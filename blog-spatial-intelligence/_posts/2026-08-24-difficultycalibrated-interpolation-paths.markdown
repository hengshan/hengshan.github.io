---
layout: post-wide
title: 'Difficulty-Calibrated Flow Matching：让生成模型在"难点"上多练一会儿'
date: 2026-08-24 08:04:24 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.21286v1
generated_by: Claude Code CLI
---

## 一句话总结

通过短暂的 pilot 训练测量各时间步的学习难度，自动调整 Conditional Flow Matching 的插值 schedule，让模型在最难学的区域获得更多训练信号——仅需约 2% 的额外开销。

## 为什么这个问题重要？

Conditional Flow Matching（CFM）已成为图像生成的主流范式之一，Stable Diffusion 3 和 FLUX 都基于此思路。CFM 的核心是学习一个速度场 $v_\theta(t, x)$，将噪声分布沿时间 $t \in [0,1]$ 传输到数据分布。

但一个长期被忽视的细节是：**训练时 $t$ 是均匀采样的**。

均匀采样意味着模型在"简单"时间步（$t$ 接近 0 或 1，噪声和数据都比较纯）和"困难"时间步（中间段，噪声数据混杂，速度场最复杂）上花同样多的资源。这就像让学生在已经会做的题和不会做的题上花同等时间——效率显然不高。

Difficulty-Calibrated Flow Matching 解决的正是这个问题：**自动发现哪里难、然后把计算资源重新分配到那里。**

## 背景：Conditional Flow Matching

### 核心框架

CFM 定义了一条从噪声 $x_0 \sim \mathcal{N}(0, I)$ 到数据 $x_1 \sim p_{data}$ 的插值路径，最常用的是线性（Optimal Transport）路径：

$$x_t = (1 - t)\, x_0 + t\, x_1, \quad t \in [0, 1]$$

对应的目标速度场（常数，这是 OT 路径的优点）为：

$$u_t(x_0, x_1) = x_1 - x_0$$

**训练目标**：让网络回归目标速度：

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_0,\, x_1} \left[\, \| v_\theta(t, x_t) - u_t \|^2 \,\right]$$

**采样（Inference）**：用学到的速度场做 ODE 积分：

$$\frac{dx}{dt} = v_\theta(t, x_t), \quad x_0 \sim \mathcal{N}(0, I)$$

### 为什么不同时间步难度不同？

对于图像生成任务，直觉上：

- **$t \approx 0$（接近纯噪声）**：模型只需输出数据集的"平均方向"，语义简单
- **$t \approx 1$（接近真实图像）**：速度场接近零，调整量小
- **$t \approx 0.5$（噪声数据混杂）**：需要精细区分不同语义内容，**回归误差最大**

## 核心方法：Difficulty-Calibrated Flow Matching

### 三步流程

**Step 1 — Pilot 训练**：用标准线性 schedule 做短暂训练（约 2% 总步数），记录每个时间步的平均损失 $L(t)$。

**Step 2 — 构造难度分布**：将 $L(t)$ 归一化为概率密度：

$$p(t) = \frac{L(t)}{\int_0^1 L(s)\, ds}$$

**Step 3 — 求分位数函数（Quantile Function）**：令 $F_L$ 为难度的累积分布函数，定义新的时间 schedule：

$$\tau(u) = F_L^{-1}(u), \quad F_L(t) = \frac{\int_0^t L(s)\, ds}{\int_0^1 L(s)\, ds}$$

正式训练时均匀采样 $u \sim \mathcal{U}[0,1]$，令 $t = \tau(u)$，等价于按难度分布采样时间步。

### 为什么叫"轨迹在困难处停留更久"？

新的 schedule $\tau(u)$ 是对时间轴的重参数化。困难区域（高损失）对应 $\tau(u)$ 变化缓慢的地方：在 $u$ 空间中，更大的 $u$ 区间映射到同一个困难的 $t$ 区间。因此轨迹在困难区域"移动更慢"，给予模型更多学习机会。

### 梯度等价性（关键技术保证）

重参数化后，梯度期望保持等价——你不需要修改损失函数，只需改变如何采样 $t$：

$$\underbrace{\mathbb{E}_{t \sim \mathcal{U}} \left[\| v_\theta(t, x_t) - u_t \|^2\right]}_{\text{原始目标}} \quad \longleftrightarrow \quad \underbrace{\mathbb{E}_{u \sim \mathcal{U}} \left[\| v_\theta(\tau(u), x_{\tau(u)}) - u_{\tau(u)} \|^2\right]}_{\text{校准后（等价）}}$$

### Pipeline 概览

```
[Pilot Run ~2% steps]
均匀采样 t → 记录 L(t) per time bin
        ↓
[Schedule Calibration]
归一化 L(t) → 计算 CDF → 求逆 → 得到 τ(u)
        ↓
[正式训练 ~98% steps]
均匀采样 u → t = τ(u) → 计算 CFM 损失
```

## 实现

### 核心类

```python
import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

class DifficultyCalibrator:
    """从 pilot 训练结果中提取难度 schedule"""

    def __init__(self, num_bins=50):
        self.num_bins = num_bins
        self.schedule_fn = None  # u -> t 的逆 CDF 映射

    def fit(self, pilot_losses: dict):
        t_vals = np.array(sorted(pilot_losses.keys()))
        l_vals = np.array([pilot_losses[t] for t in t_vals])

        # 平滑降噪，避免 pilot 期间的随机波动影响 schedule
        l_vals = gaussian_filter1d(l_vals, sigma=2.0)
        l_vals = np.clip(l_vals, 1e-8, None)

        # 归一化为概率密度，计算 CDF
        l_norm = l_vals / np.trapz(l_vals, t_vals)
        dt = np.diff(np.concatenate([[0], t_vals]))
        cdf = np.cumsum(l_norm * dt)
        cdf = cdf / cdf[-1]

        # 逆 CDF：u -> t，使得在高损失区域 t 变化缓慢
        self.schedule_fn = interp1d(
            cdf, t_vals, kind='linear',
            bounds_error=False, fill_value=(t_vals[0], t_vals[-1])
        )

    def sample_t(self, n: int) -> np.ndarray:
        u = np.random.uniform(0, 1, n)
        return self.schedule_fn(u).astype(np.float32)


class CalibratedFlowMatcher:
    """Difficulty-Calibrated Conditional Flow Matcher（OT path）"""

    def __init__(self, calibrator: DifficultyCalibrator = None):
        self.calibrator = calibrator

    def interpolate(self, x0, x1, t):
        t_shape = (-1,) + (1,) * (x0.dim() - 1)
        return (1 - t.view(t_shape)) * x0 + t.view(t_shape) * x1

    def sample_time(self, batch_size, device):
        if self.calibrator is not None and self.calibrator.schedule_fn is not None:
            return torch.from_numpy(self.calibrator.sample_t(batch_size)).to(device)
        return torch.rand(batch_size, device=device)

    def compute_loss(self, model, x0, x1):
        t = self.sample_time(x0.shape[0], x0.device)
        xt = self.interpolate(x0, x1, t)
        ut = x1 - x0  # OT path 目标速度为常数
        vt = model(xt, t)
        return ((vt - ut) ** 2).mean()
```

### Pilot 训练：测量各时间步难度

```python
def run_pilot(model, dataloader, device, num_bins=50, pilot_steps=500):
    """用均匀 schedule 做短暂训练，收集各时间步的损失分布"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    cfm = CalibratedFlowMatcher()  # 无校准，均匀采样
    bin_losses = {i: [] for i in range(num_bins)}

    step = 0
    for x1, _ in dataloader:
        if step >= pilot_steps:
            break
        x1, x0 = x1.to(device), torch.randn_like(x1.to(device))
        t = torch.rand(x1.shape[0], device=device)
        xt = cfm.interpolate(x0, x1, t)
        vt = model(xt, t)
        ut = x1 - x0

        # 逐样本损失（保留 batch 维度）
        per_sample = ((vt - ut) ** 2).flatten(1).mean(1)
        for ti, li in zip(t.cpu().numpy(), per_sample.detach().cpu().numpy()):
            bin_idx = min(int(ti * num_bins), num_bins - 1)
            bin_losses[bin_idx].append(float(li))

        optimizer.zero_grad()
        per_sample.mean().backward()
        optimizer.step()
        step += 1

    return {
        (i + 0.5) / num_bins: np.mean(v)
        for i, v in bin_losses.items() if v
    }
```

### 完整训练流程

```python
def train_with_calibration(model, dataloader, device, total_steps=50000):
    pilot_steps = max(500, total_steps // 50)  # ~2% overhead

    # Phase 1: 收集难度信息
    print(f"Pilot run: {pilot_steps} steps...")
    pilot_losses = run_pilot(model, dataloader, device, pilot_steps=pilot_steps)

    # Phase 2: 拟合 schedule
    calibrator = DifficultyCalibrator()
    calibrator.fit(pilot_losses)

    # Phase 3: 正式训练（使用校准后的 schedule）
    cfm = CalibratedFlowMatcher(calibrator=calibrator)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step, (x1, _) in enumerate(dataloader):
        if step >= total_steps:
            break
        x1, x0 = x1.to(device), torch.randn_like(x1.to(device))
        loss = cfm.compute_loss(model, x0, x1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 可视化难度曲线

```python
import matplotlib.pyplot as plt

def visualize_schedule(pilot_losses, calibrator):
    t_vals = sorted(pilot_losses.keys())
    l_vals = [pilot_losses[t] for t in t_vals]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t_vals, l_vals, 'b-', lw=2)
    axes[0].fill_between(t_vals, l_vals, alpha=0.2)
    axes[0].set(xlabel='Time t', ylabel='Loss', title='Difficulty Profile L(t)')

    u = np.linspace(0, 1, 200)
    axes[1].plot(u, calibrator.schedule_fn(u), 'r-', lw=2, label='τ(u) calibrated')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Linear (baseline)')
    axes[1].set(xlabel='u (uniform)', ylabel='t (actual)', title='Time Reparameterization τ(u)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('difficulty_schedule.png', dpi=150)
```

预期输出：左图的损失曲线通常在 $t \approx 0.3\text{–}0.6$ 形成峰值；右图的 $\tau(u)$ 曲线在对应区域斜率变小（停留更久）。

## 实验

### 数据集

| 数据集 | 分辨率 | 任务难度 |
|--------|--------|---------|
| CIFAR-10 | 32×32 | 中（主要基准）|
| MNIST | 28×28 | 低 |
| Fashion-MNIST | 28×28 | 中低 |

三个数据集使用相同的紧凑 U-Net，确保对比公平。

### 定量结果

| 方法 | CIFAR-10 FID ↓（全步数）| 大 batch + 少步数 |
|------|------------------------|------------------|
| Linear（均匀）| 基准 | 基准 |
| Cosine（固定）| 次优 | 次优 |
| **Difficulty-Calibrated** | **最优** | **明显最优 ★★★** |

论文的核心发现：在**大 batch + 少步数**（compute-efficient 训练）场景下，校准 schedule 的优势最为显著。这恰好是工业界最关心的设定——每次 GPU 小时都很宝贵。

## 工程实践

### 开销估算

```python
# 规则：pilot steps = max(500, total_steps // 50)
# 对应 2% 额外训练开销 + 一次 CPU 侧的 CDF 计算（毫秒级）
# 正式训练阶段：零额外开销
```

### 与 Classifier-Free Guidance 结合

```python
def cfg_loss(model, x0, x1, cond, cfm, uncond_prob=0.1):
    t = cfm.sample_time(x1.shape[0], x1.device)
    xt = cfm.interpolate(x0, x1, t)
    ut = x1 - x0
    # 随机 drop 条件（CFG 标准做法，schedule 无需修改）
    drop = torch.rand(len(cond)) < uncond_prob
    cond_in = [None if d else c for d, c in zip(drop, cond)]
    return ((model(xt, t, cond_in) - ut) ** 2).mean()
```

校准 schedule 与 CFG 完全兼容，只需在 pilot run 时同样随机 drop 条件即可。

### 常见坑

**1. Pilot 曲线噪声过大（小 batch 时常见）**

每个 bin 的样本数不足导致 $L(t)$ 抖动剧烈，拟合出的 schedule 不稳定。

```python
# 修复：加大平滑 sigma 或增加 pilot steps
l_vals = gaussian_filter1d(l_vals, sigma=3.0)  # 适当加强平滑
```

**2. 难度曲线单调（任务本身过于简单）**

如果 $L(t)$ 几乎是平线，校准 schedule 退化为均匀分布，无增益。这不是 bug，而是任务本身不需要校准的信号。

**3. 初始化阶段难度不代表收敛后难度**

Pilot 在训练最初期做，此时模型随机初始化，损失绝对值虚高。论文验证了 pilot 的相对难度排序在训练过程中保持稳定，但对于非常长的训练，可以考虑中途重新校准。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| Compute-efficient 训练（大 batch、少步数）| 无计算预算限制 |
| 中等难度以上的生成任务（CIFAR 量级及以上）| 任务难度均匀（如 MNIST）|
| 与现有 CFM 框架无缝集成 | 需要完全可复现的固定 schedule |
| Pilot 开销可接受的场景 | 超短训练（总步数 < 1000）|

## 与其他方法对比

| 方法 | 核心思想 | 优点 | 缺点 |
|------|---------|------|------|
| Linear CFM | 均匀时间采样 | 简单，无需额外步骤 | 忽略难度差异 |
| Cosine/Log-SNR | 固定非线性 schedule | 经验有效，借鉴自 DDPM | 与数据/模型无关 |
| Min-SNR Weighting | 对损失按 SNR 加权 | 改变优化权重，互补 | 不改变轨迹形状 |
| **Difficulty-Calibrated（本文）**| 数据驱动的自适应 schedule | 针对具体任务，等价梯度 | 需要 pilot run |

本文方法与 Min-SNR Weighting 互补：前者改变"在哪个时间步多采样"，后者改变"每个时间步的损失权重"，理论上可以叠加使用。

## 我的观点

这是一篇"小而美"的工作。核心思想用一句话说清楚，实现代价几乎为零，理论保证（梯度等价性）干净利落。

**值得关注的开放问题：**

1. **大规模验证缺失**：论文在 32×32 上做实验，对 Stable Diffusion 3、FLUX 这类 512×512 甚至更高分辨率场景，效果是否同样显著？中间时间步的难度峰值是否会因为网络容量变大而消失？

2. **动态重校准**：论文只在开头做一次 pilot。对于长程训练，难度分布可能随模型能力提升而漂移，定期重新校准（如每 10% 进度重测一次）是否有额外收益尚未探索。

3. **与 Rectified Flow 的关系**：Rectified Flow 通过"拉直"轨迹减少中间段复杂度，本质上是降低 $t \approx 0.5$ 处的内在难度。Difficulty-Calibrated 则是给难度高的地方多分配资源。两者从不同角度解决同一问题，结合使用理论上应该更好，但没有公开实验。

**实用建议**：如果你正在训练 Flow Matching 模型，2% 的 pilot overhead 几乎不值一提，加进 pipeline 试试没有任何风险。在大 batch 少步数的场景（工业预训练的常见设定），这个 trick 的性价比极高。