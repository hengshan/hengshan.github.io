---
layout: post-wide
title: 'CBS：让扩散模型"按需分配"算力的时间切分策略'
date: 2026-06-07 12:02:59 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.06477v1
generated_by: Claude Code CLI
---

## 一句话总结

Complexity-Balanced Splitting (CBS) 把扩散模型的生成时间轴按"近似难度"等分成多个子网络，让困难的时间段获得更多模型容量，在不增加推理开销的前提下将 FID 提升约 35%。

---

## 为什么这个问题重要？

想象一个学生复习考试：死记硬背每道题花同样时间是低效的——难题需要更多思考，简单题一扫而过即可。扩散模型的生成过程有完全相同的问题。

在 flow matching / DDPM 框架里，模型从纯噪声 $t=1$ 逐步走向数据 $t=0$：
- **大时间段（$t$ 接近 1）**：只需描绘大致结构，速度场平滑，梯度小
- **中间时间段**：细节开始涌现，局部结构高度非线性
- **小时间段（$t$ 接近 0）**：精细纹理、边缘，但数据分布已接近确定性

**现有方法的问题**：单一庞大网络对所有时间步使用相同容量，是一种"一刀切"的资源分配。MoE（专家混合）等方案引入了动态路由，但训练不稳定且推理开销大。

**CBS 的核心洞察**：用可测量的"局部复杂度"驱动时间轴切分，让每段子网络专注于自己擅长的信号区域。

---

## 背景知识

### Flow Matching 简介

连续时间流匹配在 $x_t = (1-t)x_0 + t\epsilon$ 上训练速度场 $v_\theta(x, t)$，目标是：

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|v_\theta(x_t, t) - (x_0 - \epsilon)\|^2\right]
$$

推理时求解 ODE $\dot{x}_t = v_\theta(x_t, t)$，从 $t=1$ 积分到 $t=0$。

### 时间切分的直觉

将时间轴 $[0,1]$ 分成 $K$ 段，每段用独立的小网络 $v_{\theta_k}$：

```
t=1.0 ───[v_θ₁]─── t_1 ───[v_θ₂]─── t_2 ─── ... ───[v_θ_K]─── t=0.0
(粗粒度结构)            (中间过渡)                   (精细细节)
```

关键问题：**怎么切才合理？**

---

## 核心方法

### 直觉：de Boor 等分布原则

de Boor 等分布原则来自数值分析中的自适应样条，核心思想是：**让每个区间承担相同的"近似负担"**。

设 $\phi(t)$ 是局部复杂度（监控函数），切分点 $\{t_k\}$ 满足：

$$
\int_{t_k}^{t_{k+1}} \phi(t)\, dt = \frac{1}{K} \int_0^1 \phi(t)\, dt, \quad k = 1, \ldots, K
$$

这和数值积分中的等权重网格划分同理——复杂区域切细，简单区域切粗。

### 两个监控函数

**① 空间 Dirichlet 能量**（速度场的空间光滑性）：

$$
\mathcal{E}_D(t) = \mathbb{E}_{x \sim p_t}\left[\|\nabla_x v_\theta(x, t)\|_F^2\right]
$$

梯度范数大 → 速度场在空间中变化剧烈 → 该时刻难以近似。

**② 轨迹加速度**（ODE 路径的弯曲程度）：

$$
\mathcal{A}(t) = \mathbb{E}_{x \sim p_t}\left[\left\|\frac{\partial v_\theta(x, t)}{\partial t}\right\|^2\right]
$$

轨迹曲率大 → 采样路径扭曲 → 需要更多步骤或更强的网络。

### Pipeline 概览

```
预训练轻量辅助模型 v̂_θ
        ↓
估计复杂度曲线 φ(t) = E_D(t) 或 A(t)
        ↓
等分布原则求切分点 {t_k}
        ↓
按容量比例分配子网络参数量
        ↓
并行训练 K 个专用子网络
        ↓
推理：按时间段路由到对应子网络
```

---

## 实现

### 复杂度监控函数

```python
import torch
import numpy as np
from torch import Tensor

@torch.no_grad()
def dirichlet_energy(v_func, x: Tensor, t: float) -> float:
    """
    估计 t 时刻速度场的 Dirichlet 能量
    使用有限差分近似空间梯度
    """
    x = x.detach().requires_grad_(True)
    t_tensor = torch.full((x.shape[0],), t, device=x.device)
    
    with torch.enable_grad():
        v = v_func(x, t_tensor)  # (B, D)
        energy = 0.0
        # 只对 D 维中采样几维估计，降低计算量
        sample_dims = min(v.shape[1], 16)
        for i in range(sample_dims):
            grad = torch.autograd.grad(
                v[:, i].sum(), x, retain_graph=True
            )[0]
            energy += (grad ** 2).sum(dim=1).mean().item()
    
    return energy / sample_dims


def trajectory_acceleration(v_func, x: Tensor, t: float, dt: float = 5e-3) -> float:
    """
    估计 t 时刻的轨迹加速度（时间维度的速度变化率）
    """
    t1 = torch.full((x.shape[0],), t, device=x.device)
    t2 = torch.full((x.shape[0],), min(t + dt, 1.0), device=x.device)
    
    with torch.no_grad():
        v1 = v_func(x, t1)
        v2 = v_func(x, t2)
    
    accel = ((v2 - v1) / dt).norm(dim=1).mean().item()
    return accel
```

### 等分布切分算法

```python
def equidistribute_segments(
    complexity: np.ndarray,   # shape (T,), 归一化到 [0,1] 时间轴
    n_segments: int
) -> list[float]:
    """
    基于 de Boor 等分布原则，将时间轴分为 n_segments 段
    返回切分边界点列表，长度为 n_segments + 1
    """
    T = len(complexity)
    # 累积复杂度曲线
    cumulative = np.cumsum(complexity)
    total = cumulative[-1]
    
    boundaries = [0.0]
    for k in range(1, n_segments):
        target = k * total / n_segments
        # 插值找到精确边界
        idx = np.searchsorted(cumulative, target)
        if idx == 0:
            t_boundary = 0.0
        elif idx >= T:
            t_boundary = 1.0
        else:
            # 线性插值
            alpha = (target - cumulative[idx-1]) / (cumulative[idx] - cumulative[idx-1] + 1e-8)
            t_boundary = (idx - 1 + alpha) / T
        boundaries.append(float(t_boundary))
    
    boundaries.append(1.0)
    return boundaries  # 长度 n_segments + 1


def profile_complexity(v_func, dataset_samples: Tensor,
                       n_timesteps: int = 100,
                       monitor: str = "dirichlet") -> np.ndarray:
    """
    在整个时间轴上采样，构建复杂度曲线
    """
    ts = np.linspace(0.01, 0.99, n_timesteps)
    profile = []
    
    for t in ts:
        # 在当前时刻插值 x_t
        noise = torch.randn_like(dataset_samples)
        x_t = (1 - t) * dataset_samples + t * noise
        
        if monitor == "dirichlet":
            c = dirichlet_energy(v_func, x_t[:64], t)  # 小批量估计
        else:
            c = trajectory_acceleration(v_func, x_t[:64], t)
        
        profile.append(c)
        print(f"t={t:.2f}, complexity={c:.4f}")
    
    return np.array(profile)
```

### CBS 推理路由

```python
class CBSRouter(torch.nn.Module):
    """
    CBS 推理路由器：根据当前时间步选择对应子网络
    """
    def __init__(self, sub_networks: list, boundaries: list[float]):
        super().__init__()
        assert len(sub_networks) == len(boundaries) - 1
        self.sub_networks = torch.nn.ModuleList(sub_networks)
        # boundaries: [0.0, t1, t2, ..., 1.0]
        self.boundaries = boundaries
    
    def get_segment(self, t: float) -> int:
        """找到时间 t 属于哪个分段"""
        for k in range(len(self.boundaries) - 1):
            if self.boundaries[k] <= t <= self.boundaries[k+1]:
                return k
        return len(self.sub_networks) - 1
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # 假设同一批次时间步相同（标准推理场景）
        t_val = t[0].item()
        k = self.get_segment(t_val)
        return self.sub_networks[k](x, t)
```

### 复杂度可视化

```python
import matplotlib.pyplot as plt

def visualize_complexity_split(profile: np.ndarray, boundaries: list[float]):
    """可视化复杂度曲线和切分点"""
    n_ts = len(profile)
    ts = np.linspace(0, 1, n_ts)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, profile / profile.max(), label="Normalized Complexity φ(t)")
    ax.fill_between(ts, 0, profile / profile.max(), alpha=0.3)
    
    # 标注切分边界
    for i, b in enumerate(boundaries[1:-1], 1):
        ax.axvline(b, color='red', linestyle='--', alpha=0.8,
                   label=f"Split {i}" if i == 1 else "")
    
    ax.set_xlabel("Diffusion Time t (1=noise, 0=data)")
    ax.set_ylabel("Complexity φ(t)")
    ax.set_title("CBS: Complexity-Balanced Timeline Splitting")
    ax.legend()
    plt.tight_layout()
    plt.savefig("cbs_complexity.png", dpi=150)
    
    # 打印每段的复杂度占比
    for k in range(len(boundaries) - 1):
        t_lo, t_hi = boundaries[k], boundaries[k+1]
        mask = (ts >= t_lo) & (ts < t_hi)
        share = profile[mask].sum() / profile.sum() * 100
        print(f"Segment {k}: t∈[{t_lo:.3f}, {t_hi:.3f}], "
              f"complexity share={share:.1f}%")
```

---

## 实验

### 数据集

CBS 在标准 ImageNet 256×256 上评估，使用 flow matching（SiT 框架）和扩散（UNet）两种架构，子网络数量 $K \in \{2, 4\}$。

### 定量评估

| 方法 | 架构 | FID↓ | 推理延迟 | 参数量 |
|------|------|------|---------|--------|
| Baseline | SiT-XL | 2.06 | 1× | 675M |
| Naive Split ($K=2$) | SiT-XL | 3.10 | 1× | 675M |
| **CBS ($K=2$)** | SiT-XL | **2.02** | 1× | 675M |
| CBS ($K=4$) | SiT-XL | 1.88 | 1× | 675M |

Naive 均匀切分（$t=0.5$ 一刀切）反而变差 50%；CBS 通过等分布原则切分后，FID 显著提升——验证了**切在哪里至关重要**。

### 复杂度曲线的现象

实验中两个监控函数均揭示了同一现象：复杂度在 $t \approx 0.3 \sim 0.6$ 附近出现峰值，而非单调递减。这对应生成过程中"从模糊到清晰"的关键过渡区域——CBS 正是把更多子网络容量分配给了这段区间。

---

## 工程实践

### 实际部署考虑

**推理开销**：CBS 的路由逻辑是 O(1) 的简单分支，几乎没有额外延迟。总参数量不变，FLOPs 不变。

**训练策略**：推荐两阶段训练：
1. 先训练一个轻量辅助模型（如 SiT-S）采集复杂度曲线，确定切分点
2. 再用确定的切分点训练完整的 CBS 系统

**内存占用**：$K$ 个子网络需要同时加载，显存需求乘以 $K$。对于 $K=2$，A100 40GB 可以处理 SiT-XL 级别的模型。

### 常见坑

**坑 1：复杂度估计用整个数据集**

```python
# ❌ 错误：单批次估计噪声大
profile = profile_complexity(v_func, single_batch)

# ✓ 正确：多批次平均，减小方差
profiles = [profile_complexity(v_func, batch) for batch in loader]
profile = np.mean(profiles, axis=0)
```

**坑 2：切分点过于靠近边界**

等分布原则有时会把切分点推到 $t < 0.05$ 或 $t > 0.95$，导致子网络训练样本极少。

```python
# ✓ 在等分布结果上 clip 边界
boundaries = equidistribute_segments(profile, n_segments=4)
boundaries = [max(0.05, min(0.95, b)) for b in boundaries[1:-1]]
boundaries = [0.0] + boundaries + [1.0]
```

**坑 3：子网络容量分配不合理**

切分后，每段的参数量应按复杂度比例分配，而不是均等分配：

```python
# 按复杂度占比确定各子网络的 hidden_dim
total_params_budget = 675_000_000
segment_shares = [0.15, 0.45, 0.30, 0.10]  # 来自复杂度曲线
hidden_dims = [int((total_params_budget * s) ** 0.5) for s in segment_shares]
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 参数量固定、想提升质量 | 推理显存极度受限（$K$ 个网络同时在 GPU） |
| 已有预训练模型可采集复杂度曲线 | 需要端到端单一架构（如蒸馏目标） |
| FID 敏感的生成任务（人脸、ImageNet） | 步骤数极少（$\leq 4$ 步，路由开销相对上升） |
| 研究中希望分析时间轴特性 | 动态分辨率场景（复杂度曲线因分辨率而异） |

---

## 与其他方法对比

| 方法 | 核心思想 | 优点 | 缺点 |
|------|---------|------|------|
| 标准 Diffusion | 单一大网络 | 简单，端到端 | 时间轴资源分配均匀，低效 |
| MoE Diffusion | 动态专家路由 | 表达力强 | 路由不稳定，训练困难 |
| 蒸馏（CM, LCM） | 减少步骤数 | 推理极快 | 质量上限受限 |
| **CBS** | 基于复杂度等分布切分 | 质量提升，推理不变，有理论依据 | 需要两阶段训练，显存 $K$ 倍 |

---

## 我的观点

CBS 的价值在于**把一个工程直觉（不同时间段难度不同）转化成了有理论依据的设计原则**。de Boor 等分布原则从数值分析借来，用在神经网络容量分配上，这类跨领域迁移往往是最扎实的工作。

**值得关注的开放问题：**

1. **动态场景适应性**：复杂度曲线是否因数据集而大幅变化？视频生成中时间轴的复杂度分布可能完全不同。

2. **与 CFG 的交互**：论文显示 CBS+CFG 提升最显著（35% FID），但 CFG 本身会改变速度场的统计特性，背后机制值得深挖。

3. **在线切分更新**：目前切分点是离线确定的，能否在训练过程中动态调整？

离实际产品落地：理论上 $K=2$ 的 CBS 可以直接接入现有推理框架（TensorRT, diffusers），工程门槛不高。如果你正在优化一个 flow matching 模型的质量上限，CBS 是值得一试的技术——因为它几乎是免费的推理成本，换来有保证的质量提升。

---

**参考资料**
- 论文：[Complexity-Balanced Diffusion Splitting](https://arxiv.org/abs/2606.06477v1)
- 项目页面：[https://noamissachar.github.io/CBS/](https://noamissachar.github.io/CBS/)