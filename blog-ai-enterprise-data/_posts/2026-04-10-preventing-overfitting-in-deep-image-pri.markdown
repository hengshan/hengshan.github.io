---
layout: post-wide
title: "深度图像先验去噪：如何对抗 DIP 的过拟合陷阱"
date: 2026-04-10 12:06:56 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.08272v1
generated_by: Claude Code CLI
---

## 一句话总结

Deep Image Prior（DIP）用网络架构作为隐式正则化，无需训练数据即可去噪——但它会随迭代过拟合噪声，本文通过 Smooth ℓ1 + 散度正则化 + 输入联合优化三者组合来压制这个问题，尤其针对高光谱图像（HSI）的混合噪声场景。

---

## 背景：DIP 的优雅与脆弱

### DIP 是什么

2018 年 Ulyanov 等人发现了一个反直觉的结论：**卷积网络的结构本身就是一种图像先验**，无需任何训练样本。

给定含噪图像 $y$，固定一个随机输入 $z$，只优化网络参数：

$$\hat{\theta} = \arg\min_{\theta} \|f_\theta(z) - y\|_2^2$$

关键现象：**网络优先拟合低频信号（真实内容），高频噪声拟合得更晚**。CNN 的卷积天然是低通滤波器，"生成平滑图像"比"生成随机噪声"更容易。

### 高光谱图像的特殊挑战

HSI 不是 RGB 三通道，而是 100-200+ 个波段的图像立方体，常见于遥感、医学成像。它同时面临三类噪声：

- **高斯噪声**：传感器热噪声，均匀分布
- **稀疏噪声**：坏像素、宇宙射线，幅值极大
- **条带噪声**：推扫传感器的列缺陷，整列偏移

MSE 损失对稀疏噪声极度脆弱——一个幅值为 10 的坏像素，梯度贡献是正常像素的 100 倍，强迫网络去拟合它。

### 过拟合问题的本质

```
         ← 真实信号被拟合 →  ← 噪声开始被拟合 →
PSNR ↑   ████████████████▓▒░░░░░░░░░░░░
         0        最佳停止点          ∞  迭代次数
```

标准 DIP 的核心矛盾：**迭代次数即是唯一的正则化参数，但最佳停止点无法预测**，尤其在无参考图像时。

---

## 三板斧：本文的解决方案

联合优化目标：

$$\hat{\theta}, \hat{z} = \arg\min_{\theta,\, z} \underbrace{\mathcal{L}_{\text{Huber}}(f_\theta(z),\, y)}_{\text{鲁棒数据保真项}} + \lambda \underbrace{R(f_\theta, z)}_{\text{敏感度正则化}}$$

### 1. Smooth ℓ1（Huber 损失）

$$\mathcal{L}_{\text{Huber}}(r;\,\delta) = \begin{cases} \dfrac{r^2}{2\delta} & |r| \leq \delta \\[6pt] |r| - \dfrac{\delta}{2} & |r| > \delta \end{cases}$$

当误差 $|r| > \delta$ 时退化为 ℓ1（线性），稀疏异常值的梯度有界，不再劫持优化。

### 2. 散度正则化（Sensitivity Regularization）

**核心直觉**：若网络输出对输入 $z$ 的微小扰动极度敏感，说明它在记忆噪声而非学习信号。

用 Jacobian 的 Frobenius 范数量化敏感度，$J = \frac{\partial f_\theta}{\partial z}$：

$$R(f_\theta, z) = \|J\|_F^2 = \text{tr}(J^\top J)$$

直接计算 $J$ 代价极高，用 **Hutchinson 估计器**：

$$\text{tr}(J^\top J) \approx \frac{1}{n} \sum_{i=1}^n \|J^\top v_i\|^2, \quad v_i \sim \mathcal{N}(0, I)$$

每次只需 $n$ 次反向传播（$n=2$ 够用）。

### 3. 联合输入优化

标准 DIP 固定 $z$，本文同时优化 $z$ 和 $\theta$。$z$ 获得了适应图像结构的自由度，而正则化防止这种自由度失控。

---

## 实现

### 最小可运行版本

```python
import torch
import torch.nn as nn
import numpy as np

def huber_loss(pred, target, delta=0.1):
    """Smooth ℓ1 损失，对稀疏噪声鲁棒"""
    diff = (pred - target).abs()
    return torch.where(diff < delta,
                       diff**2 / (2 * delta),
                       diff - delta / 2).mean()

def sensitivity_reg(output, z, n_probes=2):
    """Hutchinson 估计 ||J||_F^2，J = ∂output/∂z"""
    reg = torch.tensor(0.0, device=z.device)
    out_flat = output.reshape(-1)
    for _ in range(n_probes):
        v = torch.randn_like(output)
        # 计算 J^T v = ∂(v^T · output)/∂z
        jtv, = torch.autograd.grad(out_flat, z,
                                    grad_outputs=v.reshape(-1),
                                    retain_graph=True,
                                    create_graph=True)
        reg = reg + (jtv ** 2).mean()
    return reg / n_probes

class DIPNet(nn.Module):
    """轻量 U-Net，含跳跃连接，适合 HSI 去噪"""
    def __init__(self, in_ch, out_ch, hidden=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.2))
        self.e2 = nn.Sequential(nn.Conv2d(hidden, hidden*2, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.d1 = nn.Sequential(nn.ConvTranspose2d(hidden*2, hidden, 2, stride=2), nn.LeakyReLU(0.2))
        self.out = nn.Conv2d(hidden*2, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        d1 = self.d1(e2)
        return torch.sigmoid(self.out(torch.cat([d1, e1], dim=1)))
```

### 完整训练流程

```python
def train_dip_hsi(noisy_hsi: np.ndarray, num_iter=3000,
                   lr=0.01, lambda_reg=1e-3, delta=0.1) -> np.ndarray:
    """
    Args:
        noisy_hsi: 含噪 HSI，shape=(C, H, W)，值域 [0,1]
        lambda_reg: 敏感度正则化权重（调参重点）
    Returns:
        去噪后 HSI，shape=(C, H, W)
    """
    C, H, W = noisy_hsi.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DIPNet(in_ch=32, out_ch=C, hidden=64).to(device)
    # z 解耦维度，同时作为可训练参数
    z = torch.randn(1, 32, H, W, device=device, requires_grad=True)
    y = torch.tensor(noisy_hsi, dtype=torch.float32, device=device).unsqueeze(0)

    optimizer = torch.optim.Adam(list(model.parameters()) + [z], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iter)

    best_loss, best_output = float('inf'), None

    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(z)
        loss = huber_loss(output, y, delta=delta) + lambda_reg * sensitivity_reg(output, z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([z], max_norm=0.1)  # z 的梯度也要裁
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_output = output.detach().cpu().squeeze(0).numpy()

    return best_output
```

### 关键 Trick

**Trick 1：输入维度解耦**（最容易犯的错）

```python
# ❌ 错误：z 的维度和波段数耦合，引入维度偏置
z = torch.tensor(noisy_hsi).unsqueeze(0).requires_grad_(True)

# ✅ 正确：固定 32 维随机输入，网络自由映射到任意波段数
z = torch.randn(1, 32, H, W, requires_grad=True) * 0.1
```

**Trick 2：δ 跟着噪声走**

```python
# 噪声水平估计（MAD 估计器）
sigma_est = np.median(np.abs(noisy_hsi - np.median(noisy_hsi))) / 0.6745
delta = float(sigma_est) * 1.5  # 经验系数，稀疏噪声可调到 2.0
```

**Trick 3：梯度裁剪不可少**

Hutchinson 估计通过 `create_graph=True` 生成高阶计算图，梯度偶尔会爆炸，`clip_grad_norm_` 是保险丝，不是可选项。

---

## 实验

### 合成数据生成

```python
def make_synthetic_hsi(C=31, H=64, W=64, seed=42):
    """生成低秩干净 HSI + 混合噪声"""
    np.random.seed(seed)
    # 低秩结构（真实 HSI 的核心特征）
    U = np.random.randn(C, 6)
    V = np.random.randn(6, H * W)
    clean = (U @ V).reshape(C, H, W)
    clean = (clean - clean.min()) / clean.ptp()

    noisy = clean.copy().astype(np.float32)
    noisy += np.random.randn(C, H, W).astype(np.float32) * 0.05   # 高斯
    mask = np.random.rand(C, H, W) < 0.02
    noisy[mask] = np.random.uniform(0.8, 1.0, mask.sum())          # 稀疏
    noisy[:, :, np.random.choice(W, 4)] += 0.25                    # 条带
    return clean.astype(np.float32), np.clip(noisy, 0, 1).astype(np.float32)
```

### 方法对比

| 方法 | PSNR (dB) | SSIM | SAM (°) | 需要数据? |
|------|-----------|------|---------|----------|
| 含噪输入 | ~20 | ~0.55 | ~8.5 | - |
| DIP (MSE) | ~26 | ~0.78 | ~4.2 | 否 |
| DIP + Huber | ~28 | ~0.82 | ~3.5 | 否 |
| **DIP + Huber + 散度正则** | **~30** | **~0.87** | **~2.8** | 否 |
| DnCNN（监督）| ~33 | ~0.91 | ~2.1 | **是** |

> 散度正则化的提升在**稀疏噪声**上最显著，PSNR 差距能到 3-4 dB；纯高斯噪声下提升有限。

---

## 调试指南

### 看什么指标

训练时同时监控 loss 和（如果有 clean 图）PSNR：

```
正常收敛：loss ↓ 且 PSNR ↑，在某点趋于平稳
过拟合中：loss ↓ 但 PSNR 在 1000 轮后开始 ↓
未充分拟合：PSNR 1000 轮后仍 < 25dB
```

### 常见问题排查

**1. Loss 从头不动**
- `z.requires_grad` 是否为 True
- 学习率太低（DIP 常用 `lr=0.01`，比监督学习高 10 倍）
- BatchNorm 在网络很浅时可以试着去掉

**2. 训练到 NaN 崩溃**
- Hutchinson 估计的梯度爆炸：确认两处 `clip_grad_norm_` 都在
- `delta` 过小（分母接近 0）：改为 `max(delta, 1e-4)`
- `z` 初始化方差过大：乘以 0.1

**3. 加了正则反而更差**
- `lambda_reg` 过大：从 `1e-4` 开始，而非 `1e-2`
- `n_probes=1` 时估计方差大，可以用 `n_probes=3`
- 检查 `create_graph=True` 没有漏掉

### 超参数敏感度参考

| 参数 | 推荐起点 | 敏感度 | 调参建议 |
|------|---------|-------|---------|
| `lr` | 0.01 | 高 | 先固定这个 |
| `lambda_reg` | 1e-3 | 高 | 按噪声水平和类型调 |
| `delta` | ≈ noise σ | 中 | 稀疏噪声适当加大 |
| `num_iter` | 3000 | 中 | 条带噪声可延长 |
| `hidden` | 64 | 低 | 大多数情况够了 |

---

## 什么时候用 / 不用

| 适用场景 | 不适用场景 |
|---------|-----------|
| 完全无训练数据 | 有配对数据（监督方法更强）|
| 混合噪声（高斯 + 稀疏 + 条带）| 单一高斯噪声（BM3D 更快更稳）|
| 单张图像探索性分析 | 实时/批量处理（每张图要完整优化）|
| 需要无监督基线的科研对比 | 对结果稳定性要求高的工业场景 |

---

## 我的观点

DIP 是一个极度优雅的洞察——**网络架构即先验**。这个框架让我们重新思考"正则化"的来源：不一定是数据，可以是归纳偏置。

但说实话，**用 DIP 做实际工程很痛**。每张图都要跑完整优化（几分钟），超参数对噪声类型极度敏感，在一类噪声上调好的参数换个场景直接失效。

本文的三个机制都有道理：Huber 对稀疏噪声的鲁棒性毋庸置疑；Jacobian 正则化的物理意义也很清晰——**高敏感度 = 在记忆噪声**。但这也引入了额外超参数，并没有根本降低调参负担。

**最实在的建议**：如果你没有任何训练数据且噪声是混合型，这个方法值得一试，它会比朴素 DIP 稳定很多。但如果有任何机会收集数据，训练一个 DnCNN 或 FFDNet 会在各方面碾压它。DIP 的真正价值在于**理论理解**，而不是工程部署。