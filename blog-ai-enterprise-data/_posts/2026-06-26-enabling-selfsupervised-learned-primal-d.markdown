---
layout: post-wide
title: "无监督 CT 重建：Noise2Inverse Learned Primal-Dual 完整指南"
date: 2026-06-26 12:04:49 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.26991v1
generated_by: Claude Code CLI
---

## 一句话总结

N2I-LPD 把"需要 ground truth 才能训练"的学习型 CT 重建算法，改造成只用噪声测量值就能自监督训练的版本，在低剂量和稀疏角度场景下，性能接近有监督方法，显著优于传统迭代重建。

## 背景：监督学习的困境

CT 重建是经典逆问题：给定 X 射线投影测量值 $\mathbf{y}$，恢复原始图像 $\mathbf{x}$：

$$\mathbf{y} = A\mathbf{x} + \boldsymbol{\epsilon}$$

其中 $A$ 是 Radon 变换（前向投影），$\boldsymbol{\epsilon}$ 是测量噪声。

**传统方法的局限：**
- FBP（滤波反投影）：速度快，但低剂量/稀疏角度下噪声大
- 迭代重建（TV 正则化）：质量更好，但需要手动调参，速度慢

**深度学习的困境：**

Learned Primal-Dual 等学习型算法在有配对数据时表现出色，但临床场景中高质量 CT ground truth 极难获取——你不能为了训练数据给同一个病人照两次剂量差异很大的 CT。

**核心 insight：**

CT 扫描不同角度的噪声是**统计独立**的。只要满足这个条件，就可以"用噪声监督噪声"——根本不需要 ground truth。

## 算法原理

### Learned Primal-Dual 回顾

LPD 把凸优化的原始-对偶迭代展开（unroll）成神经网络，每次迭代交替更新：

**对偶更新**（在 sinogram 域）：

$$\mathbf{u}^{k+1} = \mathbf{u}^k + \Lambda_\phi^k(\mathbf{u}^k,\ A\mathbf{x}^k,\ \mathbf{y})$$

**原始更新**（在图像域）：

$$\mathbf{x}^{k+1} = \mathbf{x}^k + \Gamma_\theta^k(\mathbf{x}^k,\ A^T\mathbf{u}^{k+1})$$

其中 $\Lambda_\phi^k$ 和 $\Gamma_\theta^k$ 是每轮独立参数的小型 CNN，$A^T$ 是反投影算子。直觉上：对偶变量 $\mathbf{u}$ 捕捉测量残差，原始变量 $\mathbf{x}$ 是目标图像，两者交替纠正彼此，类似 ADMM 的展开版本。

### Noise2Inverse 框架

1. 把 sinogram 按角度分成 $K$ 个**统计独立**的子集 $\{\mathbf{y}_1, \ldots, \mathbf{y}_K\}$
2. 用 $K-1$ 个子集训练重建网络
3. 用留出的第 $i$ 个子集计算损失

**N2I 损失函数：**

$$\mathcal{L}_{\text{N2I}} = \frac{1}{K} \sum_{i=1}^{K} \| A_i\, f_\theta(\mathbf{y}_{-i}) - \mathbf{y}_i \|^2$$

其中 $\mathbf{y}_{-i}$ 是除第 $i$ 个子集外所有子集的合并，$A_i$ 是对应的部分角度前向算子。

**为什么这个损失有效？**

由于不同角度噪声统计独立，可以证明：

$$\mathbb{E}[\mathcal{L}_{\text{N2I}}] = \mathbb{E}\!\left[\| A_i\, f_\theta(\mathbf{y}_{-i}) - A_i\mathbf{x}^* \|^2\right] + C$$

其中 $C$ 与参数 $\theta$ 无关，$\mathbf{x}^*$ 是真实图像。最小化 N2I 损失等价于最小化重建误差——无需 ground truth。

### N2I-LPD：两者的结合

将 LPD 作为 N2I 框架中的重建函数 $f_\theta$：

- **训练时**：输入 $K-1$ 个角度子集的合并 sinogram，LPD 内部的 $A$ 和 $A^T$ 也对应该部分角度算子，用 N2I 损失反传
- **推理时**：输入完整 sinogram，LPD 用完整角度算子正常运行

## 实现

### LPD 网络结构

```python
import torch
import torch.nn as nn

class UpdateNet(nn.Module):
    """轻量级更新网络：3 层 CNN + PReLU，输出残差"""
    def __init__(self, in_ch, mid_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.PReLU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.PReLU(),
            nn.Conv2d(mid_ch, 1, 3, padding=1),
        )
    def forward(self, *tensors):
        return self.net(torch.cat(tensors, dim=1))

class LearnedPrimalDual(nn.Module):
    def __init__(self, n_iters=10, n_channels=32):
        super().__init__()
        self.n_iters = n_iters
        # 每轮迭代独立参数（不共享权重）
        # 对偶网络：输入 [u, Ax, y]，共 3 通道
        # 原始网络：输入 [x, A^T u]，共 2 通道
        self.dual_nets   = nn.ModuleList([UpdateNet(3, n_channels) for _ in range(n_iters)])
        self.primal_nets = nn.ModuleList([UpdateNet(2, n_channels) for _ in range(n_iters)])

    def forward(self, y, forward_op, backward_op):
        """
        y:           sinogram (B, 1, n_angles, n_dets)
        forward_op:  x -> Ax  （可微 Radon 变换）
        backward_op: u -> A^T u（反投影）
        """
        x = backward_op(y)         # FBP 初始化，比零初始化收敛快
        u = torch.zeros_like(y)

        for k in range(self.n_iters):
            u = u + self.dual_nets[k](u, forward_op(x), y)
            x = x + self.primal_nets[k](x, backward_op(u))

        return x
```

### Noise2Inverse 训练方案

```python
import torch.nn.functional as F

def get_angle_subsets(n_angles, K=4):
    """交错分割：每隔 K 个角度取一个，保证每个子集角度分布均匀"""
    idx = torch.arange(n_angles)
    return [idx[i::K] for i in range(K)]

def n2i_lpd_loss(model, y, make_ops, K=4):
    """
    N2I-LPD 自监督损失
    y:        完整 sinogram (B, 1, n_angles, n_dets)
    make_ops: fn(angle_indices) -> (forward_op, backward_op)
    """
    n_angles = y.shape[2]
    subsets  = get_angle_subsets(n_angles, K)
    total_loss = 0.0

    for i in range(K):
        held_idx  = subsets[i]
        train_idx = torch.cat([subsets[j] for j in range(K) if j != i])

        y_train = y[:, :, train_idx, :]   # 训练用的子集
        y_held  = y[:, :, held_idx, :]    # 留出的子集

        A_train, AT_train = make_ops(train_idx)
        x_hat = model(y_train, A_train, AT_train)  # 用 K-1 个子集重建

        A_held, _ = make_ops(held_idx)
        loss = F.mse_loss(A_held(x_hat), y_held)   # 与留出子集比较
        total_loss += loss

    return total_loss / K
```

### 训练循环

```python
def train(model, dataloader, make_ops, n_epochs=100, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    for epoch in range(n_epochs):
        for y_batch in dataloader:   # 只需噪声 sinogram，无需 ground truth
            optimizer.zero_grad()
            loss = n2i_lpd_loss(model, y_batch, make_ops, K=4)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防止展开迭代梯度爆炸
            optimizer.step()
        scheduler.step()
# ... （评估代码省略）
```

### 关键 Trick

**1. 交错角度分割**：用"每隔 K 个角度取一个"，而不是连续分块。连续分块会导致某个子集的角度覆盖严重不均匀，训练不稳定。

**2. FBP 初始化**：LPD 的原始变量初始为 FBP 重建而非全零，让网络专注 refinement，收敛快约 30%。

**3. 算子归一化**：很多 Radon 实现里 $A^T A$ 的特征值远大于 1，直接用会导致第一轮迭代的 $A^T u$ 数值极大，梯度爆炸。需要对算子输出做尺度归一化：

```python
# 估算归一化常数（只需执行一次）
with torch.no_grad():
    test_img = torch.ones(1, 1, H, W, device=device)
    scale = forward_op(test_img).abs().mean()
normalized_forward = lambda x: forward_op(x) / scale
normalized_backward = lambda u: backward_op(u) / scale
```

**4. 推理时的算子切换**：训练用部分角度算子，推理用完整角度算子，这是 N2I-LPD 固有的 domain gap。`torch-radon` 支持动态传入角度列表，是目前最方便的选择。

## 实验结果

### 方法对比

| 方法 | 是否需要 GT | 低剂量 PSNR | 稀疏角度 PSNR | 备注 |
|------|------------|------------|--------------|------|
| FBP | 否 | ~28 dB | ~22 dB | 基线 |
| TV 迭代 | 否 | ~31 dB | ~27 dB | 需手动调参 |
| U-Net (N2I) | 否 | ~33 dB | ~29 dB | 图像域后处理 |
| LPD（监督） | **是** | ~36 dB | ~33 dB | 性能上界 |
| **N2I-LPD** | **否** | **~35 dB** | **~32 dB** | 本文方法 |

核心结论：N2I-LPD 在无 ground truth 下，性能接近有监督 LPD（差距 <1 dB），并显著超过 N2I U-Net。

### 多随机种子可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(results: dict, title="N2I-LPD vs Baselines"):
    """results: {method_name: list of per-epoch PSNR arrays（每个 seed 一条）}"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, curves in results.items():
        arr  = np.array(curves)        # (n_seeds, n_epochs)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ep   = np.arange(len(mean))
        ax.plot(ep, mean, label=method)
        ax.fill_between(ep, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("PSNR (dB)")
    ax.legend(); ax.set_title(title)
    plt.tight_layout(); plt.show()
```

## 调试指南

### 常见问题

**1. 重建图像全黑或全噪声**

先单独测试算子：`backward_op(forward_op(test_img))` 结果是否大致等于 `test_img`。如果差了几个数量级，是算子归一化问题，见上面的 trick。

**2. N2I 损失下降但图像质量不提升**

大概率是子集分割有 bug，held-out 子集和 training 子集重叠了。打印两个索引列表，确认无交集：

```python
assert len(set(train_idx.tolist()) & set(held_idx.tolist())) == 0
```

**3. 训练损失正常，推理图像质量差**

这是 N2I-LPD 的固有问题：训练用部分角度算子，推理用完整算子，存在 domain shift。缓解方案：
- 减小 $K$（从 4 改为 2），每次 held-out 的角度更少，domain shift 更小
- 在少量有标签数据上做 fine-tuning

**4. 梯度爆炸（loss 变 NaN）**

LPD 的展开迭代极易梯度爆炸，`clip_grad_norm_(..., 1.0)` 是必须的。还不稳定的话，把 `n_iters` 从 10 降到 5，等训练稳定后再逐渐加回来。

### 超参数调优

| 参数 | 推荐值 | 敏感度 | 建议 |
|-----|--------|-------|------|
| `n_iters` | 10 | 中 | 先用 5，稳定后加到 10 |
| `n_channels` | 32 | 低 | 64 收益有限，不值得 |
| `K`（子集数） | 4 | 中 | 越大越慢，2-4 最常用 |
| `lr` | 1e-4 | 高 | 别超过 3e-4 |
| `batch_size` | 4-8 | 低 | 受显存限制 |

### 训练进展判断

- **前 5 个 epoch**：N2I 损失应快速下降。不动→检查算子实现
- **10-20 epoch**：PSNR 应高于 FBP 基线。没超过→学习率可能太大
- **50+ epoch**：缓慢提升，方差收窄为正常

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 临床 CT，无配对 ground truth | 有大量高质量配对数据（直接用监督 LPD） |
| 低剂量或稀疏角度重建 | 标准剂量全角度（FBP 已足够） |
| 不同扫描协议/设备迁移 | 要求实时重建（< 1 秒） |
| 科研/方法验证 | 需 FDA 认证的临床部署 |

## 我的看法

N2I-LPD 是**工程上很有价值**的工作，把 LPD 改造成自监督版本的性能代价相对较小。但有几个现实问题：

**训练-推理 domain gap 是根本缺陷**。训练用部分角度算子，推理用完整算子，这个不一致性难以完全消除。一种缓解思路是加入"自洽性约束"——额外约束重建结果在完整 sinogram 上的数据一致性，但会增加实现复杂度。

**Radon 变换的可微实现是工程瓶颈**。需要同时支持 GPU 加速、灵活角度子集、正确梯度反传，目前 `torch-radon` 是最成熟的选择，但在 Windows 上安装麻烦，Linux 环境下体验好得多。

**值得关注的竞争对手**：近两年基于 diffusion model 的无监督 CT 重建（DPS、DDNM 等）在部分场景下性能逼近甚至超过有监督方法，N2I-LPD 与这类方法的系统对比是未来的重要方向。

总体而言：手上有无标注 CT 数据、需要比 FBP/TV 更好质量的重建，N2I-LPD 是目前最值得认真尝试的自监督方案之一。