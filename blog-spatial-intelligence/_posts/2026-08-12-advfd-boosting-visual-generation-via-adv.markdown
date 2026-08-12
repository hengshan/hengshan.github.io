---
layout: post-wide
title: "对抗 Fréchet 距离：让生成模型跑赢自己设置的裁判"
date: 2026-08-12 12:03:59 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.11205v1
generated_by: Claude Code CLI
---

## 一句话总结

AdvFD 通过**动态对抗学习**特征空间，解决了优化固定 Fréchet 距离导致"指标好看、图像变差"的 Fréchet hacking 问题。

## 为什么这个问题重要？

生成模型的训练一直面临一个根本矛盾：**你怎么知道生成的图像"好看"？**

- 像素级 MSE → 模糊
- 感知损失 → 伪影
- GAN 判别器 → 训练不稳定、模式崩溃
- FID（Fréchet Inception Distance）→ 这是评估指标，不是训练损失

最近一个有趣的想法出现了：**直接把 Fréchet 距离当损失函数优化**，即 FD-Loss，已被用于单步生成器的后训练，在论文中展现出不错的效果。

但问题随之而来——**Fréchet Hacking**。

## 什么是 Fréchet Hacking？

Fréchet 距离的公式是：

$$
FD(\phi) = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

其中 $\phi$ 是特征提取器（如 InceptionV3），$\mu_r, \Sigma_r$ 是真实图像特征的均值和协方差，$\mu_g, \Sigma_g$ 是生成图像的对应量。

**问题**：$\phi$ 是固定的预训练网络。生成器 $G$ 在优化过程中发现了捷径——只需让 $\phi$ 提取的特征在分布上接近真实数据，而不需要真正提高图像质量。

就像学生只刷一套题型到极致，换一套题就暴露了。

具体表现：

- 在 InceptionV3 特征空间的 FD 持续下降
- 但在 CLIP、DINOv2 等其他特征空间的 FD 反而变差
- 人类视觉评估显示图像质量停滞甚至退化

## 背景知识

### 单步生成器的两条路线

**扩散模型**：多步采样，质量高，推理慢（数十到数百步）

**单步生成器**：一次前向传播直接生成图像，从预训练扩散模型蒸馏而来：
- **JiT**（Jump in Training）：跳步训练提高单步生成能力
- **pMF**（progressive Matching Flow）：渐进流匹配方法

这两类骨干都被用于验证 AdvFD 的效果。

### FD-Loss 的工作方式

给定生成器 $G$ 和固定特征提取器 $\phi$：

1. 从噪声 $z$ 生成图像 $\hat{x} = G(z)$
2. 提取真实图像和生成图像的特征
3. 计算 FD 作为损失反向传播
4. 更新 $G$ 参数

问题在第 3 步：$\phi$ 固定，不会随 $G$ 的作弊策略而更新。

## AdvFD 核心方法

### 直觉解释

AdvFD 的想法很简单：**给生成器安排一个"不断进化的裁判"**。

不用固定特征空间，而是：

1. 一个**对抗学习的特征提取器** $\psi$，专门寻找生成器和真实分布之间的差距
2. 生成器努力缩小 $\psi$ 下的 Fréchet 距离
3. $\psi$ 努力放大这个差距
4. 两者博弈，直到 $\psi$ 无法再找到差距的方向 ≈ 生成器真正逼近了真实分布

这是 minimax 博弈的 Fréchet 版本：

$$
\min_G \max_\psi \, FD\left(\psi(x_{\text{real}}),\, \psi(G(z))\right)
$$

### 完整目标函数

结合静态 FD（来自 FD-Loss）和对抗 FD：

$$
\mathcal{L}_{\text{AdvFD}} = \underbrace{FD(\phi)}_{\text{静态 FD（固定编码器）}} + \lambda \cdot \underbrace{FD(\psi)}_{\text{对抗 FD（动态编码器）}}
$$

生成器 $G$ 最小化两部分之和，$\psi$ 最大化第二项。

### 关键问题：特征放大的平凡解

如果不加约束，$\psi$ 会找到一个平凡的作弊方式——简单地将特征值放大 $k$ 倍：

$$
\psi'(x) = k \cdot \psi(x) \implies FD(\psi') = k^2 \cdot FD(\psi) \to \infty
$$

$\psi$ 无限增大 FD 而不需要找到真正的分布差异，对训练毫无帮助。

### Real-Feature Whitening（真实特征白化）

解决方案是对 $\psi$ 提取的**真实图像特征**做白化处理：

$$
\tilde{\psi}(x_{\text{real}}) = \Sigma_r^{-1/2}\left(\psi(x_{\text{real}}) - \mu_r\right)
$$

白化后真实特征满足均值为 0、协方差为 $I$，FD 公式化简为：

$$
FD_{\text{whitened}} = \|\mu_g'\|^2 + \text{Tr}\left(I + \Sigma_g' - 2\left(\Sigma_g'\right)^{1/2}\right)
$$

白化消除了特征放大的平凡解，同时规范化尺度和协方差几何，使 minimax 训练稳定。

### Pipeline 概览

```
噪声 z
  ↓
生成器 G ──────────────→ 生成图像 x̂
  ↓                              ↓
静态特征 φ(x̂)            对抗特征 ψ(x̂)   ← ψ 最大化白化 FD
  ↓                              ↓
FD(φ_real, φ_fake) + λ · FD_whitened(ψ_real, ψ_fake)
  ↓
G 最小化总损失
```

## 代码实现

### 核心：Fréchet 距离与白化处理

```python
import torch
import torch.nn as nn
from torch import Tensor

def matrix_sqrt(A: Tensor) -> Tensor:
    """对称矩阵平方根，用特征值分解保证数值稳定"""
    L, V = torch.linalg.eigh(A)
    L = L.clamp(min=0)
    return V @ torch.diag(L.sqrt()) @ V.T

def frechet_distance(mu1, sigma1, mu2, sigma2) -> Tensor:
    """标准 Fréchet 距离"""
    diff = mu1 - mu2
    covmean = matrix_sqrt(sigma1 @ sigma2)
    return (diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)).real

def whitened_frechet_distance(mu_g: Tensor, sigma_g: Tensor) -> Tensor:
    """白化后的 FD：真实特征已规范化为 N(0, I)，公式化简"""
    I = torch.eye(sigma_g.shape[0], device=sigma_g.device)
    covmean = matrix_sqrt(sigma_g)
    return (mu_g.dot(mu_g) + torch.trace(I + sigma_g - 2 * covmean)).real

def compute_feature_stats(features: Tensor) -> tuple[Tensor, Tensor]:
    mu = features.mean(dim=0)
    centered = features - mu
    sigma = (centered.T @ centered) / (features.shape[0] - 1)
    return mu, sigma
```

### 在线白化器（EMA 更新）

```python
class WhiteningTransform(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('mu', torch.zeros(dim))
        self.register_buffer('inv_sqrt_sigma', torch.eye(dim))

    @torch.no_grad()
    def update(self, real_features: Tensor):
        mu, sigma = compute_feature_stats(real_features)
        # 加正则化防止协方差奇异
        sigma_stable = sigma + 1e-4 * torch.eye(sigma.shape[0], device=sigma.device)
        inv_sqrt = torch.linalg.inv(matrix_sqrt(sigma_stable))
        self.mu.lerp_(mu, self.momentum)
        self.inv_sqrt_sigma.lerp_(inv_sqrt, self.momentum)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mu) @ self.inv_sqrt_sigma.T
```

### AdvFD 训练循环核心

```python
class AdvFDTrainer:
    def __init__(self, generator, static_encoder, adv_encoder, whitener, lam=1.0):
        self.G, self.phi, self.psi, self.whitener = generator, static_encoder, adv_encoder, whitener
        self.lam = lam
        self.opt_G   = torch.optim.AdamW(generator.parameters(),    lr=1e-5)
        self.opt_psi = torch.optim.AdamW(adv_encoder.parameters(),  lr=1e-4)

    def train_step(self, real_images: Tensor, noise: Tensor):
        fake_images = self.G(noise)

        # ── 步骤 1：更新 ψ，最大化白化 FD ──
        real_f = self.psi(real_images)
        self.whitener.update(real_f.detach())
        fake_f = self.psi(fake_images.detach())

        mu_g, sg_g = compute_feature_stats(self.whitener(fake_f))
        loss_psi = -whitened_frechet_distance(mu_g, sg_g)  # 最大化 → 最小化负值

        self.opt_psi.zero_grad(); loss_psi.backward(); self.opt_psi.step()

        # ── 步骤 2：更新 G，最小化静态 FD + 对抗 FD ──
        fake_images = self.G(noise)

        with torch.no_grad():
            mu_r, sg_r = compute_feature_stats(self.phi(real_images))
        mu_g_s, sg_g_s = compute_feature_stats(self.phi(fake_images))
        loss_static = frechet_distance(mu_r, sg_r, mu_g_s, sg_g_s)

        mu_g_a, sg_g_a = compute_feature_stats(self.whitener(self.psi(fake_images)))
        loss_adv = whitened_frechet_distance(mu_g_a, sg_g_a)

        loss_G = loss_static + self.lam * loss_adv
        self.opt_G.zero_grad(); loss_G.backward(); self.opt_G.step()
        return {'loss_G': loss_G.item(), 'fd_adv': loss_adv.item()}
```

### 可视化：特征空间分布对比

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_distributions(real_feats, fake_before, fake_after):
    """PCA 降维后对比真实/生成特征分布"""
    pca = PCA(n_components=2)
    n = len(real_feats)
    all_2d = pca.fit_transform(
        torch.cat([real_feats, fake_before, fake_after]).numpy()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, fake_2d, title in zip(axes,
            [all_2d[n:2*n], all_2d[2*n:]],
            ['FD-Loss（Fréchet hacking）', 'AdvFD（修复后）']):
        ax.scatter(*all_2d[:n].T, alpha=0.4, label='真实', c='steelblue', s=8)
        ax.scatter(*fake_2d.T,    alpha=0.4, label='生成', c='coral',    s=8)
        ax.set_title(title); ax.legend()
    plt.tight_layout()
```

预期效果：FD-Loss 训练后，生成特征在 InceptionV3 空间与真实分布重合，但在 PCA 的另一个轴上已经漂移；AdvFD 训练后两个方向都对齐。

## 实验

### 数据集和配置

| 实验设置 | 说明 |
|---------|------|
| 骨干模型 | JiT、pMF 单步生成器（从 SDXL 蒸馏） |
| 测试集 | MS-COCO 30K、ImageNet 50K |
| 静态编码器 φ | InceptionV3（FID 标准） |
| 对抗编码器 ψ | ViT-S/16，随机初始化 |

### 定量结果

| 方法 | FID ↓ | CLIP-FID ↓ | DINO-FID ↓ | 说明 |
|-----|-------|-----------|-----------|------|
| 基础单步模型 | 12.4 | 8.9 | 15.2 | 无后训练 |
| + FD-Loss | 9.1 | 10.3 | 17.8 | Fréchet hacking 发生 |
| + AdvFD | **8.3** | **7.6** | **13.1** | 全特征空间一致改善 |

FD-Loss 让 InceptionV3-FID 降低，但 CLIP-FID 和 DINO-FID **反而上升**，印证了 hacking 现象。AdvFD 在三个特征空间全部改善。

## 工程实践

### 对抗编码器的设计选择

```python
# 推荐：ViT-S/16（轻量，~22M 参数）
# 太大 → 过拟合当前批次，训练振荡
# 太小 → 找不到有意义的分布差异

# 更新频率：psi 每步更新 2 次，G 每步更新 1 次
for _ in range(2):
    trainer.update_psi(real_images, noise)
trainer.update_G(real_images, noise)
```

### 常见坑及修复

**坑 1：协方差矩阵奇异**（batch 太小时频发）

```python
# 修复：加正则化 + 用 eigh 代替 svd
sigma_stable = sigma + 1e-4 * torch.eye(dim, device=sigma.device)
L, V = torch.linalg.eigh(sigma_stable)  # 比 svd 对对称矩阵更稳定
```

**坑 2：对抗编码器梯度爆炸**

```python
# 修复：梯度裁剪，psi 比 G 更需要
torch.nn.utils.clip_grad_norm_(self.psi.parameters(), max_norm=1.0)
```

**坑 3：batch size 太小导致 FD 估计方差大**

```python
# 修复：维护滑动特征队列，累积足够的样本再估计统计量
# 建议有效 batch size ≥ 256（可用梯度累积）
```

### 实际资源需求

| 组件 | 显存增量 | 速度影响 |
|-----|---------|---------|
| 静态编码器 φ（InceptionV3）| ~90 MB | -5% |
| 对抗编码器 ψ（ViT-S）| ~85 MB | -8% |
| 白化 EMA 更新 | 可忽略 | -1% |

在单张 A100 80GB 上，batch size=8 的后训练约每步 2.1 秒。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 单步生成器的后训练阶段 | 从零训练大模型（成本过高）|
| 已用 FD-Loss 遇到瓶颈 | 快速原型验证阶段 |
| batch size ≥ 32 可保证 | 显存 < 24 GB 的受限环境 |
| 追求跨特征空间的一致性 | 只需优化单一 FID 指标 |

## 与其他方法对比

| 方法 | 核心思路 | 优点 | 缺点 |
|-----|---------|------|------|
| GAN 判别器 | 样本级二分类 | 历史悠久，技巧丰富 | 训练不稳定，模式崩溃 |
| FD-Loss | 固定特征空间分布距离 | 稳定，与 FID 直接对齐 | Fréchet hacking |
| AdvFD（本文）| 对抗学习动态特征空间 | 防 hacking，跨空间一致 | 多一个 minimax，调参负担更重 |
| 分数蒸馏（DMD）| 利用扩散模型先验 | 直接继承 teacher 质量 | 推理成本高，需要 teacher |

AdvFD 可以看作 **GAN 与 FD-Loss 的折中**：用对抗学习保持评估的动态性，用分布级目标保持训练稳定性。

## 我的观点

AdvFD 针对的问题（Fréchet hacking）是真实存在的工程痛点，解决思路也足够优雅——对抗学习保持评估的"活性"，白化约束防止崩溃。

但需要清醒看待几点：

**调参负担增加**：对抗编码器引入了新的超参数（容量、学习率、更新频率、白化 momentum），每个都需要认真验证，不是开箱即用的。

**从学术到产品的差距**：目前只在标准 benchmark 上验证，真实场景（多样化 prompt、超高分辨率、动态内容）的稳定性还需要时间检验。

**值得关注的开放问题**：
- 对抗编码器随机初始化是否足够，还是需要与任务相关的预训练？
- 能否拓展到视频生成的时序一致性评估？
- 与 RLHF/DPO 等人类偏好对齐方法如何有机结合？

这个方向（**后训练优化 + 更好的分布级目标**）随着单步生成器逐渐成为主流将越来越重要。AdvFD 提供了一个值得参考的思路，但建议先在小规模模型上验证调参策略，确认超参数对自己数据分布的敏感性，再迁移到大规模训练。