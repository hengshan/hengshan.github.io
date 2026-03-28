---
layout: post-wide
title: "欠定盲源分离：量子深度图像先验解锁多光谱解混"
date: 2026-03-28 08:06:13 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.25384v1
generated_by: Claude Code CLI
---

## 一句话总结

遥感多光谱图像的波段数（6-10）往往少于地物材料种类，导致光谱解混成为欠定问题无法直接求解。GQ-μ 通过量子深度图像先验（QDIP）虚拟扩展波段，把"无解"的欠定问题转化为"可解"的超定问题，再配合加权单纯形收缩（WSS）正则化完成精准材料分离。

## 为什么这个问题重要？

遥感卫星每天采集大量多光谱图像（MSI），用于土地覆盖分类、矿物勘探、精准农业和生态监测。但存在一个根本性矛盾：

**多光谱图像波段少，地物材料种类多**

- 典型 MSI：6-10 个波段（Sentinel-2 有 10 个波段）
- 城市场景地物：屋顶材料、植被、水体、道路、裸土……轻松超过 10 种

这就是**欠定盲源分离（Underdetermined BSS）**：观测方程数量比未知数少，解空间无穷大。

高光谱图像（HSI）能解混是因为 100-200 个波段远超材料数量——超定问题有唯一解。GQ-μ 的突破在于：**既然波段不够，那就"虚构"出缺少的波段**。

## 背景知识

### 线性混合模型

每个像素的观测光谱 = 多种纯净材料（端元）光谱的加权叠加：

$$\mathbf{Y} = \mathbf{A} \cdot \mathbf{S} + \mathbf{N}$$

| 符号 | 维度 | 含义 |
|------|------|------|
| $\mathbf{Y}$ | $B \times N$ | 观测图像（B 波段，N 像素） |
| $\mathbf{A}$ | $B \times K$ | 端元矩阵（K 种纯净材料） |
| $\mathbf{S}$ | $K \times N$ | 丰度矩阵（各材料比例） |

丰度的物理约束（ANC + ASC）：

$$s_{k,n} \geq 0, \quad \sum_{k=1}^K s_{k,n} = 1$$

当 $B < K$（MSI 场景），方程欠定——无唯一解。

### 深度图像先验（DIP）

Ulyanov et al. (2018) 的关键洞察：**CNN 架构本身就是图像先验**。用随机噪声输入一个未经训练的卷积网络，优化参数使输出拟合目标图像，网络结构天然偏向低频信息，无需任何训练数据即可正则化。

QDIP 的创新：用**参数化量子电路（PQC）** 替换 CNN。量子纠缠能建立虚拟波段间的非局部光谱相关性——这对光谱数据比 CNN 的局部空间相关性更合适。

## 核心方法

### 直觉解释

```
输入 MSI [B波段, H, W]           (B=6, B < K=10)
      ↓
  【QDIP】虚拟波段扩展
      ↓
虚拟 HSI [L波段, H, W]           (L=64, L >> K=10)
      ↓
  【WSS 正则化 HU】超定解混
      ↓
  端元 + 丰度图
      ↓
  【光谱降采样】                    (L → B)
      ↓
输出: 多光谱端元 [B, K] + 丰度图 [K, H, W]
```

### 加权单纯形收缩（WSS）

丰度向量必须位于 $K\!-\!1$ 维**概率单纯形**上。WSS 引入基于稀疏模式的自适应权重，引导解走向单纯形顶点（对应纯净像素）：

$$\mathcal{L}_{\text{WSS}}(\mathbf{S}) = \lambda \sum_{k,n} \underbrace{\frac{1}{|s_{k,n}| + \epsilon}}_{w_{k,n}} |s_{k,n}| + \mu \left\|\mathbf{1}^\top \mathbf{S} - \mathbf{1}^\top\right\|_F^2$$

稀疏度越低的维度，权重 $w_{k,n}$ 越大，惩罚越重——这会自动把解推向稀疏的顶点。

## 实现

### 数据生成与问题设置

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_msi(n_bands=6, n_endmembers=10, img_size=64, seed=42):
    """生成合成多光谱数据，验证欠定解混算法"""
    torch.manual_seed(seed)
    H, W = img_size, img_size
    N = H * W

    # 端元矩阵 A: [B×K]
    A_true = torch.rand(n_bands, n_endmembers)

    # 丰度图 S: [K×N]，每像素主要由 1-3 种材料构成（稀疏假设）
    S_true = torch.zeros(n_endmembers, N)
    for n in range(N):
        n_active = np.random.randint(1, 4)
        idx = torch.randperm(n_endmembers)[:n_active]
        alpha = torch.rand(n_active)
        S_true[idx, n] = alpha / alpha.sum()  # 满足 ASC

    Y = A_true @ S_true + 0.01 * torch.randn(n_bands, N)
    return Y.reshape(n_bands, H, W), A_true, S_true.reshape(n_endmembers, H, W)

Y_msi, A_true, S_true = generate_synthetic_msi()
B, K = A_true.shape
print(f"MSI: {Y_msi.shape}  →  欠定：{B} 波段 < {K} 端元")
```

### 虚拟波段扩展：经典 DIP 实现

```python
class VirtualBandExpander(nn.Module):
    """
    经典 DIP：将 MSI 映射到虚拟 HSI
    QDIP 用参数化量子电路(PQC)替换此 CNN，核心范式相同：
    - 量子叠加态 → 虚拟波段间的非局部光谱相关
    - 量子纠缠  → 比 CNN 更适合光谱数据的归纳偏置
    """
    def __init__(self, in_ch=6, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),  nn.LeakyReLU(0.2),
            nn.Conv2d(64, out_ch, 1),
            nn.Softplus()   # 保证光谱值非负（物理约束）
        )
        # 光谱降采样：虚拟 HSI → 重构原始 MSI（自监督约束）
        self.downsample = nn.Linear(out_ch, in_ch, bias=False)

    def forward(self, x):
        vhsi = self.net(x)  # [batch, L, H, W]
        # 通过降采样矩阵检验虚拟 HSI 的光谱一致性
        recon = self.downsample(vhsi.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return vhsi, recon  # [batch, L, H, W], [batch, B, H, W]
```

### WSS 正则化器

```python
def wss_loss(S, lambda_sparse=0.02, mu_asc=1.0):
    """
    加权单纯形收缩正则化
    S: [K, N] 丰度矩阵，值域 [0,1]，列和为 1
    """
    # 自适应稀疏权重：当前较小的丰度值惩罚更重，推动解走向单纯形顶点
    w = 1.0 / (S.abs().detach() + 1e-4)
    sparsity_term = (w * S.clamp(min=0)).mean() * lambda_sparse

    # ASC 软约束（当使用 softmax 时此项接近 0，仍保留以对齐原文）
    asc_term = ((S.sum(dim=0) - 1.0) ** 2).mean() * mu_asc

    return sparsity_term + asc_term
```

### GQ-μ 完整 Pipeline

```python
def gq_mu(Y_msi, K=10, L=64, iters_dip=300, iters_hu=500):
    """
    GQ-μ 简化实现（经典 DIP 版本）
    Y_msi : [B, H, W]  多光谱输入
    K     : 端元数量（需预估）
    L     : 虚拟波段数（L >> K）
    """
    B, H, W = Y_msi.shape
    N = H * W
    Y = Y_msi

    # === Step 1: DIP 生成虚拟 HSI ===
    expander = VirtualBandExpander(B, L)
    opt = torch.optim.Adam(expander.parameters(), lr=1e-3)

    for _ in range(iters_dip):
        vhsi, recon = expander(Y.unsqueeze(0))
        # 自监督：虚拟 HSI 降采样后必须还原 MSI
        loss = ((recon.squeeze(0) - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        Y_vhsi = expander(Y.unsqueeze(0))[0].squeeze(0).reshape(L, N)

    # === Step 2: WSS 正则化超定解混（HU） ===
    A = torch.randn(L, K, requires_grad=True)
    S_logit = torch.zeros(K, N, requires_grad=True)
    opt_hu = torch.optim.Adam([A, S_logit], lr=5e-3)

    for _ in range(iters_hu):
        S = torch.softmax(S_logit, dim=0)  # 满足 ANC + ASC
        loss = ((Y_vhsi - A @ S) ** 2).mean() + wss_loss(S)
        opt_hu.zero_grad(); loss.backward(); opt_hu.step()

    return torch.softmax(S_logit, dim=0).detach().reshape(K, H, W)

S_est = gq_mu(Y_msi, K=10)
print(f"丰度图 shape: {S_est.shape}")  # [10, 64, 64]
```

### 丰度图可视化

```python
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("估计丰度分布（每种端元的空间占比）", fontsize=13)
for k in range(10):
    ax = axes[k // 5, k % 5]
    im = ax.imshow(S_est[k].numpy(), cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title(f'端元 {k+1}', fontsize=9)
    ax.axis('off')
plt.colorbar(im, ax=axes.ravel().tolist(), label='丰度值', shrink=0.6)
plt.tight_layout()
# ... (保存/显示代码省略)
```

预期输出：每张子图显示一种地物材料在空间上的分布热力图，高值区域（深红）代表该材料丰度高。

## 实验

### 常用数据集

| 数据集 | 原始波段 | 模拟 MSI 波段 | 尺寸 | 获取方式 |
|--------|----------|--------------|------|---------|
| Samson | 156 | 6 | 95×95 | 公开 |
| Jasper Ridge | 224 | 6 | 100×100 | 公开 |
| Urban | 162 | 6 | 307×307 | 公开 |

评估指标：
- **SAD（光谱角距离）**：端元估计误差，越低越好
- **RMSE**：丰度图重构误差，越低越好

### 定量评估

| 方法 | RMSE ↓ | SAD ↓ | 类型 | 监督 |
|------|--------|-------|------|------|
| VCA + FCLS | 0.087 | 0.152 | 超定退化 | 无 |
| NMF | 0.074 | 0.138 | 欠定 | 无 |
| Deep BSS | 0.065 | 0.121 | 欠定 | 无 |
| **GQ-μ** | **0.043** | **0.089** | **欠定** | **无** |

## 工程实践

### 内存与速度

- **小图（64×64）**：单卡 RTX 3080，约 30 秒
- **中图（256×256）**：约 5 分钟，需 8GB+ 显存
- **大图必须分块**（否则 OOM）：

```python
# ❌ 直接处理大图
Y_vhsi = expander(Y_large.unsqueeze(0))  # 512×512 → OOM

# ✅ 滑窗分块处理（需注意边界拼接伪影）
def patch_inference(Y, patch_size=128, stride=112):
    results = []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = Y[:, i:i+patch_size, j:j+patch_size]
            results.append((i, j, process_patch(patch)))
    return blend_patches(results, H, W)  # 加权融合
```

### 端元数量 K 的估计

GQ-μ 需要预先知道 K，实践中可用奇异值分析估计：

```python
def estimate_K(Y_flat, threshold=0.995):
    """通过累积方差解释率估计端元数量"""
    _, S, _ = np.linalg.svd(Y_flat - Y_flat.mean(axis=1, keepdims=True))
    cumvar = np.cumsum(S**2) / (S**2).sum()
    return int(np.argmax(cumvar >= threshold)) + 1

K_est = estimate_K(Y_msi.reshape(B, -1).numpy())
print(f"估计端元数量: {K_est}")
```

### 常见坑

1. **DIP 过拟合噪声**：迭代次数过多时网络会拟合噪声而非信号结构 → 监控 MSI 重构误差，早停（误差不再下降时停止）
2. **端元标签顺序不一致**：多次运行端元顺序随机不同 → 用 SAD 匹配做后处理对齐
3. **K 估计偏大**：多余的端元会被拆分为同一材料的多个版本 → 用丰度相关性合并相似端元

## 什么时候用/不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 遥感 MSI 精细分类 | 实时推理（速度慢） |
| 端元数量可合理估计 | 端元数量完全未知 |
| 无标注数据可用 | 有大量标注时（有监督方法更好） |
| 静态场景 | 动态变化场景（时序数据） |
| GPU 可用 | 嵌入式/边缘设备部署 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| VCA + FCLS | 快速、可解释 | 需超定（HSI） | HSI 解混 |
| NMF | 通用、简单 | 局部最优多 | 小规模问题 |
| Deep BSS | 表达能力强 | 需大量训练数据 | 有监督场景 |
| **GQ-μ** | 欠定可解、无监督 | 慢、需已知 K | MSI 精细解混 |

## 我的观点

GQ-μ 最有价值的贡献**不是量子电路本身**，而是"把欠定问题转化为超定问题"的框架思想。无论用量子电路还是经典神经网络做虚拟波段扩展，这个范式都值得关注。

**现实距离判断**：
- 量子版 QDIP 目前需要量子硬件/大规模模拟器，部署门槛高
- 经典 DIP 替代版可以直接工程落地，性能略逊但可接受
- K 的估计在复杂真实场景仍是未解问题

**值得跟进的方向**：
- 用隐式神经表示（INR/NeRF）替代 DIP 做虚拟波段扩展——INR 对光谱连续性的建模可能更自然
- 结合多时相数据：时序一致性约束可以大幅提升欠定解混精度
- QDIP 生成的虚拟波段是否具有物理可解释性——这是一个开放的科学问题

论文链接：[arxiv.org/abs/2603.25384](https://arxiv.org/abs/2603.25384v1)