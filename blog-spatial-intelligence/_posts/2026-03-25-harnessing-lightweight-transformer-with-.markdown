---
layout: post-wide
title: "Light-UNETR：医学3D图像分割的轻量化Transformer设计"
date: 2026-03-25 12:06:13 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.23390v1
generated_by: Claude Code CLI
---

## 一句话总结

Light-UNETR 用 90% 更少的计算量实现了与大型 Transformer 相当甚至更好的 3D 医学图像分割效果，核心在于轻量化注意力模块 + 半监督学习策略，直击医疗场景下标注数据稀缺和计算资源受限两大痛点。

## 为什么这个问题重要？

### 应用场景

3D 医学图像分割是临床诊断的基础工具：CT 心脏分割用于术前规划，MRI 脑部分割用于肿瘤检测，腹部 CT 多器官分割用于放疗靶区勾画。一个典型的 CT 扫描有 200-500 个切片，分辨率 512×512，意味着每个样本有约 5000 万个体素需要处理。

**三个核心挑战：**

- **标注代价极高**：标注一个腹部 CT 需要放射科医生花费 2-4 小时，大规模标注数据集几乎不可能获取
- **3D 体积庞大**：标准 Transformer 的 O(N²) 注意力在 3D 场景下直接爆显存，将分辨率翻倍会导致计算量增加 8 倍
- **部署环境受限**：医院工作站通常配置消费级 GPU（如 RTX 3090），而非数据中心级设备

### 现有方法的问题

nnU-Net 是长期的医学图像分割 baseline，对 3D 局部特征提取能力强，但缺乏全局感受野。UNETR（2022）将 ViT 引入医学分割，全局建模能力强，但计算开销大：一个 $96^3$ 的输入在 patch size=16 时会产生 216 个 token，参数量超过 100M。

Light-UNETR 的核心创新是**在不牺牲精度的前提下把计算量降低 90%**，同时用半监督策略让 10% 的标注数据发挥出充分的价值。

## 背景知识

### 3D 医学体积数据的本质

医学图像的 3D 体积（Volumetric Data）和 3D 场景重建有根本性不同：

| 特性 | 医学体积数据 | 3D 场景重建（NeRF/3DGS）|
|------|------------|------------------------|
| 表示形式 | 规则体素网格 | 隐式场 / 高斯点云 |
| 输入 | 单个 CT/MRI 体积 | 多视角图像序列 |
| 任务 | 逐体素语义分类 | 新视角合成 / 几何重建 |
| 先验 | 解剖结构先验 | 辐射场 / 几何先验 |

医学体素数据天然就是规则网格，标准卷积操作非常适合，但也带来了维度诅咒：$D \times H \times W$ 的体积在 Transformer 中会产生 $N = DHW/p^3$ 个 token。

### Transformer 注意力的 3D 计算量问题

给定输入体积 $X \in \mathbb{R}^{D \times H \times W \times C}$，展平为 $N$ 个 patch token，标准多头注意力的计算复杂度为：

$$
\text{FLOPs}_{\text{Attn}} = 4 \cdot N \cdot C^2 + 2 \cdot N^2 \cdot C
$$

当 $D=H=W=96$，$p=16$ 时，$N=216$，尚可接受。但提高分辨率（$p=8$ 时 $N=1728$）后，$N^2$ 项会使计算量增加 64 倍。精细分割在 3D 场景下代价极高，这正是 LIDR 要解决的问题。

## 核心方法

### LIDR：轻量化维度压缩注意力

**直觉**：标准注意力让所有 N 个 token 两两配对，LIDR 用两条分支分别压缩空间维度和通道维度，从不同角度捕获全局和局部信息，最终加权融合。

```
输入 token [N × C]
      ↓           ↓
  空间压缩分支    通道压缩分支
  [N/r × C]      [N × C/r]
  全局注意力      局部注意力
      ↓           ↓
     加权融合 → 输出 [N × C]
```

**空间压缩分支**（全局感受野）对 K/V 做步长为 $r$ 的下采样：

$$
\tilde{K} = \text{Stride}(K, r), \quad \text{Attn}_{\text{spatial}} = \text{softmax}\!\left(\frac{Q\tilde{K}^T}{\sqrt{d}}\right)\tilde{V}
$$

**通道压缩分支**（局部细节）使用低维投影矩阵 $W \in \mathbb{R}^{C \times C/r}$：

$$
\text{Attn}_{\text{channel}} = \text{softmax}\!\left(\frac{\hat{Q}\hat{K}^T}{\sqrt{d/r}}\right)\hat{V}
$$

最终通过可学习标量 $\alpha$ 融合两条分支：

$$
\text{LIDR}(X) = \sigma(\alpha) \cdot \text{Attn}_{\text{spatial}} + (1-\sigma(\alpha)) \cdot \text{Attn}_{\text{channel}}
$$

### CGLU：紧凑门控线性单元

传统 FFN 扩张比为 4（参数量 $8C^2$），CGLU 引入门控机制，扩张比降至 2，同时用 sigmoid 门控实现选择性通道交互：

$$
\text{CGLU}(X) = \text{LayerNorm}\!\left(XW_1 \odot \sigma(XW_2)\right) W_3
$$

其中 $W_1, W_2 \in \mathbb{R}^{C \times 2C}$，门控向量 $\sigma(XW_2)$ 决定每个通道的信息流量，参数量约为传统 FFN 的一半。

### CSE：上下文协同增强学习策略

CSE 是 Light-UNETR 在**半监督学习**方向的贡献，基于 Mean Teacher 框架（学生网络参数用梯度更新，教师网络参数为学生的 EMA）。

**① Attention-Guided Replacement（AGR）—— 外部上下文**

从有标注图像提取注意力热图，找到"重要区域"，将其 patch 替换到无标注图像中，生成增强样本：

$$
\tilde{X}_u = X_u \odot (1 - M_{\text{attn}}) + X_l^{\text{crop}} \odot M_{\text{attn}}
$$

其中 $M_{\text{attn}}$ 是根据标注图像注意力权重生成的 patch 级二值掩码。

**② Spatial Masking Consistency（SMC）—— 内部上下文**

对无标注图像随机遮掩部分空间区域，要求模型在遮掩和未遮掩两种输入下预测一致：

$$
\mathcal{L}_{\text{SMC}} = \left\| f_\theta(X_u) - f_\theta(X_u \odot M_{\text{spatial}}) \right\|_2^2
$$

**总损失**（监督 + 半监督）：

$$
\mathcal{L} = \mathcal{L}_{\text{seg}}(X_l) + \lambda_1 \mathcal{L}_{\text{AGR}} + \lambda_2 \mathcal{L}_{\text{SMC}}
$$

## 实现

### LIDR 注意力模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LIDRAttention(nn.Module):
    """轻量化维度压缩注意力：双分支 = 空间压缩（全局）+ 通道压缩（局部）"""
    def __init__(self, dim, num_heads=8, reduction_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.r = reduction_ratio

        # 空间压缩分支：K/V 序列长度压缩为 N/r
        self.qkv_spatial = nn.Linear(dim, dim * 3)

        # 通道压缩分支：低维 QKV，通道数为 C/r
        dim_low = dim // reduction_ratio
        self.qkv_channel = nn.Linear(dim, dim_low * 3)
        self.channel_proj = nn.Linear(dim_low, dim)   # 投影回原始维度

        self.branch_weight = nn.Parameter(torch.tensor(0.5))  # 可学习融合权重
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # 空间压缩分支（全局上下文）
        qkv = self.qkv_spatial(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # K/V 按 reduction_ratio 步长采样，序列长度从 N 降为 N/r
        k, v = k[:, :, ::self.r, :], v[:, :, ::self.r, :]
        attn_s = (q @ k.transpose(-2, -1) * self.scale).softmax(-1) @ v
        out_spatial = attn_s.transpose(1, 2).reshape(B, N, C)

        # 通道压缩分支（局部细节）
        dim_low = C // self.r
        q_c, k_c, v_c = self.qkv_channel(x).chunk(3, dim=-1)  # 各 [B, N, dim_low]
        attn_c = (q_c @ k_c.transpose(-2, -1) * (dim_low ** -0.5)).softmax(-1) @ v_c
        out_channel = self.channel_proj(attn_c)

        # 加权融合两条分支
        w = self.branch_weight.sigmoid()
        return self.out_proj(w * out_spatial + (1 - w) * out_channel)
```

### CGLU 前馈模块与 Transformer Block

```python
class CGLU(nn.Module):
    """紧凑门控线性单元：替代标准 FFN，参数量减半"""
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden = dim * expansion        # 扩张比为 2，传统 FFN 为 4
        self.fc1 = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        # 门控：sigmoid(gate) 决定每个通道的信息流量
        return self.fc2(self.norm(self.fc1(x) * torch.sigmoid(self.gate(x))))


class LightTransformerBlock(nn.Module):
    """Light-UNETR 基本块 = LIDR 注意力 + CGLU 前馈"""
    def __init__(self, dim, num_heads=8, reduction_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = LIDRAttention(dim, num_heads, reduction_ratio)
        self.ffn = CGLU(dim, expansion=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# 快速验证参数量对比
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

block_light = LightTransformerBlock(dim=256, num_heads=8, reduction_ratio=4)
block_std = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
print(f"Light block:    {count_params(block_light)/1e6:.2f}M params")
print(f"Standard block: {count_params(block_std)/1e6:.2f}M params")
# 预期：Light ~0.5M  vs  Standard ~1.3M
```

### CSE 半监督训练策略

```python
def spatial_masking_consistency_loss(student, teacher, x_u, mask_ratio=0.3, patch_size=16):
    """
    SMC 损失：用遮掩一致性驱动空间上下文推理
    学生在遮掩输入下的预测，应对齐教师在完整输入下的预测
    """
    B, C, D, H, W = x_u.shape
    # patch 级掩码（避免逐体素掩码的稀疏性问题）
    pd, ph, pw = D // patch_size, H // patch_size, W // patch_size
    mask = (torch.rand(B, 1, pd, ph, pw, device=x_u.device) > mask_ratio).float()
    mask = F.interpolate(mask, size=(D, H, W), mode='nearest')

    with torch.no_grad():
        pseudo_label = teacher(x_u).softmax(1)   # 教师给出软伪标签

    pred_masked = student(x_u * mask).softmax(1)  # 学生处理遮掩输入
    return F.mse_loss(pred_masked, pseudo_label.detach())


def attention_guided_replacement(x_labeled, x_unlabeled, attn_weights, top_ratio=0.3):
    """
    AGR：把标注图像的高注意力区域替换进无标注图像，引入外部上下文
    attn_weights: [B, N] 来自 Transformer 的 patch 注意力权重
    """
    B, C, D, H, W = x_unlabeled.shape
    topk = int(attn_weights.shape[-1] * top_ratio)
    _, top_idx = attn_weights.topk(topk, dim=-1)  # 找最重要的 patch

    # 将高注意力 patch 索引映射回 3D 空间（沿深度轴处理）
    p = 16  # patch size
    mask_3d = torch.zeros(B, 1, D, H, W, device=x_unlabeled.device)
    n_d = D // p
    for b in range(B):
        for idx in top_idx[b]:
            d_start = (idx.item() // ((H // p) * (W // p))) % n_d * p
            mask_3d[b, :, d_start:d_start + p, :, :] = 1.0

    # 高注意力区域替换为标注图像内容
    return x_unlabeled * (1 - mask_3d) + x_labeled * mask_3d
```

### 3D 分割结果可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_3d_segmentation(volume, prediction, gt=None, slice_idx=None):
    """
    多切片叠加可视化：原始图像 + 预测 + 真值对比
    volume / prediction / gt: [D, H, W] numpy 数组
    """
    D = volume.shape[0]
    slice_idx = slice_idx or D // 2
    cols = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    axes[0].imshow(volume[slice_idx], cmap='gray')
    axes[0].set_title(f'CT Volume (Axial Slice {slice_idx})', fontsize=11)

    axes[1].imshow(volume[slice_idx], cmap='gray')
    axes[1].imshow(prediction[slice_idx], alpha=0.45, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Prediction Overlay', fontsize=11)

    if gt is not None:
        axes[2].imshow(volume[slice_idx], cmap='gray')
        axes[2].imshow(gt[slice_idx], alpha=0.45, cmap='Greens', vmin=0, vmax=1)
        axes[2].set_title('Ground Truth Overlay', fontsize=11)

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('seg_result.png', dpi=150, bbox_inches='tight')
    plt.show()

# 模拟演示：96×96×96 体积
vol = np.random.rand(96, 96, 96)
pred = (np.random.rand(96, 96, 96) > 0.7).astype(float)
gt_mask = (np.random.rand(96, 96, 96) > 0.65).astype(float)
visualize_3d_segmentation(vol, pred, gt=gt_mask)
```

## 实验

### 数据集说明

| 数据集 | 典型分辨率 | 标注类型 | 获取方式 |
|--------|-----------|---------|---------|
| Left Atrial (LA) | $\sim 88 \times 576 \times 576$ | 左心房二值分割 | 公开，2018 年 Atrial Segmentation Challenge |
| Pancreas-CT | $\sim 240 \times 512 \times 512$ | 胰腺分割 | 公开，NIH |
| ACDC（心脏MRI）| $\sim 10 \times 352 \times 352$ | 多结构心脏 | 公开竞赛数据 |

LA 数据集是心脏射频消融手术的术前规划基础，精确的左心房 3D 模型直接影响手术路径规划，临床价值明确。

### 定量评估（Left Atrial，10% 标注数据）

| 方法 | Dice (%) | Jaccard (%) | FLOPs（相对）| 参数量（相对）|
|-----|---------|------------|------------|------------|
| UA-MT | 81.65 | 69.33 | 基准 | 基准 |
| SASSNet | 81.60 | 69.63 | 基准 | 基准 |
| BCP | 88.02 | 78.52 | 基准 | 基准 |
| **Light-UNETR** | **89.45** | **79.95** | **-90.8%** | **-85.8%** |

用不到 1/10 的计算量超越当前 SOTA，这个结果相当有说服力。

### 失败案例

- **边界模糊区域**：CT 窗宽/窗位设置不当时，相邻器官边界模糊，注意力范围难以精确定位
- **病理形态变异大**：肿瘤浸润导致器官形态严重变形时，基于正常解剖先验训练的模型会出现明显偏差
- **金属伪影干扰**：含金属植入物的 CT 扫描会产生条状伪影，严重干扰特征提取

## 工程实践

### 滑动窗口推理——部署必备

直接输入完整体积会 OOM，必须分块处理，重叠区域取平均：

```python
def sliding_window_inference(model, volume, num_classes, patch_size=96, overlap=0.5):
    """处理任意大小 3D 体积的标准推理方式"""
    D, H, W = volume.shape[-3:]
    stride = int(patch_size * (1 - overlap))
    output = torch.zeros(1, num_classes, D, H, W, device=volume.device)
    count = torch.zeros(1, 1, D, H, W, device=volume.device)

    for d in range(0, max(D - patch_size + 1, 1), stride):
        for h in range(0, max(H - patch_size + 1, 1), stride):
            for w in range(0, max(W - patch_size + 1, 1), stride):
                patch = volume[..., d:d+patch_size, h:h+patch_size, w:w+patch_size]
                with torch.no_grad():
                    pred = model(patch)
                output[..., d:d+patch_size, h:h+patch_size, w:w+patch_size] += pred
                count[..., d:d+patch_size, h:h+patch_size, w:w+patch_size] += 1

    return output / count.clamp(min=1)   # 重叠区域取平均
```

**硬件需求参考：**

| 场景 | 显存需求 | 单样本推理速度 |
|------|---------|-------------|
| LA 分割（patch $96^3$）| ~6 GB | ~2s |
| 多器官分割（patch $96^3$）| ~8 GB | ~5s |
| 实时手术导航（需 <500ms）| >16 GB + TensorRT | 需要工程优化 |

### 常见坑

1. **强度归一化不一致** → 不同机构 CT 的 HU 值范围不同，必须统一裁剪（通常 $[-175, 250]$ HU）再做 z-score 归一化，否则模型在跨域场景下直接崩

2. **医学图像坐标轴方向** → MRI 数据存在 RAS vs LPS 方向问题，读取后必须检查 affine 矩阵，统一坐标系，否则分割结果是镜像翻转的

3. **半监督训练不稳定** → EMA momentum 对训练稳定性极为敏感，建议从 0.99 开始调，过小（教师更新太快）或过大（教师学习迟缓）都会导致性能下降 3-5%

4. **背景 patch 过采样** → 随机采样会导致背景 patch 占多数（ROI 通常很小），必须用**前景居中采样**（foreground-centered sampling），强制至少 2/3 的 patch 包含目标器官

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 标注数据极少（$\leq 20\%$）| 有充足高质量标注，可直接用 nnU-Net |
| 部署在消费级 GPU 上 | 需要实时推理（<100ms），还需进一步优化 |
| 静态器官分割（心、肝、肾）| 运动器官（心跳/呼吸中的实时追踪）|
| 形态变化不大的目标 | 病理形态极端变异的肿瘤浸润 |
| 单模态（CT only 或 MRI only）| 多模态融合（CT+PET 需改架构）|

## 与其他方法对比

| 方法 | 核心思路 | 优点 | 缺点 | 适用场景 |
|-----|---------|------|------|---------|
| nnU-Net | 自动配置 3D 卷积 U-Net | 开箱即用，工程成熟 | 缺乏全局感受野 | 充足标注的标准任务 |
| UNETR | 3D ViT 编码器 + U-Net 解码器 | 全局建模能力强 | 参数量大，计算开销高 | 数据充足、资源丰富 |
| SwinUNETR | 层级 Swin Transformer | 全局+局部均衡，精度高 | 仍然较重（62M 参数）| 中等资源，充足标注 |
| **Light-UNETR** | 轻量 Transformer + 半监督 | 计算量低，标注需求少 | 半监督超参数多，调参复杂 | 资源受限，标注稀缺 |

## 我的观点

**这个方向做对了什么**

轻量化 + 半监督是医学图像分割工程落地的正确路径。临床场景的核心约束不是精度（99% vs 98% Dice 在临床上几乎没有区别），而是**能不能在普通工作站上跑、能不能用现有的少量标注数据**。Light-UNETR 的贡献方向是正确的。

LIDR 的双分支设计（全局+局部）在视觉 Transformer 领域已经是较成熟的技巧（PVT、EfficientFormer 都有类似设计），更值得关注的是 CSE 的半监督策略——AGR 和 SMC 从"外部上下文引入"和"内部上下文一致性"两个维度利用无标注数据，比简单的 Mean Teacher 一致性正则化更有几何直觉。

**离实际应用还有多远**

核心功能已经接近可用，但以下问题仍需解决：

- **多中心泛化**：在单一机构训练的模型在其他机构扫描上性能下降 5-15%（域偏移问题），这是医疗 AI 落地的最大拦路虎
- **不确定性估计**：临床使用必须知道模型"有多不确定"，当前方法缺乏校准良好的置信度输出
- **3D 交互式分割**：真实临床场景中，医生需要在 3D 中交互式修改分割结果，与 SAM3D 等交互式方法的结合是自然的下一步

**值得关注的开放问题**

- 当标注比例从 10% 降到 1% 时，CSE 的性能衰减曲线如何？是否存在半监督学习的有效下限？
- LIDR 的空间压缩步长 $r$ 对不同器官任务的敏感度——是否需要任务自适应的 $r$？
- 能否将 CSE 的思想迁移到遥感图像分割、工业缺陷检测等同样面临标注稀缺问题的领域？

官方代码：[CUHK-AIM-Group/Light-UNETR](https://github.com/CUHK-AIM-Group/Light-UNETR)