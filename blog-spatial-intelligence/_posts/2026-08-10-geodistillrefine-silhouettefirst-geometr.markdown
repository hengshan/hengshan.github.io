---
layout: post-wide
title: "无标注航天器分割：GeoDistill-Refine 的几何蒸馏框架"
date: 2026-08-10 12:05:48 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.07405v1
generated_by: Claude Code CLI
---

## 一句话总结

用 SAM 3 作为标注引擎生成伪掩码，通过有符号距离场、骨架和面积约束蒸馏出一个 0.263M 参数、1ms/帧的轻量分割网络——全程不需要人工画一个标注框。

## 为什么这个问题重要？

在轨航天器分割（精确提取卫星轮廓）是以下场景的基础能力：

- **交会对接**：机械臂抓取目标时的视觉引导，需要实时精确轮廓
- **空间态势感知**：在轨物体识别与姿态估计
- **碎片监测**：失效卫星的轮廓与状态评估

困境在于标注成本极高。太空图像的标注需要懂航天的工程师逐帧勾画，SPEED+、TANGO 等公开数据集虽然存在，但标注规模不足以训练重型网络。现有方法：

- **迁移学习**：地面物体 → 太空目标，域差距太大
- **合成数据**：渲染器图像与真实图像存在不可忽视的 sim-to-real gap
- **全监督训练**：精度高，但需要规模化标注，成本无法接受

GeoDistill-Refine 的核心思路：让 SAM 3 这类基础大模型充当"自动标注员"，生成伪掩码（pseudo-mask），再通过几何约束将知识蒸馏到轻量网络中。

## 背景知识

### 知识蒸馏

大模型（Teacher）将知识传递给小模型（Student）的过程。在这个框架中：

- **Teacher**：SAM 3（Segment Anything Model 3），参数量庞大，无法实时运行
- **Student**：TinyUNet（0.263M 参数），推理延迟 1.1ms/张

传统蒸馏用 Teacher 的软标签（logits）监督 Student。这里的特殊性在于 Teacher 的输出（掩码）质量不稳定，需要几何约束来"修正"学生的学习方向。

### 有符号距离场（SDF）

对于一个二值掩码 $M$，其有符号距离场定义为：

$$
\text{SDF}(x) = \begin{cases}
-d(x, \partial M) & x \in M \text{（前景内部，负值）} \\
+d(x, \partial M) & x \notin M \text{（背景，正值）}
\end{cases}
$$

其中 $d(x, \partial M)$ 是点 $x$ 到掩码边界 $\partial M$ 的最短距离。边界处 SDF 值为零，梯度方向垂直于边界，这使得 SDF 损失天然地对边界对齐有强约束力。

### 形态学骨架

骨架是二值形状的"中轴"，通过不断腐蚀得到。对于卫星这类刚体（主体 + 太阳能板阵列），骨架保持了拓扑连接性。骨架损失迫使模型维持目标的整体形状结构，防止出现"断臂"或"碎片化"预测。

## 核心方法

### 整体 Pipeline

```
原始图像
   │
   ▼
SAM 3（6 个固定文本提示，并行推理）
   ├── "spacecraft"        → 掩码1
   ├── "satellite"         → 掩码2
   ├── "space vehicle"     → 掩码3
   ├── "space station component" → 掩码4
   ├── "orbital module"    → 掩码5
   └── "artificial satellite"   → 掩码6
              ↓
       多数投票（50% 阈值）→ 伪掩码
              ↓
       样本质量门控（过滤低质量样本）
              ↓
   ┌─────────────────────────────┐
   │ 阶段一（20 epochs）         │
   │ BCE Loss：学习前景轮廓       │
   └──────────┬──────────────────┘
              ↓
   ┌─────────────────────────────────────────────┐
   │ 阶段二（30 epochs）                          │
   │ BCE + SDF Loss + 骨架 Loss + 面积 Loss      │
   └──────────┬──────────────────────────────────┘
              ↓
         TinyUNet（0.263M 参数，1.1ms/张）
```

### 为什么分两阶段？

直觉上，先学"大概在哪"（前景 vs 背景），再学"边界在哪"（精确轮廓）。如果一开始就加入 SDF 和骨架约束，网络还没学会区分前景背景，几何约束只是噪声。分阶段训练让每个约束在最合适的时机介入——类似于先画草图，再精细描线。

### 样本质量门控

对每张训练样本计算置信度分数：

$$
s = 0.4 \cdot \text{agree} + 0.3 \cdot \text{valid\_ratio} + 0.3 \cdot \text{area\_score}
$$

- `agree`：6 个提示的像素级投票一致性（方差越低越好）
- `valid_ratio`：返回非空掩码的提示比例
- `area_score`：伪掩码面积是否落在合理区间（不能太大也不能太小）

门控分数低的样本，其损失被按比例缩小，防止低质量伪标注污染训练。

### 几何损失函数

阶段二的总损失：

$$
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{SDF}} + \lambda_2 \mathcal{L}_{\text{skel}} + \lambda_3 \mathcal{L}_{\text{area}}
$$

- $\mathcal{L}_{\text{SDF}}$：让模型预测的"边界"与伪掩码的 SDF 零面对齐
- $\mathcal{L}_{\text{skel}}$：骨架区域必须被预测为前景，保持形状拓扑
- $\mathcal{L}_{\text{area}}$：预测面积与伪掩码面积接近，防止预测漂移

## 实现

### 伪掩码生成与质量门控

```python
import numpy as np
from scipy.ndimage import distance_transform_edt

PROMPTS = [
    "spacecraft", "satellite", "space vehicle",
    "space station component", "orbital module", "artificial satellite"
]

def fuse_pseudo_masks(masks: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """多数投票融合 SAM 预测掩码，threshold=0.5 即超过半数提示认为是前景"""
    stack = np.stack(masks, axis=0).astype(np.float32)  # [N, H, W]
    return (stack.mean(axis=0) >= threshold).astype(np.uint8)

def compute_sample_gate(masks: list[np.ndarray], fused: np.ndarray,
                         img_area: int) -> float:
    """计算样本质量门控分数 [0, 1]"""
    stack = np.stack(masks, axis=0).astype(np.float32)
    agree = float(1.0 - stack.std(axis=0).mean())           # 一致性
    valid_ratio = sum(m.sum() > 0 for m in masks) / len(masks)  # 非空比例

    ratio = fused.sum() / img_area
    if ratio > 0.6:    # 超过60%极可能是误检
        area_score = 0.0
    elif 0.01 <= ratio <= 0.4:
        area_score = 1.0
    else:
        area_score = 0.3

    return 0.4 * agree + 0.3 * valid_ratio + 0.3 * area_score
```

### 几何特征计算

```python
from skimage.morphology import skeletonize

def compute_sdf(mask: np.ndarray) -> np.ndarray:
    """从二值掩码计算有符号距离场：内部负值，外部正值，边界为零"""
    mask_bool = mask.astype(bool)
    dist_in  = distance_transform_edt(mask_bool)   # 前景内部到边界距离
    dist_out = distance_transform_edt(~mask_bool)  # 背景到边界距离
    sdf = dist_out - dist_in
    # 归一化，仅关心边界附近 (clamp 防止大物体内部主导梯度)
    sdf = np.clip(sdf, -20, 20) / 20.0
    return sdf.astype(np.float32)

def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """形态学中轴骨架，保留刚体结构的拓扑信息"""
    return skeletonize(mask.astype(bool)).astype(np.float32)
```

### 几何损失函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryRefineLoss(nn.Module):
    def __init__(self, lambda_sdf=1.0, lambda_skel=0.5, lambda_area=0.1):
        super().__init__()
        self.w = (lambda_sdf, lambda_skel, lambda_area)

    def forward(self, logit, mask, sdf, skel, gate):
        prob = torch.sigmoid(logit)

        loss_bce  = F.binary_cross_entropy_with_logits(logit, mask)

        # SDF 对齐：预测 logit 映射到 [-1,1] 后与 SDF 对齐
        pred_sdf_proxy = (prob - 0.5) * 2
        loss_sdf  = F.mse_loss(pred_sdf_proxy, sdf)

        # 骨架损失：骨架像素必须为前景
        eps = 1e-6
        skel_count = skel.sum() + eps
        loss_skel = -(skel * torch.log(prob + eps)).sum() / skel_count

        # 面积正则：防止预测掩码整体漂移
        loss_area = F.l1_loss(prob.mean(), mask.mean())

        total = (loss_bce
                 + self.w[0] * loss_sdf
                 + self.w[1] * loss_skel
                 + self.w[2] * loss_area)
        return total * gate   # 门控加权
```

### TinyUNet 架构

```python
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class TinyUNet(nn.Module):
    """标准 UNet 缩小版，base=32 时约 0.26M 参数"""
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNReLU(in_ch, base), ConvBNReLU(base, base))
        self.enc2 = nn.Sequential(ConvBNReLU(base, base*2), ConvBNReLU(base*2, base*2))
        self.pool = nn.MaxPool2d(2)
        self.btn  = nn.Sequential(ConvBNReLU(base*2, base*4), ConvBNReLU(base*4, base*2))
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # skip concat 后通道数: bottleneck(base*2) + enc2(base*2) = base*4
        self.dec2 = ConvBNReLU(base*4, base*2)
        # skip concat 后通道数: dec2(base*2) + enc1(base) = base*3
        self.dec1 = ConvBNReLU(base*3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.btn(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(b),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.head(d1)
```

### 两阶段训练

```python
def train_geodistill(model, loader, device='cuda',
                     epochs1=20, epochs2=30, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    geo_criterion = GeometryRefineLoss()

    # 阶段一：只学前景/背景轮廓
    model.train()
    for _ in range(epochs1):
        for img, mask, _, _, gate in loader:
            img, mask, gate = img.to(device), mask.to(device), gate.to(device)
            loss = F.binary_cross_entropy_with_logits(model(img), mask) * gate.mean()
            opt.zero_grad(); loss.backward(); opt.step()

    # 阶段二：引入几何精修损失
    for _ in range(epochs2):
        for img, mask, sdf, skel, gate in loader:
            img, mask = img.to(device), mask.to(device)
            sdf, skel = sdf.to(device), skel.to(device)
            loss = geo_criterion(model(img), mask, sdf, skel, gate.mean().to(device))
            opt.zero_grad(); loss.backward(); opt.step()

    return model
```

### 可视化几何约束的作用

```python
import matplotlib.pyplot as plt

def visualize_geometry(mask: np.ndarray):
    """可视化掩码、SDF 和骨架，帮助直观理解三种几何约束"""
    sdf  = compute_sdf(mask)
    skel = compute_skeleton(mask)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(mask, cmap='gray');  axes[0].set_title("Binary Mask")
    im = axes[1].imshow(sdf, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title("Signed Distance Field")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(mask, cmap='gray', alpha=0.5)
    axes[2].imshow(skel, cmap='Reds', alpha=0.9); axes[2].set_title("Skeleton Overlay")
    for ax in axes: ax.axis('off')
    plt.tight_layout(); plt.savefig("geometry_vis.png", dpi=150); plt.show()

# 模拟卫星形状：主体 + 双侧太阳能板
mask = np.zeros((128, 128), dtype=np.uint8)
mask[45:85, 25:105] = 1   # 主体矩形
mask[58:70,  5:25]  = 1   # 左太阳能板
mask[58:70, 103:123]= 1   # 右太阳能板
visualize_geometry(mask)
```

预期输出：左图为二值轮廓；中图为 SDF 热力图（内部蓝色，外部红色，边界白色零等值线清晰可见）；右图叠加骨架（红色中轴线穿过主体中央并延伸至两翼）。SDF 零等值线和骨架共同提供边界对齐与形状拓扑的双重约束。

## 实验

### 数据集说明

| 数据集 | 场景 | 特点 |
|--------|------|------|
| SpaceSense-Bench HJM | 真实在轨 | 主要评估基准，背景复杂 |
| SPEED+ Lightbox | 模拟光箱 | 受控打光，sim-to-real 差距小 |
| SPEED+ Sunlamp | 强方向光 | 高对比度阴影，边界判断难 |
| TANGO | 在轨交会 | 真实太空背景，噪声重 |

### 定量结果

| 方法 | Image IoU | Boundary F1 | 参数量 | 推理延迟 |
|------|-----------|-------------|--------|---------|
| 直接伪标注 Student | 0.712 | 0.543 | 0.263M | 1.1ms |
| **GeoDistill-Refine** | **0.758** | **0.681** | 0.263M | 1.1ms |
| SAM 3 Teacher（参考） | ~0.74 | ~0.65 | >600M | >100ms |

关键发现：**Student 的 Boundary F1 超越了 Teacher**。这说明几何蒸馏不是单纯的模型压缩——SDF 和骨架约束给了学生比老师更强的边界对齐能力。

## 工程实践

### 实时性基准

```python
import time, torch

model = TinyUNet().cuda().eval()
dummy = torch.randn(1, 3, 512, 512).cuda()

with torch.no_grad():
    for _ in range(50): model(dummy)   # GPU 预热
    t0 = time.perf_counter()
    for _ in range(500): model(dummy)
    ms = (time.perf_counter() - t0) / 500 * 1000
    print(f"延迟: {ms:.2f} ms | FPS: {1000/ms:.0f}")
# RTX 4090 预期输出: 延迟: ~1.1 ms | FPS: ~900
```

TinyUNet 在嵌入式 GPU（如 Jetson Orin）上预计 30–60 FPS，满足在轨实时对接引导需求。

### 常见坑

**坑1：SAM 对黑色背景返回全图前景**

太空背景极暗时，SAM 可能把整张图判为前景，`area_plausibility` 得分应为 0 而非正常值：

```python
def area_plausibility(mask, img_area, hard_max=0.6):
    ratio = mask.sum() / img_area
    if ratio > hard_max:
        return 0.0   # 直接丢弃，不参与训练
    return 1.0 if 0.01 <= ratio <= 0.4 else 0.3
```

**坑2：大面积目标的 SDF 梯度消失**

空间站等大型目标内部 SDF 值极大，MSE 损失被内部像素主导，边界附近梯度反而微弱。解决方案是在 `compute_sdf` 中截断到 `[-20, 20]`（已在前面代码实现），只关注边界附近 20 像素范围。

**坑3：小目标骨架退化**

当卫星在图像中仅占几十像素时，骨架退化为单点甚至消失，骨架损失退化为零梯度：

```python
# 在 GeometryRefineLoss.forward 中加入面积检查
if mask.sum() < 150:
    loss_skel = torch.zeros(1, device=logit.device, requires_grad=False)
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 无手动标注预算 | 已有大量精标注数据（用全监督更简单） |
| 刚体目标（卫星、飞船等） | 大量形变或遮挡（柔性结构展开中） |
| 部署端算力受限 | 亚像素级边界精度要求 |
| 目标在图像中相对显著 | 目标像素数 < 100（SAM 提示失效） |
| 单目标场景 | 多目标堆叠（投票融合无法区分实例） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 直接 SAM 3 推理 | 无需训练，泛化强 | 慢（>100ms），不可嵌入 | 离线批处理 |
| 直接伪标注蒸馏 | 部署简单 | 边界粗糙，Boundary F1 低 | 精度要求宽松 |
| **GeoDistill-Refine** | 边界精准，实时，轻量 | 依赖 SAM 伪掩码基础质量 | 在轨实时感知 |
| 全监督 U-Net | 精度最高 | 需要大量标注 | 有充足标注的地面数据集 |

## 我的观点

GeoDistill-Refine 的真正价值不在于刷新精度榜单，而在于提供了一套**可复制的"零标注分割"范式**：基础大模型充当标注引擎 + 几何约束提升边界质量 + 轻量网络部署。这个框架可以迁移到任何标注成本高昂的领域——水下 ROV 目标分割、内窥镜手术器械分割、无人机工业巡检都是自然候选。

需要注意的是，这套方法的天花板取决于 Teacher（SAM）对目标的识别能力。当目标尺度极小或视觉特征过于特殊（如仅有几个像素的远距卫星），SAM 的提示效果会退化，整个伪标注链路都会失效。

值得关注的开放问题：

- 样本质量门控目前是启发式的——能否用自监督方法自动学习门控权重？
- SDF 约束是像素级的，能否用 Neural SDF 引入更强的全局形状先验？
- 多目标场景下，投票融合无法区分实例，需要结合实例感知的提示策略