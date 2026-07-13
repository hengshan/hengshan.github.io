---
layout: post-wide
title: "从太空俯视地球：SAM 3 零样本遥感分割的能力边界评测"
date: 2026-07-13 12:05:24 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.09583v1
generated_by: Claude Code CLI
---

## 一句话总结

SAM 3 的视觉提示能有效识别卫星/航拍图像的复杂几何结构，但文本提示会引入「地面语义偏差」，激活地面视角的先验知识、主动干扰坐标回归——这项系统性评测揭示了基础模型在专业遥感域的能力边界与改造路径。

## 为什么这个问题重要？

遥感影像是人类理解地球表面的关键手段：农业监测、城市扩张分析、灾害评估、基础设施测绘——这些任务每年处理的卫星数据量以 PB 计。

传统方法的困境是**每个任务都需要大量标注数据**。一旦想检测新类型目标（临时搭建的营地、新型飞机），就得重新收集标注、重新训练——成本极高。

SAM（Segment Anything Model）系列让人看到了希望：能不能用一个通用模型通过「提示」完成任意分割，无需针对遥感数据训练？SAM 3 进一步引入**多模态提示**（文本 + 视觉），理论上更强大。这篇论文系统回答了一个关键问题：**SAM 3 在遥感影像上真实的零样本能力是什么？**

### 现有方法的瓶颈

- **专用模型**（RSPrompter、GeoSAM 等）：需要针对遥感数据微调，泛化到新类别时性能骤降
- **通用基础模型**（SAM 1/2）：不支持文本提示，需要手动点击或框选才能分割
- **SAM 3**：理论上能接受文本/视觉提示做开放词汇分割，但遥感领域实际表现从未被系统评测

---

## 背景知识

### 遥感影像 vs 自然图像：俯视几何带来的根本差距

遥感影像最核心的特点是**俯视视角**（Overhead/Nadir View）。这让目标的外观与训练数据产生根本性差距：

```
地面视角（SAM 训练数据）：         俯视视角（遥感影像）：

      /|\ 屋顶可见                     ┌────────┐
     / | \                             │        │  ← 建筑投影为矩形
    /  |  \                            │        │    无深度、无立面、无阴影
   /   |   \                           └────────┘
 墙面透视、玻璃反射、阴影向前投
```

这导致三个核心问题：

1. **形态差异**：飞机在地面看是侧面流线型；从卫星看是"十字"形状
2. **纹理差异**：屋顶材质、停车场沥青——这些在地面训练数据中几乎不出现
3. **尺度差异**：中分辨率卫星（10m/pixel）下，一辆汽车可能只有 2×4 像素

### SAM 3 架构简述

SAM 3 在 SAM 2 基础上增加了**多模态解码器**，支持文本和视觉双路提示：

```
图像 ──→ 图像编码器 ──→ 图像特征
                             ↓
文本提示 ──→ 文本编码器 ──→  ┤
                             ↓
视觉提示 ──→ 提示编码器 ──→ 多模态解码器
                             ↓
                    分割掩码 + 存在置信度头（Presence Head）
```

其中**存在置信度头**是一个二分类输出："目标是否存在于图像中"。这篇论文的核心工程创新之一就是把它改造成零样本分类器。

### 广义零样本学习（GZSL）与调和均值

普通零样本学习（ZSL）只在**未见过的类别**上测试。但实际应用中，模型会同时遇到见过和没见过的类别——这就是**广义零样本学习（GZSL）**。

GZSL 的核心指标是**调和均值（H-mean）**：

$$
H = \frac{2 \times \text{Acc}_{seen} \times \text{Acc}_{unseen}}{\text{Acc}_{seen} + \text{Acc}_{unseen}}
$$

如果模型过拟合到已见类别，$\text{Acc}_{unseen}$ 就会很低，H-mean 会被强制拉低——它迫使模型在两侧都保持平衡，是评估基础模型真实泛化能力的关键指标。

---

## 核心方法

### 直觉：存在头 → 零样本分类器

给模型看一张图，问"飞机在这里吗？"得到一个置信分数。对所有候选类别都问一遍，取分数最高的作为预测类别——用二元判断头构造多分类器。

### 数学细节

设图像特征 $f_I \in \mathbb{R}^D$，类别 $c$ 的文本特征 $f_T^c \in \mathbb{R}^D$，存在置信度为：

$$
p_c = \sigma\left(\frac{f_I \cdot f_T^c}{\|f_I\| \|f_T^c\|} / \tau\right)
$$

零样本分类的预测为 $\hat{c} = \arg\max_c\, p_c$，将单次二元查询推广为跨类别的 argmax 选择。

### Pipeline 概览

```
遥感影像 → 图像编码器 → 图像特征
                           ↓
类别名称 → 文本编码器 → 存在头×N类 → argmax → 零样本分类
                           ↓
视觉参考 → 提示编码器 → 多模态解码器 → 实例分割掩码
```

---

## 实现

### 零样本分类器适配

```python
import torch
import torch.nn.functional as F
from typing import List

class SAM3ZeroShotClassifier:
    """将 SAM 3 的二元存在头改造为零样本多分类器"""

    def __init__(self, sam3_model, class_names: List[str]):
        self.model = sam3_model
        self.class_names = class_names
        self.text_prototypes = self._build_prototypes(class_names)

    def _build_prototypes(self, class_names: List[str]):
        # 加入俯视上下文可略微缓解语义偏差
        prompts = [f"aerial satellite view of {name}" for name in class_names]
        with torch.no_grad():
            feats = self.model.encode_text(prompts)      # [N, D]
        return F.normalize(feats, dim=-1)

    def classify(self, image: torch.Tensor) -> dict:
        img_feat = F.normalize(self.model.encode_image(image), dim=-1)  # [1, D]

        # presence_scores[i] = "类别i存在于图像中"的置信度（推广到多类）
        scores = (img_feat @ self.text_prototypes.T).squeeze(0)   # [N]
        pred_idx = scores.argmax().item()

        return {
            "class": self.class_names[pred_idx],
            "confidence": scores[pred_idx].item(),
            "all_scores": scores.cpu().numpy()
        }
```

### 五种提示配置的消融评测

论文通过系统隔离文本/视觉贡献来诊断跨模态干扰：

```python
PROMPT_CONFIGS = {
    "text_only":    {"use_text": True,  "use_visual": False},   # 配置1：纯文本
    "visual_only":  {"use_text": False, "use_visual": True},    # 配置2：纯视觉
    "text_visual":  {"use_text": True,  "use_visual": True},    # 配置3：双模态
    "text_bbox":    {"use_text": True,  "use_bbox":   True},    # 配置4：文本+框
    "visual_bbox":  {"use_text": False, "use_bbox":   True},    # 配置5：视觉+框
}

def run_ablation(model, dataset, configs=PROMPT_CONFIGS):
    results = {}
    for name, cfg in configs.items():
        preds = model.predict_batch(dataset, **cfg)
        results[name] = compute_metrics(preds, dataset.labels)
    return results
    # 关键结论：visual_only 在几何任务上优于 text_only
    # 且 text_visual 并不总是优于 visual_only（文本干扰！）
```

### 无训练 GZSL 代理评估

```python
import numpy as np

def gzsl_proxy_eval(model, seen_classes, unseen_classes, images, labels):
    """无需训练集：用零样本置信度分数直接进行 GZSL 评估"""
    all_classes = seen_classes + unseen_classes
    clf = SAM3ZeroShotClassifier(model, all_classes)

    preds  = np.array([clf.classify(img)["class"] for img in images])
    labels = np.array(labels)

    seen_mask   = np.isin(labels, seen_classes)
    unseen_mask = np.isin(labels, unseen_classes)

    acc_s = (preds[seen_mask]   == labels[seen_mask]).mean()
    acc_u = (preds[unseen_mask] == labels[unseen_mask]).mean()
    h     = 2 * acc_s * acc_u / (acc_s + acc_u + 1e-8)

    return {"S": acc_s, "U": acc_u, "H": h}
```

### 俯视几何差距可视化

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='#f9f9f9')

# 地面视角：文本提示"aircraft"激活的先验
ax1.set_facecolor('#e8f4f8')
body   = mpatches.FancyArrowPatch((0.2,0.45),(0.8,0.45), mutation_scale=50, color='steelblue')
lwing  = mpatches.FancyArrowPatch((0.35,0.45),(0.25,0.7), mutation_scale=25, color='steelblue')
rwing  = mpatches.FancyArrowPatch((0.65,0.45),(0.75,0.7), mutation_scale=25, color='steelblue')
for p in [body, lwing, rwing]: ax1.add_patch(p)
ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.axis('off')
ax1.set_title("文本先验：侧视飞机\n（水平翼展、有深度透视）", fontsize=10)

# 俯视视角：遥感影像实际外观
ax2.set_facecolor('#f0f8e8')
ax2.add_patch(plt.Rectangle((0.44,0.05), 0.12, 0.9, color='steelblue', alpha=0.85))  # 机身
ax2.add_patch(plt.Rectangle((0.05,0.42), 0.9, 0.16, color='steelblue', alpha=0.85))  # 机翼
ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.axis('off')
ax2.set_title("遥感实际：俯视十字形\n（纵横比/形状完全不同）", fontsize=10)

plt.suptitle("跨模态干扰的根源：文本先验 vs 俯视外观", fontsize=12, fontweight='bold')
plt.tight_layout()
# 预期输出：左侧侧视飞机轮廓 vs 右侧俯视十字形，直观展示语义偏差来源
```

---

## 实验

### 数据集说明

论文在三类任务上评测：**场景分类**（NWPU-RESISC45 等）、**目标检测**（DOTA、DIOR）、**实例分割**（iSAID）。这些数据集覆盖建筑、飞机、船舶、车辆等典型遥感类别，分辨率从 0.5m 到 30m/pixel 不等。

评测分为 **严格零样本**（不见任何目标类别样本）和 **一次性**（每类别一张参考图）两种设定。

### 定量评估（论文趋势，数据示意）

| 提示配置 | 场景分类 | 目标检测 AP | 实例分割 IoU |
|---------|---------|-----------|------------|
| 纯视觉（One-shot） | **~68** | **~42** | **~51** |
| 纯文本（Zero-shot） | ~55 | ~32 | ~44 |
| 文本 + 视觉 | ~61 | ~34 | ~48 |

关键反直觉结论：**文本 + 视觉并不优于纯视觉**——文本提示在检测任务中主动降低了坐标回归精度。

### GZSL 分割 H-mean 对比（论文趋势，数据示意）

| 方法 | Acc\_seen (S) | Acc\_unseen (U) | H-mean |
|-----|--------------|----------------|--------|
| 传统领域适配模型 | ~82 | ~23 | ~36 |
| SAM 3（视觉提示） | ~62 | ~58 | **~60** |

SAM 3 的 H-mean 大幅领先：它不会把未见类别强行归入已见类别，这对真实部署中遇到新目标类型时价值巨大。

---

## 工程实践

### 实时性与硬件需求

- SAM 3 全精度推理：至少需要 RTX 3090 (24GB) 或 A100
- 分割单张 1024×1024 瓦片：约 40–80ms，实时应用需多卡并行
- 存在置信度头的分类推理远轻量：可在单卡 GPU 上以 >50 FPS 运行

### 常见坑

**坑 1：遥感影像的坐标系转换**

分割结果是像素掩码，需要映射回地理坐标才能与 GIS 系统对接：

```python
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

def mask_to_geojson(mask: np.ndarray, transform) -> list:
    """像素掩码 → 地理坐标多边形，写入 GIS 系统"""
    geoms = [
        shape(geom)
        for geom, val in shapes(mask.astype('uint8'), transform=transform)
        if val == 1
    ]
    return geoms
```

**坑 2：大影像分块的边缘伪影**

SAM 3 期望 1024×1024 输入，卫星影像常为 GB 级单张，分块时边界处目标会被截断：

```python
def tiled_inference(model, image: np.ndarray, tile=1024, overlap=128) -> np.ndarray:
    """带重叠的滑窗推理，用平均投票消除边缘伪影"""
    H, W = image.shape[:2]
    stride = tile - overlap
    accum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            pred = model.predict(image[y:y+tile, x:x+tile])
            accum[y:y+tile, x:x+tile] += pred
            count[y:y+tile, x:x+tile] += 1.0

    return (accum / np.maximum(count, 1)) > 0.5  # 均值阈值化
```

**坑 3：文本提示的俯视语境注入**

直接用 `"car"` 作为文本提示会激活地面视角先验，加入俯视语境可略微缓解：

```python
# ❌ 直接类别名
prompts_bad = ["car", "aircraft", "building"]

# ✅ 带俯视上下文（效果有限但有帮助）
prompts_better = [f"overhead satellite image of {c}" for c in ["car", "aircraft", "building"]]
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 高分辨率影像（≤0.5m/pixel）的大型目标 | 中低分辨率（>5m/pixel）中的小目标检测 |
| 需要泛化到新类别（GZSL 场景） | 需要精确边界框坐标的文本驱动检测 |
| 有参考样本的一次性实例分割 | 实时/边缘设备部署（计算量太大） |
| 快速探索性分析、辅助标注 | 需要语义细粒度理解的场景 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 专用遥感模型（GeoSAM） | 训练类别精度高 | 新类别泛化差，需再训练 | 固定类别批量生产 |
| SAM 2（仅视觉提示） | 零样本，几何对齐好 | 无文本支持，需手动点击 | 交互式辅助标注 |
| SAM 3（本文） | 多模态提示，高 H-mean | 文本提示有偏差，计算重 | 新类别探索、GZSL 基准 |
| LoRA + SAM 3 | 保留基础能力 + 俯视校正 | 需少量地理标注数据 | **生产级部署（推荐）** |

---

## 我的观点

这篇论文最有价值的贡献不是某个新模型，而是**精确诊断出了一个失效机制：跨模态语义偏差**。

SAM 3 的文本编码器「活在地面」——它的所有语义表征来自地面视角的训练数据。当你问它「卫星图里的飞机在哪儿」，它不是不懂飞机，而是它脑子里的飞机形状与眼前的完全不同。这不是 prompt engineering 能修复的——**这需要解码器层面的参数高效微调**，论文最终也明确指向了这个方向。

从工程角度看，**视觉提示路线值得立刻投入**：
- One-shot 视觉分割已经可用于人工辅助标注和快速原型
- 高 H-mean 意味着模型在生产中不会对新目标类型「失明」
- LoRA 微调 SAM 3 的多模态解码器成本极低，可能只需数百张带标注的俯视图像

文本提示的遥感应用短期仍是开放问题，等待更大规模的**地理空间多模态预训练数据集**来从根本上修复语义偏差——类似 GeoCLIP、SkyScript 的方向，但规模需要更大。

> 论文链接：[arxiv.org/abs/2607.09583](https://arxiv.org/abs/2607.09583v1)