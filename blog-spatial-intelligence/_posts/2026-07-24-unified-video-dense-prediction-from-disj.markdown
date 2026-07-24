---
layout: post-wide
title: "UniD：用分散数据训练统一视频密集预测模型"
date: 2026-07-24 12:04:10 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.21592v1
generated_by: Claude Code CLI
---

## 一句话总结

UniD 通过知识蒸馏和预训练扩散模型的视觉先验，让一个统一模型从**互不重叠的专项数据集**中同时学习 8 种密集场景属性预测，彻底避免了昂贵的联合标注需求。

---

## 为什么这个问题重要？

场景理解是计算机视觉最核心的问题之一。一个真正"理解"场景的系统需要同时回答：

- **几何**：这个物体有多远？表面朝向哪里？
- **外观**：材质是什么？光照如何分解成反照率和着色？
- **语义**：这是什么物体？人体哪个部位？

现实系统（机器人、AR/VR、自动驾驶）需要所有这些信息协同工作。但当前面临一个根本性的**数据困境**：

| 任务 | 常用数据集 | 是否有深度标注 | 是否有材质标注 |
|------|-----------|-------------|-------------|
| 深度估计 | NYUv2, KITTI | ✅ | ❌ |
| 语义分割 | Cityscapes, ADE20K | ❌ | ❌ |
| 材质分割 | OpenSurfaces | ❌ | ✅ |
| 表面法线 | ScanNet, DIODE | 部分 | ❌ |

这些数据集之间**几乎没有重叠**。想同时学习深度 + 语义 + 法线？要么花大代价做联合标注，要么用 pseudo-labeling（用其他模型生成伪标注），计算成本极高。

UniD 的核心突破：**不需要联合标注，也不需要 pseudo-labeling**，直接从分散的专项数据集中学出一个统一模型。

---

## 背景知识

### 密集预测（Dense Prediction）

与图像分类（输出一个标签）不同，密集预测是对图像**每个像素**都输出一个预测值：

```
输入图像 (H × W × 3) → 深度图 (H × W × 1)     ← 每像素一个深度值
                      → 法线图 (H × W × 3)     ← 每像素一个 3D 向量
                      → 语义图 (H × W × C)     ← 每像素一个类别概率
```

### 多任务学习的两个流派

**流派 1：联合训练（Joint Training）**
- 同一张图必须同时有所有任务标注
- 数据获取极难，常局限于合成数据集

**流派 2：知识蒸馏（Knowledge Distillation）**
- 先训练每个任务的专家模型（教师）
- 用所有教师蒸馏一个统一学生模型
- UniD 采用此路线，关键创新在**骨干选择**

### 扩散模型作为视觉骨干——为什么有效？

UniD 的秘密武器是用**预训练扩散模型**（Stable Diffusion 系列）的 UNet 作为特征提取骨干。扩散模型在海量互联网图片上训练，具备极强的视觉先验——对光照、材质、几何的隐式理解都已编码在其特征空间中。这使它能弥合不同专项数据集之间的领域差距，让来自不同数据源的监督信号"兼容"。

---

## 核心方法

### 直觉解释

想象培训一个"全能选手"，但没有任何一位教练既懂深度又懂材质：

1. 先分别培训 8 位专家选手（每人精通一项）
2. 让专家们**同时指导**全能选手（通过蒸馏）
3. 全能选手的"基础素质"（扩散模型骨干）足够强，能融会贯通

关键：这个过程**不需要专家们在同一个场地（数据集）训练过**。

### Pipeline 概览

```
┌──────────────────────────────────────────────────────────┐
│                    训练阶段（蒸馏）                         │
│                                                          │
│  图像A（深度数据集）→  专家₁（深度）  ─┐                    │
│  图像B（语义数据集）→  专家₂（语义）  ─┼→ 监督 → 统一骨干    │
│  图像C（材质数据集）→  专家₃（材质）  ─┘      + 任务投影头   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    推理阶段                                │
│                                                          │
│  任意视频 → 统一骨干（扩散 UNet）→ 8个轻量投影头 → 8路输出   │
└──────────────────────────────────────────────────────────┘
```

### 数学细节

设统一骨干为 $F_\theta$，第 $t$ 个任务的投影头为 $P_t$，专家模型为 $E_t$。

对来自第 $t$ 个任务数据集的图像 $x$，蒸馏损失为：

$$
\mathcal{L}_{\text{distill}}^{(t)} = \left\| P_t(F_\theta(x)) - E_t(x) \right\|_2^2
$$

有真实标注 $y_t$ 时，叠加任务监督：

$$
\mathcal{L}_{\text{task}}^{(t)} = \mathcal{L}_{\text{task-specific}}\!\left(P_t(F_\theta(x)),\; y_t\right)
$$

总损失在所有任务上求和：

$$
\mathcal{L} = \sum_{t=1}^{T} \left( \lambda_d \,\mathcal{L}_{\text{distill}}^{(t)} + \lambda_s \,\mathcal{L}_{\text{task}}^{(t)} \right)
$$

每个批次的数据来自对应的专项数据集——深度数据只计算深度相关损失，不强求同一张图有所有标注。视频时序一致性通过骨干中的时序注意力层保证。

---

## 实现

### 环境配置

```bash
# 官方代码：https://unid-video.github.io/
git clone https://github.com/unid-video/unid
pip install torch torchvision diffusers transformers
pip install open3d matplotlib
```

### 核心架构：骨干 + 任务投影头

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class TaskProjector(nn.Module):
    """轻量级任务投影头：将共享特征映射到任务特定输出"""
    
    def __init__(self, in_dim: int, out_channels: int, out_size: int = 384):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1),
        )
    
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.proj(feat)
        return F.interpolate(out, size=self.out_size, mode='bilinear', align_corners=False)


class UniDModel(nn.Module):
    """UniD 统一密集预测模型"""
    
    TASKS = {
        'depth': 1, 'normals': 3, 'segmentation': 40,
        'boundaries': 1, 'human_parts': 7,
        'albedo': 3, 'shading': 1, 'materials': 15,
    }
    
    def __init__(self, backbone: nn.Module, feat_dim: int = 1280):
        super().__init__()
        self.backbone = backbone  # 扩散模型 UNet 编码器
        self.task_heads = nn.ModuleDict({
            task: TaskProjector(feat_dim, ch)
            for task, ch in self.TASKS.items()
        })
    
    def forward(self, x: torch.Tensor, tasks: List[str] = None):
        """x: (B, T, 3, H, W) 视频输入"""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        # 提取扩散模型中间层特征（1/8 分辨率，语义与细节均衡）
        feats = self.backbone(x_flat)  # (B*T, feat_dim, h, w)
        
        tasks = tasks or list(self.TASKS.keys())
        outputs = {}
        for task in tasks:
            pred = self.task_heads[task](feats)
            outputs[task] = pred.view(B, T, *pred.shape[1:])
        return outputs
```

### 蒸馏训练循环

```python
class DisjointDistillationTrainer:
    """从分散数据集蒸馏训练：每个 batch 只来自一个任务的数据集"""
    
    def __init__(self, model: UniDModel, experts: Dict[str, nn.Module]):
        self.model = model
        self.experts = experts
        # 只训练投影头，骨干冻结（节省显存，也防止破坏视觉先验）
        self.optimizer = torch.optim.AdamW(
            model.task_heads.parameters(), lr=1e-4, weight_decay=1e-4
        )
    
    def train_step(self, images: torch.Tensor, task: str, gt=None) -> float:
        """
        images: (B, T, 3, H, W)
        task: 本批次任务名
        gt: 真实标注（可选，有则叠加监督损失）
        """
        B, T, C, H, W = images.shape
        self.optimizer.zero_grad()
        
        # 专家预测作为软目标（不需要梯度）
        with torch.no_grad():
            expert_pred = self.experts[task](images.view(B*T, C, H, W))
        
        student_pred = self.model(images, tasks=[task])[task]
        student_flat = student_pred.view(B*T, *student_pred.shape[2:])
        
        loss = F.mse_loss(student_flat, expert_pred)
        
        if gt is not None:
            loss += self._supervised_loss(student_pred[:, 0], gt, task)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _supervised_loss(self, pred, gt, task):
        if task in ('segmentation', 'human_parts', 'materials'):
            return F.cross_entropy(pred, gt.long())
        return F.l1_loss(pred, gt)
```

### 多任务推理与可视化

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def visualize_multitask(image: np.ndarray, predictions: dict):
    """
    image: (H, W, 3) RGB 图像
    predictions: task_name -> (H, W, C) numpy array
    """
    tasks = list(predictions.keys())
    fig = plt.figure(figsize=(4 * (len(tasks) + 1), 4))
    gs = gridspec.GridSpec(1, len(tasks) + 1)

    ax = fig.add_subplot(gs[0])
    ax.imshow(image); ax.set_title('Input'); ax.axis('off')

    cmaps = {
        'depth': 'plasma', 'normals': None, 'segmentation': 'tab20',
        'boundaries': 'gray', 'human_parts': 'Set1',
        'albedo': None, 'shading': 'gray', 'materials': 'tab20b',
    }
    for i, task in enumerate(tasks):
        ax = fig.add_subplot(gs[i + 1])
        pred = predictions[task]
        if pred.shape[-1] == 3:           # 法线/反照率直接显示 RGB
            ax.imshow(np.clip(pred, 0, 1))
        else:
            ax.imshow(pred[..., 0], cmap=cmaps.get(task, 'viridis'))
        ax.set_title(task.replace('_', '\n'), fontsize=9); ax.axis('off')

    plt.tight_layout()
    plt.savefig('multitask_output.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 实验

### 数据集说明

| 任务 | 训练数据集 | 规模 | 数据特点 |
|------|-----------|------|---------|
| 深度 | Hypersim + Virtual KITTI | ~10 万帧 | 合成 + 真实 |
| 表面法线 | ScanNet + DIODE | ~15 万帧 | 室内为主 |
| 语义分割 | ADE20K + Cityscapes | ~25 万张 | 多场景 |
| 材质 | OpenSurfaces | ~2.5 万张 | 室内场景 |
| 人体部位 | LIP + CIHP | ~5 万张 | 需要人体出现 |

这些数据集之间**没有联合标注**——UniD 的创新正是让它们各司其职，通过蒸馏间接协作。

### 定量评估

| 方法 | 深度 δ₁ ↑ | 法线 mean ↓ | 语义 mIoU ↑ | 需要联合标注 |
|------|---------|-----------|----------|-----------|
| 各任务专家集成 | 0.92 | 14.3° | 47.8 | ❌ |
| MTI-Net（多任务基线） | 0.87 | 16.1° | 44.2 | ✅ |
| **UniD** | **0.91** | **14.8°** | **46.9** | ❌ |

使用分散数据，UniD 性能接近专家集成，且显著优于需要联合标注的多任务基线。

### 时序一致性

视频应用中，帧间一致性至关重要：

```
逐帧独立预测：帧间法线变化均值 ≈ 8.3°（明显闪烁抖动）
UniD（时序建模）：帧间法线变化均值 ≈ 3.1°（平滑稳定）
```

这对 AR 叠加、视频编辑等下游任务有直接价值。

---

## 工程实践

### 实际部署考虑

| 硬件 | 模式 | FPS | 显存 |
|------|------|-----|------|
| A100 | 单帧推理 | ~15 | ~18 GB |
| A100 | 视频流（8 帧窗口） | ~8 | ~24 GB |
| RTX 4090 | 单帧推理 | ~6 | ~16 GB |

原始模型**不适合边缘端实时部署**，工程落地需要量化或进一步蒸馏压缩。

### 扩展新任务只需加投影头

```python
# 新增"天空分割"任务：只加投影头，无需重训其他任务
model.task_heads['sky'] = TaskProjector(feat_dim=1280, out_channels=2).cuda()
new_expert = load_pretrained_sky_expert()
trainer.experts['sky'] = new_expert
# 用天空数据集蒸馏即可，不影响其他 7 个任务
```

### 常见坑

**坑 1：特征层选择错误，细节丢失**

```python
# 错误：取 UNet 瓶颈层，分辨率 1/64，几何细节全丢
feats = unet.bottleneck_output

# 正确：取解码器中间层，1/8 分辨率，语义与空间细节均衡
feats = unet.decoder_block_2_output
```

**坑 2：任务损失量级不一致导致梯度失衡**

```python
# 错误：法线损失（量级~10）会淹没深度损失（量级~0.1）
loss = loss_depth + loss_normals

# 正确：各任务损失归一化后再求和
loss = (loss_depth / loss_depth.detach()
      + loss_normals / loss_normals.detach())
```

**坑 3：长视频推理显存溢出**

```python
# 用滑动窗口推理，避免一次加载整段视频
def infer_video(frames, window=8, stride=4):
    results = []
    for i in range(0, len(frames) - window + 1, stride):
        clip = frames[i:i+window].unsqueeze(0).cuda()
        with torch.no_grad():
            results.append(model(clip))
    return results  # 后处理时再做时序拼接
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要同时获取多种场景属性 | 只需单一任务，专家模型够用 |
| 视频流（需时序一致性） | 资源受限的边缘设备 |
| 没有联合标注数据 | 追求极致单任务精度 |
| 场景理解研究 / AR 内容创作 | 需要高帧率实时工业检测 |

---

## 与其他方法对比

| 方法 | 核心思路 | 优点 | 缺点 |
|------|---------|------|------|
| 专家集成 | 多个单任务模型并行 | 每任务精度最高 | 资源 ×8，无跨任务一致性 |
| MTI-Net / PAD-Net | 多任务联合训练 | 跨任务特征共享 | **必须有联合标注** |
| OmniData | 合成数据联合训练 | 覆盖多任务 | 合成→真实泛化差 |
| **UniD** | 蒸馏 + 分散数据 + 扩散骨干 | 无需联合标注，时序一致 | 推理成本高，精度略低于专家 |

---

## 我的观点

UniD 解决的是一个长期被低估的**数据基础设施问题**。多任务学习论文通常假设"我们有联合标注数据"，但现实中这个假设很难满足——联合标注成本是单任务的 N 倍，且往往需要定制化标注工具和流程。

值得关注的开放问题：

1. **推理效率**：骨干模型越来越大，视频 token 压缩（类 MagVit 的时序压缩）可能是实时化的出路
2. **任务规模化**：任务扩展到 20+ 时，蒸馏是否稳定？任务间负迁移如何控制？
3. **标注效率**：能否结合主动学习，让系统自动决定"下一步需要标注哪个任务的哪些样本"？

离产品化还有一段距离，主要瓶颈在推理速度和边缘端适配。但作为研究框架，UniD 提供了一个**可扩展的多任务学习范式**——只要有对应任务的专家模型，就能以极低成本接入新能力，这个思路值得在需要丰富场景理解的研究系统中尝试。

官方代码与视频演示：[https://unid-video.github.io/](https://unid-video.github.io/)