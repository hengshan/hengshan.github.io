---
layout: post-wide
title: "永冻土融化预测：从3D点云到2D预测图的垂直结构保留"
date: 2026-03-18 08:04:51 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.16788v1
generated_by: Claude Code CLI
---

## 一句话总结

无人机LiDAR扫描北极森林后，朴素的Z轴平均投影会抹去"地面-林下-树冠"的垂直分层信息；本文提出带可学习高度嵌入的分层投影解码器，让网络自主学习"哪个高度的回波信号对预测地下融深最重要"。

## 为什么这个问题重要？

永冻土覆盖北半球约四分之一的陆地，其融化直接影响全球碳循环：冻土融化释放甲烷和 CO₂，进一步加速气候变暖，形成正反馈。精确监测**活跃层深度**（Active Layer Thickness, ALT）是预测这一反馈环路的关键。

传统方法是拿探针戳进土里测深度，在阿拉斯加内陆茂密的北方森林中极其费力，空间分辨率远不够。无人机LiDAR提供了希望——快速扫描、高点云密度、能捕捉植被垂直结构。但问题来了：

**LiDAR给的是3D点云，融深预测需要的是2D栅格图。如何把3D信息"压缩"到2D，同时不丢失关键的垂直结构？**

### 垂直结构为什么重要？

北方森林中，地下融冻状态与地表植被有强相关性：

```
树冠层 (Canopy)     → 树种和密度影响遮荫 → 影响地表温度
林下层 (Understory) → 苔藓、灌木提供隔热
地面层 (Ground)     → 地被物厚度直接决定热传导
地下               → 活跃层深度（融深，即我们要预测的目标）
```

朴素做法（取所有点特征的平均值）把三层信息混在一起，网络无法区分信号来自树冠还是地面，预测精度自然大打折扣。

## 背景知识

### 3D点云到2D投影的经典方法

| 方法 | 做法 | 问题 |
|------|------|------|
| 高度图 | 只取最高点 | 丢失地面信息 |
| 密度图 | 统计点数 | 无特征信息 |
| Z-mean | 对所有点特征取平均 | 垂直结构完全消失 |
| 多视图投影 | 从不同角度渲染 | 计算量大，对齐困难 |

### Point Transformer V3 简介

PTv3 是处理大规模3D点云的高效 Transformer 架构：用 Hilbert 曲线等空间填充曲线把3D点云序列化为1D序列，在序列上做注意力，避免了 KD-tree 的计算瓶颈，可处理数百万点的超大场景。本文用它作为编码器提取逐点特征。

## 核心方法

### 直觉解释

想象你是侦探，需要根据楼上楼下的线索推断地下室的状态。朴素投影就像把所有楼层的线索混在一张纸上——你看不出哪条线索来自哪层。

本文的方法：**给每条线索贴上"楼层标签"**。

1. 每个 LiDAR 点的特征向量，拼接一个与其高度相关的**可学习嵌入**
2. **分层采样**确保地面、林下、树冠三层都有代表
3. 网络自动学到"地面层信号权重更高"（因为它离地下最近）

### 数学细节

设第 $i$ 个点坐标为 $\mathbf{p}_i = (x_i, y_i, z_i)$，PTv3 输出特征为 $\mathbf{f}_i \in \mathbb{R}^C$。

**高度归一化：**

$$
\hat{z}_i = \frac{z_i - z_{\min}}{z_{\max} - z_{\min}} \in [0, 1]
$$

**高度嵌入（类比 Transformer 位置编码）：**

$$
\mathbf{e}_i = \text{HeightEmbed}\!\left(\left\lfloor \hat{z}_i \cdot K \right\rfloor\right) \in \mathbb{R}^C
$$

其中 $K$ 是高度分箱数（如32），`HeightEmbed` 是可学习的 Embedding 表。

**高度条件特征变换：**

$$
\mathbf{g}_i = \text{MLP}([\mathbf{f}_i \,\|\, \mathbf{e}_i])
$$

**投影到2D格子 $(u, v)$：**

$$
\hat{\mathbf{F}}_{u,v} = \frac{1}{|\mathcal{P}_{u,v}|} \sum_{i \in \mathcal{P}_{u,v}} \mathbf{g}_i, \quad \text{ALT}_{u,v} = \text{PredHead}(\hat{\mathbf{F}}_{u,v})
$$

其中 $\mathcal{P}_{u,v}$ 是分层采样后投影到格子 $(u,v)$ 的点集。

### Pipeline 概览

```
无人机LiDAR点云 (N×3)
    ↓
Point Transformer V3 编码器 → 3D逐点特征 (N×C)
    ↓
Z轴分层采样 (保证地面/林下/树冠各层代表性)
    ↓
高度嵌入 + MLP特征变换 (N×C)
    ↓
XY平面网格聚合 (平均池化)
    ↓
2D特征图 (H×W×C) → 预测头 → 融深图 (H×W)
```

## 实现

### 高度嵌入模块

```python
import torch
import torch.nn as nn

class HeightEmbedding(nn.Module):
    """
    可学习高度嵌入：将连续高度值映射到嵌入向量
    类比 Transformer 中的位置编码，但针对垂直方向
    """
    def __init__(self, feat_dim: int, num_bins: int = 32):
        super().__init__()
        self.num_bins = num_bins
        # 每个高度区间一个可学习向量
        self.embed_table = nn.Embedding(num_bins, feat_dim)
        nn.init.normal_(self.embed_table.weight, std=0.02)

    def forward(self, z: torch.Tensor, z_min: float, z_max: float):
        """
        z: [N] 原始高度值
        返回: [N, feat_dim] 高度嵌入向量
        """
        z_norm = (z - z_min) / (z_max - z_min + 1e-8)  # 归一化到 [0,1]
        bin_idx = (z_norm * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)
        return self.embed_table(bin_idx)
```

### 分层采样器

```python
def stratified_sample(
    points: torch.Tensor,
    features: torch.Tensor,
    strata_ratios: list = [0.2, 0.3, 0.5],  # 地面层/林下层/树冠层 z轴比例
    total_samples: int = 2048
) -> tuple:
    """
    按垂直分层采样，确保地面/林下/树冠各层都有代表。
    strata_ratios 之和应为1，每层均分 total_samples。
    """
    z = points[:, 2]
    z_min, z_max = z.min().item(), z.max().item()
    samples_per = total_samples // len(strata_ratios)
    selected, lower = [], z_min

    for ratio in strata_ratios:
        upper = lower + ratio * (z_max - z_min)
        mask = (z >= lower) & (z < upper)
        idx = mask.nonzero(as_tuple=False).squeeze(1)

        if len(idx) > 0:
            perm = torch.randperm(len(idx), device=points.device)
            selected.append(idx[perm[:min(samples_per, len(idx))]])
        lower = upper

    sel = torch.cat(selected)
    return points[sel], features[sel]
```

### 投影解码器核心

```python
class ProjectionDecoder(nn.Module):
    """将3D点特征投影到2D预测网格，同时保留垂直结构"""

    def __init__(self, feat_dim: int, grid_h: int, grid_w: int):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.height_embed = HeightEmbedding(feat_dim)
        self.transform = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.pred_head = nn.Linear(feat_dim, 1)

    def forward(self, points, features, z_min, z_max, xy_bounds):
        """
        points:    [N, 3]  xyz 坐标
        features:  [N, C]  PTv3 输出特征
        xy_bounds: (x_min, x_max, y_min, y_max)
        返回:      [H, W]  融深预测图 (单位: 米)
        """
        # 1. 高度嵌入 + 特征融合
        h_embed = self.height_embed(points[:, 2], z_min, z_max)
        fused = self.transform(torch.cat([features, h_embed], dim=-1))  # [N, C]

        # 2. xy 坐标 → 网格索引
        x_min, x_max, y_min, y_max = xy_bounds
        u = ((points[:, 0] - x_min) / (x_max - x_min) * (self.grid_w - 1)).long().clamp(0, self.grid_w - 1)
        v = ((points[:, 1] - y_min) / (y_max - y_min) * (self.grid_h - 1)).long().clamp(0, self.grid_h - 1)

        # 3. 按格子聚合（平均池化）
        C = fused.shape[-1]
        grid_feat = torch.zeros(self.grid_h, self.grid_w, C, device=fused.device)
        count = torch.zeros(self.grid_h, self.grid_w, 1, device=fused.device)
        grid_feat.index_put_((v, u), fused, accumulate=True)
        count.index_put_((v, u), torch.ones(len(u), 1, device=fused.device), accumulate=True)
        grid_feat = grid_feat / (count + 1e-8)

        return self.pred_head(grid_feat).squeeze(-1)  # [H, W]
```

### 3D 可视化：对比投影效果

```python
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(points_np, thaw_gt, thaw_naive, thaw_stratified):
    """三列对比：真值 / 朴素投影 / 分层投影"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = thaw_gt.min(), thaw_gt.max()

    for ax, data, title in zip(axes,
                                [thaw_gt, thaw_naive, thaw_stratified],
                                ["Ground Truth ALT (m)", "Naive Z-mean", "Stratified + Height Embed"]):
        im = ax.imshow(data, cmap='plasma', vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.axis('off')
        plt.colorbar(im, ax=ax, label='Thaw Depth (m)')
    plt.tight_layout()
    plt.savefig('projection_comparison.png', dpi=150)

    # 按高度着色的3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])
    z = points_np[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min())
    pcd.colors = o3d.utility.Vector3dVector(plt.cm.viridis(z_norm)[:, :3])
    o3d.io.write_point_cloud('forest_colored_by_height.ply', pcd)
    # 用 open3d 打开: o3d.visualization.draw_geometries([pcd])
```

## 实验

### 数据集说明

论文使用阿拉斯加内陆北方森林的无人机 LiDAR 数据：
- **平台**：DJI M300 + Zenmuse L1
- **点云密度**：约 200 点/m²
- **标注**：现场探针测量的 ALT，对应约 1m 分辨率栅格
- **挑战**：树冠遮挡导致地面回波仅占 5%-15%

这类数据不像 ShapeNet 或 ScanNet 那样开放。若需类似数据，可参考 OpenTopography 或 NEON（美国国家生态观测网络）的公开航空 LiDAR 数据集。

### 定量评估

| 方法 | MAE (m) ↓ | RMSE (m) ↓ | R² ↑ |
|------|-----------|------------|------|
| Z-mean 基线 | 0.18 | 0.24 | 0.61 |
| 高度直方图特征 | 0.15 | 0.20 | 0.71 |
| **分层采样 + 高度嵌入（本文）** | **0.11** | **0.15** | **0.82** |

在植被结构复杂的区域（密林、多层冠层），本文方法提升最为显著，验证了垂直结构信息的价值。

## 工程实践

### 实时性和硬件需求

这是**离线批处理**任务，不要求实时：
- PTv3 编码百万点云约需 4-8 秒（A100 GPU）
- 推理一个 200m×200m 区块约需 10-30 秒
- 显存需求：20-40 GB（取决于点云密度和批大小）
- 大场景建议分块处理，块间保留重叠区做平均融合

### 常见坑

**坑1：高度归一化用了全局极值**

离群点（高大树顶）导致大多数地面点被压缩到 $[0, 0.05]$ 区间，高度嵌入分辨率浪费。

```python
# 用百分位而非极值做归一化
z_max = torch.quantile(z, 0.95).item()
z_min = torch.quantile(z, 0.05).item()
```

**坑2：格子聚合中的空格子**

稀疏区域部分 2D 格子没有任何点，直接输出 0 导致伪影。

```python
from scipy.ndimage import distance_transform_edt
mask_empty = (count.squeeze(-1).cpu().numpy() == 0)
if mask_empty.any():
    _, nn_idx = distance_transform_edt(mask_empty, return_indices=True)
    grid_feat[mask_empty] = grid_feat[nn_idx[0][mask_empty], nn_idx[1][mask_empty]]
```

**坑3：标注对齐误差**

探针测量 GPS 精度约 1-3m，与 LiDAR 投影格子（1m 分辨率）存在对齐误差。训练时在标注点周围 ±2m 范围做软标签平均，可显著降低噪声影响。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 多层冠层的北方森林、热带雨林 | 无植被裸地（朴素投影足够） |
| 地下状态与植被垂直结构相关 | 纯地表形态分析任务 |
| UAV LiDAR，点云密度 >50 点/m² | 卫星雷达（垂直分辨率不足） |
| 离线批处理，允许分钟级推理 | 实时机器人导航 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| PointNet++ 直接回归 | 端到端简单 | 无法处理超大场景 | 小区域稠密点云 |
| 高度切片 2D CNN | 可解释性好 | 层数选择需要先验 | 植被结构已知 |
| **本文（PTv3 + 分层投影）** | 大场景可扩展，自适应垂直结构 | 需要标注数据训练 | 北方森林 ALT 预测 |
| NeRF / 3DGS | 重建质量极高 | 慢，不适合预测任务 | 视觉重建 |

## 我的观点

这篇论文的核心贡献是**把"垂直结构很重要"这个领域先验知识，转化为可学习的归纳偏置**（高度嵌入 + 分层采样），而不是让网络从头自由学习。在标注数据极度稀缺的情况下，这是最务实的工程选择。

**值得关注的开放问题：**

1. **泛化性**：在阿拉斯加训练的模型能迁移到西伯利亚或加拿大北方林吗？不同植被类型可能使高度嵌入的语义完全不同
2. **时序建模**：单时间点 LiDAR 只能预测当前 ALT；融合多年时序数据或许能预测融化速率趋势
3. **多模态融合**：Sentinel-2 光学影像 + LiDAR 点云，在非林地区域（苔原、湿地）可能比单纯依赖垂直结构更有效
4. **无监督预训练**：北方森林 LiDAR 数据量大，但 ALT 标注稀少——自监督点云重建任务预训练 + 少样本微调是自然的下一步

**离实际部署还有多远？** 算法层面已相当成熟。真正的瓶颈是标注：探针测量是苦力活，在北极野外更甚。如果能与自动化土壤传感器阵列（如 TDR 探针网络）结合，批量产生训练标注，这个方向的工业化部署会大大提速。