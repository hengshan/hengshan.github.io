---
layout: post-wide
title: "ZeD-MAP：用 Bundle Adjustment 引导零样本深度扩散模型实现实时无人机三维重建"
date: 2026-04-07 12:04:39 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.04667v1
generated_by: Claude Code CLI
---

## 一句话总结

ZeD-MAP 将零样本深度扩散模型（无需重新训练）与增量式集束调整（BA）结合，让无人机影像从"相对深度猜测"变成"绝对精度的稠密度量地图"，每帧处理时间控制在 1.5~5 秒。

## 为什么这个问题重要？

无人机（UAV）影像的实时深度重建是**灾害响应**、**精准农业**、**基础设施巡检**的核心需求——你需要在飞行结束之前就知道地面的三维结构。现有方案各有致命缺陷：

| 方案 | 核心问题 |
|-----|---------|
| 经典多视角立体（MVS/COLMAP） | 大基线、低纹理区域效果差，速度慢 |
| Transformer 单目深度 | 需要大量标注数据，泛化性弱 |
| 扩散模型深度（Marigold 等） | 输出**相对深度**，无绝对尺度，帧间不一致 |

ZeD-MAP 的核心洞察很简单：**扩散模型告诉你"形状"，BA 告诉你"尺度"，两者结合才有用。**

## 背景知识

### 零样本深度扩散模型的根本局限

Marigold、Depth Anything V2 这类模型能从单张图像预测稠密深度图，不需要针对特定场景训练。但它们的输出是**仿射不变深度**，即：

$$d_{\text{pred}} = s \cdot d_{\text{true}} + t$$

其中 $s > 0$ 是未知尺度，$t$ 是未知偏移。更糟的是，不同帧的 $s$ 和 $t$ 会随机漂移，相邻帧的深度图无法直接拼接成一致的三维地图。

### Bundle Adjustment 能给我们什么

集束调整（Bundle Adjustment, BA）通过最小化重投影误差，同时优化相机位姿和稀疏三维点：

$$\min_{\{R_i, \mathbf{t}_i\}, \{X_j\}} \sum_{i,j} \rho\left(\left\| \pi(R_i X_j + \mathbf{t}_i) - x_{ij} \right\|^2\right)$$

- $\pi$：透视投影函数
- $x_{ij}$：第 $i$ 帧中第 $j$ 个特征点的像素坐标
- $\rho$：鲁棒损失函数（如 Huber）

BA 输出**度量精确**的稀疏点云和相机位姿，但点很稀疏，没有稠密深度信息。

### ZeD-MAP 的桥梁：仿射对齐

有了 BA 的稀疏度量点，在每帧中找到这些点的像素位置，读取扩散模型预测的相对深度，然后拟合线性变换：

$$\min_{s,\, t} \sum_{p \in \mathcal{P}} \left(s \cdot d_{\text{rel}}(p) + t - d_{\text{metric}}(p)\right)^2$$

最小二乘解出 $s$ 和 $t$，应用到全图，将相对深度对齐到度量空间。锚点越密集、分布越均匀，对齐越稳定。

## 核心方法

### Pipeline 概览

```
UAV 连续帧流
    ↓
┌─────────────────────────────┐
│ 集群划分（overlapping）      │
│ 每个集群约 30-50 帧          │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│ 增量式 Bundle Adjustment     │
│ → 稀疏三维点  → 度量位姿     │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│ 稀疏点重投影 → 深度锚点      │
│ 扩散模型 → 相对深度图        │
│ 仿射对齐 → 度量稠密深度图    │
└─────────────────────────────┘
```

## 实现

### 核心：稀疏引导的深度对齐

```python
import numpy as np

def align_depth_affine(
    depth_relative: np.ndarray,      # H x W，扩散模型输出的相对深度
    sparse_depth_metric: np.ndarray, # H x W，稀疏度量深度（0 表示无效）
    min_points: int = 20,
) -> np.ndarray:
    """
    用 BA 产生的稀疏度量锚点，将相对深度对齐到绝对尺度。
    核心：在有稀疏深度的像素处拟合 s * d_rel + t = d_metric
    """
    valid_mask = sparse_depth_metric > 0
    if valid_mask.sum() < min_points:
        raise ValueError(f"锚点不足 ({valid_mask.sum()} < {min_points})")

    d_rel    = depth_relative[valid_mask].reshape(-1, 1)
    d_metric = sparse_depth_metric[valid_mask]

    # 构造线性系统 [d_rel, 1] @ [s, t]^T = d_metric
    A = np.hstack([d_rel, np.ones_like(d_rel)])
    (scale, shift), *_ = np.linalg.lstsq(A, d_metric, rcond=None)

    if scale <= 0.01:
        raise ValueError(f"尺度 s={scale:.4f} 异常，检查锚点质量")

    depth_out = scale * depth_relative + shift
    return np.clip(depth_out, 0.1, 500.0)  # 合理深度范围（单位：m）
```

### 集群划分策略

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class FrameCluster:
    frame_ids: List[int] = field(default_factory=list)
    overlap_with_prev: int = 0

def create_overlapping_clusters(
    total_frames: int,
    cluster_size: int = 40,
    overlap: int = 10,
) -> List[FrameCluster]:
    """
    将连续帧序列划分为重叠集群。
    重叠帧确保相邻集群之间的 BA 结果可以拼接对齐。
    """
    clusters = []
    stride = cluster_size - overlap  # 每次滑动步长

    for start in range(0, total_frames, stride):
        end = min(start + cluster_size, total_frames)
        frame_ids = list(range(start, end))

        if len(frame_ids) < 5:      # 集群太小则合并到前一个
            if clusters:
                clusters[-1].frame_ids.extend(frame_ids)
            break

        clusters.append(FrameCluster(
            frame_ids=frame_ids,
            overlap_with_prev=overlap if clusters else 0
        ))

    return clusters
```

### Bundle Adjustment（简化版，理解核心）

```python
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def project_point(cam_params, point_3d, K):
    """将三维点投影到像素坐标"""
    R = Rotation.from_rotvec(cam_params[:3]).as_matrix()
    p_cam = R @ point_3d + cam_params[3:6]
    return (K @ (p_cam / p_cam[2]))[:2]

def ba_cost(params, n_cams, n_pts, cam_idx, pt_idx, obs, K):
    """BA 残差函数：所有观测的重投影误差"""
    cams = params[:n_cams * 6].reshape(n_cams, 6)
    pts  = params[n_cams * 6:].reshape(n_pts, 3)
    errors = [project_point(cams[ci], pts[pi], K) - obs[i]
              for i, (ci, pi) in enumerate(zip(cam_idx, pt_idx))]
    return np.concatenate(errors)

def run_bundle_adjustment(cam_init, pts_init, cam_idx, pt_idx, obs, K):
    """运行 BA，返回优化后的位姿和三维点"""
    x0 = np.concatenate([cam_init.ravel(), pts_init.ravel()])
    n_cams, n_pts = len(cam_init), len(pts_init)
    res = least_squares(ba_cost, x0, method='lm', max_nfev=200,
                        args=(n_cams, n_pts, cam_idx, pt_idx, obs, K))
    return (res.x[:n_cams*6].reshape(n_cams, 6),
            res.x[n_cams*6:].reshape(n_pts, 3))
```

### 三维可视化

```python
import open3d as o3d
import numpy as np

def depth_to_pointcloud(depth_metric, rgb, K, max_depth=200.0):
    """将度量深度图反投影为彩色点云"""
    H, W = depth_metric.shape
    u, v  = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth_metric > 0) & (depth_metric < max_depth)

    z = depth_metric[valid]
    x = (u[valid] - K[0,2]) * z / K[0,0]
    y = (v[valid] - K[1,2]) * z / K[1,1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([x, y, z], axis=1))
    pcd.colors = o3d.utility.Vector3dVector(rgb[valid] / 255.0)
    return pcd

# 多帧点云融合可视化（数据加载/帧遍历代码省略）
# o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])
# 预期输出：俯视图下，飞行路径下方的地面地形清晰可见，建筑边缘锐利
```

## 实验

### 数据集说明

论文（[arxiv:2604.04667](https://arxiv.org/abs/2604.04667v1)）使用 DLR MACS 系统：
- 飞行高度 ≈ 50 m，地面采样距离（GSD）≈ 0.85 cm/px
- 每帧覆盖约 2,650 m² 地面
- 以手动标注点云作为真值（存在少量噪声）

公开基准推荐：

| 数据集 | 场景 | 特点 |
|-------|------|------|
| EuRoC MAV | 室内/室外 | 有 IMU，精确真值 |
| UAVID | 城市无人机 | 高分辨率 |
| SensatUrban | 城市点云 | 超大规模 |

### 定量评估

| 方向 | 误差 | 说明 |
|-----|------|------|
| 水平 XY | ≈ 0.87 m | GSD=0.85 cm 时约 100 像素的绝对误差 |
| 垂直 Z | ≈ 0.12 m | 高度方向精度远好于水平 |
| 单帧耗时 | 1.47~4.91 s | 取决于 BA 收敛和扩散步数 |

垂直精度远好于水平精度——这符合预期：深度方向的视差变化更容易被相对深度模型捕捉，而 XY 误差受相机标定和 BA 初始值影响更大。

## 工程实践

### 实时性瓶颈分析

| 步骤 | 耗时占比 | 优化方向 |
|-----|---------|---------|
| 特征提取与匹配 | ~40% | SuperPoint+LightGlue 替代 SIFT |
| Bundle Adjustment | ~35% | 减少点数，使用稀疏求解器（g2o/Ceres） |
| 扩散模型推理 | ~20% | 降低扩散步数，FP16/INT8 量化 |
| 仿射对齐 | <5% | 无需优化 |

### 常见坑

**坑 1：锚点分布不均匀导致对齐失效**

低纹理区域（大面积草地、水面）特征点稀少，BA 产生的稀疏点集中在有纹理的边缘，导致中心区域外推误差大。

```python
def stratified_anchor_sampling(sparse_depth, grid_size=8, min_per_cell=3):
    """空间分层采样，确保锚点覆盖全图而非只集中在有纹理区域"""
    H, W = sparse_depth.shape
    ch, cw = H // grid_size, W // grid_size
    selected = np.zeros_like(sparse_depth, dtype=bool)

    for i in range(grid_size):
        for j in range(grid_size):
            cell  = sparse_depth[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            valid = np.argwhere(cell > 0)
            if len(valid) >= min_per_cell:
                idx = np.random.choice(len(valid), min_per_cell, replace=False)
                for r, c in valid[idx]:
                    selected[i*ch+r, j*cw+c] = True
    return selected
```

**坑 2：集群间尺度漂移**

相邻集群的重叠帧应预测相同深度，但各自独立 BA 会产生微小全局漂移。

```python
def check_cluster_consistency(depth_a, depth_b, threshold_m=0.5):
    """检查两集群对同一重叠帧的深度预测是否一致"""
    valid = (depth_a > 0) & (depth_b > 0)
    median_diff = np.median(np.abs(depth_a[valid] - depth_b[valid]))
    if median_diff > threshold_m:
        print(f"警告：集群间不一致 {median_diff:.3f}m，考虑全局位姿图优化")
    return median_diff
```

**坑 3：扩散模型对俯视图的泛化性**

Marigold 等模型主要在自然场景训练，无人机俯视视角与训练分布差异大，可能出现**深度反转**。检测方法：计算 BA 稀疏深度与模型预测深度的 Spearman 秩相关系数，若 $\rho < 0.6$，降低该帧权重或跳过。

### 数据采集建议

- **重叠率**：航向 ≥ 80%，旁向 ≥ 60%（BA 的生命线）
- **飞行速度**：≤ 5 m/s，减少运动模糊
- **光照**：避免正午强光（镜面反射破坏特征匹配）
- **高度**：25~100 m 最佳，太低遮挡多，太高 GSD 过低

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 静态地形测量（农田、工地） | 动态目标多（车流、人群） |
| 没有 LiDAR，需快速三维估计 | 需要厘米级测量精度 |
| 灾害现场快速评估 | 需要实时（< 0.1 s/帧）的场景 |
| 低纹理区域（扩散模型补全稠密深度） | 完全无纹理场景（水面、雪地） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 经典 MVS（COLMAP） | 精度高，无学习依赖 | 低纹理失败，离线慢 | 精密离线测量 |
| NeRF / 3DGS | 渲染质量好 | 需完整采集，不能流式处理 | 小场景三维建模 |
| 单目扩散深度 | 速度快，泛化强 | 无度量尺度，帧间不一致 | 单图深度预估 |
| **ZeD-MAP** | **度量一致，流式处理** | **依赖 BA 收敛，精度受限** | **实时无人机测绘** |

## 我的观点

ZeD-MAP 的思路非常务实：它不试图训练一个"大一统"的深度网络，而是充分利用 BA 的几何约束和扩散模型的感知能力——"各司其职，互相补足"。

**值得关注的方向：**

1. **扩散骨干可替换**：论文不绑定特定模型，随着 Depth Anything V3 等模型更新，对齐质量会自然提升
2. **IMU 融合**：当前只用视觉，加入视觉-惯导紧耦合 BA 能显著提升高动态环境鲁棒性
3. **语义引导的集群划分**：固定窗口换成根据场景变化（检测到场景切换）动态调整边界，减少错误集群

**离实际部署还有多远？**

在灾害响应类场景（不追求厘米级精度），ZeD-MAP 已接近可用。主要瓶颈是单帧 1.5~5 秒——对 30 fps 飞行视频需要激进的关键帧选择和并行流水线。把扩散模型换成更快的确定性深度网络（牺牲部分泛化性），可以进一步压缩到亚秒级。

核心局限没有变：**没有密集度量真值，深度估计的绝对精度永远依赖稀疏先验的质量。** ZeD-MAP 用 BA 绕开了标注数据的需求，但 BA 本身的精度由特征点质量和场景可观测性决定。在大面积均匀场景（沙漠、雪地、水面）中，BA 失效，ZeD-MAP 也会随之失效——这是整条技术链的上限。