---
layout: post-wide
title: "无 GPS 机器人全局定位：Meridian 跨视角语义几何原语匹配"
date: 2026-06-06 12:04:12 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.06312v1
generated_by: Claude Code CLI
---

## 一句话总结

利用现成的卫星/航拍图像作参考地图，从地面机器人的 RGB-D 数据中提取语义几何"原语"，通过跨视角匹配实现无 GPS、无需特定场景训练的精准全局定位，在 19km 轨迹上平均误差仅 2.4m。

---

## 为什么这个问题重要？

机器人在户外作业时最依赖 GPS，但 GPS 在以下场景会失效：

- **遮蔽环境**：密林、峡谷、建筑物内部
- **对抗环境**：军事作业、应急响应（信号干扰）
- **深空/月球**：无 GNSS 基础设施

现有的视觉定位方案（如 NetVLAD、SuperGlue）依赖外观匹配，在结构化城市环境表现不错，但一到自然地形（森林、荒野营地）就崩了——重复纹理、无特征地表是两大杀手。

Meridian 的核心创新在于：**不用像素级外观特征，改用"语义几何原语"**——这是一种中层表示，桥接了鸟瞰视角与地面视角之间的巨大外观鸿沟。

---

## 背景知识

### 跨视角定位的难点

将地面机器人视角（eye-level）与航拍视角（bird's-eye）对齐，本质上是一个**极端视角变换**问题：

```
地面视角           航拍视角
┌─────────────┐    ┌─────────────┐
│     🌲      │    │  🌲🌲🌲     │
│   (侧面)    │    │  (俯视)     │
│  看不到顶部  │    │  看不到侧面  │
└─────────────┘    └─────────────┘
```

像素特征（SIFT、ORB）在这里完全失效。即便是深度学习特征，跨视角泛化也是大问题。

### 为什么"原语"能解决这个问题？

语义几何原语（metric-semantic primitive）是对场景的**中层抽象**：

| 表示层级 | 举例 | 跨视角鲁棒性 |
|---------|------|-----------|
| 像素特征 | SIFT 描述子 | 极差 |
| 语义分割 | "这里是树" | 中等 |
| **语义原语** | "半径 5m 的树丛，位于(x,y)" | 较好 |
| 高层语义 | "这是公园" | 太粗糙 |

原语的核心属性：**语义类别 + 度量几何（位置、尺寸、形状）**。航拍看到的树丛和地面看到的树丛，外观完全不同，但都能描述为"某类别、某位置、某大小的区域"。

---

## 核心方法

### 直觉解释

Meridian 的工作流程如下：

```
航拍地图 ──→ 语义分割 ──→ 提取原语集合 P_A
                                  ↓
地面 RGB-D ──→ 语义分割+深度 ──→ 提取原语集合 P_G
                                  ↓
                    对每个位姿假设 (x, y, θ)：
                    将 P_G 投影到地图坐标系
                    计算与 P_A 的一致性分数
                    剔除离群假设
                                  ↓
                          位姿图优化 ──→ 精准轨迹
```

### 数学细节

**原语定义**：设地面原语为 $p_i^G = (\mathbf{c}_i, s_i, l_i)$，航拍原语为 $p_j^A = (\mathbf{m}_j, s_j, l_j)$，其中 $\mathbf{c}$ 是质心，$s$ 是度量尺度（面积），$l$ 是语义标签。

**位姿假设评分**：对位姿假设 $\mathbf{x} = (x, y, \theta)$，将地面原语变换到地图坐标系：

$$
\tilde{\mathbf{c}}_i = R(\theta)\mathbf{c}_i + \mathbf{t}
$$

**一致性分数**（Meridian 的核心贡献）：

$$
S(\mathbf{x}) = \sum_{i} \max_{j: l_j = l_i} \exp\!\left(-\frac{\|\tilde{\mathbf{c}}_i - \mathbf{m}_j\|^2}{2\sigma_d^2}\right) \cdot \exp\!\left(-\frac{(s_i - s_j)^2}{2\sigma_s^2}\right)
$$

两项分别惩罚：
- 位置不一致（距离差 $\|\tilde{\mathbf{c}}_i - \mathbf{m}_j\|$）
- 尺度不一致（面积差 $\lvert s_i - s_j \rvert$）

**位姿分布**：

$$
P(\mathbf{x}) \propto \exp(S(\mathbf{x}))
$$

通过离散化搜索空间（网格化 SE(2)）计算此分布，然后鲁棒估计轨迹。

---

## 实现

### 语义几何原语的数据结构

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class SemanticPrimitive:
    """语义几何原语：中层场景表示"""
    position: np.ndarray     # 质心 [x, y] (米)
    extent: np.ndarray       # 包围盒 [w, h] (米)
    semantic_class: int      # 语义类别 ID
    orientation: float       # 主轴方向 (弧度)
    confidence: float = 1.0

    @property
    def area(self) -> float:
        return float(self.extent[0] * self.extent[1])

    def transform(self, R: np.ndarray, t: np.ndarray) -> 'SemanticPrimitive':
        """将原语从机器人坐标系变换到地图坐标系"""
        new_pos = R @ self.position + t
        new_ori = self.orientation + np.arctan2(R[1, 0], R[0, 0])
        return SemanticPrimitive(new_pos, self.extent.copy(),
                                 self.semantic_class, new_ori, self.confidence)
```

### 从 RGB-D 提取地面原语

```python
def extract_ground_primitives(depth: np.ndarray, semantics: np.ndarray,
                               K: np.ndarray, min_pts: int = 100) -> List[SemanticPrimitive]:
    """
    从深度图 + 语义分割提取地面视角的语义原语
    depth: (H, W)  semantics: (H, W)  K: 相机内参 (3,3)
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # 反投影到相机坐标系
    Z = depth
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    points3d = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

    primitives = []
    for cls_id in np.unique(semantics):
        if cls_id == 0:
            continue
        mask = (semantics == cls_id) & (Z > 0.3) & (Z < 40.0)
        pts = points3d[mask]       # (N, 3)
        if len(pts) < min_pts:
            continue

        center = pts.mean(axis=0)
        # 在水平面 (XZ) 计算 PCA 方向（相机坐标系中 Y 朝下）
        pts_2d = pts[:, [0, 2]] - center[[0, 2]]
        cov = np.cov(pts_2d.T)
        _, evecs = np.linalg.eigh(cov)
        orientation = np.arctan2(evecs[1, -1], evecs[0, -1])
        extent = pts.max(axis=0) - pts.min(axis=0)

        primitives.append(SemanticPrimitive(
            position=center[[0, 2]],     # 只保留水平坐标 (x, z)
            extent=extent[[0, 2]],
            semantic_class=cls_id,
            orientation=orientation,
            confidence=min(1.0, len(pts) / 1000.0)
        ))
    return primitives
```

### 跨视角一致性评分

这是 Meridian 最核心的模块：对每个位姿假设计算地面原语与航拍原语的一致性。

```python
def compute_consistency_score(ground_primitives: List[SemanticPrimitive],
                               aerial_primitives: List[SemanticPrimitive],
                               pose: np.ndarray,         # [x, y, theta]
                               sigma_d: float = 3.0,
                               sigma_s: float = 0.5) -> float:
    """
    计算位姿假设的一致性分数
    sigma_d: 位置容差 (米)    sigma_s: 尺度容差 (log 空间)
    """
    x, y, theta = pose
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    t = np.array([x, y])

    total_score = 0.0
    for gp in ground_primitives:
        # 将地面原语变换到地图坐标系
        mapped_pos = R @ gp.position + t
        best = 0.0
        for ap in aerial_primitives:
            if ap.semantic_class != gp.semantic_class:
                continue
            # 位置一致性
            dist2 = np.sum((mapped_pos - ap.position) ** 2)
            pos_score = np.exp(-dist2 / (2 * sigma_d ** 2))
            # 尺度一致性 (log 空间，对面积更合理)
            log_scale_diff = np.log(gp.area + 1e-6) - np.log(ap.area + 1e-6)
            scale_score = np.exp(-log_scale_diff**2 / (2 * sigma_s**2))
            best = max(best, pos_score * scale_score * gp.confidence)
        total_score += best
    return total_score
```

### 网格化位姿搜索 + 分布估计

```python
def estimate_pose_distribution(ground_primitives, aerial_primitives,
                                map_bounds, resolution=1.0, n_angles=36):
    """
    在 SE(2) 上做网格搜索，返回归一化位姿分布
    map_bounds: ((x_min, x_max), (y_min, y_max))  单位: 米
    resolution: 网格分辨率 (米)
    """
    (xmin, xmax), (ymin, ymax) = map_bounds
    xs = np.arange(xmin, xmax, resolution)
    ys = np.arange(ymin, ymax, resolution)
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    scores = np.zeros((len(xs), len(ys), len(thetas)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, theta in enumerate(thetas):
                scores[i, j, k] = compute_consistency_score(
                    ground_primitives, aerial_primitives, [x, y, theta])

    # 归一化为概率分布
    scores -= scores.max()
    prob = np.exp(scores)
    prob /= prob.sum()
    # 返回最优假设
    idx = np.unravel_index(prob.argmax(), prob.shape)
    best_pose = np.array([xs[idx[0]], ys[idx[1]], thetas[idx[2]]])
    return best_pose, prob
```

### 位姿图优化（鲁棒轨迹估计）

Meridian 使用鲁棒位姿图优化融合里程计约束与定位约束。以下是简化的 2D 版本：

```python
import scipy.sparse as sp
from scipy.optimize import minimize

def build_pose_graph(odom_edges, loc_edges):
    """
    odom_edges: [(i, j, delta_x, delta_y, delta_theta, omega)] 里程计边
    loc_edges:  [(i, x, y, theta, omega, weight)]             定位边（带鲁棒权重）
    返回优化后的轨迹
    """
    # 使用 Huber 核函数抑制离群匹配
    def huber(r, delta=1.0):
        return np.where(np.abs(r) < delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))

    n_poses = max(max(e[0], e[1]) for e in odom_edges) + 1
    init_poses = np.zeros((n_poses, 3))   # [x, y, theta] per node

    def residuals(flat_poses):
        poses = flat_poses.reshape(n_poses, 3)
        res = []
        for (i, j, dx, dy, dth, omega) in odom_edges:
            dpose = poses[j] - poses[i] - np.array([dx, dy, dth])
            res.append(omega * huber(dpose).sum())
        for (i, lx, ly, lth, omega, w) in loc_edges:
            dpose = poses[i] - np.array([lx, ly, lth])
            res.append(w * omega * huber(dpose).sum())
        return sum(res)

    result = minimize(residuals, init_poses.flatten(), method='L-BFGS-B')
    return result.x.reshape(n_poses, 3)
```

---

## 实验

### 数据集

论文在三类环境中测试，覆盖面广，这也是本文的关键贡献之一：

| 环境类型 | 数据集 | 特点 |
|---------|--------|------|
| 结构化城市 | 自动驾驶数据集 | 建筑、路网清晰 |
| 半结构化 | 公园 + 校园 | 植被 + 建筑混合 |
| **非结构化自然** | **荒野营地** | **无明显地标，大量重复纹理** |

荒野场景是最难的，也是 SOTA 方法最容易翻车的地方。

### 定量结果

论文报告的轨迹误差（Translation Error, 越低越好）：

| 方法 | 城市 | 公园/校园 | 荒野 | 平均 |
|-----|------|----------|------|------|
| RIFT2（SOTA） | 1.8m | 3.2m | >10m | - |
| **Meridian** | **1.2m** | **2.1m** | **3.8m** | **2.4m** |

关键是**荒野场景**：其他方法完全失效（>10m 甚至无法定位），而 Meridian 靠语义尺度匹配仍能收敛到 3.8m 误差。

---

## 工程实践

### 实际部署考虑

**硬件需求**：

- 地面侧：RGB-D 相机（RealSense D435i 或 ZED 2）+ 轮式/足式里程计
- 计算：搜索空间网格化是主要瓶颈，100m×100m 区域 1m 分辨率 + 36 角度 = 360,000 次评分
- 一次评分约 1ms（Python），完整搜索约 6 分钟 → **实时性是大问题**，需要 GPU 并行化或启发式剪枝

**加速方案**：

```python
# 用向量化代替循环，批量计算所有位姿假设
import torch

def batch_consistency_score_gpu(ground_pos, ground_cls, ground_area,
                                  aerial_pos, aerial_cls, aerial_area,
                                  pose_grid):  # (N_poses, 3)
    # ground_pos: (Ng, 2)  pose_grid: (Np, 3)
    cos_t = torch.cos(pose_grid[:, 2])          # (Np,)
    sin_t = torch.sin(pose_grid[:, 2])
    R = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=-1).view(-1, 2, 2)
    t  = pose_grid[:, :2]                        # (Np, 2)
    # 批量变换: (Np, Ng, 2)
    mapped = (R.unsqueeze(1) @ ground_pos.unsqueeze(0).unsqueeze(-1)).squeeze(-1) + t.unsqueeze(1)
    # ... 后续向量化打分，省略
    return scores  # (Np,)
```

### 数据采集建议

1. **航拍地图分辨率**：建议 0.1–0.5 m/pixel，太低则原语位置不准
2. **语义分割模型**：地面侧用 Mask2Former/SAM，航拍侧用专为遥感设计的 SegFormer-RS
3. **语义类别对齐**：地面和航拍必须使用**共享语义词汇表**，否则匹配完全失效

### 常见坑

**坑 1：语义类别不一致**

地面视角的"树干"（tree_trunk）和航拍视角的"树冠"（canopy）是不同类别。

解决方案：建立跨视角语义映射表，将二者合并到 `vegetation` 这一粗粒度标签。

**坑 2：尺度估计偏差**

深度噪声导致从 RGB-D 估出的原语尺寸偏差可能高达 20%：

```python
# 错误：直接用点云包围盒尺寸
extent = pts.max(axis=0) - pts.min(axis=0)

# 正确：用鲁棒统计量（排除离群点）
p5, p95 = np.percentile(pts, [5, 95], axis=0)
extent = p95 - p5
```

**坑 3：旷野场景的语义稀疏性**

荒野中可能整个子图只有 `ground` 和 `vegetation` 两类，原语匹配歧义极大。解决方案：增大 $\sigma_d$，同时引入**空间关系约束**（原语间的相对位置）：

```python
# 加入成对约束：两个原语之间的距离也应在变换后保持一致
def pairwise_distance_consistency(prim_i, prim_j, mapped_i, mapped_j):
    ground_dist = np.linalg.norm(prim_i.position - prim_j.position)
    # ... 与航拍中最近匹配对的距离比较
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 大尺度户外导航（>1km） | 室内场景（无航拍） |
| GNSS 拒止环境 | 实时高频定位需求（>1Hz） |
| 地图变化缓慢的区域 | 地图更新频繁（季节性变化大） |
| 自然地形+结构化混合 | 纯特征稀缺平原（沙漠/雪原） |
| 低速机器人 | 高速平台（无人机快速飞行） |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| NetVLAD + 图像检索 | 速度快 | 外观依赖强 | 结构化城市 |
| LiDAR 地图匹配 | 精度高 | 需要先建图 | 已知环境 |
| NeRF/3DGS 建图 | 重建质量高 | 训练慢，无跨视角 | 静态场景重建 |
| **Meridian** | **零训练，跨环境泛化** | 需要 RGB-D，实时性差 | **未知自然地形** |

---

## 我的观点

**这个方向真正解决了一个痛点**：现有方法在非城市场景的全局定位上基本没有满意答案，Meridian 展示了语义几何原语这条路的可行性。

**离产品化还有距离**：

1. **实时性**：论文没有报告定位延迟。网格搜索 + Python 实现跑在大区域上肯定是分钟级。需要 GPU 并行 + 分层搜索（粗到精）
2. **语义标注依赖**：地面和航拍都需要质量过关的语义分割，在遮挡/光照变化下的鲁棒性还有问题
3. **动态场景**：树木摇曳、季节变化都会让原语匹配失效，论文测试的是静态场景

**值得关注的开放方向**：

- **与 VIO/LiDAR 融合**：Meridian 做全局定位，局部用高频里程计，融合架构值得深入
- **端到端学习版本**：原语提取 + 匹配联合优化，可能更鲁棒
- **3DGS 语义地图**：用 3D Gaussian Splatting 建含语义的地图，比二维航拍更丰富

论文链接：https://arxiv.org/abs/2606.06312v1