---
layout: post-wide
title: "多机器人分布式 SLAM：用场景图匹配降低 90% 通信开销"
date: 2026-06-16 08:04:27 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.16881v1
generated_by: Claude Code CLI
---

## 一句话总结

SGM-SLAM 用**物体级场景图**代替特征点来实现多机器人地图融合——只交换"椅子在哪、门在哪"这类语义信息，通信量降至特征级方法的几十分之一，同时保持地图质量。

论文链接：[SGM-SLAM: Scene Graph Matching for Data-Efficient Distributed SLAM](https://arxiv.org/abs/2606.16881v1)

## 为什么这个问题重要？

多机器人协同探索时，**分布式 SLAM** 的核心问题是闭环检测：机器人 A 和机器人 B 需要判断"我们到过同一个地方吗？"传统方法用视觉特征描述子（SIFT、SuperPoint），每次同步要传输几 MB 数据。在 WiFi 不稳定的室内或低带宽野外环境，这根本行不通。

SGM-SLAM 的核心创新：**只用对象的语义标签和质心做图匹配，不依赖任何视觉特征**。本质上，它把"机器人间的共识"从像素级降维到了语义级。

## 背景知识

### 多机器人 SLAM 的共享数据结构

| 表示方式 | 数据量 | 语义 | 通信友好 |
|---------|--------|------|---------|
| 原始点云 | 极大 | 无 | 极差 |
| 特征描述子 | 大 | 弱 | 差 |
| Keyframe 压缩 | 中 | 无 | 一般 |
| **对象级场景图** | **极小** | **强** | **好** |

### 位姿图优化基础

多机器人 SLAM 的骨架是**位姿图**：节点是位姿，边是约束。当两台机器人识别出同一场景后，在位姿图中添加跨机器人约束，然后全局优化：

$$
\min_{\mathbf{x}} \sum_{(i,j) \in \mathcal{E}} \| \mathbf{x}_i \ominus \mathbf{x}_j - \mathbf{z}_{ij} \|_{\Sigma_{ij}}^2
$$

其中 $\mathbf{z}_{ij}$ 是测量约束，$\Sigma_{ij}$ 是信息矩阵，$\ominus$ 表示位姿差。

## 核心方法

### 直觉解释

想象两人各自记录："走廊里有张椅子，3 米后有扇门，门左边有台显示器"。即使没有照片，仅凭这些语义描述就能判断"你们经过了同一个地方"，并推断出相对位置。SGM-SLAM 就是把这个直觉形式化。

### Pipeline 概览

```
LiDAR + RGB + IMU
    │
    ├─→ 里程计估计 (LIO / VIO)
    │
    └─→ RGB-LiDAR 融合 → 语义点云 → 3D 目标检测
                                          │
                                          └─→ 场景图 G_i = (V_i, E_i)
                                                    │
                              多机器人通信（只传标签+质心）
                                                    │
                                          场景图匹配 → 相对位姿约束
                                                    │
                                          全局位姿图优化 → 融合地图
```

### 数学细节

**场景图定义**

对于机器人 $r$，其场景图 $\mathcal{G}^r = (\mathcal{V}^r, \mathcal{E}^r)$ 中：
- 节点 $v_i = (\ell_i,\, \mathbf{c}_i)$，$\ell_i$ 为语义标签，$\mathbf{c}_i \in \mathbb{R}^3$ 为质心
- 边权 $e_{ij} = \|\mathbf{c}_i - \mathbf{c}_j\|_2$

**图匹配打分**

节点兼容性（标签一致才为 1）：

$$
C_{\text{node}}(i, j) = \mathbf{1}[\ell_i^1 = \ell_j^2]
$$

边兼容性（相对距离的高斯相似度）：

$$
C_{\text{edge}}(i_1, j_1, i_2, j_2) = \exp\!\left(-\frac{(d_{i_1 j_1}^1 - d_{i_2 j_2}^2)^2}{2\sigma^2}\right)
$$

**相对位姿估计**

给定 $K$ 对匹配的物体质心，SVD 求解最优刚体变换：

$$
(\mathbf{R}^*, \mathbf{t}^*) = \arg\min_{\mathbf{R},\mathbf{t}} \sum_{k=1}^{K} \|\mathbf{c}_k^2 - \mathbf{R}\mathbf{c}_k^1 - \mathbf{t}\|^2
$$

## 实现

### 场景图数据结构

```python
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class SemanticObject:
    label: str            # 语义标签，如 'chair', 'door'
    centroid: np.ndarray  # 3D 质心坐标 [x, y, z]
    obj_id: int

@dataclass
class SceneGraph:
    objects: List[SemanticObject]
    robot_id: int

    def pairwise_distances(self) -> np.ndarray:
        """计算对象两两间欧氏距离矩阵"""
        n = len(self.objects)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.objects[i].centroid - self.objects[j].centroid)
                D[i, j] = D[j, i] = d
        return D

    def to_compact_msg(self) -> dict:
        """序列化为通信消息（只含标签+质心，体积极小）"""
        return {
            "robot_id": self.robot_id,
            "objects": [
                {"label": o.label, "centroid": o.centroid.tolist()}
                for o in self.objects
            ]
        }
```

### 场景图匹配

```python
def match_scene_graphs(
    g1: SceneGraph, g2: SceneGraph,
    sigma: float = 1.0, threshold: float = 0.4
) -> List[Tuple[int, int]]:
    """
    基于节点标签 + 边距离一致性的场景图匹配（匈牙利算法）
    返回匹配对列表 [(g1中索引, g2中索引), ...]
    """
    n1, n2 = len(g1.objects), len(g2.objects)
    if n1 == 0 or n2 == 0:
        return []

    D1, D2 = g1.pairwise_distances(), g2.pairwise_distances()
    cost = np.full((n1, n2), 1e6)

    for i in range(n1):
        for j in range(n2):
            # Step1：标签必须一致
            if g1.objects[i].label != g2.objects[j].label:
                continue

            # Step2：边兼容性——考虑与其他同标签节点的距离一致性
            edge_scores = []
            for k1 in range(n1):
                for k2 in range(n2):
                    if k1 == i or k2 == j:
                        continue
                    if g1.objects[k1].label == g2.objects[k2].label:
                        diff = D1[i, k1] - D2[j, k2]
                        edge_scores.append(np.exp(-(diff**2) / (2 * sigma**2)))

            edge_score = np.mean(edge_scores) if edge_scores else 0.5
            cost[i, j] = 1.0 - edge_score  # 转为最小化代价

    row_ind, col_ind = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(row_ind, col_ind) if cost[r, c] < threshold]
```

### 相对位姿估计

```python
def estimate_relative_pose(
    matches: List[Tuple[int, int]],
    g1: SceneGraph, g2: SceneGraph
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    从匹配物体质心估计机器人相对位姿（SVD 方法）
    至少需要 3 个匹配点；返回 (R, t)，使得 c2 ≈ R @ c1 + t
    """
    if len(matches) < 3:
        return None

    pts1 = np.array([g1.objects[m[0]].centroid for m in matches])
    pts2 = np.array([g2.objects[m[1]].centroid for m in matches])

    mu1, mu2 = pts1.mean(0), pts2.mean(0)
    H = (pts1 - mu1).T @ (pts2 - mu2)  # 交叉协方差矩阵

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 处理反射解（det < 0 时翻转最后一行）
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = mu2 - R @ mu1
    return R, t
```

### 完整 Demo

```python
def demo():
    # 机器人 A 观测到的场景图
    g_a = SceneGraph(robot_id=0, objects=[
        SemanticObject("chair",   np.array([1.0, 0.0, 0.5]), 0),
        SemanticObject("door",    np.array([4.0, 0.0, 1.0]), 1),
        SemanticObject("monitor", np.array([2.0, 3.0, 1.2]), 2),
        SemanticObject("cabinet", np.array([0.0, 5.0, 0.8]), 3),
    ])

    # 机器人 B 观测同一房间（绕Z轴转30°，平移[2,1,0]）
    R_true = np.array([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]])
    t_true = np.array([2.0, 1.0, 0.0])

    g_b = SceneGraph(robot_id=1, objects=[
        SemanticObject(o.label, R_true @ o.centroid + t_true, i)
        for i, o in enumerate(g_a.objects)
    ])

    msg_size = len(str(g_b.to_compact_msg()).encode())
    print(f"通信数据量: ~{msg_size} bytes（{len(g_b.objects)} 个对象）")

    matches = match_scene_graphs(g_a, g_b)
    print(f"找到 {len(matches)} 对匹配: {matches}")

    result = estimate_relative_pose(matches, g_a, g_b)
    if result:
        R_est, t_est = result
        t_err = np.linalg.norm(t_est - t_true)
        angle_err = np.degrees(np.arccos(
            np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1, 1)
        ))
        print(f"平移误差: {t_err:.4f} m | 旋转误差: {angle_err:.2f}°")

demo()
```

### 3D 可视化

```python
import open3d as o3d

def visualize_matched_graphs(g1, g2, matches):
    """可视化两个场景图及其匹配关系"""
    geoms = []
    for obj in g1.objects:  # 机器人A：蓝色
        s = o3d.geometry.TriangleMesh.create_sphere(0.15)
        s.paint_uniform_color([0.2, 0.4, 0.8])
        s.translate(obj.centroid)
        geoms.append(s)

    for obj in g2.objects:  # 机器人B：橙色
        s = o3d.geometry.TriangleMesh.create_sphere(0.15)
        s.paint_uniform_color([0.9, 0.5, 0.1])
        s.translate(obj.centroid)
        geoms.append(s)

    # 匹配连线：绿色
    pts, lines = [], []
    for idx, (m1, m2) in enumerate(matches):
        pts += [g1.objects[m1].centroid, g2.objects[m2].centroid]
        lines.append([idx * 2, idx * 2 + 1])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[0, 0.8, 0.2]] * len(lines))
    geoms.append(ls)

    o3d.visualization.draw_geometries(geoms)
    # ... (保存图片代码省略)
```

预期输出：蓝色球体表示机器人A的对象，橙色球体表示机器人B的对象，绿线连接匹配对。两组球体在空间上存在旋转+平移关系。

## 实验

### 数据集说明

论文使用了室内和室外两类数据集，由四足机器人（如 Spot）采集：

| 数据集类型 | 传感器 | 场景 | 挑战 |
|---------|--------|------|------|
| 室内模拟 | LiDAR+RGB+IMU | 办公室 | 对称场景歧义 |
| 室外真实 | LiDAR+RGB+IMU | 校园 | 光照变化、稀疏特征 |

关键评估指标：
- **位姿精度**：RMSE（平移/旋转误差）
- **通信效率**：每次同步传输数据量
- **匹配召回率**：正确识别共同观测区域的比率

### 通信量量化对比

| 方法 | 每次同步数据量 | 100帧累计 |
|-----|-------------|---------|
| 特征描述子 (NetVLAD) | ~4 KB | ~400 KB |
| Keyframe + 描述子 | ~50 KB | ~5 MB |
| DiSCo-SLAM（点云） | ~200 KB | ~20 MB |
| **SGM-SLAM（本文）** | **~0.5 KB** | **~50 KB** |

## 工程实践

### 语义分割的选择权衡

SGM-SLAM 的质量上限由语义分割决定：

```python
# 实时优先：YOLOv8-seg，~30ms/frame on RTX 3080
# 精度优先：Mask3D / OneFormer，~200ms/frame，需降帧处理
# 嵌入式部署：MobileNetV3-seg，~50ms on Jetson Orin

STATIC_LABELS = {"door", "column", "cabinet", "window", "monitor", "shelf"}
# 只保留静态物体，过滤人、推车等动态目标
objects = [o for o in raw_objects if o.label in STATIC_LABELS]
```

### 常见坑

**坑 1：标签空间不一致**

```python
# 问题：A用'chair'，B用'seat'，同物不同名导致匹配失败
# 解决：统一标签词典
LABEL_ALIASES = {"seat": "chair", "sofa": "couch", "screen": "monitor"}
label = LABEL_ALIASES.get(raw_label, raw_label)
```

**坑 2：对称走廊的歧义匹配**

```python
# 问题：走廊两侧各有4张相同椅子，匹配无法区分
# 解决：加入对象相对于机器人前进方向的角度特征
def augmented_label(obj, robot_heading):
    angle = np.arctan2(obj.centroid[1], obj.centroid[0]) - robot_heading
    sector = int(np.degrees(angle) // 45)  # 8个方位区间
    return f"{obj.label}_sector{sector}"
```

**坑 3：匹配点数不足**

```python
# 问题：两机器人重叠区域太小，共同物体少于3个，无法估位姿
# 解决：回退到基于里程计的初始对齐，扩大搜索半径后重匹配
if len(matches) < 3:
    # 用里程计给出粗对齐，变换后重新匹配
    fallback_to_odometry_alignment()
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 室内结构化环境（办公室、仓库） | 室外开阔无特征场景 |
| 带宽受限的多机器人系统 | 单机器人 SLAM（杀鸡用牛刀） |
| 静态或低动态环境 | 人群密集的高动态场景 |
| 对象密度适中（5-20 物体/帧） | 纯走廊等几何单一场景 |
| 四足/轮式机器人搭载 LiDAR+RGB | 纯相机系统（无 LiDAR） |

## 与其他方法对比

| 方法 | 通信开销 | 语义 | 大场景扩展 | 实时性 |
|-----|---------|------|----------|-------|
| COVINS（特征级） | 高 | 无 | 一般 | 是 |
| Kimera-Multi | 中 | 网格 | 差 | 勉强 |
| DiSCo-SLAM（点云） | 极高 | 无 | 差 | 否 |
| **SGM-SLAM** | **极低** | **强** | **好** | **是** |

## 我的观点

SGM-SLAM 的思路直觉且实用——**人类之间也是靠语言描述地标来协作导航的**，而不是传输像素。把这个常识形式化为图匹配算法，是一步聪明的工程决策。

几个值得关注的开放问题：

1. **稀疏环境的退化**：当场景对象少于 3 个时，位姿估计退化到纯里程计。室外大场景尤其明显，这是目前最大的短板。

2. **标签一致性假设**：不同机器人必须使用相同分割模型和类别词典，跨厂商部署时这个假设经常被打破。

3. **开放词汇语义的机会**：固定类别标签正在被 CLIP、Grounding DINO 等开放词汇模型替代。将开放词汇嵌入与场景图匹配结合，是很自然的下一步。

4. **N 机器人一致性**：论文展示了 2 台机器人的场景，当 $N > 5$ 时，分布式优化的收敛性和全局一致性保证是悬而未决的问题。

离商业部署还有一段距离，主要卡在**语义分割实时性**和**无结构环境的鲁棒性**上。但作为多机器人协作的研究框架，它提供了一个干净的抽象：**让机器人用"物体语言"而非"像素语言"交流**。