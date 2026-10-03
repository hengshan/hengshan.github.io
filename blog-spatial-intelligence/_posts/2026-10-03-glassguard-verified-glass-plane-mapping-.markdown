---
layout: post-wide
title: 'GlassGuard：当激光雷达"看不见"玻璃时，机器人该怎么走'
date: 2026-10-03 12:03:15 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2610.02110v1
generated_by: Claude Code CLI
---

## 一句话总结

GlassGuard 是一个专门针对建筑玻璃（窗户、幕墙、玻璃门）的三维重建框架：它用视觉基础模型检测玻璃的 2D 掉脚，用 LiDAR 点云中的"结构性空洞"生成平面候选，再用一个**不需要深度估计**的 2D 投影几何校验来确认这些候选是不是真的玻璃平面，最终把结果融合进全局地图，专门解决"玻璃挡住了激光雷达，但地图上却显示那里是可以走的"这个危险问题。

## 为什么这个问题重要？

室内服务机器人、园区配送车、商场导览机器人几乎都依赖 LiDAR-based SLAM 做导航。但玻璃对激光是半透明的：激光脉冲直接穿过玻璃打到后面的墙或家具上，返回的点云里，玻璃所在的位置是一个"洞"——没有任何障碍物。

这会导致一个经典但极其危险的失败模式：**占据栅格地图把玻璃幕墙标记为自由空间**，机器人规划路径直接撞上去。

已有的一些方法尝试重建缺失的玻璃表面（比如用视觉检测玻璃区域后补一个虚拟平面），但这又带来了相反的问题：如果平面位置、朝向估计不准，就会把本来可通行的空间也标记成障碍物，导致机器人绕远路甚至"卡死"在狭窄通道里走不出去。

GlassGuard 的核心创新就是把这两个失败模式**同时**作为优化目标：

- **玻璃覆盖率（glass coverage）**：有多少真实玻璃被正确识别和建模
- **自由空间污染（free-space contamination）**：有多少真实可通行空间被错误地标记为障碍物

这个"双重约束"思路贯穿了论文的候选生成、校验、全局地图构建三个阶段。

## 背景知识

### 为什么玻璃对 LiDAR 和纯几何方法都是难题

| 表示方式 | 对玻璃的处理能力 | 原因 |
|---------|----------------|------|
| 点云/体素占据栅格 | 差 | 玻璃处激光穿透，无返回点，天然是"洞" |
| 立体视觉深度 | 差 | 玻璃反射+透射混合，视差估计严重失真 |
| NeRF / 3DGS | 中等但非目标场景 | 能学到玻璃的"视觉外观"（反射高光），但不产出可供避障用的几何边界 |
| 显式平面模型（本文） | 好 | 建筑玻璃几乎总是平面结构（窗、幕墙），用平面先验强约束几何 |

这里有个关键的直觉：**玻璃本身没有几何，但玻璃所在的"洞"有几何**——它通常是一个规则的矩形（窗框）或者由墙体结构围成的开口。GlassGuard 正是利用了"建筑玻璃几乎总是镶在规则结构洞里"这个先验。

### 前置知识

读完本文你需要大致了解：
- 占据栅格地图（occupancy grid）的基本概念
- 相机投影模型 $\mathbf{x} = K[R \mid t]\mathbf{X}$
- 两视图之间由平面诱导的单应矩阵（plane-induced homography）——这是本文最核心的几何工具

## 核心方法

### 直觉解释

把整个 pipeline 想象成三道关卡：

```
LiDAR点云 ──→ 找"应该有点但没有点"的空洞 ──→ 生成矩形平面候选(粗)
图像序列 ──→ 视觉基础模型分割玻璃掉脚     ──→
                                              ↓
                        跨视角单应一致性校验 (不需要深度!)
                                              ↓
                        通过校验的平面 ──→ 写入全局地图
                                              ↓
                  同时统计: 覆盖了多少真实玻璃 / 污染了多少自由空间
```

第一关（点云侧）给出"这里大概有个洞"，第二关（视觉侧）给出"这个洞的形状/边界"，第三关是全文最巧妙的部分：**不用重建深度，只用相机相对位姿 + 平面参数，看这个平面能不能"解释"玻璃掉脚在不同视角下的变形**。

### 数学细节

**平面诱导的单应矩阵**：设平面方程为 $n^T X + d = 0$（$n$ 是法向量，$d$ 是到原点距离），两个相机之间的相对位姿为 $R_{21}, t_{21}$，则场景中属于该平面的点，在两个视图之间满足：

$$
H = K_2 \left( R_{21} - \frac{t_{21}\, n^T}{d} \right) K_1^{-1}
$$

这个公式的物理含义是：**如果一组 2D 点确实来自同一个平面**，那么用相机的相对位姿和平面参数就能精确算出它们在另一个视角下的位置，完全不需要知道每个点的深度值。这正是"depth-free"的来源——传统的立体匹配需要逐点估计深度，而这里只需要验证"平面假设"是否自洽。

**校验准则**：把视角 1 中的玻璃掉脚用 $H$ 变换到视角 2，和视角 2 中实际检测到的玻璃掉脚算 IoU：

$$
\text{IoU}(H(\text{mask}_1), \text{mask}_2) \geq \tau
$$

如果这个几何假设（平面位置+朝向）是错的，warp 之后的掉脚会和真实掉脚明显错位，IoU 会很低，候选平面就会被拒绝。

### Pipeline 概览

```
输入: LiDAR点云序列 + 多视角图像序列 + 相机/LiDAR外参标定
  ↓
[阶段1] 视觉基础模型 → 逐帧玻璃实例掉脚 (2D)
[阶段2] 点云结构性空洞检测 → 矩形平面假设 (粗略位置+朝向)
[阶段3] 跨帧单应一致性校验 (depth-free) → 保留/拒绝候选
[阶段4] 全局地图融合 → 占据栅格更新 + 覆盖率/污染率统计
  ↓
输出: 带有验证过的玻璃平面的全局导航地图
```

## 实现

以下代码是教学性的最小实现，用来理解算法骨架，并非官方实现。官方项目页面见文末。

### 环境配置

```bash
pip install numpy opencv-python open3d matplotlib
```

### 阶段1+2：模拟场景与结构性空洞检测

```python
import numpy as np

def generate_room_with_glass(width=10.0, height=3.0,
                               glass_region=((4, 6), (0, 3))):
    """模拟一面带玻璃窗的墙：玻璃区域内激光穿透，无返回点"""
    points = []
    gx0, gx1 = glass_region[0]
    gz0, gz1 = glass_region[1]
    for x in np.linspace(0, width, 200):
        for z in np.linspace(0, height, 60):
            if gx0 <= x <= gx1 and gz0 <= z <= gz1:
                continue  # 玻璃区域: 无返回点
            points.append([x, 0.0, z])
    return np.array(points)


def detect_glass_hole_boundary(points, grid_res=0.1):
    """在墙面点云中找'应该有点但没有点'的矩形空洞（玻璃候选）"""
    proj = points[:, [0, 2]]
    x_min, x_max = proj[:, 0].min(), proj[:, 0].max()
    z_min, z_max = proj[:, 1].min(), proj[:, 1].max()
    nx = int((x_max - x_min) / grid_res) + 1
    nz = int((z_max - z_min) / grid_res) + 1
    occ = np.zeros((nx, nz), dtype=bool)
    idx_x = ((proj[:, 0] - x_min) / grid_res).astype(int)
    idx_z = ((proj[:, 1] - z_min) / grid_res).astype(int)
    occ[idx_x, idx_z] = True
    rows, cols = np.where(~occ)
    if len(rows) == 0:
        return None
    return (rows.min() * grid_res + x_min, rows.max() * grid_res + x_min,
            cols.min() * grid_res + z_min, cols.max() * grid_res + z_min)
```

实际系统中空洞检测要做连通域分析和形态学滤波来剔除噪声造成的小孔洞，这里为了突出主干逻辑做了简化。

### 阶段3：depth-free 单应一致性校验（核心）

```python
import cv2

def plane_induced_homography(K1, K2, R21, t21, n, d):
    """平面 n^T X + d = 0 诱导的两视图单应矩阵
    只依赖相对位姿 + 平面参数，不需要逐点深度
    """
    A = R21 - (t21.reshape(3, 1) @ n.reshape(1, 3)) / d
    return K2 @ A @ np.linalg.inv(K1)


def verify_plane_hypothesis(mask1, mask2, H, iou_thresh=0.5):
    """把view1的玻璃掉脚warp到view2，和真实掉脚比IoU来验证平面假设"""
    h, w = mask2.shape
    warped = cv2.warpPerspective(mask1.astype(np.uint8), H, (w, h))
    inter = np.logical_and(warped > 0, mask2 > 0).sum()
    union = np.logical_or(warped > 0, mask2 > 0).sum()
    iou = inter / max(union, 1)
    return iou >= iou_thresh, iou
```

### 阶段4：全局地图融合与双重指标统计

```python
def fuse_global_map(occupancy_grid, verified_planes):
    """把验证通过的玻璃平面写入占据栅格，同时统计自由空间污染"""
    false_voxels = 0
    covered_glass_voxels = 0
    for plane in verified_planes:
        voxels = rasterize_plane_to_voxels(plane, occupancy_grid.resolution)
        for v in voxels:
            if occupancy_grid.was_marked_free_by_lidar(v):
                # LiDAR因为穿透玻璃把这里标记为free, 现在用视觉证据纠正
                occupancy_grid.mark_occupied(v, source="glass_plane")
                covered_glass_voxels += 1
            else:
                # 该体素原本已有真实障碍物, 平面放错位置, 记为误检
                false_voxels += 1
    return occupancy_grid, covered_glass_voxels, false_voxels
```

`rasterize_plane_to_voxels` 和 `occupancy_grid` 的具体实现（体素哈希、射线投射判断 free/occupied）在教学代码中省略，它们是标准的占据栅格 SLAM 组件，不是本文的创新点。

### 3D 可视化

```python
import open3d as o3d

def visualize_glass_result(points, glass_plane_bbox):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    x0, x1, z0, z1 = glass_plane_bbox
    mesh = o3d.geometry.TriangleMesh.create_box(width=x1 - x0, height=0.01,
                                                   depth=z1 - z0)
    mesh.translate([x0, -0.005, z0])
    mesh.paint_uniform_color([0.3, 0.7, 1.0])  # 淡蓝色标出重建的玻璃平面

    o3d.visualization.draw_geometries([pcd, mesh])
```

可视化的预期效果：灰色点云显示真实墙面结构，中间有一块矩形"空洞"，淡蓝色半透明平面正好嵌在空洞位置——这就是重建出来的玻璃。

## 实验

### 数据集说明

论文作者自建了 9 个建筑级场景，涵盖不同玻璃结构（窗户、幕墙、玻璃门）、不同空间尺度和光照条件，用真实机器人采集了超过 1 小时、2.1 公里的轨迹数据。这类数据采集门槛较高——需要同步标定的 LiDAR + 相机 + 精确位姿的机器人平台，普通开发者很难复现同等规模的评测，建议直接参考项目页面上公开的场景说明。

### 定量评估

| 方法 | 全景输入覆盖率 | 针孔输入覆盖率 | 每帧误检体素 |
|-----|-------------|-------------|------------|
| GlassGuard | 85% | 82% | 基准 |
| 最优 baseline | — | ≤61% | 5–17倍于GlassGuard |

这两个数字共同说明了"双重约束"设计的效果：覆盖率没有因为加入了严格的几何校验而大幅下降，同时误检体素（也就是自由空间污染）大幅减少。单独追求高覆盖率而不校验几何一致性的方法，很容易用大量误检换来虚高的覆盖率。

### 定性结果

论文展示了在导航规划器中的实际效果：重建出的玻璃平面正确地挡住了试图穿过玻璃幕墙的路径，同时相邻的真实门洞、走廊依然保持可通行——这正是"覆盖率"和"不污染自由空间"两个指标同时达标的直观体现。

## 工程实践

### 实际部署考虑

- **实时性**：视觉基础模型分割 + 跨帧单应校验都不是轻量操作，论文聚焦于**建图阶段**的离线/半在线精度，不是逐帧实时避障的直接替代，实际部署建议把 GlassGuard 当作建图/重建模块，输出结果喂给下游的实时局部规划器。
- **硬件需求**：视觉基础模型分割通常需要 GPU；单应校验和体素融合是 CPU 友好的几何运算，可以放在嵌入式端。
- **多传感器标定依赖**：整个方法的前提是 LiDAR-相机外参标定准确，以及相机间相对位姿（视觉里程计/SLAM 输出）可靠。标定误差会直接污染单应矩阵，进而让校验环节产生系统性偏差。

### 数据采集建议

- 尽量保证相邻帧之间有足够的视角变化（baseline），否则单应矩阵对平面参数的区分度会很低——视角几乎不变时，任何平面假设 warp 出来的结果都差不多，校验失去意义。
- 玻璃区域尽量保证多帧共视，单帧看到的玻璃掉脚无法做跨视角校验。

### 常见坑

1. **平面法向量符号不一致导致校验全部失败** → 统一约定法向量指向相机一侧，并在计算 $H$ 前检查 $n^T t_{21}$ 的符号：

```python
if np.dot(n, t21) < 0:
    n = -n
    d = -d
```

2. **玻璃掉脚边缘噪声导致 IoU 被严重低估** → 在比较前对掉脚做轻微膨胀：

```python
kernel = np.ones((5, 5), np.uint8)
mask1 = cv2.dilate(mask1.astype(np.uint8), kernel)
mask2 = cv2.dilate(mask2.astype(np.uint8), kernel)
```

3. **全景相机直接套用针孔单应公式会出错** → 全景图像不满足针孔投影模型，需要先把感兴趣区域重投影到局部透视子视图，再应用上述单应校验逻辑。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 建筑室内，玻璃为规则平面（窗、幕墙、玻璃门） | 曲面玻璃、异形装饰玻璃 |
| LiDAR + 多视角相机，位姿可靠 | 仅单目无里程计、无法获取跨帧相对位姿 |
| 玻璃区域有足够共视帧数 | 玻璃一次性扫过、共视帧极少 |
| 静态建筑结构建图 | 玻璃后方场景剧烈动态变化 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| NeRF / 3DGS | 能学习玻璃的视觉外观（反射、透射混合） | 不直接输出可用于避障的几何边界 | 视觉渲染、数字孪生展示 |
| 传统占据栅格 SLAM | 简单高效，实时性好 | 对玻璃完全失明，直接造成安全隐患 | 无玻璃或玻璃占比很低的场景 |
| GlassGuard | 显式建模玻璃平面，兼顾覆盖率与污染控制 | 依赖精确跨帧位姿标定，非逐帧实时 | 建筑级室内导航建图 |

## 我的观点

GlassGuard 的价值不在于提出了一个全新的感知模型，而在于**把"玻璃检测"这个长期被当作纯视觉问题的任务，重新表述成一个几何验证问题**——用平面诱导单应这种经典多视图几何工具，去校验视觉基础模型给出的 2D 线索是否几何自洽。这种"用老几何工具约束新感知模型输出"的思路，在很多 3D 感知任务里都值得借鉴，不只是玻璃检测。

离真正的产品化落地还有几个现实距离：对精确跨帧位姿和外参标定的依赖，在真实机器人长时间运行中位姿漂移、标定老化都会直接削弱校验环节的可靠性；此外论文的评测场景仍然集中在建筑级规则结构，对于玻璃碎片、曲面玻璃幕墙等更复杂的真实世界形态，覆盖率能否保持还有待验证。但对于当下大量部署在办公楼、商场的服务机器人来说，"玻璃幕墙导致误判为可通行空间"是一个真实存在且危险的问题，这类专门针对性的解决方案在工程上是有迫切需求的。

项目主页：https://glassguardproject.github.io/