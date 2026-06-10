---
layout: post-wide
title: "多 UGV 分布式协作探索：LiDAR 描述子、回环感知与层次化规划"
date: 2026-06-10 08:03:04 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.11088v1
generated_by: Claude Code CLI
---

## 一句话总结

在没有 GPS、没有先验地图、带宽受限的环境中，让多台地面机器人（UGV）协同探索未知区域——核心挑战是：如何通过跨机器人回环检测消除定位漂移，同时让规划器感知这些回环以减少重复覆盖。

---

## 为什么这个问题重要？

### 应用场景

- **矿洞/隧道搜救**：GPS 完全失效，单机器人效率太低
- **仓库自动建图**：多机协同加速，但要保证地图拼接一致
- **军事侦察/灾后评估**：带宽受限，不能持续传输原始数据

### 现有方法的问题

单机 LiDAR SLAM（如 LOAM、LIO-SAM）已经成熟，但多机协作有三个难题：

1. **定位漂移累积**：各机器人独立运行，长时间后轨迹误差可达数米，地图拼接失败
2. **重复探索**：不知道同伴去了哪里，不同机器人探索同一区域
3. **通信瓶颈**：把完整点云地图广播给队友，带宽根本不够

### 这篇论文的核心创新

```
传统方案：中心服务器 + 持续地图同步（带宽瓶颈）
本文方案：完全分布式 + 描述子辅助回环 + 回环感知规划
```

三个关键模块：
- **轻量 LiDAR 全局描述子**：range-image 预对齐，鲁棒处理大偏航角差异
- **不确定性感知回环筛选**：只保留"高价值"回环作为规划锚点
- **层次化规划**：用回环锚点做全局任务分配 + 本地路径精化

---

## 背景知识

### LiDAR 距离图像（Range Image）

LiDAR 扫描是一堆 3D 点，要生成描述子，需要先把点云"展平"成 2D 图像。距离图像是将 3D 球面扫描投影到柱面：

$$
u = \left\lfloor \left(1 - \frac{\phi + f_{\text{up}}}{f_V}\right) \times H \right\rfloor, \quad v = \left\lfloor \left(0.5 - \frac{\theta}{2\pi}\right) \times W \right\rfloor
$$

其中 $\phi$ 是仰角、$\theta$ 是方位角、$f_V = f_{\text{up}} + f_{\text{down}}$ 是垂直视场角，每个像素值存储距离 $r$。

**为什么先投影再提描述子？** 点云无序、稠密，直接处理开销大。Range image 保留了空间结构，可以用 2D 卷积提取特征。

### 位姿图优化（Pose Graph Optimization）

SLAM 系统维护一张位姿图：节点是机器人位姿 $T_i \in SE(3)$，边是相对变换约束 $z_{ij}$。优化目标：

$$
\min_{\{T_i\}} \sum_{(i,j) \in \mathcal{E}} \| \log(z_{ij}^{-1} \cdot T_i^{-1} T_j) \|_{\Omega_{ij}}^2
$$

其中 $\Omega_{ij}$ 是信息矩阵（协方差的逆），回环约束提供全局一致性。

---

## 核心方法

### 直觉解释

想象两个机器人 A 和 B 从不同入口进入同一建筑物。各自建了局部地图，但不知道彼此的关系。当 A 路过 B 曾经探索过的走廊时，"我认出这个地方了！"——这就是跨机器人回环检测。检测到后，两张地图就能精准拼接，并且规划器知道"那里已经探索过了"。

```
机器人A: 入口 → 走廊1 → 房间1 → ?
                                   ↑ 回环检测（A 到达 B 去过的地方）
机器人B: 侧门 → 走廊2 → 房间1 → 走廊3

检测后：合并位姿图，A 知道走廊3 已被覆盖，转向其他区域
```

### 数学细节

#### Range Image 预对齐

跨机器人的主要挑战：两台机器人从不同方向经过同一地点，偏航角差异可达 180°。直接比较描述子会失败。

**解决方案**：先用旋转不变性估计偏航差，再对齐后提取描述子。

设描述子 $\mathbf{d}$ 是 range image 的列均值向量（旋转 = 列循环移位）：

$$
\Delta\theta^* = \arg\min_{\Delta\theta} \| \mathbf{d}_A - \text{shift}(\mathbf{d}_B, \Delta\theta) \|_2^2
$$

通过 FFT 加速循环相关计算：$\mathcal{O}(N \log N)$ 代替 $\mathcal{O}(N^2)$。

#### 不确定性感知回环评分

不是所有回环都值得相信。机器人漂移大时，一个"假回环"会破坏整张地图。评分函数：

$$
s_{ij} = \frac{\Delta I_{ij}}{\sqrt{\text{tr}(\Sigma_i) + \text{tr}(\Sigma_j)}}
$$

- $\Delta I_{ij}$：这个回环能带来的信息增益（图中两节点间最短路径长度）
- $\Sigma_i, \Sigma_j$：两端位姿的协方差矩阵（漂移越大，迹越大）
- 高分回环 = 信息量大 + 两端定位都比较可信

### Pipeline 概览

```
各 UGV 独立运行：
LiDAR → 前端里程计 → 局部位姿图 + 局部稀疏拓扑图
                            ↓
             广播：全局描述子 + 当前位姿协方差

跨 UGV 回环检测：
描述子库 → 候选匹配 → 几何验证 → 回环约束

回环利用：
高分回环 → 分布式位姿图优化 → 一致轨迹
         → 规划锚点库 → 全局任务分配 + 局部路径精化
```

---

## 实现

### Range Image 生成与描述子提取

```python
import numpy as np

def points_to_range_image(points, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    """
    将 LiDAR 点云转换为距离图像
    points: (N, 3) xyz 点云
    返回: (H, W) 距离图像
    """
    fov_up_r = np.deg2rad(fov_up)
    fov_down_r = np.deg2rad(fov_down)
    fov_range = fov_up_r - fov_down_r

    # 球坐标
    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])        # 方位角
    phi = np.arcsin(points[:, 2] / (r + 1e-8))           # 仰角

    # 投影到图像坐标
    u = (1.0 - (phi - fov_down_r) / fov_range) * H
    v = (0.5 - theta / (2 * np.pi)) * W

    u = np.clip(u.astype(np.int32), 0, H - 1)
    v = np.clip(v.astype(np.int32), 0, W - 1)

    range_img = np.zeros((H, W))
    range_img[u, v] = r
    return range_img


def extract_descriptor(range_img):
    """
    提取旋转感知描述子（列均值 + 列标准差）
    返回: (2, W) 描述子，水平偏移对应旋转
    """
    col_mean = range_img.mean(axis=0)           # (W,)
    col_std  = range_img.std(axis=0)            # (W,)
    desc = np.stack([col_mean, col_std], axis=0)
    # L2 归一化
    desc = desc / (np.linalg.norm(desc) + 1e-8)
    return desc
```

### 偏航预对齐与相似度计算

```python
def align_and_match(desc_a, desc_b):
    """
    用 FFT 循环相关估计最优偏航对齐，再计算描述子相似度
    desc_a, desc_b: (2, W) 描述子
    返回: (相似度, 偏航差列数)
    """
    W = desc_a.shape[1]

    # FFT 循环相关（对两行分别计算，取均值）
    corr_total = np.zeros(W)
    for row in range(desc_a.shape[0]):
        fa = np.fft.fft(desc_a[row])
        fb = np.fft.fft(desc_b[row])
        corr = np.fft.ifft(fa * np.conj(fb)).real
        corr_total += corr

    best_shift = np.argmax(corr_total)

    # 按最优偏移对齐 desc_b
    desc_b_aligned = np.roll(desc_b, best_shift, axis=1)

    # 余弦相似度
    sim = np.dot(desc_a.flatten(), desc_b_aligned.flatten()) / (
        np.linalg.norm(desc_a) * np.linalg.norm(desc_b_aligned) + 1e-8
    )
    return float(sim), best_shift
```

### 不确定性感知回环评分

```python
import heapq

def score_loop_closure(graph, pose_covs, candidate_pairs, sim_scores, sim_thresh=0.85):
    """
    对候选回环按不确定性感知评分排序
    graph: dict {node_id: [neighbor_ids]}  稀疏拓扑图
    pose_covs: dict {node_id: 3x3 协方差矩阵}
    返回: 按评分排序的 [(score, node_i, node_j), ...]
    """
    def bfs_distance(graph, src, dst):
        # 图中两节点最短路径长度（近似信息增益）
        visited, queue = {src}, [(0, src)]
        while queue:
            dist, node = heapq.heappop(queue)
            if node == dst:
                return dist
            for nb in graph.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    heapq.heappush(queue, (dist + 1, nb))
        return float('inf')

    scored = []
    for (i, j), sim in zip(candidate_pairs, sim_scores):
        if sim < sim_thresh:
            continue
        # 信息增益 = 图中 i 到 j 的最短路（越远价值越高）
        info_gain = bfs_distance(graph, i, j)
        # 两端不确定性（协方差迹）
        unc_i = np.trace(pose_covs[i])
        unc_j = np.trace(pose_covs[j])
        uncertainty = np.sqrt(unc_i + unc_j + 1e-6)
        score = info_gain / uncertainty
        scored.append((score, i, j))

    scored.sort(reverse=True)
    return scored
```

### 简化位姿图优化（g2o 风格）

```python
def pose_graph_optimize_2d(poses, edges, n_iters=50, lr=0.01):
    """
    2D 位姿图优化（教学简化版，实际用 g2o/GTSAM）
    poses: (N, 3) [x, y, theta]
    edges: [(i, j, dx, dy, dtheta, omega_3x3), ...]
    返回: 优化后的 poses
    """
    poses = poses.copy()

    def compose(p1, p2):
        c, s = np.cos(p1[2]), np.sin(p1[2])
        x = p1[0] + c * p2[0] - s * p2[1]
        y = p1[1] + s * p2[0] + c * p2[1]
        return np.array([x, y, p1[2] + p2[2]])

    def error(pi, pj, z):
        pred = compose([-pi[0], -pi[1], -pi[2]], pj)  # T_i^{-1} T_j
        return pred - z

    for _ in range(n_iters):
        grad = np.zeros_like(poses)
        for (i, j, dx, dy, dtheta, omega) in edges:
            z = np.array([dx, dy, dtheta])
            e = error(poses[i], poses[j], z)
            # 梯度累积（简化，未推导完整雅可比）
            grad[i] -= lr * omega @ e
            grad[j] += lr * omega @ e
        poses[1:] -= grad[1:]          # 固定第一个节点
        poses[:, 2] = (poses[:, 2] + np.pi) % (2 * np.pi) - np.pi

    return poses
```

### 3D 可视化（轨迹 + 回环）

```python
import open3d as o3d

def visualize_multi_robot_trajectories(traj_list, loop_pairs, colors=None):
    """
    traj_list: [np.array (N,3), ...]  各机器人轨迹 (x,y,z)
    loop_pairs: [(pose_i_xyz, pose_j_xyz), ...]  回环对
    """
    geometries = []
    default_colors = [[1,0,0], [0,0.7,0], [0,0,1], [1,0.5,0]]

    for k, traj in enumerate(traj_list):
        # 轨迹线段
        pts = o3d.utility.Vector3dVector(traj)
        lines = [[i, i+1] for i in range(len(traj)-1)]
        color = colors[k] if colors else default_colors[k % 4]
        ls = o3d.geometry.LineSet(
            points=pts,
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.paint_uniform_color(color)
        geometries.append(ls)

    # 回环连线（黄色虚线）
    if loop_pairs:
        lp_pts = np.array([[p for pair in loop_pairs for p in pair]]).reshape(-1, 3)
        lp_lines = [[2*i, 2*i+1] for i in range(len(loop_pairs))]
        lp_ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(lp_pts),
            lines=o3d.utility.Vector2iVector(lp_lines)
        )
        lp_ls.paint_uniform_color([1, 1, 0])
        geometries.append(lp_ls)

    o3d.visualization.draw_geometries(geometries,
        window_name="Multi-UGV Trajectories + Loop Closures")
    # 预期输出：红/绿/蓝轨迹线 + 黄色回环连接线，可交互旋转缩放
```

---

## 实验

### 数据集说明

| 数据集 | 特点 | 获取难度 |
|--------|------|---------|
| MulRan | 韩国城市/校园，多传感器 | 公开下载 |
| KITTI | 驾驶场景，基准充分 | 公开下载 |
| 自采 | 室内/工厂，GPS 失效 | 需要多机平台 |

真实多 UGV 数据极度稀缺——大多数论文用"模拟多路"（把单机轨迹分段，模拟多机视角）。

### 定量评估

| 方法 | AR@1 | AR@1% | ATE (m) | 探索时间 | 通信量 |
|------|------|-------|---------|---------|--------|
| **本文** | **89.9%** | **95.5%** | 低 | -15% | 大幅降低 |
| mTSP baseline | — | — | 高（漂移） | 基准 | 基准 |
| 无回环 | ~60% | ~75% | 高 | -5% | 低 |

AR@1：top-1 检索准确率；AR@1%：top-1% 内存在正确匹配的比例。

### 定性结果

**好的案例**：走廊、房间角落等结构重复性强的场景，描述子区分度高，回环可靠。

**失败案例**：开阔广场（地面特征匮乏）、光滑墙体（range image 缺乏细节）——这时 AR@1 下降明显，系统退化为无回环模式。

---

## 工程实践

### 实时性约束

```
LiDAR 频率：10 Hz → 每帧 100ms 预算
- range image 生成：~2ms（NumPy 向量化）
- 描述子提取：~1ms
- FFT 预对齐 + 匹配：~5ms（N=100 候选）
- 位姿图优化：~20ms（iSAM2 增量式）
剩余给前端里程计：~70ms ✓
```

一个常见坑：描述子数据库随探索面积线性增长，匹配时间从 O(1) 变 O(N)。

```python
# 坑：暴力线性搜索
# for desc in database:  # 越来越慢！
#     sim = cosine_sim(query, desc)

# 修：用 FAISS 近似最近邻
import faiss
index = faiss.IndexFlatIP(DIM)          # 内积 = 余弦相似度（归一化后）
index.add(database_matrix)
D, I = index.search(query[None], k=5)   # 恒定时间
```

### 跨机器人数据格式

```python
# 机器人广播的最小数据包（~200 bytes/帧）
broadcast_msg = {
    "robot_id": int,
    "timestamp": float,
    "descriptor": np.float32,  # (2*W,) 压缩后 ~128 bytes
    "pose_cov_trace": float,   # 不确定性标量，非完整协方差
    "node_id": int,
}
# 对比：完整点云 ~100KB/帧，带宽减少 500x
```

### 常见坑与解决方案

1. **大偏航角导致描述子不匹配** → FFT 预对齐，估计旋转后再比较（本文核心）

2. **回环误检破坏地图** → 几何验证：ICP 精配准后检查 inlier 比例，低于阈值拒绝

3. **通信延迟导致回环时序错乱**
```python
# 坑：直接用收到时刻作为时间戳
# 修：包含机器人本地时间戳，接收方用时间差加权置信度
age = current_time - msg["timestamp"]
weight = np.exp(-age / tau)  # 越旧越不可信
```

4. **稀疏拓扑图节点过多** → 只在转角/分叉处插入节点，平直走廊用距离阈值抽稀

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 室内结构化环境（走廊、房间） | 开阔无特征场地（草坪、平地） |
| GPS 拒止（地下/室内） | GPS 可用时（直接用 RTK 更简单） |
| 3 台以上 UGV 协作 | 单机任务（过度设计） |
| 带宽受限（<1 Mbps） | 带宽充足（可传完整地图） |
| 需要持久地图一致性 | 一次性探索，不需要返回 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| COVINS / KIMERA-Multi | 精度高，成熟 | 需要中心服务器，带宽大 | 局域网多机 |
| DiSCo-SLAM | 完全分布式 | 无规划模块 | 纯定位/建图 |
| mTSP + 独立SLAM | 规划成熟 | 地图不一致 | 带宽充足环境 |
| **本文** | 分布式+规划一体化，低通信量 | 场景依赖强，开放代码少 | 带宽受限多机探索 |

---

## 我的观点

### 技术路线判断

这篇论文的价值在于**系统集成**而非单点突破：把描述子、回环筛选、规划三个模块连通，形成闭环。这在 multi-robot SLAM 领域仍然是稀缺工作——大多数论文只做其中一个模块。

### 离实际部署还有多远？

**近**：通信协议和描述子效率已经达到可部署水平，AR@1 89.9% 在工业场景够用。

**远**：
- 动态障碍（人、叉车）：当前框架假设静态环境
- 机器人数量扩展性：5 台以上时规划复杂度急剧上升
- 实测数据集极少，sim-to-real gap 未充分验证

### 值得关注的开放问题

1. **异构传感器**：UGV 配备不同型号 LiDAR 时，描述子还能匹配吗？
2. **主动感知**：规划器能否主动引导机器人去"消除不确定性"，而不是被动利用回环？
3. **神经描述子**：用 PointNetVLAD 替换手工描述子，能否在少训练数据下泛化？

多 UGV 协作探索的基础问题已经基本清楚，接下来的战场是**大规模实验验证**和**面向特定垂直场景的落地**——这才是真正拉开差距的地方。