---
layout: post-wide
title: '给会写代码的机器人加一道"安全带"：Coding Agent 的避障执行框架 SafeHarness'
date: 2026-09-19 08:04:11 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.20822v1
generated_by: Claude Code CLI
---

## 一句话总结

当我们让大语言模型直接"写代码"来控制机械臂完成抓取、推动等操作时，它在绝大多数情况下会撞上任务里明确要求"不能碰"的障碍物——不是因为它没看到障碍物，也不是因为指令不清楚，而是因为它的规划过程里从来没有把"安全"当成一个需要被满足的约束，而只是当成了任务描述里的一句装饰性文字。SafeHarness 通过把"绕障路径规划"和"接触点选择"这两个环节显式地结构化出来，让模型第一次把避障当成硬约束来执行。

## 为什么这个问题重要？

Coding agent 操控机器人是过去一年里很实际的一条路线：不需要针对具体机器人做训练，LLM 直接生成一段调用运动学/抓取 API 的 Python 脚本，机器人执行。这条路线的好处是灵活——只要换个 prompt，同一个模型就能处理抓取、码垛、开抽屉等完全不同的任务。

但这篇论文问了一个此前没人系统问过的问题：**这条路线安全吗？** 在桌面操作场景里，"目标物体旁边放着一个不能碰的杯子/传感器/易碎品"是极其常见的约束。论文构造了一批"目标 + 禁触障碍物"配对任务，发现：

- Agent 在推理轨迹里**能够正确识别障碍物**（感知没问题）；
- Prompt 里也**明确写了"不要碰 X"**（指令没问题）；
- 但执行结果**大概率还是撞上了**（规划出了问题）。

论文把失败定位在两个具体阶段：**走位阶段（route）**——模型不知道什么叫"一条不穿过障碍物的路径"，选错了就没有重规划的机制；**接触阶段（contact）**——模型完全没意识到"抓取/接触"这个动作本身也可能触碰障碍物，因为它把约束只理解成"路上不要碰"，而不是"全程不要碰"。

对机器人从业者来说，这个问题的现实意义很直接：如果 coding agent 要从 demo 走向真实部署（家庭、仓库、实验室），"能完成任务"和"不闯祸"必须同时成立，而目前的默认行为是牺牲后者。

## 背景知识

### 3D 表示的取舍：为什么用包围盒，不用点云或 mesh

在传统运动规划（RRT、CHOMP、MoveIt 这类）里，障碍物通常表示成精细的点云、体素栅格或碰撞 mesh，规划器在 C-space 里做数值优化，不需要"理解"障碍物是什么。

但 coding agent 的规划发生在**语言模型的推理过程**里，它没有能力直接对一团点云做几何运算——它需要的是能被"读进 prompt、能被写进代码、能被人和模型同时理解"的表示。SafeHarness 因此选择了最轻量的 3D 表示：**轴对齐包围盒（AABB）**，由物体检测 + 深度估计得到 6 个数字（中心坐标 + 半边长）。这是一种典型的"为 LLM 推理妥协几何精度"的设计：牺牲了对非凸/细长障碍物的精确建模能力，换来了模型可以直接在代码里做 `if box.contains(point)` 这种可读、可验证的判断。

### 关键假设

包围盒表示隐含了一个前提：障碍物的检测和位姿估计足够准，且障碍物在任务执行期间基本静止。这个假设在很多真实场景里是脆弱的（详见后文"常见坑"）。

## 核心方法

### 直觉解释

把一次操作拆成两段：

```
[走位阶段 route]                    [接触阶段 contact]
起点 ──直线试探──> 碰到障碍物包围盒?      抓取/接触前重新选点，
       └─是─> 生成绕行路点 ──> 验证      使接触方向背离障碍物
       └─否─> 直接执行
```

原始 coding agent 只把约束当"背景知识"提一嘴就去规划抓取轨迹了；SafeHarness 则强制模型在写代码前先**显式规划路径、显式验证、必要时重规划**，接触点的选择也被单独拎出来做一次避障判断，而不是默认继承走位阶段的安全性。

### 数学细节

把安全约束显式写成硬约束，而不是柔性目标里的一项：

$$
\pi^* = \arg\max_{\pi} \; \text{success}(\pi) \quad \text{s.t.} \quad \forall t,\; \text{state}(t) \notin \mathcal{C}_{\text{obs}}
$$

其中 $\mathcal{C}_{\text{obs}}$ 是障碍物在配置空间中膨胀安全裕量后的碰撞区域。论文观察到的核心失败模式，本质上是 coding agent 把这个带约束的优化问题，退化成了无约束的：

$$
\pi^* = \arg\max_{\pi} \; \text{success}(\pi)
$$

线段与包围盒的碰撞检测（走位阶段验证的核心判据）可以用 slab 方法表达：路径段 $p(t) = p_0 + t(p_1 - p_0),\ t \in [0,1]$ 与包围盒相交，当且仅当在每个坐标轴上都能找到重叠的参数区间。

### Pipeline 概览

```
物体检测+深度 → 3D 包围盒 → 障碍感知走位规划（生成/验证/重规划路点）
                                    ↓
                          障碍感知接触点选择（选背离障碍物的接触方向）
                                    ↓
                              生成机械臂控制代码并执行
```

## 实现

下面给出一个**教学用的简化实现**，还原论文两个 harness 的核心逻辑：包围盒碰撞检测、走位规划与重规划、接触点选择。这不是论文的官方代码（论文未在摘要中给出仓库地址，我们也未确认其代码是否已开源，因此不提供链接）。

### 环境配置

```bash
pip install numpy open3d
```

### 核心代码：障碍物表示与走位规划

```python
import numpy as np

class AABB:
    """轴对齐包围盒，用于表示障碍物的碰撞体积"""
    def __init__(self, center, half_extent):
        self.center = np.array(center, dtype=float)
        self.half_extent = np.array(half_extent, dtype=float)

    def inflate(self, margin):
        # 按安全裕量膨胀包围盒，避免路径贴着障碍物边缘走
        return AABB(self.center, self.half_extent + margin)

    def contains(self, point):
        return np.all(np.abs(point - self.center) <= self.half_extent)

    def segment_intersects(self, p0, p1, n_samples=20):
        # 简化实现：沿线段采样检测，替代严格的 slab 相交测试
        for t in np.linspace(0, 1, n_samples):
            if self.contains(p0 + t * (p1 - p0)):
                return True
        return False


def plan_route(start, goal, obstacle, margin=0.03, max_replans=3):
    """障碍感知的走位规划：先走直线，遇障碍则绕行并重新验证"""
    safe_box = obstacle.inflate(margin)
    route = [np.array(start), np.array(goal)]

    for _ in range(max_replans):
        collided, new_route = False, [route[0]]
        for p0, p1 in zip(route[:-1], route[1:]):
            if safe_box.segment_intersects(p0, p1):
                mid = (p0 + p1) / 2
                clearance_z = safe_box.center[2] + safe_box.half_extent[2] + 0.05
                new_route.append(np.array([mid[0], mid[1], clearance_z]))
                collided = True
            new_route.append(p1)
        route = new_route
        if not collided:
            return route, True  # 验证通过，直到全程无碰撞才返回
    return route, False  # 达到最大重规划次数仍不安全
```

### 核心代码：接触阶段的避障选点

```python
def select_contact_pose(object_box, obstacle_box, gripper_half_width=0.04):
    """选择接触点：让机械臂从远离障碍物的一侧接近物体，避免接触瞬间碰撞"""
    away_dir = object_box.center - obstacle_box.center
    away_dir[2] = 0  # 仅在水平面上判断接近方向
    away_dir /= np.linalg.norm(away_dir) + 1e-8

    approach_point = object_box.center + away_dir * (
        object_box.half_extent[0] + gripper_half_width)
    contact_point = object_box.center - away_dir * object_box.half_extent[0]
    return approach_point, contact_point
```

这两个函数分别对应论文里的 obstacle-aware route planning 和 obstacle-aware contact execution。关键区别在于：原始 coding agent 只会调用一次抓取 API 生成轨迹；这里的走位规划带有**验证-重规划循环**，接触点选择则**独立**于走位阶段重新做一次避障判断，不假设走位安全就意味着接触安全。

### 3D 可视化

```python
import open3d as o3d

def visualize(obstacle_box, route):
    obs_mesh = o3d.geometry.TriangleMesh.create_box(
        *(2 * obstacle_box.half_extent)).translate(
        obstacle_box.center - obstacle_box.half_extent)
    obs_mesh.paint_uniform_color([0.9, 0.2, 0.2])

    points = [np.array(p) for p in route]
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector([[0, 0.8, 0]] * len(lines))

    o3d.visualization.draw_geometries([obs_mesh, line_set])
```

红色方块是障碍物包围盒，绿色折线是规划出的绕行路径——直线段一旦穿过红色方块，就会被替换成一个抬升的绕行路点，这就是"验证-重规划"在几何上的直观呈现。

## 实验

以下数字来自论文本身报告的结果，不是我们复现得到的：

| 配置 | 任务成功率 | 避障成功率 |
|-----|-----------|-----------|
| 无 harness 的 coding agent | 约为 SafeHarness 的 1/2.3 | 约为 SafeHarness 的 1/1.5 |
| 此前 SOTA | 71.9% − 6.5% | 87.5% − 27.0% |
| SafeHarness | **71.9%** | **87.5%** |

需要注意的是，摘要只给出了绝对提升幅度和倍数关系，没有给出基线的绝对数值，上表中的基线数字是按倍数关系反推的近似值，实际论文正文可能有更精确的表格。

值得关注的两个信号：一是避障成功率的提升幅度（27 个百分点）远大于任务成功率的提升（6.5 个百分点），说明"加约束"本身对任务完成能力的损害并不大——这是好消息，意味着安全和效率在这个设定下不是强烈对立的；二是相对无 harness 基线 2.3 倍/1.5 倍的提升，说明原始 coding agent 的避障能力**不是差一点，而是差一大截**，印证了论文"regulator 缺失"而非"感知缺失"的诊断。

## 工程实践

### 实际部署考虑

- **感知延迟决定重规划频率**：走位阶段的验证-重规划循环依赖障碍物包围盒的实时更新，如果物体检测+深度估计的帧率跟不上机械臂运动速度，重规划会用过时的障碍物位置做判断，等于白做。
- **安全裕量（margin）的选择是个权衡**：裕量太小，检测/位姿噪声会让"验证通过"的路径实际上仍然碰撞；裕量太大，在狭窄工作空间里会导致无解或路径极度迂回。
- **包围盒对非凸物体是保守近似**：L 形支架、开口容器这类物体用 AABB 表示会显著高估碰撞体积，可能导致原本可行的路径被误判为不可行。

### 数据采集建议

这里的"数据"主要是物体检测和位姿估计的质量，而不是训练数据：

- 障碍物检测的召回率比精度更重要——漏检一个障碍物比多算一个安全裕量后果严重得多；
- 深度估计噪声会直接传导到包围盒的 `half_extent` 上，建议对连续几帧做滑动平均再生成包围盒，避免规划器在噪声抖动下反复重规划。

### 常见坑

1. **障碍物在执行过程中被别的物体遮挡，检测丢失** → 丢失期间沿用最后一次可信的包围盒，而不是直接假设障碍物消失。
2. **接触点选择只看物体和障碍物的相对位置，忽略机械臂自身连杆的碰撞** → 需要在 `select_contact_pose` 之外单独做一次机械臂本体的自碰撞/环境碰撞检查，论文的两个 harness 本身不覆盖这一层。
3. **重规划次数设得太少，狭窄空间里频繁"验证失败但放弃"** → 论文的 `max_replans` 需要结合具体工作空间的杂乱程度调参，没有通用最优值。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 桌面级抓取/推动，障碍物静止 | 障碍物本身在运动（如另一台机械臂、传送带上的物体） |
| 障碍物形状接近轴对齐立方体 | 细长、L 形、镂空等强非凸障碍物 |
| 物体检测+深度估计精度较高 | 遮挡严重、反光/透明物体导致检测不稳定的场景 |
| 单一明确禁触区域 | 多个障碍物密集排列、通道极窄的杂乱场景 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 经典运动规划（RRT/CHOMP/MoveIt） | 碰撞检测精确，理论保证强 | 需要精细环境建模，不适合语言指令驱动的任务泛化 | 已知精确环境模型的工业场景 |
| 纯 Prompt 约束的 coding agent | 灵活，任务泛化能力强 | 安全约束容易被"任务完成"目标压过，本文核心发现的问题 | 对安全要求不高的探索性任务 |
| SafeHarness | 在保持语言驱动灵活性的同时显式验证安全约束 | 包围盒表示精度有限，不覆盖机械臂自碰撞 | 目标+禁触障碍物的桌面操作任务 |

## 我的观点

这篇工作的价值不在于提出了多复杂的算法——绕行路点生成和背离方向选点都是运动规划里的基本操作——而在于它**指出了一个容易被忽视的系统性问题**：当我们把规划的"大脑"换成 LLM 之后，那些经典规划器里默认内置的安全约束（碰撞检测是规划的硬性组成部分，不是可选项）在语言模型的推理里并不会自动出现，即使模型在文字上"提到"了约束。这提醒我们，安全性不能靠 prompt 里多加一句"不要碰 X"来兜底，而需要在执行框架层面把约束验证做成结构化、可检查的步骤。

离真实部署还有明显距离：包围盒表示对复杂几何体的保守性、对动态障碍物的完全不支持、以及机械臂自身连杆碰撞未被覆盖，都是把这套框架搬到真实家庭/仓库环境前必须补上的部分。一个值得关注的开放问题是：当障碍物数量增多、约束之间可能冲突（比如两个禁触区域之间只剩一条窄缝）时，"验证-重规划"这种局部修补式的方法是否还够用，还是需要引入更接近经典采样式规划器的全局搜索。