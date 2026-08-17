---
layout: post-wide
title: "CORAL：课程学习驱动的 LiDAR 城市自动驾驶强化学习方法"
date: 2026-08-17 08:03:52 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.14332v1
generated_by: Claude Code CLI
---

## 一句话总结

CORAL 通过五阶段课程学习与阶段感知奖励权重的组合，让基于 99 维 LiDAR 向量的 PPO 策略在 CARLA 城市驾驶中达到 100% 到达率，而固定奖励的 PPO 基线只有 5-10%。

## 为什么这个问题重要？

城市自动驾驶是一个**多目标同时竞争**的难题。智能体需要同时满足：

- 到达远处目标（长程规划）
- 沿规划路线行驶（不随意抄近道）
- 避开其他车辆和行人
- 遵守交通信号和限速
- 保持驾驶平顺性

经典 RL 做法是加权求和奖励函数：

$$
r = w_1 r_{\text{goal}} + w_2 r_{\text{route}} + w_3 r_{\text{safety}} + w_4 r_{\text{smooth}} + w_5 r_{\text{rule}}
$$

问题在于训练初期：智能体连基本导航都不会，就要同时满足所有约束，往往陷入"原地不动最安全"的局部最优。**CORAL 的洞察是：人类学开车也是先学直线行驶、再学路口、最后上市区拥堵路段，课程顺序至关重要。**

## 背景知识

### 状态空间设计（99 维）

CORAL 放弃了主流的鸟瞰图（BEV）或点云编码器，使用紧凑的结构化向量：

| 模块 | 维度 | 内容 |
|------|------|------|
| 极坐标 LiDAR 直方图 | 36 | 各方向最近障碍距离（归一化） |
| 车辆状态 | 7 | 速度、航向角、转向、油门、刹车等 |
| 自车坐标系路点 | 48 | 未来若干路点的相对位置 |
| 交通规则指示 | 8 | 红绿灯状态、限速、停车线等 |

极坐标直方图的压缩逻辑：将 360° 分成 36 个角度桶，每桶记录该方向最近障碍物的距离，将原始 N×3 点云压缩为 36 维向量，推理延迟降至 ~1ms（CPU 可运行）。

### 多流 Actor-Critic 架构

不同来源的状态用分离编码器处理，再融合进共享主干：

```
LiDAR stream  (36维) → MLP → 32维 \
vehicle stream (7维) → MLP → 16维  \
route stream  (48维) → MLP → 32维  → concat(88维) → trunk → Actor (2维动作)
rule stream    (8维) → MLP →  8维  /                       → Critic (状态值)
```

## 核心方法

### 直觉解释：两个日程表同步推进

CORAL 的精髓在于**任务难度**和**优化目标**同步演进：

```
难度渐进：  ─── Stage1 ─── Stage2 ─── Stage3 ─── Stage4 ─── Stage5 ───►
             20-30m路线   40-60m     60-90m     90-120m    100-150m
             无其他车辆   稀疏流量   中等流量   较密流量   完整流量

奖励侧重：  [目标进度为主] ───────────────────────────► [安全合规为主]
```

初期阶段鼓励"向目标前进"，随着策略成熟，奖励权重逐渐向"安全合规"倾斜。

### 阶段感知奖励函数

$$
r_t = \sum_{k \in \{goal, route, safety, smooth, rule\}} w_k^{(s)} \cdot r_k^t
$$

其中 $s$ 是当前课程阶段，$w_k^{(s)}$ 随阶段变化：

| 阶段 | $w_{goal}$ | $w_{route}$ | $w_{safety}$ | $w_{smooth}$ | $w_{rule}$ |
|------|-----------|------------|------------|------------|----------|
| 1 | 1.0 | 0.2 | 0.5 | 0.1 | 0.0 |
| 3 | 0.5 | 0.8 | 1.0 | 0.4 | 0.3 |
| 5 | 0.2 | 0.7 | 1.0 | 0.7 | 1.0 |

## 实现

### LiDAR 极坐标直方图

```python
import numpy as np

def lidar_to_polar_histogram(point_cloud: np.ndarray,
                              n_bins: int = 36,
                              max_range: float = 50.0) -> np.ndarray:
    """将点云压缩为极坐标直方图，每个角度桶取最近距离"""
    x, y = point_cloud[:, 0], point_cloud[:, 1]
    angles = np.arctan2(y, x)           # [-π, π]
    dists  = np.sqrt(x**2 + y**2)

    mask = dists < max_range
    angles, dists = angles[mask], dists[mask]

    histogram = np.full(n_bins, max_range, dtype=np.float32)
    bin_width = 2 * np.pi / n_bins
    indices = ((angles + np.pi) / bin_width).astype(int) % n_bins

    for idx, d in zip(indices, dists):
        if d < histogram[idx]:
            histogram[idx] = d

    return histogram / max_range   # 归一化到 [0, 1]
```

### 多流 Actor-Critic 网络

```python
import torch
import torch.nn as nn

class MultiStreamActorCritic(nn.Module):
    def __init__(self, lidar_dim=36, vehicle_dim=7, route_dim=48, rule_dim=8):
        super().__init__()
        self.lidar_enc   = nn.Sequential(nn.Linear(lidar_dim, 64),   nn.ReLU(), nn.Linear(64, 32))
        self.vehicle_enc = nn.Sequential(nn.Linear(vehicle_dim, 32), nn.ReLU(), nn.Linear(32, 16))
        self.route_enc   = nn.Sequential(nn.Linear(route_dim, 64),   nn.ReLU(), nn.Linear(64, 32))
        self.rule_enc    = nn.Sequential(nn.Linear(rule_dim, 16),    nn.ReLU(), nn.Linear(16, 8))

        trunk_dim = 32 + 16 + 32 + 8   # = 88
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )
        self.actor_mean   = nn.Linear(64, 2)         # 转向 + 纵向控制
        self.actor_logstd = nn.Parameter(torch.zeros(2))
        self.critic       = nn.Linear(64, 1)

    def forward(self, lidar, vehicle, route, rule):
        h = self.trunk(torch.cat([
            self.lidar_enc(lidar), self.vehicle_enc(vehicle),
            self.route_enc(route), self.rule_enc(rule),
        ], dim=-1))
        mean  = torch.tanh(self.actor_mean(h))       # 动作范围 [-1, 1]
        std   = self.actor_logstd.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def get_action(self, obs_dict):
        mean, std, value = self.forward(**obs_dict)
        dist     = torch.distributions.Normal(mean, std)
        action   = dist.sample().clamp(-1, 1)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value
```

### 阶段感知奖励

```python
STAGE_WEIGHTS = {
    1: [1.0, 0.2, 0.5, 0.1, 0.0],
    2: [0.8, 0.5, 0.8, 0.2, 0.1],
    3: [0.5, 0.8, 1.0, 0.4, 0.3],
    4: [0.3, 0.8, 1.0, 0.6, 0.6],
    5: [0.2, 0.7, 1.0, 0.7, 1.0],
}

def compute_reward(info: dict, stage: int) -> float:
    w = STAGE_WEIGHTS[stage]
    r_goal   =  info["dist_reduction"]
    r_route  = -abs(info["lateral_error"])
    r_safety = -100.0 if info["collision"] else 0.0
    r_smooth = -(abs(info["jerk"]) + abs(info["steer_rate"]))
    r_rule   = (-5.0 * info["red_light_violated"]
                - 2.0 * max(0, info["speed"] - info["speed_limit"]))

    weighted = (w[0]*r_goal + w[1]*r_route
                + w[2]*(r_safety*0.01) + w[3]*r_smooth + w[4]*r_rule)
    return weighted + r_safety   # 碰撞惩罚不受权重缩放
```

### 课程调度器

```python
class CurriculumScheduler:
    def __init__(self, success_threshold=0.8, window=20):
        self.stage     = 1
        self.threshold = success_threshold
        self.history   = []
        self.window    = window

    def record(self, success: bool):
        self.history.append(float(success))
        if len(self.history) > self.window:
            self.history.pop(0)

    def maybe_advance(self) -> bool:
        if self.stage >= 5:
            return False
        if len(self.history) >= self.window and np.mean(self.history) >= self.threshold:
            self.stage += 1
            self.history.clear()
            print(f"[Curriculum] Advanced to Stage {self.stage}")
            return True
        return False

    def get_route_config(self) -> dict:
        configs = {
            1: {"min_len": 20,  "max_len": 30,  "traffic_density": 0.0},
            2: {"min_len": 40,  "max_len": 60,  "traffic_density": 0.2},
            3: {"min_len": 60,  "max_len": 90,  "traffic_density": 0.5},
            4: {"min_len": 90,  "max_len": 120, "traffic_density": 0.8},
            5: {"min_len": 100, "max_len": 150, "traffic_density": 1.0},
        }
        return configs[self.stage]
```

## 实验结果

### 定量评估（CARLA Town01，100-150m 路线）

| 方法 | 成功率 | 路线完成率 | 碰撞率 | 闯红灯率 |
|------|--------|-----------|--------|---------|
| PPO Baseline-A | 5% | 38% | 42% | 21% |
| PPO Baseline-B | 10% | 51% | 31% | 18% |
| CORAL（无课程） | 55% | 71% | 18% | 9% |
| CORAL（固定权重）| 63% | 74% | 15% | 11% |
| **CORAL（完整）** | **100%** | **97%** | **3%** | **1%** |

消融实验清楚表明：课程和动态奖励**缺一不可**，单独移除任一组件都导致显著退化。

### 零样本迁移（训练于 Town01，测试于 7 个未见城镇）

| 城镇 | 成功率 | 平均横向偏差 |
|------|--------|------------|
| Town02 | 98% | 0.21m |
| Town03 | 87% | 0.28m |
| Town04 | 82% | 0.31m |
| Town05 | 68% | 0.34m |
| Town06-07 | 78-91% | 0.27-0.33m |

Town05 最差，因为其道路拓扑包含多车道高速公路段，超出训练分布。

## 工程实践

### 实时性对比

| 组件 | CORAL | BEV-based 方法 |
|------|-------|--------------|
| 输入处理 | 99 维向量 | 256×256×C 图像 |
| 推理延迟 | ~1ms（CPU） | ~30-50ms（GPU） |
| 内存占用 | <10MB | >200MB |
| 可运行平台 | Jetson Nano | RTX 3090+ |

在嵌入式自动驾驶平台上，99 维向量的计算优势压倒性。

### 常见坑与解决方案

**坑1：奖励稀疏导致早期训练停滞**

```python
# 错误：只有到达目标才给奖励
reward = 100.0 if reached_goal else 0.0

# 正确：每步给予距离减少量作为密集奖励
reward = prev_dist_to_goal - curr_dist_to_goal
```

**坑2：课程晋级过快，策略未泛化就进入下一阶段**

```python
# 错误：只看单个 episode 结果
if last_episode_success:
    advance_stage()

# 正确：滑动窗口均值，窗口至少 20 个 episode
if len(history) >= 20 and np.mean(history[-20:]) > 0.8:
    advance_stage()
```

**坑3：LiDAR 直方图丢失垂直信息**

极坐标直方图无法区分"路边石"和"旁边的车"，对于需要精确几何感知的场景，需要分层处理：

```python
# 按高度分三层，各生成独立直方图，总维度 36×3 = 108
ground  = lidar_to_polar_histogram(pc[pc[:, 2] < 0.3])
body    = lidar_to_polar_histogram(pc[(pc[:, 2] >= 0.3) & (pc[:, 2] < 1.8)])
rooftop = lidar_to_polar_histogram(pc[pc[:, 2] >= 1.8])
lidar_feat = np.concatenate([ground, body, rooftop])
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 算力受限，需要实时推理（嵌入式平台） | 需要精细感知（行人意图、手势识别） |
| 中短距离（<200m）结构化道路导航 | 超长路线、复杂拓扑（多层立交桥） |
| RL 驾驶策略快速原型验证 | 需要可解释性与安全认证的生产系统 |
| 需要零样本迁移到新地图 | 非结构化环境（越野、施工区域） |

## 与其他方法对比

| 方法 | 感知输入 | 计算开销 | 可迁移性 | 核心局限 |
|------|---------|---------|---------|---------|
| 纯规划（HD Map） | 高精地图 | 中 | 低（需建图） | 地图维护成本高 |
| 端到端 CNN（BEV） | 图像/LiDAR | 高 | 中 | 数据量要求大 |
| NeRF/3DGS + 规划 | RGB | 极高 | 极低（需重建） | 动态场景差 |
| **CORAL** | LiDAR 向量 | **极低** | **高（零样本）** | 感知表达能力有限 |

## 我的观点

CORAL 最值得借鉴的不是具体数字，而是**课程学习 + 动态奖励**这个组合拳的设计哲学——它直接对应了一个基础问题：多目标任务应该按什么顺序学习？

**正向评价**：99 维向量输入是当前 RL 驾驶研究中被低估的方向。当学术界热衷于更大的 Transformer、更高分辨率的 BEV，CORAL 用极轻量的表示达到了不俗的效果，嵌入式平台适配性值得认真对待。

**清醒评估**：100% 成功率建立在几个局限之上——CARLA 传感器噪声模型与真实 LiDAR 存在 sim-to-real gap；100-150m 路线在实际场景中只算"停车场导航"级别；7 个迁移城镇仍在同一模拟器内，未经过真实世界验证。

**开放问题**：这套课程框架如何扩展到更长路线（公里级）？动态奖励权重能否通过元学习自动发现，而非人工设计？这些是值得跟进的方向。

对于在做机器人导航、无人机避障、仓储 AGV 调度的读者：课程学习框架本身具有高度可迁移性，奖励权重随任务复杂度动态调整的思路可以直接复用，与具体的传感器模态无关。