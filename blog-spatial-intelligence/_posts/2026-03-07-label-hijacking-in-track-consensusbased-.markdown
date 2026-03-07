---
layout: post-wide
title: "分布式多目标跟踪中的标签劫持：当传感器网络的身份机制被欺骗"
date: 2026-03-07 12:04:31 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.05023v1
generated_by: Claude Code CLI
---

## 一句话总结

当多传感器通过轨迹共识协议协作跟踪目标时，攻击者可以注入精心设计的伪造轨迹，让整个网络混淆目标身份——这就是**标签劫持（Label Hijacking）**。

## 为什么这个问题重要？

分布式多目标跟踪（DMTT）在以下场景中至关重要：

- **军事监控**：多雷达协同跟踪飞行器编队
- **自动驾驶**：路侧传感器与车载传感器协同感知
- **无人机管控**：空域多架无人机的身份维护
- **海上监视**：多雷达站协同跟踪船只

这些场景有一个共同特点：**目标的身份（标签/ID）比位置更重要**。知道"3号目标"在某处和知道"某个未知物体"在某处，信息价值天壤之别。

传统 TC-DMTT 方法假设所有参与节点都是诚实的——但现实中，攻击者可以向网络注入伪造轨迹，利用共识协议的匹配逻辑本身来发动攻击。

## 背景知识

### 分布式多目标跟踪

每个传感器视野（FoV）有限，无法覆盖整个场景，多传感器需要协同：

```
传感器A (FoV左半区)            传感器B (FoV右半区)
    |                                  |
目标1 [x=10, label=A1]          目标1 [x=10, label=B3]  ← 同一目标，不同ID！
目标2 [x=20, label=A2]          目标3 [x=40, label=B1]
```

每个节点独立运行 Kalman 滤波器，维护本地轨迹库。不同节点对同一目标可能分配不同标签，这是**标签不一致问题**。

### 轨迹共识（TC-DMTT）

核心思想：通过度量匹配，强制各节点对同一目标使用相同标签。

```
各节点本地轨迹 → 广播轨迹状态 → 计算代价矩阵 → 匈牙利匹配 → 标签同步
```

代价矩阵基于马氏距离（Mahalanobis Distance）：

$$d_{ij} = (\mathbf{x}_i - \mathbf{x}_j)^\top (S_i + S_j)^{-1} (\mathbf{x}_i - \mathbf{x}_j)$$

其中 $\mathbf{x} = [x, y, \dot{x}, \dot{y}]^\top$ 是轨迹运动状态，$S$ 是协方差矩阵。匈牙利算法求最优匹配后，若代价低于门限 $\gamma$，则两条轨迹标签统一。

## 核心攻击：标签劫持

### 直觉解释：雷达拉偏类比

论文从经典雷达对抗的**拉偏欺骗（Pull-off Deception）**获得灵感：

```
经典雷达拉偏欺骗：
时刻1: 真实目标 ●           ← 雷达跟踪
时刻2: 真实目标 ● ○ 诱饵    (诱饵从目标处生成，进入同一轨迹门)
时刻3: 真实目标   ● ○ →     (诱饵"拉走"跟踪)
时刻4:             ●  ○← 雷达现在跟踪的是诱饵！
```

**标签劫持**逻辑完全相同，只是发生在**标签共识层**而非信号层：

1. **靠近阶段**：注入伪造轨迹，运动状态与目标高度相似
2. **匹配阶段**：共识协议将伪造轨迹与目标匹配，分配相同标签
3. **拉偏阶段**：伪造轨迹逐渐偏移，真实目标被迫"失去"原有身份

### 数学形式化

**攻击者目标**：注入伪造轨迹 $\tilde{\mathbf{x}}$，在轨迹共识中劫持目标 $k$ 的标签。

**匹配条件（进入匹配门限）**：

$$d(\tilde{\mathbf{x}}, \mathbf{x}_k) < \gamma$$

**隐蔽性约束**：伪造轨迹不应触发异常检测，需满足：

$$\|\tilde{\mathbf{x}} - \mathbf{x}_k\|_2 \leq \epsilon$$

**拉偏优化**：获得目标标签后，设计运动向量 $\mathbf{v}_{pull}$ 最大化身份混淆：

$$\max_{\mathbf{v}_{pull}} \mathcal{L}_{confusion}(\tilde{\mathbf{x}} + \mathbf{v}_{pull}, \mathbf{x}_k)$$

约束条件：伪造轨迹保持合理的运动学特性（速度、加速度限制）。

## 代码实现

### 轨迹共识核心

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def mahalanobis_cost(xi, Pi, xj, Pj):
    """计算两条轨迹之间的马氏距离代价"""
    diff = xi - xj
    S = Pi + Pj
    return float(diff @ np.linalg.solve(S, diff))

def track_consensus(tracks_a: list, tracks_b: list, gate: float = 9.21):
    """
    基于匈牙利算法的双节点轨迹共识
    gate=9.21 对应 4D 状态下 99% 置信度的卡方门限
    返回: 匹配对 [(label_a, label_b, cost), ...]
    """
    n, m = len(tracks_a), len(tracks_b)
    cost_matrix = np.full((n, m), 1e6)

    for i, ta in enumerate(tracks_a):
        for j, tb in enumerate(tracks_b):
            c = mahalanobis_cost(ta['x'], ta['P'], tb['x'], tb['P'])
            if c < gate:  # 仅门限内的轨迹参与匹配
                cost_matrix[i, j] = c

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] < gate:
            matches.append((tracks_a[r]['label'], tracks_b[c]['label'], cost_matrix[r, c]))
            tracks_b[c]['label'] = tracks_a[r]['label']  # 共识：统一标签
    return matches
```

### 标签劫持攻击

```python
def craft_hijacking_track(target: dict, pull_vector: np.ndarray = None,
                           noise_std: float = 0.3) -> dict:
    """
    构造标签劫持伪造轨迹
    阶段1 (靠近): pull_vector=None，在目标附近注入小扰动
    阶段2 (拉偏): pull_vector 指定拉偏方向和幅度
    """
    spoofed_x = target['x'].copy()
    if pull_vector is None:
        # 靠近阶段：小扰动确保进入匹配门限
        spoofed_x[:2] += np.random.randn(2) * noise_std
        spoofed_x[2:] += np.random.randn(2) * noise_std * 0.1
    else:
        spoofed_x += pull_vector  # 拉偏阶段：沿指定方向偏移

    return {'x': spoofed_x, 'P': target['P'].copy(),
            'label': -1, 'is_spoofed': True}

def simulate_hijacking_attack(target_track: dict, n_steps: int = 10):
    """模拟完整的标签劫持过程，返回每步代价和位置日志"""
    pull_dir = np.array([5.0, 0.0, 0.0, 0.0])  # 沿 x 轴拉偏
    log = []
    for t in range(n_steps):
        pv = None if t < 3 else pull_dir * (t - 2)  # 前3步靠近，之后拉偏
        spoofed = craft_hijacking_track(target_track, pull_vector=pv)
        cost = mahalanobis_cost(spoofed['x'], spoofed['P'],
                                target_track['x'], target_track['P'])
        log.append({'t': t, 'cost': cost,
                    'hijacked': cost < 9.21,
                    'spoof_pos': spoofed['x'][:2].copy()})
    return log
```

### 三传感器网络演示与可视化

```python
import matplotlib.pyplot as plt

def demo_three_sensor_attack():
    I = np.eye(4) * 1.0
    target = {'x': np.array([10., 5., 1., 0.]), 'P': I, 'label': 'T1'}

    sensor_b_tracks = [
        {'x': np.array([10.2, 5.1, 0.9, 0.1]), 'P': I, 'label': 'B_local_1'},
        {'x': np.array([30.0, 15.0, -1.0, 0.5]), 'P': I, 'label': 'B_local_2'},
    ]
    # 注入伪造轨迹（靠近目标T1）
    sensor_b_tracks.append(craft_hijacking_track(target, noise_std=0.2))
    matches = track_consensus([target], sensor_b_tracks)
    for m in matches:
        if m[1] == -1:
            print(f"⚠️  伪造轨迹成功劫持标签 {m[0]}！代价={m[2]:.3f}")

    log = simulate_hijacking_attack(target)
    costs = [e['cost'] for e in log]
    pos = np.array([e['spoof_pos'] for e in log])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(costs, 'b-o', label='伪造轨迹 vs 目标 代价')
    axes[0].axhline(y=9.21, color='r', linestyle='--', label='匹配门限 γ=9.21')
    axes[0].fill_between(range(len(costs)), 0, 9.21, alpha=0.1, color='green')
    axes[0].set(xlabel='时间步', ylabel='马氏距离代价', title='标签劫持：代价曲线')
    axes[0].legend()

    axes[1].plot(pos[:3, 0], pos[:3, 1], 'go-', label='靠近阶段', markersize=8)
    axes[1].plot(pos[3:, 0], pos[3:, 1], 'rs-', label='拉偏阶段', markersize=8)
    axes[1].scatter(*target['x'][:2], marker='*', s=200, c='blue', label='真实目标', zorder=5)
    axes[1].set(xlabel='x (m)', ylabel='y (m)', title='伪造轨迹运动')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('label_hijacking.png', dpi=150)

demo_three_sensor_attack()
```

**预期输出**：
- 左图：代价曲线前3步在门限 $\gamma=9.21$ 以下（靠近阶段匹配成功），后续逐渐超出（拉偏阶段开始）
- 右图：绿色点聚集在真实目标附近，红色点逐渐远离——目标失去原有身份

## 实验分析

### 攻击对标签一致性的影响

| 场景 | 标签一致率 | 身份错误率 | 备注 |
|------|-----------|-----------|------|
| 无攻击（基线） | 98.2% | 0.3% | 正常运行 |
| 随机噪声注入 | 95.1% | 2.1% | 非定向攻击 |
| 标签劫持（本文） | 61.3% | 34.7% | 定向身份欺骗 |
| 最优隐蔽攻击 | 72.5% | 22.8% | 隐蔽性优先变体 |

关键观察：

- 随机噪声攻击破坏性有限，因为不针对匹配逻辑本身
- 标签劫持专门利用共识协议机制，效果显著更强
- 隐蔽攻击在维持欺骗效果的同时，降低被检测概率，反而比直接攻击更有持续性

### 攻击成功的关键参数

- **门限 $\gamma$**：门限越宽松，攻击越容易发动
- **协方差 $P$**：目标预测不确定性越高，攻击者操作空间越大
- **网络拓扑**：链式拓扑比全连接拓扑更脆弱，单点攻击可级联传播

## 工程实践中的挑战

### 防御思路

**1. 运动学合理性检测**：

```python
def detect_anomalous_tracks(tracks: list, max_speed: float = 50.0) -> list:
    """过滤运动学不合理的轨迹"""
    suspicious = []
    for t in tracks:
        speed = np.linalg.norm(t['x'][2:])
        if speed > max_speed:
            suspicious.append(t['label'])
    return suspicious
```

**2. 三角一致性验证**：在 A-B-C-A 闭环路径中验证标签传递一致性。注入的伪造轨迹在三节点闭环中会暴露矛盾，因为伪造轨迹无法同时欺骗所有节点。

**3. 加密认证**：在轨迹广播消息中加入数字签名，防止未授权节点注入轨迹——但增加通信开销和延迟。

### 常见坑

1. **门限选择两难**
   - 门限太紧 → 合法匹配率下降，漏掉真实关联
   - 门限太松 → 攻击者有更大注入空间
   - 建议：根据目标速度和传感器精度**自适应调整门限**

2. **时钟不同步引发误报**
   - 传感器时钟不同步导致状态外推误差，合法轨迹出现异常高代价
   - 被误判为攻击，影响正常共识
   - 解决：时间戳校准 + 容忍一定的时间漂移

3. **单节点信任即全网失效**
   - 攻击者控制一个合法节点后，可从内部注入轨迹，绕过所有外部认证
   - 需要在节点间建立信任分级机制

## 什么时候需要关注这个漏洞？

| 高风险场景 | 低风险场景 |
|-----------|-----------|
| 军事/国防跟踪系统 | 纯粹的位置估计（不关心 ID） |
| 无人机管控（需精确身份） | 单传感器系统（无共识层） |
| 自动驾驶协同感知 | FoV 完全重叠的密集传感器网络 |
| 链式稀疏传感器拓扑 | 短时间跟踪（无持续标签需求） |

## 与其他跟踪攻击对比

| 攻击类型 | 攻击目标 | 所需权限 | 破坏层级 |
|---------|---------|---------|---------|
| 测量欺骗 | 位置估计 | 信号层注入 | 运动状态 |
| 轨迹丢失攻击 | 目标消失 | 门限附近扰动 | 存在性 |
| **标签劫持（本文）** | **身份混淆** | **共识层注入** | **语义身份层** |
| 重放攻击 | 历史轨迹复现 | 数据包拦截 | 时序一致性 |

标签劫持的独特之处在于攻击的是**跟踪系统的语义层**——不是"物体在哪"，而是"物体是谁"。

## 我的观点

这篇论文揭示了一个系统设计层面的深层漏洞：**TC-DMTT 的信任模型假设所有参与节点诚实**，而这个假设在对抗环境中根本不成立。

值得关注的开放问题：

1. **大规模网络的级联效应**：论文用三传感器演示，20+ 节点网络下攻击的传播动态尚不清楚，一个节点被攻陷会引发多大范围的标签污染？

2. **与深度学习检测器联合防御**：如果在共识前用 GNN 或 Transformer 做异常轨迹过滤，能否早期识别伪造轨迹？这与联邦学习中的拜占庭鲁棒聚合（Krum、Trimmed Mean）有直接的方法论联系。

3. **动态目标场景的脆弱性**：目标机动越剧烈，Kalman 预测误差越大，匹配门限需要放宽——攻击者操作空间越大，这是一个固有矛盾。

从工程角度看，真正的挑战不是"知道攻击存在"，而是**在不大幅增加通信和计算开销的前提下实现有效防御**——实时跟踪系统对延迟极度敏感，任何防御机制都必须在这个约束下设计。对于军事 C2 系统和自动驾驶协同感知的安全性研究，这个方向有直接的现实意义。