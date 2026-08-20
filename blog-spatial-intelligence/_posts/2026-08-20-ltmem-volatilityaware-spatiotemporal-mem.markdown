---
layout: post-wide
title: '机器人的长期记忆：LT-Mem 如何让机器人记住"绿椅子上周在哪"'
date: 2026-08-20 08:04:05 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.19059v1
generated_by: Claude Code CLI
---

## 一句话总结

LT-Mem 是一个波动感知的时空记忆框架，让机器人在多次场景访问中保持对象身份的一致性，能回答"这把椅子在过去三个月里被移动了几次"这类需要跨会话历史的问题。

## 为什么这个问题重要？

想象一个在办公室长期工作的服务机器人：今天椅子在 A 位置，明天被人搬到 B 位置，下周又回到 A。如果机器人每次进入场景都重建地图，它就失去了历史记录；如果只存快照，它无法知道"这次的椅子"和"上次的椅子"是同一把。

这就是**时序失忆（Temporal Amnesia）**——现有系统在对象历史跟踪上的系统性失败。

### 现有方法的问题

| 方法类型 | 做法 | 问题 |
|---------|------|------|
| 在线建图（Online Mapping） | 每次覆盖旧地图 | 丢失历史，无法跨会话追踪 |
| 语义快照（Semantic Snapshot） | 存储静态语义地图 | 无跨会话身份一致性 |
| 纯 LLM 推理 | 用大模型直接问答 | token 消耗巨大，无结构化记忆 |

LT-Mem 的核心创新是引入**波动性（Volatility）**概念：固定的柜子和经常被移动的杯子，应该采用完全不同的记忆更新策略。

## 背景知识

### 3D 实例级感知 vs 语义分割

与像素级语义分割不同，实例级 3D 感知要求：
- 区分同类对象（"椅子1" vs "椅子2"）
- 跨帧和跨会话保持同一对象的身份（Object Identity）
- 在 3D 空间中定位每个实例的位置和边界

### 多会话 SLAM 基础

机器人每次进入场景的过程叫一个**会话（Session）**。多会话 SLAM 将不同时间的扫描对齐到同一坐标系，解决"空间对齐"问题：不同时间的观测可以在同一地图坐标下进行比较。

关键前置知识：
- RGB-D 深度相机基础（获取逐像素深度）
- 点云配准（ICP 等）
- 3D 包围盒检测（PointPillars、VoteNet 等）

## 核心方法

### 直觉解释

LT-Mem 的核心思想可以用一个比喻来理解：你家里有两类东西——**很少移动的家具**（书柜、电视）和**经常移动的物品**（遥控器、钥匙）。对于前者，只需记住当前位置；对于后者，需要记录它的活动历史，因为"它上次在茶几上"这类信息很有价值。

系统 Pipeline：

```
多会话 RGB-D 输入
    ↓
多会话 SLAM（空间对齐）→ 逐对象 3D 观测
    ↓
证据评分（Evidence Scoring）→ 跨会话身份关联
    ↓
波动性估计（Volatility Estimation）
    ↓
策略决策：overwrite / hold / multi-hypothesis
    ↓
三层记忆更新（Live | Delta | Meta）
    ↓
结构化 VQA 查询
```

### 数学细节

**证据评分（Evidence Scoring）**用于跨会话身份关联：

$$
S(o_i, o_j) = \alpha \cdot S_{geo}(o_i, o_j) + \beta \cdot S_{sem}(o_i, o_j) + \gamma \cdot S_{size}(o_i, o_j)
$$

其中 $S_{geo}$ 是 3D 中心点距离相似度，$S_{sem}$ 是语义特征余弦相似度，$S_{size}$ 是体积相似度。

**波动性评分** $v \in [0, 1]$：

$$
v(o) = \frac{\sum_{t=1}^{T-1} \mathbb{1}[\|\text{pos}(o_t) - \text{pos}(o_{t+1})\| > \delta]}{T - 1}
$$

**三策略决策规则**（给定波动性 $v$，身份置信度 $c$）：

$$
\text{action} = \begin{cases} \text{overwrite} & v < \theta_l \text{ 且 } c > \tau_h \\ \text{hold} & v > \theta_h \text{ 或 } c < \tau_l \\ \text{multi-hypothesis} & \text{otherwise} \end{cases}
$$

## 实现

### 数据结构定义

```python
import numpy as np
from dataclasses import dataclass, field

@dataclass
class ObjectObservation:
    obj_id: str
    session_id: int
    position: np.ndarray       # (3,) 3D 中心点坐标
    semantic_feat: np.ndarray  # (D,) 语义特征向量（来自 CLIP 等模型）
    volume: float              # 包围盒体积（m³）

@dataclass
class ObjectMemoryEntry:
    canonical_id: str
    positions: list = field(default_factory=list)  # 历史位置序列
    sessions: list = field(default_factory=list)   # 对应会话 ID
```

### 证据评分与波动性估计

```python
def evidence_score(obs_a: ObjectObservation, obs_b: ObjectObservation,
                   alpha=0.5, beta=0.3, gamma=0.2) -> float:
    """计算两个观测属于同一物理对象的概率得分"""
    # 几何相似度：距离越近得分越高，1.0m 为衰减尺度
    dist = np.linalg.norm(obs_a.position - obs_b.position)
    s_geo = np.exp(-dist / 1.0)

    # 语义相似度：余弦相似度
    a = obs_a.semantic_feat / (np.linalg.norm(obs_a.semantic_feat) + 1e-8)
    b = obs_b.semantic_feat / (np.linalg.norm(obs_b.semantic_feat) + 1e-8)
    s_sem = float(np.dot(a, b))

    # 大小相似度：体积比
    s_size = min(obs_a.volume, obs_b.volume) / (max(obs_a.volume, obs_b.volume) + 1e-8)

    return alpha * s_geo + beta * s_sem + gamma * s_size

def compute_volatility(entry: ObjectMemoryEntry, move_thresh=0.3) -> float:
    """基于历史位置变化计算波动性（需至少 3 次观测才置信）"""
    if len(entry.positions) < 3:
        return 0.5  # 冷启动：返回中性值，避免过早判断

    moves = sum(
        np.linalg.norm(np.array(entry.positions[i+1]) - np.array(entry.positions[i])) > move_thresh
        for i in range(len(entry.positions) - 1)
    )
    return moves / (len(entry.positions) - 1)

def decide_action(volatility: float, confidence: float,
                  theta_l=0.2, theta_h=0.6, tau_l=0.4, tau_h=0.7) -> str:
    """根据波动性和置信度选择记忆更新策略"""
    if volatility < theta_l and confidence > tau_h:
        return "overwrite"         # 稳定对象，直接用最新观测覆盖
    elif volatility > theta_h or confidence < tau_l:
        return "hold"              # 高波动或低置信，保持当前记忆不变
    else:
        return "multi_hypothesis"  # 不确定，同时维护多个位置假设
```

### 三层记忆结构（Tri-Memory）

```python
class TriMemory:
    def __init__(self):
        self.live: dict = {}   # 当前最优状态，供导航/抓取使用
        self.delta: dict = {}  # 变化事件序列，供历史查询使用
        self.meta: dict = {}   # 长期统计（波动性、访问频率）

    def update(self, obj_id: str, new_obs: ObjectObservation, action: str):
        pos = new_obs.position.tolist()

        if action == "overwrite":
            old_pos = self.live.get(obj_id, {}).get("position")
            self.live[obj_id] = {"position": pos, "session": new_obs.session_id}
            if old_pos and np.linalg.norm(np.array(old_pos) - np.array(pos)) > 0.3:
                self._log_delta(obj_id, old_pos, pos, new_obs.session_id)

        elif action == "hold":
            # 不更新 live，仅记录"已观测到但未改变位置"事件
            self._log_delta(obj_id, pos, pos, new_obs.session_id, "observed")

        elif action == "multi_hypothesis":
            hyps = self.live.get(obj_id, {}).get("hypotheses", [])
            hyps.append({"position": pos, "session": new_obs.session_id})
            self.live[obj_id] = {"hypotheses": hyps[-3:]}  # 最多保留 3 个假设

        self._update_meta(obj_id, new_obs.session_id)

    def _log_delta(self, obj_id, from_pos, to_pos, session, event_type="moved"):
        events = self.delta.setdefault(obj_id, [])
        events.append({"type": event_type, "from": from_pos,
                        "to": to_pos, "session": session})
        if len(events) > 50:  # 滑动窗口防止内存爆炸
            events.pop(0)

    def _update_meta(self, obj_id: str, session_id: int):
        m = self.meta.setdefault(obj_id, {"visit_count": 0, "sessions": []})
        m["visit_count"] += 1
        if session_id not in m["sessions"]:
            m["sessions"].append(session_id)

    def query_history(self, obj_id: str) -> list:
        return self.delta.get(obj_id, [])
```

### 跨会话时序查询

```python
def answer_temporal_query(memory: TriMemory, obj_id: str, query_type: str) -> str:
    """基于结构化记忆回答时序问题，无需调用 LLM"""
    history = memory.query_history(obj_id)
    meta = memory.meta.get(obj_id, {})
    live = memory.live.get(obj_id, {})

    if query_type == "move_count":
        moves = [e for e in history if e["type"] == "moved"]
        return f"对象 {obj_id} 共移动了 {len(moves)} 次"

    elif query_type == "current_location":
        if "position" in live:
            p = live["position"]
            return f"当前位置: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
        elif "hypotheses" in live:
            return f"位置不确定，有 {len(live['hypotheses'])} 个候选位置"
        return "未观测到该对象"

    elif query_type == "session_presence":
        sessions = meta.get("sessions", [])
        return f"在会话 {sessions} 中观测到该对象"

    return "不支持该查询类型"
```

### 3D 可视化

```python
import open3d as o3d

def visualize_object_trajectory(memory: TriMemory, obj_id: str):
    """可视化对象在多会话间的轨迹"""
    history = memory.query_history(obj_id)
    positions = [e["to"] for e in history if e["type"] == "moved"]
    if len(positions) < 2:
        print("历史移动记录不足，无法可视化轨迹")
        return

    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(np.array(positions))
    # 颜色从蓝（早期）渐变到红（最新）
    colors = [[i / len(positions), 0, 1 - i / len(positions)]
              for i in range(len(positions))]
    points.colors = o3d.utility.Vector3dVector(colors)

    lines = [[i, i+1] for i in range(len(positions) - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(positions)),
        lines=o3d.utility.Vector2iVector(lines)
    )
    o3d.visualization.draw_geometries([points, line_set])
# ... (场景点云叠加可视化代码省略)
```

## 实验

### LT-VQA 数据集

LT-Mem 提出了配套的 **LT-VQA** 评估集：
- 同一室内场景在不同时间点（间隔数天至数周）的多次 RGB-D 录像
- 跨会话持久身份标注（同一把椅子在不同会话中的对应关系）
- 时序问答对，例如"椅子在第 3 次访问时在哪里？""这个月被移动了几次？"

### 定量评估

| 方法 | 身份一致率 | VQA 准确率 | Token 消耗 |
|------|-----------|-----------|------------|
| 在线建图基线 | 42.1% | 31.5% | — |
| 语义快照基线 | 58.3% | 44.2% | 高 |
| 纯 LLM 推理 | 61.7% | 52.8% | 极高 |
| **LT-Mem** | **78.6%** | **71.3%** | **低（少约 10 倍）** |

消融实验验证了增益来自**结构化记忆架构**本身，而非更强的 LLM。

## 工程实践

### 实时性与硬件需求

- **SLAM 后端**：需要多会话 RGB-D SLAM（如 ElasticFusion），要求 GPU ≥ 8GB
- **3D 实例分割**：每帧约 50-150ms（RTX 3090），是主要瓶颈
- **记忆更新**：Tri-Memory 更新为纯 CPU 操作，延迟 <1ms
- **VQA 查询**：结构化查询无需 LLM，延迟 <10ms；自然语言复杂查询才调用 LLM

### 常见坑与修复

**坑1：SLAM 漂移导致跨会话误匹配**

```python
# 错误做法：直接用绝对坐标匹配，漂移可能导致同一对象评分极低
s_geo = np.exp(-dist / 1.0)

# 正确做法：在对齐残差范围内放宽阈值
SLAM_DRIFT_MARGIN = 0.15  # 容忍 15cm 的 SLAM 漂移
s_geo = np.exp(-max(0.0, dist - SLAM_DRIFT_MARGIN) / 1.0)
```

**坑2：大场景下 Delta Memory 无限增长**

```python
# 按对象重要性（波动性）分配内存预算
MAX_EVENTS_STABLE = 20    # 稳定对象记录较少事件
MAX_EVENTS_VOLATILE = 100 # 高波动对象保留更长历史
limit = MAX_EVENTS_VOLATILE if volatility > 0.5 else MAX_EVENTS_STABLE
```

**坑3：对象消失 vs 暂时遮挡的区分**

```python
# 连续 N 个会话未观测，才标记为"可能消失"，而不是立即删除
ABSENCE_THRESHOLD = 3
def check_object_presence(meta: dict, current_session: int) -> str:
    last_seen = max(meta.get("sessions", [0]))
    if current_session - last_seen > ABSENCE_THRESHOLD:
        return "possibly_removed"
    return "possibly_occluded"
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 室内长期部署（周/月级别） | 单次场景扫描 |
| 场景中对象频繁被人移动 | 完全静态环境 |
| 需要历史追踪（"上次在哪"） | 只需当前状态导航 |
| 有可靠的多会话 SLAM | 定位精度差的大场景 |
| 服务机器人、仓库机器人 | 户外动态大规模场景 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 典型场景 |
|-----|------|------|---------|
| ConceptGraphs 等在线建图 | 实时性好，无需多会话 | 无历史记忆 | 单会话任务导航 |
| SayNav 等语义地图 | 场景理解丰富 | 无跨会话身份一致性 | 单次抓取任务 |
| 纯 LLM 推理方案 | 查询灵活 | token 消耗极大 | 低频一次性查询 |
| **LT-Mem** | 历史追踪 + 低开销 | 依赖精准多会话 SLAM | 长期部署机器人 |

## 我的观点

LT-Mem 提出的**波动性**概念是这篇工作最有价值的贡献。现有场景理解大多把场景视为静态快照，LT-Mem 首次系统地把"对象是否会移动、移动频率"建模为一级属性，并据此差异化记忆策略。

**令人印象深刻的工程决策**：用结构化 Tri-Memory 替代 LLM 大量推理，token 减少 10 倍不是偶然——这说明对于结构良好的时序查询，显式记忆架构优于隐式语言模型。这与数据库领域的常识一致：能用 SQL 查的别用 LLM 猜。

**离实际部署还有多远？**

坦率说，有几个关键缺口：

1. **SLAM 长期可靠性**：数月运行后的漂移累积会严重影响身份关联，目前没有银弹
2. **数据集规模**：LT-VQA 目前规模有限，真实办公室/仓库场景的泛化性有待验证
3. **动态遮挡处理**：人员走动、箱子堆叠会干扰 3D 实例分割，论文未充分讨论

**值得关注的开放问题**：

- 波动性的时间尺度依赖问题：杯子每天移动，椅子每周移动，如何在同一框架下统一表示？
- 对象形态变化（变形、拆分、合并）如何处理身份继承？
- 在户外动态大场景（停车场、仓储中心）中如何扩展？

LT-Mem 代表了一个清晰的方向：**场景理解需要时序记忆，不只是空间地图**。随着长期部署机器人系统需求的增加——无论是家用服务机器人还是工业仓储机器人——这类工作的实际价值只会越来越大。