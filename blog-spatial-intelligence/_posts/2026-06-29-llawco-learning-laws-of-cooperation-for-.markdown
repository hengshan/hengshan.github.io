---
layout: post-wide
title: "具身多智能体协作：让 LLM 智能体从失败中学习合作规律"
date: 2026-06-29 12:02:29 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.28182v1
generated_by: Claude Code CLI
---

## 一句话总结

LLawCo 框架让具身智能体通过反思历史失败，自动提炼出高层行为规律（如"必要时才通话"、"等待搭档"），并通过监督微调将这些规律注入推理链，从而大幅提升多智能体在部分可观测环境中的协作效率。

## 为什么这个问题重要？

想象两个机器人在仓库里协同搬货：它们看不到对方的完整状态，通信有延迟，还得共同完成任务。现有方案普遍有两类问题：

**纯 MARL（强化学习多智能体）方案**
- 训练慢、泛化差、难以解释行为
- 换一个任务就得重新训练

**LLM-based 智能体方案**
- 推理能力强，但智能体之间行为不一致
- 一个智能体已经拿到物品了，另一个还在"讨论"去哪里拿
- 对环境当前状态感知不准确，导致幻觉性决策

核心矛盾：**LLM 有强大的推理能力，但缺乏在协作任务中自动对齐行为的机制**。LLawCo 的创新在于把"从失败中学习规律"这个人类直觉，系统化成一套可训练的框架。

## 背景知识

### 具身智能体的特殊挑战

具身智能体（Embodied Agent）不同于 NLP 任务中的 LLM：

| 特性 | NLP 任务 | 具身任务 |
|------|---------|---------|
| 观测 | 完整文本 | 部分可见（视野/传感器限制） |
| 行动 | 生成 token | 物理动作（移动、抓取） |
| 反馈 | 即时 | 延迟，且受物理约束 |
| 通信 | 无限制 | 有代价（通信本身是行动） |

### 部分可观测性（Partial Observability）

每个智能体只能观测到环境的一个子集。数学上：

$$
o_i^t = \Omega(s^t, i)
$$

其中 $s^t$ 是全局状态，$o_i^t$ 是智能体 $i$ 在时刻 $t$ 的局部观测，$\Omega$ 是观测函数（通常是视野范围内的物体和状态）。

这意味着：智能体 A 不知道智能体 B 手里有没有拿东西，除非 B 主动汇报或 A 亲自走过去看。

## 核心方法

### 直觉解释

LLawCo 的核心思路用一句话概括：**让智能体像老员工带新人一样工作——先总结出哪些合作方式不对，再提炼成规则，然后让新智能体严格遵守这些规则思考问题**。

三步走：
```
历史失败案例 → 反思提取错误模式 → 抽象成行为规律 → SFT 注入 CoT → 新任务中规律引导决策
```

### Pipeline 概览

```
Phase 1: 规律提取（离线）
失败轨迹 → LLM Reflection → 错误行为模式 → 聚类 → 行为规律集合 {L₁, L₂, ...}

Phase 2: 规律注入（SFT）
规律集合 + 任务上下文 → 构建 CoT 训练数据 → 监督微调 LLM Backbone

Phase 3: 在线推理（部署）
当前观测 + 行为规律 → 规律引导 CoT → 行动决策
```

### 行为规律的形式

规律不是复杂的数学约束，而是自然语言的高层原则，例如：

- `"Talk when necessary"`: 只在信息不对称会影响任务的时候通信，避免无效对话刷屏
- `"Wait for partner"`: 如果搭档正在执行关键动作，等待而不是干扰
- `"Claim tasks explicitly"`: 接手某个子任务前，明确告知搭档，避免重复劳动

这些规律通过以下损失函数被注入到模型的推理过程中：

$$
\mathcal{L}_{SFT} = -\sum_{t} \log P_\theta(a_t \mid o_t, \mathcal{L}_{laws}, h_t)
$$

其中 $\mathcal{L}_{laws}$ 是行为规律集合，$h_t$ 是历史上下文，$a_t$ 是目标行动（来自专家轨迹）。

## 实现

### 核心数据结构

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class ActionType(Enum):
    MOVE = "move"
    PICKUP = "pickup"
    PLACE = "place"
    COMMUNICATE = "communicate"
    WAIT = "wait"

@dataclass
class AgentObservation:
    agent_id: str
    position: tuple
    held_object: Optional[str]
    visible_objects: List[str]     # 只能看到视野内的
    received_messages: List[str]   # 来自搭档的消息

@dataclass
class CooperationLaw:
    name: str
    description: str
    trigger_condition: str         # 何时激活此规律
    guidance: str                  # 具体行为指导
    derived_from_failures: List[str] = field(default_factory=list)
```

### 失败模式提取（核心）

```python
import json
from typing import List, Dict

def extract_failure_patterns(failed_trajectories: List[Dict]) -> List[str]:
    """
    从失败轨迹中提取错误行为模式。
    实际论文用 LLM 做 reflection，这里展示模板化的简化版本。
    """
    patterns = []
    for traj in failed_trajectories:
        actions = traj["actions"]
        failure_reason = traj["failure_reason"]
        
        # 检测冗余通信模式
        comm_count = sum(1 for a in actions if a["type"] == "communicate")
        total_steps = len(actions)
        if comm_count / total_steps > 0.4:
            patterns.append("REDUNDANT_COMMUNICATION: agents spent >40% steps communicating")
        
        # 检测重复任务分配
        pickups = [a for a in actions if a["type"] == "pickup"]
        object_ids = [a["object"] for a in pickups]
        if len(object_ids) != len(set(object_ids)):
            patterns.append("DUPLICATE_TASK: multiple agents attempted same pickup")
        
        # 检测等待时机错误
        waits_before_critical = _check_premature_waits(actions)
        if waits_before_critical:
            patterns.append(f"PREMATURE_WAIT: agent waited unnecessarily at {waits_before_critical}")
    
    return list(set(patterns))  # 去重

def _check_premature_waits(actions: List[Dict]) -> Optional[str]:
    for i, action in enumerate(actions[:-1]):
        if action["type"] == "wait":
            next_action = actions[i + 1]
            # 如果等待后搭档并没有完成关键动作，视为过早等待
            if next_action.get("partner_completed_critical", False) is False:
                return f"step_{i}"
    return None
```

### 行为规律生成

```python
PATTERN_TO_LAW_TEMPLATE = {
    "REDUNDANT_COMMUNICATION": CooperationLaw(
        name="Talk when necessary",
        description="只在搭档缺少关键信息时才通信",
        trigger_condition="当你掌握搭档不知道但影响其决策的信息时",
        guidance="默认执行任务，仅在：(1)发现搭档走向无效目标 (2)你的计划改变 时才发消息"
    ),
    "DUPLICATE_TASK": CooperationLaw(
        name="Claim tasks explicitly",
        description="接手子任务前明确声明，避免重复",
        trigger_condition="准备执行可能与搭档重叠的动作时",
        guidance="执行前广播 'I will [action] [object]'，若搭档已声明同一任务则选其他"
    ),
    "PREMATURE_WAIT": CooperationLaw(
        name="Wait for partner",
        description="只在搭档执行关键动作时等待，否则继续自己的任务",
        trigger_condition="搭档正在执行会影响当前环境状态的动作",
        guidance="等待条件：搭档在移动你需要的物体；否则并行执行各自子任务"
    ),
}

def derive_laws(patterns: List[str]) -> List[CooperationLaw]:
    laws = []
    for pattern in patterns:
        pattern_key = pattern.split(":")[0]
        if pattern_key in PATTERN_TO_LAW_TEMPLATE:
            law = PATTERN_TO_LAW_TEMPLATE[pattern_key]
            law.derived_from_failures.append(pattern)
            laws.append(law)
    return laws
```

### 带规律的 Chain-of-Thought 推理

```python
def extract_failure_patterns(failed_trajectories: List[Dict]) -> List[str]:
    """从失败轨迹中提取错误行为模式（实际论文用 LLM reflection）"""
    patterns = []
    for traj in failed_trajectories:
        actions, failure_reason = traj["actions"], traj["failure_reason"]
        
        # 检测冗余通信（>40% 步骤用于通信）
        if sum(1 for a in actions if a["type"] == "communicate") / len(actions) > 0.4:
            patterns.append("REDUNDANT_COMMUNICATION")
        
        # 检测重复任务分配（多智能体抢同一目标）
        pickups = [a["object"] for a in actions if a["type"] == "pickup"]
        if len(pickups) != len(set(pickups)):
            patterns.append("DUPLICATE_TASK")
        
        # 检测过早等待
        # ... (premature wait 检测逻辑省略)
    
    return list(set(patterns))  # 去重
```

### 模拟对比实验

```python
import random

def simulate_cooperation(use_laws: bool, num_steps: int = 20) -> Dict:
    """简化的协作任务模拟，对比有无规律的效果"""
    
    # 任务：两个智能体搬运3个物体到目标区域
    objects_to_move = {"box_A", "box_B", "box_C"}
    delivered = set()
    total_comm = 0
    duplicate_attempts = 0
    claimed_tasks = set()  # 有规律时的任务声明
    
    for step in range(num_steps):
        for agent_id in ["agent_0", "agent_1"]:
            remaining = objects_to_move - delivered
            if not remaining:
                break
            
            if use_laws:
                # 有规律：先声明任务再执行
                target = random.choice(list(remaining - claimed_tasks)) if remaining - claimed_tasks else None
                if target:
                    claimed_tasks.add(target)
                    # 只在必要时通信（声明任务）
                    total_comm += 1
                    delivered.add(target)
            else:
                # 无规律：随机选目标，可能重复
                target = random.choice(list(remaining))
                # 过度通信（每步都通报）
                total_comm += 1
                if target in claimed_tasks:
                    duplicate_attempts += 1
                claimed_tasks.add(target)
                delivered.add(target)
        
        if delivered == objects_to_move:
            return {
                "success": True,
                "steps_to_complete": step + 1,
                "total_communications": total_comm,
                "duplicate_attempts": duplicate_attempts,
            }
    
    return {"success": False, "steps_to_complete": num_steps,
            "total_communications": total_comm, "duplicate_attempts": duplicate_attempts}


# 运行对比
results = {"with_laws": [], "without_laws": []}
for _ in range(100):
    results["with_laws"].append(simulate_cooperation(use_laws=True))
    results["without_laws"].append(simulate_cooperation(use_laws=False))

for condition, runs in results.items():
    success_rate = sum(r["success"] for r in runs) / len(runs)
    avg_comm = sum(r["total_communications"] for r in runs) / len(runs)
    avg_dup = sum(r["duplicate_attempts"] for r in runs) / len(runs)
    print(f"{condition}: success={success_rate:.1%}, avg_comm={avg_comm:.1f}, duplicates={avg_dup:.1f}")
```

## 实验

### 基准测试说明

论文引入了 **PARTNR-Dialog** 基准：在 PARTNR（家庭机器人任务）环境中加入对话通信能力，评估需要沟通协商的多智能体协作。

| 数据集 | 场景类型 | 任务复杂度 | 通信必要性 |
|--------|---------|-----------|-----------|
| PARTNR-Dialog | 家庭环境（厨房/卧室） | 高（多步规划） | 关键 |
| TDW-MAT | 物品运输 | 中 | 可选 |

### 定量结果

| 方法 | PARTNR-Dialog SR | TDW-MAT SR | 通信效率 |
|------|-----------------|-----------|---------|
| Vanilla LLM Agent | ~45% | ~52% | 低（过度通信） |
| CoELA（SOTA 开源） | ~58% | ~61% | 中 |
| **LLawCo** | **+4.5% vs CoELA** | **+6.8% vs CoELA** | 高 |

SR = Success Rate，以 CoELA 为基线，LLawCo 在四个不同 LLM backbone 上均取得提升。

## 工程实践

### SFT 训练数据构建

论文的核心工程难点是构建高质量的 CoT 训练数据。关键流程：

```python
def build_law_guided_prompt(observation, laws, task_description) -> str:
    laws_text = "\n".join([f"[Law: {l.name}] {l.guidance}" for l in laws])
    return f"""遵守以下合作规律：
{laws_text}

任务：{task_description} | 位置：{observation.position} | 持物：{observation.held_object or '无'}
可见：{', '.join(observation.visible_objects)} | 消息：{'; '.join(observation.received_messages) or '无'}

推理步骤：
1. [环境分析] 当前状态？搭档位置/行为？
2. [规律检查] 哪条规律与当前情境相关？
3. [行动规划] 基于规律，执行什么动作？
4. Action: <动作> | Target: <目标> | Message: <可选>"""


def parse_agent_action(llm_response: str) -> dict:
    import re
    # ... (字段提取逻辑)
    return {
        "action": (re.search(r"Action:\s*(\w+)", llm_response) or [None, "wait"])[1],
        "target": t.group(1).strip() if (t := re.search(r"Target:\s*([^\|]+)", llm_response)) else None,
        "message": m.group(1).strip() if (m := re.search(r"Message:\s*(.+)", llm_response)) else None,
    }
```

### 常见坑

**坑 1：规律冲突**
`"Talk when necessary"` 和 `"Claim tasks explicitly"` 在某些状态下会矛盾——前者要求少通信，后者要求声明任务。

解决方案：为规律添加优先级，任务声明优先于减少通信：

```python
import random

def simulate_cooperation(use_laws: bool, num_steps: int = 20) -> dict:
    objects = {"box_A", "box_B", "box_C"}
    delivered, claimed = set(), set()
    total_comm = duplicate_attempts = 0

    for step in range(num_steps):
        for agent_id in ["agent_0", "agent_1"]:
            remaining = objects - delivered
            if not remaining: break

            if use_laws:
                # 有规律：声明后执行，避免重复
                target = random.choice(list(remaining - claimed)) if remaining - claimed else None
                if target:
                    claimed.add(target); total_comm += 1; delivered.add(target)
            else:
                # 无规律：随机选择，可能重复
                target = random.choice(list(remaining))
                total_comm += 1
                if target in claimed: duplicate_attempts += 1
                claimed.add(target); delivered.add(target)

        if delivered == objects:
            return {"success": True, "steps": step + 1, "comm": total_comm, "dup": duplicate_attempts}

    return {"success": False, "steps": num_steps, "comm": total_comm, "dup": duplicate_attempts}

# ... (100次对比实验，统计成功率/通信量/重复率)
```

**坑 2：部分可观测导致规律误触发**

智能体 A 看不到 B 已经拿起了 box_A，所以 A 仍然声明要拿 box_A，规律失效。

解决方案：在 prompt 中显式区分"已知确定事实"和"推测"：

```python
def format_observation_with_uncertainty(obs: AgentObservation) -> str:
    return f"""
确定信息（我直接观测到）：{obs.visible_objects}
不确定信息（推测搭档状态）：基于最后消息 '{obs.received_messages[-1] if obs.received_messages else "无"}' 推断
"""
```

**坑 3：SFT 导致过拟合特定规律组合**

如果训练数据中某条规律总是和特定任务类型配对，模型会学到虚假关联。

解决方案：在构建 SFT 数据时，对规律的使用场景做多样化增强，确保同一条规律出现在不同任务类型中。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有历史失败数据可以挖掘 | 全新任务类型，没有失败案例 |
| 智能体数量少（2-4个） | 大规模智能体群体（蚁群类） |
| 任务需要显式协商 | 任务可以完全并行、无需协调 |
| 通信有代价的场景 | 通信完全自由且无成本 |
| 任务结构稳定，规律可复用 | 每次任务结构差异极大 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| MARL | 端到端优化 | 泛化差，样本低效 | 单一固定任务 |
| Vanilla LLM Agent | 泛化好，零样本 | 多智能体行为不对齐 | 单智能体或松耦合任务 |
| CoELA | 显式通信协议 | 协议设计需要人工 | 通信结构固定的任务 |
| **LLawCo** | 自动从失败学规律，SFT 注入推理 | 需要失败数据；SFT 开销 | 有历史数据的协作任务 |

## 我的观点

LLawCo 的思路有一个值得关注的优雅之处：**它没有试图设计更复杂的多智能体通信协议，而是让智能体学会"什么时候不该做某件事"**。这和人类团队协作的直觉高度吻合——优秀的团队成员知道何时闭嘴、何时等待，而不只是更努力地工作。

**几个值得关注的开放问题：**

1. **规律的迁移性**：从家庭任务学到的 "Wait for partner" 规律，能迁移到工业机器人场景吗？论文没有给出跨域实验。

2. **规律数量的上限**：当任务复杂度上升，规律库可能膨胀且相互矛盾，当前的优先级机制是否足够？

3. **实时性问题**：SFT 后的 LLM 每步推理仍需数百 ms，对实时机器人控制（100Hz+）几乎不可用。这个框架目前更适合高层任务规划，而非底层控制。

**离产品化还有多远？** 对于仓储机器人、室内服务机器人等高层协调场景（秒级决策），有望在 1-2 年内看到落地尝试。对于实时控制，短期内仍需混合架构：LLM 做任务分配，传统控制器做执行。