---
layout: post-wide
title: "让 LLM 的思考不再浪费 Token：ACTS 推理控制深度解析"
date: 2026-06-03 12:05:05 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.03965v1
generated_by: Claude Code CLI
---

## 一句话总结

ACTS 把推理过程建模为马尔可夫决策过程，用强化学习训练一个控制器，在推理时动态发出"策略 + 引导短语"来操控冻结推理模型的思考方向——不改任何推理模型权重，就能在保持准确率的同时大幅压缩 token 消耗。

## 背景：CoT 的代价和现有方法的局限

现代推理模型（DeepSeek-R1、QwQ-32B）在难题上确实有效，但代价高昂：一道高中数学题可能消耗 2000–8000 个思考 token，而最终答案本身可能只需要 50 个。

**现有压缩方法的共同缺陷：**

- **预算强制（Budget Forcing）**：强行截断推理，准确率断崖式下跌
- **早停（Early Stopping）**：判断何时停是个难题，容易停太早或太晚
- **KV cache 压缩**：针对内存，不针对推理质量控制

这些方法把"token 少"当目标，而不是让模型"聪明地少想"。

ACTS 的核心 insight：**推理策略是可以主动控制的**。给一个思考中的模型插入"好了，预算只剩 30% 了，快速验证然后作答"，模型真的会调整行为。关键在于：冻结推理模型，只训练一个轻量控制器。

## 算法原理

### MDP 建模

把推理过程建模为 MDP：

- **状态** $s_t = (r_{1:t},\ b_t)$：$r_{1:t}$ 是当前推理轨迹，$b_t$ 是剩余 token 预算
- **动作** $a_t = (\sigma_t,\ p_t)$：$\sigma_t$ 是推理策略，$p_t$ 是插入推理流的自然语言引导短语
- **转移**：冻结推理模型接收引导短语，生成下一段推理 $r_{t+1}$
- **奖励**：答对加分，超预算扣分

微妙但关键的设计：**推理模型是冻结的**。控制器通过在推理流中插入自然语言短语来影响推理走向，不需要对推理模型做任何修改。这使整个框架可以即插即用于任何推理模型。

### 推理策略空间

| 策略 | 触发场景 | 示例引导短语 |
|------|---------|------------|
| CONTINUE | 预算充裕，当前思路正确 | "继续..." |
| VERIFY | 有疑问，需要验证结论 | "等等，让我检查一下..." |
| SIMPLIFY | 预算紧张，推理过度展开 | "实际上，更直接的方式是..." |
| BACKTRACK | 当前路径明显走错 | "这条路走不通，换个角度..." |
| CONCLUDE | 预算快耗尽，结论已明确 | "综上所述，答案是..." |

### 预算条件奖励塑形

这是 RL 部分最重要的设计。**不同预算条件下，奖励函数不同**：

$$
R(s_T) = \underbrace{\alpha \cdot \mathbb{1}[\text{correct}]}_{\text{准确性}} + \underbrace{\mathbb{1}[\text{correct}] \cdot \beta \cdot \frac{b_T}{B}}_{\text{效率奖励（节省越多越好）}} - \underbrace{\gamma \cdot \max\!\left(0,\ \frac{B - b_T}{B} - \epsilon\right)}_{\text{超预算惩罚}}
$$

**关键设计**：效率奖励只在答对时生效——防止模型学会"快速瞎猜"。

多预算增强（Multi-budget Augmentation）：对同一道题，分别设置 25%、50%、75%、100% 的预算上限，生成不同条件下的训练轨迹。让控制器学会在不同压力下做不同选择。

## 实现

### 状态与动作定义

```python
from dataclasses import dataclass
from enum import Enum

class Strategy(str, Enum):
    CONTINUE  = "CONTINUE"
    VERIFY    = "VERIFY"
    SIMPLIFY  = "SIMPLIFY"
    BACKTRACK = "BACKTRACK"
    CONCLUDE  = "CONCLUDE"

@dataclass
class ReasoningState:
    problem: str
    trace: str
    budget_used: int
    budget_total: int

    @property
    def budget_remaining_ratio(self) -> float:
        return 1.0 - self.budget_used / self.budget_total

    def to_controller_prompt(self) -> str:
        ratio = self.budget_remaining_ratio
        urgency = "【预算紧张】" if ratio < 0.3 else "【预算充裕】"
        return (
            f"问题：{self.problem}\n\n"
            f"当前推理（最近部分）：\n{self.trace[-500:]}\n\n"
            f"{urgency} 剩余预算：{ratio:.0%}\n"
            f"请选择下一步策略并生成引导短语。\n"
            f"格式：<策略>STRATEGY</策略><引导>PHRASE</引导>"
        )

@dataclass
class ControllerAction:
    strategy: Strategy
    phrase: str       # 插入推理流的引导短语
    log_prob: float = 0.0
```

### 控制器智能体

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class ControllerAgent:
    """轻量控制器：观察推理状态，输出策略 + 引导短语"""

    DEFAULT_PHRASES = {
        Strategy.CONTINUE:  "继续，",
        Strategy.VERIFY:    "等等，让我验证一下：",
        Strategy.SIMPLIFY:  "实际上，更直接地说：",
        Strategy.BACKTRACK: "这条路不对，换个思路：",
        Strategy.CONCLUDE:  "综上，最终答案是：",
    }

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

    def act(self, state: ReasoningState) -> ControllerAction:
        prompt = state.to_controller_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 从最后一个 token 的 logits 中提取策略概率
        logits = outputs.logits[:, -1, :]
        strategy_ids = self._get_strategy_token_ids()
        strategy_logits = logits[:, strategy_ids]
        probs = F.softmax(strategy_logits, dim=-1)

        chosen_idx = probs.argmax().item()
        strategy = list(Strategy)[chosen_idx]
        log_prob = probs[0, chosen_idx].log().item()

        return ControllerAction(
            strategy=strategy,
            phrase=self.DEFAULT_PHRASES[strategy],
            log_prob=log_prob,
        )

    def _get_strategy_token_ids(self) -> list[int]:
        return [
            self.tokenizer.encode(s.value, add_special_tokens=False)[0]
            for s in Strategy
        ]
```

### 预算条件奖励函数

```python
def compute_budget_conditioned_reward(
    is_correct: bool,
    budget_used: int,
    budget_total: int,
    alpha: float = 1.0,   # 准确性权重
    beta: float = 0.2,    # 效率奖励权重（从 0.1 开始调）
    gamma: float = 0.3,   # 超预算惩罚权重
) -> float:
    budget_ratio = budget_used / budget_total

    if is_correct:
        accuracy_reward = alpha
        # 节省越多，效率奖励越高；答错则不给效率奖励
        efficiency_reward = beta * (1.0 - budget_ratio)
        return accuracy_reward + efficiency_reward
    else:
        return -alpha * 0.5  # 答错惩罚，不依赖 budget 使用量
```

### RL 训练循环

```python
def train_one_episode(
    controller: ControllerAgent,
    reasoner,          # 冻结的推理 LLM（调用 API 或本地推理）
    problem: str,
    answer: str,
    budget_total: int,
    optimizer,
    gamma_discount: float = 0.99,
) -> float:
    state = ReasoningState(
        problem=problem, trace="",
        budget_used=0, budget_total=budget_total
    )
    actions = []

    for _ in range(10):  # 最多 10 次控制介入
        if state.budget_remaining_ratio < 0.05:
            break

        action = controller.act(state)
        actions.append(action)

        # 推理模型接收引导短语，生成下一段推理
        new_chunk = reasoner.generate(
            state.trace + action.phrase,
            max_new_tokens=int(budget_total * 0.2),
        )
        tokens_used = len(new_chunk.split())
        state = ReasoningState(
            problem=problem,
            trace=state.trace + action.phrase + new_chunk,
            budget_used=state.budget_used + tokens_used,
            budget_total=budget_total,
        )
        if action.strategy == Strategy.CONCLUDE:
            break

    # 计算奖励并执行 REINFORCE 更新
    is_correct = evaluate_answer(state.trace, answer)
    reward = compute_budget_conditioned_reward(
        is_correct, state.budget_used, budget_total
    )

    # 归一化奖励（重要！降低方差）
    log_probs = torch.tensor([a.log_prob for a in actions])
    loss = -(reward * log_probs.sum())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.model.parameters(), 0.5)
    optimizer.step()

    return reward
```

### 多预算数据增强（SFT 预训练阶段）

RL 之前，先用监督学习初始化控制器。从完整推理轨迹生成多个预算条件下的样本：

```python
def create_multi_budget_trajectories(
    problem: str,
    full_trace: str,
    answer: str,
    budget_ratios: list[float] = [0.25, 0.5, 0.75, 1.0],
) -> list[dict]:
    full_tokens = len(full_trace.split())
    trajectories = []

    for ratio in budget_ratios:
        target_tokens = int(full_tokens * ratio)
        truncated = " ".join(full_trace.split()[:target_tokens])

        # 根据预算剩余程度标注理想策略（启发式规则）
        if ratio < 0.3:
            ideal_strategy = Strategy.CONCLUDE
        elif ratio < 0.6:
            ideal_strategy = Strategy.SIMPLIFY
        else:
            ideal_strategy = Strategy.CONTINUE

        trajectories.append({
            "problem": problem,
            "trace": truncated,
            "budget_ratio": ratio,
            "ideal_strategy": ideal_strategy.value,
            "answer": answer,
        })

    return trajectories
```

## 调试指南

### 常见问题

**1. 控制器总是输出 CONCLUDE，答得太快**

效率权重过高，控制器学到了"快速乱猜也能拿效率分"。

```python
# 错误：效率权重高于准确性权重
beta = 1.5  # 节省 token 的奖励远大于答对的奖励

# 修复：降低效率权重，alpha 始终 ≥ 4 * beta
beta = 0.1
```

**2. 推理模型无视引导短语**

部分模型对中途插入的短语不敏感，用格式标记强化插入：

```python
# 弱插入（可能被忽视）
inserted = f"{trace}{phrase}"

# 强插入：换行 + 显式标记
inserted = f"{trace}\n\n[Steering Signal]: {phrase}\n"
```

**3. RL 训练不收敛，loss 持续震荡**

RL 的经典问题，三步排查：

```python
# 1. 检查奖励归一化（最常被忽视）
rewards = torch.tensor(reward_buffer)
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# 2. 更保守的梯度裁剪
torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.5)  # 而非 1.0

# 3. 如果还不收敛，换 PPO：REINFORCE 方差太大
```

**4. 高预算下控制器仍然保守（一直用 SIMPLIFY）**

SFT 阶段低预算样本过多，分布不均匀：

```python
# 确保各预算比例的训练样本数量相近
from collections import Counter
dist = Counter(t["budget_ratio"] for t in trajectories)
print(dist)  # 检查分布是否均匀
```

### 判断训练是否有效

| 指标 | 好的信号 | 坏的信号 |
|------|---------|---------|
| 策略分布 | 高预算多 CONTINUE，低预算多 CONCLUDE | 策略分布几乎不变 |
| Token 消耗 | 随训练步数缓慢下降 | 始终接近预算上限 |
| 准确率 | 不低于 full-thinking 的 90% | 大幅低于 baseline |
| 奖励曲线 | 缓慢上升（1k+ episodes 才明显） | 完全平坦或剧烈震荡 |

### 超参数调优

| 参数 | 推荐范围 | 敏感度 | 建议 |
|------|---------|-------|------|
| $\alpha$（准确性权重） | 1.0（固定） | — | 不要动，作为其他参数的基准 |
| $\beta$（效率权重） | 0.1–0.3 | 高 | 从 0.1 开始，缓慢上调 |
| $\gamma$（超预算惩罚） | 0.2–0.5 | 中 | 视任务对超支容忍度调整 |
| 控制器学习率 | 1e-5–5e-5 | 高 | 比普通 SFT 低一个量级 |
| 折扣因子 | 0.95–0.99 | 低 | 0.99 通常够用 |
| 最大控制步数 | 5–15 | 中 | 对应问题复杂度，数学题用 8–10 |

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 推理模型本身能力强（QwQ、R1 级别） | 基础模型推理能力弱（7B 以下未专训） |
| 有明确的 token 预算约束（成本/延迟） | 对延迟敏感（控制器本身也有推理开销） |
| 问题难度差异大，简单题不需要深思 | 所有问题难度相近，一刀切即可 |
| 多任务场景（数学、代码、推理混合） | 单一固定任务，规则性早停更简单 |

## 我的观点

**ACTS 的真正价值不是那几个百分点的 token 节省。**

它的核心贡献是提供了一个"推理可控性"框架：以前我们只能在训练时塑造推理风格，ACTS 把推理控制变成了一个可学习、可优化的推理时决策过程。框架清晰，实验扎实。

但要清醒几件事：

1. **控制器本身有开销**：控制器也是一个 LLM，每次介入都要跑一次前向，对延迟敏感的场景需要仔细权衡
2. **插入短语有效的前提是推理模型"听话"**：对 RLHF 程度较低的模型，同一引导短语效果差异很大，需要实验验证
3. **复现难度不小**：multi-budget augmentation 的"理想策略标注"是启发式的，论文没有完全公开细节，自己构建需要多次实验

官方代码：[https://github.com/Andree-9/ACTS](https://github.com/Andree-9/ACTS)

**实用建议**：如果你有推理 token 成本压力，且已经在用 QwQ 或 R1 类模型，值得一试，先从简单的预算强制 baseline 对比开始。如果还在用 7B 以下未专门训练推理的模型，先把基础推理能力做好，再考虑控制效率。