---
layout: post-wide
title: '让 LLM 学会"真的推理"：SUPERNOVA 数据配方与 RLVR 实战'
date: 2026-04-10 08:04:43 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.08477v1
generated_by: Claude Code CLI
---

## 一句话总结

RLVR 在数学、代码上屡试不爽，但通用推理任务（因果推理、时序理解、逻辑归纳）总是跑不起来——SUPERNOVA 告诉你：问题不在算法，在于你喂的数据。

---

## 背景：为什么通用推理比数学难那么多？

### RLVR 为什么在数学上能 work？

Reinforcement Learning with Verifiable Rewards（RLVR）背后的逻辑很简单：让模型自己生成答案，用可验证的 reward 信号来更新参数，不需要人工标注偏好。

在数学上，这个框架天然适配：
- **Reward 清晰**：`"42" == "42"` 是非0即1的信号
- **数据充足**：GSM8K、MATH 早就标准化了
- **验证廉价**：字符串匹配，或调一个 Python 解释器

DeepSeek-R1、Qwen-QwQ 这条技术路线就是靠这个爆发的。

### 通用推理的三座大山

但一旦离开数学/代码领域，RLVR 就开始翻车：

1. **Reward 难以验证**：因果推理的问题往往有多个合理答案，怎么判断"对了"？
2. **数据标准不统一**：没有像 MATH 那样干净的通用推理数据集
3. **任务多样性**：时序理解、类比推理、常识推理——每个任务的难度曲线完全不同

这就是 SUPERNOVA 要解决的问题：**如何把现有的指令微调数据集（instruction-tuning datasets）改造成 RLVR 可以用的训练信号**？

### 核心 Insight

专家标注的指令微调数据集（如 FLAN、Super-NaturalInstructions）里已经有了答案——这些答案就是天然的 ground truth，可以直接当 verifiable reward 用。

问题转化为：**怎么选任务，怎么混合，以及答案格式是否需要处理**。

---

## 算法原理

### RLVR 的目标函数

RLVR 的核心是最大化期望奖励：

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} [r(x, y)]
$$

其中 $r(x, y)$ 是可验证的奖励函数（对则 1，错则 0）。

SUPERNOVA 使用 GRPO（Group Relative Policy Optimization）作为基础算法，核心更新是：

$$
\mathcal{L}_\text{GRPO} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta(y \mid x)}{\pi_{\theta_\text{old}}(y \mid x)} A, \, \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_\text{old}}}, 1 \pm \epsilon\right) A\right)\right]
$$

$A$ 是组内相对优势（group-normalized advantage），这比 PPO 的绝对优势估计更稳定。

### 三个关键设计决策（从 100+ 实验中蒸馏出来的）

SUPERNOVA 的主要贡献不是新算法，而是系统性地研究了数据侧的三个问题：

**决策一：源任务选择（Non-trivial！）**
- 不是所有任务都适合 RLVR
- 适合的任务特征：答案简短且可精确匹配、推理链条清晰、不需要世界知识更新
- 关键发现：针对特定目标任务选择源任务 >> 选平均性能最好的任务

**决策二：任务混合策略**
- 单一任务训练容易过拟合该任务的推理模式
- 但盲目混合会互相干扰，降低整体性能
- 最优策略是**按目标任务相关性加权混合**

**决策三：合成数据干预**
- 原始指令微调数据的答案格式常常不适合 RLVR（过长、格式不统一）
- 需要对 ground truth 做格式标准化

---

## 实现

### 关键组件一：可验证奖励函数

这是整个框架的基石。不同推理类型需要不同的验证逻辑：

```python
import re

def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip().lower()
    lines = text.strip().split('\n')
    return lines[-1].strip().lower() if lines else None

def exact_match_reward(prediction, ground_truth):
    pred = extract_answer(prediction)
    return 1.0 if pred and pred == ground_truth.strip().lower() else 0.0

def multi_choice_reward(prediction, ground_truth):
    pred = extract_answer(prediction)
    if not pred:
        return 0.0
    pred_l = re.search(r'\b([a-d])\b', pred)
    gt_l = re.search(r'\b([a-d])\b', ground_truth.lower())
    if pred_l and gt_l:
        return 1.0 if pred_l.group(1) == gt_l.group(1) else 0.0
    return exact_match_reward(prediction, ground_truth)

# ... (contains_reward 等其他奖励函数省略)

REWARD_REGISTRY = {
    "classification": exact_match_reward,
    "multiple_choice": multi_choice_reward,
    # ... (其他任务类型省略)
}
```

### 关键组件二：数据格式转换

把 instruction-tuning 格式转换为 RLVR 可用的格式：

```python
from dataclasses import dataclass

@dataclass
class RLVRSample:
    task_type: str
    prompt: str        # 包含 CoT 引导的完整提示
    ground_truth: str  # 标准答案（用于计算 reward）
    reward_fn: str     # 奖励函数名称

def convert_to_rlvr(raw_sample: dict, task_type: str) -> RLVRSample:
    # 添加 CoT + 结构化输出引导
    prompt = f"""{raw_sample["instruction"]}

请先逐步推理，然后用 <answer> 标签给出最终答案。
格式：<reasoning>你的推理过程</reasoning><answer>最终答案</answer>"""

    return RLVRSample(
        task_type=task_type,
        prompt=prompt,
        ground_truth=_normalize_answer(raw_sample["output"], task_type),
        reward_fn=_select_reward_fn(task_type),
    )

def _normalize_answer(answer: str, task_type: str) -> str:
    # ... (多选题提取字母、统一大小写等处理省略)
    return answer.strip().lower()

def _select_reward_fn(task_type: str) -> str:
    # task_type -> reward function 的映射表
    mapping = {"multiple_choice": "multiple_choice", ...}  # ... (完整映射省略)
    return mapping.get(task_type, "exact_match")
```

### 关键组件三：任务混合策略

```python
import numpy as np
from collections import defaultdict

class TaskMixer:
    """动态任务混合器：针对性选任务 >> 选平均最优任务"""
    def __init__(self, tasks: list[str], target_task: str):
        self.weights = {t: 1.0 / len(tasks) for t in tasks}
        self.task_performance = defaultdict(list)
        # ... (初始化代码省略)

    def sample_task(self) -> str:
        probs = np.array(list(self.weights.values()))
        return np.random.choice(list(self.weights.keys()), p=probs / probs.sum())

    def update_weights(self, task: str, target_reward: float):
        """根据迁移效果在线更新权重"""
        self.task_performance[task].append(target_reward)
        if len(self.task_performance[task]) >= 10:
            # 性能好的任务获得更多采样权重
            self.weights[task] = max(0.1, np.mean(self.task_performance[task][-20:]))
            total = sum(self.weights.values())
            self.weights = {t: w / total for t, w in self.weights.items()}
```

### 核心训练循环（GRPO）

```python
def grpo_update(model, optimizer, prompts, reward_fn, group_size=8, clip_eps=0.2, kl_coef=0.01):
    all_loss = []
    for prompt in prompts:
        outputs = model.generate(prompt, num_return_sequences=group_size, ...)  # ... (采样代码)
        rewards = torch.tensor([reward_fn(out, ...) for out in outputs])

        # 组内归一化优势（GRPO 核心，无需 value network）
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for out, adv in zip(outputs, advantages):
            log_prob_new = model.log_prob(prompt, out)
            log_prob_old = model.log_prob_detached(prompt, out)

            ratio = torch.exp(log_prob_new - log_prob_old)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            pg_loss = -torch.min(ratio * adv, clipped * adv).mean()
            kl = log_prob_old - log_prob_new  # KL 惩罚，防止偏离参考模型
            all_loss.append(pg_loss + kl_coef * kl.mean())

    total_loss = torch.stack(all_loss).mean()
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
    optimizer.step()
```

### 关键 Trick（论文没写清楚，但必须有）

**1. 答案格式必须严格约束**

```python
# 错误：模型输出太自由，reward 函数识别不了
prompt_bad = "请回答以下问题：..."

# 正确：强制输出格式
prompt_good = """请回答以下问题：...

你必须按以下格式回答：
<reasoning>一步步推理</reasoning>
<answer>最终答案（只写答案本身）</answer>"""
```

**2. 早期训练 reward 为 0 的处理**

```python
# 如果一个 batch 里所有输出的 reward 都是 0
# advantages 会全为 0，梯度消失，什么都学不到
if rewards.max() == 0:
    # 跳过这个 batch，或者降低判断严格度
    continue  # 早期训练时适当放宽 reward 条件
```

**3. 任务难度课程安排**

```python
# 不要一开始就混合所有任务
# 先在"容易"任务（精确匹配 reward 高）上热身，再引入难任务
warmup_tasks = ["classification", "multiple_choice"]  # 第 0-1k 步
full_tasks = warmup_tasks + ["causal_reasoning", "temporal"]  # 之后
```

---

## 实验

### 环境选择与基准

SUPERNOVA 选择的评测集能说明"真正的通用推理能力"，而不只是记忆：

| 基准 | 测什么 | 为什么选它 |
|------|--------|-----------|
| BBEH | 多步推理、逻辑归纳 | 不能靠记忆刷分 |
| ZebraLogic | 约束满足、逻辑推理 | 格式固定，易验证 |
| MMLU-Pro | 多学科综合 | 覆盖面广，工业常用 |

### 学习曲线可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_rlvr_curves(results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"SUPERNOVA": "#e74c3c", "Baseline": "#3498db"}

    for name, data in results.items():
        steps = data["steps"]
        all_rewards = np.array(data["rewards_per_seed"])  # [n_seeds, n_steps]
        mean, std = all_rewards.mean(axis=0), all_rewards.std(axis=0)
        axes[0].plot(steps, mean, label=name, color=colors[name])
        axes[0].fill_between(steps, mean - std, mean + std, alpha=0.2, color=colors[name])
    # ... (axes[0] 标签/标题设置省略)

    benchmarks = ["BBEH", "ZebraLogic", "MMLU-Pro"]
    x, width = np.arange(len(benchmarks)), 0.35
    for i, (name, scores) in enumerate(results.items()):
        axes[1].bar(x + i * width, scores["final"], width, label=name, color=colors[name], alpha=0.8)
    # ... (axes[1] 标签/标题设置省略)

    plt.tight_layout()
    plt.savefig("supernova_results.png", dpi=150)
```

### 与 Baseline 对比（论文报告结果）

| 算法 | BBEH | ZebraLogic | MMLU-Pro |
|------|------|-----------|---------|
| Qwen3.5 (SFT only) | 基线 | 基线 | 基线 |
| 随机任务混合 RLVR | +12% | +8% | +5% |
| SUPERNOVA（针对性选任务） | **+52.8%** | +31% | +18% |

关键结论：任务选择策略的差距远超算法差距。

### 消融实验

| 去掉什么 | BBEH 变化 |
|---------|---------|
| 去掉针对性任务选择 | -28% |
| 去掉格式标准化 | -15% |
| 去掉课程安排 | -8% |
| 去掉 KL 约束 | 训练崩溃 |

---

## 调试指南

### 常见问题

**1. Reward 始终为 0，训练不动**

最常见的原因是答案格式提取失败。诊断方法：

```python
# 打印前 10 个样本的模型输出，看看格式对不对
for i, (prompt, output, reward) in enumerate(zip(prompts, outputs, rewards)):
    if reward == 0:
        print(f"=== 样本 {i} ===")
        print(f"输出末尾: {output[-200:]}")
        print(f"期望答案: {get_ground_truth(prompt)}")
        print(f"提取到的: {extract_answer(output)}")
        if i > 10: break
```

最可能的问题：`<answer>` 标签没有出现，或者大小写不匹配。

**2. 训练曲线先涨后崩**

```
步骤 0-500: reward 从 0.2 涨到 0.6  ✓
步骤 500-600: reward 突然跌到 0.05  ✗
```

这是 KL 散度爆炸的经典症状：
- 检查 `kl_coef`，通常需要调大（从 0.001 到 0.01 甚至 0.1）
- 检查梯度裁剪是否生效
- 降低学习率（从 `1e-5` 降到 `5e-6`）

**3. 不同任务的 reward 差距太大导致梯度方向混乱**

```python
# 坏的做法：直接混合，难任务 reward=0 淹没容易任务的梯度
batch = sample_from_all_tasks()

# 好的做法：按任务分组更新，或者对 reward 做任务内归一化
def task_normalized_reward(reward, task_type, history):
    task_mean = np.mean(history[task_type][-100:]) if history[task_type] else 0.5
    return reward - task_mean  # 相对于该任务的历史均值
```

### 如何判断算法在"学习"

- **前 200 步**：reward 从 0 开始，应该能看到随机波动但均值缓慢上升
- **500 步**：至少一种任务类型的 reward 应该 > 0.3
- **2000 步**：如果还没有任何任务的 reward > 0.5，说明有问题
- **监控指标**：reward 均值、KL 散度、梯度范数（三个都要看）

### 超参数调优

| 参数 | 推荐起点 | 敏感度 | 调优建议 |
|------|---------|-------|---------|
| `lr` | `1e-5` | 极高 | 先试这个，崩了就降一个量级 |
| `kl_coef` | `0.01` | 高 | reward 崩就调大 |
| `group_size` | `8` | 中 | 显存不够就降到 4 |
| `clip_eps` | `0.2` | 低 | 一般不动 |
| `temperature` | `0.8` | 中 | 太低则探索不足，太高则输出混乱 |
| 任务权重更新频率 | 每 50 步 | 中 | 太频繁会抖动 |

---

## 什么时候用 / 不用 SUPERNOVA 的思路？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有大量人工标注 IT 数据，但缺乏 RLVR 格式 | 任务答案本身就无法精确验证 |
| 目标任务有清晰的 ground truth | 只关心数学/代码（直接用 RLVR 即可）|
| 多个推理任务需要同时提升 | 计算资源极其有限（多任务混合需要更多实验）|
| 不想用人工 reward model（成本高） | 目标是开放式生成（写作、对话）|

---

## 我的观点

SUPERNOVA 的贡献是**诚实的**：它没有提出新算法，而是系统性地回答了"RLVR 数据配方"这个被忽视的问题。

但有几点值得注意：

**真正有价值的发现**：针对特定目标任务选择源任务，比选"全局最优"任务效果好得多。这个结论很违反直觉，但在迁移学习、多任务学习中其实屡见不鲜。

**需要警惕的地方**：52.8% 的相对提升听起来惊人，但 BBEH 本来就是低分基准——绝对值的提升需要放在上下文里看。如果基线是 20%，提升到 30% 也是 50%。

**对实践者的建议**：如果你手头有大量领域内的 instruction-tuning 数据，SUPERNOVA 的数据配方值得一试。核心步骤就两个：

1. 不要盲目混合所有任务，先做任务相关性分析
2. 答案格式标准化比你想象的更重要

最后还是那句话：**RL 很脆弱，数据配方比算法设计更影响最终结果**。论文代码在 [https://github.com/asuvarna31/supernova](https://github.com/asuvarna31/supernova)，建议先跑通官方实现再尝试自己的数据。