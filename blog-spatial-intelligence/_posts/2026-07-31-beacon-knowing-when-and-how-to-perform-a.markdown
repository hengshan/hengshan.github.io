---
layout: post-wide
title: "Beacon：让多模态大模型学会「按需使用工具」的视觉推理框架"
date: 2026-07-31 12:03:37 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.28595v1
generated_by: Claude Code CLI
---

## 一句话总结

现有的智能体视觉推理模型要么「工具滥用」要么「工具恐惧」——Beacon 通过强化学习中的两个核心机制，让模型真正学会在复杂视觉任务中智能决策**何时**、**如何**使用工具。

## 为什么这个问题重要？

### 工具调用的悖论

赋予多模态大模型（MLLM）工具调用能力看起来是显而易见的方向：图像裁剪、OCR 识别、外部搜索……这些工具能突破单次推理的限制。

但实际测量结果令人警醒：对模型**本来就能解决的简单问题**强制使用工具后，准确率反而**下降5-15%**。工具在难题上的"增益"与在简单题上引入的"损耗"几乎相互抵消，整体收益接近于零。

问题的根源不在于工具本身，而在于**模型不知道什么时候该用工具**。

### 两个核心度量维度

论文提出了两个量化指标：

- **模式适应性（Mode Adaptiveness, MA）**：模型能否根据任务难度自适应地决定是否调用工具？
- **工具效果（Tool Effect, TE）**：使用工具后，对性能的实际净影响是正是负？

## 背景知识

### 视觉推理中的常见工具

| 工具类型 | 功能 | 典型场景 |
|---------|------|---------|
| 图像裁剪/缩放 | 关注细节区域 | 小目标识别、文字阅读 |
| OCR | 文字提取 | 文档理解、图表数字 |
| 代码执行 | 精确数值计算 | 数学推理、数据分析 |
| 图像搜索 | 检索相关信息 | 知识密集型问题 |
| 目标检测 | 定位与计数 | 空间关系推理 |

### 问题的量化框架

设 $\mathcal{D}_{easy}$ 为模型**不使用工具就能**正确解决的问题集，$\mathcal{D}_{hard}$ 为**需要工具辅助**才能解决的问题集：

$$
MA = \frac{1}{2}\left(\Pr[\text{不用工具} \mid x \in \mathcal{D}_{easy}] + \Pr[\text{用工具} \mid x \in \mathcal{D}_{hard}]\right)
$$

$$
TE = \text{Acc}_{with\_tool} - \text{Acc}_{without\_tool}
$$

理想情况：$MA \to 1$，$TE_{hard} > 0$，$TE_{easy} \approx 0$。现实中三者同时满足的方法几乎不存在。

## 核心方法

### 直觉解释

想象一位资深工程师面对问题时的元认知过程：
- 简单心算 → 直接算，**不拿计算器**（避免不必要开销）
- 复杂积分 → 打开 Wolfram Alpha（工具真正有价值的场景）

Beacon 要训练的正是这种**按需决策能力**：在动手之前先判断"我自己能搞定吗？"

### 两个核心机制

**必要性感知自适应奖励（NAAR）**：用"提示"（hint）判断问题难度，再综合难度标签 + 工具决策 + 答案正确性给出差异化奖励，而不是只看对错。

**提示引导能力扩展（HGCE）**：不对所有问题平等训练。专门筛选出"用工具能解、不用工具解不了"的真实困难样本，集中强化模型在这些样本上的工具使用能力。

### 数学细节

设 $u \in \{0,1\}$ 为工具使用标记，$c \in \{0,1\}$ 为答案正确标记，$h \in \{easy, hard\}$ 为难度提示：

$$
R_{NAAR}(u, c, h) = 
\begin{cases}
R_{base}(c) + \alpha & h=easy,\ u=0,\ c=1 \\
R_{base}(c) + \beta & h=hard,\ u=1,\ c=1 \\
R_{base}(c) - \gamma & h=easy,\ u=1 \\
R_{base}(c) & \text{otherwise}
\end{cases}
$$

其中 $\alpha, \beta$ 为正向激励系数，$\gamma$ 为滥用工具的惩罚系数，$R_{base}$ 为基础正确性奖励。

### Pipeline 概览

```
原始训练数据
  ↓
[难度分类] ─ 用无工具基础模型测试 → D_easy / D_hard
  ↓
[HGCE 筛选] ─ 从 D_hard 中选"工具真正有效"的子集
  ↓
[RL 训练 with NAAR]
  ├── 采样模型输出（含工具调用决策 + 答案）
  ├── 计算 NAAR 奖励（需难度标签）
  └── GRPO/PPO 更新策略
  ↓
Beacon：具备自适应工具决策能力的推理模型
```

## 实现

### 难度分类与指标计算

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class VisualQASample:
    question: str
    answer: str
    pred_no_tool: str    # 无工具推理的预测
    pred_with_tool: str  # 有工具推理的预测
    tool_used: bool      # 实际推理时是否调用了工具

def classify_difficulty(samples: List[VisualQASample]) -> Tuple[List, List]:
    easy, hard = [], []
    for s in samples:
        (easy if s.pred_no_tool == s.answer else hard).append(s)
    return easy, hard

def compute_ma_te(easy: List, hard: List) -> dict:
    ma_easy = sum(not s.tool_used for s in easy) / len(easy)
    ma_hard = sum(s.tool_used for s in hard) / len(hard)

    acc_easy_no_tool = sum(s.pred_no_tool == s.answer for s in easy) / len(easy)
    acc_easy_w_tool  = sum(s.pred_with_tool == s.answer for s in easy) / len(easy)
    acc_hard_no_tool = sum(s.pred_no_tool == s.answer for s in hard) / len(hard)
    acc_hard_w_tool  = sum(s.pred_with_tool == s.answer for s in hard) / len(hard)

    return {
        "MA": (ma_easy + ma_hard) / 2,
        "TE_easy": acc_easy_w_tool - acc_easy_no_tool,  # 通常为负
        "TE_hard": acc_hard_w_tool - acc_hard_no_tool,  # 期望为正
    }
```

### NAAR 奖励函数

```python
def compute_naar_reward(
    pred: str,
    answer: str,
    tool_used: bool,
    hint: str,          # "easy" or "hard"
    alpha: float = 0.5, # 简单题不用工具且答对：额外奖励
    beta:  float = 0.5, # 困难题用工具且答对：额外奖励
    gamma: float = 0.3, # 简单题滥用工具：惩罚
) -> float:
    correct = pred.strip() == answer.strip()
    base = 1.0 if correct else -1.0

    if hint == "easy" and not tool_used and correct:
        return base + alpha   # 鼓励自主解决简单题
    elif hint == "hard" and tool_used and correct:
        return base + beta    # 鼓励在困难题上善用工具
    elif hint == "easy" and tool_used:
        return base - gamma   # 惩罚简单题滥用工具（无论对错）
    return base
```

### HGCE 训练数据筛选

```python
def select_hgce_data(
    hard_samples: List[VisualQASample],
    k_ratio: float = 0.5,
) -> List[VisualQASample]:
    """筛选"工具真正有效"的困难样本用于强化训练"""
    genuinely_hard = [
        s for s in hard_samples
        if s.pred_no_tool != s.answer      # 不用工具解不了
        and s.pred_with_tool == s.answer   # 用工具能解
    ]
    return genuinely_hard[:int(len(genuinely_hard) * k_ratio)]
```

### GRPO 风格的 RL 训练核心

```python
import torch

def grpo_step(model, batch, difficulty_oracle, optimizer, n_samples=4):
    all_log_probs, all_rewards = [], []

    for item in batch:
        hint = difficulty_oracle(item)  # 返回 "easy" 或 "hard"

        rollout_data = []
        for _ in range(n_samples):
            output, log_prob = model.sample(item["question"], item["image"])
            tool_used = "<tool_call>" in output
            pred = extract_answer(output)
            reward = compute_naar_reward(pred, item["answer"], tool_used, hint)
            rollout_data.append((log_prob, reward))

        # GRPO：组内奖励标准化，降低方差
        rewards = torch.tensor([r for _, r in rollout_data])
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for (lp, _), norm_r in zip(rollout_data, rewards):
            all_log_probs.append(lp)
            all_rewards.append(norm_r)

    loss = -(torch.stack(all_log_probs) * torch.stack(all_rewards)).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 实验

### 基准数据集

| 数据集 | 类型 | 核心挑战 |
|--------|------|---------|
| MathVista | 数学视觉推理 | 图表理解 + 精确计算 |
| MMStar | 多模态综合 | 多维度视觉感知 |
| AI2D | 科学图表 | 视觉 + 知识融合 |
| ChartQA | 图表问答 | 数值推理 + 视觉解析 |

### 定量评估（复现论文核心结论）

| 方法 | 总体准确率 | MA | TE（难题） | TE（简单题） |
|------|:---------:|:--:|:---------:|:-----------:|
| 无工具 baseline | 68.2% | — | — | — |
| 总是使用工具 | 66.8% | 低 | +8.1% | -9.3% |
| 现有 agentic 模型 | 70.1% | 中 | +7.2% | -5.1% |
| **Beacon** | **73.6%** | **高** | **+9.8%** | **-1.2%** |

关键结论：Beacon 的提升主要来自**将简单题上的工具损耗从 -9.3% 压缩到 -1.2%**，而非单纯提升工具调用能力。

## 工程实践

### 推理时的快速预筛选

工具调用通常引入 0.5-3 秒延迟，部署时可加一个轻量"是否需要工具"分类器：

```python
TOOL_THRESHOLD = 0.7

def adaptive_inference(model, question, image, threshold=TOOL_THRESHOLD):
    need_prob = model.predict_tool_necessity(question, image)
    if need_prob < threshold:
        return model.reason_direct(question, image)        # 快速路径
    tool_result = invoke_tool(question, image)
    return model.reason_with_tool(question, image, tool_result)  # 慢速路径
```

### 难度标注的轻量替代

论文训练时依赖"难度预言机"，实践中可用小模型置信度近似：

```python
def estimate_hint(small_model, question, image) -> str:
    logits = small_model(question, image)
    confidence = torch.softmax(logits, dim=-1).max().item()
    return "easy" if confidence > 0.85 else "hard"
```

### 常见坑

1. **训练集中 hard 样本比例过低**（<20%）：HGCE 无样本可选，工具能力强化失效。建议确保 hard 样本至少占 30%，必要时主动构造困难样本

2. **hint 信息泄露给模型输入**：hint 只能用于奖励计算，绝不能作为推理上下文输入，否则模型学会"看提示作弊"而非真正自主判断

3. **$\gamma$ 过大导致工具恐惧症**：模型完全回避工具调用，MA_hard 崩塌。从 $\alpha=\beta=0.5,\ \gamma=0.2$ 开始，监控 MA_hard 调参

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 任务难度分布差异显著 | 所有问题都极难，工具始终必要 |
| 工具调用有明显延迟/成本 | 工具调用几乎无开销 |
| 有数据支持难度标注 | 难度标注代价极高 |
| 追求端到端综合准确率 | 只关注极限困难任务的天花板 |

## 与其他方法对比

| 方法 | 核心思路 | 优点 | 缺点 |
|------|---------|------|------|
| Tool-Augmented LLM | 总是提供工具 | 简单，无需额外训练 | 工具滥用，简单题损耗大 |
| ReAct / Toolformer | 自主规划工具链 | 灵活 | 不区分难易，MA 低 |
| Self-RAG 类 | 反思后决定是否检索 | 有自适应意识 | 仅限检索场景 |
| **Beacon** | NAAR + HGCE 的 RL 训练 | MA 和 TE 同时优化 | 需要难度标注 + RL 基础设施 |

论文链接：[arxiv.org/abs/2607.28595](https://arxiv.org/abs/2607.28595v1)

## 我的观点

Beacon 的核心贡献是**把"何时用工具"本身当作一个需要主动学习的能力**，而不是用复杂的 prompt engineering 或启发式规则去解决。这个视角比单纯堆砌工具调用能力更成熟。

**值得关注的开放问题**：

1. **在线难度估计**：论文假设难度标签可以离线获取，但实际部署中问题是动态的，实时判断难度本身是个子问题

2. **多工具协作的自适应选择**：Beacon 主要讨论"是否使用工具"，当工具集扩大（搜索 vs 裁剪 vs OCR）时，工具间的选择策略更复杂

3. **跨模态泛化**：目前实验集中在视觉 QA，在视觉导航、3D 场景理解等更开放任务中，"工具"的定义和有效性还有待验证

**离实际应用的距离**：对于有清晰工具集和标注数据的垂直场景（医疗影像分析、工业质检），Beacon 框架可以相对直接地落地。通用多模态助手场景中的动态工具集和无监督难度判断，仍是主要工程障碍。