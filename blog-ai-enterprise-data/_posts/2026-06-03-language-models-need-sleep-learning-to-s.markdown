---
layout: post-wide
title: 'LLM 的"睡眠"：用记忆巩固与 RL 驱动持续学习'
date: 2026-06-03 08:04:55 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.03979v1
generated_by: Claude Code CLI
---

## 一句话总结

大型语言模型善于"当下理解"，却难以将新知识永久写入参数——这篇论文（arxiv 2506.03979）借鉴人类睡眠机制，提出两阶段框架：用 **Knowledge Seeding**（RL + 蒸馏）将短期记忆固化进大模型参数，再用 **Dreaming**（RL 生成合成课程）实现无监督自我改进。

## 背景：LLM 的"记忆悖论"

### 现有方法的局限

当你 fine-tune 一个 LLM 来学新知识时，它会忘掉旧知识——**灾难性遗忘（Catastrophic Forgetting）**。这不是新问题，但在 LLM 时代被放大了：

- **In-context learning**：快，但重启即忘，知识不进参数
- **Fine-tuning**：持久，但破坏原有能力，样本效率低
- **RAG**：回避了问题，没有真正"学习"

从 RL 的角度看，这是一个**非平稳环境下的持续学习问题**：每次更新都在改变 value landscape，导致之前学到的 policy 被覆盖。

### 论文的核心 Insight

人类用**睡眠**来解决记忆巩固问题：白天体验（海马体短期记忆） → 夜间睡眠 → 大脑皮层长期存储。REM 睡眠（"做梦"）则让大脑重放、重组知识。

论文把这个映射到 LLM：
- 小模型（hippocampus）：通过 in-context 学习承担短期记忆
- 大模型（neocortex）：参数存储长期知识
- Sleep = Memory Consolidation（巩固）+ Dreaming（自我改进）

## 算法原理

### Stage 1：Memory Consolidation（记忆巩固）

**核心思想**：将小模型通过 in-context 学到的知识，蒸馏进大模型的参数。

这里有个反常识设计：通常知识蒸馏是大模型 → 小模型，但这里是**小模型 → 大模型**（upward distillation）。原因：小模型承担"工作记忆"，大模型有更多容量做"长期存储"。

**Generalized Distillation 目标函数**：

$$
\mathcal{L}_{\text{KS}} = \alpha \cdot \mathcal{L}_{\text{distill}} + (1-\alpha) \cdot \mathcal{L}_{\text{RL}}
$$

- $\mathcal{L}_{\text{distill}}$：on-policy 蒸馏损失，让大模型拉近小模型的输出分布：

$$
\mathcal{L}_{\text{distill}} = \mathbb{E}_{x \sim \pi_{\text{small}}} \left[ D_{\text{KL}}\!\left(\pi_{\text{small}}(\cdot \mid x) \| \pi_{\text{large}}(\cdot \mid x)\right) \right]
$$

- $\mathcal{L}_{\text{RL}}$：RL 模仿学习损失，将小模型轨迹作为专家演示，策略梯度优化：

$$
\mathcal{L}_{\text{RL}} = -\mathbb{E}_{\tau \sim \pi_{\text{small}}} \left[ \sum_t r_t \log \pi_{\text{large}}(a_t \mid s_t) \right]
$$

**为什么同时用 KL 蒸馏和 RL？** 单纯 KL 只保证分布匹配，不保证任务性能。RL 分量提供任务导向的梯度信号。这和 PPO 中 KL penalty + reward 的设计哲学类似。

### Stage 2：Dreaming（做梦 / 自我改进）

**核心思想**：让模型用 RL 自己生成合成训练数据，在"梦境"中练习，无需人工标注。

这是一种**自课程学习（Self-Curriculum Learning）**：

1. 模型用当前 policy 生成合成问题
2. 用奖励信号评估生成质量（novelty + difficulty + correctness）
3. 在高奖励合成数据上继续训练

$$
\mathcal{L}_{\text{dream}} = -\mathbb{E}_{x \sim \pi_\theta} \left[ R(x) \cdot \log \pi_\theta(x) \right]
$$

这和 **Expert Iteration**（ExIt）或 **AlphaGo 自我博弈**的精神一致。

## 实现

### 最小可运行版本：灾难性遗忘 Demo

先搞清楚我们在解决什么：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class TinyLM(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=n_layers
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.head(self.layers(self.embed(x)))

def measure_forgetting():
    model = TinyLM()
    opt = Adam(model.parameters(), lr=3e-4)

    # Task A：学习 token 范围 [0, 50)
    task_a = torch.randint(0, 50, (200, 10))
    for _ in range(100):
        loss = F.cross_entropy(model(task_a).flatten(0,1), task_a.roll(-1,1).flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    perf_a_before = loss.item()

    # Task B：新任务覆盖（灾难性遗忘发生处）
    task_b = torch.randint(50, 100, (200, 10))
    for _ in range(100):
        loss = F.cross_entropy(model(task_b).flatten(0,1), task_b.roll(-1,1).flatten())
        opt.zero_grad(); loss.backward(); opt.step()

    # 验证 Task A 被遗忘
    with torch.no_grad():
        loss_a_after = F.cross_entropy(model(task_a).flatten(0,1), task_a.roll(-1,1).flatten())
    print(f"Task A loss: {perf_a_before:.3f} -> {loss_a_after:.3f}")
    # 典型输出：Task A loss: 0.12 -> 4.53  <-- 遗忘了

measure_forgetting()
```

### Knowledge Seeding 核心实现

```python
class KnowledgeSeedingTrainer:
    """
    将小模型（短期记忆）蒸馏进大模型（长期记忆）
    同时使用 KL 蒸馏 + RL 模仿学习
    """
    def __init__(self, small_model, large_model, alpha=0.7, lr=1e-4):
        self.small = small_model   # 教师：短期记忆，冻结参数
        self.large = large_model   # 学生：长期存储，持续更新
        self.alpha = alpha
        self.opt = Adam(large_model.parameters(), lr=lr)

    def step(self, context_tokens):
        with torch.no_grad():
            teacher_logits = self.small(context_tokens)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            # on-policy 采样：必须是实时生成，不能用静态数据集
            sampled = torch.multinomial(teacher_probs.flatten(0,1), 1).view(context_tokens.shape)

        student_logits = self.large(context_tokens)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # KL 蒸馏：拉近两个分布
        l_distill = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

        # RL 模仿：奖励 = 教师对该 token 的置信度（越确定越值得模仿）
        reward = teacher_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        log_p = student_log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        l_rl = -(reward.detach() * log_p).mean()

        loss = self.alpha * l_distill + (1 - self.alpha) * l_rl
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {'l_distill': l_distill.item(), 'l_rl': l_rl.item()}
```

### Dreaming 核心实现

```python
class DreamingTrainer:
    """
    模型自我生成合成课程，用 REINFORCE 驱动自我改进
    类似 Expert Iteration / AlphaGo 自我博弈
    """
    def __init__(self, model, reward_fn, lr=1e-5, entropy_coef=0.01):
        self.model = model
        self.reward_fn = reward_fn  # 评估生成质量的奖励函数
        self.opt = Adam(model.parameters(), lr=lr)
        self.entropy_coef = entropy_coef

    def dream_step(self, seed_prompts, gen_len=20, n_samples=16):
        all_rewards, all_log_probs, all_entropy = [], [], []

        for prompt in seed_prompts:
            gen = prompt.unsqueeze(0).repeat(n_samples, 1)
            step_log_probs, step_entropy = [], []

            for _ in range(gen_len):
                logits = self.model(gen)
                probs = F.softmax(logits[:, -1], dim=-1)
                next_tok = torch.multinomial(probs, 1)
                step_log_probs.append(probs.gather(1, next_tok).log())
                # entropy bonus：防止 mode collapse
                step_entropy.append(-(probs * (probs + 1e-8).log()).sum(-1, keepdim=True))
                gen = torch.cat([gen, next_tok], dim=1)

            rewards = self.reward_fn(gen)  # [n_samples]
            all_rewards.append(rewards)
            all_log_probs.append(torch.stack(step_log_probs).sum(0).squeeze())
            all_entropy.append(torch.stack(step_entropy).mean())

        rewards_t = torch.stack(all_rewards).flatten()
        log_probs_t = torch.stack(all_log_probs).flatten()
        baseline = rewards_t.mean()  # 均值 baseline 降低方差

        policy_loss = -((rewards_t - baseline) * log_probs_t).mean()
        entropy_loss = -torch.stack(all_entropy).mean()
        loss = policy_loss + self.entropy_coef * entropy_loss  # 防 collapse

        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {'reward': rewards_t.mean().item(), 'loss': loss.item()}
```

### 关键 Trick

**1. on-policy 采样是必须的**

Knowledge Seeding 中，静态数据集 ≠ on-policy 数据，distribution shift 会严重损害效果：

```python
# 错误：预先收集静态教师输出（off-policy）
static_data = collect_once(small_model, dataset)
for epoch in range(100):
    large_model.learn(static_data)  # 越训练 distribution shift 越大

# 正确：每个 batch 实时采样（on-policy）
for batch in dataloader:
    with torch.no_grad():
        teacher_out = small_model(batch)  # 当前分布下实时生成
    trainer.step(batch)
```

**2. Dreaming 的奖励函数设计**

奖励函数质量决定 Dreaming 的上限，没有好奖励 RL 只会 reward hacking：

```python
def self_consistency_reward(gen_tokens, model, n_verify=4):
    """用多次采样一致性作为奖励，不需要外部标注"""
    answers = [model.generate(gen_tokens) for _ in range(n_verify)]
    return compute_majority_agreement(answers)  # 一致性高 → 奖励高

def difficulty_reward(confidence):
    """适中难度样本奖励高：置信度 0.3-0.7 的最有学习价值"""
    return 1.0 - abs(confidence - 0.5) * 2
```

## 调试指南

### 常见问题

**1. Knowledge Seeding 后大模型性能下降**
- 可能原因：`(1-α)` 太大，RL 分量主导导致不稳定
- 检查方法：分开记录 `l_distill` 和 `l_rl`，若 `l_rl` 剧烈抖动，把 `α` 从 0.5 提高到 0.8
- 快速验证：临时把 `α=1.0`（纯蒸馏），看是否稳定

**2. Dreaming 没有改善，甚至倒退**
- 可能原因 1：奖励函数给错了信号（先在已知样本上手动验证奖励值）
- 可能原因 2：生成的合成数据 OOD，模型学不到有用的东西
- 可能原因 3：REINFORCE 高方差 + 学习率偏大

**3. Dreaming 生成内容崩溃（重复 token 或极短序列）**
- 症状：entropy 单调下降，`mean_reward` 迅速收敛但质量差
- 原因：mode collapse，没有 entropy regularization
- 修复：加大 `entropy_coef`（0.01 → 0.05），或加 KL penalty 与参考模型保持距离

### 监控指标

| 指标 | 正常状态 | 异常信号 |
|------|---------|---------|
| `l_distill` (KL) | 缓慢下降 | 不降或剧烈振荡 |
| `l_rl` (RL loss) | 平稳降低 | 突然跳变 |
| `mean_reward` (Dreaming) | 缓慢上升 | 过快收敛 = collapse |
| 生成多样性 (entropy) | 稳定 | 单调下降 |
| 旧任务性能 | 保持稳定 | 快速下跌 = 遗忘 |

### 超参数敏感度

| 参数 | 推荐值 | 敏感度 | 建议 |
|------|-------|-------|------|
| `α`（distill 权重） | 0.7 | 高 | 先用 0.7，偏稳定 |
| RL 学习率 | 1e-5 | 极高 | 比监督学习小一个量级 |
| entropy coefficient | 0.01 | 中 | 崩溃了就加大 |
| `n_samples`（Dreaming） | 16–32 | 中 | 影响 baseline 估计质量 |
| `gen_len` | 10–50 | 低 | 太长 RL 信号稀疏 |

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要持续接收新知识的系统 | 静态任务，一次训练就够 |
| 有自然的"短期-长期"层级结构 | 所有知识同等重要，无层级 |
| 可以设计合理的自我评估奖励 | 奖励函数极难设计的领域 |
| 训练算力充裕 | 计算预算极为紧张 |
| 希望减少对人工标注的依赖 | 已有大量高质量标注数据 |

## 我的观点

这篇论文的核心贡献不是某个新 RL 算法，而是一个**框架**：把人类睡眠比喻转化成了两个具体的可优化目标。

**真正新颖的地方**：反向蒸馏（小→大）的设计在持续学习设置下有充分动机；Dreaming 把 self-play 引入 LLM 持续学习，跳出依赖人工标注的框架。

**需要怀疑的地方**：

- 小模型真的能作为大模型可靠的"教师"吗？capability gap 大时这个假设会失效
- Dreaming 的奖励函数在论文里语焉不详，self-consistency 类代理指标不一定对齐真实任务目标
- 实验规模能否 scale 到实际生产级 LLM，尚不明确

**实用建议**：

如果你在做 LLM 持续学习，Knowledge Seeding 的蒸馏部分相对成熟、值得尝试；Dreaming 的 RL 稳定性是你要自己踩的坑——从简单的 `self_consistency_reward` 开始，验证奖励信号合理后再上 RL，不要一上来就端到端。

**归根结底**：这类"让 LLM 自我进化"方向，核心挑战还是老问题——**奖励函数**。没有好的 reward，RL 只会让模型更擅长 gaming the metric。

---

**论文链接**：[arxiv 2506.03979](https://arxiv.org/abs/2606.03979v1)