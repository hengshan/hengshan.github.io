---
layout: post-wide
title: "级联强化学习：如何让大模型在扩展新领域时不忘旧知识"
date: 2026-03-21 12:03:14 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.19220v1
generated_by: Claude Code CLI
---

## 一句话总结

Cascade RL 通过**领域课程式强化学习 + 在策略蒸馏**，让 LLM 在扩展训练领域的同时防止旧领域性能回退，使 30B MoE 模型（仅 3B 激活参数）在数学和编程推理上媲美规模大 20 倍的前沿模型。

---

## 背景：多领域 RLHF 为什么很难

标准 RLHF 流程（SFT → RM → PPO）在单一领域效果尚可，但一旦你想同时优化数学推理、代码生成、工具调用等多个领域，就会遇到一个根本矛盾：

**问题一：Reward 信号相互干扰**

数学题的 reward 是形式验证（答案对不对），代码的 reward 是执行结果，工具调用的 reward 是任务完成率。这三类 reward 的量纲、稀疏程度、噪声水平都不同。直接混合训练往往导致某些领域支配梯度。

**问题二：扩展新领域时的性能回退**

这是 Nemotron-Cascade 2 重点解决的问题。当你在已有的数学/代码基础上，继续加入 agentic 领域的 RL 训练时，数学 benchmark 会下降。这不是 bug，而是 RL 优化的本质：策略往好的方向更新，但"好"的定义取决于当前训练域的 reward，其他领域的能力就悄悄退化了。

这在监督学习里叫**灾难性遗忘**，在多领域 RL 里情况更糟，因为策略分布的偏移会放大。

**Cascade RL 的核心 insight：两步走**

1. 按领域**级联式**（而非混合式）训练——先在核心领域建立强基础，再逐步扩展
2. 扩展时用**各领域最强中间检查点**作为教师，通过在策略蒸馏恢复回退的性能

---

## 算法原理

### 直觉解释

想象你在教一个学生：先把他训练成数学高手，然后让他去学编程。他的数学可能会生疏。这时候你的做法不是"重新教数学"（太贵），而是让他的数学老师（之前最好的数学检查点）**持续给他提示**，同时他在用自己生成的答案练习编程。

这就是在策略蒸馏（on-policy distillation）——学生用自己的策略采样数据，老师在这些数据上提供软标签监督。

### 数学基础

#### GRPO 强化学习目标

Cascade RL 底层用的是 Group Relative Policy Optimization（GRPO），对每个 prompt $x$，采样 $G$ 个回复 $\{o_1, ..., o_G\}$，计算相对优势：

$$
A_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}
$$

策略梯度目标（带 clip）：

$$
\mathcal{L}_{RL} = -\mathbb{E}\left[\min\left(\rho_i A_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right)\right]
$$

其中 $\rho_i = \frac{\pi_\theta(o_i \mid x)}{\pi_{\text{old}}(o_i \mid x)}$ 是重要性权重。

#### 在策略蒸馏目标

对于已经扩展到新领域时产生回退的旧领域，引入领域专属教师 $\pi_{\text{teacher}}^{(d)}$，在学生自己生成的 token 序列上计算 KL 蒸馏损失：

$$
\mathcal{L}_{distill}^{(d)} = \mathbb{E}_{o \sim \pi_\theta}\left[D_{KL}\left(\pi_{\text{teacher}}^{(d)}(\cdot \mid x, o_{<t})\ \|\ \pi_\theta(\cdot \mid x, o_{<t})\right)\right]
$$

#### 联合目标

$$
\mathcal{L}_{total} = \mathcal{L}_{RL}^{current} + \lambda \sum_{d \in D_{prev}} \mathcal{L}_{distill}^{(d)}
$$

- $\mathcal{L}_{RL}^{current}$：当前扩展领域的 RL 损失
- $\mathcal{L}_{distill}^{(d)}$：对已训练领域的蒸馏损失，防止遗忘
- $\lambda$：平衡系数，需要仔细调

**为什么必须是 on-policy？** 如果用旧领域的数据做蒸馏（off-policy），分布偏移会让 KL 估计失真，梯度信号嘈杂。用学生自己当前策略采样的数据，保证分布匹配。

### Cascade RL vs 标准多领域 RLHF

| 维度 | 标准混合 RLHF | Cascade RL |
|------|------------|------------|
| 训练方式 | 所有领域同时混合 | 领域逐步级联 |
| 遗忘问题 | 严重，无保护机制 | 通过蒸馏缓解 |
| 教师模型 | 无 | 各领域最强中间检查点 |
| 计算开销 | 低 | 中（多个教师推理） |
| 调试难度 | 高（reward 冲突） | 中（需要调 λ） |

---

## 实现

### 最小可运行版本

核心是 GRPO 损失 + 在策略蒸馏损失的组合：

```python
import torch
import torch.nn.functional as F

def grpo_loss(log_probs, old_log_probs, rewards, clip_eps=0.2):
    """
    GRPO 策略梯度损失
    log_probs: [batch, seq_len] 当前策略的 log prob
    rewards:   [batch] 每条回复的奖励（已归一化为组内相对优势）
    """
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    ratio = (log_probs - old_log_probs).sum(dim=-1).exp()  # [batch]
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    
    pg_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()
    return pg_loss

def on_policy_distill_loss(student_logits, teacher_logits, temperature=2.0):
    """
    在策略 KL 蒸馏损失
    student_logits, teacher_logits: [batch, seq_len, vocab_size]
    """
    # 在学生自己生成的序列上计算 token 级别 KL
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL(teacher || student)，让学生向教师靠拢
    kl = F.kl_div(student_log_probs, teacher_probs,
                  reduction='batchmean', log_target=False)
    return kl * (temperature ** 2)  # 温度缩放补偿
```

### 完整级联训练框架

```python
class CascadeRLTrainer:
    def __init__(self, model, reward_fns: dict, distill_weight=0.1):
        self.model = model
        self.reward_fns = reward_fns   # {'math': fn, 'code': fn, 'agent': fn}
        self.distill_weight = distill_weight
        self.teachers = {}             # domain -> teacher checkpoint
        self.domain_order = []         # 训练过的领域顺序

    def register_teacher(self, domain: str, teacher_model):
        """保存某领域最强检查点作为教师"""
        self.teachers[domain] = teacher_model
        self.teachers[domain].eval()
        for p in self.teachers[domain].parameters():
            p.requires_grad_(False)

    def train_step(self, current_domain: str, prompts, generated):
        """
        一步级联 RL 训练
        prompts:   List[str]，当前领域的输入
        generated: List[str]，模型生成的回复（已采样）
        """
        # 1. 计算当前领域 RL 损失
        rewards = self.reward_fns[current_domain](prompts, generated)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        log_probs = self.model.compute_log_probs(prompts, generated)
        with torch.no_grad():
            old_log_probs = log_probs.detach()

        rl = grpo_loss(log_probs, old_log_probs, rewards)

        # 2. 对已训练领域做在策略蒸馏（防止遗忘）
        distill_total = torch.tensor(0.0)
        student_logits = self.model.get_logits(prompts, generated)

        for domain, teacher in self.teachers.items():
            with torch.no_grad():
                teacher_logits = teacher.get_logits(prompts, generated)
            distill_total = distill_total + on_policy_distill_loss(
                student_logits, teacher_logits
            )

        total_loss = rl + self.distill_weight * distill_total
        return total_loss, {'rl': rl.item(), 'distill': distill_total.item()}
```

### 领域课程调度器

```python
class DomainCurriculum:
    """
    决定何时切换到下一个领域，何时保存教师检查点
    """
    def __init__(self, domains: list, switch_threshold=0.85, patience=500):
        self.domains = domains          # ['math', 'code', 'agent']
        self.threshold = switch_threshold   # 达到多少胜率才推进
        self.patience = patience
        self.current_idx = 0
        self.step_count = 0
        self.best_score = 0.0

    def should_save_teacher(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            return True
        return False

    def should_advance(self, score: float) -> bool:
        """连续 patience 步超过阈值才切换领域"""
        self.step_count += 1
        if score >= self.threshold and self.step_count >= self.patience:
            self.current_idx = min(self.current_idx + 1, len(self.domains) - 1)
            self.step_count = 0
            self.best_score = 0.0
            return True
        return False

    @property
    def current_domain(self) -> str:
        return self.domains[self.current_idx]
```

### 关键 Trick

**1. 教师选择策略**

不要用最后一个检查点，要用该领域 **benchmark 得分最高的检查点**。这需要在训练过程中持续评估并保存，增加计算开销，但效果差距明显。

**2. 蒸馏权重 λ 的调法**

```python
# 观察 rl_loss 和 distill_loss 的量级
# 目标：两者量级相近，不要让蒸馏压制 RL 梯度

# 诊断脚本：打印梯度范数比
rl_grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
# 如果 distill 梯度范数 >> rl 梯度范数，减小 λ
```

**3. 温度参数对蒸馏的影响**

温度 $T > 1$ 会软化教师分布，让学生更容易学到"次优选项"的分布信息。对于 reasoning 任务，`T=1.5~3.0` 通常比 `T=1.0` 效果更好。

**4. 在策略 vs 离策略蒸馏的区别**

| | 在策略蒸馏 | 离策略蒸馏 |
|--|---------|---------|
| 采样来源 | 学生当前策略 | 固定数据集 |
| 分布对齐 | 总是对齐 | 策略偏移后失效 |
| 计算成本 | 需要在线生成 | 可预计算 |
| 效果 | 更稳定 | 训练后期效果差 |

---

## 调试指南

### 常见问题

**1. 新领域 RL 学不动**

症状：reward 长期停在 0 附近，不上升。

可能原因：
- Reward 函数设计错误（先用少量样本手动验证 reward）
- 生成长度被截断，导致答案不完整无法验证
- `clip_eps` 太小，policy 更新太保守，可以先试 `clip_eps=0.3`

**2. 旧领域性能剧烈下降**

症状：扩展到代码领域后，数学 benchmark 掉了 10+ 点。

可能原因：
- 蒸馏权重 λ 太小，蒸馏损失被 RL 损失淹没
- 教师模型选错了（不是该领域最强点，而是最后检查点）
- 学习率太大，策略偏移过快

快速诊断：
```python
# 每 100 步评估一次所有领域的 benchmark
# 画出各领域得分曲线，观察何时开始下降
for domain in all_domains:
    score = evaluate(model, domain, n_samples=50)
    writer.add_scalar(f'eval/{domain}', score, global_step)
```

**3. RL loss 和 distill loss 相互冲突**

症状：loss 剧烈震荡，reward 不稳定。

这通常是蒸馏把模型往旧分布拉，而 RL 把模型往新 reward 推，两者梯度方向相反。

解决：用 **渐进式 λ 衰减**——新领域刚开始时 λ 大（强保护旧能力），随训练推进逐渐减小 λ，让 RL 信号主导。

### 超参数调优

| 参数 | 推荐起点 | 敏感度 | 说明 |
|------|--------|------|-----|
| clip_eps | 0.2 | 中 | 太小学不动，太大不稳定 |
| λ (蒸馏权重) | 0.05~0.2 | 高 | 最需要调的参数 |
| temperature | 2.0 | 中 | reasoning 任务用高一点 |
| 学习率 | 1e-6 | 极高 | LLM RL 阶段要比 SFT 小 10x |
| 组大小 G | 8 | 低 | 越大 advantage 估计越准，越慢 |

### 如何判断 Cascade RL 在"正常工作"

- 当前领域 reward 持续上升（不是一直震荡）
- 旧领域 benchmark **不超过 3 个点的下降**（完全不掉是不现实的）
- 蒸馏 loss 随时间缓慢下降（不是爆炸）
- Reward 在多个随机种子下趋势一致（至少跑 3 个种子）

---

## 什么时候用 / 不用

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要同时优化 3+ 个不同领域 | 单一领域微调（用标准 RLHF 就够） |
| 已有各领域强检查点可用作教师 | 没有可靠的 reward 函数 |
| 计算资源支持多教师并行推理 | 计算预算极其紧张 |
| 需要严格控制旧能力不回退 | 任务之间高度相关（遗忘不严重） |
| MoE 架构（稀疏激活，教师推理便宜） | Dense 模型且教师推理成本高 |

---

## 我的观点

Nemotron-Cascade 2 在技术上最有意思的地方，不是模型规模（30B MoE），而是**用 on-policy 蒸馏解决多领域 RL 的灾难性遗忘**这个工程决策。

它本质上是一种"持续强化学习"方案——领域课程 + 知识蒸馏。这在理念上并不新，但能工程化地在 IMO、IOI、ICPC 这三个级别的任务上同时取得好成绩，说明实现细节的抠法到位了。

几点诚实的评价：

1. **教师选择是个隐藏成本**：需要在训练过程中持续评估所有中间检查点，这意味着训练周期内需要频繁评估，实际 wall-clock 时间比论文描述的更长。

2. **超参数依赖问题没有解决**：λ 怎么调、什么时候切换领域、温度设多少——这些仍然是玄学，论文里给的消融实验也只是在他们的设置下有效。

3. **复现难度不低**：MoE 架构的 reward shaping 本身就复杂，再加上多教师推理，实际跑起来的调试工作量相当大。如果你只是想验证 on-policy 蒸馏防遗忘这个 idea，建议先在小 Dense 模型（如 Qwen2.5-7B）上复现核心 ablation。

如果你在做多任务 LLM 后训练，Cascade RL 的思路值得借鉴；如果只是在单个领域做 RLHF，用 PPO/GRPO 就足够，不需要引入这层复杂性。