---
layout: post-wide
title: "非鲁棒教师也能教出鲁棒性：测试时对抗适应新范式"
date: 2026-04-14 08:05:12 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.11590v1
generated_by: Claude Code CLI
---

## 一句话总结

不需要标签，不需要鲁棒预训练——仅凭无标签目标域数据，让一个普通预训练模型在测试时适应出对抗鲁棒性。

## 为什么这个问题重要

现实部署中，模型同时面临两种挑战：

**域偏移**：训练和测试数据分布不同（光照变化、相机参数差异、传感器噪声）。测试时适应（Test-Time Adaptation, TTA）已经很好地解决了干净准确率问题。

**对抗攻击**：精心设计的扰动可以轻易骗过模型。但现有 TTA 工作几乎完全忽视了这一点。

**现有方法的尴尬处境**：
- 标准对抗训练（AT）需要标签——测试时没有
- 鲁棒预训练代价高，且不一定适合目标域
- 直接将蒸馏和对抗训练结合，效果不稳定

这篇论文提出的问题很实际：**只用无标签目标域数据，能否在部署后提升鲁棒性？**

## 背景知识

### 测试时适应（TTA）

TTA 的基本思路：模型在测试时，看到目标域的少量无标签样本，就地更新自身参数以适应新分布。

```
预训练模型 f_θ → 目标域无标签样本 {x_1, ..., x_n} → 适应后模型 f_θ'
```

常见的 TTA 方法（TENT、TTT）通常最小化目标域上的预测熵，让模型对自己更"自信"。但熵最小化完全不考虑对抗鲁棒性。

### 对抗训练基础

标准对抗训练（PGD-AT）解的是一个 min-max 问题：

$$\min_\theta \mathbb{E}_{(x,y)\sim\mathcal{D}} \left[ \max_{\|\delta\|_\infty \leq \epsilon} \mathcal{L}(f_\theta(x+\delta), y) \right]$$

TRADES 把目标分解为干净准确率 + 鲁棒性正则化：

$$\min_\theta \mathbb{E}_{(x,y)} \left[ \mathcal{L}_{CE}(f_\theta(x), y) + \beta \max_{\|\delta\|\leq\epsilon} \text{KL}(f_\theta(x+\delta) \| f_\theta(x)) \right]$$

TRADES 的鲁棒性项——`KL(f_θ(x+δ) || f_θ(x))`——叫**自一致性正则化**：对抗样本的预测应和干净样本预测保持一致。

### 为什么把 TRADES 搬到 TTA 场景会崩？

把标签换成教师预测，直接迁移 TRADES：

**问题一**：当教师 $f_T$ 本身不鲁棒时，$f_T(x+\delta)$ 和 $f_T(x)$ 差别悬殊，提供错误的监督信号。

**问题二**：自一致性项 `KL(f_θ(x+δ) || f_θ(x))` 的内层攻击目标依赖 $f_\theta(x)$，而 $f_\theta$ 在适应过程中持续更新——这造成**目标漂移（target drift）**，PGD 每步的梯度方向都对着一个"移动的靶心"，训练极不稳定。

## 核心方法

### 直觉解释

论文的核心 idea 用一句话说：**把教师的干净预测当作固定语义锚点，无论干净路径还是对抗路径，都向这个锚点对齐。**

```
x ──── 教师 f_T（冻结） ──→ anchor = softmax(f_T(x))  ← 永远不变！
                                        ↑                ↑
x ──── 学生 f_θ ──────→ p_clean    KL↗              KL↗
x+δ ── 学生 f_θ ──────→ p_adv ─────────────────────
```

与自一致性对比：

- **自一致性**：p_adv 对齐 p_clean（两个目标都在变 → 不稳定）
- **语义锚点**：p_adv 对齐 anchor（目标固定 → 稳定）

### 数学细节

**提出的语义锚点损失**：

$$\mathcal{L}_{\text{anchor}}(\theta; x) = \underbrace{\text{KL}(f_T(x) \| f_\theta(x))}_{\text{干净对齐}} + \beta \underbrace{\max_{\|\delta\|\leq\epsilon} \text{KL}(f_T(x) \| f_\theta(x+\delta))}_{\text{对抗对齐（目标固定！）}}$$

其中 $f_T(x) = \text{softmax}(g_T(x))$ 是教师在干净样本上的软标签，在整个优化过程中保持不变。

**为什么稳定？** 内层最大化的目标 $f_T(x)$ 是常数，PGD 每步的梯度方向不受模型参数更新影响，攻击方向一致收敛。论文给出理论保证：语义锚点公式的梯度方差上界严格小于自一致性版本。

### Pipeline 概览

```
目标域样本 x
    ├──→ [教师 f_T，参数冻结] ──────────→ anchor（固定软标签）
    │                                          ↑            ↑
    ├──→ [学生 f_θ] ──────────→ p_clean ──→ KL损失        |
    │                                                       |
    └──→ [PGD攻击，目标=anchor] ──→ x_adv                 |
              └──→ [学生 f_θ] ──→ p_adv ──────────────→ KL损失
                                              ↓
                                   total = clean_loss + β·adv_loss
                                              ↓
                                        梯度更新 θ
```

## 实现

### 核心代码

```python
import torch
import torch.nn.functional as F

def pgd_attack(student, x, anchor, epsilon=8/255, alpha=2/255, steps=10):
    """
    以固定语义锚点为目标的PGD攻击
    最大化 KL(anchor || f_θ(x+δ))
    """
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
    delta.requires_grad_(True)

    for _ in range(steps):
        adv_log_pred = F.log_softmax(student(x + delta), dim=1)
        # 负号：对负KL做梯度下降 = 对KL做梯度上升（最大化）
        loss = -F.kl_div(adv_log_pred, anchor.detach(), reduction='batchmean')
        loss.backward()

        with torch.no_grad():
            delta.data += alpha * delta.grad.sign()
            delta.data.clamp_(-epsilon, epsilon)
            (x + delta.data).clamp_(0, 1)  # 投影回合法像素空间
        delta.grad.zero_()

    return (x + delta).detach()


def semantic_anchor_loss(student, teacher, x, beta=6.0,
                         epsilon=8/255, alpha=2/255, steps=10):
    """语义锚点损失——论文核心公式"""
    # Step 1: 获取固定锚点（不参与梯度）
    with torch.no_grad():
        anchor = F.softmax(teacher(x), dim=1)

    # Step 2: 干净对齐损失
    clean_loss = F.kl_div(
        F.log_softmax(student(x), dim=1), anchor, reduction='batchmean'
    )

    # Step 3: 对抗对齐损失（BN 用 eval 模式，避免统计量被污染）
    student.eval()
    x_adv = pgd_attack(student, x, anchor, epsilon, alpha, steps)
    student.train()

    adv_loss = F.kl_div(
        F.log_softmax(student(x_adv), dim=1), anchor, reduction='batchmean'
    )

    return clean_loss + beta * adv_loss
```

### 测试时适应主循环

```python
def test_time_adapt(student, teacher, data_loader, lr=1e-3, adapt_steps=1):
    """
    测试时适应主循环
    只更新 BN 层，避免灾难性遗忘
    """
    # 只更新 BatchNorm 的可学习参数
    params = [p for n, p in student.named_parameters()
              if 'bn' in n or 'norm' in n]
    optimizer = torch.optim.Adam(params, lr=lr)
    teacher.eval()  # 教师始终冻结

    for x_batch, _ in data_loader:  # 无标签，忽略 y
        for _ in range(adapt_steps):
            optimizer.zero_grad()
            loss = semantic_anchor_loss(student, teacher, x_batch)
            loss.backward()
            optimizer.step()
        yield student  # 适应后用于推理
```

### 为什么自一致性版本不稳定

```python
def trades_tta_unstable(student, teacher, x, beta=6.0):
    """直接迁移 TRADES 到无监督TTA——问题演示"""
    with torch.no_grad():
        teacher_pred = F.softmax(teacher(x), dim=1)

    # 干净损失（对齐教师）
    clean_loss = F.kl_div(
        F.log_softmax(student(x), dim=1), teacher_pred, reduction='batchmean'
    )

    # ⚠️ 问题：攻击目标是 student(x).detach()，但每次迭代后 student 已更新
    # 下一步优化时，"当前 student(x)" 已经不是同一个分布了 → 目标漂移
    moving_target = F.softmax(student(x), dim=1).detach()  # 快照，但很快过期
    x_adv = pgd_attack(student, x, moving_target)

    adv_loss = F.kl_div(
        F.log_softmax(student(x_adv), dim=1), moving_target, reduction='batchmean'
    )
    return clean_loss + beta * adv_loss  # 梯度方向混乱，loss 震荡
```

### 可视化稳定性对比

```python
import matplotlib.pyplot as plt

def visualize_stability(losses_anchor, losses_trades, losses_clean):
    """对比三种方法的适应过程稳定性"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, losses, title, color in [
        (axes[0], [losses_trades, losses_clean], 'TRADES-TTA vs 无鲁棒性约束', ['red', 'gray']),
        (axes[1], [losses_anchor, losses_clean], '语义锚点 vs 无鲁棒性约束', ['steelblue', 'gray']),
    ]:
        labels = ['对抗Loss', '干净TTA Loss']
        for loss, c, lbl in zip(losses, color, labels):
            ax.plot(loss, alpha=0.8, color=c, label=lbl)
        ax.set_title(title); ax.set_xlabel('适应步数')
        ax.set_ylabel('Loss'); ax.legend()

    plt.tight_layout()
    plt.savefig('stability_comparison.png', dpi=150)
    # 预期效果：左图（TRADES-TTA）震荡剧烈；右图（语义锚点）单调下降
```

## 实验

### 数据集和评估设置

论文在两个标准分类数据集上实验，域偏移通过**光度变换（photometric transformations）**模拟——这在自动驾驶、机器人视觉等实际部署中极为常见：

| 数据集 | 域偏移类型 | 评估攻击 |
|--------|-----------|---------|
| CIFAR-10 | 亮度、对比度、色调变换 | PGD-20 ($\epsilon=8/255$) |
| ImageNet | 同类光度变换 | PGD-20, AutoAttack |

### 定量评估

| 方法 | 需要标签 | 需要鲁棒预训练 | 干净准确率 | 鲁棒准确率 | 超参数敏感性 |
|------|---------|--------------|-----------|-----------|------------|
| 无适应（基线） | — | — | 中 | 低 | — |
| TENT（熵最小化） | 否 | 否 | 高 | 无提升 | 低 |
| TRADES-TTA | 否 | 否 | 中 | 不稳定 | **极高** |
| **语义锚点（本文）** | **否** | **否** | **高** | **稳定提升** | **低** |

核心结论：本文方法在鲁棒性-准确率权衡曲线上整体优于基线，且对 $\beta$ 等超参数的选择不敏感。

## 工程实践

### 实际部署考虑

**计算开销是主要瓶颈**：每步 TTA 需要 PGD（10步内循环）+ 前向 × 2。相比普通 TTA（仅前向），计算量约增加 **10-15×**。对实时系统不友好。

**可行的优化方案**：

```python
# 方案1：减少PGD步数（牺牲攻击质量换速度）
loss = semantic_anchor_loss(student, teacher, x, steps=3)  # 默认10，改为3

# 方案2：单步FGSM（最快，但攻击强度弱）
def fgsm_attack(student, x, anchor, epsilon=8/255):
    x.requires_grad_(True)
    loss = -F.kl_div(F.log_softmax(student(x), dim=1), anchor, reduction='batchmean')
    loss.backward()
    return (x + epsilon * x.grad.sign()).clamp(0, 1).detach()

# 方案3：只对低置信度样本做对抗适应
def selective_adapt(student, teacher, x, conf_threshold=0.9):
    with torch.no_grad():
        anchor = F.softmax(teacher(x), dim=1)
        conf = anchor.max(dim=1).values
    high_conf_mask = conf > conf_threshold
    if high_conf_mask.any():
        return semantic_anchor_loss(student, teacher, x[high_conf_mask])
    return torch.tensor(0.0)
```

### 常见坑

**坑1：PGD 攻击时忘记切换 BN 模式**
```python
# ❌ BN 在 train 模式：攻击时统计量被对抗样本污染
x_adv = pgd_attack(student, x, anchor)

# ✅ 攻击时临时切换 eval，更新时切回 train
student.eval()
x_adv = pgd_attack(student, x, anchor)
student.train()
```

**坑2：KL 散度方向和 reduction 搞错**
```python
# PyTorch 的 F.kl_div(log_p, q) = KL(q || p)
# input 必须是 log_softmax；target 是普通概率

# ❌ 常见错误：input/target 颠倒，或忘了 log_softmax
F.kl_div(anchor, student_logits)

# ✅ 正确
F.kl_div(F.log_softmax(student_logits, dim=1), anchor, reduction='batchmean')
```

**坑3：适应步数过多导致遗忘**
```python
# 少量样本 + 多步适应 = 在少量样本上严重过拟合
adapt_steps = 1   # 经验法则：通常1步足够，而非10或20
lr = 1e-4         # 学习率也要保守，避免偏离预训练权重太远
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 部署后遭遇光照/相机参数偏移 | 实时系统（PGD 计算开销大） |
| 有少量无标签目标域样本（≥32张） | 目标域样本极少（<10张，过拟合风险） |
| 安全关键应用（医疗、自动驾驶） | 攻击强度远超训练时的 ε |
| 预算不允许重新进行对抗训练 | 分布持续漂移的流式场景 |

## 与其他方法对比

| 方法 | 需要标签 | 测试时计算量 | 鲁棒性提升 | 干净准确率 |
|------|---------|------------|-----------|-----------|
| 标准 AT | 是（训练时） | 低 | 高（同域） | 略有下降 |
| TENT | 否 | 极低 | 无 | 有提升 |
| Robust 预训练 + TTA | 否 | 低 | 中 | 中 |
| **语义锚点（本文）** | **否** | **高** | **稳定提升** | **保持** |

## 我的观点

**核心洞见的普适性**："固定锚点比移动目标更稳定"——这个观察超越了这篇论文本身。任何 min-max 优化中，内层最大化的目标如果会随外层参数更新而漂移，训练都会不稳定。这是个设计原则，值得迁移到其他场景（如 GAN 的判别器/生成器更新解耦）。

**离实际应用还有多远**：
- 计算开销是硬伤——PGD-10 的对抗 TTA 在边缘设备上基本不可行
- 目前只在图像分类上验证，检测/分割等更复杂任务尚未覆盖
- 连续部署场景（分布持续变化）中，静态锚点是否还足够？

**值得关注的开放问题**：
1. 能否设计专门针对 TTA 场景的高效对抗例生成方式，将计算量降低到 FGSM 级别？
2. 当目标域分布持续漂移（Continual TTA）时，语义锚点是否需要动态更新，以及如何更新？
3. 视觉语言模型（VLM）中，"语义锚点"能否从文本空间获得更丰富、更鲁棒的监督信号，彻底绕开视觉教师的脆弱性？