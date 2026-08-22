---
layout: post-wide
title: '神经网络"过度自信"怎么破？TCP_α 后验置信度估计详解'
date: 2026-08-22 12:03:42 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.20326v1
generated_by: Claude Code CLI
---

正在基于论文摘要撰写关于 TCP_α 置信度估计的深度技术博客。


## 一句话总结

TCP_α 是一种后验置信度估计方法：在冻结分类器之上训练轻量辅助头，通过引入 margin 惩罚项，**在目标值层面保证正确预测与错误预测的完全分离**，让系统知道什么时候该拒绝回答。

## 背景：校准 vs. 置信度估计，傻傻分不清楚

你训练好一个分类器，准确率 89%。但有时候它说 99% 把握，结果还是错了；有时候说 60% 把握，却是对的。这就是**过度自信**（overconfidence）问题。

### 现有方法的局限

**温度缩放（Temperature Scaling）** 是最常见的校准方法：

$$p_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

它调整整体概率分布，但它只改变置信度的**幅度**，不能改变哪些预测应该被信任。T 对所有样本一视同仁。

**TCP（True Class Probability，Corbière et al. 2019）** 则是后验置信度估计的主流方案：训练一个辅助头，预测"这个样本分类正确的概率"。TCP 的目标值定义为：

$$\text{target}(x, y) = p_{\theta}(y \mid x)$$

即真实类别的 softmax 概率。正确预测时这个值等于最大类别概率，错误预测时这个值较小（真实类不是预测类）。

### TCP 的问题：目标值重叠

问题在于，接近决策边界的样本，**错误预测的目标值可以非常接近正确预测的目标值**：

| 样本 | 真实类概率 | 是否正确 | TCP 目标值 |
|------|-----------|---------|-----------|
| 样本 A | 0.52 | ✓ | 0.52 |
| 样本 B | 0.48 | ✗ | 0.48 |
| 样本 C | 0.91 | ✓ | 0.91 |

样本 A（正确）和样本 B（错误）的目标值极其接近，辅助头无法学到可靠的分界线。

## TCP_α：用 margin 惩罚强制分离

### 直觉解释

TCP_α 的核心想法很简单：**对错误预测的目标值施加额外惩罚 α，人为拉大两类目标值的间距**。

```
正确预测目标值: ████████████████░░░░  0.52
错误预测目标值: ████░░░░░░░░░░░░░░░░  0.02  ← 减去 α=0.5
                                  ↑
                          完全分离的间距
```

### 数学定义

$$\text{TCP}_\alpha(x, y) = \begin{cases} p_\theta(y \mid x) & \text{if } \hat{y} = y \text{（预测正确）} \\ \max\!\left(p_\theta(y \mid x) - \alpha,\ 0\right) & \text{if } \hat{y} \neq y \text{（预测错误）} \end{cases}$$

其中：
- $p_\theta(y \mid x)$ 是冻结分类器对真实类别的 softmax 概率
- $\hat{y} = \arg\max_k p_\theta(k \mid x)$ 是预测类别
- $\alpha \in (0, 1)$ 是 margin 惩罚参数

**分离保证的直觉**：对于一个准确的分类器，正确预测的 $p_\theta(y \mid x)$ 分布集中在较高区域，错误预测的 $p_\theta(y \mid x)$ 分布集中在较低区域（真实类不是最大类）。减去 $\alpha$ 后，错误目标值被进一步压低，两个分布之间的间隔变得更清晰，辅助头更容易学到判别性的阈值。

### 另一个关键问题：严重的类别不平衡

好的分类器错误样本很少。比如基础准确率 89%，错误样本只占 11%。这是**回归任务中的极端不平衡**，会导致辅助头偏向总是预测"正确"（高置信度）。

论文对此做了系统性研究，发现对错误样本加权训练是最有效的策略之一。

## 实现

### 核心组件：TCP_α 目标值计算

```python
import torch
import torch.nn.functional as F

def compute_tcp_alpha(
    probs: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算 TCP_α 置信度目标值。

    Args:
        probs: softmax 输出 [B, K]
        labels: 真实标签 [B]
        alpha: margin 惩罚参数

    Returns:
        targets: TCP_α 目标值 [B]
        correct_mask: 正确预测的布尔掩码 [B]
    """
    B = probs.shape[0]
    true_probs = probs[torch.arange(B), labels]   # 真实类别概率
    predicted  = probs.argmax(dim=-1)
    correct    = (predicted == labels)

    targets = true_probs.clone()
    # 错误预测：减去惩罚 α
    targets[~correct] = (true_probs[~correct] - alpha).clamp(min=0.0)

    return targets, correct
```

### 置信度预测头

```python
import torch.nn as nn

class ConfidenceHead(nn.Module):
    """轻量置信度预测头，接在冻结的主干网络特征之上"""
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid()          # 输出 [0, 1] 置信度分数
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

### 不平衡感知损失函数

```python
def imbalance_aware_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    correct_mask: torch.Tensor,
    error_weight: float = 10.0
) -> torch.Tensor:
    """对错误预测样本加权 MSE，缓解严重的样本不平衡"""
    weights = torch.ones_like(target)
    weights[~correct_mask] = error_weight       # 错误样本权重放大

    # 可选：对接近分界面的正确预测也加权
    # low_conf_correct = correct_mask & (target < 0.6)
    # weights[low_conf_correct] = 3.0

    return (weights * (pred - target).pow(2)).mean()
```

### 完整训练流程

```python
def train_confidence_head(
    backbone,           # 冻结的特征提取器 + 分类器
    conf_head: ConfidenceHead,
    train_loader,
    alpha: float = 0.5,
    error_weight: float = 10.0,
    epochs: int = 30,
    lr: float = 1e-3,
) -> list[float]:
    optimizer = torch.optim.Adam(conf_head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    losses = []

    backbone.eval()     # 冻结主干
    conf_head.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in train_loader:
            with torch.no_grad():
                feats  = backbone.extract_features(x)   # 倒数第二层特征
                logits = backbone.classify(feats)
                probs  = F.softmax(logits, dim=-1)

            targets, correct_mask = compute_tcp_alpha(probs, y, alpha)
            pred_conf = conf_head(feats)
            loss = imbalance_aware_loss(pred_conf, targets, correct_mask, error_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(conf_head.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        losses.append(epoch_loss / len(train_loader))

    return losses
```

### 推理：用置信度做拒绝决策

```python
def predict_with_rejection(backbone, conf_head, x, threshold=0.5):
    """
    低于阈值的预测标记为"不确定"，拒绝回答。
    """
    with torch.no_grad():
        feats  = backbone.extract_features(x)
        logits = backbone.classify(feats)
        probs  = F.softmax(logits, dim=-1)
        conf   = conf_head(feats)

    predicted = probs.argmax(dim=-1)
    accepted  = conf >= threshold

    return predicted, conf, accepted   # accepted=False 表示该预测被拒绝
```

## 关键 Trick（没有就跑不起来）

### 1. α 的选取

α 是最敏感的超参数。它直接决定分离 margin 的大小：

| α | 效果 |
|---|------|
| 太小（<0.1） | 分离不足，接近普通 TCP |
| 0.3-0.6 | 通常效果最好，从 0.5 开始 |
| 太大（>0.8） | 错误目标值都被截断到 0，辅助头退化为二分类 |

### 2. 错误样本权重

准确率越高，错误样本越少，权重需要越大：

```python
# 估算合适的 error_weight
error_rate = 1 - accuracy
suggested_weight = min(50.0, 1.0 / error_rate)
```

### 3. 特征层选取

不要用最后一层 logits，要用倒数第二层的嵌入特征。Logits 信息已经被 softmax 压缩过，丢失了细节。

## 实验：置信度估计的评估指标

置信度估计的核心指标不是 MSE，而是**失败预测（failure prediction）**：

- **AUPR-Error**：以"样本是否错误"为正例的 PR 曲线下面积，越高越能找到错误
- **覆盖率-准确率权衡**：拒绝 X% 的预测后，剩余预测的准确率

论文的关键结果：拒绝置信度最低的 8% 的预测，宏 F1 从 0.89 提升到 0.98。

```python
def coverage_accuracy_curve(conf_scores, correct_mask, thresholds):
    """计算不同拒绝阈值下的覆盖率和准确率"""
    results = []
    for t in thresholds:
        accepted = conf_scores >= t
        coverage = accepted.float().mean().item()
        if accepted.sum() > 0:
            acc = correct_mask[accepted].float().mean().item()
        else:
            acc = 1.0
        results.append((coverage, acc))
    return results
```

## 调试指南

### 置信度头总是输出接近 0.5

**原因**：不平衡问题导致头没有学到任何信息，输出均值。
**检查**：计算训练集中正确/错误样本的比例，确认 `error_weight` 是否足够大。

### 置信度分布完全不分离

```python
# 诊断代码：分别画出正确/错误预测的置信度分布
import matplotlib.pyplot as plt

conf_correct = conf_scores[correct_mask]
conf_wrong   = conf_scores[~correct_mask]

plt.hist(conf_correct.numpy(), bins=50, alpha=0.5, label='correct')
plt.hist(conf_wrong.numpy(),   bins=50, alpha=0.5, label='wrong')
plt.legend(); plt.show()
```

如果两个分布完全重叠，说明 α 太小或者 error_weight 不够。

### 拒绝阈值难以确定

不要手动定阈值，用覆盖率-准确率曲线，根据业务需求选"拒绝多少比例"而不是"阈值是多少"。这更直观也更稳定。

### 超参数敏感度

| 超参 | 推荐范围 | 敏感度 | 建议 |
|------|---------|-------|------|
| α | 0.3-0.6 | 高 | 从 0.5 开始 |
| error_weight | 5-20× | 高 | 根据准确率估算 |
| hidden_dim | 128-512 | 低 | 256 就够 |
| lr | 1e-3 到 3e-4 | 中 | 配合 scheduler |
| epochs | 20-50 | 低 | 早停即可 |

## 什么时候用 TCP_α？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 准确率已经很高（>85%），需要进一步筛除错误 | 基础准确率很低（<70%），先改分类器 |
| 有可靠的特征提取器，但想不重训练主干 | 训练集中错误样本极少（<1%），很难训练 |
| 需要拒绝选项（failure prediction） | 需要全局概率校准（用温度缩放更合适） |
| 跨域泛化时只有少量标注数据（5% 微调） | 动态环境下类别持续变化 |

## 我的观点

TCP_α 的核心贡献是清晰的：**用一个设计上保证分离性的目标函数，解决了 TCP 中目标值重叠的问题**。思路简洁，有理论保证。

但有几个现实问题需要考虑：

1. **领域依赖性强**：论文在拉格（rāga，印度音乐分类）上验证，这是个类别极多、相似度高的细粒度分类任务。在自然图像或 NLP 任务上的泛化性值得独立验证。

2. **不平衡训练的挑战被低估**：论文确实做了系统研究，但实践中这个问题非常棘手，error_weight 的选取仍然需要验证集调参。

3. **5% 标注数据的域迁移**：这个结果很有吸引力，但代表只需少量目标域数据就能恢复置信度性能。这一点如果能复现，工程价值很高。

对于实际应用：如果你有一个部署中的分类器，想在不重训练的前提下增加"拒绝"能力，TCP_α 值得一试。入门成本低（只需训练一个小 MLP），效果上限取决于基础分类器的特征质量。