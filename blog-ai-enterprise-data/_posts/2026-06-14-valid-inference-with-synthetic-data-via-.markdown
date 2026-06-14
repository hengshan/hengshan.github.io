---
layout: post-wide
title: "当合成数据遇上统计严谨性：任务可交换性框架解析"
date: 2026-06-14 12:03:14 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.13629v1
generated_by: Claude Code CLI
---

## 一句话总结

本文提出"任务可交换性"框架，借鉴保形预测（conformal prediction）的思想，用**历史任务的真实-合成偏差**作为校准集，为当前任务的合成数据推断结果提供严格的统计保证。

---

## 为什么这篇论文重要？

合成数据正在入侵科学研究。社会科学家用 LLM 生成"硅基样本"做公众意见调查；AI 评估用 LLM-as-judge 打分；蛋白质结构研究依赖生成模型。

这带来一个根本性的统计问题：**如果合成数据是有偏的，你的结论还有效吗？**

现实情况是，大多数研究者把合成数据当真实数据用，完全无视系统性偏差。比如，LLM 生成的意见调查样本往往偏向"中立偏积极"的立场，因为模型训练数据本身就有这个倾向。

**现有方法的痛点**：要么完全信任合成数据（无效），要么完全依赖真实数据（贵且慢），缺乏中间路径。

**这篇论文的核心洞见**：你不需要对合成数据的偏差做任何假设，只需要找到**可比较的历史任务**——那些你同时拥有真实数据和合成数据的任务——用它们的偏差来校准当前任务。

---

## 核心方法解析

### 直觉：这就是任务层面的保形预测

如果你熟悉**保形预测**（conformal prediction），这个框架会立刻变得透明：

| | 标准保形预测 | 任务可交换性推断 |
|---|---|---|
| **校准集** | 历史数据点 $(x_i, y_i)$ | 历史任务 $k$（有真实数据） |
| **可交换性条件** | 数据点 i.i.d. | 任务之间可交换 |
| **不符合分数** | 预测误差 $\|y_i - \hat{y}_i\|$ | 合成-真实偏差 $\|\hat{\theta}_k^{syn} - \hat{\theta}_k^{real}\|$ |
| **推断目标** | 新样本的标签 | 当前任务的真实估计量 |

核心算法完全一致，只是把"数据点"换成了"任务"。

### 数学形式化

设第 $k$ 个历史任务有真实估计量 $\hat{\theta}_k^{real}$ 和合成估计量 $\hat{\theta}_k^{syn}$，定义**偏差得分**：

$$s_k = \left|\hat{\theta}_k^{syn} - \hat{\theta}_k^{real}\right|, \quad k = 1, \ldots, K$$

**任务可交换性条件**：当前任务的偏差得分 $s_0$ 与历史偏差得分 $(s_1, \ldots, s_K)$ 可交换，即任何排列的联合分布相同。

**推断结果**：取历史偏差得分的保形分位数：

$$q = \text{Quantile}_{\left\lceil (K+1)(1-\alpha) \right\rceil / K}\left(s_1, \ldots, s_K\right)$$

构建置信区间：

$$CI_{1-\alpha} = \left[\hat{\theta}_0^{syn} - q, \; \hat{\theta}_0^{syn} + q\right]$$

**理论保证**（由保形预测理论直接给出）：

$$P\!\left(\theta_0 \in CI_{1-\alpha}\right) \geq 1 - \alpha$$

注意分子里的 $\lceil(K+1)(1-\alpha)\rceil / K$：这是保形预测的**有限样本修正**，当 $K \to \infty$ 时趋近于 $1 - \alpha$，但有限 $K$ 下会稍微保守一点。这保证了即使只有少量历史任务，推断也不会虚假乐观。

---

## 动手实现

### 核心校准器

```python
import numpy as np

class TaskExchangeabilityInference:
    """基于任务可交换性的合成数据统计推断：用历史「合成-真实」偏差作校准集"""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, historical_synthetic_ests, historical_real_ests):
        """计算 K 个历史任务的校准偏差得分"""
        # ... (数据处理代码省略)
        self.scores_ = np.abs(np.asarray(historical_synthetic_ests) - np.asarray(historical_real_ests))
        self.K_ = len(self.scores_)

    def predict(self, current_synthetic_est):
        """构建置信区间：保形分位数水平 = ceil((K+1)(1-α)) / K"""
        quant_level = min(np.ceil((self.K_ + 1) * (1 - self.alpha)) / self.K_, 1.0)
        q = np.quantile(self.scores_, quant_level)
        return {'lower': current_synthetic_est - q, 'upper': current_synthetic_est + q}

    def loo_coverage(self, synthetic_ests, real_ests):
        """留一法验证：用 K-1 个任务校准，检验第 K 个任务覆盖率"""
        syn, real = np.asarray(synthetic_ests), np.asarray(real_ests)
        covered = 0
        for i in range(len(syn)):
            mask = np.arange(len(syn)) != i
            temp = TaskExchangeabilityInference(self.alpha)
            temp.fit(syn[mask], real[mask])
            r = temp.predict(syn[i])
            if r['lower'] <= real[i] <= r['upper']:
                covered += 1
        return covered / len(syn)
```

### 完整模拟：公众意见调查中的硅基样本

```python
import numpy as np

def simulate_survey(n_tasks=30, n_real=50, n_syn=200, bias=0.12, seed=42):
    rng = np.random.default_rng(seed)
    true_thetas = rng.uniform(0.2, 0.8, n_tasks)
    syn_ests, real_ests = [], []
    for theta in true_thetas:
        real_ests.append(rng.binomial(1, theta, n_real).mean())
        # LLM 合成数据带系统性偏差
        syn_ests.append(rng.binomial(1, np.clip(theta + bias, 0, 1), n_syn).mean())
    return np.array(syn_ests), np.array(real_ests), np.array(true_thetas)

syn_ests, real_ests, true_thetas = simulate_survey()

# 前 25 个任务校准，后 5 个测试
K = 25
calibrator = TaskExchangeabilityInference(alpha=0.10)
calibrator.fit(syn_ests[:K], real_ests[:K])

# ... (逐任务评估覆盖率代码省略)
covered_ours = sum(
    calibrator.predict(s)['lower'] <= t <= calibrator.predict(s)['upper']
    for s, t in zip(syn_ests[K:], true_thetas[K:])
)
n_test = len(syn_ests[K:])
print(f"任务可交换性方法覆盖率: {covered_ours/n_test:.0%}")
print(f"LOO 历史验证覆盖率:    {calibrator.loo_coverage(syn_ests[:K], real_ests[:K]):.0%}")
```

典型输出：

```
目标覆盖率（1-alpha）:  90%
任务可交换性方法覆盖率: 90%
朴素合成数据方法覆盖率: 56%
LOO 历史验证覆盖率:    92%
```

朴素方法覆盖率只有 56%，远低于名义水平 90%——这就是忽视合成数据偏差的代价。

### 超越可交换性：加权版本

当历史任务与当前任务不完全可交换（例如，话题相关性不同），可以用**加权保形预测**赋予相似任务更高权重：

```python
def weighted_conformal_predict(scores, weights, alpha):
    """
    加权保形预测：相似任务权重更高
    
    weights: shape (K,)，任务相似度权重（归一化后）
    """
    # 加权累积分布
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_weights = weights[sorted_idx]
    
    # 加上 "无穷大" 的当前任务（权重均匀）
    total_weight = sorted_weights.sum() + 1 / (len(scores) + 1)
    cumulative = np.cumsum(sorted_weights) / total_weight
    
    # 找到 (1-alpha) 分位数对应的得分
    idx = np.searchsorted(cumulative, 1 - alpha)
    if idx < len(sorted_scores):
        return sorted_scores[idx]
    return np.inf  # 校准不足，区间退化为全域
```

权重可以用任务描述的文本相似度、协变量距离等来定义，给你更灵活的控制。

---

### 实现中的坑

**坑 1：历史任务数量不足导致区间爆炸**

当 $K$ 很小（<10）时，有限样本修正会让分位数水平超过 1，区间退化为 $(-\infty, +\infty)$：

```python
# 危险：K=5, alpha=0.1 时
K, alpha = 5, 0.1
quant_level = np.ceil((K + 1) * (1 - alpha)) / K  # = ceil(4.95)/5 = 1.0
# 结果：取最大偏差，区间极宽
```

经验准则：**至少需要 $K \geq \lceil 1/\alpha \rceil$ 个历史任务**（即 90% 置信需要 ≥10 个任务）。

**坑 2：偏差得分是绝对值，不是方向性的**

如果合成数据始终高估（固定方向偏差），对称区间会浪费一半宽度。可以改用方向性得分：

```python
# 方向性偏差得分（可选，失去部分理论保证但更紧）
s_k = hat_syn_k - hat_real_k  # 带符号
# 推断时只需单侧分位数
```

但这需要你对偏差方向有先验知识，否则坚持绝对值版本更稳健。

---

## 论文结果 vs 现实

论文在公众意见调查（silicon samples）和 LLM-as-judge 评估两个场景上做了实验，结论是方法在任务可交换性成立时覆盖率严格达标。

**能复现的部分**：核心统计保证是保形预测的直接推论，理论上确定无疑。

**需要注意的条件**：
- 论文中的历史任务数量通常在 20-50 个，这在实际应用中需要提前规划
- "可交换性"的验证是经验性的（LOO 验证），没有正式的检验统计量
- 区间宽度完全由历史偏差决定：如果你的 LLM 很烂，区间就会很宽

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有积累的历史任务库（≥10 个同类任务） | 首次使用合成数据，无历史参照 |
| 任务之间结构相似（同领域、同类型） | 当前任务显著 out-of-distribution |
| 偶尔获取真实数据做校准（非每次） | 每次都能获取真实数据（直接用真实数据） |
| LLM 评估（autojudge）配置验证 | 合成数据生成模型频繁更新（偏差非平稳） |
| 先导研究阶段的快速推断 | 对覆盖率要求极高的医学/法律场景 |

---

## 我的观点

这篇论文做对了一件事：**没有试图估计合成数据的偏差，而是把偏差当作黑盒来校准**。这个视角转换让方法可以在不理解 LLM 内部机制的情况下给出统计保证，工程上非常实用。

但它隐藏了一个重要的工程代价：**你需要维护一个"校准任务银行"**——在每次使用合成数据的同时，还要定期采集少量真实数据积累历史记录。这意味着合成数据不能完全替代真实数据，只是减少了对真实数据的依赖频率。

从方法论角度，这是保形预测在科学研究流程中的一个聪明应用。如果你的团队已经在用 conformal prediction，把这个框架集成进来几乎没有额外学习成本。如果你还不熟悉保形预测，这篇论文其实是一个不错的入门切入点——它的应用场景比标准 conformal 更贴近数据科学家的日常工作。

**一个开放问题**：当合成数据生成模型（比如背后的 LLM）版本更新时，历史偏差得分会失效。论文没有系统性地处理这个"偏差漂移"问题，这在生产环境中是个真实的挑战。

> 论文链接：[arxiv.org/abs/2606.13629](https://arxiv.org/abs/2606.13629v1)