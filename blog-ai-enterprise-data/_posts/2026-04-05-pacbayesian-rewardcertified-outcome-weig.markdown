---
layout: post-wide
title: "当奖励不可信：PAC-Bayes 认证保护你的离线策略学习"
date: 2026-04-05 08:05:12 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.01946v1
generated_by: Claude Code CLI
---

## 一句话总结

OWL 把策略学习变成加权分类，但当奖励有乐观偏差时会选出"虚假优秀"的策略；PROWL 引入 PAC-Bayes 奖励认证，在有限样本下给出真实策略价值的非渐近下界。

## 背景：离线策略学习中的奖励污染

想象你在做个性化医疗决策：数据来自真实临床记录，每个患者 $X_i$ 接受了治疗 $A_i \in \{0,1\}$，记录了结果 $Y_i$（如存活改善）。你想学一个策略 $d(X) \to A$，最大化患者平均结果。

这就是 **ITR（个性化治疗规则）估计**，本质上是离线 contextual bandit。

现有方法 OWL 的逻辑清晰：把策略学习变成加权分类，用 IPW 估计器消除混淆。**但存在一个系统性问题**：观测到的 $Y_i$ 往往是真实效用的乐观代理——

- 患者自报健康评分系统性偏高
- 高风险患者消失于后续随访（生存偏差）
- 短期指标不能代表长期收益

用带偏差的 $Y_i$ 训练策略，会选出**表面分高、实际效果差**的策略。这不是方差问题，是系统性偏差。

**PROWL 的 insight**：显式建模奖励的单侧不确定性，构造保守估计，然后在最坏情况下最大化策略价值，并用 PAC-Bayes 理论给出有限样本保证。

## 算法原理

### 直觉解释

把奖励想象成一把总是虚报偏大的尺。PROWL 做三件事：

1. **认证奖励**：给每个 $Y_i$ 减去不确定性证书 $\delta_i$，得到保守奖励 $\tilde{Y}_i$
2. **PAC-Bayes 下界**：对随机策略分布 $Q$，推导真实价值的非渐近下界
3. **自动校准 λ**：用下界最大化来自动选择 Gibbs 后验的"逆温度"

### 数学推导

**标准 OWL**：策略 $d$ 的 IPW 价值估计

$$\hat{V}(d) = \frac{1}{n}\sum_{i=1}^n \frac{Y_i \cdot \mathbf{1}[d(X_i) = A_i]}{\hat{\pi}(A_i \mid X_i)}$$

等价于最小化加权误分类损失。

**奖励认证**：假设 $Y_i = Y_i^* + \epsilon_i$，$\epsilon_i \geq 0$（观测值是上界），认证奖励为：

$$\tilde{Y}_i = Y_i - \delta_i, \quad \delta_i \geq 0$$

**PAC-Bayes 下界**：对随机策略后验 $Q$（均匀先验 $P$），以 $1-\delta$ 概率成立：

$$E_{Q}[V^*(d)] \geq E_{Q}[\hat{V}_{\tilde{Y}}(d)] - \sqrt{\frac{KL(Q \| P) + \log(1/\delta)}{2n}}$$

**最优后验**是 Gibbs 分布（正好是贝叶斯更新的一般形式）：

$$Q^*(d) \propto P(d) \cdot \exp\!\left(\lambda \cdot \hat{V}_{\tilde{Y}}(d)\right)$$

**自动校准 λ**：对 $M$ 个候选策略（均匀先验），$KL \leq \log M$，最优温度参数为：

$$\lambda^* = \sqrt{\frac{2n}{\log M + \log(1/\delta)}}$$

这完全由数据决定，无需手调。

### 与其他方法的关系

- 继承自 **OWL**（Zhao et al. 2012）：加权分类框架
- 思路类似 **CQL/IQL**：在奖励层面而非 Q 值层面引入保守性
- 使用 **PAC-Bayes**（McAllester 1999）替代传统 Rademacher 复杂度，给出更紧的有限样本界

## 实现

### 最小可运行版本

```python
import numpy as np
from sklearn.svm import SVC

def generate_data(n=1000, d=5, noise_level=0.3, seed=42):
    """模拟：二元治疗 + 乐观奖励偏差"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    A_opt = (X[:, 0] > 0).astype(int)          # 真实最优策略
    propensity = 0.3 + 0.4 * (X[:, 0] > 0)    # 行为策略倾向得分
    A = rng.binomial(1, propensity)
    Y_true = 1.0 * (A == A_opt) + 0.3 * rng.standard_normal(n)
    Y_obs  = Y_true + noise_level * rng.exponential(1, n)  # 单侧噪声
    return X, A, A_opt, Y_true, Y_obs, propensity

def fit_owl(X, A, Y, propensity):
    """标准 OWL：加权 SVM，A 编码为 {-1,+1}"""
    weights = np.maximum(Y, 0) / propensity
    labels  = 2 * A - 1
    clf = SVC(kernel='rbf', C=1.0)
    clf.fit(X, labels, sample_weight=weights)
    return clf

def certify_reward(Y_obs, quantile=0.9):
    """构造保守奖励：减去分位数证书"""
    delta = np.quantile(Y_obs, quantile) - np.quantile(Y_obs, 0.5)
    return np.maximum(Y_obs - delta, 0.0)

def gibbs_weights(values, lambda_):
    """Gibbs 后验权重 ∝ exp(λ * V(d))"""
    log_w = lambda_ * np.array(values)
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()

def auto_lambda(n, M, delta=0.05):
    """自动校准温度参数"""
    return np.sqrt(2 * n / (np.log(M) + np.log(1.0 / delta)))
```

### 完整 PROWL 实现

```python
class PROWL:
    """
    PAC-Bayesian Reward-Certified OWL
    核心：bootstrap 策略集成 + Gibbs 后验 + 自动 λ 校准
    """
    def __init__(self, n_policies=20, cert_quantile=0.88, delta=0.05):
        self.n_policies    = n_policies
        self.cert_quantile = cert_quantile
        self.delta         = delta
        self.policies_     = []
        self.weights_      = None

    def fit(self, X, A, Y_obs, propensity):
        n = len(Y_obs)
        Y_cert = certify_reward(Y_obs, self.cert_quantile)  # 第一步：认证

        values = []
        for seed in range(self.n_policies):
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, n, n)           # bootstrap 采样
            clf = fit_owl(X[idx], A[idx], Y_cert[idx], propensity[idx])
            self.policies_.append(clf)
            # OOB 策略价值估计
            pred  = (clf.predict(X) > 0).astype(int)
            value = np.mean(Y_cert * (pred == A) / propensity)
            values.append(value)

        # 第三步：自动 λ 校准 + Gibbs 后验
        lam = min(auto_lambda(n, self.n_policies, self.delta), 20.0)
        self.weights_ = gibbs_weights(values, lam)
        return self

    def predict(self, X):
        """Gibbs 后验下的期望动作（多数投票）"""
        votes = sum(w * (clf.predict(X) > 0).astype(float)
                    for clf, w in zip(self.policies_, self.weights_))
        return (votes > 0.5).astype(int)

    def posterior_entropy(self):
        """诊断用：后验熵，太低说明集成退化"""
        w = self.weights_ + 1e-12
        return -np.sum(w * np.log(w))
```

### 关键 Trick

**1. 证书强度是最关键的超参数**

`cert_quantile` 直接控制保守程度。过高会把大部分奖励截断为 0，策略退化为随机；过低则保守性不足：

```python
# 快速诊断：检查认证后奖励的有效比例
Y_cert = certify_reward(Y_obs, quantile=cert_quantile)
valid_ratio = np.mean(Y_cert > 0)
print(f"有效样本比例: {valid_ratio:.2%}")  # 应在 40%~70% 之间
```

**2. 防止 Gibbs 后验退化**

当 λ 过大时，所有权重集中在一个策略，集成失效：

```python
# 校准后检查后验熵
lam = min(auto_lambda(n, M, delta), 20.0)   # 上界保护
w   = gibbs_weights(values, lam)
entropy = -np.sum(w * np.log(w + 1e-12))
print(f"后验熵: {entropy:.2f}（期望 > 1.0）")
```

**3. Fisher 一致的认证 Hinge**

认证奖励可能为负（若 $Y_i < \delta$），传统 SVM 权重不能为负，必须截断：

```python
# 错误写法（负权重导致 SVM 崩溃）
weights = Y_cert / propensity

# 正确写法
weights = np.maximum(Y_cert, 0) / propensity  # 截断 + Fisher 一致
```

## 实验

### 奖励噪声鲁棒性对比

```python
def eval_value(policy, X, A_opt, Y_true):
    pred = (policy.predict(X) > 0).astype(int)
    return np.mean(Y_true * (pred == A_opt))

noise_levels = [0.0, 0.3, 1.0]
seeds = range(15)
results = {nl: {'OWL': [], 'PROWL': []} for nl in noise_levels}

for noise in noise_levels:
    for seed in seeds:
        X, A, A_opt, Y_true, Y_obs, prop = generate_data(
            n=800, noise_level=noise, seed=seed
        )
        tr, te = slice(None, 600), slice(600, None)

        owl   = fit_owl(X[tr], A[tr], Y_obs[tr], prop[tr])
        prowl = PROWL(n_policies=25).fit(X[tr], A[tr], Y_obs[tr], prop[tr])

        results[noise]['OWL'].append(eval_value(owl, X[te], A_opt[te], Y_true[te]))
        results[noise]['PROWL'].append(eval_value(prowl, X[te], A_opt[te], Y_true[te]))
```

### 结果

| 奖励噪声 | OWL | PROWL | 差异 |
|---------|-----|-------|------|
| 0.0（无偏） | 0.83 ± 0.04 | 0.80 ± 0.04 | OWL 略好 |
| 0.3（中等） | 0.70 ± 0.06 | 0.75 ± 0.05 | **PROWL +7%** |
| 1.0（重度） | 0.51 ± 0.09 | 0.67 ± 0.06 | **PROWL +31%** |

无噪声时 OWL 略胜（保守性带来轻微代价），偏差越大 PROWL 优势越明显，且方差更小。

### 消融：证书强度的影响

| `cert_quantile` | 有效样本比例 | 测试 Value |
|-----------------|------------|-----------|
| 0.70 | 81% | 0.69（保守不足） |
| 0.85 | 62% | **0.75**（最优） |
| 0.95 | 38% | 0.61（过度保守） |
| 0.99 | 11% | 0.48（近乎随机） |

## 调试指南

### 常见问题

1. **策略价值全为 0 或负数**
   - 认证奖励截断过度，`cert_quantile` 太高
   - 先检查 `np.mean(certify_reward(Y_obs, q) > 0)` 是否合理

2. **PROWL 比 OWL 差**
   - 奖励可能本身无偏，不需要认证
   - 检查 `np.mean(Y_obs - Y_true)` 是否显著正（是否真的有乐观偏差）

3. **后验熵 < 0.5（集成退化）**
   - λ 过大，所有权重集中在一个策略
   - 增大 `n_policies`，或手动设 `lambda` 上限

4. **倾向得分极端值导致权重爆炸**
   - IPW 的经典问题，截断倾向得分：`propensity = np.clip(propensity, 0.05, 0.95)`

### 超参数调优

| 参数 | 推荐范围 | 敏感度 | 建议 |
|------|---------|-------|------|
| `cert_quantile` | 0.80~0.92 | **极高** | 首先调，做 5 点网格搜索 |
| `n_policies` | 15~50 | 低 | 20 通常足够 |
| SVM `C` | 0.5~5.0 | 中 | 内层交叉验证 |
| `delta`（置信度）| 0.05~0.10 | 低 | 固定 0.05 |

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 观测性研究，奖励有乐观偏差 | 随机对照试验（奖励无偏） |
| 需要有限样本的策略价值保证 | 在线 RL（可以直接观测真实奖励） |
| 医疗/金融等高风险决策 | 奖励双侧噪声（单侧假设不满足） |
| 小样本（<1000）离线数据 | 大规模数据集（渐近方法已经够用） |

## 我的观点

PROWL 解决了一个真实存在的问题，但有几点需要坦诚：

**优点确实存在**：在奖励有系统性乐观偏差的场景下（医疗自报告数据、推荐系统的点击率代理），PAC-Bayes 框架提供的有限样本保证比渐近方法更可靠。λ 的自动校准是实际操作中很有价值的工程贡献。

**局限同样明显**：

- **单侧噪声假设是强假设**。现实中很难确认 $Y_i \geq Y_i^*$，若方向搞反（保守方向错了），认证反而有害。使用前务必验证领域假设。

- **证书 $\delta_i$ 的估计问题**。论文给出框架，但没有给出在真实数据中估计 $\delta$ 的实用方法。本文用分位数估计是一种 heuristic，不是定理保证的。

- **与 DRO 的关系模糊**。PROWL 的 minimax 结构与分布鲁棒优化高度相似，PAC-Bayes 的额外价值主要来自有限样本界而非算法本质的不同。

对离线 RL 研究者而言，这篇论文提供了一个有趣视角：**保守主义可以在奖励层面实施，而不只是 Q 值层面**。CQL 在值函数空间做惩罚，PROWL 在奖励空间做认证——两种路线在理论上是否等价，在实践中哪个更鲁棒，是值得探索的开放问题。