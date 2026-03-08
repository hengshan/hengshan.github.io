---
layout: post-wide
title: "统计最优性不唯一：四种不相容的预测推断可容许性几何"
date: 2026-03-08 08:06:47 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.05335v1
generated_by: Claude Code CLI
---

## 一句话总结

同一个统计程序，在 Blackwell 风险支配准则下是"最优的"，在 anytime-valid 检验框架下却未必——这篇论文证明了评估统计程序最优性的四种经典准则是**两两不可嵌套**的，"最优"这个词必须先说清楚是哪种最优。

## 为什么这个问题重要？

统计学家设计推断程序时，天然地问：**有没有比这个更好的？** 如果没有，这个程序就叫"可容许的（admissible）"。

但"更好"怎么定义？不同应用场景下，答案截然不同：

- **序列 A/B 测试**（数据边到边分析）：你希望程序在任意停止时间都控制误差
- **预测区间**（conformal prediction）：你希望覆盖率有有限样本保证
- **经典参数估计**（最小二乘、贝叶斯）：你希望期望损失最小
- **无先验在线学习**：你用博弈论的时间平均论证接近最优

麻烦在于，这四个场景对应四种不同的最优性几何，**它们互相之间并不兼容**。如果你在序列实验中用了 Blackwell 最优的程序，它在 anytime-valid 意义下可能完全不是最优的。

## 背景：可容许性的经典定义

### 风险函数

设统计程序 $\delta$ 在真实参数 $\theta$ 下的期望损失（风险）为：

$$
R(\theta, \delta) = \mathbb{E}_\theta[L(\theta, \delta(X))]
$$

**程序 $\delta_1$ Blackwell 支配 $\delta_2$**：对所有 $\theta$ 有 $R(\theta, \delta_1) \leq R(\theta, \delta_2)$，且至少一个严格不等式成立。

**$\delta$ 是可容许的**：不存在支配它的程序。

### James-Stein：经典不可容许性例子

估计 $p \geq 3$ 维正态均值时，MLE（样本均值）是**不可容许的**——James-Stein 收缩估计量在所有参数点上风险都更低：

```python
import numpy as np

def james_stein(x):
    """James-Stein 收缩估计量，维度 p >= 3"""
    p = len(x)
    norm_sq = np.dot(x, x)
    shrinkage = max(0.0, 1.0 - (p - 2) / norm_sq)
    return shrinkage * x

def compare_risks(p=5, n_sim=50000, theta_range=(0, 4)):
    """蒙特卡洛比较 MLE 和 James-Stein 的风险（MSE）"""
    thetas = np.linspace(*theta_range, 40)
    mle_risks, js_risks = [], []

    for theta_val in thetas:
        theta = np.zeros(p);  theta[0] = theta_val
        samples = np.random.default_rng(0).normal(theta, 1.0, size=(n_sim, p))

        mle_risk = np.mean(np.sum((samples - theta) ** 2, axis=1))
        js_est   = np.stack([james_stein(x) for x in samples])
        js_risk  = np.mean(np.sum((js_est - theta) ** 2, axis=1))

        mle_risks.append(mle_risk);  js_risks.append(js_risk)

    return thetas, np.array(mle_risks), np.array(js_risks)

thetas, mle_r, js_r = compare_risks(p=5)
print(f"James-Stein 在所有参数点风险均低于 MLE: {np.all(js_r < mle_r)}")
print(f"平均风险节省: {np.mean(mle_r - js_r):.3f}")
```

这说明：Blackwell 意义下，$p \geq 3$ 时 MLE 是不可容许的。但换一个准则，情况可能完全不同。

## 四种可容许性几何

### 几何 1：Blackwell 风险支配

**空间**：凸风险集合 $\mathcal{R} \subseteq \mathbb{R}^k$，每个维度对应一个参数点的风险。

**可容许性证书**：若程序 $\delta$ 可容许，则存在一个**先验分布 $\pi$**（支撑超平面），使得 $\delta$ 在 $\pi$ 下是贝叶斯最优的。这是论文标题"Bayes with No Shame"的含义——每个可容许程序背后都隐藏着一个（可能是广义的）先验。

**关键性质**：鞅相干性（martingale coherence）是 Blackwell 可容许性的**必要条件**，但不充分。

### 几何 2：Anytime-Valid 可容许性（超鞅锥）

**场景**：序列收集数据，可在任意时间停止并得出结论。

**空间**：非负超鞅锥（nonnegative supermartingale cone）。合法的序列检验等价于一个 **e-process**——一个在零假设下期望值不超过 1 的非负过程。

**可容许性证书**：在 e-process 类中，鞅相干性是 anytime-valid 可容许性的**充要条件**（与 Blackwell 不同，它在此是充分的）。

```python
def evalue(x, mu0=0.0, mu1=1.0, sigma=1.0):
    """单观测的 e-value（似然比 H1/H0）"""
    return np.exp((mu1 - mu0) * x / sigma**2 - (mu1**2 - mu0**2) / (2 * sigma**2))

def anytime_valid_test(stream, alpha=0.05, mu0=0.0, mu1=1.0):
    """
    e-process 序列检验
    核心保证：无论何时停止，I 型错误 P(拒绝 H0 | H0) <= alpha
    """
    e_product, history, stop_time = 1.0, [], None
    for t, x in enumerate(stream):
        e_product *= evalue(x, mu0, mu1)   # 乘积构成 e-process
        history.append(e_product)
        if e_product >= 1.0 / alpha and stop_time is None:
            stop_time = t                  # 可安全停止
    return np.array(history), stop_time

rng = np.random.default_rng(42)
data_h1 = rng.normal(1.0, 1.0, 300)        # 真实 mu=1（H1 成立）
e_proc, stopped = anytime_valid_test(data_h1, alpha=0.05)
print(f"在第 {stopped} 步检测到显著性（共 {len(data_h1)} 步）")
```

**与 Blackwell 的分离**：存在 Blackwell 可容许但非 anytime-valid 可容许的程序（反之亦然），两者不可嵌套。

### 几何 3：边际覆盖有效性（共形预测）

**场景**：给定历史数据，构造新观测的预测集合，要求**有限样本**覆盖率保证，不依赖分布假设。

**空间**：可交换性秩（exchangeability rank）——利用数据可交换性，无需参数假设。

**可容许性证书**：可交换性秩函数确保预测集合有效。

```python
def conformal_interval(cal_y, cal_yhat, x_new_pred, alpha=0.1):
    """
    分裂共形预测区间
    保证：P(Y_{n+1} ∈ C(X_{n+1})) >= 1 - alpha（有限样本，精确保证）
    """
    residuals = np.abs(cal_y - cal_yhat)           # 不一致性分数
    n = len(residuals)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q = np.quantile(residuals, q_level)
    return x_new_pred - q, x_new_pred + q

# 验证覆盖率
rng = np.random.default_rng(0)
X_cal = rng.uniform(0, 1, 500)
y_cal = 2 * X_cal + rng.normal(0, 0.3, 500)
lo, hi = conformal_interval(y_cal, 2 * X_cal, x_new_pred=0.5, alpha=0.1)
print(f"90% 预测区间: [{lo:.3f}, {hi:.3f}]")
# 只要数据可交换，覆盖率精确 >= 90%，无需正态假设
```

**注意**：共形预测的可容许性与 Blackwell 无关——一个 Blackwell 最优的预测程序未必有有效的覆盖率，反之亦然。

### 几何 4：Cesàro 可接近性（CAA）

**场景**：不存在显式先验时，用博弈论的**时间平均**论证来接近风险集合边界。

**核心思想**：传统 Blackwell 可容许性需要找到一个真实贝叶斯先验（支撑超平面）。但有时先验不存在（无穷维参数、非正则情形）。CAA 用 Cesàro 平均代替先验：沿着可接近性方向"驾驶"风险序列到边界。

**可容许性证书**：Cesàro 导向策略——不是先验，而是一个随时间调整的混合策略，保证时间平均风险趋近边界。

这是四种几何中最抽象的，但在在线学习、非参数函数估计等场景中有实际价值。

## 分离定理：四种几何两两非嵌套

论文最核心的结论——对任意两种准则，都存在在其中一种下可容许、在另一种下不可容许的程序：

| 准则对 | 分离例子类型 |
|--------|------------|
| Blackwell vs Anytime-Valid | 鞅相干性：对 AV 充要，对 B 只必要 |
| Blackwell vs Coverage | 覆盖率与期望损失最小化目标不同 |
| Anytime-Valid vs Coverage | 序列有效性与可交换性秩无关 |
| CAA vs 其余三种 | 无先验场景与有支撑超平面场景不同 |

**核心几何直觉**：四种准则在不同的**空间**上定义偏序——凸风险集、超鞅锥、可交换排列、Cesàro 时间序列。这些空间的偏序结构互不相容，就像用欧几里得距离和曼哈顿距离定义"最近邻"会得到不同结果。

## 四种准则可视化比较

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
rng = np.random.default_rng(123)

# 左图：H0 下 e-process 轨迹（anytime-valid 性质验证）
ax = axes[0]
for i in range(6):
    data_h0 = rng.normal(0.0, 1.0, 400)    # H0 成立，不应拒绝
    e_proc, _ = anytime_valid_test(data_h0, alpha=0.05)
    ax.semilogy(e_proc, alpha=0.6)
ax.axhline(20, color='red', linestyle='--', label='拒绝阈值 1/α=20')
ax.set_title('H0 下 e-process 轨迹\n（极少超过红线 = I 型错误受控）')
ax.set_xlabel('样本量 t');  ax.set_ylabel('累积 e-process（对数）')
ax.legend()

# 右图：共形预测覆盖率随 alpha 变化
ax2 = axes[1]
target_coverages = np.linspace(0.6, 0.95, 15)
empirical_coverages = []

for target in target_coverages:
    alpha = 1 - target
    hits = sum(
        conformal_interval(
            rng.normal(2*rng.uniform(0,1,200), 0.3),
            2*rng.uniform(0,1,200), 0.5, alpha
        )[0] <= 2*0.5 + rng.normal(0, 0.3) <=
        conformal_interval(
            rng.normal(2*rng.uniform(0,1,200), 0.3),
            2*rng.uniform(0,1,200), 0.5, alpha
        )[1]
        for _ in range(300)
    )
    empirical_coverages.append(hits / 300)

ax2.plot(target_coverages, empirical_coverages, 'bo-', label='实际覆盖率')
ax2.plot([0.6, 0.95], [0.6, 0.95], 'r--', label='理论下界 1-α')
ax2.set_title('共形预测覆盖率验证\n（点始终在红线上方）')
ax2.set_xlabel('目标覆盖率 1-α');  ax2.set_ylabel('实际覆盖率')
ax2.legend()

plt.tight_layout()
plt.savefig('admissibility_geometries.png', dpi=150, bbox_inches='tight')
```

## 什么时候用哪种准则？

| 适用场景 | 推荐准则 | 原因 |
|---------|---------|------|
| 序列 A/B 测试（随时停止） | Anytime-Valid (e-process) | 任意停止时间下 I 型错误受控 |
| 回归/分类预测区间 | Coverage（共形预测） | 有限样本、无分布假设 |
| 固定样本参数估计 | Blackwell | 最小化期望损失 |
| 无先验在线学习 | CAA | 无需知道参数空间结构 |

### 常见坑

1. **固定样本 p-value 用于连续监控** → 改用 e-process，否则 p-value 在数据边到边分析下 I 型错误膨胀至 100%

2. **把边际覆盖率当条件覆盖率** → 共形预测保证整体覆盖率 $\geq 1-\alpha$，但对子群（如特定年龄段）不保证；需要条件共形

3. **高维估计默认用 MLE** → 维度 $p \geq 3$ 时，MLE 对球对称损失被 James-Stein 支配，Blackwell 不可容许

## 与相关方法对比

| 方法 | 最优性准则 | 先验需求 | 序列有效 | 分布假设 |
|-----|---------|--------|--------|--------|
| 贝叶斯估计 | Blackwell | 需要显式先验 | 否 | 需要似然 |
| James-Stein | Blackwell（支配 MLE） | 隐式（点质量先验） | 否 | 正态性 |
| E-process 检验 | Anytime-Valid | 无 | 是 | 无 |
| 分裂共形预测 | Coverage | 无 | 有限支持 | 可交换性 |
| 在线凸优化（Hedge） | CAA 类似 | 无 | 是 | 无 |

## 我的观点

这篇论文做了一件重要的**概念清理工作**——把统计学界长期混用的"最优性"精确化，证明了它的不可约性。

**对实践者的影响**：

- 序列实验（临床试验、在线平台）必须用 anytime-valid 准则，用 Blackwell 最优的固定样本程序会导致错误率失控
- 预测区间用共形预测比用参数区间更稳健，尤其是数据分布未知时
- "最优统计程序"不存在唯一定义，面试时遇到这类问题要先追问"在哪种损失/哪种保证下最优"

**局限性**：

- 论文主要在理论层面；具体场景下如何**计算**各准则下的最优程序，仍是开放问题
- Blackwell 最优程序需要求解贝叶斯问题（可能无解析解）
- CAA 可接近性策略的计算复杂度在高维下仍不清楚

**一个开放问题**：联合优化多个准则（如既 anytime-valid 又有良好覆盖率）时，Pareto 前沿的结构是什么？这在自适应临床试验设计中有很高的实际价值，目前几乎无结果。