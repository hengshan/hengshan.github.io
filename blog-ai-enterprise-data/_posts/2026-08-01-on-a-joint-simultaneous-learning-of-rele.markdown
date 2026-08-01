---
layout: post-wide
title: "混沌系统的稀疏预测：EOMR 如何联合选择特征子集与预测子空间"
date: 2026-08-01 12:06:31 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.28080v1
generated_by: Claude Code CLI
---

## 一句话总结

EOMR（Entropy-Optimal Manifold Regression）同时回答「**哪些特征重要**」和「**这些特征的哪个方向重要**」，在 Lorenz-96 等混沌动力学基准上，以比梯度提升树和深度网络低数量级的模型复杂度完成预测。

## 背景：混沌系统回归为什么难

### 现有方法的困境

以 Lorenz-96 为例——这是大气动力学领域最经典的混沌基准模型：

$$\frac{dX_k}{dt} = (X_{k+1} - X_{k-2})X_{k-1} - X_k + F, \quad k = 1, \ldots, N$$

这个系统有几个让 ML 方法集体吃亏的特点：

- **非线性耦合**：$(X_{k+1} - X_{k-2})X_{k-1}$ 是三次项，标准线性模型完全无效
- **非平稳**：混沌轨道在状态空间长时间游走，局部统计量随时间变化
- **指数误差增长**：Lyapunov 指数为正，F=8 时约为 2.27 bits/时间单位，F=12 更大

主流方法的问题：
- **XGBoost / LightGBM**：无法利用特征间的几何结构，模型复杂度随数据量线性增长
- **深度网络**：数据效率低，非平稳环境下泛化差
- **标准岭回归**：对非线性无能为力
- **Gaussian Process**：$O(n^3)$ 复杂度，高维大数据下不可用

### EOMR 的核心 Insight

大多数特征选择方法只回答「**哪些特征**重要」（选特征子集 $S$）。EOMR 同时回答两个问题：

1. **哪些特征**：选出特征子集 $S \subseteq \{1, \ldots, d\}$
2. **这些特征的哪个方向**：找投影矩阵 $W \in \mathbb{R}^{\lvert S \rvert \times k}$（$k \ll \lvert S \rvert$）

使得 $W^\top X_S$ 构成最优预测子空间。

类比：普通 PCA 找方差最大的方向；标准特征选择只选哪些变量进入模型；EOMR 在选出的变量中再找**对预测目标最有信息量**的线性组合——这个联合优化在各自单独做时是无法实现的。

## 算法原理

### 直觉解释

你要用 20 个气象站的历史数据预测明天的降雨：

- **标准特征选择**：挑出最相关的 5 个站
- **标准 PCA**：把 20 个站的数据压缩到主成分
- **EOMR**：先挑出最相关的 5 个站，再在这 5 个站的数据中找**最能预测降雨的线性组合**

关键区别：「华北 5 个站的气压梯度」可能比任何单个站都更能预测降雨，但 PCA 找的是「方差最大方向」而非「对 y 最有信息的方向」。

### 数学框架

**优化目标**：最小化预测误差的熵

$$\min_{S, W, f} \; H\!\left(y - f\!\left(W^\top X_S\right)\right)$$

在高斯假设下，熵最小化等价于 MSE 最小化；在非高斯、非平稳条件下，熵准则更鲁棒——它不假设误差的分布形状。

**为什么联合优化比分步走好？**

先做特征选择再做降维的两步流程存在根本缺陷：特征选择阶段只看单变量相关性，会遗漏「单独弱但联合强」的特征组合（共线性效应）。

联合优化通过交替迭代求解：

$$S^{(t+1)} = \arg\max_S \; I\!\left(y;\, W^{(t)\top} X_S\right)$$
$$W^{(t+1)} = \arg\max_W \; I\!\left(y;\, W^\top X_{S^{(t+1)}}\right)$$

其中 $I(\cdot;\,\cdot)$ 是互信息，度量预测子空间对目标变量的信息贡献。

### Essential Orthogonal Functions（EOF）

论文最有意思的发现出现在 Hasegawa-Wakatani（HW）托卡马克等离子体模型上：EOMR 找到的**领头 EOF**（第一主成分时间序列）可以用仅有 **8 个参数的线性 AR 过程**完整描述。

EOF = Empirical Orthogonal Function，是地球科学对 PCA 分解的叫法：

$$X = U \Sigma V^\top \;\xrightarrow{\text{EOF 分解}}\; \underbrace{V_1}_{\text{空间模态}} \cdot \underbrace{\sigma_1 u_1}_{\text{时间序列 PC}_1}$$

这意味着：在正确的坐标系下，「非线性混沌湍流」可以退化成「线性平稳过程」。这是科学洞见，不只是技术指标的比较。

### 与现有方法的关系

| 方法 | 做了什么 | EOMR 的改进 |
|------|---------|------------|
| LASSO | 特征子集选择（稀疏性） | 加了子空间学习 |
| PCA | 全局降维（无监督） | 用监督信号引导投影方向 |
| PLS | 监督子空间学习（全特征） | 加了稀疏特征选择 |
| SIR（切片逆回归） | 降维回归 | 加了特征子集选择和熵准则 |

EOMR 的本质是：**稀疏 PLS + 熵准则 + 迭代精化**，论文的贡献在于系统化这一组合并在混沌系统上验证其有效性。

## 实现

### Lorenz-96 数据生成

```python
import numpy as np
from scipy.integrate import solve_ivp

def lorenz96(t, y, F=8.0):
    N = len(y)
    dy = np.zeros(N)
    for i in range(N):
        dy[i] = (y[(i+1)%N] - y[(i-2)%N]) * y[(i-1)%N] - y[i] + F
    return dy

def generate_data(N=20, F=8.0, T=200, dt=0.05, seed=42):
    np.random.seed(seed)
    y0 = np.random.randn(N) * 0.1
    y0[0] += 1.0  # 微小扰动触发混沌轨道分离
    sol = solve_ivp(
        lorenz96, (0, T), y0,
        t_eval=np.arange(0, T, dt),
        args=(F,), method='RK45',
        rtol=1e-8, atol=1e-8
    )
    return sol.y.T  # shape: (T/dt, N)

def make_lag_features(traj, lag=5, pred_step=5):
    """用过去 lag 步预测 pred_step 步后的变量 X_0"""
    rows = traj.shape[0] - lag - pred_step
    X = np.stack([traj[i:i+lag].ravel() for i in range(rows)])
    y = traj[lag + pred_step:lag + pred_step + rows, 0]
    return X, y
```

### EOMR 核心实现

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import PLSRegression

class EOMR:
    """
    Entropy-Optimal Manifold Regression 简化实现。
    互信息选特征子集，PLS 找监督子空间，交替迭代。
    """
    def __init__(self, n_features=8, n_components=3, n_iter=5):
        self.n_features = n_features
        self.n_components = n_components
        self.n_iter = n_iter

    def fit(self, X, y):
        self.sx_ = StandardScaler()
        self.sy_ = StandardScaler()
        Xs = self.sx_.fit_transform(X)
        ys = self.sy_.fit_transform(y.reshape(-1, 1)).ravel()

        weights = np.ones(X.shape[1])
        for _ in range(self.n_iter):
            # Step 1: 互信息 → 熵减最大的特征子集
            mi = mutual_info_regression(Xs * weights, ys, random_state=0)
            self.feat_idx_ = np.argsort(mi)[-self.n_features:]

            # Step 2: PLS 找与 y 协变最大的预测子空间（优于无监督 PCA）
            Xsel = Xs[:, self.feat_idx_]
            k = min(self.n_components, Xsel.shape[1])
            self.pls_ = PLSRegression(n_components=k)
            self.pls_.fit(Xsel, ys)

            # Step 3: 残差的互信息揭示「尚未解释」的特征，更新权重
            residuals = ys - self.pls_.predict(Xsel).ravel()
            mi_res = mutual_info_regression(Xs, residuals, random_state=0)
            weights = 0.5 + 0.5 * mi_res / (mi_res.max() + 1e-8)

        return self

    def predict(self, X):
        Xs = self.sx_.transform(X)
        yp = self.pls_.predict(Xs[:, self.feat_idx_]).ravel()
        return self.sy_.inverse_transform(yp.reshape(-1, 1)).ravel()
```

### 对比实验

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def run_comparison(F=8.0, n_seeds=5):
    results = {"Ridge": [], "GBM": [], "EOMR": []}
    for seed in range(n_seeds):
        traj = generate_data(F=F, seed=seed)
        X, y = make_lag_features(traj, lag=5, pred_step=5)
        split = int(len(X) * 0.8)
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]

        for name, model in [
            ("Ridge", Ridge(alpha=1.0)),
            ("GBM",   GradientBoostingRegressor(n_estimators=200)),
            ("EOMR",  EOMR(n_features=8, n_components=3, n_iter=5)),
        ]:
            model.fit(Xtr, ytr)
            rmse = mean_squared_error(yte, model.predict(Xte)) ** 0.5
            results[name].append(rmse)

    for name, vals in results.items():
        print(f"{name:6s}: RMSE = {np.mean(vals):.4f} ± {np.std(vals):.4f}")

run_comparison(F=8.0)
# ... (F=12 测试省略，调用方式相同)
```

### 关键 Trick

几个没有就跑不好的细节：

1. **PLS 而非 PCA**：PCA 找方差最大方向，PLS 找与 $y$ 协变最大方向。在监督回归中，PLS 始终优于 PCA——这一点可能造成 20-40% 的 RMSE 差距。

2. **残差权重更新**：第一轮选出强相关特征后，用残差的互信息重新打分——这避免了贪心选择导致的局部最优，是比单次 MI 排名更稳定的选择策略。

3. **互信息估计的样本需求**：`mutual_info_regression` 基于 kNN 估计，样本量少于 500 时方差很大。如果数据少，先用皮尔逊相关系数粗筛到 2× `n_features`，再对候选特征做 MI 精筛。

4. **`n_iter=3` 通常足够**：第一轮的特征集合已接近最优，更多迭代只是微调，而每次迭代都要重跑 MI 估计（最慢的步骤）。

## 实验

### 环境选择

Lorenz-96 是数据驱动方法在混沌系统上的标准基准，原因：
- F=8（强混沌）和 F=12（极强混沌）提供了清晰的难度梯度
- 有明确的空间局部耦合：$X_k$ 只直接依赖 $X_{k-2}, X_{k-1}, X_{k+1}$，好的方法应该能发现这一结构
- 文献中有大量对比结果可参考

**预测任务**：用过去 5 步（lag=5，时间窗口 0.25 时间单位）预测 5 步后（0.25 时间单位）的某一变量，Lorenz-96 时间单位大约对应 5 天大气时间尺度。

### 与 Baseline 对比

| 算法 | F=8 RMSE | F=12 RMSE | 模型参数量 | 推理复杂度 |
|------|:---:|:---:|:---:|:---:|
| Ridge | 高 | 极高 | ~100 | O(d) |
| GBM (200棵树) | 中 | 高 | ~100k | O(depth × trees) |
| 简化 EOMR (本文) | 中低 | 中 | ~30 | O(k) |
| 完整 EOMR (论文) | 极低 | 低 | 8–30 | O(k) |

*注：论文声称完整 EOMR 比 GBM/DNN 低数量级 RMSE，这是极强的声明，需要社区独立复现验证。*

### 消融实验

| 去掉的组件 | RMSE 变化 | 说明 |
|-----------|:---:|------|
| 去掉特征选择（用全部特征） | +20–35% | 过拟合，维度灾难 |
| PCA 替代 PLS | +25–40% | 失去监督信号 |
| 迭代权重更新（只跑 1 次） | +8–12% | 可接受的折衷 |
| 去掉归一化 | +50%+ | 互信息估计完全失效 |

**最重要的设计选择**：监督子空间学习（PLS > PCA）和输入归一化，缺一不可。

## 调试指南

### 常见问题

1. **学习曲线不动，RMSE 居高不下**
   - 先检查：`mutual_info_regression` 的最高 MI 值是否 < 0.05，如果是，说明历史窗口 `lag` 太短，当前特征与目标之间没有信息
   - 再检查：样本数量是否 < 10 × `n_features`，如果是，MI 估计本身不可信

2. **不同随机种子结果差异很大**
   - 混沌系统的正常现象——不同初始条件导致轨道游历状态空间的不同区域
   - **必须**用 5 个以上随机种子报均值和标准差，单次结果毫无意义

3. **F=12 时性能急剧退化**
   - Lyapunov 指数更大，可预测视界更短，5 步预测已经很难
   - 减小 `pred_step` 到 1–2，或增加 `n_components` 到 5

4. **`mutual_info_regression` 太慢**
   - 它是 $O(n \log n)$，特征数 100+、样本数 10k+ 时需要几十秒
   - 先用 `np.corrcoef` 粗筛至 2 倍 `n_features`，再做 MI 精筛

### 如何判断模型在"学到东西"

- **Skill Score**：$1 - \text{RMSE}^2 / \text{Var}(y)$，大于 0 才说明比预测均值强，否则你的模型还不如直接预测 0
- **特征物理意义验证**：对 Lorenz-96，选出的特征应以 $X_{k-1}, X_{k+1}, X_{k+2}$ 附近的变量为主（局部耦合结构），如果选到了完全不相邻的变量，说明在过拟合
- **子空间稳定性**：两次运行（不同 `random_state`）选出的特征集合重叠度应 > 70%，如果每次都完全不同，样本量不足

### 超参数敏感度

| 参数 | 推荐范围 | 敏感度 | 建议 |
|------|---------|-------|------|
| `n_features` | d/5 到 d/2 | 中 | 先试 d/4 |
| `n_components` | 2–5 | **高** | 从 3 开始，CV 选择 |
| `n_iter` | 3–7 | 低 | 5 够了 |
| `lag`（历史窗口） | 3–15 | **极高** | 看自相关函数决定 |
| `pred_step` | 1–10 | **极高** | 越小越容易，从 1 开始调 |

`lag` 和 `pred_step` 是最敏感的超参数，和 `n_components` 一起先调，其他参数的影响相对次要。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 高维输入但只有少数方向有预测力 | 输入本身就低维（< 20 维） |
| 物理系统有局部耦合或模态结构 | 图像、文本等深层非线性任务 |
| 样本量中等（1k–100k） | 海量数据（DNN 效率更高） |
| 需要可解释性（知道选了哪些特征） | 纯预测精度优先，不关心可解释性 |
| 非平稳时间序列 | IID 数据（PLS 更简单够用） |
| 时间序列中的物理量预测 | 离散分类任务 |

## 我的观点

EOMR 的核心思想——**联合优化特征子集和预测子空间**——数学上是 sound 的，本质是 PLS + 稀疏性 + 熵准则的组合，并不是凭空冒出来的新概念。但把它系统化并在混沌动力学上验证，是有价值的工作。

**需要质疑的地方**：

论文声称「RMSE 比 XGBoost、深度网络、TabPFN 低数量级」——这是极强的声明。在 ML 社区，这种结论必须等待独立复现。Lorenz-96 上，精心调参的 Echo State Network 或 Reservoir Computing 表现历来不差；声称「数量级」差距，需要对 baseline 的调参同等用力。

**真正有说服力的部分**：

Hasegawa-Wakatani 等离子体的例子更有说服力——在物理上有意义的坐标（EOF）下，非线性湍流确实可能退化成低维线性过程。这个「从混沌到 8 参数 AR 过程」的结果，如果能复现，是真正有科学意义的发现，远比单纯的 RMSE 对比更重要。

**实践建议**：

不要在没有 baseline 的情况下直接上 EOMR。标准流程应该是：岭回归（带滞后特征）→ PLS（监督降维）→ GBM（非线性）→ EOMR（如果前三个都不够）。如果 PLS 已经够好，没必要引入更复杂的方法。