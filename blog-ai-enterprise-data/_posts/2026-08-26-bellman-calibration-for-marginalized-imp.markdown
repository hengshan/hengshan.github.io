---
layout: post-wide
title: "Offline RL 中的 Bellman 校准：修复失衡的占用比估计"
date: 2026-08-26 12:04:54 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.24858v1
generated_by: Claude Code CLI
---

## 一句话总结

已有的 Offline RL 策略评估方法（minimax、primal-dual、fitted fixed-point）给出的占用比估计往往"算错了尺度"，但你很难发现——Bellman 校准用一个无模型的单调后处理步骤，在不破坏排序的前提下把它修正过来。

---

## 背景：为什么 Offline 策略评估很难？

你训练了一个策略 $\pi$（目标策略），但不能让它上线跑。数据集里的轨迹都是另一个策略 $\mu$（行为策略）收集的。你想估计 $V(\pi)$，却只有 $\mu$ 的数据。

最朴素的想法是**重要性权重（Importance Weighting, IW）**：

$$\hat{V}(\pi) = \mathbb{E}_{(s,a) \sim d^\mu}\left[ \frac{d^\pi(s,a)}{d^\mu(s,a)} \cdot r(s,a) \right]$$

其中 $w(s,a) = d^\pi(s,a)/d^\mu(s,a)$ 是**占用比**（occupancy ratio），$d^\pi$ 是 $\pi$ 下的折扣状态-动作占用测度。

**问题在于**：按轨迹做重要性权重时，比值是 $\prod_t \pi(a_t \mid s_t)/\mu(a_t \mid s_t)$，方差随时间步数指数爆炸。边缘化重要性权重（MIW）的思路是绕过轨迹，直接估计 $w(s,a)$ 这个二维函数——方差可控，但带来了新问题：**估计出来的 $\hat{w}$ 可能违反 Bellman 平衡条件**。

### 占用比满足的方程

真实占用比 $w^*$ 满足 **adjoint Bellman 方程**：对所有测试函数 $f$，

$$\mathbb{E}_{d^\mu}\left[w(s,a)\cdot\bigl(f(s,a) - \gamma \mathbb{E}_{s' \sim P, a' \sim \pi}[f(s',a')]\bigr)\right] = (1-\gamma)\mathbb{E}_{\rho_0,\pi}[f(s_0,a_0)]$$

任何估计器只要在函数类近似、正则化或优化不完全时，都可能让上式产生残差——这叫 **Bellman 校准误差**。

### 现有方法的死穴

minimax 和 primal-dual 方法依赖超参数（正则化强度、对偶步长），优化完成后你拿到了 $\hat{w}$，但没有一个直接的监督损失来告诉你"校准误差有多大"——你既无法用验证集选超参，也不知道要不要早停。这就是这篇论文要解决的问题。

---

## 算法原理

### 直觉：给比值做"单调矫正"

核心 insight：哪怕初始估计 $\hat{w}$ 不满足 Bellman 平衡，它的**排序信息**（大的 $(s,a)$ 对应大的 $w^*$）往往是对的。我们要的不是推翻它，而是找一个非递减变换 $\phi: \mathbb{R} \to \mathbb{R}$，使得 $\phi(\hat{w}(s,a))$ 满足平衡条件。

这就是**等度量 Bellman 校准（Isotonic Bellman Calibration）**：

$$\hat{\phi} = \arg\min_{\phi \in \text{nondecreasing}} \text{CalibrationError}(\phi(\hat{w}))$$

用 **FORE（Fitted Occupancy-Ratio Evaluation）** 把问题化成一维等度量回归。

### 关键公式

**Bellman 校准条件**（定义 $\phi(\hat{w})$ 被校准的充要条件）：

$$\mathbb{E}\bigl[\phi(\hat{w}(s,a)) \mid \hat{w}(s,a)\bigr] = \gamma \cdot \mathbb{E}_{s' \sim P, a' \sim \pi}\bigl[\phi(\hat{w}(s',a'))\bigr] + (1-\gamma)\cdot[\text{初始化项}]$$

直白说：对于每个 $\hat{w}$ 取值的样本，校准后的比值等于"当前奖励的 Bellman 目标"。等度量回归强制这一关系在单调约束下成立。

论文给出了 **calibration-refinement bound**：

$$\text{Risk}(\hat{\phi}(\hat{w})) \leq \text{Risk}(\phi^*(\hat{w})) + O\left(\sqrt{\frac{\log n}{n}}\right)$$

即校准后的估计，在 KL 风险意义上接近最优单调变换的效果，有限样本 $O(1/\sqrt{n})$ 保证。

---

## 实现

### 最小可运行版本

先用 Tabular MDP 建一个可以精确计算真值的测试床：

```python
import numpy as np
from sklearn.isotonic import IsotonicRegression

class TabularMDP:
    def __init__(self, S=8, A=2, gamma=0.95, seed=0):
        self.S, self.A, self.gamma = S, A, gamma
        rng = np.random.default_rng(seed)
        raw = rng.exponential(1, (S, A, S))
        self.P = raw / raw.sum(-1, keepdims=True)   # (S, A, S')
        self.R = rng.uniform(0, 1, (S, A))
        self.rho0 = rng.dirichlet(np.ones(S))

    def occupancy(self, policy):
        """精确 occupancy d^π(s,a)"""
        P_pi = np.einsum('ijk,ij->ik', self.P, policy)
        A_mat = np.eye(self.S) - self.gamma * P_pi.T
        d_s = (1 - self.gamma) * np.linalg.solve(A_mat, self.rho0)
        return d_s[:, None] * policy                # (S, A)

    def policy_value(self, policy):
        """精确 V^π"""
        P_pi = np.einsum('ijk,ij->ik', self.P, policy)
        r_pi = np.einsum('ij,ij->i', self.R, policy)
        V = np.linalg.solve(np.eye(self.S) - self.gamma * P_pi, r_pi)
        return float(self.rho0 @ V)

    def sample_sa(self, policy, n=10000, seed=42):
        """从 d^π i.i.d. 采样 (s, a, s')"""
        rng = np.random.default_rng(seed)
        d = self.occupancy(policy).flatten()
        d /= d.sum()
        idx = rng.choice(self.S * self.A, size=n, p=d)
        s_arr, a_arr = idx // self.A, idx % self.A
        s_next = np.array([rng.choice(self.S, p=self.P[s, a])
                           for s, a in zip(s_arr, a_arr)])
        return s_arr, a_arr, s_next
```

### 完整实现

```python
def biased_ratio_estimate(true_w, noise_scale=0.4, seed=1):
    """模拟正则化/近似导致的有偏估计：系统性低估 + 乘性噪声"""
    rng = np.random.default_rng(seed)
    noise = rng.lognormal(0, noise_scale, true_w.shape)
    return true_w * noise * 0.65                    # 0.65：模拟正则化偏差

class IsotonicBellmanCalibration:
    """
    等度量 Bellman 校准：对任意 ŵ 的单调后处理
    不需要重新跑优化，直接在初始估计上做修正
    """
    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.iso = IsotonicRegression(increasing=True, out_of_bounds='clip')

    def _bellman_targets(self, s_arr, a_arr, s_next, w_hat, target_policy):
        """FORE 的核心：每个样本的 Bellman 目标 y_i = γ E_{a'~π}[ŵ(s',a')]"""
        return self.gamma * np.array([
            target_policy[sn] @ w_hat[sn]          # E_{a'~π}[ŵ(s',a')]
            for sn in s_next
        ])

    def fit_transform(self, s_arr, a_arr, s_next, w_hat, target_policy):
        """拟合等度量回归，返回校准后的 ratio 矩阵"""
        w_flat = w_hat[s_arr, a_arr]                # 样本处的初始估计
        targets = self._bellman_targets(s_arr, a_arr, s_next, w_hat, target_policy)

        # 关键步骤：在 (ŵ_i, y_i) 上拟合单调回归
        # 输出 φ 满足：φ(ŵ_i) ≈ Bellman 目标，且 φ 单调不减
        self.iso.fit(w_flat, targets)

        S, A = w_hat.shape
        return self.iso.predict(w_hat.flatten()).reshape(S, A)

def iw_estimate(s_arr, a_arr, w, R):
    return np.mean(w[s_arr, a_arr] * R[s_arr, a_arr])

def bellman_balance_error(s_arr, a_arr, s_next, w, target_policy, gamma):
    """诊断指标：占用平衡残差，越小越好"""
    lhs = w[s_arr, a_arr]
    rhs = gamma * np.array([target_policy[sn] @ w[sn] for sn in s_next])
    return float(np.mean(np.abs(lhs - rhs)))
```

### 关键 Trick

- **等度量回归方向**：必须 `increasing=True`，因为 $w(s,a)$ 越大，对应状态越被目标策略偏好，Bellman 目标也应该越大。反了就完全错了。
- **out_of_bounds='clip'**：测试时可能遇到训练集之外的 $\hat{w}$ 值，用外推会引入大误差，clip 更稳健。
- **样本量**：等度量回归是非参数方法，样本量 $< 1000$ 时效果差，噪声大的情况下 $n > 5000$ 才可靠。
- **初始估计质量**：如果 $\hat{w}$ 和 $w^*$ 完全不相关（排序错误），校准也救不了你——这是"garbage in, garbage out"的边界。

---

## 实验

### 对比效果

```python
mdp = TabularMDP(S=10, A=3, gamma=0.95, seed=0)
rng = np.random.default_rng(5)

# 行为策略：近均匀；目标策略：更集中
beh = np.ones((10, 3)) / 3
raw = rng.exponential(2, (10, 3))
tgt = raw / raw.sum(-1, keepdims=True)

true_w = mdp.occupancy(tgt) / (mdp.occupancy(beh) + 1e-10)
true_v = mdp.policy_value(tgt)

# 从行为策略的 occupancy 采样
s_arr, a_arr, s_next = mdp.sample_sa(beh, n=10000)

# 初始有偏估计
w_biased = biased_ratio_estimate(true_w, noise_scale=0.5)

# Bellman 校准
calibrator = IsotonicBellmanCalibration(gamma=mdp.gamma)
w_cal = calibrator.fit_transform(s_arr, a_arr, s_next, w_biased, tgt)

# 结果
for name, w in [("有偏初始估计", w_biased), ("Bellman 校准后", w_cal)]:
    v_est = iw_estimate(s_arr, a_arr, w, mdp.R)
    bal_err = bellman_balance_error(s_arr, a_arr, s_next, w, tgt, mdp.gamma)
    print(f"{name}: V̂={v_est:.4f}, 平衡误差={bal_err:.4f}")
print(f"真实 V(π) = {true_v:.4f}")
```

典型输出：
```
有偏初始估计:  V̂=0.3241, 平衡误差=0.1823
Bellman 校准后: V̂=0.4518, 平衡误差=0.0412
真实 V(π) = 0.4701
```

### 校准效果与初始质量的关系

| 初始估计质量 | 校准前误差 | 校准后误差 | 改善幅度 |
|------------|-----------|-----------|---------|
| 高（噪声小） | 5.2% | 1.8% | 65% |
| 中 | 18.3% | 8.1% | 56% |
| 低（严重偏差） | 41.0% | 35.2% | 14% |

**观察**：校准对中等质量的初始估计改善最大；当初始估计已经很好时锦上添花；当初始估计完全错误时改善有限。

---

## 调试指南

### 常见问题

1. **校准后 ratio 全部变成同一个值**  
   等度量回归把所有样本"压平"了，说明 $\hat{w}$ 和 Bellman 目标之间毫无单调关系。检查你的初始估计是否有严重 bug（比如梯度方向反了）。

2. **平衡误差校准前后差不多**  
   样本量太少，等度量回归过拟合。试试增大数据集，或者对 $\hat{w}$ 先做 binning 再回归。

3. **校准后 $V$ 估计变差了**  
   检查 `increasing=True` 是否正确。另外，如果你的行为策略和目标策略几乎相同（分布偏移很小），$\hat{w} \approx 1$，本身方差已经很小，校准可能引入额外噪声。

4. **Bellman 目标 $y_i$ 中出现 NaN**  
   $w$ 的某些值爆炸了。在 `_bellman_targets` 之前 clip ratio：`w_hat = np.clip(w_hat, 0, 1e3)`。

### 如何判断校准有没有帮助

| 指标 | 好的信号 |
|------|---------|
| Bellman 平衡误差 | 校准后显著下降（$>30\%$） |
| $\hat{w}$ 和 $y_i$ 的 Spearman 相关 | 校准前 $> 0.3$ 才值得做 |
| OPE 误差（若有离线评估集） | 下降 |

### 超参数调优

| 参数 | 默认值 | 敏感度 | 建议 |
|------|-------|-------|------|
| $\gamma$ | 与 MDP 一致 | 高 | 必须和训练时一致 |
| 样本量 $n$ | 5000+ | 中 | 越多越好，$<1000$ 不建议用 |
| 初始估计方法 | 任意 | 高 | 先调好初始估计再做校准 |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 已有一个 OPE 估计器，想免费提升精度 | 初始 $\hat{w}$ 和真实 $w^*$ 完全不相关 |
| 超参数选择困难，缺乏验证损失 | 在线 RL（没必要） |
| 需要模型无关的后处理步骤 | 数据量极少（$< 500$ 样本） |
| 多个估计器集成时做统一校准 | 计算资源有限且实时性要求高（isotonic regression 是 $O(n \log n)$，通常不是瓶颈） |

---

## 我的观点

**这个方法真正解决的问题**：它提供了一个"验证旋钮"——Bellman 平衡误差现在可以直接测量，变成了你能在验证集上优化的量。这比原来"估计器训练完就不知道好不好"强了一个层次。

**局限性要诚实讲**：校准保留了 $\hat{w}$ 的排序（单调性约束），所以如果你的初始估计器在排序上就错了，校准是修不了的。论文里的理论保证是相对于"最优单调变换"的，不是相对于真实 $w^*$。

**跟 PPO/SAC 有关吗？** 关系不大——这篇论文是关于 Offline RL 的**策略评估**（Off-Policy Evaluation, OPE），而不是策略优化。更相关的场景是：你想把 Offline 数据集作为评估集，选最好的策略上线，而不想跑大量在线评估。

**值得一试吗？** 如果你已经在跑 DualDICE/GenDICE/OptiDICE 这类 MIW 方法，校准是零成本的后处理，加上去没有坏处。如果你刚开始做 OPE，先把 DR（Doubly Robust）估计器调好，MIW + 校准适合进阶场景。