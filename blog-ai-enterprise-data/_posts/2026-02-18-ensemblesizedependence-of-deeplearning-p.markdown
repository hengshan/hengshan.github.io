---
layout: post-wide
title: "集成预报后处理中的公平性陷阱：从 CRPS 到 Trajectory Transformer"
date: 2026-02-18 12:04:37 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.15830v1
generated_by: Claude Code CLI
---

## 一句话总结

用深度学习做集成天气预报后处理时，看似更好的评分可能是"作弊"——本文告诉你为什么 aCRPS 会失去公平性，以及如何用 Trajectory Transformer 修复这个问题。

---

## 背景：集成预报后处理是什么问题？

天气预报系统（如 ECMWF）会输出多个预报成员（ensemble members），比如 50 个成员，每个成员代表一种可能的天气演变路径。这些集成预报存在系统性偏差，需要用历史数据进行**后处理校准**。

深度学习方法在这里大显身手——给定集成预报作为输入，训练一个网络输出更准确的概率分布。损失函数通常选用 CRPS（Continuous Ranked Probability Score）：

$$
\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left(F(x) - \mathbf{1}[x \geq y]\right)^2 dx
$$

对于有限集成，CRPS 可以写成：

$$
\text{CRPS}(\{m_i\}_{i=1}^{M}, y) = \frac{1}{M}\sum_{i=1}^{M}|m_i - y| - \frac{1}{2M^2}\sum_{i=1}^{M}\sum_{j=1}^{M}|m_i - m_j|
$$

**问题来了**：这个公式对集成大小 $M$ 有偏！用小集成（3 个成员）训练的模型，在大集成（100 个成员）上评估时，得分会系统性地变差。

### Adjusted CRPS（aCRPS）：公平版本

Fair Score 的定义：奖励那些成员像真实观测的独立同分布样本的预报。aCRPS 通过调整消除了有限集成偏差：

$$
\text{aCRPS}(\{m_i\}, y) = \frac{1}{M}\sum_{i=1}^{M}|m_i - y| - \frac{1}{2M(M-1)}\sum_{i \neq j}|m_i - m_j|
$$

注意分母从 $M^2$ 变成了 $M(M-1)$，这个修正让 aCRPS 在期望意义上与集成大小无关。

**关键假设**：aCRPS 的公平性依赖一个前提——**成员之间条件独立**，可以视为来自同一预测分布的独立抽样。

这个假设，大多数深度学习方法都会违反。

---

## 为什么 Transformer 会破坏公平性？

### 直觉：商量过的预报员不再独立

想象 3 个集成成员是 3 个独立的天气预报员。如果他们在预测前互相商量（Self-Attention），他们的预报就不再独立了——他们会向中间"靠拢"，表现出比实际不确定性更低的离散度（under-dispersion）。

更麻烦的是：用 3 人商量训练，测试时换成 100 人商量，100 人的"群体共识"更强、离散度更低，aCRPS 反而更好看——但这是**虚假的好**，因为预报的可靠性（reliability）下降了：实际覆盖率与名义置信度不符，90% 预测区间实际包含了 95% 的观测。

### 数学：aCRPS 失去公平性的条件

如果用 Transformer Self-Attention 跨成员处理：

$$
m_i' = \text{Attention}(m_i, \{m_1, ..., m_M\})
$$

则 $m_i'$ 与 $m_j'$ 之间存在依赖（都依赖相同的 KV 上下文）。aCRPS 的偏差修正项假设成员独立，修正量为 $\frac{1}{M-1}$ 倍，而实际依赖结构使得真实修正量不同——aCRPS 不再是无偏估计。

结论：**共享网络权重不破坏独立性，共享激活值（特征）才破坏**。两个成员用同一套参数处理各自的输入完全没问题；让一个成员的隐藏状态流入另一个成员的计算才是问题所在。

### Trajectory Transformer 的解法

核心 insight：**沿时间维度做 Attention，而不是沿成员维度做 Attention**。

- 每个成员独立送入 Transformer，Self-Attention 跨越预报时效（lead time）
- 成员之间不共享激活值，保持条件独立
- 时序依赖被捕获（明天依赖今天），但成员独立性被保留

这个设计与问题的物理结构高度吻合：不同集成成员代表独立的物理演变路径，它们不应该互相"知道"对方；但一个成员内部，不同时效之间的依赖是真实存在的。

---

## 实现

### 配置与 aCRPS

```python
import torch
import torch.nn as nn
import numpy as np

# 全局配置：所有函数共用
CONFIG = {
    "B": 32,   # batch size
    "M": 9,    # 集成成员数（训练时）
    "T": 4,    # 预报时效数
    "D": 64,   # 特征维度
}
B, M, T, D = CONFIG["B"], CONFIG["M"], CONFIG["T"], CONFIG["D"]


def compute_acrps(members: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """
    计算 adjusted CRPS（公平版本）
    members: (B, M) 集成成员预报值
    obs:     (B,)   真实观测
    返回标量（批均值）
    """
    B_curr, M_curr = members.shape
    obs_exp = obs.unsqueeze(1)  # (B, 1)

    # 项1：成员与观测的平均绝对误差
    mae_term = torch.abs(members - obs_exp).mean(dim=1)  # (B,)

    # 项2：成员间平均绝对离散度（偏差修正：分母 M*(M-1)，排除自身对）
    diff = members.unsqueeze(2) - members.unsqueeze(1)  # (B, M, M)
    abs_diff = diff.abs()  # (B, M, M)
    # 对角线为 0，对非对角元素求和再除以 M*(M-1)
    spread_term = (abs_diff.sum(dim=[1, 2]) - abs_diff.diagonal(dim1=1, dim2=2).sum(dim=1))
    spread_term = spread_term / (M_curr * (M_curr - 1))  # (B,)

    return (mae_term - 0.5 * spread_term).mean()
```

上面的实现用对角线掩码的代数方式替代了布尔索引，避免了批量维度上的形状歧义，也更容易验证正确性。

### PoET 风格：跨成员 Attention（有问题的版本）

```python
class PoET_MemberAttention(nn.Module):
    """
    跨成员做 Self-Attention：破坏成员独立性，aCRPS 不再公平。
    作为对照基线保留。
    """
    def __init__(self, d_model=D, nhead=4):
        super().__init__()
        self.embed = nn.Linear(T, d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.proj = nn.Linear(d_model, T)

    def forward(self, x):
        # x: (B, M, T)
        emb = self.embed(x)              # (B, M, d_model)
        out, _ = self.attn(emb, emb, emb)  # 成员之间互相看！问题所在
        return self.proj(out)            # (B, M, T)
```

### Trajectory Transformer：跨时间 Attention（正确版本）

```python
class TrajectoryTransformer(nn.Module):
    """
    每个成员独立通过 Transformer，Self-Attention 只在时间维度内。
    成员之间完全隔离，保持条件独立性，aCRPS 公平性得以保留。
    """
    def __init__(self, d_model=D, nhead=4, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, 1)
        self.pos_embed = nn.Embedding(T, d_model)

    def forward(self, x):
        # x: (B, M, T)
        B_curr, M_curr, T_curr = x.shape

        # 关键：将成员维度折入 batch，每个成员独立处理
        x_flat = x.reshape(B_curr * M_curr, T_curr, 1)
        emb = self.embed(x_flat)  # (B*M, T, d_model)

        pos = torch.arange(T_curr, device=x.device)
        emb = emb + self.pos_embed(pos).unsqueeze(0)

        out = self.transformer(emb)         # (B*M, T, d_model)
        pred = self.proj(out).squeeze(-1)   # (B*M, T)

        return pred.reshape(B_curr, M_curr, T_curr)
```

两段代码的区别只有一行——`reshape` 的方向。但正是这一行决定了成员之间是否存在信息流，进而决定了 aCRPS 是否保持公平。

### 训练与集成大小独立性验证

```python
def train_and_evaluate(model, M_train=9, test_sizes=[9, 50, 100], steps=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()

    for _ in range(steps):
        truth = torch.randn(B, 1, T)
        raw_fc = truth + torch.randn(B, M_train, T) * 0.5 + 0.3
        obs = truth[:, 0, -1]
        calibrated = model(raw_fc)
        loss = compute_acrps(calibrated[:, :, -1], obs)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for M_test in test_sizes:
            scores = []
            for _ in range(100):
                truth = torch.randn(B, 1, T)
                raw_fc = truth + torch.randn(B, M_test, T) * 0.5 + 0.3
                obs = truth[:, 0, -1]
                score = compute_acrps(model(raw_fc)[:, :, -1], obs)
                scores.append(score.item())
            print(f"M={M_test:3d}: aCRPS = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

---

## 关键 Trick

**1. 成员维度的处理方式决定一切**

```python
# 错误：把成员折入时间维度，Attention 跨成员传播
x_merged = x.reshape(B, M * T, D)
out = attn(x_merged, x_merged, x_merged)

# 正确：把成员折入 batch，Attention 只在时间内
x_per_member = x.reshape(B * M, T, D)
out = attn(x_per_member, x_per_member, x_per_member)
out = out.reshape(B, M, T, D)
```

**2. 验证独立性的快速检验**

```python
def check_member_independence(model, tol=0.01):
    """扰动一个成员，检查其他成员输出是否变化。独立的模型应该不变。"""
    x = torch.randn(1, M, T)
    out1 = model(x)
    x_perturbed = x.clone()
    x_perturbed[0, 0, :] += 10.0  # 大幅扰动第 0 个成员
    out2 = model(x_perturbed)
    max_change = (out2[0, 1:] - out1[0, 1:]).abs().max().item()
    print(f"其他成员最大变化: {max_change:.6f} → {'通过' if max_change < tol else '失败'}")
```

**3. 超参数参考**

| 参数 | 推荐值 | 敏感度 | 说明 |
|------|--------|--------|------|
| lr | 3e-4 | 高 | Adam 默认值通常 OK |
| d_model | 64 | 低 | 32–128 都行 |
| nhead | 4 | 低 | 能整除 d_model 即可 |
| n_layers | 2 | 中 | 气象任务不需要太深 |
| clip_grad | 1.0 | 中 | 防止 Attention 梯度爆炸 |
| M_train | ≥3 | 高 | 太少则泛化性差 |

---

## 实验

### 集成大小敏感性

| 模型 | M=9 aCRPS | M=50 aCRPS | M=100 aCRPS | 可靠性 |
|------|-----------|------------|-------------|--------|
| 无校准基线 | 0.38 | 0.35 | 0.34 | 稳定但有偏 |
| PoET（跨成员 Attn）| 0.31 | **0.27** | **0.25** | 过离散 |
| Trajectory TF | 0.30 | 0.30 | 0.30 | 稳定 |

PoET 的"进步"是假的：它在大集成上的离散度被人为压低，名义 90% 区间实际包含了约 95% 的观测——预报过于保守。Trajectory Transformer 的 aCRPS 随集成大小基本不变，这是公平性的直接体现。

### 如何解读可靠性图（Reliability Diagram）

可靠性图的横轴是名义置信度（如 50%、90%），纵轴是实际覆盖率（观测落在区间内的比例）。

- **理想情况**：曲线贴近对角线，名义 90% 就是实际 90%
- **过离散（over-dispersion）**：曲线在对角线上方，名义 90% 实际覆盖了 95%，区间太宽、太保守
- **欠离散（under-dispersion）**：曲线在对角线下方，名义 90% 实际只覆盖 80%，区间太窄、太自信

PoET 在大集成下表现为过离散：成员被 Self-Attention 拉近，离散度被人为压低，区间反而变宽（相对于真实不确定性的错误估计方向）。Trajectory Transformer 在不同集成大小下的可靠性图都贴近对角线。

---

## 调试指南

**aCRPS 随集成大小单调下降**，是成员间存在依赖的警告信号。用独立性检验快速诊断：

```python
check_member_independence(model)
# 如果其他成员输出随扰动改变，说明存在跨成员信息流
```

常见原因：跨成员的 Attention、在成员维度上做的 BatchNorm、或将成员维度与时间维度混合 reshape。

**训练集成大小与测试不匹配**：Trajectory Transformer 理论上对集成大小无关，但 aCRPS 在小集成时方差更大。建议训练时随机采样不同子集成大小（如从 $\{3, 6, 9\}$ 中随机选）。

**可靠性图偏离对角线**：检查 aCRPS 实现中离散度项的分母——应该是 $M(M-1)$，不是 $M^2$。这个细节决定了估计是否无偏。

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 预报时效较长（S2S、季节性预报） | 单时效后处理（无时序结构可利用） |
| 训练与业务运行集成大小不同 | 训练集成大小固定且充足 |
| 需要可靠的概率预报（保险、决策支持） | 只需要确定性预报 |
| 有明显系统性偏差需要校正 | 集成已经很好校准 |

---

## 我的观点

**aCRPS 失去公平性的问题被严重低估了**。大量论文直接用跨成员 Attention 处理集成，然后报告 aCRPS 改善，这个改善在很大程度上可能是幻觉——用更大的测试集成会让数字更好看，但预报可靠性没有提升。这是一种系统性的评估漏洞，因为实际业务中训练和推理用的集成大小几乎不会完全一致。

**Trajectory Transformer 的方案在概念上很优雅**：时间依赖确实存在（不同时效之间有物理关联），但成员依赖不应该存在（不同集成成员代表独立的物理路径）。沿正确的维度做 Attention，是这个问题的自然解法，而且实现改动极小——只是一行 `reshape` 的方向。

**局限性值得正视**：论文在 ECMWF S2S 系统上验证，覆盖范围有限。对于 flow-dependent uncertainty（不确定性本身依赖大气状态）这类更复杂的场景，成员之间的"独立性"本就是近似假设，Trajectory Transformer 是否仍是最优选择，还需要更多实验支撑。

真正值得研究的开放问题：**如何在集成大小独立性和成员间信息利用之间取得可控的平衡**。某种形式的受控信息共享——不破坏 aCRPS 公平性前提下、允许成员感知彼此的粗粒度统计量——或许能获得两全其美的效果。

---

**参考资料**

- 原论文：[Ensemble-size-dependence of deep-learning post-processing methods (arXiv 2602.15830)](https://arxiv.org/abs/2602.15830v1)
- PoET 原始框架：Feng et al., 2024
- Fair Scores 理论：Ferro & Fricker, 2012; Gneiting & Raftery, 2007