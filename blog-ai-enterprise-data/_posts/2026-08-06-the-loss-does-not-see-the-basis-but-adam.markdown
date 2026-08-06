---
layout: post-wide
title: "损失函数看不见基底，但 Adam 看见了"
date: 2026-08-06 08:06:11 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.05136v1
generated_by: Claude Code CLI
---

## 一句话总结

同样的损失、同样的初始化——梯度下降收敛到低秩解，Adam 不会。这不是调参问题，而是 Adam 打破了损失函数的一个深层代数对称性。

## 为什么这个问题值得关注？

矩阵分解 $W = UV^\top$ 在 ML 中无处不在：LoRA、矩阵补全、注意力机制的 $W_Q, W_K$……当我们用过参数化分解训练模型时，优化器的选择会影响最终解的结构，而不只是收敛速度。

这篇论文的核心发现：**Adam 从第一步就"选了一个坐标系"，并从此锁定在那个坐标系里**。梯度下降则没有这个偏见，它让损失函数的对称性自然地引导解走向低秩结构。

这不是玄学。论文给出了精确的代数刻画，并在矩阵感知（matrix sensing）和 Transformer 训练上验证了后果。

---

## 规范对称性：损失看不见的自由度

设模型参数是因式分解 $W = UV^\top$，其中 $U, V \in \mathbb{R}^{n \times r}$，而 $r$ 远大于真实秩。

**关键观察**：对任意正交矩阵 $Q \in O(r)$，变换

$$
(U, V) \mapsto (UQ,\ VQ)
$$

不改变乘积 $W = UV^\top$（因为 $(UQ)(VQ)^\top = UQQ^\top V^\top = UV^\top$）。

这意味着损失函数 $\mathcal{L}(UV^\top)$ 在这个变换下完全不变。如果把 $(U, V)$ 参数空间画出来，每个"真实的 $W$"对应一个 $O(r)$-维的等价类——论文称之为**规范轨道（gauge orbit）**。

**损失函数对规范轨道内的运动完全"失明"**。在同一个轨道里，哪个点都等价，损失一模一样。

那么优化器在这个巨大的自由度上会怎么表现？

---

## 梯度下降：顺势而为

梯度下降的更新是：

$$
U \leftarrow U - \eta \nabla_U \mathcal{L}, \quad V \leftarrow V - \eta \nabla_V \mathcal{L}
$$

因为 $\nabla_U \mathcal{L} = \frac{\partial \mathcal{L}}{\partial W} \cdot V$，当你对 $(U, V)$ 做规范变换 $(UQ, VQ)$ 时，梯度也跟着以一致的方式变换——GD 是**规范等变（gauge-equivariant）**的。

更重要的是，这种等变性继承了**梯度流（gradient flow）的隐式偏置**。理论上已知，从小初始化出发的梯度流会收敛到核范数最小的解——即最低秩的解。等变性把这个性质"传递"给了 GD。

---

## Adam：第一步就打破对称性

Adam 的更新用二阶矩估计来归一化梯度：

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t \odot g_t, \quad \Delta\theta = -\eta \cdot \frac{g_t}{\sqrt{v_t} + \epsilon}
$$

问题出在 $\odot$（逐元素）上。当你对 $U$ 做规范变换 $U \mapsto UQ$ 时，梯度 $g_U$ 变成 $g_U Q$，但二阶矩 $v_t$ 是按坐标平方累积的——它不以等变的方式变换。

结果是：**两个规范等价的初始化 $(U_0, V_0)$ 和 $(U_0 Q, V_0 Q)$，在 Adam 的第一步之后就走向不同的轨迹，最终产生不同的 $W$**。

这个"选基底"的过程是随机的（取决于初始化数值），却有持久的后果。

---

## 动手验证

下面的代码在过参数化矩阵分解上对比两种优化器，追踪恢复矩阵的有效秩变化：

```python
import torch, torch.nn as nn

def effective_rank(W: torch.Tensor) -> float:
    """用奇异值熵估算有效秩"""
    sv = torch.linalg.svdvals(W).abs()
    sv = sv[sv > 1e-8]
    p = sv / sv.sum()
    return torch.exp(-(p * (p + 1e-10).log()).sum()).item()

def run_experiment(optimizer_name: str, steps: int = 3000):
    torch.manual_seed(42)
    n, r_true, r_over = 30, 3, 12  # 真实秩 3，用秩 12 的过参数化分解

    A = torch.randn(n, r_true) * 0.5
    W_star = A @ A.T  # 目标：低秩矩阵

    # 关键：小初始化，让隐式偏置生效
    U = nn.Parameter(torch.randn(n, r_over) * 0.01)
    V = nn.Parameter(torch.randn(n, r_over) * 0.01)

    opt = (torch.optim.SGD([U, V], lr=5e-4) if optimizer_name == 'GD'
           else torch.optim.Adam([U, V], lr=1e-3))

    rank_history, loss_history = [], []
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * ((U @ V.T - W_star) ** 2).mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            rank_history.append(effective_rank(U @ V.T))
            loss_history.append(loss.item())

    return rank_history, loss_history

gd_rank, gd_loss     = run_experiment('GD')
adam_rank, adam_loss = run_experiment('Adam')

print(f"GD   最终有效秩: {gd_rank[-1]:.2f}，最终损失: {gd_loss[-1]:.6f}")
print(f"Adam 最终有效秩: {adam_rank[-1]:.2f}，最终损失: {adam_loss[-1]:.6f}")
# GD 的有效秩收敛到接近 3；Adam 的损失可能更低，但有效秩明显偏高
```

验证规范等变性失效——Adam 第一步就分叉：

```python
def gauge_divergence_test(steps: int = 200):
    torch.manual_seed(0)
    n, r = 10, 6
    W_star = torch.randn(n, n)
    Q, _ = torch.linalg.qr(torch.randn(r, r))  # 随机正交矩阵

    U0 = torch.randn(n, r) * 0.01
    V0 = torch.randn(n, r) * 0.01

    divergences = {}
    for name in ['GD', 'Adam']:
        p1 = [nn.Parameter(U0.clone()),        nn.Parameter(V0.clone())]
        p2 = [nn.Parameter((U0 @ Q).clone()), nn.Parameter((V0 @ Q).clone())]

        def make_opt(p):
            return (torch.optim.SGD(p, lr=1e-3) if name == 'GD'
                    else torch.optim.Adam(p, lr=1e-3))

        opt1, opt2 = make_opt(p1), make_opt(p2)
        dists = []
        for _ in range(steps):
            for opt, (U, V) in [(opt1, p1), (opt2, p2)]:
                opt.zero_grad()
                (0.5 * ((U @ V.T - W_star) ** 2).mean()).backward()
                opt.step()
            W1 = (p1[0] @ p1[1].T).detach()
            W2 = (p2[0] @ p2[1].T).detach()
            dists.append(((W1 - W2).norm() / W1.norm()).item())
        divergences[name] = dists

    return divergences
    # GD:   dists 始终 ≈ 0，两条轨迹的 W 几乎相同
    # Adam: 第一步即分叉，dists 迅速拉开
```

---

## 哪些优化器是规范等变的？

论文给出了一个结构定理：**无记忆的规范等变更新规则，恰好是"Gram 确定的左预条件子"**。

直白地说：预条件子作用在 $U$ 上时，只能依赖于 $G_U = U^\top U$（Gram 矩阵），不能依赖于 $U$ 本身的具体坐标。

| 优化器 | 规范等变？ | 原因 |
|--------|-----------|------|
| GD / SGD | ✓ | 预条件子 = $I$，平凡满足 |
| Momentum | ✓ | 动量不改变方向等变性 |
| Shared-scalar Adam | ✓ | 全局标量归一化，不破坏坐标对称性 |
| Muon | ✓ | Newton-Schulz 正交化是 Gram 确定的 |
| Shampoo | ✓ | 使用 $(U^\top U)^{-1/4}$ 作为预条件子 |
| **标准 Adam** | **✗** | 逐坐标归一化依赖 $U$ 的具体基底 |
| **RMSProp** | **✗** | 同上 |

"Shared-scalar Adam"是一个有趣的中间态：把 Adam 的逐元素 $\sqrt{v_t}$ 换成所有坐标的全局 RMS，就能恢复等变性。论文用一个单参数插值（从逐坐标到全局标量）展示了低秩偏置如何随等变程度单调地恢复——干净的消融实验。

---

## Transformer 中的具体后果

注意力机制的有效信息是乘积 $W_Q^\top W_K$（决定 Query-Key 相似度）。$(W_Q, W_K)$ 存在规范对称性：$(W_Q R, W_K R)$（$R$ 为正交矩阵）给出完全相同的 $W_Q^\top W_K$。

论文实验发现：用 Adam 训练时，两个规范等价的初始化

$$
(W_Q^{(0)},\, W_K^{(0)}) \quad \text{vs.} \quad (W_Q^{(0)} R,\, W_K^{(0)} R)
$$

在第一个梯度步之后就以浮点精度分离，最终导致 $W_Q^\top W_K$ 在相对 Frobenius 距离上相差 **56%**。等变优化器（GD、Muon）在相同设置下保持在浮点精度内。

这 56% 的差异无法通过对每个头做旋转来消除——因为不存在一个旋转能同时对齐两组 $W_Q^\top W_K$。这意味着 Adam 训练的注意力头有大量参数预算花在了规范噪声上，而非实际语义信息。

在高光谱数据集验证中，换用梯度下降（匹配训练损失）在最低采样密度下将保留误差降低了 **43-44%**，同时使用了更低的有效秩。

---

## Muon 的谱调度：一个微妙的权衡

Muon（Nesterov + Newton-Schulz 正交化）是规范等变的，理论上应继承低秩偏置。但它存在一个实践问题：

- 目标是**精确低秩**时：Muon 的等速率更新（所有奇异值以相同速率更新）表现完美
- 目标有**谱尾**（低秩 + 小噪声）时：等速率更新会过度压缩小奇异值，损失精度

**谱调度（spectral schedule）** 的解法：对不同奇异值分量使用不同学习率——大奇异值用大学习率，小奇异值用小学习率。这在两种制度之间取得平衡，同时保持规范等变性。这也解释了为何文献中对 Muon 的评价两极分化：评测设置不同，所落入的制度就不同。

---

## 什么时候用哪种优化器？

| 适用场景 | 推荐优化器 |
|---------|-----------|
| 矩阵感知 / 低秩恢复 | GD、Muon（加谱调度） |
| LoRA 微调（追求紧凑低秩适配） | Shampoo、Shared-scalar Adam |
| 追求快速收敛，不关心解的结构 | 标准 Adam（仍然有效） |
| 注意力层训练（关心头的一致性） | Muon 或 GD |

| 不适用场景 | 说明 |
|-----------|------|
| 真实秩高 / 数据本身高秩 | 低秩偏置未必是好事 |
| 计算预算紧张 | Muon/Shampoo 有额外矩阵运算开销 |
| 需要 Adam 级别的自适应学习率 | 等变优化器通常更难调参 |

---

## 实现中的坑

**坑 1：初始化幅度必须小**

```python
# 正确：小初始化让理论中的梯度流机制生效
U = nn.Parameter(torch.randn(n, r) * 0.01)

# 错误：大初始化下，梯度流的低秩轨迹理论不再适用
U = nn.Parameter(torch.randn(n, r))  # 默认 std ≈ 1
```

**坑 2：Shared-scalar Adam 需手动实现**

标准 PyTorch 没有内置版本。核心修改只有一行——把逐元素方差改为全局标量：

```python
# 标准 Adam（非等变）
v = beta2 * v + (1 - beta2) * grad ** 2          # 逐元素

# Shared-scalar Adam（等变）
v = beta2 * v + (1 - beta2) * (grad ** 2).mean()  # 全局标量，保留等变性
param.data -= lr * m_hat / (v_hat.sqrt() + 1e-8)
```

**坑 3：有效秩度量在训练早期不稳定**

奇异值熵在初始化附近接近最大值，容易造成误判。建议同时追踪前 $k$ 个奇异值的能量占比：

```python
def top_k_energy(W, k):
    sv = torch.linalg.svdvals(W)
    return (sv[:k] ** 2).sum() / (sv ** 2).sum()
```

---

## 我的观点

这篇论文最有价值的不是"Adam 不好"这个结论——Adam 仍然是大多数任务的默认选择。真正的贡献是提供了一个分析框架：**规范等变性是优化器的一个可测量代数属性，它决定了优化器能否继承梯度流的隐式偏置**。

几个值得持续关注的开放问题：

1. **LoRA 的影响**：LoRA 的低秩约束是显式的，但用 Adam 优化时，等变性缺失会不会影响适配器找到的解的质量？规模上还不清楚。

2. **非线性的干扰**：论文主要处理矩阵因式分解，实际 Transformer 中的 GeLU、LayerNorm 等非线性打破了许多理论假设。真实模型中规范对称性的影响会衰减多少，需要更多实证。

3. **Muon 的工程可行性**：Newton-Schulz 正交化有额外的计算开销，大模型预训练的实际效益还需要更多社区验证。

这篇论文做了一件少见的事：用代数结构解释了优化器行为差异，而不只是经验对比。这类理解框架，长期来看比任何特定的工程建议都更有价值。

---

**论文链接**：https://arxiv.org/abs/2608.05136v1