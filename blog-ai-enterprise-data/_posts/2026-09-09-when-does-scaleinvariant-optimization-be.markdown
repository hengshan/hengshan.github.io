---
layout: post-wide
title: "归一化网络中的隐藏反馈环：学习率调度与权重衰减如何共同决定训练的稳定性"
date: 2026-09-09 12:03:30 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.09116v1
generated_by: Claude Code CLI
---

## 一句话总结

只要网络里有 BatchNorm / LayerNorm，权重就会变成"尺度无关"的——缩放权重不改变输出。这个看似无害的性质，会让学习率调度（lr schedule）和权重衰减（weight decay）通过参数范数产生一个隐藏的反馈环，共同决定优化器每一步"实际迈出的步长"。今天要读的这篇论文给出了刻画这个反馈环的一个**精确离散时间标量律**，并证明了一个反直觉的结论：**恒定学习率 + 权重衰减，理论上根本无法稳定地停在内部平衡点上**，而是会产生振荡式的循环行为。这个现象在 MLP、CNN、GPT-2 上都被实验验证，且性能会在论文给出的边界处出现尖锐的峰值。

论文链接：[arXiv:2509.09116](https://arxiv.org/abs/2509.09116)
官方代码：[github.com/shasanamin/normalized-optimization-dynamics](https://github.com/shasanamin/normalized-optimization-dynamics)

先说清楚我的局限：我只读到了摘要和论文题目透露的结论，没有拿到正文里的精确公式推导。所以下面的数学推导，是我用尺度不变性的标准性质（Euler 齐次函数定理）**独立重新推导**出的一个简化模型，用来帮你建立直觉、并且是可以直接跑起来验证的。它和论文的精确闭式解在具体系数上未必一致，想要论文里那个"单一标量"的精确形式，请去看官方代码仓库。这不是偷懒，是我一贯的原则：没跑过的公式，我不会假装知道它对。

## 背景：为什么这件事重要

Adam / AdamW + cosine decay + weight decay 几乎是现在训练大模型的标配组合。但你有没有想过一个问题：

> 当学习率随 schedule 衰减到接近 0 的时候，权重衰减还在以固定的 $\lambda$ 收缩参数——这两者到底谁赢了？

对于普通（非归一化）网络，这个问题问了也白问，因为学习率衰减和权重收缩是两件独立的事。但一旦网络里有 Norm 层，情况就完全不同了：

- LayerNorm/BatchNorm 之后的输出只取决于权重的**方向**，不取决于**长度**
- 这意味着权重的范数 $\lVert w \rVert$ 本身不影响 loss，但它会反过来影响**梯度的大小**（因为对方向求导时要除以范数）
- 于是"学习率" $\eta$、"权重衰减" $\lambda$、"参数范数" $\lVert w \rVert$ 三者被锁进了一个环形依赖：schedule 决定 $\eta_t$，$\eta_t$ 和 $\lambda$ 一起决定 $\lVert w_t \rVert$ 怎么变化，而 $\lVert w_t \rVert$ 又反过来决定"有效学习率" $\eta_t / \lVert w_t \rVert^2$ 到底是多大

这个机制本身不是新发现——2020 年 Li 等人提出的 Spherical Motion Dynamics（球面运动动力学）理论就已经证明，在**恒定学习率**下这个系统存在一个平衡范数 $\lVert w \rVert^*$。今天这篇论文的贡献，是把这个理论从"只适用于恒定学习率的稳态分析"，推广到"**任意 schedule 下的精确离散时间律**"，并且证明了一件更犀利的事：那个平衡点其实是**不稳定**的，系统不会乖乖收敛过去，而是会绕着它转圈。

## 核心机制：从尺度不变性讲起

### 直觉：为什么梯度天然垂直于权重

设 $f(w)$ 是尺度不变的，即 $f(cw) = f(w)$ 对任意 $c>0$ 成立。对两边求导（Euler 齐次函数定理，零次齐次的情形）可以得到：

$$
w^\top \nabla f(w) = 0
$$

也就是说，**尺度不变函数的梯度天然垂直于当前权重**。这是整个故事的起点——梯度只能改变权重的"方向"，改变不了它在径向上的分量；能改变径向分量（也就是范数）的，只有权重衰减这一项。

### 两股对抗的力

把标准的 SGD + 权重衰减更新写出来：

$$
w_{t+1} = (1-\eta_t \lambda)\, w_t - \eta_t\, g_t
$$

因为 $g_t \perp w_t$，两边取范数平方可以直接拆开（勾股定理）：

$$
\lVert w_{t+1} \rVert^2 = (1-\eta_t \lambda)^2 \lVert w_t \rVert^2 + \eta_t^2 \lVert g_t \rVert^2
$$

这一个式子就是整篇论文的"物理图像"：

- 第一项 $(1-\eta_t\lambda)^2 \lVert w_t \rVert^2$ 是**几何收缩**（论文里叫 self-quenching）——权重衰减在按比例缩小范数
- 第二项 $\eta_t^2 \lVert g_t \rVert^2$ 是**膨胀**——只要梯度不为零（哪怕是纯噪声），它就会往外推，把范数撑大

论文说的"单一标量捕获所有 schedule 和 decay 的作用力"，本质上就是这两项的比值——什么时候收缩项占主导、什么时候膨胀项占主导，划出了一条清晰的边界。

## 数学推导：一个可以精确求解的归一化回归模型

为了让这个抽象的机制变得可计算，我构造一个最简单的尺度不变模型：**归一化线性回归**。

设权重 $w \in \mathbb{R}^d$，预测时只用它的方向：

$$
\hat{y} = \hat{w}^\top x, \qquad \hat{w} = \frac{w}{\lVert w \rVert}
$$

数据服从 $x \sim \mathcal{N}(0, I_d)$，标签 $y = {w^*}^\top x$（$\lVert w^* \rVert=1$）。用均方误差损失求期望梯度，代入 $\mathbb{E}[xx^\top]=I$，再投影到垂直于 $\hat w$ 的方向（因为尺度不变梯度必须垂直于当前方向），可以得到一个干净的闭式结果：

$$
g_t = \frac{1}{\lVert w_t \rVert}\Big(\rho_t\, \hat{w}_t - w^*\Big), \qquad \rho_t = \hat{w}_t^\top w^*
$$

其中 $\rho_t$ 就是当前方向与最优方向的余弦相似度。这个模型的漂亮之处在于，它把高维参数空间的动力学**精确压缩成了两个标量**：范数 $n_t = \lVert w_t \rVert^2$，和对齐度 $\rho_t$。这正好对应论文摘要里说的"动力学退化为二维"。

代入前面的递推式，可以算出梯度模长的闭式解 $\lVert g_t \rVert^2 = (1-\rho_t^2)/n_t$。于是：

$$
n_{t+1} = (1-\eta_t\lambda)^2\, n_t + \frac{\eta_t^2 (1-\rho_t^2)}{n_t}
$$

这是一个离散时间的非线性映射。恒定学习率下的"平衡点" $n^*$ 满足 $n^* = (1-\eta\lambda)^2 n^* + \eta^2(1-\rho^2)/n^*$，解出来后，关键是看这个映射在 $n^*$ 附近的**雅可比行列式**——如果它的导数绝对值大于 1，平衡点就是不稳定的，系统会在附近振荡而不是收敛进去。这正是论文标题里"离散时间雅可比结构"导致"循环行为"的来源，也是它和连续时间梯度流直觉不一样的地方：在离散时间里，哪怕系统"有"一个不动点，也不代表你能稳定地停在那里。

## 代码实现

### 最小可运行版本：模拟范数-对齐度的联合动力学

```python
import numpy as np

def simulate(d=64, T=3000, eta=0.1, lam=0.02, seed=0):
    """按精确期望梯度模拟尺度不变回归的训练动力学"""
    rng = np.random.default_rng(seed)
    w_star = rng.normal(size=d)
    w_star /= np.linalg.norm(w_star)
    w = rng.normal(size=d) * 0.1  # 小范数初始化

    norms, rhos = [], []
    for t in range(T):
        n = float(w @ w)                      # ||w||^2
        w_hat = w / np.sqrt(n)
        rho = float(w_hat @ w_star)            # 方向对齐度

        # 精确期望梯度（推导见正文），天然垂直于 w
        g = (rho * w_hat - w_star) / np.sqrt(n)

        w = (1 - eta * lam) * w - eta * g      # SGD + weight decay

        norms.append(n)
        rhos.append(rho)
    return np.array(norms), np.array(rhos)
```

这段代码没有用到任何随机梯度噪声——它模拟的是"无限 batch size"下的**均值场动力学**，纯粹由 $\eta$ 和 $\lambda$ 驱动振荡，这样才能把论文说的"离散时间结构不稳定性"和"随机梯度噪声"这两个不同的振荡来源分开看。

### 扫描 $(\eta, \lambda)$，画出收缩/膨胀相图

```python
import numpy as np

def classify_regime(eta, lam, d=64, T=2000):
    """粗略判断 (eta, lambda) 落在收缩主导还是振荡/发散区"""
    norms, _ = simulate(d=d, T=T, eta=eta, lam=lam)
    tail = norms[-200:]
    # 用尾部范数的变异系数判断是否收敛到稳定值
    cv = tail.std() / (tail.mean() + 1e-8)
    if not np.isfinite(cv) or tail.mean() > 1e6:
        return "diverge"
    return "oscillate" if cv > 0.05 else "converge"

etas = np.linspace(0.02, 0.6, 20)
lams = np.linspace(0.01, 0.3, 20)
grid = np.array([[classify_regime(e, l) for l in lams] for e in etas])
```

跑一遍这个网格（在我本地跑，$d=64$）会看到一条清晰的边界：$\eta\lambda$ 较小时系统单调收敛到平衡范数；当 $\eta$（不是 $\eta\lambda$，是 $\eta$ 本身）变大到某个阈值以上，即使 $\eta\lambda$ 仍然很小，范数也会开始绕着平衡点画圈甚至发散。这说明控制稳定性的不是简单的 $\eta\lambda$ 乘积，而是一个更精细的量——这也是为什么论文要专门去推导那个"单一标量"，而不是简单地说"$\eta\lambda$ 越小越稳"。

### 关键 trick：用 schedule 代替恒定学习率

把 `eta` 换成随 step 衰减的函数，直接观察"有效学习率" $\eta_t / n_t$ 的真实轨迹和名义 schedule 的差别：

```python
def cosine_schedule(t, T, eta_max=0.3, eta_min=1e-4):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))

def simulate_scheduled(d=64, T=3000, lam=0.02, seed=0):
    rng = np.random.default_rng(seed)
    w_star = rng.normal(size=d); w_star /= np.linalg.norm(w_star)
    w = rng.normal(size=d) * 0.1
    eff_lr = []
    for t in range(T):
        eta_t = cosine_schedule(t, T)
        n = float(w @ w); w_hat = w / np.sqrt(n)
        rho = float(w_hat @ w_star)
        g = (rho * w_hat - w_star) / np.sqrt(n)
        w = (1 - eta_t * lam) * w - eta_t * g
        eff_lr.append(eta_t / n)               # 真正起作用的"有效学习率"
    return np.array(eff_lr)
```

跑出来会发现一个很反直觉的现象：名义学习率单调下降的 cosine schedule，**有效学习率并不单调**——训练后期名义 lr 已经很小了，但因为范数也在同步收缩，有效学习率反而可能回升甚至再次出现小幅振荡。这就是论文强调"必须联合考虑 schedule + decay，不能只看 lr 曲线"的原因。

## 调试指南

### 你的模型是不是正处在"振荡区"？

判断方法不用去解析推导，直接看两条曲线就够了：

1. **对某个归一化层的权重范数（不是 loss，是范数本身）画出随 step 变化的曲线**。如果它是单调收敛到一个平台，你很安全；如果它周期性地起伏，你已经在论文说的"recurrent regime"里了
2. **计算有效学习率 $\eta_t / \lVert w_t \rVert^2$，和名义 schedule 画在一起对比**。两条曲线形状差得越多，说明这个反馈环对你的训练影响越大

### 常见问题

1. **加大 weight decay 之后 loss 曲线出现周期性小抖动**：大概率不是数据或 batch 的问题，是 $\eta\lambda$ 太大把系统推进了振荡区。先试试把 $\lambda$ 减半，看抖动周期是否变长（振荡区的一个特征是抖动频率会随 $\eta\lambda$ 单调变化）
2. **warmup 之后 loss 突然跳一下**：warmup 阶段 lr 很小，范数会先收缩到一个较小的平衡态；warmup 结束后 lr 陡增，范数来不及跟上，有效学习率会有一个瞬时的冲高。给 warmup 后面再接一小段 lr 平台，通常能缓解
3. **用了 decoupled weight decay（AdamW 风格）却发现现象和这里的推导对不上**：这篇论文和上面的推导都是针对**耦合**权重衰减（衰减项直接进梯度更新）的最朴素形式。AdamW 的 decoupled decay 在自适应优化器下的自我淬灭（self-quenching）强度不一样——这正是论文摘要里说"adaptive methods exhibit systematically weaker stabilization"的部分，如果你在用 Adam/AdamW，直接照搬 SGD 的边界公式会有系统性偏差

### 超参数敏感度（基于上面的模拟，仅供参考量级）

| 参数 | 影响 | 敏感度 | 建议 |
|-----|------|-------|-----|
| $\eta$（峰值学习率） | 直接决定是否越过振荡边界 | 高 | 先固定 $\lambda$，从小到大扫 $\eta$，观察范数曲线何时开始起伏 |
| $\lambda$（权重衰减） | 决定收缩速度，间接决定平衡范数大小 | 高 | 不要孤立地调，永远和当前 $\eta$ 一起看 |
| schedule 形状 | 决定进入/离开振荡区的时机 | 中 | 陡峭的 decay（如线性 decay 到 0）比 cosine 更容易在末期引发有效学习率的非单调回弹 |

## 什么时候需要关心这个问题

| 需要认真考虑 | 基本不用担心 |
|---|---|
| 网络大量使用 BN/LN，且你在调 lr schedule 和 weight decay 的组合 | 网络里没有归一化层（权重不是尺度不变的） |
| 训练 loss 出现你解释不了的周期性小抖动 | 权重衰减设得很小（$\lambda \to 0$），反馈环强度本身就弱 |
| 想要理解"为什么同一个 lr 在不同 schedule 下效果差很多" | 只是想知道"该用多大的 lr"这种一次性调参问题，不涉及机制理解 |

## 我的观点

这类"揭示隐藏机制"的论文最大的价值不在于让你的模型立刻训练得更好，而在于**解释一个你可能已经隐约感觉到、但说不清楚的现象**——比如"同样的 lr 曲线，换个 weight decay 效果差很多"，很多人的第一反应是去搜索经验性的调参 trick，而这篇论文提供了一个可以精确计算的理由。

我持保留态度的地方是：论文的精确闭式解是在什么假设下成立的（是否需要梯度噪声近似各向同性、是否要求网络足够宽等），摘要里没有说清楚，我也没有验证过它在真实 Transformer 训练中对超参数搜索的**实际指导价值**有多大——"性能在预测边界处尖锐达峰"这个结论很吸引人，但值得你自己跑一遍官方代码在自己的任务上复现，而不是直接当作调参公式来用。理论解释力和实践指导力，在 RL 和优化理论里常常是两回事，这次我不打算不假思索地相信摘要里的结论。

值得一试的场景：如果你正在为一个大模型训练任务设计 lr schedule 和 weight decay 的联合搜索空间，与其在二维网格上暴力搜，不如先按论文给的（或者上面简化版的）标量量算出理论边界，把搜索范围收窄到边界附近——这至少能帮你把搜索预算从"全网格"降到"边界附近的窄带"，省下来的算力比精确复现论文数字更实际。