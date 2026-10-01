---
layout: post-wide
title: "BayesNDE：用贝叶斯生成模型和桥式采样做密度估计，绕开可逆网络"
date: 2026-10-01 12:05:25 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.39843v1
generated_by: Claude Code CLI
---

## 一句话总结

BayesNDE 想解决一个具体问题：给定一个任意的隐变量生成模型 $p(x, z) = p(x \mid z) p(z)$（不要求可逆、不要求可计算雅可比行列式），如何对每一个观测样本 $x$ 估计出一个相对靠谱、方差可控的密度值 $p(x)$？它的做法是——先为这个 $x$ 推断一个"专属"的后验近似（adaptive proposal），再用统计学里经典的**桥式采样（bridge sampling）**把这个 proposal 的采样结果和真实后验采样结果最优组合起来，得到一个低方差的密度估计量。

这篇文章不是对论文的逐字复现（官方代码还没细读到可以逐行对照的程度），而是把"用桥式采样做密度估计"这件事从零拆开讲清楚：它在解决重要性采样的什么痛点，代码怎么写，哪里容易炸，什么时候真的该用。

官方代码仓库：https://github.com/liuq-lab/BayesNDE

## 背景：为什么密度估计这么麻烦

如果你只是要"生成"样本，VAE、GAN、diffusion 都能做。但如果你要的是**具体的密度数值** $p(x)$（比如做异常检测、模型比较、似然比检验），情况就不一样了：

- **Normalizing Flow**：可以精确算密度，代价是网络必须可逆，且每层都要算雅可比行列式。这极大限制了网络结构的自由度，层数深了之后行列式计算也不便宜。
- **VAE 的 ELBO**：只是 $\log p(x)$ 的一个**下界**，不是真实密度。IWAE 把这个下界收紧了（$K \to \infty$ 时趋于真值），但本质仍然是对 $\log$ 取了期望内侧的近似，有系统性偏差。
- **朴素重要性采样（IS）**：如果你已经有 $p(x\mid z)$ 和 $p(z)$，理论上
$$
p(x) = \int p(x \mid z) p(z)\, dz \approx \frac{1}{K}\sum_{k=1}^K \frac{p(x, z_k)}{q(z_k)},\quad z_k \sim q
$$
这个估计量本身是**无偏**的（注意是对 $p(x)$ 无偏，不是对 $\log p(x)$）。问题出在方差：如果 proposal $q$ 和真实后验 $p(z\mid x)$ 对不上（尾部不匹配、或后验是多峰的而 $q$ 是单峰高斯），少数几个权重会主导整个求和，方差可能大到没法用。

BayesNDE 的核心想法就是冲着这个方差问题去的：先给每个样本训练/推断一个还不错的 proposal（类似 VAE 的 encoder），再用桥式采样把这个 proposal 的采样和一组"更真"的后验采样结合起来，比单纯依赖 proposal 本身更稳。

## 算法原理

### 直觉解释

把估计 $p(x)$ 想象成估计一块不规则石头的体积。朴素重要性采样相当于随便往空间里扔石子，数落在石头里的比例——如果石头很小、空间很大，大部分石子都是浪费的。用训练好的 proposal $q(z\mid x)$ 采样，相当于先大致猜到石头在哪儿，把石子集中扔在那附近，效率高很多，但如果猜错了方向（比如石头其实有两块，你只猜中了一块），估计会有系统性偏差。

桥式采样的做法是：**同时**准备两批信息——一批是"大致猜测"（proposal $q$ 的采样），一批是"真实情况"（从真实后验 $p(z\mid x)$ 采样，比如用 MCMC），然后用一个理论上方差最优的权重方式把两批信息揉在一起。这样即使 proposal 猜得不准，只要后验采样覆盖到了，总估计依然是对的、而且方差比只用后验采样（没有 proposal 辅助）要小。

### 数学推导

记 $\tilde p_1(z) = p(x, z)$（未归一化，它的归一化常数正是我们要求的 $r = p(x)$），$p_2(z) = q(z \mid x)$（已归一化的 proposal）。桥式采样的核心恒等式：对任意满足支撑条件的函数 $h(z)$，

$$
r = \frac{\mathbb{E}_{p_2}\left[\tilde p_1(Z) h(Z)\right]}{\mathbb{E}_{p_1}\left[p_2(Z) h(Z)\right]}
$$

其中 $p_1(z) = \tilde p_1(z)/r$ 就是真实后验 $p(z\mid x)$。用蒙特卡洛估计分子分母：

$$
\hat r = \frac{\frac{1}{n_2}\sum_{j=1}^{n_2} \tilde p_1(z_j) h(z_j)}{\frac{1}{n_1}\sum_{i=1}^{n_1} p_2(z_i) h(z_i)}, \quad z_j \sim p_2,\ z_i \sim p_1
$$

Meng & Wong (1996) 证明了能让这个估计量渐近方差最小的最优桥函数是

$$
h^*(z) \propto \frac{1}{n_1 p_1(z) + n_2 p_2(z)} = \frac{1}{n_1 \tilde p_1(z)/r + n_2 p_2(z)}
$$

这里 $h^*$ 本身依赖 $r$，所以只能不动点迭代：给定当前估计 $r_t$，算出 $h_t$，代入上式求出 $r_{t+1}$，反复迭代到收敛。实践中通常 10～30 次迭代就能稳定下来。

### 与其他方法的关系

桥式采样不是新东西，它是贝叶斯统计里估计模型证据（marginal likelihood）的经典工具（常用于贝叶斯模型选择），和退火重要性采样（AIS）、调和平均估计量属于同一家族，但方差表现通常显著优于后两者。BayesNDE 相当于把这套经典工具接到了"用神经网络生成模型做密度估计"这个任务上，proposal $q(z\mid x)$ 的角色和 VAE 的 encoder 完全一样，只是用途从"变分推断训练模型"换成了"辅助密度估计"。它不需要生成网络可逆，这是相对 normalizing flow 最直接的架构自由度优势；代价是推断阶段比 flow 贵得多——flow 一次前向传播就能拿到密度，BayesNDE 对每个样本都要跑 proposal 采样 + MCMC + 不动点迭代。

## 从零实现：渐进式代码讲解

下面的实现是我自己写的教学示例，不是论文官方代码，目的是让你亲手验证"桥式采样确实比朴素 IS 稳"这件事。为了能有一个可以数值积分得到的"真实密度"作为 ground truth，我用了一个人为构造的二维非单射映射（复数平方 $z \mapsto z^2$），它天然会产生双峰后验——这正好对应论文强调的"多峰数据"场景。

### 第一步：构造一个可验证真实密度的生成模型

```python
import torch
import numpy as np

torch.manual_seed(0)
LATENT_DIM, DATA_DIM, SIGMA_X = 2, 2, 0.2

def true_f(z):
    # 复数平方映射: (z1, z2) -> (z1^2 - z2^2, 2*z1*z2)
    # 非单射：z 和 -z 映射到同一个点，天然产生双峰后验
    z1, z2 = z[..., 0], z[..., 1]
    return torch.stack([z1**2 - z2**2, 2 * z1 * z2], dim=-1)

def log_p_x_given_z(x, z):
    mu = true_f(z)
    diff2 = ((x - mu) ** 2).sum(-1)
    return -0.5 * diff2 / SIGMA_X**2 - DATA_DIM * np.log(2 * np.pi * SIGMA_X**2) / 2

def log_p_z(z):
    return -0.5 * (z ** 2).sum(-1) - LATENT_DIM * np.log(2 * np.pi) / 2

def ground_truth_log_px(x, grid_size=240, bound=3.5):
    # 2 维隐变量才能暴力网格积分，仅用于验证，不是通用方法
    g = torch.linspace(-bound, bound, grid_size)
    zz1, zz2 = torch.meshgrid(g, g, indexing="ij")
    z_grid = torch.stack([zz1.flatten(), zz2.flatten()], dim=-1)
    log_joint = log_p_x_given_z(x, z_grid) + log_p_z(z_grid)
    cell_area = (2 * bound / grid_size) ** 2
    return (torch.logsumexp(log_joint, dim=0) + np.log(cell_area)).item()
```

`true_f` 和 `log_p_z` / `log_p_x_given_z` 之后会反复用到，不再重复定义。

### 第二步：朴素重要性采样——最简单但脆弱的基线

```python
def naive_is_log_px(x, proposal_sample_fn, log_q_fn, K=512):
    z = proposal_sample_fn(K)                       # [K, 2]
    log_w = log_p_x_given_z(x, z) + log_p_z(z) - log_q_fn(z)
    return torch.logsumexp(log_w, dim=0) - np.log(K)

# proposal = 先验本身
prior_sample = lambda K: torch.randn(K, LATENT_DIM)
prior_logq = log_p_z
```

用先验当 proposal 时，`naive_is_log_px` 完全不知道后验大概在哪，K 小的时候方差极大——这是我们要改进的基线。

### 第三步：训练一个自适应 proposal（amortized encoder）

```python
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DATA_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.mu_head = nn.Linear(64, LATENT_DIM)
        self.logvar_head = nn.Linear(64, LATENT_DIM)

    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h)

encoder = Encoder()
opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)

for step in range(3000):
    z0 = torch.randn(256, LATENT_DIM)
    x_data = true_f(z0) + SIGMA_X * torch.randn(256, DATA_DIM)   # 造训练数据
    mu, logvar = encoder(x_data)
    eps = torch.randn_like(mu)
    z_sample = mu + eps * (0.5 * logvar).exp()
    log_q = (-0.5 * eps**2 - 0.5 * logvar - 0.5 * np.log(2 * np.pi)).sum(-1)
    elbo = log_p_x_given_z(x_data, z_sample) + log_p_z(z_sample) - log_q
    loss = -elbo.mean()
    opt.zero_grad(); loss.backward(); opt.step()
```

训练目标就是标准的 ELBO（重建项 + 隐式 KL），只训练 encoder——这里把 `true_f` 当作已知、不训练的"真实解码器"，专注验证密度估计本身。

### 第四步：用 MCMC 拿到真实后验样本

proposal 再准，也可能漏掉后验的某个峰。桥式采样需要第二批**真正来自后验**的样本，这里用最简单的随机游走 Metropolis-Hastings：

```python
def mh_posterior_samples(x, n_samples=512, step=0.4, burn_in=200, init=None):
    z = init if init is not None else torch.randn(LATENT_DIM)
    log_target = lambda z: log_p_x_given_z(x, z) + log_p_z(z)
    cur_log_p = log_target(z)
    samples = []
    for t in range(n_samples + burn_in):
        z_prop = z + step * torch.randn(LATENT_DIM)
        prop_log_p = log_target(z_prop)
        if torch.log(torch.rand(1)) < prop_log_p - cur_log_p:
            z, cur_log_p = z_prop, prop_log_p
        if t >= burn_in:
            samples.append(z.clone())
    return torch.stack(samples)
```

**这是一个已知的坑**：由于 `true_f` 是双峰的，单条随机游走链极易困在其中一个峰，很难自己跳到另一个峰。稳妥的做法是从多个随机初始点各跑一条短链再合并样本，下面的实验里我是这么做的。

### 第五步：Optimal Bridge Sampling——把两批样本最优组合

```python
def bridge_sampling_log_px(x, encoder, n_iter=30, K_q=256, n_chains=4, n_per_chain=128):
    mu, logvar = encoder(x)
    std = (0.5 * logvar).exp()
    log_q = lambda z: (-0.5 * ((z - mu) / std) ** 2 - torch.log(std) - 0.5 * np.log(2 * np.pi)).sum(-1)

    z2 = mu + std * torch.randn(K_q, LATENT_DIM)                      # 来自 proposal q
    z1 = torch.cat([mh_posterior_samples(x, n_per_chain,               # 来自真实后验 p1
                     init=torch.randn(LATENT_DIM) * 2) for _ in range(n_chains)])
    n1, n2 = z1.shape[0], z2.shape[0]

    l1_tilde = log_p_x_given_z(x, z1) + log_p_z(z1)   # log p~1(z1_i)
    l2_tilde = log_p_x_given_z(x, z2) + log_p_z(z2)   # log p~1(z2_j)
    l1_q, l2_q = log_q(z1), log_q(z2)                 # log p2(z1_i), log p2(z2_j)

    log_r = torch.logsumexp(l2_tilde - l2_q, 0) - np.log(n2)   # 朴素IS估计作初值
    for _ in range(n_iter):
        log_h1 = -torch.logsumexp(torch.stack([np.log(n1) + l1_tilde - log_r, np.log(n2) + l1_q]), 0)
        log_h2 = -torch.logsumexp(torch.stack([np.log(n1) + l2_tilde - log_r, np.log(n2) + l2_q]), 0)
        num = torch.logsumexp(l2_tilde + log_h2, 0) - np.log(n2)
        den = torch.logsumexp(l1_q + log_h1, 0) - np.log(n1)
        log_r = num - den
    return log_r
```

全程在 log 空间用 `logsumexp` 完成，这是数值稳定的关键——直接在线性空间算会因为 $p(x,z)$ 的数值范围跨度太大而上溢/下溢。

### 调试诊断：有效样本数（ESS）

判断一次 IS/桥式采样靠不靠谱，光看点估计没用，要看权重分布有多"集中"：

```python
def effective_sample_size(log_w):
    w = torch.softmax(log_w, dim=0)           # 归一化权重
    return 1.0 / (w ** 2).sum()                # ESS, 理想情况接近样本数K

# 用法：log_w = log_p_x_given_z(x,z) + log_p_z(z) - log_q(z)
# 若 ESS / K < 5%~10%，说明 proposal 和目标分布严重失配，估计不可信
```

## 实验：多峰合成数据上的对比

用上面的组件，对若干测试点 $x$ 同时跑三种估计并对比网格积分得到的真值：

```python
test_zs = torch.randn(5, LATENT_DIM)
test_xs = true_f(test_zs) + SIGMA_X * torch.randn(5, DATA_DIM)

print(f"{'方法':<18}{'平均|误差|':>12}")
for name, estimate_fn in [
    ("朴素IS-先验",   lambda x: naive_is_log_px(x, prior_sample, prior_logq, K=512)),
    ("朴素IS-encoder", lambda x: naive_is_log_px(x,
         lambda K: encoder(x)[0] + (0.5*encoder(x)[1]).exp()*torch.randn(K, LATENT_DIM),
         lambda z: (-0.5*((z-encoder(x)[0])/(0.5*encoder(x)[1]).exp())**2
                    - 0.5*encoder(x)[1] - 0.5*np.log(2*np.pi)).sum(-1), K=256)),
    ("桥式采样",       lambda x: bridge_sampling_log_px(x, encoder)),
]:
    errs = [abs(estimate_fn(x).item() - ground_truth_log_px(x)) for x in test_xs]
    print(f"{name:<18}{np.mean(errs):>12.3f}")
```

在我本地跑这个脚本的多次重复实验中，一个稳定出现的模式是：**朴素 IS-先验**的误差方差极大（有时候某次采样几乎命中后验、误差很小，换个随机种子又误差一两个数量级）；**朴素 IS-encoder** 平均误差明显更低、更稳定，但因为 encoder 是单峰高斯，对双峰后验里"没猜中"的那个峰经常系统性低估；**桥式采样**在样本数相近的情况下通常误差最小、波动也最小——因为它同时利用了 MCMC 采到的两个峰的信息，不完全依赖 encoder 猜得准不准。我没有做到严格复现论文量级的实验（没有用论文的真实数据集和官方超参数），这里报告的是在这个小玩具任务上观察到的定性趋势，不是定量结论。

## 性能分析与调试指南

**计算成本对比**：normalizing flow 算一次密度是一次前向传播，$O(1)$（per layer 的行列式开销摊在训练里）。BayesNDE 这类方法对每一个测试样本都要：encoder 前向 + $K$ 次 proposal 采样、MCMC 跑若干条链（每条链又要多步）、再做十几次不动点迭代——单样本推断成本比 flow 高出一到两个数量级。如果你需要大批量、实时的密度打分（比如在线异常检测流水线逐条打分），这个成本差异是要认真掂量的；如果是离线批量分析（比如科研里对一批数据算似然做模型比较），这笔开销通常可以接受。

**常见问题**：

1. **桥式采样结果忽大忽小、不收敛**：几乎总是因为 MCMC 链没有混合好——单链困在双峰的一个峰里。解决办法是多条链、不同初始化，再检查链内样本的自相关/接受率（理想接受率大致在 20%～50% 之间，步长 `step` 需要按这个来调）。
2. **ESS 很低（比如远小于样本数的 10%)**：说明 proposal 和真实后验差距太大，这时候再怎么加大 $K$ 也救不回来，先去改善 encoder 或者增加后验采样的多样性，而不是盲目加样本数。
3. **log 密度出现 `nan`/`inf`**：基本是在线性空间算了指数导致上溢/下溢，检查是不是哪一步漏了用 `logsumexp`，或者方差项 `SIGMA_X` 设得太小导致 $\log p(x\mid z)$ 数值爆炸。
4. **不动点迭代发散**：通常是初值 $\log r_0$ 离真值太远，用朴素 IS 的结果作为初值（如上面代码所做）通常足够稳健；如果仍不收敛，减小迭代步幅或限制每步变化幅度。

**可调超参数一览**：

| 参数 | 推荐范围 | 敏感度 | 备注 |
|------|---------|-------|------|
| proposal 采样数 $K_q$ | 128–512 | 中 | 太小时 encoder 的误差主导 |
| MCMC 链数 $n_{chains}$ | 3–8 | 高（多峰场景） | 单链在多峰后验下严重不可靠 |
| MCMC 步长 | 需按接受率调 | 高 | 目标接受率约 20%–50% |
| 不动点迭代次数 | 15–30 | 低 | 通常很快收敛，次数再多无明显收益 |

## 什么时候用，什么时候别用

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要精确似然值，架构上不想被"可逆"约束死（异常检测、模型比较） | 需要对海量样本做实时/在线密度打分 |
| 数据/后验天然多峰，flow 式单调变换不好建模 | 隐变量维度很高（网格积分做验证的方法本身就用不了，真实场景也要靠更贵的 MCMC/HMC 兜底后验采样） |
| 离线分析，能接受"一次推断需要多次采样 + 迭代"的开销 | 需要对密度估计本身求梯度并反传（采样+迭代的估计量，端到端可微性没有 flow 自然） |

## 我的观点

桥式采样不是一个新算法，它是贝叶斯统计里几十年前就有的估计模型证据的经典工具；BayesNDE 真正的贡献点在于把它和 amortized inference（像 VAE 的 encoder）接到一起，让"自适应 proposal + 后验采样"这套组合可以批量作用在神经网络生成模型上，从而绕开了 flow 必须可逆这个架构约束。这个思路是合理的，代价也很诚实地摆在那里：每个样本的密度估计都要付出比 flow 贵得多的推断成本，而且多峰场景下 MCMC 混合不好依然是个实打实的工程难题，不会因为换了个"桥式采样"的外壳就自动消失。

是否比最新的基于分数的（score-based / diffusion）密度估计方法更有优势，我没有做过正面对比，这里不下结论。如果你的场景本来就需要精确似然、能接受离线批处理的延迟、且数据有明显的多峰/非线性结构（比如论文提到的异常检测场景），这条路值得一试；如果你只是想要一个能打分的生成模型、对密度精度要求不那么苛刻，VAE+IWAE 或者训练一个 flow 可能更省心。