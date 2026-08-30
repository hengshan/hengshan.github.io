---
layout: post-wide
title: "用条件 Wasserstein GAN 摊销保险复合损失模型的贝叶斯推断"
date: 2026-08-30 12:03:27 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.27229v1
generated_by: Claude Code CLI
---

## 一句话总结

训练一个以充分统计量和先验参数为条件的 WGAN，让同一个生成器能在毫秒内为数千个保单场景提供后验样本——替代每次都要跑几小时的 MCMC。

---

## 背景：重复推断的代价

在精算建模中，**复合损失模型**是标准工具：

$$
S = \sum_{i=1}^{N} X_i
$$

其中 $N \sim \text{Poisson}(\lambda)$ 是索赔次数，$X_i \sim \text{Pareto}(\alpha)$ 是单次损失金额。

贝叶斯推断的挑战不在于单次推断，而在于**规模**。一家保险公司有数万个保单，每个对应不同的历史数据和先验假设。Poisson-Gamma 共轭情形下后验有解析解；但换成混合先验族（Gamma、逆高斯、对数正态的混合），或者要同时推断 Pareto 形状参数，就只能上 MCMC。每次 MCMC 耗时分钟级，数万次叠加让计算成本不可接受。

### 摊销推断的核心 Insight

**摊销推断（Amortized Inference）** 的思路是：与其每次从头推断，不如训练一个神经网络，学会从数据直接映射到后验分布。

这篇论文的贡献在于，用条件 WGAN 来近似后验：

$$
G(z;\, \mathbf{c}) \approx \text{样本} \sim p(\theta \mid \mathbf{c})
$$

条件向量 $\mathbf{c}$ 包含：
- 充分统计量（索赔次数 $n$，损失对数之和 $\sum \log x_i$）
- 先验参数（均值、变异系数 CV）
- 先验族混合权重

训练完成后，推断速度从分钟降到毫秒。

---

## 算法：为什么用 WGAN 而不是 VAE

VAE 的近似后验 $q_\phi(\theta \mid \mathbf{c})$ 通常限定在高斯族，对重尾后验（Pareto 参数的后验往往偏斜）拟合不好。WGAN 不对分布族做假设，通过 Wasserstein 距离驱动生成器学习任意形状的后验。

WGAN-GP 的目标：

$$
\min_G \max_{D:\,\|D\|_L \leq 1} \; \mathbb{E}_{\theta \sim p(\theta \mid \mathbf{c})}[D(\theta, \mathbf{c})] - \mathbb{E}_{z}[D(G(z, \mathbf{c}), \mathbf{c})]
$$

训练数据来自**模拟**：先从先验采样真实参数，再从模型采样观测，这样我们就有了无限量的 $(\mathbf{c}, \theta_{\text{true}})$ 配对。这是摊销推断的关键优势——不需要真实数据的标注。

---

## 实现

### 数据生成

```python
import numpy as np

def simulate_dataset(n_policies=50000):
    """从先验-似然联合分布模拟训练数据"""
    records = []
    for _ in range(n_policies):
        # 随机化先验超参数（让模型学会跨先验泛化）
        a_lam = np.random.uniform(1, 8)
        b_lam = np.random.uniform(0.5, 4)
        a_alpha = np.random.uniform(2, 6)
        b_alpha = np.random.uniform(1, 3)

        lam_true = np.random.gamma(a_lam, 1.0 / b_lam)
        alpha_true = np.random.gamma(a_alpha, 1.0 / b_alpha)

        n_claims = np.random.poisson(lam_true)
        # Pareto(alpha): F(x) = 1 - x^{-alpha}, x >= 1
        losses = np.random.pareto(alpha_true, n_claims) + 1 if n_claims > 0 else np.array([])

        records.append({
            'lam': lam_true, 'alpha': alpha_true,
            'n': n_claims, 'losses': losses,
            'prior': (a_lam, b_lam, a_alpha, b_alpha)
        })
    return records
```

### 条件向量构造

```python
def build_condition_vector(record):
    """
    将观测数据和先验参数压缩为固定长度向量
    Poisson 充分统计量: n
    Pareto 充分统计量:  n, sum(log x_i)
    """
    n = record['n']
    losses = record['losses']
    a_lam, b_lam, a_alpha, b_alpha = record['prior']

    sum_log_x = np.sum(np.log(losses)) if n > 0 else 0.0

    prior_mean_lam = a_lam / b_lam
    prior_cv_lam   = 1.0 / np.sqrt(a_lam)
    prior_mean_alpha = a_alpha / b_alpha
    prior_cv_alpha   = 1.0 / np.sqrt(a_alpha)

    return np.array([
        float(n), sum_log_x,
        prior_mean_lam, prior_cv_lam,
        prior_mean_alpha, prior_cv_alpha
    ], dtype=np.float32)
```

### 网络结构

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    """从噪声和条件向量生成后验样本 (lambda, alpha)"""
    def __init__(self, z_dim=32, cond_dim=6, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, out_dim),
            nn.Softplus()  # 保证输出为正
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=-1))

class Critic(nn.Module):
    """WGAN critic：输出实数，不加 sigmoid"""
    def __init__(self, sample_dim=2, cond_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sample_dim + cond_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, theta, c):
        return self.net(torch.cat([theta, c], dim=-1))
```

### 梯度惩罚与训练步骤

```python
def gradient_penalty(critic, real, fake, cond, lam=10.0):
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(interp, cond)
    grads = torch.autograd.grad(score, interp,
                                grad_outputs=torch.ones_like(score),
                                create_graph=True)[0]
    return lam * ((grads.norm(2, dim=1) - 1) ** 2).mean()

def train_step(G, C, theta_real, cond, opt_G, opt_C, z_dim=32, n_critic=5):
    for _ in range(n_critic):
        z = torch.randn(len(cond), z_dim, device=cond.device)
        theta_fake = G(z, cond).detach()
        gp = gradient_penalty(C, theta_real, theta_fake, cond)
        loss_C = (C(theta_fake, cond) - C(theta_real, cond)).mean() + gp
        opt_C.zero_grad(); loss_C.backward(); opt_C.step()

    z = torch.randn(len(cond), z_dim, device=cond.device)
    loss_G = -C(G(z, cond), cond).mean()
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    return loss_G.item(), loss_C.item()
```

### 推断接口

```python
@torch.no_grad()
def posterior_samples(generator, record, n_samples=2000, z_dim=32):
    """给定一个保单的数据，生成后验样本"""
    cond = torch.tensor(build_condition_vector(record)).unsqueeze(0).repeat(n_samples, 1)
    z = torch.randn(n_samples, z_dim)
    samples = generator(z, cond).numpy()
    # samples[:, 0]: lambda 后验样本
    # samples[:, 1]: alpha 后验样本
    return samples
```

---

## 实验：用 SBC 检验校准性

**SBC（Simulation-Based Calibration）** 是验证摊销推断最可靠的工具。思路：从先验采样真实参数 $\theta^*$，再用模拟数据生成后验样本，计算 $\theta^*$ 在这些样本中的**秩**。如果校准良好，秩应该服从 $\text{Uniform}[0, 1]$。

```python
import matplotlib.pyplot as plt

def run_sbc(generator, test_records, n_samples=500):
    ranks_lam, ranks_alpha = [], []
    for rec in test_records:
        samples = posterior_samples(generator, rec, n_samples=n_samples)
        ranks_lam.append((samples[:, 0] < rec['lam']).mean())
        ranks_alpha.append((samples[:, 1] < rec['alpha']).mean())
    return np.array(ranks_lam), np.array(ranks_alpha)

def plot_sbc(ranks, param_name):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(ranks, bins=20, density=True, alpha=0.7, color='steelblue')
    ax.axhline(1.0, color='red', linestyle='--', label='理想均匀分布')
    ax.set(title=f'SBC — {param_name}', xlabel='分位数秩', ylabel='密度')
    ax.legend(); plt.tight_layout(); plt.show()
```

**解读 SBC 图**：

| 形状 | 含义 | 修复方向 |
|------|------|---------|
| 接近水平线 | 校准良好 ✓ | — |
| U 形（中间低） | 后验过宽，不确定性虚高 | 增加训练数据，减小 z_dim |
| 拱形（中间高） | 后验过窄，生成器过度自信 | 检查梯度惩罚系数 |
| 单侧偏斜 | 系统性偏差 | 检查充分统计量是否完整 |

---

## 调试指南

RL 很难调，摊销推断也一样——只是崩溃的方式不同。

### 常见问题

**1. mode collapse：生成器输出崩塌到一个点**

```python
# 训练时监控方差
samples = posterior_samples(G, rec, n_samples=500)
print(f"lambda std: {samples[:,0].std():.4f}")  # 接近 0 说明崩塌
# 修复：降低 generator 学习率，增加 n_critic
```

**2. Critic 分数单调下降不收敛**

```python
# 修复：critic 学习率设为 generator 的 2-4 倍
opt_C = torch.optim.Adam(C.parameters(), lr=2e-4, betas=(0.0, 0.9))
opt_G = torch.optim.Adam(G.parameters(), lr=5e-5, betas=(0.0, 0.9))
# betas=(0, 0.9) 是 WGAN-GP 的标准配置，不要用默认的 (0.9, 0.999)
```

**3. SBC 严重偏斜，先检查这里**

```python
# 条件向量量纲差异悬殊时必须标准化
# sum_log_x 随 n 线性增长，不标准化会让网络无法泛化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cond_train = scaler.fit_transform(cond_train)
# 推断时用同一个 scaler.transform()
```

### 超参数敏感度

| 参数 | 推荐值 | 敏感度 | 备注 |
|------|--------|--------|------|
| 学习率 | 5e-5 / 2e-4 | 高 | 先调这个 |
| `n_critic` | 5 | 中 | 小于 3 容易不稳定 |
| GP 系数 `lam` | 10 | 低 | 5–20 都可以 |
| `z_dim` | 32–64 | 低 | 太小限制表达力 |
| 批大小 | 256–512 | 中 | 太小则 GP 估计噪声大 |
| 训练样本数 | ≥ 50000 | 高 | 摊销推断最贵的地方 |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要对同一类问题反复推断（批量保单） | 一次性推断，MCMC 跑一次就够了 |
| 先验族固定，只是参数变化 | 模型结构本身经常变化 |
| 后验计算是性能瓶颈（实时定价） | 需要精确后验（近似误差不可接受） |
| 模拟成本低，真实标注贵 | 先验分布分散，训练分布难以覆盖 |

---

## 我的观点

这篇论文的核心贡献不是 WGAN 本身，而是**把摊销推断引入精算学**，并诚实地用 SBC 来评估近似质量——这一点值得学习，很多做摊销推断的论文只报告均值和方差对比，而不跑校准测试。

几点诚实评价：

**训练数据要求高**。5 万条模拟只是起点，先验参数范围宽时可能需要更多。但模拟便宜，这是摊销推断的本质优势。

**泛化边界要谨慎**。生成器只在训练时覆盖的先验参数范围内可靠。新保单的先验参数一旦超出训练分布，结果会悄无声息地变差，这是摊销推断的通病，不是这篇论文特有的问题。

**SBC 不能省**。没有校准测试就上生产，等于盲开车。SBC 是这类方法的生命线。

如果你做的是单次推断，MCMC 依然是更可靠的选择。如果你面对的是需要反复推断的批量场景，Neural Posterior Estimation（NPE）系列方法值得一并了解，它们在校准性和样本效率方面有更系统的理论支撑。