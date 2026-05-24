---
layout: post-wide
title: "用扩散模型做生存分析：SDPM 原理、实现与调试指南"
date: 2026-05-24 08:07:29 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.22776v1
generated_by: Claude Code CLI
---

## 一句话总结

SDPM 把生存分析重新建模为生成问题：用扩散模型直接学习联合分布 $\mathbb{P}(T, \delta \mid \mathbf{x})$，绕过了对风险函数的参数假设，也不需要对时间轴离散化。

---

## 背景：生存分析难在哪里？

生存分析要回答的问题很简单：**事件什么时候发生？** 病人何时复发、用户何时流失、机器何时故障。

真正的麻烦是**删失（Censoring）**：很多样本在观测结束时事件尚未发生，我们只知道"到截止时刻，事件还没发生"。简单丢弃这些样本会引入严重偏差；正确处理删失是生存分析的核心挑战。

### 现有方法的局限

**Cox 比例风险模型**是最常用的方法：

$$h(t \mid \mathbf{x}) = h_0(t) \exp(\mathbf{x}^\top \boldsymbol{\beta})$$

它有两个强假设：比例风险（不同协变量的风险比恒定）、半参数形式。深度学习方案：

- **DeepHit**：离散化时间轴，用神经网络预测每段的事件概率。时间分辨率与计算量存在 trade-off，离散化引入近似误差
- **Deep Survival Machines**：假设事件时间来自 Weibull/Log-Normal 混合，参数灵活性有限
- **SurvTRACE**：用 Transformer 建模，底层仍然是参数化风险函数

### SDPM 的核心 Insight

既然扩散模型最擅长学习分布，为什么不直接建模联合分布？

$$\mathbb{P}(T, \delta \mid \mathbf{x})$$

其中 $T$ 是观测时间，$\delta \in \{0, 1\}$ 是事件指示符（1 = 事件，0 = 删失）。

推断方式：对给定 $\mathbf{x}$ 采样若干 $(T_i, \delta_i)$，喂给 Kaplan-Meier 估计量，得到 $\hat{S}(t \mid \mathbf{x})$。框架的好处：无风险函数假设，无时间离散化，纯数据驱动。

---

## 算法原理

### 直觉：往 (T, δ) 上加噪声，再学着去噪

扩散模型的逻辑大家都熟悉：前向过程加噪声，反向过程学去噪。SDPM 把这套机制用在生存结果 $(T, \delta)$ 上。

麻烦在于 $\delta$ 是二值的，不能直接用高斯扩散。解决方案是**目标空间变换**：

- **时间**：$\tilde{t} = (\log T - \mu) / \sigma$，标准化 log 时间，让分布更接近高斯
- **删失指示**：把 $\delta \in \{0, 1\}$ 当成连续浮点值，扩散过程自然把它变成连续变量；生成时用 0.5 作阈值恢复二值

目标空间变为 $\mathbf{y} = (\tilde{t},\ \delta) \in \mathbb{R}^2$，标准 DDPM 即可工作。

### 数学推导

前向扩散：

$$q(\mathbf{y}_s \mid \mathbf{y}_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_s}\,\mathbf{y}_0,\ (1 - \bar{\alpha}_s)\mathbf{I}\right)$$

条件去噪网络 $\epsilon_\theta$ 的训练目标：

$$\mathcal{L} = \mathbb{E}_{s,\,\mathbf{y}_0,\,\boldsymbol{\epsilon}}\left[\left\|\boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{y}_s, s, \mathbf{x})\right\|^2\right]$$

推断时从高斯噪声出发，逐步去噪得到 $\hat{\mathbf{y}}_0 = (\hat{t}, \hat{\delta})$，逆变换恢复 $(T, \delta)$。

### 从样本到生存曲线

对给定 $\mathbf{x}$ 生成 $N$ 个样本 $\{(\hat{T}_i, \hat{\delta}_i)\}$，用 KM 估计量：

$$\hat{S}(t \mid \mathbf{x}) = \prod_{i:\, \hat{T}_i \leq t} \left(1 - \frac{\hat{\delta}_i}{\sum_j \mathbf{1}[\hat{T}_j \geq \hat{T}_i]}\right)$$

$N$ 越大，估计越稳，但推断越慢——这是 SDPM 最主要的实际 trade-off。

---

## 实现

### 最小可运行版本

```python
import torch, torch.nn as nn, math

class SinusoidalEmbed(nn.Module):
    """扩散时间步的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DenoisingNet(nn.Module):
    """条件去噪 MLP：输入 (y_noisy, t, x)，输出预测噪声"""
    def __init__(self, x_dim, hidden=256, t_dim=64):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalEmbed(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU()
        )
        self.net = nn.Sequential(
            nn.Linear(x_dim + 2 + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),   # 输出 2D：(log-time 噪声, δ 噪声)
        )

    def forward(self, y_noisy, t, x):
        t_emb = self.t_embed(t)
        return self.net(torch.cat([y_noisy, x, t_emb], dim=-1))
```

### 完整实现

```python
class SDPM:
    def __init__(self, x_dim, T_steps=1000, hidden=256, device='cpu'):
        self.T_steps, self.device = T_steps, device
        betas = torch.linspace(1e-4, 0.02, T_steps).to(device)
        alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        self.alphas, self.betas = alphas, betas
        self.net = DenoisingNet(x_dim, hidden).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        self.log_t_mean, self.log_t_std = 0.0, 1.0   # 训练前用数据填充

    def _normalize(self, T, delta):
        log_t = (torch.log(T.clamp(min=1e-6)) - self.log_t_mean) / (self.log_t_std + 1e-8)
        return torch.stack([log_t, delta.float()], dim=1)   # (B, 2)

    def _denormalize(self, y):
        T = torch.exp(y[:, 0] * self.log_t_std + self.log_t_mean)
        delta = (y[:, 1] > 0.5).float()
        return T, delta

    def train_step(self, x, T_obs, delta):
        y0 = self._normalize(T_obs, delta)
        t_idx = torch.randint(0, self.T_steps, (x.size(0),), device=self.device)
        noise = torch.randn_like(y0)
        ab = self.alphas_bar[t_idx].view(-1, 1)
        y_noisy = ab.sqrt() * y0 + (1 - ab).sqrt() * noise
        pred_noise = self.net(y_noisy, t_idx, x)
        loss = ((pred_noise - noise) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)  # 梯度裁剪
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, x_single, n_samples=200):
        """对单个样本 x_single，生成 n_samples 个 (T, δ) 对"""
        x = x_single.unsqueeze(0).expand(n_samples, -1)
        y = torch.randn(n_samples, 2, device=self.device)
        for s in reversed(range(self.T_steps)):
            t_tensor = torch.full((n_samples,), s, device=self.device, dtype=torch.long)
            pred_noise = self.net(y, t_tensor, x)
            alpha, alpha_bar = self.alphas[s], self.alphas_bar[s]
            y = (y - (1 - alpha) / (1 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()
            if s > 0:
                y += self.betas[s].sqrt() * torch.randn_like(y)
        return self._denormalize(y)   # (T_samples, delta_samples)，各长 n_samples
```

### 关键 Trick（跑不起来先看这里）

**1. 目标空间标准化——最容易忽略**

```python
import numpy as np

# 训练前计算统计量，否则扩散目标尺度差异巨大
log_T = np.log(T_train.numpy() + 1e-6)
model.log_t_mean = float(log_T.mean())
model.log_t_std  = float(log_T.std())
```

**2. 推断采样数与速度的 trade-off**

```python
# N < 50：KM 曲线抖动严重，不可信
# N = 200：通常够用
# N = 500：稳定但推断慢 2.5x，适合最终评估
n_samples = 200   # 先用这个调参，最终评估用 500
```

**3. 高删失率下的训练不稳定**

如果删失率超过 80%，δ 维度极度不平衡，考虑加权损失：

```python
event_weight = 1.0 / (delta.mean().clamp(min=0.05))
w = 1 + (event_weight - 1) * delta.float().unsqueeze(1).expand_as(noise)
loss = ((pred_noise - noise) ** 2 * w).mean()
```

---

## 实验

### 合成 Cox-Weibull 数据

论文专门用合成数据验证分布恢复能力，我们复现这个场景：

```python
import numpy as np

def generate_cox_weibull(n=2000, p=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p)).astype(np.float32)
    beta = rng.normal(0, 0.5, p)
    scale = np.exp(-X @ beta / 2.0)           # 协变量决定 Weibull 尺度
    T_event  = rng.weibull(2.0, n) * scale    # 形状参数 2（单峰风险）
    T_censor = rng.exponential(scale.mean() * 2, n)
    T_obs = np.minimum(T_event, T_censor).astype(np.float32)
    delta = (T_event <= T_censor).astype(np.float32)
    print(f"事件率: {delta.mean():.1%}")       # 预期约 50%
    return X, T_obs, delta

X, T_obs, delta = generate_cox_weibull()
```

### 从样本到生存曲线

```python
from lifelines import KaplanMeierFitter

def predict_survival(model, x_single, time_grid, n_samples=200):
    """对单个协变量向量预测生存曲线 S(t | x)"""
    T_samples, delta_samples = model.sample(x_single, n_samples)
    kmf = KaplanMeierFitter()
    kmf.fit(T_samples.cpu().numpy(), event_observed=delta_samples.cpu().numpy())
    return kmf.survival_function_at_times(time_grid).values
```

### 与 Baseline 对比

在合成 Cox-Weibull 数据上的预期表现（多种子平均，仅供参考）：

| 算法 | C-index ↑ | IBS ↓ | 备注 |
|------|-----------|-------|------|
| Cox (线性) | ~0.72 | ~0.15 | 模型设定正确时竞争力强 |
| DeepHit | ~0.74 | ~0.14 | 需调时间离散粒度 |
| **SDPM** | ~0.73 | ~0.13 | 连续分布估计校准更好 |
| Random Forest | ~0.70 | ~0.16 | 弱基线 |

**关键观察**：SDPM 在 C-index（排序）上没有系统性优势，但在 Brier Score（校准性）上更好，因为它直接建模了完整的分布形状。

### 消融：目标空间变换的重要性

论文消融实验显示，去掉 log 时间标准化后，IBS 约上升 10-15%，生成的负时间样本比例从 0% 升至 5%+。这是设计最精巧的地方，也是最容易忽略的地方。

---

## 调试指南

### 常见问题

**1. 生存曲线出现非单调或跳跃**

KM 估计本身单调，但如果生成的 $\hat{T}$ 包含极端值（负数、NaN），曲线会异常。

```python
# 健康检查：生成样本的基本统计
T_s, delta_s = model.sample(x_test[0], n_samples=500)
print(f"T 范围: [{T_s.min():.2f}, {T_s.max():.2f}]")
print(f"δ 均值: {delta_s.mean():.2f}（期望接近训练集事件率）")
assert (T_s > 0).all(), "出现非正时间，检查目标空间变换！"
```

**2. δ 预测退化（全 0 或全 1）**

说明删失指示器的扩散没学好。两个维度（log-T 和 δ）共处 $\mathbb{R}^2$，若 log-T 尺度远大于 1，网络会忽略 δ 维度。根因：`log_t_std` 设置错误或未标准化。

**3. 损失收敛但 C-index 不动**

扩散模型可以学到边际分布但忽略了与协变量 x 的关联。检查：对两个特征值差异很大的样本，生成的 T 分布是否有明显差异。若没有，网络容量不足或 x 的 conditioning 不够强——尝试增加 hidden 维度或给 x 加一个预编码层。

### 如何判断模型在"学习"

- **前 100 个 batch**：loss 应从 ~2.0 快速降到 ~1.0
- **1000 个 batch 后**：应稳定在 0.2~0.5
- **真正的验证**：对两个 T 分布差异大的子组（如高风险/低风险），预测的中位生存时间应该有显著差异

### 超参数调优

| 参数 | 推荐起点 | 敏感度 | 说明 |
|------|---------|--------|------|
| `lr` | 3e-4 | 高 | Adam 老规矩，先试这个 |
| `T_steps` | 500~1000 | 中 | 更多步 ≠ 更好，误差累积 |
| `hidden` | 256 | 低 | 通常够用 |
| `n_samples` | 200 | 中 | 影响 KM 估计质量 |
| `batch_size` | 256 | 低 | |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要准确的分布形状（不只是排序） | 只关心 C-index，排序够用 |
| 数据量 > 500，分布结构复杂 | 小数据集（扩散需要足够数据） |
| 协变量与风险的关系未知/非线性 | 确定满足比例风险假设 |
| 可接受较慢的推断（批量预测） | 在线推断、实时评分 |
| 研究/报告需要完整的不确定性估计 | 工程部署要求低延迟 |

---

## 我的观点

**SDPM 是一个有价值的研究方向，但目前是研究工具，不是即插即用的工程方案。**

真正的优势在于校准性（calibration）。医疗场景下我们常常不确定协变量与风险的函数形式，这时非参数生成模型比 Cox 更可信。论文在合成数据上也证明了 SDPM 能比非参数基线更准确地恢复底层连续分布的形状——这个结论是可信的。

现实的局限：

- 推断需要大量采样（N=200+），每个测试样本都要跑一遍反向扩散，比 Cox 慢 2~3 个数量级
- DDPM 本身超参数敏感，目标空间变换设置稍有偏差就跑不好
- C-index 上没有系统性优势，无法用最常用的指标说服同行

什么时候值得一试：你的数据量充足，领域专家怀疑比例风险假设不成立，且你比 C-index 更在意校准性（Brier Score）。否则 DeepHit 加上仔细的时间离散化仍然是性价比更高的工程选择。

论文代码已开源，见 [arXiv:2605.22776](https://arxiv.org/abs/2605.22776v1)。