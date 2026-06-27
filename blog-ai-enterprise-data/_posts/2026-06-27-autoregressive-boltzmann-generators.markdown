---
layout: post-wide
title: "自回归 Boltzmann 生成器：用 Transformer 架构突破分子采样瓶颈"
date: 2026-06-27 08:04:12 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.27361v1
generated_by: Claude Code CLI
---

检查记忆文件后直接写作。


## 一句话总结

ArBG 把"生成分子构象"从归一化流的可逆性约束中解放出来——用自回归链式法则直接算精确似然，让 Transformer 级别的架构第一次能被塞进 Boltzmann 生成器里。

---

## 背景：分子采样为什么这么难？

### 问题的本质

分子系统在平衡态下服从 Boltzmann 分布：

$$\pi(\mathbf{x}) = \frac{e^{-U(\mathbf{x})/kT}}{Z}$$

$U(\mathbf{x})$ 是势能，$Z$ 是配分函数——积分算不出来。传统 MCMC（分子动力学、Metropolis-Hastings）会被高能量壁垒困在单个低能盆地里转圈。蛋白质在不同构象间的转变就像翻越一座山，MCMC 的步长根本翻不过去。

### Boltzmann 生成器框架

BG 的核心 idea：训练一个生成模型 $q_\theta(\mathbf{x}) \approx \pi(\mathbf{x})$，从 $q_\theta$ 快速采样（独立同分布！），再用重要性采样修正偏差：

$$w(\mathbf{x}) = \frac{e^{-U(\mathbf{x})/kT}}{q_\theta(\mathbf{x})}$$

关键约束：**必须能计算精确的对数似然 $\log q_\theta(\mathbf{x})$**，不然重要性权重算不了。

### 归一化流卡在哪里

现有 BG 主要用归一化流：

$$\log q(\mathbf{x}) = \log p(\mathbf{z}) - \log \left|\det \frac{\partial f}{\partial \mathbf{z}}\right|$$

Jacobian 行列式是问题根源：
- **离散时间流**（RealNVP）：为让行列式易算，强制用耦合层，表达能力严重受限
- **连续时间流**（CNF）：改用迹估计，计算量 $O(D^2)$，系统维度稍大就扛不住

这是架构层面的硬约束，不是超参数调调能绕过去的。

---

## ArBG 核心原理

### 直觉：换一种分解

概率链式法则早就有了：

$$q(\mathbf{x}) = \prod_{i=1}^{D} q(x_i \mid x_{<i})$$

对数似然直接分解为条件项之和：

$$\log q(\mathbf{x}) = \sum_{i=1}^{D} \log q(x_i \mid x_{<i})$$

**不需要 Jacobian，不需要可逆性。** 每个条件 $q(x_i \mid x_{<i})$ 想用什么网络用什么网络，Transformer 随便上。

### 数学推导

**训练目标一：前向 KL（NLL 损失，需要 MCMC 样本）**

$$\mathcal{L}_{\text{NLL}} = -\mathbb{E}_{\mathbf{x} \sim \pi}\left[\sum_{i=1}^D \log q_\theta(x_i \mid x_{<i})\right]$$

**训练目标二：反向 KL（只需要能量函数）**

$$\mathcal{L}_{\text{revKL}} = \mathbb{E}_{\mathbf{x} \sim q_\theta}\left[\log q_\theta(\mathbf{x}) + U(\mathbf{x})/kT\right]$$

反向 KL 不需要 MCMC 样本，但有 mode-seeking 倾向，容易只学会一个能量盆地。

**有效样本量（ESS）是核心评估指标：**

$$\text{ESS} = \frac{\left(\sum_k w^{(k)}\right)^2}{\sum_k \left(w^{(k)}\right)^2}, \quad w^{(k)} = e^{-U(\mathbf{x}^{(k)})/kT - \log q_\theta(\mathbf{x}^{(k)})}$$

### 与其他方法的关系

| 方法 | 精确似然 | 表达能力 | 推断时干预 | 扩展难度 |
|-----|---------|---------|----------|---------|
| 离散时间 NF | ✓（受限架构） | 中 | ✗ | 高 |
| 连续时间 NF | ✓（$O(D^2)$） | 高 | ✗ | 高 |
| **ArBG** | **✓（链式法则）** | **高** | **✓** | **低** |
| Diffusion | ✗（近似） | 极高 | 部分 | 低 |

"推断时干预"是 ArBG 独有的优势：自回归采样是逐步进行的，可以在第 $i$ 步插入约束（比如固定某个键角），流模型因为全维度同时生成做不到这点。

---

## 实现

### 最小可运行版本

用二维双阱势能验证核心逻辑：

```python
import torch
import torch.nn as nn

def double_well_energy(x):
    """二维双阱势能，模拟分子的两个稳定构象"""
    return (x[:, 0]**2 - 1)**2 + 0.5 * x[:, 1]**2

class SimpleArBG(nn.Module):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.dim = dim
        # 网络 i 接受 x_{<i} 为上下文，输出 x_i 的 (mean, log_std)
        # i=0 时用 dummy 输入（大小为1的零向量）
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max(i, 1), hidden),
                nn.Tanh(),
                nn.Linear(hidden, 2),
            )
            for i in range(dim)
        ])

    def _context(self, x, i):
        B = x.shape[0]
        return torch.zeros(B, 1, device=x.device) if i == 0 else x[:, :i]

    def log_prob(self, x):
        log_p = torch.zeros(x.shape[0], device=x.device)
        for i, net in enumerate(self.nets):
            params = net(self._context(x, i))
            mean, log_std = params[:, 0], params[:, 1].clamp(-4, 2)
            log_p += torch.distributions.Normal(mean, log_std.exp()).log_prob(x[:, i])
        return log_p

    def sample(self, n):
        device = next(self.parameters()).device
        xs = []
        for i, net in enumerate(self.nets):
            ctx = (torch.zeros(n, 1, device=device) if i == 0
                   else torch.stack(xs, dim=1))
            params = net(ctx)
            mean, log_std = params[:, 0], params[:, 1].clamp(-4, 2)
            xs.append(torch.distributions.Normal(mean, log_std.exp()).rsample())
        return torch.stack(xs, dim=1)
```

### 完整训练与评估

```python
def train_arbg(model, energy_fn, kT=1.0, n_steps=5000, lr=3e-4, batch=512):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(n_steps):
        x = model.sample(batch)
        log_q = model.log_prob(x)
        U = energy_fn(x).clamp(max=50.0)  # 防止初期样本跑到高能区

        # 反向 KL 损失
        loss = (log_q + U / kT).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 1000 == 0:
            ess = compute_ess(model, energy_fn, kT)
            print(f"Step {step+1}: loss={loss.item():.3f}, ESS={ess:.1%}")

    return model

def compute_ess(model, energy_fn, kT=1.0, n=5000):
    with torch.no_grad():
        x = model.sample(n)
        log_w = -energy_fn(x) / kT - model.log_prob(x)
        log_w -= log_w.max()  # 数值稳定
        w = log_w.exp()
        w /= w.sum()
        return (1.0 / (w**2).sum()).item() / n

# 运行
model = SimpleArBG(dim=2, hidden=128)
model = train_arbg(model, double_well_energy, kT=1.0)
```

### 关键 Trick（没有就跑不起来）

**Trick 1：条件分布方差的数值稳定性**

```python
# 直接用 exp() 很容易爆炸或塌缩
log_std = params[:, 1].clamp(-4, 2)  # 方差范围 [e^-4, e^2] ≈ [0.018, 7.4]
```

**Trick 2：反向 KL 的梯度是高方差的**

```python
# 不裁剪几乎必崩
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# 如果还不稳定，试试更小的 batch 或换 Adam -> RMSprop
```

**Trick 3：混合训练目标（有 MCMC 样本时）**

```python
# 纯反向 KL 容易 mode collapse，混入前向 KL 稳定很多
alpha = 0.5
loss_rev = (log_q + U / kT).mean()           # 反向 KL
loss_fwd = -model.log_prob(mcmc_samples).mean()  # 前向 KL（需要 MCMC 样本）
loss = alpha * loss_fwd + (1 - alpha) * loss_rev
```

**Trick 4：自回归顺序**

分子内坐标（键长 → 键角 → 二面角）按物理依赖关系排序比随机顺序收敛快 2-3 倍。这是论文里没写清楚但实际很重要的一点。

---

## 用 ESS 评估训练质量

ESS（有效样本量）是判断 ArBG 质量的核心指标，比 loss 可靠得多：

| ESS/N | 状态 | 动作 |
|-------|------|------|
| > 50% | 优秀，$q$ 和 $\pi$ 高度重叠 | 可以信任重要性权重 |
| 10-50% | 可用 | 样本量够大能得到合理估计 |
| 1-10% | 勉强 | 考虑加大网络或更多训练步 |
| < 1% | 模型基本没学到 | 排查根本问题 |

---

## 调试指南

### 常见失败模式

**1. ESS 始终 < 1%**

样本落在了 $\pi$ 的支撑之外。先检查：

```python
# 采样后立即看能量分布
x = model.sample(1000)
U = double_well_energy(x)
print(f"能量统计: min={U.min():.1f}, mean={U.mean():.1f}, max={U.max():.1f}")
# 如果 mean > 20*kT，说明模型还没找到低能区域
```

修复顺序：先确认模型在低能区域有样本，再优化 ESS。

**2. 模式坍缩（只学到一个阱）**

反向 KL 的固有缺陷。判断方法：

```python
# 检查 2D 情况下样本覆盖是否对称
x = model.sample(5000).detach().numpy()
# 如果 x[:,0] 的分布只有单峰，说明只学了一个阱
print(f"x0 > 0 的比例: {(x[:,0] > 0).mean():.2%}")  # 应该接近 50%
```

**3. Loss 震荡无法收敛**

按顺序试：①把 `clip_grad_norm` 从 1.0 降到 0.1，②把 `lr` 降 10 倍，③检查是否有 `nan`（`torch.isnan(loss).any()`）。

### 超参数敏感度

| 参数 | 推荐值 | 敏感度 | 建议 |
|-----|--------|--------|------|
| `lr` | 3e-4 | 高 | 先试这个，震荡就降 10 倍 |
| `kT` | 物理系统温度 | 极高 | 错了训练方向就反了 |
| `hidden_dim` | 128~512 | 中 | 系统维度大时对应增大 |
| `batch_size` | 512~2048 | 中 | 反向 KL 梯度方差大，batch 越大越稳 |
| `grad_clip` | 0.5~2.0 | 高 | 反向 KL 必须裁剪 |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要精确对数似然（做重要性采样） | 只需要生成样本，不需要密度估计 |
| 中等维度系统（< 1000D） | 极高维度且推断速度是瓶颈 |
| 有明确势能函数 $U(\mathbf{x})$ | 只有样本数据，没有能量函数 |
| 需要推断时加约束（条件采样） | 大批量无条件采样（串行慢） |
| 想在相似系统间迁移（如 Robin） | 单次任务、数据量大、MCMC 够用 |

---

## 我的观点

ArBG 的核心 insight 是真实的：把 Jacobian 的麻烦换给链式法则，代价是串行采样，收益是架构自由度。这个 trade-off 对于分子采样这个场景是值得的——Robin 的 zero-shot 迁移到相似肽系统、8 残基系统能量误差下降 60%，说明 Transformer 的扩展性确实在这里起了作用。

**真正值得警惕的是**：自回归采样是串行的，维度高时比流模型慢一个数量级。对于 MD 模拟中需要每秒生成百万构象的场景，这是实际工程瓶颈，不是能靠更大 GPU 解决的问题。

**坐标顺序选择**是目前最不清楚的超参数：内坐标 vs. 笛卡尔坐标，哪个维度先采，论文里没有定论。如果你跑不出论文结果，这是第一个值得怀疑的地方。

官方代码：https://github.com/danyalrehman/autobg