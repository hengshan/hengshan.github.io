---
layout: post-wide
title: "多任务贝叶斯上下文学习：让 Transformer 在推理时切换先验"
date: 2026-06-20 08:05:41 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.20538v1
generated_by: Claude Code CLI
---

## 一句话总结

把"历史任务数据集"当作先验的上下文表示，训练 Transformer 动态适配不同先验——以 PFN 的推理速度，实现接近 Oracle 贝叶斯的分布外泛化能力。

---

## 背景：先验不该烙印在权重里

贝叶斯预测推断要计算：

$$p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) \, p(\theta \mid \mathcal{D}) \, d\theta$$

这个积分几乎总是无解析解。现有摊销方案的主要问题：

- **MCMC**：理论完备，实际太慢
- **变分推断**：假设太强，后验质量难以保证
- **Prior-Data Fitted Networks（PFNs）**：Transformer 直接输出预测分布，速度极快——但先验被硬编码进了权重。**测试时换先验，只能重新训练。**

### 层级贝叶斯的视角

很多现实问题有层级结构：超参数 $\varphi$ 决定任务族，每个任务的参数 $\theta_i$ 从 $p(\theta \mid \varphi)$ 中采样：

$$\varphi \;\longrightarrow\; \theta_i \sim p(\theta \mid \varphi) \;\longrightarrow\; \mathcal{D}_i \sim p(\mathcal{D} \mid \theta_i)$$

我们往往不直接观测 $\varphi$，但能观测到来自同一 $\varphi$ 的多个任务数据集。**这些数据集本身就隐含了先验信息。**

本文的 Insight：不把 $\varphi$ 编进权重——把来自同一 $\varphi$ 的先验任务数据集作为**上下文前缀**，让 Transformer 从中读取先验，再对目标任务做预测。

---

## 算法原理

### 直觉：先验即数据

传统 PFN 的序列结构：

```
[目标训练集 D_target] → 查询 x* → 预测 y*
```

多任务贝叶斯 ICL 的序列结构：

```
[先验 D_1, D_2, ..., D_k] + [目标训练集 D_target] → 查询 x* → 预测 y*
│← 先验前缀（隐式编码 φ）→│                           ↑
                                                  y=0 占位，待预测
```

Transformer 通过全注意力同时看到所有上下文，从先验前缀推断 $\varphi$，再对目标任务做"贝叶斯更新"。

### 训练目标

模型学习的映射：

$$f_\psi\!\left(\mathcal{D}_1, \ldots, \mathcal{D}_k,\, \mathcal{D}_{\text{target}},\, x^*\right) \approx p\!\left(y^* \mid x^*, \mathcal{D}_{\text{target}}, \varphi\right)$$

训练对整个层级先验分布上采样，最大化查询点的对数似然。

### 与 PFN 的关键区别

| | PFN | MT-BICL（本文）|
|---|---|---|
| 先验表示 | 隐含在权重中 | 显式为上下文前缀 |
| 测试时换先验 | 需重新训练 | 换前缀即可 |
| 分布外先验 | 性能显著降级 | 通过前缀自适应 |
| 推理速度 | 一次前向传播 | 一次前向传播 |

---

## 实现

### 最小可运行版本：数据生成

用 GP 回归作为玩具任务，lengthscale 就是超参数 $\varphi$：

```python
import torch
import torch.nn as nn

def sample_gp_task(lengthscale, n_pts=20, noise_std=0.1):
    """从单个 GP 任务采样（固定 lengthscale = θ）"""
    x = torch.rand(n_pts) * 6 - 3
    y = torch.sin(x / lengthscale) + torch.randn(n_pts) * noise_std
    return x, y

def sample_hierarchical_batch(n_prior=5, n_train=10, n_query=5):
    """
    层级结构: φ → {D_i}
    - prior_xys:    k 个先验任务（暗示 φ）
    - target_train: 目标任务训练集
    - target_query: 目标任务查询集（y 是监督信号）
    """
    lengthscale = torch.exp(torch.randn(1) * 0.3 + 0.3).item()

    prior_xys = [sample_gp_task(lengthscale) for _ in range(n_prior)]

    x_all, y_all = sample_gp_task(lengthscale, n_train + n_query)
    return prior_xys, (x_all[:n_train], y_all[:n_train]), \
                      (x_all[n_train:], y_all[n_train:])
```

### 完整模型实现

```python
class MTBayesICL(nn.Module):
    """
    多任务贝叶斯上下文学习
    核心：先验任务数据集 → 上下文前缀 → Transformer 自适应
    """
    def __init__(self, d_model=64, n_heads=4, n_layers=4):
        super().__init__()
        self.embed = nn.Linear(2, d_model)       # (x, y) → d_model
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True        # dropout=0 见 Trick #3
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model, 2)        # 输出 (μ, log σ)

    def build_context(self, prior_xys, target_train_xy):
        """将先验任务 + 目标训练集打包为 token 序列 [L, 2]"""
        tokens = [torch.stack([x, y], dim=-1) for x, y in prior_xys]
        x_t, y_t = target_train_xy
        tokens.append(torch.stack([x_t, y_t], dim=-1))
        return torch.cat(tokens, dim=0)

    def forward(self, context_xy, query_x):
        """
        context_xy: [B, L, 2]  先验前缀 + 目标训练
        query_x:    [B, Q]     查询点（y 未知）
        返回: μ [B,Q], log σ [B,Q]
        """
        B, Q = query_x.shape
        # 查询 token：y 用 0 占位
        q_tok = torch.stack([query_x, torch.zeros_like(query_x)], dim=-1)
        seq = torch.cat([context_xy, q_tok], dim=1)   # [B, L+Q, 2]

        h = self.embed(seq)
        h = self.transformer(h)

        out = self.head(h[:, -Q:, :])                 # [B, Q, 2]
        return out[..., 0], out[..., 1]               # μ, log σ
```

### 训练循环

```python
def train(model, n_steps=5000, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(n_steps):
        # 随机化先验任务数量（关键 Trick，见下文）
        n_prior = torch.randint(1, 10, (1,)).item()
        prior_xy, target_train, (x_q, y_q) = sample_hierarchical_batch(n_prior)

        ctx = model.build_context(prior_xy, target_train).unsqueeze(0)  # [1,L,2]
        x_q_in = x_q.unsqueeze(0)                                       # [1,Q]

        mu, log_sigma = model(ctx, x_q_in)
        sigma = log_sigma[0].exp().clamp(min=1e-4)

        # 高斯负对数似然
        loss = (0.5 * ((y_q - mu[0]) / sigma) ** 2 + log_sigma[0]).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 必须！
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step:5d} | loss={loss.item():.4f}")
```

### 关键 Trick

**1. 任务内标准化（影响大）**

```python
def normalize_task(x, y):
    """每个任务独立归一化，消除不同 φ 的量纲差异"""
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return x, y
```

不做这一步，不同 lengthscale 的函数幅度相差悬殊，模型会被尺度信息干扰，忽视真正的先验信号。

**2. 训练时随机化 n_prior**

让模型学会从 1 个到 10 个先验任务中推断 $\varphi$，否则测试时一旦前缀长度变化，性能直接崩。

**3. Dropout = 0**

元学习中，dropout 会随机丢弃跨任务的信息传递通路，实验上有害无益。

**4. 梯度裁剪不可省**

序列长度随 n_prior 变化，梯度范数波动极大。`clip_grad_norm_(..., 1.0)` 是稳定训练的底线。

---

## 实验

下表定性反映论文方法趋势（数值为示意）：

| 方法 | 训练分布内 RMSE | OOD 先验 RMSE | 推理速度 |
|------|:---:|:---:|:---:|
| MCMC（Oracle）| 0.10 | 0.10 | 极慢 |
| PFN（固定先验）| 0.12 | 0.51 | 快 |
| Flat ICL（无前缀）| 0.18 | 0.35 | 快 |
| **MT-BICL（本文）** | **0.11** | **0.14** | 快 |

核心结论：OOD 先验场景下，固定先验的 PFN 性能崩溃，MT-BICL 几乎追平 Oracle。

### 先验任务数量的影响

先验任务从 0 增加到 5 时，OOD RMSE 显著下降；超过 5 之后收益递减。这是模型确实在利用前缀信息的直接证明——而不是"学到了什么偏捷径"。

---

## 调试指南

### 症状 1：模型完全忽略先验前缀

**诊断**：分别用 n_prior=0 和 n_prior=5 测试，预测结果几乎一样。

**原因与解法**：

- 训练时 n_prior 固定，模型没有学到"利用前缀"的动机 → 随机化 n_prior
- n_train 太大，目标训练集已经足够推断 $\varphi$，前缀变成了噪声 → 减小 n_train，逼迫模型依赖前缀
- 学习率过高，优化路径直接收敛到忽视前缀的局部最优 → 降到 1e-4 重新训练

### 症状 2：loss 振荡剧烈，不收敛

原因几乎总是梯度爆炸。检查：

```python
# 在 loss.backward() 后、optimizer.step() 前加一行
grad_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)**0.5
print(f"grad_norm={grad_norm:.2f}")
```

如果 grad_norm > 10，立即加上 `clip_grad_norm_`，同时考虑降低 lr。

### 症状 3：训练 loss 下降，OOD 性能仍然差

这是"伪收敛"——模型学会了拟合训练分布内的任务，但先验适配能力没有泛化。

**解法**：训练时主动引入 OOD 先验。从更宽的 $\varphi$ 分布采样（如把 lengthscale 范围翻倍），或加入"对抗先验"样本。

### 超参数参考

| 参数 | 推荐范围 | 敏感度 | 建议 |
|------|---------|--------|------|
| lr | 1e-4 ~ 3e-4 | 高 | 从 3e-4 开始，不收敛就降 |
| d_model | 64 ~ 256 | 中 | 128 通常够用 |
| n_layers | 3 ~ 6 | 中 | 4 是好起点 |
| n_prior（训练）| 随机 1~10 | **极高** | 固定值会导致 OOD 性能差 |
| n_heads | 4 ~ 8 | 低 | |

---

## 什么时候用，什么时候不用

| 适用场景 | 不适用场景 |
|---------|-----------|
| 测试先验可能与训练先验不同 | 先验完全固定且已知（直接用 PFN）|
| 有相关历史任务可用作先验 | 每个任务完全独立无关 |
| 需要快速推理，无法跑 MCMC | 数据极少，必须精确贝叶斯 |
| 小样本回归、分类、时序预测 | 大规模标准监督学习 |
| 先验随时间/场景动态变化 | 计算资源极度有限 |

---

## 我的观点

这篇论文解决了 PFN 最尴尬的缺陷——**先验不可变**。把先验表示为数据而非权重，这个 insight 简洁而有力，是 in-context learning 思路在贝叶斯推断领域的自然延伸。

但有几点要诚实地说：

1. **先验任务需要足够相关**。如果先验任务和目标任务来自不同的分布族（不共享 $\varphi$），前缀不仅无益，可能还会干扰预测。

2. **上下文长度是瓶颈**。先验任务越多效果越好，但序列长度线性增长，Transformer 的二次注意力代价在任务很多时仍然明显。FlashAttention 可以缓解，但不能根治。

3. **本质是贝叶斯元学习**。和 MAML、ProtoNet 做的事情有相似之处——用任务历史适配新任务——只是用了贝叶斯视角，且避免了显式的元梯度更新。如果你的场景已经有成熟的元学习方案，不必急着切换。

最值得一试的场景：**先验随时间或用户动态变化的预测问题**，比如个性化推荐、气象建模、自适应实验设计。前缀机制天然支持"在线更新先验"，只需把最新的相关任务数据追加到前缀。

官方代码：[https://github.com/martianmartina/multi-task-bayesian-icl/](https://github.com/martianmartina/multi-task-bayesian-icl/)