---
layout: post-wide
title: "用凸包重建加速 LLM 测试时微调：HullFT 解析"
date: 2026-05-31 08:03:28 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.30337v1
generated_by: Claude Code CLI
---

## 一句话总结

HullFT 将每条 query 的 embedding 表达为训练样本的稀疏凸组合，同时解决了测试时微调的两个瓶颈：如何快速选出相关且多样的支撑集，以及如何复用重复样本的梯度计算。

---

## 为什么测试时微调很难？

**测试时微调（Test-Time Finetuning，TTFT）**的思路很直接：推理前，针对当前 query 检索若干相关训练序列，在这些序列上短暂微调模型，再做推理。

问题在于，每条 query 都要做一次微调，延迟直接叠在推理路径上。现有方法陷入两难：

- **快速检索（k-NN）**：用向量相似度取 Top-k，速度快但选出来的样本高度冗余，对质量提升有限
- **多样性感知检索**：显式优化覆盖度，效果好但每次都要跑一个优化问题，单次开销就把延迟打爆

HullFT 的核心洞见是：**这两个问题其实共享同一个几何结构**。把 query 表达成训练样本的凸组合，相关性和多样性都自然涌现——你需要足够近的点（相关性），也需要足够多方向的点才能"撑起" query 所在位置（多样性）。

---

## 核心方法：三步走

### 第一步：凸包重建（Convex Reconstruction）

给定 query embedding $q \in \mathbb{R}^d$ 和训练集 embeddings $X = \{x_1, \ldots, x_n\} \subset \mathbb{R}^d$，求：

$$\min_{w \in \Delta^n} \left\| \sum_{i=1}^n w_i x_i - q \right\|^2$$

其中 $\Delta^n = \{w \mid w_i \geq 0,\ \sum_i w_i = 1\}$ 是概率单纯形。

这个问题的解 $w^*$ 是稀疏的——大多数 $w_i = 0$，只有少量样本有正权重，这些样本就是支撑集。与 k-NN 不同，这里的选择是**全局最优**的：给定预算 k，它找到能最好还原 query embedding 的 k 个样本组合。

### 第二步：Frank-Wolfe 求解

直接对单纯形投影开销 $O(n \log n)$，而 Frank-Wolfe（条件梯度法）只需每轮做一次**线性极小化**，对单纯形而言就是 argmin——时间复杂度 $O(n)$。

迭代格式：

$$s^{(t)} = e_{i^*}, \quad i^* = \arg\min_i \left[ \nabla_w f(w^{(t)}) \right]_i$$

$$w^{(t+1)} = (1 - \gamma_t)\, w^{(t)} + \gamma_t\, s^{(t)}$$

目标函数梯度为 $\nabla_w f = 2X(Xw - q)^T$，精确步长可解析求得：

$$\gamma^* = \frac{-\nabla f(w)^\top d}{2\, \|Xd\|^2}, \quad d = s^{(t)} - w^{(t)}$$

Frank-Wolfe 每步把一个新样本"加进来"，天然产生稀疏解，且每次迭代只扫一遍候选集。

### 第三步：整数化 + 梯度复用

凸组合权重 $w^*$ 是连续的，但微调需要离散的训练样本集合。HullFT 把 $w^*$ 转为整数多重集：给定总预算 $B$，令 $m_i = \lfloor w_i^* \cdot B \rfloor$，再把剩余配额分给小数部分最大的样本。

整数化后，某些样本会出现多次（multiplicity $m_i > 1$）。对于这些重复样本，梯度计算只需做一次，然后乘以倍数：

$$g_i^{\text{reuse}} = m_i \cdot \nabla_\theta \mathcal{L}(\theta;\, x_i)$$

这就是 **Gradient Reuse**——把本来 $\sum_i m_i$ 次前向-反向传播，降到只做 $|\text{support}|$ 次。

---

## 代码实现

### Frank-Wolfe 凸组合求解器

```python
import torch
import torch.nn.functional as F

def frank_wolfe_reconstruct(query: torch.Tensor,
                            corpus: torch.Tensor,
                            n_iter: int = 50,
                            tol: float = 1e-6) -> torch.Tensor:
    """
    将 query 表达为 corpus 行向量的稀疏凸组合。
    query:  (d,)
    corpus: (n, d)  —— 预先 L2 归一化效果更好
    返回:  w (n,)，稀疏的概率权重向量
    """
    n = corpus.shape[0]
    # 初始化：选最相似的单个样本
    sims = corpus @ query  # (n,)
    w = torch.zeros(n, dtype=query.dtype)
    w[sims.argmax()] = 1.0

    for _ in range(n_iter):
        recon = w @ corpus          # 当前重建 (d,)
        residual = recon - query    # (d,)

        # 梯度 ∂f/∂w_i = 2 * x_i · residual
        grad = 2.0 * (corpus @ residual)  # (n,)

        # FW 线性步：在单纯形上极小化线性近似 = 选 argmin
        s = torch.zeros_like(w)
        s[grad.argmin()] = 1.0

        d = s - w  # 下降方向
        Xd = d @ corpus  # (d,)

        denom = 2.0 * (Xd @ Xd)
        if denom < 1e-12:
            break

        # 精确线搜索
        gamma = (-(grad @ d) / denom).clamp(0.0, 1.0)
        w_new = w + gamma * d

        if (w_new - w).norm() < tol:
            w = w_new
            break
        w = w_new

    return w.clamp(min=0)  # 数值误差修正
```

### 权重整数化

```python
def integerize_weights(w: torch.Tensor, budget: int) -> torch.Tensor:
    """
    把连续权重转为整数多重集，总和恰好等于 budget。
    采用"最大余数法"（Hamilton method），避免舍入偏差。
    """
    scaled = w * budget
    m = scaled.floor().long()
    remainder = budget - m.sum().item()

    if remainder > 0:
        # 把剩余名额分给小数部分最大的样本
        fracs = scaled - m.float()
        top_idx = fracs.topk(int(remainder)).indices
        m[top_idx] += 1

    return m  # shape (n,), sum == budget
```

### 梯度复用微调

```python
def gradient_reuse_step(model, loss_fn, examples, multiplicities, optimizer):
    """
    对支撑集做一步微调，重复样本只计算一次梯度。
    examples:      List[样本]
    multiplicities: torch.Tensor (n,)，整数倍数
    """
    optimizer.zero_grad()
    support_idx = multiplicities.nonzero(as_tuple=False).squeeze(-1)

    for idx in support_idx:
        m = multiplicities[idx].item()
        loss = loss_fn(model, examples[idx])
        # 反向传播，梯度乘以倍数，累积到 .grad
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(m)  # 梯度复用：乘以出现次数

    # 所有支撑样本的梯度已累积
    optimizer.step()
```

### 完整 HullFT 推理流程

```python
class HullFT:
    def __init__(self, model, corpus_embeddings, corpus_examples,
                 budget=10, n_fw_iter=50, finetune_steps=1):
        self.model = model
        self.corpus_emb = F.normalize(corpus_embeddings, dim=-1)
        self.corpus_ex  = corpus_examples
        self.budget = budget
        self.n_fw_iter = n_fw_iter
        self.ft_steps = finetune_steps

    @torch.no_grad()
    def select(self, query_emb: torch.Tensor):
        q = F.normalize(query_emb, dim=-1)
        w = frank_wolfe_reconstruct(q, self.corpus_emb, self.n_fw_iter)
        m = integerize_weights(w, self.budget)
        return m  # 各样本在微调集中出现次数

    def adapt_and_infer(self, query, query_emb, loss_fn, infer_fn):
        # 保存原始权重
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # 选择支撑集
        m = self.select(query_emb)

        # 微调
        opt = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        for _ in range(self.ft_steps):
            gradient_reuse_step(self.model, loss_fn,
                                self.corpus_ex, m, opt)

        # 推理
        result = infer_fn(self.model, query)

        # 恢复原始权重（每次 query 互不干扰）
        self.model.load_state_dict(original_state)
        return result
```

### 实现中的坑

**坑 1：corpus embedding 必须预先归一化**

Frank-Wolfe 用内积衡量相似度，不归一化会让高范数样本被过度选中：

```python
# 错误：直接用原始 embedding
corpus_emb = model.encode(corpus_texts)

# 正确：L2 归一化后存储
corpus_emb = F.normalize(model.encode(corpus_texts), dim=-1)
```

**坑 2：多步微调时梯度复用会引入偏差**

当 `finetune_steps > 1` 时，第二步的梯度是在更新后的参数上计算的，无法再复用第一步的梯度。论文结果主要在 `finetune_steps=1` 下成立，多步时需重新计算：

```python
# 仅单步时梯度复用是精确的
# 多步微调时应退化为标准循环
if self.ft_steps > 1:
    # 展开重复样本，正常训练
    expanded = [self.corpus_ex[i] for i, cnt in enumerate(m)
                for _ in range(cnt.item())]
```

**坑 3：Frank-Wolfe 收敛速度对初始点敏感**

对高维 embedding（≥ 768d），从随机初始点出发可能需要数百次迭代。用最近邻作为起点可以把迭代次数降到 20-30 次。

---

## 论文结果 vs 现实

论文报告 HullFT 在语言建模（bits-per-byte 指标）上优于 kNN-TTFT 等基线，同时总 runtime 更低。我认为有几点需要注意：

**可信度较高的部分：**
- Gradient Reuse 的加速是精确的，savings 随 support 的稀疏度线性增长
- Frank-Wolfe 在低支撑数（k ≤ 10）时确实比暴力最优化快

**需要谨慎的部分：**
- 论文的实验集中在语言建模任务；对 reasoning、code 等任务是否同样有效尚不明确
- budget B 和 n_fw_iter 是新增超参数，实验室场景下调好了，生产环境不一定迁移

---

## 适用边界

| 适用场景 | 不适用场景 |
|---------|-----------|
| query 分布与训练集接近（in-distribution） | OOD query 超出训练集凸包范围 |
| 已经在用 TTFT，想降低每次微调开销 | 对延迟极度敏感、不能接受任何额外检索开销 |
| 训练集足够大（> 10K），凸包覆盖度好 | 训练集极小（< 1K），凸包近似质量差 |
| 单步或少步微调（1-3 步） | 需要大量微调步数才能收敛 |

---

## 我的看法

HullFT 最聪明的地方不是 Frank-Wolfe，也不是梯度复用——这两个技术都不新。聪明的是**把它们串联成一个整体**：凸组合权重自然产生重复样本，重复样本自然触发梯度复用。整个流程里没有多余的设计，每步都在服务下一步。

真正的开放问题是：**TTFT 本身是否值得**？每次推理都要微调模型，这在延迟预算有限的场景下依然是奢侈的。HullFT 改进了效率，但没有改变 TTFT 的根本代价。更可能的未来是把这种"基于检索的自适应"思路融合进 in-context learning 或 LoRA adapter 缓存里，而不是每次都动原始参数。

不过，对于离线批处理、医疗/法律等高准确率要求的场景，TTFT 是有实际意义的。HullFT 在这个细分赛道里是目前最干净的解法之一。

---

**论文链接**：[arxiv 2605.30337](https://arxiv.org/abs/2605.30337v1)