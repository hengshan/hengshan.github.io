---
layout: post-wide
title: "UniPool：用全局共享专家池重新设计混合专家（MoE）架构"
date: 2026-05-08 12:05:50 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.06665v1
generated_by: Claude Code CLI
---

## 一句话总结

UniPool 用一个全局共享专家池替代传统 MoE 的 per-layer 独立专家集，在用 41.6%-66.7% 专家参数的情况下仍能持平甚至超越 vanilla MoE，同时揭示了深层路由的本质冗余。

---

## 背景：per-layer MoE 设计有什么问题？

MoE 架构的核心吸引力是**稀疏激活**：模型参数很多，但每次前向只激活一小部分。Mixtral、DeepSeek-MoE、Switch Transformer 都继承了同一个隐性假设：

> **每个 Transformer 层需要一套独立的专家集合。**

这个设计带来了一个被忽视的代价：**专家参数随深度线性增长**。32 层 × 8 专家 = 256 个独立 FFN 模块，专家间零共享。

UniPool 的出发点是一个刁钻的实验：**把深层学习到的 top-k 路由替换成均匀随机路由，验证精度只下降 1.0-1.6 个百分点**。这说明深层 MoE 路由在很大程度上是冗余的——专家被调用，但"调用谁"并不那么重要。

### 核心洞察

深层 transformer 层不需要"专属"专家，它们只需要访问一个足够大的**全局专家库**。把 per-layer 专家集合换成一个中央共享池，每层只保留独立的路由器。

| | Vanilla per-layer MoE | UniPool |
|---|---|---|
| 专家参数增长 | 随深度线性增长 | 可次线性增长 |
| 深层路由利用率 | 低（随机化损失 < 2 点） | 全局视角统一分配 |
| 池大小 | 隐式（层数 × 专家数） | 显式超参数 |

---

## 算法原理

### 直觉：中央图书馆 vs. 每层独立书架

传统 MoE 像每个楼层都有独立书架（per-layer 专家集），每次只能用本层的书。UniPool 建了一个中央图书馆（global expert pool），每层有独立的管理员（per-layer router）决定借哪本书，专家本身不"属于"任何层。

### 数学推导

**Vanilla MoE 第 $l$ 层：**

$$
\text{MoE}_l(x) = \sum_{i \in \text{TopK}(G_l(x))} g_{l,i}(x) \cdot E_{l,i}(x)
$$

**UniPool（所有层共享专家池 $\mathcal{P}$）：**

$$
\text{UniPool}_l(x) = \sum_{i \in \text{TopK}(G_l(x))} g_{l,i}(x) \cdot E_i(x), \quad E_i \in \mathcal{P}
$$

关键变化只有一处：$E_{l,i}$ → $E_i$，去掉了层索引 $l$。专家不再属于任何层，每层路由器 $G_l$ 从同一个池中选。

### NormRouter

多层共享同一批专家时，不同层的激活尺度差异会破坏路由稳定性。NormRouter 在路由前对输入和权重都做 L2 归一化：

$$
s(x) = \frac{x}{\|x\|_2} \cdot \frac{W_r^T}{\|W_r\|_2}
$$

路由分数只依赖**方向**，不受激活幅度影响。

### Pool-level 辅助损失

Per-layer 辅助损失只在单层内平衡负载，无法阻止不同层聚焦于池子的相同区域。UniPool 从**全局视角**计算均衡损失：

$$
\mathcal{L}_{\text{pool}} = \alpha \cdot \sum_{i=1}^{N_e} \left( f_i - \frac{1}{N_e} \right)^2
$$

$f_i$ 是专家 $i$ 在所有层、所有 token 上的全局平均路由概率。

---

## 实现

### 最小可运行版本

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class NormRouter(nn.Module):
    """归一化路由器：方向相似度替代原始点积"""
    def __init__(self, d_model, pool_size, top_k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(pool_size, d_model))
        self.top_k = top_k

    def forward(self, x):
        x_n = F.normalize(x, dim=-1)
        w_n = F.normalize(self.weight, dim=-1)
        probs = F.softmax(x_n @ w_n.T, dim=-1)           # (N, pool_size)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        # 重归一化，使 top-k 权重之和为 1
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)
        return topk_probs, topk_idx, probs  # probs 保留用于辅助损失

class UniPoolLayer(nn.Module):
    """单个 UniPool 层：使用外部共享的专家池"""
    def __init__(self, d_model, expert_pool, top_k):
        super().__init__()
        self.router = NormRouter(d_model, len(expert_pool), top_k)
        self.expert_pool = expert_pool  # 共享引用，不复制参数
        self.top_k = top_k

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        topk_w, topk_idx, full_probs = self.router(x_flat)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for i, expert in enumerate(self.expert_pool):
                mask = topk_idx[:, k] == i
                if mask.any():
                    out[mask] += topk_w[mask, k:k+1] * expert(x_flat[mask])

        return out.view(B, T, D), full_probs  # 返回完整概率分布
```

### 完整实现：含全局辅助损失

```python
class UniPoolMoE(nn.Module):
    """UniPool 完整实现，含 pool-level 负载均衡"""
    def __init__(self, d_model, d_ff, pool_size, num_layers, top_k=2):
        super().__init__()
        # 所有层共享同一个专家池
        self.expert_pool = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(pool_size)
        ])
        # 每层有独立的路由器（参数不共享）
        self.moe_layers = nn.ModuleList([
            UniPoolLayer(d_model, self.expert_pool, top_k)
            for _ in range(num_layers)
        ])
        self.pool_size = pool_size

    def pool_aux_loss(self, all_probs):
        """
        all_probs: list of (B*T, pool_size) 张量，每层一个
        使用连续概率而非离散 indices，保证梯度回传
        """
        # 所有层、所有 token 的平均专家使用率
        total_freq = sum(p.mean(0) for p in all_probs) / len(all_probs)
        target = torch.ones_like(total_freq) / self.pool_size
        return ((total_freq - target) ** 2).sum()

    def forward(self, x, aux_coef=0.01):
        all_probs = []
        for layer in self.moe_layers:
            x, probs = layer(x)
            all_probs.append(probs)
        aux_loss = self.pool_aux_loss(all_probs) * aux_coef
        return x, aux_loss
```

### 对比：Vanilla per-layer MoE

```python
class VanillaMoELayer(nn.Module):
    """传统 MoE：每层独立专家集"""
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        probs = F.softmax(self.router(x_flat), dim=-1)
        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for i, expert in enumerate(self.experts):
                mask = topk_idx[:, k] == i
                if mask.any():
                    out[mask] += topk_w[mask, k:k+1] * expert(x_flat[mask])
        return out.view(B, T, D)

# 参数量对比（32层，每层8专家，每专家参数量 P）
# Vanilla:          32 × 8 × P = 256P
# UniPool (pool=16): 16 × P + 32 个路由器 ≈ 16P  (节省约 37.5% 专家参数)
```

---

## 关键 Trick（不写这些就跑不起来）

### 1. NormRouter 是必需的，不是可选的

不加归一化时，共享池会出现严重的**专家崩溃**：少数专家被所有层疯狂抢占，其余饿死。原因是不同层的激活尺度差异会主导路由决策。

```python
# 不稳定：裸 softmax 受激活幅度干扰
scores = F.softmax(self.router(x), dim=-1)

# 稳定：只用方向信息
scores = F.softmax(F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).T, dim=-1)
```

### 2. 辅助损失必须用连续概率，不能用 indices

```python
# 错误：indices 是离散的，梯度无法回传路由器
usage = (indices == i).float().mean()  # 无梯度！

# 正确：用路由概率的均值，梯度完整保留
freq = probs.mean(0)  # (pool_size,)，有梯度
aux = ((freq - 1/pool_size) ** 2).sum()
```

### 3. pool_size 是显式超参数，必须主动调

论文建议从 vanilla 总专家数的 50% 开始，然后按验证损失做二分搜索。盲目用 100% 并非最优，反而浪费计算。

---

## 实验分析

论文在 LLaMA 架构（182M-978M 参数，30B tokens from Pile）上的核心结论：

| 配置 | 专家参数占比 | 验证损失变化（↓越好） |
|------|------------|------------------|
| Vanilla MoE | 100% | 基线 |
| UniPool（完整池） | 100% | 最多 -0.0386 |
| UniPool（缩减池） | 41.6%-66.7% | ≈ 基线或更低 |

**反直觉结论**：减少专家参数反而效果更好。全局共享迫使专家变得更通用，减少了"专家死亡"问题。

### 随机路由探针的深层含义

"深层路由换成随机路由只损失 1-2 点"——这说明深层 MoE 的瓶颈不是路由质量，而是**专家本身的表达能力**。UniPool 通过让更多专家对深层可见，直接解决了这个瓶颈。

---

## 调试指南

### 常见问题

**1. 训练初期损失正常，但池辅助损失一直不下降**

大概率是辅助损失梯度断了。检查 `pool_aux_loss` 的输入是否是连续概率张量（带 `requires_grad=True`），而不是 `torch.topk` 返回的 indices。

**2. 学习曲线比 vanilla MoE 更差**

先排查 `pool_size` 是否太小。UniPool 在池太小时退化为强迫所有层共享极少专家，等于大幅降低了模型容量。建议先用 `pool_size = num_layers * experts_per_layer`（等于 vanilla 总专家数），确认能持平后再缩减。

**3. 不同层的路由几乎完全相同**

用以下代码监控每层路由多样性：

```python
def routing_entropy(probs_list):
    """诊断每层路由是否真的在使用不同专家"""
    for l, probs in enumerate(probs_list):
        avg_usage = probs.mean(0)  # (pool_size,)
        H = -(avg_usage * (avg_usage + 1e-9).log()).sum()
        H_max = torch.tensor(len(avg_usage)).float().log()
        print(f"Layer {l}: entropy = {H:.3f} / {H_max:.3f} = {H/H_max:.1%}")
        # 如果所有层的熵都接近 100%，说明路由均匀但无差异化
        # 如果熵很低（< 50%），说明专家崩溃
```

### 超参数敏感度

| 参数 | 推荐起点 | 敏感度 | 调优建议 |
|------|---------|-------|--------|
| `pool_size` | vanilla 总专家数 × 0.5 | 高 | 最先调，二分搜索 |
| `top_k` | 2 | 低 | 先固定 |
| `aux_coef` | 0.01 | 中 | 太大压制专业化，太小导致崩溃 |
| 学习率 | 3e-4 | 高 | NormRouter 允许略大的 lr |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 预训练大语言模型（>1B 参数） | 浅层模型（< 8 层），共享收益小 |
| 专家参数预算受限 | 已有精调完善的 per-layer MoE 基线 |
| 研究 MoE 路由机制 | 推理延迟极度敏感（共享池索引有额外开销） |
| 想把 pool_size 作为深度-参数权衡旋钮 | 任务高度多领域（路由专业化需求强） |

---

## 我的观点

UniPool 的真正贡献是**把一个模糊的工程直觉变成了可测量的架构选择**。"深层路由冗余"这个想法不新鲜，但随机路由探针定量地证明它，再基于此把 `pool_size` 提升为深度-参数权衡的显式超参——这是扎实的系统性工作。

几个疑虑值得注意：

- **规模上限未知**：实验最大 978M 参数，7B+ 规模是否成立没有证据
- **专家专业化代价**：全局共享是否削弱专家的领域专业化？论文没有深入分析这一点
- **工程开销**：共享池的内存访问模式比 per-layer 差，实际训练吞吐量需要实测，不能只看参数量

如果你在做 MoE 预训练实验，建议把 UniPool（缩减池版本）作为一个**参数效率基线**来对比。能在自己规模上复现"少参数等效果"这一条，pool_size 就成了你调参工具箱里的新旋钮。

论文链接：https://arxiv.org/abs/2605.06665v1