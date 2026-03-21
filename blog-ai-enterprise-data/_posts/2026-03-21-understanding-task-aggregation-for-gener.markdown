---
layout: post-wide
title: "超声基础模型的任务聚合：联合训练何时有益，何时有害？"
date: 2026-03-21 08:03:46 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.18123v1
generated_by: Claude Code CLI
---

## 一句话总结

联合训练多个临床超声任务不一定比任务专属模型更好——数据规模和任务类型共同决定了聚合策略的成败，而非临床分类。

---

## 为什么这篇论文重要？

医学 AI 的"基础模型梦"是训练一个模型搞定所有临床任务。但现实残酷：很多超声基础模型研究发现，联合训练的统一模型**反而比任务专属模型更差**。

直觉上我们会怀疑是模型容量不够。但这篇论文的核心主张是：**问题出在任务聚合策略上，而不是模型容量上**。被忽视的是两个关键变量的交互：

- **任务异质性**：分割、分类、检测、回归之间有多大差异？
- **训练数据规模**：每个任务有多少标注数据？

现有研究的常见做法是按临床类别分组（器官系统、疾病组），认为"临床上相关的任务，特征上也应该相似"。这篇论文通过 27 个任务的系统实验证明：**这个假设是危险的**。

> 核心洞见：临床分类 ≠ 特征空间分类。按临床分组训练在数据充足时有效，在数据稀缺时可能造成严重负迁移。

---

## 核心方法：M2DINO

论文提出 **M2DINO**（Multi-organ, Multi-task DINO），建立在 DINOv2 骨干网络上，关键创新是**任务条件化的混合专家（Task-conditioned MoE）块**，用于自适应容量分配。

### 直觉解释

标准 MoE 的路由决策发生在 **token 级别**：同一张图像内的不同 patch 可能被路由到不同专家，导致内部不一致。

M2DINO 的路由决策发生在 **任务级别**：由一个 task token 决定激活哪些专家，整张图像的所有特征走同一批专家。这保证了任务内的一致性，路由也更稳定。

### 数学表达

设 $\mathbf{x} \in \mathbb{R}^{N \times D}$ 为图像特征序列，$\mathbf{t} \in \mathbb{R}^D$ 为任务嵌入：

$$\text{Router}(\mathbf{t}) = \text{Softmax}(W_r \mathbf{t}) \in \mathbb{R}^{E}$$

$$\text{TopK}(\text{Router}(\mathbf{t})) \rightarrow \{(i_1, w_1), \ldots, (i_k, w_k)\}$$

$$\text{MoE}(\mathbf{x}, \mathbf{t}) = \sum_{j=1}^{k} w_j \cdot \text{Expert}_{i_j}(\mathbf{x})$$

其中 $E$ 是专家总数，$k \ll E$ 是激活的专家数。权重 $w_j$ 在 top-k 选出后重归一化。

---

## 代码实现

### 任务条件化 MoE 模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskConditionedMoE(nn.Module):
    """路由决策由 task token 驱动，而非逐 token 路由"""

    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 路由器：从任务嵌入预测专家分配权重
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # 每个专家是一个独立 FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, task_token: torch.Tensor) -> torch.Tensor:
        """
        x:          图像特征序列 (B, N, D)
        task_token: 任务嵌入     (B, D)
        """
        B, N, D = x.shape
        router_probs = F.softmax(self.router(task_token), dim=-1)   # (B, E)
        top_k_w, top_k_idx = router_probs.topk(self.top_k, dim=-1)  # (B, k)
        top_k_w = top_k_w / top_k_w.sum(-1, keepdim=True)           # 重归一化

        output = torch.zeros_like(x)
        for k in range(self.top_k):
            weight = top_k_w[:, k].view(B, 1, 1)      # (B, 1, 1)
            for e_id in range(self.num_experts):
                mask = (top_k_idx[:, k] == e_id)       # (B,)
                if mask.any():
                    output[mask] += weight[mask] * self.experts[e_id](x[mask])
        return output
```

### 完整 M2DINO Transformer Block

```python
class M2DINOBlock(nn.Module):
    """标准 ViT Block，将 FFN 替换为 Task-conditioned MoE"""

    def __init__(self, d_model: int, n_heads: int, num_experts: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe   = TaskConditionedMoE(d_model, num_experts)

    def forward(self, x: torch.Tensor, task_token: torch.Tensor) -> torch.Tensor:
        # Self-attention：所有任务共享
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MoE FFN：任务条件化，专家自适应分配
        x = x + self.moe(self.norm2(x), task_token)
        return x


class TaskEmbedding(nn.Module):
    """为每个任务分配可学习的 task token"""

    def __init__(self, num_tasks: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(num_tasks, d_model)
        nn.init.normal_(self.embed.weight, std=d_model ** -0.5)  # 见坑1

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(task_ids)   # (B, D)
```

### 负迁移检测工具

论文没有提供、但工程实践中必须有的东西——在训练中持续监测任务间是否发生负迁移：

```python
class NegativeTransferMonitor:
    """比较联合训练 vs 任务专属训练的性能差距，delta < -threshold 时报警"""

    def __init__(self, tasks: list, threshold: float = 0.02):
        self.tasks = tasks
        self.threshold = threshold
        self.baseline: dict = {}   # 任务专属基线
        self.joint: dict    = {}   # 联合训练结果

    def record_baseline(self, task: str, metric: float):
        self.baseline[task] = metric

    def record_joint(self, task: str, metric: float):
        self.joint[task] = metric

    def report(self):
        print(f"{'Task':<20} {'Baseline':>10} {'Joint':>10} {'Delta':>10} {'Status':>14}")
        print("-" * 68)
        for task in self.tasks:
            if task not in self.baseline or task not in self.joint:
                continue
            delta  = self.joint[task] - self.baseline[task]
            status = "OK" if delta >= -self.threshold else "[NEG TRANSFER]"
            print(f"{task:<20} {self.baseline[task]:>10.4f} "
                  f"{self.joint[task]:>10.4f} {delta:>+10.4f} {status:>14}")
```

---

## 三个核心实验结论

论文对比了三种训练范式：**任务专属（Task-specific）**、**临床分组（Clinically-grouped）**、**全任务统一（All-task unified）**。

### 结论一：数据规模决定聚合是否有利

| 数据规模 | 临床分组训练 | 全任务统一训练 |
|--------|------------|--------------|
| 数据充足 | 通常有益 | 相当或略低 |
| 数据稀缺 | 可能严重有害（-10% 以上） | 相对稳健 |

**反直觉发现**：在数据稀缺时，全任务统一训练比临床分组更稳定。更多任务提供了隐式多样性正则化，有害梯度被稀释——类似 dropout 的效果。

### 结论二：任务类型决定脆弱程度

按对负迁移的敏感性排序（高 → 低）：

**分割** >> 分类 ≈ 回归 > 检测

分割需要精细的空间位置特征，这类特征高度任务专属，最难共享。分类和回归依赖全局语义特征，共享成本更低。

### 结论三：临床分类不等于特征空间分类

心脏分割和心脏分类在特征需求上可能差异巨大。同器官 ≠ 同特征空间。这是最重要的工程教训。

---

## 实现中的坑

**坑 1：Task token 初始化敏感性**

路由器早期如果随机分配专家，会破坏专家专业化学习：

```python
# 不好：默认随机初始化方差过大
self.embed = nn.Embedding(num_tasks, d_model)

# 更好：正态分布，方差与 d_model 挂钩（同 BERT/ViT 的做法）
nn.init.normal_(self.embed.weight, std=d_model ** -0.5)
```

**坑 2：专家负载不均衡**

Top-k 路由容易导致热门专家被反复激活，其余专家"饿死"，需要在 loss 中加入负载均衡项：

```python
def load_balancing_loss(router_probs: torch.Tensor) -> torch.Tensor:
    # 希望每个专家被激活的概率接近均匀分布
    mean_usage = router_probs.mean(dim=0)              # (E,)
    target = torch.ones_like(mean_usage) / mean_usage.numel()
    return F.kl_div(mean_usage.log(), target, reduction='sum')
```

**坑 3：多任务数据采样比例**

数据量大的任务会主导训练，压制小任务：

```python
import numpy as np

task_sizes = np.array([1000, 5000, 200, 8000])   # 各任务样本数
# 按平方根采样：平衡大小任务，避免直接按比例带来的过度倾斜
sampling_weights = np.sqrt(task_sizes)
sampling_weights /= sampling_weights.sum()
```

---

## 什么时候用 / 不用联合训练？

| 场景 | 推荐策略 | 理由 |
|------|---------|------|
| 数据充足，任务异质性低 | 临床分组或全任务统一 | 足够数据可克服负迁移 |
| 数据稀缺（任意一个任务） | 全任务统一 或 任务专属 | 避免临床分组的负迁移 |
| 核心任务是分割 | 谨慎聚合，单独验证 | 分割对负迁移最敏感 |
| 核心任务是回归或分类 | 大胆聚合 | 对聚合更鲁棒 |
| 各任务数据量差异 >10x | 避免临床分组 | 数据不平衡加剧负迁移 |
| 部署资源受限 | 全任务统一 | 单一模型，性能一致性更好 |

---

## 我的观点

这篇论文做的是**系统性诊断**，而不是提出全新算法。它的价值在于用规模化实验给出可操作的设计准则，这在大量只追求 SOTA 的论文中相对稀缺。

两个值得深思的地方：

第一，"全任务统一在低数据时更稳定"这个结论出乎意料。直觉上更多任务 = 更多干扰，结果反而相反。如果这个结论能被独立复现，它意味着在医疗 AI 领域，当数据稀缺时，**加入更多任务**是一种可行的正则化策略——这与传统多任务学习建议的"谨慎添加任务"相悖。

第二，这篇论文没有讨论推理效率。MoE 模块在超声床旁设备部署时，`num_experts` 和 `top_k` 的选择会直接影响推理延迟。实际部署者需要在聚合收益和推理开销之间自行权衡，论文没有给出这个方向的指导。

**核心外卖（takeaway）**：在规划多任务超声模型时，不要问"临床上哪些任务相关"，而要问"数据量是否支撑联合训练，以及有没有分割任务在里面"。

---

## 延伸阅读

- [原论文：arxiv.org/abs/2603.18123](https://arxiv.org/abs/2603.18123v1)
- **DINOv2**（Oquab et al., 2023）：M2DINO 的骨干基础，自监督 ViT 预训练
- **ST-MoE**（Zoph et al., 2022）：Google 提出的稳定化 MoE 训练方案，解决专家负载均衡问题