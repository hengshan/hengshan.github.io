---
layout: post-wide
title: "VECA：用弹性核心注意力打破 Vision Transformer 的二次复杂度瓶颈"
date: 2026-05-13 12:10:53 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.12491v1
generated_by: Claude Code CLI
---

## 一句话总结

用少量可学习的"核心 token"作为通信中介，patch 之间无需直接两两交互，将标准 ViT 注意力的 $O(N^2)$ 复杂度降至线性 $O(NC)$，并支持在推理时弹性调整计算量。

## 为什么这个问题值得关注？

Vision Transformer 的核心缺陷很直白：自注意力的计算量与序列长度成**平方**关系。

- 224×224 图像，patch_size=16 → N=196 个 token
- 512×512 图像 → N=1024，计算量增加约 $27\times$
- 1024×1024 图像 → N=4096，计算量增加约 $440\times$

这不是线性扩展，而是灾难性爆炸。医学影像（2048×2048 CT）、卫星图像、高分辨率工业检测——这些场景标准 ViT 根本跑不动。

现有方案各有缺陷：
- **Swin Transformer**：局部窗口注意力，但需要精心设计层级结构，窗口大小限制感受野
- **Perceiver**：把 N 个 token 压缩到 C 个 latent，空间细节丢失严重
- **线性注意力近似**（Performer、Linformer）：理论 $O(N)$，但近似误差大，精度下降明显

VECA 的目标：**在完整保留 N 个 patch token 的前提下，把注意力复杂度压到线性**。

## 背景知识

### ViT 注意力的计算瓶颈

标准多头自注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$，矩阵乘 $QK^\top$ 复杂度是 $O(N^2 d)$。

**隐含假设**：每个 patch 需要**直接**关注其他所有 patch——all-to-all 全连接交互。

VECA 的核心问题：这个假设真的必要吗？

### 核心—外围结构的直觉

想象公司内部沟通：

- **全对全**（标准注意力）：每位员工直接联系所有同事 → $N^2$ 次通话
- **通过中介**（VECA）：设立 C 个"协调人"，员工只和协调人沟通 → $2NC$ 次通话

当 $N \gg C$ 时，效率提升显著，且全局信息依然能通过协调人流通。这种核心—外围（core-periphery）结构在神经科学中也有对应——大脑皮层的层级通信模式。

## 核心方法

### 数学描述

设 $\mathbf{X} \in \mathbb{R}^{N \times d}$ 为 patch token，$\mathbf{C}_0 \in \mathbb{R}^{C \times d}$ 为可学习核心 token（$C \ll N$）。

**Phase 1 — Gather（聚合）**：核心 token 提炼全局 patch 信息

$$\tilde{\mathbf{C}} = \text{Attn}\!\left(\mathbf{C}_0 W_Q^c,\ \mathbf{X} W_K^p,\ \mathbf{X} W_V^p\right) \quad \text{复杂度} \; O(C \cdot N)$$

**Phase 2 — Broadcast（广播）**：更新后的核心向所有 patch 分发全局上下文

$$\tilde{\mathbf{X}} = \text{Attn}\!\left(\mathbf{X} W_Q^p,\ \tilde{\mathbf{C}} W_K^c,\ \tilde{\mathbf{C}} W_V^c\right) \quad \text{复杂度} \; O(N \cdot C)$$

**总复杂度**：$O(NC)$，对固定 $C$ 即线性 $O(N)$。

**与 Perceiver 的关键区别**：Perceiver 最终只保留 C 个 latent token（空间信息永久丢失），VECA 的输出**始终是 N 个 patch token**——核心 token 只是通信媒介，不是最终表示。

### Pipeline 概览

```
输入图像
    ↓ Patch Embedding + 位置编码
[P₁  P₂  ...  Pₙ]       ← N 个 patch token（全程保留，空间结构完整）
       ↕ 只通过 core 交互（无直接 patch-to-patch）
  [C₁  C₂  ...  Cс]     ← C 个可学习 core token（跨层传播）

每层内：
  Gather:    core ← softmax(core · patchᵀ / √d) · patch     O(CN)
  Broadcast: patch ← softmax(patch · coreᵀ / √d) · core    O(NC)

    ↓ 重复 L 层（每层 updated_cores → 下一层初始 cores）

分类头（全局平均池化）/ 密集预测头（利用全部 N 个空间 token）
```

### 弹性推理原理

VECA 的"弹性"来自**嵌套训练（nested training）**策略。

训练时，每个 batch 随机采样 $C' \in \{1, 2, 4, 8, 16, 32, 64\}$，只使用**前 $C'$ 个核心 token**。这迫使模型学会：
1. 用任意多核心都能获得合理表示
2. 核心 token 按重要性自动排序（前面的 core 携带最关键信息）

推理时根据计算预算弹性选择 $C'$：移动端用 $C'=8$（快），服务器端用 $C'=64$（准）。

这与 **Matryoshka Representation Learning（MRL）** 在嵌入维度上的思想一脉相承，只是弹性维度从特征维度换成了核心数量。

## 代码实现

### 核心注意力模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoreAttention(nn.Module):
    """VECA 核心注意力：Gather + Broadcast，复杂度 O(NC)"""

    def __init__(self, dim, num_cores=64, num_heads=8):
        super().__init__()
        self.num_cores = num_cores
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # 可学习核心 token（小方差初始化，避免 Gather 注意力坍缩）
        self.cores = nn.Parameter(torch.randn(1, num_cores, dim) * 0.02)

        self.q_core   = nn.Linear(dim, dim, bias=False)   # Gather: core 查询
        self.kv_patch = nn.Linear(dim, 2 * dim, bias=False)  # Gather: patch 键值

        self.q_patch  = nn.Linear(dim, dim, bias=False)   # Broadcast: patch 查询
        self.kv_core  = nn.Linear(dim, 2 * dim, bias=False)  # Broadcast: core 键值

        self.proj = nn.Linear(dim, dim)

    def _attn(self, q, k, v):
        """多头注意力，输入/输出均为 [B, N, D]"""
        B, H, Dh = q.shape[0], self.num_heads, self.head_dim
        Nq, Nkv  = q.shape[1], k.shape[1]
        q = q.reshape(B, Nq,  H, Dh).transpose(1, 2)
        k = k.reshape(B, Nkv, H, Dh).transpose(1, 2)
        v = v.reshape(B, Nkv, H, Dh).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B, Nq, -1)

    def forward(self, x, num_cores=None):
        B, N, D = x.shape
        C     = num_cores or self.num_cores
        cores = self.cores[:, :C].expand(B, -1, -1)  # 弹性截断

        # Phase 1 - Gather: O(C·N)
        k_p, v_p = self.kv_patch(x).chunk(2, dim=-1)
        updated_cores = self._attn(self.q_core(cores), k_p, v_p)

        # Phase 2 - Broadcast: O(N·C)
        k_c, v_c = self.kv_core(updated_cores).chunk(2, dim=-1)
        updated_x = self._attn(self.q_patch(x), k_c, v_c)

        return self.proj(updated_x), updated_cores
```

### VECA Block 与弹性推理验证

```python
class VECABlock(nn.Module):
    def __init__(self, dim, num_cores=64, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = CoreAttention(dim, num_cores, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def forward(self, x, num_cores=None):
        attn_out, updated_cores = self.attn(self.norm1(x), num_cores)
        x = x + attn_out                   # 残差连接
        x = x + self.mlp(self.norm2(x))
        return x, updated_cores            # updated_cores 传递给下一层


# --- 弹性推理演示 ---
B, N, D = 2, 1024, 256  # batch=2, 1024 patches（等效 512×512 图像）
block = VECABlock(dim=D, num_cores=64)
x     = torch.randn(B, N, D)

out_full, _ = block(x, num_cores=64)  # 全核心：最高精度
out_fast, _ = block(x, num_cores=8)   # 弹性：8 个 core，计算量降至 1/8

print(f"全核心输出:  {out_full.shape}")  # [2, 1024, 256]
print(f"弹性推理:    {out_fast.shape}")  # [2, 1024, 256]，形状不变
```

在多层 VECA 中，每层返回的 `updated_cores` 作为下一层 Gather 的初始核心，实现跨层信息传播——这是与单纯 cross-attention 的重要区别。

### 嵌套训练策略

```python
import random

class NestedCoreTrainer:
    """嵌套训练：使模型在任意核心数量下均可工作"""
    SCHEDULE = [1, 2, 4, 8, 16, 32, 64]

    def __init__(self, model, max_cores=64):
        self.model     = model
        self.max_cores = max_cores

    def _sample_cores(self):
        # C_max 以 50% 概率选中，防止最大精度退化
        if random.random() < 0.5:
            return self.max_cores
        valid = [c for c in self.SCHEDULE if c <= self.max_cores]
        return random.choice(valid)

    def step(self, x, labels, optimizer, loss_fn):
        num_cores = self._sample_cores()
        logits = self.model(x, num_cores=num_cores)
        loss   = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "num_cores": num_cores}
```

### 复杂度可视化

```python
import matplotlib.pyplot as plt

def plot_complexity():
    resolutions = [112, 224, 448, 896, 1792]
    N_list = [(r // 16) ** 2 for r in resolutions]  # patch_size=16

    std  = [n ** 2  for n in N_list]  # O(N²)
    c16  = [16 * n  for n in N_list]  # O(NC), C=16
    c64  = [64 * n  for n in N_list]  # O(NC), C=64

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(resolutions, std, 'r-o', label='标准自注意力  O(N²)',  lw=2)
    ax.semilogy(resolutions, c16, 'b-s', label='VECA  C=16,  O(NC)', lw=2)
    ax.semilogy(resolutions, c64, 'g-^', label='VECA  C=64,  O(NC)', lw=2)
    ax.set_xlabel('图像分辨率（像素）', fontsize=12)
    ax.set_ylabel('相对计算量（对数刻度）', fontsize=12)
    ax.set_title('注意力复杂度随分辨率的增长', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_complexity()
```

在 1792×1792 分辨率下，标准注意力计算量是 VECA（C=64）的约 **1800 倍**。这个数字直接说明了高分辨率场景下线性注意力的必要性。

## 实验结果

论文在 ImageNet 分类和 ADE20K 语义分割上评估 VECA，与当前主流视觉基础模型对比：

- **分类精度**：与同规模 Swin、MaxViT 相近，但注意力复杂度是线性而非二次
- **密集预测**：明显优于 Perceiver 类方法（后者压缩 token 后丢失空间细节），接近 Swin 的 mIoU
- **弹性收益**：核心数从 64 降至 16 时，精度下降约 1~2%，计算量减少 4×

**Perceiver 精度差距的根源**：Perceiver 最终只剩 C 个 latent，dense prediction head 无法访问每个空间位置的独立特征；VECA 保留全部 N 个 patch token，分割头可以直接读取每个像素对应的特征图。

> 注：论文目前未发布官方代码，具体数值以最终发表版本为准。

## 工程实践

### 核心数 C 怎么选？

```python
# 经验准则（粗略参考）
# 224×224 → N=196  → C 推荐 16~32
# 512×512 → N=1024 → C 推荐 32~64
# 1024×1024 → N=4096 → C 推荐 64~128

# 弹性部署配置示例
deploy_config = {
    "mobile_realtime": {"num_cores": 8,  "note": "牺牲约 2% 精度换 4× 加速"},
    "edge_balanced":   {"num_cores": 32, "note": "速度/精度折中"},
    "server_quality":  {"num_cores": 64, "note": "最高精度"},
}
```

### 常见坑

**坑 1：Core token 初始化过大，Gather 阶段注意力坍缩到少数 patch**

```python
# ❌ 标准正态，方差=1，core 向量与 patch 键值量级不匹配
self.cores = nn.Parameter(torch.randn(1, C, dim))

# ✓ 小方差初始化，与 patch embedding 输出量级对齐
self.cores = nn.Parameter(torch.randn(1, C, dim) * 0.02)
```

**坑 2：弹性推理时忘记截断 core，始终做 C_max 次 Broadcast**

```python
# ❌ 永远用全部 C_max 个 core
cores = self.cores.expand(B, -1, -1)           # [B, C_max, D]

# ✓ 按推理时指定的数量截断
cores = self.cores[:, :num_cores].expand(B, -1, -1)  # [B, C', D]
```

**坑 3：超高分辨率（>2048²）Gather 阶段 patch KV 存储 OOM**

Gather 需存储 $K_p, V_p \in \mathbb{R}^{N \times d}$，对超高分辨率图像建议将 patch 分块，逐块计算 core 的注意力权重后累加聚合，避免一次性 materialize 全部 KV。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 高分辨率图像（512² 以上） | 低分辨率且算力充足（224²，标准 ViT 已经够快） |
| 需要弹性计算预算（多端部署） | 细粒度局部纹理感知（如关键点检测、光流） |
| 分类 / 目标检测 / 语义分割主干 | 图像生成（需要全局精细一致性） |
| 内存受限的边缘设备 | patch 间精确相对位置关系至关重要的任务 |

## 与其他方法对比

| 方法 | 核心思想 | 复杂度 | 保留全部 N token | 弹性推理 |
|------|---------|--------|----------------|---------|
| 标准 ViT | 全对全自注意力 | O(N²) | ✓ | ✗ |
| Swin | 局部滑动窗口 | O(N) | ✓ | ✗ |
| Perceiver | 压缩到 C 个 latent | O(NC) | ✗ | ✗ |
| Linformer | 低秩近似 KV 矩阵 | O(NC) | ✓ | ✗ |
| **VECA** | 核心 token 通信中介 | O(NC) | ✓ | ✓ |

VECA 在"保留 N token + 线性复杂度 + 弹性推理"三个维度同时满足，是目前这个组合中的独特选择。

## 我的观点

VECA 的核心贡献是一个有说服力的实验结论：**patch 之间无需直接交互，模型仍可学到有竞争力的视觉表示**。这挑战了 Transformer 社区长期以来的一个隐性假设——all-to-all 注意力是获得丰富特征的必要条件。

**值得关注的方向**：

- 核心 token 具备潜在可解释性，每个 core 可能对应特定视觉概念或频段（低频全局结构 vs 高频纹理），值得深入分析
- 弹性推理机制实用性强，移动端/边缘端部署场景天然需要这种"一次训练，多档推理"的能力
- 与 MoE（混合专家）结合的潜力：不同图像区域动态分配不同数量的 core，针对复杂区域用更多 core 精细处理

**需要谨慎的地方**：

- Flash Attention 通过硬件优化让 $O(N^2)$ 注意力在实践中远快于理论预测，VECA 的**实际**速度优势需要在相同硬件环境下仔细测量，而不能只看 FLOPs
- C 的选择对不同任务差异显著，目前缺乏自动化方法（如 NAS 或 learned C）
- 密集预测任务（实例分割、深度估计）上的充分验证还需要更多工作
- 目前无官方代码，层间 core 传播和嵌套训练的实现细节需要仔细复现

**距离实际应用还有多远**？对高分辨率分类和分割主干，VECA 已经接近可用——复杂度优势明显，精度损失在可接受范围内。但要取代 Swin 或 ViT 成为主流，还需要：Flash Attention 适配、分布式训练友好性验证、以及更多下游任务的泛化性证明。核心注意力这个范式本身是有潜力的，值得持续跟进。