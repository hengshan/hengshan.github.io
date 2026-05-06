---
layout: post-wide
title: "RD-ViT：用循环深度 Transformer 打破医学分割的数据瓶颈"
date: 2026-05-06 12:06:01 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.03999v1
generated_by: Claude Code CLI
---

## 一句话总结

RD-ViT 用一个共享权重的 Transformer Block 循环 T 次代替 T 个独立 Block，在参数量减半的同时保持分割精度——在小数据集上甚至反超标准 ViT。

## 为什么这个问题重要？

医学图像分割（心脏 MRI、肿瘤 CT）是 3D 空间理解的核心任务，但数据标注贵、样本量小是行业常态。

**现有方法的痛点：**
- 标准 ViT-Base：12 层独立参数 ≈ 86M，没有大量标注数据根本收敛不了
- CNN 在 3D 体积数据上感受野受限，捕捉不到全局解剖结构
- 数据增强只能治标，根本问题是模型对每一层都从零学习参数

**RD-ViT 的三板斧：**
参数共享（减少学习负担）+ 自适应计算（难区域多想几步）+ MoE 专家（自发分工），共同解决数据稀缺问题。

## 背景知识

### 循环深度 vs 标准深度

标准 ViT 的 12 层就像 12 个独立工人，每人都要从零学习所有操作；RD-ViT 只雇一个工人，让他重复干 T 次，每次带着上一次的经验，但通过 **LoRA 适配器** 在不同深度有细微调整。

```
标准 ViT:  x → Block₁ → Block₂ → ... → Block₁₂ → y
                ↑独立参数↑  ↑独立参数↑       ↑独立参数↑

RD-ViT:    x → Block(t=1) → Block(t=2) → ... → Block(t=T) → y
                ↑___________ 共享权重 + LoRA_t ___________↑
```

参数量从 $O(T \cdot d^2)$ 降至 $O(d^2 + T \cdot r \cdot d)$，其中 $r \ll d$。

### LTI 状态注入：为什么权重共享不能裸用？

纯粹权重共享等价于把同一 Block 展开成 T 层，梯度通过 T 次乘法反传时会爆炸或消失——和 vanilla RNN 一样的老问题。

线性时不变（LTI）状态注入引入一个衰减记忆：

$$s_t = \lambda \cdot s_{t-1} + W_s \cdot h_{t-1}, \quad \lambda < 1$$

$\lambda < 1$ 保证状态序列有界，从而保证训练收敛。

## 核心方法

### 直觉解释：心脏边界需要更多"思考"

心脏 MRI 中，心肌（MYO）和血池（LV/RV）的边界模糊且形态变化大，是最难分割的区域。普通模型对所有像素分配同样的计算——浪费在简单背景上。

**ACT（自适应计算时间）** 让每个空间位置自主决定停止时机：
- 均匀背景：迭代 1-2 次即停
- 心脏边界：迭代 4-5 次才停

论文的 halting map 可视化完美印证了这一点：计算资源自动聚焦到解剖学难点。

### 关键公式

**深度 LoRA 适配：**

$$W_t = W_{\text{shared}} + A_t B_t^\top, \quad A_t \in \mathbb{R}^{d \times r},\; B_t \in \mathbb{R}^{d \times r}$$

每个迭代步有独立的低秩矩阵，以极小的参数开销赋予不同深度不同的"性格"。

**ACT 停机机制：**

$$\text{ponder}(x) = \min\!\left\{N : \sum_{t=1}^{N} p_t(x) \geq 1 - \epsilon\right\}$$

训练时加正则项 $\mathcal{L}_{\text{ponder}} = \beta \cdot \mathbb{E}[\text{ponder}(x)]$，鼓励模型越训越"懒"（高效）。论文报告全局平均 ponder time 从 2.6 降至 1.4。

**MoE 前馈层：**

$$\text{FFN}_{\text{MoE}}(x) = \sum_{k \in \text{TopK}} g_k(x) \cdot E_k(x)$$

### Pipeline 概览

```
输入（2D 切片 or 3D 体积）
    ↓ Patch Embedding + Positional Encoding
    h₀: [B, N, d]
    ┌──────────────────────────────────────────┐
    │  for t = 1 to T:                        │
    │    s_t = λ·s_{t-1} + W_s·h_{t-1}       │  ← LTI 状态
    │    h_t = SharedBlock(h_{t-1} + s_t,     │
    │                      LoRA_t, MoE_FFN)   │  ← 共享 + 适配
    │    p_t = sigmoid(W_halt · h_t)          │  ← 停机概率
    │    if ACT_halt(p_t): break              │
    └──────────────────────────────────────────┘
    ↓ ACT 加权聚合
    ↓ 分割头（展开 patch → 像素级预测）
输出分割图 [B, H, W, num_classes]
```

## 实现

### 核心 RecurrentBlock（含 LoRA 深度适配）

```python
import torch
import torch.nn as nn

class LoRAAdapter(nn.Module):
    """每个循环步独立的低秩适配器"""
    def __init__(self, d_model: int, rank: int = 8, num_steps: int = 6):
        super().__init__()
        # 初始化：A 小随机，B 全零 → 初始 ΔW = 0，不破坏预训练权重
        self.A = nn.Parameter(torch.randn(num_steps, d_model, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(num_steps, d_model, rank))

    def delta_W(self, step: int) -> torch.Tensor:
        return self.A[step] @ self.B[step].T  # [d, d]

class RecurrentBlock(nn.Module):
    """共享参数的 Transformer Block，LoRA 赋予深度差异化"""
    def __init__(self, d_model: int, num_heads: int, num_steps: int = 6, rank: int = 8):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn   = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(),
                                   nn.Linear(4*d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.lora  = LoRAAdapter(d_model, rank, num_steps)

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        # Self-attention（所有步共享权重）
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN + LoRA 残差（差异化当前深度的语义）
        delta_w  = self.lora.delta_W(step)       # [d, d]
        ffn_out  = self.ffn(x) + x @ delta_w.T  # LoRA 残差叠加
        return self.norm2(x + ffn_out)
```

### ACT 自适应计算时间

```python
class AdaptiveComputationTime(nn.Module):
    """每个空间 token 自主决定停止迭代的时机"""
    def __init__(self, d_model: int, epsilon: float = 0.01):
        super().__init__()
        self.halt_proj = nn.Linear(d_model, 1)
        self.epsilon   = epsilon

    def forward(self, hiddens: list) -> tuple:
        # hiddens: T 步隐状态，每个 [B, N, d]
        accum  = torch.zeros(*hiddens[0].shape[:2], 1, device=hiddens[0].device)
        output = torch.zeros_like(hiddens[0])

        for t, h in enumerate(hiddens):
            p = torch.sigmoid(self.halt_proj(h))    # [B, N, 1]，停机概率
            p = torch.min(p, 1.0 - accum) if t < len(hiddens)-1 else (1.0 - accum)
            output = output + p * h
            accum  = accum  + p

        ponder_cost = accum.mean()                   # 正则化项：惩罚多余的步数
        return output, ponder_cost
```

### Mixture-of-Experts FFN

```python
import torch.nn.functional as F

class MoEFFN(nn.Module):
    """混合专家 FFN，不同专家自发专精不同心脏结构"""
    def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(),
                          nn.Linear(4*d_model, d_model))
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.top_k  = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores  = self.router(x)                                  # [B, N, E]
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)                      # 归一化权重

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e, expert in enumerate(self.experts):
                mask = (indices[..., k] == e).unsqueeze(-1).float()
                out += mask * weights[..., k:k+1] * expert(x)
        return out
```

### 完整 RD-ViT 前向传播

```python
class RDViT(nn.Module):
    """循环深度 ViT，支持 2D 医学图像分割"""
    def __init__(self, img_size=224, patch_size=16, d_model=256,
                 num_heads=8, num_steps=6, num_classes=4):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(1, d_model, patch_size, stride=patch_size)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.state_proj  = nn.Linear(d_model, d_model)  # LTI 状态转换
        self.lti_lambda  = 0.9                           # 衰减系数，< 1 保稳定
        self.block       = RecurrentBlock(d_model, num_heads, num_steps)
        self.act         = AdaptiveComputationTime(d_model)
        self.seg_head    = nn.Linear(d_model, num_classes)
        self.num_steps   = num_steps

    def forward(self, x: torch.Tensor):
        # Patch embedding: [B,1,H,W] → [B,N,d]
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        state, hiddens = torch.zeros_like(h), []

        for t in range(self.num_steps):
            state = self.lti_lambda * state + self.state_proj(h)  # LTI 注入
            h     = self.block(h + state, step=t)
            hiddens.append(h)

        h, ponder_cost = self.act(hiddens)     # ACT 加权聚合
        logits = self.seg_head(h)              # [B, N, C]，需 reshape 回 H×W
        return logits, ponder_cost
```

### 训练损失（含 ACT 正则化）

```python
def rd_vit_loss(logits, targets, ponder_cost, beta=0.01):
    """
    logits:      [B, N, C] → 转置后输入 cross_entropy
    targets:     [B, N]
    ponder_cost: ACT 步数惩罚（鼓励模型少想几步）
    """
    seg_loss = F.cross_entropy(logits.transpose(1, 2), targets)
    return seg_loss + beta * ponder_cost
```

## 实验

### ACDC 心脏 MRI 数据集

ACDC（Automated Cardiac Diagnosis Challenge）是心脏 MRI 分割标准测试集：
- 100 例患者，4 类：背景、右心室（RV）、心肌（MYO）、左心室（LV）
- 支持 2D 切片和 3D 体积两种输入模式
- 数据格式：`.nii.gz`（NIfTI 3D 体积）

```bash
pip install torch torchvision nibabel einops
# ACDC 数据需在官网注册下载：creatis.insa-lyon.fr/Challenge/acdc/
```

### 定量结果

**2D 切片分割（Dice Score）**

| 方法 | 10% 标注数据 | 全量数据 | 参数量 |
|------|------------|---------|-------|
| 标准 ViT | 0.762 | 0.872 | ~5.7M |
| RD-ViT | **0.774** | **0.882** | ~2.8M |

**3D 体积分割（Dice Score）**

| 方法 | Dice | 参数量 | 相对性能 |
|------|------|--------|---------|
| 标准 ViT | 0.817 | ~5.7M | 100% |
| RD-ViT + MoE | 0.812 | 3.0M | **99.4%** |

核心结论：**53% 的参数量 → 99.4% 的性能**，小数据下直接反超。

### MoE 专家分工（涌现行为）

无需任何显式监督，4 个专家在训练中自发形成解剖学分工：

| 专家 | 主要激活区域 | 解剖学对应 |
|------|------------|----------|
| Expert 0 | 大面积均匀区域 | 背景 + RV 血池 |
| Expert 1 | 薄壁环状结构 | 心肌（MYO） |
| Expert 2 | 致密圆形区域 | 左心室（LV）内腔 |
| Expert 3 | 边缘过渡带 | 各结构交界处 |

这种涌现分工说明 MoE 确实在学习形态学差异，而非随机路由。

## 工程实践

### 实际部署注意事项

- **推理速度**：ACT 让每个 batch 的迭代步数不同，GPU 并行利用率下降，延迟不稳定。如果需要实时推理（如术中导航），建议固定步数（去掉 ACT）
- **硬件需求**：2D 版本在 Colab T4 可跑，3D 体积建议 16GB+ VRAM
- **显存控制**：3D 体积用 2D 逐切片 + 切片间融合，避免直接展 3D patch

### 常见坑

**坑1：ACT 的 beta 系数过大**

ponder_cost 正则化太强，模型学会"偷懒"，边界区域只迭代 1 次，精度崩溃。

```python
beta = 1.0   # 危险：模型会强行压缩步数
beta = 0.01  # 推荐：论文用值，平衡精度和效率
```

**坑2：LoRA rank 选择两难**

```python
rank = 2    # 太小：深度差异化失效，接近纯共享权重
rank = 64   # 太大：参数量骤增，失去参数高效的意义
rank = max(4, d_model // 32)  # 推荐：d=256 → rank=8
```

**坑3：Depth Extrapolation 时 LoRA 越界**

训练用 6 步，推理用 8 步时 `lora.A[step]` 越界：

```python
# 修复：推理时对步数取模或截断
lora_step = min(step, self.num_steps - 1)  # 超出范围复用最后一步
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 标注数据少（< 500 例患者） | 数据充足时标准 ViT 更容易调 |
| 医学图像（MRI/CT）分割 | 需要固定延迟的实时系统 |
| 参数预算严格（边缘部署） | 类别数极多时 MoE 专家数需相应扩大 |
| 结构边界精度要求高 | 动态场景（心脏搏动序列，需要时序建模） |

## 与其他方法对比

| 方法 | 核心思想 | 优点 | 缺点 |
|-----|---------|------|------|
| 标准 ViT | 独立层堆叠 | 容量大，灵活 | 数据饥渴 |
| Universal Transformer | 纯权重共享 | 参数少 | 无深度差异，无 ACT |
| **RD-ViT（本文）** | 共享+LoRA+ACT+MoE | 小数据强，计算自适应 | 推理延迟不稳定 |
| nnU-Net | CNN 自动配置 | 开箱即用，工程成熟 | 缺全局建模 |
| SwinUNETR | Swin ViT + UNet | 分层特征，3D 效果好 | 参数量大 |

## 我的观点

**真正有价值的地方：**

ACT 的空间自适应计算在医学图像上天然契合——解剖边界本就需要更多关注。MoE 的涌现专家分工实验是本文最有趣的发现：类别语义信息从 routing 机制中自发涌现，这对理解 Transformer 的特征组织方式有启发。

**离实用还有距离：**

1. **对比基线偏弱**：只和标准 ViT 比，没有对比 nnU-Net、MedSAM 等工程化更成熟的医学分割方法
2. **3D 实验数据少**：仅 ACDC 一个数据集，泛化性存疑
3. **ACT 的工程成本**：变长计算在 GPU 上难以批处理，实际部署吞吐量比论文数字低

**值得跟进的方向：**

- 权重共享 + LoRA 的思路能否迁移到医学图像 foundation model，用极少的 fine-tuning 数据适配新器官？
- ACT 能否扩展到 3D，为不同切片分配不同计算量（病灶切片 vs 正常切片）？
- MoE 的专家专化能否被显式监督强化，变成一种轻量级多任务学习框架？

在数据稀缺的医学分割场景（这几乎是常态），RD-ViT 的参数效率值得认真考量。论文代码已开源：[arxiv 2605.03999](https://arxiv.org/abs/2605.03999v1)。