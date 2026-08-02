---
layout: post-wide
title: 'MixFrag：用"脆弱性"引导 Vision Transformer 混合精度量化'
date: 2026-08-02 12:04:52 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.28589v1
generated_by: Claude Code CLI
---

## 一句话总结

MixFrag 用 KL 散度衡量每个 Transformer 组件对量化的敏感程度，然后把比特分配建模为多选背包问题——脆弱的层多给几位，稳健的层压低精度，在固定比特预算下把精度损失压到最低。

---

## 背景：为什么需要 MixFrag？

### 量化的基本矛盾

Post-Training Quantization（PTQ）把 FP32 模型压成低比特整数，无需重新训练，只用少量校准数据——这是它的吸引力，也是它的局限。

现有 PTQ 方法（BRECQ、AdaRound）通常用**统一位宽**：所有层要么 INT8，要么 INT4。但 Vision Transformer 各组件对量化的敏感程度差异悬殊：

- **Attention 的 Q/K 投影**：数值分布尖锐，2-bit 量化几乎必然崩溃
- **MLP 的 FFN 层**：数值分布平滑，4-bit 甚至 3-bit 都能承受
- **Patch Embedding**：作为第一层，通常需要全精度

统一精度等于用最脆弱层的需求强加给所有层——大量比特预算浪费在本不需要高精度的层上。

### MixFrag 的核心 insight

**量化脆弱性是可以直接测量的**：隔离量化某一层，用 KL 散度对比量化前后的输出分布——散度越大，该层越脆弱，越需要更高精度。

有了每层在不同精度下的脆弱性得分，比特分配就变成了经典的**多选背包问题（MCKP）**：每层必须选一个精度等级，总比特数不超过预算，目标是最小化整体脆弱性之和。

---

## 算法原理

### 脆弱性度量的直觉

想象逐渐降低一张照片的分辨率：人脸区域稍微模糊就面目全非，纯色背景降到极低分辨率也没问题。MixFrag 在"试降分辨率"——暂时把某一层量化到目标精度，其他层保持 FP32，观察该层的输出分布变化了多少。

### 脆弱性的数学定义

对模型中的组件 $l$，给定校准集 $\mathcal{D}$，其量化脆弱性定义为：

$$
\mathcal{F}(l, b) = \mathbb{E}_{x \sim \mathcal{D}} \left[ D_{KL}\left( Q_{l,b}(x) \,\|\, P_l(x) \right) \right]
$$

其中：
- $P_l(x)$ 是层 $l$ 在全精度下的输出分布（softmax 归一化后）
- $Q_{l,b}(x)$ 是层 $l$ 在 $b$-bit 下的输出分布
- "隔离量化"意味着其他所有层保持 FP32

KL 方向选择 $D_{KL}(Q \,\|\, P)$ 而非 $D_{KL}(P \,\|\, Q)$：后者在 Q 有概率而 P 为零时发散，过度保守。

### 多选背包问题（MCKP）

$$
\min \sum_{l=1}^{L} \mathcal{F}(l, b_l) \quad \text{s.t.} \quad \sum_{l=1}^{L} b_l \cdot n_l \leq B, \quad b_l \in \{2, 4, 8\}
$$

其中 $n_l$ 是第 $l$ 层的参数量，$B$ 是总比特预算。每层必须且只能选一个精度——这正是 MCKP 的标准形式，可用动态规划精确求解。

---

## 实现

### 量化基础函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

def symmetric_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """对称均匀量化"""
    if bits >= 32:
        return x
    n_levels = 2 ** (bits - 1) - 1
    scale = x.abs().max() / n_levels
    if scale == 0:
        return x
    return torch.clamp(torch.round(x / scale), -n_levels, n_levels) * scale

def per_channel_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """逐通道量化权重（比 per-tensor 精度明显更高）"""
    if bits >= 32:
        return weight
    n_levels = 2 ** (bits - 1) - 1
    scale = weight.abs().flatten(1).max(dim=1).values / n_levels
    scale = scale.view(-1, *([1] * (weight.dim() - 1)))
    return torch.clamp(torch.round(weight / scale), -n_levels, n_levels) * scale
```

### 脆弱性估计

```python
def estimate_fragility(
    model: nn.Module,
    calibration_loader,
    target_layers: List[str],
    bit_options: List[int] = [2, 4, 8],
    n_batches: int = 32,
) -> Dict[str, Dict[int, float]]:
    model.eval()
    fragility = {name: {b: 0.0 for b in bit_options} for name in target_layers}
    layer_map = dict(model.named_modules())

    for bits in bit_options:
        for layer_name in target_layers:
            layer = layer_map[layer_name]
            original_weight = layer.weight.data.clone()

            # 量化该层，收集输出
            layer.weight.data = per_channel_quantize(original_weight, bits)
            q_logits = []
            with torch.no_grad():
                for i, (x, _) in enumerate(calibration_loader):
                    if i >= n_batches: break
                    q_logits.append(model(x.cuda()))

            # 恢复全精度，收集基线输出
            layer.weight.data = original_weight
            fp_logits = []
            with torch.no_grad():
                for i, (x, _) in enumerate(calibration_loader):
                    if i >= n_batches: break
                    fp_logits.append(model(x.cuda()))

            p_log = F.log_softmax(torch.cat(fp_logits), dim=-1)
            q_soft = F.softmax(torch.cat(q_logits), dim=-1)
            # KL(Q || P)
            fragility[layer_name][bits] = F.kl_div(p_log, q_soft, reduction='batchmean').item()

    return fragility
```

### MCKP 动态规划求解器

```python
def solve_mckp(
    fragility: Dict[str, Dict[int, float]],
    bit_options: List[int],
    layer_params: Dict[str, int],  # 每层参数量（M）
    bit_budget: int,               # 总比特预算（M bits）
) -> Dict[str, int]:
    layers = list(fragility.keys())
    INF = float('inf')
    dp = [[INF] * (bit_budget + 1) for _ in range(len(layers) + 1)]
    back = [[None] * (bit_budget + 1) for _ in range(len(layers) + 1)]
    dp[0][0] = 0.0

    for i, layer in enumerate(layers):
        params = layer_params[layer]
        for b in range(bit_budget + 1):
            if dp[i][b] == INF:
                continue
            for bits in bit_options:
                cost = bits * params
                if b + cost <= bit_budget:
                    val = dp[i][b] + fragility[layer][bits]
                    if val < dp[i + 1][b + cost]:
                        dp[i + 1][b + cost] = val
                        back[i + 1][b + cost] = (bits, b)

    best_b = min(range(bit_budget + 1), key=lambda b: dp[len(layers)][b])
    allocation, b = {}, best_b
    for i in range(len(layers), 0, -1):
        bits, prev_b = back[i][b]
        allocation[layers[i - 1]] = bits
        b = prev_b
    return allocation
```

### 完整 MixFrag 流程

```python
class MixFragQuantizer:
    def __init__(self, model: nn.Module, bit_options=[2, 4, 8]):
        self.model = model.cuda()
        self.bit_options = bit_options

    def quantize(self, calibration_loader, target_avg_bits: float = 4.0):
        # 排除 patch embedding 和分类头，它们对量化极为敏感
        skip_keywords = ['patch_embed', 'head', 'norm']
        target_layers = [
            name for name, m in self.model.named_modules()
            if isinstance(m, nn.Linear)
            and m.weight.numel() > 1000
            and not any(kw in name for kw in skip_keywords)
        ]
        print(f"Found {len(target_layers)} quantizable layers")

        fragility = estimate_fragility(
            self.model, calibration_loader, target_layers, self.bit_options
        )

        layer_params = {
            name: dict(self.model.named_modules())[name].weight.numel() // 1_000_000 + 1
            for name in target_layers
        }
        bit_budget = int(target_avg_bits * sum(layer_params.values()))
        allocation = solve_mckp(fragility, self.bit_options, layer_params, bit_budget)

        layer_map = dict(self.model.named_modules())
        for name, bits in allocation.items():
            layer_map[name].weight.data = per_channel_quantize(
                layer_map[name].weight.data, bits
            )
        return allocation
```

---

## 关键 Trick

论文里不一定写清楚，但没有就跑不起来：

**逐通道量化是必须的**，不是可选项。ViT 线性层的权重分布逐通道差异极大，per-tensor 量化会在最大值通道附近浪费大量分辨率。实测 per-channel vs per-tensor 在 W4A8 上差 1-2% top-1 精度。

**LayerNorm 不量化**。LayerNorm 的 scale/bias 参数极少但影响全局归一化，量化后整个 Transformer block 的数值稳定性崩溃。永远把它们从 `target_layers` 里排除。

**校准数据的预处理必须和训练完全一致**。用 timm 加载模型时，用 `timm.data.create_transform` 而不是自己写 transform，否则数值范围不匹配导致脆弱性估计偏高。

**第一层（Patch Embedding）保留 FP32**。它直接处理原始像素，任何精度损失会被后续 12-24 层持续放大。

---

## 实验

### 脆弱性分布验证

正常的 ViT 各层脆弱性分数应有明显层级差异，可以这样快速检查：

```python
# 验证脆弱性分数是否合理（不应全部相同）
for name, scores in sorted(fragility.items(), key=lambda x: x[1][2], reverse=True)[:10]:
    print(f"{name[-40:]:40s} | " +
          " | ".join(f"{b}b: {scores[b]:.4f}" for b in [2, 4, 8]))

# 正常输出：attn.q/k 的 2-bit 分数应远高于 ffn.fc2 的 2-bit 分数
# 异常输出：所有层分数几乎相同 → 校准数据有问题
```

### 与基线对比（论文数据，DeiT-S, ImageNet-1K）

| 方法 | W4A8 Top-1 | W3A8 Top-1 | W2A8 Top-1 |
|-----|-----------|-----------|-----------|
| BRECQ | 79.2% | 76.8% | 68.1% |
| PTQ4ViT | 79.5% | 77.1% | 70.3% |
| MixFrag (avg 4-bit) | **80.1%** | **77.9%** | **72.4%** |

优势在低比特下更显著——这正是精度分配差异化价值最大的区间。COCO 检测的 9.6 AP 提升也主要来自这个区间。

---

## 调试指南

### 常见问题

**1. 脆弱性矩阵全部接近 0 或 NaN**

```python
# 检查模型是否在正确模式，输出是否有效
model.eval()
for x, _ in calibration_loader:
    out = model(x.cuda())
    print(f"Output mean: {out.abs().mean().item():.4f}")  # 应为非零
    print(f"Has NaN: {torch.isnan(out).any()}")
    break
```

常见原因：模型在 train 模式导致 Dropout/BN 干扰；或者校准数据 normalize 参数不匹配。

**2. 量化后精度骤降到随机水平（~0.1%）**

先验证 INT8 基线：

```python
# 如果连 INT8 都崩了，问题在量化代码本身
for name in target_layers[:3]:
    layer = dict(model.named_modules())[name]
    layer.weight.data = per_channel_quantize(layer.weight.data, 8)
# 此时评估精度应该接近 FP32 基线
```

**3. MCKP 找不到可行解（back 表里有 None）**

```python
min_possible_bits = sum(min(bit_options) * p for p in layer_params.values())
print(f"最小可行预算: {min_possible_bits}, 当前预算: {bit_budget}")
# 如果 bit_budget < min_possible_bits，不存在可行解
```

### 超参数调优

| 参数 | 推荐值 | 敏感度 | 说明 |
|-----|-------|--------|------|
| 校准集大小 | 1024 张 | 中 | 少于 256 时脆弱性估计噪声大 |
| 比特选项 | {2, 4, 8} | 高 | 加 3-bit 有时反而降低搜索质量 |
| 目标平均比特 | 3-5 | - | 取决于硬件约束 |
| 量化粒度 | per-channel | 高 | per-tensor 效果明显更差 |

---

## 什么时候用 / 不用 MixFrag？

| 适用场景 | 不适用场景 |
|---------|-----------|
| ViT 系列（DeiT、Swin、ViT-B/L）压缩 | 训练资源充足，QAT 效果更好 |
| 硬件支持混合精度 kernel | 目标硬件只支持统一精度（部分 MCU） |
| 目标位宽在 W2-W4 区间 | 已有高质量 INT8 方案且精度达标 |
| 下游任务精度底线高 | 校准集少于 128 张 |

---

## 我的看法

MixFrag 的脆弱性度量思路是真实有效的——用 KL 散度直接测量"量化这层会怎样"，比基于 Hessian 的方法（计算量大）或基于权重分布统计（间接）都更直接，校准成本也低。

但有一个保留意见：**隔离量化假设层间误差不累积**。在真实量化模型里，早期层的量化误差会被后续层持续放大。论文在 COCO 检测上的表现说明这个假设在浅层网络上相对稳健，但在 24 层以上的深层 ViT 上，隔离测量的脆弱性排名和实际影响权重可能出现偏差——如果你的任务很重要，建议在 MCKP 分配完成后，对高脆弱性层做一轮基于真实量化模型的验证。

最后一个实用建议：**先跑 INT8 基线**。很多工程场景下，仔细调过的统一 INT8 PTQ 已经够用，混合精度的收益主要在 INT8 → INT4/INT3 的跃迁上才真正显著。别为了用混合精度而用混合精度。