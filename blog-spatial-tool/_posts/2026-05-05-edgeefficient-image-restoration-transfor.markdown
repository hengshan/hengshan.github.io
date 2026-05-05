---
layout: post-wide
title: "图像修复的效率革命：将 Transformer 蒸馏为 SSM 并在边缘端提速 3.4x"
date: 2026-05-05 12:05:41 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.02794v1
generated_by: Claude Code CLI
---

## 一句话总结

通过特征蒸馏训练 SSM（状态空间模型）块作为 Transformer 块的代理，再用多目标 ENS 搜索找到最优混合架构，图像去模糊在 Snapdragon 8 Elite 上从 10119 ms 降至 2973 ms（**3.4x 加速**），同时保持接近原始的修复质量。

---

## 为什么需要这个？

### 边缘端的 Transformer 困境

Restormer 是图像修复领域的强力模型，PSNR 指标优秀。但有一个绕不开的现实：

```
Restormer 在 Snapdragon 8 Elite CPU 上推理一张图：10119 ms ≈ 10 秒
```

即便 Restormer 使用了转置注意力（在通道维度而非空间维度做 attention），在移动端仍有多层瓶颈：

- **计算层面**：标准自注意力是 $O(N^2)$ 复杂度（N = 特征图像素数），转置注意力产生 $(H \times W) \times (H \times W)$ 的临时注意力图，严重压缩移动端内存
- **内存访问**：Softmax + QKV 乘法产生大量随机内存访问，对 ARM 的 L1/L2 缓存极不友好
- **NPU 支持受限**：移动端 NPU 对 Transformer 的 attention kernel 支持远不如卷积和矩阵乘法

### 为什么不直接用 SSM？

Mamba 等 SSM 提供线性复杂度 $O(N)$，边缘端确实更快。但问题是：**纯 SSM 在细粒度修复任务上精度不够**。

SSM 的归纳偏置是长程序列依赖，对局部边缘细节（去模糊的核心需求）的捕获弱于 Transformer。这就是本文的出发点：**让 SSM 学到 Transformer 对局部特征的建模能力。**

---

## 核心原理

### 直觉：让学生模仿老师的中间推理过程

传统知识蒸馏是对最终输出对齐。但这篇论文的关键是**特征蒸馏**：让 SSM 学生模仿 Transformer 教师在每层的**中间特征图**，而不只是最终输出。

这样，即使 SSM 的计算路径完全不同（递推状态 vs 全局注意力），它的"思维方式"——特征空间的表达——可以被对齐。

### 硬件层面：SSM 为什么在移动端更快？

状态空间模型的递推方程：

$$h_t = A h_{t-1} + B x_t, \quad y_t = C h_t$$

| 操作 | Transformer Attention | SSM（Mamba 风格） |
|------|----------------------|-----------------|
| 时间复杂度 | $O(N^2)$ | $O(N)$ |
| 内存访问模式 | 随机（attention map） | 顺序（状态递推） |
| 移动端缓存友好度 | 差 | 好（状态 $h$ 小且连续） |
| ARM SIMD 利用率 | 受限（softmax 瓶颈） | 高（向量乘加） |

### 三步框架

1. **对齐预训练**：对每个 Transformer 块，独立训练 SSM 代理块，通过特征蒸馏使输出特征尽量匹配
2. **ENS 搜索**：枚举 Transformer/SSM 不同混合比例，用多目标评分找到质量-速度最优配置
3. **端到端微调**：对搜索到的混合架构进行任务特定的微调

---

## 代码实现

### Baseline：Restormer 风格的 Transformer 块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransposedAttention(nn.Module):
    """Restormer 风格的转置注意力：在通道维度做 attention，避免 O(N^2) 空间复杂度"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = [t.reshape(B, self.num_heads, -1, H * W) for t in qkv]

        # 注意力在通道维度(C/heads)上，但仍产生 (H*W)^2 大小的临时张量
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)  # 移动端的 softmax 是主要瓶颈
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)

        return self.proj(out.reshape(B, C, H, W))
```

**性能分析**：对于 256×256 输入，`attn` 张量大小为 `[B, heads, H*W, H*W]` = 65536² 个元素，在移动端内存不足时会触发频繁的内存分配和交换。

---

### 核心优化：SSM 代理块

```python
class SSMImageBlock(nn.Module):
    """用于图像修复的 SSM 块，线性复杂度替代 Transformer 注意力"""
    def __init__(self, channels, d_state=16, expand=2):
        super().__init__()
        d_inner = channels * expand
        self.in_proj  = nn.Linear(channels, d_inner * 2)
        # 短程 1D 卷积捕获局部上下文（类 Mamba 的 depthwise conv）
        self.conv1d   = nn.Conv1d(d_inner, d_inner, 4, padding=3, groups=d_inner)
        # 选择性 SSM 参数投影
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1)  # B, C, dt
        self.A_log    = nn.Parameter(torch.randn(d_inner, d_state))
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, channels)
        self.norm     = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # 将 2D 特征图展平为序列（行优先扫描）
        x_seq = x.flatten(2).permute(0, 2, 1)       # [B, H*W, C]
        residual = x_seq
        xz = self.in_proj(self.norm(x_seq))          # [B, N, 2*d_inner]
        x_part, z = xz.chunk(2, dim=-1)

        # 局部卷积 + 选择性扫描（完整实现见 mamba-ssm 官方库）
        x_conv = self.conv1d(x_part.transpose(1,2))[:,:,:H*W].transpose(1,2)
        y = F.silu(x_conv) * F.silu(z)              # 门控输出

        return (self.out_proj(y) + residual).permute(0,2,1).reshape(B, C, H, W)
```

> **注意**：完整的 Mamba 选择性扫描依赖专用 CUDA 算子实现并行扫描。这里展示的是 PyTorch 简化版，实际部署建议使用 [mamba-ssm](https://github.com/state-spaces/mamba) 官方库。

---

### 特征蒸馏：让 SSM 学习 Transformer 的特征表达

论文的关键创新是**中间特征对齐**，而非输出对齐：

```python
class FeatureDistillationTrainer:
    def __init__(self, teacher_block, student_block, channels):
        self.teacher  = teacher_block.eval()  # Teacher 权重冻结
        self.student  = student_block
        # 对齐投影，应对 Teacher/Student 输出空间不同的情况
        self.align    = nn.Conv2d(channels, channels, 1)

    def distill_loss(self, x_input, alpha=0.8):
        """
        同一输入分别过 Teacher（Transformer）和 Student（SSM）
        计算中间特征的对齐损失
        """
        with torch.no_grad():
            feat_t = self.teacher(x_input)   # Teacher 特征，不参与梯度

        feat_s = self.student(x_input)       # Student 特征
        feat_aligned = self.align(feat_s)

        # L2 + Cosine 双重对齐，防止特征尺度偏移
        l2   = F.mse_loss(feat_aligned, feat_t)
        cos  = 1 - F.cosine_similarity(
            feat_aligned.flatten(1), feat_t.flatten(1)
        ).mean()

        return alpha * l2 + (1 - alpha) * cos
```

训练策略：每个 Transformer 块对应独立训练一个 SSM 代理块，蒸馏阶段只优化 Student 参数。

---

### ENS：用 Transformer 块数量作为延迟代理指标

ENS 的核心洞察：不需要在真实硬件上反复 profile，"Transformer 块数量"本身就是延迟的可靠代理。

```python
import numpy as np

def ens_search(block_configs, quality_scores, n_transformer_blocks,
               quality_threshold=0.95):
    """
    block_configs:       所有候选混合配置（如 ['T','S','S','T','S',...]）
    quality_scores:      每个配置的 PSNR/SSIM（由预训练代理块评估）
    n_transformer_blocks: 每个配置中 Transformer 块数量（延迟代理）
    """
    q = np.array(quality_scores)
    t = np.array(n_transformer_blocks)

    # 归一化后做多目标评分：最大化质量，最小化 Transformer 占比
    q_norm = (q - q.min()) / (q.ptp() + 1e-8)
    t_norm = t / (t.max() + 1e-8)
    ens_score = q_norm - t_norm

    # 过滤低于质量阈值的配置（保证最低精度要求）
    valid = q >= q.max() * quality_threshold
    ens_score[~valid] = -np.inf

    best = ens_score.argmax()
    return block_configs[best], {'quality': q[best], 'n_transformer': t[best]}
```

---

### 混合 U-Net 架构

```python
class HybridRestorationNet(nn.Module):
    """
    根据 ENS 搜索结果动态组合 Transformer/SSM 块
    block_types: ENS 返回的最优配置，如 ['T','S','S','T']
    """
    def __init__(self, in_ch=3, base_ch=48, block_types=None):
        super().__init__()
        block_types = block_types or ['T', 'S', 'S', 'T']
        self.head = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.blocks = nn.ModuleList([
            TransposedAttention(base_ch) if bt == 'T'
            else SSMImageBlock(base_ch)
            for bt in block_types
        ])

        self.tail = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        for block in self.blocks:
            feat = block(feat)
        return self.tail(feat) + x  # 全局残差：网络只学残差，收敛更快
```

---

## 性能实测

论文在 **Snapdragon 8 Elite CPU** 上的实测数据（输入约 256×256）：

| 模型 | 延迟 (ms) | 加速比 | PSNR 损失 |
|------|----------|--------|----------|
| Restormer（纯 Transformer） | 10119 | 1.0x | 基准 |
| ENS-Deblurring | 2973 | **3.4x** | < 0.1 dB |
| ENS-Deraining | 5816 | **1.74x** | < 0.05 dB |
| ENS-Denoising | 8666 | **1.17x** | < 0.03 dB |

**规律分析**：去模糊加速最大，因为去模糊任务中 SSM 对全局模糊场的建模能力足够；去噪对细粒度噪声分布更敏感，ENS 只能替换少量 Transformer 块，加速有限。

---

## 常见坑与调试技巧

### 坑 1：直接替换不做蒸馏，精度暴跌

```python
# 错误：直接换块，不做预对齐
model.blocks[2] = SSMImageBlock(channels=48)  # PSNR 可能掉 1-2 dB

# 正确：先蒸馏训练，再替换集成
trainer = FeatureDistillationTrainer(teacher_blocks[2], ssm_block, channels=48)
# 用训练集蒸馏若干步，再将 ssm_block 集成进整体模型
```

### 坑 2：SSM 单向扫描造成方向偏置

```python
# 单向扫描：右下角的像素看不到左上角的信息
out = ssm_scan(x_seq, direction='forward')

# 双向扫描消除偏置（类似 VMamba 的多方向扫描）
out = (ssm_scan(x_seq, 'forward') + ssm_scan(x_seq, 'backward')) / 2
```

### 坑 3：ENS 搜索空间过大

若模型有 24 层，每层可选 T 或 S，搜索空间是 $2^{24} \approx 1.6 \times 10^7$。缓解策略：
- 按 U-Net 阶段分组，同阶段共享类型选择
- 贪心逐层替换（从替换收益最大的块开始）
- 代理指标（Transformer 块数量）替代实测延迟，避免重复 profiling

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 移动端/边缘端部署（Snapdragon、Apple A 系列） | 数据中心 GPU（Transformer 有 FlashAttention 等高度优化） |
| 延迟敏感的实时图像处理 | 精度优先、延迟不限的离线批处理 |
| 有 Transformer baseline 模型需要迁移 | 从头训练轻量模型（直接训练纯 SSM 更简单） |
| 去模糊、去雨等全局相关性任务 | 超分辨率（对局部高频细节要求极高） |

---

## 局限性：诚实的评估

1. **SSM CUDA 算子在 ARM 上缺乏等效优化**：Mamba 的高效并行扫描依赖 CUDA，ARM CPU 需要专门的 NEON/SVE 实现，实际部署工程复杂度不低

2. **蒸馏训练增加工程成本**：需要维护 Teacher 模型，训练时间约为普通训练的 1.5-2x，且每个任务（去模糊/去雨/去噪）需要单独搜索

3. **加速比任务依赖性强**：去模糊 3.4x vs 去噪 1.17x，说明该方法的收益高度依赖任务特性，无法简单迁移

4. **ENS 代理指标在不同硬件上不一定准确**：若目标硬件有硬件加速的 Transformer 算子（如 Apple Neural Engine），以 Transformer 块数量作为延迟代理可能失效

---

## 延伸阅读

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — SSM 基础，理解选择性扫描的必读论文
- [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881) — 本文的 Transformer baseline
- [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.13260) — 将 Mamba 扩展到视觉任务，提出四方向扫描解决方向偏置问题
- 本文原论文：https://arxiv.org/abs/2605.02794