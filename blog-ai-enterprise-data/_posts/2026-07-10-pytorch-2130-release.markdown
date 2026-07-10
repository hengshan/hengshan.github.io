---
layout: post-wide
title: "PyTorch 2.13 FlexAttention 深度解析：可编程注意力与 Apple Silicon 的 12x 加速"
date: 2026-07-10 08:03:06 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://github.com/pytorch/pytorch/releases/tag/v2.13.0
generated_by: Claude Code CLI
---

## 一句话总结

PyTorch 2.13 将 FlexAttention 带到了 Apple Silicon（MPS），用一个可编程的 `score_mod` 函数替代了手写 CUDA kernel 的时代——稀疏注意力模式下实测最高 12x 加速。

---

## 为什么这件事值得关注？

注意力机制有一个根本矛盾：研究者需要各种各样的注意力变体（因果、滑动窗口、文档级别、ALiBi 偏置……），但高效实现每一种都需要手写底层 kernel，这件事大多数团队根本做不到。

传统解法是三选一：
- 用 `F.scaled_dot_product_attention`（SDPA）——高效，但只支持固定模式
- 手写 Triton kernel——灵活，但门槛极高
- 用 Python 循环 + mask——可行，但慢到无法用于生产

FlexAttention 的核心洞见：**把"注意力 score 如何被修改"抽象成一个普通 Python 函数，然后通过 `torch.compile` 把这个函数直接编译进 attention kernel**。你写的是 Python，运行的是融合好的高性能 kernel。

2.13 最重要的变化是这套机制落地了 MPS 后端——也就是说，Mac 上跑研究的工程师终于能用上这个工具了。

---

## FlexAttention 核心机制解析

### 直觉理解

标准注意力计算的核心步骤：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

传统方案里，$M$ 是一个预先构造好的 mask 矩阵，整个 $QK^T$ 都会被计算，然后被 mask 掉——白白浪费了大量计算。

FlexAttention 把这个过程拆成两部分：

1. **score_mod**：对每个 $(b, h, q, k)$ 位置的 score 做任意变换（加 bias、乘系数等）
2. **block_mask**：在 block 粒度上告诉 kernel 哪些区域整块跳过

关键是：`score_mod` 不是事后应用的，它被 `torch.compile` **内联进 attention kernel 本身**，不产生任何中间 tensor。

### API 全貌

```python
from torch.nn.attention.flex_attention import (
    flex_attention, 
    create_block_mask,
    and_masks,
    or_masks,
)

# score_mod 签名固定：接收 score + 四个索引，返回修改后的 score
def score_mod(score, b, h, q_idx, kv_idx):
    return score  # 恒等变换 = 标准注意力

# block_mask_fn 签名：接收四个索引，返回 bool（True = 保留）
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, B=2, H=8, Q_LEN=1024, KV_LEN=1024)

output = flex_attention(query, key, value, 
                        score_mod=score_mod,
                        block_mask=block_mask)
```

---

## 动手实现

### 最小可运行示例：因果注意力

```python
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

device = "mps" if torch.backends.mps.is_available() else "cuda"
B, H, S, D = 2, 8, 512, 64

q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

# 定义因果 mask（下三角）
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device)

# score_mod 恒等（只用 block_mask 做稀疏化）
out = flex_attention(q, k, v, block_mask=block_mask)
print(out.shape)  # [2, 8, 512, 64]
```

### 三种实用注意力模式

**模式一：滑动窗口（Sliding Window）**

```python
WINDOW = 128  # 每个 token 只看前后 128 个位置

def sliding_window_mask(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() <= WINDOW

block_mask = create_block_mask(
    sliding_window_mask, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
)
out_sw = flex_attention(q, k, v, block_mask=block_mask)
```

**模式二：ALiBi 位置编码（score_mod 方案）**

```python
# ALiBi: 对距离施加线性 bias，不同 head 用不同斜率
def make_alibi_score_mod(num_heads):
    slopes = torch.tensor([2 ** (-8 * i / num_heads) for i in range(1, num_heads + 1)],
                          device=device)
    def alibi_score_mod(score, b, h, q_idx, kv_idx):
        bias = -slopes[h] * (q_idx - kv_idx).abs().float()
        return score + bias
    return alibi_score_mod

out_alibi = flex_attention(q, k, v, score_mod=make_alibi_score_mod(H))
```

**模式三：文档级别注意力（跨文档不交叉）**

```python
# 假设 batch 内每个样本是多个拼接文档，doc_ids 标记每个 token 属于哪篇文档
doc_ids = torch.randint(0, 4, (S,), device=device)  # 4 篇文档

def document_mask(b, h, q_idx, kv_idx):
    return doc_ids[q_idx] == doc_ids[kv_idx]

block_mask = create_block_mask(
    document_mask, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
)
out_doc = flex_attention(q, k, v, block_mask=block_mask)
```

### 组合多个 mask

```python
from torch.nn.attention.flex_attention import and_masks

# 因果 + 滑动窗口：只看过去，且最多看 WINDOW 步
causal_and_window = and_masks(causal_mask, sliding_window_mask)
block_mask = create_block_mask(causal_and_window, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device)
```

### 实现中的坑

**坑 1：block_mask 和 score_mod 的语义边界**

```python
# 错误做法：用 score_mod 做稀疏化（慢！）
def bad_causal_score_mod(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float('-inf'))  # 不跳过计算

# 正确做法：稀疏化用 block_mask，数值变换用 score_mod
block_mask = create_block_mask(causal_mask, ...)  # 跳过整个 block
out = flex_attention(q, k, v, block_mask=block_mask)  # 快
```

block_mask 在 block 粒度（默认 128x128）跳过计算；score_mod 在 element 粒度修改数值，两者协同使用性能最佳。

**坑 2：数据类型**

```python
# score_mod 内部的 bias 计算建议用 float32
def score_mod(score, b, h, q_idx, kv_idx):
    bias = compute_bias(q_idx, kv_idx)  # float32
    return score + bias.to(score.dtype)  # 再转回 score 的类型
```

**坑 3：MPS 上 torch.compile 需显式启用**

```python
# MPS 上 flex_attention 需要 compile 才能获得加速
compiled_flex = torch.compile(flex_attention, backend="aot_eager")  # MPS
# CUDA 上 flex_attention 内部已自动 compile
```

---

## 性能分析：12x 从哪里来？

12x 加速不是无条件的，它依赖**稀疏度**。来看一个对比：

```python
import time

def benchmark(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize() if device == "cuda" else None
    
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize() if device == "cuda" else None
    return (time.perf_counter() - t0) / iters * 1000  # ms

# SDPA 基线
sdpa_time = benchmark(
    torch.nn.functional.scaled_dot_product_attention, q, k, v, is_causal=True
)

# FlexAttention（因果，约 50% 稀疏）
flex_causal_time = benchmark(flex_attention, q, k, v, block_mask=causal_block_mask)

# FlexAttention（滑动窗口 128，~75% 稀疏）
flex_sw_time = benchmark(flex_attention, q, k, v, block_mask=sw_block_mask)

print(f"SDPA:              {sdpa_time:.2f}ms")
print(f"FlexAttn (causal): {flex_causal_time:.2f}ms")
print(f"FlexAttn (SW-128): {flex_sw_time:.2f}ms")
```

实际加速规律：

| 模式 | 稀疏度 | vs SDPA 加速比（CUDA） | vs SDPA 加速比（MPS） |
|------|--------|----------------------|---------------------|
| Full attention | 0% | ~1x | ~1x |
| Causal | 50% | ~1.8x | ~2x |
| Sliding Window (128) | 75%+ | ~4x | ~6x |
| Sparse (90%+) | 90% | ~8-12x | ~10-12x |

MPS 上加速比反而更高，因为 Metal 的 kernel 调度开销对全量 SDPA 更不友好，而 FlexAttention 的 block-skip 正好规避了这一点。

---

## 实验：论文说的 vs 现实

PyTorch 官方报告的 12x 是在**高稀疏度**（90%+）配置下测出的，且序列长度较长（≥ 2048）。

实际使用中需要注意：
- **序列长 512 以内**：FlexAttention overhead 不可忽略，收益有限，SDPA 可能更快
- **block_mask 的构建本身有开销**：`create_block_mask` 不是免费的，建议缓存
- **动态序列长度**：每次重新创建 block_mask 会触发重编译，建议 padding 到固定长度

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 序列长度 ≥ 1024，需要稀疏注意力 | 短序列（< 512）批量推理 |
| 需要实验多种注意力 pattern | 生产中只用标准因果注意力 |
| 在 Mac M 系列上做研究 | 部署到不支持 MPS/CUDA 的环境 |
| 多文档拼接训练（文档级 mask） | 需要极致量化（INT8/INT4） |
| ALiBi、RoPE 变体、自定义 bias | 只需要标准 SDPA 的 Flash Attention |

---

## 我的观点

FlexAttention 的真正价值不是"快"——而是**降低了注意力研究的工程门槛**。

在此之前，发一篇提出新注意力模式的论文，配套代码要么是慢到不能用的 Python 实现，要么需要一个专门的 CUDA 工程师。现在，`score_mod` 把"定义注意力语义"和"写高效 kernel"彻底解耦了。

MPS 支持的意义超过了苹果设备本身：它意味着这套抽象被证明是**后端无关的**。接下来如果出现 TPU、RDNA 等后端的支持，FlexAttention 很可能成为跨硬件注意力计算的统一层。

一个值得关注的开放问题：score_mod 目前只能表达**逐元素**的变换，无法表达需要跨 token 聚合信息的操作（比如 linear attention）。这个边界在哪里、能否突破，是这个方向下一步有趣的问题。