---
layout: post-wide
title: "当 torch.compile 把 Matmul 交给 cuBLAS，谁来优化剩下的 Triton 代码？"
date: 2026-09-25 08:05:16 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.30059v1
generated_by: Claude Code CLI
---

## 一句话总结

KernelOPT 是一套多智能体系统，专门优化 `torch.compile` (PyTorch Inductor) 编译出来的 **Triton 子内核**，同时保留 cuBLAS / cuDNN 这些 vendor library 调用不动。在 KernelBench 的 250 个问题上，它相对 `torch.compile` 基线取得了 Level 1（纯 kernel，1.40x，51/100 题改进）、Level 2（融合算子，1.15x，31/100 题改进）、Level 3（完整网络，1.07x，12/50 题改进）的几何平均加速。

这篇文章不复述论文，而是带你理解一个更底层的问题：**为什么 Inductor 生成的 Triton 代码天生有优化空间，以及这个空间具体长什么样**。我们会写一个真实的 Triton kernel、指出它的性能问题、手工优化它，并展示 KernelOPT 论文里最有意思的部分——用 float64 fallback 做数值真值校验的思路，这个技巧本身就值得搬进你自己的 kernel 测试框架。

## 为什么需要这个？

先搞清楚 `torch.compile` 到底生成了什么。当你写：

```python
y = torch.nn.functional.gelu(x @ w + b)
```

Inductor 不会把整段代码变成一个 Triton kernel。它会做**图切分**：

- `x @ w` 这种规整的矩阵乘法，Inductor 认为手写 Triton 很难打过 cuBLAS 的专家实现（NVIDIA 工程师针对每一代 Tensor Core 微架构调过的 GEMM kernel），所以直接生成一次 **extern call**，走 cuBLAS。
- 剩下的 `+ b` 和 `gelu(...)` 是 elementwise 操作，Inductor 会把它们**融合**成一个 Triton kernel，减少一次 global memory 往返。

问题出在这个 Triton 融合 kernel 上。Inductor 的 codegen 是规则驱动的：它知道怎么融合、怎么选 `BLOCK_SIZE`，但它的 autotuning 搜索空间是有限的、通用的，不会针对你这个具体 shape（比如 `n_cols` 很小、或者 batch 维度非常大）做专门优化。这就是论文里 1.40x 加速的来源——不是重写 matmul，而是把这些"边角料" Triton kernel 打磨得更贴合硬件。

KernelOPT 的核心洞察是：**不要把编译后的模型当黑盒**。之前的 LLM kernel 优化器往往拿到一个 kernel 就直接重写，可能连 matmul 都想用 Triton 重新实现——这通常是在做无用功甚至倒退。KernelOPT 先respect 编译器已经做出的"这部分该用库、那部分该生成代码"的结构决策，只在 Triton 生成的部分动刀。

## 核心原理

### 直觉：GPU 上的时间都花在哪

一个 fused bias+GELU 的 Triton kernel，性能瓶颈几乎总是内存带宽，不是算力。GELU 的 tanh 近似有几次乘加，但相比一次 global memory load/store，算力开销可以忽略。所以优化这类 kernel 的核心问题永远是：**每个字节从 HBM 搬进 SM 之后，能不能被充分复用，线程能不能在等待内存的时候不闲着**。

### 硬件层面：Warp 调度与 Occupancy

Inductor 为一个 `[batch, n_cols]` 的 tensor 生成 kernel 时，常见策略是"一行一个 program"（`program_id` 对应行号）。如果 `n_cols` 很小（比如 128），每个 program 只需要几个 warp 就能覆盖整行，SM 上能同时驻留的 program 数量受限于 `num_warps` 和 shared memory 占用，而不是计算量本身。这时候瓶颈是**occupancy 不够**——SM 没有足够多的 warp 在轮转，内存延迟没被隐藏。

优化方向通常是：让一个 program 处理多行（提高每次 launch 的计算密度），或者调整 `num_warps` 让调度器有更多 warp 可切换。这些正是 KernelOPT 里 profiling-guided agent 会去搜索的维度——它不是瞎猜，而是先跑 profiler，看 kernel 是 memory-bound 还是 occupancy-bound，再决定往哪个方向改。

### 四道验证关卡

论文里最值得借鉴的不是搜索算法，是**验证流水线**。LLM 生成的 kernel 代码，"看起来对"和"真的对"之间的差距经常是灾难性的（比如 mask 边界写错，只在特定 shape 下才暴露）。KernelOPT 用四道 gate 过滤候选：

1. **静态验证**：语法、语义检查，能不能编译。
2. **多随机种子正确性**：换几个随机输入种子跑，输出必须和参考实现在容差内一致——防止候选 kernel 只是"蒙对了一组测试数据"。
3. **模型级 float64 fallback 验证**：把整个模型切到 float64 精度跑一遍，得到"数值真值"，再用它去校验低精度（fp16/bf16）候选 kernel 的误差是否在合理范围——这一步专门用来区分"kernel 有 bug"和"低精度本来就有误差"。
4. **性能门槛**：候选必须在真实硬件上measure出比基线快，理论分析不算。四道全过才会替换编译器基线，否则保留 Inductor 原始版本——这是个重要的工程决策：**优化失败的代价是零**，不会比 `torch.compile` 更差。

下面我们把关卡 2 和 3 的思路写成可复用的验证代码。

## 代码实现

### Baseline：Inductor 风格的融合 kernel

这是一个典型的 "bias + GELU" 融合 kernel，风格上模拟 Inductor 对 `linear` 后接 `gelu` 的 Triton codegen（matmul 部分假设已经是 cuBLAS extern call，这里只优化 epilogue）：

```python
import triton
import triton.language as tl

@triton.jit
def fused_bias_gelu_baseline(
    x_ptr, bias_ptr, out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(x_ptr + row * n_cols + col_offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    z = x + b

    # GELU 的 tanh 近似
    inner = 0.7978845608 * (z + 0.044715 * z * z * z)
    gelu = 0.5 * z * (1.0 + tl.math.tanh(inner))

    tl.store(out_ptr + row * n_cols + col_offsets, gelu, mask=mask)
```

**性能分析**：这个 kernel 的问题是"一行一个 program"，当 `n_cols` 较小（比如 Transformer 里常见的 head_dim=128）时，`BLOCK_SIZE=128` 只需要 4 个 warp，如果 SM 支持驻留更多 warp，剩下的调度槛位是空的——kernel 跑得快，但 SM 大部分时间没被喂满，整体吞吐低于带宽上限。用 Nsight Compute 看的话，典型症状是 **Achieved Occupancy 远低于 Theoretical Occupancy**，而 Memory Throughput 也没跑满。

### 优化版本：多行合并 + autotune

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_PROGRAM": r, "BLOCK_SIZE": 128}, num_warps=w)
        for r in [1, 2, 4, 8] for w in [2, 4, 8]
    ],
    key=["n_cols"],
)
@triton.jit
def fused_bias_gelu_opt(
    x_ptr, bias_ptr, out_ptr,
    n_rows, n_cols,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROGRAM
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    # bias 在所有行之间复用，一次加载即可
    b = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0)

    for r in range(ROWS_PER_PROGRAM):
        row = row_start + r
        if row < n_rows:
            ptr = x_ptr + row * n_cols + col_offsets
            x = tl.load(ptr, mask=col_mask, other=0.0)
            z = x + b
            inner = 0.7978845608 * (z + 0.044715 * z * z * z)
            gelu = 0.5 * z * (1.0 + tl.math.tanh(inner))
            tl.store(out_ptr + row * n_cols + col_offsets, gelu, mask=col_mask)
```

**为什么更快**：

- `ROWS_PER_PROGRAM` 让一个 program 处理多行，`bias` 只需从 global memory 加载一次并在寄存器里复用，而不是每行都重复读取——直接减少了对同一份数据的重复内存访问。
- `@triton.autotune` 让 Triton 在真实硬件上试跑不同的 `(ROWS_PER_PROGRAM, num_warps)` 组合，选出实测最快的配置，而不是像 Inductor 默认策略那样用一套通用规则一次定死。这正是 KernelOPT 缩小的那个 gap：Inductor 的 autotune 空间是"够用但不够细"，agent 搜索出的组合往往能踩中 Inductor 没试过的点。

### 常见错误：把 matmul 也用 Triton 重写

```python
# 反例:看到 x @ w 就想用 Triton "优化" 掉
@triton.jit
def naive_matmul_kernel(...):
    # 手写 tiled matmul,没有针对 Tensor Core 的
    # ldmatrix / mma 指令做特化,也没有 cuBLASLt
    # 的 kernel selection heuristics
    ...
```

对规整、大尺寸的 GEMM，cuBLAS 背后是 NVIDIA 针对每代架构专门调过的 kernel 库（包括 Tensor Core 的 `mma` 指令排布、多级流水线），业余重写基本不可能打平，更谈不上超越。这也是论文强调"dispatch-aware"的原因：KernelOPT **明确排除**已经走 vendor library 的部分，只对 Inductor 生成的 Triton 子内核下手。如果你的优化器不做这个区分,大概率会在 matmul 上浪费大量搜索预算,还得到一个更慢的结果。

### 验证关卡的实现思路

这是论文里"多随机种子"和"float64 fallback"两道 gate 的简化实现，值得直接搬进你自己的 kernel 测试脚本：

```python
import torch

def check_multi_seed(candidate_fn, ref_fn, shape, n_seeds=5):
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        bias = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
        out_c = candidate_fn(x, bias)
        out_r = ref_fn(x, bias)
        if not torch.allclose(out_c, out_r, atol=1e-2, rtol=1e-2):
            return False, seed
    return True, None

def check_float64_fallback(candidate_fn, ref_fn, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    bias = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
    # 用 float64 建立数值真值,把"低精度本身的误差"
    # 和"kernel 逻辑错误"分开判断
    truth = ref_fn(x.double(), bias.double())
    err = (candidate_fn(x, bias).double() - truth).abs().max().item()
    return err
```

`check_multi_seed` 防的是"候选 kernel 只在某一组输入上碰巧对了"；`check_float64_fallback` 防的是反过来的问题——**候选和参考在 fp16 下有差异，但这差异只是低精度舍入误差，不是 bug**，如果不建立 float64 真值，很容易把一个正确的优化错误地判定为失败。

## 性能实测

论文在 KernelBench（250 个问题，覆盖单算子 / 融合算子 / 完整网络三个层级）上的结果：

| Level | 问题数 | 改进的问题数 | 几何平均加速（相对 torch.compile） |
|-------|--------|-------------|-----------------------------------|
| Level 1（单 kernel） | 100 | 51 | 1.40x |
| Level 2（融合算子） | 100 | 31 | 1.15x |
| Level 3（完整网络） | 50 | 12 | 1.07x |

有个规律很直观：越往上层（Level 3 完整网络），可优化空间越小。原因和我们前面分析的一致——完整网络里 matmul/conv 占比更大，这些已经走 cuBLAS/cuDNN，KernelOPT 根本不碰；能优化的只是网络里 Triton 生成的那部分 elementwise/reduction 逻辑，占总耗时比例本来就不高，所以整体加速被"摊薄"了。

对于本文的 demo kernel，我没有在本机跑通完整 benchmark（这次写作环境没有可用 GPU），所以不编造具体的 ms 数字。如果你想验证上面 baseline 和优化版本的差距，用 `triton.testing.do_bench` 在自己的 GPU 上跑一次就是最诚实的答案：

```python
import triton
ms_baseline = triton.testing.do_bench(lambda: fused_bias_gelu_baseline[grid](...))
ms_opt = triton.testing.do_bench(lambda: fused_bias_gelu_opt[grid](...))
print(f"speedup: {ms_baseline / ms_opt:.2f}x")
```

不同 GPU 架构（Ampere vs Hopper）、不同 `n_cols`，最优的 `ROWS_PER_PROGRAM` 会不一样，这也是为什么 autotune / agent 搜索这类"实测驱动"的方法比手工猜配置更稳健。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 模型里有大量 Inductor 生成的 elementwise/reduction 融合 kernel（LayerNorm、GELU、softmax epilogue 等） | 模型主要耗时在 matmul/conv，且 shape 规整，cuBLAS/cuDNN 本身已经跑满带宽 |
| 你在跑 KernelBench 类型的搜索/benchmark 场景，能接受多候选生成+验证的开销 | 线上服务对编译/优化耗时敏感，无法承受多轮 profiling-验证循环 |
| 团队没有精力手工为每个 shape 调 Triton autotune 空间 | 已经有专家手写的 kernel（比如 FlashAttention），再搜索大概率不会超越 |

## 调试技巧

- 遇到 Inductor 生成的 kernel 想看它到底长什么样，设置 `TORCH_LOGS="output_code"` 环境变量，`torch.compile` 会把生成的 Triton 源码打印出来，这是理解"编译器切分了什么、留了什么给 Triton"最直接的方式。
- 怀疑某个融合 kernel 是 occupancy 瓶颈还是 memory 瓶颈，用 Nsight Compute 的 `Occupancy` 和 `Memory Throughput` 两个 section 对照看——两者都低通常是 launch 配置问题（比如网格太小），只有 occupancy 低但带宽已经跑满则说明 kernel 已经接近理论上限，别再浪费时间调 block size。
- 验证一个"优化后更快"的候选 kernel 时，永远先跑多个随机种子再跑 float64 对照，顺序反了容易被 fp16 的正常舍入误差误导，浪费时间去"修一个不存在的 bug"。

## 延伸阅读

- 论文原文：[KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization](https://arxiv.org/abs/2609.30059)
- PyTorch 官方文档中关于 `torch.compile` 的图切分与 Inductor 后端说明：[pytorch.org/docs/stable/torch.compiler.html](https://pytorch.org/docs/stable/torch.compiler.html)
- Triton 官方文档中的 autotune 与 kernel benchmark 工具：[triton-lang.org](https://triton-lang.org/main/index.html)