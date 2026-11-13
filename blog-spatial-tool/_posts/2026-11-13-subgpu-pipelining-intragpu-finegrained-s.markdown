---
layout: post-wide
title: "SubGPU Pipelining：把一张 GPU 的 SM 资源切开，流水线式跑 LLM Decode"
date: 2026-11-13 08:03:32 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2611.14904v1
generated_by: Claude Code CLI
---

## 一句话总结

论文在 GPT-OSS 120B + 32×B200 部署设定下验证：把一张 GPU 的 SM 资源在空间上切成多个子池，用流水线方式调度多个 microbatch 的 decode 计算，能把 DRAM 读写带宽利用率从 71.8% 提升到 83.6%，36 层 decoder 的单次迭代耗时降低 10.33%，在匹配吞吐的条件下最高能拿到 2.2x per-user 吞吐或 1.9x per-GPU 吞吐（论文模拟结果）。这不是新算子、不是新硬件，而是用 CUDA 12.4+ 引入的 **Green Context** API 做了一次"空间上的流水线"。

## 为什么需要这个？

LLM decode 阶段有个很别扭的特性：batch size 小，每个 kernel 跑得很快（几十到几百微秒），而且绝大多数 kernel 是 memory-bound 的。原因很直接——decode 阶段每次只生成一个 token，attention 和 MLP 里的矩阵乘法退化成矩阵-向量乘法，算术强度（arithmetic intensity）很低，SM 算力根本吃不满，瓶颈永远落在 HBM 带宽上。

问题在于，不同 kernel 的资源使用模式是"脉冲式"、错位的：attention 的 KV cache 读取是纯粹的带宽消耗型操作，SM 占用率其实很低；MLP 的矩阵乘法则相对更吃算力。如果用传统方式——一个 microbatch 从头跑到尾，依次占满整张 GPU 执行每个 kernel（论文称之为 **temporal execution**）——那么任意时刻，GPU 上只有一种资源在被充分使用，另一种资源大概率在空闲。这就是为什么论文测出来的 baseline DRAM 带宽利用率只有 71.8%：不是带宽不够用，是没有被"填满"。

现有的多 stream 并发方案看起来像是能解决这个问题，但实际上没有触及根本：CUDA 的 stream 并发只保证 kernel 可以并发*提交*，具体哪些 SM 去执行哪个 kernel，仍然由硬件的 GigaThread Engine 统一动态调度，没有静态的资源隔离。如果两个 kernel 本身的 grid 配置都足够大（能打满所有 SM），调度器大概率还是会把它们近似串行化处理，"并发"只停留在提交层面。

## 核心原理

### 先给直觉

把这个问题想象成工厂流水线。传统方式是一个工人从头做到尾，做完一件产品再开始下一件——对应 temporal execution：一个 microbatch 独占全部 SM，跑完整个算子图再轮到下一个 microbatch。SubGPU Pipelining 则是把工厂拆成两个工位：工位 A 专做前半道工序（比如 attention），工位 B 专做后半道工序（比如 MLP）。两个工位同时开工，产品在工位间流动——某一时刻,工位 A 在加工产品 3 的前半道，工位 B 同时在加工产品 2 的后半道。只要两个工位各自够忙，整条产线的综合利用率就比"单人全流程"高得多。

### 再讲硬件

关键在于怎么把"工位"落地到真实的 GPU 硬件上。GPU 由上百个 SM 组成，传统的多 stream 并发并不会把 SM 划分给固定的 stream——这是 NVIDIA 在 CUDA 12.4 之后补上的能力：**Green Context（绿色上下文）**。它允许在 driver API 层面，把一个物理设备的 SM 资源静态切成若干个互不重叠的分区，每个分区绑定一个独立的 context 和 stream。提交到这个 stream 上的 kernel，只会调度到分区内的 SM 执行，不会被其他分区抢占，也不会越界使用别的分区的资源。这就给了"空间隔离"一个硬件层面的保证，而不只是软件层面的期望。

SubGPU Pipelining 做的事情是：

1. **Graph cut**：把 decode 阶段的算子图按执行顺序切成若干连续段（segment），比如 attention 段、MLP 段；
2. **Spatial binding**：用 Green Context 给每个段分配一个独立的 SM 子池，作为一个"stage"；
3. **Pipeline schedule**：让多个 microbatch 依次流过这些 stage，同一时刻不同 stage 上跑着流水线里不同阶段、不同 microbatch 的 kernel。

论文特别强调，这四个决策——怎么切图、segment 怎么绑定 stage、每个 stage 分配多少 SM、microbatch 大小选多大——是一个**耦合优化问题**：SM 分配决定每个 segment 的执行时间，执行时间决定流水线里有没有 bubble（气泡/空等），而 microbatch 大小又决定了流水线能流多深、每次调用的固定开销占比有多大。

### 最后，公式

两级流水线稳定运行时，吞吐由较慢的那一级决定，而不是两级时间之和：

$$\text{throughput} \propto \frac{1}{\max(t_1, t_2)}$$

其中 $t_1$、$t_2$ 分别是两个 stage 处理一个 microbatch 的耗时。这解释了为什么 SM 配比要按**执行时间**而不是按算子数量去平分——只有 $t_1 \approx t_2$ 时，流水线才不会有一侧空等。

## 代码实现

下面的代码用来展示思路，不是完整可编译项目（我本地也没有 GPU 环境可以实测编译，Green Context 这套 driver API 在 CUDA 12.4 之后才引入，具体函数签名请以你本机 CUDA Toolkit 文档为准）。

### Baseline：朴素实现（多 stream，但没有 SM 空间分区）

```cuda
// ===== Baseline: 两个 microbatch 用两个普通 stream 跑同一套算子图 =====
__global__ void attn_kernel(float* q, float* k, float* v, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = q[idx] * k[idx] + v[idx]; // 占位：真实实现省略
}

__global__ void mlp_kernel(float* in, float* w, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * w[idx]; // 占位：真实实现省略
}

void run_baseline(int num_microbatches, dim3 grid, dim3 block /* ... */) {
    cudaStream_t stream_a, stream_b;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    for (int mb = 0; mb < num_microbatches; mb += 2) {
        attn_kernel<<<grid, block, 0, stream_a>>>(/* mb 的参数 */);
        attn_kernel<<<grid, block, 0, stream_b>>>(/* mb+1 的参数 */);
        mlp_kernel<<<grid, block, 0, stream_a>>>(/* mb 的参数 */);
        mlp_kernel<<<grid, block, 0, stream_b>>>(/* mb+1 的参数 */);
    }
    // 问题：如果 grid 本身就足够大（已能占满所有 SM），
    // stream_a / stream_b 上的 kernel 会被硬件调度器近似串行化，
    // "并发"只停留在提交层面，执行层面仍是排队
}
```

### 优化版本：用 Green Context 做空间分区

```cuda
// ===== 优化版: Green Context 做 SM 空间分区 + 流水线调度 =====
#include <cuda.h>  // Green Context 是 driver API (cu*)，不是 runtime API

CUgreenCtx g_ctx_attn, g_ctx_mlp;
CUstream stream_attn, stream_mlp;

void setup_spatial_partition(CUdevice dev, unsigned int sm_count) {
    CUdevResource dev_sm;
    cuDeviceGetDevResource(dev, &dev_sm, CU_DEV_RESOURCE_TYPE_SM);

    // 配比按实测耗时调整：attention 段带宽敏感、SM 需求小；
    // MLP 段算力敏感，通常需要更多 SM 才能跟 attention 段耗时对齐
    CUdevResource result[2];
    unsigned int nb_groups = 2;
    cuDevSmResourceSplitByCount(result, &nb_groups, &dev_sm,
                                 nullptr, 0, /*minCount=*/sm_count / 4);

    CUdevResourceDesc desc_attn, desc_mlp;
    cuDevResourceGenerateDesc(&desc_attn, &result[0], 1);
    cuDevResourceGenerateDesc(&desc_mlp,  &result[1], 1);

    cuGreenCtxCreate(&g_ctx_attn, desc_attn, dev, CU_GREEN_CTX_DEFAULT_STREAM);
    cuGreenCtxCreate(&g_ctx_mlp,  desc_mlp,  dev, CU_GREEN_CTX_DEFAULT_STREAM);
    cuGreenCtxStreamCreate(&stream_attn, g_ctx_attn, CU_STREAM_NON_BLOCKING, 0);
    cuGreenCtxStreamCreate(&stream_mlp,  g_ctx_mlp,  CU_STREAM_NON_BLOCKING, 0);
}

void run_subgpu_pipeline(int num_mb, dim3 g_attn, dim3 g_mlp, dim3 block) {
    // 流水线核心：mb 的 attention 跟 mb-1 的 MLP 物理上并发，
    // 因为它们绑定在两个互不重叠的 SM 子池上
    for (int mb = 0; mb < num_mb; mb++) {
        attn_kernel<<<g_attn, block, 0, (cudaStream_t)stream_attn>>>(/* mb */);
        if (mb > 0) {
            mlp_kernel<<<g_mlp, block, 0, (cudaStream_t)stream_mlp>>>(/* mb-1 */);
        }
    }
    mlp_kernel<<<g_mlp, block, 0, (cudaStream_t)stream_mlp>>>(/* 最后一个 microbatch */);
}
```

**为什么更快**：两个 stream 分别绑定在不重叠的 SM 子池上，attention 的带宽消耗和 MLP 的算力消耗被安排到同一时刻、不同 SM 上发生，而不是在时间轴上排队等待。这正是论文把 DRAM 带宽利用率从 71.8% 拉到 83.6% 的来源——瓶颈资源（带宽）在更多时间片里被用上了。

### 常见错误

```cuda
// 错误 1：以为多个 stream 自动获得独立 SM —— 没有 Green Context 分区时不成立
cudaStreamCreate(&stream_a); cudaStreamCreate(&stream_b);
kernel_a<<<1024, 256, 0, stream_a>>>(...); // grid=1024，已经能占满所有 SM
kernel_b<<<1024, 256, 0, stream_b>>>(...); // 实际在排队等 kernel_a 让出 SM
```

```cuda
// 错误 2：按算子数量而不是按实测耗时平分 SM
unsigned int min_count = sm_count / 2; // 想当然地对半分
// 如果 attention 段只需 1/4 的 SM 就能打满带宽，
// 而 MLP 段需要 3/4 的 SM 才能打满算力，对半分会让 MLP 段拖慢整条流水线
```

## 性能实测

> 下表数据来自论文原文的硅片实验和分析模拟结果，并非本文作者实测。测试环境：GPT-OSS 120B，用单张 B200 GPU 模拟 32-GPU B200 部署（通过 CUDA Green Contexts + TensorRT-LLM kernels 验证一个两段流水线配置，global batch = 2048，36 层 decoder）。

| 对比维度 | Temporal baseline | SubGPU Pipelining | 数据来源 |
|---|---|---|---|
| DRAM 读写带宽利用率 | 71.8% peak | 83.6% peak | 硅片实验 |
| 36 层 decoder 单次迭代耗时 | 基准 | -10.33% | 硅片实验 |
| 匹配 per-GPU 吞吐下的 per-user 吞吐 | 基准 | 最高 +2.2x | 分析模拟（Pareto frontier 搜索） |
| 匹配 per-user 吞吐下的 per-GPU 吞吐 | 基准 | 最高 +1.9x | 分析模拟（Pareto frontier 搜索） |

需要注意的是，2.2x / 1.9x 是模拟器在搜索整个流水线结构和 batch size 空间后画出的 Pareto frontier 上的最优点，不是任意配置都能达到；硅片实验验证的是其中**一个**具体的两段流水线配置，用来确认模拟结果在真实硬件上物理可行。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---|---|
| Decode 阶段，batch 偏小，kernel 以 memory-bound 为主 | Prefill 阶段——本身就是 compute-bound，单 kernel 已能打满 SM，没有"空闲资源"可切 |
| 单卡内在 TP/PP 之后仍有明显空闲的 SM 或带宽余量 | 对延迟极度敏感、不接受流水线引入的额外调度/同步开销的场景 |
| CUDA 12.4+ 且驱动支持 Green Context 的架构（论文验证于 Blackwell B200） | 更早的架构或驱动版本不支持 Green Context，需要先确认平台兼容性 |
| 愿意为了吞吐投入离线搜索（segment 切分、SM 配比、microbatch 大小）的工程成本 | 模型/batch 配置经常变化，没有余力为每种配置重新搜索最优流水线参数 |

## 调试技巧

验证 SM 分区是否真的生效，光看代码跑没跑通不够，要靠 Nsight Systems 看 timeline：

```bash
nsys profile --trace=cuda,nvtx -o subgpu_trace ./your_decode_binary
```

重点看两件事：一是两个 stage 的 kernel 在时间轴上是否真正重叠（而不是前后相继）；二是重叠之后每个 kernel 自身的耗时是否稳定——如果分区没生效，你会看到两个"并发"的 kernel 实际执行时长都比单独跑时变长，这说明它们还在抢同一批 SM。

另一个容易踩的坑是 Green Context 创建出来的 resource descriptor 和实际绑定的 stream 要对应检查：如果调用链里某一步传错了 context handle，kernel 很可能悄悄跑回了默认 context，分区形同虚设，但程序不会报错，只是性能数据对不上预期——这种 silent failure 比编译错误更难排查，建议在搭建阶段先用小规模 kernel 做分区验证，再接入真实的算子图。

## 延伸阅读

- 论文原文：[SubGPU Pipelining: Intra-GPU Fine-Grained Spatial Parallelism for LLM Inference](https://arxiv.org/abs/2611.14904v1)
- 论文中用到的推理框架：[NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- CUDA Driver API 官方文档（Green Context 相关接口的权威定义以此为准）：[https://docs.nvidia.com/cuda/cuda-driver-api/](https://docs.nvidia.com/cuda/cuda-driver-api/)