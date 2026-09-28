---
layout: post-wide
title: "EAServe：让 Encode 阶段成为多模态 LLM 服务的控制中枢"
date: 2026-09-28 08:05:50 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.31551v1
generated_by: Claude Code CLI
---

## 一句话总结

针对多模态 LLM（图像/视频/音频输入）推理服务中新增的 Encode 阶段，EAServe 把它从"附属的第一步"重新定位为整条 Encode-Prefill-Decode（EPD）流水线的控制点，通过负载自适应微批、部分 Prefill 卸载和动态 SM 分区三项机制协同工作，在相同 SLO 约束下相比 NVIDIA Dynamo 提升最多 4.3x goodput，相比 vLLM 提升 1.7x。

## 为什么需要这个？

纯文本 LLM 服务把 Prefill 和 Decode 拆到不同 GPU 池上，已经是业界标准做法——两者的算力/带宽比（compute-to-memory ratio）差异巨大，分离部署能各自跑满硬件特长。但多模态 LLM 在最前面多了一道 **Encode**：把图像 patch、视频帧或音频波形转换成 embedding，再喂给语言模型。

这道工序看起来只是"预处理"，实际上制造了一个结构性资源失衡：

- 每个请求都必须先经过 Encode，才能进入下游 Prefill/Decode——它是流水线的硬性入口，天然是延迟链路上不可跳过的一环。
- 但 Encode 通常是**逐请求执行**的：一张图、一段音频到达就单独跑一次前向。论文观察到的现象是，即便整体负载很高，Encode GPU 的时间平均利用率仍然很低——因为请求不是持续、密集地占满这块卡的执行队列，而是稀疏地一个一个到达再触发一次计算。
- 结果是：Encode GPU 大量时间在"等下一个请求"，而这段空闲期本可以用来分担 Prefill 的算力压力；与此同时下游 Prefill/Decode worker 因为等 Encode 结果而被"饿死"。

现有系统只能提供部分答案：纯文本 PD 系统压根没有 Encode 这个概念；已有的 EPD 框架虽然把 Encode 单独拆成一个服务，却没有对下游请求流量做任何调控——Encode 算完就一股脑扔给 Prefill，没有节奏控制。

## 核心原理

### 直觉：收费站类比

把 Encode 想象成高速公路入口的收费站。如果收费员坚持"一辆车来了就抬杆放行，放完关闸等下一辆"，即使后面车道很宽，车流量也上不去——大量时间浪费在了"等车"和"开关闸"上，而不是收费本身。解决办法要么是攒够一批车再统一放行（微批），要么是收费站空闲时干脆借用它旁边的车道分流部分车辆去下一个路口（跨阶段资源共享）。EAServe 同时做了这两件事，外加对"车道宽度"本身的动态调整。

### 硬件视角：问题出在哪一层

Encode 阶段的核心算子通常是 GEMM（比如 ViT 的 patch embedding 线性投影：`[num_patches, patch_dim] × [patch_dim, embed_dim]`）。单个请求的 `num_patches`（矩阵的 M 维）往往有限，如果逐请求单独发射 kernel：

- 这次 kernel 调用与下一次之间存在真实的**时间间隔**（等待网络请求、等待其他请求排队），GPU 在这段间隔里没有别的工作可做，是纯粹的空闲。
- 即使单次 kernel 内部 block 数量够用（现代 GPU 动辄 100+ SM），矩阵的 M 维越小，权重矩阵（K×N）在一次 kernel 执行中被复用的次数就越少，算术强度（arithmetic intensity）越低，Encode 更容易落入 memory-bound 区间，GPU 算力用不满。

而如果把多个请求的 patch 在 M 维上拼起来一起算（微批），既减少了 kernel 发射次数、填平了请求间的空闲期，又提高了权重复用率——这是批处理对 GEMM 效率的经典增益，只是这里的批次来源是"多模态请求"而非"多个 token"。

### EAServe 的三个控制维度

论文把 Encode 重新定位为控制点后，暴露出三个相互耦合的调节旋钮：

1. **When**：负载自适应微批（load-adaptive micro-batching）——用多长的等待窗口去攒批，直接决定 Encode 侧的吞吐/延迟权衡。
2. **Where**：速率受控的部分 Prefill 卸载（rate-controlled partial offload）——把一部分 Prefill 计算迁移到与 Encode 共处一卡（co-resident）的 worker 上执行，利用 Encode 卡的空闲周期。
3. **How**：动态 SM 分区（dynamic SM partitioning）——为共处同一张卡的 Encode kernel 和 Prefill kernel 划定各自可用的 SM 份额，保证两者并发时互不过度抢占，从而让"部分卸载"的延迟可预测。

下面用代码把这三个旋钮分别具体化。以下代码均为我们自己编写的示意实现，用于讲解机制原理，不是 EAServe 论文的原始实现，读者需要在自己的 GPU 上运行才能得到真实数值。

## 代码实现

### 1. 为什么单请求 Encode 吃不满带宽：一个朴素 GEMM kernel

```cuda
// 朴素 patch embedding：把线性投影当作 [M, K] x [K, N] 的 GEMM
// M = num_patches（batch=1 时很小，micro-batch 后线性增大）
// K = patch_dim, N = embed_dim
__global__ void patch_embed_naive(const float* patches, const float* weight,
                                   float* out, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 第几个 patch
    int col = blockIdx.x * blockDim.x + threadIdx.x; // embedding 第几维
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += patches[row * K + k] * weight[k * N + col];
    }
    out[row * N + col] = acc;
}

// 关键观察：weight[k*N+col] 会被每一行 row 重复读取
// M 越小（单请求场景），weight 矩阵在一次 kernel 生命周期内的复用次数越少
// -> 算术强度低，kernel 更容易受限于显存带宽而非算力
```

理论上按 occupancy calculator 估算：M=197（单张图 ViT-B/16 的 patch 数）时，`grid.y` 只有约 13 个 block；把 32 个请求的 patch 在 M 维拼接后，`grid.y` 增大到约 400+，单次 kernel 内权重的复用倍数也随之提升。注意这是基于 occupancy 公式的理论计算，不是实测值——真实的加速比取决于具体 GPU 型号和显存带宽，请以 Nsight Compute 的 `achieved_occupancy` 和 `dram__throughput` 指标为准。

### 2. When：负载自适应微批调度器

真正让利用率低的根源，是请求到达的稀疏节奏，而不只是单次 kernel 太小。下面用离散事件模拟展示"攒批窗口"如何影响 GPU 忙碌时间占比：

```python
import heapq

def simulate(arrival_times, max_batch, max_wait_ms, encode_ms_per_item):
    """离散事件模拟：请求到达 -> 攒批 -> 触发一次 encode kernel。
    返回 GPU 忙碌时间占总时长的比例（一个近似的"利用率"指标）。
    """
    events = [(t, "arrive") for t in arrival_times]
    heapq.heapify(events)
    pending, busy_time, last_flush = [], 0.0, 0.0

    def flush(now):
        nonlocal busy_time, pending, last_flush
        if not pending:
            return
        # 微批大小越大，单位请求的 kernel 发射开销被摊薄得越多
        busy_time += encode_ms_per_item * len(pending) * 0.6  # 批处理带来的效率折扣
        pending.clear()
        last_flush = now

    while events:
        t, _ = heapq.heappop(events)
        pending.append(t)
        if len(pending) >= max_batch or (t - last_flush) >= max_wait_ms:
            flush(t)
    flush(arrival_times[-1] if arrival_times else 0)

    total_span = arrival_times[-1] - arrival_times[0] if len(arrival_times) > 1 else 1
    return busy_time / total_span
```

**常见错误**：把 `max_wait_ms` 设得很大以追求高批处理效率，会直接推高首 token 时间（TTFT），违反 SLO——这正是论文强调"负载自适应"而非固定窗口的原因：窗口大小应该随实时到达率动态收缩/放大，而不是写死一个常数。

### 3. Where / How：部分 Prefill 卸载与并发执行

要让 Encode 卡的空闲周期被 Prefill 复用，首先需要两者能在同一张卡上并发跑而不互相拖慢。最基础的做法是用 stream 优先级让调度器倾向于优先执行 Prefill kernel：

```cuda
cudaStream_t encode_stream, prefill_stream;
int least_prio, greatest_prio;
cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio);

// Prefill 对延迟敏感，给更高优先级；Encode 用剩余算力见缝插针
cudaStreamCreateWithPriority(&prefill_stream, cudaStreamNonBlocking, greatest_prio);
cudaStreamCreateWithPriority(&encode_stream, cudaStreamNonBlocking, least_prio);

prefill_kernel<<<grid_p, block_p, 0, prefill_stream>>>(/* ... */);
patch_embed_naive<<<grid_e, block_e, 0, encode_stream>>>(/* ... */);
```

需要诚实说明的是：stream 优先级只影响**调度顺序**，本质上是时间片抢占式的机会性并发，并不能保证空间上的 SM 隔离——高优先级 kernel 抢占后，低优先级 kernel 仍可能因为长尾 block 占着 SM 而延迟释放。要做到论文中"动态 SM 分区"这种真正的空间隔离，需要 CUDA 12.4+ 的 Green Context（`cuGreenCtxCreate` / `cuDevSmResourceSplitByCount`）或 MPS 的 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`，两者的 API 细节和驱动版本要求变化较快，本文不展开完整示例，建议直接对照 NVIDIA 官方 Multi-Process Service 和 Green Context 文档验证。

### HAS：配置搜索的简化示意

论文的 Hybrid Auto Selection 先用每阶段容量画像剪枝掉明显失衡的分配方案，再用 TPE（Tree-structured Parzen Estimator）贝叶斯优化细化剩余空间。用 `optuna` 表达这个思路：

```python
import optuna

def objective(trial):
    encode_gpu_frac = trial.suggest_float("encode_gpu_frac", 0.1, 0.5)
    micro_batch_timeout = trial.suggest_float("timeout_ms", 1, 20)
    offload_ratio = trial.suggest_float("offload_ratio", 0.0, 0.6)

    # goodput_model 是对当前配置下系统吞吐/SLO达成率的估计
    # 真实系统中来自离线 profiling + 排队模型，这里仅作接口示意
    goodput = goodput_model(encode_gpu_frac, micro_batch_timeout, offload_ratio)
    return -goodput  # optuna 默认最小化

study = optuna.create_study(sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)
```

## 性能实测

以下数据来自论文摘要本身报告的结果（跨图像、视频、音频三种 MLLM 架构，在相同 SLO 约束下测得），并非我们在本地复现的实验：

| 对比对象 | Goodput 提升 | 备注 |
|---------|-------------|------|
| NVIDIA Dynamo | 最高 4.3x | 相同 SLO 约束下 |
| vLLM | 最高 1.7x | 相同 SLO 约束下 |
| 基线搜索方法 | 更快收敛到近最优配置 | HAS 相比纯网格/随机搜索 |

论文没有在摘要中公开具体 GPU 型号、CUDA 版本和三个 MLLM 的具体名称，本文也不做猜测性补充。如果要在自己的环境里验证类似效果，建议先用本文的离散事件模拟脚本量化"微批窗口 vs 利用率"曲线，再决定是否值得投入实现完整的跨阶段调度。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 图像/视频/音频编码器本身跑在 GPU 上，且并发请求量较大 | Encode 在 CPU 上完成（如简单音频特征提取），GPU 利用率问题根本不存在 |
| Encode 与 Prefill/Decode 分别部署在独立 GPU 池，追求提升整体 goodput | 单卡小规模部署，没有多余显存同时驻留 Encode 和 Prefill 模型 |
| GPU/驱动支持 MPS 或 Green Context 做空间级 SM 隔离 | 老架构 GPU 或驱动版本过低，无法做细粒度 SM 分区，只能退化为 stream 优先级的机会性并发 |
| 请求到达模式波动大，固定 batch/timeout 难以兼顾吞吐与 SLO | QPS 很低、Encode 本身耗时占比很小，调度复杂度带来的收益不明显 |

## 调试技巧

- 用 `nsys profile` 抓 timeline，确认 Encode kernel 和 Prefill kernel 是否真的在时间轴上重叠，而不只是逻辑上并发（stream 优先级不保证真正重叠）。
- 用 `nvidia-smi dmon -s u` 采样 SM 利用率随时间的变化，验证微批窗口调整前后忙碌区间是否被填平。
- 如果使用 MPS，务必检查 `CUDA_MPS_PIPE_DIRECTORY` / `CUDA_MPS_LOG_DIRECTORY` 环境变量和 `nvidia-cuda-mps-control` 守护进程是否真的启动，否则多进程 kernel 仍会退化为串行执行而不自知。
- 微批超时参数是最容易踩坑的地方：先固定 `max_batch` 网格搜索一遍 `max_wait_ms` 对 TTFT 的影响曲线，再考虑做成自适应，避免一上来就写复杂的自适应逻辑却没有基准对照。

## 延伸阅读

- CUDA Multi-Process Service（MPS）官方文档：重点看 `Volta MPS` 之后的显存/算力隔离能力边界。
- CUDA 12.4+ Green Context 相关文档：这是目前官方提供的更细粒度、进程内可控的 SM 分区机制，适合替代本文示例中的 stream 优先级方案。
- NVIDIA Dynamo 与 vLLM 的 PD 分离实现，作为理解"为什么多模态场景需要在此基础上再加一层 Encode 调度"的对照基线。