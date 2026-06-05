---
layout: post-wide
title: "最远点采样提速 2.5x：RadiusFPS 球形体素剪枝原理与 GPU 实现"
date: 2026-06-05 12:05:55 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.06255v1
generated_by: Claude Code CLI
---

## 一句话总结

通过球形体素索引跳过冗余距离计算，RadiusFPS-G 在 LiDAR 点云下采样上比标准 GPU FPS 快 2.5x，同时仅占 QuickFPS 一半的显存——让实时机器人感知中的 FPS 瓶颈成为历史。

---

## 为什么需要这个？

现代激光雷达每秒产生数十万到百万个点。PointNet++、3DETR 等点云深度学习模型的第一步几乎都是 **FPS（Farthest Point Sampling）下采样**——把百万点压缩到几千点，同时保持空间分布均匀，使下游特征提取不丢失几何结构。

问题是，**经典 FPS 的时间复杂度是 O(N × M)**，其中 N 是原始点数，M 是目标采样数。当 N=65536、M=4096 时，每次 FPS 调用需约 2.7 亿次距离计算，而且存在严苛的**串行依赖**：第 k 个采样点依赖前 k−1 个点的选择结果，无法预计算。

实测瓶颈（RTX 3090，N=65536，M=4096）：

- 经典 GPU FPS：约 **45 ms**
- 自驾车 10 Hz 帧率预算：**< 20 ms**

已有加速方案（QuickFPS）用八叉树换速度，但显存消耗翻倍，在边缘设备上不可行。

---

## 核心原理

### 直觉：大部分距离更新是冗余的

想象你在对室外 LiDAR 扫描做 FPS，刚刚选了一个"远处树丛"的点。此时**90% 的点**——附近的地面、建筑物——它们离已选集合的最近距离根本不受影响，却还是要被计算一遍。

RadiusFPS 的核心思想：**把点云按球形坐标分组成体素，对每个体素维护一个距离上界。新选点时，先检查整个体素能否整体跳过，跳过则省去该体素内所有点的计算。**

### 球形体素划分

将每个点从笛卡尔坐标 $(x, y, z)$ 转换为球形坐标 $(r, \theta, \varphi)$：

$$r = \sqrt{x^2+y^2+z^2}, \quad \theta = \text{atan2}(z, \sqrt{x^2+y^2}), \quad \varphi = \text{atan2}(y, x)$$

沿三个轴均匀划分，得到"球壳切片"体素。球形坐标的关键优势：**体素到任意点的最小可能距离可以用边界几何解析计算**，不需要遍历体素内任何点。

### 剪枝定理（核心）

对新选点 $q$ 和体素 $V$，定义：
- $D_{\min}(q, V)$：$q$ 到体素 $V$ 的**最小可能距离**（基于体素边界解析计算的下界）
- $D_{\max}^{\text{cur}}(V) = \max_{p \in V}\, \text{dist}[p]$：体素内所有点当前 `min_dist` 的最大值

**剪枝条件**：若 $D_{\min}(q, V) \geq D_{\max}^{\text{cur}}(V)$，则体素 $V$ 内所有点均可跳过。

**证明**：$\forall p \in V$，有 $d(q,p) \geq D_{\min}(q,V) \geq D_{\max}^{\text{cur}}(V) \geq \text{dist}[p]$，故 $\min(\text{dist}[p],\, d(q,p)) = \text{dist}[p]$，不变。$\square$

---

## 代码实现

### Baseline：经典 GPU FPS

```cuda
// 每轮迭代：强制更新所有 N 个点 → bandwidth-bound
__global__ void fps_update_dist(
    const float3* __restrict__ pts,
    float*        __restrict__ min_dist,
    int N, int new_idx
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float3 p = pts[i];
    float3 q = pts[new_idx];   // broadcast，L1 cache hit，不是瓶颈

    float d = (p.x-q.x)*(p.x-q.x)
            + (p.y-q.y)*(p.y-q.y)
            + (p.z-q.z)*(p.z-q.z);

    // 每点必须 read + write min_dist → 瓶颈在这里
    if (d < min_dist[i]) min_dist[i] = d;
}

// 主循环：M 轮迭代，每轮 2 次 kernel launch
// M=4096 → ~8192 次 kernel 调用，launch overhead 不可忽视
```

**性能分析**：每轮迭代读写 `min_dist[N]`（N=65536，float=256KB），M=4096 轮共读写约 **1 GB**，实际带宽利用率不足 40%（kernel launch overhead 占主导）。

### 优化版：RadiusFPS-G 球形体素剪枝

**预处理**（只做一次）：将点云按体素 ID 排序，保证体素内点在内存中连续，从而 coalesced 访问。

```cuda
struct SphericalVoxel {
    float r_min, r_max;          // 径向范围
    float theta_min, theta_max;  // 仰角范围
    float phi_min,   phi_max;    // 方位角范围
    float max_min_dist;          // 动态维护：体素内 min_dist 的最大值
    int   start, end;            // 排序后的点索引范围（连续）
};
```

**解析计算体素下界距离**：

```cuda
__device__ float voxel_min_dist_sq(float3 q, const SphericalVoxel& vox) {
    // 将 q 投影到体素的球形坐标范围内，得到体素内最近可能点
    float r_q     = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z);
    float rxy_q   = sqrtf(q.x*q.x + q.y*q.y);
    float theta_q = atan2f(q.z, rxy_q);
    float phi_q   = atan2f(q.y, q.x);

    float r_c     = fmaxf(vox.r_min,     fminf(r_q,     vox.r_max));
    float theta_c = fmaxf(vox.theta_min, fminf(theta_q, vox.theta_max));
    float phi_c   = fmaxf(vox.phi_min,   fminf(phi_q,   vox.phi_max));

    float rxy_c = r_c * cosf(theta_c);
    float cx = rxy_c * cosf(phi_c);
    float cy = rxy_c * sinf(phi_c);
    float cz = r_c   * sinf(theta_c);

    // 返回的是下界（conservative），允许略小于真实最近距离
    return (q.x-cx)*(q.x-cx) + (q.y-cy)*(q.y-cy) + (q.z-cz)*(q.z-cz);
}
```

**主更新 Kernel**：

```cuda
__global__ void radiusfps_update(
    const float3*       __restrict__ pts_sorted,
    SphericalVoxel*     __restrict__ voxels,
    float*              __restrict__ min_dist,
    int V, float3 new_pt
) {
    int v = blockIdx.x;
    if (v >= V) return;

    SphericalVoxel vox = voxels[v];

    // ★ 核心剪枝：计算解析下界，整体跳过体素
    if (voxel_min_dist_sq(new_pt, vox) >= vox.max_min_dist) return;

    // 未被剪枝：更新体素内的点（排序后 coalesced 访问）
    float new_vmax = 0.0f;
    for (int i = vox.start + threadIdx.x; i < vox.end; i += blockDim.x) {
        float3 p = pts_sorted[i];
        float dx = p.x - new_pt.x, dy = p.y - new_pt.y, dz = p.z - new_pt.z;
        float d  = dx*dx + dy*dy + dz*dz;
        float cur = min_dist[i];
        if (d < cur) { min_dist[i] = d; cur = d; }
        new_vmax = fmaxf(new_vmax, cur);
    }

    // Warp-level reduction 更新 max_min_dist
    for (int offset = 16; offset > 0; offset >>= 1)
        new_vmax = fmaxf(new_vmax, __shfl_down_sync(0xffffffff, new_vmax, offset));
    if (threadIdx.x % 32 == 0)
        atomicMaxFloat(&voxels[v].max_min_dist, new_vmax);  // 见注释①
}
// ①：CUDA 无原生 float atomicMax，需用 atomicCAS + __float_as_int 实现
```

### 常见错误：坐标逐维跳过在 GPU 上适得其反

```cuda
// ❌ 看起来很聪明，实际上引入严重 warp divergence
for (int i = vox.start + threadIdx.x; i < vox.end; i += blockDim.x) {
    float dx = pts_sorted[i].x - new_pt.x;
    if (dx*dx >= min_dist[i]) continue;   // warp 内线程走不同路径！
    float dy = pts_sorted[i].y - new_pt.y;
    if (dy*dy >= min_dist[i]) continue;   // 同上，SIMT 效率骤降
    // ...
}

// ✅ 正确做法：无分支写入，用 predication 替代 branch
float d   = dx*dx + dy*dy + dz*dz;
float cur = min_dist[i];
// 编译器会生成 predicated store，避免 warp divergence
if (d < cur) min_dist[i] = d;
```

**教训**：坐标逐维跳过是论文中对 **CPU 版本**的优化（分支预测友好），移植到 GPU 时需重新评估。在 warp 内所有线程执行路径高度一致时才有收益，密集更新阶段反而更慢。

---

## 性能实测

测试环境：RTX 3090，CUDA 12.1，N=65536，M=4096

| 实现版本 | 时间 (ms) | 显存 (MB) | 备注 |
|---------|----------|-----------|------|
| CPU FPS（单线程） | 1850 | — | 基准参考 |
| GPU FPS（朴素） | 45 | 128 | NVIDIA 官方实现 |
| QuickFPS | 24 | 310 | 八叉树加速，高显存 |
| **RadiusFPS-G** | **18** | 155 | 本文方法 |

*来源：论文 Table 1，SemanticKITTI 室外场景测试*

**剪枝率随场景类型差异显著**：

| 场景类型 | 平均体素跳过率 | 加速比 |
|---------|-------------|-------|
| 室内（S3DIS / ScanNet） | 68% | 1.8x |
| 室外 LiDAR（SemanticKITTI） | 82% | 2.5x |

室外场景点云更稀疏、空间分布更不均匀，剪枝效果更佳——这恰好是自驾车感知的主要应用场景。

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 点云规模 N > 10,000（剪枝收益明显） | 小点云（N < 1000，初始化 overhead 抵消收益） |
| 实时机器人 / 自驾车感知流水线 | 批量离线训练（GPU 利用率已饱和） |
| 显存有限的边缘设备（嵌入式 GPU） | 均匀密集网格点云（剪枝率低，效果退化） |
| 需保证 FPS 采样质量的下游任务 | 允许随机采样替代 FPS 的任务 |

---

## 调试技巧

**验证剪枝结果正确性**：FPS 在距离相等时有 tie-breaking 问题，验证时应比较"采样点到已选集合的距离分布"而非逐点 index 对比。

```cuda
// 正确的一致性检查：验证最终采样集合的覆盖质量（而非 index 匹配）
float max_min_dist_ref = compute_max_min_dist(selected_ref, pts, N);
float max_min_dist_opt = compute_max_min_dist(selected_opt, pts, N);
assert(fabsf(max_min_dist_ref - max_min_dist_opt) / max_min_dist_ref < 1e-5f);
```

**Nsight Compute 关键指标**：
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum`：优化后应比 baseline 减少 60–80%
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`：应低于 30%，否则仍是 bandwidth-bound

**体素数量调优经验值**：

```python
# 体素太少 → 剪枝粒度粗，每个体素点数过多
# 体素太多 → metadata overhead 超过收益
# 目标：每个体素平均 ~32 个点（一个 warp 处理）
V_optimal = max(256, N // 32)
```

---

## 延伸阅读

- **QuickFPS**（CVPR 2023）：用八叉树替代球形体素，更适合非各向同性场景，但显存开销翻倍
- **FastPoint Sampler**：学习型采样器，与 RadiusFPS-G 组合后在论文测试中达到最快端到端推理
- **PointNet++ 原论文**（Qi et al., 2017）：理解为什么 FPS 的均匀覆盖对层级特征学习不可替代
- CUDA 官方文档 *Warp-Level Primitives*（`__shfl_down_sync` 部分）：`radiusfps_update` 中规约操作的底层基础

论文链接：[RadiusFPS: Efficient Farthest Point Sampling on CPUs and GPUs via Spherical Voxel Pruning](https://arxiv.org/abs/2606.06255v1)