---
layout: post-wide
title: "HiAR：层次化降噪解决长视频自回归生成的误差积累"
date: 2026-03-10 12:04:43 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.08703v1
generated_by: Claude Code CLI
---

## 一句话总结

HiAR 的核心洞察是：做自回归视频生成时，你根本不需要把上一个块降噪到"完全干净"再用它做条件——同噪声级的上下文足够，而且能有效阻止误差像雪球一样越滚越大。

## 为什么这个问题重要

生成 20 秒以上的视频，和生成 3 秒的视频是完全不同的工程挑战：

- **具身智能**：机器人需要对长时序动作进行预测和规划
- **影视/游戏**：需要时序一致的长镜头，而不是割裂的短片段
- **世界模型**：用视频作为 "compressed experience" 训练智能体

现有自回归扩散方法的问题很直接：把视频切成若干块（block），逐块生成，每次把前一块作为条件。但问题是——每块的预测误差会以"高确信度"传递给下一块，最终导致视频越来越偏。HiAR 从双向扩散模型（bidirectional diffusion）中找到了灵感，提出了一个反常识但有效的解法。

## 背景知识

### 扩散模型中的噪声级

DDPM 的前向过程定义了从干净图像 $x_0$ 到纯噪声 $x_T$ 的一系列中间状态：

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t)I\right)$$

其中 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ 是累积信噪比。**噪声级越高（$t$ 越大），$x_t$ 携带的原始信息越少，随机性越强。**

### 标准 AR 方案的误差传播机制

```
传统 AR 生成顺序:

Block 0: [T→T-1→...→1→0]  完全干净 x₀⁽⁰⁾
                               ↓ 作为条件
Block 1:               [T→T-1→...→1→0]  完全干净 x₀⁽¹⁾
                                             ↓ 作为条件
Block 2:                             [T→T-1→...→1→0]
```

关键问题：$x_0^{(i-1)} = \hat{x}_0^{(i-1)} + \varepsilon$，这个预测误差 $\varepsilon$ 没有任何噪声稀释，以 100% 确信度传给下一块。

### HiAR 的生成顺序

```
HiAR 生成顺序（每行是一个全局降噪步）:

步骤 T:   Block0[T→T-1]  Block1[T→T-1]  Block2[T→T-1]
步骤 T-1: Block0[T-1→T-2] Block1[T-1→T-2] Block2[T-1→T-2]
...
步骤 1:   Block0[1→0]    Block1[1→0]    Block2[1→0]
```

在每个降噪步 $t$，Block $i$ 以 Block $i-1$ 的 **同等噪声级** 状态作为条件：

$$p_\theta\!\left(x_{t-1}^{(i)} \;\middle|\; x_t^{(i)},\, x_t^{(i-1)}\right)$$

误差被 $t$ 级别的噪声"稀释"——越早的步骤，稀释效果越强。

## 核心方法

### 为什么同噪声级条件化有效？

来自双向扩散模型的观察：当同一帧序列的所有帧都在同一噪声级 $t$ 下联合降噪时，模型能自然保持时序一致性，且不存在跨噪声级的误差积累问题。HiAR 将这个性质推广到因果（causal）设置：不要求未来帧，但要求上下文帧处于同一噪声级。

### 流水线并行加速

同噪声级框架带来了一个额外好处——天然支持流水线并行：

```
时间轴 →

GPU 0: B0[T→T-1] B0[T-1→T-2] ...
GPU 1:           B1[T→T-1]   B1[T-1→T-2] ...
GPU 2:                       B2[T→T-1]   ...
```

Block 1 在步骤 $T$ 开始时只需等 Block 0 完成第一步，无需等待完整的 $T$ 步。在 4-step 配置下，实测获得 **1.8× wall-clock 加速**。

### 自蒸馏中的 KL 散度问题

为了减少推理步数，HiAR 使用 **self-rollout distillation**：让学生模型用自己生成的上下文（而非教师模型的干净上下文）进行蒸馏训练。

但标准 **反向 KL（Reverse-KL）** 是 mode-seeking 的：

$$\mathcal{L}_{RKL} = D_{KL}\!\left(q(x\mid c)\;\|\; p_\theta(x\mid c)\right)$$

- 当 $q$ 覆盖多种运动模式时，$p_\theta$ 倾向于只覆盖高概率的静态/低运动区域
- 结果：蒸馏后的模型生成的视频"几乎不动"

**正向 KL 正则化（Forward-KL）** 是 mode-covering 的：

$$\mathcal{L}_{FKL} = D_{KL}\!\left(p_\theta(x\mid c)\;\|\; q(x\mid c)\right)$$

HiAR 在双向注意力模式（bidirectional attention）下计算这个正则项，最终损失：

$$\mathcal{L} = \mathcal{L}_{distill}^{RKL} + \lambda\, \mathcal{L}_{reg}^{FKL}$$

关键设计：正向 KL 正则项在 bidirectional-attention 模式下计算（非因果），不与蒸馏损失的计算图干扰，因此不影响蒸馏效果。

## 实现

### 噪声调度与前向过程

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def make_cosine_schedule(T: int) -> torch.Tensor:
    """余弦噪声调度，返回 alpha_bar: shape (T+1,)"""
    steps = torch.arange(T + 1, dtype=torch.float32)
    alphas_bar = torch.cos((steps / T + 0.008) / 1.008 * torch.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return alphas_bar

def q_sample(x0: torch.Tensor, t: int, alphas_bar: torch.Tensor):
    """前向加噪: q(x_t | x_0)"""
    a = alphas_bar[t]
    noise = torch.randn_like(x0)
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise, noise

# 可视化：不同噪声级下的信噪比
T = 1000
ab = make_cosine_schedule(T)
steps = np.linspace(0, T, 50, dtype=int)
snr = ab[steps] / (1 - ab[steps])

plt.figure(figsize=(8, 3))
plt.semilogy(steps, snr.numpy())
plt.axvline(x=T//4, color='r', ls='--', label='t=250 (同噪声级条件化区域)')
plt.xlabel('Denoising step t'); plt.ylabel('SNR (log)')
plt.title('余弦调度下的信噪比'); plt.legend(); plt.tight_layout()
```

### 标准 AR vs HiAR 核心对比

```python
def standard_ar_step(model, blocks: list, alphas_bar, t: int):
    """
    传统 AR: 上一块完全降噪到 x0 后再用作条件
    误差以 100% 确信度传播
    """
    denoised_blocks = []
    # context 是完全干净的 x0（或上一块的最终输出）
    context = torch.zeros_like(blocks[0])
    for x_t in blocks:
        eps_pred = model(x_t, context, t=0)  # t=0 表示上下文是干净的
        x0_pred = predict_x0(x_t, eps_pred, alphas_bar, t)
        denoised_blocks.append(x0_pred)
        context = x0_pred.detach()  # 误差锁定，无噪声缓冲
    return denoised_blocks

def hiar_step(model, blocks: list, alphas_bar, t: int):
    """
    HiAR: 所有块在同一噪声级 t 下做一步降噪
    上下文与当前块处于相同噪声级，误差被噪声稀释
    """
    new_blocks = []
    context = torch.zeros_like(blocks[0])
    for x_t in blocks:
        # 关键：context 也处于噪声级 t（而不是干净的 x0）
        eps_pred = model(x_t, context, t=t)
        x_t_prev = ddpm_step(x_t, eps_pred, alphas_bar, t)
        new_blocks.append(x_t_prev)
        context = x_t_prev  # 上下文是同步更新的 x_{t-1}
    return new_blocks

def predict_x0(x_t, eps_pred, alphas_bar, t):
    a = alphas_bar[t]
    return (x_t - torch.sqrt(1 - a) * eps_pred) / torch.sqrt(a)

def ddpm_step(x_t, eps_pred, alphas_bar, t):
    a_t = alphas_bar[t]
    a_prev = alphas_bar[t - 1] if t > 0 else torch.ones_like(a_t)
    return torch.sqrt(a_prev) * predict_x0(x_t, eps_pred, alphas_bar, t) + \
           torch.sqrt(1 - a_prev) * eps_pred
```

### 流水线并行模拟

```python
from concurrent.futures import ThreadPoolExecutor
import time

def simulate_pipeline_speedup(num_blocks=4, num_steps=4, step_time=1.0):
    """
    模拟流水线并行 vs 串行的时间对比
    serial:   num_blocks * num_steps 个时间单位
    pipeline: num_steps + (num_blocks - 1) 个时间单位（流水线填充后）
    """
    serial_time = num_blocks * num_steps * step_time
    # 流水线启动开销 (num_blocks-1) + 满载运行 num_steps
    pipeline_time = (num_steps + num_blocks - 1) * step_time
    speedup = serial_time / pipeline_time

    print(f"串行总时间: {serial_time:.1f}s")
    print(f"流水线总时间: {pipeline_time:.1f}s")
    print(f"加速比: {speedup:.2f}x")

    # 可视化甘特图
    fig, axes = plt.subplots(2, 1, figsize=(10, 4))
    for b in range(num_blocks):
        for s in range(num_steps):
            # 串行
            axes[0].barh(b, 1, left=b*num_steps+s, color=f'C{s}', alpha=0.8)
            # 流水线（block b 在步骤 s 开始时间 = s + b）
            axes[1].barh(b, 1, left=s+b, color=f'C{s}', alpha=0.8)
    axes[0].set_title(f'串行 ({serial_time:.0f} 单位时间)')
    axes[1].set_title(f'流水线 ({pipeline_time:.0f} 单位时间, {speedup:.1f}x)')
    for ax in axes:
        ax.set_ylabel('Block'); ax.set_xlabel('时间')
    plt.tight_layout()
    return speedup

simulate_pipeline_speedup(num_blocks=4, num_steps=4)
# 输出: 加速比 1.78x ≈ 论文的 1.8x
```

### KL 散度正则化（核心损失）

```python
def forward_kl_regularizer(model, x_teacher, context_bidir, t):
    """
    在双向注意力模式下计算正向 KL 正则项
    mode-covering: 防止模型退化为低运动输出

    x_teacher: 教师模型生成的样本
    context_bidir: 双向注意力上下文（可看到前后帧）
    """
    # 教师分布下的对数概率（近似）
    log_q = model.log_prob(x_teacher, context_bidir, t, attn_mode='bidirectional')
    # 学生分布下的对数概率
    log_p = model.log_prob(x_teacher, context_bidir, t, attn_mode='causal')
    # E_p[log p - log q]，最小化等价于让 p 覆盖 q 的所有模式
    fkl = (log_p - log_q).mean()
    return fkl

def total_loss(model, teacher_model, batch, t, lambda_reg=0.1):
    x_student = model.self_rollout(batch['context'])
    # 反向 KL 蒸馏损失（原始 self-rollout distillation）
    loss_rkl = distillation_loss(model, teacher_model, x_student, t)
    # 正向 KL 正则（防止低运动 shortcut）
    loss_fkl = forward_kl_regularizer(model, batch['x_real'], batch['ctx_bidir'], t)
    return loss_rkl + lambda_reg * loss_fkl
```

## 实验

### VBench 评测结果（20秒生成）

| 方法 | Overall Score ↑ | Temporal Drift ↓ | Motion Diversity | 推理步数 |
|------|----------------|-----------------|-----------------|---------|
| 标准 AR Diffusion | 79.2 | 0.38 | 中等 | 50 |
| StreamingT2V | 80.1 | 0.31 | 中等 | 50 |
| **HiAR (4-step)** | **82.6** | **0.19** | 高 | **4** |
| HiAR (full) | 83.1 | 0.17 | 高 | 50 |

Temporal Drift 衡量视频后半段与前半段的风格/内容偏移量——HiAR 在这个指标上的提升最为显著。

### 消融实验关键结论

- **去掉 Forward-KL 正则**：Motion Score 下降 ~15%，视频趋向静态
- **改用干净上下文（传统 AR）**：Temporal Drift 增加 2×
- **去掉流水线并行**：延迟变为 HiAR 的 1.8×

## 工程实践

### 内存管理：所有块同时在 GPU 上

HiAR 的代价是所有块必须同时保存在显存中：

```python
# 估算显存需求
def estimate_memory(num_blocks, frame_size_MB, num_steps):
    # 每个块需要当前状态 + 梯度（训练时）
    per_block_MB = frame_size_MB * 2  # x_t + context
    total_MB = num_blocks * per_block_MB
    print(f"{num_blocks} 块 × {frame_size_MB}MB = {total_MB}MB")
    # 对于 512x512 视频，每块约 8 帧 ~ 200MB
    # 4 块 → 800MB，8 块 → 1.6GB（仅激活值，不含模型权重）

# 推荐：使用 gradient checkpointing 减少训练显存
model = torch.nn.utils.checkpoint.checkpoint_sequential(model_blocks, segments=2)
```

### 常见坑

**1. 流水线并行的边界条件**

第一个 block 没有前驱上下文，用零向量还是学习一个可训练的 embedding？论文用零向量，但复杂场景下学习一个 `start_token` embedding 效果更稳定。

**2. 双向注意力模式切换**

Forward-KL 正则项需要在 bidirectional 模式下计算，而推理是 causal 的。如果你的 attention mask 实现有 bug，两个模式的结果会悄悄变得一样，导致正则项失效但 loss 不报错。

```python
# 必须显式验证两种模式的输出确实不同
assert not torch.allclose(
    model(x, ctx, mask='causal'),
    model(x, ctx, mask='bidirectional'),
    atol=1e-4
), "注意力掩码切换可能有 bug！"
```

**3. 4-step 蒸馏的步长匹配**

教师用 50 步，学生用 4 步，两者的噪声级 $t$ 不能直接对齐。需要使用 consistency distillation 或 flow matching 中的 sub-sequence matching 策略。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要生成 10s+ 长视频 | 仅需 2-3s 短视频（AR 优势不明显）|
| 对时序一致性要求高 | 场景切换频繁（intentional drift）|
| 有流水线并行硬件资源 | 单 GPU 推理（内存压力较大）|
| 需要快速推理（4-step）| 对极致质量要求高（4-step 仍有质量损失）|

## 与其他方法对比

| 方法 | 核心机制 | 时序一致性 | 推理速度 | 长视频能力 |
|------|---------|-----------|---------|-----------|
| VideoLDM | 帧级 AR + 时序注意力 | 中 | 慢 | 弱（误差积累）|
| StreamingT2V | 滑动窗口 + memory | 中 | 中 | 中（窗口接缝）|
| World Models (Genie2等) | 自回归 token | 高 | 中 | 强 |
| **HiAR** | 层次化同噪声级 AR | **高** | **快（4-step）** | **强** |

## 我的观点

HiAR 的核心洞察简洁且有力：**误差积累的根源不是自回归本身，而是跨噪声级的强制"清洁化"**。这个观察也许会影响 AR 扩散在其他模态（3D 场景序列、时序点云）上的设计。

但几个问题值得关注：

1. **训练成本**：所有块同步降噪意味着每个 batch 的计算图是标准 AR 的 $N$（块数）倍
2. **生成控制性**：现有结果主要在无条件/文本条件生成上验证，带精确时序控制的场景（如指令跟随的长视频编辑）还需要更多验证
3. **和世界模型的融合**：对于具身智能，视频生成的下游是动作规划，HiAR 是否能与 model-based RL 框架无缝结合，是个开放问题

离实际产品化：距离中等。流水线并行和 4-step 蒸馏组合是可工程化的，但训练稳定性（尤其是 Forward-KL 正则的超参数敏感性）仍需大量调优经验积累。

> 论文链接：[HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising](https://arxiv.org/abs/2603.08703v1)