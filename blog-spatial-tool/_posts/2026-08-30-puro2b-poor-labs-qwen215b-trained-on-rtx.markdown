---
layout: post-wide
title: "用 RTX 5090 花 $7000 训练 2B 大模型：Puro-2B 低成本预训练技术解析"
date: 2026-08-30 08:06:03 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.27370v1
generated_by: Claude Code CLI
---

## 一句话总结

Puro-2B 通过 FP8 精度训练、超球优化（Hyperball）和课程模型平均三项技术，在消费级 RTX 5090 上以不到 $7000 训练出媲美 Qwen2.5-1.5B 的模型——同等规模的 Llama-3.2-3B 需要超过 $150 万。

## 为什么大模型预训练这么贵？

先看几个数据点：

| 模型 | 参数量 | 训练成本 |
|------|--------|---------|
| Llama-3.2-3B | 3B | $1,500,000+ |
| SmolLM3-3B | 3B | $700,000+ |
| Puro-2B（本文） | 2B | $6,900 |

成本主要来自**算力租用费**：H100/A100 集群的机时。大多数团队默认使用 BF16 精度在 H100 上训练，但这里有两个优化机会长期被忽视：

1. **消费级 GPU 比云 GPU 便宜得多**：RTX 5090 的 FP8 峰值算力与 A100 BF16 相当，但购置成本是 $2000 量级
2. **FP8 精度训练让算力利用效率翻倍**：同硬件下，FP8 Tensor Core 吞吐量是 BF16 的 2 倍

但为什么大多数人不用这两个优化？因为**难**。FP8 数值范围极窄（E4M3 最大值只有 448，BF16 是 65504），一不小心梯度爆炸或数值下溢。Puro-2B 的价值在于把这套方法打通成一个可复现的完整方案。

## FP8 训练：深入硬件层理解

### Blackwell 的 FP8 Tensor Core

RTX 5090（Blackwell 架构）的 FP8 Tensor Core 理论吞吐量约为 BF16 的 2 倍。原理直接：Tensor Core 每个周期处理的数据量固定，精度越低，同样的内存装得下的数据越多，计算吞吐越高。

FP8 有两种格式：

- **E4M3**（`float8_e4m3fn`）：4 位指数 + 3 位尾数，最大值 448，精度高但动态范围窄。用于**权重和激活值**（前向传播）
- **E5M2**（`float8_e5m2`）：5 位指数 + 2 位尾数，最大值 57344，动态范围宽但精度低。用于**梯度**（反向传播）

内存节省方面：一个 2B 参数模型，FP8 权重只需 2GB，BF16 需要 4GB，FP32 需要 8GB。这让你可以用更大的 batch size，直接提升硬件利用率。

### 核心挑战：Scale Factor 管理

FP8 最大的坑在于**动态范围太窄**。梯度值如果是 0.001 或者 1000，直接转成 E4M3 都会丢失精度甚至溢出。解决方案是为每个张量维护一个 scale factor：

```python
import torch
import torch.nn as nn

class FP8Linear(nn.Module):
    """FP8 精度线性层，使用延迟缩放（Delayed Scaling）策略"""

    E4M3_MAX = 448.0

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (in_features ** -0.5)
        )
        # amax 通过 EMA 更新，不参与梯度计算
        self.register_buffer('w_amax', torch.ones(1))
        self.register_buffer('x_amax', torch.ones(1))

    def _update_amax(self, tensor, amax_buf, beta=0.99):
        """EMA 更新 amax，避免单个异常值破坏 scale"""
        current_amax = tensor.abs().max().detach()
        amax_buf.mul_(beta).add_(current_amax * (1 - beta))

    def _quantize(self, x, amax):
        scale = (amax / self.E4M3_MAX).clamp(min=1e-12)
        x_scaled = (x / scale).clamp(-self.E4M3_MAX, self.E4M3_MAX)
        return x_scaled.to(torch.float8_e4m3fn), scale

    def forward(self, x):
        self._update_amax(x, self.x_amax)
        self._update_amax(self.weight, self.w_amax)

        x_fp8, x_scale = self._quantize(x.bfloat16(), self.x_amax)
        w_fp8, w_scale = self._quantize(self.weight.bfloat16(), self.w_amax)

        # 反量化后做矩阵乘（生产中用 Transformer Engine 直接走 FP8 GEMM）
        x_dq = x_fp8.bfloat16() * x_scale
        w_dq = w_fp8.bfloat16() * w_scale
        return torch.nn.functional.linear(x_dq, w_dq)
```

**实际生产建议**：上面的代码用于理解原理。生产中应使用 NVIDIA Transformer Engine（`te.Linear`），它内置了 FP8 GEMM 的 CUDA 优化，直接对接 cuBLAS FP8 kernel，实测 MFU 能达到 BF16 的 1.5x–2x。

### 常见错误：直接转换而不加 Scale

```python
# 错误：没有 scale，精度损失严重
x = torch.randn(1024, 1024) * 10
x_fp8 = x.to(torch.float8_e4m3fn)  # 貌似没溢出，但量化误差极大

# 正确：先计算 scale，再量化
scale = x.abs().max() / 448.0
x_fp8 = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
```

## Hyperball 优化：让 FP8 训练不爆炸

### 为什么 Adam 在 FP8 下容易出问题？

Adam 依赖梯度的二阶矩 $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ 来自适应学习率。当梯度本身带有 FP8 量化误差时，这个估计会系统性偏移，导致某些参数的有效学习率失控。

### Hyperball 的核心思路

**只用梯度的方向，丢弃幅值**。把每步梯度更新投影到单位超球面（unit hypersphere）：

$$g_t^{\text{norm}} = \frac{g_t}{\|g_t\|_2}$$

这和 Lion 优化器的 `sign(g_t)` 策略类似（都只用方向），但超球投影保留了完整的向量方向信息，而不是逐元素取符号。

```python
class HyperballSGD(torch.optim.Optimizer):
    """
    将梯度投影到单位超球面后更新。
    完全回避二阶统计量，适合 FP8 低精度训练。
    """

    def __init__(self, params, lr=1e-4, weight_decay=0.1, eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                # 关键：将梯度归一化到单位超球面
                g_norm = torch.norm(g).clamp(min=group['eps'])
                g_normalized = g / g_norm

                p.mul_(1.0 - lr * wd)          # 解耦 weight decay
                p.add_(g_normalized, alpha=-lr) # 用归一化梯度更新
```

代价是：由于幅值信息被丢弃，学习率的选择比 Adam 更关键——建议使用 cosine schedule，并在前 1% 步做 warmup。

## 课程模型平均：数据配方 × 权重融合

Puro-2B 的另一项技术是**课程模型平均（Curriculum Model Averaging）**，流程如下：

1. 设计多个数据课程（不同领域配比：偏重代码、偏重数学、均衡混合等）
2. 从相同的 checkpoint 出发，分别用不同课程继续训练数千步
3. 对多个最终 checkpoint 的权重做**加权平均**

```python
def curriculum_model_average(state_dicts, weights=None):
    """
    对多个课程训练的 checkpoint 做加权平均。
    从同一 checkpoint 出发的模型在 loss landscape 上线性连通，
    平均后的模型泛化性通常优于任意单个模型。
    """
    n = len(state_dicts)
    if weights is None:
        weights = [1.0 / n] * n

    avg_state = {}
    for key in state_dicts[0].keys():
        avg_tensor = torch.zeros_like(
            state_dicts[0][key], dtype=torch.float32
        )
        for sd, w in zip(state_dicts, weights):
            avg_tensor.add_(sd[key].float(), alpha=w)
        avg_state[key] = avg_tensor.to(state_dicts[0][key].dtype)

    return avg_state
```

**为什么有效？** 不同课程训练的模型处于 loss landscape 的不同区域，但由于出发点相同，这些区域在权重空间中通常"线性连通"（路径上没有高 loss 峡谷）。平均后的模型往往比任何单个模型的下游任务泛化性更好——这是 Model Soup（Wortsman et al., 2022）的核心结论，Puro-2B 将其应用到了课程选择上。

## 成本是怎么算出来的？

$$\text{Cost} = T_{\text{wall}} \times N_{\text{GPU}} \times \text{Cost per GPU-hour}$$

Puro-2B 报告的 $6.9K 来自自有 RTX 5090 机器的电费和折旧，而非云端 GPU 租用。FP8 加速让相同计算量的挂钟时间缩短约 40%–50%，直接体现在成本上。

论文进一步拟合了 **Puro Cost Scaling Law**：约 $4400 就能达到 Qwen2-1.5B 的性能，给出了"给定预算能训出什么水平"的量化依据。

## 性能对比

（基于论文评测数据）

| 对标模型 | 参数量 | 训练成本 | 性能 |
|---------|--------|---------|------|
| Qwen2-1.5B | 1.5B | 未公开（估计 $100K+） | 基准 |
| Qwen2.5-1.5B | 1.5B | 未公开 | 略高于 Qwen2 |
| Puro-2B（$4.4K 版） | 2B | $4,400 | ≈ Qwen2-1.5B |
| Puro-2B（最优版） | 2B | $6,900 | 接近 Qwen2.5-1.5B |

## 适用场景 / 局限性

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有 RTX 40/50 系 GPU（支持 FP8） | Volta/Ampere 之前的老卡（无 FP8 Tensor Core） |
| 预算有限（$1K–$50K 量级） | 需要快速出结果（调试 FP8 本身费时） |
| 需要完全掌控预训练数据 | 目标是 7B+ 模型（消费卡显存不够） |
| 研究预训练数据对后训练的影响 | 已有适合任务的 open-weight 模型可直接微调 |

## 调试 FP8 训练的关键步骤

1. **先跑 BF16**：验证完整训练流程和 loss 曲线，再切 FP8
2. **监控 amax 收敛**：如果 loss 爆炸，先检查 scale factor 的 EMA 是否在前几百步收敛
3. **验证 kernel 真的触发**：用 Nsight Systems 确认 FP8 GEMM kernel 实际调用，不要只看理论吞吐
4. **打出 gradient norm**：持续上升说明 Hyperball 归一化没有正确生效

典型坑：

- 不同层的 scale factor 收敛速度不同，embedding 层通常需要单独处理（建议保持 BF16）
- LayerNorm 的 running statistics 必须保持 FP32 精度
- Gradient checkpoint 和 FP8 同时开启时，重计算的量化会引入额外误差，需要测试是否影响收敛

## 延伸阅读

- **Puro-2B 官方代码和权重**：[https://huggingface.co/collections/thu-pacman/puro-2b](https://huggingface.co/collections/thu-pacman/puro-2b)，数据、代码、权重 Apache 2.0 开源
- **NVIDIA Transformer Engine**：FP8 训练的工业级实现，支持 PyTorch/JAX，是生产环境的首选
- **Model Soup**（Wortsman et al., 2022）：权重平均的理论基础，解释了为什么平均后模型更好
- **Scaling Laws for Neural Language Models**（Kaplan et al., 2020）：理解 token 预算和模型大小之间的权衡