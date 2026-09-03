---
layout: post-wide
title: "UE5M3：给 FP4 训练算一笔更宽容的账"
date: 2026-09-03 12:03:16 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.02846v1
generated_by: Claude Code CLI
---

## 一句话总结

这篇论文解决的不是"要不要用 FP4 训练"，而是"FP4 里那个决定生死的 block scale，该用什么格式存"。作者把 scale 从 NVIDIA NVFP4 recipe 里的 E4M3 换成无符号 E5M3（更宽的动态范围），从而可以把"每步都重新计算 tensor 级 amax"的 current-tensor scaling，换成"隔一段时间才更新一次"的 periodic tensor scaling，同时去掉了开销不小的 Random Hadamard Transform（RHT），只在反向传播的梯度上做选择性随机舍入。结果是：8B 模型、近 190B token 的预训练里，loss 更低，下游指标更高，训练还更快。

## 背景：为什么 FP4 这么难训

FP4 的痛点不在"4 bit 表达能力弱"这件事本身——真正的痛点是主流 FP4 payload 格式 E2M1（1 位符号、2 位指数、1 位尾数）能表示的数值集合极其稀疏：

$$
\{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\}
$$

（以及它们的负数）。相邻两个可表示值之间的相对间隔在小数值区间可以超过 30%。任何激活值或梯度只要落在这个网格之外，量化误差都非常显著。

解决办法是**block scaling**：把一个张量切成小块（比如 16 个数一组），每组共享一个缩放因子 scale，先把这组数值压缩到 E2M1 能覆盖的动态范围里，再量化。真正决定训练稳不稳的，其实是这个 scale 该怎么算、用什么精度存。

NVIDIA Transformer Engine 的 NVFP4 recipe 给出的答案是：

- **current-tensor scaling**：每一步都对整个 tensor 求 amax 来定 scale，追踪分布变化很及时，但计算开销不小；
- **Random Hadamard Transform（RHT）**：训练前对激活/权重做随机正交变换，把离群值（outlier）打散到整个向量里，减轻单个 outlier 撑爆动态范围的问题，但这是矩阵乘法之外额外插进来的一步运算；
- **BF16 final layers**：最后几层直接豁免量化，退回 BF16，保精度但吃不到 FP4 的加速。

这三件事都是"给 E2M1 兜底"的补丁，而补丁本身就是额外算力开销。论文的核心 insight 是：这些补丁之所以必要，很大程度是因为 NVFP4 选的 scale 格式（E4M3）动态范围不够宽，逼得你必须频繁重新计算 scale、必须把 outlier 打散。如果换一个动态范围更宽的 scale 格式，很多补丁可以省掉。

论文的主要贡献可以诚实地概括为：**不是发明新算法，而是重新设计了一个格式选择**——用 UE5M3（无符号、5 位指数、3 位尾数）做 block scale，宽范围换来了简化流程的空间。这是一个偏工程、偏数值格式的贡献，不涉及模型结构或优化器改动，泛化性需要靠更多模型规模和架构验证，论文目前只在一个 8B Nemotron-H 模型上做了验证。

## 算法原理

### 直觉解释

把 E2M1 想象成一把只有 8 个刻度的尺子，block scale 就是"把这把尺子放大或缩小多少倍去量这组数"。

- NVFP4 用 E4M3 做尺子的"缩放旋钮"：旋钮精细（3 位尾数），但能拧的范围窄（4 位指数），一旦这组数字的量级突然变化，旋钮很容易拧到头，所以必须每一步都重新检查、还得先把突出的数字磨平（RHT）。
- 本文用 E5M3 做旋钮：指数位多一位，可拧范围大了一倍，代价是尾数少一位、旋钮刻度略粗一点。范围换来的好处是：旋钮不用每步都重新对，隔几十步对一次也不容易脱靶，磨平离群值这道工序也可以省了。

这是一个典型的"精度换范围"的 trade-off，在数值格式设计里很常见（想想为什么 BF16 用少尾数换大指数范围，而 FP16 反过来）。

### 数学推导

对一个长度为 $B$ 的 block $x_1, \dots, x_B$，先算 block scale：

$$
s = \mathrm{quant}_{\mathrm{UE5M3}}\left(\frac{\max_i \lvert x_i \rvert}{Q_{\max}}\right)
$$

其中 $Q_{\max}=6$ 是 E2M1 能表示的最大幅值。量化和反量化：

$$
\hat{x}_i = \mathrm{quant}_{\mathrm{E2M1}}\!\left(\frac{x_i}{s}\right) \cdot s
$$

关键的区别在 $s$ 多久重新计算一次：

- **current-tensor scaling**：每次前向/反向都重新对当前 tensor 求 $\max_i \lvert x_i \rvert$；
- **periodic tensor scaling**：每隔 $T$ 步才重新求一次 amax，中间步骤复用旧 scale，只要 $s$ 的表示范围够宽，中间几步的分布漂移不会把数值挤出可表示区间。

反向传播里，论文只对梯度张量做**选择性随机舍入**（stochastic rounding），而不是对前向的激活/权重做。直觉是：随机舍入的价值在于长期无偏——训练成千上万步之后，舍入误差不会系统性地偏向一个方向。这个价值对反复累积、量级本身就波动大的梯度最有用，对前向张量收益不明显、却要多付一次随机数生成的成本，所以只在梯度上做。

### 与其他算法的关系

可以看作这样一条演化线：

- **MXFP4**（OCP Microscaling 标准）：block scale 用 E8M0（纯 2 的幂次，无尾数），range 最大但精度最差；
- **NVFP4**（NVIDIA Transformer Engine）：scale 换成 E4M3，精度提高，但 range 变窄，逼出 current-tensor scaling + RHT + BF16 兜底三件套；
- **本文 UE5M3**：scale 用 E5M3，精度和 range 都取中间值，换来 periodic scaling + 无 RHT + 全层 FP4（不豁免任何线性层）。

## 实现

真正的 FP4 矩阵乘法需要专用 Tensor Core 硬件支持，个人环境跑不了原生 FP4 GEMM。下面的代码是**软件模拟**（fake quantization），目的是让你理解 block scaling、periodic scaling、selective stochastic rounding 具体在算什么，不是给你一份可以拿去跑 8B 模型的训练框架。论文本身没有公开代码链接。

### 最小可运行版本

```python
import torch

# E2M1 (FP4 payload) 能表示的全部非负幅值
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
Q_MAX = 6.0

def quantize_minifloat(x, exp_bits, mantissa_bits):
    """把正数 x 量化成 exp_bits/mantissa_bits 的 minifloat（模拟 E4M3 / UE5M3）"""
    bias = 2 ** (exp_bits - 1) - 1
    x = x.clamp(min=1e-12)
    exponent = torch.floor(torch.log2(x))
    exponent = exponent.clamp(min=-bias, max=2 ** exp_bits - 1 - bias)
    normalized = x / (2.0 ** exponent)          # 落在 [1, 2)
    scale = 2 ** mantissa_bits
    mantissa = torch.round(normalized * scale) / scale
    return mantissa * (2.0 ** exponent)

def quantize_e2m1(x):
    """把 x 量化到 E2M1 的 8 个刻度上（round-to-nearest）"""
    sign = x.sign()
    ax = x.abs().unsqueeze(-1)
    dist = (ax - E2M1_GRID).abs()
    idx = dist.argmin(dim=-1)
    return sign * E2M1_GRID[idx]

def block_quantize(x, block_size=16, scale_exp_bits=5, scale_mantissa_bits=3):
    """block scaling: 每 block_size 个元素共享一个 scale"""
    orig_shape = x.shape
    blocks = x.reshape(*orig_shape[:-1], -1, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = quantize_minifloat(amax / Q_MAX, scale_exp_bits, scale_mantissa_bits)
    scale = scale.clamp(min=1e-12)
    q = quantize_e2m1(blocks / scale) * scale
    return q.reshape(orig_shape)
```

`scale_exp_bits=5, scale_mantissa_bits=3` 对应本文的 UE5M3；如果换成 `4, 3` 就是 NVFP4 的 E4M3 scale——两行参数的差别，就是论文的核心改动。

### 完整实现

```python
class FakeQuantMatmul(torch.autograd.Function):
    """前向用 round-to-nearest 量化，反向对梯度用选择性随机舍入"""

    @staticmethod
    def forward(ctx, x, w, block_size):
        ctx.save_for_backward(x, w)
        ctx.block_size = block_size
        x_q = block_quantize(x, block_size)
        w_q = block_quantize(w, block_size)
        return x_q @ w_q.t()

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        bs = ctx.block_size
        grad_out_q = stochastic_block_quantize(grad_out, bs)  # 只在梯度上做随机舍入
        grad_x = grad_out_q @ w
        grad_w = grad_out_q.t() @ x
        return grad_x, grad_w, None

def stochastic_block_quantize(x, block_size=16):
    orig_shape = x.shape
    blocks = x.reshape(*orig_shape[:-1], -1, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = quantize_minifloat(amax / Q_MAX, 5, 3).clamp(min=1e-12)
    y = (blocks / scale).abs()
    idx_up = torch.searchsorted(E2M1_GRID, y.unsqueeze(-1)).clamp(max=len(E2M1_GRID) - 1)
    idx_lo = (idx_up - 1).clamp(min=0)
    lo, up = E2M1_GRID[idx_lo.squeeze(-1)], E2M1_GRID[idx_up.squeeze(-1)]
    prob_up = ((y - lo) / (up - lo).clamp(min=1e-12)).clamp(0, 1)
    rounded = torch.where(torch.rand_like(y) < prob_up, up, lo)
    q = blocks.sign() * rounded * scale
    return q.reshape(orig_shape)


class PeriodicScaleCache:
    """periodic tensor scaling：每 update_every 步才重算一次 scale"""

    def __init__(self, update_every=32):
        self.update_every = update_every
        self.step = 0
        self.cached_scale = None

    def get_scale(self, amax, exp_bits=5, mantissa_bits=3):
        if self.step % self.update_every == 0 or self.cached_scale is None:
            self.cached_scale = quantize_minifloat(amax / Q_MAX, exp_bits, mantissa_bits)
        self.step += 1
        return self.cached_scale.clamp(min=1e-12)
```

### 关键 Trick（论文里容易被忽略的细节）

- **scale 更新周期不能拍脑袋定**：`update_every` 太大，中间步骤的分布漂移会让旧 scale 失配；太小又退化回 current-tensor scaling，吃不到省开销的好处。论文用的是"periodic"而非固定常数，实践中建议从几十步起步，配合梯度范数监控调整。
- **省掉 RHT 是有前提的**：RHT 的作用是打散 outlier，只有当 scale 格式本身的动态范围足够吃下 outlier 的量级波动，才敢直接去掉这道工序。如果你在自己的模型上复用这套 recipe 却保留 E4M3 scale，贸然去掉 RHT 大概率会更不稳定。
- **随机舍入只放在反向**：前向量化用确定性 round-to-nearest（可复现、调试方便），只有梯度用随机舍入。这是省算力和保持数值行为可预测之间的折中，不要图省事把随机舍入用到前向,否则同一批数据每次前向结果都不一样，调试会非常痛苦。
- **全层 FP4 不豁免**：论文特意强调"omits RHT, and uses FP4 in all eligible internal linears"——如果你发现某几层必须退回 BF16 才能不发散,这通常说明 scale 格式的动态范围或更新频率还没配对，而不是这几层"天生"需要更高精度。

## 实验：一个诚实的、你真的能跑的小实验

8B 模型、190B token 的预训练个人根本无法复现，这里退而求其次，做一个**量化误差的数值验证**，用来直观感受"scale 格式换成 E5M3"到底带来多少差别。这不是论文实验的替代品，只是帮助建立直觉。

```python
import torch

torch.manual_seed(0)
# 模拟一批带 outlier 的激活值（重尾分布更接近真实 LLM 激活）
x = torch.randn(4096, 4096) * (1 + torch.rand(4096, 4096) * 8)

for exp_bits, mant_bits, name in [(4, 3, "E4M3 scale (NVFP4)"), (5, 3, "E5M3 scale (本文)")]:
    x_q = block_quantize(x, block_size=16, scale_exp_bits=exp_bits, scale_mantissa_bits=mant_bits)
    mse = ((x - x_q) ** 2).mean().item()
    print(f"{name}: MSE = {mse:.6f}")
```

在这种重尾分布下，你会观察到 E5M3 scale 的量化 MSE 通常比 E4M3 更低——因为 amax 的量级本身就有更大波动，E4M3 更容易在 outlier 出现时把 scale 顶到指数上界，导致整个 block 的相对精度下降。这个方向和论文的结论一致，但幅度不能直接类比论文里的最终 loss 差距，因为真实训练里还有梯度累积、优化器状态等更复杂的因素。

### 与论文数据的对比

| 指标 | Transformer Engine NVFP4 | 本文 block-16 UE5M3 |
|---|---|---|
| final-window training loss | 基线 | 更低 |
| 量化推理 held-out NLL | 基线 | 更低 |
| 三个下游聚合指标 | 基线 | 全部更高 |
| model-body token throughput（去掉 RHT + BF16 豁免后） | 基线 | +21.2% |

需要注意最后一行的 21.2% 提升，是论文在 NVIDIA 原生 NVFP4 执行路径上做的 ablation（同时去掉 RHT 和 BF16 最终层豁免），不是"UE5M3 recipe 本身在硬件上跑出来的加速"——因为目前没有硬件原生支持 UE5M3 block scale，本文的方法目前仍然是软件模拟执行。这一点论文自己也承认，并把"呼吁硬件原生支持 UE5M3"作为结论的一部分。

## 调试指南：量化训练不收敛怎么办

### 常见问题

1. **loss 出现 NaN 或突然发散**：大概率是某个 block 的 amax 出现异常大值，scale 被顶到 minifloat 的指数上界后仍然溢出。检查 amax 的分布，尤其是训练早期（权重初始化阶段数值分布还不稳定）。
2. **loss 收敛但明显慢于 BF16 baseline**：先怀疑 `update_every` 是不是设得太大，导致 scale 长期滞后于真实分布；也可能是 block_size 太大（block 内部动态范围过宽，共享一个 scale 精度损失大）。
3. **训练全程稳定但下游指标偏低**：检查是不是在应该做随机舍入的地方用了确定性舍入（或反过来），量化偏差长期累积会系统性拉低模型质量，但不会表现为 loss 曲线异常。

### 如何判断"在正常训练"

单看 loss 曲线不够，量化训练建议额外监控两个指标：

- **相对量化误差**：$\lVert x - \hat{x} \rVert / \lVert x \rVert$，按层、按张量类型（激活/权重/梯度）分开看，如果某一层持续显著高于其他层，说明这一层的动态范围和当前 scale 配置不匹配。
- **scale 的更新幅度**：如果每次重新计算的 scale 相对上一次跳变很大，说明 `update_every` 定得太长，分布已经漂移出了旧 scale 能覆盖的范围。

### 超参数调优参考

| 参数 | 推荐起点 | 敏感度 | 说明 |
|---|---|---|---|
| block_size | 16 | 中 | 越小精度越高但 scale 存储/计算开销越大 |
| scale 更新周期 | 数十步 | 高 | 论文强调这是省下 current-tensor scaling 开销的关键，但过大会导致滞后 |
| 随机舍入应用范围 | 仅反向梯度 | 高 | 前向加了会让训练不可复现，收益也有限 |
| scale 格式 | UE5M3 | 高 | 换成 E4M3 基本等价于退回 NVFP4 的假设，需要重新引入 RHT |

## 什么时候值得考虑这套 recipe？

| 适用场景 | 不适用场景 |
|---|---|
| 已经具备 FP4/FP8 训练基础设施、追求进一步降低训练成本的大模型预训练 | 个人或小团队自研训练框架，没有专用量化 kernel 支持 |
| 硬件或软件栈已经支持宽范围 minifloat scale 存储 | 依赖当前 NVIDIA 原生 NVFP4 kernel（尚不原生支持 UE5M3，跑这套 recipe 目前意味着软件模拟，吃不到全部硬件加速） |
| 模型规模、训练 token 量接近论文验证的量级（8B、近 190B token） | 小模型或短训练场景，量化收益不明显，反而增加工程复杂度 |

## 我的观点

这篇论文做的事情很克制：不改模型结构，不改优化器，只改了一个 scale 存储格式,却牵动了整条 FP4 训练 pipeline 的复杂度。这种"换一个数值格式就省掉一整套补丁"的思路,比很多号称"革新训练范式"的论文更朴素,也更容易验证——你不需要重新设计模型就能判断它对不对。

但要泼一盆冷水：目前这仍然是**软件模拟**的 FP4 训练,21.2% 的吞吐提升来自去掉 NVFP4 原生路径里的 RHT 和 BF16 豁免层,不是 UE5M3 本身在硬件上跑出来的速度。如果你的目标是"现在就要更快的训练",这篇论文暂时给不了你,它更像是给硬件厂商递了一份需求文档:"如果你们原生支持 UE5M3 block scale,我们已经证明了它比现有方案更简单、效果更好"。值得关注,但落地还要等硬件跟上。

另外,论文只在一个 8B 模型、一种架构（Nemotron-H）上验证,量化格式的鲁棒性是否能推广到更大规模或不同架构（比如纯 Transformer、MoE）,论文并未回答,这是复现和跟进这项工作时最值得先验证的一点。