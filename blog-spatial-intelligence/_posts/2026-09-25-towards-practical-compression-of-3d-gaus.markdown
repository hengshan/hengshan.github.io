---
layout: post-wide
title: "COSA-GS：让 3D Gaussian Splatting 压缩真正能跨平台部署"
date: 2026-09-25 12:02:55 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.30245v1
generated_by: Claude Code CLI
---

## 一句话总结

COSA-GS 用"按属性通道因果分解 + 整数化上下文模型"replace 了主流方法依赖的空间上下文聚合，在保持高压缩率的同时，解决了神经压缩里一个经常被忽视但致命的工程问题：**浮点上下文推理在不同平台上算出不同的概率分布，导致算术解码直接崩溃**。

## 为什么这个问题重要？

3DGS 训练出来的场景，动辄几十万到几百万个高斯基元，每个基元携带位置、缩放、旋转、透明度、球谐系数——原始存储经常是几百 MB 到几 GB。这对于：

- **移动端/Web 端渲染**：下载一个场景等于下载一个中等大小的游戏资源包
- **流式 AR/VR**：场景要边传边渲，带宽预算根本不允许裸存
- **多场景云端库**：存储成本随场景数量线性爆炸

现有的压缩思路大多借鉴了图像/视频神经压缩里的"上下文模型"（context model）：用已解码的邻居信息预测当前符号的概率分布，再用算术编码逼近这个概率的信息量。问题是，3DGS 的高斯基元在空间上是不规则分布的，要建立"邻居"关系就得做 KNN、八叉树、哈希网格聚合——这类空间上下文模型（如 ContextGS、HAC 系方法）复杂度高，训练慢，而且更麻烦的是：

**上下文概率是用浮点网络算出来的，CPU 和 GPU、甚至不同 GPU 架构的浮点舍入都可能不一致。** 算术编码要求编码端和解码端算出的概率必须逐比特相同，否则解码端会算出错误的符号边界，直接解码失败或者输出错误内容。这在学术评测里往往被忽略（同一台机器编解码），但一旦要跨平台部署（训练在 GPU，解码在浏览器/手机 CPU），就是硬伤。

COSA-GS 的核心贡献，正是同时解决"压缩率"和"跨平台一致性"这两件事。

## 背景知识

### 3D 表示方式：为什么是 anchor-based 3DGS

原始 3DGS 里每个高斯都是独立参数，冗余度极高（相邻高斯的属性高度相关但没有显式建模）。Scaffold-GS 提出的 **anchor-based** 结构把场景表示为：

- 一组稀疏的 **anchor 点**（有坐标 + 一个紧凑 feature）
- 每个 anchor 通过一个小 MLP 解码出 k 个 neural gaussian 的偏移、透明度、缩放、颜色

这样做的好处是天然把"要压缩的东西"从几百万个独立高斯，降维成几万到几十万个 anchor + 解码器权重，压缩问题变成了"怎么高效编码 anchor 的坐标和属性"。COSA-GS 就是建立在这种 anchor 表示之上的。

### 神经压缩基础：熵编码 + 上下文模型

神经压缩的标准范式（源自 Ballé 等人的 learned image compression）：

1. 把要编码的量（这里是 anchor 属性）看作离散随机变量
2. 训练一个网络预测每个符号的概率分布（通常用高斯分布近似）
3. 用算术编码器按预测的概率把符号编成比特流，概率越准，比特数越少
4. 训练目标是 **rate-distortion**：既要重建质量好（distortion 小），又要码率低（rate 小）

关键约束：**解码时必须用和编码时完全相同的概率分布**，因为算术解码是按累积概率区间逐位剥离符号的，一旦编解码两端算出的 mu/sigma 有哪怕 1e-6 的浮点误差，都可能导致区间判断错位。

## 核心方法

### 直觉解释

主流方法（ContextGS/HAC 一类）的上下文是"空间聚合"式的：预测 anchor A 的属性概率时，要去看空间上邻近的 anchor B、C、D 已经解码的信息，再聚合。这意味着：

- 编解码必须严格按空间遍历顺序进行（类似 PixelCNN 逐像素）
- 邻居查找（KNN/哈希）本身增加复杂度
- 聚合网络通常涉及较深的结构，浮点误差在多层传播中被放大

COSA-GS 换了一个维度做因果分解：**不看空间邻居，只看同一个 anchor 内部、已经解码的前几个属性通道**（比如先解码 x 坐标残差，再基于它解码 y，再基于 x,y 解码 scale……），上下文来自两个稳定的东西：

1. **几何上下文**：直接从这个 anchor 自己的（已量化、无损可得的）坐标算出来，编解码双方都能独立复现，不依赖任何邻居
2. **anchor latent**：一个训练出来的紧凑向量，本身也要被熵编码传输，但一旦解出来就固定了

这样把"空间自回归"简化成"通道内自回归"，网络结构也可以简化成纯线性变换+激活，误差传播路径短得多，这也是后面做整数化推理可行的关键前提。

### 数学细节

给定 anchor 的第 $i$ 个属性通道 $x_i$，其编码概率用高斯分布近似：

$$
P(x_i \mid x_{<i}, c) \approx \int_{x_i-0.5}^{x_i+0.5} \mathcal{N}\big(t; \mu_i(x_{<i}, c),\, \sigma_i(x_{<i}, c)\big)\, dt
$$

其中 $c$ 是几何上下文 + anchor latent 拼接得到的上下文向量，$x_{<i}$ 是同一 anchor 内已解码的前 $i-1$ 个通道——注意这里完全没有空间邻居项。

训练目标是标准 rate-distortion，加上自适应剪枝项：

$$
\mathcal{L} = \mathcal{L}_{\text{render}} + \lambda \Big( \underbrace{\sum_i -\log_2 P(x_i \mid x_{<i}, c)}_{\text{比特率}} + \beta \sum_a \text{sigmoid}(m_a) \Big)
$$

$m_a$ 是每个 anchor 的可学习"保留 mask" logit，训练时用直通估计器把它推向稀疏解，实现自适应剪枝——冗余 anchor 会被压到 mask 接近 0 而被移除，既降低了存储又降低了渲染开销。

### Pipeline 概览

```
Anchor坐标(量化,无损) ──┬─> 几何上下文编码 ──┐
                       │                    ├─> 拼接上下文 c
Anchor feature ────────┴─> anchor latent ───┘
                                              │
        c + 已解码通道 x_{<i} ──> 线性网络 ──> (mu_i, sigma_i)
                                              │
                              算术编码/解码第 i 个通道
                                              │
                    （训练：QAT浮点近似 / 部署：纯整数推理）
```

## 实现

下面的代码是对论文思路的**教学性精简复现**，不是官方实现，重点是让你理解"通道因果分解"和"整数化推理"这两个核心机制。官方代码见 [pengpeng-yu/COSA-GS](https://github.com/pengpeng-yu/COSA-GS)。

### 环境配置

```bash
# PyTorch + 3DGS 常用依赖(以官方仓库为准)
pip install torch torchvision
pip install plyfile tqdm
# 官方实现通常还依赖 diff-gaussian-rasterization 等CUDA光栅化算子
```

### Anchor 表示（Scaffold-GS 风格）

```python
import torch
import torch.nn as nn

class AnchorGS(nn.Module):
    """每个anchor通过小MLP解码出k个neural gaussian的属性"""
    def __init__(self, n_anchors, k=10, feat_dim=32):
        super().__init__()
        self.anchor_xyz = nn.Parameter(torch.randn(n_anchors, 3))
        self.anchor_feat = nn.Parameter(torch.randn(n_anchors, feat_dim))
        self.anchor_scale = nn.Parameter(torch.zeros(n_anchors, 3))
        self.k = k
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim + 3, 128), nn.ReLU(),
            nn.Linear(128, k * (3 + 1 + 3))  # offset, opacity, scale(简化)
        )

    def forward(self):
        n = self.anchor_xyz.shape[0]
        inp = torch.cat([self.anchor_feat, self.anchor_scale], dim=-1)
        raw = self.decoder(inp).view(n, self.k, -1)
        offset, opacity, scale = raw.split([3, 1, 3], dim=-1)
        gaussian_xyz = self.anchor_xyz.unsqueeze(1) + offset * self.anchor_scale.unsqueeze(1)
        return gaussian_xyz, opacity, scale
```

### 核心：Anchor-wise 因果上下文模型（无空间聚合）

```python
class AnchorContextModel(nn.Module):
    """
    COSA-GS核心思路：上下文只来自(1)anchor自身坐标的几何编码
    (2)可学习anchor latent，不依赖任何空间邻居聚合
    """
    def __init__(self, feat_dim=32, latent_dim=16, n_channels=8):
        super().__init__()
        self.n_channels = n_channels
        self.geo_embed = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(), nn.Linear(64, latent_dim)
        )
        self.latent_proj = nn.Linear(feat_dim, latent_dim)
        # 每个通道一个预测头：输入 = 上下文 + 已解码的前i个通道值
        self.channel_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim * 2 + i, 64), nn.GELU(), nn.Linear(64, 2))
            for i in range(n_channels)
        ])

    def build_context(self, anchor_xyz_q, anchor_feat):
        geo_ctx = self.geo_embed(anchor_xyz_q)       # 双端可无损复现
        anchor_latent = self.latent_proj(anchor_feat) # 需要额外熵编码传输
        return torch.cat([geo_ctx, anchor_latent], dim=-1)

    def predict_channel(self, ctx, decoded_prev, i):
        """因果分解：只依赖ctx和本anchor内已解码通道，不看空间邻居"""
        mu, log_sigma = self.channel_heads[i](
            torch.cat([ctx, decoded_prev], dim=-1)
        ).chunk(2, dim=-1)
        return mu, log_sigma.exp().clamp(min=1e-6)
```

### Rate-Distortion 损失 + 自适应剪枝

```python
def gaussian_entropy_bits(x, mu, sigma):
    """用连续高斯近似离散符号概率，估计可微的编码比特数"""
    dist = torch.distributions.Normal(mu, sigma)
    prob = (dist.cdf(x + 0.5) - dist.cdf(x - 0.5)).clamp(min=1e-9)
    return -torch.log2(prob).sum()

def rd_loss(render_out, gt, bits_total, mask_logits, lambda_rd=1e-2, beta=1e-3):
    distortion = torch.nn.functional.mse_loss(render_out, gt)
    rate = bits_total / render_out.numel()
    keep_prob = torch.sigmoid(mask_logits)          # 越接近0，anchor越倾向被剪掉
    sparsity_penalty = keep_prob.mean()
    return distortion + lambda_rd * rate + beta * sparsity_penalty
```

### 量化感知训练 + 整数推理（跨平台 bit-exact 的关键）

```python
class QuantizedLinear(nn.Module):
    """
    训练用STE(直通估计器)模拟量化误差；部署时用纯整数矩阵乘法，
    保证任何平台算出的概率完全一致，避免算术解码崩溃
    """
    def __init__(self, in_dim, out_dim, bits=8):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bits = bits
        self.act_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, integer_mode=False):
        w = self.linear.weight
        qmax = 2 ** (self.bits - 1) - 1
        w_scale = w.abs().max() / qmax
        w_q = torch.round(w / w_scale).clamp(-qmax - 1, qmax)
        if integer_mode:
            x_q = torch.round(x / self.act_scale)
            out_int = torch.matmul(x_q, w_q.t())       # 纯整数域运算
            return out_int * (self.act_scale * w_scale)
        w_ste = w + (w_q * w_scale - w).detach()        # 前向量化,反向直通
        return torch.nn.functional.linear(x, w_ste, self.linear.bias)
```

### 可视化：码率在场景中的分布

```python
import matplotlib.pyplot as plt

def visualize_rate_allocation(anchor_xyz, bits_per_anchor):
    """直观看到细节丰富的区域分配更多码率、冗余区域被剪枝"""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    xyz = anchor_xyz.detach().cpu().numpy()
    bits = bits_per_anchor.detach().cpu().numpy()
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=bits, cmap='viridis', s=3)
    plt.colorbar(sc, label='bits / anchor')
    ax.set_title('Anchor码率分配（细节区域码率更高）')
```

## 实验

### 数据集说明

3DGS 压缩方法通常在标准新视角合成基准上评测：**Mip-NeRF360**（室内外真实场景）、**Tanks & Temples**、**Deep Blending**。这些数据集提供多视角标定图像，训练流程和普通 3DGS 一致（COLMAP 位姿 + 稀疏点云初始化），压缩是在训练完成后（或联合训练中）对 anchor 属性做熵编码，不需要额外采集数据。

### 定量评估（定性结论，具体数值以论文为准）

| 方法 | 上下文来源 | 压缩率 | 解码跨平台一致性 | 模型复杂度 |
|-----|-----------|-------|----------------|-----------|
| Scaffold-GS（无压缩） | — | 基线 | N/A | 低 |
| ContextGS/HAC 类 | 空间邻居聚合 | 高 | ⚠️ 浮点上下文可能不一致 | 高 |
| COSA-GS | 通道内因果 + 整数化 | 高（论文报告达 SOTA） | ✅ bit-exact | 低（纯线性+激活） |

论文报告 COSA-GS 在压缩率上达到 SOTA 水平，同时解码更快，且是几类方法中唯一明确解决跨平台一致性问题的方案。具体 PSNR/SSIM/存储字节数请以原论文表格为准，这里不做二次编造。

## 工程实践（重要！）

### 实际部署考虑

- **解码速度**：因为上下文网络只有线性+激活，没有空间查找，解码端可以做到比空间自回归方法快得多的逐符号推理，这对 Web/移动端流式加载场景很关键
- **内存占用**：anchor latent 本身需要额外传输，latent_dim 是一个需要权衡的超参——太大会抵消压缩收益，太小会让上下文信息不足、码率上升
- **整数推理的精度损失**：8-bit 量化对大场景（高斯数量多、属性动态范围大）可能不够，需要针对不同属性通道（坐标 vs 颜色 vs 透明度）分别设计量化 bit 数和 scale

### 常见坑

1. **浮点上下文跨平台不一致导致解码失败** → 用 QAT + 整数推理彻底避开，不要指望"固定随机种子"或"关闭 TF32"这类临时手段，那些在不同硬件间依然不可靠
2. **过度剪枝导致细节丢失** → 剪枝正则项 β 需要 warm-up，训练初期不要剪太狠，否则会在细节区域造成不可逆的信息丢失
3. **通道因果顺序的选择影响压缩效率** → 通道解码顺序（先编哪个属性）会影响后续通道能利用的上下文量，建议按信息量从大到小排序（比如先编码坐标残差，再编码强相关的 scale/opacity）
4. **量化训练不稳定** → STE 的梯度是近似的，训练初期建议用较低的学习率或先用浮点训练稳定后再切换 QAT

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要压缩后的模型跨设备（训练GPU→浏览器/手机CPU）解码 | 只在单一机器/单一硬件内训练和渲染，没有跨平台需求 |
| 场景规模大、anchor 数量多，空间上下文聚合开销明显 | 场景很小，压缩收益本来就有限 |
| 对解码速度有较高要求（流式加载） | 只追求极限压缩率，不在乎解码速度和一致性 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| Scaffold-GS（无压缩） | 实现简单，anchor结构本身已省了不少冗余 | 存储仍然大 | 基线/快速验证 |
| ContextGS / HAC 类 | 空间上下文利用充分，压缩率高 | 复杂度高，训练慢，跨平台解码有风险 | 单机部署、不需要跨平台 |
| COSA-GS | 结构简单、解码快、bit-exact跨平台一致 | anchor latent仍需额外传输，通道顺序需要调优 | 需要真正落地部署、跨设备解码 |

## 我的观点

这篇工作让我觉得有意思的地方，不是压缩率本身刷了多少个百分点——3DGS 压缩这条赛道这两年已经卷得很厉害了，SOTA 数字每隔几个月就被刷新一轮。真正值得关注的是它把"神经压缩要能实际部署"这件事认真当了一回事：**算术编码要求 bit-exact 一致性，这在学术论文里几乎从不被提及，但在真实产品里是决定"能不能用"的红线**。这和图像/视频神经编解码器工业化过程中踩过的坑是同一个坑——学术评测默认编解码在同一进程里跑完，工程部署默认编码在云端 GPU、解码在客户端各种硬件上跑。

从趋势上看，"用简单的因果分解代替复杂的空间上下文"这个思路，本质上是在用一点点压缩率换取工程可用性和速度，这个权衡在实际产品化里往往是划算的。3DGS 压缩离真正大规模落地（比如做成类似 GLTF 的标准流式格式）还差一个共识：目前各家方法的比特流格式互不兼容，谁能把"压缩率 + 跨平台稳定性 + 标准化格式"三者一起做好，谁就有机会成为事实标准，而不只是又一篇论文。