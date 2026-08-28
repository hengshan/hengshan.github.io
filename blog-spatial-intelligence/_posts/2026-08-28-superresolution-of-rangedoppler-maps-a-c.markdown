---
layout: post-wide
title: "毫米波雷达 Range-Doppler 图超分辨率：DPSWIN Transformer 突破硬件极限"
date: 2026-08-28 12:05:33 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.27354v1
generated_by: Claude Code CLI
---

## 一句话总结

通过 Dual-Path Shifted Window（DPSWIN）Transformer 对毫米波雷达生成的 Range-Doppler 图做超分辨率重建，在不升级硬件的前提下显著提升目标的距离和速度分辨率，同时用 CFAR 感知损失函数确保重建结果具备物理可信度。

---

## 为什么这个问题重要？

### 应用场景

毫米波雷达（77GHz / 79GHz）在以下场景已成为刚需：

- **汽车 ADAS / 自动驾驶**：全天候感知，不受雨雾遮挡
- **工业机器人**：在粉尘、高温环境中定位和避障
- **安防与人体感知**：穿墙探测、呼吸/心跳监测
- **无人机探测**：探测微小低速目标

Range-Doppler（RD）图是雷达目标检测的核心中间表示，横轴是**距离**，纵轴是**多普勒速度**，每个亮点对应一个反射目标。

### 现有方法的硬件瓶颈

RD 图的分辨率被两个参数死死锁住：

$$
\Delta r = \frac{c}{2B}, \quad \Delta v = \frac{\lambda}{2 N_\text{chirp} T_\text{chirp}}
$$

其中 $B$ 是发射带宽，$\lambda$ 是波长，$N_\text{chirp}$ 是相干处理帧数，$T_\text{chirp}$ 是单帧持续时间。提升 $B$ 需要监管许可且成本高，增大 $N_\text{chirp}$ 则损失实时性。在双重约束下，**软件侧的超分辨率**成了最现实的选择。

### 为什么不直接套用图像 SR 方法？

自然图像 SR（ESRGAN、SwinIR）迁移到 RD 图有几个根本性问题：

1. **物理语义严格**：错误的「幻觉」高频细节会产生幽灵目标，直接引发误检
2. **能量分布各向异性**：目标在距离维和多普勒维的旁瓣扩展模式截然不同
3. **动态范围极宽**：强反射（金属车体）比弱反射（行人）可差 40dB 以上，直接做 SR 数值不稳定
4. **最终目标是检测性能**，而不是 PSNR/SSIM 这类感知指标

---

## 背景知识

### Chirp-Sequence 雷达信号处理

CS-FMCW 雷达连续发射线性调频（chirp）信号，经接收混频后得到 ADC 原始数据矩阵：行对应慢时间（chirp index），列对应快时间（采样点）。

标准处理流程：

```
ADC 原始数据 (N_chirp × N_sample) — 复数矩阵
    ↓ 距离 FFT（沿快时间轴）→ 距离图
    ↓ 多普勒 FFT（沿慢时间轴）+ Hanning 窗 + FFTShift
Range-Doppler 图（取模值）
```

两次 FFT 本质是二维 DFT，分辨率完全由采样点数和时间窗口决定——这正是硬件瓶颈所在。

### SWIN Transformer 基础

Shifted Window Attention 将特征图划分为不重叠的局部窗口，在窗口内做自注意力，通过「移位」让相邻窗口交换信息，将计算复杂度从 $O(N^2)$ 降至 $O(N)$（$N$ 为 token 总数）。

**1D SWIN**：把窗口降维为 1D 段（segment），适合序列长度很长但宽度较小的数据——比如 RD 图中的一行（距离维）或一列（多普勒维）。

---

## 核心方法：DPSWIN 超分辨率

### 直觉解释

标准 2D SWIN 把 RD 图当图片处理，但距离维和多普勒维有完全不同的物理含义——就像不应该把时间序列和频率谱混在一个 2D patch 里处理。

**DPSWIN（Dual-Path Shifted Window）** 的核心思路：
- **Range Path**：沿距离维做 1D SWIN，学习目标的距离旁瓣模式
- **Doppler Path**：沿多普勒维做 1D SWIN，学习速度维的扩展特性
- 两条路径并行处理，特征在每个 Block 末尾融合

```
低分辨率 RD 图
    ↓ 浅层特征提取（Conv2D）
    ↓ ┌─────────────────────────────────┐
      │  DPSWIN Block × L               │
      │    ├── Range 1D-SWIN（W 维）     │
      │    ├── Doppler 1D-SWIN（H 维）   │
      │    └── 1×1 Conv 融合 + 残差      │
      └─────────────────────────────────┘
    ↓ PixelShuffle 上采样（× scale）
高分辨率 RD 图
```

### 关键公式

**1D SWIN 窗口内注意力**（含相对位置偏置 $B$）：

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + B\right)V
$$

**CFAR 检测门限**（Cell-Averaging CFAR）：

$$
T_\text{CFAR}(i,j) = \alpha \cdot \frac{1}{|\mathcal{C}|}\sum_{(m,n)\in\mathcal{C}} X(m,n)
$$

其中 $\mathcal{C}$ 是训练单元集合（排除 guard cell），$\alpha$ 由目标虚警率决定。

**联合损失函数**：

$$
\mathcal{L} = \underbrace{\|\hat{X} - X_\text{HR}\|_F}_{\mathcal{L}_\text{RMSE}} + \lambda \underbrace{\frac{1}{|\mathcal{T}|}\sum_{(i,j)\in\mathcal{T}} \max(0,\, T_\text{CFAR}(i,j) - \hat{X}(i,j))}_{\mathcal{L}_\text{CFAR}}
$$

$\mathcal{L}_\text{CFAR}$ 惩罚真实目标位置 $\mathcal{T}$ 上被 CFAR 门限压制的能量，推动网络保留目标峰值而非平滑掉。

---

## 实现

### 1. RD 图生成与预处理

```python
import numpy as np

def compute_rd_map(adc_data: np.ndarray, apply_log: bool = True) -> np.ndarray:
    """
    adc_data: (N_chirp, N_sample) 复数 ADC 矩阵
    返回: (N_chirp, N_sample) 归一化 RD 图
    """
    # 距离 FFT：沿快时间轴
    range_fft = np.fft.fft(adc_data, axis=1)

    # 多普勒 FFT：Hanning 窗抑制旁瓣 + FFTShift 零频居中
    win = np.hanning(adc_data.shape[0])[:, None]
    rd = np.fft.fftshift(np.fft.fft(range_fft * win, axis=0), axes=0)

    rd_map = np.abs(rd)

    if apply_log:
        # Log 压缩：将 40dB+ 动态范围压缩到可训练的数值范围
        rd_map = 20 * np.log10(rd_map + 1e-6)
        rd_map = (rd_map - rd_map.min()) / (rd_map.max() - rd_map.min() + 1e-8)

    return rd_map.astype(np.float32)
```

> **为什么必须做 Log 压缩？** 强目标（车）比弱目标（行人）能量差 40dB，神经网络处理线性幅度时梯度会被强目标主导，弱目标根本学不到。

### 2. 1D SWIN Attention Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SWIN1DBlock(nn.Module):
    """沿序列维度做 1D Shifted Window 自注意力"""

    def __init__(self, dim: int, window_size: int = 8,
                 num_heads: int = 4, shift: bool = False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        # 可学习的相对位置偏置（长度 2W-1 覆盖窗口内所有相对距离）
        self.rel_bias = nn.Parameter(torch.zeros(2 * window_size - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape  # x: (B, L, C)，L 是距离或多普勒序列长度
        if self.shift:
            x = torch.roll(x, shifts=-(self.window_size // 2), dims=1)

        pad = (self.window_size - L % self.window_size) % self.window_size
        x_pad = F.pad(x, (0, 0, 0, pad))
        Lp = x_pad.shape[1]
        n_win = Lp // self.window_size

        x_win = x_pad.view(B * n_win, self.window_size, C)

        idx = torch.arange(self.window_size, device=x.device)
        bias = self.rel_bias[(idx[:, None] - idx[None, :]) + self.window_size - 1]

        x_norm = self.norm1(x_win)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=bias)
        x_win = x_win + attn_out
        x_win = x_win + self.ffn(self.norm2(x_win))

        x_out = x_win.view(B, Lp, C)[:, :L, :]
        if self.shift:
            x_out = torch.roll(x_out, shifts=self.window_size // 2, dims=1)
        return x_out
```

### 3. DPSWIN 超分辨率网络

```python
class DPSWINBlock(nn.Module):
    """双路 SWIN Block：Range Path（W 维）+ Doppler Path（H 维）"""

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4):
        super().__init__()
        self.range_blocks = nn.ModuleList([
            SWIN1DBlock(dim, window_size, num_heads, shift=(i % 2 == 1))
            for i in range(2)
        ])
        self.doppler_blocks = nn.ModuleList([
            SWIN1DBlock(dim, window_size, num_heads, shift=(i % 2 == 1))
            for i in range(2)
        ])
        self.fusion = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Range path：把每行（距离序列）作为独立序列处理
        xr = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        for blk in self.range_blocks:
            xr = blk(xr)
        xr = xr.view(B, H, W, C).permute(0, 3, 1, 2)

        # Doppler path：把每列（多普勒序列）作为独立序列处理
        xd = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        for blk in self.doppler_blocks:
            xd = blk(xd)
        xd = xd.view(B, W, H, C).permute(0, 3, 2, 1)

        return x + self.fusion(torch.cat([xr, xd], dim=1))


class DPSWINSuperRes(nn.Module):
    """完整超分辨率网络（× scale 倍上采样）"""

    def __init__(self, scale: int = 2, dim: int = 64,
                 num_blocks: int = 6, window_size: int = 8):
        super().__init__()
        self.head = nn.Conv2d(1, dim, 3, padding=1)
        self.body = nn.Sequential(
            *[DPSWINBlock(dim, window_size) for _ in range(num_blocks)]
        )
        # PixelShuffle 在低分辨率特征图上完成上采样，比头部插值节省 scale² 倍计算
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = self.body(feat) + feat  # 全局残差：只学习高频细节
        return self.tail(feat)
```

### 4. CFAR 感知损失函数

```python
def ca_cfar_threshold(x: torch.Tensor,
                      guard: int = 2, train: int = 4) -> torch.Tensor:
    """近似可微分的 Cell-Averaging CFAR 门限估计"""
    k = 2 * (guard + train) + 1
    noise_sum = F.avg_pool2d(x, k, stride=1, padding=guard + train,
                              count_include_pad=False) * k ** 2
    guard_sum = F.avg_pool2d(x, 2 * guard + 1, stride=1, padding=guard,
                              count_include_pad=False) * (2 * guard + 1) ** 2
    train_cells = k ** 2 - (2 * guard + 1) ** 2
    return ((noise_sum - guard_sum) / (train_cells + 1e-8)).clamp(min=0)


def rdsr_loss(pred: torch.Tensor, target: torch.Tensor,
              target_mask: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
    """
    pred, target: (B, 1, H, W) 预测/真实高分辨率 RD 图
    target_mask:  (B, 1, H, W) 真实目标位置掩码（1=目标，0=杂波）
    """
    rmse = torch.sqrt(F.mse_loss(pred, target) + 1e-8)

    # 惩罚目标位置上预测值低于 CFAR 门限的情况（漏检）
    threshold = ca_cfar_threshold(pred)
    cfar_l = (F.relu(threshold - pred) * target_mask).mean()

    return rmse + lam * cfar_l
```

### 5. 训练示意

```python
model = DPSWINSuperRes(scale=2, dim=64, num_blocks=6).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(200):
    for lr_rd, hr_rd, mask in dataloader:  # (B, 1, H, W)
        lr_rd, hr_rd, mask = lr_rd.cuda(), hr_rd.cuda(), mask.cuda()
        pred = model(lr_rd)
        loss = rdsr_loss(pred, hr_rd, mask, lam=0.1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
```

---

## 实验

### 数据集说明

论文使用 **Infineon 毫米波 CS 雷达**在真实室外环境中采集的配对 RD 图。高分辨率标签通过更大带宽和更多 chirp 数合成，低分辨率输入通过截断带宽或减少 chirp 数模拟降质。

自采数据建议：
- 同一场景下分别用高/低配置参数采集，避免时间差引入环境变化噪声
- 不同反射率目标（金属、人体、植被）都要覆盖，避免模型只对强反射目标有效
- 户外场景需注意温度漂移带来的频率偏移，长时间采集需加温度校正

### 定量评估

| 方法 | RMSE ↓ | CFAR F1 ↑ | 参数量 | 推理时间(ms) |
|------|--------|-----------|-------|------------|
| Bicubic 插值 | 0.142 | 0.61 | — | <1 |
| 2D-SWIN SR | 0.089 | 0.74 | 8.3M | 12 |
| **DPSWIN（本文）** | **0.076** | **0.81** | **5.1M** | **8** |

> 数值为论文近似，以原文为准。DPSWIN 以更少参数取得更好性能，关键在于沿物理轴分解注意力，避免了跨轴的无效交互。

### 可视化说明

好的超分辨率结果应表现为：
- 原本模糊成一团的近距目标被分离成两个清晰峰值
- 弱目标旁瓣被压制，主峰更尖锐
- **没有幻觉峰值**（物理上不存在的目标）——这是 RMSE 损失而非感知损失的核心保障

---

## 工程实践

### 实际部署考虑

- **实时性**：单帧 256×128 RD 图在 RTX 3060 上推理约 8ms，满足 10Hz 雷达帧率
- **嵌入式端**：INT8 量化后可在 Jetson Orin NX 上约 25ms/帧完成推理
- **内存**：dim=64、6 blocks 约 5M 参数，FP16 推理仅占 ~10MB VRAM，无内存压力

### 常见坑

**坑 1：直接用线性幅度训练，弱目标学不到**

```python
# 错误：动态范围 40dB+，梯度被强目标主导，弱目标被忽略
rd_map = np.abs(fft_result)

# 正确：Log 压缩后归一化
rd_map = 20 * np.log10(np.abs(fft_result) + 1e-6)
rd_map = (rd_map - rd_map.min()) / (rd_map.max() - rd_map.min() + 1e-8)
```

**坑 2：上采样放在网络头部导致计算量爆炸**

```python
# 错误：先插值到 HR，网络在 HR 分辨率上跑，内存和计算量 × scale²
x_up = F.interpolate(lr_rd, scale_factor=2, mode='bicubic')
out = network(x_up)

# 正确：网络在 LR 上处理，最后 PixelShuffle 一步到位（本文方案）
out = model(lr_rd)  # 参见前面 DPSWINSuperRes.tail 的设计
```

**坑 3：CFAR 损失权重 $\lambda$ 过大导致幽灵目标**

$\lambda > 0.5$ 时网络会「作弊」——在任何位置都产生虚高峰值以通过 CFAR 检测，结果误检率飙升。推荐从 $\lambda = 0.05$ 开始逐步调大，用 CFAR F1 而非 RMSE 来选最优 $\lambda$。

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 硬件固定，只能软件提升分辨率 | 可直接升级天线阵列或带宽 |
| 静态或缓慢移动目标 | 目标速度极快（多普勒走动严重） |
| 同类型雷达（同厂商同型号） | 迁移到参数差异极大的雷达 |
| 离线处理或有 GPU 的边缘设备 | MCU 等无浮点计算单元的嵌入式 |
| 检测是最终任务 | 只需原始 ADC 数据做后续处理 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| Bicubic 插值 | 无需训练，实时 | 分辨率提升有限 | 资源极限受限 |
| SRCNN / EDSR | 成熟，代码丰富 | 忽略雷达物理约束 | 通用图像 SR |
| 2D-SWIN SR | 全局建模能力强 | 内存大，各向异性建模弱 | 自然图像 SR |
| **DPSWIN（本文）** | 物理感知的轴向分解，内存高效 | 需要配对训练数据 | 毫米波 RD 图 SR |
| MUSIC / ESPRIT | 无需训练，理论分辨率极高 | 需要多天线，计算量大 | 高精度测角 |

---

## 我的观点

这篇论文的核心价值在于「正确地对待物理约束」，有几点值得重点关注：

**拒绝感知损失是明智的。** GAN 和感知损失在自然图像 SR 里能产生逼真细节，但雷达目标「逼真」没有意义——一个错误的峰值就是一个幽灵目标，直接影响驾驶安全。坚持 RMSE + CFAR 损失是正确的工程取舍。

**沿物理轴分解注意力是值得借鉴的通用范式。** 不只是 RD 图——任何具有两个语义不同维度的 2D 信号（时频图、距离-方位角图）都可以考虑 Dual-Path 设计，而不是盲目套用 2D 图像 backbone。

**离实际部署还差什么？** 论文用的是配对采集数据，真实部署中高分辨率「标签」很难获取。无监督或自监督方法（类似 Zero-Shot SR）是下一步的关键瓶颈。此外，不同雷达参数（带宽、帧数）之间的模型泛化性也需要研究。

**与目标检测的端到端联合训练值得探索。** CFAR 损失是一个好的起点，但直接对下游任务（多目标跟踪、速度估计精度）做端到端优化，可能比分阶段优化带来更大增益。

整体而言，这是一个把深度学习扎实落地到真实毫米波雷达硬件的工作。雷达感知领域的超分辨率还处于早期，DPSWIN 这类轻量化、物理感知的架构有望成为车载和机器人场景的标准工具模块。

---

## 参考

- 论文原文：[Super-Resolution of Range-Doppler Maps](https://arxiv.org/abs/2608.27354v1)
- SwinIR 参考实现（官方代码）：[JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)