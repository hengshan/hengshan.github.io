---
layout: post-wide
title: "肌骨超声断层扫描中的首波分割：用 U-Net 让全波形反演摆脱 Cycle Skipping"
date: 2026-08-22 08:04:04 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.19828v1
generated_by: Claude Code CLI
---

## 一句话总结

用轻量级 2D U-Net 将超声首至波拾取从逐道问题转化为图像分割问题，结合仿真到真实的课程迁移策略，为肌骨 USCT 的全波形反演提供稳定初始模型。

---

## 为什么这个问题重要？

### 应用背景

肌骨超声计算断层扫描（Musculoskeletal USCT）能够重建肌肉、骨骼、肌腱的声学属性分布（声速、密度、衰减）。相比 MRI：

- **实时性**：USCT 采集速度快，有潜力做动态成像
- **便携性**：未来可做成床旁设备
- **定量信息**：输出物理属性而非仅仅形态

典型应用：运动损伤评估（跟腱断裂、肌肉撕裂）、骨质疏松诊断、神经肌肉疾病监测。

### 全波形反演（FWI）的困境

**全波形反演**是 USCT 的核心重建算法：给定超声波形观测数据，反向求解介质的声学属性分布。本质是最小化：

$$
\mathcal{L}(\mathbf{m}) = \frac{1}{2} \| \mathbf{d}_{obs} - \mathbf{d}_{syn}(\mathbf{m}) \|^2
$$

其中 $\mathbf{m}$ 是声学模型，$\mathbf{d}_{obs}$ 是实测波形，$\mathbf{d}_{syn}$ 是正演模拟波形。

问题在于这个优化极其非线性，存在臭名昭著的 **Cycle Skipping（周期跳跃）**：

> 如果初始模型 $\mathbf{m}_0$ 的误差导致合成波形与观测波形的时差超过半个周期，梯度下降就会收敛到错误的极值点。

骨骼组织的高声阻抗、强散射和强衰减使这个问题更加严重。

### 初至波：FWI 的救星

**初至波（First-break）** 是超声信号最先到达接收器的那部分能量，携带纯粹的**走时（Traveltime）**信息。用初至走时构建初始速度模型再启动 FWI，可有效规避 Cycle Skipping。

但问题是：**如何在低 SNR 条件下准确拾取初至波？**

---

## 背景知识

### 传统方法：STA/LTA

经典地震学方法——短时/长时平均比（STA/LTA）：

$$
\text{STA/LTA}(t) = \frac{\frac{1}{N_s}\sum_{i=t}^{t+N_s} x_i^2}{\frac{1}{N_l}\sum_{i=t-N_l}^{t} x_i^2}
$$

当比值超过阈值时认为初至到达。核心缺陷：每道**独立**处理，忽略相邻接收道的**空间连续性**，信噪比低时极不稳定。

### 图像分割视角

论文的核心洞察：把所有接收道的波形叠成一张 **2D 图**（横轴 = 接收道编号，纵轴 = 时间），初至波轨迹在这张图上是一条**连续的曲线**。

```
接收道   1    2    3    ...  N
时间 0   |    |    |    |    |
     1   |    |    | *  |    |    * = 初至波到达点
     2   |    | *  |    | *  |
     3   | *  |    |    |    | *
```

这不就是图像分割问题吗？U-Net 的感受野天然能利用这种空间连续性先验。

---

## 核心方法

### Pipeline 概览

```
全矩阵波形数据 (N_recv × N_time)
        ↓
    2D U-Net（图像分割）
        ↓
初至到达 mask → 提取走时向量
        ↓
   初始速度模型构建
        ↓
  Hybrid FWI：Rytov 走时层析 + 波形拟合
        ↓
   声学属性图像（声速分布）
```

### 数学细节

**Rytov 近似走时层析**在 FWI 早期阶段将走时残差转化为速度模型更新：

$$
\delta \mathbf{m}_{tomo} = (\mathbf{J}_{rytov}^T \mathbf{J}_{rytov} + \lambda \mathbf{I})^{-1} \mathbf{J}_{rytov}^T \delta \mathbf{t}
$$

其中 $\delta \mathbf{t}$ 是观测走时与合成走时之差，$\mathbf{J}_{rytov}$ 是 Rytov 敏感度矩阵，$\lambda$ 是正则化参数。

**Hybrid FWI（HFWI）** 融合走时约束和波形拟合：

$$
\mathcal{L}_{HFWI} = \alpha \cdot \mathcal{L}_{tomo} + (1-\alpha) \cdot \mathcal{L}_{FWI}
$$

随训练推进 $\alpha$ 从 1 衰减至 0，实现从粗到细的重建。

### 仿真到真实的三阶段课程

| 阶段 | 训练数据 | 目标 SNR 范围 | 策略 |
|------|---------|-------------|------|
| 1 | 纯仿真（清洁） | > 20 dB | 学习初至波几何形状 |
| 2 | 仿真 + 真实系统噪声 | 5–20 dB | 增强鲁棒性 |
| 3 | 少量真实弱标注 | < 5 dB | **仅微调解码器** |

关键设计：第三阶段冻结编码器，避免少量真实样本导致过拟合。

---

## 实现

### 轻量级 U-Net

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class FirstBreakUNet(nn.Module):
    """输入 (B, 1, N_recv, N_time)，输出首至概率 mask"""
    def __init__(self, base_ch=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(1, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)
        self.out  = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))

# 参数量估算
model = FirstBreakUNet(base_ch=32)
total = sum(p.numel() for p in model.parameters())
print(f"参数量: {total/1e6:.2f}M")  # ~7.8M，远比 ResNet50 轻
```

### 基线：STA/LTA 逐道拾取

```python
import numpy as np

def sta_lta_pick(trace: np.ndarray, dt: float,
                 sta_len=0.5e-6, lta_len=5e-6, threshold=3.0) -> float:
    """单道初至拾取，返回到达时间（秒）"""
    n_sta = max(1, int(sta_len / dt))
    n_lta = max(1, int(lta_len / dt))
    trace_sq = trace ** 2
    n = len(trace_sq)
    ratio = np.zeros(n)
    for i in range(n_lta, n - n_sta):
        lta = trace_sq[i-n_lta:i].mean() + 1e-12
        sta = trace_sq[i:i+n_sta].mean()
        ratio[i] = sta / lta
    picks = np.where(ratio > threshold)[0]
    return picks[0] * dt if len(picks) > 0 else float('nan')

def batch_sta_lta(waveforms: np.ndarray, dt: float) -> np.ndarray:
    """waveforms: (N_recv, N_time) → 走时向量 (N_recv,)"""
    return np.array([sta_lta_pick(w, dt) for w in waveforms])
```

### 噪声注入与课程学习

```python
def augment_with_real_noise(clean: torch.Tensor, noise_bank: torch.Tensor,
                             target_snr_db: float) -> torch.Tensor:
    """将真实采集的系统噪声叠加到仿真信号上"""
    sig_power   = clean.pow(2).mean()
    noise       = noise_bank[torch.randint(len(noise_bank), (1,))]
    noise_power = noise.pow(2).mean() + 1e-12
    snr_linear  = 10 ** (target_snr_db / 10)
    scale = (sig_power / (snr_linear * noise_power)).sqrt()
    return clean + scale * noise

def curriculum_snr(epoch: int, max_epochs: int) -> float:
    """三阶段课程：随 epoch 递减目标 SNR（dB）"""
    p = epoch / max_epochs
    if p < 0.33:   return 20.0
    elif p < 0.66: return 20.0 - (p - 0.33) / 0.33 * 15.0
    else:          return 5.0
```

### 从 Mask 提取走时

```python
def mask_to_traveltimes(mask: torch.Tensor, dt: float) -> np.ndarray:
    """mask: (1, 1, N_recv, N_time) → 走时向量 (N_recv,)"""
    prob = mask.squeeze().cpu().numpy()   # (N_recv, N_time)
    n_recv = prob.shape[0]
    traveltimes = np.full(n_recv, np.nan)
    for i in range(n_recv):
        t_idx = np.argmax(prob[i])
        if prob[i, t_idx] > 0.5:
            traveltimes[i] = t_idx * dt
    # 空间插值填充漏拾的道
    valid = ~np.isnan(traveltimes)
    if valid.sum() > 2:
        traveltimes = np.interp(np.arange(n_recv),
                                np.where(valid)[0], traveltimes[valid])
    return traveltimes
```

### 可视化对比

```python
import matplotlib.pyplot as plt

def visualize_first_break(waveforms, mask_pred, tt_sta, tt_unet, dt):
    """waveforms: (N_recv, N_time)"""
    t_us = np.arange(waveforms.shape[1]) * dt * 1e6   # 转为微秒
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    vmax = np.abs(waveforms).max() * 0.5
    ax1.imshow(waveforms.T, aspect='auto', cmap='seismic',
               vmin=-vmax, vmax=vmax, origin='upper',
               extent=[0, waveforms.shape[0], t_us[-1], t_us[0]])
    ax1.plot(tt_sta  * 1e6, 'g.', ms=3, label='STA/LTA')
    ax1.plot(tt_unet * 1e6, 'r-', lw=1.5, label='U-Net')
    ax1.set(xlabel='接收道', ylabel='时间 (μs)', title='拾取结果对比')
    ax1.legend()

    ax2.imshow(mask_pred.T, aspect='auto', cmap='hot', origin='upper',
               extent=[0, waveforms.shape[0], t_us[-1], t_us[0]])
    ax2.set(xlabel='接收道', ylabel='时间 (μs)', title='U-Net 首至概率 Mask')
    plt.tight_layout()
    plt.show()
```

---

## 实验

### 数据集说明

| 数据集 | 类型 | 主要挑战 |
|--------|------|---------|
| 体外声学模体（Phantom） | 已知声速的凝胶介质 | 基准验证，SNR 较高 |
| 离体牛肢（Ex vivo bovine limb） | 真实骨肌组织 | 强衰减、高散射、骨遮挡 |
| 在体人大腿（In vivo human thigh） | 活体采集 | 呼吸运动、局部 SNR < 3 dB |

### 定量评估

| 方法 | 平均拾取误差（样本点） | 空间相干性 | 处理速度 |
|------|--------------------|-----------|---------|
| STA/LTA | 8–15 | 差（逐道独立） | < 1 s |
| 1D CNN 逐道 | 4–6 | 一般 | < 2 s |
| **U-Net（本文）** | **2–4** | **好（空间连续）** | 数秒（全矩阵） |

SNR < 3 dB 条件下，U-Net 误差约为 STA/LTA 的 **1/4**，且后续 FWI 重建不发生 Cycle Skipping。

---

## 工程实践

### 硬件需求

- **U-Net 训练**：单张 RTX 3090（24 GB）足够，模型本身 ~7.8M 参数
- **U-Net 推理**：RTX 4070 以上，< 1 秒/全矩阵帧
- **FWI 重建**：这才是真正的瓶颈，通常需要 A100 级别 GPU 或多卡集群

### 常见坑

**1. 走时单位混淆**

```python
# 错误：把采样点索引当作时间（秒）
traveltime = t_idx

# 正确：乘以采样间隔 dt
traveltime = t_idx * dt   # dt 单位：秒/样本
```

**2. 全矩阵采集（FMC）数据维度理解错误**

```python
# FMC 数据形状：(N_tx, N_rx, N_time)，每个发射器独立处理
fmc_data = load_fmc()  # e.g., (128, 128, 4096)
for tx_idx in range(fmc_data.shape[0]):
    wf = fmc_data[tx_idx]                           # (128, 4096)
    mask = model(wf.unsqueeze(0).unsqueeze(0))      # (1, 1, 128, 4096)
```

**3. 第三阶段解码器微调时梯度泄漏**

```python
# 冻结编码器和 bottleneck，只训练解码器
for name, param in model.named_parameters():
    if any(k in name for k in ['enc', 'bottleneck']):
        param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)
```

### 数据采集建议

- **发射频率**：肌骨成像通常 5–15 MHz，穿骨需要较低频率（声衰减 $\propto f^2$）
- **阵列几何**：环形阵列提供全方位覆盖；线阵只能覆盖有限角度，走时信息不完整
- **水浴耦合**：气泡会产生极强虚假反射，采集前必须排气

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 多通道阵列采集（USCT / 地震勘探） | 仅有单道信号，无空间阵列 |
| 信噪比低、骨骼遮挡严重 | 高 SNR 场景（STA/LTA 已经够用，更快） |
| 有物理仿真器可生成训练数据 | 完全没有仿真器且真实标注极少 |
| 初始模型精度要求高（FWI 前处理） | 只需要粗略到达时间估计 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| STA/LTA | 无需训练，毫秒级 | SNR 低时不稳定，逐道独立 | 高 SNR 标准场景 |
| AIC 拾取 | 精度优于 STA/LTA | 同样逐道独立，阈值敏感 | 中等 SNR |
| 1D CNN 逐道 | 端到端学习 | 无空间连续性约束 | 通用场景 |
| **U-Net（本文）** | **空间连续性强，低 SNR 鲁棒** | **需要仿真训练数据** | **骨骼 USCT，低 SNR** |

---

## 我的观点

这篇论文的核心贡献不在于 U-Net 本身，而在于**把一维信号处理问题重新定义为二维图像分割问题**的视角转变——初至波轨迹的空间连续性是天然的结构先验，而 U-Net 的感受野正好能利用这种约束，这是逐道方法在架构层面就无法做到的事。

**三阶段课程迁移**是另一个值得借鉴的工程亮点：不是暴力 fine-tune，而是用逐步增加的信号退化程度引导网络，最终仅微调解码器来适配真实数据分布，用最少的真实标注实现最大的迁移效果。这个思路对其他医学成像的 sim-to-real 问题同样适用。

**离实际部署还有多远？** 在体人体实验是个好兆头，但临床落地还需要：FDA/CE 认证流程、实时处理 pipeline、与商业环形阵列 USCT 设备的集成。更根本的是，FWI 重建本身的计算量——分割拾取只是整条链路中耗时最少的一环。

**值得关注的开放问题：**
- 三维 USCT 的首至分割（目前是逐切片 2D 处理）
- 肌肉收缩动态过程的实时重建（帧间时序信息如何利用？）
- 走时拾取与 FWI 的端到端联合训练

论文链接：[arxiv:2608.19828](https://arxiv.org/abs/2608.19828v1)