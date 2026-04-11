---
layout: post-wide
title: "MonoUNet：用可训练单义信号特征打造边缘端超轻量医学图像分割"
date: 2026-04-11 08:05:20 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.07780v1
generated_by: Claude Code CLI
---

## 一句话总结

MonoUNet 将经典信号处理中的"单义信号理论"变成可微的神经网络组件，在参数量仅为同类轻量模型 1/10–1/700 的情况下，实现了跨设备鲁棒的膝关节软骨超声分割。

---

## 为什么这篇论文重要？

### POCUS 的跨设备泛化困境

手持式床旁超声（POCUS）正在快速普及，但不同厂商、不同档次的设备产生的图像风格差异极大——高端推车式设备信噪比高、图像清晰；手持式设备噪声多、伪影明显。

一个在推车式设备上训练的分割模型，搬到手持设备上精度往往骤降 5–10%。这不是边缘情况，而是 POCUS 规模化部署的核心障碍。

### 现有轻量化方案的盲区

MobileNetV2-UNet、EfficientUNet 等方案的压缩逻辑是"更少通道 + 深度可分离卷积"。代价是：模型对输入分布高度敏感——图像增益变了，精度就掉。没有人问过：**我们能不能把"对亮度不敏感的特征"作为先验嵌入网络？**

MonoUNet 问了这个问题。答案是局部相位特征。

### 核心洞见：相位比幅度更稳定

软骨边界在不同设备下，**亮度（幅度）会变**，但从暗到亮的**相位结构**是稳定的。单义信号理论恰好能分离这两者。论文的创新不是"做了个小 U-Net"，而是**把经典信号处理中的固定 Riesz 变换变成了可学习的组件**，让网络自适应地找到最有鉴别力的局部相位表示。

---

## 核心方法解析

### 单义信号的直觉与数学

对一张图像 $f(\mathbf{x})$，单义信号分解出三个量：

$$A(\mathbf{x}) = \sqrt{f^2 + \|\mathbf{R}f\|^2}$$

$$\phi(\mathbf{x}) = \arctan\!\left(\frac{\|\mathbf{R}f\|}{f}\right)$$

$$\theta(\mathbf{x}) = \arctan\!\left(\frac{R_2 f}{R_1 f}\right)$$

其中 $\mathbf{R}f = (R_1f, R_2f)$ 是 **Riesz 变换**（二维 Hilbert 变换），频域定义为：

$$\hat{R}_i f(\mathbf{u}) = -j\,\frac{u_i}{|\mathbf{u}|}\,\hat{f}(\mathbf{u}), \quad i=1,2$$

关键：$\phi$ 和 $\theta$ 对图像整体亮度变化**不敏感**，这正是我们需要的跨设备鲁棒性来源。论文让带通滤波器（施加于 Riesz 变换之前）变成可学习参数，赋予网络自适应调整相位敏感尺度的能力。

### 架构设计

```
输入图像
   │
   ├──→ [MonogenicBlock] → 多尺度局部相位特征 (3个尺度)
   │
   └──→ [极度压缩的 U-Net 编码器]
            enc1 (8ch)  ──→ [GatedInjection] ←── 细尺度相位
            enc2 (16ch) ──→ [GatedInjection] ←── 中尺度相位
            enc3 (32ch) ──→ [GatedInjection] ←── 粗尺度相位
            │
            非对称解码器（仅1层）→ 分割输出
```

两个关键设计决策：
- 基础通道数从标准 U-Net 的 64 压缩到 **8**，解码器仅保留**一层**
- 相位特征通过**残差门控**注入，而非简单拼接——门控值决定"相位信息贡献多少"

---

## 动手实现

### 可微 Riesz 变换

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RieszTransform(nn.Module):
    """频域实现可微 Riesz 变换，无可学习参数"""
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        X = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2+1], complex

        fy = torch.fft.fftfreq(H, device=x.device).view(-1, 1)   # [H, 1]
        fx = torch.fft.rfftfreq(W, device=x.device).view(1, -1)  # [1, W//2+1]
        eps = 1e-8
        r = torch.sqrt(fx**2 + fy**2) + eps

        h1 = (-1j * fx / r).to(X.dtype)  # x 方向 Riesz 核
        h2 = (-1j * fy / r).to(X.dtype)  # y 方向 Riesz 核

        R1 = torch.fft.irfft2(X * h1, s=(H, W), norm='ortho')
        R2 = torch.fft.irfft2(X * h2, s=(H, W), norm='ortho')
        return R1, R2
```

### 多尺度单义特征模块

```python
class MonogenicBlock(nn.Module):
    """多尺度可训练单义特征：带通滤波器可学习，Riesz 变换固定"""
    def __init__(self, in_ch: int, scales: int = 3):
        super().__init__()
        self.riesz = RieszTransform()
        # 可学习的多尺度带通滤波器（深度卷积，不改变通道数）
        self.bp = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch, kernel_size=2**(s+1)+1,
                      padding=2**s, groups=in_ch, bias=False)
            for s in range(scales)
        ])

    def _monogenic_features(self, f, R1, R2):
        amp    = torch.sqrt(f**2 + R1**2 + R2**2 + 1e-8)
        phase  = torch.atan2(torch.sqrt(R1**2 + R2**2 + 1e-8), f)
        orient = torch.atan2(R2, R1 + 1e-8)
        return torch.cat([amp, phase, orient], dim=1)  # [B, 3C, H, W]

    def forward(self, x):
        feats = []
        for bp in self.bp:
            band = x - bp(x)       # 带通响应（可学习）
            R1, R2 = self.riesz(band)
            feats.append(self._monogenic_features(band, R1, R2))
        return feats  # 3 × [B, 3*in_ch, H, W]
```

### 门控特征注入

```python
class GatedFeatureInjection(nn.Module):
    """残差门控：相位特征软增强编码器特征，而非硬替换"""
    def __init__(self, enc_ch: int, phase_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(phase_ch, enc_ch, 1, bias=False)
        self.gate = nn.Conv2d(enc_ch, enc_ch, 1)
        self.norm = nn.BatchNorm2d(enc_ch)
        # 初始化 gate bias 为负值，使初始注入量接近零
        nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, enc_feat, phase_feat):
        gate = torch.sigmoid(self.gate(self.proj(phase_feat)))
        return self.norm(enc_feat * (1.0 + gate))
```

### MonoUNet 完整结构

```python
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )

class MonoUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=8):
        super().__init__()
        b = base
        self.enc1 = conv_block(in_ch, b)
        self.enc2 = conv_block(b,     b*2)
        self.enc3 = conv_block(b*2,   b*4)
        self.pool = nn.MaxPool2d(2)

        phase_ch = 3 * in_ch  # amp + phase + orient
        self.mono = MonogenicBlock(in_ch, scales=3)
        self.gi1  = GatedFeatureInjection(b,   phase_ch)
        self.gi2  = GatedFeatureInjection(b*2, phase_ch)
        self.gi3  = GatedFeatureInjection(b*4, phase_ch)

        # 非对称解码器（仅 2 层，无 bottleneck）
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = conv_block(b*4 + b*2, b*2)
        self.dec2 = conv_block(b*2 + b,   b)
        self.head = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        phases = self.mono(x)  # 全部在原始分辨率计算

        e1 = self.gi1(self.enc1(x),                phases[0])
        e2 = self.gi2(self.enc2(self.pool(e1)),     F.avg_pool2d(phases[1], 2))
        e3 = self.gi3(self.enc3(self.pool(e2)),     F.avg_pool2d(phases[2], 4))

        d = self.dec1(torch.cat([self.up(e3), e2], dim=1))
        d = self.dec2(torch.cat([self.up(d),  e1], dim=1))
        return self.head(d)

model = MonoUNet()
n = sum(p.numel() for p in model.parameters())
print(f"参数量: {n:,}")   # ~60K，视 base 和 scales 而定
x = torch.randn(1, 1, 256, 256)
print(model(x).shape)     # torch.Size([1, 1, 256, 256])
```

### 实现中的坑

**坑1：相位特征必须在原始输入上计算，而非中间特征图**

```python
# 错误：在下采样后的特征上算 Riesz，H/W < 16 时 FFT 精度极差
e2_feats = monogenic(self.enc2_feat)   # ❌

# 正确：MonogenicBlock 只吃原始输入 x
phases = self.mono(x)                  # ✓ 始终全分辨率
```

**坑2：带通滤波器初始化要避免恒等映射**

```python
# 如果 bp(x) ≈ x，则 band = x - bp(x) ≈ 0，梯度消失
# 用小随机初始化 + 偏置确保初始响应非零
nn.init.normal_(bp_layer.weight, std=0.01)
```

**坑3：atan2 在 x=0 处的梯度**

```python
# torch.atan2(y, x) 在 x=0 时梯度为 NaN
# 修复：对 x 加 epsilon
orient = torch.atan2(R2, R1 + 1e-8)  # ✓
```

---

## 实验：论文说的 vs 现实

| 设备类型 | Dice (%) | MASD (mm) |
|---------|---------|-----------|
| 推车式 | 94.82 | 0.133 |
| 便携式 | 93.47 | 0.198 |
| 手持式 | 92.62 | 0.254 |

与轻量基线对比（参数量匹配时）：

| 模型 | 参数量 | FLOPs | 手持 Dice |
|-----|-------|-------|---------|
| MobileNetV2-UNet | ~2.1M | ~280M | ~87% |
| EfficientUNet-B0 | ~4.7M | ~390M | ~88% |
| **MonoUNet** | **~60K** | **~14M** | **92.6%** |

跨设备精度衰减（推车→手持）：MonoUNet 降 **2.2%**，无相位模块的等规模基线降 **5.8%**。

**可复现性说明**：官方代码已开源，但多站点数据集未公开，无法完全复现论文数字。从 CAMUS 等公开超声数据集做迁移实验可验证方法有效性，但绝对指标会有出入。

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要部署到手持/嵌入式设备 | 服务端推理，算力充足 |
| 跨设备、跨中心泛化是核心需求 | 只在单一设备类型上部署 |
| 超声图像分割（相位特征与模态匹配） | CT/MRI（不受增益变化影响，相位优势减弱） |
| 模型文件需要 < 1MB | 需要极高精度，可接受大模型 |
| 标注数据有限，需要强先验 | 大规模数据充足，端到端学习足够 |

---

## 我的观点

### 这个框架比结果更值得关注

论文最有价值的贡献不是 Dice 数字，而是**演示了"将领域物理先验以可微方式嵌入网络"的具体路径**。

相同的框架可以迁移到：

- **OCT 视网膜分割**：光学相干断层扫描同样有强烈的相干噪声，局部相位对斑点噪声天然鲁棒
- **弹性超声的应变估计**：相位在位移编码中有直接物理意义
- 任何**图像风格变化大但几何结构稳定**的跨域分割问题

### 两个没回答的问题

**1. 可学习带通滤波器学到了什么？** 论文没有可视化分析——学到的核是否仍然保持 Log-Gabor 的频率响应特性？还是退化成了普通卷积？这直接影响我们对"相位先验是否真的起作用"的判断。

**2. 重度骨关节炎患者的软骨极薄（< 1mm），恰好是临床最需要精确测量的人群。** 论文报告的是平均指标，没有按软骨厚度分层分析。这是方法进入临床前最需要补上的实验。

官方代码：[https://github.com/alvinkimbowa/monounet](https://github.com/alvinkimbowa/monounet)