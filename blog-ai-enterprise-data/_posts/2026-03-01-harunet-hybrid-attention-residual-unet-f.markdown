---
layout: post-wide
title: 'HARU-Net：混合注意力 + 残差 U-Net，如何在 CBCT 降噪中"保住"齿根边界？'
date: 2026-03-01 06:50:53 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.22544v1
generated_by: Claude Code CLI
---

## 一句话总结

HARU-Net 把 CBAM 风格的混合注意力（通道+空间）嵌入残差 U-Net，解决了传统降噪方法"去噪越狠，边缘越糊"的内在矛盾，专为 CBCT 牙科影像设计。

---

## 为什么这篇论文重要？

### 问题的根源：CBCT 的噪声不是普通噪声

锥束 CT（Cone-Beam CT）的噪声来源是**光子计数统计**——泊松分布，不是高斯分布。低剂量 CBCT 为了减少辐射，光子数更少，噪声更重。

更麻烦的是，CBCT 还有结构性伪影：
- **光束硬化（Beam Hardening）**：金属修复体旁边的条纹，看起来像噪声，但不是
- **环形伪影（Ring Artifacts）**：探测器不均匀性导致的同心圆纹路

这意味着：**任何"均匀平滑"的降噪策略都是错的。**

### 现有方法的痛点

| 方法类别 | 代表 | 问题 |
|---------|------|------|
| 滤波器法 | BM3D、NLM | 对 CBCT 非高斯噪声假设错误 |
| 标准 U-Net | DnCNN | 去噪即平滑，损伤高频边缘 |
| 自注意力 Transformer | Restormer | 医学图像数据少，容易过拟合 |

### HARU-Net 的核心洞见

> 用注意力机制教网络区分"这里是噪声"和"这里是边缘"——两者在频率域上重叠，但在语义上完全不同。

---

## 核心方法解析

### 直觉：为什么"混合注意力"能保边缘？

想象你在 Photoshop 里降噪，你会：
1. 先找到"这张图哪些区域是平坦的纹理（可以大力平滑）"
2. 再找到"哪些是牙根/骨骼边界（不能碰）"

HARU-Net 的注意力模块做的正是这件事——只不过是自动学习的：

- **通道注意力（Channel Attention）**：学习"哪些特征图编码了边缘信息"，给这些 channel 更高权重
- **空间注意力（Spatial Attention）**：学习"哪些像素位置是边缘"，在那里抑制降噪力度

两者串联（先通道后空间），就是 CBAM 结构——这里被称为"Hybrid Attention"。

### 数学公式

**通道注意力：**

$$M_c(F) = \sigma\!\left(W_1 \operatorname{ReLU}\!\left(W_0 F^{avg}_c\right) + W_1 \operatorname{ReLU}\!\left(W_0 F^{max}_c\right)\right)$$

其中 $F^{avg}_c, F^{max}_c$ 分别是全局平均池化和最大池化结果。

**空间注意力：**

$$M_s(F) = \sigma\!\left(f^{7\times7}\!\left([\operatorname{AvgPool}(F);\, \operatorname{MaxPool}(F)]\right)\right)$$

**残差降噪（Residual Learning）：**

论文采用噪声估计而非直接映射——网络学的不是干净图像，而是**噪声本身**：

$$\hat{x} = y - \mathcal{F}_\theta(y)$$

这让残差连接有了双重含义：ResNet 的梯度通路 + DnCNN 的噪声残差学习。

### 整体架构

```
输入(噪声CBCT) → [编码器] → [瓶颈] → [解码器] → 输出(干净图像)
                     ↕ skip connections（保留空间细节）
每个块内部：Conv → BN → ReLU → Conv → BN → HybridAttn → +残差
```

---

## 动手实现

### 核心模块：混合注意力（CBAM）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力：学习"哪些特征图"重要"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        avg = x.mean(dim=[2, 3])          # (B, C) 全局平均
        mx  = x.amax(dim=[2, 3])          # (B, C) 全局最大
        # 两路共享 MLP，相加后激活
        attn = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * attn.view(B, C, 1, 1)  # 广播到空间维度

class SpatialAttention(nn.Module):
    """空间注意力：学习"哪些像素位置"重要"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)    # (B,1,H,W)
        mx  = x.amax(dim=1, keepdim=True)   # (B,1,H,W)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class HybridAttention(nn.Module):
    """先通道后空间——顺序很重要"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))   # 通道 → 空间
```

### 残差注意力块

```python
class ResAttnBlock(nn.Module):
    """U-Net 每级的基础块：Conv-BN-ReLU × 2 + HybridAttn + 残差"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.attn = HybridAttention(out_ch)
        # 通道数不同时需要投影
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.attn(self.conv_path(x))
        return self.relu(out + self.proj(x))   # 残差相加
```

### HARU-Net 完整骨架

```python
class HARUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=(64, 128, 256, 512)):
        super().__init__()
        # 编码器：逐级下采样
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.encoders.append(ResAttnBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # 瓶颈
        self.bottleneck = ResAttnBlock(features[-1], features[-1] * 2)

        # 解码器：逐级上采样 + skip connection
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoders.append(ResAttnBlock(f * 2, f))  # *2 因为 concat skip
            ch = f

        self.head = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips, inp = [], x
        for enc, pool in zip(self.encoders, self.pools):
            inp = enc(inp); skips.append(inp); inp = pool(inp)

        inp = self.bottleneck(inp)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            inp = up(inp)
            # 处理奇数尺寸的边界情况
            if inp.shape != skip.shape:
                inp = F.interpolate(inp, size=skip.shape[2:])
            inp = dec(torch.cat([inp, skip], dim=1))

        # 残差输出：预测噪声，干净 = 输入 - 噪声
        return x - self.head(inp)
```

### 边缘感知损失函数

这是论文最关键的工程决策——**只用 MSE 是不够的**：

```python
class EdgePreservingLoss(nn.Module):
    def __init__(self, edge_weight=0.3):
        super().__init__()
        self.w = edge_weight
        # Sobel 算子检测边缘，固定权重不参与训练
        sobel = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]],
                               [[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32)
        self.register_buffer('sobel', sobel.unsqueeze(1))  # (2,1,3,3)

    def _edge_map(self, x):
        # x: (B,1,H,W) → 计算梯度幅值
        grad = F.conv2d(x, self.sobel, padding=1)          # (B,2,H,W)
        return grad.pow(2).sum(dim=1, keepdim=True).sqrt()  # (B,1,H,W)

    def forward(self, pred, target):
        loss_pixel = F.l1_loss(pred, target)               # L1 比 MSE 更保边缘
        loss_edge  = F.l1_loss(self._edge_map(pred),
                               self._edge_map(target))
        return loss_pixel + self.w * loss_edge
```

### 训练配置

```python
model     = HARUNet(in_ch=1, out_ch=1).cuda()
criterion = EdgePreservingLoss(edge_weight=0.3)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    for noisy, clean in dataloader:
        noisy, clean = noisy.cuda(), clean.cuda()
        pred = model(noisy)
        loss = criterion(pred, clean)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    scheduler.step()
```

---

## 实现中的坑

### 坑 1：CBCT 图像的 HU 值范围

CBCT 原始数据是 Hounsfield Unit（HU），范围 $[-1000, +3000]$。直接送入网络会导致梯度爆炸：

```python
# 错误做法
img = dicom.pixel_array.astype(np.float32)  # 原始 HU 值

# 正确做法：窗宽窗位归一化（牙科常用 W=4000, L=1000）
def normalize_cbct(img, window=4000, level=1000):
    low, high = level - window/2, level + window/2
    return np.clip((img - low) / window, 0.0, 1.0)
```

### 坑 2：注意力的 reduction ratio 在小图像上会崩

当 `in_channels=64, reduction=16` 时 `mid=4`，没问题。但如果误设 `reduction=32`，`mid=2`，MLP 表达力严重不足：

```python
# HybridAttention 构造时加保护
mid = max(in_channels // reduction, 4)   # 前面代码已经处理了这个
```

### 坑 3：跳跃连接的尺寸错位

CBCT 图像常见非 2 的幂次尺寸（如 $512 \times 492$）。下采样后上采样回来会差 1 个像素——已在 `forward` 中用 `F.interpolate` 修复，但要确认 `align_corners=False`（PyTorch 默认）。

### 坑 4：边缘损失权重不是越大越好

`edge_weight > 0.5` 时，网络倾向于"制造边缘"来降低损失，在平坦区域产生锐化伪影。建议从 `0.1` 开始网格搜索。

---

## 实验：论文说的 vs 现实

### 论文预期结果

在 CBCT 牙科数据集上，HARU-Net 应当在 PSNR/SSIM 上超过：
- BM3D：约 +2-3 dB PSNR
- 标准 U-Net：约 +1 dB PSNR
- 主观评价：齿根边界更清晰

### 复现时的现实条件

| 条件 | 影响 |
|------|------|
| 训练数据 < 200 对 | 注意力模块过拟合风险高，建议关掉 SpatialAttention 或加 Dropout |
| 金属植入物区域 | 模型往往在此失效，边缘损失反而学到了金属伪影轮廓 |
| 不同剂量设备间迁移 | 泛化性差，需要 domain adaptation 或 fine-tune |
| 3D CBCT vs 2D 切片 | 论文大概率用 2D 切片训练，3D 推理需要逐层或改 3D 卷积 |

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 低剂量 CBCT 降噪（泊松噪声为主） | 金属伪影去除（需要专门的 MAR 方法） |
| 数据量 > 500 对，且有配对干净/噪声图 | 无配对数据（需换用自监督方法如 Noise2Void） |
| 边缘精度要求高（齿根、骨骼轮廓测量） | 实时推理（注意力计算有额外开销） |
| 单中心固定设备 | 多中心多设备部署（域偏移问题显著） |

---

## 我的观点

**值得肯定的部分**：CBAM 加入 U-Net 用于医学图像降噪，技术路线是合理的。边缘感知损失是必须的设计，不是可选项——任何面向诊断的降噪系统都应该包含某种边缘保护机制。

**我的疑虑**：

1. **"Hybrid"的命名有点虚**。CBAM 是 2018 年的工作，在这里被重新包装为"Hybrid Attention"。真正的创新点需要看论文是否在注意力结构上有实质改动（比如专门设计的边缘引导注意力），还是直接套用 CBAM。

2. **配对数据的获取是真正的瓶颈**。医院能提供的配对数据（同一患者低剂量+高剂量扫描）极为稀少。论文可能使用合成噪声（往干净图上加泊松噪声），但真实低剂量噪声的分布远比合成复杂。

3. **与 Diffusion-based 方法的比较缺失**。2024-2025 年，扩散模型在医学图像重建上的表现已经相当出色（如 DiffuseIR、score-based CT denoising）。HARU-Net 这类判别式方法的推理速度有优势，但效果上可能已被超越。

**结论**：如果你在做 CBCT/CT 降噪的工程落地，HARU-Net 的架构值得参考，特别是残差学习 + 混合注意力 + 边缘损失这个组合。但如果追求 SOTA，建议同时评估基于 score matching 的生成式方法。

---

*注：本文基于论文摘要进行架构推断，完整实验细节（超参数、数据集划分、具体注意力变体）请参考原论文 [arxiv: 2602.22544](https://arxiv.org/abs/2602.22544v1)。*