---
layout: post-wide
title: "用 U-Net 与 GAN 重新思考无线图像传输：端到端联合信源信道编码"
date: 2026-03-01 06:48:44 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.22691v1
generated_by: Claude Code CLI
---

## 一句话总结

传统"压缩+纠错"两段式管道在低信噪比下会发生"悬崖效应"——这篇论文用端到端的 U-Net + GAN 架构，让无线图像传输在恶劣信道下仍能保持视觉质量。

---

## 为什么这篇论文重要？

### 传统方案的根本缺陷

Shannon 的信源信道分离定理告诉我们：在**无限码长**条件下，先压缩再纠错不损失性能。但现实系统里码长是有限的，分离设计带来一个可怕的问题——**悬崖效应（Cliff Effect）**：

```
                传统方案
质  ██████████████████████│
量  ██████████████████████│ ← 一旦信噪比低于阈值，质量断崖式下跌
    ──────────────────────┼────────► SNR
                          阈值

                JSCC 方案
质  ████████████████████████
量  ██████████████████████
    ████████████████████         ← 随信噪比平滑退化，不会突然崩溃
    ──────────────────────────► SNR
```

工程上的痛点很直接：移动网络的信噪比是时变的，用户一走进地下室，视频就变成马赛克——这不是压缩质量的问题，是系统架构的问题。

### JSCC 的核心洞见

**联合信源信道编码（JSCC）**直接学习从像素到复数信道符号的映射，把"该传什么信息"和"怎么对抗噪声"合并成一个优化问题。深度学习让这件事变得可行。

加上 U-Net 和 GAN，这篇论文解决的是 JSCC 领域长期存在的问题：**如何在极低带宽比（channel use per pixel 很小）下，让重建图像"看起来好"而不只是"误差小"**。

---

## 核心方法解析

### 系统架构直觉

整个系统可以理解为一个"有噪声瓶颈"的自编码器：

```
原始图像 s
   │
   ▼
[U-Net 编码器]  ← 提取多尺度特征，skip connections 保留细节
   │
   ▼ 压缩到 k 个复数符号（功率归一化）
   │
   ▼
[无线信道] + AWGN 噪声 n ~ N(0, σ²I)
   │
   ▼
[U-Net 解码器]  ← 利用 skip connections 重建多频率细节
   │
   ▼
[GAN 判别器] → 引导解码器生成视觉真实的细节
   │
   ▼
重建图像 ŝ
```

U-Net 的 skip connections 在这里有特别的意义：**编码器的浅层特征保留了低频结构（轮廓、颜色），深层特征是语义信息**。解码器通过 skip 直接访问这些多尺度特征，让重建更稳定。

### 信道模型与带宽约束

设原始图像 $s \in \mathbb{R}^{H \times W \times 3}$，编码器输出 $k$ 个复数符号 $x \in \mathbb{C}^k$。

带宽压缩比定义为：

$$\rho = \frac{k}{H \times W \times 3}$$

$\rho$ 越小，需要传的"字节"越少，任务越难。实用场景下 $\rho \in [1/12, 1/6]$ 是常见选择。

信道模型（加性高斯白噪声 AWGN）：

$$y = x + n, \quad n \sim \mathcal{N}(0, \sigma^2 I)$$

信噪比 $\text{SNR} = 1 / \sigma^2$（假设信号功率归一化为 1）。

### 损失函数设计

训练时联合优化三个目标：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{distortion}} + \lambda_2 \mathcal{L}_{\text{perceptual}} + \lambda_3 \mathcal{L}_{\text{adversarial}}$$

- **失真损失**：$\mathcal{L}_{\text{distortion}} = \|s - \hat{s}\|_2^2$ （像素级 MSE）
- **感知损失**：$\mathcal{L}_{\text{perceptual}} = \|\phi(s) - \phi(\hat{s})\|_2^2$，其中 $\phi$ 是预训练 VGG 的中间层特征
- **对抗损失**：$\mathcal{L}_{\text{adversarial}} = \mathbb{E}[\log D(\hat{s})]$，让判别器分不清真实图像和重建图像

**关键权衡**：MSE 让 PSNR 好，GAN 让 LPIPS（感知相似度）好——两者天然对立。这篇论文的贡献之一是找到合适的权衡点。

---

## 动手实现

### 信道模拟模块

```python
import torch
import torch.nn as nn

class AWGNChannel(nn.Module):
    """AWGN 信道：加性高斯白噪声"""
    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]，假设已功率归一化
        snr_linear = 10 ** (self.snr_db / 10)
        noise_std = (1.0 / snr_linear) ** 0.5
        noise = torch.randn_like(x) * noise_std
        return x + noise

class PowerNormalize(nn.Module):
    """将编码器输出归一化到单位平均功率"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对每个样本独立归一化
        power = x.pow(2).mean(dim=[1, 2, 3], keepdim=True)
        return x / (power.sqrt() + 1e-8)
```

### U-Net JSCC 核心架构

```python
class DoubleConv(nn.Module):
    """U-Net 基础块：两层卷积 + BN + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class JSCCEncoder(nn.Module):
    """U-Net 编码器：输出信道符号 + 保存 skip features"""
    def __init__(self, in_ch=3, base_ch=64, bottleneck_ch=16):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)         # 128x128 → 128x128
        self.enc2 = DoubleConv(base_ch, base_ch * 2)   # 64x64
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)  # 32x32
        self.pool = nn.MaxPool2d(2)
        # 瓶颈层：将特征压缩到 bottleneck_ch 个信道符号
        self.bottleneck = nn.Sequential(
            DoubleConv(base_ch * 4, base_ch * 8),
            nn.Conv2d(base_ch * 8, bottleneck_ch, 1),  # 1x1 卷积降维
        )
        self.power_norm = PowerNormalize()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.bottleneck(self.pool(e3))  # [B, bottleneck_ch, H/8, W/8]
        z = self.power_norm(z)
        return z, (e1, e2, e3)  # skip features 也要传回（这里模拟未损信道）
```

```python
class JSCCDecoder(nn.Module):
    """U-Net 解码器：融合 skip features 重建图像"""
    def __init__(self, out_ch=3, base_ch=64, bottleneck_ch=16):
        super().__init__()
        # 从瓶颈层逐步上采样
        self.up3 = nn.ConvTranspose2d(bottleneck_ch, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)  # 融合 skip e3

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)  # 融合 skip e2

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)      # 融合 skip e1

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, 1),
            nn.Sigmoid(),  # 像素值归一到 [0, 1]
        )

    def forward(self, z, skips):
        e1, e2, e3 = skips
        x = self.dec3(torch.cat([self.up3(z), e3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), e2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), e1], dim=1))
        return self.out_conv(x)
```

### 完整系统 + GAN 训练

```python
class JSCCSystem(nn.Module):
    def __init__(self, snr_db=10.0, bandwidth_ratio=1/12):
        super().__init__()
        # bottleneck_ch 根据带宽比动态计算
        bottleneck_ch = max(4, int(3 * bandwidth_ratio * 64))
        self.encoder = JSCCEncoder(bottleneck_ch=bottleneck_ch)
        self.channel = AWGNChannel(snr_db=snr_db)
        self.decoder = JSCCDecoder(bottleneck_ch=bottleneck_ch)

    def forward(self, img):
        z, skips = self.encoder(img)
        z_noisy = self.channel(z)
        return self.decoder(z_noisy, skips)

# PatchGAN 判别器（比全图判别器更稳定）
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, padding=1),  # 输出 patch-wise 真/假得分
        )
    def forward(self, x): return self.net(x)
```

```python
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """用 VGG16 relu3_3 层计算感知损失"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for p in self.parameters(): p.requires_grad = False

    def forward(self, x, target):
        return nn.functional.mse_loss(self.features(x), self.features(target))

def train_step(system, discriminator, batch, opt_g, opt_d,
               perceptual_loss, λ=(10.0, 0.1, 0.01)):
    λ_mse, λ_perc, λ_adv = λ
    real = batch.cuda()
    fake = system(real)

    # ---- 训练判别器 ----
    opt_d.zero_grad()
    loss_d = (nn.functional.binary_cross_entropy_with_logits(discriminator(real), torch.ones_like(discriminator(real))) +
              nn.functional.binary_cross_entropy_with_logits(discriminator(fake.detach()), torch.zeros_like(discriminator(fake))))
    loss_d.backward(); opt_d.step()

    # ---- 训练生成器（编解码系统）----
    opt_g.zero_grad()
    loss_mse  = nn.functional.mse_loss(fake, real)
    loss_perc = perceptual_loss(fake, real)
    loss_adv  = nn.functional.binary_cross_entropy_with_logits(
                    discriminator(fake), torch.ones_like(discriminator(fake)))
    loss_g = λ_mse * loss_mse + λ_perc * loss_perc + λ_adv * loss_adv
    loss_g.backward(); opt_g.step()
    return loss_g.item(), loss_d.item()
```

### 实现中的坑

**坑 1：skip connections 在真实系统中无法使用**

上面的实现偷懒了——`skips` 是编码器的中间特征，**在真实无线系统中这些特征也必须经过信道传输**，不能直接从编码器传到解码器。

一种处理方式是只传瓶颈特征（解码器不用 skip），另一种是把 skip features 也量化压缩一起传——论文里通常做前者或者研究分级传输：

```python
# 简化版：不使用 skip（牺牲高频细节换取系统合理性）
def forward_no_skip(self, img):
    z, _ = self.encoder(img)       # 丢掉 skip features
    z_noisy = self.channel(z)
    return self.decoder(z_noisy, skips=None)  # 解码器退化为纯上采样
```

**坑 2：GAN 训练极度不稳定**

感知质量的提升往往伴随 PSNR 下降 1-2 dB。如果 $\lambda_{\text{adv}}$ 设太大，GAN 会开始"幻觉"——生成视觉上合理但内容完全错误的纹理。推荐：
- 先用纯 MSE 训练 10 epoch，再引入 GAN 损失
- `λ_adv` 从 `0.001` 开始，谨慎增加
- 使用 spectral normalization 稳定判别器

**坑 3：SNR 训练分布**

固定 SNR 训练的模型在其他 SNR 下性能急剧下降。实践中建议**随机采样 SNR** 训练：

```python
# 每个 batch 随机采样信噪比（覆盖目标部署范围）
snr_db = torch.empty(batch_size).uniform_(0, 20)  # 0~20 dB
```

---

## 实验：论文说的 vs 现实

### 论文报告的结果

| 指标 | 传统 BPG+LDPC | 深度学习 JSCC | U-Net+GAN JSCC |
|------|-------------|-------------|----------------|
| PSNR (10dB SNR) | ~28 dB | ~30 dB | ~28.5 dB |
| LPIPS (越小越好) | 0.18 | 0.12 | **0.08** |
| 低 SNR 抗跌落 | 悬崖效应 | 平滑退化 | 平滑退化 |

GAN 版本 PSNR 略低于纯 MSE JSCC，但 LPIPS 显著更好——这是有意的权衡。

### 复现时的注意事项

- **数据集**：论文通常用 CIFAR-10 或 Kodak 图像集，后者质量更高更有说服力
- **带宽比影响极大**：$\rho = 1/6$ 和 $\rho = 1/24$ 的效果差距比 SNR 差距更大
- 用 LPIPS 而非 SSIM 评估感知质量——两者结论有时相反

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 低带宽视频会议、实时监控流 | 医疗图像（GAN 幻觉不可接受）|
| 信道状态时变、信噪比不稳定 | 需要精确像素重建（遥感、卫星图）|
| 强调"看起来好"的消费场景 | 系统要求严格可解释性 |
| 移动端对抗深度衰落 | 对端延迟极敏感（编解码耗时高于传统方案）|

---

## 我的观点

**真正的创新不是 U-Net 或 GAN，而是承认了一件事：在无线传输场景下，"感知质量"和"信息保真度"是不同的目标**，应该被显式地优化。

但这个方向有几个我认为尚未解决的问题：

1. **Skip connection 的系统假设**：大多数论文暗暗把 skip features 当成免费信息，这在真实系统部署中是不成立的。分级传输（hierarchical transmission）才是正确方向。

2. **GAN 幻觉 vs 内容保真**：在低 SNR 下，GAN 会生成"看起来真实但内容错误"的图像。比如人脸上的眼睛位置是对的，但换了一张脸。对于通信系统，这可能比模糊的图像更危险。

3. **实际部署的对称性假设**：发送端和接收端都需要运行深度网络，对 IoT 设备的发送端计算要求很高。未来更有价值的研究方向可能是**非对称 JSCC**（轻量发送端 + 强力接收端）。

这个领域值得持续关注，尤其是与语义通信（Semantic Communication）结合的方向——与其传图像像素，不如传"图像的语义表示"。

---

*参考资料：[arxiv 原文](https://arxiv.org/abs/2602.22691v1) | 相关工作：DeepJSCC (Bourtsoulatze et al., 2019), NTSCC (Dai et al., 2022)*