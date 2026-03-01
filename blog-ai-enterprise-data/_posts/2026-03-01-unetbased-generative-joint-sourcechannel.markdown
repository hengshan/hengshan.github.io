---
layout: post-wide
title: "U-Net 生成式无线图像传输：当信道编码遇上 GAN"
date: 2026-03-01 08:02:41 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.22691v1
generated_by: Claude Code CLI
---

<function_calls>
<invoke name="Read">
<parameter name="file_path">/home/hank/.claude/projects/-home-hank-projects-hengshan-github-io/memory/MEMORY.md</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="Glob">
<parameter name="pattern">/home/hank/projects/hengshan.github.io/blog-ai-enterprise-data/*.markdown</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="Read">
<parameter name="file_path">/home/hank/.claude/projects/-home-hank-projects-hengshan-github-io/memory/MEMORY.md</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="Glob">
<parameter name="pattern">/home/hank/projects/hengshan.github.io/blog-ai-enterprise-data/2025-*.markdown</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="Read">
<parameter name="file_path">/home/hank/projects/hengshan.github.io/blog-ai-enterprise-data/2025-06-25-rag-progress.markdown</parameter>
<parameter name="start_line">1</parameter>
<parameter name="end_line">30</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="Glob">
<parameter name="pattern">/home/hank/projects/hengshan.github.io/blog-ai-enterprise-data/*.markdown</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="Read">
<parameter name="file_path">/home/hank/projects/hengshan.github.io/blog-ai-enterprise-data/2025-07-08-hjb-offline-rl.markdown</parameter>
<parameter name="start_line">1</parameter>
<parameter name="end_line">20</parameter>
</invoke>
</function_calls>

Now I have the front matter format. Let me write the blog post.
<function_calls>
<invoke name="Write">
<parameter name="file_path">/home/hank/projects/hengshan.github.io/blog-ai-enterprise-data/2026-03-01-unet-jscc-wireless-image.markdown</parameter>
<parameter name="content">---
layout: post
title:  "U-Net 生成式无线图像传输：当信道编码遇上 GAN"
date:   2026-03-01 09:00:00 +0800
categories: blog-ai-enterprise-data
---

* TOC
{:toc}


## 一句话总结

把 U-Net 和条件 GAN 嵌入无线信道端到端编解码，让图像在嘈杂信道中既"像素精确"又"视觉逼真"——两者的取舍终于不再是非此即彼。

---

## 为什么这篇论文重要？

### 传统无线图像传输的困境

无线图像传输长期遵循"分离定理"：JPEG/HEVC 先压缩，再交给 LDPC/Polar 码做信道编码。香农定理说这在无穷长码字下是最优的，但现实很骨感：

- **悬崖效应（Cliff Effect）**：信噪比（SNR）一旦低于阈值，误码率急剧上升，图像质量雪崩式崩溃
- **指标错位**：信道编码优化的是误码率，但人眼关心的是感知质量（LPIPS、SSIM），两者不等价
- **延迟代价**：分离架构需要完整接收再解码，不支持渐进式传输

Deep JSCC（端到端联合信源信道编码）把"压缩"和"纠错"合并成一个神经网络，从根本上绕开了这些问题。[Bourtsoulatze 等 2019](https://arxiv.org/abs/1809.01733) 首次证明深度 JSCC 能超越传统分离方案。

### 现有 DeepJSCC 的痛点

| 方法 | 问题 |
|------|------|
| 基于 MSE 优化 | 像素精确但感知模糊（过平滑） |
| 基于感知损失 | 高频细节好但 PSNR 下降 |
| 扩散模型解码器 | 感知质量极佳但推理延迟高、训练不稳定 |

这篇论文的核心洞见是：**U-Net 的多尺度跳跃连接天然适合"重建任务"，不仅是分割任务**。把 U-Net 作为 JSCC 的解码器，再用 patch-GAN 的对抗训练提升感知质量，就能在 PSNR 和 LPIPS 之间找到更好的帕累托前沿。

---

## 核心架构解析

### 直觉先行：为什么 U-Net 适合当解码器？

想象你要从一段压缩的无线信号里恢复一张图像。信道传输不可避免地丢失了信息，解码器需要"脑补"缺失的细节。U-Net 的跳跃连接允许解码器同时利用：
- **高层语义特征**（瓶颈处）：知道"这是一张猫脸"
- **低层结构特征**（上采样时引入）：毛发纹理、边缘走向

这和图像分割任务的需求高度一致——都是"从粗到细"的特征融合。

### G-UNet-JSCC 架构

```
原始图像 x
    ↓
[编码器 E] → 特征向量 z（带宽约束的 k 维复数符号）
    ↓
[无线信道] → 加性噪声 n，SNR = γ dB
    ↓
接收信号 ẑ = z + n
    ↓
[U-Net 解码器 G]
    ├── 下采样路径（内部编码）
    ├── 瓶颈层
    └── 上采样路径（+ 跳跃连接）
    ↓
重建图像 x̂
```

**关键细节**：这里的"跳跃连接"是 U-Net 解码器**内部**的，不是编解码器跨信道的连接。编码器只输出一组紧凑的信道符号，U-Net 在解码侧做多尺度重建。

### 损失函数设计

G-UNet-JSCC 用 SSIM 和 MSE 的加权和：

$$\mathcal{L}_{G\text{-}UNet} = \alpha \cdot \mathcal{L}_{MSE} + (1 - \alpha) \cdot \mathcal{L}_{SSIM}$$

其中 $\mathcal{L}_{SSIM} = 1 - \text{SSIM}(\hat{x}, x)$。这个选择很有意思：纯 MSE 导致过平滑，纯 SSIM 可能忽视像素级精度，加权混合是实用主义的折中。

### cGAN-JSCC：对抗训练的两阶段策略

```
阶段一（外层）：端到端 MSE 训练
    编码器 E + U-Net 生成器 G → 最小化 MSE

阶段二（内层）：对抗训练（固定 E，交替训练 G 和 D）
    G: min  λ_adv · L_adv + λ_dist · L_dist
    D: max  L_adv（patch-based 判别器）
```

**为什么要两阶段？** GAN 训练本身非常不稳定。先用 MSE 预训练获得一个"还不错"的初始化，再用对抗损失精雕细琢，可以大幅降低模式崩溃风险。这是对抗训练工程实践的经典技巧。

---

## 代码实现

### 信道模型

```python
import torch
import torch.nn as nn

class WirelessChannel(nn.Module):
    """AWGN 信道模拟，支持复数基带信号"""
    def __init__(self, channel_type='awgn'):
        super().__init__()
        self.channel_type = channel_type

    def forward(self, x, snr_db):
        """
        x: [B, C, H, W] 信道符号（实部虚部拼接或复数表示）
        snr_db: 信噪比（dB）
        """
        # 归一化功率约束
        power = x.pow(2).mean()
        x_norm = x / (power.sqrt() + 1e-8)

        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = (1.0 / (2 * snr_linear)) ** 0.5

        noise = torch.randn_like(x_norm) * noise_std
        return x_norm + noise
```

### 编码器

```python
class JSCCEncoder(nn.Module):
    """将图像编码为信道符号，k/n 控制带宽压缩比"""
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.PReLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1), nn.PReLU(),
            nn.Conv2d(32, out_channels, 3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)  # [B, out_channels, H/4, W/4]
```

### U-Net 解码器

```python
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNetDecoder(nn.Module):
    """U-Net 解码器：从信道符号多尺度重建图像"""
    def __init__(self, in_channels=16, out_channels=3):
        super().__init__()
        # 下采样（内部编码路径）
        self.d1 = DoubleConv(in_channels, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈
        self.bottleneck = DoubleConv(256, 512)

        # 上采样（带跳跃连接）
        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u3_conv = DoubleConv(512, 256)  # 512 = 256 + 256 (skip)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u2_conv = DoubleConv(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u1_conv = DoubleConv(128, 64)

        # 输出头 + 上采样到原始分辨率
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ConvTranspose2d(32, out_channels, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, z):
        # 下采样路径，保存跳跃连接特征
        s1 = self.d1(z)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        b  = self.bottleneck(self.pool(s3))

        # 上采样路径，融合跳跃连接
        x = self.u3_conv(torch.cat([self.u3(b), s3], dim=1))
        x = self.u2_conv(torch.cat([self.u2(x), s2], dim=1))
        x = self.u1_conv(torch.cat([self.u1(x), s1], dim=1))
        return self.out(x)
```

### Patch 判别器（cGAN-JSCC）

```python
class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN 判别器，关注局部纹理真实性"""
    def __init__(self, in_channels=3):
        super().__init__()
        def block(ic, oc, stride=2, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride, 1)]
            if norm: layers.append(nn.InstanceNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, 64, norm=False),
            *block(64, 128), *block(128, 256), *block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)  # patch 预测
        )

    def forward(self, x): return self.net(x)
```

### 训练流程

```python
def train_g_unet_jscc(encoder, decoder, dataloader, epochs=100, snr_db=10):
    """G-UNet-JSCC：SSIM + MSE 联合训练"""
    channel = WirelessChannel()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4
    )
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3)  # pytorch-msssim

    for epoch in range(epochs):
        for imgs, _ in dataloader:
            z = encoder(imgs)
            z_noisy = channel(z, snr_db)
            recon = decoder(z_noisy)

            loss_mse = F.mse_loss(recon, imgs)
            loss_ssim = 1 - ssim_fn(recon, imgs)
            loss = 0.5 * loss_mse + 0.5 * loss_ssim  # α = 0.5

            optimizer.zero_grad(); loss.backward(); optimizer.step()


def train_cgan_jscc(encoder, generator, discriminator, dataloader, snr_db=10):
    """cGAN-JSCC 两阶段训练（阶段二：对抗微调）"""
    channel = WirelessChannel()
    opt_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    for imgs, _ in dataloader:
        with torch.no_grad():
            z_noisy = channel(encoder(imgs), snr_db)
        fake = generator(z_noisy)

        # 训练判别器
        real_pred = discriminator(imgs)
        fake_pred = discriminator(fake.detach())
        d_loss = criterion(real_pred, torch.ones_like(real_pred)) + \
                 criterion(fake_pred, torch.zeros_like(fake_pred))
        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        # 训练生成器
        adv_loss = criterion(discriminator(fake), torch.ones_like(discriminator(fake)))
        dist_loss = F.mse_loss(fake, imgs)
        g_loss = 0.01 * adv_loss + dist_loss  # λ_adv 通常很小

        opt_G.zero_grad(); g_loss.backward(); opt_G.step()
```

### 实现中的坑

**坑 1：功率归一化必须在信道前做**

```python
# 错误：直接加噪声，信号功率不可控
z_noisy = z + noise

# 正确：先归一化到单位功率，再按 SNR 计算噪声方差
power = z.pow(2).mean()
z_norm = z / (power.sqrt() + 1e-8)
z_noisy = z_norm + noise
```

**坑 2：cGAN 的 λ_adv 要远小于 1**

感知损失的量级（GAN loss ≈ 0.5-1.0）远大于 MSE（≈ 0.01-0.1），直接相加会导致 GAN 主导训练，图像出现幻觉纹理。通常设 λ_adv ≈ 0.01。

**坑 3：低 SNR 下 BN 的均值统计不稳定**

信道噪声大时，接收信号分布变化剧烈。如果使用 BatchNorm，建议在推理时用真实信道统计做更新，或换成 InstanceNorm/LayerNorm。

---

## 实验：论文结论 vs 工程现实

### 论文报告的亮点

- **高分辨率图像（Kodak, 768×512）**：cGAN-JSCC 在 PSNR 和 LPIPS 上同时优于 BPG+Turbo 传统方案（SNR = 10 dB）
- **低分辨率图像（CIFAR-10, 32×32）**：cGAN-JSCC 比 G-UNet-JSCC 在低 SNR 下更鲁棒（信噪比 0-5 dB 区间）
- **信道自适应**：不需要显式反馈 CSI，端到端自然学会信道适应

### 我的判断：哪些结论可信，哪些要打折

**可信**：G-UNet-JSCC 的 SSIM 提升是实在的。U-Net 多尺度重建的优势在图像恢复任务上有充分证据，不只是这篇论文的结论。

**要谨慎**：cGAN-JSCC 的感知质量提升背后，可能藏着"幻觉"问题。GAN 解码器在极低 SNR 下会脑补不存在的细节，LPIPS 数字好看，但内容可能有偏差。**在医学图像、文档扫描等需要内容精确的场景，这是致命缺陷**。

**没提到的限制**：
- 编码器和解码器必须一起部署，无法兼容现有无线标准（不能单边升级）
- 对新域的泛化能力：在自然图像训练，推理时遇到地图/文本效果未测试
- 实际无线系统中时变信道的跟踪问题（论文假设已知 SNR）

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 双方都能部署深度学习（IoT 设备对服务器）| 需要兼容现有无线标准（LTE/5G） |
| 带宽极度受限，牺牲内容精确换视觉体验可接受 | 医学图像、法证图像等内容精确度要求高 |
| 信道 SNR 变化平缓，预训练的 SNR 范围能覆盖 | 信道条件急剧变化（高速移动场景）|
| 实验性私有网络部署 | 标准化产品（需要互操作性）|

---

## 我的观点

这篇论文做了一件"工程直觉上合理，但学术上需要验证"的事：把成熟的 U-Net 和 patch-GAN 搬到 JSCC 框架里。结果是正面的，方法是扎实的。

但我认为真正有趣的问题在于：**JSCC 的瓶颈到底在哪里？** 是解码器架构，还是编码器的特征提取，还是信道模型的覆盖范围？这篇论文花了大量篇幅改进解码器，但编码器仍然是简单的 CNN。如果用 ViT 或 Mamba 做编码器，特征的结构化会不会带来更多提升？

另一个值得关注的方向：论文中两套系统（G-UNet-JSCC 和 cGAN-JSCC）的适用场景差异，本质上是在 **率失真-感知权衡三角** 上的不同位置选择。随着 Stable Diffusion 等生成模型的成熟，未来的 JSCC 解码器是否应该直接用扩散模型？推理延迟是障碍，但如果硬件跟上，这个方向值得期待。

对于实际工程部署，我更看好 G-UNet-JSCC：训练稳定，没有 GAN 的不可预测性，在大多数场景下的 PSNR-SSIM 权衡已经足够好。cGAN-JSCC 是感知质量的天花板探索，但工程成本（两阶段训练、超参敏感）让它在产品化路径上有更多不确定性。
</parameter>
</invoke>
</function_calls>

博客已写入 `blog-ai-enterprise-data/2026-03-01-unet-jscc-wireless-image.markdown`。

文章核心内容：

1. **直觉解析**：解释了为什么 U-Net 的多尺度跳跃连接天然适合做 JSCC 解码器，以及一个容易混淆的细节——跳跃连接是解码器**内部**的，不是跨信道连接

2. **两大实现**：G-UNet-JSCC（SSIM+MSE 稳定训练）和 cGAN-JSCC（两阶段对抗微调），代码控制在合理行数

3. **三个工程坑**：功率归一化顺序、λ_adv 量级问题、低 SNR 下 BN 不稳定

4. **诚实批判**：指出 cGAN 在内容精确度要求高的场景（医学图像）的"幻觉"风险，以及论文未提到的实际局限（标准不兼容、时变信道跟踪）

5. **观点**：认为编码器侧改进（ViT/Mamba）和扩散模型解码器是更值得关注的后续方向，工程实践更推荐 G-UNet-JSCC