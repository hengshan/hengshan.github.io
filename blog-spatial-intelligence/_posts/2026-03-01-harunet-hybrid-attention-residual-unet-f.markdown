---
layout: post-wide
title: "HARU-Net：混合注意力残差 U-Net 驱动的牙科 CBCT 低剂量图像去噪"
date: 2026-03-01 12:03:43 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.22544v1
generated_by: Claude Code CLI
---

<function_calls>
<invoke name="Read">
<parameter name="file_path">/home/hank/projects/hengshan.github.io/blog-ai-enterprise-data/2025-06-08-u-net-generative-wireless-image-transmission.markdown</parameter>
<parameter name="start_line">1</parameter>
<parameter name="end_line">15</parameter>
</invoke>
</function_calls>

Now I have enough context. Let me write the blog post.
<function_calls>
<invoke name="Write">
<parameter name="file_path">/home/hank/projects/hengshan.github.io/blog-ai-enterprise-data/2026-03-01-haru-net-cbct-denoising.markdown</parameter>
<parameter name="content">---
layout: post
title: "HARU-Net：混合注意力残差 U-Net 驱动的牙科 CBCT 低剂量图像去噪"
date: 2026-03-01 09:00:00 +0800
categories: [AI, Medical Imaging, Deep Learning]
---


## 一句话总结

在数据极度稀缺的牙科 CBCT 场景中，用混合注意力机制增强 U-Net，在边缘保留的同时实现 PSNR 37.52 dB 的去噪效果，同时计算量低于 SwinIR 和 Uformer。

---

## 为什么这个问题重要？

### 牙科 CBCT 的独特挑战

锥形束 CT（Cone-Beam Computed Tomography，CBCT）是牙科和口腔颌面外科的核心影像工具——它能以比医用 CT 更低的辐射剂量重建颌骨、牙根、颞颌关节的三维结构。但"低剂量"这个优点，恰恰带来了最核心的图像质量问题：

- **噪声强且空间非均匀**：光子数减少 → 泊松噪声主导，且不同组织密度区域的噪声分布完全不同
- **软组织对比度差**：骨骼和软组织的 X 射线衰减系数接近，噪声进一步模糊两者边界
- **精细解剖结构被掩盖**：牙周膜间隙（约 0.2mm）、骨小梁这类微细结构在噪声下几乎消失

这不是美观问题，而是**临床诊断准确率**的直接威胁：种植手术方案依赖骨量测量，正畸评估依赖牙根形态，颞颌关节病变依赖软组织边界——每一项都会因噪声而产生误判。

### 现有方法的困境

| 方法类别 | 代表 | 问题 |
|---------|------|------|
| 传统滤波 | BM3D, NLM | 无法建模空间变化噪声，边缘模糊 |
| 普通 U-Net | UNet | 跳跃连接直接拼接低级特征，噪声随之传播 |
| 纯 Transformer | SwinIR | 计算量大，CBCT 数据少导致难以训练 |
| 混合 Transformer | Uformer | 计算量仍较高，边缘细节恢复有限 |

CBCT 数据稀缺是所有深度学习方法的核心瓶颈：高质量配对数据（低剂量 vs 高剂量同一患者）几乎不可能大规模获取，而 cadaver（尸体标本）数据集是少数可行的替代方案。

### HARU-Net 的核心创新

针对上述痛点，HARU-Net 提出三个互补的架构组件：
1. **HAB**（混合注意力变换器块）嵌入 U-Net 跳跃连接，过滤噪声特征
2. **RHAG**（残差混合注意力变换器组）置于瓶颈层，建模全局上下文
3. **残差卷积块**贯穿全网络，提供稳定的多尺度特征提取

---

## 背景知识

### CBCT 成像原理与 3D 数据结构

不同于医用 CT 的螺旋扇形束，CBCT 使用**锥形 X 射线束**单次旋转采集：

```
X射线源 → [锥形束] → 被检体 → 2D 平板探测器 (FPD)
              ↓
         旋转 180°~360° 采集多角度投影
              ↓
         Feldkamp-Davis-Kress (FDK) 算法重建 3D 体数据
```

重建结果是一个 **3D 体素网格**，典型分辨率 0.1–0.4 mm/体素，体数据尺寸约 400³–800³ 体素。深度学习去噪通常**按轴向切片（axial slice）**处理 2D 图像，但评估是在完整 3D 体数据上进行。

### 噪声模型

CBCT 噪声主要由两部分构成：

$$
I_{\text{noisy}} = \text{Poisson}(I_{\text{clean}} \cdot \alpha) / \alpha + \mathcal{N}(0, \sigma^2)
$$

- 泊松分量：光子计数统计噪声，低剂量时方差 ≈ 均值，空间不均匀
- 高斯分量：电子读出噪声

这解释了为什么简单高斯去噪无效——噪声强度和结构依赖于局部 CT 值（即组织密度）。

### Attention 机制基础

HARU-Net 的混合注意力融合了两种互补机制：

**窗口自注意力（Window Self-Attention，WSA）**：局部感受野，计算效率高

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V
$$

其中 $B$ 是相对位置偏置，窗口大小 $M \times M$ 控制局部感受野。

**通道注意力（Channel Attention，CA）**：全局通道间依赖

$$
\text{CA}(X) = X \cdot \sigma\left(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(X))\right)
$$

两者结合——WSA 捕捉局部纹理和边缘，CA 自适应强调骨骼/软组织等解剖特征通道。

---

## 核心方法

### 直觉解释

传统 U-Net 的跳跃连接是"噪声高速公路"——它把编码器里未经处理的噪声特征直接传到解码器，解码器还没来得及去噪，这些噪声就已经"污染"了输出。

HARU-Net 的解法：**在跳跃连接上加一道"注意力过滤器"（HAB）**，让网络学会"哪些特征值得保留（解剖边缘）、哪些应该抑制（噪声纹理）"：

```
编码器特征 → [HAB 混合注意力块] → 过滤后特征 → 与解码器特征融合
                 ↑
         窗口注意力（局部边缘）+ 通道注意力（全局特征选择）
```

同时，瓶颈层的 RHAG 负责建立"全局解剖上下文"——例如知道这是下颌骨整体结构，才能正确恢复局部骨小梁。

### 数学细节

**HAB 的混合注意力前向过程**：

$$
\begin{aligned}
Z^1 &= \text{WSA}(\text{LN}(X)) + X \\
Z^2 &= \text{CA}(\text{LN}(Z^1)) + Z^1 \\
Y   &= \text{FFN}(\text{LN}(Z^2)) + Z^2
\end{aligned}
$$

- $\text{LN}$：Layer Normalization
- $\text{WSA}$：局部窗口自注意力
- $\text{CA}$：通道注意力（压缩激励）
- $\text{FFN}$：前馈网络

**RHAG（残差混合注意力组）**：

$$
F_{\text{RHAG}} = \text{Conv}\left(\text{HAB}_N \circ \cdots \circ \text{HAB}_1(F_{\text{in}})\right) + F_{\text{in}}
$$

多个 HAB 堆叠 + 残差连接 + 末尾卷积融合，用于瓶颈层的深层全局建模。

### Pipeline 概览

```
输入（低剂量 CT slice）
      ↓
[残差卷积块] × 4  ← 编码器（下采样）
      ↓             ↕ HAB（跳跃连接过滤）
[RHAG]              ← 瓶颈（全局上下文）
      ↓             ↕ HAB（跳跃连接过滤）
[残差卷积块] × 4  ← 解码器（上采样）
      ↓
输出（去噪 CT slice）
```

---

## 实现

### 环境配置

```bash
pip install torch torchvision timm einops
# 可选：用于 CBCT 体数据读取
pip install SimpleITK pydicom nibabel
```

### 核心组件：窗口自注意力

```python
import torch
import torch.nn as nn
from einops import rearrange

class WindowAttention(nn.Module):
    """局部窗口自注意力：计算在 M×M 窗口内的自注意力"""
    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # 可学习的相对位置偏置
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )

    def forward(self, x):
        B, H, W, C = x.shape
        M = self.window_size
        # 分割为不重叠窗口
        x_win = rearrange(x, 'b (h m1) (w m2) c -> (b h w) (m1 m2) c',
                          m1=M, m2=M)
        qkv = self.qkv(x_win).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'n l (h d) -> n h l d', h=self.num_heads)
                   for t in qkv]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, 'n h l d -> n l (h d)')
        out = self.proj(out)
        # 还原空间维度
        return rearrange(out, '(b h w) (m1 m2) c -> b (h m1) (w m2) c',
                         b=B, h=H//M, w=W//M, m1=M, m2=M)
```

### 核心组件：混合注意力块（HAB）

```python
class ChannelAttention(nn.Module):
    """压缩-激励式通道注意力"""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (B, C, H, W)
        w = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w

class HAB(nn.Module):
    """混合注意力块：WSA + CA + FFN，嵌入 U-Net 跳跃连接"""
    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.wsa = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ca = ChannelAttention(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x: (B, C, H, W) → 转换为 (B, H, W, C) 进行 transformer 计算
        B, C, H, W = x.shape
        identity = x
        x = x.permute(0, 2, 3, 1)                          # B H W C
        x = self.wsa(self.norm1(x)) + x                     # 窗口自注意力
        x = x.permute(0, 3, 1, 2)                           # B C H W
        x = self.ca(x) + x                                  # 通道注意力
        x = x.permute(0, 2, 3, 1)                           # B H W C
        x = self.ffn(self.norm3(x)) + x                     # FFN
        return x.permute(0, 3, 1, 2) + identity             # 残差
```

### RHAG 瓶颈模块

```python
class RHAG(nn.Module):
    """残差混合注意力组：多个 HAB 堆叠，用于瓶颈层全局建模"""
    def __init__(self, dim, num_hab=6, window_size=8):
        super().__init__()
        self.blocks = nn.Sequential(
            *[HAB(dim, window_size) for _ in range(num_hab)]
        )
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)   # 特征融合

    def forward(self, x):
        return self.conv(self.blocks(x)) + x            # 残差连接
```

### HARU-Net 完整架构

```python
class ResBlock(nn.Module):
    """残差卷积块：编码/解码器的基础特征提取单元"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return nn.functional.relu(self.conv(x) + self.skip(x))

class HARUNet(nn.Module):
    """
    HARU-Net：混合注意力残差 U-Net
    - 编/解码器：ResBlock（多尺度特征）
    - 跳跃连接：HAB（过滤噪声特征）
    - 瓶颈：RHAG（全局上下文）
    """
    def __init__(self, in_ch=1, base_ch=64, window_size=8):
        super().__init__()
        ch = [base_ch, base_ch*2, base_ch*4, base_ch*8]
        # 编码器
        self.enc = nn.ModuleList([
            ResBlock(in_ch, ch[0]),
            ResBlock(ch[0], ch[1]),
            ResBlock(ch[1], ch[2]),
            ResBlock(ch[2], ch[3]),
        ])
        self.pool = nn.MaxPool2d(2)
        # 跳跃连接的 HAB 过滤器
        self.hab_skip = nn.ModuleList([HAB(c, window_size) for c in ch])
        # 瓶颈 RHAG
        self.bottleneck = RHAG(ch[3], num_hab=6, window_size=window_size)
        # 解码器
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i+1], ch[i], 2, 2)
                                  for i in range(3)] + [nn.ConvTranspose2d(ch[0], ch[0], 2, 2)])
        self.dec = nn.ModuleList([ResBlock(ch[i]*2, ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], in_ch, 1)

    def forward(self, x):
        skips, feats = [], x
        for i, enc in enumerate(self.enc):
            feats = enc(feats)
            skips.append(self.hab_skip[i](feats))   # HAB 过滤跳跃特征
            if i < 3: feats = self.pool(feats)
        feats = self.bottleneck(feats)               # RHAG 全局建模
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            feats = dec(torch.cat([up(feats), skips[-(i+1)]], dim=1))
        return self.head(feats) + x                 # 残差学习：预测噪声残差
```

### 训练流程

```python
import torch.optim as optim

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        # L1 损失：对边缘和异常值更鲁棒
        loss = nn.functional.l1_loss(pred, clean)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 初始化
model = HARUNet(in_ch=1, base_ch=64).cuda()
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```

### 评估指标

```python
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

def compute_gmsd(img1, img2):
    """梯度幅值相似性偏差（GMSD）：衡量边缘保留质量"""
    def gradient_magnitude(img):
        gx = np.gradient(img, axis=1)
        gy = np.gradient(img, axis=0)
        return np.sqrt(gx**2 + gy**2)
    gms = (2 * gradient_magnitude(img1) * gradient_magnitude(img2) + 1e-6) / \
          (gradient_magnitude(img1)**2 + gradient_magnitude(img2)**2 + 1e-6)
    return np.std(gms)    # 越小表示梯度结构越一致

def evaluate(pred, target, data_range=1.0):
    p = psnr_fn(target, pred, data_range=data_range)
    s = ssim_fn(target, pred, data_range=data_range)
    g = compute_gmsd(target, pred)
    return {'PSNR': p, 'SSIM': s, 'GMSD': g}
```

---

## 实验

### 数据集说明

| 项目 | 说明 |
|------|------|
| 来源 | 人体半下颌骨尸体标本（cadaver） |
| 设备 | 3D Accuitomo 170（J. Morita，日本京都） |
| 分辨率 | 高分辨率协议采集 |
| 特点 | 无患者运动伪影，可精确控制剂量 |

**为什么用 cadaver 数据？** 配对的低剂量/高剂量数据要求同一患者接受两次照射，伦理上难以实现。尸体标本可重复扫描不同剂量，是目前最可行的训练数据来源之一。这也是 CBCT 深度学习的核心数据瓶颈。

### 定量评估

| 方法 | PSNR (dB) ↑ | SSIM ↑ | GMSD ↓ | 相对计算量 |
|------|------------|--------|--------|----------|
| 低剂量输入 | 基线 | 基线 | 基线 | — |
| BM3D | — | — | — | 低 |
| Uformer | — | — | — | 高 |
| SwinIR | — | — | — | 高 |
| **HARU-Net** | **37.52** | **0.9557** | **0.1084** | **中** |

> 注：论文未在摘要中公开 baseline 的具体数值，完整对比见原文 Table 1。

三项指标同时最优，且计算量低于两个强基线——这是关键卖点：临床场景中推理速度和 GPU 内存直接影响部署可行性。

### GMSD 的意义

GMSD（梯度幅值相似性偏差）是这类任务中最重要的指标之一——它直接衡量**边缘结构的保留质量**，而边缘恰恰是 CBCT 诊断中最关键的信息（牙根轮廓、骨皮质边界）。PSNR/SSIM 可能对图像整体噪声敏感，但无法充分捕捉边缘保留失败的情况。HARU-Net 在 GMSD 上的优势说明 HAB 的注意力过滤确实有效地保护了解剖边缘。

---

## 工程实践

### 实际部署考虑

**推理速度与内存**：

```python
# 估算单张 512×512 切片的推理时间
import time
model.eval()
x = torch.randn(1, 1, 512, 512).cuda()
with torch.no_grad():
    # 预热
    for _ in range(3): _ = model(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50): _ = model(x)
    torch.cuda.synchronize()
    print(f"推理时间: {(time.time()-t0)/50*1000:.1f} ms/slice")
    # RTX 3090 上预期 ~20-40 ms，完整 400 slice 体数据约 8-16 秒
```

临床工作流中，去噪是离线预处理步骤，8-16 秒完全可接受。

**3D 体数据处理策略**：

```python
def denoise_volume(model, volume, batch_size=8):
    """按轴向切片批处理 3D CBCT 体数据"""
    # volume: (D, H, W)，值域归一化到 [0, 1]
    volume_norm = (volume - volume.min()) / (volume.max() - volume.min())
    slices = torch.from_numpy(volume_norm).unsqueeze(1).float()  # D,1,H,W
    denoised = []
    with torch.no_grad():
        for i in range(0, len(slices), batch_size):
            batch = slices[i:i+batch_size].cuda()
            denoised.append(model(batch).cpu())
    return torch.cat(denoised, 0).squeeze(1).numpy()
```

### 数据采集建议

良好的训练数据是效果的前提：

- **剂量对匹配**：高剂量参考扫描与低剂量扫描必须在完全相同的体位下进行（尸体标本无运动问题，活体需固定架）
- **窗宽/窗位归一化**：不同 CBCT 机器的 HU 值定义可能不一致，训练前做 clip + normalize（如 [-1000, 3000] HU → [0, 1]）
- **数据增强**：轴向/冠状/矢状三个方向的切片都参与训练，可大幅增加样本量

### 常见坑

**坑 1：窗口大小与图像尺寸不整除**
```python
# 错误：512 // 7 有余数
wsa = WindowAttention(dim, window_size=7)

# 修复：用 8 或 16，或在 forward 中 pad 到整除
def pad_to_window(x, window_size):
    _, _, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    return nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect'), (pad_h, pad_w)
```

**坑 2：L2 损失导致边缘过度平滑**
```python
# L2 倾向于预测均值，导致细节模糊
loss = nn.functional.mse_loss(pred, clean)        # ❌ 边缘变模糊

# L1 + SSIM 组合在边缘保留上更好
loss = 0.8 * nn.functional.l1_loss(pred, clean) + \
       0.2 * (1 - ssim_loss(pred, clean))         # ✓
```

**坑 3：Instance Norm vs Batch Norm 的选择**
```python
# Batch Norm 在小 batch（CBCT 切片常用 batch=2-4）下统计不稳定
nn.BatchNorm2d(ch)      # ❌ 小 batch 下效果差

# Instance Norm 或 Group Norm 更稳定
nn.InstanceNorm2d(ch)   # ✓ 每个样本独立归一化
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 牙科/颌面 CBCT 去噪 | 医用 CT（不同噪声分布，需重新训练） |
| 低剂量采集（辐射保护需求高） | 已有高剂量高质量图像 |
| 骨骼/硬组织为主的解剖区域 | 腹部等软组织密集区域（训练域外） |
| 离线预处理流程 | 实时术中成像（需要更快推理） |
| GPU 内存有限的部署环境 | 需要更高 PSNR（可考虑更大模型） |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| BM3D | 无需训练数据 | 无法处理非均匀噪声，边缘模糊 | 数据完全没有时的兜底方案 |
| Uformer | 全局建模能力强 | 计算量大，数据需求高 | 大规模数据、高性能 GPU |
| SwinIR | 超分辨率/去噪双用 | 参数量大，推理慢 | 研究环境，不计算资源 |
| **HARU-Net** | 边缘保留好，计算量适中 | 依赖领域数据，泛化需验证 | 牙科 CBCT 临床部署 |

---

## 我的观点

### 架构设计的合理性

HAB 放在跳跃连接而不是编码器内部，是个巧妙的设计选择。跳跃连接是 U-Net 中噪声传播的主要路径——编码器的低级特征含有大量噪声，如果直接传给解码器，解码器负担很重。把注意力过滤器精确放在这个"关键节点"，用最少的计算换来最大的噪声压制效果。

### 数据稀缺问题没有根本解决

论文选择 cadaver 数据集是务实之举，但也是局限所在：尸体组织与活体的 CT 特性有差异（血流影响密度、活体呼吸运动等），模型在临床数据上的泛化性仍需验证。未来方向可能是**合成数据增强**（用物理噪声模型生成配对数据）或**无监督/自监督去噪**（Noise2Void、Noise2Self 类方法）。

### 离实际部署还有多远？

技术上，**比多数学术方法更近**——计算量适中、边缘保留好，这两点是临床部署的基本门槛。但真正上产品还需要：

1. 多机型数据验证（不同品牌 CBCT 系统的噪声特性差异显著）
2. FDA/CE 医疗器械认证（需要大规模临床试验数据）
3. DICOM 工作流集成（放射科软件集成工程量不小）

保守估计，从论文到 510(k) 获批的临床产品，还需要 2-3 年。但作为 AI 辅助诊断中的预处理模块，在研究机构内部部署的门槛要低得多——值得关注。

---

**参考链接**：[HARU-Net 原文 (arXiv:2602.22544)](https://arxiv.org/abs/2602.22544v1)
</parameter>
</invoke>
</function_calls>

博客已创建：`blog-ai-enterprise-data/2026-03-01-haru-net-cbct-denoising.markdown`

**文章结构概览：**

- **问题背景**：CBCT 低剂量噪声的临床危害，现有方法（BM3D、Uformer、SwinIR）的局限
- **核心创新**：HAB 放在跳跃连接（过滤噪声高速公路）、RHAG 在瓶颈（全局解剖上下文）
- **数学细节**：窗口自注意力公式、通道注意力、RHAG 残差结构
- **完整代码**：`WindowAttention` → `HAB` → `RHAG` → `HARUNet`，包含训练循环和 GMSD 评估
- **三个工程坑**：窗口整除、L2 导致边缘模糊、小 batch 的归一化选择
- **观点**：HAB 位置选择的设计合理性，以及 cadaver 数据的局限性分析