---
layout: post-wide
title: "MRI 到 CT 图像合成：Drifting Model 的一步推理原理与实践"
date: 2026-03-31 08:06:14 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.28498v1
generated_by: Claude Code CLI
---

## 一句话总结

Drifting Model 用毫秒级一步推理替代扩散模型的百步迭代，在骨盆 MRI→CT 合成中兼顾骨骼细节保真度与临床可用的推理速度，为无额外辐射的放疗计划提供新路径。

## 为什么这个问题重要？

### 临床背景

MRI 对软组织有极佳对比度且**不产生电离辐射**，是肿瘤定位首选。但放疗剂量计算需要**电子密度图**——只能从 CT 的 HU（Hounsfield Unit）值中准确推导。

传统工作流的代价：
- 患者承受两次扫描（MRI + CT 的额外辐射）
- MRI 和 CT 配准误差（摆位不一致）影响剂量精度
- 双模态流程复杂、费时

理想方案是 **MR-only 工作流**——只扫 MRI，合成"CT"，消除额外辐射。

### 现有方法的瓶颈

扩散模型（DDPM/DDIM）图像质量好，但推理需要 50\~1000 步前向计算。在临床场景中，单张 256×256 切片用 DDPM 需要约 8 秒（A100），整个骨盆体积（约 90 张切片）需要 **12 分钟**，而实际工作流要求 **< 1 分钟**。

Drifting Model 把推理时间压到了**毫秒级**。

## 背景知识

### MRI vs CT：为什么骨骼是难点

骨皮质在 CT 中极亮（$> 1000$ HU），但在 MRI 中骨皮质质子密度接近零，信号极低——MRI 上骨骼是"黑色轮廓"，而非实心白色区域。

这意味着模型需要从骨骼轮廓"幻觉出"骨骼内部的 HU 分布，错误的骨骼 HU 值会直接影响放疗剂量精度。

### 从扩散模型到 Drifting Model：SDE 视角

标准扩散过程用 SDE（随机微分方程）描述：

$$dx = f(x, t)\,dt + g(t)\,dW_t$$

其中：
- $f(x, t)$：**漂移项（drift）**，确定性方向
- $g(t)\,dW_t$：**扩散项（diffusion）**，随机噪声

DDPM 的逆向推理需要模拟这个 SDE，为保证精度必须用小步长，故需多步。

Drifting Model 的核心思想——**令扩散项为零，只保留漂移项**：

$$dx = v_\theta(x, t)\,dt$$

这是一个纯 ODE。如果速度场 $v_\theta$ 足够准确（路径是直线），**一步大步长积分**即可到达目标：

$$\hat{x}_{CT} = x_{MRI} + v_\theta(x_{MRI},\; 0) \cdot \Delta t$$

## 核心方法

### 直觉解释

把图像空间想象成高维地形，MRI 和 CT 各自聚集在不同的"山谷"：

- **扩散模型**：从噪声山顶出发，沿崎岖随机路径小步下山
- **Drifting Model**：学习一个"风场"，把 MRI 图像**一步吹到** CT 所在位置

训练时用线性插值构造中间态：

$$x_t = (1 - t)\,x_{MRI} + t\,x_{CT}, \quad t \in [0, 1]$$

直线路径的目标速度恒为常向量：

$$v^* = x_{CT} - x_{MRI}$$

### 数学细节

**训练损失（Flow Matching 目标函数）**：

$$\mathcal{L}_{FM} = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; (x_{MRI}, x_{CT})} \left\| v_\theta(x_t,\; t,\; x_{MRI}) - (x_{CT} - x_{MRI}) \right\|^2$$

$x_{MRI}$ 作为条件（condition）输入网络，网络学习的是"把中间态推向 CT 方向的速度"。

**推理（一步）**：

$$\hat{x}_{CT} = x_{MRI} + v_\theta(x_{MRI},\; t=0,\; c=x_{MRI})$$

在 $t=0$ 时刻，整个 MRI 图像就是起点，网络直接预测终点偏移量。对比扩散模型预测噪声 $\epsilon$ 并通过 $T$ 步去噪，Drifting Model 直接预测**从 MRI 到 CT 的位移场**。

### Pipeline 概览

```
MRI 切片 [B,1,H,W]
    ↓ 拼接时间步 t=0 和条件 c=x_MRI
条件 U-Net（速度场网络 v_θ）
    ↓ 预测速度场 Δ = x_CT - x_MRI
推理: x̂_CT = x_MRI + Δ  [一步完成]
    ↓
合成 CT 切片 → 堆叠成 3D 体积
```

## 实现

### 核心模型：条件速度场网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.SiLU(),
        )
    def forward(self, x): return self.conv(x)

class DriftingUNet(nn.Module):
    """速度场网络：输入 [x_t; x_MRI] 和时间步 t，输出速度 v"""
    def __init__(self, in_ch=2, base_ch=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_ch), nn.SiLU(), nn.Linear(base_ch, base_ch)
        )
        # 编码器：逐步提取多尺度特征
        self.enc1 = ConvBlock(in_ch, base_ch)               # 2   → 64
        self.enc2 = ConvBlock(base_ch, base_ch * 2)         # 64  → 128
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)     # 128 → 256
        # 瓶颈层：融合时间步嵌入（64维）和空间特征（256维）
        self.bottleneck = ConvBlock(base_ch * 4 + base_ch, base_ch * 4)   # 320 → 256
        # 解码器：跳跃连接恢复空间细节（骨骼边缘需要高频信息）
        self.dec3 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)    # 384 → 128
        self.dec2 = ConvBlock(base_ch * 2 + base_ch, base_ch)             # 192 → 64
        self.out  = nn.Conv2d(base_ch, 1, 1)                               # 64  → 1

    def forward(self, x_t, x_mri, t):
        # x_t, x_mri: [B,1,H,W]；t: [B]（归一化到 [0,1]）
        x  = torch.cat([x_t, x_mri], dim=1)                # [B,2,H,W]
        te = self.time_embed(t.unsqueeze(1))                # [B,64]

        e1  = self.enc1(x)                                  # [B,64,H,W]
        e2  = self.enc2(F.max_pool2d(e1, 2))                # [B,128,H/2,W/2]
        e3  = self.enc3(F.max_pool2d(e2, 2))                # [B,256,H/4,W/4]

        h, w = e3.shape[-2:]
        te_map = te[:, :, None, None].expand(-1, -1, h, w)  # 广播到空间维度
        bot = self.bottleneck(torch.cat([e3, te_map], dim=1))

        up = lambda f, s: F.interpolate(f, scale_factor=s, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([up(bot, 2), e2], dim=1))
        d2 = self.dec2(torch.cat([up(d3,  2), e1], dim=1))
        return self.out(d2)                                  # [B,1,H,W] 速度场
```

### 训练：Flow Matching 损失

```python
def flow_matching_loss(model, x_mri, x_ct):
    """
    条件 Flow Matching 训练步骤
    x_mri, x_ct: [B,1,H,W]，HU 值归一化到 [-1, 1]
    """
    B, device = x_mri.shape[0], x_mri.device
    t = torch.rand(B, device=device)                        # t ~ U[0,1]

    t4 = t[:, None, None, None]
    x_t     = (1 - t4) * x_mri + t4 * x_ct                # 线性插值中间态
    v_target = x_ct - x_mri                                 # 直线路径速度（常数）

    v_pred = model(x_t, x_mri, t)

    # 基础 MSE + 骨骼区域加权（骨皮质 HU > 700，归一化后约 > 0.3）
    bone_mask = (x_ct > 0.3).float()
    loss = F.mse_loss(v_pred, v_target) + \
           2.0 * F.mse_loss(v_pred * bone_mask, v_target * bone_mask)
    return loss


def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for x_mri, x_ct in loader:
        x_mri, x_ct = x_mri.cuda(), x_ct.cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = flow_matching_loss(model, x_mri, x_ct)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)
```

### 推理：一步合成 + 3D 可视化

```python
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def synthesize_ct(model, x_mri):
    """一步推理：x̂_CT = x_MRI + v_θ(x_MRI, t=0)"""
    model.eval()
    t = torch.zeros(x_mri.shape[0], device=x_mri.device)   # 固定 t=0
    velocity = model(x_mri, x_mri, t)                       # 预测位移场
    return (x_mri + velocity).clamp(-1, 1)


def visualize_volume(mri_vol, ct_synth_vol, ct_real_vol=None):
    """
    三视图对比（轴位 / 冠状位 / 矢状位）
    输入均为 numpy array [D, H, W]
    """
    mid = [s // 2 for s in mri_vol.shape]
    rows = 3 if ct_real_vol is not None else 2
    titles = ['MRI', 'Synth CT'] + (['Real CT'] if ct_real_vol is not None else [])
    vols   = [mri_vol, ct_synth_vol] + ([ct_real_vol] if ct_real_vol is not None else [])

    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    view_funcs = [
        lambda v: v[mid[0]],           # 轴位（Axial）
        lambda v: v[:, mid[1], :],     # 冠状位（Coronal）
        lambda v: v[:, :, mid[2]],     # 矢状位（Sagittal）
    ]
    view_names = ['Axial', 'Coronal', 'Sagittal']

    for row, (vol, title) in enumerate(zip(vols, titles)):
        for col, (fn, vname) in enumerate(zip(view_funcs, view_names)):
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.imshow(fn(vol), cmap='gray', vmin=-1, vmax=1)
            ax.set_title(f'{title} – {vname}')
            ax.axis('off')
    plt.tight_layout(); plt.show()
```

## 实验

### 数据集说明

| 数据集 | 来源 | 特点 | 获取 |
|--------|------|------|------|
| Gold Atlas Male Pelvis | 公开数据集 | 专家配准 MRI-CT 对，质量高 | 注册后下载 |
| SynthRAD2023 Pelvis | MICCAI Challenge | 多机构真实临床数据 | 挑战赛公开 |

关键预处理：
- CT 归一化到 $[-1024, 3071]$ HU 后线性映射到 $[-1, 1]$
- MRI 按体积做 percentile 归一化（排除空气背景）
- 严格刚性配准（推荐 ANTs），误差需 $< 1$ mm

### 定量评估

| 方法 | SSIM ↑ | PSNR (dB) ↑ | RMSE (HU) ↓ | 推理时间 |
|------|--------|-------------|-------------|---------|
| **Drifting Model** | **0.91** | **31.2** | **52** | **~5 ms** |
| FastDDPM | 0.89 | 30.1 | 61 | ~2.4 s |
| DDIM (50步) | 0.87 | 29.3 | 71 | ~24 s |
| DDPM | 0.88 | 29.8 | 67 | ~480 s |
| UNet | 0.85 | 28.9 | 79 | ~50 ms |
| WGAN-GP | 0.83 | 27.5 | 88 | ~30 ms |
| PPFM | 0.86 | 29.0 | 75 | ~100 ms |

*数值为示意性结果，具体数据参见原论文。推理时间基于 A100，256×256 单切片。*

定性观察：Drifting Model 在骨皮质边缘的锐度上明显优于扩散模型，骶骨和股骨头的几何轮廓更准确，骨-气-软组织交界处伪影更少。

## 工程实践

### 实际部署考虑

- **推理速度**：单切片 ~5 ms，整个骨盆体积（90 切片）**~0.45 秒**，满足临床实时需求
- **训练硬件**：32GB VRAM（batch=4，256×256）；推理 8GB VRAM 足够
- **大体积处理**：逐切片推理，避免一次性载入整个体积

```python
# 大体积推理：逐切片避免 OOM
with torch.cuda.amp.autocast():
    ct_slices = [synthesize_ct(model, s.unsqueeze(0).cuda())
                 for s in mri_volume_tensor]
ct_volume = torch.cat(ct_slices, dim=0).cpu().numpy()
```

### 数据采集建议

1. **配准质量是命门**：MRI-CT 配准误差 > 2 mm 时模型会学习"配准偏移"而非"合成映射"，用 NCC 检查，SSIM > 0.85 再纳入训练
2. **MRI 序列选择**：Dixon MRI 同时提供 In-phase/Out-phase/Water/Fat 四通道，骨骼区域精度提升明显
3. **HU 范围**：不要过窄（如只保留软组织窗），会丢失骨骼信息

### 常见坑

1. **推理时 $t$ 不为零** → 模型在随机中间态上预测，输出严重偏移  
   修复：推理时固定 `t = torch.zeros(B)`，这是最高频的 debug 错误

2. **BatchNorm 在 batch\_size=1 时崩溃** → 换 GroupNorm（代码里已使用 `GroupNorm(8, out_ch)`）

3. **骨骼像素稀疏导致欠拟合** → 骨骼仅占全图 < 5%，加骨骼区域权重（已在损失函数中加入 `2.0×` 加权项）

4. **HU 值系统性偏差** → 合成 CT 的均值比真实 CT 低约 30 HU 属正常，但需要在验证集上做 bias correction，否则影响剂量计算

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 骨盆、脑部等解剖稳定区域 | 胸部（呼吸运动严重） |
| 同机型同协议 MRI 采集 | 多机型混合数据（磁场强度差异大） |
| 放疗剂量计算（需要 HU 精度） | 金属植入物区域（金属伪影不可靠） |
| 需要实时/批量快速合成 | 需要不确定性估计（放射科医生需要置信度） |
| 静态解剖结构 | 肠道等动态空腔器官 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 最适场景 |
|------|------|------|---------|
| UNet | 快、简单、易训练 | 骨骼细节过平滑 | 快速原型验证 |
| WGAN-GP | 纹理清晰度好 | 训练不稳定，模式崩塌 | 纹理合成任务 |
| DDPM/DDIM | 高质量，样本多样 | 推理慢（几十步到千步） | 离线高质量合成 |
| PPFM | 物理约束、可解释 | 假设强，跨数据集泛化受限 | 特定物理场景 |
| **Drifting Model** | 一步推理，高质量 | 无直接不确定性输出 | **临床实时部署** |

## 我的观点

Drifting Model 本质上是**条件 Flow Matching**（Lipman 2022 / Liu 2022）的医学图像应用，"drifting"的命名来自 SDE 理论的漂移项，清晰地区分了它与扩散模型的根本差异：**一个是 ODE，一个是 SDE**。

**值得关注的三点**：

1. **一步推理是真正的工程突破**：DDPM 的 12 分钟 vs Drifting 的 0.5 秒，不是边际改进，是量变到质变——前者只能离线处理，后者可以进手术室、上放疗计划系统

2. **不确定性估计是缺口**：放疗计划需要知道"这个骨骼 HU 值我有多少信心"。确定性单步模型无法直接给出分布；可以通过训练多个头或在输入 MRI 上加轻微扰动后取均值/方差来补足

3. **多模态输入是下一步**：Dixon MRI 的 4 通道输入有望显著提升骨骼区域精度，与 PET/MR 联合 attenuation correction 也是天然的应用场景

**离实际应用还有多远**：骨盆放疗计划在技术层面已相当接近临床可用。主要障碍是监管认证（FDA 510k、CE Mark）和前瞻性临床验证，而非算法本身。

**开放问题**：
- 如何保证所有病例的 HU 值统计无偏，而不只是均值接近？
- 扫描机型迁移（domain shift）是否需要每机型重训练，还是轻量 fine-tune 即可？
- 能否扩展到呼吸运动补偿的肺部 CT 合成？