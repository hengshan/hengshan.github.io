---
layout: post-wide
title: "城市热岛逆问题：用扩散模型生成多样化降温植被方案"
date: 2026-04-15 08:02:45 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.13028v1
generated_by: Claude Code CLI
---

## 一句话总结

给定"降温 2°C"的目标，自动生成多种在空间上合理的城市绿化布局方案——这是一个用扩散模型解决的条件逆问题。

## 为什么这个问题重要？

城市热岛效应是真实的工程问题：高密度建成区的地表温度（Land Surface Temperature, LST）可比周边郊区高出 5-10°C，直接影响能耗、健康和气候适应性。

传统工作流是**前向**的：

```
植被覆盖图 + 城市形态 → 地表温度预测
```

城市规划师真正需要的是**逆向**工作流：

```
目标降温幅度 → 应该在哪里种树？种多少？
```

这个逆问题有两个核心难点：

1. **病态性（ill-posed）**：多种不同的植被布局方案可以产生相同的区域平均降温效果
2. **数据稀缺**：卫星图像与温度联合数据集规模有限，很难训练大模型

传统回归模型面对多解问题只能输出"平均解"，扩散模型则天然适合建模这种**一对多**的条件分布。

这篇论文（[arxiv:2604.13028](https://arxiv.org/abs/2604.13028v1)）提出的 **Conflated Inverse Modeling** 框架把前向预测模型与扩散生成模型结合，在保证物理合理性的同时输出多样化方案。

---

## 背景知识

### 前向模型 vs 逆模型

设 $\mathbf{v}$ 为植被图，$\Delta T$ 为温度变化量：

- **前向模型**：$\hat{\Delta T} = F(\mathbf{v}, \mathbf{u})$，其中 $\mathbf{u}$ 是城市形态特征
- **逆模型**：$\mathbf{v} \sim p(\mathbf{v} \mid \Delta T^*, \mathbf{u})$，从后验分布采样

逆问题的后验 $p(\mathbf{v} \mid \Delta T^*)$ 是多峰的（multimodal）——这正是扩散模型的主场。

### 扩散模型用于条件生成

DDPM 的逆过程：

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{c}) = \mathcal{N}\left(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t, \mathbf{c}),\ \sigma_t^2 \mathbf{I}\right)
$$

其中 $\mathbf{c}$ 是条件（目标降温幅度 + 城市形态）。

**前向模型引导（Forward Model Guidance）** 在推理时施加额外梯度：

$$
\tilde{\mu} = \mu_\theta(\mathbf{x}_t, t, \mathbf{c}) + \lambda \cdot \nabla_{\mathbf{x}_t} \log p(\Delta T^* \mid F(\hat{\mathbf{x}}_0))
$$

"Conflated"的核心就在这里：逆向扩散过程被前向物理模型的梯度持续校正。

---

## 核心方法

### 直觉解释

```
城市形态图 + 目标降温Δ T*
        │
        ▼
┌──────────────────────────┐
│   扩散逆模型（生成器）     │  ← 产生多样化植被候选
│   + 前向模型梯度引导       │  ← 确保物理合理性
└──────────────────────────┘
        │
        ▼
植被方案 1、方案 2、方案 3...   (每次采样不同但都能达到Δ T*)
```

### Pipeline 概览

```
Landsat/Sentinel 影像
    │
    ├─→ [前向模型训练]  (植被图 + 城市形态) → 预测 LST
    │
    └─→ [扩散模型训练]  以 LST 目标 + 城市形态为条件
                           ↓
                   [推理：conflated 采样]
                   扩散去噪 + 前向模型梯度引导
                           ↓
                   多样化植被布局方案
```

---

## 实现

### 环境配置

```bash
pip install torch torchvision einops matplotlib rasterio
```

### 前向模型：UNet 预测地表温度变化

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)

class ForwardUNet(nn.Module):
    """
    前向模型：(植被图, 城市形态) → 地表温度变化图
    输入: [B, 2, H, W]  (植被 NDVI + 不透水面覆盖度)
    输出: [B, 1, H, W]  (LST 相对变化量, 单位 °C)
    """
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(2, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.bottleneck = ConvBlock(128, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b  = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))
        return self.head(d1)
```

### 条件扩散模型：生成植被方案

```python
class SinusoidalPE(nn.Module):
    """时间步位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-torch.arange(half, device=t.device) * 8.0 / half)
        emb = t[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ConditionalDenoiser(nn.Module):
    """
    条件去噪网络
    输入: 含噪植被图 x_t [B,1,H,W] + 城市形态 u [B,1,H,W]
          + 时间步 t [B] + 目标降温标量 delta_T [B]
    输出: 预测噪声 [B,1,H,W]
    """
    def __init__(self, base_ch=64):
        super().__init__()
        self.time_emb = SinusoidalPE(base_ch)
        self.temp_proj = nn.Linear(1, base_ch)  # 将标量 ΔT 投影为向量

        # 将 x_t 和城市形态拼接作为输入
        self.enc1 = ConvBlock(2, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.mid  = ConvBlock(base_ch * 2, base_ch * 2)
        self.dec2 = ConvBlock(base_ch * 2 * 2, base_ch)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x_t, u, t, delta_T):
        # 时间和温度条件嵌入
        cond = self.time_emb(t) + self.temp_proj(delta_T.unsqueeze(-1))
        # ... (cond 通过 AdaGN 注入各层，此处简化为加法)
        inp = torch.cat([x_t, u], dim=1)
        e1 = self.enc1(inp)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        m  = self.mid(e2)
        d2 = self.dec2(torch.cat([m, e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))
        return self.head(d1)
```

### Conflated 推理：前向模型引导的扩散采样

这是整个框架最关键的部分——在去噪每一步，前向模型提供梯度将生成方向拉向物理可行解：

```python
@torch.no_grad()
def conflated_sample(denoiser, forward_model, u, delta_T_target,
                     T=1000, guidance_scale=5.0, device='cuda'):
    """
    Conflated 采样：扩散去噪 + 前向模型梯度引导
    delta_T_target: 目标降温幅度 (标量, 负值表示降温)
    """
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)

    B, _, H, W = u.shape
    x = torch.randn(B, 1, H, W, device=device)  # 从纯噪声开始
    delta_T = torch.full((B,), delta_T_target, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        alpha_t = alphas_bar[t]

        # 标准扩散去噪
        eps_pred = denoiser(x, u, t_batch, delta_T)
        x0_pred = (x - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # 前向模型引导：确保 x0_pred 能产生目标降温
        with torch.enable_grad():
            x0_g = x0_pred.detach().requires_grad_(True)
            veg_input = torch.cat([x0_g, u], dim=1)  # [B, 2, H, W]
            lst_pred = forward_model(veg_input).mean()  # 区域平均温度
            loss = (lst_pred - delta_T_target) ** 2
            grad = torch.autograd.grad(loss, x0_g)[0]

        x0_pred = x0_pred - guidance_scale * grad.detach()

        # DDPM 后验均值
        if t > 0:
            noise = torch.randn_like(x)
            x = alphas[t].sqrt() * x0_pred + betas[t].sqrt() * noise
        else:
            x = x0_pred

    return x  # 返回生成的植被布局 (NDVI map)
```

### 可视化多样性验证

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_diverse_solutions(samples, u, delta_T_target, forward_model):
    """
    对比多次采样结果，验证方案多样性和降温有效性
    samples: [N, 1, H, W]  N 个独立采样的植被方案
    """
    N = len(samples)
    fig, axes = plt.subplots(2, N + 1, figsize=(4 * (N + 1), 8))

    # 显示城市底图
    axes[0, 0].imshow(u[0, 0].cpu(), cmap='gray')
    axes[0, 0].set_title('城市形态\n(不透水面)')
    axes[1, 0].set_title(f'目标降温: {delta_T_target:.1f}°C')
    axes[1, 0].axis('off')

    actual_temps = []
    for i, veg in enumerate(samples):
        # 显示生成的植被方案
        axes[0, i+1].imshow(veg[0, 0].cpu(), cmap='Greens', vmin=-1, vmax=1)
        axes[0, i+1].set_title(f'方案 {i+1}')

        # 用前向模型计算实际降温效果
        with torch.no_grad():
            inp = torch.cat([veg, u], dim=1)
            pred_temp = forward_model(inp).mean().item()
        actual_temps.append(pred_temp)
        axes[1, i+1].bar(['预测降温'], [pred_temp], color='steelblue')
        axes[1, i+1].axhline(delta_T_target, color='r', linestyle='--', label='目标')
        axes[1, i+1].set_ylim(delta_T_target - 1, delta_T_target + 1)
        axes[1, i+1].set_title(f'Δ T = {pred_temp:.2f}°C')

    plt.tight_layout()
    plt.savefig('diverse_vegetation_solutions.png', dpi=150)
    print(f"方案温度方差: {np.std(actual_temps):.3f}°C（多样性），"
          f"均值误差: {abs(np.mean(actual_temps) - delta_T_target):.3f}°C（准确性）")
```

---

## 实验

### 数据集说明

论文使用的数据来自 Landsat 8/9 和 Sentinel-2 卫星影像：

| 数据源 | 内容 | 分辨率 |
|--------|------|--------|
| Landsat 热红外波段 | 地表温度 (LST) | 100m |
| Sentinel-2 NDVI | 植被覆盖指数 | 10m |
| 城市建成区数据 | 不透水面比例 | 30m |

数据获取难点：LST 受云覆盖影响严重，有效样本量有限，这正是论文强调"数据稀缺"场景的背景。

### 定量评估

| 方法 | 温度预测误差 (MAE) | 生成多样性 (FID) | 方案可行率 |
|------|-----------------|----------------|----------|
| 直接回归 | ~0.5°C | — | 100%（但无多样性）|
| CVAE | ~0.8°C | 较高 | ~75% |
| **Conflated (本文)** | **~0.4°C** | **低（高质量）** | **~90%** |

> 注：以上数字为根据论文描述的量级估计，具体数值请参考原文。

---

## 工程实践

### 训练注意事项

前向模型和扩散模型分阶段训练，顺序很重要：

```python
# 阶段 1：先训练前向模型，确保温度预测可靠
optimizer_F = torch.optim.AdamW(forward_model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
# ... 在 (vegetation, urban) → LST 配对数据上训练

# 阶段 2：冻结前向模型，训练扩散去噪器
for p in forward_model.parameters():
    p.requires_grad_(False)
optimizer_D = torch.optim.AdamW(denoiser.parameters(), lr=2e-5)
```

### 推理时 guidance_scale 的影响

```python
import torch, torch.nn as nn, torch.nn.functional as F

# ... (ConvBlock: Conv2d→GroupNorm→GELU ×2 省略)

class ForwardUNet(nn.Module):
    """前向模型：(植被图, 城市形态) → 地表温度变化图
    输入: [B, 2, H, W]  输出: [B, 1, H, W] (LST 变化, °C)
    """
    def __init__(self):
        super().__init__()
        # Encoder: 2→32→64→128, Bottleneck: 128→256
        self.enc1, self.enc2, self.enc3 = ConvBlock(2,32), ConvBlock(32,64), ConvBlock(64,128)
        self.bottleneck = ConvBlock(128, 256)
        # Decoder with skip connections
        self.dec3, self.dec2, self.dec1 = ConvBlock(384,128), ConvBlock(192,64), ConvBlock(96,32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1, e2, e3 = self.enc1(x), self.enc2(F.max_pool2d(e1:=self.enc1(x),2)), ...
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b  = self.bottleneck(F.max_pool2d(e3, 2))
        # Decoder (upsample + skip cat)
        d3 = self.dec3(torch.cat([F.interpolate(b,  scale_factor=2), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.head(d1)
```

### 常见坑

1. **LST 数据标准化问题** → 温度值范围跨度大（250~330K），训练前必须归一化到 [-1, 1]，否则梯度引导 scale 难以调整
2. **空间分辨率不匹配** → Landsat LST 100m vs Sentinel NDVI 10m，拼接前需重采样到统一分辨率（推荐 30m）
3. **季节性偏差** → 夏季和冬季的植被-温度关系差异很大，建议按季节分别训练或加入月份编码
4. **梯度引导中的 NaN** → `x0_pred.clamp(-1, 1)` 在梯度路径上可能产生零梯度，改用 `tanh` 更稳定

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 城市绿化规划、热岛缓解方案设计 | 需要精确像素级控制（该方法输出概率性） |
| 数据稀缺条件（卫星数据有限） | 实时决策（扩散推理慢，>10s/样本）|
| 需要多样化方案供规划师比选 | 动态场景（洪涝、火灾等快速变化） |
| 探索"可能解空间"而非单一最优解 | 精度优先而非多样性 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 线性回归 | 快速、可解释 | 无法建模非线性，单一解 | 初探/基准 |
| CVAE | 可生成多样样本 | 后验坍塌问题，多样性有限 | 中等复杂度 |
| GAN (条件) | 生成质量高 | 训练不稳定，模式崩塌 | 大数据场景 |
| **Conflated Diffusion** | 多样性 + 物理约束 | 推理慢，需要前向模型 | **数据稀缺 + 多解问题** |

---

## 我的观点

这篇论文的技术贡献在于将"物理前向模型作为引导信号"和"扩散生成多样解"两个思路显式地结合，而不是隐式地把物理知识编码进训练数据。

**值得关注的方向：**

- **更快的采样**：目前扩散推理慢，一致性模型（Consistency Models）或 DDIM 可以将步数从 1000 压缩到 10-50 步，让这类工具进入实际规划工作流
- **多目标扩展**：除降温外，同时优化雨水渗透、生物多样性等指标——本质上是多条件引导问题
- **3D 城市建模集成**：结合 CityGML 或 NeRF 的三维城市模型，植被布局可以更精细地考虑阴影遮挡等几何效应

**离实际应用的差距：**

卫星数据的空间分辨率（30-100m）仍然粗糙，城市里一棵树的尺度（~5m）无法直接分辨。结合无人机高分辨率数据是下一步，但标注成本极高。

这个框架的思路——"前向物理模型 + 扩散逆向生成"——是一种通用范式，不限于城市热岛，在气候适应、建筑节能、城市水文等领域都有潜力复用。