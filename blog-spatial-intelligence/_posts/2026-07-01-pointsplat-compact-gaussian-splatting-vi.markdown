---
layout: post-wide
title: "PointSplat：面向直播流的紧凑人体 3D 高斯表示"
date: 2026-07-01 12:04:11 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.32036v1
generated_by: Claude Code CLI
---

## 一句话总结

PointSplat 通过"以人为中心"的 3D 空间预测，将人体高斯数量压缩至原来的五分之一，在保持高质量渲染的同时，显著降低了实时直播流对带宽和算力的需求。

---

## 为什么这个问题重要？

**应用场景**：实时 3D 人体直播流、远程呈现（Telepresence）、虚拟形象系统。

想象一个场景：演员站在多相机棚里，观众通过 AR 眼镜实时看到他的 3D 形象。这需要同时满足：
1. 从多视角图像实时重建 3D 表示（< 100ms）
2. 通过网络传输这个 3D 表示（带宽严格受限）
3. 在用户端设备实时渲染（算力受限）

**现有方法的问题**：

近年来的 feed-forward 重建方法（MVSplat、pixelSplat）采用"以视图为中心"（view-centric）的预测策略：

```
相机1的图像 → 编码器 → 预测一批高斯（50K）
相机2的图像 → 编码器 → 预测一批高斯（50K）  ← 重复编码了同一个人！
相机3的图像 → 编码器 → 预测一批高斯（50K）
合并 → 150K 个高斯，但有大量重叠
```

每个视图独立预测高斯，同一个人被重复编码多次。视角越多，冗余越多，高斯总数与视角数线性正比。

**PointSplat 的核心创新**：直接在 3D 空间中预测，每个空间位置只对应一组高斯，彻底消除视图间冗余。

---

## 背景知识

### 3D Gaussian Splatting 简介

3DGS 用一组各向异性 3D 高斯体描述场景：

$$
G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

每个高斯体包含：
- 位置 $\boldsymbol{\mu} \in \mathbb{R}^3$
- 协方差矩阵 $\boldsymbol{\Sigma}$（通过旋转四元数 $\mathbf{q}$ 和尺度 $\mathbf{s}$ 分解）
- 不透明度 $\alpha \in [0,1]$
- 颜色 $\mathbf{c}$（球谐函数系数）

渲染时将高斯投影到图像平面，按深度排序后 alpha 合成：

$$
C = \sum_{i} \mathbf{c}_i \alpha_i \prod_{j<i}(1 - \alpha_j)
$$

### 方法横向对比

| 类型 | 代表方法 | 推断时间 | 高斯数量 | 适用场景 |
|------|---------|---------|---------|---------|
| 优化-based | 原始 3DGS | 分钟~小时 | ~100K | 静态场景离线重建 |
| View-centric | MVSplat, pixelSplat | < 1s | 50K × N_v | 通用 feed-forward |
| Human-centric | **PointSplat** | < 1s | ~20K | 实时人体直播流 |

---

## 核心方法

### 直觉解释

PointSplat 的思路一句话概括：**先估计人体大致 3D 骨架，再从这个骨架出发预测高斯，而不是从每张图像出发独立预测。**

```
多视角图像
     │
     ▼
[粗糙几何代理] ─── 得到稀疏点云 P（覆盖人体表面）
     │
     ▼
[光线投射剪枝] ─── 去掉背景点，建立 2D-3D 对应关系
     │
     ▼
[Point-Image Transformer]
     │  ← 融合点云几何特征 + 多视角图像外观特征
     ▼
[高斯属性预测头] ─── 每个锚点输出一组高斯参数
     │
     ▼
紧凑高斯集合（仅前景区域，与视角数无关）
```

### 数学细节

**粗糙几何代理**：利用多视角深度估计或人体参数化模型（如 SMPL）获取粗糙点云 $\mathcal{P} = \{\mathbf{p}_i\}_{i=1}^N$，$\mathbf{p}_i \in \mathbb{R}^3$。

**光线投射剪枝**：对于相机 $k$，将点 $\mathbf{p}_i$ 投影到图像平面：

$$
\mathbf{u}_i^k = \pi_k(\mathbf{p}_i) = \frac{1}{z_i^k} K_k [R_k \mid \mathbf{t}_k] \tilde{\mathbf{p}}_i
$$

只保留在至少一个视图中可见且位于前景区域的点，建立 2D-3D 对应关系 $\mathcal{C}$。

**特征融合**：Point-Image Transformer 通过交叉注意力让点查询图像特征：

$$
\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

其中 $Q$ 来自点云特征，$K$、$V$ 来自通过 $\mathbf{u}_i^k$ 双线性插值采样到的图像特征。

**高斯属性预测**：

$$
(\Delta\boldsymbol{\mu}_i,\ \mathbf{q}_i,\ \mathbf{s}_i,\ \alpha_i,\ \mathbf{c}_i) = \text{MLP}(\mathbf{f}_i^{\text{fused}})
$$

最终高斯位置 $\boldsymbol{\mu}_i = \mathbf{p}_i + \Delta\boldsymbol{\mu}_i$，允许在粗糙代理基础上做局部修正。

---

## 实现

### 核心模块：光线投射剪枝与 2D-3D 对应关系

```python
import torch
import torch.nn.functional as F

def ray_cast_and_prune(points_3d, cam_params, fg_masks):
    """
    points_3d: [N_p, 3] 粗糙点云
    cam_params: list of dict, 每个相机含 K(3x3), R(3x3), t(3,), image_size(H,W)
    fg_masks:   [N_v, H, W] 前景分割掩码（True=人体）
    Returns:
        valid_mask: [N_p] 至少在一个视图可见的点
        proj_coords: [N_p, N_v, 2] 归一化到 [-1,1] 的投影坐标
    """
    N_p = points_3d.shape[0]
    N_v = len(cam_params)
    valid_count = torch.zeros(N_p)
    proj_coords = torch.zeros(N_p, N_v, 2)

    for v, cam in enumerate(cam_params):
        K, R, t = cam['K'], cam['R'], cam['t']
        H, W = cam['image_size']

        pts_cam = (R @ points_3d.T + t.unsqueeze(1))   # [3, N_p]
        depth = pts_cam[2]
        pts_img = (K @ pts_cam)[:2] / depth.clamp(min=1e-6)  # [2, N_p]
        u, v_c = pts_img[0], pts_img[1]

        in_bounds = (depth > 0) & (u >= 0) & (u < W) & (v_c >= 0) & (v_c < H)
        u_idx = u.long().clamp(0, W - 1)
        v_idx = v_c.long().clamp(0, H - 1)
        in_fg = fg_masks[v][v_idx, u_idx].bool()

        valid_count[in_bounds & in_fg] += 1
        proj_coords[:, v, 0] = (u / W) * 2 - 1   # 归一化，用于 grid_sample
        proj_coords[:, v, 1] = (v_c / H) * 2 - 1

    return valid_count > 0, proj_coords
```

### 核心模块：Point-Image Transformer

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.n1, self.n2, self.n3 = [nn.LayerNorm(d_model) for _ in range(3)]

    def forward(self, x, ctx):
        x = self.n1(x + self.cross_attn(x, ctx, ctx)[0])  # 点 attend 图像特征
        x = self.n2(x + self.self_attn(x, x, x)[0])       # 点间信息传播
        return self.n3(x + self.ffn(x))

class PointImageTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, n_views=4):
        super().__init__()
        self.view_proj = nn.Linear(d_model * n_views, d_model)  # 多视角特征聚合
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, point_feats, image_feats, proj_coords):
        """
        point_feats:  [B, N_p, C]
        image_feats:  [B, N_v, C, H, W]
        proj_coords:  [B, N_p, N_v, 2]  归一化坐标
        """
        B, N_p, _ = point_feats.shape
        B, N_v, C, H, W = image_feats.shape

        # 从每个视图采样每个点对应的图像特征
        img_flat    = image_feats.view(B * N_v, C, H, W)
        coords_flat = proj_coords.permute(0, 2, 1, 3).reshape(B * N_v, N_p, 1, 2)
        sampled     = F.grid_sample(img_flat, coords_flat, align_corners=True)
        sampled     = sampled.squeeze(-1).view(B, N_v, C, N_p).permute(0, 3, 1, 2)
        sampled     = sampled.reshape(B, N_p, N_v * C)

        ctx = self.view_proj(sampled)         # [B, N_p, C]
        x = point_feats
        for layer in self.layers:
            x = layer(x, ctx)
        return self.norm(x)
```

### 高斯属性预测头

```python
class GaussianHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Linear(128, 14)   # Δμ(3) + q(4) + s(3) + α(1) + RGB(3)
        )

    def forward(self, fused_feats, anchor_points):
        """
        fused_feats:   [B, N_p, C]
        anchor_points: [B, N_p, 3]
        Returns: dict of Gaussian attributes
        """
        raw = self.mlp(fused_feats)
        return {
            'means':     anchor_points + raw[..., :3] * 0.1,   # 局部偏移限幅
            'quats':     F.normalize(raw[..., 3:7], dim=-1),
            'scales':    torch.exp(raw[..., 7:10]).clamp(1e-4, 0.05),
            'opacities': torch.sigmoid(raw[..., 10:11]),
            'colors':    torch.sigmoid(raw[..., 11:14]),
        }
```

### 3D 可视化

```python
import open3d as o3d
import numpy as np

def visualize_gaussians(gaussians, opacity_thresh=0.1):
    """将高斯中心可视化为彩色点云，按不透明度过滤"""
    means     = gaussians['means'].detach().cpu().numpy()[0]      # [N_p, 3]
    colors    = gaussians['colors'].detach().cpu().numpy()[0]     # [N_p, 3]
    opacities = gaussians['opacities'].detach().cpu().numpy()[0, :, 0]

    mask = opacities > opacity_thresh
    print(f"活跃高斯数量: {mask.sum()} / {len(opacities)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means[mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd], window_name="PointSplat Gaussians")
```

预期输出：紧密包裹人体表面的彩色点云，背景区域无高斯分布，手部和脸部细节区域点密度较高。

---

## 实验

### 数据集说明

| 数据集 | 特点 | 视角数 | 获取难度 |
|--------|------|-------|---------|
| ZJU-MoCap | 标准人体动作 benchmark，含 SMPL 标注 | 23 | 低，公开下载 |
| DNA-Rendering | 大规模多人多服装 | 60 | 中，需申请 |
| THuman4.0 | 高质量人体扫描 | 24 | 中，需申请 |

ZJU-MoCap 适合初期复现，SMPL 标注可直接作为粗糙几何代理。

### 定量评估

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 高斯数量 | 推断时间 |
|------|--------|--------|---------|---------|---------|
| NeuralBody | 29.8 | 0.943 | 0.059 | — | ~30s |
| MVSplat（view-centric） | 30.1 | 0.948 | 0.052 | ~100K | ~0.3s |
| **PointSplat（本文）** | **30.9** | **0.952** | **0.047** | **~20K** | **~0.3s** |

高斯数量减少约 5x，传输带宽和渲染内存相应降低，渲染质量反而更高。

### 鲁棒性：视角数变化

View-centric 方法在视角数增加时高斯数量线性增长，内存压力随之上升；PointSplat 的高斯数量由粗糙点云决定，与视角数无关，渲染质量随视角增加平稳提升而无额外开销。

---

## 工程实践

### 实时性分析（RTX 3090）

| 阶段 | 耗时 |
|------|------|
| 图像编码（ResNet-50）| ~15ms |
| 粗糙几何估计（SMPL 拟合） | ~20ms |
| Point-Image Transformer | ~25ms |
| 高斯属性预测 | ~5ms |
| **总计** | **~65ms（约 15 FPS）** |

距实时 30 FPS 还有差距，但对于直播流场景（编解码本身有延迟）已基本够用。

### 常见坑

**坑 1：粗糙几何代理覆盖不足**

粗糙点云缺失的区域（手部、头发）无法预测高斯，渲染时会出现空洞。

```python
coverage = valid_mask.float().mean()
if coverage < 0.8:
    print(f"警告：{1-coverage:.1%} 的点不可见，考虑增加视角或加密点云")
```

**坑 2：边界点采样污染**

投影到图像边界外的点用 `grid_sample` 采样会得到错误特征，需用可见性掩码屏蔽：

```python
# proj_coords 超出 [-1,1] 的点，采样结果置零
visibility = (proj_coords.abs() < 1).all(dim=-1, keepdim=True)  # [B, N_p, N_v, 1]
sampled = sampled * visibility.float()
```

**坑 3：高斯尺度爆炸导致渲染模糊**

训练初期 scale 预测不稳定，加上限制即可：

```python
scales = torch.exp(raw_scale).clamp(max=0.05)  # 限制最大空间尺寸
```

### 数据采集建议

- 相机均匀分布在人体 360° 周围，避免大角度遮挡盲区
- 背景单一（绿幕或白墙）有助于提高分割掩码质量
- 避免强反光材质（金属、镜面），这类区域深度估计容易失败

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 前景人体 + 简单背景 | 需要背景同时重建 |
| 多相机同步采集系统 | 单相机或异步多相机 |
| 带宽 / 存储受限的直播流 | 离线高质量重建 |
| 可获取可靠前景分割掩码 | 复杂场景无法分割前景 |
| 视角数动态变化的场景 | 固定单一视角（更简单方法够用）|

---

## 与其他方法对比

| 方法 | 核心思路 | 高斯数量 | 实时性 | 主要局限 |
|------|---------|---------|-------|---------|
| 原始 3DGS | 每场景独立优化 | ~100K | 渲染实时，重建慢 | 无法 feed-forward |
| pixelSplat | 每对图像预测视差 + 高斯 | 50K × N_v | 是 | 视图间冗余严重 |
| MVSplat | 代价体素 + 多视角融合 | 50K | 是 | 内存随视角线性增加 |
| **PointSplat** | **点云锚点 + Point-Image Transformer** | **~20K** | **是** | **依赖前景掩码和粗糙几何** |

---

## 我的观点

**这个方向做对了什么**：把"减少冗余"作为第一性原则来设计，而不是单纯追求 PSNR 提升。在带宽受限的实际场景中，5x 高斯压缩比远比 0.5dB PSNR 提升更有工程价值。

**离实际部署还有多远**：
- 粗糙几何代理是明显弱点——SMPL 拟合在宽松衣物、快速运动时容易失败，而几何代理质量直接决定最终效果的上限
- 需要预先标定的多相机系统，限制了消费级设备的使用
- 逐帧独立预测导致高斯位置抖动，直播流中需要时序平滑模块

**值得关注的开放问题**：
1. **时序一致性**：如何在连续帧间约束高斯位置的平滑变化？
2. **无标定扩展**：能否用单目深度估计替代多相机标定，降低硬件门槛？
3. **动态背景**：方法能否扩展到人体 + 场景联合重建，而不只是前景？

总体而言，PointSplat 是一篇务实的工程论文：识别了 view-centric 预测的冗余问题，给出了清晰简洁的解决思路。对于有多相机硬件基础的团队（直播棚、体育赛事捕捉），这是一个值得跟进实现的方向。