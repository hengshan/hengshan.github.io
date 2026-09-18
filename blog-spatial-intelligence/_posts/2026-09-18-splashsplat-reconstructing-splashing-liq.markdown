---
layout: post-wide
title: "SplashSplat：给会破碎的水花一个物理骨架"
date: 2026-09-18 12:02:39 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.20818v1
generated_by: Claude Code CLI
---

## 一句话总结

SplashSplat 用"每帧液体 SDF + 帧间水平集光流 + 拉格朗日粒子平流"搭出一副物理骨架，再让骨架上的粒子解码成局部高斯去做可微渲染，从而在真实多视角视频里重建出会撕裂、飞溅、重新聚合的液体——这是目前少数专门针对"破碎水花"而不是烟雾、清澈液柱或缓慢形变表面设计的重建方法，论文同时放出了 20 组同步多视角真实水花数据集。

## 为什么这个问题重要？

液体飞溅的重建，比大部分人想象得难得多。原因不是渲染难，而是**观测本身就很稀缺**：

- 一片水花从撕裂成韧带（ligament）到碎成雾状液滴，往往只存在几帧的时间，根本来不及像刚体或人体那样做长时程特征跟踪；
- 水面几乎无纹理，外观强烈依赖视角（高光、折射、反射混杂），传统基于光度一致性的特征匹配会直接失效；
- 拓扑在不断变化——一片水膜可以在下一帧分裂成十几颗液滴——这意味着"跟踪同一个点随时间运动"这件事本身在物理上就是不适定的。

正因如此，此前的重建研究要么退而求其次去做烟雾（体密度场，拓扑变化温和），要么做游泳池水面这种缓慢形变的显式曲面，要么用合成水替代真实拍摄数据。**没有人发布过同步多视角的真实水花数据集**，这也是这篇工作愿意先把数据集这件苦活做完的原因。

应用场景也很直接：影视特效的真实水花采集、机器人操作液体（倒水、搅拌）时的感知与仿真校准、以及任何需要"液体数字资产"的图形学管线。

核心创新可以浓缩成一句话：**只在观测能够约束的地方强加物理结构**——不是硬套一个全局 Navier-Stokes 求解器，而是用多视角 mask 能可靠估计出的隐式曲面（SDF）和曲面间的运动场，去约束一小撮携带高斯的粒子，粒子之间的间隙留给可微渲染自己去优化。

## 背景知识

### 3D 表示方式怎么选

| 表示 | 优点 | 对"破碎水花"的问题 |
|---|---|---|
| 点云 / 体素 | 直接、易融合多视角 | 分辨率与拓扑变化的粒子数矛盾，稀疏区域没法表达连续曲面 |
| NeRF 隐式场 | 视角相关外观建模强 | 每帧独立优化代价高，缺乏帧间物理一致性，动态液体几乎无法收敛 |
| 3D Gaussian Splatting（静态/动态） | 渲染快、可微、质量高 | 动态 3DGS 通常靠"跟踪同一批高斯随时间变形"，但水花的高斯身份在撕裂/合并时根本不存在，会导致漂移和伪影 |
| SDF（符号距离场） | 天然表达闭合曲面，适合从多视角 mask 融合 | 单独用 SDF 无法直接渲染出高频外观细节 |
| 拉格朗日粒子 | 天然承载"物质点"的运动语义，适合和流体求解器语言对接 | 单独用粒子渲染需要额外解码到可渲染基元 |

SplashSplat 的选择是**分层组合**：SDF 负责"这一帧液体占据了哪里"（观测直接可约束），粒子负责"物质怎么从上一帧流到这一帧"（用 SDF 间的水平集光流间接约束），高斯负责"渲染出来长什么样"（用照片级损失约束）。三层各司其职，谁的信号强就让谁主导。

### 相机模型

数据集用 7 台同步标定的 4K/60fps 相机，也就是标准的多视角标定设置：已知内参 $K_i$、外参 $[R_i \mid t_i]$，每个体素/粒子投影到像平面后与该视角的液体 mask 做比对。这是后面视觉壳（visual hull）式 SDF 融合的前提。

## 核心方法

### 直觉解释

想象水花的每一帧先被"雕"成一个隐式曲面（像一块半透明的果冻，内部为负、外部为正的距离场）。相邻两帧的果冻形状不一样，先算出一个"这块果冻怎么变形才能变成下一块"的粗糙速度场——这一步只关心几何，不关心颜色。然后撒一把"记号笔粒子"在果冻表面，让它们顺着速度场游动，游到新位置后再看一眼新的真实观测把它们轻轻拉回曲面上；如果某处水花突然炸出了新的碎屑而记号笔粒子没跟上，就在那里补种新粒子。最后每颗粒子背个小高斯，负责把自己周围渲染成照片。

### 数学细节

**第一层：SDF 观测项。** 给定第 $t$ 帧的 $N$ 个视角 mask $M_t^{(i)}$，融合出符号距离场 $\phi_t(x)$，目标是让曲面的零水平集与所有视角的轮廓一致：

$$
\phi_t = \arg\min_\phi \sum_{i=1}^{N} \left\| \pi_i(\{x : \phi(x)=0\}) - M_t^{(i)} \right\|^2
$$

其中 $\pi_i$ 是第 $i$ 台相机的投影算子。

**第二层：水平集平流约束。** 给定 $\phi_t$ 和 $\phi_{t+1}$，求一个速度场 $v(x)$ 使得沿速度场平流后的曲面与下一帧观测吻合，这本质是水平集方程的离散形式：

$$
\phi_{t+1}(x) \approx \phi_t(x - v(x)\,\Delta t)
$$

再加平滑正则（相邻体素速度不应突变）：

$$
\mathcal{L}_{\text{flow}} = \left\| \phi_{t+1} - \phi_t(x - v\Delta t) \right\|_1 + \lambda \left\| \nabla v \right\|_2^2
$$

**第三层：粒子修正。** 粒子 $p$ 平流后位置为 $p' = p + v(p)\Delta t$，再用 SDF 的一步牛顿投影把它拉回新曲面：

$$
p'' = p' - \phi_{t+1}(p') \cdot \frac{\nabla \phi_{t+1}(p')}{\|\nabla \phi_{t+1}(p')\|^2}
$$

**第四层：渲染损失。** 每个粒子的特征通过一个小 MLP 解码出局部高斯参数 $(\mu, s, q, \alpha, c)$（均值偏移、缩放、旋转、不透明度、颜色/球谐系数），标准可微高斯光栅化后与真实图像做光度损失。

### Pipeline 概览

```
多视角视频+mask
   → 逐帧 SDF 融合（视觉壳/多视角一致性）
   → 相邻帧水平集光流 → 粗速度场
   → 粒子平流 + SDF 修正 + 覆盖缺失处重播种
   → 粒子特征 → MLP → 局部高斯
   → 可微光栅化 → 与真实图像做光度损失，反传更新粒子特征/MLP
```

## 实现

下面代码是为了理解算法骨架而写的精简版本，不是论文的官方实现，省略了标定读取、mask 前处理、marching cubes 网格化和完整的可微光栅化内核。

### 环境配置

```bash
# 核心依赖：几何、优化、可微渲染
pip install torch numpy scipy trimesh
# 可微高斯光栅化建议直接用现成库，而不是自己写 CUDA kernel
pip install gsplat
```

### 1. 多视角 mask 融合出逐帧 SDF

```python
import numpy as np
from scipy.ndimage import distance_transform_edt

def fuse_sdf_from_masks(masks, cameras, grid_res=128, bounds=(-1, 1)):
    """masks: list[(H,W) bool array]; cameras: list[(K, R, t)]"""
    xs = np.linspace(*bounds, grid_res)
    grid = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1)  # (G,G,G,3)
    votes = np.zeros(grid.shape[:3], dtype=np.int32)

    for mask, (K, R, t) in zip(masks, cameras):
        # 世界坐标 -> 相机坐标 -> 像素坐标
        pts_cam = (grid.reshape(-1, 3) - t) @ R.T
        uvw = pts_cam @ K.T
        uv = (uvw[:, :2] / uvw[:, 2:3]).astype(int)
        H, W = mask.shape
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        inside = np.zeros(len(uv), dtype=bool)
        inside[valid] = mask[uv[valid, 1], uv[valid, 0]]
        votes += inside.reshape(grid.shape[:3])

    # 全部视角都投在 mask 内部才算占据（视觉壳）
    occupancy = votes == len(masks)
    inside_dist = distance_transform_edt(occupancy)
    outside_dist = distance_transform_edt(~occupancy)
    phi = outside_dist - inside_dist  # 内部为负，外部为正
    return phi
```

### 2. 相邻帧水平集光流：估计粗速度场

```python
import torch
import torch.nn.functional as F

def estimate_flow(phi_t, phi_next, iters=200, lr=1e-2, smooth_w=0.1):
    """phi_t, phi_next: (G,G,G) torch tensor, 单位体素网格坐标系"""
    G = phi_t.shape[0]
    v = torch.zeros(1, 3, G, G, G, requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr)

    base_grid = make_identity_grid(G)  # (1,G,G,G,3)，归一化到 [-1,1]
    for _ in range(iters):
        opt.zero_grad()
        warp_grid = base_grid - v.permute(0, 2, 3, 4, 1)
        warped_phi = F.grid_sample(
            phi_t[None, None], warp_grid, align_corners=True
        )
        data_term = F.l1_loss(warped_phi[0, 0], phi_next)
        smooth_term = (v[:, :, 1:] - v[:, :, :-1]).pow(2).mean()
        loss = data_term + smooth_w * smooth_term
        loss.backward()
        opt.step()
    return v.detach()[0]  # (3,G,G,G)
```

### 3. 拉格朗日粒子：平流 + SDF 修正 + 重播种

```python
def advect_and_correct(particles, v_field, phi_next, dt=1.0, lr=1.0):
    """particles: (N,3) torch tensor；v_field/phi_next 支持三线性采样"""
    v_at_p = trilinear_sample(v_field, particles)      # (N,3)
    p_next = particles + v_at_p * dt

    phi_at_p = trilinear_sample(phi_next, p_next)       # (N,1)
    grad_phi = trilinear_grad(phi_next, p_next)          # (N,3)
    grad_phi = grad_phi / (grad_phi.norm(dim=-1, keepdim=True) + 1e-6)
    p_corrected = p_next - lr * phi_at_p * grad_phi       # 牛顿投影回零水平集

    return p_corrected

def reseed_missing_coverage(particles, phi_next, surface_samples, coverage_thresh=0.05):
    """在真实曲面上采样点，若离最近粒子太远，说明覆盖丢失（例如新炸出的水滴），补种新粒子"""
    dists = torch.cdist(surface_samples, particles).min(dim=1).values
    uncovered = surface_samples[dists > coverage_thresh]
    return torch.cat([particles, uncovered], dim=0)
```

### 4. 粒子解码为局部高斯 + 渲染损失

```python
import torch.nn as nn
from gsplat import rasterization  # 现成的可微高斯光栅化

class ParticleToGaussian(nn.Module):
    def __init__(self, feat_dim=16, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3 + 3 + 4 + 1 + 3)  # 偏移、缩放、旋转、不透明度、颜色
        )

    def forward(self, positions, features):
        out = self.net(torch.cat([positions, features], dim=-1))
        d_mu, s, q, alpha, color = torch.split(out, [3, 3, 4, 1, 3], dim=-1)
        means = positions + torch.tanh(d_mu) * 0.02   # 限制偏移幅度
        scales = torch.exp(s) * 0.01
        quats = F.normalize(q, dim=-1)
        opacities = torch.sigmoid(alpha)
        colors = torch.sigmoid(color)
        return means, scales, quats, opacities, colors

def photometric_step(decoder, particles, features, gt_image, camera):
    means, scales, quats, opacities, colors = decoder(particles, features)
    rendered, _, _ = rasterization(means, quats, scales, opacities, colors, **camera)
    loss = F.l1_loss(rendered, gt_image)
    return loss
```

### 完整训练循环（引用前面定义的模块）

一个训练迭代大致是：用第 2 步得到的速度场对第 3 步的粒子做平流修正与重播种，再用第 4 步的 `ParticleToGaussian` 渲染并计算光度损失，同时保留一个较小权重的 SDF 一致性项（渲染出的高斯的隐式占据应大致落在 `phi_next` 的零水平集内），反传更新粒子特征与解码器权重。速度场本身在第 2 步预计算好后通常保持冻结，只在少数实现里联合微调。

## 实验

### 数据集说明

论文自建了 **20 个真实场景**，覆盖从连贯水柱到剧烈飞溅的谱系，用 **7 台同步标定的 4K/60fps 相机**拍摄，并**人工精修**了每个视角的液体与容器 mask，还给出了固定的训练/评测切分。这类数据的采集门槛很高——多相机硬件同步、标定精度、以及逐帧 mask 精修的人工成本，都不是普通实验室能轻松复现的，这也是为什么此前一直没有这样的数据集。此外论文还用了一个合成基准做交叉验证。

### 定量评估

摘要中没有公布具体的 PSNR/SSIM 数值，这里只按论文陈述的结论做定性对比，不编造具体分数：

| 维度 | Dynamic 3DGS 基线 | SplashSplat |
|---|---|---|
| 渲染质量（真实拍摄） | 中 | 优于基线 |
| 运动物理合理性 | 低（高斯轨迹易漂移/瞬移） | 更连贯，符合流体运动趋势 |
| 训练成本 | 高 | 更低 |
| 拓扑变化（撕裂/合并）处理 | 弱 | 有专门机制（重播种） |

### 定性结果

预期的可视化对比应该是：在水膜撕裂成液滴的瞬间，基线动态 3DGS 容易出现高斯"瞬移"或拖影伪影（因为它假设了高斯身份在时间上连续），而 SplashSplat 由于有 SDF 观测约束 + 重播种机制，新生成的液滴能够"平滑地长出来"而不是凭空出现。失败案例大概率出现在：多层水花互相遮挡导致 mask 融合出的 SDF 本身就错误，以及粒子密度不足以覆盖极细的韧带结构时。

## 工程实践

### 实际部署考虑

- **实时性**：这是一个逐场景优化（per-scene optimization）的重建方法，不是前馈网络，训练本身就要跑到收敛，谈不上实时重建；训练完成后的渲染可以复用 3DGS 光栅化速度，理论上能做到交互帧率，但论文重点在离线重建质量。
- **硬件需求**：7 路 4K/60fps 视频的多视角 SDF 融合和粒子优化，显存和 IO 压力都不小，建议用体素分辨率分级（先粗后细）控制显存。
- **内存占用**：粒子数会随重播种不断增长，长序列必须做粒子数上限/合并策略，否则显存会线性爆炸。

### 数据采集建议

- 相机同步精度直接决定 SDF 融合质量，硬件触发同步优于软件时间戳对齐；
- mask 质量是整个方法的地基——水花几乎无纹理，自动分割模型在高光/半透明区域经常失败，人工精修几乎不可省。

### 常见坑

1. **速度场只在物质存在的区域有意义**：在空气区域强行拟合光流会产生伪速度，导致粒子被吸到不该去的地方 → 修复：只在 `phi_t` 附近的窄带（narrow band）内计算和使用速度场。

```python
narrow_band_mask = phi_t.abs() < band_width
v_field = v_field * narrow_band_mask  # 带外速度清零
```

2. **重播种阈值设太松会产生噪声粒子，设太紧会漏掉真实的新飞溅**：建议阈值与体素分辨率成比例，而不是固定常数。

```python
coverage_thresh = 1.5 * voxel_size  # 而不是写死一个绝对值
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---|---|
| 需要重建带拓扑变化的飞溅、水柱、倒水过程 | 静止/规则形状的清澈液面（游泳池、玻璃杯） |
| 有多台同步标定相机可用 | 只有单目/非同步视频 |
| 关心物理合理的运动而非单帧渲染质量 | 只需要单帧新视角合成，不关心动态一致性 |
| 需要时序插值或风格迁移而不想重新优化 | 需要实时在线重建（如机器人闭环感知） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| NeRF | 视角相关外观建模成熟 | 动态场景优化代价高，缺乏物理一致性 | 静态或缓慢变化场景的新视角合成 |
| 静态 3DGS | 渲染快，训练快 | 不处理时间维度 | 单帧/静态场景重建 |
| Dynamic 3DGS | 复用 3DGS 渲染速度，扩展到动态 | 依赖高斯身份跨帧跟踪，拓扑剧变下漂移严重 | 刚体/温和形变的动态场景（人体、缓慢流体） |
| SplashSplat | SDF+粒子提供物理骨架，支持插值/风格迁移 | 依赖多视角同步硬件与精细 mask，非实时 | 破碎/飞溅类液体的离线高保真重建 |

## 我的观点

这篇工作真正有意思的地方不是"又一个动态 3DGS 变种"，而是它显式地承认了**纯数据驱动的跟踪在拓扑剧变面前是不适定问题**，于是退回到"能用几何观测约束的地方就用 SDF，不能约束的地方才交给神经渲染自由发挥"。这种分层思路和流体仿真里 PIC/FLIP 混合欧拉-拉格朗日方法的精神是一致的，只是这里的"物理"更弱——没有显式求解 Navier-Stokes，速度场只是从观测反推出来的运动学近似，不保证质量守恒或不可压缩性。

离实际部署还有明显距离：7 台同步 4K 相机加人工精修 mask 的数据门槛，决定了它目前更适合影视特效或高价值数字资产制作，而不是机器人或 AR/VR 这类需要轻量采集的场景。真正值得关注的开放问题是：能不能把更强的物理先验（如不可压缩性约束）加进速度场估计而不损失对真实观测的拟合；以及能不能把这套"SDF+粒子"骨架泛化到更大尺度的水体（河流、海浪）而不是桌面尺度的水花实验。如果这条路能走通，"给动态高斯一个显式的物理骨架"很可能会成为动态场景重建的一个通用范式，不止用于液体。