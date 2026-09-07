---
layout: post-wide
title: "CrossDepth：用几何约束注意力解决环视深度估计的跨图像不一致问题"
date: 2026-09-07 12:03:09 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.05397v1
generated_by: Claude Code CLI
---

## 一句话总结

CrossDepth 是一种自监督的多相机环视深度估计方法，通过"相机感知射线编码"消除不同相机内参带来的单目线索差异，并用"几何约束的跨图像注意力"让每个像素能看到相邻相机中真正几何相关的区域，从而在自动驾驶环视场景下获得更准确、更一致的深度估计。

## 为什么这个问题重要？

自动驾驶车辆通常配备 6 个左右的环视相机，覆盖车身周围 360°。这些相机之间的视场重叠很小——相邻两个相机可能只在图像边缘有几十像素的重叠区域。这意味着：

- **绝大多数像素的深度只能靠单目线索推断**（纹理、遮挡、透视规律），立体三角测量帮不上忙。
- 单目深度估计天生是尺度模糊的，不同相机的焦距、畸变、安装角度不同，同一物体在不同相机里呈现的"单目线索"会不一样，模型对同一物理表面在不同图像中可能给出不一致的深度。
- 现有的自监督环视深度方法（如 FSM、SurroundDepth）大多是"每个相机独立跑单目深度网络 + 后处理拼接"，跨相机一致性缺乏显式约束。

CrossDepth 的核心创新是：**不再把跨相机一致性当作后处理问题，而是在网络内部用几何先验直接约束特征聚合的范围**。这对下游的占据栅格（occupancy）、BEV 感知、3D 检测都有直接影响——深度不一致会导致车辆周围点云"缝合"处出现台阶和鬼影。

## 背景知识

### 自监督深度估计的基本套路

没有激光雷达真值时，深度网络靠**光度一致性**训练：用预测深度 + 已知的相机相对位姿，把一张图像"warp"到另一个视角，如果深度预测正确，warp 后的图像应该和目标图像在光度上一致（像素级 L1 + SSIM）。这个信号既可以来自**同一相机的相邻帧**（依赖车辆自运动），也可以来自**同一时刻相邻相机的重叠区域**（依赖已知的相机间外参，不依赖运动估计，更稳定）。

### 极线几何回顾

给定两台标定好的相机（内参 K、外参 R/t 已知），相机 A 中一个像素对应的 3D 射线，投影到相机 B 的图像平面上会形成一条**极线**。由于深度未知，物理点具体落在这条极线的哪个位置不确定，但**搜索范围被极线严格限制**——这正是 CrossDepth 用来约束注意力范围的几何先验。

### 3D 表示方式的定位

CrossDepth 输出的是逐像素深度图（可反投影为点云），不是 NeRF 的隐式辐射场，也不是 3D Gaussian Splatting 的显式基元集合。它更接近传统的 MVS / 单目深度范式，胜在**自监督、无需真值深度、可直接部署在现有环视相机 rig 上**。

## 核心方法

### 直觉解释

想象你站在车顶中心，六个相机像风车叶片一样朝外看。对于左前相机图像里的一根电线杆：

1. **它在图像里的样子**取决于这个相机的焦距和畸变——如果不告诉网络"这是哪台相机拍的、每个像素对应哪根射线"，网络很难跨相机复用单目先验。→ CrossDepth 给每个像素加上**射线编码**（这个像素的 3D 射线方向和位置），让网络显式知道自己在"看哪个方向"。
2. **它是否在相邻相机（左侧相机）里也出现**，取决于极线几何——网络不应该在整张左侧图像里瞎找对应关系，而应该**只在极线附近**找。→ CrossDepth 用**几何约束注意力**，把注意力矩阵中几何不可行的位置直接屏蔽掉。

### 数学细节

**射线编码（Plücker 坐标）**：对相机中像素 $(u,v)$，方向向量与力矩向量为

$$
\mathbf{d} = \frac{R \, K^{-1} [u, v, 1]^\top}{\lVert R \, K^{-1} [u, v, 1]^\top \rVert}, \qquad \mathbf{m} = \mathbf{o} \times \mathbf{d}
$$

其中 $\mathbf{o}$ 是相机光心在车体坐标系下的位置。$(\mathbf{d}, \mathbf{m})$ 六维向量唯一确定了这条射线在车体坐标系下的位置和方向，与具体相机内参解耦。

**几何约束的注意力**：对查询相机中像素 $p$，沿其射线在 $[d_{min}, d_{max}]$ 范围采样一组候选深度，投影到相邻相机得到一条轨迹曲线，轨迹邻域内的像素构成可行集合 $\mathcal{M}(p)$。注意力计算为

$$
\text{Attn}(p, \cdot) = \text{softmax}\!\left(\frac{q_p k^\top}{\sqrt{d}} + B(p)\right), \quad
B(p)_j = \begin{cases} 0 & j \in \mathcal{M}(p) \\ -\infty & j \notin \mathcal{M}(p) \end{cases}
$$

**光度一致性损失**（简化形式）：

$$
\mathcal{L}_{photo} = \alpha \cdot \frac{1-\text{SSIM}(I_t, \hat{I}_t)}{2} + (1-\alpha) \lVert I_t - \hat{I}_t \rVert_1
$$

其中 $\hat{I}_t$ 是用预测深度和已知相对位姿从源图像 warp 得到的重建图像。

### Pipeline 概览

```
6 路环视图像
   │
   ├─ 逐相机 CNN/Transformer 主干 → 图像特征
   │
   ├─ 逐像素射线编码 (Plücker) → 与图像特征拼接/相加
   │
   ├─ 几何约束跨图像注意力（相邻相机对之间，多层堆叠）
   │
   ├─ 深度解码头 → 每路相机的深度图
   │
   └─ 光度一致性损失（跨相机重叠区 + 跨帧）→ 反向传播训练
```

## 实现

以下代码是对核心思想的最小可运行实现，用于理解算法骨架，不是论文的完整训练 pipeline（数据加载、多尺度金字塔、遮挡处理等已省略）。

### 环境配置

```bash
pip install torch torchvision matplotlib numpy
```

### 核心代码：射线编码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_ray_embedding(K, R, t, H, W):
    """
    计算相机的逐像素 Plücker 射线编码
    K: (3,3) 内参矩阵；R: (3,3) 相机到车体系旋转；t: (3,) 光心在车体系下位置
    返回: (H, W, 6) 射线嵌入 (方向 d, 力矩 m = o × d)
    """
    device = K.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)   # (H,W,3)
    K_inv = torch.inverse(K)
    d_cam = pix @ K_inv.T                                       # 相机系下方向
    d_rig = d_cam @ R.T                                          # 转到车体坐标系
    d_rig = d_rig / d_rig.norm(dim=-1, keepdim=True)
    o_rig = t.expand_as(d_rig)
    m_rig = torch.cross(o_rig, d_rig, dim=-1)                    # Plücker 力矩
    return torch.cat([d_rig, m_rig], dim=-1)                     # (H,W,6)
```

### 核心代码：几何约束区域生成

```python
def compute_epipolar_mask(K_a, R_a, t_a, K_b, R_b, t_b, u, v,
                           depth_min, depth_max, n_samples, H_b, W_b, band=2.0):
    """
    为相机 A 中像素 (u,v)，在相机 B 图像上生成几何可行的注意力区域：
    沿射线采样候选深度 → 投影到相机 B → 轨迹邻域内标记为可行 (mask=1)
    """
    device = K_a.device
    ray_cam = torch.inverse(K_a) @ torch.tensor([u, v, 1.0], device=device)
    depths = torch.linspace(depth_min, depth_max, n_samples, device=device)
    pts_cam_a = ray_cam.unsqueeze(0) * depths.unsqueeze(1)          # (N,3)
    pts_rig = pts_cam_a @ R_a.T + t_a                                # 转到车体坐标系
    pts_cam_b = (pts_rig - t_b) @ R_b                                # 转到相机 B 坐标系
    valid = pts_cam_b[:, 2] > 0.1                                    # B 前方才可见
    proj = pts_cam_b @ K_b.T
    proj_xy = proj[:, :2] / proj[:, 2:3].clamp(min=1e-6)

    mask = torch.zeros(H_b, W_b, device=device)
    ys, xs = torch.meshgrid(torch.arange(H_b, device=device, dtype=torch.float32),
                             torch.arange(W_b, device=device, dtype=torch.float32),
                             indexing="ij")
    for i in range(n_samples):
        if not valid[i]:
            continue
        px, py = proj_xy[i]
        dist = (xs - px) ** 2 + (ys - py) ** 2
        mask = torch.maximum(mask, (dist < band ** 2).float())
    return mask   # (H_b, W_b)，1 表示几何可行的对应区域
```

### 核心代码：几何约束跨图像注意力

```python
class GeometryConstrainedAttention(nn.Module):
    """跨图像注意力：仅在极线约束区域内做特征聚合"""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.scale = (dim // n_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, feat_q, feat_kv, mask):
        """
        feat_q:  (Lq, dim) 查询图像展平特征（已叠加射线编码）
        feat_kv: (Lk, dim) 相邻图像展平特征
        mask:    (Lq, Lk)  几何可行区域，1=可行 0=不可行
        """
        q, k, v = self.q_proj(feat_q), self.k_proj(feat_kv), self.v_proj(feat_kv)
        attn = (q @ k.T) * self.scale
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        return self.out_proj(attn @ v)
```

### 核心代码：自监督光度损失

```python
def photometric_loss(img_tgt, img_src, depth_tgt, K_tgt, K_src, T_tgt2src, alpha=0.85):
    """自监督光度一致性损失：用预测深度把源图像 warp 到目标视角后比较"""
    B, _, H, W = depth_tgt.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=depth_tgt.device, dtype=torch.float32),
                             torch.arange(W, device=depth_tgt.device, dtype=torch.float32),
                             indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=0).view(3, -1)
    cam_pts = torch.inverse(K_tgt) @ pix * depth_tgt.view(B, 1, -1)
    cam_pts_h = torch.cat([cam_pts, torch.ones(B, 1, H * W, device=cam_pts.device)], dim=1)
    src_pts = T_tgt2src @ cam_pts_h
    src_pix = K_src @ src_pts[:, :3]
    src_pix = src_pix[:, :2] / src_pix[:, 2:3].clamp(min=1e-3)
    grid = src_pix.view(B, 2, H, W).permute(0, 2, 3, 1)
    grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
    warped = F.grid_sample(img_src, grid, align_corners=True, padding_mode="zeros")

    l1 = (warped - img_tgt).abs().mean(1, keepdim=True)
    ssim_approx = F.avg_pool2d((warped - img_tgt) ** 2, 3, 1, 1)   # 简化版 SSIM 近似
    return (alpha * ssim_approx + (1 - alpha) * l1).mean()
```

### 3D 可视化

```python
import matplotlib.pyplot as plt

def visualize_epipolar_mask(img_a, img_b, mask, u, v):
    """可视化：相机 A 中查询像素及其在相机 B 上的几何可行区域"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_a); axes[0].scatter([u], [v], c="red", s=30)
    axes[0].set_title("Camera A - query pixel")
    axes[1].imshow(img_b); axes[1].imshow(mask.cpu().numpy(), cmap="jet", alpha=0.4)
    axes[1].set_title("Camera B - geometry-constrained region")
    plt.tight_layout(); plt.savefig("epipolar_mask.png", dpi=150)
```

预期效果：左图红点是查询像素，右图会看到一条随深度范围延伸的窄带状高亮区域，而不是整张图都高亮——这正是几何约束带来的搜索空间收窄。

官方代码与项目主页：https://abualhanud.github.io/CrossDepthPage/

## 实验

### 数据集说明

- **DDAD**：Toyota 发布的自动驾驶数据集，6 路环视相机，提供激光雷达点云作为深度评估真值（训练仍是自监督，仅评估用真值）。
- **nuScenes**：业界最常用的环视数据集之一，6 路相机 + 32 线激光雷达，场景多样（雨天、夜间、拥堵路口），常用于跨域泛化评估。

两者相机安装布局不同（焦距、朝向、离地高度均不同），因此论文用 DDAD 训练、nuScenes 测试（或反向）来验证"相机感知射线编码"是否真的能让模型泛化到未见过的相机配置，这是本方法的一个关键卖点。

### 定量评估（趋势参考）

论文报告的核心指标是标准单目深度指标：Abs Rel、Sq Rel、RMSE（越低越好）和 $\delta < 1.25$（越高越好），并额外报告了**跨图像深度一致性**指标（衡量相邻相机重叠区域深度预测的差异）。精确数值请以原文 Table 为准，这里只总结方向性结论：

| 对比维度 | 结论 |
|---------|------|
| 域内评估（DDAD→DDAD, nuScenes→nuScenes） | 优于此前的自监督环视 SOTA 方法 |
| 跨域评估（DDAD→nuScenes 或反向） | 相机感知射线编码带来的泛化收益更明显 |
| 相邻相机重叠区一致性 | 几何约束注意力显著降低重叠区深度差异 |

### 定性结果

论文展示的可视化对比中，基线方法在相邻相机拼接处（例如左前与正前相机交界的车道线、护栏）经常出现深度突变形成的"台阶"，而 CrossDepth 的深度图在该区域过渡更平滑。失败案例集中在**动态物体**（行人、其他车辆穿过重叠区）——因为极线约束依赖静态场景假设，动态物体会破坏光度一致性和几何对应关系。

## 工程实践（重要！）

### 实际部署考虑

- **实时性**：跨图像注意力的计算量和相机对数量、注意力层数成正比，直接在原始分辨率上做逐像素极线掩码计算是不现实的（上面的 `compute_epipolar_mask` 是逐像素 Python 循环，仅用于教学，实际需要在特征图分辨率（如 1/8 或 1/16 下采样）上向量化批处理）。
- **硬件需求**：6 路相机同时推理，加上跨图像注意力的显存开销，训练通常需要多卡（24GB+ 显存/卡），推理端如果要上车，需要对注意力模块做 TensorRT 量化和分辨率降采样。
- **内存占用**：极线掩码本质是稀疏的（每个查询像素只关联极线附近少量位置），工程实现应该用稀疏索引/gather 而不是稠密 mask 矩阵，否则内存随分辨率平方增长。

### 数据采集建议

- 相机 rig 标定精度直接决定极线约束是否有效——**外参标定误差会直接体现为极线掩码偏移**，导致跨图像注意力"看错地方"。上车前务必做精细的多相机联合标定（棋盘格/激光雷达辅助标定）。
- 需要相邻相机有一定重叠视场角（哪怕只有 5°-10°）才能提供光度监督信号；完全无重叠的 rig 布局无法用这套自监督方案训练。

### 常见坑

1. **动态物体污染光度损失** → 结合语义分割或运动一致性掩码，训练时屏蔽动态区域的光度损失。
2. **曝光/白平衡在不同相机间不一致，导致光度损失失真** → 训练前做逐相机的曝光归一化，或在损失中加入局部对比度归一化（如 census transform）。
3. **深度尺度模糊**（自监督深度方法通用问题）→ 若有稀疏激光雷达可用，加入尺度对齐的辅助监督；否则评估时需要用 median scaling 对齐真值尺度。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 已标定的多相机环视 rig（自动驾驶、机器人） | 单相机、无重叠视场的场景 |
| 相机间有小范围重叠视场 | 完全无重叠或宽 baseline 立体配置（更适合传统 MVS） |
| 静态或准静态场景为主 | 动态物体密集的场景（遮挡、光度不一致严重） |
| 需要跨相机深度一致性（BEV/occupancy 下游任务） | 只需单张图深度、不关心跨图一致性 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 传统单目自监督深度（Monodepth2 类） | 简单、成熟、每相机独立训练 | 跨相机不一致，无法利用重叠信息 | 单相机或对一致性无要求 |
| SurroundDepth / FSM 等环视方法 | 已引入跨相机监督信号 | 跨图像对应关系缺乏几何约束，容易学到错误对应 | 环视 rig，但对一致性要求不极致 |
| NeRF / 3DGS | 高质量新视角合成，隐式/显式场景重建 | 需要逐场景优化或大量视角，不适合实时在线感知 | 离线重建、数字孪生 |
| CrossDepth | 相机感知 + 几何约束注意力，跨图一致性更好，泛化到不同 rig 配置 | 依赖精确标定，动态物体场景表现下降，注意力计算开销较大 | 环视自动驾驶感知 pipeline |

## 我的观点

CrossDepth 代表的思路很实用：与其在网络后期"硬拼"多路深度图，不如把标定信息（相机内外参）当作先验直接注入到特征学习和注意力计算中。这类"用已知几何约束深度网络自由度"的做法，在多相机感知领域会越来越常见——因为它不需要额外标注，只需要把已经有的标定参数用起来。

离实际车规级部署还有距离：极线掩码计算和跨图注意力的实时化、量化，以及对动态物体的鲁棒性，都需要针对具体车型和算力平台做工程适配。一个值得关注的开放问题是——能否把这种几何约束注意力的思路，进一步和时序信息（多帧）、以及下游 occupancy/BEV 检测任务联合训练，做成端到端的感知系统，而不是"深度估计→占据栅格"两阶段流水线。