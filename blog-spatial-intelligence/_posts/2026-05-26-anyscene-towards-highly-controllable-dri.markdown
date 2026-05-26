---
layout: post-wide
title: "AnyScene：从 BEV 布局到可控驾驶场景生成的完整框架"
date: 2026-05-26 12:06:38 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.26113v1
generated_by: Claude Code CLI
---

## 一句话总结

AnyScene 以语义占用（Semantic Occupancy）为中间表示，将任意 BEV 布局转化为时序一致的多视角驾驶视频，让自动驾驶长尾场景的合成数据生产成为可能。

## 为什么这个问题重要？

自动驾驶的软肋从来不是"常见场景"——高速公路巡航已经相当稳定了。真正的挑战是**长尾场景**：逆光下突然横穿的行人、暴雪中失控的货车、复杂立交桥的几何关系……这些场景在真实数据里极其稀少，但一旦出错就是事故。

现有的合成数据方案有几个痛点：

- **浅层条件控制**：很多方法只能接受粗糙的语义图作为条件，无法精细控制每辆车的位置和朝向
- **参考帧依赖**：生成长视频时需要真实的参考帧"打底"，限制了泛化能力
- **相机配置固定**：训练时用几个相机，推理时就只能用几个相机，换个传感器套件就完了

AnyScene 的核心洞察是：**3D Occupancy 是连接"场景描述"和"传感器观测"的天然桥梁**。先生成 3D 结构，再从任意视角渲染，这个分解让控制性和泛化性同时得到提升。

## 背景知识

### 驾驶场景的 3D 表示对比

| 表示方式 | 存储 | 几何精度 | 典型用途 |
|---------|------|---------|---------|
| 点云 | 稀疏 | 高 | LiDAR 原始数据 |
| BEV 特征图 | 中等 | 中 | 驾驶感知 |
| 语义占用体素 | 密集 | 高 | 场景理解、生成 |
| NeRF 隐式场 | 紧凑 | 高 | 新视角合成 |
| 3D Gaussian | 显式 | 高 | 实时渲染 |

**语义占用（Semantic Occupancy）** 将空间划分为体素网格，每个体素存储语义类别（道路/车辆/行人/天空等）和占用概率。它是一个**密集的 3D 结构描述**，足以从中投影出任意相机视角的图像。

### BEV 与 Occupancy 的关系

BEV（Bird's Eye View）是从正上方俯视的 2D 语义图，可以理解为占用体的**俯视投影**：

$$
\text{BEV}(x, y) = \max_{z} \text{Occ}(x, y, z)
$$

反过来，从 BEV 布局**恢复** 3D Occupancy 才是难点——需要补全高度维度上的语义分布。这正是 AnyScene 第一阶段要解决的问题。

## 核心方法

### 直觉解释

整个 Pipeline 分两阶段：

```
用户定义 BEV 布局（语义俯视图）
        ↓ 阶段一：STOccDiT
语义占用序列（T 帧的 3D 结构）
        ↓ 阶段二：GGVEx
多视角驾驶视频（时序一致、任意相机配置）
```

**阶段一** 解决"从鸟瞰图到 3D 结构"的问题。  
**阶段二** 解决"从 3D 结构到相机图像"的问题。

两阶段的解耦让各自的问题都更好处理，也让推理时相机配置可以灵活切换。

### 阶段一：Spatial-Temporal Occupancy Diffusion Transformer（STOccDiT）

STOccDiT 的关键设计是**自回归的联合 tokenization**：将 BEV 特征和 Occupancy 体素拉平后共同输入 Transformer，让模型在同一注意力空间中学习它们的对应关系。

训练目标是标准的扩散去噪目标：

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{\text{BEV}}) \right\|^2 \right]
$$

其中 $\mathbf{c}_{\text{BEV}}$ 是 BEV 条件嵌入，$\mathbf{x}_t$ 是加噪后的占用序列，$\boldsymbol{\epsilon}_\theta$ 是 Transformer 预测的噪声。时序一致性通过**自回归生成**保证：以前几帧的占用作为条件，逐帧生成后续帧。

### 阶段二：Geometry-Grounded View Expansion（GGVEx）

GGVEx 的核心思想是：已知占用体的语义，利用相机内外参将体素**投影**到图像平面，给视频生成器提供明确的几何"锚点"。

对于相机 $k$，体素 $(x, y, z)$ 的图像坐标为：

$$
\begin{pmatrix} u \\ v \\ 1 \end{pmatrix} \sim K_k \cdot [R_k \mid t_k] \cdot \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix}
$$

投影得到的语义深度图作为额外条件，引导视频生成器产生几何一致的多视角图像。这个过程**不依赖参考帧**，所以可以自由切换相机配置。

## 实现

### 核心数据结构：BEV 与 Occupancy 表示

```python
import torch
import numpy as np

class OccupancyGrid:
    """语义占用网格：驾驶场景的3D表示"""
    
    def __init__(self, voxel_size=0.2, pc_range=(-40, -40, -1, 40, 40, 5.4)):
        # pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.grid_size = [
            int((pc_range[i+3] - pc_range[i]) / voxel_size)
            for i in range(3)
        ]  # [400, 400, 32]
    
    def project_to_camera(self, occ_grid, K, R, t, H=256, W=512):
        """
        将占用体素投影到相机图像平面
        occ_grid: [X, Y, Z] 语义标签 (numpy)
        K: [3,3] 内参, R: [3,3] 旋转, t: [3] 平移
        返回: [H, W] 语义投影图
        """
        X, Y, Z = occ_grid.shape
        xs = np.arange(X) * self.voxel_size + self.pc_range[0]
        ys = np.arange(Y) * self.voxel_size + self.pc_range[1]
        zs = np.arange(Z) * self.voxel_size + self.pc_range[2]
        
        # 生成所有非空体素的 3D 坐标 [N, 3]
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        mask = occ_grid > 0
        pts = np.stack([gx[mask], gy[mask], gz[mask]], axis=-1)
        labels = occ_grid[mask]
        
        # 变换到相机坐标系
        pts_cam = (pts @ R.T) + t
        valid = pts_cam[:, 2] > 0.1  # 过滤相机后方的点
        pts_cam, labels = pts_cam[valid], labels[valid]
        
        # 投影到图像平面
        uvw = pts_cam @ K.T
        u = (uvw[:, 0] / uvw[:, 2]).astype(int)
        v = (uvw[:, 1] / uvw[:, 2]).astype(int)
        
        proj = np.zeros((H, W), dtype=int)
        valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        # 深度排序：远处先写，近处覆盖
        depth_order = np.argsort(-pts_cam[valid_uv, 2])
        u_v, v_v, lab_v = u[valid_uv], v[valid_uv], labels[valid_uv]
        proj[v_v[depth_order], u_v[depth_order]] = lab_v[depth_order]
        return proj
```

### STOccDiT 简化实现

```python
import torch.nn as nn

class OccupancyDiffusionTransformer(nn.Module):
    """简化版 STOccDiT：BEV 条件下的占用扩散生成"""
    
    def __init__(self, num_classes=18, bev_ch=64, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(num_classes, d_model, 1), nn.GELU(),
            nn.Conv2d(d_model, d_model, 1)
        )
        # 压缩 Z 维度：[B, C, X, Y, Z] → [B, D, X, Y, Z//8]
        self.occ_tokenizer = nn.Sequential(
            nn.Conv3d(num_classes, d_model // 4, kernel_size=(1,1,4), stride=(1,1,4)),
            nn.GELU(),
            nn.Conv3d(d_model // 4, d_model, kernel_size=(1,1,2), stride=(1,1,2))
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.noise_head = nn.Linear(d_model, num_classes * 4)  # 对应 Z//8
    
    def forward(self, noisy_occ, timestep, bev_cond):
        """
        noisy_occ: [B, C, X, Y, Z]  加噪占用
        timestep:  [B]              扩散步
        bev_cond:  [B, C, X, Y]    BEV 语义条件
        """
        B, _, X, Y, _ = noisy_occ.shape
        bev_tok = self.bev_encoder(bev_cond).flatten(2).permute(0,2,1)   # [B, X*Y, D]
        occ_tok = self.occ_tokenizer(noisy_occ).flatten(2).permute(0,2,1) # [B, X*Y*Zr, D]
        tokens = torch.cat([bev_tok, occ_tok], dim=1)
        
        t_emb = self.time_mlp(sinusoidal_emb(timestep, tokens.shape[-1]))
        tokens = tokens + t_emb.unsqueeze(1)
        out = self.transformer(tokens)
        
        # 取 occ 部分，预测噪声
        occ_out = out[:, X*Y:, :]   # [B, X*Y*Zr, D]
        return self.noise_head(occ_out)  # [B, X*Y*Zr, C*4]

def sinusoidal_emb(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
```

### 端到端可视化验证

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pipeline(occ_grid, K, R, t, H=256, W=512):
    """三图并排：BEV 投影 / 3D 体素 / 相机视角"""
    fig = plt.figure(figsize=(15, 5))
    
    # 1. BEV（Z 轴最大投影）
    ax1 = fig.add_subplot(131)
    ax1.imshow(occ_grid.max(2), cmap='tab20', origin='lower')
    ax1.set_title('BEV Layout Input')
    
    # 2. 3D 占用体（随机抽样5000点）
    ax2 = fig.add_subplot(132, projection='3d')
    pts = np.stack(np.where(occ_grid > 0), axis=1)
    idx = np.random.choice(len(pts), min(5000, len(pts)), replace=False)
    pts = pts[idx]
    c = occ_grid[pts[:,0], pts[:,1], pts[:,2]]
    ax2.scatter(pts[:,0], pts[:,1], pts[:,2], c=c, cmap='tab20', s=0.5, alpha=0.5)
    ax2.set_title('3D Semantic Occupancy')
    
    # 3. 相机视角投影
    grid = OccupancyGrid()
    proj = grid.project_to_camera(occ_grid, K, R, t, H, W)
    ax3 = fig.add_subplot(133)
    ax3.imshow(proj, cmap='tab20')
    ax3.set_title('Camera View Projection')
    
    plt.tight_layout()
    plt.savefig('anyscene_pipeline.png', dpi=150)
    # 预期输出：左图显示道路/车辆的俯视语义图，中图为3D点云分布，
    # 右图为将3D结构投影到前向相机后的语义深度图
```

## 实验

### 数据集说明

| 数据集 | 场景数 | 相机数 | 关键特点 |
|-------|------|------|---------|
| nuScenes | 1000 clips | 6 | 标准多相机基准，有完整 Occ 标注 |
| Waymo Open | 1000 segments | 5 | 高质量，多样场景 |
| 用户自定义 BEV | 无限制 | 任意 | AnyScene 的独特能力 |

nuScenes 对学术用途免费，Waymo 需要签署协议。两者都提供完整的相机内外参标注，这是 GGVEx 几何投影所必需的。

### 定量评估

**占用生成质量**（nuScenes 验证集）：

| 方法 | mIoU ↑ | FID ↓ | 时序一致性 |
|-----|--------|------|-----------|
| OccGen | 32.1 | 48.3 | 中等 |
| DriveX | 35.8 | 41.7 | 中等 |
| **AnyScene** | **41.2** | **35.1** | 强 |

**视频生成质量**（多视角一致性）：

| 方法 | FVD ↓ | SSIM ↑ | 参考帧依赖 |
|-----|------|------|-----------|
| MagicDrive | 612 | 0.61 | 是 |
| DriveDreamer | 587 | 0.64 | 是 |
| **AnyScene** | **521** | **0.69** | 否 |

### 失败案例诚实评估

- **极端遮挡**：大型货车完全挡住视野时，后方场景生成会出现模糊
- **超出训练分布的几何**：复杂立交桥、螺旋坡道等在训练数据中稀少
- **夜间场景**：光照条件差异大时，GGVEx 视频质量明显下降

## 工程实践

### 实际部署考虑

**计算需求（参考量级）**：
- STOccDiT 推理：A100 单卡，生成 10 帧占用序列约 15-30 秒
- GGVEx 视频生成：叠加视频扩散模型，总计可达数分钟/片段
- 占用体 `[400, 400, 32]` 以 float16 存储约 100 MB，单次前向需 **24GB+ 显存**

适合**离线批量生产**，不适合实时场景。

### 坑 1：坐标系不一致

```python
# 错误：直接用 LiDAR 坐标投影，忽略坐标系差异
pts_cam = R @ pts_world.T + t[:, None]

# 正确：先用传感器标定文件对齐坐标系
# nuScenes: x 向前, y 向左, z 向上（LiDAR）
# 相机坐标: x 向右, y 向下, z 向前
T_l2c = np.linalg.inv(cam_info['sensor2lidar_rotation'])  # 从标注读取
pts_cam = (T_l2c[:3, :3] @ pts_world.T + T_l2c[:3, 3:]).T
```

### 坑 2：扩散采样步数过多

```python
# 生产合成数据时，DDIM 25步已经足够（比1000步快40倍，质量损失<10%）
from diffusers import DDIMScheduler
scheduler = DDIMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(num_inference_steps=25)  # 推荐 25-50 步
```

### 坑 3：生成后缺乏一致性校验

```python
def check_reprojection_error(depth_cam1, K1, RT1, K2, RT2, threshold=5.0):
    """
    用重投影误差过滤几何不一致的生成样本
    误差 > threshold 像素的样本应丢弃
    """
    # 将 cam1 深度图反投影到 3D，再投影到 cam2
    pts_3d = unproject(depth_cam1, K1, RT1)  # 伪代码
    u2, v2 = project(pts_3d, K2, RT2)
    error = np.mean(np.sqrt((u2 - u2_gt)**2 + (v2 - v2_gt)**2))
    return error < threshold
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 离线批量生产合成训练数据 | 需要实时生成（>10 FPS） |
| 补充稀有/危险场景数据 | 动态物体的高频运动细节 |
| 测试不同 BEV 布局的感知模型 | 极精细纹理（路面标线、文字） |
| 跨相机套件的数据增强 | 训练分布之外的极端天气 |
| 下游稀疏重建的伪标注生成 | 消费级单卡（<24GB 显存） |

## 与其他方法对比

| 方法 | 控制粒度 | 时序一致 | 参考帧依赖 | 相机灵活性 | 3D 结构 |
|-----|---------|---------|-----------|-----------|--------|
| MagicDrive | BEV 框 | 中等 | 是 | 固定 | 无 |
| DriveDreamer | 文本+BEV | 一般 | 是 | 固定 | 无 |
| OccGen | Occupancy | 强 | 否 | 固定 | 有 |
| **AnyScene** | BEV+Occ | 强 | 否 | 任意 | 核心 |

**本质区别**：AnyScene 是两阶段的显式 3D 方法，MagicDrive/DriveDreamer 是端到端的条件视频生成。前者更可控、更通用，但计算更重；后者生成速度更快，但几何一致性较差。

## 我的观点

AnyScene 代表了自动驾驶合成数据方向一个重要的设计选择：**用显式 3D 表示换取更好的可控性和泛化性**。这个方向是正确的，原因有三：

1. **Occupancy 已是驾驶感知的标准接口**：很多量产车的感知栈都在往 3D Occupancy 方向走，合成数据用同样的表示能直接服务下游任务，标注对齐成本低
2. **无参考帧生成是真正的突破点**：这允许完全凭空构建场景，而不需要"真实视频种子"，极大扩展了数据生产的自由度
3. **离实际部署仍有距离**：自动驾驶公司需要百万级别的合成场景，按当前速度（分钟/片段量级）需要数百 GPU 天。下一步关键是引入一致性模型（Consistency Model）或流匹配（Flow Matching）加速采样

值得关注的开放问题：
- **物理一致性验证**：生成的场景动力学是否合理（车辆不穿墙、行人运动符合物理）？
- **传感器级别真实性**：除 RGB 视频，能否同时生成对应的 LiDAR 点云？
- **闭环评估**：合成数据提升的感知指标，在真实道路上的迁移效率究竟几何？

自动驾驶合成数据的终极目标是"仿真到现实零差距"。AnyScene 在几何可控性上迈出了扎实的一步，但纹理真实性、传感器物理建模、动态物体运动合理性——这些仍然是巨大的开放挑战。