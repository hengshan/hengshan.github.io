---
layout: post-wide
title: "VLGA：用密集点图监督让自动驾驶大模型真正理解3D空间"
date: 2026-06-11 08:02:58 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.12396v1
generated_by: Claude Code CLI
---

## 一句话总结

VLGA 将几何理解作为第四模态引入视觉-语言-动作模型，通过逐像素点图回归损失迫使策略网络真正利用 3D 信息，而不是让几何特征"穿肠而过"。

---

## 为什么这个问题重要？

自动驾驶是空间智能最典型的应用场景：车辆必须在三维世界里做决策，而不是在二维图像里猜测。

**现有 VLA 方法的困境**有两种：

- **冻结 3D 骨干网络（Frozen 3D backbone）**：把预训练的深度估计或点云模型提取的特征注入 VLA，但没有任务目标约束策略网络必须用好这些几何特征。结果往往是网络学会忽略几何分支，靠视觉语言部分"猜"动作。
- **稀疏几何监督（Sparse geometric loss）**：用 3D 检测框、HD 地图车道线等监督信号约束空间感知，但这些信号覆盖密度低，无法提供密集的空间梯度。

**VLGA 的核心创新**：让模型在训练时重建它驾驶穿过的密集 3D 世界。每个像素都要预测对应的 3D 坐标，这个约束无法"走捷径"——网络必须真正理解几何才能通过监督。

---

## 背景知识

### 3D 表示方式：为什么选点图（Pointmap）？

| 表示 | 密度 | 可微性 | 计算开销 | 适用场景 |
|------|------|--------|----------|----------|
| 点云 | 稀疏 | 较差 | 低 | LiDAR 输入 |
| 体素 | 密集 | 好 | 高（$O(n^3)$）| 室内场景 |
| NeRF 隐式场 | 密集 | 好 | 极高 | 静态重建 |
| 深度图 | 密集 | 好 | 低 | 单帧感知 |
| **点图（Pointmap）** | **密集** | **好** | **低** | **端到端学习** |

**Pointmap** 是 DUSt3R 推广的表示方法：对于输入图像中每个像素 $(u, v)$，直接预测其对应的 3D 空间坐标 $(X, Y, Z)$，形成一个 $H \times W \times 3$ 的张量。

与深度图相比，点图不依赖相机内参就可以直接计算空间距离，梯度可以直接在 3D 坐标上回传，适合作为端到端训练的监督信号。

### 混合专家（MoE）简介

VLGA 的几何模块以"专家"形式嵌入：在 Transformer 的 FFN 层中，几何 token 会路由到专门的几何专家网络，语言/动作 token 走普通 FFN。这样几何计算不干扰语言推理，同时共享注意力层实现模态融合。

---

## 核心方法

### 直觉解释

想象一个学生驾驶教练的测验方式：

- **旧方法**：考学生"前方有没有行人？该不该刹车？"（只考动作）
- **VLGA**：还要求学生说出"前方行人距离 8.3 米，左侧护栏距离 1.2 米"（必须量化 3D 理解）

只有真正建立了 3D 空间模型，才能通过密集点图考试。这个考试无法靠背规律通过，必须"看懂"。

### 数学细节

**Pointmap 预测**：给定第 $t$ 帧图像 $I_t$，几何专家输出点图：

$$
\hat{P}_t \in \mathbb{R}^{H \times W \times 3}
$$

其中 $\hat{P}_t[u, v] = (\hat{X}, \hat{Y}, \hat{Z})$ 是像素 $(u,v)$ 对应的预测 3D 坐标（在自车坐标系下）。

**点图回归损失**：

$$
\mathcal{L}_{\text{geo}} = \frac{1}{\lvert \mathcal{V} \rvert} \sum_{(u,v) \in \mathcal{V}} \left\| \hat{P}_t[u,v] - P^*_t[u,v] \right\|_2
$$

其中 $\mathcal{V}$ 是有效 LiDAR 投影点的掩码集合（LiDAR 只覆盖部分像素）。

**总损失**（三项联合训练）：

$$
\mathcal{L} = \lambda_{\text{action}} \mathcal{L}_{\text{action}} + \lambda_{\text{geo}} \mathcal{L}_{\text{geo}} + \lambda_{\text{lang}} \mathcal{L}_{\text{lang}}
$$

**为什么 dense 比 sparse 信号强**：bounding box 损失对每帧只提供 $O(N_{\text{obj}})$ 个梯度（$N_{\text{obj}} \sim 20-50$），而点图损失提供 $O(H \times W) \sim 10^5$ 个梯度，空间约束密度提升约 3 个数量级。

### Pipeline 概览

```
摄像头图像 (6×H×W×3)
    │
    ▼
视觉编码器 (ViT/SwinT)
    │
    ▼
多模态 Transformer ←── 语言 token（场景描述/导航指令）
    │           └──── 几何 token → 几何专家 FFN
    │                                    │
    │                                    ▼ 点图头
    │                              Pointmap (H×W×3)
    │                                    │ LiDAR GT 监督
    │                              L_geo 损失
    │
    ▼
动作解码器
    │
    ▼
轨迹输出 (waypoints)
```

---

## 实现

### 环境配置

```bash
pip install torch torchvision transformers
pip install nuscenes-devkit open3d
# 官方代码（论文提交后发布）：暂未公开
# nuScenes 数据集：https://www.nuscenes.org/download
```

### 几何专家模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryExpert(nn.Module):
    """几何专家 FFN：替换标准 Transformer FFN 中的几何 token 处理"""
    
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_geo, D] 几何 token
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.norm(x + residual)


class PointmapHead(nn.Module):
    """从几何 token 解码出逐像素 3D 坐标"""
    
    def __init__(self, hidden_dim: int, H: int, W: int):
        super().__init__()
        self.H, self.W = H, W
        # 上采样路径：token → 特征图 → 点图
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 3)  # 输出 X, Y, Z
        )
    
    def forward(self, geo_tokens: torch.Tensor) -> torch.Tensor:
        """
        geo_tokens: [B, H*W, D] (假设 token 与像素一一对应)
        return: [B, H, W, 3] pointmap
        """
        B, N, D = geo_tokens.shape
        pointmap = self.decoder(geo_tokens)          # [B, N, 3]
        pointmap = pointmap.view(B, self.H, self.W, 3)
        return pointmap
```

### 点图回归损失

```python
class PointmapLoss(nn.Module):
    """
    逐像素 L2 损失，只计算 LiDAR 有效投影区域
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred_pointmap: torch.Tensor,   # [B, H, W, 3] 预测点图
        gt_pointmap: torch.Tensor,     # [B, H, W, 3] LiDAR 投影GT
        valid_mask: torch.Tensor       # [B, H, W] bool，LiDAR 有效区域
    ) -> torch.Tensor:
        
        # 只在有效 LiDAR 点处计算损失
        diff = pred_pointmap - gt_pointmap          # [B, H, W, 3]
        l2 = torch.norm(diff, dim=-1)               # [B, H, W]
        
        # 掩码平均，避免 LiDAR 稀疏带来的梯度不均
        loss = (l2 * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-6)
        return loss


def project_lidar_to_image(
    lidar_points: torch.Tensor,   # [N, 3] 点云 (X, Y, Z)
    cam_intrinsic: torch.Tensor,  # [3, 3]
    cam_extrinsic: torch.Tensor,  # [4, 4] lidar2cam
    H: int, W: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 LiDAR 点云投影为点图 GT"""
    
    # 变换到相机坐标系
    ones = torch.ones(lidar_points.shape[0], 1, device=lidar_points.device)
    pts_hom = torch.cat([lidar_points, ones], dim=1)    # [N, 4]
    pts_cam = (cam_extrinsic @ pts_hom.T).T[:, :3]      # [N, 3]
    
    # 只保留相机前方点
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]
    
    # 投影到像素坐标
    uv_hom = (cam_intrinsic @ pts_cam.T).T              # [N, 3]
    uv = (uv_hom[:, :2] / uv_hom[:, 2:3]).long()        # [N, 2]
    
    # 构建点图和掩码
    pointmap = torch.zeros(H, W, 3, device=lidar_points.device)
    mask = torch.zeros(H, W, dtype=torch.bool, device=lidar_points.device)
    
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv, pts_cam = uv[in_bounds], pts_cam[in_bounds]
    
    pointmap[uv[:, 1], uv[:, 0]] = pts_cam   # 注意 (row=y, col=x)
    mask[uv[:, 1], uv[:, 0]] = True
    
    return pointmap, mask
```

### 简化的 VLGA 前向传播

```python
class VLGASimplified(nn.Module):
    """VLGA 核心逻辑示意（省略完整 VLA 基础架构）"""
    
    def __init__(self, hidden_dim=768, H=32, W=64):
        super().__init__()
        # 几何专家（实际中嵌入 Transformer 每层）
        self.geo_expert = GeometryExpert(hidden_dim, hidden_dim * 4)
        self.pointmap_head = PointmapHead(hidden_dim, H, W)
        self.pointmap_loss = PointmapLoss()
        
        # 动作解码器：预测轨迹 waypoints
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 6 * 2)   # 6个 waypoint，每个 (x, y)
        )
    
    def forward(self, visual_tokens, lang_tokens, lidar_gt=None, valid_mask=None):
        # 分离几何 token（实际中通过路由机制）
        geo_tokens = self.geo_expert(visual_tokens)     # [B, H*W, D]
        
        # 点图预测
        pointmap_pred = self.pointmap_head(geo_tokens)  # [B, H, W, 3]
        
        # 几何感知的动作预测：几何特征 + 语言特征 → 动作
        fused = geo_tokens.mean(dim=1) + lang_tokens.mean(dim=1)  # 简化融合
        waypoints = self.action_head(fused).view(-1, 6, 2)
        
        losses = {}
        if lidar_gt is not None:
            losses['geo'] = self.pointmap_loss(pointmap_pred, lidar_gt, valid_mask)
        
        return waypoints, pointmap_pred, losses
```

### 3D 可视化

```python
import open3d as o3d
import numpy as np

def visualize_pointmap(pointmap: np.ndarray, valid_mask: np.ndarray, 
                        image: np.ndarray = None):
    """
    可视化点图：将 H×W×3 的点图渲染为带颜色的点云
    pointmap: [H, W, 3] float32
    valid_mask: [H, W] bool
    image: [H, W, 3] uint8（可选，作为点云颜色）
    """
    # 提取有效点
    pts_3d = pointmap[valid_mask]           # [N, 3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    
    if image is not None:
        colors = image[valid_mask] / 255.0  # [N, 3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 用距离着色（无图像时）
    else:
        depth = np.linalg.norm(pts_3d, axis=1)
        cmap = plt.cm.viridis((depth - depth.min()) / (depth.max() + 1e-6))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(cmap)
    
    o3d.visualization.draw_geometries([pcd])
    # 预期输出：彩色点云，显示车辆周围的道路、建筑、行人的3D结构
    # ... (数据加载和坐标系对齐代码省略)
```

---

## 实验

### 数据集说明

**nuScenes**（开环评估）：6 路摄像头 + 32 线 LiDAR，700 训练/150 验证 scene，城市场景，标注完善。获取难度低，官网注册后可免费下载。

**Bench2Drive**（闭环评估）：CARLA 仿真环境，支持直接运行策略并测量碰撞/完成率。适合闭环评估但与真实场景有 domain gap。

### 定量评估

**nuScenes 开环（无 ego status 输入，越低越好）**：

| 方法 | L2 (1s) | L2 (2s) | L2 (3s) | 碰撞率 (3s) |
|------|---------|---------|---------|------------|
| UniAD | 0.45 | 0.70 | 1.05 | 0.37% |
| VAD | 0.41 | 0.70 | 1.05 | 0.38% |
| SparseDrive | 0.43 | 0.67 | 1.01 | 0.31% |
| **VLGA（本文）** | **0.29** | **0.45** | **0.50** | **0.18%** |

**Bench2Drive 闭环（越高越好）**：

| 方法 | 驾驶得分 | 路线完成率 |
|------|---------|-----------|
| DriveVLM | 75.6 | - |
| DriveLLM-2 | 78.37 | - |
| **VLGA（本文）** | **79.08** | - |

L2 误差从 1.05m 降到 0.50m（3s 处），提升 52%，说明密集几何监督对长期轨迹预测帮助显著。

### 失败案例分析

- **强逆光/夜间**：点图预测质量下降，LiDAR GT 与图像特征对不上
- **高速公路直道**：L2 误差本身小，VLGA 改进不明显
- **遮挡行人**：LiDAR 和相机均看不到，dense supervision 也帮不上忙

---

## 工程实践

### 实际部署考虑

**硬件需求**：
- 训练：8×A100 80GB（基于论文实验规模估计），LiDAR GT 计算需要额外显存
- 推理：单 A100 或 RTX 4090，推理时几何专家仍然激活（增加约 20% 计算量）
- **关键**：推理时不需要 LiDAR，只用摄像头。LiDAR 仅在训练时作为监督信号

**延迟估计**：VLA 类模型通常 100-300ms/帧（含语言解码），不满足实时（<50ms）要求，实际部署需要异步规划架构。

### 数据采集建议

LiDAR-Camera 时间同步是最大坑。点图 GT 质量直接影响几何损失有效性：

```python
# 时间戳对齐：LiDAR 和相机帧率不同时的线性插值
def sync_lidar_to_camera(lidar_sweep, cam_timestamp, lidar_timestamps):
    # 找最近的两帧 LiDAR
    idx = np.searchsorted(lidar_timestamps, cam_timestamp)
    t0, t1 = lidar_timestamps[idx-1], lidar_timestamps[idx]
    alpha = (cam_timestamp - t0) / (t1 - t0 + 1e-9)
    # 对点云做线性插值（严格应用 ego motion 补偿）
    return lidar_sweep[idx-1] * (1 - alpha) + lidar_sweep[idx] * alpha
```

### 常见坑

**坑1：LiDAR 稀疏导致损失信号弱**

现象：只有 5-10% 的像素有 LiDAR GT，几何损失数值不稳定。

解决：使用深度补全（depth completion）预先稠密化 LiDAR，或调低 $\lambda_{\text{geo}}$ 权重并用 focal-style weighting 强调近距离点。

**坑2：点图预测单位尺度漂移**

```python
# 错误：直接回归绝对坐标，scale 随场景变化大
loss = F.mse_loss(pred, gt)

# 正确：归一化到场景范围内，或用 log-depth 形式
gt_normalized = gt / gt_norm.clamp(min=1.0)  # 按场景尺度归一化
loss = F.huber_loss(pred_normalized, gt_normalized, delta=0.1)
```

**坑3：几何专家权重不被利用**

症状：$\mathcal{L}_{\text{geo}}$ 收敛但 $\mathcal{L}_{\text{action}}$ 没有改善。原因：几何梯度没有有效回传到动作路径。

解决：检查几何 token 是否真正参与了跨注意力（cross-attention），确保动作解码器能 attend 到几何 token。

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有配套 LiDAR 的训练数据 | 纯摄像头数据集（无 LiDAR 监督） |
| 城市复杂交通场景 | 结构化停车场/简单直道 |
| 对 3s 长期轨迹精度有要求 | 只需要 1s 短期控制 |
| 算力充足（A100 级别训练） | 边缘端嵌入式部署 |
| 闭环仿真验证 | 实时嵌入式控制器（<50ms） |

---

## 与其他方法对比

| 方法 | 几何建模 | 监督信号 | 语言能力 | 适用场景 |
|------|---------|---------|---------|---------|
| UniAD | BEV 特征 | 稀疏（检测/分割） | 无 | 端到端规划 |
| DriveVLM | 无显式 3D | 无几何损失 | 强 | 场景理解 + 规划 |
| SparseDrive | 稀疏实例 | BBox 损失 | 弱 | 快速推理 |
| **VLGA** | 逐像素点图 | **密集 LiDAR** | 强（VLA基础） | **精准长程规划** |

VLGA 的定位更接近"精度优先"而非"效率优先"，与 SparseDrive 的取舍方向相反。

---

## 我的观点

**这个方向的核心价值**在于提出了一个优雅的问题：如果你不能重建驾驶场景的 3D 结构，你凭什么说你理解了它？密集点图作为自监督信号是合理的，比 bounding box 回归有更强的理论依据。

**离实际应用还有的距离**：VLA 类模型的推理延迟目前是最大瓶颈。VLGA 的几何专家在推理时仍然存在，额外计算不可避免。在 L4 自动驾驶的感知-规划-控制全栈里，100ms 的规划周期勉强可用，但需要配合低层控制器做补偿。

**值得关注的开放问题**：
1. 能否用 4D Radar 替换 LiDAR 降低硬件成本？（4D Radar 点云更稀疏，但便宜）
2. 点图监督对动态目标（行人、自行车）是否有独特帮助？遮挡情况下的预测质量？
3. 几何专家学到的 3D 表征能否迁移到其他任务（3D 目标检测）？

VLGA 更像是一个"正确方向上的有力论证"，而非即插即用的产品方案。对于研究者来说，密集几何监督这个思路值得在更多 VLA 架构上验证。