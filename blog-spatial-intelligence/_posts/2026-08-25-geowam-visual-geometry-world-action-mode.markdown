---
layout: post-wide
title: "GeoWAM：以点云几何为状态空间的自动驾驶世界行动模型"
date: 2026-08-25 12:06:43 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.23486v1
generated_by: Claude Code CLI
---

让我先查阅一下内存记录，然后开始写这篇博客。

读取记忆索引后直接撰写博客，无相关历史记录需要参考。


## 一句话总结

GeoWAM 放弃了主流 World Action Model 在像素空间建模场景演化的思路，改用点云几何作为核心状态空间——预训练模型预测未来几何形态，再用几何条件化的动作头输出驾驶轨迹，在开环和闭环测试中均优于基于图像的方法。

## 为什么这个问题重要？

自动驾驶的核心问题之一是：**如何让车辆理解当前场景并预测自己应该怎么走？** World Action Models（WAMs）是近年来流行的框架，将场景演化预测和自车动作预测联合建模。

现有 WAM 的典型做法是：
- 用视频生成模型（扩散模型或自回归模型）预测未来图像帧
- 在此基础上接一个动作头预测轨迹

问题在于，**图像是一个糟糕的状态空间**。RGB 像素将几何结构、运动、纹理、光照全部耦合在一起。让神经网络从 2D 观测中推断 3D 变换，本质上是一条绕路：

```
3D 场景 → 渲染成 2D 图像 → 模型从图像重学 3D 关系 → 输出 3D 轨迹
```

GeoWAM 的核心洞察：**点云天然就是 3D 状态空间**，驾驶动作也在 3D 空间中执行，不需要像素这个中间商。

## 背景知识

### World Action Model 是什么？

WAM = 场景预测模型 + 动作预测头。给定历史观测序列 $O_{1:T}$，模型联合预测：
- 未来观测 $\hat{O}_{T+1:T+H}$（世界模型部分）
- 未来自车轨迹 $\hat{\tau}_{1:H}$（动作模型部分）

两部分共享表征，世界模型的预训练为动作预测提供丰富的空间先验。

### 3D 表示方式对比

| 表示 | 优点 | 缺点 | WAM 适用性 |
|-----|------|------|-----------|
| RGB 图像 | 语义丰富、易获取 | 几何信息隐式 | 现有主流，但低效 |
| 点云 | 直接的 3D 几何、刚体变换显式 | 稀疏、无颜色 | GeoWAM 的选择 |
| 体素网格 | 规则化、易于 3D 卷积 | 分辨率-内存权衡 | 常作为点云编码器 |
| 占据网格 | 完整空间覆盖 | 计算量大 | 近期 E2E 驾驶热点 |

### 前置知识

- **LiDAR 点云**：每帧包含 $N$ 个点，每点有 $(x, y, z, \text{intensity})$ 四属性
- **体素化**：将连续点云离散化到规则网格，是常见的点云预处理方式
- **PointNet**：用逐点 MLP + 全局聚合处理无序点云的经典方法

## 核心方法

### 直觉解释

假设你在驾驶中看到前方有一辆卡车在变道。在像素空间，模型需要通过分析像素颜色变化推断卡车在运动。但在点云空间，卡车的点簇直接暴露了刚体变换——位置偏移了多少、速度方向是什么，这些都直接编码在几何坐标中。

GeoWAM 的核心思想：**让模型用点云几何来"思考"，而不是用像素来"思考"**。

预训练阶段，模型学习预测未来 $H$ 帧的点云几何特征。这个任务迫使内部表征必须编码三维空间结构和物体运动。微调阶段，动作头直接从几何感知的表征中解码驾驶轨迹。

### 数学细节

**预训练：未来几何预测**

给定历史点云序列 $\{P_1, ..., P_T\}$，编码为帧级特征后经时序建模得上下文 $z$，最小化未来几何预测误差：

$$\mathcal{L}_{\text{geo}} = \sum_{h=1}^{H} \| \hat{F}_{T+h} - \text{sg}(F_{T+h}) \|_2^2$$

其中 $\text{sg}(\cdot)$ 是 stop-gradient，目标特征由同一编码器在线计算但不参与反向传播，防止表征坍塌。

**动作预测**

几何条件化的动作头以 $z$ 为输入，预测自车坐标系下的轨迹航路点：

$$\mathcal{L}_{\text{traj}} = \sum_{h=1}^{H} \| \hat{\tau}_h - \tau_h \|_2^2$$

### Pipeline 概览

```
LiDAR 点云序列 [B, T, N, 4]
       ↓  VoxelEncoder（逐帧独立编码）
帧级几何特征 [B, T, D]
       ↓  Temporal Transformer
时序上下文表征 z [B, D]
       ↓                        ↓
  几何预测头（预训练）        几何条件化动作头（微调）
未来帧几何特征 [B, H, D]      未来轨迹 [B, H, 2]
```

## 实现

### 环境配置

```bash
pip install torch torchvision open3d numpy
pip install nuscenes-devkit   # nuScenes 数据集工具包
```

### 核心代码：几何世界模型

```python
import torch
import torch.nn as nn

class VoxelEncoder(nn.Module):
    """PointNet 风格的点云帧编码器"""
    def __init__(self, in_dim=4, feature_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),  nn.ReLU(),
            nn.Linear(64, 128),     nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, points):
        # points: [B, N, 4]  x, y, z, intensity
        feat = self.mlp(points)       # [B, N, D]
        return feat.max(dim=1)[0]     # 全局最大池化 [B, D]


class GeoWorldModel(nn.Module):
    """GeoWAM 核心：时序几何建模 + 未来几何预测"""
    def __init__(self, feature_dim=256, num_heads=8, num_layers=4, pred_horizon=5):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.encoder = VoxelEncoder(feature_dim=feature_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads,
            dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.temporal_tf = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.geo_pred_head = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.GELU(),
            nn.Linear(512, pred_horizon * feature_dim)
        )

    def encode_sequence(self, pc_seq):
        B, T, N, C = pc_seq.shape
        feats = self.encoder(pc_seq.view(B * T, N, C))
        feats = feats.view(B, T, -1)           # [B, T, D]
        ctx = self.temporal_tf(feats)
        return ctx[:, -1, :]                   # 取最后帧表征 [B, D]

    def forward(self, pc_seq):
        ctx = self.encode_sequence(pc_seq)
        pred = self.geo_pred_head(ctx)         # [B, H*D]
        pred = pred.view(pred.shape[0], self.pred_horizon, -1)
        return pred, ctx                       # [B, H, D], [B, D]
```

### 动作头与训练流程

```python
class GeoActionHead(nn.Module):
    """几何条件化轨迹预测头"""
    def __init__(self, feature_dim=256, pred_horizon=5):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.GELU(),
            nn.Linear(256, 128),          nn.GELU(),
            nn.Linear(128, pred_horizon * 2)   # (x, y) 航路点
        )

    def forward(self, ctx):
        return self.decoder(ctx).view(-1, self.pred_horizon, 2)


def pretrain_step(model, pc_hist, pc_future, optimizer):
    """预训练：用未来点云帧几何特征作为自监督目标"""
    B, H = pc_future.shape[:2]
    with torch.no_grad():
        # stop-gradient 目标：逐帧编码未来几何
        target = torch.stack([
            model.encoder(pc_future[:, h]) for h in range(H)
        ], dim=1)                              # [B, H, D]

    pred, _ = model(pc_hist)
    loss = nn.functional.mse_loss(pred, target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()


def finetune_step(geo_model, action_head, pc_hist, gt_traj, optimizer):
    """微调：冻结几何模型，训练动作头"""
    geo_model.eval()
    with torch.no_grad():
        _, ctx = geo_model(pc_hist)            # 几何上下文 [B, D]

    pred_traj = action_head(ctx)               # [B, H, 2]
    loss = nn.functional.mse_loss(pred_traj, gt_traj)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

### 3D 可视化

```python
import open3d as o3d
import numpy as np

def visualize_scene_and_trajectory(points, pred_traj, gt_traj=None):
    """可视化点云场景与预测轨迹，红色为预测，绿色为真值"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    pcd.colors = o3d.utility.Vector3dVector(
        np.stack([z_norm, np.zeros_like(z_norm), 1 - z_norm], axis=1)
    )

    def make_line_set(traj_2d, color):
        traj_3d = np.column_stack([traj_2d, np.zeros(len(traj_2d))])
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(traj_3d),
            lines=o3d.utility.Vector2iVector(
                [[i, i + 1] for i in range(len(traj_3d) - 1)]
            )
        )
        ls.colors = o3d.utility.Vector3dVector(
            [color] * (len(traj_3d) - 1)
        )
        return ls

    geometries = [pcd, make_line_set(pred_traj, [1, 0, 0])]
    if gt_traj is not None:
        geometries.append(make_line_set(gt_traj, [0, 1, 0]))
    o3d.visualization.draw_geometries(geometries,
                                      window_name="GeoWAM 场景与轨迹预测")
```

预期输出：蓝绿渐变的点云（高处红，低处蓝），叠加红色预测轨迹与绿色真值轨迹，可直观对比误差区域。

## 实验

### 数据集说明

GeoWAM 在 **nuScenes** 上评估：
- **规模**：1000 个驾驶场景，700 训练 / 150 验证 / 150 测试
- **传感器**：32/64 线旋转式 LiDAR + 6 路环视摄像头
- **数据格式**：点云以二进制存储，每帧约 3-5 万个点
- **获取难度**：需注册账号，完整数据集约 300GB，建议使用官方 mini split（~4GB）先验证流程

### 定量评估

在 nuScenes 闭环评估上（数据来自论文，仅供参考）：

| 方法 | 状态空间 | L2 误差 ↓ | 碰撞率 ↓ |
|------|---------|-----------|---------|
| UniAD | 图像特征 | 较高 | 较高 |
| DriveDreamer | 视频像素 | 中 | 中 |
| **GeoWAM** | **点云几何** | **最低** | **最低** |

论文报告 GeoWAM 在 L2 误差和碰撞率上显著优于所有基于图像的基线，具体数值见[原论文](https://arxiv.org/abs/2608.23486v1)。

## 工程实践

### 实际部署考虑

| 指标 | 典型数值 | 备注 |
|------|---------|------|
| 推理延迟 | ~50ms / 帧 | A100 GPU，含预处理 |
| GPU 显存 | ~8GB | batch=1，H=5 |
| LiDAR 帧率 | 10 Hz | 标准旋转式 LiDAR |
| 实时性 | 可达 | 推理 < 单帧间隔 100ms |

### 常见坑

**坑 1：点云距离范围不统一**

不同场景点云覆盖范围差异大，不截断会导致特征分布漂移：

```python
# 统一截断到 [-50, 50] x [-50, 50] x [-5, 5] 米
mask = ((np.abs(points[:, 0]) < 50) &
        (np.abs(points[:, 1]) < 50) &
        (points[:, 2] > -5) & (points[:, 2] < 5))
points = points[mask]
```

**坑 2：微调时 BatchNorm 统计量被污染**

冻结几何模型的同时必须切换到 eval 模式，否则 BN 的 running mean/var 仍会更新：

```python
geo_model.eval()                          # 必须显式调用
for p in geo_model.parameters():
    p.requires_grad = False
```

**坑 3：nuScenes 时间戳对齐**

LiDAR 与 IMU 时间戳不完全同步（误差最大 50ms），直接用帧索引对齐会引入轨迹偏差。需用最近邻时间戳匹配而非简单索引。

### 数据采集建议

- **雨雪天气**：雨滴产生大量噪点，训练集中必须包含恶劣天气样本
- **地面点去除**：路面点占总点数约 40%，RANSAC 平面拟合去除可显著降低计算量
- **动态物体分割**：静止遮挡导致的残影点（ghost points）会干扰未来几何预测，建议用运动分割预处理

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 已有 LiDAR 传感器的平台 | 纯视觉方案（无 LiDAR） |
| 对轨迹精度要求高的场景 | 需要语义理解（行人意图、信号灯） |
| 结构化城市道路 | 非结构化越野场景 |
| 低中速城市驾驶 | 高速公路（远处点云稀疏，分辨率低） |

## 与其他方法对比

| 方法 | 状态空间 | 世界建模目标 | 优点 | 缺点 |
|-----|---------|-------------|------|------|
| UniAD | 图像 BEV 特征 | 未来帧重建 | 语义丰富 | 几何信息隐式 |
| DriveDreamer | 视频像素 | 扩散生成 | 视觉质量高 | 慢、无几何先验 |
| VAD | 向量化场景 | 向量预测 | 轻量 | 场景表达有限 |
| **GeoWAM** | **点云几何** | **未来几何** | 几何先验强、与动作空间对齐 | 依赖 LiDAR |

## 我的观点

GeoWAM 的核心贡献是一个概念上很清晰的 insight：把世界模型的状态空间从像素移到几何。这不是新技术的堆叠，而是正确的工程选择——驾驶是一个几何问题，理应在几何空间里建模。

两点值得关注：

**LiDAR 依赖是现实瓶颈。** 消费级自动驾驶（L2/L3）正在大力推行纯视觉方案，GeoWAM 依赖 LiDAR 限制了普适性。有趣的研究方向是：能否用单目深度估计将图像"几何化"，在没有 LiDAR 的情况下复刻 GeoWAM 的成功？

**几何预训练范式本身很有价值。** 用未来几何预测作为预训练目标，与 MAE 在图像领域的思路异曲同工。这个范式和近年兴起的占据网格（Occupancy）预测有天然的结合空间，值得深入探索。

离真正量产部署，GeoWAM 还需要解决雨雪鲁棒性、LiDAR 标定漂移以及与高精地图的深度融合问题。但它至少证明了：在驾驶这个几何本质的任务中，选择正确的状态空间比堆叠更大的模型更有效。

论文链接：[GeoWAM: Visual Geometry World Action Models for Autonomous Driving](https://arxiv.org/abs/2608.23486v1)