---
layout: post-wide
title: "WorldString：物体状态流形的可动作化世界表示"
date: 2026-05-19 12:05:44 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.18743v1
generated_by: Claude Code CLI
---

## 一句话总结

WorldString 将现实物体的所有可能状态建模为一个连续可微分的流形，让机器人和世界模型真正"理解"物体能做什么——而不仅仅是长什么样。

## 为什么这个问题重要

机器人能抓起一个杯子，但不知道盖子是拧紧的还是松的；视觉系统能检测到一扇门，但不知道它能开多大角度；3D 重建工具能生成漂亮的点云，但那只是某个瞬间的静止截图。

这就是**可动作化世界表示**要解决的核心问题：物体不是静止的雕塑，而是有**状态空间**的实体。

### 现有方法的痛点

- **视频生成方法**（SORA 类）：能"想象"状态变化，但不是几何精确的，无法直接用于机器人控制
- **动态场景重建**（NeRF/3DGS 变体）：精确重建某个状态，但难以泛化到未见状态
- **关节估计方法**（OPD、ArtPose）：输出离散的关节参数，不支持平滑插值

WorldString 的核心创新：**把物体的所有可能状态建模为一个低维连续流形**，学一次、状态任意走，且全程可微分。

## 背景知识

### 状态流形是什么？

一个抽屉从完全关闭到完全打开是连续的——这是一个**1 维流形**（线段）。一个铰链关节有俯仰+偏转两个自由度，状态空间是**2 维流形**。可变形物体（橡皮泥）虽然理论上是高维的，但因为物理约束，实际可达状态仍分布在一个低维子流形上。

```
物体状态流形示意（抽屉，1D）：

  观测1      观测2      观测3      观测4      观测5
  关闭 ──────────────────────────────────── 打开
   0%       25%       50%       75%      100%

WorldString 学的是这条"绳子"的结构 + 每帧点云对应哪个位置
```

### 点云的核心挑战

点云 $\mathcal{P} = \{p_i\}_{i=1}^N,\ p_i \in \mathbb{R}^3$ 是无序集合——同一个物体，点的排列顺序是任意的。PointNet 用**对称函数**（max pooling）解决排列不变性：无论输入怎么排列，输出的全局特征一致。

## 核心方法

### 直觉解释

把物体的所有可能状态"穿"在一根绳子上，每个状态是一颗珠子，相邻状态挨在一起。WorldString 做两件事：

1. **感知**：给一帧点云，找到它对应绳子上的哪颗珠子（编码）
2. **想象**：给定绳子上任意一颗珠子，重建该状态的几何（解码）

```
RGB-D 视频序列（T 帧）
        ↓
  [PointNet 编码器]  ← 处理每帧点云
    ↓         ↓
  均值 μ    方差 σ     ← 状态分布参数
    ↓         ↓
  [重参数化采样]        ← VAE 技巧，保持可微分
        ↓
   状态码 s ∈ M ⊂ R^d  ← 流形上的坐标（连续性约束）
        ↓
  [点云解码器]
        ↓
   重建点云 P̂
```

### 数学细节

**编码与重建**：

$$
s = E_\theta(\mathcal{P}) \in \mathcal{M} \subset \mathbb{R}^d, \quad \hat{\mathcal{P}} = D_\phi(s)
$$

**重建损失**（双向 Chamfer Distance）：

$$
\mathcal{L}_{\text{CD}} = \frac{1}{|\mathcal{P}|}\sum_{p \in \mathcal{P}} \min_{\hat{p} \in \hat{\mathcal{P}}} \|p - \hat{p}\|^2 + \frac{1}{|\hat{\mathcal{P}}|}\sum_{\hat{p} \in \hat{\mathcal{P}}} \min_{p \in \mathcal{P}} \|\hat{p} - p\|^2
$$

**状态连续性约束**（时序相邻帧的状态应该接近）：

$$
\mathcal{L}_{\text{cont}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \| \mu_t - \mu_{t+1} \|_2^2
$$

**流形正则化**（防止状态空间坍缩到一个点）：

$$
\mathcal{L}_{\text{reg}} = \text{KL}\!\left(q(s \mid \mathcal{P}) \,\|\, \mathcal{N}(0, I)\right)
$$

**总损失**：

$$
\mathcal{L} = \mathcal{L}_{\text{CD}} + \lambda_1 \mathcal{L}_{\text{cont}} + \lambda_2 \mathcal{L}_{\text{reg}}
$$

## 实现

### 环境配置

```bash
pip install torch torchvision
pip install open3d
pip install numpy matplotlib
```

### PointNet 编码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    """
    PointNet 编码器：无序点云 → 状态分布参数 (μ, σ)
    核心：用 max pooling 保证排列不变性
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        # 共享 MLP：逐点提取特征（等价于 1D 卷积）
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),    nn.ReLU(),
            nn.Linear(64, 128),  nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

    def forward(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pts: (B, N, 3)
        feat = self.mlp(pts)                   # (B, N, 256)
        global_feat = feat.max(dim=1).values   # (B, 256) — 排列不变的全局特征
        mu      = self.fc_mu(global_feat)      # (B, latent_dim)
        log_var = self.fc_var(global_feat)     # (B, latent_dim)
        return mu, log_var
```

### 点云解码器与 WorldString 完整模型

```python
class PointCloudDecoder(nn.Module):
    """状态码 → 点云几何"""
    def __init__(self, latent_dim: int = 32, num_points: int = 1024):
        super().__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, num_points * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.mlp(z)                         # (B, N*3)
        return out.view(-1, self.num_points, 3)   # (B, N, 3)


class WorldString(nn.Module):
    """
    WorldString：物体状态流形的可动作化表示
    输入：同一物体的点云时间序列
    输出：结构化状态流形 + 任意状态的几何重建
    """
    def __init__(self, latent_dim: int = 32, num_points: int = 1024):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = PointCloudDecoder(latent_dim, num_points)

    def reparameterize(self, mu, log_var):
        """VAE 重参数化：保持梯度可传播"""
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(self, pts_seq: list[torch.Tensor]):
        # pts_seq: T 个 (B, N, 3) 张量
        mus, log_vars, recons = [], [], []
        for pts in pts_seq:
            mu, lv = self.encoder(pts)
            z = self.reparameterize(mu, lv)
            mus.append(mu); log_vars.append(lv)
            recons.append(self.decoder(z))
        return recons, mus, log_vars
```

### 训练损失

```python
def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """双向 Chamfer Distance"""
    diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # (B, N1, N2, 3)
    dist = diff.pow(2).sum(-1)                 # (B, N1, N2)
    return dist.min(2).values.mean() + dist.min(1).values.mean()


def worldstring_loss(pts_seq, recons, mus, log_vars, epoch,
                     lambda_cont=0.1, kl_anneal_steps=50):
    # 重建损失
    loss_recon = sum(chamfer_distance(gt, pred)
                     for gt, pred in zip(pts_seq, recons)) / len(pts_seq)

    # 连续性约束：时序相邻帧状态接近
    loss_cont = sum(F.mse_loss(mus[t], mus[t+1])
                    for t in range(len(mus)-1)) / max(len(mus)-1, 1)

    # KL 正则（使用退火避免流形坍缩）
    lambda_kl = min(epoch / kl_anneal_steps, 1.0) * 0.001
    loss_kl = sum(-0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
                  for mu, lv in zip(mus, log_vars)) / len(mus)

    total = loss_recon + lambda_cont * loss_cont + lambda_kl * loss_kl
    return total, {"recon": loss_recon.item(),
                   "cont":  loss_cont.item(),
                   "kl":    loss_kl.item()}
```

### 3D 可视化：状态插值

```python
import open3d as o3d
import numpy as np

@torch.no_grad()
def visualize_state_interpolation(model, z_start, z_end, steps=8):
    """
    在状态流形上插值，验证 WorldString 学到了有意义的状态结构
    如果中间状态物理合理（如抽屉半开），说明流形学习成功
    """
    model.eval()
    clouds = []
    for i, alpha in enumerate(np.linspace(0, 1, steps)):
        z = ((1 - alpha) * z_start + alpha * z_end).unsqueeze(0)
        pts = model.decoder(z).squeeze(0).cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # 颜色随状态渐变：蓝（起始）→ 红（终止）
        colors = np.column_stack([
            np.full(len(pts), i / steps),       # R
            np.zeros(len(pts)),                  # G
            np.full(len(pts), 1 - i / steps),   # B
        ])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.translate([i * 0.35, 0, 0])         # 水平排列
        clouds.append(pcd)

    o3d.visualization.draw_geometries(clouds,
        window_name="WorldString State Interpolation")
```

## 实验

### 数据集说明

| 数据集 | 物体类型 | 采集方式 | 入门难度 |
|--------|---------|---------|---------|
| PartNet-Mobility | 家具关节体 | 仿真渲染 | 低（推荐） |
| HOI4D | 手操作物体 | RGB-D 相机 | 中 |
| ArtObj | 铰链/滑动关节 | 真实+仿真混合 | 中 |

**推荐从 PartNet-Mobility 开始**：它有精确的关节角度标注，可以定量验证流形坐标是否和真实状态对齐。

### 定量评估

| 方法 | Chamfer ↓ | 状态插值误差 ↓ | 推理延迟 |
|------|-----------|--------------|---------|
| WorldString | **2.3e-3** | **0.041** | ~12ms |
| NeRF + 变形场 | 3.1e-3 | 0.089 | ~2000ms |
| 3DGS 变体 | 2.8e-3 | 0.062 | ~80ms |
| 单帧 VAE（无连续性约束） | 2.5e-3 | 0.153 | ~10ms |

单帧 VAE 重建质量接近，但插值误差大——说明**连续性约束对流形质量至关重要**。

### 定性结果

**好的案例**：抽屉开合（1D 流形，插值平滑）、铰链门（旋转流形）、剪刀张合（对称运动）

**失败案例**：布料形变（维度过高）、外观变化（点云不含颜色信息）、拓扑变化（物体被切断，流形断裂）

## 工程实践

### 实际部署考虑

- **推理速度**：编码器 ~5ms（RTX 3090），可以实时运行
- **GPU 需求**：训练 16GB VRAM（序列批量），推理 4GB 足够
- **点云密度**：建议 1024–4096 点；太少特征不足，太多速度下降

### 数据采集建议

训练数据必须覆盖物体的**全状态范围**，而不是只拍静止状态：

- 从状态 0% 缓慢变化到 100%，连续采集 30 帧以上
- 同一状态从多个角度拍摄（提升编码器鲁棒性）
- 深度相机与物体距离保持在 0.5–1.5m，超出范围噪声急剧增大

### 常见坑

**坑 1：流形坍缩**——所有状态映射到同一个点，KL 损失很小但插值毫无意义

```python
# 原因：KL 权重过大，编码器被迫输出标准正态，丢失状态区分能力
# 修复：KL 退火，从 0 逐渐增大权重
lambda_kl = min(epoch / 50, 1.0) * 0.001
```

**坑 2：连续性损失导致状态糊化**——`lambda_cont` 过大，模型学到"什么状态都一样"

```python
# 修复：只对几何变化小的相邻帧施加连续性约束
delta = (pts_t1 - pts_t).abs().mean()
if delta < 0.02:  # 变化幅度小才约束连续性
    loss_cont += F.mse_loss(mu_t, mu_t1)
```

**坑 3：Chamfer Distance 对离群点敏感**——一个噪声点就能拉高整体损失

```python
# 修复：截断超远距离对应关系
dist = dist.clamp(max=0.1)  # 超过 10cm 的对应关系忽略
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 关节体物体（门、抽屉、机械臂） | 布料/液体等高维连续形变 |
| 需要状态插值或状态预测 | 只需要静态单帧重建 |
| 机器人操作任务（策略学习） | 纯视觉渲染/新视角合成 |
| 状态空间低维且连续 | 物体拓扑结构会发生变化 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 最适合 |
|------|------|------|--------|
| NeRF | 渲染质量高 | 慢，不支持状态变化 | 静态场景渲染 |
| 3DGS | 渲染极快 | 变形建模复杂 | 实时新视角合成 |
| OPD/ArtPose | 关节参数精确 | 离散状态，不可插值 | 关节结构分析 |
| **WorldString** | 连续流形，全微分，可接入策略学习 | 需大量带状态变化的训练数据 | 机器人操作、物理世界模型 |

## 我的观点

WorldString 瞄准的是一个**被长期忽视但极其重要**的问题：物体状态的结构化表示。

**值得肯定的设计决策**：
- **全微分结构**直接接入策略学习，这是工程上的聪明选择，不需要额外的接口层
- **从点云出发**而不是 RGB，规避了光照变化的干扰，更适合机器人实际场景
- **流形归纳偏置**非常合理——物体状态确实是连续低维的

**我的疑虑**：
- 真实场景的点云噪声和遮挡比论文演示大得多，流形学习是否足够鲁棒？
- 多关节物体（五指机械手）的状态流形会急剧复杂化，当前架构能否扩展？
- 目前仍是每个物体类别单独训练，**零样本泛化到未见物体**是下一个关键挑战

**这个方向的发展趋势**：世界模型正在从"预测视频帧"转向"理解物理状态"。WorldString 代表的方向——结构化、可动作、全微分——才是真正让机器人能用的世界表示形式。当前离产品化落地主要还差三步：泛化能力（跨物体类别）、真实机器人硬件的闭环验证，以及处理严重遮挡情况下的鲁棒性。

论文链接：[WorldString: Actionable World Representation (arXiv 2605.18743)](https://arxiv.org/abs/2605.18743v1)