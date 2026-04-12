---
layout: post-wide
title: "Scal3R：大规模场景3D重建的测试时训练方案"
date: 2026-04-12 12:03:19 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.08542v1
generated_by: Claude Code CLI
---

## 一句话总结

Scal3R 通过**测试时训练（Test-Time Training, TTT）**将神经全局上下文压缩进轻量子网络，让前馈式3D重建模型在长视频序列中不再"失忆"，从根本上解决了大场景重建的漂移和不一致问题。

---

## 为什么这个问题重要？

自动驾驶汽车沿着街道行驶1公里，机器人探索一栋楼，AR眼镜扫描一个展览馆——这些场景的共同问题是：**序列长、场景大、全局一致性难维持**。

近两年的前馈式3D重建方法（DUSt3R、MASt3R 等）通过直接从 RGB 图像回归三维几何，不需要 SfM 预处理就能快速重建。但这类方法有一个根本缺陷：

```
帧1 帧2 → [滑动窗口处理] → 局部准确 ✓
帧1 ...  帧500 → [同样处理] → 累积漂移 ✗
```

**问题的本质**：模型的感受野是有限的（通常几帧到几十帧），处理第500帧时已经"忘掉"了前面建立的全局结构。人类不会这样——我们会用对整栋楼的全局理解来纠正当前位置的判断。

Scal3R 的核心贡献就是给模型装上一个"全局记忆"。

---

## 背景知识

### 3D 表示回顾

| 表示方式 | 典型方法 | 优点 | 缺点 |
|---------|---------|------|------|
| 点云 | SfM, LiDAR | 直观、轻量 | 无纹理、稀疏 |
| 密集点图 | DUSt3R, MASt3R | 稠密重建 | 内存随帧数线性增长 |
| 隐式 NeRF | NeRF, iNeRF | 连续、紧凑 | 训练慢 |
| 3D Gaussian | 3DGS | 实时渲染 | 大场景存储大 |

Scal3R 的基础是**点图（Point Map）**表示：网络直接输出每像素对应的3D点坐标 $X \in \mathbb{R}^{H \times W \times 3}$，比深度图更直接（不需要相机内参）。

### 测试时训练是什么？

测试时训练（TTT）是一种在推理阶段对模型部分参数进行快速适应的技术。核心思想：

$$\phi^* = \arg\min_{\phi} \mathcal{L}_{self}(x_{test}; \theta, \phi)$$

其中 $\theta$ 是冻结的主干参数，$\phi$ 是可适应的轻量子网络参数，$\mathcal{L}_{self}$ 是无需标注的自监督损失。

---

## 核心方法

### 直觉解释

想象你在用手机拍一段走廊视频，设备需要实时重建。传统滑动窗口做法：

```
[帧1-10]  → 重建局部A
[帧11-20] → 重建局部B（不知道A在哪）
[帧21-30] → 重建局部C（漂移累积）
```

Scal3R 的做法：

```
[帧1-10]  → 重建局部A + 压缩全局上下文 c₁
[帧11-20] → 重建局部B（有 c₁ 指导）+ 更新 c₂
[帧21-30] → 重建局部C（有 c₂ 指导，零漂移）
```

全局上下文 $\mathbf{c}$ 存储在**轻量子网络 $\phi$ 的参数**里，而不是显式存储所有历史帧。这是关键设计：参数 = 压缩的记忆。

### 数学细节

**点图预测**：主干网络 $g_\theta$ 接收当前帧图像和全局上下文，输出点图：

$$\hat{X}_t = g_\theta(I_t, \phi(z_t))$$

其中 $z_t$ 是上下文查询向量，$\phi(\cdot)$ 是轻量子网络对全局信息的响应。

**测试时自监督损失**包含两部分：

**几何一致性损失**：相邻帧预测的点图在公共区域应一致
$$\mathcal{L}_{geo} = \sum_{(i,j) \in \mathcal{E}} \| \Pi(X_i \to j) - X_j \|_1$$

其中 $\Pi(\cdot)$ 表示从帧 $i$ 投影到帧 $j$ 坐标系的操作。

**光度一致性损失**：从预测点图和相机位姿重投影后的图像应与原图一致
$$\mathcal{L}_{photo} = \sum_{t} \| I_t - \hat{I}_t(\{X_s\}, \{P_s\}) \|_1$$

总损失为：
$$\mathcal{L}_{TTT} = \mathcal{L}_{geo} + \lambda \mathcal{L}_{photo}$$

子网络 $\phi$ 通过在测试序列上最小化 $\mathcal{L}_{TTT}$ 快速适应，学到了该场景的全局结构。

### Pipeline 概览

```
长视频序列 (N帧)
    ↓
滑动窗口分组 (每组K帧, 步长S)
    ↓
[窗口内] 前馈重建 g_θ
    ↓
[全局] 上下文子网络 φ 压缩场景信息
    ↓
[测试时] 自监督优化 φ → φ*
    ↓
带全局上下文的点图预测
    ↓
全局点云拼接 + 位姿估计
```

---

## 实现

### 神经全局上下文的核心实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralContextNetwork(nn.Module):
    """
    轻量全局上下文子网络
    接收局部特征，输出全局上下文向量
    测试时这个网络的参数会被快速适应
    """
    def __init__(self, feat_dim=256, context_dim=128, n_slots=64):
        super().__init__()
        # 上下文槽：可学习的"记忆单元"
        self.context_slots = nn.Parameter(torch.randn(n_slots, context_dim))
        
        # 注意力机制：局部特征查询全局槽
        self.query_proj = nn.Linear(feat_dim, context_dim)
        self.key_proj = nn.Linear(context_dim, context_dim)
        self.value_proj = nn.Linear(context_dim, context_dim)
        
        # 槽更新网络（GRU风格）
        self.slot_update = nn.GRUCell(feat_dim, context_dim)
    
    def forward(self, local_features):
        """
        local_features: [B, N, feat_dim] 当前窗口的局部特征
        返回: [B, n_slots, context_dim] 全局上下文
        """
        B, N, _ = local_features.shape
        slots = self.context_slots.unsqueeze(0).expand(B, -1, -1)
        
        # 交叉注意力：用局部特征更新全局槽
        q = self.query_proj(local_features)           # [B, N, C]
        k = self.key_proj(slots)                      # [B, n_slots, C]
        v = self.value_proj(slots)                    # [B, n_slots, C]
        
        attn = torch.softmax(q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5), dim=-1)
        context = attn.transpose(-2, -1) @ local_features  # [B, n_slots, feat_dim]
        
        # GRU更新槽
        updated_slots = self.slot_update(
            context.reshape(B * slots.shape[1], -1),
            slots.reshape(B * slots.shape[1], -1)
        ).reshape(B, slots.shape[1], -1)
        
        return updated_slots


class PointMapHead(nn.Module):
    """从图像特征+全局上下文预测点图"""
    def __init__(self, feat_dim=256, context_dim=128):
        super().__init__()
        self.context_attn = nn.MultiheadAttention(feat_dim, 8, batch_first=True)
        self.context_proj = nn.Linear(context_dim, feat_dim)
        self.point_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(),
            nn.Linear(128, 3)  # 输出 (X, Y, Z)
        )
    
    def forward(self, image_features, global_context):
        """
        image_features: [B, H*W, feat_dim]
        global_context: [B, n_slots, context_dim]
        """
        ctx = self.context_proj(global_context)   # 维度对齐
        # 图像特征用全局上下文做交叉注意力增强
        enhanced, _ = self.context_attn(image_features, ctx, ctx)
        point_map = self.point_head(enhanced + image_features)  # 残差连接
        return point_map  # [B, H*W, 3]
```

### 测试时训练循环

```python
def test_time_adapt(model, context_net, video_frames, n_adapt_steps=5, lr=1e-4):
    """
    model:        冻结的主干网络 (g_θ)
    context_net:  待适应的上下文子网络 (φ)
    video_frames: 测试视频帧列表
    """
    model.eval()
    # 只优化上下文子网络的参数
    optimizer = torch.optim.Adam(context_net.parameters(), lr=lr)
    
    reconstructed_points = []
    
    for win_start in range(0, len(video_frames), 8):
        window = video_frames[win_start : win_start + 16]
        if len(window) < 2:
            break
        
        # 测试时适应：用自监督损失快速调整 context_net
        for step in range(n_adapt_steps):
            optimizer.zero_grad()
            
            # 前向：提取特征 + 全局上下文 + 预测点图
            with torch.no_grad():
                feats = model.encode(window)          # 冻结编码器
            
            ctx = context_net(feats)                  # 可优化的上下文
            point_maps = model.decode(feats, ctx)     # [T, H, W, 3]
            
            # 自监督损失1：相邻帧几何一致性
            loss_geo = compute_geometric_consistency(point_maps, window)
            
            # 自监督损失2：光度重投影一致性
            poses = estimate_poses_from_points(point_maps)
            loss_photo = compute_photometric_loss(window, point_maps, poses)
            
            loss = loss_geo + 0.1 * loss_photo
            loss.backward()
            optimizer.step()
        
        # 适应后的最终预测
        with torch.no_grad():
            ctx = context_net(model.encode(window))
            final_points = model.decode(model.encode(window), ctx)
            reconstructed_points.append(final_points)
    
    return torch.cat(reconstructed_points, dim=0)


def compute_geometric_consistency(point_maps, frames):
    """相邻帧点图的几何一致性损失（简化版）"""
    T = point_maps.shape[0]
    loss = 0.0
    for t in range(T - 1):
        # 从帧t的点图预测帧t+1中对应点的位置
        pts_t = point_maps[t]         # [H, W, 3]
        pts_t1 = point_maps[t + 1]    # [H, W, 3]
        # 用光流做对应，检查3D点的一致性
        # ... (对应关系计算省略)
        loss += F.l1_loss(pts_t[..., 2], pts_t1[..., 2])  # 深度一致性
    return loss / (T - 1)
```

### 3D 可视化

```python
import open3d as o3d
import numpy as np

def visualize_reconstruction(point_maps, colors=None, voxel_size=0.05):
    """
    point_maps: numpy array [N, H, W, 3]
    colors:     numpy array [N, H, W, 3] RGB (0-1)
    """
    # 合并所有帧的点云
    all_pts = point_maps.reshape(-1, 3)
    
    # 过滤无效点（距离太远或坐标异常）
    valid = (np.abs(all_pts).max(axis=-1) < 100) & (all_pts[:, 2] > 0)
    all_pts = all_pts[valid]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    
    if colors is not None:
        all_colors = colors.reshape(-1, 3)[valid]
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # 体素下采样，减少点数
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 法向量估计（用于渲染）
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Scal3R 重建结果",
        zoom=0.5, front=[0, 0, -1], up=[0, -1, 0]
    )
    return pcd
```

---

## 实验

### 数据集说明

| 数据集 | 场景类型 | 序列长度 | 获取难度 |
|-------|---------|---------|---------|
| KITTI Odometry | 城市驾驶（室外） | 数千帧 | 公开下载 |
| Oxford Spires | 历史建筑（室外） | 超长序列 | 公开，需注册 |

两个数据集都是户外大场景，典型特点：光照变化大、重复纹理多（路面、草地）、场景尺度跨度大。这些正是让大多数方法头疼的条件。

### 定量评估

论文报告的 KITTI Odometry 结果（Translation Error, Rotation Error 越低越好）：

| 方法 | 类型 | Trans. Err (%) | Rot. Err (°/100m) | 速度 |
|-----|------|---------------|-------------------|------|
| DUSt3R | 前馈（无全局上下文）| 较高 | 较高 | 快 |
| MASt3R | 前馈（特征匹配）| 中等 | 中等 | 中 |
| **Scal3R** | **前馈+TTT** | **最低** | **最低** | 中 |
| COLMAP | 传统 SfM | 低 | 低 | 非常慢 |

关键结论：Scal3R 在前馈方法里精度最高，对比传统离线 SfM 有竞争力，且保留了前馈方法的速度优势。

---

## 工程实践

### 实际部署考虑

**硬件需求**：
- 训练（如果需要微调主干）：2~4张 A100 80G
- 测试时推理：单张 RTX 4090 可运行，但测试时训练会增加约 30-50% 时间开销
- 内存：上下文子网络很轻量（~几十 MB），主要开销在主干特征提取

**吞吐量**：以 KITTI 为例，每帧大约需要 50-200ms（取决于 TTT 迭代次数），不是严格实时，但比离线 SfM 快一个数量级。

### 常见坑

**1. 上下文槽坍塌（Slot Collapse）**

所有槽学到相同信息，全局上下文退化。

```python
# 诊断：检查槽的多样性
def check_slot_diversity(context):
    # context: [B, n_slots, C]
    # 计算槽间余弦相似度
    norm = F.normalize(context, dim=-1)
    sim = (norm @ norm.transpose(-2, -1)).mean()
    if sim > 0.9:
        print("警告：上下文槽坍塌！相似度:", sim.item())
    return sim

# 修复：加入槽多样性损失
def diversity_loss(context):
    norm = F.normalize(context, dim=-1)
    sim = norm @ norm.transpose(-2, -1)
    # 惩罚过高相似度（对角线除外）
    eye = torch.eye(sim.shape[-1], device=sim.device)
    return (sim * (1 - eye)).mean()
```

**2. 测试时过拟合（TTT Overfitting）**

TTT 迭代步数过多，上下文子网络过拟合到当前窗口的噪声。

```python
import torch
import torch.nn as nn

class NeuralContextNetwork(nn.Module):
    """轻量全局上下文子网络：局部特征→全局上下文向量"""
    def __init__(self, feat_dim=256, context_dim=128, n_slots=64):
        super().__init__()
        self.context_slots = nn.Parameter(torch.randn(n_slots, context_dim))  # 可学习记忆单元
        self.query_proj = nn.Linear(feat_dim, context_dim)
        self.key_proj = nn.Linear(context_dim, context_dim)
        self.slot_update = nn.GRUCell(feat_dim, context_dim)

    def forward(self, local_features):  # [B, N, feat_dim]
        B, N, _ = local_features.shape
        slots = self.context_slots.unsqueeze(0).expand(B, -1, -1)

        # 交叉注意力：局部特征查询全局槽
        q = self.query_proj(local_features)
        k = self.key_proj(slots)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5), dim=-1)
        context = attn.transpose(-2, -1) @ local_features  # [B, n_slots, feat_dim]

        # GRU更新槽状态
        S = slots.shape[1]
        updated_slots = self.slot_update(
            context.reshape(B * S, -1), slots.reshape(B * S, -1)
        ).reshape(B, S, -1)
        return updated_slots  # [B, n_slots, context_dim]


class PointMapHead(nn.Module):
    """图像特征 + 全局上下文 → 点图"""
    def __init__(self, feat_dim=256, context_dim=128):
        super().__init__()
        self.context_attn = nn.MultiheadAttention(feat_dim, 8, batch_first=True)
        self.context_proj = nn.Linear(context_dim, feat_dim)
        self.point_head = nn.Sequential(nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, image_features, global_context):
        ctx = self.context_proj(global_context)
        enhanced, _ = self.context_attn(image_features, ctx, ctx)
        return self.point_head(enhanced + image_features)  # 残差连接 → [B, H*W, 3]
```

**3. 跨窗口拼接漂移**

```python
def test_time_adapt(model, context_net, video_frames, n_adapt_steps=5, lr=1e-4):
    model.eval()
    optimizer = torch.optim.Adam(context_net.parameters(), lr=lr)
    reconstructed_points = []

    for win_start in range(0, len(video_frames), 8):
        window = video_frames[win_start : win_start + 16]

        # 测试时适应：用自监督损失快速调整 context_net
        for step in range(n_adapt_steps):
            optimizer.zero_grad()
            with torch.no_grad():
                feats = model.encode(window)   # 冻结编码器
            ctx = context_net(feats)           # 可优化的上下文
            point_maps = model.decode(feats, ctx)

            loss = compute_geometric_consistency(point_maps, window)
            # ... (光度重投影损失省略)
            loss.backward()
            optimizer.step()

        # 适应后的最终预测
        with torch.no_grad():
            feats = model.encode(window)
            reconstructed_points.append(model.decode(feats, context_net(feats)))

    return torch.cat(reconstructed_points, dim=0)


def compute_geometric_consistency(point_maps, frames):
    T = point_maps.shape[0]
    loss = 0.0
    for t in range(T - 1):
        # ... (光流对应关系计算省略)
        loss += F.l1_loss(point_maps[t][..., 2], point_maps[t+1][..., 2])  # 深度一致性
    return loss / (T - 1)
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 超长视频（>200帧）重建 | 需要严格实时（<30ms/帧） |
| 室外大场景（城市、建筑） | 简单小场景（前馈已够用）|
| 对全局一致性要求高 | 动态场景（大量运动物体）|
| 有测试时计算预算 | 嵌入式/边缘设备部署 |
| 无 GT 标注的新场景迁移 | 场景与训练分布高度吻合 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| DUSt3R | 速度快，无需标定 | 长序列漂移严重 | 短序列、原型验证 |
| MASt3R | 特征匹配更鲁棒 | 仍受限于窗口大小 | 中等长度序列 |
| COLMAP | 精度高、经过工程验证 | 极慢、需特征点 | 离线精度优先 |
| NeRF/3DGS | 渲染质量高 | 需要位姿，慢 | 有位姿的小场景 |
| **Scal3R** | 大场景精度高，端到端 | TTT 有额外时间开销 | 大场景在线重建 |

---

## 我的观点

**这个方向真正的价值**在于它找到了一个工程上可行的折中点：不像 NeRF 那样需要全量优化，也不像纯前馈那样放弃全局信息。把全局状态压缩进网络参数是一个优雅的设计——参数本身就是记忆。

**测试时训练的趋势**值得认真关注。随着 NLP 领域 TTT 论文（如 TTT-LM）的成功，这套思路正在向视觉领域渗透。对 3D 重建来说尤其合适，因为每个场景本身就是独特的，通用模型天生有适应空间。

**离实际部署还差什么**：

1. **动态物体**：KITTI 里有行人和车辆，但当前方法主要针对静态背景。真实城市场景的动态处理还是个开放问题。
2. **场景尺度**：超大场景（几公里）需要分层存储策略，单一上下文向量是否够用存疑。
3. **测试时计算**：自动驾驶对延迟敏感，TTT 的额外开销需要更好的工程优化（如异步适应）。

总体来说，Scal3R 代表了前馈3D重建走向实用化的一个重要步骤，但在工业落地之前，还需要在动态场景鲁棒性和硬件效率上下功夫。官方代码在 [https://zju3dv.github.io/scal3r](https://zju3dv.github.io/scal3r)，值得复现研究。