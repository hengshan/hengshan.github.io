---
layout: post-wide
title: "PPMem：用平面基元构建视频世界模型的持久三维记忆"
date: 2026-10-19 12:04:07 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2610.23795v1
generated_by: Claude Code CLI
---

## 一句话总结

PPMem 是一个视频世界模型（video world model），它在逐段（chunk）自回归生成视频的同时，联合预测一组"定向平面基元"（oriented plane primitives）作为场景的持久三维记忆——每个基元只需 7 个参数，比逐像素点云紧凑三个数量级，却能在生成数百帧、大幅视角变化的过程中保持几何一致性。

## 为什么这个问题重要？

视频生成模型（尤其是视频扩散 Transformer）最近被当作"世界模型"来用：给定一段起始视频和相机轨迹，模型自回归地"生成"接下来会看到的画面，用于游戏引擎替代、机器人仿真、AR/VR 内容生成等场景。这类应用的核心痛点是**长时程三维一致性**——当相机绕一圈回到起点，生成的画面必须和之前生成的内容对得上，而不是凭空想象出一堵新墙。

纯 2D 的视频扩散模型没有显式三维状态，只靠注意力机制隐式"记住"之前生成的帧，一旦序列变长、视角变化变大，很容易出现几何漂移甚至场景"穿帮"。此前也有工作尝试用**逐像素点云**或**体素**作为显式三维记忆来约束生成，但点云/体素在场景规模变大时内存和计算开销会线性甚至更快地增长，很难支撑"数百帧、跨越整个场景"的生成长度。

PPMem 的核心创新是：室内外场景里大量结构（墙面、地板、天花板、建筑立面）本身就接近平面，用一小撮"定向平面"而不是海量像素点就能描述场景骨架。这把三维记忆的规模从几十万到几百万个点，压缩到约 10 万个平面基元、整场景仅 5MB，同时保留了显式的度量深度、法向量和可导出表面。

## 背景知识

### 3D 表示方式对比

| 表示 | 紧凑性 | 几何显式性 | 适合做"持久记忆"吗 |
|-----|-------|----------|------------------|
| 逐像素点云 | 差（随帧数线性增长） | 高 | 大场景下内存爆炸 |
| 体素 | 差（立方级增长） | 高 | 分辨率受限，难覆盖大场景 |
| NeRF 隐式场 | 好 | 低（需要渲染才能查询几何） | 训练慢，难以增量更新 |
| 3D Gaussian Splatting | 中等（百万级高斯） | 中等 | 适合重建已有场景，但不是为"生成式增量记忆"设计 |
| 平面基元（PPMem） | 很好（~10 万个，7 参数/个） | 高（度量深度、法向、可导出面片） | 专为增量生成+融合设计 |

需要强调的是，**PPMem 解决的是"生成"问题而不是"重建"问题**：它不是在拟合一个已经存在的真实场景（这是 NeRF/3DGS 的强项），而是在生成一个此前并不存在、只在模型想象中成立的场景，并要求这个想象在三维上自洽。

### 相机模型

PPMem 沿用标准针孔相机模型：已知内参矩阵 $K$ 和每帧的相机位姿（外参）$T = [R \mid t]$，像素坐标 $(u, v)$ 对应一条从相机光心出发的射线。平面基元的渲染（即把平面集合转换成某个视角下的深度图）本质上就是求"这条射线最先打到哪个平面"的经典光线-平面求交问题。

## 核心方法

### 直觉解释

想象生成过程被切成一段段视频"块"（chunk）。每生成完一个块，模型除了吐出 RGB 像素，还顺带吐出这个块里看到的场景由哪些平面构成。这些新平面会跟"记忆库"里已有的平面做匹配：如果是同一面墙的新观测，就把两者融合、让参数更准；如果是从没见过的新结构，就追加进记忆库。下一个块开始生成之前，先把当前相机视角下的记忆库"渲染"成一张深度图，作为强先验喂给扩散模型——这样模型在画新内容时，就知道"这里应该是一堵墙，离相机多远"，而不用凭空猜。

### 数学细节

一个平面基元用 7 个参数描述：中心点 $c \in \mathbb{R}^3$（3 维）、法向方向用球坐标角 $(\theta, \phi)$ 表示（2 维，保证单位范数且连续可微）、平面内旋转角 $\psi$（1 维，确定矩形块的朝向）、以及一个尺寸标量 $s$（1 维，近似矩形块的半边长）。平面方程为：

$$
n \cdot (x - c) = 0, \quad n = \begin{bmatrix}\sin\theta\cos\phi \\ \sin\theta\sin\phi \\ \cos\theta\end{bmatrix}
$$

给定某视角下一条像素射线 $x(t) = o + t d$（$o$ 为相机光心，$d$ 为归一化的世界系射线方向），与平面的交点参数为：

$$
t^\star = \frac{n \cdot (c - o)}{n \cdot d}
$$

把所有平面对每个像素求交、取最近的有效交点（$t^\star > 0$ 且落在矩形块范围内），就得到了记忆库在当前视角下的渲染深度图——这正是 Depth Token Fusion（DTF）模块要用来条件化扩散模型的信号。

训练时的监督目标是光栅化深度与法向的几何损失，加上视频扩散本身的去噪损失，两者联合优化，使得生成的 RGB 内容与同时预测出的平面几何相互一致。

### Pipeline 概览

```
起始帧/相机轨迹
      │
      ▼
┌───────────────────────────┐
│  Chunk k: 视频扩散 Transformer │
│  （RGB分支 + 平面预测头）       │
└───────────────────────────┘
      │ 新平面基元                │ 渲染出的条件深度图(来自记忆)
      ▼                           ▲
┌───────────────┐          ┌──────────────┐
│  平面记忆合并     │─────────►│ DTF: 记忆渲染  │
│  (匹配/融合/追加) │          │ 为Chunk k+1条件│
└───────────────┘          └──────────────┘
      │
      ▼
  持久平面记忆（~10万个基元，5MB）
```

## 实现

以下代码是对 PPMem 核心思想的最小化复现，用来理解算法骨架，不是论文的官方实现（论文未公开代码仓库，只给出了项目主页）。

### 环境配置

```bash
pip install torch numpy open3d
```

### 平面渲染：从基元到深度图

```python
import numpy as np

class PlanePrimitive:
    """7参数平面基元: 中心(3) + 法向角theta,phi(2) + 内旋转(1) + 尺寸(1)"""
    def __init__(self, params: np.ndarray):
        self.center = params[0:3]
        theta, phi = params[3], params[4]
        self.normal = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])
        self.inplane_rot = params[5]
        self.size = max(params[6], 1e-3)

def render_depth_from_planes(planes, K, T_c2w, H, W, eps=1e-6):
    """把平面集合渲染成当前相机位姿下的深度图（z-buffer最近面）"""
    o = T_c2w[:3, 3]
    R = T_c2w[:3, :3]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us, vs, np.ones_like(us)], -1).reshape(-1, 3).astype(np.float32)
    dirs_cam = pix @ np.linalg.inv(K).T
    dirs_world = dirs_cam @ R.T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)

    depth = np.full(H * W, np.inf, dtype=np.float32)
    for p in planes:
        denom = dirs_world @ p.normal
        valid = np.abs(denom) > eps
        t = np.full(H * W, np.inf, dtype=np.float32)
        t[valid] = (p.normal @ (p.center - o)) / denom[valid]
        hit = dirs_world[valid] * t[valid, None] + o
        within = np.linalg.norm(hit - p.center, axis=-1) <= p.size
        update = np.zeros(H * W, dtype=bool)
        update[valid] = within
        closer = (t > 0) & (t < depth) & update
        depth[closer] = t[closer]
    depth[np.isinf(depth)] = 0.0
    return depth.reshape(H, W)
```

### 记忆合并：匹配、融合与追加

```python
def merge_plane_memory(memory, new_planes, normal_thresh=0.97, dist_thresh=0.1):
    """将新一轮chunk预测的平面与已有记忆匹配融合，未匹配的追加入库"""
    matched_new = set()
    for i, m in enumerate(memory):
        best_j, best_score = -1, -1.0
        for j, p in enumerate(new_planes):
            if j in matched_new:
                continue
            cos_sim = np.dot(m.normal, p.normal)
            offset_diff = abs(np.dot(m.normal, m.center) - np.dot(m.normal, p.center))
            if cos_sim > normal_thresh and offset_diff < dist_thresh and cos_sim > best_score:
                best_j, best_score = j, cos_sim
        if best_j >= 0:
            p = new_planes[best_j]
            # 简单加权平均融合（可替换为基于观测置信度的加权）
            m.center = 0.5 * (m.center + p.center)
            m.normal = (m.normal + p.normal)
            m.normal /= np.linalg.norm(m.normal)
            m.size = max(m.size, p.size)
            matched_new.add(best_j)
    for j, p in enumerate(new_planes):
        if j not in matched_new:
            memory.append(p)
    return memory
```

### Depth Token Fusion（DTF）模块

```python
import torch
import torch.nn as nn

class DepthTokenFusion(nn.Module):
    """把渲染深度图token化, 以ControlNet式零初始化方式注入视频token流"""
    def __init__(self, patch_size=16, dim=768):
        super().__init__()
        self.patchify = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(dim, dim)
        # 零初始化: 训练初期不扰动预训练好的视频扩散主干
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, video_tokens, rendered_depth):
        # video_tokens: (B, N, C)   rendered_depth: (B, 1, H, W)
        d_tok = self.patchify(rendered_depth)          # (B, C, H/p, W/p)
        d_tok = d_tok.flatten(2).transpose(1, 2)        # (B, N, C)
        cond = self.proj(d_tok)
        return video_tokens + cond  # 加性条件注入, 供后续自注意力层使用
```

### 3D 可视化：记忆库中的平面

```python
import open3d as o3d

def visualize_plane_memory(memory):
    meshes = []
    for p in memory:
        box = o3d.geometry.TriangleMesh.create_box(width=p.size*2, height=p.size*2, depth=0.01)
        box.translate(-np.array([p.size, p.size, 0.005]))
        # 把box从默认z轴朝向旋转到平面法向
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, p.normal)
        angle = np.arccos(np.clip(np.dot(z_axis, p.normal), -1, 1))
        if np.linalg.norm(axis) > 1e-6:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle)
            box.rotate(R, center=(0, 0, 0))
        box.translate(p.center)
        box.paint_uniform_color(np.random.rand(3))
        meshes.append(box)
    o3d.visualization.draw_geometries(meshes)
```

预期效果：随着自回归块数增加，`memory` 中平面数量会先快速增长（探索新区域），后趋于稳定（相机回到已生成区域时触发融合而非新增）——这正是论文强调的"自适应记忆"特性，可以通过打印 `len(memory)` 随块数变化的曲线直观看到。

## 实验

PPMem 在 **DL3DV** 和 **RealEstate10K** 两个带位姿的真实视频数据集上评测，这两个数据集本身就是训练/评测相机受控视频模型的常用选择（多为室内外步行/驾驶轨迹拍摄）。论文摘要给出的结论是：在视频质量上超过此前"3D 约束"的视频世界模型基线，同时额外产出度量深度、法向量和可导出的平面几何，并能在数百帧、大视角变化下保持连贯性。

需要诚实说明的是，论文摘要本身没有给出具体的 PSNR/SSIM/FPS 数值，因此本文不编造定量对比表格——如果需要精确数字，请以项目主页或正式论文中的表格为准。从方法设计上可以推断出的定性结论是：

- **紧凑性**是实打实的优势：约 10 万个平面基元、5MB，比动辄百万级点的点云记忆小三个数量级，这意味着可以支撑更长的生成序列而不耗尽显存。
- **跨视图一致性**来自"平面预测头和 RGB 分支共享同一个扩散骨干网络联合预测"，而不是先生成视频再离线做三维重建——这是它区别于"生成后处理估计深度"类方法的关键。

## 工程实践

### 实际部署考虑

视频扩散 Transformer 本身的推理成本很高，逐块自回归生成意味着每个 chunk 都要走一次完整的去噪采样过程（多步迭代）。这类模型目前的定位更接近**离线/半交互式世界生成**（比如预渲染一段可探索的场景、为机器人仿真提供素材），而不是能插入实时控制回路（比如游戏手柄输入直接驱动下一帧）的系统。如果追求低延迟交互，需要配合蒸馏采样（few-step diffusion）、KV 缓存复用等工程手段，论文摘要本身没有涉及这部分优化。

### 常见坑

1. **射线几乎平行于平面（$n \cdot d \approx 0$）导致求交发散**：上面 `render_depth_from_planes` 里用 `eps` 过滤 `denom` 接近零的情况，否则 `t` 会出现数值爆炸，深度图里出现孤立亮点。
2. **记忆库中相似平面因为融合阈值过松而被错误合并**（比如把两面平行但实际不同的墙合成一面）：`normal_thresh` 和 `dist_thresh` 需要结合场景尺度调参，场景越大，`dist_thresh` 应该相应放宽。
3. **长序列生成中平面记忆无限增长拖慢渲染**：论文强调记忆"随生成过程被合并精化"而非单纯累加，工程实现上需要定期做一次全局去重/合并，而不是只做相邻 chunk 间的增量匹配。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 以平面结构为主的室内/建筑场景生成 | 植被、人群等非平面、高频几何为主的自然场景 |
| 需要长序列、大视角变化下的三维一致性 | 只需要单帧或短视频生成，三维一致性要求不高 |
| 需要同时拿到可导出几何（供下游仿真/物理引擎使用） | 只关心像素级视觉效果，不需要几何输出 |
| 离线/半交互式内容生产流程 | 要求低延迟实时交互的应用 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| NeRF | 几何连续、渲染质量高 | 训练慢、难增量更新、不是为生成设计 | 已有场景的高质量重建 |
| 3D Gaussian Splatting | 渲染快、适合实时查看 | 重建导向，百万级基元仍不算轻量 | 已采集场景的实时渲染 |
| 纯2D视频扩散世界模型 | 架构简单，不需要显式三维标注 | 长序列三维一致性差，容易"穿帮" | 短视频生成、对几何一致性要求低的场景 |
| PPMem | 记忆紧凑、几何显式、跨视图一致 | 依赖场景以平面结构为主，复杂有机几何表现未知 | 以平面结构为主的长序列场景生成 |

## 我的观点

PPMem 把"视频生成"和"三维表示"结合得比较巧妙的地方，是它没有把三维记忆当作生成之外的附加模块，而是让平面预测头和 RGB 分支共享同一个骨干、联合训练——这比"先生成视频再做深度估计/重建"的两阶段方案更有机会避免误差累积。平面基元作为记忆载体的选择也很务实：它牺牲了对非平面结构（植被、曲面家具）的表达能力，换来了数量级的压缩和清晰的几何语义，这个取舍对室内外建筑类场景是合理的，但对自然场景或高杂乱度场景的泛化能力还有待验证——论文摘要并未覆盖这类场景的评测。

更值得关注的开放问题是实时性：目前这类自回归视频扩散世界模型离"交互式可控探索"还有明显距离，平面记忆解决的是"生成内容是否几何自洽"的问题，而不是"生成速度是否足够快"的问题。两者结合可能需要配合扩散加速技术才能真正落地到需要实时反馈的应用（如机器人仿真训练循环）中。