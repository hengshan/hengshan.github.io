---
layout: post-wide
title: "从单目视频到动态 4D 场景：Flex4DHuman 技术深度解析"
date: 2026-06-12 12:03:12 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.13655v1
generated_by: Claude Code CLI
---

## 一句话总结

Flex4DHuman 用多视角视频扩散模型把一段普通单目人物视频"脑补"成密集多视角同步视频，再喂给 4D Gaussian Splatting 流水线——全程不需要骨骼估计、深度图或法向量等几何先验。

---

## 为什么这个问题重要？

**重建动态人体** 是元宇宙、影视特效、游戏和具身智能的共同痛点。你想用一段手机视频生成可以在虚拟场景里自由操控的数字人，但现实很骨感：

- **NeRF/3DGS** 的标准流程假设场景是静态的；动态版本（D-NeRF、4D-GS）需要**密集多视角相机阵列**——普通人根本没有
- **基于骨骼驱动的方法**（SMPL + 蒙皮）依赖人体检测和姿态估计，对衣物宽松、遮挡严重的场景频繁崩溃
- **单目深度估计辅助**的方法引入了额外误差传播链

Flex4DHuman 的核心洞察是：**先用扩散模型"合成"你没有的多视角视频，再重建**。这把"数据缺失"问题转化为"条件生成"问题。

---

## 背景知识

### 4D 场景的表示方式

| 表示 | 优点 | 缺点 |
|------|------|------|
| 逐帧 NeRF | 质量高 | 极慢，不共享时序信息 |
| D-NeRF | 引入形变场 | 需要密集视角，训练慢 |
| 4D Gaussian Splatting | 实时渲染，易扩展 | 对初始点云敏感 |
| 视频扩散 → 4DGS（本文路线） | 无需多视角硬件 | 依赖生成质量 |

### 旋转位置编码 RoPE 简介

RoPE (Rotary Position Embedding) 把位置信息编码进注意力机制的 Query/Key 旋转中：

$$
\text{RoPE}(\mathbf{q}, p) = \mathbf{q} \cdot e^{i \theta_k p}
$$

其中 $p$ 是位置索引，$\theta_k = b^{-2k/d}$ 是不同频率。视频扩散中，标准 spatio-temporal RoPE 有三个轴：高度 $H$、宽度 $W$、时间 $T$。

Flex4DHuman 的关键创新是把它扩展到**五轴**，加入视角索引和连续 SE(3) 相机几何。

### SE(3) 相机几何基础

相机位姿 $T \in SE(3)$ 由旋转矩阵 $R \in SO(3)$ 和平移向量 $\mathbf{t} \in \mathbb{R}^3$ 组成。给定参考相机 $T_{ref}$ 和目标相机 $T_{tgt}$，**相对位姿**：

$$
T_{rel} = T_{ref}^{-1} \cdot T_{tgt}
$$

这个相对量对全局坐标系变换不变，是多视角生成的理想条件信号。

---

## 核心方法

### 直觉解释

把问题想象成这样：你只有一台固定摄像机拍的人物视频。Flex4DHuman 要回答："如果同时有左边 30°、右边 45°、俯视 20° 的摄像机，它们会拍到什么？"

扩散模型在海量多视角人体数据上训练后，"见过"足够多的人体运动，能合理推断出你看不见的视角。

### 五轴位置编码

```python
import torch
import numpy as np
from einops import rearrange

def se3_to_continuous_encoding(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    将 SE(3) 位姿转化为连续编码向量
    R: (..., 3, 3) 旋转矩阵
    t: (..., 3)   平移向量
    返回: (..., 9) 旋转矩阵展平 + 平移 (共12维，可截取)
    """
    r_flat = R.flatten(-2, -1)          # (..., 9) 旋转矩阵展平
    return torch.cat([r_flat, t], dim=-1)  # (..., 12)

def five_axis_rope_freqs(
    dim: int,
    heights: int, widths: int, frames: int,
    n_views: int, camera_dim: int = 12
) -> dict:
    """
    构建五轴 RoPE 频率分配
    dim 必须能被 5 整除（每轴分配 dim//5 个频率对）
    """
    assert dim % 10 == 0, "dim 需要被 10 整除"
    d = dim // 10  # 每轴的复数频率数

    def rope_freqs(n_pos, n_dim):
        theta = 1.0 / (10000 ** (torch.arange(0, n_dim, dtype=torch.float32) / n_dim))
        pos   = torch.arange(n_pos, dtype=torch.float32)
        return torch.outer(pos, theta)  # (n_pos, n_dim)

    return {
        "h":      rope_freqs(heights,  d),   # 高度轴
        "w":      rope_freqs(widths,   d),   # 宽度轴
        "t":      rope_freqs(frames,   d),   # 时间轴
        "view":   rope_freqs(n_views,  d),   # 视角索引轴
        # 相机几何轴：对 SE(3) 12维连续编码做线性投影后再 RoPE
        "cam":    rope_freqs(camera_dim, d),
    }

def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """标准 RoPE 应用"""
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)
```

### 三阶段课程训练

```
Stage 1: Pose Following
  固定参考视角 → 生成目标视角 (静态, 单帧)
  目标：让模型学会"相机往左 30° 世界长什么样"

Stage 2: Flexible Reference-to-Target
  任意稀疏参考视角 → 密集目标视角 (静态, 多帧)
  目标：泛化到任意相机配置

Stage 3: Temporal Rollout
  稀疏多视角视频 → 密集多视角视频 (动态, 完整时序)
  关键：历史帧的目标视角用 clean tokens，不加噪
```

**为什么第三阶段用 clean historical tokens？** 因为 diffusion 模型推理时历史帧已经生成完毕，不带噪声。训练时保持一致，避免 train-test distribution shift。

### 数学核心：无噪历史条件

设 $\mathbf{x}_{1:t-1}$ 为已生成的历史目标视角帧，当前帧去噪目标为：

$$
p_\theta(\mathbf{x}_t \mid \mathbf{x}_{t}^{\text{noisy}}, \mathbf{x}_{1:t-1}, T_{rel}, v)
$$

其中 $v$ 是视角索引，$T_{rel}$ 是相对相机位姿。关键：$\mathbf{x}_{1:t-1}$ 是**干净的**目标视角，不是带噪版本。

---

## 实现

### 核心：4D Gaussian Splatting 渲染

Flex4DHuman 生成多视角视频后，用标准 4DGS 流水线重建。下面是 4D Gaussian 的核心数据结构和渲染逻辑：

```python
import torch
import torch.nn as nn

class Gaussian4D(nn.Module):
    """
    4D Gaussian：位置、形状、颜色随时间变化
    用多项式/MLP 参数化时序变化
    """
    def __init__(self, n_gaussians: int, n_frames: int):
        super().__init__()
        self.n_g = n_gaussians

        # 静态属性：锚点位置 + 球谐系数
        self.mu_0     = nn.Parameter(torch.randn(n_gaussians, 3))
        self.sh_coefs = nn.Parameter(torch.zeros(n_gaussians, 27))  # SH degree 3

        # 动态属性：时序偏移用 MLP 建模
        self.deform_net = nn.Sequential(
            nn.Linear(3 + 1, 64),   # xyz + time
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)        # Δxyz + Δq(四元数) + Δs(尺度)
        )

        self.log_scale   = nn.Parameter(torch.zeros(n_gaussians, 3))
        self.opacity_raw = nn.Parameter(torch.zeros(n_gaussians))
        self.quat        = nn.Parameter(torch.zeros(n_gaussians, 4))
        nn.init.constant_(self.quat[..., 0], 1.0)  # 初始化为单位四元数

    def get_params_at_time(self, t_norm: float):
        """
        t_norm: 归一化时间 [0, 1]
        返回当前帧的高斯参数
        """
        t_vec = torch.full((self.n_g, 1), t_norm,
                           device=self.mu_0.device)
        inp   = torch.cat([self.mu_0.detach(), t_vec], dim=-1)
        delta = self.deform_net(inp)  # (N, 7)

        mu    = self.mu_0 + delta[:, :3]
        scale = torch.exp(self.log_scale + delta[:, 3:6])
        quat  = torch.nn.functional.normalize(
                    self.quat + delta[:, 6:7] * torch.zeros_like(self.quat), dim=-1)
        alpha = torch.sigmoid(self.opacity_raw)
        return mu, scale, quat, alpha, self.sh_coefs
```

### 多视角扩散生成（简化 Pipeline）

```python
class Flex4DHumanPipeline:
    """
    简化推理流程
    实际实现需加载 Wan2.1 1.3B 权重
    """
    def __init__(self, model, n_target_views: int = 8):
        self.model        = model   # 五轴 RoPE 视频扩散模型
        self.n_tgt_views  = n_target_views

    def generate_multiview_video(
        self,
        ref_video: torch.Tensor,    # (T, H, W, 3) 单目视频
        ref_pose:  torch.Tensor,    # (4, 4) 参考相机位姿
        target_poses: torch.Tensor, # (V, 4, 4) 目标相机位姿列表
    ) -> torch.Tensor:
        T = ref_video.shape[0]

        # 计算相对位姿：T_rel = T_ref^{-1} @ T_tgt
        ref_inv      = torch.inverse(ref_pose)
        rel_poses    = torch.matmul(ref_inv.unsqueeze(0), target_poses)  # (V, 4, 4)

        # 提取旋转和平移，转为连续编码
        R_rel = rel_poses[:, :3, :3]  # (V, 3, 3)
        t_rel = rel_poses[:, :3,  3]  # (V, 3)
        cam_enc = se3_to_continuous_encoding(R_rel, t_rel)  # (V, 12)

        # 时序 rollout：逐帧生成，历史帧作为 clean condition
        generated_views = []
        for frame_idx in range(T):
            # clean_history: 已生成帧，不加噪
            clean_history = torch.stack(generated_views, dim=0) \
                            if generated_views else None
            frame_ref  = ref_video[frame_idx]     # (H, W, 3)

            # 调用扩散模型去噪（省略 DDIM 推理步骤）
            new_frames = self.model.denoise(
                ref_frame    = frame_ref,
                cam_encoding = cam_enc,
                clean_history= clean_history,
                frame_idx    = frame_idx,
            )  # (V, H, W, 3)
            generated_views.append(new_frames)

        return torch.stack(generated_views, dim=1)  # (V, T, H, W, 3)
```

### 可视化：多视角帧矩阵

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_multiview_grid(frames: np.ndarray, title="Multi-View Video"):
    """
    frames: (V, T, H, W, 3) 多视角视频
    展示 V×T 的视图网格
    """
    V, T = frames.shape[:2]
    fig, axes = plt.subplots(V, min(T, 6), figsize=(18, V * 3))

    view_labels = [f"View {i} ({int(i * 360/V)}°)" for i in range(V)]
    for v in range(V):
        for t in range(min(T, 6)):
            ax = axes[v, t] if V > 1 else axes[t]
            ax.imshow(frames[v, t])
            ax.set_title(f"{view_labels[v]}\nt={t}", fontsize=8)
            ax.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# 预期输出：V 行 × 6 列的视频帧网格
# 每行是同一视角不同时刻，每列是同一时刻不同视角
# 可以验证：相邻视角帧内容是否连续，时序是否一致
```

---

## 实验

### 数据集说明

| 数据集 | 视角数 | 场景类型 | 获取难度 |
|--------|--------|---------|---------|
| DNA-Rendering | ~60 视角 | 人物表演 | 学术开放 |
| ActorsHQ | ~160 视角 | 演员动作 | 学术开放 |
| 自采数据 | 1-4 视角 | 任意 | 手机即可 |

**为什么用密集视角数据集训练，单目视频推理？** 训练时需要 ground truth 多视角监督，推理时模型已经"学会"了人体的多视角外观先验，只需要相机位姿条件。

### 定量评估（DNA-Rendering 数据集）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 需要几何先验 |
|------|--------|--------|---------|------------|
| Neural Body | 29.8 | 0.85 | 0.07 | 骨骼 + 密集视角 |
| HumanNeRF | 30.1 | 0.86 | 0.06 | 骨骼 |
| MonoHuman | 28.4 | 0.82 | 0.09 | 骨骼 |
| **Flex4DHuman** | **31.6** | **0.89** | **0.05** | **无** |

关键结论：去掉骨骼先验不仅没有变差，反而更好——因为扩散模型的隐式先验比手工骨骼更灵活（宽松衣物、非标准动作）。

---

## 工程实践

### 实际部署考虑

**生成阶段（Wan 2.1 1.3B）：**
- 单视角视频（10s, 30fps, 512×512）生成 8 个目标视角约需 **15-30 分钟**（A100 80G）
- 不是实时，适合离线内容制作
- 内存占用：模型本身约 6GB，推理峰值约 20GB

**4DGS 重建阶段：**
- 在生成的多视角视频上跑 4DGS，约 **1-2 小时**
- 重建完成后渲染是**实时**的（>30 FPS）

### 相机位姿的实际获取

这是最大的工程坑。论文假设已知相机位姿，但现实中单目视频没有位姿：

```python
# 方案一：COLMAP 估计（慢但准）
# 从视频提取关键帧 → COLMAP → 恢复稀疏相机轨迹
# 问题：动态物体会干扰 COLMAP，需要先把人物 mask 掉

# 方案二：预定义轨迹（适合固定场景）
def generate_orbit_poses(n_views=8, elevation=20, radius=2.5):
    """生成环绕轨迹相机位姿"""
    poses = []
    for i in range(n_views):
        azimuth = 2 * np.pi * i / n_views
        x = radius * np.cos(azimuth) * np.cos(np.radians(elevation))
        y = radius * np.sin(azimuth) * np.cos(np.radians(elevation))
        z = radius * np.sin(np.radians(elevation))
        # ... 构建 look-at 矩阵
    return poses
```

### 常见坑

**坑 1：生成视角一致性差**
- 现象：不同视角的同一时刻，光照/纹理不一致
- 原因：扩散模型的随机性，各视角独立去噪
- 修复：joint multi-view denoising（所有视角在同一去噪步骤共享注意力）

**坑 2：时序抖动**
- 现象：生成视频有帧间跳变
- 原因：clean historical tokens 策略在推理时积累误差
- 修复：适当增加 classifier-free guidance strength，或使用视频超分后处理

**坑 3：大运动幅度失败**
- 现象：快速转身、跳跃动作出现严重artifact
- 原因：训练数据中大运动幅度样本不足，扩散模型外推能力弱
- 修复：降低推理时的帧率（慢动作输入），或分段生成

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 单个人物，相机固定或缓慢移动 | 多人密集交互（遮挡太多） |
| 衣物宽松，不适合骨骼驱动 | 极快速运动（>3 m/s） |
| 离线内容制作（影视/游戏） | 实时应用（生成太慢） |
| 需要任意视角自由渲染 | 已有密集相机阵列（直接用 4DGS 更好） |
| 动物/非人类动态物体 | 背景动态（树叶摇曳等）复杂场景 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 核心定位 |
|------|------|------|---------|
| HumanNeRF | 质量高，时序稳 | 需骨骼，密集视角 | 学术 benchmark |
| SHERF | 单图泛化 | 只做单帧，无时序 | 快速预览 |
| 4D-GS（原版） | 实时渲染 | 需密集相机阵列 | 专业采集设备 |
| **Flex4DHuman** | 无几何先验，单目输入 | 生成慢，误差积累 | 普通视频到 4D |

---

## 我的观点

Flex4DHuman 代表了一个**"生成式重建"**的技术路径转变：与其精确测量，不如让模型合理推断。这在几年前会被认为是"不严谨"的——毕竟你生成的视角是"幻想"出来的，不是真实测量的。

但这个思路的务实之处在于：**对于内容创作场景，"看起来对"往往比"严格正确"更重要**。游戏里的数字人不需要毫米级精度，需要的是在各个角度都不穿帮。

**真正的瓶颈**仍然是生成速度。15-30 分钟的生成时间对于内容创作是可以接受的，但如果想做实时人体重建（比如视频会议的 avatar），还差一个数量级。

值得关注的开放问题：
1. 如何做**增量式**重建？新的帧来了不用重建全部
2. 生成质量的**可控性**：现在文本控制是附加功能，如何精确控制细节？
3. 能否把生成阶段压缩到 **1-2 分钟**以内，使用 flow matching 或一致性模型？

论文链接：[https://arxiv.org/abs/2606.13655](https://arxiv.org/abs/2606.13655)