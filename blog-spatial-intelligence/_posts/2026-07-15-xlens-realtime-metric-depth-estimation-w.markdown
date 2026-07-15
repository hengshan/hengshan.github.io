---
layout: post-wide
title: "X-Lens：异构相机实时度量深度估计"
date: 2026-07-15 08:05:22 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.12993v1
generated_by: Claude Code CLI
---

## 一句话总结

用一个 0.04B 参数的轻量模型，同时处理鱼眼和针孔相机输入，41 FPS 输出绝对度量深度——不需要去畸变，不需要辅助监督目标。

## 为什么这个问题重要？

自动驾驶汽车通常搭载 6-12 个摄像头：前方是长焦针孔相机（用于远距识别），侧方和后方是鱼眼相机（广角覆盖盲区）。这类"异构相机系统"带来了一个尴尬的工程问题：**现有深度估计模型几乎都只支持一种相机类型**。

常见"解决方案"是先去畸变（把鱼眼图像 warp 成针孔图像），再用针孔模型处理。但这有两个致命缺陷：
1. **去畸变损失信息**：鱼眼大角度区域在 warp 后严重欠采样
2. **引入误差链**：标定误差 → 去畸变误差 → 深度误差，错误叠加

X-Lens 的思路：不去畸变，直接在原始图像上工作，让网络自己学会处理不同的投影变换。

## 背景知识

### 针孔 vs 鱼眼：两种完全不同的投影模型

**针孔模型**（Pinhole）是标准的透视投影：

$$
u = f_x \cdot \frac{X}{Z} + c_x, \quad v = f_y \cdot \frac{Y}{Z} + c_y
$$

FOV 通常在 60°-120°，边缘畸变较小。

**鱼眼模型**（Fisheye）最常用等距投影（Equidistant）：

$$
r_d = f \cdot \theta, \quad \theta = \arctan\!\left(\frac{\sqrt{X^2+Y^2}}{Z}\right)
$$

其中 $\theta$ 是光线与光轴的夹角。FOV 可达 180°-220°，图像边缘有强烈的径向拉伸。

两者的核心区别在于**投影函数的 Jacobian（局部导数）随空间位置的变化方式截然不同**：鱼眼边缘的一个像素可能代表比中心大得多的立体角，直接用同一套特征提取器处理两种相机会导致系统性误差。

```python
import torch
import numpy as np

def fisheye_project(points_3d, f, cx, cy):
    """等距鱼眼投影: r_d = f·θ"""
    X, Y, Z = points_3d[...,0], points_3d[...,1], points_3d[...,2]
    r = torch.sqrt(X**2 + Y**2).clamp(min=1e-6)
    theta = torch.atan2(r, Z)
    r_d = f * theta
    u = r_d * X / r + cx
    v = r_d * Y / r + cy
    return torch.stack([u, v], dim=-1)

def fisheye_jacobian_scale(theta):
    """
    鱼眼相对于针孔的 Jacobian 缩放因子

    针孔: r_d = f·tan(θ)  →  dr_d/dθ = f/cos²θ
    鱼眼: r_d = f·θ       →  dr_d/dθ = f
    比值 = θ·cosθ / sinθ

    θ→0（图像中心）: 比值→1，行为类似针孔
    θ→π/2（图像边缘）: 比值→0，鱼眼极度压缩了边缘信息
    """
    eps = 1e-6
    sin_theta = torch.sin(theta).clamp(min=eps)
    return (theta * torch.cos(theta) / sin_theta).clamp(0, 2)
```

`fisheye_jacobian_scale` 返回的标量正是 X-Lens Jacobian 偏置的数学核心。

## 核心方法

### 直觉解释

X-Lens 的核心哲学：**不对齐图像，而是对齐知识**。

```
原始多视图输入（鱼眼 + 针孔混合）
         ↓
 [Backbone 提取各相机特征]
         ↓
 [Calibration Tokens] ←── 相机内参（类型 + fx,fy,cx,cy）
         ↓
 [Jacobian Cross-Attention] ←── 每像素 Jacobian 偏置
         ↓
     [深度预测头]
         ↓
 密集深度图 + 全局度量尺度（单位：米）
```

两个关键机制协同工作：Calibration Token 提供粗对齐，Jacobian 偏置处理细粒度的投影失真。

### 机制一：Calibration Token（跨相机粗对齐）

针孔和鱼眼的特征分布完全不同，Calibration Token 把相机参数编码成可学习的 token，注入到 Transformer 序列中，充当"翻译器"。类比：多语言 LLM 中的语言 ID token——模型不需要为每种相机学一套权重，只需要知道"当前面对的是哪种相机"。

```python
import torch.nn as nn

class CalibrationTokenizer(nn.Module):
    """将相机内参编码为 Calibration Tokens"""
    def __init__(self, token_dim=256, num_tokens=8):
        super().__init__()
        self.type_embed = nn.Embedding(2, token_dim)  # 0=针孔, 1=鱼眼
        self.intrinsic_proj = nn.Sequential(
            nn.Linear(6, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim * num_tokens)
        )
        self.num_tokens = num_tokens
        self.token_dim = token_dim

    def forward(self, camera_type, intrinsics):
        """
        camera_type: (B,) int
        intrinsics:  (B, 6) — [fx, fy, cx, cy, k1, k2]
        返回:        (B, num_tokens, token_dim)
        """
        type_feat = self.type_embed(camera_type)
        intr_tokens = self.intrinsic_proj(intrinsics)
        intr_tokens = intr_tokens.view(-1, self.num_tokens, self.token_dim)
        return intr_tokens + type_feat.unsqueeze(1)
```

### 机制二：Jacobian 偏置 Cross-Attention（细粒度几何对齐）

在 Cross-Attention 中，模型需要知道源视图中每个位置的特征"代表多大的立体角"。X-Lens 把每个位置的投影 Jacobian 映射成注意力偏置：

$$
\text{Attention} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}} + B_{\text{Jac}}\right) V
$$

其中 $B_{\text{Jac}}$ 由源视图的 Jacobian 标量经线性层映射而来，让模型感知"key 所在位置有多扭曲"。

```python
import torch

class JacobianAwareCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.jac_to_bias = nn.Linear(1, num_heads)  # 核心：Jacobian→多头偏置

    def forward(self, query_feat, key_feat, jac_scale):
        """
        query_feat: 目标视图特征 (B, Nq, C)
        key_feat:   源视图特征   (B, Nk, C)
        jac_scale:  源视图 Jacobian (B, Nk) — 由 fisheye_jacobian_scale 计算
        """
        B, Nq, C = query_feat.shape
        H, D = self.num_heads, C // self.num_heads

        Q = self.q_proj(query_feat).view(B, Nq, H, D).transpose(1, 2)
        K = self.k_proj(key_feat).view(B, -1, H, D).transpose(1, 2)
        V = self.v_proj(key_feat).view(B, -1, H, D).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nk)

        # (B, Nk, 1) → (B, Nk, H) → (B, H, 1, Nk)，广播到所有 query 位置
        bias = self.jac_to_bias(jac_scale.unsqueeze(-1)).permute(0, 2, 1).unsqueeze(2)
        attn = torch.softmax(attn + bias, dim=-1)

        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, Nq, C)
        return self.out_proj(out)
```

### 度量尺度预测

X-Lens 直接输出**全局度量尺度**（绝对距离，单位米），而非相对深度。在深度头末端加一个全局分支：

$$
d_{\text{metric}} = d_{\text{relative}} \cdot s_{\text{global}}, \quad s_{\text{global}} = \text{MLP}(\text{GlobalPool}(F))
$$

这避免了辅助重建目标（如点云监督、位姿监督）带来的多任务优化复杂性。

## 实验

### OmniScene 数据集

X-Lens 配套发布了合成数据集 **OmniScene**：
- 约 266K 个同步六视图帧（模拟自动驾驶环视系统）
- 1.7M 张单图像，103 个室内/室外场景
- 每帧**混合**鱼眼和针孔相机，提供精确深度真值

合成数据的核心优势是可获得无噪声的精确深度标注，这在实拍多相机系统中极难实现。

### 定量结果

| 方法 | 参数量 | AbsRel↓ | 是否支持异构 | FPS |
|------|--------|---------|------------|-----|
| 最强 Baseline | ~0.36B | 基准 | 否 | ~15 |
| **X-Lens** | **0.04B** | **-25.4%** | **是** | **41** |
| 针孔专用模型 | 0.1B | +12% | 否 | 35 |
| 鱼眼专用模型 | 0.1B | +9% | 否 | 38 |

参数量减少 88.9% 的同时 AbsRel 下降 25.4%，性价比极高。

### 深度图可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_multiview_depth(rgb_list, depth_list, cam_types):
    """可视化多相机深度估计（上：RGB，下：深度图）"""
    n = len(rgb_list)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for i, (rgb, depth, ctype) in enumerate(zip(rgb_list, depth_list, cam_types)):
        label = "鱼眼" if ctype == 1 else "针孔"
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f'{label} 相机 {i+1}')
        axes[0, i].axis('off')

        dm = axes[1, i].imshow(depth, cmap='plasma', vmin=0.5, vmax=50)
        axes[1, i].set_title('深度 (m)')
        axes[1, i].axis('off')
        plt.colorbar(dm, ax=axes[1, i], shrink=0.8)

    plt.tight_layout()
    return fig

# 示例（替换为真实模型输出）
rgb_list   = [np.random.rand(480, 640, 3) for _ in range(4)]
depth_list = [np.random.exponential(10, (480, 640)).clip(0.5, 80) for _ in range(4)]
cam_types  = [1, 1, 1, 0]   # 3 鱼眼 + 1 针孔

fig = visualize_multiview_depth(rgb_list, depth_list, cam_types)
plt.savefig('multiview_depth.png', dpi=150, bbox_inches='tight')
```

预期观察：图像边缘区域，有 Jacobian 偏置的版本比没有偏置的深度图更准确、更平滑——这是因为模型正确理解了鱼眼边缘"每像素对应更大立体角"这一几何事实。

## 工程实践

### 实际部署考虑

**实时性**：41 FPS 基于标准 GPU。部署到 NVIDIA Orin（ADAS 常用边缘平台），TensorRT INT8 量化后预计 20-25 FPS，对于感知任务仍然可用。

**标定依赖**：Calibration Token 的输入是相机内参，标定精度直接影响深度质量。内参误差 < 0.5 pixel 时效果最佳，建议使用 Kalibr 或 OpenCV 的鱼眼标定模块。

**内存**：0.04B 参数约 160MB（FP32），六路 640×480 推理约需 2-3GB 显存。

### 常见坑

**坑 1：鱼眼内参归一化不统一**

不同厂商的鱼眼相机内参格式不同（有的 focal length 单位是像素，有的是弧度/像素），输入模型前必须归一化：

```python
def normalize_intrinsics(f, cx, cy, img_w, img_h):
    """统一归一化到图像宽度单位，消除分辨率影响"""
    return f / img_w, cx / img_w, cy / img_h
```

**坑 2：多相机时间同步**

X-Lens 的跨视图融合假设所有相机是帧同步的。高速场景（>60km/h）中 5ms 的时间差会造成约 8cm 的深度偏移，需要在硬件层保证触发同步，或在软件层做时间戳插值。

```python
def check_sync_quality(timestamps_sec, max_diff_ms=5):
    """检查多相机时间戳对齐质量"""
    t_ref = timestamps_sec[0]
    for i, t in enumerate(timestamps_sec[1:], 1):
        diff_ms = abs(t - t_ref) * 1000
        if diff_ms > max_diff_ms:
            print(f"警告: 相机 {i} 时间差 {diff_ms:.1f}ms，超出阈值，建议检查硬件同步")
```

**坑 3：合成到真实的域差（Sim-to-Real Gap）**

OmniScene 是合成数据，真实相机的镜头光晕、雨天模糊、ISO 噪声都不在训练分布内。在目标域采集 500 帧真实数据做 fine-tuning，通常能带来 10-20% 的 AbsRel 改善。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 自动驾驶环视系统（混合相机） | 纯针孔单目/双目（用 Depth Pro 更简单） |
| 机器人鱼眼导航相机 | 动态物体密集场景（无时序建模） |
| 需要实时（>30 FPS）的边缘部署 | 工业精密检测（需要 < 1mm 精度） |
| 资源受限的嵌入式平台 | 相机标定不精确（误差 > 2 pixel） |
| 多相机需要统一的度量深度尺度 | 需要自由视角 360° 深度图 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| Depth Pro | 针孔精度高，零样本泛化好 | 完全不支持鱼眼 | 单相机针孔 |
| UniDepth | 统一框架，多相机类型 | 对鱼眼支持弱，速度慢 | 通用精度优先 |
| 去畸变流水线 | 可复用现有针孔模型 | 信息损失，误差叠加 | 无法更换模型时 |
| **X-Lens** | 轻量、异构、实时 | 合成训练数据，真实泛化存疑 | 自动驾驶异构相机系统 |

## 我的观点

X-Lens 的技术路线是对的：**不要强行统一坐标系，而是让模型学会理解不同的投影几何**。Jacobian 偏置这个设计很优雅——只用一个额外的 `nn.Linear(1, num_heads)` 就把复杂的几何先验注入到了注意力机制中，参数开销几乎为零，但几何意义很清晰。

有两个值得关注的开放问题：

**真实泛化**：OmniScene 是合成的，现实世界的光照变化、镜头老化、恶劣天气都是未知量。在 nuScenes 或真实车载数据上的测评，是检验这个方法落地价值的真正标准。

**动态场景**：X-Lens 是纯单帧前向推理，对运动物体没有显式建模。自动驾驶中移动的车辆和行人会导致系统性深度误差——这是这一类单帧方法的共同短板，需要结合光流或视频时序建模来解决。

0.04B 参数实现 SOTA 的异构深度估计，是扎实的工程成果。如果你的系统里同时有鱼眼和针孔相机，X-Lens 是目前最值得试的基线。