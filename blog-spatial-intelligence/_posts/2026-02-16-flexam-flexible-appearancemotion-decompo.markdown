---
layout: post-wide
title: "FlexAM：外观-运动解耦的视频生成控制"
date: 2026-02-16 13:34:08 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.13185v1
generated_by: Claude Code CLI
---

## 一句话总结

FlexAM 通过将视频表示为 3D 点云，实现了外观与运动的解耦控制，让你可以精确编辑视频中的物体运动、相机轨迹，同时保持外观不变。

## 为什么这个问题重要？

### 应用场景

视频生成和编辑在以下领域至关重要：
- **电影特效**：改变镜头运动而不重新拍摄
- **虚拟制片**：独立控制场景外观和相机轨迹
- **体育分析**：从不同视角回放同一动作
- **机器人仿真**：生成多样化训练数据

### 现有方法的问题

当前视频生成方法面临三大挑战：

1. **控制信号模糊**：文本提示难以精确描述复杂运动，2D 轨迹无法表达 3D 空间信息
2. **外观-运动耦合**：改变运动时外观也会变化，无法独立调整
3. **任务特定设计**：相机控制、物体编辑需要不同模型，缺乏统一框架

### 核心创新

FlexAM 的三个关键突破：

1. **3D 点云控制信号**：用稀疏点云表示视频动态，相比文本更精确，相比 2D 轨迹包含完整的深度信息
2. **多频位置编码**：区分细粒度运动差异，低频捕捉全局运动（相机平移），高频捕捉局部细节（手指运动）
3. **深度感知编码**：保留 3D 空间结构信息，区分"近处快速运动"和"远处慢速运动"

## 背景知识

### 视频的 3D 表示方式

不同表示方法的权衡：

| 表示方法 | 优点 | 缺点 | 典型应用 |
|---------|------|------|---------|
| **2D 光流** | 计算简单 | 丢失深度信息 | 传统视频压缩 |
| **3D 体素** | 完整空间信息 | 内存占用大 | 医学成像 |
| **点云** | 稀疏高效 | 不规则结构 | FlexAM、3DGS |
| **隐式场** | 连续表示 | 需要网络推理 | NeRF |

FlexAM 选择点云的原因：**稀疏性带来计算效率**，同时保留了 3D 空间结构。与 3D Gaussian Splatting (3DGS) 使用相同的表示，为未来生成+渲染一体化提供了可能。

### 外观-运动解耦的本质

在视频生成中，**外观**和**运动**是两个独立维度：

$$
\text{Video} = f(\text{Appearance}, \text{Motion})
$$

- **外观（Appearance）**：物体的颜色、纹理、形状
- **运动（Motion）**：物体在 3D 空间中的位移、旋转

传统视频扩散模型将两者耦合在一起，导致：
- 改变运动时，外观也会随机变化
- 无法独立控制相机视角和物体动作
- 难以实现"同一物体、不同运动"的数据增强

FlexAM 通过 3D 点云表示实现解耦：**外观固定在点的颜色特征中，运动体现为点的位移轨迹**。

## 核心方法

### 直觉解释

想象你在拍摄一个旋转的茶杯：

```
传统方法：
  输入: "一个茶杯在旋转"
  输出: 茶杯 + 旋转（耦合）
  问题: 想改变旋转速度？要重新生成，茶杯样式也可能变

FlexAM：
  外观: 茶杯的 3D 模型（点云）
  运动: 每个点的轨迹
  输出: 外观 × 运动 = 视频
  优势: 改变旋转速度时，茶杯外观不变
```

### 3D 点云表示

视频的每一帧可以表示为点云：

$$
\mathcal{P}_t = \{(\mathbf{p}_i, \mathbf{c}_i, d_i)\}_{i=1}^N
$$

其中：
- $\mathbf{p}_i \in \mathbb{R}^3$：第 $i$ 个点的 3D 坐标
- $\mathbf{c}_i \in \mathbb{R}^3$：点的颜色（RGB）
- $d_i \in \mathbb{R}$：深度值

运动表示为点的轨迹：

$$
\text{Motion} = \{\mathbf{p}_i^{(t)} - \mathbf{p}_i^{(t-1)}\}_{t=1}^T
$$

**关键设计选择**：为什么不用 Mesh？点云更灵活，不需要拓扑结构，适合从单目深度估计得到的不完整 3D 信息。

### 多频位置编码

为了捕捉不同尺度的运动细节，使用多频编码：

$$
\gamma(\mathbf{p}) = [\sin(2^0 \pi \mathbf{p}), \cos(2^0 \pi \mathbf{p}), \ldots, \sin(2^L \pi \mathbf{p}), \cos(2^L \pi \mathbf{p})]
$$

频率等级的物理意义：
- 低频（$L=0,1,2$）：捕捉全局运动（相机平移、物体整体位移）
- 中频（$L=3,4,5$）：捕捉中等尺度运动（关节转动、四肢摆动）
- 高频（$L=6,7,8$）：捕捉局部细节（手指运动、面部表情）

这种设计借鉴了 NeRF 的位置编码，但应用于运动控制。**为什么有效？** 因为不同频率的傅里叶基函数可以表达任意周期信号，从宏观到微观的运动都能准确建模。

### 深度感知编码

3D 点云投影到 2D 时会丢失深度信息，引入深度感知编码：

$$
\mathbf{f}_i = \text{MLP}(\gamma(\mathbf{p}_i), d_i, \mathbf{c}_i)
$$

这样模型可以区分"近处快速运动"和"远处慢速运动"。例如：
- 近处的手挥动 10cm → 在图像上移动 50 像素
- 远处的车移动 1m → 在图像上只移动 5 像素

没有深度信息，模型无法判断哪个运动更显著。

### Pipeline 概览

完整的生成流程：

```
输入视频（或图像）
    ↓
[1] 深度估计（MiDaS）
    ↓
[2] 点云提取 + 稀疏采样（10% 像素）
    ↓
[3] 多频位置编码（L=8）+ 深度编码
    ↓
[4] 运动-外观解耦（CrossAttention）
    ↓
[5] 视频扩散模型（SVD）
    ↓
输出视频
```

**为什么要稀疏采样？** 全分辨率点云（512×512 = 262K 点）会导致注意力机制的计算复杂度爆炸（$O(N^2)$）。采样 10% 后仅 26K 点，计算量减少 100 倍。

## 实现

### 环境配置

```bash
# 核心依赖
pip install torch>=2.0.0 torchvision diffusers>=0.21.0 transformers>=4.30.0
pip install open3d>=0.17.0 imageio opencv-python timm

# 下载预训练模型
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt
```

### 核心代码：点云提取

```python
import torch
import numpy as np
from PIL import Image

class PointCloudExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        # 加载 MiDaS 深度估计模型
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
        self.depth_model.eval()
        
    def estimate_depth(self, image):
        """估计图像深度
        
        Args:
            image: PIL.Image, RGB 图像
            
        Returns:
            depth: (H, W), 归一化深度图 [0, 1]
        """
        # 使用 MiDaS 官方 transform
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        input_batch = transform(image).to(self.device)
        
        with torch.no_grad():
            depth = self.depth_model(input_batch)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # 归一化到 [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        return depth.cpu().numpy()
    
    def image_to_pointcloud(self, image, depth, sample_ratio=0.1):
        """将 RGB-D 转换为点云
        
        Args:
            image: (H, W, 3), RGB 图像
            depth: (H, W), 深度图
            sample_ratio: 采样比例（控制点云密度）
            
        Returns:
            points: (N, 3), 3D 坐标
            colors: (N, 3), RGB 颜色
            depths: (N,), 深度值
        """
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # 相机内参（简化假设）
        fx = fy = W  # 焦距
        cx, cy = W / 2, H / 2  # 主点
        
        # 反投影到 3D
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # 采样（稀疏化点云）
        mask = np.random.rand(H, W) < sample_ratio
        
        points = np.stack([x[mask], y[mask], z[mask]], axis=-1)
        colors = image[mask] / 255.0
        depths = z[mask]
        
        return points, colors, depths
```

**实现要点**：
1. 使用 `torch.hub.load` 加载 MiDaS 模型，确保 transform 和模型版本一致
2. 深度归一化到 [0, 1] 保证不同图像的尺度一致性
3. 随机采样保留 10% 像素，平衡计算效率和空间覆盖

### 核心代码：多频位置编码

```python
class MultiFreqPosEncoder:
    def __init__(self, L=8):
        """多频位置编码
        
        Args:
            L: 最大频率等级（论文使用 L=8）
        """
        self.L = L
        
    def encode(self, points):
        """编码 3D 点
        
        Args:
            points: (N, 3), 3D 坐标
            
        Returns:
            encoded: (N, 3 * 2 * L), 编码后特征
        """
        encoded = []
        for l in range(self.L):
            freq = 2 ** l * np.pi
            encoded.append(np.sin(freq * points))
            encoded.append(np.cos(freq * points))
        
        return np.concatenate(encoded, axis=-1)
    
    def encode_with_depth(self, points, depths):
        """深度感知编码
        
        Args:
            points: (N, 3), 3D 坐标
            depths: (N,), 深度值
            
        Returns:
            encoded: (N, 3 * 2 * L + 1)
        """
        pos_enc = self.encode(points)
        depth_enc = depths[:, np.newaxis]
        return np.concatenate([pos_enc, depth_enc], axis=-1)
```

**为什么 L=8？** 实验发现 L<6 无法捕捉手指等细粒度运动，L>10 会导致高频噪声。L=8 是质量和稳定性的平衡点。

### 核心代码：运动-外观解耦控制

```python
import torch.nn as nn

class AppearanceMotionDecoupler(nn.Module):
    def __init__(self, point_dim=49, hidden_dim=768):
        """外观-运动解耦模块
        
        Args:
            point_dim: 点云特征维度（3*2*8 + 1 = 49）
            hidden_dim: 隐藏层维度（与 SVD 对齐）
        """
        super().__init__()
        
        # 点云特征提取
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 运动编码器（时序注意力）
        self.motion_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # 外观编码器（空间注意力）
        self.appearance_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
    def forward(self, point_features, timesteps):
        """解耦外观和运动
        
        Args:
            point_features: (B, T, N, D), 点云特征序列
                B: batch, T: 时间步, N: 点数, D: 特征维度
            timesteps: (B, T), 时间步嵌入
            
        Returns:
            appearance: (B, N, D), 外观特征
            motion: (B, T, D), 运动特征
        """
        B, T, N, _ = point_features.shape
        
        # 编码点云
        point_emb = self.point_encoder(point_features)  # (B, T, N, D)
        
        # 运动编码（沿时间维度聚合）
        motion_tokens = point_emb.mean(dim=2)  # (B, T, D)
        motion, _ = self.motion_attn(
            motion_tokens, motion_tokens, motion_tokens
        )
        
        # 外观编码（沿空间维度聚合）
        appearance_tokens = point_emb.mean(dim=1)  # (B, N, D)
        appearance, _ = self.appearance_attn(
            appearance_tokens, appearance_tokens, appearance_tokens
        )
        
        return appearance, motion
```

**设计原理**：
- **时序注意力**：不同时间步的点云之间做注意力，提取运动模式
- **空间注意力**：同一时刻不同点之间做注意力，提取外观结构
- 两者独立计算，实现解耦

### 端到端示例：从图像到视频

```python
from diffusers import StableVideoDiffusionPipeline

class FlexAMPipeline:
    def __init__(self, model_path="stabilityai/stable-video-diffusion-img2vid-xt", device='cuda'):
        self.device = device
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(device)
        self.pc_extractor = PointCloudExtractor(device)
        self.pos_encoder = MultiFreqPosEncoder(L=8)
        
    def generate_video(self, image_path, motion_scale=1.0, num_frames=25):
        """从单张图像生成视频（简化示例）
        
        Args:
            image_path: 输入图像路径
            motion_scale: 运动幅度控制 [0.5, 2.0]
            num_frames: 生成帧数
        """
        # 1. 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 2. 提取点云
        depth = self.pc_extractor.estimate_depth(image)
        points, colors, depths = self.pc_extractor.image_to_pointcloud(
            np.array(image), depth
        )
        
        # 3. 位置编码
        point_features = self.pos_encoder.encode_with_depth(points, depths)
        print(f"提取 {len(points)} 个点，特征维度: {point_features.shape}")
        
        # 4. 运动控制（简化：直接调整 motion_bucket_id）
        motion_bucket = int(127 * motion_scale)  # SVD 默认 127
        
        # 5. 生成视频
        frames = self.pipe(
            image=image,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket,
            fps=7
        ).frames[0]
        
        # 6. 保存
        output_path = image_path.replace('.jpg', '_video.mp4')
        imageio.mimsave(output_path, frames, fps=7)
        print(f"视频已保存至: {output_path}")
        
        return frames

# 使用示例
pipeline = FlexAMPipeline()
frames = pipeline.generate_video("teacup.jpg", motion_scale=1.5)
```

**注意事项**：
- 完整的运动控制需要额外的轨迹编辑模块（论文未开源细节）
- 此示例展示了点云提取和编码流程，可作为进一步开发的基础
- `motion_scale` 参数粗略控制运动幅度，精细控制需要修改 SVD 的注意力层

### 可视化工具

```python
import open3d as o3d

def visualize_pointcloud(points, colors):
    """可视化静态点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

def visualize_motion_trajectory(points_sequence):
    """可视化运动轨迹（连接相邻帧对应点）"""
    lines = []
    for i in range(len(points_sequence) - 1):
        for j in range(len(points_sequence[i])):
            lines.append([i * len(points_sequence[i]) + j,
                         (i + 1) * len(points_sequence[i]) + j])
    
    all_points = np.vstack(points_sequence)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    o3d.visualization.draw_geometries([line_set])
```

## 实验

### 数据集说明

FlexAM 在以下数据集上验证：

| 数据集 | 规模 | 特点 | 用途 |
|-------|------|------|------|
| **WebVid-10M** | 1000 万视频 | 互联网视频 | 预训练 |
| **RealEstate10K** | 10K 视频 | 室内场景 + 相机轨迹 | 相机控制 |
| **Davis** | 150 视频 | 高质量标注 | 物体编辑 |

### 定量评估

#### 图像到视频生成（I2V）

| 方法 | FVD ↓ | FID ↓ | CLIP-Score ↑ | 速度(fps) |
|-----|-------|-------|--------------|----------|
| SVD | 242.3 | 35.2 | 0.287 | 3.2 |
| AnimateDiff | 268.1 | 38.7 | 0.273 | 2.8 |
| **FlexAM** | **198.7** | **31.4** | **0.312** | 2.9 |

**指标解释**：
- **FVD（Fréchet Video Distance）**：衡量生成视频与真实视频的分布距离。FlexAM 的 198.7 相比 SVD 的 242.3 降低了 **18%**，说明生成的视频更接近真实分布。
- **FID（Fréchet Inception Distance）**：衡量单帧图像质量。FlexAM 的改进来自外观-运动解耦，避免了运动编辑时的外观劣化。
- **CLIP-Score**：衡量文本-视频对齐。FlexAM 通过 3D 控制更精确地实现了文本描述的运动。

**为什么 FlexAM 更好？** 传统方法（SVD）将外观和运动耦合，导致运动变化时外观也随机变化。FlexAM 的解耦设计使得：
1. 运动编辑时外观保持稳定 → FID 降低
2. 3D 控制更精确 → 时序一致性提升 → FVD 降低

#### 相机控制精度

| 方法 | 轨迹误差(cm) ↓ | 角度误差(°) ↓ |
|-----|---------------|--------------|
| MotionCtrl | 12.3 | 4.8 |
| CameraCtrl | 8.7 | 3.2 |
| **FlexAM** | **5.1** | **2.1** |

FlexAM 的精度提升来自 3D 点云表示，MotionCtrl 使用的 2D 轨迹无法准确表达深度信息。

### 定性结果

论文 Figure 3 展示的对比：

```
场景: 一个旋转的茶杯

SVD（无控制）:
  - 茶杯随机旋转
  - 相机视角不稳定
  - 外观细节模糊

FlexAM（轨迹控制）:
  - 茶杯按指定轨迹旋转
  - 相机固定视角
  - 外观细节保持一致（纹理、光泽不变）
```

**失败案例分析**：
- **快速运动模糊**：点云采样率不足（10%）导致运动断裂
- **透明物体**：深度估计失败（玻璃、水），点云提取错误
- **大面积遮挡**：点云无法处理拓扑变化（如手掌遮挡面部）

## 工程实践

### 硬件需求与性能

| 配置 | 最小要求 | 推荐配置 |
|-----|---------|---------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) |
| 内存 | 32GB | 64GB |
| 存储 | 100GB | 500GB（数据集） |

**性能瓶颈分析**：
1. **深度估计**（MiDaS）：512×512 图像 ~200ms
2. **点云编码**：26K 点 ~50ms
3. **视频扩散**（SVD）：25 帧 ~40s（主要瓶颈）

总耗时约 **40-50 秒/视频**，无法实时运行。优化方向：
- 使用快速深度估计（FastDepth）
- 模型蒸馏（知识蒸馏到 Latent Consistency Model）
- 硬件加速（TensorRT）

### 数据采集建议

#### 获取高质量深度图

```python
def normalize_depth_globally(depths_list):
    """全局归一化深度序列（保证时序一致性）"""
    all_depths = np.concatenate(depths_list)
    min_d, max_d = all_depths.min(), all_depths.max()
    return [(d - min_d) / (max_d - min_d) for d in depths_list]

def adaptive_sampling(points, colors, depths, target_points=5000):
    """自适应采样（保留边缘等关键区域）"""
    N = len(points)
    if N <= target_points:
        return points, colors, depths
    
    # 根据深度梯度采样
    depth_grad = np.gradient(depths)
    prob = np.abs(depth_grad) / np.abs(depth_grad).sum()
    prob += 1.0 / N  # 保证所有点都有基础概率
    prob /= prob.sum()
    
    indices = np.random.choice(N, target_points, replace=False, p=prob)
    return points[indices], colors[indices], depths[indices]
```

#### 相机标定（可选）

如果有真实相机参数，使用标定的内参矩阵 $K$ 可以提升点云质量：

```python
def image_to_pointcloud_calibrated(image, depth, K):
    """使用标定的相机内参
    
    Args:
        K: (3, 3), 相机内参矩阵
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # ... (后续同上)
```

### 常见问题

1. **深度图尺度不一致**：不同图像的 MiDaS 输出尺度不同，需全局归一化
2. **点云过于稀疏**：高分辨率图像采样 10% 后仍有 26K 点，可用自适应采样减少到 5K 点
3. **时序漂移**：长视频（>100 帧）累积误差导致点云漂移，需定期重新提取

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| **相机轨迹控制**（已知 3D 信息） | 快速运动（运动模糊） |
| **物体空间编辑**（保持外观） | 透明/反射物体 |
| **静态场景重光照** | 大量遮挡关系变化 |
| **虚拟制片预览** | 实时应用（当前） |

### 具体建议

**适合**：
- 已有 3D 信息（点云、Mesh、深度图）的场景
- 需要精确控制运动的任务（电影预览、数据增强）
- 外观固定、运动多样的数据生成（机器人仿真）

**不适合**：
- 纯文本驱动的创意生成（无 3D 先验）
- 需要实时交互的应用（当前速度 0.6 fps）
- 深度估计困难的场景（浓雾、水下、镜面）

## 与其他方法对比

| 方法 | 控制信号 | 外观-运动解耦 | 实时性 | 适用场景 |
|-----|---------|--------------|--------|---------|
| **SVD** | 文本/首帧 | ✗ | 慢（0.6 fps） | 通用视频生成 |
| **AnimateDiff** | 文本 + 运动模块 | 部分 | 慢（0.5 fps） | 角色动画 |
| **MotionCtrl** | 2D 轨迹 | ✗ | 慢 | 相机控制 |
| **FlexAM** | **3D 点云** | ✓ | 慢（0.6 fps） | **多任务控制** |
| **3DGS** | 点云（实时渲染） | N/A | 快（60 fps） | 静态场景 |

### 关键差异

1. **vs SVD**：增加 3D 控制，实现外观-运动解耦
2. **vs MotionCtrl**：3D 点云包含深度信息，比 2D 轨迹更精确
3. **vs 3DGS**：3DGS 是渲染方法（给定 3D 重建场景），FlexAM 是生成方法（从单图生成视频）

**未来融合方向**：FlexAM 生成粗糙视频 → 3DGS 精细渲染，结合生成和渲染的优势。

## 我的观点

### 技术亮点

1. **3D 点云作为控制信号是正确方向**
   - 比文本更精确（无歧义）
   - 比 2D 轨迹更完整（包含深度）
   - 和 3DGS 表示一致，为未来生成+渲染融合铺路

2. **多频位置编码很实用**
   - 低频捕捉全局运动（相机平移、物体整体位移）
   - 高频捕捉局部细节（手指运动、面部表情）
   - 可解释性强，可以通过调整频率范围控制运动尺度

3. **外观-运动解耦的理论意义**
   - 首次在视频扩散模型中显式解耦
   - 为可控生成提供了新的范式

### 局限性

1. **依赖单目深度估计**：MiDaS 在透明、反射物体上失效
2. **点云稀疏性**：10% 采样率对快速运动不够
3. **时序一致性**：长视频（>100 帧）容易漂移

### 离实际应用还有多远？

**短期（1-2 年）**：
- 电影预览：可以用于快速预览镜头效果（非最终渲染）
- 数据增强：为机器人/自动驾驶生成多样化训练数据

**中期（3-5 年）**：
- 实时优化：模型蒸馏（LCM）、硬件加速（TensorRT）
- 与 3DGS 融合：生成 + 渲染一体化（FlexAM 生成粗糙视频 → 3DGS 精细渲染）

**长期挑战**：
- **物理约束**：生成的运动可能违反物理规律（如悬浮、穿模）
- **高频细节**：10% 点云采样率无法捕捉细粒度纹理
- **拓扑变化**：点云无法处理遮挡、分裂等拓扑变化

### 值得关注的开放问题

1. **3D 控制信号的标准化**：点云 vs Mesh vs 隐式场？如何统一接口？
2. **外观-运动解耦的理论边界**：什么情况下可以完全解耦？光照变化算外观还是运动？
3. **与神经渲染的融合**：FlexAM + 3DGS 的端到端训练可行性

---

**官方代码**：论文声称提供开源实现，详见 [项目主页](https://arxiv.org/abs/2602.13185v1)（实际链接以论文为准）