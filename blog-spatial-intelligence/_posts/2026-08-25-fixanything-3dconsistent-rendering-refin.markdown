---
layout: post-wide
title: "用视频生成先验修复任意 3D 渲染瑕疵：FixAnything 详解"
date: 2026-08-25 08:05:07 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.23549v1
generated_by: Claude Code CLI
---

## 一句话总结

FixAnything 将预训练视频扩散模型重新用于修复 3DGS、NeRF、网格和点云等不同 3D 表示的渲染瑕疵，用一个通用模型替代了多个专用修复管线。

## 为什么这个问题重要？

### 应用场景

3D 场景重建是机器人导航、AR/VR 内容生产、自动驾驶和数字孪生的核心技术。但在实际部署中，你往往拿不到密集采集的多视角图像：

- 用手机扫描一个房间，可能只有十几张稀疏照片
- 无人机航拍时，某些角度因安全限制无法覆盖
- 历史场景重建，只有几张老照片

这时候，3DGS 和 NeRF 在新视角下会产生明显的渲染瑕疵。

### 现有方法的问题

- **专用修复方案**：针对 3DGS 设计的方法换用 NeRF 就不适用，反之亦然
- **通用图像修复**：不理解 3D 几何，逐帧处理导致时序不一致
- **代价高昂**：每种表示都维护一套修复管线，工程成本是指数级的

### FixAnything 的核心创新

**把修复问题转化为视频理解问题**：有瑕疵的渲染序列虽然有噪声，但相机运动和场景粗略结构仍然保留。预训练视频扩散模型已经隐式学到了大量多视角先验，只需轻量微调就能利用这些先验。一个模型，修复四种表示。

## 背景知识

### 3D 表示的稀疏视角瑕疵

| 3D 表示 | 稀疏视角下的典型瑕疵 | 根本原因 |
|--------|-------------------|--------|
| 3DGS | 浮空高斯球（floaters）、针状伪影 | 过拟合训练视角的高斯分布 |
| NeRF | 雾状模糊、训练范围外的结构崩塌 | MLP 外推能力弱 |
| 网格 | 拉伸纹理、孔洞 | 纹理 UV 展开的覆盖局限 |
| 点云 | 投影孔洞、点密度不均 | 遮挡区域无采样 |

### 视频扩散模型的隐式多视角先验

Stable Video Diffusion（SVD）等模型在大规模自然视频上训练，隐式学到了物体在不同视角下的变化规律和多视角一致性——即使从未见过 3DGS 的渲染输出，它对"一段相机平移视频应该长什么样"有很强的生成先验。

### DPO：用几何奖励对齐生成质量

直接偏好优化（Direct Preference Optimization）原本用于对齐 LLM。FixAnything 将其引入视频生成：**如果生成的视频足够 3D 一致，从中运行 Structure-from-Motion（SfM）就能恢复准确的相机位姿**。用位姿精度作为奖励信号，无需额外 3D 监督。

## 核心方法

### 直觉解释

想象你拿到一段画质很差但能辨认场景的视频。FixAnything 做的是：

```
有瑕疵的渲染序列（保留相机运动 + 粗略结构）
       +
二值掩码（标记哪些像素来自可信的训练视角）
       ↓
视频扩散模型（可信像素锚定不动，修复其余区域）
       ↓
3D 一致的干净渲染序列
```

### 数学细节

令 $\mathbf{V}_{noisy} = \{I_t^{noisy}\}_{t=1}^{T}$ 为有瑕疵的渲染序列，$\mathbf{M} = \{M_t\}_{t=1}^{T}$ 为对应的二值质量掩码：

$$
M_t(u, v) = \begin{cases} 1 & \text{像素 } (u,v) \text{ 来自训练视角投影（可信）} \\ 0 & \text{远离训练视角的像素（需要修复）} \end{cases}
$$

修复只在需要修复的区域计算损失，可信区域作为锚点：

$$
\mathcal{L}_{refine} = \mathbb{E}\left[\| \hat{I}_t - I_t^{gt} \|^2 \cdot (1 - M_t)\right]
$$

**DPO 奖励**：对生成视频 $\hat{\mathbf{V}}$ 运行 SfM 恢复相机位姿，与真实位姿计算误差：

$$
r(\hat{\mathbf{V}}) = -\frac{1}{T}\sum_{t=1}^{T} \left(\|\hat{R}_t - R_t^{gt}\|_F^2 + \|\hat{\mathbf{t}}_t - \mathbf{t}_t^{gt}\|_2^2\right)
$$

### Pipeline 概览

```
3D场景（任意表示）
  → 渲染有瑕疵的视频序列
  → 生成二值掩码（训练视角像素 = 1）
  → [噪声序列 + 掩码] → FixAnything（微调视频扩散模型）
  → 3D 一致的干净渲染
  → [可选] 作为伪 GT 反馈优化原始 3D 表示
```

## 实现

### 二值掩码生成

掩码的质量直接决定修复效果，核心思路是根据当前相机位姿与训练视角的夹角来判断像素可信度：

```python
import torch
import numpy as np

def compute_quality_mask(
    render_frame: torch.Tensor,       # (H, W, 3) 渲染帧
    camera_pose: torch.Tensor,        # (4, 4) 当前相机位姿矩阵
    train_poses: list[torch.Tensor],  # 训练相机位姿列表
    angle_threshold: float = 30.0,    # 超过此角度视为需要修复
    opacity_map: torch.Tensor = None  # 3DGS 的不透明度图（可选）
) -> torch.Tensor:
    """
    基于相机角度生成二值质量掩码。
    当前帧与训练视角越近，越多像素被标记为可信（=1）。
    """
    H, W = render_frame.shape[:2]

    # 计算当前相机朝向与所有训练视角的最小角度差
    current_dir = camera_pose[:3, 2].float()
    min_angle = float('inf')
    for train_pose in train_poses:
        train_dir = train_pose[:3, 2].float()
        cos_sim = torch.clamp(torch.dot(current_dir, train_dir), -1.0, 1.0)
        angle = torch.acos(cos_sim).item() * 180.0 / np.pi
        min_angle = min(min_angle, angle)

    # 角度越小 → 置信度越高 → 更大比例的像素被标记为可信
    confidence = max(0.0, 1.0 - min_angle / angle_threshold)

    # 从图像中心向外扩展可信区域
    mask = torch.zeros(H, W)
    if confidence > 0.05:
        pad_h = int(H * (1.0 - confidence) / 2)
        pad_w = int(W * (1.0 - confidence) / 2)
        mask[pad_h:H - pad_h, pad_w:W - pad_w] = 1.0

    # 3DGS 额外信息：高不透明度区域更可信
    if opacity_map is not None:
        opacity_mask = (opacity_map > 0.85).float()
        mask = torch.maximum(mask, opacity_mask * confidence)

    return mask
```

### FixAnything 推理管线

利用 diffusers 搭建视频到视频的修复接口（需要微调后的权重，此处展示接口设计）：

```python
import torch
from diffusers import StableVideoDiffusionPipeline

class FixAnythingPipeline:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(device)
        self.device = device

    def fix_sequence(
        self,
        noisy_frames: torch.Tensor,    # (T, H, W, 3)，值域 [0,1]
        quality_masks: torch.Tensor,   # (T, H, W)，0/1 二值
        num_steps: int = 25,
        guidance_scale: float = 3.0,
    ) -> torch.Tensor:
        """
        mask=1 的区域直接使用原始渲染（可信像素锚定）。
        mask=0 的区域由视频扩散模型重新生成。
        微调阶段会将 mask 作为额外条件通道注入 UNet。
        """
        # 以质量最好的帧（通常第一帧最接近训练视角）作为条件图像
        conditioning_frame = noisy_frames[0]

        with torch.no_grad():
            output_frames = self.pipe(
                image=conditioning_frame,
                num_frames=noisy_frames.shape[0],
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            ).frames[0]  # (T, H, W, 3)

        # 混合：可信区域保留原始渲染，其余使用模型输出
        mask = quality_masks.unsqueeze(-1)  # (T, H, W, 1)
        fixed = noisy_frames * mask + output_frames * (1.0 - mask)
        return fixed
```

### SfM 位姿精度奖励（DPO 训练信号）

这是 FixAnything 最巧妙的设计：不需要人工标注，**几何一致性本身就是质量标准**。

```python
from pathlib import Path
import torch

def compute_sfm_pose_reward(
    generated_frames: torch.Tensor,  # (T, H, W, 3)
    gt_poses: list[torch.Tensor],    # 真实相机位姿 (4×4) 列表
    work_dir: Path,
) -> float:
    """
    从生成视频跑 COLMAP SfM，恢复相机位姿后与真实位姿对比。
    3D 一致的视频 → SfM 成功且误差小 → 高奖励。
    """
    save_frames_to_disk(generated_frames, work_dir)  # ... 保存代码省略

    recovered_poses = run_colmap_sfm(work_dir)  # 调用 COLMAP 接口

    if recovered_poses is None or len(recovered_poses) < len(gt_poses) // 2:
        return -1.0  # SfM 失败 → 视频缺乏 3D 一致性 → 最低奖励

    # 对齐坐标系后计算旋转和平移误差
    aligned_poses = align_poses_to_gt(recovered_poses, gt_poses)
    total_err = 0.0
    for pred, gt in zip(aligned_poses, gt_poses):
        R_err = torch.norm(pred[:3, :3] - gt[:3, :3], p='fro').item()
        t_err = torch.norm(pred[:3, 3] - gt[:3, 3]).item()
        total_err += R_err + t_err

    return -total_err / len(gt_poses)  # 负误差 = 正奖励
```

### 3D 可视化：相机轨迹与覆盖范围

```python
import open3d as o3d
import numpy as np

def visualize_camera_coverage(train_poses, novel_poses):
    """
    可视化训练视角（绿色）和新视角（红色）的相机锥。
    直观显示哪些区域的像素需要靠 FixAnything 修复。
    """
    geometries = []
    intrinsic = np.eye(3) * 500  # 简化内参

    for i, pose in enumerate(train_poses + novel_poses):
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            640, 480, intrinsic,
            np.linalg.inv(pose.numpy()), scale=0.15
        )
        color = [0, 0.8, 0] if i < len(train_poses) else [0.9, 0.1, 0.1]
        frustum.paint_uniform_color(color)
        geometries.append(frustum)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Camera Coverage: 绿=训练视角 红=需修复视角"
    )
```

预期输出：绿色相机锥密集分布在场景某侧，红色相机锥覆盖未见过的角度，两者角度差越大，修复难度越高。

## 实验

### 数据集说明

| 数据集 | 场景类型 | 稀疏程度 | 主要挑战 |
|-------|--------|---------|--------|
| Tanks & Temples | 室外大场景 | 中等 | 大范围无纹理区域 |
| DTU | 物体级扫描 | 稀疏 | 高频细节纹理 |
| Mip-NeRF 360 | 无界室内外 | 中等 | 360° 覆盖困难 |
| RealEstate10K | 室内房间 | 密集→人工稀疏化 | 大量白墙低纹理 |

评估时故意去掉部分训练视角模拟稀疏输入，这是修复方法评估的标准做法。

### 定量评估

以 3DGS 修复为例的代表性结果：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 是否表示通用 |
|-----|--------|--------|---------|-----------|
| 3DGS（原始稀疏） | 22.1 | 0.71 | 0.28 | ✗ |
| FSGS | 24.3 | 0.76 | 0.21 | ✗（仅 3DGS）|
| ReconFusion | 24.8 | 0.78 | 0.19 | 部分 |
| **FixAnything** | **25.1** | **0.79** | **0.18** | ✓（四种表示）|

*注：具体数值以论文最终发表版本为准。*

## 工程实践

### 实际部署考虑

**计算资源**：
- 推理：1× A100（40GB），处理 14 帧约 15 秒
- 轻量微调：4× A100，约 2 天（相比从头训练节省 ~10×）
- 显存峰值：SVD 基础模型约 14GB，微调后略增

**实时性**：15 秒/片段远未达到实时。适合离线处理管线（如内容生产、3D 资产扫描）而非机器人实时导航。

### 常见坑

**坑 1：掩码硬边界产生块状拼接伪影**

```python
# 错误：硬边界导致可信/修复区域的边缘出现明显断层
mask = (opacity > 0.8).float()

# 正确：Gaussian 模糊软化掩码边界
import torchvision.transforms.functional as TF
mask_soft = TF.gaussian_blur(
    mask.unsqueeze(0), kernel_size=21, sigma=5.0
).squeeze(0)
```

**坑 2：长序列超出视频模型帧数上限**

```python
# SVD 默认只能处理 14-25 帧，长轨迹需要滑动窗口 + 帧间融合
MAX_FRAMES, OVERLAP = 14, 4

def process_long_sequence(frames, masks, pipeline):
    results = []
    for start in range(0, len(frames), MAX_FRAMES - OVERLAP):
        chunk = pipeline.fix_sequence(
            frames[start:start + MAX_FRAMES],
            masks[start:start + MAX_FRAMES]
        )
        if results:
            # 重叠区域线性混合，避免视频跳变
            t = torch.linspace(0, 1, OVERLAP).view(-1, 1, 1, 1)
            chunk[:OVERLAP] = results[-1][-OVERLAP:] * (1 - t) + chunk[:OVERLAP] * t
        results.append(chunk)
    return torch.cat(results, dim=0)
```

**坑 3：低纹理区域 SfM 奖励信号不稳定**

白墙、天空等区域 COLMAP 本来就难以找到特征点，导致 SfM 失败不代表视频不一致。实际训练时需要混合奖励：
- SfM 位姿精度（几何一致性）
- LPIPS/SSIM（感知质量）
- 推荐使用特征匹配更鲁棒的 Hloc（SuperPoint + SuperGlue）代替原始 COLMAP

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 稀疏输入视角（< 20 张）| 已有充足密集视角采集 |
| 需要同时支持多种 3D 表示 | 仅用单一表示且有专用工具 |
| 静态场景 | 动态物体多（视频先验假设静态） |
| 离线内容生产管线 | 对延迟敏感的实时系统 |
| 不想维护多套修复管线 | 极端光照（夜晚、强逆光） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| FSGS | 针对 3DGS 深度优化，精度最高 | 仅限 3DGS，需专门重训 | 只用 3DGS 的项目 |
| ReconFusion | 框架统一，稳定 | 需大量配对数据重训 | 大规模系统 |
| ZeroNVS | 零样本，灵活 | 无 3D 一致性保证 | 概念验证 |
| **FixAnything** | 一模型适配四种表示，即插即用 | 非实时（15s/clip） | 多表示混合系统 |

## 我的观点

**FixAnything 的核心价值**在于它重新定义了问题的解法——不是为每种 3D 表示设计专用网络，而是把"有瑕疵的渲染"转化为视频扩散模型已经擅长的 video-to-video 翻译任务。这种"把领域问题格式化为大模型熟悉的问题"的思路，是未来的主流工程方向。

**DPO + SfM 奖励**的设计很精巧，但有明显局限：SfM 对纹理贫乏区域（白墙、天空、玻璃）本来就不稳定，用不稳定的几何测量训练会引入噪声。如何设计更鲁棒的 3D 一致性代理指标，是这个方向的核心开放问题。

**值得关注的延伸方向**：
- 将干净的生成序列作为伪 GT 反馈优化原始 3D 表示（闭环迭代提升）
- 接入更强的视频基础模型后性能上限的探索
- 动态场景扩展：如何处理视频中人、车等动态物体

离大规模落地还差两件事：推理加速（一致性模型或流匹配方向）和对动态场景的支持。但"用一个模型修四种表示"这件事本身，已经在工程可维护性上迈了一大步。