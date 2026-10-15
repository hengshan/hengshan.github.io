---
layout: post-wide
title: '从"生成再压缩"到端到端编码：MFGS 如何把 3D 高斯的传输码率砍掉 99.8%'
date: 2026-10-15 12:03:43 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2610.20590v1
generated_by: Claude Code CLI
---

## 一句话总结

MFGS 不压缩显式的 3D 高斯参数，而是直接对多视角图像提取的中间特征做端到端码率-失真编码，让接收端从解码出的紧凑特征中预测高斯属性。相比传统"先生成完整高斯、再分别压缩几何和外观"的流程，MFGS 在 RE10K、ACID、DL3DV 三个数据集上实现了 99.76%–99.86% 的 BD-rate 码率节省，同时计算量降低 52%–55%，端到端速度提升约 12 倍。

## 为什么这个问题重要？

Feed-forward 3D Gaussian Splatting（如 pixelSplat、MVSplat、DepthSplat 等）已经可以从 2~10 张稀疏视角图像一次前向推理出完整的高斯场景表示，不再需要像原始 3DGS 那样对每个场景做几分钟到几十分钟的逐场景优化。这让 3D 场景的"即拍即建"成为可能，对云渲染、AR/VR 远程协作、机器人临场感等场景非常关键——想象一台机器人拍几张照片，把场景"发送"给远端的操作员实时渲染查看。

但"发送"这个动作本身就是问题：一个典型场景可能包含几十万到上百万个高斯点，每个点有位置（3）、旋转四元数（4）、各向异性缩放（3）、不透明度（1）、球谐系数颜色（通常几十维）——总参数量远超一张压缩后的图像。现有做法是先把这些高斯"生成出来"，再对几何和外观分别设计 codec 去压缩，这个"生成后压缩"（generate-then-compress）的流程存在三个问题：

1. 很多几何压缩操作（八叉树构建、KNN 排序）不可导，整条 pipeline 无法端到端联合优化；
2. 几何和外观被当成独立信号分别压缩，丢失了两者之间共享的结构冗余（它们本质上都来自同一组输入图像）；
3. 码率-失真性能被每个属性 codec 的独立设计上限锁死。

MFGS 的核心思路是把编码这一步"往前挪"：不压缩已经生成好的显式高斯，而是压缩生成高斯之前的多视角特征本身，让整个链路从特征提取到最终渲染都可导，可以用标准的 rate-distortion 训练目标端到端优化。

## 背景知识

**3D 表示方式对比**：点云稀疏、不能直接高质量渲染；NeRF 用 MLP 权重隐式编码场景，渲染质量高但速度慢、权重本身也不好流式传输；3DGS 显式参数渲染快但参数量大，直接传输代价高；MFGS 传输的既不是点云也不是最终的高斯参数，而是一组中间特征图——接收端有一个解码网络，根据这组特征"重建出"需要渲染的高斯。

**Feed-forward 3DGS vs 逐场景优化 3DGS**：前者用预训练网络一次前向推理（几十毫秒），后者需要针对每个场景跑梯度下降优化（分钟级）。MFGS 建立在前者之上。

**码率-失真编码基础**：标准有损编码框架是编码器 → 量化 → 熵编码 → 比特流 → 熵解码 → 解码器，训练目标是在码率 $R$ 和失真 $D$ 之间做权衡，这正是图像/视频压缩（如 JPEG、学习式编码 compressai）里的经典范式，MFGS 把它第一次系统性地搬到了 3D 高斯场景表示上。

## 核心方法

### 直觉解释

传统做法像是：先把看到的几张照片在脑子里"完整还原"成一个 3D 模型，再把这个模型压缩打包发出去。MFGS 的做法是：直接压缩"看完这几张照片后脑海里留下的印象特征"，接收端拿到这个压缩后的印象特征，自己"想象"出渲染需要的 3D 高斯——不需要先在发送端生成完整模型再压缩。

### 数学细节

编码器 $E$ 把多视角图像 $\{I_1, \dots, I_N\}$ 映射为紧凑特征 $F$，量化后得到 $\hat{F} = Q(F)$，熵编码器估计其码率。解码器 $D$ 从 $\hat{F}$ 直接预测每个高斯的属性集合 $G = \{\mu, q, s, \alpha, c\}$（中心、旋转、缩放、不透明度、颜色），再经标准 3DGS 光栅化器渲染出图像 $\hat{I}$。整体训练目标是：

$$
\mathcal{L} = \mathcal{D}(I, \hat{I}) + \lambda \cdot R(\hat{F})
$$

其中 $\mathcal{D}$ 是渲染失真（通常是 L1/SSIM 混合），$R(\hat{F}) \approx -\log_2 p(\hat{F})$ 是熵编码器估计的比特数，$\lambda$ 控制码率与质量的权衡。关键在于从图像到渲染结果的整条链路都可导，梯度可以一路回传到特征编码器，使编码器学会提炼出"既能省码率又能准确重建场景"的特征，而不是事后对一个已经固定下来的几何/外观表示做压缩。

为了支持不同的渲染预算（比如带宽紧张时少用点高斯），论文还给每个高斯预测一个重要性分数，接收端可以从同一份比特流里只取 top-K 个高斯渲染：

$$
G_{\text{keep}} = \text{TopK}(G, \text{score}, K)
$$

这样同一次编码可以服务于不同算力/带宽的终端，而不需要为每个预算重新编码。

### Pipeline 概览

```
多视角图像 → 多视角特征提取与融合 → 特征量化 + 熵编码 → 比特流
         → 熵解码 → 特征→高斯属性解码 → 重要性引导的高斯选择 → 3DGS 光栅化 → 渲染图像
```

## 实现

以下代码是对 MFGS 核心思路的最小化复现，重点展示"特征编码代替参数编码"这一设计，省略了真正的跨视角 epipolar attention、CDF 查表熵编码等工程细节。

### 环境配置

```bash
pip install torch open3d numpy
# 真实渲染需要 3DGS 官方光栅化器，例如 gsplat 或 diff-gaussian-rasterization
```

### 编码端：特征提取 + 熵瓶颈

```python
import torch
import torch.nn as nn

class MultiViewFeatureEncoder(nn.Module):
    """从N个视角图像提取并融合为紧凑的场景特征（省略真实的跨视角注意力机制）"""
    def __init__(self, in_ch=3, feat_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, feat_dim, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fuse = nn.Conv2d(feat_dim, feat_dim, 3, padding=1)

    def forward(self, imgs):
        # imgs: [B, N, 3, H, W]，逐视角提特征后跨视角融合
        B, N, C, H, W = imgs.shape
        feats = self.backbone(imgs.view(B * N, C, H, W))
        feats = feats.view(B, N, feats.shape[1], feats.shape[2], feats.shape[3])
        fused = feats.mean(dim=1)  # 用均值池化代替论文中的注意力融合
        return self.fuse(fused)  # [B, feat_dim, H/4, W/4]


class EntropyBottleneck(nn.Module):
    """简化的量化 + 熵编码代理：训练用加性噪声近似量化，推理用真实四舍五入"""
    def __init__(self, feat_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, feat_dim, 1, 1))

    def forward(self, x, training=True):
        if training:
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            x_hat = x + noise  # 可导的量化近似
        else:
            x_hat = torch.round(x)
        # 真实实现用学习到的 CDF 查表做熵编码，这里用拉普拉斯近似估计码率
        rate = torch.log2(1 + (x_hat.abs() / self.scale)).sum(dim=[1, 2, 3])
        return x_hat, rate
```

### 解码端：特征到高斯属性 + 预算自适应选择

```python
class FeatureToGaussianDecoder(nn.Module):
    """从解码后的特征图预测每个像素对应的 pixel-aligned 高斯属性"""
    def __init__(self, feat_dim=64):
        super().__init__()
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # 深度偏移(1) + 旋转四元数(4) + 缩放(3) + 不透明度(1) + RGB(3) + 重要性分数(1)
        self.head = nn.Conv2d(feat_dim, 1 + 4 + 3 + 1 + 3 + 1, 3, padding=1)

    def forward(self, feat, ray_origins, ray_dirs):
        feat = self.up(feat)
        out = self.head(feat)
        depth_offset, quat, scale, opacity, rgb, importance = torch.split(
            out, [1, 4, 3, 1, 3, 1], dim=1)
        mu = ray_origins + ray_dirs * torch.sigmoid(depth_offset)  # 沿射线确定高斯中心
        return {
            'mu': mu, 'quat': torch.nn.functional.normalize(quat, dim=1),
            'scale': torch.exp(scale), 'opacity': torch.sigmoid(opacity),
            'rgb': torch.sigmoid(rgb), 'importance': importance,
        }


def select_by_budget(gaussians, num_keep):
    """重要性引导的高斯选择：同一比特流按不同渲染预算截取不同数量的高斯"""
    flat_imp = gaussians['importance'].flatten(1)
    _, idx = torch.topk(flat_imp, k=num_keep, dim=1)
    selected = {k: torch.gather(v.flatten(2).transpose(1, 2), 1,
                idx.unsqueeze(-1).expand(-1, -1, v.shape[1]))
                for k, v in gaussians.items() if k != 'importance'}
    return selected
```

### 端到端训练目标

```python
def train_step(encoder, bottleneck, decoder, imgs, target, ray_o, ray_d, lambda_rd=0.01):
    feat = encoder(imgs)
    feat_hat, rate = bottleneck(feat, training=True)
    gaussians = decoder(feat_hat, ray_o, ray_d)
    rendered = gaussian_splatting_render(gaussians, camera=target['camera'])  # 调用官方3DGS光栅化器
    distortion = torch.nn.functional.mse_loss(rendered, target['rgb'])
    return distortion + lambda_rd * rate.mean()
```

这里 `gaussian_splatting_render` 代表标准的 3DGS 可导光栅化器（例如 `gsplat`），没有重新实现——这部分已经是成熟的开源组件，MFGS 的贡献在于它前面的编码-解码环节。

### 3D 可视化

```python
import open3d as o3d

def visualize_gaussians(gaussians):
    centers = gaussians['mu'].detach().cpu().numpy().reshape(-1, 3)
    colors = gaussians['rgb'].detach().cpu().numpy().reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
```

实际效果应该是：随着 $\lambda$ 增大（更看重码率），点云会变得稀疏、颜色过渡更粗糙；随着保留高斯数量 `num_keep` 减小，细节（如边缘、薄结构）先消失，大面积的颜色块保留得更久——这符合重要性分数倾向于保留"信息密度高"的高斯这一设计目标。

## 实验

### 数据集说明

- **RE10K（RealEstate10K）**：室内房产场景视频，视角密集且视角基线变化丰富，是 feed-forward 3DGS 领域的标准训练/评测集；
- **ACID**：无人机航拍的自然景观，用于验证室外大尺度场景的泛化能力；
- **DL3DV**：大规模真实场景数据集，覆盖室内外多种场景类型，场景数量级更大，常用于检验方法的鲁棒性。

三者都是公开可获取的多视角图像+相机位姿数据集，获取难度不高，但训练 feed-forward 模型本身对算力要求不低。

### 定量评估

论文报告的是跨三个数据集的汇总结果，原文没有给出逐数据集的细分数字，这里按摘要中的区间原样呈现：

| 对比对象 | BD-rate 码率节省 | GMACs 降低 | 端到端加速 |
|---|---|---|---|
| vs. generate-then-compress 基线 | 99.76%–99.86% | 52%–55% | ~12x |

需要注意，这个"99%+节省"是相对于论文设定的"生成再压缩"基线，不是相对于某个理论最优编码的绝对数字——在理解这个数字时要带上这个参照系。

### 定性结果

论文中应给出不同码率下的渲染对比图（原始 3DGS 质量 vs 低码率下 MFGS 重建质量），以及不同 `num_keep` 下渲染预算-质量曲线。典型模式是：MFGS 在极低码率区间相比传统 codec 能保持更连贯的几何结构（因为几何和外观共享特征，不会出现"几何量化导致颜色贴图错位"之类的伪影），但在追求绝对峰值质量（高码率、不计算资源）时，直接传输原始高斯仍可能略胜一筹——压缩总是有代价的，只是 MFGS 把这个代价压得很低。

## 工程实践（重要！）

### 实际部署考虑

- **实时性**：论文强调的是单次前向推理+解码的速度（约 12 倍于基线），但实际流式场景下还要考虑网络延迟、熵编解码的硬件实现（CPU 熵编码通常是瓶颈，需要专门优化或用 GPU 并行熬编码）；
- **硬件需求**：训练阶段需要能跑 feed-forward 3DGS 的 GPU（通常是 24GB+ 显存级别，因为要同时渲染+反传）；推理/解码端如果部署在边缘设备（AR 眼镜等），需要评估解码网络+光栅化器能否跑在移动 GPU 上；
- **内存占用**：重要性引导选择机制天然提供了一个"内存-质量"旋钮，大场景可以直接调小 `num_keep` 而不用重新编码，这比传统 codec 要重新调码率参数方便得多。

### 数据采集建议

Feed-forward 方法对训练数据的视角分布比较敏感：训练时视角数量、基线范围（相机间距）要覆盖推理时实际会遇到的情况，否则泛化会明显下降。采集时尽量保证场景静态、光照稳定，多视角之间有足够重叠但又不要退化成单视角。

### 常见坑

1. **训练-推理量化不一致**：训练时用加性均匀噪声近似量化，推理时用真实的 `round`，如果熵模型没有针对真实量化边界校准，会导致实际码率与训练时估计的码率有偏差 → 建议在验证阶段额外跑一次真实量化+真实熵编码来评估实际比特数，不要只信训练时的噪声近似估计。
2. **重要性分数训练不稳定**：如果不显式约束不同保留比例下的渲染质量，模型可能"偷懒"——把重要性分数学成近乎均匀分布，导致截断时质量骤降 → 训练时应随机采样不同的 `num_keep` 比例，强制模型学会把真正重要的高斯排到前面。
3. **视角数量泛化差**：训练时固定用固定数量的输入视角，推理时视角数变化（比如训练用4视角，部署时只有2视角）会导致融合模块（均值池化/注意力）性能骤降 → 训练时随机采样视角数量。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要把 3D 场景从一端传到另一端渲染（云渲染、远程协作） | 动态场景、视频流重建（当前方法面向静态场景） |
| 输入是稀疏多视角图像（2~10张），带宽/码率是瓶颈 | 密集多视角输入（几十张以上），直接逐场景优化3DGS可能质量更好 |
| 需要同一份数据服务不同算力终端（重要性分级渲染） | 单张图像的3D生成任务（这是重建+压缩，不是生成式任务） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| NeRF | 渲染质量高，表示本身较紧凑（MLP权重） | 渲染慢，难以流式传输和编辑 | 离线高质量重建 |
| 3DGS（逐场景优化） | 渲染快、质量高 | 需要场景特定优化，参数量大，直接传输代价高 | 单场景实时渲染，不涉及传输 |
| Feed-forward 3DGS（pixelSplat等） | 单次前向即可重建，无需逐场景优化 | 生成的显式高斯仍需额外codec压缩才能传输 | 快速重建，但传输仍是瓶颈 |
| MFGS | 端到端编码训练，码率极低，计算量更小、速度更快 | 牺牲部分峰值质量换压缩效率，依赖训练数据分布，目前只支持静态场景 | 需要把重建场景传到远端渲染的流式/协作场景 |

## 我的观点

这篇工作本质上是把图像/视频压缩领域早已验证过的结论——端到端学习的码率-失真联合优化远优于"先固定表示再手工设计codec"——搬到了 3D 高斯场景表示上。把压缩这一步从"显式参数之后"挪到"特征提取之后"，是个很自然但此前被忽视的设计选择，原因可能是过去大家默认 3DGS 的"生成"和"传输/存储"是两个独立阶段，各自优化。MFGS 证明把它们打通之后收益很大。

但距离真正产品化落地还有几个问题没解决：一是目前所有数字都是相对于论文自定义的"generate-then-compress"基线，业界还没有一个像 H.264 那样统一的 3DGS 比特流标准，不同框架之间不兼容；二是方法目前局限于静态场景，真正的远程协作/临场感场景往往涉及人物、物体的动态变化，这需要额外的时序建模；三是熵编解码在真实网络环境下的延迟和鲁棒性（丢包、抖动）还没有被充分讨论，这些是编码研究进入到流媒体系统时绕不开的工程问题。总的来说，这是一个方向正确、效果显著的工作，但"3D场景的标准化流式传输"这件事，可能还需要几年才能真正落地到产品里。