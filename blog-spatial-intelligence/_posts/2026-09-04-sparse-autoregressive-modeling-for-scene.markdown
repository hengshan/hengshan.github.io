---
layout: post-wide
title: '稀疏体素自回归：如何用扫描一半的房间"脑补"出完整场景'
date: 2026-09-04 08:03:07 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.03931v1
generated_by: Claude Code CLI
---

## 一句话总结

SPAR3S 把 3D 场景生成问题变成了"稀疏体素上的完形填空"：只对被占据的体素编码 latent token，用一个 masked autoregressive transformer 联合预测"哪里应该有内容"和"那里的内容是什么"，训练全程只用多视角图片做光度监督（可微 3D Gaussian Splatting），不需要任何 3D ground truth。

## 为什么这个问题重要？

想象一个室内机器人推门进入一个房间，只拍到了三四张照片——沙发的一角、半扇窗、走廊尽头的柜子。人类可以立刻脑补出"这大概率是个客厅，沙发对面应该有电视柜，墙角可能还有一盏灯"。但目前主流的 3D 视觉管线做不到这件事：

- **Feed-forward 重建方法**（如 LRM、pixelSplat、MVSplat 这类前馈式高斯泼溅重建）本质上是"看见什么建什么"，输入图像没覆盖到的区域，输出就是空洞或者拉伸出的鬼影。它们做的是几何插值，不是内容生成。
- **纯 3D 生成模型**（在体素或 NeRF 网格上做扩散）理论上可以补全未见区域，但稠密体素在分辨率提高时显存和计算量是立方级增长，很难扩展到房间尺度的场景；而且大规模、干净的 3D ground truth 数据集本身就稀缺——3D 世界不像图像那样能从互联网上无限爬取标注。

SPAR3S 的核心创新是绕开这两个坑：用**稀疏体素**只存有内容的地方（避免立方级开销），用**可微渲染的光度损失**代替 3D 监督（避免数据稀缺问题），再用**自回归 transformer** 把"生成结构"和"生成内容"统一到一个模型里。

## 背景知识

### 3D 表示方式怎么选？

| 表示 | 优点 | 缺点 |
|-----|------|------|
| 点云 | 轻量、直接对应观测 | 无拓扑、渲染需额外处理 |
| 稠密体素 | 结构规整，适合卷积/attention | 显存随分辨率立方增长 |
| 隐式场（NeRF） | 连续、分辨率无关 | 训练/渲染慢，编辑困难 |
| 3D Gaussian | 渲染极快、可微、显式 | 数量大时优化不稳定 |
| **稀疏体素** | 只存有效区域，省显存 | 需要额外结构（八叉树/哈希）管理占据关系 |

SPAR3S 选择稀疏体素 + latent token 的组合，本质上是把图像生成里"VQ-GAN 把图像压成离散 token，MaskGIT 在 token 网格上做双向/自回归生成"的思路搬到了 3D：**体素网格里的每个占据格子对应一个 latent token，未占据的格子直接不参与计算**。这样计算量只和场景中"有东西的地方"成正比，而不是和整个空间体积成正比。

### 为什么用 3D Gaussian Splatting 做监督信号

3DGS 的渲染是完全可微的（高斯参数 → 光栅化 → 像素颜色），这意味着你可以把"渲染出的图像和真实图像的差异"反传回 3D 表示本身，而不需要知道每个 3D 点的真实几何或颜色标签。这正是 SPAR3S 摆脱 3D ground truth 依赖的关键：latent 空间是通过"能不能渲染出正确的图片"这个代理任务学出来的，不是通过"这个体素的真实占据值是多少"这种直接监督学出来的。

## 核心方法

### 直觉解释

整个 pipeline 可以理解为三步：

```
稀疏输入视角 → 编码为稀疏体素 latent（部分占据） → 
掩码自回归 transformer 补全占据关系 + latent 值 → 
解码为 3D Gaussian → 渲染任意新视角
```

训练时的关键循环是：编码器把已知视角"往回投影"到体素网格上，产生一部分带 latent 的体素；transformer 学习"给定部分体素，预测剩余体素在哪、值是什么"；解码器把完整的体素 latent 转成高斯参数，用可微渲染和真实图片比较误差。

### 数学细节

设场景的稀疏体素集合为 $\mathcal{V} = \{v_i\}$，每个体素 $v_i$ 有一个占据标记 $o_i \in \{0,1\}$ 和一个 latent 编码 $z_i \in \mathbb{R}^d$（只有 $o_i=1$ 时 $z_i$ 才有意义）。给定观测视角集合 $\mathcal{I}_{\text{obs}}$ 反投影得到的部分体素集合 $\mathcal{V}_{\text{obs}} \subset \mathcal{V}$，场景补全等价于对未观测体素建模联合分布：

$$
p(o_i, z_i \mid \mathcal{V}_{\text{obs}}), \quad v_i \in \mathcal{V} \setminus \mathcal{V}_{\text{obs}}
$$

训练目标由两部分组成。第一部分是掩码自回归的 token 预测损失（类似 MaskGIT，随机掩掉一部分体素，让模型预测被掩掉的占据状态和 latent 值）：

$$
\mathcal{L}_{\text{tok}} = \mathbb{E}\left[ -\log p_\theta(o_i \mid \mathcal{V}_{\text{ctx}}) - \mathbb{1}[o_i=1]\log p_\theta(z_i \mid \mathcal{V}_{\text{ctx}}) \right]
$$

第二部分是光度重建损失，把解码后的高斯 $\{\mu_k, \Sigma_k, c_k, \alpha_k\} = D(z_i)$ 渲染到新视角相机 $\pi$ 下：

$$
\mathcal{L}_{\text{photo}} = \sum_{\pi \in \mathcal{I}_{\text{train}}} \left\| R(\{\mu_k,\Sigma_k,c_k,\alpha_k\}, \pi) - I_\pi \right\|_1
$$

其中 $R(\cdot)$ 是可微高斯光栅化渲染器。最终损失是两者的加权和，这使得 latent 空间既服从 transformer 容易建模的分布结构，又保真于真实的视觉内容。

### Pipeline 概览

```
多视角图像 → 图像特征提取（ViT/CNN）
           → 反投影到稀疏体素网格（只保留可见占据体素）
           → 稀疏体素编码器 → 部分 latent token 序列
           → Masked Autoregressive Transformer（预测占据 + latent）
           → 完整 latent token 序列
           → 逐体素解码为局部 3D Gaussian
           → 3DGS 光栅化渲染 → 任意新视角图像
```

## 实现

以下代码是为理解算法骨架而写的**简化教学实现**，不是论文官方代码（本文未找到公开代码库链接）。

### 环境配置

```bash
# 核心依赖
pip install torch torchvision einops
# 真实 3DGS 渲染需要额外的可微光栅化算子，例如：
# pip install diff-gaussian-rasterization
```

### 稀疏体素结构

只存储被占据的体素坐标和索引，避免稠密网格的立方级开销。

```python
import torch

class SparseVoxelGrid:
    """用哈希表管理稀疏体素坐标 -> 索引的映射"""
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.coord_to_idx = {}   # (x, y, z) -> token 索引
        self.coords = []         # 索引 -> (x, y, z)

    def insert(self, coords: torch.Tensor):
        # coords: (N, 3) 整数体素坐标
        for c in coords.tolist():
            key = tuple(c)
            if key not in self.coord_to_idx:
                self.coord_to_idx[key] = len(self.coords)
                self.coords.append(key)

    def neighbor_mask(self, idx: int, radius: int = 1):
        # 返回某体素在网格中的空间邻居 token 索引，供 transformer 做局部注意力
        x, y, z = self.coords[idx]
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    key = (x + dx, y + dy, z + dz)
                    if key in self.coord_to_idx:
                        neighbors.append(self.coord_to_idx[key])
        return neighbors
```

### 从多视角特征到体素 latent（编码器）

把图像特征反投影到体素中心，聚合成每个占据体素的初始 latent。

```python
import torch.nn as nn
import torch.nn.functional as F

class MultiViewVoxelEncoder(nn.Module):
    """将多视角图像特征反投影并聚合为稀疏体素 latent"""
    def __init__(self, feat_dim=256, latent_dim=128):
        super().__init__()
        self.proj = nn.Linear(feat_dim, latent_dim)
        self.aggregate = nn.GRUCell(latent_dim, latent_dim)

    def forward(self, image_feats, voxel_centers, cams):
        # image_feats: list of (H, W, feat_dim) 每视角特征图
        # voxel_centers: (N, 3) 世界坐标
        # cams: 每视角的内外参，用于把体素中心投影到像素坐标
        latents = torch.zeros(voxel_centers.shape[0], self.proj.out_features,
                               device=voxel_centers.device)
        for feat, cam in zip(image_feats, cams):
            uv, valid = project_to_pixels(voxel_centers, cam)  # 用户自定义投影函数
            sampled = bilinear_sample(feat, uv)  # (N, feat_dim)，无效点为 0
            step = self.proj(sampled) * valid.unsqueeze(-1)
            latents = self.aggregate(step, latents)
        return latents  # (N, latent_dim) 只对应观测到的体素
```

`project_to_pixels` 和 `bilinear_sample` 是标准的相机投影和双线性采样，为了控制代码长度这里省略实现（工程细节和 pixelSplat/MVSplat 中的反投影模块类似）。

### 掩码自回归 Transformer：核心生成模块

这是整个方法的关键——联合建模"体素在哪"和"体素是什么"。

```python
class MaskedVoxelTransformer(nn.Module):
    """在稀疏体素 token 序列上做掩码自回归生成"""
    def __init__(self, latent_dim=128, n_layers=8, n_heads=8):
        super().__init__()
        self.pos_embed = nn.Linear(3, latent_dim)     # 体素坐标位置编码
        self.mask_token = nn.Parameter(torch.randn(latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, batch_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, n_layers)
        self.occ_head = nn.Linear(latent_dim, 1)       # 占据概率
        self.latent_head = nn.Linear(latent_dim, latent_dim)  # 预测 latent 值

    def forward(self, tokens, coords, observed_mask):
        # tokens: (N, D) 已知位置填真实 latent，未知位置填 mask_token
        # coords: (N, 3) 所有候选体素（包括待生成的空位）
        # observed_mask: (N,) bool，True 表示已观测（不参与预测 loss）
        x = tokens + self.pos_embed(coords.float())
        x[~observed_mask] = self.mask_token  # 用可学习 mask token 替换未观测位置
        h = self.blocks(x.unsqueeze(0)).squeeze(0)
        occ_logits = self.occ_head(h)
        pred_latent = self.latent_head(h)
        return occ_logits, pred_latent
```

推理阶段采用类似 MaskGIT 的**迭代解码**：每一步只保留置信度最高的一批预测结果并"钉住"，其余体素继续掩码，反复迭代直到全部体素被生成，这样比逐体素自回归快得多，同时保留了生成的空间一致性。

### 训练：光度损失驱动的端到端优化

```python
def train_step(encoder, transformer, decoder, renderer,
                images, cams, gt_images, gt_cams, voxel_grid):
    obs_coords = torch.tensor(voxel_grid.coords)
    obs_latents = encoder(extract_features(images), obs_coords, cams)

    # 随机采样候选空位体素（简化：从已知场景边界内均匀采样）
    empty_coords = sample_candidate_voxels(voxel_grid, n=obs_coords.shape[0] * 2)
    all_coords = torch.cat([obs_coords, empty_coords], dim=0)
    tokens = torch.cat([obs_latents, torch.zeros_like(obs_latents[:1]).expand(
        empty_coords.shape[0], -1)], dim=0)
    observed_mask = torch.cat([
        torch.ones(obs_coords.shape[0], dtype=torch.bool),
        torch.zeros(empty_coords.shape[0], dtype=torch.bool)])

    occ_logits, pred_latent = transformer(tokens, all_coords, observed_mask)

    gaussians = decoder(pred_latent, occ_logits, all_coords)  # -> 高斯参数
    rendered = renderer(gaussians, gt_cams)  # 可微 3DGS 渲染新视角
    photo_loss = F.l1_loss(rendered, gt_images)

    total_loss = photo_loss  # 训练全流程只用光度监督，无需 3D GT
    total_loss.backward()
    return total_loss.item()
```

注意这里没有对 `occ_logits` 单独加占据分类损失——这是刻意的：论文的核心思路是占据结构本身也通过渲染信号间接学习，只有在做纯粹的掩码 token 预训练阶段（重构已观测体素）才会用到显式的 occupancy 交叉熵。

## 实验

### 数据集说明

论文在两类数据上验证：

- **合成室内场景**：可以获得干净的多视角渲染和完整场景布局，用来验证补全质量的上限。
- **RealEstate10k**：真实房产视频数据集，视角轨迹自然、光照真实，但没有 3D ground truth，用来验证方法在真实数据上的泛化能力——这也呼应了它"不需要 3D 监督"的设计初衷。

### 定量与定性结果

论文报告在合成室内场景上，新视角渲染质量优于此前的补全/生成基线；在 RealEstate10k 上也验证了方法的适用性。由于这是一篇新发布的 arXiv 论文，具体的 PSNR/SSIM 数值和消融实验结果建议直接查阅原文（arXiv:2609.03931），本文不做数字上的复述，以免与论文实际数据出入。

从方法设计上可以预期的定性趋势是：观测视角越少、场景先验越强（如常见室内布局），生成质量越稳定；观测覆盖率越高，方法退化为接近纯重建，此时和 feed-forward 方法差距应该不大。

## 工程实践

### 实际部署考虑

- **实时性**：反投影编码 + 单次前馈 transformer 推理理论上可以做到交互式速度，但 masked autoregressive 的**迭代解码**（多轮掩码-预测-钉住）会引入额外的串行步骤，具体帧率取决于迭代轮数和体素数量，论文场景下大概率不是严格实时（不适合直接上机器人闭环控制，更适合离线/半在线的场景补全）。
- **硬件需求**：稀疏体素 + transformer attention 的组合，显存主要消耗在 token 序列长度上（占据体素数），场景越大越复杂，token 数量越多，attention 的 $O(N^2)$ 复杂度会成为瓶颈——这也是为什么"稀疏"是核心卖点而不是可有可无的优化。
- **内存占用**：相比稠密体素扩散模型，稀疏表示在大场景下的内存优势会非常明显，但代价是需要额外维护哈希/索引结构，工程实现复杂度更高。

### 数据采集建议

- 输入视角之间需要有一定的相机基线（baseline），太近的视角反投影后体素占据几乎重叠，等价于没有额外信息；太远则反投影误差增大。
- 由于监督信号来自光度损失，训练数据的**相机位姿精度**直接影响 latent 空间质量，位姿误差会直接污染生成结果，这一点在用 COLMAP 等 SfM 工具处理自采数据时要特别注意。

### 常见坑

1. **体素分辨率选择过高**：稀疏体素虽然省显存，但分辨率一旦拉高，token 序列长度暴涨，attention 计算量爆炸。→ 从低分辨率（如 32³ 或 64³ 有效范围）开始验证 pipeline，再逐步提高。
2. **掩码策略过于简单（纯随机）**：如果训练时只做均匀随机掩码，模型学不到"整块连续区域缺失"时该怎么补全（更接近真实的稀疏视角场景）。→ 训练时应模拟真实的视角遮挡模式做结构化掩码，而不仅仅是逐点随机。
3. **忽视光度损失的多视角一致性**：只用单视角监督容易让生成结果在几何上不一致（不同角度看到"违和"的内容）。→ 每步训练尽量用多个 novel view 同时算损失。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 视角稀疏、需要合理"脑补"未见区域（如房产漫游、AR 预览） | 需要严格几何精度的测量/重建任务（如工业质检） |
| 场景类型有较强先验（室内布局等） | 开放世界、结构高度不规则的场景 |
| 可离线/半在线处理，容忍多轮迭代解码延迟 | 需要硬实时闭环（机器人避障、SLAM 前端） |
| 训练数据只有图像、缺乏 3D 标注 | 已有高质量 3D 扫描数据，补全需求不强 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| NeRF | 高质量连续场重建 | 训练/渲染慢，无法补全未见区域 | 稠密视角静态场景重建 |
| 3DGS | 渲染极快、可编辑 | 同样只能重建可见内容 | 实时新视角合成 |
| Feed-forward LRM 类方法 | 单次前向、速度快 | 输入外内容无法生成，遮挡区域伪影明显 | 视角相对稠密的快速重建 |
| 稠密体素 3D 扩散 | 可以做真正的生成补全 | 显存/计算随分辨率立方增长，训练慢 | 小场景/物体级生成 |
| SPAR3S | 稀疏表示 + 无需 3D 监督的生成补全 | 迭代解码非严格实时，依赖位姿精度 | 稀疏视角房间级场景补全 |

## 我的观点

SPAR3S 这类工作代表了一个明确的趋势：**把图像生成领域已经验证过的"离散 token + 掩码自回归"范式，结合稀疏 3D 表示，迁移到场景级生成**上。这个方向的吸引力在于它同时解决了两个 3D 生成领域长期的痛点——3D 数据稀缺（用可微渲染代替 3D 监督）和计算量爆炸（用稀疏结构代替稠密体素）。

但我认为距离真正的产品化落地还有几个明显的差距：一是迭代解码的延迟问题，在需要交互式反馈的场景（比如 AR 应用里用户移动手机实时看到补全结果）还不够快；二是光度监督本身存在的多解性——同一组观测视角在理论上可以对应无数种"合理"的补全结果，模型学到的往往是训练分布里最常见的模式，遇到不寻常的房间布局容易生成"看起来对但其实错"的内容，这类幻觉问题目前在 3D 生成里比 2D 图像生成更难验证和纠错，因为普通用户很难像检查一张图片那样直觉地判断一个 3D 场景补全是否合理。

值得持续关注的开放问题是：如何把这类稀疏体素生成模型和真实的几何一致性约束（比如多视角深度、语义先验）结合得更紧，让"合理"和"正确"之间的差距进一步缩小。