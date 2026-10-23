---
layout: post-wide
title: "数据稀缺场景下的天空图像辐照度估计：物理先验 + 双路 ViT 实战"
date: 2026-10-23 08:03:15 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2610.30940v1
generated_by: Claude Code CLI
---

## 一句话总结

这篇论文要解决的问题是：在只有**一个月**天空图像数据的情况下，如何训练出一个预测散射水平辐照度（DHI）的模型，效果能接近用**全年数据**训练的 SOTA Vision Transformer。它的做法不是换一个更大的模型，而是往输入端和监督信号里塞物理先验——地外辐照度、周期时间编码、GHI 中间预测、无监督云层分割 overlay——用领域知识换数据量。

## 背景：为什么需要这个方法

DHI（diffuse horizontal irradiance，散射水平辐照度）是光伏系统性能评估和发电预测的关键输入，但测量它需要专门的分光辐射计（pyranometer + shadow band 或分离式仪器），贵且维护成本高。于是有人想用便宜的鱼眼天空相机拍图像，训练 ViT 直接回归出 DHI，省掉仪器。

这条路径本身在数据充足时是成立的——用几年的高质量天空图像 + 对应的地面观测值训练 ViT，效果不错。但现实中大部分想部署这套系统的场景（新建电站、发展中地区、刚装相机的站点）恰恰**没有**多年历史数据，这就是论文标题里 "Data Scarce" 的含义。

论文的核心 insight 是：图像里包含的信息不是辐照度的唯一来源。太阳几何位置、季节、时间这些**可以直接计算、不需要任何观测仪器**的物理量，本身就携带了大量辐照度相关的先验信息。把这些先验显式注入模型，可以大幅降低模型对图像训练数据量的依赖。

诚实地说，这不是一个"新算法"论文，ViT 结构本身没有改动，创新点全部在输入特征工程和训练流程设计上。这类工作的价值在于证明了"领域知识 + 现成模型"比"堆数据 + 大模型"在小数据场景下更可靠，这个思路比具体数字更值得学。

## 算法原理

### 直觉解释

想象教一个刚入行的气象观测员估算辐照度。如果只给他看一张天空照片，他需要见过成千上万张照片才能建立直觉。但如果你先告诉他："现在是几月几号几点，太阳本该有多高，理论上晴空时辐照度是多少"，他的判断起点就已经对了大半，剩下的只需要看云层来修正这个理论值——这正是论文中"计算的大气参数"（地外辐照度 + 周期时间编码）扮演的角色。

接着，论文让模型先猜一个总量（GHI，总水平辐照度），再拿这个总量作为额外特征去细化散射部分（DHI）的预测——这类似于先估算"今天一共能挣多少钱"，再据此推算"其中有多少是小费"，比直接盲猜小费更稳。

最后，模型同时看原图和一张"圈出云在哪里"的叠加图（overlay），这张图是用传统图像处理方法（非深度学习）生成的，不需要标注数据，相当于免费给模型加了一个注意力提示。

### 数学背景

地外辐照度（extraterrestrial irradiance）完全由天文几何决定，不依赖任何测量：

$$
I_0 = I_{sc}\left(1 + 0.033\cos\left(\frac{360n}{365}\right)\right)\cos\theta_z
$$

其中 $I_{sc}$ 是太阳常数，$n$ 是一年中的第几天，$\theta_z$ 是太阳天顶角。这个值可以在任何地点任何时刻精确计算出来，不存在"数据稀缺"的问题。

时间信息如果直接用小时数（0-23）当特征，会在 23 点到 0 点产生一个巨大的数值跳变，模型会误以为深夜和白天在数值上"很远"。周期编码解决这个问题：

$$
t_{\sin} = \sin\left(\frac{2\pi h}{24}\right),\qquad t_{\cos} = \cos\left(\frac{2\pi h}{24}\right)
$$

这样 23 点和 0 点在特征空间里是相邻的，符合物理直觉。

### 与标准 ViT 的关系

骨架完全是标准 ViT：patch embedding + transformer encoder + 回归 head。改动的是：

- **输入端**：图像 token 之外，拼接一个由物理计算特征组成的小型 MLP 分支。
- **监督信号**：不是一步到位回归 DHI，而是先回归 GHI，把 GHI 的预测值（或其中间表示）当作额外特征喂给第二阶段的 DHI 回归器——这是一个两阶段的序列化管道，不是端到端联合优化。
- **架构**：双并行分支分别处理原始鱼眼图和物理分割后的 overlay 图，两路特征在最后融合。

没有改动 attention 机制本身，所以如果你已经会用 `timm` 加载一个 ViT，这篇论文的核心代码量其实不大。

## 实现

下面的代码是我按论文思路写的简化复现，不是官方代码（论文摘要未提供代码仓库链接，我也没有去猜测/编造链接）。目的是帮你理解结构，不是提供一套可以直接拿去生产的框架。

### 最小可运行版本：图像 + 物理特征的 ViT 回归器

```python
import torch
import torch.nn as nn
import timm

class MiniIrradianceViT(nn.Module):
    """图像分支(预训练ViT) + 物理特征分支(MLP)，融合后回归辐照度"""
    def __init__(self, phys_dim=4, embed_dim=192):
        super().__init__()
        # 用ImageNet预训练权重，小数据场景下这一步几乎是必须的
        self.vit = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True, num_classes=0
        )
        self.phys_mlp = nn.Sequential(
            nn.Linear(phys_dim, 32), nn.ReLU(), nn.Linear(32, 32)
        )
        self.head = nn.Sequential(
            nn.Linear(self.vit.num_features + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img, phys_feat):
        # img: [B,3,224,224]  phys_feat: [B,phys_dim]
        # phys_feat = [地外辐照度(归一化), sin(h), cos(h), 太阳天顶角]
        img_feat = self.vit(img)
        phys_feat = self.phys_mlp(phys_feat)
        fused = torch.cat([img_feat, phys_feat], dim=-1)
        return self.head(fused).squeeze(-1)
```

这一版就能跑起来，它已经体现了论文第一个关键点：物理特征不是直接拼在图像像素上，而是走独立的小 MLP 分支再融合，避免量级差异巨大的数值（辐照度上千 W/m² vs 图像像素 0-1）互相干扰。

### 无监督云层分割（生成 overlay，不需要训练）

论文强调分割是"unsupervised, physics-informed"，经典做法之一是红蓝比法（red-blue ratio）：晴空天空因瑞利散射偏蓝，云层因多次散射趋于灰白，红蓝比值可以作为简单且免标注的云检测信号。

```python
import numpy as np

def cloud_overlay(rgb_img, threshold=0.72):
    """rgb_img: HxWx3, float in [0,1]。红蓝比越高(越接近1)越可能是云"""
    r, b = rgb_img[..., 0], rgb_img[..., 2]
    ratio = r / (b + 1e-6)
    ratio = np.clip(ratio / ratio.max(), 0, 1)
    cloud_mask = (ratio > threshold).astype(np.float32)
    # overlay: 在原图基础上把云区域标红，作为第二路输入
    overlay = rgb_img.copy()
    overlay[..., 0] = np.where(cloud_mask > 0, 1.0, overlay[..., 0])
    return overlay, cloud_mask
```

注意 `threshold=0.72` 是我随手给的示例值，**这个阈值是强站点相关的**：不同纬度、不同相机白平衡下红蓝比分布完全不同，换一个站点基本要重新标定，这是这类"免训练"方法最大的隐藏成本——免去了标注数据，但换成了人工调阈值，别被"unsupervised"这个词骗了以为完全不需要人工介入。

### 序列化管道：GHI 作为 DHI 的中间特征

```python
class SequentialPipeline(nn.Module):
    """先预测GHI，把GHI预测值拼回特征里再预测DHI"""
    def __init__(self, phys_dim=4):
        super().__init__()
        self.ghi_model = MiniIrradianceViT(phys_dim=phys_dim)
        # DHI模型的物理特征维度+1，多出的1维是GHI的预测值
        self.dhi_model = MiniIrradianceViT(phys_dim=phys_dim + 1)

    def forward(self, img, phys_feat):
        ghi_pred = self.ghi_model(img, phys_feat)
        dhi_input_feat = torch.cat([phys_feat, ghi_pred.unsqueeze(-1)], dim=-1)
        dhi_pred = self.dhi_model(img, dhi_input_feat)
        return ghi_pred, dhi_pred
```

训练时两阶段的 loss 分开算再相加，`ghi_model` 和 `dhi_model` 可以联合训练，也可以先单独训好 `ghi_model` 再冻结——论文里"sequential"这个词暗示更像后者，但我没看到明确说明，这里我按联合训练来写，实际用的时候两种都值得试。

### 双路并行架构：原图 + overlay 图

```python
class DualParallelViT(nn.Module):
    """原始图像分支 + 物理分割overlay图像分支，特征级融合"""
    def __init__(self, phys_dim=4):
        super().__init__()
        self.raw_vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
        self.overlay_vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
        self.phys_mlp = nn.Sequential(nn.Linear(phys_dim, 32), nn.ReLU())
        feat_dim = self.raw_vit.num_features * 2 + 32
        self.head = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, raw_img, overlay_img, phys_feat):
        f_raw = self.raw_vit(raw_img)
        f_overlay = self.overlay_vit(overlay_img)
        f_phys = self.phys_mlp(phys_feat)
        fused = torch.cat([f_raw, f_overlay, f_phys], dim=-1)
        return self.head(fused).squeeze(-1)
```

### 关键 Trick（论文摘要没细说，但没有就跑不起来）

- **预训练权重几乎是硬性要求**：一个月的数据对 ViT 来说极小，从零训练几乎必然过拟合或学不动，必须用 ImageNet 预训练初始化，论文能在单月数据上取得接近全年训练的效果，预训练权重是隐藏功臣。
- **物理特征必须单独归一化**：地外辐照度量级在几百到一千多 W/m²，和周期编码（-1 到 1）量级差几百倍，不归一化直接拼接会让梯度被大数值特征主导。
- **两路 ViT 分支不要共享权重**：原图和 overlay 图的视觉统计特性不同（overlay 多了人工标红区域），共享权重会让其中一路训练信号被稀释，论文称为"dual-parallel"也暗示两路是独立网络。
- **数据增强要谨慎**：天空图像里颜色本身就是信号（蓝天 vs 灰云），常规的颜色抖动（color jitter）会破坏这个语义，适合用的是几何类增强（随机裁剪、轻微旋转），颜色类增强建议关掉或幅度调到很小。
- **月份选择会影响太阳高度角分布**：如果训练月恰好是冬季（太阳高度角一直很低），模型可能没见过夏季正午那种高辐照度场景，泛化到全年测试集时这块误差会特别大，这也是论文选 Chilbolton（温带、云层多变）而不是热带晴空站点做验证的原因——想证明方法在"季节变化大"的环境里依然稳健。

## 实验设计

论文选 Chilbolton Observatory 的原因很直接：温带气候、云层变化频繁，能同时测试模型在"低辐照度冬季"和"高辐照度夏季"之间的泛化能力，这比选一个常年晴空的站点更有说服力——常年晴空的话，随便一个简单回归模型都能"蒙"对大半时间的数值。

实验设置是用单月（如一月）训练，在时间上完全不重叠的全年测试集上评估，这就是"模拟数据稀缺"的具体做法，比随机划分训练/测试集更贴近真实部署场景（真实场景里你拿到的历史数据永远是"过去某段连续时间"，不是随机打散的）。

论文报告的结果（按摘要）：

| 方法 | 相对全量数据 SOTA ViT 的 RMSE 差距 |
|---|---|
| 序列化管道（Sequential） | 2–3 W/m² |
| 双路并行（Dual-Parallel） | 1–6 W/m² |

这个差距量级是合理的——辐照度本身量级在几十到上千 W/m²，个位数到小几位数的 RMSE 差距，说明单月训练确实可以逼近全年训练的效果。但要注意双路并行的区间上限到了 6 W/m²，波动范围比序列化管道大，这提示 overlay 分支的收益没有序列化管道那么稳定，大概率依赖云分割阈值标定的好坏——这正是对应我前面提到的"阈值强站点相关"的隐藏成本。

### 消融实验（论文未必给出完整数据，以下是我基于结构的推测方向）

值得做消融验证的三个组件：

1. 去掉周期时间编码，换成线性小时数——预期在跨午夜样本上误差上升。
2. 去掉 GHI 中间监督，直接端到端回归 DHI——预期在数据量更小时差距会更明显，因为 GHI 相当于一个更容易学的辅助任务，帮模型先建立"总量级"的判断。
3. 去掉 overlay 分支，只用原图——预期在云层快速变化的阴天样本上误差上升，因为 overlay 本质是把云的边界信息显式标出来，减轻了 ViT 自己从像素里学"什么是云边缘"的负担。

## 调试指南

### 常见问题

1. **Loss 几乎不降**：先检查物理特征是否做了归一化，辐照度量级的数值不归一化基本会直接把梯度冲爆；再检查云分割生成的 mask 是不是全黑或全白（阈值设得不合适），全黑/全白的 overlay 等于没提供任何信息。
2. **单月训练后在其他月份性能骤降**：检查训练月的太阳天顶角分布范围是否覆盖了测试月的范围，如果训练月太阳高度角一直很低（冬季），模型在夏季高辐照度场景会系统性低估。
3. **双路并行比单路原图还差**：大概率是 overlay 分支的阈值在训练集上标定过头（过拟合某种光照条件下的红蓝比分布），换个时间段/季节这个阈值就失效了，建议先单独检查分割质量（肉眼看几十张 overlay），再决定要不要用。

### 如何判断模型在"学习"

看验证集 RMSE 是否在前几个 epoch 内明显下降（因为用了预训练权重，收敛应该比从零训练快很多，通常 5-10 个 epoch 内就能看出趋势），如果 20 个 epoch 后验证 RMSE 还和随机初始化差不多，大概率是数据管道（归一化、标签对齐）有问题，不是模型容量问题。

### 超参数建议

| 参数 | 推荐范围 | 敏感度 | 说明 |
|---|---|---|---|
| 学习率 | 1e-4 ~ 3e-4（ViT主干），1e-3（MLP分支） | 高 | 小数据集容易过拟合，偏大学习率+早停比小学习率+长训练更稳 |
| patch size | 16 | 中 | 鱼眼图像畸变区域大，太小的patch会把畸变边缘切碎 |
| 云分割阈值 | 站点相关，需人工标定 | 高 | 换站点必须重新看样本调整，不要复用别的站点的值 |
| 是否预训练 | 必须用 | — | 单月数据从零训练几乎不可行 |

## 什么时候用 / 不用

| 适用场景 | 不适用场景 |
|---|---|
| 新建电站，历史数据不足一年 | 已有多年高质量数据（直接训练全量ViT更简单可靠） |
| 温带/多云多变气候，季节覆盖是关键问题 | 卫星遥感图像输入（鱼眼相机特有的畸变校正和overlay方法不适用） |
| 能接受人工标定分割阈值的运维投入 | 需要全自动、零人工干预的流水线部署 |
| 研究/原型验证阶段 | 需要实时高频推理（双路ViT比单路慢接近一倍） |

## 我的观点

这篇论文的价值不在模型架构，ViT 本身一行都没改，真正的贡献是证明了"物理先验 + 简单无监督预处理"在数据稀缺场景下能顶替掉大量训练数据的作用，这个思路本身比具体的 RMSE 数字更值得借鉴，可以迁移到其他遥感/物联网回归任务（比如用卫星云图估算地面温度、用少量传感器数据外推区域降水）。

但双路并行方案的误差区间（1-6 W/m²）比序列化管道（2-3 W/m²）波动更大，说明 overlay 分支收益不稳定，很可能卡在云分割阈值的站点相关性上——这是我认为论文里最容易在实际部署中"翻车"的一环，换个站点、换个相机型号，这个阈值基本要重新标定，而这部分工作量在"cost-effective"的叙事里经常被低估。如果要在生产环境用，我会优先选序列化管道而不是双路并行，省掉一套需要持续维护的分割阈值。

论文链接：[arxiv.org/abs/2610.30940](https://arxiv.org/abs/2610.30940)（本文代码为按论文思路的自行复现，非官方实现）。