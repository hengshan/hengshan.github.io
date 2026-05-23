---
layout: post-wide
title: "用 AlphaEarth 地理空间嵌入做作物识别：卫星遥感语义分割实战"
date: 2026-05-23 12:03:57 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.21804v1
generated_by: Claude Code CLI
---

## 一句话总结

用 Google DeepMind 的预训练地理空间基础模型（AlphaEarth）替代手工特征工程，配合 U-Net + Monte Carlo Dropout，在加州实现 99% 以上精度的番茄田地块识别，并输出可信的不确定性地图。

## 为什么这个问题重要？

作物地图（Crop Map）是农业供应链预测、政策制定和保险定损的核心数据源。加州是美国最大的加工番茄产区，年产量占全美 90% 以上，及时、准确的地块识别直接影响亿级美元的合同交易。

传统遥感工作流的三大痛点：

- **手工特征工程**：NDVI、EVI 等植被指数需要大量预处理，每年都要重新调参
- **跨年泛化差**：作物物候期受气候影响，2018 年调好的模型在干旱年表现可能崩塌
- **时效性差**：现有方案依赖事后调查（retrospective surveys），农时早已过去

这篇论文的核心思路：**用预训练地理空间基础模型输出的 64 维嵌入向量作为即用特征，完全跳过特征工程步骤**。这和 NLP 里用 BERT embedding 替代 TF-IDF 的逻辑完全一致。

## 背景知识

### 遥感影像的表示方式对比

| 表示方式 | 特点 | 缺点 |
|---------|------|------|
| 原始多光谱影像 | 信息完整，直接可读 | 需大量预处理，波段多达 10+ |
| 手工植被指数 (NDVI 等) | 直观，领域经验丰富 | 跨年/跨域泛化弱 |
| **基础模型嵌入（AlphaEarth）** | 无需特征工程，语义丰富 | 依赖预训练模型，黑盒 |

### AlphaEarth 嵌入是什么？

AlphaEarth 是 Google DeepMind 开发的地理空间基础模型，类比 CV 领域的 CLIP 或 DINOv2。在大规模卫星影像上预训练后，它输出的嵌入向量同时编码了：

- **空间结构**：田块边界、纹理
- **时序信息**：多时相物候特征（春种、夏长、秋收）
- **光谱特征**：不同作物的光谱响应差异

本文提取的是 **64 维嵌入图（64-band embedding chip）**，可直接送入下游模型，不需要理解 Sentinel-2 各波段的物理含义。

### U-Net：语义分割的标准选择

U-Net 最早用于医学图像分割，核心思想是编码器压缩特征、解码器恢复分辨率，跳跃连接（skip connection）保留空间细节。在遥感语义分割中依然是主力架构：

```
输入 (B, 64, H, W)  [64 维 AlphaEarth embedding chip]
  → Encoder: 特征提取 + 下采样（3 层）
  → Bottleneck: 语义压缩
  → Decoder: 上采样 + 跳跃连接拼接
  → Head: 1×1 卷积 → 输出 (B, 1, H, W) [番茄概率图]
```

### Monte Carlo Dropout：廉价的不确定性估计

标准神经网络只给点预测，不知道"模型在哪里不确定"。Monte Carlo Dropout 的思路：

- 训练时 Dropout 正常开启
- **推理时也保持 Dropout 开启**（关键异常操作）
- 重复推理 T 次（本文 T=100）
- 用 T 次结果的方差作为不确定性度量

## 核心方法

### 关键损失函数

论文使用带掩膜的复合损失（Masked BCE + Soft Dice Loss）：

$$
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda \cdot \mathcal{L}_{\text{Dice}}
$$

Soft Dice Loss 对类别不平衡更鲁棒（田块边缘像素少，内部像素多）：

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}
$$

**掩膜的作用**：只对多边形标注内部的有效像素计算损失，忽略背景区域——避免大量简单负样本主导梯度。

### 空间独立数据集划分

地理空间 ML 中最容易被忽视的坑：**不能随机划分训练/验证/测试集**。

地理上相邻的田块共享光照、土壤、灌溉系统，随机 split 会严重高估泛化能力。正确做法是按**空间区域块**划分，确保测试区域在训练阶段完全不可见。

## 实现

### 核心架构：U-Net

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """U-Net 基本单元: Conv → BN → ReLU (含 MC Dropout)"""
    def __init__(self, in_ch, out_ch, dropout_p=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),   # Dropout2d 用于 MC 采样
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    """适配 64 维 AlphaEarth 嵌入的轻量 U-Net"""
    def __init__(self, in_channels=64, base_ch=32, dropout_p=0.2):
        super().__init__()
        C = base_ch
        self.enc1 = ConvBlock(in_channels, C, dropout_p)
        self.enc2 = ConvBlock(C,   C*2, dropout_p)
        self.enc3 = ConvBlock(C*2, C*4, dropout_p)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(C*4, C*8, dropout_p)
        self.up3  = nn.ConvTranspose2d(C*8, C*4, 2, stride=2)
        self.dec3 = ConvBlock(C*8, C*4, dropout_p)   # concat skip
        self.up2  = nn.ConvTranspose2d(C*4, C*2, 2, stride=2)
        self.dec2 = ConvBlock(C*4, C*2, dropout_p)
        self.up1  = nn.ConvTranspose2d(C*2, C,   2, stride=2)
        self.dec1 = ConvBlock(C*2, C, dropout_p)
        self.head = nn.Conv2d(C, 1, 1)               # 输出 logits

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
```

### 复合损失函数

```python
import torch.nn.functional as F

class MaskedCompositeLoss(nn.Module):
    """掩膜 BCE + Soft Dice，只对标注区域内像素计算损失"""
    def __init__(self, dice_weight=0.5, eps=1e-6):
        super().__init__()
        self.w, self.eps = dice_weight, eps

    def forward(self, logits, targets, mask):
        probs = torch.sigmoid(logits)
        # 掩膜 BCE：只对 mask=1 的像素求均值
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        bce_loss = (bce * mask).sum() / (mask.sum() + self.eps)
        # Soft Dice（有效区域内）
        p = (probs * mask).flatten(1)
        g = (targets * mask).flatten(1)
        dice = 1 - (2*(p*g).sum(1) + self.eps) / (p.sum(1) + g.sum(1) + self.eps)
        return bce_loss + self.w * dice.mean()
```

### Monte Carlo Dropout 推理与不确定性估计

```python
def mc_dropout_predict(model, x, n_samples=100):
    """
    推理时保持 Dropout 激活，重复采样估计预测分布
    返回: (均值概率图, 方差不确定性图)
    """
    model.train()   # 关键：train 模式激活 Dropout，eval 模式不行
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(torch.sigmoid(model(x)))
    preds = torch.stack(preds)   # (T, B, 1, H, W)
    return preds.mean(0), preds.var(0)  # 均值 + 方差

# 可视化不确定性（田块边缘方差最高）
# mean_map, var_map = mc_dropout_predict(model, chip.unsqueeze(0))
# plt.imshow(var_map.squeeze().cpu(), cmap='hot')  # 高方差 = 红色 = 不确定
```

### 训练与评估

```python
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for chips, masks, labels in loader:
        chips  = chips.to(device)   # (B, 64, H, W)  AlphaEarth 嵌入
        masks  = masks.to(device)   # (B, 1, H, W)   有效像素掩膜
        labels = labels.to(device)  # (B, 1, H, W)   番茄=1 / 非番茄=0
        optimizer.zero_grad()
        loss = criterion(model(chips), labels.float(), masks.float())
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def compute_metrics(probs, targets, thresh=0.5):
    pred = (probs > thresh).float()
    tp = (pred * targets).sum()
    fp = (pred * (1 - targets)).sum()
    fn = ((1 - pred) * targets).sum()
    p  = tp / (tp + fp + 1e-6)
    r  = tp / (tp + fn + 1e-6)
    return {"f1": (2*p*r/(p+r+1e-6)).item(),
            "iou": (tp/(tp+fp+fn+1e-6)).item()}
```

## 实验结果

### 定量评估

| 指标 | AlphaEarth + U-Net（本文）| 传统 NDVI 方法（参考范围）|
|------|--------------------------|------------------------|
| 像素准确率 | **99.19%** | 94–96% |
| Precision | 98.69% | 91–95% |
| Recall | 99.40% | 90–94% |
| F1 | **99.04%** | 92–95% |
| IoU | **98.11%** | 86–92% |

*传统方法数值为遥感作物识别文献常见范围，非本文直接对比。*

### 不确定性地图的空间模式

MC Dropout 方差图呈现清晰的空间规律：

- **田块内部**：方差接近 0，模型高度确定
- **田块边缘**：方差最高，对应标注本身模糊的地带（路边植被混入）
- **道路/灌溉渠**：中等不确定性

这和人工审核结果高度一致——可以用不确定性地图触发**主动学习**：只复核高不确定区域，标注成本大幅降低。

## 工程实践

### 实际部署考虑

**硬件需求**：
- 训练：AWS SageMaker（论文），或本地 RTX 3090/4090 可复现
- 推理：单张 512×512 chip，MC Dropout 100 次约 2–3 秒（GPU）；大面积批量处理需分块（tile-based）推理

**生产环境加速**：MC Dropout 100 次全量推理成本高，可用自适应策略：

```python
def adaptive_mc_predict(model, x, fast_n=10, full_n=100, var_thresh=0.03):
    """快速初判：方差低则直接返回；方差高才做完整采样"""
    mean, var = mc_dropout_predict(model, x, n_samples=fast_n)
    if var.max().item() < var_thresh:
        return mean, var          # 低不确定性：快速路径
    return mc_dropout_predict(model, x, n_samples=full_n)
```

### 常见坑

1. **空间数据泄露** → 相邻田块共享环境因素，随机 split 导致测试集虚高 3–5%。修复：按经纬度网格做空间块划分，训练区和测试区地理边界框无重叠。

2. **MC Dropout 推理忘记 `model.train()`** → 所有采样完全相同，方差为零。修复：推理函数开头显式调用 `model.train()`，推理结束后恢复 `model.eval()`。

3. **掩膜逻辑错误** → 把多边形外的大面积背景纳入损失，模型偏向预测背景类，召回率崩塌。修复：用田块多边形栅格化生成精确的有效像素掩膜。

4. **Dice Loss 在极小 batch 下不稳定** → `batch_size=1` 时分子分母极小，加 `eps` 仍可能梯度爆炸。建议 `batch_size >= 4`，或切换为 batch-level Dice。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有 AlphaEarth API 访问权限 | 无法访问预训练嵌入（闭源限制） |
| 已知田块边界多边形 | 只有点标注或像素标注 |
| 单一目标作物识别 | 多类作物细粒度区分（10+类） |
| 光照/物候稳定的正常年份 | 极端气候年份（嵌入分布偏移） |
| 大规模区域批量离线处理 | 实时流式推理（MC Dropout 延迟高） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 手工光谱指数 (NDVI/EVI) | 可解释，无 GPU 需求 | 跨年泛化差，调参繁琐 | 小规模、预算受限 |
| CNN + 原始多光谱 | 端到端，可定制 | 预处理复杂，需大量标注 | 有充足标注数据时 |
| **AlphaEarth + U-Net（本文）** | 无特征工程，精度高，有不确定性 | 依赖闭源基础模型 | 大规模作物制图 |
| SAM（Segment Anything） | 通用分割，零样本 | 无时序感知，遥感域外泛化弱 | 快速原型验证 |

## 我的观点

这篇论文最大的价值不在于 99% 的数字，而在于它验证了一个迁移学习范式：**把地理空间基础模型当作 feature extractor，下游任务用轻量监督模型微调**。路径和 NLP 的 BERT → fine-tune 完全对应，遥感领域正在经历同样的范式迁移。

真正的瓶颈是 AlphaEarth 是 Google DeepMind 的闭源产品，外部访问依赖 API。没有 API 权限时，开源替代品值得关注：
- **Prithvi**（NASA/IBM）：开源遥感基础模型，Hugging Face 可直接下载
- **SatMAE**：基于 ViT 的时序遥感 MAE 预训练
- **GeoCLIP**：地理感知的对比学习嵌入

不确定性量化是高风险农业应用的硬需求，但 100 次 MC Dropout 的推理成本在大面积时会成为实际瓶颈。工程上 **Deep Ensemble（3–5 个模型）** 往往比 MC Dropout 有更好的速度/质量权衡，且实现更简单。

最后一个开放问题：论文只验证了 2018 年一个年份。**跨年泛化**——2018 年训练的模型在 2022 年极端干旱季是否依然可靠——是这条技术路线进入产品级部署前必须回答的问题。