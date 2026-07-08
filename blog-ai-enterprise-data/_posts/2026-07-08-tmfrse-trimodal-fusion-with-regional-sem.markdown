---
layout: post-wide
title: '不只是预测，还要知道"我有多不确定"：肺部严重度评分中的三模态融合与证据回归'
date: 2026-07-08 12:02:07 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.06356v1
generated_by: Claude Code CLI
---

## 一句话总结

TMF-RSE 把胸部影像、肺部分割掩码、VLM 语义特征三路融合，并用**证据回归（Evidential Regression）**同时输出严重度预测和不确定性估计——后者才是这篇论文真正值得关注的技术点。

---

## 为什么这篇论文重要？

医生看 CT 时做的事情是连续评分，不是二分类。肺部 COVID 严重度通常用 0-100% 的受累比例来衡量，或者用多区域的地理范围评分（RALO 数据集）。现有方法的问题：

1. **只给点估计**：预测"肺受累 43%"，但不告诉你这个预测有多可靠
2. **多模态没用好**：要么只用图像，要么用晚期融合堆特征，没有让模态之间真正"对话"
3. **分割信息被浪费**：分割掩码通常只用来做预处理，而不是作为独立的结构化信息通道

TMF-RSE 的核心洞见是：**结构先验（分割掩码）+ 语义引导（VLM 文本特征）+ 不确定性量化（证据回归）三者缺一不可**。在临床决策中，不确定性和预测值本身同样重要——一个置信度低的预测应该触发人工复核，而不是直接进入流程。

---

## 核心方法解析

### 三路输入是什么？

```
输入 1：2D 胸部影像  →  外观特征（纹理、密度异常）
输入 2：肺部分割掩码  →  结构特征（解剖位置、受累区域边界）
输入 3：VLM 文本提示  →  语义特征（"双肺多发磨玻璃影，以下叶为主"）
```

分割掩码单独作为一路输入，而不是只用来裁剪图像，这个设计让模型能显式学习"哪个区域受累"与"受累程度"的对应关系。

### 证据回归：最值得深挖的部分

标准回归预测一个数 $\hat{y}$。证据回归（基于 Normal-Inverse-Gamma 先验）预测四个参数 $(\gamma, \nu, \alpha, \beta)$，分别编码：

- $\gamma$：预测均值（等价于点估计）
- $\nu, \alpha, \beta$：编码对这个预测"有多少证据支撑"

不确定性可以分解为两部分：

$$U_{aleatoric} = \frac{\beta}{\alpha - 1} \quad \text{（数据噪声，不可约）}$$

$$U_{epistemic} = \frac{\beta}{\nu(\alpha - 1)} \quad \text{（模型不确定性，可通过更多数据降低）}$$

训练用 NIG-NLL 损失加正则项：

$$\mathcal{L} = \mathcal{L}_{NIG\text{-}NLL} + \lambda \cdot |y - \gamma| \cdot (2\nu + \alpha)$$

正则项的直觉：当预测误差大时，惩罚"证据量"$\nu$ 和 $\alpha$ 过高——即强迫模型不能在预测错误时还"自信满满"。

---

## 动手实现

### 证据回归头（核心）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialRegressionHead(nn.Module):
    """输出 NIG 分布的四个参数，而非单一点估计"""
    
    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 4)  # γ, log_ν, log_α, log_β
    
    def forward(self, x):
        out = self.fc(x)
        gamma = out[:, 0]
        # ν > 0, α > 1, β > 0，用 softplus 保证约束
        nu    = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0   # α > 1 使方差有定义
        beta  = F.softplus(out[:, 3]) + 1e-6
        return gamma, nu, alpha, beta
    
    def uncertainty(self, nu, alpha, beta):
        aleatoric  = beta / (alpha - 1)
        epistemic  = beta / (nu * (alpha - 1))
        return aleatoric, epistemic


def nig_nll_loss(y, gamma, nu, alpha, beta):
    """Normal-Inverse-Gamma 负对数似然"""
    two_beta_lambda = 2.0 * beta * (1.0 + nu)
    
    loss = (0.5 * torch.log(torch.tensor(torch.pi) / nu)
            - alpha * torch.log(two_beta_lambda)
            + (alpha + 0.5) * torch.log((y - gamma)**2 * nu + two_beta_lambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5))
    return loss.mean()


def evidential_loss(y, gamma, nu, alpha, beta, lam=0.1):
    """NIG-NLL + 正则项，防止模型对错误预测过度自信"""
    nll = nig_nll_loss(y, gamma, nu, alpha, beta)
    reg = (torch.abs(y - gamma) * (2 * nu + alpha)).mean()
    return nll + lam * reg
```

### 三模态融合骨架

```python
class EvidentialRegressionHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 4)  # γ, ν, α, β

    def forward(self, x):
        out = self.fc(x)
        gamma = out[:, 0]
        nu    = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0   # α > 1 使方差有定义
        beta  = F.softplus(out[:, 3]) + 1e-6
        return gamma, nu, alpha, beta


def evidential_loss(y, gamma, nu, alpha, beta, lam=0.1):
    # NIG 负对数似然
    two_beta_lambda = 2.0 * beta * (1.0 + nu)
    nll = (0.5 * torch.log(torch.pi / nu)
           - alpha * torch.log(two_beta_lambda)
           + (alpha + 0.5) * torch.log((y - gamma)**2 * nu + two_beta_lambda)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)).mean()
    # 正则项：防止对错误预测过度自信
    reg = (torch.abs(y - gamma) * (2 * nu + alpha)).mean()
    return nll + lam * reg
```

### 训练循环

```python
import torchvision.models as tvm

class SegmentationEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)  # 单通道输入
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(512, out_dim)

    def forward(self, mask):
        return self.proj(self.encoder(mask).flatten(1))


class CrossModalAttention(nn.Module):
    def __init__(self, dim=256, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        q, kv = query.unsqueeze(1), key_value.unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        return self.norm(query + out.squeeze(1))


class TMFRSE(nn.Module):
    """三模态融合 + 证据回归"""
    def __init__(self, feat_dim=256):
        super().__init__()
        backbone = tvm.resnet50(weights='IMAGENET1K_V1')
        self.img_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.img_proj = nn.Linear(2048, feat_dim)
        self.seg_encoder = SegmentationEncoder(feat_dim)
        self.vlm_encoder = VLMSemanticExtractor(feat_dim)  # CLIP 文本编码器，dim=512
        # 掩码结构 + 语义分别引导图像特征
        self.img_seg_attn = CrossModalAttention(feat_dim)
        self.img_vlm_attn = CrossModalAttention(feat_dim)
        self.fusion = nn.Sequential(nn.Linear(feat_dim * 3, feat_dim), nn.GELU(), nn.Dropout(0.3))
        self.head = EvidentialRegressionHead(feat_dim)

    def forward(self, image, mask, text_emb):
        img_feat = self.img_proj(self.img_encoder(image).flatten(1))
        seg_feat, vlm_feat = self.seg_encoder(mask), self.vlm_encoder(text_emb)
        img_refined = self.img_vlm_attn(self.img_seg_attn(img_feat, seg_feat), vlm_feat)
        return self.head(self.fusion(torch.cat([img_refined, seg_feat, vlm_feat], dim=-1)))
```

---

## 实现中的坑

**1. α 初始化问题**

`alpha > 1` 的约束是证据回归的硬性要求（确保方差有限）。`F.softplus` 输出范围是 `(0, +∞)`，加 `1.0` 后变为 `(1, +∞)`，但训练初期 $\alpha$ 非常接近 1 会导致不确定性估计爆炸。在批归一化层前加一个梯度裁剪很重要。

**2. λ（正则强度）对结果影响极大**

论文没有详细讨论这个超参数，但实践中 `λ ∈ [0.01, 0.5]` 的范围内结果差异显著。过小：模型对错误预测过度自信；过大：模型退化为保守估计，MAE 反而升高。建议先用 `0.1` 跑出基线，再用较小的学习率微调 λ。

```python
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks, text_embs, labels in loader:
        # ... (数据转移到设备)
        gamma, nu, alpha, beta = model(images, masks, text_embs)
        loss = evidential_loss(labels, gamma, nu, alpha, beta, lam=0.1)
        optimizer.zero_grad()
        loss.backward()
        # ... (梯度裁剪)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def predict_with_uncertainty(model, image, mask, text_emb, device):
    model.eval()
    gamma, nu, alpha, beta = model(...)
    aleatoric, epistemic = model.head.uncertainty(nu, alpha, beta)
    return {
        "severity_score": gamma,
        "aleatoric_uncertainty": aleatoric,  # 数据噪声
        "epistemic_uncertainty": epistemic,   # 模型不确定
    }
```

**3. VLM 提示词工程被严重低估**

论文用了分割感知的视觉描述，但没说明提示词如何生成。实际实现中，提示词质量对最终结果影响可能超过架构本身。建议用模板：`"Chest CT showing {region} involvement with {pattern} pattern, estimated {severity} severity."`，并对 severity 做离散桶化后在模板中填入。

---

## 论文说的 vs 现实

| 指标 | 论文报告 | 现实注意点 |
|------|---------|-----------|
| Per-COVID MAE 4.02 | 验证集（非测试集） | 数据集不大，小样本波动明显 |
| RALO MAE 0.339 | 地理范围评分 | 离散打分被当成连续回归，标签噪声高 |
| Pearson 0.973 | 相关性 vs 绝对误差 | 高相关不代表临床可用，量纲误差才是关键 |

不确定性估计的校准结果论文没有报告，这是一个明显的遗漏。ECE（Expected Calibration Error）或可靠性图才能真正验证不确定性的临床价值。

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有高质量分割掩码可用（或自动分割质量可接受） | 分割掩码缺失或质量差 |
| 需要不确定性量化用于触发人工复核 | 只需要最高精度的点估计 |
| 有放射科报告可以生成 VLM 文本特征 | 纯影像场景，无文本辅助信息 |
| 小数据集（百级别），需要用预训练先验补偿 | 大规模数据集（万级），端到端训练更强 |

---

## 我的观点

TMF-RSE 最有价值的贡献不是三模态融合本身，而是把**证据回归引入医学图像定量评分**这件事。多模态融合在影像AI里已经不新鲜，但大多数论文输出的仍然是置信区间模糊的点估计，在临床场景中可用性有限。

不确定性分解（偶发 vs 认知）这个框架很有吸引力，但论文没有提供校准实验，这是一个重要缺口。一个值得做的后续工作是：高认知不确定性的样本是否能有效识别出"需要人工审核"的边界案例？如果能，这个方法在 AI 辅助诊断流程中的落地价值会远大于几个 MAE 点的提升。

VLM 分支目前是这个架构里最"黑盒"的部分。提示词如何生成、VLM 权重是否微调、文本特征的 domain gap 问题，论文都没有讨论清楚。这个分支既可能是关键增益来源，也可能是在 in-domain 数据上过拟合的噪声源。分离实验（消融掉 VLM 分支）的结果我会特别关注。