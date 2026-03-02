---
layout: post-wide
title: "脑卒中 CT 分割：DINOv3 + 区域门控损失实现临床对齐的 ASPECTS 评分"
date: 2026-03-02 08:02:28 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.23961v1
generated_by: Claude Code CLI
---

## 一句话总结

在非增强 CT（NCCT）上做缺血性脑卒中分割时，单纯的像素级监督忽略了临床 ASPECTS 评分的解剖结构约束——本文提出的 **Territory-Aware Gated Loss（TAGL）** 在训练阶段强制基底节层（BG）与上节层（SG）一致性，Dice 从 0.698 提升到 0.767，且**推理时零额外开销**。

---

## 背景：为什么脑卒中分割这么难？

### ASPECTS 评分是什么

ASPECTS（Alberta Stroke Program Early CT Score）是急性缺血性卒中临床决策的核心工具：把大脑中动脉供血区分成 **10 个子区域**（BG 层 4 个 + SG 层 6 个），每个区域梗死扣 1 分，满分 10 分，分数越低梗死越重。

问题在于：**BG 层和 SG 层是解剖上耦合的**。临床医生看片时会联合两个层面判断，而大多数深度学习模型把每张切片独立处理，完全忽视了这种跨层一致性。

### 现有方法的局限

- **CNN（U-Net 系列）**：擅长局部纹理，但 NCCT 上梗死区域与正常组织对比度极低（Hounsfield 差值 < 10 HU），边界模糊
- **基础模型直接迁移（DINOv2/SAM）**：特征表达能力强，但没有领域先验，不懂 BG-SG 解剖关系
- **像素级监督**：对每个像素一视同仁，无法编码"BG 区域的梗死应当在 SG 层有对应"这种结构约束

### 本文的核心 insight

> 不改推理架构，只在**训练时**注入结构先验：把 BG/SG 的一致性要求编码进损失函数。

---

## 算法原理

### 整体架构

```
NCCT 切片 → 冻结 DINOv3 骨干 → patch token 特征
                                        ↓
                              轻量级解码器（几个卷积层）
                                        ↓
                              像素级分割 logits
                                        ↓
                              TAGL 损失（训练期）
```

冻结骨干有两个好处：节省显存、避免在小医学数据集上过拟合大模型。

### 直觉解释：什么是 Territory-Aware Gated Loss

普通 Dice Loss 对每个像素独立计算，不知道"这个像素属于 BG 层还是 SG 层"。

TAGL 的思路是：给每个切片打上解剖层标签（BG 或 SG），然后在计算损失时**按层级加权，并惩罚跨层不一致的预测**。

形象地说：如果模型在 BG 层预测了一大块梗死，但在对应的 SG 层却说没有梗死，TAGL 会额外加罚。

### 数学推导

设切片 $i$ 的预测为 $\hat{y}_i$，真值为 $y_i$，解剖层标签为 $t_i \in \{\text{BG}, \text{SG}\}$。

**标准 Dice Loss：**

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i \hat{y}_i y_i}{\sum_i \hat{y}_i + \sum_i y_i + \epsilon}
$$

**Territory-Aware Gated Loss：**

$$
\mathcal{L}_{\text{TAGL}} = \mathcal{L}_{\text{Dice}} + \lambda \cdot \mathcal{L}_{\text{consistency}}
$$

其中一致性项惩罚 BG-SG 预测的不匹配：

$$
\mathcal{L}_{\text{consistency}} = \frac{1}{|P|} \sum_{(i,j) \in P} \left\| \text{pool}(\hat{y}_i^{\text{BG}}) - \text{pool}(\hat{y}_j^{\text{SG}}) \right\|_2^2
$$

- $P$ 是同一患者 BG-SG 切片配对集合
- $\text{pool}(\cdot)$ 是区域平均池化（把分割图压成区域级别的激活向量）
- $\lambda$ 控制一致性约束强度，论文默认 0.1

**关键点：** 这个损失只在训练时计算，推理时模型结构不变，无额外延迟。

---

## 实现

### 最小可运行版本

下面是核心思想的最小实现，聚焦 TAGL 损失本身：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TAGLoss(nn.Module):
    """
    Territory-Aware Gated Loss
    在标准 Dice Loss 基础上，增加 BG-SG 一致性约束
    """
    def __init__(self, lambda_consistency=0.1, eps=1e-6):
        super().__init__()
        self.lam = lambda_consistency
        self.eps = eps

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred).view(pred.size(0), -1)
        target = target.view(target.size(0), -1).float()
        inter = (pred * target).sum(dim=1)
        return 1 - (2 * inter + self.eps) / (pred.sum(dim=1) + target.sum(dim=1) + self.eps)

    def consistency_loss(self, pred_bg, pred_sg):
        """
        pred_bg, pred_sg: [N, 1, H, W]
        惩罚同一患者 BG 层和 SG 层预测的区域级不匹配
        """
        # 压成区域均值（区域级别激活）
        act_bg = torch.sigmoid(pred_bg).mean(dim=[2, 3])  # [N, 1]
        act_sg = torch.sigmoid(pred_sg).mean(dim=[2, 3])  # [N, 1]
        return F.mse_loss(act_bg, act_sg)

    def forward(self, pred, target, pred_bg=None, pred_sg=None):
        base = self.dice_loss(pred, target).mean()
        if pred_bg is not None and pred_sg is not None:
            consist = self.consistency_loss(pred_bg, pred_sg)
            return base + self.lam * consist
        return base
```

### 完整模型实现

```python
import torch
import torch.nn as nn

class LightweightDecoder(nn.Module):
    """
    接在冻结 DINOv3 后的轻量解码器
    输入: patch tokens [B, N, C]，输出: 分割图 [B, 1, H, W]
    """
    def __init__(self, in_dim=768, img_size=224, patch_size=14):
        super().__init__()
        self.H = self.W = img_size // patch_size  # token grid 尺寸
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, tokens):
        # tokens: [B, N, C]，去掉 cls token
        x = tokens[:, 1:, :]          # [B, N-1, C]
        x = self.proj(x)               # [B, N-1, 256]
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.H, self.W)
        return self.upsample(x)        # [B, 1, H', W']


class StrokeSegModel(nn.Module):
    def __init__(self, dino_model, img_size=224):
        super().__init__()
        self.backbone = dino_model     # 冻结的 DINOv3
        self.decoder = LightweightDecoder(img_size=img_size)

        # 冻结骨干
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            tokens = self.backbone.get_intermediate_layers(x, n=1)[0]
        return self.decoder(tokens)
```

### 训练循环骨架

```python
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for batch in loader:
        # batch 包含: image, mask, level ("BG" or "SG"), patient_id
        imgs, masks, levels, pids = batch

        optimizer.zero_grad()
        preds = model(imgs)            # [B, 1, H, W]

        # 拆分 BG / SG 切片
        bg_mask = [l == "BG" for l in levels]
        sg_mask = [l == "SG" for l in levels]

        pred_bg = preds[bg_mask] if any(bg_mask) else None
        pred_sg = preds[sg_mask] if any(sg_mask) else None

        loss = criterion(preds, masks, pred_bg, pred_sg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

# ... (评估循环省略)
```

### 关键 Trick（重要！）

这些在论文里可能一笔带过，但没有就很难跑起来：

**1. DINOv3 特征归一化**

```python
# DINOv3 输出的 token 方差很大，直接接解码器会不稳定
tokens = F.layer_norm(tokens, [tokens.size(-1)])
```

**2. NCCT 窗宽窗位预处理**

```python
def apply_stroke_window(ct_array, wl=40, ww=80):
    """脑窗：窗位40HU，窗宽80HU，这是卒中最关键的预处理"""
    lo, hi = wl - ww / 2, wl + ww / 2
    ct_array = ct_array.clip(lo, hi)
    return (ct_array - lo) / (ww + 1e-8)
```

**3. BG-SG 配对采样**

同一个 batch 里要尽量包含同一患者的 BG 和 SG 切片，否则一致性损失计算不到：

```python
# 在 DataLoader 的 sampler 里按患者分组采样
# 或者简单做法：batch_size 设成 2 的倍数，每个患者取 BG+SG 各一张
```

**4. 类别不平衡**

梗死区域一般只占切片的 1-5%，纯 Dice Loss 不够，要加 BCE 或 Focal：

```python
loss = 0.5 * dice_loss + 0.5 * F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor([10.0]))
```

---

## 实验与结果分析

### 数据集

- **AISD**（公开）：来自 397 例患者的 NCCT，包含像素级梗死标注
- **私有 ASPECTS 数据集**：有 ASPECTS 子区域标注，更接近临床任务

### 核心结果

| 方法 | AISD Dice | ASPECTS Dice |
|------|-----------|-------------|
| U-Net | 0.58 | 0.672 |
| DINOv2 直接迁移 | 0.61 | 0.698 |
| DINOv3 + TAGL（本文） | **0.6385** | **0.767** |

TAGL 带来的提升在 ASPECTS 数据集上更明显（+6.9%），因为 ASPECTS 本身就是区域级任务，结构先验更重要。

### 消融实验

| 配置 | ASPECTS Dice |
|------|-------------|
| DINOv3 backbone only | 0.698 |
| + Dice Loss | 0.731 |
| + TAGL（完整） | **0.767** |

一致性约束单独带来约 3.6% 提升，是本文最核心的贡献。

---

## 调试指南

### 常见问题

**1. 一致性损失不降，甚至上升**

- **原因**：batch 里 BG/SG 样本极度不均衡，配对为空
- **解法**：检查 batch 构造逻辑，确保每个 batch 有足够的 BG-SG 对；或者把 `lambda_consistency` 降低到 0.01 先试

**2. 分割图全黑（全预测为背景）**

- **原因**：类别不平衡没处理，正样本太少
- **解法**：加 `pos_weight`（建议先试 10，再根据正负比例调整）

**3. DINOv3 token 维度对不上**

- 不同 DINOv3 变体（ViT-S/B/L）输出维度不同（384/768/1024），`LightweightDecoder` 的 `in_dim` 要对应修改
- 用 `print(tokens.shape)` 确认后再接解码器

**4. Dice 在 0.3-0.4 震荡不动**

- NCCT 对比度低，纯靠 DINOv3 冻结特征可能不够
- 尝试解冻最后 2 个 transformer block，用更小的学习率（1e-5）微调

### 超参数调优

| 参数 | 推荐初始值 | 敏感度 | 建议 |
|------|-----------|-------|------|
| lr（解码器） | 1e-3 | 高 | 先用这个，不收敛再降 |
| lr（骨干微调） | 1e-5 | 高 | 比解码器小 100x |
| lambda_consistency | 0.1 | 中 | 范围 0.01-0.5 |
| pos_weight | 10 | 高 | 按实际正负比例调 |
| batch_size | 8-16 | 低 | 越大 BG-SG 配对越多 |

### 如何判断模型在"学习"

- **前 5 个 epoch**：Dice 应该从接近 0 开始爬升，如果卡在 0 看 pos_weight
- **10-20 epoch**：一致性损失应该开始下降，说明 BG-SG 预测开始协调
- **收敛标志**：验证集 Dice 连续 5 个 epoch 不再上升

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| NCCT 缺血性卒中分割 | 增强 CT / MRI（窗宽逻辑完全不同） |
| 有解剖层标注（BG/SG）的数据集 | 数据集没有层级标注 |
| 小数据集（DINOv3 冻结骨干抗过拟合） | 数据量超过 10k，可以考虑端到端训练 |
| 需要零推理额外开销 | 训练资源极度受限（DINO 特征提取慢） |

---

## 我的观点

**TAGL 的贡献是真实的，但不神奇。** 6.9% 的 Dice 提升来自一个很朴素的 insight：把临床先验编码进损失函数。这件事本身并不新鲜（医学图像分割领域有大量类似工作），但做在 DINOv3 冻结骨干上、且保持推理零开销，是有实际价值的工程选择。

**值得关注的点：**

1. **冻结骨干 + 轻量解码器**的范式在医学小数据集上越来越主流，本文是个不错的参考实现
2. TAGL 的一致性约束思路可以推广到其他有解剖层级结构的任务（如腹部多器官分割）
3. 私有数据集的结果无法独立复现，AISD 上 0.6385 的 Dice 才是可以参考的公允数字

**局限：**

- 文章没有报告多随机种子结果，单次实验的 Dice 波动可能在 ±0.02
- BG-SG 配对依赖切片级别的解剖标注，实际落地时这个标注成本不低
- DINOv3 是自然图像预训练，医学领域迁移效果因数据分布差异而不稳定

**如果你在做类似项目**，建议先用 U-Net baseline 跑通流程，确认数据预处理（窗宽窗位）正确，再引入 TAGL——别把基础做不好的锅甩给算法。