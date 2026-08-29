---
layout: post-wide
title: "AnatoProto：用解剖先验和序列内原型监督解决胎儿超声盲扫中的极端类别不平衡问题"
date: 2026-08-29 08:03:12 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.27051v1
generated_by: Claude Code CLI
---

读取记忆文件，了解用户背景和写作偏好。


## 一句话总结

在产科盲扫超声序列中，正样本帧不足 3%，AnatoProto 通过冻结 BiomedCLIP 特征 + 解剖区域加权池化 + 序列内原型损失三者协同，将标准切面检测 F1 从 54.52 提升至 67.72。

## 为什么这个问题重要？

### 产科盲扫的现实约束

在资源匮乏地区，超声检查通常由非专业人员执行"盲扫"（blind sweep）：探头按固定轨迹扫过腹部，摄像机全程录制，事后再由算法或远端专家找出关键切面。这种模式极大降低了产前检查的门槛，但也带来一个棘手问题：

- 一次扫描产生数百帧图像
- 腹围（AC）标准切面只出现在其中 **2–3%** 的连续帧内
- 标准切面的判定依赖腹部解剖结构（肝脏、胃泡、脐静脉）的几何关系

这是一个比普通类别不平衡更难的问题：正负样本比可达 1:50，且正样本必须形成时间上连续的短片段。

### 基础模型在这里为何失效？

BiomedCLIP、FetalCLIP 这类医学视觉基础模型在单帧分类上表现优秀，但在盲扫场景下有两个结构性缺陷：

1. **空间聚合方式不对**：CLS token 或均匀平均池化会把背景噪声和解剖区域等权处理
2. **时序结构未利用**：每帧独立预测，没有利用"同一次扫描的正样本帧彼此相似"这一强先验

AnatoProto 的核心贡献就是用极轻量的适配层解决这两个问题，同时保持基础模型的冻结状态（节省训练成本）。

---

## 背景知识

### 医学超声的视觉表示

超声图像与自然图像有本质差异：

| 特征 | 自然图像 | 超声图像 |
|------|---------|---------|
| 纹理 | 丰富、确定 | 斑点噪声（speckle） |
| 解剖结构 | 任意 | 严格的几何约束 |
| 标注难度 | 低 | 需要专科培训 |

正因如此，超声分析需要**解剖学约束**作为先验——不能只靠像素级特征。

### 类原型学习的直觉

原型学习（Prototype Learning）来自少样本分类：用同类样本的均值向量作为"原型"，训练时拉近类内距离。AnatoProto 将这个思路迁移到时序场景：**同一次扫描内的正样本帧**共享相同的解剖切面，它们的嵌入应该相互靠近。

---

## 核心方法

### 直觉解释

想象你在看一段扫描视频，脑中快速过一遍所有帧：

1. **解剖加权池化**：你的视线自动聚焦到腹部区域，忽略皮肤和背景——这就是 nnU-Net 分割概率图做的事
2. **序列内原型**：你记住了"这次扫描中出现过的标准切面长什么样"，用这个记忆来校准当前帧的判断
3. **级联精炼**：先逐帧判断，再合并连续片段，最后对整个 case 做全局检验

### 数学细节

**解剖加权空间池化**

BiomedCLIP（ViT 架构）产生 $N$ 个 patch token，记为 $\{f_i\}_{i=1}^N$，$f_i \in \mathbb{R}^D$。

nnU-Net 输出腹部区域概率图 $M \in \mathbb{R}^{H \times W}$，下采样到 patch 网格后得到权重 $w_i$：

$$
\hat{f} = \sum_{i=1}^{N} \tilde{w}_i \cdot f_i, \quad \tilde{w}_i = \frac{w_i}{\sum_j w_j}
$$

这替代了原始的 CLS token 或均匀平均池化，使语义特征聚合到解剖意义区域。

**序列内原型损失**

对于第 $k$ 次扫描，设正样本帧集合为 $\mathcal{P}_k$，原型为：

$$
\mu_k = \frac{1}{|\mathcal{P}_k|} \sum_{t \in \mathcal{P}_k} \hat{f}_t
$$

原型损失拉近正样本帧与其原型：

$$
\mathcal{L}_{proto} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{|\mathcal{P}_k|} \sum_{t \in \mathcal{P}_k} \|\hat{f}_t - \text{sg}(\mu_k)\|^2
$$

其中 $\text{sg}(\cdot)$ 表示 stop-gradient，避免梯度通过原型反传。

总损失为：
$$
\mathcal{L} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{proto}
$$

### Pipeline 概览

```
盲扫视频
    │
    ├─→ BiomedCLIP (冻结) ──→ patch tokens {f_i}
    │
    ├─→ nnU-Net (冻结)   ──→ 解剖概率图 M
    │
    └─→ 解剖加权池化 ──→ 帧嵌入 f̂_t
                              │
                    序列内原型损失 (训练期间)
                              │
                    混合预测头 (稳定性 + 边界)
                              │
                    三阶段级联精炼
                    帧级 → 片段级 → Case 级拒识
                              │
                         最终预测
```

---

## 实现

### 环境配置

```bash
pip install torch torchvision transformers open-clip-torch nnunetv2
```

### 核心代码

**组件一：解剖加权空间池化**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomyWeightedPooling(nn.Module):
    def __init__(self, patch_grid_size=16):
        super().__init__()
        self.patch_grid_size = patch_grid_size

    def forward(self, patch_tokens: torch.Tensor, anatomy_probs: torch.Tensor):
        """
        patch_tokens:  (B, N, D) — BiomedCLIP patch token 序列
        anatomy_probs: (B, H, W) — nnU-Net 腹部区域概率图，值域 [0,1]
        返回:          (B, D)    — 解剖加权帧嵌入
        """
        B, N, D = patch_tokens.shape
        h = w = self.patch_grid_size  # 假设方形 patch 网格

        # 将解剖概率图下采样到 patch 分辨率
        weights = F.adaptive_avg_pool2d(
            anatomy_probs.unsqueeze(1).float(), (h, w)
        ).squeeze(1)           # (B, h, w)
        weights = weights.view(B, N)  # (B, N)

        # softmax 归一化权重（避免解剖区域面积差异影响量级）
        weights = F.softmax(weights * 5.0, dim=1)  # temperature=5 增强对比度

        # 加权求和
        pooled = torch.einsum('bn,bnd->bd', weights, patch_tokens)
        return pooled
```

**组件二：序列内原型损失**

```python
class WithinCasePrototypeLoss(nn.Module):
    def __init__(self, stop_gradient=True):
        super().__init__()
        self.stop_gradient = stop_gradient

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                case_ids: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (N, D) — 当前 batch 所有帧的嵌入
        labels:     (N,)   — 二值标签，1=标准切面帧
        case_ids:   (N,)   — 每帧所属的扫描序列编号
        """
        total_loss = torch.tensor(0.0, device=embeddings.device)
        count = 0

        for case_id in case_ids.unique():
            mask = case_ids == case_id
            case_embs = embeddings[mask]
            case_labels = labels[mask]

            pos_mask = case_labels == 1
            if pos_mask.sum() == 0:
                continue  # 该 case 无正样本，跳过

            # 正样本原型（stop-gradient 防止梯度流回原型）
            prototype = case_embs[pos_mask].mean(dim=0)
            if self.stop_gradient:
                prototype = prototype.detach()

            # 拉近正样本帧与原型
            pos_embs = case_embs[pos_mask]
            loss = F.mse_loss(pos_embs,
                              prototype.unsqueeze(0).expand_as(pos_embs))
            total_loss += loss
            count += 1

        return total_loss / max(count, 1)
```

**组件三：混合预测头（稳定性 + 边界转换）**

```python
class HybridPredictionHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # 分支1：逐帧稳定性评分
        self.stability = nn.Linear(d_model, 1)
        # 分支2：帧间边界转换（1D 卷积捕捉局部时序模式）
        self.boundary_conv = nn.Conv1d(d_model, 1, kernel_size=5, padding=2)
        # 融合
        self.fuse = nn.Linear(2, 1)

    def forward(self, frame_embs: torch.Tensor) -> torch.Tensor:
        """
        frame_embs: (B, T, D) — 序列帧嵌入
        返回:       (B, T)    — 每帧的 logit
        """
        # 稳定性分支
        stab = self.stability(frame_embs)          # (B, T, 1)

        # 边界转换分支
        x = frame_embs.permute(0, 2, 1)            # (B, D, T)
        bound = self.boundary_conv(x).permute(0, 2, 1)  # (B, T, 1)

        logits = self.fuse(torch.cat([stab, bound], dim=-1)).squeeze(-1)
        return logits
```

**组件四：三阶段级联精炼**

```python
def cascade_refinement(frame_probs: torch.Tensor,
                        min_seg_len: int = 5,
                        seg_conf_thresh: float = 0.45) -> torch.Tensor:
    """
    三阶段级联：帧级 → 片段级 → Case 级拒识
    frame_probs: (T,) 每帧正类概率
    """
    T = len(frame_probs)
    frame_preds = (frame_probs > 0.5).long()

    # Stage 2: 合并连续正样本帧为片段，过滤短片段
    segment_preds = torch.zeros(T, dtype=torch.long)
    i = 0
    while i < T:
        if frame_preds[i] == 1:
            j = i
            while j < T and frame_preds[j] == 1:
                j += 1
            seg_len = j - i
            seg_conf = frame_probs[i:j].mean().item()
            # 片段须够长且置信度充足，才保留
            if seg_len >= min_seg_len and seg_conf >= seg_conf_thresh:
                segment_preds[i:j] = 1
            i = j
        else:
            i += 1

    # Stage 3: Case 级拒识 —— 若整段序列正样本过少，直接全部拒识
    pos_ratio = segment_preds.float().mean().item()
    if pos_ratio > 0.15:  # 超过15%被标为正样本，不符合先验，拒识
        return torch.zeros(T, dtype=torch.long)

    return segment_preds
```

**完整训练循环骨架**

```python
# 模型组装（BiomedCLIP 冻结，只训练适配层）
encoder = BiomedCLIPEncoder(freeze=True)     # 冻结基础模型
pooling = AnatomyWeightedPooling(patch_grid_size=16)
head    = HybridPredictionHead(d_model=512)

proto_loss_fn = WithinCasePrototypeLoss()
cls_loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0))  # 补偿不平衡

optimizer = torch.optim.AdamW(
    list(pooling.parameters()) + list(head.parameters()), lr=1e-4
)

for frames, anatomy_maps, labels, case_ids in dataloader:
    patch_tokens = encoder(frames)              # (B, N, D)
    embeddings   = pooling(patch_tokens, anatomy_maps)  # (B, D)

    # 构造时序序列（假设 batch 内同一 case 连续排列）
    logits = head(embeddings.unsqueeze(0)).squeeze(0)   # (B,)

    loss = (cls_loss_fn(logits, labels.float())
            + 0.5 * proto_loss_fn(embeddings, labels, case_ids))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 实验

### 数据集：ACOUSLIC-AI Benchmark

| 属性 | 值 |
|------|---|
| 任务 | 胎儿腹围标准切面检测 |
| 正样本比例 | ~2.5% |
| 评估指标 | F1（宽容帧级匹配） |
| 基线模型 | FetalCLIP, BiomedCLIP, TriDet |

### 定量结果

| 方法 | F1 | 备注 |
|------|-----|-----|
| FetalCLIP + PRS | 54.52 | 最强基础模型 baseline |
| TriDet + PRS | 51.96 | 最强视频动作检测 baseline |
| AnatoProto (仅原型损失) | ~42 | 原型损失单独使用反而降低 recall 12pt |
| AnatoProto (仅解剖池化) | ~61 | 单独使用已超越所有 baseline |
| **AnatoProto (完整)** | **67.72** | 两者协同，recall 提升 6.5pt |

### 关键发现：协同效应而非叠加

这是论文最有价值的分析结论，值得重点理解。

**原型损失单独使用时效果反而更差（recall 下降 12 点）**。

原因在于：原型损失的效果完全依赖于"原型质量"。如果嵌入特征本身噪声很大（未经解剖加权，背景干扰严重），那么正样本原型 $\mu_k$ 本身就不准确——把所有正样本帧往一个错误的方向拉，自然会损害性能。

只有当解剖加权池化让特征先聚焦到腹部区域，原型才具备足够的语义纯粹度，此时序列内原型损失才能发挥校准作用——这就是**符号翻转（sign-flip）**的几何解释。

---

## 工程实践

### 实际部署考虑

| 指标 | 情况 |
|------|------|
| 推理速度 | BiomedCLIP 冻结，适配层轻量，单帧 ~5ms (A100) |
| 内存占用 | 主要来自 BiomedCLIP (~400MB)，适配层 <10MB |
| nnU-Net 依赖 | 离线预处理即可，不需要实时分割 |

### 常见坑

**坑 1：正样本权重设置不当**

极端不平衡（1:50）必须在损失函数层面补偿，否则模型倾向于全预测为负：

```python
# pos_weight 告诉模型正样本的"权重"
pos_weight = torch.tensor([neg_count / pos_count])  # 约 40-50
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**坑 2：batch 采样策略错误**

随机采样 batch 会导致某个 batch 内完全没有正样本，原型损失退化为 0，梯度消失：

```python
# 使用 case-balanced sampler：每个 batch 保证包含若干正样本 case
from torch.utils.data import WeightedRandomSampler
# 按 case 级别的正样本比例赋权，而非帧级别
```

**坑 3：级联阈值需要针对序列长度校准**

不同扫描时长不同，固定阈值 `min_seg_len=5` 在短序列中会过滤掉真正的标准切面：

```python
# 使用相对长度而非绝对帧数
min_seg_len = max(3, int(total_frames * 0.01))  # 至少1%的序列长度
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 极端类别不平衡（正样本 <5%） | 正负样本相对均衡（可直接用分类头） |
| 有成熟分割模型提供解剖先验 | 缺乏可靠的空间先验 |
| 序列内正样本结构稳定（同类切面） | 正样本在同一序列内变化极大 |
| 基础模型计算资源充足 | 边缘设备实时推理（BiomedCLIP 太重） |
| 有 case 级别标注 | 只有帧级标注，无法构造序列内原型 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| FetalCLIP | 领域预训练，单帧强 | 不建模时序 | 单帧标准切面分类 |
| TriDet (视频动作检测) | 时序建模完整 | 大量正样本假设，对极端不平衡脆弱 | 动作识别、体育视频 |
| AnatoProto | 轻量适配，协同先验 | 依赖 nnU-Net 分割结果质量 | 医学盲扫、稀疏事件检测 |

---

## 我的观点

AnatoProto 最值得关注的贡献不是最终指标，而是那个**符号翻转的消融实验**：它清楚地说明了"两个 reasonable 的模块叠加不一定有效，必须分析模块间的前提依赖关系"。这在工程中是高频坑——很多时候单独验证有效的组件组合后反而退化，根因正是这种隐式依赖。

从方向上看，这类"冻结基础模型 + 轻量任务适配 + 结构先验"的范式会越来越重要。随着医学 AI 数据标注成本居高不下，如何最大化利用已有分割 / 检测模型的输出作为软先验，是比继续堆数据更现实的路径。

离实际部署还有一段距离：nnU-Net 的腹部分割本身需要在目标设备 / 数据分布上验证，而不同医院的超声机型、探头频率都会影响分割质量，进而影响原型损失的效果。这个链式依赖是工程化的主要挑战。

值得关注的开放问题：在**无标注**新 case 上的实时自适应（test-time prototype update）——如果能在推理阶段动态更新原型，对分布外数据的鲁棒性会有质的提升。