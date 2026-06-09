---
layout: post-wide
title: "开放集医学图像分割的主动无源域适应：三重困境下的优雅破局"
date: 2026-06-09 12:02:33 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.08749v1
generated_by: Claude Code CLI
---

## 一句话总结

在没有源域数据、目标域含未知类别的双重约束下，用少量标注样本完成医学图像分割的跨域迁移——通过分解不确定性和原型差异来精准选样。

---

## 为什么这个问题难到令人头疼？

想象这样一个真实场景：你在 A 医院的 CT 数据上训练了一个肝脏分割模型，现在要部署到 B 医院。麻烦接踵而至：

1. **域偏移**：B 医院用的是不同品牌的扫描仪，图像风格差异显著
2. **隐私墙**：A 医院的数据由于患者隐私合规无法带走（无源域数据）
3. **新病种**：B 医院的数据里有 A 医院没有的肿瘤类型（开放集问题）
4. **标注贵**：你只有预算让放射科医生标注 B 医院 5% 的数据（主动学习预算）

现有方法通常只解决这四个问题中的一两个。ASFOSDA（Active Source-free Open-set Domain Adaptation）试图同时应对全部四个挑战。

**核心洞见**：样本选择不是一维问题。一个好的候选样本应该同时满足：模型对它不确定（信息量大）、它跟已标注样本不重复（多样性够）、它可能包含未知类别（探测新类）。CDU 负责前者，CPD 负责后者两点。

---

## 核心方法解析

### 1. CDU：知道自己不知道什么

CDU（Class-aware Decomposed Uncertainty）的核心是把预测不确定性分解成两部分：

**直觉**：医生拿到一张模糊 CT 片会说"这个区域怎么看都模糊"（认知不确定性，数据本身的问题），或者"这个区域超出了我的知识范围"（知识不确定性，模型没见过类似的）。两种原因需要不同的对策。

做法是**测试时数据增强（TTA）**：对同一张图做 $T$ 次随机扰动（随机裁剪、翻转、亮度抖动），分别预测，再聚合。

设 $\{\hat{y}^{(t)}\}_{t=1}^T$ 为 $T$ 次 TTA 预测结果，每次预测给出像素级的类别概率。

**认知不确定性（Aleatoric）** — 平均每次预测的熵：

$$U_{al}(x) = \frac{1}{T} \sum_{t=1}^{T} H[\hat{p}^{(t)}(y \mid x)]$$

**知识不确定性（Epistemic）** — 总熵减去认知熵：

$$U_{ep}(x) = H\!\left[\frac{1}{T}\sum_{t=1}^{T} \hat{p}^{(t)}(y \mid x)\right] - U_{al}(x)$$

其中 $H[p] = -\sum_c p_c \log p_c$ 是香农熵。

"Class-aware"体现在：对已知类别和未知类别（开放集）分别聚合这两种不确定性，让查询策略感知到"这是已知类不确定"还是"这可能是新类"。

### 2. CPD：不要挑重复的样本

CPD（Class-agnostic Prototype Discrepancy）解决多样性问题，分两个维度：

**跨域差异（Cross-domain）**：衡量目标样本与源域原型（source prototypes，迁移前模型的类中心嵌入）的距离。距离越大，说明这个样本越"陌生"，越值得标注。

$$D_{cross}(x) = \min_{c \in \mathcal{C}_s} \|f(x) - \mathbf{p}_c^s\|_2$$

**自域差异（Self-domain）**：衡量候选样本与已选样本集 $\mathcal{Q}$ 的距离。越远越不重复，多样性越好：

$$D_{self}(x, \mathcal{Q}) = \min_{q \in \mathcal{Q}} \|f(x) - f(q)\|_2$$

最终查询分数是四项的加权组合：

$$S(x) = \alpha \cdot U_{al} + \beta \cdot U_{ep} + \gamma \cdot D_{cross} + \delta \cdot D_{self}$$

---

## 动手实现

### CDU 核心实现

```python
import torch
import torch.nn.functional as F

def tta_predict(model, image, n_augments=10):
    model.train()  # 开启 dropout，增加随机性
    predictions = []
    with torch.no_grad():
        for _ in range(n_augments):
            # ... (数据增强代码省略)
            logits = model(image)
            predictions.append(F.softmax(logits, dim=1))  # [B, C, H, W]
    model.eval()
    return torch.stack(predictions, dim=0)  # [T, B, C, H, W]


def compute_cdu(model, image, n_augments=10, known_classes=None):
    """计算 Class-aware Decomposed Uncertainty"""
    preds = tta_predict(model, image, n_augments)  # [T, B, C, H, W]

    # ... (已知/未知类别掩码及重归一化省略)
    preds_known = preds

    mean_pred = preds_known.mean(dim=0)  # [B, C, H, W]
    eps = 1e-8

    # 认知不确定性：各次预测熵的平均
    per_aug_entropy = -(preds_known * (preds_known + eps).log()).sum(dim=2)  # [T, B, H, W]
    al_unc = per_aug_entropy.mean(dim=0)  # [B, H, W]

    # 知识不确定性：均值预测的熵 - 认知熵
    mean_entropy = -(mean_pred * (mean_pred + eps).log()).sum(dim=1)  # [B, H, W]
    ep_unc = (mean_entropy - al_unc).clamp(min=0)

    return al_unc.mean(dim=[-1, -2]), ep_unc.mean(dim=[-1, -2])  # [B], [B]
```

### CPD 核心实现

```python
def build_source_prototypes(model, source_loader, num_classes, device):
    """从源模型提取类别原型"""
    prototype_sum = torch.zeros(num_classes, model.feature_dim).to(device)
    prototype_count = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for images, labels in source_loader:
            features = model.encode(images.to(device))  # [B, D, H, W]
            for c in range(num_classes):
                mask = (labels.to(device) == c).float().unsqueeze(1)
                prototype_sum[c] += (features * mask).sum(dim=[0, 2, 3])
                prototype_count[c] += mask.sum()
    
    return prototype_sum / (prototype_count.unsqueeze(1) + 1e-8)  # [C, D]


def compute_cpd(model, candidate_images, source_prototypes, selected_features=None):
    """计算 Class-agnostic Prototype Discrepancy"""
    with torch.no_grad():
        feat_global = model.encode(candidate_images).mean(dim=[-1, -2])  # [B, D]
    
    # 跨域差异：到最近源类原型的距离
    cross_domain_disc = torch.cdist(feat_global, source_prototypes).min(dim=1).values

    # 自域差异：到已选样本的最小距离（越大越多样）
    if selected_features:
        self_domain_disc = torch.cdist(feat_global, torch.stack(selected_features)).min(dim=1).values
    else:
        self_domain_disc = torch.zeros(len(candidate_images))
    
    return cross_domain_disc, self_domain_disc, feat_global
```

### 主动查询策略

```python
def active_query(model, target_loader, source_prototypes, budget,
                 num_augments=10, known_classes=None,
                 alpha=0.3, beta=0.3, gamma=0.2, delta=0.2):
    all_scores, all_features = {}, {}

    # 第一轮：计算所有候选样本的 CDU + cross-domain CPD
    for images, _, indices in target_loader:
        al, ep = compute_cdu(model, images, n_augments=num_augments, known_classes=known_classes)
        cross, _, feats = compute_cpd(model, images, source_prototypes)
        for i, idx in enumerate(indices):
            all_scores[idx.item()] = {'al': al[i].item(), 'ep': ep[i].item(), 'cross': cross[i].item()}
            all_features[idx.item()] = feats[i].detach().cpu()

    # 归一化各分量到 [0, 1]
    for key in ['al', 'ep', 'cross']:
        vals = torch.tensor([s[key] for s in all_scores.values()])
        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        for i, idx in enumerate(all_scores.keys()):
            all_scores[idx][f'{key}_norm'] = vals[i].item()

    # 贪心迭代：每轮选分最高样本，self-domain 差异随已选集合动态更新
    selected_indices, selected_features = [], []
    for _ in range(budget):
        best_score, best_idx = -1, None
        for idx, s in all_scores.items():
            if idx in selected_indices:
                continue
            feat = all_features[idx].unsqueeze(0)
            self_disc_norm = (torch.cdist(feat, torch.stack(selected_features)).min().item()
                              if selected_features else 1.0)
            score = alpha * s['al_norm'] + beta * s['ep_norm'] + gamma * s['cross_norm'] + delta * self_disc_norm
            if score > best_score:
                best_score, best_idx = score, idx
        selected_indices.append(best_idx)
        selected_features.append(all_features[best_idx])

    return selected_indices
```

### 目标域自训练（Target-refined Self-training）

```python
def self_training_step(model, labeled_loader, unlabeled_loader, optimizer,
                       conf_threshold=0.85, lambda_u=0.5):
    model.train()
    for (x_l, y_l), (x_u, _) in zip(labeled_loader, unlabeled_loader):
        # 有监督损失
        loss_sup = F.cross_entropy(model(x_l), y_l, ignore_index=255)

        # 生成伪标签（仅保留高置信度像素）
        with torch.no_grad():
            probs_u = F.softmax(model(x_u), dim=1)
            max_prob, pseudo_label = probs_u.max(dim=1)
            mask = (max_prob > conf_threshold).long()

        # 无监督损失（置信度加权）
        loss_unsup = (F.cross_entropy(model(x_u), pseudo_label, reduction='none') * mask).mean()

        (loss_sup + lambda_u * loss_unsup).backward()
        optimizer.step(); optimizer.zero_grad()
```

### 实现中的坑

**坑 1：TTA 时忘记还原 model.train() 的影响**

```python
# 错误：用完 TTA 后忘记切换回 eval
model.train()  # 开 dropout
_ = tta_predict(...)
model.predict(test_data)  # 此时 dropout 还开着！

# 正确：在 tta_predict 内部管理状态（见上面实现）
```

**坑 2：原型归一化导致 cross-domain 距离失真**

```python
# 特征归一化必须在源域和目标域使用相同的统计量
# 不能各自归一化，否则距离无意义
feat_target = (feat_target - source_mean) / source_std  # 用源域统计量
```

**坑 3：自训练的确认偏差（Confirmation Bias）**

伪标签错误会被模型持续强化。缓解方法：
- 提高 `conf_threshold`（0.85–0.95），宁缺毋滥
- 对已知类和开放集类分别设阈值
- 每隔若干轮重新生成伪标签，而不是固定一次

---

## 论文说的 vs 现实

| 方面 | 论文声称 | 现实考量 |
|------|---------|---------|
| 少量标注（5–10%）即超越有监督基线 | 在特定数据集上成立 | 高度依赖 budget 分配和数据集多样性 |
| TTA 增强不确定性估计 | 显著提升 CDU 质量 | TTA 每次推理 10x 成本，3D 体积数据时极慢 |
| 开放集检测 | CPD 能找到新类样本 | 新类与已知类视觉相似时，CPD 容易失效 |
| 无需源数据 | 仅需冻结的源模型和原型 | 原型存储需要提前规划，不是真正"零源信息" |

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 跨医院/跨设备部署，源数据受 HIPAA/GDPR 保护 | 源数据可自由使用，无隐私顾虑 |
| 目标域可能出现新的解剖结构或病理类型 | 已知目标域类别集合与源域完全相同 |
| 标注预算极度受限（<10% 目标数据） | 可以标注大量目标数据 |
| 3D 体积分割（CT、MRI），域偏移主要来自设备差异 | 实时推理场景（TTA 计算开销不可接受） |
| 有能力保存并传递源模型权重和原型 | 连源模型都无法访问（完全盲迁移） |

---

## 我的观点

**这个方法的真正价值**在于它诚实地面对了临床部署的实际约束，而不是在清洁实验室条件下优化一个简化问题。同时处理隐私、开放集和标注成本，这三个约束在现实中经常同时出现，偏偏之前的文献喜欢逐一单独解决。

**令我存疑的地方**：CDU 的权重参数 $\alpha, \beta, \gamma, \delta$ 如何调？论文可能通过交叉验证给出了默认值，但这四个权重对目标域非常敏感，实际部署时需要额外的调优策略。另外，TTA 的开销在 3D 分割任务中是实实在在的工程障碍——10 次增强意味着单次查询评估需要 10 倍推理时间，面对数千个候选 volume 时不是小事。

**更值得关注的方向**：能否用更轻量的不确定性估计（如 evidential deep learning 或 conformal prediction）替代 TTA，在保留 CDU 精度的同时大幅降低计算成本？这是这条技术路线最值得跟进的下一步。