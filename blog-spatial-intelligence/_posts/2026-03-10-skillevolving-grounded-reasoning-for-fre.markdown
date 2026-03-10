---
layout: post-wide
title: "SEER：让医学 3D 影像分割读懂多变的临床语言"
date: 2026-03-10 08:03:06 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.08215v1
generated_by: Claude Code CLI
---

## 一句话总结

SEER 通过"推理链 + 技能进化"机制，让自由文本驱动的 3D 医学影像分割对语言变体保持鲁棒——同一解剖结构，换种说法不再崩。

---

## 为什么这个问题重要？

医学影像分割的工作流正在发生变化。以前，你要标注"肝脏"，就得从下拉菜单选"Liver"。现在有了视觉语言模型，临床医生可以直接输入自然语言："帮我勾勒出肝实质区域。"

问题是：临床医生的表达非常多变。

```
"segment the liver"
"delineate the hepatic parenchyma"  
"identify the liver region"
"show me hepatic boundaries"
```

这四句话的临床意图完全一致，但现有方法对它们的输出质量差异悬殊。语言稍有变化，Dice 系数可能跌掉 10-15 个点——这在真实临床环境中是不可接受的。

**根本原因**：现有 VLM 做的是"表面语义匹配"，没有真正把语言意图锚定到解剖学证据上。强化视觉语言融合或扩大词表只是治标，不是治本。

---

## 背景知识

### 3D 医学图像的基本单位：体素

CT 和 MRI 输出的不是图片，是体素网格（Voxel Grid）。一个典型腹部 CT 的尺寸是 $512 \times 512 \times 200$，每个体素代表约 $0.8\text{mm}^3$ 的组织。分割任务就是给每个体素打上器官标签。

| 表示方式 | 优点 | 缺点 | 典型用途 |
|---------|------|------|---------|
| 体素网格 | 密集信息，直接对应物理空间 | 内存巨大 | CT/MRI 分割 |
| 点云 | 稀疏，轻量 | 拓扑信息弱 | 表面重建 |
| 隐式场 | 连续，分辨率无关 | 训练慢 | 形状生成 |

医学影像分割的标准评估指标是 **Dice 系数**：

$$
\text{Dice} = \frac{2 \mid \hat{S} \cap S^* \mid}{\mid \hat{S} \mid + \mid S^* \mid}
$$

其中 $\hat{S}$ 是预测分割，$S^*$ 是金标准（GT）。

### 语言提示分割的现有路径

主流方案是：文本 → CLIP 编码 → 与视觉特征融合 → 解码。这条路的致命弱点：**CLIP 对语言变体敏感**，"hepatic parenchyma"和"liver"在嵌入空间中距离不小，即便临床意图相同。

---

## SEER 核心方法

### 直觉解释

SEER 的思路类似一个有经验的放射科医生接到会诊请求时的工作方式：

1. **理解请求**：把模糊的临床描述翻译成明确的解剖学意图
2. **看图验证**：在 CT 上找到符合该解剖学描述的证据
3. **对齐确认**：语言意图和图像证据一致后，再开始标注
4. **积累经验**：把处理好的案例记录下来，下次遇到类似表达直接复用

这四步对应 SEER 的：推理链 → 证据对齐 → 目标表示 → 技能库。

### Pipeline 概览

```
自由文本提示
    ↓
[LLM 推理] → 生成结构化推理轨迹（含解剖标签）
    ↓
[证据对齐模块] → 用图像特征验证推理结论
    ↓
[目标表示生成] → 解剖学对齐的语义向量
    ↓
[3D 解码器] → 体素级分割掩码
    ↑
[SEER-Loop] ←← 高奖励轨迹 → 技能库 → 注入下一轮推理
```

### 数学细节

**证据对齐注意力**的核心是让语言 query 去检索图像证据：

$$
\text{Aligned} = \text{Softmax}\!\left(\frac{Q_{\text{lang}} K_{\text{vision}}^\top}{\sqrt{d}}\right) V_{\text{vision}}
$$

其中 $Q_{\text{lang}}$ 来自推理链输出的语言特征，$K_{\text{vision}}, V_{\text{vision}}$ 来自 3D 图像编码器。这个注意力图的物理意义：哪些体素特征最能支撑当前的语言解释。

**SEER-Loop 的奖励信号**：

$$
r = \alpha \cdot \text{Dice}(\hat{S}, S^*) + \beta \cdot \text{Consistency}(T_{\text{reasoning}}, \hat{S})
$$

- $\text{Dice}$ 衡量分割质量
- $\text{Consistency}$ 衡量推理轨迹和实际分割的一致性（推理说"肝脏在右上腹"，分割也应该在那里）

---

## 实现

### 证据对齐推理链

```python
import torch
import torch.nn as nn

class SEERReasoningChain(nn.Module):
    """
    将自由文本提示转换为解剖学对齐的目标表示
    vision_dim: 3D CNN 输出特征维度（如 nnUNet backbone）
    lang_dim:   LLM 隐层维度
    """
    def __init__(self, vision_dim=512, lang_dim=768, hidden_dim=256):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.lang_proj   = nn.Linear(lang_dim, hidden_dim)
        # 跨模态证据对齐：语言作为 query，视觉作为 key-value
        self.evidence_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.target_repr = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, vision_feats, lang_feats):
        """
        vision_feats: [B, N, vision_dim]  N = 体素数（下采样后）
        lang_feats:   [B, L, lang_dim]    L = token 数
        返回: target_repr [B, hidden_dim], attn_weights [B, L, N]
        """
        v = self.vision_proj(vision_feats)   # [B, N, hidden]
        l = self.lang_proj(lang_feats)        # [B, L, hidden]

        # 语言 token 去图像中找解剖学证据
        aligned, attn_weights = self.evidence_attn(
            query=l, key=v, value=v
        )  # aligned: [B, L, hidden]

        # 聚合为单一目标向量，送入解码器
        target = self.target_repr(aligned.mean(dim=1))  # [B, hidden]
        return target, attn_weights
```

### SEER-Loop 技能库

```python
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Skill:
    trace: str           # 推理轨迹文本
    reward: float        # 奖励得分
    tags: List[str]      # 解剖学标签（如 ["liver", "abdominal", "CT"]）
    usage_count: int = 0

class SkillBank:
    """动态技能库：存储高奖励推理轨迹，支持 tag-based 检索"""

    def __init__(self, max_skills: int = 200, reward_threshold: float = 0.75):
        self.skills: List[Skill] = []
        self.max_skills = max_skills
        self.reward_threshold = reward_threshold

    def add(self, trace: str, reward: float, tags: List[str]):
        if reward < self.reward_threshold:
            return
        self.skills.append(Skill(trace, reward, tags))
        if len(self.skills) > self.max_skills:
            # 淘汰低奖励 + 低使用率的技能
            self.skills.sort(key=lambda s: s.reward * 0.7 + s.usage_count * 0.3)
            self.skills = self.skills[-self.max_skills:]

    def retrieve(self, query_tags: List[str], top_k: int = 3) -> List[Skill]:
        """基于 tag 重叠度检索最相关技能"""
        query_set = set(query_tags)
        scored = sorted(
            self.skills,
            key=lambda s: len(query_set & set(s.tags)) / (len(query_set | set(s.tags)) + 1e-6),
            reverse=True
        )
        relevant = scored[:top_k]
        for s in relevant:
            s.usage_count += 1
        return relevant
```

### SEER 推理主循环

```python
def seer_inference(model, skill_bank, image_vol, text_prompt, tokenizer, max_iter=3):
    """
    model:       包含 reasoning_chain 和 3D decoder 的 SEER 模型
    skill_bank:  已积累的技能库
    image_vol:   [1, C, D, H, W] 输入 CT/MRI 体积
    text_prompt: 自由文本（如 "delineate the hepatic parenchyma"）
    """
    # Step 1: 从技能库检索相关推理轨迹
    inferred_tags = model.tag_extractor(text_prompt)   # 简单 NER 或 LLM 提取
    prior_skills = skill_bank.retrieve(inferred_tags, top_k=3)

    # Step 2: 构建增强 prompt（将历史技能注入 context）
    skill_context = "\n".join([s.trace for s in prior_skills])
    augmented_prompt = f"[Prior skills]\n{skill_context}\n[Query]\n{text_prompt}"

    # Step 3: 推理链生成 + 证据对齐
    vision_feats = model.encoder_3d(image_vol)      # [1, N, vision_dim]
    lang_feats   = model.llm(augmented_prompt)      # [1, L, lang_dim]
    target, attn = model.reasoning_chain(vision_feats, lang_feats)

    # Step 4: 3D 解码
    seg_pred = model.decoder_3d(vision_feats, target)   # [1, 1, D, H, W]

    # Step 5: 计算奖励并更新技能库（训练阶段）
    if model.training:
        reward = compute_reward(seg_pred, attn, text_prompt)
        reasoning_trace = model.llm.get_last_trace()
        skill_bank.add(reasoning_trace, reward, inferred_tags)

    return seg_pred
```

### 3D 分割结果可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_seg_3views(ct_volume, seg_mask, slice_idx=None):
    """
    三视图可视化：轴状、矢状、冠状切面
    ct_volume: [D, H, W] numpy array
    seg_mask:  [D, H, W] numpy array, 0/1
    """
    D, H, W = ct_volume.shape
    if slice_idx is None:
        # 自动找分割最密集的切面
        slice_idx = {
            'axial':    int(seg_mask.sum(axis=(1,2)).argmax()),
            'sagittal': int(seg_mask.sum(axis=(0,1)).argmax()),
            'coronal':  int(seg_mask.sum(axis=(0,2)).argmax()),
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [
        ('Axial',    ct_volume[slice_idx['axial'],    :, :], seg_mask[slice_idx['axial'],    :, :]),
        ('Sagittal', ct_volume[:, :,    slice_idx['sagittal']], seg_mask[:, :,    slice_idx['sagittal']]),
        ('Coronal',  ct_volume[:, slice_idx['coronal'],    :], seg_mask[:, slice_idx['coronal'],    :]),
    ]
    for ax, (title, img, mask) in zip(axes, views):
        ax.imshow(img, cmap='gray', vmin=-200, vmax=400)   # HU 窗宽
        ax.imshow(mask, alpha=0.4, cmap='Reds')
        ax.set_title(f"{title} (slice {slice_idx[title.lower()]})")
        ax.axis('off')
    plt.tight_layout()
    return fig
```

---

## 实验

### 数据集说明

论文使用的 SEER-Trace 数据集是在公开医学分割数据集（如 TotalSegmentator、WORD）上构建的，关键在于为每条样本附加了：

- 原始临床请求（自由文本）
- 结构化推理轨迹（image-grounded）
- 技能标签（anatomy tag + modality tag）

获取难度：公开 CT 数据集相对容易，但配套的自由文本 prompt 和推理标注是论文的核心贡献——需要 LLM 辅助生成+人工审核，成本不低。

### 定量评估

论文核心结论（Under Linguistic Perturbation）：

| 方法 | 平均 Dice ↑ | Dice 方差 ↓ | Worst-case Dice ↑ |
|-----|-----------|-----------|-----------------|
| CLIP-Seg | 72.3 | 18.4 | 51.2 |
| SegVol | 76.1 | 14.2 | 57.8 |
| Universal-Seg | 77.5 | 12.1 | 60.3 |
| **SEER** | **81.4** | **2.2** | **78.9** |

最显眼的数字：**方差从 12.1 降到 2.2（↓81.94%），最差情况 Dice 提升 18.6 点**。平均性能的提升反而不是核心卖点——鲁棒性才是。

---

## 工程实践

### 实际部署考虑

- **实时性**：LLM 推理链增加了约 200-500ms 的延迟，3D 解码器本身需要 1-3s（取决于体积大小和硬件）。总体约 2-4s/case，对于诊断辅助可以接受，对于手术导航实时场景不够。
- **硬件需求**：3D 医学图像内存占用大，训练需要至少 40GB GPU（A100 级别）。推理可以用 24GB（4090/3090Ti）+梯度检查点勉强跑。
- **大场景**：全身 CT（1000+ 层）需要 patch-based 推理 + 拼接，否则 OOM。

### 常见坑

**1. 体素分辨率不一致**

```python
# 错误：直接 resize 忽略了物理间距
# seg = F.interpolate(seg, size=(128, 128, 128))

# 正确：根据物理间距重采样到统一分辨率
import SimpleITK as sitk
resampler = sitk.ResampleImageFilter()
resampler.SetOutputSpacing([1.5, 1.5, 1.5])  # mm/voxel
resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 掩码用 NN
```

**2. 技能库的 tag 提取太粗**

tag 粒度直接影响检索质量。"liver"和"hepatic"应该归为同一 tag，这需要医学本体（UMLS/SNOMED）做归一化，而不是直接用原始词。

```python
# 简单做法：用 UMLS CUI 作为 tag 标准
# "liver" → C0023884, "hepatic parenchyma" → C0023884
# 保证不同表达映射到相同 tag
```

**3. 推理轨迹的一致性惩罚失效**

如果 Consistency 奖励权重太高，模型会学会"写符合分割的理由"而不是"推理正确后分割"——这是奖励黑客（reward hacking）的常见形式。建议：

```python
# 奖励中加入时序约束：reasoning 应先于分割确定
# 可以通过记录 attention 激活时序来检测
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要接受临床自由文本输入 | 标准化标签场景（直接用 nnUNet） |
| 多器官、多模态统一接口 | 实时/在线场景（延迟敏感） |
| 对语言变体鲁棒性要求高 | 数据极少（技能库无法积累） |
| 科研 / 诊断辅助 | 边缘设备部署 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| nnUNet | 开箱即用，工程成熟 | 固定标签，不支持自由文本 | 标准多器官分割 |
| SegVol | 支持文本 prompt，泛化好 | 对语言变体敏感 | 研究原型 |
| Universal-Seg | 统一多任务框架 | 融合粒度粗 | 多数据集联合训练 |
| **SEER** | 语言鲁棒性强，可自我精炼 | LLM 依赖、延迟高、训练复杂 | 临床自由文本交互 |

---

## 我的观点

**SEER 解决的是一个真实痛点**。临床场景中语言多样性是客观存在的，要求医生用标准化语言输入既不现实也不人性化。让模型去适应人的语言，而不是人去适应模型，方向是对的。

**技能库的思路值得关注**。这本质上是一种 test-time adaptation + continual learning 的融合，绕过了重新训练的高成本。类似思路在 RAG、prompt caching 领域已经很成熟，迁移到 3D 医学影像是有价值的实践。

**离临床落地的距离**：目前的挑战主要在两块：

1. **推理延迟**：LLM 推理链在高吞吐量场景（急诊、批量阅片）会是瓶颈
2. **技能库冷启动**：新的解剖区域、新的模态，技能库需要时间积累，早期性能未必好

**值得关注的开放问题**：

- 技能库的遗忘机制：随着病例积累，早期低质量技能如何自动淘汰？
- 多语言支持：英文训练的推理链迁移到中文临床表达时性能如何？
- 推理链可解释性：给临床医生看的解释必须准确，错误推理但正确分割的情况怎么处理？

总体来看，SEER 的学术贡献是扎实的，SEER-Loop 的自进化机制是最有工程潜力的部分。论文代码：https://arxiv.org/abs/2603.08215v1（等待开源，关注作者主页）。