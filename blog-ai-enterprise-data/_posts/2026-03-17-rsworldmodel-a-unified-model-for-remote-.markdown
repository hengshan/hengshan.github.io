---
layout: post-wide
title: "遥感世界模型：用强化学习统一变化理解与未来场景预测"
date: 2026-03-17 08:02:42 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.14941v1
generated_by: Claude Code CLI
---

## 一句话总结

RS-WorldModel 用三阶段训练流程（地理感知预训练 → 协同指令微调 → 可验证强化优化）统一了遥感变化理解和文本引导未来场景生成，2B 参数在多数评测上超越 120 倍大的模型。其中最值得关注的是第三阶段的 VRO——一个将"可验证奖励 + RL"引入多模态遥感的实践。

---

## 背景：为什么需要这个方向？

遥感图像分析有两类典型任务：

**变化理解（Understanding）**：给定双时相图像，解释"发生了什么变化"，输出文本描述或分类标签。

**未来预测（Forecasting）**：给定历史图像和文本条件，生成"未来可能是什么样子"的图像。

这两个任务在表面上看完全不同——一个是判别式（图→文），一个是生成式（文+图→图）。但它们共享一个关键先验：**时空因果规律**。建筑拆迁后是空地，植被覆盖度在季节间周期变化，这些规律对两个任务都有用。

现有方法分开处理这两件事，导致：
- 变化理解模型学不到生成未来图像所需的时序先验
- 预测模型学不到从真实变化描述中抽取的语义约束
- 数据利用率低，同一批卫星图像要训练两个模型

RS-WorldModel 的核心 insight：**用一个共享骨干同时优化两个目标，让理解任务的监督信号反过来约束生成质量**。

---

## 三阶段训练流水线

整体架构是一个 2B 参数的多模态语言模型（文本 + 图像编码器），三个阶段逐步"解锁"能力：

```
Stage 1: GAGP  →  Stage 2: SIT  →  Stage 3: VRO
地理感知预训练    协同指令微调      可验证强化优化
（条件生成基础）  （双任务对齐）    （奖励驱动精炼）
```

### Stage 1：地理感知生成预训练（GAGP）

问题：卫星图像的"外观"强烈依赖于拍摄时间、地理位置、传感器参数。同一块土地在不同季节、不同分辨率下看起来可以完全不同。

解法：将地理元数据（经纬度、拍摄日期、分辨率）编码后拼接到条件向量，作为生成的显式条件。

$$
p_\theta(x_{t+1} \mid x_t, c_{\text{geo}}) = p_\theta(x_{t+1} \mid x_t, [\text{lat}, \text{lon}, \text{date}, \text{gsd}])
$$

这一步的目标是让模型知道"季节→植被状态"、"纬度→建筑风格"之类的地理先验，而不是学一个盲目的图像扩散模型。

### Stage 2：协同指令微调（SIT）

用 RSWBench-1.1M 数据集同时训练两个任务，输入格式统一为：

```
[双时相图像] + [指令文本] → [文本描述] 或 [生成图像]
```

"协同"的关键：loss 同时包含两个任务头，梯度更新共享骨干参数。

### Stage 3：可验证强化优化（VRO）

这是最有意思的部分。

---

## VRO：把 GRPO 用到遥感多模态模型上

### 直觉解释

SIT 之后，模型"大概知道"怎么做两个任务，但存在两个问题：
1. **格式不稳定**：输出可能缺少关键字段，或格式不符合评测要求
2. **幻觉**：对于变化理解，模型可能生成"看起来合理但不准确"的描述

VRO 的核心思想来自 GRPO（Group Relative Policy Optimization）——对于每个输入，采样多个输出，用可验证的规则打分，用分数差驱动策略更新，**不需要额外的奖励模型**。

这一路线因 DeepSeek-R1 而广为人知，RS-WorldModel 把它搬到了遥感多模态场景。

### 数学推导

设策略 $\pi_\theta$ 在输入 $q$（图像 + 指令）下采样一组输出 $\{o_1, o_2, ..., o_G\}$，用规则奖励函数 $r(o_i, y^*)$ 计算分数，GRPO 目标为：

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{o_i \sim \pi_\theta} \left[ A_i \log \pi_\theta(o_i \mid q) \right] + \beta \cdot \text{KL}[\pi_\theta \| \pi_{\text{ref}}]
$$

其中优势估计（归一化组内得分差）：

$$
A_i = \frac{r(o_i) - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}
$$

**关键点**：KL 散度项防止策略偏离 SIT 之后的参考模型太远，这对多模态模型尤其重要——RL 很容易把生成图像的质量搞坏。

### 奖励函数设计

对于**变化理解任务**（文本输出），奖励是组合式的：

$$
r_{\text{understand}} = \lambda_1 r_{\text{format}} + \lambda_2 r_{\text{accuracy}}
$$

- $r_{\text{format}}$：输出是否包含规定的 XML/JSON 字段，0/1 奖励
- $r_{\text{accuracy}}$：与标注答案的 token 匹配率或 F1

对于**未来预测任务**（图像输出），奖励基于生成图像质量：

$$
r_{\text{forecast}} = -\text{FID}(\hat{x}, x_{\text{ref}}) \cdot \mathbb{1}[\hat{x} \text{ is valid}]
$$

这是"可验证"奖励的精髓：奖励函数本身不需要神经网络，规则可验证，不会被 reward hacking。

---

## 实现

### 最小 VRO 训练循环

```python
import torch
import torch.nn.functional as F

def compute_grpo_loss(model, ref_model, batch, G=4, beta=0.01):
    """
    batch: {"input_ids", "images", "labels"}
    G: 每个输入采样的输出数量
    """
    # Step 1: 对每个输入采样 G 个输出
    with torch.no_grad():
        outputs = []
        for _ in range(G):
            out = model.generate(
                batch["input_ids"], 
                images=batch["images"],
                do_sample=True, temperature=0.8, max_new_tokens=256
            )
            outputs.append(out)  # [B, L]
    
    # Step 2: 计算可验证奖励
    rewards = torch.zeros(len(outputs), batch["input_ids"].shape[0])
    for i, out in enumerate(outputs):
        rewards[i] = compute_verifiable_reward(out, batch["labels"])
    # rewards: [G, B]
    
    # Step 3: 组内归一化得到优势
    mean_r = rewards.mean(dim=0, keepdim=True)  # [1, B]
    std_r  = rewards.std(dim=0, keepdim=True) + 1e-8
    advantages = (rewards - mean_r) / std_r  # [G, B]
    
    # Step 4: 计算策略梯度 + KL 惩罚
    total_loss = 0.0
    for i, out in enumerate(outputs):
        logp_new = model.log_prob(out, batch["input_ids"], batch["images"])
        with torch.no_grad():
            logp_ref = ref_model.log_prob(out, batch["input_ids"], batch["images"])
        
        kl = (logp_new - logp_ref).mean()
        pg_loss = -(advantages[i] * logp_new).mean()
        total_loss += pg_loss + beta * kl
    
    return total_loss / G
```

### 可验证奖励函数

```python
import re
from torchmetrics.image.fid import FrechetInceptionDistance

def compute_verifiable_reward(outputs, labels):
    """
    对 understanding 任务用规则打分，对 forecasting 用 FID
    outputs: list of decoded strings or image tensors
    """
    rewards = []
    for out, label in zip(outputs, labels):
        if label["task"] == "understanding":
            # 格式奖励：检查必要字段是否存在
            has_change_type  = bool(re.search(r"<change_type>.*</change_type>", out))
            has_description  = bool(re.search(r"<description>.*</description>", out))
            format_reward    = float(has_change_type and has_description) * 0.3
            
            # 准确率奖励：token F1（简化版）
            pred_tokens  = set(out.lower().split())
            label_tokens = set(label["text"].lower().split())
            f1 = 2 * len(pred_tokens & label_tokens) / (len(pred_tokens) + len(label_tokens) + 1e-8)
            rewards.append(format_reward + 0.7 * f1)
        
        elif label["task"] == "forecasting":
            # 图像质量奖励（用负 FID 的代理：SSIM 更轻量）
            from torchmetrics.functional import structural_similarity_index_measure as ssim
            score = ssim(out.unsqueeze(0), label["image"].unsqueeze(0)).item()
            rewards.append(score)
    
    return torch.tensor(rewards)
```

### 地理条件编码（GAGP 核心）

```python
import math
import torch.nn as nn

class GeoConditionEncoder(nn.Module):
    """将地理元数据编码为条件向量，注入到生成模型"""
    
    def __init__(self, d_model=512):
        super().__init__()
        # 经纬度用正弦位置编码，日期用循环编码
        self.date_proj  = nn.Linear(2, d_model // 4)   # sin/cos of day-of-year
        self.latlon_proj = nn.Linear(4, d_model // 4)  # sin/cos of lat, sin/cos of lon
        self.gsd_proj   = nn.Linear(1, d_model // 4)   # ground sampling distance
        self.out_proj   = nn.Linear(3 * d_model // 4, d_model)
    
    def forward(self, lat, lon, date_of_year, gsd):
        # 循环编码：把日期映射到圆上，避免年末/年初的跳变
        day_enc = torch.stack([
            torch.sin(2 * math.pi * date_of_year / 365),
            torch.cos(2 * math.pi * date_of_year / 365)
        ], dim=-1)
        
        latlon_enc = torch.stack([
            torch.sin(lat * math.pi / 180), torch.cos(lat * math.pi / 180),
            torch.sin(lon * math.pi / 180), torch.cos(lon * math.pi / 180)
        ], dim=-1)
        
        geo_feat = torch.cat([
            self.date_proj(day_enc),
            self.latlon_proj(latlon_enc),
            self.gsd_proj(gsd.unsqueeze(-1))
        ], dim=-1)
        return self.out_proj(geo_feat)  # [B, d_model]
```

---

## 关键 Trick（论文里不一定写清楚的）

### 1. KL 系数调度

VRO 中 $\beta$（KL 惩罚系数）不能是固定值。太大→模型不敢偏离 SIT 结果，RL 无效；太小→生成图像快速退化。

```python
# 线性 warmup + 余弦衰减
def get_kl_beta(step, warmup_steps=500, total_steps=5000, beta_max=0.05, beta_min=0.001):
    if step < warmup_steps:
        return beta_max * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return beta_min + 0.5 * (beta_max - beta_min) * (1 + math.cos(math.pi * progress))
```

### 2. 组大小 G 的选择

G 太小（2-3）：优势估计方差太大，训练不稳定。G 太大（>8）：显存爆炸，推理成本翻倍。论文用 G=4，这个值在大多数场景都是合理的起点。

### 3. 理解任务和预测任务的 loss 权重

两个任务的奖励量纲不同（F1 ∈ [0,1] vs SSIM ∈ [-1,1]），必须归一化到同一尺度，否则一个任务会主导梯度。简单做法：各自用 z-score 归一化后再加权。

### 4. 参考模型的更新策略

VRO 的参考模型（KL 锚点）应该是 SIT 结束后的快照，**不要随策略更新**。如果用动态参考模型（如 EMA），KL 约束会逐渐失效，导致生成质量崩溃。

---

## 实验与对比

### 环境选择思路

论文用的评测维度：
- **变化问答**：AUC、Recall、精确率
- **场景生成**：FID（越低越好）

FID=43.13 这个数字需要上下文：在遥感领域，FID 参考分布是真实卫星图像，而不是 ImageNet，数值不能直接跨领域比较。

### 与 Baseline 对比（论文数据）

| 方法 | 参数量 | 变化理解 (AUC) | 未来预测 (FID ↓) |
|------|--------|--------------|----------------|
| 专用变化检测模型 | 0.3B | 高 | - |
| 通用 VLM (7B+) | 7B+ | 中 | - |
| Gemini-2.5-Flash | 闭源 | - | > 43.13 |
| **RS-WorldModel** | **2B** | **SOTA** | **43.13** |

2B 参数超越 120 倍大的开源模型，核心原因不是架构创新，而是**任务协同** + **VRO 的精炼效果**。

---

## 调试指南

### VRO 常见问题

**1. 奖励方差极大，loss 震荡**
- 原因：可验证奖励函数设计有问题，大多数样本得分为 0 或 1（两极分化）
- 修复：检查格式奖励是否过于严苛，适当引入软奖励（partial credit）

**2. KL 爆炸（KL > 10）**
- 原因：学习率太高，或 beta 太小
- 修复：先检查 beta 调度，再降低 RL 阶段的 lr（通常应比 SIT 小 5-10 倍）

**3. 生成图像质量退化（FID 越来越差）**
- 原因：RL 优化文本描述质量时，破坏了生成图像的分布
- 修复：确保两个任务的 batch 按比例混合（不要全 RL 步骤只用理解任务）

### 如何判断 VRO 在"工作"

| 指标 | 健康信号 | 危险信号 |
|------|---------|---------|
| 组内奖励方差 | 稳定在 0.1-0.5 | 接近 0（模型输出退化为单一解） |
| KL 散度 | < 2.0 | > 5（策略偏离太远） |
| 格式奖励 | 稳步上升 | 振荡不收敛 |
| FID（预测任务） | 缓慢下降 | 先降后升（过拟合奖励代理） |

---

## 超参数调优

| 参数 | 推荐范围 | 敏感度 | 建议 |
|------|---------|-------|-----|
| 组大小 G | 4-8 | 高 | 先用 G=4，显存够再加 |
| KL 系数 β | 0.001-0.05 | 高 | 从 0.01 开始，看 KL 曲线调整 |
| RL 学习率 | 5e-7 ~ 5e-6 | 很高 | 比 SFT 低一个数量级 |
| RL 步数 | 500-2000 | 中 | 看验证集，不要 RL 太久 |
| 格式奖励权重 λ₁ | 0.2-0.4 | 中 | 确保格式不对准时有明显惩罚 |

---

## 什么时候用 / 不用

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要同时支持变化检测和场景生成 | 只需要单任务，用专用模型更简单 |
| 有丰富的地理元数据（经纬度、日期） | 元数据缺失，GAGP 优势消失 |
| 数据量 > 10 万样本，VRO 有足够信号 | 数据稀少场景，RL 信号太噪 |
| 对格式化输出有严格要求 | 输出格式自由，不需要可验证奖励 |

---

## 我的观点

**VRO 的设计是亮点，也是局限所在。**

可验证奖励在文本任务上效果很好——格式对不对、关键词有没有，规则可以写得很清晰。但在图像生成任务上，FID 和 SSIM 作为奖励信号都有明显缺陷：FID 需要大批量样本才稳定，单样本估计噪声极大；SSIM 对纹理不敏感，容易被模型糊弄（生成一张模糊但结构正确的图就能得高分）。

论文的 2B 参数效率论证是真实的，但主要来自**任务协同**的数据效率，而不是 VRO。如果你只关心变化检测，直接用 SIT 阶段的模型就够了，VRO 的增益更多体现在格式稳定性上。

**值得复现的核心**：GAGP 中地理条件编码的设计——这个思路简单、有效，而且被严重低估。在任何需要处理多地区、多季节卫星数据的项目里，把经纬度和拍摄日期作为显式条件注入都是值得一试的工程改进。