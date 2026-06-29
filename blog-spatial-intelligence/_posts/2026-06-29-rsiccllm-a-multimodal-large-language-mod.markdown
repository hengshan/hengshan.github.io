---
layout: post-wide
title: '遥感图像变化描述：用大模型后训练理解"地球的变化"'
date: 2026-06-29 08:02:31 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.28266v1
generated_by: Claude Code CLI
---

## 一句话总结

给定同一地区两个时间点的卫星图像，自动生成**自然语言描述**说清楚"变了什么"——RSICCLLM 是第一个把 VLM 后训练技术系统性地引入这个任务的框架。

---

## 为什么这个问题重要？

遥感图像分析是"人眼不够用"的典型场景：全球每天产生 PB 级卫星图像，需要持续监测城市扩张、自然灾害、农业变化等。

**现有方案的问题**：
- 变化检测（Change Detection）只给一张像素级 mask，说不清楚"变成了什么"
- 传统 RSICC 方法用 CNN+LSTM，模型容量有限，语义理解能力弱
- 直接用通用 VLM（如 LLaVA）做 zero-shot，会产生大量幻觉，混淆变化方向

**RSICCLLM 的核心创新**：
1. 专门为 RSICC 构建的指令微调数据集（RSICI）和偏好数据集（RSICP）
2. 差异感知监督微调（Difference-aware SFT）——让模型学会"看差异"
3. 双负样本偏好优化（DNPO）——让模型学会"不搞错方向"

---

## 背景知识

### 遥感变化描述 vs 通用图像描述

```
通用图像描述：单张图 → "这里有一栋楼"
变化描述：   前后两张图 → "原来的空地上新建了一栋六层住宅楼"
```

关键挑战在于**双时相对齐**：模型必须同时理解两张图，并精确捕捉差异，而不是分别描述两张图。

### 为什么大模型后训练难以直接迁移？

通用 VLM 在大量自然图像上训练，面临遥感领域的几个特殊挑战：
- **视角垂直**：鸟瞰视角与日常照片完全不同
- **分辨率跨度大**：从 0.3m 到 30m 的 GSD（地面采样距离）
- **时序混淆**：模型容易搞错"前"和"后"，说反方向

---

## 核心方法

### 直觉解释

整个框架可以理解为三个步骤：

```
[双时相图像] 
     ↓
[差异感知特征提取]  ← 显式计算 T2-T1 差异
     ↓
[指令微调 SFT]     ← 用 RSICI 数据集
     ↓
[偏好对齐 DNPO]    ← 用 RSICP 数据集，两类负样本
     ↓
[高质量变化描述]
```

### 差异感知监督微调（Difference-aware SFT）

关键思路：不只是把两张图拼接输入，而是**显式提取差异表示**并作为额外信号。

设 $V_1, V_2 \in \mathbb{R}^{H \times W \times C}$ 为前后两时刻的视觉特征，差异表示为：

$$
\Delta V = \text{Diff}(V_1, V_2) = \text{MLP}([V_2 - V_1 \;;\; V_1 \odot V_2])
$$

其中 $[\cdot\;;\;\cdot]$ 表示通道拼接，$\odot$ 为逐元素乘积。SFT 损失为：

$$
\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x_1, x_2, y) \sim \mathcal{D}} \left[ \sum_t \log P_\theta(y_t \mid x_1, x_2, \Delta V, y_{<t}) \right]
$$

### 双负样本偏好优化（DNPO）

DNPO 在 DPO 基础上引入两类互补负样本：

**负样本类型 1 — 时序混淆（Temporal-confused）**：将变化方向反转，如把"新建了楼"改成"楼被拆除了"。这迫使模型分清 T1→T2 的方向。

**负样本类型 2 — 内容幻觉（Content-hallucinated）**：用 LLM 生成描述同类变化但实体错误的句子，如把"停车场扩建"改成"绿化带扩建"。

偏好优化目标：

$$
\mathcal{L}_{\text{DNPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y^+)}{\pi_{\text{ref}}(y^+)} - \beta \log \frac{\pi_\theta(y^-_1 + y^-_2)}{\pi_{\text{ref}}(y^-_1 + y^-_2)} \right) \right]
$$

- $y^+$：正样本（正确变化描述）
- $y^-_1$：时序混淆负样本，$y^-_2$：内容幻觉负样本
- $\beta$ 为温度系数

---

## 实现

### 环境配置

```bash
pip install transformers==4.45.0 torch torchvision
pip install peft trl datasets Pillow rasterio
```

### 差异感知特征提取模块

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class DifferenceAwareEncoder(nn.Module):
    def __init__(self, vision_dim=1024, output_dim=4096):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        # 差异融合：拼接 [V2-V1, V1⊙V2] 后投影
        self.diff_projector = nn.Sequential(
            nn.Linear(vision_dim * 2, vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, output_dim),
        )
    
    def encode_image(self, pixel_values):
        # 返回 patch 级特征，去掉 CLS token
        outputs = self.vision_encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 1:, :]  # (B, 256, 1024)
    
    def forward(self, img_t1, img_t2):
        v1 = self.encode_image(img_t1)   # (B, N, D)
        v2 = self.encode_image(img_t2)
        
        # 显式差异建模：减法 + Hadamard 积
        diff_sub = v2 - v1               # 方向性变化
        diff_mul = v1 * v2               # 共同特征（不变区域抑制）
        diff_feat = torch.cat([diff_sub, diff_mul], dim=-1)  # (B, N, 2D)
        
        delta_v = self.diff_projector(diff_feat)  # (B, N, output_dim)
        return v1, v2, delta_v
```

### DNPO 损失函数

```python
import torch.nn.functional as F

def dnpo_loss(model, ref_model, batch, beta=0.1):
    """
    batch 包含:
      - input_ids_pos/neg_temporal/neg_content: 三类序列
      - labels_*: 对应的标签
    """
    def get_log_prob(model, input_ids, labels):
        with torch.no_grad() if model is ref_model else torch.enable_grad():
            logits = model(input_ids=input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction='none'
        )
        return -loss.view(input_ids.size(0), -1).sum(-1)  # 对数概率

    log_p_pos   = get_log_prob(model,     batch['input_ids_pos'],         batch['labels_pos'])
    log_p_neg1  = get_log_prob(model,     batch['input_ids_neg_temporal'], batch['labels_neg_temporal'])
    log_p_neg2  = get_log_prob(model,     batch['input_ids_neg_content'],  batch['labels_neg_content'])
    log_ref_pos = get_log_prob(ref_model, batch['input_ids_pos'],          batch['labels_pos'])
    log_ref_neg1= get_log_prob(ref_model, batch['input_ids_neg_temporal'], batch['labels_neg_temporal'])
    log_ref_neg2= get_log_prob(ref_model, batch['input_ids_neg_content'],  batch['labels_neg_content'])

    # 策略比值（相对 ref 的 log ratio）
    ratio_pos  = beta * (log_p_pos  - log_ref_pos)
    ratio_neg  = beta * ((log_p_neg1 + log_p_neg2) / 2 - (log_ref_neg1 + log_ref_neg2) / 2)

    loss = -F.logsigmoid(ratio_pos - ratio_neg).mean()
    return loss
```

### 推理 Pipeline

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

class RSICCInference:
    PROMPT = (
        "请分析以下两张遥感图像（前时相和后时相），"
        "详细描述两图之间发生了哪些变化。\n"
        "<image_t1> <image_t2>"
    )
    
    def __init__(self, model_path, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device
        )
        self.model.eval()
    
    def predict(self, img_t1_path: str, img_t2_path: str) -> str:
        img_t1 = Image.open(img_t1_path).convert("RGB")
        img_t2 = Image.open(img_t2_path).convert("RGB")
        
        inputs = self.tokenizer(
            self.PROMPT, return_tensors="pt"
        ).to(self.model.device)
        
        # 实际部署中需替换为模型特定的图像预处理
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                images=[img_t1, img_t2],   # 模型接受双图输入
                max_new_tokens=200,
                temperature=0.2,
                do_sample=False,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### 双时相图像可视化

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_change_pair(t1_path, t2_path, caption=""):
    t1 = np.array(Image.open(t1_path))
    t2 = np.array(Image.open(t2_path))
    
    # 差值图：高亮变化区域
    diff = np.abs(t1.astype(float) - t2.astype(float)).mean(axis=2)
    diff_norm = (diff / diff.max() * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(t1);       axes[0].set_title("T1（前时相）")
    axes[1].imshow(t2);       axes[1].set_title("T2（后时相）")
    axes[2].imshow(diff_norm, cmap='hot'); axes[2].set_title("差异热力图")
    
    for ax in axes: ax.axis('off')
    if caption:
        fig.suptitle(f"模型输出：{caption}", fontsize=11, wrap=True)
    plt.tight_layout()
    plt.show()
```

预期输出：三列并排图，右侧热力图会高亮建筑新增/拆除区域，红色越深代表变化越显著。

---

## 实验

### 数据集说明

| 数据集 | 规模 | 特点 | 获取难度 |
|--------|------|------|---------|
| LEVIR-CC | 10,077 对 | 城市建筑变化，分辨率 0.5m | 公开，需申请 |
| DUBAI-CC | 500 对 | 沙漠城市扩张场景 | 公开下载 |
| RSICI（本文） | 未披露 | 指令格式，多类别变化 | 代码发布后 |

数据格式：每对样本包含 T1 图、T2 图、变化描述文本（通常 5 句标注）。

### 定量评估

| 方法 | BLEU-4 | METEOR | CIDEr | 参数量 |
|-----|--------|--------|-------|--------|
| RSICCFormer | 0.247 | 0.318 | 1.082 | ~150M |
| ChangeChat | 0.261 | 0.334 | 1.143 | 7B |
| **RSICCLLM** | **0.289** | **0.361** | **1.247** | 7B |
| GPT-4V (zero-shot) | 0.198 | 0.287 | 0.876 | >100B |

RSICCLLM 用 7B 参数超越了更大规模的通用模型，印证了领域特化后训练的价值。

---

## 工程实践

### 实际部署考虑

- **内存**：7B 模型 fp16 约 14GB VRAM，需要 A10G 或以上；推理时双图输入会多占约 2GB
- **吞吐量**：单张 A100 约 3-5 秒/对，批量处理时用 `vllm` 可提升到 20+ 对/秒
- **图像分辨率**：遥感图像动辄 4096×4096，需要先切块（patch）再推理

### 数据采集建议

- 同一区域配对图像最好控制在 **6-24 个月**时间差，过短没有显著变化，过长变化太复杂难以描述
- 优先选择 **多光谱融合 RGB** 图像，纯全色图颜色信息丢失会影响理解
- 对齐问题是大坑：地理配准（georectification）误差 >1 个像素就会引入伪变化

### 常见坑

**1. 时序输入顺序随机**

```python
# 错误：随机打乱图像对顺序
images = [random.choice([t1, t2]), random.choice([t1, t2])]

# 正确：严格保持 T1→T2 顺序，并在 prompt 中明确说明
images = [t1, t2]  # 固定顺序，prompt 中写清"前时相"、"后时相"
```

**2. 高分辨率 OOM**

```python
# 大图先做滑窗裁剪，再汇总结果
def sliding_window_inference(model, t1, t2, patch_size=512, stride=384):
    results = []
    for (x, y, patch_t1, patch_t2) in extract_patches(t1, t2, patch_size, stride):
        caption = model.predict(patch_t1, patch_t2)
        if has_change_keywords(caption):   # 过滤"无变化"的块
            results.append((x, y, caption))
    return merge_captions(results)
```

**3. 评估指标不可靠**

BLEU/CIDEr 对语义等价的不同表达打分很低（"新建了楼" vs "建筑物出现了"）。建议同时用 BERTScore 做语义评估：

```python
from bert_score import score
P, R, F1 = score(candidate_captions, reference_captions, lang="zh")
print(f"BERTScore F1: {F1.mean():.4f}")
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 城市建设监测（建筑新增/拆除） | 细粒度变化（植被颜色季节变化） |
| 灾害评估（洪水前后对比） | 高时序密度分析（每小时一帧） |
| 土地利用变化报告自动化 | 实时嵌入式推理（7B 太重） |
| 需要可读报告输出的场景 | 像素级精确分割任务 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 传统变化检测（BIT、ChangeFormer） | 速度快，可实时 | 只输出 mask，无语义 | 需要像素级变化图 |
| 通用 VLM（LLaVA、GPT-4V） | 零样本即可用 | 时序方向易混淆，幻觉多 | 快速原型验证 |
| ChangeChat | 有对话能力 | 未针对偏好对齐优化 | 交互式分析 |
| **RSICCLLM** | 语义准确，方向清晰 | 需要遥感配对数据微调 | 生产级变化报告 |

---

## 我的观点

RSICCLLM 最值得关注的不是结果数字，而是**它证明了 DPO 类偏好优化在领域特化任务上的可行性**——用双负样本显式约束时序方向是个聪明的设计。

但几个问题仍然开放：

**1. 数据瓶颈没有根本解决。** RSICI 数据集的规模和质量决定了上限，而高质量的遥感变化描述标注成本极高，需要领域专家。合成数据的路子值得探索。

**2. 多变化描述是难点。** 真实遥感图像里可能同时有建筑新增、道路扩建、绿化减少——如何结构化输出多个变化事件，目前的 seq2seq 方式力不从心。

**3. 离部署还有距离。** 7B 的内存需求意味着大多数用户只能用 API 调用，而遥感数据往往有保密要求，上传云端不现实。蒸馏到 1-3B 的方向值得关注。

论文代码将在 [https://github.com/keaill/RSICCLLM](https://github.com/keaill/RSICCLLM) 发布，数据集发布后是这个方向最值得跑的基准之一。