---
layout: post-wide
title: 'Hide and Seek：不碰模型权重也能做"概念级"黑盒归因'
date: 2026-10-12 12:03:38 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2610.13935v1
generated_by: Claude Code CLI
---

## 一句话总结

HnS（Hide and Seek）是一种**纯黑盒**的概念归因方法：不需要访问模型权重、不需要人工标注的概念数据集，用视觉语言模型（VLM）检索概念相关图像、用分割模型定位概念区域、遮挡该区域后测量黑盒模型输出的变化——这个变化量就是"这个概念对模型预测有多重要"的归因分数。它的关键卖点是**任务无关**：分类、回归、图像描述（captioning）、语义分割都能用同一套流程解释，这是之前的概念归因方法做不到的。

## 背景：为什么需要这种方法？

先说清楚现有概念归因方法的两个硬约束：

1. **TCAV 一类的方法**：需要在模型的某一中间层提取激活向量，训练一个线性探针来区分"有这个概念"和"没有这个概念"。这意味着你必须能拿到模型权重和中间层输出——对闭源 API 模型、商用黑盒模型根本用不了。而且训练探针需要一批**带概念标注**的图像（比如"戴眼镜"的正负样本），标注成本不低。
2. **Grad-CAM 系列**：不需要概念标注集，但需要梯度，本质上还是白盒方法；而且输出的是像素级显著性图，不是"概念"这个语义单位。
3. **两类方法共同的限制**：它们都假设模型输出是一个固定的类别 logit 或回归值。遇到 captioning（输出是一串词）、segmentation（输出是像素级类别图）这种结构化输出，这些方法直接失效——"对某个类别的梯度"这个概念本身就不存在。

HnS 的思路是绕开这两个约束：**不需要看模型内部，只需要能调用模型做前向推理（query access）；不需要人工标注概念数据集，用 VLM 的 zero-shot 能力去检索"长得像这个概念"的图片。** 遮挡-测量输出变化这个机制是通用的，不管模型输出是 logit、caption 还是分割图，只要能定义一个"输出变化量"的度量，流程就能跑。

这是一个典型的"用工程手段换掉强假设"的设计：原来的强假设是"可以拿到模型内部/有标注数据"，换成了"VLM 和分割模型的 zero-shot 能力足够好"。这个 trade-off 好不好，取决于你有没有可靠的 VLM 和分割模型可用——后面调试部分会讲这个假设什么时候会崩。

## 算法原理

### 直觉解释

想象你要验证"模型的猫分类器是不是真的在看猫耳朵，而不是在看背景的沙发"。传统做法要么打开模型内部看激活，要么人工标注一堆"有猫耳朵/没猫耳朵"的图。HnS 的做法更粗暴：

1. 找一堆带猫的图片；
2. 用 VLM 在这些图片里找到"像猫耳朵"的图片子集；
3. 用分割模型在每张图里圈出猫耳朵的位置；
4. 把猫耳朵区域抠掉（用某种方式填充），喂给模型看分类结果变化多少；
5. 变化越大，说明模型越依赖猫耳朵这个概念。

本质上这是**遮挡敏感性分析（occlusion sensitivity）** 的概念级升级版：把"手动划定遮挡区域"换成了"VLM + 分割模型自动划定概念区域"。

### 数学形式化

给定概念 $c$（比如一段自然语言描述）、图像池 $\mathcal{X}$、黑盒模型 $f$，HnS 定义归因分数：

$$
A(c) = \mathbb{E}_{x \sim \mathcal{R}(c, \mathcal{X})}\Big[\, f(x) - f\big(\text{mask}(x, \ell(x, c))\big) \,\Big]
$$

其中：
- $\mathcal{R}(c, \mathcal{X})$ 是用 VLM 检索出的"概念相关"图像分布（实际上就是取 top-k 相似度的图片集合）；
- $\ell(x, c)$ 是分割/定位模型给出的概念区域 mask；
- $\text{mask}(\cdot)$ 是某种遮挡函数（填均值、填噪声、inpainting 都可以）；
- 如果 $f$ 输出是标量 logit，$f(x) - f(x_{\text{masked}})$ 就是直接的数值差；如果是 caption 或分割图这种结构化输出，需要先定义一个标量化的距离度量（比如 caption 的 CLIP 相似度下降量、分割图的 IoU 下降量）。

这个公式的好处是：**只要能定义"输出的变化"，$f$ 是什么黑盒都无所谓**。这正是它能用在 captioning/segmentation 上的原因——以前的方法做不到这一步，是因为它们绑死在"对某个类别求梯度"这个操作上。

### 与其他方法的关系

HnS 继承的是 LIME / RISE / occlusion sensitivity 这一条"扰动输入、观察输出变化"的谱系，而不是 Grad-CAM 这条"反向传播梯度"的谱系。它的创新点不在"遮挡-测量"这个机制本身（这个机制很老），而在于**用 VLM 把"概念"这个语义单位和"图像区域"对齐起来**，让遮挡法第一次能在概念粒度上工作，而不是在像素/超像素粒度上工作。

## 实现

下面给出一个精简实现，用 HuggingFace 的 CLIP 做概念检索，CLIPSeg 做开放词表定位。这是我自己写的示意代码，**不是论文官方实现**——论文代码是匿名提交在补充材料里的，目前没有公开仓库链接，想跑官方版本请直接去 arxiv 页面找补充材料。

### 最小可运行版本

```python
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def retrieve_concept_images(concept: str, image_pool: list, top_k: int = 20):
    """用 CLIP 的 zero-shot 相似度，从图像池里挑出最像这个概念的图片"""
    inputs = clip_proc(text=[concept], images=image_pool,
                        return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip(**inputs)
    sims = (out.image_embeds @ out.text_embeds.T).squeeze(-1)
    idx = sims.topk(min(top_k, len(image_pool))).indices.tolist()
    return [image_pool[i] for i in idx]

def mean_mask(image: Image.Image, mask: torch.Tensor):
    """最简单的遮挡策略：用整张图的均值颜色填充 mask 区域"""
    arr = torch.tensor(np.array(image)).float()
    arr[mask.bool()] = arr.mean(dim=(0, 1))
    return Image.fromarray(arr.byte().numpy())

def attribution_score(black_box_fn, image: Image.Image, mask: torch.Tensor):
    """核心度量：遮挡概念区域前后，黑盒输出的变化量"""
    masked = mean_mask(image, mask)
    return black_box_fn(image) - black_box_fn(masked)
```

这个版本只有检索和一种最朴素的遮挡方式，缺了"定位"这一步（直接假设 mask 已知）。下面补上定位和完整流程。

### 完整实现

```python
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch.nn.functional as F

seg_proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
seg_model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined").to(device).eval()

def localize_concept(image: Image.Image, concept: str, threshold: float = 0.5):
    """用 CLIPSeg 做开放词表分割，返回概念区域的二值 mask"""
    inputs = seg_proc(text=[concept], images=[image],
                       return_tensors="pt").to(device)
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    prob = logits.sigmoid()[0]
    prob = F.interpolate(prob[None, None], size=image.size[::-1],
                          mode="bilinear")[0, 0]
    return (prob > threshold).float()

def hide_and_seek(concept, image_pool, black_box_fn,
                   top_k: int = 20, mask_threshold: float = 0.5,
                   min_mask_ratio: float = 0.01):
    """HnS 主流程：检索 -> 定位 -> 遮挡 -> 测量，返回该概念的平均归因分数"""
    images = retrieve_concept_images(concept, image_pool, top_k)
    scores = []
    for img in images:
        mask = localize_concept(img, concept, mask_threshold)
        # 定位失败（mask 面积过小）直接跳过，否则会引入噪声，见下文 trick
        if mask.mean() < min_mask_ratio:
            continue
        scores.append(attribution_score(black_box_fn, img, mask))
    return torch.stack(scores).mean().item() if scores else float("nan")
```

`black_box_fn` 可以是任何接受 PIL 图片、输出标量的函数：对分类模型就是某个类别的 logit，对分割模型可以是目标类别的平均置信度，对 captioning 模型可以是生成 caption 与某个参考描述的 CLIP 相似度。

### 关键 Trick

这几个点论文摘要里不会细讲，但你跑起来发现结果不对劲，大概率是踩了这些坑：

- **遮挡方式会引入分布外（OOD）伪影**。用纯黑色或均值颜色填充会在 mask 边缘产生不自然的硬边缘，有些模型对这种边缘本身敏感（而不是对"内容消失"敏感），导致归因分数虚高。更稳的做法是用 inpainting 模型（比如 LaMa）生成语义合理的填充，代价是速度慢很多。
- **要有"无关区域"遮挡作为对照 baseline**。单纯遮挡概念区域测出的分数里，混杂了"遮挡动作本身"带来的偏差。正确做法是额外遮挡一块随机无关区域，把这部分的输出变化当作噪声基线减掉。
- **mask 面积过滤**。CLIPSeg 对抽象概念的定位经常是"全图均匀低置信度"而不是"精确的一小块"，这种情况下 mask 面积会接近 0 或接近全图，两种极端都要过滤掉，不然分数无意义。
- **检索质量决定一切**。如果 VLM 对这个概念理解弱，`retrieve_concept_images` 返回的根本不是你想要的图片，后面定位、遮挡做得再好也没用。实践中最好对检索结果做人工抽查，而不是直接信任 top-k。

## 实验

论文在 CelebA（人脸属性，有标注概念如"戴眼镜""微笑"）和 COCO-Stuff（语义分割标注可以当概念定位的 ground truth）上做了验证，并报告 HnS 与真实归因（ground-truth attribution）的一致性优于现有方法。**具体数值我没有独立复现，不在这里编造表格**——如果你要验证这个结论，建议的流程是：

1. 在 CelebA 上选几个有明确标注的属性（比如"戴帽子"），用标注本身作为"ground truth concept region"；
2. 分别用 HnS 的全自动检索+定位流程、以及用标注直接给出的 mask，跑一遍归因分数；
3. 对比两者的排序一致性（比如 Spearman 相关系数），这样能单独衡量"自动检索+定位"这一步引入的误差，而不是笼统地说"HnS 好不好"。

这个拆分很重要：HnS 的误差来源是两段式的（VLM 检索误差 + 分割定位误差），把它们分开测，才能知道瓶颈在哪一段。

## 调试指南

### 常见问题

1. **归因分数在所有概念上都差不多，区分度很低**：大概率是遮挡策略引入的分布外伪影主导了分数，建议先排查 baseline（无关区域遮挡）的分数是不是同样很高——如果是，说明问题不在概念本身，在遮挡方式。
2. **某个概念的分数突然异常高或异常低**：先去肉眼检查检索回来的图片是不是真的和概念相关，CLIPSeg 对非常细碎或非常抽象的概念（比如"忧郁的表情"）定位经常是乱的。
3. **分数方差极大，换一批检索图片结果完全不一样**：提高 top_k，增加聚合样本量；同时考虑给检索结果按相似度加权平均，而不是简单平均。

### 超参数敏感度（基于个人实践推测，论文未必详细讨论）

| 超参数 | 推荐范围 | 敏感度 | 备注 |
|---|---|---|---|
| top_k（检索图片数） | 15~30 | 中 | 太少方差大，太多会混入弱相关图片 |
| mask_threshold（CLIPSeg 阈值） | 0.4~0.6 | 高 | 阈值附近 mask 面积变化剧烈，建议对每个概念单独调 |
| min_mask_ratio | 0.01~0.05 | 中 | 太低会放过无效 mask，太高会误杀小物体概念 |
| 遮挡策略 | 均值/inpainting | 高 | 均值简单但伪影重，inpainting 更真实但慢且依赖额外模型质量 |

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要解释非分类任务（captioning、segmentation、检测） | 概念抽象到 VLM 难以检索/分割（如"构图的紧张感"） |
| 只有 query 权限的黑盒/商用 API 模型 | 需要实时/交互式解释（pipeline 包含检索+分割+多次前向，延时高） |
| 有大量未标注图像池但没有概念标注数据集 | 需要严格因果证明的高风险场景（遮挡法衡量的是相关性，不是因果干预） |

## 我的观点

HnS 解决的核心痛点是真实的：之前的概念归因方法确实被"白盒访问"和"标注数据"这两个假设卡得很死，遇到 captioning/segmentation 这类结构化输出任务直接没法用。HnS 用 VLM + 分割模型这两个"外部零件"替换掉这两个假设，这个思路本身是合理的工程权衡。

但要诚实地说，这个权衡不是免费的：你把"需要标注数据"的问题，换成了"依赖 VLM 和分割模型的 zero-shot 质量"的问题。误差会从检索和定位两个环节传播到最终归因分数里，而这两个环节的失败模式都不透明——你很难判断一个异常的归因分数是模型真的这么想，还是 CLIPSeg 定位错了。这跟 RL 里的 reward hacking 有点像：你测量的是"遮挡前后输出的变化"，但模型可能对遮挡这个操作本身的统计特征（边缘、色块）有反应，而不是对"概念缺失"有反应——如果不做无关区域遮挡的对照实验，很容易把伪影当成洞见。

值得一试的场景是：你手上有个黑盒模型，输出不是简单分类（比如一个 captioning 或分割模型），又没有现成的概念标注数据，且能接受较高的推理延时——这正是以前的方法完全覆盖不到的空白。如果你的场景本身就是标准分类任务、又有白盒访问权限，TCAV 或 Grad-CAM 系列仍然更便宜、更可控，没必要上这套更重的 pipeline。