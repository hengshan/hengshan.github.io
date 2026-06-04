---
layout: post-wide
title: '两阶段车辆精细分类：当 Vision Transformer 学会"说不知道"'
date: 2026-06-04 08:04:34 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.05149v1
generated_by: Claude Code CLI
---

## 一句话总结

用 RT-DETR 做粗定位、ViT 做细粒度分类，加上一个会主动放弃的置信度阈值——这套开源流水线在真实路侧视频上达到 0.94 准确率，更重要的是它知道自己什么时候不靠谱。

## 为什么这篇论文重要？

标准目标检测器（YOLO 系列、Faster R-CNN）给你的标签只有 `car`、`truck`、`bus`——这对大多数任务够用，但在**自行车道安全研究**中远远不够。当你需要知道超车的是皮卡还是厢式货车时，粗标签毫无意义：两者对骑行者的伤害风险截然不同。

现有的细粒度车辆识别数据集（CompCars、Stanford Cars）大多是干净的正面或侧面照片，在真实路侧视频中——背光、遮挡、运动模糊——这些模型往往**悄无声息地出错**，没有任何警示。

这篇论文的核心洞见是：**一个承认自己不确定的系统，比一个自信犯错的系统更有价值**。这个原则远比车辆分类本身更通用。

## 核心方法解析

### 直觉理解

把整个流水线想象成两个专家合作：

1. **第一个专家（RT-DETR）**：负责喊"那里有辆车！"并圈出位置，但不管具体车型
2. **第二个专家（ViT）**：只看被裁剪出来的车辆区域，做精细分类；如果他不确定，他会说"不知道"而不是瞎猜

关键设计决策是两阶段而非端到端。这样做的原因在于**任务分离**：检测需要感知全局上下文，细粒度分类需要聚焦局部纹理差异，强行合并往往导致两头都不讨好。

### 架构流程

```
路侧视频帧
    ↓
[Stage 1] RT-DETR → 粗检测框 + 类别（car/truck/bus...）
    ↓
裁剪车辆区域（+ 适当 padding）
    ↓
[Stage 2] ViT-Base/16 → softmax 概率分布
    ↓
置信度 ≥ 0.60？ → 是：输出细粒度标签（SUV/皮卡/面包车...）
               → 否：输出 "unknown"
```

### 置信度弃权机制

给定 ViT 输出的 softmax 概率向量 $\mathbf{p} \in \mathbb{R}^6$，预测规则为：

$$\hat{y} = \begin{cases} \arg\max_k p_k & \text{if } \max_k p_k \geq \tau \\ \text{unknown} & \text{otherwise} \end{cases}$$

其中 $\tau = 0.60$ 是弃权阈值。论文中 in-distribution 弃权率为 2.4%，域外（OOD）弃权率上升到 25.0%——这个行为完全符合预期：**遇到不熟悉的数据，模型变得更谨慎，而不是更自信**。

## 动手实现

### Stage 1：RT-DETR 车辆检测

```python
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import torch
from PIL import Image

# COCO 数据集中的车辆类别 ID
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
detector = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
detector.eval()

def detect_vehicles(image: Image.Image, threshold: float = 0.5) -> list[dict]:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = detector(**inputs)

    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=[image.size[::-1]]
    )[0]

    vehicles = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        if label.item() in VEHICLE_CLASS_IDS:
            vehicles.append({
                "box": box.tolist(),   # [x1, y1, x2, y2]
                "coarse_label": VEHICLE_CLASS_IDS[label.item()],
                "score": score.item(),
            })
    return vehicles
```

### Stage 2：ViT 细粒度分类 + 弃权机制

```python
from transformers import ViTForImageClassification, ViTImageProcessor
import torch.nn.functional as F

FINE_CLASSES = ["passenger_car", "suv", "pickup_truck", "minivan", "large_van", "commercial_truck"]
ABSTENTION_THRESHOLD = 0.60

vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
classifier = ViTForImageClassification.from_pretrained(
    "your-finetuned-checkpoint",  # 替换为实际路径
    num_labels=len(FINE_CLASSES),
)
classifier.eval()

def classify_with_abstention(crop: Image.Image) -> dict:
    inputs = vit_processor(images=crop, return_tensors="pt")
    with torch.no_grad():
        logits = classifier(**inputs).logits

    probs = F.softmax(logits, dim=-1)[0]  # shape: (6,)
    confidence, pred_idx = probs.max(dim=-1)
    confidence = confidence.item()

    if confidence < ABSTENTION_THRESHOLD:
        return {"label": "unknown", "confidence": confidence, "abstained": True}

    return {
        "label": FINE_CLASSES[pred_idx.item()],
        "confidence": confidence,
        "abstained": False,
        "all_probs": {cls: probs[i].item() for i, cls in enumerate(FINE_CLASSES)},
    }
```

### 完整推理流水线

```python
def crop_with_padding(image: Image.Image, box: list, pad_ratio: float = 0.1) -> Image.Image:
    """裁剪车辆区域，加少量 padding 防止边缘被截断"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * pad_ratio)
    y1 = max(0, y1 - h * pad_ratio)
    x2 = min(image.width, x2 + w * pad_ratio)
    y2 = min(image.height, y2 + h * pad_ratio)
    return image.crop((x1, y1, x2, y2))

def run_pipeline(frame: Image.Image) -> list[dict]:
    """对单帧运行完整两阶段流水线"""
    vehicles = detect_vehicles(frame)
    results = []
    for vehicle in vehicles:
        crop = crop_with_padding(frame, vehicle["box"])
        fine_result = classify_with_abstention(crop)
        results.append({**vehicle, **fine_result})
    return results
```

### 如何微调 ViT 分类头

```python
from transformers import ViTForImageClassification, TrainingArguments
import torchvision.transforms as T

# 模拟路侧视频的常见变形做数据增强
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3),   # 模拟不同光照
    T.RandomAffine(degrees=5, translate=(0.05, 0.05)),  # 轻微视角抖动
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=6,
    ignore_mismatched_sizes=True,   # 重置原始 ImageNet 1000 类分类头
    id2label={i: c for i, c in enumerate(FINE_CLASSES)},
    label2id={c: i for i, c in enumerate(FINE_CLASSES)},
)

training_args = TrainingArguments(
    output_dir="./vit-vehicle-finetune",
    num_train_epochs=15,
    per_device_train_batch_size=32,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    evaluation_strategy="epoch",
    metric_for_best_model="f1",
    load_best_model_at_end=True,
)
```

### 实现中的坑

**坑 1：弃权阈值 0.60 不是通用最优值**

论文的 0.60 是在其数据集上调出来的。用校准曲线找你的最优值：

```python
import numpy as np

def find_optimal_threshold(confidences, correct_flags, target_precision=0.95):
    """找到使精度 ≥ target_precision 的最低弃权阈值"""
    for tau in np.arange(0.50, 1.0, 0.05):
        mask = np.array(confidences) >= tau
        if mask.sum() == 0:
            continue
        precision = np.array(correct_flags)[mask].mean()
        coverage = mask.mean()
        if precision >= target_precision:
            return tau, precision, coverage  # 精度达标时的最低阈值
    return 1.0, 1.0, 0.0
```

**坑 2：softmax 置信度天然过度自信**

ViT 在 OOD 数据上仍然会输出虚高的置信度（已知校准问题）。用 Temperature Scaling 缓解：

```python
class TemperatureScaledClassifier(torch.nn.Module):
    def __init__(self, model, temperature=1.5):
        super().__init__()
        self.model = model
        # temperature 可以在验证集上优化，一般 1.0-3.0 之间
        self.temperature = torch.nn.Parameter(torch.tensor([temperature]))

    def forward(self, **inputs):
        logits = self.model(**inputs).logits
        return logits / self.temperature  # 软化分布，降低过度自信
```

**坑 3：类别不平衡时看 F1，别看准确率**

minivan 样本量少，模型倾向于把它错认为 SUV 或 passenger car，但整体准确率看起来仍然不错。训练时给小类更高权重：

```python
# 按类别频率的倒数设置权重
class_weights = torch.tensor([1.0, 0.8, 1.2, 2.5, 1.5, 1.1])  # minivan 权重最高
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
```

## 实验：论文说的 vs 现实

| 指标 | In-Distribution | OOD 域外 |
|------|----------------|---------|
| 整体准确率 | 0.94 | 0.89 |
| Minivan F1 | 0.91 | 0.72 |
| SUV F1 | 0.97 | ~0.92 |
| 弃权率 | 2.4% | 25.0% |

OOD 弃权率从 2.4% 跳到 25.0%，而不是出现大量错误分类——这正是弃权机制设计正确的标志。如果弃权率没有上升但准确率下降，说明模型对自己的错误"浑然不知"，那才是真正危险的。

Minivan 的大幅衰退是最诚实的部分：北美 minivan 和其他地区的同类车型外形差异很大，这不是技术问题，是**数据分布问题**。89% OOD 准确率本身相当不错，但需要注意 OOD 测试集只有 311 个样本，置信区间很宽，结论需要谨慎解读。

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 路侧固定摄像头，角度变化较小 | 无人机俯视或极端仰视角度 |
| 需要区分 SUV/皮卡/厢货等细粒度类别 | COCO 粗类别标签就够用的场景 |
| 可以接受一定比例的 unknown 输出 | 每帧必须输出确定标签的场景 |
| 安全敏感应用（宁可不知道，不能猜错） | 延迟要求极高（< 30ms/frame）的实时流 |
| 有百级以上标注样本可供微调 | 完全零样本、无法采集标注数据的场景 |

## 我的观点

**弃权机制是这篇论文真正的贡献，不是分类器本身。**

两阶段流水线是计算机视觉工程的标准模式，ViT 微调没有什么新意。但把"我不知道"设计为一等公民输出，并且用**弃权率上升来诊断域漂移**——这是值得借鉴的系统设计思想，在医疗影像、工业质检、自动驾驶感知标注等场景同样适用。

大多数工业部署中，"沉默的误分类"（模型以高置信度给出错误答案）比"已知的未知"（模型说不知道）危害更大。

不容忽视的局限性：

- OOD 评估只有 311 个样本，统计显著性有限
- 0.60 阈值的选取没有经过系统的消融实验
- 没有讨论夜间、雨天等极端场景
- 模型基于北美道路数据训练，用于其他地区建议重新收集数据微调

**官方开源代码**：论文（[arxiv.org/abs/2606.05149](https://arxiv.org/abs/2606.05149v1)）中提及完整开源，包括推理脚本、训练代码和模型权重，可直接用于路侧视频档案的批量处理。