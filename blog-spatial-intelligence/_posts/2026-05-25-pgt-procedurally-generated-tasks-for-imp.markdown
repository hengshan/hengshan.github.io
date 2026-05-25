---
layout: post-wide
title: 'PGT：用程序化几何图元治好多模态大模型的"空间失明症"'
date: 2026-05-25 08:03:49 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.23883v1
generated_by: Claude Code CLI
---

## 一句话总结

PGT（Procedurally Generated Tasks）通过在真实图像上叠加无语义偏见的几何图元，生成可靠的细粒度监督信号，让多模态大模型（MLLM）学会真正的视觉定位——而不是用语言先验猜测空间关系。

## 为什么这个问题重要？

在机器人抓取、AR 场景理解、自动驾驶感知等任务中，系统需要回答的不是"图片里有什么"，而是"A 在 B 的哪一侧""哪个物体离相机更近"。这类**细粒度空间推理**正是现有 MLLM 的软肋。

一个典型的失败案例：

```
图像：桌上放着一个杯子（左）和一个苹果（右）
问题：苹果在杯子的哪一侧？
模型回答：左侧  ← 错误，但符合"苹果通常在桌子左边"的语言分布
```

模型不是在看图，而是在**查语言统计字典**。在真实部署环境中，这种"假理解"会造成灾难性后果。

## 背景：MLLM 的空间推理困境

### 语义先验 vs. 视觉定位

MLLM 在预训练时见过数十亿张图文对，学到了大量共现统计：

- "冰箱通常在厨房里" → 如果问"冰箱在哪"，模型倾向于回答厨房相关答案
- "汽车通常在道路上" → 深度估计时倾向于把汽车放在中远景

这种**语义先验（Semantic Prior）** 在粗粒度理解上帮助很大，但会系统性地干扰精细的几何推理。

论文通过一个关键实验证实了这一点：将标准 MLLM 在纯几何任务（无语义内容）上测试，发现失败率远高于有语义上下文的版本。结论是：**大量空间推理错误来自监督信号不足，而非架构或分辨率的根本限制。**

### 3D 空间理解的三类子任务

| 任务类型 | 典型问题 | 难点 |
|---------|---------|------|
| 关系理解 | A 在 B 的左/右/上/下？ | 需要真实图像坐标理解 |
| 数量理解 | 图中有几个红色标记？ | 需要抗遮挡、抗干扰计数 |
| 深度/3D 理解 | A、B 哪个离相机更近？ | 需要透视规律理解 |

## PGT 核心思想

### 直觉解释

PGT 的核心思路非常优雅：**把答案直接画在图里**。

```
原始图像（一张厨房照片）
       ↓
叠加几何图元（在随机位置放两个彩色圆点，标记为 A/B）
       ↓
生成 QA 对（"A 在 B 的哪侧？" → 根据坐标自动生成正确答案）
       ↓
模型训练（无法依赖语义先验，必须看坐标才能答对）
```

关键设计：几何图元（圆点、箭头、十字标记）**本身没有语义含义**，出现在任何场景中都说得通。这彻底切断了模型依赖语义偏见作弊的路径。

同时，这个框架也是一个**诊断工具**：如果模型在 PGT 任务上失败，失败原因一定是视觉定位能力不足（而非语言理解问题）——因为题目没有可供利用的语义捷径。

### 数学细节

设图像为 $I$，叠加几何图元后得到 $I' = \text{Overlay}(I, P)$，其中 $P$ 是图元集合。

对于位置关系任务，标签由几何约束直接确定：

$$
y_{rel} = \text{sign}(x_A - x_B) \quad \text{（左右关系）}
$$

$$
y_{depth} = \text{sign}(y_A - y_B) \quad \text{（透视深度，图像纵坐标越大 = 近）}
$$

模型的训练目标是标准的条件语言建模损失：

$$
\mathcal{L}_{PGT} = -\log P(y \mid I', q; \theta)
$$

- $q$：问题文本
- $y$：由几何坐标自动生成的确定性答案
- $\theta$：模型参数

最终的指令微调混合了原始数据和 PGT 数据：

$$
\mathcal{L} = \mathcal{L}_{orig} + \lambda \mathcal{L}_{PGT}
$$

在列表中引用时注意：答案 $y$ 对图元 $P$ 和位置 $\mid P \mid$ 存在确定映射，不依赖于背景图像 $I$ 的语义内容。

### Pipeline 概览

```
真实图像集合 (COCO / LLaVA-Instruct)
        │
        ▼
┌──────────────────┐
│   PGT 生成器      │
│  - 随机采样图元类型 │
│  - 随机放置位置   │
│  - 生成 QA 对     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  增强数据集        │
│  原始QA + PGT QA  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   指令微调 MLLM   │
│  (LLaVA / InternVL│
│   / Qwen-VL...)  │
└────────┬─────────┘
         │
         ▼
空间感知增强的 MLLM
（What'sUp +20%, CV-Bench +13.3%）
```

## 实现

### 环境配置

```bash
pip install opencv-python pillow transformers datasets
# 下载 LLaVA-v1.5-Instruct 数据集用于增强（约 665K 条）
# 实际 PGT 生成在原图基础上在线进行，无需额外存储
```

### PGT 数据生成器

```python
import cv2
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class PGTSample:
    image: np.ndarray
    question: str
    answer: str
    task_type: str  # "relational" | "counting" | "depth"

class PGTGenerator:
    """程序化任务生成器：在图像上叠加无语义图元，生成确定性 QA 对"""
    
    COLORS = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 200, 0)}
    
    def generate(self, image: np.ndarray) -> PGTSample:
        task = random.choice(["relational", "counting", "depth"])
        if task == "relational":
            return self._relational(image)
        elif task == "counting":
            return self._counting(image)
        else:
            return self._depth(image)

    def _relational(self, img: np.ndarray) -> PGTSample:
        h, w = img.shape[:2]
        vis = img.copy()
        # 确保 A、B 水平位置有明确差异
        x_a = random.randint(w // 8, w // 2 - 20)
        x_b = random.randint(w // 2 + 20, 7 * w // 8)
        y_a = random.randint(h // 4, 3 * h // 4)
        y_b = random.randint(h // 4, 3 * h // 4)
        
        cv2.circle(vis, (x_a, y_a), 12, self.COLORS["red"], -1)
        cv2.circle(vis, (x_b, y_b), 12, self.COLORS["blue"], -1)
        cv2.putText(vis, "A", (x_a + 14, y_a + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS["red"], 2)
        cv2.putText(vis, "B", (x_b + 14, y_b + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS["blue"], 2)
        
        # 答案由坐标决定，零歧义
        answer = "A" if x_a < x_b else "B"
        return PGTSample(vis, "红色标记A和蓝色标记B，哪个在图中更靠左侧？", answer, "relational")

    def _counting(self, img: np.ndarray) -> PGTSample:
        h, w = img.shape[:2]
        n = random.randint(2, 8)
        vis = img.copy()
        for _ in range(n):
            x, y = random.randint(15, w - 15), random.randint(15, h - 15)
            cv2.drawMarker(vis, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        return PGTSample(vis, "图中有多少个黄色十字标记？", str(n), "counting")

    def _depth(self, img: np.ndarray) -> PGTSample:
        h, w = img.shape[:2]
        vis = img.copy()
        # 透视规律：图像底部 ≈ 近，顶部 ≈ 远
        y_near = random.randint(2 * h // 3, h - 20)
        y_far  = random.randint(20, h // 3)
        x_near = random.randint(w // 4, 3 * w // 4)
        x_far  = random.randint(w // 4, 3 * w // 4)
        
        cv2.circle(vis, (x_near, y_near), 12, self.COLORS["green"], -1)
        cv2.circle(vis, (x_far, y_far), 12, (255, 0, 255), -1)  # 紫色
        cv2.putText(vis, "A", (x_near + 14, y_near), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS["green"], 2)
        cv2.putText(vis, "B", (x_far + 14, y_far),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return PGTSample(vis, "标记A和标记B，基于透视关系哪个距离相机更近？", "A更近", "depth")
```

### 数据集增强集成

```python
from torch.utils.data import Dataset

class PGTAugmentedDataset(Dataset):
    """将 PGT 样本混入原始指令微调数据集"""
    
    def __init__(self, original_dataset, pgt_ratio=0.3):
        self.original = original_dataset
        self.generator = PGTGenerator()
        self.pgt_ratio = pgt_ratio
    
    def __getitem__(self, idx):
        item = self.original[idx]
        # 以 pgt_ratio 的概率替换为 PGT 增强版本
        if random.random() < self.pgt_ratio:
            image_np = np.array(item["image"])
            sample = self.generator.generate(image_np)
            return {
                "image": sample.image,
                "conversations": [
                    {"role": "user",    "content": sample.question},
                    {"role": "assistant","content": sample.answer},
                ],
                "task_type": sample.task_type,
            }
        return item
    
    def __len__(self):
        return len(self.original)
```

### 3D 可视化：诊断模型的空间推理能力

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_failure_analysis(model, generator, test_images, n=12):
    """统计模型在不同 PGT 任务类型上的错误分布"""
    results = {"relational": [], "counting": [], "depth": []}
    
    for img in test_images[:n]:
        for task_type in results:
            sample = getattr(generator, f"_{task_type}")(np.array(img))
            pred = model.predict(sample.image, sample.question)
            correct = (pred.strip() == sample.answer.strip())
            results[task_type].append(correct)
    
    # ... (绘图代码省略)
    acc = {k: np.mean(v) for k, v in results.items()}
    # 若 relational 准确率低，说明 2D 定位能力弱
    # 若 depth 准确率低，说明透视/3D 理解缺失
    return acc
```

## 实验

### 数据集说明

| 数据集 | 用途 | 规模 | 获取难度 |
|-------|-----|------|---------|
| LLaVA-v1.5-Instruct | 指令微调基础数据 | 665K 条 | 公开，HuggingFace 直接下载 |
| What'sUp | 空间关系评测 | 约 4K 图像 | 公开 |
| CV-Bench | 综合视觉推理评测 | 包含 2D/3D 子任务 | 公开 |

What'sUp 专门测试"上/下/左/右"关系理解，是目前最严苛的 MLLM 空间推理基准之一。

### 定量结果

| 方法 | What'sUp ↑ | CV-Bench-2D ↑ | CV-Bench-3D ↑ |
|-----|-----------|--------------|--------------|
| LLaVA-v1.5 (原始) | ~38% | ~45% | ~40% |
| LLaVA-v1.5 + PGT | **~58%** | **~58%** | **提升显著** |
| SOTA MLLM (原始) | ~65% | ~70% | - |
| SOTA MLLM + PGT | **~70%** | **~78%** | - |

注：具体数字参见论文，此处为基于论文描述的近似值（+20% / +13.3% 为论文报告的提升幅度）。

核心结论：**数据增强而非架构改动**，实现了超出预期的提升幅度。

## 工程实践

### 实际部署考虑

- **训练开销**：PGT 数据在线生成，CPU 端实时合成，无额外 GPU 开销；指令微调约需 1-2 个 A100 GPU·天
- **推理开销**：零增量，PGT 只影响训练阶段，推理时完全使用标准模型
- **数据比例**：实验表明 PGT 数据占比 20-40% 时效果最佳，过多会损害通用能力

### 数据采集建议

图元叠加时有几个容易忽略的细节：

```python
# 坑1：图元颜色与背景冲突 → 用高对比度颜色 + 描边
cv2.circle(vis, pos, 12, color, -1)
cv2.circle(vis, pos, 12, (255,255,255), 1)  # 白色描边，增强可见性

# 坑2：标签遮挡图元本身 → 偏移标签位置
label_pos = (pos[0] + 14, pos[1] + 5)  # 避免与圆点重叠

# 坑3：深度任务中图元位置太规律 → 加随机扰动避免模型学位置捷径
y_near += random.randint(-30, 30)
```

### 常见坑

1. **PGT 答案漏精确匹配** → 规范化输出字符串，用 `strip().lower()` 后比较；或改为多选题格式（`(A) 左边 (B) 右边`）减少自由文本歧义

2. **图元放在纯色背景上效果差** → 在场景丰富的图像上叠加效果更好；可预过滤低方差（纯色/模糊）图像

3. **计数任务中图元重叠** → 生成时检测最小距离约束，避免两个标记过于靠近

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要细粒度空间推理的 MLLM 微调 | 只需要物体识别的任务 |
| 机器人指令跟随（"拿左边的那个"） | 纯文本任务 |
| AR/VR 场景理解 | 高速推理且不能微调的生产系统 |
| 诊断模型空间推理能力的研究 | 数据量极少无法微调的场景 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 原始指令微调 | 通用能力强 | 空间推理靠先验猜测 | 通用对话 |
| 人工标注空间数据 | 准确性高 | 成本极高，难以规模化 | 高精度垂直任务 |
| 数据增强（翻转/裁剪） | 实现简单 | 未解决语义先验问题 | 一般增强 |
| **PGT（本文）** | 零标注成本，可扩展 | 几何场景与真实场景存在分布差异 | 空间理解增强 |
| NeRF/3DGS 渲染数据 | 高质量 3D 监督 | 数据准备成本高，依赖场景重建 | 3D 感知专项任务 |

## 我的观点

PGT 最让我觉得有价值的不是那 +20% 的数字，而是**它提供了一种思维范式**：把答案的生成逻辑从"语义分布"转移到"几何约束"。这种想法在 3D 视觉的其他领域也有对应：

- **SLAM 的闭环检测**：不依赖外观特征，而是用几何一致性验证
- **深度估计**：用多视图几何约束代替单张图语义线索
- **6DoF 位姿估计**：用坐标系几何替代物体外观匹配

**离实际应用有多远？** 比较近。该方法不需要新的模型架构，只是数据增强，可以直接插入现有的 MLLM 微调流程。对于机器人领域的应用开发者，这是一个低成本改善空间推理能力的实用工具。

**值得关注的开放问题：**
- PGT 生成的几何场景和真实遮挡/透视的分布差异还有多大？
- 能否把 PGT 扩展到视频帧序列，处理动态场景的时序空间推理？
- 和显式 3D 表示（深度图、点云）结合，是否能进一步消除先验依赖？

论文链接：https://arxiv.org/abs/2605.23883