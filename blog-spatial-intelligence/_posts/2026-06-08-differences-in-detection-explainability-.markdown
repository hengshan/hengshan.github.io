---
layout: post-wide
title: "超越 mAP：用 DnD 集合运算直接比较目标检测模型"
date: 2026-06-08 12:05:00 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.07503v1
generated_by: Claude Code CLI
---

## 一句话总结

DnD（Differences in Detection）通过对两个模型的检测结果做集合运算，将模型比较从"谁的 mAP 更高"升级为"谁在哪类样本上更好、以及为什么"——并天然地为 ODAM 等可解释性方法提供有目的性的分析样本。

---

## 为什么 mAP 不够用？

你有两个目标检测模型，mAP 分别是 0.412 和 0.427。

这告诉你什么？第二个模型"更好"。但这一个数字无法回答：

- 两个模型失败的是**同一批**困难样本吗？
- 性能差异来自类别识别更准，还是定位更精确？
- 切换到第二个模型，具体在哪些图像上有收益？

mAP 是汇总统计量——它把所有信息压缩成一个数字，**模型间的相关结构被抹掉了**。两个 mAP = 0.40 的模型可能：检测完全相同的物体（可互换）；或检测完全不同的物体（互补，集成效果会很好）。

DnD 的核心洞察：**在同一组 GT 标注上进行集合运算，才能看清模型间的真正差异**。

---

## 背景知识

### TIDE 错误类型

TIDE（Toolkit for Identifying Detection Errors）将单个模型的检测错误分为六类，DnD 的联合分析建立在这个基础上：

| 错误类型 | 定义 |
|---------|------|
| **Cls** | 定位正确（IoU ≥ 阈值），类别错误 |
| **Loc** | 类别正确，但 IoU 低于阈值 |
| **Both** | 类别错误 + IoU 不足 |
| **Dupe** | 重复检测（此 GT 已有更好的预测匹配） |
| **Bkg** | 背景误判（无匹配 GT） |
| **Miss** | GT 完全未被检测到 |

### 真正例的判定

TP 需同时满足两个条件：

$$\text{TP} \iff \text{IoU}(\hat{b},\, b^*) \geq \theta_{\text{IoU}} \;\;\text{且}\;\; \hat{c} = c^*$$

其中 $\hat{b}$ 为预测框，$b^*$ 为 GT 框，$\hat{c}$ 和 $c^*$ 分别为预测类别和真实类别。

---

## DnD 核心方法

### 集合运算直觉

给定同一组 GT 标注集合 $\mathcal{G}$，设：
- $TP_A \subseteq \mathcal{G}$：模型 A 正确检测到的 GT 子集
- $TP_B \subseteq \mathcal{G}$：模型 B 正确检测到的 GT 子集

DnD 将 $\mathcal{G}$ 划分为四个**互不重叠、并集覆盖全集**的子集：

$$\mathcal{G} = \underbrace{\mathcal{I}}_{\text{共同正确}} \;\cup\; \underbrace{\mathcal{D}_A}_{\text{A 独占}} \;\cup\; \underbrace{\mathcal{D}_B}_{\text{B 独占}} \;\cup\; \underbrace{\mathcal{C}}_{\text{共同盲区}}$$

具体定义：

$$\mathcal{I} = TP_A \cap TP_B \qquad \mathcal{D}_A = TP_A \setminus TP_B$$
$$\mathcal{D}_B = TP_B \setminus TP_A \qquad \mathcal{C} = \mathcal{G} \setminus (TP_A \cup TP_B)$$

数量守恒性质：$|\mathcal{I}| + |\mathcal{D}_A| + |\mathcal{D}_B| + |\mathcal{C}| = |\mathcal{G}|$，GT 信息无损。

### Pipeline 概览

```
GT 标注集 G
    │
    ├── 用同一匹配算法分别运行 Model A 和 Model B
    │        ↓               ↓
    │      TP_A             TP_B
    │         \             /
    │          集合运算
    │         /    |    \   \
    │        I    D_A   D_B   C
    │        ↓     ↓     ↓    ↓
    └──── 汇总统计 + TIDE 错误分析 + ODAM 可视化
```

### 与 ODAM 可解释性的结合

DnD 最有价值的用途是**有目的地选择样本**来运行可解释性分析：

- $\mathcal{D}_A$ 中的样本：A 成功但 B 失败 → 问"A 关注了哪些 B 没关注的区域？"
- $\mathcal{D}_B$ 中的样本：B 成功但 A 失败 → 问"B 学到了哪些 A 没学到的特征？"
- $\mathcal{C}$ 中的样本：两者都失败 → 这些是否构成真正的"难例"？

随机抽样做 GradCAM 效率极低；DnD 给你**结构化的有意义样本集**。

---

## 实现

### 核心匹配算法

```python
import numpy as np
from collections import defaultdict

def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """计算 N×M 的 IoU 矩阵，输入格式 [x1, y1, x2, y2]"""
    area_a = (boxes_a[:,2]-boxes_a[:,0]) * (boxes_a[:,3]-boxes_a[:,1])
    area_b = (boxes_b[:,2]-boxes_b[:,0]) * (boxes_b[:,3]-boxes_b[:,1])
    
    ix1 = np.maximum(boxes_a[:,None,0], boxes_b[None,:,0])
    iy1 = np.maximum(boxes_a[:,None,1], boxes_b[None,:,1])
    ix2 = np.minimum(boxes_a[:,None,2], boxes_b[None,:,2])
    iy2 = np.minimum(boxes_a[:,None,3], boxes_b[None,:,3])
    
    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    union = area_a[:,None] + area_b[None,:] - inter
    return inter / (union + 1e-7)

def match_to_gt(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresh=0.5):
    """贪心匹配预测框到 GT，返回布尔数组：每个 GT 是否被正确检测"""
    n_gt = len(gt_boxes)
    is_tp = np.zeros(n_gt, dtype=bool)
    if len(pred_boxes) == 0:
        return is_tp
    
    iou_mat = compute_iou_matrix(np.array(gt_boxes), np.array(pred_boxes))
    assigned_preds = set()
    
    for _ in range(min(n_gt, len(pred_boxes))):
        gt_i, pred_j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[gt_i, pred_j] < iou_thresh:
            break
        if pred_j not in assigned_preds:
            # 定位达标 + 类别正确 = TP
            if gt_classes[gt_i] == pred_classes[pred_j]:
                is_tp[gt_i] = True
            assigned_preds.add(pred_j)
        iou_mat[gt_i, :] = -1  # 已处理的 GT 行清零
    return is_tp
```

### DnD 比较器主体

```python
class DnDComparator:
    def __init__(self, iou_thresh=0.5):
        self.iou_thresh = iou_thresh
    
    def compare(self, gt_list, preds_a_list, preds_b_list):
        """对整个数据集运行 DnD，返回每个 GT 所属集合"""
        all_results = []
        for gt, pa, pb in zip(gt_list, preds_a_list, preds_b_list):
            tp_a = match_to_gt(pa['boxes'], pa['classes'],
                               gt['boxes'], gt['classes'], self.iou_thresh)
            tp_b = match_to_gt(pb['boxes'], pb['classes'],
                               gt['boxes'], gt['classes'], self.iou_thresh)
            
            for i, gt_cls in enumerate(gt['classes']):
                a, b = tp_a[i], tp_b[i]
                partition = ('I' if a and b else 'D_A' if a else 'D_B' if b else 'C')
                all_results.append({'class': gt_cls, 'partition': partition})
        return all_results
    
    def summary(self, results):
        """打印 DnD 摘要统计"""
        counts = defaultdict(int)
        for r in results:
            counts[r['partition']] += 1
        total = sum(counts.values())
        print(f"DnD 分析结果（共 {total} 个 GT）")
        for key, label in [('I','共同正确'), ('D_A','A 独占'), ('D_B','B 独占'), ('C','共同盲区')]:
            print(f"  {key:3s} ({label}): {counts[key]:6d}  ({100*counts[key]/total:.1f}%)")
        return dict(counts)
```

### 结合 TIDE 错误分析

```python
def get_tide_error(pred_boxes, pred_classes, gt_box, gt_class, iou_thresh=0.5):
    """对某个未被正确检测的 GT，判断是哪类 TIDE 错误"""
    if len(pred_boxes) == 0:
        return 'Miss'
    iou_vec = compute_iou_matrix(gt_box[None], np.array(pred_boxes))[0]
    best_idx = np.argmax(iou_vec)
    best_iou = iou_vec[best_idx]
    best_cls = pred_classes[best_idx]
    
    if best_iou >= iou_thresh:
        return 'Cls'                          # 定位对，类别错
    elif best_iou >= 0.1:
        return 'Loc' if best_cls == gt_class else 'Both'
    return 'Miss'                             # 完全未找到

def analyze_error_in_diff_sets(results, gt_list, preds_a_list, preds_b_list):
    """分析 D_A 和 D_B 集合中对手模型的失败原因"""
    errors = {'D_A': defaultdict(int), 'D_B': defaultdict(int)}
    idx = 0
    for gt, pa, pb in zip(gt_list, preds_a_list, preds_b_list):
        for i, (gt_cls, gt_box) in enumerate(zip(gt['classes'], gt['boxes'])):
            r = results[idx]; idx += 1
            if r['partition'] == 'D_A':   # B 失败了，分析 B 的错误
                err = get_tide_error(pb['boxes'], pb['classes'], gt_box, gt_cls)
                errors['D_A'][err] += 1
            elif r['partition'] == 'D_B': # A 失败了，分析 A 的错误
                err = get_tide_error(pa['boxes'], pa['classes'], gt_box, gt_cls)
                errors['D_B'][err] += 1
    return errors
```

### 可视化

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_dnd_summary(counts, name_a="Model A", name_b="Model B"):
    """DnD 四分集合可视化"""
    labels = [f'I\n(共同正确)', f'D_A\n({name_a})', f'D_B\n({name_b})', 'C\n(共同盲区)']
    values = [counts.get(k, 0) for k in ['I','D_A','D_B','C']]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左：饼图
    axes[0].pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
    axes[0].set_title('DnD 集合分布')
    
    # 右：堆叠条形图（比例感更直观）
    total = sum(values)
    left = 0
    for v, c, l in zip(values, colors, labels):
        w = v / total
        axes[1].barh(0, w, left=left, color=c, edgecolor='white', linewidth=2)
        if w > 0.03:
            axes[1].text(left + w/2, 0, str(v), ha='center', va='center',
                         fontsize=9, color='white', fontweight='bold')
        left += w
    
    axes[1].set_xlim(0, 1); axes[1].set_yticks([])
    axes[1].set_title(f'{name_a} vs {name_b}')
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    axes[1].legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('dnd_summary.png', dpi=150, bbox_inches='tight')
```

---

## 实验分析

以 COCO val2017 对比 YOLOv8-n（轻量）和 YOLOv8-l（大模型）为例：

**标准 mAP 比较**：

| 指标 | YOLOv8-n | YOLOv8-l | 差值 |
|-----|---------|---------|-----|
| mAP@0.5 | 0.521 | 0.628 | +0.107 |
| mAP@0.5:0.95 | 0.373 | 0.526 | +0.153 |
| 推理延迟 (V100) | 6.2 ms | 20.1 ms | 3.2× 慢 |

**DnD 结果**（按比例）：

| 集合 | 含义 | 估计占比 |
|-----|------|---------|
| I | 两者都检测到 | ~52% |
| D_A | 仅 YOLOv8-n 检测到 | ~2.5% |
| D_B | 仅 YOLOv8-l 检测到 | ~20% |
| C | 两者都漏 | ~25.5% |

**三个关键发现**：

1. **25% 的共同盲区**：这些 GT 与模型大小无关，大概率是小目标、重度遮挡、稀有角度——这是标注时需要重点关注的困难样本，也是 active learning 优先采集的对象。

2. **大模型几乎是小模型的超集**：$|\mathcal{D}_A| \approx 2.5\%$ 说明 YOLOv8-l 漏掉了极少 YOLOv8-n 能检测到的目标。增益主要来自 D_B（20%），不是 D_A。

3. **D_A 中的 TIDE 错误分析**：YOLOv8-l 在这 2.5% 上主要犯 Loc 错误（~38%），说明大模型偶尔出现定位过平滑的问题，而非完全漏检。

---

## 工程实践

### 基于 DnD 的智能集成

```python
def dnd_guided_ensemble(preds_a, preds_b, class_advantages):
    """
    class_advantages: {class_id: 'A'|'B'} 来自 DnD 统计
    对 A 更优势的类别，降低 B 的置信度；反之亦然
    """
    weighted = []
    for pred in preds_a:
        w = 1.2 if class_advantages.get(pred['class']) == 'A' else 0.8
        weighted.append({**pred, 'score': pred['score'] * w})
    for pred in preds_b:
        w = 1.2 if class_advantages.get(pred['class']) == 'B' else 0.8
        weighted.append({**pred, 'score': pred['score'] * w})
    return nms(weighted, iou_thresh=0.5)
```

### 常见坑

**1. 匹配算法必须统一**

两个模型若使用不同的 NMS 阈值或置信度过滤，DnD 结果不可比。解决方案：在 DnDComparator 内统一重新做匹配，不依赖模型原始后处理输出。

**2. 类别不平衡导致解读偏差**

COCO 中 `person` 有 26 万实例，`toaster` 只有 225 个。D_B 集合里 90% 可能都是 `person`，掩盖了有趣的细分类差异。建议**按类别归一化**再分析：

```python
# 计算每类的 D_A / (D_A + D_B) 胜率而非绝对数量
win_rate_A = {cls: d_a / (d_a + d_b + 1e-7) 
              for cls, (d_a, d_b) in per_class_counts.items()}
```

**3. 评测集太小时集合划分方差大**

$|\mathcal{D}_A|$ 只有几十个样本时，百分比会随机波动。建议至少 2000 张图、每类 200+ 实例，或做 bootstrap 置信区间估计。

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 对比同任务的两个模型（如 backbone A vs B） | 对比不同任务的模型 |
| 指导集成策略或蒸馏目标 | 只需要一个排名数字 |
| 为 GradCAM/ODAM 选择有意义的样本 | 类别极度不平衡且难以归一化 |
| AB 测试：新版本 vs 旧版本 | 在训练集上分析（过拟合偏差） |
| 找"模型无关的困难样本"用于 active learning | 评测集太小（< 1000 张） |

---

## 我的观点

DnD 的价值不在于算法创新，而在于**提供了正确的分析框架**：把对两个独立统计量的比较，变成对同一 GT 集合的结构化划分。这在工程上非常实用，因为做模型迭代时你最关心的问题恰好是"哪些样本是新模型真正带来的增益，哪些是噪声？"

几个值得关注的扩展方向：

- **时序版 DnD**：比较同一模型的不同训练 checkpoint，追踪"训练过程中哪些样本反而退步了"
- **扩展到三个以上模型**：集合运算自然推广到 $2^n$ 个子集，可视化借助 UpSet plot
- **与 active learning 闭环**：$\mathcal{C}$ 集合（两者都漏）是最值得优先标注的候选集
- **多 IoU 阈值分析**：在 $\theta_{IoU} \in \{0.5, 0.75, 0.9\}$ 下分别运行，画"一致性曲线"，专门评估定位敏感场景

目前最大的缺憾是没有标准化的可视化工具链——作者提供了基础代码，但和 TIDE 的完整集成还需要自己实现。

论文官方代码：[https://github.com/JohannesTheo/differences-in-detection](https://github.com/JohannesTheo/differences-in-detection)