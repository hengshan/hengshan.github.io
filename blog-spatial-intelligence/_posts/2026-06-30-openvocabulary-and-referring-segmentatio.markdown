---
layout: post-wide
title: "GaussDet：让 2D 开放词汇检测器为 3D 高斯做语义定位"
date: 2026-06-30 12:03:54 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.30638v1
generated_by: Claude Code CLI
---

## 一句话总结

GaussDet 把"给 3D 高斯场景打语义标签"这件事，从"把 CLIP 特征硬塞进每个高斯里"换成了"用现成的 2D 开放词汇检测器在多个视角投票"，不仅能做开放词汇分割，还能零样本支持"桌子左边那把椅子"这种指代表达定位（referring expression grounding）。

## 为什么这个问题重要？

3D Gaussian Splatting（3DGS）已经成为新视角合成的事实标准之一：渲染快、质量高、可编辑性强。但一个纯几何/外观的 3DGS 场景是"哑巴"的——它知道每个点的颜色和位置，却不知道"这是什么"。在具身智能（embodied AI）、机器人抓取、AR 标注这些场景里，我们需要的是"把杯子拿过来"而不是"渲染出杯子所在位置的像素"。这就要求场景具备**开放词汇的语义理解能力**，并且最好能支持自然语言里更复杂的指代表达，而不只是简单名词。

现有主流方案（如 LangSplat、Gaussian Grouping）的做法是把 CLIP 的高维视觉-语言特征蒸馏进每个高斯的属性里，渲染时插值出像素级的语言特征，再和文本编码做相似度匹配。这条路有两个绕不开的坑：

- **实例分组依赖弱监督**：要么需要预先指定实例数量，要么依赖 SAM 等工具做自底向上分组，分组噪声会直接污染语义。
- **CLIP 的语义粒度太粗**：CLIP 擅长"这是一张猫的图片"这类全局语义匹配，但很难处理"靠近窗户的那张椅子"这种需要空间关系推理的指代表达。

GaussDet 的核心创新在于：不再依赖稠密的 CLIP 特征蒸馏，而是直接调用**离散的、带指代定位能力的 2D 开放词汇检测器**（如 Grounding DINO 一类模型），把检测器在多视角下的判断**投票聚合**到 3D 实例上，用聚合本身作为去噪手段。

## 背景知识

### 3D 表示方式怎么选

| 表示 | 优点 | 缺点 |
|-----|------|------|
| 点云 | 直接、稀疏高效 | 缺少连续表面信息 |
| 体素/隐式（NeRF） | 连续、可微 | 渲染慢，编辑困难 |
| 3D Gaussian Splatting | 渲染快（实时）、显式、可编辑 | 表示离散，实例边界天然模糊 |

3DGS 用一堆带位置、协方差、颜色、不透明度的各向异性高斯椭球表示场景，渲染时按深度排序做 alpha blending。这个"显式 + 可附加属性"的特性，正是语言蒸馏类方法能给每个高斯挂上一个特征向量的原因——也是 GaussDet 给每个高斯挂"实例特征"而非"语言特征"的基础。

### 现有语言蒸馏方法回顾

LangSplat、Gaussian Grouping 这类方法本质上是：

$$
F(\mathbf{p}) = \sum_{i \in N} f_i \, \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)
$$

这和 3DGS 原始的颜色渲染公式形式完全一致，只是把颜色 $c_i$ 换成了高维语言特征 $f_i$。问题在于 $f_i$ 要么是 CLIP 特征本身（维度高、训练慢、需要 PCA 压缩），要么是依赖 SAM 分组得到的实例 ID 编码——一旦 SAM 在某个视角下分错了，错误会被蒸馏进场景里，难以靠单视角自我纠正。

### 开放词汇 2D 检测器

像 Grounding DINO、OWL-ViT 这类模型，输入图片 + 文本（可以是名词，也可以是带空间关系的指代表达），直接输出对应的检测框，并且经过大规模图文对训练，具备较强的**短语定位（phrase grounding）**能力。GaussDet 的思路是：与其把这种能力"蒸馏"进 3D 场景，不如在推理时多视角**调用**它，再把结果投影回 3D。

## 核心方法

### 直觉解释

把场景想象成一群学生在不同角度观察同一个物体，每个学生（视角）独立用检测器给出判断："这块区域是椅子，置信度 0.8"。GaussDet 做的事情是：

1. 先把高斯分成若干"3D 实例组"（不依赖语义，只是几何/外观上的聚类）；
2. 把每个实例组投影渲染到所有可见视角；
3. 在每个视角上跑 2D 开放词汇检测器，看检测框和渲染出来的实例区域重叠多少；
4. 把所有视角的"投票"汇总成一个该实例的标签分布——这就是论文里的 **View-Aggregated Semantic Label Distribution（VASD）**。

多视角投票本质上是一种集成（ensemble）去噪：单个视角检测器可能因遮挡、角度问题判断失误，但只要多数视角判断一致，错误就会被平均掉。

### 数学细节

**实例特征渲染**：和颜色渲染同构，每个高斯 $i$ 附带一个低维实例嵌入 $e_i$（不是 CLIP 特征，只用于做 3D 聚类）：

$$
E(\mathbf{p}) = \sum_{i \in N} e_i \, \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)
$$

通过对比学习（同一 2D mask 内的像素嵌入拉近，不同 mask 拉远）训练 $e_i$，再聚类得到离散的 3D 实例划分 $\{G_k\}$。

**VASD 聚合**：对实例 $k$，在视角集合 $\mathcal{V}_k$（实例可见的视角）上聚合标签 $\ell$ 的票数：

$$
\text{VASD}_k(\ell) = \frac{1}{Z_k}\sum_{v \in \mathcal{V}_k} w_v \cdot \text{IoU}\!\left(M_k^v, B_\ell^v\right) \cdot s_\ell^v
$$

其中 $M_k^v$ 是实例 $k$ 在视角 $v$ 下的渲染掩码，$B_\ell^v$ 是检测器给出的标签 $\ell$ 的检测框，$s_\ell^v$ 是该检测的置信度，$w_v$ 是视角可见度权重（越正对、面积越大权重越高），$Z_k$ 归一化。

查询时：

- 简单名词查询：直接在 VASD 上取 $\arg\max_\ell \text{VASD}_k(\ell)$ 做开放词汇分割；
- 指代表达查询（如"桌子左边的椅子"）：把整句话直接喂给检测器的 grounding 接口（检测器本身支持），在多视角下重复上面的投票流程，**不需要重新训练**，这是"零样本扩展"的关键。

### Pipeline 概览

```
多视角RGB+位姿 → 3DGS重建(+实例嵌入属性)
    → 渲染实例特征图 → 3D聚类得到实例组 {G_k}
    → 渲染每个实例组到所有可见视角
    → 每视角跑开放词汇检测器(简单查询/指代表达)
    → IoU加权投票聚合 → VASD_k
    → 查询时取argmax或重复指代表达流程
```

## 实现

下面是根据论文思路编写的最小化示例代码，用于理解算法骨架，并非论文官方实现。摘要中没有公开代码仓库链接。

### 环境配置

```bash
pip install torch torchvision open3d numpy
# 实际工程中还需要 gsplat / diff-gaussian-rasterization 做高斯渲染
# 以及一个开放词汇检测器，如 Grounding DINO 的推理接口
```

### 1. 高斯实例嵌入的对比训练

```python
import torch
import torch.nn.functional as F

def instance_contrastive_loss(rendered_embed, sam_mask_ids):
    """
    rendered_embed: (H, W, D) 渲染出的实例嵌入图
    sam_mask_ids:   (H, W) 同一张图上 SAM 给出的 2D mask id（仅用于训练监督）
    """
    feats = rendered_embed.reshape(-1, rendered_embed.shape[-1])
    ids = sam_mask_ids.reshape(-1)
    feats = F.normalize(feats, dim=-1)

    # 随机采样像素对，避免 H*W 平方级计算
    idx = torch.randint(0, feats.shape[0], (4096,))
    a, b = feats[idx], feats[idx[torch.randperm(4096)]]
    same = (ids[idx] == ids[idx[torch.randperm(4096)]]).float()

    sim = (a * b).sum(-1)  # cosine similarity
    # 同实例拉近，不同实例推远（margin=0.2）
    loss = same * (1 - sim) + (1 - same) * F.relu(sim - 0.2)
    return loss.mean()
```

### 2. 3D 实例分组（聚类）

```python
from sklearn.cluster import DBSCAN

def group_gaussians(instance_embed):
    """
    instance_embed: (N, D) 每个高斯学到的实例嵌入
    返回每个高斯所属的 3D 实例 id，-1 表示噪声/未分组
    """
    embed_np = F.normalize(instance_embed, dim=-1).detach().cpu().numpy()
    clustering = DBSCAN(eps=0.15, min_samples=20, metric="cosine").fit(embed_np)
    return torch.from_numpy(clustering.labels_)
```

### 3. 多视角语义投票聚合（VASD，核心贡献）

```python
import numpy as np

def aggregate_vasd(instance_masks_per_view, detections_per_view, view_weights):
    """
    instance_masks_per_view: dict[view_id] -> {instance_id: bool_mask(H,W)}
    detections_per_view:     dict[view_id] -> list of (label, box_mask, score)
    view_weights:            dict[view_id] -> float，越正对/面积越大权重越高
    """
    votes = {}  # instance_id -> {label: accumulated_score}

    for view_id, inst_masks in instance_masks_per_view.items():
        dets = detections_per_view.get(view_id, [])
        w_v = view_weights[view_id]
        for inst_id, mask in inst_masks.items():
            votes.setdefault(inst_id, {})
            for label, box_mask, score in dets:
                inter = np.logical_and(mask, box_mask).sum()
                union = np.logical_or(mask, box_mask).sum() + 1e-6
                iou = inter / union
                if iou > 0:
                    votes[inst_id][label] = votes[inst_id].get(label, 0.0) + w_v * iou * score

    # 归一化为分布 VASD_k(label)
    vasd = {}
    for inst_id, label_scores in votes.items():
        total = sum(label_scores.values()) + 1e-6
        vasd[inst_id] = {l: s / total for l, s in label_scores.items()}
    return vasd
```

### 4. 查询接口

```python
def query_open_vocab(vasd, query_label):
    return [inst for inst, dist in vasd.items()
            if dist.get(query_label, 0) == max(dist.values(), default=0) and query_label in dist]

def query_referring(scene_renderer, detector, instance_groups, expression):
    """
    指代表达查询：直接把整句表达喂给检测器的 grounding 接口，
    在多视角下复用 aggregate_vasd 的投票逻辑，无需重新训练。
    """
    masks_per_view = scene_renderer.render_instance_masks(instance_groups)
    dets_per_view = {v: detector.ground(img, expression)
                      for v, img in scene_renderer.get_views()}
    vasd = aggregate_vasd(masks_per_view, dets_per_view, scene_renderer.view_weights)
    return max(vasd, key=lambda k: sum(vasd[k].values()))
```

### 3D 可视化

```python
import open3d as o3d

def visualize_instance(gaussian_centers, instance_ids, target_id):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gaussian_centers)
    colors = np.tile([0.7, 0.7, 0.7], (len(gaussian_centers), 1))
    colors[instance_ids == target_id] = [1.0, 0.2, 0.2]  # 高亮目标实例
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
```

实际效果是：选中实例后，对应的高斯点云在 3D 空间中以红色高亮显示，可以直接旋转视角检查分割边界是否贴合物体几何。

## 实验

### 数据集说明

- **LeRF-OVS**：LeRF 论文配套的开放词汇分割基准，场景含复杂物体和长尾类别，常用于检验 CLIP 蒸馏类方法。
- **ScanNet**：室内真实扫描数据集，带稠密语义标注，适合评估开放词汇分割在真实噪声数据下的鲁棒性。
- **Ref-LeRF**：论文为评估指代表达定位新构建/扩展的基准，标注形式是自然语言指代表达 + 对应 3D 实例，而非简单类别名。

这三个数据集的共同点是都需要多视角图像 + 相机位姿，获取门槛和普通 3DGS 重建一致（手机环绕拍摄 + COLMAP 位姿估计即可），不需要额外的语言标注（推理阶段用现成检测器，不需要为每个场景标注语言数据）。

### 定量评估

摘要中明确给出的硬指标是：在严格零样本设置下，指代表达定位任务相比已有方法取得 **16.7% 的 mIoU 提升**。下表结构供参考，具体绝对数值需以论文正文表格为准：

| 方法 | 开放词汇分割 mIoU | 指代定位 mIoU（零样本） | 备注 |
|-----|------|------|------|
| LangSplat | 中等 | 不支持/弱 | 依赖 CLIP 蒸馏 |
| Gaussian Grouping | 中等 | 不支持/弱 | 实例分组依赖 SAM |
| GaussDet（本文） | 持续提升 | +16.7%（相对最佳基线） | 检测器直接做指代定位 |

### 定性结果

论文报告的典型现象是：在物体密集、相互遮挡较多的场景里，CLIP 蒸馏类方法容易出现"语义渗透"——相邻物体的语言特征互相污染，边界模糊；而 GaussDet 因为投票是基于离散检测框的 IoU，实例边界更干净，但代价是依赖前置的 3D 聚类质量——如果聚类把两个物体并成一组，再好的投票也救不回来。

## 工程实践

### 实际部署考虑

- **实时性**：3DGS 渲染本身可以做到实时（数十到上百 FPS），但 GaussDet 的语义查询阶段需要对多个视角逐一跑一次 2D 检测器推理，这部分不是实时的，更适合"离线建库 + 在线查询缓存结果"的模式，而不是每帧实时跑检测器。
- **硬件需求**：3DGS 训练/渲染一张消费级 GPU（如 RTX 3090/4090，24GB 显存）足够；2D 检测器（Grounding DINO 等）额外占用显存，多视角批量推理建议单独排队执行，避免和高斯渲染抢显存。
- **内存占用**：每个高斯额外挂一个低维实例嵌入（如 16~32 维）相比挂高维 CLIP 特征（512/768 维）省内存得多，这是相对 LangSplat 类方法的一个工程优势。

### 数据采集建议

- 场景需要足够的视角覆盖度，尤其是目标物体的多个侧面都要被拍到——投票机制依赖多视角一致性，单视角覆盖不足会让 VASD 退化成"单视角结果"，丧失去噪能力。
- 避免强烈光照变化导致检测器在不同视角给出不一致的判断，这会直接稀释票数。

### 常见坑

1. **3D 聚类把多个物体并成一组** → 后续投票无法修正语义；建议适当调小 DBSCAN 的 `eps` 或换用基于密度自适应的聚类，并对聚类数量做合理性检查。
2. **检测器在小物体/远距离视角误检** → 用渲染掩码面积过滤掉过小、过于倾斜的视角贡献，避免低质量视角主导投票。
3. **指代表达涉及场景级关系（"离门最近的箱子"）** → 单帧 2D 检测器通常理解不了全局空间关系，这类查询效果会明显下降，需要额外的几何后处理而不能完全依赖检测器本身。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 静态室内/物体场景，多视角覆盖充分 | 动态场景、视角覆盖稀疏 |
| 需要支持自然语言指代表达查询 | 只需要固定类别的语义分割（专用分割模型更划算） |
| 物体边界清晰、遮挡适中 | 物体高度堆叠粘连，3D 聚类难以分离 |
| 离线建图 + 在线查询的应用模式 | 需要逐帧实时语义理解（如高速机器人避障） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| LangSplat | 像素级连续语言特征，查询灵活 | CLIP 语义粒度粗，无法做指代定位 | 简单开放词汇分割 |
| Gaussian Grouping | 实例分组与重建联合优化 | 依赖 SAM 分组质量，预设实例数敏感 | 实例数量已知或可控的场景 |
| GaussDet（本文） | 复用强大 2D 检测器能力，支持零样本指代定位，显存友好 | 推理阶段需多次调用检测器，依赖前置 3D 聚类质量 | 需要复杂语言查询的具身智能/AR 场景 |

## 我的观点

GaussDet 体现的是一个挺务实的工程思路：与其费力把一个能力（CLIP 的语义理解）蒸馏进 3D 表示里再损失掉一部分精度，不如在推理时直接复用更强、更新更快的 2D 基础模型，3D 重建只负责提供几何一致性和多视角聚合的"骨架"。这种"3D 提供结构，2D 模型提供语义"的分工，未来大概率会随着 2D 检测/分割基础模型继续变强而持续受益，不需要重新训练 3D 部分。

但它也没有摆脱开放词汇 3D 理解的根本难题：3D 实例分组质量始终是上限，投票聚合只能减噪、不能纠正系统性的分组错误；而涉及全局空间关系的复杂查询，单帧 2D 检测器的理解力依然有限。距离真正可靠的"自然语言操控 3D 场景"，还需要在 3D 几何推理和语言模型之间做更深的结合，而不只是多视角投票这一层去噪。