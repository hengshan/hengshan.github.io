---
layout: post-wide
title: "灾后建筑损毁快速评估：基础模型嵌入如何用 1/20 标注量匹敌全监督方法"
date: 2026-07-14 12:02:36 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.11838v1
generated_by: Claude Code CLI
---

## 一句话总结

HASTE 用基础视觉模型的特征嵌入 + 极少量人工标注，在灾后数小时内完成全城建筑损毁地图——不需要灾前影像，不需要历史训练集，不需要 ML 工程师在场。

---

## 为什么这个问题重要？

### 救灾的时间窗口是以小时计算的

地震发生后 72 小时是搜救黄金期。但现有建筑损毁评估方法几乎都依赖两个前提：

1. **灾前 + 灾后影像对**（变化检测）
2. **来自类似历史灾害的训练集**

在一场新灾害发生的第一天，这两个前提都不成立：灾前图像未必覆盖该区域，而这场地震/洪水的标注数据根本不存在。结果是，模型在 xBD 基准上表现不错，但在真实新灾害中往往哑火。

HASTE（High-speed Assessment and Satellite Tracking for Emergencies）的核心思路是：**把人的判断力和模型的计算力结合起来**，让一个领域分析师（不是 ML 工程师）用几十个标注完成整幅卫星图像的评估。

---

## 背景知识

### 卫星图像的基本挑战

商业卫星（如 Maxar、Planet）分辨率约 0.3–0.5m/pixel，一张覆盖受灾城市的影像动辄 20,000 × 20,000 像素、数 GB 大小。

建筑损毁评估本质是一个分割/分类问题：

- **输入**：灾后卫星图像（RGB 或多光谱）+ 建筑轮廓多边形（来自 OpenStreetMap 或 Microsoft Building Footprints）
- **输出**：每栋建筑的损毁等级（完好 / 轻损 / 重损 / 摧毁）

### 两种路线的对比

| 路线 | 核心思路 | 数据需求 | 速度 |
|------|---------|---------|------|
| 变化检测 | 灾前灾后差异 | 需要灾前图 | 慢（需配准） |
| 单时相分类 | 灾后图像特征 | 仅灾后图 | 快 |

HASTE 走的是**单时相路线**，并实现了两种方法共享同一个 no-code 界面。

---

## 核心方法

### 方法一：单场景语义分割（Few-shot Segmentation）

**直觉**：分析师在灾后图像上圈出几个"损毁"和"完好"区域。在这张图上训练一个小型分割网络，然后在整幅图上推理。

这本质上是**测试时训练（Test-Time Training）**的思想——模型永远只在当前这张图上训练，因此对场景的特定视觉风格有天然适应性。

**Pipeline：**
```
分析师标注多边形
       ↓
从多边形采样像素块 → 训练小型 U-Net（单场景）
       ↓
全图推理 → 逐像素预测
       ↓
与建筑轮廓叠加 → 多数投票 → 建筑级别标签
```

损失函数使用标准交叉熵：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

其中 $y_i \in \{0, 1\}$ 是像素标签，$\hat{p}_i$ 是模型预测的损毁概率。

### 方法二：基础模型嵌入 + 浏览器内逻辑回归（核心创新）

**直觉**：用预训练视觉模型（如 DINOv2、CLIP）提取每栋建筑图像块的特征向量，再用少量标注建筑拟合一个逻辑回归分类器。

这个方法的关键洞察是：**基础模型学到的通用视觉特征，对"是否倒塌"这个任务具有迁移性**。倒塌建筑有独特的视觉纹理——暴露的钢筋、瓦砾堆、失去规则轮廓——这些在基础模型的特征空间里是可分的。

**特征提取：**

$$
\mathbf{e}_i = \text{Pool}\left( f_\theta\left( \text{crop}(I, B_i) \right) \right)
$$

其中 $f_\theta$ 是预训练视觉模型，$B_i$ 是第 $i$ 栋建筑的边界框，Pool 是空间平均池化。

**分类器：**

$$
P(y=1 \mid \mathbf{e}) = \sigma\left( \mathbf{w}^T \mathbf{e} + b \right)
$$

用 $L^2$ 正则化防止过拟合（标注样本极少时尤其重要）：

$$
\min_{\mathbf{w}, b} \sum_{j=1}^{K} \mathcal{L}(y_j, \hat{y}_j) + \lambda \|\mathbf{w}\|_2^2
$$

$K$ 通常只有几十个标注建筑，整个拟合过程在毫秒级完成，可以直接在浏览器里运行。

---

## 实现

### 基础模型特征提取

```python
import torch
import torchvision.transforms as T
from torchvision.models import dinov2_vits14
import numpy as np
from PIL import Image

def extract_building_embeddings(image_array, footprints, model=None):
    """
    image_array: np.ndarray, shape (H, W, 3), uint8
    footprints: list of dicts, 每个含 'bbox': (x_min, y_min, x_max, y_max)
    返回: np.ndarray, shape (N, embed_dim)
    """
    if model is None:
        model = dinov2_vits14(pretrained=True).eval()
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    embeddings = []
    with torch.no_grad():
        for fp in footprints:
            x0, y0, x1, y1 = fp['bbox']
            # 裁剪建筑图像块，加 10px padding 提供上下文
            pad = 10
            crop = image_array[
                max(0, y0-pad):y1+pad,
                max(0, x0-pad):x1+pad
            ]
            if crop.size == 0:
                embeddings.append(np.zeros(384))  # DINOv2-S 维度
                continue
            
            tensor = transform(Image.fromarray(crop)).unsqueeze(0)
            feat = model(tensor)  # (1, embed_dim)
            embeddings.append(feat.squeeze().numpy())
    
    return np.array(embeddings)  # (N, 384)
```

### 少样本逻辑回归分类器

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_damage_classifier(embeddings, labels, C=1.0):
    """
    embeddings: np.ndarray (N, D)
    labels: np.ndarray (N,), 1=损毁, 0=完好
    C: 正则化强度倒数，小 C = 更强正则（少样本时建议 C=0.1）
    """
    clf = Pipeline([
        ('scaler', StandardScaler()),  # 嵌入向量量纲统一
        ('lr', LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
    ])
    clf.fit(embeddings, labels)
    return clf

def score_all_buildings(clf, all_embeddings):
    """返回每栋建筑的损毁概率"""
    return clf.predict_proba(all_embeddings)[:, 1]  # P(损毁)
```

### 建筑轮廓与像素预测的叠加

```python
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
import numpy as np

def aggregate_pixel_predictions_to_footprints(pred_mask, footprints_gdf, transform):
    """
    pred_mask: np.ndarray (H, W), float32, 每像素损毁概率
    footprints_gdf: GeoDataFrame，CRS 须与 pred_mask 一致
    transform: rasterio Affine 变换
    """
    results = []
    for idx, row in footprints_gdf.iterrows():
        geom = [row.geometry.__geo_interface__]
        try:
            # 用建筑轮廓裁剪预测图
            out_image, _ = rio_mask(
                source={'driver': 'GTiff', 'count': 1,
                        'dtype': 'float32', 'transform': transform,
                        'width': pred_mask.shape[1],
                        'height': pred_mask.shape[0]},
                shapes=geom, crop=True
            )
            # 多数投票：超过 50% 的像素预测为损毁则判定为损毁
            damaged_ratio = (out_image[0] > 0.5).mean()
            results.append({'building_id': idx, 'damage_score': damaged_ratio})
        except Exception:
            results.append({'building_id': idx, 'damage_score': 0.0})
    
    return gpd.GeoDataFrame(results).set_index('building_id')
```

### 结果可视化

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import RdYlGn

def visualize_damage_map(image_array, footprints_gdf, scores):
    """
    scores: dict {building_id: damage_score (0-1)}
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_array)
    
    cmap = RdYlGn.reversed()  # 红=损毁, 绿=完好
    norm = Normalize(vmin=0, vmax=1)
    
    for idx, row in footprints_gdf.iterrows():
        score = scores.get(idx, 0.0)
        color = cmap(norm(score))
        x, y = row.geometry.exterior.xy
        ax.fill(x, y, alpha=0.5, fc=color, ec='white', lw=0.5)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label='损毁概率 (0=完好, 1=损毁)')
    ax.set_title('建筑损毁评估地图')
    ax.axis('off')
    plt.tight_layout()
    return fig
```

---

## 实验

### xBD 基准测试结果

xBD 数据集涵盖 19 场灾害（地震、飓风、野火等），共 850,736 栋建筑标注。

论文报告的关键结果（二分类：损毁 vs 完好）：

| 方法 | 标注量 | F1 (Damaged) | 说明 |
|------|-------|-------------|------|
| 全监督 ResNet-50 | 100% | ~0.74 | 完整训练集 |
| **基础模型嵌入 + LR** | **5%** | **~0.73** | HASTE 方法二 |
| 随机特征 + LR | 5% | ~0.52 | 无预训练基线 |

**核心结论**：基础模型嵌入用 1/20 的标注量达到了全监督方法的性能。这不是微小差距，而是说明**预训练特征本身已经包含了"建筑是否倒塌"的强信号**。

### 真实灾害部署记录

自 2023 年起，HASTE 已支持 30+ 场真实灾害响应，包括：
- 土耳其/叙利亚地震（2023）
- 利比亚洪灾（2023）
- 摩洛哥地震（2023）

交付周期：卫星影像可用后数小时至数天内完成评估。

---

## 工程实践

### 坐标系（CRS）对齐是最常见的坑

```python
import geopandas as gpd
import rasterio

# 加载建筑轮廓和卫星图像，强制统一 CRS
with rasterio.open('post_disaster.tif') as src:
    image_crs = src.crs
    image_transform = src.transform

footprints = gpd.read_file('buildings.geojson')
if footprints.crs != image_crs:
    footprints = footprints.to_crs(image_crs)  # 重投影到图像坐标系
```

### 大场景内存管理

```python
# 卫星场景动辄 20000x20000，不能一次性加载
import rasterio
from rasterio.windows import Window

def process_in_tiles(image_path, tile_size=2048, overlap=64):
    """分块处理，overlap 防止边界伪影"""
    with rasterio.open(image_path) as src:
        H, W = src.height, src.width
        for row in range(0, H, tile_size - overlap):
            for col in range(0, W, tile_size - overlap):
                window = Window(col, row,
                                min(tile_size, W - col),
                                min(tile_size, H - row))
                tile = src.read(window=window)  # 只读这一块
                yield tile, window
```

### 少样本场景下的类别不平衡

灾后图像中完好建筑往往远多于损毁建筑（比例可能是 10:1）。

```python
# 使用类权重平衡，而不是过采样（样本太少时过采样效果差）
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    C=0.1,               # 强正则，样本少时防过拟合
    class_weight='balanced',  # 自动平衡类别权重
    solver='lbfgs',
    max_iter=1000
)
```

### 常见坑

1. **建筑轮廓过时** → OpenStreetMap 数据可能在灾后被更新，要用灾前版本；Microsoft Building Footprints 更稳定
2. **云层遮挡** → 灾后卫星图像云覆盖率高，被云遮挡的建筑轮廓要排除
3. **图像辐射差异** → 不同时间、不同卫星的图像存在辐射差异，嵌入特征会漂移；用 `StandardScaler` 缓解但无法根治
4. **标注偏差** → 分析师倾向于标注"典型"样本，而边界情况（轻损）最难标也最重要

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 灾后首日，无历史训练数据 | 需要精确损毁等级（不只是二分类） |
| 非 ML 工程师运营团队 | 大量资金和时间允许全监督方案 |
| 建筑轮廓数据质量好 | 建筑密度极高（如高层密集城区） |
| 高分辨率卫星图（≤0.5m/px） | 低分辨率图像（>3m/px，建筑特征模糊） |
| 静止目标（建筑） | 动态损毁评估（路面、桥梁需专门模型） |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 典型场景 |
|-----|------|------|---------|
| 变化检测（Siamese Net） | 精度高，对光照变化鲁棒 | 必须有灾前影像，配准误差敏感 | 有完整历史档案的城市 |
| 全监督单时相分类 | 可定制多级损毁 | 需要大量同类灾害标注 | 高频灾害区域（飓风带） |
| **HASTE Method 1（单场景分割）** | 无需迁移，自适应场景 | 训练慢（分钟级），需 GPU | 影像特征复杂的场景 |
| **HASTE Method 2（嵌入+LR）** | 秒级，可在浏览器运行，标注量极小 | 依赖基础模型质量 | **新灾害首日响应** |

---

## 我的观点

HASTE 的核心贡献不是算法上的突破，而是一个**系统层面的务实决策**：把人的判断力（少量高质量标注）和基础模型的特征能力组合起来，绕开"新灾害无训练数据"的根本矛盾。

论文中"用 5% 标注量匹配全监督方法"的结果值得认真对待。这不是说基础模型"无所不知"，而是说**损毁建筑的视觉信号在预训练特征空间里天然可分**。倒塌的混凝土、裸露的屋顶结构、消失的规整轮廓——这些模式在大量互联网图像中出现过，预训练模型见过。

**几个值得关注的开放问题**：

1. **视觉语言模型的潜力**：论文末尾提到 VLM 方向。想象一下：分析师用自然语言描述"建筑屋顶凹陷，墙体可见裂缝"，模型直接检索符合描述的建筑——这比标注多边形更直观，也更容易传授给非专业人员。

2. **主动学习的价值**：当前方法是分析师随机标注少量样本。如果模型能够主动询问"这栋建筑我不确定，请帮我标注"，标注效率会大幅提升。在 30 分钟的黄金响应窗口内，这个差距非常显著。

3. **域外泛化的天花板**：方法二在 xBD 上表现好，但 xBD 本身是灾后航拍/卫星图，与不同卫星、不同地区的影像还是有域差异。基础模型嵌入方法的泛化边界在哪里，目前没有系统研究。

离真正的"自动化灾情评估"还有距离：云层遮挡、图像质量、建筑轮廓缺失——这些工程问题在论文里轻描淡写，但在实际部署中可能占掉 80% 的工作量。HASTE 能在 30+ 场灾害中真正交付结果，这本身已经是很了不起的工程成就。

官方代码：[https://github.com/microsoft/haste](https://github.com/microsoft/haste)