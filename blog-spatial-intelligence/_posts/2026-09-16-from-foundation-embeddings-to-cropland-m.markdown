---
layout: post-wide
title: "冻结的地理空间基础模型嵌入能不能做好耕地制图？"
date: 2026-09-16 12:02:47 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.17138v1
generated_by: Claude Code CLI
---

## 一句话总结

这篇论文用 AlphaEarth 提供的**冻结（不微调）年度卫星嵌入向量**，加一个轻量级分类器（逻辑回归/随机森林），在美国缅因州做二分类耕地制图，达到 93.7% 总体精度、90.8% 平衡精度，效果逼近专门微调的分割模型，且计算成本低得多。更有意思的是：它还找了两位人工解译员盲评 385 个点做"第三方裁判"，结果显示这套方法甚至比官方的 USDA CDL 参考图更准确（95.3% vs 91.7%，McNemar 检验 p=0.0161）。

## 为什么这个问题重要？

耕地/非耕地制图听起来朴素，但背后是农业普查、碳汇核算、粮食安全预警等一系列应用的基础输入。传统做法有两条路：

- **专家规则 + 时序光谱指数**（NDVI 物候曲线等）：可解释，但对新区域、新年份泛化差，需要大量领域知识调参。
- **端到端深度分割模型**（U-Net、Swin、TerraMind 之类）：精度高，但需要标注掩膜、微调训练、算力投入，换一个地区往往要重新训练。

**地理空间基础模型**（Geospatial Foundation Model）试图打破这个二选一：在海量卫星影像上预训练一个通用编码器，输出一个固定维度的嵌入向量，下游任务只需要在这个嵌入上训练一个简单分类器——不碰编码器本身。这篇论文验证的核心问题是：**这种"即插即用"的冻结嵌入，够不够格替代专门微调的分割模型？**

这个问题对遥感、农业遥感创业公司、政府制图部门都有直接意义：如果答案是"够格"，意味着一个没有深度学习团队的机构，也能用逻辑回归做出接近 SOTA 的耕地地图。

## 背景知识

### 什么是 AlphaEarth 嵌入

AlphaEarth Foundations（Google DeepMind）是一个在 Sentinel-1/2、Landsat 等多源卫星数据上预训练的编码器，输出**每个像元每年一个 64 维向量**，分辨率约 10 米。这个向量融合了光谱、纹理、时序物候等信息，可以理解成"这一小块地在这一年的语义指纹"。

关键设计是它**按年发布**，同一像元不同年份的嵌入可以直接比较，这为下面要讲的"时序可迁移性"实验提供了基础。

### 三种表示范式的对比

| 表示方式 | 特点 | 制图适用场景 |
|---------|------|------------|
| 原始波段（多光谱时序） | 信息完整但维度高、噪声大 | 需要专家特征工程 |
| 手工指数（NDVI/EVI 时序） | 可解释、轻量 | 物候特征明显的作物 |
| 基础模型嵌入（本文） | 冻结、通用、低维 | 下游任务标注少、算力有限 |
| 端到端微调分割网络 | 精度上限高 | 标注充足、算力充足 |

## 核心方法

### 直觉解释

把每个像元想象成 64 维空间里的一个点。如果基础模型训练得好，"耕地"和"非耕地"在这个空间里会自然分成两团，中间只需要一条（或者几条）简单的分界线就能区分——不需要复杂的非线性模型。论文用**最近类别质心**（Nearest Class Centroid，不拟合任何参数，只算两类的均值点，谁离得近就归哪类）作为下限基线，逻辑回归作为上限对照，结果两者差距很小，说明这个 64 维空间里的类别可分性本身就很好，复杂模型带来的增量有限。

### 数学细节

最近类别质心分类规则：

$$
\hat{y} = \arg\min_{c \in \{0,1\}} \| x - \mu_c \|_2, \quad \mu_c = \frac{1}{|S_c|}\sum_{x_i \in S_c} x_i
$$

平衡精度（类别不平衡时比总体精度更可靠）：

$$
\text{BA} = \frac{1}{2}\left(\text{Sensitivity} + \text{Specificity}\right) = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)
$$

配对分类器比较用的 McNemar 检验（判断两个模型在同一批样本上的差异是否显著，而不是简单比较总体精度数字）：

$$
\chi^2 = \frac{(|b-c|-1)^2}{b+c}
$$

其中 $b$、$c$ 分别是"仅模型 A 答对"和"仅模型 B 答对"的样本数。这也是论文能说出"95.3% vs 91.7% 显著不同（$p=0.0161$），但 95.3% vs 93.5% 不显著（$p=0.14$）"的统计依据。

### Pipeline 概览

```
卫星影像（Sentinel/Landsat）
      ↓ 预训练编码器（冻结，不参与本任务训练）
每像元 64 维年度嵌入
      ↓ 轻量分类器（Logistic Regression / RF / Nearest Centroid）
耕地 / 非耕地 二分类栅格图
      ↓ 与 USDA CDL 参考标签 + 人工盲评对比
精度评估（总体精度 / 平衡精度 / Kappa / McNemar）
```

## 实现

真实 AlphaEarth 嵌入需要通过 Google Earth Engine 获取，本文用模拟数据复现论文的**方法论骨架**——分类器对比、空间分组交叉验证、标签效率曲线、时序迁移测试——帮助理解每一步在验证什么。

### 环境配置

```bash
pip install numpy scikit-learn matplotlib statsmodels
```

### 核心代码

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

# 模拟 AlphaEarth 年度嵌入：每个像元 64 维向量（真实数据集就是 64-band 年度嵌入）
def simulate_embeddings(n_pixels=20000, n_patches=192, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    patch_id = rng.integers(0, n_patches, size=n_pixels)          # 模拟空间分块，避免像元级数据泄漏
    is_crop = rng.integers(0, 2, size=n_pixels)
    centroid_crop = rng.normal(0, 1, dim)
    centroid_noncrop = rng.normal(0, 1, dim)
    noise = rng.normal(0, 1.5, (n_pixels, dim))
    X = np.where(is_crop[:, None] == 1, centroid_crop, centroid_noncrop) + noise
    return X, is_crop, patch_id

class NearestCentroidClassifier:
    """不拟合可训练参数，只计算类别质心，用作下限基线"""
    def fit(self, X, y):
        self.centroids_ = {c: X[y == c].mean(axis=0) for c in np.unique(y)}
        return self
    def predict(self, X):
        dists = np.stack([np.linalg.norm(X - c, axis=1) for c in self.centroids_.values()], axis=1)
        labels = np.array(list(self.centroids_.keys()))
        return labels[dists.argmin(axis=1)]

X, y, patch_id = simulate_embeddings()
gkf = GroupKFold(n_splits=5)                                       # 按地块分组，保证同一地块不同时出现在训练/测试集
train_idx, test_idx = next(gkf.split(X, y, groups=patch_id))
X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

models = {
    "NearestCentroid": NearestCentroidClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0),
}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"{name}: balanced_accuracy={balanced_accuracy_score(y_test, pred):.3f}")
```

这段代码复现了论文最核心的对照实验：三种复杂度递增的分类器，看精度差距有多大。用 `GroupKFold` 按 `patch_id` 分组，是为了避免像元级数据泄漏——这一点论文特别强调，也是遥感制图里最容易踩的坑之一（下文详述）。

### 标签效率曲线

```python
import matplotlib.pyplot as plt

def label_efficiency_curve(X_train, y_train, X_test, y_test, sizes):
    scores = []
    rng = np.random.default_rng(1)
    for n in sizes:
        idx = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)
        clf = LogisticRegression(max_iter=1000).fit(X_train[idx], y_train[idx])
        scores.append(balanced_accuracy_score(y_test, clf.predict(X_test)))
    return scores

sizes = [200, 500, 1000, 3000, 6000, len(X_train)]
scores = label_efficiency_curve(X_train, y_train, X_test, y_test, sizes)

plt.plot(sizes, scores, marker="o")
plt.xscale("log")
plt.xlabel("训练像元数（对数尺度）")
plt.ylabel("Balanced Accuracy")
plt.title("标签效率曲线：小样本也能接近全量表现")
plt.savefig("label_efficiency.png", dpi=150)
```

论文原文用 6 万像元达到了 860 万像元全量的 98.7% 水平（差 1.3 个百分点）。用这段代码在自己的数据上画一条类似曲线，可以快速判断"到底还需要标多少样本"——这是实际项目里最直接有用的产出。

### 时序迁移测试

```python
def simulate_temporal_drift(X, drift_std=0.3, seed=2):
    rng = np.random.default_rng(seed)
    drift = rng.normal(0, drift_std, X.shape[1])                   # 传感器/大气校正带来的年际系统性偏移
    return X + drift + rng.normal(0, 0.5, X.shape)

clf_2023 = LogisticRegression(max_iter=1000).fit(X_train, y_train)

for year_offset in [1, 3, 5]:
    X_future = simulate_temporal_drift(X_test, drift_std=0.1 * year_offset)
    ba = balanced_accuracy_score(y_test, clf_2023.predict(X_future))
    print(f"训练年 -> +{year_offset}年迁移: balanced_accuracy={ba:.3f}")
```

这段用人工加噪模拟"年际漂移"，对应论文里"2018-2023 年跨年迁移仍保持高精度"的实验。真实场景中漂移来自传感器更替、大气校正版本变化等系统性因素，不是纯随机噪声，因此这里只是方法论示意，不能替代真实的多年数据验证。

### 可视化

```python
def plot_prediction_map(y_true, y_pred, patch_shape=(50, 50)):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(y_true[:patch_shape[0]*patch_shape[1]].reshape(patch_shape), cmap="YlGn")
    axes[0].set_title("参考标签（如 USDA CDL）")
    axes[1].imshow(y_pred[:patch_shape[0]*patch_shape[1]].reshape(patch_shape), cmap="YlGn")
    axes[1].set_title("AlphaEarth + LR 预测")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig("cropland_map_compare.png", dpi=150)
```

在真实项目里，这种"参考图 vs 预测图"的并排栅格对比是判断误差**空间分布**（是否集中在地块边界、林地边缘等特定区域）的最快方式，比单一精度数字有信息量得多。

## 实验

### 数据集说明

- **区域**：美国缅因州，192 个空间上互不重叠的 patch，共约 860 万个像元。
- **训练标签来源**：USDA Cropland Data Layer（CDL），30 米分辨率的年度土地覆盖产品——本身是模型输出，存在标注噪声，这是本研究一个重要的局限。
- **独立验证**：在一个连续的 2023 年区块中随机抽 385 个点，由两位人工解译员盲评并取得共识作为"更接近真值"的参考。

### 定量评估

| 方法 | 总体精度 | 平衡精度 | Kappa | 备注 |
|-----|---------|---------|-------|------|
| Nearest Class Centroid | — | 90.2% | — | 零训练参数 |
| Logistic Regression（冻结嵌入） | 93.7% | 90.8% | — | 主推荐配置 |
| Gradient Boosted Ensemble | ~93.4-94.0% | 与 LR 差 0.3pp 以内 | — | 复杂度换来的增益有限 |
| AlphaEarth + Random Forest（对人工共识验证） | 95.3% | — | 0.82 | 385 个盲评点 |
| USDA CDL（对人工共识验证） | 91.7% | — | 0.72 | 官方参考产品本身 |
| TerraMind（微调分割模型，对人工共识验证） | 93.5% | — | — | 与冻结嵌入方案无显著差异（p=0.14） |

### 定性结果

论文没有提供逐像元的详细可视化对比图，但从 Kappa 差异（0.82 vs 0.72）可以推断：AlphaEarth 方案在地块边界、破碎小地块这类容易被 30 米分辨率参考标签"抹平"的区域，误判更少。用上面的 `plot_prediction_map` 函数在自己项目的真实数据上跑一遍，重点看边界区域和林地/耕地过渡带的误差，会比单纯看精度数字更有诊断价值。

## 工程实践（重要！）

### 实际部署考虑

- **计算成本**：冻结嵌入 + 逻辑回归的训练在 CPU 上几分钟内完成，这是它相对微调分割模型最大的工程优势。但论文自己也承认，这不是一次"受控的算力对比实验"——TerraMind 微调需要 GPU 训练时间，两者没有在同一基准下量化对比，实际项目里这笔账要自己算。
- **推理规模**：一个州级区域 860 万像元用简单分类器推理是秒级的；如果扩展到全国级别，瓶颈会转移到嵌入本身的存储和读取（64 维 × 每年 × 全美像元，数据量相当可观）。

### 数据采集建议

- 用**空间分组交叉验证**（如 `GroupKFold` 按地块/patch 分组），而不是随机像元级切分，否则同一地块的像元同时出现在训练和测试集，精度会被严重高估。
- 论文强调"6 万像元 vs 860 万像元"的效率结论是**像元采样效率**，不是**独立标注点效率**——因为像元存在空间自相关，60000 个像元可能只对应几十到几百个真正独立的地理位置。规划标注预算时不要把这两者混淆。

### 常见坑

1. **把参考标签当真值**：CDL 本身是模型产物，含系统性噪声。用它训练+用它评估，会低估真实误差，也会让"进步"部分只是在拟合参考标签的偏差模式。缓解方法是像本文一样引入独立人工验证集。
2. **精度差异不做显著性检验**：两个 93% vs 95% 的数字，不代表方法真的有优劣区别。用 McNemar 检验（下面 10 行代码）能避免"看数字拍脑袋"的结论：

```python
from statsmodels.stats.contingency_tables import mcnemar

# b, c 分别是"仅模型 A 答对"和"仅模型 B 答对"的人工验证点数
table = [[0, 14], [3, 368]]
result = mcnemar(table, exact=True)
print(f"McNemar p-value = {result.pvalue:.4f}")
```

3. **单一区块验证外推到全区域**：385 个点集中在一个连续区块，论文明确说这不能证明"CDL 在全州范围内都被冻结嵌入方案纠正"，只能说这个局部结果符合"部分平滑了 CDL 标签噪声"的假设。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 二分类或少类别的区域级制图任务 | 需要精细类别（如具体作物种类）的细粒度分类 |
| 标注预算有限、没有深度学习工程能力 | 需要亚像元级别的边界精度（如地块面积精确测量） |
| 需要跨年快速复用同一套嵌入 | 研究区域超出基础模型预训练数据覆盖范围 |
| 作为快速原型/基线，验证问题是否可行 | 追求 SOTA 上限、有充足标注和算力做端到端微调 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| NDVI 时序 + 规则/决策树 | 可解释、无需基础模型 | 泛化能力弱，需专家调参 | 单一作物类型、物候特征明显 |
| 冻结基础模型嵌入 + 轻量分类器（本文） | 低算力、易复用、精度接近上限 | 依赖预训练质量，上限受编码器限制 | 标注少、快速原型、区域级二分类 |
| 微调端到端分割模型（如 TerraMind） | 精度上限高，支持细粒度类别 | 训练成本高，标注需求大 | 标注充足、算力充足、需要精细分割 |
| 官方参考产品（CDL 等） | 覆盖广、免费可得 | 30 米分辨率、含系统性噪声 | 快速获取粗略标签，不适合当真值 |

## 我的观点

这篇工作最有价值的地方不是精度数字本身，而是**方法论上的诚实**：作者反复强调"像元采样效率不等于标注效率""局部验证不能外推到全州""与微调模型的对比不是受控的算力实验"。这种自我设限的写法在基础模型应用类论文里并不常见，值得学习。

趋势上，"冻结嵌入 + 简单分类器"正在把地理空间深度学习的门槛拉到接近传统机器学习的水平——这对没有算法团队的农业部门、环保组织是实打实的好消息。但离真正的规模化部署还有几个开放问题没解决：基础模型本身每年重新训练/重新发布时，历史年份的嵌入是否保持数值一致性（否则时序可迁移性会被悄悄破坏）；以及单一州、单一年份区块的人工验证，能不能支撑起跨生物气候带的信心，都还需要更大规模的独立验证来回答。