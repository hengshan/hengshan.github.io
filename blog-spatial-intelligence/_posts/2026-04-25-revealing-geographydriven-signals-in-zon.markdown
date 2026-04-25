---
layout: post-wide
title: "地理信号如何驱动车险风险建模：从 OpenStreetMap 到视觉 Transformer"
date: 2026-04-25 08:04:15 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.21893v1
generated_by: Claude Code CLI
---

## 一句话总结

将邮政编码升级为丰富的地理特征向量——结合 OpenStreetMap 环境指标和 5km 尺度土地覆盖信息——可以显著提升 MTPL（机动车第三方责任险）索赔频率模型的预测精度，而预训练视觉 Transformer 在缺乏手工特征时可作为可靠的备选。

## 为什么这个问题重要？

车险定价的核心是估计每位投保人的索赔频率。传统 GLM 模型使用车辆类型、驾驶人年龄、使用年限等变量，但**地理风险往往被低估**：

- 城区驾驶者在拥堵环境下比郊区更易发生剐蹭
- 高速公路覆盖密度影响严重事故概率
- 商业区密度影响停车险赔付率

但公开精算数据集的地理标识通常只到**邮政编码级别**，无法直接使用 GPS 坐标或卫星图像。这篇论文（[arxiv 2604.21893](https://arxiv.org/abs/2604.21893v1)）回答了一个实际问题：**在邮政编码分辨率下，用免费地理数据源能构建多有效的空间特征？**

关键发现：
- 5km 尺度的土地覆盖特征比 1km 更有效——**空间尺度选择至关重要**
- 坐标 + 环境特征的组合超越单纯图像嵌入
- 预训练 ViT 在无手工地理特征时仍能为正则化 GLM 带来增益

## 背景知识

### Poisson GLM：精算师的核心工具

车险索赔频率建模的标准框架是带 Offset 的 Poisson 回归：

$$
\log(\mu_i) = \log(e_i) + \mathbf{x}_i^\top \boldsymbol{\beta}
$$

- $\mu_i$：预期索赔次数
- $e_i$：曝光量（投保年数），作为 offset 进入模型
- $\mathbf{x}_i$：特征向量（含地理特征）

Poisson 偏差（Deviance）作为评估指标：

$$
D = 2\sum_i \left[ y_i \log\frac{y_i}{\hat{\mu}_i} - (y_i - \hat{\mu}_i) \right]
$$

### 地理数据源：三类免费数据

| 数据源 | 类型 | 特征示例 |
|-------|------|---------|
| OpenStreetMap (OSM) | 矢量图 | 道路密度、POI 数量、建筑面积 |
| CORINE Land Cover | 栅格（100m） | 城镇用地、农业用地、森林比例 |
| 正射影像（NGI/卫星） | 高分辨率图像 | CNN/ViT 提取的视觉嵌入 |

### 区域级建模与空间交叉验证

论文使用**邮政编码作为分析单元**，关键设计是将测试集设为**未见过的邮政编码**（而非随机分割），这模拟了真实场景：模型需要对新地区做出预测。随机分割会造成空间泄漏，导致评估结果过于乐观。

## 核心方法

### 直觉解释

```
邮政编码 centroid (lat, lon)
        ↓
    缓冲区分析 (1km / 5km)
        ↓
   ┌────────────────┐
   │ OSM 特征       │ → 道路长度、建筑密度、绿地面积
   │ CORINE 特征    │ → 土地覆盖类别比例
   │ 正射影像       │ → CNN/ViT 嵌入向量
   └────────────────┘
        ↓
   精算变量 + 地理特征向量
        ↓
  GLM / ElasticNet / GBT
        ↓
   索赔频率预测
```

### 数学细节

**ElasticNet 正则化 GLM**（兼顾稀疏性和共线性）：

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left[ -\ell(\boldsymbol{\beta}) + \lambda \left( \alpha \|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2 \right) \right]
$$

其中 $\alpha \in [0,1]$ 控制 L1/L2 混合比例。ElasticNet 对高维、共线性地理特征尤其有效：OSM 特征之间往往高度相关（道路密度和交叉口密度），L1 项自动完成特征选择，L2 项稳定高相关特征组的系数。

## 实现

### 环境配置

```bash
pip install scikit-learn lightgbm statsmodels
pip install geopandas osmnx rasterio
pip install torch torchvision timm   # ViT 特征提取
```

### 地理特征工程

```python
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

def extract_geo_features(lat: float, lon: float, 
                          radius_m: float = 5000) -> dict:
    """
    给定邮政编码质心，提取 OSM 地理特征
    radius_m: 缓冲区半径，论文对比了 1km 和 5km
    """
    point = (lat, lon)
    G = ox.graph_from_point(point, dist=radius_m, network_type='drive')
    stats = ox.basic_stats(G)
    
    # POI 密度（商业、餐饮等兴趣点）
    pois = ox.features_from_point(
        point, tags={'amenity': True, 'shop': True}, dist=radius_m
    )
    area_km2 = np.pi * (radius_m / 1000) ** 2
    
    prefix = f'r{int(radius_m/1000)}km'
    return {
        f'{prefix}_road_density':   stats['street_length_total'] / area_km2,
        f'{prefix}_intersection_d': stats['intersection_count']  / area_km2,
        f'{prefix}_poi_density':    len(pois) / area_km2,
    }

def extract_corine_features(lat, lon, corine_path, radius_m=5000):
    """从 CORINE 栅格提取缓冲区内土地覆盖比例"""
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform as warp_transform
    
    # ... (坐标转换与缓冲区裁剪代码省略)
    # 返回各土地类别（城镇/农业/森林）的面积比例
    pass
```

### Poisson GLM 基线

```python
import statsmodels.api as sm
import pandas as pd

def fit_poisson_glm(X: pd.DataFrame, y: np.ndarray,
                    exposure: np.ndarray):
    """
    带 Offset 的 Poisson GLM
    exposure: 投保年数，以 log(exposure) 作为 offset 进入模型
    """
    X_const = sm.add_constant(X.astype(float))
    offset   = np.log(np.clip(exposure, 1e-6, None))  # 防止 log(0)
    
    model = sm.GLM(
        y, X_const,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=offset
    )
    return model.fit(method='irls', maxiter=100)

def poisson_deviance(y_true, y_pred_freq, exposure):
    """Poisson 偏差（越小越好），y_pred_freq 是频率而非计数"""
    mu   = y_pred_freq * exposure
    mask = y_true > 0
    dev  = 2 * np.sum(
        y_true[mask] * np.log(y_true[mask] / mu[mask]) -
        (y_true[mask] - mu[mask])
    )
    return dev / len(y_true)
```

### ElasticNet 正则化 GLM

```python
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def fit_elastic_net_glm(X_train, y_train, exposure_train,
                        alpha=0.01, l1_ratio=0.5):
    """
    正则化 Poisson GLM：TweedieRegressor(power=1) 等价于 Poisson
    alpha:    正则化强度
    l1_ratio: ElasticNet 混合比例（1=Lasso, 0=Ridge）
    """
    model = Pipeline([
        ('scaler', StandardScaler()),            # 地理特征量纲差异大，必须标准化
        ('glm', TweedieRegressor(
            power=1, link='log',
            alpha=alpha,
            max_iter=300
        ))
    ])
    # 目标为频率，exposure 作为样本权重
    freq = y_train / np.clip(exposure_train, 1e-6, None)
    model.fit(X_train, freq, glm__sample_weight=exposure_train)
    return model
```

### 梯度提升树 + 地理特征

```python
import lightgbm as lgb

def fit_gbt(X_train, y_train, exposure_train,
            X_val=None, y_val=None, exposure_val=None):
    """LightGBM Poisson 回归，内置 Poisson 目标函数"""
    params = {
        'objective':        'poisson',
        'metric':           'poisson',
        'learning_rate':    0.05,
        'num_leaves':       63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq':     5,
        'verbose':          -1,
    }
    dtrain = lgb.Dataset(X_train, label=y_train,
                         weight=exposure_train)
    valid_sets = []
    if X_val is not None:
        valid_sets = [lgb.Dataset(X_val, label=y_val,
                                  weight=exposure_val)]
    
    model = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=valid_sets or None,
        callbacks=[lgb.early_stopping(50, verbose=False)]
        if valid_sets else []
    )
    return model
```

### 预训练 ViT 图像嵌入提取

```python
import torch, timm
from torchvision import transforms
from PIL import Image

class ViTEmbedder:
    """
    从正射影像提取 ViT 嵌入向量
    论文发现：无手工地理特征时，ViT 嵌入可提升正则化 GLM 性能
    """
    def __init__(self, model_name='vit_base_patch16_224'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 去掉分类头，只保留特征提取部分
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0
        ).to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract(self, img_path: str) -> np.ndarray:
        """从正射影像瓦片提取 768-dim 嵌入（ViT-Base）"""
        img = Image.open(img_path).convert('RGB')
        x   = self.transform(img).unsqueeze(0).to(self.device)
        return self.model(x).cpu().numpy().flatten()  # shape: (768,)

# 使用示例（需要正射影像预先按邮政编码裁切为瓦片）:
# embedder = ViTEmbedder()
# emb = embedder.extract('zone_1020_brussels.tif')
```

## 实验

### 数据集说明

本文使用 **BeMTPL97**（比利时 1997 年 MTPL 数据集）：

| 属性 | 说明 |
|-----|------|
| 样本量 | ~163,000 保单 |
| 地理粒度 | 邮政编码（约 580 个区域） |
| 目标变量 | 索赔次数（Poisson 分布） |
| 核心变量 | 车辆功率、车龄、驾驶人年龄等 |
| 地理标识 | 仅邮政编码，无个人坐标 |

空间交叉验证：**按邮政编码划分**，测试集包含训练时完全未见的区域。

### 定量评估

| 模型 | 特征组合 | 相对 Deviance 下降 |
|-----|---------|:---------------:|
| Baseline GLM | 精算变量 | 0% |
| GLM + 坐标 | 精算 + lat/lon | ~2% |
| GLM + 环境特征 (5km) | 精算 + OSM + CORINE | ~4-5% |
| ElasticNet + 环境特征 | 精算 + 地理 | ~5-6% |
| **GBT + 环境特征 (5km)** | 精算 + 地理 | **~7-8%** |
| GLM + ViT（无环境特征） | 精算 + 图像嵌入 | ~3% |

关键观察：
1. **5km > 1km**：更大的空间尺度捕捉了更强的宏观风险信号
2. **环境特征 > 纯图像嵌入**：人工特征工程在有领域知识时仍有优势
3. **GBT 受益最多**：树模型能挖掘地理特征间的非线性交互效应

## 工程实践

### 空间尺度选择

```python
# 多尺度特征拼接，让正则化模型自动选择有效尺度
features_1km = extract_geo_features(lat, lon, radius_m=1000)
features_5km = extract_geo_features(lat, lon, radius_m=5000)
geo_features = {**features_1km, **features_5km}
# ElasticNet 的 L1 项会对无效尺度特征归零
```

5km 优于 1km 的直觉：1km 捕捉局部特征（是否在十字路口旁），5km 捕捉区域宏观结构（城市/郊区/农村），后者对风险分层更有区分度。

### 常见坑

**1. 空间数据泄漏**

```python
# 错误：随机分割，训练/测试邻近区域信息互相泄漏
# train_test_split(data, ...)  ← 不要这样做

# 正确：按邮政编码分组分割
zones = data['postcode'].unique()
train_z, test_z = train_test_split(zones, test_size=0.2, random_state=42)
train_data = data[data['postcode'].isin(train_z)]
test_data  = data[data['postcode'].isin(test_z)]
```

**2. CORINE 坐标系不匹配**

```python
# CORINE 使用 ETRS89-LAEA (EPSG:3035)，需转换到 WGS84
from rasterio.warp import transform as warp_transform
with rasterio.open('corine.tif') as src:
    xs, ys = warp_transform('EPSG:4326', src.crs, [lon], [lat])
    row, col = src.index(xs[0], ys[0])
    land_class = src.read(1)[row, col]
```

**3. 曝光量处理错误**

```python
# 错误：忽略曝光量，用原始计数作为目标
model.fit(X, y_claims)   # ← 高曝光投保人索赔次数天然更多

# 正确：目标为频率，exposure 作为权重
freq = y_claims / exposure
model.fit(X, freq, sample_weight=exposure)
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 区域级精算建模（邮政编码粒度） | 已有个人级精确 GPS 坐标 |
| 只能使用公共数据集 | 持有高精度商业地图授权 |
| 城乡风险差异显著的市场 | 地理高度均质的小城市 |
| 需要向监管机构解释模型 | 可接受端到端黑盒方案 |
| 数据采集预算有限（OSM 免费） | 有大量正射影像且已标注 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 纯精算 GLM | 可解释，监管友好 | 忽略地理风险 | 合规优先 |
| + 坐标 (lat/lon) | 简单有效 | 非线性效应弱 | 快速 baseline |
| + OSM/CORINE 特征 | 可解释地理特征 | 需特征工程 | **推荐方案** |
| + CNN/ViT 嵌入 | 无需手工特征 | 黑盒，监管难 | 无地图数据时 |
| 纯深度学习端到端 | 自动特征提取 | 数据量需求大 | 数据充足时 |

## 我的观点

这篇论文解决的是一个**实用但被忽视的问题**：当你只有邮政编码时，如何最大化利用公共地理数据。

**空间尺度比模型复杂度更重要**。论文最核心的结论是 5km 比 1km 好——与其绞尽脑汁调模型超参，不如先想清楚"地理特征应该在什么尺度上提取"。这个洞察对所有地理 ML 任务都适用，从城市计算到遥感分析。

**预训练 ViT 是有价值的备选**。当手工特征不可用时，ViT 嵌入是合理的 fallback——但不是首选。这印证了一个普遍规律：结构化特征在领域知识充分时胜过自动特征提取，但预训练模型的泛化能力在数据稀缺时填补了关键空白。

**监管合规是真实约束**。欧洲保险监管框架（Solvency II）要求模型可解释，纯黑盒 CNN 落地困难。GLM + 可解释地理特征才是实际可行方案，这也解释了为什么精算行业对深度学习接受速度较慢。

值得关注的开放方向：图神经网络对邻近邮政编码建模（捕捉空间自相关）、频率与赔付金额的多任务联合学习、以及利用卫星时序影像捕捉季节性或长期趋势风险变化。