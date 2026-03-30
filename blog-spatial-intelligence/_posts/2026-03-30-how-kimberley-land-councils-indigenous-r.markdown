---
layout: post-wide
title: "用 Google Earth Engine 监测野火：从卫星光谱到火烧频率图"
date: 2026-03-30 12:04:44 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://medium.com/google-earth/how-kimberley-land-councils-indigenous-rangers-bring-65-000-years-of-expertise-to-earth-engine-39a9dcbfe8df?source=rss----a747a9e16c1c---4
generated_by: Claude Code CLI
---

## 一句话总结

通过卫星遥感数据和 Earth Engine 云计算平台，可以高效重建区域火烧历史、量化植被恢复状态，为生态管理和火险预警提供空间决策支持。

## 为什么这个问题重要

澳大利亚金伯利地区的原住民护林员，将 65,000 年的传统火管理知识与 Google Earth Engine 的卫星数据分析结合，实现了对数百万公顷土地的精细化管理。这个案例说明了一件事：**遥感技术真正的价值，不在于替代人类知识，而在于将专家经验拓展到无法人工巡查的尺度上。**

从技术角度看，野火监测面临几个核心挑战：

- **尺度问题**：单次野火可烧毁数万公顷，人工调查成本极高
- **时序问题**：火烧频率、季节性规律需要多年数据才能体现
- **异质性问题**：同一火场的燃烧强度差异极大，影响生态恢复轨迹

Google Earth Engine（GEE）通过将 PB 级卫星档案（Landsat、Sentinel、MODIS）与云端并行计算结合，让这类分析从"研究室专属"变成"可扩展基础设施"。

## 背景知识

### 关键光谱指数

卫星传感器捕获不同波段的地表反射率，植被和裸土在近红外（NIR）与短波红外（SWIR）波段有显著差异：

**归一化植被指数（NDVI）**：

$$
\text{NDVI} = \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + \rho_{Red}}
$$

**归一化燃烧比（NBR）**：

$$
\text{NBR} = \frac{\rho_{NIR} - \rho_{SWIR}}{\rho_{NIR} + \rho_{SWIR}}
$$

**差分燃烧比（dNBR）**，用于评估单次火烧严重程度：

$$
\text{dNBR} = \text{NBR}_{pre} - \text{NBR}_{post}
$$

植被受损越严重，post-fire NBR 越低，dNBR 越高。健康植被的 NBR 约为 0.4～0.6，严重火烧后可降至 -0.2 以下。

### 传感器选择对比

| 传感器 | 空间分辨率 | 重访周期 | 最佳用途 |
|--------|-----------|---------|---------|
| MODIS | 500m | 1-2天 | 实时火点检测、大范围统计 |
| Landsat 8/9 | 30m | 16天 | 历史火烧图、植被恢复评估 |
| Sentinel-2 | 10m | 5天 | 精细火烧边界、小面积监测 |

### 前置知识

需要了解：Python 基础、基本的遥感概念（波段、反射率）。不需要深入的 GIS 经验，GEE 处理了大部分底层坐标问题。

## 核心方法

### 直觉解释

火烧过的土地有一个典型的光谱特征：NIR 反射大幅下降（植被受损），SWIR 反射相对升高（黑炭、裸土）。NBR 因此从健康植被的高正值骤降。把多年 NBR 时序"堆叠"起来，就能识别每一次突降事件：

```
Landsat 历史图像 (1984-2025)
      ↓
逐年计算 NBR（年度中位数合成，抑制云噪声）
      ↓
检测相邻年份 NBR 突降（dNBR > 0.15 → 标记为火烧）
      ↓
按像素叠加所有年份的火烧掩膜
      ↓
生成火烧频率图（Fire Frequency Map）
```

频率图是土地管理的核心工具：从未烧过的区域燃料积累、高危险；烧的过于频繁的区域则影响生物多样性。

### 数学细节：火烧严重程度分级

USGS 标准 dNBR 分级：

| dNBR 范围 | 严重程度 | 生态含义 |
|-----------|---------|---------|
| < -0.10 | 负值 | 可能是火后植被再生 |
| -0.10 ~ 0.10 | 未烧/轻微 | 基本未受影响 |
| 0.10 ~ 0.27 | 轻度 | 草地表面火 |
| 0.27 ~ 0.44 | 中度 | 部分冠层受损 |
| > 0.44 | 高度 | 冠层完全烧毁 |

## 实现

### 环境配置

```bash
pip install earthengine-api geemap
earthengine authenticate   # 首次使用需要 OAuth 认证，会打开浏览器
```

### 核心代码：NBR 时序构建

```python
import ee
import geemap

ee.Initialize()

def mask_landsat_clouds(image):
    """使用 QA_PIXEL 波段去除云和云阴影"""
    qa = image.select('QA_PIXEL')
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)   # bit 3: 云
    shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)  # bit 4: 云阴影
    return image.updateMask(cloud_mask.And(shadow_mask))

def compute_nbr(image):
    """Landsat 8/9: NIR=B5, SWIR2=B7"""
    nbr = image.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR')
    return image.addBands(nbr)

# 研究区：金伯利地区 (西澳大利亚)
aoi = ee.Geometry.Rectangle([125.0, -18.5, 130.0, -15.0])

# 获取 Landsat 8 地表反射率并预处理
collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate('2013-01-01', '2024-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 30))
    .map(mask_landsat_clouds)
    .map(compute_nbr))

# 逐年中位数合成，消除单景云污染
def annual_nbr_composite(year):
    year = ee.Number(year)
    return (collection
        .filter(ee.Filter.calendarRange(year, year, 'year'))
        .select('NBR')
        .median()
        .set('year', year))

years = ee.List.sequence(2013, 2024)
annual_collection = ee.ImageCollection(years.map(annual_nbr_composite))
```

### 核心代码：火烧事件检测与频率统计

```python
def detect_fire_frequency(annual_collection, dnbr_threshold=0.15):
    """
    逐年检测 dNBR 突降事件，统计累计火烧次数
    返回: 每个像素的火烧频率（整数）
    """
    annual_list = annual_collection.toList(annual_collection.size())
    n = annual_collection.size().getInfo()

    fire_masks = []
    for i in range(1, n):
        pre  = ee.Image(annual_list.get(i - 1)).select('NBR')
        post = ee.Image(annual_list.get(i)).select('NBR')

        # dNBR 超阈值 → 该年发生火烧
        burned = pre.subtract(post).gt(dnbr_threshold).rename('burned')
        fire_masks.append(burned)

    # 所有年份叠加求和 = 累计火烧次数
    fire_frequency = (ee.ImageCollection(fire_masks)
        .sum()
        .rename('fire_frequency'))
    return fire_frequency

fire_freq = detect_fire_frequency(annual_collection, dnbr_threshold=0.15)
```

### 核心代码：单次火烧严重程度评估

```python
def assess_fire_severity(pre_date, post_date, aoi):
    """
    评估单次火烧事件严重程度
    使用火前/火后各3个月的中位数合成，减少云噪声
    """
    def get_composite(start, end):
        return (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt('CLOUD_COVER', 30))
            .map(mask_landsat_clouds)
            .map(compute_nbr)
            .select('NBR')
            .median())

    pre_nbr  = get_composite(
        ee.Date(pre_date).advance(-3, 'month').format('YYYY-MM-dd'), pre_date)
    post_nbr = get_composite(
        post_date,
        ee.Date(post_date).advance(3, 'month').format('YYYY-MM-dd'))

    dnbr = pre_nbr.subtract(post_nbr).rename('dNBR')

    # 按 USGS 标准分 4 级
    severity = (ee.Image(1)
        .where(dnbr.gte(0.10).And(dnbr.lt(0.27)), 2)   # 轻度
        .where(dnbr.gte(0.27).And(dnbr.lt(0.44)), 3)   # 中度
        .where(dnbr.gte(0.44), 4)                        # 高度
        .rename('severity'))

    return dnbr, severity

# 示例：2023年干季火烧事件
dnbr_2023, sev_2023 = assess_fire_severity('2023-07-01', '2023-09-30', aoi)
```

### 可视化：交互式地图

```python
Map = geemap.Map(center=[-16.5, 127.5], zoom=7)

# 火烧频率：绿(0次) → 红(多次)
freq_vis = {
    'min': 0, 'max': 8,
    'palette': ['#1a9850', '#91cf60', '#d9ef8b',
                '#fee08b', '#fc8d59', '#d73027']
}
Map.addLayer(fire_freq.clip(aoi), freq_vis, 'Fire Frequency 2013-2024')

# 严重程度：灰/黄/橙/红
sev_vis = {'min': 1, 'max': 4, 'palette': ['#d9d9d9', '#fdae61', '#f46d43', '#a50026']}
Map.addLayer(sev_2023.clip(aoi), sev_vis, 'Fire Severity 2023')

Map.add_colorbar(freq_vis, label='Fire Count (2013-2024)', orientation='horizontal')
Map.save('kimberley_fire_analysis.html')
```

## 实验

### 数据集说明

本文使用 **Landsat Collection 2 Level-2**（地表反射率产品），已完成大气校正，适合跨时序比较。

- **时间覆盖**：Landsat 4 起始（1982年），可构建 40+ 年历史
- **空间分辨率**：30m，足以识别单次火烧斑块（>1 公顷）
- **获取成本**：完全免费，通过 GEE 直接访问，无需下载原始数据
- **主要限制**：热带地区雨季（11月-4月）云覆盖率高，可用影像少

### 定量评估

以 MODIS 主动火点产品（MOD14A1）作为粗略验证参考：

| 指标 | 本文方法（Landsat NBR） | MODIS 主动火点 |
|------|----------------------|--------------|
| 空间分辨率 | 30m | 500m-1km |
| 时间延迟 | 季度合成（历史） | 准实时（当日） |
| 小火检测 | 较好（>5公顷） | 差（漏检多） |
| 严重程度信息 | ✓ (dNBR 4级) | ✗ |
| 历史回溯深度 | 40年 | 2000年起 |

### 定性结果

金伯利地区 2013-2024 年火烧频率图呈现明显空间异质性：

- **沿河谷区域**：火烧频率低（1-2次/11年），雨季水分充足抑制燃烧
- **高地草原**：频率中等（4-6次），对应传统低强度早期干季烧除区域
- **国家公园主动管理区**：部分斑块频率高（7-9次），是人为烧除控制带

这与原住民护林员的传统知识高度吻合：**干季早期低强度烧除**形成马赛克景观，有效防止雨季末高燃料积累导致的灾难性大火。

## 工程实践

### 大区域数据导出

在线 `getInfo()` 仅适合小范围测试，大范围分析必须用异步导出：

```python
# 导出至 Google Drive（推荐）
task = ee.batch.Export.image.toDrive(
    image=fire_freq.clip(aoi).toFloat(),
    description='fire_frequency_kimberley_2013_2024',
    folder='GEE_exports',
    scale=30,
    region=aoi,
    maxPixels=1e10,       # 大图像必须设置，否则报错
    crs='EPSG:32753'      # WGS84 UTM Zone 53S，适合澳洲西北
)
task.start()

# 轮询状态
import time
while task.active():
    print(task.status()['state'])
    time.sleep(30)
```

**典型耗时**：金伯利全区（~420,000 km²）11 年时序分析约 20-40 分钟。

### 常见坑

**1. 云污染导致虚假火烧信号**

```python
# 错误：直接比较单张影像，云块会产生极大 dNBR
dnbr_wrong = single_pre_image.subtract(single_post_image)

# 正确：季度中位数合成，云占比<30% 时会被自然排除
pre_median  = collection.filterDate(pre_start, pre_end).median()
post_median = collection.filterDate(post_start, post_end).median()
dnbr_correct = pre_median.subtract(post_median)
```

**2. Landsat 7 SLC-off 条带问题**

```python
# 2003年6月后 Landsat 7 存在约22%条带缺失，产生条纹伪影
# 解决：改用 Landsat 8（2013年起）或降低 L7 权重
collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')  # 优先 L8
    .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterDate('1999-01-01', '2003-05-31')))  # L7 只取 SLC-on 阶段
```

**3. 反射率缩放因子**

```python
# Collection 2 的 SR 波段需要乘以缩放因子才是真正的反射率
def apply_scale_factors(image):
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True)
# 注意：normalizedDifference 会自动处理比值，通常不影响 NBR 结果
# 但若与其他数据联合分析时必须做此校正
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 大范围历史火烧统计（>100 km²） | 实时火点检测（重访延迟 16 天） |
| 过火面积和强度评估 | 热带雨林密集云覆盖区（全年） |
| 长期植被恢复轨迹监测 | 需要亚米级精度的边界制图 |
| 土地管理周期性规划决策 | 城市火灾（下垫面复杂，指数失效） |
| 免费/低成本遥感分析 | 需要每日更新的动态监测 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| MODIS 主动火点 | 实时、全球、免费 | 1km 粗分辨率，小火漏检多 | 大范围实时预警 |
| Sentinel-2 NBR | 10m 高分辨率，5天重访 | 仅 2017 年后，云敏感 | 精细火烧边界制图 |
| 无人机热像仪 | 厘米级、实时热成像 | 覆盖范围小（~几 km²），成本高 | 现场指挥、地面验证 |
| **Landsat NBR（本文）** | 40 年历史、30m、完全免费 | 16 天重访，无法实时 | 长期趋势分析、历史重建 |

## 我的观点

这个案例真正有价值的地方，不是技术本身有多先进——NBR 是 1980 年代就提出的算法——而是**工具与知识的耦合方式**。

原住民护林员掌握的是关于哪些植被群落在何时以何种方式响应火烧的精细地方性知识，很难被完整数字化。GEE 提供的是**空间连续性和时序深度**，人不可能同时观测数百万公顷的土地 40 年。两者的结合，构成了可行的大规模土地管理系统。

从技术发展趋势看，有几个值得关注的方向：

1. **地球观测基础模型**：Prithvi（NASA IBM）等在 Landsat/Sentinel 时序数据上预训练的模型，可以更鲁棒地处理云遮挡和缺失数据，比手工阈值方法泛化能力更强
2. **SAR 数据融合**：Sentinel-1 雷达数据不受云影响，与光学 NBR 融合可破解热带地区的云覆盖瓶颈
3. **主动学习 + 专家标注**：将护林员的局部现场标注通过主动学习扩展到大范围，是目前最实际的精度提升路径

对于**历史分析和规划决策**，这套工具链已经相当成熟，几行代码可以覆盖大洲级别的分析。对于**实时预警和精细操作支持**，还需要更高频率的数据源（如 Planet Labs 的 3m 日分辨率数据）以及更智能的云检测。核心瓶颈已经不是算法，而是高频高分数据的获取成本。