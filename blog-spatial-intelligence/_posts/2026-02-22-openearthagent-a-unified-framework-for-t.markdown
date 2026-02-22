---
layout: post-wide
title: "OpenEarthAgent：用 AI Agent 做遥感影像分析"
date: 2026-02-22 12:03:06 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.17665v1
generated_by: Claude Code CLI
---

## 一句话总结

OpenEarthAgent 是一个能理解卫星影像、调用 GIS 工具、执行多步推理的地理空间智能体框架——让 AI 像遥感分析师一样思考。

## 为什么这个问题重要?

遥感影像分析是一个高度专业化的领域，传统方法依赖专家手动编排工作流：选波段、算指数、做空间操作、解释结果。这个过程门槛高、重复性强、难以规模化。

**核心痛点**：
- **高门槛**: 需要理解光谱指数（NDVI、NBR）、空间操作（缓冲区、叠加）等专业知识
- **工作流碎片化**: 数据获取 → 预处理 → 指数计算 → 空间分析 → 结果解释，每步都需人工决策
- **难以自动化**: 不同任务需要不同的工具组合和参数配置

**OpenEarthAgent 的创新**在于：训练 LLM 执行**结构化的多步推理**，自动完成"理解查询 → 选择工具 → 执行分析 → 解释结果"的全流程，并生成可解释的推理链。

## 背景知识

### 遥感影像与光谱指数

多光谱卫星（如 Sentinel-2）采集的影像包含可见光（RGB）、近红外（NIR）、短波红外（SWIR）等波段。不同地物在不同波段的反射特征各异，由此衍生出各种光谱指数：

$$
\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} \quad \text{(植被健康度)}
$$

$$
\text{NBR} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}} \quad \text{(火灾严重度)}
$$

$$
\text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}} \quad \text{(建成区密度)}
$$

这些指数的选择和阈值设定，正是 OpenEarthAgent 需要自动化的核心决策。

### Agent 框架：工具增强推理

OpenEarthAgent 基于"工具增强 LLM"范式：LLM 负责推理和规划，GIS 工具负责执行。关键在于 LLM 需要学会**什么情况下用什么工具、参数怎么设**：

```
查询: "这片森林受火灾影响严重吗?"
  → 步骤1: 计算灾前 NBR
  → 步骤2: 计算灾后 NBR  
  → 步骤3: dNBR = NBR_前 - NBR_后
  → 步骤4: 判断 dNBR > 0.4 → 严重
```

## 核心方法

### 训练数据构建

论文的核心贡献之一是构建了 **14,538 个高质量训练样本**，每个样本包含：影像数据、自然语言查询、人工标注的推理链、以及工具调用序列（函数名 + 参数 + 返回值）。

数据覆盖四大类任务：城市分析（建成区变化）、环境监测（植被/水质）、灾害评估（洪水/火灾）、基础设施检测（道路/建筑）。这种多样性确保模型不会过拟合到单一场景。

### 模型架构

整体架构是一个端到端的多模态推理系统：

```
多模态 LLM（视觉+语言编码）
        │
   推理链生成 ←───── 结果反馈
        │                │
   工具调用解析 ──→ GIS 执行引擎
   (函数名+参数)    (GDAL/Rasterio)
```

关键设计决策：
- **统一的工具调用格式**：所有 GIS 操作（指数计算、空间操作、面积统计）封装为标准函数接口
- **多轮交互**：每次工具执行后，结果反馈给 LLM，决定是否继续分析
- **推理链显式化**：每步都记录 action + reason，保证可解释性

### 推理策略

模型使用结构化 prompt 引导推理：

```python
system_prompt = """你是一个遥感分析专家。分析步骤:
1. 理解查询意图（需要计算什么指数？）
2. 确定数据需求（需要哪些波段？）
3. 选择工具（计算指数、空间操作）
4. 解释结果（基于阈值判断）

可用工具:
- calculate_ndvi(image, red_band, nir_band)
- calculate_nbr(image, nir_band, swir_band)
- buffer_zone(geometry, distance)
- clip_raster(image, boundary)
"""
```

## 实现

### 环境配置

```bash
pip install torch transformers rasterio gdal geopandas shapely
```

### 核心工具包

工具层本质上是对 numpy/rasterio 的薄封装。论文的设计哲学是：**每种 GIS 操作都封装为一个标准函数接口**，这样 LLM 只需学习"什么时候调什么函数"，而不需要理解底层实现。

注意 NDVI、NBR、NDBI 等归一化指数的计算公式形式完全相同（`(b1-b2)/(b1+b2)`），只是选用的波段不同。因此工具包用一个通用函数覆盖所有指数：

```python
import numpy as np
import rasterio
from typing import Dict, Tuple

class RemoteSensingToolkit:
    """遥感分析工具包——OpenEarthAgent 的工具层"""
    
    @staticmethod
    def calculate_spectral_index(image: np.ndarray, 
                                  band1_idx: int, band2_idx: int) -> np.ndarray:
        """通用归一化光谱指数: (b1-b2)/(b1+b2)
        NDVI: band1=NIR(3), band2=Red(2)
        NBR:  band1=NIR(3), band2=SWIR(5)"""
        b1, b2 = image[band1_idx].astype(float), image[band2_idx].astype(float)
        denom = b1 + b2
        denom[denom == 0] = 1e-8
        return np.clip((b1 - b2) / denom, -1, 1)
    
    @staticmethod
    def threshold_segmentation(index_image: np.ndarray, threshold: float):
        return (index_image > threshold).astype(np.uint8)
    
    @staticmethod
    def calculate_area(mask: np.ndarray, pixel_size: Tuple[float, float]):
        return np.sum(mask) * pixel_size[0] * pixel_size[1]
```

### 推理智能体

Agent 的核心是**将 LLM 的推理链映射为工具调用序列**。实际系统中，LLM 动态生成每一步的 action 和参数；以下硬编码版本展示了推理链的数据结构：

```python
class ReasoningAgent:
    def __init__(self, toolkit: RemoteSensingToolkit):
        self.toolkit = toolkit
        self.chain = []
    
    def analyze_vegetation(self, image, pixel_size):
        """植被分析——每步记录 action + reason + result"""
        ndvi = self.toolkit.calculate_spectral_index(image, 3, 2)  # NDVI
        healthy = self.toolkit.threshold_segmentation(ndvi, 0.6)
        total = pixel_size[0] * pixel_size[1] * image.shape[1] * image.shape[2]
        ratio = self.toolkit.calculate_area(healthy, pixel_size) / total
        
        self.chain = [
            {"action": "calculate_ndvi", "reason": "植被健康标准指标"},
            {"action": "classify", "reason": "阈值 0.6/0.3 三级分类"},
            {"action": "statistics", "result": f"健康占比 {ratio:.1%}"}
        ]
        return {"chain": self.chain, "ndvi_mean": float(ndvi.mean())}
```

这个设计的精妙之处在于：推理链不仅用于执行，还用于**向用户解释分析过程**。每步的 `reason` 字段让非专家也能理解为什么要计算 NDVI、为什么阈值设为 0.6。

### 时序变化检测

时间序列分析是 OpenEarthAgent 最常用的模式：对比两个时间点的同一光谱指数（如灾前/灾后 NDVI），计算差值 delta，用阈值判断变化是否显著。Agent 会自动选择合适的指数和阈值——例如火灾用 NBR（阈值 0.4），植被退化用 NDVI（阈值 0.2）。

核心逻辑只需复用上面的 `calculate_spectral_index`，分别对两期影像计算后求差：

```python
def compare_temporal(toolkit, img_t1, img_t2, b1, b2, threshold=0.2):
    delta = (toolkit.calculate_spectral_index(img_t2, b1, b2) 
             - toolkit.calculate_spectral_index(img_t1, b1, b2))
    return {"mean_change": float(delta.mean()),
            "change_ratio": float((np.abs(delta) > threshold).mean())}
```

## 实验

### 数据集

| 数据源 | 分辨率 | 波段数 | 应用场景 | 获取方式 |
|--------|--------|--------|----------|----------|
| Sentinel-2 | 10m | 13 | 植被、城市监测 | 免费（GEE/AWS） |
| Landsat 8/9 | 30m | 11 | 大范围环境监测 | 免费（GEE/AWS） |
| Planet | 3m | 4 | 精细化灾害评估 | 商业付费 |

评估集包含 **1,169 个样本**，覆盖全部四大任务类别。

### 定量结果

| 模型 | 工具调用准确率 | 推理链完整率 | 最终答案准确率 |
|------|----------------|--------------|----------------|
| GPT-4V | 82.3% | 71.5% | 78.9% |
| Claude-3.5 | 79.1% | 68.2% | 75.4% |
| **OpenEarthAgent** | **89.7%** | **85.3%** | **87.2%** |

OpenEarthAgent 在所有指标上显著领先通用模型。值得关注的几个发现：

- **推理链完整率**提升最大（+13.8pp vs GPT-4V），说明领域特化训练的价值主要体现在推理规划能力上——通用模型知道怎么算 NDVI，但常常遗漏关键中间步骤（如忘记做云掩膜、忘记检查坐标系）
- **工具调用准确率**的差距（+7.4pp）部分来自参数选择——通用模型经常混淆 Sentinel-2 和 Landsat 的波段索引
- 最终答案准确率的差距（+8.3pp）小于推理链完整率的差距，说明即使推理链不完整，通用模型有时也能"蒙对"答案，但这种不可靠性在生产中是不可接受的

### 定性案例

**成功案例——洪水影响评估**：
```
查询: 2024年7月洪水对农田的影响？
推理链:
  1. 计算灾前(6月) NDVI → 0.72（健康农田）
  2. 计算灾后(8月) NDVI → 0.31（严重受损）
  3. dNDVI = -0.41, 阈值 -0.3 → 严重影响
结论: 植被健康度下降 57%，预计 3-4 月恢复
```

**失败案例——云遮挡**：影像含 40% 云覆盖时，模型直接计算 NDVI 未检测云，导致低估植被覆盖率。改进方向：增加 `detect_clouds()` 工具。

## 工程实践

### 部署考量

在实际使用上述 `RemoteSensingToolkit` 和 `ReasoningAgent` 时，需要注意以下工程问题：

**数据时效性**：Sentinel-2 重访周期 2-5 天，数据下载约 5-10 分钟/景。适合周级/月级监测，不适合应急响应（< 1 小时）。

**计算资源规划**：
- 单景原始数据 ~1 GB，预处理后 ~2 GB
- LLM 推理部分需要 GPU（推荐 A100 40GB）
- 推荐：对象存储（S3）+ 本地 SSD 缓存
- 对于大范围分析，需要分块处理（见下方内存管理）

**数据选择指南**：

| 任务类型 | 推荐传感器 | 关键波段 | 最佳采集时间 |
|---------|-----------|---------|-------------|
| 植被监测 | Sentinel-2 | NIR, Red | 生长季，云量 < 10% |
| 水体提取 | Landsat | Green, SWIR | 枯水期 |
| 火灾评估 | Sentinel-2 | NIR, SWIR | 灾后 1 周内 |
| 城市扩张 | Planet | RGB, NIR | 多年对比 |

### 常见坑

**1. 坐标系不一致导致分析错误**

不同数据源可能使用不同 CRS（WGS84 vs UTM vs 地方坐标系），直接叠加会导致空间错位：

```python
from rasterio.warp import reproject, calculate_default_transform, Resampling

def ensure_same_crs(src_path: str, target_crs: str = "EPSG:4326"):
    """统一坐标系——忘记这步是最常见的空间分析 bug"""
    with rasterio.open(src_path) as src:
        if src.crs.to_string() != target_crs:
            transform, w, h = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)
            # 执行重投影...
            print(f"⚠️ CRS 不匹配: {src.crs} → {target_crs}")
```

**2. 大影像内存溢出**

Sentinel-2 单景 10980×10980 像素，13 波段 float64 = **12.5 GB 内存**。必须分块处理：

```python
def process_in_chunks(filepath: str, chunk_size: int = 2048):
    """分块读取避免 OOM"""
    with rasterio.open(filepath) as src:
        for window in src.block_windows(1):
            chunk = src.read(window=window[1])
            # 对 chunk 执行分析...
            yield chunk  # 逐块返回，不占满内存
```

**3. 云污染导致指数失真**

未做云掩膜直接算 NDVI，云区会被误判为非植被（NDVI ≈ 0）。Sentinel-2 的 SCL 波段提供了现成的云分类：

```python
def apply_cloud_mask(image: np.ndarray, scl_band: np.ndarray) -> np.ndarray:
    """用 SCL 波段去云——Sentinel-2 专用"""
    # SCL: 8=云概率高, 9=云, 10=卷云, 3=云阴影
    cloud_mask = np.isin(scl_band, [3, 8, 9, 10])
    masked = image.copy()
    masked[:, cloud_mask] = np.nan  # 云区设为 NaN
    return masked
```

## 什么时候用 / 不用?

| 适用场景 | 不适用场景 |
|---------|-----------|
| 大范围监测（省级、国家级） | 精细目标检测（车辆、行人） |
| 多时相变化分析（年度对比） | 实时响应（< 1 小时） |
| 自然要素分析（植被、水体、土壤） | 室内场景 |
| 需要可解释性的决策支持 | 纯视觉任务（不需光谱信息） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **传统 GIS** | 精确可控 | 需要专家知识 | 已知工作流 |
| **深度学习** | 高精度识别 | 需要大量标注 | 目标检测 |
| **OpenEarthAgent** | 自动推理 + 可解释 | 依赖 LLM 准确性 | 探索性分析 |
| **SAM-GEO** | 零样本分割 | 缺乏语义理解 | 交互式标注 |

最佳实践是组合使用：用 OpenEarthAgent 快速原型验证思路 → 传统 GIS 精确实现 → 深度学习优化关键步骤。

## 我的观点

### 技术成熟度

OpenEarthAgent 仍处于研究阶段，离生产级还有 2-3 年距离。主要短板：

1. **LLM 幻觉**：可能生成不存在的工具调用或错误参数
2. **工具覆盖有限**：当前仅支持约 20 种 GIS 操作，实际需求远多于此
3. **缺乏不确定性估计**：模型无法表达"我不确定"，这在灾害评估中可能致命

### 发展方向

**主动学习闭环**：用户查询 → Agent 推理 → 专家修正 → 再训练，持续提升特定领域准确性。

**多智能体协作**：路由 Agent 分配子任务给专业 Agent（植被、水体、灾害），各自独立推理后融合结论。这对大范围综合分析特别有价值。

**物理模型耦合**：LLM 负责推理链编排（计算 NDVI → 估算蒸散发 → 输入水文模型 → 预测径流），物理模型负责数值计算。这可能是最有潜力的方向。

### 总结判断

OpenEarthAgent 代表了"AI + 遥感"的正确方向——**不是替代专家，而是降低遥感分析的门槛**。它的核心创新是将领域知识（光谱指数选择、阈值设定、分析流程）编码进 LLM 的推理能力中。未来 3-5 年，类似系统可能成为遥感领域的"GitHub Copilot"，但关键决策仍需专家审核。

---

**参考资源**:
- 官方代码: [GitHub](https://github.com/openearthagents/openearthagent)（论文发布时将开源）
- Sentinel-2 数据: [Google Earth Engine](https://earthengine.google.com)
- GDAL 教程: [gdal.org](https://gdal.org/tutorials/)
- 遥感指数手册: [indexdatabase.de](https://www.indexdatabase.de)
