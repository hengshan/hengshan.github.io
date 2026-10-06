---
layout: post-wide
title: 'FairRSFM:遥感基础模型的"隐藏偏见"诊断框架'
date: 2026-10-06 08:02:11 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2610.05790v1
generated_by: Claude Code CLI
---

Let me check for relevant memory about blog-writing preferences before drafting.

---

I checked memory — no prior feedback stored specifically on blog-writing style, so I'll follow the detailed template provided. Writing the post now.


## 一句话总结

遥感基础模型（RSFM）在聚合指标上表现亮眼（比如宏观 F1 做到 90%+），但这个数字可能掩盖了它在某些生态区域（比如荒漠、苔原）上表现很差的事实。FairRSFM 提出了一套按生物群落（biome）分组评估的基准协议，专门用来揪出这种"平均分很高、最差组很差"的鲁棒性陷阱。

## 为什么这个问题重要？

### 应用场景

遥感基础模型的下游应用几乎都带有强烈的地理属性：

- **农业监测**：作物分类模型要在热带雨林农田和温带草原农田上都work
- **生态保护**：森林退化检测要覆盖从赤道雨林到北方针叶林
- **灾害响应**：洪水/火灾分割模型要在不同地貌下保持可靠
- **碳核算**：土地覆盖分类直接影响碳排放估算的准确性

这些场景的共同特点是：**训练数据的地理分布天然不均衡**。欧洲、北美的标注数据多，中亚荒漠、高山苔原的数据少。

### 现有方法的问题

遥感基础模型的评测几乎都停留在"整体准确率"这一个数字上。论文指出的核心问题是：

> 聚合指标（aggregate metrics）系统性地隐藏了跨生态区域的性能差异。

这不是一个抽象的担忧，论文给出了具体数字：

- Prithvi-EO-2.0 在 m-EuroSAT 上整体宏观 F1 达到 **90.98%**，但最差分组（worst-group）只有 **83.72%**
- 在 m-SA-Crop-Type 任务上，整体 mIoU 是 27.30%，但在 Xeric（干旱）和矿质土壤组上掉到 **18.47%**——几乎是腰斩

对于一个要在真实世界部署的模型，这意味着：如果你的应用场景恰好落在某个"弱势生物群落"，模型的实际表现会远低于论文汇报的数字。

### 这个方法的核心创新

FairRSFM 不是提出一个新的模型架构，而是提出一套**诊断协议**：

1. 把地理坐标映射到生态学上有意义的分组（而不是任意的地理网格）
2. 在统一的 frozen-backbone 协议下评估多个 RSFM
3. 同时测试几种"偏见缓解"方法，看它们是否真的有效

这种"benchmark + mitigation baseline"的组合，让它既是诊断工具，也是改进起点。

## 背景知识

### 遥感数据的特殊性

和自然图像不同，遥感影像有几个关键特征决定了这篇论文的方法论：

- **多光谱/高光谱通道**：Sentinel-2 有 13 个波段，不只是 RGB
- **地理坐标元数据**：每个样本都带经纬度，这是做生物群落分组的前提
- **标签语义的地理依赖性**：同样是"裸地"类别，在荒漠和冻土带的光谱特征完全不同

### 什么是"生物群落"（Biome）分组

论文使用的是陆地生态系统的 14 类生物群落分类（如热带雨林、温带草原、苔原、荒漠等），这是生态学界广泛使用的 WWF/Olson 生物群落体系的变体。FairRSFM 把这 14 类进一步聚合成 **6 个宏观生态组**，原因是：

- 14 类太细，很多类别样本量太少，统计不稳定
- 6 个宏观组保留了生态学意义（比如把"热带雨林"和"热带季雨林"合并为"湿润热带"组），同时保证每组有足够样本做可靠评估

### Worst-group 指标：鲁棒性评估的核心工具

这是从分布鲁棒优化（Distributionally Robust Optimization, DRO）领域借来的概念。与"整体准确率"不同：

$$
\text{Worst-group score} = \min_{g \in \mathcal{G}} \text{Metric}(g)
$$

其中 $\mathcal{G}$ 是所有分组的集合。这个指标回答的问题是："模型在**最差的那个子群体**上表现如何？"——这正是整体准确率会掩盖的信息。

### 读者需要的前置知识

- 了解基础的图像分类/分割评估指标（F1、mIoU）
- 了解 frozen-backbone linear probing 的评估范式（即不微调主干网络，只训练一个线性探针头）
- 不需要深入的生态学背景，论文的生物群落分组可以直接当作一种"地理元数据驱动的分组标签"来理解

## 核心方法

### 直觉解释

把整个评估流程想象成一次"分层体检"：

```
传统评估：   全体样本 → 模型 → 一个平均分（90.98%）
                ↓ 隐藏了问题
FairRSFM：   全体样本 → 按生物群落分组 → 每组单独算分
             组1: 92%   组2: 88%   组3: 83.72% ← 这才是真实的下限
```

传统评估就像只看一个班级的平均分，FairRSFM 则是看每个学习小组的平均分——如果有一个小组远低于平均水平，说明教学方法对这个小组是失效的。

### 数学细节

**Step 1：地理坐标到生物群落的映射**

每个样本的地理坐标 $(\text{lat}, \text{lon})$ 通过空间查找表映射到一个生物群落标签 $b \in \{1, ..., 14\}$，再聚合为宏观组 $g \in \{1, ..., 6\}$：

$$
g = \phi(b), \quad \phi: \{1,...,14\} \rightarrow \{1,...,6\}
$$

**Step 2：分组内指标计算**

对每个宏观组 $g$，在该组样本子集 $\mathcal{D}_g$ 上计算任务指标（分类用 macro-F1，分割用 mIoU）：

$$
\text{Metric}(g) = \frac{1}{|\mathcal{D}_g|}\sum_{i \in \mathcal{D}_g} \ell(y_i, \hat{y}_i)
$$

**Step 3：三个缓解方法的优化目标**

- **GroupDRO**：最小化最差组的损失，而不是平均损失

$$
\min_\theta \max_{g \in \mathcal{G}} \mathbb{E}_{(x,y)\sim \mathcal{D}_g}[\ell(f_\theta(x), y)]
$$

- **Dynamic Biome Reweighting (DBR)**：训练过程中动态调整各组的采样权重 $w_g$，使表现差的组获得更多采样概率

$$
w_g^{(t+1)} = w_g^{(t)} \cdot \exp(\eta \cdot (1 - \text{Metric}_g^{(t)}))
$$

- **Biome-Orthogonal Linear Probing (BOLP)**：在冻结的 backbone 特征上，训练一个线性探针，但对探针权重加正交约束，使不同组的决策方向相互解耦，减少组间干扰

### Pipeline 概览

```
georeferenced 原始影像(多光谱)
      ↓
Frozen RSFM Backbone (Prithvi-EO-2.0 / SatMAE / DOFA)
      ↓ 提取特征
按地理坐标分组 (14 biome → 6 macro-group)
      ↓
线性探针头 (Linear Probe / BOLP / DBR-weighted / GroupDRO)
      ↓
分组指标评估 (per-group F1 / mIoU)
      ↓
Worst-group score + Overall score 对比
```

## 实现

下面的代码展示 FairRSFM 评估协议的核心骨架：分组指标计算 + GroupDRO 训练循环。完整的数据集加载和生物群落映射表请参考官方仓库。

### 环境配置

```bash
# 核心依赖
pip install torch torchvision scikit-learn numpy

# 数据集与官方代码
git clone https://github.com/aminurhossain/FairRSFM
```

### 核心代码：分组评估协议

```python
import torch
import numpy as np
from collections import defaultdict

def group_metric(preds, labels, groups, metric_fn, num_groups=6):
    """按生物群落宏观组计算指标，返回每组得分和 worst-group 得分"""
    group_scores = {}
    for g in range(num_groups):
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        # 只在该组子集上计算任务指标（如 macro-F1 / mIoU）
        group_scores[g] = metric_fn(preds[mask], labels[mask])

    overall = metric_fn(preds, labels)
    worst_group = min(group_scores.values())
    return {
        "overall": overall,
        "worst_group": worst_group,
        "per_group": group_scores,
    }
```

### 核心代码：GroupDRO 训练循环

```python
def train_group_dro(probe, features, labels, groups, num_groups=6,
                     epochs=50, lr=1e-3, eta=0.1):
    """
    在 frozen RSFM 特征上训练线性探针，
    使用 GroupDRO 优化最差组损失而非平均损失
    """
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    # 组权重初始化为均匀分布
    group_weights = torch.ones(num_groups) / num_groups

    for epoch in range(epochs):
        logits = probe(features)
        group_losses = torch.zeros(num_groups)

        for g in range(num_groups):
            mask = (groups == g)
            if mask.sum() == 0:
                continue
            loss_g = torch.nn.functional.cross_entropy(
                logits[mask], labels[mask]
            )
            group_losses[g] = loss_g

        # 动态上调损失高的组的权重（指数梯度上升）
        with torch.no_grad():
            group_weights *= torch.exp(eta * group_losses)
            group_weights /= group_weights.sum()  # 归一化

        # 用加权组损失反向传播，而非简单平均
        weighted_loss = (group_weights.detach() * group_losses).sum()
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

    return probe
```

### 可视化：每组得分对比

```python
import matplotlib.pyplot as plt

def plot_group_scores(results, biome_names):
    """柱状图展示整体得分 vs 各生物群落组得分，突出 worst-group gap"""
    groups = list(results["per_group"].keys())
    scores = [results["per_group"][g] for g in groups]
    names = [biome_names[g] for g in groups]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, scores, color="steelblue")
    ax.axhline(results["overall"], color="red", linestyle="--",
               label=f"Overall: {results['overall']:.2f}")
    # 标红最差组
    worst_idx = scores.index(min(scores))
    bars[worst_idx].set_color("firebrick")

    ax.set_ylabel("Macro-F1 / mIoU")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("biome_gap.png", dpi=150)
```

这类柱状图是这篇论文论证的核心证据形式——一条红色虚线（整体分数）和若干柱子（各组分数），一眼就能看出哪个生物群落被模型"牺牲"了。

## 实验

### 数据集说明

FairRSFM 覆盖四个下游数据集，都是遥感基准测试里常用的：

| 数据集 | 任务 | 分辨率/通道 |
|-------|------|-----------|
| m-EuroSAT | 场景分类 | Sentinel-2，13 波段 |
| m-BigEarthNet | 多标签分类 | Sentinel-2，13 波段 |
| m-SA-Crop-Type | 作物类型分割 | Sentinel-2 时序 |
| MMEarth20K (Dynamic World) | 土地覆盖分割 | 多模态 |

这些数据集本身公开可下载，但**生物群落标签需要额外用地理坐标做空间关联查询**（通过 WWF Olson biome shapefile），这是 FairRSFM 的工程增量，不是数据采集增量。

### 定量评估

论文用三个 backbone（Prithvi-EO-2.0、SatMAE、DOFA）跑三个随机种子，核心发现整理如下：

| 模型 | 数据集 | Overall | Worst-group |
|-----|-------|---------|-------------|
| Prithvi-EO-2.0 | m-EuroSAT | 90.98% | 83.72% |
| Prithvi-EO-2.0 | m-SA-Crop-Type | 27.30% (mIoU) | 18.47% (mIoU) |

缓解方法效果（以 Prithvi-EO-2.0 + m-BigEarthNet 为例）：

| 方法 | Worst-group F1@opt |
|-----|---------------------|
| 普通 Linear Probe | 46.12% |
| BOLP | 50.27% |

注意论文强调：**这些缓解方法的效果是模型和任务依赖的**，不存在一个"放之四海而皆准"的最优方案——这是诚实的实验报告，不是在吹嘘某个方法万能。

### 定性结果

论文中 m-SA-Crop-Type 在 Xeric/矿质土壤组上的断崖式下跌（27.30% → 18.47%）特别值得关注：这类地貌的作物光谱特征和训练数据主体（可能偏向湿润农业区）差异大，模型等于是在用"错配"的先验做推断。

## 工程实践（重要！）

### 实际部署考虑

- **这不是一个实时系统**：FairRSFM 是离线评估协议，frozen-backbone + linear probe 的组合计算开销很小（backbone 前向一次，探针训练是轻量级的），在单张消费级 GPU（如 RTX 3090）上跑完一个数据集的多组评估通常是分钟到小时级别
- **内存瓶颈在特征缓存**：如果数据集较大（如 MMEarth20K），建议先把 frozen backbone 的特征提取并缓存到磁盘，而不是每个 epoch 都重新跑一遍 backbone
- **GroupDRO 的训练不稳定性**：组权重的指数更新（`eta` 参数）如果设得太大，会导致权重震荡，论文里用的多个随机种子平均正是为了应对这种不稳定性

### 数据采集建议

如果你要把这套框架用在自己的项目上：

- 确保你的地理坐标元数据是**准确的**（不是数据集常见的"中心点近似"坐标），否则生物群落映射会出错
- 小样本的生物群落组（比如极地、高山）要特别小心——样本量不足会让 worst-group 指标本身变得不可靠，论文用 3 个随机种子来缓解这个问题，实际使用中建议种子数更多

### 常见坑

1. **问题**：直接用整体准确率/F1 做模型选型，选出来的模型在边缘地区部署后效果远不及预期
   **解决方案**：在模型选型阶段就加入 worst-group 指标作为筛选条件，而不是上线后才发现问题

2. **问题**：生物群落分组样本量严重不均衡（比如热带雨林组有 10 万样本，苔原组只有 200 个），导致 worst-group 指标噪声很大
   **解决方案**：对小样本组做 bootstrap 置信区间估计，或者在报告 worst-group 分数时同时报告该组的样本量

3. **问题**：GroupDRO 训练时某个组因为样本极少导致损失剧烈震荡，拖累整体收敛
   **解决方案**：对组权重更新加平滑/动量项，或者对样本量过小的组设置权重上限

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要跨地理区域部署的遥感模型选型 | 单一地理区域、场景同质的应用（比如只做某一个国家的城市用地分类） |
| 做 RSFM 的公平性/鲁棒性审计 | 需要实时在线评估的生产监控系统 |
| 下游任务有明确的地理坐标元数据 | 坐标信息缺失或不准确的数据集 |
| 对比不同 backbone 的鲁棒性差异 | 需要评估模型架构本身的创新点（这是诊断工具，不是模型） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 传统聚合指标评估 | 简单、计算快、行业标准 | 掩盖分布不均衡导致的性能差异 | 快速模型筛选、初步对比 |
| 通用 DRO/公平性框架（如 WILDS） | 通用性强，跨领域适用 | 分组定义不具遥感领域的生态学意义 | 自然图像、医疗影像等领域 |
| FairRSFM | 分组有生态学依据，覆盖多个 RSFM 和任务 | 依赖准确的地理坐标元数据；缓解方法效果不稳定 | 遥感基础模型的鲁棒性审计 |

## 我的观点

FairRSFM 本质上不是一个"新模型"，而是把 DRO/公平性研究的方法论严肃地搬到了遥感基础模型评估上——这个动作本身姗姗来迟但很有必要，因为遥感数据的地理不均衡性比大多数视觉领域都更结构化、更容易被生态学知识明确分组。

值得关注的几点：

- 论文诚实地报告了缓解方法"model- and task-dependent"的效果，没有强行包装成银弹方案，这种态度在基准测试论文里是值得肯定的
- 开放问题在于：当某个生物群落组本身样本稀少时，"提升 worst-group 表现"和"过拟合到噪声大的小组"之间的边界并不清晰，这需要更细致的统计显著性检验
- 离工业落地还有距离——目前的工作停留在 frozen-backbone linear probing 评估，还没有验证这些分组鲁棒性结论在端到端微调或更大规模部署中是否依然成立

对于任何要把 RSFM 用在跨地理区域项目（尤其是农业、生态监测相关）的团队，这篇论文提供的评估协议值得直接拿来用，哪怕你用的不是论文里测试的三个 backbone——"按生物群落分组看 worst-group 指标"这个思路本身就是立刻可以落地的最佳实践。