---
layout: post-wide
title: "遥感图像的分布漂移检测：三款开源工具的实测对比"
date: 2026-11-12 12:01:53 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2611.13510v1
generated_by: Claude Code CLI
---

## 一句话总结

无人机（UAV）部署的机器学习模型，训练集和真实部署环境几乎必然存在分布差异——这篇文章基于一项实证研究（Alibi-Detect、NannyML、PyTorch-OOD 三种工具在跨数据集遥感场景下的对比），复现其核心思路，并给出一套可以直接套用的检测流程、延迟测算方法，以及调参踩坑经验。

## 背景：为什么遥感模型需要漂移检测

遥感图像分类/检测模型训练时用的往往是某个特定地区、特定季节、特定传感器拍摄的数据集（比如夏季、某型号相机、某纬度的农田影像）。一旦部署到实际无人机上：

- **地理漂移**：换了个地区，地表纹理、植被覆盖完全不同
- **传感器漂移**：换了摄像头型号，色彩标定、分辨率不同
- **季节漂移**：训练时是夏季绿色植被，部署时是冬季裸地
- **光照/视角漂移**：飞行高度、太阳角度变化导致阴影和曝光不同

模型本身不会"知道"自己正在对分布外的数据做预测，它会照常输出一个看起来很自信的结果——这才是最危险的地方。更麻烦的是，UAV 是资源受限平台（算力、电量、带宽都有限），不可能跑一个很重的监控系统。

所以问题变成：**用什么方法、花多大代价，能尽早发现"这批图像跟训练时不一样了"？**

## 三种检测思路

在深入代码前，先搞清楚三个工具分别在解决什么问题，因为它们的哲学完全不同。

### Alibi-Detect：统计检验驱动

核心思路是经典的双样本假设检验：把训练集（reference）和线上数据（current）的特征分布做对比，用 MMD（Maximum Mean Discrepancy）或 KS 检验判断"这两批数据是否来自同一分布"。

$$
\text{MMD}^2(P, Q) = \left\| \mathbb{E}_{x \sim P}[\phi(x)] - \mathbb{E}_{y \sim Q}[\phi(y)] \right\|_{\mathcal{H}}^2
$$

直觉：把样本映射到核空间，看两组样本的均值嵌入差多远。优点是统计意义明确、有 p-value；缺点是高维图像特征上计算核矩阵开销不小，而且需要先用编码器把图像降维成特征向量再检测，原始像素直接测基本没意义。

### NannyML：性能估计驱动

NannyML 的切入点不一样——它不直接问"分布变了吗"，而是问"模型性能可能掉了多少"，即使**没有真实标签**。核心方法是 CBPE（Confidence-Based Performance Estimation）：用模型输出的置信度分布来反推预期准确率/F1。

这对 UAV 场景特别实用，因为线上根本没有即时标签（人工标注遥感图像要等很久）。它也提供传统的 univariate drift（每个特征单独做 KS/Chi2 检验）作为辅助诊断。

### PyTorch-OOD：样本级异常检测

前两者是"批次级"判断（这一批数据整体有没有漂移），PyTorch-OOD 做的是**单样本级别**的 OOD（Out-of-Distribution）打分，比如 Mahalanobis 距离、能量分数（Energy Score）、Softmax 最大值等：

$$
E(x) = -\log \sum_{i} e^{f_i(x)}
$$

能量分数越低（更负），说明模型对这张图越"陌生"。适合逐帧实时判断，但天然是无监督的异常点检测，容易受模型校准程度影响——一个过度自信的模型，能量分数区分度会很差。

## 最小可运行示例：模拟遥感特征漂移

真实卫星/无人机图像数据集体量太大，这里用一个可以直接跑的简化版本：用随机生成的"特征向量"模拟 CNN 提取出的遥感图像 embedding，制造一次季节性漂移（均值和方差同时偏移），分别用三类方法检测。

```python
import numpy as np
from scipy.stats import ks_2samp

# 模拟 CNN 提取的 128 维特征向量
# reference: 训练集分布（比如夏季影像）
# current: 部署数据分布（比如冬季影像，均值和方差都偏移了）
rng = np.random.default_rng(42)
reference = rng.normal(loc=0.0, scale=1.0, size=(2000, 128))
current_no_shift = rng.normal(loc=0.0, scale=1.0, size=(500, 128))
current_shifted = rng.normal(loc=0.6, scale=1.4, size=(500, 128))

def ks_drift_score(ref, cur, alpha=0.05):
    """逐维做 KS 检验，返回触发漂移的特征维度占比"""
    n_dims = ref.shape[1]
    n_triggered = 0
    for d in range(n_dims):
        _, p_value = ks_2samp(ref[:, d], cur[:, d])
        if p_value < alpha:
            n_triggered += 1
    return n_triggered / n_dims

print("无漂移场景触发比例:", ks_drift_score(reference, current_no_shift))
print("有漂移场景触发比例:", ks_drift_score(reference, current_shifted))
```

这是 Alibi-Detect 和 NannyML 的 univariate drift 检测在做的事情的简化版——本质就是逐维统计检验，再用某个阈值（比如超过 X% 维度触发）判定整批数据是否漂移。注意：**这里省略了多重检验校正**（Bonferroni），真实场景 128 维逐个做 KS 检验，不校正的话假阳性率会被严重低估。

### Mahalanobis 距离做样本级 OOD 打分

```python
import numpy as np

def fit_mahalanobis(reference):
    """用参考集估计均值和协方差（PyTorch-OOD 内部做法的简化版）"""
    mean = reference.mean(axis=0)
    cov = np.cov(reference, rowvar=False)
    cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))  # 加正则防止奇异
    return mean, cov_inv

def mahalanobis_score(x, mean, cov_inv):
    diff = x - mean
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, cov_inv, diff))

mean, cov_inv = fit_mahalanobis(reference)
scores_no_shift = mahalanobis_score(current_no_shift, mean, cov_inv)
scores_shifted = mahalanobis_score(current_shifted, mean, cov_inv)

print("无漂移场景平均距离:", scores_no_shift.mean())
print("有漂移场景平均距离:", scores_shifted.mean())
```

这段代码的 `pinv` 和正则化项不是摆设——128 维特征如果参考集样本数不够（比如小于特征维度的几倍），协方差矩阵会接近奇异，直接求逆会得到数值爆炸的距离值，所有样本看起来都"极度异常"。这是实践中最容易踩的坑，论文/教程里基本不会提。

## 延迟与检测能力的权衡

这篇论文最有价值的部分其实是**工程视角**：不是比较哪个工具"检测得更准"，而是比较"检测代价"。几个关键观察（基于方法论特性，不是具体复现数字）：

| 方法类型 | 典型延迟来源 | 适合的部署位置 |
|---------|------------|--------------|
| 统计检验（Alibi-Detect MMD/KS） | 核矩阵计算 $O(n^2)$，batch 越大越慢 | 云端/地面站批量巡检 |
| 性能估计（NannyML CBPE） | 轻量，主要是概率分布拟合 | 可在边缘跑，适合周期性汇报 |
| 样本级 OOD（PyTorch-OOD） | 单次前向 + 简单距离计算，延迟低 | 适合机载实时逐帧判断 |

一个实用结论：**批次级统计检验适合"事后巡检"，样本级 OOD 适合"实时拦截"**。如果 UAV 本身算力紧张，机载端可以只跑便宜的能量分数或 Softmax 置信度打分，一旦触发报警再把这批原始数据回传到地面站，用 Alibi-Detect 做更严谨的统计确认——这是典型的分级检测策略，论文摘要里提到的"latency-aware monitoring"大概也是指这个方向。

## 调试指南

### 常见问题

1. **报警风暴（all triggered）**：几乎每批数据都报漂移。最常见原因是参考集太小、方差估计不稳定，或者逐维检验没做多重检验校正——128 维里哪怕每维只有 5% 假阳性率，整体触发概率会被显著放大。
2. **漂移明显但检测不出来（silent failure）**：如果是用预训练编码器提特征再检测，编码器本身对新分布的图像可能提取出"看起来正常"的特征（表征坍缩），这时统计检验在特征空间上看不出异常，但下游任务性能已经掉了。这种情况必须配合 NannyML 式的性能估计，而不能只依赖特征层的分布检验。
3. **Mahalanobis 距离全是 inf 或 nan**：协方差矩阵求逆数值不稳定，加正则化项或者先用 PCA 降维。

### 如何判断"真的在漂移"而不是噪声

不要只看单个 batch 的检验结果，要看**连续多个 batch 的趋势**。偶发的单次触发大概率是采样噪声；如果连续 N 个 batch（N 根据飞行频率定，比如连续 3-5 趟飞行任务）都触发，才值得人工介入复核。

### 阈值设置

| 参数 | 推荐起点 | 敏感度 | 备注 |
|-----|---------|-------|------|
| KS 检验显著性水平 α | 0.05（做多重校正后等效更严格） | 高 | 维度越多越要校正，否则假阳性爆炸 |
| 触发比例阈值（多少维超标算整体漂移） | 10%~20% | 中 | 太低容易过敏感，太高会漏检 |
| Mahalanobis 距离阈值 | 参考集距离分布的 95~99 分位数 | 高 | 不要用固定数值，要用参考集自己校准 |

## 什么时候用 / 不用

| 适用场景 | 不适用场景 |
|---------|-----------|
| 模型部署后缺乏及时标签反馈（NannyML 性能估计） | 有充足实时标签，直接监控准确率更直接 |
| 需要逐帧实时拦截异常输入（PyTorch-OOD） | 批量离线巡检，用统计检验更严谨（Alibi-Detect） |
| 已知会有季节性/地理性漂移的长期部署任务 | 一次性短期任务，漂移风险低，监控系统的开销可能不值得 |

## 我的观点

这三个工具没有互相替代关系，它们回答的是三个不同的问题："分布变了吗"（Alibi-Detect）、"性能可能掉了吗"（NannyML）、"这一张图正常吗"（PyTorch-OOD）。在资源受限的 UAV 场景下，单独依赖任何一个都有盲区——纯统计检验对小样本敏感，纯 OOD 打分依赖模型校准质量，纯性能估计对协变量漂移不一定敏感（因为它本质看的是输出置信度，不是输入分布）。

比较现实的做法是分级部署：机载端跑便宜的样本级打分做"初筛"，一旦触发再回传数据到地面站做更严谨的批次级统计检验，同时用性能估计工具周期性地评估"即使没报警，模型可能已经悄悄退化了多少"。这篇论文提到的延迟差异其实提示了一个方向——与其追求单一"最优"检测器，不如把检测拆成不同成本层级，按需触发。