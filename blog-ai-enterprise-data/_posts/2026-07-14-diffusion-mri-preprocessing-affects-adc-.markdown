---
layout: post-wide
title: "前列腺 DWI 预处理：你的深度学习模型可能在学习伪影"
date: 2026-07-14 08:04:38 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.11385v1
generated_by: Claude Code CLI
---

## 一句话总结

DWI 预处理（去噪 → Gibbs 校正 → 磁化率失真校正）不是可选项——跳过它，你的 PI-RADS 分类器正在把几何形变和振铃伪影当成病理特征来学习。

## 背景：为什么前列腺 DWI 特别难处理

DWI 是 bi-parametric 前列腺 MRI 的核心序列：癌变区域水分子扩散受限，ADC 值偏低，PI-RADS v2.1 评分正是依赖这一特性识别高危病灶。

但 DWI 有三个系统性缺陷：

- **信噪比低**：高 b 值（≥1000 s/mm²）信号极弱，热噪声占主导
- **Gibbs 振铃**：k 空间截断在组织边界产生条纹伪影，直接污染 ADC 估计
- **磁化率失真**：直肠气体导致磁场不均匀，DWI 与 T2w 出现几何形变，两者解剖不对齐

脑科学社区早有 MRtrix3、FSL 这套成熟工具链，但前列腺成像社区长期没有标准化流程。基于 fastMRI 前列腺数据集（268 例）的最新研究系统量化了每个步骤的影响，结论很清晰：**完整预处理流水线对高危 PI-RADS 分类的假阴性率有显著改善**。

## ADC 估计的数学基础

标准单指数模型：

$$
S(b) = S_0 \cdot e^{-b \cdot \text{ADC}}
$$

对数线性化后变成普通线性方程组，$[\ln S_0,\ \text{ADC}]$ 为待求参数。

### 线性最小二乘（LLS）

```python
import numpy as np

def compute_adc_lls(signals, b_values):
    """
    LLS 估计 ADC
    signals: (n_b, n_voxels) — 各 b 值下的信号强度
    b_values: (n_b,) — 单位 s/mm²，别用 ms/mm²（常见 bug）
    returns: adc (n_voxels,), s0 (n_voxels,)
    """
    b = np.array(b_values, dtype=float)
    log_S = np.log(np.clip(signals, 1e-6, None))  # clip 防 log(0)

    # 设计矩阵 [1, -b]，对应参数 [log_S0, ADC]
    A = np.column_stack([np.ones_like(b), -b])

    # lstsq 一次处理所有体素列
    result, _, _, _ = np.linalg.lstsq(A, log_S, rcond=None)  # (2, n_voxels)

    return result[1], np.exp(result[0])  # adc, s0
```

### 迭代加权最小二乘（IWLLS）

LLS 假设对数域噪声是高斯的，但 DWI 真实噪声在信号域服从 Rician 分布。IWLLS 用预测信号的平方作为权重来修正：

$$
w_i^{(t)} = \hat{S}_i^{(t-1)^2}
$$

```python
def compute_adc_iwlls(signals, b_values, n_iter=5):
    """IWLLS：迭代加权，对低 SNR 体素理论更准"""
    b = np.array(b_values, dtype=float)
    adc, s0 = compute_adc_lls(signals, b_values)  # LLS 热启动

    A = np.column_stack([np.ones_like(b), -b])
    log_S = np.log(np.clip(signals, 1e-6, None))

    for _ in range(n_iter):
        S_hat = s0[None, :] * np.exp(-b[:, None] * adc[None, :])
        w = np.clip(S_hat, 1e-8, None) ** 2  # Rician 最优权重

        # 逐体素 2×2 加权最小二乘（生产环境用 DIPY 向量化实现）
        for v in range(signals.shape[1]):
            W = np.diag(w[:, v])
            AtWA = A.T @ W @ A
            AtWb = A.T @ W @ log_S[:, v]
            r = np.linalg.solve(AtWA, AtWb)
            adc[v], s0[v] = r[1], np.exp(r[0])

    return adc, s0
```

**关键结论**：论文报告两种方法在充分预处理后数值几乎等价（PCC ~0.99）。别在估计方法上钻牛角尖，先把预处理做对。

## 三步预处理流水线

```
原始 DWI → [1] MP-PCA 去噪 → [2] Gibbs 校正 → [3] 磁化率失真校正 → ADC 估计
```

顺序不能乱：去噪必须在 Gibbs 校正前，否则会把振铃当信号保留。

### 第一步：MP-PCA 去噪

DWI 信号本质上是低秩的（梯度方向相关），MP-PCA 用 Marchenko-Pastur 分布确定噪声阈值并截断奇异值：

```python
def mp_pca_denoise(patch):
    """
    patch: (n_voxels_in_patch, n_gradients)
    生产环境直接用：from dipy.denoise.localpca import mppca
    """
    m, n = patch.shape
    U, S, Vt = np.linalg.svd(patch, full_matrices=False)

    # MP 分布上界确定噪声阈值
    beta = min(m, n) / max(m, n)
    lambda_plus = (1 + np.sqrt(beta)) ** 2
    sigma = np.median(S) / np.sqrt(lambda_plus * max(m, n))
    threshold = sigma * np.sqrt(lambda_plus * max(m, n))

    S_clean = S * (S > threshold)
    return (U * S_clean) @ Vt
```

### 第二步：Gibbs 振铃校正

Gibbs 伪影在组织边界产生条带，在前列腺包膜处尤其明显，会系统性地偏移边缘体素的 ADC 值。子体素位移法通过平均不同位移版本消除振铃：

```python
from scipy.ndimage import shift as nd_shift

def gibbs_correction_subvoxel(img_slice, n_shifts=4):
    """
    局部子体素位移平均（简化版）
    完整实现：from dipy.denoise.gibbs import gibbs_removal
    """
    shifts = np.linspace(0, 1, n_shifts, endpoint=False)
    result = np.zeros_like(img_slice, dtype=float)
    for s in shifts:
        result += nd_shift(img_slice.astype(float), [0, s], mode='wrap')
    return result / n_shifts
```

### 第三步：磁化率失真校正（SDC）

这是影响最大的步骤——论文报告 SDC 前后 ADC PCC 从 0.99 降至 0.90，**但这不是变差了**，而是把 DWI 的解剖位置校准到真实位置，改变了空间分布。

策略：将 DWI b=0 配准到 T2w，把相同形变场应用到所有 b 值体积：

```python
import SimpleITK as sitk

def get_dwi_to_t2w_transform(dwi_b0_path, t2w_path):
    """配准 DWI b0 → T2w，返回可重用的 transform"""
    fixed  = sitk.ReadImage(t2w_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(dwi_b0_path, sitk.sitkFloat32)

    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(50)
    demons.SetStandardDeviations(1.5)
    disp_field = demons.Execute(fixed, moving)

    return sitk.DisplacementFieldTransform(disp_field)

def apply_transform_to_volume(volume_path, transform, reference_path):
    """将同一 transform 应用到所有 DWI 体积"""
    volume = sitk.ReadImage(volume_path, sitk.sitkFloat32)
    reference = sitk.ReadImage(reference_path, sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(volume)
```

## PI-RADS 自动分类器

论文用 DenseNet 做 3 类分类（低危/中危/高危），输入是三通道 MRI（T2w + ADC + b1000 DWI）。

### 网络结构

```python
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, growth_rate, 3, padding=1, bias=False)
        )
    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)

class PIRADSClassifier(nn.Module):
    """
    输入: (B, 3, H, W) — T2w + ADC + b1000 三通道
    输出: (B, 3) — PI-RADS 低/中/高危 logits
    """
    def __init__(self, growth_rate=32):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)

        self.block1 = nn.Sequential(*[DenseLayer(64 + i*growth_rate, growth_rate) for i in range(6)])
        ch1 = 64 + 6 * growth_rate

        self.trans = nn.Sequential(
            nn.BatchNorm2d(ch1), nn.ReLU(),
            nn.Conv2d(ch1, ch1 // 2, 1), nn.AvgPool2d(2)
        )
        self.block2 = nn.Sequential(*[DenseLayer(ch1//2 + i*growth_rate, growth_rate) for i in range(6)])
        ch2 = ch1 // 2 + 6 * growth_rate

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch2, 3)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans(x)
        x = self.block2(x)
        return self.head(x)
```

### 训练：类别不平衡是最大坑

PI-RADS 高危病例天然稀少，必须处理类别不平衡，否则模型直接预测"全部低危"也能得到不错的 accuracy：

```python
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, growth_rate, 3, padding=1, bias=False)
        )
    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)  # 密集连接：拼接输入与输出

class PIRADSClassifier(nn.Module):
    """输入: (B, 3, H, W) — T2w + ADC + b1000 三通道 → 输出: (B, 3) — PI-RADS 低/中/高危 logits"""
    def __init__(self, growth_rate=32):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.block1 = nn.Sequential(*[DenseLayer(64 + i*growth_rate, growth_rate) for i in range(6)])
        ch1 = 64 + 6 * growth_rate
        self.trans = nn.Sequential(nn.BatchNorm2d(ch1), nn.ReLU(),
                                   nn.Conv2d(ch1, ch1//2, 1), nn.AvgPool2d(2))  # 过渡层：压缩通道+下采样
        self.block2 = nn.Sequential(*[DenseLayer(ch1//2 + i*growth_rate, growth_rate) for i in range(6)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch1//2 + 6*growth_rate, 3))

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans(x)   # 两段 DenseBlock，中间 transition
        x = self.block2(x)
        return self.head(x)  # 全局池化 → 三分类
```

## 实验结果解读

| 预处理阶段 | ADC 与基准 PCC | 高危 AUROC | 高危假阴性置信度 |
|-----------|--------------|-----------|---------------|
| 无预处理（原始）| —（基准）| 最低 | 最高（最危险）|
| + 去噪 + Gibbs | ~0.99 | 中 | 中 |
| + SDC（完整流水线）| ~0.90* | **最高** | **最低（最安全）** |

`*` SDC 后 PCC 降低是因为空间位置改变，不是质量变差。

**最值得关注的指标是假阴性分析**：完整预处理流水线在高危病例上的错误预测置信度最低（less overconfident）。一个把 PI-RADS 5 误判为低危但置信度 95% 的模型，比一个置信度 55% 的模型危险得多——后者至少会触发人工复查。

## 调试指南

### ADC 图看起来不对

1. **全图 ADC 系统偏高**：b 值单位错误，确认用 s/mm²，不是 ms/mm²
2. **边缘出现负 ADC**：Gibbs 伪影压低了边缘信号，先做 Gibbs 校正再估计 ADC
3. **DWI 和 T2w 解剖错位**：跳过了 SDC，或配准参数不合适（前列腺动度大，需要形变配准，刚体不够）

### 分类器训练异常

1. **验证 AUROC 大幅波动**：268 例数据集偏小，必须用分层 5 折交叉验证，单次 split 结论不可信
2. **高危召回率极低**：类别权重不够，或没用 focal loss；检查每个 batch 的类别分布
3. **训练收敛但泛化差**：检查多通道输入的归一化——ADC 和 T2w 数值范围差异悬殊，分别做 z-score

### 如何判断预处理真的有帮助

别只看最终 AUROC，做消融实验：

```python
# 关键诊断指标
def evaluate_preprocessing(adc_before, adc_after, mask):
    """比较预处理前后的 ADC 分布变化"""
    roi_before = adc_before[mask > 0]
    roi_after  = adc_after[mask > 0]
    print(f"ADC 均值变化: {roi_before.mean():.4f} → {roi_after.mean():.4f}")
    print(f"ADC 标准差变化: {roi_before.std():.4f} → {roi_after.std():.4f}")
    # 去噪后 std 应减小；SDC 后均值可能改变（解剖位置变了）

    from scipy.stats import pearsonr
    pcc, _ = pearsonr(roi_before, roi_after)
    print(f"PCC: {pcc:.4f}  （SDC 后约 0.90，其余步骤约 0.99）")
```

## 什么时候必须用完整流水线？

| 场景 | 建议 |
|-----|------|
| 多中心研究 | 必须完整流水线，中心间磁场差异会系统性影响 ADC |
| 临床 AI 部署 | SDC 必须，否则模型在不同采集方向/机器上泛化很差 |
| 单中心回顾性研究 | 至少做 Gibbs 校正 + SDC |
| 纯定性阅片辅助 | 可简化，但要在 limitation 里说清楚 |

不适用：如果下游任务依赖 DWI 的原始几何形状（某些粒子植入手术导航），SDC 引入的形变场可能带来负效应。

## 我的观点

这个工作的价值不在于提出新算法，而在于**把脑科学社区已知的东西系统化地移植到前列腺，并用数字说话**。医学 AI 社区有个根深蒂固的坏习惯：在未预处理的原始数据上训练深度学习模型，在同分布测试集上报告漂亮的 AUROC，却从不问"模型到底学到了什么"。

这篇论文的假阴性置信度分析给出了一个更诚实的视角：预处理后模型在犯错时更"不确定"，这恰恰是临床部署最需要的特性——不确定的模型触发人工复查，过度自信的模型把漏诊变成医疗事故。

如果你在做前列腺 MRI 的工作：
- 用 MRtrix3 (`dwidenoise` + `mrdegibbs`) 或 DIPY 搭预处理流水线，别自己手写
- 在有无 SDC 的数据上分别训练，对比两者在跨中心数据上的泛化能力
- 评估高危类别的假阴性置信度，不要只看 AUROC