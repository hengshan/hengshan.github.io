---
layout: post-wide
title: '分割模型的频谱脆弱性：当你的模型在特征层"依赖高频"'
date: 2026-09-01 08:04:00 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.29167v1
generated_by: Claude Code CLI
---

## 一句话总结

对分割模型的**内部特征图**做低通滤波，发现不同架构的脆弱层位置截然不同，数据集差异比架构差异更显著——而且输入域的 Fourier 增广完全修不好这个问题。

## 背景：我们一直在测错地方

传统鲁棒性评估的做法：给输入加噪声、做高斯模糊、对抗扰动，然后看 Dice 有没有下降。这听起来合理，但有一个根本性盲点：

**输入层鲁棒性 ≠ 特征层鲁棒性。**

考虑这个场景：你在结肠镜息肉数据集（CVC-ClinicDB）上训练了一个 ResNet50-UNet，测试 Dice 达到 0.90，加了 Fourier 增广，跑了对抗攻击，结论是"模型很鲁棒"。

然而，如果直接在模型内部某个中间层的特征图上做低通滤波（保留低频、扔掉高频细节），Dice 立刻掉到接近 0——**下降 100%**。

同样的操作放到皮肤病变数据集（ISIC2018）上？Dice 只下降 9.4%。

同一架构，同样的滤波，完全不同的结果。这就是**特征域频谱脆弱性（Feature-Spectral Fragility）**的核心问题。

### 为什么值得关心？

- **部署风险被低估**：如果模型必须依赖高频特征才能工作，任何导致高频丢失的因素（图像压缩、不同内镜设备、下采样后处理）都可能让性能崩溃，而你在标准 benchmark 上根本看不出来。
- **蒸馏和剪枝的隐患**：不同架构的脆弱层位置不同，在错误的层做特征对齐可能破坏鲁棒性。
- **Fourier 增广不是银弹**：输入域的频谱增广提升了对输入扰动的鲁棒性，但对特征域脆弱性几乎没有帮助。

## 方法：特征域低通滤波探针

### 直觉解释

把每个中间层特征图 $F \in \mathbb{R}^{B \times C \times H \times W}$ 看成一组二维信号。对其做 2D 傅里叶变换后，低频成分对应全局轮廓，高频成分对应边缘细节。

**探针逻辑**：在推理时，对某一层的输出做 2D FFT，只保留频率半径在 $\rho \cdot \rho_{max}$ 以内的成分，然后用滤波后的特征继续前向传播。如果 Dice 崩了，说明这一层高度依赖高频信息。

### 数学基础

对特征图 $F$ 做 2D FFT，得到 $\hat{F} = \mathcal{F}(F)$，定义圆形低通掩码：

$$
M_\rho(u, v) = \begin{cases}
1 & \text{if } \sqrt{(u - u_0)^2 + (v - v_0)^2} \leq \rho \cdot \rho_{max} \\
0 & \text{otherwise}
\end{cases}
$$

其中 $(u_0, v_0)$ 是频域中心，$\rho_{max} = \min(H, W) / 2$。滤波后重建：

$$
F_{\text{filtered}} = \mathcal{F}^{-1}(\hat{F} \cdot M_\rho)
$$

$\rho = 0.25$ 表示只保留最低 25% 频率半径内的成分，是相当激进的滤波。

## 实现

### 核心低通滤波器

```python
import torch
import torch.fft as fft

def low_pass_filter_2d(feature_map: torch.Tensor, cutoff_ratio: float = 0.25) -> torch.Tensor:
    """对特征图做 2D 圆形低通滤波"""
    B, C, H, W = feature_map.shape

    # 对空间维度做 FFT 并移到中心
    f_shifted = fft.fftshift(fft.fft2(feature_map, dim=(-2, -1)), dim=(-2, -1))

    # 构造圆形低通掩码
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, device=feature_map.device, dtype=torch.float32),
        torch.arange(W, device=feature_map.device, dtype=torch.float32),
        indexing='ij'
    )
    radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (radius <= cutoff_ratio * min(H, W) / 2).float()  # [H, W]
    mask = mask.unsqueeze(0).unsqueeze(0)  # 广播到 [1, 1, H, W]

    # 滤波并逆变换
    f_filtered = f_shifted * mask
    filtered = fft.ifft2(fft.ifftshift(f_filtered, dim=(-2, -1)), dim=(-2, -1)).real
    return filtered
```

### 探针：Hook 注入

用 PyTorch 的 `register_forward_hook` 拦截中间层的输出并替换为滤波后的版本。注意：每次探针完**必须**移除 hook，否则下一层探针会叠加在上一层之上。

```python
class SpectralProbe:
    """对指定层注入低通滤波探针，支持上下文管理器确保 hook 清理"""

    def __init__(self, model: torch.nn.Module, cutoff_ratio: float = 0.25):
        self.model = model
        self.cutoff_ratio = cutoff_ratio
        self._hooks = []

    def probe_layer(self, layer_path: str):
        """layer_path 支持点路径，如 'encoder.layer3'"""
        target = self.model
        for attr in layer_path.split('.'):
            target = getattr(target, attr)

        def hook_fn(module, input, output):
            # 只处理 4D 特征图，Transformer 可能返回 tuple
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                return low_pass_filter_2d(output, self.cutoff_ratio)
            return output

        self._hooks.append(target.register_forward_hook(hook_fn))
        return self

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self): return self
    def __exit__(self, *args): self.remove_hooks()
```

### 逐层敏感度扫描

```python
import numpy as np

def evaluate_dice(model, dataloader, device='cuda'):
    model.eval()
    scores = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.long().to(device)
            preds = (torch.sigmoid(model(images)) > 0.5).long()
            # Dice per batch: 2*TP / (2*TP + FP + FN)
            tp = (preds * masks).sum(dim=(1,2,3))
            denom = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
            scores.extend((2 * tp / denom.clamp(min=1)).tolist())
    return np.mean(scores)


def scan_layer_sensitivity(model, dataloader, layer_names, cutoff_ratio=0.25, device='cuda'):
    baseline = evaluate_dice(model, dataloader, device)
    print(f"Baseline Dice: {baseline:.4f}")

    results = {}
    for name in layer_names:
        with SpectralProbe(model, cutoff_ratio).probe_layer(name):
            probed = evaluate_dice(model, dataloader, device)
        drop = baseline - probed
        results[name] = drop
        print(f"  {name}: drop = {drop:.4f} ({drop/baseline*100:.1f}%)")
    return results
```

### 可视化

```python
import matplotlib.pyplot as plt

def plot_sensitivity(results: dict, title: str = "Layer-wise Spectral Fragility"):
    layers = list(results.keys())
    drops = [results[l] for l in layers]
    max_idx = int(np.argmax(drops))

    fig, ax = plt.subplots(figsize=(max(8, len(layers)*0.6), 4))
    colors = ['crimson' if i == max_idx else 'steelblue' for i in range(len(layers))]
    ax.bar(range(len(layers)), drops, color=colors, alpha=0.85)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Dice Drop (↑ 越高越脆弱)')
    ax.set_title(f"{title}\n红色 = 最脆弱层: {layers[max_idx]}")
    plt.tight_layout()
    plt.savefig('spectral_fragility.png', dpi=150)
    plt.show()
```

## 关键发现

### 发现 1：数据集差异是主效应

截止比例 $\rho = 0.25$ 时的 Dice 下降：

| 架构 | CVC-ClinicDB | ISIC2018 |
|------|-------------|---------|
| ResNet50-UNet (CNN) | **100%** | 9.4% |
| VM-UNet (SSM) | 73.2% | 10.3% |
| Swin-UNETR (Transformer) | 30.9% | 0.6% |

结肠镜数据比皮肤镜数据脆弱得多，且差异在三种架构上都统计显著。**可能原因**：息肉的边界依赖高频纹理对比，而皮肤病变有更明显的颜色/形状信息，低频成分已经够做区分了。

### 发现 2：脆弱层位置是架构特异的

- **CNN（ResNet50-UNet）**：中后期编码器最脆弱——特征图较小，但对高频细节的依赖在此时达到峰值。
- **SSM（VM-UNet）**：早期编码器就出问题——说明 SSM 从第一阶段就强依赖高频全局上下文。

这对**知识蒸馏**有直接影响：在 CNN 的浅层和 SSM 的浅层做特征对齐，风险程度截然不同。

### 发现 3：Fourier 增广无法解决特征域脆弱性

用输入域的频谱增广（如 FDA）训练后，对输入扰动的鲁棒性确实提升，但对特征域低通滤波的抵抗力**几乎没有改善**。两者是相对独立的问题。

## 调试指南

### 问题 1：所有层探针后 Dice 都接近 0

**原因**：多个 hook 叠加，或 hook 在 batch 间没有清除。

**修复**：务必用上下文管理器（见 `SpectralProbe.__enter__/__exit__`），不要手动管理 hook 生命周期。

### 问题 2：Transformer 架构 hook 无效

Swin 等 Transformer 的 block 输出可能是 `(tensor, mask)` 的 tuple。当前 `hook_fn` 已处理这种情况（检查 `isinstance` + `ndim`），但如果你发现滤波没有生效，打印 `type(output)` 确认：

```python
def hook_fn(module, input, output):
    if isinstance(output, tuple):
        # 对 tuple 中的第一个 tensor 做滤波
        filtered = low_pass_filter_2d(output[0], self.cutoff_ratio)
        return (filtered,) + output[1:]
    if isinstance(output, torch.Tensor) and output.ndim == 4:
        return low_pass_filter_2d(output, self.cutoff_ratio)
    return output
```

### 问题 3：结果在 train/eval 模式下差异很大

BatchNorm 在 `train()` 模式使用 batch 统计，会部分"消化"滤波引入的均值漂移，导致探针效果被低估。探针实验应统一在 `model.eval()` 下进行。

### 截止比例 $\rho$ 怎么选？

| $\rho$ | 含义 | 建议用途 |
|--------|------|---------|
| 0.10 | 极激进，只保留最粗轮廓 | 定位最脆弱层 |
| 0.25 | 标准设置（论文采用） | 与论文对比 |
| 0.50 | 温和，保留中频 | 建立脆弱性梯度 |
| 0.75 | 接近原始信号 | 健全性检查基线 |

## 适用场景

| 适用 | 不适用 |
|------|--------|
| 医疗影像跨设备部署前评估 | 替代标准数据增广 |
| 决定在哪一层做蒸馏/特征对齐 | 作为训练损失（FFT 断计算图） |
| 诊断跨域泛化差的根因 | 大规模超参搜索（速度慢） |

## 我的观点

这项工作最有价值的地方不是某个新算法，而是**把一个诊断工具方法化**，并在足够多的架构和数据集上系统验证了。

**数据集依赖性比架构差异更大**这个结论值得认真对待。当你看到"Transformer 在鲁棒性上优于 CNN"的论文结论时，要问一句：这是架构的性质，还是数据集的性质？

**Fourier 增广无效这个发现是真正的实践价值所在**。很多团队用 FDA/AugMax 提升泛化，然后声称模型在频谱扰动下鲁棒——但这只在输入域成立。如果你的部署场景引入了特征域的频谱偏移（比如跨医院、跨采集协议），输入增广解决不了问题，你需要直接在特征层做干预。

**局限性要说清楚**：这是一个诊断工具，不是修复工具。论文没有给出如何提升特征域鲁棒性的有效方法，这仍是开放问题。一个可能的方向是在训练时对中间层特征随机做轻度低通滤波，但这还缺乏系统验证。

如果你在做医疗影像分割且需要跨设备部署，在上线前跑一次逐层敏感度扫描，会让你对模型的信心有更准确的估计——或者及时发现需要重新设计的地方。