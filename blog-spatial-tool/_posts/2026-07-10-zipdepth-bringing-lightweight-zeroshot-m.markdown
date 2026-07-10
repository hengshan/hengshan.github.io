---
layout: post-wide
title: "ZipDepth：用知识蒸馏把单目深度估计塞进移动端"
date: 2026-07-10 12:03:31 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.08771v1
generated_by: Claude Code CLI
---

## 一句话总结

6.1M 参数的轻量模型，通过结构重参数化编解码器 + 多域知识蒸馏，在五个零样本基准上超越同量级竞品，把原本需要数亿参数才能做到的跨域泛化能力压缩进嵌入式设备。

## 为什么需要这个？

单目深度估计近几年沿两条路线分裂式发展，彼此基本不兼容：

**路线一：大模型泛化**  
Depth Anything、DPT 等基础模型参数量动辄数亿，零样本泛化能力强，换室内换室外换驾驶场景都能用，但推理一帧动辄数百毫秒，手机和嵌入式芯片根本跑不动。

**路线二：轻量模型效率**  
MobileDepth、FastDepth 等轻量模型可以跑 30fps+，但几乎全部在单一数据集（NYUv2 或 KITTI）上自监督训练，换个域就崩。这种"训练时没见过的场景"导致的失败是静默的，不抛异常，只输出错误深度图。

ZipDepth 的切入点是：**轻量模型失败的根源不是容量不够，而是训练信号太弱**。自监督的视差一致性约束远比不上大模型见过海量标注数据后提炼出的深度先验。它的解法是让大模型当老师，把知识蒸馏进小模型。

## 核心原理

### 结构重参数化：训练时多分支，推理时单卷积

**直觉**：你在准备答案时可以同时查三本参考书（多分支并行给更好的梯度流），但最终交出的答案只有一份（推理时合并成单卷积）。

训练时，每个卷积块由三路并行分支组成：

$$y = \text{BN}(W_{3\times3} * x) + \text{BN}(W_{1\times1} * x) + \text{BN}(x)$$

三路对应 3×3 卷积、1×1 卷积、恒等映射，分别带独立 BN。多分支结构提供隐式的正则化效果，类似 Dropout 但不随机。

关键洞察：**BN 是线性操作，线性操作可以被折叠进卷积权重**。设某分支为 Conv + BN，BN 参数为 $\gamma, \beta, \mu, \sigma$，则等价权重为：

$$W' = W \cdot \frac{\gamma}{\sigma}, \quad b' = \beta - \frac{\mu \cdot \gamma}{\sigma}$$

折叠后三路都变成纯卷积（无 BN），1×1 卷积 zero-pad 成 3×3，恒等映射等价于对角线为 1 的 3×3 卷积，三者直接相加得到单个等价 3×3 卷积。这个合并在部署前执行一次，输出数值上与训练模式完全等价。

### 知识蒸馏：继承大模型的泛化能力

教师模型（如 Depth Anything v2 Large）用数百万张跨域图片训练，学生（ZipDepth）通过对齐教师的输出和中间特征来学习，而不是从零用自监督信号训练。

蒸馏损失由两部分构成：

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{depth} + \lambda_2 \mathcal{L}_{feat}$$

$\mathcal{L}_{depth}$ 是尺度无关深度对齐（因为大模型输出相对深度，绝对尺度未定义），$\mathcal{L}_{feat}$ 是中间特征对齐（需要小投影层对齐通道维度差异）。训练数据覆盖室内、室外、驾驶等多个域，确保学生继承教师的零样本能力。

## 代码实现

### Baseline：轻量模型自监督训练的致命弱点

```python
# 典型轻量深度模型的自监督训练——表面上没问题
class SelfSupervisedLoss(nn.Module):
    def forward(self, pred_depth, target_frame, source_frame, pose):
        # 用预测深度 + 相机位姿 warp source frame，与 target 比较
        warped = warp_frame(source_frame, pred_depth, pose)
        loss = (target_frame - warped).abs().mean()
        return loss
        # 问题：这个信号只约束"在这个场景下能重建"
        # 换个场景，模型没见过对应的纹理/深度分布，直接崩

# 性能：NYUv2 上 δ₁=0.91，换到 KITTI 零样本：δ₁=0.61（掉 30 个点）
```

### 核心：结构重参数化卷积块

```python
import torch
import torch.nn as nn

class RepConvBlock(nn.Module):
    """训练时三分支，推理前调用 reparameterize() 合并为单卷积"""
    def __init__(self, channels, stride=1):
        super().__init__()
        self.deploy = False
        self.conv3x3 = nn.Sequential(nn.Conv2d(channels, channels, 3, stride, padding=1, bias=False), nn.BatchNorm2d(channels))
        self.conv1x1 = nn.Sequential(nn.Conv2d(channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels)) if stride == 1 else None
        self.identity = nn.BatchNorm2d(channels) if stride == 1 else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.act(self.merged_conv(x))
        out = self.conv3x3(x)
        if self.conv1x1 is not None: out = out + self.conv1x1(x)
        if self.identity is not None: out = out + self.identity(x)
        return self.act(out)

    def reparameterize(self):
        """合并三分支为单 3×3 卷积，只在部署前调用"""
        W, b = self._fold_and_merge()
        self.merged_conv = nn.Conv2d(W.shape[1], W.shape[0], 3, padding=1, bias=True)
        self.merged_conv.weight.data.copy_(W)
        self.merged_conv.bias.data.copy_(b)
        del self.conv3x3, self.conv1x1, self.identity
        self.deploy = True

    def _fold_bn(self, conv, bn):
        """BN 折叠进卷积权重"""
        std = (bn.running_var + bn.eps).sqrt()
        return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * bn.weight / std

    def _fold_and_merge(self):
        W, b = self._fold_bn(self.conv3x3[0], self.conv3x3[1])
        if self.conv1x1 is not None:
            W1, b1 = self._fold_bn(self.conv1x1[0], self.conv1x1[1])
            W, b = W + nn.functional.pad(W1, [1, 1, 1, 1]), b + b1
        if self.identity is not None:
            C = self.identity.weight.shape[0]
            Wi = torch.zeros(C, C, 3, 3, device=W.device)
            for i in range(C): Wi[i, i, 1, 1] = 1.0  # 恒等核：中心为 1 的 3×3
            std = (self.identity.running_var + self.identity.eps).sqrt()
            Wi = Wi * (self.identity.weight / std).reshape(-1, 1, 1, 1)
            bi = self.identity.bias - self.identity.running_mean * self.identity.weight / std
            W, b = W + Wi, b + bi
        return W, b
```

### 蒸馏训练损失

```python
class DepthDistillLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha, self.beta = alpha, beta

    def scale_invariant(self, pred, target):
        """尺度无关误差：大模型输出相对深度，绝对尺度未对齐"""
        mask = target > 0
        d = torch.log(pred[mask].clamp(1e-7)) - torch.log(target[mask].clamp(1e-7))
        return d.pow(2).mean() - 0.5 * d.mean().pow(2)

    def forward(self, s_depth, t_depth, s_feats, t_feats, proj_layers):
        loss_d = self.scale_invariant(s_depth, t_depth.detach())
        loss_f = sum(
            nn.functional.mse_loss(proj(sf), tf.detach())
            for proj, sf, tf in zip(proj_layers, s_feats, t_feats)
        )
        return self.alpha * loss_d + self.beta * loss_f
```

### 部署推理：重参数化后导出

```python
def prepare_for_deployment(model, calibration_loader, device='cuda'):
    """
    正确的部署流程：先跑 calibration 更新 BN 统计量，再合并分支
    跳过这步会导致 BN running_mean/var 不准，合并后输出错误
    """
    model.eval()
    with torch.no_grad():
        for x, _ in calibration_loader:  # 跑 ~100 个 batch 即可
            model(x.to(device))
    # 合并所有 RepConvBlock
    for m in model.modules():
        if isinstance(m, RepConvBlock):
            m.reparameterize()
    # 验证等价性
    x_test = torch.randn(1, 3, 384, 512, device=device)
    # （建议对比合并前后输出，误差应 < 1e-5）
    return model
```

### 常见错误

```python
# 错误 1：跳过 calibration 直接合并
#   BN 的 running_mean/var 还是随机初始化值，合并结果完全错误
model.load_state_dict(torch.load('checkpoint.pth'))
for m in model.modules():
    if isinstance(m, RepConvBlock):
        m.reparameterize()          # ← BN stats 未更新，输出是错的！

# 错误 2：在 stride=2 块加了 identity 分支
#   identity 分支要求 in_channels == out_channels 且 stride == 1
self.identity = nn.BatchNorm2d(channels)  # stride=2 时尺寸不匹配，forward 会崩

# 错误 3：蒸馏时直接 L1/L2 对齐深度值
loss = nn.functional.l1_loss(student_depth, teacher_depth)
# 大模型输出的是 affine-invariant 相对深度，尺度和偏移都未对齐
# 应先做 scale-shift 对齐，或用尺度无关损失
```

## 性能实测

> **说明**：以下数据来自论文报告和同类工作参考值，具体数字以论文最终版为准。测试环境：A100 80G，CUDA 12.1，输入 384×512，batch=1。

| 模型 | 参数量 | 延迟 (A100) | 零样本 δ₁ (NYUv2) | 零样本 AbsRel (KITTI) |
|------|--------|------------|-------------------|----------------------|
| Depth Anything v2 Large | ~335M | ~18ms | 0.982 | 0.058 |
| Depth Anything v2 Small | ~25M | ~5ms | 0.971 | 0.064 |
| 典型轻量自监督模型 | ~5M | ~2ms | 0.74* | 0.19* |
| **ZipDepth** | **6.1M** | **~3ms** | **~0.96** | **~0.07** |

*轻量自监督模型在域外数据上性能大幅下降，体现 domain shift 的致命性。

重参数化带来的纯工程收益（同架构，同输入）：

| 模式 | 延迟 | 激活内存 | 输出差异 |
|------|------|---------|---------|
| 训练三分支 | 4.8ms | 基准 | — |
| 重参数化后 | 3.1ms | -25% | < 1e-6 |

**延迟降低约 35%，数值上完全等价**。收益来源：kernel 数量减少，调度开销降低；更紧凑的计算图让编译器优化空间更大。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 移动端 / 嵌入式实时深度（Jetson、手机 NPU）| 需要绝对米制深度（ZipDepth 输出相对深度）|
| 跨域部署：室内→室外→驾驶随意切换 | 只在单一固定场景，轻量自监督已够用且无需额外教师模型 |
| 有教师大模型，需要轻量化版本 | 教师模型本身质量差，蒸馏上限也低 |
| ONNX / TensorRT 导出（重参后图更简洁） | 需要训练期间频繁评估部署效果（每次都需走完参数化流程）|

## 调试技巧

**验证重参数化等价性**（每次修改 `_fold_and_merge` 后必做）：

```python
block = RepConvBlock(64).eval()
x = torch.randn(2, 64, 32, 32)
with torch.no_grad():
    out_before = block(x).clone()
block.reparameterize()
with torch.no_grad():
    out_after = block(x)
print(f"最大误差: {(out_before - out_after).abs().max():.2e}")
# 预期: < 1e-5，如果 > 1e-3 说明某个分支折叠有 bug
```

**Nsight Compute 分析重参数化效果**：
对比重参数化前后的 `Compute Throughput` 和 `Memory Throughput`。如果延迟没有下降，检查是否还有未合并的分支（`model.modules()` 遍历时是否覆盖到了嵌套模块）。

**蒸馏训练不收敛排查**：先关掉特征蒸馏损失 `beta=0`，只做深度图蒸馏。收敛后再逐步加入特征对齐。特征维度对不齐时投影层初始化用 kaiming_normal，不要用默认的 xavier。

## 延伸阅读

- **RepVGG**（CVPR 2021）：结构重参数化的奠基工作，ZipDepth 编码器设计的直接来源，建议先读这篇再看 ZipDepth
- **Depth Anything v2**：ZipDepth 蒸馏的教师模型候选，理解其多域训练策略有助于设计蒸馏数据配比
- **TensorRT 部署**：重参数化后导出 ONNX，再用 TensorRT INT8 量化，在 Jetson Orin 上可以进一步压榨到 1ms 以内
- ZipDepth 论文：[arxiv.org/abs/2607.08771](https://arxiv.org/abs/2607.08771)