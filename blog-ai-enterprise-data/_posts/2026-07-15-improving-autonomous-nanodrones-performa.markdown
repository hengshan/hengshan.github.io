---
layout: post-wide
title: "纳米无人机上的 CNN 部署：当算力预算只剩 1.6% 时"
date: 2026-07-15 12:03:27 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.12593v1
generated_by: Claude Code CLI
---

## 一句话总结

通过自动化量化、剪枝和部署流水线，将视觉导航 CNN 压缩后跑在一颗比指甲盖还小的芯片上——等精度前提下，内存减半、推理提速 1.6 倍，把无人机最高飞行速度从 0.5 m/s 推到 1.96 m/s。

## 为什么这篇论文重要？

很多 ML 工程师有过这种经历：训练好一个模型，准确率不错，开始部署到嵌入式设备——然后发现地狱才刚开始。

纳米无人机（sub-10 cm UAV）的情况比普通嵌入式还极端：

- 算力：一颗 PULP GAP8，8 个 RISC-V 核
- 内存：512 KB L2 SRAM，**没有外部 DRAM**
- 功耗：整机约 100mW，CNN 能用不到 1.6mW
- 实时要求：控制频率 10+ Hz

在这种约束下，`model.half()` 是远远不够的。你需要一整套工具链：量化感知训练 → 结构化剪枝 → 硬件特定内核映射 → 闭环验证。

**这篇论文真正的贡献**不是 PULP-Dronet（那是他们之前的工作），而是把这个乱糟糟的手工过程**自动化**了。

有一个现象值得单独拿出来说：优化后的模型在 benchmark 上与原始模型精度相同，但实际飞行速度提升了 4 倍。原因是推理更快了，控制频率更高，系统响应性更好。**精度不变，行为改变了**——这是只做 offline evaluation 的研究者最容易忽略的一点。

## 核心方法解析

### 整体流水线

```
原始模型 → 量化感知训练 → 结构化剪枝 → 编译器优化 → 硬件部署 → 闭环测试
```

### 1. 量化感知训练（QAT）

GAP8 支持 INT8/INT16 运算，比浮点快得多。直接截断的 Post-Training Quantization（PTQ）在小模型上精度损失通常不可接受，因此需要在训练时就模拟量化误差。

量化正向传播的核心公式：

$$x_q = \text{round}\left(\frac{x}{\Delta}\right) \cdot \Delta, \quad \Delta = \frac{x_{\max} - x_{\min}}{2^b - 1}$$

反向传播时，`round` 不可导，用 Straight-Through Estimator（STE）绕过：梯度直接通过，不经过 round 操作。

### 2. 结构化剪枝

不是随机稀疏化，而是整 filter 级别的移除。这样剪完的模型可以直接利用 SIMD 指令，不需要稀疏矩阵运算库。

用 L1-norm 评估 filter 重要性：

$$I_f = \sum_{i,j,k} \left|W_{f,i,j,k}\right|$$

重要性低的 filter 被移除，剩余部分微调恢复精度。

### 3. 自动化部署（最关键的部分）

论文使用 **NEMO + DORY** 工具链，自动完成：
- 用少量校准数据确定每层量化参数
- 生成 GAP8 特定的 C 代码
- 分析并安排 L1/L2 缓存调度，保证激活值不溢出

这把原本需要数周的手工迭代压缩到了自动化流程。

## 动手实现

### 量化感知训练核心模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeQuantize(nn.Module):
    """前向模拟量化，STE 处理反向传播"""
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        x_q = (x / self.scale + self.zero_point).clamp(qmin, qmax).round()
        x_dq = (x_q - self.zero_point) * self.scale
        return x + (x_dq - x).detach()  # STE

    def calibrate(self, x):
        with torch.no_grad():
            self.scale.data = x.abs().max() / (2 ** (self.bits - 1) - 1)
            self.zero_point.data = torch.zeros(1)


class QuantizedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bits=8, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, **kwargs)
        self.weight_quant = FakeQuantize(bits)
        self.act_quant = FakeQuantize(bits)

    def forward(self, x):
        w_q = self.weight_quant(self.conv.weight)
        out = F.conv2d(x, w_q, self.conv.bias,
                       self.conv.stride, self.conv.padding)
        return self.act_quant(out)
```

### 简化版 PULP-Dronet 架构

```python
class PULPDronet(nn.Module):
    """
    输入 200x200 灰度图，输出转向角 + 碰撞概率
    双头设计：regression + binary classification
    """
    def __init__(self, bits=8):
        super().__init__()
        self.stem = QuantizedConv2d(1, 32, 5, bits=bits, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.blocks = nn.Sequential(
            self._res_block(32, 32, bits),
            self._res_block(32, 64, bits, stride=2),
            self._res_block(64, 64, bits),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_steering = nn.Linear(64, 1)
        self.head_collision = nn.Linear(64, 1)

    def _res_block(self, in_ch, out_ch, bits, stride=1):
        return nn.Sequential(
            QuantizedConv2d(in_ch, out_ch, 3, bits=bits,
                            stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(),  # ReLU6 对量化更友好，见下文
            QuantizedConv2d(out_ch, out_ch, 3, bits=bits, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.stem(x)))
        x = self.pool(self.blocks(x)).flatten(1)
        return self.head_steering(x), torch.sigmoid(self.head_collision(x))
```

### 嵌入式部署前的模型剖析

```python
def profile_for_embedded(model, input_shape=(1, 1, 200, 200)):
    from torchinfo import summary
    result = summary(model, input_size=input_shape, verbose=0)

    # GAP8 硬性约束
    L2_LIMIT_KB = 512
    param_kb = result.total_params / 1024  # INT8: 1 byte/param

    print(f"参数量:       {result.total_params:,}")
    print(f"INT8 内存:    {param_kb:.1f} KB / {L2_LIMIT_KB} KB")
    print(f"MACs:         {result.total_mult_adds:,}")

    # 留 30% 给激活值缓冲
    if param_kb > L2_LIMIT_KB * 0.7:
        print("⚠ 内存超限风险，需进一步剪枝")
    return result
```

### 实现中的坑

**坑1：BatchNorm 要在推理时 Fold 进卷积**

单独的 BN 层在嵌入式推理时是额外开销。正确做法是 inference 前把 BN 参数吸收进卷积权重：

```python
def fold_bn_into_conv(conv, bn):
    std = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / std
    conv.weight.data *= scale[:, None, None, None]
    conv.bias.data = (conv.bias.data - bn.running_mean) * scale + bn.bias
    return conv
```

**坑2：用 ReLU6 代替 ReLU**

ReLU 输出范围 $[0, +\infty)$，INT8 量化时范围无法确定，scale 会被极端值拉歪。ReLU6 截断到 $[0, 6]$，量化步长固定，精度损失更小，MobileNet 系列已将其作为标准。

**坑3：校准数据集的质量比数量重要**

PTQ 的 scale 估计依赖校准数据的分布。100-200 张有代表性的图（覆盖走廊、转弯、障碍物场景）远胜过用随机噪声校准：

```python
def calibrate_model(model, calib_loader, device='cpu'):
    model.eval()
    with torch.no_grad():
        for imgs, _ in calib_loader:
            model(imgs.to(device))  # 触发 FakeQuantize.calibrate 中的统计
    print(f"校准完成，共 {len(calib_loader.dataset)} 张图")
```

## 实验：论文说的 vs 现实

| 指标 | 原始手工版 | 自动化优化版 |
|------|-----------|------------|
| 内存占用 | 2x | 1x（减半） |
| 推理延迟 | 1.6x | 1x（提速）|
| 转向 MAE | 相同 | 相同 |
| 最高飞行速度 | 0.5 m/s | 1.96 m/s |
| 障碍制动速度 | - | 1.65 m/s |

**需要注意的地方**：

1. **论文没说清楚的泛化边界**：90 度转弯 lane following 实验的训练数据怎么采集的？赛道环境能泛化到多宽？对运动模糊、光照变化的鲁棒性没有系统测试。
2. **可复现性门槛**：NEMO + DORY 工具链是开源的（ETH 维护），但依赖 GAP8 硬件。如果你只有 Jetson Nano 或树莓派，这套流程**不能直接迁移**，需要替换底层代码生成器。
3. **飞行速度提升的真正原因**：是推理频率提高带来的控制增益，不是模型本身更"聪明"了。离线精度相同，在线表现迥异——这正是论文最值得学习的系统视角。

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 目标硬件是 PULP/GAP8 系列 | 有充足算力（Jetson、手机 NPU） |
| 功耗是硬约束（<10mW 量级） | 任务需要复杂推理或记忆 |
| 任务可以用端到端 CNN 建模 | 需要强泛化到未见环境 |
| 有能力做闭环飞行测试 | 只做 offline benchmark 验证 |

## 我的观点

**这篇论文的价值被摘要低估了。**

表面上是"又一个无人机模型压缩"，但它展示了一个更普适的结论：**端到端自动化工具链的成熟度，直接决定 edge AI 的应用天花板**，不只是 benchmark 分数。

对做嵌入式 AI 的团队，这个教训在任何领域都成立：手工优化和自动化优化的模型，offline 指标可能一模一样，但 online 行为可能截然不同。闭环测试不是锦上添花，是必须项。

**这个方向的限制**：PULP-Dronet 本质上是反应式控制——看到什么做什么，没有记忆，没有规划。在复杂动态环境中迟早撞墙。下一代方向很可能是将这类高效感知模块与轻量状态空间模型结合，赋予纳米无人机短时序记忆能力。

1.96 m/s 在室内赛道听起来不错，但室外真实导航中，风扰、GPS 漂移和电池续航才是真正的天花板——那不是 CNN 能解决的问题。