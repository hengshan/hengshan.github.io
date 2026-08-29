---
layout: post-wide
title: "InSAR 差分相位优化：用深度展开网络解决稀疏 SAR 重建的真正目标"
date: 2026-08-29 12:03:30 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.26605v1
generated_by: Claude Code CLI
---

## 一句话总结

DP-JMRNet 通过联合双时相重建和相干性感知门控，将稀疏 SAR 重建的优化目标从幅度精度转向差分相位精度，在 50% 采样率下将差分相位 RMSE 降低 51%。

## 为什么相位比幅度更重要？

大多数人看 SAR 图像时只关注幅度（amplitude）——建筑物、道路、植被的空间分布。但在 InSAR（干涉合成孔径雷达）应用中，**相位差才是金矿**。

地面下沉 1 毫米，卫星雷达能看到吗？能。InSAR 通过比较同一地区两次飞越时的雷达相位差，可以测量到毫米级的地表形变，这是地震变形监测、地面沉降预警的核心技术。

但这里有个被忽视已久的问题：**现有的稀疏 SAR 重建方法都在优化幅度精度，而不是相位精度。**

当你只有 30-50% 的采样数据，重建算法会最小化幅度误差。这对可视化没问题，但差分相位误差可能被放大到无法接受的程度。DP-JMRNet 直接把差分相位精度作为训练目标，效果提升 47-51%。

这个想法听起来简单，实现起来有三个非平凡的挑战。

## 核心方法解析

### 问题建模：从单时相到双时相

标准稀疏 SAR 重建：给定子采样观测 $y = \Phi x$，恢复完整复数图像 $x$。

双时相 InSAR 问题同时处理两次过境：

$$y_1 = \Phi_1 x_1, \quad y_2 = \Phi_2 x_2$$

差分相位定义为：

$$\Delta\phi = \angle(x_1 \cdot x_2^*)$$

它与地表形变 $d$ 的关系为 $\Delta\phi = \frac{4\pi}{\lambda} d \cos\theta$，其中 $\lambda$ 是雷达波长，$\theta$ 是入射角。所以差分相位误差直接转化为形变测量误差。

### 深度展开：让迭代算法可学习

深度展开（Deep Unfolding）将经典迭代优化算法"展开"成神经网络。以求解压缩感知的 ISTA 算法为例：

$$x^{(k+1)} = \text{prox}_\lambda\!\left(x^{(k)} + \frac{1}{L}\Phi^H(y - \Phi x^{(k)})\right)$$

展开 $K$ 步后，把固定的 $\lambda$ 替换成可学习参数，把近端算子替换成神经网络模块。这给了模型**可解释的归纳偏置**（每一层对应一次迭代），同时保留端到端学习的灵活性。

DP-JMRNet 把两个时相的重建迭代捆绑在一起，在每一层同时更新 $x_1^{(k)}$ 和 $x_2^{(k)}$。

### 两个关键设计

**交换等变交互模块（Exchange-Equivariant Interaction）**

两个时相没有先后之分。如果把时相 1 和 2 对调，差分相位只是符号翻转，重建质量不应改变。

数学要求：若 $f(x_1, x_2) = (x_1', x_2')$，则 $f(x_2, x_1) = (x_2', x_1')$。

实现方式：跨时相特征融合走两条权重共享的路径，天然满足等变性。

**相干性感知门控（Coherence-Aware Gate）**

这是论文最精妙的设计。相干性 $\gamma$ 衡量两次观测的相位一致性：

$$\gamma = \frac{\lvert\langle x_1 x_2^* \rangle\rvert}{\sqrt{\langle\lvert x_1\rvert^2\rangle\langle\lvert x_2\rvert^2\rangle}}$$

- 当 $\gamma \approx 1$（建筑物、农田）：两次相位差只来自形变，跨时相共享信息有益
- 当 $\gamma \approx 0$（水体、热带植被）：两次相位完全不相关，跨时相共享会污染结果

门控网络学习：何时打开跨时相信息通道，何时关闭它。

## 动手实现

### 差分相位损失函数

这是论文的核心贡献，值得先看懂再看网络结构：

```python
import torch
import torch.nn.functional as F

def differential_phase_loss(pred1: torch.Tensor, pred2: torch.Tensor,
                            target1: torch.Tensor, target2: torch.Tensor) -> torch.Tensor:
    """
    差分相位 RMSE 损失
    输入: 复数张量 shape (B, 1, H, W, 2)，最后一维为实部/虚部
    """
    def to_complex(x):
        return torch.view_as_complex(x.contiguous())

    pred_diff   = to_complex(pred1)   * torch.conj(to_complex(pred2))
    target_diff = to_complex(target1) * torch.conj(to_complex(target2))

    # 用复数乘法求相位误差，比直接 angle 相减数值更稳定
    phase_error = torch.angle(pred_diff * torch.conj(target_diff))
    return torch.sqrt(torch.mean(phase_error ** 2))
```

### 相干性感知门控

```python
import torch.nn as nn

class CoherenceAwareGate(nn.Module):
    """根据局部相干性决定跨时相特征融合的权重"""

    def __init__(self, channels: int, window_size: int = 5):
        super().__init__()
        self.window_size = window_size
        self.gate_net = nn.Sequential(
            nn.Conv2d(1, channels // 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 1), nn.Sigmoid()
        )
        # 偏置初始化为正值，让门控初始倾向于打开
        # 若初始全关，网络无法收到跨时相梯度，易陷入局部最优
        nn.init.constant_(self.gate_net[-2].bias, 2.0)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        cross  = F.avg_pool2d(feat1 * feat2,  self.window_size, 1, pad)
        power1 = F.avg_pool2d(feat1 ** 2,     self.window_size, 1, pad)
        power2 = F.avg_pool2d(feat2 ** 2,     self.window_size, 1, pad)
        coherence = (cross / (torch.sqrt(power1 * power2) + 1e-8)).abs()
        return self.gate_net(coherence)  # (B, 1, H, W) ∈ [0, 1]
```

### 完整 DP-JMRNet 骨架

```python
class UnfoldingStep(nn.Module):
    """一次深度展开迭代：梯度步 + 可学习近端算子 + 跨时相交互"""

    def __init__(self, channels: int):
        super().__init__()
        self.step_size = nn.Parameter(torch.tensor(0.1))
        self.coherence_gate = CoherenceAwareGate(channels)
        self.denoiser = nn.Sequential(          # 近端算子（去噪网络）
            nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 2, 3, padding=1)
        )
        # 权重共享保证交换等变性
        self.cross_conv = nn.Conv2d(2, 2, 1)

    def gradient_step(self, x, y, mask):
        return x - self.step_size * mask * (mask * x - y)

    def forward(self, x1, x2, y1, y2, mask1, mask2):
        x1 = self.gradient_step(x1, y1, mask1)
        x2 = self.gradient_step(x2, y2, mask2)

        gate = self.coherence_gate(x1[:, :1], x2[:, :1])

        # 同一个 cross_conv 分别作用于对方 → 交换等变
        x1 = x1 + gate * self.cross_conv(x2)
        x2 = x2 + gate * self.cross_conv(x1)

        x1 = x1 + self.denoiser(x1)
        x2 = x2 + self.denoiser(x2)
        return x1, x2


class DPJMRNet(nn.Module):

    def __init__(self, num_stages: int = 6, channels: int = 32):
        super().__init__()
        self.stages = nn.ModuleList(
            [UnfoldingStep(channels) for _ in range(num_stages)]
        )

    def forward(self, y1, y2, mask1, mask2):
        x1, x2 = mask1 * y1, mask2 * y2   # 零填充初始化
        for stage in self.stages:
            x1, x2 = stage(x1, x2, y1, y2, mask1, mask2)
        return x1, x2
```

### 训练脚本

```python
def train_step(model, optimizer, batch):
    y1, y2, mask1, mask2, gt1, gt2 = batch

    pred1, pred2 = model(y1, y2, mask1, mask2)

    loss_phase = differential_phase_loss(pred1, pred2, gt1, gt2)
    loss_amp = (F.mse_loss(pred1.norm(dim=-1), gt1.norm(dim=-1)) +
                F.mse_loss(pred2.norm(dim=-1), gt2.norm(dim=-1)))
    loss = loss_phase + 0.1 * loss_amp   # 幅度辅助损失防止相位退化

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_phase.item()
```

### 实现中的坑

**相位缠绕引发的虚假误差**

差分相位接近 $\pm\pi$ 时，直接相减会产生虚假大误差：

```python
# 错误：相位差 = 1.9π 和 -1.9π 应该接近，直接相减得到 3.8π
error = pred_phase - target_phase

# 正确：将误差折叠到 [-π, π]
error = torch.atan2(torch.sin(pred_phase - target_phase),
                    torch.cos(pred_phase - target_phase))
```

**复数梯度稳定性**

```python
# 避免：torch.angle 在 0 附近导数不稳定
phase_error = torch.angle(pred) - torch.angle(target)

# 推荐：复数除法，自动微分路径更稳定
phase_error = torch.angle(pred * torch.conj(target))
```

## 实验：论文说的 vs 现实

论文在模拟双时相 SAR 数据上，30%/40%/50% 采样率下差分相位 RMSE 较最优基线降低 47.5-51.3%，参数量只有基线的 1/3。三个 Sentinel-1 真实场景验证了相同趋势。

| 条件 | 论文结论可靠性 |
|------|--------------|
| 两时相相同孔径支撑 | 高 |
| 两时相不同孔径支撑 | 论文已证实性能大幅下降 |
| 高相干场景（城区、农田） | 高 |
| 低相干场景（热带雨林、水体） | 未充分验证 |
| 大形变（超过半波长） | 未解决 |

## 工程师最该记住的一个发现

论文有一个容易被忽视的结论：**优化随机采样掩码的形状对差分相位没有帮助，但两个时相必须使用相同的孔径支撑**。

设计 InSAR 采集策略时，花时间优化掩码形状是徒劳的。但把两次过境的有效孔径对齐，是差分相位精度的硬约束。

直觉解释：如果两次用相同的稀疏采样模式，重建引入的系统性偏差会在相减时互相抵消；如果采样模式不同，偏差就变成了噪声。

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| InSAR 地表形变监测 | 只需要幅度图像（目标检测、分类） |
| 30-50% 稀疏采样 SAR | 已有全采样数据 |
| 城区、农田等高相干场景 | 热带雨林、水体等低相干场景 |
| 两时相孔径支撑相同 | 两时相孔径差异较大 |
| 微小形变（毫米级，无相位缠绕） | 大形变场景（需要额外解缠步骤） |

## 我的观点

DP-JMRNet 解决了一个真实存在但被系统性忽视的问题：**优化目标和应用目标的错位**。SAR 重建的优化目标是幅度 MSE，但 InSAR 的真正目标是差分相位。这个错位的代价被量化为 47-51% 的性能差距，说明这不是小问题。

深度展开的选择也是明智的。相比纯黑箱网络，展开结构允许工程师检查每一层的中间结果，定位重建失败的原因——在地球科学应用中，可解释性和精度同样重要。

**开放问题：** 相位缠绕依然悬而未决。当地表形变超过半个波长（C 波段约 2.8 cm），差分相位就会缠绕。未来的工作需要把解缠（phase unwrapping）整合进重建过程，才能覆盖大地震等极端场景。

官方代码：[https://github.com/JasonBao05/coherent-sar-unfolding](https://github.com/JasonBao05/coherent-sar-unfolding)