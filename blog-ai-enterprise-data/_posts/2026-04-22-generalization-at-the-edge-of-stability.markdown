---
layout: post-wide
title: "边缘稳定性（Edge of Stability）：混沌训练动力学与泛化的理论解释"
date: 2026-04-22 12:05:54 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.19740v1
generated_by: Claude Code CLI
---

## 一句话总结

大学习率让训练进入混沌状态（Edge of Stability），但这种混沌往往比"安全"的小学习率泛化更好——本文给出理论框架和可实验验证的代码。

## 背景：一个让人困惑的经验现象

你可能遇到过这种情况：学习率调小，训练稳定，但测试集表现反而差；学习率调大，loss 跳来跳去，测试集却更好。这不是你的错觉。

这个现象有个名字：**边缘稳定性（Edge of Stability, EOS）**。

经典优化理论告诉我们：梯度下降在学习率 $\eta$ 满足 $\eta \cdot \lambda_{\max} < 2$ 时才稳定，其中 $\lambda_{\max}$ 是 Hessian 矩阵的最大特征值（又叫**锐度 sharpness**）。超过这个界，理论上应该发散。

但实践中，现代神经网络常常在 $\eta \cdot \lambda_{\max} \approx 2$ 附近训练，曲线震荡，却收敛到更好的解。

**这篇论文的核心问题：为什么在混沌边缘训练能泛化更好？**

他们把随机优化器建模为**随机动力系统**，发现其吸引子是低维的分形集合，并引入"锐度维度（sharpness dimension）"证明泛化界——同时顺带解释了 grokking 现象。

## 核心概念

### 直觉：三个区域

| 区域 | 条件 | 现象 | 泛化 |
|------|------|------|------|
| 稳定区 | $\eta \cdot \lambda_{\max} < 2$ | loss 单调下降 | 一般 |
| 边缘稳定（EOS） | $\eta \cdot \lambda_{\max} \approx 2$ | 震荡但收敛 | 往往更好 |
| 不稳定区 | $\eta \cdot \lambda_{\max} \gg 2$ | 发散 | 无 |

EOS 的直觉：优化器无法在某个尖锐极小值上停下来，被迫"弹出"，最终稳定在一个**更平坦、更泛化的解附近**。

### Hessian 锐度的计算

```python
import torch

def compute_sharpness(model, loss_fn, data_loader, n_iters=20):
    """用幂迭代估算 Hessian 最大特征值（锐度）"""
    params = [p for p in model.parameters() if p.requires_grad]
    
    # 随机初始化方向向量并归一化
    v = [torch.randn_like(p) for p in params]
    norm = sum((vi**2).sum() for vi in v).sqrt()
    v = [vi / norm for vi in v]
    
    x, y = next(iter(data_loader))
    
    for _ in range(n_iters):
        loss = loss_fn(model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Hessian-vector product: H·v
        Hv = torch.autograd.grad(
            sum((g * vi).sum() for g, vi in zip(grads, v)), params
        )
        
        eigenvalue = sum((hv * vi).sum() for hv, vi in zip(Hv, v)).item()
        norm = sum((hv**2).sum() for hv in Hv).sqrt()
        v = [hv / norm for hv in Hv]
    
    return eigenvalue
```

### 论文的理论核心：锐度维度

经典动力系统中，Kaplan-Yorke 维度描述吸引子复杂度：

$$
d_{KY} = j + \frac{\sum_{i=1}^{j} \lambda_i}{|\lambda_{j+1}|}
$$

其中 $\lambda_1 \geq \lambda_2 \geq \cdots$ 是 Lyapunov 指数，$j$ 是使前缀和仍为正的最大下标。

论文类比定义**锐度维度**，用 Hessian 特征值 $h_1 \geq h_2 \geq \cdots \geq h_n$ 替代：

$$
d_{\text{sharpness}} = j^* + \frac{\sum_{i=1}^{j^*} \log h_i}{|\log h_{j^*+1}|}
$$

**关键贡献**：这个维度依赖 Hessian 的**完整谱**，而不仅仅是：
- trace（$\sum h_i$，很多 PAC-Bayes flatness 工作用这个）
- 谱范数（只看 $h_1$）

论文证明：基于锐度维度的泛化界在混沌训练区（EOS）更紧，解释了为什么 EOS 能泛化。

## 实验：亲手验证

### 观测 EOS 现象

```python
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 10)
        )
    def forward(self, x): return self.net(x.view(x.size(0), -1))

def train_with_sharpness_tracking(lr, n_epochs=50):
    model = MLP()
    # 注意：用 full-batch GD 更容易观测 EOS（论文也在这个设置下分析）
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    eos_threshold = 2.0 / lr

    log = {"sharpness": [], "train_loss": [], "test_acc": []}

    for epoch in range(n_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()

        s = compute_sharpness(model, loss_fn, train_loader)
        log["sharpness"].append(s)
        # EOS ratio > 1 说明进入了 EOS 区
        print(f"LR={lr} | Epoch {epoch:3d} | sharpness={s:.1f} | EOS={eos_threshold:.1f} | ratio={s/eos_threshold:.2f}")

    return log

# 对比三个学习率：稳定区 / EOS 区 / 不稳定区
results = {lr: train_with_sharpness_tracking(lr) for lr in [0.001, 0.01, 0.05]}
```

**你应该观察到的规律**：
- lr=0.001：sharpness 持续上升，远低于 EOS 阈值，训练"稳定"
- lr=0.01：sharpness 爬升到 ≈ 2/lr 后**停留在那里**，loss 轻微震荡——这就是 EOS
- lr=0.05：sharpness 超过阈值，loss 明显震荡，但可能测试集仍然更好

### Grokking：延迟的泛化——EOS 的极端案例

Grokking 指模型先训练到 100% 训练精度（过拟合），然后经过很长时间，测试精度突然跳升。论文给出了新的解释。

```python
def make_modular_addition_data(p=97, train_frac=0.3):
    """模块化加法：(a + b) mod p，经典 grokking 任务"""
    import random
    all_data = [(a, b, (a + b) % p) for a in range(p) for b in range(p)]
    random.shuffle(all_data)
    split = int(len(all_data) * train_frac)

    def encode(data):
        X = torch.zeros(len(data), 2 * p)
        Y = torch.zeros(len(data), dtype=torch.long)
        for i, (a, b, c) in enumerate(data):
            X[i, a] = 1.0; X[i, p + b] = 1.0; Y[i] = c
        return TensorDataset(X, Y)

    return encode(all_data[:split]), encode(all_data[split:])

# 用 AdamW + weight decay 训练，观察 train/test acc 随时间变化
# train acc 很快到 100%，test acc 在数千步后突然跳升
train_ds, test_ds = make_modular_addition_data(p=97, train_frac=0.3)
# ... (训练循环省略，需要 5000-50000 步)
```

**论文对 grokking 的解释**（基于锐度维度）：
- 阶段 1：模型找到"记忆型"解，高锐度，高 sharpness dimension，不泛化
- 阶段 2：EOS 训练让优化器在吸引子集合上游荡，**慢慢漂移**向低维部分
- 阶段 3：模型抵达低锐度维度的区域，泛化突然出现

这比"weight decay 压缩范数"的解释更细致，因为它解释了为什么需要等那么久。

## 工程实践

### Sharpness 监控仪表盘

```python
class SharpnessMonitor:
    """训练循环中实时监控 EOS 状态"""
    def __init__(self, model, loss_fn, loader, lr):
        self.model = model
        self.loss_fn = loss_fn
        self.loader = loader
        self.lr = lr
        self.eos_threshold = 2.0 / lr

    def log(self, step):
        if step % 200 != 0:
            return
        s = compute_sharpness(self.model, self.loss_fn, self.loader, n_iters=10)
        ratio = s / self.eos_threshold
        status = "EOS ✓" if 0.8 < ratio < 1.5 else ("UNSTABLE !" if ratio > 1.5 else "sub-EOS")
        print(f"Step {step:5d} | sharpness={s:.1f} | EOS ratio={ratio:.2f} | {status}")
        return {"sharpness": s, "eos_ratio": ratio}
```

### 推荐配置（基于 EOS 视角）

```python
# 进入 EOS 的推荐起点
config = {
    "optimizer": "SGD",           # SGD 比 Adam 更容易进入 EOS
    "lr": 0.1,                    # 足够大；从这里开始调
    "momentum": 0.9,
    "weight_decay": 1e-4,         # 轻微正则，不要太强（会阻止 EOS 游荡）
    "gradient_clip": 1.0,         # EOS 区梯度会波动，必须裁剪
    "lr_warmup": True,            # 预热后再让 sharpness 爬到 EOS
    "lr_schedule": "cosine",      # 尽量晚才显著降低 lr
}
```

## 调试指南

### 常见问题

| 现象 | 可能原因 | 建议 |
|------|---------|------|
| sharpness 持续上升，不稳定 | lr 过大，进了不稳定区 | 减小 lr 或加大 gradient_clip |
| sharpness 远低于 2/lr | 没进入 EOS | 增大 lr |
| 训练 loss 下降但测试不动 | 记忆型解（高锐度） | 增大 weight decay，延长训练 |
| 想复现 grokking 但看不到 | 训练集太大或训练步数太少 | 用 ≤30% 数据，训练 ≥ 10k 步 |

### 如何判断你在 EOS

- sharpness ≈ 2/lr，且相对稳定（不是一直爬）
- EOS ratio 在 0.9–1.2 之间
- loss 有轻微但持续的震荡（不是单调下降）

### 超参数敏感度

| 参数 | 推荐范围 | 敏感度 | 说明 |
|------|---------|-------|------|
| lr | 0.01–0.1（SGD） | 高 | 决定是否进入 EOS |
| weight_decay | 1e-5–1e-3 | 中 | 太大会阻止 EOS 游荡 |
| gradient_clip | 0.5–2.0 | 中 | EOS 区必需 |
| batch_size | 32–512 | 高 | 大 batch 等效于小 lr，可能退出 EOS |

## 局限性：诚实评价

**论文的贡献是真实的**：锐度维度比 trace 或谱范数更细粒度，grokking 的解释有新意，实验结果支持理论。

**但也要看清限制**：

- 锐度维度**很难直接计算**：需要完整的 Hessian 谱，在大模型上不现实（参数量一到几亿，Hessian 连存都存不下）
- 泛化界的常数因子通常很松，"存在性"意义大于"可量化"意义
- "分形吸引子"的直觉漂亮，但严格验证困难，目前更多是类比

**对从业者的实际价值**：这篇论文更多是**解释**已有实践，而不是提供新的调参手册。但理解背后机制，能让你调参时更有方向感——比如知道"sharpness 稳定在 2/lr 附近"是个好信号，而不是要急着降学习率。

## 什么时候值得关注这个框架？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 解释某个 lr 为什么比另一个好 | 直接计算大模型的锐度维度 |
| 理解 / 复现 grokking 现象 | 替代 hyperparameter search |
| 设计训练监控指标（sharpness） | 需要精确泛化界的理论证明 |
| 小模型消融实验 | 超大规模训练场景 |

## 我的观点

EOS 是一个先被从业者发现（大 lr 往往更好），然后理论跑去解释的典型案例。这篇论文的理论框架比之前的工作更精细——用锐度维度替代 trace 或谱范数，确实捕捉到了更多结构信息。

但我对泛化理论工作有个通用评价：理论界通常很松，对大模型基本没有定量指导意义。这篇论文也不例外。

真正有价值的实用 takeaway 是：**把 `sharpness / (2/lr)` 这个 EOS ratio 加到你的训练 dashboard 里**。不需要理解全部理论，只需要知道这个比值在 1.0 附近是正常的，高得离谱说明不稳定，低得很说明 lr 还有上调空间。

至于 grokking 的解释——如果你有需要研究 grokking 的任务（比如数学推理、算法任务），这个框架给了你一个新角度：延长训练时间，让优化器在 EOS 上慢慢游荡，泛化会来的。

---

**参考文献**
- 本文论文：[Generalization at the Edge of Stability, 2025](https://arxiv.org/abs/2604.19740v1)
- EOS 原始发现：Cohen et al. 2021, [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065)
- Grokking 原论文：Power et al. 2022, [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)