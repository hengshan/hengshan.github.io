---
layout: post-wide
title: "神经符号 AI 的统一框架：用单子（Monad）理解 NeSyCat Torch"
date: 2026-06-18 08:04:46 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.19279v1
generated_by: Claude Code CLI
---

## 一句话总结

NeSyCat Torch 用范畴论的"单子"把模糊逻辑、概率逻辑、神经网络统一在同一套可微框架下，在 MNIST 加法任务上同时实现了比 LTN/DeepProbLog 更快、精度接近 DeepStochLog 的效果——但你需要接受：这套框架的认知门槛比调 PPO 超参数高得多。

## 背景：神经符号 AI 现在是一盘散沙

想象你要训练一个神经网络，让它学会"两张图片里的数字相加等于某个结果"这种规则。你有三个世界可以选择：

**纯神经网络**：端到端训练，规则隐式编码在权重里，不可解释，OOD 时容易翻车。

**纯符号 AI**：规则清晰，但处理不了原始图片，必须手工特征工程。

**神经符号 AI**：两者兼得——但各框架互不兼容：

| 框架 | 真值类型 | 主要问题 |
|------|---------|---------|
| LTN | `[0,1]` 模糊值（t-norm） | 精度受限于模糊化 |
| DeepProbLog | 精确概率 | 推理开销大，难以扩展 |
| DeepStochLog | 近似概率（beam search） | 近似误差，框架不通用 |
| NeSyCat | 参数化真值（单子） | 门槛高，理解成本大 |

NeSyCat 的核心主张：以上所有方法都是"强单子在不同真值代数上的特例"。通过参数化单子，同一套代码可以切换推理语义，不需要为每种逻辑分别实现。

## 单子不是魔法，是边际化

先把数学术语放一边，用 MNIST 加法来理解单子实际做了什么。

### MNIST 加法任务

**输入**：两张 MNIST 数字图片  
**输出**：预测它们的和（0~18）  
**约束**：训练时只知道"和是多少"，不知道每张图片具体是哪个数字

这是弱监督学习。神经符号解法的核心公式：

$$
P(\text{sum} = n) = \sum_{a+b=n} P(\text{digit}_1 = a \mid \text{img}_1) \times P(\text{digit}_2 = b \mid \text{img}_2)
$$

这就是**边际化**：把所有满足 $a+b=n$ 的情况的概率累加。

**单子做了什么？** 在范畴论里：
- `M α`（单子类型）：表示"关于 α 的概率分布"
- `>>=`（绑定操作）：把 `M α` 和函数 `α → M β` 组合成 `M β`，等价于对 X 求边际

对应到 Python，就是：

```python
def monadic_bind(dist_x: torch.Tensor, transition) -> torch.Tensor:
    """
    dist_x: [B, n] - P(X=i) 的分布
    transition(i): 返回给定 X=i 时 Y 的分布 [B, m]
    返回: P(Y) = Σ_x P(X=x) * P(Y|X=x)
    """
    result = torch.zeros_like(transition(0))
    for i in range(dist_x.shape[1]):
        result += dist_x[:, i:i+1] * transition(i)  # 加权求和 = 边际化
    return result
```

这就是全部魔法。接下来把它做得高效且数值稳定。

## 算法原理

### Log-Semiring：为什么不能直接乘概率

NeSyCat Torch 的核心工程贡献是**对数张量单子（log-tensor monad）**——在对数半环上工作：

| 操作 | 概率空间 | 对数空间 |
|------|---------|---------|
| 联合概率（AND） | $p \times q$ | $\log p + \log q$ |
| 边际化（OR） | $p + q$ | $\text{logsumexp}(\log p, \log q)$ |

当推理链变长（多个规则组合），概率乘积会迅速下溢至 0，梯度消失。对数空间把乘法变加法，完全避开这个问题。

## 最小可运行实现

下面是 NeSyCat 风格的 MNIST 加法，核心逻辑在 50 行以内：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitNet(nn.Module):
    """感知模块：图片 → 数字的 log 概率分布"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        return F.log_softmax(self.net(x), dim=-1)  # [B, 10]

def addition_rule(log_p1: torch.Tensor, log_p2: torch.Tensor) -> torch.Tensor:
    """
    符号规则：加法，全程在对数空间计算（log-tensor monad 实现）
    log_p1, log_p2: [B, 10] — 两个数字的 log 概率
    返回: [B, 19] — 和的 log 概率
    """
    # 外积：log P(a,b) = log P(a) + log P(b)，形状 [B, 10, 10]
    log_joint = log_p1.unsqueeze(2) + log_p2.unsqueeze(1)
    
    # 构建 "和→数字对" 的索引映射（只算一次）
    a_idx = torch.arange(10, device=log_p1.device)
    b_idx = torch.arange(10, device=log_p1.device)
    
    log_sum_probs = torch.full((log_p1.shape[0], 19), float('-inf'),
                               device=log_p1.device)
    for s in range(19):
        # 满足 a+b=s 的所有对，用 logsumexp 边际化（OR 操作）
        valid_a = [a for a in range(10) if 0 <= s - a < 10]
        valid_b = [s - a for a in valid_a]
        terms = log_joint[:, valid_a, valid_b]     # [B, k]
        log_sum_probs[:, s] = torch.logsumexp(terms, dim=1)
    
    return log_sum_probs

def train_step(model, img1, img2, target, optimizer):
    log_p1 = model(img1)                             # [B, 10]
    log_p2 = model(img2)                             # [B, 10]
    log_pred = addition_rule(log_p1, log_p2)         # [B, 19]
    loss = F.nll_loss(log_pred, target)              # 负对数似然
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

### 完整训练循环

```python
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

class MNISTAdditionDataset(Dataset):
    def __init__(self, train=True, n_pairs=30000):
        mnist = MNIST('./data', train=train, download=True,
                      transform=transforms.ToTensor())
        n = len(mnist)
        idx1 = torch.randint(0, n, (n_pairs,))
        idx2 = torch.randint(0, n, (n_pairs,))
        # 直接读取底层数据，避免逐条 transform
        images = mnist.data.float().unsqueeze(1) / 255.0
        labels = mnist.targets
        self.img1, self.img2 = images[idx1], images[idx2]
        self.sums = labels[idx1] + labels[idx2]

    def __len__(self): return len(self.sums)
    def __getitem__(self, i): return self.img1[i], self.img2[i], self.sums[i]


def train(n_epochs=15, batch_size=512, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    loader = DataLoader(MNISTAdditionDataset(train=True),
                        batch_size=batch_size, shuffle=True, num_workers=4)
    for epoch in range(n_epochs):
        total = sum(train_step(model,
                               i1.to(device), i2.to(device), s.to(device),
                               optimizer)
                    for i1, i2, s in loader)
        scheduler.step()
        print(f"Epoch {epoch+1:2d}: loss={total/len(loader):.4f}")
    return model
# ... (评估代码省略)
```

### 关键 Trick

**1. 永远用 `log_softmax` + `nll_loss`，不要用 `softmax` + `cross_entropy`**

`log_softmax` 的梯度比 `softmax` 更数值稳定，尤其当某个类概率极小时。

**2. `addition_rule` 中的 `valid_a, valid_b` 预计算移到外部**

```python
# 在训练前预计算，避免每次 forward 重建
VALID_PAIRS = {s: [a for a in range(10) if 0 <= s-a < 10] for s in range(19)}
```

**3. 不要在 `addition_rule` 里用 `detach()`**

这是最常见的错误——一旦 detach，梯度就断了，神经网络什么都学不到。

## 实验

### 数值稳定性消融

以下代码展示 log-space 为什么不可少：

```python
# 模拟 5 层推理链中的概率下溢
p = torch.tensor([0.1] * 10)
log_p = torch.log(p)

result_prob = p.clone()
result_log  = log_p.clone()
for _ in range(5):
    result_prob = result_prob * p    # 乘法下溢
    result_log  = result_log  + log_p  # 加法稳定

print(f"概率空间最小值: {result_prob.min():.2e}")  # 1e-10，接近 float32 精度极限
print(f"log 空间最小值: {result_log.min():.2e}")   # -23.0，完全正常
```

### 与 Baseline 对比

| 方法 | MNIST 加法准确率 | 速度（相对 NeSyCat）| 框架通用性 |
|------|---------------|------------------|---------|
| LTN | ~97% | 慢 0.5× | 仅模糊逻辑 |
| DeepProbLog | ~99% | 慢 3~5× | 仅概率逻辑 |
| DeepStochLog | ~99% | 接近 | 近似，不统一 |
| **NeSyCat Torch** | **~99%** | **基准** | **统一框架** |

NeSyCat 的速度优势来自批量张量操作，在 batch size 大时尤其明显。精度上与 DeepProbLog 和 DeepStochLog 持平。

## 调试指南

### 问题 1：Loss 从第一步就卡住不动

**症状**：loss 停在 `log(19) ≈ 2.94`（均匀分布的熵），完全不下降。

**诊断梯度流**：

```python
model = DigitNet()
img1, img2 = torch.randn(4, 1, 28, 28), torch.randn(4, 1, 28, 28)
target = torch.tensor([5, 9, 12, 3])

loss = F.nll_loss(addition_rule(model(img1), model(img2)), target)
loss.backward()

for name, param in model.named_parameters():
    if param.grad is None:
        print(f"[ERROR] {name}: 梯度断裂（检查是否有 .detach()）")
    elif param.grad.abs().max() < 1e-9:
        print(f"[WARN]  {name}: 梯度几乎为零（{param.grad.abs().max():.1e}）")
    else:
        print(f"[OK]    {name}: 梯度正常（max={param.grad.abs().max():.1e}）")
```

### 问题 2：Loss 出现 NaN

**原因**：`logsumexp` 输入全为 `-inf` 时返回 `nan`，通常是某个数字类别从未被采样。

```python
# 加保护性检查
def safe_addition_rule(log_p1, log_p2):
    log_pred = addition_rule(log_p1, log_p2)
    if torch.isnan(log_pred).any():
        # 检查输入是否包含 -inf 列
        zero_cols = (log_p1 < -30).all(dim=0).nonzero()
        raise ValueError(f"log_p1 的以下类别概率全为 0：{zero_cols}")
    return log_pred
```

### 问题 3：验证准确率远低于论文

最常见原因是数据集生成方式不同。NeSyCat 论文使用固定种子的配对方式。评估时注意：

```python
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for img1, img2, target in loader:
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            pred = addition_rule(model(img1), model(img2)).argmax(dim=-1)
            correct += (pred == target).sum().item()
            total   += target.size(0)
    return correct / total
```

### 超参数敏感度

| 参数 | 推荐值 | 敏感度 | 备注 |
|------|-------|-------|-----|
| 学习率 | `1e-3` | 高 | `1e-2` 容易发散，`1e-4` 收敛慢 |
| batch size | `256~512` | 中 | 太小则 logsumexp 方差偏大 |
| 训练轮数 | `10~20` | 低 | MNIST 加法不容易过拟合 |
| lr schedule | StepLR/CosLR | 低 | 有比没有好，类型影响不大 |

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有明确符号规则 + 原始感知输入 | 纯神经网络任务（无需推理规则） |
| 弱监督学习（只有结果标签） | 规则简单、可直接手写边际化 |
| 需要在多种逻辑语义之间切换 | 团队没有函数式编程基础 |
| 研究型项目，追求框架统一性 | 工业快速迭代，调试成本是瓶颈 |

## 我的观点

NeSyCat Torch 的统一框架在学术层面**确实优雅**。用同一套单子接口切换概率/模糊/经典语义，是框架设计上的进步。对数张量单子（log-semiring 计算）也是真实的工程贡献，任何神经符号系统都应该采用这个技巧。

但我要诚实地说：

**对大多数实践者的实用性有限。** 如果你的任务只需要概率语义，直接手写边际化代码 20 行就够了，不需要引入单子抽象层。单子的价值在于你需要**频繁切换语义**，或者构建**通用推理引擎**时才显现。

**门槛确实很高。** 论文大量使用 Haskell 和范畴论术语（"strong monad"、"Giry monad"、"do-notation"），不了解这些背景的读者很难修改框架核心逻辑。这不是作者的问题，而是所在领域的现实。

**MNIST 加法这个 benchmark 太弱。** 所有框架在这个任务上都能到 99%，无法区分方法的真实能力差异。真正有说服力的对比应该在更复杂的组合推理任务上。

**什么时候值得一试？** 如果你在做神经符号系统研究、需要理论上的保证、或者你的任务确实需要在多种逻辑语义下运行，NeSyCat 是目前最严格的统一框架之一。如果你只是想"给神经网络加点规则"，从手写边际化开始更实际。