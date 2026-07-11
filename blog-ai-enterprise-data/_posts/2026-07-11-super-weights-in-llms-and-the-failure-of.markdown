---
layout: post-wide
title: '参数重要性 ≠ 参数可训练性：LLM "超级权重" 悖论'
date: 2026-07-11 12:02:10 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.08733v1
generated_by: Claude Code CLI
---

## 一句话总结

论文证明了一个反直觉的发现：LLM 中那些"删一个就崩溃"的超级权重（Super Weights），恰恰是你**不能单独微调**的参数——对它们定向训练会让模型直接退化到随机猜测水平。

---

## 为什么这篇论文重要？

大语言模型的参数不是均质的。过去一年，研究者们发现了一种极端现象：模型中存在少数"超级权重"（Super Weights），仅仅移除其中几个，模型性能就会从正常水平断崖式下跌到随机猜测。这听起来很像系统中的单点故障——如果这些参数如此关键，那么：

1. 它们是模型能力的"存储中心"吗？
2. 如果专门训练这些参数，能不能高效地注入新知识？

这篇论文的回答是：**不能，而且会让模型彻底崩溃。**

这个结论打破了"参数重要性"和"参数可训练性"之间天然等价的直觉，并为 LoRA 为何有效提供了一个新的理解视角。

---

## 什么是超级权重？

在深入讨论之前，先建立直觉。

LLM 的权重矩阵中，绝大多数值分布在某个正态范围内。但存在极少数离群值（outlier），其绝对值比均值大 10-100 倍。这些就是超级权重。

**类比**：想象一栋拱门建筑，拱顶的那块楔石（keystone）承受着来自两侧的所有压力。它不是"最重的石头"，但是最不可或缺的结构件。

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_super_weights(model, layer_name="down_proj", top_k=100):
    """识别模型中的超级权重位置"""
    super_weight_coords = {}
    
    for name, param in model.named_parameters():
        if layer_name not in name:
            continue
        
        w = param.data.float()
        # 超级权重定义：绝对值远超均值 + 3*标准差的参数
        threshold = w.abs().mean() + 3 * w.abs().std()
        mask = w.abs() > threshold
        
        # 按绝对值降序返回坐标
        flat_indices = w.abs().flatten().topk(top_k).indices
        rows = flat_indices // w.shape[1]
        cols = flat_indices % w.shape[1]
        
        super_weight_coords[name] = list(zip(rows.tolist(), cols.tolist()))
        print(f"{name}: 找到 {mask.sum().item()} 个超级权重，"
              f"最大值 {w.abs().max():.2f} vs 均值 {w.abs().mean():.4f}")
    
    return super_weight_coords

# 示例：加载模型并检测
# model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
# coords = find_super_weights(model)
```

超级权重有两个特征：
- **位置固定**：在不同的 prompt 下，它们的位置不变
- **删除敏感**：置零后，输出的 token 分布立刻崩坏

---

## 核心发现：定向训练为何失败？

### 实验设计

论文的实验逻辑非常清晰：

| 训练策略 | 训练参数数量 | 结果 |
|---------|------------|------|
| 只训练超级权重坐标 | 100 ~ 8,192 | 崩溃（随机猜测） |
| 扩展到超级权重的局部邻域 | 最多 36K | 仍然崩溃 |
| 在同层随机选等量坐标训练 | 100 ~ 8,192 | **优于基线** |
| LoRA（全层低秩分解） | ~0.16% 参数 | **成功** |
| LoRA 但排除超级权重坐标 | ~0.16% 参数 | **与普通 LoRA 无显著差异** |

最关键的对照实验：**同样的稀疏度、同样的层、随机坐标成功了，但超级权重坐标失败了。**

这排除了"稀疏训练本身有问题"的假说。问题是针对超级权重**坐标**的，而不是稀疏性。

### 为什么定向训练会崩溃？

我的理解是梯度更新的"尺度错配"问题：

超级权重的值 $w_{ij}$ 极大，这意味着：

$$\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial w_{ij}}$$

其中 $\mathbf{h} = \mathbf{W}\mathbf{x}$，输出激活对这个特定权重极度敏感。学习率的任何微小扰动，都会在输出空间产生放大的影响，破坏模型已经建立的精细表示。

而其他参数（随机选的）不存在这个问题——它们的梯度处于正常尺度，更新是稳定的。

---

## 代码实验：复现失败场景

下面用一个简化实验演示选择性训练的行为差异：

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from copy import deepcopy

class MinimalMLP(nn.Module):
    """简化版 MLP，模拟 LLM 中的 down_proj 行为"""
    def __init__(self, d_in=256, d_out=128):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)
        # 手动注入几个"超级权重"
        with torch.no_grad():
            self.proj.weight[0, :5] = 50.0  # 极大值，模拟超级权重
    
    def forward(self, x):
        return self.proj(x)

def selective_train(model, data, target_coords, n_steps=200, lr=1e-3):
    """只训练指定坐标的参数"""
    model = deepcopy(model)
    
    # 创建仅更新目标坐标的梯度钩子
    mask = torch.zeros_like(model.proj.weight)
    for r, c in target_coords:
        mask[r, c] = 1.0
    
    optimizer = AdamW([model.proj.weight], lr=lr)
    losses = []
    
    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(data)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        
        # 只保留目标坐标的梯度
        model.proj.weight.grad *= mask
        optimizer.step()
        losses.append(loss.item())
    
    return losses, model

# 生成测试数据
torch.manual_seed(42)
x = torch.randn(64, 256)
target = torch.randn(64, 128)

model = MinimalMLP()

# 方案A：定向训练超级权重（前5列第0行）
super_coords = [(0, i) for i in range(5)]
losses_super, m_super = selective_train(model, x, target, super_coords)

# 方案B：定向训练等量随机坐标
random_coords = [(i // 256, i % 256) 
                 for i in torch.randperm(256*128)[:5].tolist()]
losses_random, m_random = selective_train(model, x, target, random_coords)

print(f"超级权重定向训练  最终 loss: {losses_super[-1]:.4f}")
print(f"随机坐标定向训练  最终 loss: {losses_random[-1]:.4f}")
```

即使是这个玩具模型，你也会观察到超级权重坐标的训练曲线更不稳定——在真实的 OLMo 规模下，这种不稳定会演变为完全崩溃。

---

## LoRA 为何成功：结构胜于选择

这是论文最深刻的一个隐含结论。

LoRA 不是因为"避开了超级权重"而成功——论文专门做了一个实验：把 LoRA 的更新矩阵中对应超级权重坐标的部分强制归零，结果**几乎没有性能差异**。

这说明：
1. 超级权重**不需要被更新**，微调也能成功
2. LoRA 成功的原因是**低秩结构强制了层级协同更新**

LoRA 的更新可以写成：

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)$$

这个低秩约束意味着：每一个参数的更新都不是独立的，而是受整个秩空间约束的。超级权重所在位置的更新量，自然被整个矩阵的方向性结构所制约，不会产生爆炸式的偏移。

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, base_weight, rank=8):
        super().__init__()
        d_out, d_in = base_weight.shape
        self.base = base_weight  # 冻结
        # 低秩更新矩阵
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        self.scaling = 1.0 / rank
    
    def forward(self, x):
        base_out = x @ self.base.T
        lora_out = x @ self.A.T @ self.B.T * self.scaling
        return base_out + lora_out

# LoRA 的关键：base_weight 完全冻结，超级权重不会被直接修改
# 但整层的表达能力通过低秩扰动得到调整
```

**工程直觉**：低秩分解就像给整个层套上一个"协调约束"，没有哪个单独的坐标能够脱离整体结构独立暴走。

---

## 实现中的坑

### 坑1：学习率对超级权重极度敏感

```python
# 错误示范：用统一学习率训练含超级权重的稀疏子集
optimizer = AdamW(super_weight_params, lr=1e-4)  # 看似合理

# 问题：Adam 会对梯度进行归一化，但超级权重的梯度本身就不稳定
# 更好的做法是使用极小学习率或直接 LoRA
optimizer = AdamW([
    {"params": super_weight_params, "lr": 1e-6},  # 极保守
    {"params": other_params, "lr": 1e-4},
])
```

### 坑2：把"删除敏感"误判为"更新有益"

移除超级权重 → 性能崩溃  
≠  
更新超级权重 → 性能提升

这两个命题**互相独立**。"不可缺少"不等于"可以优化"。就像你的脊椎骨对生命至关重要，但这不意味着你应该单独去"训练"脊椎骨。

### 坑3：局部邻域扩展没有帮助

论文尝试了从单个超级权重扩展到其局部 $k \times k$ 邻域（最多 36K 参数），依然崩溃。这暗示问题不是"上下文不足"，而是这个坐标本身就不适合作为训练目标。

---

## 实验：论文结论的适用边界

论文使用 OLMo-1B 和 OLMo-7B。这里有几点需要注意：

**能复现的部分**：
- 超级权重在 `down_proj` 层中存在（跨模型普遍现象）
- LoRA 以极少参数达到较好效果（已有大量工程实践验证）

**不确定的部分**：
- 论文中提到"Super Weights 的危害性不普遍适用于所有 LLM"——不同架构、不同训练数据的模型行为可能不同
- 实验局限于 OLMo 系列（开放权重但非主流部署模型）
- 在 Llama、Mistral 等模型上是否完全等效，需要单独验证

---

## 什么时候用 / 不用定向权重训练？

| 适用场景 | 不适用场景 |
|---------|-----------|
| LoRA / QLoRA 微调 | 针对超级权重坐标的定向稀疏训练 |
| 全层参数的低秩更新 | 基于"重要性排序"选参数做 sparse fine-tuning |
| 随机稀疏子集训练（如 random pruning + retrain） | 以超级权重为核心的 PEFT 方案 |
| 按层冻结（freeze early layers，train later） | 假设"重要参数 = 训练收益高"的方案 |

---

## 我的观点

这篇论文的意义不止于"超级权重不能单独训练"这个结论，它更深层的贡献是：

**它把"参数重要性"和"参数可训练性"解耦了。**

这两个概念在过去的文献中经常被混用。很多剪枝、PEFT 方法都隐含地假设"重要的参数应该被优先训练/保留"——但这篇论文告诉我们，这个假设在极端情况下是错的。

对实践者的最大启示是：**不要根据权重的量级或重要性来选择微调目标**，而应该依赖结构化的更新方案（如 LoRA）。结构不仅仅是一种参数效率的技巧，它本质上是一种稳定训练的约束。

一个有趣的开放问题是：超级权重究竟编码了什么？如果它们不是通过直接训练产生贡献的，那么它们在预训练中是怎么形成的、扮演什么角色？这可能是理解大模型涌现能力的一个切入点。

---

**论文链接**：https://arxiv.org/abs/2607.08733v1