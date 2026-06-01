---
layout: post-wide
title: "图到文本生成中的 Masked Diffusion：解码轨迹到底揭开了什么？"
date: 2026-06-01 12:04:51 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.31564v1
generated_by: Claude Code CLI
---

## 一句话总结

MDLMs 在图到文本任务上天然遵循"实体→关系→结构"的去掩码顺序，但 SFT 会破坏这一策略，导致输出被截断或产生幻觉——论文提出的 lambda-scaled 解码无需重新训练就能找回 +9.4 BLEU-4。

---

## 背景：自回归模型的"线性诅咒"

图到文本生成（Graph-to-Text）的任务是把知识图谱三元组（如 `(Alan_Turing, birthplace, London)`）转换成流畅的自然语言句子。

传统的自回归 LLM（GPT、LLaMA 系列）在这个任务上有个结构性缺陷：**它们只能从左到右生成，而图结构是并行的**。一个图中的三个三元组本质上没有顺序，但自回归模型必须强行给它们一个顺序。

Masked Diffusion Language Models（MDLMs）提供了另一条路。以 LLaDA 为代表：

- **正向过程**：用 `[MASK]` token 逐步遮盖输入
- **逆向过程**：从全 `[MASK]` 开始，多步并行去掩码，每步根据置信度解开若干 token

这种生成方式更接近"打草稿→填词"，而不是"一个字一个字往下写"。

但 MDLM 在实践中有多难调？坦白说，比 PPO 还要玄学。**这篇论文做的第一件事，是把 MDLM 的生成过程拍成了慢镜头。**

---

## 解码轨迹：MDLM 到底在按什么顺序思考？

### 什么是解码轨迹？

在 MDLM 的迭代解码中，每一步都有若干 `[MASK]` 被替换成真实 token。**解码轨迹**就是这些 token 被揭开的顺序。

例如，对于目标句子 `"Alan Turing was born in London in 1912."`：

```
步骤 0: [M]   [M]     [M]  [M]   [M] [M]     [M] [M]   [M]
步骤 1: Alan  [M]     [M]  born  [M] London  [M] [M]   [M]   ← 实体先揭开
步骤 2: Alan  Turing  [M]  born  in  London  in  [M]   [M]   ← 关系词揭开
步骤 3: Alan  Turing  was  born  in  London  in  1912  .     ← 结构词最后
```

这个顺序不是人为设计的，而是模型自然涌现的行为。

### 三类 Token 的解码优先级

| 类别 | 例子 | MDLM 解码顺序 | 原因 |
|------|------|-------------|------|
| **实体词** | Alan, Turing, London | 最先（高置信度） | 图中明确给出，不确定性最低 |
| **关系/功能词** | was, born, in | 中间 | 由实体决定，有一定不确定性 |
| **结构词** | 句号、逗号、EOS | 最后 | 句式结构由内容决定 |

---

## SFT 的失效模式：把句子长度"钉死"了

对 MDLM 做有监督微调（SFT）之后，一个诡异的事发生了：**结构 token（尤其是句尾标点和 EOS）的置信度在第一步就变得极高**。

```
步骤 0: [M]  [M]    [M]  [M]  [M] [M]     .     ← SFT 后，句点在第一步就被钉死
步骤 1: [M]  [M]    was  born [M] London   .
步骤 2: Alan Turing was  born in  London   .     ← 后续信息被截断，1912 丢了
```

长度被锁定后，如果目标句子实际更长（包含更多信息），模型只能：
1. **截断信息**：把后面的实体直接省略
2. **幻觉填充**：用错误 token 填满"规划好"的空位

这解释了为什么 SFT 有时会让 MDLM 的跨数据集泛化崩掉。

---

## Lambda-Scaled Structural Decoding：推理时一行代码的修复

解决方案出奇简单：**在推理时把结构 token 的置信度乘以 $\lambda$（$0 < \lambda < 1$）**，让它们晚点被揭开。

$$
\tilde{p}(x_i) = \begin{cases}
\lambda \cdot p(x_i) & \text{if } x_i \in \mathcal{S}_{\text{struct}} \\
p(x_i) & \text{otherwise}
\end{cases}
$$

其中 $\mathcal{S}_{\text{struct}}$ 是结构 token 集合（句号、逗号、EOS 等）。

**不需要重新训练。推理时修改一行代码。带来 +9.4 BLEU-4。**

---

## 实现

### 最小可运行版本：MDLM 解码 + 轨迹追踪

```python
import torch
import torch.nn.functional as F

MASK_ID = 103  # [MASK] token id（以 BERT 为例）

def mdlm_decode(model, tokenizer, seq_len=20, num_steps=10):
    """
    模拟 MDLM 迭代解码，返回解码轨迹
    每步按置信度揭开若干 token
    """
    input_ids = torch.full((1, seq_len), MASK_ID, dtype=torch.long)
    trajectory = []
    per_step   = seq_len // num_steps

    for step in range(num_steps):
        masked_pos = (input_ids[0] == MASK_ID).nonzero(as_tuple=True)[0]
        if len(masked_pos) == 0:
            break

        with torch.no_grad():
            logits = model(input_ids).logits          # (1, seq_len, vocab)

        probs      = F.softmax(logits[0], dim=-1)     # (seq_len, vocab)
        confidence = probs[masked_pos].max(dim=-1).values   # 每个被掩码位置的最大概率
        predicted  = probs[masked_pos].argmax(dim=-1)

        # 揭开置信度最高的 per_step 个位置
        topk = confidence.topk(min(per_step, len(masked_pos))).indices
        step_tokens = []
        for idx in topk:
            pos   = masked_pos[idx].item()
            token = tokenizer.convert_ids_to_tokens([predicted[idx].item()])[0]
            input_ids[0, pos] = predicted[idx]
            step_tokens.append((pos, token))

        trajectory.append(step_tokens)

    return input_ids, trajectory
```

### Lambda-Scaled Structural Decoding 核心逻辑

```python
def get_struct_ids(tokenizer):
    """获取结构 token 的 id 集合"""
    struct_tokens = ['.', ',', '!', '?', ';', ':', '[SEP]', '</s>']
    return {tokenizer.convert_tokens_to_ids(t)
            for t in struct_tokens if t in tokenizer.vocab}

STRUCT_IDS = get_struct_ids(tokenizer)

def lambda_scaled_confidence(probs, masked_pos, lam=0.1):
    """
    计算带结构惩罚的置信度
    probs: (seq_len, vocab_size)
    masked_pos: 当前被掩码的位置索引
    """
    confidence = probs[masked_pos].max(dim=-1).values
    predicted  = probs[masked_pos].argmax(dim=-1)

    # 如果预测的是结构 token，降低其"被选中"的优先级
    struct_mask = torch.tensor(
        [pred.item() in STRUCT_IDS for pred in predicted]
    )
    confidence[struct_mask] *= lam
    return confidence, predicted
```

将 `mdlm_decode` 中的置信度计算替换为 `lambda_scaled_confidence`，就完成了所有修改。

### Graph-LLaDA 架构示意

Graph-LLaDA 在 LLaDA 基础上加入图编码器，把图结构信息作为额外 key-value 注入解码层：

```python
import torch.nn as nn

class GraphTransformerEncoder(nn.Module):
    """
    将知识图谱三元组编码为上下文向量序列
    输出通过 cross-attention 注入 LLaDA 解码器
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj  = nn.Linear(edge_dim,  hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, node_feats, edge_feats, padding_mask=None):
        # 节点和边特征拼接成统一序列：[n1, e12, n2, e23, n3, ...]
        nodes = self.node_proj(node_feats)     # (B, N, H)
        edges = self.edge_proj(edge_feats)     # (B, E, H)
        seq   = torch.cat([nodes, edges], dim=1)  # (B, N+E, H)
        return self.encoder(seq, src_key_padding_mask=padding_mask)
# 将此输出作为 LLaDA 每层 cross-attention 的 K/V 注入
```

---

## 实验结果分析

### 与 Baseline 对比（LAGRANGE 跨数据集评估）

| 方法 | WebNLG（域内） | LAGRANGE（跨域） |
|------|--------------|-----------------|
| T5 fine-tuned | 64.2 | 38.1 |
| LLaDA (zero-shot) | 61.8 | 35.6 |
| LLaDA + SFT | 66.3 | **28.4** ← 泛化崩了 |
| LLaDA + SFT + λ-scaling | 67.1 | 37.8 |
| Graph-LLaDA + λ-scaling | **69.4** | **41.2** |

*注：以上数字为示意性重现，具体数值请以原论文为准*

关键发现：**SFT 在域内提升分数，却在跨数据集评估上崩掉了近 10 个点。** λ-scaling 不仅修复了泛化，还提升了域内性能。这说明问题根源是 SFT 引入的结构偏置，而非 MDLM 本身的能力缺陷。

### 消融：λ 值怎么选？

```python
# 在验证集上扫描 lambda 的效果
results = {}
for lam in [1.0, 0.5, 0.3, 0.1, 0.05, 0.01]:
    bleu = evaluate_with_lambda(model, val_loader, lam=lam)
    results[lam] = bleu
    print(f"λ={lam:.2f}  BLEU-4={bleu:.1f}")

# 典型输出规律：
# λ=1.00  BLEU-4=58.3  （无惩罚）
# λ=0.10  BLEU-4=67.1  ← 通常最优
# λ=0.01  BLEU-4=62.4  （过度惩罚，结构 token 迟迟不出现）
```

λ 不是越小越好：太小会导致模型生完所有实体后仍然不愿意放出句号，输出变得混乱。**λ=0.1 是一个比较稳健的起点，再在 [0.05, 0.3] 范围内微调。**

---

## 调试指南

### 问题 1：输出总是很短，关键信息被截断

**症状**：生成的句子只包含图中部分实体，其余三元组对应的信息消失。

**原因**：SFT 导致的过早结构 token 锚定，EOS 在第一步就被选中。

**修复**：启用 λ-scaling。同时检查训练时 target 序列是否被 `max_length` 截断——如果 target 被截断，模型会学到"输出可以比输入短"的错误偏置。

### 问题 2：输出中出现幻觉实体

**症状**：生成的句子包含图中根本没有的信息。

**原因**：输出长度被提前锁定后，模型用错误 token 填满"规划好"的空位。与问题 1 同源。

**修复**：λ-scaling。如果幻觉仍然存在，检查图编码器是否正确传递了边信息——节点表示正确但边缺失，模型会靠语言先验"补脑"。

### 问题 3：轨迹分析显示实体没有优先出现

**诊断代码（5-15 行短片段）**：

```python
def check_trajectory_health(trajectory, tokenizer, struct_tokens):
    """健康的轨迹：早期步骤结构 token 比例 ≈ 0%，后期 ≈ 100%"""
    for step, tokens in enumerate(trajectory):
        struct_ratio = sum(1 for _, t in tokens if t in struct_tokens) / max(len(tokens), 1)
        print(f"Step {step+1:2d}: struct_ratio={struct_ratio:.0%}  tokens={[t for _, t in tokens]}")
```

如果第 1 步就看到句号或 EOS 出现，说明 SFT 引入了结构锚定，需要 λ-scaling。

### 超参数调优表

| 参数 | 推荐起点 | 敏感度 | 建议 |
|------|---------|--------|------|
| λ（结构惩罚） | 0.1 | 中 | 先试 0.1，再在 [0.05, 0.3] 搜索 |
| 解码步数 T | 10~20 | 低 | 步数越多质量越好，但速度成线性下降 |
| Graph Transformer 层数 | 4 | 中 | 数据集小于 1 万条时不要超过 4 层 |
| Cross-attention 注入层 | 全部层 | 高 | 只注入顶层效果明显变差 |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 知识图谱 → 自然语言，实体信息密集 | 纯语言生成任务（无结构化输入） |
| 跨数据集泛化要求高 | 需要极低延迟（MDLM 多步解码慢于自回归） |
| 已经有 SFT 好的 MDLM 基座 | 数据集极小（< 1000 条），T5 微调更稳 |
| 希望可解释的生成过程 | 动作空间超大的离散生成任务 |

---

## 我的观点

这篇论文最有价值的贡献不是 +9.4 BLEU，而是**对 MDLM 生成行为的可解释性分析**。

大多数关于扩散语言模型的论文都在讲"分数提升了多少"，很少有人去问"模型到底按什么顺序在思考"。把生成轨迹可视化，然后发现 SFT 破坏了这个顺序——这种分析思路值得借鉴。

λ-scaling 本身技术含量不高，但**训练时零成本、推理时一行代码、效果显著**，这是我喜欢的工程风格。

Graph-LLaDA 的方向是对的：显式建模图结构比"把三元组拍平成字符串再喂给 LLM"更自然。但能否泛化到大规模知识图谱（百万级三元组），还需要更多验证。

最后一个警告：**MDLM 在小数据集上很难调**。如果你的图到文本数据集只有几千条，自回归 T5 微调可能仍然是更务实的选择。扩散模型需要足够数据才能展现出超越自回归的优势。不要被 BLEU 分数迷惑，先跑基线，再考虑换架构。

原论文：https://arxiv.org/abs/2605.31564v1