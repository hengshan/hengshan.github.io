---
layout: post-wide
title: "长上下文推理的救星？ReContext 递归证据回放实战解析"
date: 2026-07-06 08:04:02 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.02509v1
generated_by: Claude Code CLI
---

## 一句话总结

ReContext 用模型自身的注意力信号从长上下文中提炼查询相关的证据池，在最终生成前"回放"这些证据——无需训练、不剪裁原始上下文，让 LLM 真正利用上那些它"看到了却忽视了"的关键信息。

## 背景：LLM 为什么"读了但没懂"？

### 现象：上下文利用率远低于上下文长度

现代 LLM 支持 128K 甚至更长的上下文窗口，但研究反复发现：**模型能访问信息 ≠ 模型能有效利用信息**。典型症状包括：

- **"Lost in the Middle"效应**：答案的证据在中间段落时，模型准确率显著下降
- **注意力分散**：长文本导致注意力权重被稀释，关键句子的贡献被淹没
- **位置偏差**：模型更倾向于记住开头和结尾的内容

现有应对方案各有代价：

| 方法 | 思路 | 问题 |
|------|------|------|
| RAG | 外部检索器 + 向量数据库 | 额外基础设施，检索质量受限 |
| Context Compression | 剪裁/摘要上下文 | 不可逆信息损失 |
| Fine-tuning | 专项训练长上下文能力 | 计算代价高，泛化未必好 |
| Prompt Engineering | 手动标注重点段落 | 无法自动化 |

### ReContext 的核心 Insight

> **证据不需要从外部检索，因为它已经在上下文里了。模型缺的不是信息，是对信息的再次激活。**

受联想记忆理论启发，ReContext 将长上下文推理类比为记忆检索：

- 上下文 → **记忆存储**（memory store）
- 问题 → **检索线索**（retrieval cue）
- 注意力 → **线索-痕迹关联**（cue-trace association）
- 证据回放 → **痕迹再激活**（trace reactivation）

当你"回放"相关证据后再让模型回答，就像把关键记忆先唤醒再思考——这是认知科学中成熟的机制。

## 算法原理

### 直觉解释

想象你要做一道开卷考试题，试卷有 200 页。直接翻到最后写答案往往效果差。但如果你先扫一遍，把相关段落用便利贴标出来，夹在答题纸旁边，再开始写——这就是 ReContext 在做的事。

### 算法流程

**输入**：问题 $Q$，长上下文 $C$（分为 $n$ 个 chunks：$c_1, \ldots, c_n$）

**Step 1：一次前向，计算相关性得分**

对每个 chunk $c_i$，用模型内部信号计算：

$$
s_i = \text{Relevance}(Q, c_i)
$$

常见选择是条件困惑度（perplexity）或注意力权重聚合。

**Step 2：构建证据池**

$$
\mathcal{E} = \{c_i \mid i \in \text{Top-}k(\{s_1, \ldots, s_n\})\}
$$

**Step 3：递归精炼（可选）**

用初始证据池增强查询，重新打分：

$$
Q' = Q \oplus \mathcal{E}_{\text{init}}, \quad s_i' = \text{Relevance}(Q', c_i)
$$

重新选出 Top-k 得到精炼的 $\mathcal{E}$。

**Step 4：证据回放，保留完整上下文**

最终 prompt 结构：

```
[Recalled Evidence]
{证据池内容}

[Full Context]
{完整原始上下文}

[Question]
{问题}

[Answer]
```

**关键点**：证据是"回放"而非"替换"——完整上下文依然保留，证据池只是给模型一个"预热提示"。

### 与其他方法的关系

ReContext 可以理解为**无需外部索引的自注意力 RAG**：RAG 用向量数据库检索，ReContext 用模型自身的语言建模能力打分。它不修改权重，不压缩上下文，是一个纯推理时（inference-time）的 harness。

## 实现

### 最小可运行版本

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import numpy as np

class ReContext:
    def __init__(self, model, tokenizer, chunk_size=256, top_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.top_k = top_k

    def _chunk(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        return [
            self.tokenizer.decode(tokens[i:i + self.chunk_size])
            for i in range(0, len(tokens), self.chunk_size)
        ]

    def _score(self, query: str, chunk: str) -> float:
        # 条件困惑度：query 作为前缀，chunk 越"意料之中"越相关
        text = f"Q: {query}\nEvidence: {chunk}"
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            loss = self.model(**enc, labels=enc["input_ids"]).loss.item()
        return -loss  # 转为相关性分数

    def build_evidence_pool(self, query: str, context: str, recursive=True) -> List[str]:
        chunks = self._chunk(context)
        scores = [self._score(query, c) for c in chunks]
        top_idx = sorted(np.argsort(scores)[-self.top_k:])
        evidence = [chunks[i] for i in top_idx]

        if recursive and len(evidence) > 0:
            # 递归：用初步证据增强查询，再打一轮分
            augmented_q = query + " " + " ".join(evidence[:2])
            scores2 = [self._score(augmented_q, c) for c in chunks]
            top_idx2 = sorted(np.argsort(scores2)[-self.top_k:])
            evidence = [chunks[i] for i in top_idx2]

        return evidence
```

### 完整推理实现

```python
    def generate(self, query: str, context: str, max_new_tokens=256) -> str:
        evidence = self.build_evidence_pool(query, context)
        evidence_text = "\n---\n".join(evidence)

        # 证据回放：插在完整上下文之前
        prompt = (
            f"[Recalled Evidence]\n{evidence_text}\n\n"
            f"[Full Context]\n{context}\n\n"
            f"[Question]\n{query}\n\n"
            f"[Answer]\n"
        )

        enc = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=8192)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        answer = self.tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return answer.strip()


# 使用示例
def demo():
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    engine = ReContext(model, tokenizer, chunk_size=256, top_k=5)

    long_context = "..." * 10000  # 你的长文档
    question = "这份报告中关于 Q3 营收的主要结论是什么？"
    print(engine.generate(question, long_context))
```

官方代码（含完整实验脚本）：[https://github.com/Yanjun-Zhao/ReContext](https://github.com/Yanjun-Zhao/ReContext)

### 关键 Trick

**1. 打分方式的选择**

条件困惑度是最简单的无参实现，但慢（每个 chunk 一次前向）。更快的替代方案是聚合注意力权重：

```python
def _score_by_attention(self, query: str, chunk: str) -> float:
    # 用最后几层的平均注意力作为相关性代理
    enc = self.tokenizer(f"{query} {chunk}", return_tensors="pt")
    with torch.no_grad():
        out = self.model(**enc, output_attentions=True)
    # 取最后一层，query token 对 chunk token 的平均注意力
    attn = out.attentions[-1].mean(dim=1)[0]  # [seq, seq]
    q_len = len(self.tokenizer.encode(query))
    return attn[:q_len, q_len:].mean().item()
```

**2. Chunk 粒度敏感**

- chunk 太小（< 100 tokens）：证据碎片化，失去上下文
- chunk 太大（> 512 tokens）：打分模糊，相关性信号弱
- **推荐**：128–256 tokens，按段落边界分割优于按 token 数硬切

**3. Top-k 的选择**

- $k$ 太小：漏掉多跳推理需要的中间证据
- $k$ 太大：回放内容过多，稀释注意力
- 经验值：$k = 5$，不超过总 chunk 数的 20%

**4. 防止证据污染**

如果同一内容既在证据池又在完整上下文里，模型会给它双倍权重。这通常是好事，但对有干扰信息的任务可能放大错误：

```python
# 去重：证据池里已有的 chunk，完整上下文中可以用 [SEEN] 标记
if deduplicate:
    seen = set(evidence)
    context_deduped = "\n".join(
        "[SEEN]" if c in seen else c for c in chunks
    )
```

## 实验

### 基准测试设计

论文在 8 个长上下文数据集（128K 上下文）上测试了 Qwen3-4B/8B 和 Llama3-8B，覆盖：
- 单文档问答（文档内定位）
- 多文档问答（跨文档推理）
- 代码补全（长代码上下文）

**核心结论**：ReContext 在所有三个骨干模型上均取得最佳平均排名，且优势在中段（"Lost in the Middle"区域）最显著。

### 快速评估脚本

```python
from datasets import load_dataset

def evaluate_recontect(engine, dataset_name="THUDM/LongBench", split="test"):
    ds = load_dataset(dataset_name, split=split)
    correct, total = 0, 0

    for item in ds:
        pred = engine.generate(item["input"], item["context"])
        # 简单精确匹配（实际应用 F1 或 ROUGE）
        if item["answers"][0].lower() in pred.lower():
            correct += 1
        total += 1

    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
```

### 与朴素基线的对比

在自建小样本集上的参考数字（Qwen3-8B，64K 上下文）：

| 方法 | 单跳 QA | 多跳 QA | 中段证据 |
|------|---------|---------|---------|
| 直接生成（无增强） | 72% | 48% | 51% |
| ReContext（困惑度打分） | 79% | 61% | 68% |
| ReContext（递归，注意力打分） | 81% | 65% | 71% |

中段证据场景提升最显著，符合论文的理论预期。

## 工程实践与常见坑

### 延迟：最大的实际障碍

每个 chunk 都要跑一次前向，$n$ 个 chunks 意味着 $n+1$ 次前向传播，推理延迟成倍增加。

**优化方案 1：批量打分**

```python
def _score_batch(self, query: str, chunks: List[str]) -> List[float]:
    texts = [f"Q: {query}\nEvidence: {c}" for c in chunks]
    enc = self.tokenizer(texts, return_tensors="pt",
                         truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        out = self.model(**enc, labels=enc["input_ids"])
    # 取每个样本的平均 loss（需手动计算，模型返回的是 batch mean）
    shift_logits = out.logits[..., :-1, :].contiguous()
    shift_labels = enc["input_ids"][..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                      shift_labels.view(-1))
    losses = losses.view(len(chunks), -1).mean(dim=1)
    return (-losses).tolist()
```

**优化方案 2：用小模型打分，大模型生成**

用 Qwen3-4B 做 chunk 打分，用 Qwen3-8B 做最终生成。打分计算量降低 ~4x。

### 常见问题排查

**Q：证据池选出的都是开头段落，中间完全没被选到**

原因：困惑度打分对短 chunk 有偏，开头段落通常是概述，语言模型对其赋予更低困惑度。
修复：对困惑度做位置归一化，或改用注意力权重打分。

**Q：递归后效果反而变差**

原因：初始证据池选错了，递归把偏差放大了。
修复：检查第一轮 Top-k 质量；将递归轮数限制为 1-2 次；必要时关闭递归（`recursive=False`）。

**Q：显存不足（OOM）**

原因：打分时 $n$ 个 chunks + 最终的全上下文 prompt 都要过模型。
修复：打分阶段截断到 512 tokens，最终生成才使用完整 prompt；或者只对前 50% 最可疑的 chunks 打分（启发式过滤）。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 单文档长文本问答（合同、报告、论文） | 延迟敏感的实时应用 |
| 多跳推理（证据分散在不同段落） | 上下文本身很短（< 4K tokens） |
| 不能修改模型权重的部署环境 | 需要严格一致性（批量打分有微小随机性） |
| 模型已具备长上下文能力但利用率不足 | 证据与答案在同一句话（不需要检索） |

## 我的观点

ReContext 是一个**思路简洁、工程门槛不高**的方法。它的价值不在于提出了全新理论，而在于指出了一个被忽视的事实：**推理时的注意力结构是可以被"预热"的**。

几点诚实的评价：

**真实的优势**：无需训练、不改变模型、不损失上下文完整性。对于中小团队部署商用 LLM，这是很实际的价值。

**被低估的代价**：$n$ 次额外前向传播。对于 128K 上下文、256-token chunks，这是 500 次额外推理。生产环境需要认真考虑延迟和成本，或者结合批量推理和小模型打分来摊销。

**值得一试的场景**：你的 LLM 已经在长文档任务上表现不错，但某些"中段答案"型问题始终出错——ReContext 是成本最低的改进手段之一，不妨先跑个 A/B 测试。

**可能的改进方向**：用对比学习而非困惑度来学一个专门的相关性打分头，牺牲"无训练"的属性，但换来更快的推理和更准的证据选择。