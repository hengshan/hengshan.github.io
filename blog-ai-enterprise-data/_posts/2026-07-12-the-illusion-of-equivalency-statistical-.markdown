---
layout: post-wide
title: '量化模型的"等价幻觉"：为什么困惑度骗了你'
date: 2026-07-12 08:03:46 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.08734v1
generated_by: Claude Code CLI
---

## 一句话总结

Post-training quantization 让模型变小，但准确率不变 ≠ 模型行为没变。论文引入**正确性一致率**（correctness agreement）揭示了被主流评估指标系统性忽略的行为分歧。

---

## 背景：你的量化评估可能是错的

你量化了一个模型，发现：

- 准确率从 72.3% → 72.1%，差 0.2%，可接受
- 困惑度从 8.2 → 8.4，几乎没变

结论："量化后模型等价于原始模型"。

这个结论可能是错的。

准确率和困惑度是**群体统计量**——它们反映整体分布是否接近，但不回答：

> 对于同一道题，原始模型答对了，量化模型是否也答对了？

这就是论文的核心观察：**等价幻觉**。两个模型可以在群体层面统计相似，但在决策层面却完全不同。

论文的主要贡献：
1. 提出 correctness agreement 作为决策级评估指标
2. 将量化建模为注意力权重上的**结构算子**，定量分析逐层失真
3. 发现 Q/K 投影比 V/O 投影对量化显著更敏感
4. 找到低比特宽度下的非线性失效断点（breakpoint）

---

## 核心概念

### 直觉类比

想象两个学生各得 60 分（满分 100）：

- 学生 A（原始模型）：第 1-60 题答对
- 学生 B（量化模型）：第 41-100 题答对

两人分数相同，但重叠的答对题只有 20 题。这两个学生有相同的"知识"吗？

### Correctness Agreement 的数学定义

设 $M$ 是基础模型，$M_q$ 是量化版本，在测试集 $\mathcal{D}$ 上评估：

$$
\text{CA}(M, M_q) = \frac{|\{x \in \mathcal{D} : \text{correct}(M, x) = \text{correct}(M_q, x)\}|}{|\mathcal{D}|}
$$

这个指标独立于绝对准确率：即使两个模型准确率相同，CA 也可以很低。

### 注意力权重失真

论文将量化视为权重上的加性噪声算子：

$$
\hat{W} = Q(W) = W + \Delta W
$$

用相对 Frobenius 范数定量逐层失真：

$$
D_{\text{layer}} = \frac{\|W - Q(W)\|_F}{\|W\|_F}
$$

关键发现：**Q（query）和 K（key）投影对量化噪声的敏感度系统性地高于 V（value）和 O（output）投影。** 这在直觉上合理——Q/K 的乘积决定注意力分配模式，细微扰动可以改变"看哪里"，而 V 的失真只影响"看到什么"。

---

## 实现

### 最小可运行版本：计算 Correctness Agreement

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_predictions(model, tokenizer, dataset, device="cuda"):
    """在多选题上获取模型对每道题的对/错标记"""
    model.eval()
    correct_flags = []

    for item in dataset:
        question = item["question"]
        choices = item["choices"]["text"]
        answer_idx = ord(item["answerKey"]) - ord("A")

        scores = []
        for choice in choices:
            prompt = f"Question: {question}\nAnswer: {choice}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                # 用负 loss（= log 概率之和）给每个选项打分
                loss = model(**inputs, labels=inputs["input_ids"]).loss
            scores.append(-loss.item())

        pred = int(torch.tensor(scores).argmax())
        correct_flags.append(pred == answer_idx)

    return correct_flags

def correctness_agreement(flags_base, flags_quant):
    """计算两模型的决策一致率，独立于绝对准确率"""
    n = len(flags_base)
    agree = sum(a == b for a, b in zip(flags_base, flags_quant))
    return agree / n
```

### 完整实现：逐层失真分析

```python
import matplotlib.pyplot as plt

def compute_layer_distortion(base_model, quant_model, layer_idx):
    """计算单层 Q/K/V/O 投影的相对量化失真"""
    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    base_attn = base_model.model.layers[layer_idx].self_attn
    quant_attn = quant_model.model.layers[layer_idx].self_attn

    results = {}
    for name in proj_names:
        W_base = getattr(base_attn, name).weight.float()
        # 4-bit 量化层需要先 dequantize 才能做差
        W_quant = getattr(quant_attn, name).weight.float()
        delta = W_base - W_quant
        results[name] = (delta.norm("fro") / W_base.norm("fro")).item()

    return results

def analyze_all_layers(base_model, quant_model):
    """汇总所有层的失真，返回按投影类型分组的时序列表"""
    n_layers = len(base_model.model.layers)
    layer_data = {p: [] for p in ["q_proj", "k_proj", "v_proj", "o_proj"]}

    for i in range(n_layers):
        dist = compute_layer_distortion(base_model, quant_model, i)
        for proj, val in dist.items():
            layer_data[proj].append(val)

    return layer_data

def plot_distortion(layer_data, save_path="distortion.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"q_proj": "red", "k_proj": "orange",
               "v_proj": "blue", "o_proj": "green"}
    for proj, vals in layer_data.items():
        ax.plot(vals, label=proj, color=colors[proj], alpha=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Relative Frobenius Distortion")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
```

### 关键 Trick

**不要用 logit 打分，用负 loss：**

```python
# 错误：logit 对不同长度的 completion 不可比
logits = model(**inputs).logits[:, -1, :]

# 正确：整个序列的平均 NLL 作为分数
loss = model(**inputs, labels=inputs["input_ids"]).loss
score = -loss.item()
```

**加载量化模型（bitsandbytes）：**

```python
from transformers import BitsAndBytesConfig

config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,   # 对量化参数本身再量化
)
quant_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=config_4bit, device_map="auto"
)
```

---

## 实验

### 用 ARC-Challenge 快速验证

ARC-Challenge 是好的测试床：题目够难，随机猜测基线低（25%），量化导致的行为漂移会被放大。

```python
from datasets import load_dataset

dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
# 先跑基础模型，缓存结果
flags_base = get_predictions(base_model, tokenizer, dataset)

for bits in [8, 4, 2]:
    q_model = load_quantized(model_name, bits=bits)  # 用前面的 BitsAndBytesConfig
    flags_q = get_predictions(q_model, tokenizer, dataset)
    ca = correctness_agreement(flags_base, flags_q)
    acc_q = sum(flags_q) / len(flags_q)
    print(f"{bits}-bit | acc={acc_q:.3f} | CA={ca:.3f}")
```

### 预期结果

| 指标 | 8-bit | 4-bit | 2-bit |
|------|-------|-------|-------|
| 准确率下降 | < 0.5% | ~1-2% | ~10%+ |
| Correctness Agreement | ~95% | ~80-85% | < 60% |
| Q/K 相对失真 | 低 | 中 | 非线性激增 |

**关键观察：4-bit 量化看起来"安全"，但 CA 已经下降到约 80%。** 这意味着每 5 道题里就有 1 道的对/错发生翻转——哪些题翻转了，你从准确率上完全看不出来。

---

## 调试指南

### 常见问题

**1. CA 计算结果异常低（< 60%），即使 8-bit**

先排查：两个模型是否加载在相同精度？padding 策略是否一致？用一道题手动对比两个模型的 loss 数值——如果差距超过 0.5，说明数值环境不对，而不是量化问题。

**2. 失真图中某层出现孤立尖峰**

这是**离群权重**（outlier weights）的标志，是量化的已知痛点。孤立尖峰层往往是模型中对量化最敏感的瓶颈。考虑 SmoothQuant 或 AWQ——它们的核心思想就是在量化前先将离群值"迁移"到激活层。

**3. V/O 层失真比 Q/K 高**

不同架构（MLA、GQA、标准 MHA）的敏感度分布不同。论文结论基于 Llama 系列，在 Qwen、Phi、Mixtral 上可能有差异。记录下来，这是有价值的 negative result，不是错误。

### 更完整的安全性报告

```python
def quantization_safety_report(base_flags, quant_flags):
    ca = correctness_agreement(base_flags, quant_flags)
    n = len(base_flags)
    # 原本对、量化后错：真正的"损失"
    flip_bad = sum(b and not q for b, q in zip(base_flags, quant_flags)) / n
    # 原本错、量化后对：侥幸"增益"（不可依赖）
    flip_good = sum(not b and q for b, q in zip(base_flags, quant_flags)) / n

    print(f"Correctness Agreement : {ca:.3f}")
    print(f"Flip-to-wrong rate    : {flip_bad:.3f}")
    print(f"Flip-to-correct rate  : {flip_good:.3f}")

    if ca > 0.90 and flip_bad < 0.05:
        print("=> 量化较安全")
    elif ca > 0.80:
        print("=> 中等风险，建议任务专项评估")
    else:
        print("=> 高风险，行为已显著改变")
```

### 量化方案对 CA 的影响参考

| 量化方案 | CA 影响 | 适用场景 |
|---------|---------|---------|
| INT8 (LLM.int8) | 极小 | 生产部署，内存受限 |
| NF4 (QLoRA) | 小-中 | 微调阶段 |
| GPTQ 4-bit | 中 | 推理加速，需校准数据 |
| GGUF Q2_K | 大 | 极端压缩，行为改变显著 |

---

## 什么时候在意 CA？

| 需要关注 CA | 可以忽略 CA |
|-----------|-----------|
| 安全关键应用 | 创意写作、摘要生成 |
| 多步推理（Chain-of-Thought） | 快速 demo 原型 |
| 知识密集型 QA | 粗粒度分类 |
| 部署多个量化版本需对比 | 单次生成评估 |

特别提醒：如果你的应用依赖 CoT 推理，CA 下降比准确率下降更危险——错误发生在推理链的中间步骤，最终答案不一定暴露问题。

---

## 我的观点

这篇论文做的是**"提出更好的评估指标"**，不是提出更好的量化方法。这类工作往往被低估，但实际上很重要。

值得采纳的部分：CA 确实捕捉到了准确率和困惑度遗漏的信息，实现成本低，可以直接加到现有的评估 pipeline 里，不需要改任何训练代码。

需要警惕的部分：论文实验以 Llama 系列为主，Q/K 比 V/O 更敏感的结论在其他架构上未必成立。如果你的模型用了 GQA（Llama 3 以后标配）或 MLA（DeepSeek），需要验证。

实际建议：在生产中部署量化模型之前，跑一次 CA 评估——只需要测试集和两个模型，几小时内可以出结果。4-bit 量化下"准确率没变"和"行为没变"是两件事，这个 gap 在你的任务上可能无关紧要，也可能很重要，但你得先测了才知道。不测就上线，是在赌运气。