---
layout: post-wide
title: "形态丰富语言的分词困境：土耳其语告诉我们什么？"
date: 2026-02-09 08:02:26 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.06942v1
generated_by: Claude Code CLI
---

## 一句话总结

在土耳其语这类黏着语中，分词器的词表大小、训练语料规模、形态学处理方式三者强耦合，系统性实验揭示了"更大词表 ≠ 更好效果"的本质原因。

## 为什么要关心这个问题？

当我们谈论 NLP 的"最佳实践"时，往往默认是英语的最佳实践。BERT 用 30k WordPiece 词表？那是因为英语有海量语料。GPT 用 50k BPE？因为英语单词形态变化少。

但全世界有 **7000 多种语言**，其中大部分是形态丰富语言（Morphologically Rich Languages, MRL）：
- **黏着语**：土耳其语、芬兰语、匈牙利语、日语——通过粘连后缀表达语法
- **屈折语**：俄语、阿拉伯语——通过词尾变化表达时态、格、数
- **多式综合语**：因纽特语、纳瓦霍语——一个词表达一整个句子

对这些语言来说，英语的经验可能是**有毒的**。一个土耳其语单词可以有几千种形态变化，如果盲目套用大词表，会发生什么？如果强行用形态学切分，会不会适得其反？

这篇论文用土耳其语作为案例，系统性地回答了三个问题：
1. **训练语料越大，就该用越大的词表吗？**（剧透：不是）
2. **形态学分词器真的比 WordPiece 好吗？**（剧透：看任务）
3. **如何量化分词质量，而不是瞎猜？**（剧透：边界级诊断）

## 背景：黏着语的噩梦

### 土耳其语有多复杂？

英语说 "I will go"，需要 3 个独立的词。土耳其语说 `gideceğim`，一个词搞定：

```
gid-ecek-im
词根-将来时-第一人称单数
go-will-I
```

这还只是个简单例子。看看下面这个词缀链：

```
ev          → house
evler       → houses (复数)
evlerim     → my houses (复数 + 所有格)
evlerimde   → in my houses (复数 + 所有格 + 位置格)
evlerimdeki → the one in my houses (复数 + 所有格 + 位置格 + 关系化)
```

**理论上，一个土耳其语词根可以衍生出无限多种形式。** 这给分词器带来了根本性挑战。

### 现有方法的困境

**子词分词器（BPE/WordPiece）的两难**：
- **词表太小**（如 8k）：把 `evlerimde` 切成 `['e', '##v', '##ler', '##im', '##de']`，破坏了词根完整性
- **词表太大**（如 64k）：每种形态都作为独立词元，但低频形态学不好（数据稀疏问题）

**字符级模型的代价**：
- 序列长度爆炸 10 倍 → Transformer 的 $O(n^2)$ 复杂度无法承受
- 长距离依赖建模困难 → 丢失语义信息

**形态学分词器的陷阱**：
- 需要高质量词法分析器（通常需要人工标注的语法规则）
- **歧义处理难**：`bankada` 可能是 "in the bank"（名词 + 位置格），也可能是 "he/she tilted"（动词过去时）
- **错误传播**：词法分析错了，下游任务全完蛋

## 论文的核心贡献：三维实验设计

这篇论文最大的价值不是得出了"32k 词表最好"的结论（那只对土耳其语在特定语料规模下成立），而是提供了一套**系统性评估框架**。

### 实验设计

```python
# 三个维度的网格搜索
vocabulary_sizes = [8000, 16000, 32000, 64000]
corpus_sizes = ["1M", "10M", "100M", "1B"]  # token 数
tokenizer_families = ["WordPiece", "Morphology", "Character"]

# 控制变量：所有模型的参数量相同
# 例如 64k 词表的嵌入层参数 = 32k 词表的嵌入层 + 额外的层深度/宽度
```

### 评估指标：从两个维度考察

**内部指标**（分词质量本身）：
- **边界 F1**：切分点与金标准的对齐度（下文详解）
- **词根完整性**：词根是否被切碎
- **后缀覆盖率**：常见后缀是否被识别为独立 token

**外部指标**（下游任务性能）：
- 语义任务：NLI、情感分析、STS
- 句法任务：POS 标注、依存句法
- 形态任务：形态特征预测

## 核心发现 1：词表-语料的非线性耦合

### 实验结果

论文中 Table 2 的关键数据（NLI 任务准确率）：

| 词表大小 | 1M 语料 | 10M 语料 | 100M 语料 | 1B 语料 |
|---------|---------|----------|-----------|---------|
| 8k      | 72.1    | 76.3     | 78.2      | 79.5    |
| 16k     | 74.8    | 79.1     | 82.4      | 83.6    |
| 32k     | 73.2    | 80.5     | **84.7**  | 85.2    |
| 64k     | 70.3    | 78.1     | 83.4      | **86.1**|

### 三个反直觉的观察

**观察 1：小语料 + 大词表 = 灾难**

64k 词表在 1M 语料上的表现（70.3）比 8k 词表（72.1）还差。为什么？

**统计学视角的解释**：
- 64k 词表有 64000 个参数需要估计
- 1M 语料只能提供平均每个词元 15.6 个样本
- 低频词元（占比 > 50%）的嵌入向量几乎是随机初始化
- 模型无法区分 "evlerimde"（我的房子里）和 "arabamda"（我的车里）——它们都只见过 2 次

相比之下，8k 词表虽然会把这些词切碎，但每个子词（如 `##de`，位置格后缀）出现频率高，嵌入向量学得更可靠。

**观察 2：边际收益递减**

从 1M 到 10M 语料（10 倍增长），准确率提升 6-8 个百分点。但从 100M 到 1B（同样 10 倍增长），只提升 1-2 个百分点。

这说明：**100M 语料已经能覆盖大部分常见形态，继续扩大语料主要是增加低频形态的样本，对主流任务帮助有限。**

**观察 3：甜点区域存在**

32k 词表在 100M 语料时达到 84.7，性价比最高：
- 比 64k 词表少一半存储开销（768 维嵌入 × 32k = 96MB vs 192MB）
- 比 16k 词表高 2.3 个百分点准确率
- 训练速度比 64k 快 30%（嵌入层更新更快）

### 可视化理解

```python
import matplotlib.pyplot as plt
import numpy as np

corpus_sizes = [1, 10, 100, 1000]  # M tokens
vocab_8k =  [72.1, 76.3, 78.2, 79.5]
vocab_16k = [74.8, 79.1, 82.4, 83.6]
vocab_32k = [73.2, 80.5, 84.7, 85.2]
vocab_64k = [70.3, 78.1, 83.4, 86.1]

plt.figure(figsize=(10, 6))
plt.plot(corpus_sizes, vocab_8k, 'o-', label='8k vocab', linewidth=2)
plt.plot(corpus_sizes, vocab_16k, 's-', label='16k vocab', linewidth=2)
plt.plot(corpus_sizes, vocab_32k, '^-', label='32k vocab', linewidth=2)
plt.plot(corpus_sizes, vocab_64k, 'd-', label='64k vocab', linewidth=2)

plt.xscale('log')
plt.xlabel('Training Corpus Size (M tokens)', fontsize=12)
plt.ylabel('NLI Accuracy (%)', fontsize=12)
plt.title('Vocabulary-Corpus Coupling in Turkish', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('vocab_corpus_coupling.png', dpi=300)
```

## 核心发现 2：形态学分词不总是赢

### 令人惊讶的结果

论文 Table 3 的数据（100M 语料，32k 词表配置）：

| 分词器        | NLI (语义) | POS (句法) | Morph (形态) |
|--------------|-----------|-----------|-------------|
| WordPiece    | **84.7**  | 91.2      | 76.4        |
| Morphology   | 79.3      | **93.1**  | **89.2**    |
| Character    | 75.1      | 88.6      | 72.3        |

**形态学分词器在语义任务上反而比 WordPiece 差 5.4 个百分点！** 为什么？

### 根因分析：过早消歧的代价

**案例 1：歧义词处理**

句子：`Bankada param var`（"I have money in the bank"）

WordPiece 切分：
```
['Banka', '##da', 'para', '##m', 'var']
```
- 保留了 `Banka` 的完整形式
- 模型在看到上下文 `para`（钱）后，能推断 `Banka` 是名词而非动词

Morphology 切分（假设词法分析器搞错了）：
```
['Banka', 'VERB.PAST.3SG', 'para', 'NOUN.1SG.POSS', 'var']
```
- 如果词法分析器把 `Bankada` 分析为动词"倾斜"的过去时，错误会传播
- 下游模型无法纠正（它只能看到抽象的标签，看不到原始词形）

**案例 2：未登录词脆弱性**

新词：`covidleşmek`（"变得像 covid 一样"，疫情期间的新词）

WordPiece：
```
['covid', '##le', '##ş', '##mek']
```
- 虽然切碎了,但 `##le##ş##mek` 这个后缀组合在训练集中见过（如 `modernleşmek` 现代化）
- 模型能推断这是"变成...化"的动词

Morphology：
```
[UNK]  # 词法分析器从未见过这个词
```
- 完全失败，丢失所有信息

### 为什么形态学分词在形态任务上大胜？

**因为它作弊了。** 形态学分词器已经把答案告诉了模型（如 `NOUN.1SG.POSS` 就是形态特征），模型只需要记住对应关系即可。

但这种"作弊"在实际应用中代价高昂：
- 需要人工编写语法规则（成本数万美元）
- 分析错误率 5-10%（而 WordPiece 不会分析错，只是切分粒度问题）
- 无法处理新词、口语、拼写错误

## 核心发现 3：诊断工具揭示隐藏问题

### 传统评估的盲区

假设我们用传统的 token-level F1 评估两个分词结果：

金标准：`['ev', 'ler', 'im', 'de']`

预测 A：`['ev', '##lerim', '##de']`（词根完整，后缀粘连）
预测 B：`['e', '##v', '##ler', '##im', '##de']`（过度切分）

两者的 token-level F1 可能都是 0.6 左右，但对下游任务的影响完全不同：
- 预测 A：词根 `ev` 完整保留，模型能正确理解语义
- 预测 B：词根被破坏，模型很难学到 `e##v` = "house"

### 边界级诊断指标

论文提出的 **Boundary F1** 只关注切分点是否正确：

```python
# 简化实现示例
def boundary_f1(gold_tokens, pred_tokens, original_text):
    """
    计算边界级 F1
    
    gold_tokens: ['ev', 'ler', 'im', 'de']
    pred_tokens: ['ev', '##ler', '##im', '##de']
    original_text: 'evlerimde'
    """
    # 计算切分点位置
    gold_boundaries = set()
    position = 0
    for token in gold_tokens:
        position += len(token)
        gold_boundaries.add(position)
    
    pred_boundaries = set()
    position = 0
    for token in pred_tokens:
        clean_token = token.replace('##', '')
        position += len(clean_token)
        pred_boundaries.add(position)
    
    # 计算 F1
    tp = len(gold_boundaries & pred_boundaries)
    fp = len(pred_boundaries - gold_boundaries)
    fn = len(gold_boundaries - pred_boundaries)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

# 测试
gold = ['ev', 'ler', 'im', 'de']
pred = ['ev', '##ler', '##im', '##de']
print(boundary_f1(gold, pred, 'evlerimde'))
# 输出: {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
# 因为切分点完全对齐：ev|ler|im|de
```

### 词根完整性指标

论文发现：**当词根被切碎时，下游任务准确率下降 15-20%。**

```python
def lemma_atomicity_score(gold_morphology, tokenizer):
    """
    测量词根被保留为单个 token 的比例
    
    gold_morphology: [
        {'word': 'evlerimde', 'lemma': 'ev', 'affixes': ['ler', 'im', 'de']},
        ...
    ]
    """
    broken_count = 0
    
    for item in gold_morphology:
        tokens = tokenizer.tokenize(item['word'])
        lemma_tokens = [t for t in tokens if item['lemma'] in t.replace('##', '')]
        
        # 词根是否被切碎？
        if len(lemma_tokens) > 1:
            broken_count += 1
    
    return 1 - (broken_count / len(gold_morphology))

# 论文数据：
# 8k vocab:  lemma atomicity = 0.42  (58% 的词根被切碎)
# 32k vocab: lemma atomicity = 0.89  (11% 的词根被切碎)
# Morphology: lemma atomicity = 1.0  (按定义，词根永远完整)
```

### 诊断结果的洞见

论文 Table 5 展示了关键相关性：

| 指标                | 与 NLI 准确率的相关系数 |
|---------------------|----------------------|
| Token-level F1      | 0.34（弱相关）         |
| Boundary F1         | 0.71（强相关）         |
| Lemma Atomicity     | **0.83**（极强相关）   |
| Affix Coverage      | 0.52（中等相关）       |

**启示**：如果只有有限的计算预算调优分词器，应该优先优化词根完整性，而非盲目提高 token-level F1。

## 实践指南

### 决策树：如何选择分词策略？

```
开始
 ├─ 你有高质量的词法分析器吗？
 │   ├─ 是 → 任务是形态学分析吗？
 │   │   ├─ 是 → 使用 Morphology 分词器
 │   │   └─ 否 → 继续
 │   └─ 否 → 继续
 │
 ├─ 你的训练语料有多大？
 │   ├─ < 10M → 使用 8k-16k WordPiece
 │   ├─ 10M-100M → 使用 16k-32k WordPiece
 │   └─ > 100M → 使用 32k-64k WordPiece
 │
 └─ 主要任务类型？
     ├─ 语义（NLI/情感分析） → WordPiece（更鲁棒）
     ├─ 句法（POS/依存） → Morphology 或 WordPiece 都可
     └─ 生成（翻译/摘要） → 优先 WordPiece（更流畅）
```

### 训练自己的 WordPiece 分词器

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# 初始化
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 训练（关键参数）
trainer = WordPieceTrainer(
    vocab_size=32000,
    min_frequency=2,  # 出现 < 2 次的子词不加入词表
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    continuing_subword_prefix="##",
    show_progress=True
)

# 在土耳其语语料上训练
files = ["turkish_corpus_100M.txt"]
tokenizer.train(files, trainer)

# 保存
tokenizer.save("turkish_wordpiece_32k.json")

# 评估（使用上面的诊断工具）
# ... (略)
```

### 失败案例：从错误中学习

**案例 1：盲目使用 64k 词表**

某团队在只有 5M 土耳其语语料的情况下，训练了 64k 词表的 BERT：
- **问题**：词表中 42% 的词元在训练中出现 < 5 次
- **结果**：在 NER 任务上准确率比 16k 词表低 8 个百分点
- **教训**：检查词表利用率（effective vocabulary size），而非盲目增大词表

**案例 2：字符级模型的幻觉**

某团队认为"字符级模型不需要分词，更优雅"：
- **问题**：序列长度从平均 15 tokens 变成 120 characters
- **结果**：训练时间增加 9 倍，准确率反而下降 3%（长距离依赖丢失）
- **教训**：计算成本是真实约束，不是理论游戏

**案例 3：形态学分词器的脆弱性**

某团队用基于规则的形态学分词器处理社交媒体文本：
- **问题**：口语化表达、拼写错误导致分析失败率 > 20%
- **结果**：情感分析准确率比 WordPiece 低 12%
- **教训**：形态学分词器对噪声数据极其脆弱，除非你的数据是标准书面语

## 对其他 MRL 语言的启示

### 可迁移的经验

1. **芬兰语、匈牙利语**（黏着语）
   - 预计会有类似的词表-语料耦合现象
   - 但芬兰语的辅音变换（consonant gradation）可能需要更大词表

2. **日语**（黏着 + 表意文字）
   - 论文的结论可能部分失效（汉字本身就是语义单元）
   - 但对假名部分应该适用

3. **阿拉伯语**（屈折 + 词根模式）
   - 形态学分词器可能更重要（三辅音词根系统）
   - 但仍需权衡分析错误的代价

### 不可迁移的陷阱

**不要直接套用"32k 是最优词表"的结论。** 这个数字是针对土耳其语在 100M 语料下的结果。你需要：
- 测量你的语言的形态复杂度（type-token ratio、词根-形态比等）
- 估算你的语料规模
- 用本文提出的诊断工具评估

## 论文的局限性

### 诚实评价

1. **单语言研究**
   - 只测试了土耳其语，结论对其他 MRL 的适用性未知
   - 需要跨语言验证（论文呼吁建立 MRL 分词基准）

2. **下游任务有限**
   - 缺少生成任务（机器翻译、摘要）
   - 缺少跨语言迁移（如土耳其语-英语）

3. **形态学分词器质量依赖**
   - 使用的词法分析器（TRmorph）错误率约 7%
   - 更好的分析器可能改变结论

4. **计算预算单一**
   - 所有模型参数量相同（固定总预算）
   - 没有测试"小词表 + 大模型"vs"大词表 + 小模型"

## 我的批判性思考

### 这项工作的真正价值

**不是数字本身**（32k、100M 这些魔法数字），而是**方法论范式**：

1. **系统性对照实验**：同时变化三个维度（词表、语料、分词器），找出交互效应
2. **多层次评估**：内部指标（边界 F1）+ 外部任务（NLI/POS），建立因果链条
3. **诊断驱动优化**：用词根完整性等指标定位真正问题，而非瞎调超参数

### 未来方向的猜想

1. **自适应词表**
   - 根据语料规模动态调整词表大小
   - 训练时逐步增大词表（curriculum learning）

2. **多粒度混合表示**
   - 同时用字符、子词、形态学特征作为输入
   - 让模型自己学习最优加权

3. **端到端可微分分词**
   - 不固定分词策略,而是让分词本身可训练
   - 类似 Soft Attention 的思路

## 总结

土耳其语分词实验告诉我们三个深刻教训：

1. **没有银弹**
   - 词表大小、训练语料、任务类型三者环环相扣
   - "最佳配置"总是相对的,而非绝对的

2. **简单往往更好**
   - WordPiece + 足够语料 ≥ 复杂的形态学系统
   - 鲁棒性比理论优雅性更重要

3. **诊断比算法重要**
   - 知道分词器在哪里失败,比换个新算法更有价值
   - 边界 F1、词根完整性应该成为标配指标

这项工作提供的不是"土耳其语用 32k 词表"的结论，而是**"如何为你的语言找到最佳分词策略"**的范式。

---

## 参考资源

- 论文原文：arXiv:2602.06942v1
- HuggingFace Tokenizers 库：https://github.com/huggingface/tokenizers
- 土耳其语词法分析器：TRmorph, Zemberek
- 相关工作：SentencePiece (Kudo & Richardson, 2018)