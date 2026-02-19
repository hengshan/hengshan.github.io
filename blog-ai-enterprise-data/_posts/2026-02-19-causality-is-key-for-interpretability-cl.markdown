---
layout: post-wide
title: "因果推断是 LLM 可解释性的关键：为什么你的消融实验可能什么都没说明"
date: 2026-02-19 08:03:23 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.16698v1
generated_by: Claude Code CLI
---

* TOC
{:toc}

## 一句话总结

LLM 可解释性研究长期存在两个痼疾：发现不能泛化，以及用观察性证据支撑因果性结论。Pearl 因果层级（PCH）提供了一个诊断框架，帮你搞清楚自己的实验到底在说什么——以及说不了什么。

---

## 背景：可解释性研究的两个坑

过去几年，机械可解释性（Mechanistic Interpretability）领域出了不少漂亮的工作：电路发现、超级位置（superposition）、稀疏自编码器（SAE）…… 但如果你仔细看这些论文的 limitation 部分，或者尝试把发现迁移到别的模型上，你会发现一个共同的模式：**发现往往是局部的、脆弱的，结论往往超越了证据所能支持的范围**。

**坑一：发现不能泛化。** 在 GPT-2 上找到的"间接宾语识别电路"，换一批 prompt 后还成立吗？换到 GPT-4o 还成立吗？很多论文没有系统地测试过这个问题。

**坑二：因果越界。** 激活值与行为相关 → 于是我们说"这个注意力头负责 X"。但相关性不等于因果性，更不等于反事实意义上的必要性。这个跳跃是隐蔽的，评审者和读者都容易忽视。

这篇论文（[arxiv:2602.16698](https://arxiv.org/abs/2602.16698v1)）的核心论点是：**Pearl 因果层级明确规定了什么证据能支持什么论断**。没有这个框架，你的解释很可能是在讲故事，而非揭示机制。

---

## Pearl 因果层级：三层能力，三种问题

Judea Pearl 把推断能力分成三层，每一层需要不同类型的数据和假设：

### 第一层：关联（Association）

$$P(Y \mid X)$$

问题类型：*如果我观察到 X，Y 是什么？*

在可解释性里的对应：
- 注意力熵与任务性能的相关性分析
- 探测器（probe）的线性可分性测试
- 某个 token 的激活值分布统计

**能说什么**：A 和 B 在数据中共同出现。

**不能说什么**：A 导致 B，或者去掉 A 之后 B 会消失。

---

### 第二层：干预（Intervention）

$$P(Y \mid do(X=x))$$

问题类型：*如果我强制把 X 设为 x，Y 会怎样？*

在可解释性里的对应：
- 激活修补（Activation Patching）：把干净运行的激活替换进被污染的运行
- 消融实验（Ablation）：把某个注意力头的输出清零
- 因果中介分析（Causal Mediation Analysis）

**能说什么**：这次编辑在**平均意义上**改变了多少行为，对应的是总体干预效果。

**不能说什么**：对于**某个特定 prompt**，如果当初的激活不同，输出具体会是什么——这是第三层的问题。

---

### 第三层：反事实（Counterfactual）

$$P(Y_{X=x} \mid X=x', Y=y')$$

问题类型：*在已知 X=x' 且 Y=y' 的情况下，如果 X 反事实地等于 x，Y 会是什么？*

在可解释性里的对应：
- "如果这个 prompt 的第 15 层激活换成另一个 prompt 的，输出**还会是同样的错误**吗？"
- 个体层面的充分性/必要性判断

**为什么难**：反事实推断需要完整的结构因果模型（SCM），从观测数据中通常**无法识别**（not identified）。实践中，大多数可解释性实验根本没有能力回答反事实问题，但论文的措辞往往暗示了这个层级的结论。

---

## 三层的实践对应

下面用具体示例把这三层落地。

### 第一层：探测器——相关性分析

```python
class LinearProbe(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, activations):
        return self.linear(activations[:, -1, :])

def train_probe(model, tokenizer, texts, labels, layer_idx=6):
    """
    训练线性探测器，评估某层激活与标签的线性相关性。
    
    注意：高探测精度只说明"激活与标签线性相关"。
    它不说明该层"负责"该特征，更不说明去掉这层特征
    会改变行为——这需要第二层的干预实验。
    """
    model.eval()
    probe = LinearProbe(model.config.n_embd, n_classes=2)
    # ... 训练过程省略 ...
    return probe
```

典型的因果越界写法是：探测精度 95% → 结论"第 8 层负责存储性别信息"。正确的结论只能是"第 8 层激活与性别标签线性相关"。两者差别很大：前者隐含了功能性归因和反事实不变性，后者只陈述了一个统计事实。

---

### 第二层：激活修补——干预分析

激活修补是目前最接近"因果"的工具，也是机械可解释性的主力方法：

```python
def activation_patching(model, clean_tokens, corrupted_tokens, target_layer):
    """
    激活修补：将干净运行的指定层激活，替换进被污染的运行中，
    测量输出的变化量。
    
    这回答的是 do() 层级的问题：
      "强制把第 target_layer 层的激活设为干净值后，
       输出的期望变化是多少？"
    
    关键限制：这是对一批 prompt 的平均效果，不是对单个
    prompt 的反事实。某层效果显著，可能只是因为信息在
    那层汇聚，而非该层是唯一的功能载体。
    """
    clean_acts = {}

    # 步骤一：记录干净运行的激活
    def save_hook(module, input, output):
        clean_acts[target_layer] = output[0].detach()
        return output

    handle = model.transformer.h[target_layer].register_forward_hook(save_hook)
    with torch.no_grad():
        model(clean_tokens)
    handle.remove()

    # 步骤二：在被污染的运行中替换该层激活
    def patch_hook(module, input, output):
        return (clean_acts[target_layer],) + output[1:]

    handle = model.transformer.h[target_layer].register_forward_hook(patch_hook)
    with torch.no_grad():
        patched_output = model(corrupted_tokens)
    handle.remove()

    return patched_output.logits
```

激活修补能做什么：在控制其他层不变的条件下，测量"这一层的激活差异"对输出的平均贡献。

激活修补不能做什么：解释为什么某个具体 prompt 答错了，或者证明该层是某个能力的"必要条件"。

---

### 分解直接/间接效应：因果中介分析

```python
def compute_total_indirect_effect(model, clean_tokens, corrupted_tokens,
                                   candidate_layers, metric_fn):
    """
    总间接效应（TIE）：冻结最终输出层，逐层 patch，量化各层对
    metric 恢复的贡献。对应 do-calculus 中的路径特定效应。

    仍然是 do() 层级，不是反事实层级。
    适合用于"哪些层传递了关键信息"的研究问题。
    """
    with torch.no_grad():
        baseline = metric_fn(model(corrupted_tokens).logits)

    layer_contributions = {}
    for layer_idx in candidate_layers:
        patched_logits = activation_patching(
            model, clean_tokens, corrupted_tokens, layer_idx
        )
        layer_contributions[layer_idx] = metric_fn(patched_logits) - baseline

    return layer_contributions
```

---

## 因果表示学习（CRL）：可识别性是什么意思

论文的第二个贡献是用**因果表示学习**（CRL）精确化"我们能从激活中恢复什么"这个问题。

### 问题设置

激活空间 $\mathbf{h} \in \mathbb{R}^d$ 由潜在因果变量 $\mathbf{z}$ 生成：

$$\mathbf{h} = f(\mathbf{z}), \quad \mathbf{z} \sim P_{\text{causal}}$$

**可识别性**（identifiability）：给定激活 $\mathbf{h}$，在什么条件下我们能唯一恢复 $\mathbf{z}$？

答案高度依赖假设：

| 假设 | 可识别的内容 |
|------|-------------|
| 无额外假设 | 几乎没有保证 |
| 线性混合 + 非高斯噪声 | 独立成分（ICA）可识别 |
| 受监督的干预标签可用 | 干预目标变量可识别 |
| 稀疏因果图约束 | 稀疏性约束下可识别 |

### 哪类可解释性实验满足了哪类假设？

这是论文最实用的部分，也是很多研究者没有明说的盲点：

**ICA / 稀疏自编码器（SAE）**：隐含假设潜在特征是统计独立且稀疏的。如果模型内部特征存在强依赖关系（比如"情感"和"主观性"高度相关），ICA 分解出来的方向可能没有独立的因果意义。

**探测器（Linear Probe）**：不做可识别性假设，因此结论也最弱——它只能说激活与标签线性相关，不能说激活"编码了"该概念（编码意味着更强的表示唯一性）。

**激活修补**：隐含假设"干净激活"和"被污染激活"是同一个因果模型下的不同实例，且替换激活不会产生分布外的副作用。当两个输入在语义上差异很大时，这个假设可能不成立，修补的效果也就难以解释。

**受监督干预数据**（如 Geiger et al. 的 Interchange Intervention）：最接近第二层的严格实验，且在特定条件下支持可识别性，但需要事先知道干预目标。

```python
def check_identifiability_empirically(activations, intervention_labels):
    """
    经验验证：在有监督干预标签下，分解出的成分是否能预测干预目标？
    
    这提供了可识别性的经验证据——但不是理论保证。
    如果交叉验证分数远高于随机，说明分解方向与因果变量对齐；
    如果接近随机，说明分解可能只捕捉了统计相关性。
    """
    from sklearn.decomposition import FastICA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    ica = FastICA(n_components=20, random_state=42, max_iter=500)
    components = ica.fit_transform(activations)  # [n_samples, 20]

    clf = LogisticRegression()
    scores = cross_val_score(clf, components, intervention_labels, cv=5)

    print(f"均值准确率: {scores.mean():.3f} ± {scores.std():.3f}")
    print("高于随机基线 → 分解方向与因果变量存在对齐")
    print("接近随机基线 → 分解可能只捕捉了统计模式")
    return scores
```

---

## 诊断工具：你的实验在哪一层？

### 快速检查清单

**在写"component X 负责 behavior Y"之前，先问自己：**

- [ ] 我的证据是相关性（第一层）还是干预（第二层）？
  → 只有探测器结果，不能声称因果性
- [ ] 我的干预是在分布内还是分布外？
  → 两个语义差异很大的 prompt 之间的激活修补，结果可能是 artifact
- [ ] 我的结论是对一批 prompt 的平均，还是对单个 prompt 的声明？
  → 个体层面的反事实（第三层）通常无法从实验中识别
- [ ] 我的发现在不同 prompt 集合上是否稳健？
  → 5 个 prompt 上的结果，不能外推到全部分布
- [ ] 我声明的"电路"在更大/不同的模型上还成立吗？
  → 没测试过就是假设，不是发现

### 稳健性诊断代码

```python
def causal_claim_diagnostics(results_set1, results_set2, threshold=0.3):
    """
    诊断激活修补结果在两组不同 prompt 上的一致性。
    
    Args:
        results_set1: 第一组 prompt 的每层 TIE 分数 [n_layers]
        results_set2: 第二组 prompt 的每层 TIE 分数 [n_layers]
        threshold:    认为某层"重要"的 TIE 阈值
    """
    r1, r2 = np.array(results_set1), np.array(results_set2)

    important_1 = set(np.where(r1 > threshold)[0])
    important_2 = set(np.where(r2 > threshold)[0])
    jaccard = len(important_1 & important_2) / max(len(important_1 | important_2), 1)
    rank_corr = np.corrcoef(r1, r2)[0, 1]

    strength = (
        "strong"   if jaccard > 0.7 and rank_corr > 0.8 else
        "moderate" if jaccard > 0.4 else
        "weak"
    )

    print(f"Jaccard 相似度 (重要层集合): {jaccard:.3f}")
    print(f"Rank 相关性 (层重要性排序): {rank_corr:.3f}")
    print(f"结论强度: {strength}")

    if strength == "weak":
        print("\n⚠ 警告：两组 prompt 上的结果不一致。")
        print("  发现可能不能泛化，降低因果声明的强度。")

    return {"jaccard": jaccard, "rank_corr": rank_corr, "strength": strength}
```

---

## 常见错误模式：论文写法对比

### 错误模式 1：探测精度即功能归因

| | 写法 | 对应层级 |
|-|------|---------|
| **实际证据** | 第 8 层激活上的线性探测器，性别分类准确率 95% | 第一层（关联） |
| **错误结论** | "第 8 层**负责存储**性别信息" | 第二层甚至第三层 |
| **正确结论** | "第 8 层激活与性别标签**线性相关**" | 第一层（匹配） |

为什么错？探测器可能利用了与性别相关但功能不同的特征；删掉该层的性别信息，模型可能从其他层恢复。只有消融/修补实验才能支持"负责"的说法。

---

### 错误模式 2：平均效应即个体反事实

| | 写法 | 对应层级 |
|-|------|---------|
| **实际证据** | patch 第 15 层后，1000 个 prompt 上平均准确率提升 30% | 第二层（干预，平均） |
| **错误结论** | "对于 prompt P，如果第 15 层激活不同，它就会答对" | 第三层（个体反事实） |
| **正确结论** | "在这组 prompt 上，第 15 层的干预**平均**改善了 30%" | 第二层（匹配） |

平均效应和个体反事实是不同的量。有些 prompt 可能在 patch 后反而变差，被平均掩盖。个体层面的反事实需要 SCM，通常无法从实验中直接识别。

---

### 错误模式 3：单一配置的实验

```python
# 错误：结论基于单一 prompt 集和单一随机种子
results = run_activation_patching(model, prompts_A, seed=42)
print("发现：layer 12 最重要")

# 正确：系统性地检验跨配置的一致性
all_results = []
for seed in [42, 123, 456]:
    for prompt_set in [prompts_A, prompts_B, prompts_C]:
        r = run_activation_patching(model, prompt_set, seed=seed)
        all_results.append(r)

# 只有在多个配置下一致的发现才值得报告
consistency = causal_claim_diagnostics(all_results[0], all_results[1])
```

---

## 我的观点

这篇论文是一篇 position paper，没有新实验，但它提供了一个真正有用的**语言系统**。

**最有价值的贡献**：给了研究者一个精确说清楚"我在 claim 什么"的工具。就像统计学里区分"置信区间"和"预测区间"一样，可解释性领域也需要区分"这个干预平均有效"和"这个组件是因果必要的"。

**不要过度解读**：这篇论文不是说现有工作都是错的，而是说 claim 要和证据匹配。消融实验仍然有价值，只是应该被定位为第二层的证据，不是第三层的。

**真正的难题没有解决**：框架是清晰的，但论文没有解决"如何在实践中获得足够的受控干预数据"这个核心困难。反事实实验的数据要求很高，大多数实验室没有这个条件，所以第三层的结论在实践中依然罕见且昂贵。

**给实践者的建议**：
- 写论文时，对每个实验明确标注它在 PCH 的哪一层
- 审论文时，检查结论是否超出了证据的层级
- 做实验时，至少用两组 prompt 集验证发现的稳健性，再声称泛化性

---

## 参考资料

- [Causality is Key for Interpretability Claims to Generalise (arxiv:2602.16698)](https://arxiv.org/abs/2602.16698v1)
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Wang et al. (2022). [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)
- Geiger et al. (2021). [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997)
- Schölkopf et al. (2021). [Toward Causal Representation Learning](https://arxiv.org/abs/2102.11107)