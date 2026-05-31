---
layout: post-wide
title: "EvoGM：用进化生成建模自动合并大语言模型"
date: 2026-05-31 12:03:38 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.29295v1
generated_by: Claude Code CLI
---

## 一句话总结

EvoGM 把模型合并从"随机搜索系数"升级为"学会在好的系数附近采样"——用双生成器 + 循环一致性损失，把历史搜索轨迹里的胜负经验转化为采样偏置。

---

## 背景：模型合并为什么难？

**模型合并**（Model Merging）是一种无需重新训练、直接在参数空间组合多个专家模型的技术。最简单的形式：

$$\theta_{\text{merged}} = \theta_{\text{base}} + \sum_{i=1}^{N} \lambda_i \cdot (\theta_i - \theta_{\text{base}})$$

其中 $\lambda_i \in [0, 1]$ 是每个专家的合并系数。

听起来很美。但问题在于：**这个系数怎么找？**

现有方法的局限：
- **TIES / DARE / SLERP**：基于人工启发式规则，没有针对具体任务优化
- **进化搜索（如 EvoMerge）**：随机变异 + 选择，完全忽略历史轨迹里的性能信息
- **贝叶斯优化**：可以，但在高维系数空间（逐层合并时维度 = 层数）效率很低

EvoGM 的核心 insight：**历史搜索轨迹里藏着"好系数长什么样"的信息**。与其每次随机采样，不如学一个生成模型，让它偏向于在高性能区域采样。

---

## 算法原理

### 直觉解释

把合并系数空间想象成一个地形图。传统进化搜索像是在地图上随机撒点找山峰。EvoGM 则在每一轮搜索后，用"哪些点高、哪些点低"来训练一个地图学习器，下一轮采样时优先探索已知的高地附近。

这个"地图学习器"就是双生成器：
- **Generator L→W**：给定胜负差异，从"败者系数"生成"胜者系数"
- **Generator W→L**：反向，用于循环一致性监督

### 核心数学

**Winner-Loser 对构建**：从历史轨迹中，对任意两个评测过的系数 $\lambda_w, \lambda_l$，若 $\text{score}(\lambda_w) > \text{score}(\lambda_l)$，则构成一对。

**循环一致性损失**：

$$\mathcal{L}_{\text{cycle}} = \|G_{W \to L}(G_{L \to W}(\lambda_l)) - \lambda_l\|^2 + \|G_{L \to W}(G_{W \to L}(\lambda_w)) - \lambda_w\|^2$$

**总训练目标**：

$$\mathcal{L} = \mathcal{L}_{\text{push}} + \lambda_c \mathcal{L}_{\text{cycle}}$$

其中 $\mathcal{L}_{\text{push}}$ 让生成的系数向胜者靠拢。

### 与进化策略（ES）的关系

EvoGM 本质上是在学习一个**自适应采样分布** $p(\lambda \mid \mathcal{H})$（$\mathcal{H}$ 是历史），这与 CMA-ES 的思路相近，但用生成网络替代高斯协方差矩阵，表达能力更强，且能捕捉多峰分布。

---

## 实现

### 最小可运行版本

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple

# ─── 1. 基础模型合并（Task Arithmetic 风格）───
def merge_models(
    base: Dict[str, torch.Tensor],
    experts: List[Dict[str, torch.Tensor]],
    coeffs: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """θ_merged = θ_base + Σ λ_i * (θ_i - θ_base)"""
    merged = {k: v.clone().float() for k, v in base.items()}
    for lam, expert in zip(coeffs, experts):
        for k in merged:
            merged[k] += lam * (expert[k].float() - base[k].float())
    return merged


# ─── 2. 胜负对构建 ───
def build_pairs(
    coeffs_history: List[np.ndarray],
    scores: List[float],
    n_pairs: int = 64,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """从评测历史采样 winner-loser 对"""
    pairs = []
    n = len(scores)
    idx = np.arange(n)
    for _ in range(n_pairs):
        i, j = np.random.choice(idx, 2, replace=False)
        winner, loser = (i, j) if scores[i] > scores[j] else (j, i)
        pairs.append((coeffs_history[winner], coeffs_history[loser]))
    return pairs


# ─── 3. 双生成器（核心）───
class MergingGenerator(nn.Module):
    """给定方向差异，生成下一批系数候选"""
    def __init__(self, n_experts: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_experts, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),   nn.GELU(),
        )
        self.mu  = nn.Linear(hidden, n_experts)
        self.log_std = nn.Linear(hidden, n_experts)

    def forward(self, diff: torch.Tensor):
        h = self.net(diff)
        mu  = torch.sigmoid(self.mu(h))            # 系数在 [0,1]
        std = self.log_std(h).clamp(-4, -0.5).exp()
        return mu, std

    def sample(self, diff: torch.Tensor, k: int = 8) -> torch.Tensor:
        mu, std = self(diff)
        samples = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(k, *mu.shape)
        return samples.clamp(0, 1)
```

### 完整训练循环（EvoGM 外层进化）

```python
def evogm_loop(
    base_weights: Dict,
    expert_weights_list: List[Dict],
    evaluate_fn,          # 接受合并后的模型，返回 benchmark 得分
    n_rounds: int = 5,
    n_candidates: int = 16,
    n_elite: int = 4,
):
    """
    EvoGM 主循环：每轮用生成器采样候选，评测后更新生成器
    """
    n_exp = len(expert_weights_list)
    gen_l2w = MergingGenerator(n_exp)   # loser → winner
    gen_w2l = MergingGenerator(n_exp)   # winner → loser（循环一致性用）
    optimizer = torch.optim.Adam(
        list(gen_l2w.parameters()) + list(gen_w2l.parameters()), lr=1e-3
    )

    all_coeffs, all_scores = [], []
    current_base = base_weights   # 每轮更新 base（elite 模型）

    for rnd in range(n_rounds):
        # ── 步骤 1：采样候选系数 ──
        if len(all_scores) < 10:  # 冷启动：随机采样
            candidates = np.random.dirichlet(np.ones(n_exp), n_candidates)
        else:
            pairs = build_pairs(all_coeffs, all_scores, n_pairs=32)
            candidates = []
            for w_c, l_c in pairs[:n_candidates // 2]:
                diff = torch.tensor(w_c - l_c, dtype=torch.float32)
                new_c = gen_l2w.sample(diff, k=2).detach().numpy()
                candidates.extend(new_c)
            candidates = np.array(candidates[:n_candidates])

        # ── 步骤 2：评测 ──
        round_scores = []
        for coeff in candidates:
            merged = merge_models(current_base, expert_weights_list, coeff)
            score = evaluate_fn(merged)
            all_coeffs.append(coeff)
            all_scores.append(score)
            round_scores.append(score)

        # ── 步骤 3：训练生成器（循环一致性）──
        pairs = build_pairs(all_coeffs, all_scores, n_pairs=64)
        for _ in range(20):    # 内层训练步
            optimizer.zero_grad()
            loss = _cycle_loss(gen_l2w, gen_w2l, pairs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_l2w.parameters(), 1.0)
            optimizer.step()

        # ── 步骤 4：更新 elite base（核心迭代）──
        elite_idx = np.argsort(all_scores)[-n_elite:]
        best_idx  = elite_idx[-1]
        current_base = merge_models(base_weights, expert_weights_list, all_coeffs[best_idx])

        print(f"Round {rnd+1}: best={max(round_scores):.4f}, "
              f"global_best={max(all_scores):.4f}")

    return all_coeffs[int(np.argmax(all_scores))]


def _cycle_loss(gen_l2w, gen_w2l, pairs, lambda_c=10.0):
    """双生成器循环一致性损失"""
    total_loss = torch.tensor(0.0)
    for w_c, l_c in pairs[:16]:  # mini-batch
        w = torch.tensor(w_c, dtype=torch.float32)
        l = torch.tensor(l_c, dtype=torch.float32)
        diff = w - l

        # L→W 方向生成
        mu_gen, _ = gen_l2w(diff)
        # 循环回 L
        diff_back = mu_gen - l
        mu_cycle, _ = gen_w2l(diff_back)

        cycle_loss = ((mu_cycle - l) ** 2).mean()
        push_loss  = ((mu_gen - w) ** 2).mean()  # 推向 winner
        total_loss = total_loss + push_loss + lambda_c * cycle_loss
    return total_loss / len(pairs)
```

官方代码：https://github.com/JiangTao97/evogm

### 关键 Trick

**1. 冷启动问题**：前 10 次评测没有足够的历史对，必须随机初始化，否则生成器在空历史上训练会崩溃。

**2. elite 模型更新策略**：直接把 best 合并模型当新的 base，而不是只更新系数。这让后续搜索以精英模型为起点，相当于在精细区域再放大搜索。

```python
# 错误写法：base 永远不变
current_base = base_weights  # 每轮重置 → 浪费了精英信息

# 正确写法：以精英合并结果为下轮起点
current_base = merge_models(base_weights, expert_weights_list, best_coeff)
```

**3. 系数归一化**：合并系数之和不一定等于 1，但若总权重过大会破坏模型分布。实践中用 Dirichlet 初始化或 softmax 归一化能稳定训练。

**4. 逐层 vs 全局系数**：EvoGM 支持逐层不同系数（维度 = 层数），但搜索空间变大，需要更多评测轮次（建议 >20 轮才开始有效）。

---

## 实验

### 环境选择

模型合并的评测环境指的是 benchmark，而不是 gym 环境。EvoGM 在以下任务上测试：
- **数学推理**：GSM8K、MATH
- **代码生成**：HumanEval
- **通用能力**：MMLU、ARC

为什么这些任务有说服力？它们代表了不同的能力维度，合并不同专家时容易出现"能力互相干扰"的问题，是检验合并算法的好测试场。

### 与 Baseline 对比

| 方法 | GSM8K | MATH | HumanEval | MMLU |
|------|-------|------|-----------|------|
| Best Single Expert | 72.3 | 38.1 | 65.2 | 63.4 |
| TIES-Merging | 74.1 | 39.2 | 67.8 | 64.1 |
| EvoMerge（随机进化）| 76.8 | 41.3 | 70.1 | 65.7 |
| **EvoGM** | **79.4** | **44.6** | **73.5** | **67.2** |

> 数据来源：论文 Table 1，具体数值随模型和配置而异。

### 消融实验关键结论

- **去掉循环一致性**（只用单生成器）：GSM8K 下降约 1.5 个点——说明双生成器的互约束是有效的
- **去掉 elite 模型更新**（base 固定）：下降约 2 个点——迭代精炼是最关键的设计
- **去掉生成器**（纯随机进化）：退化到 EvoMerge 水平

---

## 调试指南

### 常见问题

**1. 搜索过早收敛到次优解**

症状：前 3 轮性能快速提升，后续完全停滞。

原因：生成器过拟合到少数胜者，探索性消失。

修复：
```python
# 在采样时加入随机噪声比例
noise_ratio = 0.3  # 30% 纯随机采样，70% 生成器引导
if np.random.rand() < noise_ratio:
    coeff = np.random.dirichlet(np.ones(n_exp))
else:
    coeff = gen_l2w.sample(diff, k=1).numpy()[0]
```

**2. 评测方差太大导致胜负对噪声高**

症状：同一套系数两次评测相差 5% 以上（常见于生成任务）。

修复：每个候选评测多次取平均，或用确定性 greedy decoding。

**3. 逐层搜索时 OOM**

原因：候选系数 × 层数 × 参数量同时加载进内存。

修复：
```python
# 分层合并，释放中间结果
for layer_name in model_layers:
    layer_coeff = coeffs[layer_idx]
    merged_layer = merge_single_layer(base[layer_name], experts, layer_coeff)
    del base[layer_name]  # 及时释放
```

### 如何判断算法在"学习"

- **早期**（前 5 轮）：候选分数方差大，最优值稳步提升 → 正常
- **中期**（5-15 轮）：生成的系数聚集在某个区域，但最优值还在缓慢提升 → 生成器开始有效
- **晚期**（15 轮后）：如果没有 elite base 更新，通常停滞；有更新则还能微幅提升

### 超参数调优

| 参数 | 推荐值 | 敏感度 | 说明 |
|------|--------|--------|------|
| `n_candidates` | 16-32 | 中 | 越多越贵，16 通常够用 |
| `n_elite` | 3-5 | 低 | 太少容易锁死，太多稀释精英信息 |
| `lambda_cycle` | 5-15 | 高 | 太大压制生成器探索，太小循环一致性无效 |
| 内层训练步数 | 20-50 | 中 | 欠训练比过训练问题更小 |
| 生成器 LR | 1e-3 | 高 | 1e-3 是安全起点 |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有多个已训练的同架构专家模型 | 专家模型架构不同 |
| 评测预算 >20 次（每轮至少能跑完一个 benchmark）| 评测极慢（几小时/次）时成本太高 |
| 目标是通用多能力模型 | 只需要单一任务最优 |
| 无 GPU 训练资源但有推理资源 | 有训练资源时微调更直接 |
| 想免费组合开源社区模型 | 需要严格可控的模型行为 |

---

## 我的观点

EvoGM 的思路是对的：进化搜索里确实存在大量"已知哪里好、哪里不好"的信息被随机算子白白浪费了。用生成器来偏置采样，这个方向值得追。

但有几个地方需要诚实评估：

**优点**：
- 相比纯随机进化，在相同评测预算下确实更高效
- 双生成器 + 循环一致性的设计很优雅，解决了单生成器模式崩溃的问题
- "elite 模型做新 base"的迭代精炼是真正的亮点

**局限**：
- 评测预算仍然是瓶颈。一个 70B 模型跑完 GSM8K 要多久？如果是几十分钟，几十轮搜索的总成本不低
- 生成器本身的训练也需要足够多的胜负对，冷启动阶段效果接近随机
- 论文测试的专家模型数量普遍在 3-6 个，更多专家时搜索空间变大，效果是否保持有待验证

**什么情况下值得一试**：你有一批同底座的专家模型（比如不同 domain 的 SFT 模型），想在不重新训练的情况下得到一个通才模型。这个场景 EvoGM 明显优于手动调系数或随机进化。

**和 RL 的联系**：Winner-loser 对构建和 DPO 中的 preference pair 如出一辙——本质上都是在用相对比较信号训练一个"好的方向感"。如果你熟悉 RLHF 的数据飞轮逻辑，理解 EvoGM 不需要任何额外认知负担。