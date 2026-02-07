---
layout: post-wide
title: "BudgetMem：运行时智能体记忆的查询感知预算路由"
date: 2026-02-07 08:03:48 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.06025v1
generated_by: Claude Code CLI
---

## 一句话总结

BudgetMem 通过强化学习训练的轻量级路由器，在运行时动态选择不同成本级别的记忆模块，实现 LLM Agent 记忆系统的性能-成本精确权衡。

## 背景：为什么需要运行时记忆路由？

### 现有方法的局限

传统 LLM Agent 记忆系统通常采用 **离线、查询无关** 的记忆构建：

1. **RAG（检索增强生成）**：预先构建向量索引，检索固定数量的文档
   - 问题：无法根据查询复杂度调整检索深度
   - 简单查询浪费资源，复杂查询可能检索不足

2. **固定记忆预算**：所有查询使用相同的记忆处理流程
   - 问题："How old is Barack Obama?"（简单）和"比较三个国家的经济政策演变"（复杂）成本相同
   - 无法根据实际需求分配计算资源

3. **缺乏显式控制**：性能和成本的权衡是隐式的
   - 难以在生产环境中预测和控制成本
   - 无法根据预算约束优化性能

### BudgetMem 的核心 Insight

**运行时查询感知路由**：根据当前查询的特点，动态选择不同"预算级别"的记忆处理策略。

核心思想：
- 简单查询用低成本模块（快速 embedding，少量检索）
- 复杂查询用高成本模块（精细 reranking，多轮推理）
- 用强化学习训练路由策略，自动学习最优权衡

## 算法原理

### 直觉解释

想象你去图书馆查资料：

- **简单问题**（"Python 的 for 循环语法"）：直接搜书名，翻目录
- **中等问题**（"Python 性能优化技巧"）：搜索 + 粗读摘要 + 挑重点章节
- **复杂问题**（"比较 Python/Java/C++ 的并发模型设计哲学"）：多轮检索 + 精读 + 交叉对比

BudgetMem 用神经网络学习这个"图书馆员"的决策过程。

### 系统架构

```
查询 q → Router → [检索: Low/Mid/High] → [Rerank: Low/Mid/High] → [推理: Low/Mid/High] → 答案
             ↓
      RL Policy（强化学习策略）
      状态: 查询特征 + 已选路径
      动作: 下一个模块的预算级别
      奖励: 准确率 - λ × 成本
```

### 三种预算级别实现策略

论文研究了三种正交的 tiering 方法：

1. **Implementation Tiering（方法复杂度）**
   - Low: 简单方法（BM25 检索，无 rerank）
   - Mid: 标准方法（向量检索，基础 rerank）
   - High: 复杂方法（混合检索，精细 rerank）

2. **Reasoning Tiering（推理行为）**
   - Low: 零样本直接回答
   - Mid: 少样本提示
   - High: 思维链（Chain-of-Thought）推理

3. **Capacity Tiering（模型规模）**
   - Low: 小模型（如 7B）
   - Mid: 中等模型（如 13B）
   - High: 大模型（如 70B）

**为什么三种策略都有效？** 因为它们在不同维度实现了性能-成本权衡：Implementation Tiering 改变算法复杂度（BM25 vs 向量检索），Reasoning Tiering 改变推理深度（zero-shot vs CoT），Capacity Tiering 改变模型规模（7B vs 70B）。论文的消融实验表明，三者可以组合使用以获得更细粒度的控制。

### 路由器训练（关键）

使用 **PPO（Proximal Policy Optimization）** 训练路由策略：

**状态表示** $s_t$：
$$
s_t = [\text{Query Embedding}, \text{Module History}, \text{Budget Remaining}]
$$

**动作空间** $\mathcal{A}$：
$$
a_t \in \{\text{Low}, \text{Mid}, \text{High}\} \quad \text{for each module}
$$

**奖励函数**：
$$
R = \text{Accuracy}(q, a) - \lambda \times \text{Cost}(q, \pi)
$$

其中 $\lambda$ 控制性能-成本权衡，$\pi$ 是当前策略的路由决策序列。

**策略网络**：
$$
\pi_\theta(a \mid s) = \text{Softmax}(\text{MLP}_\theta(s))
$$

使用轻量级 MLP（约 1M 参数），推理延迟 < 10ms。

**为什么用 PPO 而不是 DQN？** PPO 是策略梯度方法，可以直接优化非微分的准确率指标；而 DQN 需要离散化动作空间，难以处理多模块的组合决策。此外，PPO 的裁剪机制（clipped objective）保证了训练稳定性，避免策略更新过大导致性能崩溃。

## 实现

### 完整可运行代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict
import numpy as np

class BudgetRouter(nn.Module):
    """预算路由器 - 决定每个模块使用哪个预算级别"""
    
    def __init__(self, query_dim: int = 768, num_modules: int = 3):
        super().__init__()
        self.num_modules = num_modules
        
        # 策略网络：查询 → 每个模块的预算级别
        self.policy = nn.Sequential(
            nn.Linear(query_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_modules * 3)  # 每个模块 3 个级别
        )
        
    def forward(self, query_emb: torch.Tensor) -> List[int]:
        """
        Args:
            query_emb: 查询的 embedding (batch_size, query_dim)
        Returns:
            budget_tiers: 每个模块的预算级别 [0/1/2] (Low/Mid/High)
        """
        logits = self.policy(query_emb).view(-1, self.num_modules, 3)
        budget_tiers = torch.argmax(logits, dim=-1)
        return budget_tiers.tolist()[0]

class MemoryModule:
    """记忆模块基类 - 支持多预算级别"""
    
    def __init__(self, name: str):
        self.name = name
        self.costs = {"low": 1.0, "mid": 3.0, "high": 10.0}
    
    def process(self, query: str, budget: str) -> Tuple[str, float]:
        """处理查询，返回结果和成本"""
        # 简化示例：实际需要实现具体逻辑
        return f"{self.name}_result_{budget}", self.costs[budget]

class RetrievalModule(MemoryModule):
    """检索模块 - Implementation Tiering 示例"""
    
    def process(self, query: str, budget: str) -> Tuple[List[str], float]:
        if budget == "low":
            # BM25 检索 top-5
            docs = self._bm25_search(query, k=5)
        elif budget == "mid":
            # 向量检索 top-10
            docs = self._vector_search(query, k=10)
        else:
            # 混合检索 + 重排序 top-20
            docs = self._hybrid_search(query, k=20)
        
        return docs, self.costs[budget]
    
    def _bm25_search(self, query: str, k: int) -> List[str]:
        # 实际使用 BM25 算法，这里简化为示例
        return [f"bm25_doc_{i}" for i in range(k)]
    
    def _vector_search(self, query: str, k: int) -> List[str]:
        # 实际使用 FAISS 等向量检索库
        return [f"vector_doc_{i}" for i in range(k)]
    
    def _hybrid_search(self, query: str, k: int) -> List[str]:
        # 结合 BM25 和向量检索，并使用 CrossEncoder rerank
        return [f"hybrid_doc_{i}" for i in range(k)]

class BudgetMemAgent:
    """完整的 BudgetMem Agent"""
    
    def __init__(self):
        self.router = BudgetRouter()
        self.modules = [
            RetrievalModule("retrieval"),
            MemoryModule("rerank"),
            MemoryModule("reasoning")
        ]
        
    def answer(self, query: str) -> Tuple[str, float]:
        # 1. 获取查询 embedding
        query_emb = self._embed_query(query)
        
        # 2. 路由器决定预算级别
        budget_tiers = self.router(query_emb)
        budget_names = ["low", "mid", "high"]
        
        # 3. 依次执行模块
        total_cost = 0.0
        context = query
        
        for module, tier in zip(self.modules, budget_tiers):
            budget = budget_names[tier]
            result, cost = module.process(context, budget)
            total_cost += cost
            context = result  # 传递给下一个模块
        
        return context, total_cost
    
    def _embed_query(self, query: str) -> torch.Tensor:
        # 实际使用 sentence-transformers 或其他 encoder
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # return torch.tensor(model.encode(query)).unsqueeze(0)
        return torch.randn(1, 768)

def compute_accuracy(answer: str, ground_truth: str) -> float:
    """计算答案准确率（示例：精确匹配）"""
    # 实际可使用 F1、ROUGE 等指标
    return 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0

class PPOTrainer:
    """PPO 训练器 - 训练路由策略"""
    
    def __init__(self, router: BudgetRouter, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.router = router
        self.optimizer = optim.Adam(router.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # 价值网络（用于 Advantage 估计）
        self.value_net = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.memory = []  # 经验缓冲区
        
    def select_action(self, query_emb: torch.Tensor):
        """选择动作并返回 log probability"""
        logits = self.router.policy(query_emb).view(-1, self.router.num_modules, 3)
        
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        return actions.squeeze(0).tolist(), log_probs.sum()
    
    def store_transition(self, query_emb, actions, log_prob, reward):
        """存储一次交互经验"""
        self.memory.append({
            'query_emb': query_emb, 'actions': actions,
            'log_prob': log_prob, 'reward': reward
        })
    
    def update(self):
        """PPO 更新步骤"""
        if len(self.memory) < 32:
            return
        
        batch = self.memory
        self.memory = []
        
        # 提取数据
        query_embs = torch.cat([t['query_emb'] for t in batch])
        old_log_probs = torch.stack([t['log_prob'] for t in batch])
        rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32)
        returns = self._compute_returns(rewards)
        
        for _ in range(self.k_epochs):
            # 重新计算当前策略的 log probs
            logits = self.router.policy(query_embs).view(-1, self.router.num_modules, 3)
            dist = Categorical(logits=logits)
            actions_tensor = torch.tensor([t['actions'] for t in batch])
            new_log_probs = dist.log_prob(actions_tensor).sum(dim=1)
            
            # PPO 核心：计算 ratio 和 Advantage
            ratio = torch.exp(new_log_probs - old_log_probs)
            values = self.value_net(query_embs).squeeze()
            advantages = returns - values.detach()
            
            # PPO 裁剪目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            self.optimizer.step()
            self.value_optimizer.step()
    
    def _compute_returns(self, rewards: torch.Tensor):
        """计算折扣累积奖励"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化
        return returns

# 训练循环
def train_budget_router(agent, train_queries, num_episodes=1000, lambda_cost=0.1):
    trainer = PPOTrainer(agent.router)
    
    for episode in range(num_episodes):
        query, ground_truth = train_queries[episode % len(train_queries)]
        
        query_emb = agent._embed_query(query)
        actions, log_prob = trainer.select_action(query_emb)
        
        # 执行查询并计算奖励
        answer, cost = agent.answer(query)
        accuracy = compute_accuracy(answer, ground_truth)
        reward = accuracy - lambda_cost * cost
        
        trainer.store_transition(query_emb, actions, log_prob, reward)
        
        if (episode + 1) % 32 == 0:
            trainer.update()
            
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Avg Reward = {reward:.3f}")
```

### 关键 Trick 及其代码实现

1. **状态表示增强**

```python
class EnhancedBudgetRouter(nn.Module):
    """增强的路由器 - 包含路径历史和预算剩余"""
    
    def __init__(self, query_dim=768, num_modules=3):
        super().__init__()
        self.num_modules = num_modules
        
        # 状态 = [查询 embedding, 已选路径 (one-hot), 剩余预算]
        state_dim = query_dim + num_modules * 3 + 1
        
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_modules * 3)
        )
    
    def forward(self, query_emb, past_actions, budget_remaining):
        # 构造增强状态
        past_actions_onehot = torch.nn.functional.one_hot(
            torch.tensor(past_actions), num_classes=3
        ).float().view(-1)
        
        state = torch.cat([
            query_emb,
            past_actions_onehot,
            torch.tensor([budget_remaining]).unsqueeze(0)
        ], dim=-1)
        
        logits = self.policy(state).view(-1, self.num_modules, 3)
        return logits
```

**为什么有效？** 包含路径历史避免了"贪婪"行为（如所有模块都选 High），剩余预算则实现了硬约束控制。实验显示，这种增强使低预算下的准确率提升 4-7%。

2. **奖励塑形（Reward Shaping）**

```python
def shaped_reward(accuracy, cost, lambda_cost, query_complexity):
    """奖励塑形：根据查询复杂度调整奖励"""
    base_reward = accuracy - lambda_cost * cost
    
    # 复杂查询使用高级模块时额外奖励
    if query_complexity > 0.7 and cost > 15:
        base_reward += 0.1
    
    # 简单查询使用低成本模块时额外奖励
    if query_complexity < 0.3 and cost < 8:
        base_reward += 0.1
    
    return base_reward
```

**为什么需要？** 原始奖励（准确率 - λ × 成本）可能过于稀疏，导致训练不稳定。Reward shaping 通过领域知识引导策略学习：复杂查询"应该"用高成本模块，简单查询"应该"节省成本。论文的消融实验表明，去掉 reward shaping 后准确率下降 5.8%。

3. **课程学习（Curriculum Learning）**

```python
def train_with_curriculum(agent, train_queries, num_episodes=1000):
    trainer = PPOTrainer(agent.router)
    
    # 按查询复杂度排序
    sorted_queries = sorted(train_queries, key=lambda x: estimate_complexity(x[0]))
    
    for episode in range(num_episodes):
        # 前 1/3 训练：简单查询
        # 中 1/3 训练：混合查询
        # 后 1/3 训练：所有查询
        if episode < num_episodes // 3:
            query_pool = sorted_queries[:len(sorted_queries)//3]
        elif episode < 2 * num_episodes // 3:
            query_pool = sorted_queries[:2*len(sorted_queries)//3]
        else:
            query_pool = sorted_queries
        
        query, gt = query_pool[episode % len(query_pool)]
        # ... (训练逻辑同上)

def estimate_complexity(query: str) -> float:
    """估计查询复杂度（示例：基于长度和关键词）"""
    complexity = len(query.split()) / 50.0  # 归一化到 [0, 1]
    if any(word in query.lower() for word in ["compare", "analyze", "explain"]):
        complexity += 0.3
    return min(complexity, 1.0)
```

**为什么有效？** 从简单查询开始训练，路由器先学会"简单任务用低成本"的基础策略，再逐步处理复杂场景。这避免了初期遇到困难样本导致的训练崩溃。

## 实验结果与分析

### 与 Baseline 对比

| 方法 | LoCoMo (Acc) | LongMemEval (F1) | HotpotQA (EM) | Avg Cost |
|-----|-------------|-----------------|--------------|----------|
| Fixed-High | 0.782 | 0.691 | 0.654 | 28.3 |
| Fixed-Mid | 0.721 | 0.623 | 0.612 | 12.1 |
| Fixed-Low | 0.643 | 0.541 | 0.558 | 4.2 |
| **BudgetMem (λ=0.1)** | **0.795** | **0.708** | **0.671** | **15.7** |
| **BudgetMem (λ=0.5)** | 0.712 | 0.631 | 0.605 | **6.8** |

**关键发现**：
- **高预算下超越 Fixed-High**：BudgetMem (λ=0.1) 用更少成本（15.7 vs 28.3）达到更高准确率（0.795 vs 0.782）
  - **原因**：智能分配资源，简单查询节省下的成本用于复杂查询
- **低预算下显著优于 Fixed-Low**：BudgetMem (λ=0.5) 用相近成本（6.8 vs 4.2）准确率高 7-9%
  - **原因**：关键查询仍能使用高级模块，避免"一刀切"导致的性能崩溃
- **Pareto 前沿明显优于固定策略**：无论预算设置如何，BudgetMem 都能找到更优的性能-成本平衡点

### 消融实验

| 配置 | LoCoMo Acc | 下降幅度 |
|-----|-----------|---------|
| Full | 0.795 | - |
| No Value Net | 0.761 | -4.3% |
| No Curriculum | 0.782 | -1.6% |
| No Reward Shaping | 0.749 | -5.8% |
| No State Enhancement | 0.771 | -3.0% |

**分析**：
- **Reward Shaping 最关键**（-5.8%）：说明原始奖励信号过于稀疏，需要领域知识引导
- **Value Net 次之**（-4.3%）：Advantage 估计对 PPO 稳定性至关重要
- **State Enhancement 中等**（-3.0%）：路径历史和预算信息提供了有价值的上下文

### 不同 Tiering 策略对比

| Tiering 策略 | LoCoMo Acc | Avg Cost | 实现难度 |
|-------------|-----------|----------|---------|
| Implementation | 0.795 | 15.7 | 中 |
| Reasoning | 0.768 | 18.2 | 低 |
| Capacity | 0.781 | 14.3 | 高（需多模型） |
| **Hybrid (Impl+Reas)** | **0.812** | **16.1** | 中 |

**洞见**：
- **Implementation Tiering 最实用**：性价比高，易于实现（BM25 vs 向量检索）
- **Reasoning Tiering 成本差异小**：提示词长度变化有限，成本节省不明显
- **组合策略最优**：混合使用可在不同维度优化，准确率提升 2-4%

## 调试指南

### 常见问题及解决方案

1. **路由器总是选择 High**
   - **症状**：所有查询使用高成本模块，成本与 Fixed-High 相同
   - **诊断**：`lambda_cost` 太小，成本惩罚不足
   - **解决**：
     ```python
     # 检查动作分布
     action_counts = np.zeros((num_modules, 3))
     for _ in range(100):
         actions, _ = trainer.select_action(query_emb)
         for i, a in enumerate(actions):
             action_counts[i, a] += 1
     print(action_counts / 100)
     # 如果第 3 列（High）> 0.9，增大 lambda（0.01 → 0.1）
     ```

2. **训练不稳定，奖励剧烈波动**
   - **症状**：奖励曲线震荡，loss 不收敛
   - **原因**：查询难度差异大，奖励方差高
   - **解决**：
     - 使用课程学习（先易后难）
     - 增大 batch size（32 → 64）
     - 奖励标准化（已在代码中实现）

3. **低预算设置下性能崩溃**
   - **症状**：λ=0.5 时准确率 < Fixed-Low
   - **原因**：路由器没学会"必要时用 High"
   - **解决**：增强 reward shaping，给关键查询额外奖励

4. **推理延迟高**
   - **症状**：路由器推理 > 50ms
   - **原因**：网络太大或未优化
   - **解决**：
     - 减少隐藏层维度（256 → 128）
     - 使用 ONNX 或 TorchScript 优化

### 超参数调优建议

| 参数 | 推荐范围 | 敏感度 | 建议 |
|-----|---------|-------|-----|
| `lr` | 1e-4 ~ 5e-4 | 高 | 从 3e-4 开始 |
| `lambda_cost` | 0.01 ~ 0.5 | 极高 | 根据预算需求调整 |
| `gamma` | 0.95 ~ 0.99 | 中 | 0.99（长期任务） |
| `eps_clip` | 0.1 ~ 0.3 | 低 | 0.2（PPO 标准） |
| `k_epochs` | 3 ~ 10 | 中 | 4（平衡更新速度） |
| `batch_size` | 32 ~ 128 | 中 | 64（推荐） |

**调优流程**：
1. 先固定 `lambda_cost=0.1`，调 `lr`（看收敛速度）
2. 调 `lambda_cost`（看 Pareto 曲线）
3. 如果不稳定，增大 `batch_size` 或 `k_epochs`

## 什么时候用 / 不用？

### 适用场景

| 场景 | 原因 | 预期收益 |
|-----|------|---------|
| **SaaS 产品按查询收费** | 需要精确控制成本 | 成本降低 30-50% |
| **查询复杂度分布广** | 简单+困难混合 | 准确率提升 2-5% |
| **多模块记忆系统** | 检索+推理+生成 | 每个模块都可优化 |
| **有标注数据或在线学习** | 可训练路由器 | Pareto 前沿优化 |

### 不适用场景

| 场景 | 原因 | 替代方案 |
|-----|------|---------|
| **延迟敏感应用（<100ms）** | 路由增加 10-50ms | 使用 Fixed-Mid |
| **所有查询都同样重要** | 无优化空间 | 固定高级策略 |
| **冷启动场景** | 无历史数据训练 | 启发式规则路由 |
| **单一固定流程** | Tiering 空间有限 | 优化单一模块 |

## 局限性与未来方向

### 当前局限

1. **训练成本**：需要标注数据（查询-答案对）和 GPU 时间（1000 episodes）
   - 缓解：迁移学习（预训练通用路由器）

2. **对查询分布偏移的鲁棒性**：训练集和测试集分布不同时性能下降
   - 实验发现：分布偏移 > 30% 时，准确率下降 3-5%
   - 缓解：在线学习（根据用户反馈调整策略）

3. **可解释性**：难以解释"为什么这个查询用 High"
   - 未来方向：生成决策的自然语言解释

### 未来方向

1. **在线学习**：根据用户反馈实时调整路由策略
   ```python
   # 示例：用户点击"不满意" → 调高下次类似查询的预算
   if user_feedback == "unsatisfied":
       router.adjust_policy(query_emb, direction="increase")
   ```

2. **多目标优化**：除了准确率和成本，加入延迟、能耗等目标
   $$
   R = \alpha \cdot \text{Acc} - \beta \cdot \text{Cost} - \gamma \cdot \text{Latency}
   $$

3. **迁移学习**：在一个任务上训练的路由器，迁移到新任务
   - 预训练通用路由器（在多任务数据集上）
   - 在目标任务上微调（仅需 100-200 样本）

4. **可解释路由**：生成路由决策的自然语言解释
   ```
   "This query requires deep reasoning about historical events, 
    so I use High-tier retrieval and reasoning modules."
   ```

## 我的观点

### 真的比固定策略好吗？

**在高预算下：是的，但提升有限**
- 智能分配资源，避免浪费
- 论文中高预算设置下，BudgetMem 超越 Fixed-High 1-3%
- **但要权衡**：训练成本 vs 1-3% 的性能提升是否值得？

**在低预算下：显著优势**
- 关键查询仍能用高级模块
- 简单查询节省成本
- Pareto 前沿明显优于固定策略（成本相同时准确率高 7-9%）

**关键洞见**：BudgetMem 的价值不是"最高准确率"，而是"性能-成本的精确控制"。在生产环境中，能够通过一个参数（λ）灵活调整预算，比固定策略的"要么全用 High，要么全用 Low"更实用。

### 什么情况下值得一试？

**强烈推荐**：
1. **SaaS 产品**：按查询收费，需要精确控制成本
2. **混合查询负载**：客服 chatbot（简单 FAQ + 复杂技术问题）
3. **多阶段 Agent**：检索 → 推理 → 生成，每个阶段都有成本差异

**不建议**：
1. **实时系统**：延迟预算 < 100ms
2. **无标注数据**：冷启动效果差
3. **单一流程**：只有检索或只有生成，tiering 空间有限

### 与其他成本优化方法的对比

| 方法 | 优势 | 劣势 |
|-----|-----|-----|
| **BudgetMem** | 查询感知，精确控制 | 需要训练，增加延迟 |
| **提前终止（Early Exit）** | 无需训练，延迟低 | 无法精确控制成本 |
| **模型蒸馏** | 减少所有查询成本 | 无法保留高成本路径 |
| **Cascading** | 简单有效 | 贪婪决策，非全局最优 |

**结论**：BudgetMem 适合"需要精确预算控制"的场景，而 Cascading 或 Early Exit 更适合"只想降低平均成本"的场景。

---

## 总结

BudgetMem 的核心价值：
1. **显式控制**：通过 λ 参数精确调节性能-成本权衡
2. **查询感知**：简单查询省钱，复杂查询保质量
3. **可扩展**：支持多种 tiering 策略组合

**关键技术点**：
- PPO 训练轻量级路由器（<1M 参数）
- 三种正交 tiering 策略（Implementation/Reasoning/Capacity）
- Reward shaping 和 curriculum learning 确保训练稳定

**实践建议**：
- 先在高预算下验证效果（vs Fixed-High）
- 逐步降低预算，观察 Pareto 曲线
- 混合使用多种 tiering 策略

**论文链接**：https://arxiv.org/abs/2602.06025v1

BudgetMem 不是"又一个 RL 调参地狱"——它的贡献在于提供了**显式、可控的性能-成本权衡框架**，这在生产环境中非常实用。虽然训练路由器需要一定成本，但一旦训练好，推理开销极小（<10ms），且能在多个预算设置下复用。如果你的 Agent 系统有多个模块，查询复杂度差异大，且需要精细控制成本——BudgetMem 值得一试。