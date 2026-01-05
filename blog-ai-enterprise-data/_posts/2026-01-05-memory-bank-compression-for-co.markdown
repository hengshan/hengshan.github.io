---
layout: post-wide
title: "Memory Bank Compression for Continual Adaptation of Large Language Models"
date: 2026-01-05 15:48:52 +0800
category: AI
author: Hank Li
source_url: https://arxiv.org/abs/2601.00756v1
generated_by: AI Agent
---

# 大模型持续学习的记忆库压缩：从理论到实践

## 问题设定：持续学习中的记忆困境

在强化学习和持续学习（Continual Learning）的交叉领域中，我们面临一个核心挑战：如何让大语言模型（LLM）在不断接收新知识的同时，既不忘记旧知识（避免灾难性遗忘），又不让存储成本无限增长？

**MDP视角下的问题建模**：
- **状态 (State)**：当前LLM的参数状态 + 外部记忆库内容
- **动作 (Action)**：选择更新哪些参数、如何压缩记忆库
- **奖励 (Reward)**：新任务准确率 - 旧任务遗忘率 - 记忆库存储成本
- **转移 (Transition)**：数据流到达后的模型更新过程

这是一个**在线学习（Online Learning）**问题，类似于off-policy RL，我们需要在数据流中持续优化策略，同时平衡exploration（学习新知识）和exploitation（保持旧知识）。

**与传统RL算法的关系**：
- 类似DQN的经验回放机制（记忆库存储历史信息）
- 借鉴Actor-Critic思想（LLM主体 + 记忆库辅助）
- 引入VQ-VAE的离散化表示（码本压缩）

---

## 算法原理：记忆库压缩的数学基础

### 核心思想

传统记忆增强方法将每个样本的表示直接存入记忆库，导致存储量线性增长。MBC（Memory Bank Compression）通过**向量量化（Vector Quantization）**将连续的记忆表示映射到离散码本，实现指数级压缩。

### 数学推导

**1. 记忆库的价值函数**

定义记忆库的价值为其对模型性能的贡献：

$$
V(\mathcal{M}) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{test}}} [\log P_\theta(y|x, \mathcal{M})]
$$

其中 $\mathcal{M}$ 是记忆库，$\theta$ 是LLM参数。

**2. 压缩目标函数**

引入码本 $\mathcal{C} = \{c_1, ..., c_K\}$（K个码字），优化目标：

$$
\min_{\mathcal{C}, \phi} \mathbb{E}_{m \sim \mathcal{M}} [\|m - c_{\phi(m)}\|^2] + \lambda \cdot \text{Size}(\mathcal{C})
$$

- $\phi(m)$：编码器，将记忆 $m$ 映射到最近的码字索引
- $\lambda$：存储成本权重

**3. 在线更新的Bellman方程**

在时间步 $t$，接收新数据 $(x_t, y_t)$：

$$
\mathcal{C}_{t+1} = \arg\min_{\mathcal{C}} \sum_{i=1}^{t} \alpha^{t-i} \|m_i - c_{\phi(m_i)}\|^2
$$

指数衰减因子 $\alpha$ 平衡新旧数据重要性。

**4. 防止码本坍缩的重置机制**

当码字 $c_k$ 的使用频率低于阈值 $\tau$ 时：

$$
c_k \leftarrow m_{\text{worst}} + \epsilon, \quad \text{where} \quad m_{\text{worst}} = \arg\max_{m \in \mathcal{M}_t} \|m - c_{\phi(m)}\|
$$

将未使用码字重置为重建误差最大的记忆附近。

### 算法伪代码

```
Algorithm: MBC (Memory Bank Compression)

Input: 
  - LLM with LoRA adapters θ
  - Codebook C = {c₁, ..., cₖ}
  - Data stream {(xₜ, yₜ)}ₜ₌₁^∞
  
Hyperparameters:
  - Codebook size K
  - Reset threshold τ
  - Learning rate η

For each timestep t:
  1. 提取新样本的键值表示：
     m_t = Encoder(x_t)
  
  2. 向量量化（找最近码字）：
     idx = argmin_k ||m_t - c_k||
     m̂_t = c_idx
  
  3. 使用压缩记忆增强推理：
     ŷ_t = LLM(x_t | {m̂_1, ..., m̂_t})
  
  4. 计算损失并更新：
     L = CrossEntropy(ŷ_t, y_t) + β||m_t - m̂_t||²
     θ ← θ - η∇_θ L
     C ← C - η∇_C L
  
  5. 码本重置（防止坍缩）：
     For each c_k:
       If usage(c_k) < τ:
         c_k ← m_worst + noise
  
  6. 更新记忆库索引：
     M_indices.append(idx)
```

**关键创新点**：
1. **在线码本优化**：不需要预训练，边学习边压缩
2. **自适应重置**：防止码字"死亡"，保持表达能力
3. **LoRA集成**：仅更新低秩适配器，降低计算成本

---

## 实现：简单环境（问答任务）

### 环境定义

我们使用一个简化的问答数据流环境：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# 模拟持续学习的问答数据流
class ContinualQADataset(Dataset):
    """
    模拟数据流：每个时间段有不同主题的问答对
    模拟概念漂移（concept drift）场景
    """
    def __init__(self, num_topics=5, samples_per_topic=100, vocab_size=1000):
        self.data = []
        self.topics = []
        
        # 生成不同主题的问答对
        for topic_id in range(num_topics):
            for _ in range(samples_per_topic):
                # 简化表示：每个主题有特定的token分布
                question = torch.randint(
                    topic_id * 200, 
                    (topic_id + 1) * 200, 
                    (20,)  # 问题长度20
                )
                answer = torch.randint(
                    topic_id * 200, 
                    (topic_id + 1) * 200, 
                    (10,)  # 答案长度10
                )
                self.data.append((question, answer))
                self.topics.append(topic_id)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.topics[idx]

# 创建数据流
def create_data_stream(batch_size=8):
    dataset = ContinualQADataset(num_topics=5, samples_per_topic=100)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

### 核心组件实现

#### 1. 向量量化码本

```python
class VectorQuantizer(nn.Module):
    """
    向量量化模块：将连续记忆映射到离散码本
    
    核心思想：类似VQ-VAE，但针对在线学习优化
    """
    def __init__(
        self, 
        codebook_size: int = 256,      # 码本大小（K）
        embedding_dim: int = 768,       # 记忆向量维度
        commitment_cost: float = 0.25,  # 承诺损失权重（β）
        decay: float = 0.99,            # EMA衰减率
        epsilon: float = 1e-5
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 码本：K个d维向量
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1/codebook_size, 1/codebook_size)
        
        # 用于EMA更新的统计量
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', self.codebook.weight.data.clone())
        
        self.decay = decay
        self.epsilon = epsilon
        
        # 记录每个码字的使用频率（用于重置）
        self.register_buffer('usage_count', torch.zeros(codebook_size))
        self.reset_threshold = 0.01  # 使用率低于1%时重置
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            z: 输入记忆 [batch_size, embedding_dim]
        
        Returns:
            z_q: 量化后的记忆
            loss: VQ损失
            indices: 码字索引
        """
        # 展平输入
        z_flattened = z.view(-1, self.embedding_dim)
        
        # 计算距离：||z - c_k||^2 = ||z||^2 + ||c_k||^2 - 2<z, c_k>
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.codebook.weight.t())
        )
        
        # 找到最近的码字索引
        indices = torch.argmin(distances, dim=1)
        
        # 更新使用计数
        self.usage_count.index_add_(
            0, indices, torch.ones_like(indices, dtype=torch.float)
        )
        
        # 量化：用码字替换原始向量
        z_q = self.codebook(indices).view(z.shape)
        
        # 计算VQ损失
        # 1. Codebook loss: 让码字靠近输入
        # 2. Commitment loss: 让输入靠近码字（防止输入空间膨胀）
        codebook_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = F.mse_loss(z_q, z.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator：前向用量化值，反向传梯度给输入
        z_q = z + (z_q - z).detach()
        
        return z_q, loss, indices
    
    def update_codebook_ema(self, z: torch.Tensor, indices: torch.Tensor):
        """
        使用指数移动平均（EMA）更新码本
        比梯度下降更稳定，适合在线学习
        """
        indices_onehot = F.one_hot(indices, self.codebook_size).float()
        
        # 更新聚类大小
        self.cluster_size.mul_(self.decay).add_(
            indices_onehot.sum(0), alpha=1 - self.decay
        )
        
        # 拉普拉斯平滑
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.epsilon) 
            / (n + self.codebook_size * self.epsilon) * n
        )
        
        # 更新码字的EMA
        embed_sum = torch.matmul(indices_onehot.t(), z.view(-1, self.embedding_dim))
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        
        # 更新码本权重
        self.codebook.weight.data.copy_(self.embed_avg / cluster_size.unsqueeze(1))
    
    def reset_unused_codes(self):
        """
        重置未充分使用的码字
        防止码本坍缩（codebook collapse）
        """
        total_usage = self.usage_count.sum()
        if total_usage == 0:
            return
        
        usage_freq = self.usage_count / total_usage
        unused_mask = usage_freq < self.reset_threshold
        
        if unused_mask.any():
            # 找到重建误差最大的样本
            # 实际中应该从最近的batch中采样
            # 这里简化为随机重置
            num_reset = unused_mask.sum().item()
            new_codes = torch.randn(num_reset, self.embedding_dim).to(self.codebook.weight.device)
            self.codebook.weight.data[unused_mask] = new_codes * 0.1
            
            # 重置使用计数
            self.usage_count[unused_mask] = 0
            
            print(f"重置了 {num_reset} 个未使用的码字")
```

#### 2. LoRA低秩适配器

```python
class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) 层
    
    原理：W' = W + BA，其中B∈R^(d×r), A∈R^(r×k), r<<d
    只训练A和B，冻结原始权重W
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int = 8,
        alpha: float = 16.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA的两个低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, in_features]
        """
        # LoRA增量：x @ A @ B
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return lora_output

class LoRALinear(nn.Module):
    """
    带LoRA的线性层：组合原始权重和LoRA增量
    """
    def __init__(
        self, 
        linear_layer: nn.Linear, 
        rank: int = 8, 
        alpha: float = 16.0
    ):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False  # 冻结原始权重
        
        self.lora = LoRALayer(
            linear_layer.in_features, 
            linear_layer.out_features, 
            rank, 
            alpha
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出 + LoRA增量
        return self.linear(x) + self.lora(x)
```

#### 3. 记忆增强的LLM

```python
class MemoryAugmentedLLM(nn.Module):
    """
    带记忆库的大语言模型
    
    架构：
    1. 编码器：提取输入的键值表示
    2. 向量量化：压缩记忆
    3. 记忆检索：从码本中检索相关记忆
    4. LLM推理：使用检索到的记忆增强生成
    """
    def __init__(
        self, 
        hidden_dim: int = 768,
        codebook_size: int = 256,
        lora_rank: int = 8,
        max_memory_size: int = 1000
    ):
        super().__init__()
        
        # 简化的编码器（实际中应使用预训练LLM的encoder）
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 向量量化器
        self.vq = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=hidden_dim
        )
        
        # LoRA适配器（应用于注意力层的Q、K、V）
        self.lora_q = LoRALayer(hidden_dim, hidden_dim, rank=lora_rank)
        self.lora_k = LoRALayer(hidden_dim, hidden_dim, rank=lora_rank)
        self.lora_v = LoRALayer(hidden_dim, hidden_dim, rank=lora_rank)
        
        # 简化的解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 记忆库：存储码字索引而非原始向量
        self.memory_indices = deque(maxlen=max_memory_size)
        self.memory_keys = deque(maxlen=max_memory_size)  # 用于检索的键
    
    def encode_memory(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码输入为记忆表示并量化
        
        Returns:
            memory: 原始记忆向量
            quantized_memory: 量化后的记忆
            vq_loss: 向量量化损失
            indices: 码字索引
        """
        # 提取记忆表示
        memory = self.encoder(x.mean(dim=1))  # [batch, hidden_dim]
        
        # 向量量化
        quantized_memory, vq_loss, indices = self.vq(memory)
        
        return memory, quantized_memory, vq_loss, indices
    
    def retrieve_memory(self, query: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """
        从记忆库中检索相关记忆
        
        Args:
            query: 查询向量 [batch, hidden_dim]
            top_k: 检索top-k个记忆
        
        Returns:
            retrieved: 检索到的记忆 [batch, top_k, hidden_dim]
        """
        if len(self.memory_indices) == 0:
            return torch.zeros(query.size(0), top_k, query.size(-1)).to(query.device)
        
        # 计算查询与所有记忆键的相似度
        memory_keys_tensor = torch.stack(list(self.memory_keys)).to(query.device)
        similarities = torch.matmul(query, memory_keys_tensor.t())  # [batch, num_memories]
        
        # 选择top-k
        top_k = min(top_k, len(self.memory_indices))
        topk_indices = torch.topk(similarities, k=top_k, dim=1).indices
        
        # 从码本中检索
        retrieved = []
        for batch_idx in range(query.size(0)):
            batch_memories = []
            for k_idx in topk_indices[batch_idx]:
                code_idx = self.memory_indices[k_idx.item()]
                memory_vector = self.vq.codebook.weight[code_idx]
                batch_memories.append(memory_vector)
            retrieved.append(torch.stack(batch_memories))
        
        return torch.stack(retrieved)
    
    def forward(
        self, 
        x: torch.Tensor, 
        use_memory: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入 [batch, seq_len, hidden_dim]
            use_memory: 是否使用记忆增强
        
        Returns:
            output: 模型输出
            vq_loss: 向量量化损失
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # 1. 编码当前输入为记忆
        memory, quantized_memory, vq_loss, indices = self.encode_memory(x)
        
        # 2. 检索相关记忆
        if use_memory and len(self.memory_indices) > 0:
            retrieved_memories = self.retrieve_memory(memory, top_k=5)
            # 将检索到的记忆拼接到输入
            memory_context = retrieved_memories.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        else:
            memory_context = torch.zeros_like(x)
        
        # 3. 使用LoRA增强的注意力
        q = x + self.lora_q(x)
        k = (x + memory_context) + self.lora_k(x + memory_context)
        v = (x + memory_context) + self.lora_v(x + memory_context)
        
        # 简化的注意力计算
        attn_weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (hidden_dim ** 0.5), 
            dim=-1
        )
        attn_output = torch.matmul(attn_weights, v)
        
        # 4. 解码
        output = self.decoder(attn_output)
        
        return output, vq_loss
    
    def update_memory(self, keys: torch.Tensor, indices: torch.Tensor):
        """
        更新记忆库（只存储索引，不存储原始向量）
        """
        for key, idx in zip(keys, indices):
            self.memory_keys.append(key.detach().cpu())
            self.memory_indices.append(idx.item())
```

### 训练循环

```python
class MBCTrainer:
    """
    MBC训练器：在线持续学习
    """
    def __init__(
        self, 
        model: MemoryAugmentedLLM,
        learning_rate: float = 1e-4,
        vq_weight: float = 0.25,
        reset_interval: int = 100
    ):
        self.model = model
        self.vq_weight = vq_weight
        self.reset_interval = reset_interval
        
        # 只优化LoRA参数和VQ码本
        self.optimizer = torch.optim.Adam([
            {'params': model.lora_q.parameters()},
            {'params': model.lora_k.parameters()},
            {'params': model.lora_v.parameters()},
            {'params': model.vq.parameters()},
        ], lr=learning_rate)
        
        # 记录训练指标
        self.metrics = {
            'loss': [],
            'vq_loss': [],
            'accuracy': [],
            'memory_size': [],
            'codebook_usage': []
        }
    
    def train_step(
        self, 
        questions: torch.Tensor, 
        answers: torch.Tensor
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            questions: [batch, seq_len, hidden_dim]
            answers: [batch, seq_len, hidden_dim]
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        output, vq_loss = self.model(questions, use_memory=True)
        
        # 计算任务损失（简化为MSE，实际应为交叉熵）
        task_loss = F.mse_loss(output, answers)
        
        # 总损失
        total_loss = task_loss + self.vq_weight * vq_loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        # 更新码本（EMA）
        with torch.no_grad():
            memory, _, _, indices = self.model.encode_memory(questions)
            self.model.vq.update_codebook_ema(memory, indices)
            self.model.update_memory(memory, indices)
        
        return {
            'loss': total_loss.item(),
            'vq_loss': vq_loss.item(),
            'task_loss': task_loss.item()
        }
    
    def evaluate(
        self, 
        test_loader: DataLoader, 
        topic_id: int = None
    ) -> float:
        """
        评估模型在特定主题上的性能
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for questions, answers, topics in test_loader:
                if topic_id is not None:
                    mask = topics == topic_id
                    if not mask.any():
                        continue
                    questions = questions[mask]
                    answers = answers[mask]
                
                # 转换为浮点并添加特征维度
                questions = questions.float().unsqueeze(-1).expand(-1, -1, 768)
                answers = answers.float().unsqueeze(-1).expand(-1, -1, 768)
                
                output, _ = self.model(questions, use_memory=True)
                loss = F.mse_loss(output, answers)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train_online(
        self, 
        data_stream: DataLoader, 
        num_steps: int = 1000,
        eval_interval: int = 100
    ):
        """
        在线训练主循环
        """
        step = 0
        data_iter = iter(data_stream)
        
        # 用于评估的历史主题数据
        topic_test_data = {i: [] for i in range(5)}
        
        while step < num_steps:
            try:
                questions, answers, topics = next(data_iter)
            except StopIteration:
                data_iter = iter(data_stream)
                questions, answers, topics = next(data_iter)
            
            # 保存测试数据
            for i in range(len(topics)):
                topic_id = topics[i].item()
                if len(topic_test_data[topic_id]) < 20:
                    topic_test_data[topic_id].append(
                        (questions[i], answers[i])
                    )
            
            # 转换为模型输入格式
            questions = questions.float().unsqueeze(-1).expand(-1, -1, 768)
            answers = answers.float().unsqueeze(-1).expand(-1, -1, 768)
            
            # 训练步
            metrics = self.train_step(questions, answers)
            
            # 记录指标
            self.metrics['loss'].append(metrics['loss'])
            self.metrics['vq_loss'].append(metrics['vq_loss'])
            self.metrics['memory_size'].append(len(self.model.memory_indices))
            
            # 计算码本使用率
            usage = (self.model.vq.usage_count > 0).sum().item()
            self.metrics['codebook_usage'].append(
                usage / self.model.vq.codebook_size
            )
            
            # 定期重置未使用的码字
            if step % self.reset_interval == 0 and step > 0:
                self.model.vq.reset_unused_codes()
            
            # 定期评估
            if step % eval_interval == 0:
                print(f"\n步骤 {step}/{num_steps}")
                print(f"损失: {metrics['loss']:.4f} | VQ损失: {metrics['vq_loss']:.4f}")
                print(f"记忆库大小: {len(self.model.memory_indices)}")
                print(f"码本使用率: {self.metrics['codebook_usage'][-1]:.2%}")
                
                # 评估所有历史主题（检测遗忘）
                for topic_id in range(5):
                    if len(topic_test_data[topic_id]) > 0:
                        # 创建临时测试集
                        test_questions = torch.stack([x[0] for x in topic_test_data[topic_id]])
                        test_answers = torch.stack([x[1] for x in topic_test_data[topic_id]])
                        test_questions = test_