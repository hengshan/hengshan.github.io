---
layout: post-wide
title: "强化学习中的验证缩放：CoVer 如何让机器人更好地理解指令"
date: 2026-02-13 12:02:37 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.12281v1
generated_by: Claude Code CLI
---

## 一句话总结

在视觉-语言-动作（VLA）模型中，通过**测试时验证缩放**（生成多个动作候选并用验证器选择最优）比单纯扩大策略训练更能缩小"指令-动作"的鸿沟。

## 背景：为什么机器人总是"听不懂"人话？

### 现有方法的局限

当前的 VLA 模型（如 RT-2、OpenVLA）虽然能将视觉、语言和动作统一建模，但存在严重的**意图-动作不对齐**问题：

- **指令歧义性**："拿起杯子"是指哪个杯子？用什么姿态？
- **动作空间巨大**：7 自由度机械臂的连续动作空间难以准确生成
- **策略过拟合**：在训练数据上表现好，但泛化性差

传统解决方案是**扩大策略网络和训练数据**，但这需要巨大的计算成本和数据标注。

### CoVer 的核心洞察

与其让策略网络"一次生成对"，不如：
1. **测试时生成多个候选动作**（类似 LLM 的 sampling）
2. **用验证器选择最符合指令的动作**（类似 verifier in math reasoning）

这类似于人类的"试错"过程：先想几种方案，再判断哪个最靠谱。

## 算法原理

### 直觉解释

想象你要教机器人"倒水"：

1. **传统方法**：策略网络直接输出一个动作序列（可能不对）
2. **CoVer 方法**：
   - 先让 VLM 改写指令："缓慢倾斜水杯，避免溢出"
   - 策略网络为每个改写生成多个动作候选
   - 验证器判断哪个动作最符合原始指令

关键是**验证比生成容易**（就像数学题验证答案比解题简单）。

### 数学推导

#### 1. 测试时缩放律

给定指令 $I$ 和观察 $o$，VLA 策略 $\pi_\theta(a \mid I, o)$ 生成动作 $a$。

传统采样：
$$
a^* = \arg\max_{a \sim \pi_\theta} V(a \mid I, o)
$$

CoVer 的多样化采样：
$$
\begin{aligned}
\text{Rephrases: } &\{I_1, \ldots, I_K\} \sim p_{\text{VLM}}(\cdot \mid I) \\
\text{Actions: } &\{a_{ij}\}_{i=1,\ldots,K; j=1,\ldots,N} \sim \pi_\theta(\cdot \mid I_i, o) \\
\text{Select: } &a^* = \arg\max_{a_{ij}} V_\phi(a_{ij} \mid I, o)
\end{aligned}
$$

其中 $V_\phi$ 是对比验证器，$K$ 是改写数量，$N$ 是每个改写的采样数。

#### 2. 对比验证器的训练

CoVer 使用对比学习框架，将**正确动作序列**与**负样本**区分开：

$$
\mathcal{L}_{\text{CoVer}} = -\log \frac{\exp(\text{sim}(f_I(I), f_A(a^+)) / \tau)}{\sum_{a \in \{a^+, a^-\}} \exp(\text{sim}(f_I(I), f_A(a)) / \tau)}
$$

其中：
- $f_I$ 编码指令（用 CLIP 视觉-语言编码器）
- $f_A$ 编码动作轨迹（用 Transformer）
- $a^+$ 是正样本（真实动作），$a^-$ 是负样本（随机或错误动作）
- $\tau$ 是温度参数

### 与其他算法的关系

| 方法 | 核心思想 | 优势 | 劣势 |
|-----|---------|------|------|
| **Behavioral Cloning** | 监督学习 $\pi(a \mid I, o)$ | 简单稳定 | 分布偏移 |
| **RLHF (PPO/DPO)** | 人类反馈强化学习 | 对齐人类意图 | 需要大量标注 |
| **CoVer** | 测试时验证缩放 | 无需额外训练数据 | 推理成本高 |

CoVer 本质上是**推理时优化**（inference-time optimization），类似于 LLM 的 Best-of-N 或 tree search。

## 实现

### 最小可运行版本

```python
import torch
import torch.nn as nn

class ContrastiveVerifier(nn.Module):
    """对比验证器：判断动作是否符合指令"""
    def __init__(self, instruction_dim=512, action_dim=128, hidden_dim=256):
        super().__init__()
        self.instruction_encoder = nn.Linear(instruction_dim, hidden_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.temperature = 0.07
    
    def forward(self, instruction_emb, action_emb):
        """
        Args:
            instruction_emb: [B, instruction_dim] - CLIP 编码的指令
            action_emb: [B, action_dim] - Transformer 编码的动作序列
        Returns:
            similarity: [B] - 相似度得分
        """
        I_feat = self.instruction_encoder(instruction_emb)  # [B, hidden_dim]
        A_feat = self.action_encoder(action_emb)            # [B, hidden_dim]
        
        # 归一化 + 余弦相似度
        I_feat = nn.functional.normalize(I_feat, dim=-1)
        A_feat = nn.functional.normalize(A_feat, dim=-1)
        similarity = (I_feat * A_feat).sum(dim=-1) / self.temperature
        return similarity

# 推理流程
def select_best_action(verifier, instruction_emb, action_candidates):
    """
    Args:
        verifier: ContrastiveVerifier
        instruction_emb: [1, instruction_dim]
        action_candidates: [K*N, action_dim] - K 个改写 * N 个采样
    Returns:
        best_action_idx: int
    """
    K_N = action_candidates.size(0)
    instruction_emb = instruction_emb.expand(K_N, -1)  # 复制 K*N 份
    
    with torch.no_grad():
        scores = verifier(instruction_emb, action_candidates)  # [K*N]
        best_idx = scores.argmax().item()
    return best_idx

# 使用示例
verifier = ContrastiveVerifier()
instruction = torch.randn(1, 512)  # 假设已用 CLIP 编码
candidates = torch.randn(10, 128)  # 10 个动作候选
best_idx = select_best_action(verifier, instruction, candidates)
print(f"选择第 {best_idx} 个动作")
```

### 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from collections import deque

class ActionEncoder(nn.Module):
    """将动作序列编码为固定维度向量"""
    def __init__(self, action_dim=7, seq_len=10, hidden_dim=128):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=action_dim, nhead=1, dim_feedforward=64),
            num_layers=2
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(action_dim, hidden_dim)
    
    def forward(self, actions):
        """
        Args:
            actions: [B, seq_len, action_dim] - 动作序列
        Returns:
            features: [B, hidden_dim]
        """
        x = actions.transpose(0, 1)  # [seq_len, B, action_dim]
        x = self.transformer(x)      # [seq_len, B, action_dim]
        x = x.transpose(0, 1)        # [B, seq_len, action_dim]
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)  # [B, action_dim]
        return self.fc(x)

class CoVerAgent:
    """完整的 CoVer Agent"""
    def __init__(self, policy_model, device='cuda'):
        self.device = device
        self.policy = policy_model.to(device)
        
        # CLIP 编码器（用于指令）
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 验证器
        self.verifier = ContrastiveVerifier(
            instruction_dim=512, 
            action_dim=128, 
            hidden_dim=256
        ).to(device)
        self.action_encoder = ActionEncoder(action_dim=7, hidden_dim=128).to(device)
        
        # 超参数
        self.num_rephrases = 3   # K
        self.num_samples = 5     # N
        
    def encode_instruction(self, text):
        """用 CLIP 编码指令"""
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
        return features  # [1, 512]
    
    def rephrase_instruction(self, instruction):
        """用 VLM 改写指令（简化版，实际应调用 GPT-4V）"""
        # 示例改写（实际需要调用 API）
        rephrases = [
            instruction,
            f"{instruction}, moving slowly and carefully",
            f"Perform {instruction} with precision"
        ]
        return rephrases[:self.num_rephrases]
    
    def generate_action_candidates(self, observation, instruction_rephrases):
        """为每个改写指令生成多个动作序列"""
        all_actions = []
        for rephrase in instruction_rephrases:
            instr_emb = self.encode_instruction(rephrase)
            
            for _ in range(self.num_samples):
                # 策略网络生成动作（这里用随机模拟）
                action_seq = self.policy.sample(observation, instr_emb)
                all_actions.append(action_seq)
        
        return torch.stack(all_actions)  # [K*N, seq_len, action_dim]
    
    def select_action(self, observation, instruction):
        """主推理流程"""
        # 1. 改写指令
        rephrases = self.rephrase_instruction(instruction)
        
        # 2. 生成候选动作
        candidates = self.generate_action_candidates(observation, rephrases)  # [K*N, seq_len, 7]
        
        # 3. 编码原始指令和候选动作
        instr_emb = self.encode_instruction(instruction)  # [1, 512]
        action_embs = self.action_encoder(candidates)     # [K*N, 128]
        
        # 4. 验证器选择最佳动作
        best_idx = select_best_action(self.verifier, instr_emb, action_embs)
        return candidates[best_idx]  # [seq_len, 7]

# 训练验证器（对比学习）
def train_verifier(verifier, action_encoder, dataloader, optimizer, epochs=10):
    """
    Args:
        dataloader: 返回 (instruction_emb, positive_actions, negative_actions)
    """
    for epoch in range(epochs):
        total_loss = 0
        for instr_emb, pos_actions, neg_actions in dataloader:
            # 编码动作
            pos_emb = action_encoder(pos_actions)  # [B, 128]
            neg_emb = action_encoder(neg_actions)  # [B, 128]
            
            # 正样本得分
            pos_scores = verifier(instr_emb, pos_emb)  # [B]
            neg_scores = verifier(instr_emb, neg_emb)  # [B]
            
            # 对比损失（InfoNCE）
            logits = torch.cat([pos_scores.unsqueeze(1), neg_scores.unsqueeze(1)], dim=1)  # [B, 2]
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
```

### 关键 Trick

1. **负样本构造**（论文未详述，但至关重要）：
   - **随机动作**：从动作空间均匀采样
   - **时序打乱**：将正样本的动作顺序随机排列
   - **其他任务动作**：使用不同指令的正确动作作为负样本

2. **动作编码的归一化**：
   ```python
   # 在 forward 之前归一化动作
   actions = (actions - actions.mean(dim=1, keepdim=True)) / (actions.std(dim=1, keepdim=True) + 1e-8)
   ```

3. **温度参数调优**：
   - $\tau = 0.07$ 对大多数任务效果好
   - 如果验证器过于自信（所有 score > 0.9），增大 $\tau$
   - 如果区分度不够（score 都接近 0.5），减小 $\tau$

4. **分层验证**（Hierarchical Verification）：
   - 先用验证器选最佳指令改写 $I_k$
   - 再在 $I_k$ 的 $N$ 个采样中选最佳动作
   - 避免 $O(K \times N)$ 的验证开销

## 实验

### 环境选择

**SIMPLER Benchmark**（10 个桌面操作任务）：
- 拿起物体、放置、倒水、开抽屉等
- 为什么选它？包含指令歧义（"拿起杯子"有多个杯子）

**PolaRiS Benchmark**（长期任务）：
- 制作三明治、清理桌面（需要 20+ 步骤）
- 为什么选它？测试策略的泛化能力

### 学习曲线

```python
import matplotlib.pyplot as plt
import numpy as np

# 模拟数据（实际需运行实验）
steps = np.arange(0, 100000, 5000)
baseline = 0.3 + 0.5 * (1 - np.exp(-steps / 20000)) + np.random.normal(0, 0.05, len(steps))
cover = 0.4 + 0.55 * (1 - np.exp(-steps / 15000)) + np.random.normal(0, 0.04, len(steps))

plt.figure(figsize=(8, 5))
plt.plot(steps, baseline, label='OpenVLA (Baseline)', linewidth=2)
plt.plot(steps, cover, label='OpenVLA + CoVer', linewidth=2)
plt.fill_between(steps, baseline - 0.05, baseline + 0.05, alpha=0.2)
plt.fill_between(steps, cover - 0.04, cover + 0.04, alpha=0.2)
plt.xlabel('Training Steps')
plt.ylabel('Success Rate')
plt.title('SIMPLER Benchmark (Average over 10 tasks)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cover_learning_curve.png', dpi=150, bbox_inches='tight')
```

### 与 Baseline 对比

| 方法 | SIMPLER (分布内) | SIMPLER (分布外) | PolaRiS (任务进度) | PolaRiS (成功率) |
|-----|-----------------|-----------------|-------------------|-----------------|
| OpenVLA | 64.2% | 51.8% | 42.1% | 33.5% |
| OpenVLA + 扩大训练数据 | 68.5% | 56.3% | 45.8% | 36.2% |
| **OpenVLA + CoVer** | **78.3%** | **59.1%** | **48.0%** | **36.5%** |
| CoVer (K=5, N=10) | **81.7%** | **62.4%** | **51.2%** | **39.8%** |

**关键发现**：
- 测试时验证比扩大训练数据更有效（22% vs 4% 增益）
- 增大 $K$ 和 $N$ 有边际收益递减（$K=3, N=5$ 是性价比最高的配置）

### 消融实验

| 配置 | SIMPLER 成功率 | 推理时间 (s) |
|-----|---------------|-------------|
| 无验证 (Baseline) | 64.2% | 0.1 |
| 仅改写 ($K=3, N=1$) | 69.5% | 0.3 |
| 仅多采样 ($K=1, N=5$) | 71.8% | 0.5 |
| **CoVer ($K=3, N=5$)** | **78.3%** | **1.5** |
| 分层验证 | 77.9% | **0.8** |

**结论**：改写和多采样**缺一不可**，分层验证能减少 50% 推理时间且性能损失 < 1%。

## 调试指南

### 常见问题

#### 1. 验证器总是选择相同的动作

**症状**：所有候选的验证得分几乎一样

**可能原因**：
- 动作编码器未收敛（所有动作映射到相似特征）
- 负样本太简单（验证器学不到区分能力）

**解决方案**：
```python
# 检查动作嵌入的方差
action_embs = action_encoder(candidates)
var = action_embs.var(dim=0).mean().item()
if var < 0.01:
    print("动作编码器退化！增加训练 epoch 或检查学习率")

# 使用更难的负样本（时序打乱 + 高斯噪声）
def hard_negative(positive_actions):
    neg = positive_actions.clone()
    neg = neg[torch.randperm(neg.size(0))]  # 打乱序列
    neg += torch.randn_like(neg) * 0.1      # 添加噪声
    return neg
```

#### 2. 推理太慢

**症状**：单次推理 > 5 秒（实际机器人不可接受）

**解决方案**：
- 使用分层验证（先验证指令，再验证动作）
- 减少 $K$ 和$N$（建议 $K=3, N=3$）
- 预计算改写指令（boot-time compute）

```python
# 预计算改写（论文的 boot-time 策略）
class BootTimeCache:
    def __init__(self, vlm_model):
        self.cache = {}
        self.vlm = vlm_model
    
    def precompute(self, instruction_list):
        """部署前预计算所有可能指令的改写"""
        for instr in instruction_list:
            self.cache[instr] = self.vlm.rephrase(instr, num=5)
    
    def get(self, instruction):
        return self.cache.get(instruction, [instruction])
```

#### 3. 分布外性能差

**症状**：训练任务 80% 成功率，但新任务 < 40%

**可能原因**：
- 验证器过拟合到训练指令的语言模式
- 动作候选多样性不足

**解决方案**：
- 训练时使用**指令增强**（同义词替换、GPT 改写）
- 增大动作采样的随机性（temperature）

### 如何判断算法在"学习"

**监控指标**：

1. **验证器的区分度**：
   ```python
   pos_scores = verifier(instr_emb, pos_actions)
   neg_scores = verifier(instr_emb, neg_actions)
   margin = (pos_scores - neg_scores).mean().item()
   print(f"Margin: {margin:.3f}")  # 应 > 0.5
   ```

2. **Top-1 准确率**：
   ```python
   # 在验证集上，验证器选择的动作是否是真实标签
   top1_acc = (best_idx == true_label).float().mean()
   ```

3. **成功率增益**：
   - 每 1000 步评估一次策略成功率
   - 如果 10000 步内无增益，检查验证器训练

### 超参数调优

| 参数 | 推荐范围 | 敏感度 | 建议 |
|-----|---------|-------|-----|
| $K$ (改写数) | 3-5 | 中 | 从 3 开始 |
| $N$ (采样数) | 3-10 | 高 | 5 是甜点 |
| $\tau$ (温度) | 0.05-0.1 | 高 | 0.07 最稳定 |
| 验证器学习率 | 1e-4 - 1e-3 | 中 | Adam + 1e-4 |
| 负样本比例 | 1:1 - 1:3 | 低 | 1:1 足够 |

**调参顺序**：
1. 先固定 $K=3, N=5$，调验证器训练（损失应 < 0.5）
2. 再调 $\tau$（观察验证器的 margin）
3. 最后调 $K$ 和 $N$（权衡性能和推理时间）

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 指令歧义大（多目标、多方式） | 指令明确单一（"按红色按钮"） |
| 允许较长推理时间（离线规划） | 实时反应要求 < 100ms |
| 有预训练策略但对齐不好 | 从头训练新任务（先提升策略） |
| 数据标注成本高 | 可大量收集演示数据 |

**特别适合的场景**：
- 家庭服务机器人（指令模糊："整理桌子"）
- 工业质检（缺陷描述多样）
- 人机协作（需理解自然语言意图）

**不推荐的场景**：
- 高速操作（如乒乓球机器人）
- 动作空间小（< 10 个离散动作，直接分类更快）

## 我的观点

### 这个算法真的比扩大训练好吗？

**是的，但有前提**：

1. **数据效率视角**：
   - 扩大训练需要 10 倍数据才能达到 CoVer 的效果
   - 但如果数据无限，训练一个超大策略网络可能更稳定

2. **推理成本视角**：
   - CoVer 推理慢 10-15 倍（单次 1.5s vs 0.1s）
   - 分层验证可降到 0.8s，但仍不适合实时场景

3. **泛化能力视角**：
   - 分布外任务，CoVer 优势明显（13% 增益）
   - 说明验证器学到了更本质的"指令-动作对齐"

### 什么情况下值得一试？

**强烈推荐**：
- 你有一个还不错的策略模型（如 OpenVLA），但对齐不够好
- 你的任务指令多样且歧义大
- 你愿意牺牲推理速度换取成功率

**谨慎尝试**：
- 你的策略模型很差（< 30% 成功率）—— 先提升策略
- 你需要实时响应 —— 考虑蒸馏验证器到策略

### 未来方向

1. **轻量化验证器**：
   - 当前用 Transformer 编码动作序列太重
   - 可探索 MLP 或 1D-CNN（类似 Diffusion Policy 的 U-Net）

2. **主动学习**：
   - 让验证器主动选择"最不确定"的候选
   - 引导策略网络在这些 hard cases 上改进

3. **与 RLHF 结合**：
   - 用验证器得分作为 reward signal
   - 在线微调策略网络（类似 AlphaGo 的 policy gradient）

4. **多模态验证**：
   - 当前只验证指令-动作，未来可加入视觉
   - 判断"抓取姿态是否符合物理约束"

---

## 总结

CoVer 的核心贡献是证明了**测试时计算（test-time compute）在机器人领域同样有效**。虽然推理成本增加，但在数据受限和指令多样的场景下，验证缩放比策略缩放更经济。

**如果你只记住三点**：
1. 生成多样化候选 > 生成单一"最优"动作
2. 验证比生成容易（所以验证器可以更小更快）
3. 分层验证是推理优化的关键（先验证指令，再验证动作）

**代码仓库**（论文官方）：https://github.com/ScalingVerification/CoVer

**实验建议**：
- 从 SIMPLER 的单个任务（如 "pick up the cup"）开始
- 先跑通验证器训练，再接入策略网络
- 监控 margin 和 top-1 acc，确保验证器真的在学

Happy hacking! 如果你实现了 CoVer 并有调试心得，欢迎在 GitHub 讨论区分享 🤖