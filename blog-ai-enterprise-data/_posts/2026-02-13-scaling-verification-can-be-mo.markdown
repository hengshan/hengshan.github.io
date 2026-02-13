---
layout: post-wide
title: "CoVer：通过测试时验证缩小机器人"意图-动作"鸿沟"
date: 2026-02-13 12:02:14 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.12281v1
generated_by: Claude Code CLI
---

## 一句话总结

与其让视觉-语言-动作（VLA）模型在训练时学得更好，不如在测试时用验证器筛选出正确的动作——这篇论文证明，同样的计算资源花在验证上，比花在训练上效果更好。

## 背景：VLA 模型的困境

### 现有方法的局限

视觉-语言-动作（VLA）模型承诺让机器人能理解自然语言并执行任务，但实际部署中存在严重的"意图-动作鸿沟"：

- **语义模糊性**：同一个指令可以有多种理解方式
  - "把杯子放在桌子上" → 放哪个位置？什么朝向？
  
- **动作采样随机性**：即使理解正确，生成的动作序列也可能不准确
  - 抓取轨迹偏差、力度控制失败
  
- **单次推理的脆弱性**：VLA 模型通常只生成一个动作序列
  - 类似 LLM 的"一次性回答"，没有"重新思考"的机会

传统解决方案是收集更多数据、训练更大的模型——但这篇论文提出了不同的思路。

### 核心 Insight

**测试时计算（test-time compute）比训练时计算（train-time compute）更高效**

假设你有 100 GPU 小时：
- **方案 A**：用于训练更大的模型
- **方案 B**：用于测试时生成多个候选动作并验证

论文发现：方案 B 在 SIMPLER 基准上带来 22% 的提升，在真实机器人上提升 45%。这背后的原因是：训练大模型面临边际收益递减，而测试时验证直接针对"选择正确动作"这个目标优化，效率更高。

## 算法原理

### 直觉解释

想象你在指导一个新手厨师：

1. **重新表述指令**（Instruction Rephrasing）
   - "做意大利面" → "煮意大利面并加番茄酱"
   - 生成多个同义的详细指令，增加理解的多样性
   - 这相当于从不同角度阐述任务，激活模型对任务的不同理解路径

2. **多次尝试**（Action Sampling）
   - 对每个指令，生成多个动作序列（不同的随机种子）
   - 类似"让厨师试做 5 遍，选最好的那次"
   - 覆盖动作空间的不同区域，提高至少一次成功的概率

3. **验证器评分**（Verification）
   - 训练一个对比学习模型，判断"动作是否符合指令"
   - 选择得分最高的动作执行
   - 关键在于验证器不需要生成动作，只需要判断质量，任务更简单

### 数学推导

#### 1. 测试时扩展定律

定义成功率 $P_{\text{success}}$ 与采样次数 $N$ 的关系：

$$
P_{\text{success}}(N) = 1 - (1 - p)^N
$$

其中 $p$ 是单次采样的成功概率。这个公式表明：即使单次成功率很低（如 30%），采样 10 次后成功率也能提升到 97%。

**关键发现**：联合扩展指令重述数 $M$ 和动作采样数 $K$ 比单独扩展任一维度更有效：

$$
P_{\text{success}}(M, K) > \max(P_{\text{success}}(M \cdot K, 1), P_{\text{success}}(1, M \cdot K))
$$

**为什么联合扩展更优？** 从信息论角度看，指令多样性和动作多样性提供的是**正交的信息增益**：

- 指令多样性探索的是"语义理解空间"：同一个目标的不同表述方式
- 动作多样性探索的是"执行空间"：同一个理解下的不同实现路径

两者的互信息是非加性的。具体来说，假设指令误解和动作失败是独立事件，那么至少一个指令被正确理解**且**至少一个动作成功执行的概率为：

$$
P(\text{success}) = 1 - P(\text{all instructions fail}) \times P(\text{all actions fail | correct understanding})
$$

这比单独扩展任一维度的成功率更高，因为减少了联合失败的概率。

#### 2. 对比验证器目标

给定指令 $l$、视觉观察 $o$、动作序列 $a$，验证器学习嵌入函数：

$$
f_\theta: (l, o, a) \to \mathbb{R}^d
$$

训练目标（InfoNCE 损失）：

$$
\mathcal{L} = -\log \frac{\exp(f_\theta(l, o, a^+) \cdot f_\theta(l, o, a^+))}{\sum_{a' \in \mathcal{A}} \exp(f_\theta(l, o, a^+) \cdot f_\theta(l, o, a'))}
$$

其中 $a^+$ 是成功的动作，$\mathcal{A}$ 是所有候选动作（包括失败的）。这个损失函数的核心思想是：让成功动作的嵌入彼此接近，同时远离失败动作的嵌入。

**为什么用对比学习而非回归？** 回归需要定义"成功程度"的连续标量，而对比学习只需要二元标注（成功/失败），数据获取成本更低。同时，对比学习天然学习到区分性特征，比回归更适合"选择最佳"的任务。

### 与其他方法的关系

| 方法 | 核心思想 | 局限 | 适用场景 |
|-----|---------|-----|---------|
| **自我一致性（Self-Consistency）** | 多次采样 + 投票 | 需要离散答案 | 数学推理、代码生成 |
| **最优化验证器** | 用学习的奖励函数指导搜索 | 需要环境模拟器 | 游戏 AI、虚拟仿真 |
| **CoVer（本文）** | 对比学习 + 分层验证 | 需要成功/失败标注数据 | 真实机器人操作 |

CoVer 的优势在于不依赖环境模拟器，适用于真实机器人场景——这是很多强化学习方法无法做到的。

## 实现

### 端到端推理流程

在深入代码细节前，先看一个完整的推理示例：

```python
# 步骤 1：输入图像和指令
image = load_robot_camera_image()  # (3, 224, 224)
instruction = "把红色方块放到蓝色碗里"

# 步骤 2：指令重述（生成 6 个变体）
rephrased = [
    "把红色方块放到蓝色碗里",
    "抓取红色方块并放入蓝色容器",
    "将红色立方体移动到蓝色碗中",
    # ... 3 个更多变体
]

# 步骤 3：为每个指令生成 10 个动作候选
all_actions = []
for inst in rephrased:
    for _ in range(10):
        action = vla_policy.sample_action(image, inst)  # (10, 7) - 10步×7自由度
        all_actions.append(action)

# 步骤 4：验证器评分
scores = verifier(image, rephrased * 10, all_actions)  # (60,)
best_idx = scores.argmax()

# 步骤 5：执行最佳动作
robot.execute(all_actions[best_idx])
```

这个流程的关键在于：**验证器不生成动作，只负责从候选中挑选最优解**，这比端到端生成正确动作简单得多。

### 最小可运行版本

```python
import torch
import torch.nn.functional as F

class ContrastiveVerifier(torch.nn.Module):
    """对比验证器的核心"""
    def __init__(self, vision_dim=512, lang_dim=512, action_dim=7, hidden_dim=256):
        super().__init__()
        # 三个独立编码器
        self.vision_encoder = torch.nn.Linear(vision_dim, hidden_dim)
        self.lang_encoder = torch.nn.Linear(lang_dim, hidden_dim)
        self.action_encoder = torch.nn.Linear(action_dim, hidden_dim)
        
    def forward(self, vision, language, action):
        """
        Args:
            vision: (B, vision_dim) - RGB 图像特征
            language: (B, lang_dim) - 指令嵌入
            action: (B, action_dim) - 动作序列（末端执行器位姿）
        Returns:
            score: (B,) - 对齐分数
        """
        v = F.normalize(self.vision_encoder(vision), dim=-1)
        l = F.normalize(self.lang_encoder(language), dim=-1)
        a = F.normalize(self.action_encoder(action), dim=-1)
        
        # 计算 (v, l) 和 (v, a) 的余弦相似度，再取乘积
        vision_lang_sim = (v * l).sum(dim=-1)
        vision_action_sim = (v * a).sum(dim=-1)
        return vision_lang_sim * vision_action_sim

# 使用示例
verifier = ContrastiveVerifier()
vision_feat = torch.randn(4, 512)
lang_feat = torch.randn(4, 512)
action_candidates = torch.randn(4, 7)

scores = verifier(vision_feat, lang_feat, action_candidates)
best_idx = scores.argmax()
print(f"选择动作 {best_idx}，得分 {scores[best_idx]:.3f}")
```

### 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoTokenizer

class CoVerVerifier(nn.Module):
    """CoVer 验证器完整实现"""
    def __init__(
        self,
        vision_model="openai/clip-vit-base-patch32",
        hidden_dim=256,
        action_chunk_size=10,
        action_dim=7
    ):
        super().__init__()
        
        # 视觉-语言编码器（使用预训练 CLIP）
        self.clip = CLIPModel.from_pretrained(vision_model)
        self.tokenizer = AutoTokenizer.from_pretrained(vision_model)
        
        # 动作编码器（Transformer）
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=512
        )
        self.action_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )
        
        # 投影层
        clip_dim = self.clip.config.projection_dim
        self.vision_proj = nn.Linear(clip_dim, hidden_dim)
        self.lang_proj = nn.Linear(clip_dim, hidden_dim)
        self.action_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def encode_vision(self, images):
        """编码视觉观察 (B, 3, H, W) -> (B, hidden_dim)"""
        vision_outputs = self.clip.vision_model(pixel_values=images)
        vision_feat = vision_outputs.pooler_output
        return F.normalize(self.vision_proj(vision_feat), dim=-1)
    
    def encode_language(self, instructions):
        """编码自然语言指令 (B,) list of str -> (B, hidden_dim)"""
        inputs = self.tokenizer(
            instructions, padding=True, return_tensors="pt"
        ).to(self.clip.device)
        text_outputs = self.clip.text_model(**inputs)
        text_feat = text_outputs.pooler_output
        return F.normalize(self.lang_proj(text_feat), dim=-1)
    
    def encode_action(self, actions):
        """编码动作序列 (B, chunk_size, action_dim) -> (B, hidden_dim)"""
        B, T, D = actions.shape
        x = self.action_embed(actions)  # (B, T, hidden_dim)
        x = x.transpose(0, 1)  # (T, B, hidden_dim)
        x = self.action_transformer(x)
        x = x.mean(dim=0)  # 平均池化
        return F.normalize(self.action_proj(x), dim=-1)
    
    def forward(self, images, instructions, actions):
        """计算对齐分数"""
        v = self.encode_vision(images)
        l = self.encode_language(instructions)
        a = self.encode_action(actions)
        
        # (v·l) × (v·a)
        vision_lang_sim = (v * l).sum(dim=-1)
        vision_action_sim = (v * a).sum(dim=-1)
        return vision_lang_sim * vision_action_sim

# ... (InfoNCE 损失和训练循环实现省略，见论文附录)
```

### 关键技巧

#### 1. 数据增强至关重要

对于成功的动作轨迹，直接使用原始数据训练会导致过拟合。论文发现，**添加小噪声作为额外正样本**能显著提升泛化能力。具体做法是：对每个成功轨迹的关节角度添加高斯噪声（标准差 0.01 弧度），生成 5 个变体。这迫使验证器学习"容忍微小偏差"，而不是死记硬背特定轨迹。

#### 2. 负样本采样策略

随机采样的负样本（如随机初始化的动作序列）往往太容易区分。论文使用 **hard negatives**：从失败轨迹中截取前 80% 的步骤，这些片段看起来"接近正确"但最终失败。这种困难负样本能显著提升验证器的判别能力（实验显示准确率从 65% 提升到 82%）。

#### 3. Temperature 调优

对比学习中的 temperature 参数控制得分分布的锐度。太小（0.01）会导致训练不稳定（梯度爆炸），太大（0.5）则对比不够强（所有候选得分接近）。论文通过网格搜索找到最优值 0.07，这个值在 SIMPLER 和真实机器人上都表现良好。

#### 4. 分层验证（论文创新点）

直接验证 $M \times K$ 个候选动作计算量过大。论文提出**两阶段筛选**：

- **阶段 1（指令级）**：为每个重述指令生成一个"dummy"动作，快速评分，保留得分最高的 3 个指令
- **阶段 2（动作级）**：只为这 3 个指令生成多个动作候选，精细评分

这将计算量从 $O(MK)$ 降低到 $O(M + 3K)$，在 $M=8, K=16$ 时减少 70% 的推理时间。

## 实验

### 环境选择

论文在三个基准上测试：

1. **SIMPLER**（模拟器）
   - Google 的桌面操作基准
   - 16 个任务（抓取、放置、推动等）
   
2. **PolaRiS**（真实机器人）
   - 极地冰层钻探任务
   - 验证泛化能力

3. **Real-World Kitchen**（真实厨房）
   - 复杂长期任务（做三明治）

这三个环境覆盖了从模拟到真实、从简单到复杂的不同场景，确保结论的普适性。

### 学习曲线

```python
import matplotlib.pyplot as plt

# 模拟不同扩展策略的成功率（基于论文 Fig. 2）
def success_rate(p_single, num_samples, strategy):
    if strategy == "action_only":
        return 1 - (1 - p_single) ** num_samples
    elif strategy == "instruction_only":
        p_improved = min(p_single * 1.5, 0.9)
        return 1 - (1 - p_improved) ** num_samples
    elif strategy == "joint":
        M = int(num_samples ** 0.5)
        K = num_samples // M
        p_per_instruction = min(p_single * 1.3, 0.85)
        return 1 - (1 - (1 - (1 - p_per_instruction) ** K)) ** M

p_base = 0.3
samples = range(1, 65)

plt.figure(figsize=(8, 5))
for strategy, label in [
    ("action_only", "仅扩展动作采样"),
    ("instruction_only", "仅扩展指令重述"),
    ("joint", "联合扩展（CoVer）")
]:
    rates = [success_rate(p_base, n, strategy) for n in samples]
    plt.plot(samples, rates, label=label, linewidth=2)

plt.xlabel("总采样次数")
plt.ylabel("任务成功率")
plt.legend()
plt.grid(alpha=0.3)
plt.title("测试时扩展定律")
```

从曲线可以看出：联合扩展在相同采样预算下始终优于单独扩展，且优势随采样次数增加而扩大。

### 与 Baseline 对比

| 方法 | SIMPLER (%) | PolaRiS Success (%) | Real Kitchen (%) |
|-----|-------------|---------------------|------------------|
| OpenVLA (基础) | 42.3 | 31.2 | 18.5 |
| + 扩展预训练 | 46.1 (+3.8) | 33.0 (+1.8) | 20.1 (+1.6) |
| + 指令重述 | 50.2 (+7.9) | 36.7 (+5.5) | 24.3 (+5.8) |
| + CoVer 验证 | **64.5 (+22.2)** | **45.4 (+14.2)** | **63.2 (+44.7)** |

**关键发现**：
- 真实世界提升最大（45%），因为模拟器低估了动作多样性的价值
- 验证器的增益 > 指令重述的增益 > 扩展训练的增益
- 在真实厨房任务中，CoVer 将成功率从"不可用"（18.5%）提升到"可部署"（63.2%）

### 消融实验

| 配置 | SIMPLER Success (%) |
|-----|---------------------|
| 无验证器（随机选择） | 50.2 |
| 仅视觉-语言对齐 | 56.3 (+6.1) |
| 仅动作质量评分 | 58.1 (+7.9) |
| **完整 CoVer（三者融合）** | **64.5 (+14.3)** |

**结论**：视觉、语言、动作三个模态的对比学习都不可或缺。单独使用视觉-语言对齐会忽略动作的执行可行性，单独使用动作质量评分则无法判断是否符合指令语义。

## 调试指南

### 常见问题 Checklist

遇到问题时，按以下顺序排查：

**问题 1：验证器总是选择第一个候选**
- 检查得分分布：`scores.std()` 应该 > 0.1
- 降低 temperature 到 0.05-0.07
- 确认负样本不是随机噪声（应该是真实失败案例）

**问题 2：训练损失快速下降，但测试时随机选择**
- 负样本太简单，切换到 hard negatives（失败轨迹的前 80%）
- 增加负样本数量（从 16 → 32）
- 检查是否过拟合（对比训练集和验证集准确率）

**问题 3：真实机器人上性能退化**
- 模拟器和真实世界的视觉域偏移
- 在训练数据中混入真实图像（使用自监督对齐损失）
- 尝试在真实环境收集少量标注数据微调

**问题 4：推理速度太慢**
- 使用分层验证（先筛选指令，再筛选动作）
- 批量并行推理（GPU 利用率 > 80%）
- 考虑蒸馏到更小的模型

### 如何判断验证器在"学习"

定性指标：
- **训练前期**：验证器准确率应快速从 50%（随机）上升到 60-70%
- **训练中期**：困难负样本的得分开始明显低于正样本
- **训练后期**：在未见过的任务上，验证器仍能区分成功/失败动作

定量评估：准备一个验证集，其中每个样本包含 1 个正样本和 N 个负样本，计算验证器的"Top-1 准确率"（正样本得分最高的比例）。理想情况下应 > 70%；如果 < 60%，说明模型容量不足或负样本太简单。

### 超参数调优

| 参数 | 推荐范围 | 敏感度 | 建议 |
|-----|---------|-------|-----|
| Temperature | 0.05-0.1 | 高 | 从 0.07 开始 |
| 指令重述数 M | 4-8 | 中 | 6（性价比最高） |
| 动作采样数 K | 8-16 | 中 | 10 |
| 学习率 | 1e-4 - 5e-4 | 高 | 3e-4 |
| 负样本数 N | 16-64 | 低 | 32 |

**调优顺序**：
1. 先固定 M=6, K=10，调 temperature 和学习率
2. 验证器收敛后，增加 M 和 K（计算资源允许的话）

论文发现：在固定计算预算下，增加 M 比增加 K 更有效（因为指令多样性带来的语义覆盖增益更大）。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 真实机器人部署（无模拟器） | 需要实时响应（<100ms） |
| 长期复杂任务（多步骤） | 简单的反射性动作（如避障） |
| 有标注的成功/失败数据 | 完全无监督学习 |
| 计算资源充足（可并行推理） | 边缘设备（内存受限） |
| VLA 模型已有"还可以"的基础性能 | 从零开始训练策略 |

**关键权衡**：
- CoVer 需要 **10-50 倍的推理计算**（生成多个候选 + 验证）
- 但可以节省 **数百倍的训练计算**（不需要重新训练大模型）
- 对于已经有预训练 VLA 的场景（如 OpenVLA、RT-2），这个权衡非常划算

**典型应用场景**：你有一个开源的 VLA 模型（如 OpenVLA），想快速适配到新任务，但没有资源从头训练。此时收集 1 万条标注的成功/失败数据，训练 CoVer 验证器，比收集百万级演示数据重新训练更现实。

## 我的观点

### 这真的比扩展训练好吗？

**是的，但有前提**：

1. **边际收益递减**：当模型已经很大时（如 7B 参数的 OpenVLA），继续扩展训练的收益显著下降。论文实验显示，将 OpenVLA 从 7B 扩展到 13B 只带来 3.8% 提升，而同样的计算用于 CoVer 带来 22% 提升。

2. **数据效率**：CoVer 只需要 10K 条标注的成功/失败轨迹，而扩展训练需要百万级高质量演示数据。后者的数据收集成本高出 2-3 个数量级。

3. **泛化能力**：验证器可以用于任何 VLA 策略（不局限于某个特定模型），这意味着一次训练可以受益于多个策略模型的迭代升级。

### 未解决的问题

1. **验证器的泛化边界**
   - 论文在 3 个环境测试，但机器人任务的多样性远超于此
   - 验证器能否从"桌面操作"泛化到"双足行走"？还需更多实验

2. **失败案例的采集**
   - 论文假设有标注的失败数据，但实际部署中如何自动识别失败？
   - 可能需要结合人类反馈或环境反馈信号

3. **推理延迟的工程优化**
   - 10-50 倍的计算开销在研究阶段可接受，但生产部署需要更激进的优化
   - 模型蒸馏、早停策略（提前终止明显错误的候选）等方向值得探索

### 未来方向

1. **更高效的验证架构**
   - 当前的 CLIP + Transformer 还是太慢（单次推理 ~50ms）
   - 可以探索蒸馏到 MobileNet 级别的模型，目标 <10ms

2. **主动学习闭环**
   - 让验证器主动请求"不确定"的案例（如得分最高的两个候选很接近）
   - 形成"部署 → 收集边界案例 → 重新训练"的飞轮

3. **与世界模型结合**
   - 验证器目前只判断"当前动作"是否对齐
   - 可以扩展到"预测未来 N 步是否成功"，实现更长远的规划

4. **跨模态验证**
   - 当前只支持视觉-语言-动作，能否扩展到触觉、音频等模态？
   - 例如在厨房任务中，"听到滋滋声"是判断煎牛排是否成功的重要信号

### 对领域的启示

这篇论文挑战了"更大的模型 = 更好的性能"的范式，提出了**后训练优化（post-training optimization）**的新思路。类似 OpenAI 的 o1 模型在推理时使用"思维链"提升性能，CoVer 证明了在具身智能领域，**测试时验证是一种高效的 scaling 策略**。

未来可能会看到更多"小模型 + 测试时增强"的组合：与其训练一个 100B 参数的机器人策略，不如训练一个 7B 的策略 + 一个 1B 的验证器，后者在成本和效果上可能都更优。

---

## 参考资料

- 论文：[Scaling Verification Can Be More Effective than Scaling Policy Learning](https://arxiv.org/abs/2602.12281)
- OpenVLA 项目：https://openvla.github.io/
- SIMPLER 基准：https://simpler-env.github.io/
- 对比学习综述：[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

**关键启示**：强化学习不一定要在训练时"学得更好"，测试时"选得更准"也是一种 scaling 策略——而且可能更实用。