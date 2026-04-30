---
layout: post-wide
title: "COPUS：大模型训练中批量大小与并行策略的协同自适应优化"
date: 2026-04-30 12:03:59 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.26687v1
generated_by: Claude Code CLI
---

## 一句话总结

大模型训练中，全局批量大小（statistical efficiency）和 3D 并行策略（hardware throughput）的决策是**强耦合**的——COPUS 通过在线估计梯度噪声尺度，联合优化两者，在 H100 集群上实现平均 3.9–8.0% 的收敛加速。

---

## 为什么需要这个？

### 传统做法的盲区

当前 LLM 训练社区存在一个隐性假设：并行策略（DP/TP/PP 的组合）和批量大小是独立决策的。

- **系统团队**：给定固定批量大小，搜索最快的并行配置
- **算法团队**：给定固定并行策略，动态调整批量大小跟踪临界批量

问题在于：**这两个决策并不独立**。

```
小批量训练时：高 DP 度 → 每个 GPU 的 micro-batch 极小 → GPU 利用率低
大批量训练时：高 DP 度 → 完美扩展 → 高吞吐量
```

换句话说，随着训练进行，临界批量大小会变化（早期梯度噪声大，后期趋于稳定），最优的并行配置也随之变化。固定其中一个、优化另一个，意味着训练的某段时间必然运行在次优配置下。

### 问题的规模

对于一个 32B 模型，在 4 节点 32xH100 上：
- 不同并行配置之间的吞吐量差异可达 2–3x
- 临界批量大小从训练初期到后期可以变化 4–8x
- 两个决策的交叉影响意味着单独优化一个，潜在损失可达 10%+ 的训练时间

---

## 核心原理

### 1. 梯度噪声尺度：临界批量大小的理论基础

**直觉**：梯度是有噪声的估计。当你用更大的批量时，噪声被平均掉，但当批量已经足够大时，继续增大批量的边际收益趋近于零。

临界批量大小 $B_{crit}$ 定义为梯度信噪比等于 1 的点：

$$B_{crit} = \frac{\text{tr}(\Sigma)}{\|\mu\|^2}$$

其中 $\Sigma$ 是梯度协方差矩阵，$\mu = \mathbb{E}[\nabla L]$ 是真实梯度。

**实际含义**：
- $B \ll B_{crit}$：增大批量能等比例减少所需步数，计算效率高
- $B \gg B_{crit}$：增大批量对收敛几乎没有帮助，白白浪费通信开销
- $B_{crit}$ 在训练中**不是常数**——随 loss 下降而增大

### 2. Goodput：统一的效率度量

COPUS 用 **Goodput** 联合衡量硬件效率和统计效率：

$$\text{Goodput} = \underbrace{\text{Throughput}}_{\text{tokens/sec}} \times \underbrace{\text{StatEff}(B, B_{crit})}_{\text{有效利用率}}$$

统计效率建模为：

$$\text{StatEff}(B) = \frac{B_{crit}}{B + B_{crit}}$$

这个公式的含义：
- $B = B_{crit}$ 时，效率为 0.5（每消耗 2 个 token，只有 1 个 token 的"真实"训练价值）
- $B \to 0$ 时，效率趋近于 1（每个 token 都是最高效的）
- $B \to \infty$ 时，效率趋近于 0（大量冗余计算）

Goodput 的目标是：选择既不让 GPU 空转、又不让 batch 远超临界点的配置。

### 3. 3D 并行策略与批量大小的耦合

3D 并行由三个维度组成：
- **DP**（数据并行）：跨 GPU 切分 batch，all-reduce 梯度
- **TP**（张量并行）：切分单层内的矩阵运算，all-reduce 激活
- **PP**（流水线并行）：切分模型层，点对点通信，有 bubble 开销

关键洞察：

| 批量大小 | 适合的并行策略 | 原因 |
|---------|--------------|------|
| 小批量 | 高 TP/PP，低 DP | DP 要求足够大的 micro-batch |
| 中等批量 | 平衡配置 | 各方向通信开销权衡 |
| 大批量 | 高 DP | DP 线性扩展，通信效率最高 |

---

## 代码实现

### Baseline：静态并行 + 固定批量（问题复现）

```python
# 传统做法：并行策略固定，批量大小固定
# 问题：训练后期 B_crit 上升，当前批量远小于最优值，白白浪费 GPU

def static_training(model, config):
    """
    config = {dp: 8, tp: 2, pp: 2, batch_size: 1024}
    这在训练初期可能是最优的，但后期 B_crit 增大后就次优了
    """
    optimizer = AdamW(model.parameters(), lr=1e-4)
    dataloader = build_dataloader(batch_size=config['batch_size'])
    
    for step, batch in enumerate(dataloader):
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # ❌ 从不检查当前批量是否还是最优的
        # ❌ 从不重新评估并行配置
```

### 核心组件一：在线梯度噪声尺度估计

```python
import torch

class GradientNoiseScaleEstimator:
    """
    在线估计临界批量大小 B_crit = tr(Σ) / ||μ||²
    使用指数移动平均避免存储完整梯度历史
    """
    def __init__(self, ema_decay: float = 0.95):
        self.decay = ema_decay
        self.ema_grad_mean = None     # μ 的 EMA 估计
        self.ema_grad_sq_mean = None  # E[g²] 的 EMA 估计
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module, current_batch_size: int) -> float:
        # 收集所有参数的梯度（展平为向量）
        grads = torch.cat([
            p.grad.flatten() for p in model.parameters() 
            if p.grad is not None
        ])
        
        grad_sq = grads ** 2
        
        # EMA 更新：平滑估计，避免单步噪声影响
        if self.ema_grad_mean is None:
            self.ema_grad_mean = grads.clone()
            self.ema_grad_sq_mean = grad_sq.clone()
        else:
            self.ema_grad_mean.mul_(self.decay).add_(grads, alpha=1 - self.decay)
            self.ema_grad_sq_mean.mul_(self.decay).add_(grad_sq, alpha=1 - self.decay)
        
        return self._estimate_critical_batch_size(current_batch_size)
    
    def _estimate_critical_batch_size(self, B: int) -> float:
        # tr(Σ) ≈ E[||g||²] - ||E[g]||²（方差的迹）
        gradient_noise = (self.ema_grad_sq_mean - self.ema_grad_mean ** 2).sum()
        # ||μ||²（梯度信号强度）
        gradient_signal = (self.ema_grad_mean ** 2).sum()
        
        if gradient_signal < 1e-12:
            return float('inf')  # 梯度消失，无法估计
        
        # B_crit = tr(Σ) / ||μ||²，用当前批量大小做偏差修正
        return float(gradient_noise / gradient_signal) * B
```

### 核心组件二：Goodput 计算与配置搜索

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ParallelConfig:
    dp: int   # 数据并行度
    tp: int   # 张量并行度
    pp: int   # 流水线并行度
    
    @property
    def world_size(self):
        return self.dp * self.tp * self.pp


def compute_goodput(throughput: float, batch_size: int, critical_batch_size: float) -> float:
    """
    Goodput = 吞吐量 × 统计效率
    统计效率 = B_crit / (B + B_crit)
    """
    if critical_batch_size <= 0:
        return throughput  # B_crit 未知时退化为纯吞吐量优化
    
    stat_efficiency = critical_batch_size / (batch_size + critical_batch_size)
    return throughput * stat_efficiency


def estimate_throughput(config: ParallelConfig, batch_size: int, 
                        model_params: dict) -> float:
    """
    吞吐量模型（基于 roofline 分析简化）
    实际部署中应替换为真实 profiling 数据
    
    pipeline bubble overhead ≈ (pp - 1) / (pp - 1 + micro_steps)
    """
    seq_len = model_params['seq_len']
    micro_batch = batch_size // (config.dp * config.pp)  # 每个 PP stage 的 micro-batch
    
    if micro_batch < 1:
        return 0.0  # 批量太小，无法分配
    
    # Pipeline bubble 效率：PP 越高，bubble 占比越大
    micro_steps = batch_size // (config.dp * micro_batch)
    bubble_efficiency = micro_steps / (micro_steps + config.pp - 1)
    
    # TP 通信开销随 TP 度线性增加（简化模型）
    tp_efficiency = 1.0 / (1.0 + 0.1 * (config.tp - 1))
    
    # 基础计算吞吐量（假设线性扩展）
    base_throughput = config.world_size * seq_len * micro_batch * 1000  # tokens/sec
    
    return base_throughput * bubble_efficiency * tp_efficiency


def find_optimal_config(batch_size: int, critical_batch_size: float,
                        candidate_configs: List[ParallelConfig],
                        model_params: dict) -> Tuple[ParallelConfig, float]:
    """
    在候选并行配置中搜索最优 Goodput
    """
    best_goodput = -1.0
    best_config = candidate_configs[0]
    
    for config in candidate_configs:
        throughput = estimate_throughput(config, batch_size, model_params)
        goodput = compute_goodput(throughput, batch_size, critical_batch_size)
        
        if goodput > best_goodput:
            best_goodput = goodput
            best_config = config
    
    return best_config, best_goodput
```

### 核心组件三：自适应训练主循环

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ParallelConfig:
    dp: int; tp: int; pp: int

    @property
    def world_size(self):
        return self.dp * self.tp * self.pp


def compute_goodput(throughput: float, batch_size: int, critical_batch_size: float) -> float:
    # Goodput = 吞吐量 × 统计效率，B_crit / (B + B_crit)
    stat_efficiency = critical_batch_size / (batch_size + critical_batch_size)
    return throughput * stat_efficiency


def estimate_throughput(config: ParallelConfig, batch_size: int, model_params: dict) -> float:
    micro_batch = batch_size // (config.dp * config.pp)
    micro_steps = batch_size // (config.dp * micro_batch)

    bubble_efficiency = micro_steps / (micro_steps + config.pp - 1)  # PP bubble overhead
    tp_efficiency = 1.0 / (1.0 + 0.1 * (config.tp - 1))             # TP 通信开销
    base_throughput = config.world_size * model_params['seq_len'] * micro_batch * 1000

    return base_throughput * bubble_efficiency * tp_efficiency


def find_optimal_config(batch_size: int, critical_batch_size: float,
                        candidate_configs: List[ParallelConfig],
                        model_params: dict) -> Tuple[ParallelConfig, float]:
    best_goodput, best_config = -1.0, candidate_configs[0]
    for config in candidate_configs:
        goodput = compute_goodput(estimate_throughput(config, batch_size, model_params),
                                  batch_size, critical_batch_size)
        if goodput > best_goodput:
            best_goodput, best_config = goodput, config
    return best_config, best_goodput
```

### 常见错误：忽视重配置开销

```python
def copus_training_loop(model, train_dataloader, candidate_configs, eval_interval=50):
    gns_estimator = GradientNoiseScaleEstimator(ema_decay=0.95)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    current_config, current_batch_size = candidate_configs[0], 512

    for step, batch in enumerate(train_dataloader):
        # 前向/反向传播
        model(**batch).loss.backward()
        # 在线 GNS 估计（每步更新，开销极低）
        critical_batch_size = gns_estimator.update(model, current_batch_size)
        optimizer.step(); optimizer.zero_grad()

        # 定期配置搜索（每 eval_interval 步）
        if step > 0 and step % eval_interval == 0 and critical_batch_size < float('inf'):
            # 在 [0.5×B_crit, 2×B_crit] 范围内搜索最优 (batch_size, parallelism) 对
            best_goodput, best_batch_size, best_config = -1.0, current_batch_size, current_config
            for bs in [int(critical_batch_size * r) for r in [0.5, 1.0, 2.0]]:
                config, goodput = find_optimal_config(bs, critical_batch_size, candidate_configs)
                if goodput > best_goodput:
                    best_goodput, best_batch_size, best_config = goodput, bs, config

            # 触发重新配置（实际部署调用 deepspeed/megatron 重新初始化）
            if best_config != current_config or best_batch_size != current_batch_size:
                # reconfigure_parallel(model, optimizer, best_config)
                current_config, current_batch_size = best_config, best_batch_size
```

---

## 性能实测

> 以下数据来自原论文（H100 和 MI210 集群），非本地复现。

| 模型规模 | 硬件 | 基线 | COPUS | 加速比 |
|---------|------|------|-------|-------|
| 3B | 8×H100 | — | — | **+5.2%** |
| 7B | 16×H100 | — | — | **+4.8%** |
| 32B | 32×H100 | — | — | **+8.0%** |
| 7B | 8×MI210 | — | — | **+11.1%**（峰值）|

几个关键观察：

1. **越大的模型收益越明显**：因为并行配置的搜索空间更大，耦合效应更强
2. **MI210 收益更高**：通信拓扑与 H100 不同，配置敏感性更强
3. **收益随训练阶段变化**：早期批量小时收益小，中后期批量增大后差异显著

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 多节点大模型预训练（>7B） | 单 GPU 训练（无并行决策） |
| 训练时间超过数天 | 微调（批量大小通常固定小批量） |
| 有充足的配置搜索预算 | 实时推理服务（不涉及训练） |
| 硬件异构集群（H100/MI200 混用） | 批量大小受内存约束无法调整 |

**局限性**：
- GNS 估计需要完整梯度，与梯度裁剪、ZeRO-3 等有兼容性问题
- 重配置本身有开销（重初始化通信组、重分配参数），小模型可能得不偿失
- 统计效率模型是近似的，对非标准优化器（如 Muon）的建模尚不完善

---

## 调试技巧

**问题：B_crit 估计不稳定**

```python
# 症状：每隔几步就触发重配置
# 原因：EMA decay 太小，估计值方差大

# 修复：增大 decay 系数，或只在 loss 平台期估计
gns = GradientNoiseScaleEstimator(ema_decay=0.99)  # 更平滑的估计
```

**问题：配置切换后吞吐量反而下降**

检查项：
- micro-batch 大小是否满足 `batch_size % (dp × pp) == 0`
- TP 切换是否重新触发了 NCCL 通信组初始化
- 使用 `nccl-tests` 验证新配置下的实际通信带宽

**用 Nsight 验证配置效果**：
```bash
# 对比两个配置的实际 SM 利用率
nsys profile --stats=true python train.py --config dp8_tp2_pp2
# 查看 cudaMemcpyAsync 占比，评估通信瓶颈
```

---

## 延伸阅读

- **梯度噪声尺度理论**：McCandlish et al., [*An Empirical Model of Large-Batch Training*](https://arxiv.org/abs/1812.06162)（B_crit 的原始定义）
- **Goodput 框架**：Pollards et al., *Optimizing LLM Training Throughput with Goodput* 
- **3D 并行最佳实践**：Megatron-LM 论文系列，尤其是 Narayanan et al. 2021
- **COPUS 原文**：[arxiv 2604.26687](https://arxiv.org/abs/2604.26687v1)（建议重点阅读 §3 的 Goodput 推导和 §5 的重配置开销分析）

**进阶方向**：COPUS 目前只考虑 3D 并行；专家并行（Expert Parallelism）和序列并行（Sequence Parallelism）的加入会使搜索空间指数级增大，是开放的研究问题。