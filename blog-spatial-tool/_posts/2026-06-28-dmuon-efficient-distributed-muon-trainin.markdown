---
layout: post-wide
title: "DMuon：让矩阵正交化优化器达到接近 AdamW 的分布式训练开销"
date: 2026-06-28 12:04:05 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.27153v1
generated_by: Claude Code CLI
---

## 一句话总结

通过将 Newton-Schulz 正交化计算与梯度通信流水线化，DMuon 将分布式场景下的优化器步骤加速 6.85x–163x，端到端步骤时间加速 1.48x–3.01x，使 Muon 类优化器在大规模训练中真正可用。

---

## 为什么需要矩阵感知优化器？

AdamW 是当前主流，但它有一个根本性缺陷：**逐元素处理梯度**，完全忽略权重矩阵内部的几何结构。

对于权重矩阵 $W \in \mathbb{R}^{m \times n}$，AdamW 把它当作 $m \times n$ 个独立标量更新：

$$W_{t+1} = W_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

这在现代 Transformer 中是个问题——注意力投影矩阵、FFN 权重的行向量正交性对模型表达能力有实际影响。**Muon** 的解法是：不用原始梯度，而是用梯度矩阵的**正交化版本**来更新权重。

类比：调整多台摄像机朝向，AdamW 独立调每个旋钮，Muon 把所有朝向当整体处理，保证调整后它们彼此正交。

---

## Muon 核心：Newton-Schulz 迭代

Muon 的目标是求梯度矩阵 $G$ 的**极分解正交因子** $UV^T$（其中 $G = U\Sigma V^T$），但不真正做 SVD。

Newton-Schulz 迭代通过反复做矩阵乘法来逼近这个结果：

$$G_0 = \frac{G}{\|G\|_F}$$
$$G_{t+1} = \frac{3}{2}G_t - \frac{1}{2}G_t G_t^T G_t$$

经过 5–10 次迭代后，$G_t$ 的奇异值全部收敛到 1，即 $G_t \approx UV^T$。

**为什么比 SVD 快**：N-S 每步只做两次矩阵乘法（$O(m^2 n)$），没有 SVD 的分支逻辑，天然适合 GPU 批量执行。

代码实现：

```python
import torch

def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    将 2D 梯度矩阵 G 正交化，返回极分解正交因子近似
    要求 dtype=float32 或 bfloat16
    """
    assert G.ndim == 2
    m, n = G.shape
    # 保证行数 <= 列数（否则转置处理）
    transposed = m > n
    if transposed:
        G = G.T

    # 归一化是关键——防止迭代数值爆炸
    G = G / (G.norm() + 1e-7)

    for _ in range(steps):
        A = G @ G.T                    # (m, m)
        G = 1.5 * G - 0.5 * (A @ G)  # cubic Newton-Schulz step

    return G.T if transposed else G
```

---

## 分布式 Muon 的性能瓶颈

Vanilla Muon 在分布式训练中的执行顺序：

```
Forward → Backward → [All-Reduce] → [N-S 迭代 × 5] → 权重更新
                        串行，全程等待
```

以 8 卡 A100、矩阵 4096×4096 为例（论文数据数量级）：

| 阶段 | 时间占比 |
|------|---------|
| Forward + Backward | ~30% |
| All-Reduce | ~25% |
| Newton-Schulz（5 步） | ~45% |

两个最慢的步骤完全串行执行，导致优化器步骤开销**超过 Forward+Backward 的 2 倍**。这就是 Muon 在大规模训练中很少被采用的真实原因。

---

## DMuon 的三个关键优化

### 1. 计算-通信流水线

**核心洞察**：All-Reduce 等待通信期间，GPU 的 SM 处于空闲状态。N-S 是纯计算密集型操作，两者可以**真正并行**。

具体做法：
- 用独立 CUDA Stream 运行 N-S 计算
- All-Reduce 在主流上以异步模式启动
- 两者通过 CUDA Event 同步，保证正确性

### 2. 参数分片减少 N-S 计算量

每个 rank 只对**自己负责的参数分片**做 N-S，然后 All-Gather 汇总结果。N 卡并行时，每个 rank 的 N-S 计算量降至 $1/N$。

### 3. 跨层异步流水

Layer $i$ 的 All-Reduce 进行时，同步计算 Layer $i+1$ 的 N-S。Backward pass 天然提供了这种层间流水的机会。

---

## 代码实现

### Vanilla Muon（Baseline，单 GPU）

```python
class VanillaMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(grad)
                state['buf'].mul_(group['momentum']).add_(grad)

                if grad.dim() == 2:
                    # 矩阵参数：N-S 正交化
                    update = newton_schulz(state['buf'], steps=group['ns_steps'])
                else:
                    # 1D 参数（bias 等）：直接用动量梯度
                    update = state['buf']

                p.add_(update, alpha=-group['lr'])
```

### DMuon（分布式，计算-通信重叠）

```python
import torch.distributed as dist

class DMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))
        self._ns_stream = torch.cuda.Stream()  # 专用于 N-S 计算的独立 Stream

    @torch.no_grad()
    def step(self):
        pending = []  # (param, all_reduce_handle, lr)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.dim() != 2:
                    continue

                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(p.grad)
                state['buf'].mul_(group['momentum']).add_(p.grad)

                # 在独立 Stream 上异步运行 N-S，不阻塞主流的通信
                update = torch.empty_like(p)
                event = torch.cuda.Event()
                with torch.cuda.stream(self._ns_stream):
                    result = newton_schulz(state['buf'].clone(), steps=group['ns_steps'])
                    update.copy_(result)
                    event.record()

                # 主流等待 N-S 完成后再启动 All-Reduce（异步，不阻塞后续循环）
                torch.cuda.current_stream().wait_event(event)
                handle = dist.all_reduce(update, op=dist.ReduceOp.AVG, async_op=True)
                pending.append((p, update, handle, group['lr']))

        # 下一层的 N-S 已在 _ns_stream 上开始；等待本批次通信完成后更新参数
        for p, update, handle, lr in pending:
            handle.wait()
            p.add_(update, alpha=-lr)
```

### 常见错误

```python
# 错误 1：忘记初始归一化，梯度范数大时迭代爆炸
for _ in range(5):
    G = 1.5 * G - 0.5 * G @ G.T @ G  # G.norm() >> 1 时直接 nan

# 错误 2：对 1D 参数（bias、LayerNorm）做 N-S
# bias 形状 (d,)，newton_schulz 会 assert 失败
# 必须按 dim 区分处理（见 VanillaMuon 实现）

# 错误 3：用同一 Stream 跑 N-S 和 All-Reduce（失去并行性）
with torch.cuda.stream(self._ns_stream):
    update = newton_schulz(buf)
    dist.all_reduce(update)  # 正确的并行消失了
```

---

## 性能实测

以下数据来自论文（8× A100，NVLink，LLM/具身智能基础模型工作负载）：

| 实现版本 | 端到端步骤加速 | 优化器步骤加速 | 备注 |
|---------|-------------|-------------|------|
| Vanilla Muon | 1.0x（基准）| 1.0x（基准）| 通信+N-S 完全串行 |
| DMuon | **1.48x–3.01x** | **6.85x–163x** | 与 AdamW 开销接近 |
| AdamW | — | — | 参照对象 |

优化器步骤的极大加速比（最高 163x）出现在模型规模大、通信延迟高的配置中——此时 N-S 与长时间 All-Reduce 的重叠效果最显著。

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 大型 Transformer（大量 2D 权重矩阵） | Embedding 层为主（1D 参数无法 N-S） |
| 多 GPU 分布式训练（有通信开销可重叠） | 单 GPU 训练（无通信，Vanilla Muon 就够） |
| 追求更好收敛质量（Muon > AdamW 在多数任务）| 内存极度受限（N-S 需要临时矩阵） |
| LLM、VLM、具身智能基础模型 | 卷积网络（4D 权重需要 reshape，需额外处理） |

---

## 调试技巧

**验证 N-S 正交性**：

```python
def check_orthogonality(G_out: torch.Tensor) -> float:
    eye = torch.eye(G_out.shape[0], device=G_out.device)
    err = (G_out @ G_out.T - eye).norm().item()
    return err  # 正常应 < 0.01，超过 0.1 说明迭代次数不够或数值问题
```

**Nsight Systems 验证流水线效果**：

在 Nsight 时间线中，`nccl:AllReduce` 和 N-S 的 CUDA kernel（通常是 `gemm_*`）应该在时间轴上**有重叠**。如果发现完全串行，检查：
1. `_ns_stream` 是否是独立 Stream（不能和 default stream 相同）
2. `wait_event` 的位置是否正确（应在 `all_reduce` 启动前，而不是之前）
3. PyTorch 版本是否支持 `async_op=True`（需 PyTorch ≥ 1.10）

**N-S 步数选择**：

```python
# 快速验证：步数越多越正交，但收益递减
for steps in [3, 5, 7, 10]:
    G_out = newton_schulz(G.clone(), steps=steps)
    print(f"steps={steps}, 正交误差={check_orthogonality(G_out):.5f}")
# 大多数场景 steps=5 是甜点
```

---

## 延伸阅读

- **DMuon 论文**：[arxiv 2606.27153](https://arxiv.org/abs/2606.27153v1)，含开源实现
- **Muon 原始工作**：Keller Jordan 等人提出的矩阵正交化优化器概念，搜索 "Muon optimizer" 可找到多个独立复现
- **ZeRO 系列**：理解参数分片思想的必读，DMuon 的分片策略与 ZeRO-1/2 思路相通
- **SOAP 优化器**：另一个矩阵感知优化器方向，与 Muon 互为参照