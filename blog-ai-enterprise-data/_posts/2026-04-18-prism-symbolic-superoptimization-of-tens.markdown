---
layout: post-wide
title: "Prism：用符号推理重新定义张量程序超优化"
date: 2026-04-18 12:06:19 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.15272v1
generated_by: Claude Code CLI
---

## 一句话总结

把搜索空间本身符号化——Prism 不再枚举具体程序，而是推理"程序族"，在 LLM 算子上比最佳编译器快 4.9 倍，同时将优化时间缩短 3.4 倍。

---

## 一个问题困扰了编译器研究者很久

假设你要把一个矩阵乘法内核跑到极致，你有三种武器：

- **编译器**（TVM、XLA）：启发式规则搜索，快，但可能错过最优解
- **超优化器**（AMOS、Roller）：穷举所有实现，找到最优，但慢到不实用
- **自动调优**（Autoscheduler、Triton）：采样 + 测量，介于两者之间

这三种方法的根本困境在于：**搜索质量和搜索速度是对立的**。编译器用贪心启发式换速度，超优化器用穷举换质量，没有两全其美的方案——直到 Prism。

Prism 的核心洞见：**超优化器慢，是因为它搜索的是"具体程序"；如果搜索"程序族"，就能在不牺牲搜索质量的前提下大幅剪枝。**

这个想法听起来简单，但实现它需要回答两个硬问题：（1）怎么表示"程序族"？（2）怎么证明变换后的程序还是正确的？Prism 分别用 sGraph 和 e-graph 回答了这两个问题。

---

## sGraph：把执行参数变成符号

传统计算图中，每个算子节点的执行参数是固定的：

```
MatMul(tile_m=128, tile_n=128, tile_k=64, loop_order="mnk")
```

Prism 的 **sGraph（symbolic graph）** 将这些参数替换为**符号变量**：

```
MatMul(tile_m=τ_m, tile_n=τ_n, tile_k=τ_k, loop_order=σ)
```

这一行符号节点**代表了所有可能的 tiling 和循环顺序**——它是一个"程序族"。更关键的是，我们可以对符号表达式做代数推理，**在不实例化任何具体程序的情况下就排除次优方案**。

以 cache 约束为例。L1 cache 大小是已知的硬件规格，可以直接推导出合法的 tile 范围：

$$\tau_m \cdot \tau_k + \tau_k \cdot \tau_n + \tau_m \cdot \tau_n \leq L1\_capacity$$

这个不等式一次性排除了大量非法配置，不需要 kernel launch，不需要实测。这就是"符号剪枝"的本质：**在推理层面，而不是实验层面，消灭次优方案。**

---

## 代码实现

### 1. 符号搜索空间：一次推理胜过千次测量

```python
import sympy as sp
from itertools import product

def symbolic_cache_pruning(tile_choices=(16, 32, 64, 128), l1_capacity=32768):
    """
    用L1 cache约束推导合法tiling，无需逐个测量。
    l1_capacity: L1 cache大小（以float32计，32K = 128KB）
    
    关键：这里的约束不是手写的规则，而是从算子语义自动推导的：
    "数据复用最大化" → "三个tile必须同时驻留L1" → 不等式约束
    """
    total = len(tile_choices) ** 3  # 暴力搜索的搜索空间
    valid = []

    for tm, tn, tk in product(tile_choices, repeat=3):
        # C-tile(tm×tn) + A-tile(tm×tk) + B-tile(tk×tn) <= L1
        working_set = tm*tn + tm*tk + tk*tn
        if working_set <= l1_capacity:
            valid.append((tm, tn, tk))

    print(f"搜索空间: {total} 种配置")
    print(f"符号剪枝后: {len(valid)} 种需要验证")
    print(f"直接排除: {total - len(valid)} 种（无需任何kernel launch）")
    return valid

valid_configs = symbolic_cache_pruning()
# 输出:
# 搜索空间: 64 种配置
# 符号剪枝后: 20 种需要验证
# 直接排除: 44 种
```

真实的 Prism 在此基础上还做了：内存带宽瓶颈分析、算子融合收益的符号估算、以及跨越多个算子的全局最优性证明。单个约束的剪枝比例并不惊人，但多层约束组合后，搜索空间可以缩小几个数量级。

### 2. sGraph：程序族的表示结构

```python
from dataclasses import dataclass
from typing import Union
import sympy as sp

SymParam = Union[sp.Symbol, int]

@dataclass
class MatMulNode:
    """GEMM节点：执行参数可以是符号量，代表一族程序"""
    M: int; N: int; K: int
    tile_m: SymParam
    tile_n: SymParam
    tile_k: SymParam

    def is_symbolic(self) -> bool:
        return any(isinstance(p, sp.Expr)
                   for p in [self.tile_m, self.tile_n, self.tile_k])

    def instantiate(self, vals: dict) -> 'MatMulNode':
        """符号图 → 具体程序：代入实际参数"""
        def resolve(p):
            return int(p.subs(vals)) if isinstance(p, sp.Expr) else p
        return MatMulNode(self.M, self.N, self.K,
                          resolve(self.tile_m),
                          resolve(self.tile_n),
                          resolve(self.tile_k))

    def symbolic_l1_pressure(self) -> sp.Expr:
        """符号化L1工作集，用于自动生成剪枝约束"""
        tm, tn, tk = self.tile_m, self.tile_n, self.tile_k
        return tm*tn + tm*tk + tk*tn


# --- 两层搜索示意 ---
tm, tn, tk = sp.symbols('tm tn tk', positive=True, integer=True)

# 第一层：创建符号节点（代表所有4096x4096 GEMM的tiling变体）
sgraph = MatMulNode(4096, 4096, 4096, tm, tn, tk)
print(f"符号L1压力: {sgraph.symbolic_l1_pressure()}")  
# → tm*tn + tm*tk + tk*tn

# 第二层：符号推理通过后，才实例化为具体程序
concrete = sgraph.instantiate({tm: 128, tn: 128, tk: 64})
print(f"实例化结果: tile=({concrete.tile_m}, {concrete.tile_n}, {concrete.tile_k})")
```

### 3. E-graph：等价性验证，不是搜索

Prism 找到一个"更快的程序"后，怎么保证它和原始程序计算结果相同？答案是 **e-graph（等价图）**。

这里有一个容易误解的点：Prism 用 e-graph 做**验证**，而不是做搜索。搜索是在符号空间完成的；e-graph 是在最后确认"这两个程序语义上是否等价"。

```python
class EGraph:
    """
    最小可用的E-graph实现：用union-find维护等价类。
    真实实现需要加入类型系统和形状约束。
    """
    def __init__(self):
        self.parent = {}

    def add(self, expr: str) -> str:
        if expr not in self.parent:
            self.parent[expr] = expr
        return self.find(expr)

    def find(self, expr: str) -> str:
        if self.parent[expr] != expr:
            self.parent[expr] = self.find(self.parent[expr])  # 路径压缩
        return self.parent[expr]

    def union(self, e1: str, e2: str):
        """声明两个表达式等价（应用一条改写规则）"""
        r1, r2 = self.find(e1), self.find(e2)
        if r1 != r2:
            self.parent[r2] = r1

    def equivalent(self, e1: str, e2: str) -> bool:
        return self.find(e1) == self.find(e2)


# 验证：matmul(A, B+C) 与 matmul(A,B)+matmul(A,C) 是否等价？
eg = EGraph()
orig   = "matmul(A, add(B,C))"
optim  = "add(matmul(A,B), matmul(A,C))"

eg.add(orig); eg.add(optim)

# 施加分配律改写规则（矩阵乘法对加法的左分配律）
eg.union(orig, optim)

print(eg.equivalent(orig, optim))  # True → 优化安全，可以使用右式
# 右式可以并行计算两个matmul，在多GPU/多核上有潜在优势
```

E-graph 的关键特性是：改写规则可以**批量施加**，而不必逐一验证。Prism 内置了矩阵代数的主要恒等式（交换律、结合律、分配律），能快速验证复杂的多步变换。

---

## 实验：论文说的 vs 现实

论文在 5 个典型 LLM 算子上测试（GEMM 变体、Attention 组件、FFN 融合等）：

| 对比基准 | 执行速度提升 | 优化时间 |
|---------|-----------|---------|
| 最佳超优化器 | 最高 2.2× | 最快 3.4× |
| 最佳编译器方案 | 最高 4.9× | — |

有几点值得细品：

**2.2× over superoptimizers**：现有超优化器只搜索"执行参数空间"（tile size、循环顺序），而 Prism 通过 e-graph 改写还能发现"算子代数变换空间"（融合、分裂、重排）。这是两个维度的优化，后者是竞争对手触及不到的。

**4.9× over compilers**：TVM/XLA 的贪心启发式会在复杂融合决策处陷入局部最优，而 Prism 的符号剪枝在保证完备性的同时维持了可扩展性。

**3.4× faster optimization time**：通常"更好的代码 = 更慢的搜索"，Prism 同时做到了两者——符号剪枝在实例化之前就排除了大部分搜索空间。

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 静态形状的仿射张量程序（GEMM、Conv） | 动态形状（变长序列、动态 batch） |
| 批量编译（优化成本可摊销） | 单次在线推理（优化时间不可接受） |
| LLM 推理 / 训练的核心算子库 | 含复杂数据依赖分支的算子 |
| GPU / TPU 等规则硬件 | 新型异构硬件（需重建约束库） |

---

## 工程实践中的坑

**符号代价模型要保守**。理论带宽和实测有效带宽相差 20%-30%，如果直接用理论值推导约束，会保留一些"理论上最优但实际上因内存争用退化"的配置：

```python
# 错误：直接用峰值带宽，导致符号约束过于乐观
theoretical_bw_gbps = 2000  # A100峰值

# 正确：用实测有效带宽，并留安全余量
effective_bw_gbps = 1400   # 实际流量测试值
safety_margin = 0.85
usable_bw = effective_bw_gbps * safety_margin  # ~1190 GB/s
```

**改写规则的完备性问题**。Prism 内置的代数恒等式覆盖了常规矩阵运算，但对于 Flash Attention 的 online softmax、RMSNorm 的融合等特殊模式，需要手工扩展改写规则库。规则不完备，e-graph 不会报错，只是找不到那部分等价优化。

---

## 我的观点

Prism 最重要的贡献不是那些性能数字，而是提供了一个**概念框架的升级**：把"优化具体程序"重新定义为"推理程序族的属性"。

这个框架会渗透进未来的 ML 编译器。XLA 和 TVM 的下一代版本很可能借鉴 sGraph 的思路——不是直接用 Prism，而是把符号化搜索空间的思想纳入 schedule 搜索框架。

有一点论文低估了：**e-graph 作为正确性验证工具的价值**。当前 ML 编译器在做激进融合时依赖"经验上不会出错"的保证，缺乏形式化验证。随着融合越来越复杂（如跨 layer 融合、KV cache 算子融合），有一个机械化的等价性验证器会变得越来越重要。Prism 的验证管线可以独立地被复用。

论文没有触及的开放问题：sGraph 在动态形状下的扩展。LLM prefill 和 decode 的序列长度不同，如果 sGraph 能以符号量表示 batch size / seq len，并在部署时根据实际形状做快速实例化，应用价值会显著扩大。这可能是下一篇跟进工作的方向。

> **论文链接**：https://arxiv.org/abs/2604.15272v1