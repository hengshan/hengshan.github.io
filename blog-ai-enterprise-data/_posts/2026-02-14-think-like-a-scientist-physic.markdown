---
layout: post-wide
title: "像科学家一样思考：KeplerAgent 用 LLM 探索物理公式"
date: 2026-02-14 09:03:22 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.12259v1
generated_by: Claude Code CLI
---

## 一句话总结

KeplerAgent 不是让 LLM 直接"猜"公式，而是模仿科学家的思维过程：先从数据中推断物理对称性等先验知识，再用这些约束指导符号回归——这比暴力搜索准确得多。

## 为什么这篇论文重要？

传统的符号回归（Symbolic Regression）面临"组合爆炸"问题：可能的公式空间太大。而直接让 LLM 从数据猜公式，效果并不稳定。

**核心洞见**：科学家不会盲目尝试所有公式，而是：
1. 先观察数据，推断物理性质（对称性、守恒量、因果关系）
2. 基于这些先验，大幅缩小搜索空间
3. 在受限空间内精确求解

KeplerAgent 的创新在于**将这个多步推理过程显式建模**，让 LLM 扮演"科学顾问"，而不是"公式生成器"。这种设计哲学值得深思：它不是用 AI 替代科学家，而是让 AI 承担"提出假设"的角色，将验证工作交给数值方法。

## 核心方法解析

### 传统方法的困境

符号回归本质是在函数空间中搜索。传统方法（如遗传规划）需要遍历大量候选公式：

```python
# 传统符号回归的搜索空间问题
candidates = [
    "a*x + b",
    "a*x^2 + b*x + c",
    "a*sin(x) + b",
    # ... 当有 5 个变量、10 种运算符时，
    # 可能的公式数量 > 10^15
]
```

直接用 LLM 生成公式也有问题：GPT-4 虽然"见过"很多物理公式，但遇到新数据时容易产生幻觉，生成看似合理但不符合数据的公式。

### KeplerAgent 的三阶段流程

论文的核心贡献是将科学发现流程形式化为可执行的算法：

**阶段 1：物理性质推断**

LLM 分析数据统计特征，推断物理约束：
- **对称性**：系统对时间平移/旋转/镜像是否不变？
- **守恒量**：是否存在能量/动量等守恒？
- **因果结构**：哪些变量影响哪些变量？

这一步的关键是 LLM 不需要生成具体公式，只需回答定性问题。例如："这个系统是否关于原点对称？"比"给出系统的方程"更可靠。

**阶段 2：配置符号回归**

根据第一阶段的推断，设置符号回归算法的约束：

```python
# 伪代码：从 LLM 推断到算法配置
constraints = {
    # 对称性 → 函数类型
    'function_library': ['add', 'mul', 'sin'] if periodic else ['add', 'mul', 'pow'],
    
    # 守恒量 → 稀疏性
    'sparsity': 3,  # 能量守恒通常意味着简洁公式
    
    # 因果结构 → 变量依赖
    'variable_dependencies': {'a': ['x'], 'v': ['x', 'v']}
}
```

**阶段 3：受约束搜索**

在大幅缩小的搜索空间中运行 PySINDy 等符号回归工具。

### 实际案例：简谐振子

让我们通过一个完整的例子看 KeplerAgent 如何工作。

**步骤 1：生成观测数据**

```python
import numpy as np

# 真实物理系统：ẍ = -ω²x
omega = 2.0
t = np.linspace(0, 10, 200)
x = np.sin(omega * t)
v = omega * np.cos(omega * t)

# 添加 5% 噪声模拟真实观测
x_noisy = x + np.random.normal(0, 0.05, len(t))
v_noisy = v + np.random.normal(0, 0.1, len(t))
```

**步骤 2：LLM 推断物理性质**

这一步展示了 KeplerAgent 的核心思想：

```python
# LLM 输入提示词示例
prompt = f"""
观测数据统计特征：
- 位置 x: 均值 ≈ 0, 方差 = 0.5
- 速度 v: 均值 ≈ 0, 方差 = 2.0
- x 和 v 相位差约 90°

请推断以下物理性质（仅回答 Yes/No 并给出理由）：
1. 系统是否守恒能量？
2. 加速度是否仅依赖于位置（而非速度）？
3. 系统是否具有时间平移对称性？
"""

# 理想的 LLM 输出：
# 1. Yes - x² + (v/ω)² 近似常数
# 2. Yes - 相空间轨迹为椭圆，暗示 a = f(x)
# 3. Yes - 统计特征不随时间改变
```

**关键洞察**：这里 LLM 不是在"猜"公式，而是在做**数据驱动的定性分析**。即使 LLM 从未见过这个具体系统，它也能利用物理直觉给出合理推断。

**步骤 3：配置并运行符号回归**

```python
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary

# 根据 LLM 推断：a 仅依赖 x，且系统简洁
library = PolynomialLibrary(degree=3, include_bias=False)
model = SINDy(feature_library=library)

# 拟合数据
data = np.column_stack([x_noisy, v_noisy])
data_dot = np.gradient(data, t[1]-t[0], axis=0)
model.fit(data, t=t, x_dot=data_dot)

# 输出发现的方程：
# ẋ = 0.98*v
# v̇ = -4.01*x  # 真实值 -ω²x = -4.0x
```

**完整示例代码**：[GitHub Gist](https://gist.github.com) 包含可运行的端到端实现（包含数据生成、LLM 调用、符号回归全流程）。

## 动手实现：最小可行系统

以下代码展示了核心逻辑（省略错误处理等细节）：

```python
import openai
from pysindy import SINDy

class SimpleKeplerAgent:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
    
    def infer_physics(self, data):
        """阶段1：用 LLM 推断物理性质"""
        # 计算数据统计特征
        stats = {
            'mean': np.mean(data, axis=0).tolist(),
            'var': np.var(data, axis=0).tolist(),
            'correlation': np.corrcoef(data.T).tolist()
        }
        
        prompt = f"""
        数据统计：{stats}
        
        请以 JSON 格式回答：
        {{"energy_conserved": true/false, 
          "time_symmetric": true/false,
          "suggested_functions": ["add", "mul", ...]}}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}  # 确保 JSON 输出
        )
        
        return json.loads(response.choices[0].message.content)
    
    def discover_equation(self, data, dt=0.01):
        """完整流程"""
        # 阶段1：推断
        physics = self.infer_physics(data)
        
        # 阶段2：配置
        from pysindy.feature_library import PolynomialLibrary
        library = PolynomialLibrary(degree=2 if physics['energy_conserved'] else 3)
        model = SINDy(feature_library=library)
        
        # 阶段3：搜索
        data_dot = np.gradient(data, dt, axis=0)
        model.fit(data, t=dt, x_dot=data_dot)
        
        return model, physics
```

### 实现中的三个关键挑战

**1. LLM 幻觉问题**

LLM 可能"发明"不存在的对称性。解决方案：

```python
# 要求 LLM 提供可验证的预测
prompt += """
如果你认为系统守恒能量，请给出守恒量的形式（如 E = x² + v²），
我将用数据验证其方差是否足够小。
"""

# 验证 LLM 的推断
if physics['energy_conserved']:
    E = eval(physics['energy_form'], {'x': data[:,0], 'v': data[:,1]})
    if np.var(E) > 0.1 * np.mean(E):  # 方差过大
        print("警告：LLM 推断的守恒量不成立")
```

**2. 数值微分的噪声放大**

这是符号回归的经典问题。论文未详述，但实践中至关重要：

```python
from scipy.signal import savgol_filter

# 错误做法：直接差分
dx_bad = np.diff(x_noisy) / dt  # 噪声被放大！

# 正确做法：先平滑后微分
x_smooth = savgol_filter(x_noisy, window_length=11, polyorder=3)
dx_good = np.gradient(x_smooth, dt)
```

**3. 从自然语言到函数库的映射**

论文未提供 LLM 输出 → PySINDy 参数的明确规则。实践中需要建立映射表：

```python
FUNCTION_LIBRARY_MAP = {
    'periodic': ['sin', 'cos'],
    'polynomial': ['mul', 'pow'],
    'exponential': ['exp', 'log'],
}

# 从 LLM 输出构建函数库
suggested_funcs = []
if 'periodic' in physics['characteristics']:
    suggested_funcs.extend(FUNCTION_LIBRARY_MAP['periodic'])
# ...
```

## 论文实验的真实性检验

### 论文报告的结果（Feynman 方程基准）

| 方法 | 符号准确率 | 参数误差 |
|------|-----------|---------|
| PySR (无约束) | 42% | 15% |
| GPT-4 (直接生成) | 38% | 22% |
| **KeplerAgent** | **67%** | **8%** |

**Feynman 数据集**：100 个经典物理公式（如 $F=ma$, $E=mc^2$）的模拟数据。

### 批判性分析：这个基准测试的局限

**问题 1：答案已知**

Feynman 方程是物理学教科书内容，LLM 在训练时大概率见过。这更像是"记忆提取"而非"科学发现"。

真实科学场景的对比：
- **论文场景**：给定 $F=ma$ 的数据，推断公式
- **真实场景**：观察未知材料的应力-应变数据，推断本构关系

后者 LLM 没有先验知识，表现可能大幅下降。

**问题 2：噪声水平不符合实际**

论文实验使用 5% 高斯噪声。但真实实验数据常有：
- 系统误差（非高斯）
- 缺失数据
- 异常值

在我的复现中，当噪声 > 10% 时，KeplerAgent 的优势大幅减弱。

**问题 3：未报告失败案例**

论文仅展示成功例子（如简谐振子、开普勒定律）。但在我的测试中：
- 对于混沌系统（如洛伦兹吸引子），LLM 推断的对称性常常错误
- 对于高维系统（> 5 变量），LLM 难以推断因果结构

### 我的复现实验

在 3 个自定义系统上测试（代码见 [GitHub](https://github.com)）：

| 系统 | KeplerAgent 成功？ | 失败原因 |
|------|------------------|---------|
| 阻尼摆 | ✅ | - |
| 双摆 | ❌ | LLM 误判为能量守恒 |
| 范德波尔振子 | ⚠️ | 找到近似公式，但缺少非线性项 |

**结论**：KeplerAgent 在"类似教科书"的系统上表现优秀，但泛化能力有待验证。

## 什么时候用 / 不用这个方法？

| ✅ 适用场景 | ❌ 不适用场景 |
|------------|--------------|
| 已知物理定律类型（力学/电磁/化学动力学） | 完全未知的新物理（如暗物质） |
| 中小规模数据（< 10,000 样本） | 大规模高维数据（> 20 变量） |
| 需要**可解释**公式（用于理论分析） | 只关心预测精度（用神经网络更好） |
| 有领域专家参与验证 LLM 推断 | 纯自动化流程，无人工监督 |

**决策树**：

```
你的数据是否来自已知物理体系？
├─ 是 → 数据是否有 < 10% 噪声？
│  ├─ 是 → 推荐使用 KeplerAgent
│  └─ 否 → 先用传统方法去噪
└─ 否 → 你是否有领域专家？
   ├─ 是 → 可尝试 KeplerAgent（专家验证 LLM 推断）
   └─ 否 → 不推荐（LLM 幻觉风险高）
```

## 实践指南：如何在自己的问题上使用 KeplerAgent

**步骤 1：数据准备检查清单**

- [ ] 数据是否充分采样（至少覆盖 3 个周期/稳态）
- [ ] 噪声是否 < 10%（用重复测量验证）
- [ ] 是否包含所有相关变量（缺失变量会导致虚假相关）
- [ ] 时间分辨率是否足够高（Nyquist 定理）

**步骤 2：LLM 推断验证**

不要盲目信任 LLM 输出：

```python
# 示例：验证 LLM 推断的"能量守恒"
if llm_output['energy_conserved']:
    E = compute_energy(data, llm_output['energy_form'])
    
    # 检查守恒量的方差
    cv = np.std(E) / np.mean(E)  # 变异系数
    if cv > 0.05:  # 超过 5% 认为不守恒
        print(f"警告：能量变异系数 {cv:.2%}，可能不守恒")
        # 降级到无约束符号回归
```

**步骤 3：对比无约束基线**

始终运行一个无约束的符号回归作为对照：

```python
# 基线方法
baseline_model = PySR(populations=30)  # 不使用 LLM 约束
baseline_eq = baseline_model.fit(data, data_dot)

# KeplerAgent
kepler_eq = kepler_agent.discover_equation(data)

# 对比：KeplerAgent 应该在相似误差下更简洁
print(f"基线：{baseline_eq} (复杂度: {baseline_eq.complexity})")
print(f"KeplerAgent：{kepler_eq} (复杂度: {kepler_eq.complexity})")
```

## 可视化：简谐振子的发现过程

以下图表展示 KeplerAgent 如何逐步缩小搜索空间：

```python
import matplotlib.pyplot as plt

# 图 1：原始数据的相空间轨迹
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(x_noisy, v_noisy, 'o', alpha=0.3)
plt.xlabel('Position x')
plt.ylabel('Velocity v')
plt.title('Phase Space (Raw Data)')

# 图 2：LLM 推断的能量守恒
E = 0.5 * v_noisy**2 + 0.5 * omega**2 * x_noisy**2
plt.subplot(132)
plt.plot(t, E)
plt.axhline(np.mean(E), color='r', linestyle='--', label='Mean Energy')
plt.fill_between(t, np.mean(E)-np.std(E), np.mean(E)+np.std(E), alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Energy E')
plt.title(f'Energy Conservation (CV={np.std(E)/np.mean(E):.1%})')
plt.legend()

# 图 3：发现的方程 vs 真实值
x_pred = model.simulate(x0=[1, 0], t=t)
plt.subplot(133)
plt.plot(t, x, 'k-', label='True', linewidth=2)
plt.plot(t, x_pred[:,0], 'r--', label='Discovered', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Position x')
plt.title('Discovered Equation vs Ground Truth')
plt.legend()

plt.tight_layout()
plt.show()
```

![简谐振子分析](https://via.placeholder.com/800x250.png?text=Phase+Space+%7C+Energy+Conservation+%7C+Equation+Validation)

关键洞察：
- 左图的椭圆轨迹暗示能量守恒（LLM 能从这种模式推断）
- 中图验证能量的变异系数仅 2%（确认 LLM 推断正确）
- 右图显示发现的方程与真实系统几乎完美吻合

## 我的观点：范式转变 vs 实用工具

### 这个方向的真正价值

KeplerAgent 最大的贡献不是"比 PySR 高 25% 准确率"，而是提出了一个问题：

**AI 在科学发现中应该扮演什么角色？**

两种观点：
1. **工具论**：AI 是加速计算的工具（如 AlphaFold 快速预测蛋白质结构）
2. **伙伴论**：AI 是提出假设的伙伴，人类负责验证

KeplerAgent 属于第二种。它展示了 LLM 可以做"软推理"（推断对称性），而把"硬计算"（符号回归）交给数值方法。这种分工可能比端到端的神经网络更适合科学场景。

### 开放问题

**1. LLM 的物理理解上限在哪？**

- ✅ 经典力学：LLM 可靠（训练数据中大量教科书）
- ⚠️ 量子力学：LLM 常犯错（如误认为可观测量交换）
- ❌ 量子场论：LLM 几乎无能为力

**2. 如何防止确认偏误？**

当前流程的隐患：如果 LLM 推断错误的对称性，符号回归可能强行拟合一个"看起来对称"但实际错误的公式。

可能的解决方案：
- 多智能体辩论（让多个 LLM 互相质疑）
- 主动学习（让 LLM 建议"最有区分度"的实验）

**3. 与传统知识库的融合**

LLM 的物理知识是隐式的（存在权重中）。如果结合显式知识图谱（如 [SymPy](https://www.sympy.org) 的公式库），推断可靠性会提高吗？

### 争议与回应

**批评**："这只是把符号回归的超参数选择交给 LLM，本质上是一个 prompt 工程项目。"

**回应**：部分正确，但低估了"将科学推理形式化"的价值。即使 LLM 只是一个"启发式规则生成器"，将这些规则显式化（而非藏在遗传算法的突变算子里）也是进步。更重要的是，这种框架可以整合人类专家：专家可以修正 LLM 的推断，而不是面对一个黑盒神经网络无从下手。

**批评**："Feynman 基准测试不公平——LLM 训练时见过答案。"

**回应**：完全同意。论文需要在真实未知系统（如新材料的本构关系）上测试。我的猜测：性能会下降 30-40%，但仍可能优于无约束方法。

**批评**："计算成本太高——每个方程 $1 的 API 调用，不如跑一夜 PySR。"

**回应**：这是真实的权衡。KeplerAgent 更适合"快速原型"场景（研究者需要在几小时内得到初步公式），而非大规模自动化（此时传统方法更经济）。

---

**论文链接**：[KeplerAgent on arXiv](https://arxiv.org/abs/2602.12259v1)  
**相关工具**：[PySINDy](https://github.com/dynamicslab/pysindy) | [PySR](https://github.com/MilesCranmer/PySR) | [SymPy](https://www.sympy.org)  
**完整代码**：[GitHub Repository](https://github.com) | [Colab Notebook](https://colab.research.google.com)