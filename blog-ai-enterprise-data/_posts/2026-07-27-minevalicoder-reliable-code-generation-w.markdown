---
layout: post-wide
title: "用二部图互验证让 LLM 代码生成更可靠：MineValiCoder 解析"
date: 2026-07-27 12:03:17 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.22471v1
generated_by: Claude Code CLI
---

## 一句话总结

MineValiCoder 通过**过滤劣质测试用例 + 二部图迭代评分**，解决 LLM 生成的测试用例本身不可靠这一根本问题，将 HumanEval Pass@1 推到 96%。

---

## 背景：测试驱动开发遇上 LLM 随机性

让 LLM 写代码的主流思路是 Test-Driven Development（TDD）：先生成测试用例，再让代码通过测试。这个 pipeline 有一个致命假设——**测试用例本身是正确的**。

当测试用例由人类编写时，这个假设基本成立。但当测试也由 LLM 自动生成时，问题就来了：

```
需求描述 → LLM 生成测试 → LLM 生成代码 → 代码跑通测试 ✓
                    ↑
              测试本身可能是错的
```

两个具体的 failure mode：

1. **Faulty test problem**：一个错误的测试断言 `assert add(2, 3) == 6`，LLM 为了通过它而写出错误代码，且毫不自知
2. **Mixed-quality problem**：20 个测试里有 5 个是错的，正确与错误信号混在一起，代码选择环节无从判断哪个候选更好

MineValiCoder 的核心 insight：**代码质量和测试质量可以互相验证**——好代码倾向于通过好测试，好测试倾向于被好代码通过。这是一个可以迭代求解的不动点。

---

## 三个核心模块

### 模块一：测试用例质量挖掘（TCQM）

核心思路：用**多个独立代码实现对同一测试投票**，如果它们的输出不一致，说明这个测试存在歧义或错误。

对测试用例 $t$，生成 $N$ 个独立代码实现 $c_1, \ldots, c_N$，计算一致性分数：

$$
\text{consistency}(t) = \frac{|\{(i,j) : i < j,\ \text{outcome}(c_i, t) = \text{outcome}(c_j, t)\}|}{\binom{N}{2}}
$$

一致性低于阈值 $\theta$ 的测试被过滤掉。直觉上：如果多个合理的实现对某个测试用例的判断相互矛盾，这个测试本身大概率有问题。

### 模块二：并行 TDD 精炼

用**验证后的测试集**作为反馈信号，并行优化多个代码候选：

```
validated_tests → 并行生成 K 个代码候选 → 各候选跑测试 → 失败用例作为 prompt → 迭代修复
```

关键点是"并行"：不是一个代码迭代精炼，而是维护一个**候选池**，最终从中选最优。

### 模块三：二部图互验证（BiCoTeV）

这是最有意思的部分，本质上是 HITS 算法在代码-测试关系上的应用。

构建一个二部图：
- 左节点：代码候选 $c_1, \ldots, c_M$
- 右节点：测试用例 $t_1, \ldots, t_K$
- 边：$A_{ij} = 1$ 当且仅当代码 $c_i$ 通过测试 $t_j$

迭代更新两组分数：

$$
s_c(i) = \sum_j A_{ij} \cdot s_t(j), \quad s_t(j) = \sum_i A_{ij} \cdot s_c(i)
$$

收敛后，$s_c(i)$ 最高的代码候选即为最终答案。直觉：通过了高质量测试的代码分高；被高质量代码通过的测试分高。

---

## 实现

### 核心实现（TCQM + BiCoTeV）

```python
import numpy as np
from typing import Callable

class TestCaseQualityMiner:
    def __init__(self, n_validators: int = 5, threshold: float = 0.7):
        self.n_validators = n_validators
        self.threshold = threshold

    def filter(
        self,
        test_cases: list[str],
        generate_code: Callable[[], str],
        execute: Callable[[str, str], bool],
    ) -> list[str]:
        # 生成 N 个独立代码实现用于投票
        validators = [generate_code() for _ in range(self.n_validators)]
        
        valid_tests = []
        for tc in test_cases:
            outcomes = [execute(code, tc) for code in validators]
            # 计算 pairwise 一致性
            pairs = self.n_validators * (self.n_validators - 1) / 2
            agree = sum(
                outcomes[i] == outcomes[j]
                for i in range(self.n_validators)
                for j in range(i + 1, self.n_validators)
            )
            if agree / pairs >= self.threshold:
                valid_tests.append(tc)
        
        return valid_tests


class BipartiteGraphValidator:
    def __init__(self, alpha: float = 0.85, max_iter: int = 50, tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def score(self, pass_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        pass_matrix: (n_codes, n_tests), 1 表示通过
        返回: (code_scores, test_scores)
        """
        n_codes, n_tests = pass_matrix.shape
        code_scores = np.ones(n_codes) / n_codes
        test_scores = np.ones(n_tests) / n_tests

        for _ in range(self.max_iter):
            new_test = pass_matrix.T @ code_scores
            new_code = pass_matrix @ test_scores

            # L1 归一化防止分数爆炸
            norm_t = new_test.sum() + 1e-8
            norm_c = new_code.sum() + 1e-8
            new_test = self.alpha * new_test / norm_t + (1 - self.alpha) / n_tests
            new_code = self.alpha * new_code / norm_c + (1 - self.alpha) / n_codes

            if np.abs(new_code - code_scores).max() < self.tol:
                break
            code_scores, test_scores = new_code, new_test

        return code_scores, test_scores

    def select_best(self, candidates: list[str], pass_matrix: np.ndarray) -> str:
        code_scores, _ = self.score(pass_matrix)
        return candidates[np.argmax(code_scores)]
```

### 完整 Pipeline

```python
def mineval_pipeline(
    requirement: str,
    llm_generate_tests: Callable[[str], list[str]],
    llm_generate_code: Callable[[str, list[str]], str],
    execute_code: Callable[[str, str], bool],
    n_candidates: int = 10,
    n_refinement_rounds: int = 3,
) -> str:
    # Step 1: 生成并过滤测试用例
    raw_tests = llm_generate_tests(requirement)
    miner = TestCaseQualityMiner(n_validators=5, threshold=0.7)
    valid_tests = miner.filter(
        raw_tests,
        generate_code=lambda: llm_generate_code(requirement, []),
        execute=execute_code,
    )

    # Step 2: 并行生成多个候选代码并迭代精炼
    candidates = []
    for _ in range(n_candidates):
        code = llm_generate_code(requirement, valid_tests)
        for _ in range(n_refinement_rounds):
            failed = [t for t in valid_tests if not execute_code(code, t)]
            if not failed:
                break
            code = llm_generate_code(requirement, failed)  # 以失败用例为反馈重生成
        candidates.append(code)

    # Step 3: BiCoTeV 选出最优候选
    pass_matrix = np.array([
        [int(execute_code(c, t)) for t in valid_tests]
        for c in candidates
    ])
    validator = BipartiteGraphValidator(alpha=0.85)
    return validator.select_best(candidates, pass_matrix)

# ... (LLM 接口封装和测试执行沙箱省略)
```

### 关键工程细节

**测试执行沙箱**：绝对不能直接 `exec()` LLM 生成的代码。必须用进程隔离：

```python
import subprocess, tempfile, os

def safe_execute(code: str, test_case: str, timeout: int = 5) -> bool:
    script = code + "\n" + test_case
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(script)
        fname = f.name
    try:
        result = subprocess.run(
            ["python", fname], timeout=timeout,
            capture_output=True, text=True
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)
```

**TCQM 阈值怎么选**：0.7 是论文默认值。如果你的任务需求很精确（如竞赛题），可以调高到 0.8；如果测试用例本来就少，调低到 0.6 以避免过度过滤。

---

## 实验结果

| 方法 | HumanEval | MBPP | APPS | LiveCodeBench |
|------|-----------|------|------|---------------|
| GPT-4 直接生成 | ~90% | ~82% | ~48% | ~38% |
| AlphaCode | 91.0% | - | 56.6% | - |
| **MineValiCoder** | **96.34%** | **87.40%** | **64.00%** | **51.33%** |

注意：这些数字依赖于底层 LLM 的能力，用更弱的 backbone 效果下降明显。

---

## 工程实践和常见坑

**坑 1：TCQM 过滤掉了太多测试**

```python
# 诊断：打印过滤比例
valid_ratio = len(valid_tests) / len(raw_tests)
if valid_ratio < 0.3:
    # 说明生成的测试质量本来就很差，先检查 test generation prompt
    print(f"Warning: only {valid_ratio:.0%} tests passed quality filter")
```

如果过滤比例过低，通常是 test generation prompt 有问题，而不是 threshold 需要调整。

**坑 2：BiCoTeV 所有候选分数相同**

这发生在 pass_matrix 接近全零或全一时：

```python
# 快速检查
print(f"Pass rate: {pass_matrix.mean():.2f}")
# 期望值：0.3 ~ 0.8，太极端说明测试太难或太简单
```

Pass rate 接近 0：测试太难，候选代码都不会；接近 1：测试太简单，区分不了好坏代码。两种情况下二部图评分都退化。

**坑 3：并行精炼轮次浪费**

```python
# 早停：如果所有测试都通过了，不需要继续精炼
failed_tests = [t for t in valid_tests if not execute_code(code, t)]
if not failed_tests:
    break  # 别忘了这个！不然会白白多调几次 LLM
```

---

## 适用边界

| 适合场景 | 不适合场景 |
|---------|-----------|
| 有自然语言描述的函数级编程题 | 需要复杂系统上下文的代码（如修 bug） |
| 输出可执行验证（有明确 assert） | 生成式任务（如写文章、写 SQL 注释） |
| 有足够算力跑 LLM N 次 | token 预算极度受限的场景 |
| 竞赛题、算法题 | UI/前端代码（难以自动化测试） |

---

## 我的观点

MineValiCoder 的核心贡献不是某个新的神经网络结构，而是**重新思考了 LLM 输出的可靠性问题**：当你不能信任模型的单次输出时，用多次采样 + 互验证来建立置信度。这个思路本质上和 Self-Consistency（CoT 多数投票）一脉相承，只不过把它扩展到了代码-测试的二部关系上。

HITS 算法在 1990 年代就被提出来做网页排名，现在被拿来做代码选择——这种"老算法解决新场景"的路子，在 LLM 工程里会越来越常见。

**值得一试的条件**：你有一个可以自动执行的测试框架，LLM API 调用成本不是瓶颈，且任务有清晰的正确性标准（能用测试验证）。

**不值得的情况**：如果你的瓶颈是底层 LLM 能力而非选择策略，换更强的模型比加复杂 pipeline 更有效。

论文链接：[arxiv.org/abs/2607.22471](https://arxiv.org/abs/2607.22471v1)