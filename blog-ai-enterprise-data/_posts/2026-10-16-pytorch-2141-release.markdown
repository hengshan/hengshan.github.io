---
layout: post-wide
title: 'PyTorch 2.14.1 "静默正确性"补丁:当 GPU 算出错误答案却不报错时'
date: 2026-10-16 08:03:38 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://github.com/pytorch/pytorch/releases/tag/v2.14.1
generated_by: Claude Code CLI
---

## 一句话总结

PyTorch 2.14.1 修复了 MPS(Apple Silicon GPU)后端两个**不报错但结果错误**的线性代数 bug——`torch.linalg.lstsq` 在复数欠定批量系统上出错、`torch.linalg.svd` 在秩亏/病态矩阵上给出非正交的 `U` 和不准确的小奇异值——这提醒所有做数值计算的工程师:GPU 后端的正确性不能只靠"没有 crash"来判断,必须主动验证。

## 为什么这个补丁值得花时间看?

大多数 release note 我们扫一眼标题就划过去了,但"silent correctness fix"(静默正确性修复)这个分类值得单独拿出来讲,原因很简单:

- **它不会报错**。类型不匹配、shape 不对、NaN、Inf——这些 bug 会让程序崩溃,工程师第一时间就能发现。但静默正确性 bug 只是把正确答案换成了一个"看起来正常"的错误答案,程序跑完、loss 下降、不报任何异常。
- **它会沿着计算图传播**。如果你的 pipeline 里有一步用 `lstsq` 做最小二乘拟合,或者用 `svd` 做 PCA / 低秩近似,这个错误会一直传到最终结果,而且没有任何 log 提示你"这里出问题了"。
- **MPS 后端相对年轻**。相比 CPU(背后是几十年打磨的 LAPACK)和 CUDA(NVIDIA 自己维护 cuSOLVER),MPS 后端的线性代数 kernel 是 PyTorch 团队相对较新、测试覆盖相对较薄的一块,复数运算、批量运算、退化矩阵(欠定、秩亏、病态)这些 edge case 更容易被漏掉。

核心洞见是:**不同后端在遇到数值 edge case 时,可能走了不同的算法分支**,而这些分支在"正常"矩阵上结果一致,只有在欠定、秩亏、病态这些特殊输入上才会暴露差异。这正是这次补丁修复的两个问题的共同模式。

## 核心概念解析

### 1. 欠定最小二乘问题(`lstsq`)

给定 $A \in \mathbb{C}^{m \times n}$($m < n$,方程数少于未知数)和 $b$,欠定系统有无穷多个解。`torch.linalg.lstsq` 在这种情况下应该返回**最小范数解**:

$$
x^* = \arg\min_{x \,:\, Ax=b} \|x\|_2
$$

标准做法是通过 SVD 求解:设 $A = U\Sigma V^H$,则

$$
x^* = V \Sigma^{+} U^H b
$$

其中 $\Sigma^+$ 是伪逆(非零奇异值取倒数)。如果底层 SVD 实现本身在复数、批量场景下有数值偏差,那么算出来的"最小范数解"就不是真正的最小范数解——而且程序不会报错,只会给你一个看起来合理但错误的 $x$。

### 2. SVD 的正交性约束

SVD 分解 $A = U\Sigma V^H$ 有一个硬性数学约束:$U$ 和 $V$ 必须是正交(酉)矩阵,也就是

$$
U^H U = I, \qquad V^H V = I
$$

这次修复提到的问题是:在**秩亏**(矩阵实际秩小于 $\min(m,n)$)或**病态**(条件数很大)的输入上,MPS 算出的 $U$ 不满足正交性,而且对应小奇异值的数值不准。

直觉上这个问题的根源是:当矩阵接近奇异时,小奇异值对应的奇异向量在数值上是"不稳定"的方向——微小的浮点误差会导致这些方向发生较大旋转。好的数值算法(比如 LAPACK 里经过多年调优的分治法或 QR 迭代)会用额外的技巧(如 deflation、重新正交化)来控制这种误差;如果 GPU 实现省略了这类步骤,结果就会在这些 edge case 上劣化,但在"正常"的满秩、良态矩阵上完全看不出问题——这也是为什么这种 bug 能潜伏很久才被发现。

## 动手实现:写一个"静默正确性检测器"

与其等官方告诉你哪里错了,更稳健的做法是自己对关键数值 kernel 写验证逻辑。下面是一个可以直接运行的诊断脚本,核心思路是:**SVD 和 lstsq 的输出都有数学上可验证的不变量,我们直接检查这些不变量,而不是信任"没报错就是对的"**。

```python
import torch

def check_svd_health(A: torch.Tensor, rcond: float = 1e-10):
    """
    检查 torch.linalg.svd 的结果是否满足数学不变量:
    1. 正交性: U^H U ≈ I, V^H V ≈ I
    2. 重构误差: U @ diag(S) @ V^H ≈ A
    返回一个 dict,包含两个误差指标,数值越小越可信。
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    eye_u = torch.eye(U.shape[-1], dtype=U.dtype, device=U.device)
    eye_v = torch.eye(Vh.shape[-2], dtype=Vh.dtype, device=Vh.device)

    # 正交性缺陷: Frobenius 范数衡量偏离单位矩阵的程度
    orth_defect_U = torch.linalg.norm(U.mH @ U - eye_u)
    orth_defect_V = torch.linalg.norm(Vh @ Vh.mH - eye_v)

    # 重构误差: 用分解结果重建原矩阵,看差多少
    recon = U @ torch.diag(S).to(U.dtype) @ Vh
    recon_err = torch.linalg.norm(recon - A) / torch.linalg.norm(A)

    return {
        "orth_defect_U": orth_defect_U.item(),
        "orth_defect_V": orth_defect_V.item(),
        "recon_rel_err": recon_err.item(),
        "min_singular_value": S.min().item(),
        "condition_number": (S.max() / S.min().clamp_min(1e-30)).item(),
    }


def check_lstsq_health(A: torch.Tensor, b: torch.Tensor):
    """
    检查 lstsq 的解是否满足最小二乘的最优性条件:
    残差应与 A 的列空间正交,即 A^H (Ax - b) ≈ 0。
    """
    result = torch.linalg.lstsq(A, b)
    x = result.solution
    residual = A @ x - b
    # 最优性条件: A^H r ≈ 0
    optimality_gap = torch.linalg.norm(A.mH @ residual)
    return {"residual_norm": torch.linalg.norm(residual).item(),
            "optimality_gap": optimality_gap.item()}
```

这两个函数的关键在于**不依赖 PyTorch 内部实现是否正确**,而是用分解结果重新验证数学定义——这是任何数值库都应该能通过的检查,如果通不过,说明后端实现有问题,跟是不是 edge case 无关。

### 构造能暴露问题的病态/秩亏矩阵

```python
import torch

def make_rank_deficient_complex(m, n, rank, device="cpu"):
    """构造一个复数、秩亏的矩阵,用于触发 edge case"""
    assert rank <= min(m, n)
    U = torch.randn(m, rank, dtype=torch.complex64, device=device)
    V = torch.randn(rank, n, dtype=torch.complex64, device=device)
    return U @ V  # 秩最多为 rank

def make_ill_conditioned(n, cond=1e8, device="cpu"):
    """构造指定条件数的病态矩阵,用奇异值谱直接控制"""
    U, _ = torch.linalg.qr(torch.randn(n, n, device=device))
    V, _ = torch.linalg.qr(torch.randn(n, n, device=device))
    s = torch.logspace(0, -torch.log10(torch.tensor(cond)).item(), n)
    return U @ torch.diag(s) @ V.T

# 跑一遍诊断
A_rank_def = make_rank_deficient_complex(50, 30, rank=10)
report = check_svd_health(A_rank_def)
print(report)

A_ill = make_ill_conditioned(40, cond=1e9)
b = torch.randn(40, dtype=torch.float32)
report2 = check_lstsq_health(A_ill, b)
print(report2)
```

在 CPU(LAPACK)后端上跑这段代码,`orth_defect_U`、`orth_defect_V`、`optimality_gap` 应该都在 `1e-5` 量级以下(float32 精度)。**如果你在 MPS 设备上跑同样的代码,数值明显偏大(比如 `1e-2` 以上),就说明你撞上了类似这次修复的问题。**

> 诚实说明:我没有 Apple Silicon 设备,无法实际对比 PyTorch 2.14.0 与 2.14.1 在 MPS 上跑这段脚本的差异。上面的诊断逻辑是通用的数值验证方法,建议有 Mac 的读者分别在两个版本上跑一遍,对比 `orth_defect` 和 `optimality_gap` 的量级变化,这比单纯相信 release note 的文字描述更可靠。

## 实验:Release Note 说的 vs 我们能验证的

官方 release note 给出的信息是:

- `torch.linalg.lstsq` 在 MPS 上对**复数、批量、欠定**系统给出错误解([#196113](https://github.com/pytorch/pytorch/issues/196113),修复于 [#196128](https://github.com/pytorch/pytorch/pull/196128))
- `torch.linalg.svd` 在 MPS 上对**秩亏、病态**矩阵给出非正交的 `U` 和不准确的小奇异值([#196112](https://github.com/pytorch/pytorch/issues/196112))

这两个描述都没有给出具体的误差数量级(比如"正交性缺陷从 0.3 降到 1e-6"),这也是 release note 的一个共性问题——"修复了"告诉你方向,但没告诉你修复前后差多少、你的使用场景是否真的会撞上。**这正是上面那个诊断脚本存在的价值**:比起等官方公告,自己对关键 kernel 跑不变量检查,可以更早、更精确地发现问题,而且不依赖是否读到了这篇 release note。

## 什么时候需要警惕这类问题?

| 适用场景(风险较低) | 需要警惕/额外验证的场景 |
|---|---|
| 矩阵满秩、条件数适中(< 1e6) | 矩阵秩亏或条件数很大(> 1e8) |
| 实数、非批量的简单线性代数运算 | 复数运算,尤其是复数 + 批量组合 |
| CPU 或 CUDA 后端,生产环境已大规模验证 | MPS 后端,尤其是刚升级 PyTorch 版本时 |
| 下游只用于非关键展示(可视化、探索性分析) | 下游结果影响决策(科学计算、金融建模、医疗) |
| 有完整的数值回归测试套件 | 没有对 `svd`/`lstsq` 等结果做不变量校验 |

## 我的观点

这类 bug 暴露的不是某个具体 kernel 写错了,而是一个更普遍的事实:**线性代数库的"正确性"不是二元的 pass/fail,而是在不同输入分布下连续变化的数值质量**。GPU 后端(尤其是相对新的 MPS)在覆盖所有 edge case 上天然落后于几十年积累的 LAPACK,这不是 PyTorch 团队的失误,而是任何新后端都要经历的过程。

对工程师的实际建议:

- 如果你的 pipeline 里有 `svd`、`lstsq`、`eig`、`cholesky` 这类分解,**不要只测 shape 和 dtype,加上正交性/残差这类数值不变量检查**作为单测的一部分,这比等下游模型效果变差再排查便宜得多。
- 关键数值计算(尤其涉及秩亏、病态矩阵)优先用 CPU(double precision)做一次 sanity check,再决定是否信任 MPS/CUDA 的结果。
- 升级 PyTorch 版本后,如果你的代码涉及这几个 API,值得花十分钟跑一遍本文的诊断脚本,而不是假设"patch release 不会影响我"。

## 局限性说明

本文的诊断方法只能发现"分解结果违反数学不变量"这类问题,无法定位具体是哪一行 kernel 代码出错,也无法保证覆盖所有可能的静默正确性 bug(比如数值上满足不变量但仍偏离真实解的情况,虽然这种情况在实践中较少见)。如果你需要更严格的保证,应该结合已知解析解的合成数据做端到端对比,而不是只依赖不变量检查。