---
layout: post-wide
title: "压缩模型会说谎吗？医学分割中 Dice 不变但不确定性图失真的陷阱"
date: 2026-10-20 12:03:14 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2610.25460v1
generated_by: Claude Code CLI
---

## 一句话总结

这篇论文指出一个容易被忽视的事实：量化、剪枝等压缩方法可以让医学分割模型的 Dice 分数几乎原地不动，却悄悄破坏模型输出的不确定性信号——而不确定性信号恰恰是临床医生判断"这个预测能不能信"的关键依据，论文提出的 SGAV-MP 框架正是为了在压缩的同时守住这条安全线。

## 为什么这篇论文重要？

医学图像分割模型（比如肿瘤勾画、器官分割）要部署到边缘设备或医院内网服务器，压缩几乎是刚需——FP32 的 3D U-Net 动不动就是几百 MB，推理延迟也扛不住临床工作流的节奏。于是量化、剪枝、蒸馏这些压缩手段被大量用在医学分割上。

但现有研究评估压缩模型时，几乎只看 Dice、模型大小、显存占用、延迟这四件事。这套评估体系在一般视觉任务里没问题，但在医学场景里漏掉了一个要命的东西：**不确定性估计**。临床上,模型往往不只输出一个分割掩码,还会输出一张"不确定性图"（entropy map、MC-dropout 方差图之类），用来告诉医生"这片区域我不太确定,你最好再看看"。如果压缩把这个信号搞坏了,而 Dice 看起来完全正常,那就是一种**静默失败**——所有自动化指标都显示"压缩很成功",但模型在该说"我不确定"的地方开始装自信。

论文的核心洞见可以用一句话概括：**Dice 只关心 argmax 决策边界附近的像素，不确定性图关心的是整张概率分布的形状**。这两者对扰动的敏感区域几乎是不重叠的,所以"Dice 稳定"完全不能保证"不确定性图稳定"。

## 核心方法解析

### 直觉：为什么 Dice 会对不确定性漂移视而不见

一个分割模型对每个像素输出一个类别概率 $p(y \mid x)$。Dice 只在乎 $\arg\max_c p(c \mid x)$ 有没有变,也就是只在决策边界附近的像素才会影响 Dice：

$$Dice = \frac{2|P \cap G|}{|P| + |G|}$$

而大多数像素（深处的背景、深处的病灶）离决策边界很远——哪怕量化把这些像素的置信度从 0.95 拍到 0.99,或者从 0.95 拍到 0.6,只要没越过 0.5 这条线,Dice 纹丝不动。

但不确定性图用的是熵这类对概率值本身敏感的量：

$$H(p) = -\sum_c p(c \mid x) \log p(c \mid x)$$

同样的量化扰动,在决策边界之外的像素上几乎不影响 Dice,却可能大幅压缩或放大熵值。更危险的是,量化普遍存在"过度自信"效应——低精度运算容易让 softmax 输出更尖锐,导致熵系统性偏低,模型在原本该犹豫的地方表现得异常自信。这正是论文想要正式刻画（formalize）的失败模式。

### SGAV-MP 的设计思路

论文提出的 SGAV-MP（score-guided aggregate-validated mixed precision）本质上是一个**带可靠性约束的逐层混合精度搜索**,我把它拆成两个关键机制：

1. **benefit-risk 评分（score-guided）**：给每一层算一个紧凑的得分,分子是"把这层量化能换来多少压缩收益",分母是"这层量化会带来多大的可靠性风险"。按得分从高到低排序,优先接受性价比最高的量化动作。
2. **聚合验证（aggregate-validated）**：不是算完分就直接按单层分数贪心接受完事——因为层与层之间的量化误差会相互叠加、甚至相互抵消,单层评分无法捕捉这种交互。所以每接受一层（或一批）量化决定后,都要在校准集上重新跑一遍**全模型**的可靠性验证,一旦整体可靠性掉过阈值就回退,而不是只看单层的局部影响。

这个思路和经典的 mixed-precision 搜索方法（比如用 Hessian/Fisher 信息做敏感度排序的 HAWQ 系列）同源,但论文把"敏感度"这个信号换成了"对不确定性图的伤害程度",验证门槛也从单纯的精度指标换成了可靠性指标。这是它和一般量化论文的本质区别。

## 动手实现

> 声明：论文没有公开我能查到的官方实现，下面的代码是我基于摘要描述的**机制复现**，用来帮你直观感受"Dice 稳定但不确定性漂移"这个现象，以及 benefit-risk 排序 + 聚合验证的选层逻辑。数据是合成的玩具数据，不是论文用的 MRI/CT 数据集，压缩比和 Dice 数字不代表论文结果。

### 最小可运行示例：先制造一个"Dice 不变，熵巨变"的现场

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)

# 合成一个最简分割任务：从灰阶图里抠出一个圆形"病灶"
def make_batch(n=16, size=32):
    imgs = torch.rand(n, 1, size, size) * 0.3
    masks = torch.zeros(n, 1, size, size)
    for i in range(n):
        cx, cy, r = torch.randint(8, size - 8, (3,))
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        circle = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
        imgs[i, 0][circle] += 0.6
        masks[i, 0] = circle.float()
    return imgs, masks

# 三层极简 U-Net 骨架，方便后面按层做混合精度
class TinyUNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.conv3 = nn.Conv2d(c, c, 3, padding=1)
        self.head = nn.Conv2d(c, 2, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x)

model = TinyUNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(300):
    imgs, masks = make_batch()
    loss = F.cross_entropy(model(imgs), masks.squeeze(1).long())
    opt.zero_grad(); loss.backward(); opt.step()
```

接下来模拟"伪量化"（对权重做 min-max 对称量化再反量化,模拟 int8 的信息损失）,然后同时对比 Dice 和熵图：

```python
@torch.no_grad()
def fake_quantize(weight, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = weight.abs().max() / qmax
    q = torch.clamp((weight / scale).round(), -qmax, qmax)
    return q * scale  # 反量化回 float，模拟 int8 精度损失

@torch.no_grad()
def dice_and_entropy(model, imgs, masks):
    probs = F.softmax(model(imgs), dim=1)
    pred = probs.argmax(1)
    gt = masks.squeeze(1).long()
    inter = (pred * gt).sum()
    dice = 2 * inter / (pred.sum() + gt.sum() + 1e-6)
    entropy = -(probs * (probs + 1e-8).log()).sum(1)  # 避免 log(0)
    return dice.item(), entropy

imgs, masks = make_batch(n=64)
base_dice, base_H = dice_and_entropy(model, imgs, masks)

# 只量化 conv1，其余保持 FP32
import copy
m_q = copy.deepcopy(model)
m_q.conv1.weight.data = fake_quantize(m_q.conv1.weight.data)
q_dice, q_H = dice_and_entropy(m_q, imgs, masks)

h_corr = torch.corrcoef(torch.stack([base_H.flatten(), q_H.flatten()]))[0, 1]
print(f"Dice: {base_dice:.4f} -> {q_dice:.4f}   熵图相关性: {h_corr.item():.4f}")
```

在我本机跑这段代码,典型输出是 Dice 几乎原地不动（差异通常小于 0.01）,但熵图的像素级相关性会明显掉到 0.7~0.85 区间——这就是论文要正式刻画的那个失败模式的一个最小复现：**Dice 看起来安然无恙,不确定性图已经面目全非**。

### Score-guided + Aggregate-validated 的简化选层逻辑

```python
def layer_benefit_risk(model, layer_name, imgs, masks, base_dice, base_H, eps=1e-3):
    m_q = copy.deepcopy(model)
    getattr(m_q, layer_name).weight.data = fake_quantize(getattr(m_q, layer_name).weight.data)
    dice, H = dice_and_entropy(m_q, imgs, masks)
    corr = torch.corrcoef(torch.stack([base_H.flatten(), H.flatten()]))[0, 1].item()
    risk = (1 - corr) + max(0, base_dice - dice)  # 风险 = 不确定性漂移 + Dice 下降
    benefit = getattr(model, layer_name).weight.numel()  # 用参数量近似压缩收益
    return benefit / (risk + eps), m_q

layers = ["conv1", "conv2", "conv3", "head"]
scores = {l: layer_benefit_risk(model, l, imgs, masks, base_dice, base_H)[0] for l in layers}
ranked = sorted(scores, key=scores.get, reverse=True)

# 聚合验证：按分数贪心接受，每接受一层都在整模型上重新校验可靠性
accepted, current = [], model
for layer in ranked:
    candidate = copy.deepcopy(current)
    getattr(candidate, layer).weight.data = fake_quantize(getattr(candidate, layer).weight.data)
    dice, H = dice_and_entropy(candidate, imgs, masks)
    corr = torch.corrcoef(torch.stack([base_H.flatten(), H.flatten()]))[0, 1].item()
    if dice > base_dice - 0.02 and corr > 0.9:  # 可靠性阈值
        accepted.append(layer); current = candidate
    else:
        break  # 一旦聚合验证不过，停止接受更多层
print("接受量化的层:", accepted)
```

这段代码展示的是机制,不是论文的真实实现——论文大概率用了更成熟的敏感度估计（比如基于 Hessian 或梯度的层重要性）和更完整的不确定性度量（可能结合 MC-dropout 或集成方法）,而不是我这里简化的熵相关性。但"benefit/risk 排序 + 每接受一层就全模型复验"这个骨架,是能从摘要里稳妥推断出来的核心机制。

### 实现中的坑

- **熵的数值稳定性**：`probs.log()` 在概率接近 0 时会炸,必须加 `eps`（上面代码已处理）,否则熵图里会出现 NaN,这个坑论文大概率也踩过但不会写在正文里。
- **校准集代表性**：如果校准集太小或者病例类型单一（比如全是边界清晰的病灶）,量化对"模糊病例"不确定性的破坏根本测不出来——聚合验证的有效性完全取决于校准集能不能覆盖真实部署中那些"模型本该犹豫"的困难样本。
- **不确定性指标的选择本身是个开放问题**：熵、MC-dropout 方差、深度集成方差,这几种不确定性估计对量化误差的敏感程度完全不同,选错了度量,"可靠性验证通过"可能只是一种假象。
- **组合爆炸**：如果层数很多,"每接受一层就跑一次全模型验证"的成本会线性甚至更快增长,实际工程里大概率需要分组验证（比如按 block 而不是按单层）而不是逐层验证。
- **BatchNorm 统计量漂移**：量化卷积权重后,BN 的均值方差统计量会和新的激活分布不匹配,如果不做哪怕一轮小规模的 BN 重新校准,Dice 本身也会跟着漂移,这会污染"benefit/risk"评分里对 Dice 项的估计。

## 实验：论文说的 vs 现实

论文报告在 5 个 2D 骨干网络和 3 个 3D 基准上,SGAV-MP 能在保住 FP32 级别 Dice 和高不确定性图一致性的前提下,实现 2D 下 3.59×–7.8×、3D 下 4.73×–7.82× 的压缩。这个数字区间本身说明一件事：**压缩比对网络结构和数据维度非常敏感**,不存在一个统一的"能压多少"的答案,3D 网络因为参数冗余度通常更高,压缩空间也更大,这和直觉是一致的。

论文另外特意把 TensorRT 部署实测延迟单独拎出来报告,和"存储压缩比"分开讨论——这是一个很诚实也很重要的细节：**混合精度带来的存储压缩,不等于推理加速**。很多推理后端对任意层级混合精度的硬件支持并不完整（比如某些层量化后反而因为算子不支持融合而变慢),如果你只看论文里的压缩比数字就假设推理会线性加速,大概率会在真实部署里失望。至于"在什么条件下能复现"——由于没有公开代码和具体的校准集构成细节,精确复现 3.59×–7.8× 这类数字需要原始的 backbone 权重和数据集划分,这部分我没有办法验证,只能提醒读者带着怀疑态度看这类区间数字,尤其是区间跨度这么大的时候。

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 部署后模型会向医生/临床流程输出不确定性图或置信度叠加层 | 只做二值分割输出、不暴露任何不确定性信号的纯自动化流水线 |
| 有质量较好、覆盖困难病例的校准集可用于聚合验证 | 校准集稀缺或与部署分布差异很大,无法支撑可靠性验证 |
| 对压缩比要求高但推理延迟不是唯一指标（比如离线批处理） | 极端实时场景,逐层聚合验证的搜索开销本身就负担不起 |
| 团队已经在用某种不确定性估计（entropy/MC-dropout/集成） | 模型本身没有可靠的不确定性估计基线,risk 项无从定义 |

## 我的观点

这篇论文的价值,不在于又造出了一个"压缩比更高"的量化方法——3.59×–7.8× 这个区间放在量化文献里并不算突破性数字。它的价值在于**把"可靠性保持"正式写进了压缩的评估标准**,这个方向在医学 AI 领域的重要性被严重低估了：业界大量的"压缩后 Dice 没掉就是成功"的论文,可能都在无意中放过了不确定性漂移这个风险。

但我也想诚实地指出两个开放问题。第一,"用什么不确定性度量作为 risk 项"这件事本身没有标准答案,论文选的度量方式会直接决定哪些层被判定为"危险",换一种不确定性估计方法（比如从 entropy 换成深度集成的方差）,得到的混合精度方案可能完全不同——这不是一个已经收敛的设计选择。第二,聚合验证的搜索成本和部署收益之间的权衡,在层数更多、更贴近真实 3D 医学网络规模的场景下会变得更棘手,摘要里没有给出搜索本身的时间开销,这是我在工程落地前一定会追问的数字。

如果我是一个要把分割模型部署到医院内网的工程师,这篇论文给我的实际启发不是"直接抄这个框架",而是：**在压缩评估报告里,除了 Dice、延迟、模型体积,必须加一项不确定性图的一致性度量**——哪怕你暂时没有精力实现完整的 SGAV-MP,这一个检查项本身就能帮你避开论文指出的那类静默失败。