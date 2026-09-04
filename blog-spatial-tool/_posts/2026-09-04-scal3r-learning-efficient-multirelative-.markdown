---
layout: post-wide
title: "Scal3R 深度解析：冻结主干 + 1% 可训练 Token，如何解决在线 3D 重建的位姿漂移"
date: 2026-09-04 12:02:23 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.04201v1
generated_by: Claude Code CLI
---

## 一句话总结

Scal3R 把"在线 3D 重建"的位姿回归问题从"相对于第一帧的绝对外推"重新定义为"相对于多个历史关键帧的相对查询"，用约 1% 的可训练参数（轻量级 Token + 非对称注意力）配合在线位姿图优化，在 KITTI 上把平均轨迹误差（ATE）降低 60% 以上，单卡训练仅需 8 小时。

先说明一点：这篇论文本质上是模型架构 + 系统设计工作，不是底层 CUDA kernel 优化。但它的核心思路——**冻结绝大部分网络、只让极少量参数参与梯度更新**——本身就是一个非常值得从 GPU 显存/算力角度拆解的工程决策。本文会在实现部分具体分析这个决策省了多少显存、少算了多少次反向传播。

## 为什么需要这个？

### 问题的根源：训练分布之外的外推

在线 3D 重建模型（比如 DUSt3R/MASt3R 一类的基于回归的方法）通常这样工作：给定第一帧作为锚点，后续每一帧都直接回归出"相对于第一帧"的位姿。这个设计在训练时没问题，因为训练用的视频片段通常很短（几十帧量级）。

但推理时如果输入是几千帧的长视频，模型就要预测"相对于几千帧之前的锚点"的位姿——这个相对变换的量级（平移距离、旋转累积）远远超出了训练时见过的分布范围。神经网络对分布外输入的外推能力本来就差，位姿头在这种情况下给出的估计会带有系统性偏差。更麻烦的是，在线系统是逐帧累积的，早期的小偏差会被后续帧继续放大，最终表现为几何结构整体"塌陷"（geometric collapse）。

### 关键观察：不是所有模块都坏了

论文里一个很有意思的发现是：即使在位姿严重漂移的长视频里，**逐帧深度估计依然是稳定准确的**。也就是说，backbone 提取的局部几何信息（每一帧内部的结构）没有问题，出问题的只是"全局位姿头"这一个组件。这说明失败不是网络整体退化，而是一个局部模块在做超出能力范围的外推任务。

这个观察直接决定了修复方案的方向：**不需要重训整个网络，只需要修正位姿这一个环节的查询方式**。

## 核心原理

### 直觉类比

把绝对位姿回归想象成"站在起点，凭记忆估算自己走了多远、转了多少角度"——走的时间越长，误差累积越离谱。而相对位姿查询更像是"每走几步就低头看一眼刚才踩过的脚印，用最近的参照物校正方向"——参照物离得近，估计任务就没有偏离训练分布太远。

### 从单参考到多参考

原来的做法：`pose(frame_t) = f(frame_t, frame_0)`，t 越大，`frame_t` 和 `frame_0` 的间隔越超出训练分布。

Scal3R 的做法：维护一组历史关键帧 `{k_1, k_2, ..., k_n}`，对当前帧计算它相对每个关键帧的相对位姿：

$$
\Delta P_{t,i} = g(\text{frame}_t, \text{keyframe}_i), \quad i = 1, \dots, n
$$

因为关键帧是滑动维护的，任意一对 `(frame_t, keyframe_i)` 的时间间隔都被限制在训练分布覆盖的范围内，模型不再需要做长距离外推。这些成对的相对位姿估计随后被送入位姿图做全局一致性优化。

### 用非对称注意力注入轻量 Token，而不是微调整个网络

要实现"多参考查询"，直觉上你需要一个新的、能处理多个参考帧的位姿头。但重训整个 backbone 成本很高。Scal3R 的做法是：backbone 权重完全冻结，额外引入一小批**可学习的 Token**（约占总参数量 1%），这些 Token 通过一种**非对称注意力**机制和冻结的 backbone 特征交互——Token 作为 query，backbone 输出的多帧特征作为 key/value，注意力权重只反向传播到 Token 这一侧，不更新 backbone 权重。

这本质上是一种参数高效微调（PEFT）思路，类似 LoRA/Prompt-tuning 的家族，但专门针对"多参考位姿查询"这个任务设计了 Token 的接入方式。

## 代码实现

需要先说明：论文项目页在 `https://linjohnss.github.io/scal3r/`，截至本文撰写时页面上未明确标注代码仓库地址，所以以下代码是**基于论文描述复现的教学示例**，用于说明架构思路，不是论文官方实现，也没有做完整可复现性验证。数据部分我会明确标注是论文报告的数字。

### Baseline：绝对位姿回归（问题所在）

```python
import torch
import torch.nn as nn

class AbsolutePoseHead(nn.Module):
    """朴素做法：位姿头只看 backbone 输出的第 0 帧 + 当前帧特征"""
    def __init__(self, feat_dim=768):
        super().__init__()
        self.proj = nn.Linear(feat_dim * 2, 256)
        self.pose_out = nn.Linear(256, 7)  # 平移(3) + 四元数(4)

    def forward(self, feat_anchor, feat_t):
        # feat_anchor: 固定为 frame_0 的特征，t 越大问题越严重
        x = torch.cat([feat_anchor, feat_t], dim=-1)
        x = torch.relu(self.proj(x))
        return self.pose_out(x)  # 相对于 frame_0 的位姿，训练分布外会失真
```

**问题分析**：`feat_anchor` 在整个视频中固定不变，`pose_out` 需要学会表达"任意长度间隔"的相对变换。训练集里间隔通常只有几十帧，一旦推理视频长达数千帧，这个映射就在做严重的分布外插值。

### 优化版本：冻结主干 + 轻量 Token 的多参考查询

```python
class ReferencePoseQuery(nn.Module):
    """非对称注意力：可学习 Token 作为 query，
    冻结 backbone 的多关键帧特征作为 key/value"""
    def __init__(self, feat_dim=768, num_ref=4, num_heads=8):
        super().__init__()
        # 可学习 token，数量远小于 backbone 参数
        self.query_tokens = nn.Parameter(torch.randn(num_ref, feat_dim))
        self.cross_attn = nn.MultiheadAttention(
            feat_dim, num_heads, batch_first=True)
        self.pose_head = nn.Linear(feat_dim, 7)  # 每个参考帧一个相对位姿

    def forward(self, backbone_feats_keyframes, backbone_feat_current):
        # backbone_feats_keyframes: [B, num_ref, feat_dim] —— 已冻结的多关键帧特征
        # backbone_feat_current:    [B, 1, feat_dim]      —— 当前帧特征
        B = backbone_feat_current.shape[0]
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        # 非对称：query 来自可学习 token，key/value 来自冻结特征拼接当前帧
        kv = torch.cat([backbone_feats_keyframes, backbone_feat_current], dim=1)
        out, attn_w = self.cross_attn(q, kv, kv)
        rel_poses = self.pose_head(out)  # [B, num_ref, 7]
        return rel_poses  # 每个 token 对应一个关键帧的相对位姿估计
```

**为什么这样更省 GPU 资源**：backbone 全程 `requires_grad=False`，前向计算依然要跑（提取特征），但反向传播不需要计算 backbone 权重的梯度，也不需要给 backbone 参数分配 Adam 的一阶/二阶动量。举个数量级例子：如果 backbone 是 3 亿参数的 ViT，`query_tokens` + `cross_attn` + `pose_head` 总共只有几百万参数（约 1%），单是 Adam 优化器状态就能省下：

$$
300\text{M} \times 2 \times 4\text{bytes} \approx 2.4\text{GB}
$$

这还没算反向传播过程中省下的激活值梯度计算，这也是论文能在**单卡 8 小时内收敛**的直接原因之一——可训练参数少，梯度计算图短，显存压力小，能用更大 batch 或更长序列窗口。

### 在线位姿图优化（简化示意）

```python
class SlidingPoseGraph:
    """维护关键帧位姿图，用相对位姿估计做在线优化，
    并在检测到回环时加入约束抑制长程漂移"""
    def __init__(self, max_keyframes=20):
        self.keyframes = []          # 关键帧全局位姿估计
        self.edges = []              # (i, j, relative_pose, weight)
        self.max_keyframes = max_keyframes

    def add_frame(self, frame_id, rel_poses, ref_ids):
        # rel_poses: 当前帧相对每个 ref 关键帧的位姿, 来自 ReferencePoseQuery
        for pose, ref_id in zip(rel_poses, ref_ids):
            self.edges.append((ref_id, frame_id, pose, weight=1.0))
        if self._detect_loop_closure(frame_id):
            self._add_loop_constraint(frame_id)
        self._optimize()  # 实际实现可用 g2o / GTSAM 等库做位姿图优化

    def _detect_loop_closure(self, frame_id):
        # 简化：基于外观/特征相似度检索历史关键帧，此处省略具体实现
        return False

    def _add_loop_constraint(self, frame_id):
        pass  # 省略：加入回环边并提高该约束权重

    def _optimize(self):
        pass  # 省略：非线性最小二乘优化，最小化所有边的残差
```

**为什么这一步不可省略**：单靠"多参考相对位姿"只是让每次估计的误差变小，但误差依然会累积。位姿图优化把所有相对位姿约束联合求解，回环检测则在系统重新经过已知区域时提供一个"强锚点"，直接把累积误差拉回来，这是抑制长程漂移的关键，而不是位姿头本身。

### 常见错误

```python
# 错误 1：忘记把 backbone 设为 eval + 冻结梯度，导致 BatchNorm/Dropout 状态被污染
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False  # 少了这一步，优化器仍会更新 backbone

# 错误 2：关键帧窗口无限增长，显存/计算量线性上升
# 应该用固定大小的滑动窗口 + 淘汰策略，而不是保留所有历史帧
if len(self.keyframes) > self.max_keyframes:
    self.keyframes.pop(0)  # 简单先进先出，实际中常结合关键帧多样性挑选
```

## 性能实测

以下数据均来自论文摘要中报告的结果，不是本文作者的实测，特此说明：

| 指标 | 数值 | 说明 |
|------|------|------|
| KITTI 平均 ATE 降低 | > 60% | 相对于在线 baseline |
| 可训练参数占比 | ~1% | backbone 完全冻结 |
| 训练收敛时间 | 8 小时 | 单 GPU |
| 其他数据集表现 | SOTA | Virtual KITTI、Sintel、TUM-Dynamic、ScanNet、7-Scenes |

论文没有在摘要中给出具体显存占用数字，前文"2.4GB 优化器状态"是基于参数量做的估算，用于说明冻结策略的收益来源，不代表论文实测值。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 长视频在线 3D 重建，绝对位姿外推明显失效 | 短序列离线重建，训练/测试分布本就接近 |
| 已有强大预训练 backbone，只想低成本适配新任务 | 需要端到端联合优化 backbone 的场景（如目标域和预训练域差异极大） |
| 计算/显存资源有限，无法全参数微调 | 对实时性要求极高、无法承受在线位姿图优化开销的嵌入式场景 |

## 调试技巧

- **位姿图发散**：先检查单帧相对位姿估计本身的量级是否合理（平移/旋转是否明显超出物理常识），发散往往不是优化器的问题，而是上游位姿头输出了异常值。
- **冻结但仍在更新**：用 `sum(p.numel() for p in backbone.parameters() if p.requires_grad)` 打印确认可训练参数真的只有 Token 相关模块。
- **回环检测误报**：外观相似不代表位姿相近，建议在加入回环约束前用几何一致性（比如重投影误差）做二次校验，否则错误的回环约束会把整个位姿图"拉歪"。
- **Token 初始化敏感**：可学习 Token 数量少，初始化方差过大容易导致训练早期注意力权重塌缩到极少数关键帧上，建议用较小方差（如 0.02）初始化并观察注意力权重分布是否均匀。

## 延伸阅读

- 相对位姿回归与绝对位姿回归的分布外泛化问题，可以参考 DUSt3R/MASt3R 系列工作中关于成对位姿估计的设计动机。
- 非对称注意力 + 冻结 backbone 的参数高效微调思路，和 LoRA、Prompt Tuning 的核心动机一致，值得对比阅读理解"为什么只训练极少参数也能适配新任务"。
- 位姿图优化与回环检测是 SLAM 领域的经典问题，g2o、GTSAM 的官方文档对非线性最小二乘位姿图求解讲得比较清楚，值得作为工程实现参考。