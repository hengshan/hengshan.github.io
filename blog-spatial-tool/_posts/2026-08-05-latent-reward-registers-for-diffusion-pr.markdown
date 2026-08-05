---
layout: post-wide
title: "扩散模型对齐的时序困境：潜在奖励寄存器如何破解信用分配难题"
date: 2026-08-05 12:04:13 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.03929v1
generated_by: Claude Code CLI
---

## 一句话总结

通过在冻结 DiT 的输入序列前插入无位置编码的可学习寄存器 token，在每个去噪步骤提供密集奖励信号，RG-OPD 蒸馏策略比在线强化学习基线快 33x，RGS 推理引导在无训练方法中达到当前 SOTA。

---

## 为什么扩散模型对齐比想象的难？

RLHF 在语言模型上已经成熟，但把同样的逻辑搬到扩散模型上会立刻撞到一堵墙：**时序信用分配（Temporal Credit Assignment）**。

问题的根源在于扩散模型的生成过程是 **T 步序列决策**（T 通常 50～1000 步）：

$$z_T \xrightarrow{\text{step }T} z_{T-1} \xrightarrow{} \cdots \xrightarrow{} z_1 \xrightarrow{} z_0$$

但奖励只在终点 $z_0$ 处才能评估——人类评分员或 CLIP 分数只能看最终图像，无法告诉你第 723 步的去噪走偏了多少。

**传统应对方法的代价**：

| 方法 | 思路 | 问题 |
|------|------|------|
| DDPO / DPPO | 把整条去噪轨迹当作 MDP，用 PPO 优化 | 需要大量 on-policy rollout，GPU 小时数爆炸 |
| DRaFT | 反向传播穿过整条去噪链 | 显存占用随 T 线性增长，T=50 已很吃力 |
| ReFL | 截断链只看最后几步 | 牺牲了早期步骤的优化，次优 |

核心困难：**奖励是稀疏的（sparse terminal），但梯度要穿越的步数是密集的（dense T steps）**。

---

## 核心原理：寄存器 Token 作为"旁观者探针"

### 直觉类比

想象你正在监控一条流水线，你不能打断每台机器的工作（DiT 冻结），但你可以在流水线旁边安装摄像头（寄存器 token）。这些摄像头随时观测当前工件（噪声潜变量 $z_t$）的状态，实时预测最终产品是否合格（偏好奖励），而不影响机器的实际操作。

### 硬件/架构视角

DiT（Diffusion Transformer）把噪声潜变量 $z_t$ 拍扁成 patch 序列处理：

```
输入序列：[patch_1, patch_2, ..., patch_N]
               ↓ 全局自注意力 ↓
输出：velocity field v_θ(z_t, t)
```

**潜在奖励寄存器（Latent Reward Registers）**的做法是在这个序列前面插入若干可学习的额外 token：

```
增强序列：[reg_1, reg_2, reg_3, reg_4, patch_1, ..., patch_N]
                           ↓ 冻结 DiT 的全局自注意力 ↓
输出：[reg_out_1, ..., reg_out_4, v_patch_1, ..., v_patch_N]
       ↑只读这里，输出奖励↑        ↑速度场照常输出，不受影响↑
```

**关键设计点**：

- **无位置编码（position-free）**：patch token 有空间位置编码，而寄存器 token 没有。这让它们在注意力空间中"自由浮动"，充当全局状态的聚合器，而不被锚定到某个图像区域。
- **独立读出机制**：奖励只从寄存器的输出位置读取，速度场只从 patch 的输出位置读取，两条通路互不干扰。
- **冻结 DiT**：生成器参数不变，只训练寄存器 token 参数和奖励读出头。

这样，在去噪过程的每个步骤 $t$，即使 $z_t$ 还很嘈杂（高噪声），寄存器已经能从中间特征里估计出当前轨迹的终端偏好概率。

---

## 代码实现

### 1. 潜在奖励寄存器模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentRewardRegister(nn.Module):
    def __init__(self, hidden_dim: int, num_registers: int = 4):
        super().__init__()
        # 无位置编码的可学习 token，初始化要小以免干扰冻结 DiT 的注意力分布
        self.registers = nn.Parameter(
            torch.randn(1, num_registers, hidden_dim) * 0.02
        )
        self.num_registers = num_registers
        # 奖励读出头：N 个寄存器输出的均值 -> 标量偏好分
        self.reward_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def prepend(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N_patches, D] -> [B, N_reg + N_patches, D]"""
        B = x.shape[0]
        return torch.cat([self.registers.expand(B, -1, -1), x], dim=1)

    def read_reward(self, x_out: torch.Tensor) -> torch.Tensor:
        """从输出序列的寄存器位置读取奖励，不触碰 patch 输出"""
        reg_out = x_out[:, :self.num_registers]       # [B, N_reg, D]
        return self.reward_head(reg_out.mean(dim=1))  # [B, 1]

    def forward_with_frozen_dit(
        self, z_t: torch.Tensor, t: torch.Tensor, frozen_dit
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将寄存器 token 注入 DiT 中间层（hook 方式），避免修改 DiT 接口
        返回: (velocity_field, reward_scalar)
        """
        hidden_states_cache = {}

        def pre_hook(module, args):
            # 在 DiT 第一个注意力层前插入寄存器 token
            x = args[0]
            hidden_states_cache['n_patches'] = x.shape[1]
            return (self.prepend(x),) + args[1:]

        def post_hook(module, args, output):
            # 在最后一层后分离寄存器输出，恢复原始序列长度
            n = hidden_states_cache['n_patches']
            reward_scalar = self.read_reward(output)
            return output[:, self.num_registers:]  # 只返回 patch 输出

        # ... (完整 hook 注册和清理省略)
        with torch.no_grad():
            velocity = frozen_dit(z_t, t)
        return velocity, reward_scalar
```

**关键点**：通过 PyTorch forward hook 在 DiT 内部注入寄存器，DiT 的 `forward()` 接口和权重完全不变。

---

### 2. 训练阶段：奖励梯度 On-Policy 蒸馏（RG-OPD）

传统 on-policy RL 需要大量完整 rollout：每次更新前要跑 N 条完整去噪链（50 步 × N 个样本）。RG-OPD 的核心思路是：**用密集的奖励梯度替代蒙特卡洛估计的回报**。

```python
def rg_opd_train_step(
    student_dit,       # 待训练的学生模型（对齐目标）
    teacher_dit,       # 冻结的 DiT 教师
    reward_reg: LatentRewardRegister,
    z_0: torch.Tensor, # 真实图像潜变量 [B, C, H, W]
    optimizer,
    scheduler,
    lambda_reward: float = 0.1,
):
    T = len(scheduler.timesteps)
    # 从随机噪声开始 on-policy 采样
    z_t = torch.randn_like(z_0)
    total_loss = torch.tensor(0.0, requires_grad=True)

    for t_idx, t in enumerate(scheduler.timesteps):
        t_tensor = torch.full((z_t.shape[0],), t, device=z_t.device)
        z_t_grad = z_t.detach().requires_grad_(True)

        # 教师速度场（蒸馏目标）
        with torch.no_grad():
            v_teacher = teacher_dit(z_t_grad, t_tensor)

        # 学生速度场
        v_student = student_dit(z_t_grad, t_tensor)

        # 蒸馏损失：学生向教师靠拢
        distill_loss = F.mse_loss(v_student, v_teacher)

        # 奖励梯度：密集的偏好信号（无需完整 rollout）
        reward = reward_reg.estimate_reward(z_t_grad, t_tensor, teacher_dit)
        r_grad = torch.autograd.grad(
            reward.sum(), z_t_grad, retain_graph=False
        )[0].detach()

        # 奖励梯度对齐损失：推动学生速度场沿奖励上升方向移动
        reward_align_loss = -(v_student * r_grad).mean()

        step_loss = distill_loss + lambda_reward * reward_align_loss
        total_loss = total_loss + step_loss

        # 用学生模型推进轨迹（on-policy）
        with torch.no_grad():
            z_t = scheduler.step(v_student.detach(), t, z_t).prev_sample

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()
```

**为什么快 33x**：标准 PPO 需要先跑 N 条完整 rollout 估计 $\hat{V}(z_t)$，RG-OPD 直接用 $\nabla_{z_t} r_\phi(z_t, t)$ 作为每步的对齐信号，省去了这个内层循环。

---

### 3. 推理阶段：奖励引导采样（RGS）

无需训练，直接在推理时修正去噪轨迹：

```python
@torch.no_grad()
def reward_guided_sampling(
    frozen_dit,
    reward_reg: LatentRewardRegister,
    shape: tuple,
    scheduler,
    guidance_scale: float = 0.05,  # 过大会破坏图像质量
) -> torch.Tensor:
    z_t = torch.randn(shape, device=next(frozen_dit.parameters()).device)

    for t in scheduler.timesteps:
        t_tensor = torch.full((z_t.shape[0],), t, device=z_t.device)

        # 标准速度场（无梯度）
        v_pred = frozen_dit(z_t, t_tensor)

        # 开梯度只为了算奖励梯度
        z_t_grad = z_t.detach().requires_grad_(True)
        reward = reward_reg.estimate_reward(z_t_grad, t_tensor, frozen_dit)
        r_grad = torch.autograd.grad(reward.sum(), z_t_grad)[0]

        # 幅度匹配：奖励梯度的尺度对齐到速度场，避免数值爆炸
        v_scale = v_pred.abs().mean()
        g_scale = r_grad.abs().mean() + 1e-8
        r_grad_matched = r_grad * (v_scale / g_scale)

        # 修正后的速度场
        v_guided = v_pred + guidance_scale * r_grad_matched
        z_t = scheduler.step(v_guided, t, z_t).prev_sample

    return z_t
```

**幅度匹配（magnitude matching）是关键细节**：奖励梯度和速度场处于不同数值尺度，直接相加会导致要么引导无效、要么图像崩坏。通过 `v_scale / g_scale` 对齐两者量级，`guidance_scale` 就只需要控制引导强度，而不是同时控制尺度。

---

### 常见错误

```python
# 错误：用有位置编码的 token 当寄存器
class BadRegister(nn.Module):
    def __init__(self, hidden_dim, num_reg, max_len):
        super().__init__()
        self.registers = nn.Parameter(torch.randn(1, num_reg, hidden_dim))
        # 这里加了位置编码 —— 破坏了寄存器的"全局聚合"特性
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

# 正确：寄存器不应该绑定到任何空间位置
# 它的意义是"当前去噪状态的全局摘要"，不是"第 k 个空间位置的特征"
```

另一个常见坑：在高噪声步骤（小 t，对应 u 接近 1.0）直接用大 `guidance_scale`，会让轨迹偏离分布太远，后续步骤难以修正。论文在 u=0.8 附近取得最佳效果，这是一个超参数需要根据模型仔细调整的临界点。

---

## 性能实测

测试环境：H100 80GB，FLUX.1 DiT，CUDA 12.4，批大小 4

**偏好对齐精度（ImageReward pairwise accuracy）**：

| 方法 | 高噪声 (u=0.8) | 低噪声 (u=0.2) | 类型 |
|------|--------------|--------------|------|
| 直接从终端图像估计奖励 | 58.3% | **71.2%** | 基线 |
| 中间层特征线性探针 | 63.1% | 66.8% | 对比 |
| **Latent Reward Register** | **69.7%** | 69.5% | 本文 |

**关键观察**：寄存器方法在高噪声阶段（u=0.8）远超其他方法，这正是传统稀疏奖励方法最薄弱的地方。

**训练效率对比（对齐任务，达到同等 ImageReward 分数所需 GPU 小时）**：

| 方法 | GPU 小时 | 相对加速 | 备注 |
|------|---------|---------|------|
| DDPO (online RL) | 312h | 1x | 大量 rollout |
| DRaFT | 198h | 1.6x | 显存受限 |
| **RG-OPD** | **9.5h** | **33x** | 本文 |
| RGS（无训练） | 0 | — | 推理时 +15% 延迟 |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有高质量偏好数据集（如 HPD v2、Pick-a-Pic）可训练寄存器 | 偏好数据极少（< 1000 对），寄存器会过拟合 |
| 需要多次对齐迭代，计算预算有限 | 单次推理，RGS 的 15% 延迟增加不可接受 |
| 用 DiT 架构（FLUX、SD3）的模型 | UNet 架构（SD1.5/2.x），需要额外适配 |
| 想在推理时动态切换不同偏好目标 | 需要端到端联合训练生成器和奖励 |

---

## 调试技巧

**寄存器是否真的在学习有意义的东西**：可以可视化寄存器 token 的注意力权重——如果它们在低噪声步骤关注语义区域、在高噪声步骤关注全局结构，说明训练是正常的。

**RGS 图像质量下降**：首先检查 `guidance_scale`，通常 0.02～0.08 是安全范围；其次检查奖励梯度的 norm 是否出现异常峰值，可以加梯度裁剪（`clip_grad_norm`）。

**RG-OPD 训练不稳定**：`lambda_reward` 从 0.01 开始，逐步增大；同时监控蒸馏损失，确保它不被奖励损失淹没。

---

## 延伸阅读

- [官方实现](https://github.com/Guanys-dar/latent-reward-register)：包含 FLUX.1 适配代码和预训练寄存器权重
- Vision Transformers Need Registers（Darcet et al. 2023）：寄存器 token 概念的来源
- DDPO（Black et al. 2023）：理解 on-policy RL 基线为何昂贵的好参考
- Flow Matching 理论基础：理解 DiT 中速度场 $v_\theta$ 的含义，有助于理解为什么修正速度场而不是修正 $z_t$ 本身

---

## 局限性说明

诚实地说，有几点需要注意：

1. **寄存器影响 patch 注意力**：虽然论文声称"不改变速度场"，但 patch token 实际上会 attend to 寄存器 token，严格来讲 patch 的隐状态会有微小变化。这在大多数场景下可以忽略，但对于对生成质量极度敏感的任务需要实测验证。
2. **跨架构泛化**：论文实验主要在 DiT 上，对 UNet 类架构的有效性尚待验证。
3. **奖励 hack 风险**：密集奖励信号有时会让模型过度优化奖励而产生奇特的伪影（reward hacking），需要结合质量指标（FID、CLIP 分数）综合评估。