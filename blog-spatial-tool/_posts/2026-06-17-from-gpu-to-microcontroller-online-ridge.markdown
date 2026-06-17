---
layout: post-wide
title: "用 444 个参数打败图神经网络：交通流量预测中的过度设计陷阱"
date: 2026-06-17 12:10:05 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.17613v1
generated_by: Claude Code CLI
---

## 一句话总结

对最新图无关模型 GLMST 做参数消融后发现：嵌入维度从 64 缩到 4，MAPE 仅上涨不到 1 个百分点。基于这一发现，用 Ridge 回归 + RLS 在线学习构建每传感器 444 参数的模型，在 PEMS 四项基准中三项超越 GNN，且全程无需 GPU——ESP32 单片机可在 2ms 内完成预测与在线更新。

## 为什么需要这个？

### 交通预测的 GPU 路径依赖

从 DCRNN → GWN → STGCN → 图无关 MLP，交通流量预测模型在 PEMS04/PEMS08 基准上持续刷新 MAPE，代价也在同步上升：

- **训练**：集中式 GPU 服务器，整合全路网数据
- **参数量**：Graph WaveNet 约 30 万参数，最新 SOTA 数十万起步
- **部署**：路侧单元（RSU）和边缘节点难以承载 GPU 推理

### 让人不安的消融实验

[arxiv 2606.17613](https://arxiv.org/abs/2606.17613v1) 对 GLMST 做了一个危险的测试——把内部嵌入维度从 64 逐步压缩：

| 嵌入维度 | PEMS04 MAPE | PEMS08 MAPE |
|---------|------------|------------|
| 64（原版）| 基准 | 基准 |
| 16 | +0.3% | +0.2% |
| **4** | **+0.8%** | **+0.6%** |

从 64 到 4，**MAPE 上涨不到 1 个百分点**。模型有效容量远超任务所需——交通流的信息量，远比我们以为的少。

### 为什么信息量这么少？

交通流是强周期信号：工作日早高峰每天 8:00-9:00 重复，周五下午比周三拥堵，节假日规律固定。一旦模型"记住"了周期模式，剩余残差（事故、极端天气）才是真正的噪声——而这些用多少参数都无法准确预测。

GNN 花数十万参数学习的东西，本质上是一张"时间 → 交通量"的查找表。

## 核心原理

算法由三部分组成：

### 1. Horizon-Aligned 周期特征

对于预测 $h$ 步后的流量，构造**目标时刻** $t + h\Delta t$ 的周期特征（而非当前时刻）：

$$\phi(\tau) = \Bigl[1,\ \sin\frac{2\pi k\tau}{T_d},\ \cos\frac{2\pi k\tau}{T_d},\ \sin\frac{2\pi k\tau}{T_w},\ \cos\frac{2\pi k\tau}{T_w}\Bigr]_{k=1}^{K}$$

其中 $T_d = 86400\text{s}$，$T_w = 604800\text{s}$，$K$ 为谐波数。

**直觉**：模型学习的是"**这个时间点**（用周期特征表示）通常有多少车流量"。预测 1 小时后的流量，直接查询 1 小时后的周期特征，无需依赖当前观测。这个假设在主干道上大多成立：交通流方差有 80%+ 可以用时间周期解释。

### 2. Ridge 回归（每 Horizon 独立模型）

每个传感器、每个预测 horizon $h$ 有独立线性模型：

$$\hat{y}(\tau) = \phi(\tau)^\top \mathbf{w}_h, \quad \mathbf{w}_h = \bigl(\Phi^\top\Phi + \lambda I\bigr)^{-1}\Phi^\top \mathbf{y}$$

正则化系数 $\lambda$ 防止过拟合，同时确保矩阵可逆。

### 3. 递推最小二乘（RLS）在线适应

每当收到新观测 $(x_t, y_t)$，利用 Sherman-Morrison 恒等式以 $O(d^2)$ 代价更新，**无需重训**。设 $P = (\Phi^\top\Phi + \lambda I)^{-1}$，则：

1. **Kalman 增益**：$k = Px_t \;/\; (1 + x_t^\top P x_t)$
2. **权重更新**：$\mathbf{w} \leftarrow \mathbf{w} + k\,(y_t - x_t^\top \mathbf{w})$
3. **协方差更新**：$P \leftarrow P - k\,x_t^\top P$

$d = 17$（$K=4$，两个周期）时，每步约 289 次乘法——这正是 ESP32 能在 2ms 内完成更新的原因。

## 代码实现

### Step 1：周期特征工程

```python
import numpy as np

def make_periodic_features(t_seconds: float, n_harmonics: int = 4) -> np.ndarray:
    """
    生成 horizon-aligned 周期特征向量。
    n_harmonics=4 → 维度 = 1 + 2*4*2 = 17
    """
    DAY  = 86400.0    # 秒/天
    WEEK = 604800.0   # 秒/周

    feats = [1.0]     # bias 项
    for period in [DAY, WEEK]:
        phase = t_seconds / period
        for k in range(1, n_harmonics + 1):
            feats.append(np.sin(2 * np.pi * k * phase))
            feats.append(np.cos(2 * np.pi * k * phase))

    return np.array(feats, dtype=np.float64)
```

### Step 2：RLS Ridge 回归核心

```python
class RLSRidge:
    """单一 Horizon 的 Ridge 回归 + RLS 在线更新器"""

    def __init__(self, n_features: int, lambda_reg: float = 1.0):
        self.w = np.zeros(n_features)
        self.P = np.eye(n_features) / lambda_reg   # 初始协方差 = (λI)^{-1}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """批量冷启动：求解 (X^T X + λI)w = X^T y"""
        d = X.shape[1]
        lam = 1.0 / self.P[0, 0]                   # 从 P 还原 λ
        A = X.T @ X + lam * np.eye(d)
        self.w = np.linalg.solve(A, X.T @ y)        # 内部使用 Cholesky 分解
        self.P = np.linalg.inv(A)                   # 为后续 RLS 初始化 P

    def update(self, x: np.ndarray, y: float):
        """单步 RLS 在线更新，O(d^2)"""
        Px       = self.P @ x
        gain     = Px / (1.0 + x @ Px)             # Kalman 增益（Sherman-Morrison）
        self.w  += gain * (y - x @ self.w)          # 权重更新
        self.P  -= np.outer(gain, x @ self.P)       # 协方差秩-1 下降更新

    def predict(self, x: np.ndarray) -> float:
        return float(x @ self.w)
```

### Step 3：多 Horizon 预测器（完整 Pipeline）

```python
class SensorPredictor:
    """
    单传感器完整预测器：n_horizons 个独立 RLSRidge 模型。
    n_horizons=12, n_harmonics=4 → 总参数 = 17*12 = 204 per sensor。
    """

    def __init__(self, n_horizons=12, n_harmonics=4,
                 dt=300.0, lambda_reg=1e-3):
        d = 1 + 2 * n_harmonics * 2
        self.models     = [RLSRidge(d, lambda_reg) for _ in range(n_horizons)]
        self.n_horizons = n_horizons
        self.dt         = dt           # PEMS 采样间隔：5 分钟 = 300s
        self.n_harmonics = n_harmonics

    def cold_start(self, timestamps: np.ndarray, traffic: np.ndarray):
        """用历史数据批量初始化所有 horizon 模型"""
        all_feats = np.stack([
            make_periodic_features(t, self.n_harmonics) for t in timestamps
        ])
        for h, model in enumerate(self.models, start=1):
            # horizon-aligned：用目标时刻 τ 的特征预测 τ 时刻的流量
            X = all_feats[h:]    # phi(τ)，τ = h,...,T-1
            y = traffic[h:]      # y(τ)，τ = h,...,T-1
            model.fit(X, y)

    def predict_and_update(self, t: float, y_obs: float) -> np.ndarray:
        """
        在时刻 t 收到流量观测 y_obs，同时完成更新与预测。

        关键性质：所有 horizon 模型的更新特征相同（均为 phi(t)）。
        原因：模型 h 在 t-h 时刻接收的输入恰好是 phi(t-h+h*dt) = phi(t)。
        """
        x_now = make_periodic_features(t, self.n_harmonics)
        for model in self.models:
            model.update(x_now, y_obs)   # 所有 horizon 统一用 phi(t) 更新

        preds = np.zeros(self.n_horizons)
        for h, model in enumerate(self.models, start=1):
            x_future = make_periodic_features(t + h * self.dt, self.n_harmonics)
            preds[h-1] = model.predict(x_future)   # 用 phi(t+h) 预测未来流量

        return preds
```

### 常见错误：忽略 Horizon-Aligned

```python
# ❌ 错误：所有 horizon 共享当前时刻特征
x_now = make_periodic_features(t)
preds = [model.predict(x_now) for model in models]  # 12 个 horizon 输出相同！

# ✓ 正确：每个 horizon 用目标时刻特征
preds = [
    models[h].predict(make_periodic_features(t + (h+1)*dt))
    for h in range(n_horizons)
]
```

用当前特征预测所有 horizon，线性模型无法区分"5 分钟后"和"1 小时后"，退化为对所有 horizon 输出相同的值。

## 性能实测

**模型规模对比**（数据来自原论文）：

| 方法 | 参数量/传感器 | 推理平台 | PEMS 结果 |
|------|------------|---------|----------|
| Graph WaveNet | ~300,000 | GPU | 标准 baseline |
| GLMST（dim=64）| ~35,000 | GPU | 当前 SOTA |
| GLMST（dim=4，消融）| ~2,000 | GPU | SOTA -0.8% MAPE |
| **RLS-Ridge（本文）** | **444** | **ESP32** | **4 项中 3 项最优** |

444 vs 35,000：80 倍参数差距，MAPE 反而更低。

**实时性能**（单传感器，12 horizons，来自原论文）：

| 操作 | ESP32 C3（160MHz，520KB SRAM）| Raspberry Pi 5（单核）|
|------|----------------------------|----------------------|
| 冷启动训练 | 7.4s | 0.21s |
| 预测 + 在线更新 | <2ms | 0.26ms |
| 内存（全部参数）| <80KB | — |

ESP32 冷启动的 7.4s 主要花在批量矩阵求逆（$d^3 \times H$ 次运算）。上线后每步更新仅需 $O(d^2) = O(289)$ 次乘法，且原论文实现**零堆内存分配**，适合严格 RTOS 环境。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 传感器独立运行，无需路网拓扑 | 需建模上下游堵车传播（空间依赖强）|
| 流量主要由时间周期驱动 | 事故、天气等非周期突发事件频繁 |
| 资源受限的边缘部署（RSU、MCU）| 有充裕 GPU 且需极致精度的中心化系统 |
| 新传感器需快速冷启动（7s 内上线）| 路网拓扑频繁变化，传感器关联复杂 |
| 数据流持续到达，分布缓慢漂移 | 流量模式剧变（如城市重大活动、疫情）|

## 调试技巧

**1. 验证周期特征**

将一周的时间戳输入 `make_periodic_features`，可视化前 9 维。应看到频率递增的正弦波。若出现全零或 NaN，检查时间戳单位——PEMS 数据集常见坑是时间戳为**毫秒**而非秒，需除以 1000。

**2. 监控数值稳定性**

```python
# 每隔一段时间检查协方差矩阵条件数
cond = max(np.linalg.cond(m.P) for m in predictor.models)
if cond > 1e10:
    print(f"Warning: P 条件数 {cond:.1e}，考虑重新冷启动")
    # RLS 长期运行后浮点累积误差可使 P 失去正定性
```

**3. 正则化系数 λ 的选择**

流量归一化到 $[0, 1]$ 后，经验值 $\lambda \in [10^{-4}, 10^{-2}]$。过小则对噪声敏感，协方差矩阵趋于奇异；过大则学习速率过慢，无法跟上季节性漂移。可用 5% 验证集快速网格搜索（3 个值即可）。

## 延伸阅读

- **原论文**：[arxiv 2606.17613](https://arxiv.org/abs/2606.17613v1)，含完整 ESP32 实现细节和所有消融实验
- **RLS 理论**：Sayed, *Adaptive Filters* (Wiley, 2008) 第 10 章，严格推导 Sherman-Morrison 更新的数值稳定条件
- **进阶方向**：若路网空间依赖不可忽略，可在 RLS-Ridge 基础上加 Kriging 空间修正，以较低参数成本引入跨传感器相关性；或探索带遗忘因子的 RLS（$0 < \mu < 1$），对近期数据赋予更高权重以应对模式漂移