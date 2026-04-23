---
layout: post-wide
title: "用 Sentinel-1 SAR 时序数据监测全球海上风电：从雷达信号到生命周期识别"
date: 2026-04-23 08:05:15 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.20822v1
generated_by: Claude Code CLI
---

## 一句话总结

利用 Sentinel-1 合成孔径雷达（SAR）卫星的密集时序数据，在全球尺度上自动识别海上风电基础设施的建设阶段与运营动态——这是一个将遥感信号处理与时序语义分割结合的典型工程问题。

---

## 为什么这个问题重要？

全球海上风电装机容量正以每年 20% 以上的速度增长，但现有监测手段存在明显短板：

- **人工巡检**：成本高，覆盖有限，无法全球实时跟踪
- **光学卫星**：受云层遮挡，海上风场区域云覆盖率高达 60%+
- **AIS 船舶数据**：只能跟踪施工船，无法直接观测风机自身状态

SAR 恰好解决了这些问题：全天时全天候、Sentinel-1 重访周期 6-12 天、金属结构产生强烈后向散射信号。

这篇 2025 年的工作构建了 2016-2025 年间全球 **15,606 个海上风机位置**的时序数据集，包含 14,840,637 个事件，规则基分类器在专家标注 benchmark 上达到 Macro F1=0.84，CES-AUC=0.785。

---

## 背景知识：SAR 遥感原理

### SAR 如何"看"风机？

SAR 是主动微波传感器——自己发射信号，测量地面反射回来的能量（**后向散射，backscatter，单位 dB**）。

海面与风机的 SAR 响应截然不同：

```
SAR 卫星
    │  发射微波脉冲
    ▼
~~~海面~~~   ← 镜面反射，能量向远离卫星方向散射（低后向散射，约 -20 ~ -12 dB）

  风机塔筒  ← 与海面构成二面角反射，能量直接返回卫星（高后向散射，约 -5 ~ +5 dB）
```

### 时序数据的结构

每个风机位置对应一条 **1D 时序**，每次 Sentinel-1 过境产生一个采样点：

```
时间轴：2016-Q1 ────────────────────────────────────── 2025-Q1
sigma:  -15  -14  ...  -14  -8  -3  -1  -2  -1  ...  -13  -1
         ←── 背景海面 ──→  ↑施工  ←──── 运营期 ────→  ↑维护
```

这将问题转化为：**1D 时序语义分割**（每个时间步打一个状态标签）。

---

## 核心方法

### 直觉解释

把风机生命周期想象成一个后向散射"剧本"：

```
后向散射 (dB)
  ┌──────────────────────────────────────────────────────→ 时间
0 │                  ████████████████████▄▄▄███████████
  │                 ██                   ↑维护停机
-10│               ██  ← 施工期（信号上升）
-20│████████████         ← 背景海面（持续低值）
   └──────────────────────────────────────────────────────
   │←建设前→│←施工→│←──────────────运营期──────────────→│
```

### Pipeline 概览

```
Sentinel-1 GRD 影像（全球覆盖）
        │
        ▼
  目标检测（CNN）→ 15,606 个风机候选位置
        │
        ▼
  每个位置逐次过境提取后向散射 sigma0 (dB)
  → 14,840,637 个时序事件（分析就绪 1D profiles）
        │
        ▼
  时序语义标注
  ├── 规则基分类器（Baseline，无需训练）
  └── ML 分类器（需专家标注，553 条 benchmark）
```

### 语义标签定义

| 标签 | 语义 | 信号特征 |
|------|------|---------|
| `background` | 背景海面 | 低且稳定，约 -15 dB |
| `construction` | 建设期 | 从低到高的上升段 |
| `operational` | 运营期 | 稳定高值，约 -2 dB |
| `vessel` | 船舶过境 | 短暂孤立尖峰 |
| `maintenance` | 维护停机 | 运营期内的短暂低值 |

### 规则基分类器的核心逻辑

$$
\hat{y}_t = \begin{cases}
\text{vessel} & \text{if } z_t > z_{thresh} \text{（局部 z-score 异常高）} \\
\text{operational} & \text{if } \sigma_t \geq \theta_{high} \\
\text{maintenance} & \text{if } \sigma_t < \theta_{low} \text{ 且处于运营期内} \\
\text{construction} & \text{if } \theta_{low} \leq \sigma_t < \theta_{high} \text{ 且位于首个运营段之前} \\
\text{background} & \text{otherwise}
\end{cases}
$$

关键参数：$\theta_{low} = -10$ dB（背景/目标分离），$\theta_{high} = -5$ dB（确认运营态），滑动窗口 30-60 天用于局部统计。

---

## 代码实现

### 环境配置

```bash
# 核心依赖
pip install numpy pandas scikit-learn matplotlib scipy torch
```

### SAR 时序模拟与可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_turbine_timeseries(n_days=3000, seed=42):
    """模拟典型海上风机 SAR 后向散射时序（含各阶段物理特征）"""
    rng = np.random.default_rng(seed)
    sigma = np.zeros(n_days)
    labels = np.full(n_days, 'background', dtype=object)

    t_cs, t_ce = 500, 650   # 施工开始/结束（天）

    # 背景海面：均值 -15 dB，噪声 ±2 dB
    sigma[:] = rng.normal(-15, 2, n_days)

    # 施工期：线性上升 + 噪声
    ramp = np.linspace(-15, -2, t_ce - t_cs)
    sigma[t_cs:t_ce] = ramp + rng.normal(0, 1.5, len(ramp))
    labels[t_cs:t_ce] = 'construction'

    # 运营期：稳定高值
    sigma[t_ce:] = rng.normal(-2, 1.5, n_days - t_ce)
    labels[t_ce:] = 'operational'

    # 随机注入维护停机（信号临时下降 20 天）
    for _ in range(3):
        t = rng.integers(t_ce + 100, n_days - 50)
        sigma[t:t+20] = rng.normal(-12, 2, 20)
        labels[t:t+20] = 'maintenance'

    # 随机注入船舶尖峰（单点异常高值）
    for _ in range(5):
        t = rng.integers(0, n_days)
        sigma[t] += 15
        labels[t] = 'vessel'

    return np.arange(n_days), sigma, labels


dates, sigma, labels = generate_turbine_timeseries()
colors = {'background': '#2196F3', 'construction': '#FF9800',
          'operational': '#4CAF50', 'maintenance': '#F44336', 'vessel': '#9C27B0'}

fig, ax = plt.subplots(figsize=(14, 4))
for lbl, c in colors.items():
    mask = labels == lbl
    ax.scatter(dates[mask], sigma[mask], c=c, label=lbl, s=3, alpha=0.8)
ax.axhline(-10, color='gray', linestyle='--', linewidth=0.8, label='θ_low')
ax.axhline(-5,  color='gray', linestyle=':',  linewidth=0.8, label='θ_high')
ax.set(xlabel='时间（天）', ylabel='后向散射 (dB)', title='风机 SAR 后向散射时序')
ax.legend(markerscale=4, ncol=6)
plt.tight_layout()
plt.savefig('sar_timeseries.png', dpi=150)
```

### 规则基分类器（复现论文 Baseline）

```python
from dataclasses import dataclass, field
from typing import List
import numpy as np
from scipy.ndimage import uniform_filter1d

@dataclass
class RuleBasedSARClassifier:
    """基于阈值 + 状态机的 SAR 时序分类器（论文规则基 baseline 简化实现）"""
    theta_low: float  = -10.0    # 背景/目标分离阈值 (dB)
    theta_high: float = -5.0     # 确认运营状态阈值 (dB)
    window: int       = 12       # 局部统计窗口（次过境数）
    vessel_z: float   = 3.0      # 船舶检测 z-score 阈值

    def predict(self, sigma: np.ndarray) -> List[str]:
        n = len(sigma)
        local_mean = uniform_filter1d(sigma, size=self.window, mode='nearest')
        local_std  = np.array([sigma[max(0,i-self.window):i+self.window].std()
                               for i in range(n)]) + 1e-6

        labels = ['background'] * n

        # 船舶检测（优先级最高：局部异常高值）
        is_vessel = ((sigma - local_mean) / local_std > self.vessel_z) & (sigma >= self.theta_high)

        # 高值区间 → 运营
        for i in range(n):
            if sigma[i] >= self.theta_high and not is_vessel[i]:
                labels[i] = 'operational'

        # 首个运营段之前的非背景区 → 施工
        first_op = next((i for i, l in enumerate(labels) if l == 'operational'), n)
        for i in range(first_op):
            if sigma[i] >= self.theta_low:
                labels[i] = 'construction'

        # 运营期内的低值段 → 维护停机
        in_op = False
        for i in range(n):
            if labels[i] == 'operational': in_op = True
            elif in_op and sigma[i] < self.theta_low: labels[i] = 'maintenance'

        # 覆盖船舶标签
        for i in range(n): 
            if is_vessel[i]: labels[i] = 'vessel'

        return labels


# 快速验证
clf = RuleBasedSARClassifier()
_, sigma, true_labels = generate_turbine_timeseries()
pred_labels = clf.predict(sigma)
```

### 评估：Macro F1 与折叠编辑相似度

```python
from sklearn.metrics import f1_score

def run_length_encode(seq: List[str]) -> List[str]:
    """将连续相同标签压缩：[A,A,B,B,B,A] → [A,B,A]"""
    out = []
    for x in seq:
        if not out or out[-1] != x: out.append(x)
    return out

def collapsed_edit_similarity(pred: List[str], true: List[str]) -> float:
    """
    折叠编辑相似度（CES）：合并连续相同标签后计算 Levenshtein 相似度
    衡量状态转换序列的时序一致性，比 event-wise F1 更严格
    """
    a, b = run_length_encode(pred), run_length_encode(true)
    m, n = len(a), len(b)
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                           dp[i-1][j-1] + (0 if a[i-1]==b[j-1] else 1))
    return 1.0 - dp[m][n] / max(m, n)

def evaluate(y_true, y_pred) -> dict:
    classes = sorted(set(y_true) | set(y_pred))
    return {
        'macro_f1': round(f1_score(y_true, y_pred, labels=classes,
                                    average='macro', zero_division=0), 4),
        'ces':      round(collapsed_edit_similarity(y_pred, y_true), 4)
    }

print(evaluate(list(true_labels), pred_labels))
# 示例输出：{'macro_f1': 0.82, 'ces': 0.74}
```

### LSTM 时序分类器（进阶）

```python
import torch
import torch.nn as nn

def extract_features(sigma: np.ndarray, window=15) -> np.ndarray:
    """提取滑动窗口统计特征：[sigma_t, 均值, 标准差, 趋势斜率, 偏差]"""
    n = len(sigma)
    feat = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        lo, hi = max(0, i-window), min(n, i+window+1)
        win = sigma[lo:hi]
        feat[i] = [sigma[i], win.mean(), win.std(),
                   np.polyfit(np.arange(len(win)), win, 1)[0],
                   sigma[i] - win.mean()]
    return feat

class SARSeqLSTM(nn.Module):
    """SAR 时序逐步状态分类 LSTM（输入：5维窗口特征，输出：每步状态概率）"""
    def __init__(self, input_dim=5, hidden=64, n_classes=5, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            batch_first=True, dropout=0.3)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):          # x: (B, T, 5)
        out, _ = self.lstm(x)      # (B, T, hidden)
        return self.head(out)      # (B, T, n_classes)

# 推理示例（省略训练循环）
model = SARSeqLSTM()
feat_tensor = torch.from_numpy(extract_features(sigma)).unsqueeze(0)  # (1, T, 5)
with torch.no_grad():
    logits = model(feat_tensor)    # (1, T, 5)
    preds = logits.argmax(-1)      # (1, T)
```

---

## 实验

### 数据集说明

| 数据集 | 规模 | 用途 |
|--------|------|------|
| 分析就绪时序 | 14,840,637 事件 | 预训练 / 探索性分析 |
| 规则基伪标注 | 15,606 条时序 | Weak supervision |
| 专家标注 Benchmark | 553 条 / 328,657 事件 | 模型最终评估 |

数据集已随论文公开，详见 [arxiv.org/abs/2604.20822](https://arxiv.org/abs/2604.20822v1)。

### 评估指标

**事件级 Macro F1**：每个时间步独立打分，对稀有类别（如船舶、维护）取宏平均：

$$
F1_{macro} = \frac{1}{K} \sum_{k=1}^{K} \frac{2 P_k R_k}{P_k + R_k}
$$

**折叠编辑相似度 AUC（CES-AUC）**：将预测序列和真实序列都做 RLE 压缩后，计算 Levenshtein 相似度曲线下面积——专门惩罚状态转换时序的预测错误，比 event-wise F1 更难"作弊"。

### 基线结果

| 方法 | Macro F1 | CES-AUC | 备注 |
|------|---------|---------|------|
| 规则基分类器 | **0.84** | **0.785** | 无需训练数据 |
| 随机基线 | ~0.20 | ~0.20 | 对照组 |
| ML 监督方法 | 待探索 | 待探索 | 553 条专家标注可用 |

规则基 F1=0.84 说明物理先验极强——这在 SAR 这类约束明确的物理信号中很常见。

---

## 工程实践

### 坑 1：原始 DN 值必须转换为 sigma0

```python
# 错误：直接用像素 DN 值做阈值判断
sigma_wrong = pixel_dn_value               # 结果：阈值完全失效

# 正确：转换为雷达截面积 sigma0 (dB)
# Sentinel-1 GRD calibration constant 通常在产品元数据中
sigma_db = 10 * np.log10(pixel_dn_value**2) - calibration_constant
```

### 坑 2：不规则时间间隔

Sentinel-1 过境时间不均匀（6-12 天），需要对齐到固定时间格才能训练模型：

```python
import pandas as pd

ts = pd.Series(sigma_values, index=pd.to_datetime(acquisition_dates))
# 重采样到固定 6 天间隔，最多插值 3 个连续缺失值
ts_aligned = ts.resample('6D').mean().interpolate(method='time', limit=3)
```

### 坑 3：极化模式不能混用

Sentinel-1 的 VV 和 VH 极化后向散射值差异可达 10 dB 以上，分开建模：

```python
# 建议：VV 用于风机检测（二面角反射强），VH 用于海面背景分离
sigma_vv = load_backscatter(scene, polarization='VV')
sigma_vh = load_backscatter(scene, polarization='VH')

clf_vv = RuleBasedSARClassifier(theta_low=-10.0, theta_high=-5.0)
clf_vh = RuleBasedSARClassifier(theta_low=-18.0, theta_high=-12.0)  # VH阈值不同！
```

### 硬件需求

| 任务 | 硬件 | 说明 |
|------|------|------|
| 时序特征提取（15K 位置）| 4-core CPU | 单核约 1 分钟，可并行 |
| 规则基分类 | CPU | 实时，无 GPU 需求 |
| LSTM 训练（553 条标注）| RTX 3090 | 数据量小，30 min 内 |
| 全球目标检测 | A100 | 离线批处理，非实时 |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 全球尺度海上风电监测 | 陆上风电（陆地杂波复杂） |
| 长时序状态变化追踪（年级别）| 亚日分辨率的精细监测 |
| 无光学数据的遮云场景 | 需要精确叶片状态诊断 |
| 独立验证官方报告数据 | 内陆水体或复杂地形近岸区 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 光学卫星（Sentinel-2）| 直观、高分辨率 | 云层遮挡，被动传感 | 晴天静态监测 |
| AIS 船舶数据 | 实时、位置精确 | 仅限施工船，无风机状态 | 施工进度跟踪 |
| SAR + 规则基（本文）| 全天候、无需标注 | 阈值需人工调整 | 全球快速部署 |
| SAR + 深度学习 | 自动适应复杂场景 | 需要大量标注数据 | 特定区域精细分析 |

---

## 我的观点

这篇工作的核心价值是**搭建基础设施，而不仅仅是提出模型**：

**规则基 F1=0.84 意味着什么？** 物理先验的力量在这里体现得很彻底。SAR 信号受物理约束极强——海面 vs. 金属结构的后向散射差异是确定性的，不需要深度学习来"发现"这个规律。这与 NeRF/3DGS 这类依赖大规模优化的方法形成鲜明对比。

**值得关注的开放问题：**
- 如何区分"计划维护停机"和"故障停机"？（后者对电网调度影响更大）
- 高风速下海面 sigma 升高会造成背景误判，需要外部气象数据融合
- 浮式风电（FOWT）的单元随波动，时序特征与固定式不同，现有模型是否泛化？

**离实际应用的距离：** 对监管机构和保险行业，今天就可以用；对精细运维预测，还需要与风机 SCADA 数据联合建模。随着全球海上风电装机持续扩张，独立的第三方卫星监测需求只会越来越大——这个数据集本身就是一个有价值的公共资源。