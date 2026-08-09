---
layout: post-wide
title: '把 LLM 推理请求路由到最"绿"的机房：碳感知调度工程实践'
date: 2026-08-09 08:04:57 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.06188v1
generated_by: Claude Code CLI
---

## 一句话总结

通过把 LLM 推理流量实时路由到边际碳强度最低的 GPU 集群，无需重训练、无需换硬件，可削减高达 **50.9%** 的 GPU 推理碳排放——代价是更高的工程复杂度和会话锁定引入的约 3 个百分点损失。

## 为什么这个问题值得工程师认真对待？

同样一张 H100，在美国西北部（水电为主）每度电排放约 40 g CO₂，在中西部（煤电为主）同样一度电可能排放 900 g 以上——差距超过 **20 倍**。这个差距在时间维度同样显著：太阳能充足的中午是 200 g/kWh，深夜调峰靠气电时变成 600 g/kWh。

意味着：**把请求放在哪里比 kernel 优化 10% 的碳影响更大**。而且这个优化不需要触碰模型权重。

## 核心概念：MOER 不是 AER

在深入实现之前，必须分清两个概念：

- **AER（Average Emissions Rate）**：电网所有在线发电机的加权平均排放强度
- **MOER（Marginal Operating Emissions Rate）**：你多消耗一度电时，电网边际上启动的那台发电机的排放强度

高峰时段增加负载，调度员会启动调峰机组，通常是煤电或天然气单循环机组。MOER 捕获的正是这个边际效应，比 AER 更能反映你的实际碳影响。WattTime 等服务提供实时 MOER 信号，这是论文驱动路由决策的数据源。

## 系统架构

```
请求流量
    │
    ▼
[碳感知路由层]  ←── MOER 信号（5 分钟刷新）
    │
    ├── us-west-2   MOER:  42 lbs/MWh  ←── 本小时最优
    ├── us-central  MOER: 380 lbs/MWh
    └── us-east-1   MOER: 210 lbs/MWh
         │
         ▼
    [生产路由器]  ←── 碳感知层是无损 overlay，不替换原有逻辑
```

关键设计原则：碳感知路由作为**严格可逆的 overlay** 叠加在现有路由器上。MOER 信号不可用或区域故障时，立即回退到原始路由器。

## GPU 能耗逐请求归因

论文的重要贡献之一：用 **NVIDIA DCGM** 对每个请求单独计量 GPU 能耗，而非用铭牌 TDP 估算。

H100 SXM5 的 TDP 是 700W，但实际推理功率高度依赖并发数和请求长度——从 1 并发的 150W 到 64 并发的 680W，用 TDP 均摊每个请求的能耗误差超过 4 倍。

```python
import pynvml, time
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass
class EnergyMeasurement:
    energy_kwh: float
    duration_s: float
    avg_power_w: float

class GPUEnergyMeter:
    def __init__(self, device_index: int = 0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.last: EnergyMeasurement | None = None

    @contextmanager
    def measure(self):
        # nvmlDeviceGetTotalEnergyConsumption 返回单调递增的累计值（mJ）
        start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        t0 = time.monotonic()

        yield  # 在此期间执行推理

        delta_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - start_mj
        dt = time.monotonic() - t0
        self.last = EnergyMeasurement(
            energy_kwh=delta_mj / 3.6e9,        # mJ → kWh
            duration_s=dt,
            avg_power_w=(delta_mj / 1000) / dt   # mJ → J，除以秒得 W
        )
```

> **多卡注意**：上述接口只反映单卡能耗，TP/PP 并行时需对所有参与卡求和。此外，DCGM 只测 GPU 本身，PUE（通常 1.1-1.5）引入的冷却和配电损耗需单独乘上。

### 并发修正

多请求同时执行时，能耗要按实测并发功率曲线分摊，而非简单除以并发数（功率随并发数非线性增长）：

```python
def attribute_energy_per_request(
    request_durations: list[float],
    concurrency_curve: dict[int, float],  # {并发数: 实测功率(W)}
) -> list[float]:
    n = len(request_durations)
    # 查实测功率曲线；超出范围取最大并发的实测值
    power_w = concurrency_curve.get(n, concurrency_curve[max(concurrency_curve)])
    # 每个请求按自身占用时长归因
    return [(power_w * dur / 3600) / 1000 for dur in request_durations]
```

## 碳感知路由器实现

```python
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class Region:
    name: str
    endpoint: str
    moer: float = float("inf")  # lbs CO₂e/MWh，必须用绝对值
    healthy: bool = True
    last_updated: float = field(default_factory=time.time)

class CarbonAwareRouter:
    def __init__(self, regions: list[Region], stale_s: float = 600):
        self.regions = {r.name: r for r in regions}
        self.stale_s = stale_s  # 10 分钟无更新视为信号过期
        self.session_pins: dict[str, str] = {}

    def route(self, session_id: Optional[str] = None) -> Optional[Region]:
        # Session pinning：对话必须发到同一区域以复用 KV cache
        if session_id and (pinned := self.session_pins.get(session_id)):
            if (r := self.regions.get(pinned)) and r.healthy:
                return r

        now = time.time()
        candidates = [
            r for r in self.regions.values()
            if r.healthy and (now - r.last_updated) < self.stale_s
        ]
        if not candidates:
            return None  # 信号全部过期，回退到生产路由器

        # 关键：按绝对 MOER 值排序，不是百分位
        best = min(candidates, key=lambda r: r.moer)
        if session_id:
            self.session_pins[session_id] = best.name
        return best

    def update_moer(self, region_name: str, moer: float):
        if r := self.regions.get(region_name):
            r.moer = moer
            r.last_updated = time.time()
```

## 一个容易犯的错误：百分位信号 vs 绝对值

WattTime API 提供两种信号：

| 信号类型 | 含义 | 适用场景 |
|---------|------|---------|
| `moer`（绝对值） | lbs CO₂e/MWh | **跨区域比较** |
| `signal_index`（百分位） | 0-100，区域内历史分位数 | 判断本区域内"现在是否是好时机" |

**百分位信号回答的是时序问题，不是空间问题。**

具体例子：
- 区域 A（西北水电）：MOER = 85 lbs/MWh，本区域历史第 90 百分位（此时比平时脏）
- 区域 B（中西部煤电）：MOER = 750 lbs/MWh，本区域历史第 20 百分位（此时比平时干净）

按百分位路由会选区域 B——实际碳排放是区域 A 的 **9 倍**。

## 历史数据回放模拟

论文核心量化结果来自用一年历史 MOER 数据回放（非预测信号驱动），这给出了无预测误差的理论上界：

```python
import pandas as pd

def simulate_routing(moer_df: pd.DataFrame) -> pd.DataFrame:
    """
    回放历史 MOER 比较三种策略
    moer_df columns: [timestamp, region, moer]（单位：lbs CO₂e/MWh）
    """
    pivot = moer_df.pivot(index="timestamp", columns="region", values="moer")
    static_best = pivot.mean().idxmin()  # 年均 MOER 最低的区域
    rr_idx = 0
    regions = pivot.columns.tolist()
    rows = []

    for ts, row in pivot.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        rows.append({
            "timestamp": ts,
            "round_robin": valid.iloc[rr_idx % len(valid)],  # 轮询
            "static_best": valid.get(static_best, valid.mean()),
            "carbon_aware": valid.min(),  # 每小时选最低 MOER
        })
        rr_idx += 1

    df = pd.DataFrame(rows)
    baseline = df["round_robin"].mean()
    summary = {s: {"mean_moer": df[s].mean(),
                   "reduction": f"{(baseline - df[s].mean()) / baseline:.1%}"}
               for s in ["round_robin", "static_best", "carbon_aware"]}
    return pd.DataFrame(summary).T
```

## 性能实测数据

以下是论文报告的结果（CONUS 多区域机群，一年历史 MOER 回放）：

| 策略 | vs 轮询降幅 | 备注 |
|------|------------|------|
| 轮询（baseline） | — | 实际生产压力路由器 |
| 静态最优区域（年均最低） | ~32% | 选对机房，全年不变 |
| 逐小时最低 MOER | ~54% | 理论上界，无预测误差 |
| 实际碳感知路由（含 session pinning） | **50.9%**（CI: 48.5-53.3%） | 对历史 MOER 结算 |

关键解读：**静态路由**选对数据中心就能省约 32%，是最大的单一杠杆。动态路由额外贡献约 22 个百分点，Session pinning 消耗约 3 个百分点。

## 工程实践中的坑

### 坑 1：MOER 信号是预测值

实时 MOER 是预测值，事后结算值才是真实碳成本。预测误差在高波动时段可达 30-50 lbs/MWh。50.9% 是历史 MOER 回放的上界，实操中应打折扣。在误差范围内优先选低延迟区域是合理的降级策略：

```python
def route_with_uncertainty(candidates: list[Region],
                            moer_tolerance_lbs: float = 50.0) -> Region:
    best_moer = min(r.moer for r in candidates)
    # MOER 差距在预测误差范围内时，优先选负载低的区域
    near_optimal = [r for r in candidates if r.moer - best_moer <= moer_tolerance_lbs]
    return min(near_optimal, key=lambda r: r.active_sessions)
```

### 坑 2：Session pinning 的碳代价

多轮对话 pin 到同一区域以复用 KV cache。会话开始时 us-west-2 是最优，4 小时后 MOER 飙升，该用户后续请求无法迁移。会话越长，锁定期内的 MOER 波动越大，碳代价越高。

### 坑 3：网络传输碳成本

跨区域路由增加了骨干网流量，骨干网本身也有碳成本。论文的数据未计入此项。对大 batch 离线推理（图像/视频处理）场景，传输比例会显著增大，需单独核算。

## 什么时候用，什么时候不用

| 适用场景 | 不适用场景 |
|---------|-----------|
| 多区域 GPU 集群，区域间 MOER 差距 > 2x | 单区域部署（无可路由地址） |
| 批量推理、异步离线任务 | 实时流式应用（跨区延迟不可接受） |
| 无状态短会话（单轮 Q&A） | 长上下文多轮对话（session pinning 代价高） |
| 有 ESG 合规或碳预算约束的企业应用 | 跨区域带宽费用超过碳收益的场景 |

## 调试思路

- **验证 MOER 信号是绝对值**：打印各区域的原始 MOER 数值，如果全部在 0-100 范围内，你拿到的是百分位信号，而非绝对值
- **检查信号新鲜度**：在路由日志里记录 `last_updated`，监控 stale region 的比例——超过 20% 说明 MOER 服务有问题
- **碳结算审计**：每次请求记录 `(region, energy_kwh, moer_at_dispatch)`，事后用历史 MOER 重算实际碳成本，对比预测值的偏差

## 延伸阅读

- 原论文：[Routing LLM Inference to the Cleanest Grid in Real Time](https://arxiv.org/abs/2608.06188v1)
- WattTime API 文档：MOER 信号获取与 `signal_index` vs `moer` 的区别
- NVIDIA DCGM 文档：`nvmlDeviceGetTotalEnergyConsumption` 的精度说明与多卡聚合方法
- Green Software Foundation：软件碳强度（SCI）评分规范，提供了统一的碳归因框架