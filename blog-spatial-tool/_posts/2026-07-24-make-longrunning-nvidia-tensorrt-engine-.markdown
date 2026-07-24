---
layout: post-wide
title: "让 TensorRT 引擎构建过程可观测、可中断"
date: 2026-07-24 08:04:15 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://developer.nvidia.com/blog/make-long-running-nvidia-tensorrt-engine-builds-observable-and-cancelable-in-python-or-c/
generated_by: Claude Code CLI
---

## 一句话总结

通过实现 `IProgressMonitor` 接口，可以将 TensorRT 引擎构建的"黑盒等待"变为实时进度反馈，并支持任意时刻优雅中断——对需要在生产环境管理构建流程的工程师尤为关键。

---

## 为什么需要这个？

TensorRT 引擎构建的耗时差异极大：

| 模型规模 | 估算耗时 | 备注 |
|---------|---------|------|
| ResNet-50 | 10–30 秒 | FP16，热 timing cache |
| BERT-large | 2–5 分钟 | INT8，深度 tactic 搜索 |
| GPT-2 medium | 5–15 分钟 | 强类型模式 |
| 大型 LLM | 30 分钟以上 | 冷 cache + 全精度校准 |

**硬件层面发生了什么**：TensorRT 在构建期间会针对目标 GPU 实际运行数百个候选 kernel（tactic），选出最快的组合。这不是编译，是真实的 benchmark 过程，数据量越大、精度配置越复杂，搜索空间越大。

**两个核心痛点**：

1. **不可观测**：构建过程没有任何进度输出，用户无法区分"正在搜索"和"进程卡死"
2. **不可中断**：只能 `kill -9`，无法在服务热更新时优雅地放弃当前构建

TensorRT 8.6+ 引入了 `IProgressMonitor` 接口，正是为了解决这两个问题。

---

## 接口设计：阶段树模型

`IProgressMonitor` 抽象了一个**嵌套阶段树**。构建过程由多个**阶段**（phase）组成，每个阶段内部有若干**步骤**（step），阶段之间存在父子关系。

```
BuilderPhase (root)
├── NetworkPreprocessing
│   ├── step 0..N
├── TacticSearch
│   ├── LayerA
│   │   ├── step 0..M
│   └── LayerB
│       ├── step 0..K
└── SerializationPhase
    └── step 0..P
```

接口定义三个回调：

- `phase_start(phase_name, parent_phase, num_steps)` — 一个阶段开始
- `step_complete(phase_name, step) → bool` — 单步完成；**返回 `False` 即取消整个构建**
- `phase_finish(phase_name)` — 阶段结束

`stepComplete` 的返回值是唯一的取消信号——这是设计的精妙之处：不需要线程锁来中断，只需在下一次回调时翻转标志位。

---

## Python 实现：基础进度监控

先写一个有用的基础版本：带时间戳、层级缩进、进度百分比。

```python
import tensorrt as trt
import time
from collections import defaultdict

class TRTProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._phases = {}          # phase_name -> {total, done, start, depth}
        self._depth_map = {}       # phase_name -> 缩进层级
        self._should_cancel = False

    def phase_start(self, phase_name: str, parent_phase: str, num_steps: int):
        depth = self._depth_map.get(parent_phase, -1) + 1 if parent_phase else 0
        self._depth_map[phase_name] = depth
        self._phases[phase_name] = {
            "total": num_steps,
            "done": 0,
            "start": time.time(),
        }
        indent = "  " * depth
        print(f"{indent}▶ [{phase_name}] 开始，共 {num_steps} 步")

    def step_complete(self, phase_name: str, step: int) -> bool:
        if phase_name in self._phases:
            info = self._phases[phase_name]
            info["done"] = step
            total = info["total"]
            pct = (step + 1) / total * 100 if total > 0 else 0
            elapsed = time.time() - info["start"]
            depth = self._depth_map.get(phase_name, 0)
            indent = "  " * depth
            # 只在每 10% 或最后一步时打印，避免刷屏
            if total <= 10 or (step + 1) % max(1, total // 10) == 0 or step + 1 == total:
                eta = elapsed / (step + 1) * (total - step - 1) if step >= 0 else 0
                print(f"{indent}  {step+1}/{total} ({pct:.0f}%) | 已用 {elapsed:.1f}s | 预计剩余 {eta:.1f}s")
        return not self._should_cancel

    def phase_finish(self, phase_name: str):
        if phase_name in self._phases:
            elapsed = time.time() - self._phases[phase_name]["start"]
            depth = self._depth_map.get(phase_name, 0)
            indent = "  " * depth
            print(f"{indent}✓ [{phase_name}] 完成，耗时 {elapsed:.1f}s")
            del self._phases[phase_name]
```

挂载到构建配置：

```python
import tensorrt as trt
import time

class TRTProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._phases = {}      # phase_name -> {total, done, start}
        self._depth_map = {}   # phase_name -> 缩进层级
        self._should_cancel = False

    def phase_start(self, phase_name: str, parent_phase: str, num_steps: int):
        depth = self._depth_map.get(parent_phase, -1) + 1 if parent_phase else 0
        self._depth_map[phase_name] = depth
        self._phases[phase_name] = {"total": num_steps, "done": 0, "start": time.time()}
        # ... (打印缩进进度信息)

    def step_complete(self, phase_name: str, step: int) -> bool:
        if phase_name in self._phases:
            info = self._phases[phase_name]
            # 每 10% 或最后一步打印，避免刷屏
            # ... (计算 pct/elapsed/eta 并打印)
        return not self._should_cancel

    def phase_finish(self, phase_name: str):
        if phase_name in self._phases:
            # ... (打印完成耗时，清理 _phases 条目)
            del self._phases[phase_name]
```

---

## Python 实现：带超时取消

生产环境中，构建超时是常见需求。用 `threading.Timer` 实现软超时：

```python
import threading

class CancelableTRTMonitor(TRTProgressMonitor):
    def __init__(self, timeout_seconds: float | None = None):
        super().__init__()
        self._timer = None
        self._cancel_reason = ""
        if timeout_seconds:
            self._timer = threading.Timer(timeout_seconds, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()

    def _on_timeout(self):
        self._cancel_reason = f"超时（已等待 {self._timer.interval:.0f}s）"
        self._should_cancel = True
        print(f"\n⚠ 构建取消：{self._cancel_reason}")

    def cancel(self, reason: str = "用户取消"):
        self._cancel_reason = reason
        self._should_cancel = True
        if self._timer:
            self._timer.cancel()

    def cleanup(self):
        if self._timer:
            self._timer.cancel()
```

使用场景：微服务中构建超时后降级到 FP32 引擎：

```python
monitor = CancelableTRTMonitor(timeout_seconds=300)  # 5 分钟超时

try:
    engine = builder.build_serialized_network(network, config)
finally:
    monitor.cleanup()

if engine is None:
    print(f"构建失败或被取消：{monitor._cancel_reason}，降级到备用引擎")
    # ... 加载预构建的 FP32 engine ...
```

**注意**：`build_serialized_network` 返回 `None` 既可能是构建失败，也可能是被取消。通过 `monitor._cancel_reason` 区分两种情况。

---

## C++ 实现

C++ 场景下，多线程取消更为自然，使用 `std::atomic<bool>` 保证无锁安全：

```cpp
#include <NvInfer.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <unordered_map>

class TRTProgressMonitor : public nvinfer1::IProgressMonitor {
public:
    void phaseStart(char const* phaseName, char const* parentPhase,
                    int32_t nbSteps) noexcept override {
        PhaseInfo info{nbSteps, 0, std::chrono::steady_clock::now()};
        std::lock_guard<std::mutex> lk(mMutex);
        mPhases[phaseName] = info;
        std::cout << "▶ [" << phaseName << "] 开始，共 " << nbSteps << " 步\n";
    }

    bool stepComplete(char const* phaseName, int32_t step) noexcept override {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            auto it = mPhases.find(phaseName);
            if (it != mPhases.end()) {
                it->second.done = step;
                auto elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - it->second.start).count();
                int32_t total = it->second.total;
                if (total <= 10 || (step + 1) % std::max(1, total / 10) == 0) {
                    double pct = total > 0 ? (step + 1.0) / total * 100.0 : 0;
                    std::cout << "  " << step + 1 << "/" << total
                              << " (" << static_cast<int>(pct) << "%)"
                              << " | " << elapsed << "s\n";
                }
            }
        }
        return !mShouldCancel.load(std::memory_order_relaxed);
    }

    void phaseFinish(char const* phaseName) noexcept override {
        std::lock_guard<std::mutex> lk(mMutex);
        auto it = mPhases.find(phaseName);
        if (it != mPhases.end()) {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - it->second.start).count();
            std::cout << "✓ [" << phaseName << "] 完成，耗时 " << elapsed << "s\n";
            mPhases.erase(it);
        }
    }

    void cancel() noexcept { mShouldCancel.store(true); }

private:
    struct PhaseInfo { int32_t total, done; std::chrono::steady_clock::time_point start; };
    std::unordered_map<std::string, PhaseInfo> mPhases;
    std::mutex mMutex;
    std::atomic<bool> mShouldCancel{false};
};
```

在另一个线程中触发取消：

```cpp
TRTProgressMonitor monitor;
config->setProgressMonitor(&monitor);

// 后台线程：5 分钟后自动取消
std::thread watchdog([&monitor] {
    std::this_thread::sleep_for(std::chrono::minutes(5));
    monitor.cancel();
});

auto engine = builder->buildSerializedNetwork(*network, *config);
watchdog.detach();

if (!engine) {
    std::cerr << "构建失败或被取消\n";
}
```

---

## 常见陷阱

**1. `stepComplete` 中不能做耗时操作**

```python
# 错误：在回调中读磁盘/网络，会拖慢 tactic 搜索本身
def step_complete(self, phase_name, step):
    self.log_to_database(phase_name, step)  # ❌ I/O 阻塞
    return True

# 正确：累积到内存，构建完成后批量写
def step_complete(self, phase_name, step):
    self._events.append((phase_name, step, time.time()))  # ✅ 内存操作
    return True
```

**2. 取消后 `build_serialized_network` 返回 `None`，不代表引擎损坏**

取消是优雅退出，不会留下半成品文件。重新构建时如果 timing cache 已有部分数据，会略快于完全冷构建。

**3. `num_steps` 可能为 `-1`**

某些内部阶段的步骤数在开始时未知，会传入 `-1`。做进度百分比时需要判断：

```python
def phase_start(self, phase_name, parent_phase, num_steps):
    total = num_steps if num_steps > 0 else None  # None = 未知总量
    self._phases[phase_name] = {"total": total, ...}
```

---

## 与 Timing Cache 的协同

进度监控和 timing cache 是最佳搭档：

```python
# 热身构建：记录 timing cache + 监控进度
cache_path = "engine.trt_cache"
config.progress_monitor = TRTProgressMonitor()

# 首次构建：缓存为空，耗时最长
engine = builder.build_serialized_network(network, config)
with open(cache_path, "wb") as f:
    f.write(config.get_timing_cache().serialize())

# 后续构建：加载 cache，大幅缩短 TacticSearch 阶段
timing_cache = config.create_timing_cache(open(cache_path, "rb").read())
config.set_timing_cache(timing_cache, ignore_mismatch=False)
engine = builder.build_serialized_network(network, config)
# TacticSearch 阶段步骤数基本不变，但每步耗时从 ~50ms 降到 ~1ms
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 服务启动时构建引擎，需要 readiness probe | 离线批量构建脚本（直接等就行） |
| 构建超时需降级到备用路径 | 引擎已序列化到磁盘，直接反序列化 |
| 用户需要看到构建进度（CLI/Web UI） | 使用 trtexec 工具（已内置进度输出） |
| 需要在信号（SIGTERM）时优雅退出 | 构建时间 < 5s 的小模型 |

---

## 延伸阅读

- TensorRT 官方文档 `IProgressMonitor` 章节：重点读 `stepComplete` 的线程安全保证（回调在构建线程中执行，不跨线程）
- Timing Cache 最佳实践：`BuilderConfig::setTimingCache` 和 `getTimingCache` 的序列化格式
- TensorRT 强类型模式（`STRONGLY_TYPED`）：这是 LLM 构建耗时特别长的主要原因，理解它才能有效估算构建时间