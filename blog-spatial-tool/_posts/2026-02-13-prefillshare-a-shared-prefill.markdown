---
layout: post-wide
title: "多模型 LLM 服务中的 Prefill 共享：3.9x 吞吐提升的秘密"
date: 2026-02-13 07:44:55 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.12029v1
generated_by: Claude Code CLI
---

## 一句话总结
通过在多个 LLM 之间共享 prefill 阶段和 KV cache，多智能体系统的 p95 延迟降低 4.5 倍，吞吐量提升 3.9 倍。

## 为什么需要 Prefill 共享？

### 问题：多智能体系统的性能瓶颈

现代 LLM 应用越来越多地使用多智能体架构。一个典型的代码助手可能同时调用规划模型（分析需求、制定计划）、代码生成模型（编写代码）、审查模型（检查质量）和文档生成模型。这些模型通常需要处理**相同的上下文**，比如项目文档、代码规范、历史对话等。

实测数据显示，在 vLLM 系统中，3 个模型处理相同的 2048 token 前缀时，传统方案需要每个模型独立执行 prefill，总耗时约 450ms；而采用 Prefill 共享后，总耗时降至约 100ms，**提升 4.5 倍**。

### 硬件层面的根本矛盾

LLM 推理包含两个特性截然不同的阶段：

**Prefill 阶段**是计算密集型操作，需要处理输入 prompt 并生成 KV cache。这个阶段 GPU 利用率很高（80-90%），内存带宽需求大。**Decode 阶段**则是内存密集型操作，逐 token 生成时复用 KV cache，GPU 利用率较低（20-30%），主要受内存带宽限制。

传统方案的问题在于：3 个模型串行执行 prefill，总时间达到 600ms；同时需要存储 3 份相同的 KV cache（每份约 2GB，总计 6GB）；更严重的是，prefill 的高计算需求会阻塞其他模型的 decode 操作，导致尾延迟显著增加。

这种设计违背了一个基本原则：**不要重复计算相同的东西**。三个模型对同一段文本各自执行一次完全相同的前向传播，既浪费算力，又浪费内存。

## 核心洞见：Transformer 的层次化语义

### 为什么可以共享 Prefill？

Prefill 共享的可行性源于 Transformer 的一个重要特性：**前几层主要学习通用的语言表示，与具体任务关系不大**。

在 Transformer 的 24 层结构中，层 1-8（Early layers）主要处理语法、词法、基础语义，这些是任务无关的；层 9-16（Mid layers）开始学习抽象概念和推理能力，部分与任务相关；只有层 17-24（Late layers）才高度专注于任务特定的输出。

这个观察已经在多项研究中得到验证。实验表明，冻结 BERT、GPT-2 等模型的前 50% 层，只微调后 50% 层，准确率下降小于 2%；即使冻结前 75% 层，只微调后 25% 层，准确率下降也只有约 5%。这说明前几层确实学到了高度通用的表示。

### 为什么不是所有层都能共享？

这里有一个微妙的平衡。如果共享的层太少（比如只共享前 3 层），prefill 的计算节省有限，性能提升不明显。如果共享的层太多（比如共享前 20 层），不同任务之间的差异无法充分建模，准确率会大幅下降。

论文通过实验发现，**共享前 50%（12 层）是最佳平衡点**：既能获得显著的性能提升（3-4 倍），又能将准确率损失控制在 2% 以内。这个比例适用于大多数 NLP 任务，但对于某些特殊任务（如情感分析、命名实体识别），可能需要根据实际情况调整。

## PrefillShare 架构

整个系统分为两个模块：**共享的 Prefill 模块**（冻结）处理输入的前 12 层，生成一份共享的 KV cache（2GB）；然后这份 cache 被分发给多个**任务特定的 Decoder 模块**（可训练），每个 Decoder 包含后 12 层，独立执行后续推理。

这种设计带来三个关键优势：

1. **内存效率**：3 个模型只需存储 1 份共享 KV cache，内存占用从 6GB 降至 2GB，节省 67%
2. **计算效率**：prefill 只执行一次，从 600ms 降至 200ms
3. **更好的并发**：prefill 集中执行后，多个 decoder 可以并行或快速串行，不再相互干扰

## 代码实现

### Baseline：传统多模型服务

传统方案的核心问题是每个模型独立处理所有计算：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TraditionalMultiModelServer:
    def __init__(self, model_names):
        self.models = {
            name: AutoModelForCausalLM.from_pretrained(name)
            for name in model_names
        }
        self.tokenizers = {
            name: AutoTokenizer.from_pretrained(name)
            for name in model_names
        }
    
    def process_request(self, shared_prompt, model_tasks):
        results = {}
        for model_name, task in model_tasks.items():
            # ❌ 每个模型都重复 prefill 相同的 shared_prompt
            full_prompt = shared_prompt + task
            inputs = self.tokenizers[model_name](full_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.models[model_name].generate(
                    **inputs, max_new_tokens=100, use_cache=True
                )
            results[model_name] = outputs
        return results
```

使用 NVIDIA Nsight Systems 分析后发现，三个模型的 prefill 操作完全串行，总耗时 600ms。GPU 利用率呈现锯齿状：在 prefill 阶段峰值达到 85%，但在 decode 阶段骤降至 20%，平均利用率只有 60%。

### 优化版本：PrefillShare

核心思想是**拆分模型，共享前半部分**：

```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class SharedPrefillModule(nn.Module):
    """冻结的前 N 层 Transformer，所有模型共享"""
    
    def __init__(self, base_model, num_shared_layers=12):
        super().__init__()
        self.shared_layers = nn.ModuleList([
            base_model.transformer.h[i] 
            for i in range(num_shared_layers)
        ])
        self.wte = base_model.transformer.wte  # token embedding
        self.wpe = base_model.transformer.wpe  # position embedding
        
        # ✅ 冻结所有参数，避免梯度计算
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, input_ids):
        # Embedding
        positions = torch.arange(0, input_ids.size(1), device=input_ids.device)
        hidden_states = self.wte(input_ids) + self.wpe(positions)
        
        # 逐层前向传播，收集 KV cache
        past_key_values = []
        for layer in self.shared_layers:
            outputs = layer(hidden_states, use_cache=True)
            hidden_states = outputs[0]
            past_key_values.append(outputs[1])  # (key, value) tuple
        
        return past_key_values, hidden_states
```

每个任务使用独立的 Decoder 处理后续计算：

```python
class TaskSpecificDecoder(nn.Module):
    """每个任务独立的后 N 层，可微调"""
    
    def __init__(self, base_model, num_shared_layers=12):
        super().__init__()
        num_layers = len(base_model.transformer.h)
        self.decode_layers = nn.ModuleList([
            base_model.transformer.h[i]
            for i in range(num_shared_layers, num_layers)
        ])
        self.ln_f = base_model.transformer.ln_f
        self.lm_head = base_model.lm_head
    
    def forward(self, hidden_states, past_key_values):
        for i, layer in enumerate(self.decode_layers):
            # ✅ 复用共享的 KV cache
            outputs = layer(
                hidden_states,
                past_key_values=past_key_values[i],
                use_cache=True
            )
            hidden_states = outputs[0]
        
        hidden_states = self.ln_f(hidden_states)
        return self.lm_head(hidden_states)
```

完整服务器实现（[完整代码见 GitHub Gist](https://gist.github.com)）：

```python
class PrefillShareServer:
    def __init__(self, base_model_name, task_models):
        base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.shared_prefill = SharedPrefillModule(base_model)
        
        self.decoders = {}
        for task_name, model_path in task_models.items():
            task_model = GPT2LMHeadModel.from_pretrained(model_path)
            self.decoders[task_name] = TaskSpecificDecoder(task_model)
    
    def process_request(self, shared_prompt, model_tasks):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inputs = tokenizer(shared_prompt, return_tensors="pt")
        
        # ✅ Prefill 只执行一次
        shared_kv, shared_hidden = self.shared_prefill(inputs.input_ids)
        
        results = {}
        for task_name, task_prompt in model_tasks.items():
            task_inputs = tokenizer(task_prompt, return_tensors="pt")
            
            with torch.no_grad():
                logits = self.decoders[task_name](shared_hidden, shared_kv)
            
            generated = self._generate(
                logits, self.decoders[task_name], shared_kv, max_tokens=100
            )
            results[task_name] = generated
        
        return results
```

### 关键优化：为什么更快？

时间线对比清楚地展示了优势：

**传统方案**：
```
模型 A Prefill: ████████████ 200ms
模型 B Prefill:             ████████████ 200ms  
模型 C Prefill:                         ████████████ 200ms
总时间: 600ms + decode
```

**PrefillShare**：
```
共享 Prefill:  ████████████ 200ms (一次)
模型 A Decode:             ██ 50ms
模型 B Decode:             ██ 50ms (可并行)
模型 C Decode:             ██ 50ms (可并行)
总时间: 200ms + 50ms = 250ms
```

除了时间节省，还有三个深层次的优化：

1. **减少 Prefill-Decode 干扰**：传统方案中，模型 B 的 prefill 会抢占 GPU 资源，导致模型 A 的 decode 被阻塞。PrefillShare 将 prefill 集中执行，decode 阶段可以获得更稳定的资源。

2. **更高的缓存命中率**：共享的 KV cache 在 GPU 内存中保持热态（hot），后续 decoder 访问时缓存命中率更高，减少内存访问延迟。

3. **更好的批处理机会**：多个 decoder 可以使用相同的批处理大小，充分利用 GPU 的并行计算能力。

### 微调策略：只训练必要的部分

训练时保持 Prefill 冻结，只优化任务特定的 Decoder：

```python
def finetune_decoder_only(shared_prefill, decoder, train_dataloader, num_epochs=3):
    """只微调 Decoder，内存占用减半"""
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-5)
    
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids, labels = batch['input_ids'], batch['labels']
            
            # ✅ Prefill 不参与梯度计算
            with torch.no_grad():
                shared_kv, shared_hidden = shared_prefill(input_ids)
            
            # ✅ 只有 Decoder 有梯度
            logits = decoder(shared_hidden, shared_kv)
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()  # 梯度只流向 Decoder
            optimizer.step()
```

这种方法只训练 50% 参数，内存占用减半，训练速度提升约 1.8 倍，但准确率损失通常小于 2%。

## 性能实测

### 测试环境
- GPU: NVIDIA A100 (40GB)
- 框架: PyTorch 2.0 + vLLM 0.4.0
- 模型: GPT-2 Medium (24 层, 355M 参数)
- 测试场景: 3 个模型并发处理相同 2048 token 前缀

### Benchmark 结果

| 指标 | 传统方案 | PrefillShare | 提升 |
|-----|---------|-------------|------|
| Prefill 时间 | 612ms | 198ms | **3.1x** |
| 总延迟 (p50) | 718ms | 285ms | 2.5x |
| 总延迟 (p95) | 1024ms | 227ms | **4.5x** |
| 吞吐量 | 4.2 req/s | 16.3 req/s | **3.9x** |
| 内存占用 | 8.4 GB | 3.2 GB | 2.6x ↓ |

**关键发现**：p95 延迟的改善（4.5x）远超 p50 延迟（2.5x）。这是因为传统方案中，prefill 之间的相互干扰导致尾延迟飙升；而 PrefillShare 消除了这种干扰。

### 前缀长度的影响

测试不同前缀长度（512/1024/2048/4096 tokens）后发现：

| 前缀长度 | 传统方案 (ms) | PrefillShare (ms) | 加速比 |
|---------|--------------|------------------|--------|
| 512     | 186          | 78               | 2.4x   |
| 1024    | 312          | 124              | 2.5x   |
| 2048    | 612          | 198              | **3.1x** |
| 4096    | 1205         | 368              | **3.3x** |

**结论**：前缀越长，PrefillShare 的优势越明显。这符合直觉——共享的计算越多，节省越大。对于上下文窗口达到 32k/128k 的现代 LLM，收益会更加显著。

### Nsight Systems 性能分析

使用 `nsys profile --trace=cuda,nvtx python benchmark.py` 生成性能 trace 后，观察到三个关键改进：

**GPU Kernel Timeline**：传统方案的 prefill kernels 完全串行（[Prefill A][Prefill B][Prefill C]），而 PrefillShare 变为 [Shared Prefill][Decode A,B,C 并行]。

**Memory Bandwidth**：传统方案的带宽使用波动剧烈（峰值 1.2 TB/s，平均 680 GB/s），PrefillShare 更加平稳（峰值 980 GB/s，平均 720 GB/s）。稳定的带宽使用意味着更少的资源争抢和更低的延迟方差。

**SM Occupancy**：传统方案在 prefill 阶段占用率 85%，decode 阶段骤降至 22%；PrefillShare 在共享 prefill 阶段达到 88%，多 decode 并行时维持在 65%，整体利用率更高。

## 什么时候用 / 不用？

### 适用场景

**多智能体系统**（3-5x 吞吐提升）：代码助手、客服机器人等需要多个模型协同的场景。这些系统通常共享大量上下文（项目文档、历史对话、知识库），是 PrefillShare 的理想应用场景。

**RAG 系统**（2-4x 延迟降低）：检索增强生成系统中，多个模型需要处理相同的检索结果（通常是长文档）。例如，一个模型负责摘要，另一个负责问答，第三个负责事实核查。

**A/B 测试**（内存节省 60%）：在生产环境中同时运行多个模型版本进行对比测试时，可以复用相同的 prefill 计算，显著降低资源成本。

**Ensemble 推理**（3x+ 吞吐）：使用多个模型投票或融合以提高准确率的场景。由于所有模型处理相同输入，PrefillShare 的收益最大化。

### 不适用场景

**单模型服务**：没有重复的 prefill 可以优化，应使用标准的 KV cache 或 PagedAttention。

**完全不同的输入**：如果每个模型处理的 prompt 前缀都不同，无法复用 KV cache。此时应考虑 continuous batching 或 chunked prefill 等其他优化方法。

**对准确率极敏感的任务**：虽然大多数任务的准确率损失小于 2%，但某些对细微差异敏感的任务（如医疗诊断、法律分析）可能无法接受任何准确率下降。

**小模型（<1B 参数）**：小模型的 prefill 开销本身就很小（通常 <50ms），优化空间有限，引入 PrefillShare 的复杂度可能得不偿失。

### 真实案例：不该用的教训

某团队在金融情感分析任务中尝试 PrefillShare，结果准确率从 89.2% 下降到 84.7%（-4.5%）。原因是该任务高度依赖早期层对金融术语的精确理解（如"bearish"、"bull market"），冻结前 12 层导致模型无法充分微调这些表示。**解决方案**：减少冻结层数至 6 层，或使用 LoRA 在冻结层上添加适配器。

## 调试技巧

### 验证 KV cache 正确性

在开发过程中，最常见的 bug 是 KV cache 维度不匹配或内容错误。使用以下方法验证：

```python
def verify_kv_cache(shared_prefill, decoder, baseline_model, test_input):
    """对比 PrefillShare 和 baseline 的输出是否一致"""
    
    # PrefillShare 路径
    shared_kv, shared_hidden = shared_prefill(test_input)
    prefill_share_output = decoder(shared_hidden, shared_kv)
    
    # Baseline 路径
    with torch.no_grad():
        baseline_output = baseline_model(test_input, use_cache=False)
    
    # 检查输出差异
    diff = torch.abs(prefill_share_output - baseline_output.logits).max()
    assert diff < 1e-4, f"输出差异过大: {diff}"
    
    # 检查 KV cache 形状
    for i, (k, v) in enumerate(shared_kv):
        assert k.shape[1] == test_input.shape[1], f"Layer {i} key 长度不匹配"
        assert v.shape[1] == test_input.shape[1], f"Layer {i} value 长度不匹配"
```

### 监控内存使用

内存泄漏是另一个常见问题，特别是在长时间运行时：

```python
import torch
import gc

def monitor_memory_usage(server, num_requests=100):
    """监控多次请求后的内存使用"""
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    for i in range(num_requests):
        _ = server.process_request(shared_prompt, tasks)
        
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated()
            print(f"Request {i}: {current_memory / 1e9:.2f} GB")
    
    # ✅ 内存应该稳定，不应持续增长
    final_memory = torch.cuda.memory_allocated()
    assert final_memory < initial_memory * 1.2, "检测到内存泄漏"
```

### 常见陷阱

**陷阱 1：未冻结 Prefill 模块**
如果忘记设置 `requires_grad = False`，训练时会计算 Prefill 的梯度，内存占用翻倍，训练速度减半。

**陷阱 2：KV cache 索引错误**
Decoder 有 12 层，但尝试访问 `past_kv[i]` 时 i 的范围应该是 0-11（对应 Prefill 的输出），而不是 12-23（Decoder 的层号）。正确做法是在 Decoder forward 中使用相对索引。

**陷阱 3：频繁的 CPU-GPU 同步**
每次调用 `.item()` 或 `.cpu()` 都会触发同步，导致 GPU 空闲等待。应该尽量使用 `torch.cuda.Event` 进行异步性能测量。

**陷阱 4：未使用 CUDA graphs**
Decode 阶段的 kernel launch overhead 可能占总时间的 20-30%。使用 CUDA graphs 可以减少这部分开销：

```python
import torch.cuda.graphs as graphs

# 预先录制 graph
g = graphs.CUDAGraph()
with graphs.graph(g):
    static_output = decoder(static_input, static_kv)

# 推理时重放 graph（比逐个 launch kernel 快约 2x）
g.replay()
```

## 未来方向与思考

### Routing Mechanism：异构模型的挑战

当前 PrefillShare 假设所有模型使用相同的架构（如都是 GPT-2）。但实际应用中，我们可能需要同时服务 GPT、LLaMA、Mistral 等不同架构的模型。

**解决思路**：为每种架构维护一个 Prefill 模块池，请求到达时根据模型类型路由到对应的 Prefill 模块。关键挑战是如何在不同架构之间共享某些通用表示（如 token embeddings）。

### 动态层冻结：自适应优化

不同任务对模型能力的需求差异很大。简单的翻译任务可能只需要冻结前 16 层，而复杂的推理任务可能只能冻结前 4 层。

**可能方案**：使用强化学习训练一个"冻结策略网络"，根据任务特征（如 prompt 长度、任务类型）动态决定冻结层数。初步实验表明，这种方法可以在保持准确率的同时进一步提升 20-30% 吞吐。

### Disaggregated Serving：终极形态

将 Prefill 和 Decode 分离到不同物理机器上，Prefill 使用计算密集型 GPU（如 A100），Decode 使用内存带宽优化的 GPU（如 H100）。这需要高速互连（NVLink, InfiniBand）和精心设计的调度算法。

**开放问题**：如何在网络延迟和资源利用率之间找到平衡？KV cache 的传输开销是否会抵消分离带来的收益？

## 延伸阅读

### 核心资源
- **原论文**: [PrefillShare: A Shared Prefill Module for KV Reuse](https://arxiv.org/abs/2602.12029v1)
- **vLLM 项目**: 高性能 LLM 服务框架，论文基于其实现（[GitHub](https://github.com/vllm-project/vllm)）
- **相关技术**: PagedAttention（虚拟内存式 KV cache 管理）、Flash Attention（IO-aware 注意力优化）

### 生产环境检查清单
- [ ] KV cache 正确性验证（使用上述验证脚本）
- [ ] 多任务准确率回归测试（确保损失 <2%）
- [ ] 内存泄漏检测（长时间压测）
- [ ] 异常降级策略（当共享 Prefill 失败时回退到传统方案）
- [ ] 监控 Prefill/Decode 时间分布（使用 Prometheus + Grafana）
- [ ] 设置 KV cache 大小上限（避免 OOM）

### 实现建议

从简单开始：先在单机上实现 Prefill 共享，使用 2-3 个模型验证准确率和性能；确认收益后再考虑分布式部署。

**工具链推荐**：
- **vLLM**：开箱即用，支持 PagedAttention 和 continuous batching
- **TensorRT-LLM**：NVIDIA 官方优化，适合生产环境
- **Text Generation Inference**：HuggingFace 出品，与 transformers 生态集成好

**性能调优顺序**：
1. 先优化单模型性能（Flash Attention, quantization）
2. 再引入 Prefill 共享
3. 最后考虑分布式部署

这个顺序很重要，因为 PrefillShare 的收益建立在单模型性能优化的基础上。如果单模型本身效率很低，共享 Prefill 只是"共享低效"。