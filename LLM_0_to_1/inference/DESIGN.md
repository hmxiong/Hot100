# 从零推理引擎：设计说明（当前实现版）

本设计文档描述 `inference/` 当前已经落地的推理原型。目标仍然不是性能最优，而是把 vLLM / SGLang 风格的关键抽象逐步拆出来，并确保每一步都可运行、可验证、可继续扩展。

## 1. 当前设计目标
- 把 `model.generate` 的黑盒生成循环拆开，让 engine 显式接管 `prefill` 与 `decode`。
- 将模型与引擎解耦：
  - 模型只负责前向与 logits 计算。
  - 引擎负责生成控制流、batch 状态、采样和 KV 生命周期。
- 先做“语义正确”的 batch inference：
  - 支持单条生成
  - 支持 batch 内独立状态
  - 支持 batch 语义 KV cache
  - 支持第一版变长 batch（padding + attention_mask）
- 在此基础上再向更高阶能力推进：
  - continuous batching
  - chunked prefill
  - paged attention
  - radix / prefix cache

## 2. 当前模块划分

### 2.1 模型层：`models/model_infer.py`
- 这是当前为了引擎演进而单独维护的“干净模型路径”。
- 设计原则：
  - `forward()` 返回 hidden states，而不是直接返回最终 logits。
  - `compute_logits()` 单独做 `lm_head` 投影。
  - `prefill()` / `decode()` 作为轻量包装，方便 engine 调用。
- 当前位置编码：
  - 使用显式 `positions` 输入，而不是从单一 `start_pos` 推断。
  - 接口形态向 nano-vllm / Qwen3 的 `rotary_emb(positions, q, k)` 靠齐。
- 当前 KV 语义：
  - 每层 attention 自己维护：
    - `k_cache`
    - `v_cache`
    - `cache_lens`
  - `cache_lens` 表示每个 batch slot 当前的有效历史长度。

### 2.2 引擎层：`engine/engine.py`
- `InferenceEngine` 是当前生成控制流的核心。
- 当前承担的职责：
  - 调用 `model.prefill()` / `model.decode()`
  - 维护 batch 状态
  - 组织采样
  - 处理 eos / finished / 生成长度
  - 管理 batch 语义下的 decode 循环
- 当前公开的生成接口：
  - `generate()`：单条或 `B=1` 路径
  - `generate_batch()`：batch 路径
  - 二者都收敛到 `_generate_batch_impl()`

### 2.3 验证与基准脚本
- `basic.py`
  - 作为原生 baseline，直接使用 `models/model.py` 中的 `generate()`
- `run_model_infer.py`
  - 作为当前 `model_infer + engine` 的最小验证入口
  - 支持单条和 batch 推理
- `compare_infer_paths.py`
  - 统一对比三条路径：
    - `basic.generate`
    - `engine.generate`
    - `engine.generate_batch`
  - 输出吞吐与显存指标

## 3. 当前生成路径

### 3.1 Prefill
- 输入：
  - `input_ids`
  - `positions`
  - 可选 `attention_mask`
- 行为：
  - 对完整 prompt 做一次前向
  - 建立第一版 KV 语义
  - 返回每个样本“最后一个有效 token”的 logits
- 对变长 batch 的处理：
  - 如果有 `attention_mask`，最后 logits 不是固定取 `[:, -1, :]`
  - 而是按每个样本最后一个有效 token 位置抽取

### 3.2 Decode
- 输入：
  - 当前 step 的 `next_token`
  - 对应的 `positions`
  - `active_mask`
- 行为：
  - 活跃样本继续追加真实 KV
  - 已结束样本冻结 batch slot，但维持规则张量形状
  - 返回当前 step 的 last-token logits

### 3.3 采样与停止条件
- engine 侧当前负责：
  - temperature
  - top_p
  - repetition_penalty
  - eos 停止
  - max_new_tokens 停止
- 每个样本独立采样、独立 finished

## 4. Batch 状态设计

### 4.1 `_BatchGenerateState`
当前 batch 生成已经显式维护以下状态：
- `prompt_lens`
- `generated_lens`
- `finished`
- `eos_hit`
- `generated_token_ids`

这些状态的意义是：
- 让 batch 中每个样本拥有独立生命周期
- 为后续 KV 压缩、连续调度、动态活跃集合提供基础

### 4.2 为什么要先做状态管理
如果没有显式 batch 状态，就很难继续做：
- finished 样本冻结
- batch 内独立 KV 语义
- 变长 batch
- continuous batching

所以当前设计先把“batch 是一个状态集合”立住，再去做更复杂的调度与内存管理。

## 5. 当前 KV Cache 设计

### 5.1 现状
当前 `model_infer.py` 中的 KV cache 仍然是“规则张量 + batch slot”语义，而不是 paged/block 语义。

每层 attention 维护：
- `k_cache: [B, T_kv, num_kv_heads, head_dim]`
- `v_cache: [B, T_kv, num_kv_heads, head_dim]`
- `cache_lens: [B]`

### 5.2 当前 decode 的语义
- batch 第 `b` 个样本始终对应第 `b` 个 slot
- 如果样本已 finished：
  - slot 保留
  - `cache_lens[b]` 不再增长
  - 不再追加真实 token 的 KV
- 如果样本仍活跃：
  - 追加当前 step 的真实 KV

### 5.3 为什么这样设计
- 这不是最终的高性能方案，但它有两个优点：
  - 保持 batch 张量规则，便于先验证正确性
  - 提前建立“每个 slot 有自己的有效上下文长度”这一关键语义
- 这正是后续演进到：
  - padding batch
  - packed / varlen
  - paged attention
  的桥梁

## 6. 变长 Batch 设计

### 6.1 当前方案
当前变长 batch 使用：
- `padding=True`
- `attention_mask`

也就是说：
- batch 输入可以有不同长度 prompt
- 通过 mask 控制 prefill 时哪些 token 有效
- 通过 `cache_lens` 控制 decode 时每个样本可见的历史长度

### 6.2 当前限制
虽然已经支持变长 batch，但这仍是“正确性优先”的第一版：
- 没有 packed / varlen kernel
- 没有 page table
- 没有动态压缩活跃样本
- finished 样本仍保留在 batch 中

这意味着：
- 语义已经打通
- 性能还没有接近 vLLM / SGLang

## 7. 和 vLLM / SGLang 的差距
当前原型距离 vLLM / SGLang 还差的核心点主要有：
- 没有 continuous batching 的成熟调度器
- 没有 chunked prefill
- 没有 paged KV / block allocator
- 没有 prefix cache / radix tree
- 没有 packed / varlen attention kernel

但已经具备了这些后续演进最需要的前置条件：
- 引擎接管生成循环
- batch 状态显式化
- KV cache 具备 batch slot 语义
- positions 显式化
- attention_mask / 变长 batch 第一版打通

## 8. 当前建议的演进顺序
1. 稳定当前变长 batch 正确性
2. 完善 continuous batching 状态机
3. 引入 chunked prefill
4. 将 KV cache 继续外提，演进到 paged / block 语义
5. 再做 prefix cache / radix attention

## 9. 当前设计的刻意限制
- 当前重点仍是“教学可控、路径清晰”，不是“一步到位复刻 vLLM”。
- 当前 KV 还在模型内部维护，没有完全外提到独立的 cache manager。
- 当前 decode 仍保持固定 batch slot，不做活跃样本压缩。
- 当前 attention 仍使用 PyTorch 的常规路径，没有引入专用 varlen/paged kernel。

这些限制都是刻意保留的，因为它们让每一步重构都更容易验证和对照。
