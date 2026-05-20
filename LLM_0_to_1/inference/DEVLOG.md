# 开发记录（DEVLOG）

记录原则：每次只记录“改了什么 + 为什么 + 怎么验”，确保你回看源码时能沿着动机复现 vLLM/SGLang 的设计决策。

## 2026-05-19
### 新增：最小推理引擎骨架（拆出 generate）
- 改动
  - 新增 `inference/engine/`：Request / Scheduler / Sampler / KVCache / Engine。
  - 新增 `inference/run_engine.py`：并发提交多个 prompt，通过 step 循环展示 continuous batching 雏形。
  - 新增文档：ROADMAP.md、DESIGN.md。
- 动机
  - `model.generate` 是黑盒：无法在“每一步 decode”之间插入调度与 cache 管理，自然也无法实现 continuous batching / chunked prefill / paged attention / radix attention / PD 分离。
  - 需要把推理过程降解成可组合的 primitive：prefill/decode、batch 组织、KV 生命周期。
- 验证
  - 直接运行 `python -m LLM_0_to_1.inference.run_engine --load_from <模型路径>`，观察多请求同时推进并输出总 tokens/s。

### 重构：基于 MiniMind 的 KVCache 形态整理 Engine.step（去除 legacy/DynamicCache 兼容分支）
- 改动
  - `InferenceEngine.step` 改为自管理 running/waiting 队列，不再依赖 scheduler。
  - 推理一步拆成两条确定路径：prefill（无 KV）与 decode（有 KV），并按长度分组做 micro-batch。
  - KVCache 直接存 MiniMind 的 `past_key_values: list[(k,v)]`（每个 request 保存 batch=1 的切片）。
- 动机
  - MiniMind 的 `past_key_values` 形态稳定，便于你后续在 KV 上做 paged/radix/chunked 的实验。
  - MiniMind 当前实现里 `start_pos` 从 `past_key_values[0][0].shape[1]` 推断，batch 内必须同长度；因此用分组 micro-batch 保证正确性。
- 验证
  - `python -m LLM_0_to_1.inference.run_engine --load_from <MiniMind/HF目录> --use_kv_cache 1`

## 2026-05-20
### 新增：`model_infer.py` 干净模型路径（为引擎接管生成做准备）
- 改动
  - 新增/重构 `inference/models/model_infer.py`，将 `forward()` 固定为返回 hidden states，`compute_logits()` 单独负责投影到词表。
  - 补齐 `from_pretrained()`，可直接从 `/root/autodl-tmp/minimind/minimind-3` 加载配置与权重。
  - 将 RoPE 改为显式 `positions` 输入，接口形态向 nano-vllm / Qwen3 的写法靠齐。
- 动机
  - 把模型本体和生成控制流解耦，后续才能将 prefill / decode / KVCache 管理彻底移到引擎侧。
  - `positions` 显式化后，更容易继续演进到 packed / varlen / paged attention。
- 验证
  - `python run_model_infer.py --load_from /root/autodl-tmp/minimind/minimind-3 --device cuda`

### 重构：在 `InferenceEngine` 中实现基础自回归生成
- 改动
  - 在 `inference/engine/engine.py` 中新增 `generate()` / `generate_batch()`。
  - 生成流程改为：`prefill` 一次拿首个 logits，之后循环调用 `decode` 完成自回归生成。
  - 新增 `run_model_infer.py` 作为最小验证脚本，支持单条与 batch 推理。
- 动机
  - 后续需要在 decode 循环里插入 batch 状态管理、KV 生命周期管理、性能统计，必须先把生成控制权从 `model.generate` 收回到 engine。
- 验证
  - `python run_model_infer.py --load_from /root/autodl-tmp/minimind/minimind-3 --device cuda --temperature 0`

### 新增：batch 内独立状态管理
- 改动
  - 在 `InferenceEngine` 中新增 `_BatchGenerateState`，显式记录：
    - `prompt_lens`
    - `generated_lens`
    - `finished`
    - `eos_hit`
    - `generated_token_ids`
  - 将 `generate()` 与 `generate_batch()` 统一收敛到 `_generate_batch_impl()`。
- 动机
  - 后续要支持 batch 语义 KV、变长 batch、动态压缩活跃样本，必须先把每个 batch slot 的生命周期抽象清楚。
- 验证
  - `run_model_infer.py` 中单条与 batch 推理共享同一套生成逻辑，输出结果一致性更容易对比。

### 升级：KV cache 从“整批张量”变为“带 batch slot 语义的缓存”
- 改动
  - `model_infer.py` 的 attention 层新增：
    - `k_cache`
    - `v_cache`
    - `cache_lens`
  - decode 路径新增 `active_mask`，活跃样本追加真实 KV，已结束样本冻结 slot 并保持 batch 形状稳定。
  - `check_kv_cache()` 负责校验 batch_size / device / dtype / kv 维度是否匹配。
- 动机
  - 这是从“单条语义 KV”演进到“batch 语义 KV”的关键步骤。
  - 后续做变长 batch、packed、paged attention 时，都需要先明确每个 slot 的有效上下文长度。
- 验证
  - `engine.generate_batch()` 在 batch 内样本提前结束时，其他样本仍能继续 decode，且 cache 语义不混乱。

### 升级：支持变长 batch（padding + attention_mask）
- 改动
  - `model_infer.py` 的 `Attention.forward()`、`MiniMindModel.forward()`、`MiniMindForCausalLM.prefill()/decode()` 全面接入 `attention_mask`。
  - prefill 改为按每个样本最后一个有效 token 位置取 logits，而不是固定取 `[:, -1, :]`。
  - `InferenceEngine.generate()/generate_batch()` 支持直接接收 padding 后的 `input_ids + attention_mask`。
  - `run_model_infer.py` 的 batch 输入改为 `tokenizer(..., padding=True)`，不再要求 prompt 等长。
- 动机
  - 这是从“教学型等长 batch”迈向“可用 batch inference”的第一步。
  - 先打通 padding + mask 语义，再考虑 packed / varlen 的性能优化。
- 验证
  - `run_model_infer.py` 中手动 batch 输入不同长度 prompt，能够正常推理并输出结果。

### 新增：三路径对比脚本
- 改动
  - 新增 `inference/compare_infer_paths.py`，对比以下三种路径：
    - `basic.generate`
    - `model_infer + engine.generate`
    - `model_infer + engine.generate_batch`
  - 统一使用相同 prompt、相同生成参数，输出：
    - 总 tokens
    - wall time
    - tokens/s
    - CUDA 峰值显存
- 动机
  - 后续每轮迭代都需要有稳定的回归对比入口，观察“正确性是否回退、吞吐是否提升、显存是否异常”。
- 验证
  - `python compare_infer_paths.py --load_from /root/autodl-tmp/minimind/minimind-3 --device cuda --temperature 0`
