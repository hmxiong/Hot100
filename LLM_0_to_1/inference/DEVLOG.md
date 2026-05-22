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

### 修复：`basic.generate` 与 `engine.generate` 输出不对齐
- 改动
  - 对 `compare_infer_paths.py` 连续补充多轮诊断模式：
    - `diagnose`
    - `diagnose_forward`
    - `diagnose_layers`
    - `diagnose_attention`
  - 逐步比较：
    - 首次分叉 step
    - 指定 step 的 cache/full logits
    - embedding / layer hidden / attention 内部张量
  - 最终确认 `model_infer.py` 的 `from_pretrained()` 不应在加载阶段提前执行 `to(torch.bfloat16)`。
  - 调整为仅加载权重、迁移 device、`eval()`；具体 dtype 统一由外层 `init_infer_model()` 控制。
- 动机
  - `basic` 与 `engine` 早期存在 token 数不一致问题，表面像 cache 或 attention 语义偏差，但深入定位后发现更早在 embedding 就已经出现微小差异。
  - 真正根因是 `model_infer` 在加载后被提前转成 `bf16`，随后外层又转成 `fp16`，导致权重路径与 `basic model` 不一致。
- 验证
  - `diagnose_layers` 显示：
    - embedding 差异归零
    - 各层 attention / mlp / hidden 差异归零
    - final logits 差异归零
  - `diagnose_forward` 显示：
    - `basic-cache == infer-cache`
    - `basic.generate` 与 `engine.generate` 的 `per_prompt_tokens` 完全一致

### 结论更新：当前正确性已对齐，主线重新回到 serving engine 演进
- 当前状态
  - `basic.generate` 与 `engine.generate` 在相同 prompt 和 greedy 参数下已对齐。
  - `engine.generate_batch` 与 `basic.generate` 在当前 8 条 prompt 测试集上文本输出对齐。
  - 当前代表性结果：
    - `basic.generate`: `2022 tokens / 11.18s / 180.87 tokens/s`
    - `engine.generate_batch`: `2022 tokens / 3.89s / 519.31 tokens/s`
- 含义
  - “正确性排障”阶段可以收口。
  - 后续工作重点不再是“为什么生成不一致”，而是：
    - 为什么单条 `engine.generate` 仍慢于 `basic.generate`
    - 如何把当前静态 batch 优势扩展到更专业的 serving engine 能力

## 2026-05-21
### 新增：`generate_step` 最小闭环，打通 step-based 调度路径
- 改动
  - 基于当前 `InferenceEngine / SimpleScheduler / SimpleModelRunner`，补齐：
    - `prepare_prefill()`
    - `prepare_decode()`
    - `run_model()` 的 `prefill/decode` 两条分支
    - `scheduler.postprocess()`
    - `engine.generate_step()`
  - `Sampler` 增加 `temperature <= 0` 时的 greedy 路径，避免除零和采样异常。
- 动机
  - 后续要从静态 batch 走向 shrinking batch / continuous batching，必须先把“逐 step 运行 + 每步调度 + 每步回写状态”这条路径独立出来。
  - 这一步的目标不是性能，而是先把调度和模型执行拆开，建立最小可验证闭环。
- 验证
  - `verify_generate_step.py` 在 `temperature=0`、4 条 prompt、`max_new_tokens=64` 下验证：
    - `generate_step` 可完整跑通
    - 最终输出与 `reference engine.generate_batch` 完全一致

### 新增：shrinking-batch 验证脚本，专门观察 batch 收缩行为
- 改动
  - 新增 `verify_shrinking_batch.py`：
    - 逐 step 打印 `waiting/running/finished` 变化
    - 支持 `--per_prompt_max_new_tokens`
    - 打印最终文本输出与 `token_ids`
    - 最后对齐 `reference engine.generate_batch`
- 动机
  - 仅验证 `generate_step` 最终输出还不够，必须显式观察：
    - 哪个 step 开始有样本完成
    - `running` 是否真的从 `N -> N-1`
    - shrinking 后剩余样本是否仍然正确推进
- 验证
  - 通过 `--per_prompt_max_new_tokens 4,8,16,32` 稳定制造 `running 4->3->2->1->0` 的收缩过程

### 定位问题：shrinking 生效了，但 batch 收缩后输出开始分叉
- 现象
  - `verify_shrinking_batch.py` 的第一次运行中：
    - trace 已经显示 `running 4->3->2->1`
    - 但最终第 1 条样本从第 5 个 token 开始与 reference 分叉
    - 后续样本出现明显重复生成，如 `jjjjj`、`使用使用使用...`
- 诊断结论
  - 调度器本身没有问题，`finished` 样本确实被从 `running` 中移除了。
  - 真正的问题在模型内部 KV：
    - batch 收缩后，模型各层的 `k_cache/v_cache/cache_lens` 仍保留旧 batch row 布局
    - 下一轮 decode 时 `check_kv_cache()` 发现 `batch_size` 不匹配，直接 `reset_kv_cache()`
    - 之后 decode 只喂最后 1 个 token，但历史 KV 已丢失，导致生成轨迹漂移
- 含义
  - 这一步非常关键：它说明 shrinking-batch 的第一类核心问题不是“调度怎么删样本”，而是“删完样本后，模型内部状态怎么跟着一起重排”

### 修复：增加 KV compaction，让 shrinking-batch 闭环
- 改动
  - 在 `model_infer.py` 中新增：
    - `Attention.compact_kv_cache()`
    - `MiniMindModel.compact_kv_cache()`
    - `MiniMindForCausalLM.compact_kv_cache()`
  - 在 `engine.step()` 中新增：
    - decode 后根据当前 surviving `running` 顺序，对模型内部 KV cache 做 compact
- 动机
  - shrinking 后剩余样本必须继承原来的历史 KV，不能因为 batch 维度变化而整批 reset。
  - 这是从“固定 batch slot + inactive mask”走向“真实 shrinking batch”必须补上的那一步。
- 验证
  - 重新运行：
    - `python verify_shrinking_batch.py --load_from /root/autodl-tmp/minimind/minimind-3 --device cuda --max_batch_size 8 --num_prompts 4 --max_new_tokens 64 --per_prompt_max_new_tokens 4,8,16,32 --temperature 0 --top_p 1.0 --show_outputs 1`
  - 结果显示：
    - `shrink_happened=True`
    - trace 中稳定出现 `running 4->3->2->1->0`
    - 最终 4 条输出全部与 `reference engine.generate_batch` 完全一致

### 阶段性结论：从 dynamic batch 正式迈到 shrinking-batch scheduler
- 当前状态
  - 我们不再只是“支持 batch 内不同样本独立 finished”
  - 而是已经实现：
    - step-based `generate_step`
    - shrinking-batch scheduler
    - KV compaction
    - shrinking 过程的可观测验证脚本
- 含义
  - 这说明当前系统已经跨过了“静态 batch + 冻结 inactive slot”的阶段。
  - 下一阶段的重点可以顺理成章转向：
    - 空位补新请求
    - prefill/decode 共存调度
    - 真正的 continuous batching
