# 从零推理引擎：路线图

目标：从一个可跑的 HuggingFace baseline 出发，逐步演进到具备 vLLM / SGLang 核心能力的推理引擎原型。每一步都用“需求→原理→实现点→验证方式”记录，并保持可运行。

## 0. Baseline（你现在的 basic.py）
- 需求：单请求/串行对话，验证模型可用、采样参数正确。
- 现状：直接调用 `model.generate`，缺少对 prefill/decode、KV、批处理、调度的显式控制。
- 短板（后续每一项都会对应一个模块/实现）：
  - 无法做 continuous batching：generate 内部封装了循环与 batch 组织。
  - 无法观察/控制 KV cache：没有显式 `past_key_values` 管理。
  - 无法实现 chunked prefill：prefill 不可分块。
  - 无法实现 paged attention：KV 不可分页/不可回收。
  - 无法实现 radix attention：缺少前缀共享与 KV 复用机制。
  - 无法做 PD disaggregate：没有将 prefill/decode 变成可拆分的“服务边界”。

## 1. 最小可控引擎骨架（当前已落地）
- 需求：把 generate 的黑盒拆开，让“一步 decode”变成一个可调度的 primitive。
- 原理：自回归生成 = prefill（prompt 一次前向 + 建 KV）+ decode（每步 1 token 前向 + 追加 KV）。
- 实现点：
  - `InferenceEngine.step()`：每次对一批请求生成 1 token（可插队=continuous batching 的雏形）。
  - `SimpleScheduler`：先来先服务，控制 batch size。
  - `NaiveKVCache`：按 request 保存 `past_key_values`，先不做分页/复用。
- 验证：`run_engine.py` 同时提交多个 prompt，观察同一轮 step 内的批处理与吞吐变化。

## 2. Engine 接管生成循环（已落地）
- 需求：将生成控制流从 `model.generate` 移到引擎，显式区分 prefill 与 decode。
- 原理：先在引擎中完成单请求/等长 batch 的自回归控制，再逐步向更复杂的 batch 组织和调度扩展。
- 实现点：
  - `model_infer.py`：`forward()` 返回 hidden states，`prefill()` / `decode()` 返回 last-token logits。
  - `engine.generate()`：prefill 一次，之后循环 decode。
  - `run_model_infer.py`：最小验证入口，便于和 `basic.py` 对照。
- 验证：`run_model_infer.py` 可直接加载权重并输出生成文本。

## 3. Batch 状态与 KV 语义（已落地）
- 需求：让 batch 成为一等公民，并为后续变长 batch / continuous batching 做状态准备。
- 原理：batch 不只是一个 `(B, T)` 张量，还应包含每个 slot 的独立生命周期与 KV 生命周期。
- 实现点：
  - `_BatchGenerateState`：显式维护 `prompt_lens / generated_lens / finished / eos_hit / generated_token_ids`。
  - `Attention` 维护 `k_cache / v_cache / cache_lens`。
  - decode 通过 `active_mask` 控制活跃样本继续追加 KV，已结束样本冻结 slot。
- 验证：batch 内不同样本在不同时间结束时，其它样本仍可继续生成。

## 4. 变长 Batch（当前已落地第一版）
- 需求：不再要求 batch 内 prompt 等长。
- 原理：通过 `padding + attention_mask` 先打通正确性；性能优化（packed/varlen）延后。
- 实现点：
  - prefill 支持 `attention_mask`，并按每个样本最后一个有效 token 抽取 logits。
  - decode 使用 `cache_lens` 限制可见历史。
  - `run_model_infer.py` 的 batch 输入改为 `padding=True`。
- 验证：可直接输入不同长度 prompt 组成 batch 并完成推理。

## 4.5 近期执行顺序（浓缩版）
- 阶段 A：当前正确性基线已完成
  - `basic.generate`、`engine.generate`、`engine.generate_batch` 在当前测试集上已经重新对齐。
  - `compare_infer_paths.py` 现在既是性能基线入口，也是后续 serving engine 重构的正确性回归入口。
- 阶段 B：从“教学型引擎”过渡到“专业 serving engine”
  - 将当前 `InferenceEngine` 逐步拆成 `Request / Scheduler / ModelRunner / KVManager` 四层。
  - 明确职责边界：
    - `Request`：状态机、长度、优先级、取消/完成
    - `Scheduler`：挑选每一轮参与 prefill/decode 的请求集合
    - `ModelRunner`：只负责执行模型前向
    - `KVManager`：只负责 KV 分配、复用和释放
- 阶段 C：优先完善调度与资源编排
  - 先实现 continuous batching，再实现 chunked prefill。
  - 先把“请求动态进入、动态退出、prefill/decode 交织”这条主线打通。
- 阶段 D：再继续外提 KV 与 attention 形态
  - 将 KV cache 从模型内部进一步外提到独立管理层。
  - 再演进到 paged/block、packed/varlen、prefix cache。

## 5. Continuous Batching（下一阶段）
- 需求：请求随时到达；decode 批次每步都可能变化；吞吐优先。
- 原理：把“活跃请求集合”视作一个动态队列，每步选择可运行子集；调度策略影响吞吐与尾延迟。
- 实现点：
  - 请求状态：WAITING/RUNNING/FINISHED/CANCELLED/FAILED + 超时/最大长度约束。
  - 调度策略：FCFS → SRPT（剩余 token 最短优先）/ priority / fairness。
  - 预留：prefill 与 decode 分别调度（为 chunked prefill 做铺垫）。
- 验证：压测脚本（固定 prompt 长度/不同到达率），输出吞吐与 P50/P99 延迟。

### 5.1 当前已完成的前置里程碑
- 我们已经从“dynamic batch inference”继续推进到：
  - `generate_step()` 最小闭环
  - shrinking-batch scheduler
  - KV compaction
- 这一步的核心收获不是性能，而是验证了一件事：
  - running batch 可以在 decode 过程中真实收缩
  - shrinking 后 surviving 请求的历史 KV 可以被正确保留

### 5.2 这一步中实际踩到的问题
- 第一个版本的 shrinking-batch 虽然在调度器层面已经实现：
  - finished 样本会从 `running` 中移除
  - trace 中已经出现 `running 4->3->2->1`
- 但最终输出会分叉，原因不是调度器，而是模型内部 KV：
  - batch 收缩后，旧的 `k_cache/v_cache/cache_lens` 仍按旧 row 布局保存
  - 下一轮 decode 时 `check_kv_cache()` 因 batch size 不匹配而整批 reset cache
  - 剩余样本丢失历史上下文，生成开始退化和重复

### 5.3 当前解决方案
- 在 `model_infer.py` 中新增 `compact_kv_cache()`
- 在 `engine.step()` 中在 decode 后按 surviving `running` 顺序执行 KV compaction
- 验证结果：
  - `verify_shrinking_batch.py --per_prompt_max_new_tokens 4,8,16,32`
  - 能稳定看到 `running 4->3->2->1->0`
  - 最终输出与 `reference engine.generate_batch` 完全一致

### 5.4 因此下一步应是什么
- 当前最自然的下一步不再是“是否能 shrink”
- 而是：
  - shrinking 后是否能把 waiting 请求补进空位
  - prefill 新请求与 decode 旧请求如何共存调度
- 换句话说，下一步已经从：
  - `shrinking batch`
  进入：
  - `refillable shrinking batch`
  - 再进一步才是真正更完整的 continuous batching

### 5.5 当前 refillable shrinking-batch 的实际结论
- 我们已经完成了第一版 refill：
  - `max_batch_size` 真实受 `SchedulerConfig` 控制
  - shrinking 后 waiting 请求可以补进空位
  - 通过 `verify_refill_batch.py` 可以稳定看到：
    - `step 0`: `waiting 6->2`, `running 0->4`
    - `step 4`: `admitted_now=[4]`
    - `step 8`: `admitted_now=[5]`
    - `shrink_happened=True`
    - `refill_happened=True`
- 最终 6 条输出与 `reference engine.generate_batch` 完全一致

### 5.6 在 refill 阶段踩到的新坑
- 第一版 refill 曾尝试使用 mixed prefill：
  - 旧 running 的 pending token
  - 新 waiting 的完整 prompt
  - 混在同一轮 step 中执行
- 这在当前 attention 语义下不成立，导致：
  - 队列 trace 正确
  - 但 refill 后旧样本输出开始分叉
- 因此当前我们明确了一个重要结论：
  - 真正高性能的 mixed prefill / decode 共存调度，需要更成熟的 attention / runner 语义支持

### 5.7 当前解决方案与后续方向
- 当前采取的方案是：
  - `correctness-first rebuild-on-refill`
  - 即补位时先重建当前 running 集合的 KV，再继续 decode
- 这不是最终高性能实现，但它已经让 refill 行为在当前工程里闭环。
- 因此下一步更合理的顺序是：
  - 先接受 `rebuild-on-refill` 作为正确性基线
  - 再继续优化到真正的 mixed prefill / decode 调度
  - 最后再考虑 chunked prefill、packed/varlen、paged/block KV

## 6. Chunked Prefill（把 prefill 也拆成 step）
- 需求：长 prompt 会独占算力导致短请求排队；希望长 prompt 分块并与 decode 交织。
- 原理：prefill 可按 token chunk 分段计算并增量写入 KV（注意：需要模型支持 cache + 位置编码一致）。
- 实现点：
  - Request 增加 `prefill_offset` / `prefill_done`。
  - `Engine.step()` 支持两类 micro-batch：prefill-chunk 与 decode-one-token。
  - chunk 大小自适应（受显存与吞吐影响）。
- 验证：混合长短 prompt 的延迟曲线，观察尾延迟下降。

## 7. Paged Attention（KV 分页/块管理）
- 需求：KV cache 是显存大头；需要可回收、可碎片整理、可按 token 增长分配。
- 原理：把 KV 按固定 block/page 分配，逻辑连续序列映射到物理非连续块；attention 用 page table 做 gather。
- 实现点（教学原型可以分两阶段）：
  - 阶段 A：只做“paged KV allocator”，attention 仍用 HF 实现（先把 KV 结构变成 blocks）。
  - 阶段 B：实现 paged attention kernel（CUDA/Triton）+ page table gather。
- 验证：并发请求数拉高，观察 OOM 边界提升与显存曲线更平滑。

## 8. Radix Attention（前缀共享 / prefix cache）
- 需求：多轮对话、检索增强等场景大量共享前缀；希望复用前缀 KV，减少 prefill。
- 原理：用 radix tree（前缀树）存 prompt token 序列到 KV 的映射；最长公共前缀匹配后只 prefill 增量部分。
- 实现点：
  - PrefixCache：token 序列哈希 + radix tree 节点引用计数。
  - KV 引用计数与生命周期：请求结束后释放/衰减。
  - 与 paged KV 结合：prefix KV 对应 blocks 可共享。
- 验证：共享前缀的 N 个请求，prefill FLOPs 与 wall-time 显著下降。

## 9. PD Disaggregate（Prefill/Decode 分离）
- 需求：prefill 与 decode 的算子/并发特性不同；可用不同 GPU/不同实例；提升整体利用率。
- 原理：把推理拆成两个服务：Prefill Service 产出 KV handle；Decode Service 消费 KV handle 继续生成。
- 实现点（先做单机模拟，再做进程/网络）：
  - 抽象 KV handle（page table + blocks id）可序列化/可传递。
  - 队列协议：prefill 完成后将 handle 交给 decode worker。
  - 一致性：tokenization、rope scaling、eos 等必须一致。
- 验证：分别跑 prefill-heavy 与 decode-heavy workload，观察资源利用与吞吐改善。

## 10. 可选补充（强烈建议加入）
- 推理正确性与复现：固定随机种子、采样一致性、对齐 HF generate（小样本）。
- 监控与可观测：step 级 timeline（prefill/decode 时间、batch size、cache hit）。
- 基本工程化：配置文件、日志分级、基准脚本、最小单测。

## 11. 当前阶段结论
- 已完成：
  - `model_infer` 与 `model.py` 前向对齐
  - `engine.generate` 与 `basic.generate` 输出对齐
  - `engine.generate_batch` 与 `basic.generate` 在当前 8 条 prompt 测试集上输出对齐
  - `generate_step` 与 `engine.generate_batch` 在 deterministic 场景下输出对齐
  - shrinking-batch scheduler + KV compaction 已打通，并通过独立脚本验证
  - refillable shrinking-batch 已打通，并通过独立脚本验证
- 当前代表性结果：
  - `basic.generate`: `2022 tokens / 11.18s / 180.87 tokens/s`
  - `engine.generate_batch`: `2022 tokens / 3.89s / 519.31 tokens/s`
- 现阶段主问题：
  - 单条 `engine.generate` 仍慢于 `basic.generate`
  - 当前虽然已经支持 refill，但 refill 仍依赖 `rebuild-on-refill`
  - KV 仍是规则张量语义，离真正的专业 serving engine 还有调度与内存管理差距
