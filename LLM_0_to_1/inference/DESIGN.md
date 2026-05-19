# 从零推理引擎：设计说明（最小骨架版）

本设计文档对应当前代码落地的最小引擎骨架，目标不是性能最优，而是把 vLLM / SGLang 的关键抽象“拆出来、跑起来、可扩展”。

## 1. 需求拆解（为什么要改）
- 从 `model.generate` 迁移到显式循环：否则无法插入调度、无法跨请求共享 batch、无法管理 KV。
- 从“单请求”迁移到“请求集合”：continuous batching 的本质是每一步都在一个动态集合上做 decode。
- 从“文本输入”迁移到“token 请求”：paged/radix/chunked/PD 都在 token/KV 这一层发生。

## 2. 核心对象（最小集合）
- Request：一次生成任务的最小状态机载体。
  - 固定部分：prompt 的 input_ids/attention_mask、prompt_len。
  - 变化部分：generated_ids、last_logits、status。
- Scheduler：每个 step 选择哪些请求进入 micro-batch。
- KVCache：按 request_id 保存/加载 `past_key_values`（后续替换为 paged / shared）。
- Engine：把“调度 + 组 batch + 一步前向 + 采样 + 更新状态”串起来。

## 3. 关键路径（decode one token）
1. Scheduler 从 WAITING 队列选择 batch（受 max_batch_size 控制）。
2. Engine 构造每个 request 的“完整序列”（prompt + 已生成 token）。
3. Model 前向：
   - `use_cache=True`，返回 `past_key_values`
   - 取 `logits[:, -1, :]` 作为下一 token 分布
4. Sampler（temperature/top_p/repetition_penalty）采样得到 token_id。
5. 更新 request：
   - append token
   - 命中 eos 或达到 max_new_tokens → FINISHED
6. 把 batch 的 `past_key_values` 写回 KVCache（按 request_id 分散）。

## 4. 现阶段的刻意限制（为了教学可控）
- KV 仍用 HF 的 `past_key_values` 形态，未做 blocks/page table。
- batched past_key_values 通过拼接构造，兼容性依赖模型的 cache 维度约定。
- attention_mask 目前用 padding mask，尚未做 position_ids/rope 细节处理。

## 5. 扩展点映射（对应后续路线）
- Chunked Prefill：Request 增加 prefill 进度；Engine.step 支持 prefill chunk 微批。
- Paged Attention：KVCache 替换为 block allocator + page table；Engine 不再拼接 past，而是传 page table 给 kernel。
- Radix Attention：KVCache 之上增加 PrefixCache（radix tree），把 prompt 的部分 KV 变成共享引用。
- PD Disaggregate：Engine 的“prefill 产物”变成可传递的 KV handle；decode worker 消费 handle。

