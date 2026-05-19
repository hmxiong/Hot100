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

## 2. Continuous Batching（完善调度与状态机）
- 需求：请求随时到达；decode 批次每步都可能变化；吞吐优先。
- 原理：把“活跃请求集合”视作一个动态队列，每步选择可运行子集；调度策略影响吞吐与尾延迟。
- 实现点：
  - 请求状态：WAITING/RUNNING/FINISHED/CANCELLED/FAILED + 超时/最大长度约束。
  - 调度策略：FCFS → SRPT（剩余 token 最短优先）/ priority / fairness。
  - 预留：prefill 与 decode 分别调度（为 chunked prefill 做铺垫）。
- 验证：压测脚本（固定 prompt 长度/不同到达率），输出吞吐与 P50/P99 延迟。

## 3. Chunked Prefill（把 prefill 也拆成 step）
- 需求：长 prompt 会独占算力导致短请求排队；希望长 prompt 分块并与 decode 交织。
- 原理：prefill 可按 token chunk 分段计算并增量写入 KV（注意：需要模型支持 cache + 位置编码一致）。
- 实现点：
  - Request 增加 `prefill_offset` / `prefill_done`。
  - `Engine.step()` 支持两类 micro-batch：prefill-chunk 与 decode-one-token。
  - chunk 大小自适应（受显存与吞吐影响）。
- 验证：混合长短 prompt 的延迟曲线，观察尾延迟下降。

## 4. Paged Attention（KV 分页/块管理）
- 需求：KV cache 是显存大头；需要可回收、可碎片整理、可按 token 增长分配。
- 原理：把 KV 按固定 block/page 分配，逻辑连续序列映射到物理非连续块；attention 用 page table 做 gather。
- 实现点（教学原型可以分两阶段）：
  - 阶段 A：只做“paged KV allocator”，attention 仍用 HF 实现（先把 KV 结构变成 blocks）。
  - 阶段 B：实现 paged attention kernel（CUDA/Triton）+ page table gather。
- 验证：并发请求数拉高，观察 OOM 边界提升与显存曲线更平滑。

## 5. Radix Attention（前缀共享 / prefix cache）
- 需求：多轮对话、检索增强等场景大量共享前缀；希望复用前缀 KV，减少 prefill。
- 原理：用 radix tree（前缀树）存 prompt token 序列到 KV 的映射；最长公共前缀匹配后只 prefill 增量部分。
- 实现点：
  - PrefixCache：token 序列哈希 + radix tree 节点引用计数。
  - KV 引用计数与生命周期：请求结束后释放/衰减。
  - 与 paged KV 结合：prefix KV 对应 blocks 可共享。
- 验证：共享前缀的 N 个请求，prefill FLOPs 与 wall-time 显著下降。

## 6. PD Disaggregate（Prefill/Decode 分离）
- 需求：prefill 与 decode 的算子/并发特性不同；可用不同 GPU/不同实例；提升整体利用率。
- 原理：把推理拆成两个服务：Prefill Service 产出 KV handle；Decode Service 消费 KV handle 继续生成。
- 实现点（先做单机模拟，再做进程/网络）：
  - 抽象 KV handle（page table + blocks id）可序列化/可传递。
  - 队列协议：prefill 完成后将 handle 交给 decode worker。
  - 一致性：tokenization、rope scaling、eos 等必须一致。
- 验证：分别跑 prefill-heavy 与 decode-heavy workload，观察资源利用与吞吐改善。

## 7. 可选补充（强烈建议加入）
- 推理正确性与复现：固定随机种子、采样一致性、对齐 HF generate（小样本）。
- 监控与可观测：step 级 timeline（prefill/decode 时间、batch size、cache hit）。
- 基本工程化：配置文件、日志分级、基准脚本、最小单测。

