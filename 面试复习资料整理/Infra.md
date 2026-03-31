# AI Infra 面试复习（总览）

## 归档规则（后续我会自动执行）

- 算子/推理加速/CUDA/kernel/GEMM/attention/KV cache/continuous batching/推理引擎与调度 → 算子方向.md
- 训练框架/分布式训练/并行策略（DP/TP/PP）/ZeRO/FSDP/通信与重叠/数据与 checkpoint/集群调度与多租户 → 系统方向.md
- 跨域或系统设计题（端到端推理服务、训推一体、指标与容量规划等）→ Infra.md（本文件）
- 不确定归类时：先收录到 Infra.md，并在后续整理时移动

## 记录格式

- Q：一句话问清楚
  A：用要点回答（必要时给公式/对比/工程取舍）

## 显存 / 并行 / 训练系统

- Q：训练时显存主要花在哪？
   A：三大块：参数（weights）、梯度（grads）、优化器状态（optimizer states，Adam 典型是 m/v 两份）。再加上激活（activations，用于反向传播）、临时 buffer（通信/算子 workspace）、以及数据/缓存。工程上最先区分“常驻（参数/状态）”和“随 batch/seq 增长的激活”。
- Q：FP16/BF16 训练为什么通常还要 FP32 master weights？
   A：FP16 动态范围小，直接用 FP16 权重更新容易数值不稳定；常见做法是：前向/反向用 FP16/BF16（省带宽/提吞吐），权重更新用 FP32 master（稳定），再 cast 回低精度用于下一轮计算。
- Q：activation checkpointing（重计算）在省什么、代价是什么？
   A：省激活显存：不保存全部中间激活，只保存少量 checkpoint，反向时对缺失部分再做一次前向重算。代价是额外算力（增加 forward 次数），通常换显存、提升可训练 batch/seq 或模型规模。
- Q：DP/TP/PP 分别解决什么瓶颈？
   A：
  
  - DP（数据并行）：复制模型，分数据；主要瓶颈是梯度 AllReduce 通信，适合算力足、通信也强的场景。
  - TP（张量并行）：把单层矩阵切分到多卡；解决单卡放不下/单卡算不动，瓶颈是层内高频通信（如 AllGather/ReduceScatter）。
  - PP（流水并行）：按层切成 stage；解决模型太深放不下，瓶颈是 pipeline bubble、以及 micro-batch 调度复杂度。
- Q：3D 并行（DP+TP+PP）怎么配比？
   A：先按“能否放下”决定 TP/PP 的最小需求，再在剩余卡数上尽量提高 DP 以提升吞吐；同时约束通信：TP 通信最频繁，通常 TP 不宜过大；PP 要看层数与 micro-batch，bubble 可用更多 micro-batch 缓解。最终用 profiling 以 MFU/吞吐/显存为指标调参。
- Q：ZeRO/FSDP 的核心思想？ZeRO-1/2/3 区别？
   A：核心是“分片（shard）+按需聚合（gather）”，减少 DP 下的冗余常驻显存。
  
  - ZeRO-1：分片 optimizer states。
  - ZeRO-2：再分片 grads。
  - ZeRO-3：再分片 params（最省显存，但通信/复杂度最高）。
     工程取舍：越往后越省显存，但参数聚合更频繁、对通信和实现要求更高。
- Q：训练为什么会“算得很快但吞吐上不去”？
   A：常见是被通信/IO/调度卡住：AllReduce 占比高、梯度桶太小导致频繁启动、数据加载慢、CPU/GPU pipeline 不均衡、或者 kernel launch overhead 多。定位方法：分层 profile（step time 拆分 compute/comm/data），看 NCCL 时间、PCIe/NVLink 利用率、GPU idle。
- Q：如何估算一个 Transformer 的显存（面试口径）？
   A：给出分项与数量级：
  
  - 参数：N_params × dtype_bytes（若有 FP32 master 则再 +4B/param）。
  - 梯度：≈ N_params × dtype_bytes。
  - Adam 状态：≈ 2 × N_params × 4B（m/v FP32 常见）。
  - 激活：与 batch×seq×hidden×layers 成正比，长序列时常是主要增量。
     说清楚“常驻与可控项”：常驻用 ZeRO/FSDP/分片降，激活用 checkpoint/seq parallel/flash-attn 降。
## CUDA / 算子 / 性能优化

- Q：GEMM 优化你会从哪几层做？
   A：三层：
  
  - 算法/调度：tile 分块，block/warp mapping，减少全局访存。
  - 存储：shared memory cache、vectorized load、避免 bank conflict、合并访问（coalescing）。
  - 计算：寄存器复用、指令级并行、利用 Tensor Cores（WMMA/HMMA）、合理设置 pipeline stages。衡量指标：算力利用率、内存带宽利用率、occupancy、以及算子 Roofline 对齐。
- Q：怎么判断一个 kernel 是 compute-bound 还是 memory-bound？
   A：看算术强度（FLOPs/byte）与实际带宽/吞吐：如果带宽接近峰值但 FLOPs 上不去，多半 memory-bound；如果带宽不高但 SM 利用率高、接近峰值 FLOPs，则 compute-bound。用 profiler 看 DRAM throughput、SM occupancy、pipe utilization。
- Q：warp/block 划分策略与 SIMT 的关系？
   A：warp 是调度最小单位，分支发散会导致同 warp 串行执行不同路径；block 决定并发与资源分配（寄存器/共享内存）。目标是在不超寄存器/共享内存限制的前提下，提高 occupancy，同时减少 divergence 与全局内存访问次数。
- Q：reduce（归约）为什么要分层？
   A：避免原子冲突与全局同步开销。典型做法：block 内共享内存/warp shuffle 先局部归约，再用多 block 归约或两阶段 kernel。注意数值稳定（尤其 FP16）时用 FP32 accumulator。
- Q：数值稳定性常见坑（FP16/softmax）怎么处理？
   A：softmax 用 x - max(x) ，累加用 FP32，必要时用 log-sum-exp；混合精度训练配合 loss scaling；归约/方差等避免 catastrophic cancellation（如用 Welford 算法）。
## 推理系统 / KV Cache / 调度

- Q：LLM 推理为什么分 prefill 和 decode？它们的性能瓶颈分别是什么？
   A：prefill 处理整段 prompt，attention 计算量大、偏 compute；decode 每步生成一个 token，属于小 batch 的 반복执行，常被 memory/launch overhead/调度限制，且 KV cache 读写成为关键瓶颈。
- Q：KV cache 是什么？为什么会成为瓶颈？
   A：KV cache 存储每层 attention 的 key/value（按 seq 增长），decode 每步都要读历史 KV；它占显存大、访问频繁，且带宽敏感。优化方向：分页/块管理提高利用率（减少碎片）、压缩/量化、以及更好的 batch/调度提升复用。
- Q：连续批处理（continuous batching）解决什么问题？
   A：解决在线请求到达不齐、固定 batch 会造成等待与浪费的问题。通过动态把新请求插入正在运行的 batch，在保证时延约束下提升吞吐与 GPU 利用率。代价是调度复杂、需要 KV 管理与抢占/回滚机制支持。
- Q：吞吐 vs 首 token 延迟（TTFT）怎么权衡？
   A：吞吐倾向大 batch、更多合并；TTFT 倾向小 batch、优先 prompt。工程上常用：分队列（短/长请求）、prefill/decode 分离调度、设置最大等待窗口、对超长请求做拆分或降优先级。指标上同时看 QPS、P99 TTFT、tokens/s。
- Q：PD 分离（prefill/decode 分离）为什么有效？
   A：两阶段资源形态不同：prefill 更 compute-heavy、decode 更 memory/调度敏感；分离后可用不同并行策略/批策略/实例规格分别优化，并减少阶段间互相拖累（比如长 prompt 把 decode 队列堵住）。
- Q：DeepSeek-V3 的优化点有哪些（从训练成本与推理效率看）？
   A：核心目标是“用更少激活参数 + 更低精度与更强并行/通信优化”把训练与推理成本压下来：
  
  - 架构侧（兼顾性能与成本）：MoE（总参数很大，但每 token 只激活一小部分专家参数）+ MLA（降低注意力相关的推理开销与 KV 压力），在保持能力的同时控制每 token 的计算与显存带宽压力。（DeepSeek-V3 Technical Report）[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
  - MoE 负载均衡：提出“auxiliary-loss-free”的负载均衡策略，减少为了做均衡而引入的性能损失。（DeepSeek-V3 Technical Report）[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
  - 训练侧 FP8 混合精度：构建 FP8 mixed precision 训练框架，并在超大规模模型上验证可行性；同时在 MoE 训练里用 FP8 缓存/分发 activation、BF16 存低精度优化器状态，以降低显存与通信开销。（DeepSeek-V3 Technical Report）[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
  - 通信与计算重叠：针对 MoE all-to-all/dispatch 的通信延迟做 overlap，尽量把通信隐藏到计算里，提升端到端 MFU。（DeepSeek-V3 Technical Report）[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
  - 训练目标与推理加速联动：引入 Multi-Token Prediction（MTP）目标提升训练效果，并可在推理端用于 speculative decoding 加速；MTP 模块推理时可按需关闭，仅在需要加速时启用。（DeepSeek-V3 Technical Report）[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Q：推理服务如何做限流与隔离（多租户）？
   A：从入口到 GPU 全链路：请求级配额（QPS/tokens/s）、队列隔离（按租户/优先级）、并发上限、最大上下文限制、以及 GPU 侧调度的公平策略（如按租户轮转/权重）。配合熔断/降级：返回更小 max_tokens、降低采样复杂度、或转 CPU/低优实例（若业务允许）。
## 平台 / 集群 / 调度（训推一体、潮汐调度）

- Q：什么是训推一体/潮汐调度？核心难点？
   A：同一集群里白天推理高峰、夜间训练高峰，做弹性切换以提高 GPU 利用率。难点：推理 SLA 需要强保障（低延迟、稳定），训练任务可被抢占但要处理 checkpoint、数据一致性与恢复；还要做资源隔离、优先级、以及容量规划。
- Q：如果推理要抢占训练资源，怎么做“可恢复”的训练？
   A：训练侧支持周期性 checkpoint（模型/优化器/随机种子/数据进度），调度器支持 preemption 通知与优雅停机；数据侧支持可重放（确定性 shard/seed）；恢复时做一致性校验并从最近 checkpoint 继续。
- Q：如何提升 GPU 集群整体利用率？
   A：三条线：
  
  - 任务侧：提高单任务 MFU（并行策略、通信优化、数据 pipeline）。
  - 调度侧：减少碎片与空洞（bin packing、gang scheduling、亲和性/拓扑感知、抢占与回填）。
  - 系统侧：加速启动与分发（镜像、权重分发、缓存、RDMA/高速存储）。
- Q：分布式训练中“网络/存储”分别怎么影响？
   A：网络影响梯度同步与参数聚合（NCCL collective），拓扑（NVLink/IB/RDMA）决定通信代价；存储影响数据与 checkpoint 吞吐，数据加载不足会造成 GPU idle。常用手段：数据本地缓存、并行读取、异步预取、checkpoint 分层存储（本地+远端）。
   
## 系统设计题（高频模板）

- Q：设计一个云端大模型推理服务，你会怎么拆模块？
   A：入口网关（鉴权/限流/计费）→ 调度层（队列/优先级/批处理/路由）→ 执行层（模型实例、prefill/decode、并行与 KV 管理）→ 观测与运维（指标/日志/追踪/回放/灰度）→ 控制面（扩缩容、发布、配额、模型管理）。明确三类指标：吞吐、时延（P95/P99、TTFT）、成本（单 token 成本、GPU 利用率）。
- Q：怎么把一个推理服务从 1 卡扩到多卡/多机？
   A：先做数据并行实例扩容（水平扩），再做单实例内的张量并行/流水并行（垂直扩）；路由按模型版本与并行组绑定；KV cache 与会话粘性要处理（粘到同一并行组，或实现可迁移缓存但复杂度高）。同时要做健康检查、自动摘除、热升级策略。
