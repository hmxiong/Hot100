完成minimind3和minimind3-moe模型的基础训练
在minimind的训练代码基础上叠加tinyllama的训练过程，尝试第一次将模型参数和训练数据扩充到0.5B的参数级别
首次尝试在8卡4090上训练348M参数量的模型，max seq len=760，结果显示OOM，将max seq len调整到340+开启flash attention成果降低显存占用
训练失败，8卡训练出现nccl错误，loss下降速度过快，最终决定：放弃继续scaling 模型参数，但是需要进行FlashAtttention的部分优化学习

一些理论知识的补充：
## FSDP 学习与实践要点（MiniMind）

### 1. FSDP 是什么
- FSDP（Fully Sharded Data Parallel）用于训练阶段的显存优化并行方案。
- 核心思想：**每张卡模型结构相同，但参数/梯度/优化器状态按 shard 分片存储**。
- 与 PP（Pipeline Parallel）不同：FSDP 不是按层把模型切到不同卡上。

### 2. FSDP 的计算与通信流程
- 前向到某层前：通过 `all-gather` 聚合该层完整参数（临时）。
- 前向计算后：可释放该层完整参数，仅保留本地 shard。
- 反向时：通过 `reduce-scatter` 规约并分发梯度分片。
- 因此 FSDP 的主要开销是通信（all-gather / reduce-scatter），不是 PP 的层间串行等待。

### 3. 三个核心通信原语
- `all-reduce`：规约后广播，所有卡拿到同样完整结果（DDP 常用）。
- `all-gather`：各卡提交分片，所有卡拿到完整张量（FSDP 前向常用）。
- `reduce-scatter`：先规约再切片，每卡只拿结果的一片（FSDP 反向常用）。

### 4. 本项目里的 FSDP 关键实现点
- 新增脚本：`trainer/train_pretrain_fsdp.py`。
- 使用 `FSDP(...)` 包裹模型并配置 `auto_wrap_policy`（按 `MiniMindBlock`）。
- 支持分片策略：`--sharding full|grad|hybrid|none`。
- 训练中使用 `no_sync` 做梯度累积，累积窗口末尾强制同步并更新参数。

### 5. 已修复的关键问题
- 修复 `auto_wrap_policy` 传参方式（避免运行时错误）。
- 修复梯度累积尾步同步问题（避免多卡参数漂移）。
- 优化续训加载流程，避免各 rank 读取完整模型造成内存/IO 压力。

### 6. 调参经验（FSDP 场景）
- 全局 batch：`global_batch = batch_size * accumulation_steps * gpu_num`。
- OOM 优先调参顺序：
  1. 降 `batch_size`
  2. 增 `accumulation_steps` 维持 global batch
  3. 再降 `max_seq_len`
- `max_seq_len` 主要影响激活显存和计算开销；参数分片显存主要由模型规模和分片策略决定。
- `full` 更省显存但可能更慢；显存允许时 `grad` 常更快。

### 7. 结论
- FSDP 解决的是“单卡放不下完整模型”的训练显存问题。
- 代价是通信开销增加，需要在分片策略、batch、accumulation、seq_len 之间做平衡。
- 推理场景通常不以 FSDP 为首选（更常见 TP / vLLM 等方案）。