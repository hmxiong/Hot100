python run_engine.py \
  --load_from /root/autodl-tmp/minimind/minimind-3 \
  --use_kv_cache 1 \
  --mode compare \
  --max_batch_size 8 \
  --num_requests 128 \
  --arrival_rate 16 \
  --arrival_process fixed \
  --baseline static \
  --use_pretrain_prompt 0 \
  --temperature 0 \
  --top_p 1.0

python compare_infer_paths.py --mode basic --load_from /root/autodl-tmp/minimind/minimind-3 --device cuda --max_new_tokens 256 --temperature 0 --use_chat_template 1 --show_outputs 0

python compare_infer_paths.py --mode basic_batch --load_from /root/autodl-tmp/minimind/minimind-3 --device cuda --max_new_tokens 256 --temperature 0 --use_chat_template 1 --show_outputs 0 --num_prompts 8

python compare_infer_paths.py --mode engine --load_from /root/autodl-tmp/minimind/minimind-3 --device cuda --max_new_tokens 256 --temperature 0 --use_chat_template 1 --show_outputs 0

python compare_infer_paths.py --mode engine_batch --load_from /root/autodl-tmp/minimind/minimind-3 --device cuda --max_new_tokens 256 --temperature 0 --use_chat_template 1 --show_outputs 0 --num_prompts 8

# serving_benchmark.py

## 作用

- `serving_benchmark.py` 专门用于模拟更接近真实在线 serving 的请求到达过程。
- 它和 `compare_infer_paths.py` 的定位不同：
- `compare_infer_paths.py` 主要用于正确性对齐、单条/静态 batch 对比、诊断分叉问题。
- `serving_benchmark.py` 主要用于性能评测，包括吞吐、时延、队列等待时间、显存峰值等指标。

## 支持的 benchmark 模式

- `hf_single`
  - 使用基础模型逐请求执行 `model.generate`
  - 近似模拟“完全不做批处理”的基线
- `hf_batch`
  - 使用基础模型做静态 batch `model.generate`
  - 用于观察 HuggingFace 原生 batch 在当前工作负载下的吞吐表现
- `engine_batch`
  - 使用当前引擎的 `generate_batch`
  - 用于观察当前静态 batch serving 实现的表现
- `engine_continuous`
  - 预留给 continuous batching 路径
  - 如果当前 `InferenceEngine` 还没有完整的 `add_request` / `step` / `get_request` 接口，脚本会自动跳过
- `all`
  - 按顺序运行上面全部模式

## 支持的核心参数

- `--mode`
  - 选择 benchmark 模式，可选：`all`、`hf_single`、`hf_batch`、`engine_batch`、`engine_continuous`
- `--num_requests`
  - 请求总数
- `--max_batch_size`
  - 最大 batch 大小
- `--arrival_rate`
  - 请求到达速率，单位是 `req/s`
- `--arrival_process`
  - 到达分布，可选：`fixed`、`poisson`
- `--max_new_tokens`
  - 每个请求最大生成长度
- `--temperature`
  - 采样温度；`0` 时近似 greedy，更适合做稳定性能对比
- `--top_p`
  - nucleus sampling 阈值；通常和 `temperature > 0` 一起使用
- `--repetition_penalty`
  - 重复惩罚
- `--use_pretrain_prompt`
  - 是否使用预训练风格 prompt 包装
- `--prompts`
  - 自定义 prompt 池，使用 `||` 分隔

## 推荐用法

### 1. 跑完整 benchmark

```bash
python serving_benchmark.py \
  --load_from /root/autodl-tmp/minimind/minimind-3 \
  --device cuda \
  --mode all \
  --num_requests 128 \
  --max_batch_size 8 \
  --arrival_rate 16 \
  --arrival_process poisson \
  --max_new_tokens 256 \
  --temperature 0 \
  --top_p 1.0 \
  --repetition_penalty 1.0 \
  --use_pretrain_prompt 0
```

### 2. 只看当前引擎静态 batch 路径

```bash
python serving_benchmark.py \
  --load_from /root/autodl-tmp/minimind/minimind-3 \
  --device cuda \
  --mode engine_batch \
  --num_requests 128 \
  --max_batch_size 8 \
  --arrival_rate 16 \
  --arrival_process poisson \
  --max_new_tokens 256 \
  --temperature 0 \
  --top_p 1.0
```

### 3. 只看 HuggingFace batch 基线

```bash
python serving_benchmark.py \
  --load_from /root/autodl-tmp/minimind/minimind-3 \
  --device cuda \
  --mode hf_batch \
  --num_requests 128 \
  --max_batch_size 8 \
  --arrival_rate 16 \
  --arrival_process poisson \
  --max_new_tokens 256 \
  --temperature 0 \
  --top_p 1.0
```

### 4. 评估采样场景

```bash
python serving_benchmark.py \
  --load_from /root/autodl-tmp/minimind/minimind-3 \
  --device cuda \
  --mode all \
  --num_requests 128 \
  --max_batch_size 8 \
  --arrival_rate 16 \
  --arrival_process poisson \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.0
```

## 输出指标说明

- `req_throughput`
  - 每秒完成多少个请求
- `token_throughput`
  - 每秒生成多少个 token
- `latency`
  - 从请求到达到请求完成的总延迟
- `queue_wait`
  - 从请求到达到被调度执行的等待时间
- `service`
  - 从开始执行到请求完成的时间
- `ttft`
  - first token time，当前对非流式路径是近似统计
- `output_tokens`
  - 每个请求输出 token 数的分布
- `tpot`
  - time per output token，近似衡量 decode 阶段平均每 token 耗时
- `memory`
  - CUDA 基线显存与峰值显存统计

## 结果解读建议

- 如果目标是建立稳定性能基线，优先使用：
  - `--temperature 0`
  - `--top_p 1.0`
- 如果目标是模拟真实采样场景，可以使用：
  - `--temperature 0.7`
  - `--top_p 0.9`
- 开启采样后，请求输出长度波动会更明显，因此：
  - 吞吐会抖动
  - 延迟分位会更不稳定
  - 更适合多跑几轮再取均值
- `engine_continuous` 当前更像“接口预留位”，等后续真正补齐 continuous batching 主线后，这个模式会变成后续优化的核心对比入口

# 当前阶段结论
# - basic.generate 与 engine.generate 已重新对齐
# - engine.generate_batch 与 basic.generate 在当前 8 条 prompt 测试集上输出对齐
# - 当前后续主线：从“可控教学型引擎”继续演进到“更专业的 serving engine”
