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

