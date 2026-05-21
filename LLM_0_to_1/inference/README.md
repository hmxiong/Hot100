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

python LLM_0_to_1/inference/compare_infer_paths.py \
  --load_from <模型目录> \
  --mode basic_batch \
  --min_equal_batch_size 4