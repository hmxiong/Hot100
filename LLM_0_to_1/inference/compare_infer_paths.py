import argparse
import gc
import math
import time
import warnings

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from engine.engine import EngineConfig, InferenceEngine
from engine.sampler import sample_next_token
from models.model import MiniMindForCausalLM as BasicMiniMindForCausalLM
from models.model_infer import MiniMindForCausalLM as InferMiniMindForCausalLM

warnings.filterwarnings("ignore")


def default_prompts() -> list[str]:
    return [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        '解释一下"光合作用"的基本过程',
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
    ]


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    name = str(dtype_name).lower()
    if name == "auto":
        return torch.float16 if str(device).startswith("cuda") else torch.float32
    if name in ("float16", "fp16"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype_name}")


def maybe_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_cuda_stats(device: str) -> tuple[float, float]:
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return 0.0, 0.0
    torch.cuda.empty_cache()
    gc.collect()
    baseline_alloc = torch.cuda.memory_allocated(device) / 1024**2
    baseline_reserved = torch.cuda.memory_reserved(device) / 1024**2
    torch.cuda.reset_peak_memory_stats(device)
    return baseline_alloc, baseline_reserved


def collect_cuda_peaks(device: str) -> tuple[float, float]:
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return 0.0, 0.0
    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    return peak_alloc, peak_reserved


def build_text(tokenizer, prompt: str, use_chat_template: bool, open_thinking: bool) -> str:
    if use_chat_template and callable(getattr(tokenizer, "apply_chat_template", None)):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=bool(open_thinking),
        )
    return (getattr(tokenizer, "bos_token", None) or "") + prompt


def select_equal_length_texts(
    tokenizer,
    prompts: list[str],
    texts: list[str],
    min_batch_size: int = 2,
) -> tuple[list[str], list[str], int]:
    if len(prompts) != len(texts):
        raise ValueError("prompts 与 texts 长度不一致")

    buckets: dict[int, list[int]] = {}
    for i, text in enumerate(texts):
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        token_len = int(encoded["input_ids"].shape[1])
        buckets.setdefault(token_len, []).append(i)

    candidates = [(token_len, idxs) for token_len, idxs in buckets.items() if len(idxs) >= int(min_batch_size)]
    if not candidates:
        bucket_sizes = {token_len: len(idxs) for token_len, idxs in buckets.items()}
        raise ValueError(
            f"找不到满足最小 batch 大小的等长 prompt 子集，min_batch_size={min_batch_size}, buckets={bucket_sizes}"
        )

    best_len, best_indices = sorted(candidates, key=lambda x: (-len(x[1]), -x[0], x[0]))[0]
    selected_prompts = [prompts[i] for i in best_indices]
    selected_texts = [texts[i] for i in best_indices]
    return selected_prompts, selected_texts, int(best_len)


def init_basic_model(load_from: str, device: str, dtype: torch.dtype):
    model = BasicMiniMindForCausalLM.from_pretrained(load_from, trust_remote_code=True)
    model = model.eval().to(device=device, dtype=dtype)
    return model


def init_infer_model(load_from: str, device: str, dtype: torch.dtype):
    model = InferMiniMindForCausalLM.from_pretrained(load_from, device=device)
    model = model.eval().to(device=device, dtype=dtype)
    return model


def token_to_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def collect_top_token_info(tokenizer, logits: torch.Tensor, top_k: int) -> list[dict]:
    k = max(1, min(int(top_k), int(logits.shape[-1])))
    values, indices = torch.topk(logits, k=k)
    summary: list[dict] = []
    for token_id, value in zip(indices.tolist(), values.tolist()):
        summary.append(
            {
                "token_id": int(token_id),
                "token_text": token_to_text(tokenizer, int(token_id)),
                "logit": float(value),
            }
        )
    return summary


def summarize_top_tokens(tokenizer, logits: torch.Tensor, top_k: int) -> list[str]:
    return [
        f"{item['token_id']}:{item['token_text']}:{item['logit']:.4f}"
        for item in collect_top_token_info(tokenizer, logits, top_k)
    ]


def eos_rank(logits: torch.Tensor, eos_token_id: int | None) -> int | None:
    if eos_token_id is None:
        return None
    eos_logit = logits[int(eos_token_id)]
    return int((logits > eos_logit).sum().item()) + 1


def trace_basic_single(
    model,
    tokenizer,
    text: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    top_k_to_show: int,
) -> list[dict]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    eos_token_id = tokenizer.eos_token_id
    do_sample = bool(temperature > 0)
    basic_top_p = float(top_p) if temperature is not None and temperature > 0 else 1.0
    basic_top_k = 50 if temperature is not None and temperature > 0 else 0
    past_key_values = None
    finished = False
    trace: list[dict] = []

    for step in range(int(max_new_tokens)):
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        outputs = model.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :].clone()
        if temperature is not None and temperature > 0 and temperature != 1.0:
            logits = logits / float(temperature)
        if repetition_penalty != 1.0:
            logits[0, torch.unique(input_ids[0])] /= repetition_penalty
        if basic_top_k > 0:
            threshold = torch.topk(logits, basic_top_k)[0][..., -1, None]
            logits[logits < threshold] = -float("inf")
        if basic_top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > basic_top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            logits[mask.scatter(1, sorted_indices, mask)] = -float("inf")
        logits_1d = logits[0]
        next_token = (
            torch.multinomial(torch.softmax(logits_1d, dim=-1), num_samples=1)
            if do_sample
            else torch.argmax(logits_1d, dim=-1, keepdim=True)
        )
        token_id = int(next_token.item())
        trace.append(
            {
                "step": step,
                "token_id": token_id,
                "token_text": token_to_text(tokenizer, token_id),
                "eos_logit": float(logits_1d[int(eos_token_id)].item()) if eos_token_id is not None else None,
                "eos_rank": eos_rank(logits_1d, eos_token_id),
                "top_token_info": collect_top_token_info(tokenizer, logits_1d, top_k_to_show),
                "top_tokens": summarize_top_tokens(tokenizer, logits_1d, top_k_to_show),
            }
        )
        input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
        past_key_values = outputs.past_key_values
        if eos_token_id is not None and token_id == int(eos_token_id):
            finished = True
        if finished:
            break

    return trace


def trace_engine_single(
    model,
    tokenizer,
    text: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    top_k_to_show: int,
) -> list[dict]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    eos_token_id = tokenizer.eos_token_id
    generated_token_ids: list[int] = []
    trace: list[dict] = []
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()

    prompt_len = int(input_ids.shape[1])
    positions = torch.arange(prompt_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
    logits_last = model.prefill(input_ids, positions=positions, attention_mask=attention_mask)[0].clone()

    for step in range(int(max_new_tokens)):
        sampled = sample_next_token(
            logits=logits_last,
            temperature=float(temperature) if temperature is not None and temperature > 0 else 0.0,
            top_p=float(top_p),
            generated_token_ids=generated_token_ids,
            repetition_penalty=repetition_penalty,
            generator=None,
        )
        token_id = int(sampled.token_id)
        trace.append(
            {
                "step": step,
                "token_id": token_id,
                "token_text": token_to_text(tokenizer, token_id),
                "eos_logit": float(logits_last[int(eos_token_id)].item()) if eos_token_id is not None else None,
                "eos_rank": eos_rank(logits_last, eos_token_id),
                "top_token_info": collect_top_token_info(tokenizer, logits_last, top_k_to_show),
                "top_tokens": summarize_top_tokens(tokenizer, logits_last, top_k_to_show),
            }
        )
        generated_token_ids.append(token_id)
        if eos_token_id is not None and token_id == int(eos_token_id):
            break
        next_token = torch.tensor([[token_id]], device=input_ids.device, dtype=torch.long)
        step_pos = torch.tensor([[prompt_len + len(generated_token_ids) - 1]], device=input_ids.device, dtype=torch.long)
        logits_last = model.decode(next_token, positions=step_pos, active_mask=torch.tensor([True], device=input_ids.device))[0].clone()

    return trace


def print_trace_comparison(
    prompt_index: int,
    prompt: str,
    basic_trace: list[dict],
    engine_trace: list[dict],
    extra_token_ids: list[int] | None = None,
) -> None:
    print(f"\n### Diagnose Prompt {prompt_index}")
    print(f"Prompt: {prompt}")
    max_steps = max(len(basic_trace), len(engine_trace))
    first_divergence: int | None = None
    for step in range(max_steps):
        basic_step = basic_trace[step] if step < len(basic_trace) else None
        engine_step = engine_trace[step] if step < len(engine_trace) else None
        basic_token = basic_step["token_id"] if basic_step is not None else None
        engine_token = engine_step["token_id"] if engine_step is not None else None
        status = "MATCH" if basic_token == engine_token else "DIFF"
        print(
            f"step={step} status={status} "
            f"basic={basic_token}:{basic_step['token_text'] if basic_step else '<none>'} "
            f"engine={engine_token}:{engine_step['token_text'] if engine_step else '<none>'}"
        )
        if status == "DIFF" and first_divergence is None:
            first_divergence = step
            if basic_step is not None:
                print(
                    f"  basic: eos_rank={basic_step['eos_rank']}, eos_logit={basic_step['eos_logit']}, "
                    f"top={basic_step['top_tokens']}"
                )
            if engine_step is not None:
                print(
                    f"  engine: eos_rank={engine_step['eos_rank']}, eos_logit={engine_step['eos_logit']}, "
                    f"top={engine_step['top_tokens']}"
                )
            if basic_step is not None and engine_step is not None:
                basic_map = {item["token_id"]: item for item in basic_step["top_token_info"]}
                engine_map = {item["token_id"]: item for item in engine_step["top_token_info"]}
                compared_ids = list(dict.fromkeys(
                    list(basic_map.keys())
                    + list(engine_map.keys())
                    + [int(x) for x in (extra_token_ids or [])]
                ))
                if compared_ids:
                    print("  compared_logits:")
                    for token_id in compared_ids:
                        basic_item = basic_map.get(int(token_id))
                        engine_item = engine_map.get(int(token_id))
                        token_text = (
                            (basic_item or {}).get("token_text")
                            or (engine_item or {}).get("token_text")
                            or "<unknown>"
                        )
                        basic_logit = basic_item["logit"] if basic_item is not None else None
                        engine_logit = engine_item["logit"] if engine_item is not None else None
                        delta = None
                        if basic_logit is not None and engine_logit is not None:
                            delta = basic_logit - engine_logit
                        print(
                            f"    token={int(token_id)}:{token_text} "
                            f"basic={basic_logit if basic_logit is not None else '<not-in-topk>'} "
                            f"engine={engine_logit if engine_logit is not None else '<not-in-topk>'} "
                            f"delta={delta if delta is not None else '<na>'}"
                        )
            break
    if first_divergence is None:
        print(f"首 {max_steps} 步完全一致")


def capture_basic_cache_logits(
    model,
    tokenizer,
    text: str,
    device: str,
    forced_token_ids: list[int],
    target_step: int,
) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    past_key_values = None
    for step in range(int(target_step) + 1):
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        outputs = model.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :].clone()[0]
        if step == int(target_step):
            return logits
        next_token_id = int(forced_token_ids[step])
        next_token = torch.tensor([[next_token_id]], device=input_ids.device, dtype=torch.long)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
        past_key_values = outputs.past_key_values
    raise RuntimeError(f"无法捕获 basic cache logits, target_step={target_step}")


def capture_infer_cache_logits(
    model,
    tokenizer,
    text: str,
    device: str,
    forced_token_ids: list[int],
    target_step: int,
) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    prompt_len = int(input_ids.shape[1])
    positions = torch.arange(prompt_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
    logits_last = model.prefill(input_ids, positions=positions, attention_mask=attention_mask)[0].clone()
    for step in range(int(target_step) + 1):
        if step == int(target_step):
            return logits_last
        next_token_id = int(forced_token_ids[step])
        next_token = torch.tensor([[next_token_id]], device=input_ids.device, dtype=torch.long)
        step_pos = torch.tensor([[prompt_len + step]], device=input_ids.device, dtype=torch.long)
        logits_last = model.decode(
            next_token,
            positions=step_pos,
            active_mask=torch.tensor([True], device=input_ids.device),
        )[0].clone()
    raise RuntimeError(f"无法捕获 infer cache logits, target_step={target_step}")


def capture_infer_full_recompute_logits(
    model,
    tokenizer,
    text: str,
    device: str,
    forced_token_ids: list[int],
    target_step: int,
) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    if int(target_step) > 0:
        forced_prefix = torch.tensor(forced_token_ids[: int(target_step)], device=input_ids.device, dtype=torch.long).view(1, -1)
        input_ids = torch.cat([input_ids, forced_prefix], dim=-1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
    logits_last = model.prefill(input_ids, positions=positions, attention_mask=attention_mask)[0].clone()
    return logits_last


def infer_token_text_from_maps(tokenizer, token_id: int, info_maps: list[dict[int, dict]]) -> str:
    for info_map in info_maps:
        item = info_map.get(int(token_id))
        if item is not None:
            return str(item.get("token_text", ""))
    return token_to_text(tokenizer, int(token_id))


def print_forward_diagnose_result(
    tokenizer,
    prompt_index: int,
    prompt: str,
    target_step: int,
    forced_prefix_ids: list[int],
    basic_cache_logits: torch.Tensor,
    infer_cache_logits: torch.Tensor,
    infer_full_logits: torch.Tensor,
    top_k: int,
    extra_token_ids: list[int] | None = None,
) -> None:
    print(f"\n### Diagnose Forward Prompt {prompt_index}")
    print(f"Prompt: {prompt}")
    print(f"target_step={target_step}")
    if forced_prefix_ids:
        preview_ids = forced_prefix_ids[-8:]
        preview_text = "".join(token_to_text(tokenizer, token_id) for token_id in preview_ids)
        print(f"forced_prefix_tokens={len(forced_prefix_ids)}; tail={preview_text}")
    else:
        print("forced_prefix_tokens=0")

    basic_top = collect_top_token_info(tokenizer, basic_cache_logits, top_k)
    infer_cache_top = collect_top_token_info(tokenizer, infer_cache_logits, top_k)
    infer_full_top = collect_top_token_info(tokenizer, infer_full_logits, top_k)
    print(f"basic-cache top={summarize_top_tokens(tokenizer, basic_cache_logits, top_k)}")
    print(f"infer-cache top={summarize_top_tokens(tokenizer, infer_cache_logits, top_k)}")
    print(f"infer-full top={summarize_top_tokens(tokenizer, infer_full_logits, top_k)}")

    basic_map = {item["token_id"]: item for item in basic_top}
    infer_cache_map = {item["token_id"]: item for item in infer_cache_top}
    infer_full_map = {item["token_id"]: item for item in infer_full_top}
    compared_ids = list(
        dict.fromkeys(
            list(basic_map.keys())
            + list(infer_cache_map.keys())
            + list(infer_full_map.keys())
            + [int(x) for x in (extra_token_ids or [])]
        )
    )
    info_maps = [basic_map, infer_cache_map, infer_full_map]
    print("compared_logits:")
    for token_id in compared_ids:
        token_text = infer_token_text_from_maps(tokenizer, int(token_id), info_maps)
        basic_logit = float(basic_cache_logits[int(token_id)].item())
        infer_cache_logit = float(infer_cache_logits[int(token_id)].item())
        infer_full_logit = float(infer_full_logits[int(token_id)].item())
        print(
            f"  token={int(token_id)}:{token_text} "
            f"basic-cache={basic_logit:.6f} "
            f"infer-cache={infer_cache_logit:.6f} "
            f"infer-full={infer_full_logit:.6f} "
            f"delta(cache-basic)={infer_cache_logit - basic_logit:.6f} "
            f"delta(full-basic)={infer_full_logit - basic_logit:.6f} "
            f"delta(cache-full)={infer_cache_logit - infer_full_logit:.6f}"
        )


def tensor_diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).abs().float()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def format_diff_stats(name: str, stats: dict[str, float]) -> str:
    return f"{name}: max_abs={stats['max_abs']:.6f}, mean_abs={stats['mean_abs']:.6f}"


def capture_basic_layerwise_outputs(
    model,
    tokenizer,
    text: str,
    device: str,
    forced_token_ids: list[int],
) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    if forced_token_ids:
        forced_prefix = torch.tensor(forced_token_ids, device=input_ids.device, dtype=torch.long).view(1, -1)
        input_ids = torch.cat([input_ids, forced_prefix], dim=-1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    hidden_states = model.model.dropout(model.model.embed_tokens(input_ids))
    embedding = hidden_states.clone()
    if model.model.freqs_cos[0, 0] == 0:
        from models.model import precompute_freqs_cis as basic_precompute_freqs_cis

        freqs_cos, freqs_sin = basic_precompute_freqs_cis(
            dim=model.config.head_dim,
            end=model.config.max_position_embeddings,
            rope_base=model.config.rope_theta,
            rope_scaling=model.config.rope_scaling,
        )
        model.model.freqs_cos = freqs_cos.to(hidden_states.device)
        model.model.freqs_sin = freqs_sin.to(hidden_states.device)
    seq_length = int(input_ids.shape[1])
    position_embeddings = (
        model.model.freqs_cos[:seq_length],
        model.model.freqs_sin[:seq_length],
    )
    layers: list[dict] = []
    for layer in model.model.layers:
        residual = hidden_states
        attn_input = layer.input_layernorm(hidden_states)
        attn_out, _ = layer.self_attn(
            attn_input,
            position_embeddings,
            past_key_value=None,
            use_cache=False,
            attention_mask=attention_mask,
        )
        after_attn = residual + attn_out
        mlp_input = layer.post_attention_layernorm(after_attn)
        mlp_out = layer.mlp(mlp_input)
        hidden_states = after_attn + mlp_out
        layers.append(
            {
                "attn_out": attn_out.clone(),
                "mlp_out": mlp_out.clone(),
                "hidden": hidden_states.clone(),
            }
        )
    final_hidden = model.model.norm(hidden_states)
    final_logits = model.lm_head(final_hidden)[:, -1, :]
    return {
        "embedding": embedding,
        "layers": layers,
        "final_hidden": final_hidden,
        "final_logits": final_logits,
    }


def capture_infer_layerwise_outputs(
    model,
    tokenizer,
    text: str,
    device: str,
    forced_token_ids: list[int],
) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    if forced_token_ids:
        forced_prefix = torch.tensor(forced_token_ids, device=input_ids.device, dtype=torch.long).view(1, -1)
        input_ids = torch.cat([input_ids, forced_prefix], dim=-1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
    hidden_states = model.model.dropout(model.model.embed_tokens(input_ids))
    embedding = hidden_states.clone()
    layers: list[dict] = []
    for layer in model.model.layers:
        residual = hidden_states
        attn_input = layer.input_layernorm(hidden_states)
        attn_out = layer.self_attn(
            attn_input,
            positions,
            active_mask=None,
            attention_mask=attention_mask,
        )
        after_attn = residual + attn_out
        mlp_input = layer.post_attention_layernorm(after_attn)
        mlp_out = layer.mlp(mlp_input)
        hidden_states = after_attn + mlp_out
        layers.append(
            {
                "attn_out": attn_out.clone(),
                "mlp_out": mlp_out.clone(),
                "hidden": hidden_states.clone(),
            }
        )
    final_hidden = model.model.norm(hidden_states)
    final_logits = model.lm_head(final_hidden)[:, -1, :]
    return {
        "embedding": embedding,
        "layers": layers,
        "final_hidden": final_hidden,
        "final_logits": final_logits,
    }


def print_layer_diagnose_result(
    tokenizer,
    prompt_index: int,
    prompt: str,
    target_step: int,
    forced_prefix_ids: list[int],
    basic_outputs: dict,
    infer_outputs: dict,
    top_k: int,
    extra_token_ids: list[int] | None = None,
) -> None:
    print(f"\n### Diagnose Layers Prompt {prompt_index}")
    print(f"Prompt: {prompt}")
    print(f"target_step={target_step}")
    if forced_prefix_ids:
        preview_ids = forced_prefix_ids[-8:]
        preview_text = "".join(token_to_text(tokenizer, token_id) for token_id in preview_ids)
        print(f"forced_prefix_tokens={len(forced_prefix_ids)}; tail={preview_text}")
    else:
        print("forced_prefix_tokens=0")

    print(format_diff_stats("embedding", tensor_diff_stats(basic_outputs["embedding"], infer_outputs["embedding"])))
    for layer_idx, (basic_layer, infer_layer) in enumerate(zip(basic_outputs["layers"], infer_outputs["layers"])):
        attn_stats = tensor_diff_stats(basic_layer["attn_out"], infer_layer["attn_out"])
        mlp_stats = tensor_diff_stats(basic_layer["mlp_out"], infer_layer["mlp_out"])
        hidden_stats = tensor_diff_stats(basic_layer["hidden"], infer_layer["hidden"])
        print(
            f"layer={layer_idx} "
            f"{format_diff_stats('attn', attn_stats)}; "
            f"{format_diff_stats('mlp', mlp_stats)}; "
            f"{format_diff_stats('hidden', hidden_stats)}"
        )

    final_hidden_stats = tensor_diff_stats(basic_outputs["final_hidden"], infer_outputs["final_hidden"])
    final_logits_stats = tensor_diff_stats(basic_outputs["final_logits"], infer_outputs["final_logits"])
    print(format_diff_stats("final_hidden", final_hidden_stats))
    print(format_diff_stats("final_logits", final_logits_stats))

    basic_logits = basic_outputs["final_logits"][0]
    infer_logits = infer_outputs["final_logits"][0]
    print(f"basic top={summarize_top_tokens(tokenizer, basic_logits, top_k)}")
    print(f"infer top={summarize_top_tokens(tokenizer, infer_logits, top_k)}")
    compared_ids = list(
        dict.fromkeys(
            [item["token_id"] for item in collect_top_token_info(tokenizer, basic_logits, top_k)]
            + [item["token_id"] for item in collect_top_token_info(tokenizer, infer_logits, top_k)]
            + [int(x) for x in (extra_token_ids or [])]
        )
    )
    print("compared_logits:")
    for token_id in compared_ids:
        token_text = token_to_text(tokenizer, int(token_id))
        basic_logit = float(basic_logits[int(token_id)].item())
        infer_logit = float(infer_logits[int(token_id)].item())
        print(
            f"  token={int(token_id)}:{token_text} "
            f"basic={basic_logit:.6f} "
            f"infer={infer_logit:.6f} "
            f"delta={infer_logit - basic_logit:.6f}"
        )


def set_flash_enabled(model, enabled: bool) -> None:
    if hasattr(model, "config"):
        setattr(model.config, "flash_attn", bool(enabled))
    model_impl = getattr(model, "model", None)
    if model_impl is None:
        return
    for layer in getattr(model_impl, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "flash"):
            attn.flash = bool(enabled)


def build_forced_inputs(tokenizer, text: str, device: str, forced_token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    if forced_token_ids:
        forced_prefix = torch.tensor(forced_token_ids, device=input_ids.device, dtype=torch.long).view(1, -1)
        input_ids = torch.cat([input_ids, forced_prefix], dim=-1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    return input_ids, attention_mask


def get_basic_hidden_before_layer(
    model,
    tokenizer,
    text: str,
    device: str,
    forced_token_ids: list[int],
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    input_ids, attention_mask = build_forced_inputs(tokenizer, text, device, forced_token_ids)
    hidden_states = model.model.dropout(model.model.embed_tokens(input_ids))
    if model.model.freqs_cos[0, 0] == 0:
        from models.model import precompute_freqs_cis as basic_precompute_freqs_cis

        freqs_cos, freqs_sin = basic_precompute_freqs_cis(
            dim=model.config.head_dim,
            end=model.config.max_position_embeddings,
            rope_base=model.config.rope_theta,
            rope_scaling=model.config.rope_scaling,
        )
        model.model.freqs_cos = freqs_cos.to(hidden_states.device)
        model.model.freqs_sin = freqs_sin.to(hidden_states.device)
    seq_length = int(input_ids.shape[1])
    position_embeddings = (
        model.model.freqs_cos[:seq_length],
        model.model.freqs_sin[:seq_length],
    )
    for layer in model.model.layers[: int(layer_idx)]:
        residual = hidden_states
        attn_input = layer.input_layernorm(hidden_states)
        attn_out, _ = layer.self_attn(
            attn_input,
            position_embeddings,
            past_key_value=None,
            use_cache=False,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_out
        hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
    return hidden_states, attention_mask, position_embeddings


def get_infer_hidden_before_layer(
    model,
    tokenizer,
    text: str,
    device: str,
    forced_token_ids: list[int],
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = build_forced_inputs(tokenizer, text, device, forced_token_ids)
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    hidden_states = model.model.dropout(model.model.embed_tokens(input_ids))
    positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
    for layer in model.model.layers[: int(layer_idx)]:
        residual = hidden_states
        attn_input = layer.input_layernorm(hidden_states)
        attn_out = layer.self_attn(
            attn_input,
            positions,
            active_mask=None,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_out
        hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    return hidden_states, attention_mask, positions


def summarize_score_positions(scores_row: torch.Tensor, top_k: int) -> list[str]:
    k = max(1, min(int(top_k), int(scores_row.shape[-1])))
    values, indices = torch.topk(scores_row, k=k)
    return [f"pos={int(pos)}:{float(val):.6f}" for pos, val in zip(indices.tolist(), values.tolist())]


def capture_basic_attention_details(
    model,
    tokenizer,
    text: str,
    device: str,
    forced_token_ids: list[int],
    layer_idx: int,
) -> dict:
    from models.model import apply_rotary_pos_emb as basic_apply_rotary_pos_emb
    from models.model import repeat_kv as basic_repeat_kv

    hidden_states, attention_mask, position_embeddings = get_basic_hidden_before_layer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        forced_token_ids=forced_token_ids,
        layer_idx=layer_idx,
    )
    layer = model.model.layers[int(layer_idx)]
    attn = layer.self_attn
    attn_input = layer.input_layernorm(hidden_states)
    bsz, seq_len, _ = attn_input.shape
    xq = attn.q_proj(attn_input).view(bsz, seq_len, attn.n_local_heads, attn.head_dim)
    xk = attn.k_proj(attn_input).view(bsz, seq_len, attn.n_local_kv_heads, attn.head_dim)
    xv = attn.v_proj(attn_input).view(bsz, seq_len, attn.n_local_kv_heads, attn.head_dim)
    xq_norm = attn.q_norm(xq)
    xk_norm = attn.k_norm(xk)
    cos, sin = position_embeddings
    xq_rope, xk_rope = basic_apply_rotary_pos_emb(xq_norm, xk_norm, cos, sin)
    xq_t = xq_rope.transpose(1, 2)
    xk_t = basic_repeat_kv(xk_rope, attn.n_rep).transpose(1, 2)
    xv_t = basic_repeat_kv(xv, attn.n_rep).transpose(1, 2)
    scores = (xq_t @ xk_t.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    if attn.is_causal:
        scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
    if attention_mask is not None:
        scores = scores + (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=scores.dtype)) * -1e9
    probs = torch.softmax(scores.float(), dim=-1).type_as(xq_t)
    use_flash_path = bool(attn.flash and (seq_len > 1) and torch.all(attention_mask == 1).item())
    if use_flash_path:
        actual_out = F.scaled_dot_product_attention(
            xq_t,
            xk_t,
            xv_t,
            dropout_p=attn.dropout if attn.training else 0.0,
            is_causal=attn.is_causal,
        )
        use_flash_path = True
    else:
        actual_out = attn.attn_dropout(probs) @ xv_t
        use_flash_path = False
    actual_out = actual_out.transpose(1, 2).reshape(bsz, seq_len, -1)
    attn_out = attn.resid_dropout(attn.o_proj(actual_out))
    return {
        "use_flash_path": use_flash_path,
        "attn_input": attn_input,
        "q_norm": xq_norm,
        "k_norm": xk_norm,
        "q_rope": xq_rope,
        "k_rope": xk_rope,
        "scores": scores,
        "probs": probs,
        "attn_out": attn_out,
    }


def capture_infer_attention_details(
    model,
    tokenizer,
    text: str,
    device: str,
    forced_token_ids: list[int],
    layer_idx: int,
) -> dict:
    from models.model_infer import repeat_kv as infer_repeat_kv

    hidden_states, attention_mask, positions = get_infer_hidden_before_layer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        forced_token_ids=forced_token_ids,
        layer_idx=layer_idx,
    )
    layer = model.model.layers[int(layer_idx)]
    attn = layer.self_attn
    attn_input = layer.input_layernorm(hidden_states)
    bsz, seq_len, _ = attn_input.shape
    xq = attn.q_proj(attn_input).view(bsz, seq_len, attn.n_local_heads, attn.head_dim)
    xk = attn.k_proj(attn_input).view(bsz, seq_len, attn.n_local_kv_heads, attn.head_dim)
    xv = attn.v_proj(attn_input).view(bsz, seq_len, attn.n_local_kv_heads, attn.head_dim)
    xq_norm = attn.q_norm(xq)
    xk_norm = attn.k_norm(xk)
    xq_rope, xk_rope = attn.rotary_emb(positions, xq_norm, xk_norm)
    token_mask = attention_mask.to(dtype=torch.bool, device=attn_input.device)
    token_mask_4d = token_mask.view(bsz, seq_len, 1, 1)
    k_all = torch.where(token_mask_4d, xk_rope, torch.zeros_like(xk_rope))
    v_all = torch.where(token_mask_4d, xv, torch.zeros_like(xv))
    xq_t = xq_rope.transpose(1, 2)
    xk_t = infer_repeat_kv(k_all, attn.n_rep).transpose(1, 2)
    xv_t = infer_repeat_kv(v_all, attn.n_rep).transpose(1, 2)
    scores = (xq_t @ xk_t.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    if attn.is_causal:
        scores[:, :, :, -seq_len:] += torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=scores.device,
            dtype=scores.dtype,
        ).triu(1)
    scores = scores + (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=scores.dtype)) * -1e9
    probs = torch.softmax(scores.float(), dim=-1).type_as(xq_t)
    use_flash_path = bool(attn.flash and (seq_len > 1) and torch.all(token_mask).item())
    if use_flash_path:
        actual_out = F.scaled_dot_product_attention(
            xq_t,
            xk_t,
            xv_t,
            dropout_p=attn.dropout if attn.training else 0.0,
            is_causal=attn.is_causal,
        )
    else:
        actual_out = attn.attn_dropout(probs) @ xv_t
    actual_out = actual_out.transpose(1, 2).reshape(bsz, seq_len, -1)
    actual_out = actual_out * token_mask.unsqueeze(-1).to(dtype=actual_out.dtype)
    attn_out = attn.resid_dropout(attn.o_proj(actual_out))
    return {
        "use_flash_path": use_flash_path,
        "attn_input": attn_input,
        "q_norm": xq_norm,
        "k_norm": xk_norm,
        "q_rope": xq_rope,
        "k_rope": xk_rope,
        "scores": scores,
        "probs": probs,
        "attn_out": attn_out,
    }


def print_attention_diagnose_result(
    tokenizer,
    prompt_index: int,
    prompt: str,
    target_step: int,
    forced_prefix_ids: list[int],
    layer_idx: int,
    head_idx: int,
    basic_details: dict,
    infer_details: dict,
    top_k: int,
) -> None:
    print(f"\n### Diagnose Attention Prompt {prompt_index}")
    print(f"Prompt: {prompt}")
    print(f"target_step={target_step}; layer_idx={layer_idx}; head_idx={head_idx}")
    if forced_prefix_ids:
        preview_ids = forced_prefix_ids[-8:]
        preview_text = "".join(token_to_text(tokenizer, token_id) for token_id in preview_ids)
        print(f"forced_prefix_tokens={len(forced_prefix_ids)}; tail={preview_text}")
    else:
        print("forced_prefix_tokens=0")
    print(
        f"use_flash_path: basic={basic_details['use_flash_path']} "
        f"infer={infer_details['use_flash_path']}"
    )
    for name in ("attn_input", "q_norm", "k_norm", "q_rope", "k_rope", "attn_out"):
        print(format_diff_stats(name, tensor_diff_stats(basic_details[name], infer_details[name])))
    print(format_diff_stats("scores", tensor_diff_stats(basic_details["scores"], infer_details["scores"])))
    print(format_diff_stats("probs", tensor_diff_stats(basic_details["probs"], infer_details["probs"])))
    basic_last_scores = basic_details["scores"][0, int(head_idx), -1, :]
    infer_last_scores = infer_details["scores"][0, int(head_idx), -1, :]
    basic_last_probs = basic_details["probs"][0, int(head_idx), -1, :]
    infer_last_probs = infer_details["probs"][0, int(head_idx), -1, :]
    print(format_diff_stats("last_scores", tensor_diff_stats(basic_last_scores, infer_last_scores)))
    print(format_diff_stats("last_probs", tensor_diff_stats(basic_last_probs, infer_last_probs)))
    print(f"basic last_scores top={summarize_score_positions(basic_last_scores, top_k)}")
    print(f"infer last_scores top={summarize_score_positions(infer_last_scores, top_k)}")
    print(f"basic last_probs top={summarize_score_positions(basic_last_probs, top_k)}")
    print(f"infer last_probs top={summarize_score_positions(infer_last_probs, top_k)}")


def benchmark_basic_generate(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> dict:
    baseline_alloc, baseline_reserved = reset_cuda_stats(device)
    maybe_sync(device)
    st = time.time()
    responses: list[str] = []
    total_tokens = 0
    per_prompt_tokens: list[int] = []
    basic_top_p = float(top_p) if temperature is not None and temperature > 0 else 1.0
    basic_top_k = 50 if temperature is not None and temperature > 0 else 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=basic_top_p,
            top_k=basic_top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=bool(temperature > 0),
            use_cache=True,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        response_ids = outputs[0][prompt_len:]
        total_tokens += int(response_ids.shape[0])
        per_prompt_tokens.append(int(response_ids.shape[0]))
        responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))
    maybe_sync(device)
    wall_s = time.time() - st
    peak_alloc, peak_reserved = collect_cuda_peaks(device)
    return {
        "name": "basic.generate",
        "responses": responses,
        "per_prompt_tokens": per_prompt_tokens,
        "total_tokens": total_tokens,
        "wall_s": wall_s,
        "throughput_tps": (total_tokens / wall_s) if wall_s > 0 else 0.0,
        "baseline_alloc_mb": baseline_alloc,
        "baseline_reserved_mb": baseline_reserved,
        "peak_alloc_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
    }


def benchmark_basic_generate_batch(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> dict:
    encoded = [tokenizer(text, return_tensors="pt", truncation=True) for text in texts]
    prompt_lens = [int(x["input_ids"].shape[1]) for x in encoded]
    if len(set(prompt_lens)) != 1:
        raise ValueError(
            f"benchmark_basic_generate_batch 要求 batch 内长度一致，当前 prompt_lens={prompt_lens}"
        )

    batch_inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
    baseline_alloc, baseline_reserved = reset_cuda_stats(device)
    maybe_sync(device)
    st = time.time()
    basic_top_p = float(top_p) if temperature is not None and temperature > 0 else 1.0
    basic_top_k = 50 if temperature is not None and temperature > 0 else 0
    outputs = model.generate(
        inputs=batch_inputs["input_ids"],
        attention_mask=batch_inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=basic_top_p,
        top_k=basic_top_k,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=bool(temperature > 0),
        use_cache=True,
    )
    maybe_sync(device)
    wall_s = time.time() - st
    peak_alloc, peak_reserved = collect_cuda_peaks(device)

    prompt_width = int(batch_inputs["input_ids"].shape[1])
    responses: list[str] = []
    total_tokens = 0
    per_prompt_tokens: list[int] = []
    for i in range(outputs.shape[0]):
        response_ids = outputs[i][prompt_width:]
        gen_len = int(response_ids.shape[0])
        total_tokens += gen_len
        per_prompt_tokens.append(gen_len)
        responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))

    return {
        "name": "basic.generate_batch",
        "responses": responses,
        "per_prompt_tokens": per_prompt_tokens,
        "total_tokens": total_tokens,
        "wall_s": wall_s,
        "throughput_tps": (total_tokens / wall_s) if wall_s > 0 else 0.0,
        "baseline_alloc_mb": baseline_alloc,
        "baseline_reserved_mb": baseline_reserved,
        "peak_alloc_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
    }


def benchmark_engine_generate(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> dict:
    engine = InferenceEngine(model=model, tokenizer=tokenizer, config=EngineConfig(device=device, use_kv_cache=True))
    baseline_alloc, baseline_reserved = reset_cuda_stats(device)
    maybe_sync(device)
    st = time.time()
    responses: list[str] = []
    total_tokens = 0
    per_prompt_tokens: list[int] = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        outputs = engine.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=bool(temperature > 0),
        )
        state = engine._last_batch_state
        gen_len = int(state.generated_lens[0].item()) if state is not None else int(outputs.shape[1] - inputs["input_ids"].shape[1])
        prompt_width = int(inputs["input_ids"].shape[1])
        response_ids = outputs[0][prompt_width : prompt_width + gen_len]
        total_tokens += gen_len
        per_prompt_tokens.append(gen_len)
        responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))
    maybe_sync(device)
    wall_s = time.time() - st
    peak_alloc, peak_reserved = collect_cuda_peaks(device)
    return {
        "name": "engine.generate",
        "responses": responses,
        "per_prompt_tokens": per_prompt_tokens,
        "total_tokens": total_tokens,
        "wall_s": wall_s,
        "throughput_tps": (total_tokens / wall_s) if wall_s > 0 else 0.0,
        "baseline_alloc_mb": baseline_alloc,
        "baseline_reserved_mb": baseline_reserved,
        "peak_alloc_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
    }


def benchmark_engine_generate_batch(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> dict:
    engine = InferenceEngine(model=model, tokenizer=tokenizer, config=EngineConfig(device=device, use_kv_cache=True))
    batch_inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
    baseline_alloc, baseline_reserved = reset_cuda_stats(device)
    maybe_sync(device)
    st = time.time()
    outputs = engine.generate_batch(
        input_ids=batch_inputs["input_ids"],
        attention_mask=batch_inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=bool(temperature > 0),
    )
    maybe_sync(device)
    wall_s = time.time() - st
    peak_alloc, peak_reserved = collect_cuda_peaks(device)

    state = engine._last_batch_state
    prompt_width = int(batch_inputs["input_ids"].shape[1])
    responses: list[str] = []
    total_tokens = 0
    per_prompt_tokens: list[int] = []
    for i in range(outputs.shape[0]):
        gen_len = int(state.generated_lens[i].item()) if state is not None else int(outputs.shape[1] - prompt_width)
        response_ids = outputs[i][prompt_width : prompt_width + gen_len]
        total_tokens += gen_len
        per_prompt_tokens.append(gen_len)
        responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))

    return {
        "name": "engine.generate_batch",
        "responses": responses,
        "per_prompt_tokens": per_prompt_tokens,
        "total_tokens": total_tokens,
        "wall_s": wall_s,
        "throughput_tps": (total_tokens / wall_s) if wall_s > 0 else 0.0,
        "baseline_alloc_mb": baseline_alloc,
        "baseline_reserved_mb": baseline_reserved,
        "peak_alloc_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
    }


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_summary(result: dict) -> None:
    peak_alloc_delta = result["peak_alloc_mb"] - result["baseline_alloc_mb"]
    peak_reserved_delta = result["peak_reserved_mb"] - result["baseline_reserved_mb"]
    print(
        f"[{result['name']}] total={result['total_tokens']} tokens; "
        f"wall={result['wall_s']:.2f}s; throughput={result['throughput_tps']:.2f} tokens/s; "
        f"mem_base_alloc={result['baseline_alloc_mb']:.1f}MB; "
        f"mem_base_reserved={result['baseline_reserved_mb']:.1f}MB; "
        f"mem_peak_alloc={result['peak_alloc_mb']:.1f}MB (+{peak_alloc_delta:.1f}MB); "
        f"mem_peak_reserved={result['peak_reserved_mb']:.1f}MB (+{peak_reserved_delta:.1f}MB)"
    )
    if "per_prompt_tokens" in result:
        print(f"per_prompt_tokens={result['per_prompt_tokens']}")
    if "note" in result and result["note"]:
        print(f"note={result['note']}")


def print_outputs(prompts: list[str], results: list[dict]) -> None:
    print("\n### Outputs")
    for result in results:
        result_prompts = result.get("prompts", prompts)
        print(f"\n[{result['name']}]")
        for i, prompt in enumerate(result_prompts):
            print(f"[{i}] Prompt: {prompt}")
            print(result["responses"][i])
            print()


def main():
    parser = argparse.ArgumentParser(
        description="对比 basic.generate / basic.generate_batch / engine.generate / engine.generate_batch"
    )
    parser.add_argument("--load_from", default="/root/autodl-tmp/minimind/minimind-3", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--use_chat_template", default=1, type=int, choices=[0, 1])
    parser.add_argument("--open_thinking", default=0, type=int, choices=[0, 1])
    parser.add_argument("--num_prompts", default=8, type=int)
    parser.add_argument("--prompts", default="", type=str, help="自定义 prompt，使用 || 分隔")
    parser.add_argument("--auto_select_equal_length", default=1, type=int, choices=[0, 1])
    parser.add_argument("--min_equal_batch_size", default=2, type=int)
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "all",
            "basic",
            "basic_batch",
            "engine",
            "engine_batch",
            "diagnose",
            "diagnose_forward",
            "diagnose_layers",
            "diagnose_attention",
        ],
        help="选择运行哪一条路径；分别运行时可得到更干净的显存/吞吐数据",
    )
    parser.add_argument("--show_outputs", default=1, type=int, choices=[0, 1])
    parser.add_argument("--diagnose_indices", default="0", type=str, help="诊断 prompt 下标，使用逗号分隔")
    parser.add_argument("--diagnose_steps", default=64, type=int, help="逐步诊断时最多比较多少步")
    parser.add_argument("--diagnose_top_k", default=5, type=int, help="分叉步打印多少个 top token")
    parser.add_argument("--diagnose_target_step", default=-1, type=int, help="定点前向诊断的目标 step")
    parser.add_argument("--diagnose_layer_idx", default=0, type=int, help="attention 诊断时要看的层号")
    parser.add_argument("--diagnose_head_idx", default=0, type=int, help="attention 诊断时要看的 head 号")
    parser.add_argument(
        "--diagnose_extra_token_ids",
        default="",
        type=str,
        help="首次分叉步额外对比的 token id，使用逗号分隔；为空时自动比较双方 top-k",
    )
    parser.add_argument("--disable_flash", default=0, type=int, choices=[0, 1], help="是否关闭 flash attention 快路径")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    raw_prompts = [x.strip() for x in str(args.prompts).split("||") if x.strip()] if args.prompts else default_prompts()
    prompts = raw_prompts[: int(args.num_prompts)]
    texts = [
        build_text(
            tokenizer=tokenizer,
            prompt=prompt,
            use_chat_template=bool(args.use_chat_template),
            open_thinking=bool(args.open_thinking),
        )
        for prompt in prompts
    ]
    dtype = resolve_dtype(args.dtype, args.device)

    results = []

    run_basic = args.mode in ("all", "basic")
    run_basic_batch = args.mode in ("all", "basic_batch")
    run_engine = args.mode in ("all", "engine")
    run_engine_batch = args.mode in ("all", "engine_batch")
    run_diagnose = args.mode == "diagnose"
    run_diagnose_forward = args.mode == "diagnose_forward"
    run_diagnose_layers = args.mode == "diagnose_layers"
    run_diagnose_attention = args.mode == "diagnose_attention"

    if run_diagnose:
        diagnose_indices = [int(x.strip()) for x in str(args.diagnose_indices).split(",") if x.strip()]
        diagnose_extra_token_ids = [
            int(x.strip()) for x in str(args.diagnose_extra_token_ids).split(",") if x.strip()
        ]
        if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) not in diagnose_extra_token_ids:
            diagnose_extra_token_ids.append(int(tokenizer.eos_token_id))
        if not diagnose_indices:
            raise ValueError("diagnose_indices 不能为空")
        basic_model = init_basic_model(args.load_from, args.device, dtype)
        infer_model = init_infer_model(args.load_from, args.device, dtype)
        if int(args.disable_flash):
            set_flash_enabled(basic_model, False)
            set_flash_enabled(infer_model, False)
        try:
            for idx in diagnose_indices:
                if idx < 0 or idx >= len(prompts):
                    raise IndexError(f"prompt 下标越界: idx={idx}, num_prompts={len(prompts)}")
                basic_trace = trace_basic_single(
                    model=basic_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    max_new_tokens=int(args.diagnose_steps),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    repetition_penalty=float(args.repetition_penalty),
                    top_k_to_show=int(args.diagnose_top_k),
                )
                engine_trace = trace_engine_single(
                    model=infer_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    max_new_tokens=int(args.diagnose_steps),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    repetition_penalty=float(args.repetition_penalty),
                    top_k_to_show=int(args.diagnose_top_k),
                )
                print_trace_comparison(
                    prompt_index=idx,
                    prompt=prompts[idx],
                    basic_trace=basic_trace,
                    engine_trace=engine_trace,
                    extra_token_ids=diagnose_extra_token_ids,
                )
        finally:
            cleanup_model(basic_model)
            cleanup_model(infer_model)
        return

    if run_diagnose_forward:
        diagnose_indices = [int(x.strip()) for x in str(args.diagnose_indices).split(",") if x.strip()]
        diagnose_extra_token_ids = [
            int(x.strip()) for x in str(args.diagnose_extra_token_ids).split(",") if x.strip()
        ]
        if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) not in diagnose_extra_token_ids:
            diagnose_extra_token_ids.append(int(tokenizer.eos_token_id))
        if not diagnose_indices:
            raise ValueError("diagnose_indices 不能为空")
        if int(args.diagnose_target_step) < 0:
            raise ValueError("diagnose_forward 模式要求 diagnose_target_step >= 0")
        basic_model = init_basic_model(args.load_from, args.device, dtype)
        infer_model = init_infer_model(args.load_from, args.device, dtype)
        if int(args.disable_flash):
            set_flash_enabled(basic_model, False)
            set_flash_enabled(infer_model, False)
        try:
            for idx in diagnose_indices:
                if idx < 0 or idx >= len(prompts):
                    raise IndexError(f"prompt 下标越界: idx={idx}, num_prompts={len(prompts)}")
                basic_trace = trace_basic_single(
                    model=basic_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    max_new_tokens=int(args.diagnose_target_step) + 1,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    repetition_penalty=float(args.repetition_penalty),
                    top_k_to_show=int(args.diagnose_top_k),
                )
                if len(basic_trace) <= int(args.diagnose_target_step):
                    raise ValueError(
                        f"prompt {idx} 在 target_step={args.diagnose_target_step} 前已经结束，"
                        f"当前可用 steps={len(basic_trace)}"
                    )
                forced_prefix_ids = [int(step_info["token_id"]) for step_info in basic_trace[: int(args.diagnose_target_step)]]
                basic_cache_logits = capture_basic_cache_logits(
                    model=basic_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    forced_token_ids=forced_prefix_ids,
                    target_step=int(args.diagnose_target_step),
                )
                infer_cache_logits = capture_infer_cache_logits(
                    model=infer_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    forced_token_ids=forced_prefix_ids,
                    target_step=int(args.diagnose_target_step),
                )
                infer_full_logits = capture_infer_full_recompute_logits(
                    model=infer_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    forced_token_ids=forced_prefix_ids,
                    target_step=int(args.diagnose_target_step),
                )
                print_forward_diagnose_result(
                    tokenizer=tokenizer,
                    prompt_index=idx,
                    prompt=prompts[idx],
                    target_step=int(args.diagnose_target_step),
                    forced_prefix_ids=forced_prefix_ids,
                    basic_cache_logits=basic_cache_logits,
                    infer_cache_logits=infer_cache_logits,
                    infer_full_logits=infer_full_logits,
                    top_k=int(args.diagnose_top_k),
                    extra_token_ids=diagnose_extra_token_ids,
                )
        finally:
            cleanup_model(basic_model)
            cleanup_model(infer_model)
        return

    if run_diagnose_layers:
        diagnose_indices = [int(x.strip()) for x in str(args.diagnose_indices).split(",") if x.strip()]
        diagnose_extra_token_ids = [
            int(x.strip()) for x in str(args.diagnose_extra_token_ids).split(",") if x.strip()
        ]
        if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) not in diagnose_extra_token_ids:
            diagnose_extra_token_ids.append(int(tokenizer.eos_token_id))
        if not diagnose_indices:
            raise ValueError("diagnose_indices 不能为空")
        if int(args.diagnose_target_step) < 0:
            raise ValueError("diagnose_layers 模式要求 diagnose_target_step >= 0")
        basic_model = init_basic_model(args.load_from, args.device, dtype)
        infer_model = init_infer_model(args.load_from, args.device, dtype)
        if int(args.disable_flash):
            set_flash_enabled(basic_model, False)
            set_flash_enabled(infer_model, False)
        try:
            for idx in diagnose_indices:
                if idx < 0 or idx >= len(prompts):
                    raise IndexError(f"prompt 下标越界: idx={idx}, num_prompts={len(prompts)}")
                basic_trace = trace_basic_single(
                    model=basic_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    max_new_tokens=int(args.diagnose_target_step),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    repetition_penalty=float(args.repetition_penalty),
                    top_k_to_show=int(args.diagnose_top_k),
                )
                if len(basic_trace) < int(args.diagnose_target_step):
                    raise ValueError(
                        f"prompt {idx} 在 target_step={args.diagnose_target_step} 前已经结束，"
                        f"当前可用 prefix steps={len(basic_trace)}"
                    )
                forced_prefix_ids = [int(step_info["token_id"]) for step_info in basic_trace[: int(args.diagnose_target_step)]]
                basic_outputs = capture_basic_layerwise_outputs(
                    model=basic_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    forced_token_ids=forced_prefix_ids,
                )
                infer_outputs = capture_infer_layerwise_outputs(
                    model=infer_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    forced_token_ids=forced_prefix_ids,
                )
                print_layer_diagnose_result(
                    tokenizer=tokenizer,
                    prompt_index=idx,
                    prompt=prompts[idx],
                    target_step=int(args.diagnose_target_step),
                    forced_prefix_ids=forced_prefix_ids,
                    basic_outputs=basic_outputs,
                    infer_outputs=infer_outputs,
                    top_k=int(args.diagnose_top_k),
                    extra_token_ids=diagnose_extra_token_ids,
                )
        finally:
            cleanup_model(basic_model)
            cleanup_model(infer_model)
        return

    if run_diagnose_attention:
        diagnose_indices = [int(x.strip()) for x in str(args.diagnose_indices).split(",") if x.strip()]
        if not diagnose_indices:
            raise ValueError("diagnose_indices 不能为空")
        if int(args.diagnose_target_step) < 0:
            raise ValueError("diagnose_attention 模式要求 diagnose_target_step >= 0")
        basic_model = init_basic_model(args.load_from, args.device, dtype)
        infer_model = init_infer_model(args.load_from, args.device, dtype)
        if int(args.disable_flash):
            set_flash_enabled(basic_model, False)
            set_flash_enabled(infer_model, False)
        try:
            for idx in diagnose_indices:
                if idx < 0 or idx >= len(prompts):
                    raise IndexError(f"prompt 下标越界: idx={idx}, num_prompts={len(prompts)}")
                basic_trace = trace_basic_single(
                    model=basic_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    max_new_tokens=int(args.diagnose_target_step),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    repetition_penalty=float(args.repetition_penalty),
                    top_k_to_show=int(args.diagnose_top_k),
                )
                if len(basic_trace) < int(args.diagnose_target_step):
                    raise ValueError(
                        f"prompt {idx} 在 target_step={args.diagnose_target_step} 前已经结束，"
                        f"当前可用 prefix steps={len(basic_trace)}"
                    )
                forced_prefix_ids = [int(step_info["token_id"]) for step_info in basic_trace[: int(args.diagnose_target_step)]]
                basic_details = capture_basic_attention_details(
                    model=basic_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    forced_token_ids=forced_prefix_ids,
                    layer_idx=int(args.diagnose_layer_idx),
                )
                infer_details = capture_infer_attention_details(
                    model=infer_model,
                    tokenizer=tokenizer,
                    text=texts[idx],
                    device=args.device,
                    forced_token_ids=forced_prefix_ids,
                    layer_idx=int(args.diagnose_layer_idx),
                )
                print_attention_diagnose_result(
                    tokenizer=tokenizer,
                    prompt_index=idx,
                    prompt=prompts[idx],
                    target_step=int(args.diagnose_target_step),
                    forced_prefix_ids=forced_prefix_ids,
                    layer_idx=int(args.diagnose_layer_idx),
                    head_idx=int(args.diagnose_head_idx),
                    basic_details=basic_details,
                    infer_details=infer_details,
                    top_k=int(args.diagnose_top_k),
                )
        finally:
            cleanup_model(basic_model)
            cleanup_model(infer_model)
        return

    if run_basic:
        basic_model = init_basic_model(args.load_from, args.device, dtype)
        if int(args.disable_flash):
            set_flash_enabled(basic_model, False)
        result = benchmark_basic_generate(
            model=basic_model,
            tokenizer=tokenizer,
            texts=texts,
            device=args.device,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        results.append(result)
        result["prompts"] = prompts
        print_summary(result)
        cleanup_model(basic_model)

    if run_basic_batch:
        batch_prompts = prompts
        batch_texts = texts
        batch_note = ""
        if int(args.auto_select_equal_length):
            batch_prompts, batch_texts, token_len = select_equal_length_texts(
                tokenizer=tokenizer,
                prompts=prompts,
                texts=texts,
                min_batch_size=int(args.min_equal_batch_size),
            )
            batch_note = (
                f"auto_selected_equal_length_batch: batch_size={len(batch_texts)}, token_len={token_len}"
            )
        basic_model_batch = init_basic_model(args.load_from, args.device, dtype)
        if int(args.disable_flash):
            set_flash_enabled(basic_model_batch, False)
        result = benchmark_basic_generate_batch(
            model=basic_model_batch,
            tokenizer=tokenizer,
            texts=batch_texts,
            device=args.device,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        results.append(result)
        result["prompts"] = batch_prompts
        result["note"] = batch_note
        print_summary(result)
        cleanup_model(basic_model_batch)

    if run_engine:
        infer_model_single = init_infer_model(args.load_from, args.device, dtype)
        if int(args.disable_flash):
            set_flash_enabled(infer_model_single, False)
        result = benchmark_engine_generate(
            model=infer_model_single,
            tokenizer=tokenizer,
            texts=texts,
            device=args.device,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        results.append(result)
        result["prompts"] = prompts
        print_summary(result)
        cleanup_model(infer_model_single)

    if run_engine_batch:
        infer_model_batch = init_infer_model(args.load_from, args.device, dtype)
        if int(args.disable_flash):
            set_flash_enabled(infer_model_batch, False)
        result = benchmark_engine_generate_batch(
            model=infer_model_batch,
            tokenizer=tokenizer,
            texts=texts,
            device=args.device,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        results.append(result)
        result["prompts"] = prompts
        print_summary(result)
        cleanup_model(infer_model_batch)

    if int(args.show_outputs):
        print_outputs(prompts, results)


if __name__ == "__main__":
    main()
