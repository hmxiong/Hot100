import argparse
import gc
import time
import warnings

import torch
from transformers import AutoTokenizer

from engine.engine import EngineConfig, InferenceEngine
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


def init_basic_model(load_from: str, device: str, dtype: torch.dtype):
    model = BasicMiniMindForCausalLM.from_pretrained(load_from, trust_remote_code=True)
    model = model.eval().to(device=device, dtype=dtype)
    return model


def init_infer_model(load_from: str, device: str, dtype: torch.dtype):
    model = InferMiniMindForCausalLM.from_pretrained(load_from, device=device)
    model = model.eval().to(device=device, dtype=dtype)
    return model


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


def print_outputs(prompts: list[str], results: list[dict]) -> None:
    print("\n### Outputs")
    for i, prompt in enumerate(prompts):
        print(f"[{i}] Prompt: {prompt}")
        for result in results:
            print(f"{result['name']}:")
            print(result["responses"][i])
        print()


def main():
    parser = argparse.ArgumentParser(description="对比 basic.generate / engine.generate / engine.generate_batch")
    parser.add_argument("--load_from", default="/root/autodl-tmp/minimind/minimind-3", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--use_chat_template", default=1, type=int, choices=[0, 1])
    parser.add_argument("--open_thinking", default=0, type=int, choices=[0, 1])
    parser.add_argument("--num_prompts", default=4, type=int)
    parser.add_argument("--prompts", default="", type=str, help="自定义 prompt，使用 || 分隔")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "basic", "engine", "engine_batch"],
        help="选择运行哪一条路径；分别运行时可得到更干净的显存/吞吐数据",
    )
    parser.add_argument("--show_outputs", default=1, type=int, choices=[0, 1])
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
    run_engine = args.mode in ("all", "engine")
    run_engine_batch = args.mode in ("all", "engine_batch")

    if run_basic:
        basic_model = init_basic_model(args.load_from, args.device, dtype)
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
        print_summary(result)
        cleanup_model(basic_model)

    if run_engine:
        infer_model_single = init_infer_model(args.load_from, args.device, dtype)
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
        print_summary(result)
        cleanup_model(infer_model_single)

    if run_engine_batch:
        infer_model_batch = init_infer_model(args.load_from, args.device, dtype)
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
        print_summary(result)
        cleanup_model(infer_model_batch)

    if int(args.show_outputs):
        print_outputs(prompts, results)


if __name__ == "__main__":
    main()
