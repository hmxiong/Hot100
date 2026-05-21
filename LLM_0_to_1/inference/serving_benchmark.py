import argparse
import gc
import math
import random
import time
import warnings
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from engine import GenerationParams, InferenceEngine
from engine.engine import EngineConfig
from models.model import MiniMindForCausalLM as BasicMiniMindForCausalLM
from models.model_infer import MiniMindForCausalLM

warnings.filterwarnings("ignore")


def default_prompt_pool() -> list[str]:
    return [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        '解释一下"光合作用"的基本过程',
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
        "讲一下Transformer的核心思想",
        "请写一段简短的自我介绍",
        "如何提升学习效率",
        "介绍一下中国四大发明",
    ]


def maybe_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


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


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if q <= 0:
        return float(ys[0])
    if q >= 100:
        return float(ys[-1])
    k = (len(ys) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return float(ys[f])
    return float(ys[f] * (c - k) + ys[c] * (k - f))


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _make_arrivals(num_requests: int, arrival_rate: float, arrival_process: str, seed: int) -> list[float]:
    rng = random.Random(seed)
    if num_requests <= 0:
        return []
    if num_requests == 1:
        return [0.0]
    if arrival_rate <= 0:
        return [0.0 for _ in range(num_requests)]

    arrivals = [0.0]
    for _ in range(num_requests - 1):
        if arrival_process == "poisson":
            u = max(rng.random(), 1e-12)
            dt = -math.log(u) / arrival_rate
        else:
            dt = 1.0 / arrival_rate
        arrivals.append(arrivals[-1] + dt)
    return arrivals


def _build_text(tokenizer, prompt: str, use_pretrain_prompt: bool) -> str:
    if use_pretrain_prompt:
        return (getattr(tokenizer, "bos_token", None) or "") + prompt
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        conv = [{"role": "user", "content": prompt}]
        return apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    return prompt


def init_model(load_from: str, device: str):
    # tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = MiniMindForCausalLM.from_pretrained(load_from)
    return model.half().eval().to(device)

def init_model_basic(load_from: str, device: str):
    # tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = BasicMiniMindForCausalLM.from_pretrained(load_from, trust_remote_code=True)
    return model.half().eval().to(device)

def init_tokenizer(load_from: str):
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    return tokenizer

@dataclass(slots=True)
class RequestStat:
    idx: int
    arrival_s: float
    dispatch_s: float
    first_token_s: float
    finish_s: float
    generated_tokens: int


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    wall_s: float
    total_requests: int
    total_gen_tokens: int
    request_stats: list[RequestStat]
    batch_sizes: list[int]
    baseline_alloc_mb: float
    baseline_reserved_mb: float
    peak_alloc_mb: float
    peak_reserved_mb: float
    note: str = ""


def _build_result(
    name: str,
    start_wall: float,
    request_stats: list[RequestStat],
    batch_sizes: list[int],
    baseline_alloc_mb: float,
    baseline_reserved_mb: float,
    peak_alloc_mb: float,
    peak_reserved_mb: float,
    note: str = "",
) -> BenchmarkResult:
    wall_s = max(time.time() - start_wall, 1e-12)
    total_gen_tokens = int(sum(x.generated_tokens for x in request_stats))
    return BenchmarkResult(
        name=name,
        wall_s=wall_s,
        total_requests=len(request_stats),
        total_gen_tokens=total_gen_tokens,
        request_stats=request_stats,
        batch_sizes=batch_sizes,
        baseline_alloc_mb=baseline_alloc_mb,
        baseline_reserved_mb=baseline_reserved_mb,
        peak_alloc_mb=peak_alloc_mb,
        peak_reserved_mb=peak_reserved_mb,
        note=note,
    )


def _run_generate_scheduler(
    name: str,
    model,
    tokenizer,
    device: str,
    prompts: list[str],
    arrivals: list[float],
    max_batch_size: int,
    params: GenerationParams,
    use_pretrain_prompt: bool,
    poll_sleep_s: float,
) -> BenchmarkResult:
    baseline_alloc, baseline_reserved = reset_cuda_stats(device)
    maybe_sync(device)
    start_wall = time.time()

    next_idx = 0
    queue: list[int] = []
    request_stats: list[RequestStat] = []
    batch_sizes: list[int] = []
    finished = 0

    while finished < len(prompts):
        now = time.time()
        t = now - start_wall
        while next_idx < len(prompts) and arrivals[next_idx] <= t:
            queue.append(next_idx)
            next_idx += 1

        if not queue:
            time.sleep(poll_sleep_s)
            continue

        take = 1 if name == "hf_single" else min(max_batch_size, len(queue))
        batch_indices = [queue.pop(0) for _ in range(take)]
        batch_sizes.append(len(batch_indices))
        dispatch_t = time.time()

        texts = [_build_text(tokenizer, prompts[i], use_pretrain_prompt) for i in batch_indices]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                eos_token_id=params.eos_token_id,
                do_sample=(params.temperature is not None and params.temperature > 0),
                repetition_penalty=params.repetition_penalty,
                use_cache=True,
            )
        maybe_sync(device)
        finish_t = time.time()

        attn = inputs.get("attention_mask", None)
        for row, idx in enumerate(batch_indices):
            if attn is None:
                prompt_len = int(inputs["input_ids"][row].shape[0])
            else:
                prompt_len = int(attn[row].sum().item())
            out_len = int(outputs[row].shape[0])
            gen_len = max(out_len - prompt_len, 0)
            request_stats.append(
                RequestStat(
                    idx=idx,
                    arrival_s=start_wall + arrivals[idx],
                    dispatch_s=dispatch_t,
                    first_token_s=finish_t,
                    finish_s=finish_t,
                    generated_tokens=gen_len,
                )
            )
            finished += 1

    peak_alloc, peak_reserved = collect_cuda_peaks(device)
    return _build_result(
        name=name,
        start_wall=start_wall,
        request_stats=sorted(request_stats, key=lambda x: x.idx),
        batch_sizes=batch_sizes,
        baseline_alloc_mb=baseline_alloc,
        baseline_reserved_mb=baseline_reserved,
        peak_alloc_mb=peak_alloc,
        peak_reserved_mb=peak_reserved,
        note="first_token 统计为近似值（非流式，按整次 generate 完成时刻记录）",
    )


def _run_engine_batch_scheduler(
    model,
    tokenizer,
    device: str,
    prompts: list[str],
    arrivals: list[float],
    max_batch_size: int,
    params: GenerationParams,
    use_pretrain_prompt: bool,
    poll_sleep_s: float,
) -> BenchmarkResult:
    engine = InferenceEngine(
        model,
        tokenizer,
        EngineConfig(device=device, max_batch_size=max_batch_size, use_kv_cache=True),
    )
    baseline_alloc, baseline_reserved = reset_cuda_stats(device)
    maybe_sync(device)
    start_wall = time.time()

    next_idx = 0
    queue: list[int] = []
    request_stats: list[RequestStat] = []
    batch_sizes: list[int] = []
    finished = 0

    while finished < len(prompts):
        now = time.time()
        t = now - start_wall
        while next_idx < len(prompts) and arrivals[next_idx] <= t:
            queue.append(next_idx)
            next_idx += 1

        if not queue:
            time.sleep(poll_sleep_s)
            continue

        take = min(max_batch_size, len(queue))
        batch_indices = [queue.pop(0) for _ in range(take)]
        batch_sizes.append(len(batch_indices))
        dispatch_t = time.time()

        texts = [_build_text(tokenizer, prompts[i], use_pretrain_prompt) for i in batch_indices]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            outputs = engine.generate_batch(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=params.max_new_tokens,
                eos_token_id=params.eos_token_id,
                temperature=params.temperature,
                top_p=params.top_p,
                repetition_penalty=params.repetition_penalty,
                do_sample=(params.temperature is not None and params.temperature > 0),
            )
        maybe_sync(device)
        finish_t = time.time()

        state = engine._last_batch_state
        attn = inputs.get("attention_mask", None)
        prompt_width = int(inputs["input_ids"].shape[1])
        for row, idx in enumerate(batch_indices):
            if state is not None:
                gen_len = int(state.generated_lens[row].item())
            else:
                if attn is None:
                    prompt_len = int(inputs["input_ids"][row].shape[0])
                else:
                    prompt_len = int(attn[row].sum().item())
                out_len = int(outputs[row].shape[0])
                gen_len = max(out_len - max(prompt_len, prompt_width), 0)
            request_stats.append(
                RequestStat(
                    idx=idx,
                    arrival_s=start_wall + arrivals[idx],
                    dispatch_s=dispatch_t,
                    first_token_s=finish_t,
                    finish_s=finish_t,
                    generated_tokens=gen_len,
                )
            )
            finished += 1

    peak_alloc, peak_reserved = collect_cuda_peaks(device)
    return _build_result(
        name="engine_batch",
        start_wall=start_wall,
        request_stats=sorted(request_stats, key=lambda x: x.idx),
        batch_sizes=batch_sizes,
        baseline_alloc_mb=baseline_alloc,
        baseline_reserved_mb=baseline_reserved,
        peak_alloc_mb=peak_alloc,
        peak_reserved_mb=peak_reserved,
        note="first_token 统计为近似值（非流式，按整次 generate_batch 完成时刻记录）",
    )


def _run_engine_continuous(
    model,
    tokenizer,
    device: str,
    prompts: list[str],
    arrivals: list[float],
    max_batch_size: int,
    params: GenerationParams,
    use_pretrain_prompt: bool,
    poll_sleep_s: float,
) -> BenchmarkResult:
    engine = InferenceEngine(
        model,
        tokenizer,
        EngineConfig(device=device, max_batch_size=max_batch_size, use_kv_cache=True),
    )
    required = ("add_request", "step", "get_request")
    if not all(hasattr(engine, x) for x in required):
        return BenchmarkResult(
            name="engine_continuous",
            wall_s=0.0,
            total_requests=0,
            total_gen_tokens=0,
            request_stats=[],
            batch_sizes=[],
            baseline_alloc_mb=0.0,
            baseline_reserved_mb=0.0,
            peak_alloc_mb=0.0,
            peak_reserved_mb=0.0,
            note="当前 InferenceEngine 不包含 add_request/step/get_request 接口，跳过该模式",
        )

    baseline_alloc, baseline_reserved = reset_cuda_stats(device)
    maybe_sync(device)
    start_wall = time.time()

    next_idx = 0
    request_ids: list[str] = []
    request_to_idx: dict[str, int] = {}
    arrival_wall: dict[str, float] = {}
    dispatch_wall: dict[str, float] = {}
    first_token_wall: dict[str, float] = {}
    finish_wall: dict[str, float] = {}
    finished: set[str] = set()
    generated_tokens: dict[str, int] = {}

    while len(finished) < len(prompts):
        now = time.time()
        t = now - start_wall
        while next_idx < len(prompts) and arrivals[next_idx] <= t:
            rid = engine.add_request(prompts[next_idx], params=params, is_pretrain=use_pretrain_prompt)
            request_ids.append(rid)
            request_to_idx[rid] = next_idx
            arrival_wall[rid] = start_wall + arrivals[next_idx]
            dispatch_wall[rid] = time.time()
            next_idx += 1

        progressed = int(engine.step())
        if progressed == 0:
            time.sleep(poll_sleep_s)

        for rid in request_ids:
            if rid in finished:
                continue
            req = engine.get_request(rid)
            if req is None:
                continue
            gen_len = int(len(req.generated_ids))
            if gen_len > 0 and rid not in first_token_wall:
                first_token_wall[rid] = time.time()
            if req.is_finished:
                finished.add(rid)
                finish_wall[rid] = time.time()
                generated_tokens[rid] = gen_len

    maybe_sync(device)
    peak_alloc, peak_reserved = collect_cuda_peaks(device)
    stats = []
    for rid in request_ids:
        finish_t = finish_wall.get(rid, time.time())
        first_t = first_token_wall.get(rid, finish_t)
        stats.append(
            RequestStat(
                idx=request_to_idx[rid],
                arrival_s=arrival_wall[rid],
                dispatch_s=dispatch_wall[rid],
                first_token_s=first_t,
                finish_s=finish_t,
                generated_tokens=int(generated_tokens.get(rid, 0)),
            )
        )

    return _build_result(
        name="engine_continuous",
        start_wall=start_wall,
        request_stats=sorted(stats, key=lambda x: x.idx),
        batch_sizes=[],
        baseline_alloc_mb=baseline_alloc,
        baseline_reserved_mb=baseline_reserved,
        peak_alloc_mb=peak_alloc,
        peak_reserved_mb=peak_reserved,
    )


def _print_result(result: BenchmarkResult) -> None:
    if result.total_requests == 0:
        print(f"[{result.name}] skipped: {result.note}")
        return

    latencies = [x.finish_s - x.arrival_s for x in result.request_stats]
    waits = [x.dispatch_s - x.arrival_s for x in result.request_stats]
    services = [x.finish_s - x.dispatch_s for x in result.request_stats]
    ttfts = [x.first_token_s - x.arrival_s for x in result.request_stats]
    gen_tokens = [float(x.generated_tokens) for x in result.request_stats]
    tpot_samples = []
    for x in result.request_stats:
        decode_tokens = max(int(x.generated_tokens) - 1, 0)
        decode_time = max(float(x.finish_s - x.first_token_s), 0.0)
        if decode_tokens > 0:
            tpot_samples.append(decode_time / decode_tokens)
    req_tps = result.total_requests / result.wall_s if result.wall_s > 0 else 0.0
    tok_tps = result.total_gen_tokens / result.wall_s if result.wall_s > 0 else 0.0
    peak_alloc_delta = result.peak_alloc_mb - result.baseline_alloc_mb
    peak_reserved_delta = result.peak_reserved_mb - result.baseline_reserved_mb
    avg_bs = _mean([float(x) for x in result.batch_sizes]) if result.batch_sizes else 0.0
    max_bs = max(result.batch_sizes) if result.batch_sizes else 0

    print(
        f"[{result.name}] reqs={result.total_requests}; total_gen_tokens={result.total_gen_tokens}; "
        f"wall={result.wall_s:.2f}s; req_throughput={req_tps:.2f} req/s; token_throughput={tok_tps:.2f} tok/s"
    )
    print(
        f"  latency: p50={_percentile(latencies, 50):.3f}s p90={_percentile(latencies, 90):.3f}s "
        f"p99={_percentile(latencies, 99):.3f}s mean={_mean(latencies):.3f}s"
    )
    print(
        f"  queue_wait: p50={_percentile(waits, 50):.3f}s p90={_percentile(waits, 90):.3f}s "
        f"p99={_percentile(waits, 99):.3f}s mean={_mean(waits):.3f}s"
    )
    print(
        f"  service: p50={_percentile(services, 50):.3f}s p90={_percentile(services, 90):.3f}s "
        f"p99={_percentile(services, 99):.3f}s mean={_mean(services):.3f}s"
    )
    print(
        f"  ttft: p50={_percentile(ttfts, 50):.3f}s p90={_percentile(ttfts, 90):.3f}s "
        f"p99={_percentile(ttfts, 99):.3f}s mean={_mean(ttfts):.3f}s"
    )
    print(
        f"  output_tokens: p50={_percentile(gen_tokens, 50):.1f} p90={_percentile(gen_tokens, 90):.1f} "
        f"p99={_percentile(gen_tokens, 99):.1f} mean={_mean(gen_tokens):.1f}"
    )
    print(
        f"  tpot: p50={_percentile(tpot_samples, 50):.4f}s p90={_percentile(tpot_samples, 90):.4f}s "
        f"p99={_percentile(tpot_samples, 99):.4f}s mean={_mean(tpot_samples):.4f}s"
    )
    print(
        f"  batch: count={len(result.batch_sizes)} avg={avg_bs:.2f} max={max_bs}"
    )
    print(
        f"  memory: base_alloc={result.baseline_alloc_mb:.1f}MB "
        f"base_reserved={result.baseline_reserved_mb:.1f}MB "
        f"peak_alloc={result.peak_alloc_mb:.1f}MB (+{peak_alloc_delta:.1f}MB) "
        f"peak_reserved={result.peak_reserved_mb:.1f}MB (+{peak_reserved_delta:.1f}MB)"
    )
    if result.note:
        print(f"  note: {result.note}")


def main():
    parser = argparse.ArgumentParser(description="Serving benchmark: 模拟请求到达并评估不同推理策略")
    parser.add_argument("--load_from", default="/root/autodl-tmp/minimind/minimind-3", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "hf_single", "hf_batch", "engine_batch", "engine_continuous"],
    )
    parser.add_argument("--num_requests", default=128, type=int)
    parser.add_argument("--max_batch_size", default=8, type=int)
    parser.add_argument("--arrival_rate", default=16.0, type=float, help="请求到达速率（req/s）")
    parser.add_argument("--arrival_process", default="fixed", choices=["fixed", "poisson"], type=str)
    parser.add_argument("--poll_sleep_ms", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--use_pretrain_prompt", default=0, type=int, choices=[0, 1])
    parser.add_argument("--prompts", default="", type=str, help="自定义 prompt 池，使用 || 分隔")
    args = parser.parse_args()

    tokenizer = init_tokenizer(args.load_from)

    rng = random.Random(int(args.seed))
    prompt_pool = [x.strip() for x in str(args.prompts).split("||") if x.strip()] if args.prompts else default_prompt_pool()
    if not prompt_pool:
        raise ValueError("prompt 池为空")
    chosen_prompts = [prompt_pool[rng.randrange(0, len(prompt_pool))] for _ in range(int(args.num_requests))]
    arrivals = _make_arrivals(
        num_requests=int(args.num_requests),
        arrival_rate=float(args.arrival_rate),
        arrival_process=str(args.arrival_process),
        seed=int(args.seed),
    )

    params = GenerationParams(
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    poll_sleep_s = max(int(args.poll_sleep_ms), 0) / 1000.0
    use_pretrain_prompt = bool(args.use_pretrain_prompt)

    print("=== Serving Benchmark Config ===")
    print(
        f"device={args.device}, mode={args.mode}, num_requests={args.num_requests}, "
        f"max_batch_size={args.max_batch_size}, arrival_rate={args.arrival_rate}, "
        f"arrival_process={args.arrival_process}, max_new_tokens={args.max_new_tokens}, "
        f"temperature={args.temperature}, top_p={args.top_p}, repetition_penalty={args.repetition_penalty}"
    )

    runners = []
    if args.mode == "all":
        runners = ["hf_single", "hf_batch", "engine_batch", "engine_continuous"]
    else:
        runners = [str(args.mode)]

    results: list[BenchmarkResult] = []
    for runner in runners:
        if runner in ("hf_single", "hf_batch"):
            model = init_model_basic(args.load_from, args.device)
            result = _run_generate_scheduler(
                name=runner,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                prompts=chosen_prompts,
                arrivals=arrivals,
                max_batch_size=int(args.max_batch_size),
                params=params,
                use_pretrain_prompt=use_pretrain_prompt,
                poll_sleep_s=poll_sleep_s,
            )
        elif runner == "engine_batch":
            model = init_model(args.load_from, args.device)
            result = _run_engine_batch_scheduler(
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                prompts=chosen_prompts,
                arrivals=arrivals,
                max_batch_size=int(args.max_batch_size),
                params=params,
                use_pretrain_prompt=use_pretrain_prompt,
                poll_sleep_s=poll_sleep_s,
            )
        else:
            model = init_model(args.load_from, args.device)
            result = _run_engine_continuous(
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                prompts=chosen_prompts,
                arrivals=arrivals,
                max_batch_size=int(args.max_batch_size),
                params=params,
                use_pretrain_prompt=use_pretrain_prompt,
                poll_sleep_s=poll_sleep_s,
            )
        results.append(result)
        _print_result(result)
        print()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
