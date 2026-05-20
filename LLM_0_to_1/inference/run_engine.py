import argparse
import time
import warnings
import random
import math
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from engine import GenerationParams, InferenceEngine
from engine.engine import EngineConfig

from models.model import MiniMindForCausalLM

warnings.filterwarnings("ignore")


def init_model(load_from: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = MiniMindForCausalLM.from_pretrained(load_from, trust_remote_code=True)
    return model.half().eval().to(device), tokenizer


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    wall_s: float
    total_gen_tokens: int
    throughput_tps: float
    latencies_s: list[float]


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


def _make_arrivals(num_requests: int, arrival_rate: float, arrival_process: str, seed: int) -> list[float]:
    rng = random.Random(seed)
    arrivals = [0.0]
    if num_requests <= 1:
        return [0.0]
    if arrival_rate <= 0:
        return [0.0 for _ in range(num_requests)]
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


def _run_engine_continuous(
    model,
    tokenizer,
    device: str,
    prompts: list[str],
    arrivals: list[float],
    max_batch_size: int,
    params: GenerationParams,
    use_kv_cache: bool,
    use_pretrain_prompt: bool,
    poll_sleep_s: float,
) -> BenchmarkResult:
    engine = InferenceEngine(
        model,
        tokenizer,
        EngineConfig(device=device, max_batch_size=max_batch_size, use_kv_cache=use_kv_cache),
    )

    start_wall = time.time()
    next_idx = 0
    request_ids: list[str] = []
    arrival_wall: dict[str, float] = {}
    finished: set[str] = set()
    end_wall: dict[str, float] = {}

    while len(finished) < len(prompts):
        now = time.time()
        t = now - start_wall
        while next_idx < len(prompts) and arrivals[next_idx] <= t:
            rid = engine.add_request(prompts[next_idx], params=params, is_pretrain=use_pretrain_prompt)
            request_ids.append(rid)
            arrival_wall[rid] = start_wall + arrivals[next_idx]
            next_idx += 1

        progressed = engine.step()
        if progressed == 0:
            time.sleep(poll_sleep_s)

        for rid in request_ids:
            if rid in finished:
                continue
            req = engine.get_request(rid)
            if req is not None and req.is_finished:
                finished.add(rid)
                end_wall[rid] = time.time()

    total_gen_tokens = sum(len(engine.get_request(rid).generated_ids) for rid in request_ids)
    wall_s = time.time() - start_wall
    latencies = [end_wall[rid] - arrival_wall[rid] for rid in request_ids]
    return BenchmarkResult(
        name="engine_continuous",
        wall_s=wall_s,
        total_gen_tokens=total_gen_tokens,
        throughput_tps=(total_gen_tokens / wall_s) if wall_s > 0 else 0.0,
        latencies_s=latencies,
    )


def _run_generate_static(
    model,
    tokenizer,
    device: str,
    prompts: list[str],
    arrivals: list[float],
    max_batch_size: int,
    params: GenerationParams,
    use_pretrain_prompt: bool,
    batch_mode: str,
    poll_sleep_s: float,
) -> BenchmarkResult:
    start_wall = time.time()
    next_idx = 0
    queue: list[int] = []
    finished = 0
    arrival_wall: dict[int, float] = {}
    end_wall: dict[int, float] = {}
    gen_tokens: dict[int, int] = {}

    while finished < len(prompts):
        now = time.time()
        t = now - start_wall
        while next_idx < len(prompts) and arrivals[next_idx] <= t:
            queue.append(next_idx)
            arrival_wall[next_idx] = start_wall + arrivals[next_idx]
            next_idx += 1

        if not queue:
            time.sleep(poll_sleep_s)
            continue

        if batch_mode == "single":
            batch_indices = [queue.pop(0)]
        else:
            take = min(max_batch_size, len(queue))
            batch_indices = [queue.pop(0) for _ in range(take)]

        texts = [_build_text(tokenizer, prompts[i], use_pretrain_prompt) for i in batch_indices]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.inference_mode():
            out = model.generate(
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

        end_t = time.time()
        attn = inputs.get("attention_mask", None)
        for row, i in enumerate(batch_indices):
            out_len = int(out[row].shape[0])
            if attn is None:
                prompt_len = int(inputs["input_ids"][row].shape[0])
            else:
                prompt_len = int(attn[row].sum().item())
            gen_tokens[i] = out_len - prompt_len
            end_wall[i] = end_t
            finished += 1

    total_gen_tokens = sum(gen_tokens.values())
    wall_s = time.time() - start_wall
    latencies = [end_wall[i] - arrival_wall[i] for i in range(len(prompts))]
    return BenchmarkResult(
        name=f"generate_{batch_mode}",
        wall_s=wall_s,
        total_gen_tokens=total_gen_tokens,
        throughput_tps=(total_gen_tokens / wall_s) if wall_s > 0 else 0.0,
        latencies_s=latencies,
    )


def main():
    parser = argparse.ArgumentParser(description="从零推理引擎：最小 continuous batching 版本")
    parser.add_argument("--load_from", default="model", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--max_batch_size", default=4, type=int)
    parser.add_argument("--use_kv_cache", default=0, type=int, choices=[0, 1])
    parser.add_argument("--decode_during_run", default=0, type=int, choices=[0, 1])
    parser.add_argument("--mode", default="demo", type=str, choices=["demo", "compare"])
    parser.add_argument("--num_requests", default=64, type=int)
    parser.add_argument("--arrival_rate", default=8.0, type=float)
    parser.add_argument("--arrival_process", default="fixed", type=str, choices=["fixed", "poisson"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--baseline", default="static", type=str, choices=["static", "single"])
    parser.add_argument("--use_pretrain_prompt", default=1, type=int, choices=[0, 1])
    parser.add_argument("--poll_sleep_ms", default=1, type=int)
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.85, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    args = parser.parse_args()

    model, tokenizer = init_model(args.load_from, args.device)

    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    params = GenerationParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    poll_sleep_s = max(int(args.poll_sleep_ms), 0) / 1000.0
    use_pretrain_prompt = bool(args.use_pretrain_prompt)

    if args.mode == "compare":
        num_requests = int(args.num_requests)
        rng = random.Random(int(args.seed))
        chosen_prompts = [prompts[rng.randrange(0, len(prompts))] for _ in range(num_requests)]
        arrivals = _make_arrivals(
            num_requests=num_requests,
            arrival_rate=float(args.arrival_rate),
            arrival_process=str(args.arrival_process),
            seed=int(args.seed),
        )

        baseline_mode = "static" if args.baseline == "static" else "single"
        print(f"running generated static")
        r1 = _run_generate_static(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            prompts=chosen_prompts,
            arrivals=arrivals,
            max_batch_size=int(args.max_batch_size),
            params=params,
            use_pretrain_prompt=use_pretrain_prompt,
            batch_mode=baseline_mode,
            poll_sleep_s=poll_sleep_s,
        )
        print(f"running generated continuous")
        r2 = _run_engine_continuous(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            prompts=chosen_prompts,
            arrivals=arrivals,
            max_batch_size=int(args.max_batch_size),
            params=params,
            use_kv_cache=bool(args.use_kv_cache),
            use_pretrain_prompt=use_pretrain_prompt,
            poll_sleep_s=poll_sleep_s,
        )

        for r in (r1, r2):
            print(
                f"[{r.name}] total={r.total_gen_tokens} tokens; wall={r.wall_s:.2f}s; "
                f"throughput={r.throughput_tps:.2f} tokens/s; "
                f"latency_p50={_percentile(r.latencies_s, 50):.3f}s; "
                f"p90={_percentile(r.latencies_s, 90):.3f}s; "
                f"p99={_percentile(r.latencies_s, 99):.3f}s"
            )
        return

    engine = InferenceEngine(
        model,
        tokenizer,
        EngineConfig(device=args.device, max_batch_size=args.max_batch_size, use_kv_cache=bool(args.use_kv_cache)),
    )

    request_ids = [engine.add_request(p, params=params, is_pretrain=use_pretrain_prompt) for p in prompts]
    buffers = {rid: "" for rid in request_ids}
    last_lens = {rid: 0 for rid in request_ids}

    print(f"提交请求数: {len(request_ids)}; max_batch_size={args.max_batch_size}\n")
    st = time.time()
    finished = set()
    while len(finished) < len(request_ids):
        progressed = engine.step()
        if progressed == 0:
            time.sleep(poll_sleep_s)
            continue

        for rid in request_ids:
            req = engine.get_request(rid)
            if req is None:
                continue
            if args.decode_during_run:
                cur_len = len(req.generated_ids)
                if cur_len != last_lens[rid]:
                    buffers[rid] = tokenizer.decode(req.generated_ids, skip_special_tokens=True)
                    last_lens[rid] = cur_len
            if req.is_finished and rid not in finished:
                finished.add(rid)

    elapsed = time.time() - st
    total_tokens = sum(len(engine.get_request(rid).generated_ids) for rid in request_ids)
    print(f"\n总生成: {total_tokens} tokens; wall={elapsed:.2f}s; {total_tokens/elapsed:.2f} tokens/s\n")
    for i, rid in enumerate(request_ids):
        if not args.decode_during_run:
            req = engine.get_request(rid)
            if req is not None:
                buffers[rid] = tokenizer.decode(req.generated_ids, skip_special_tokens=True)
        print(f"[{i}] {prompts[i]}\n{buffers[rid]}\n")


if __name__ == "__main__":
    main()
