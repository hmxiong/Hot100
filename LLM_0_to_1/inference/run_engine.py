import argparse
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .engine import GenerationParams, InferenceEngine
from .engine.engine import EngineConfig

warnings.filterwarnings("ignore")


def init_model(load_from: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = AutoModelForCausalLM.from_pretrained(load_from, trust_remote_code=True)
    return model.half().eval().to(device), tokenizer


def _decode_one(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="从零推理引擎：最小 continuous batching 版本")
    parser.add_argument("--load_from", default="model", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--max_batch_size", default=4, type=int)
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.85, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    args = parser.parse_args()

    model, tokenizer = init_model(args.load_from, args.device)
    engine = InferenceEngine(model, tokenizer, EngineConfig(device=args.device, max_batch_size=args.max_batch_size))

    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        "解释一下光合作用的基本过程",
    ]

    params = GenerationParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    request_ids = [engine.add_request(p, params=params, is_pretrain=False) for p in prompts]
    buffers = {rid: "" for rid in request_ids}

    print(f"提交请求数: {len(request_ids)}; max_batch_size={args.max_batch_size}\n")
    st = time.time()
    finished = set()
    while len(finished) < len(request_ids):
        progressed = engine.step()
        if progressed == 0:
            time.sleep(0.001)
            continue

        for rid in request_ids:
            req = engine.get_request(rid)
            if req is None:
                continue
            if req.generated_ids:
                token_id = req.generated_ids[-1]
                buffers[rid] += _decode_one(tokenizer, token_id)
            if req.is_finished and rid not in finished:
                finished.add(rid)

    elapsed = time.time() - st
    total_tokens = sum(len(engine.get_request(rid).generated_ids) for rid in request_ids)
    print(f"\n总生成: {total_tokens} tokens; wall={elapsed:.2f}s; {total_tokens/elapsed:.2f} tokens/s\n")
    for i, rid in enumerate(request_ids):
        print(f"[{i}] {prompts[i]}\n{buffers[rid]}\n")


if __name__ == "__main__":
    main()

