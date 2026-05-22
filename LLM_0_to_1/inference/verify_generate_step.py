import argparse
import traceback

import torch
from transformers import AutoTokenizer

from engine.engine import EngineConfig, InferenceEngine
from engine.request import GenerationParams
from models.model_infer import MiniMindForCausalLM


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


def build_text(tokenizer, prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        conv = [{"role": "user", "content": prompt}]
        return apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    return prompt


def init_engine(load_from: str, device: str, max_batch_size: int) -> tuple[InferenceEngine, object]:
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = MiniMindForCausalLM.from_pretrained(load_from)
    model = model.half().eval().to(device)
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        config=EngineConfig(device=device, max_batch_size=max_batch_size, use_kv_cache=True),
    )
    return engine, tokenizer


def normalize_generate_step_outputs(outputs) -> list[dict]:
    if not isinstance(outputs, list):
        raise TypeError(f"generate_step outputs must be a list, got {type(outputs)}")

    normalized = []
    for i, item in enumerate(outputs):
        if isinstance(item, dict):
            token_ids = item.get("token_ids")
            text = item.get("text")
            if token_ids is None:
                raise ValueError(f"generate_step outputs[{i}] missing token_ids")
            normalized.append(
                {
                    "token_ids": [int(x) for x in token_ids],
                    "text": str(text) if text is not None else "",
                }
            )
            continue
        raise TypeError(f"unsupported generate_step output item type at index {i}: {type(item)}")
    return normalized


def run_reference_batch(
    engine: InferenceEngine,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> list[dict]:
    texts = [build_text(tokenizer, prompt) for prompt in prompts]
    batch_inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(engine.config.device)
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
    state = engine._last_batch_state
    if state is None:
        raise RuntimeError("reference engine.generate_batch did not populate _last_batch_state")

    prompt_width = int(batch_inputs["input_ids"].shape[1])
    normalized = []
    for i in range(outputs.shape[0]):
        gen_len = int(state.generated_lens[i].item())
        response_ids = outputs[i][prompt_width : prompt_width + gen_len].tolist()
        normalized.append(
            {
                "token_ids": [int(x) for x in response_ids],
                "text": tokenizer.decode(response_ids),
            }
        )
    return normalized


def compare_outputs(step_outputs: list[dict], ref_outputs: list[dict]) -> bool:
    if len(step_outputs) != len(ref_outputs):
        print(f"[FAIL] 输出条数不一致: generate_step={len(step_outputs)} reference={len(ref_outputs)}")
        return False

    all_match = True
    for i, (step_item, ref_item) in enumerate(zip(step_outputs, ref_outputs)):
        step_ids = step_item["token_ids"]
        ref_ids = ref_item["token_ids"]
        if step_ids != ref_ids:
            all_match = False
            print(f"[FAIL] 第 {i} 条输出 token_ids 不一致")
            print(f"  generate_step: {step_ids}")
            print(f"  reference    : {ref_ids}")
            print(f"  generate_step text: {step_item['text']}")
            print(f"  reference text    : {ref_item['text']}")
            break
        print(f"[OK] 第 {i} 条输出一致, tokens={len(step_ids)}")
    return all_match


def main():
    parser = argparse.ArgumentParser(description="验证 InferenceEngine.generate_step 的输出是否正确")
    parser.add_argument("--load_from", default="/root/autodl-tmp/minimind/minimind-3", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--max_batch_size", default=8, type=int)
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--prompts", default="", type=str, help="自定义 prompt，使用 || 分隔")
    parser.add_argument("--num_prompts", default=4, type=int)
    args = parser.parse_args()

    prompts = [x.strip() for x in str(args.prompts).split("||") if x.strip()] if args.prompts else default_prompts()
    prompts = prompts[: int(args.num_prompts)]
    if not prompts:
        raise ValueError("prompts 为空")

    params = [
        GenerationParams(
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            eos_token_id=None,
            pad_token_id=None,
            ignore_eos=False,
        )
        for _ in prompts
    ]

    step_engine, tokenizer = init_engine(
        load_from=args.load_from,
        device=args.device,
        max_batch_size=int(args.max_batch_size),
    )
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    for p in params:
        p.eos_token_id = eos_token_id
        p.pad_token_id = pad_token_id

    print("=== Verify generate_step ===")
    print(
        f"num_prompts={len(prompts)}, max_batch_size={args.max_batch_size}, "
        f"max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}"
    )

    try:
        step_outputs_raw = step_engine.generate_step(prompts, params)
        step_outputs = normalize_generate_step_outputs(step_outputs_raw)
    except Exception as exc:
        print("[FAIL] generate_step 运行失败")
        print(f"{type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return

    ref_engine, ref_tokenizer = init_engine(
        load_from=args.load_from,
        device=args.device,
        max_batch_size=int(args.max_batch_size),
    )
    ref_outputs = run_reference_batch(
        engine=ref_engine,
        tokenizer=ref_tokenizer,
        prompts=prompts,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
    )

    ok = compare_outputs(step_outputs, ref_outputs)
    print()
    if ok:
        print("[PASS] generate_step 输出与 reference engine.generate_batch 完全一致")
    else:
        print("[FAIL] generate_step 输出与 reference engine.generate_batch 不一致")


if __name__ == "__main__":
    main()
