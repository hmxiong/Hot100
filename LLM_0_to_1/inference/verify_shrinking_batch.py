import argparse
import traceback

import torch
from transformers import AutoTokenizer

from engine.engine import EngineConfig, InferenceEngine
from engine.request import GenerationParams
from models.model_infer import MiniMindForCausalLM


def default_prompts() -> list[str]:
    return [
        "请只回答一个字：好。",
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        '解释一下"光合作用"的基本过程',
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
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


def build_params(
    tokenizer,
    max_new_tokens_list: list[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> list[GenerationParams]:
    return [
        GenerationParams(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            ignore_eos=False,
        )
        for max_new_tokens in max_new_tokens_list
    ]


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
        print(f"[FAIL] 输出条数不一致: step={len(step_outputs)} reference={len(ref_outputs)}")
        return False

    ok = True
    for i, (step_item, ref_item) in enumerate(zip(step_outputs, ref_outputs)):
        if step_item["token_ids"] != ref_item["token_ids"]:
            ok = False
            print(f"[FAIL] 第 {i} 条输出不一致")
            print(f"  step token_ids: {step_item['token_ids']}")
            print(f"  ref  token_ids: {ref_item['token_ids']}")
            print(f"  step text: {step_item['text']}")
            print(f"  ref  text: {ref_item['text']}")
            break
        print(f"[OK] 第 {i} 条输出一致, tokens={len(step_item['token_ids'])}")
    return ok


def format_seq_summary(engine: InferenceEngine, seq_id: int) -> str:
    seq = None
    for item in list(engine.scheduler.waiting) + list(engine.scheduler.running):
        if item.seq_id == seq_id:
            seq = item
            break
    if seq is None:
        return f"seq_id={seq_id}"
    return (
        f"seq_id={seq.seq_id}, cached={seq.num_cached_tokens}, tokens={seq.num_tokens}, "
        f"generated={seq.num_generated_tokens}, status={seq.status.name}"
    )


def main():
    parser = argparse.ArgumentParser(description="验证 step-based shrinking batch 行为，并打印逐 step 调度过程")
    parser.add_argument("--load_from", default="/root/autodl-tmp/minimind/minimind-3", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--max_batch_size", default=8, type=int)
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument(
        "--per_prompt_max_new_tokens",
        default="",
        type=str,
        help="为每条 prompt 单独指定 max_new_tokens，使用逗号分隔，例如 4,8,16,32",
    )
    parser.add_argument("--prompts", default="", type=str, help="自定义 prompt，使用 || 分隔")
    parser.add_argument("--num_prompts", default=4, type=int)
    parser.add_argument("--show_outputs", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    prompts = [x.strip() for x in str(args.prompts).split("||") if x.strip()] if args.prompts else default_prompts()
    prompts = prompts[: int(args.num_prompts)]
    if not prompts:
        raise ValueError("prompts 为空")

    engine, tokenizer = init_engine(
        load_from=args.load_from,
        device=args.device,
        max_batch_size=int(args.max_batch_size),
    )
    if args.per_prompt_max_new_tokens:
        max_new_tokens_list = [int(x.strip()) for x in str(args.per_prompt_max_new_tokens).split(",") if x.strip()]
        if len(max_new_tokens_list) != len(prompts):
            raise ValueError(
                "per_prompt_max_new_tokens 数量必须与 prompts 数量一致，"
                f"got {len(max_new_tokens_list)} vs {len(prompts)}"
            )
    else:
        max_new_tokens_list = [int(args.max_new_tokens) for _ in prompts]
    params = build_params(
        tokenizer=tokenizer,
        max_new_tokens_list=max_new_tokens_list,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
    )

    seq_id_to_prompt_idx: dict[int, int] = {}
    for idx, (prompt, sp) in enumerate(zip(prompts, params)):
        engine.add_request(prompt, sp)
        seq = engine.scheduler.waiting[-1]
        seq_id_to_prompt_idx[int(seq.seq_id)] = idx

    print("=== Shrinking Batch Trace ===")
    print(
        f"num_prompts={len(prompts)}, max_batch_size={args.max_batch_size}, "
        f"max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}"
    )
    print(f"per_prompt_max_new_tokens={max_new_tokens_list}")
    print(f"initial waiting_seq_ids={[seq.seq_id for seq in engine.scheduler.waiting]}")
    print()

    outputs_by_seq_id: dict[int, list[int]] = {}
    shrink_happened = False
    step_idx = 0

    try:
        while not engine.is_finished():
            waiting_before = [seq.seq_id for seq in engine.scheduler.waiting]
            running_before = [seq.seq_id for seq in engine.scheduler.running]
            output, num_tokens = engine.step()
            waiting_after = [seq.seq_id for seq in engine.scheduler.waiting]
            running_after = [seq.seq_id for seq in engine.scheduler.running]
            finished_now = [int(seq_id) for seq_id, _ in output]
            phase = "prefill" if num_tokens >= 0 else "decode"
            if phase == "decode" and len(running_after) < len(running_before):
                shrink_happened = True

            print(
                f"[step {step_idx}] phase={phase} num_tokens={num_tokens} "
                f"waiting {len(waiting_before)}->{len(waiting_after)} "
                f"running {len(running_before)}->{len(running_after)} "
                f"finished_now={finished_now}"
            )
            print(f"  waiting_before={waiting_before}")
            print(f"  running_before={running_before}")
            print(f"  waiting_after ={waiting_after}")
            print(f"  running_after ={running_after}")

            if finished_now:
                for seq_id, token_ids in output:
                    outputs_by_seq_id[int(seq_id)] = [int(x) for x in token_ids]
                    prompt_idx = seq_id_to_prompt_idx[int(seq_id)]
                    print(
                        f"  finished seq_id={seq_id} prompt_idx={prompt_idx} "
                        f"generated_tokens={len(token_ids)}"
                    )

            if running_after:
                print("  running_state:")
                for seq_id in running_after:
                    print(f"    {format_seq_summary(engine, int(seq_id))}")
            print()
            step_idx += 1
    except Exception as exc:
        print("[FAIL] shrinking-batch trace 运行失败")
        print(f"{type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return

    ordered_outputs = []
    for prompt_idx in range(len(prompts)):
        target_seq_id = next(seq_id for seq_id, idx in seq_id_to_prompt_idx.items() if idx == prompt_idx)
        token_ids = outputs_by_seq_id.get(int(target_seq_id))
        if token_ids is None:
            raise RuntimeError(f"missing final output for seq_id={target_seq_id}, prompt_idx={prompt_idx}")
        ordered_outputs.append(
            {
                "token_ids": token_ids,
                "text": tokenizer.decode(token_ids),
            }
        )

    print("=== Shrinking Batch Summary ===")
    print(f"total_steps={step_idx}")
    print(f"shrink_happened={shrink_happened}")
    if not shrink_happened:
        print("note=本次样本没有出现明显 shrinking，可能是都跑满 max_new_tokens 或都未提前结束")
    print()

    if int(args.show_outputs):
        print("=== Final Outputs ===")
        for i, item in enumerate(ordered_outputs):
            print(f"[{i}] Prompt: {prompts[i]}")
            print(item["text"])
            print(f"token_ids={item['token_ids']}")
            print()

    ref_engine, ref_tokenizer = init_engine(
        load_from=args.load_from,
        device=args.device,
        max_batch_size=int(args.max_batch_size),
    )
    ref_outputs = run_reference_batch(
        engine=ref_engine,
        tokenizer=ref_tokenizer,
        prompts=prompts,
        max_new_tokens=max(max_new_tokens_list),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
    )

    ref_outputs = [
        {
            "token_ids": item["token_ids"][: max_new_tokens_list[i]],
            "text": ref_tokenizer.decode(item["token_ids"][: max_new_tokens_list[i]]),
        }
        for i, item in enumerate(ref_outputs)
    ]

    print("=== Output Check ===")
    ok = compare_outputs(ordered_outputs, ref_outputs)
    print()
    if ok:
        print("[PASS] shrinking-batch 路径最终输出与 reference engine.generate_batch 完全一致")
    else:
        print("[FAIL] shrinking-batch 路径最终输出与 reference engine.generate_batch 不一致")


if __name__ == "__main__":
    main()
