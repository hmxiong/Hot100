import argparse
import time
import warnings

import torch
from transformers import AutoTokenizer

from engine.engine import EngineConfig, InferenceEngine
from models.model_infer import MiniMindForCausalLM

warnings.filterwarnings("ignore")


def init_model(load_from: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = MiniMindForCausalLM.from_pretrained(load_from, device=device)
    return model.eval(), tokenizer


def build_text(tokenizer, prompt: str, use_chat_template: bool, open_thinking: bool) -> str:
    if use_chat_template and callable(getattr(tokenizer, "apply_chat_template", None)):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=bool(open_thinking),
        )
    return (getattr(tokenizer, "bos_token", None) or "") + prompt


def build_batch_inputs(tokenizer, texts: list[str], device: str) -> dict[str, torch.Tensor]:
    return tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)


def main():
    parser = argparse.ArgumentParser(description="使用 model_infer + InferenceEngine 的最小生成脚本")
    parser.add_argument("--load_from", default="/root/autodl-tmp/minimind/minimind-3", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--use_chat_template", default=1, type=int, choices=[0, 1])
    parser.add_argument("--open_thinking", default=0, type=int, choices=[0, 1])
    parser.add_argument("--show_speed", default=1, type=int, choices=[0, 1])
    parser.add_argument("--batch_size", default=4, type=int)
    args = parser.parse_args()

    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        '解释一下"光合作用"的基本过程',
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
    ]

    model, tokenizer = init_model(args.load_from, args.device)
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        config=EngineConfig(device=args.device, use_kv_cache=True),
    )

    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("💬: "), "")

    total_tokens = 0
    total_start = time.time()
    for prompt in prompt_iter:
        if input_mode == 0:
            print(f"💬: {prompt}")

        text = build_text(
            tokenizer=tokenizer,
            prompt=prompt,
            use_chat_template=bool(args.use_chat_template),
            open_thinking=bool(args.open_thinking),
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(args.device)

        print("🧠: ", end="")
        st = time.time()
        output_ids = engine.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=int(args.max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            do_sample=bool(float(args.temperature) > 0),
        )
        response_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        print(response)

        gen_tokens = int(response_ids.shape[0])
        total_tokens += gen_tokens
        if int(args.show_speed):
            elapsed = max(time.time() - st, 1e-6)
            print(f"\n[Speed]: {gen_tokens / elapsed:.2f} tokens/s\n")
        else:
            print()

    total_elapsed = max(time.time() - total_start, 1e-6)
    print(f"\n总生成: {total_tokens} tokens; wall={total_elapsed:.2f}s; {total_tokens / total_elapsed:.2f} tokens/s\n")

    batch_mode = int(input("[2] 自动 batch 测试\n[3] 手动 batch 输入\n其他键跳过\n") or "-1")
    if batch_mode not in (2, 3):
        return

    if batch_mode == 2:
        batch_prompts = prompts[: int(args.batch_size)]
        batch_texts = [
            build_text(
                tokenizer=tokenizer,
                prompt=prompt,
                use_chat_template=bool(args.use_chat_template),
                open_thinking=bool(args.open_thinking),
            )
            for prompt in batch_prompts
        ]
    else:
        raw = input("请输入多条 prompt，使用 || 分隔: ").strip()
        batch_prompts = [x.strip() for x in raw.split("||") if x.strip()]
        if not batch_prompts:
            print("未提供有效 prompt，跳过 batch 测试。")
            return
        batch_texts = [
            build_text(
                tokenizer=tokenizer,
                prompt=prompt,
                use_chat_template=bool(args.use_chat_template),
                open_thinking=bool(args.open_thinking),
            )
            for prompt in batch_prompts
        ]

    batch_inputs = build_batch_inputs(tokenizer, batch_texts, args.device)
    st = time.time()
    batch_output_ids = engine.generate_batch(
        input_ids=batch_inputs["input_ids"],
        attention_mask=batch_inputs.get("attention_mask"),
        max_new_tokens=int(args.max_new_tokens),
        eos_token_id=tokenizer.eos_token_id,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        do_sample=bool(float(args.temperature) > 0),
    )
    elapsed = max(time.time() - st, 1e-6)

    print("\n### Batch Results")
    batch_total_tokens = 0
    prompt_lens = batch_inputs.get("attention_mask").sum(dim=1).tolist() if "attention_mask" in batch_inputs else [int(batch_inputs["input_ids"].shape[1])] * len(batch_prompts)
    padded_prompt_len = int(batch_inputs["input_ids"].shape[1])
    for i, prompt in enumerate(batch_prompts):
        response_ids = batch_output_ids[i][padded_prompt_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        batch_total_tokens += int(response_ids.shape[0])
        print(f"[{i}] Prompt(len={int(prompt_lens[i])}): {prompt}")
        print(response)
        print()
    if int(args.show_speed):
        print(f"[Batch Speed]: {batch_total_tokens / elapsed:.2f} tokens/s")


if __name__ == "__main__":
    main()
