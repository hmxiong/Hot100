import argparse
import json
import os
from contextlib import ExitStack
from types import SimpleNamespace

import torch
from accelerate import Accelerator, dispatch_model, infer_auto_device_map
from transformers import AutoModelForCausalLM
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
except Exception:
    FSDP = None
    FullStateDictConfig = None
    StateDictType = None

from lib.prune import prune_permllm_stage2_accelerate


def build_max_memory(max_memory_per_gpu, cpu_max_memory="128GiB"):
    if max_memory_per_gpu is None:
        return None
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        return {"cpu": cpu_max_memory}
    max_memory = {i: max_memory_per_gpu for i in range(gpu_count)}
    max_memory["cpu"] = cpu_max_memory
    return max_memory


def load_model_from_stage_args(stage_args, accelerator):
    # Under FSDP/accelerate multi-process launch, model sharding must be handled by FSDP.
    # Keeping HF device_map=auto here would make every rank try to map layers to the same GPUs
    # (often gpu0), which causes contention and OOM.
    use_device_map = None if accelerator.num_processes > 1 else stage_args.device_map
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "cache_dir": stage_args.cache_dir,
        "low_cpu_mem_usage": True,
        "device_map": use_device_map,
        "trust_remote_code": stage_args.trust_remote_code,
    }
    max_memory_per_gpu = getattr(stage_args, "max_memory_per_gpu", None)
    cpu_max_memory = getattr(stage_args, "cpu_max_memory", "128GiB")
    if accelerator.num_processes > 1:
        max_memory_per_gpu = None
    max_memory = build_max_memory(max_memory_per_gpu, cpu_max_memory)
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(stage_args.model, **model_kwargs)
    model.seqlen = stage_args.seqlen
    model.eval()
    return model


def print_rank_device_debug(accelerator, model, tag):
    rank = accelerator.process_index
    local_rank = os.environ.get("LOCAL_RANK", "NA")
    world_size = accelerator.num_processes
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    if torch.cuda.is_available():
        cur_idx = torch.cuda.current_device()
        cur_name = torch.cuda.get_device_name(cur_idx)
        cur_str = f"cuda:{cur_idx} ({cur_name})"
    else:
        cur_str = "cpu"
    try:
        model_dev = next(model.parameters()).device
    except StopIteration:
        model_dev = "NA"
    print(
        f"[{tag}] rank={rank}/{world_size} local_rank={local_rank} "
        f"visible={visible} current={cur_str} model_first_param={model_dev}",
        flush=True,
    )


def build_full_state_dict_for_saving(model, accelerator):
    pre_sd = model.state_dict()
    suspicious = {}
    for k, v in pre_sd.items():
        if torch.is_tensor(v) and (v.numel() == 0 or (hasattr(v, "full_tensor") and callable(getattr(v, "full_tensor")))):
            suspicious[k] = tuple(v.shape)

    fsdp_modules = []
    if FSDP is not None:
        fsdp_modules = [m for m in model.modules() if isinstance(m, FSDP)]

    with ExitStack() as stack:
        if fsdp_modules and FullStateDictConfig is not None and StateDictType is not None:
            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=accelerator.is_main_process)
            for m in fsdp_modules:
                stack.enter_context(FSDP.state_dict_type(m, StateDictType.FULL_STATE_DICT, full_cfg))
        elif fsdp_modules:
            for m in fsdp_modules:
                stack.enter_context(FSDP.summon_full_params(m, recurse=False, writeback=False))
        state_dict = model.state_dict()

    cpu_state_dict = {} if accelerator.is_main_process else None
    printed = 0
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            if accelerator.is_main_process:
                cpu_state_dict[k] = v
            continue

        full_v = v
        if hasattr(v, "full_tensor") and callable(getattr(v, "full_tensor")):
            try:
                full_v = v.full_tensor()
            except Exception:
                full_v = v

        if accelerator.is_main_process:
            if k in suspicious and printed < 100:
                print(f"[save_state] {k} {suspicious[k]} -> {tuple(full_v.shape)}", flush=True)
                printed += 1
            cpu_state_dict[k] = full_v.detach().cpu()
    return cpu_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_bundle", type=str, required=True, help="Path created by main_permllm_stage1.py")
    parser.add_argument("--save_bundle_out", type=str, default=None, help="Optional output bundle path after stage2.")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--bsz", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--save_model_out", type=str, default=None, help="Optional output directory to save model for inference.")
    parser.add_argument("--dispatch_for_infer", action="store_true", help="Run infer_auto_device_map + dispatch_model after stage2.")
    parser.add_argument("--save_device_map_out", type=str, default=None, help="Optional json path to save inferred device_map.")
    parser.add_argument("--infer_max_memory_per_gpu", type=str, default=None, help='Optional infer max memory per gpu, e.g. "22GiB".')
    parser.add_argument("--infer_cpu_max_memory", type=str, default=None, help='Optional infer max cpu memory, e.g. "128GiB".')
    args = parser.parse_args()

    accelerator = Accelerator()
    if torch.cuda.is_available() and accelerator.num_processes > 1:
        torch.cuda.set_device(accelerator.local_process_index)

    if accelerator.is_main_process:
        print(f"[Stage2] loading bundle: {args.stage1_bundle}")
    bundle = torch.load(args.stage1_bundle, map_location="cpu")
    stage_args = SimpleNamespace(**bundle["args"])
    for key in ("iters", "bsz", "lr", "weight_decay"):
        value = getattr(args, key)
        if value is not None:
            setattr(stage_args, key, value)

    if accelerator.num_processes > 1 and accelerator.is_main_process:
        print("[Stage2] FSDP mode detected: force device_map=None and let accelerate/FSDP shard parameters.")
    model = load_model_from_stage_args(stage_args, accelerator)
    print_rank_device_debug(accelerator, model, tag="before_stage2")
    stage_state = bundle["stage_state"]

    prune_permllm_stage2_accelerate(stage_args, model, stage_state, accelerator)
    accelerator.wait_for_everyone()

    if args.save_model_out:
        if accelerator.is_main_process:
            os.makedirs(args.save_model_out, exist_ok=True)
        accelerator.wait_for_everyone()
        raw_model = accelerator.unwrap_model(model)
        full_state_dict = build_full_state_dict_for_saving(raw_model, accelerator)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            raw_model.save_pretrained(args.save_model_out, state_dict=full_state_dict)
            print(f"[Stage2] full model saved to: {args.save_model_out}")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and args.save_bundle_out:
        torch.save({"args": vars(stage_args)}, args.save_bundle_out)
        print(f"[Stage2] output bundle saved to: {args.save_bundle_out}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
