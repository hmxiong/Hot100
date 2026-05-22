from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional
import torch

from .kv_cache import KVCache, KVCacheView, NaiveKVCache
from .request import GenerationParams, Request, RequestStatus
from .sampler import sample_next_token
from .scheduler import SchedulerConfig, SimpleScheduler, Sequence
from .model_runner import SimpleModelRunner


@dataclass(slots=True)
class EngineConfig:
    device: str = "cuda"
    max_batch_size: int = 8
    use_kv_cache: bool = False


@dataclass(slots=True)
class _DecodeGroup:
    past_len: int
    requests: list[Request]
    past_key_values: object


@dataclass(slots=True)
class _BatchGenerateState:
    prompt_lens: torch.Tensor
    generated_lens: torch.Tensor
    finished: torch.Tensor
    eos_hit: torch.Tensor
    generated_token_ids: list[list[int]]

class InferenceEngine:
    def __init__(
        self,
        model,
        tokenizer,
        config: EngineConfig,
        kv_cache: Optional[KVCache] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.kv_cache = kv_cache or NaiveKVCache()

        self.scheduler = SimpleScheduler(
            SchedulerConfig(
                max_batch_size=int(config.max_batch_size),
            )
        )
        self.model_runner = SimpleModelRunner(model)

        self._waiting: list[Request] = []
        self._running: list[Request] = []
        self._finished: list[Request] = []
        self._requests: dict[str, Request] = {}
        self._decode_groups: list[_DecodeGroup] = []
        self._last_batch_state: _BatchGenerateState | None = None
        self._seqs: dict[str, Sequence] = {}

    def _init_batch_state(self, input_ids: torch.Tensor) -> _BatchGenerateState:
        batch_size, prompt_len = input_ids.shape
        prompt_lens = torch.full((batch_size,), int(prompt_len), device=input_ids.device, dtype=torch.long)
        return _BatchGenerateState(
            prompt_lens=prompt_lens,
            generated_lens=torch.zeros((batch_size,), device=input_ids.device, dtype=torch.long),
            finished=torch.zeros((batch_size,), device=input_ids.device, dtype=torch.bool),
            eos_hit=torch.zeros((batch_size,), device=input_ids.device, dtype=torch.bool),
            generated_token_ids=[[] for _ in range(batch_size)],
        )

    def _sample_batch_next_tokens(
        self,
        logits_last: torch.Tensor,
        state: _BatchGenerateState,
        temperature: float,
        top_p: float | None,
        repetition_penalty: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        batch_size = int(logits_last.shape[0])
        token_ids: list[int] = []
        for b in range(batch_size):
            if bool(state.finished[b].item()):
                token_ids.append(0)
                continue
            sampled = sample_next_token(
                logits=logits_last[b],
                temperature=temperature,
                top_p=top_p,
                generated_token_ids=state.generated_token_ids[b],
                repetition_penalty=repetition_penalty,
                generator=generator,
            ).token_id
            token_ids.append(int(sampled))
        return torch.tensor(token_ids, device=logits_last.device, dtype=torch.long).view(batch_size, 1)

    def _update_batch_state(
        self,
        state: _BatchGenerateState,
        next_token: torch.Tensor,
        eos_token_id: int | None,
        max_new_tokens: int,
    ) -> None:
        batch_size = int(next_token.shape[0])
        token_column = next_token.squeeze(-1)
        prev_finished = state.finished.clone()
        for b in range(batch_size):
            if bool(prev_finished[b].item()):
                continue
            token_id = int(token_column[b].item())
            state.generated_token_ids[b].append(token_id)
            state.generated_lens[b] += 1
            if eos_token_id is not None and token_id == int(eos_token_id):
                state.eos_hit[b] = True
                state.finished[b] = True
            elif int(state.generated_lens[b].item()) >= int(max_new_tokens):
                state.finished[b] = True

    @torch.inference_mode()
    def _generate_batch_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        eos_token_id: int | None = None,
        temperature: float = 0.85,
        top_p: float | None = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D (B,T), got shape={tuple(input_ids.shape)}")
        if max_new_tokens <= 0:
            return input_ids
        if temperature is None or temperature <= 0:
            do_sample = False
            temperature = 1.0

        if hasattr(self.model, "reset_kv_cache"):
            self.model.reset_kv_cache()

        generated = input_ids
        batch_size, prompt_len = generated.shape
        state = self._init_batch_state(generated)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.long)
        state.prompt_lens = attention_mask.sum(dim=1).to(dtype=torch.long)
        self._last_batch_state = state

        positions = torch.arange(prompt_len, device=generated.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        logits_last = self.model.prefill(generated, positions=positions, attention_mask=attention_mask)

        for _ in range(int(max_new_tokens)):
            active_mask = ~state.finished.clone()
            sample_temperature = float(temperature) if do_sample else 0.0
            next_token = self._sample_batch_next_tokens(
                logits_last=logits_last,
                state=state,
                temperature=sample_temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generator=generator,
            )

            if eos_token_id is not None:
                eos = int(eos_token_id)
                next_token = torch.where(
                    state.finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos), next_token
                )

            generated = torch.cat([generated, next_token], dim=-1)
            self._update_batch_state(
                state=state,
                next_token=next_token,
                eos_token_id=eos_token_id,
                max_new_tokens=int(max_new_tokens),
            )

            if bool(state.finished.all().item()):
                break

            step_positions = (state.prompt_lens + state.generated_lens - 1).clamp(min=0).view(batch_size, 1)
            logits_last = self.model.decode(next_token, positions=step_positions, active_mask=active_mask)
        
        return generated
    
    def add_request(self, prompt: str | list[int], sampling_params: GenerationParams):
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            conv = [{"role": "user", "content": prompt}]
            prompt = apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

        input_ids = self.tokenizer.encode(prompt)
        seq = Sequence(input_ids, sampling_params)
        self.scheduler.add(seq)
        request_id = str(seq.seq_id)
        self._seqs[request_id] = seq
        return request_id
    
    def is_finished(self):
        return self.scheduler.is_finished()

    def get_request(self, request_id: str) -> Request | None:
        seq = self._seqs.get(str(request_id))
        if seq is None:
            return None
        status = RequestStatus.RUNNING
        if seq.status.name == "WAITING":
            status = RequestStatus.WAITING
        elif seq.status.name == "FINISHED":
            status = RequestStatus.FINISHED
        prompt_ids = torch.tensor(seq.token_ids[: seq.num_prompt_tokens], dtype=torch.long)
        attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        return Request(
            request_id=str(request_id),
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            params=GenerationParams(
                max_new_tokens=seq.max_tokens,
                temperature=seq.temperature,
                top_p=seq.top_p,
                repetition_penalty=seq.repetition_penalty,
                eos_token_id=seq.eos_token_id,
                ignore_eos=seq.ignore_eos,
            ),
            prompt_len=seq.num_prompt_tokens,
            status=status,
            generated_ids=list(seq.completion_token_ids),
        )

    def _append_empty_model_kv_rows(self, num_new_rows: int) -> None:
        append_fn = getattr(self.model, "append_empty_kv_cache", None)
        if callable(append_fn) and int(num_new_rows) > 0:
            append_fn(int(num_new_rows))

    def _compact_model_kv_cache_for_running(self, seqs: list[Sequence]) -> None:
        compact_fn = getattr(self.model, "compact_kv_cache", None)
        if not callable(compact_fn):
            return
        running_ids = [seq.seq_id for seq in self.scheduler.running]
        if not running_ids:
            reset_fn = getattr(self.model, "reset_kv_cache", None)
            if callable(reset_fn):
                reset_fn()
            return
        id_to_old_index = {seq.seq_id: idx for idx, seq in enumerate(seqs)}
        keep_indices = [id_to_old_index[seq_id] for seq_id in running_ids if seq_id in id_to_old_index]
        compact_fn(keep_indices)
    
    def step(self):
        # 单个step需要检测batch队列中所有序列的状态，并根据状态决定处于running还是waiting
        running_before_ids = [seq.seq_id for seq in self.scheduler.running]
        seqs, step_kind = self.scheduler.schedule()
        if not seqs:
            return [], 0
        if step_kind == "rebuild":
            reset_fn = getattr(self.model, "reset_kv_cache", None)
            if callable(reset_fn):
                reset_fn()
        elif step_kind == "prefill" and running_before_ids:
            running_before_set = set(running_before_ids)
            num_new_rows = sum(1 for seq in seqs if seq.seq_id not in running_before_set)
            self._append_empty_model_kv_rows(num_new_rows)
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if step_kind != "decode" else -len(seqs)
        token_ids = self.model_runner.run(seqs, step_kind)
        self.scheduler.postprocess(seqs, token_ids, step_kind)
        if step_kind != "rebuild" and seqs:
            self._compact_model_kv_cache_for_running(seqs)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def generate_step(
            self,
            prompts: list[str] | list[list[int]],
            sampling_params: GenerationParams | list[GenerationParams],
        ) -> torch.Tensor:

        if hasattr(self.model, "reset_kv_cache"):
            self.model.reset_kv_cache()

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        
        while not self.is_finished():
            t = time.perf_counter()
            output, num_tokens = self.step()

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs


    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        eos_token_id: int | None = None,
        temperature: float = 0.85,
        top_p: float | None = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self._generate_batch_impl(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            generator=generator,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        eos_token_id: int | None = None,
        temperature: float = 0.85,
        top_p: float | None = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self._generate_batch_impl(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            generator=generator,
        )
            
        
