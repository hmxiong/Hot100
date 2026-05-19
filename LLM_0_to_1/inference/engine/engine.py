from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import torch

from .kv_cache import KVCache, KVCacheView, NaiveKVCache
from .request import GenerationParams, Request, RequestStatus
from .sampler import sample_next_token
from .scheduler import SchedulerConfig, SimpleScheduler


@dataclass(slots=True)
class EngineConfig:
    device: str = "cuda"
    max_batch_size: int = 8
    scheduler_policy: str = "fcfs"


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
            SchedulerConfig(max_batch_size=config.max_batch_size, policy=config.scheduler_policy)
        )

        self._waiting: list[Request] = []
        self._running: list[Request] = []
        self._finished: list[Request] = []

    def add_request(
        self,
        prompt: str,
        params: GenerationParams,
        request_id: Optional[str] = None,
        conversation: Optional[list[dict]] = None,
        is_pretrain: bool = False,
        open_thinking: bool = False,
    ) -> str:
        rid = request_id or uuid.uuid4().hex
        if is_pretrain:
            text = (self.tokenizer.bos_token or "") + prompt
        else:
            conv = conversation or [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, open_thinking=bool(open_thinking)
            )

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.config.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.config.device)
        req = Request(
            request_id=rid,
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=params,
            prompt_len=int(input_ids.shape[-1]),
        )
        self._waiting.append(req)
        return rid

    def get_request(self, request_id: str) -> Optional[Request]:
        for r in self._waiting + self._running + self._finished:
            if r.request_id == request_id:
                return r
        return None

    def step(self) -> int:
        batch = self.scheduler.select_next_batch(self._waiting, self._running)
        if not batch:
            self._running = self.scheduler.iter_active(self._running)
            return 0

        self._waiting = [r for r in self._waiting if r.status != RequestStatus.RUNNING]
        self._running.extend(batch)

        self._run_decode_step(batch)

        still_running: list[Request] = []
        for r in self._running:
            if r.is_finished:
                self._finished.append(r)
                self.kv_cache.free(r.request_id)
            else:
                still_running.append(r)
        self._running = still_running
        return len(batch)

    def _run_decode_step(self, batch: list[Request]) -> None:
        if not batch:
            return

        input_ids, attention_mask = self._build_batched_inputs(batch)
        past_key_values = self._build_batched_past(batch)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

        logits = outputs.logits[:, -1, :]
        new_past = self._to_legacy_cache_if_possible(outputs.past_key_values)

        for i, req in enumerate(batch):
            req.last_logits = logits[i]
            token_id = sample_next_token(
                logits=logits[i],
                temperature=req.params.temperature,
                top_p=req.params.top_p,
                generated_token_ids=req.generated_ids,
                repetition_penalty=req.params.repetition_penalty,
            ).token_id

            req.generated_ids.append(token_id)
            eos_id = req.params.eos_token_id
            if eos_id is None:
                eos_id = self.tokenizer.eos_token_id

            if eos_id is not None and token_id == int(eos_id):
                req.status = RequestStatus.FINISHED
            elif req.num_generated >= int(req.params.max_new_tokens):
                req.status = RequestStatus.FINISHED

        self._scatter_past_to_requests(batch, new_past)

    @staticmethod
    def _to_legacy_cache_if_possible(past_key_values):
        if past_key_values is None:
            return None
        to_legacy = getattr(past_key_values, "to_legacy_cache", None)
        if callable(to_legacy):
            return to_legacy()
        return past_key_values

    def _build_batched_inputs(self, batch: list[Request]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(int(r.input_ids.shape[-1] + r.num_generated) for r in batch)
        batch_size = len(batch)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        input_ids = torch.full((batch_size, max_len), int(pad_id), device=self.config.device, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), device=self.config.device, dtype=torch.long)

        for i, r in enumerate(batch):
            full = self._get_full_sequence(r)
            seq_len = int(full.shape[-1])
            input_ids[i, :seq_len] = full
            attention_mask[i, :seq_len] = 1

        return input_ids, attention_mask

    def _get_full_sequence(self, r: Request) -> torch.Tensor:
        if not r.generated_ids:
            return r.input_ids[0]
        gen = torch.tensor(r.generated_ids, device=self.config.device, dtype=torch.long)
        return torch.cat([r.input_ids[0], gen], dim=0)

    def _build_batched_past(self, batch: list[Request]):
        views = [self.kv_cache.get_view(r.request_id) for r in batch]
        if all(v.past_key_values is None for v in views):
            return None

        pkvs = [self._to_legacy_cache_if_possible(v.past_key_values) for v in views]
        return self._stack_past_key_values(pkvs)

    def _stack_past_key_values(self, pkvs: list[Optional[tuple]]):
        template = next((p for p in pkvs if p is not None), None)
        if template is None:
            return None

        num_layers = len(template)
        out = []
        for layer_idx in range(num_layers):
            keys = []
            values = []
            for p in pkvs:
                if p is None:
                    k0, v0 = template[layer_idx]
                    empty_k = torch.zeros_like(k0[:1, :, :0, :])
                    empty_v = torch.zeros_like(v0[:1, :, :0, :])
                    keys.append(empty_k)
                    values.append(empty_v)
                    continue
                k, v = p[layer_idx]
                keys.append(k)
                values.append(v)
            out.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        return tuple(out)

    def _scatter_past_to_requests(self, batch: list[Request], batched_past):
        if batched_past is None:
            return

        batch_split = getattr(batched_past, "batch_split", None)
        if callable(batch_split):
            per_request = batch_split(batch_size=len(batch))
            for req, pkv in zip(batch, per_request):
                self.kv_cache.set_view(req.request_id, KVCacheView(past_key_values=pkv))
            return

        per_request = self._unstack_past_key_values(batched_past, batch_size=len(batch))
        for req, pkv in zip(batch, per_request):
            self.kv_cache.set_view(req.request_id, KVCacheView(past_key_values=pkv))

    def _unstack_past_key_values(self, batched_past, batch_size: int) -> list[tuple]:
        num_layers = len(batched_past)
        outs = [None] * batch_size
        for b in range(batch_size):
            layers = []
            for l in range(num_layers):
                k, v = batched_past[l]
                layers.append((k[b : b + 1].contiguous(), v[b : b + 1].contiguous()))
            outs[b] = tuple(layers)
        return outs

    def run_until_complete(self, poll_interval_s: float = 0.0) -> list[Request]:
        while self._waiting or self._running:
            progressed = self.step()
            if progressed == 0 and poll_interval_s > 0:
                time.sleep(poll_interval_s)
        return list(self._finished)
