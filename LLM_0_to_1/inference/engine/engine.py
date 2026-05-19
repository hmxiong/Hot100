from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import torch

from .kv_cache import KVCache, KVCacheView, NaiveKVCache
from .request import GenerationParams, Request, RequestStatus
from .sampler import sample_next_token


@dataclass(slots=True)
class EngineConfig:
    device: str = "cuda"
    max_batch_size: int = 8
    use_kv_cache: bool = False


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
            text = (getattr(self.tokenizer, "bos_token", None) or "") + prompt
        else:
            apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
            if callable(apply_chat_template):
                conv = conversation or [{"role": "user", "content": prompt}]
                text = apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True, open_thinking=bool(open_thinking)
                )
            else:
                text = prompt

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
        self._running = [r for r in self._running if not r.is_finished]

        use_cache = bool(self.config.use_kv_cache)
        buckets: dict[tuple[str, int], list[Request]] = {}

        for r in self._running:
            if r.status != RequestStatus.RUNNING or r.is_finished:
                continue
            view = self.kv_cache.get_view(r.request_id)
            if use_cache and view.past_key_values is not None:
                past_len = int(view.past_key_values[0][0].shape[1])
                key = ("decode", past_len)
            else:
                key = ("prefill", int(r.prompt_len))
            buckets.setdefault(key, []).append(r)

        for r in self._waiting:
            if r.status != RequestStatus.WAITING or r.is_finished:
                continue
            key = ("prefill", int(r.prompt_len))
            buckets.setdefault(key, []).append(r)

        if not buckets:
            return 0

        sorted_keys = sorted(
            buckets.keys(), key=lambda k: (0 if k[0] == "decode" else 1, -len(buckets[k]), k)
        )
        chosen_key = sorted_keys[0]
        chosen = buckets[chosen_key][: self.config.max_batch_size]
        chosen_ids = {c.request_id for c in chosen}

        if chosen_key[0] == "prefill":
            newly_started = []
            still_waiting = []
            for r in self._waiting:
                if r.request_id in chosen_ids:
                    r.status = RequestStatus.RUNNING
                    newly_started.append(r)
                else:
                    still_waiting.append(r)
            if newly_started:
                self._waiting = still_waiting
                running_ids = {r.request_id for r in self._running}
                for r in newly_started:
                    if r.request_id not in running_ids:
                        self._running.append(r)

        self._run_one_step(chosen)

        still_running: list[Request] = []
        for r in self._running:
            if r.is_finished:
                self._finished.append(r)
                self.kv_cache.free(r.request_id)
            else:
                still_running.append(r)
        self._running = still_running
        return len(chosen)

    def _run_one_step(self, batch: list[Request]) -> None:
        if not batch:
            return

        use_cache = bool(self.config.use_kv_cache)
        prefill: list[Request] = []
        decode: list[Request] = []
        for r in batch:
            view = self.kv_cache.get_view(r.request_id)
            if use_cache and view.past_key_values is not None:
                decode.append(r)
            else:
                prefill.append(r)

        if prefill:
            self._run_prefill(prefill, use_cache=use_cache)
        if decode:
            self._run_decode(decode, use_cache=use_cache)

    def _run_prefill(self, requests: list[Request], use_cache: bool) -> None:
        groups: dict[int, list[Request]] = {}
        for r in requests:
            groups.setdefault(int(r.prompt_len), []).append(r)

        for _, reqs in groups.items():
            input_ids = torch.cat([r.input_ids for r in reqs], dim=0)
            attention_mask = torch.cat([r.attention_mask for r in reqs], dim=0)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_values=None,
                )

            logits = outputs.logits[:, -1, :]
            if use_cache:
                self._write_past_per_request(reqs, outputs.past_key_values)

            for i, req in enumerate(reqs):
                req.last_logits = logits[i]
                token_id = sample_next_token(
                    logits=logits[i],
                    temperature=req.params.temperature,
                    top_p=req.params.top_p,
                    generated_token_ids=req.generated_ids,
                    repetition_penalty=req.params.repetition_penalty,
                ).token_id
                req.generated_ids.append(token_id)
                req.attention_mask = torch.cat(
                    [req.attention_mask, req.attention_mask.new_ones((1, 1))], dim=1
                )
                self._maybe_finish(req, token_id)

    def _run_decode(self, requests: list[Request], use_cache: bool) -> None:
        groups: dict[int, list[Request]] = {}
        for r in requests:
            view = self.kv_cache.get_view(r.request_id)
            past = view.past_key_values
            if past is None:
                continue
            past_len = int(past[0][0].shape[1])
            groups.setdefault(past_len, []).append(r)

        for past_len, reqs in groups.items():
            token_ids = [int(r.generated_ids[-1]) for r in reqs]
            input_ids = torch.tensor(token_ids, device=self.config.device, dtype=torch.long).view(-1, 1)

            attention_mask_rows = []
            batched_past = self._stack_minimind_past([self.kv_cache.get_view(r.request_id).past_key_values for r in reqs])
            for r in reqs:
                if int(r.attention_mask.shape[1]) != past_len + 1:
                    attention_mask_rows.append(torch.ones((1, past_len + 1), device=self.config.device, dtype=torch.long))
                else:
                    attention_mask_rows.append(r.attention_mask)
            attention_mask = torch.cat(attention_mask_rows, dim=0)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_values=batched_past,
                )

            logits = outputs.logits[:, -1, :]
            if use_cache:
                self._write_past_per_request(reqs, outputs.past_key_values)

            for i, req in enumerate(reqs):
                req.last_logits = logits[i]
                token_id = sample_next_token(
                    logits=logits[i],
                    temperature=req.params.temperature,
                    top_p=req.params.top_p,
                    generated_token_ids=req.generated_ids,
                    repetition_penalty=req.params.repetition_penalty,
                ).token_id
                req.generated_ids.append(token_id)
                req.attention_mask = torch.cat(
                    [req.attention_mask, req.attention_mask.new_ones((1, 1))], dim=1
                )
                self._maybe_finish(req, token_id)

    def _maybe_finish(self, req: Request, token_id: int) -> None:
        eos_id = req.params.eos_token_id
        if eos_id is None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)

        if eos_id is not None and token_id == int(eos_id):
            req.status = RequestStatus.FINISHED
            return
        if req.num_generated >= int(req.params.max_new_tokens):
            req.status = RequestStatus.FINISHED

    @staticmethod
    def _stack_minimind_past(pasts: list):
        template = next((p for p in pasts if p is not None), None)
        if template is None:
            return None
        num_layers = len(template)
        out = []
        for layer_idx in range(num_layers):
            ks = []
            vs = []
            for p in pasts:
                k, v = p[layer_idx]
                ks.append(k)
                vs.append(v)
            out.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))
        return out

    def _write_past_per_request(self, requests: list[Request], batched_past) -> None:
        if batched_past is None:
            for r in requests:
                self.kv_cache.set_view(r.request_id, KVCacheView(past_key_values=None))
            return

        batch_size = len(requests)
        num_layers = len(batched_past)
        for b, r in enumerate(requests):
            per_layer = []
            for l in range(num_layers):
                k, v = batched_past[l]
                per_layer.append((k[b : b + 1].contiguous(), v[b : b + 1].contiguous()))
            self.kv_cache.set_view(r.request_id, KVCacheView(past_key_values=per_layer))

    def run_until_complete(self, poll_interval_s: float = 0.0) -> list[Request]:
        while self._waiting or self._running:
            progressed = self.step()
            if progressed == 0 and poll_interval_s > 0:
                time.sleep(poll_interval_s)
        return list(self._finished)
