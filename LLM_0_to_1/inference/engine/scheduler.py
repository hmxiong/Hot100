from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .request import Request, RequestStatus, GenerationParams
from enum import Enum, auto
from copy import copy


@dataclass(slots=True)
class SchedulerConfig:
    max_batch_size: int = 8
    max_num_batched_tokens: int=1024
    policy: str = "fcfs"

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    counter = itertools.count()

    def __init__(self, token_ids: list[int], sampling_params: GenerationParams):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_new_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.eos_token_id = sampling_params.eos_token_id
        self.top_p = sampling_params.top_p
        self.repetition_penalty = sampling_params.repetition_penalty
    
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_generated_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens
    
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]


class SimpleScheduler:
    def __init__(self, config: SchedulerConfig | None = None):
        self._config = config or SchedulerConfig()
        self.max_num_seqs = self._config.max_batch_size
        self.max_num_batched_tokens = self._config.max_num_batched_tokens
        self.waiting = deque()
        self.running = deque()
    
    def add(self,
            seq):
        self.waiting.append(seq)
    
    def is_finished(self):
        return not self.waiting and not self.running
    
    def schedule(self):
        running_seqs = list(self.running)[: self.max_num_seqs]

        # refill / rebuild:
        # when there are free running slots and waiting requests, admit new requests
        # and rebuild KV for the whole running set using full sequence histories.
        if self.waiting and 0 < len(running_seqs) < self.max_num_seqs:
            scheduled_seqs = list(running_seqs)
            num_batched_tokens = sum(seq.num_tokens for seq in scheduled_seqs)
            admitted_new = 0
            free_slots = self.max_num_seqs - len(running_seqs)

            while self.waiting and admitted_new < free_slots:
                seq = self.waiting[0]
                remaining = self.max_num_batched_tokens - num_batched_tokens
                if remaining <= 0:
                    break
                if seq.num_tokens > remaining and scheduled_seqs:
                    break
                seq.num_scheduled_tokens = seq.num_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_seqs.append(seq)
                num_batched_tokens += seq.num_tokens
                admitted_new += 1

            if admitted_new > 0:
                for seq in running_seqs:
                    seq.num_scheduled_tokens = seq.num_tokens
                return scheduled_seqs, "rebuild"

        # initial prefill
        if self.waiting and not running_seqs:
            scheduled_seqs = []
            num_batched_tokens = 0
            while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.waiting[0]
                remaining = self.max_num_batched_tokens - num_batched_tokens
                if remaining == 0:
                    break
                num_tokens = seq.num_tokens - seq.num_cached_tokens
                if remaining < num_tokens and scheduled_seqs:
                    break
                seq.num_scheduled_tokens = min(num_tokens, remaining)
                num_batched_tokens += seq.num_scheduled_tokens
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                scheduled_seqs.append(seq)
            if scheduled_seqs:
                return scheduled_seqs, "prefill"

        # decode
        decode_seqs = list(self.running)[: self.max_num_seqs]
        for seq in decode_seqs:
            seq.num_scheduled_tokens = 1
            seq.status = SequenceStatus.RUNNING
        return decode_seqs, "decode"

    def _finish_sequence(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED
        try:
            self.running.remove(seq)
        except ValueError:
            pass

    def _maybe_finish_after_sampling(self, seq: Sequence, token_id: int) -> bool:
        if (not seq.ignore_eos) and seq.eos_token_id is not None and int(token_id) == int(seq.eos_token_id):
            self._finish_sequence(seq)
            return True
        if seq.num_generated_tokens >= int(seq.max_tokens):
            self._finish_sequence(seq)
            return True
        return False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], step_kind: str) -> None:
        if len(seqs) != len(token_ids):
            raise ValueError(f"postprocess length mismatch: len(seqs)={len(seqs)} len(token_ids)={len(token_ids)}")

        if step_kind == "prefill":
            for seq, token_id in zip(seqs, token_ids):
                seq.num_cached_tokens += int(seq.num_scheduled_tokens)
                seq.num_scheduled_tokens = 0
                if seq.status != SequenceStatus.RUNNING:
                    continue
                seq.token_ids.append(int(token_id))
                seq.last_token = int(token_id)
                seq.num_tokens += 1
                seq.is_prefill = False
                self._maybe_finish_after_sampling(seq, int(token_id))
            return

        if step_kind == "rebuild":
            for seq, token_id in zip(seqs, token_ids):
                seq.num_cached_tokens = seq.num_tokens
                seq.num_scheduled_tokens = 0
                if seq.status != SequenceStatus.RUNNING:
                    continue
                seq.token_ids.append(int(token_id))
                seq.last_token = int(token_id)
                seq.num_tokens += 1
                seq.is_prefill = False
                self._maybe_finish_after_sampling(seq, int(token_id))
            return

        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += int(seq.num_scheduled_tokens)
            seq.num_scheduled_tokens = 0
            seq.token_ids.append(int(token_id))
            seq.last_token = int(token_id)
            seq.num_tokens += 1
            self._maybe_finish_after_sampling(seq, int(token_id))

    def select_next_batch(self, waiting: list[Request], running: list[Request]) -> list[Request]:
        del running
        candidates = [r for r in waiting if r.status == RequestStatus.WAITING]
        if not candidates:
            return []

        if self._config.policy == "fcfs":
            batch = candidates[: self._config.max_batch_size]
        else:
            batch = candidates[: self._config.max_batch_size]

        for r in batch:
            r.status = RequestStatus.RUNNING
        return batch

    @staticmethod
    def iter_active(requests: Iterable[Request]) -> list[Request]:
        return [r for r in requests if not r.is_finished]
