from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .request import Request, RequestStatus


@dataclass(slots=True)
class SchedulerConfig:
    max_batch_size: int = 8
    policy: str = "fcfs"


class SimpleScheduler:
    def __init__(self, config: SchedulerConfig):
        self._config = config

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

