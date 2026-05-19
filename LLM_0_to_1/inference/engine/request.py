from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class GenerationParams:
    max_new_tokens: int = 256
    temperature: float = 0.85
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None


@dataclass(slots=True)
class Request:
    request_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    params: GenerationParams
    prompt_len: int
    status: RequestStatus = RequestStatus.WAITING
    generated_ids: list[int] = field(default_factory=list)
    last_logits: Optional[torch.Tensor] = None
    error: Optional[str] = None

    @property
    def num_generated(self) -> int:
        return len(self.generated_ids)

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.CANCELLED, RequestStatus.FAILED)

