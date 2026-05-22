from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

@dataclass(slots=True)
class SampleResult:
    token_id: int
    logprob: Optional[float] = None


class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy_mask = temperatures <= 0
        safe_temperatures = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
        scaled_logits = logits / safe_temperatures.unsqueeze(dim=1)
        probs = torch.softmax(scaled_logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        greedy_tokens = torch.argmax(logits, dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)

def _apply_repetition_penalty(logits: torch.Tensor, generated_token_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty is None or penalty == 1.0 or not generated_token_ids:
        return logits
    token_ids = torch.tensor(generated_token_ids, device=logits.device, dtype=torch.long)
    token_logits = logits.index_select(dim=-1, index=token_ids)
    penalized = torch.where(token_logits < 0, token_logits * penalty, token_logits / penalty)
    logits = logits.clone()
    logits.index_copy_(dim=-1, index=token_ids, source=penalized)
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generated_token_ids: list[int],
    repetition_penalty: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> SampleResult:
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1D (vocab,), got shape={tuple(logits.shape)}")

    logits = _apply_repetition_penalty(logits, generated_token_ids, repetition_penalty)

    if temperature is None or temperature <= 0:
        token_id = int(torch.argmax(logits).item())
        return SampleResult(token_id=token_id)

    logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)

    if top_p is not None and 0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_in_sorted = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
        token_id = int(sorted_indices.gather(dim=-1, index=sampled_in_sorted).item())
        return SampleResult(token_id=token_id)

    token_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
    return SampleResult(token_id=token_id)
