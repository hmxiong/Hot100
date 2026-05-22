import torch

from .scheduler import Sequence
from .sampler import Sampler

class SimpleModelRunner:

    def __init__(self, model):
        self.model = model
        self.sampler = Sampler()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda":
            return tensor.pin_memory().to(device=self.device, non_blocking=True)
        return tensor.to(device=self.device)
    
    def prepare_prefill(self, seqs: list[Sequence]):
        if not seqs:
            raise ValueError("prepare_prefill requires at least one sequence")

        batch_input_ids = []
        batch_positions = []
        batch_attention_mask = []
        max_seq_len = 0

        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            if seqlen_q <= 0:
                raise ValueError(
                    f"sequence {seq.seq_id} has invalid num_scheduled_tokens={seqlen_q} during prefill"
                )
            token_chunk = seq.token_ids[start:end]
            if len(token_chunk) != seqlen_q:
                raise ValueError(
                    f"sequence {seq.seq_id} prefill chunk length mismatch: "
                    f"expected={seqlen_q}, actual={len(token_chunk)}, start={start}, end={end}"
                )
            position_chunk = list(range(start, end))
            batch_input_ids.append(token_chunk)
            batch_positions.append(position_chunk)
            batch_attention_mask.append([1] * seqlen_q)
            max_seq_len = max(max_seq_len, seqlen_q)

        if max_seq_len <= 0:
            raise ValueError("prepare_prefill produced empty scheduled chunks")

        padded_input_ids = []
        padded_positions = []
        padded_attention_mask = []
        for token_chunk, position_chunk, attn_chunk in zip(batch_input_ids, batch_positions, batch_attention_mask):
            pad_len = max_seq_len - len(token_chunk)
            padded_input_ids.append(token_chunk + [0] * pad_len)
            padded_positions.append(position_chunk + [0] * pad_len)
            padded_attention_mask.append(attn_chunk + [0] * pad_len)

        input_ids = self._to_device(torch.tensor(padded_input_ids, dtype=torch.int64))
        positions = self._to_device(torch.tensor(padded_positions, dtype=torch.int64))
        attention_mask = self._to_device(torch.tensor(padded_attention_mask, dtype=torch.long))

        return {
            "input_ids": input_ids,
            "positions": positions,
            "attention_mask": attention_mask,
        }

    def prepare_decode(self, seqs: list[Sequence]):
        if not seqs:
            raise ValueError("prepare_decode requires at least one sequence")

        input_ids = []
        positions = []
        active_mask = []
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            if seqlen_q != 1:
                raise ValueError(
                    f"sequence {seq.seq_id} decode expects num_scheduled_tokens=1, got {seqlen_q}"
                )
            token_chunk = seq.token_ids[start:end]
            if len(token_chunk) != 1:
                raise ValueError(
                    f"sequence {seq.seq_id} decode chunk length mismatch: "
                    f"expected=1, actual={len(token_chunk)}, start={start}, end={end}"
                )
            input_ids.append(token_chunk)
            positions.append([start])
            active_mask.append(True)

        return {
            "input_ids": self._to_device(torch.tensor(input_ids, dtype=torch.int64)),
            "positions": self._to_device(torch.tensor(positions, dtype=torch.int64)),
            "active_mask": self._to_device(torch.tensor(active_mask, dtype=torch.bool)),
        }

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = self._to_device(torch.tensor(temperatures, dtype=torch.float32))
        return temperatures
    
    @torch.inference_mode()
    def run_model(self, model_inputs: dict[str, torch.Tensor], is_prefill: bool):
        if is_prefill:
            return self.model.prefill(
                input_ids=model_inputs["input_ids"],
                positions=model_inputs["positions"],
                attention_mask=model_inputs.get("attention_mask"),
            )
        return self.model.decode(
            input_ids=model_inputs["input_ids"],
            positions=model_inputs["positions"],
            active_mask=model_inputs.get("active_mask"),
        )

    def run(self, seqs: list[Sequence], is_prefill: bool):
        model_inputs = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(model_inputs, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist()
        return token_ids
