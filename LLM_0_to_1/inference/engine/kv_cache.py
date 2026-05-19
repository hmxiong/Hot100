from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(slots=True)
class KVCacheView:
    past_key_values: Optional[object]


class KVCache:
    def allocate_for_batch(self, batch_size: int) -> None:
        del batch_size

    def get_view(self, request_id: str) -> KVCacheView:
        raise NotImplementedError

    def set_view(self, request_id: str, view: KVCacheView) -> None:
        raise NotImplementedError

    def free(self, request_id: str) -> None:
        raise NotImplementedError


class NaiveKVCache(KVCache):
    def __init__(self):
        self._store: dict[str, KVCacheView] = {}

    def get_view(self, request_id: str) -> KVCacheView:
        return self._store.get(request_id, KVCacheView(past_key_values=None))

    def set_view(self, request_id: str, view: KVCacheView) -> None:
        self._store[request_id] = view

    def free(self, request_id: str) -> None:
        self._store.pop(request_id, None)
