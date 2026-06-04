from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendConfig:
    model: str = "toy"
    device: str = "auto"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False


@dataclass
class DecodeState:
    """Per-request backend decode state.

    backend_cache stores framework-specific cache objects such as Hugging Face
    past_key_values/DynamicCache or a CoreX runtime cache handle.
    """

    backend_cache: Any | None = None
    cache_length: int = 0


@dataclass
class KVTensorRef:
    """Reference to one logical K/V cache slice.

    key/value may be real tensors, device buffers, or backend-specific handles.
    The engine does not inspect their contents; it records the block mapping.
    """

    layer_index: int
    key: Any
    value: Any


@dataclass
class DecodeResult:
    logits: list[float]
    state: DecodeState
    kv_tensors: list[KVTensorRef] | None = None


class ModelBackend(ABC):
    def init_state(self, token_ids: list[int]) -> DecodeState:
        return DecodeState()

    @abstractmethod
    def decode(self, token_ids: list[int], state: DecodeState) -> DecodeResult:
        """Run one prefill/decode step and return next-token logits."""

    def next_token_logits(self, token_ids: list[int], vocab_size: int) -> list[float]:
        """Compatibility helper for older tests and simple backends."""
        return self.decode(token_ids, DecodeState()).logits
