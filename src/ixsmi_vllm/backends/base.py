from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendConfig:
    model: str = "toy"
    device: str = "auto"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False


class ModelBackend(ABC):
    @abstractmethod
    def next_token_logits(self, token_ids: list[int], vocab_size: int) -> list[float]:
        """Return logits for the next token."""
