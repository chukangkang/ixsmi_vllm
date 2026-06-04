from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class SamplingParams:
    """Generation sampling options.

    The field names intentionally mirror common vLLM/OpenAI-style options so
    callers can migrate to a production engine with minimal changes.
    """

    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: Optional[int] = None
    stop: Optional[str | list[str]] = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 or a positive integer")

    @property
    def stop_sequences(self) -> list[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return list(self.stop)
