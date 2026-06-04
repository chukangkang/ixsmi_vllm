from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    finish_reason: str | None = None


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool = True
