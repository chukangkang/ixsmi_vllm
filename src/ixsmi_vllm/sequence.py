from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sequence:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str | None = None

    @property
    def all_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.generated_token_ids]
