from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KVCacheBlock:
    block_id: int
    token_ids: list[int] = field(default_factory=list)


class KVCacheManager:
    """Placeholder for future paged KV cache management.

    The first implementation only records token ids per request. It gives later
    steps a stable API for replacing this with BI-V150S/corex cache buffers.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, KVCacheBlock] = {}

    def append(self, request_id: str, token_id: int) -> None:
        block = self._blocks.setdefault(
            request_id, KVCacheBlock(block_id=len(self._blocks))
        )
        block.token_ids.append(token_id)

    def get(self, request_id: str) -> list[int]:
        block = self._blocks.get(request_id)
        return [] if block is None else list(block.token_ids)

    def free(self, request_id: str) -> None:
        self._blocks.pop(request_id, None)
