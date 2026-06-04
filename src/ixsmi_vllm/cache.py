from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KVCacheBlock:
    block_id: int
    block_size: int
    token_ids: list[int] = field(default_factory=list)

    @property
    def is_full(self) -> bool:
        return len(self.token_ids) >= self.block_size


class KVCacheManager:
    """Block-table KV cache manager inspired by PagedAttention.

    The blocks currently store token ids instead of real key/value tensors. The
    API mirrors the data structure needed later for BI-V150S/corex KV buffers:
    request id -> logical block table -> fixed-size cache blocks.
    """

    def __init__(self, block_size: int = 16) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._next_block_id = 0
        self._block_tables: dict[str, list[KVCacheBlock]] = {}

    def append(self, request_id: str, token_id: int) -> None:
        table = self._block_tables.setdefault(request_id, [])
        if not table or table[-1].is_full:
            table.append(self._allocate_block())
        block = table[-1]
        block.token_ids.append(token_id)

    def get(self, request_id: str) -> list[int]:
        return [
            token_id
            for block in self._block_tables.get(request_id, [])
            for token_id in block.token_ids
        ]

    def block_table(self, request_id: str) -> list[int]:
        return [block.block_id for block in self._block_tables.get(request_id, [])]

    def free(self, request_id: str) -> None:
        self._block_tables.pop(request_id, None)

    def _allocate_block(self) -> KVCacheBlock:
        block = KVCacheBlock(
            block_id=self._next_block_id,
            block_size=self.block_size,
        )
        self._next_block_id += 1
        return block
