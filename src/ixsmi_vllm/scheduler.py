from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from ixsmi_vllm.sequence import Sequence


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int = 8


class FIFOScheduler:
    """Minimal request scheduler.

    This establishes the extension point used later for continuous batching.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self._waiting: deque[Sequence] = deque()

    def add(self, sequence: Sequence) -> None:
        self._waiting.append(sequence)

    def extend(self, sequences: Iterable[Sequence]) -> None:
        for sequence in sequences:
            self.add(sequence)

    def schedule(self) -> list[Sequence]:
        batch: list[Sequence] = []
        while self._waiting and len(batch) < self.config.max_num_seqs:
            batch.append(self._waiting.popleft())
        return batch
