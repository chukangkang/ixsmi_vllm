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


class ContinuousBatchScheduler:
    """Step-wise scheduler that keeps unfinished requests active.

    Each call to schedule returns up to max_num_seqs active sequences for one
    decode step. Newly waiting requests are admitted as active slots become free.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self._waiting: deque[Sequence] = deque()
        self._running: list[Sequence] = []

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def add(self, sequence: Sequence) -> None:
        self._waiting.append(sequence)

    def extend(self, sequences: Iterable[Sequence]) -> None:
        for sequence in sequences:
            self.add(sequence)

    def schedule(self) -> list[Sequence]:
        self._running = [sequence for sequence in self._running if not sequence.finished]
        while self._waiting and len(self._running) < self.config.max_num_seqs:
            self._running.append(self._waiting.popleft())
        return list(self._running)

    def update(self, sequence: Sequence) -> None:
        if sequence.finished:
            self._running = [
                running for running in self._running if running.request_id != sequence.request_id
            ]
