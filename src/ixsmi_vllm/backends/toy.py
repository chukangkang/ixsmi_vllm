from __future__ import annotations

import hashlib

from ixsmi_vllm.backends.base import BackendConfig, ModelBackend


class ToyBackend(ModelBackend):
    """Deterministic pseudo model used before real weights/runtime are wired in."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    def next_token_logits(self, token_ids: list[int], vocab_size: int) -> list[float]:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        digest = hashlib.sha256(",".join(map(str, token_ids)).encode("utf-8")).digest()
        preferred = int.from_bytes(digest[:4], "big") % vocab_size
        logits = [-8.0 for _ in range(vocab_size)]
        logits[preferred] = 8.0
        # Keep EOS reachable for stop-like behavior in tests/future demos.
        if vocab_size > 2:
            logits[2] = max(logits[2], -1.0)
        return logits
