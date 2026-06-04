from __future__ import annotations

import hashlib

from ixsmi_vllm.backends.base import BackendConfig, DecodeResult, DecodeState, ModelBackend


class ToyBackend(ModelBackend):
    """Deterministic pseudo model used before real weights/runtime are wired in."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    def decode(self, token_ids: list[int], state: DecodeState) -> DecodeResult:
        return DecodeResult(logits=self._logits(token_ids), state=state)

    def _logits(self, token_ids: list[int]) -> list[float]:
        vocab_size = max([*token_ids, 2], default=2) + 1
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        first_regular_token_id = 3
        if vocab_size <= first_regular_token_id:
            return [0.0 for _ in range(vocab_size)]
        digest = hashlib.sha256(",".join(map(str, token_ids)).encode("utf-8")).digest()
        preferred = first_regular_token_id + (
            int.from_bytes(digest[:4], "big") % (vocab_size - first_regular_token_id)
        )
        logits = [-8.0 for _ in range(vocab_size)]
        for token_id in range(first_regular_token_id):
            logits[token_id] = -1_000_000.0
        logits[preferred] = 8.0
        return logits
