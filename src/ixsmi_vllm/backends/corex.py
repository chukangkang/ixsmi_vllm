from __future__ import annotations

from typing import Protocol

from ixsmi_vllm.backends.base import BackendConfig, DecodeResult, DecodeState, ModelBackend
from ixsmi_vllm.backends.toy import ToyBackend


class CoreXRuntime(Protocol):
    """Protocol expected from a BI-V150S corex.4.4.0 runtime adapter."""

    def init_state(self, model: str) -> object:
        ...

    def decode(self, token_ids: list[int], state: object) -> DecodeResult:
        ...


class CoreXBackend(ModelBackend):
    """BI-V150S corex.4.4.0 backend adapter placeholder.

    Replace the fallback with vendor SDK/runtime calls when the concrete Python
    API is available. Keeping this class from day one prevents engine code from
    depending on a specific execution backend.
    """

    def __init__(self, config: BackendConfig, runtime: CoreXRuntime | None = None) -> None:
        self.config = config
        self.runtime = runtime
        self._fallback = ToyBackend(config)
        self.available = runtime is not None

    def init_state(self, token_ids: list[int]) -> DecodeState:
        if self.runtime is None:
            return DecodeState()
        return DecodeState(backend_cache=self.runtime.init_state(self.config.model))

    def decode(self, token_ids: list[int], state: DecodeState) -> DecodeResult:
        if self.runtime is None:
            return self._fallback.decode(token_ids, state)
        result = self.runtime.decode(token_ids, state.backend_cache)
        return DecodeResult(
            logits=result.logits,
            state=DecodeState(
                backend_cache=result.state.backend_cache,
                cache_length=result.state.cache_length,
            ),
            kv_tensors=result.kv_tensors,
        )

    def next_token_logits(self, token_ids: list[int], vocab_size: int) -> list[float]:
        return self.decode(token_ids, DecodeState()).logits
