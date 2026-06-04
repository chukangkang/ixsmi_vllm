from __future__ import annotations

from ixsmi_vllm.backends.base import BackendConfig, ModelBackend
from ixsmi_vllm.backends.toy import ToyBackend


class CoreXBackend(ModelBackend):
    """BI-V150S corex.4.4.0 backend adapter placeholder.

    Replace the fallback with vendor SDK/runtime calls when the concrete Python
    API is available. Keeping this class from day one prevents engine code from
    depending on a specific execution backend.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._fallback = ToyBackend(config)
        self.available = False

    def next_token_logits(self, token_ids: list[int], vocab_size: int) -> list[float]:
        if not self.available:
            return self._fallback.next_token_logits(token_ids, vocab_size)
        raise NotImplementedError("CoreX runtime integration is not implemented yet")
