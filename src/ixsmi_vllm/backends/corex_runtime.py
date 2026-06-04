from __future__ import annotations

import importlib
from dataclasses import dataclass

from ixsmi_vllm.backends.base import DecodeResult, DecodeState, KVTensorRef


@dataclass
class CoreXCacheHandle:
    """Backend-specific handle for BI-V150S corex KV cache buffers."""

    model: str
    handle: object | None = None
    cache_length: int = 0


class CoreXRuntimeAdapter:
    """Adapter boundary for BI-V150S corex.4.4.0 runtime kernels.

    The concrete vendor Python package/API is not available in this workspace, so
    this class does not pretend to execute kernels. Once the SDK is installed,
    wire these methods to runtime calls such as model loading, prefill/decode,
    and KV cache block allocation.
    """

    candidate_modules = ("corex", "corex_runtime", "ixsmi_corex")

    def __init__(self) -> None:
        self.runtime = self._load_runtime_module()
        if self.runtime is None:
            raise RuntimeError(
                "BI-V150S corex.4.4.0 runtime module was not found. "
                "Install the vendor SDK and implement CoreXRuntimeAdapter.decode()."
            )

    def init_state(self, model: str) -> CoreXCacheHandle:
        return CoreXCacheHandle(model=model)

    def decode(self, token_ids: list[int], state: object) -> DecodeResult:
        if not isinstance(state, CoreXCacheHandle):
            state = CoreXCacheHandle(model="unknown", handle=state)
        raise NotImplementedError(
            "CoreX kernel invocation is not wired yet. Map this method to "
            "corex.4.4.0 prefill/decode APIs and return DecodeResult(logits, state, kv_tensors)."
        )

    def make_kv_ref(self, layer_index: int, key_buffer: object, value_buffer: object) -> KVTensorRef:
        return KVTensorRef(layer_index=layer_index, key=key_buffer, value=value_buffer)

    def _load_runtime_module(self) -> object | None:
        for module_name in self.candidate_modules:
            try:
                return importlib.import_module(module_name)
            except ImportError:
                continue
        return None
