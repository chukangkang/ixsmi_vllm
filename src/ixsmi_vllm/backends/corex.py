from __future__ import annotations

from typing import Protocol

from ixsmi_vllm.backends.base import BackendConfig, DecodeResult, DecodeState, ModelBackend
from ixsmi_vllm.backends.corex_runtime import CoreXRuntimeAdapter
from ixsmi_vllm.backends.hf import HuggingFaceBackend
from ixsmi_vllm.backends.toy import ToyBackend


class CoreXRuntime(Protocol):
    """Protocol expected from a BI-V150S corex.4.4.0 runtime adapter."""

    def init_state(self, model: str) -> object:
        ...

    def decode(self, token_ids: list[int], state: object) -> DecodeResult:
        ...


class CoreXBackend(ModelBackend):
    """BI-V150S corex.4.4.0 backend.

    CoreX is CUDA/PyTorch compatible, so real inference can run through the
    Hugging Face PyTorch backend after corex driver libraries and environment
    variables are initialized. A custom runtime may still be injected for future
    vendor-specific kernels.
    """

    def __init__(self, config: BackendConfig, runtime: CoreXRuntime | None = None) -> None:
        self.config = config
        self.runtime = runtime
        self._fallback = ToyBackend(config)
        self._delegate: ModelBackend | None = None
        if runtime is None and config.model != "toy":
            self._delegate = self._create_pytorch_delegate(config)
        self.available = runtime is not None or self._delegate is not None

    def init_state(self, token_ids: list[int]) -> DecodeState:
        if self._delegate is not None:
            return self._delegate.init_state(token_ids)
        if self.runtime is None:
            return DecodeState()
        return DecodeState(backend_cache=self.runtime.init_state(self.config.model))

    def decode(self, token_ids: list[int], state: DecodeState) -> DecodeResult:
        if self._delegate is not None:
            return self._delegate.decode(token_ids, state)
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

    def _create_pytorch_delegate(self, config: BackendConfig) -> ModelBackend:
        try:
            CoreXRuntimeAdapter()
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "CoreX backend requires BI-V150S corex.4.4.0 driver initialization. "
                "Run `ixsmi-vllm corex-info` first. If you only want a CPU/GPU "
                "PyTorch run, use `--backend hf` instead."
            ) from exc
        self._ensure_torch_cuda_available()
        delegate_config = BackendConfig(
            model=config.model,
            device="cuda" if config.device == "auto" else config.device,
            dtype=config.dtype,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=config.trust_remote_code,
        )
        return HuggingFaceBackend(delegate_config)

    def _ensure_torch_cuda_available(self) -> None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            raise RuntimeError(
                "CoreX driver libraries were loaded, but PyTorch CUDA initialization failed. "
                "Install a PyTorch build that matches BI-V150S corex.4.4.0, or run "
                "with `--backend hf --device cpu` to validate the API on CPU."
            ) from exc

        if not cuda_available:
            raise RuntimeError(
                "CoreX driver libraries were loaded, but PyTorch reports cuda is unavailable. "
                "This usually means the installed PyTorch build is not compatible with the "
                "corex.4.4.0 CUDA driver/runtime. Install the vendor-compatible PyTorch wheel, "
                "then rerun `ixsmi-vllm corex-info` and start with `--backend corex`."
            )
