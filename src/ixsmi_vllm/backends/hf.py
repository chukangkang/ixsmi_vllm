from __future__ import annotations

from ixsmi_vllm.backends.base import (
    BackendConfig,
    DecodeResult,
    DecodeState,
    KVTensorRef,
    ModelBackend,
)


class HuggingFaceBackend(ModelBackend):
    """Causal LM backend powered by transformers + torch.

    It intentionally exposes only next-token logits so the educational engine can
    keep its own scheduler, sampler, and cache abstractions.
    """

    def __init__(self, config: BackendConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Hugging Face backend requires optional dependencies. "
                "Install them with `python -m pip install -e .[hf]`."
            ) from exc

        self.torch = torch
        model_kwargs = {"trust_remote_code": config.trust_remote_code}
        if config.device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
        self.model.eval()
        if config.device != "auto":
            self.model.to(config.device)

    def init_state(self, token_ids: list[int]) -> DecodeState:
        return DecodeState()

    def decode(self, token_ids: list[int], state: DecodeState) -> DecodeResult:
        if not token_ids:
            raise ValueError("HuggingFaceBackend requires at least one input token")

        device = next(self.model.parameters()).device
        uncached_token_ids = token_ids[state.cache_length :] if state.backend_cache is not None else token_ids
        if not uncached_token_ids:
            uncached_token_ids = [token_ids[-1]]

        input_ids = self.torch.tensor(
            [uncached_token_ids],
            dtype=self.torch.long,
            device=device,
        )
        attention_mask = self.torch.ones(
            (1, state.cache_length + len(uncached_token_ids)),
            dtype=self.torch.long,
            device=device,
        )
        with self.torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=state.backend_cache,
                use_cache=True,
            )
        logits = output.logits[0, -1].detach().float().cpu().tolist()
        past_key_values = getattr(output, "past_key_values", None)
        next_state = DecodeState(
            backend_cache=past_key_values,
            cache_length=state.cache_length + len(uncached_token_ids),
        )
        return DecodeResult(
            logits=logits,
            state=next_state,
            kv_tensors=self._latest_kv_refs(past_key_values),
        )

    def _latest_kv_refs(self, past_key_values) -> list[KVTensorRef]:
        if past_key_values is None:
            return []

        refs: list[KVTensorRef] = []
        iterable = past_key_values
        if hasattr(past_key_values, "to_legacy_cache"):
            iterable = past_key_values.to_legacy_cache()

        for layer_index, layer_cache in enumerate(iterable):
            if len(layer_cache) < 2:
                continue
            key, value = layer_cache[0], layer_cache[1]
            refs.append(
                KVTensorRef(
                    layer_index=layer_index,
                    key=key[..., -1:, :].detach(),
                    value=value[..., -1:, :].detach(),
                )
            )
        return refs
