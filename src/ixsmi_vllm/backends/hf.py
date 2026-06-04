from __future__ import annotations

from ixsmi_vllm.backends.base import BackendConfig, ModelBackend


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

    def next_token_logits(self, token_ids: list[int], vocab_size: int) -> list[float]:
        if not token_ids:
            raise ValueError("HuggingFaceBackend requires at least one input token")

        device = next(self.model.parameters()).device
        input_ids = self.torch.tensor([token_ids], dtype=self.torch.long, device=device)
        with self.torch.no_grad():
            output = self.model(input_ids=input_ids)
        logits = output.logits[0, -1].detach().float().cpu().tolist()
        return logits
