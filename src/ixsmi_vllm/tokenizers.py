from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int:
        ...

    @property
    def eos_token_id(self) -> int | None:
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        ...


class HuggingFaceTokenizer:
    """Adapter around transformers.AutoTokenizer."""

    def __init__(self, model: str, *, trust_remote_code: bool = False) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Hugging Face tokenizer requires optional dependencies. "
                "Install them with `python -m pip install -e .[hf]`."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return str(
            self._tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=True,
            )
        )
