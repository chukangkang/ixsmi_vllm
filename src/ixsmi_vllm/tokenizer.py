from __future__ import annotations

import re
from collections import OrderedDict


class SimpleTokenizer:
    """A tiny deterministic tokenizer for framework bring-up tests.

    It is deliberately simple: words and punctuation become tokens, and unknown
    tokens are added to an in-memory vocabulary. Production work should replace
    this with a model tokenizer adapter.
    """

    def __init__(self) -> None:
        self._token_to_id: OrderedDict[str, int] = OrderedDict()
        self._id_to_token: dict[int, str] = {}
        for token in ["<pad>", "<bos>", "<eos>"]:
            self._add_token(token)

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def encode(self, text: str) -> list[int]:
        pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return [self._add_token(piece) for piece in pieces]

    def decode(self, token_ids: list[int]) -> str:
        pieces = [self._id_to_token[token_id] for token_id in token_ids]
        text = ""
        for piece in pieces:
            if not text or re.match(r"[^\w\s]", piece, flags=re.UNICODE):
                text += piece
            else:
                text += " " + piece
        return text

    def token_for_id(self, token_id: int) -> str:
        return self._id_to_token[token_id]

    def _add_token(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = len(self._token_to_id)
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]
