from __future__ import annotations

from ixsmi_vllm import LLM
from ixsmi_vllm.entrypoints.openai.api_server import (
    build_chat_completion_response,
    build_completion_response,
)


def test_completion_response_matches_openai_shape() -> None:
    llm = LLM(model="toy")
    response = build_completion_response(
        llm,
        {
            "model": "toy",
            "prompt": ["Hello", "World"],
            "max_tokens": 2,
            "temperature": 0,
        },
    )

    assert response["id"].startswith("cmpl-")
    assert response["object"] == "text_completion"
    assert response["model"] == "toy"
    assert len(response["choices"]) == 2
    assert response["choices"][0]["logprobs"] is None
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"]["completion_tokens"] == 4
    assert response["usage"]["total_tokens"] >= response["usage"]["completion_tokens"]


def test_chat_completion_response_matches_openai_shape() -> None:
    llm = LLM(model="toy")
    response = build_chat_completion_response(
        llm,
        {
            "model": "toy",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Say hello."},
            ],
            "max_tokens": 3,
            "temperature": 0,
        },
    )

    assert response["id"].startswith("chatcmpl-")
    assert response["object"] == "chat.completion"
    assert response["model"] == "toy"
    assert len(response["choices"]) == 1
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(response["choices"][0]["message"]["content"], str)
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"]["completion_tokens"] == 3
