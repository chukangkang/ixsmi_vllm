from __future__ import annotations

import argparse
import time
from typing import Any
import uuid

from ixsmi_vllm import LLM, SamplingParams


def create_default_llm(
    model: str = "toy",
    backend: str = "toy",
    device: str = "auto",
    trust_remote_code: bool = False,
) -> LLM:
    return LLM(
        model=model,
        backend=backend,
        device=device,
        trust_remote_code=trust_remote_code,
    )


def create_app(llm: LLM | None = None) -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - exercised only without extra deps
        raise RuntimeError(
            "Install server dependencies with `python -m pip install -e .[server]`."
        ) from exc

    engine = llm or create_default_llm()
    app = FastAPI(title="ixsmi-vLLM OpenAI-compatible API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.config.model,
                    "object": "model",
                    "owned_by": "ixsmi-vllm",
                }
            ],
        }

    @app.post("/v1/completions")
    def completions(payload: dict[str, Any]) -> dict[str, Any]:
        return build_completion_response(engine, payload)

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        return build_chat_completion_response(engine, payload)

    return app


def build_completion_response(llm: LLM, payload: dict[str, Any]) -> dict[str, Any]:
    prompt = payload.get("prompt", "")
    prompts = _normalize_prompts(prompt)
    params = _sampling_params_from_payload(payload)
    outputs = llm.generate(prompts, params)

    choices = []
    completion_tokens = 0
    prompt_tokens = 0
    for index, output in enumerate(outputs):
        completion = output.outputs[0]
        prompt_tokens += len(output.prompt_token_ids)
        completion_tokens += len(completion.token_ids)
        choices.append(
            {
                "index": index,
                "text": completion.text,
                "logprobs": None,
                "finish_reason": _openai_finish_reason(completion.finish_reason),
            }
        )

    return {
        "id": _new_id("cmpl"),
        "object": "text_completion",
        "created": int(time.time()),
        "model": payload.get("model", llm.config.model),
        "choices": choices,
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def build_chat_completion_response(llm: LLM, payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages", [])
    prompt = _messages_to_prompt(messages)
    params = _sampling_params_from_payload(payload)
    outputs = llm.generate(prompt, params)
    output = outputs[0]
    completion = output.outputs[0]

    completion_tokens = len(completion.token_ids)
    prompt_tokens = len(output.prompt_token_ids)
    return {
        "id": _new_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", llm.config.model),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion.text,
                },
                "finish_reason": _openai_finish_reason(completion.finish_reason),
            }
        ],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _sampling_params_from_payload(payload: dict[str, Any]) -> SamplingParams:
    return SamplingParams(
        max_tokens=int(payload.get("max_tokens", 16)),
        temperature=float(payload.get("temperature", 1.0)),
        top_p=float(payload.get("top_p", 1.0)),
        top_k=int(payload.get("top_k", -1)),
        seed=payload.get("seed"),
        stop=payload.get("stop"),
    )


def _normalize_prompts(prompt: Any) -> str | list[str]:
    if isinstance(prompt, list):
        return [str(item) for item in prompt]
    return str(prompt)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(str(part.get("text", part)) for part in content)
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def _openai_finish_reason(reason: str | None) -> str | None:
    if reason == "length":
        return "length"
    if reason == "stop":
        return "stop"
    return reason


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the ixsmi-vLLM API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="toy")
    parser.add_argument("--backend", default="toy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app(
            create_default_llm(
                model=args.model,
                backend=args.backend,
                device=args.device,
                trust_remote_code=args.trust_remote_code,
            )
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
