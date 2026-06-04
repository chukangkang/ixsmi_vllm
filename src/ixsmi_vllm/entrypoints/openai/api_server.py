from __future__ import annotations

import argparse
from typing import Any

from ixsmi_vllm import LLM, SamplingParams


llm = LLM(model="toy")


def create_app() -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - exercised only without extra deps
        raise RuntimeError(
            "Install server dependencies with `python -m pip install -e .[server]`."
        ) from exc

    app = FastAPI(title="ixsmi-vLLM OpenAI-compatible API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions")
    def completions(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload.get("prompt", "")
        params = SamplingParams(
            max_tokens=int(payload.get("max_tokens", 16)),
            temperature=float(payload.get("temperature", 1.0)),
            top_p=float(payload.get("top_p", 1.0)),
            seed=payload.get("seed"),
            stop=payload.get("stop"),
        )
        outputs = llm.generate(prompt, params)
        choices = []
        for index, output in enumerate(outputs):
            completion = output.outputs[0]
            choices.append(
                {
                    "index": index,
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                }
            )
        return {
            "object": "text_completion",
            "model": llm.config.model,
            "choices": choices,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the ixsmi-vLLM API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
