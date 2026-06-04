from __future__ import annotations

import argparse

from ixsmi_vllm import LLM, SamplingParams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ixsmi-vLLM offline generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate text for a prompt")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--model", default="toy")
    generate.add_argument(
        "--backend",
        default="toy",
        choices=["toy", "hf", "huggingface", "corex", "bi-v150s", "biv150s"],
    )
    generate.add_argument("--device", default="auto")
    generate.add_argument("--trust-remote-code", action="store_true")
    generate.add_argument("--max-tokens", type=int, default=16)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        llm = LLM(
            model=args.model,
            backend=args.backend,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        outputs = llm.generate(args.prompt, params)
        print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
