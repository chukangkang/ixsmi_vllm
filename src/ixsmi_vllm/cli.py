from __future__ import annotations

import argparse

from ixsmi_vllm import LLM, SamplingParams
from ixsmi_vllm.backends.corex_runtime import CoreXRuntimeAdapter


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

    corex_info = subparsers.add_parser(
        "corex-info",
        help="Check BI-V150S corex.4.4.0 CUDA driver visibility",
    )
    corex_info.add_argument("--corex-root", default="/usr/local/corex-4.4.0")
    corex_info.add_argument(
        "--visible-devices",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
    )
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
    elif args.command == "corex-info":
        runtime = CoreXRuntimeAdapter(
            corex_root=args.corex_root,
            visible_devices=args.visible_devices,
        )
        info = runtime.device_info()
        if info.initialized:
            print("✅ BI-V150S corex.4.4.0 调用成功！")
            print(f"✅ 检测到 GPU 总数：{info.device_count}")
        else:
            print(f"❌ BI-V150S corex 初始化失败，错误码：{info.error_code}")
        print(f"corex_root={info.corex_root}")
        print(f"visible_devices={info.visible_devices}")


if __name__ == "__main__":
    main()
