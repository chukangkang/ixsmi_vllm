from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ixsmi_vllm import LLM, SamplingParams


prompts = [
    "Hello, my name is",
    "天数智凯 BI-V150S 可以用于",
]

sampling_params = SamplingParams(max_tokens=8, temperature=0, seed=42)
llm = LLM(model="toy", backend="corex")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print(f"Finish reason: {output.outputs[0].finish_reason}\n")
