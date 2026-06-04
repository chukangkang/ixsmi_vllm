import pytest

from ixsmi_vllm import LLM, SamplingParams
from ixsmi_vllm.cache import KVCacheManager
from ixsmi_vllm.tokenizers import HuggingFaceTokenizer


def test_generate_single_prompt_shape() -> None:
    llm = LLM(model="toy")
    outputs = llm.generate("Hello world", SamplingParams(max_tokens=4, temperature=0))

    assert len(outputs) == 1
    assert outputs[0].prompt == "Hello world"
    assert outputs[0].finished is True
    assert outputs[0].outputs[0].finish_reason == "length"
    assert len(outputs[0].outputs[0].token_ids) == 4


def test_generate_batch_preserves_order() -> None:
    llm = LLM(model="toy", max_num_seqs=1)
    outputs = llm.generate(["first", "second"], SamplingParams(max_tokens=1, seed=7))

    assert [output.prompt for output in outputs] == ["first", "second"]


def test_sampling_params_validation() -> None:
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=-1)

    with pytest.raises(ValueError):
        SamplingParams(top_p=0)


def test_corex_backend_falls_back_safely() -> None:
    llm = LLM(model="toy", backend="corex")
    outputs = llm.generate("BI-V150S", SamplingParams(max_tokens=2, temperature=0))

    assert len(outputs[0].outputs[0].token_ids) == 2


def test_generate_text_skips_special_tokens() -> None:
    llm = LLM(model="toy")
    outputs = llm.generate(
        "Hello, my name is",
        SamplingParams(max_tokens=8, temperature=0),
    )

    generated_text = outputs[0].outputs[0].text
    assert "<pad>" not in generated_text
    assert "<bos>" not in generated_text
    assert "<eos>" not in generated_text


def test_continuous_batching_completes_mixed_lengths() -> None:
    llm = LLM(model="toy", max_num_seqs=2)
    outputs = llm.generate(
        ["a", "b", "c"],
        SamplingParams(max_tokens=3, temperature=0),
    )

    assert [output.prompt for output in outputs] == ["a", "b", "c"]
    assert all(len(output.outputs[0].token_ids) == 3 for output in outputs)


def test_kv_cache_uses_fixed_size_blocks() -> None:
    cache = KVCacheManager(block_size=2)
    for token_id in [10, 11, 12, 13, 14]:
        cache.append("req-1", token_id)

    assert cache.get("req-1") == [10, 11, 12, 13, 14]
    assert cache.block_table("req-1") == [0, 1, 2]

    cache.free("req-1")
    assert cache.get("req-1") == []


def test_hf_tokenizer_reports_missing_optional_dependency(monkeypatch) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match=r"pip install -e \.\[hf\]"):
        HuggingFaceTokenizer("sshleifer/tiny-gpt2")
