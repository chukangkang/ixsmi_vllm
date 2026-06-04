from __future__ import annotations

import random
import uuid
from typing import Iterable

from ixsmi_vllm.backends.base import BackendConfig, DecodeState, ModelBackend
from ixsmi_vllm.backends.corex import CoreXBackend
from ixsmi_vllm.backends.hf import HuggingFaceBackend
from ixsmi_vllm.backends.toy import ToyBackend
from ixsmi_vllm.cache import KVCacheManager
from ixsmi_vllm.outputs import CompletionOutput, RequestOutput
from ixsmi_vllm.sampler import sample_from_logits
from ixsmi_vllm.sampling_params import SamplingParams
from ixsmi_vllm.scheduler import ContinuousBatchScheduler, SchedulerConfig
from ixsmi_vllm.sequence import Sequence
from ixsmi_vllm.tokenizer import SimpleTokenizer
from ixsmi_vllm.tokenizers import HuggingFaceTokenizer, Tokenizer


class LLM:
    """Small vLLM-style offline inference facade.

    Parameters mirror vLLM where possible, while unsupported values are stored in
    BackendConfig for future backend integration.
    """

    def __init__(
        self,
        model: str = "toy",
        *,
        backend: str = "toy",
        device: str = "auto",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 8,
        block_size: int = 16,
        trust_remote_code: bool = False,
    ) -> None:
        self.config = BackendConfig(
            model=model,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = self._create_tokenizer(backend, trust_remote_code)
        self.backend = self._create_backend(backend)
        self.scheduler = ContinuousBatchScheduler(SchedulerConfig(max_num_seqs=max_num_seqs))
        self.cache = KVCacheManager(block_size=block_size)

    def generate(
        self,
        prompts: str | Iterable[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        params = sampling_params or SamplingParams()
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        sequences = [self._make_sequence(prompt) for prompt in prompt_list]
        self.scheduler.extend(sequences)
        rngs = {
            sequence.request_id: random.Random(
                None if params.seed is None else params.seed + index
            )
            for index, sequence in enumerate(sequences)
        }
        decode_states = {
            sequence.request_id: self.backend.init_state(sequence.prompt_token_ids)
            for sequence in sequences
        }

        results: list[RequestOutput] = []
        while self.scheduler.has_unfinished_requests:
            batch = self.scheduler.schedule()
            for sequence in batch:
                decode_states[sequence.request_id] = self._decode_step(
                    sequence,
                    params,
                    rngs[sequence.request_id],
                    decode_states[sequence.request_id],
                )
                self.scheduler.update(sequence)
                if not sequence.finished:
                    continue
                results.append(self._to_request_output(sequence))
                self.cache.free(sequence.request_id)
                decode_states.pop(sequence.request_id, None)
        return results

    def _create_tokenizer(self, backend: str, trust_remote_code: bool) -> Tokenizer:
        normalized = backend.lower()
        if normalized in {"hf", "huggingface", "corex", "bi-v150s", "biv150s"} and self.config.model != "toy":
            return HuggingFaceTokenizer(
                self.config.model,
                trust_remote_code=trust_remote_code,
            )
        return SimpleTokenizer()

    def _create_backend(self, backend: str) -> ModelBackend:
        normalized = backend.lower()
        if normalized == "toy":
            return ToyBackend(self.config)
        if normalized in {"hf", "huggingface"}:
            return HuggingFaceBackend(self.config)
        if normalized in {"corex", "bi-v150s", "biv150s"}:
            return CoreXBackend(self.config)
        raise ValueError(f"Unsupported backend: {backend}")

    def _make_sequence(self, prompt: str) -> Sequence:
        return Sequence(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            prompt_token_ids=self.tokenizer.encode(prompt),
        )

    def _decode_step(
        self,
        sequence: Sequence,
        params: SamplingParams,
        rng: random.Random,
        decode_state: DecodeState,
    ) -> DecodeState:
        if len(sequence.generated_token_ids) >= params.max_tokens:
            sequence.finished = True
            sequence.finish_reason = "length"
            return decode_state

        decode_result = self.backend.decode(sequence.all_token_ids, decode_state)
        logits = decode_result.logits
        next_token_id = sample_from_logits(logits, params, rng)
        sequence.generated_token_ids.append(next_token_id)
        self.cache.append(sequence.request_id, next_token_id, decode_result.kv_tensors)

        if self.tokenizer.eos_token_id is not None and next_token_id == self.tokenizer.eos_token_id:
            sequence.finished = True
            sequence.finish_reason = "stop"
            return decode_result.state

        generated_text = self.tokenizer.decode(sequence.generated_token_ids)
        if self._matches_stop(generated_text, params.stop_sequences):
            sequence.finished = True
            sequence.finish_reason = "stop"
            return decode_result.state

        if len(sequence.generated_token_ids) >= params.max_tokens:
            sequence.finished = True
            sequence.finish_reason = "length"
        return decode_result.state

    def _matches_stop(self, text: str, stop_sequences: list[str]) -> bool:
        return any(stop and stop in text for stop in stop_sequences)

    def _to_request_output(self, sequence: Sequence) -> RequestOutput:
        return RequestOutput(
            request_id=sequence.request_id,
            prompt=sequence.prompt,
            prompt_token_ids=sequence.prompt_token_ids,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=self.tokenizer.decode(sequence.generated_token_ids),
                    token_ids=sequence.generated_token_ids,
                    finish_reason=sequence.finish_reason,
                )
            ],
            finished=sequence.finished,
        )
