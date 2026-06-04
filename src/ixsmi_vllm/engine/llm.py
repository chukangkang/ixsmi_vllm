from __future__ import annotations

import random
import uuid
from typing import Iterable

from ixsmi_vllm.backends.base import BackendConfig, ModelBackend
from ixsmi_vllm.backends.corex import CoreXBackend
from ixsmi_vllm.backends.toy import ToyBackend
from ixsmi_vllm.cache import KVCacheManager
from ixsmi_vllm.outputs import CompletionOutput, RequestOutput
from ixsmi_vllm.sampler import sample_from_logits
from ixsmi_vllm.sampling_params import SamplingParams
from ixsmi_vllm.scheduler import FIFOScheduler, SchedulerConfig
from ixsmi_vllm.sequence import Sequence
from ixsmi_vllm.tokenizer import SimpleTokenizer


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
    ) -> None:
        self.config = BackendConfig(
            model=model,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer = SimpleTokenizer()
        self.backend = self._create_backend(backend)
        self.scheduler = FIFOScheduler(SchedulerConfig(max_num_seqs=max_num_seqs))
        self.cache = KVCacheManager()

    def generate(
        self,
        prompts: str | Iterable[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        params = sampling_params or SamplingParams()
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        sequences = [self._make_sequence(prompt) for prompt in prompt_list]
        self.scheduler.extend(sequences)

        results: list[RequestOutput] = []
        while batch := self.scheduler.schedule():
            for sequence in batch:
                self._decode_sequence(sequence, params)
                results.append(self._to_request_output(sequence))
                self.cache.free(sequence.request_id)
        return results

    def _create_backend(self, backend: str) -> ModelBackend:
        normalized = backend.lower()
        if normalized == "toy":
            return ToyBackend(self.config)
        if normalized in {"corex", "bi-v150s", "biv150s"}:
            return CoreXBackend(self.config)
        raise ValueError(f"Unsupported backend: {backend}")

    def _make_sequence(self, prompt: str) -> Sequence:
        return Sequence(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            prompt_token_ids=self.tokenizer.encode(prompt),
        )

    def _decode_sequence(self, sequence: Sequence, params: SamplingParams) -> None:
        rng = random.Random(params.seed)
        for _ in range(params.max_tokens):
            logits = self.backend.next_token_logits(
                sequence.all_token_ids,
                self.tokenizer.vocab_size,
            )
            next_token_id = sample_from_logits(logits, params, rng)
            sequence.generated_token_ids.append(next_token_id)
            self.cache.append(sequence.request_id, next_token_id)

            generated_text = self.tokenizer.decode(sequence.generated_token_ids)
            if self._matches_stop(generated_text, params.stop_sequences):
                sequence.finished = True
                sequence.finish_reason = "stop"
                return

        sequence.finished = True
        sequence.finish_reason = "length"

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
