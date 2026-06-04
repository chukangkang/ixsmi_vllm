from __future__ import annotations

import math
import random

from ixsmi_vllm.sampling_params import SamplingParams


def sample_from_logits(logits: list[float], params: SamplingParams, rng: random.Random) -> int:
    if not logits:
        raise ValueError("logits must not be empty")

    if params.temperature == 0:
        return max(range(len(logits)), key=logits.__getitem__)

    indexed = list(enumerate(logits))
    if params.top_k > 0:
        indexed = sorted(indexed, key=lambda item: item[1], reverse=True)[: params.top_k]

    temperature = max(params.temperature, 1e-6)
    max_logit = max(logit for _, logit in indexed)
    probs = [math.exp((logit - max_logit) / temperature) for _, logit in indexed]
    total = sum(probs)
    normalized = [(token_id, prob / total) for (token_id, _), prob in zip(indexed, probs)]

    if params.top_p < 1.0:
        cumulative = 0.0
        filtered: list[tuple[int, float]] = []
        for token_id, prob in sorted(normalized, key=lambda item: item[1], reverse=True):
            filtered.append((token_id, prob))
            cumulative += prob
            if cumulative >= params.top_p:
                break
        total = sum(prob for _, prob in filtered)
        normalized = [(token_id, prob / total) for token_id, prob in filtered]

    threshold = rng.random()
    cumulative = 0.0
    for token_id, prob in normalized:
        cumulative += prob
        if threshold <= cumulative:
            return token_id
    return normalized[-1][0]
