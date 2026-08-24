"""Cache-correctness helpers that do not require Triton at import time."""

from __future__ import annotations

from typing import Any, Iterable


def reset_synthetic_prompt_cache_markers(sequences: Iterable[Any], tts_embeddings: Any) -> None:
    """Force synthetic TTS embeddings through prefill instead of a stale cache hit."""

    if tts_embeddings is not None:
        for sequence in sequences:
            sequence.num_cached_tokens = 0


__all__ = ["reset_synthetic_prompt_cache_markers"]
