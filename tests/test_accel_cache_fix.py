from __future__ import annotations

from indextts.accel_cache_guard import reset_synthetic_prompt_cache_markers


class _Sequence:
    def __init__(self, cached: int):
        self.num_cached_tokens = cached


def test_synthetic_prompt_forces_full_prefill():
    sequences = [_Sequence(32), _Sequence(64)]
    reset_synthetic_prompt_cache_markers(sequences, object())
    assert [item.num_cached_tokens for item in sequences] == [0, 0]


def test_token_prompt_keeps_allocated_cache_markers():
    sequences = [_Sequence(32)]
    reset_synthetic_prompt_cache_markers(sequences, None)
    assert sequences[0].num_cached_tokens == 32
