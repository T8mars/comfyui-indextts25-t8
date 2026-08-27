from __future__ import annotations

import torch

from runtime.candidates import (
    combined_candidate_score,
    select_best_candidate,
    technical_audio_review,
)


def _audio(waveform: torch.Tensor) -> dict:
    return {"waveform": waveform.reshape(1, 1, -1), "sample_rate": 1000}


def test_technical_review_penalizes_clipping() -> None:
    clean = technical_audio_review(_audio(torch.sin(torch.linspace(0, 20, 1000)) * 0.4))
    clipped = technical_audio_review(_audio(torch.ones(1000)))
    assert clean["score"] > clipped["score"]
    assert clipped["clipped_ratio"] == 1.0


def test_combined_score_prefers_asr_when_available() -> None:
    assert combined_candidate_score(0.4, 0.95) > combined_candidate_score(1.0, 0.7)
    assert combined_candidate_score(0.7, None) == 0.7


def test_select_best_candidate_is_stable_on_tie() -> None:
    reviews = [
        {"combined_score": 0.8, "technical": {"score": 0.9}},
        {"combined_score": 0.8, "technical": {"score": 0.9}},
    ]
    assert select_best_candidate(reviews) == 0
