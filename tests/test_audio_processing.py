from __future__ import annotations

import torch

from runtime.audio_processing import (
    apply_duration_policy,
    concatenate_with_pauses,
    postprocess_audio,
)


def _audio(samples=100, sample_rate=1000):
    return {"waveform": torch.linspace(-0.5, 0.5, samples).view(1, 1, -1), "sample_rate": sample_rate}


def test_duration_pad_never_trims_an_overrun():
    result, report = apply_duration_policy(_audio(200), 0.1, "pad")
    assert result["waveform"].shape[-1] == 200
    assert report["action"] == "overrun_preserved"


def test_duration_exact_pads_or_trims_to_exact_sample_count():
    padded, padded_report = apply_duration_policy(_audio(100), 0.15, "exact")
    trimmed, trimmed_report = apply_duration_policy(_audio(200), 0.1, "exact")
    assert padded["waveform"].shape[-1] == 150
    assert trimmed["waveform"].shape[-1] == 100
    assert padded_report["action"] == "padded"
    assert trimmed_report["action"] == "trimmed"
    assert float(trimmed["waveform"][..., -1]) == 0.0


def test_normalize_postprocess_is_optional_and_reports_peak():
    unchanged, off_report = postprocess_audio(_audio(), "off", 1.0)
    normalized, report = postprocess_audio(_audio(), "normalize", 1.0, -1.0)
    assert torch.equal(unchanged["waveform"], _audio()["waveform"])
    assert off_report["preset"] == "off"
    assert report["preset"] == "normalize"
    assert float(normalized["waveform"].abs().max()) < 1.0


def test_all_voice_presets_produce_finite_bounded_audio():
    source = _audio(2205, 22050)
    for preset in ("voice_clarity", "clear_narration", "deharsh", "warm"):
        result, report = postprocess_audio(source, preset, 0.75)
        assert report["preset"] == preset
        assert torch.isfinite(result["waveform"]).all()
        assert float(result["waveform"].abs().max()) <= 1.0


def test_concatenation_preserves_leading_and_between_block_pauses_exactly():
    result = concatenate_with_pauses(
        [_audio(10), _audio(20)],
        [100, 0],
        leading_pause_ms=250,
    )
    waveform = result["waveform"]
    assert waveform.shape[-1] == 380
    assert torch.count_nonzero(waveform[..., :250]) == 0
    assert torch.count_nonzero(waveform[..., 260:360]) == 0
