from __future__ import annotations

import numpy as np
import pytest
import torch

from runtime.audio_adapter import indextts_result_to_audio, validate_comfy_audio
from runtime import reference_cache


def test_indextts_int16_result_becomes_standard_comfy_audio():
    raw = np.array([[0], [32767], [-32767]], dtype=np.int16)
    audio = indextts_result_to_audio((22050, raw))
    assert audio["sample_rate"] == 22050
    assert audio["waveform"].shape == (1, 1, 3)
    assert audio["waveform"].dtype == torch.float32
    assert float(audio["waveform"].max()) <= 1.0


def test_validate_comfy_audio_rejects_batch_and_nonfinite():
    with pytest.raises(ValueError, match="batch"):
        validate_comfy_audio({"waveform": torch.zeros(2, 1, 10), "sample_rate": 22050})
    with pytest.raises(ValueError, match="NaN"):
        validate_comfy_audio({"waveform": torch.tensor([[[float("nan")]]]), "sample_rate": 22050})


def test_reference_audio_is_resampled_truncated_and_content_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_cache, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(reference_cache, "MAX_REFERENCE_SECONDS", 0.5)
    source_rate = 16_000
    waveform = torch.linspace(-0.25, 0.25, int(source_rate * 0.75)).view(1, 1, -1)
    audio = {"waveform": waveform, "sample_rate": source_rate}

    first, notes = reference_cache.comfy_audio_to_reference_wav(audio, kind="speaker")
    second, second_notes = reference_cache.comfy_audio_to_reference_wav(audio, kind="speaker")

    import torchaudio

    cached, sample_rate = torchaudio.load(str(first))
    assert first == second
    assert notes == second_notes
    assert first.is_file()
    assert sample_rate == 22_050
    assert cached.shape == (1, int(22_050 * 0.5))
    assert any("0.5" in note for note in notes)
