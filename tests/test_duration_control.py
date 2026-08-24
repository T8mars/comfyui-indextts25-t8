from __future__ import annotations

import pytest
import torch

from indextts.utils.duration_control import allocate_target_frames, fit_waveform_length


def test_native_target_duration_helpers_are_sample_exact():
    frames, samples = allocate_target_frames(4.0, [1, 1], 22050, 256, 200)
    assert samples == 88200
    assert sum(frames) == round((samples - 4410) / 256)
    waveform = torch.ones(1, 10)
    assert fit_waveform_length(waveform, 12).shape[-1] == 12
    assert fit_waveform_length(waveform, 8).shape[-1] == 8


def test_native_target_duration_rejects_impossible_pause_budget():
    with pytest.raises(ValueError, match="too short"):
        allocate_target_frames(0.1, [1, 1, 1], 22050, 256, 200)
