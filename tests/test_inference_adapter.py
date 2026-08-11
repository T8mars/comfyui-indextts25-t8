from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from runtime import inference_adapter
from runtime.types import EmotionConfig, ModelHandle, SamplingConfig


class FakeModel:
    def __init__(self):
        self.gr_progress = None
        self.kwargs = None
        self.qwen_loaded = False

    def ensure_qwen_emotion(self):
        self.qwen_loaded = True

    def infer(self, **kwargs):
        self.kwargs = kwargs
        if self.gr_progress:
            self.gr_progress(0.5, desc="test")
        return 22050, np.array([[0], [1000], [-1000]], dtype=np.int16)


def test_adapter_maps_all_controls_without_global_side_effects(tmp_path: Path, monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(inference_adapter, "_progress_callback", lambda: (lambda value, desc="": None))
    monkeypatch.setattr(inference_adapter.MODEL_CACHE, "acquire", lambda handle: SimpleNamespace(model=fake, lock=__import__("threading").RLock()))
    completed = []
    monkeypatch.setattr(
        inference_adapter.MODEL_CACHE,
        "done",
        lambda handle, entry, release=False: completed.append(release),
    )
    monkeypatch.setattr(
        inference_adapter,
        "comfy_audio_to_reference_wav",
        lambda audio, kind: (tmp_path / f"{kind}.wav", ()),
    )
    handle = ModelHandle(tmp_path, "cpu", False, release_after_run=True)
    audio, status = inference_adapter.run_inference(
        handle,
        {"waveform": torch.zeros(1, 1, 100), "sample_rate": 22050},
        "hello",
        "EN",
        0.75,
        123,
        EmotionConfig(mode="text", text="happy", strength=0.6),
        SamplingConfig(do_sample=True, temperature=0.7, segment_silence_ms=321),
    )
    assert fake.qwen_loaded
    assert fake.kwargs["duration_factor"] == 0.75
    assert fake.kwargs["use_emo_text"] is True
    assert fake.kwargs["emo_text"] == "happy"
    assert fake.kwargs["interval_silence"] == 321
    assert fake.kwargs["do_sample"] is True
    assert audio["waveform"].shape == (1, 1, 3)
    assert "seed=123" in status
    assert completed == [True]
