from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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


class NativeDurationFakeModel(FakeModel):
    def infer(self, target_duration=None, **kwargs):
        self.target_duration = target_duration
        return super().infer(**kwargs)


def test_interrupt_check_only_uses_an_already_loaded_comfy_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    inference_adapter.throw_if_processing_interrupted()

    checks = []
    monkeypatch.setitem(
        sys.modules,
        "comfy.model_management",
        SimpleNamespace(
            throw_exception_if_processing_interrupted=lambda: checks.append(True)
        ),
    )
    inference_adapter.throw_if_processing_interrupted()
    assert checks == [True]


def _patch_plan(monkeypatch, text="hello"):
    plan = SimpleNamespace(
        chunks=(SimpleNamespace(text=text, pause_after_ms=0),),
        segments=(SimpleNamespace(),),
        max_tokens=60,
        total_pause_ms=0,
    )
    monkeypatch.setattr(inference_adapter, "build_generation_plan", lambda *args, **kwargs: plan)


def test_adapter_maps_all_controls_without_global_side_effects(tmp_path: Path, monkeypatch):
    _patch_plan(monkeypatch)
    fake = FakeModel()
    monkeypatch.setattr(inference_adapter, "_progress_callback", lambda: (lambda value, desc="": None))
    interrupt_checks = []
    monkeypatch.setattr(
        inference_adapter,
        "throw_if_processing_interrupted",
        lambda: interrupt_checks.append(True),
    )
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
    assert callable(fake.kwargs["interrupt_callback"])
    criteria = fake.kwargs["stopping_criteria"]
    assert len(criteria) == 1
    stopped = criteria(torch.ones(1, 1, dtype=torch.long), torch.zeros(1, 1))
    assert not bool(stopped.any())
    assert audio["waveform"].shape == (1, 1, 3)
    assert "seed=123" in status
    assert completed == [True]
    assert len(interrupt_checks) == 3


def test_adapter_maps_cfm_controls_and_native_target_duration(tmp_path: Path, monkeypatch):
    _patch_plan(monkeypatch)
    fake = NativeDurationFakeModel()
    monkeypatch.setattr(inference_adapter, "_progress_callback", lambda: (lambda value, desc="": None))
    monkeypatch.setattr(
        inference_adapter.MODEL_CACHE,
        "acquire",
        lambda handle: SimpleNamespace(model=fake, lock=__import__("threading").RLock()),
    )
    monkeypatch.setattr(inference_adapter.MODEL_CACHE, "done", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference_adapter,
        "comfy_audio_to_reference_wav",
        lambda audio, kind: (tmp_path / f"{kind}.wav", ()),
    )
    sampling = SamplingConfig(
        diffusion_steps=40,
        inference_cfg_rate=0.85,
        cfm_temperature=0.8,
    )
    _audio, status = inference_adapter.run_inference(
        ModelHandle(tmp_path, "cpu", False),
        {"waveform": torch.zeros(1, 1, 100), "sample_rate": 22050},
        "hello",
        "EN",
        1.0,
        99,
        sampling=sampling,
        target_duration_seconds=2.5,
    )
    assert fake.target_duration == 2.5
    assert fake.kwargs["diffusion_steps"] == 40
    assert fake.kwargs["inference_cfg_rate"] == 0.85
    assert fake.kwargs["cfm_temperature"] == 0.8
    assert "native_target=2.500s" in status


def test_legacy_core_native_duration_fallback_does_not_disable_acceleration(tmp_path: Path, monkeypatch):
    _patch_plan(monkeypatch)
    fake = FakeModel()
    monkeypatch.setattr(
        inference_adapter.MODEL_CACHE,
        "acquire",
        lambda handle: SimpleNamespace(model=fake, lock=__import__("threading").RLock()),
    )
    monkeypatch.setattr(inference_adapter.MODEL_CACHE, "done", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference_adapter,
        "comfy_audio_to_reference_wav",
        lambda audio, kind: (tmp_path / f"{kind}.wav", ()),
    )
    handle = ModelHandle(
        tmp_path,
        "cpu",
        False,
        use_torch_compile=True,
        acceleration_requested="torch_compile",
        acceleration_effective="torch_compile",
    )
    with pytest.raises(inference_adapter.NativeTargetDurationUnsupported):
        inference_adapter.run_inference(
            handle,
            {"waveform": torch.zeros(1, 1, 100), "sample_rate": 22050},
            "hello",
            "EN",
            1.0,
            99,
            target_duration_seconds=2.5,
        )
    assert handle.acceleration_effective == "torch_compile"
    assert handle.use_torch_compile is True


def test_low_vram_adapter_releases_qwen_before_speech_generation(tmp_path: Path, monkeypatch):
    _patch_plan(monkeypatch)
    fake = FakeModel()

    def ensure_qwen():
        fake.qwen_loaded = True
        fake.qwen_emo = SimpleNamespace(
            inference=lambda text: {
                "happy": 1.0,
                "angry": 0.0,
                "sad": 0.0,
                "afraid": 0.0,
                "disgusted": 0.0,
                "melancholic": 0.0,
                "surprised": 0.0,
                "calm": 0.0,
            }
        )

    fake.ensure_qwen_emotion = ensure_qwen
    monkeypatch.setattr(inference_adapter, "_progress_callback", lambda: (lambda value, desc="": None))
    monkeypatch.setattr(
        inference_adapter.MODEL_CACHE,
        "acquire",
        lambda handle: SimpleNamespace(model=fake, lock=__import__("threading").RLock()),
    )
    monkeypatch.setattr(inference_adapter.MODEL_CACHE, "done", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference_adapter,
        "comfy_audio_to_reference_wav",
        lambda audio, kind: (tmp_path / f"{kind}.wav", ()),
    )
    monkeypatch.setattr(inference_adapter.torch.cuda, "is_available", lambda: False)
    handle = ModelHandle(tmp_path, "cuda:0", True, low_vram=True)

    _audio, status = inference_adapter.run_inference(
        handle,
        {"waveform": torch.zeros(1, 1, 100), "sample_rate": 22050},
        "hello",
        "EN",
        1.0,
        1,
        EmotionConfig(mode="text", text="happy"),
    )

    assert fake.kwargs["use_emo_text"] is False
    assert fake.kwargs["emo_vector"][0] == 1.0
    assert fake.qwen_emo is None
    assert "释放 QwenEmotion" in status


def test_optional_runtime_failure_reloads_normal_mode(tmp_path: Path, monkeypatch):
    _patch_plan(monkeypatch)
    failing = FakeModel()
    normal = FakeModel()
    failing_calls = []

    def failing_infer(**kwargs):
        failing_calls.append(kwargs)
        raise RuntimeError("optional kernel failed")

    failing.infer = failing_infer
    monkeypatch.setattr(inference_adapter, "_progress_callback", lambda: (lambda value, desc="": None))
    monkeypatch.setattr(
        inference_adapter.MODEL_CACHE,
        "acquire",
        lambda handle: SimpleNamespace(
            model=failing if handle.acceleration_effective != "off" else normal,
            lock=__import__("threading").RLock(),
        ),
    )
    releases = []
    monkeypatch.setattr(
        inference_adapter.MODEL_CACHE,
        "done",
        lambda handle, entry, release=False: releases.append((handle.acceleration_effective, release)),
    )
    monkeypatch.setattr(
        inference_adapter,
        "comfy_audio_to_reference_wav",
        lambda audio, kind: (tmp_path / f"{kind}.wav", ()),
    )
    handle = ModelHandle(
        tmp_path,
        "cpu",
        False,
        use_torch_compile=True,
        acceleration_requested="torch_compile",
        acceleration_effective="torch_compile",
    )
    _audio, status = inference_adapter.run_inference(
        handle,
        {"waveform": torch.zeros(1, 1, 100), "sample_rate": 22050},
        "hello",
        "EN",
        1.0,
        7,
    )
    assert normal.kwargs is not None
    assert releases[0] == ("torch_compile", True)
    assert "自动重载普通模式" in status
    assert handle.acceleration_effective == "off"
    assert handle.use_torch_compile is False

    inference_adapter.run_inference(
        handle,
        {"waveform": torch.zeros(1, 1, 100), "sample_rate": 22050},
        "second line",
        "EN",
        1.0,
        8,
    )
    assert len(failing_calls) == 1


def test_cross_segment_slowdown_retries_only_the_suspect_segment(
    tmp_path: Path, monkeypatch
):
    texts = (
        "one two three four five six",
        "seven eight nine ten eleven twelve",
        "this tail segment becomes much too slow",
    )
    plan = SimpleNamespace(
        chunks=tuple(
            SimpleNamespace(text=text, pause_after_ms=0, pause_before_ms=0)
            for text in texts
        ),
        segments=tuple(
            SimpleNamespace(token_count=12, speech_block=index + 1)
            for index in range(3)
        ),
        max_tokens=60,
        total_pause_ms=0,
    )
    monkeypatch.setattr(
        inference_adapter, "build_generation_plan", lambda *args, **kwargs: plan
    )

    class RateGuardFakeModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.calls = []

        def infer(self, **kwargs):
            self.calls.append(kwargs)
            collector = kwargs.get("segment_collector")
            text = kwargs["text"]
            is_retry = collector is None and text == texts[-1]
            seconds = 3.0 if is_retry else (12.0 if text == texts[-1] else 2.4)
            waveform = torch.zeros(1, round(22_050 * seconds))
            if collector is not None:
                collector.append(
                    {
                        "index": 1,
                        "text": text,
                        "language": "EN",
                        "token_count": 12,
                        "sample_rate": 22_050,
                        "duration_seconds": seconds,
                        "waveform": waveform,
                    }
                )
            return 22_050, waveform

    fake = RateGuardFakeModel()
    monkeypatch.setattr(
        inference_adapter.MODEL_CACHE,
        "acquire",
        lambda handle: SimpleNamespace(
            model=fake, lock=__import__("threading").RLock()
        ),
    )
    monkeypatch.setattr(
        inference_adapter.MODEL_CACHE, "done", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        inference_adapter,
        "comfy_audio_to_reference_wav",
        lambda audio, kind: (tmp_path / f"{kind}.wav", ()),
    )

    audio, status = inference_adapter.run_inference(
        ModelHandle(tmp_path, "cpu", False),
        {"waveform": torch.zeros(1, 1, 100), "sample_rate": 22_050},
        "ignored",
        "EN",
        1.0,
        42,
        sampling=SamplingConfig(do_sample=True),
    )

    assert len(fake.calls) == 4
    assert fake.calls[-1]["text"] == texts[-1]
    assert "segment_rate_guard=" in status
    assert '"accepted":true' in status
    assert audio["waveform"].shape[-1] == round(22_050 * 7.8)
