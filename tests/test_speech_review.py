from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from runtime import speech_review


def test_review_transcript_and_waveform_only_asr(monkeypatch):
    assert speech_review.review_transcript("重庆银行！", "重庆银行", "ZH", 0.9)[
        "passed"
    ]
    assert not speech_review.review_transcript("第一句话", "不同内容", "ZH", 0.8)[
        "passed"
    ]

    captured = {}

    class FakeWhisper:
        def transcribe(self, audio, **kwargs):
            captured["audio"] = audio
            captured.update(kwargs)
            return {"text": "hello", "language": "en", "segments": [{}]}

    monkeypatch.setattr(
        speech_review, "load_asr_model", lambda *args, **kwargs: (FakeWhisper(), "cpu")
    )
    monkeypatch.setattr(
        speech_review, "resolve_asr_backend", lambda _backend: "openai_whisper"
    )
    result = speech_review.transcribe_waveform(
        torch.zeros(2, 22050), 22050, language="EN", model_name="tiny"
    )
    assert isinstance(captured["audio"], np.ndarray)
    assert len(captured["audio"]) == 16000
    assert result["text"] == "hello"


def test_language_aware_metrics_and_normalization():
    zh = speech_review.review_transcript("第二十五條臺詞", "第25条台词", "ZH", 0.99)
    assert zh["passed"] and zh["metric"] == "cer"
    en = speech_review.review_transcript("one small test", "one test", "EN", 0.5)
    assert en["metric"] == "wer" and en["wer"] == 0.333333


def test_asr_cache_status_and_clear_release_all_entries(monkeypatch):
    speech_review._CACHE.clear()
    speech_review._CACHE[("openai_whisper", "base", "cpu", "")] = object()
    speech_review._CACHE[("faster_whisper", "tiny", "cuda", "models")] = object()
    monkeypatch.setattr(speech_review.torch.cuda, "is_available", lambda: False)

    status = speech_review.asr_cache_status()
    assert status["cached_models"] == 2
    assert speech_review.clear_asr_cache() == 2
    assert speech_review.asr_cache_status() == {"cached_models": 0, "entries": []}


def test_clear_asr_cache_waits_for_active_transcription(monkeypatch):
    inference_started = threading.Event()
    allow_finish = threading.Event()
    clear_finished = threading.Event()
    errors = []

    class FakeWhisper:
        def transcribe(self, _audio, **_kwargs):
            inference_started.set()
            if not allow_finish.wait(5):
                raise TimeoutError("test transcription was not released")
            return {"text": "ok", "language": "en", "segments": []}

    monkeypatch.setattr(
        speech_review, "load_asr_model", lambda *args, **kwargs: (FakeWhisper(), "cpu")
    )
    monkeypatch.setattr(
        speech_review, "resolve_asr_backend", lambda _backend: "openai_whisper"
    )
    speech_review._CACHE.clear()
    speech_review._CACHE[("openai_whisper", "base", "cpu", "")] = object()

    def transcribe():
        try:
            speech_review.transcribe_waveform(
                torch.zeros(1, 16000), 16000, language="EN"
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    worker = threading.Thread(target=transcribe)
    worker.start()
    assert inference_started.wait(5)

    clearer = threading.Thread(
        target=lambda: (speech_review.clear_asr_cache(), clear_finished.set())
    )
    clearer.start()
    assert not clear_finished.wait(0.1)

    allow_finish.set()
    worker.join(5)
    clearer.join(5)
    assert not worker.is_alive()
    assert not clearer.is_alive()
    assert errors == []
    assert clear_finished.is_set()


def test_faster_whisper_checks_interruption_while_consuming_segments(monkeypatch):
    consumed = []

    class FakeFasterWhisper:
        def transcribe(self, _audio, **_kwargs):
            def segments():
                for index in range(3):
                    consumed.append(index)
                    yield SimpleNamespace(text=str(index), words=[])

            return segments(), SimpleNamespace(language="en")

    monkeypatch.setattr(
        speech_review,
        "load_asr_model",
        lambda *args, **kwargs: (FakeFasterWhisper(), "cpu"),
    )
    monkeypatch.setattr(
        speech_review, "resolve_asr_backend", lambda _backend: "faster_whisper"
    )
    checks = 0

    def interrupt():
        nonlocal checks
        checks += 1
        if checks == 5:
            raise RuntimeError("processing interrupted")

    with pytest.raises(RuntimeError, match="processing interrupted"):
        speech_review.transcribe_waveform(
            torch.zeros(1, 16000),
            16000,
            language="EN",
            interrupt_callback=interrupt,
        )

    assert consumed == [0, 1]
