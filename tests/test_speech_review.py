from __future__ import annotations

import numpy as np
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
