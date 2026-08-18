from __future__ import annotations

import torch

from indextts.infer_v2_5 import QwenEmotion
from indextts.utils import common, model_download


def _emotion_converter():
    emotion = QwenEmotion.__new__(QwenEmotion)
    emotion.cn_key_to_en = {
        "高兴": "happy",
        "愤怒": "angry",
        "悲伤": "sad",
        "恐惧": "afraid",
        "反感": "disgusted",
        "低落": "melancholic",
        "惊讶": "surprised",
        "自然": "calm",
    }
    emotion.desired_vector_order = list(emotion.cn_key_to_en)
    emotion.max_score = 1.2
    emotion.min_score = 0.0
    return emotion


def test_qwen_label_output_normalization_is_synced():
    emotion = _emotion_converter()
    assert emotion.convert({"emotion_label": "calm"})["calm"] == 1.0
    redirected = emotion.convert({"高兴": "自然"})
    assert redirected["happy"] == 0.0
    assert redirected["calm"] == 1.0


def test_pcm_wav_save_is_normalized_for_torchcodec(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        common.torchaudio,
        "save",
        lambda path, wav, rate, **kwargs: captured.update(wav=wav.clone()),
    )
    common.save_pcm_wav(
        tmp_path / "out.wav", torch.tensor([[32767.0, 0.0, -16383.5]]), 22050
    )
    assert captured["wav"].dtype == torch.float32
    assert float(captured["wav"].abs().max()) <= 1.0


def test_config_download_targets_index_tts_25(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        model_download,
        "_download_single_file",
        lambda repo, filename, target: calls.append((repo, filename, target)),
    )
    model_download.ensure_config_available(str(tmp_path), version="2.5")
    assert calls[0][:2] == ("IndexTeam/IndexTTS-2.5", "config.yaml")
