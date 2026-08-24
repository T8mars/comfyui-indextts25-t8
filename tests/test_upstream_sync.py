from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from indextts.gpt import model_v2
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


def test_pcm_tail_fade_is_synced_and_ends_at_zero():
    source = torch.full((1, 100), 32767.0)
    faded = common.fade_out_pcm_tail(source, 1000, duration_ms=20)
    assert torch.equal(source, torch.full((1, 100), 32767.0))
    assert torch.equal(faded[..., :-20], source[..., :-20])
    assert faded[0, -1] == 0
    assert torch.all(faded[..., -20:-1].diff() <= 0)


def test_config_download_targets_index_tts_25(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        model_download,
        "_download_single_file",
        lambda repo, filename, target: calls.append((repo, filename, target)),
    )
    model_download.ensure_config_available(str(tmp_path), version="2.5")
    assert calls[0][:2] == ("IndexTeam/IndexTTS-2.5", "config.yaml")


def test_deepspeed_bfloat16_dtype_fix_is_synced(monkeypatch):
    class FakeInference:
        def eval(self):
            return self

    captured = {}

    def init_inference(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(module=FakeInference())

    monkeypatch.setattr(model_v2, "GPT2Config", lambda **kwargs: object())
    monkeypatch.setattr(model_v2, "GPT2InferenceModel", lambda *args, **kwargs: FakeInference())
    monkeypatch.setattr(model_v2.torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(init_inference=init_inference))

    voice = model_v2.UnifiedVoice.__new__(model_v2.UnifiedVoice)
    voice.max_mel_tokens = 10
    voice.max_text_tokens = 10
    voice.number_mel_codes = 20
    voice.model_dim = 8
    voice.layers = 1
    voice.heads = 1
    voice.use_accel = False
    voice.gpt = SimpleNamespace(wte=None)
    voice.mel_pos_embedding = object()
    voice.mel_embedding = object()
    voice.final_norm = object()
    voice.mel_head = object()
    voice.post_init_gpt2_config(
        use_deepspeed=True,
        kv_cache=True,
        half=True,
        deepspeed_dtype=torch.bfloat16,
    )
    assert captured["dtype"] is torch.bfloat16
