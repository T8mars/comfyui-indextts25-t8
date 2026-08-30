from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys

import pytest
import torch

from indextts.gpt.transformers_generation_utils import GenerationMixin
from indextts.gpt.transformers_gpt2 import GPT2Model
from indextts.utils import audio_io
from transformers import DynamicCache, GPT2Config


@pytest.mark.parametrize(
    "module_name",
    ("indextts.gpt.model", "indextts.gpt.model_v2", "indextts.gpt.model_v2_5"),
)
def test_gpt_inference_models_explicitly_inherit_generation_mixin(module_name):
    module = __import__(module_name, fromlist=["GPT2InferenceModel"])
    assert GenerationMixin in module.GPT2InferenceModel.__bases__
    assert module.GPT2InferenceModel._supports_cache_class is True


def test_gpt2_model_returns_and_reuses_dynamic_cache():
    config = GPT2Config(vocab_size=32, n_positions=16, n_embd=8, n_layer=1, n_head=2)
    model = GPT2Model(config).eval()
    first = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
    assert isinstance(first.past_key_values, DynamicCache)
    assert first.past_key_values.get_seq_length() == 3
    second = model(input_ids=torch.tensor([[4]]), past_key_values=first.past_key_values, use_cache=True)
    assert isinstance(second.past_key_values, DynamicCache)
    assert second.past_key_values.get_seq_length() == 4


def test_torchaudio_28_uses_legacy_backend(monkeypatch, tmp_path):
    expected = torch.zeros(1, 16)
    calls = []
    monkeypatch.setattr(audio_io.torchaudio, "__version__", "2.8.0+cu128")
    monkeypatch.setattr(audio_io.torchaudio, "load", lambda _path: (expected, 22050))
    monkeypatch.setattr(
        audio_io.torchaudio,
        "save",
        lambda path, waveform, sample_rate, **kwargs: calls.append(
            (path, waveform, sample_rate, kwargs)
        ),
    )
    waveform, sample_rate = audio_io.load_audio_file(tmp_path / "input.wav")
    audio_io.save_audio_file(tmp_path / "output.wav", waveform, sample_rate, pcm16=True)
    assert waveform is expected
    assert sample_rate == 22050
    assert calls[0][3] == {"encoding": "PCM_S", "bits_per_sample": 16}


def test_torchaudio_29_uses_native_torchcodec(monkeypatch, tmp_path):
    expected = torch.zeros(1, 32)
    saved = []
    monkeypatch.setattr(audio_io.torchaudio, "__version__", "2.9.1+cu128")
    monkeypatch.setattr(audio_io, "_load_with_torchcodec", lambda _path: (expected, 24000))
    monkeypatch.setattr(
        audio_io,
        "_save_with_torchcodec",
        lambda path, waveform, sample_rate: saved.append((Path(path), waveform, sample_rate)),
    )
    waveform, sample_rate = audio_io.load_audio_file(tmp_path / "input.wav")
    audio_io.save_audio_file(tmp_path / "output.wav", waveform, sample_rate)
    assert waveform is expected
    assert sample_rate == 24000
    assert saved[0][0].name == "output.wav"
    assert saved[0][1].dtype == torch.float32
    assert saved[0][2] == 24000


def test_torchcodec_failure_has_actionable_abi_message(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_io.torchaudio, "__version__", "2.9.0")

    def fail(_path):
        raise ImportError("missing libtorchcodec")

    monkeypatch.setattr(audio_io, "_load_with_torchcodec", fail)
    with pytest.raises(RuntimeError, match="0.8/0.9 with torch 2.9"):
        audio_io.load_audio_file(tmp_path / "input.wav")


def test_torchcodec_preflight_reports_legacy_and_missing_runtime(monkeypatch):
    monkeypatch.setattr(audio_io.torchaudio, "__version__", "2.8.0")
    assert audio_io.probe_torchcodec_runtime()["ready"] is True
    monkeypatch.setattr(audio_io.torchaudio, "__version__", "2.9.0")
    monkeypatch.setattr(audio_io.metadata, "version", lambda _name: (_ for _ in ()).throw(audio_io.metadata.PackageNotFoundError()))
    report = audio_io.probe_torchcodec_runtime()
    assert report["required"] is True
    assert report["ready"] is False
    assert "requires a compatible TorchCodec" in report["reason"]


def test_torchcodec_preflight_accepts_loadable_native_modules(monkeypatch):
    monkeypatch.setattr(audio_io.torchaudio, "__version__", "2.9.0")
    monkeypatch.setattr(audio_io.metadata, "version", lambda _name: "0.9.0")
    package = ModuleType("torchcodec")
    package.__path__ = []
    decoders, encoders = ModuleType("torchcodec.decoders"), ModuleType("torchcodec.encoders")
    decoders.AudioDecoder, encoders.AudioEncoder = object, object
    monkeypatch.setitem(sys.modules, "torchcodec", package)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    monkeypatch.setitem(sys.modules, "torchcodec.encoders", encoders)
    report = audio_io.probe_torchcodec_runtime()
    assert report["ready"] is True
    assert report["ffmpeg_shared_libraries"] == "loaded"


@pytest.mark.skipif(
    not audio_io.uses_torchcodec_io(),
    reason="native TorchCodec roundtrip runs in the Torchaudio 2.9 CI matrix",
)
def test_real_torchcodec_roundtrip(tmp_path):
    sample_rate = 24000
    timeline = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    original = (0.2 * torch.sin(2 * torch.pi * 440 * timeline)).unsqueeze(0)
    output = tmp_path / "torchcodec-roundtrip.wav"
    audio_io.save_audio_file(output, original, sample_rate, pcm16=True)
    decoded, decoded_rate = audio_io.load_audio_file(output)
    assert decoded_rate == sample_rate
    assert decoded.shape == original.shape
    assert torch.isfinite(decoded).all()
    assert float(decoded.abs().max()) == pytest.approx(0.2, abs=0.01)
