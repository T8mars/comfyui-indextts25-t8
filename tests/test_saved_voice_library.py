from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

from runtime import voice_library


def _bundle(path: Path, *, unsafe: bool = False) -> Path:
    manifest = {
        "schemaVersion": 1,
        "profiles": [
            {
                "profile_id": "hero",
                "name": "主角",
                "audio_path": "../voice.wav" if unsafe else "audio/voice.wav",
                "language": "ZH",
                "emotion_mode": "vector",
                "emotion_vector": [0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                "emotion_strength": 0.75,
                "tags": ["男声", "主角"],
                "favorite": True,
                "notes": "稳定测试音色",
                "quality": {"score": 92},
            }
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
        archive.writestr("../voice.wav" if unsafe else "audio/voice.wav", b"fake-wave")
    return path


def test_scan_and_load_desktop_voice_bundle(tmp_path: Path, monkeypatch):
    _bundle(tmp_path / "demo.t8voice.zip")
    entries = voice_library.scan_saved_voices(tmp_path)
    assert len(entries) == 1
    assert entries[0].label.startswith("主角 · demo.t8voice")
    assert "男声" in entries[0].label

    monkeypatch.setattr(
        voice_library.torchaudio,
        "load",
        lambda _path: (torch.zeros((1, 320)), 24000),
    )
    profile, report = voice_library.load_saved_voice(
        entries[0].label,
        root=tmp_path,
        role_name_override="英雄",
        language_override="EN",
    )
    assert profile.name == "英雄"
    assert profile.language == "EN"
    assert profile.speaker_audio["waveform"].shape == (1, 1, 320)
    assert profile.emotion is not None
    assert profile.emotion.mode == "vector"
    assert profile.emotion.vector[4] == pytest.approx(0.2)
    assert report["favorite"] is True
    assert report["quality"]["score"] == 92


def test_saved_voice_fingerprint_changes_with_bundle(tmp_path: Path):
    bundle = _bundle(tmp_path / "demo.t8voice.zip")
    before = voice_library.saved_voice_fingerprint(tmp_path)
    with bundle.open("ab") as output:
        output.write(b"changed")
    after = voice_library.saved_voice_fingerprint(tmp_path)
    assert before != after


def test_unsafe_voice_bundle_is_not_listed_and_cannot_load(tmp_path: Path):
    bundle = _bundle(tmp_path / "unsafe.t8voice.zip", unsafe=True)
    assert voice_library.scan_saved_voices(tmp_path) == []
    with pytest.raises(ValueError, match="不安全路径"):
        voice_library._read_manifest(bundle)
