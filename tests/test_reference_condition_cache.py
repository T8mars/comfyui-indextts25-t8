from __future__ import annotations

from pathlib import Path

import torch

from indextts.utils.reference_condition_cache import ReferenceConditionCache


def test_reference_condition_cache_uses_safe_tensor_files(tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"reference-audio")
    cache = ReferenceConditionCache(tmp_path / "cache", "model-fingerprint")
    tensor = torch.arange(12).reshape(1, 3, 4).float()
    saved = cache.save("speaker", audio, {"spk_cond": tensor})
    assert saved is not None and saved.suffix == ".safetensors"
    loaded = cache.load("speaker", audio, "cpu")
    assert loaded is not None
    assert torch.equal(loaded["spk_cond"], tensor)


def test_disabled_cache_does_not_write(tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"reference-audio")
    cache = ReferenceConditionCache(None, "model")
    assert cache.save("speaker", audio, {"spk_cond": torch.ones(1)}) is None
    assert cache.load("speaker", audio, "cpu") is None


def test_cache_management_reports_session_stats_and_preserves_other_files(tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"reference-audio")
    cache = ReferenceConditionCache(tmp_path / "cache", "model")
    assert cache.load("speaker", audio, "cpu") is None
    cache.save("speaker", audio, {"spk_cond": torch.ones(1)})
    assert cache.load("speaker", audio, "cpu") is not None
    unrelated = tmp_path / "cache" / "notes.json"
    unrelated.write_text("{}", encoding="utf-8")

    before = cache.status()
    assert before["entries"] == 1
    assert before["hits"] == 1
    assert before["misses"] == 1
    assert before["hit_rate"] == 0.5
    assert cache.clear() == 1
    assert cache.status()["entries"] == 0
    assert unrelated.exists()
