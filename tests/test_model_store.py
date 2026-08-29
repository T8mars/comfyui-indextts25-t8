from __future__ import annotations

import hashlib
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from services import downloader, model_store


def _metadata(data: bytes) -> dict:
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_discover_and_validate_model_directory(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "IndexTTS-2.5"
    model_dir.mkdir()
    files = {
        "config.yaml": b"version: 2.5",
        "gpt.pth": b"gpt",
        "nested/model.bin": b"model",
    }
    for relative, data in files.items():
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    manifest = {
        "modelRevision": "formal-2.5",
        "files": {relative: _metadata(data) for relative, data in files.items()},
    }
    monkeypatch.setattr(model_store, "load_manifest", lambda: manifest)
    discovered = model_store.discover_models([tmp_path])
    assert discovered == {"IndexTTS-2.5": model_dir.resolve()}
    assert model_store.validate_model_dir(model_dir).valid
    assert model_store.validate_model_dir(model_dir, verify_hashes=True).valid

    (model_dir / "gpt.pth").write_bytes(b"broken")
    report = model_store.validate_model_dir(model_dir)
    assert not report.valid
    assert report.mismatched == ("gpt.pth",)


def test_model_hash_cache_detects_same_stat_replacement(tmp_path: Path, monkeypatch):
    original = b"good-model"
    replacement = b"evil-model"
    model = tmp_path / "gpt.pth"
    model.write_bytes(original)
    manifest = {
        "modelRevision": "formal-2.5",
        "files": {"gpt.pth": _metadata(original)},
    }
    monkeypatch.setattr(model_store, "load_manifest", lambda: manifest)
    model_store._HASH_CACHE.clear()

    assert model_store.validate_model_dir(tmp_path, verify_hashes=True).valid
    before = model.stat()
    model.write_bytes(replacement)
    os.utime(model, ns=(before.st_atime_ns, before.st_mtime_ns))

    report = model_store.validate_model_dir(tmp_path, verify_hashes=True)
    assert report.mismatched == ("gpt.pth",)


def test_model_hash_cache_is_thread_safe_and_bounded(tmp_path: Path):
    paths = []
    expected = {}
    for index in range(192):
        data = bytes([index % 251]) * 1024
        path = tmp_path / f"model-{index}.bin"
        path.write_bytes(data)
        paths.append(path)
        expected[path] = hashlib.sha256(data).hexdigest()

    with model_store._HASH_CACHE_LOCK:
        model_store._HASH_CACHE.clear()
    previous_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(1e-6)
        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(model_store._sha256, paths))
    finally:
        sys.setswitchinterval(previous_interval)

    assert results == [expected[path] for path in paths]
    with model_store._HASH_CACHE_LOCK:
        assert len(model_store._HASH_CACHE) <= model_store._HASH_CACHE_MAX_ENTRIES


def test_manifest_is_pinned_to_formal_index_tts_25():
    manifest = model_store.load_manifest()
    assert manifest["codeRevision"] == "ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c"
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["modelRevision"])
    assert (
        manifest["files"]["config.yaml"]["sha256"]
        == "18adf417be3e8f5e2e48e30f7420c719170a6870619436250f360d626877870e"
    )
    assert manifest["modelRepository"] == "t8star/IndexTTS-2.5-Comfy"
    assert manifest["upstreamModelRepository"] == "IndexTeam/IndexTTS-2.5"
    assert (
        manifest["upstreamModelRevision"] == "c39ce5ba981572cb187443877ff559dfb246ce63"
    )
    assert manifest["files"]["bpe.model"] == {
        "size": 475997,
        "sha256": "b2a5ce8090d32da3642cc4f81fdc996376bc6dd3f4cd5e3d165f71120d9f2bc8",
        "sourceRepository": "IndexTeam/IndexTTS-2",
        "sourceRevision": "740dcaff396282ffb241903d150ac011cd4b1ede",
    }
    assert downloader._file_source(manifest, "bpe.model") == (
        "t8star/IndexTTS-2.5-Comfy",
        manifest["modelRevision"],
    )
    auxiliary = {
        relative
        for relative, metadata in manifest["files"].items()
        if metadata.get("group") == "auxiliary"
    }
    assert auxiliary == {
        "hf_cache/w2v-bert-2.0/config.json",
        "hf_cache/w2v-bert-2.0/model.safetensors",
        "hf_cache/w2v-bert-2.0/preprocessor_config.json",
        "hf_cache/campplus_cn_common.bin",
        "hf_cache/bigvgan/config.json",
        "hf_cache/bigvgan/bigvgan_generator.pt",
    }
    assert not any("semantic_codec" in relative for relative in manifest["files"])


def test_validation_can_skip_or_require_auxiliary_files(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "IndexTTS-2.5"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_bytes(b"config")
    manifest = {
        "modelRevision": "formal-2.5",
        "files": {
            "config.yaml": _metadata(b"config"),
            "hf_cache/helper.bin": {
                **_metadata(b"helper"),
                "group": "auxiliary",
            },
        },
    }
    monkeypatch.setattr(model_store, "load_manifest", lambda: manifest)

    assert model_store.validate_model_dir(model_dir, include_auxiliary=False).valid
    report = model_store.validate_model_dir(model_dir, include_auxiliary=True)
    assert report.missing == ("hf_cache/helper.bin",)


def test_huggingface_download_repairs_only_requested_bundle_files(
    tmp_path: Path, monkeypatch
):
    manifest = {
        "modelRepository": "t8star/IndexTTS-2.5-Comfy",
        "modelRevision": "a" * 40,
        "files": {
            "config.yaml": _metadata(b"config"),
            "bpe.model": _metadata(b"bpe"),
            "hf_cache/helper.bin": {
                **_metadata(b"helper"),
                "group": "auxiliary",
            },
        },
    }
    captured = []

    def fake_hf_hub_download(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(downloader, "load_manifest", lambda: manifest)
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        type("Hub", (), {"hf_hub_download": staticmethod(fake_hf_hub_download)}),
    )

    downloader.download_main_model(
        tmp_path,
        "huggingface",
        missing=("bpe.model", "hf_cache/helper.bin"),
    )
    assert [item["filename"] for item in captured] == [
        "bpe.model",
        "hf_cache/helper.bin",
    ]
    assert all(item["repo_id"] == "t8star/IndexTTS-2.5-Comfy" for item in captured)
    assert all(item["revision"] == "a" * 40 for item in captured)
    assert all(item["force_download"] is False for item in captured)


def test_model_download_progress_is_monotonic_and_reports_disk_preflight():
    events = []
    tick = [0.0]

    def clock():
        tick[0] += 1.0
        return tick[0]

    manifest = {
        "files": {
            "a.bin": {"size": 100, "sha256": "0" * 64},
            "b.bin": {"size": 300, "sha256": "1" * 64},
        }
    }
    progress = downloader.ModelDownloadProgress(
        manifest,
        "huggingface",
        events.append,
        include_auxiliary=True,
        clock=clock,
    )
    progress.preflight(["b.bin"], free_bytes=10_000)
    progress.begin_file("b.bin", 1)
    progress.resume_file(100)
    progress.update_file(200)
    progress.complete_file()
    progress.verify("b.bin", 300, 300)
    progress.done()

    assert events[0]["phase"] == "preflight"
    assert events[0]["required_bytes"] == 300
    assert events[0]["available_bytes"] == 10_000
    assert events[-1]["phase"] == "complete"
    assert events[-1]["overall_fraction"] == 1.0
    fractions = [event["overall_fraction"] for event in events]
    assert fractions == sorted(fractions)


def test_model_download_stops_before_network_when_space_is_insufficient(
    tmp_path: Path, monkeypatch
):
    manifest = {
        "files": {
            "large.bin": {"size": 2 * 1024**3, "sha256": "0" * 64},
        }
    }
    report = model_store.ValidationReport(
        model_dir=tmp_path,
        missing=("large.bin",),
    )
    download_started = []
    monkeypatch.setattr(downloader, "load_manifest", lambda: manifest)
    monkeypatch.setattr(
        downloader, "validate_model_dir", lambda *args, **kwargs: report
    )
    monkeypatch.setattr(
        downloader.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=600 * 1024**2),
    )
    monkeypatch.setattr(
        downloader,
        "download_main_model",
        lambda *args, **kwargs: download_started.append(True),
    )

    with pytest.raises(RuntimeError, match="空间不足"):
        downloader.ensure_model_bundle(
            tmp_path,
            accept_license=True,
            skip_auxiliary=True,
        )

    assert download_started == []


def test_huggingface_download_interrupt_terminates_worker_without_waiting(
    monkeypatch,
):
    class UserInterrupt(BaseException):
        pass

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.killed = False
            self.waits = []

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return self.returncode

    process = FakeProcess()
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(downloader.subprocess, "Popen", fake_popen)

    def interrupt():
        raise UserInterrupt()

    with pytest.raises(UserInterrupt):
        downloader._download_hf_file_cancellable(
            {
                "repo_id": "t8star/IndexTTS-2.5-Comfy",
                "revision": "a" * 40,
                "filename": "model.safetensors",
                "local_dir": "models",
                "force_download": False,
            },
            interrupt,
        )

    assert process.terminated is True
    assert process.killed is False
    assert process.waits == [downloader.HF_DOWNLOAD_STOP_SECONDS]
    assert captured["command"][:4] == [
        sys.executable,
        "-m",
        "services.downloader",
        "--internal-hf-download",
    ]
    assert captured["kwargs"]["cwd"] == str(downloader.PLUGIN_ROOT)


def test_comfy_requirements_do_not_replace_torch_or_pull_training_stacks():
    requirements = (
        (model_store.PLUGIN_ROOT / "requirements.txt")
        .read_text(encoding="utf-8")
        .lower()
    )
    active = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(
        line.startswith(("torch", "torchaudio", "torchvision")) for line in active
    )
    assert not any(
        line.startswith(("descript-audiotools", "openai-whisper")) for line in active
    )
    assert len(active) == 8
    assert not any(line.startswith(("matplotlib", "modelscope")) for line in active)
    assert [line for line in active if "<" in line] == ["transformers>=4.52.1,<5"]

    optional = (
        (model_store.PLUGIN_ROOT / "requirements-modelscope.txt")
        .read_text(encoding="utf-8")
        .lower()
    )
    optional_active = [
        line.strip()
        for line in optional.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert optional_active == ["modelscope>=1.27"]
