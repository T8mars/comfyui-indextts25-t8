from __future__ import annotations

import hashlib
import re
from pathlib import Path

from services import downloader, model_store


def _metadata(data: bytes) -> dict:
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_discover_and_validate_model_directory(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "IndexTTS-2.5"
    model_dir.mkdir()
    files = {"config.yaml": b"version: 2.5", "gpt.pth": b"gpt", "nested/model.bin": b"model"}
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


def test_manifest_is_pinned_to_formal_index_tts_25():
    manifest = model_store.load_manifest()
    assert manifest["codeRevision"] == "ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c"
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["modelRevision"])
    assert manifest["files"]["config.yaml"]["sha256"] == "18adf417be3e8f5e2e48e30f7420c719170a6870619436250f360d626877870e"
    assert manifest["modelRepository"] == "t8star/IndexTTS-2.5-Comfy"
    assert manifest["upstreamModelRepository"] == "IndexTeam/IndexTTS-2.5"
    assert manifest["upstreamModelRevision"] == "c39ce5ba981572cb187443877ff559dfb246ce63"
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
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(downloader, "load_manifest", lambda: manifest)
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        type("Hub", (), {"snapshot_download": staticmethod(fake_snapshot_download)}),
    )

    downloader.download_main_model(
        tmp_path,
        "huggingface",
        missing=("bpe.model", "hf_cache/helper.bin"),
    )
    assert captured["repo_id"] == "t8star/IndexTTS-2.5-Comfy"
    assert captured["revision"] == "a" * 40
    assert "bpe.model" in captured["allow_patterns"]
    assert "hf_cache/helper.bin" in captured["allow_patterns"]
    assert "config.yaml" not in captured["allow_patterns"]


def test_comfy_requirements_do_not_replace_torch_or_pull_training_stacks():
    requirements = (model_store.PLUGIN_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    active = [line.strip() for line in requirements.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert not any(line.startswith(("torch", "torchaudio", "torchvision")) for line in active)
    assert not any(line.startswith(("descript-audiotools", "openai-whisper")) for line in active)
    assert len(active) == 8
    assert not any(line.startswith(("matplotlib", "modelscope")) for line in active)
    assert [line for line in active if "<" in line] == ["transformers>=4.52.1,<5"]

    optional = (model_store.PLUGIN_ROOT / "requirements-modelscope.txt").read_text(encoding="utf-8").lower()
    optional_active = [
        line.strip() for line in optional.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    assert optional_active == ["modelscope>=1.27"]
