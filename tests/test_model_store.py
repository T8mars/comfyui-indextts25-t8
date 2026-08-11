from __future__ import annotations

import hashlib
from pathlib import Path

from services import model_store


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
    assert manifest["codeRevision"] == "56eead7eb0888ecac6abbf9d777c27f798a2c730"
    assert manifest["modelRevision"] == "ba2480d9f7f629eb18f6acaebb357679d9ba88a4"
    assert manifest["modelRepository"] == "IndexTeam/IndexTTS-2.5"


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
