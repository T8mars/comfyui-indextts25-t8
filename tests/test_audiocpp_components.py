from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

from runtime import audiocpp_components as manager


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def getcode(self):
        return self.status


def test_component_download_verifies_hash(tmp_path: Path, monkeypatch):
    payload = b"verified-component"
    monkeypatch.setattr(manager.urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload))
    target = manager._download(
        "https://example.invalid/file",
        tmp_path / "file.zip",
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        label="test",
    )
    assert target.read_bytes() == payload


def test_component_download_recovers_completed_part(tmp_path: Path, monkeypatch):
    payload = b"complete-part"
    target = tmp_path / "file.zip"
    target.with_suffix(".zip.part").write_bytes(payload)
    monkeypatch.setattr(
        manager.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network not expected")),
    )
    assert manager._download(
        "https://example.invalid/file",
        target,
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        label="test",
    ).read_bytes() == payload


def test_component_runtime_install_and_status(tmp_path: Path, monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("bin/audiocpp_cli.exe", b"exe")
    data = payload.getvalue()
    release = {
        "tag_name": "v9.9.9",
        "html_url": "https://example.invalid/release",
        "assets": [
            {
                "name": "audio-v9.9.9-windows-x64-cpu.zip",
                "size": len(data),
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                "browser_download_url": "https://example.invalid/runtime.zip",
            }
        ],
    }
    monkeypatch.setattr(manager, "_request_json", lambda _url: release)
    monkeypatch.setattr(manager, "_is_windows_platform", lambda: True)
    monkeypatch.setattr(manager.urllib.request, "urlopen", lambda *args, **kwargs: _Response(data))
    result = manager.install_runtime("cpu", data_root=tmp_path)
    assert Path(result["executable"]).read_bytes() == b"exe"
    assert manager.component_status(tmp_path)["runtimeReady"] is True


def test_component_model_install_records_revision(tmp_path: Path, monkeypatch):
    payload = b"gguf"
    metadata = {
        "filename": "index-tts2_5-q8_0.gguf",
        "repositoryPath": "IndexTTS2.5-GGUF/index-tts2_5-q8_0.gguf",
        "revision": "a" * 40,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": "https://example.invalid/model",
    }
    monkeypatch.setattr(manager, "_model_metadata", lambda _quantization: metadata)
    monkeypatch.setattr(manager.urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload))
    result = manager.install_model("q8_0", data_root=tmp_path)
    manifest = json.loads(
        (Path(result["modelPath"]).parent / "t8-model.json").read_text(encoding="utf-8")
    )
    assert manifest["revision"] == "a" * 40
