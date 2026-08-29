"""Verified, opt-in audio.cpp runtime and GGUF installer for ComfyUI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable


AUDIOCPP_REPOSITORY = "0xShug0/audio.cpp"
AUDIOCPP_RELEASE_API = f"https://api.github.com/repos/{AUDIOCPP_REPOSITORY}/releases/latest"
AUDIOCPP_MODEL_REPOSITORY = "audio-cpp/audio.cpp-gguf"
AUDIOCPP_MODEL_API = f"https://huggingface.co/api/models/{AUDIOCPP_MODEL_REPOSITORY}"
AUDIOCPP_MODEL_FOLDER = "IndexTTS2.5-GGUF"
DOWNLOAD_CHUNK = 8 * 1024 * 1024
DISK_RESERVE_BYTES = 1024**3
ProgressCallback = Callable[[dict[str, Any]], None]

_BACKEND_ASSETS = {
    "cpu": (r"-windows-x64-cpu\.zip$",),
    "vulkan": (r"-windows-x64-vulkan\.zip$",),
    "cuda": (
        r"-windows-x64-cuda12\.4\.zip$",
        r"-cudart-windows-x64-cuda12\.4\.zip$",
    ),
}
_MODEL_FILES = {
    "q8_0": "index-tts2_5-q8_0.gguf",
    "f16": "index-tts2_5-f16.gguf",
    "original": "index-tts2_5-orig.gguf",
}


def default_component_data_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "TTS" / "IndexTTS-2.5"
    except Exception:
        return Path(tempfile.gettempdir()) / "t8_indextts25_components"


def _root(data_root: str | Path | None = None) -> Path:
    base = Path(data_root or default_component_data_root()).expanduser().resolve()
    return base / "optional_components" / "audio.cpp"


def _emit(callback: ProgressCallback | None, **event: Any) -> None:
    if callback is not None:
        callback(event)


def _request_json(url: str, timeout: float = 30.0) -> dict[str, Any] | list[Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json, application/json",
            "User-Agent": "T8star-Aix-IndexTTS25",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _checksum(value: str) -> str:
    normalized = str(value or "").lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.split(":", 1)[1]
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError("下载源没有提供有效 SHA-256，已拒绝安装。")
    return normalized


def _sha256(path: Path, callback=None) -> str:
    digest = hashlib.sha256()
    processed = 0
    total = path.stat().st_size
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK):
            digest.update(chunk)
            processed += len(chunk)
            if callback:
                callback(processed, total)
    return digest.hexdigest()


def _download(
    url: str,
    target: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
    callback: ProgressCallback | None = None,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    if target.is_file():
        if target.stat().st_size == expected_size and _sha256(target) == expected_sha256:
            _emit(callback, phase="cached", label=label, received=expected_size, total=expected_size)
            return target
        target.unlink()
    existing = part.stat().st_size if part.is_file() else 0
    if existing > expected_size:
        part.write_bytes(b"")
        existing = 0
    elif existing == expected_size:
        if _sha256(part) == expected_sha256:
            part.replace(target)
            _emit(callback, phase="cached", label=label, received=expected_size, total=expected_size)
            return target
        part.write_bytes(b"")
        existing = 0
    headers = {"User-Agent": "T8star-Aix-IndexTTS25"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", response.getcode())
            resumed = bool(existing and status == 206)
            if existing and not resumed:
                existing = 0
            received = existing
            with part.open("ab" if resumed else "wb") as output:
                while chunk := response.read(DOWNLOAD_CHUNK):
                    output.write(chunk)
                    received += len(chunk)
                    elapsed = max(0.001, time.monotonic() - started)
                    _emit(
                        callback,
                        phase="downloading",
                        label=label,
                        received=received,
                        total=expected_size,
                        percent=received / max(1, expected_size),
                        bytesPerSecond=round((received - existing) / elapsed),
                    )
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"下载 {label} 失败，断点文件已保留：{exc}") from exc
    if not part.is_file() or part.stat().st_size != expected_size:
        raise RuntimeError(f"{label} 下载大小不匹配。")

    def verify_progress(received: int, total: int) -> None:
        _emit(
            callback,
            phase="verifying",
            label=label,
            received=received,
            total=total,
            percent=received / max(1, total),
        )

    if _sha256(part, verify_progress) != expected_sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{label} SHA-256 校验失败，已删除损坏断点。")
    part.replace(target)
    return target


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            candidate = PurePosixPath(item.filename.replace("\\", "/"))
            if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
                raise ValueError(f"audio.cpp 压缩包包含不安全路径：{item.filename}")
        archive.extractall(destination)


def _release_assets(backend: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    backend = str(backend).lower()
    if backend not in _BACKEND_ASSETS:
        raise ValueError("audio.cpp 自动安装仅支持 Windows CUDA、Vulkan 和 CPU。")
    release = _request_json(AUDIOCPP_RELEASE_API)
    if not isinstance(release, dict):
        raise RuntimeError("audio.cpp Release 元数据格式无效。")
    assets = release.get("assets") or []
    selected: list[dict[str, Any]] = []
    for pattern in _BACKEND_ASSETS[backend]:
        match = next(
            (item for item in assets if re.search(pattern, str(item.get("name") or ""), re.I)),
            None,
        )
        if match is None:
            raise RuntimeError(f"audio.cpp 最新 Release 缺少安装资产：{pattern}")
        asset_name = str(match.get("name") or "")
        if Path(asset_name).name != asset_name or "/" in asset_name or "\\" in asset_name:
            raise RuntimeError("audio.cpp Release 返回了不安全的资产名称。")
        _checksum(match.get("digest"))
        selected.append(match)
    return release, selected


def _is_windows_platform() -> bool:
    """Keep platform probing patchable without mutating the process-wide os module."""
    return os.name == "nt"


def install_runtime(
    backend: str,
    *,
    data_root: str | Path | None = None,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if not _is_windows_platform():
        raise RuntimeError("当前一键安装器面向 Windows；其他系统请使用 audio.cpp 官方包。")
    backend = str(backend).lower()
    release, assets = _release_assets(backend)
    tag = str(release.get("tag_name") or "").strip()
    if not tag:
        raise RuntimeError("audio.cpp Release 缺少版本号。")
    root = _root(data_root)
    downloads = root / "downloads" / tag
    install_dir = root / "runtime" / tag / backend
    required = sum(int(item.get("size") or 0) for item in assets)
    root.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(root).free
    if available < required + DISK_RESERVE_BYTES:
        raise RuntimeError("audio.cpp 安装空间不足。")
    downloaded = [
        _download(
            str(asset["browser_download_url"]),
            downloads / str(asset["name"]),
            expected_size=int(asset["size"]),
            expected_sha256=_checksum(asset.get("digest")),
            label=str(asset["name"]),
            callback=callback,
        )
        for asset in assets
    ]
    with tempfile.TemporaryDirectory(prefix="t8_audiocpp_install_") as temporary:
        staging = Path(temporary) / "payload"
        for archive in downloaded:
            _safe_extract(archive, staging)
        if not list(staging.rglob("audiocpp_cli.exe")):
            raise RuntimeError("官方压缩包内没有找到 audiocpp_cli.exe。")
        if install_dir.exists():
            shutil.rmtree(install_dir)
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, install_dir)
    executable = next(iter(install_dir.rglob("audiocpp_cli.exe")), None)
    if executable is None:
        raise RuntimeError("audio.cpp 安装完成后可执行文件缺失。")
    manifest = {
        "release": tag,
        "releaseUrl": release.get("html_url"),
        "backend": backend,
        "executable": str(executable.resolve()),
        "installedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (root / "current-runtime.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _emit(callback, phase="complete", label="audio.cpp runtime", percent=1.0)
    return manifest


def _model_metadata(quantization: str) -> dict[str, Any]:
    filename = _MODEL_FILES.get(str(quantization).lower())
    if filename is None:
        raise ValueError("GGUF 精度必须是 q8_0、f16 或 original。")
    repository = _request_json(AUDIOCPP_MODEL_API)
    if not isinstance(repository, dict) or not repository.get("sha"):
        raise RuntimeError("GGUF 仓库元数据格式无效。")
    revision = str(repository["sha"])
    folder = urllib.parse.quote(AUDIOCPP_MODEL_FOLDER, safe="")
    tree = _request_json(f"{AUDIOCPP_MODEL_API}/tree/{revision}/{folder}?recursive=true&expand=true")
    wanted = f"{AUDIOCPP_MODEL_FOLDER}/{filename}"
    if not isinstance(tree, list):
        raise RuntimeError("GGUF 文件清单格式无效。")
    item = next((entry for entry in tree if entry.get("path") == wanted), None)
    if item is None:
        raise RuntimeError(f"GGUF 仓库缺少 {wanted}。")
    return {
        "filename": filename,
        "repositoryPath": wanted,
        "revision": revision,
        "size": int(item.get("size") or 0),
        "sha256": _checksum((item.get("lfs") or {}).get("oid")),
        "url": (
            f"https://huggingface.co/{AUDIOCPP_MODEL_REPOSITORY}/resolve/{revision}/"
            f"{urllib.parse.quote(wanted, safe='/')}?download=true"
        ),
    }


def install_model(
    quantization: str = "q8_0",
    *,
    data_root: str | Path | None = None,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    metadata = _model_metadata(quantization)
    root = _root(data_root)
    model_dir = root / "models" / AUDIOCPP_MODEL_FOLDER
    model_path = model_dir / metadata["filename"]
    root.mkdir(parents=True, exist_ok=True)
    remaining = max(0, int(metadata["size"]) - (model_path.stat().st_size if model_path.is_file() else 0))
    if shutil.disk_usage(root).free < remaining + DISK_RESERVE_BYTES:
        raise RuntimeError("GGUF 模型安装空间不足。")
    _download(
        metadata["url"],
        model_path,
        expected_size=int(metadata["size"]),
        expected_sha256=metadata["sha256"],
        label=metadata["filename"],
        callback=callback,
    )
    manifest = {
        "repository": AUDIOCPP_MODEL_REPOSITORY,
        "revision": metadata["revision"],
        "quantization": str(quantization).lower(),
        "modelPath": str(model_path.resolve()),
        "size": int(metadata["size"]),
        "sha256": metadata["sha256"],
        "installedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "t8-model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _emit(callback, phase="complete", label=metadata["filename"], percent=1.0)
    return manifest


def component_status(data_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(data_root)
    runtime: dict[str, Any] = {}
    model: dict[str, Any] = {}
    try:
        runtime = json.loads((root / "current-runtime.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    try:
        model = json.loads(
            (root / "models" / AUDIOCPP_MODEL_FOLDER / "t8-model.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    executable = Path(str(runtime.get("executable") or ""))
    model_path = Path(str(model.get("modelPath") or ""))
    return {
        "root": str(root),
        "runtime": runtime,
        "model": model,
        "runtimeReady": executable.is_file(),
        "modelReady": model_path.is_file(),
        "executable": str(executable) if executable.is_file() else "",
        "modelPath": str(model_path) if model_path.is_file() else "",
    }


__all__ = [
    "AUDIOCPP_MODEL_REPOSITORY",
    "AUDIOCPP_REPOSITORY",
    "component_status",
    "default_component_data_root",
    "install_model",
    "install_runtime",
]
