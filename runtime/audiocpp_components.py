"""Verified, opt-in audio.cpp runtime and GGUF installer for ComfyUI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Callable


AUDIOCPP_REPOSITORY = "0xShug0/audio.cpp"
AUDIOCPP_RELEASE_API = (
    f"https://api.github.com/repos/{AUDIOCPP_REPOSITORY}/releases/latest"
)
AUDIOCPP_MODEL_REPOSITORY = "audio-cpp/audio.cpp-gguf"
AUDIOCPP_MODEL_API = f"https://huggingface.co/api/models/{AUDIOCPP_MODEL_REPOSITORY}"
AUDIOCPP_MODEL_FOLDER = "IndexTTS2.5-GGUF"
COMPONENT_SCHEMA_VERSION = 2
DOWNLOAD_CHUNK = 8 * 1024 * 1024
DISK_RESERVE_BYTES = 1024**3
ProgressCallback = Callable[[dict[str, Any]], None]
_COMPONENT_LOCK = threading.RLock()
_HASH_CACHE: dict[tuple[str, int, int, int, str], str] = {}

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


def _synchronized(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        with _COMPONENT_LOCK:
            return method(*args, **kwargs)

    return wrapped


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _contained_file(root: Path, path: Path) -> Path:
    root = root.resolve()
    candidate = path.expanduser().resolve()
    if root not in candidate.parents or not candidate.is_file():
        return Path()
    return candidate


def _resolved_manifest_path(
    root: Path,
    manifest: dict[str, Any],
    relative_key: str,
    absolute_key: str,
    fallback: Path | None = None,
) -> Path:
    relative = str(manifest.get(relative_key) or "").strip()
    if relative:
        candidate = _contained_file(root, root / PurePosixPath(relative))
        if candidate.is_file():
            return candidate
    absolute_value = str(manifest.get(absolute_key) or "").strip()
    if absolute_value:
        absolute = _contained_file(root, Path(absolute_value))
        if absolute.is_file():
            return absolute
    if fallback is not None:
        fallback = _contained_file(root, fallback)
        if fallback.is_file():
            return fallback
    return Path()


def _file_change_token(path: Path, stat_result: os.stat_result) -> int:
    """Return a content-change token that cannot be reset with ordinary utime calls."""

    if os.name != "nt":
        return int(getattr(stat_result, "st_ctime_ns", 0))
    try:
        import ctypes
        from ctypes import wintypes

        class FileBasicInfo(ctypes.Structure):
            _fields_ = [
                ("creation_time", ctypes.c_longlong),
                ("last_access_time", ctypes.c_longlong),
                ("last_write_time", ctypes.c_longlong),
                ("change_time", ctypes.c_longlong),
                ("file_attributes", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        get_info = kernel32.GetFileInformationByHandleEx
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        handle = create_file(
            str(path),
            0x0080,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x02000000,
            None,
        )
        if handle == wintypes.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), "CreateFileW failed")
        try:
            info = FileBasicInfo()
            if not get_info(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
                raise OSError(
                    ctypes.get_last_error(), "GetFileInformationByHandleEx failed"
                )
            return int(info.change_time)
        finally:
            close_handle(handle)
    except (AttributeError, OSError, ValueError):
        return int(getattr(stat_result, "st_ctime_ns", 0))


def _verified_sha256(path: Path, expected: str) -> bool:
    stat = path.stat()
    key = (
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        _file_change_token(path, stat),
        expected,
    )
    cached = _HASH_CACHE.get(key)
    if cached is None:
        cached = _sha256(path)
        _HASH_CACHE[key] = cached
        while len(_HASH_CACHE) > 32:
            _HASH_CACHE.pop(next(iter(_HASH_CACHE)))
    return cached == expected


def _file_integrity(
    path: Path,
    *,
    expected_size: int = 0,
    expected_sha256: str = "",
    verify_hash: bool,
    manifest_valid: bool,
) -> tuple[bool, str]:
    if not manifest_valid:
        return False, "manifest-invalid"
    if not path.is_file():
        return False, "missing"
    if expected_size <= 0:
        return False, "metadata-invalid"
    if path.stat().st_size != expected_size:
        return False, "size-mismatch"
    checksum = str(expected_sha256 or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        return False, "metadata-invalid"
    if not verify_hash:
        return True, "size-verified"
    return (
        (True, "verified")
        if _verified_sha256(path, checksum)
        else (False, "sha256-mismatch")
    )


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


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _checksum(value: str) -> str:
    normalized = str(value or "").lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.split(":", 1)[1]
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError("下载源没有提供有效 SHA-256，已拒绝安装。")
    return normalized


def _resumable_download_bytes(
    target: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> int:
    """Return usable staged bytes after removing misleading/corrupt files."""

    part = target.with_suffix(target.suffix + ".part")
    if target.is_file():
        if (
            target.stat().st_size == expected_size
            and _sha256(target) == expected_sha256
        ):
            return expected_size
        target.unlink(missing_ok=True)
    if not part.is_file():
        return 0
    size = part.stat().st_size
    if size > expected_size:
        part.unlink(missing_ok=True)
        return 0
    if size == expected_size:
        if _sha256(part) == expected_sha256:
            return expected_size
        part.unlink(missing_ok=True)
        return 0
    return size


def _expanded_archive_bytes(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            total += sum(max(0, int(item.file_size)) for item in archive.infolist())
    return total


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
        if (
            target.stat().st_size == expected_size
            and _sha256(target) == expected_sha256
        ):
            _emit(
                callback,
                phase="cached",
                label=label,
                received=expected_size,
                total=expected_size,
            )
            return target
        target.unlink()
    existing = part.stat().st_size if part.is_file() else 0
    if existing > expected_size:
        part.write_bytes(b"")
        existing = 0
    elif existing == expected_size:
        if _sha256(part) == expected_sha256:
            part.replace(target)
            _emit(
                callback,
                phase="cached",
                label=label,
                received=expected_size,
                total=expected_size,
            )
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
            if candidate.is_absolute() or any(
                part in {"", ".", ".."} for part in candidate.parts
            ):
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
            (
                item
                for item in assets
                if re.search(pattern, str(item.get("name") or ""), re.I)
            ),
            None,
        )
        if match is None:
            raise RuntimeError(f"audio.cpp 最新 Release 缺少安装资产：{pattern}")
        asset_name = str(match.get("name") or "")
        if (
            Path(asset_name).name != asset_name
            or "/" in asset_name
            or "\\" in asset_name
        ):
            raise RuntimeError("audio.cpp Release 返回了不安全的资产名称。")
        _checksum(match.get("digest"))
        selected.append(match)
    return release, selected


def _is_windows_platform() -> bool:
    """Keep platform probing patchable without mutating the process-wide os module."""
    return os.name == "nt"


@_synchronized
def install_runtime(
    backend: str,
    *,
    data_root: str | Path | None = None,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if not _is_windows_platform():
        raise RuntimeError(
            "当前一键安装器面向 Windows；其他系统请使用 audio.cpp 官方包。"
        )
    backend = str(backend).lower()
    release, assets = _release_assets(backend)
    tag = str(release.get("tag_name") or "").strip()
    if not tag:
        raise RuntimeError("audio.cpp Release 缺少版本号。")
    root = _root(data_root)
    downloads = root / "downloads" / tag
    install_dir = root / "runtime" / tag / backend
    root.mkdir(parents=True, exist_ok=True)
    remaining_download_bytes = sum(
        max(
            0,
            int(item.get("size") or 0)
            - _resumable_download_bytes(
                downloads / str(item["name"]),
                expected_size=int(item.get("size") or 0),
                expected_sha256=_checksum(item.get("digest")),
            ),
        )
        for item in assets
    )
    available = shutil.disk_usage(root).free
    if available < remaining_download_bytes + DISK_RESERVE_BYTES:
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
    expanded_bytes = _expanded_archive_bytes(downloaded)
    available = shutil.disk_usage(root).free
    if available < expanded_bytes * 2 + DISK_RESERVE_BYTES:
        raise RuntimeError("audio.cpp 解压空间不足：需要同时保留解压暂存和候选运行时。")
    with tempfile.TemporaryDirectory(
        prefix="t8_audiocpp_install_", dir=root
    ) as temporary:
        staging = Path(temporary) / "payload"
        for archive in downloaded:
            _safe_extract(archive, staging)
        if not list(staging.rglob("audiocpp_cli.exe")):
            raise RuntimeError("官方压缩包内没有找到 audiocpp_cli.exe。")
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        candidate_dir = install_dir.with_name(
            f".{install_dir.name}.new-{os.getpid()}-{time.time_ns()}"
        )
        backup_dir = install_dir.with_name(
            f".{install_dir.name}.backup-{os.getpid()}-{time.time_ns()}"
        )
        try:
            shutil.copytree(staging, candidate_dir)
        except Exception:
            shutil.rmtree(candidate_dir, ignore_errors=True)
            raise
        candidate_executable = next(iter(candidate_dir.rglob("audiocpp_cli.exe")), None)
        if candidate_executable is None:
            shutil.rmtree(candidate_dir, ignore_errors=True)
            raise RuntimeError("audio.cpp 候选运行时缺少可执行文件。")
        executable_relative_to_install = candidate_executable.relative_to(candidate_dir)
        executable = (install_dir / executable_relative_to_install).resolve()
        manifest = {
            "schemaVersion": COMPONENT_SCHEMA_VERSION,
            "release": tag,
            "releaseUrl": release.get("html_url"),
            "backend": backend,
            "executable": str(executable),
            "executableRelative": _relative_path(root, executable),
            "executableSize": candidate_executable.stat().st_size,
            "executableSha256": _sha256(candidate_executable),
            "installedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            _write_json_atomic(candidate_dir / "t8-component.json", manifest)
        except Exception:
            shutil.rmtree(candidate_dir, ignore_errors=True)
            raise
        had_original = install_dir.exists()
        try:
            if had_original:
                install_dir.replace(backup_dir)
        except Exception:
            shutil.rmtree(candidate_dir, ignore_errors=True)
            raise
        try:
            candidate_dir.replace(install_dir)
            _write_json_atomic(root / "current-runtime.json", manifest)
        except Exception:
            shutil.rmtree(install_dir, ignore_errors=True)
            if had_original and backup_dir.exists():
                backup_dir.replace(install_dir)
            raise
        else:
            shutil.rmtree(backup_dir, ignore_errors=True)
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
    tree = _request_json(
        f"{AUDIOCPP_MODEL_API}/tree/{revision}/{folder}?recursive=true&expand=true"
    )
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
            f"https://huggingface.co/{AUDIOCPP_MODEL_REPOSITORY}/resolve/{revision}/{urllib.parse.quote(wanted, safe='/')}?download=true"
        ),
    }


@_synchronized
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
    download_target = model_path.with_suffix(model_path.suffix + ".download")
    root.mkdir(parents=True, exist_ok=True)
    installed_valid = (
        model_path.is_file()
        and model_path.stat().st_size == int(metadata["size"])
        and _verified_sha256(model_path, metadata["sha256"])
    )
    existing = _resumable_download_bytes(
        download_target,
        expected_size=int(metadata["size"]),
        expected_sha256=metadata["sha256"],
    )
    remaining = max(0, int(metadata["size"]) - existing)
    if (
        not installed_valid
        and shutil.disk_usage(root).free < remaining + DISK_RESERVE_BYTES
    ):
        raise RuntimeError("GGUF 模型安装空间不足。")
    manifest = {
        "schemaVersion": COMPONENT_SCHEMA_VERSION,
        "repository": AUDIOCPP_MODEL_REPOSITORY,
        "revision": metadata["revision"],
        "quantization": str(quantization).lower(),
        "modelPath": str(model_path.resolve()),
        "modelRelative": _relative_path(root, model_path),
        "size": int(metadata["size"]),
        "sha256": metadata["sha256"],
        "installedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if installed_valid:
        _write_json_atomic(model_dir / "t8-model.json", manifest)
    else:
        downloaded_model = _download(
            metadata["url"],
            download_target,
            expected_size=int(metadata["size"]),
            expected_sha256=metadata["sha256"],
            label=metadata["filename"],
            callback=callback,
        )
        candidate_model = model_path.with_name(
            f".{model_path.name}.new-{os.getpid()}-{time.time_ns()}"
        )
        backup_model = model_path.with_name(
            f".{model_path.name}.backup-{os.getpid()}-{time.time_ns()}"
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        downloaded_model.replace(candidate_model)
        had_original = model_path.exists()
        try:
            if had_original:
                model_path.replace(backup_model)
            candidate_model.replace(model_path)
            _write_json_atomic(model_dir / "t8-model.json", manifest)
        except Exception:
            candidate_model.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)
            if had_original and backup_model.exists():
                backup_model.replace(model_path)
            raise
        else:
            backup_model.unlink(missing_ok=True)
    _emit(callback, phase="complete", label=metadata["filename"], percent=1.0)
    return manifest


@_synchronized
def component_status(
    data_root: str | Path | None = None,
    *,
    verify_hash: bool = False,
) -> dict[str, Any]:
    root = _root(data_root)
    runtime: dict[str, Any] = {}
    model: dict[str, Any] = {}
    try:
        runtime = json.loads(
            (root / "current-runtime.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    if not isinstance(runtime, dict):
        runtime = {}
    try:
        model = json.loads(
            (root / "models" / AUDIOCPP_MODEL_FOLDER / "t8-model.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    if not isinstance(model, dict):
        model = {}
    release = str(runtime.get("release") or "").strip()
    backend = str(runtime.get("backend") or "").strip()
    runtime_fallback = None
    if release and Path(release).name == release and backend in _BACKEND_ASSETS:
        runtime_dir = root / "runtime" / release / backend
        runtime_fallback = next(iter(runtime_dir.rglob("audiocpp_cli.exe")), None)
    executable = _resolved_manifest_path(
        root,
        runtime,
        "executableRelative",
        "executable",
        runtime_fallback,
    )
    model_filename = _MODEL_FILES.get(str(model.get("quantization") or "").lower())
    model_fallback = (
        root / "models" / AUDIOCPP_MODEL_FOLDER / model_filename
        if model_filename
        else None
    )
    model_path = _resolved_manifest_path(
        root,
        model,
        "modelRelative",
        "modelPath",
        model_fallback,
    )
    runtime_ready, runtime_integrity = _file_integrity(
        executable,
        expected_size=_nonnegative_int(runtime.get("executableSize")),
        expected_sha256=str(runtime.get("executableSha256") or ""),
        verify_hash=True,
        manifest_valid=_nonnegative_int(runtime.get("schemaVersion"))
        == COMPONENT_SCHEMA_VERSION,
    )
    model_ready, model_integrity = _file_integrity(
        model_path,
        expected_size=_nonnegative_int(model.get("size")),
        expected_sha256=str(model.get("sha256") or ""),
        verify_hash=verify_hash,
        manifest_valid=_nonnegative_int(model.get("schemaVersion"))
        == COMPONENT_SCHEMA_VERSION,
    )
    return {
        "root": str(root),
        "runtime": runtime,
        "model": model,
        "runtimeReady": runtime_ready,
        "modelReady": model_ready,
        "runtimeIntegrity": runtime_integrity,
        "modelIntegrity": model_integrity,
        "installedBackend": backend,
        "executable": str(executable) if runtime_ready else "",
        "modelPath": str(model_path) if model_ready else "",
    }


__all__ = [
    "AUDIOCPP_MODEL_REPOSITORY",
    "AUDIOCPP_REPOSITORY",
    "component_status",
    "default_component_data_root",
    "install_model",
    "install_runtime",
]
