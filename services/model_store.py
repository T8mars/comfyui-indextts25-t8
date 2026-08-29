from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PLUGIN_ROOT / "manifests" / "model_2_5.json"
MODEL_FOLDER_NAME = "IndexTTS-2.5"
MISSING_MODEL_OPTION = "[未找到] 请将模型放入 models/TTS/IndexTTS-2.5"
MODEL_REPOSITORY_URL = "https://huggingface.co/t8star/IndexTTS-2.5-Comfy"
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True, slots=True)
class ValidationReport:
    model_dir: Path
    missing: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    hashes_verified: bool = False

    @property
    def valid(self) -> bool:
        return not self.missing and not self.mismatched

    def require_valid(self) -> None:
        if self.valid:
            return
        details: list[str] = []
        if self.missing:
            details.append("缺少：" + ", ".join(self.missing))
        if self.mismatched:
            details.append("大小或哈希不匹配：" + ", ".join(self.mismatched))
        raise FileNotFoundError(
            f"IndexTTS 2.5 模型目录不完整：{self.model_dir}；"
            + "；".join(details)
            + "。可在模型加载器中勾选“缺失时自动下载”并接受模型许可证，"
            + f"或从 {MODEL_REPOSITORY_URL} 下载完整目录。"
        )


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def register_model_paths() -> None:
    import folder_paths

    default_root = Path(folder_paths.models_dir) / "TTS"
    default_root.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path("TTS", str(default_root), is_default=True)


def configured_model_roots() -> list[Path]:
    import folder_paths

    register_model_paths()
    return [Path(item).expanduser().resolve() for item in folder_paths.get_folder_paths("TTS")]


def _looks_like_model(path: Path) -> bool:
    return (path / "config.yaml").is_file() and (path / "gpt.pth").is_file()


def discover_models(roots: Iterable[Path] | None = None) -> dict[str, Path]:
    roots = list(roots) if roots is not None else configured_model_roots()
    candidates: list[tuple[str, Path]] = []
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if _looks_like_model(root):
            candidates.append((root.name, root))
        if not root.is_dir():
            continue
        for current, directories, files in os.walk(root):
            model_dir = Path(current).resolve()
            try:
                relative = model_dir.relative_to(root)
            except ValueError:
                continue
            depth = len(relative.parts)
            if depth >= 3:
                directories.clear()
            else:
                directories[:] = [
                    name for name in directories if name not in {".git", "__pycache__", "hf_cache"}
                ]
            if "config.yaml" not in files or not _looks_like_model(model_dir):
                continue
            candidates.append((relative.as_posix(), model_dir))

    result: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for label, path in sorted(candidates, key=lambda item: (item[0].lower(), str(item[1]).lower())):
        if path in seen_paths:
            continue
        seen_paths.add(path)
        option = label or path.name
        suffix = 2
        while option in result:
            option = f"{label} [{suffix}]"
            suffix += 1
        result[option] = path
    return result


def model_options() -> list[str]:
    values = list(discover_models())
    return values or [MISSING_MODEL_OPTION]


def resolve_model(model_name: str, custom_model_path: str = "") -> Path:
    if custom_model_path.strip():
        return Path(custom_model_path.strip().strip('"')).expanduser().resolve()
    models = discover_models()
    if model_name == MISSING_MODEL_OPTION or model_name not in models:
        expected = configured_model_roots()[0] / MODEL_FOLDER_NAME
        raise FileNotFoundError(
            "未找到完整的 IndexTTS 2.5 模型。可在模型加载器中勾选“缺失时自动下载”"
            "并接受模型许可证，或运行节点目录中的 scripts/download_models.py。"
            f"完整模型：{MODEL_REPOSITORY_URL}；标准位置：{expected}"
        )
    return models[model_name]


def _sha256(path: Path, on_chunk: Callable[[int], None] | None = None) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _HASH_CACHE.get(key)
    if cached is not None:
        if on_chunk is not None:
            on_chunk(stat.st_size)
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            if on_chunk is not None:
                on_chunk(len(chunk))
    value = digest.hexdigest()
    _HASH_CACHE[key] = value
    return value


def validate_model_dir(
    model_dir: Path | str,
    *,
    verify_hashes: bool = False,
    include_auxiliary: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> ValidationReport:
    model_dir = Path(model_dir).expanduser().resolve()
    manifest = load_manifest()
    missing: list[str] = []
    mismatched: list[str] = []
    selected = [
        (relative, metadata)
        for relative, metadata in manifest["files"].items()
        if include_auxiliary or metadata.get("group") != "auxiliary"
    ]
    total = sum(int(metadata["size"]) for _, metadata in selected)
    processed = 0
    for relative, metadata in selected:
        path = model_dir.joinpath(*relative.split("/"))
        if not path.is_file():
            missing.append(relative)
            processed += int(metadata["size"])
            if progress is not None:
                progress(relative, processed, total)
            continue
        if path.stat().st_size != int(metadata["size"]):
            mismatched.append(relative)
            processed += int(metadata["size"])
            if progress is not None:
                progress(relative, processed, total)
            continue
        if verify_hashes:
            hashed = 0

            def update_hash(size: int) -> None:
                nonlocal hashed
                hashed += size
                if progress is not None:
                    progress(relative, processed + hashed, total)

            if _sha256(path, update_hash).lower() != str(metadata["sha256"]).lower():
                mismatched.append(relative)
            processed += int(metadata["size"])
        else:
            processed += int(metadata["size"])
        if progress is not None:
            progress(relative, processed, total)
    return ValidationReport(
        model_dir=model_dir,
        missing=tuple(missing),
        mismatched=tuple(mismatched),
        hashes_verified=verify_hashes,
    )


def model_fingerprint(model_dir: Path | str) -> str:
    model_dir = Path(model_dir)
    manifest = load_manifest()
    parts = [manifest.get("modelRevision", "")]
    for relative in manifest["files"]:
        path = model_dir.joinpath(*relative.split("/"))
        try:
            stat = path.stat()
            parts.append(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}")
        except FileNotFoundError:
            parts.append(f"{relative}:missing")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def default_model_target(comfy_root: Path | str) -> Path:
    return Path(comfy_root).expanduser().resolve() / "models" / "TTS" / MODEL_FOLDER_NAME
