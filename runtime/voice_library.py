"""Portable saved-voice bundles shared by the Desktop and ComfyUI editions."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import torchaudio

from .types import EmotionConfig, VoiceProfile


VOICE_BUNDLE_SUFFIX = ".t8voice.zip"
VOICE_BUNDLE_SCHEMA_VERSION = 1
SUPPORTED_LANGUAGES = {"ZH", "EN", "JA", "ES", "AR"}
MAX_VOICE_BUNDLE_MEMBER_BYTES = 2 * 1024**3
MAX_VOICE_BUNDLE_TOTAL_BYTES = 4 * 1024**3
MAX_VOICE_BUNDLE_MEMBERS = 1024
MAX_VOICE_PROFILES = 256


@dataclass(frozen=True, slots=True)
class SavedVoiceEntry:
    key: str
    label: str
    bundle_path: Path
    profile: dict[str, Any]


def default_voice_library_root() -> Path:
    """Return the user-writable folder scanned by the saved-voice node."""

    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "TTS" / "IndexTTS-2.5" / "voices"
    except Exception:
        return Path(tempfile.gettempdir()) / "t8_indextts25_voices"


def _safe_member(name: str) -> PurePosixPath:
    value = PurePosixPath(str(name).replace("\\", "/"))
    if value.is_absolute() or not value.parts or any(
        part in {"", ".", ".."} for part in value.parts
    ):
        raise ValueError(f"音色包包含不安全路径：{name}")
    return value


def _read_manifest(bundle_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            members = archive.infolist()
            if len(members) > MAX_VOICE_BUNDLE_MEMBERS:
                raise ValueError(f"音色包文件数量超过安全上限：{bundle_path.name}")
            if any(member.file_size > MAX_VOICE_BUNDLE_MEMBER_BYTES for member in members):
                raise ValueError(f"音色包包含超过 2 GiB 的单个文件：{bundle_path.name}")
            if sum(member.file_size for member in members) > MAX_VOICE_BUNDLE_TOTAL_BYTES:
                raise ValueError(f"音色包解压后总大小超过 4 GiB：{bundle_path.name}")
            for member in members:
                _safe_member(member.filename)
            manifest_info = archive.getinfo("manifest.json")
            if manifest_info.file_size > 4 * 1024 * 1024:
                raise ValueError(f"音色包 manifest.json 超过 4 MiB：{bundle_path.name}")
            payload = json.loads(archive.read("manifest.json").decode("utf-8"))
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"音色包 manifest.json 缺失或损坏：{bundle_path.name}") from exc
    if not isinstance(payload, dict) or payload.get("schemaVersion") != VOICE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"不支持的音色包版本：{bundle_path.name}")
    if not isinstance(payload.get("profiles"), list):
        raise ValueError(f"音色包角色列表无效：{bundle_path.name}")
    if len(payload["profiles"]) > MAX_VOICE_PROFILES:
        raise ValueError(f"单个音色包最多包含 256 个角色：{bundle_path.name}")
    return payload


def scan_saved_voices(root: str | Path | None = None) -> list[SavedVoiceEntry]:
    library_root = Path(root or default_voice_library_root()).expanduser().resolve()
    result: list[SavedVoiceEntry] = []
    if not library_root.is_dir():
        return result
    for bundle_path in sorted(library_root.glob(f"*{VOICE_BUNDLE_SUFFIX}")):
        try:
            manifest = _read_manifest(bundle_path)
        except ValueError:
            continue
        for index, profile in enumerate(manifest["profiles"]):
            if not isinstance(profile, dict):
                continue
            name = str(profile.get("name") or "").strip()
            if not name or not profile.get("audio_path"):
                continue
            profile_id = str(profile.get("profile_id") or index)
            key = f"{bundle_path.name}::{profile_id}"
            tags = " / ".join(str(item) for item in profile.get("tags") or [] if str(item).strip())
            label = f"{name} · {bundle_path.stem}"
            if tags:
                label += f" · {tags}"
            result.append(SavedVoiceEntry(key, label, bundle_path, dict(profile)))
    return sorted(result, key=lambda item: item.label.casefold())


def saved_voice_options(root: str | Path | None = None) -> list[str]:
    entries = scan_saved_voices(root)
    return [item.label for item in entries] or ["未找到音色包（请放入 voices 目录）"]


def saved_voice_fingerprint(root: str | Path | None = None) -> str:
    library_root = Path(root or default_voice_library_root()).expanduser().resolve()
    digest = hashlib.sha256()
    if library_root.is_dir():
        for path in sorted(library_root.glob(f"*{VOICE_BUNDLE_SUFFIX}")):
            try:
                stat = path.stat()
            except OSError:
                continue
            digest.update(path.name.encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _resolve_entry(selection: str, root: str | Path | None = None) -> SavedVoiceEntry:
    entries = scan_saved_voices(root)
    needle = str(selection or "").strip()
    for entry in entries:
        if needle in {entry.label, entry.key}:
            return entry
    raise ValueError(
        "没有找到所选音色。请把 Desktop 导出的 .t8voice.zip 放到 "
        f"{default_voice_library_root()}，再刷新节点下拉列表。"
    )


def _cache_file(entry: SavedVoiceEntry, member_name: str, root: Path) -> Path:
    member = _safe_member(member_name)
    stat = entry.bundle_path.stat()
    bundle_key = hashlib.sha256(
        f"{entry.bundle_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:20]
    destination = (root / ".cache" / bundle_key / Path(*member.parts)).resolve()
    cache_root = (root / ".cache" / bundle_key).resolve()
    if cache_root not in destination.parents:
        raise ValueError("音色包音频路径越界。")
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(entry.bundle_path) as archive:
            try:
                source = archive.open(str(member).replace("\\", "/"))
            except KeyError as exc:
                raise ValueError(f"音色包缺少音频文件：{member}") from exc
            with source, destination.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    return destination


def _load_audio(path: Path) -> dict[str, Any]:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.to(dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def load_saved_voice(
    selection: str,
    *,
    root: str | Path | None = None,
    role_name_override: str = "",
    language_override: str = "saved",
) -> tuple[VoiceProfile, dict[str, Any]]:
    entry = _resolve_entry(selection, root)
    library_root = Path(root or default_voice_library_root()).expanduser().resolve()
    raw = entry.profile
    speaker_path = _cache_file(entry, str(raw.get("audio_path") or ""), library_root)
    emotion_mode = str(raw.get("emotion_mode") or "speaker")
    emotion: EmotionConfig | None = None
    if emotion_mode == "reference_audio":
        member = str(raw.get("emotion_audio_path") or "")
        if not member:
            raise ValueError("保存的角色使用情感参考模式，但音色包缺少情感音频。")
        emotion_path = _cache_file(entry, member, library_root)
        emotion = EmotionConfig(
            mode="reference_audio",
            reference_audio=_load_audio(emotion_path),
            strength=float(raw.get("emotion_strength", 1.0)),
            use_random=bool(raw.get("emotion_use_random", False)),
        )
    elif emotion_mode == "vector":
        vector = tuple(float(item) for item in raw.get("emotion_vector") or ())
        if len(vector) != 8:
            raise ValueError("保存的八维情感向量必须正好包含 8 个值。")
        emotion = EmotionConfig(
            mode="vector",
            vector=vector,
            strength=float(raw.get("emotion_strength", 1.0)),
            use_random=bool(raw.get("emotion_use_random", False)),
        )
    elif emotion_mode == "text":
        emotion = EmotionConfig(
            mode="text",
            text=str(raw.get("emotion_text") or ""),
            strength=float(raw.get("emotion_strength", 1.0)),
            use_random=bool(raw.get("emotion_use_random", False)),
        )
    elif emotion_mode != "speaker":
        raise ValueError(f"保存的角色情感模式无效：{emotion_mode}")
    language = str(raw.get("language") or "ZH").upper()
    override = str(language_override or "saved").upper()
    if override != "SAVED":
        language = override
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"保存的角色语言无效：{language}")
    name = str(role_name_override or raw.get("name") or "").strip()
    if not name:
        raise ValueError("保存的角色名称为空。")
    profile = VoiceProfile(name, _load_audio(speaker_path), language, emotion)
    report = {
        "name": name,
        "language": language,
        "bundle": str(entry.bundle_path),
        "profileId": raw.get("profile_id"),
        "tags": list(raw.get("tags") or []),
        "favorite": bool(raw.get("favorite", False)),
        "notes": str(raw.get("notes") or ""),
        "quality": dict(raw.get("quality") or {}),
        "emotionMode": emotion_mode,
        "speakerAudio": str(speaker_path),
    }
    return profile, report


__all__ = [
    "SavedVoiceEntry",
    "default_voice_library_root",
    "load_saved_voice",
    "saved_voice_fingerprint",
    "saved_voice_options",
    "scan_saved_voices",
]
