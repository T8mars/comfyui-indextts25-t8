from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

import torch

from .audio_adapter import INDEXTTS_SAMPLE_RATE, validate_comfy_audio


MAX_REFERENCE_SECONDS = 15.0
MIN_REFERENCE_SECONDS = 0.25
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.Lock] = {}


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _cache_root() -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.get_temp_directory())
    except Exception:
        import tempfile

        root = Path(tempfile.gettempdir())
    path = root / "indextts25_t8" / "references"
    path.mkdir(parents=True, exist_ok=True)
    return path


def comfy_audio_to_reference_wav(audio: dict[str, Any], *, kind: str) -> tuple[Path, tuple[str, ...]]:
    waveform = validate_comfy_audio(audio, name=kind)
    sample_rate = int(audio["sample_rate"])
    mono = waveform.mean(dim=1)
    seconds = mono.shape[-1] / sample_rate
    if seconds < MIN_REFERENCE_SECONDS:
        raise ValueError(f"{kind} 太短（{seconds:.2f} 秒）；至少需要 {MIN_REFERENCE_SECONDS:.2f} 秒。")

    notes: list[str] = []
    max_samples = int(MAX_REFERENCE_SECONDS * sample_rate)
    if mono.shape[-1] > max_samples:
        mono = mono[..., :max_samples]
        limit = f"{MAX_REFERENCE_SECONDS:g}"
        notes.append(f"{kind} 超过 {limit} 秒，已截取前 {limit} 秒。")

    if sample_rate != INDEXTTS_SAMPLE_RATE:
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("缺少 torchaudio，无法转换参考音频采样率。") from exc
        mono = torchaudio.functional.resample(mono, sample_rate, INDEXTTS_SAMPLE_RATE)

    mono = mono.clamp(-1.0, 1.0).contiguous()
    digest = hashlib.sha256()
    digest.update(str(INDEXTTS_SAMPLE_RATE).encode("ascii"))
    digest.update(mono.numpy().tobytes())
    key = digest.hexdigest()
    target = _cache_root() / f"{kind}-{key}.wav"
    if target.is_file():
        return target, tuple(notes)

    with _lock_for(key):
        if not target.is_file():
            try:
                import torchaudio
            except ImportError as exc:
                raise RuntimeError("缺少 torchaudio，无法保存参考音频。") from exc
            temporary = target.with_name(f".{target.stem}-{os.getpid()}-{threading.get_ident()}.tmp.wav")
            try:
                torchaudio.save(str(temporary), mono, INDEXTTS_SAMPLE_RATE)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink(missing_ok=True)
    return target, tuple(notes)
