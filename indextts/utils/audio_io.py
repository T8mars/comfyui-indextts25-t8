"""Audio I/O compatibility for Torchaudio 2.8 and TorchCodec-backed 2.9+."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import torchaudio


def _version_tuple(value: str) -> tuple[int, int]:
    parts: list[int] = []
    for chunk in str(value or "").split("+", 1)[0].split(".")[:2]:
        if not chunk.isdigit():
            return (0, 0)
        parts.append(int(chunk))
    return tuple(parts) if len(parts) == 2 else (0, 0)


def uses_torchcodec_io() -> bool:
    return _version_tuple(getattr(torchaudio, "__version__", "")) >= (2, 9)


def probe_torchcodec_runtime() -> dict[str, Any]:
    """Check the 2.9 audio runtime without opening a user audio file."""

    torchaudio_version = str(getattr(torchaudio, "__version__", ""))
    required = uses_torchcodec_io()
    try:
        torchcodec_version = metadata.version("torchcodec")
    except metadata.PackageNotFoundError:
        torchcodec_version = None
    if not required:
        return {
            "required": False,
            "ready": True,
            "torchaudio": torchaudio_version,
            "torchcodec": torchcodec_version,
            "ffmpeg_shared_libraries": "not_required",
            "reason": "Torchaudio 2.8 uses its legacy audio backend.",
        }
    if torchcodec_version is None:
        return {
            "required": True,
            "ready": False,
            "torchaudio": torchaudio_version,
            "torchcodec": None,
            "ffmpeg_shared_libraries": "unknown",
            "reason": "Torchaudio 2.9+ requires a compatible TorchCodec package.",
        }
    try:
        from torchcodec.decoders import AudioDecoder  # noqa: F401
        from torchcodec.encoders import AudioEncoder  # noqa: F401
    except Exception as exc:
        return {
            "required": True,
            "ready": False,
            "torchaudio": torchaudio_version,
            "torchcodec": torchcodec_version,
            "ffmpeg_shared_libraries": "unavailable",
            "reason": (
                "TorchCodec could not load its native audio libraries. On Windows, "
                "install discoverable FFmpeg shared DLLs; ffmpeg.exe alone is not enough. "
                f"Original error: {str(exc).strip() or type(exc).__name__}"
            ),
        }
    return {
        "required": True,
        "ready": True,
        "torchaudio": torchaudio_version,
        "torchcodec": torchcodec_version,
        "ffmpeg_shared_libraries": "loaded",
        "reason": "TorchCodec decoder/encoder and FFmpeg shared libraries loaded successfully.",
    }


def _torchcodec_error(operation: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"Torchaudio {getattr(torchaudio, '__version__', '2.9+')} requires a "
        f"TorchCodec build compatible with the installed torch for audio {operation}. "
        "Use torchcodec 0.6/0.7 with torch 2.8 or 0.8/0.9 with torch 2.9, "
        "and make sure FFmpeg is discoverable. "
        f"Original error: {exc}"
    )


def _load_with_torchcodec(source: Any) -> tuple[torch.Tensor, int]:
    from torchcodec.decoders import AudioDecoder

    samples = AudioDecoder(str(source) if isinstance(source, Path) else source).get_all_samples()
    return samples.data.to(dtype=torch.float32), int(samples.sample_rate)


def load_audio_file(source: Any) -> tuple[torch.Tensor, int]:
    if not uses_torchcodec_io():
        waveform, sample_rate = torchaudio.load(source)
        return waveform.to(dtype=torch.float32), int(sample_rate)
    try:
        return _load_with_torchcodec(source)
    except Exception as exc:
        raise _torchcodec_error("decoding", exc) from exc


def _save_with_torchcodec(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    from torchcodec.encoders import AudioEncoder

    AudioEncoder(waveform, sample_rate=int(sample_rate)).to_file(str(path))


def save_audio_file(
    path: str | Path,
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    pcm16: bool = False,
) -> None:
    audio = waveform.detach().to(device="cpu", dtype=torch.float32)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError("audio waveform must have shape [time] or [channels, time]")
    audio = audio.contiguous().clamp_(-1.0, 1.0)
    if uses_torchcodec_io():
        try:
            _save_with_torchcodec(path, audio, int(sample_rate))
            return
        except Exception as exc:
            raise _torchcodec_error("encoding", exc) from exc
    kwargs = {"encoding": "PCM_S", "bits_per_sample": 16} if pcm16 else {}
    torchaudio.save(str(path), audio, int(sample_rate), **kwargs)


__all__ = [
    "load_audio_file",
    "probe_torchcodec_runtime",
    "save_audio_file",
    "uses_torchcodec_io",
]
