"""Non-destructive duration and speech post-processing for ComfyUI AUDIO."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .audio_adapter import validate_comfy_audio


DURATION_MODES = ("off", "natural", "pad", "exact")
POSTPROCESS_PRESETS = ("off", "voice_clarity", "clear_narration", "deharsh", "warm", "normalize")


def audio_duration_ms(audio: dict[str, Any]) -> float:
    waveform = validate_comfy_audio(audio)
    return waveform.shape[-1] * 1000.0 / int(audio["sample_rate"])


def apply_duration_policy(
    audio: dict[str, Any],
    target_seconds: float,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_mode = str(mode or "off").lower()
    if normalized_mode not in DURATION_MODES:
        raise ValueError("目标时长模式只能是 off、natural、pad 或 exact。")
    waveform = validate_comfy_audio(audio)
    sample_rate = int(audio["sample_rate"])
    original_samples = waveform.shape[-1]
    target = float(target_seconds or 0)
    if normalized_mode == "off" or target <= 0:
        return {"waveform": waveform, "sample_rate": sample_rate}, {
            "mode": "off",
            "target_ms": 0,
            "original_ms": round(original_samples * 1000 / sample_rate),
            "final_ms": round(original_samples * 1000 / sample_rate),
            "action": "unchanged",
        }
    if not 0.1 <= target <= 3600:
        raise ValueError("目标时长必须在 0.1–3600 秒。")
    target_samples = max(1, round(target * sample_rate))
    action = "unchanged"
    result = waveform
    if original_samples < target_samples and normalized_mode in {"pad", "exact"}:
        result = F.pad(waveform, (0, target_samples - original_samples))
        action = "padded"
    elif original_samples > target_samples and normalized_mode == "exact":
        result = waveform[..., :target_samples].clone()
        fade_samples = min(round(sample_rate * 0.02), target_samples)
        if fade_samples > 1:
            fade = torch.linspace(1.0, 0.0, fade_samples, dtype=result.dtype)
            result[..., -fade_samples:] *= fade
        action = "trimmed"
    elif original_samples > target_samples and normalized_mode == "pad":
        action = "overrun_preserved"
    return {"waveform": result.contiguous(), "sample_rate": sample_rate}, {
        "mode": normalized_mode,
        "target_ms": round(target_samples * 1000 / sample_rate),
        "original_ms": round(original_samples * 1000 / sample_rate),
        "final_ms": round(result.shape[-1] * 1000 / sample_rate),
        "action": action,
    }


def _peak_normalize(waveform: torch.Tensor, target_db: float) -> torch.Tensor:
    peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    target = 10.0 ** (float(target_db) / 20.0)
    return waveform * (target / peak).clamp(max=20.0)


def _compress(waveform: torch.Tensor, threshold_db: float = -18.0, ratio: float = 3.0) -> torch.Tensor:
    threshold = 10.0 ** (threshold_db / 20.0)
    magnitude = waveform.abs().clamp_min(1e-8)
    compressed = torch.where(
        magnitude > threshold,
        threshold * torch.pow(magnitude / threshold, 1.0 / ratio),
        magnitude,
    )
    return waveform.sign() * compressed


def _filters(waveform: torch.Tensor, sample_rate: int, preset: str) -> torch.Tensor:
    try:
        import torchaudio.functional as AF
    except Exception as exc:  # pragma: no cover - ComfyUI always supplies torchaudio
        raise RuntimeError("音频后处理需要与 PyTorch 匹配的 torchaudio。") from exc
    channels = waveform.reshape(-1, waveform.shape[-1])
    if preset in {"voice_clarity", "clear_narration", "warm"}:
        channels = AF.highpass_biquad(channels, sample_rate, 70.0)
    if preset == "voice_clarity":
        channels = AF.equalizer_biquad(channels, sample_rate, 3200.0, gain=3.0, Q=0.7)
    elif preset == "clear_narration":
        channels = AF.equalizer_biquad(channels, sample_rate, 2800.0, gain=2.5, Q=0.8)
        channels = _compress(channels, -20.0, 3.0)
    elif preset == "deharsh":
        channels = AF.equalizer_biquad(channels, sample_rate, 4800.0, gain=-4.0, Q=1.1)
        channels = AF.lowpass_biquad(channels, sample_rate, min(10_500.0, sample_rate * 0.45))
    elif preset == "warm":
        channels = AF.equalizer_biquad(channels, sample_rate, 220.0, gain=3.0, Q=0.8)
        channels = AF.equalizer_biquad(channels, sample_rate, 5200.0, gain=-1.5, Q=0.9)
    elif preset == "normalize":
        pass
    return channels.reshape_as(waveform)


def postprocess_audio(
    audio: dict[str, Any],
    preset: str = "off",
    strength: float = 1.0,
    target_peak_db: float = -1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = str(preset or "off").lower()
    if normalized not in POSTPROCESS_PRESETS:
        raise ValueError("未知音频后处理预设。")
    amount = float(strength)
    if not 0 <= amount <= 1:
        raise ValueError("后处理强度必须在 0–1。")
    if not -12 <= float(target_peak_db) <= -0.1:
        raise ValueError("目标峰值必须在 -12 至 -0.1 dBFS。")
    source = validate_comfy_audio(audio)
    sample_rate = int(audio["sample_rate"])
    before_peak = float(source.abs().max())
    if normalized == "off" or amount == 0:
        return {"waveform": source, "sample_rate": sample_rate}, {
            "preset": "off", "strength": amount, "peak_before": before_peak, "peak_after": before_peak
        }
    processed = _filters(source, sample_rate, normalized)
    mixed = source * (1.0 - amount) + processed * amount
    mixed = _peak_normalize(mixed, float(target_peak_db)).clamp(-1.0, 1.0).contiguous()
    if not torch.isfinite(mixed).all():
        raise RuntimeError("音频后处理产生了非法数值。")
    return {"waveform": mixed, "sample_rate": sample_rate}, {
        "preset": normalized,
        "strength": amount,
        "target_peak_db": float(target_peak_db),
        "peak_before": before_peak,
        "peak_after": float(mixed.abs().max()),
    }


def concatenate_with_pauses(
    audios: list[dict[str, Any]], pause_after_ms: list[int], leading_pause_ms: int = 0
) -> dict[str, Any]:
    if not audios or len(audios) != len(pause_after_ms):
        raise ValueError("音频块和停顿数量必须一致且不能为空。")
    sample_rate = int(audios[0]["sample_rate"])
    waveforms: list[torch.Tensor] = []
    channels = 1
    validated: list[torch.Tensor] = []
    for audio in audios:
        if int(audio["sample_rate"]) != sample_rate:
            raise ValueError("音频块采样率不一致。")
        waveform = validate_comfy_audio(audio)
        channels = max(channels, waveform.shape[1])
        validated.append(waveform)
    leading_samples = round(max(0, int(leading_pause_ms)) * sample_rate / 1000)
    if leading_samples:
        waveforms.append(torch.zeros((1, channels, leading_samples), dtype=validated[0].dtype))
    for waveform, millis in zip(validated, pause_after_ms):
        if waveform.shape[1] != channels:
            waveform = waveform[:, :1].repeat(1, channels, 1)
        waveforms.append(waveform)
        silence_samples = round(max(0, int(millis)) * sample_rate / 1000)
        if silence_samples:
            waveforms.append(torch.zeros((1, channels, silence_samples), dtype=waveform.dtype))
    return {"waveform": torch.cat(waveforms, dim=-1), "sample_rate": sample_rate}


__all__ = [
    "DURATION_MODES",
    "POSTPROCESS_PRESETS",
    "apply_duration_policy",
    "audio_duration_ms",
    "concatenate_with_pauses",
    "postprocess_audio",
]
