"""Reference-audio diagnostics, safe trimming, and waveform rendering."""

from __future__ import annotations

import html
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def _mono_waveform(waveform) -> torch.Tensor:
    audio = torch.as_tensor(waveform).detach().float().cpu()
    while audio.ndim > 2:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or audio.shape[-1] == 0:
        raise ValueError("参考音频为空或波形维度无效。")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if not torch.isfinite(audio).all():
        raise ValueError("参考音频包含 NaN 或 Inf。")
    if audio.numel() and float(audio.abs().max()) > 2.0:
        audio = audio / 32768.0
    return audio.clamp(-1, 1).contiguous()


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-9))


def _frame_rms(audio: torch.Tensor, sample_rate: int, frame_ms: int = 30) -> torch.Tensor:
    frame = max(1, round(int(sample_rate) * frame_ms / 1000))
    padded = F.pad(audio, (0, (-audio.shape[-1]) % frame))
    return padded.square().unfold(-1, frame, frame).mean(-1).sqrt().squeeze(0)


def analyze_reference_audio(waveform, sample_rate: int) -> dict[str, Any]:
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("采样率必须大于 0。")
    audio = _mono_waveform(waveform)
    frames = _frame_rms(audio, sample_rate)
    silence_threshold = 10 ** (-45 / 20)
    active = frames >= silence_threshold
    active_indexes = torch.nonzero(active, as_tuple=False).flatten()
    frame_seconds = 0.03
    duration = audio.shape[-1] / sample_rate
    if active_indexes.numel():
        leading = min(duration, float(active_indexes[0]) * frame_seconds)
        trailing = min(duration, max(0.0, duration - (float(active_indexes[-1]) + 1) * frame_seconds))
        active_rms = frames[active]
        noise_rms = frames[~active]
        signal_level = float(active_rms.median())
        noise_level = float(noise_rms.median()) if noise_rms.numel() else max(signal_level / 20, 1e-6)
        snr_db = max(0.0, min(80.0, _db(signal_level / max(noise_level, 1e-9))))
    else:
        leading, trailing, snr_db = duration, 0.0, 0.0
    peak = float(audio.abs().max())
    rms = float(audio.square().mean().sqrt())
    clipped_ratio = float((audio.abs() >= 0.999).float().mean())
    silence_ratio = float((~active).float().mean()) if frames.numel() else 1.0
    dc_offset = float(audio.mean().abs())
    score = 100
    issues: list[str] = []
    recommendations: list[str] = []
    if duration < 1.5:
        score -= 45
        issues.append("有效时长过短")
        recommendations.append("请提供至少 3 秒的单人清晰语音。")
    elif duration < 3:
        score -= 15
        issues.append("参考音频短于推荐的 3 秒")
    elif duration > 15:
        score -= 8
        issues.append("参考音频超过 15 秒")
        recommendations.append("建议裁剪为信息密度最高的 3–10 秒，最长不超过 15 秒。")
    if silence_ratio > 0.45:
        score -= 25
        issues.append("静音占比过高")
        recommendations.append("裁掉首尾空白并避免句间长停顿。")
    elif silence_ratio > 0.25:
        score -= 10
        issues.append("静音占比较高")
    if clipped_ratio >= 0.01:
        score -= 35
        issues.append("存在明显削波失真")
        recommendations.append("降低录音增益，避免波形顶到 0 dBFS。")
    elif clipped_ratio > 0:
        score -= 10
        issues.append("检测到少量削波样本")
    if _db(rms) < -35:
        score -= 15
        issues.append("整体音量过低")
        recommendations.append("提高人声录制电平或先做响度归一化。")
    elif _db(rms) > -6:
        score -= 10
        issues.append("整体音量过高")
    if snr_db < 15:
        score -= 25
        issues.append("估算信噪比较低")
        recommendations.append("换用更干净、无背景音乐和环境噪声的参考音频。")
    elif snr_db < 25:
        score -= 10
        issues.append("估算信噪比一般")
    if dc_offset > 0.02:
        score -= 10
        issues.append("直流偏移较明显")
    score = max(0, min(100, score))
    usable = bool(duration >= 1.5 and active.any() and clipped_ratio < 0.05)
    grade = "优秀" if score >= 85 else "可用" if score >= 65 else "较差"
    return {
        "sample_rate": sample_rate,
        "channels": 1,
        "duration_seconds": round(duration, 3),
        "active_seconds": round(max(0.0, duration - leading - trailing), 3),
        "leading_silence_seconds": round(leading, 3),
        "trailing_silence_seconds": round(trailing, 3),
        "silence_ratio": round(silence_ratio, 6),
        "peak_dbfs": round(_db(peak), 3),
        "rms_dbfs": round(_db(rms), 3),
        "clipped_ratio": round(clipped_ratio, 8),
        "snr_estimate_db": round(snr_db, 3),
        "dc_offset": round(dc_offset, 8),
        "score": score,
        "grade": grade,
        "usable": usable,
        "issues": issues,
        "recommendations": recommendations,
        "limits": {"recommended_seconds": [3, 10], "maximum_seconds": 15, "silence_threshold_dbfs": -45},
    }


def prepare_reference_audio(
    waveform,
    sample_rate: int,
    *,
    trim_silence: bool = True,
    max_seconds: float = 15.0,
    padding_ms: int = 150,
) -> tuple[torch.Tensor, dict[str, Any]]:
    audio = _mono_waveform(waveform)
    sample_rate = int(sample_rate)
    original_samples = audio.shape[-1]
    start, end = 0, original_samples
    if trim_silence:
        frames = _frame_rms(audio, sample_rate)
        active = torch.nonzero(frames >= 10 ** (-45 / 20), as_tuple=False).flatten()
        frame = max(1, round(sample_rate * 0.03))
        padding = max(0, round(sample_rate * int(padding_ms) / 1000))
        if active.numel():
            start = max(0, int(active[0]) * frame - padding)
            end = min(original_samples, (int(active[-1]) + 1) * frame + padding)
    audio = audio[..., start:end]
    maximum = max(1, round(sample_rate * float(max_seconds)))
    selected_offset = 0
    if audio.shape[-1] > maximum:
        hop = max(1, round(sample_rate * 0.1))
        starts = torch.arange(0, audio.shape[-1] - maximum + 1, hop)
        if starts[-1] != audio.shape[-1] - maximum:
            starts = torch.cat([starts, torch.tensor([audio.shape[-1] - maximum])])
        energy_prefix = F.pad(audio.square().squeeze(0).cumsum(0), (1, 0))
        energies = energy_prefix[starts + maximum] - energy_prefix[starts]
        selected_offset = int(starts[int(torch.argmax(energies))])
        audio = audio[..., selected_offset : selected_offset + maximum]
    final_start = start + selected_offset
    report = {
        "original": analyze_reference_audio(waveform, sample_rate),
        "prepared": analyze_reference_audio(audio, sample_rate),
        "trimmed": bool(final_start or final_start + audio.shape[-1] < original_samples),
        "selected_start_seconds": round(final_start / sample_rate, 3),
        "selected_end_seconds": round((final_start + audio.shape[-1]) / sample_rate, 3),
    }
    return audio.contiguous(), report


def _waveform_columns(audio: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    samples = audio.squeeze(0)
    width = max(1, int(width))
    if samples.numel() < width:
        samples = F.interpolate(samples.view(1, 1, -1), size=width, mode="linear", align_corners=False).view(-1)
        return samples, samples
    edges = torch.linspace(0, samples.numel(), width + 1, dtype=torch.long)
    lows, highs = [], []
    for index in range(width):
        part = samples[edges[index] : max(edges[index] + 1, edges[index + 1])]
        lows.append(part.min())
        highs.append(part.max())
    return torch.stack(lows), torch.stack(highs)


def render_waveform_image(
    waveform,
    sample_rate: int,
    word_timestamps: Sequence[dict[str, Any]] | None = None,
    *,
    width: int = 1200,
    height: int = 280,
) -> torch.Tensor:
    audio = _mono_waveform(waveform)
    width, height = max(320, int(width)), max(120, int(height))
    image = torch.full((1, height, width, 3), 0.045, dtype=torch.float32)
    center, amplitude = height // 2, max(20, height // 2 - 24)
    image[:, center : center + 1, :, :] = 0.18
    lows, highs = _waveform_columns(audio, width)
    for x, (low, high) in enumerate(zip(lows, highs)):
        y0 = max(4, min(height - 5, center - round(float(high) * amplitude)))
        y1 = max(y0 + 1, min(height - 4, center - round(float(low) * amplitude)))
        image[:, y0:y1, x, :] = torch.tensor([0.98, 0.25, 0.62])
    duration = max(audio.shape[-1] / int(sample_rate), 1e-6)
    for item in word_timestamps or ():
        start = max(0.0, float(item.get("start") or 0.0))
        x = min(width - 1, round(width * start / duration))
        image[:, :, x : x + 1, :] = torch.tensor([0.25, 0.78, 1.0])
    return image


def waveform_html(
    waveform,
    sample_rate: int,
    word_timestamps: Sequence[dict[str, Any]] | None = None,
    *,
    width: int = 1000,
    height: int = 220,
) -> str:
    audio = _mono_waveform(waveform)
    lows, highs = _waveform_columns(audio, width)
    center, amplitude = height / 2, height / 2 - 16
    bars = "".join(
        f'<line x1="{x}" y1="{center-float(high)*amplitude:.1f}" x2="{x}" y2="{center-float(low)*amplitude:.1f}" />'
        for x, (low, high) in enumerate(zip(lows, highs))
    )
    duration = max(audio.shape[-1] / int(sample_rate), 1e-6)
    markers = []
    for item in word_timestamps or ():
        start, end = float(item.get("start") or 0), float(item.get("end") or 0)
        x = max(0.0, min(width, width * start / duration))
        label = html.escape(str(item.get("word") or ""), quote=True)
        markers.append(
            f'<line class="word" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" '
            f'data-end="{end:.3f}"><title>{label} · {start:.3f}–{end:.3f}s</title></line>'
        )
    return (
        '<div class="t8-waveform-scroll"><svg class="t8-waveform" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="音频波形与逐字时间戳">'
        f'<g class="wave">{bars}</g><g>{"".join(markers)}</g></svg>'
        f'<div class="t8-waveform-caption">0s · {duration:.2f}s · 蓝线为逐字时间戳</div></div>'
    )


__all__ = [
    "analyze_reference_audio",
    "prepare_reference_audio",
    "render_waveform_image",
    "waveform_html",
]
