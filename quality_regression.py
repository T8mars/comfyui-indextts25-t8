"""Deterministic multilingual audio-quality regression helpers.

The real-model runner lives in ``desktop/scripts/smoke-multilingual-quality.py``.
This module intentionally contains only lightweight analysis and comparison logic
so the quality gates can be unit-tested without loading IndexTTS or Whisper.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from statistics import median
from typing import Any

import numpy as np


QUALITY_CASES: tuple[dict[str, str], ...] = (
    {
        "id": "zh_narration",
        "language": "ZH",
        "text": (
            "清晨的城市刚刚苏醒，街角的面包店已经亮起温暖的灯。"
            "行人放慢脚步，听见树叶在微风里轻轻作响。"
            "这段测试用于检查长文本分段后的语速、清晰度和稳定性。"
        ),
    },
    {
        "id": "en_narration",
        "language": "EN",
        "text": (
            "The station was quiet when the first train arrived. "
            "A soft announcement echoed across the platform while the morning light grew brighter. "
            "This sample checks pronunciation, pacing, and long-form stability."
        ),
    },
    {
        "id": "ja_narration",
        "language": "JA",
        "text": (
            "朝の駅には静かな風が吹いていました。"
            "最初の電車が到着すると、ホームに案内放送が響きます。"
            "この文章で発音と速度の安定性を確認します。"
        ),
    },
    {
        "id": "es_narration",
        "language": "ES",
        "text": (
            "La ciudad despertaba lentamente cuando llegó el primer tren. "
            "Una voz tranquila anunció la salida mientras la luz llenaba la estación. "
            "Esta prueba revisa la pronunciación, el ritmo y la estabilidad."
        ),
    },
    {
        "id": "ar_narration",
        "language": "AR",
        "text": (
            "استيقظت المدينة بهدوء مع وصول القطار الأول. "
            "انتشر صوت الإعلان في المحطة بينما أصبح ضوء الصباح أكثر وضوحا. "
            "يختبر هذا النص النطق والإيقاع واستقرار المقاطع الطويلة."
        ),
    },
)

SUPPORTED_LANGUAGES = tuple(case["language"] for case in QUALITY_CASES)


def _mono_float32(waveform: Any) -> np.ndarray:
    """Return a finite mono float32 array in nominal ``[-1, 1]`` scale."""

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    array = np.asarray(waveform)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        channel_axis = 0 if array.shape[0] <= 8 else -1
        array = np.mean(array, axis=channel_axis)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.size and float(np.nanmax(np.abs(array))) > 2.0:
        array = array / 32768.0
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)


def analyze_waveform(
    waveform: Any,
    sample_rate: int,
    *,
    silence_dbfs: float = -45.0,
    frame_ms: float = 20.0,
) -> dict[str, float | int]:
    """Measure duration, peak/RMS, clipping, DC offset, and framed silence."""

    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError("sample_rate must be positive")
    mono = _mono_float32(waveform)
    if mono.size == 0:
        raise ValueError("waveform must contain at least one sample")
    absolute = np.abs(mono)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    clip_ratio = float(np.mean(absolute >= 0.999))
    frame_size = max(1, round(rate * float(frame_ms) / 1000.0))
    padded = int(math.ceil(mono.size / frame_size) * frame_size)
    framed = np.pad(mono, (0, padded - mono.size)).reshape(-1, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1))
    silence_threshold = 10.0 ** (float(silence_dbfs) / 20.0)
    return {
        "sample_rate": rate,
        "samples": int(mono.size),
        "duration_seconds": round(mono.size / rate, 4),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-12)), 3),
        "dc_offset": round(float(np.mean(mono, dtype=np.float64)), 7),
        "clipping_ratio": round(clip_ratio, 8),
        "silence_ratio": round(float(np.mean(frame_rms <= silence_threshold)), 6),
    }


def summarize_segment_rates(
    segment_reports: Iterable[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Summarize eligible internal-segment rates without hiding outliers."""

    rates = [
        float(item.get("units_per_second") or 0.0)
        for item in segment_reports
        if item.get("eligible") and float(item.get("units_per_second") or 0.0) > 0
    ]
    suspects = sum(bool(item.get("suspect")) for item in segment_reports)
    accepted = sum(bool(item.get("accepted")) for item in segment_reports)
    if not rates:
        return {
            "eligible_segments": 0,
            "suspect_segments": suspects,
            "accepted_retries": accepted,
            "median_units_per_second": 0.0,
            "coefficient_of_variation": 0.0,
            "slowest_to_median_ratio": 0.0,
        }
    rate_array = np.asarray(rates, dtype=np.float64)
    center = float(median(rates))
    return {
        "eligible_segments": len(rates),
        "suspect_segments": suspects,
        "accepted_retries": accepted,
        "median_units_per_second": round(center, 4),
        "coefficient_of_variation": round(
            float(np.std(rate_array) / center) if center > 0 else 0.0, 4
        ),
        "slowest_to_median_ratio": round(
            float(np.min(rate_array) / center) if center > 0 else 0.0, 4
        ),
    }


def _case_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("id")): item
        for item in report.get("cases", [])
        if isinstance(item, Mapping) and item.get("id")
    }


def build_baseline_snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a portable quality baseline without local paths or full transcripts."""

    cases: list[dict[str, Any]] = []
    for case in report.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        asr = case.get("asr") if isinstance(case.get("asr"), Mapping) else {}
        cases.append(
            {
                "id": case.get("id"),
                "language": case.get("language"),
                "rtf": case.get("rtf"),
                "peak_vram_bytes": case.get("peak_vram_bytes"),
                "audio": dict(case.get("audio") or {}),
                "segment_rates": dict(case.get("segment_rates") or {}),
                "asr": {
                    key: asr.get(key)
                    for key in (
                        "enabled",
                        "backend",
                        "package_version",
                        "model",
                        "metric",
                        "error_rate",
                        "similarity",
                    )
                    if key in asr
                },
            }
        )
    return {
        "schema_version": report.get("schema_version", 1),
        "baseline_kind": "indextts25-multilingual-gpu",
        "created_at": report.get("created_at"),
        "platform": report.get("platform"),
        "python": report.get("python"),
        "torch": report.get("torch"),
        "device": report.get("device"),
        "precision": report.get("precision"),
        "vram_profile": dict(report.get("vram_profile") or {}),
        "seed": report.get("seed"),
        "asr_runtime": dict(report.get("asr_runtime") or {}),
        "cases": cases,
    }


def compare_quality_reports(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two reports using conservative, explainable regression gates."""

    previous = _case_map(baseline)
    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []
    for case in current.get("cases", []):
        if not isinstance(case, Mapping) or not case.get("id"):
            continue
        case_id = str(case["id"])
        old = previous.get(case_id)
        if old is None:
            warnings.append(f"{case_id}: baseline case is missing")
            continue
        audio = case.get("audio") or {}
        old_audio = old.get("audio") or {}
        current_rtf = float(case.get("rtf") or 0.0)
        old_rtf = float(old.get("rtf") or 0.0)
        rtf_ratio = current_rtf / old_rtf if current_rtf > 0 and old_rtf > 0 else 0.0
        current_error = case.get("asr", {}).get("error_rate")
        old_error = old.get("asr", {}).get("error_rate")
        error_delta = (
            float(current_error) - float(old_error)
            if current_error is not None and old_error is not None
            else None
        )
        clip_ratio = float(audio.get("clipping_ratio") or 0.0)
        silence_delta = float(audio.get("silence_ratio") or 0.0) - float(
            old_audio.get("silence_ratio") or 0.0
        )
        case_failures: list[str] = []
        case_warnings: list[str] = []
        if rtf_ratio > 1.35:
            case_failures.append(f"RTF increased {rtf_ratio:.2f}x")
        elif rtf_ratio > 1.20:
            case_warnings.append(f"RTF increased {rtf_ratio:.2f}x")
        if error_delta is not None and error_delta > 0.10:
            case_failures.append(f"ASR error rate increased {error_delta:.3f}")
        elif error_delta is not None and error_delta > 0.05:
            case_warnings.append(f"ASR error rate increased {error_delta:.3f}")
        if clip_ratio > 0.001:
            case_failures.append(f"clipping ratio is {clip_ratio:.6f}")
        if silence_delta > 0.15:
            case_warnings.append(f"silence ratio increased {silence_delta:.3f}")
        failures.extend(f"{case_id}: {item}" for item in case_failures)
        warnings.extend(f"{case_id}: {item}" for item in case_warnings)
        comparisons.append(
            {
                "id": case_id,
                "rtf_ratio": round(rtf_ratio, 4),
                "asr_error_rate_delta": (
                    round(error_delta, 4) if error_delta is not None else None
                ),
                "silence_ratio_delta": round(silence_delta, 4),
                "failures": case_failures,
                "warnings": case_warnings,
            }
        )
    return {
        "status": "failed" if failures else "passed_with_warnings" if warnings else "passed",
        "failures": failures,
        "warnings": warnings,
        "cases": comparisons,
    }


def evaluate_quality_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Apply absolute sanity gates when no previous baseline is available."""

    failures: list[str] = []
    warnings: list[str] = []
    for case in report.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        case_id = str(case.get("id") or "unknown")
        audio = case.get("audio") or {}
        duration = float(audio.get("duration_seconds") or 0.0)
        clipping = float(audio.get("clipping_ratio") or 0.0)
        silence = float(audio.get("silence_ratio") or 0.0)
        if duration < 1.0:
            failures.append(f"{case_id}: generated audio is too short ({duration:.2f}s)")
        if clipping > 0.001:
            failures.append(f"{case_id}: clipping ratio is {clipping:.6f}")
        if silence > 0.75:
            failures.append(f"{case_id}: silence ratio is {silence:.3f}")
        elif silence > 0.50:
            warnings.append(f"{case_id}: silence ratio is {silence:.3f}")
        rate = case.get("segment_rates") or {}
        if int(rate.get("eligible_segments") or 0) >= 3:
            slow_ratio = float(rate.get("slowest_to_median_ratio") or 0.0)
            if 0 < slow_ratio < 0.35:
                failures.append(
                    f"{case_id}: slowest segment is only {slow_ratio:.3f} of the median"
                )
    return {
        "status": "failed" if failures else "passed_with_warnings" if warnings else "passed",
        "failures": failures,
        "warnings": warnings,
    }


__all__ = [
    "QUALITY_CASES",
    "SUPPORTED_LANGUAGES",
    "analyze_waveform",
    "build_baseline_snapshot",
    "compare_quality_reports",
    "evaluate_quality_report",
    "summarize_segment_rates",
]
