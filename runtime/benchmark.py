from __future__ import annotations

import statistics


def summarize_measurements(measurements: list[dict]) -> dict:
    if not measurements:
        raise ValueError("基准测量结果不能为空。")
    rtf_values = [float(item["rtf"]) for item in measurements]
    inference_values = [float(item["inference_seconds"]) for item in measurements]
    audio_values = [float(item["audio_seconds"]) for item in measurements]
    peak_values = [
        float(item["peak_vram_gb"])
        for item in measurements
        if item.get("peak_vram_gb") is not None
    ]
    return {
        "repeat_count": len(measurements),
        "median_rtf": round(statistics.median(rtf_values), 6),
        "best_rtf": round(min(rtf_values), 6),
        "median_inference_seconds": round(statistics.median(inference_values), 6),
        "median_audio_seconds": round(statistics.median(audio_values), 6),
        "peak_vram_gb": round(max(peak_values), 6) if peak_values else None,
        "realtime": statistics.median(rtf_values) < 1.0,
    }
