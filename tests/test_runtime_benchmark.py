from __future__ import annotations

from runtime.benchmark import summarize_measurements


def test_summarize_measurements_uses_median_and_peak() -> None:
    report = summarize_measurements([
        {"rtf": 0.8, "inference_seconds": 4, "audio_seconds": 5, "peak_vram_gb": 7.5},
        {"rtf": 0.6, "inference_seconds": 3, "audio_seconds": 5, "peak_vram_gb": 8.0},
        {"rtf": 1.2, "inference_seconds": 6, "audio_seconds": 5, "peak_vram_gb": 7.8},
    ])
    assert report["median_rtf"] == 0.8
    assert report["best_rtf"] == 0.6
    assert report["peak_vram_gb"] == 8.0
    assert report["realtime"] is True
