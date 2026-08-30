import numpy as np

from quality_regression import (
    analyze_waveform,
    build_baseline_snapshot,
    compare_quality_reports,
    evaluate_quality_report,
    summarize_segment_rates,
)


def test_waveform_metrics_measure_clipping_silence_and_duration():
    waveform = np.concatenate(
        [np.zeros(1000, dtype=np.float32), np.full(1000, 0.5, dtype=np.float32)]
    )
    report = analyze_waveform(waveform, 1000, frame_ms=100)
    assert report["duration_seconds"] == 2.0
    assert report["peak"] == 0.5
    assert report["clipping_ratio"] == 0.0
    assert 0.45 <= report["silence_ratio"] <= 0.55


def test_segment_rate_summary_keeps_suspects_visible():
    summary = summarize_segment_rates(
        [
            {"eligible": True, "units_per_second": 4.0},
            {"eligible": True, "units_per_second": 4.2},
            {
                "eligible": True,
                "units_per_second": 1.0,
                "suspect": True,
                "accepted": True,
            },
        ]
    )
    assert summary["eligible_segments"] == 3
    assert summary["suspect_segments"] == 1
    assert summary["accepted_retries"] == 1
    assert summary["slowest_to_median_ratio"] == 0.25


def test_quality_report_comparison_detects_speed_and_asr_regressions():
    baseline = {
        "cases": [
            {
                "id": "en_narration",
                "rtf": 0.5,
                "audio": {"clipping_ratio": 0.0, "silence_ratio": 0.1},
                "asr": {"error_rate": 0.05},
            }
        ]
    }
    current = {
        "cases": [
            {
                "id": "en_narration",
                "rtf": 0.8,
                "audio": {"clipping_ratio": 0.0, "silence_ratio": 0.12},
                "asr": {"error_rate": 0.2},
            }
        ]
    }
    result = compare_quality_reports(current, baseline)
    assert result["status"] == "failed"
    assert any("RTF" in item for item in result["failures"])
    assert any("ASR" in item for item in result["failures"])


def test_absolute_quality_gate_rejects_nearly_empty_audio():
    result = evaluate_quality_report(
        {
            "cases": [
                {
                    "id": "zh_narration",
                    "audio": {
                        "duration_seconds": 0.2,
                        "clipping_ratio": 0.0,
                        "silence_ratio": 0.9,
                    },
                    "segment_rates": {"eligible_segments": 0},
                }
            ]
        }
    )
    assert result["status"] == "failed"
    assert len(result["failures"]) == 2


def test_portable_baseline_strips_paths_and_transcripts():
    result = build_baseline_snapshot(
        {
            "model_dir": "C:/private/model",
            "reference_voice": "C:/private/voice.wav",
            "asr_runtime": {"package_version": "20250625", "model": "base"},
            "cases": [
                {
                    "id": "en_narration",
                    "language": "EN",
                    "output": "C:/private/result.wav",
                    "rtf": 0.5,
                    "audio": {"duration_seconds": 5.0},
                    "segment_rates": {"eligible_segments": 1},
                    "asr": {
                        "enabled": True,
                        "backend": "openai_whisper",
                        "package_version": "20250625",
                        "model": "base",
                        "metric": "WER",
                        "error_rate": 0.1,
                        "recognized_text": "private transcript",
                    },
                }
            ],
        }
    )
    serialized = str(result)
    assert "C:/private" not in serialized
    assert "private transcript" not in serialized
    assert result["cases"][0]["asr"]["error_rate"] == 0.1
