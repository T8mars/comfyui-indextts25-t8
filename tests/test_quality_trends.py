from quality_trends import build_quality_trend, render_quality_trend_markdown, render_quality_trend_svg


def test_quality_trend_is_path_free_and_renders_all_formats():
    report = {"created_at": "2026-08-30T12:00:00+0800", "model_dir": "C:/private/model", "vram_profile": {"name": "8gb"}, "cases": [{"id": "ar_narration", "language": "AR", "rtf": 0.7, "peak_vram_bytes": 6 * 1024**3, "audio": {"duration_seconds": 5}, "asr": {"model": "small", "error_rate": 0.2}}]}
    trend = build_quality_trend([report])
    assert trend["points"][0]["profile"] == "8gb"
    assert trend["points"][0]["mean_asr_error_rate"] == 0.2
    assert "C:/private" not in str(trend)
    assert "8gb" in render_quality_trend_markdown(trend)
    assert "<polyline" in render_quality_trend_svg(trend)
