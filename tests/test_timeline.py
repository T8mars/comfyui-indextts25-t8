from __future__ import annotations

import pytest

from runtime.dialogue import DialogueLine
from runtime.timeline import apply_timeline_edits, rewrite_srt, timeline_json, timeline_rows


def test_timeline_json_edits_and_srt_rewrite():
    lines = [
        DialogueLine(1, "A", "第一句", "ZH", 0, 1000, 1.0),
        DialogueLine(2, "B", "second", "EN", 1100, 2000, 1.0),
    ]
    rows = timeline_rows(lines)
    rows[0][3:5] = [100, 900]
    edited = apply_timeline_edits(lines, rows)
    assert edited[0].slot_ms == 800
    assert "\"lines\"" in timeline_json(edited)

    reports = [
        {"index": 1, "timeline": {"actual_start_ms": 50, "actual_end_ms": 850}, "asr": {"recognized_text": "识别一", "passed": True}},
        {"index": 2, "timeline": {"actual_start_ms": 900, "actual_end_ms": 1800}, "asr": {"recognized_text": "wrong", "passed": False}},
    ]
    srt, report = rewrite_srt(edited, reports, timing_mode="actual", text_mode="asr_passed")
    assert "00:00:00,050 --> 00:00:00,850" in srt
    assert "[A] 识别一" in srt
    assert "[B] second" in srt
    assert report["lines"][0]["source"] == "asr"

    rows[1][4] = None
    with pytest.raises(ValueError, match="开始/结束"):
        apply_timeline_edits(lines, rows)


def test_srt_rewrite_accepts_legacy_numeric_and_shifted_widget_values():
    lines = [DialogueLine(1, "旁白", "保留原始字幕", "ZH", 0, 1000, 1.0)]
    rewritten, report = rewrite_srt(
        lines,
        [],
        timing_mode=0,
        text_mode="actual",
    )
    assert "保留原始字幕" in rewritten
    assert report["timing_mode"] == "actual"
    assert report["text_mode"] == "asr_passed"
    assert len(report["warnings"]) == 2

def test_timeline_editor_rejects_unbounded_allocations():
    lines = [DialogueLine(1, "旁白", "测试", "ZH")]
    with pytest.raises(ValueError, match="0–86400000"):
        apply_timeline_edits(
            lines,
            [
                {
                    "index": 1,
                    "role": "旁白",
                    "text": "测试",
                    "language": "ZH",
                    "start_ms": 0,
                    "end_ms": 90_000_000,
                    "duration_factor": 1.0,
                }
            ],
        )
