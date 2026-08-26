from __future__ import annotations

import pytest

from runtime import text_planner


class _CharacterTokenizer:
    def encode(self, text, allowed_special="all"):
        return list(text)


def test_language_aware_limits_are_conservative_for_latin_text():
    assert text_planner.auto_segment_limit("EN") == 60
    assert text_planner.auto_segment_limit("ES") == 60
    assert text_planner.auto_segment_limit("ZH") == 120
    assert text_planner.effective_segment_limit("JA", "custom", 75) == 75


def test_pause_planner_preserves_pronunciation_annotations_and_explicit_seconds():
    chunks = text_planner.split_speech_chunks(
        "<银行|YIN2 HANG2>到了。<pause=0.5>下一句，继续。",
        pause_preset="custom",
        comma_pause_ms=100,
        sentence_pause_ms=300,
        paragraph_pause_ms=600,
    )
    assert chunks[0].text == "<银行|YIN2 HANG2>到了。"
    assert chunks[0].pause_after_ms == 500
    assert chunks[1].text == "下一句，"
    assert chunks[1].pause_after_ms == 100
    assert chunks[-1].pause_after_ms == 300


def test_plan_reports_segments_pauses_and_gpt_cache_risk(tmp_path, monkeypatch):
    monkeypatch.setattr(text_planner, "_load_tokenizer", lambda _path: _CharacterTokenizer())
    plan = text_planner.build_generation_plan(
        "First sentence. Second sentence.",
        "EN",
        tmp_path,
        segmentation_mode="auto",
        pause_preset="natural",
    )
    payload = plan.to_dict()
    assert payload["max_tokens"] == 60
    assert payload["speech_blocks"] == 2
    assert payload["total_pause_ms"] == 520
    assert payload["gpt_accel_risk"] is False
    assert payload["gpt_accel_cache_fix"] is True
    assert all(item["token_count"] <= 60 for item in payload["segments"])


def test_explicit_pause_rejects_unbounded_silence():
    with pytest.raises(ValueError, match="0–30 秒"):
        text_planner.split_speech_chunks("hello<pause=31>world")


def test_leading_explicit_pause_is_preserved_in_preview_and_total(tmp_path, monkeypatch):
    monkeypatch.setattr(text_planner, "_load_tokenizer", lambda _path: _CharacterTokenizer())
    plan = text_planner.build_generation_plan(
        "<pause=250ms>Hello.",
        "EN",
        tmp_path,
        pause_preset="off",
    )
    assert plan.chunks[0].pause_before_ms == 250
    assert plan.segments[0].pause_before_ms == 250
    assert plan.total_pause_ms == 250


def test_decimal_punctuation_does_not_create_false_pause_boundaries():
    chunks = text_planner.split_speech_chunks(
        "Version 3.14 costs 1,000.50 dollars.",
        pause_preset="custom",
        comma_pause_ms=120,
        sentence_pause_ms=300,
        paragraph_pause_ms=600,
    )
    assert len(chunks) == 1
    assert chunks[0].text == "Version 3.14 costs 1,000.50 dollars."
    assert chunks[0].pause_after_ms == 300
