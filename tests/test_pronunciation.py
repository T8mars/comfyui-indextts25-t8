from __future__ import annotations

import pytest

from runtime.pronunciation import (
    PronunciationEntry,
    PronunciationValidationError,
    format_pronunciation_report,
    parse_dictionary_text,
    process_pronunciation_text,
)


def test_pipe_dictionary_is_portable_and_longest_term_wins():
    entries = parse_dictionary_text("行|XING2|ZH\n银行|YIN2 HANG2|ZH")
    result = process_pronunciation_text("银行能办理。", "ZH", entries, strict=True)
    assert result.text == "<银行|YIN2 HANG2>能办理。"
    assert "银行 → YIN2 HANG2" in format_pronunciation_report(result)


def test_existing_inline_annotation_has_priority():
    entries = [PronunciationEntry("行", "XING2", "ZH")]
    result = process_pronunciation_text("银<行|HANG2>里行走", "ZH", entries, strict=True)
    assert result.text == "银<行|HANG2>里<行|XING2>走"


def test_english_and_japanese_validation():
    english = process_pronunciation_text("a <minute|M IH1 . N AH0 T>", "EN", strict=True)
    japanese = process_pronunciation_text("<上手|じょうず>", "JA", strict=True)
    assert english.ok and japanese.ok


def test_strict_mode_rejects_invalid_dictionary_reading():
    entries = parse_dictionary_text("坏词|BAD9|ZH")
    with pytest.raises(PronunciationValidationError):
        process_pronunciation_text("坏词", "ZH", entries, strict=True)


def test_issue_792_whole_word_annotation_is_the_safe_form():
    risky = process_pronunciation_text(
        "小明<要|YAO4>求这个题的答案是多少，该做什么呢？", "ZH", strict=True
    )
    assert any("<要求|YAO4 QIU2>" in item for item in risky.warnings)
    robust = process_pronunciation_text(
        "小明<要求|YAO4 QIU2>这个题的答案是多少，该做什么呢？", "ZH", strict=True
    )
    assert robust.warnings == ()


def test_chinese_annotation_warns_on_mismatched_syllable_count():
    result = process_pronunciation_text("请到<银行|HANG2>办理。", "ZH", strict=True)
    assert any("2 个汉字" in item and "1 个拼音音节" in item for item in result.warnings)
