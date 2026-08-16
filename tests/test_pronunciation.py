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
