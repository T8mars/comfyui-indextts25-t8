from __future__ import annotations

from runtime.context_emotion import (
    build_context_prompt,
    normalize_emotion_scores,
    suggest_context_emotions,
)
from runtime.dialogue import DialogueLine


def _lines():
    return [
        DialogueLine(1, "甲", "我们终于成功了。"),
        DialogueLine(2, "乙", "等等，结果好像不对。"),
        DialogueLine(3, "甲", "什么？你再说一遍！"),
    ]


def test_context_prompt_marks_target_and_surrounding_roles() -> None:
    prompt, indexes = build_context_prompt(_lines(), 1, 1)
    assert indexes == [1, 2, 3]
    assert prompt.endswith("【只分析这一句】#2 乙：等等，结果好像不对。")
    assert "输出 IndexTTS 八维" not in prompt


def test_normalize_scores_supports_chinese_keys_and_caps_sum() -> None:
    vector, raw = normalize_emotion_scores({"愤怒": 1.2, "惊讶": 0.4})
    assert len(vector) == 8
    assert round(sum(vector), 6) == 0.8
    assert raw["angry"] == 1.2


def test_suggestions_require_confirmation_and_preserve_existing() -> None:
    source = _lines()
    source[0] = DialogueLine(
        1, "甲", source[0].text, emotion_mode="text", emotion_text="克制地高兴"
    )
    updated, report = suggest_context_emotions(
        source, lambda _prompt: {"愤怒": 0.9, "惊讶": 0.3}, context_window=1
    )
    assert updated[0].emotion_mode == "text"
    assert updated[1].emotion_mode == "vector"
    assert report["preserved_count"] == 1
    assert report["classified_count"] == 2
    assert report["requires_user_confirmation"] is True
    assert report["started_synthesis"] is False
