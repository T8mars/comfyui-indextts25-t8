"""Context-aware per-line emotion suggestions for ComfyUI dialogue workflows."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence


EMOTION_ORDER = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)
EMOTION_LABELS_ZH = ("喜", "怒", "哀", "惧", "厌恶", "低落", "惊喜", "平静")
_ALIASES = {
    "happy": ("happy", "高兴", "喜"),
    "angry": ("angry", "愤怒", "怒"),
    "sad": ("sad", "悲伤", "哀"),
    "afraid": ("afraid", "恐惧", "惧"),
    "disgusted": ("disgusted", "反感", "厌恶"),
    "melancholic": ("melancholic", "低落"),
    "surprised": ("surprised", "惊讶", "惊喜"),
    "calm": ("calm", "自然", "平静"),
}
CONTEXT_WEIGHT = 0.35


def _line_excerpt(line: Any, limit: int = 240) -> str:
    text = " ".join(str(getattr(line, "text", "") or "").split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return f"#{getattr(line, 'index', '?')} {getattr(line, 'role', '角色')}：{text}"


def build_context_prompt(
    lines: Sequence[Any], target_position: int, context_window: int = 2
) -> tuple[str, list[int]]:
    if not lines:
        raise ValueError("情感分析至少需要一条台词。")
    position = int(target_position)
    if not 0 <= position < len(lines):
        raise IndexError("目标台词序号超出范围。")
    window = max(0, min(5, int(context_window)))
    start = max(0, position - window)
    end = min(len(lines), position + window + 1)
    context_indexes = [int(getattr(lines[index], "index", index + 1)) for index in range(start, end)]
    before = [_line_excerpt(lines[index]) for index in range(start, position)]
    after = [_line_excerpt(lines[index]) for index in range(position + 1, end)]
    sections = []
    if before:
        sections.append("【此前对话】\n" + "\n".join(before))
    if after:
        sections.append("【随后对话】\n" + "\n".join(after))
    # Keep the target last: the bundled classifier follows this compact form much
    # more reliably than a long instruction block that enumerates emotion labels.
    sections.append("【只分析这一句】" + _line_excerpt(lines[position]))
    return "\n\n".join(sections), context_indexes


def normalize_emotion_scores(values: Any) -> tuple[tuple[float, ...], dict[str, float]]:
    if isinstance(values, dict):
        lowered = {str(key).strip().lower(): value for key, value in values.items()}
        raw = [
            next(
                (lowered[alias.lower()] for alias in _ALIASES[key] if alias.lower() in lowered),
                0.0,
            )
            for key in EMOTION_ORDER
        ]
    elif isinstance(values, (list, tuple)):
        raw = list(values)
    else:
        raise ValueError("情感分类器必须返回八维数组或情感字典。")
    if len(raw) != 8:
        raise ValueError("情感分类器必须返回正好 8 个情感数值。")
    try:
        scores = [max(0.0, min(1.2, float(item))) for item in raw]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("情感分类结果包含无效数值。") from exc
    if not any(scores):
        scores[-1] = 1.0
    total = sum(scores)
    scale = min(1.0, 0.8 / total) if total else 1.0
    vector = tuple(round(score * scale, 6) for score in scores)
    return vector, dict(zip(EMOTION_ORDER, scores))


def suggest_context_emotions(
    lines: Sequence[Any],
    classifier: Callable[[str], Any],
    *,
    context_window: int = 2,
    overwrite_existing: bool = False,
    progress: Callable[[int, int, Any], None] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    if not callable(classifier):
        raise TypeError("classifier 必须是可调用对象。")
    source = list(lines)
    if not source:
        raise ValueError("情感分析至少需要一条台词。")
    updated, suggestions = [], []
    classified = preserved = 0
    for position, line in enumerate(source):
        if progress is not None:
            progress(position, len(source), line)
        existing_mode = str(getattr(line, "emotion_mode", "inherit") or "inherit")
        if existing_mode != "inherit" and not overwrite_existing:
            updated.append(line)
            preserved += 1
            suggestions.append(
                {
                    "index": int(getattr(line, "index", position + 1)),
                    "role": str(getattr(line, "role", "")),
                    "text": str(getattr(line, "text", "")),
                    "action": "preserved_existing",
                    "emotion_mode": existing_mode,
                }
            )
            continue
        prompt, context_indexes = build_context_prompt(source, position, context_window)
        _, target_scores = normalize_emotion_scores(
            classifier(str(getattr(line, "text", "")))
        )
        if len(context_indexes) > 1:
            _, context_scores = normalize_emotion_scores(classifier(prompt))
            blended_scores = {
                key: round(
                    target_scores[key] * (1.0 - CONTEXT_WEIGHT)
                    + context_scores[key] * CONTEXT_WEIGHT,
                    6,
                )
                for key in EMOTION_ORDER
            }
        else:
            context_scores = dict(target_scores)
            blended_scores = dict(target_scores)
        vector, raw_scores = normalize_emotion_scores(blended_scores)
        dominant_index = max(range(8), key=lambda index: raw_scores[EMOTION_ORDER[index]])
        dominant_score = raw_scores[EMOTION_ORDER[dominant_index]]
        strength = round(max(0.35, min(1.0, dominant_score / 1.2)), 2)
        updated.append(
            replace(
                line,
                emotion_mode="vector",
                emotion_text="",
                emotion_vector=vector,
                emotion_strength=strength,
                emotion_use_random=False,
            )
        )
        classified += 1
        suggestions.append(
            {
                "index": int(getattr(line, "index", position + 1)),
                "role": str(getattr(line, "role", "")),
                "text": str(getattr(line, "text", "")),
                "action": "suggested",
                "context_indexes": context_indexes,
                "dominant_emotion": EMOTION_LABELS_ZH[dominant_index],
                "dominant_key": EMOTION_ORDER[dominant_index],
                "target_scores": target_scores,
                "context_scores": context_scores,
                "context_weight": CONTEXT_WEIGHT if len(context_indexes) > 1 else 0.0,
                "raw_scores": raw_scores,
                "emotion_vector": list(vector),
                "strength": strength,
            }
        )
    if progress is not None:
        progress(len(source), len(source), None)
    return updated, {
        "mode": "context_aware_qwen_emotion",
        "prompt_version": 3,
        "context_window": max(0, min(5, int(context_window))),
        "overwrite_existing": bool(overwrite_existing),
        "line_count": len(source),
        "classified_count": classified,
        "preserved_count": preserved,
        "requires_user_confirmation": True,
        "started_synthesis": False,
        "emotion_order": list(EMOTION_LABELS_ZH),
        "suggestions": suggestions,
    }


__all__ = [
    "EMOTION_LABELS_ZH",
    "EMOTION_ORDER",
    "CONTEXT_WEIGHT",
    "build_context_prompt",
    "normalize_emotion_scores",
    "suggest_context_emotions",
]
