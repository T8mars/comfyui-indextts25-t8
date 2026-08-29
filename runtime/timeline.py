"""Timeline editing, visualization metadata, and SRT rewriting."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from typing import Any, Sequence

import torch

from .dialogue import (
    MAX_TIMELINE_MS,
    DialogueLine,
    format_emotion_override,
    parse_emotion_override,
)


TIMELINE_HEADERS = [
    "index",
    "role",
    "language",
    "start_ms",
    "end_ms",
    "duration_factor",
    "text",
    "emotion",
]


def timeline_rows(lines: Sequence[DialogueLine]) -> list[list[Any]]:
    return [
        [
            line.index,
            line.role,
            line.language,
            line.start_ms,
            line.end_ms,
            line.duration_factor,
            line.text,
            format_emotion_override(line),
        ]
        for line in lines
    ]


def _optional_ms(value, position: int, label: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        result = int(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"时间轴第 {position} 行{label}必须是整数毫秒。") from exc
    if result < 0 or result > MAX_TIMELINE_MS:
        raise ValueError(
            f"时间轴第 {position} 行{label}必须在 0–{MAX_TIMELINE_MS} 毫秒之间。"
        )
    return result


def apply_timeline_edits(
    original_lines: Sequence[DialogueLine], rows
) -> list[DialogueLine]:
    if isinstance(rows, str):
        try:
            rows = json.loads(rows)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"时间轴 JSON 格式错误：{exc.msg}（第 {exc.lineno} 行）"
            ) from exc
    if isinstance(rows, dict):
        rows = rows.get("lines") or rows.get("data") or []
    if not rows:
        return list(original_lines)
    if len(rows) != len(original_lines):
        raise ValueError("时间轴编辑行数必须与脚本台词数量一致。")
    originals = {line.index: line for line in original_lines}
    result, seen = [], set()
    for position, row in enumerate(rows, 1):
        value = row if isinstance(row, dict) else dict(zip(TIMELINE_HEADERS, row))
        index = int(value.get("index", position))
        if index not in originals or index in seen:
            raise ValueError(f"时间轴第 {position} 行序号重复或不存在：{index}")
        seen.add(index)
        role, text = (
            str(value.get("role") or "").strip(),
            str(value.get("text") or "").strip(),
        )
        language = str(value.get("language") or "").upper()
        if not role or not text:
            raise ValueError(f"时间轴第 {position} 行角色和台词不能为空。")
        if language not in {"ZH", "EN", "JA", "ES", "AR"}:
            raise ValueError(f"时间轴第 {position} 行语言无效：{language}")
        start = _optional_ms(value.get("start_ms"), position, "开始时间")
        end = _optional_ms(value.get("end_ms"), position, "结束时间")
        if (start is None) != (end is None) or (start is not None and end <= start):
            raise ValueError(f"时间轴第 {position} 行开始/结束时间无效。")
        factor = float(value.get("duration_factor", 1.0))
        if not 0.5 <= factor <= 2.0:
            raise ValueError(f"时间轴第 {position} 行时长系数必须在 0.5–2.0。")
        original = originals[index]
        language_explicit = bool(
            value.get("language_explicit", original.language_explicit)
        ) or language != original.language
        if "emotion" in value:
            emotion = parse_emotion_override(value.get("emotion"), index=index)
        elif any(
            key in value
            for key in (
                "emotion_mode",
                "emotion_text",
                "emotion_vector",
                "emotion_strength",
                "emotion_use_random",
            )
        ):
            emotion = parse_emotion_override(value, index=index)
        else:
            emotion = (
                original.emotion_mode,
                original.emotion_text,
                original.emotion_vector,
                original.emotion_strength,
                original.emotion_use_random,
            )
        result.append(
            replace(
                original,
                role=role,
                text=text,
                language=language,
                start_ms=start,
                end_ms=end,
                duration_factor=factor,
                emotion_mode=emotion[0],
                emotion_text=emotion[1],
                emotion_vector=emotion[2],
                emotion_strength=emotion[3],
                emotion_use_random=emotion[4],
                language_explicit=language_explicit,
            )
        )
    return result


def format_srt_timestamp(milliseconds: int) -> str:
    value = max(0, int(milliseconds))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _normalize_choice(
    value, choices: tuple[str, ...], numeric_choices: tuple[str, ...], default: str
) -> tuple[str, str | None]:
    raw = str(value if value is not None else "").strip().lower()
    if raw in choices:
        return raw, None
    try:
        index = int(float(raw))
    except (TypeError, ValueError, OverflowError):
        index = -1
    if 0 <= index < len(numeric_choices):
        normalized = numeric_choices[index]
        return normalized, f"旧工作流字幕选项 {value!r} 已兼容为 {normalized!r}。"
    return default, f"无法识别字幕选项 {value!r}，已使用安全默认值 {default!r}。"


def rewrite_srt(
    lines: Sequence[DialogueLine],
    line_reports: Sequence[dict] | None = None,
    *,
    timing_mode: str = "actual",
    text_mode: str = "asr_passed",
    include_role: bool = True,
) -> tuple[str, dict[str, Any]]:
    requested_timing, requested_text = timing_mode, text_mode
    timing_mode, timing_warning = _normalize_choice(
        timing_mode,
        ("original", "actual"),
        ("actual", "original"),
        "actual",
    )
    text_mode, text_warning = _normalize_choice(
        text_mode,
        ("original", "asr_passed", "asr_all"),
        ("asr_passed", "asr_all", "original"),
        "asr_passed",
    )
    reports = {
        int(item.get("index", position)): item
        for position, item in enumerate(line_reports or (), 1)
        if isinstance(item, dict)
    }
    blocks, rows, cursor = [], [], 0
    timing_warnings: list[str] = []
    for output_index, line in enumerate(lines, 1):
        report, timeline = (
            reports.get(line.index, {}),
            (reports.get(line.index, {}).get("timeline") or {}),
        )
        actual_range: tuple[int, int] | None = None
        if timing_mode == "actual" and timeline:
            try:
                actual_start = int(timeline.get("actual_start_ms", cursor))
                actual_end = int(timeline.get("actual_end_ms", cursor + 1))
            except (TypeError, ValueError, OverflowError):
                actual_start = actual_end = -1
            if 0 <= actual_start < actual_end <= MAX_TIMELINE_MS:
                actual_range = (actual_start, actual_end)
            else:
                timing_warnings.append(
                    f"第 {line.index} 条实际时间无效，已回退到原始或顺延时间。"
                )
        if actual_range is not None:
            start, end = actual_range
        elif line.start_ms is not None and line.end_ms is not None:
            start, end = int(line.start_ms), int(line.end_ms)
        else:
            try:
                duration = int(report.get("actual_duration_ms", 1000))
            except (TypeError, ValueError, OverflowError):
                duration = 1000
            duration = max(1, duration)
            if cursor >= MAX_TIMELINE_MS:
                raise ValueError(f"字幕时间不能超过 {MAX_TIMELINE_MS} 毫秒。")
            start, end = (
                cursor,
                min(MAX_TIMELINE_MS, cursor + duration),
            )
        end, cursor = max(start + 1, end), max(cursor, end)
        asr = report.get("asr") or {}
        recognized = str(asr.get("recognized_text") or "").strip()
        use_asr = bool(
            recognized
            and (
                text_mode == "asr_all"
                or (text_mode == "asr_passed" and asr.get("passed"))
            )
        )
        text = recognized if use_asr else line.text
        if include_role:
            emotion_tag = format_emotion_override(line)
            role_tag = f"{line.role}|emotion={emotion_tag}" if emotion_tag else line.role
            rendered = f"[{role_tag}] {text}"
        else:
            rendered = text
        blocks.append(
            f"{output_index}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{rendered}"
        )
        rows.append(
            {
                "index": line.index,
                "start_ms": start,
                "end_ms": end,
                "source": "asr" if use_asr else "original",
                "text": text,
            }
        )
    warnings = [item for item in (timing_warning, text_warning) if item]
    warnings.extend(timing_warnings)
    return "\n\n".join(blocks) + ("\n" if blocks else ""), {
        "timing_mode": timing_mode,
        "text_mode": text_mode,
        "include_role": bool(include_role),
        "requested_timing_mode": requested_timing,
        "requested_text_mode": requested_text,
        "warnings": warnings,
        "lines": rows,
    }


def timeline_json(
    lines: Sequence[DialogueLine], line_reports: Sequence[dict] | None = None
) -> str:
    return json.dumps(
        {
            "lines": [line.to_dict() for line in lines],
            "reports": list(line_reports or ()),
        },
        ensure_ascii=False,
        indent=2,
    )


def render_timeline_image(
    lines: Sequence[DialogueLine], width: int = 1200, row_height: int = 56
) -> torch.Tensor:
    """Return a standard ComfyUI IMAGE tensor with one colored track per dialogue line."""
    width = max(320, int(width))
    row_height = max(36, int(row_height))
    height = max(96, 32 + len(lines) * row_height)
    image = torch.full((1, height, width, 3), 0.055, dtype=torch.float32)
    if not lines:
        return image

    entries, cursor = [], 0
    for line in lines:
        start = int(line.start_ms if line.start_ms is not None else cursor)
        end = int(line.end_ms if line.end_ms is not None else start + 1000)
        end = max(start + 1, end)
        cursor = max(cursor, end)
        entries.append((line, start, end))
    total = max(end for _line, _start, end in entries)
    left, right = 20, width - 20
    track_width = max(1, right - left)

    for fraction in range(11):
        x = min(width - 1, left + round(track_width * fraction / 10))
        image[:, :, x : x + 1, :] = 0.16
    for row_index, (line, start, end) in enumerate(entries):
        y0 = 32 + row_index * row_height
        y1 = min(height, y0 + row_height - 8)
        image[:, y0:y1, left:right, :] = 0.09
        x0 = left + round(track_width * start / total)
        x1 = left + round(track_width * end / total)
        x0 = min(right - 1, max(left, x0))
        x1 = min(right, max(x0 + 2, x1))
        digest = hashlib.sha256(line.role.encode("utf-8")).digest()
        color = torch.tensor(
            [0.35 + digest[0] / 425, 0.35 + digest[1] / 425, 0.35 + digest[2] / 425],
            dtype=torch.float32,
        ).clamp(max=0.95)
        image[:, y0 + 8 : y1 - 6, x0:x1, :] = color
        image[:, y0 + 8 : y1 - 6, x0 : min(x0 + 3, x1), :] = 1.0
    return image


__all__ = [
    "TIMELINE_HEADERS",
    "apply_timeline_edits",
    "format_srt_timestamp",
    "render_timeline_image",
    "rewrite_srt",
    "timeline_json",
    "timeline_rows",
]
