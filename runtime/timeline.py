"""Timeline editing, visualization metadata, and SRT rewriting."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from typing import Any, Sequence

import torch

from .dialogue import DialogueLine


TIMELINE_HEADERS = ["index", "role", "language", "start_ms", "end_ms", "duration_factor", "text"]


def timeline_rows(lines: Sequence[DialogueLine]) -> list[list[Any]]:
    return [[line.index, line.role, line.language, line.start_ms, line.end_ms, line.duration_factor, line.text] for line in lines]


def _optional_ms(value, position: int, label: str) -> int | None:
    if value in {None, ""}:
        return None
    try:
        result = int(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"时间轴第 {position} 行{label}必须是整数毫秒。") from exc
    if result < 0 or result > 86_400_000:
        raise ValueError(f"时间轴第 {position} 行{label}必须在 0–86400000 毫秒之间。")
    return result


def apply_timeline_edits(original_lines: Sequence[DialogueLine], rows) -> list[DialogueLine]:
    if isinstance(rows, str):
        try:
            rows = json.loads(rows)
        except json.JSONDecodeError as exc:
            raise ValueError(f"时间轴 JSON 格式错误：{exc.msg}（第 {exc.lineno} 行）") from exc
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
        role, text = str(value.get("role") or "").strip(), str(value.get("text") or "").strip()
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
        result.append(replace(originals[index], role=role, text=text, language=language, start_ms=start, end_ms=end, duration_factor=factor))
    return result


def format_srt_timestamp(milliseconds: int) -> str:
    value = max(0, int(milliseconds))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def rewrite_srt(lines: Sequence[DialogueLine], line_reports: Sequence[dict] | None = None, *, timing_mode: str = "actual", text_mode: str = "asr_passed", include_role: bool = True) -> tuple[str, dict[str, Any]]:
    if timing_mode not in {"original", "actual"} or text_mode not in {"original", "asr_passed", "asr_all"}:
        raise ValueError("字幕回写模式无效。")
    reports = {int(item.get("index", position)): item for position, item in enumerate(line_reports or (), 1) if isinstance(item, dict)}
    blocks, rows, cursor = [], [], 0
    for output_index, line in enumerate(lines, 1):
        report, timeline = reports.get(line.index, {}), (reports.get(line.index, {}).get("timeline") or {})
        if timing_mode == "actual" and timeline:
            start, end = int(timeline.get("actual_start_ms", cursor)), int(timeline.get("actual_end_ms", cursor + 1))
        elif line.start_ms is not None and line.end_ms is not None:
            start, end = int(line.start_ms), int(line.end_ms)
        else:
            start, end = cursor, cursor + max(1, int(report.get("actual_duration_ms", 1000)))
        end, cursor = max(start + 1, end), max(cursor, end)
        asr = report.get("asr") or {}
        recognized = str(asr.get("recognized_text") or "").strip()
        use_asr = bool(recognized and (text_mode == "asr_all" or (text_mode == "asr_passed" and asr.get("passed"))))
        text = recognized if use_asr else line.text
        rendered = f"[{line.role}] {text}" if include_role else text
        blocks.append(f"{output_index}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{rendered}")
        rows.append({"index": line.index, "start_ms": start, "end_ms": end, "source": "asr" if use_asr else "original", "text": text})
    return "\n\n".join(blocks) + ("\n" if blocks else ""), {"timing_mode": timing_mode, "text_mode": text_mode, "include_role": bool(include_role), "lines": rows}


def timeline_json(lines: Sequence[DialogueLine], line_reports: Sequence[dict] | None = None) -> str:
    return json.dumps({"lines": [line.to_dict() for line in lines], "reports": list(line_reports or ())}, ensure_ascii=False, indent=2)


def render_timeline_image(lines: Sequence[DialogueLine], width: int = 1200, row_height: int = 56) -> torch.Tensor:
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


__all__ = ["TIMELINE_HEADERS", "apply_timeline_edits", "format_srt_timestamp", "render_timeline_image", "rewrite_srt", "timeline_json", "timeline_rows"]
