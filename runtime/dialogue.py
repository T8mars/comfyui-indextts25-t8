"""Dialogue/SRT data and waveform composition for the ComfyUI nodes."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch


LANGUAGES = {"ZH", "EN", "JA", "ES", "AR"}
_TIME = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})$")
_RANGE = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)(?:\s+.*)?$")
_BRACKET = re.compile(r"^\s*[\[【]([^\]】]+)[\]】]\s*([\s\S]*)$")
_COLON = re.compile(r"^\s*([^:：|]{1,40})\s*[:：]\s*([\s\S]+)$")


@dataclass(frozen=True, slots=True)
class DialogueLine:
    index: int
    role: str
    text: str
    language: str = "ZH"
    start_ms: int | None = None
    end_ms: int | None = None
    duration_factor: float = 1.0

    @property
    def slot_ms(self) -> int | None:
        return None if self.start_ms is None or self.end_ms is None else max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimelinePlacement:
    index: int
    requested_start_ms: int
    actual_start_ms: int
    actual_end_ms: int
    overlap_ms: int
    overrun_ms: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def parse_timestamp(value: str) -> int:
    match = _TIME.match(str(value).strip())
    if not match:
        raise ValueError(f"无效的 SRT 时间：{value}")
    hours, minutes, seconds, millis = map(int, match.groups())
    if minutes > 59 or seconds > 59:
        raise ValueError(f"无效的 SRT 时间：{value}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def split_role_text(text: str, default_role: str) -> tuple[str, str]:
    normalized = str(text).strip()
    for pattern in (_BRACKET, _COLON):
        match = pattern.match(normalized)
        if match and match.group(1).strip() and match.group(2).strip():
            return match.group(1).strip(), match.group(2).strip()
    return str(default_role).strip(), normalized


def _language(value: Any, default: str) -> str:
    result = str(value or default).strip().upper()
    if result not in LANGUAGES:
        raise ValueError(f"不支持的语言：{result}")
    return result


def parse_srt(content: str, default_role: str = "旁白", default_language: str = "ZH") -> list[DialogueLine]:
    raw = str(content).lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        raise ValueError("SRT 内容不能为空。")
    result: list[DialogueLine] = []
    for number, block in enumerate(re.split(r"\n\s*\n", raw), 1):
        rows = [row.rstrip() for row in block.splitlines() if row.strip()]
        position = next((index for index, row in enumerate(rows[:2]) if "-->" in row), -1)
        if position < 0:
            raise ValueError(f"SRT 第 {number} 段缺少时间范围。")
        match = _RANGE.match(rows[position])
        if not match:
            raise ValueError(f"SRT 第 {number} 段时间格式错误。")
        start, end = parse_timestamp(match.group(1)), parse_timestamp(match.group(2))
        if end <= start:
            raise ValueError(f"SRT 第 {number} 段结束时间必须晚于开始时间。")
        text = "\n".join(rows[position + 1 :]).strip()
        if not text:
            raise ValueError(f"SRT 第 {number} 段没有字幕文本。")
        role, text = split_role_text(text, default_role)
        result.append(DialogueLine(len(result) + 1, role, text, _language(default_language, "ZH"), start, end))
    return result


def _mapping_line(value: dict[str, Any], index: int, default_role: str, default_language: str) -> DialogueLine:
    role, text = str(value.get("role") or default_role).strip(), str(value.get("text") or "").strip()
    if not role or not text:
        raise ValueError(f"第 {index} 条台词必须包含 role 和 text。")
    factor = float(value.get("duration_factor", 1.0))
    if not 0.5 <= factor <= 2.0:
        raise ValueError(f"第 {index} 条台词的 duration_factor 必须在 0.5–2.0。")
    start, end = value.get("start_ms"), value.get("end_ms")
    return DialogueLine(
        index,
        role,
        text,
        _language(value.get("language"), default_language),
        None if start is None else int(start),
        None if end is None else int(end),
        factor,
    )


def parse_batch_script(content: str, default_role: str = "旁白", default_language: str = "ZH") -> list[DialogueLine]:
    raw = str(content).lstrip("\ufeff").strip()
    if not raw:
        raise ValueError("批量台词不能为空。")
    if raw.startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 台词格式错误：{exc.msg}（第 {exc.lineno} 行）") from exc
        if not isinstance(payload, list):
            raise ValueError("JSON 台词必须是数组。")
        invalid = next((index for index, item in enumerate(payload, 1) if not isinstance(item, dict)), None)
        if invalid is not None:
            raise ValueError(f"JSON 第 {invalid} 条台词必须是对象。")
        result = [
            _mapping_line(item, index, default_role, default_language)
            for index, item in enumerate(payload, 1)
        ]
    else:
        result = []
        for source_line, raw_line in enumerate(raw.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split("|", 3)]
            if len(parts) < 2:
                role, text = split_role_text(line, default_role)
                if role == default_role and text == line:
                    raise ValueError(f"第 {source_line} 行应为：角色|台词|语言|时长系数")
                parts = [role, text]
            result.append(_mapping_line({"role": parts[0], "text": parts[1], "language": parts[2] if len(parts) > 2 and parts[2] else default_language, "duration_factor": parts[3] if len(parts) > 3 and parts[3] else 1.0}, len(result) + 1, default_role, default_language))
    if not result:
        raise ValueError("脚本中没有可生成内容。")
    return result


def missing_roles(lines: Iterable[DialogueLine], roles: Iterable[str]) -> list[str]:
    known = {str(role).strip() for role in roles}
    return sorted({line.role for line in lines if line.role not in known})


def fit_duration_factor(current: float, actual_ms: float, target_ms: float) -> float:
    if actual_ms <= 0 or target_ms <= 0:
        return max(0.5, min(2.0, float(current)))
    return max(0.5, min(2.0, float(current) * float(target_ms) / float(actual_ms)))


def compose_timeline(clips: Sequence[torch.Tensor], lines: Sequence[DialogueLine], sample_rate: int, policy: str = "shift", gap_ms: int = 0) -> tuple[torch.Tensor, list[TimelinePlacement]]:
    if len(clips) != len(lines):
        raise ValueError("音频片段数量与台词数量不一致。")
    if policy not in {"shift", "overlay"}:
        raise ValueError("时间轴策略只能是 shift 或 overlay。")
    normalized = []
    for value in clips:
        clip = torch.as_tensor(value).detach().float().cpu()
        if clip.ndim == 1:
            clip = clip[None, None, :]
        elif clip.ndim == 2:
            clip = clip[None, :, :]
        if clip.ndim != 3:
            raise ValueError("音频张量必须为 T、CT 或 BCT。")
        normalized.append(clip[:1])
    if not normalized:
        return torch.zeros((1, 1, 0)), []
    channels, cursor, starts, reports = max(clip.shape[1] for clip in normalized), 0, [], []
    for line, clip in zip(lines, normalized):
        requested = max(0, int(line.start_ms or 0) * sample_rate // 1000)
        start = max(requested, cursor) if policy == "shift" else requested
        end = start + clip.shape[-1]
        slot_end = None if line.end_ms is None else int(line.end_ms) * sample_rate // 1000
        reports.append(TimelinePlacement(line.index, round(requested * 1000 / sample_rate), round(start * 1000 / sample_rate), round(end * 1000 / sample_rate), round(max(0, cursor - requested) * 1000 / sample_rate), 0 if slot_end is None else round(max(0, end - slot_end) * 1000 / sample_rate)))
        starts.append(start)
        cursor = max(cursor, end + max(0, int(gap_ms)) * sample_rate // 1000)
    output = torch.zeros((1, channels, max(start + clip.shape[-1] for start, clip in zip(starts, normalized))))
    active = torch.zeros(output.shape[-1])
    for start, clip in zip(starts, normalized):
        if clip.shape[1] != channels:
            clip = clip[:, :1].repeat(1, channels, 1)
        output[..., start : start + clip.shape[-1]] += clip
        active[start : start + clip.shape[-1]] += 1
    if policy == "overlay":
        output /= active.clamp_min(1)[None, None, :]
    return output.clamp(-1, 1), reports


__all__ = ["DialogueLine", "TimelinePlacement", "compose_timeline", "fit_duration_factor", "missing_roles", "parse_batch_script", "parse_srt"]
