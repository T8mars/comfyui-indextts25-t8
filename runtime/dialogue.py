"""Dialogue/SRT data and waveform composition for the ComfyUI nodes."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch


LANGUAGES = {"ZH", "EN", "JA", "ES", "AR"}
EMOTION_OVERRIDE_MODES = {"inherit", "speaker", "vector", "text"}
MAX_TIMELINE_MS = 86_400_000
MAX_TIMELINE_ALLOCATION_BYTES = 1_073_741_824
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
    emotion_mode: str = "inherit"
    emotion_text: str = ""
    emotion_vector: tuple[float, ...] | None = None
    emotion_strength: float = 1.0
    emotion_use_random: bool = False

    @property
    def slot_ms(self) -> int | None:
        return (
            None
            if self.start_ms is None or self.end_ms is None
            else max(0, self.end_ms - self.start_ms)
        )

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
    result = ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis
    if result > MAX_TIMELINE_MS:
        raise ValueError(f"SRT 时间不能超过 {MAX_TIMELINE_MS} 毫秒：{value}")
    return result


def parse_emotion_override(
    value: Any,
    *,
    index: int | None = None,
) -> tuple[str, str, tuple[float, ...] | None, float, bool]:
    """Normalize a per-line emotion override from compact text or JSON data."""

    label = f"第 {index} 条台词" if index is not None else "逐句情感"
    if value is None:
        value = ""
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("{") or raw.startswith("["):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} JSON 格式错误：{exc.msg}") from exc
        else:
            for prefix in ("emotion=", "emotion:", "情感=", "情感："):
                if raw.lower().startswith(prefix.lower()):
                    raw = raw[len(prefix) :].strip()
                    break
            lowered = raw.lower()
            if lowered in {"", "inherit", "default", "继承", "角色默认"}:
                value = {"mode": "inherit"}
            elif lowered in {"speaker", "voice", "跟随音色", "音色"}:
                value = {"mode": "speaker"}
            elif lowered.startswith(("vector:", "vector：", "向量:", "向量：")):
                value = {"mode": "vector", "vector": re.split(r"[:：]", raw, maxsplit=1)[1]}
            elif lowered.startswith(("text:", "text：", "文本:", "文本：")):
                value = {"mode": "text", "text": re.split(r"[:：]", raw, maxsplit=1)[1]}
            else:
                value = {"mode": "text", "text": raw}
    elif isinstance(value, (list, tuple)):
        value = {"mode": "vector", "vector": value}
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是文本、八维数组或 JSON 对象。")

    mode = str(value.get("mode") or value.get("emotion_mode") or "").strip().lower()
    vector_value = value.get("vector", value.get("emotion_vector"))
    text = str(value.get("text", value.get("emotion_text", "")) or "").strip()
    if not mode:
        mode = "vector" if vector_value is not None else ("text" if text else "inherit")
    mode = {
        "default": "inherit",
        "role": "inherit",
        "voice": "speaker",
        "description": "text",
    }.get(mode, mode)
    if mode not in EMOTION_OVERRIDE_MODES:
        raise ValueError(f"{label}模式无效：{mode}；可用 inherit、speaker、text、vector。")
    try:
        strength = float(value.get("strength", value.get("emotion_strength", 1.0)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}强度必须是 0–1 的数值。") from exc
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"{label}强度必须在 0–1。")
    use_random = bool(value.get("use_random", value.get("emotion_use_random", False)))

    vector: tuple[float, ...] | None = None
    if mode == "vector":
        if isinstance(vector_value, str):
            vector_value = [item for item in re.split(r"[,，\s]+", vector_value.strip()) if item]
        try:
            vector = tuple(float(item) for item in (vector_value or ()))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label}八维向量必须全部是数值。") from exc
        if len(vector) != 8:
            raise ValueError(f"{label}八维向量必须正好包含 8 个数值。")
        if any(not 0.0 <= item <= 1.0 for item in vector):
            raise ValueError(f"{label}八维向量的每个数值必须在 0–1。")
        total = sum(vector)
        if total > 0.8:
            vector = tuple(item * 0.8 / total for item in vector)
    return mode, text, vector, strength, use_random


def format_emotion_override(line: DialogueLine) -> str:
    if line.emotion_mode == "inherit":
        return ""
    payload: dict[str, Any] = {"mode": line.emotion_mode}
    if line.emotion_mode == "text":
        payload["text"] = line.emotion_text
    elif line.emotion_mode == "vector":
        payload["vector"] = list(line.emotion_vector or ())
    if line.emotion_strength != 1.0:
        payload["strength"] = line.emotion_strength
    if line.emotion_use_random:
        payload["use_random"] = True
    if set(payload) == {"mode"}:
        return str(payload["mode"])
    if line.emotion_strength == 1.0 and not line.emotion_use_random:
        if line.emotion_mode == "text":
            return f"text:{line.emotion_text}"
        if line.emotion_mode == "vector":
            return "vector:" + ",".join(f"{item:g}" for item in line.emotion_vector or ())
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def split_role_text_emotion(text: str, default_role: str) -> tuple[str, str, Any]:
    normalized = str(text).strip()
    for pattern in (_BRACKET, _COLON):
        match = pattern.match(normalized)
        if match and match.group(1).strip() and match.group(2).strip():
            role = match.group(1).strip()
            emotion: Any = None
            if pattern is _BRACKET and "|" in role:
                role, emotion = (part.strip() for part in role.split("|", 1))
            return role, match.group(2).strip(), emotion
    return str(default_role).strip(), normalized, None


def split_role_text(text: str, default_role: str) -> tuple[str, str]:
    role, body, _emotion = split_role_text_emotion(text, default_role)
    return role, body


def _language(value: Any, default: str) -> str:
    result = str(value or default).strip().upper()
    if result not in LANGUAGES:
        raise ValueError(f"不支持的语言：{result}")
    return result


def parse_srt(
    content: str, default_role: str = "旁白", default_language: str = "ZH"
) -> list[DialogueLine]:
    raw = (
        str(content).lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    if not raw:
        raise ValueError("SRT 内容不能为空。")
    result: list[DialogueLine] = []
    for number, block in enumerate(re.split(r"\n\s*\n", raw), 1):
        rows = [row.rstrip() for row in block.splitlines() if row.strip()]
        position = next(
            (index for index, row in enumerate(rows[:2]) if "-->" in row), -1
        )
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
        role, text, emotion = split_role_text_emotion(text, default_role)
        emotion_mode, emotion_text, emotion_vector, emotion_strength, emotion_use_random = (
            parse_emotion_override(emotion, index=len(result) + 1)
        )
        result.append(
            DialogueLine(
                len(result) + 1,
                role,
                text,
                _language(default_language, "ZH"),
                start,
                end,
                emotion_mode=emotion_mode,
                emotion_text=emotion_text,
                emotion_vector=emotion_vector,
                emotion_strength=emotion_strength,
                emotion_use_random=emotion_use_random,
            )
        )
    return result


def _mapping_line(
    value: dict[str, Any], index: int, default_role: str, default_language: str
) -> DialogueLine:
    role, text = (
        str(value.get("role") or default_role).strip(),
        str(value.get("text") or "").strip(),
    )
    if not role or not text:
        raise ValueError(f"第 {index} 条台词必须包含 role 和 text。")
    factor = float(value.get("duration_factor", 1.0))
    if not 0.5 <= factor <= 2.0:
        raise ValueError(f"第 {index} 条台词的 duration_factor 必须在 0.5–2.0。")
    start = _optional_timeline_ms(value.get("start_ms"), index, "start_ms")
    end = _optional_timeline_ms(value.get("end_ms"), index, "end_ms")
    if (start is None) != (end is None) or (start is not None and end <= start):
        raise ValueError(
            f"第 {index} 条台词的 start_ms/end_ms 必须同时填写，且结束时间晚于开始时间。"
        )
    emotion_value = value.get("emotion")
    if emotion_value is None and any(
        key in value
        for key in (
            "emotion_mode",
            "emotion_text",
            "emotion_vector",
            "emotion_strength",
            "emotion_use_random",
        )
    ):
        emotion_value = value
    emotion_mode, emotion_text, emotion_vector, emotion_strength, emotion_use_random = (
        parse_emotion_override(emotion_value, index=index)
    )
    return DialogueLine(
        index,
        role,
        text,
        _language(value.get("language"), default_language),
        start,
        end,
        factor,
        emotion_mode,
        emotion_text,
        emotion_vector,
        emotion_strength,
        emotion_use_random,
    )


def _optional_timeline_ms(value: Any, index: int, label: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        result = int(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"第 {index} 条台词的 {label} 必须是整数毫秒。") from exc
    if not 0 <= result <= MAX_TIMELINE_MS:
        raise ValueError(
            f"第 {index} 条台词的 {label} 必须在 0–{MAX_TIMELINE_MS} 毫秒之间。"
        )
    return result


def _split_batch_fields(line: str) -> list[str]:
    """Split role/text/language/factor without breaking <text|reading> annotations."""
    fields: list[str] = []
    current: list[str] = []
    annotation_depth = 0
    for character in line:
        if character == "<":
            annotation_depth += 1
        elif character == ">" and annotation_depth:
            annotation_depth -= 1
        if character == "|" and annotation_depth == 0 and len(fields) < 4:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    fields.append("".join(current).strip())
    return fields


def parse_batch_script(
    content: str, default_role: str = "旁白", default_language: str = "ZH"
) -> list[DialogueLine]:
    raw = str(content).lstrip("\ufeff").strip()
    if not raw:
        raise ValueError("批量台词不能为空。")
    if raw.startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON 台词格式错误：{exc.msg}（第 {exc.lineno} 行）"
            ) from exc
        if not isinstance(payload, list):
            raise ValueError("JSON 台词必须是数组。")
        invalid = next(
            (
                index
                for index, item in enumerate(payload, 1)
                if not isinstance(item, dict)
            ),
            None,
        )
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
            parts = _split_batch_fields(line)
            if len(parts) < 2:
                role, text = split_role_text(line, default_role)
                if role == default_role and text == line:
                    raise ValueError(
                        f"第 {source_line} 行应为：角色|台词|语言|时长系数|逐句情感（可选）"
                    )
                parts = [role, text]
            result.append(
                _mapping_line(
                    {
                        "role": parts[0],
                        "text": parts[1],
                        "language": parts[2]
                        if len(parts) > 2 and parts[2]
                        else default_language,
                        "duration_factor": parts[3]
                        if len(parts) > 3 and parts[3]
                        else 1.0,
                        "emotion": parts[4]
                        if len(parts) > 4 and parts[4]
                        else None,
                    },
                    len(result) + 1,
                    default_role,
                    default_language,
                )
            )
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


def compose_timeline(
    clips: Sequence[torch.Tensor],
    lines: Sequence[DialogueLine],
    sample_rate: int,
    policy: str = "shift",
    gap_ms: int = 0,
) -> tuple[torch.Tensor, list[TimelinePlacement]]:
    if len(clips) != len(lines):
        raise ValueError("音频片段数量与台词数量不一致。")
    if policy not in {"shift", "overlay"}:
        raise ValueError("时间轴策略只能是 shift 或 overlay。")
    if int(sample_rate) <= 0:
        raise ValueError("时间轴采样率必须大于 0。")
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
    channels, cursor, starts, reports = (
        max(clip.shape[1] for clip in normalized),
        0,
        [],
        [],
    )
    for line, clip in zip(lines, normalized):
        start_ms = _optional_timeline_ms(line.start_ms, line.index, "start_ms")
        end_ms = _optional_timeline_ms(line.end_ms, line.index, "end_ms")
        if (start_ms is None) != (end_ms is None) or (
            start_ms is not None and end_ms <= start_ms
        ):
            raise ValueError(
                f"第 {line.index} 条台词的 start_ms/end_ms 必须同时填写，且结束时间晚于开始时间。"
            )
        requested = int(start_ms or 0) * sample_rate // 1000
        start = max(requested, cursor) if policy == "shift" else requested
        end = start + clip.shape[-1]
        slot_end = None if end_ms is None else end_ms * sample_rate // 1000
        reports.append(
            TimelinePlacement(
                line.index,
                round(requested * 1000 / sample_rate),
                round(start * 1000 / sample_rate),
                round(end * 1000 / sample_rate),
                round(max(0, cursor - requested) * 1000 / sample_rate),
                0
                if slot_end is None
                else round(max(0, end - slot_end) * 1000 / sample_rate),
            )
        )
        starts.append(start)
        cursor = max(cursor, end + max(0, int(gap_ms)) * sample_rate // 1000)
    output_samples = max(
        start + clip.shape[-1] for start, clip in zip(starts, normalized)
    )
    allocation_bytes = (
        (channels + 1)
        * output_samples
        * torch.tensor([], dtype=torch.float32).element_size()
    )
    if allocation_bytes > MAX_TIMELINE_ALLOCATION_BYTES:
        estimated_gib = allocation_bytes / (1024**3)
        raise ValueError(
            f"时间轴需要约 {estimated_gib:.2f} GiB 临时内存，超过 1 GiB 安全上限；"
            "请缩短开始时间或拆分工程。"
        )
    output = torch.zeros((1, channels, output_samples))
    active = torch.zeros(output.shape[-1])
    for start, clip in zip(starts, normalized):
        if clip.shape[1] != channels:
            clip = clip[:, :1].repeat(1, channels, 1)
        output[..., start : start + clip.shape[-1]] += clip
        active[start : start + clip.shape[-1]] += 1
    if policy == "overlay":
        output /= active.clamp_min(1)[None, None, :]
    return output.clamp(-1, 1), reports


__all__ = [
    "DialogueLine",
    "EMOTION_OVERRIDE_MODES",
    "MAX_TIMELINE_MS",
    "TimelinePlacement",
    "compose_timeline",
    "fit_duration_factor",
    "format_emotion_override",
    "missing_roles",
    "parse_batch_script",
    "parse_emotion_override",
    "parse_srt",
]
