"""Language-aware segmentation and deterministic pause planning.

The planner is intentionally lightweight: it loads only the official tiktoken
vocabulary from the selected model directory, never the neural model weights.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


LANGUAGE_AUTO_LIMITS = {
    "ZH": 120,
    "EN": 60,
    "JA": 100,
    "ES": 60,
    "AR": 80,
}

PAUSE_PRESETS = {
    "off": (0, 0, 0),
    "natural": (0, 260, 500),
    "narration": (120, 360, 700),
    "dialogue": (80, 250, 450),
}

_EXPLICIT_PAUSE = re.compile(
    r"(?:<|\[)\s*pause\s*(?:=|:)\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*(?:>|\])",
    re.IGNORECASE,
)
_PROTECTED = re.compile(r"<[^>\n]+>")
_BOUNDARY = re.compile(
    r"(?:<|\[)\s*pause\s*(?:=|:)\s*\d+(?:\.\d+)?\s*(?:ms|s)?\s*(?:>|\])"
    r"|<[^>\n]+>|\r?\n(?:[ \t]*\r?\n)*|[。！？!?；;：:]|(?<!\d)\.(?!\d)|[，、]|(?<!\d),(?!\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    text: str
    pause_after_ms: int = 0
    pause_before_ms: int = 0


@dataclass(frozen=True, slots=True)
class SegmentPreview:
    index: int
    speech_block: int
    token_count: int
    text: str
    pause_after_ms: int
    pause_before_ms: int = 0


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    language: str
    segmentation_mode: str
    max_tokens: int
    pause_preset: str
    chunks: tuple[SpeechChunk, ...]
    segments: tuple[SegmentPreview, ...]

    @property
    def total_pause_ms(self) -> int:
        return sum(chunk.pause_before_ms + chunk.pause_after_ms for chunk in self.chunks)

    @property
    def max_segment_tokens(self) -> int:
        return max((segment.token_count for segment in self.segments), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "segmentation_mode": self.segmentation_mode,
            "max_tokens": self.max_tokens,
            "pause_preset": self.pause_preset,
            "speech_blocks": len(self.chunks),
            "segment_count": len(self.segments),
            "total_pause_ms": self.total_pause_ms,
            "gpt_accel_risk": gpt_accel_risk(self),
            "chunks": [asdict(item) for item in self.chunks],
            "segments": [asdict(item) for item in self.segments],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def auto_segment_limit(language: str) -> int:
    normalized = str(language).strip().upper()
    if normalized not in LANGUAGE_AUTO_LIMITS:
        raise ValueError(f"不支持的语言：{normalized}")
    return LANGUAGE_AUTO_LIMITS[normalized]


def effective_segment_limit(language: str, mode: str, custom_limit: int) -> int:
    normalized_mode = str(mode or "auto").strip().lower()
    if normalized_mode not in {"auto", "custom"}:
        raise ValueError("分段模式只能是 auto 或 custom。")
    if normalized_mode == "auto":
        return auto_segment_limit(language)
    value = int(custom_limit)
    if not 20 <= value <= 300:
        raise ValueError("自定义每段文本 Token 必须在 20–300。")
    return value


def resolve_pause_values(
    preset: str,
    comma_ms: int,
    sentence_ms: int,
    paragraph_ms: int,
) -> tuple[int, int, int]:
    normalized = str(preset or "off").strip().lower()
    if normalized == "custom":
        values = (int(comma_ms), int(sentence_ms), int(paragraph_ms))
    elif normalized in PAUSE_PRESETS:
        values = PAUSE_PRESETS[normalized]
    else:
        raise ValueError("停顿预设只能是 off、natural、narration、dialogue 或 custom。")
    if any(value < 0 or value > 5000 for value in values):
        raise ValueError("标点停顿必须在 0–5000 毫秒。")
    return values


def _explicit_pause_ms(match: re.Match[str]) -> int:
    value = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    millis = round(value if unit == "ms" else value * 1000)
    if not 0 <= millis <= 30_000:
        raise ValueError("显式停顿必须在 0–30 秒；例如 <pause=0.5> 或 <pause=500ms>。")
    return millis


def split_speech_chunks(
    text: str,
    *,
    pause_preset: str = "off",
    comma_pause_ms: int = 0,
    sentence_pause_ms: int = 0,
    paragraph_pause_ms: int = 0,
) -> tuple[SpeechChunk, ...]:
    """Split text only where an external silence must be inserted.

    Pronunciation annotations such as ``<银行|YIN2 HANG2>`` remain atomic.
    Explicit pause tags are always honoured, even when the preset is ``off``.
    """

    source = str(text or "").strip()
    if not source:
        raise ValueError("待合成文本不能为空。")
    comma_ms, sentence_ms, paragraph_ms = resolve_pause_values(
        pause_preset, comma_pause_ms, sentence_pause_ms, paragraph_pause_ms
    )
    chunks: list[SpeechChunk] = []
    buffer: list[str] = []
    pending_leading_pause = 0

    def flush(pause_ms: int = 0) -> None:
        nonlocal pending_leading_pause
        content = "".join(buffer).strip()
        buffer.clear()
        if content:
            chunks.append(SpeechChunk(content, int(pause_ms), pending_leading_pause))
            pending_leading_pause = 0
        elif chunks and pause_ms:
            previous = chunks[-1]
            chunks[-1] = SpeechChunk(
                previous.text,
                max(previous.pause_after_ms, int(pause_ms)),
                previous.pause_before_ms,
            )
        elif pause_ms:
            pending_leading_pause = max(pending_leading_pause, int(pause_ms))

    position = 0
    for match in _BOUNDARY.finditer(source):
        buffer.append(source[position : match.start()])
        token = match.group(0)
        explicit = _EXPLICIT_PAUSE.fullmatch(token)
        if explicit:
            flush(_explicit_pause_ms(explicit))
        elif _PROTECTED.fullmatch(token):
            buffer.append(token)
        elif "\n" in token or "\r" in token:
            flush(paragraph_ms)
        elif token in "，、,":
            buffer.append(token)
            if comma_ms:
                flush(comma_ms)
        else:
            buffer.append(token)
            if sentence_ms:
                flush(sentence_ms)
        position = match.end()
    buffer.append(source[position:])
    flush(0)
    if not chunks:
        raise ValueError("文本只包含停顿标记，没有可合成内容。")
    # A trailing silence is intentional only when explicitly present in source.
    return tuple(chunks)


@lru_cache(maxsize=8)
def _load_tokenizer(model_dir: str):
    from indextts.utils.tokenizer import get_tokenizer

    return get_tokenizer(multilingual=True, model_dir=str(Path(model_dir)))


def _token_len(tokenizer: Any, value: str) -> int:
    return len(tokenizer.encode(value, allowed_special="all"))


def _split_atomic_pieces(text: str) -> list[tuple[str, bool]]:
    pieces: list[tuple[str, bool]] = []
    position = 0
    for match in _PROTECTED.finditer(text):
        if match.start() > position:
            pieces.append((text[position : match.start()], False))
        pieces.append((match.group(0), True))
        position = match.end()
    if position < len(text):
        pieces.append((text[position:], False))
    return pieces


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    language: str,
    token_counter: Callable[[str], int],
) -> list[str]:
    """Mirror the official punctuation-first splitter with an injected counter."""

    prefix = f"<|{str(language).lower()}|> "
    budget = max(1, int(max_tokens) - token_counter(prefix))
    if token_counter(text) <= budget:
        return [text]
    pieces: list[str] = []
    for piece, atomic in _split_atomic_pieces(text):
        if atomic:
            pieces.append(piece)
            continue
        for part in re.split(r"(?<=[，。！？、；：,\.!?;:\n])", piece):
            if not part:
                continue
            if token_counter(part) <= budget:
                pieces.append(part)
                continue
            current = ""
            for char in part:
                if current and token_counter(current + char) > budget:
                    pieces.append(current)
                    current = char
                else:
                    current += char
            if current:
                pieces.append(current)
    segments: list[str] = []
    current = ""
    for piece in pieces:
        if current and token_counter(current + piece) > budget:
            segments.append(current)
            current = piece
        else:
            current += piece
    if current:
        segments.append(current)
    return segments or [text]


def build_generation_plan(
    text: str,
    language: str,
    model_dir: str | Path,
    *,
    segmentation_mode: str = "auto",
    max_text_tokens_per_segment: int = 120,
    pause_preset: str = "off",
    comma_pause_ms: int = 0,
    sentence_pause_ms: int = 0,
    paragraph_pause_ms: int = 0,
) -> GenerationPlan:
    normalized_language = str(language).strip().upper()
    limit = effective_segment_limit(
        normalized_language, segmentation_mode, max_text_tokens_per_segment
    )
    chunks = split_speech_chunks(
        text,
        pause_preset=pause_preset,
        comma_pause_ms=comma_pause_ms,
        sentence_pause_ms=sentence_pause_ms,
        paragraph_pause_ms=paragraph_pause_ms,
    )
    tokenizer = _load_tokenizer(str(Path(model_dir).resolve()))
    prefix = f"<|{normalized_language.lower()}|> "
    counter = lambda value: _token_len(tokenizer, value)
    previews: list[SegmentPreview] = []
    for block_index, chunk in enumerate(chunks, 1):
        parts = split_text_by_tokens(chunk.text, limit, normalized_language, counter)
        for part_index, part in enumerate(parts):
            previews.append(
                SegmentPreview(
                    len(previews) + 1,
                    block_index,
                    counter(prefix + part),
                    part,
                    chunk.pause_after_ms if part_index == len(parts) - 1 else 0,
                    chunk.pause_before_ms if part_index == 0 else 0,
                )
            )
    return GenerationPlan(
        normalized_language,
        str(segmentation_mode).lower(),
        limit,
        str(pause_preset).lower(),
        chunks,
        tuple(previews),
    )


def gpt_accel_risk(plan: GenerationPlan) -> bool:
    """Guard the still-unmerged upstream synthetic-prompt KV-cache edge case."""

    return len(plan.chunks) > 1 or plan.max_segment_tokens >= 180


__all__ = [
    "GenerationPlan",
    "LANGUAGE_AUTO_LIMITS",
    "PAUSE_PRESETS",
    "SegmentPreview",
    "SpeechChunk",
    "auto_segment_limit",
    "build_generation_plan",
    "effective_segment_limit",
    "gpt_accel_risk",
    "resolve_pause_values",
    "split_speech_chunks",
    "split_text_by_tokens",
]
