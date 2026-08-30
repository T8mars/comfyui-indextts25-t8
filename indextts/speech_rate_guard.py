"""Conservative cross-segment speech-rate anomaly detection.

The guard compares only sufficiently long neighbouring segments.  It is meant
to catch the rare long-text failure where the tail suddenly becomes several
times slower, not to flatten intentional emotion or prosody.
"""

from __future__ import annotations

import math
import re
from statistics import median
from typing import Any, Iterable, Mapping


SLOW_RATE_RATIO = 0.45
MIN_BASELINE_SEGMENTS = 2
_LATIN_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_EMBEDDED_LATIN_WORD = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ]+)*"
)
_CJK_OR_KANA = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]"
)
_PROTECTED = re.compile(r"<[^>\n]+>|\[(?:pause|停顿)[^\]\n]*\]", re.IGNORECASE)


def speech_unit_count(text: str, language: str) -> int:
    """Count stable, language-aware units without counting punctuation/tags."""

    source = _PROTECTED.sub(" ", str(text or ""))
    language = str(language or "").strip().upper()
    if language in {"ZH", "JA"}:
        return len(_CJK_OR_KANA.findall(source)) + len(
            _EMBEDDED_LATIN_WORD.findall(source)
        )
    return len(_LATIN_WORD.findall(source))


def assess_segment_speech_rates(
    segments: Iterable[Mapping[str, Any]],
    *,
    ratio_threshold: float = SLOW_RATE_RATIO,
    min_baseline_segments: int = MIN_BASELINE_SEGMENTS,
) -> list[dict[str, Any]]:
    """Return sequential rate reports and flag only strong slow-down outliers."""

    threshold = max(0.2, min(0.8, float(ratio_threshold)))
    baseline_needed = max(2, int(min_baseline_segments))
    stable_rates: list[float] = []
    reports: list[dict[str, Any]] = []
    for position, raw in enumerate(segments):
        text = str(raw.get("text") or "")
        language = str(raw.get("language") or "").strip().upper()
        units = speech_unit_count(text, language)
        try:
            seconds = float(raw.get("duration_seconds") or 0.0)
        except (TypeError, ValueError):
            seconds = 0.0
        valid_duration = math.isfinite(seconds) and seconds > 0
        rate = units / seconds if valid_duration and units > 0 else 0.0
        minimum_units = 6 if language in {"EN", "ES", "AR"} else 10
        eligible = valid_duration and seconds >= 2.0 and units >= minimum_units
        baseline = (
            float(median(stable_rates[-5:]))
            if eligible and len(stable_rates) >= baseline_needed
            else 0.0
        )
        ratio = rate / baseline if baseline > 0 else 0.0
        suspect = bool(
            baseline >= 0.8
            and ratio > 0
            and ratio <= threshold
            and baseline - rate >= 0.5
        )
        report = {
            "position": position,
            "index": raw.get("index", position + 1),
            "speech_block": raw.get("speech_block"),
            "language": language,
            "text": text[:160],
            "speech_units": units,
            "duration_seconds": round(seconds, 4) if valid_duration else 0.0,
            "units_per_second": round(rate, 4),
            "baseline_units_per_second": round(baseline, 4),
            "rate_ratio": round(ratio, 4),
            "eligible": eligible,
            "suspect": suspect,
            "reason": "cross_segment_slowdown" if suspect else "",
        }
        reports.append(report)
        if eligible and not suspect:
            stable_rates.append(rate)
    return reports


def retry_candidate_improves_rate(
    original_rate: float,
    retry_rate: float,
    baseline_rate: float,
) -> bool:
    """Accept a retry only when it is materially closer to the stable baseline."""

    original = float(original_rate)
    retry = float(retry_rate)
    baseline = float(baseline_rate)
    if not all(math.isfinite(value) and value > 0 for value in (original, retry, baseline)):
        return False
    original_ratio = original / baseline
    retry_ratio = retry / baseline
    if retry_ratio > 1.8 or retry <= original * 1.2:
        return False
    return abs(1.0 - retry_ratio) + 0.08 < abs(1.0 - original_ratio)


__all__ = [
    "MIN_BASELINE_SEGMENTS",
    "SLOW_RATE_RATIO",
    "assess_segment_speech_rates",
    "retry_candidate_improves_rate",
    "speech_unit_count",
]
