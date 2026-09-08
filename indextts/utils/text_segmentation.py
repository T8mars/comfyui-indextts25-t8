"""Language-aware token budgets shared by previews and model inference."""

from __future__ import annotations

import re


CJK_SEGMENT_LANGUAGES = frozenset({"zh", "zhen", "ja", "ko", "yue"})
NON_CJK_BUDGET_SCALE = 3.1 / 4.3
REFERENCE_MAX_TEXT_TOKENS = 120
_LANGUAGE_PREFIX = re.compile(r"<\|([^|]+)\|>")


def language_from_prefix(lang_prefix: str) -> str:
    """Extract the normalized language code from an IndexTTS prefix."""

    match = _LANGUAGE_PREFIX.match(str(lang_prefix or ""))
    return match.group(1).strip().lower() if match else ""


def language_aware_token_budget(
    max_tokens: int,
    prefix_tokens: int,
    *,
    language: str,
    capacity: int | None = None,
) -> int:
    """Return a safe content-token budget without double-scaling small limits.

    The upstream non-CJK proposal derives a 3.1/4.3 density ratio from English
    long-text failures.  Desktop and ComfyUI auto modes already use conservative
    per-language limits (for example EN/ES=60), so multiplying those limits again
    would over-segment them.  Instead, the ratio defines a cap for large non-CJK
    requests while smaller explicit limits remain unchanged.
    """

    requested = int(max_tokens)
    if capacity is not None:
        requested = min(requested, max(1, int(capacity) - 2))
    prefix = max(0, int(prefix_tokens))
    budget = max(1, requested - prefix)
    normalized = str(language or "").strip().lower()
    if not normalized or normalized in CJK_SEGMENT_LANGUAGES:
        return budget

    reference = REFERENCE_MAX_TEXT_TOKENS
    if capacity is not None:
        reference = min(reference, max(1, int(capacity) - 2))
    safe_cap = max(1, int(max(1, reference - prefix) * NON_CJK_BUDGET_SCALE))
    return min(budget, safe_cap)


__all__ = [
    "CJK_SEGMENT_LANGUAGES",
    "NON_CJK_BUDGET_SCALE",
    "REFERENCE_MAX_TEXT_TOKENS",
    "language_aware_token_budget",
    "language_from_prefix",
]
