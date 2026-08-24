"""Optional local Whisper ASR and transcript accuracy scoring."""

from __future__ import annotations

import importlib.util
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any

import torch
import torchaudio


ASR_MODELS = ("tiny", "base", "small", "medium", "turbo")
ASR_LANGUAGE_CODES = {"AUTO": None, "ZH": "zh", "EN": "en", "JA": "ja", "ES": "es", "AR": "ar"}
_CACHE: dict[tuple[str, str, str], Any] = {}
_LOCK = threading.RLock()
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def asr_available() -> bool:
    return importlib.util.find_spec("whisper") is not None


def normalize_review_text(text: str, language: str = "AUTO") -> str:
    return _NON_WORD.sub("", unicodedata.normalize("NFKC", str(text or "")).casefold()).replace("_", "")


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def review_transcript(expected_text: str, recognized_text: str, language: str = "AUTO", threshold: float = 0.82) -> dict[str, Any]:
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("ASR 通过阈值必须在 0 到 1 之间。")
    expected, recognized = normalize_review_text(expected_text, language), normalize_review_text(recognized_text, language)
    distance = edit_distance(expected, recognized)
    similarity = max(0.0, 1.0 - distance / max(len(expected), len(recognized), 1))
    return {
        "expected_text": str(expected_text), "recognized_text": str(recognized_text),
        "normalized_expected": expected, "normalized_recognized": recognized,
        "edit_distance": distance, "cer": round(distance / max(len(expected), 1), 6),
        "similarity": round(similarity, 6), "threshold": threshold,
        "passed": bool(expected and recognized and similarity >= threshold), "language": str(language).upper(),
    }


def _resolve_device(device: str) -> str:
    value = str(device or "auto").lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("所选 ASR CUDA 不可用，请改用 auto 或 cpu。")
    if value not in {"cpu", "cuda"}:
        raise ValueError("ASR 设备只能是 auto、cpu 或 cuda。")
    return value


def _load_model(model_name: str, device: str, download_root: str | Path | None):
    model_name = str(model_name).lower()
    if model_name not in ASR_MODELS:
        raise ValueError("ASR 模型只能是：" + "、".join(ASR_MODELS))
    if not asr_available():
        raise RuntimeError("缺少可选 ASR 依赖 openai-whisper；请在 ComfyUI Python 环境中手动安装。")
    resolved = _resolve_device(device)
    root = "" if download_root is None else str(Path(download_root).resolve())
    key = (model_name, resolved, root)
    with _LOCK:
        if key not in _CACHE:
            import whisper
            if root:
                Path(root).mkdir(parents=True, exist_ok=True)
            _CACHE[key] = whisper.load_model(model_name, device=resolved, download_root=root or None)
        return _CACHE[key], resolved


def transcribe_waveform(waveform, sample_rate: int, *, language: str = "AUTO", model_name: str = "base", device: str = "auto", download_root: str | Path | None = None) -> dict[str, Any]:
    language = str(language or "AUTO").upper()
    if language not in ASR_LANGUAGE_CODES:
        raise ValueError("ASR 语言只能是 AUTO、ZH、EN、JA、ES 或 AR。")
    audio = torch.as_tensor(waveform).detach().float().cpu()
    while audio.ndim > 2:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if int(sample_rate) != 16000:
        audio = torchaudio.functional.resample(audio, int(sample_rate), 16000)
    model, resolved = _load_model(model_name, device, download_root)
    result = model.transcribe(audio.squeeze(0).clamp(-1, 1).numpy(), language=ASR_LANGUAGE_CODES[language], task="transcribe", fp16=resolved == "cuda", verbose=False, condition_on_previous_text=False, temperature=0.0)
    return {
        "text": str(result.get("text") or "").strip(),
        "detected_language": str(result.get("language") or ASR_LANGUAGE_CODES[language] or ""),
        "requested_language": language, "model": str(model_name).lower(), "device": resolved,
        "segments": len(result.get("segments") or ()),
    }


__all__ = ["ASR_MODELS", "asr_available", "edit_distance", "normalize_review_text", "review_transcript", "transcribe_waveform"]
