"""Optional Whisper backends plus language-aware transcript review."""

from __future__ import annotations

import difflib
import importlib.util
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import torch
import torchaudio

ASR_MODELS = ("tiny", "base", "small", "medium", "turbo")
ASR_BACKENDS = ("auto", "openai_whisper", "faster_whisper")
ASR_LANGUAGE_CODES = {"AUTO": None, "ZH": "zh", "EN": "en", "JA": "ja", "ES": "es", "AR": "ar"}
_CACHE: dict[tuple[str, str, str, str], Any] = {}
_LOCK = threading.RLock()
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_WORD_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff]")
_CHINESE_NUMBER = re.compile(r"[负負]?[零〇一二两兩三四五六七八九十百千万萬亿億]+(?:点[零〇一二两兩三四五六七八九]+)?")
_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_LARGE_UNITS = {"万": 10_000, "萬": 10_000, "亿": 100_000_000, "億": 100_000_000}
_FALLBACK_T2S = str.maketrans({"臺": "台", "灣": "湾", "語": "语", "詞": "词", "體": "体", "實": "实", "長": "长", "點": "点", "數": "数", "據": "据", "後": "后", "發": "发", "聲": "声", "識": "识", "別": "别", "與": "与", "為": "为", "門": "门", "開": "开", "關": "关", "歡": "欢", "來": "来", "這": "这", "個": "个", "條": "条", "裡": "里", "裏": "里", "測": "测", "試": "试", "銀": "银", "慶": "庆", "時": "时", "間": "间", "萬": "万", "億": "亿"})
_OPENCC = None
_OPENCC_CHECKED = False


def _backend_installed(backend: str) -> bool:
    return importlib.util.find_spec("whisper" if backend == "openai_whisper" else "faster_whisper") is not None


def resolve_asr_backend(backend: str = "auto") -> str:
    value = str(backend or "auto").lower()
    if value not in ASR_BACKENDS:
        raise ValueError("ASR 后端只能是 auto、openai_whisper 或 faster_whisper。")
    if value == "auto":
        for candidate in ("faster_whisper", "openai_whisper"):
            if _backend_installed(candidate):
                return candidate
        raise RuntimeError("未安装 ASR 后端；请安装 openai-whisper 或 faster-whisper。")
    if not _backend_installed(value):
        raise RuntimeError(f"未安装所选 ASR 后端 {'openai-whisper' if value == 'openai_whisper' else 'faster-whisper'}。")
    return value


def asr_available(backend: str = "auto") -> bool:
    try:
        resolve_asr_backend(backend)
        return True
    except (RuntimeError, ValueError):
        return False


def _simplify_chinese(text: str) -> str:
    global _OPENCC, _OPENCC_CHECKED
    if not _OPENCC_CHECKED:
        _OPENCC_CHECKED = True
        if importlib.util.find_spec("opencc") is not None:
            from opencc import OpenCC
            _OPENCC = OpenCC("t2s")
    return _OPENCC.convert(text) if _OPENCC is not None else text.translate(_FALLBACK_T2S)


def _chinese_integer(value: str) -> int:
    if not any(char in _SMALL_UNITS or char in _LARGE_UNITS for char in value):
        return int("".join(str(_DIGITS[char]) for char in value))
    total = section = number = 0
    for char in value:
        if char in _DIGITS:
            number = _DIGITS[char]
        elif char in _SMALL_UNITS:
            section += (number or 1) * _SMALL_UNITS[char]
            number = 0
        elif char in _LARGE_UNITS:
            section += number
            total += section * _LARGE_UNITS[char]
            section = number = 0
    return total + section + number


def _canonicalize_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        negative = raw.startswith(("负", "負"))
        raw = raw[1:] if negative else raw
        if "点" in raw:
            integer, fraction = raw.split("点", 1)
            number = f"{_chinese_integer(integer)}.{''.join(str(_DIGITS[c]) for c in fraction)}"
        else:
            number = str(_chinese_integer(raw))
        return ("-" if negative else "") + number
    return _CHINESE_NUMBER.sub(replace, text)


def _normalized_value(text: str) -> str:
    return _canonicalize_numbers(_simplify_chinese(unicodedata.normalize("NFKC", str(text or "")).casefold()))


def normalize_review_text(text: str, language: str = "AUTO") -> str:
    del language
    return _NON_WORD.sub("", _normalized_value(text)).replace("_", "")


def _word_tokens(text: str) -> list[str]:
    return _WORD_TOKEN.findall(_normalized_value(text).replace("_", " "))


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, 1):
        current = [left_index]
        for right_index, right_item in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_item != right_item)))
        previous = current
    return previous[-1]


def _difference_details(expected: Sequence[str], recognized: Sequence[str], separator: str) -> list[dict[str, str]]:
    details = []
    for operation, left_start, left_end, right_start, right_end in difflib.SequenceMatcher(a=list(expected), b=list(recognized), autojunk=False).get_opcodes():
        if operation != "equal":
            details.append({"operation": operation, "expected": separator.join(expected[left_start:left_end]), "recognized": separator.join(recognized[right_start:right_end])})
    return details


def review_transcript(expected_text: str, recognized_text: str, language: str = "AUTO", threshold: float = 0.82) -> dict[str, Any]:
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("ASR 通过阈值必须在 0 到 1 之间。")
    expected, recognized = normalize_review_text(expected_text, language), normalize_review_text(recognized_text, language)
    char_distance = edit_distance(expected, recognized)
    cer = char_distance / max(len(expected), 1)
    expected_words, recognized_words = _word_tokens(expected_text), _word_tokens(recognized_text)
    word_distance = edit_distance(expected_words, recognized_words)
    wer = word_distance / max(len(expected_words), 1)
    language = str(language or "AUTO").upper()
    metric = "cer" if language in {"ZH", "JA"} or (language == "AUTO" and _CJK.search(expected + recognized)) else "wer"
    metric_error = cer if metric == "cer" else wer
    metric_expected, metric_recognized = (list(expected), list(recognized)) if metric == "cer" else (expected_words, recognized_words)
    similarity = max(0.0, 1.0 - metric_error)
    return {"expected_text": str(expected_text), "recognized_text": str(recognized_text), "normalized_expected": expected, "normalized_recognized": recognized, "edit_distance": char_distance, "cer": round(cer, 6), "word_edit_distance": word_distance, "wer": round(wer, 6) if metric == "wer" else None, "metric": metric, "metric_error_rate": round(metric_error, 6), "similarity": round(similarity, 6), "threshold": threshold, "passed": bool(expected and recognized and similarity >= threshold), "language": language, "differences": _difference_details(metric_expected, metric_recognized, "" if metric == "cer" else " "), "normalization": ["NFKC", "casefold", "traditional_to_simplified", "number_canonicalization", "punctuation_ignored"]}


def resolve_asr_device(device: str = "auto") -> str:
    value = str(device or "auto").lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("所选 ASR CUDA 不可用，请改用 auto 或 cpu。")
    if value not in {"cpu", "cuda"}:
        raise ValueError("ASR 设备只能是 auto、cpu 或 cuda。")
    return value


def load_asr_model(model_name: str = "base", device: str = "auto", download_root: str | Path | None = None, backend: str = "auto"):
    model_name = str(model_name).lower()
    if model_name not in ASR_MODELS:
        raise ValueError("ASR 模型只能是：" + "、".join(ASR_MODELS))
    resolved_backend, resolved_device = resolve_asr_backend(backend), resolve_asr_device(device)
    root = "" if download_root is None else str(Path(download_root).resolve())
    key = (resolved_backend, model_name, resolved_device, root)
    with _LOCK:
        if key not in _CACHE:
            if root:
                Path(root).mkdir(parents=True, exist_ok=True)
            if resolved_backend == "openai_whisper":
                import whisper
                model = whisper.load_model(model_name, device=resolved_device, download_root=root or None)
            else:
                from faster_whisper import WhisperModel
                model = WhisperModel(model_name, device=resolved_device, compute_type="float16" if resolved_device == "cuda" else "int8", download_root=root or None)
            _CACHE[key] = model
        return _CACHE[key], resolved_device


def _openai_words(segments) -> list[dict[str, Any]]:
    words = []
    for segment_index, segment in enumerate(segments or ()):
        for word in segment.get("words") or ():
            words.append({"word": str(word.get("word") or "").strip(), "start": round(float(word.get("start") or 0), 3), "end": round(float(word.get("end") or 0), 3), "probability": round(float(word.get("probability") or 0), 6), "segment": segment_index})
    return words


def transcribe_waveform(waveform, sample_rate: int, *, language: str = "AUTO", model_name: str = "base", device: str = "auto", download_root: str | Path | None = None, backend: str = "auto") -> dict[str, Any]:
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
    samples = audio.squeeze(0).clamp(-1, 1).numpy()
    resolved_backend = resolve_asr_backend(backend)
    model, resolved_device = load_asr_model(model_name, device, download_root, resolved_backend)
    if resolved_backend == "openai_whisper":
        result = model.transcribe(samples, language=ASR_LANGUAGE_CODES[language], task="transcribe", fp16=resolved_device == "cuda", verbose=False, condition_on_previous_text=False, temperature=0.0, word_timestamps=True)
        segments = result.get("segments") or ()
        text, detected_language, words = str(result.get("text") or "").strip(), str(result.get("language") or ASR_LANGUAGE_CODES[language] or ""), _openai_words(segments)
    else:
        segment_iter, info = model.transcribe(samples, language=ASR_LANGUAGE_CODES[language], task="transcribe", beam_size=5, temperature=0.0, condition_on_previous_text=False, word_timestamps=True)
        segments = list(segment_iter)
        text, detected_language, words = "".join(str(segment.text) for segment in segments).strip(), str(getattr(info, "language", None) or ASR_LANGUAGE_CODES[language] or ""), []
        for segment_index, segment in enumerate(segments):
            for word in getattr(segment, "words", None) or ():
                words.append({"word": str(word.word).strip(), "start": round(float(word.start), 3), "end": round(float(word.end), 3), "probability": round(float(getattr(word, "probability", 0)), 6), "segment": segment_index})
    return {"text": text, "detected_language": detected_language, "requested_language": language, "model": model_name, "device": resolved_device, "backend": resolved_backend, "segments": len(segments), "word_timestamps": words}


__all__ = ["ASR_BACKENDS", "ASR_MODELS", "asr_available", "edit_distance", "normalize_review_text", "resolve_asr_backend", "review_transcript", "transcribe_waveform"]
