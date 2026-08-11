from __future__ import annotations

from typing import Any

from .audio_adapter import indextts_result_to_audio
from .model_cache import MODEL_CACHE
from .reference_cache import comfy_audio_to_reference_wav
from .seed_scope import scoped_seed
from .types import DEFAULT_EMOTION, DEFAULT_SAMPLING, EmotionConfig, ModelHandle, SamplingConfig


def _progress_callback():
    try:
        import comfy.model_management
        import comfy.utils

        progress = comfy.utils.ProgressBar(100)

        def update(value: float, desc: str = "") -> None:
            comfy.model_management.throw_exception_if_processing_interrupted()
            progress.update_absolute(max(0, min(100, round(float(value) * 100))))

        return update
    except Exception:
        return lambda value, desc="": None


def run_inference(
    handle: ModelHandle,
    speaker_audio: dict[str, Any],
    text: str,
    language: str,
    duration_factor: float,
    seed: int,
    emotion: EmotionConfig | None = None,
    sampling: SamplingConfig | None = None,
) -> tuple[dict[str, Any], str]:
    text = str(text).strip()
    if not text:
        raise ValueError("待合成文本不能为空。")
    if language.upper() not in {"ZH", "EN", "JA", "ES", "AR"}:
        raise ValueError(f"不支持的语言：{language}")
    if not 0.5 <= float(duration_factor) <= 2.0:
        raise ValueError("语速/时长系数必须在 0.5 到 2.0 之间。")

    emotion = emotion or DEFAULT_EMOTION
    sampling = sampling or DEFAULT_SAMPLING
    speaker_path, speaker_notes = comfy_audio_to_reference_wav(speaker_audio, kind="speaker")
    notes = list(speaker_notes) + list(emotion.notes)

    emo_audio_prompt = None
    emo_vector = None
    use_emo_text = False
    emo_text = None
    if emotion.mode == "reference_audio":
        if emotion.reference_audio is None:
            raise ValueError("情感参考音频模式缺少 emotion_audio。")
        emotion_path, emotion_notes = comfy_audio_to_reference_wav(emotion.reference_audio, kind="emotion")
        emo_audio_prompt = str(emotion_path)
        notes.extend(emotion_notes)
    elif emotion.mode == "vector":
        if emotion.vector is None or len(emotion.vector) != 8:
            raise ValueError("八维情感向量模式需要 8 个数值。")
        emo_vector = list(emotion.vector)
    elif emotion.mode == "text":
        use_emo_text = True
        emo_text = (emotion.text or text).strip()
    elif emotion.mode != "speaker":
        raise ValueError(f"未知情感模式：{emotion.mode}")

    entry = MODEL_CACHE.acquire(handle)
    result = None
    try:
        with entry.lock:
            entry.model.gr_progress = _progress_callback()
            try:
                if use_emo_text:
                    entry.model.ensure_qwen_emotion()
                with scoped_seed(seed, handle.device):
                    result = entry.model.infer(
                        spk_audio_prompt=str(speaker_path),
                        text=text,
                        output_path=None,
                        lang=language.upper(),
                        emo_audio_prompt=emo_audio_prompt,
                        emo_alpha=float(emotion.strength),
                        emo_vector=emo_vector,
                        use_emo_text=use_emo_text,
                        emo_text=emo_text,
                        use_random=bool(emotion.use_random),
                        interval_silence=int(sampling.segment_silence_ms),
                        verbose=False,
                        max_text_tokens_per_segment=int(sampling.max_text_tokens_per_segment),
                        duration_factor=float(duration_factor),
                        text_normalization=bool(sampling.text_normalization),
                        **sampling.generation_kwargs(),
                    )
            finally:
                entry.model.gr_progress = None
    finally:
        MODEL_CACHE.done(handle, entry, release=handle.release_after_run)

    if result is None:
        raise RuntimeError("IndexTTS 2.5 未生成音频。请缩短文本或提高 max_mel_tokens 后重试。")
    audio = indextts_result_to_audio(result)
    duration = audio["waveform"].shape[-1] / audio["sample_rate"]
    status = (
        f"IndexTTS 2.5 | {language.upper()} | {duration:.2f}s | "
        f"duration_factor={float(duration_factor):.2f} | seed={int(seed)}"
    )
    if notes:
        status += " | " + "；".join(dict.fromkeys(notes))
    return audio, status
