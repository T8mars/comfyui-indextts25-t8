from __future__ import annotations

import gc
import time
from typing import Any

import torch

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
    inference_error: Exception | None = None
    started_at = time.perf_counter()
    peak_memory_mb = None
    try:
        with entry.lock:
            if handle.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            entry.model.gr_progress = _progress_callback()
            accel_engine = getattr(getattr(entry.model, "gpt", None), "accel_engine", None)
            accel_compatible = bool(
                sampling.do_sample
                and sampling.top_p == 1.0
                and sampling.top_k == 0
                and sampling.num_beams == 1
                and sampling.repetition_penalty == 1.0
                and sampling.length_penalty == 0.0
            )
            temporarily_disabled_accel = accel_engine is not None and not accel_compatible
            try:
                if temporarily_disabled_accel:
                    entry.model.gpt.accel_engine = None
                    notes.append("当前采样参数与 GPT 加速语义不兼容，本次自动使用普通 GPT 路径")
                if use_emo_text:
                    entry.model.ensure_qwen_emotion()
                    if handle.low_vram:
                        try:
                            emotion_dict = entry.model.qwen_emo.inference(emo_text)
                            emo_vector = list(emotion_dict.values())
                            use_emo_text = False
                        finally:
                            entry.model.qwen_emo = None
                            gc.collect()
                            if handle.device.startswith("cuda") and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        notes.append("低显存模式已在生成前释放 QwenEmotion")
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
                if temporarily_disabled_accel:
                    entry.model.gpt.accel_engine = accel_engine
                entry.model.gr_progress = None
                if handle.device.startswith("cuda") and torch.cuda.is_available():
                    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception as exc:
        inference_error = exc
    finally:
        MODEL_CACHE.done(
            handle,
            entry,
            release=handle.release_after_run
            or bool(inference_error is not None and handle.acceleration_effective != "off"),
        )

    if inference_error is not None:
        if handle.acceleration_effective != "off":
            handle.use_cuda_kernel = False
            handle.use_torch_compile = False
            handle.use_accel = False
            handle.use_deepspeed = False
            handle.acceleration_effective = "off"
            handle.acceleration_note = (
                f"可选加速运行失败（{type(inference_error).__name__}: {inference_error}），"
                "已自动重载普通模式"
            )
            try:
                return run_inference(
                    handle,
                    speaker_audio,
                    text,
                    language,
                    duration_factor,
                    seed,
                    emotion,
                    sampling,
                )
            except Exception as fallback_error:
                raise fallback_error from inference_error
        raise inference_error

    if result is None:
        raise RuntimeError("IndexTTS 2.5 未生成音频。请缩短文本或提高 max_mel_tokens 后重试。")
    audio = indextts_result_to_audio(result)
    elapsed = time.perf_counter() - started_at
    duration = audio["waveform"].shape[-1] / audio["sample_rate"]
    rtf = elapsed / duration if duration > 0 else 0.0
    status = (
        f"IndexTTS 2.5 | {language.upper()} | {duration:.2f}s | "
        f"耗时={elapsed:.2f}s | RTF={rtf:.3f} | "
        f"duration_factor={float(duration_factor):.2f} | seed={int(seed)} | "
        f"accel={handle.acceleration_effective}"
    )
    if peak_memory_mb is not None:
        status += f" | CUDA峰值={peak_memory_mb:.0f}MiB"
    if handle.acceleration_note:
        notes.append(handle.acceleration_note)
    if notes:
        status += " | " + "；".join(dict.fromkeys(notes))
    return audio, status
