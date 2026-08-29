from __future__ import annotations

import gc
import inspect
import json
import time
from typing import Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .audio_adapter import indextts_result_to_audio
from .audio_processing import concatenate_with_pauses
from .model_cache import MODEL_CACHE
from .reference_cache import comfy_audio_to_reference_wav
from .seed_scope import scoped_seed
from .text_planner import build_generation_plan, run_with_long_text_guard
from .types import DEFAULT_EMOTION, DEFAULT_SAMPLING, EmotionConfig, ModelHandle, SamplingConfig


class NativeTargetDurationUnsupported(RuntimeError):
    """Raised when a selected external runtime predates native duration control."""


class _ComfyInterruptStoppingCriteria(StoppingCriteria):
    """Check ComfyUI's stop flag between autoregressive token steps."""

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        throw_if_processing_interrupted()
        return False


def _native_chunk_durations(plan, target_duration_seconds: float | None) -> list[float | None]:
    if target_duration_seconds is None:
        return [None] * len(plan.chunks)
    duration = float(target_duration_seconds)
    pause_ms = sum(int(chunk.pause_after_ms) for chunk in plan.chunks)
    if plan.chunks:
        pause_ms += int(getattr(plan.chunks[0], "pause_before_ms", 0))
    pause_seconds = pause_ms / 1000.0
    speech_seconds = duration - pause_seconds
    if speech_seconds <= 0:
        raise ValueError(
            f"目标时长 {duration:.3f} 秒不足以容纳已配置的 {pause_seconds:.3f} 秒停顿。"
        )
    weights = [max(1, len(str(chunk.text))) for chunk in plan.chunks]
    total_weight = sum(weights)
    return [speech_seconds * weight / total_weight for weight in weights]


def _supports_native_target_duration(model) -> bool:
    parameters = inspect.signature(model.infer).parameters.values()
    return any(parameter.name == "target_duration" for parameter in parameters)


def _progress_callback():
    try:
        import comfy.model_management
        import comfy.utils

        progress = comfy.utils.ProgressBar(100)

        def update(value: float, desc: str = "") -> None:
            throw_if_processing_interrupted()
            progress.update_absolute(max(0, min(100, round(float(value) * 100))))

        return update
    except Exception:
        return lambda value, desc="": None


def throw_if_processing_interrupted() -> None:
    """Honor ComfyUI's stop request without requiring ComfyUI in unit tests."""

    try:
        import comfy.model_management
    except (ImportError, ModuleNotFoundError):
        return
    comfy.model_management.throw_exception_if_processing_interrupted()


def _result_duration_seconds(result: Any) -> float:
    audio = indextts_result_to_audio(result)
    return float(audio["waveform"].shape[-1]) / float(audio["sample_rate"])


def run_inference(
    handle: ModelHandle,
    speaker_audio: dict[str, Any],
    text: str,
    language: str,
    duration_factor: float,
    seed: int,
    emotion: EmotionConfig | None = None,
    sampling: SamplingConfig | None = None,
    target_duration_seconds: float | None = None,
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
    plan = build_generation_plan(
        text,
        language,
        handle.model_dir,
        segmentation_mode=sampling.segmentation_mode,
        max_text_tokens_per_segment=sampling.max_text_tokens_per_segment,
        pause_preset=sampling.pause_preset,
        comma_pause_ms=sampling.comma_pause_ms,
        sentence_pause_ms=sampling.sentence_pause_ms,
        paragraph_pause_ms=sampling.paragraph_pause_ms,
    )
    native_chunk_durations = _native_chunk_durations(plan, target_duration_seconds)
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
    results: list[Any] = []
    long_text_guard_reports: list[dict[str, Any]] = []
    inference_error: Exception | None = None
    started_at = time.perf_counter()
    peak_memory_mb = None
    try:
        with entry.lock:
            if handle.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            entry.model.gr_progress = _progress_callback()
            if target_duration_seconds is not None and not _supports_native_target_duration(entry.model):
                raise NativeTargetDurationUnsupported(
                    "当前 IndexTTS 推理核心不支持原生 target_duration，请更新内置核心。"
                )
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
                    if not accel_compatible:
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
                for block_index, chunk in enumerate(plan.chunks):
                    throw_if_processing_interrupted()
                    with scoped_seed(int(seed) + block_index, handle.device):
                        def generate_with_limit(limit: int):
                            infer_kwargs = dict(
                                spk_audio_prompt=str(speaker_path),
                                text=chunk.text,
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
                                max_text_tokens_per_segment=int(limit),
                                duration_factor=float(duration_factor),
                                text_normalization=bool(sampling.text_normalization),
                                interrupt_callback=throw_if_processing_interrupted,
                                stopping_criteria=StoppingCriteriaList(
                                    [_ComfyInterruptStoppingCriteria()]
                                ),
                                **sampling.generation_kwargs(),
                            )
                            if native_chunk_durations[block_index] is not None:
                                infer_kwargs["target_duration"] = native_chunk_durations[block_index]
                            return entry.model.infer(**infer_kwargs)

                        block_token_count = sum(
                            int(getattr(segment, "token_count", 0))
                            for segment in plan.segments
                            if int(getattr(segment, "speech_block", 1)) == block_index + 1
                        )
                        result, guard_report = run_with_long_text_guard(
                            generate_with_limit,
                            _result_duration_seconds,
                            text=chunk.text,
                            language=language,
                            token_count=block_token_count,
                            max_tokens=plan.max_tokens,
                            duration_factor=duration_factor,
                            check_duration=native_chunk_durations[block_index] is None,
                        )
                        throw_if_processing_interrupted()
                        results.append(result)
                        guard_report["speech_block"] = block_index + 1
                        long_text_guard_reports.append(guard_report)
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
        if isinstance(inference_error, NativeTargetDurationUnsupported):
            raise inference_error
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
                    target_duration_seconds,
                )
            except Exception as fallback_error:
                raise fallback_error from inference_error
        raise inference_error

    if not results or any(result is None for result in results):
        raise RuntimeError("IndexTTS 2.5 未生成音频。请缩短文本或提高 max_mel_tokens 后重试。")
    block_audios = [indextts_result_to_audio(result) for result in results]
    audio = concatenate_with_pauses(
        block_audios,
        [chunk.pause_after_ms for chunk in plan.chunks],
        getattr(plan.chunks[0], "pause_before_ms", 0),
    )
    elapsed = time.perf_counter() - started_at
    duration = audio["waveform"].shape[-1] / audio["sample_rate"]
    rtf = elapsed / duration if duration > 0 else 0.0
    status = (
        f"IndexTTS 2.5 | {language.upper()} | {duration:.2f}s | "
        f"耗时={elapsed:.2f}s | RTF={rtf:.3f} | "
        f"duration_factor={float(duration_factor):.2f} | seed={int(seed)} | "
        f"segments={len(plan.segments)}@{plan.max_tokens}token | "
        f"pause={plan.total_pause_ms}ms | accel={handle.acceleration_effective}"
    )
    status += (
        f" | CFM={sampling.diffusion_steps}steps/cfg{sampling.inference_cfg_rate:.2f}"
        f"/temp{sampling.cfm_temperature:.2f}"
    )
    if target_duration_seconds is not None:
        status += f" | native_target={float(target_duration_seconds):.3f}s"
    if peak_memory_mb is not None:
        status += f" | CUDA峰值={peak_memory_mb:.0f}MiB"
    latin_guards = [item for item in long_text_guard_reports if item.get("enabled")]
    if latin_guards:
        status += " | long_text_guard=" + json.dumps(
            latin_guards, ensure_ascii=False, separators=(",", ":")
        )
    if handle.acceleration_note:
        notes.append(handle.acceleration_note)
    if notes:
        status += " | " + "；".join(dict.fromkeys(notes))
    return audio, status
