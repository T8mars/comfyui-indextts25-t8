from __future__ import annotations

import gc
import json
import logging
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch
import torchaudio
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .project_meta import PROJECT_VERSION
from .runtime.inference_adapter import (
    NativeTargetDurationUnsupported,
    _progress_callback,
    run_inference,
)
from .runtime.audio_processing import (
    POSTPROCESS_PRESETS,
    apply_duration_policy,
    audio_duration_ms,
    postprocess_audio,
)
from .runtime.audio_quality import (
    analyze_reference_audio,
    prepare_reference_audio,
    render_waveform_image,
)
from .runtime.candidates import (
    combined_candidate_score,
    select_best_candidate,
    technical_audio_review,
)
from .runtime.context_emotion import suggest_context_emotions
from .runtime.benchmark import summarize_measurements
from .runtime.audiocpp_backend import probe as probe_audiocpp
from .runtime.audiocpp_backend import run as run_audiocpp
from .runtime.model_cache import MODEL_CACHE
from .runtime.reference_cache import comfy_audio_to_reference_wav
from .runtime.acceleration import (
    MODES,
    probe_acceleration,
    recommend_runtime_config,
    resolve_acceleration,
)
from .runtime.dialogue import (
    compose_timeline,
    fit_duration_factor,
    missing_roles,
    parse_batch_script,
    parse_srt,
)
from .runtime.pronunciation import (
    PronunciationValidationError,
    format_pronunciation_report,
    parse_dictionary_text,
    process_pronunciation_text,
)
from .runtime.text_planner import build_generation_plan
from .runtime.speech_review import (
    ASR_BACKENDS,
    ASR_MODELS,
    asr_available,
    asr_cache_status,
    clear_asr_cache,
    review_transcript,
    transcribe_waveform,
)
from .runtime.timeline import (
    apply_timeline_edits,
    render_timeline_image,
    rewrite_srt,
    timeline_json,
)
from .runtime.update_check import check_updates
from .runtime.types import (
    DialogueScript,
    EmotionConfig,
    ModelHandle,
    RoleLibrary,
    SamplingConfig,
    VoiceProfile,
)
from .services.model_store import (
    MISSING_MODEL_OPTION,
    MODEL_FOLDER_NAME,
    MODEL_REPOSITORY_URL,
    configured_model_roots,
    load_manifest,
    model_fingerprint,
    model_options,
    register_model_paths,
    resolve_model,
    validate_model_dir,
)
from .services.downloader import ensure_model_bundle


LOGGER = logging.getLogger("comfyui-indextts25-T8")
CATEGORY = "T8star-Aix/Audio/IndexTTS 2.5"
ModelType = io.Custom("T8_INDEXTTS25_MODEL")
EmotionType = io.Custom("T8_INDEXTTS25_EMOTION")
SamplingType = io.Custom("T8_INDEXTTS25_SAMPLING")
VoiceType = io.Custom("T8_INDEXTTS25_VOICE")
RoleLibraryType = io.Custom("T8_INDEXTTS25_ROLE_LIBRARY")
DialogueScriptType = io.Custom("T8_INDEXTTS25_DIALOGUE_SCRIPT")


def _asr_download_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "TTS" / "Whisper"
    except Exception:
        return Path(tempfile.gettempdir()) / "t8_indextts25_whisper"


def _reference_condition_cache_root() -> Path:
    try:
        import folder_paths

        get_user_directory = getattr(folder_paths, "get_user_directory", None)
        if callable(get_user_directory):
            return Path(get_user_directory()) / "indextts25_t8" / "reference_condition_cache"
        return Path(folder_paths.models_dir) / "TTS" / ".indextts25_t8_cache"
    except Exception:
        return Path(tempfile.gettempdir()) / "t8_indextts25_reference_condition_cache"


def _device_options() -> list[str]:
    values = ["auto"]
    if torch.cuda.is_available():
        values.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    values.append("cpu")
    return values


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("当前 ComfyUI 的 PyTorch 未检测到 CUDA。")
        return requested
    try:
        import comfy.model_management

        selected = str(comfy.model_management.get_torch_device())
    except Exception:
        selected = "cuda:0" if torch.cuda.is_available() else "cpu"
    if selected == "cuda":
        selected = f"cuda:{torch.cuda.current_device()}"
    if not (
        selected.startswith("cuda")
        or selected.startswith("cpu")
        or selected.startswith("xpu")
        or selected.startswith("mps")
    ):
        raise RuntimeError(f"IndexTTS 2.5 暂不支持 ComfyUI 当前设备：{selected}")
    return selected


def _native_bf16(device: str) -> bool:
    if device.startswith("xpu"):
        return True
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return False
    index = (
        int(device.split(":", 1)[1])
        if ":" in device
        else torch.cuda.current_device()
    )
    try:
        if hasattr(torch.cuda, "device"):
            with torch.cuda.device(index):
                return bool(
                    torch.cuda.is_bf16_supported(including_emulation=False)
                )
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        try:
            return bool(torch.cuda.is_bf16_supported(index))
        except TypeError:
            return bool(torch.cuda.is_bf16_supported())


def _precision_flags(precision: str, device: str) -> tuple[bool, bool, str]:
    requested = str(precision or "auto").strip().lower()
    if requested not in {"auto", "bfloat16", "float16", "float32"}:
        raise RuntimeError(f"未知精度：{precision}")
    accelerator = device.startswith("cuda") or device.startswith("xpu")
    if requested in {"bfloat16", "float16"} and not accelerator:
        raise RuntimeError(f"{requested} 仅建议在 CUDA/XPU 设备上使用。")
    if requested == "float32" or not accelerator:
        return False, False, "float32"
    if requested == "bfloat16":
        return True, False, "bfloat16"
    if requested == "float16":
        return False, True, "float16"
    if _native_bf16(device):
        return True, False, "bfloat16"
    return False, True, "float16"


def _use_bf16(precision: str, device: str) -> bool:
    """Compatibility helper retained for existing integrations and tests."""

    return _precision_flags(precision, device)[0]


def _resolve_reference_device(mode: str, device: str, low_vram: bool) -> str:
    normalized = str(mode or "auto").strip().lower()
    if normalized not in {"auto", "same", "cpu"}:
        raise RuntimeError(f"未知参考编码器设备：{mode}")
    if normalized == "cpu" or (
        normalized == "auto" and low_vram and device.startswith("cuda")
    ):
        return "cpu"
    return device


def _is_low_vram(device: str, threshold_gb: float = 10.0) -> bool:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return False
    index = (
        int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
    )
    total_gb = torch.cuda.get_device_properties(index).total_memory / (1024**3)
    return total_gb < threshold_gb


def _normalize_model_loader_values(
    recycle_after_runs,
    verify_hashes,
    custom_model_path,
) -> tuple[int, bool, str, str]:
    """Restore v0.10 UI values after recycle_after_runs was inserted in v0.11."""
    legacy_flag = isinstance(recycle_after_runs, (bool, int, float)) and float(
        recycle_after_runs
    ) in {0.0, 1.0}
    if (
        legacy_flag
        and isinstance(verify_hashes, str)
        and not str(custom_model_path or "").strip()
    ):
        return (
            0,
            bool(float(recycle_after_runs)),
            verify_hashes,
            "已自动还原 v0.10 工作流中错位的模型校验与自定义路径。",
        )
    try:
        recycle = max(0, min(1000, int(float(recycle_after_runs or 0))))
    except (TypeError, ValueError, OverflowError):
        recycle = 0
    return recycle, bool(verify_hashes), str(custom_model_path or ""), ""


class T8IndexTTS25ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = model_options()
        return io.Schema(
            node_id="T8_IndexTTS25_ModelLoader",
            display_name="IndexTTS 2.5 模型加载器 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["IndexTTS 2.5", "T8star-Aix", "TTS model loader"],
            description=(
                "发现并校验 ComfyUI/models/TTS 下的正式 IndexTTS 2.5 模型；"
                "可由用户明确授权后下载或修复完整模型，权重在首次生成时按需载入。"
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    display_name="IndexTTS 2.5 模型",
                    options=options,
                    default=options[0],
                    tooltip="标准位置：ComfyUI/models/TTS/IndexTTS-2.5。安装模型后需刷新/重启 ComfyUI。",
                ),
                io.Combo.Input(
                    "device",
                    display_name="推理设备",
                    options=_device_options(),
                    default="auto",
                ),
                io.Combo.Input(
                    "precision",
                    display_name="精度",
                    options=["auto", "bfloat16", "float16", "float32"],
                    default="auto",
                    tooltip="auto 优先原生 bfloat16；旧显卡自动使用 float16，CPU 使用 float32。",
                ),
                io.Combo.Input(
                    "acceleration_mode",
                    display_name="可选加速模式",
                    options=list(MODES),
                    default="off",
                    tooltip=(
                        "默认关闭。缺少可选依赖会自动回退；DeepSpeed 不属于基础依赖，"
                        "不会被自动安装或自动启用。"
                    ),
                ),
                io.Boolean.Input(
                    "use_cuda_kernel",
                    display_name="旧工作流：BigVGAN CUDA 融合核",
                    default=False,
                    advanced=True,
                    tooltip="仅为兼容旧工作流；新工作流请使用上方可选加速模式。",
                ),
                io.Boolean.Input(
                    "release_after_run",
                    display_name="生成后释放本模型",
                    default=False,
                    advanced=True,
                    tooltip="适合显存紧张环境；会降低连续生成速度，不会清空其他 ComfyUI 模型。",
                ),
                io.Int.Input(
                    "recycle_after_runs",
                    display_name="连续生成多少次后重载模型（0=关闭）",
                    default=0,
                    min=0,
                    max=1000,
                    step=1,
                    advanced=True,
                    tooltip="长批量任务的显存稳定保护；达到次数后安全释放并在下一句自动重载。",
                ),
                io.Boolean.Input(
                    "verify_hashes",
                    display_name="完整 SHA-256 校验",
                    default=False,
                    advanced=True,
                    tooltip="首次校验约需读取 7.7 GiB 文件；平时仅做文件大小校验即可。",
                ),
                io.String.Input(
                    "custom_model_path",
                    display_name="自定义模型绝对路径",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="留空时使用上方模型列表；仅用于已有的完整 IndexTTS 2.5 目录。",
                ),
                io.Combo.Input(
                    "reference_device",
                    display_name="参考编码器设备",
                    options=["auto", "same", "cpu"],
                    default="auto",
                    advanced=True,
                    tooltip="auto：低于 10GB 显存时把 Wav2Vec/CAMPPlus 放到 CPU；same：与主模型相同；cpu：始终节省显存。",
                ),
                io.Boolean.Input(
                    "reuse_spk_cond_for_emo",
                    display_name="快速默认情感",
                    default=False,
                    advanced=True,
                    tooltip="未连接独立情感时复用音色条件，减少一次参考编码；可能轻微改变音色或情感。",
                ),
                io.Boolean.Input(
                    "persistent_reference_cache",
                    display_name="持久化参考条件缓存",
                    default=True,
                    advanced=True,
                    tooltip="按音频内容和模型版本缓存已提取的音色/情感张量；使用 safetensors，模型重载后仍可复用。",
                ),
                io.Boolean.Input(
                    "download_missing",
                    display_name="缺失时自动下载/修复完整模型",
                    default=False,
                    tooltip=(
                        "默认关闭。启用后会从 t8star/IndexTTS-2.5-Comfy 下载主模型、"
                        "bpe.model 与运行所需辅助模型，完整目录约 7.7 GiB。"
                    ),
                ),
                io.Boolean.Input(
                    "accept_model_license",
                    display_name="我已阅读并接受模型许可证",
                    default=False,
                    tooltip=(
                        "仅自动下载时需要勾选。许可证、免责声明与完整模型说明："
                        f"{MODEL_REPOSITORY_URL}"
                    ),
                ),
            ],
            outputs=[
                ModelType.Output("model", display_name="IndexTTS 2.5 模型"),
                io.String.Output("model_info", display_name="模型信息"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        model_name: str,
        device: str,
        precision: str,
        acceleration_mode: str,
        use_cuda_kernel: bool,
        release_after_run: bool,
        recycle_after_runs: int = 0,
        verify_hashes: bool = False,
        custom_model_path: str = "",
        reference_device: str = "auto",
        reuse_spk_cond_for_emo: bool = False,
        persistent_reference_cache: bool = True,
        download_missing: bool = False,
        accept_model_license: bool = False,
    ) -> str:
        recycle_after_runs, verify_hashes, custom_model_path, _ = (
            _normalize_model_loader_values(
                recycle_after_runs,
                verify_hashes,
                custom_model_path,
            )
        )
        try:
            path = resolve_model(model_name, custom_model_path)
            return model_fingerprint(path)
        except Exception as exc:
            return (
                f"missing:{model_name}:{custom_model_path}:"
                f"download={bool(download_missing)}:license={bool(accept_model_license)}:{exc}"
            )

    @classmethod
    def validate_inputs(
        cls,
        model_name: str,
        custom_model_path: str = "",
        recycle_after_runs: int = 0,
        verify_hashes: bool = False,
        download_missing: bool = False,
        accept_model_license: bool = False,
        **kwargs,
    ) -> bool | str:
        _, _, custom_model_path, _ = _normalize_model_loader_values(
            recycle_after_runs,
            verify_hashes,
            custom_model_path,
        )
        if download_missing and not accept_model_license:
            return "启用自动下载前，请勾选“我已阅读并接受模型许可证”。"
        if (
            model_name == MISSING_MODEL_OPTION
            and not custom_model_path.strip()
            and not download_missing
        ):
            return (
                "未找到 IndexTTS 2.5 模型；可启用“缺失时自动下载/修复完整模型”"
                "并接受许可证，或先运行 scripts/download_models.py。"
            )
        return True

    @classmethod
    def execute(
        cls,
        model_name: str,
        device: str,
        precision: str,
        acceleration_mode: str,
        use_cuda_kernel: bool,
        release_after_run: bool,
        recycle_after_runs: int = 0,
        verify_hashes: bool = False,
        custom_model_path: str = "",
        reference_device: str = "auto",
        reuse_spk_cond_for_emo: bool = False,
        persistent_reference_cache: bool = True,
        download_missing: bool = False,
        accept_model_license: bool = False,
    ) -> io.NodeOutput:
        recycle_after_runs, verify_hashes, custom_model_path, compatibility_note = (
            _normalize_model_loader_values(
                recycle_after_runs,
                verify_hashes,
                custom_model_path,
            )
        )
        try:
            model_dir = resolve_model(model_name, custom_model_path)
        except FileNotFoundError:
            if not download_missing:
                raise
            model_dir = (
                Path(custom_model_path.strip().strip('"')).expanduser().resolve()
                if custom_model_path.strip()
                else configured_model_roots()[0] / MODEL_FOLDER_NAME
            )

        downloaded_or_repaired = False
        if download_missing:
            if not accept_model_license:
                raise ValueError(
                    "启用自动下载前，请勾选“我已阅读并接受模型许可证”。"
                )
            initial_report = validate_model_dir(model_dir, verify_hashes=False)
            downloaded_or_repaired = not initial_report.valid
            report = ensure_model_bundle(
                model_dir,
                "huggingface",
                accept_license=True,
                verify_hashes=True,
            )
        else:
            report = validate_model_dir(model_dir, verify_hashes=verify_hashes)
            report.require_valid()
        fingerprint = model_fingerprint(model_dir)
        resolved_device = _resolve_device(device)
        low_vram = _is_low_vram(resolved_device)
        use_bf16, use_fp16, precision_name = _precision_flags(
            precision, resolved_device
        )
        resolved_reference_device = _resolve_reference_device(
            reference_device, resolved_device, low_vram
        )
        requested_acceleration = (
            "bigvgan_cuda"
            if use_cuda_kernel and acceleration_mode in {"off", "auto_safe"}
            else acceleration_mode
        )
        acceleration = resolve_acceleration(requested_acceleration, resolved_device)
        manifest = load_manifest()
        handle = ModelHandle(
            model_dir=model_dir,
            device=resolved_device,
            use_bf16=use_bf16,
            use_fp16=use_fp16,
            reference_device=resolved_reference_device,
            reuse_spk_cond_for_emo=bool(reuse_spk_cond_for_emo),
            use_cuda_kernel=acceleration.use_cuda_kernel,
            use_torch_compile=acceleration.use_torch_compile,
            use_accel=acceleration.use_accel,
            use_deepspeed=acceleration.use_deepspeed,
            acceleration_requested=acceleration.requested,
            acceleration_effective=acceleration.effective,
            acceleration_note=acceleration.reason,
            release_after_run=bool(release_after_run),
            recycle_after_runs=recycle_after_runs,
            model_revision=str(manifest["modelRevision"]),
            model_fingerprint=fingerprint,
            low_vram=low_vram,
            reference_cache_dir=(
                _reference_condition_cache_root()
                if bool(persistent_reference_cache)
                else None
            ),
        )
        verification = "SHA-256 已校验" if report.hashes_verified else "文件大小已校验"
        info = (
            f"IndexTTS 2.5 | node={PROJECT_VERSION} | core={str(manifest['codeRevision'])[:8]} | "
            f"model={str(manifest['modelRevision'])[:8]} | {model_dir} | device={resolved_device} | "
            f"precision={precision_name} | reference={resolved_reference_device} | {verification} | "
            f"accel={acceleration.effective}（{acceleration.reason}）"
            + (f" | {compatibility_note}" if compatibility_note else "")
            + (" | 低显存自动适配" if low_vram else "")
            + (" | 快速默认情感" if reuse_spk_cond_for_emo else "")
            + (" | 持久参考缓存" if persistent_reference_cache else "")
            + (" | 完整模型已自动下载/修复" if downloaded_or_repaired else "")
        )
        return io.NodeOutput(handle, info)


def _strength_input() -> io.Float.Input:
    return io.Float.Input(
        "strength",
        display_name="情感强度",
        default=1.0,
        min=0.0,
        max=1.0,
        step=0.01,
        display_mode=io.NumberDisplay.slider,
    )


class T8IndexTTS25EmotionControl(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        vector_inputs = [
            io.Float.Input(
                name, display_name=label, default=0.0, min=0.0, max=1.0, step=0.01
            )
            for name, label in (
                ("happy", "高兴 happy"),
                ("angry", "愤怒 angry"),
                ("sad", "悲伤 sad"),
                ("afraid", "恐惧 afraid"),
                ("disgusted", "厌恶 disgusted"),
                ("melancholic", "低落 melancholic"),
                ("surprised", "惊讶 surprised"),
                ("calm", "自然 calm"),
            )
        ]
        vector_inputs.extend(
            [
                _strength_input(),
                io.Boolean.Input(
                    "use_random",
                    display_name="随机情感原型",
                    default=False,
                    tooltip="开启后由 seed 决定每种情感采用的原型；关闭时匹配音色参考。",
                ),
            ]
        )
        return io.Schema(
            node_id="T8_IndexTTS25_EmotionControl",
            display_name="IndexTTS 2.5 情感控制 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["IndexTTS emotion", "情感向量", "情感参考音频"],
            description="在跟随音色、情感参考音频、八维向量和文本描述四种官方控制方式之间切换。",
            inputs=[
                io.DynamicCombo.Input(
                    "mode",
                    display_name="情感模式",
                    options=[
                        io.DynamicCombo.Option("speaker", []),
                        io.DynamicCombo.Option(
                            "reference_audio",
                            [
                                io.Audio.Input(
                                    "emotion_audio", display_name="情感参考音频"
                                ),
                                _strength_input(),
                            ],
                        ),
                        io.DynamicCombo.Option("vector", vector_inputs),
                        io.DynamicCombo.Option(
                            "text",
                            [
                                io.String.Input(
                                    "emotion_text",
                                    display_name="情感描述",
                                    multiline=True,
                                    default="",
                                    placeholder="例如：克制但难掩喜悦，语气温柔。留空则分析待合成文本。",
                                ),
                                _strength_input(),
                            ],
                        ),
                    ],
                    tooltip="speaker 最省显存；text 会按需加载额外的 Qwen 情感模型。",
                ),
            ],
            outputs=[
                EmotionType.Output("emotion", display_name="情感控制"),
                io.String.Output("emotion_info", display_name="情感信息"),
            ],
        )

    @classmethod
    def execute(cls, mode: dict) -> io.NodeOutput:
        selected = str(mode["mode"])
        if selected == "speaker":
            config = EmotionConfig(mode="speaker")
            info = "情感跟随音色参考（不额外加载文本情感模型）"
        elif selected == "reference_audio":
            config = EmotionConfig(
                mode="reference_audio",
                reference_audio=mode["emotion_audio"],
                strength=float(mode.get("strength", 1.0)),
            )
            info = f"情感参考音频 | strength={config.strength:.2f}"
        elif selected == "vector":
            names = (
                "happy",
                "angry",
                "sad",
                "afraid",
                "disgusted",
                "melancholic",
                "surprised",
                "calm",
            )
            values = [max(0.0, min(1.0, float(mode.get(name, 0.0)))) for name in names]
            notes: list[str] = []
            total = sum(values)
            if total > 0.8:
                scale = 0.8 / total
                values = [value * scale for value in values]
                notes.append("向量总强度超过 0.8，已等比归一化")
            config = EmotionConfig(
                mode="vector",
                vector=tuple(values),
                strength=float(mode.get("strength", 1.0)),
                use_random=bool(mode.get("use_random", False)),
                notes=tuple(notes),
            )
            populated = ", ".join(
                f"{name}={value:.2f}" for name, value in zip(names, values) if value > 0
            )
            info = "八维情感向量 | " + (populated or "全部为 0（保留基础情感）")
            if notes:
                info += " | " + "；".join(notes)
        elif selected == "text":
            emotion_text = str(mode.get("emotion_text", "")).strip()
            config = EmotionConfig(
                mode="text",
                text=emotion_text or None,
                strength=float(mode.get("strength", 1.0)),
            )
            info = "文本情感分析 | " + (emotion_text or "使用待合成文本")
        else:
            raise ValueError(f"未知情感模式：{selected}")
        return io.NodeOutput(config, info)


class T8IndexTTS25SamplingConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_SamplingConfig",
            display_name="IndexTTS 2.5 采样设置 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["IndexTTS sampling", "TTS sampling config"],
            description="集中配置确定性、采样、语言感知长文本分段、标点/显式停顿和文本归一化参数。",
            inputs=[
                io.Boolean.Input(
                    "do_sample",
                    display_name="启用随机采样",
                    default=False,
                    tooltip="关闭时结果更稳定；开启后 temperature/top_p/top_k 生效。",
                ),
                io.Float.Input(
                    "temperature",
                    display_name="temperature",
                    default=0.8,
                    min=0.1,
                    max=2.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Float.Input(
                    "top_p",
                    display_name="top_p",
                    default=0.8,
                    min=0.05,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "top_k",
                    display_name="top_k",
                    default=30,
                    min=0,
                    max=200,
                    step=1,
                    advanced=True,
                ),
                io.Int.Input(
                    "num_beams",
                    display_name="num_beams",
                    default=3,
                    min=1,
                    max=10,
                    step=1,
                    advanced=True,
                ),
                io.Float.Input(
                    "repetition_penalty",
                    display_name="repetition_penalty",
                    default=10.0,
                    min=0.1,
                    max=20.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Float.Input(
                    "length_penalty",
                    display_name="length_penalty",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_mel_tokens",
                    display_name="最大语音 token",
                    default=1500,
                    min=256,
                    max=4096,
                    step=16,
                    advanced=True,
                ),
                io.Int.Input(
                    "diffusion_steps",
                    display_name="CFM 扩散步数",
                    default=25,
                    min=5,
                    max=100,
                    step=1,
                    advanced=True,
                    tooltip="官方默认 25；更高通常更稳定但更慢，旁白可尝试 40–50。",
                ),
                io.Float.Input(
                    "inference_cfg_rate",
                    display_name="CFM 引导强度",
                    default=0.7,
                    min=0.0,
                    max=1.5,
                    step=0.05,
                    advanced=True,
                    tooltip="提高后更贴近参考音色/音高；过高可能过度平滑。",
                ),
                io.Float.Input(
                    "cfm_temperature",
                    display_name="CFM 温度",
                    default=1.0,
                    min=0.1,
                    max=1.5,
                    step=0.05,
                    advanced=True,
                    tooltip="降低可减少抖动；稳定旁白可尝试 0.8。",
                ),
                io.Combo.Input(
                    "segmentation_mode",
                    display_name="长文本分段模式",
                    options=["auto", "custom"],
                    default="auto",
                    tooltip="auto：EN/ES=60、AR=80、JA=100、ZH=120 Token；custom 使用下方数值。",
                ),
                io.Int.Input(
                    "max_text_tokens_per_segment",
                    display_name="每段最大文本 token",
                    default=120,
                    min=20,
                    max=300,
                    step=5,
                ),
                io.Int.Input(
                    "segment_silence_ms",
                    display_name="段间静音（毫秒）",
                    default=200,
                    min=0,
                    max=3000,
                    step=10,
                ),
                io.Combo.Input(
                    "pause_preset",
                    display_name="标点停顿预设",
                    options=["off", "natural", "narration", "dialogue", "custom"],
                    default="off",
                    tooltip="显式 <pause=0.5> 或 <pause=500ms> 在任意预设下都有效。",
                ),
                io.Int.Input(
                    "comma_pause_ms",
                    display_name="逗号停顿（毫秒）",
                    default=100,
                    min=0,
                    max=5000,
                    step=10,
                    advanced=True,
                ),
                io.Int.Input(
                    "sentence_pause_ms",
                    display_name="句末停顿（毫秒）",
                    default=300,
                    min=0,
                    max=5000,
                    step=10,
                    advanced=True,
                ),
                io.Int.Input(
                    "paragraph_pause_ms",
                    display_name="段落停顿（毫秒）",
                    default=600,
                    min=0,
                    max=5000,
                    step=10,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "text_normalization", display_name="文本归一化", default=True
                ),
            ],
            outputs=[
                SamplingType.Output("sampling", display_name="采样设置"),
                io.String.Output("sampling_info", display_name="采样信息"),
            ],
        )

    @classmethod
    def execute(
        cls,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: int,
        repetition_penalty: float,
        length_penalty: float,
        max_mel_tokens: int,
        diffusion_steps: int,
        inference_cfg_rate: float,
        cfm_temperature: float,
        segmentation_mode: str,
        max_text_tokens_per_segment: int,
        segment_silence_ms: int,
        pause_preset: str,
        comma_pause_ms: int,
        sentence_pause_ms: int,
        paragraph_pause_ms: int,
        text_normalization: bool,
    ) -> io.NodeOutput:
        config = SamplingConfig(
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
            length_penalty=float(length_penalty),
            max_mel_tokens=int(max_mel_tokens),
            diffusion_steps=int(diffusion_steps),
            inference_cfg_rate=float(inference_cfg_rate),
            cfm_temperature=float(cfm_temperature),
            segmentation_mode=str(segmentation_mode),
            max_text_tokens_per_segment=int(max_text_tokens_per_segment),
            segment_silence_ms=int(segment_silence_ms),
            pause_preset=str(pause_preset),
            comma_pause_ms=int(comma_pause_ms),
            sentence_pause_ms=int(sentence_pause_ms),
            paragraph_pause_ms=int(paragraph_pause_ms),
            text_normalization=bool(text_normalization),
        )
        mode = "随机采样" if config.do_sample else "确定性/束搜索"
        info = (
            f"{mode} | beams={config.num_beams} | max_mel={config.max_mel_tokens} | "
            f"segment={config.segmentation_mode}/{config.max_text_tokens_per_segment} | "
            f"pause={config.pause_preset} | internal_silence={config.segment_silence_ms}ms | "
            f"CFM={config.diffusion_steps}steps/cfg{config.inference_cfg_rate:.2f}/temp{config.cfm_temperature:.2f}"
        )
        return io.NodeOutput(config, info)


class T8IndexTTS25TextPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_TextPreview",
            display_name="IndexTTS 2.5 分段与停顿预览 · T8star-Aix",
            category=CATEGORY,
            search_aliases=[
                "text segment preview",
                "长文本分段",
                "pause preview",
                "停顿预览",
            ],
            description="仅加载官方轻量 Token 词表，预览模型输入前的 Token 分段、外部停顿和 GPT 加速风险；文本原样透传。",
            inputs=[
                ModelType.Input("model", display_name="IndexTTS 2.5 模型"),
                io.String.Input(
                    "text",
                    display_name="待预览文本",
                    multiline=True,
                    dynamic_prompts=True,
                ),
                io.Combo.Input(
                    "language",
                    display_name="语言",
                    options=["ZH", "EN", "JA", "ES", "AR"],
                    default="ZH",
                ),
                SamplingType.Input(
                    "sampling", display_name="采样/分段设置", optional=True
                ),
            ],
            outputs=[
                io.String.Output("text", display_name="原样文本"),
                io.String.Output("plan_json", display_name="分段与停顿预览 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(cls, text: str, **kwargs) -> bool | str:
        return True if str(text).strip() else "待预览文本不能为空。"

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        text: str,
        language: str,
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        config = sampling or SamplingConfig()
        plan = build_generation_plan(
            text,
            language,
            model.model_dir,
            segmentation_mode=config.segmentation_mode,
            max_text_tokens_per_segment=config.max_text_tokens_per_segment,
            pause_preset=config.pause_preset,
            comma_pause_ms=config.comma_pause_ms,
            sentence_pause_ms=config.sentence_pause_ms,
            paragraph_pause_ms=config.paragraph_pause_ms,
        )
        return io.NodeOutput(str(text), plan.to_json())


class T8IndexTTS25Pronunciation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_Pronunciation",
            display_name="IndexTTS 2.5 发音控制 · T8star-Aix",
            category=CATEGORY,
            search_aliases=[
                "IndexTTS pronunciation",
                "多音字",
                "拼音",
                "CMU phoneme",
                "日语假名",
            ],
            description=(
                "将持久词典规则转换成 IndexTTS 2.5 官方 <文字|读音> 标注；"
                "已有手工标注优先，输出可直接连接语音生成节点。"
            ),
            inputs=[
                io.String.Input(
                    "text",
                    display_name="原始文本",
                    multiline=True,
                    default="他在银行里工作，行长正在开会。",
                    dynamic_prompts=True,
                ),
                io.Combo.Input(
                    "language",
                    display_name="默认语言",
                    options=["ZH", "EN", "JA", "ES", "AR"],
                    default="ZH",
                    tooltip="每条词典记录也可在第三列单独指定 ZH、EN 或 JA。",
                ),
                io.String.Input(
                    "dictionary",
                    display_name="发音词典",
                    multiline=True,
                    default="银行|YIN2 HANG2|ZH\n行长|HANG2 ZHANG3|ZH",
                    placeholder="每行：文字|读音|语言，例如 银行|YIN2 HANG2|ZH",
                    tooltip=(
                        "按长词优先替换；词典内容保存在工作流中。也支持本项目导出的 YAML/JSON。"
                    ),
                ),
                io.Boolean.Input(
                    "strict",
                    display_name="严格校验",
                    default=True,
                    tooltip="无效拼音、CMU 音素、日语假名或损坏标注会阻止排队。",
                ),
            ],
            outputs=[
                io.String.Output("annotated_text", display_name="发音标注文本"),
                io.String.Output("pronunciation_report", display_name="替换/校验报告"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        text: str,
        language: str,
        dictionary: str,
        strict: bool,
        **kwargs,
    ) -> bool | str:
        if not str(text).strip():
            return "原始文本不能为空。"
        try:
            entries = parse_dictionary_text(dictionary, language)
            process_pronunciation_text(text, language, entries, strict=bool(strict))
        except PronunciationValidationError as exc:
            return str(exc)
        return True

    @classmethod
    def execute(
        cls,
        text: str,
        language: str,
        dictionary: str,
        strict: bool,
    ) -> io.NodeOutput:
        entries = parse_dictionary_text(dictionary, language)
        result = process_pronunciation_text(
            text, language, entries, strict=bool(strict)
        )
        return io.NodeOutput(result.text, format_pronunciation_report(result))


class T8IndexTTS25VoiceProfile(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_VoiceProfile",
            display_name="IndexTTS 2.5 角色音色 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["角色音色", "voice profile", "character voice"],
            description="把角色名称、参考音频、默认语言和可选情感保存为可连线的工作流内音色。",
            inputs=[
                io.String.Input("role_name", display_name="角色名称", default="旁白"),
                io.Audio.Input("speaker_audio", display_name="音色参考音频"),
                io.Combo.Input(
                    "language",
                    display_name="默认语言",
                    options=["ZH", "EN", "JA", "ES", "AR"],
                    default="ZH",
                ),
                EmotionType.Input(
                    "emotion",
                    display_name="该角色默认情感",
                    optional=True,
                    tooltip="连接情感控制后，只影响当前角色；不连接时跟随该角色的音色参考。",
                ),
            ],
            outputs=[
                VoiceType.Output("voice", display_name="角色音色"),
                io.String.Output("voice_info", display_name="音色信息"),
            ],
        )

    @classmethod
    def validate_inputs(cls, role_name: str, **kwargs) -> bool | str:
        return True if str(role_name).strip() else "角色名称不能为空。"

    @classmethod
    def execute(
        cls,
        role_name: str,
        speaker_audio: dict,
        language: str,
        emotion: EmotionConfig | None = None,
    ) -> io.NodeOutput:
        profile = VoiceProfile(
            str(role_name).strip(), speaker_audio, str(language).upper(), emotion
        )
        return io.NodeOutput(
            profile,
            f"角色={profile.name} | language={profile.language} | 情感={_emotion_mode_label(emotion)}",
        )


def _emotion_mode_label(emotion: EmotionConfig | None) -> str:
    if emotion is None or emotion.mode == "speaker":
        return "跟随音色"
    return {
        "reference_audio": "参考音频",
        "vector": "八维向量",
        "text": "文本描述",
    }.get(emotion.mode, emotion.mode)


def _resolve_line_emotion(line, profile: VoiceProfile) -> tuple[EmotionConfig | None, str]:
    """Return the line override, or the saved role emotion when it is inherited."""

    mode = str(getattr(line, "emotion_mode", "inherit") or "inherit")
    if mode == "inherit":
        return profile.emotion, "role_default"
    if mode == "speaker":
        return (
            EmotionConfig(mode="speaker", strength=float(line.emotion_strength)),
            "line_override",
        )
    if mode == "vector":
        vector = getattr(line, "emotion_vector", None)
        if vector is None or len(vector) != 8:
            raise ValueError(f"第 {line.index} 条台词的八维情感向量无效。")
        return (
            EmotionConfig(
                mode="vector",
                vector=tuple(float(item) for item in vector),
                strength=float(line.emotion_strength),
                use_random=bool(line.emotion_use_random),
            ),
            "line_override",
        )
    if mode == "text":
        return (
            EmotionConfig(
                mode="text",
                text=str(line.emotion_text or line.text),
                strength=float(line.emotion_strength),
            ),
            "line_override",
        )
    raise ValueError(f"第 {line.index} 条台词使用了未知情感模式：{mode}")


def _voice_profiles(value) -> list[VoiceProfile]:
    if isinstance(value, VoiceProfile):
        return [value]
    if isinstance(value, dict):
        result: list[VoiceProfile] = []
        for nested in value.values():
            result.extend(_voice_profiles(nested))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for nested in value:
            result.extend(_voice_profiles(nested))
        return result
    return []


def _build_role_library(value) -> tuple[RoleLibrary, str]:
    profiles = _voice_profiles(value)
    if not profiles:
        raise ValueError("角色音色/情感合并至少需要连接一个角色音色。")
    result: dict[str, VoiceProfile] = {}
    for profile in profiles:
        if profile.name in result:
            raise ValueError(f"角色名称重复：{profile.name}")
        result[profile.name] = profile
    summary = "、".join(
        f"{profile.name}（{_emotion_mode_label(profile.emotion)}）"
        for profile in result.values()
    )
    return RoleLibrary(result), "角色音色/情感：" + summary


class T8IndexTTS25RoleLibrary(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_RoleLibrary",
            display_name="IndexTTS 2.5 角色音色 / 情感合并 · T8star-Aix",
            category=CATEGORY,
            search_aliases=[
                "多角色",
                "role library",
                "音色库",
                "Merge Voice Emotions",
                "合并角色情感",
                "emotion merge",
            ],
            description="自动增长输入，可汇总 1–16 个角色各自的音色与情感；同名角色会被拒绝。",
            inputs=[
                io.Autogrow.Input(
                    "voices",
                    display_name="角色音色 / 情感",
                    template=io.Autogrow.TemplatePrefix(
                        VoiceType.Input("voice"), prefix="voice_", min=1, max=16
                    ),
                )
            ],
            outputs=[
                RoleLibraryType.Output("role_library", display_name="角色音色库"),
                io.String.Output("role_info", display_name="角色列表"),
            ],
        )

    @classmethod
    def execute(cls, voices: dict) -> io.NodeOutput:
        library, info = _build_role_library(voices)
        return io.NodeOutput(library, info)


class T8IndexTTS25MergeVoiceEmotions(io.ComfyNode):
    """Search-friendly equivalent of the role-library aggregator."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_MergeVoiceEmotions",
            display_name="IndexTTS 2.5 Merge Voice Emotions · T8star-Aix",
            category=CATEGORY,
            search_aliases=["合并角色情感", "角色 emotion 汇总", "多角色情感"],
            description=(
                "把 1–16 个已包含角色名、参考音色和可选情感的角色音色汇总为角色库；"
                "这是角色配置汇总，不会把多个八维情绪数值混成一个新情绪。"
            ),
            inputs=[
                io.Autogrow.Input(
                    "voices",
                    display_name="角色音色 / 情感",
                    template=io.Autogrow.TemplatePrefix(
                        VoiceType.Input("voice"), prefix="voice_", min=1, max=16
                    ),
                )
            ],
            outputs=[
                RoleLibraryType.Output(
                    "role_library", display_name="角色音色 / 情感库"
                ),
                io.String.Output("role_info", display_name="角色与情感列表"),
            ],
        )

    @classmethod
    def execute(cls, voices: dict) -> io.NodeOutput:
        library, info = _build_role_library(voices)
        return io.NodeOutput(library, info)


class T8IndexTTS25DialogueScript(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_DialogueScript",
            display_name="IndexTTS 2.5 批量台词 / SRT · T8star-Aix",
            category=CATEGORY,
            search_aliases=["SRT", "字幕配音", "批量台词", "dialogue script"],
            description=(
                "批量格式：角色|台词|语言|时长系数|逐句情感；最后一列支持 text:描述 或 "
                "vector:喜,怒,哀,惧,厌恶,低落,惊喜,平静。SRT 可写 "
                "[角色|emotion=text:生气、激动] 台词；留空继承角色默认情感。"
            ),
            inputs=[
                io.Combo.Input(
                    "script_type",
                    display_name="脚本格式",
                    options=["batch", "srt"],
                    default="batch",
                ),
                io.String.Input(
                    "script",
                    display_name="批量台词或 SRT",
                    multiline=True,
                    dynamic_prompts=False,
                    default=(
                        "旁白|先用平静语气介绍。|ZH|1.0|text:平静、从容\n"
                        "旁白|同一个角色突然非常生气！|ZH|1.0|vector:0,0.8,0,0,0,0,0,0\n"
                        "旁白|这一句恢复角色默认情感。|ZH|1.0"
                    ),
                    tooltip=(
                        "支持 角色|台词|语言|时长系数|逐句情感、JSON 数组或 SRT。"
                        "此输入已关闭 ComfyUI 动态提示词解析，JSON 的大括号不会被改写。"
                    ),
                ),
                io.String.Input(
                    "default_role", display_name="默认角色", default="旁白"
                ),
                io.Combo.Input(
                    "default_language",
                    display_name="默认语言",
                    options=["ZH", "EN", "JA", "ES", "AR"],
                    default="ZH",
                ),
            ],
            outputs=[
                DialogueScriptType.Output("dialogue_script", display_name="台词脚本"),
                io.String.Output("script_preview", display_name="解析预览 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        script_type: str,
        script: str,
        default_role: str,
        default_language: str,
        **kwargs,
    ) -> bool | str:
        try:
            (parse_srt if script_type == "srt" else parse_batch_script)(
                script, default_role, default_language
            )
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def execute(
        cls, script_type: str, script: str, default_role: str, default_language: str
    ) -> io.NodeOutput:
        lines = (parse_srt if script_type == "srt" else parse_batch_script)(
            script, default_role, default_language
        )
        payload = [line.to_dict() for line in lines]
        return io.NodeOutput(
            DialogueScript(lines, script_type),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )


class T8IndexTTS25DialogueEmotionSuggest(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_DialogueEmotionSuggest",
            display_name="IndexTTS 2.5 上下文逐句情感建议 · T8star-Aix",
            category=CATEGORY,
            search_aliases=[
                "context emotion",
                "上下文情感",
                "逐句情感分析",
                "SRT 情感建议",
            ],
            description=(
                "使用本地 QwenEmotion 结合目标台词的前后文，为每句建议八维情感向量和强度。"
                "节点只输出可编辑建议，不会生成音频；请先查看/修改 JSON，再连接时间轴编辑或生成节点。"
            ),
            inputs=[
                ModelType.Input("model", display_name="IndexTTS 2.5 模型"),
                DialogueScriptType.Input("dialogue_script", display_name="台词脚本"),
                io.Int.Input(
                    "context_window",
                    display_name="每侧上下文台词数",
                    default=2,
                    min=0,
                    max=5,
                    step=1,
                    tooltip="2 表示参考目标台词前 2 句和后 2 句；分析时会区分角色。",
                ),
                io.Boolean.Input(
                    "overwrite_existing",
                    display_name="覆盖已有逐句情感",
                    default=False,
                    tooltip="关闭时保留脚本里已有的 text:/vector: 人工设置。",
                ),
            ],
            outputs=[
                DialogueScriptType.Output(
                    "dialogue_script", display_name="带建议的台词脚本"
                ),
                io.String.Output(
                    "editable_suggestions_json", display_name="可编辑建议 JSON"
                ),
                io.String.Output("summary", display_name="分析摘要"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        dialogue_script: DialogueScript,
        context_window: int,
        overwrite_existing: bool,
    ) -> io.NodeOutput:
        if not isinstance(model, ModelHandle):
            raise TypeError("model 输入无效，请连接本项目的 IndexTTS 2.5 模型加载节点。")
        if not isinstance(dialogue_script, DialogueScript) or not dialogue_script.lines:
            raise ValueError("台词脚本为空，请先连接批量台词 / SRT 节点。")
        entry = MODEL_CACHE.acquire(model)
        released_temporary_qwen = False
        error: Exception | None = None
        try:
            with entry.lock:
                had_qwen = getattr(entry.model, "qwen_emo", None) is not None
                try:
                    entry.model.ensure_qwen_emotion()
                    if getattr(entry.model, "qwen_emo", None) is None:
                        raise RuntimeError("QwenEmotion 加载后仍不可用。")
                    progress = _progress_callback()

                    def update_progress(position, total, line):
                        if line is None:
                            progress(1.0, "上下文情感分析完成，等待用户确认")
                        else:
                            progress(
                                0.05 + 0.9 * position / max(1, total),
                                f"分析第 {getattr(line, 'index', position + 1)} 条上下文情感",
                            )

                    lines, report = suggest_context_emotions(
                        dialogue_script.lines,
                        entry.model.qwen_emo.inference,
                        context_window=int(context_window),
                        overwrite_existing=bool(overwrite_existing),
                        progress=update_progress,
                    )
                finally:
                    if (
                        model.low_vram
                        and not had_qwen
                        and getattr(entry.model, "qwen_emo", None) is not None
                    ):
                        entry.model.qwen_emo = None
                        gc.collect()
                        if model.device.startswith("cuda") and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        released_temporary_qwen = True
        except Exception as exc:
            error = exc
        finally:
            MODEL_CACHE.done(model, entry, release=model.release_after_run)
        if error is not None:
            raise RuntimeError(
                "上下文情感分析失败："
                f"{type(error).__name__}: {error}。可减少上下文句数或切换低显存/参考编码设备设置后重试。"
            ) from error
        report["temporary_qwen_released"] = released_temporary_qwen
        report["instruction"] = (
            "本节点没有生成音频。请检查 suggestions 与 lines，必要时复制 lines 到时间轴编辑节点修改；"
            "确认后再连接多角色 / SRT 生成节点。"
        )
        payload = {**report, "lines": [line.to_dict() for line in lines]}
        summary = (
            f"已分析 {report['classified_count']} 条，保留 {report['preserved_count']} 条人工设置；"
            "尚未合成音频，请先确认可编辑建议 JSON。"
        )
        return io.NodeOutput(
            DialogueScript(lines, dialogue_script.script_type),
            json.dumps(payload, ensure_ascii=False, indent=2),
            summary,
        )


class T8IndexTTS25TimelineEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_TimelineEditor",
            display_name="IndexTTS 2.5 可视化时间轴编辑 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["timeline", "时间轴", "字幕编辑", "SRT editor"],
            description=(
                "用 JSON 编辑每句角色、语言、开始/结束毫秒、语速和文本；输出仍是可连接的台词脚本，"
                "预览 JSON 可直接复制修改。"
            ),
            inputs=[
                DialogueScriptType.Input("dialogue_script", display_name="台词脚本"),
                io.String.Input(
                    "timeline_edits_json",
                    display_name="时间轴编辑 JSON（留空保持原样）",
                    multiline=True,
                    default="",
                    dynamic_prompts=False,
                    tooltip="先运行节点取得预览 JSON；复制 lines 数组后修改并填回。时间单位为毫秒。",
                ),
            ],
            outputs=[
                DialogueScriptType.Output(
                    "dialogue_script", display_name="编辑后的台词脚本"
                ),
                io.String.Output("timeline_preview", display_name="时间轴预览 JSON"),
                io.Image.Output("timeline_image", display_name="可视化时间轴"),
            ],
        )

    @classmethod
    def execute(
        cls, dialogue_script: DialogueScript, timeline_edits_json: str
    ) -> io.NodeOutput:
        raw = str(timeline_edits_json or "").strip()
        lines = (
            apply_timeline_edits(dialogue_script.lines, raw)
            if raw
            else list(dialogue_script.lines)
        )
        payload = {
            "unit": "milliseconds",
            "columns": [
                "index",
                "role",
                "language",
                "start_ms",
                "end_ms",
                "duration_factor",
                "text",
                "emotion_mode",
                "emotion_text",
                "emotion_vector",
                "emotion_strength",
                "emotion_use_random",
            ],
            "lines": [line.to_dict() for line in lines],
        }
        return io.NodeOutput(
            DialogueScript(lines, dialogue_script.script_type),
            json.dumps(payload, ensure_ascii=False, indent=2),
            render_timeline_image(lines),
        )


class T8IndexTTS25DialogueGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_DialogueGenerate",
            display_name="IndexTTS 2.5 多角色 / SRT 生成 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["SRT generate", "多角色配音", "batch dialogue"],
            description="按角色依次推理，再按顺延或原始时间轴合成为标准 AUDIO。",
            inputs=[
                ModelType.Input("model", display_name="IndexTTS 2.5 模型"),
                RoleLibraryType.Input("role_library", display_name="角色音色库"),
                DialogueScriptType.Input("dialogue_script", display_name="台词脚本"),
                SamplingType.Input("sampling", display_name="采样设置", optional=True),
                io.Int.Input(
                    "seed",
                    display_name="起始 seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Combo.Input(
                    "timeline_policy",
                    display_name="时间冲突策略",
                    options=["shift", "overlay"],
                    default="shift",
                    tooltip="shift 顺延避免重叠；overlay 保留 SRT 起点并安全混音。",
                ),
                io.Boolean.Input(
                    "fit_srt_slots",
                    display_name="适配字幕槽位",
                    default=False,
                    tooltip="仅 SRT 有效；native 单次控制，其他模式保留兼容回退。",
                ),
                io.Combo.Input(
                    "slot_duration_mode",
                    display_name="字幕槽位收尾模式",
                    options=["native", "natural", "pad", "exact"],
                    default="pad",
                    tooltip=(
                        "pad（推荐）短则补静音、超长保留；natural 不裁剪；"
                        "native/exact 最后可能裁掉超过槽位的句尾。"
                    ),
                ),
                io.Int.Input(
                    "fit_tolerance_ms",
                    display_name="允许时长误差（毫秒）",
                    default=180,
                    min=0,
                    max=2000,
                    step=10,
                    advanced=True,
                ),
                io.Int.Input(
                    "batch_gap_ms",
                    display_name="批量句间静音（毫秒）",
                    default=200,
                    min=0,
                    max=5000,
                    step=10,
                ),
                io.Combo.Input(
                    "postprocess_preset",
                    display_name="合并音频后处理",
                    options=list(POSTPROCESS_PRESETS),
                    default="off",
                ),
                io.Float.Input(
                    "postprocess_strength",
                    display_name="后处理强度",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "asr_enabled", display_name="逐句自动 ASR 校对", default=False
                ),
                io.Combo.Input(
                    "asr_backend",
                    display_name="ASR 后端",
                    options=list(ASR_BACKENDS),
                    default="auto",
                ),
                io.Combo.Input(
                    "asr_model",
                    display_name="ASR 模型",
                    options=list(ASR_MODELS),
                    default="base",
                ),
                io.Combo.Input(
                    "asr_device",
                    display_name="ASR 设备",
                    options=["auto", "cuda", "cpu"],
                    default="auto",
                    advanced=True,
                ),
                io.Float.Input(
                    "asr_threshold",
                    display_name="ASR 通过阈值",
                    default=0.82,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Combo.Input(
                    "subtitle_timing_mode",
                    display_name="回写字幕时间",
                    options=["actual", "original"],
                    default="actual",
                ),
                io.Combo.Input(
                    "subtitle_text_mode",
                    display_name="回写字幕文本",
                    options=["asr_passed", "asr_all", "original"],
                    default="asr_passed",
                ),
                io.Boolean.Input(
                    "subtitle_include_role",
                    display_name="字幕保留角色前缀",
                    default=True,
                ),
                io.Int.Input(
                    "asr_retry_count",
                    display_name="校对失败自动重试次数",
                    default=0,
                    min=0,
                    max=3,
                    step=1,
                    advanced=True,
                    tooltip="追加在旧版控件之后以兼容已保存工作流；识别失败只写报告，不阻断音频输出。",
                ),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="合并音频"),
                io.Audio.Output(
                    "line_audios", display_name="逐句音频", is_output_list=True
                ),
                io.String.Output("generation_report", display_name="生成报告 JSON"),
                io.String.Output("rewritten_srt", display_name="自动回写 SRT"),
                io.String.Output("timeline_report", display_name="可视化时间轴 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls, role_library: RoleLibrary, dialogue_script: DialogueScript, **kwargs
    ) -> bool | str:
        if not isinstance(role_library, RoleLibrary) or not isinstance(
            dialogue_script, DialogueScript
        ):
            return True
        missing = missing_roles(dialogue_script.lines, role_library.profiles)
        return "以下角色没有连接音色：" + "、".join(missing) if missing else True

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        role_library: RoleLibrary,
        dialogue_script: DialogueScript,
        seed: int,
        timeline_policy: str,
        fit_srt_slots: bool,
        slot_duration_mode: str,
        fit_tolerance_ms: int,
        batch_gap_ms: int,
        postprocess_preset: str,
        postprocess_strength: float,
        asr_enabled: bool,
        asr_backend: str,
        asr_model: str,
        asr_device: str,
        asr_threshold: float,
        subtitle_timing_mode: str,
        subtitle_text_mode: str,
        subtitle_include_role: bool,
        asr_retry_count: int = 0,
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        missing = missing_roles(dialogue_script.lines, role_library.profiles)
        if missing:
            raise ValueError("以下角色没有连接音色：" + "、".join(missing))
        asr_warning = ""
        legacy_retry_value = subtitle_timing_mode
        legacy_timing_value = str(subtitle_text_mode).strip().lower()
        legacy_text_value = str(subtitle_include_role).strip().lower()
        if legacy_timing_value in {"actual", "original"} and legacy_text_value in {
            "asr_passed",
            "asr_all",
            "original",
        }:
            try:
                legacy_retry_count = int(float(legacy_retry_value))
            except (TypeError, ValueError, OverflowError):
                legacy_retry_count = -1
            if 0 <= legacy_retry_count <= 3:
                subtitle_timing_mode = legacy_timing_value
                subtitle_text_mode = legacy_text_value
                subtitle_include_role = bool(asr_retry_count)
                asr_retry_count = legacy_retry_count
                asr_warning = "已自动还原 v0.11.0 工作流中错位的字幕与 ASR 重试选项。"
        try:
            retry_count = max(0, min(3, int(float(asr_retry_count or 0))))
        except (TypeError, ValueError, OverflowError):
            retry_count = 0
            asr_warning = (
                f"旧工作流 ASR 重试次数 {asr_retry_count!r} 无效，已按 0 次处理。"
            )
        review_requested = bool(asr_enabled or retry_count > 0)
        review_enabled = review_requested
        if review_enabled and not asr_available(asr_backend):
            review_enabled = False
            unavailable = "所选 ASR 后端不可用；已跳过校对并保留原字幕，音频正常输出。"
            asr_warning = "；".join(item for item in (asr_warning, unavailable) if item)
        work_handle = replace(model, release_after_run=False)
        clips: list[dict] = []
        line_reports: list[dict] = []
        sample_rate: int | None = None
        try:
            for offset, line in enumerate(dialogue_script.lines):
                profile = role_library.profiles[line.role]
                language = line.language or profile.language
                resolved_emotion, emotion_source = _resolve_line_emotion(line, profile)
                native_slot = bool(
                    fit_srt_slots and line.slot_ms and slot_duration_mode == "native"
                )
                native_fallback = False
                try:
                    audio, status = run_inference(
                        work_handle,
                        profile.speaker_audio,
                        line.text,
                        language,
                        line.duration_factor,
                        int(seed) + offset,
                        emotion=resolved_emotion,
                        sampling=sampling,
                        target_duration_seconds=(line.slot_ms / 1000.0)
                        if native_slot
                        else None,
                    )
                except NativeTargetDurationUnsupported:
                    native_slot = False
                    native_fallback = True
                    audio, status = run_inference(
                        work_handle,
                        profile.speaker_audio,
                        line.text,
                        language,
                        line.duration_factor,
                        int(seed) + offset,
                        emotion=resolved_emotion,
                        sampling=sampling,
                    )
                actual_ms = audio["waveform"].shape[-1] * 1000 / audio["sample_rate"]
                used_factor = line.duration_factor
                regenerated = False
                duration_adjustment = {"mode": "off", "action": "unchanged"}
                if native_slot:
                    audio, duration_adjustment = apply_duration_policy(
                        audio, line.slot_ms / 1000.0, "exact"
                    )
                    duration_adjustment["mode"] = "native"
                    actual_ms = audio_duration_ms(audio)
                elif (
                    fit_srt_slots
                    and line.slot_ms
                    and abs(actual_ms - line.slot_ms) > int(fit_tolerance_ms)
                ):
                    fitted = fit_duration_factor(used_factor, actual_ms, line.slot_ms)
                    if abs(fitted - used_factor) >= 0.02:
                        audio, status = run_inference(
                            work_handle,
                            profile.speaker_audio,
                            line.text,
                            language,
                            fitted,
                            int(seed) + offset,
                            emotion=resolved_emotion,
                            sampling=sampling,
                        )
                        used_factor = fitted
                        actual_ms = (
                            audio["waveform"].shape[-1] * 1000 / audio["sample_rate"]
                        )
                        regenerated = True
                if (
                    fit_srt_slots
                    and line.slot_ms
                    and slot_duration_mode in {"pad", "exact"}
                ):
                    audio, duration_adjustment = apply_duration_policy(
                        audio, line.slot_ms / 1000.0, slot_duration_mode
                    )
                    actual_ms = audio_duration_ms(audio)
                asr_attempts: list[dict] = []
                selected_review: dict | None = None
                selected_seed = int(seed) + offset
                if review_enabled:
                    for retry_index in range(retry_count + 1):
                        candidate_seed = int(seed) + offset + retry_index * 100_003
                        if retry_index > 0:
                            try:
                                candidate, candidate_status = run_inference(
                                    work_handle,
                                    profile.speaker_audio,
                                    line.text,
                                    language,
                                    used_factor,
                                    candidate_seed,
                                    emotion=resolved_emotion,
                                    sampling=sampling,
                                    target_duration_seconds=(line.slot_ms / 1000.0)
                                    if native_slot
                                    else None,
                                )
                            except NativeTargetDurationUnsupported:
                                candidate, candidate_status = run_inference(
                                    work_handle,
                                    profile.speaker_audio,
                                    line.text,
                                    language,
                                    used_factor,
                                    candidate_seed,
                                    emotion=resolved_emotion,
                                    sampling=sampling,
                                )
                            candidate_adjustment = duration_adjustment
                            if native_slot:
                                candidate, candidate_adjustment = apply_duration_policy(
                                    candidate, line.slot_ms / 1000.0, "exact"
                                )
                                candidate_adjustment["mode"] = "native"
                            elif (
                                fit_srt_slots
                                and line.slot_ms
                                and slot_duration_mode in {"pad", "exact"}
                            ):
                                candidate, candidate_adjustment = apply_duration_policy(
                                    candidate,
                                    line.slot_ms / 1000.0,
                                    slot_duration_mode,
                                )
                            candidate_ms = audio_duration_ms(candidate)
                        else:
                            candidate = audio
                            candidate_status = status
                            candidate_adjustment = duration_adjustment
                            candidate_ms = actual_ms
                        try:
                            transcript = transcribe_waveform(
                                candidate["waveform"],
                                int(candidate["sample_rate"]),
                                language=language,
                                backend=asr_backend,
                                model_name=asr_model,
                                device=asr_device,
                                download_root=_asr_download_root(),
                            )
                            review = {
                                **transcript,
                                **review_transcript(
                                    line.text,
                                    transcript["text"],
                                    language,
                                    asr_threshold,
                                ),
                            }
                        except Exception as exc:
                            review = {
                                "expected_text": line.text,
                                "recognized_text": "",
                                "passed": False,
                                "similarity": 0.0,
                                "threshold": float(asr_threshold),
                                "language": language,
                                "backend": asr_backend,
                                "model": asr_model,
                                "error": str(exc).strip() or type(exc).__name__,
                            }
                        asr_attempts.append(
                            {
                                "attempt": retry_index + 1,
                                "seed": candidate_seed,
                                "passed": bool(review.get("passed")),
                                "similarity": float(review.get("similarity", 0.0)),
                                "recognized_text": review.get(
                                    "recognized_text", review.get("text", "")
                                ),
                                **(
                                    {"error": review["error"]}
                                    if review.get("error")
                                    else {}
                                ),
                            }
                        )
                        if selected_review is None or float(
                            review.get("similarity", 0.0)
                        ) > float(selected_review.get("similarity", 0.0)):
                            audio = candidate
                            status = candidate_status
                            duration_adjustment = candidate_adjustment
                            actual_ms = candidate_ms
                            selected_review = review
                            selected_seed = candidate_seed
                        if bool(review.get("passed")):
                            audio = candidate
                            status = candidate_status
                            duration_adjustment = candidate_adjustment
                            actual_ms = candidate_ms
                            selected_review = review
                            selected_seed = candidate_seed
                            break
                        if review.get("error"):
                            break
                if sample_rate is None:
                    sample_rate = int(audio["sample_rate"])
                elif sample_rate != int(audio["sample_rate"]):
                    raise RuntimeError("逐句输出采样率不一致，无法合并。")
                clips.append(audio)
                line_report = {
                    **line.to_dict(),
                    "actual_duration_ms": round(actual_ms),
                    "used_duration_factor": round(used_factor, 4),
                    "regenerated_for_slot": regenerated,
                    "native_duration": native_slot,
                    "native_duration_fallback": native_fallback,
                    "duration_adjustment": duration_adjustment,
                    "emotion_source": emotion_source,
                    "effective_emotion_mode": (
                        resolved_emotion.mode if resolved_emotion is not None else "speaker"
                    ),
                    "status": status,
                }
                if review_enabled:
                    line_report["asr"] = {
                        **(selected_review or {}),
                        "selected_seed": selected_seed,
                        "attempt_count": len(asr_attempts),
                        "retry_count": max(0, len(asr_attempts) - 1),
                        "attempts": asr_attempts,
                    }
                line_reports.append(line_report)
        finally:
            if model.release_after_run:
                MODEL_CACHE.release(work_handle)
        assert sample_rate is not None
        effective_timeline_policy = timeline_policy
        timeline_warning = ""
        if dialogue_script.script_type == "batch" and timeline_policy == "overlay":
            effective_timeline_policy = "shift"
            timeline_warning = (
                "批量台词没有绝对起点，overlay 已自动改为 shift，避免所有台词重叠。"
            )
        waveform, placements = compose_timeline(
            [audio["waveform"] for audio in clips],
            dialogue_script.lines,
            sample_rate,
            effective_timeline_policy,
            batch_gap_ms if dialogue_script.script_type == "batch" else 0,
        )
        for line_report, placement in zip(line_reports, placements):
            line_report["timeline"] = placement.to_dict()
        combined_audio, postprocess_report = postprocess_audio(
            {"waveform": waveform, "sample_rate": sample_rate},
            postprocess_preset,
            postprocess_strength,
        )
        waveform = combined_audio["waveform"]
        try:
            rewritten_srt, subtitle_report = rewrite_srt(
                dialogue_script.lines,
                line_reports,
                timing_mode=subtitle_timing_mode,
                text_mode=subtitle_text_mode,
                include_role=subtitle_include_role,
            )
        except Exception as exc:
            rewritten_srt, subtitle_report = rewrite_srt(
                dialogue_script.lines,
                line_reports,
                timing_mode="actual",
                text_mode="original",
                include_role=True,
            )
            subtitle_report["error"] = str(exc).strip() or type(exc).__name__
            subtitle_report["fallback"] = "actual/original/include_role"
        report = {
            "script_type": dialogue_script.script_type,
            "timeline_policy": effective_timeline_policy,
            "requested_timeline_policy": timeline_policy,
            "timeline_warning": timeline_warning,
            "fit_srt_slots": bool(fit_srt_slots),
            "slot_duration_mode": slot_duration_mode,
            "postprocess": postprocess_report,
            "asr": {
                "requested": review_requested,
                "enabled": review_enabled,
                "backend": asr_backend,
                "model": asr_model,
                "device": asr_device,
                "threshold": float(asr_threshold),
                "maximum_retries": retry_count,
                "reviewed": sum("asr" in item for item in line_reports),
                "passed": sum(
                    bool((item.get("asr") or {}).get("passed")) for item in line_reports
                ),
                "warning": asr_warning,
            },
            "subtitle_rewrite": subtitle_report,
            "sample_rate": sample_rate,
            "duration_ms": round(waveform.shape[-1] * 1000 / sample_rate),
            "lines": line_reports,
        }
        return io.NodeOutput(
            {"waveform": waveform, "sample_rate": sample_rate},
            clips,
            json.dumps(report, ensure_ascii=False, indent=2),
            rewritten_srt,
            timeline_json(dialogue_script.lines, line_reports),
        )


class T8IndexTTS25ASRProofread(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_ASRProofread",
            display_name="IndexTTS 2.5 ASR 自动校对 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["Whisper", "ASR", "语音校对", "CER"],
            description=(
                "使用可选的本地 Whisper 后端识别 AUDIO；中文/日文计算 CER，其余语言计算 WER；"
                "统一简繁体、数字和标点，并输出差异明细及词级时间戳；"
                "不安装时只影响本节点。"
            ),
            inputs=[
                io.Audio.Input("audio", display_name="待校对音频"),
                io.String.Input(
                    "expected_text",
                    display_name="原始文本",
                    multiline=True,
                    dynamic_prompts=False,
                ),
                io.Combo.Input(
                    "language",
                    display_name="语言",
                    options=["AUTO", "ZH", "EN", "JA", "ES", "AR"],
                    default="AUTO",
                ),
                io.Combo.Input(
                    "backend",
                    display_name="ASR 后端",
                    options=list(ASR_BACKENDS),
                    default="auto",
                ),
                io.Combo.Input(
                    "model_name",
                    display_name="ASR 模型",
                    options=list(ASR_MODELS),
                    default="base",
                ),
                io.Combo.Input(
                    "device",
                    display_name="ASR 设备",
                    options=["auto", "cuda", "cpu"],
                    default="auto",
                    advanced=True,
                ),
                io.Float.Input(
                    "threshold",
                    display_name="通过阈值",
                    default=0.82,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
            ],
            outputs=[
                io.String.Output("recognized_text", display_name="识别文本"),
                io.Boolean.Output("passed", display_name="是否通过"),
                io.Float.Output("similarity", display_name="相似度"),
                io.String.Output("word_timestamps", display_name="词级时间戳 JSON"),
                io.String.Output("review_report", display_name="校对报告 JSON"),
                io.Image.Output("alignment_image", display_name="波形与逐字时间轴"),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio: dict,
        expected_text: str,
        language: str,
        backend: str,
        model_name: str,
        device: str,
        threshold: float,
    ) -> io.NodeOutput:
        if not asr_available(backend):
            raise RuntimeError(
                "所选 ASR 后端不可用；请安装 openai-whisper / faster-whisper，或切换后端。"
            )
        transcript = transcribe_waveform(
            audio["waveform"],
            int(audio["sample_rate"]),
            language=language,
            backend=backend,
            model_name=model_name,
            device=device,
            download_root=_asr_download_root(),
        )
        review = {
            **transcript,
            **review_transcript(expected_text, transcript["text"], language, threshold),
        }
        return io.NodeOutput(
            transcript["text"],
            bool(review["passed"]),
            float(review["similarity"]),
            json.dumps(
                transcript.get("word_timestamps") or [], ensure_ascii=False, indent=2
            ),
            json.dumps(review, ensure_ascii=False, indent=2),
            render_waveform_image(
                audio["waveform"],
                int(audio["sample_rate"]),
                transcript.get("word_timestamps") or [],
            ),
        )


class T8IndexTTS25SubtitleRewrite(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_SubtitleRewrite",
            display_name="IndexTTS 2.5 字幕自动回写 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["SRT rewrite", "字幕回写", "ASR subtitle"],
            description="根据多角色生成报告中的实际时间和 ASR 结果生成可保存的 SRT 文本。",
            inputs=[
                DialogueScriptType.Input("dialogue_script", display_name="台词脚本"),
                io.String.Input(
                    "generation_report",
                    display_name="多角色生成报告 JSON",
                    multiline=True,
                    dynamic_prompts=False,
                ),
                io.Combo.Input(
                    "timing_mode",
                    display_name="字幕时间",
                    options=["actual", "original"],
                    default="actual",
                ),
                io.Combo.Input(
                    "text_mode",
                    display_name="字幕文本",
                    options=["asr_passed", "asr_all", "original"],
                    default="asr_passed",
                ),
                io.Boolean.Input(
                    "include_role", display_name="保留角色前缀", default=True
                ),
            ],
            outputs=[
                io.String.Output("srt", display_name="回写 SRT"),
                io.String.Output("rewrite_report", display_name="回写报告 JSON"),
            ],
        )

    @classmethod
    def execute(
        cls,
        dialogue_script: DialogueScript,
        generation_report: str,
        timing_mode: str,
        text_mode: str,
        include_role: bool,
    ) -> io.NodeOutput:
        try:
            payload = json.loads(str(generation_report or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"生成报告 JSON 格式错误：{exc.msg}（第 {exc.lineno} 行）"
            ) from exc
        reports = payload.get("lines") if isinstance(payload, dict) else None
        if not isinstance(reports, list):
            raise ValueError("生成报告中缺少 lines 数组。")
        content, report = rewrite_srt(
            dialogue_script.lines,
            reports,
            timing_mode=timing_mode,
            text_mode=text_mode,
            include_role=include_role,
        )
        return io.NodeOutput(content, json.dumps(report, ensure_ascii=False, indent=2))


class T8IndexTTS25ReferenceQuality(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_ReferenceQuality",
            display_name="IndexTTS 2.5 参考音频质量检测 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=[
                "reference quality",
                "参考音频检测",
                "自动裁剪",
                "waveform",
            ],
            description=(
                "检测参考音频的时长、静音、削波、响度、信噪比和直流偏移；"
                "可自动裁掉首尾静音并从超长音频中选择能量最集中的片段。"
            ),
            inputs=[
                io.Audio.Input("audio", display_name="参考音频"),
                io.Boolean.Input(
                    "auto_prepare", display_name="自动裁剪优化", default=True
                ),
                io.Float.Input(
                    "maximum_seconds",
                    display_name="最长保留秒数",
                    default=15.0,
                    min=3.0,
                    max=30.0,
                    step=0.5,
                ),
                io.Int.Input(
                    "silence_padding_ms",
                    display_name="首尾保留毫秒",
                    default=150,
                    min=0,
                    max=1000,
                    step=10,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Audio.Output("prepared_audio", display_name="检测/优化后的参考音频"),
                io.String.Output("quality_report", display_name="质量报告 JSON"),
                io.Image.Output("waveform_image", display_name="参考音频波形"),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio: dict,
        auto_prepare: bool,
        maximum_seconds: float,
        silence_padding_ms: int,
    ) -> io.NodeOutput:
        sample_rate = int(audio["sample_rate"])
        if auto_prepare:
            prepared, report = prepare_reference_audio(
                audio["waveform"],
                sample_rate,
                max_seconds=float(maximum_seconds),
                padding_ms=int(silence_padding_ms),
            )
        else:
            prepared, preparation = prepare_reference_audio(
                audio["waveform"],
                sample_rate,
                trim_silence=False,
                max_seconds=86_400,
            )
            quality = analyze_reference_audio(prepared, sample_rate)
            report = {
                "original": quality,
                "prepared": quality,
                "trimmed": False,
                "selected_start_seconds": 0.0,
                "selected_end_seconds": quality["duration_seconds"],
            }
        result = {"waveform": prepared.unsqueeze(0), "sample_rate": sample_rate}
        return io.NodeOutput(
            result,
            json.dumps(report, ensure_ascii=False, indent=2),
            render_waveform_image(prepared, sample_rate),
        )


class T8IndexTTS25MemoryControl(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_MemoryControl",
            display_name="IndexTTS 2.5 显存管理 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["释放显存", "VRAM", "model cache", "memory watchdog"],
            description=(
                "只管理本扩展加载的 IndexTTS 2.5 与可选 ASR 模型；可查看状态、释放空闲模型或全部释放，"
                "不会清空 ComfyUI 的其他模型。"
            ),
            inputs=[
                io.Combo.Input(
                    "action",
                    display_name="操作",
                    options=["status", "release_idle", "release_all"],
                    default="status",
                ),
                io.Float.Input(
                    "idle_seconds",
                    display_name="空闲秒数",
                    default=300.0,
                    min=0.0,
                    max=86_400.0,
                    step=1.0,
                ),
            ],
            outputs=[
                io.String.Output(
                    "memory_report", display_name="显存与模型缓存报告 JSON"
                )
            ],
        )

    @classmethod
    def execute(cls, action: str, idle_seconds: float) -> io.NodeOutput:
        before = MODEL_CACHE.status()
        asr_before = asr_cache_status()
        released_asr = 0
        if action == "release_all":
            released = MODEL_CACHE.clear()
            released_asr = clear_asr_cache()
        elif action == "release_idle":
            released = MODEL_CACHE.evict_idle(float(idle_seconds))
        elif action == "status":
            released = 0
        else:
            raise ValueError(f"未知显存管理操作：{action}")
        return io.NodeOutput(
            json.dumps(
                {
                    "action": action,
                    "released_models": released,
                    "released_asr_models": released_asr,
                    "before": before,
                    "after": MODEL_CACHE.status(),
                    "asr_before": asr_before,
                    "asr_after": asr_cache_status(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


class T8IndexTTS25AudioCppGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_AudioCppGenerate",
            display_name="IndexTTS 2.5 audio.cpp 实验生成 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["audio.cpp", "GGUF", "IndexTTS quantized", "实验后端"],
            description=(
                "调用用户单独安装的 audio.cpp CLI 和 IndexTTS2.5 GGUF；"
                "与默认 Python 模型加载器完全隔离，不会改变现有工作流。"
            ),
            inputs=[
                io.String.Input(
                    "executable_path",
                    display_name="audiocpp_cli 可执行文件绝对路径",
                    default="",
                ),
                io.String.Input(
                    "gguf_model_path",
                    display_name="IndexTTS2.5-GGUF 目录或文件路径",
                    default="",
                ),
                io.Audio.Input("speaker_audio", display_name="音色参考音频"),
                io.String.Input(
                    "text",
                    display_name="待合成文本",
                    multiline=True,
                    default="这是隔离的 audio.cpp IndexTTS 2.5 实验后端。",
                    dynamic_prompts=True,
                ),
                io.Combo.Input(
                    "language",
                    display_name="语言",
                    options=["AUTO", "ZH", "EN", "JA", "ES", "AR"],
                    default="ZH",
                ),
                io.Combo.Input(
                    "backend",
                    display_name="audio.cpp 后端",
                    options=["cuda", "cpu", "vulkan", "hip", "metal"],
                    default="cuda",
                    tooltip="NVIDIA 选 cuda，AMD 可选 hip/vulkan，Apple Silicon 选 metal。",
                ),
                io.Float.Input(
                    "duration_factor",
                    display_name="官方时长适配倍率（无单位）",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                ),
                io.Boolean.Input(
                    "memory_saver",
                    display_name="请求阶段结束后释放临时图",
                    default=True,
                    advanced=True,
                ),
                EmotionType.Input(
                    "emotion", display_name="可选情感控制", optional=True
                ),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="audio.cpp 生成音频"),
                io.String.Output("report", display_name="实验后端报告 JSON"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        executable_path: str,
        gguf_model_path: str,
        speaker_audio: dict,
        text: str,
        language: str,
        backend: str,
        duration_factor: float,
        memory_saver: bool,
        emotion: EmotionConfig | None = None,
    ) -> io.NodeOutput:
        probe = await probe_audiocpp(executable_path)
        if not probe.get("available"):
            raise RuntimeError(
                "audio.cpp CLI 探测失败："
                + str(probe.get("error") or probe.get("summary") or "未知原因")
            )
        speaker_path, speaker_notes = comfy_audio_to_reference_wav(
            speaker_audio, kind="audiocpp_speaker"
        )
        emotion_text = ""
        emotion_audio = None
        emotion_vector = None
        emotion_alpha = 1.0
        if emotion is not None:
            emotion_alpha = float(emotion.strength)
            if emotion.mode == "text":
                emotion_text = str(emotion.text or text)
            elif (
                emotion.mode == "reference_audio"
                and emotion.reference_audio is not None
            ):
                emotion_audio, _notes = comfy_audio_to_reference_wav(
                    emotion.reference_audio, kind="audiocpp_emotion"
                )
            elif emotion.mode == "vector":
                emotion_vector = emotion.vector
        with tempfile.TemporaryDirectory(prefix="t8_audiocpp_") as temporary:
            output = Path(temporary) / "output.wav"
            report = await run_audiocpp(
                executable_path,
                gguf_model_path,
                speaker_path,
                output,
                text,
                language,
                backend=backend,
                duration_factor=duration_factor,
                memory_saver=memory_saver,
                emotion_text=emotion_text,
                emotion_audio=emotion_audio,
                emotion_vector=emotion_vector,
                emotion_alpha=emotion_alpha,
            )
            waveform, sample_rate = torchaudio.load(str(output))
        report.update(
            probe=probe,
            notes=speaker_notes,
            limitations=(
                "实验后端不共享 Python 模型；GGUF 量化与文本归一化可能产生听感差异，"
                "正式任务请先做五语种、情感和发音标注对比。"
            ),
        )
        return io.NodeOutput(
            {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)},
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class T8IndexTTS25Environment(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_Environment",
            display_name="IndexTTS 2.5 环境与可选加速 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["加速诊断", "DeepSpeed", "FlashAttention", "Triton"],
            description="只做能力探测，不安装依赖、不加载模型；DeepSpeed 等附加包缺失属于正常情况。",
            inputs=[
                io.Combo.Input(
                    "device",
                    display_name="检查设备",
                    options=_device_options(),
                    default="auto",
                )
            ],
            outputs=[
                io.String.Output("environment_report", display_name="环境报告 JSON")
            ],
        )

    @classmethod
    def execute(cls, device: str) -> io.NodeOutput:
        resolved = _resolve_device(device)
        capabilities = probe_acceleration(resolved)
        modes = {
            mode: {
                "effective": selected.effective,
                "available": selected.available,
                "reason": selected.reason,
            }
            for mode in MODES
            for selected in [resolve_acceleration(mode, resolved, capabilities)]
        }
        return io.NodeOutput(
            json.dumps(
                {
                    "device": resolved,
                    "recommended": recommend_runtime_config(capabilities),
                    "capabilities": capabilities,
                    "modes": modes,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


class T8IndexTTS25UpdateCheck(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_UpdateCheck",
            display_name="IndexTTS 2.5 检查更新 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["update", "上游动态", "version check"],
            description="手动联网比较官方代码、官方模型和本节点版本；只报告，不下载、不覆盖。",
            inputs=[
                io.Int.Input(
                    "refresh_token",
                    display_name="刷新编号（修改后重新检查）",
                    default=0,
                    min=0,
                    max=1_000_000,
                    step=1,
                ),
                io.Float.Input(
                    "timeout_seconds",
                    display_name="单项超时（秒）",
                    default=12.0,
                    min=3.0,
                    max=60.0,
                    step=1.0,
                    advanced=True,
                ),
            ],
            outputs=[
                io.String.Output("report", display_name="更新检查 JSON"),
                io.String.Output("summary", display_name="检查摘要"),
            ],
        )

    @classmethod
    def execute(cls, refresh_token: int, timeout_seconds: float) -> io.NodeOutput:
        del refresh_token
        report = check_updates(PROJECT_VERSION, load_manifest(), timeout_seconds)
        return io.NodeOutput(
            json.dumps(report, ensure_ascii=False, indent=2), report["summary"]
        )


class T8IndexTTS25RuntimeBenchmark(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_RuntimeBenchmark",
            display_name="IndexTTS 2.5 真实性能基准 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["benchmark", "RTF", "加速实测", "显存"],
            description="实测当前模型加载器所选模式的 RTF 与峰值显存；切换加速模式后重复运行即可对比。",
            inputs=[
                ModelType.Input("model", display_name="IndexTTS 2.5 模型"),
                io.Audio.Input("speaker_audio", display_name="固定参考音频"),
                io.String.Input(
                    "text",
                    display_name="固定测试文本",
                    multiline=True,
                    default="这是 IndexTTS 2.5 真实性能基准。所有运行使用相同文本、参考音频和随机种子。",
                ),
                io.Combo.Input(
                    "language",
                    display_name="语言",
                    options=["ZH", "EN", "JA", "ES", "AR"],
                    default="ZH",
                ),
                io.Float.Input(
                    "duration_factor",
                    display_name="官方时长适配倍率",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                ),
                io.Int.Input(
                    "repeat_count",
                    display_name="正式测量次数",
                    default=2,
                    min=1,
                    max=5,
                    step=1,
                ),
                io.Boolean.Input(
                    "warmup",
                    display_name="先预热一次（推荐）",
                    default=True,
                    tooltip="预热会完成模型加载、参考条件提取和编译，不计入正式 RTF。",
                ),
                io.Int.Input(
                    "seed",
                    display_name="固定 seed",
                    default=20250827,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                ),
                EmotionType.Input("emotion", display_name="情感控制", optional=True),
                SamplingType.Input("sampling", display_name="采样设置", optional=True),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="最后一次测试音频"),
                io.String.Output("benchmark_report", display_name="真实基准 JSON"),
                io.String.Output("summary", display_name="基准摘要"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        speaker_audio: dict,
        text: str,
        language: str,
        duration_factor: float,
        repeat_count: int,
        warmup: bool,
        seed: int,
        emotion: EmotionConfig | None = None,
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        repeat_count = max(1, min(5, int(repeat_count)))
        warmup_seconds = None
        last_audio = None
        if warmup:
            started = time.perf_counter()
            last_audio, _ = run_inference(
                model, speaker_audio, text, language, duration_factor, int(seed), emotion, sampling
            )
            warmup_seconds = time.perf_counter() - started

        measurements = []
        for index in range(repeat_count):
            if model.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            last_audio, status = run_inference(
                model, speaker_audio, text, language, duration_factor, int(seed), emotion, sampling
            )
            if model.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            audio_seconds = audio_duration_ms(last_audio) / 1000.0
            measurements.append({
                "run": index + 1,
                "inference_seconds": round(elapsed, 6),
                "audio_seconds": round(audio_seconds, 6),
                "rtf": round(elapsed / max(audio_seconds, 1e-9), 6),
                "peak_vram_gb": (
                    round(torch.cuda.max_memory_allocated() / (1024**3), 6)
                    if model.device.startswith("cuda") and torch.cuda.is_available()
                    else None
                ),
                "status": status,
            })
        assert last_audio is not None
        aggregate = summarize_measurements(measurements)
        report = {
            "acceleration_requested": model.acceleration_requested,
            "acceleration_effective": model.acceleration_effective,
            "acceleration_note": model.acceleration_note,
            "device": model.device,
            "precision": "bfloat16" if model.use_bf16 else "float16" if model.use_fp16 else "float32",
            "reference_device": model.reference_device or model.device,
            "warmup": bool(warmup),
            "warmup_seconds": round(warmup_seconds, 6) if warmup_seconds is not None else None,
            "aggregate": aggregate,
            "measurements": measurements,
        }
        summary = (
            f"生效模式 {model.acceleration_effective}｜中位 RTF {aggregate['median_rtf']:.3f}｜"
            f"最佳 RTF {aggregate['best_rtf']:.3f}｜"
            + (f"峰值显存 {aggregate['peak_vram_gb']:.2f}GB｜" if aggregate['peak_vram_gb'] is not None else "")
            + "RTF<1 表示快于实时。切换模型加载器的加速模式后用相同输入重跑即可公平比较。"
        )
        return io.NodeOutput(last_audio, json.dumps(report, ensure_ascii=False, indent=2), summary)


class T8IndexTTS25Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_Generate",
            display_name="IndexTTS 2.5 语音生成 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["IndexTTS TTS", "voice clone", "语音克隆", "T8star-Aix"],
            description=(
                "使用正式 IndexTTS 2.5 模型进行多语种零样本音色克隆，并输出标准 ComfyUI AUDIO；"
                "支持手写 <文字|读音> 或连接发音控制节点。"
            ),
            inputs=[
                ModelType.Input("model", display_name="IndexTTS 2.5 模型"),
                io.Audio.Input("speaker_audio", display_name="音色参考音频"),
                io.String.Input(
                    "text",
                    display_name="待合成文本",
                    multiline=True,
                    default="欢迎使用 IndexTTS 2.5，来自 B 站：T8star-Aix。",
                    dynamic_prompts=True,
                    tooltip=(
                        "可直接使用 <文字|读音>：中文 <行|XING2>、英文 CMU 音素、日语假名；"
                        "批量规则建议连接“发音控制”节点。"
                    ),
                ),
                io.Combo.Input(
                    "language",
                    display_name="语言",
                    options=["ZH", "EN", "JA", "ES", "AR"],
                    default="ZH",
                ),
                io.Float.Input(
                    "duration_factor",
                    display_name="官方时长适配倍率（越小越短）",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    display_mode=io.NumberDisplay.slider,
                    tooltip=(
                        "控制目标声学长度，不等同于自然语气语速；建议 0.8–1.25，"
                        "极端值可能拉长或失真。"
                    ),
                ),
                io.Combo.Input(
                    "target_duration_mode",
                    display_name="目标时长模式",
                    options=["off", "native", "natural", "pad", "exact"],
                    default="off",
                    tooltip=(
                        "natural/pad 不裁掉超长句；native/exact 会在最后精确裁剪，"
                        "只适合必须严格对齐且确认台词能完整放入的场景。"
                    ),
                ),
                io.Float.Input(
                    "target_duration_seconds",
                    display_name="目标时长（秒）",
                    default=0.0,
                    min=0.0,
                    max=3600.0,
                    step=0.1,
                ),
                io.Combo.Input(
                    "postprocess_preset",
                    display_name="音频后处理",
                    options=list(POSTPROCESS_PRESETS),
                    default="off",
                ),
                io.Float.Input(
                    "postprocess_strength",
                    display_name="后处理强度",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Int.Input(
                    "seed",
                    display_name="seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Int.Input(
                    "quality_retry_count",
                    display_name="追加候选数量",
                    default=0,
                    min=0,
                    max=3,
                    step=1,
                    advanced=True,
                    tooltip="0 为单次；1–3 会更换 seed 生成额外候选并自动选优，同时从“全部候选”输出供试听或单段替换。",
                ),
                io.Combo.Input(
                    "quality_asr_backend",
                    display_name="质检 ASR 后端",
                    options=list(ASR_BACKENDS),
                    default="auto",
                    advanced=True,
                ),
                io.Combo.Input(
                    "quality_asr_model",
                    display_name="质检 ASR 模型",
                    options=list(ASR_MODELS),
                    default="base",
                    advanced=True,
                ),
                io.Combo.Input(
                    "quality_asr_device",
                    display_name="质检 ASR 设备",
                    options=["auto", "cuda", "cpu"],
                    default="auto",
                    advanced=True,
                ),
                io.Float.Input(
                    "quality_threshold",
                    display_name="质检通过阈值",
                    default=0.82,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                EmotionType.Input(
                    "emotion",
                    display_name="情感控制",
                    optional=True,
                    tooltip="不连接时跟随音色参考。",
                ),
                SamplingType.Input(
                    "sampling",
                    display_name="采样设置",
                    optional=True,
                    tooltip="不连接时使用稳定默认值。",
                ),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="生成音频"),
                io.String.Output("generation_info", display_name="生成信息"),
                io.Audio.Output(
                    "all_candidates",
                    display_name="全部候选音频",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        text: str,
        duration_factor: float,
        target_duration_mode: str = "off",
        target_duration_seconds: float = 0.0,
        **kwargs,
    ) -> bool | str:
        if not str(text).strip():
            return "待合成文本不能为空。"
        if not 0.5 <= float(duration_factor) <= 2.0:
            return "时长系数必须在 0.5 到 2.0 之间。"
        if (
            target_duration_mode != "off"
            and not 0.1 <= float(target_duration_seconds) <= 3600
        ):
            return "启用目标时长时，目标时长必须在 0.1–3600 秒。"
        return True

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        speaker_audio: dict,
        text: str,
        language: str,
        duration_factor: float,
        target_duration_mode: str,
        target_duration_seconds: float,
        postprocess_preset: str,
        postprocess_strength: float,
        seed: int,
        quality_retry_count: int = 0,
        quality_asr_backend: str = "auto",
        quality_asr_model: str = "base",
        quality_asr_device: str = "auto",
        quality_threshold: float = 0.82,
        emotion: EmotionConfig | None = None,
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        try:
            retry_count = max(0, min(3, int(float(quality_retry_count or 0))))
        except (TypeError, ValueError, OverflowError):
            retry_count = 0
        quality_requested = retry_count > 0
        quality_enabled = quality_requested and asr_available(quality_asr_backend)
        quality_warning = ""
        if quality_requested and not quality_enabled:
            quality_warning = "所选 ASR 后端不可用；已改用音频技术指标选优并保留全部候选。"

        def generate_candidate(candidate_seed: int) -> tuple[dict, str, dict]:
            native_requested = (
                target_duration_mode == "native" and float(target_duration_seconds) > 0
            )
            native_fallback = False
            try:
                candidate_audio, candidate_status = run_inference(
                    handle=model,
                    speaker_audio=speaker_audio,
                    text=text,
                    language=language,
                    duration_factor=duration_factor,
                    seed=candidate_seed,
                    emotion=emotion,
                    sampling=sampling,
                    target_duration_seconds=float(target_duration_seconds)
                    if native_requested
                    else None,
                )
            except NativeTargetDurationUnsupported:
                native_requested = False
                native_fallback = True
                candidate_audio, candidate_status = run_inference(
                    handle=model,
                    speaker_audio=speaker_audio,
                    text=text,
                    language=language,
                    duration_factor=duration_factor,
                    seed=candidate_seed,
                    emotion=emotion,
                    sampling=sampling,
                )
            duration_report: dict = {"mode": "off", "action": "unchanged"}
            used_factor = float(duration_factor)
            if native_requested:
                candidate_audio, duration_report = apply_duration_policy(
                    candidate_audio, target_duration_seconds, "exact"
                )
                duration_report["mode"] = "native"
                candidate_status += (
                    f" | target={float(target_duration_seconds):.2f}s/native"
                )
            elif target_duration_mode != "off" and float(target_duration_seconds) > 0:
                actual_ms = audio_duration_ms(candidate_audio)
                fitted = fit_duration_factor(
                    used_factor, actual_ms, float(target_duration_seconds) * 1000.0
                )
                if abs(fitted - used_factor) >= 0.02:
                    candidate_audio, candidate_status = run_inference(
                        handle=model,
                        speaker_audio=speaker_audio,
                        text=text,
                        language=language,
                        duration_factor=fitted,
                        seed=candidate_seed,
                        emotion=emotion,
                        sampling=sampling,
                    )
                    used_factor = fitted
                if target_duration_mode in {"pad", "exact"}:
                    candidate_audio, duration_report = apply_duration_policy(
                        candidate_audio, target_duration_seconds, target_duration_mode
                    )
                else:
                    duration_report = {
                        "mode": "natural",
                        "target_ms": round(float(target_duration_seconds) * 1000),
                        "final_ms": round(audio_duration_ms(candidate_audio)),
                        "action": "regenerated"
                        if used_factor != float(duration_factor)
                        else "unchanged",
                    }
                candidate_status += (
                    f" | target={float(target_duration_seconds):.2f}s/{target_duration_mode}"
                    f" | fitted_factor={used_factor:.3f}"
                )
            if native_fallback:
                candidate_status += " | 原生目标时长不可用，已回退为二次推理适配"
            candidate_audio, postprocess_report = postprocess_audio(
                candidate_audio, postprocess_preset, postprocess_strength
            )
            candidate_status += " | duration=" + json.dumps(
                duration_report, ensure_ascii=False, separators=(",", ":")
            )
            candidate_status += " | post=" + json.dumps(
                postprocess_report, ensure_ascii=False, separators=(",", ":")
            )
            return (
                candidate_audio,
                candidate_status,
                {
                    "duration": duration_report,
                    "postprocess": postprocess_report,
                    "used_duration_factor": used_factor,
                },
            )

        audio, status, selected_metadata = generate_candidate(int(seed))
        all_candidates = [audio]
        if quality_requested:
            attempts: list[dict] = []
            candidate_records: list[tuple[dict, str, dict, dict]] = []
            for attempt_index in range(retry_count + 1):
                candidate_seed = int(seed) + attempt_index * 100_003
                if attempt_index:
                    candidate, candidate_status, candidate_metadata = (
                        generate_candidate(candidate_seed)
                    )
                else:
                    candidate, candidate_status, candidate_metadata = (
                        audio,
                        status,
                        selected_metadata,
                    )
                technical = technical_audio_review(candidate)
                review: dict = {
                    "expected_text": text,
                    "recognized_text": "",
                    "passed": False,
                    "similarity": None,
                    "threshold": float(quality_threshold),
                    "language": language,
                    "technical": technical,
                }
                if quality_enabled:
                    try:
                        transcript = transcribe_waveform(
                            candidate["waveform"],
                            int(candidate["sample_rate"]),
                            language=language,
                            backend=quality_asr_backend,
                            model_name=quality_asr_model,
                            device=quality_asr_device,
                            download_root=_asr_download_root(),
                        )
                        review.update(transcript)
                        review.update(
                            review_transcript(
                                text,
                                transcript["text"],
                                language,
                                quality_threshold,
                            )
                        )
                    except Exception as exc:
                        review["error"] = str(exc).strip() or type(exc).__name__
                review["combined_score"] = combined_candidate_score(
                    float(technical["score"]), review.get("similarity")
                )
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "seed": candidate_seed,
                        "passed": bool(review.get("passed")),
                        "similarity": review.get("similarity"),
                        "combined_score": review["combined_score"],
                        "technical": technical,
                        "recognized_text": review.get(
                            "recognized_text", review.get("text", "")
                        ),
                        **({"error": review["error"]} if review.get("error") else {}),
                    }
                )
                candidate_records.append(
                    (candidate, candidate_status, candidate_metadata, review)
                )
                if attempt_index:
                    all_candidates.append(candidate)

            selected_index = select_best_candidate(
                [record[3] for record in candidate_records]
            )
            audio, status, selected_metadata, selected_review = candidate_records[
                selected_index
            ]
            selected_seed = int(seed) + selected_index * 100_003
            errors = [str(item["error"]) for item in attempts if item.get("error")]
            if errors:
                quality_warning = "部分 ASR 质检失败，已使用其余评分或音频技术指标选优：" + errors[0]
            quality_report = {
                "requested": True,
                "enabled": quality_enabled,
                "selection_method": "asr+technical" if quality_enabled else "technical",
                "selected_seed": selected_seed,
                "selected_candidate": selected_index + 1,
                "attempt_count": len(attempts),
                "additional_candidates": retry_count,
                "review": selected_review,
                "attempts": attempts,
                "generation": selected_metadata,
                "warning": quality_warning,
            }
            status += " | quality=" + json.dumps(
                quality_report, ensure_ascii=False, separators=(",", ":")
            )
        return io.NodeOutput(audio, status, all_candidates)


class T8IndexTTS25AudioPostProcess(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_AudioPostProcess",
            display_name="IndexTTS 2.5 人声后处理 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["voice clarity", "人声清晰", "温暖", "去刺耳", "normalize"],
            description="对任意 ComfyUI AUDIO 应用可选、可混合强度的人声预设；off 时原样输出。",
            inputs=[
                io.Audio.Input("audio", display_name="输入音频"),
                io.Combo.Input(
                    "preset",
                    display_name="预设",
                    options=list(POSTPROCESS_PRESETS),
                    default="voice_clarity",
                ),
                io.Float.Input(
                    "strength",
                    display_name="强度",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
                io.Float.Input(
                    "target_peak_db",
                    display_name="目标峰值（dBFS）",
                    default=-1.0,
                    min=-12.0,
                    max=-0.1,
                    step=0.1,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="处理后音频"),
                io.String.Output("report", display_name="后处理报告 JSON"),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio: dict,
        preset: str,
        strength: float,
        target_peak_db: float,
    ) -> io.NodeOutput:
        result, report = postprocess_audio(audio, preset, strength, target_peak_db)
        return io.NodeOutput(result, json.dumps(report, ensure_ascii=False, indent=2))


class T8IndexTTS25Extension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        register_model_paths()
        LOGGER.info("Loaded comfyui-indextts25-T8 (V3 nodes, IndexTTS 2.5 only)")

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            T8IndexTTS25ModelLoader,
            T8IndexTTS25EmotionControl,
            T8IndexTTS25SamplingConfig,
            T8IndexTTS25TextPreview,
            T8IndexTTS25Pronunciation,
            T8IndexTTS25Generate,
            T8IndexTTS25VoiceProfile,
            T8IndexTTS25RoleLibrary,
            T8IndexTTS25MergeVoiceEmotions,
            T8IndexTTS25DialogueScript,
            T8IndexTTS25DialogueEmotionSuggest,
            T8IndexTTS25TimelineEditor,
            T8IndexTTS25DialogueGenerate,
            T8IndexTTS25ASRProofread,
            T8IndexTTS25SubtitleRewrite,
            T8IndexTTS25ReferenceQuality,
            T8IndexTTS25MemoryControl,
            T8IndexTTS25AudioCppGenerate,
            T8IndexTTS25AudioPostProcess,
            T8IndexTTS25Environment,
            T8IndexTTS25UpdateCheck,
            T8IndexTTS25RuntimeBenchmark,
        ]


async def comfy_entrypoint() -> T8IndexTTS25Extension:
    return T8IndexTTS25Extension()


__all__ = [
    "T8IndexTTS25ModelLoader",
    "T8IndexTTS25EmotionControl",
    "T8IndexTTS25SamplingConfig",
    "T8IndexTTS25TextPreview",
    "T8IndexTTS25Pronunciation",
    "T8IndexTTS25Generate",
    "T8IndexTTS25VoiceProfile",
    "T8IndexTTS25RoleLibrary",
    "T8IndexTTS25MergeVoiceEmotions",
    "T8IndexTTS25DialogueScript",
    "T8IndexTTS25DialogueEmotionSuggest",
    "T8IndexTTS25TimelineEditor",
    "T8IndexTTS25DialogueGenerate",
    "T8IndexTTS25ASRProofread",
    "T8IndexTTS25SubtitleRewrite",
    "T8IndexTTS25ReferenceQuality",
    "T8IndexTTS25MemoryControl",
    "T8IndexTTS25AudioCppGenerate",
    "T8IndexTTS25AudioPostProcess",
    "T8IndexTTS25Environment",
    "T8IndexTTS25UpdateCheck",
    "T8IndexTTS25RuntimeBenchmark",
    "T8IndexTTS25Extension",
    "comfy_entrypoint",
]
