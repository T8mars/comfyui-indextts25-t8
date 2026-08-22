from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .runtime.inference_adapter import run_inference
from .runtime.audio_processing import (
    DURATION_MODES,
    POSTPROCESS_PRESETS,
    apply_duration_policy,
    audio_duration_ms,
    postprocess_audio,
)
from .runtime.model_cache import MODEL_CACHE
from .runtime.acceleration import MODES, probe_acceleration, resolve_acceleration
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
    load_manifest,
    model_fingerprint,
    model_options,
    register_model_paths,
    resolve_model,
    validate_model_dir,
)


LOGGER = logging.getLogger("comfyui-indextts25-T8")
CATEGORY = "T8star-Aix/Audio/IndexTTS 2.5"
ModelType = io.Custom("T8_INDEXTTS25_MODEL")
EmotionType = io.Custom("T8_INDEXTTS25_EMOTION")
SamplingType = io.Custom("T8_INDEXTTS25_SAMPLING")
VoiceType = io.Custom("T8_INDEXTTS25_VOICE")
RoleLibraryType = io.Custom("T8_INDEXTTS25_ROLE_LIBRARY")
DialogueScriptType = io.Custom("T8_INDEXTTS25_DIALOGUE_SCRIPT")


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
    if not (selected.startswith("cuda") or selected.startswith("cpu") or selected.startswith("xpu") or selected.startswith("mps")):
        raise RuntimeError(f"IndexTTS 2.5 暂不支持 ComfyUI 当前设备：{selected}")
    return selected


def _use_bf16(precision: str, device: str) -> bool:
    if precision == "float32":
        return False
    if precision == "bfloat16":
        if device == "cpu" or device.startswith("mps"):
            raise RuntimeError("bfloat16 仅建议在支持该格式的 CUDA/XPU 设备上使用。")
        return True
    if device.startswith("cuda"):
        index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
        try:
            return bool(torch.cuda.is_bf16_supported(index))
        except TypeError:
            with torch.cuda.device(index):
                return bool(torch.cuda.is_bf16_supported())
    return device.startswith("xpu")


def _is_low_vram(device: str, threshold_gb: float = 10.0) -> bool:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return False
    index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
    total_gb = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
    return total_gb < threshold_gb


class T8IndexTTS25ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = model_options()
        return io.Schema(
            node_id="T8_IndexTTS25_ModelLoader",
            display_name="IndexTTS 2.5 模型加载器 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["IndexTTS 2.5", "T8star-Aix", "TTS model loader"],
            description="发现并校验 ComfyUI/models/TTS 下的正式 IndexTTS 2.5 模型；权重在首次生成时按需载入。",
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
                    options=["auto", "bfloat16", "float32"],
                    default="auto",
                    tooltip="auto 会在支持的 GPU 上使用 bfloat16，否则使用 float32。",
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
                io.Boolean.Input(
                    "verify_hashes",
                    display_name="完整 SHA-256 校验",
                    default=False,
                    advanced=True,
                    tooltip="首次校验约需读取 5GB 文件；平时仅做文件大小校验即可。",
                ),
                io.String.Input(
                    "custom_model_path",
                    display_name="自定义模型绝对路径",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="留空时使用上方模型列表；仅用于已有的完整 IndexTTS 2.5 目录。",
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
        verify_hashes: bool,
        custom_model_path: str = "",
    ) -> str:
        try:
            path = resolve_model(model_name, custom_model_path)
            return model_fingerprint(path)
        except Exception as exc:
            return f"missing:{model_name}:{custom_model_path}:{exc}"

    @classmethod
    def validate_inputs(cls, model_name: str, custom_model_path: str = "", **kwargs) -> bool | str:
        if model_name == MISSING_MODEL_OPTION and not custom_model_path.strip():
            return "未找到 IndexTTS 2.5 模型；请先运行 scripts/download_models.py。"
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
        verify_hashes: bool,
        custom_model_path: str = "",
    ) -> io.NodeOutput:
        model_dir = resolve_model(model_name, custom_model_path)
        report = validate_model_dir(model_dir, verify_hashes=verify_hashes)
        report.require_valid()
        resolved_device = _resolve_device(device)
        low_vram = _is_low_vram(resolved_device)
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
            use_bf16=_use_bf16(precision, resolved_device),
            use_cuda_kernel=acceleration.use_cuda_kernel,
            use_torch_compile=acceleration.use_torch_compile,
            use_accel=acceleration.use_accel,
            use_deepspeed=acceleration.use_deepspeed,
            acceleration_requested=acceleration.requested,
            acceleration_effective=acceleration.effective,
            acceleration_note=acceleration.reason,
            release_after_run=bool(release_after_run),
            model_revision=str(manifest["modelRevision"]),
            low_vram=low_vram,
        )
        verification = "SHA-256 已校验" if report.hashes_verified else "文件大小已校验"
        info = (
            f"IndexTTS 2.5 | {model_dir} | device={resolved_device} | "
            f"precision={'bfloat16' if handle.use_bf16 else 'float32'} | {verification} | "
            f"model revision={manifest['modelRevision'][:12]} | "
            f"accel={acceleration.effective}（{acceleration.reason}）"
            + (" | 低显存自动适配" if low_vram else "")
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
            io.Float.Input(name, display_name=label, default=0.0, min=0.0, max=1.0, step=0.01)
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
                                io.Audio.Input("emotion_audio", display_name="情感参考音频"),
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
            names = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
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
            populated = ", ".join(f"{name}={value:.2f}" for name, value in zip(names, values) if value > 0)
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
                io.Float.Input("temperature", display_name="temperature", default=0.8, min=0.1, max=2.0, step=0.05, advanced=True),
                io.Float.Input("top_p", display_name="top_p", default=0.8, min=0.05, max=1.0, step=0.01, advanced=True),
                io.Int.Input("top_k", display_name="top_k", default=30, min=0, max=200, step=1, advanced=True),
                io.Int.Input("num_beams", display_name="num_beams", default=3, min=1, max=10, step=1, advanced=True),
                io.Float.Input(
                    "repetition_penalty",
                    display_name="repetition_penalty",
                    default=10.0,
                    min=0.1,
                    max=20.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Float.Input("length_penalty", display_name="length_penalty", default=0.0, min=-2.0, max=2.0, step=0.05, advanced=True),
                io.Int.Input("max_mel_tokens", display_name="最大语音 token", default=1500, min=256, max=4096, step=16, advanced=True),
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
                io.Int.Input("segment_silence_ms", display_name="段间静音（毫秒）", default=200, min=0, max=3000, step=10),
                io.Combo.Input(
                    "pause_preset",
                    display_name="标点停顿预设",
                    options=["off", "natural", "narration", "dialogue", "custom"],
                    default="off",
                    tooltip="显式 <pause=0.5> 或 <pause=500ms> 在任意预设下都有效。",
                ),
                io.Int.Input("comma_pause_ms", display_name="逗号停顿（毫秒）", default=100, min=0, max=5000, step=10, advanced=True),
                io.Int.Input("sentence_pause_ms", display_name="句末停顿（毫秒）", default=300, min=0, max=5000, step=10, advanced=True),
                io.Int.Input("paragraph_pause_ms", display_name="段落停顿（毫秒）", default=600, min=0, max=5000, step=10, advanced=True),
                io.Boolean.Input("text_normalization", display_name="文本归一化", default=True),
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
            f"pause={config.pause_preset} | internal_silence={config.segment_silence_ms}ms"
        )
        return io.NodeOutput(config, info)


class T8IndexTTS25TextPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_TextPreview",
            display_name="IndexTTS 2.5 分段与停顿预览 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["text segment preview", "长文本分段", "pause preview", "停顿预览"],
            description="仅加载官方轻量 Token 词表，预览模型输入前的 Token 分段、外部停顿和 GPT 加速风险；文本原样透传。",
            inputs=[
                ModelType.Input("model", display_name="IndexTTS 2.5 模型"),
                io.String.Input("text", display_name="待预览文本", multiline=True, dynamic_prompts=True),
                io.Combo.Input("language", display_name="语言", options=["ZH", "EN", "JA", "ES", "AR"], default="ZH"),
                SamplingType.Input("sampling", display_name="采样/分段设置", optional=True),
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
            search_aliases=["IndexTTS pronunciation", "多音字", "拼音", "CMU phoneme", "日语假名"],
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
        result = process_pronunciation_text(text, language, entries, strict=bool(strict))
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
                io.Combo.Input("language", display_name="默认语言", options=["ZH", "EN", "JA", "ES", "AR"], default="ZH"),
                EmotionType.Input("emotion", display_name="默认情感", optional=True),
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
        profile = VoiceProfile(str(role_name).strip(), speaker_audio, str(language).upper(), emotion)
        return io.NodeOutput(profile, f"角色={profile.name} | language={profile.language} | " + ("含默认情感" if emotion else "情感跟随音色"))


class T8IndexTTS25RoleLibrary(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_RoleLibrary",
            display_name="IndexTTS 2.5 角色音色库 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["多角色", "role library", "音色库"],
            description="自动增长输入，可连接 1–16 个角色音色；同名角色会被拒绝。",
            inputs=[
                io.Autogrow.Input(
                    "voices",
                    display_name="角色音色",
                    template=io.Autogrow.TemplatePrefix(VoiceType.Input("voice"), prefix="voice_", min=1, max=16),
                )
            ],
            outputs=[
                RoleLibraryType.Output("role_library", display_name="角色音色库"),
                io.String.Output("role_info", display_name="角色列表"),
            ],
        )

    @staticmethod
    def _profiles(value) -> list[VoiceProfile]:
        if isinstance(value, VoiceProfile):
            return [value]
        if isinstance(value, dict):
            result: list[VoiceProfile] = []
            for nested in value.values():
                result.extend(T8IndexTTS25RoleLibrary._profiles(nested))
            return result
        if isinstance(value, (list, tuple)):
            result = []
            for nested in value:
                result.extend(T8IndexTTS25RoleLibrary._profiles(nested))
            return result
        return []

    @classmethod
    def execute(cls, voices: dict) -> io.NodeOutput:
        profiles = cls._profiles(voices)
        if not profiles:
            raise ValueError("角色音色库至少需要连接一个角色音色。")
        result: dict[str, VoiceProfile] = {}
        for profile in profiles:
            if profile.name in result:
                raise ValueError(f"角色名称重复：{profile.name}")
            result[profile.name] = profile
        library = RoleLibrary(result)
        return io.NodeOutput(library, "角色音色：" + "、".join(result))


class T8IndexTTS25DialogueScript(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_DialogueScript",
            display_name="IndexTTS 2.5 批量台词 / SRT · T8star-Aix",
            category=CATEGORY,
            search_aliases=["SRT", "字幕配音", "批量台词", "dialogue script"],
            description=(
                "批量格式：角色|台词|语言|时长系数；SRT 支持 [角色] 台词 或 角色: 台词。"
            ),
            inputs=[
                io.Combo.Input("script_type", display_name="脚本格式", options=["batch", "srt"], default="batch"),
                io.String.Input(
                    "script",
                    display_name="批量台词或 SRT",
                    multiline=True,
                    dynamic_prompts=False,
                    default="旁白|欢迎使用多角色批量配音。|ZH|1.0\n角色A|这是第二句。|ZH|0.9",
                    tooltip=(
                        "支持 角色|台词|语言|时长系数、JSON 数组或 SRT。"
                        "此输入已关闭 ComfyUI 动态提示词解析，JSON 的大括号不会被改写。"
                    ),
                ),
                io.String.Input("default_role", display_name="默认角色", default="旁白"),
                io.Combo.Input("default_language", display_name="默认语言", options=["ZH", "EN", "JA", "ES", "AR"], default="ZH"),
            ],
            outputs=[
                DialogueScriptType.Output("dialogue_script", display_name="台词脚本"),
                io.String.Output("script_preview", display_name="解析预览 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(cls, script_type: str, script: str, default_role: str, default_language: str, **kwargs) -> bool | str:
        try:
            (parse_srt if script_type == "srt" else parse_batch_script)(script, default_role, default_language)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def execute(cls, script_type: str, script: str, default_role: str, default_language: str) -> io.NodeOutput:
        lines = (parse_srt if script_type == "srt" else parse_batch_script)(script, default_role, default_language)
        payload = [line.to_dict() for line in lines]
        return io.NodeOutput(DialogueScript(lines, script_type), json.dumps(payload, ensure_ascii=False, indent=2))


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
                io.Int.Input("seed", display_name="起始 seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Combo.Input("timeline_policy", display_name="时间冲突策略", options=["shift", "overlay"], default="shift", tooltip="shift 顺延避免重叠；overlay 保留 SRT 起点并安全混音。"),
                io.Boolean.Input("fit_srt_slots", display_name="二次推理适配字幕槽位", default=False, tooltip="仅 SRT 有效；不保证逐帧精确，超时会写入报告。"),
                io.Combo.Input(
                    "slot_duration_mode",
                    display_name="字幕槽位收尾模式",
                    options=["natural", "pad", "exact"],
                    default="natural",
                    tooltip="natural 只二次适配；pad 不足补静音且保留超长；exact 会补静音或强制裁剪。",
                ),
                io.Int.Input("fit_tolerance_ms", display_name="允许时长误差（毫秒）", default=180, min=0, max=2000, step=10, advanced=True),
                io.Int.Input("batch_gap_ms", display_name="批量句间静音（毫秒）", default=200, min=0, max=5000, step=10),
                io.Combo.Input("postprocess_preset", display_name="合并音频后处理", options=list(POSTPROCESS_PRESETS), default="off"),
                io.Float.Input("postprocess_strength", display_name="后处理强度", default=1.0, min=0.0, max=1.0, step=0.05, advanced=True),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="合并音频"),
                io.Audio.Output("line_audios", display_name="逐句音频", is_output_list=True),
                io.String.Output("generation_report", display_name="生成报告 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(cls, role_library: RoleLibrary, dialogue_script: DialogueScript, **kwargs) -> bool | str:
        if not isinstance(role_library, RoleLibrary) or not isinstance(dialogue_script, DialogueScript):
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
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        missing = missing_roles(dialogue_script.lines, role_library.profiles)
        if missing:
            raise ValueError("以下角色没有连接音色：" + "、".join(missing))
        work_handle = replace(model, release_after_run=False)
        clips: list[dict] = []
        line_reports: list[dict] = []
        sample_rate: int | None = None
        try:
            for offset, line in enumerate(dialogue_script.lines):
                profile = role_library.profiles[line.role]
                language = line.language or profile.language
                audio, status = run_inference(
                    work_handle,
                    profile.speaker_audio,
                    line.text,
                    language,
                    line.duration_factor,
                    int(seed) + offset,
                    emotion=profile.emotion,
                    sampling=sampling,
                )
                actual_ms = audio["waveform"].shape[-1] * 1000 / audio["sample_rate"]
                used_factor = line.duration_factor
                regenerated = False
                duration_adjustment = {"mode": "off", "action": "unchanged"}
                if fit_srt_slots and line.slot_ms and abs(actual_ms - line.slot_ms) > int(fit_tolerance_ms):
                    fitted = fit_duration_factor(used_factor, actual_ms, line.slot_ms)
                    if abs(fitted - used_factor) >= 0.02:
                        audio, status = run_inference(
                            work_handle,
                            profile.speaker_audio,
                            line.text,
                            language,
                            fitted,
                            int(seed) + offset,
                            emotion=profile.emotion,
                            sampling=sampling,
                        )
                        used_factor = fitted
                        actual_ms = audio["waveform"].shape[-1] * 1000 / audio["sample_rate"]
                        regenerated = True
                if fit_srt_slots and line.slot_ms and slot_duration_mode in {"pad", "exact"}:
                    audio, duration_adjustment = apply_duration_policy(
                        audio, line.slot_ms / 1000.0, slot_duration_mode
                    )
                    actual_ms = audio_duration_ms(audio)
                if sample_rate is None:
                    sample_rate = int(audio["sample_rate"])
                elif sample_rate != int(audio["sample_rate"]):
                    raise RuntimeError("逐句输出采样率不一致，无法合并。")
                clips.append(audio)
                line_reports.append({
                    **line.to_dict(),
                    "actual_duration_ms": round(actual_ms),
                    "used_duration_factor": round(used_factor, 4),
                    "regenerated_for_slot": regenerated,
                    "duration_adjustment": duration_adjustment,
                    "status": status,
                })
        finally:
            if model.release_after_run:
                MODEL_CACHE.release(work_handle)
        assert sample_rate is not None
        waveform, placements = compose_timeline(
            [audio["waveform"] for audio in clips],
            dialogue_script.lines,
            sample_rate,
            timeline_policy,
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
        report = {
            "script_type": dialogue_script.script_type,
            "timeline_policy": timeline_policy,
            "fit_srt_slots": bool(fit_srt_slots),
            "slot_duration_mode": slot_duration_mode,
            "postprocess": postprocess_report,
            "sample_rate": sample_rate,
            "duration_ms": round(waveform.shape[-1] * 1000 / sample_rate),
            "lines": line_reports,
        }
        return io.NodeOutput(
            {"waveform": waveform, "sample_rate": sample_rate},
            clips,
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
            inputs=[io.Combo.Input("device", display_name="检查设备", options=_device_options(), default="auto")],
            outputs=[io.String.Output("environment_report", display_name="环境报告 JSON")],
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
        return io.NodeOutput(json.dumps({"device": resolved, "capabilities": capabilities, "modes": modes}, ensure_ascii=False, indent=2))


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
                    display_name="时长系数（越小越快）",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="官方语速适配：0.5 更快、1.0 原速、2.0 更慢。",
                ),
                io.Combo.Input(
                    "target_duration_mode",
                    display_name="目标时长模式",
                    options=list(DURATION_MODES),
                    default="off",
                    tooltip="natural 二次推理；pad 不足补静音且不裁剪；exact 会补静音或强制裁剪。",
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
                io.Float.Input("postprocess_strength", display_name="后处理强度", default=1.0, min=0.0, max=1.0, step=0.05, advanced=True),
                io.Int.Input(
                    "seed",
                    display_name="seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
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
        if target_duration_mode != "off" and not 0.1 <= float(target_duration_seconds) <= 3600:
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
        emotion: EmotionConfig | None = None,
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        audio, status = run_inference(
            handle=model,
            speaker_audio=speaker_audio,
            text=text,
            language=language,
            duration_factor=duration_factor,
            seed=seed,
            emotion=emotion,
            sampling=sampling,
        )
        duration_report: dict = {"mode": "off", "action": "unchanged"}
        used_factor = float(duration_factor)
        if target_duration_mode != "off" and float(target_duration_seconds) > 0:
            actual_ms = audio_duration_ms(audio)
            fitted = fit_duration_factor(
                used_factor, actual_ms, float(target_duration_seconds) * 1000.0
            )
            if abs(fitted - used_factor) >= 0.02:
                audio, status = run_inference(
                    handle=model,
                    speaker_audio=speaker_audio,
                    text=text,
                    language=language,
                    duration_factor=fitted,
                    seed=seed,
                    emotion=emotion,
                    sampling=sampling,
                )
                used_factor = fitted
            if target_duration_mode in {"pad", "exact"}:
                audio, duration_report = apply_duration_policy(
                    audio, target_duration_seconds, target_duration_mode
                )
            else:
                duration_report = {
                    "mode": "natural",
                    "target_ms": round(float(target_duration_seconds) * 1000),
                    "final_ms": round(audio_duration_ms(audio)),
                    "action": "regenerated" if used_factor != float(duration_factor) else "unchanged",
                }
            status += f" | target={float(target_duration_seconds):.2f}s/{target_duration_mode} | fitted_factor={used_factor:.3f}"
        audio, postprocess_report = postprocess_audio(
            audio, postprocess_preset, postprocess_strength
        )
        status += " | duration=" + json.dumps(duration_report, ensure_ascii=False, separators=(",", ":"))
        status += " | post=" + json.dumps(postprocess_report, ensure_ascii=False, separators=(",", ":"))
        return io.NodeOutput(audio, status)


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
                io.Combo.Input("preset", display_name="预设", options=list(POSTPROCESS_PRESETS), default="voice_clarity"),
                io.Float.Input("strength", display_name="强度", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Float.Input("target_peak_db", display_name="目标峰值（dBFS）", default=-1.0, min=-12.0, max=-0.1, step=0.1, advanced=True),
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
            T8IndexTTS25DialogueScript,
            T8IndexTTS25DialogueGenerate,
            T8IndexTTS25AudioPostProcess,
            T8IndexTTS25Environment,
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
    "T8IndexTTS25DialogueScript",
    "T8IndexTTS25DialogueGenerate",
    "T8IndexTTS25AudioPostProcess",
    "T8IndexTTS25Environment",
    "T8IndexTTS25Extension",
    "comfy_entrypoint",
]
