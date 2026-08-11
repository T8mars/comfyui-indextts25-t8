from __future__ import annotations

import logging
from pathlib import Path

import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .runtime.inference_adapter import run_inference
from .runtime.types import EmotionConfig, ModelHandle, SamplingConfig
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
                io.Boolean.Input(
                    "use_cuda_kernel",
                    display_name="BigVGAN CUDA 融合核",
                    default=False,
                    advanced=True,
                    tooltip="首次使用可能编译扩展；不确定时保持关闭。",
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
        use_cuda_kernel: bool,
        release_after_run: bool,
        verify_hashes: bool,
        custom_model_path: str = "",
    ) -> io.NodeOutput:
        model_dir = resolve_model(model_name, custom_model_path)
        report = validate_model_dir(model_dir, verify_hashes=verify_hashes)
        report.require_valid()
        resolved_device = _resolve_device(device)
        manifest = load_manifest()
        handle = ModelHandle(
            model_dir=model_dir,
            device=resolved_device,
            use_bf16=_use_bf16(precision, resolved_device),
            use_cuda_kernel=bool(use_cuda_kernel and resolved_device.startswith("cuda")),
            release_after_run=bool(release_after_run),
            model_revision=str(manifest["modelRevision"]),
        )
        verification = "SHA-256 已校验" if report.hashes_verified else "文件大小已校验"
        info = (
            f"IndexTTS 2.5 | {model_dir} | device={resolved_device} | "
            f"precision={'bfloat16' if handle.use_bf16 else 'float32'} | {verification} | "
            f"model revision={manifest['modelRevision'][:12]}"
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
            description="集中配置确定性、采样、长文本分段、停顿和文本归一化参数。",
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
                io.Int.Input(
                    "max_text_tokens_per_segment",
                    display_name="每段最大文本 token",
                    default=120,
                    min=20,
                    max=300,
                    step=5,
                ),
                io.Int.Input("segment_silence_ms", display_name="段间静音（毫秒）", default=200, min=0, max=3000, step=10),
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
        max_text_tokens_per_segment: int,
        segment_silence_ms: int,
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
            max_text_tokens_per_segment=int(max_text_tokens_per_segment),
            segment_silence_ms=int(segment_silence_ms),
            text_normalization=bool(text_normalization),
        )
        mode = "随机采样" if config.do_sample else "确定性/束搜索"
        info = (
            f"{mode} | beams={config.num_beams} | max_mel={config.max_mel_tokens} | "
            f"segment_tokens={config.max_text_tokens_per_segment} | silence={config.segment_silence_ms}ms"
        )
        return io.NodeOutput(config, info)


class T8IndexTTS25Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_IndexTTS25_Generate",
            display_name="IndexTTS 2.5 语音生成 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["IndexTTS TTS", "voice clone", "语音克隆", "T8star-Aix"],
            description="使用正式 IndexTTS 2.5 模型进行多语种零样本音色克隆，并输出标准 ComfyUI AUDIO。",
            inputs=[
                ModelType.Input("model", display_name="IndexTTS 2.5 模型"),
                io.Audio.Input("speaker_audio", display_name="音色参考音频"),
                io.String.Input(
                    "text",
                    display_name="待合成文本",
                    multiline=True,
                    default="欢迎使用 IndexTTS 2.5，来自 B 站：T8star-Aix。",
                    dynamic_prompts=True,
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
    def validate_inputs(cls, text: str, duration_factor: float, **kwargs) -> bool | str:
        if not str(text).strip():
            return "待合成文本不能为空。"
        if not 0.5 <= float(duration_factor) <= 2.0:
            return "时长系数必须在 0.5 到 2.0 之间。"
        return True

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        speaker_audio: dict,
        text: str,
        language: str,
        duration_factor: float,
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
        return io.NodeOutput(audio, status)


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
            T8IndexTTS25Generate,
        ]


async def comfy_entrypoint() -> T8IndexTTS25Extension:
    return T8IndexTTS25Extension()


__all__ = [
    "T8IndexTTS25ModelLoader",
    "T8IndexTTS25EmotionControl",
    "T8IndexTTS25SamplingConfig",
    "T8IndexTTS25Generate",
    "T8IndexTTS25Extension",
    "comfy_entrypoint",
]

