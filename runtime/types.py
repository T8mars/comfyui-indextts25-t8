from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelHandle:
    """A lightweight, workflow-local reference to a lazily loaded model."""

    model_dir: Path
    device: str
    use_bf16: bool
    use_cuda_kernel: bool = False
    use_torch_compile: bool = False
    use_accel: bool = False
    use_deepspeed: bool = False
    acceleration_requested: str = "off"
    acceleration_effective: str = "off"
    acceleration_note: str = ""
    release_after_run: bool = False
    recycle_after_runs: int = 0
    model_revision: str = ""
    model_fingerprint: str = ""
    low_vram: bool = False

    @property
    def cache_key(self) -> tuple:
        return (
            str(self.model_dir.resolve()),
            self.device,
            self.use_bf16,
            self.use_cuda_kernel,
            self.use_torch_compile,
            self.use_accel,
            self.use_deepspeed,
            self.model_revision,
            self.model_fingerprint,
            self.low_vram,
        )


@dataclass(slots=True)
class VoiceProfile:
    """Workflow-local named voice and its optional default emotion."""

    name: str
    speaker_audio: dict[str, Any]
    language: str = "ZH"
    emotion: "EmotionConfig | None" = None


@dataclass(slots=True)
class RoleLibrary:
    profiles: dict[str, VoiceProfile]


@dataclass(slots=True)
class DialogueScript:
    lines: list[Any]
    script_type: str = "batch"


@dataclass(slots=True)
class EmotionConfig:
    """Emotion guidance passed between the emotion and generation nodes."""

    mode: str = "speaker"
    reference_audio: dict[str, Any] | None = None
    vector: tuple[float, ...] | None = None
    text: str | None = None
    strength: float = 1.0
    use_random: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Generation controls supported by the IndexTTS 2.5 inference API."""

    do_sample: bool = False
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    num_beams: int = 3
    repetition_penalty: float = 10.0
    length_penalty: float = 0.0
    max_mel_tokens: int = 1500
    diffusion_steps: int = 25
    inference_cfg_rate: float = 0.7
    cfm_temperature: float = 1.0
    segmentation_mode: str = "auto"
    max_text_tokens_per_segment: int = 120
    segment_silence_ms: int = 200
    pause_preset: str = "off"
    comma_pause_ms: int = 0
    sentence_pause_ms: int = 0
    paragraph_pause_ms: int = 0
    text_normalization: bool = True

    def __post_init__(self) -> None:
        if not 1 <= int(self.diffusion_steps) <= 200:
            raise ValueError("diffusion_steps 必须在 1 到 200 之间。")
        if not 0 <= float(self.inference_cfg_rate) <= 3:
            raise ValueError("inference_cfg_rate 必须在 0 到 3 之间。")
        if not 0.05 <= float(self.cfm_temperature) <= 2:
            raise ValueError("cfm_temperature 必须在 0.05 到 2 之间。")

    def effective_segment_tokens(self, language: str) -> int:
        from .text_planner import effective_segment_limit

        return effective_segment_limit(
            language, self.segmentation_mode, self.max_text_tokens_per_segment
        )

    def generation_kwargs(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "max_mel_tokens": self.max_mel_tokens,
            "diffusion_steps": self.diffusion_steps,
            "inference_cfg_rate": self.inference_cfg_rate,
            "cfm_temperature": self.cfm_temperature,
        }


DEFAULT_SAMPLING = SamplingConfig()
DEFAULT_EMOTION = EmotionConfig()
