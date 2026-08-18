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
    release_after_run: bool = False
    model_revision: str = ""
    low_vram: bool = False

    @property
    def cache_key(self) -> tuple[str, str, bool, bool, str, bool]:
        return (
            str(self.model_dir.resolve()),
            self.device,
            self.use_bf16,
            self.use_cuda_kernel,
            self.model_revision,
            self.low_vram,
        )


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
    max_text_tokens_per_segment: int = 120
    segment_silence_ms: int = 200
    text_normalization: bool = True

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
        }


DEFAULT_SAMPLING = SamplingConfig()
DEFAULT_EMOTION = EmotionConfig()
