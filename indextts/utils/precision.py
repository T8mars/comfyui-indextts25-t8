"""Precision selection helpers for IndexTTS 2.5 inference."""

from __future__ import annotations


def resolve_gpt_precision(*, use_fp16=False, use_bf16=False, device=None):
    """Return ``fp16``, ``bf16`` or ``None`` after device validation."""

    if use_fp16 and use_bf16:
        raise ValueError("use_fp16 and use_bf16 are mutually exclusive")
    device_type = str(device).split(":", 1)[0] if device is not None else None
    if device_type in {"cpu", "mps"}:
        return None
    if use_fp16:
        return "fp16"
    if use_bf16:
        return "bf16"
    return None


def cuda_supports_native_bf16(torch_module, device_index=None) -> bool:
    """Return whether CUDA offers native BF16, excluding emulation."""

    if not torch_module.cuda.is_available():
        return False
    probe = torch_module.cuda.is_bf16_supported
    try:
        if device_index is None:
            return bool(probe(including_emulation=False))
        with torch_module.cuda.device(device_index):
            return bool(probe(including_emulation=False))
    except TypeError:
        # PyTorch versions before ``including_emulation`` use the hardware probe.
        if device_index is None:
            return bool(probe())
        with torch_module.cuda.device(device_index):
            return bool(probe())


def select_half_precision(*, enabled, cuda_available, native_bf16_supported):
    """Prefer native BF16 and otherwise use FP16 when half precision is enabled."""

    if not enabled or not cuda_available:
        return None
    return "bf16" if native_bf16_supported else "fp16"


__all__ = [
    "cuda_supports_native_bf16",
    "resolve_gpt_precision",
    "select_half_precision",
]
