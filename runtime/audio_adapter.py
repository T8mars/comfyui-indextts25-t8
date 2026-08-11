from __future__ import annotations

from typing import Any

import numpy as np
import torch


INDEXTTS_SAMPLE_RATE = 22_050


def validate_comfy_audio(audio: dict[str, Any], *, name: str = "audio") -> torch.Tensor:
    if not isinstance(audio, dict):
        raise TypeError(f"{name} 必须是 ComfyUI AUDIO 字典。")
    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError(f"{name} 缺少 waveform 或 sample_rate。")

    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"{name}.waveform 必须是 torch.Tensor。")
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3:
        raise ValueError(f"{name}.waveform 形状必须为 [B,C,T]，实际为 {tuple(waveform.shape)}。")
    if waveform.shape[0] != 1:
        raise ValueError(f"当前版本只支持单条参考音频，batch 必须为 1，实际为 {waveform.shape[0]}。")
    if waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError(f"{name} 是空音频。")
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError(f"{name}.sample_rate 必须大于 0。")

    waveform = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(waveform).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf。")
    return waveform


def indextts_result_to_audio(result: Any) -> dict[str, Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"IndexTTS 返回了无法识别的结果：{type(result).__name__}")
    sample_rate, raw = result
    if int(sample_rate) != INDEXTTS_SAMPLE_RATE:
        raise RuntimeError(f"IndexTTS 返回了异常采样率：{sample_rate}")

    if isinstance(raw, torch.Tensor):
        tensor = raw.detach().to(device="cpu")
    else:
        array = np.asarray(raw)
        if array.size == 0:
            raise RuntimeError("IndexTTS 返回了空音频。")
        tensor = torch.from_numpy(array.copy())

    if tensor.ndim == 2:
        if tensor.shape[-1] == 1:
            tensor = tensor[:, 0]
        elif tensor.shape[0] == 1:
            tensor = tensor[0]
        else:
            tensor = tensor.mean(dim=-1)
    elif tensor.ndim != 1:
        tensor = tensor.reshape(-1)

    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        tensor = tensor.to(torch.float32) / 32768.0
    else:
        tensor = tensor.to(torch.float32)
        peak = float(tensor.abs().max()) if tensor.numel() else 0.0
        if peak > 2.0:
            tensor = tensor / 32768.0

    tensor = tensor.clamp(-1.0, 1.0).contiguous().view(1, 1, -1)
    if tensor.shape[-1] == 0 or not torch.isfinite(tensor).all():
        raise RuntimeError("IndexTTS 返回的音频为空或包含非法数值。")
    return {"waveform": tensor, "sample_rate": INDEXTTS_SAMPLE_RATE}

