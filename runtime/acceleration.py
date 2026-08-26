"""Optional acceleration probe; this module has no optional dependency imports."""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass

import torch


MODES = ("off", "auto_safe", "bigvgan_cuda", "torch_compile", "gpt_accel", "deepspeed")


@dataclass(frozen=True, slots=True)
class AccelerationSelection:
    requested: str
    effective: str
    use_cuda_kernel: bool = False
    use_torch_compile: bool = False
    use_accel: bool = False
    use_deepspeed: bool = False
    available: bool = True
    reason: str = ""


def _module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def probe_acceleration(device: str) -> dict:
    cuda = bool(torch.cuda.is_available() and str(device).startswith("cuda"))
    try:
        bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    except Exception:
        bf16 = False
    return {
        "cuda": cuda,
        "bf16": bf16,
        "modules": {"deepspeed": _module("deepspeed"), "flash_attn": _module("flash_attn"), "triton": _module("triton"), "ninja": _module("ninja") or shutil.which("ninja") is not None},
        "tools": {
            "nvcc": shutil.which("nvcc") is not None,
            "cl": shutil.which("cl") is not None,
            "cxx": any(shutil.which(name) is not None for name in ("c++", "g++", "clang++")),
        },
    }


def resolve_acceleration(mode: str, device: str, capabilities: dict | None = None) -> AccelerationSelection:
    requested = str(mode or "off").lower()
    if requested not in MODES:
        raise ValueError(f"未知加速模式：{requested}")
    caps = capabilities or probe_acceleration(device)
    modules, tools = caps["modules"], caps["tools"]
    if requested == "off":
        return AccelerationSelection(requested, "off", reason="可选加速已关闭")
    if not caps["cuda"]:
        return AccelerationSelection(requested, "off", available=False, reason="没有 CUDA，已回退普通模式")
    cuda_kernel_ready = bool(
        modules["ninja"] and tools["nvcc"] and (tools.get("cl", False) or tools.get("cxx", False))
    )
    if requested == "auto_safe":
        ready = cuda_kernel_ready
        return AccelerationSelection(requested, "bigvgan_cuda" if ready else "off", use_cuda_kernel=ready, available=ready, reason=("自动启用 BigVGAN CUDA 融合核" if ready else "缺少编译工具链，使用普通模式"))
    if requested == "bigvgan_cuda":
        ready = cuda_kernel_ready
        return AccelerationSelection(requested, requested if ready else "off", use_cuda_kernel=ready, available=ready, reason=("依赖可用" if ready else "缺少 ninja 或 CUDA/C++ 编译工具链，已回退"))
    if requested == "torch_compile":
        ready = modules["triton"]
        return AccelerationSelection(requested, requested if ready else "off", use_torch_compile=ready, available=ready, reason=("Triton 可用；首次生成会编译" if ready else "未安装可选 Triton，已回退"))
    if requested == "gpt_accel":
        ready = modules["flash_attn"] and modules["triton"]
        return AccelerationSelection(requested, requested if ready else "off", use_accel=ready, available=ready, reason=("FlashAttention/Triton 可用" if ready else "未安装可选 FlashAttention/Triton，已回退"))
    ready = modules["deepspeed"]
    ready_reason = "DeepSpeed 可用"
    if os.name == "nt":
        ready_reason += "；Windows BF16 请求将使用 FP16 workspace"
    return AccelerationSelection(requested, requested if ready else "off", use_deepspeed=ready, available=ready, reason=(ready_reason if ready else "未安装可选 DeepSpeed，已回退；它不是必需依赖"))


__all__ = ["AccelerationSelection", "MODES", "probe_acceleration", "resolve_acceleration"]
