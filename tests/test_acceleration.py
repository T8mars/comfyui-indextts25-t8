import ast
from pathlib import Path
import importlib.util

import torch

from runtime.acceleration import (
    describe_acceleration_failure,
    probe_acceleration,
    recommend_runtime_config,
    resolve_acceleration,
)


def test_waves_per_eu_error_is_reported_as_a_torch_triton_mismatch():
    message = describe_acceleration_failure(
        RuntimeError("Keyword argument waves_per_eu was specified but unrecognised")
    )
    assert "仅适用于 AMD" in message
    assert "GPT/torch.compile 加速与本机环境不兼容" in message


def caps(cuda=True, *, nvcc=False, cxx=False, **modules):
    return {
        "cuda": cuda,
        "bf16": cuda,
        "fp16": cuda,
        "gpu": {"total_vram_gb": 8.0 if cuda else 0.0},
        "modules": {"deepspeed": False, "flash_attn": False, "triton": False, "ninja": False} | modules,
        "tools": {"nvcc": nvcc, "cl": False, "cxx": cxx},
    }


def test_missing_optional_dependencies_are_safe_fallbacks():
    for mode in ("bigvgan_cuda", "torch_compile", "gpt_accel", "deepspeed"):
        result = resolve_acceleration(mode, "cuda:0", caps())
        assert result.effective == "off"
        assert not result.available


def test_auto_safe_never_enables_deepspeed():
    result = resolve_acceleration("auto_safe", "cuda:0", caps(deepspeed=True))
    assert not result.use_deepspeed


def test_bigvgan_needs_cuda_and_cpp_compilers():
    assert resolve_acceleration(
        "bigvgan_cuda", "cuda:0", caps(ninja=True, nvcc=True)
    ).effective == "off"
    assert resolve_acceleration(
        "bigvgan_cuda", "cuda:0", caps(ninja=True, nvcc=True, cxx=True)
    ).use_cuda_kernel


def test_preflight_recommends_fp16_and_cpu_reference_for_old_low_vram_gpu():
    report = caps(cuda=True)
    report["bf16"] = False
    recommendation = recommend_runtime_config(report)
    assert recommendation["precision"] == "float16"
    assert recommendation["reference_device"] == "cpu"
    assert recommendation["acceleration_mode"] == "off"


def test_preflight_only_recommends_auto_safe_when_toolchain_is_ready():
    recommendation = recommend_runtime_config(
        caps(cuda=True, ninja=True, nvcc=True, cxx=True)
    )
    assert recommendation["precision"] == "bfloat16"
    assert recommendation["acceleration_mode"] == "auto_safe"


def test_preflight_includes_dependency_versions_without_importing_models():
    report = probe_acceleration("cpu")
    assert report["cuda"] is False
    assert report["versions"]["torch"] == str(torch.__version__)
    assert set(report["versions"]) == {
        "torch",
        "cuda_runtime",
        "torchaudio",
        "torchcodec",
        "deepspeed",
        "flash_attn",
        "triton",
        "ninja",
    }
    torchcodec = report["runtime_checks"]["torchcodec"]
    if torchcodec["required"] and importlib.util.find_spec("torchcodec") is None:
        assert torchcodec["ready"] is False
    else:
        assert torchcodec["ready"] is True, torchcodec["reason"]


def test_gpt_acceleration_never_uses_implicit_default_cuda_device():
    root = Path(__file__).resolve().parents[1]
    files = (
        root / "indextts" / "gpt" / "model_v2.py",
        root / "indextts" / "gpt" / "model_v2_5.py",
        root / "indextts" / "accel" / "accel_engine.py",
        root / "indextts" / "accel" / "kv_manager.py",
    )
    violations = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
                violations.append((path.name, node.lineno, "implicit .cuda()"))
            for keyword in node.keywords:
                if (
                    keyword.arg == "device"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "cuda"
                ):
                    violations.append((path.name, node.lineno, "device='cuda'"))
    assert violations == []
