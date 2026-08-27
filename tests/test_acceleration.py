import torch

from runtime.acceleration import (
    probe_acceleration,
    recommend_runtime_config,
    resolve_acceleration,
)


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
        "deepspeed",
        "flash_attn",
        "triton",
        "ninja",
    }
