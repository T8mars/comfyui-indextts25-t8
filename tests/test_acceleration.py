from runtime.acceleration import resolve_acceleration


def caps(cuda=True, *, nvcc=False, cxx=False, **modules):
    return {
        "cuda": cuda,
        "bf16": cuda,
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
