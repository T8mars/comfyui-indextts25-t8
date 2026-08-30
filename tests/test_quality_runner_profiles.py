from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_multilingual_quality_regression.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("indextts25_quality_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCuda:
    def __init__(self, total_gib: float):
        self.total_bytes = int(total_gib * 1024**3)
        self.calls = []

    def get_device_properties(self, _device):
        return type("Properties", (), {"total_memory": self.total_bytes})()

    def set_per_process_memory_fraction(self, fraction, device):
        self.calls.append((fraction, device))


class _FakeTorch:
    def __init__(self, total_gib: float):
        self.cuda = _FakeCuda(total_gib)

    @staticmethod
    def device(value):
        return value


def test_formal_vram_profiles_enforce_and_report_the_budget():
    runner = _load_runner()
    torch_module = _FakeTorch(24)
    limited = runner._configure_vram_profile(torch_module, "8gb", "cuda:0")
    assert limited["name"] == "8gb"
    assert limited["budget_bytes"] == 8 * 1024**3
    assert limited["simulated"] is True
    assert torch_module.cuda.calls[0][0] == pytest.approx(1 / 3)

    native_24 = runner._configure_vram_profile(torch_module, "24gb", "cuda:0")
    assert native_24["name"] == "24gb"
    assert native_24["budget_bytes"] == 24 * 1024**3
    assert native_24["simulated"] is False


def test_formal_vram_profile_rejects_inadequate_hardware():
    runner = _load_runner()
    with pytest.raises(SystemExit, match="at least 22 GiB"):
        runner._configure_vram_profile(_FakeTorch(16), "24gb", "cuda:0")
    with pytest.raises(SystemExit, match="requires CUDA"):
        runner._configure_vram_profile(_FakeTorch(24), "8gb", "cpu")


def test_arabic_asr_model_override_is_validated():
    runner = _load_runner()
    assert runner._parse_asr_model_overrides(["AR=small", "EN=base"]) == {
        "AR": "small",
        "EN": "base",
    }
    with pytest.raises(SystemExit, match="expected LANG=MODEL"):
        runner._parse_asr_model_overrides(["AR=unknown"])
