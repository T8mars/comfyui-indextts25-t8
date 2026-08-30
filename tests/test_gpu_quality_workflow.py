from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_quality_workflow_isolated_from_normal_ci():
    source = (ROOT / ".github/workflows/gpu_quality_regression.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch:" in source
    assert "schedule:" in source
    assert "pull_request:" not in source
    assert re.search(r"^\s+push:\s*$", source, re.MULTILINE) is None
    assert "runs-on: [self-hosted, Windows, X64, gpu, indextts25]" in source
    assert "--asr-backend openai_whisper" in source
    assert "--asr-model base" in source
    assert "--asr-model-for AR=small" in source
    assert "--vram-profile ${{ matrix.profile }}" in source
    assert "max-parallel: 1" in source
    assert "quality_baselines/openai-whisper-mixed-8gb-gpu.json" in source
    assert "quality_baselines/openai-whisper-mixed-24gb-gpu.json" in source
    assert "build_quality_trend_report.py" in source
    assert "gpu-quality-trend" in source


def test_whisper_version_and_gpu_baselines_are_pinned():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'openai-whisper==20250625' in pyproject
    for profile in ("8gb", "24gb"):
        baseline_path = ROOT / f"quality_baselines/openai-whisper-mixed-{profile}-gpu.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert baseline["baseline_kind"] == "indextts25-multilingual-gpu"
        assert baseline["vram_profile"]["name"] == profile
        assert baseline["asr_runtime"]["package_version"] == "20250625"
        assert baseline["asr_runtime"]["model"] == "base"
        assert baseline["asr_runtime"]["model_by_language"]["AR"] == "small"
        assert [case["language"] for case in baseline["cases"]] == [
            "ZH", "EN", "JA", "ES", "AR"
        ]
        assert [case["asr"]["model"] for case in baseline["cases"]] == [
            "base", "base", "base", "base", "small"
        ]
        assert all(case["asr"]["error_rate"] is not None for case in baseline["cases"])
        serialized = json.dumps(baseline)
        assert "recognized_text" not in serialized
        assert "reference_voice" not in serialized
        assert re.search(r"[A-Za-z]:\\", serialized) is None


def test_normal_ci_covers_torchaudio_29_without_models():
    source = (ROOT / ".github/workflows/test_action.yml").read_text(encoding="utf-8")
    assert "torch==2.9.0" in source
    assert "torchaudio==2.9.0" in source
    assert "torchcodec==0.9.0" in source
    assert "run_multilingual_quality_regression.py" not in source
