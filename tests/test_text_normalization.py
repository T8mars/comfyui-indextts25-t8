from __future__ import annotations

from pathlib import Path
import sys

from indextts.utils import front
from indextts.utils.front import (
    TEXT_NORMALIZATION_SMOKE_EXPECTED,
    TEXT_NORMALIZATION_SMOKE_INPUT,
    TextNormalizer,
    probe_text_normalization,
)


def test_optional_backend_normalizes_chinese_year_by_context():
    normalizer = TextNormalizer()
    normalizer.load()

    assert normalizer.normalize(TEXT_NORMALIZATION_SMOKE_INPUT) == TEXT_NORMALIZATION_SMOKE_EXPECTED
    assert normalizer.normalize("1939个人") == "一千九百三十九个人"

    report = probe_text_normalization()
    assert report["available"] is True
    assert report["verified"] is True
    assert report["example_output"] == TEXT_NORMALIZATION_SMOKE_EXPECTED
    assert report["backend"] in {"wetext", "WeTextProcessing"}


def test_missing_optional_backend_falls_back_to_original_text(monkeypatch):
    monkeypatch.setattr(front.platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "wetext", None)

    report = probe_text_normalization()

    assert report["available"] is False
    assert report["verified"] is False
    assert report["backend"] == "identity"
    assert report["example_output"] == TEXT_NORMALIZATION_SMOKE_INPUT
    assert report["error"]


def test_optional_dependency_has_a_dedicated_cross_platform_install_file():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements-text-normalization.txt").read_text(encoding="utf-8")
    project = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert "wetext>=0.1.7,<0.2" in requirements
    assert "WeTextProcessing>=1.2.0,<2" in requirements
    assert "text-normalization = [" in project
