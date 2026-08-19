from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bump_version.py"
SPEC = importlib.util.spec_from_file_location("bump_version", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUMP_VERSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUMP_VERSION)


def test_bump_version_text_updates_only_project_version() -> None:
    source = '[project]\nversion = "0.5.1"\n\n[tool.example]\nversion = "9.9.9"\n'

    updated, old_version, new_version = BUMP_VERSION.bump_version_text(source, "patch")

    assert old_version == "0.5.1"
    assert new_version == "0.5.2"
    assert 'version = "0.5.2"' in updated
    assert 'version = "9.9.9"' in updated


def test_bump_version_text_supports_minor_and_major() -> None:
    source = '[project]\nversion = "1.2.3"\n'

    minor_text, _, minor_version = BUMP_VERSION.bump_version_text(source, "minor")
    major_text, _, major_version = BUMP_VERSION.bump_version_text(source, "major")

    assert minor_version == "1.3.0"
    assert 'version = "1.3.0"' in minor_text
    assert major_version == "2.0.0"
    assert 'version = "2.0.0"' in major_text


def test_bump_version_text_rejects_missing_project_version() -> None:
    with pytest.raises(ValueError, match="Could not find"):
        BUMP_VERSION.bump_version_text('[tool.example]\nversion = "1.2.3"\n', "patch")
