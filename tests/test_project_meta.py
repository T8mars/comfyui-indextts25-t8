from __future__ import annotations

import re
from pathlib import Path

from project_meta import PROJECT_VERSION, read_project_version


def test_runtime_version_matches_pyproject() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert declared is not None
    assert PROJECT_VERSION == read_project_version() == declared.group(1)
