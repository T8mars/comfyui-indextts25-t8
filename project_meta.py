"""Single-source project metadata for runtime status messages."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
_VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def read_project_version() -> str:
    match = _VERSION_PATTERN.search(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    if match is None:
        raise RuntimeError("pyproject.toml 缺少 [project].version。")
    return match.group(1)


PROJECT_VERSION = read_project_version()


__all__ = ["PROJECT_VERSION", "read_project_version"]
