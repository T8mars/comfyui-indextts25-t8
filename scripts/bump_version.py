#!/usr/bin/env python3
"""Increment the Comfy Registry version stored in pyproject.toml."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VERSION_LINE = re.compile(
    r'^(?P<prefix>\s*version\s*=\s*")(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?P<suffix>"\s*(?:#.*)?)$'
)


def bump_version_text(text: str, part: str) -> tuple[str, str, str]:
    """Return updated TOML text plus the old and new project versions."""
    lines = text.splitlines(keepends=True)
    in_project = False
    match_index: int | None = None
    match: re.Match[str] | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if in_project and stripped.startswith("["):
            break
        if in_project:
            candidate = VERSION_LINE.match(line.rstrip("\r\n"))
            if candidate is not None:
                match_index = index
                match = candidate
                break

    if match_index is None or match is None:
        raise ValueError('Could not find a three-part `version = "X.Y.Z"` in [project].')

    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))
    old_version = f"{major}.{minor}.{patch}"

    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unsupported version part: {part}")

    new_version = f"{major}.{minor}.{patch}"
    original_line = lines[match_index]
    newline = "\r\n" if original_line.endswith("\r\n") else "\n" if original_line.endswith("\n") else ""
    lines[match_index] = f'{match.group("prefix")}{new_version}{match.group("suffix")}{newline}'
    return "".join(lines), old_version, new_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("patch", "minor", "major"), nargs="?", default="patch")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pyproject.toml",
        help="Path to pyproject.toml",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the new version without changing the file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.pyproject.resolve()
    original = path.read_text(encoding="utf-8")
    updated, old_version, new_version = bump_version_text(original, args.part)
    if not args.dry_run:
        path.write_text(updated, encoding="utf-8", newline="")
    print(f"{old_version} -> {new_version}" + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
