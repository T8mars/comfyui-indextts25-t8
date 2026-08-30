"""Build JSON, Markdown, and SVG history from GPU quality-report.json files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quality_trends import (
    build_quality_trend,
    render_quality_trend_markdown,
    render_quality_trend_svg,
)


def _report_paths(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for source in inputs:
        if source.is_file():
            paths.add(source.resolve())
        elif source.is_dir():
            paths.update(path.resolve() for path in source.rglob("quality-report.json"))
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = _report_paths(args.input)
    if not paths:
        raise SystemExit("No quality-report.json files were found.")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    trend = build_quality_trend(reports)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "quality-trend.json").write_text(
        json.dumps(trend, ensure_ascii=False, indent=2) + os.linesep,
        encoding="utf-8",
    )
    (args.output_dir / "quality-trend.md").write_text(
        render_quality_trend_markdown(trend), encoding="utf-8"
    )
    (args.output_dir / "quality-trend.svg").write_text(
        render_quality_trend_svg(trend), encoding="utf-8"
    )
    print(f"Quality trend points: {len(trend['points'])}")
    print(f"Quality trend output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
