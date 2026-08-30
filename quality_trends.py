"""Portable trend summaries for multilingual GPU quality reports."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from html import escape
from statistics import mean, median
from typing import Any

from quality_regression import build_baseline_snapshot


def _finite(values: Sequence[Any]) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _profile_name(report: Mapping[str, Any]) -> str:
    profile = report.get("vram_profile")
    if isinstance(profile, Mapping):
        return str(profile.get("name") or "native")
    return str(profile or "native")


def build_quality_trend(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build path-free aggregate and per-language trend points."""

    points: list[dict[str, Any]] = []
    for index, source in enumerate(reports):
        report = build_baseline_snapshot(source)
        cases = list(report.get("cases") or [])
        error_rates = _finite([case.get("asr", {}).get("error_rate") for case in cases])
        rtfs = _finite([case.get("rtf") for case in cases])
        peaks = _finite([case.get("peak_vram_bytes") for case in cases])
        points.append(
            {
                "captured_at": report.get("created_at") or f"unknown-{index:04d}",
                "profile": _profile_name(source),
                "torch": report.get("torch"),
                "precision": report.get("precision"),
                "mean_asr_error_rate": round(mean(error_rates), 6) if error_rates else None,
                "median_rtf": round(median(rtfs), 6) if rtfs else None,
                "peak_vram_gb": round(max(peaks) / (1024**3), 4) if peaks else None,
                "cases": [
                    {
                        "id": case.get("id"),
                        "language": case.get("language"),
                        "asr_model": case.get("asr", {}).get("model"),
                        "asr_error_rate": case.get("asr", {}).get("error_rate"),
                        "rtf": case.get("rtf"),
                        "peak_vram_gb": (round(float(case.get("peak_vram_bytes")) / (1024**3), 4) if case.get("peak_vram_bytes") is not None else None),
                    }
                    for case in cases
                ],
            }
        )
    points.sort(key=lambda point: (str(point["captured_at"]), str(point["profile"])))
    return {"schema_version": 1, "trend_kind": "indextts25-gpu-quality-history", "points": points}


def render_quality_trend_markdown(trend: Mapping[str, Any]) -> str:
    lines = ["# IndexTTS 2.5 GPU quality trend", "", "| Captured | VRAM profile | Mean CER/WER | Median RTF | Peak VRAM (GiB) |", "|---|---:|---:|---:|---:|"]
    for point in trend.get("points", []):
        def value(name: str) -> str:
            current = point.get(name)
            return "—" if current is None else f"{float(current):.4f}"
        lines.append(f"| {point.get('captured_at')} | {point.get('profile')} | {value('mean_asr_error_rate')} | {value('median_rtf')} | {value('peak_vram_gb')} |")
    return "\n".join(lines) + "\n"


def render_quality_trend_svg(trend: Mapping[str, Any]) -> str:
    """Render three compact aggregate curves without plotting dependencies."""
    points = list(trend.get("points") or [])
    width, height, left, right, panel_height = 1000, 620, 85, 25, 145
    plot_width = width - left - right
    metrics = (("mean_asr_error_rate", "Mean CER/WER", "#e84a8a"), ("median_rtf", "Median RTF", "#4f8cff"), ("peak_vram_gb", "Peak VRAM (GiB)", "#25a36f"))
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#10131a"/>', '<text x="30" y="35" fill="#f4f6fb" font-size="22" font-family="sans-serif">IndexTTS 2.5 GPU quality trend</text>']
    for panel_index, (key, label, color) in enumerate(metrics):
        top = 65 + panel_index * 180
        values = _finite([point.get(key) for point in points])
        maximum = max(values) if values else 1.0
        maximum = maximum if maximum > 0 else 1.0
        content.extend([f'<text x="{left}" y="{top - 10}" fill="#dbe1ed" font-size="15" font-family="sans-serif">{escape(label)}</text>', f'<line x1="{left}" y1="{top + panel_height}" x2="{width-right}" y2="{top + panel_height}" stroke="#4a5263"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" stroke="#4a5263"/>', f'<text x="10" y="{top + 6}" fill="#8f9aae" font-size="12" font-family="sans-serif">{maximum:.3f}</text>', f'<text x="48" y="{top + panel_height + 5}" fill="#8f9aae" font-size="12" font-family="sans-serif">0</text>'])
        coordinates = []
        for index, point in enumerate(points):
            value = point.get(key)
            if value is None:
                continue
            x = left + (plot_width * index / max(len(points) - 1, 1))
            y = top + panel_height - (float(value) / maximum * panel_height)
            coordinates.append((x, y, point))
        if coordinates:
            polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coordinates)
            content.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3"/>')
            for x, y, point in coordinates:
                title = escape(f"{point.get('captured_at')} {point.get('profile')}: {point.get(key)}")
                content.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"><title>{title}</title></circle>')
    content.append("</svg>")
    return "\n".join(content) + "\n"


__all__ = ["build_quality_trend", "render_quality_trend_markdown", "render_quality_trend_svg"]
