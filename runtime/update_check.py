from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone


URLS = {
    "official_code": "https://api.github.com/repos/index-tts/index-tts/commits/main",
    "official_model": "https://huggingface.co/api/models/IndexTeam/IndexTTS-2.5",
    "node_project": "https://raw.githubusercontent.com/T8mars/comfyui-indextts25-t8/main/pyproject.toml",
}


def compare_versions(left: str, right: str) -> int:
    def parts(value: str) -> list[int]:
        return [int(item) for item in re.findall(r"\d+", str(value))[:3]]

    a, b = parts(left), parts(right)
    size = max(len(a), len(b), 3)
    a += [0] * (size - len(a))
    b += [0] * (size - len(b))
    return (a > b) - (a < b)


def _fetch(url: str, timeout_seconds: float) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "comfyui-indextts25-t8-update-check",
            "Accept": "application/json, text/plain;q=0.9, */*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
        return response.read().decode("utf-8")


def check_updates(
    project_version: str,
    manifest: dict,
    timeout_seconds: float = 12.0,
    fetcher=_fetch,
) -> dict:
    errors: list[str] = []
    values: dict[str, str] = {}
    for name, url in URLS.items():
        try:
            values[name] = fetcher(url, timeout_seconds)
        except Exception as exc:
            errors.append(f"{name}: {str(exc).strip() or type(exc).__name__}")

    latest_code = ""
    latest_model = ""
    latest_node = ""
    try:
        latest_code = str(json.loads(values.get("official_code", "{}"))["sha"])
    except Exception:
        if "official_code" in values:
            errors.append("official_code: 返回格式无效")
    try:
        latest_model = str(json.loads(values.get("official_model", "{}"))["sha"])
    except Exception:
        if "official_model" in values:
            errors.append("official_model: 返回格式无效")
    if "node_project" in values:
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', values["node_project"], re.M)
        if match:
            latest_node = match.group(1)
        else:
            errors.append("node_project: 未找到版本号")

    pinned_code = str(manifest.get("codeRevision", ""))
    pinned_model = str(
        manifest.get("upstreamModelRevision") or manifest.get("modelRevision", "")
    )
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "node": {
            "current": str(project_version),
            "latest": latest_node,
            "update_available": bool(
                latest_node and compare_versions(latest_node, project_version) > 0
            ),
        },
        "official_code": {
            "pinned": pinned_code,
            "latest": latest_code,
            "update_available": bool(
                latest_code
                and not latest_code.startswith(pinned_code)
                and not pinned_code.startswith(latest_code)
            ),
        },
        "official_model": {
            "pinned": pinned_model,
            "latest": latest_model,
            "update_available": bool(latest_model and latest_model != pinned_model),
        },
        "errors": errors,
    }
    count = sum(
        int(report[name]["update_available"])
        for name in ("node", "official_code", "official_model")
    )
    report["summary"] = (
        "全部检查失败，请确认网络后重试。"
        if len(errors) >= 3 and not any((latest_node, latest_code, latest_model))
        else f"发现 {count} 项新版本；本节点只提示，不会自动下载或覆盖。"
        if count
        else "当前未发现新版本。"
    )
    return report
