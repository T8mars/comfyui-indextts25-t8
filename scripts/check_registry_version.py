"""Check whether the current Comfy Registry version already exists.

The release workflow uses this as an idempotency guard.  A 200 response means
the immutable version is already registered (including Pending versions), while
only a 404 permits a new upload.  Transient lookup failures stop the workflow
instead of risking a duplicate publish.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_REGISTRY_API = "https://api.comfy.org"
_SECTION_PATTERN = re.compile(r"^\s*\[([^]]+)]\s*$")
_STRING_ASSIGNMENT_PATTERN = re.compile(
    r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*"([^"]*)"\s*$'
)
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class RegistryLookupError(RuntimeError):
    """Raised when Registry state cannot be determined safely."""


def read_release_identity(pyproject_path: Path) -> tuple[str, str]:
    """Return ``(project.name, project.version)`` without third-party TOML deps."""

    values: dict[tuple[str, str], str] = {}
    section = ""
    for line in pyproject_path.read_text(encoding="utf-8").splitlines():
        section_match = _SECTION_PATTERN.match(line)
        if section_match:
            section = section_match.group(1).strip().lower()
            continue
        assignment = _STRING_ASSIGNMENT_PATTERN.match(line)
        if assignment:
            values[(section, assignment.group(1).lower())] = assignment.group(2)

    node_id = values.get(("project", "name"), "")
    version = values.get(("project", "version"), "")
    for label, value in (("project.name", node_id), ("project.version", version)):
        if not _SAFE_ID_PATTERN.fullmatch(value):
            raise ValueError(f"pyproject.toml 中的 {label} 缺失或格式不安全。")
    return node_id, version


def registry_version_url(base_url: str, node_id: str, version: str) -> str:
    return (
        f"{base_url.rstrip('/')}/nodes/{quote(node_id, safe='')}"
        f"/versions/{quote(version, safe='')}"
    )


def lookup_registry_version(
    node_id: str,
    version: str,
    *,
    base_url: str = DEFAULT_REGISTRY_API,
    attempts: int = 3,
    timeout: float = 20.0,
    open_url: Callable[..., object] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, str]:
    """Return ``(exists, status)``; retry transient failures and fail closed."""

    url = registry_version_url(base_url, node_id, version)
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "indextts25-t8-release-guard"},
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with open_url(request, timeout=timeout) as response:  # type: ignore[attr-defined]
                payload = json.loads(response.read().decode("utf-8"))
            returned_version = str(payload.get("version", ""))
            if returned_version != version:
                raise RegistryLookupError(
                    f"Registry 返回了意外版本：请求 {version!r}，收到 {returned_version!r}。"
                )
            return True, str(payload.get("status", "unknown"))
        except HTTPError as exc:
            if exc.code == 404:
                return False, "missing"
            last_error = exc
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = exc

        if attempt < attempts:
            sleep(float(2 ** (attempt - 1)))

    raise RegistryLookupError(
        f"查询 Comfy Registry 失败（已重试 {attempts} 次）：{last_error}"
    ) from last_error


def write_github_output(path: Path, *, exists: bool) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"exists={'true' if exists else 'false'}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument(
        "--registry-base-url",
        default=os.environ.get("COMFY_REGISTRY_API_URL", DEFAULT_REGISTRY_API),
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=Path(os.environ["GITHUB_OUTPUT"]) if os.environ.get("GITHUB_OUTPUT") else None,
    )
    args = parser.parse_args(argv)

    try:
        node_id, version = read_release_identity(args.pyproject)
        exists, status = lookup_registry_version(
            node_id,
            version,
            base_url=args.registry_base_url,
        )
        if args.github_output is not None:
            write_github_output(args.github_output, exists=exists)
    except (OSError, ValueError, RegistryLookupError) as exc:
        print(f"Registry version guard failed: {exc}", file=sys.stderr)
        return 2

    if exists:
        print(
            f"Comfy Registry 已存在 {node_id} {version}（status={status}），跳过重复上传。"
        )
    else:
        print(f"Comfy Registry 尚无 {node_id} {version}，允许发布。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
