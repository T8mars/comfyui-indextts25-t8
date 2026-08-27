"""Wait until the uploaded Comfy Registry version is Active.

The publish action returning successfully only proves that the archive was
accepted.  Registry security scanning is asynchronous, so this command treats
Flagged/Banned as a release failure and does not report Pending as success.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

try:
    from .check_registry_version import (
        DEFAULT_REGISTRY_API,
        RegistryLookupError,
        lookup_registry_version,
        read_release_identity,
    )
except ImportError:
    from check_registry_version import (
        DEFAULT_REGISTRY_API,
        RegistryLookupError,
        lookup_registry_version,
        read_release_identity,
    )


ACTIVE = "NodeVersionStatusActive"
FAILED = {
    "NodeVersionStatusFlagged",
    "NodeVersionStatusBanned",
    "NodeVersionStatusDeleted",
}


class RegistryStatusError(RuntimeError):
    """Raised when the uploaded version cannot become an installable release."""


def wait_for_active(
    node_id: str,
    version: str,
    *,
    base_url: str = DEFAULT_REGISTRY_API,
    attempts: int = 30,
    interval: float = 20.0,
    lookup: Callable[..., tuple[bool, str]] = lookup_registry_version,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last_status = "missing"
    for attempt in range(1, attempts + 1):
        exists, status = lookup(node_id, version, base_url=base_url)
        last_status = status if exists else "missing"
        print(
            f"Registry activation check {attempt}/{attempts}: "
            f"{node_id} {version} status={last_status}",
            flush=True,
        )
        if exists and status == ACTIVE:
            return status
        if exists and status in FAILED:
            raise RegistryStatusError(
                f"Comfy Registry security review returned {status}; "
                "the upload exists but is not installable from Manager."
            )
        if attempt < attempts:
            sleep(max(0.0, float(interval)))
    raise RegistryStatusError(
        f"Comfy Registry did not activate {node_id} {version}; "
        f"last status={last_status}."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument(
        "--registry-base-url",
        default=os.environ.get("COMFY_REGISTRY_API_URL", DEFAULT_REGISTRY_API),
    )
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--interval", type=float, default=20.0)
    args = parser.parse_args(argv)
    try:
        node_id, version = read_release_identity(args.pyproject)
        wait_for_active(
            node_id,
            version,
            base_url=args.registry_base_url,
            attempts=args.attempts,
            interval=args.interval,
        )
    except (OSError, ValueError, RegistryLookupError, RegistryStatusError) as exc:
        print(f"Registry activation guard failed: {exc}", file=sys.stderr)
        return 2
    print(f"Comfy Registry 已激活 {node_id} {version}，Manager 可以安装该版本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
