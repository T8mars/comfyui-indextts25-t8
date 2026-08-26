from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from scripts.check_registry_version import (
    RegistryLookupError,
    lookup_registry_version,
    read_release_identity,
    registry_version_url,
    write_github_output,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _http_error(url: str, code: int) -> HTTPError:
    return HTTPError(url, code, "test", hdrs=None, fp=None)


def test_read_release_identity_uses_project_section(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "indextts25-t8"\nversion = "0.11.5"\n\n'
        '[tool.other]\nname = "wrong"\n',
        encoding="utf-8",
    )
    assert read_release_identity(pyproject) == ("indextts25-t8", "0.11.5")


def test_registry_version_url_escapes_path_segments() -> None:
    assert registry_version_url("https://api.example/", "node+id", "1.0+cu") == (
        "https://api.example/nodes/node%2Bid/versions/1.0%2Bcu"
    )


def test_existing_pending_version_is_treated_as_published() -> None:
    def open_url(_request, **_kwargs):
        return _Response({"version": "0.11.4", "status": "NodeVersionStatusPending"})

    assert lookup_registry_version(
        "indextts25-t8", "0.11.4", open_url=open_url
    ) == (True, "NodeVersionStatusPending")


def test_missing_version_allows_publish() -> None:
    def open_url(request, **_kwargs):
        raise _http_error(request.full_url, 404)

    assert lookup_registry_version(
        "indextts25-t8", "0.11.5", open_url=open_url
    ) == (False, "missing")


def test_transient_lookup_failure_retries_before_success() -> None:
    attempts = 0
    sleeps: list[float] = []

    def open_url(_request, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise URLError("temporary")
        return _Response({"version": "0.11.5", "status": "NodeVersionStatusActive"})

    assert lookup_registry_version(
        "indextts25-t8", "0.11.5", open_url=open_url, sleep=sleeps.append
    ) == (True, "NodeVersionStatusActive")
    assert attempts == 3
    assert sleeps == [1.0, 2.0]


def test_lookup_fails_closed_when_registry_cannot_be_reached() -> None:
    def open_url(_request, **_kwargs):
        raise URLError("offline")

    with pytest.raises(RegistryLookupError, match="已重试 3 次"):
        lookup_registry_version(
            "indextts25-t8", "0.11.5", open_url=open_url, sleep=lambda _delay: None
        )


def test_github_output_is_boolean_only(tmp_path: Path) -> None:
    output = tmp_path / "github-output.txt"
    write_github_output(output, exists=True)
    assert output.read_text(encoding="utf-8") == "exists=true\n"
