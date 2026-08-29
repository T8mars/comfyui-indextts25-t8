from __future__ import annotations

import base64
import json

import pytest

import runtime.update_check as update_check
from runtime.update_check import URLS, check_updates, compare_versions


def test_compare_versions() -> None:
    assert compare_versions("0.14.0", "0.13.9") > 0
    assert compare_versions("0.13.0", "0.13.0") == 0


def test_check_updates_reports_each_source() -> None:
    values = {
        URLS["official_code"]: json.dumps({"sha": "new-code"}),
        URLS["official_model"]: json.dumps({"sha": "same-model"}),
        URLS["node_project"]: json.dumps(
            {
                "content": base64.b64encode(
                    b'[project]\nversion = "0.14.0"\n'
                ).decode("ascii")
            }
        ),
    }

    def fetcher(url: str, timeout: float) -> str:
        return values[url]

    report = check_updates(
        "0.13.0",
        {
            "codeRevision": "old-code",
            "modelRevision": "bundle-model",
            "upstreamModelRevision": "same-model",
        },
        fetcher=fetcher,
    )
    assert report["node"]["update_available"] is True
    assert report["official_code"]["update_available"] is True
    assert report["official_model"]["update_available"] is False
    assert report["official_model"]["pinned"] == "same-model"


def test_fetch_rejects_non_builtin_endpoint() -> None:
    with pytest.raises(ValueError, match="固定端点"):
        update_check._fetch("https://example.com/untrusted", 2.0)


def test_fetch_disables_redirects_and_caps_response(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        chunks = [b"{}"]
        headers = {}
        closed = False

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size):
            captured["chunk_size"] = chunk_size
            yield from self.chunks

        def close(self):
            self.closed = True
            captured["closed"] = True

    class FakeSession:
        def get(self, url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return FakeResponse()

    monkeypatch.setattr(update_check, "get_session", lambda: FakeSession())

    assert update_check._fetch(URLS["official_code"], 2.0) == "{}"
    assert captured["url"] == URLS["official_code"]
    assert captured["kwargs"]["allow_redirects"] is False
    assert captured["kwargs"]["timeout"] == 2.0
    assert captured["kwargs"]["stream"] is True
    assert captured["chunk_size"] == 64 * 1024

    FakeResponse.headers = {"Content-Length": "unknown"}
    assert update_check._fetch(URLS["official_code"], 2.0) == "{}"

    FakeResponse.chunks = [
        b"x" * update_check.MAX_RESPONSE_BYTES,
        b"y",
        b"this chunk must not be consumed",
    ]
    with pytest.raises(ValueError, match="1 MiB"):
        update_check._fetch(URLS["official_code"], 2.0)
    assert captured["closed"] is True
