from __future__ import annotations

import json

from runtime.update_check import URLS, check_updates, compare_versions


def test_compare_versions() -> None:
    assert compare_versions("0.14.0", "0.13.9") > 0
    assert compare_versions("0.13.0", "0.13.0") == 0


def test_check_updates_reports_each_source() -> None:
    values = {
        URLS["official_code"]: json.dumps({"sha": "new-code"}),
        URLS["official_model"]: json.dumps({"sha": "same-model"}),
        URLS["node_project"]: '[project]\nversion = "0.14.0"\n',
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
