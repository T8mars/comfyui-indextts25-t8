import json

import pytest
import torch

from runtime.dialogue import (
    DialogueLine,
    compose_timeline,
    fit_duration_factor,
    missing_roles,
    parse_batch_script,
    parse_srt,
)


def test_parse_batch_and_srt_roles():
    batch = parse_batch_script("A|你好|ZH|0.8\nB|Hello|EN|1.0")
    assert [(line.role, line.language) for line in batch] == [("A", "ZH"), ("B", "EN")]
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[A] 第一行\n第二行")
    assert srt[0].role == "A"
    assert srt[0].text == "第一行\n第二行"
    assert srt[0].slot_ms == 1000


def test_json_script_and_role_validation():
    lines = parse_batch_script(
        json.dumps([{"role": "A", "text": "Hi", "language": "EN"}])
    )
    assert missing_roles(lines, ["B"]) == ["A"]
    assert fit_duration_factor(1.0, 2000, 1500) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="第 2 条.*对象"):
        parse_batch_script(json.dumps([{"role": "A", "text": "ok"}, 42]))


def test_plain_batch_preserves_pronunciation_annotation_delimiters():
    lines = parse_batch_script(
        "旁白|小明<要求|YAO4 QIU2>这个题。|ZH|1.0\n"
        "A|He waited a <minute|M IH1 . N AH0 T>.|EN|0.9"
    )
    assert lines[0].text == "小明<要求|YAO4 QIU2>这个题。"
    assert lines[0].language == "ZH"
    assert lines[1].text == "He waited a <minute|M IH1 . N AH0 T>."
    assert lines[1].duration_factor == pytest.approx(0.9)


@pytest.mark.parametrize(
    "payload",
    [
        [{"role": "A", "text": "x", "start_ms": 1000}],
        [{"role": "A", "text": "x", "start_ms": -1, "end_ms": 1000}],
        [{"role": "A", "text": "x", "start_ms": 2000, "end_ms": 1000}],
        [{"role": "A", "text": "x", "start_ms": 864_000_000, "end_ms": 864_001_000}],
    ],
)
def test_json_script_rejects_unsafe_timeline_values(payload):
    with pytest.raises(ValueError, match="start_ms/end_ms|0–86400000"):
        parse_batch_script(json.dumps(payload))


def test_srt_rejects_timestamp_past_one_day():
    with pytest.raises(ValueError, match="不能超过"):
        parse_srt("1\n25:00:00,000 --> 25:00:01,000\n[A] too late")


def test_compose_timeline_rejects_oversized_dense_allocation():
    lines = [DialogueLine(1, "A", "x", "ZH", 86_399_000, 86_400_000)]
    with pytest.raises(ValueError, match="1 GiB 安全上限"):
        compose_timeline([torch.zeros(1, 1, 100)], lines, 48_000, "overlay")


def test_compose_overlay_averages_overlap():
    lines = parse_srt(
        "1\n00:00:00,000 --> 00:00:01,000\n[A] one\n\n"
        "2\n00:00:00,500 --> 00:00:01,500\n[B] two"
    )
    mixed, report = compose_timeline(
        [torch.ones(1, 1, 1000), torch.zeros(1, 1, 1000)], lines, 1000, "overlay"
    )
    assert mixed.shape[-1] == 1500
    assert mixed[0, 0, 750].item() == pytest.approx(0.5)
    assert report[1].actual_start_ms == 500
