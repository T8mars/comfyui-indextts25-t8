import json

import pytest
import torch

from runtime.dialogue import compose_timeline, fit_duration_factor, missing_roles, parse_batch_script, parse_srt


def test_parse_batch_and_srt_roles():
    batch = parse_batch_script("A|你好|ZH|0.8\nB|Hello|EN|1.0")
    assert [(line.role, line.language) for line in batch] == [("A", "ZH"), ("B", "EN")]
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[A] 第一行\n第二行")
    assert srt[0].role == "A"
    assert srt[0].text == "第一行\n第二行"
    assert srt[0].slot_ms == 1000


def test_json_script_and_role_validation():
    lines = parse_batch_script(json.dumps([{"role": "A", "text": "Hi", "language": "EN"}]))
    assert missing_roles(lines, ["B"]) == ["A"]
    assert fit_duration_factor(1.0, 2000, 1500) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="第 2 条.*对象"):
        parse_batch_script(json.dumps([{"role": "A", "text": "ok"}, 42]))


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
