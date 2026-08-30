from types import SimpleNamespace

import pytest

from indextts.cli import (
    build_parser,
    parse_emotion_vector,
    resolve_device,
    resolve_precision,
)


def test_cli_defaults_to_indextts25_multilingual_path():
    args = build_parser().parse_args(["hello", "--voice", "voice.wav"])
    assert args.language == "ZH"
    assert args.model_dir.name == "checkpoints"
    assert args.precision == "auto"
    assert args.acceleration == "off"


def test_cli_accepts_all_official_25_controls():
    args = build_parser().parse_args(
        [
            "hello",
            "--voice",
            "voice.wav",
            "--language",
            "EN",
            "--duration-factor",
            "1.2",
            "--emotion-vector",
            "0.1,0.2,0,0,0,0,0,0.7",
            "--acceleration",
            "gpt_accel",
        ]
    )
    assert args.language == "EN"
    assert args.duration_factor == 1.2
    assert args.acceleration == "gpt_accel"


def test_emotion_vector_validation_is_explicit():
    assert parse_emotion_vector("0.1,0.2,0,0,0,0,0,0.7")[-1] == 0.7
    with pytest.raises(ValueError, match="exactly eight"):
        parse_emotion_vector("0.1,0.2")
    with pytest.raises(ValueError, match="sum"):
        parse_emotion_vector("1,1,0,0,0,0,0,0")


def test_precision_auto_uses_bf16_only_when_supported():
    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)
    torch_stub = SimpleNamespace(cuda=cuda)
    assert resolve_precision(torch_stub, "auto", "cuda:0") == (True, False, "bf16")
    assert resolve_precision(torch_stub, "bf16", "cpu") == (False, False, "fp32")
    assert resolve_device(torch_stub, None) == "cuda:0"
