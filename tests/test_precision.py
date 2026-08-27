from __future__ import annotations

from types import SimpleNamespace

import pytest

from indextts.utils.precision import (
    cuda_supports_native_bf16,
    resolve_gpt_precision,
    select_half_precision,
)
from runtime.types import ModelHandle


@pytest.mark.parametrize(
    "use_fp16,use_bf16,device,expected",
    [
        (False, False, "cuda:0", None),
        (True, False, "cuda:0", "fp16"),
        (False, True, "cuda:0", "bf16"),
        (True, False, "xpu", "fp16"),
        (True, False, "cpu", None),
        (False, True, "mps", None),
    ],
)
def test_resolve_gpt_precision(use_fp16, use_bf16, device, expected):
    assert resolve_gpt_precision(
        use_fp16=use_fp16, use_bf16=use_bf16, device=device
    ) == expected


def test_mutually_exclusive_precision_is_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_gpt_precision(use_fp16=True, use_bf16=True, device="cuda:0")


def test_native_bf16_probe_explicitly_disables_emulation():
    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda **kwargs: calls.append(kwargs) or True,
    )
    assert cuda_supports_native_bf16(SimpleNamespace(cuda=cuda)) is True
    assert calls == [{"including_emulation": False}]


def test_half_precision_prefers_native_bf16_then_fp16():
    assert select_half_precision(
        enabled=True, cuda_available=True, native_bf16_supported=True
    ) == "bf16"
    assert select_half_precision(
        enabled=True, cuda_available=True, native_bf16_supported=False
    ) == "fp16"


def test_model_cache_identity_includes_low_vram_controls(tmp_path):
    base = ModelHandle(tmp_path, "cuda:0", False)
    fp16 = ModelHandle(tmp_path, "cuda:0", False, use_fp16=True)
    cpu_reference = ModelHandle(
        tmp_path, "cuda:0", False, reference_device="cpu"
    )
    reuse = ModelHandle(
        tmp_path, "cuda:0", False, reuse_spk_cond_for_emo=True
    )
    assert len({base.cache_key, fp16.cache_key, cpu_reference.cache_key, reuse.cache_key}) == 4


def test_default_emotion_reuse_is_strictly_opt_in():
    from indextts.infer_v2_5 import IndexTTS2

    model = IndexTTS2.__new__(IndexTTS2)
    model.reuse_spk_cond_for_emo = True
    assert model._should_reuse_spk_cond_for_emo(None, None, False) is True
    assert model._should_reuse_spk_cond_for_emo("emotion.wav", None, False) is False
    assert model._should_reuse_spk_cond_for_emo(None, [0.0] * 8, False) is False
    assert model._should_reuse_spk_cond_for_emo(None, None, True) is False


def test_reused_default_emotion_skips_merge_path():
    from indextts.infer_v2_5 import IndexTTS2

    calls = []

    class Gpt:
        def get_emovec(self, cond, lengths):
            calls.append(("get", cond, lengths))
            return "fast"

        def merge_emovec(self, *args, **kwargs):
            calls.append(("merge", args, kwargs))
            return "merged"

    model = IndexTTS2.__new__(IndexTTS2)
    model.gpt = Gpt()
    assert model._get_emovec("spk", "spk", "length", "length", 1.0, True) == "fast"
    assert [item[0] for item in calls] == ["get"]


def test_reference_embedding_moves_only_result_to_synthesis_device():
    from indextts.infer_v2_5 import IndexTTS2

    class Tensor:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(str(device))
            return self

    features, mask, embedding = Tensor(), Tensor(), Tensor()
    model = IndexTTS2.__new__(IndexTTS2)
    model.reference_device = "cpu"
    model.device = "cuda:0"
    model.extract_features = lambda *_args, **_kwargs: {
        "input_features": features,
        "attention_mask": mask,
    }
    model.get_emb = lambda *_args: embedding
    assert model._get_reference_embedding(object()) is embedding
    assert features.moves == ["cpu"]
    assert mask.moves == ["cpu"]
    assert embedding.moves == ["cuda:0"]
