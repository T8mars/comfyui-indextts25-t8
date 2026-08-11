from __future__ import annotations

from types import SimpleNamespace

from packaging.version import Version
from transformers import GenerationConfig, LogitsProcessorList, __version__ as transformers_version

from indextts.gpt.model_v2 import GPT2InferenceModel
from indextts.gpt.transformers_generation_utils import GenerationMixin


def test_supported_transformers_v4_generation_api():
    version = Version(transformers_version)
    assert Version("4.52.1") <= version < Version("5")
    assert GenerationMixin in GPT2InferenceModel.__bases__

    model = GenerationMixin()
    model.config = SimpleNamespace(vocab_size=1024, is_encoder_decoder=False)
    generation_config = GenerationConfig()
    model._prepare_special_tokens(generation_config, device="cpu")
    processors = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=1,
        encoder_input_ids=None,
        prefix_allowed_tokens_fn=None,
        logits_processor=LogitsProcessorList(),
        device="cpu",
    )
    assert isinstance(processors, LogitsProcessorList)
