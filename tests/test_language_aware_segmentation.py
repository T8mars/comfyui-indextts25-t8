from __future__ import annotations

from types import SimpleNamespace

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.text_segmentation import language_aware_token_budget


def test_non_cjk_budget_cap_and_existing_small_limit():
    assert language_aware_token_budget(120, 7, language="EN") == 81
    assert language_aware_token_budget(300, 7, language="ES") == 81
    assert language_aware_token_budget(60, 7, language="EN") == 53


def test_inference_splitter_uses_language_aware_budget():
    class CharacterTokenizer:
        @staticmethod
        def encode(text, allowed_special="all"):
            return list(text)

    splitter = IndexTTS2.__new__(IndexTTS2)
    splitter.tokenizer = CharacterTokenizer()
    splitter.gpt = SimpleNamespace(
        text_pos_embedding=SimpleNamespace(
            emb=SimpleNamespace(num_embeddings=1024)
        )
    )

    capped = splitter.split_text_by_tokens("A" * 90, 120, "<|en|> ")
    assert list(map(len, capped)) == [81, 9]
    assert splitter.split_text_by_tokens("B" * 50, 60, "<|en|> ") == ["B" * 50]
    assert splitter.split_text_by_tokens("甲" * 100, 120, "<|zh|> ") == ["甲" * 100]
