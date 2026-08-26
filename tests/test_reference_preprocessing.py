from __future__ import annotations

import ast
from pathlib import Path


INFERENCE_SOURCE = Path(__file__).resolve().parents[1] / "indextts" / "infer_v2_5.py"


def _speaker_reference_calls() -> tuple[int, int]:
    tree = ast.parse(INFERENCE_SOURCE.read_text(encoding="utf-8"))
    method = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == "infer_generator"
    )
    embedding_calls = 0
    audio_loads = 0
    for item in ast.walk(method):
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute):
            continue
        if item.func.attr == "get_emb" and any(
            isinstance(argument, ast.Name) and argument.id == "input_features"
            for argument in item.args
        ):
            embedding_calls += 1
        if item.func.attr == "_load_and_cut_audio" and any(
            isinstance(argument, ast.Name) and argument.id == "spk_audio_prompt"
            for argument in item.args
        ):
            audio_loads += 1
    return embedding_calls, audio_loads


def test_speaker_reference_is_encoded_and_loaded_once_per_cache_miss() -> None:
    assert _speaker_reference_calls() == (1, 1)
