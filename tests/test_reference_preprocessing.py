from __future__ import annotations

import ast
from pathlib import Path


INFERENCE_SOURCE = Path(__file__).resolve().parents[1] / "indextts" / "infer_v2_5.py"


def _method(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == name
    )


def _speaker_reference_calls() -> tuple[int, int]:
    tree = ast.parse(INFERENCE_SOURCE.read_text(encoding="utf-8"))
    embedding_method = _method(tree, "_get_reference_embedding")
    speaker_method = _method(tree, "_prepare_speaker_reference")
    embedding_calls = 0
    audio_loads = 0
    for item in ast.walk(embedding_method):
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute):
            continue
        if item.func.attr == "get_emb" and any(
            isinstance(argument, ast.Name) and argument.id == "input_features"
            for argument in item.args
        ):
            embedding_calls += 1
    for item in ast.walk(speaker_method):
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute):
            continue
        if item.func.attr == "_load_and_cut_audio" and any(
            isinstance(argument, ast.Name) and argument.id == "spk_audio_prompt"
            for argument in item.args
        ):
            audio_loads += 1
    return embedding_calls, audio_loads


def test_speaker_reference_is_encoded_and_loaded_once_per_cache_miss() -> None:
    assert _speaker_reference_calls() == (1, 1)


def test_low_vram_constructor_options_are_backward_compatible() -> None:
    tree = ast.parse(INFERENCE_SOURCE.read_text(encoding="utf-8"))
    constructor = _method(tree, "__init__")
    names = [argument.arg for argument in constructor.args.args]
    original = [
        "self",
        "cfg_path",
        "model_dir",
        "use_bf16",
        "device",
        "use_cuda_kernel",
        "use_deepspeed",
        "use_accel",
        "use_torch_compile",
        "use_qwen_emo",
    ]
    assert names[: len(original)] == original
    assert names[len(original) :] == [
        "use_fp16",
        "reference_device",
        "reuse_spk_cond_for_emo",
    ]
