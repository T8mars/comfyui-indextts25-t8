from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest


pytest.importorskip("comfy_api.latest")


def _load_plugin():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "comfyui_indextts25_t8_test",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registers_all_pure_v3_nodes():
    plugin = _load_plugin()
    extension = asyncio.run(plugin.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    schemas = [node.GET_SCHEMA() for node in nodes]
    assert [schema.node_id for schema in schemas] == [
        "T8_IndexTTS25_ModelLoader",
        "T8_IndexTTS25_EmotionControl",
        "T8_IndexTTS25_SamplingConfig",
        "T8_IndexTTS25_Pronunciation",
        "T8_IndexTTS25_Generate",
        "T8_IndexTTS25_VoiceProfile",
        "T8_IndexTTS25_RoleLibrary",
        "T8_IndexTTS25_DialogueScript",
        "T8_IndexTTS25_DialogueGenerate",
        "T8_IndexTTS25_Environment",
    ]
    assert all(schema.category == "T8star-Aix/Audio/IndexTTS 2.5" for schema in schemas)
    assert schemas[4].outputs[0].io_type == "AUDIO"
    assert schemas[8].outputs[0].io_type == "AUDIO"
    dialogue_script_input = next(item for item in schemas[7].inputs if item.id == "script")
    assert dialogue_script_input.dynamic_prompts is False
    assert dialogue_script_input.as_dict()["dynamicPrompts"] is False
    assert not hasattr(plugin, "NODE_CLASS_MAPPINGS")


def test_pronunciation_node_outputs_portable_annotated_text():
    _load_plugin()
    from comfyui_indextts25_t8_test.nodes_v3 import T8IndexTTS25Pronunciation

    result = T8IndexTTS25Pronunciation.execute(
        "银行的行长到了。",
        "ZH",
        "银行|YIN2 HANG2|ZH\n行长|HANG2 ZHANG3|ZH",
        True,
    )
    assert result[0] == "<银行|YIN2 HANG2>的<行长|HANG2 ZHANG3>到了。"
    assert "已应用 2 处" in result[1]


def test_emotion_vector_is_safely_normalized():
    plugin = _load_plugin()
    from comfyui_indextts25_t8_test.nodes_v3 import T8IndexTTS25EmotionControl

    result = T8IndexTTS25EmotionControl.execute(
        {
            "mode": "vector",
            "happy": 1.0,
            "angry": 1.0,
            "sad": 0.0,
            "afraid": 0.0,
            "disgusted": 0.0,
            "melancholic": 0.0,
            "surprised": 0.0,
            "calm": 0.0,
            "strength": 1.0,
            "use_random": False,
        }
    )
    emotion = result[0]
    assert sum(emotion.vector) == pytest.approx(0.8)
    assert emotion.notes
