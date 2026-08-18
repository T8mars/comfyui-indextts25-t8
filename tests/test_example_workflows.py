from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "01_basic_voice_clone",
    "02_speed_comparison",
    "03_emotion_reference_audio",
    "04_emotion_vector",
    "05_emotion_text",
    "06_random_sampling_long_text",
    "07_multilingual_generation",
    "08_chinese_pronunciation",
    "09_english_cmu_pronunciation",
    "10_japanese_kana_pronunciation",
    "11_multi_role_dialogue",
    "12_batch_dialogue_json",
    "13_srt_multi_role",
    "14_optional_acceleration",
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_ui_and_api_examples_are_present_and_valid():
    ui_root = PLUGIN_ROOT / "example_workflows" / "ui"
    api_root = PLUGIN_ROOT / "example_workflows" / "api"
    assert {path.stem for path in ui_root.glob("*.json")} == set(EXAMPLES)
    assert {path.stem for path in api_root.glob("*.json")} == set(EXAMPLES)

    for name in EXAMPLES:
        workflow = _load(ui_root / f"{name}.json")
        prompt = _load(api_root / f"{name}.json")
        assert workflow["version"] == 0.4
        assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
        assert workflow["last_link_id"] == len(workflow["links"])
        assert any(node["type"] in {"T8_IndexTTS25_Generate", "T8_IndexTTS25_DialogueGenerate"} for node in workflow["nodes"])
        assert any(node["class_type"] in {"T8_IndexTTS25_Generate", "T8_IndexTTS25_DialogueGenerate"} for node in prompt.values())


def test_examples_cover_every_emotion_mode_speed_sampling_and_language():
    api_root = PLUGIN_ROOT / "example_workflows" / "api"
    prompts = [_load(api_root / f"{name}.json") for name in EXAMPLES]
    emotion_modes = {
        node["inputs"]["mode"]
        for prompt in prompts
        for node in prompt.values()
        if node["class_type"] == "T8_IndexTTS25_EmotionControl"
    }
    assert emotion_modes == {"reference_audio", "vector", "text"}

    generation_inputs = [
        node["inputs"]
        for prompt in prompts
        for node in prompt.values()
        if node["class_type"] == "T8_IndexTTS25_Generate"
    ]
    assert {item["language"] for item in generation_inputs} == {"ZH", "EN", "JA", "ES", "AR"}
    assert {item["duration_factor"] for item in generation_inputs}.issuperset({0.7, 1.0, 1.3})
    assert any(
        node["class_type"] == "T8_IndexTTS25_SamplingConfig" and node["inputs"]["do_sample"]
        for prompt in prompts
        for node in prompt.values()
    )
    pronunciation_nodes = [
        node
        for prompt in prompts
        for node in prompt.values()
        if node["class_type"] == "T8_IndexTTS25_Pronunciation"
    ]
    assert {node["inputs"]["language"] for node in pronunciation_nodes} == {"ZH", "EN", "JA"}
    assert any("银行|YIN2 HANG2|ZH" in node["inputs"]["dictionary"] for node in pronunciation_nodes)


def test_v3_dynamic_combo_api_inputs_are_flattened():
    api_root = PLUGIN_ROOT / "example_workflows" / "api"
    for name in EXAMPLES:
        prompt = _load(api_root / f"{name}.json")
        for node in prompt.values():
            if node["class_type"] != "T8_IndexTTS25_EmotionControl":
                continue
            assert isinstance(node["inputs"]["mode"], str)
            assert all(not isinstance(value, dict) for value in node["inputs"].values())


def test_api_prompts_expand_with_the_current_comfyui_v3_schema():
    _io = pytest.importorskip("comfy_api.latest")._io

    package_name = "comfyui_indextts25_t8_workflow_test"
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(PLUGIN_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    nodes_module = sys.modules[f"{package_name}.nodes_v3"]

    node_classes = {
        "T8_IndexTTS25_ModelLoader": nodes_module.T8IndexTTS25ModelLoader,
        "T8_IndexTTS25_EmotionControl": nodes_module.T8IndexTTS25EmotionControl,
        "T8_IndexTTS25_SamplingConfig": nodes_module.T8IndexTTS25SamplingConfig,
        "T8_IndexTTS25_Pronunciation": nodes_module.T8IndexTTS25Pronunciation,
        "T8_IndexTTS25_Generate": nodes_module.T8IndexTTS25Generate,
        "T8_IndexTTS25_VoiceProfile": nodes_module.T8IndexTTS25VoiceProfile,
        "T8_IndexTTS25_RoleLibrary": nodes_module.T8IndexTTS25RoleLibrary,
        "T8_IndexTTS25_DialogueScript": nodes_module.T8IndexTTS25DialogueScript,
        "T8_IndexTTS25_DialogueGenerate": nodes_module.T8IndexTTS25DialogueGenerate,
        "T8_IndexTTS25_Environment": nodes_module.T8IndexTTS25Environment,
    }
    api_root = PLUGIN_ROOT / "example_workflows" / "api"
    for name in EXAMPLES:
        prompt = _load(api_root / f"{name}.json")
        for node in prompt.values():
            node_class = node_classes.get(node["class_type"])
            if node_class is None:
                continue
            live_inputs = node["inputs"]
            finalized, _, v3_data = _io.get_finalized_class_inputs(node_class.INPUT_TYPES(), live_inputs)
            recognized = set(finalized.get("required", {})) | set(finalized.get("optional", {}))
            assert set(live_inputs).issubset(recognized)
            nested = _io.build_nested_inputs(live_inputs, v3_data)
            if node["class_type"] == "T8_IndexTTS25_EmotionControl":
                assert isinstance(nested["mode"], dict)
                assert nested["mode"]["mode"] == live_inputs["mode"]
            if node["class_type"] == "T8_IndexTTS25_RoleLibrary":
                assert set(nested["voices"]) == {key.split(".", 1)[1] for key in live_inputs if key.startswith("voices.voice_")}


def test_examples_cover_multi_role_batch_srt_and_optional_acceleration():
    api_root = PLUGIN_ROOT / "example_workflows" / "api"
    prompts = {name: _load(api_root / f"{name}.json") for name in EXAMPLES}
    dialogue_types = {
        node["inputs"]["script_type"]
        for prompt in prompts.values()
        for node in prompt.values()
        if node["class_type"] == "T8_IndexTTS25_DialogueScript"
    }
    assert dialogue_types == {"batch", "srt"}
    assert any(
        node["class_type"] == "T8_IndexTTS25_RoleLibrary" and len(node["inputs"]) >= 2
        for prompt in prompts.values()
        for node in prompt.values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_DialogueGenerate" and node["inputs"]["fit_srt_slots"]
        for node in prompts["13_srt_multi_role"].values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_ModelLoader" and node["inputs"]["acceleration_mode"] == "auto_safe"
        for node in prompts["14_optional_acceleration"].values()
    )
