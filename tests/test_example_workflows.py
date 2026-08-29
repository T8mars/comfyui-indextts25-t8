from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest

from runtime.dialogue import parse_batch_script, parse_srt
from scripts import build_example_workflows


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
    "15_auto_segment_preview",
    "16_pause_control",
    "17_target_duration",
    "18_audio_postprocess",
    "19_cfm_advanced",
    "20_asr_proofread",
    "21_timeline_editor",
    "22_subtitle_rewrite",
    "23_multi_role_emotions",
    "24_reference_quality",
    "25_quality_retry",
    "26_memory_control",
    "27_audiocpp_experimental",
    "28_low_vram_fp16",
    "29_runtime_benchmark",
    "30_update_check",
    "31_per_line_emotion",
    "32_context_emotion_suggestions",
    "33_saved_voice_library",
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
        generation_types = {
            "T8_IndexTTS25_Generate",
            "T8_IndexTTS25_DialogueGenerate",
            "T8_IndexTTS25_AudioPostProcess",
            "T8_IndexTTS25_AudioCppGenerate",
            "T8_IndexTTS25_RuntimeBenchmark",
            "T8_IndexTTS25_UpdateCheck",
            "T8_IndexTTS25_DialogueEmotionSuggest",
        }
        assert any(node["type"] in generation_types for node in workflow["nodes"])
        assert any(node["class_type"] in generation_types for node in prompt.values())


def test_ui_widget_values_keep_declared_order_types_and_release_version():
    ui_root = PLUGIN_ROOT / "example_workflows" / "ui"
    for name in EXAMPLES:
        workflow = _load(ui_root / f"{name}.json")
        build_example_workflows.validate_ui(workflow)
        for node in workflow["nodes"]:
            if node["properties"].get("cnr_id") == "comfyui-indextts25-t8":
                assert (
                    node["properties"]["ver"] == build_example_workflows.PROJECT_VERSION
                )
            elif node["properties"].get("cnr_id") == "comfy-core":
                assert "ver" not in node["properties"]


def test_embedded_dialogue_scripts_are_parsed_not_just_outer_workflow_json():
    ui_root = PLUGIN_ROOT / "example_workflows" / "ui"
    api_root = PLUGIN_ROOT / "example_workflows" / "api"
    for name in (
        "11_multi_role_dialogue",
        "12_batch_dialogue_json",
        "13_srt_multi_role",
        "21_timeline_editor",
        "22_subtitle_rewrite",
        "23_multi_role_emotions",
    ):
        workflow = _load(ui_root / f"{name}.json")
        ui_node = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "T8_IndexTTS25_DialogueScript"
        )
        script_type, script, default_role, default_language = ui_node["widgets_values"]
        parser = parse_srt if script_type == "srt" else parse_batch_script
        assert parser(script, default_role, default_language)

        prompt = _load(api_root / f"{name}.json")
        api_inputs = next(
            node["inputs"]
            for node in prompt.values()
            if node["class_type"] == "T8_IndexTTS25_DialogueScript"
        )
        parser = parse_srt if api_inputs["script_type"] == "srt" else parse_batch_script
        assert parser(
            api_inputs["script"],
            api_inputs["default_role"],
            api_inputs["default_language"],
        )


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
    assert {item["language"] for item in generation_inputs} == {
        "ZH",
        "EN",
        "JA",
        "ES",
        "AR",
    }
    assert {item["duration_factor"] for item in generation_inputs}.issuperset(
        {0.7, 1.0, 1.3}
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_SamplingConfig"
        and node["inputs"]["do_sample"]
        for prompt in prompts
        for node in prompt.values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_SamplingConfig"
        and node["inputs"]["diffusion_steps"] == 40
        and node["inputs"]["inference_cfg_rate"] == 0.85
        and node["inputs"]["cfm_temperature"] == 0.8
        for node in _load(api_root / "19_cfm_advanced.json").values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_Generate"
        and node["inputs"]["target_duration_mode"] == "native"
        for node in _load(api_root / "17_target_duration.json").values()
    )
    pronunciation_nodes = [
        node
        for prompt in prompts
        for node in prompt.values()
        if node["class_type"] == "T8_IndexTTS25_Pronunciation"
    ]
    assert {node["inputs"]["language"] for node in pronunciation_nodes} == {
        "ZH",
        "EN",
        "JA",
    }
    assert any(
        "银行|YIN2 HANG2|ZH" in node["inputs"]["dictionary"]
        for node in pronunciation_nodes
    )


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
        "T8_IndexTTS25_TextPreview": nodes_module.T8IndexTTS25TextPreview,
        "T8_IndexTTS25_Pronunciation": nodes_module.T8IndexTTS25Pronunciation,
        "T8_IndexTTS25_Generate": nodes_module.T8IndexTTS25Generate,
        "T8_IndexTTS25_SavedVoice": nodes_module.T8IndexTTS25SavedVoice,
        "T8_IndexTTS25_VoiceProfile": nodes_module.T8IndexTTS25VoiceProfile,
        "T8_IndexTTS25_RoleLibrary": nodes_module.T8IndexTTS25RoleLibrary,
        "T8_IndexTTS25_MergeVoiceEmotions": nodes_module.T8IndexTTS25MergeVoiceEmotions,
        "T8_IndexTTS25_DialogueScript": nodes_module.T8IndexTTS25DialogueScript,
        "T8_IndexTTS25_DialogueEmotionSuggest": nodes_module.T8IndexTTS25DialogueEmotionSuggest,
        "T8_IndexTTS25_TimelineEditor": nodes_module.T8IndexTTS25TimelineEditor,
        "T8_IndexTTS25_DialogueGenerate": nodes_module.T8IndexTTS25DialogueGenerate,
        "T8_IndexTTS25_ASRProofread": nodes_module.T8IndexTTS25ASRProofread,
        "T8_IndexTTS25_SubtitleRewrite": nodes_module.T8IndexTTS25SubtitleRewrite,
        "T8_IndexTTS25_ReferenceQuality": nodes_module.T8IndexTTS25ReferenceQuality,
        "T8_IndexTTS25_MemoryControl": nodes_module.T8IndexTTS25MemoryControl,
        "T8_IndexTTS25_AudioCppGenerate": nodes_module.T8IndexTTS25AudioCppGenerate,
        "T8_IndexTTS25_AudioPostProcess": nodes_module.T8IndexTTS25AudioPostProcess,
        "T8_IndexTTS25_Environment": nodes_module.T8IndexTTS25Environment,
        "T8_IndexTTS25_RuntimeBenchmark": nodes_module.T8IndexTTS25RuntimeBenchmark,
        "T8_IndexTTS25_UpdateCheck": nodes_module.T8IndexTTS25UpdateCheck,
    }
    api_root = PLUGIN_ROOT / "example_workflows" / "api"
    for name in EXAMPLES:
        prompt = _load(api_root / f"{name}.json")
        for node in prompt.values():
            node_class = node_classes.get(node["class_type"])
            if node_class is None:
                continue
            live_inputs = node["inputs"]
            finalized, _, v3_data = _io.get_finalized_class_inputs(
                node_class.INPUT_TYPES(), live_inputs
            )
            recognized = set(finalized.get("required", {})) | set(
                finalized.get("optional", {})
            )
            assert set(live_inputs).issubset(recognized)
            nested = _io.build_nested_inputs(live_inputs, v3_data)
            if node["class_type"] == "T8_IndexTTS25_EmotionControl":
                assert isinstance(nested["mode"], dict)
                assert nested["mode"]["mode"] == live_inputs["mode"]
            if node["class_type"] in {
                "T8_IndexTTS25_RoleLibrary",
                "T8_IndexTTS25_MergeVoiceEmotions",
            }:
                assert set(nested["voices"]) == {
                    key.split(".", 1)[1]
                    for key in live_inputs
                    if key.startswith("voices.voice_")
                }

    ui_root = PLUGIN_ROOT / "example_workflows" / "ui"
    for name in EXAMPLES:
        workflow = _load(ui_root / f"{name}.json")
        for node in workflow["nodes"]:
            node_class = node_classes.get(node["type"])
            if node_class is None or node["type"] == "T8_IndexTTS25_EmotionControl":
                continue
            live = node_class.INPUT_TYPES()
            live_order = list(live.get("required", {})) + list(live.get("optional", {}))
            ui_widgets = [item for item in node["inputs"] if "widget" in item]
            ui_names = [item["name"] for item in ui_widgets]
            assert ui_names == [item for item in live_order if item in ui_names]
            for item in ui_widgets:
                live_type = (live.get("required", {}) | live.get("optional", {}))[item["name"]][0]
                assert item["type"] == live_type


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
        node["class_type"] == "T8_IndexTTS25_DialogueGenerate"
        and node["inputs"]["fit_srt_slots"]
        for node in prompts["13_srt_multi_role"].values()
    )
    srt_script = next(
        node["inputs"]
        for node in prompts["13_srt_multi_role"].values()
        if node["class_type"] == "T8_IndexTTS25_DialogueScript"
    )
    assert srt_script["default_language"] == "ES"
    assert len(srt_script["script"].split()) >= 30
    long_english = next(
        node["inputs"]
        for node in prompts["06_random_sampling_long_text"].values()
        if node["class_type"] == "T8_IndexTTS25_Generate"
    )
    assert long_english["language"] == "EN"
    assert len(long_english["text"].split()) >= 60
    assert any(
        node["class_type"] == "T8_IndexTTS25_ModelLoader"
        and node["inputs"]["acceleration_mode"] == "auto_safe"
        for node in prompts["14_optional_acceleration"].values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_ASRProofread"
        for node in prompts["20_asr_proofread"].values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_TimelineEditor"
        for node in prompts["21_timeline_editor"].values()
    )
    assert any(
        node["class_type"] == "PreviewImage" and node["inputs"]["images"] == ["8", 2]
        for node in prompts["21_timeline_editor"].values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_SubtitleRewrite"
        for node in prompts["22_subtitle_rewrite"].values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_ReferenceQuality"
        for node in prompts["24_reference_quality"].values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_Generate"
        and node["inputs"]["quality_retry_count"] == 2
        for node in prompts["25_quality_retry"].values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_ModelLoader"
        and node["inputs"]["recycle_after_runs"] == 20
        for node in prompts["26_memory_control"].values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_MemoryControl"
        and node["inputs"]["action"] == "reference_cache_status"
        for node in prompts["26_memory_control"].values()
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_AudioCppGenerate"
        for node in prompts["27_audiocpp_experimental"].values()
    )
    low_vram_loader = next(
        node["inputs"]
        for node in prompts["28_low_vram_fp16"].values()
        if node["class_type"] == "T8_IndexTTS25_ModelLoader"
    )
    assert low_vram_loader["precision"] == "float16"
    assert low_vram_loader["reference_device"] == "cpu"
    assert low_vram_loader["reuse_spk_cond_for_emo"] is True
    role_emotions = prompts["23_multi_role_emotions"]
    assert (
        sum(
            node["class_type"] == "T8_IndexTTS25_EmotionControl"
            for node in role_emotions.values()
        )
        == 2
    )
    assert any(
        node["class_type"] == "T8_IndexTTS25_MergeVoiceEmotions"
        and len(node["inputs"]) == 2
        for node in role_emotions.values()
    )
    voice_emotion_links = [
        node["inputs"].get("emotion")
        for node in role_emotions.values()
        if node["class_type"] == "T8_IndexTTS25_VoiceProfile"
    ]
    assert voice_emotion_links == [["4", 0], ["5", 0]]
    context_suggestion = prompts["32_context_emotion_suggestions"]
    suggestion_inputs = next(
        node["inputs"]
        for node in context_suggestion.values()
        if node["class_type"] == "T8_IndexTTS25_DialogueEmotionSuggest"
    )
    assert suggestion_inputs["context_window"] == 2
    assert suggestion_inputs["overwrite_existing"] is False
    assert not any(
        node["class_type"] == "T8_IndexTTS25_DialogueGenerate"
        for node in context_suggestion.values()
    )
