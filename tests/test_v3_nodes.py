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
        "T8_IndexTTS25_TextPreview",
        "T8_IndexTTS25_Pronunciation",
        "T8_IndexTTS25_Generate",
        "T8_IndexTTS25_VoiceProfile",
        "T8_IndexTTS25_RoleLibrary",
        "T8_IndexTTS25_MergeVoiceEmotions",
        "T8_IndexTTS25_DialogueScript",
        "T8_IndexTTS25_TimelineEditor",
        "T8_IndexTTS25_DialogueGenerate",
        "T8_IndexTTS25_ASRProofread",
        "T8_IndexTTS25_SubtitleRewrite",
        "T8_IndexTTS25_AudioPostProcess",
        "T8_IndexTTS25_Environment",
    ]
    assert all(schema.category == "T8star-Aix/Audio/IndexTTS 2.5" for schema in schemas)
    assert schemas[5].outputs[0].io_type == "AUDIO"
    assert schemas[11].outputs[0].io_type == "AUDIO"
    assert schemas[14].outputs[0].io_type == "AUDIO"
    dialogue_script_input = next(item for item in schemas[9].inputs if item.id == "script")
    assert dialogue_script_input.dynamic_prompts is False
    assert dialogue_script_input.as_dict()["dynamicPrompts"] is False
    assert not hasattr(plugin, "NODE_CLASS_MAPPINGS")


def test_timeline_asr_and_subtitle_nodes_form_a_complete_editing_chain(monkeypatch):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.dialogue import DialogueLine
    from comfyui_indextts25_t8_test.runtime.types import DialogueScript

    script = DialogueScript([DialogueLine(1, "旁白", "原始字幕", "ZH", 0, 1000, 1.0)], "srt")
    edited = nodes_v3.T8IndexTTS25TimelineEditor.execute(
        script,
        '[{"index":1,"role":"旁白","language":"ZH","start_ms":100,"end_ms":900,"duration_factor":1.0,"text":"修改字幕"}]',
    )
    assert edited[0].lines[0].start_ms == 100
    assert "milliseconds" in edited[1]
    assert tuple(edited[2].shape) == (1, 96, 1200, 3)

    monkeypatch.setattr(nodes_v3, "asr_available", lambda *args: True)
    monkeypatch.setattr(
        nodes_v3,
        "transcribe_waveform",
        lambda *args, **kwargs: {"text": "修改字幕", "model": "tiny", "device": "cpu"},
    )
    asr = nodes_v3.T8IndexTTS25ASRProofread.execute(
        {"waveform": __import__("torch").zeros(1, 1, 16000), "sample_rate": 16000},
        "修改字幕",
        "ZH",
        "auto",
        "tiny",
        "cpu",
        0.8,
    )
    assert asr[0] == "修改字幕"
    assert asr[1] is True
    assert asr[2] == pytest.approx(1.0)

    generation_report = '{"lines":[{"index":1,"timeline":{"actual_start_ms":150,"actual_end_ms":950},"asr":{"recognized_text":"识别字幕","passed":true}}]}'
    rewritten = nodes_v3.T8IndexTTS25SubtitleRewrite.execute(
        edited[0], generation_report, "actual", "asr_passed", True
    )
    assert "00:00:00,150 --> 00:00:00,950" in rewritten[0]
    assert "[旁白] 识别字幕" in rewritten[0]


def test_dialogue_generation_can_auto_review_and_return_rewritten_srt(tmp_path, monkeypatch):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.dialogue import DialogueLine
    from comfyui_indextts25_t8_test.runtime.types import DialogueScript, ModelHandle, RoleLibrary, VoiceProfile

    audio = {"waveform": __import__("torch").zeros(1, 1, 22050), "sample_rate": 22050}
    monkeypatch.setattr(nodes_v3, "run_inference", lambda *args, **kwargs: (audio, "fake inference"))
    monkeypatch.setattr(nodes_v3, "asr_available", lambda *args: True)
    monkeypatch.setattr(
        nodes_v3,
        "transcribe_waveform",
        lambda *args, **kwargs: {"text": "自动校对字幕", "model": "tiny", "device": "cpu"},
    )
    script = DialogueScript([DialogueLine(1, "角色A", "自动校对字幕", "ZH", 100, 1100)], "srt")
    library = RoleLibrary({"角色A": VoiceProfile("角色A", audio, "ZH")})
    result = nodes_v3.T8IndexTTS25DialogueGenerate.execute(
        ModelHandle(tmp_path, "cpu", False),
        library,
        script,
        1,
        "overlay",
        False,
        "native",
        180,
        200,
        "off",
        1.0,
        True,
        "auto",
        "tiny",
        "cpu",
        0.8,
        "actual",
        "asr_passed",
        True,
    )
    report = __import__("json").loads(result[2])
    assert report["lines"][0]["asr"]["passed"] is True
    assert "00:00:00,100 --> 00:00:01,100" in result[3]
    assert "[角色A] 自动校对字幕" in result[3]
    assert "\"reports\"" in result[4]


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
    _load_plugin()
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


def test_role_library_and_merge_alias_preserve_each_roles_emotion():
    _load_plugin()
    from comfyui_indextts25_t8_test.nodes_v3 import (
        T8IndexTTS25MergeVoiceEmotions,
        T8IndexTTS25RoleLibrary,
    )
    from comfyui_indextts25_t8_test.runtime.types import EmotionConfig, VoiceProfile

    audio = {"waveform": __import__("torch").zeros(1, 1, 22050), "sample_rate": 22050}
    happy = EmotionConfig(mode="vector", vector=(0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2))
    tense = EmotionConfig(mode="text", text="克制而紧张", strength=0.7)
    voices = {
        "voice_0": VoiceProfile("角色A", audio, "ZH", happy),
        "voice_1": [VoiceProfile("角色B", audio, "ZH", tense)],
    }

    regular = T8IndexTTS25RoleLibrary.execute(voices)
    compatible = T8IndexTTS25MergeVoiceEmotions.execute(voices)

    for result in (regular, compatible):
        library, info = result
        assert library.profiles["角色A"].emotion is happy
        assert library.profiles["角色B"].emotion is tense
        assert "角色A（八维向量）" in info
        assert "角色B（文本描述）" in info


def test_role_emotion_merge_rejects_duplicate_names():
    _load_plugin()
    from comfyui_indextts25_t8_test.nodes_v3 import T8IndexTTS25MergeVoiceEmotions
    from comfyui_indextts25_t8_test.runtime.types import VoiceProfile

    audio = {"waveform": __import__("torch").zeros(1, 1, 1), "sample_rate": 22050}
    with pytest.raises(ValueError, match="角色名称重复"):
        T8IndexTTS25MergeVoiceEmotions.execute(
            {
                "voice_0": VoiceProfile("同名", audio),
                "voice_1": VoiceProfile("同名", audio),
            }
        )


@pytest.mark.parametrize("count", [1, 2, 16])
def test_role_emotion_merge_accepts_documented_role_counts(count):
    _load_plugin()
    from comfyui_indextts25_t8_test.nodes_v3 import T8IndexTTS25MergeVoiceEmotions
    from comfyui_indextts25_t8_test.runtime.types import VoiceProfile

    audio = {"waveform": __import__("torch").zeros(1, 1, 1), "sample_rate": 22050}
    result = T8IndexTTS25MergeVoiceEmotions.execute(
        {f"voice_{index}": VoiceProfile(f"角色{index}", audio) for index in range(count)}
    )
    assert len(result[0].profiles) == count


def test_dialogue_generation_routes_each_roles_emotion_without_leaking(tmp_path, monkeypatch):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.dialogue import DialogueLine
    from comfyui_indextts25_t8_test.runtime.types import (
        DialogueScript,
        EmotionConfig,
        ModelHandle,
        RoleLibrary,
        VoiceProfile,
    )

    audio = {"waveform": __import__("torch").zeros(1, 1, 2205), "sample_rate": 22050}
    emotions = (
        EmotionConfig(mode="speaker"),
        EmotionConfig(mode="reference_audio", reference_audio=audio, strength=0.7),
        EmotionConfig(mode="vector", vector=(0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2)),
        EmotionConfig(mode="text", text="克制而紧张", strength=0.8),
    )
    routed = []

    def fake_inference(*_args, **kwargs):
        routed.append(kwargs["emotion"])
        return audio, "fake inference"

    monkeypatch.setattr(nodes_v3, "run_inference", fake_inference)
    library = RoleLibrary(
        {
            f"角色{index}": VoiceProfile(f"角色{index}", audio, "ZH", emotion)
            for index, emotion in enumerate(emotions)
        }
    )
    script = DialogueScript(
        [
            DialogueLine(index + 1, f"角色{index}", f"第{index + 1}句", "ZH")
            for index in range(len(emotions))
        ],
        "batch",
    )
    nodes_v3.T8IndexTTS25DialogueGenerate.execute(
        ModelHandle(tmp_path, "cpu", False),
        library,
        script,
        1,
        "shift",
        False,
        "native",
        180,
        0,
        "off",
        1.0,
        False,
        "auto",
        "base",
        "cpu",
        0.82,
        "actual",
        "original",
        True,
    )
    assert routed == list(emotions)


def test_sampling_exposes_auto_segmentation_and_real_pause_controls():
    _load_plugin()
    from comfyui_indextts25_t8_test.nodes_v3 import T8IndexTTS25SamplingConfig

    result = T8IndexTTS25SamplingConfig.execute(
        False, 0.8, 0.8, 30, 3, 10.0, 0.0, 1500,
        25, 0.7, 1.0,
        "auto", 120, 200, "narration", 100, 300, 600, True,
    )
    config = result[0]
    assert config.effective_segment_tokens("EN") == 60
    assert config.effective_segment_tokens("ZH") == 120
    assert config.pause_preset == "narration"
    assert config.diffusion_steps == 25
    assert config.inference_cfg_rate == pytest.approx(0.7)
    assert config.cfm_temperature == pytest.approx(1.0)
