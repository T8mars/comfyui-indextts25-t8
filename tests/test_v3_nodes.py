from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


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
        "T8_IndexTTS25_SavedVoice",
        "T8_IndexTTS25_VoiceProfile",
        "T8_IndexTTS25_RoleLibrary",
        "T8_IndexTTS25_MergeVoiceEmotions",
        "T8_IndexTTS25_DialogueScript",
        "T8_IndexTTS25_DialogueEmotionSuggest",
        "T8_IndexTTS25_TimelineEditor",
        "T8_IndexTTS25_DialogueGenerate",
        "T8_IndexTTS25_ASRProofread",
        "T8_IndexTTS25_SubtitleRewrite",
        "T8_IndexTTS25_ReferenceQuality",
        "T8_IndexTTS25_MemoryControl",
        "T8_IndexTTS25_AudioCppGenerate",
        "T8_IndexTTS25_AudioPostProcess",
        "T8_IndexTTS25_Environment",
        "T8_IndexTTS25_UpdateCheck",
        "T8_IndexTTS25_RuntimeBenchmark",
    ]
    assert all(schema.category == "T8star-Aix/Audio/IndexTTS 2.5" for schema in schemas)
    schemas_by_id = {schema.node_id: schema for schema in schemas}
    assert schemas_by_id["T8_IndexTTS25_Generate"].outputs[0].io_type == "AUDIO"
    assert schemas_by_id["T8_IndexTTS25_DialogueGenerate"].outputs[0].io_type == "AUDIO"
    assert schemas_by_id["T8_IndexTTS25_ReferenceQuality"].outputs[0].io_type == "AUDIO"
    assert schemas_by_id["T8_IndexTTS25_MemoryControl"].outputs[0].io_type == "STRING"
    assert schemas_by_id["T8_IndexTTS25_AudioCppGenerate"].outputs[0].io_type == "AUDIO"
    dialogue_script_input = next(
        item for item in schemas_by_id["T8_IndexTTS25_DialogueScript"].inputs if item.id == "script"
    )
    assert dialogue_script_input.dynamic_prompts is False
    assert dialogue_script_input.as_dict()["dynamicPrompts"] is False
    dialogue_outputs = schemas_by_id["T8_IndexTTS25_DialogueScript"].outputs
    assert [item.id for item in dialogue_outputs] == [
        "dialogue_script",
        "script_preview",
        "human_script",
    ]
    assert "无需手写 JSON" in schemas_by_id["T8_IndexTTS25_DialogueScript"].description
    suggestion_schema = schemas_by_id["T8_IndexTTS25_DialogueEmotionSuggest"]
    suggestion_inputs = {item.id: item for item in suggestion_schema.inputs}
    assert suggestion_inputs["context_window"].as_dict()["default"] == 2
    assert suggestion_inputs["overwrite_existing"].as_dict()["default"] is False
    assert "不会生成音频" in suggestion_schema.description
    audiocpp_backend = next(
        item for item in schemas_by_id["T8_IndexTTS25_AudioCppGenerate"].inputs if item.id == "backend"
    )
    assert "metal" in audiocpp_backend.as_dict()["options"]
    loader_inputs = {
        item.id: item for item in schemas_by_id["T8_IndexTTS25_ModelLoader"].inputs
    }
    assert "float16" in loader_inputs["precision"].as_dict()["options"]
    assert loader_inputs["reference_device"].as_dict()["default"] == "auto"
    assert loader_inputs["reuse_spk_cond_for_emo"].as_dict()["default"] is False
    assert loader_inputs["download_missing"].as_dict()["default"] is False
    assert loader_inputs["accept_model_license"].as_dict()["default"] is False
    sampling_inputs = {
        item.id: item
        for item in schemas_by_id["T8_IndexTTS25_SamplingConfig"].inputs
    }
    normalization_input = sampling_inputs["text_normalization"].as_dict()
    assert normalization_input["display_name"] == "文本归一化（数字/日期）"
    assert "1939年" in normalization_input["tooltip"]
    assert "数字/日期" in schemas_by_id["T8_IndexTTS25_Environment"].description
    assert not hasattr(plugin, "NODE_CLASS_MAPPINGS")


def test_environment_report_includes_verified_text_normalization():
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3

    result = nodes_v3.T8IndexTTS25Environment.execute("cpu")
    report = json.loads(result[0])

    assert report["text_normalization"]["verified"] is True
    assert report["text_normalization"]["example_output"] == "一九三九年"


def test_audiocpp_generation_uses_explicit_local_paths(
    tmp_path,
    monkeypatch,
):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3

    executable = tmp_path / "audiocpp_cli.exe"
    model = tmp_path / "model.gguf"
    executable.write_bytes(b"exe")
    model.write_bytes(b"model")
    captured = {}

    async def fake_probe(_path):
        return {"available": True, "summary": "ok"}

    async def fake_run(*args, **kwargs):
        captured["backend"] = kwargs["backend"]
        Path(args[3]).write_bytes(b"wav")
        return {"backend": kwargs["backend"]}

    monkeypatch.setattr(nodes_v3, "probe_audiocpp", fake_probe)
    monkeypatch.setattr(nodes_v3, "run_audiocpp", fake_run)
    monkeypatch.setattr(
        nodes_v3,
        "comfy_audio_to_reference_wav",
        lambda *_args, **_kwargs: (tmp_path / "speaker.wav", []),
    )
    monkeypatch.setattr(
        nodes_v3.torchaudio,
        "load",
        lambda _path: (torch.zeros((1, 100)), 24000),
    )

    result = asyncio.run(
        nodes_v3.T8IndexTTS25AudioCppGenerate.execute(
            str(executable),
            str(model),
            {"waveform": torch.zeros(1, 1, 100), "sample_rate": 24000},
            "测试",
            "ZH",
            "cuda",
            1.0,
            True,
        )
    )

    assert captured["backend"] == "cuda"
    assert json.loads(result[1])["backend"] == "cuda"


def test_audiocpp_generation_requires_manual_local_paths():
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3

    with pytest.raises(RuntimeError, match="不会联网安装组件"):
        asyncio.run(
            nodes_v3.T8IndexTTS25AudioCppGenerate.execute(
                "",
                "",
                {"waveform": torch.zeros(1, 1, 100), "sample_rate": 24000},
                "测试",
                "ZH",
                "cuda",
                1.0,
                True,
            )
        )


def test_context_emotion_node_returns_editable_script_without_generating_audio(
    tmp_path, monkeypatch
):
    _load_plugin()
    import threading
    from types import SimpleNamespace

    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.dialogue import DialogueLine
    from comfyui_indextts25_t8_test.runtime.types import DialogueScript, ModelHandle

    class FakeQwenEmotion:
        @staticmethod
        def inference(prompt):
            assert str(prompt).strip()
            return {
                "angry": 0.9,
                "surprised": 0.3,
                "calm": 0.0,
            }

    class FakeCore:
        def __init__(self):
            self.qwen_emo = None

        def ensure_qwen_emotion(self):
            self.qwen_emo = FakeQwenEmotion()

    core = FakeCore()
    entry = SimpleNamespace(model=core, lock=threading.RLock())
    done_calls = []
    monkeypatch.setattr(nodes_v3.MODEL_CACHE, "acquire", lambda _handle: entry)
    monkeypatch.setattr(
        nodes_v3.MODEL_CACHE,
        "done",
        lambda handle, cache_entry, release=False: done_calls.append(
            (handle, cache_entry, release)
        ),
    )
    handle = ModelHandle(tmp_path, "cpu", False, low_vram=True)
    script = DialogueScript(
        [
            DialogueLine(1, "角色A", "你为什么骗我？", "ZH"),
            DialogueLine(
                2,
                "角色B",
                "我只是想保护你。",
                "ZH",
                emotion_mode="text",
                emotion_text="克制而难过",
            ),
        ],
        "batch",
    )

    result = nodes_v3.T8IndexTTS25DialogueEmotionSuggest.execute(
        handle, script, 1, False
    )

    suggested_script = result[0]
    report = json.loads(result[1])
    assert suggested_script.lines[0].emotion_mode == "vector"
    assert suggested_script.lines[0].emotion_vector[1] == pytest.approx(0.6)
    assert suggested_script.lines[1].emotion_mode == "text"
    assert report["classified_count"] == 1
    assert report["preserved_count"] == 1
    assert report["started_synthesis"] is False
    assert report["requires_user_confirmation"] is True
    assert report["temporary_qwen_released"] is True
    assert "尚未合成音频" in result[2]
    assert core.qwen_emo is None
    assert done_calls == [(handle, entry, False)]


def test_context_emotion_node_releases_temporary_qwen_after_analysis_error(
    tmp_path, monkeypatch
):
    _load_plugin()
    import threading
    from types import SimpleNamespace

    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.dialogue import DialogueLine
    from comfyui_indextts25_t8_test.runtime.types import DialogueScript, ModelHandle

    class FailingQwenEmotion:
        @staticmethod
        def inference(_prompt):
            raise RuntimeError("fake classifier failure")

    class FakeCore:
        qwen_emo = None

        def ensure_qwen_emotion(self):
            self.qwen_emo = FailingQwenEmotion()

    core = FakeCore()
    entry = SimpleNamespace(model=core, lock=threading.RLock())
    done_calls = []
    monkeypatch.setattr(nodes_v3.MODEL_CACHE, "acquire", lambda _handle: entry)
    monkeypatch.setattr(
        nodes_v3.MODEL_CACHE,
        "done",
        lambda *_args, **_kwargs: done_calls.append(True),
    )
    handle = ModelHandle(tmp_path, "cpu", False, low_vram=True)
    script = DialogueScript([DialogueLine(1, "角色A", "测试。", "ZH")])

    with pytest.raises(RuntimeError, match="fake classifier failure"):
        nodes_v3.T8IndexTTS25DialogueEmotionSuggest.execute(
            handle, script, 1, False
        )

    assert core.qwen_emo is None
    assert done_calls == [True]


def test_v010_model_loader_widget_values_are_restored():
    _load_plugin()
    from comfyui_indextts25_t8_test.nodes_v3 import _normalize_model_loader_values

    values = _normalize_model_loader_values(False, r"D:\models\IndexTTS-2.5", "")
    assert values[:3] == (0, False, r"D:\models\IndexTTS-2.5")
    assert "v0.10" in values[3]
    coerced = _normalize_model_loader_values(1, r"D:\models\IndexTTS-2.5", "")
    assert coerced[:3] == (0, True, r"D:\models\IndexTTS-2.5")
    current = _normalize_model_loader_values(20, True, r"D:\new-model")
    assert current[:3] == (20, True, r"D:\new-model")
    assert current[3] == ""


def test_model_loader_reports_node_core_and_model_versions(tmp_path, monkeypatch):
    _load_plugin()
    from types import SimpleNamespace

    from comfyui_indextts25_t8_test import nodes_v3

    class Report:
        hashes_verified = False

        def require_valid(self):
            return None

    monkeypatch.setattr(nodes_v3, "resolve_model", lambda *args: tmp_path)
    monkeypatch.setattr(nodes_v3, "validate_model_dir", lambda *args, **kwargs: Report())
    monkeypatch.setattr(nodes_v3, "model_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(nodes_v3, "_resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(nodes_v3, "_is_low_vram", lambda _device: False)
    monkeypatch.setattr(
        nodes_v3,
        "resolve_acceleration",
        lambda *args: SimpleNamespace(
            use_cuda_kernel=False,
            use_torch_compile=False,
            use_accel=False,
            use_deepspeed=False,
            requested="off",
            effective="off",
            reason="普通模式",
        ),
    )
    monkeypatch.setattr(
        nodes_v3,
        "load_manifest",
        lambda: {"codeRevision": "ee40fa7d6c", "modelRevision": "c39ce5ba98"},
    )

    result = nodes_v3.T8IndexTTS25ModelLoader.execute(
        "official",
        "auto",
        "auto",
        "off",
        False,
        False,
        0,
        False,
        "",
    )
    assert f"node={nodes_v3.PROJECT_VERSION}" in result[1]
    assert "core=ee40fa7d" in result[1]
    assert "model=c39ce5ba" in result[1]
    assert "reference=cpu" in result[1]


def test_model_loader_auto_download_is_explicit_and_license_gated(
    tmp_path, monkeypatch
):
    _load_plugin()
    from types import SimpleNamespace

    from comfyui_indextts25_t8_test import nodes_v3

    missing = nodes_v3.MISSING_MODEL_OPTION
    assert "自动下载" in nodes_v3.T8IndexTTS25ModelLoader.validate_inputs(
        missing,
        download_missing=False,
    )
    assert "许可证" in nodes_v3.T8IndexTTS25ModelLoader.validate_inputs(
        missing,
        download_missing=True,
        accept_model_license=False,
    )
    assert (
        nodes_v3.T8IndexTTS25ModelLoader.validate_inputs(
            missing,
            download_missing=True,
            accept_model_license=True,
        )
        is True
    )

    target = tmp_path / "IndexTTS-2.5"

    class MissingReport:
        valid = False
        hashes_verified = False

    class CompleteReport:
        valid = True
        hashes_verified = True

    monkeypatch.setattr(nodes_v3, "configured_model_roots", lambda: [tmp_path])
    monkeypatch.setattr(
        nodes_v3,
        "resolve_model",
        lambda *args: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(
        nodes_v3, "validate_model_dir", lambda *args, **kwargs: MissingReport()
    )
    calls = []

    def fake_ensure(path, source, **kwargs):
        calls.append((path, source, kwargs))
        return CompleteReport()

    monkeypatch.setattr(nodes_v3, "ensure_model_bundle", fake_ensure)
    monkeypatch.setattr(nodes_v3, "model_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(nodes_v3, "_resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(nodes_v3, "_is_low_vram", lambda _device: False)
    monkeypatch.setattr(
        nodes_v3,
        "resolve_acceleration",
        lambda *args: SimpleNamespace(
            use_cuda_kernel=False,
            use_torch_compile=False,
            use_accel=False,
            use_deepspeed=False,
            requested="off",
            effective="off",
            reason="普通模式",
        ),
    )
    monkeypatch.setattr(
        nodes_v3,
        "load_manifest",
        lambda: {"codeRevision": "ee40fa7d6c", "modelRevision": "a" * 40},
    )

    result = nodes_v3.T8IndexTTS25ModelLoader.execute(
        model_name=missing,
        device="auto",
        precision="auto",
        acceleration_mode="off",
        use_cuda_kernel=False,
        release_after_run=False,
        download_missing=True,
        accept_model_license=True,
    )
    assert len(calls) == 1
    assert calls[0][0:2] == (target, "huggingface")
    assert calls[0][2]["accept_license"] is True
    assert calls[0][2]["verify_hashes"] is True
    assert callable(calls[0][2]["progress"])
    assert "完整模型已自动下载/修复" in result[1]


def test_timeline_asr_and_subtitle_nodes_form_a_complete_editing_chain(monkeypatch):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.dialogue import DialogueLine
    from comfyui_indextts25_t8_test.runtime.types import DialogueScript

    script = DialogueScript(
        [DialogueLine(1, "旁白", "原始字幕", "ZH", 0, 1000, 1.0)], "srt"
    )
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
    assert tuple(asr[5].shape) == (1, 280, 1200, 3)

    generation_report = '{"lines":[{"index":1,"timeline":{"actual_start_ms":150,"actual_end_ms":950},"asr":{"recognized_text":"识别字幕","passed":true}}]}'
    rewritten = nodes_v3.T8IndexTTS25SubtitleRewrite.execute(
        edited[0], generation_report, "actual", "asr_passed", True
    )
    assert "00:00:00,150 --> 00:00:00,950" in rewritten[0]
    assert "[旁白] 识别字幕" in rewritten[0]


def test_asr_stop_check_after_transcription_is_not_swallowed(monkeypatch):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3

    monkeypatch.setattr(nodes_v3, "asr_available", lambda *args: True)
    monkeypatch.setattr(
        nodes_v3,
        "transcribe_waveform",
        lambda *args, **kwargs: {"text": "识别完成"},
    )
    checks = 0

    def interrupt_after_asr():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("processing interrupted")

    monkeypatch.setattr(
        nodes_v3, "throw_if_processing_interrupted", interrupt_after_asr
    )
    with pytest.raises(RuntimeError, match="processing interrupted"):
        nodes_v3.T8IndexTTS25ASRProofread.execute(
            {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000},
            "识别完成",
            "ZH",
            "auto",
            "tiny",
            "cpu",
            0.8,
        )


def test_dialogue_generation_can_auto_review_and_return_rewritten_srt(
    tmp_path, monkeypatch
):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.dialogue import DialogueLine
    from comfyui_indextts25_t8_test.runtime.types import (
        DialogueScript,
        ModelHandle,
        RoleLibrary,
        VoiceProfile,
    )

    audio = {"waveform": __import__("torch").zeros(1, 1, 22050), "sample_rate": 22050}
    monkeypatch.setattr(
        nodes_v3, "run_inference", lambda *args, **kwargs: (audio, "fake inference")
    )
    monkeypatch.setattr(nodes_v3, "asr_available", lambda *args: True)
    monkeypatch.setattr(
        nodes_v3,
        "transcribe_waveform",
        lambda *args, **kwargs: {
            "text": "自动校对字幕",
            "model": "tiny",
            "device": "cpu",
        },
    )
    script = DialogueScript(
        [DialogueLine(1, "角色A", "自动校对字幕", "ZH", 100, 1100)], "srt"
    )
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
    assert '"reports"' in result[4]


def test_dialogue_keeps_audio_when_asr_is_unavailable_and_normalizes_old_widgets(
    tmp_path, monkeypatch
):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.dialogue import DialogueLine
    from comfyui_indextts25_t8_test.runtime.types import (
        DialogueScript,
        ModelHandle,
        RoleLibrary,
        VoiceProfile,
    )

    audio = {"waveform": __import__("torch").zeros(1, 1, 22050), "sample_rate": 22050}
    monkeypatch.setattr(
        nodes_v3, "run_inference", lambda *args, **kwargs: (audio, "fake inference")
    )
    monkeypatch.setattr(nodes_v3, "asr_available", lambda *args: False)
    script = DialogueScript(
        [DialogueLine(1, "角色A", "音频必须保留", "ZH", 0, 1000)], "srt"
    )
    library = RoleLibrary({"角色A": VoiceProfile("角色A", audio, "ZH")})
    result = nodes_v3.T8IndexTTS25DialogueGenerate.execute(
        model=ModelHandle(tmp_path, "cpu", False),
        role_library=library,
        dialogue_script=script,
        seed=1,
        timeline_policy="shift",
        fit_srt_slots=False,
        slot_duration_mode="native",
        fit_tolerance_ms=180,
        batch_gap_ms=200,
        postprocess_preset="off",
        postprocess_strength=1.0,
        asr_enabled=True,
        asr_backend="auto",
        asr_model="base",
        asr_device="auto",
        asr_threshold=0.82,
        subtitle_timing_mode=0,
        subtitle_text_mode="actual",
        subtitle_include_role="asr_passed",
        asr_retry_count=True,
    )
    report = __import__("json").loads(result[2])
    assert result[0]["waveform"].shape[-1] == 22050
    assert report["asr"]["requested"] is True
    assert report["asr"]["enabled"] is False
    assert report["asr"]["maximum_retries"] == 0
    assert "v0.11.0" in report["asr"]["warning"]
    assert "音频正常输出" in report["asr"]["warning"]
    assert report["subtitle_rewrite"]["timing_mode"] == "actual"
    assert report["subtitle_rewrite"]["text_mode"] == "asr_passed"
    assert report["subtitle_rewrite"]["include_role"] is True
    assert "音频必须保留" in result[3]


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


def test_reference_quality_node_can_prepare_and_render_audio():
    _load_plugin()
    import json

    import torch

    from comfyui_indextts25_t8_test.nodes_v3 import T8IndexTTS25ReferenceQuality

    waveform = torch.cat(
        (torch.zeros(2000), torch.full((8000,), 0.1), torch.zeros(2000))
    )
    result = T8IndexTTS25ReferenceQuality.execute(
        {"waveform": waveform.reshape(1, 1, -1), "sample_rate": 16000},
        True,
        15.0,
        100,
    )
    assert result[0]["waveform"].shape[-1] < waveform.numel()
    assert json.loads(result[1])["trimmed"] is True
    assert tuple(result[2].shape) == (1, 280, 1200, 3)


def test_single_generation_quality_retry_selects_first_passing_seed(
    tmp_path, monkeypatch
):
    _load_plugin()
    import json

    import torch

    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.types import ModelHandle

    generated_seeds = []
    transcripts = iter(("错误文本", "目标文本", "另一段错误文本"))

    def fake_inference(*args, **kwargs):
        generated_seeds.append(int(kwargs["seed"]))
        return {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000}, "generated"

    monkeypatch.setattr(nodes_v3, "run_inference", fake_inference)
    monkeypatch.setattr(nodes_v3, "asr_available", lambda *_args: True)
    monkeypatch.setattr(
        nodes_v3,
        "transcribe_waveform",
        lambda *_args, **_kwargs: {"text": next(transcripts)},
    )

    result = nodes_v3.T8IndexTTS25Generate.execute(
        model=ModelHandle(tmp_path, "cpu", False),
        speaker_audio={"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000},
        text="目标文本",
        language="ZH",
        duration_factor=1.0,
        target_duration_mode="off",
        target_duration_seconds=0.0,
        postprocess_preset="off",
        postprocess_strength=1.0,
        seed=7,
        quality_retry_count=2,
        quality_asr_backend="auto",
        quality_asr_model="tiny",
        quality_asr_device="cpu",
        quality_threshold=0.8,
    )

    report = json.loads(result[1].split(" | quality=", 1)[1])
    assert generated_seeds == [7, 100010, 200013]
    assert report["selected_seed"] == 100010
    assert report["attempt_count"] == 3
    assert report["additional_candidates"] == 2
    assert report["review"]["passed"] is True
    assert len(result[2]) == 3


def test_generate_keeps_audio_when_quality_asr_is_unavailable(tmp_path, monkeypatch):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.types import ModelHandle

    audio = {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000}
    monkeypatch.setattr(
        nodes_v3, "run_inference", lambda *args, **kwargs: (audio, "generated")
    )
    monkeypatch.setattr(nodes_v3, "asr_available", lambda *_args: False)

    result = nodes_v3.T8IndexTTS25Generate.execute(
        model=ModelHandle(tmp_path, "cpu", False),
        speaker_audio=audio,
        text="目标文本",
        language="ZH",
        duration_factor=1.0,
        target_duration_mode="off",
        target_duration_seconds=0.0,
        postprocess_preset="off",
        postprocess_strength=1.0,
        seed=7,
        quality_retry_count=2,
    )
    report = json.loads(result[1].split(" | quality=", 1)[1])
    assert torch.equal(result[0]["waveform"], audio["waveform"])
    assert result[0]["sample_rate"] == audio["sample_rate"]
    assert report["requested"] is True and report["enabled"] is False
    assert report["attempt_count"] == 3
    assert report["selection_method"] == "technical"
    assert len(result[2]) == 3
    assert "保留全部候选" in report["warning"]


def test_generate_keeps_audio_when_quality_asr_runtime_fails(tmp_path, monkeypatch):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3
    from comfyui_indextts25_t8_test.runtime.types import ModelHandle

    audio = {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000}
    monkeypatch.setattr(
        nodes_v3, "run_inference", lambda *args, **kwargs: (audio, "generated")
    )
    monkeypatch.setattr(nodes_v3, "asr_available", lambda *_args: True)

    def fail_asr(*_args, **_kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(nodes_v3, "transcribe_waveform", fail_asr)
    result = nodes_v3.T8IndexTTS25Generate.execute(
        model=ModelHandle(tmp_path, "cpu", False),
        speaker_audio=audio,
        text="目标文本",
        language="ZH",
        duration_factor=1.0,
        target_duration_mode="off",
        target_duration_seconds=0.0,
        postprocess_preset="off",
        postprocess_strength=1.0,
        seed=7,
        quality_retry_count=2,
    )
    report = json.loads(result[1].split(" | quality=", 1)[1])
    assert torch.equal(result[0]["waveform"], audio["waveform"])
    assert result[0]["sample_rate"] == audio["sample_rate"]
    assert report["attempt_count"] == 3
    assert len(result[2]) == 3
    assert all(item["error"] == "download failed" for item in report["attempts"])
    assert "音频技术指标选优" in report["warning"]


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
    happy = EmotionConfig(
        mode="vector", vector=(0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2)
    )
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


def test_role_default_language_is_used_only_when_line_omits_language():
    _load_plugin()
    from comfyui_indextts25_t8_test.nodes_v3 import _resolve_line_language
    from comfyui_indextts25_t8_test.runtime.dialogue import DialogueLine
    from comfyui_indextts25_t8_test.runtime.types import VoiceProfile

    profile = VoiceProfile("角色A", {}, "EN")
    inherited = DialogueLine(
        1, "角色A", "hello", "ZH", language_explicit=False
    )
    explicit = DialogueLine(
        2, "角色A", "你好", "ZH", language_explicit=True
    )
    legacy = DialogueLine(3, "角色A", "兼容旧构造", "JA")

    assert _resolve_line_language(inherited, profile) == "EN"
    assert _resolve_line_language(explicit, profile) == "ZH"
    assert _resolve_line_language(legacy, profile) == "JA"


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
        {
            f"voice_{index}": VoiceProfile(f"角色{index}", audio)
            for index in range(count)
        }
    )
    assert len(result[0].profiles) == count


def test_dialogue_generation_routes_each_roles_emotion_without_leaking(
    tmp_path, monkeypatch
):
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
        ]
        + [
            DialogueLine(
                len(emotions) + 1,
                "角色0",
                "同一个角色逐句改成生气",
                "ZH",
                emotion_mode="vector",
                emotion_vector=(0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                emotion_strength=0.65,
            )
        ],
        "batch",
    )
    result = nodes_v3.T8IndexTTS25DialogueGenerate.execute(
        ModelHandle(tmp_path, "cpu", False),
        library,
        script,
        1,
        "overlay",
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
    assert routed[:4] == list(emotions)
    assert routed[4].mode == "vector"
    assert routed[4].vector[1] == pytest.approx(0.8)
    assert routed[4].strength == pytest.approx(0.65)
    report = json.loads(result[2])
    assert report["requested_timeline_policy"] == "overlay"
    assert report["timeline_policy"] == "shift"
    assert "避免所有台词重叠" in report["timeline_warning"]
    assert report["lines"][4]["emotion_source"] == "line_override"


def test_memory_control_release_all_includes_asr_cache(monkeypatch):
    _load_plugin()
    from comfyui_indextts25_t8_test import nodes_v3

    model_statuses = iter(
        (
            {"cached_models": 1, "entries": [], "cuda": {"available": False}},
            {"cached_models": 0, "entries": [], "cuda": {"available": False}},
        )
    )
    asr_statuses = iter(
        (
            {"cached_models": 2, "entries": []},
            {"cached_models": 0, "entries": []},
        )
    )
    monkeypatch.setattr(nodes_v3.MODEL_CACHE, "status", lambda: next(model_statuses))
    monkeypatch.setattr(nodes_v3.MODEL_CACHE, "clear", lambda: 1)
    monkeypatch.setattr(nodes_v3, "asr_cache_status", lambda: next(asr_statuses))
    monkeypatch.setattr(nodes_v3, "clear_asr_cache", lambda: 2)

    report = json.loads(nodes_v3.T8IndexTTS25MemoryControl.execute("release_all", 0)[0])
    assert report["released_models"] == 1
    assert report["released_asr_models"] == 2
    assert report["asr_after"]["cached_models"] == 0


def test_sampling_exposes_auto_segmentation_and_real_pause_controls():
    _load_plugin()
    from comfyui_indextts25_t8_test.nodes_v3 import T8IndexTTS25SamplingConfig

    result = T8IndexTTS25SamplingConfig.execute(
        False,
        0.8,
        0.8,
        30,
        3,
        10.0,
        0.0,
        1500,
        25,
        0.7,
        1.0,
        "auto",
        120,
        200,
        "narration",
        100,
        300,
        600,
        True,
    )
    config = result[0]
    assert config.effective_segment_tokens("EN") == 60
    assert config.effective_segment_tokens("ZH") == 120
    assert config.pause_preset == "narration"
    assert config.diffusion_steps == 25
    assert config.inference_cfg_rate == pytest.approx(0.7)
    assert config.cfm_temperature == pytest.approx(1.0)
