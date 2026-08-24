from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PLUGIN_ROOT / "example_workflows"
UI_ROOT = EXAMPLES_ROOT / "ui"
API_ROOT = EXAMPLES_ROOT / "api"

MODEL_NODE = "T8_IndexTTS25_ModelLoader"
EMOTION_NODE = "T8_IndexTTS25_EmotionControl"
SAMPLING_NODE = "T8_IndexTTS25_SamplingConfig"
TEXT_PREVIEW_NODE = "T8_IndexTTS25_TextPreview"
PRONUNCIATION_NODE = "T8_IndexTTS25_Pronunciation"
GENERATE_NODE = "T8_IndexTTS25_Generate"
VOICE_NODE = "T8_IndexTTS25_VoiceProfile"
ROLE_LIBRARY_NODE = "T8_IndexTTS25_RoleLibrary"
DIALOGUE_SCRIPT_NODE = "T8_IndexTTS25_DialogueScript"
TIMELINE_EDITOR_NODE = "T8_IndexTTS25_TimelineEditor"
DIALOGUE_GENERATE_NODE = "T8_IndexTTS25_DialogueGenerate"
ASR_PROOFREAD_NODE = "T8_IndexTTS25_ASRProofread"
SUBTITLE_REWRITE_NODE = "T8_IndexTTS25_SubtitleRewrite"
AUDIO_POSTPROCESS_NODE = "T8_IndexTTS25_AudioPostProcess"
ENVIRONMENT_NODE = "T8_IndexTTS25_Environment"


def widget_input(name: str, data_type: str, *, optional: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": name,
        "type": data_type,
        "widget": {"name": name},
        "link": None,
    }
    if optional:
        item["shape"] = 7
    return item


def slot_input(name: str, data_type: str, *, optional: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {"name": name, "type": data_type, "link": None}
    if optional:
        item["shape"] = 7
    return item


def output(name: str, data_type: str) -> dict[str, Any]:
    return {"name": name, "type": data_type, "links": []}


def properties(node_type: str, *, core: bool = False) -> dict[str, str]:
    return {
        "Node name for S&R": node_type,
        "cnr_id": "comfy-core" if core else "comfyui-indextts25-t8",
        "ver": "0.8.0",
    }


@dataclass
class Workflow:
    title: str
    nodes: list[dict[str, Any]] = field(default_factory=list)
    links: list[list[Any]] = field(default_factory=list)

    def add(
        self,
        node_type: str,
        pos: tuple[int, int],
        size: tuple[int, int],
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        widgets: list[Any] | None = None,
        *,
        title: str | None = None,
        core: bool = False,
    ) -> int:
        node_id = len(self.nodes) + 1
        node: dict[str, Any] = {
            "id": node_id,
            "type": node_type,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": len(self.nodes),
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": properties(node_type, core=core),
        }
        if widgets is not None:
            node["widgets_values"] = widgets
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node_id

    def connect(self, origin_id: int, origin_name: str, target_id: int, target_name: str) -> int:
        origin = self.nodes[origin_id - 1]
        target = self.nodes[target_id - 1]
        origin_slot = next(i for i, item in enumerate(origin["outputs"]) if item["name"] == origin_name)
        target_slot = next(i for i, item in enumerate(target["inputs"]) if item["name"] == target_name)
        data_type = origin["outputs"][origin_slot]["type"]
        link_id = len(self.links) + 1
        self.links.append([link_id, origin_id, origin_slot, target_id, target_slot, data_type])
        origin["outputs"][origin_slot]["links"].append(link_id)
        target["inputs"][target_slot]["link"] = link_id
        return link_id

    def as_dict(self) -> dict[str, Any]:
        width = max((node["pos"][0] + node["size"][0] for node in self.nodes), default=1200) + 80
        height = max((node["pos"][1] + node["size"][1] for node in self.nodes), default=700) + 80
        return {
            "last_node_id": len(self.nodes),
            "last_link_id": len(self.links),
            "nodes": self.nodes,
            "links": self.links,
            "groups": [
                {
                    "title": f"{self.title} · B站：T8star-Aix",
                    "bounding": [-40, -60, width, height],
                    "color": "#3f789e",
                    "font_size": 24,
                    "flags": {},
                }
            ],
            "config": {},
            "extra": {"ds": {"scale": 0.9, "offset": [80, 60]}},
            "version": 0.4,
        }


def add_model(workflow: Workflow, pos=(0, 0), *, release_after_run: bool = False) -> int:
    return workflow.add(
        MODEL_NODE,
        pos,
        (390, 290),
        [
            widget_input("model_name", "COMBO"),
            widget_input("device", "COMBO"),
            widget_input("precision", "COMBO"),
            widget_input("acceleration_mode", "COMBO"),
            widget_input("use_cuda_kernel", "BOOLEAN"),
            widget_input("release_after_run", "BOOLEAN"),
            widget_input("verify_hashes", "BOOLEAN"),
            widget_input("custom_model_path", "STRING", optional=True),
        ],
        [output("model", "T8_INDEXTTS25_MODEL"), output("model_info", "STRING")],
        ["IndexTTS-2.5", "auto", "auto", "off", False, release_after_run, False, ""],
    )


def add_load_audio(workflow: Workflow, filename: str, pos=(0, 340), *, title: str) -> int:
    return workflow.add(
        "LoadAudio",
        pos,
        (330, 130),
        [widget_input("audio", "COMBO")],
        [output("AUDIO", "AUDIO")],
        [filename],
        title=title,
        core=True,
    )


def add_emotion(workflow: Workflow, mode: str, pos=(440, 0), **values: Any) -> int:
    inputs = [widget_input("mode", "COMFY_DYNAMICCOMBO_V3")]
    widgets: list[Any] = [mode]
    size = (390, 170)
    if mode == "reference_audio":
        inputs.extend(
            [
                slot_input("mode.emotion_audio", "AUDIO"),
                widget_input("mode.strength", "FLOAT"),
            ]
        )
        widgets.append(values.get("strength", 0.8))
        size = (390, 220)
    elif mode == "vector":
        names = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
        inputs.extend(widget_input(f"mode.{name}", "FLOAT") for name in names)
        inputs.extend([widget_input("mode.strength", "FLOAT"), widget_input("mode.use_random", "BOOLEAN")])
        widgets.extend(values.get(name, 0.0) for name in names)
        widgets.extend([values.get("strength", 1.0), values.get("use_random", False)])
        size = (390, 520)
    elif mode == "text":
        inputs.extend(
            [
                widget_input("mode.emotion_text", "STRING"),
                widget_input("mode.strength", "FLOAT"),
            ]
        )
        widgets.extend([values.get("emotion_text", ""), values.get("strength", 0.85)])
        size = (390, 280)
    elif mode != "speaker":
        raise ValueError(f"Unsupported emotion mode: {mode}")
    return workflow.add(
        EMOTION_NODE,
        pos,
        size,
        inputs,
        [output("emotion", "T8_INDEXTTS25_EMOTION"), output("emotion_info", "STRING")],
        widgets,
    )


def sampling_values(**overrides: Any) -> dict[str, Any]:
    values = {
        "do_sample": False,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 30,
        "num_beams": 3,
        "repetition_penalty": 10.0,
        "length_penalty": 0.0,
        "max_mel_tokens": 1500,
        "diffusion_steps": 25,
        "inference_cfg_rate": 0.7,
        "cfm_temperature": 1.0,
        "segmentation_mode": "auto",
        "max_text_tokens_per_segment": 120,
        "segment_silence_ms": 200,
        "pause_preset": "off",
        "comma_pause_ms": 100,
        "sentence_pause_ms": 300,
        "paragraph_pause_ms": 600,
        "text_normalization": True,
    }
    values.update(overrides)
    return values


def add_sampling(workflow: Workflow, pos=(440, 330), **overrides: Any) -> int:
    values = sampling_values(**overrides)
    names_types = (
        ("do_sample", "BOOLEAN"),
        ("temperature", "FLOAT"),
        ("top_p", "FLOAT"),
        ("top_k", "INT"),
        ("num_beams", "INT"),
        ("repetition_penalty", "FLOAT"),
        ("length_penalty", "FLOAT"),
        ("max_mel_tokens", "INT"),
        ("diffusion_steps", "INT"),
        ("inference_cfg_rate", "FLOAT"),
        ("cfm_temperature", "FLOAT"),
        ("segmentation_mode", "COMBO"),
        ("max_text_tokens_per_segment", "INT"),
        ("segment_silence_ms", "INT"),
        ("pause_preset", "COMBO"),
        ("comma_pause_ms", "INT"),
        ("sentence_pause_ms", "INT"),
        ("paragraph_pause_ms", "INT"),
        ("text_normalization", "BOOLEAN"),
    )
    return workflow.add(
        SAMPLING_NODE,
        pos,
        (390, 760),
        [widget_input(name, data_type) for name, data_type in names_types],
        [output("sampling", "T8_INDEXTTS25_SAMPLING"), output("sampling_info", "STRING")],
        [values[name] for name, _ in names_types],
    )


def add_generate(
    workflow: Workflow,
    text: str,
    language: str,
    duration_factor: float,
    seed: int,
    pos=(900, 100),
    *,
    title: str | None = None,
    target_duration_mode: str = "off",
    target_duration_seconds: float = 0.0,
    postprocess_preset: str = "off",
    postprocess_strength: float = 1.0,
) -> int:
    return workflow.add(
        GENERATE_NODE,
        pos,
        (470, 560),
        [
            slot_input("model", "T8_INDEXTTS25_MODEL"),
            slot_input("speaker_audio", "AUDIO"),
            widget_input("text", "STRING"),
            widget_input("language", "COMBO"),
            widget_input("duration_factor", "FLOAT"),
            widget_input("target_duration_mode", "COMBO"),
            widget_input("target_duration_seconds", "FLOAT"),
            widget_input("postprocess_preset", "COMBO"),
            widget_input("postprocess_strength", "FLOAT"),
            widget_input("seed", "INT"),
            slot_input("emotion", "T8_INDEXTTS25_EMOTION", optional=True),
            slot_input("sampling", "T8_INDEXTTS25_SAMPLING", optional=True),
        ],
        [output("audio", "AUDIO"), output("generation_info", "STRING")],
        [
            text,
            language,
            duration_factor,
            target_duration_mode,
            target_duration_seconds,
            postprocess_preset,
            postprocess_strength,
            seed,
            "fixed",
        ],
        title=title,
    )


def add_pronunciation(
    workflow: Workflow,
    text: str,
    language: str,
    dictionary: str,
    pos=(440, 0),
    *,
    strict: bool = True,
) -> int:
    return workflow.add(
        PRONUNCIATION_NODE,
        pos,
        (430, 360),
        [
            widget_input("text", "STRING"),
            widget_input("language", "COMBO"),
            widget_input("dictionary", "STRING"),
            widget_input("strict", "BOOLEAN"),
        ],
        [output("annotated_text", "STRING"), output("pronunciation_report", "STRING")],
        [text, language, dictionary, strict],
    )


def add_text_preview(
    workflow: Workflow,
    text: str,
    language: str,
    pos=(880, 0),
) -> int:
    return workflow.add(
        TEXT_PREVIEW_NODE,
        pos,
        (460, 300),
        [
            slot_input("model", "T8_INDEXTTS25_MODEL"),
            widget_input("text", "STRING"),
            widget_input("language", "COMBO"),
            slot_input("sampling", "T8_INDEXTTS25_SAMPLING", optional=True),
        ],
        [output("text", "STRING"), output("plan_json", "STRING")],
        [text, language],
    )


def add_audio_postprocess(
    workflow: Workflow,
    preset: str,
    strength: float,
    pos=(1420, 160),
) -> int:
    return workflow.add(
        AUDIO_POSTPROCESS_NODE,
        pos,
        (410, 260),
        [
            slot_input("audio", "AUDIO"),
            widget_input("preset", "COMBO"),
            widget_input("strength", "FLOAT"),
            widget_input("target_peak_db", "FLOAT"),
        ],
        [output("audio", "AUDIO"), output("report", "STRING")],
        [preset, strength, -1.0],
    )


def add_save(workflow: Workflow, prefix: str, pos=(1460, 180), *, title: str | None = None) -> int:
    return workflow.add(
        "SaveAudio",
        pos,
        (320, 130),
        [slot_input("audio", "AUDIO"), widget_input("filename_prefix", "STRING")],
        [output("audio", "AUDIO")],
        [prefix],
        title=title,
        core=True,
    )


def add_voice_profile(workflow: Workflow, role: str, language: str, pos=(400, 0)) -> int:
    return workflow.add(
        VOICE_NODE,
        pos,
        (360, 240),
        [
            widget_input("role_name", "STRING"),
            slot_input("speaker_audio", "AUDIO"),
            widget_input("language", "COMBO"),
            slot_input("emotion", "T8_INDEXTTS25_EMOTION", optional=True),
        ],
        [output("voice", "T8_INDEXTTS25_VOICE"), output("voice_info", "STRING")],
        [role, language],
        title=f"角色音色：{role}",
    )


def add_role_library(workflow: Workflow, count: int, pos=(800, 40)) -> int:
    return workflow.add(
        ROLE_LIBRARY_NODE,
        pos,
        (360, 140 + count * 35),
        [slot_input(f"voices.voice_{index}", "T8_INDEXTTS25_VOICE", optional=index > 0) for index in range(count)],
        [output("role_library", "T8_INDEXTTS25_ROLE_LIBRARY"), output("role_info", "STRING")],
    )


def add_dialogue_script(workflow: Workflow, script_type: str, script: str, default_role: str, pos=(400, 620)) -> int:
    return workflow.add(
        DIALOGUE_SCRIPT_NODE,
        pos,
        (520, 390),
        [
            widget_input("script_type", "COMBO"),
            widget_input("script", "STRING"),
            widget_input("default_role", "STRING"),
            widget_input("default_language", "COMBO"),
        ],
        [output("dialogue_script", "T8_INDEXTTS25_DIALOGUE_SCRIPT"), output("script_preview", "STRING")],
        [script_type, script, default_role, "ZH"],
    )


def add_dialogue_generate(workflow: Workflow, pos=(1220, 240), *, policy="shift", fit=False) -> int:
    return workflow.add(
        DIALOGUE_GENERATE_NODE,
        pos,
        (500, 850),
        [
            slot_input("model", "T8_INDEXTTS25_MODEL"),
            slot_input("role_library", "T8_INDEXTTS25_ROLE_LIBRARY"),
            slot_input("dialogue_script", "T8_INDEXTTS25_DIALOGUE_SCRIPT"),
            slot_input("sampling", "T8_INDEXTTS25_SAMPLING", optional=True),
            widget_input("seed", "INT"),
            widget_input("timeline_policy", "COMBO"),
            widget_input("fit_srt_slots", "BOOLEAN"),
            widget_input("slot_duration_mode", "COMBO"),
            widget_input("fit_tolerance_ms", "INT"),
            widget_input("batch_gap_ms", "INT"),
            widget_input("postprocess_preset", "COMBO"),
            widget_input("postprocess_strength", "FLOAT"),
            widget_input("asr_enabled", "BOOLEAN"),
            widget_input("asr_model", "COMBO"),
            widget_input("asr_device", "COMBO"),
            widget_input("asr_threshold", "FLOAT"),
            widget_input("subtitle_timing_mode", "COMBO"),
            widget_input("subtitle_text_mode", "COMBO"),
            widget_input("subtitle_include_role", "BOOLEAN"),
        ],
        [
            output("audio", "AUDIO"),
            output("line_audios", "AUDIO"),
            output("generation_report", "STRING"),
            output("rewritten_srt", "STRING"),
            output("timeline_report", "STRING"),
        ],
        [20260818, "fixed", policy, fit, "native", 180, 200, "off", 1.0, False, "base", "auto", 0.82, "actual", "asr_passed", True],
    )


def add_timeline_editor(workflow: Workflow, edits_json: str = "", pos=(1040, 620)) -> int:
    return workflow.add(
        TIMELINE_EDITOR_NODE,
        pos,
        (520, 330),
        [
            slot_input("dialogue_script", "T8_INDEXTTS25_DIALOGUE_SCRIPT"),
            widget_input("timeline_edits_json", "STRING"),
        ],
        [
            output("dialogue_script", "T8_INDEXTTS25_DIALOGUE_SCRIPT"),
            output("timeline_preview", "STRING"),
            output("timeline_image", "IMAGE"),
        ],
        [edits_json],
    )


def add_preview_image(workflow: Workflow, pos=(1760, 650), *, title="可视化时间轴预览") -> int:
    return workflow.add(
        "PreviewImage",
        pos,
        (420, 360),
        [slot_input("images", "IMAGE")],
        [],
        [],
        title=title,
        core=True,
    )


def add_asr_proofread(workflow: Workflow, expected_text: str, language: str = "ZH", pos=(1460, 40)) -> int:
    return workflow.add(
        ASR_PROOFREAD_NODE,
        pos,
        (430, 420),
        [
            slot_input("audio", "AUDIO"),
            widget_input("expected_text", "STRING"),
            widget_input("language", "COMBO"),
            widget_input("model_name", "COMBO"),
            widget_input("device", "COMBO"),
            widget_input("threshold", "FLOAT"),
        ],
        [
            output("recognized_text", "STRING"),
            output("passed", "BOOLEAN"),
            output("similarity", "FLOAT"),
            output("review_report", "STRING"),
        ],
        [expected_text, language, "base", "auto", 0.82],
    )


def add_subtitle_rewrite(workflow: Workflow, pos=(1800, 560)) -> int:
    return workflow.add(
        SUBTITLE_REWRITE_NODE,
        pos,
        (460, 390),
        [
            slot_input("dialogue_script", "T8_INDEXTTS25_DIALOGUE_SCRIPT"),
            slot_input("generation_report", "STRING"),
            widget_input("timing_mode", "COMBO"),
            widget_input("text_mode", "COMBO"),
            widget_input("include_role", "BOOLEAN"),
        ],
        [output("srt", "STRING"), output("rewrite_report", "STRING")],
        ["actual", "asr_passed", True],
    )


def wire_generation(
    workflow: Workflow,
    model: int,
    speaker: int,
    generate: int,
    save: int,
    *,
    emotion: int | None = None,
    sampling: int | None = None,
) -> None:
    workflow.connect(model, "model", generate, "model")
    workflow.connect(speaker, "AUDIO", generate, "speaker_audio")
    if emotion is not None:
        workflow.connect(emotion, "emotion", generate, "emotion")
    if sampling is not None:
        workflow.connect(sampling, "sampling", generate, "sampling")
    workflow.connect(generate, "audio", save, "audio")


def api_model(*, release_after_run: bool = False) -> dict[str, Any]:
    return {
        "class_type": MODEL_NODE,
        "inputs": {
            "model_name": "IndexTTS-2.5",
            "device": "auto",
            "precision": "auto",
            "acceleration_mode": "off",
            "use_cuda_kernel": False,
            "release_after_run": release_after_run,
            "verify_hashes": False,
            "custom_model_path": "",
        },
    }


def api_audio(filename: str) -> dict[str, Any]:
    return {"class_type": "LoadAudio", "inputs": {"audio": filename}}


def api_generate(
    model_id: str,
    speaker_id: str,
    text: str,
    language: str,
    duration_factor: float,
    seed: int,
    *,
    emotion_id: str | None = None,
    sampling_id: str | None = None,
    target_duration_mode: str = "off",
    target_duration_seconds: float = 0.0,
    postprocess_preset: str = "off",
    postprocess_strength: float = 1.0,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "model": [model_id, 0],
        "speaker_audio": [speaker_id, 0],
        "text": text,
        "language": language,
        "duration_factor": duration_factor,
        "target_duration_mode": target_duration_mode,
        "target_duration_seconds": target_duration_seconds,
        "postprocess_preset": postprocess_preset,
        "postprocess_strength": postprocess_strength,
        "seed": seed,
    }
    if emotion_id:
        inputs["emotion"] = [emotion_id, 0]
    if sampling_id:
        inputs["sampling"] = [sampling_id, 0]
    return {"class_type": GENERATE_NODE, "inputs": inputs}


def api_pronunciation(text: str, language: str, dictionary: str, *, strict: bool = True) -> dict[str, Any]:
    return {
        "class_type": PRONUNCIATION_NODE,
        "inputs": {
            "text": text,
            "language": language,
            "dictionary": dictionary,
            "strict": strict,
        },
    }


def api_text_preview(model_id: str, text: str, language: str, sampling_id: str | None = None) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "model": [model_id, 0],
        "text": text,
        "language": language,
    }
    if sampling_id:
        inputs["sampling"] = [sampling_id, 0]
    return {"class_type": TEXT_PREVIEW_NODE, "inputs": inputs}


def api_audio_postprocess(audio_id: str, preset: str, strength: float = 1.0) -> dict[str, Any]:
    return {
        "class_type": AUDIO_POSTPROCESS_NODE,
        "inputs": {
            "audio": [audio_id, 0],
            "preset": preset,
            "strength": strength,
            "target_peak_db": -1.0,
        },
    }


def api_save(generate_id: str, prefix: str) -> dict[str, Any]:
    return {
        "class_type": "SaveAudio",
        "inputs": {"audio": [generate_id, 0], "filename_prefix": prefix},
    }


def api_emotion(mode: str, **values: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = {"mode": mode}
    if mode == "reference_audio":
        inputs.update(
            {
                "mode.emotion_audio": values["emotion_audio"],
                "mode.strength": values.get("strength", 0.8),
            }
        )
    elif mode == "vector":
        for name in ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"):
            inputs[f"mode.{name}"] = values.get(name, 0.0)
        inputs["mode.strength"] = values.get("strength", 1.0)
        inputs["mode.use_random"] = values.get("use_random", False)
    elif mode == "text":
        inputs["mode.emotion_text"] = values.get("emotion_text", "")
        inputs["mode.strength"] = values.get("strength", 0.85)
    elif mode != "speaker":
        raise ValueError(f"Unsupported emotion mode: {mode}")
    return {"class_type": EMOTION_NODE, "inputs": inputs}


def api_voice(role: str, audio_id: str, language: str = "ZH") -> dict[str, Any]:
    return {
        "class_type": VOICE_NODE,
        "inputs": {"role_name": role, "speaker_audio": [audio_id, 0], "language": language},
    }


def api_role_library(voice_ids: list[str]) -> dict[str, Any]:
    return {
        "class_type": ROLE_LIBRARY_NODE,
        "inputs": {f"voices.voice_{index}": [voice_id, 0] for index, voice_id in enumerate(voice_ids)},
    }


def api_dialogue_script(script_type: str, script: str, default_role: str) -> dict[str, Any]:
    return {
        "class_type": DIALOGUE_SCRIPT_NODE,
        "inputs": {
            "script_type": script_type,
            "script": script,
            "default_role": default_role,
            "default_language": "ZH",
        },
    }


def api_timeline_editor(script_id: str, edits_json: str = "") -> dict[str, Any]:
    return {
        "class_type": TIMELINE_EDITOR_NODE,
        "inputs": {
            "dialogue_script": [script_id, 0],
            "timeline_edits_json": edits_json,
        },
    }


def api_dialogue_generate(model_id: str, library_id: str, script_id: str, *, policy="shift", fit=False) -> dict[str, Any]:
    return {
        "class_type": DIALOGUE_GENERATE_NODE,
        "inputs": {
            "model": [model_id, 0],
            "role_library": [library_id, 0],
            "dialogue_script": [script_id, 0],
            "seed": 20260818,
            "timeline_policy": policy,
            "fit_srt_slots": fit,
            "slot_duration_mode": "native",
            "fit_tolerance_ms": 180,
            "batch_gap_ms": 200,
            "postprocess_preset": "off",
            "postprocess_strength": 1.0,
            "asr_enabled": False,
            "asr_model": "base",
            "asr_device": "auto",
            "asr_threshold": 0.82,
            "subtitle_timing_mode": "actual",
            "subtitle_text_mode": "asr_passed",
            "subtitle_include_role": True,
        },
    }


def api_asr(audio_id: str, expected_text: str, language: str = "ZH") -> dict[str, Any]:
    return {
        "class_type": ASR_PROOFREAD_NODE,
        "inputs": {
            "audio": [audio_id, 0],
            "expected_text": expected_text,
            "language": language,
            "model_name": "base",
            "device": "auto",
            "threshold": 0.82,
        },
    }


def api_subtitle_rewrite(script_id: str, generate_id: str) -> dict[str, Any]:
    return {
        "class_type": SUBTITLE_REWRITE_NODE,
        "inputs": {
            "dialogue_script": [script_id, 0],
            "generation_report": [generate_id, 2],
            "timing_mode": "actual",
            "text_mode": "asr_passed",
            "include_role": True,
        },
    }


def api_sampling(**overrides: Any) -> dict[str, Any]:
    return {"class_type": SAMPLING_NODE, "inputs": sampling_values(**overrides)}


def basic_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = Workflow("01 基础音色克隆")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    generate = add_generate(
        workflow,
        "欢迎使用 IndexTTS 2.5，来自 B 站：T8star-Aix。",
        "ZH",
        1.0,
        20260811,
    )
    save = add_save(workflow, "IndexTTS25_T8/basic")
    wire_generation(workflow, model, speaker, generate, save)
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_generate("1", "2", "欢迎使用 IndexTTS 2.5，来自 B 站：T8star-Aix。", "ZH", 1.0, 20260811),
        "4": api_save("3", "IndexTTS25_T8/basic"),
    }
    return workflow.as_dict(), api


def speed_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = Workflow("02 官方语速适配对比")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    text = "这是 IndexTTS 二点五官方语速适配功能的对比测试。"
    api: dict[str, Any] = {"1": api_model(), "2": api_audio("voice_reference.wav")}
    for index, (factor, label) in enumerate(((0.7, "fast"), (1.0, "normal"), (1.3, "slow"))):
        y = index * 470
        generate = add_generate(
            workflow,
            text,
            "ZH",
            factor,
            20260811,
            pos=(900, y),
            title=f"时长系数 {factor:.1f} · {label}",
        )
        save = add_save(workflow, f"IndexTTS25_T8/speed_{factor:.1f}", pos=(1460, y + 130))
        wire_generation(workflow, model, speaker, generate, save)
        generate_id = str(3 + index)
        save_id = str(6 + index)
        api[generate_id] = api_generate("1", "2", text, "ZH", factor, 20260811)
        api[save_id] = api_save(generate_id, f"IndexTTS25_T8/speed_{factor:.1f}")
    return workflow.as_dict(), api


def reference_emotion_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = Workflow("03 独立情感参考音频")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    emotion_audio = add_load_audio(workflow, "emotion_reference.wav", pos=(0, 520), title="情感参考音频")
    emotion = add_emotion(workflow, "reference_audio", strength=0.8)
    workflow.connect(emotion_audio, "AUDIO", emotion, "mode.emotion_audio")
    generate = add_generate(workflow, "虽然经历了许多困难，我们依然看见了新的希望。", "ZH", 1.0, 20260811)
    save = add_save(workflow, "IndexTTS25_T8/emotion_reference")
    wire_generation(workflow, model, speaker, generate, save, emotion=emotion)
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_audio("emotion_reference.wav"),
        "4": api_emotion("reference_audio", emotion_audio=["3", 0], strength=0.8),
        "5": api_generate("1", "2", "虽然经历了许多困难，我们依然看见了新的希望。", "ZH", 1.0, 20260811, emotion_id="4"),
        "6": api_save("5", "IndexTTS25_T8/emotion_reference"),
    }
    return workflow.as_dict(), api


def vector_emotion_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    vector = {"happy": 0.55, "surprised": 0.1, "calm": 0.15, "strength": 1.0, "use_random": False}
    workflow = Workflow("04 八维情感向量")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    emotion = add_emotion(workflow, "vector", **vector)
    generate = add_generate(workflow, "太好了，这个版本终于加入了真正可控的语速适配功能！", "ZH", 0.95, 20260811)
    save = add_save(workflow, "IndexTTS25_T8/emotion_vector")
    wire_generation(workflow, model, speaker, generate, save, emotion=emotion)
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_emotion("vector", **vector),
        "4": api_generate("1", "2", "太好了，这个版本终于加入了真正可控的语速适配功能！", "ZH", 0.95, 20260811, emotion_id="3"),
        "5": api_save("4", "IndexTTS25_T8/emotion_vector"),
    }
    return workflow.as_dict(), api


def text_emotion_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    emotion_text = "克制但难掩喜悦，声音温柔，结尾带一点轻松的笑意。"
    workflow = Workflow("05 文本情感描述")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    emotion = add_emotion(workflow, "text", emotion_text=emotion_text, strength=0.85)
    generate = add_generate(workflow, "等了这么久，我们终于可以一起出发了。", "ZH", 1.05, 20260811)
    save = add_save(workflow, "IndexTTS25_T8/emotion_text")
    wire_generation(workflow, model, speaker, generate, save, emotion=emotion)
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_emotion("text", emotion_text=emotion_text, strength=0.85),
        "4": api_generate("1", "2", "等了这么久，我们终于可以一起出发了。", "ZH", 1.05, 20260811, emotion_id="3"),
        "5": api_save("4", "IndexTTS25_T8/emotion_text"),
    }
    return workflow.as_dict(), api


def sampling_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    controls = {
        "do_sample": True,
        "temperature": 0.85,
        "top_p": 0.9,
        "top_k": 40,
        "num_beams": 1,
        "max_text_tokens_per_segment": 80,
        "segment_silence_ms": 350,
    }
    text = (
        "第一段用于演示随机采样，并通过固定种子保持结果可复现。"
        "第二段用于演示长文本自动分段，以及段落之间的静音控制。"
        "你可以修改温度、顶部概率和顶部候选数量，观察语气细节的变化。"
    )
    workflow = Workflow("06 随机采样与长文本")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    sampling = add_sampling(workflow, **controls)
    generate = add_generate(workflow, text, "ZH", 1.0, 998877, pos=(900, 180))
    save = add_save(workflow, "IndexTTS25_T8/random_long_text", pos=(1460, 300))
    wire_generation(workflow, model, speaker, generate, save, sampling=sampling)
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_sampling(**controls),
        "4": api_generate("1", "2", text, "ZH", 1.0, 998877, sampling_id="3"),
        "5": api_save("4", "IndexTTS25_T8/random_long_text"),
    }
    return workflow.as_dict(), api


def multilingual_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    cases = (
        ("ZH", "这是中文语音生成示例。"),
        ("EN", "This is an English speech generation example."),
        ("JA", "これは日本語の音声生成サンプルです。"),
        ("ES", "Este es un ejemplo de generación de voz en español."),
        ("AR", "هذا مثال على توليد الكلام باللغة العربية."),
    )
    workflow = Workflow("07 五语种生成")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    api: dict[str, Any] = {"1": api_model(), "2": api_audio("voice_reference.wav")}
    for index, (language, text) in enumerate(cases):
        y = index * 430
        generate = add_generate(workflow, text, language, 1.0, 20260811, pos=(900, y), title=f"{language} 生成")
        save = add_save(workflow, f"IndexTTS25_T8/{language.lower()}", pos=(1460, y + 120))
        wire_generation(workflow, model, speaker, generate, save)
        generate_id = str(3 + index)
        save_id = str(8 + index)
        api[generate_id] = api_generate("1", "2", text, language, 1.0, 20260811)
        api[save_id] = api_save(generate_id, f"IndexTTS25_T8/{language.lower()}")
    return workflow.as_dict(), api


def _pronunciation_pair(
    title: str,
    text: str,
    language: str,
    dictionary: str,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = Workflow(title)
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    pronunciation = add_pronunciation(workflow, text, language, dictionary, pos=(440, 0))
    generate = add_generate(workflow, text, language, 1.0, 20260811, pos=(950, 60))
    save = add_save(workflow, prefix, pos=(1510, 170))
    workflow.connect(pronunciation, "annotated_text", generate, "text")
    wire_generation(workflow, model, speaker, generate, save)
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_pronunciation(text, language, dictionary),
        "4": api_generate("1", "2", ["3", 0], language, 1.0, 20260811),
        "5": api_save("4", prefix),
    }
    return workflow.as_dict(), api


def chinese_pronunciation_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    return _pronunciation_pair(
        "08 中文多音字词典",
        "他在银行里工作，行长正在开会；这段手工标注的银<行|HANG2>不会被词典覆盖。",
        "ZH",
        "银行|YIN2 HANG2|ZH\n行长|HANG2 ZHANG3|ZH\n重庆|CHONG2 QING4|ZH",
        "IndexTTS25_T8/pronunciation_zh",
    )


def english_pronunciation_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    return _pronunciation_pair(
        "09 英文 CMU 音素",
        "He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details.",
        "EN",
        "Bilibili|B IY1 . L IY1 . B IY1 . L IY1|EN",
        "IndexTTS25_T8/pronunciation_en",
    )


def japanese_pronunciation_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    return _pronunciation_pair(
        "10 日语假名发音",
        "彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。",
        "JA",
        "",
        "IndexTTS25_T8/pronunciation_ja",
    )


def _dialogue_pair(title: str, script_type: str, script: str, *, policy="shift", fit=False) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = Workflow(title)
    model = add_model(workflow, pos=(0, 0))
    audio_a = add_load_audio(workflow, "role_a.wav", pos=(0, 340), title="角色 A 参考音频")
    audio_b = add_load_audio(workflow, "role_b.wav", pos=(0, 520), title="角色 B 参考音频")
    voice_a = add_voice_profile(workflow, "角色A", "ZH", pos=(400, 0))
    voice_b = add_voice_profile(workflow, "角色B", "ZH", pos=(400, 300))
    library = add_role_library(workflow, 2, pos=(800, 60))
    script_node = add_dialogue_script(workflow, script_type, script, "角色A", pos=(650, 520))
    generate = add_dialogue_generate(workflow, pos=(1220, 220), policy=policy, fit=fit)
    save = add_save(workflow, f"IndexTTS25_T8/{script_type}_dialogue", pos=(1760, 360))
    workflow.connect(audio_a, "AUDIO", voice_a, "speaker_audio")
    workflow.connect(audio_b, "AUDIO", voice_b, "speaker_audio")
    workflow.connect(voice_a, "voice", library, "voices.voice_0")
    workflow.connect(voice_b, "voice", library, "voices.voice_1")
    workflow.connect(model, "model", generate, "model")
    workflow.connect(library, "role_library", generate, "role_library")
    workflow.connect(script_node, "dialogue_script", generate, "dialogue_script")
    workflow.connect(generate, "audio", save, "audio")
    api = {
        "1": api_model(),
        "2": api_audio("role_a.wav"),
        "3": api_audio("role_b.wav"),
        "4": api_voice("角色A", "2"),
        "5": api_voice("角色B", "3"),
        "6": api_role_library(["4", "5"]),
        "7": api_dialogue_script(script_type, script, "角色A"),
        "8": api_dialogue_generate("1", "6", "7", policy=policy, fit=fit),
        "9": api_save("8", f"IndexTTS25_T8/{script_type}_dialogue"),
    }
    return workflow.as_dict(), api


def multi_role_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    return _dialogue_pair(
        "11 多角色对话",
        "batch",
        "角色A|你终于来了，我们开始吧。|ZH|1.0\n角色B|好，我已经准备好了。|ZH|0.95\n角色A|第一幕，现在开始。|ZH|1.05",
    )


def batch_dialogue_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    script = json.dumps(
        [
            {"role": "角色A", "text": "第一条批量台词。", "language": "ZH", "duration_factor": 0.9},
            {"role": "角色A", "text": "第二条批量台词。", "language": "ZH", "duration_factor": 1.0},
            {"role": "角色B", "text": "最后由另一个角色收尾。", "language": "ZH", "duration_factor": 1.1},
        ],
        ensure_ascii=False,
        indent=2,
    )
    return _dialogue_pair("12 JSON 批量台词", "batch", script)


def srt_dialogue_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    script = (
        "1\n00:00:00,500 --> 00:00:02,600\n[角色A] 这是第一条字幕。\n\n"
        "2\n00:00:02,800 --> 00:00:05,000\n角色B：这是第二条字幕，会尝试贴合时间槽位。"
    )
    return _dialogue_pair("13 SRT 多角色配音", "srt", script, policy="shift", fit=True)


def acceleration_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = Workflow("14 可选加速与环境诊断")
    model = add_model(workflow)
    workflow.nodes[model - 1]["widgets_values"][3] = "auto_safe"
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    environment = workflow.add(
        ENVIRONMENT_NODE,
        (440, 0),
        (390, 140),
        [widget_input("device", "COMBO")],
        [output("environment_report", "STRING")],
        ["auto"],
    )
    generate = add_generate(workflow, "这是可选加速模式的安全回退测试。", "ZH", 1.0, 20260818, pos=(900, 180))
    save = add_save(workflow, "IndexTTS25_T8/optional_acceleration")
    wire_generation(workflow, model, speaker, generate, save)
    model_api = api_model()
    model_api["inputs"]["acceleration_mode"] = "auto_safe"
    api = {
        "1": model_api,
        "2": api_audio("voice_reference.wav"),
        "3": {"class_type": ENVIRONMENT_NODE, "inputs": {"device": "auto"}},
        "4": api_generate("1", "2", "这是可选加速模式的安全回退测试。", "ZH", 1.0, 20260818),
        "5": api_save("4", "IndexTTS25_T8/optional_acceleration"),
    }
    return workflow.as_dict(), api


def auto_segment_preview_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    text = (
        "This is a deliberately longer English paragraph used to demonstrate language-aware automatic "
        "segmentation. The preview node shows every token segment before synthesis and protects the GPT "
        "acceleration path when a risky long-text cache pattern is detected."
    )
    workflow = Workflow("15 英文自动分段与预览")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    sampling = add_sampling(workflow, pos=(420, 300), segmentation_mode="auto")
    preview = add_text_preview(workflow, text, "EN", pos=(840, 0))
    generate = add_generate(workflow, text, "EN", 1.0, 20260822, pos=(1360, 80))
    save = add_save(workflow, "IndexTTS25_T8/auto_segment", pos=(1900, 240))
    workflow.connect(model, "model", preview, "model")
    workflow.connect(sampling, "sampling", preview, "sampling")
    workflow.connect(preview, "text", generate, "text")
    wire_generation(workflow, model, speaker, generate, save, sampling=sampling)
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_sampling(segmentation_mode="auto"),
        "4": api_text_preview("1", text, "EN", "3"),
        "5": api_generate("1", "2", ["4", 0], "EN", 1.0, 20260822, sampling_id="3"),
        "6": api_save("5", "IndexTTS25_T8/auto_segment"),
    }
    return workflow.as_dict(), api


def pause_control_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    text = "第一句结束。这里会自然停顿，接着说<pause=0.8>这一句前有八百毫秒显式停顿。"
    workflow = Workflow("16 标点与显式停顿")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    sampling = add_sampling(workflow, pause_preset="narration")
    generate = add_generate(workflow, text, "ZH", 1.0, 20260822, pos=(900, 120))
    save = add_save(workflow, "IndexTTS25_T8/pause_control", pos=(1460, 280))
    wire_generation(workflow, model, speaker, generate, save, sampling=sampling)
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_sampling(pause_preset="narration"),
        "4": api_generate("1", "2", text, "ZH", 1.0, 20260822, sampling_id="3"),
        "5": api_save("4", "IndexTTS25_T8/pause_control"),
    }
    return workflow.as_dict(), api


def target_duration_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    text = "这段语音会使用原生长度调节器，在一次推理中适配到五秒。"
    workflow = Workflow("17 目标时长秒数")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    generate = add_generate(
        workflow,
        text,
        "ZH",
        1.0,
        20260822,
        target_duration_mode="native",
        target_duration_seconds=5.0,
    )
    save = add_save(workflow, "IndexTTS25_T8/target_5s")
    wire_generation(workflow, model, speaker, generate, save)
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_generate(
            "1", "2", text, "ZH", 1.0, 20260822,
            target_duration_mode="native", target_duration_seconds=5.0,
        ),
        "4": api_save("3", "IndexTTS25_T8/target_5s"),
    }
    return workflow.as_dict(), api


def audio_postprocess_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = Workflow("18 独立人声后处理")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    generate = add_generate(workflow, "这是清晰旁白后处理的真实节点连接示例。", "ZH", 1.0, 20260822)
    post = add_audio_postprocess(workflow, "clear_narration", 0.8)
    save = add_save(workflow, "IndexTTS25_T8/clear_narration", pos=(1900, 240))
    workflow.connect(model, "model", generate, "model")
    workflow.connect(speaker, "AUDIO", generate, "speaker_audio")
    workflow.connect(generate, "audio", post, "audio")
    workflow.connect(post, "audio", save, "audio")
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_generate("1", "2", "这是清晰旁白后处理的真实节点连接示例。", "ZH", 1.0, 20260822),
        "4": api_audio_postprocess("3", "clear_narration", 0.8),
        "5": api_save("4", "IndexTTS25_T8/clear_narration"),
    }
    return workflow.as_dict(), api


def cfm_advanced_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    text = "这个示例使用四十步扩散、零点八五引导强度和零点八温度，适合比较稳定旁白。"
    workflow = Workflow("19 CFM 高级稳定性参数")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    sampling = add_sampling(
        workflow,
        diffusion_steps=40,
        inference_cfg_rate=0.85,
        cfm_temperature=0.8,
    )
    generate = add_generate(workflow, text, "ZH", 1.0, 20260824, pos=(900, 120))
    save = add_save(workflow, "IndexTTS25_T8/cfm_stable", pos=(1460, 280))
    wire_generation(workflow, model, speaker, generate, save, sampling=sampling)
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_sampling(
            diffusion_steps=40,
            inference_cfg_rate=0.85,
            cfm_temperature=0.8,
        ),
        "4": api_generate("1", "2", text, "ZH", 1.0, 20260824, sampling_id="3"),
        "5": api_save("4", "IndexTTS25_T8/cfm_stable"),
    }
    return workflow.as_dict(), api


def asr_proofread_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    text = "这是 ASR 自动校对示例，节点会识别生成音频并计算相似度和字符错误率。"
    workflow = Workflow("20 ASR 自动校对")
    model = add_model(workflow)
    speaker = add_load_audio(workflow, "voice_reference.wav", title="音色参考音频")
    generate = add_generate(workflow, text, "ZH", 1.0, 20260825, pos=(900, 40))
    asr = add_asr_proofread(workflow, text, "ZH", pos=(1460, 20))
    save = add_save(workflow, "IndexTTS25_T8/asr_proofread", pos=(1960, 300))
    workflow.connect(model, "model", generate, "model")
    workflow.connect(speaker, "AUDIO", generate, "speaker_audio")
    workflow.connect(generate, "audio", asr, "audio")
    workflow.connect(generate, "audio", save, "audio")
    api = {
        "1": api_model(),
        "2": api_audio("voice_reference.wav"),
        "3": api_generate("1", "2", text, "ZH", 1.0, 20260825),
        "4": api_asr("3", text, "ZH"),
        "5": api_save("3", "IndexTTS25_T8/asr_proofread"),
    }
    return workflow.as_dict(), api


def timeline_editor_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    script = (
        "1\n00:00:00,000 --> 00:00:02,000\n[角色A] 第一条编辑后从零点二秒开始。\n\n"
        "2\n00:00:02,200 --> 00:00:04,500\n[角色B] 第二条保留独立时间槽位。"
    )
    edits = json.dumps(
        [
            {"index": 1, "role": "角色A", "language": "ZH", "start_ms": 200, "end_ms": 2000, "duration_factor": 1.0, "text": "第一条编辑后从零点二秒开始。"},
            {"index": 2, "role": "角色B", "language": "ZH", "start_ms": 2300, "end_ms": 4600, "duration_factor": 0.95, "text": "第二条保留独立时间槽位。"},
        ],
        ensure_ascii=False,
        indent=2,
    )
    workflow = Workflow("21 可视化时间轴编辑")
    model = add_model(workflow, pos=(0, 0))
    audio_a = add_load_audio(workflow, "role_a.wav", pos=(0, 340), title="角色 A 参考音频")
    audio_b = add_load_audio(workflow, "role_b.wav", pos=(0, 520), title="角色 B 参考音频")
    voice_a = add_voice_profile(workflow, "角色A", "ZH", pos=(400, 0))
    voice_b = add_voice_profile(workflow, "角色B", "ZH", pos=(400, 300))
    library = add_role_library(workflow, 2, pos=(800, 60))
    script_node = add_dialogue_script(workflow, "srt", script, "角色A", pos=(650, 520))
    editor = add_timeline_editor(workflow, edits, pos=(1200, 650))
    preview = add_preview_image(workflow, pos=(1760, 650))
    generate = add_dialogue_generate(workflow, pos=(1760, 80), policy="overlay", fit=True)
    save = add_save(workflow, "IndexTTS25_T8/timeline_edited", pos=(2320, 920))
    workflow.connect(audio_a, "AUDIO", voice_a, "speaker_audio")
    workflow.connect(audio_b, "AUDIO", voice_b, "speaker_audio")
    workflow.connect(voice_a, "voice", library, "voices.voice_0")
    workflow.connect(voice_b, "voice", library, "voices.voice_1")
    workflow.connect(script_node, "dialogue_script", editor, "dialogue_script")
    workflow.connect(editor, "timeline_image", preview, "images")
    workflow.connect(model, "model", generate, "model")
    workflow.connect(library, "role_library", generate, "role_library")
    workflow.connect(editor, "dialogue_script", generate, "dialogue_script")
    workflow.connect(generate, "audio", save, "audio")
    api = {
        "1": api_model(), "2": api_audio("role_a.wav"), "3": api_audio("role_b.wav"),
        "4": api_voice("角色A", "2"), "5": api_voice("角色B", "3"),
        "6": api_role_library(["4", "5"]), "7": api_dialogue_script("srt", script, "角色A"),
        "8": api_timeline_editor("7", edits),
        "9": api_dialogue_generate("1", "6", "8", policy="overlay", fit=True),
        "10": api_save("9", "IndexTTS25_T8/timeline_edited"),
        "11": {"class_type": "PreviewImage", "inputs": {"images": ["8", 2]}},
    }
    return workflow.as_dict(), api


def subtitle_rewrite_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    script = (
        "1\n00:00:00,000 --> 00:00:02,500\n[角色A] 自动校对第一条字幕。\n\n"
        "2\n00:00:02,700 --> 00:00:05,000\n[角色B] 识别通过后自动回写文本。"
    )
    workflow = Workflow("22 ASR 字幕自动回写")
    model = add_model(workflow, pos=(0, 0))
    audio_a = add_load_audio(workflow, "role_a.wav", pos=(0, 340), title="角色 A 参考音频")
    audio_b = add_load_audio(workflow, "role_b.wav", pos=(0, 520), title="角色 B 参考音频")
    voice_a = add_voice_profile(workflow, "角色A", "ZH", pos=(400, 0))
    voice_b = add_voice_profile(workflow, "角色B", "ZH", pos=(400, 300))
    library = add_role_library(workflow, 2, pos=(800, 60))
    script_node = add_dialogue_script(workflow, "srt", script, "角色A", pos=(650, 520))
    generate = add_dialogue_generate(workflow, pos=(1220, 100), policy="shift", fit=True)
    workflow.nodes[generate - 1]["widgets_values"][9] = True
    subtitle = add_subtitle_rewrite(workflow, pos=(1780, 520))
    save = add_save(workflow, "IndexTTS25_T8/subtitle_rewrite", pos=(2300, 120))
    workflow.connect(audio_a, "AUDIO", voice_a, "speaker_audio")
    workflow.connect(audio_b, "AUDIO", voice_b, "speaker_audio")
    workflow.connect(voice_a, "voice", library, "voices.voice_0")
    workflow.connect(voice_b, "voice", library, "voices.voice_1")
    workflow.connect(model, "model", generate, "model")
    workflow.connect(library, "role_library", generate, "role_library")
    workflow.connect(script_node, "dialogue_script", generate, "dialogue_script")
    workflow.connect(script_node, "dialogue_script", subtitle, "dialogue_script")
    workflow.connect(generate, "generation_report", subtitle, "generation_report")
    workflow.connect(generate, "audio", save, "audio")
    generate_api = api_dialogue_generate("1", "6", "7", policy="shift", fit=True)
    generate_api["inputs"]["asr_enabled"] = True
    api = {
        "1": api_model(), "2": api_audio("role_a.wav"), "3": api_audio("role_b.wav"),
        "4": api_voice("角色A", "2"), "5": api_voice("角色B", "3"),
        "6": api_role_library(["4", "5"]), "7": api_dialogue_script("srt", script, "角色A"),
        "8": generate_api, "9": api_subtitle_rewrite("7", "8"),
        "10": api_save("8", "IndexTTS25_T8/subtitle_rewrite"),
    }
    return workflow.as_dict(), api


EXAMPLES = {
    "01_basic_voice_clone": basic_pair,
    "02_speed_comparison": speed_pair,
    "03_emotion_reference_audio": reference_emotion_pair,
    "04_emotion_vector": vector_emotion_pair,
    "05_emotion_text": text_emotion_pair,
    "06_random_sampling_long_text": sampling_pair,
    "07_multilingual_generation": multilingual_pair,
    "08_chinese_pronunciation": chinese_pronunciation_pair,
    "09_english_cmu_pronunciation": english_pronunciation_pair,
    "10_japanese_kana_pronunciation": japanese_pronunciation_pair,
    "11_multi_role_dialogue": multi_role_pair,
    "12_batch_dialogue_json": batch_dialogue_pair,
    "13_srt_multi_role": srt_dialogue_pair,
    "14_optional_acceleration": acceleration_pair,
    "15_auto_segment_preview": auto_segment_preview_pair,
    "16_pause_control": pause_control_pair,
    "17_target_duration": target_duration_pair,
    "18_audio_postprocess": audio_postprocess_pair,
    "19_cfm_advanced": cfm_advanced_pair,
    "20_asr_proofread": asr_proofread_pair,
    "21_timeline_editor": timeline_editor_pair,
    "22_subtitle_rewrite": subtitle_rewrite_pair,
}


def validate_ui(workflow: dict[str, Any]) -> None:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"])
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == len(workflow["links"])
    for link_id, origin_id, origin_slot, target_id, target_slot, data_type in workflow["links"]:
        origin = nodes[origin_id]
        target = nodes[target_id]
        assert origin["outputs"][origin_slot]["type"] == data_type
        assert link_id in origin["outputs"][origin_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert target["inputs"][target_slot]["type"] == data_type


def validate_api(prompt: dict[str, Any]) -> None:
    node_ids = set(prompt)
    for node_id, node in prompt.items():
        assert str(node_id).isdigit()
        assert "class_type" in node and isinstance(node.get("inputs"), dict)
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids
                assert isinstance(value[1], int) and value[1] >= 0
        if node["class_type"] == EMOTION_NODE:
            assert isinstance(node["inputs"].get("mode"), str)
            assert not any(isinstance(value, dict) for value in node["inputs"].values())


def main() -> int:
    UI_ROOT.mkdir(parents=True, exist_ok=True)
    API_ROOT.mkdir(parents=True, exist_ok=True)
    for name, factory in EXAMPLES.items():
        ui_workflow, api_prompt = factory()
        validate_ui(ui_workflow)
        validate_api(api_prompt)
        (UI_ROOT / f"{name}.json").write_text(
            json.dumps(ui_workflow, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (API_ROOT / f"{name}.json").write_text(
            json.dumps(api_prompt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Generated {len(EXAMPLES)} UI workflows and {len(EXAMPLES)} API prompts in {EXAMPLES_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
