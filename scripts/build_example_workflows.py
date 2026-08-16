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
PRONUNCIATION_NODE = "T8_IndexTTS25_Pronunciation"
GENERATE_NODE = "T8_IndexTTS25_Generate"


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
        "ver": "0.2.0",
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
            widget_input("use_cuda_kernel", "BOOLEAN"),
            widget_input("release_after_run", "BOOLEAN"),
            widget_input("verify_hashes", "BOOLEAN"),
            widget_input("custom_model_path", "STRING", optional=True),
        ],
        [output("model", "T8_INDEXTTS25_MODEL"), output("model_info", "STRING")],
        ["IndexTTS-2.5", "auto", "auto", False, release_after_run, False, ""],
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
        "max_text_tokens_per_segment": 120,
        "segment_silence_ms": 200,
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
        ("max_text_tokens_per_segment", "INT"),
        ("segment_silence_ms", "INT"),
        ("text_normalization", "BOOLEAN"),
    )
    return workflow.add(
        SAMPLING_NODE,
        pos,
        (390, 490),
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
) -> int:
    return workflow.add(
        GENERATE_NODE,
        pos,
        (470, 390),
        [
            slot_input("model", "T8_INDEXTTS25_MODEL"),
            slot_input("speaker_audio", "AUDIO"),
            widget_input("text", "STRING"),
            widget_input("language", "COMBO"),
            widget_input("duration_factor", "FLOAT"),
            widget_input("seed", "INT"),
            slot_input("emotion", "T8_INDEXTTS25_EMOTION", optional=True),
            slot_input("sampling", "T8_INDEXTTS25_SAMPLING", optional=True),
        ],
        [output("audio", "AUDIO"), output("generation_info", "STRING")],
        [text, language, duration_factor, seed, "fixed"],
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
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "model": [model_id, 0],
        "speaker_audio": [speaker_id, 0],
        "text": text,
        "language": language,
        "duration_factor": duration_factor,
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
