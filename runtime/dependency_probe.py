from __future__ import annotations

import importlib.util
from pathlib import Path


REQUIRED_MODULES = {
    "torch": "torch（请使用 ComfyUI 自带版本）",
    "torchaudio": "torchaudio（版本需与 torch 对应）",
    "librosa": "librosa",
    "omegaconf": "omegaconf",
    "einops": "einops",
    "transformers": "transformers",
    "sentencepiece": "sentencepiece",
    "tiktoken": "tiktoken",
    "fugashi": "fugashi",
    "munch": "munch",
    "scipy": "scipy",
}


def missing_dependencies() -> list[str]:
    return [package for module, package in REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None]


def require_runtime_dependencies(plugin_root: Path) -> None:
    missing = missing_dependencies()
    if not missing:
        return
    requirements = plugin_root / "requirements.txt"
    raise RuntimeError(
        "IndexTTS 2.5 推理依赖不完整："
        + ", ".join(missing)
        + "。请先确认 ComfyUI 自身依赖安装完整，再使用 ComfyUI 的 Python 执行：python -m pip install -r \""
        + str(requirements)
        + "\"。不要单独降级或重装 torch。"
    )
