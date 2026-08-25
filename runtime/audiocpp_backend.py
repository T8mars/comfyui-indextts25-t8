"""Isolated optional audio.cpp CLI runner for ComfyUI."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


LANGUAGES = {"AUTO": "auto", "ZH": "zh", "EN": "en", "JA": "ja", "ES": "es", "AR": "ar"}


def _path(value, label: str, *, file: bool | None = None) -> Path:
    path = Path(str(value or "").strip()).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{label}不存在：{path}")
    if file is True and not path.is_file():
        raise ValueError(f"{label}必须是文件：{path}")
    return path


def probe(executable, timeout: float = 15.0) -> dict[str, Any]:
    try:
        binary = _path(executable, "audio.cpp 可执行文件", file=True)
        completed = subprocess.run(
            [str(binary), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, float(timeout)),
            check=False,
            shell=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        compatible = "--family" in output and "--voice-ref" in output
        return {
            "available": completed.returncode == 0 and compatible,
            "compatible_cli": compatible,
            "returncode": completed.returncode,
            "executable": str(binary),
            "summary": output[:2000],
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def build_command(
    executable,
    model_dir,
    speaker_wav,
    output_wav,
    text: str,
    language: str,
    *,
    backend: str,
    duration_factor: float,
    memory_saver: bool,
    emotion_text: str = "",
    emotion_audio=None,
    emotion_vector: Sequence[float] | None = None,
    emotion_alpha: float = 1.0,
) -> list[str]:
    binary = _path(executable, "audio.cpp 可执行文件", file=True)
    model = _path(model_dir, "audio.cpp IndexTTS2.5 GGUF 模型")
    speaker = _path(speaker_wav, "音色参考音频", file=True)
    output = Path(output_wav).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = str(text or "").strip()
    if not source:
        raise ValueError("待合成文本不能为空。")
    code = LANGUAGES.get(str(language).upper())
    if code is None:
        raise ValueError(f"audio.cpp 不支持该语言：{language}")
    if backend not in {"cuda", "cpu", "vulkan", "hip", "metal"}:
        raise ValueError(f"未知 audio.cpp 后端：{backend}")
    if not 0.5 <= float(duration_factor) <= 2.0:
        raise ValueError("时长系数必须在 0.5–2.0。")
    command = [
        str(binary), "--task", "clon", "--family", "index_tts2",
        "--model", str(model), "--backend", backend, "--text", source,
        "--voice-ref", str(speaker), "--out", str(output),
        "--request-option", f"language={code}",
        "--request-option", f"duration_factor={float(duration_factor):.6g}",
        "--session-option", f"index_tts2.mem_saver={'true' if memory_saver else 'false'}",
    ]
    if emotion_text.strip():
        command.extend(["--emotion", emotion_text.strip(), "--request-option", "use_emotion_text=true"])
    if emotion_audio:
        command.extend(["--audio", str(_path(emotion_audio, "情感参考音频", file=True))])
    if emotion_vector is not None:
        vector = [max(0.0, float(item)) for item in emotion_vector]
        if len(vector) != 8:
            raise ValueError("情感向量必须包含 8 个数值。")
        command.extend(["--request-option", "emotion_vector=" + ",".join(f"{item:.6g}" for item in vector)])
    if emotion_text.strip() or emotion_audio or emotion_vector is not None:
        command.extend(["--request-option", f"emotion_alpha={max(0.0, min(1.0, float(emotion_alpha))):.6g}"])
    return command


def run(*args, timeout: float = 3600, **kwargs) -> dict[str, Any]:
    command = build_command(*args, **kwargs)
    output = Path(command[command.index("--out") + 1])
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(30.0, float(timeout)),
        check=False,
        shell=False,
    )
    if completed.returncode != 0 or not output.is_file():
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"audio.cpp 推理失败（exit={completed.returncode}）：{detail[-3000:]}")
    return {
        "output_path": str(output),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "backend": kwargs.get("backend"),
        "family": "index_tts2",
        "experimental": True,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


__all__ = ["LANGUAGES", "build_command", "probe", "run"]
