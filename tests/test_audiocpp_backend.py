from __future__ import annotations

import asyncio

import runtime.audiocpp_backend as audiocpp_backend
from runtime.audiocpp_backend import build_command


def test_builds_isolated_index_25_cli_command(tmp_path):
    executable = tmp_path / "audiocpp_cli.exe"
    model = tmp_path / "IndexTTS2.5-GGUF"
    speaker = tmp_path / "speaker.wav"
    executable.write_bytes(b"binary")
    model.mkdir()
    speaker.write_bytes(b"wave")

    command = build_command(
        executable,
        model,
        speaker,
        tmp_path / "output.wav",
        "测试",
        "ZH",
        backend="cuda",
        duration_factor=1.2,
        memory_saver=True,
        emotion_vector=(0.1, 0.2, 0, 0, 0, 0, 0, 0.7),
        emotion_alpha=0.6,
    )

    assert command[:5] == [str(executable), "--task", "clon", "--family", "index_tts2"]
    assert "language=zh" in command
    assert "duration_factor=1.2" in command
    assert "index_tts2.mem_saver=true" in command
    assert "emotion_vector=0.1,0.2,0,0,0,0,0,0.7" in command


def test_probe_uses_async_fixed_argument_vector(monkeypatch, tmp_path):
    executable = tmp_path / "audiocpp_cli.exe"
    executable.write_bytes(b"binary")
    captured = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"--family index_tts2 --voice-ref file.wav", b""

    async def fake_create_process(*command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        audiocpp_backend.asyncio,
        "create_subprocess_exec",
        fake_create_process,
    )

    report = asyncio.run(audiocpp_backend.probe(executable))

    assert report["available"] is True
    assert captured["command"] == [str(executable.resolve()), "--help"]
    assert captured["kwargs"] == {"stdout": -1, "stderr": -1}
