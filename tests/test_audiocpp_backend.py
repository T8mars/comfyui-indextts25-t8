from __future__ import annotations

import asyncio

import pytest

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


class _CancellableProcess:
    def __init__(self):
        self.returncode = None
        self.started = asyncio.Event()
        self.terminated = False
        self.killed = False

    async def communicate(self):
        self.started.set()
        await asyncio.Event().wait()

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


async def _cancel_and_assert_reaped(coroutine, process: _CancellableProcess):
    task = asyncio.create_task(coroutine)
    await process.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated is True
    assert process.returncode is not None


def test_probe_cancellation_terminates_and_reaps_child(monkeypatch, tmp_path):
    executable = tmp_path / "audiocpp_cli.exe"
    executable.write_bytes(b"binary")
    process = _CancellableProcess()

    async def fake_create_process(*command, **kwargs):
        return process

    monkeypatch.setattr(
        audiocpp_backend.asyncio, "create_subprocess_exec", fake_create_process
    )
    asyncio.run(_cancel_and_assert_reaped(audiocpp_backend.probe(executable), process))


def test_generation_cancellation_terminates_and_reaps_child(monkeypatch, tmp_path):
    executable = tmp_path / "audiocpp_cli.exe"
    speaker = tmp_path / "speaker.wav"
    model = tmp_path / "model"
    executable.write_bytes(b"binary")
    speaker.write_bytes(b"wave")
    model.mkdir()
    process = _CancellableProcess()

    async def fake_create_process(*command, **kwargs):
        return process

    monkeypatch.setattr(
        audiocpp_backend.asyncio, "create_subprocess_exec", fake_create_process
    )
    generation = audiocpp_backend.run(
        executable,
        model,
        speaker,
        tmp_path / "output.wav",
        "测试",
        "ZH",
        backend="cpu",
        duration_factor=1.0,
        memory_saver=True,
    )
    asyncio.run(_cancel_and_assert_reaped(generation, process))


def test_process_cleanup_escalates_to_kill_and_awaits_exit(monkeypatch):
    process = _CancellableProcess()
    waits = 0

    async def slow_then_exit():
        nonlocal waits
        waits += 1
        if waits == 1:
            await asyncio.Event().wait()
        return process.returncode

    process.wait = slow_then_exit
    monkeypatch.setattr(audiocpp_backend, "PROCESS_TERMINATE_GRACE_SECONDS", 0.001)
    asyncio.run(audiocpp_backend._terminate_process(process))

    assert process.terminated is True
    assert process.killed is True
    assert waits == 2
