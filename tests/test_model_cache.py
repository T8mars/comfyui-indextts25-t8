from __future__ import annotations

from pathlib import Path

from runtime import model_cache
from runtime.model_cache import CacheEntry, ModelCache
from runtime.types import ModelHandle


def _cached_entry(cache: ModelCache, handle: ModelHandle, *, users: int = 0) -> CacheEntry:
    entry = CacheEntry(model=object(), users=users)
    cache._entries[handle.cache_key] = entry
    return entry


def test_release_after_run_waits_for_all_current_users(tmp_path: Path, monkeypatch):
    cache = ModelCache()
    handle = ModelHandle(tmp_path, "cpu", False, release_after_run=True)
    model = object()
    entry = CacheEntry(model=model, users=2)
    cache._entries[handle.cache_key] = entry
    disposed = []
    monkeypatch.setattr(cache, "_dispose", lambda item, device: disposed.append((item, device)))

    cache.done(handle, entry, release=True)
    assert cache.size() == 1
    assert disposed == []

    cache.done(handle, entry)
    assert cache.size() == 0
    assert disposed == [(entry, "cpu")]


def test_manual_release_is_deferred_for_active_user(tmp_path: Path, monkeypatch):
    cache = ModelCache()
    handle = ModelHandle(tmp_path, "cpu", False)
    entry = CacheEntry(model=object(), users=1)
    cache._entries[handle.cache_key] = entry
    disposed = []
    monkeypatch.setattr(cache, "_dispose", lambda item, device: disposed.append(item))

    assert cache.release(handle) is True
    assert cache.size() == 1
    cache.done(handle, entry)
    assert cache.size() == 0
    assert disposed == [entry]


def test_optional_initialization_failure_falls_back(tmp_path: Path, monkeypatch):
    calls = []

    class FakeIndexTTS:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if kwargs["use_torch_compile"]:
                raise RuntimeError("compile init failed")
            self.use_cuda_kernel = False

    monkeypatch.setattr(model_cache, "_load_core_class", lambda: FakeIndexTTS)
    cache = ModelCache()
    handle = ModelHandle(
        tmp_path,
        "cpu",
        False,
        use_torch_compile=True,
        acceleration_requested="torch_compile",
        acceleration_effective="torch_compile",
    )
    entry = cache.acquire(handle)
    assert entry.model is not None
    assert len(calls) == 2
    assert calls[1]["use_torch_compile"] is False
    assert handle.acceleration_effective == "off"
    assert "自动回退" in handle.acceleration_note

    fresh_handle = ModelHandle(
        tmp_path,
        "cpu",
        False,
        use_torch_compile=True,
        acceleration_requested="torch_compile",
        acceleration_effective="torch_compile",
    )
    assert cache.acquire(fresh_handle) is entry
    assert fresh_handle.acceleration_effective == "off"
    assert "自动回退" in fresh_handle.acceleration_note


def test_recycle_after_runs_releases_at_threshold(tmp_path: Path, monkeypatch):
    cache = ModelCache()
    handle = ModelHandle(tmp_path, "cpu", False, recycle_after_runs=2)
    entry = _cached_entry(cache, handle, users=1)
    disposed = []
    monkeypatch.setattr(cache, "_dispose", lambda item, device: disposed.append((item, device)))

    cache.done(handle, entry)
    assert cache.size() == 1
    entry.users = 1
    cache.done(handle, entry)

    assert cache.size() == 0
    assert disposed == [(entry, "cpu")]


def test_idle_eviction_only_releases_unused_entries(tmp_path: Path, monkeypatch):
    cache = ModelCache()
    idle_handle = ModelHandle(tmp_path / "idle", "cpu", False)
    active_handle = ModelHandle(tmp_path / "active", "cpu", False)
    idle = _cached_entry(cache, idle_handle)
    active = _cached_entry(cache, active_handle, users=1)
    idle.last_used -= 600
    active.last_used -= 600
    disposed = []
    monkeypatch.setattr(cache, "_dispose", lambda item, device: disposed.append(item))

    assert cache.evict_idle(300) == 1
    assert cache.size() == 1
    assert disposed == [idle]
    assert cache.status()["entries"][0]["users"] == 1
