from __future__ import annotations

from pathlib import Path

from runtime.model_cache import CacheEntry, ModelCache
from runtime.types import ModelHandle


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
