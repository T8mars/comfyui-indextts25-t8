from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class ReferenceConditionCache:
    """Safe, content-addressed cache for extracted reference tensors."""

    def __init__(self, cache_dir=None, namespace: str = "", max_entries: int = 128):
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.namespace = str(namespace or "")
        self.max_entries = max(8, int(max_entries))
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._corruptions = 0
        self._pruned = 0
        self._clears = 0

    @property
    def enabled(self) -> bool:
        return self.cache_dir is not None

    def _path(self, kind: str, audio_path) -> Path | None:
        if not self.enabled or not audio_path:
            return None
        source = Path(audio_path)
        if not source.is_file():
            return None
        digest = hashlib.sha256()
        digest.update(b"indextts25-reference-condition-v1\0")
        digest.update(self.namespace.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(str(kind).encode("ascii", errors="ignore"))
        digest.update(b"\0")
        try:
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError:
            return None
        return self.cache_dir / str(kind) / f"{digest.hexdigest()}.safetensors"

    def load(self, kind: str, audio_path, device) -> dict[str, torch.Tensor] | None:
        target = self._path(kind, audio_path)
        if target is None:
            return None
        if not target.is_file():
            with self._lock:
                self._misses += 1
            return None
        try:
            with self._lock:
                tensors = load_file(str(target), device="cpu")
                os.utime(target, None)
                self._hits += 1
            return {name: tensor.to(device) for name, tensor in tensors.items()}
        except Exception:
            with self._lock:
                self._misses += 1
                self._corruptions += 1
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def save(self, kind: str, audio_path, tensors: dict[str, torch.Tensor]) -> Path | None:
        target = self._path(kind, audio_path)
        if target is None:
            return None
        prepared = {
            name: tensor.detach().to("cpu").contiguous()
            for name, tensor in tensors.items()
        }
        temporary = target.with_name(
            f".{target.stem}.{os.getpid()}.{threading.get_ident()}.tmp.safetensors"
        )
        try:
            with self._lock:
                target.parent.mkdir(parents=True, exist_ok=True)
                save_file(prepared, str(temporary))
                os.replace(temporary, target)
                self._writes += 1
                self._prune()
        except Exception:
            return None
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return target

    def _prune(self) -> None:
        if not self.cache_dir or not self.cache_dir.exists():
            return
        try:
            files = sorted(
                self.cache_dir.glob("*/*.safetensors"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        removed = 0
        for stale in files[self.max_entries :]:
            try:
                stale.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
        self._pruned += removed

    def clear(self) -> int:
        """Delete only safetensors entries below this cache's resolved directory."""

        if not self.enabled or not self.cache_dir.exists():
            return 0
        removed = 0
        with self._lock:
            try:
                files = list(self.cache_dir.glob("*/*.safetensors"))
            except OSError:
                files = []
            for target in files:
                try:
                    target.unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    pass
            self._clears += 1
        return removed

    def status(self) -> dict:
        with self._lock:
            try:
                files = list(self.cache_dir.glob("*/*.safetensors")) if self.enabled and self.cache_dir.exists() else []
                total_bytes = sum(path.stat().st_size for path in files)
            except OSError:
                files, total_bytes = [], 0
            requests = self._hits + self._misses
            return {
                "enabled": self.enabled,
                "directory": str(self.cache_dir) if self.cache_dir else "",
                "entries": len(files),
                "bytes": total_bytes,
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / requests, 4) if requests else None,
                "writes": self._writes,
                "corruptions": self._corruptions,
                "pruned": self._pruned,
                "clears": self._clears,
                "checked_at": time.time(),
            }
