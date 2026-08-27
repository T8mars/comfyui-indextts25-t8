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
        if target is None or not target.is_file():
            return None
        try:
            tensors = load_file(str(target), device="cpu")
            os.utime(target, None)
            return {name: tensor.to(device) for name, tensor in tensors.items()}
        except Exception:
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
        for stale in files[self.max_entries :]:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass

    def status(self) -> dict:
        try:
            files = list(self.cache_dir.glob("*/*.safetensors")) if self.enabled and self.cache_dir.exists() else []
            total_bytes = sum(path.stat().st_size for path in files)
        except OSError:
            files, total_bytes = [], 0
        return {
            "enabled": self.enabled,
            "directory": str(self.cache_dir) if self.cache_dir else "",
            "entries": len(files),
            "bytes": total_bytes,
            "checked_at": time.time(),
        }
