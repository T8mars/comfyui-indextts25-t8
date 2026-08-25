from __future__ import annotations

import gc
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .dependency_probe import require_runtime_dependencies
from .types import ModelHandle


LOGGER = logging.getLogger("comfyui-indextts25-T8")
PLUGIN_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class CacheEntry:
    model: Any
    lock: threading.RLock = field(default_factory=threading.RLock)
    last_used: float = field(default_factory=time.monotonic)
    users: int = 0
    pending_release: bool = False
    acceleration_effective: str | None = None
    acceleration_note: str = ""
    completed_runs: int = 0


def _load_core_class():
    require_runtime_dependencies(PLUGIN_ROOT)
    root_string = str(PLUGIN_ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    existing = sys.modules.get("indextts")
    if existing is not None:
        existing_file = Path(getattr(existing, "__file__", "")).resolve()
        if PLUGIN_ROOT not in existing_file.parents:
            raise RuntimeError(
                "检测到其他 IndexTTS 包已先被载入，可能导致版本串用。请移除重复节点后重启 ComfyUI。"
            )
    try:
        from indextts.infer_v2_5 import IndexTTS2
    except Exception as exc:
        raise RuntimeError(
            f"无法导入节点内置的 IndexTTS 2.5 核心（{type(exc).__name__}: {exc}）。"
            f"请检查 {PLUGIN_ROOT / 'requirements.txt'}。"
        ) from exc
    return IndexTTS2


class ModelCache:
    def __init__(self) -> None:
        self._entries: dict[tuple, CacheEntry] = {}
        self._guard = threading.RLock()

    def acquire(self, handle: ModelHandle) -> CacheEntry:
        key = handle.cache_key
        with self._guard:
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_used = time.monotonic()
                entry.users += 1
                if entry.acceleration_effective is not None:
                    handle.acceleration_effective = entry.acceleration_effective
                    handle.acceleration_note = entry.acceleration_note
                return entry

            IndexTTS2 = _load_core_class()
            LOGGER.info("Loading IndexTTS 2.5 from %s on %s", handle.model_dir, handle.device)
            constructor_kwargs = {
                "cfg_path": str(handle.model_dir / "config.yaml"),
                "model_dir": str(handle.model_dir),
                "use_bf16": handle.use_bf16,
                "device": handle.device,
                "use_cuda_kernel": handle.use_cuda_kernel,
                "use_deepspeed": handle.use_deepspeed,
                "use_accel": handle.use_accel,
                "use_torch_compile": handle.use_torch_compile,
                "use_qwen_emo": False,
            }
            try:
                model = IndexTTS2(**constructor_kwargs)
            except Exception as exc:
                if handle.acceleration_effective == "off":
                    raise
                LOGGER.exception("Optional acceleration initialization failed; reloading normal mode")
                gc.collect()
                if handle.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                constructor_kwargs.update(
                    use_cuda_kernel=False,
                    use_deepspeed=False,
                    use_accel=False,
                    use_torch_compile=False,
                )
                model = IndexTTS2(**constructor_kwargs)
                handle.acceleration_effective = "off"
                handle.acceleration_note = (
                    f"可选加速初始化失败（{type(exc).__name__}: {exc}），已自动回退普通模式"
                )
            if handle.use_cuda_kernel and not bool(getattr(model, "use_cuda_kernel", False)):
                handle.acceleration_effective = "off"
                handle.acceleration_note = "BigVGAN CUDA 融合核加载失败，上游已自动回退普通实现"
            entry = CacheEntry(
                model=model,
                users=1,
                acceleration_effective=handle.acceleration_effective,
                acceleration_note=handle.acceleration_note,
            )
            self._entries[key] = entry
            return entry

    def release(self, handle: ModelHandle) -> bool:
        """Request eviction, deferring disposal until all current users finish."""
        entry = None
        with self._guard:
            current = self._entries.get(handle.cache_key)
            if current is None:
                return False
            current.pending_release = True
            if current.users == 0:
                entry = self._entries.pop(handle.cache_key)
        if entry is None:
            return True
        self._dispose(entry, handle.device)
        return True

    def done(self, handle: ModelHandle, entry: CacheEntry, *, release: bool = False) -> None:
        """Return an acquired entry and evict it safely when requested."""
        dispose = None
        with self._guard:
            current = self._entries.get(handle.cache_key)
            if current is not entry:
                return
            current.users = max(0, current.users - 1)
            current.completed_runs += 1
            current.last_used = time.monotonic()
            if release or (
                int(handle.recycle_after_runs) > 0
                and current.completed_runs >= int(handle.recycle_after_runs)
            ):
                current.pending_release = True
            if current.users == 0 and current.pending_release:
                dispose = self._entries.pop(handle.cache_key)
        if dispose is not None:
            self._dispose(dispose, handle.device)

    @staticmethod
    def _dispose(entry: CacheEntry, device: str) -> None:
        entry.model = None
        del entry
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self) -> int:
        with self._guard:
            count = len(self._entries)
            self._entries.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return count

    def evict_idle(self, idle_seconds: float) -> int:
        """Evict only this extension's idle, currently unused model entries."""
        threshold = max(0.0, float(idle_seconds))
        now = time.monotonic()
        removed: list[tuple[CacheEntry, str]] = []
        with self._guard:
            for key, entry in list(self._entries.items()):
                if entry.users == 0 and now - entry.last_used >= threshold:
                    self._entries.pop(key)
                    removed.append((entry, str(key[1])))
        for entry, device in removed:
            self._dispose(entry, device)
        return len(removed)

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._guard:
            entries = [
                {
                    "device": str(key[1]),
                    "precision": "bfloat16" if bool(key[2]) else "float32",
                    "users": entry.users,
                    "completed_runs": entry.completed_runs,
                    "idle_seconds": round(max(0.0, now - entry.last_used), 3),
                    "pending_release": entry.pending_release,
                    "acceleration": entry.acceleration_effective or "off",
                }
                for key, entry in self._entries.items()
            ]
        cuda = {"available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            cuda.update(
                allocated_mb=round(torch.cuda.memory_allocated() / (1024**2), 2),
                reserved_mb=round(torch.cuda.memory_reserved() / (1024**2), 2),
                max_allocated_mb=round(torch.cuda.max_memory_allocated() / (1024**2), 2),
            )
        return {"cached_models": len(entries), "entries": entries, "cuda": cuda}

    def size(self) -> int:
        with self._guard:
            return len(self._entries)


MODEL_CACHE = ModelCache()
