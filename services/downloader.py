from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from filelock import FileLock

try:
    from .model_store import (
        PLUGIN_ROOT,
        ValidationReport,
        default_model_target,
        load_manifest,
        validate_model_dir,
    )
except ImportError:
    from services.model_store import (
        PLUGIN_ROOT,
        ValidationReport,
        default_model_target,
        load_manifest,
        validate_model_dir,
    )


LICENSE_NOTICE = (
    "下载即表示你已阅读并接受节点目录中的 LICENSE、LICENSE_ZH.txt 与 DISCLAIMER。\n"
    "本项目是第三方衍生集成；原始权利人不对本衍生品背书、担保或承担责任。"
)
MINIMUM_FREE_BYTES = 512 * 1024 * 1024
DISK_RESERVE_BYTES = 1024 * 1024 * 1024


class ModelDownloadProgress:
    """Translate a resumable model transfer into stable bundle-level events."""

    def __init__(
        self,
        manifest: dict,
        source: str,
        callback: Callable[[dict], None] | None,
        *,
        include_auxiliary: bool,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.files = {
            relative: metadata
            for relative, metadata in manifest["files"].items()
            if include_auxiliary or metadata.get("group") != "auxiliary"
        }
        self.source = source
        self.callback = callback or (lambda event: None)
        self.clock = clock
        self.bundle_total = sum(int(item["size"]) for item in self.files.values())
        self.required: list[str] = []
        self.required_total = 0
        self.completed = 0
        self.current_file = ""
        self.current_index = 0
        self.current_received = 0
        self.current_total = 0
        self.network_bytes = 0
        self.started_at = self.clock()
        self.last_emit = 0.0

    def emit(self, phase: str, *, force: bool = False, **values) -> None:
        now = self.clock()
        if not force and now - self.last_emit < 0.2:
            return
        self.last_emit = now
        self.callback(
            {
                "phase": phase,
                "source": self.source,
                "bundle_total": self.bundle_total,
                **values,
            }
        )

    def scan(self, relative: str, processed: int, total: int) -> None:
        fraction = processed / max(1, total)
        self.emit(
            "scanning",
            file=relative,
            overall_fraction=fraction * 0.1,
            phase_fraction=fraction,
            message=f"校验现有模型：{relative}",
        )

    def preflight(self, required: list[str], free_bytes: int) -> None:
        self.required = list(required)
        self.required_total = sum(int(self.files[item]["size"]) for item in required)
        warning = free_bytes < self.required_total + DISK_RESERVE_BYTES
        self.started_at = self.clock()
        self.emit(
            "preflight",
            force=True,
            required_bytes=self.required_total,
            available_bytes=free_bytes,
            disk_warning=warning,
            file_count=len(required),
            overall_fraction=0.1,
            phase_fraction=1.0,
            message=(
                f"磁盘预检完成，需要下载或修复 {len(required)} 个文件"
                + ("；可用空间低于保守估算" if warning and required else "")
            ),
        )

    def begin_file(self, relative: str, index: int) -> None:
        self.current_file = relative
        self.current_index = index
        self.current_received = 0
        self.current_total = int(self.files[relative]["size"])
        self._transfer(force=True)

    def resume_file(self, received: int) -> None:
        self.current_received = min(self.current_total, max(self.current_received, int(received)))
        self._transfer(force=True)

    def update_file(self, received: int) -> None:
        value = min(self.current_total, max(self.current_received, int(received)))
        self.network_bytes += max(0, value - self.current_received)
        self.current_received = value
        self._transfer()

    def complete_file(self) -> None:
        self.current_received = self.current_total
        self._transfer(force=True)
        self.completed += self.current_total
        self.current_received = 0
        self.current_total = 0

    def _transfer(self, *, force: bool = False) -> None:
        transferred = min(self.required_total, self.completed + self.current_received)
        fraction = transferred / max(1, self.required_total)
        elapsed = max(0.001, self.clock() - self.started_at)
        speed = self.network_bytes / elapsed
        remaining = max(0, self.required_total - transferred)
        self.emit(
            "downloading",
            force=force,
            file=self.current_file,
            file_index=self.current_index,
            file_count=len(self.required),
            received=transferred,
            total=self.required_total,
            bytes_per_second=round(speed),
            eta_seconds=round(remaining / speed) if speed > 0 else None,
            overall_fraction=0.1 + fraction * 0.8,
            phase_fraction=fraction,
            message=f"下载 {self.current_file}",
        )

    def verify(self, relative: str, processed: int, total: int) -> None:
        fraction = processed / max(1, total)
        self.emit(
            "verifying",
            file=relative,
            overall_fraction=0.9 + fraction * 0.1,
            phase_fraction=fraction,
            message=f"SHA-256 校验：{relative}",
        )

    def done(self) -> None:
        self.emit(
            "complete",
            force=True,
            overall_fraction=1.0,
            phase_fraction=1.0,
            message="完整模型下载和 SHA-256 校验完成",
        )


def _copy_snapshot(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def _file_source(manifest: dict, relative_path: str) -> tuple[str, str]:
    return (
        str(manifest["modelRepository"]),
        str(manifest["modelRevision"]),
    )


def _selected_files(manifest: dict, *, include_auxiliary: bool) -> list[str]:
    return [
        relative
        for relative, metadata in manifest["files"].items()
        if include_auxiliary or metadata.get("group") != "auxiliary"
    ]


def download_main_model(
    target: Path,
    source: str,
    *,
    missing: tuple[str, ...] = (),
    mismatched: tuple[str, ...] = (),
    include_auxiliary: bool = True,
    progress: ModelDownloadProgress | None = None,
) -> None:
    manifest = load_manifest()
    repository = str(manifest["modelRepository"])
    revision = str(manifest["modelRevision"])
    target.mkdir(parents=True, exist_ok=True)
    if source == "huggingface":
        from huggingface_hub import hf_hub_download

        if progress is not None:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            from huggingface_hub.file_download import get_local_download_paths

        selected = set(_selected_files(manifest, include_auxiliary=include_auxiliary))
        requested = sorted(
            selected.intersection([*missing, *mismatched])
            or selected
        )
        for index, relative in enumerate(requested, start=1):
            if progress is not None:
                progress.begin_file(relative, index)
            force_download = relative in mismatched
            incomplete_path = None
            if progress is not None:
                try:
                    metadata = get_hf_file_metadata(
                        hf_hub_url(repository, relative, revision=revision)
                    )
                    if metadata.etag:
                        incomplete_path = get_local_download_paths(
                            target, relative
                        ).incomplete_path(metadata.etag)
                except Exception:
                    incomplete_path = None

            started_wall = time.time()
            if (
                progress is not None
                and not force_download
                and incomplete_path is not None
                and incomplete_path.is_file()
            ):
                progress.resume_file(incomplete_path.stat().st_size)

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    hf_hub_download,
                    repo_id=repository,
                    revision=revision,
                    filename=relative,
                    local_dir=str(target),
                    force_download=force_download,
                )
                while not future.done():
                    if progress is not None and incomplete_path is not None:
                        try:
                            stat = incomplete_path.stat()
                            if not force_download or stat.st_mtime >= started_wall:
                                progress.update_file(stat.st_size)
                        except FileNotFoundError:
                            pass
                    time.sleep(0.2)
                future.result()
                if progress is not None:
                    progress.complete_file()
    elif source == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            optional_requirements = PLUGIN_ROOT / "requirements-modelscope.txt"
            raise RuntimeError(
                "ModelScope 下载支持是可选项，请先使用 ComfyUI 的 Python 执行："
                f'python -m pip install -r "{optional_requirements}"'
            ) from exc

        upstream_repository = str(manifest["upstreamModelRepository"])
        downloaded = Path(snapshot_download(model_id=upstream_repository)).resolve()
        _copy_snapshot(downloaded, target)
        supplemental = {
            relative_path
            for relative_path in [*missing, *mismatched]
            if manifest["files"][relative_path].get("group") != "auxiliary"
            and manifest["files"][relative_path].get("sourceRepository")
            not in {None, upstream_repository}
        }
        if supplemental:
            from modelscope.hub.file_download import model_file_download

            for relative_path in sorted(supplemental):
                file_repository = str(
                    manifest["files"][relative_path]["sourceRepository"]
                )
                source_path = Path(
                    model_file_download(
                        model_id=file_repository, file_path=relative_path
                    )
                ).resolve()
                destination = target / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
    else:
        raise ValueError(f"未知下载源：{source}")


def download_auxiliary_models(target: Path, source: str) -> None:
    root_string = str(PLUGIN_ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from indextts.utils.model_download import ensure_models_available, set_download_source

    set_download_source(source)
    ensure_models_available(str(target))


def ensure_model_bundle(
    target: Path,
    source: str = "huggingface",
    *,
    accept_license: bool,
    verify_hashes: bool = True,
    skip_auxiliary: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> ValidationReport:
    """Download or repair one complete, pinned IndexTTS 2.5 model directory."""

    if not accept_license:
        raise ValueError("下载模型前必须阅读并接受模型许可证和免责声明。")
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    reporter = ModelDownloadProgress(
        manifest,
        source,
        progress,
        include_auxiliary=not skip_auxiliary,
    )
    with FileLock(str(target / ".download.lock")):
        initial = validate_model_dir(
            target,
            verify_hashes=verify_hashes,
            include_auxiliary=not skip_auxiliary,
            progress=reporter.scan,
        )
        required = list(dict.fromkeys([*initial.missing, *initial.mismatched]))
        free_bytes = shutil.disk_usage(target).free
        reporter.preflight(required, free_bytes)
        if required and free_bytes < MINIMUM_FREE_BYTES:
            raise RuntimeError(
                "模型目录可用空间不足 512 MiB，无法安全继续下载；请清理空间或更换目录。"
            )
        if not initial.valid:
            download_main_model(
                target,
                source,
                missing=initial.missing,
                mismatched=initial.mismatched,
                include_auxiliary=not skip_auxiliary,
                progress=reporter,
            )
            if source == "modelscope":
                for relative_path in initial.mismatched:
                    if manifest["files"][relative_path].get("group") == "auxiliary":
                        target.joinpath(*relative_path.split("/")).unlink()
        if not skip_auxiliary:
            download_auxiliary_models(target, source)
        report = validate_model_dir(
            target,
            verify_hashes=verify_hashes,
            include_auxiliary=not skip_auxiliary,
            progress=reporter.verify,
        )
        report.require_valid()
        reporter.done()
        return report


def _default_target_from_layout() -> Path | None:
    if PLUGIN_ROOT.parent.name.lower() == "custom_nodes":
        return default_model_target(PLUGIN_ROOT.parent.parent)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载并校验 IndexTTS 2.5 正式模型")
    parser.add_argument("--target", type=Path, help="目标模型目录，例如 ComfyUI/models/TTS/IndexTTS-2.5")
    parser.add_argument("--comfy-root", type=Path, help="ComfyUI 根目录；目标将自动放入 models/TTS/IndexTTS-2.5")
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="huggingface")
    parser.add_argument("--accept-license", action="store_true", help="确认已接受模型许可证和免责声明")
    parser.add_argument("--verify-only", action="store_true", help="只执行完整 SHA-256 校验，不下载")
    parser.add_argument("--skip-aux", action="store_true", help="跳过 Wav2Vec2-BERT、BigVGAN 等辅助模型")
    return parser


def resolve_target(args: argparse.Namespace) -> Path:
    if args.target and args.comfy_root:
        raise ValueError("--target 与 --comfy-root 只能使用一个。")
    if args.target:
        return args.target.expanduser().resolve()
    if args.comfy_root:
        return default_model_target(args.comfy_root)
    detected = _default_target_from_layout()
    if detected is not None:
        return detected
    raise ValueError("当前节点不在 ComfyUI/custom_nodes 下，请显式提供 --target 或 --comfy-root。")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        target = resolve_target(args)
    except ValueError as exc:
        parser.error(str(exc))

    print(f">> 目标目录：{target}")
    if args.verify_only:
        report = validate_model_dir(
            target,
            verify_hashes=True,
            include_auxiliary=not args.skip_aux,
        )
        report.require_valid()
        print(">> IndexTTS 2.5 正式模型 SHA-256 校验通过。")
        return 0

    if not args.accept_license:
        parser.error("下载模型前必须阅读许可证并传入 --accept-license。")
    print(LICENSE_NOTICE)

    print(f">> 从 {args.source} 下载或修复 IndexTTS 2.5 完整模型……")
    ensure_model_bundle(
        target,
        args.source,
        accept_license=True,
        verify_hashes=True,
        skip_auxiliary=args.skip_aux,
    )
    print(">> 正式模型 SHA-256 校验通过。")

    manifest = load_manifest()
    print(f">> 模型已就绪：{target}")
    print(f">> 固定版本：{manifest['modelRevision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
