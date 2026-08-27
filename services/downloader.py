from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

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
) -> None:
    manifest = load_manifest()
    repository = str(manifest["modelRepository"])
    revision = str(manifest["modelRevision"])
    target.mkdir(parents=True, exist_ok=True)
    if source == "huggingface":
        from huggingface_hub import snapshot_download

        selected = set(_selected_files(manifest, include_auxiliary=include_auxiliary))
        requested = sorted(
            selected.intersection([*missing, *mismatched])
            or selected
        )
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=str(target),
            allow_patterns=[
                *requested,
                "README.md",
                "LICENSE",
                "LICENSE_ZH.txt",
                "DISCLAIMER",
                "THIRD_PARTY_NOTICES.md",
            ],
            force_download=bool(set(mismatched).intersection(requested)),
        )
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
) -> ValidationReport:
    """Download or repair one complete, pinned IndexTTS 2.5 model directory."""

    if not accept_license:
        raise ValueError("下载模型前必须阅读并接受模型许可证和免责声明。")
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target / ".download.lock")):
        initial = validate_model_dir(
            target,
            verify_hashes=verify_hashes,
            include_auxiliary=not skip_auxiliary,
        )
        if not initial.valid:
            download_main_model(
                target,
                source,
                missing=initial.missing,
                mismatched=initial.mismatched,
                include_auxiliary=not skip_auxiliary,
            )
            if source == "modelscope":
                manifest = load_manifest()
                for relative_path in initial.mismatched:
                    if manifest["files"][relative_path].get("group") == "auxiliary":
                        target.joinpath(*relative_path.split("/")).unlink()
        if not skip_auxiliary:
            download_auxiliary_models(target, source)
        report = validate_model_dir(
            target,
            verify_hashes=verify_hashes,
            include_auxiliary=not skip_auxiliary,
        )
        report.require_valid()
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
