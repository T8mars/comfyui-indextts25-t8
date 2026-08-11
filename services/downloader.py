from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from filelock import FileLock

try:
    from .model_store import PLUGIN_ROOT, default_model_target, load_manifest, validate_model_dir
except ImportError:
    from services.model_store import PLUGIN_ROOT, default_model_target, load_manifest, validate_model_dir


LICENSE_NOTICE = (
    "下载即表示你已阅读并接受节点目录中的 LICENSE、LICENSE_ZH.txt 与 DISCLAIMER。\n"
    "本项目是第三方衍生集成；原始权利人不对本衍生品背书、担保或承担责任。"
)


def _copy_snapshot(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def download_main_model(target: Path, source: str) -> None:
    manifest = load_manifest()
    repository = str(manifest["modelRepository"])
    revision = str(manifest["modelRevision"])
    target.mkdir(parents=True, exist_ok=True)
    if source == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=str(target),
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

        downloaded = Path(snapshot_download(model_id=repository, revision=revision)).resolve()
        _copy_snapshot(downloaded, target)
    else:
        raise ValueError(f"未知下载源：{source}")


def download_auxiliary_models(target: Path) -> None:
    root_string = str(PLUGIN_ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from indextts.utils.model_download import ensure_models_available

    ensure_models_available(str(target))


def _default_target_from_layout() -> Path | None:
    if PLUGIN_ROOT.parent.name.lower() == "custom_nodes":
        return default_model_target(PLUGIN_ROOT.parent.parent)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载并校验 IndexTTS 2.5 正式模型")
    parser.add_argument("--target", type=Path, help="目标模型目录，例如 ComfyUI/models/TTS/IndexTTS-2.5")
    parser.add_argument("--comfy-root", type=Path, help="ComfyUI 根目录；目标将自动放入 models/TTS/IndexTTS-2.5")
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope")
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
        report = validate_model_dir(target, verify_hashes=True)
        report.require_valid()
        print(">> IndexTTS 2.5 正式模型 SHA-256 校验通过。")
        return 0

    if not args.accept_license:
        parser.error("下载模型前必须阅读许可证并传入 --accept-license。")
    print(LICENSE_NOTICE)

    target.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target / ".download.lock")):
        quick = validate_model_dir(target, verify_hashes=False)
        if not quick.valid:
            print(f">> 从 {args.source} 下载正式 IndexTTS 2.5 模型……")
            download_main_model(target, args.source)
        report = validate_model_dir(target, verify_hashes=True)
        report.require_valid()
        print(">> 正式模型 SHA-256 校验通过。")
        if not args.skip_aux:
            print(">> 准备 IndexTTS 2.5 辅助模型……")
            download_auxiliary_models(target)

    manifest = load_manifest()
    print(f">> 模型已就绪：{target}")
    print(f">> 固定版本：{manifest['modelRevision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
