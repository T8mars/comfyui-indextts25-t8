"""Command-line interface for the official IndexTTS 2.5 inference path."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any


LANGUAGES = ("ZH", "EN", "JA", "ES", "AR")
ACCELERATION_MODES = (
    "off",
    "bigvgan_cuda",
    "gpt_accel",
    "torch_compile",
    "deepspeed",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndexTTS 2.5 multilingual command-line synthesis"
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("text", nargs="?", help="Text to synthesize")
    text_group.add_argument("--text-file", type=Path, help="UTF-8 text file")
    parser.add_argument(
        "-v", "--voice", type=Path, required=True, help="Speaker reference audio"
    )
    parser.add_argument(
        "-o", "--output-path", type=Path, default=Path("gen.wav"), help="Output WAV"
    )
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("-l", "--language", choices=LANGUAGES, default="ZH")
    parser.add_argument("--device", default=None, help="cuda:0, cpu, mps, or xpu")
    parser.add_argument("--reference-device", default=None)
    parser.add_argument(
        "--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Deprecated alias for --precision fp16"
    )
    parser.add_argument("--acceleration", choices=ACCELERATION_MODES, default="off")
    parser.add_argument("--duration-factor", type=float, default=1.0)
    parser.add_argument("--target-duration", type=float, default=None, help="Seconds")
    parser.add_argument("--emotion-reference", type=Path)
    parser.add_argument("--emotion-alpha", type=float, default=1.0)
    parser.add_argument("--emotion-text", default=None)
    parser.add_argument(
        "--emotion-vector",
        default=None,
        help="Eight comma-separated values: happy,angry,sad,fear,disgust,melancholy,surprise,calm",
    )
    parser.add_argument("--random-emotion", action="store_true")
    parser.add_argument("--reuse-speaker-condition-for-emotion", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--max-mel-tokens", type=int, default=1500)
    parser.add_argument("--max-text-tokens", type=int, default=120)
    parser.add_argument("--segment-silence-ms", type=int, default=200)
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--cfg-rate", type=float, default=0.7)
    parser.add_argument("--cfm-temperature", type=float, default=1.0)
    parser.add_argument(
        "--no-text-normalization", action="store_true", help="Keep text as entered"
    )
    parser.add_argument("-f", "--force", action="store_true")
    return parser


def parse_emotion_vector(value: str | None) -> list[float] | None:
    if value is None or not str(value).strip():
        return None
    try:
        vector = [float(item.strip()) for item in str(value).split(",")]
    except ValueError as exc:
        raise ValueError("emotion vector must contain numeric values") from exc
    if len(vector) != 8:
        raise ValueError("emotion vector must contain exactly eight values")
    if any(not 0.0 <= item <= 1.0 for item in vector):
        raise ValueError("emotion vector values must be between 0 and 1")
    if sum(vector) > 1.000001:
        raise ValueError("emotion vector values must sum to at most 1")
    return vector


def resolve_device(torch_module: Any, requested: str | None) -> str:
    if requested:
        return str(requested)
    if torch_module.cuda.is_available():
        return "cuda:0"
    if hasattr(torch_module, "xpu") and torch_module.xpu.is_available():
        return "xpu"
    if (
        hasattr(torch_module, "backends")
        and hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def resolve_precision(
    torch_module: Any, requested: str, device: str
) -> tuple[bool, bool, str]:
    precision = str(requested)
    if not str(device).startswith("cuda"):
        return False, False, "fp32"
    if precision == "auto":
        precision = (
            "bf16"
            if bool(getattr(torch_module.cuda, "is_bf16_supported", lambda: False)())
            else "fp16"
        )
    return precision == "bf16", precision == "fp16", precision


def _read_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")
    else:
        text = str(args.text or "")
    text = text.strip()
    if not text:
        raise ValueError("text is empty")
    return text


def _consume(generator: Any) -> Any:
    last = None
    for last in generator:
        pass
    if last is None:
        raise RuntimeError("IndexTTS did not return an output")
    return last


def validate_args(args: argparse.Namespace) -> None:
    if not 0.5 <= float(args.duration_factor) <= 2.0:
        raise ValueError("duration-factor must be between 0.5 and 2.0")
    if args.target_duration is not None and not 0.1 <= float(args.target_duration) <= 3600:
        raise ValueError("target-duration must be between 0.1 and 3600 seconds")
    if not 0.0 <= float(args.emotion_alpha) <= 1.0:
        raise ValueError("emotion-alpha must be between 0 and 1")
    if not 0 <= int(args.seed) <= 0xFFFFFFFF:
        raise ValueError("seed must be between 0 and 4294967295")
    if not 5 <= int(args.diffusion_steps) <= 100:
        raise ValueError("diffusion-steps must be between 5 and 100")
    if not 0.0 <= float(args.cfg_rate) <= 1.5:
        raise ValueError("cfg-rate must be between 0 and 1.5")
    if not 0.1 <= float(args.cfm_temperature) <= 1.5:
        raise ValueError("cfm-temperature must be between 0.1 and 1.5")
    parse_emotion_vector(args.emotion_vector)


def run(args: argparse.Namespace) -> Path:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    validate_args(args)
    text = _read_text(args)
    voice = args.voice.resolve()
    model_dir = args.model_dir.resolve()
    config = (args.config or (model_dir / "config.yaml")).resolve()
    output = args.output_path.resolve()
    if not voice.is_file():
        raise FileNotFoundError(f"speaker reference does not exist: {voice}")
    if args.emotion_reference is not None and not args.emotion_reference.is_file():
        raise FileNotFoundError(
            f"emotion reference does not exist: {args.emotion_reference}"
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    if not config.is_file():
        raise FileNotFoundError(f"config does not exist: {config}")
    if output.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    from indextts.infer_v2_5 import IndexTTS2

    device = resolve_device(torch, args.device)
    requested_precision = "fp16" if args.fp16 else args.precision
    use_bf16, use_fp16, effective_precision = resolve_precision(
        torch, requested_precision, device
    )
    if device == "cpu":
        print("WARNING: CPU inference can be very slow.")
    print(
        f"IndexTTS 2.5 CLI: device={device}, precision={effective_precision}, "
        f"language={args.language}, acceleration={args.acceleration}"
    )
    tts = IndexTTS2(
        cfg_path=str(config),
        model_dir=str(model_dir),
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        device=device,
        reference_device=args.reference_device,
        use_cuda_kernel=args.acceleration == "bigvgan_cuda",
        use_deepspeed=args.acceleration == "deepspeed",
        use_accel=args.acceleration == "gpt_accel",
        use_torch_compile=args.acceleration == "torch_compile",
        use_qwen_emo=bool(args.emotion_text),
        reuse_spk_cond_for_emo=bool(args.reuse_speaker_condition_for_emotion),
    )
    vector = parse_emotion_vector(args.emotion_vector)
    _consume(
        tts.infer(
            spk_audio_prompt=str(voice),
            text=text,
            output_path=str(output),
            lang=args.language,
            emo_audio_prompt=(
                str(args.emotion_reference.resolve())
                if args.emotion_reference is not None
                else None
            ),
            emo_alpha=float(args.emotion_alpha),
            emo_vector=vector,
            use_emo_text=bool(args.emotion_text),
            emo_text=args.emotion_text,
            use_random=bool(args.random_emotion),
            do_sample=True,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
            num_beams=int(args.num_beams),
            repetition_penalty=float(args.repetition_penalty),
            length_penalty=float(args.length_penalty),
            max_mel_tokens=int(args.max_mel_tokens),
            max_text_tokens_per_segment=int(args.max_text_tokens),
            interval_silence=int(args.segment_silence_ms),
            text_normalization=not bool(args.no_text_normalization),
            duration_factor=float(args.duration_factor),
            target_duration=args.target_duration,
            seed=int(args.seed),
            diffusion_steps=int(args.diffusion_steps),
            inference_cfg_rate=float(args.cfg_rate),
            cfm_temperature=float(args.cfm_temperature),
        )
    )
    if not output.is_file():
        raise RuntimeError("IndexTTS finished without creating the output WAV")
    print(f"Saved: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
