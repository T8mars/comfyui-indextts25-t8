"""Run a reproducible five-language IndexTTS 2.5 quality regression.

This is an opt-in real-model test.  It never downloads the IndexTTS model or
reference audio and writes every WAV plus a machine-readable report into
``--output-dir``.  When ASR is enabled, the selected Whisper model may be
downloaded into ``<output-dir>/asr_models`` by that optional backend.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indextts.speech_rate_guard import assess_segment_speech_rates
from indextts.utils.audio_io import load_audio_file
from quality_regression import (
    QUALITY_CASES,
    SUPPORTED_LANGUAGES,
    analyze_waveform,
    build_baseline_snapshot,
    compare_quality_reports,
    evaluate_quality_report,
    summarize_segment_rates,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("quality-regression"))
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=SUPPORTED_LANGUAGES,
        default=list(SUPPORTED_LANGUAGES),
    )
    parser.add_argument("--device", default=None, help="cuda:0, cpu, mps, or xpu")
    parser.add_argument(
        "--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--reference-device", default=None)
    parser.add_argument(
        "--vram-profile",
        choices=("native", "8gb", "24gb"),
        default="native",
        help="Run natively or enforce the formal 8 GB / 24 GB regression profile.",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-text-tokens", type=int, default=48)
    parser.add_argument("--segment-silence-ms", type=int, default=180)
    parser.add_argument(
        "--asr-backend",
        choices=("off", "auto", "openai_whisper", "faster_whisper"),
        default="off",
    )
    parser.add_argument("--asr-model", default="base")
    parser.add_argument(
        "--asr-model-for",
        action="append",
        default=[],
        metavar="LANG=MODEL",
        help="Override one language, for example AR=small.",
    )
    parser.add_argument("--asr-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--asr-download-root",
        type=Path,
        help="Persistent Whisper cache directory (defaults to <output-dir>/asr_models).",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--write-baseline",
        type=Path,
        help="Write a portable baseline snapshot without local paths/transcripts.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _resolve_precision(torch_module: Any, requested: str, device: str) -> tuple[bool, bool]:
    if requested == "bf16":
        return True, False
    if requested == "fp16":
        return False, True
    if requested == "fp32" or not str(device).startswith("cuda"):
        return False, False
    bf16_supported = bool(
        getattr(torch_module.cuda, "is_bf16_supported", lambda: False)()
    )
    return bf16_supported, not bf16_supported


def _consume_generation(generator: Any) -> Any:
    last = None
    for last in generator:
        pass
    if last is None:
        raise RuntimeError("IndexTTS did not return an output")
    return last


def _parse_asr_model_overrides(values: list[str]) -> dict[str, str]:
    allowed_models = {"tiny", "base", "small", "medium", "turbo"}
    result: dict[str, str] = {}
    for raw in values:
        language, separator, model = str(raw).partition("=")
        language, model = language.strip().upper(), model.strip().lower()
        if not separator or language not in SUPPORTED_LANGUAGES or model not in allowed_models:
            raise SystemExit(f"Invalid --asr-model-for {raw!r}; expected LANG=MODEL.")
        result[language] = model
    return result


def _configure_vram_profile(torch_module: Any, profile: str, device: str) -> dict[str, Any]:
    if profile == "native":
        return {"name": profile, "budget_bytes": None, "simulated": False}
    if not str(device).startswith("cuda"):
        raise SystemExit(f"VRAM profile {profile} requires CUDA.")
    cuda_device = torch_module.device(device)
    total_bytes = int(torch_module.cuda.get_device_properties(cuda_device).total_memory)
    target_gb = 8 if profile == "8gb" else 24
    minimum_gb = 7.5 if profile == "8gb" else 22.0
    if total_bytes / (1024**3) < minimum_gb:
        raise SystemExit(f"VRAM profile {profile} requires at least {minimum_gb:g} GiB physical VRAM.")
    budget_bytes = min(total_bytes, target_gb * 1024**3)
    fraction = min(1.0, budget_bytes / total_bytes)
    if fraction < 0.999:
        torch_module.cuda.set_per_process_memory_fraction(fraction, cuda_device)
    return {
        "name": profile,
        "target_gb": target_gb,
        "physical_bytes": total_bytes,
        "budget_bytes": budget_bytes,
        "memory_fraction": round(fraction, 6),
        "simulated": fraction < 0.999,
    }


def _run_asr(
    audio_path: Path,
    expected_text: str,
    language: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.asr_backend == "off":
        return {"enabled": False, "error_rate": None}
    try:
        from runtime.speech_review import clear_asr_cache, review_transcript, transcribe_waveform

        asr_model = args.asr_model_by_language.get(language, args.asr_model)
        previous_model = getattr(args, "_active_asr_model", None)
        if previous_model is not None and previous_model != asr_model:
            clear_asr_cache()
        args._active_asr_model = asr_model
        waveform, sample_rate = load_audio_file(audio_path)
        transcript = transcribe_waveform(
            waveform,
            sample_rate,
            language=language,
            backend=args.asr_backend,
            model_name=asr_model,
            device=args.asr_device,
            download_root=args.asr_download_root or args.output_dir / "asr_models",
        )
        resolved_backend = transcript.get("backend", args.asr_backend)
        package = "openai-whisper" if resolved_backend == "openai_whisper" else "faster-whisper"
        try:
            package_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_version = None
        review = review_transcript(expected_text, transcript["text"], language, 0.0)
        error_rate = review.get("cer") if language in {"ZH", "JA"} else review.get("wer")
        return {
            "enabled": True,
            "backend": resolved_backend,
            "package_version": package_version,
            "model": asr_model,
            "recognized_text": transcript.get("text", ""),
            "metric": "CER" if language in {"ZH", "JA"} else "WER",
            "error_rate": error_rate,
            "similarity": review.get("similarity"),
            "differences": review.get("differences", []),
        }
    except Exception as exc:  # real-model diagnostic must preserve the WAV/report
        return {
            "enabled": True,
            "error_rate": None,
            "error": str(exc).strip() or type(exc).__name__,
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.asr_model_by_language = _parse_asr_model_overrides(args.asr_model_for)
    model_dir = args.model_dir.resolve()
    voice = args.voice.resolve()
    config = model_dir / "config.yaml"
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_dir}")
    if not config.is_file():
        raise SystemExit(f"Missing IndexTTS 2.5 config: {config}")
    if not voice.is_file():
        raise SystemExit(f"Reference voice does not exist: {voice}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torchaudio
    from indextts.infer_v2_5 import IndexTTS2

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    vram_profile = _configure_vram_profile(torch, args.vram_profile, device)
    use_bf16, use_fp16 = _resolve_precision(torch, args.precision, device)
    reference_device = args.reference_device or ("cpu" if args.vram_profile == "8gb" else None)
    tts = IndexTTS2(
        cfg_path=str(config),
        model_dir=str(model_dir),
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        device=device,
        reference_device=reference_device,
        use_qwen_emo=False,
        reuse_spk_cond_for_emo=True,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
        "vram_profile": vram_profile,
        "model_dir": str(model_dir),
        "reference_voice": str(voice),
        "seed": args.seed,
        "asr_runtime": {
            "backend": args.asr_backend,
            "model": args.asr_model,
            "model_by_language": args.asr_model_by_language,
        },
        "cases": [],
    }
    selected = {str(language).upper() for language in args.languages}
    for case_index, case in enumerate(QUALITY_CASES):
        if case["language"] not in selected:
            continue
        output_path = args.output_dir / f"{case_index + 1:02d}_{case['id']}.wav"
        collector: list[dict[str, Any]] = []
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
            torch.cuda.synchronize(torch.device(device))
        started = time.perf_counter()
        _consume_generation(
            tts.infer(
                spk_audio_prompt=str(voice),
                text=case["text"],
                output_path=str(output_path),
                lang=case["language"],
                do_sample=True,
                seed=args.seed + case_index,
                max_text_tokens_per_segment=args.max_text_tokens,
                interval_silence=args.segment_silence_ms,
                text_normalization=True,
                duration_factor=1.0,
                segment_collector=collector,
            )
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        elapsed = time.perf_counter() - started
        waveform, sample_rate = load_audio_file(output_path)
        audio = analyze_waveform(waveform, sample_rate)
        segment_reports = assess_segment_speech_rates(collector)
        duration = float(audio["duration_seconds"])
        case_report = {
            **case,
            "output": str(output_path),
            "elapsed_seconds": round(elapsed, 4),
            "rtf": round(elapsed / duration, 4) if duration > 0 else None,
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated(torch.device(device)))
                if device.startswith("cuda")
                else 0
            ),
            "audio": audio,
            "segment_rates": summarize_segment_rates(segment_reports),
            "segment_rate_report": segment_reports,
            "asr": _run_asr(output_path, case["text"], case["language"], args),
        }
        report["cases"].append(case_report)
        print(
            f"[{case['language']}] {output_path.name}: "
            f"{duration:.2f}s, RTF={case_report['rtf']}, "
            f"clip={audio['clipping_ratio']}, silence={audio['silence_ratio']}",
            flush=True,
        )

    report["absolute_gate"] = evaluate_quality_report(report)
    enabled_asr = next(
        (case.get("asr") for case in report["cases"] if case.get("asr", {}).get("enabled")),
        None,
    )
    if enabled_asr:
        report["asr_runtime"].update(
            {
                "backend": enabled_asr.get("backend", args.asr_backend),
                "package_version": enabled_asr.get("package_version"),
            }
        )
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        report["baseline"] = str(args.baseline.resolve())
        report["comparison"] = compare_quality_reports(report, baseline)
    report_path = args.output_dir / "quality-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + os.linesep,
        encoding="utf-8",
    )
    print(f"Quality report: {report_path.resolve()}")
    if args.write_baseline:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(
            json.dumps(build_baseline_snapshot(report), ensure_ascii=False, indent=2)
            + os.linesep,
            encoding="utf-8",
        )
        print(f"Portable baseline: {args.write_baseline.resolve()}")
    failed = report["absolute_gate"]["status"] == "failed" or (
        report.get("comparison", {}).get("status") == "failed"
    )
    return 2 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
