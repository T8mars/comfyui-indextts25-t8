from __future__ import annotations

import platform
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from runtime.dependency_probe import missing_dependencies
from services.model_store import load_manifest, validate_model_dir


def main() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                print(f"  cuda:{index}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
    except Exception as exc:
        print(f"PyTorch: unavailable ({exc})")

    missing = missing_dependencies()
    print("Dependencies:", "OK" if not missing else "missing " + ", ".join(missing))
    manifest = load_manifest()
    print(f"Code revision: {manifest['codeRevision']}")
    print(f"Model revision: {manifest['modelRevision']}")
    if len(sys.argv) > 1:
        report = validate_model_dir(Path(sys.argv[1]), verify_hashes="--hash" in sys.argv[2:])
        print(f"Model directory: {report.model_dir}")
        print(f"Model files: {'OK' if report.valid else 'INCOMPLETE'}")
        if report.missing:
            print("Missing:", ", ".join(report.missing))
        if report.mismatched:
            print("Mismatched:", ", ".join(report.mismatched))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

