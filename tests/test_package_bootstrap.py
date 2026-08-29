from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNUSED_BUNDLED_WEIGHTS = {
    "indextts/utils/maskgct/models/codec/facodec/modules/JDC/bst.t7",
    "indextts/utils/maskgct/models/tts/maskgct/ckpt/wav2vec2bert_stats.pt",
}


def test_package_bootstrap_exposes_bundled_indextts_without_comfy_sys_path():
    root = ROOT
    script = f"""
import importlib.util
import os
import pathlib
import re
import sys
import tempfile
import types

root = pathlib.Path({str(root)!r})
os.chdir(tempfile.gettempdir())
sys.path = [
    item for item in sys.path
    if item and pathlib.Path(item).resolve() not in {{root.resolve(), root.parent.resolve()}}
]
spec = importlib.util.spec_from_file_location(
    "t8_plugin_probe",
    root / "__init__.py",
    submodule_search_locations=[str(root)],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
fake_nodes = types.ModuleType("t8_plugin_probe.nodes_v3")
fake_nodes.comfy_entrypoint = object()
sys.modules[fake_nodes.__name__] = fake_nodes
spec.loader.exec_module(module)
import indextts
assert pathlib.Path(indextts.__file__).resolve() == (root / "indextts" / "__init__.py").resolve()
assert sys.path[0] == str(root)
declared = re.search(r'^version\\s*=\\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.MULTILINE)
assert declared is not None and module.__version__ == declared.group(1)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_registry_package_contract_excludes_unused_weights_and_keeps_licenses():
    ignored = {
        line.strip().replace("\\", "/")
        for line in (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert UNUSED_BUNDLED_WEIGHTS <= ignored

    expected_licenses = {
        ROOT / "incl_licenses" / f"LICENSE_{index}" for index in range(1, 9)
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected_licenses)
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "incl_licenses/LICENSE_1" in notices
    assert "incl_licenses/LICENSE_8" in notices
