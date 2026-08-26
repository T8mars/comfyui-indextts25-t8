from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_package_bootstrap_exposes_bundled_indextts_without_comfy_sys_path():
    root = Path(__file__).resolve().parents[1]
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
