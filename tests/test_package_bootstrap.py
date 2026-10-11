from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


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


@pytest.mark.parametrize(
    "package_name",
    ["t8_plugin_collision_probe", "custom_nodes.t8_plugin_collision_probe"],
)
def test_plugin_registers_with_a_foreign_indextts_already_loaded(package_name):
    comfy_api = pytest.importorskip("comfy_api.latest")
    comfy_root = Path(comfy_api.__file__).resolve().parents[2]
    script = f"""
import asyncio
import importlib
import importlib.util
import pathlib
import sys
import types

root = pathlib.Path({str(ROOT)!r})
sys.path.insert(0, {str(comfy_root)!r})
foreign = types.ModuleType("indextts")
foreign.__file__ = str(root.parent / "other-index-tts" / "indextts" / "__init__.py")
foreign.__path__ = [str(pathlib.Path(foreign.__file__).parent)]
sys.modules["indextts"] = foreign
foreign_utils = types.ModuleType("indextts.utils")
foreign_utils.__path__ = [str(pathlib.Path(foreign.__file__).parent / "utils")]
foreign_common = types.ModuleType("indextts.utils.common")
foreign_common.tokenize_by_CJK_char = lambda text: text
foreign_common.de_tokenized_by_CJK_char = lambda text: text
sys.modules["indextts.utils"] = foreign_utils
sys.modules["indextts.utils.common"] = foreign_common
package_name = {package_name!r}
if "." in package_name:
    parent_name = package_name.rpartition(".")[0]
    parent = types.ModuleType(parent_name)
    parent.__path__ = [str(root.parent)]
    sys.modules[parent_name] = parent
spec = importlib.util.spec_from_file_location(
    package_name, root / "__init__.py", submodule_search_locations=[str(root)]
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[package_name] = plugin
spec.loader.exec_module(plugin)
extension = asyncio.run(plugin.comfy_entrypoint())
nodes = asyncio.run(extension.get_node_list())
assert len(nodes) >= 20
assert all(node.GET_SCHEMA().node_id for node in nodes)
adapter = sys.modules[package_name + ".runtime.inference_adapter"]
planner = sys.modules[package_name + ".runtime.text_planner"]
processing = sys.modules[package_name + ".runtime.audio_processing"]
acceleration = sys.modules[package_name + ".runtime.acceleration"]
assert adapter.assess_segment_speech_rates.__module__ == package_name + ".indextts.speech_rate_guard"
assert planner.language_aware_token_budget.__module__ == package_name + ".indextts.utils.text_segmentation"
assert processing.fade_out_pcm_tail.__module__ == package_name + ".indextts.utils.common"
assert acceleration.probe_torchcodec_runtime.__module__ == package_name + ".indextts.utils.audio_io"
tokenizer = importlib.import_module(package_name + ".indextts.utils.tokenizer")
tokenizer_marker = object()
tokenizer.get_tokenizer = lambda **kwargs: tokenizer_marker
assert planner._load_tokenizer(str(root)) is tokenizer_marker
downloader = importlib.import_module(package_name + ".services.downloader")
model_download = importlib.import_module(package_name + ".indextts.utils.model_download")
downloads = []
model_download.set_download_source = lambda source: downloads.append(source)
model_download.ensure_models_available = lambda target: downloads.append(target)
downloader.download_auxiliary_models(root, "huggingface")
assert downloads == ["huggingface", str(root)]
assert sys.modules["indextts"] is foreign
assert sys.modules["indextts.utils.common"] is foreign_common
cache = sys.modules[package_name + ".runtime.model_cache"]
cache.require_runtime_dependencies = lambda root: None
try:
    cache._load_core_class()
except RuntimeError as error:
    assert foreign.__file__ in str(error), str(error)
    assert "重启 ComfyUI" in str(error)
else:
    raise AssertionError("A conflicting inference core must not be silently replaced")
assert sys.modules["indextts"] is foreign
assert sys.modules["indextts.utils.common"] is foreign_common
print("foreign-core registration and diagnostics passed")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, completed.stderr


def test_registry_package_contract_excludes_unused_weights_and_keeps_licenses():
    ignored = {
        line.strip().replace("\\", "/")
        for line in (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert UNUSED_BUNDLED_WEIGHTS <= ignored
    assert (ROOT / "indextts" / "speech_rate_guard.py").is_file()
    assert "indextts/" not in ignored
    assert "indextts/speech_rate_guard.py" not in ignored

    expected_licenses = {
        ROOT / "incl_licenses" / f"LICENSE_{index}" for index in range(1, 9)
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected_licenses)
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "incl_licenses/LICENSE_1" in notices
    assert "incl_licenses/LICENSE_8" in notices
