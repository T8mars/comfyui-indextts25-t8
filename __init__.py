"""ComfyUI IndexTTS 2.5 nodes by Bilibili creator T8star-Aix."""

import sys
from pathlib import Path


# ComfyUI 2026.08 no longer guarantees that every custom-node directory is
# present on ``sys.path`` while importing an extension package.  The bundled
# upstream core still uses its official absolute ``indextts.*`` imports, so the
# repository root must be discoverable before importing any runtime module.
_PLUGIN_ROOT = Path(__file__).resolve().parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

try:
    if __package__:
        from .nodes_v3 import comfy_entrypoint
    else:  # Allows direct-file import by test runners despite the distribution folder's hyphen.
        import importlib.util
        import types

        _root = _PLUGIN_ROOT
        _package_name = "_comfyui_indextts25_t8_bootstrap"
        _package = sys.modules.get(_package_name)
        if _package is None:
            _package = types.ModuleType(_package_name)
            _package.__path__ = [str(_root)]
            _package.__package__ = _package_name
            sys.modules[_package_name] = _package
        _nodes_name = f"{_package_name}.nodes_v3"
        _nodes = sys.modules.get(_nodes_name)
        if _nodes is None:
            _spec = importlib.util.spec_from_file_location(_nodes_name, _root / "nodes_v3.py")
            _nodes = importlib.util.module_from_spec(_spec)
            sys.modules[_nodes_name] = _nodes
            _spec.loader.exec_module(_nodes)
        comfy_entrypoint = _nodes.comfy_entrypoint
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("comfy_api"):
        raise
    _COMFY_IMPORT_ERROR = exc

    async def comfy_entrypoint():
        raise RuntimeError("comfy_api.latest is required; install this directory inside a current ComfyUI build.") from _COMFY_IMPORT_ERROR

__version__ = "0.11.0"
__all__ = ["comfy_entrypoint", "__version__"]
