"""ComfyUI IndexTTS 2.5 nodes by Bilibili creator T8star-Aix."""

try:
    if __package__:
        from .nodes_v3 import comfy_entrypoint
    else:  # Allows direct-file import by test runners despite the distribution folder's hyphen.
        import importlib.util
        import sys
        import types
        from pathlib import Path

        _root = Path(__file__).resolve().parent
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

__version__ = "0.5.1"
__all__ = ["comfy_entrypoint", "__version__"]
