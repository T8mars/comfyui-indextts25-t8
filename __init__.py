"""ComfyUI IndexTTS 2.5 nodes by Bilibili creator T8star-Aix."""

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

__version__ = "0.1.0"
__all__ = ["comfy_entrypoint", "__version__"]
