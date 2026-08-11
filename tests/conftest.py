from __future__ import annotations

import os
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

comfy_root = os.environ.get("COMFYUI_ROOT")
if comfy_root and comfy_root not in sys.path:
    sys.path.insert(0, comfy_root)

