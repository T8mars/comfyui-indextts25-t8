from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def scoped_seed(seed: int, device: str = "cpu"):
    """Seed one inference without contaminating ComfyUI's global RNG state."""

    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices: list[int] = []
    if device.startswith("cuda") and torch.cuda.is_available():
        cuda_devices = [int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()]

    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

