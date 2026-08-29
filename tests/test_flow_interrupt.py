from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from indextts.s2mel.modules.flow_matching import BASECFM


class _ZeroEstimator(nn.Module):
    def forward(self, x, prompt, x_lens, t, style, mu):
        return torch.zeros_like(x)


def test_cfm_checks_interruption_between_diffusion_steps():
    args = SimpleNamespace(
        DiT=SimpleNamespace(in_channels=2, zero_prompt_speech_token=False),
        reg_loss_type="l2",
    )
    cfm = BASECFM(args)
    cfm.estimator = _ZeroEstimator()
    checks = 0

    def interrupt():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("processing interrupted")

    with pytest.raises(RuntimeError, match="processing interrupted"):
        cfm.solve_euler(
            torch.zeros(1, 2, 4),
            torch.tensor([4]),
            torch.zeros(1, 2, 1),
            torch.zeros(1, 4, 3),
            torch.zeros(1, 3),
            None,
            torch.linspace(0, 1, 5),
            inference_cfg_rate=0.0,
            interrupt_callback=interrupt,
        )

    assert checks == 2
