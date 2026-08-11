from __future__ import annotations

import random

import numpy as np
import torch

from runtime.seed_scope import scoped_seed


def _numpy_state_equal(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def test_scoped_seed_is_repeatable_and_restores_rng_state():
    random.seed(987)
    np.random.seed(987)
    torch.manual_seed(987)
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()

    with scoped_seed(42):
        first = (random.random(), float(np.random.rand()), torch.rand(4))
    with scoped_seed(42):
        second = (random.random(), float(np.random.rand()), torch.rand(4))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert random.getstate() == before_python
    assert _numpy_state_equal(np.random.get_state(), before_numpy)
    assert torch.equal(torch.random.get_rng_state(), before_torch)

