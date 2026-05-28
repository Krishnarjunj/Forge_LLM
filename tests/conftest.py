"""Shared pytest fixtures.

The autouse ``_seed_all`` fixture enforces CLAUDE.md sec 5 determinism for
every test: Python, NumPy, and PyTorch RNGs all reseeded to 0, with PyTorch
configured to use deterministic algorithms only. A test that genuinely needs
a non-deterministic kernel must :func:`pytest.skip` with a written reason
(CLAUDE.md sec 11), not silently relax the autouse setting.

Larger fixtures (``tiny_config``, ``tmp_ckpt``, ``cpu_device``, ``cuda_device``)
will land when M8 introduces ``ForgeConfig`` and the model assembly tests
need them.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_all() -> None:
    """Seed every RNG deterministically and lock PyTorch into deterministic ops.

    Applied to every test via ``autouse`` so a forgotten ``seed(0)`` call in
    one file can't leak nondeterminism into another.
    """
    # Env vars first: cuBLAS reads CUBLAS_WORKSPACE_CONFIG at init, so it must
    # be set before any CUDA op. Harmless on CPU.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
