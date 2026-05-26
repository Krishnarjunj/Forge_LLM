"""Shared pytest fixtures.

Real fixtures (seed_all, tiny_config, tmp_ckpt, cpu_device, cuda_device) land
in M2/M8 when the matching `src/` modules exist. Until then this file only
holds the autouse seeding hook so future tests are deterministic by default.
"""

from __future__ import annotations

import os
import random

import pytest


@pytest.fixture(autouse=True)
def _seed_all() -> None:
    """Seed Python's RNG deterministically for every test.

    Numpy/torch seeding is added in conftest when those deps are present
    (M2 onwards). Keeping the fixture autouse from day one avoids surprises
    when a non-deterministic test sneaks in.
    """
    random.seed(0)
    os.environ.setdefault("PYTHONHASHSEED", "0")
