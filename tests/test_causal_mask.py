"""Adversarial causal-mask-leak test — see docs/02_correctness_plan.md §1.7.

Mutate the value of token T-1, assert that outputs at all positions t < T-1
are byte-identical to the unmutated forward. Implemented when the M4 MHA
forward exists.
"""

import pytest


def test_placeholder() -> None:
    pytest.skip("not yet implemented — lands in M4 (causal mask adversarial)")
