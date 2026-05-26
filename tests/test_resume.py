"""Resume-safety test (slow) — see docs/02_correctness_plan.md §1.15.

Train 200 steps uninterrupted; train 100 + ckpt + kill + resume to 200;
assert steps 101..200 of both runs match bitwise on CPU fp32.
"""

import pytest


@pytest.mark.slow
def test_placeholder() -> None:
    pytest.skip("not yet implemented — lands in M9 (resume safety)")
