"""Tests for sampling primitives (M11).

Spec: ``docs/02_correctness_plan.md`` sec 1.12. Four cases on the pure
functions exported by ``forge_llm.sampling``. All run on fixed input vectors
so the assertions are exact.

* ``test_top_k_keeps_only_k_logits`` -- after ``top_k(logits, k=3)`` exactly 3
  logits are finite and the rest are ``-inf``; catches off-by-one on k.
* ``test_top_p_distribution_sanity`` -- a tight-spike distribution at p=0.5
  keeps exactly 1 token; a flat distribution at p=0.5 keeps the smallest
  set whose cumulative probability >= 0.5. Catches cumulative-sum bugs.
* ``test_temperature_extremes`` -- T=0 emits a one-hot at the argmax (no
  divide-by-zero crash); T very large makes the softmax KL-close to uniform.
* ``test_repetition_penalty_known_input`` -- the CTRL-paper formula: divide
  positive logits by penalty and multiply negative ones by penalty, for the
  specific token IDs in history.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from forge_llm.sampling import (
    apply_temperature,
    repetition_penalty,
    top_k,
    top_p,
)


def test_top_k_keeps_only_k_logits() -> None:
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    out = top_k(logits, k=3)
    finite_mask = torch.isfinite(out)
    assert int(finite_mask.sum().item()) == 3
    # The 3 finite values should be the top 3 (3, 4, 5).
    assert torch.equal(out[finite_mask].sort().values, torch.tensor([3.0, 4.0, 5.0]))
    # The masked positions are -inf.
    assert torch.all(out[~finite_mask] == float("-inf"))


def test_top_p_distribution_sanity() -> None:
    # Tight spike: one logit much larger -> probability ~1.0 on one token.
    spike = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])
    spike_out = top_p(spike, p=0.5)
    assert int(torch.isfinite(spike_out).sum().item()) == 1, (
        "tight-spike + p=0.5 should keep exactly one token"
    )

    # Flat distribution: uniform 0.2 per token; cumulative <0.5 after 2,
    # crosses 0.5 at the 3rd -> keep the smallest set that reaches p,
    # i.e. 3 tokens (0.2 + 0.2 + 0.2 = 0.6 >= 0.5).
    flat = torch.zeros(5)
    flat_out = top_p(flat, p=0.5)
    assert int(torch.isfinite(flat_out).sum().item()) == 3, (
        "flat dist + p=0.5 should keep the smallest cumulative-prob >=0.5 set"
    )


def test_temperature_extremes() -> None:
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # T = 0 -> one-hot at argmax. No NaN, no -inf at the argmax.
    out_zero = apply_temperature(logits, temperature=0.0)
    probs_zero = F.softmax(out_zero, dim=-1)
    assert torch.isfinite(out_zero).any(), "T=0 must not produce all -inf"
    assert int(probs_zero.argmax().item()) == int(logits.argmax().item())
    torch.testing.assert_close(probs_zero.max(), torch.tensor(1.0), rtol=0, atol=1e-6)

    # Very large T -> softmax ~uniform. Compare to uniform via KL.
    out_hot = apply_temperature(logits, temperature=1e6)
    probs_hot = F.softmax(out_hot, dim=-1)
    uniform = torch.full_like(probs_hot, 1.0 / probs_hot.numel())
    kl = (probs_hot * (probs_hot.log() - uniform.log())).sum()
    assert float(kl.item()) < 1e-2, f"T->inf should approach uniform; KL={kl.item()}"


def test_repetition_penalty_known_input() -> None:
    """CTRL paper: ``logit /= penalty`` if logit > 0, else ``logit *= penalty``."""
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0, 0.0])
    history = torch.tensor([0, 1])  # only penalise positions 0 and 1
    penalty = 1.2

    out = repetition_penalty(logits.clone(), history, penalty=penalty)

    # Position 0: 2.0 (positive) / 1.2.
    torch.testing.assert_close(out[0], torch.tensor(2.0 / 1.2), rtol=0, atol=1e-6)
    # Position 1: -1.0 (negative) * 1.2.
    torch.testing.assert_close(out[1], torch.tensor(-1.0 * 1.2), rtol=0, atol=1e-6)
    # Positions 2..4 must be unchanged.
    assert math.isclose(float(out[2].item()), 0.5)
    assert math.isclose(float(out[3].item()), -3.0)
    assert math.isclose(float(out[4].item()), 0.0)
