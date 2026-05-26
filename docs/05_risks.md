# 05 — Top 10 Risks

> Ranked by expected harm × likelihood. Each risk has an owner phase, a leading indicator, and a concrete mitigation. "Mitigation" must be an action we will actually take — not "be careful."

---

## R1 — Subtle math bugs that pass the loss-decrease check

**Likelihood:** High (the cohort's #1 silent failure mode).
**Harm:** Catastrophic — invalidates the entire training run; PPL claims become false.

**Examples:**
- Attention scaling missing or off (`1/sqrt(head_dim)`).
- Softmax dim wrong (over heads instead of over keys).
- RoPE rotation in the wrong direction or wrong convention (interleaved vs paired).
- SwiGLU gate and up projections swapped.
- KV-cache returning a slice that includes uninitialised zeros for unfilled positions.

These bugs let loss decrease — just not as fast as it should. They don't crash. They don't NaN. They produce a worse model.

**Mitigation.**
- Every layer has a value-vs-HF-Llama oracle test (`docs/02_correctness_plan.md`). If the per-element diff exceeds tolerance, CI fails before the bug enters main.
- The full-model value-vs-Llama test at M8 catches accumulated drift across layers that survives individual layer tests.
- The causal-mask-leak adversarial test catches the most common silent mask bug.
- A 200-step "overfit one batch" preflight (Phase F) is the canary — a math-correct ~30M model should reach near-zero loss; a math-broken one will plateau.

**Owner phase:** Phase E (caught by tests) + Phase F (caught by overfit canary).

---

## R2 — Training instability on fp16 (NaN, gradient explosion)

**Likelihood:** Medium-high — fp16 + T4 (no bf16) is the most fragile config; cohort precedent (CS23S011 used Muon for stability) hints at why.
**Harm:** Wasted Kaggle session quota; potentially requires LR drop and rerun.

**Mitigation.**
- `GradScaler` with initial scale `2**16`, growth_interval 2000.
- fp32 softmax in attention (per `CLAUDE.md` §6).
- fp32 RMSNorm compute (per `CLAUDE.md` §6).
- NaN/Inf detection — `train.py` logs `train/nan_skip_count`. Auto-halt if >5% of steps in a 1000-step window NaN-skip. Restart with LR × 0.5.
- Gradient clipping at norm 1.0 (`torch.nn.utils.clip_grad_norm_`).
- LR warmup is 1000 steps (conservative — 13% of total) so the first big steps don't blow up.

**Owner phase:** Phase F (preflight catches it on 100-step smoke run) + Phase G (in-flight monitor).

---

## R3 — T4 OOM at chosen micro_batch

**Likelihood:** Medium — our estimate is comfortable (~1.8 GB used of 16 GB) but estimates underweight allocator fragmentation and cuDNN workspace.
**Harm:** Phase F blocks; we drop micro_bs and lose tokens/sec.

**Mitigation.**
- `docs/03_training_plan.md` §2.2 has an explicit fallback ladder: micro_bs 8 → 4 → 2 → 1 → activation checkpointing.
- Phase F memory test uses `torch.cuda.max_memory_allocated()` and `torch.cuda.reset_peak_memory_stats()` at the start of every 10 steps and the peak is logged to wandb.
- We require ≥10% headroom (~1.6 GB free) to commit to a given micro_bs.

**Owner phase:** Phase F.

---

## R4 — Kaggle session kill mid-training

**Likelihood:** High — sessions die for many reasons (12h cap, idle timeout, kernel restart on commit, network blip during checkpoint upload).
**Harm:** Lose 30 min of progress if checkpointing is poorly engineered; complete restart if resume is broken.

**Mitigation.**
- Checkpoint every 500 steps OR 30 minutes, whichever first.
- Atomic upload (write `step_N.tmp`, rename to `step_N.pt`).
- HF Hub + Kaggle Dataset double-write — if Hub upload fails, the Kaggle Dataset mirror is the backup.
- The session-budget watchdog in `train.py` triggers a final checkpoint at 11h elapsed (Kaggle's 12h cap minus 1h cushion).
- The resume-safety test (`test_resume_safety_loss_curve_indistinguishable`) is **mandatory before any real training session begins** — a green resume test on CPU fp32 catches RNG/iterator/optimizer-state losses early.
- `git_sha` and `config_hash` in the checkpoint prevent silently resuming with mismatched code/config.

**Owner phase:** Phase E (M9 — resume implementation + tests) + Phase G (in-flight).

---

## R5 — FineWeb-Edu access / licensing / availability

**Likelihood:** Low-medium — HF datasets occasionally rate-limit, FineWeb-Edu is permissively licensed but a Kaggle notebook needs the HF token configured.
**Harm:** Can't stream data; training blocked entirely.

**Mitigation.**
- FineWeb-Edu is CC-BY-4.0 (no restrictive license issues).
- HF token configured via Kaggle Secret (`HF_TOKEN`); notebook prints a clear setup banner if missing.
- Fallback dataset: pre-tokenized OpenWebText shard from a Kaggle Dataset we prepare locally and upload. If FineWeb-Edu is unreachable on Kaggle for any reason, we point `--dataset` at the fallback.
- The notebook caches the first 1M sequences locally so a transient network blip after start doesn't kill the session.

**Owner phase:** Phase F (verifies FineWeb-Edu loads on a fresh Kaggle fork) + Phase G.

---

## R6 — Reproducibility breaks on a clean machine

**Likelihood:** Medium-high — the #1 thing that turns a "PEAK" project into "not actually reproducible" is undocumented env dependencies.
**Harm:** Reviewers / interviewers `pip install forge-llm` and it crashes; the entire deployable-artifact differentiator collapses.

**Mitigation.**
- `pyproject.toml` pins major.minor of every runtime dep with `>=X.Y,<X.Y+10` ranges so a future incompatible release doesn't break installs.
- CI matrix on Python 3.11 and 3.12.
- A `tests/test_install.py` (Phase H) verifies `pip install forge-llm` in a clean venv, imports succeed, `forge-llm generate` runs.
- The Kaggle notebook starts with `!pip install forge-llm` (after publish) and a smoke cell that runs `forge-llm generate "test"` end-to-end before any training begins.
- The architecture diagram in the README and the blog explain assumptions (Python 3.11+, PyTorch ≥2.2, free Kaggle T4).

**Owner phase:** Phase H (release).

---

## R7 — RoPE convention drift (Llama interleaved vs original paper)

**Likelihood:** High — every from-scratch RoPE implementation accidentally swaps the convention at least once.
**Harm:** Generated output is gibberish; perplexity ~doubles silently; bug looks like "model is just bad" instead of "rotation is wrong".

**Mitigation.**
- ADR-007 (committed in M3) explicitly locks the Llama interleaved convention.
- `test_rope_value_vs_llama` compares against the HF Llama oracle to rtol=1e-5.
- `test_rope_rotation_identity` (rotation at position 0 = identity) catches sign-flip bugs.
- `test_rope_relative_position_invariance` catches absolute-vs-relative confusion.
- The full-model test at M8 catches accumulated drift if individual RoPE tests are too forgiving.

**Owner phase:** Phase E M3.

---

## R8 — KV-cache correctness drift between training and inference paths

**Likelihood:** Medium — easy to write the cached path slightly differently from the uncached path (e.g., different RoPE position arithmetic, different mask).
**Harm:** Generation quality silently degrades; reported "KV-cache equivalence" claim is false.

**Mitigation.**
- The attention module's `forward()` is the *same code path* for cached and uncached calls; the difference is whether `cache` and `input_pos` are passed.
- `test_kvcache_full_vs_token_by_token_equivalence` (`docs/02_correctness_plan.md` §1.8) compares full-sequence forward against token-by-token cached forward at rtol=1e-5.
- The benchmark script (`scripts/benchmark_kv_cache.py`) runs the equivalence check as a sanity gate before reporting tokens/sec numbers.
- A regression test stays in CI even after M11 lands.

**Owner phase:** Phase E M11.

---

## R9 — Tokenizer mismatch between training and inference

**Likelihood:** Medium — easy to retrain the tokenizer mid-project and forget to re-tokenize, or to publish a checkpoint with the wrong tokenizer.
**Harm:** Inference produces tokens the model has never seen; PPL evaluation reports garbage.

**Mitigation.**
- The tokenizer JSON is hashed and the hash is stored in `ForgeConfig` (`tokenizer_hash`). Checkpoint loading verifies the hash matches the tokenizer on disk; mismatch refuses to load unless `--force`.
- `save_pretrained` writes the tokenizer alongside the checkpoint (`tokenizer.json` in the same directory).
- `from_pretrained` loads both in lock-step.
- A Phase H integration test (`test_pretrained_roundtrip`) saves a model, loads it from disk in a new process, encodes a fixed string with the loaded tokenizer, runs through the loaded model, and asserts identical logits to pre-save.

**Owner phase:** Phase E M9 (hub.py) + Phase H.

---

## R10 — Time slip past Oct 25, 2026 deadline

**Likelihood:** Medium-high — 4 weeks is tight given the from-scratch scope, and Phase E's L milestone (M9) typically over-runs.
**Harm:** Misses resume v3 window; project becomes interview-only artifact rather than recruited-on-paper signal.

**Mitigation.**
- Aggressive milestone scoping in `docs/04_roadmap.md` — M9 is explicitly flagged Large and budgeted 2–3 review iterations.
- Stretch items (FlashAttention-2 Triton kernel, speculative decoding, fine-tuning experiments) are explicitly out of scope unless ahead of schedule.
- The "scale-up rerun on A100" item from the brief is **deferrable indefinitely** without blocking the resume bullet — the resume bullet relies on T4 numbers.
- Calendar discipline: if M9 isn't green by week 2 day 5, we cut the per-parameter efficiency chart (move #3) before cutting any test or any resume-safety guarantee. Architecture and reproducibility take priority over comparative analysis.
- Final fallback: ship M1–M8 + M10–M12 + Phase F preflight without a full training run; the resume bullet becomes "implemented end-to-end and verified at preflight scale" — still PEAK-tier, just minus the training brag.

**Owner phase:** Phase E (in-flight schedule monitoring) + week-by-week user check-ins.

---

## Secondary risks (tracked, not in top 10)

- **Wandb account quota exceeded** — solved by free tier limits + CSV fallback.
- **HF Hub upload bandwidth on Kaggle** — Kaggle has good outbound; if it slows, Kaggle Dataset mirror is the primary.
- **Determinism breaks across CUDA versions** — pinned in `pyproject.toml`; if the Kaggle PyTorch version drifts, we pin a Kaggle Notebook setup cell.
- **Blog post writer's block** — Phase H scope is loose; the brief allows up to a 3000-word target but a 2000-word post is still PEAK-tier.

---

## Risk → mitigation matrix (one-page summary)

| # | Risk | Phase | Indicator | Mitigation |
|---|------|-------|-----------|------------|
| R1 | Silent math bug | E, F | Loss > expected after overfit | Oracle tests + overfit canary |
| R2 | fp16 instability | F, G | NaN-skip rate climbs | GradScaler + fp32 softmax + LR drop |
| R3 | T4 OOM | F | `max_memory_allocated` > 14.4 GB | Micro_bs fallback ladder |
| R4 | Session kill | E (M9), G | Wandb run gap | Checkpoint every 500 steps + watchdog at 11h |
| R5 | Data unavailable | F | Stream fails on fresh fork | HF_TOKEN secret + fallback dataset |
| R6 | Repro broken | H | `pip install` smoke test fails | Pinned deps + matrix CI + clean-venv test |
| R7 | RoPE convention drift | E (M3) | `test_rope_value_vs_llama` fails | ADR-007 locks convention; rtol=1e-5 |
| R8 | KV-cache drift | E (M11) | Equivalence test fails | Same code path + equivalence test in CI |
| R9 | Tokenizer mismatch | E (M9), H | `tokenizer_hash` mismatch | Hash-stamped config + lockstep load |
| R10 | Time slip past Oct 25 | E | M9 not green by week 2 | Scope cuts (move #3 first) |
