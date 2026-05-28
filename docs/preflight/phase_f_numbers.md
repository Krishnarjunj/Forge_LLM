# Phase F — Preflight measurements

> **Status:** Empty template. Populated during the Phase F preflight session on a Kaggle T4 (per `docs/03_training_plan.md` §10 the canonical entrypoint is `notebooks/02_kaggle_train.ipynb`). Each gate below records its measurement once and only once; values become the resolution criteria for ADR-004, ADR-005, ADR-006, ADR-008, ADR-009, ADR-010 in `docs/DECISIONS.md`. ADR closures are committed separately per `resume_here.md` §5 step 3 (max-2-word commit messages, no push).

---

## 0. Run metadata

| Field | Value |
|-------|-------|
| Date (UTC) | `<YYYY-MM-DD>` |
| Platform | Kaggle / Colab T4 |
| GPU name (`nvidia-smi`) | `<Tesla T4>` |
| Driver / CUDA | `<driver> / <cuda>` |
| `torch.__version__` | `<x.y.z+cuXXX>` |
| Notebook commit (`git_sha`) | `<sha>` |
| Wandb run ID | `<run_id>` |
| Wandb run URL | `<https://wandb.ai/...>` |
| Model preset | `forge-30m` (n_layer=6 baseline) |
| Tokenizer | `configs/tokenizer.json` SHA: `<sha>` |
| Tokens/optimizer step (target) | 131,072 |

---

## 1. Headline numbers (fill these in; everything else references them)

```
tokens/sec (mean, steps 50..100):           <measured>
peak GPU memory (GB):                       <measured>
gpu_util (mean last 50 steps, %):           <measured>
generate compile speedup ctx=128:           <measured>×
generate compile speedup ctx=512:           <measured>×
generate compile speedup ctx=2048:          <measured>×
generate cold-start with compile (s):       <measured>
```

These four bullets close every open ADR. The rest of this doc is the show-your-work scaffolding.

---

## 2. Gate results (`BUILD_PLAN.md` §8 order)

### Gate 1 — Overfit-one-batch

Tile a single batch on `forge-30m` (or a tiny `forge-1m`-style debug config), train ~200 steps, expect loss to collapse toward 0. Catches a broken loss / data path / optim setup.

| Field | Value |
|-------|-------|
| Config used | `forge-30m` / debug |
| Batch shape `(B, T)` | `<8, 1024>` |
| Steps run | 200 |
| Initial loss | `<~10.x>` |
| Loss @ step 50 | `<>` |
| Loss @ step 200 | `<should be ≪ 1>` |
| Pass / fail | **`<pass / FAIL>`** |
| Notes | `<anomalies, NaN skips, etc.>` |

If FAIL: stop here, do not proceed. Loss not collapsing on a tiled batch indicates a bug in loss, data, or optimizer — none of which Phase F is meant to debug. Open a new ADR with the failure mode.

### Gate 2 — 100-step smoke run on FineWeb-Edu stream

Real data path, real tokenizer, real model. `gpu_util` and tokens/sec measured over **steps 50..100** (skip the first 50 as warmup).

| Field | Value |
|-------|-------|
| `(micro_bs, grad_accum, seq_len)` | `(8, 16, 1024)` baseline |
| Tokens per optimizer step | 131,072 |
| Steps run | 100 |
| Tokens/sec (mean steps 50..100) | `<measured>` |
| Mean gpu_util steps 50..100 (%) | `<measured>` |
| Peak GPU memory (GB) | `<measured>` (`torch.cuda.max_memory_allocated() / 1024**3`) |
| NaN skips | `<n>` (must be 0 for a clean smoke) |
| Wall-clock per optimizer step (s) | `<measured>` |

### Gate 3 — Tokens/sec measurement

This is the same run as Gate 2, but recorded separately because it's the input to ADR-009.

| Measurement window | Tokens/sec |
|---------------------|-----------|
| Steps 50..100 (steady-state) | `<measured>` |
| Steps 0..50 (warmup, for reference) | `<measured>` |

Sanity reference (per `docs/03_training_plan.md` §8): plan budgets 6,000 tokens/sec. Naive A100→T4 scaling predicts ~750; the 8× gap is closed by GQA + memory-bandwidth-bound regime at 30M params.

### Gate 4 — Memory headroom on T4

T4 = 16 GB. Required headroom: ≥10% of total → peak ≤ 14.4 GB.

| `(micro_bs, grad_accum)` | Peak mem (GB) | Headroom (%) | Outcome |
|--------------------------|---------------|--------------|---------|
| (8, 16) baseline | `<>` | `<>` | `<keep / step down>` |
| (4, 32) | `<only if (8,16) fails>` | `<>` | `<>` |
| (2, 64) | `<only if (4,32) fails>` | `<>` | `<>` |
| (1, 128) | `<last-resort>` | `<>` | `<>` |
| activation-ckpt | `<if (1,128) still OOMs>` | `<>` | `<>` |

Walk the ladder from `docs/03_training_plan.md` §2.2 until the first row with ≥10% headroom. That row becomes ADR-008.

### Gate 5 — Checkpoint resume across a hard process kill

Already passes the bit-exact contract on single-GPU fp32 CPU (`tests/test_resume.py::test_resume_safety_loss_curve_indistinguishable`, slow). Gate 5 validates it in the real Kaggle environment.

| Step | Result |
|------|--------|
| Train 50 steps on T4 | `<loss curve>` |
| `os.kill(os.getpid(), SIGKILL)` (or stop kernel) | `<killed>` |
| Restart kernel, `Trainer.load_checkpoint(...)` | `<loaded ok / FAIL>` |
| Continue 50 steps | `<loss curve>` |
| Loss curves on steps 51..100 match the uninterrupted reference (tolerance per `CLAUDE.md` §7 single-GPU fp32: bitwise; T4 fp16: `atol≤1e-4` is acceptable for the smoke) | **`<pass / FAIL>`** |
| Wandb run ID rejoined (same run, appended metrics) | `<pass / FAIL>` |

### Gate 6 — Wandb config dump

Verify the wandb dashboard shows these fields at run start (the Trainer already records them in the checkpoint; this gate confirms they reach the dashboard):

- [ ] `git_sha`
- [ ] Resolved model config (full `ForgeConfig` JSON)
- [ ] `torch.__version__`, `torch.version.cuda`
- [ ] GPU name + driver
- [ ] Full `pip freeze`
- [ ] `config_hash` (SHA256 of resolved config dict)
- [ ] OS / Python version

---

## 3. ADR closures — measured criteria

For each ADR, the resolution criterion and the measurement that closes it. After filling in, transcribe the **Decision** and **Consequences** sections of the ADR in `docs/DECISIONS.md` and flip its Status to **Accepted**. Each closure is its own commit (e.g. `adr004 closed`, `adr005 closed`).

### ADR-004 — On-the-fly vs pre-packed tokenization

| Criterion | Measured |
|-----------|----------|
| Mean `gpu_util` over steps 50..100 | `<measured>` |
| Threshold | ≥70% → on-the-fly stays; <70% → pre-pack |
| **Decision** | `<on-the-fly / pre-pack via scripts/prepare_data.py>` |

If pre-pack: file an action item to write `scripts/prepare_data.py` and add a `--data-path` flag (per `docs/DECISIONS.md` ADR-004 Consequences).

### ADR-005 — Final `n_layer`: 6 vs 8

| Criterion | Measured |
|-----------|----------|
| Peak GPU mem at `n_layer=6, micro_bs=8` | `<measured>` GB |
| Headroom at n_layer=6 (%) | `<measured>` |
| Peak GPU mem at `n_layer=8, micro_bs=8` (only if headroom-at-6 > 30%) | `<measured>` GB |
| Headroom at n_layer=8 (%) | `<measured>` |
| Threshold | n_layer=6 headroom >30% (peak <11 GB) → bump to 8; n_layer=8 headroom <10% → revert to 6 (rename preset to `forge-25m`) |
| **Decision** | `<n_layer=6 / n_layer=8>` |
| Resulting param count | `<25.3M / 31.2M>` |

If preset changes: update `configs/model_30m.yaml`, `src/forge_llm/config.py::PRESETS["forge-30m"]` (or rename to `forge-25m`), and `tests/test_model.py::test_model_param_count` expected value with a comment citing this ADR closure.

### ADR-006 — `torch.compile` default in `generate()`

| ctx | tokens/sec no-compile | tokens/sec --compile | speedup | notes |
|-----|-----------------------|----------------------|---------|-------|
| 128 | `<>` | `<>` | `<>×` | |
| 512 | `<>` | `<>` | `<>×` | |
| 2048 | `<>` | `<>` | `<>×` | |

Cold-start with compile (compile time included): `<measured>` s.

| Criterion | Measured |
|-----------|----------|
| Speedup at all three ctxs ≥1.5× | `<yes / no>` |
| `forge-llm generate` cold start <60 s including compile | `<yes / no>` |
| **Decision** | `<default-on (with --no-compile) / opt-in (--compile)>` |

If default-on: wire `--no-compile` opt-out in `src/forge_llm/cli.py` and `src/forge_llm/generate.py`; if opt-in: ensure the existing `--compile` flag stays wired and document the recommended use case in README.

### ADR-008 — micro_bs / grad_accum split

Take from Gate 4 ladder above — the first row with ≥10% headroom.

| Field | Value |
|-------|-------|
| Chosen `(micro_bs, grad_accum)` | `<>` |
| Peak GPU mem at chosen pair (GB) | `<>` |
| Headroom (%) | `<>` |
| Tokens per optimizer step | must equal 131,072 (verify) |
| Activation checkpointing? | `<no / yes>` |

### ADR-009 — Token budget

| Criterion | Measured |
|-----------|----------|
| Steady-state tokens/sec (steps 50..100, T4) | `<>` |
| Decision threshold | ≥4,500 → keep 1B; 2,500–4,500 → 500M; <2,500 → 250M |
| **Decision** | `<1B / 500M / 250M>` total training tokens |
| Updated total optimizer steps | `<budget / 131,072>` |
| Updated README per-param efficiency claim | `<unchanged / scaled down>` |

If <1B: update `docs/03_training_plan.md` §8 budget block in the same commit as the ADR closure.

### ADR-010 — Number of Kaggle sessions allocated

Function of ADR-008 + ADR-009. Wall-clock = `total_tokens / measured_tokens_per_sec`. Sessions = `ceil(wall_clock / 11h)` (the 11h soft cap per `docs/03_training_plan.md` §10, leaving a 1h cushion under the 12h Kaggle hard cap).

| Field | Value |
|-------|-------|
| Total tokens (from ADR-009) | `<>` |
| Tokens/sec (steady-state) | `<>` |
| Wall-clock (hours) | `<total / tokens_per_sec / 3600>` |
| Sessions @ 11h soft cap | `<ceil(wall_clock / 11)>` |
| **Decision — N sessions** | `<3 / 4 / 5>` |
| Calendar implication | `<weeks needed under 30h/wk quota>` |

---

## 4. Anomalies / followups

> Anything that surfaced during preflight that doesn't fit an existing ADR. Examples: unexpected dataloader stalls, NaN-skip rate higher than expected, wandb logging gaps, Kaggle Dataset upload failures, etc.

- `<none yet>`

---

## 5. Sign-off

- [ ] Gates 1–6 all pass
- [ ] ADR-004 closed
- [ ] ADR-005 closed
- [ ] ADR-006 closed
- [ ] ADR-008 closed
- [ ] ADR-009 closed
- [ ] ADR-010 closed
- [ ] If any preset / budget changed: `configs/`, `src/forge_llm/config.py`, `tests/test_model.py`, and `docs/03_training_plan.md` updated in the same commit chain
- [ ] User confirms Phase G can begin

Then **stop** — wait for the user before kicking off Phase G (the actual training run). No push (per `CLAUDE.md` §0).
