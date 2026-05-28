# DECISIONS — ADR log

> ADR-lite. Each entry is a tight 3-section record: **Decision**, **Context**, **Consequences**. Append-only — never delete; mark superseded with a status line.
>
> One-line justification rule (per `BUILD_PLAN.md` §2 Hard rule 6): every new entry to `pyproject.toml` must have at least a one-line entry here.

---

## ADR-001 — Config system: `@dataclass` (CLOSED)

**Status:** Accepted — 2026-05-26 (Phase C).

**Decision.** A single `@dataclass(frozen=True)` `ForgeConfig` in `src/forge_llm/config.py`, with a `PRESETS: dict[str, dict]` registry and YAML files in `configs/` loadable into the dataclass. CLI overrides via a parser (final parser choice deferred to ADR-003).

**Context.** Three candidates considered:
- **Hydra** — powerful, but heavyweight for a single-model project. Composable config groups don't pay off when you have one model and one training recipe.
- **Plain YAML** — untyped; a fat-fingered `n_kv_head: "2"` (string) propagates silently.
- **Dataclass** — typed, IDE-friendly, no `exec()` hostility (vs nanoGPT/llama2.c `configurator.py`), trivially serialises to JSON for HF Hub checkpoints.

**Consequences.** `ForgeConfig.__post_init__` does shape validation (`n_head % n_kv_head == 0`, `head_dim * n_head == d_model`, `head_dim % 2 == 0` for RoPE). Saved checkpoints carry the config as JSON via `dataclasses.asdict()` — HF Hub artifacts are self-describing. We cannot do Hydra-style sweep configs without writing them ourselves; acceptable.

---

## ADR-002 — Experiment tracking: wandb (CLOSED)

**Status:** Accepted — 2026-05-26 (Phase C).

**Decision.** Use `wandb` for experiment tracking. Run resume via `wandb.init(id=..., resume="must")`. If `WANDB_API_KEY` is absent, fall back loudly to CSV logging into `runs/<run_id>/metrics.csv`.

**Context.** MLflow is the other contender. wandb's free-tier streaming charts and Kaggle integration are better; the run-resume API (resume by ID) maps cleanly to our session-kill recovery story. MLflow free-tier UX on Kaggle for streaming metrics is weaker.

**Consequences.** Adds `wandb` to `pyproject.toml` deps. Wandb run ID becomes one of the 9 keys in our resume checkpoint (per `CLAUDE.md` §8). CSV fallback is `WARNING`-logged, never silent.

---

## ADR-003 — CLI parser: `argparse` (CLOSED)

**Status:** Accepted — 2026-05-28 (M12).

**Decision.** Use stdlib `argparse` for the `forge-llm` console entrypoint and for the `python -m forge_llm.train` direct module entrypoint. No `tyro` dependency.

**Context.** Two candidates: `tyro` (auto-generates CLI from `@dataclass` fields) and `argparse` (stdlib). The deciding factors at M12 lock-in:
- Every script and module CLI already written through M11 uses `argparse` (`scripts/train_bpe.py`, `scripts/eval_perplexity.py`, `scripts/benchmark_kv_cache.py`, `forge_llm.train.main`). Switching to `tyro` would mean rewriting four working CLIs to gain a single new dep.
- The `forge-llm` entrypoint has four subcommands (`generate`, `train`, `eval`, `bench-cache`), each with a different argument surface. `tyro`'s "one dataclass → one CLI" sweet spot doesn't map cleanly onto subcommands without extra plumbing.
- `argparse` subparsers handle this in ~100 LoC; the per-subcommand argument list is short enough that hand-writing it is not a maintenance burden.

**Consequences.** No external CLI-parser dep. `forge-llm --help` renders the standard argparse subcommand help. Future evolution: if any one subcommand grows enough config surface to warrant dataclass-driven auto-generation, that subcommand can be migrated to `tyro` in isolation without touching the others.

---

## ADR-004 — Tokenization: on-the-fly vs pre-packed (OPEN)

**Status:** Open — resolves in Phase F (preflight).

**Decision (pending).** Default to on-the-fly BPE tokenization in the dataloader workers. Switch to pre-packed `.bin` shards (nanoGPT pattern) on a Kaggle Dataset if Phase F shows GPU utilisation < 70% with tokenisation on the critical path.

**Context.** On-the-fly is zero-setup and "Run All" friendly for forkers. Pre-packed gives steady throughput at the cost of a one-time data-prep step. The trade-off depends on T4 dataloader CPU contention.

**Consequences.** Resolution criterion: in Phase F's 100-step smoke run, measure `gpu_util` over the last 50 steps. If mean > 70%, on-the-fly stays. If < 70%, we add `scripts/prepare_data.py` to pre-tokenize a shard and add a `--data-path` flag.

---

## ADR-005 — Final `n_layer`: 6 vs 8 (OPEN)

**Status:** Open — resolves in Phase F.

**Decision (pending).** Final `n_layer` is 6 (per BUILD_PLAN spec, parameter count ~25.3M) or 8 (lands ~31.2M).

**Context.** The BUILD_PLAN names the config "model_30m" but the strict 6-layer spec lands at ~25M. T4 16 GB has more than enough headroom for 8 layers if micro_bs=8 fits.

**Consequences.** Resolution criterion: in Phase F, measure peak GPU memory at n_layer=6, micro_bs=8. If headroom > 30% (i.e., peak < 11 GB), bump to n_layer=8 and re-measure. If headroom at n_layer=8 is < 10%, revert to n_layer=6 and rename the preset to `forge-25m`. Either way, the README perplexity table cites the actual param count, not the aspirational label.

---

## ADR-006 — `torch.compile` default in `generate()` (OPEN)

**Status:** Open — resolves in M11 / Phase F.

**Decision (pending).** Whether `generate()` enables `torch.compile` by default.

**Context.** gpt-fast shows compile gives ~2× generation speedup for Llama-class models. The cost: compile time (~30 s warmup), incompatibility with some debugging workflows, and a hard requirement on PyTorch ≥2.2.

**Consequences.** Resolution criterion: in M11, benchmark `generate()` with and without compile at ctx=128 / 512 / 2048. If `--compile` gives ≥1.5× speedup *and* `forge-llm generate` cold-start time stays under 60 s (compile included), enable by default with `--no-compile` opt-out. Otherwise opt-in via `--compile`.

---

## ADR-007 — RoPE convention: HF Llama half-split (CLOSED)

**Status:** Accepted — 2026-05-26 (Phase C). **Amended — 2026-05-28 (M3)** after verifying the actual HF source; see "Amendment" below.

**Decision.** Use the **HF Llama half-split RoPE convention**: rotate halves `(x[:d/2], x[d/2:])` via the formula `y = x*cos + rotate_half(x)*sin` where `rotate_half([a, b]) = [-b, a]`. This is what `transformers.models.llama.modeling_llama.apply_rotary_pos_emb` actually does.

**Context.** Two conventions exist in the wild:
- **Meta interleaved** — rotate adjacent pairs `(x_{2i}, x_{2i+1})`. Used by Meta's original Llama reference and by some llama.cpp paths.
- **HF half-split** — rotate halves `(x_i, x_{i + d/2})`. Used by HuggingFace `transformers`.

The two are isomorphic up to a per-head permutation of W_Q and W_K, but they are *not* bitwise-compatible on the same raw weights. Our value-test oracle is HF's `apply_rotary_pos_emb`; matching the oracle by construction (rather than via permutation gymnastics inside the test) is operationally simpler and means M4+ attention can consume rotated q/k without an extra layout transform.

**Consequences.** `apply_rotary(q, k, freqs_cis)` in `src/forge_llm/rope.py` MUST implement HF half-split. The test `test_rope_value_vs_llama` is the binding correctness gate and compares against HF's `apply_rotary_pos_emb` directly with no permutation around the call. If we ever load a Meta-Llama checkpoint, W_Q/W_K need permuting at load time — a one-line transform documented at that load site.

**Amendment (2026-05-28, M3).** The original ADR text claimed "HF Llama oracle uses interleaved" and prescribed Meta interleaved. That was factually wrong — `transformers/models/llama/modeling_llama.py::rotate_half` splits halves, not adjacent pairs. M3 corrects the decision to match what HF actually does. The intent (match the HF oracle) is preserved; only the description of HF's convention is fixed.

---

## ADR-008 — micro_bs / grad_accum split (OPEN)

**Status:** Open — resolves in Phase F.

**Decision (pending).** Final `(micro_bs, grad_accum)` pair such that `micro_bs × grad_accum × seq_len == 128K` tokens and Phase F memory measurement leaves ≥10% T4 headroom.

**Context.** Baseline plan: micro_bs=8, grad_accum=16. Fallback ladder in `docs/03_training_plan.md` §2.2.

**Consequences.** Resolution criterion: walk the ladder until ≥10% headroom is achieved. The chosen pair is recorded here with the measured peak memory.

---

## ADR-009 — Total token budget (OPEN)

**Status:** Open — resolves in Phase F.

**Decision (pending).** Total training tokens: 1B (baseline) or 500M (downscaled if Phase F throughput falls short).

**Context.** Phase F measures tokens/sec on T4. The plan budgets ~46 hours wall-clock at 6,000 tokens/sec. If measured tokens/sec < 3,000, the projected wall-clock balloons to ~90 hours = ~8 sessions, exceeding the 30-hour weekly quota.

**Consequences.** Resolution criterion: if measured tokens/sec ≥ 4,500, keep 1B. If 2,500–4,500, halve to 500M. If < 2,500, halve to 250M and reduce per-param efficiency chart claims accordingly.

---

## ADR-010 — Final number of Kaggle sessions allocated (OPEN)

**Status:** Open — resolves in Phase F.

**Decision (pending).** Allocate N Kaggle sessions for the full training run.

**Context.** Function of ADR-008 and ADR-009. Likely 3–4.

**Consequences.** Calendar implication: 3 sessions ≈ 1 week (with weekly 30h quota), 4 ≈ 1.5 weeks. Phase G schedule is set after this resolves.

---

## ADR-011 — License: MIT (CLOSED)

**Status:** Accepted — 2026-05-26 (Phase D).

**Decision.** MIT license. `LICENSE` file in repo root.

**Context.** Three considered: MIT, Apache-2.0, BSD-3-Clause. MIT is the simplest, most-permissive, and the default for `pip`-installable single-author projects.

**Consequences.** Anyone can use Forge-LLM for any purpose including commercial. We forgo Apache-2.0's explicit patent-grant clause; acceptable for a 30M research model.

---

## ADR-012 — Initial dependency set (CLOSED)

**Status:** Accepted — 2026-05-26 (Phase D).

**Decision.** Initial runtime dependencies pinned in `pyproject.toml`:
- `torch>=2.2,<2.10` — the only deep-learning runtime; the framework we build on.
- `numpy>=1.26,<3` — RNG state, array ops in tokenizer training.
- `datasets>=2.18,<4` — FineWeb-Edu streaming.
- `wandb>=0.17,<1` — experiment tracking (per ADR-002).
- `huggingface_hub>=0.23,<1` — HF Hub checkpoint upload (per `docs/03_training_plan.md` §6).
- `pyyaml>=6,<7` — `configs/*.yaml` loading.

Dev dependencies:
- `pytest>=8`, `pytest-cov>=5`, `ruff>=0.4`, `mypy>=1.10` — testing + lint + types.

Test-only dependencies (oracles, not for `src/`):
- `transformers>=4.41,<5` — HF Llama oracle for value tests.
- `tiktoken>=0.7,<1` — BPE sanity comparison.

**Context.** Hard rule 6 in `BUILD_PLAN.md` §2 requires a one-line justification per dep.

**Consequences.** No `accelerate`, no `deepspeed`, no `xformers`, no `flash_attn`. Each new dep added later requires a new ADR entry here.

---

## ADR-013 — Build backend: hatchling (CLOSED)

**Status:** Accepted — 2026-05-26 (Phase D).

**Decision.** `hatchling` as the build backend, declared in `pyproject.toml [build-system]`.

**Context.** `setuptools` is the legacy default; `hatchling` is the modern PEP 621-native choice and what PyPA officially recommends for new projects. Either works.

**Consequences.** `pyproject.toml [build-system].requires = ["hatchling"]` and `build-backend = "hatchling.build"`. No `setup.py`.
