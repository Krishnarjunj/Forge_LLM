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

## ADR-003 — CLI parser: `tyro` (OPEN)

**Status:** Open — resolves in M12.

**Decision (pending).** `tyro` or `argparse` for the `forge-llm` console entrypoint and for `python -m forge_llm.train`.

**Context.** `tyro` integrates cleanly with `@dataclass` configs — it generates the CLI from the dataclass automatically. `argparse` is the stdlib default and has no external dep. The Forge-LLM dataclass has ~12 fields; both work.

**Consequences.** If `tyro` is chosen, adds one dep. If `argparse` is chosen, we hand-write the field list. Decision: try `tyro` first in M12; if it adds friction or doesn't compose with subcommands cleanly, drop to `argparse`. Resolution criterion: write the CLI in `tyro` first; if `forge-llm generate --help` doesn't render cleanly within 30 minutes of effort, switch to `argparse`.

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

## ADR-007 — RoPE convention: Llama interleaved (CLOSED)

**Status:** Accepted — 2026-05-26 (Phase C; will be re-confirmed in M3 once code lands).

**Decision.** Use the **Llama interleaved RoPE convention**: rotate adjacent pairs `(x_{2i}, x_{2i+1})` rather than the original-paper paired convention `(x_i, x_{i + d/2})`.

**Context.** The two conventions are isomorphic up to a permutation of the head-dimension axis, but they are *not* bitwise-compatible. The HF Llama oracle uses interleaved. We compare against the HF Llama oracle for value tests. Therefore Forge-LLM must use interleaved too — otherwise the oracle test produces a misleading failure.

**Consequences.** `apply_rotary(q, k, freqs_cis)` in `src/forge_llm/rope.py` MUST implement interleaved rotation. The test `test_rope_value_vs_llama` is the binding correctness gate. The decision is recorded here so a future re-read of `rope.py` doesn't accidentally "fix" it to the paper convention.

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
