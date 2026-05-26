# 04 — Roadmap

> 12 milestones M1–M12, bottom-up build order. Each milestone is small enough to review in one sitting. Phase E executes one milestone at a time with the procedure in `BUILD_PLAN.md` §7.

---

## Milestone summary

| # | Title | Complexity | Files touched (primary) | Test gate |
|---|-------|------------|------------------------|-----------|
| M1 | BPE tokenizer | **M** | `tokenizer.py`, `scripts/train_bpe.py` | `test_tokenizer.py` (all green incl. slow tiktoken sanity) |
| M2 | RMSNorm | S | `norm.py` | `test_norm.py` (all 5 cases green; oracle-skipped if HF missing) |
| M3 | RoPE | M | `rope.py` | `test_rope.py` (all 6 cases green; rotation-identity strict) |
| M4 | Vanilla MHA | M | `attention.py` (MHA path) | `test_attention.py::test_mha_*` green; causal-mask adversarial green |
| M5 | GQA | M | `attention.py` (GQA path) | `test_attention.py::test_gqa_*` green; GQA-reduces-to-MHA green |
| M6 | SwiGLU MLP | S | `mlp.py` | `test_mlp.py` all green |
| M7 | Block assembly | S | `block.py` | `test_block.py` all green; residual routing green |
| M8 | Full model | M | `model.py`, `config.py` | `test_model.py` all green incl. tied-embedding, param-count, vs-Llama |
| M9 | Train loop with resume | **L** | `train.py`, `data.py`, `hub.py` | `test_data.py`, `test_resume.py` (slow) green |
| M10 | Eval (perplexity on WikiText-103) | M | `eval.py`, `scripts/eval_perplexity.py` | Reproduces a known baseline within ±5%; smoke test on tiny model green |
| M11 | Sampling + KV-cache | M | `cache.py`, `sampling.py`, `generation.py`, `scripts/benchmark_kv_cache.py` | `test_cache.py`, `test_sampling.py`, `test_generation.py`; KV-cache equivalence strict |
| M12 | CLI + packaging | S | `cli.py`, `pyproject.toml`, `README.md` | `pip install -e .` works; `forge-llm --help` works; `forge-llm generate` runs on CPU with a fixture checkpoint |

---

## M1 — BPE tokenizer

**Title.** Train and ship a 16K-vocab BPE tokenizer on a FineWeb-Edu subset.

**Files touched (write):**
- `src/forge_llm/tokenizer.py` — `BPETokenizer.train`, `.encode`, `.decode`, `.save`, `.load`. Implement BPE merges from scratch over byte-level tokens; vocab includes `<bos>`, `<eos>`, `<pad>`, `<unk>`.
- `scripts/train_bpe.py` — load ~50MB of FineWeb-Edu via streaming, call `BPETokenizer.train`, save to `configs/tokenizer.json`.
- `tests/test_tokenizer.py` — replace placeholder.

**Entry criteria.**
- Phase D scaffolding done.
- `pytest -q` passes (all-skip).

**Exit criteria.**
- `pytest tests/test_tokenizer.py -v` — all green (5 tests).
- `python scripts/train_bpe.py` produces `configs/tokenizer.json` non-empty.
- Round-trip on 100 random FineWeb-Edu paragraphs is byte-identical.
- `import_audit` shell command in `CLAUDE.md` §2 returns zero matches.
- `ruff check src/forge_llm/tokenizer.py tests/test_tokenizer.py` clean.

**Complexity.** M (~400 LoC tokenizer + ~150 LoC tests).

---

## M2 — RMSNorm

**Title.** Implement RMSNorm in fp32-compute, dtype-preserving I/O.

**Files touched (write):**
- `src/forge_llm/norm.py` — `class RMSNorm(nn.Module)`.
- `tests/test_norm.py` — replace placeholder.

**Entry criteria.**
- M1 green (so we have a working `pytest` pipeline).

**Exit criteria.**
- `pytest tests/test_norm.py -v` — all 5 cases green (or `pytest.skip` with a written reason for the oracle case if HF model isn't installable in the local env).
- Value-vs-LlamaRMSNorm at rtol=1e-6 fp32.
- gradcheck on fp64 passes.
- `import_audit` clean.

**Complexity.** S (~30 LoC norm + ~120 LoC tests).

---

## M3 — RoPE

**Title.** Implement Rotary Positional Embeddings (Llama interleaved convention).

**Files touched (write):**
- `src/forge_llm/rope.py` — `precompute_freqs_cis(head_dim, max_seq, theta) -> Tensor`, `apply_rotary(q, k, freqs_cis) -> (q, k)`. **Llama interleaved convention** (ADR-007 closes here).
- `tests/test_rope.py` — replace placeholder.

**Entry criteria.**
- M2 green.

**Exit criteria.**
- `pytest tests/test_rope.py -v` — all 6 cases green.
- Rotation-identity at position 0 holds to atol=1e-7 fp32.
- Relative-position-invariance holds: `attn(q_m, k_n) == attn(q_{m+s}, k_{n+s})` to rtol=1e-5.
- Long-context (pos=4096) produces finite outputs.
- ADR-007 written in `docs/DECISIONS.md` confirming interleaved convention.
- `import_audit` clean.

**Complexity.** M (~80 LoC rope + ~200 LoC tests).

---

## M4 — Vanilla MHA

**Title.** Implement standard multi-head attention as a stepping stone to GQA. Reuse the GQA module with `n_kv_head == n_head`.

**Files touched (write):**
- `src/forge_llm/attention.py` — `class GroupedQueryAttention(nn.Module)` with code paths that handle both MHA (n_kv_head == n_head) and GQA (n_kv_head < n_head). Implement MHA path first; the GQA-specific path lands in M5.
- `tests/test_attention.py` — write `test_mha_*` cases.

**Entry criteria.**
- M3 green.

**Exit criteria.**
- `pytest tests/test_attention.py::test_mha_* -v` — all 5 MHA cases green.
- Value-vs-`torch.nn.MultiheadAttention` at rtol=1e-5 (oracle imported only in `tests/`).
- **Causal-mask adversarial test** (`tests/test_causal_mask.py::test_causal_mask_adversarial_no_future_leak` on an MHA-only mini-model) passes byte-identically.
- gradcheck on fp64 passes.
- `import_audit` clean.

**Complexity.** M (~200 LoC attention skeleton + ~250 LoC tests).

---

## M5 — GQA

**Title.** Extend attention with KV-head grouping (8 Q, 2 KV).

**Files touched (modify):**
- `src/forge_llm/attention.py` — add the `n_kv_head < n_head` path: project K, V to `n_kv_head` heads, repeat by `n_head // n_kv_head` for the matmul. RoPE applied to Q and to the *unrepeated* K (then K is repeated).
- `tests/test_attention.py` — add `test_gqa_*` cases.

**Entry criteria.**
- M4 green.

**Exit criteria.**
- `pytest tests/test_attention.py -v` — both MHA and GQA cases green (10 total in `test_attention.py`).
- `test_gqa_reduces_to_mha_when_kv_eq_q` passes (regression: M4 still works).
- `test_gqa_with_rope_value_vs_llama` at rtol=1e-5 (oracle from HF Llama).
- KV head grouping count test passes (K shape after repeat is `(B, n_head, T, head_dim)`).
- `import_audit` clean.

**Complexity.** M (~70 LoC added + ~200 LoC tests).

---

## M6 — SwiGLU MLP

**Title.** Implement SwiGLU feedforward (3 linears: gate, up, down).

**Files touched (write):**
- `src/forge_llm/mlp.py` — `class SwiGLU(nn.Module)` with `w_gate`, `w_up`, `w_down`, all bias-free.
- `tests/test_mlp.py` — replace placeholder.

**Entry criteria.**
- M5 green.

**Exit criteria.**
- `pytest tests/test_mlp.py -v` — all 5 cases green.
- Value-vs-`LlamaMLP` at rtol=1e-5 (oracle in `tests/`).
- `test_swiglu_uses_silu_not_gelu` passes (defends against the silent SiLU/GELU swap).
- gradcheck on fp64 passes.
- `import_audit` clean.

**Complexity.** S (~40 LoC mlp + ~150 LoC tests).

---

## M7 — Block assembly

**Title.** Wire RMSNorm + GQA + RMSNorm + SwiGLU into a pre-norm decoder block with residuals.

**Files touched (write):**
- `src/forge_llm/block.py` — `class TransformerBlock(nn.Module)`. Pre-norm: `x + attn(norm1(x))`, then `x + mlp(norm2(x))`.
- `tests/test_block.py` — replace placeholder.

**Entry criteria.**
- M6 green.

**Exit criteria.**
- `pytest tests/test_block.py -v` — all 4 cases green.
- Value-vs-`LlamaDecoderLayer` at rtol=1e-5.
- `test_block_residual_routing` passes (zero-out attn submodule, output equals `input + MLP(norm2(input))`).
- gradcheck on fp64 passes.
- `import_audit` clean.

**Complexity.** S (~60 LoC block + ~150 LoC tests).

---

## M8 — Full model

**Title.** Stack 6 blocks, add embedding (tied with LM head), final RMSNorm. Implement `ForgeConfig`, `PRESETS`, `ForgeModel`, `ForgeForCausalLM`.

**Files touched (write):**
- `src/forge_llm/config.py` — `@dataclass ForgeConfig` with `__post_init__` validation; `PRESETS = {"forge-30m": {...}}`.
- `src/forge_llm/model.py` — `class ForgeModel(nn.Module)` (trunk), `class ForgeForCausalLM(nn.Module)` (LM head, tied).
- `src/forge_llm/__init__.py` — expose `ForgeConfig`, `PRESETS`, `ForgeForCausalLM`, `ForgeModel`.
- `tests/test_model.py` — replace placeholder.

**Entry criteria.**
- M7 green.

**Exit criteria.**
- `pytest tests/test_model.py -v` — all 5 cases green.
- `test_model_param_count` matches the table in `docs/01_architecture.md` §6 exactly.
- `test_model_embeddings_are_tied` — identity check on weight tensor passes.
- `test_model_value_vs_llama_smallcfg` at rtol=1e-4 (loosened from 1e-5 due to accumulated drift across 6 layers).
- `ForgeConfig` validation rejects invalid shapes (e.g., n_head=8, n_kv_head=3 — not a divisor).
- `import_audit` clean.

**Complexity.** M (~150 LoC model + config + ~200 LoC tests).

---

## M9 — Train loop with resume

**Title.** Build the training loop with full resume semantics. Implements differentiation move #1 (Kaggle reproducibility).

**Files touched (write):**
- `src/forge_llm/data.py` — `PackedFineWebEdu` streaming iterable with packing and iterator-state save/load.
- `src/forge_llm/train.py` — `Trainer` class with `train_steps()`, `save_checkpoint()`, `load_checkpoint()`. AdamW + cosine LR + GradScaler. Wandb integration with resume-by-id. Session-budget watchdog (per `docs/03_training_plan.md` §10).
- `src/forge_llm/hub.py` — `save_pretrained`, `from_pretrained`, `push_to_hub`, `pull_latest_checkpoint`. HF Hub primary, Kaggle Dataset mirror.
- `tests/test_data.py` — replace placeholder.
- `tests/test_resume.py` — replace placeholder. Slow test.
- `notebooks/02_kaggle_train.ipynb` — published Kaggle training notebook stub (real cells filled in Phase F).

**Entry criteria.**
- M8 green.

**Exit criteria.**
- `pytest tests/test_data.py -v` — all 3 cases green.
- `pytest tests/test_resume.py -v -m slow` — `test_resume_safety_loss_curve_indistinguishable` passes with `atol=1e-6` on CPU fp32 (this is the headline resume-safety test).
- `python -m forge_llm.train --config configs/model_30m.yaml --steps 5 --no-wandb` runs end-to-end on CPU with mock data without errors (smoke).
- Checkpoint round-trip: train 10 steps, save, load in new process, continue 10 more steps, loss curve byte-identical to uninterrupted run.
- HF Hub upload helper has a `--dry-run` mode tested.
- `import_audit` clean.

**Complexity.** **L** (~600 LoC train+data+hub + ~400 LoC tests). The largest milestone; expect 2–3 reviewer back-and-forth iterations.

---

## M10 — Eval (perplexity on WikiText-103)

**Title.** Implement perplexity evaluation, validate against a known baseline.

**Files touched (write):**
- `src/forge_llm/eval.py` — `eval_perplexity(model, dataset, max_tokens) -> float`. Byte-level PPL (so it's tokenizer-agnostic for cross-model comparison).
- `scripts/eval_perplexity.py` — CLI wrapper: `python scripts/eval_perplexity.py --model <ckpt> --dataset wikitext-103`.
- `tests/test_eval.py` — write tests for byte-PPL math on a known small case.

**Entry criteria.**
- M8 green (eval doesn't strictly need M9, but the saved-checkpoint format does).

**Exit criteria.**
- `pytest tests/test_eval.py -v` — all green.
- On an untrained `forge-30m`, PPL on WikiText-103 valid is in `[5000, 100000]` (sanity bound — random tokens give ~vocab_size PPL = 16384; the actual model is *initialised* not random, so we'd see ~10K).
- Reproduces nanoGPT-124M reported WikiText-103 valid PPL within ±5% when run on a downloaded nanoGPT checkpoint (this is the "known baseline" sanity check).
- `import_audit` clean.

**Complexity.** M (~100 LoC eval + ~150 LoC tests + a sanity reproduction).

---

## M11 — Sampling + KV-cache

**Title.** Implement statically-allocated KV cache (gpt-fast pattern), sampling (top-p/top-k/temperature/repetition penalty), and the generation loop. Implements differentiation move #2 (KV-cache speedup table).

**Files touched (write):**
- `src/forge_llm/cache.py` — `class KVCache(nn.Module)` with `allocate(cfg, max_batch)`, `.update(input_pos, k, v) -> (k_full, v_full)`, statically pre-allocated buffers.
- `src/forge_llm/sampling.py` — pure functions: `top_k`, `top_p`, `apply_temperature`, `repetition_penalty`.
- `src/forge_llm/generation.py` — `generate(model, tokenizer, prompt, *, max_new, top_p, top_k, temperature, rep_penalty, seed, use_cache) -> Iterator[str]`.
- `src/forge_llm/__init__.py` — expose `generate`.
- `scripts/benchmark_kv_cache.py` — tokens/sec measurement, cache off vs on, at ctx 128/512/2048. Writes results to `docs/results/kv_cache_bench.md`.
- `tests/test_cache.py`, `tests/test_sampling.py`, `tests/test_generation.py` — replace placeholders.
- `src/forge_llm/attention.py` — extend `forward()` to accept optional `cache` and `input_pos` kwargs; existing tests must still pass.

**Entry criteria.**
- M10 green (so the saved-checkpoint format is stable).

**Exit criteria.**
- `pytest tests/test_cache.py -v` — all 3 cases green.
- **`test_kvcache_full_vs_token_by_token_equivalence` at rtol=1e-5** (the second headline test).
- `pytest tests/test_sampling.py -v` — all 4 cases green.
- `pytest tests/test_generation.py -v` — all 4 cases green, including the deterministic 100-token gen.
- `python scripts/benchmark_kv_cache.py` produces `docs/results/kv_cache_bench.md` with a real table.
- `import_audit` clean.

**Complexity.** M (~350 LoC cache+sampling+generation + ~300 LoC tests + the benchmark script).

---

## M12 — CLI + packaging

**Title.** Wire up the `forge-llm` console entrypoint. Make the package pip-installable. Implements differentiation move #5 (`pip install` + 30-second demo).

**Files touched (write):**
- `src/forge_llm/cli.py` — `main()` dispatch on subcommand: `generate`, `train`, `eval`, `bench-cache`.
- `pyproject.toml` — already exists from Phase D; verify `[project.scripts]` entrypoint, version bump to `0.1.0a1`.
- `README.md` — fill in `<TODO>` placeholders that depend on M11/M12 (CLI usage, quickstart). Architecture diagram, training-curve, PPL table remain `<TODO>` until Phase G/H.
- ADR-003 closes here: CLI parser choice locked.

**Entry criteria.**
- M11 green.

**Exit criteria.**
- `pip install -e .` from repo root succeeds in a clean venv.
- `forge-llm --help` shows the subcommands.
- `forge-llm generate "Hello" --max-new 20 --checkpoint tests/fixtures/tiny_model.pt` produces (gibberish but non-empty) output on CPU in <30 seconds.
- `forge-llm bench-cache --max-seq 2048` runs and produces a table.
- All previous tests still green: `pytest -q -m "not slow"` clean.
- `import_audit` clean.
- ADR-003 written in `docs/DECISIONS.md`.

**Complexity.** S (~100 LoC CLI + ~50 LoC tests).

---

## Cross-milestone gates

Between milestones, before the user is asked to commit:

1. **`grep -rE` import audit** in `src/` — zero matches. Output pasted in the PR description.
2. **`ruff check src/ tests/`** — clean.
3. **`mypy src/`** — clean (strict mode).
4. **`pytest -q -m "not slow"`** — all pass.
5. **Coverage** — `pytest --cov=src/forge_llm` shows ≥85% on `src/forge_llm/` (excluding `train.py` and `data.py` per `docs/02_correctness_plan.md` §3).

The user sees these outputs before saying "commit M<N>".

---

## Phase F (preflight) — sits after M12, before training

Per `BUILD_PLAN.md` §8, after M12 we run the preflight checklist:
- Overfit-one-batch
- 100-step smoke run on FineWeb-Edu stream
- Tokens/sec measurement
- Memory headroom on T4
- Checkpoint resume across a hard process kill
- Wandb config dump confirms `git_sha`, model config, env

Preflight closes the open ADRs:
- ADR-004 (pre-pack vs on-the-fly)
- ADR-005 (n_layer=6 vs 8)
- ADR-008 (micro_bs / grad_accum split)
- ADR-009 (token budget)
- ADR-010 (number of sessions)

---

## Differentiation move ↔ milestone mapping (carried from `docs/research/01_landscape.md`)

| Diff move | Lands in |
|-----------|---------|
| 1. Public Kaggle notebook reproducing training end-to-end | M9 (notebook stub) → Phase F (preflight) → Phase G (execution) |
| 2. KV-cache speedup benchmark table | M11 (benchmark script) → Phase H (README) |
| 3. Per-parameter efficiency chart | M10 (eval) → Phase G (results) → Phase H (blog + README) |
| 4. RoPE-vs-learned-pos + GQA-vs-MHA ablation | Phase H (short rerun, scope-bound) |
| 5. `forge-llm generate` CLI + asciinema demo | M12 (CLI) → Phase H (asciinema recording) |
