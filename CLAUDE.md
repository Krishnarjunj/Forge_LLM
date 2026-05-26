# CLAUDE.md — Forge-LLM Persistent Context

This file is loaded into every Claude Code conversation in this repo. Treat it as the binding contract for how this project is built. When in doubt, this file wins over any conversational suggestion. Source documents that informed this file: `./forge_llm.md` (project brief) and `./BUILD_PLAN.md` (build runbook).

---

## 0. Absolute hard rules (set by user, override everything else)

These rules are non-negotiable. They override §11, §13, and any conversational instruction that contradicts them. Set on 2026-05-27 at the start of Phase E.

- **NEVER PUSH.** Do not run `git push`, `git push --force`, `git push origin`, or any variant. Commits stay local. The user pushes manually if and when they choose. This applies to every branch, every remote, every milestone.
- **Commit messages: max 2 words.** Every commit message subject is at most two words. No body. No trailer (no `Co-Authored-By`, no anything). Examples that fit: `init scaffold`, `add tokenizer`, `fix rope`, `M1 tokenizer`. Examples that do not fit: anything three words or longer, anything with a body, anything with a trailing co-author line.

If a workflow step in §13 or elsewhere in this file says "push", treat it as "stop after commit and wait for the user."

---

## 1. Project goal

Forge-LLM is a ~30M-parameter decoder-only transformer written from scratch in PyTorch, trained on FineWeb-Edu using the free Kaggle T4 GPU tier ($0 compute), and released as a pip-installable package with a HuggingFace Hub checkpoint and a 3000-word architecture deep-dive blog. The goal is **depth signal**: every layer (BPE tokenizer, RMSNorm, RoPE, Grouped-Query Attention, SwiGLU, KV-cache) is implemented from the underlying math without leaning on `transformers`, `xformers`, or fused attention kernels. The success bar is: beats untrained baseline by ≥10× perplexity, reproduces a scaled-down ~30M nanoGPT-equivalent on WikiText-103 within a reasonable margin, ships with green CI, resumable training, and a publicly-forkable Kaggle training notebook.

---

## 2. Hard import rules

### Forbidden in `src/` (zero matches in `grep` audit)
- `transformers` (HuggingFace) — neither `import transformers` nor `from transformers ...`
- `xformers`
- `flash_attn`
- `apex`
- `torch.nn.MultiheadAttention`
- `torch.nn.functional.scaled_dot_product_attention`
- `triton` (unless the optional FlashAttention-2 stretch is in scope, which it is not by default)
- Any pre-built positional embedding or attention module that does the math for us

### Allowed only in `tests/` (as oracles for value-vs-reference checks)
- `transformers.models.llama.modeling_llama.LlamaRMSNorm`
- `transformers.models.llama.modeling_llama.LlamaRotaryEmbedding` (or equivalent reference)
- `transformers.LlamaModel` for end-to-end attention/block comparisons on toy configs
- `tiktoken` for BPE comparison checks (vocab sanity, encode/decode round-trips against a known tokenizer)

### Always allowed everywhere
- `torch`, `torch.nn`, `torch.nn.functional` (except the forbidden symbols above)
- `numpy`
- `datasets` (HuggingFace streaming only — for FineWeb-Edu)
- `wandb` (experiment tracking — see §10 for the wandb-vs-mlflow choice, locked in `docs/DECISIONS.md`)

### Audit command (run before every commit and in CI)
```
grep -rE "(MultiheadAttention|scaled_dot_product_attention|xformers|flash_attn|^from transformers|^import transformers|apex)" src/
```
Must return zero matches.

---

## 3. Reference implementations (test oracles only)

Reference code is **never copied** into `src/`. It exists to answer: "does our from-scratch layer produce the same numbers as a known-correct implementation on identical inputs?"

- **nanoGPT** (Karpathy) — used as a structural sanity reference for the training loop, init, and config style. Not imported.
- **HuggingFace `LlamaModel`** — used in `tests/` as a value oracle for RMSNorm, RoPE, attention output, and full-block forward pass. Imported only under `tests/`.
- **`tiktoken`** — used as a BPE round-trip / encoding-soundness oracle in `tests/`, never in `src/`.
- **`torch.nn.MultiheadAttention`** — used as a vanilla-MHA oracle in `tests/test_attention.py::test_mha_matches_torch_reference` to validate the MHA implementation on a non-GQA config.

If an oracle cannot be set up (e.g., HF model fails to load on the test runner), the dependent test is `pytest.skip("oracle not available: <reason>")` with a TODO referencing the milestone. **Never fabricate expected values.**

---

## 4. Numerical tolerances

Default tolerances for `torch.allclose` / `torch.testing.assert_close` when comparing against a reference oracle:

| Compute dtype | atol  | rtol  |
|---------------|-------|-------|
| fp32          | 1e-5  | 1e-5  |
| bf16          | 1e-3  | 1e-3  |
| fp16          | 1e-3  | 1e-3  |

Tighter tolerances may be set per-test if the operation is exactly equivalent (e.g., RMSNorm with the same epsilon and same dtype should reach 1e-6). Looser tolerances require a justification comment in the test naming the source of additional drift (e.g., softmax + matmul accumulation order differs across kernels).

---

## 5. Determinism rules

Every test and every training entrypoint **must** call this seeding block before any random op:

```python
import random, numpy as np, torch
def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

- Training and resume checkpoints persist `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, `np.random.get_state()`, and `random.getstate()`.
- The data iterator's step index is checkpointed so resume continues from the same shard offset, not the start.
- Wandb run ID is checkpointed so resumed runs append to the same run.
- The resume-safety test (see §7) requires that loss curves are bitwise-indistinguishable for ≥100 steps after resume on a single GPU; on multi-GPU only `atol=1e-4` is acceptable, but multi-GPU is out of scope for this build.

If a kernel does not have a deterministic implementation, the test must `pytest.skip("non-deterministic kernel: <op>")` rather than silently relaxing the tolerance.

---

## 6. Dtype & device rules

- **Master weights**: fp32, held in optimizer state.
- **Compute on Kaggle T4**: fp16 forward/backward via `torch.autocast(device_type="cuda", dtype=torch.float16)`. T4 does **not** support native bf16 — we explicitly choose fp16 and document the loss-scaling requirement.
- **Loss-scaling**: `torch.cuda.amp.GradScaler`. Initial scale 2**16, growth interval 2000 steps. Skip optimizer step on inf/NaN scale.
- **Softmax in attention**: cast to fp32 before softmax, cast back after — required for fp16 stability in the attention kernel.
- **LayerNorm/RMSNorm**: compute in fp32, return in input dtype.
- **CPU dev**: fp32 throughout. Tests run in fp32 by default; mixed-precision tests must explicitly switch.
- **No bf16 anywhere** in this project (would silently fail on T4).

---

## 7. Test-first discipline (per layer)

Every implementation file in `src/forge_llm/` is preceded by a matching test file in `tests/`. The test must exist and have these four cases (where applicable) before the implementation is written:

1. **Shape test** — forward pass on a tiny config produces the expected output shape (no value assertion). Catches indexing/transpose bugs.
2. **Value-vs-oracle test** — same input fed to our impl and the reference oracle (HF `LlamaRMSNorm`, etc.) produces outputs within the tolerances in §4. Skipped (`pytest.skip(...)`) with a TODO if the oracle cannot be set up.
3. **Backward test** — `torch.autograd.gradcheck` on a tiny fp64 config, or a finite-difference check on the loss with respect to a chosen parameter. Catches incorrect gradient routing.
4. **Determinism test** — same input + same seed twice produces bitwise-identical outputs. Catches accidental nondeterministic ops (e.g., `torch.nn.functional.dropout` without seeding, scatter without sort).

In addition, **three specific adversarial tests** must exist:

- **Causal-mask-leak test**: mutate the value of token `T-1`, assert that the output at all positions `t < T-1` is **byte-identical** to the unmutated forward. If a single bit differs, future information leaked through the mask.
- **KV-cache equivalence test**: run full-sequence forward producing logits `L_full`; run the same sequence token-by-token using the KV-cache producing `L_cache`. Assert `torch.allclose(L_full, L_cache, atol, rtol)` per §4.
- **Resume-safety test**: train for 200 steps with checkpoint at step 100, kill the run, resume from the checkpoint, continue to step 300, and assert that steps 101–200 of the uninterrupted run match steps 101–200 of the resumed run on a single GPU. Tolerance: `atol=1e-6` for single-GPU fp32; the test runs in fp32 on CPU to keep CI cheap.

---

## 8. Resumable training rules

A Kaggle session can die at any minute (12h cap + flakiness). Resume must always work. A checkpoint is exactly the tuple:

```
checkpoint = {
    "step": int,
    "model": state_dict (fp32 master copy),
    "optimizer": AdamW state_dict,
    "scheduler": cosine scheduler state_dict,
    "scaler": GradScaler state_dict,
    "rng": {"torch": ..., "cuda": ..., "numpy": ..., "python": ...},
    "data_iterator": {"shard_id": int, "shard_offset": int, "global_step": int},
    "wandb_run_id": str,
    "config_hash": str,  # SHA256 of the resolved config dict at run start
    "git_sha": str,
}
```

- Saved **every 500 steps OR every 30 minutes, whichever comes first.**
- Saved to **HuggingFace Hub or Kaggle Dataset** (configurable). Never to session-local `/kaggle/working/` alone — that disk is wiped on session kill.
- On resume, the loader rebuilds the model with the same config (verified by `config_hash`), restores all states, fast-forwards the data iterator to `shard_offset`, and rejoins the same wandb run via `wandb.init(id=wandb_run_id, resume="must")`.
- If `git_sha` in the checkpoint does not match the current code's SHA, refuse to resume unless `--force` is passed. This catches accidental code drift between sessions.

---

## 9. Free-tier discipline (Kaggle T4)

- **Session limit**: 12 hours per session, 30 hours per week of GPU. Plan for 2–3 sessions across the training window.
- **Storage**: Kaggle session disk is wiped on session end. Persist all checkpoints to HF Hub or a Kaggle Dataset.
- **Checkpoint cadence**: 500 steps or 30 min (whichever first).
- **Eval cadence**: WikiText-103 valid perplexity every 1000 steps (cheap enough at this size).
- **Token budget**: 1B FineWeb-Edu tokens target. Effective batch size 128K tokens. → ~7800 optimizer steps. At ~6K tokens/sec on T4, training fits in ~46 wall-clock hours = ~4 sessions. (Concrete numbers locked in `docs/03_training_plan.md`.)
- **No paid services.** No Modal/Lambda/AWS/GCP. No `wandb` paid-tier features beyond the free dashboards.
- The published Kaggle notebook must run end-to-end on a fresh fork without any private credentials. Secrets (wandb API key, HF token) are read from Kaggle Secrets and skipped gracefully if absent (with a warning, not a crash).

---

## 10. Logging / experiment tracking

We use **wandb** (locked in `docs/DECISIONS.md` ADR-002). Reason: superior streaming charts on free tier, easy run-resume by ID, well-supported on Kaggle. MLflow is rejected for this project because its UX on free-tier notebooks is weaker for streaming metrics.

Logged on every step: `loss`, `lr`, `grad_norm`, `tokens_per_sec`, `gpu_util` (if available), `gpu_mem_alloc`, `gpu_mem_reserved`.
Logged on eval cadence: `valid/perplexity`, `valid/loss`, sample generations (5 fixed prompts).
Logged at run start: full resolved config, `git_sha`, `torch.__version__`, `torch.version.cuda`, GPU name, OS, full pip freeze.

---

## 11. Forbidden anti-patterns

- **Commented-out code.** Delete it; git remembers.
- **Magic numbers.** Constants get named symbols in `configs/` or as module-level `Final[int]`.
- **Untyped tensors.** Public function signatures in `src/` must declare tensor shapes in docstrings using a `Shape:` line, or use `jaxtyping` annotations (`Float[Tensor, "B T D"]`). No bare `Tensor` returns from public functions.
- **Skipped tests without a reason string.** Every `pytest.skip()` requires a string explaining why (oracle missing, kernel non-deterministic, etc.).
- **`# type: ignore` without a justification comment** on the same line.
- **`try: ... except Exception: pass`.** Always name the expected exception class, always handle or re-raise.
- **Silent fallbacks.** If a feature is unavailable (e.g., wandb token missing), log a warning loudly; do not silently downgrade.
- **`global` mutable state.** Pass config objects explicitly.
- **Test fixtures that reach out to the network at collection time.** Tests must be runnable offline (oracles loaded once, cached, or skipped).
- **Commit messages**: max 2 words (see §0). The earlier "describe what and why in present tense" guidance is superseded by §0.

---

## 12. Definition of done

Mirrors `forge_llm.md` "Done definition" — every item must be ticked at release:

- [ ] Public GitHub repo with green CI (lint + type-check + tests on push and PR).
- [ ] `pip install forge-llm` works on a clean Python 3.11+ machine.
- [ ] HuggingFace Hub checkpoint downloadable and loadable via the package.
- [ ] Perplexity table in README: beats untrained baseline by ≥10× **and** within reasonable margin of a scaled-down ~30M nanoGPT-equivalent on WikiText-103.
- [ ] 3000-word blog post live with code excerpts pulled verbatim from `src/` (extracted by a script, not hand-copied), architecture deep-dive, training-curve image, free-tier story, and "what I learned" section.
- [ ] Architecture diagram in README (Mermaid or rendered image).
- [ ] `notebooks/01_load_and_generate.ipynb` runs end-to-end on free Colab.
- [ ] Public Kaggle training notebook published so anyone can fork and reproduce training for free.
- [ ] All 12 milestones in `docs/04_roadmap.md` complete with their exit criteria satisfied.
- [ ] Test coverage ≥ threshold set in `docs/02_correctness_plan.md`.

---

## 13. Workflow rules for Claude Code in this repo

- Every milestone follows: write tests → show tests to user → wait for "tests look good" → write implementation → run tests → run import audit → run ruff → show diff → wait for "commit M<N>" → commit → **stop (no push — see §0)**.
- Every new dependency added to `pyproject.toml` requires a one-line entry in `docs/DECISIONS.md`.
- Every non-obvious design choice (config system, logging stack, license, scheduler curve, etc.) is recorded as an ADR in `docs/DECISIONS.md`.
- Commit messages are constrained by §0 (max 2 words).
- Coding decisions not covered by `docs/01_architecture.md` or `docs/02_correctness_plan.md` are surfaced to the user before implementation and logged as a new ADR.
- All planning artefacts live in `docs/`. Nothing important lives "in chat only."
