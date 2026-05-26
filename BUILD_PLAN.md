# BUILD_PLAN.md — Forge-LLM Autonomous Build Runbook

> **Single source of truth for the entire build.** Claude Code reads this file once and executes phases A–D autonomously, then stops at GATE 1 for human action. No back-and-forth prompting in between.

---

## 0. How to use this file

### One-time kickoff prompt (paste into Claude Code, exactly once):

```
Read ./BUILD_PLAN.md and ./forge_llm.md in this directory.
Execute Phases A through D sequentially and autonomously.
Do not pause between phases. Do not ask for confirmation 
between phases. Do not perform ANY git operations 
(no `git init`, `git add`, `git commit`, `git remote`, `git push`).
When you reach GATE 1, stop completely and wait for my reply.
Begin Phase A.
```

### Execution model

- Phases tagged **[auto]** run without confirmation. Claude Code proceeds straight to the next phase.
- Phases tagged **⛔ GATE** are hard stops. Claude Code must stop and surface the gate-block text to the user, then wait for the user's next message.
- All planning artefacts are written to `docs/` as files. Nothing is "in chat only."
- When a phase has subagent steps, Claude Code spawns subagents in parallel (Task tool) and synthesizes their outputs into one deliverable.

### Stop conditions

Claude Code must stop and surface the issue (not silently degrade) if:

- A required input file (forge_llm.md, ../Placements_25_26/) is missing
- A reference implementation cannot be located when needed as a test oracle
- A gate is reached
- Any rule in Section 2 (Hard rules) is about to be violated

---

## 1. Inputs & assumptions

**Inputs in this repo (current working directory):**
- `./BUILD_PLAN.md` — this file
- `./forge_llm.md` — full project brief (architecture, validation bar, done definition)

**Inputs outside this repo (read-only reference):**
- `../Placements_25_26/` — cohort of student projects, including but not limited to CS23S011 (Qwen3-mini), ME22B032 (GPT-2 from scratch), AE22B022 (FFN in C++). Used for differentiation analysis only — never copy.

**Assumed environment:**
- Python 3.11+
- A modern PyTorch install (≥2.2)
- Local development on CPU or modest GPU; the real training run happens later on free Kaggle T4
- Git is installed but **no remote is configured yet**

**Assumed not present yet (set up at GATE 1):**
- GitHub account for the new repo
- GitHub credentials configured locally
- The remote repository itself

---

## 2. Hard rules (NO-GO list)

These rules override anything else in this file. If a step would violate one, stop instead.

1. **No git network/state operations before GATE 1.** Forbidden until then: `git init`, `git add`, `git commit`, `git remote`, `git push`, `git config user.*`, `gh repo create`, `git tag`. Reading git state with `git status` is fine (will show "not a git repo"; expected).
2. **No `transformers`, `xformers`, `flash_attn`, `apex`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`** anywhere in `src/`. Allowed only as test oracles inside `tests/` (imported, run on identical inputs, output compared to our implementation).
3. **No fabricated values in tests.** If a reference oracle cannot be set up, the test is skipped with a `pytest.skip("oracle not available: <reason>")` and a TODO. Never invent expected numbers.
4. **No paid services.** No Modal/Lambda/AWS/GCP API calls. No PyPI uploads or HF Hub uploads during planning (those happen post-coding, separately approved).
5. **No commits-on-completion habit.** Even after GATE 1, every commit requires Claude Code to show the staged diff and wait for "go" before running `git commit`.
6. **No silent dependency creep.** Every new entry added to `pyproject.toml` requires a one-line justification in `docs/DECISIONS.md` (an ADR-lite log).
7. **No "any-type" Python.** Tensor shapes documented in docstrings or jaxtyping; no untyped function signatures in `src/`.

---

## 3. Phase A — Bootstrap & CLAUDE.md `[auto]`

**Goal:** Set up the persistent context file Claude Code will use throughout the project.

**Steps:**

1. Read `./forge_llm.md` fully into context.
2. Verify `../Placements_25_26/` exists and is non-empty. If missing, stop with a clear error.
3. Create `./CLAUDE.md` with these sections (content based on the brief + Section 2 hard rules above):
   - Project goal (one paragraph)
   - Hard import rules (forbidden, allowed)
   - Reference implementations (test oracles only — nanoGPT, HuggingFace LlamaModel)
   - Numerical tolerances (1e-5 fp32, 1e-3 fp16/bf16)
   - Determinism rules (seeds, deterministic algorithms)
   - Dtype & device rules (fp16 on T4, fp32 master copy, fp32 softmax)
   - Test-first per layer (shape / value-vs-oracle / backward / determinism)
   - Resumable training rules (save model + optimizer + scheduler + RNG + iterator + wandb id)
   - Free-tier discipline (Kaggle session limits, checkpoint cadence, storage to HF Hub/Kaggle Dataset, not session disk)
   - Forbidden anti-patterns (commented-out code, magic numbers, untyped tensors, skipped tests without reason)
   - Definition of done (mirror forge_llm.md)

**Exit criteria:**
- `./CLAUDE.md` exists and covers every section above.
- Proceed to Phase B without confirmation.

---

## 4. Phase B — Landscape exploration `[auto]`

**Goal:** Position Forge-LLM against cohort and open-source baselines so Phase C makes informed design decisions.

**Steps:**

1. Spawn three subagents in parallel:

   **Subagent B1 — Cohort review.** Walk `../Placements_25_26/`. For every project touching ML, transformers, NLP, or training-from-scratch (with special focus on CS23S011, ME22B032, AE22B022), extract:
   - What was actually implemented vs claimed (read READMEs and code)
   - Depth ceiling (initialized only? trained? evaluated?)
   - Missing pieces a stronger version would have
   - One paragraph each

   **Subagent B2 — Reference repos.** Survey nanoGPT (Karpathy), llama2.c, gpt-fast, and HuggingFace LlamaModel source (use web_fetch if available; else use the agent's knowledge and note the assumption). For each, capture:
   - Code organization patterns
   - Config style
   - Test/eval setup
   - What Forge-LLM should steal vs improve on

   **Subagent B3 — Differentiation.** Given Forge-LLM's brief, identify the specific delta vs:
   - The 3 cohort projects above
   - nanoGPT (MHA + learned pos emb + LayerNorm + GELU — Forge-LLM uses GQA + RoPE + RMSNorm + SwiGLU)
   - What claim can Forge-LLM make that none of these can?

2. Synthesize outputs into a single file: `docs/research/01_landscape.md`. Include a "Top 5 differentiation moves" section at the end.

**Exit criteria:**
- `docs/research/01_landscape.md` exists with all three subagent outputs synthesized.
- The Top 5 differentiation moves are concrete (e.g., "publish reproducible Kaggle notebook" not "be reproducible").
- Proceed to Phase C without confirmation.

**Reasoning depth:** `think hard` before synthesizing.

---

## 5. Phase C — Architecture & plans `[auto]`

**Goal:** Produce every planning document needed for coding. After this phase, coding should be mechanical translation.

**Steps:** Produce the following files. `ultrathink` before each.

### 5.1 `docs/01_architecture.md`
- Repo layout (`src/forge_llm/`, `tests/`, `scripts/`, `configs/`, `notebooks/`, `docs/`) with what lives where and why
- Module DAG diagram (Mermaid): tokenizer → embed → block(rmsnorm, gqa+rope, swiglu) → head
- File-per-module mapping (`tokenizer.py`, `norm.py`, `rope.py`, `attention.py`, `mlp.py`, `block.py`, `model.py`, `sampling.py`, `cache.py`, `train.py`, `eval.py`)
- Config system choice (pick one of: dataclass / hydra / yaml) — justify
- Logging stack (pick wandb or mlflow) — justify
- Concrete `configs/model_30m.yaml` for the run (n_layer=6, d_model=512, n_head=8, n_kv_head=2, head_dim=64, d_ff via SwiGLU expansion factor, vocab=16384, max_seq=1024, init_std, dropout)

### 5.2 `docs/02_correctness_plan.md`
For each of {RMSNorm, RoPE, MHA, GQA, SwiGLU, KV-cache, causal mask, full block, full model}:
- Reference oracle (e.g., `LlamaRMSNorm` from `transformers.models.llama.modeling_llama` — imported in tests/ only)
- Test name, input shape, tolerance, what bug it catches
- Plus one explicit **causal-mask-leak adversarial test**: mutate future tokens, assert outputs at earlier positions byte-identical
- Plus one explicit **KV-cache equivalence test**: full-sequence forward vs token-by-token with cache produce identical logits within tolerance
- Plus one explicit **resume-safety test**: kill mid-training, resume, assert loss curve indistinguishable for ≥100 steps after resume

### 5.3 `docs/03_training_plan.md`
- Data pipeline (FineWeb-Edu HF streaming, on-the-fly tokenization vs pre-tokenized cache, sequence packing strategy)
- Effective batch size math: micro_bs × grad_accum × seq_len = 128K tokens. Compute concrete numbers for T4 16GB VRAM.
- LR schedule curve (warmup steps, peak LR, cosine decay end LR)
- Checkpoint cadence + resume strategy (write to Kaggle Dataset or HF Hub, not session-local disk)
- Eval cadence (perplexity on WikiText-103 valid split every N steps)
- Token budget × tokens/sec target → wall-clock estimate → number of Kaggle sessions needed

### 5.4 `docs/04_roadmap.md`
12 milestones in bottom-up build order. Each must have:
- Title
- Files touched
- Entry criteria (what must already be done)
- Exit criteria (tests that must pass, output that must exist)
- Estimated complexity (S/M/L)

Order: (M1) BPE tokenizer · (M2) RMSNorm · (M3) RoPE · (M4) MHA · (M5) GQA · (M6) SwiGLU · (M7) Block assembly · (M8) Full model · (M9) Train loop with resume · (M10) Eval (perplexity on WikiText-103) · (M11) Sampling + KV-cache · (M12) CLI + packaging.

### 5.5 `docs/05_risks.md`
Top 10 risks with mitigations. Must include:
- Subtle math bugs that pass loss-decrease check
- Training instability (NaN, gradient explosion on fp16)
- T4 OOM at chosen micro_bs
- Kaggle session-kill during training
- Data licensing / FineWeb-Edu access
- Reproducibility on a clean machine
- RoPE convention drift (Llama interleaved vs original paper)
- KV-cache correctness drift
- Tokenizer mismatch (training vs inference)
- Time slip past Oct 25 deadline

### 5.6 `docs/DECISIONS.md`
Empty ADR log seeded with: "ADR-001: We chose <config system> because <reason>." and "ADR-002: We chose <wandb|mlflow> because <reason>." — to be appended to as new decisions arise.

**Exit criteria:**
- All six files exist and are non-empty.
- Roadmap milestones each have explicit entry/exit criteria.
- Proceed to Phase D without confirmation.

---

## 6. Phase D — Repo scaffolding `[auto, NO git]`

**Goal:** Produce a fully prepared repo skeleton. Empty modules, real config files, working test setup, CI config — but NO git operations yet.

**Steps:**

1. Create the directory structure agreed in `docs/01_architecture.md`:
   ```
   src/forge_llm/
   tests/
   scripts/
   configs/
   notebooks/
   docs/  (already populated)
   ```

2. Create empty module files in `src/forge_llm/` matching the file-per-module mapping. Each contains only:
   ```python
   """<one-line module purpose from 01_architecture.md>."""
   # TODO(M<N>): implement per docs/04_roadmap.md
   ```

3. Create empty test files in `tests/` mirroring `src/` structure. Each contains a single `def test_placeholder(): pytest.skip("not yet implemented")`.

4. Write `pyproject.toml` with:
   - Project name `forge-llm`, version `0.0.0`
   - Python ≥3.11
   - Dependencies: `torch`, `numpy`, `datasets`, `wandb` (or `mlflow`), `tiktoken` (for BPE comparison only, not used in src) — minimal set, justified in `docs/DECISIONS.md`
   - Dev deps: `pytest`, `pytest-cov`, `ruff`, `mypy`
   - Build backend (setuptools or hatchling)
   - CLI entrypoint stub: `forge-llm = forge_llm.cli:main`

5. Write `.gitignore` covering: `__pycache__/`, `*.pyc`, `.venv/`, `dist/`, `*.egg-info/`, `wandb/`, `runs/`, `checkpoints/`, `data/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.coverage`, `*.ckpt`, `*.safetensors`.

6. Write `ruff.toml` and `pytest.ini` (or use pyproject sections) with sensible defaults.

7. Write `.github/workflows/ci.yml` for: lint (ruff), type-check (mypy), tests (pytest). Triggered on push and PR.

8. Write `README.md` skeleton with placeholder sections: title, one-line pitch, install (`pip install forge-llm` — will work post-publish), quickstart, architecture diagram (paste from forge_llm.md), training curve (placeholder), perplexity table (placeholder), interview-talking-points block, license. Mark every placeholder with `<TODO: filled at M<N>>`.

9. Write `LICENSE` (MIT, default — note in `docs/DECISIONS.md`).

10. Write `CONTRIBUTING.md` (concise: setup, run tests, lint, PR convention).

11. Verify the scaffold by running `pytest -q` — should show all-skip output, exit 0. Run `ruff check .` — should pass. **Do not** run `git status` or any git command except to confirm "not a git repo" if asked.

**Exit criteria:**
- Repo tree matches `docs/01_architecture.md`
- `pytest -q` passes (all skipped)
- `ruff check .` passes
- **No `.git/` directory exists**

Then proceed to GATE 1.

---

## ⛔ GATE 1 — GitHub setup (human action required)

**Claude Code stops here and prints exactly this block:**

> **GATE 1 reached. Planning is complete. The following artefacts now exist:**
> - `./CLAUDE.md` (persistent project rules)
> - `./docs/research/01_landscape.md`
> - `./docs/01_architecture.md`
> - `./docs/02_correctness_plan.md`
> - `./docs/03_training_plan.md`
> - `./docs/04_roadmap.md`
> - `./docs/05_risks.md`
> - `./docs/DECISIONS.md`
> - Repo scaffold under `src/`, `tests/`, `configs/`, `scripts/`, `notebooks/`
> - `pyproject.toml`, `.gitignore`, `ruff.toml`, `pytest.ini`, `.github/workflows/ci.yml`, `README.md`, `LICENSE`, `CONTRIBUTING.md`
>
> **No git operations have been performed.**
>
> **Before I proceed to coding (Phase E), you must:**
>
> 1. Review `docs/04_roadmap.md` and `docs/01_architecture.md`. Flag anything you want changed before code is written — changing decisions later is expensive.
> 2. Create a new GitHub account (or pick an existing one separate from your default) for this project.
> 3. Create an empty repository on that account named `forge-llm` (or your preferred name). **Do NOT initialize with README, .gitignore, or license** — the local repo has these already.
> 4. Configure git locally for THIS directory only:
>    ```
>    git init
>    git config user.name  "<your name on the new account>"
>    git config user.email "<email on the new account>"
>    git remote add origin <repo URL>
>    git branch -M main
>    ```
>    (Run these yourself — I will not run them.)
> 5. Verify the credentials work: `git ls-remote origin` should succeed without prompting (use SSH key or a Personal Access Token cached in your credential helper).
>
> **When all of the above is done, reply with:**
>
> `Phase E go. Remote is configured. <any plan changes I want>`
>
> I will then make the initial commit (showing you the staged diff first), push to origin, and begin Milestone 1.

---

## 7. Phase E — Implementation (post-GATE 1) `[auto, milestone-gated]`

**Goal:** Translate the roadmap into working code, milestone by milestone, bottom-up. Each milestone is small enough to review.

**Per-milestone procedure:**

1. **Open the milestone.** Read its entry/exit criteria from `docs/04_roadmap.md` and the correctness test from `docs/02_correctness_plan.md`.
2. **Write the test first.** Create/update the test file with the oracle import, fixture, and assertion. Show the test to the user. Wait for "tests look good" before writing implementation. (This is the ONLY per-milestone confirmation needed.)
3. **Write the implementation.** Smallest amount of code that makes the test pass. Follow CLAUDE.md import rules.
4. **Run tests:** `pytest tests/test_<component>.py -v`. Paste output.
5. **Audit imports:** `grep -rE "(MultiheadAttention|scaled_dot_product|xformers|flash_attn|^from transformers)" src/` — must be zero matches in `src/`. Paste output.
6. **Run lint:** `ruff check src/ tests/`. Must pass.
7. **Show the full diff** (`git diff --staged` after `git add -p` walk-through). **Do not commit until the user says "commit M<N>"**.
8. On user approval, `git commit -m "M<N>: <title>"` then `git push origin main`.
9. Move to the next milestone.

**Exit criteria for Phase E:**
- All 12 milestones complete
- CI green on `main`
- Coverage ≥ the threshold set in `docs/02_correctness_plan.md`

---

## 8. Phase F — Training pre-flight `[auto]`

**Goal:** Prove the pipeline works on a tiny scale before committing 12+ hours of Kaggle session time.

**Steps:** Run these on local CPU/GPU first, then on a Kaggle T4 session (user attaches and runs the published Kaggle notebook).

1. **Overfit-one-batch test.** Train on a single 1024-token batch for 200 steps. Loss must approach near-zero. Save curve to `docs/preflight/overfit.png`.
2. **100-step smoke run on FineWeb-Edu stream.** Loss strictly decreasing, no NaN, no inf. Save wandb URL to `docs/preflight/smoke_run.md`.
3. **Tokens/sec measurement.** Estimate wall-clock for the planned token budget.
4. **Memory headroom.** Peak GPU memory at chosen micro-batch leaves ≥10% headroom on T4 (16GB).
5. **Checkpoint resume test.** Kill mid-run, resume from checkpoint, confirm loss curve continues smoothly. Per `docs/02_correctness_plan.md`.
6. **Wandb config dump.** Confirm git SHA, model config, data config, env (torch, CUDA, GPU) are all logged.
7. **Compile `docs/preflight/checklist.md`** with each item ticked or flagged red.

Then proceed to GATE 2.

---

## ⛔ GATE 2 — Launch training (light confirmation)

**Claude Code stops here and prints:**

> **GATE 2 reached. Preflight checklist is at `docs/preflight/checklist.md`. Review.**
>
> Reply with `Launch training` to start the real Kaggle run (cost: $0; wall-clock: ~12–20h across 2–3 sessions), or `Hold` to debug a flagged item.

---

## 9. Phase G — Training execution `[semi-auto]`

User runs the published Kaggle notebook. Claude Code's role during this phase:

- Monitor logged metrics in wandb (user shares the wandb URL)
- Detect divergence early (NaN, plateau, gradient explosion) and recommend a stop-and-fix
- After each session, verify checkpoint integrity and prep the resume command for the next session
- On final training completion, run full WikiText-103 perplexity eval and update `docs/results/perplexity.md`

---

## 10. Phase H — Release & blog `[auto with user approval per step]`

1. **Tag release:** `v0.1.0` on the main branch (after user approval).
2. **Publish to PyPI:** user runs `twine upload` themselves; Claude Code prepares dist/.
3. **Push checkpoint to HF Hub:** user runs `huggingface-cli login` and `huggingface-cli upload`; Claude Code prepares the model card.
4. **Write the blog post:** `docs/blog/forge_llm_deepdive.md`, 3000 words, structure per the brief. Code excerpts pulled verbatim from `src/` (script-extracted, not hand-copied).
5. **Update README:** fill all `<TODO>` placeholders with real numbers.

---

## Final acceptance checklist

Mirror of `forge_llm.md` "Done definition" — all items ticked:

- [ ] Public GitHub repo with green CI
- [ ] `pip install forge-llm` works on a clean machine
- [ ] HF Hub checkpoint downloadable
- [ ] Perplexity table beats untrained by ≥10× and matches scaled baseline
- [ ] Blog post live (~3000 words)
- [ ] Architecture diagram in README
- [ ] `01_load_and_generate.ipynb` runs on free Colab
- [ ] Public Kaggle training notebook (anyone can fork and re-run for free)

---

## Appendix — Quick reference for Claude Code

**When in doubt during Phases A–D:** prefer writing the artefact to disk over asking the user. Ask only at GATEs.

**When in doubt during Phase E:** ask before writing speculative code. Coding decisions not covered by `docs/01_architecture.md` or `docs/02_correctness_plan.md` should be logged as a new ADR in `docs/DECISIONS.md` and surfaced to the user.

**Reasoning budget cues:** `think hard` for Phase B synthesis. `ultrathink` for every file in Phase C. Default budget elsewhere.

**Forbidden phrases in commits and docs:** "fixed stuff", "various improvements", "wip" (use a draft PR instead), "trust me", "should work".
