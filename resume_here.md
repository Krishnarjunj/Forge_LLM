# resume_here.md — Continue Forge-LLM from GATE 1

> Drop this into the next Claude Code session as the first message (or just say "read `resume_here.md` and continue from GATE 1"). Authored 2026-05-26 at the end of Phases A–D.

---

## Where we are

**Status:** GATE 1 reached. Phases A, B, C, D complete. **No code, no git, no commits yet.**

Phases completed (per `BUILD_PLAN.md`):
- ✅ Phase A — `CLAUDE.md` written.
- ✅ Phase B — Landscape research synthesised at `docs/research/01_landscape.md` (3 parallel subagents: cohort review, OSS reference repos, differentiation).
- ✅ Phase C — Six planning docs in `docs/`: `01_architecture.md`, `02_correctness_plan.md`, `03_training_plan.md`, `04_roadmap.md`, `05_risks.md`, `DECISIONS.md`.
- ✅ Phase D — Repo scaffold complete: `src/forge_llm/` (17 placeholder modules), `tests/` (17 placeholder tests + `conftest.py`), `configs/model_30m.yaml`, `configs/train_kaggle_t4.yaml`, `scripts/` placeholders, `notebooks/` placeholders, `pyproject.toml`, `.gitignore`, `ruff.toml`, `pytest.ini`, `.github/workflows/ci.yml`, `README.md`, `LICENSE` (MIT), `CONTRIBUTING.md`.
- ⏸️ **GATE 1 — waiting on you (human action required, see below).**

Next milestone after GATE 1: **M1 — BPE tokenizer** (`docs/04_roadmap.md` M1).

---

## What you need to do before Phase E (the GATE 1 actions)

1. **Review `docs/04_roadmap.md` and `docs/01_architecture.md`.** Flag anything you want changed before code is written — changing decisions later is expensive. Also worth a skim:
   - `docs/02_correctness_plan.md` — the test contract per milestone.
   - `docs/DECISIONS.md` — closed ADRs (001, 002, 007, 011, 012, 013) and open ones (003, 004, 005, 006, 008, 009, 010).
   - `docs/research/01_landscape.md` §"Top 5 differentiation moves" — these drive M9/M10/M11/M12 and Phase H deliverables.

2. **Create a new GitHub account** (or pick an existing one separate from your default) for this project.

3. **Create an empty repo** on that account named `forge-llm` (or your preferred name). **Do NOT initialise with README, .gitignore, or license** — the local repo has these already.

4. **Configure git locally** in `/Users/krishnarjun.j/Krish/Forge_LLM/` (THIS directory only):
   ```bash
   cd /Users/krishnarjun.j/Krish/Forge_LLM
   git init
   git config user.name  "<your name on the new account>"
   git config user.email "<email on the new account>"
   git remote add origin <repo URL>
   git branch -M main
   ```

5. **Verify credentials work**: `git ls-remote origin` should succeed without prompting (use SSH key or a PAT cached in your credential helper).

6. **In the next Claude Code session**, reply with:
   > `Phase E go. Remote is configured. <any plan changes I want>`

   Claude will then show the staged diff for the initial commit, wait for your "go", commit, push to `origin`, and begin **M1 (BPE tokenizer)**.

---

## Repo state snapshot (so you can verify nothing drifted)

Run from `/Users/krishnarjun.j/Krish/Forge_LLM/`:

```bash
# Should show: configs, docs, notebooks, scripts, src, tests + root files.
ls -la

# Should be zero matches.
grep -rE "(MultiheadAttention|scaled_dot_product_attention|xformers|flash_attn|^from transformers|^import transformers|apex)" src/

# Should show "no .git dir" until you run `git init` per the GATE 1 actions above.
test -d .git && echo "GIT DIR EXISTS" || echo "no .git dir — expected before GATE 1 actions"

# Verifications that passed at GATE 1 — re-run before kicking off Phase E:
./.venv/bin/pytest -q             # expected: 15 skipped, exit 0
./.venv/bin/ruff check .          # expected: "All checks passed!"
```

The local `.venv/` was created during Phase D and has only `pytest` and `ruff` installed (for scaffold verification). It is `.gitignore`d. You'll want a real environment with `pip install -e ".[dev]"` for Phase E.

---

## Open ADRs to keep in mind (these will close during Phases E / F)

| ADR | Topic | Resolves in |
|-----|-------|-------------|
| 003 | CLI parser (tyro vs argparse) | M12 |
| 004 | Tokenisation: on-the-fly vs pre-packed `.bin` | Phase F preflight |
| 005 | Final `n_layer` (6 vs 8) — based on T4 memory headroom | Phase F preflight |
| 006 | `torch.compile` default in `generate()` | M11 / Phase F |
| 008 | Final `(micro_bs, grad_accum)` split | Phase F preflight |
| 009 | Total token budget (1B vs 500M) — based on measured tokens/sec | Phase F preflight |
| 010 | Final number of Kaggle sessions allocated | Phase F preflight |

The closed ADRs are decisions you've accepted by reaching GATE 1; if you want to revisit any, do it **before** sending `Phase E go`.

---

## Quick mental model of the project (for fresh-session re-priming)

- **Goal:** ~30M-param Llama-family decoder (RoPE + GQA 8:2 + RMSNorm + SwiGLU + KV-cache) from scratch in PyTorch. Trained on FineWeb-Edu on free Kaggle T4 ($0). Released as `pip install forge-llm` + HF Hub checkpoint + Kaggle notebook + 3000-word blog. Deadline: Oct 25, 2026.
- **Hard rule:** no `transformers`, `xformers`, `flash_attn`, `torch.nn.MultiheadAttention`, `torch.nn.functional.scaled_dot_product_attention` in `src/`. Allowed in `tests/` as oracles only.
- **Defensible headline:** "from-scratch Llama-family 30M model that you can `pip install`, fork on Kaggle, and resume-train end-to-end on a free T4 — for $0." Defensibility comes from the conjunction, not any single piece.
- **12 milestones**, bottom-up: tokenizer → norm → rope → MHA → GQA → SwiGLU → block → model → train+resume → eval → cache+sampling+generate → CLI+packaging. After M12: Phase F preflight, GATE 2, Phase G training, Phase H release.

---

## What Claude should expect when this resumes

1. You'll say "Phase E go. Remote is configured." (plus any plan tweaks you want).
2. Claude will:
   - Re-read `BUILD_PLAN.md`, `CLAUDE.md`, `docs/04_roadmap.md`, this file.
   - Show the staged initial-commit diff and wait for your `go`.
   - Commit + push initial scaffold.
   - Begin **M1 (BPE tokenizer)** per the per-milestone procedure in `BUILD_PLAN.md` §7: write the test first, show it to you, wait for "tests look good", then implement, then run import-audit / ruff / pytest, then show the diff, wait for "commit M1", commit, push, move on.

If anything in `docs/04_roadmap.md` or `docs/01_architecture.md` needs changing, say so in the same `Phase E go ...` message; Claude will revise the docs (logging an ADR) before code starts.
