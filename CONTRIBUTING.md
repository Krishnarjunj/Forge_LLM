# Contributing to Forge-LLM

Thanks for your interest. Forge-LLM is small and opinionated; PRs are welcome
within the scope below.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest -q -m "not slow"      # fast suite (always)
pytest -q -m "slow"          # oracle + resume-safety (requires HF transformers installed)
pytest --cov=src/forge_llm   # with coverage
```

## Lint and types

```bash
ruff check src/ tests/ scripts/
ruff format src/ tests/ scripts/
mypy src/
```

## Forbidden-import audit

Forge-LLM is from-scratch. The audit gates every PR (see `CLAUDE.md` §2):

```bash
grep -rE "(MultiheadAttention|scaled_dot_product_attention|xformers|flash_attn|^from transformers|^import transformers|apex)" src/
```

This MUST return zero matches. The same check runs in CI.

## PR conventions

- One milestone per PR (see `docs/04_roadmap.md`).
- The test for the milestone lands **before** the implementation, and the user reviews the test before the implementation is written (`BUILD_PLAN.md` §7).
- Commit messages describe the *what* and the *why*; no `"fixed stuff"`, `"various improvements"`, `"wip"`, `"trust me"`, `"should work"`.
- New deps require an ADR-lite entry in `docs/DECISIONS.md` (`BUILD_PLAN.md` §2 hard rule 6).
- New design choices (config system, scheduler curve, license, etc.) also become ADRs.

## Scope

In scope:
- Bug fixes, test additions, doc improvements.
- Performance work that doesn't violate the forbidden-import rules.
- New ADRs proposing alternative design choices (we'll discuss before implementation).

Out of scope (for now):
- Adding more model architectures.
- Adding training distributed-parallel support (DDP/FSDP) — small model, free T4, no need.
- Using `transformers`, `xformers`, `flash_attn` in `src/`. Allowed only as test oracles in `tests/`.
