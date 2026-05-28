# Forge-LLM

A from-scratch ~30M-parameter Llama-family decoder, trained on free Kaggle T4
for $0 compute.

> **Status:** planning + scaffolding complete (Phases A–D). Code lands milestone
> by milestone in Phase E — see [`docs/04_roadmap.md`](docs/04_roadmap.md).

## One-line pitch

A ~30M-parameter decoder-only transformer with RoPE, Grouped-Query Attention
(8Q/2KV), RMSNorm, SwiGLU, and a statically-allocated KV cache — implemented
from scratch in PyTorch with **no `transformers`, no `xformers`, no fused
attention** — trained on FineWeb-Edu using free Kaggle T4 GPU quota and
released as `pip install forge-llm`.

## Install

```bash
pip install forge-llm   # <TODO: filled at Phase H after PyPI publish>
```

For local development:

```bash
git clone <repo-url> && cd forge-llm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q -m "not slow"
```

## Quickstart

Python API:

```python
from forge_llm import generate
from forge_llm.hub import from_pretrained
from forge_llm.tokenizer import BPETokenizer

tokenizer = BPETokenizer.load("path/to/checkpoint/tokenizer.json")
trainer = from_pretrained("path/to/checkpoint", data_factory=lambda: iter([]))

for piece in generate(
    trainer.model, tokenizer, "Once upon a time",
    max_new=50, seed=0,
):
    print(piece, end="", flush=True)
```

Console:

```bash
forge-llm generate "Once upon a time" \
    --checkpoint path/to/checkpoint \
    --max-new 50 --seed 0

# Other subcommands (each forwards to its module main):
forge-llm train --config configs/model_30m.yaml --steps 100 --no-wandb
forge-llm eval  --checkpoint path/to/checkpoint --dataset wikitext-103
forge-llm bench-cache --ctx-lengths 128 512 2048
```

## Architecture

```
                   ┌─────────────────────────┐
   "Once upon a   │  BPE Tokenizer (16K vocab)│
    time"   ─────▶│  (trained from scratch)   │
                   └────────────┬──────────────┘
                                │ token IDs
                                ▼
                   ┌─────────────────────────┐
                   │   Token Embedding (tied) │
                   └────────────┬──────────────┘
                                │ [B, T, 512]
                                ▼
                ┌──────────────────────────────────┐
                │  6× Transformer Block:            │
                │    ┌─ RMSNorm ─▶ MHA (8Q/2KV-GQA  │
                │    │              + RoPE) ──────┐ │
                │    │                            ▼ │
                │    └──────────────────────────▶ + │
                │    ┌─ RMSNorm ─▶ SwiGLU MLP ───┐  │
                │    │                            ▼ │
                │    └──────────────────────────▶ + │
                └────────────────┬──────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────┐
                   │  RMSNorm + Tied LM Head   │
                   └────────────┬──────────────┘
                                │ logits [B, T, 16K]
                                ▼
                   ┌──────────────────────────┐
                   │  Sampling (top-p / top-k) │
                   │  with KV-cache            │
                   └──────────────────────────┘
```

Full module DAG (Mermaid) in [`docs/01_architecture.md`](docs/01_architecture.md).

## Why Forge-LLM

| Choice                  | Forge-LLM                              | GPT-2 / nanoGPT       |
|-------------------------|----------------------------------------|----------------------|
| Positional embedding    | RoPE (HF half-split, ADR-007)          | learned absolute     |
| Attention               | Grouped-Query Attention (8Q / 2KV)     | vanilla MHA          |
| Normalisation           | RMSNorm (pre-norm)                     | LayerNorm            |
| Feedforward             | SwiGLU                                 | GELU MLP             |
| KV cache                | statically pre-allocated (gpt-fast)    | absent / on-the-fly  |
| Resume safety           | full state checkpoint, validated test  | best-val-loss only   |
| Compute used            | free Kaggle T4 ($0)                    | A100s / personal GPU |

## Training

Trained on **FineWeb-Edu** (~1B tokens) using **free Kaggle T4 GPU quota**
(30h/week, $0 cost). Resumable across hard 12-hour session caps via a 9-key
checkpoint (model + optimizer + scheduler + scaler + RNG × 4 + data iterator
+ wandb run id). Full plan in [`docs/03_training_plan.md`](docs/03_training_plan.md).

Training curve: `<TODO: filled at Phase G>`

## Perplexity

| Model           | Params | WikiText-103 valid PPL |
|-----------------|--------|------------------------|
| Forge-LLM       | ~30M   | `<TODO: filled at Phase G>` |
| Untrained init  | ~30M   | `<TODO: filled at Phase G>` |
| nanoGPT (re-run)| 124M   | `<TODO: filled at Phase G>` |
| GPT-2-small     | 124M   | `<TODO: cited>`            |

Per-parameter efficiency chart: `<TODO: filled at Phase G>`

KV-cache speedup (tokens/sec, batch=1):

| Context | Cache off | Cache on |
|---------|-----------|----------|
| 128     | `<TODO>`  | `<TODO>` |
| 512     | `<TODO>`  | `<TODO>` |
| 2048    | `<TODO>`  | `<TODO>` |

## Interview talking points

- Why GQA over MHA — KV-cache memory savings, quality tradeoff.
- Why RoPE over learned positional embeddings — length extrapolation, relative position bias.
- Why RMSNorm over LayerNorm — fewer parameters, similar perf, Llama convention.
- Why SwiGLU over GELU MLP — gating mechanism, empirical perplexity win.
- Numerical-stability gotchas: fp32 softmax in attention; fp16 loss-scaling on T4 (no native bf16); fp32 norm compute.
- The causal-mask adversarial test that defends against silent leak bugs.
- **Why resumable training matters in production**: free-tier session kill ≈ production cluster pre-emption. Same engineering rigour, free to demonstrate.
- **Scaling-law honesty**: 30M doesn't match 124M absolute PPL, but per-param efficiency should — chart below.

## Reproducing the training

A public Kaggle notebook (Phase G) reproduces training end-to-end across two
T4 sessions from a fresh fork — no private credentials required (Wandb and HF
tokens read from Kaggle Secrets; absent secrets fall back loudly to CSV /
local-only). Link: `<TODO: filled at Phase G>`

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — persistent project rules for collaboration
- [`docs/01_architecture.md`](docs/01_architecture.md) — module DAG, configs, design choices
- [`docs/02_correctness_plan.md`](docs/02_correctness_plan.md) — test contract
- [`docs/03_training_plan.md`](docs/03_training_plan.md) — data, batch math, schedule, sessions
- [`docs/04_roadmap.md`](docs/04_roadmap.md) — 12 milestones (M1–M12) bottom-up
- [`docs/05_risks.md`](docs/05_risks.md) — top-10 risks and mitigations
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — ADR log
- [`docs/research/01_landscape.md`](docs/research/01_landscape.md) — cohort + OSS comparator analysis

Blog post (3000-word architecture deep-dive): `<TODO: filled at Phase H>`

## License

[MIT](LICENSE).
