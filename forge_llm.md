# P1 — Forge-LLM : GPT-2-mini from scratch in PyTorch

**Role on resumes:** Depth signal. Appears on **R1** (Systems & Big-Tech SWE) and **R2** (Applied GenAI) as the 4th project when space allows. Demoable in-interview even if not on resume v1.

**Timebox:** 4 weeks. Calendar slot: **Sep 28 – Oct 25, 2026** (post-Sep-1 resume freeze; lands on resume v3 ahead of Nov interviews).

**Compute budget:** **$0 — runs entirely on Kaggle's free T4 GPU tier (30h/week quota).**

**Reference winners this clones:**
- **CS23S011** (CS M.S., 9.0 → Intel India): *"Qwen3-mini: Implemented a 'mini' version of the Qwen3 architecture from scratch using PyTorch. Implemented all the key architectural components: Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE)..."*
- **ME22B032** (Mech, 7.5 → Niva Bupa GenAI): *"GPT-2 transformer also built from scratch in PyTorch for foundational understanding."*
- **AE22B022** (Aerospace, 7.26 → NxtWave): *"Engineered a feedforward neural network from scratch in C++, manually implementing all core components including forward/backpropagation, activation functions, and stochastic gradient descent."*

## The problem

Every applied-AI candidate has called the OpenAI API. Almost none can explain attention. Build a fully-working decoder-only transformer (~30M params) from scratch in PyTorch: write the attention kernel, train it on real data using free compute, beat published baselines at scale, and explain every component on a whiteboard.

## Scope — what must exist

### Architecture (from scratch — no `transformers`, no `xformers`)
- **Tokenizer:** Byte-Pair Encoding (BPE) trained on your own corpus subset; ~16K vocab (scaled down so embedding doesn't dominate the ~30M param budget).
- **Embedding layer** + tied output projection.
- **Rotary Position Embeddings (RoPE)** — implemented from the paper, not copied.
- **Multi-head attention** with **Grouped-Query Attention (GQA)**: 8 query heads, 2 KV heads (4:1 ratio, same as Llama-2-7B).
- **RMSNorm** (pre-norm, like Llama).
- **SwiGLU** feedforward (not vanilla GELU MLP).
- **KV-cache** for inference (verify it actually accelerates: measure tokens/sec with cache on vs off).
- **Causal mask** done right (no information leakage — write a unit test).
- 6 layers, hidden 512, head dim 64 → ~30M params target.

### Training
- Data: **FineWeb-Edu** subset (~500M–1B tokens, streamed). Stream from HuggingFace datasets API.
- Compute: **free Kaggle T4 GPU** (30h/week quota) — $0 cost; training split across 2–3 sessions with checkpoint resume.
- Optimizer: **AdamW**, cosine LR schedule with warmup, weight decay 0.1.
- Mixed precision: `fp16` with autocast on T4 (T4 lacks native bf16; document the tradeoff and any loss-scaling needed).
- Gradient accumulation to hit effective batch size **128K tokens** (scaled for T4's 16GB VRAM).
- **Resumable training is mandatory** (Kaggle sessions cap at 12h): checkpoint every 500 steps OR every 30 min, whichever first — saves model weights, optimizer state, scheduler state, RNG states (torch/numpy/python), data iterator step, wandb run ID. Checkpoints written to Kaggle Dataset / HF Hub (not session-local disk).
- Wandb / MLflow logged metrics: loss, perplexity, tokens/sec, GPU util, gradient norms.

### Inference
- Sampling: top-p, top-k, temperature, repetition penalty — all implemented yourself.
- Streaming output (yield tokens as generated).
- Small CLI: `forge-llm generate "Once upon a time"` works on CPU.

### Validation — the bar that makes it PEAK
- **Beats untrained baseline** by ≥10× on held-out test set (perplexity).
- Reproduces scaled-down baseline (~30M nanoGPT-equivalent) perplexity within reasonable margin on WikiText-103; documents the scaling-law gap to GPT-2-small (124M) honestly with a per-param efficiency chart.
- Unit tests for: causal mask, RoPE rotation, attention numerical stability, KV-cache equivalence, **resume-safety** (resume produces loss curve indistinguishable from uninterrupted run for >100 steps).
- One end-to-end test that loads a checkpoint and generates 100 tokens deterministically.

### Documentation
- **3000-word blog post** breaking down each component with code excerpts and diagrams (target: HN front page, r/MachineLearning).
- README has the training curve, the perplexity table vs untrained + scaled baseline, and a "what I learned" section.
- Architecture diagram (Mermaid or hand-drawn) in README.
- Explicit "trained for $0 on Kaggle T4" framing in README — resourcefulness signal recruiters care about.

### Optional stretches (only if ahead of schedule)
- **Speculative decoding** for inference (uses a tiny draft model).
- **FlashAttention-2** reimplementation in Triton — extreme stretch.
- Fine-tune on a custom small dataset (e.g., Indian-English style transfer) to show downstream adaptation.
- **Scale-up rerun** on a paid A100 once free version validates — converts into a "before/after" blog section showing scaling-law obeyed; can be deferred indefinitely without blocking the resume bullet.

## "Done" definition

- [ ] Public GitHub repo with green CI (unit tests + lint).
- [ ] Pip-installable via `pip install forge-llm` (PyPI).
- [ ] Released checkpoint downloadable from HuggingFace Hub.
- [ ] Perplexity table in README beats untrained baseline ≥10× and matches scaled-down baseline within reasonable margin.
- [ ] Blog post live with code excerpts + training curve + design rationale + free-tier story.
- [ ] Architecture diagram in README.
- [ ] One demo notebook: `01_load_and_generate.ipynb` runs end-to-end on free Colab.
- [ ] Kaggle training notebook published publicly — anyone can fork and re-run the training for free.

## Stack

`Python · PyTorch · fp16 · HuggingFace datasets · Kaggle Notebooks (free T4) · MLflow/Wandb · PyPI · HuggingFace Hub`

## Resume bullets (winner-style, fill in actuals during build)

### Variant A — Systems/depth framing (R1)
- Implemented a 30M-parameter decoder-only transformer **from scratch in PyTorch** — RoPE, Grouped-Query Attention (8Q/2KV), RMSNorm, SwiGLU, KV-cache — without using `transformers` or `xformers`; reproduces scaled-baseline perplexity on WikiText-103 within `<X%>`.
- Trained on `<1B>` FineWeb-Edu tokens using **free Kaggle T4 GPUs at $0 compute cost** across `<3>` sessions with resumable training (model + optimizer + RNG + data-iterator state); released as `pip install forge-llm` with HuggingFace checkpoint and `<3000-word>` architecture deep-dive blog post.

### Variant B — AI/foundational-understanding framing (R2)
- Built **Forge-LLM**: a 30M-param GPT-2-style transformer from scratch in PyTorch, implementing GQA, RoPE, RMSNorm, and SwiGLU; trained end-to-end on FineWeb-Edu **at $0 compute cost** using free Kaggle T4 quota.
- Released as a PyPI package with HuggingFace checkpoint, KV-cache-accelerated inference (`<Z>` tokens/sec), session-kill-resilient resumable training, and a 3000-word component-by-component blog deep-dive.

## Architecture diagram (for README)

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
                │    ┌─ RMSNorm ─▶ MHA (8Q/2KV-GQA │
                │    │              + RoPE) ──────┐│
                │    │                            ▼│
                │    └──────────────────────────▶ +│
                │    ┌─ RMSNorm ─▶ SwiGLU MLP ───┐ │
                │    │                            ▼│
                │    └──────────────────────────▶ +│
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
                   └────────────┬──────────────┘
                                ▼
                            "Once upon a time
                             there was a..."
```

## What makes it PEAK

Six things stacked:
1. **From-scratch implementation** of every layer including RoPE and GQA — depth signal no API-caller can fake.
2. **Real training run** on real data — not a 1-epoch toy.
3. **$0 compute cost** via free Kaggle tier — resourcefulness signal recruiters explicitly value.
4. **Resumable training** engineered around free-tier session limits — ships as a real systems-engineering story (same shape as production pre-emption handling).
5. **PyPI + HF Hub release** — "deployable artifact" cross-cutting differentiator (`03_patterns.md` finding #1).
6. **Blog post with architecture deconstruction** — interview talking points pre-written.

Combined, this is the project that lets you answer "what's the most technically deep thing you've worked on?" with a 60-second answer that lands.

## Interview talking points (write these into the README)

- Why GQA over MHA — KV-cache memory savings, quality tradeoff.
- Why RoPE over learned positional embeddings — length extrapolation, relative position bias.
- Why RMSNorm over LayerNorm — fewer parameters, similar performance, used in Llama.
- Why SwiGLU over GELU MLP — gating mechanism, empirical perplexity win.
- Numerical-stability gotchas: log-sum-exp in attention softmax, fp32 accumulation in attention, fp16 loss-scaling on T4.
- The unit test that caught the causal-mask bug (write a real one).
- **Why resumable training matters in production**: free-tier sessions die at 12h; production cluster pre-emptions look the same. Same engineering rigor, free to demonstrate.
- **The scaling-law honest take**: 30M can't match 124M GPT-2 absolute perplexity, but per-param efficiency curves should match — show the chart, own the gap.
