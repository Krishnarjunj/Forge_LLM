# 03 — Training Plan

> The plan to train Forge-LLM on free Kaggle T4 GPU in 2–4 sessions, end-to-end resumable. Numbers below are baselined for the run config in `docs/01_architecture.md` §6. Phase F (preflight) will tighten the micro-batch and tokens/sec estimates against measured T4 numbers; any change ≥20% becomes an ADR.

---

## 1. Data pipeline

### 1.1 Source — FineWeb-Edu (HF streaming)

- Dataset: `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT` (10B-token curated sample).
- Access: `datasets.load_dataset(..., streaming=True)`. Streaming is mandatory — the full dataset is multi-TB and we don't have session-disk space.
- Filtering: take only the `educational_value >= 3` rows (already pre-filtered in `sample-10BT`).
- Target consumption for this build: **~1B tokens** out of the 10B sample.

### 1.2 Tokenization strategy — on-the-fly with optional pre-pack cache

- **Default path (M9):** on-the-fly BPE encoding in the dataloader worker. Pros: zero disk usage, can fork the Kaggle notebook and `Run All` with no setup. Cons: tokenization cost per worker; if it bottlenecks the GPU, we switch.
- **Fallback path (ADR-004, pending Phase F):** pre-tokenize a shard to a memmap `.bin` file (nanoGPT pattern) saved to a Kaggle Dataset, mount as a Kaggle Notebook input. Switch if Phase F shows GPU util < 70% with tokenization on the critical path.

### 1.3 Sequence packing

- Strategy: concat-and-chunk. Tokenize each doc, append the BPE-trained `<eos>` token, accumulate into a buffer, emit fixed-size chunks of `max_seq=1024`.
- **EOS between docs only.** Not inserted mid-doc. Asserted by `tests/test_data.py::test_packed_dataset_no_eos_at_arbitrary_position_leak`.
- Loss masking: the position immediately *after* an EOS is included in the loss (we want the model to learn to start new docs). No special "doc boundary" masking — this is intentional and matches Llama's training recipe.

### 1.4 Iterator state for resume

The `PackedFineWebEdu` iterable holds three integers:

```python
state = {
    "shard_id": int,        # which FineWeb-Edu file we're streaming from
    "shard_offset": int,    # how many docs we've consumed in this shard
    "global_step": int,     # how many sequences we've emitted total
}
```

`save_state()` returns the dict; `load_state(state)` resumes from there. Used by `Trainer.save_checkpoint()`. The data-iterator-step test in `docs/02_correctness_plan.md` §1.14 covers this.

---

## 2. Effective batch math

| Variable | Value | Notes |
|----------|-------|-------|
| Target effective batch size | **128K tokens / optimizer step** | Llama-2-7B convention scaled-down; standard for ~30M-param models |
| Sequence length | 1024 | from `configs/model_30m.yaml` |
| Micro-batch (per fwd/bwd) | **8** | baseline; Phase F may bump to 16 if headroom exists |
| Tokens per micro-batch | 8 × 1024 = 8,192 | |
| Gradient accumulation | **16** | to hit 128K tokens |
| Tokens per optimizer step | 8,192 × 16 = **131,072** | ≈ 128K target |
| Effective batch size (sequences) | 8 × 16 = 128 sequences | |

### 2.1 Memory headroom on T4 (16GB) — estimate

Rough back-of-envelope at micro_bs=8, seq=1024, fp16 mixed precision, ~25M params:

| Component | Bytes (approx) |
|-----------|---------------:|
| Model weights (fp32 master + fp16 compute) | 25M × 6 = 150 MB |
| Optimizer state (AdamW m, v in fp32) | 25M × 8 = 200 MB |
| Gradients (fp16) | 25M × 2 = 50 MB |
| Activations (fp16, peak) — `n_layer × micro_bs × seq × d_model × ~12` | 6 × 8 × 1024 × 512 × 12 = 302 MB |
| KV temporaries during attention | ~100 MB |
| Misc (cuDNN workspace, allocator overhead) | ~1 GB |
| **Total estimate** | **~1.8 GB** |

T4 has 16 GB. **Headroom: very large.** The estimate above is loose; the dominant variable is activation memory at long sequence. Phase F measures the real number with `torch.cuda.max_memory_allocated()` — if peak < 14 GB at micro_bs=8, we bump to micro_bs=16 and drop grad_accum to 8 (same effective batch, fewer kernel launches, higher tokens/sec).

### 2.2 If micro_bs=8 doesn't fit

Fallback ladder (Phase F decides at the first option that fits with ≥10% headroom):
1. micro_bs=8, grad_accum=16 (the baseline)
2. micro_bs=4, grad_accum=32 (half the per-step memory)
3. micro_bs=2, grad_accum=64 (cuts memory ~4×, doubles overhead)
4. micro_bs=1, grad_accum=128 (last resort)
5. Activation checkpointing on every block (memory roughly halves, ~25% speed cost) — **only** if step 4 still OOMs

We will not silently change this; whatever Phase F picks becomes ADR-008.

---

## 3. LR schedule

| Phase | Steps | LR |
|-------|-------|----|
| Warmup (linear) | 0 → 1,000 | 0 → 3e-4 |
| Cosine decay | 1,000 → 7,800 | 3e-4 → 3e-5 |
| Final | ≥ 7,800 | 3e-5 (sustained, in case we extend) |

- **Peak LR 3e-4.** Lower than GPT-2's 6e-4 because the model is smaller; cohort precedent (CS23S011) and gpt-fast Llama recipes use 3e-4 at this scale.
- **Decay to 3e-5** (10× drop) — standard cosine final.
- **Warmup is 1000 steps**, ~13% of total. Conservative end of the typical 1–10% range; helps stability on fp16 T4.
- Implemented as a single `torch.optim.lr_scheduler.LambdaLR` whose state dict is included in the resume checkpoint.

---

## 4. Optimizer

- **AdamW** (`torch.optim.AdamW`).
- `betas=(0.9, 0.95)` — Llama convention; β₂=0.95 stabilises late training vs Adam default 0.999.
- `eps=1e-8`.
- `weight_decay=0.1` applied to all `nn.Linear` weights and `nn.Embedding`; **not** applied to RMSNorm γ or to biases (there are none — but we filter anyway in `configure_optimizers()`).
- The decay/no-decay parameter split is taken from nanoGPT's `configure_optimizers()`; the test `test_configure_optimizers_param_groups` in `tests/test_train.py` asserts the split.

---

## 5. Mixed precision

- **fp16 autocast** via `torch.autocast(device_type="cuda", dtype=torch.float16)` (T4 has no native bf16).
- **`torch.cuda.amp.GradScaler`** with initial scale `2**16`, `growth_interval=2000`, `growth_factor=2.0`, `backoff_factor=0.5`.
- On `inf`/`NaN` gradient: skip the optimizer step (`scaler.step()` returns `None`), do NOT zero the unscaled gradients (let GradScaler retry on next step).
- **fp32 softmax** in attention (per `CLAUDE.md` §6).
- **fp32 norm**: RMSNorm computes in fp32 and returns in input dtype (per `CLAUDE.md` §6).
- All NaN/Inf checks log to wandb as `train/nan_skip_count`. If `nan_skip_count > 5%` of steps in any 1000-step window, training auto-halts with a fatal error — drop LR by 0.5× and resume.

---

## 6. Checkpoint cadence + resume strategy

- **Cadence**: every **500 steps OR 30 minutes**, whichever first.
- **What's saved**: see `CLAUDE.md` §8 — the full 9-key dict (`step`, `model`, `optimizer`, `scheduler`, `scaler`, `rng`, `data_iterator`, `wandb_run_id`, `config_hash`, `git_sha`).
- **Where it's saved**:
  - Primary: **HuggingFace Hub** `<user>/forge-llm-ckpts` repo, branch `main`, file `step_<N>.pt`.
  - Mirror (per-session, fast access): **Kaggle Dataset** `<user>/forge-llm-resume` updated at every checkpoint via `kaggle datasets version`. This gets pulled fast on a new session.
  - **Never** Kaggle session disk (`/kaggle/working`) alone — wiped on session end.
- **What's kept**:
  - Last 3 checkpoints kept on Hub (rolling).
  - Best-val-perplexity checkpoint kept indefinitely (separate file `best.pt`).
- **Resume contract**:
  - Loader reads the latest checkpoint from Hub or Kaggle Dataset (Kaggle Dataset preferred if mounted).
  - Verifies `config_hash` matches; mismatch → refuse unless `--force`.
  - If `git_sha` differs from current `HEAD`, log a `WARNING` (do not refuse — code can change between sessions; the user accepts the drift).
  - Restores all 9 keys.
  - Calls `wandb.init(id=wandb_run_id, resume="must")` so the same wandb run gets new metrics appended.
- **Recovery if Hub upload fails mid-checkpoint**: the upload is atomic (write to `step_<N>.tmp`, then rename). If the rename never happens, the next checkpoint cycle will retry. Crashed-during-upload state is safe.

---

## 7. Eval cadence

- **WikiText-103 valid perplexity** every **1000 steps**. (Subset: first 256 sequences of `wikitext-103-raw-v1` valid split, ~256K tokens — enough signal, cheap.)
- **Held-out FineWeb-Edu slice perplexity** every 1000 steps. (Same 256K tokens, frozen at training start.)
- **Sample generations** every 1000 steps from 5 fixed prompts (logged to wandb as text tables).
- **Full WikiText-103 valid perplexity** at end of every session (not every 1000 steps — too expensive).
- All eval runs are fp32 (no autocast), `model.eval()`, `torch.no_grad()`. The `eval()` mode flips dropout off (we use dropout=0 anyway, but the discipline matters).

---

## 8. Token budget + wall-clock + sessions

| Variable | Value |
|----------|-------|
| Total tokens | **1B** (= 1,000,000,000) |
| Tokens per optimizer step | 131,072 |
| Total optimizer steps | **~7,630** |
| Target tokens/sec on T4 (post-warmup) | **6,000** |
| Total seconds (training-only) | 1B / 6,000 = **166,667 s** ≈ **46.3 hours** |
| Per-session ceiling (Kaggle) | 12 h |
| Sessions required | **~4** (~46 / 12 with overhead) |
| Weekly Kaggle GPU quota | 30 h |
| Calendar weeks for training | **~2 weeks** (4 sessions / week 1 of 30h burn ≈ 24h, week 2 ≈ 22h) |

**Tokens/sec sanity check.** llama2.c reports ~3K tokens/sec on a single A100 for a similar-size model; T4 has ~1/4 the FP16 throughput of A100. A naive scaling would predict ~750 tokens/sec on T4. Our 6,000 target is **8× higher** than naive scaling — this gap is closed because (a) Llama2.c targets MHA, we use GQA which reduces KV memory traffic, (b) we use grad-accumulation rather than huge per-step batch, and (c) at 30M params the model is small enough that the T4 is largely memory-bandwidth-bound rather than compute-bound, where the throughput-vs-FLOPS ratio is more favourable. **If Phase F measures < 3,000 tokens/sec, the token budget drops to 500M and we rerun the math.** This becomes ADR-009.

---

## 9. Eval harness

For the final perplexity table in the README:

| Eval | Dataset | Scoring | Notes |
|------|---------|---------|-------|
| WikiText-103 valid PPL | `wikitext-103-raw-v1` valid | byte-level PPL, full split | Differentiation move #3 baseline |
| Untrained baseline PPL | same | byte-level PPL, init-only | "beats untrained by ≥10×" target |
| nanoGPT-124M PPL | same | reference value from Karpathy / re-run | Per-param efficiency curve |
| Held-out FineWeb-Edu PPL | 0.1% of FineWeb-Edu held out at training start | byte-level PPL | In-distribution generalisation |
| LM-eval-harness | hellaswag, winogrande, arc-easy | accuracy | Stretch — only if quota allows in Phase G |

The "per-parameter efficiency chart" of differentiation move #3 plots (params on x-axis, PPL on y-axis, log-log) with three points: Forge-LLM-30M, nanoGPT-124M (re-run), and GPT-2-small as cited (no re-run).

---

## 10. Free-tier operational discipline

- **Notebook is the artefact.** The published Kaggle notebook (differentiation move #1) is the canonical training entry point. Local `python -m forge_llm.train` works but is not the supported path for full-budget runs.
- **Wandb API key** is read from Kaggle Secrets (`WANDB_API_KEY`). If absent, the notebook prints a banner explaining how to set it and falls back to CSV logging.
- **HF token** for checkpoint upload read from Kaggle Secrets (`HF_TOKEN`). If absent, the notebook prints a banner; checkpoints write to the Kaggle Dataset mirror only.
- **Session monitor**. A background thread in `train.py` watches wall-clock against an 11-hour budget (Kaggle's 12h cap minus 1h cushion). At 11h elapsed, it triggers a final checkpoint and exits cleanly. This avoids losing the last 30 min of progress to a hard kill.
- **Kaggle interactive mode** while developing the notebook; **Kaggle batch (committed) mode** for the actual training runs — gives the full 12h budget.

---

## 11. Open items

- **ADR-004** — pre-pack vs on-the-fly tokenization. Resolves in Phase F.
- **ADR-008** — final micro_bs / grad_accum split. Resolves in Phase F.
- **ADR-009** — token budget (1B vs 500M) if measured tokens/sec falls short. Resolves in Phase F.
- **ADR-010** — final number of Kaggle sessions allocated (3 vs 4 vs 5). Resolves in Phase F.
