# 01 — Architecture

> The skeleton Phase E will fill in. Decisions here are binding; changes after Phase C require an ADR in `docs/DECISIONS.md` and Claude Code must surface them to the user before changing code.

---

## 1. Repo layout

```
Forge_LLM/
├── BUILD_PLAN.md              # the runbook (binding)
├── forge_llm.md               # original brief
├── CLAUDE.md                  # persistent context for Claude Code
├── README.md                  # user-facing readme (placeholders until Phase H)
├── LICENSE                    # MIT
├── CONTRIBUTING.md
├── pyproject.toml             # project metadata + deps + entrypoints
├── ruff.toml                  # lint config
├── pytest.ini                 # test config
├── .gitignore
├── .github/workflows/ci.yml   # lint + mypy + pytest on push/PR
├── src/
│   └── forge_llm/
│       ├── __init__.py        # public API: ForgeConfig, ForgeForCausalLM, generate
│       ├── config.py          # ForgeConfig dataclass + PRESETS registry
│       ├── tokenizer.py       # BPE tokenizer (train + encode/decode)
│       ├── norm.py            # RMSNorm
│       ├── rope.py            # precompute_freqs / apply_rotary
│       ├── attention.py       # GQA module (uses rope + cache)
│       ├── mlp.py             # SwiGLU FFN
│       ├── block.py           # TransformerBlock (pre-norm)
│       ├── cache.py           # KVCache (pre-allocated buffers)
│       ├── model.py           # ForgeModel + ForgeForCausalLM
│       ├── sampling.py        # top-p / top-k / temperature / rep-penalty
│       ├── generation.py      # generate() loop using KVCache
│       ├── data.py            # FineWeb-Edu streaming + packing
│       ├── train.py           # training loop with resume
│       ├── eval.py            # perplexity on WikiText-103 + held-out FineWeb
│       ├── hub.py             # save/load checkpoint, HF Hub upload helpers
│       └── cli.py             # `forge-llm` console entrypoint
├── tests/
│   ├── conftest.py            # shared fixtures: seed_all, tiny_config, tmp_ckpt
│   ├── test_tokenizer.py
│   ├── test_norm.py           # vs LlamaRMSNorm
│   ├── test_rope.py           # vs LlamaRotaryEmbedding (rotated cosine identity)
│   ├── test_attention.py      # MHA vs torch.nn.MultiheadAttention; GQA vs LlamaAttention
│   ├── test_mlp.py            # vs LlamaMLP
│   ├── test_block.py          # vs LlamaDecoderLayer
│   ├── test_cache.py          # full-seq vs token-by-token logits equality
│   ├── test_model.py          # forward shape + parameter count + tied weights
│   ├── test_sampling.py       # top-p/k/temperature distributional sanity
│   ├── test_generation.py     # deterministic 100-token gen from fixed seed
│   ├── test_causal_mask.py    # adversarial: mutate token T-1, check t<T-1 byte-identical
│   ├── test_resume.py         # train→checkpoint→kill→resume, compare loss curve
│   └── test_data.py           # streaming reader + packing + iterator step preservation
├── configs/
│   ├── model_30m.yaml         # the run config (locked in §6 below)
│   └── train_kaggle_t4.yaml   # training hyperparameters + batch math for T4
├── scripts/
│   ├── train_bpe.py           # train BPE on FineWeb-Edu subset → tokenizer.json
│   ├── prepare_data.py        # (optional) pre-pack a tokenized shard to .bin
│   ├── eval_perplexity.py     # WikiText-103 eval entrypoint
│   ├── benchmark_kv_cache.py  # tokens/sec with cache off vs on at ctx 128/512/2048
│   └── export_to_hub.py       # push checkpoint + tokenizer to HF Hub
├── notebooks/
│   ├── 01_load_and_generate.ipynb        # Colab demo
│   └── 02_kaggle_train.ipynb             # the public Kaggle training notebook
└── docs/
    ├── research/01_landscape.md
    ├── 01_architecture.md     # this file
    ├── 02_correctness_plan.md
    ├── 03_training_plan.md
    ├── 04_roadmap.md
    ├── 05_risks.md
    ├── DECISIONS.md
    ├── preflight/             # filled in Phase F
    ├── results/               # filled in Phase G
    └── blog/                  # filled in Phase H
```

**Why this layout.** Mirrors the llama2.c class hierarchy (`norm.py`, `rope.py`, `attention.py`, `mlp.py`, `block.py`, `model.py`) at a file-per-module granularity instead of llama2.c's single `model.py` — easier to test in isolation and to read in a code-reading interview. `cache.py` lives next to `attention.py` because they are co-developed (gpt-fast pattern). `generation.py` is separate from `sampling.py` so unit tests for sampling distributions don't need to instantiate a model.

---

## 2. Module DAG

```mermaid
flowchart TD
    Text[Input text] --> Tok[tokenizer.py<br/>BPE encode]
    Tok -->|token IDs| Embed[model.py<br/>nn.Embedding tied]
    Embed -->|"[B,T,D=512]"| Block

    subgraph Block[block.py × 6]
      direction TB
      In["[B,T,D]"] --> Norm1[norm.py<br/>RMSNorm]
      Norm1 --> Attn[attention.py<br/>GQA<br/>8 Q heads / 2 KV heads]
      Rope[rope.py<br/>apply_rotary] -.->|"rotate Q,K"| Attn
      Cache[cache.py<br/>KVCache<br/>pre-allocated"] -.->|"K,V history"| Attn
      Attn --> Add1[(+)]
      In --> Add1
      Add1 --> Norm2[norm.py<br/>RMSNorm]
      Norm2 --> MLP[mlp.py<br/>SwiGLU<br/>3 linears]
      MLP --> Add2[(+)]
      Add1 --> Add2
      Add2 --> Out["[B,T,D]"]
    end

    Block --> FinalNorm[norm.py<br/>RMSNorm]
    FinalNorm --> LMHead[model.py<br/>tied LM head]
    LMHead -->|"[B,T,V=16384]"| Logits[Logits]
    Logits --> Sample[sampling.py<br/>top-p / top-k / temp / rep-penalty]
    Sample --> NextTok[Next token]
```

Key points:
- **Pre-norm.** RMSNorm before attention and before MLP, residual added after the sub-block.
- **Tied embeddings.** Same weight matrix for input embedding and output LM head (saves ~8.4M params at vocab=16384).
- **Causal mask** is applied inside `attention.py`; it is not a separate module. A unit test asserts no future-token leakage (see `docs/02_correctness_plan.md`).
- **KV cache** is owned by `cache.py` and passed *into* `attention.py.forward(...)`. The attention module itself is stateless — caching is opt-in per call. This matches the gpt-fast pattern and keeps training (no cache) and generation (cache) on the same code path.

---

## 3. File-per-module mapping

| File | Purpose | Public surface |
|------|---------|----------------|
| `config.py` | Typed config + presets | `ForgeConfig`, `PRESETS["forge-30m"]` |
| `tokenizer.py` | BPE: train, encode, decode, save/load | `BPETokenizer.train(corpus, vocab_size)`, `.encode(str) -> list[int]`, `.decode(list[int]) -> str` |
| `norm.py` | RMSNorm in fp32 | `class RMSNorm(nn.Module)` |
| `rope.py` | RoPE precompute and apply | `precompute_freqs_cis(head_dim, max_seq, theta) -> Tensor`, `apply_rotary(q, k, freqs_cis) -> (q, k)` |
| `attention.py` | GQA with rope + optional cache | `class GroupedQueryAttention(nn.Module)` |
| `mlp.py` | SwiGLU FFN | `class SwiGLU(nn.Module)` |
| `block.py` | Pre-norm decoder block | `class TransformerBlock(nn.Module)` |
| `cache.py` | Statically-allocated KV cache | `class KVCache(nn.Module)` with `.update(input_pos, k, v) -> (k_full, v_full)` |
| `model.py` | Trunk + LM head | `class ForgeModel(nn.Module)`, `class ForgeForCausalLM(nn.Module)` |
| `sampling.py` | Pure functions | `top_p`, `top_k`, `apply_temperature`, `repetition_penalty` |
| `generation.py` | Generation loop | `generate(model, tokenizer, prompt, ...) -> Iterator[str]` |
| `data.py` | FineWeb-Edu streaming + packing | `class PackedFineWebEdu(IterableDataset)` |
| `train.py` | Training loop + resume | `train(cfg) -> None`, `Trainer` class |
| `eval.py` | Perplexity eval | `eval_perplexity(model, dataset) -> float` |
| `hub.py` | Save/load + Hub I/O | `save_pretrained(model, dir)`, `from_pretrained(dir_or_repo) -> ForgeForCausalLM`, `push_to_hub(repo_id)` |
| `cli.py` | Console entrypoint | `main()` dispatches: `generate`, `train`, `eval`, `bench-cache` |

Public API exposed from `src/forge_llm/__init__.py`:

```python
from forge_llm.config import ForgeConfig, PRESETS
from forge_llm.model import ForgeForCausalLM, ForgeModel
from forge_llm.tokenizer import BPETokenizer
from forge_llm.generation import generate
from forge_llm.hub import from_pretrained, save_pretrained, push_to_hub
__all__ = ["ForgeConfig", "PRESETS", "ForgeForCausalLM", "ForgeModel",
           "BPETokenizer", "generate", "from_pretrained", "save_pretrained", "push_to_hub"]
```

---

## 4. Config system choice — `@dataclass` (ADR-001)

**Decision.** A single `@dataclass(frozen=True)` `ForgeConfig` in `src/forge_llm/config.py`. Driven via:
- **Defaults** baked into the dataclass.
- **Presets registry** `PRESETS: dict[str, dict]` with named variants (`"forge-30m"`, `"forge-30m-base"`).
- **YAML files** in `configs/` loaded into the dataclass at the CLI layer.
- **CLI overrides** through `tyro` (or `argparse` if `tyro` is heavyweight) — `--n_layer=8` overrides the YAML.
- **Disk format** for saved checkpoints: JSON via `dataclasses.asdict()` so HF Hub checkpoints are self-describing.

**Why dataclass over Hydra.** Hydra is excellent for hyperparameter sweeps and grouped configs but heavyweight for a single-model project. Dataclasses are typed, IDE-friendly, no `exec()` hostility (vs nanoGPT/llama2.c `configurator.py`), and trivially serialise to JSON for Hub uploads. Hydra wins when you have many composable groups; we don't.

**Why dataclass over plain YAML.** YAML is untyped — a fat-finger `n_kv_head: "2"` (string) would propagate. The dataclass enforces types at load time.

**Validation rules** baked into `__post_init__`:
- `n_head % n_kv_head == 0` (GQA group size must divide evenly).
- `head_dim * n_head == d_model` (no projection bias from concat).
- `head_dim % 2 == 0` (RoPE needs even head dim for the rotation pairing).
- `max_seq <= 4096` (sanity bound for this build).

---

## 5. Logging stack — wandb (ADR-002)

**Decision.** wandb. (Already justified in `CLAUDE.md` §10 and in `docs/research/01_landscape.md`.)

**Why.** Free-tier streaming charts are better than MLflow's free-tier UX, run-resume by ID is one line (`wandb.init(id=..., resume="must")`), Kaggle integration is well-supported, and the wandb run ID is one of the few resume-state items we already need.

**MLflow rejected** for this project because its free hosted UX on Kaggle notebooks for streaming metrics is weaker.

**No-network fallback.** If `WANDB_API_KEY` is missing on a fresh Kaggle fork, `train.py` logs a `WARNING` and continues with local CSV logging into `runs/<run_id>/metrics.csv`. This is the "silent fallback" exception called out in `CLAUDE.md` §11 — it must log loudly, not silently.

---

## 6. `configs/model_30m.yaml` — the run config

```yaml
# configs/model_30m.yaml
# Locked in Phase C. Changes require an ADR.
name: forge-30m

# Model
n_layer: 6
d_model: 512
n_head: 8
n_kv_head: 2          # GQA 4:1 (Llama-2-7B ratio)
head_dim: 64          # d_model // n_head = 512 // 8
d_ff: 1408            # round_multiple(8/3 * d_model, multiple_of=256)
vocab_size: 16384
max_seq: 1024
rope_theta: 10000.0
norm_eps: 1.0e-5
tie_embeddings: true
init_std: 0.02
dropout: 0.0          # Llama uses 0; we follow

# Activation + norm choices (informational; locked in code)
activation: swiglu
norm: rmsnorm
position: rope
attention: gqa
```

### Parameter budget (verified)

| Component                                 | Count        |
|-------------------------------------------|--------------|
| Token embedding (16384 × 512)             | 8,388,608    |
| 6 × TransformerBlock                      |              |
| &nbsp;&nbsp;RMSNorm × 2 per block (512×2) | 6 × 1,024 = 6,144 |
| &nbsp;&nbsp;Attention W_q (512×512)       | 6 × 262,144 = 1,572,864 |
| &nbsp;&nbsp;Attention W_k (512×128)       | 6 × 65,536  = 393,216 |
| &nbsp;&nbsp;Attention W_v (512×128)       | 6 × 65,536  = 393,216 |
| &nbsp;&nbsp;Attention W_o (512×512)       | 6 × 262,144 = 1,572,864 |
| &nbsp;&nbsp;SwiGLU w_gate (512×1408)      | 6 × 720,896 = 4,325,376 |
| &nbsp;&nbsp;SwiGLU w_up (512×1408)        | 6 × 720,896 = 4,325,376 |
| &nbsp;&nbsp;SwiGLU w_down (1408×512)      | 6 × 720,896 = 4,325,376 |
| Final RMSNorm                             | 512          |
| LM head                                   | tied → 0     |
| **Total**                                 | **~25.3M**   |

**Honesty about "30M".** The label `model_30m` is aspirational. At the BUILD_PLAN's exact spec (n_layer=6, d_model=512, n_head=8, head_dim=64) the model lands at ~25.3M. To hit ≥30M while preserving the spec, we can either (a) bump SwiGLU rounding to `multiple_of=512` → d_ff=1536 → ~26.9M, or (b) add 2 layers (n_layer=8) → ~31.2M. **Phase F (preflight) will pick** based on T4 16GB memory headroom at micro_bs=8. If micro_bs=8 fits with ≥10% headroom at n_layer=6, we'll bump to n_layer=8. This is an explicit Phase F decision and will become ADR-005.

A no-cache configuration `forge-30m-base` is registered in `PRESETS` alongside the run config, used for KV-cache benchmark comparisons (differentiation move #2).

---

## 7. Cross-cutting design choices

### Initialisation

- All `nn.Linear` weights: `nn.init.normal_(mean=0.0, std=0.02)`.
- All biases: zeros. **Most linears are bias-free** per Llama convention; only the LM head's tied-embed has the embedding bias-free too.
- `nn.Embedding`: `nn.init.normal_(mean=0.0, std=0.02)`.
- RMSNorm weight (γ): `nn.init.ones_()`.
- Output projection scaling: scale residual paths' second-linear by `1 / sqrt(2 * n_layer)` per GPT-2 init recipe (catches the variance-blowup-with-depth bug).

### Bias usage

- `nn.Linear(bias=False)` everywhere in attention and MLP (Llama convention).
- `RMSNorm` has only a learnable scale γ — no bias term (this is *why* RMSNorm vs LayerNorm).

### Embedding tying

`ForgeForCausalLM` reuses the input embedding weight as the LM head:

```python
self.lm_head.weight = self.embed_tokens.weight
```

This is asserted in `test_model.py::test_embeddings_are_tied`.

### Mask construction

Causal mask is built **once** per max_seq at model construction and registered as a `register_buffer("causal_mask", ...)` so it moves with `.to(device)`. Same mask is sliced for shorter sequences. This avoids per-step mask allocation and is tested for byte-identical-with-the-vanilla-construction in `test_causal_mask.py`.

### Determinism

`seed_all(seed)` per `CLAUDE.md` §5 is called in:
- Test fixtures (`conftest.py`).
- `train.py:main()` before model construction.
- `generation.py:generate()` when a `seed=` kwarg is passed.

### Forbidden imports

The audit grep in `CLAUDE.md` §2 runs in CI on every push. Any match in `src/` fails CI.

---

## 8. Open questions deferred to ADRs

- **ADR-003 (open):** Which CLI parser? `tyro` vs `argparse`. Pending: try both during M12.
- **ADR-004 (open):** Pre-pack vs on-the-fly tokenisation. Pending Phase F tokens/sec measurement — if tokenisation is the bottleneck on T4, switch to pre-packed `.bin` files (nanoGPT pattern) and pay a one-time prep cost.
- **ADR-005 (open):** Final `n_layer` (6 or 8) — pending Phase F memory headroom measurement.
- **ADR-006 (open):** `torch.compile` toggle default in `generate()`. Pending Phase F + M11 benchmarks.

Each will be closed in `docs/DECISIONS.md` when its resolution criterion is hit.
