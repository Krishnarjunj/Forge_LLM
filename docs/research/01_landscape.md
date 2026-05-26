# 01 — Landscape: Forge-LLM Against the Field

> Phase B output. Synthesizes Subagents B1 (cohort review), B2 (reference repos), B3 (differentiation). Inputs the Phase C planning decisions in `docs/01_architecture.md` and `docs/04_roadmap.md`.

---

## Executive summary

Forge-LLM occupies a deliberately narrow niche: a Llama-family decoder (RoPE + GQA 8:2 + RMSNorm + SwiGLU) at the smallest scale where the architectural choices still matter, trained end-to-end on real data using free Kaggle T4 sessions, and released as a usable Python package. The combination is rare in the cohort and unmatched by the canonical from-scratch OSS reference (nanoGPT, GPT-2 stack). The single closest peer is **CS23S011 (Qwen3-mini)**, who has the same architecture spec and a partial training run but no shipped artefact. Forge-LLM differentiates by **closing the artefact loop**: PyPI package, HuggingFace Hub checkpoint, public Kaggle notebook, blog post, and engineered resume-safety across Kaggle's 12-hour session kill.

---

## Part 1 — Cohort review (B1)

### Focus projects

**CS23S011 — Qwen3-mini from scratch in PyTorch.** Sri Saravanan R (MS by Research, AI Specialization, CGPA 9.0) implemented a mini Qwen3 architecture covering Grouped-Query Attention, RoPE, QK-Norm with RMSNorm, and SwiGLU. The training pipeline combined Muon and AdamW hybrid optimisation with cosine LR scheduling, gradient accumulation, and mixed precision over 500K tokens. Beyond this he fine-tuned Qwen3-4B-Base via GRPO for reasoning, fine-tuned Qwen2-VL-7B via LoRA for LaTeX OCR, implemented a Stable Diffusion v1.5 inference pipeline, and built a LangChain-based PDF RAG chatbot. Offers from Intel and progress at PIMIC/IDfy/EY GDS. **Depth ceiling:** implemented, trained, and used at inference scale; evaluated via downstream fine-tuning. **What Forge-LLM adds vs CS23S011:** a public release to PyPI and HF Hub, a published 3000-word blog post, explicit scaling-law honesty, training at ~2000× the token count (1B vs 500K), a custom-trained BPE tokenizer rather than borrowing Qwen's, and engineered resume-safety across session caps.

**ME22B032 — GPT-2 from scratch in PyTorch.** Maringanti Sree Venu Gopal Seshu (ME, CGPA 7.5) built a GPT-2-style transformer from scratch in PyTorch with multi-head self-attention, trained on Tiny Shakespeare to a loss of 1.61, and produced attention-weight visualisations. Paired with production-grade RAG work: a scalable retrieval system on Azure using Qdrant VectorDB + AzureOpenAI GPT-4o embeddings + mxbai-rerank-base reranking, improving query time from 30s to <10s with 35% accuracy and 40% relevance gains over 30+ policy documents. Multi-agent chatbot with WhatsApp via Twilio, MCP server design for a banking RM app. Offer from Niva Bupa as AI Engineer (GenAI). **Depth ceiling:** transformer implemented and trained on toy data; significant production depth in RAG and multi-agent orchestration but the from-scratch artefact itself is a toy. **What Forge-LLM adds vs ME22B032:** modern Llama-family architecture (RoPE, GQA, RMSNorm, SwiGLU) instead of the 2019 GPT-2 stack, training on real web data instead of Tiny Shakespeare, a released checkpoint, KV-cache with measured speedup, and an architecture deep-dive blog.

**AE22B022 — FFN in C++ from scratch.** Jaimin Malhotra (Aerospace, CGPA 7.26) engineered a feedforward neural network from scratch in C++ with manual forward/backward propagation and stochastic gradient descent, trained on 124,000+ EMNIST samples to 88% test accuracy, no ML libraries. Codeforces 1408, top-7% LeetCode, Google ML cert. Strong DSA + low-level numeric implementation, but architecturally narrow (no attention, no convolutions, no language modelling). **Depth ceiling:** from-scratch numeric implementation at the level of a single layer family. **What Forge-LLM adds vs AE22B022:** different category entirely — modern attention variants, language modelling, tokenizer, inference optimisation, evaluation on perplexity, and a shipped package. AE22B022 is a depth signal for systems roles; Forge-LLM is depth + breadth for LLM roles.

### Other relevant cohort hits

**AE21B026 (Jay Gupta, CGPA 9.02).** CUDA kernels for tiled matmul, transpose, softmax, and a flash-attention kernel achieving 3× speedup; Caltech SURF on a Physics-Informed Neural Operator under Anandkumar achieving 45% drag reduction; seq2seq transliteration with attention from scratch; ViT fine-tuning with LoRA on iNaturalist; active research on an LLM-as-reward-model framework for RL agents. Closer to research depth than product depth. **Gap vs Forge-LLM:** no end-to-end LLM training, no public release.

**BS21B040 (Vaibhav Lakhvaiya, CGPA 7.2).** Achieved 60% → 88% accuracy on legal document classification via fine-tuning a LegalBERT + BigBird ensemble at Cardinality, deployed via FastAPI. LSTM/ARIMA forecasting, spectral clustering, conference paper on railway passenger-flow modelling. **Gap vs Forge-LLM:** transformer knowledge is API-level (fine-tuning), not architecture-building.

**AE21B036 (Kishore M, CGPA 8.66).** Deep RL for UAV obstacle avoidance, manuscript under review at Indian Control Conference 2025 on energy-optimal multi-UAV trajectories. Implemented SARSA / Q-Learning / Dueling-DQN / REINFORCE / hierarchical Q-Learning from scratch. **Gap vs Forge-LLM:** narrow domain (control/RL); no LLM or generative modelling work.

**BS21B003 (Afroze Nazira K, CGPA 8.05).** Hybrid log classification combining regex + sentence-transformers + LR + LLM querying via FastAPI, 40% accuracy improvement over baselines; Gemini-based resume analyzer with Streamlit for ATS scoring. **Gap vs Forge-LLM:** API-level LLM integration, no training or architecture work.

### Cohort synthesis — depth ceiling pattern

The cohort stratifies cleanly into three tiers:

1. **From-scratch implementers** — CS23S011 (Qwen3-mini), ME22B032 (GPT-2), AE22B022 (FFN in C++). Real architecture work, but the artefact stops short of a public, runnable release.
2. **Research-adjacent depth** — AE21B026, AE21B036. CUDA kernels, RL from scratch, publication-track research. Strong signals but not LLM-deliverables.
3. **API-layer practitioners** — BS21B040, BS21B003, and most of the cohort. Fine-tuning, RAG, ensembles, prompts. Useful production skills, no architecture depth.

**The pattern that stops at the edge of "PEAK":** every from-scratch project in the cohort halts at the implementation step. None ships a `pip install`-able package, a downloadable HF checkpoint, a public training notebook, a measured KV-cache benchmark, an architecture-ablation chart, or a multi-thousand-word public deep-dive. The cohort proves you can implement a transformer; nobody proves they can ship one. **Forge-LLM is engineered to occupy exactly that vacated space.**

---

## Part 2 — Reference repo survey (B2)

### nanoGPT (Karpathy)

- **Code organisation.** Flat single-directory layout. `model.py` (~300 LoC, all classes: `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, `GPT`), `train.py`, `sample.py`, `bench.py`, `configurator.py`. Per-dataset prep in `data/<dataset>/prepare.py` producing memmap `.bin` files.
- **Config style.** Hybrid: inline `@dataclass GPTConfig` in `model.py` (`block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True`) plus `configurator.py` which `exec()`s a Python config file and `--key=value` CLI overrides into module globals. Karpathy openly admits this is unconventional.
- **Test/eval.** No unit tests. Eval is periodic validation-loss in `train.py`; reproduces GPT-2 OWT loss curves; `eval_gpt2*.py` loads HF weights for zero-shot loss.
- **Steal for Forge-LLM.** Flat readable layout; inline `@dataclass` config next to the model; `configure_optimizers()` decay/no-decay split; `from_pretrained()` pattern (Forge-LLM analogue: `from_hf(repo_id)`); per-dataset `prepare.py` producing `.bin` memmap.
- **Improve on.** Replace `configurator.py`'s `exec()`-into-globals with a typed dataclass + `tyro`/`argparse`/Hydra. Add unit tests. Arbitrary-step resume with full state (optimizer/scheduler/RNG/iterator), not just best-val-loss. HF Hub upload helper. We diverge architecturally (RoPE/GQA/RMSNorm/SwiGLU vs learned-pos/MHA/LayerNorm/GELU).

### llama2.c (Karpathy)

- **Code organisation.** Mixed Python + C. Python: `model.py` (PyTorch Llama-2), `train.py`, `tinystories.py`, `export.py` (PyTorch → custom binary), `tokenizer.py`, same `configurator.py` trick. C: `run.c` (fp32 inference), `runq.c` (int8 quantised), `Makefile`. Tests: `test.c` + `test_all.py`.
- **Config style.** `@dataclass ModelArgs` with `dim, n_layers, n_heads, n_kv_heads, vocab_size, hidden_dim, multiple_of=256, norm_eps=1e-5, max_seq_len, dropout`. Training params via CLI + `configurator.py`.
- **Test/eval.** `test_all.py` cross-validates PyTorch model vs C binary produce matching logits/tokens — parity test, not unit tests. Eval is loss on TinyStories.
- **Steal for Forge-LLM.** Field set is essentially what Forge-LLM needs (`n_kv_heads` for GQA, `multiple_of` for SwiGLU rounding, `norm_eps`). Class layout — `RMSNorm`, free functions `precompute_freqs_cis` / `apply_rotary_emb` / `repeat_kv`, then `Attention`, `FeedForward`, `TransformerBlock`, `Transformer` — is the cleanest Llama PyTorch reference; Forge-LLM should mirror it almost verbatim. `export.py` as template for the HF Hub upload script. Parity test as a regression gate.
- **Improve on.** `model.py.generate()` explicitly has no KV cache ("super inefficient version of sampling") — Forge-LLM must ship a proper KV cache from day one. Training is single-GPU/DDP only; no documented grad-accumulation niceties. No HF Hub integration. Same `configurator.py` smell. Light on API docs.

### gpt-fast (PyTorch Labs)

- **Code organisation.** Inference-only, compact. `model.py` (304 LoC), `generate.py`, `quantize.py`, `tp.py` (tensor parallel), `eval.py` (lm-eval-harness wrapper), `tokenizer.py`. No `train.py`.
- **Config style.** `@dataclass ModelArgs` with `__post_init__` filling `intermediate_size` from `dim`. A `transformer_configs: dict[str, dict]` registry maps names (`"llama-3-8b"`, etc.) to kwarg dicts; `ModelArgs.from_name()` does exact-then-fuzzy lookup.
- **Test/eval.** No unit tests visible. Eval delegated to EleutherAI `lm-eval-harness` via `eval.py`.
- **Steal for Forge-LLM.** **`KVCache(nn.Module)` with statically pre-allocated** `k_cache`/`v_cache` buffers of shape `(max_batch, n_heads, max_seq_len, head_dim)` and an `update(input_pos, k, v)` method using `index_copy_` — this is the pattern that makes `torch.compile` happy and is the right shape for Forge-LLM. Named-preset registry (`forge_llm.configs.PRESETS = {"forge-30m": ...}`). `torch.compile` integration with explicit `compile_prefill` toggle. Tiny model file demonstrates you don't need a framework. Delegation to `lm-eval-harness` instead of rolling your own eval.
- **Improve on.** No training at all — Forge-LLM owns that. Config has no validation beyond dataclass defaults. No tests. Tensor parallel + speculative decoding overkill for 30M; skip.

### HuggingFace LlamaModel

- **Code organisation.** `transformers/models/llama/`: `modeling_llama.py` (~521 LoC), `configuration_llama.py`, `tokenization_llama.py`, `tokenization_llama_fast.py`, `convert_llama_weights_to_hf.py`. KV cache lives in shared `cache_utils.py` (`Cache`, `DynamicCache`, `StaticCache`).
- **Config style.** `LlamaConfig(PretrainedConfig)` — JSON-serialisable, `from_pretrained()`, threaded through submodule `__init__(self, config, layer_idx)`. Heavy: every field documented, many `rope_scaling` knobs.
- **Test/eval.** Real `pytest` suite at `tests/models/llama/test_modeling_llama.py` covering forward shapes, generation, and integration against published checkpoints. Eval delegated to user code.
- **Steal for Forge-LLM.** Class layering: `LlamaRMSNorm → LlamaRotaryEmbedding → LlamaMLP → LlamaAttention → LlamaDecoderLayer → LlamaModel → LlamaForCausalLM`. Forge-LLM should expose both `ForgeModel` (trunk) and `ForgeForCausalLM` (LM head). `Cache` abstraction with pluggable `DynamicCache`/`StaticCache` — support at least one of each. JSON config on disk so HF Hub checkpoints are self-describing. Parallel `tests/` tree.
- **Improve on.** Massive surface area; many `**kwargs`; legacy branches (`rope_scaling` variants, `_attn_implementation` dispatch). Forge-LLM keeps config flat and explicit. Only `eager` and (optionally) `sdpa` attention — `flash_attn` forbidden by our hard rules. HF is hard to read end-to-end as a learning artefact; Forge-LLM optimises for the opposite.

### Cross-cutting recommendations

- **File layout.** Mirror llama2.c's class order but split into a package: `src/forge_llm/{model.py, attention.py, rope.py, cache.py, config.py, generation.py, train.py, data.py, tokenizer.py, hub.py}`. `configs/` (YAML or Python presets) and `tests/` at top level. Each module under ~300 LoC.
- **Config.** Single `@dataclass ForgeConfig` in `config.py` with the llama2.c field set plus `n_kv_heads=2, n_heads=8` for GQA, `rope_theta`, `tie_embeddings`. Named `PRESETS` registry. JSON `from_pretrained`/`save_pretrained`. **Do not** copy `configurator.py`'s `exec()` pattern.
- **Tests/eval.** Real `pytest` suite: shape tests per module, forward/backward smoke on 1-layer model, KV-cache equivalence test (full vs token-by-token, gpt-fast pattern), eval script wrapping `lm-eval-harness` for hellaswag/winogrande/arc-easy *plus* perplexity on WikiText-103 and a held-out FineWeb-Edu slice.
- **Key patterns to mirror.** llama2.c class hierarchy + `ModelArgs` field set; gpt-fast `KVCache(nn.Module)` with pre-allocated static buffers and `update(input_pos, ...)`; gpt-fast named-preset registry; HF `Model` vs `ForCausalLM` split and JSON-on-disk config; nanoGPT `configure_optimizers()` and per-dataset `prepare.py`; llama2.c `export.py` as model for HF Hub upload.
- **Patterns to avoid.** `configurator.py`'s `exec()`-into-globals; HF-style `**kwargs` soup and legacy branches; shipping `generate()` without a KV cache; making `torch.compile` / tensor parallel / speculative decoding mandatory; skipping tests "because the repo is small."

---

## Part 3 — Differentiation analysis (B3)

### Delta vs CS23S011 (Qwen3-mini) — the dangerous comparator
Qwen3-mini has the highest architectural overlap (GQA + RoPE + RMSNorm + SwiGLU). The deltas live in the *delivery surface*, not the architecture spec. Assuming the typical scope of "implemented the architecture in a notebook, trained briefly on a toy corpus, no published artefact," Forge-LLM differentiates on five concrete fronts:
1. **Publishable HF Hub checkpoint** trained on ~1B FineWeb-Edu tokens (real data, real scale for the param budget) — 2000× more tokens than the cited 500K.
2. **`pip install forge-llm`** with a usable inference API.
3. **Public Kaggle training notebook** that any interviewer can fork and re-run for free.
4. **Resume-safety engineering** (model + optim + sched + RNG + iterator + wandb-run-id state) that survives Kaggle's 12h kill — a system-design problem Qwen3-mini cloners typically don't address because they trained on Colab Pro or a personal GPU without session caps.
5. **Custom BPE tokenizer trained on the project's own corpus** rather than reusing Qwen's tokenizer.

Qwen3-mini is an architecture exercise; Forge-LLM is an architecture exercise plus a shipped product plus a $0-compute logistics solution.

### Delta vs ME22B032 (GPT-2 from scratch)
The easiest delta to articulate because every modern-LLM choice in Forge-LLM is a direct upgrade over GPT-2: **RoPE > learned absolute positional embeddings** (length extrapolation, no fixed context); **GQA 8:2 > vanilla MHA** (4× KV-cache memory reduction at inference, same ratio as Llama-2-7B); **RMSNorm > LayerNorm** (one fewer mean-subtraction; the production choice in Llama/Mistral/Qwen); **SwiGLU > GELU MLP** (the gated-FFN variant every frontier open model adopted post-2023). Forge-LLM frames ME22B032 as "the 2019 stack" and itself as "the 2024 stack." Plus: KV-cache with measured speedup, resume-safe training, deployed package + checkpoint — none of which the GPT-2-from-scratch toy ships.

### Delta vs AE22B022 (FFN in C++)
Different category. AE22B022 demonstrates low-level numeric implementation; Forge-LLM owns the full LLM stack (tokenizer training, modern attention variants, mixed-precision training mechanics with grad accumulation to 128K-token effective batch, KV-cache inference, perplexity eval, PyPI + HF Hub). For LLM/NLP/applied-research roles, Forge-LLM is in-domain; AE22B022 is adjacent-domain.

### Delta vs nanoGPT — the dangerous OSS comparator
nanoGPT is famous and works. Forge-LLM cannot win on fame or code minimalism, so it wins on **architectural modernity** + **operational completeness**:
- Forge-LLM ships **RoPE, GQA, RMSNorm, SwiGLU, KV-cache** — nanoGPT ships none.
- Forge-LLM ships **resume-safety across hard session caps** — nanoGPT does not.
- Forge-LLM ships a **custom-trained BPE tokenizer** — nanoGPT uses tiktoken/GPT-2's.
- Forge-LLM ships **a `pip install` package** — nanoGPT is clone-and-hack.
- Forge-LLM ships **unit tests for causal masking, KV-cache equivalence, and resume safety** — nanoGPT has none.
- Forge-LLM ships **a $0-compute Kaggle reproducibility path** — nanoGPT assumes you own GPUs.

Framing: "nanoGPT is a 2022 GPT-2 reference; Forge-LLM is a 2024 Llama-family reference, trainable for $0."

### The defensible headline claim
**Headline:** Forge-LLM is a from-scratch ~30M-parameter Llama-family decoder (RoPE + GQA 8:2 + RMSNorm + SwiGLU) that any interviewer can `pip install`, fork on Kaggle, and resume-train end-to-end on a free T4 across the 12-hour session cap — for $0.

Defensibility comes from the *conjunction*, not any single piece. Each element has a counter ("nanoGPT trains too," "Qwen3-mini has GQA too," "anyone can publish a PyPI package"), but no comparator clears all four gates simultaneously: modern Llama-family architecture; public reproducibility on free hardware; resume-safety across hard session caps; shipped artefacts (package + Hub checkpoint + blog). The conjunction is the moat.

---

## Top 5 differentiation moves (FINAL — locks Phase C design)

1. **Public Kaggle notebook reproducing training end-to-end across two T4 sessions from a fresh fork.** Concrete artefact: a Kaggle URL where session 1 trains and checkpoints to HF Hub, and session 2 resumes from that checkpoint to completion. Verifiable by an interviewer in <5 minutes. Beats every cohort peer with a private notebook and beats "worked on my machine" claims. **Owned by:** M9 (training loop with resume) and Phase G (training execution).

2. **KV-cache speedup benchmark table in the README: tokens/sec at batch 1 with cache off vs on, at context lengths 128 / 512 / 2048.** Turns "I implemented KV-cache" from a bullet into a measured engineering claim (e.g., "3.8× at ctx=2048"). nanoGPT does not have this; cohort peers almost certainly do not. **Owned by:** M11 (sampling + KV-cache) and Phase H (release).

3. **Per-parameter efficiency chart: Forge-LLM (30M) vs nanoGPT-124M vs GPT-2-small validation perplexity on a held-out FineWeb-Edu slice, plotted against params and against training tokens.** Pre-empts the "your model is small" objection by reframing the axis. Even if Forge-LLM loses on absolute perplexity, the per-param or per-FLOP curve is a citeable defensible result. **Owned by:** M10 (eval) and the blog post.

4. **Architecture-choice ablation in the blog: same training recipe, swap RoPE→learned-pos and GQA→MHA, report perplexity + KV-cache memory delta.** Converts the architecture choices from resume bullets into measured design decisions. Separates "I copied Llama" from "I understand why Llama is built this way." **Owned by:** Phase H (blog) — short ablation runs on Kaggle, gated on remaining session quota.

5. **`forge-llm generate` CLI + 30-second asciinema demo in the README running `pip install forge-llm && forge-llm generate "..."` on a clean machine.** Turns the package from a directory of code into a *thing an interviewer touches in 30 seconds*. Conversion from "looked at the repo" to "remembers the candidate" is dominated by this artefact. Zero cohort peers will have it. **Owned by:** M12 (CLI + packaging) and Phase H (release).

---

## Phase C inputs derived from this landscape

- **File layout** → mirror llama2.c class hierarchy in a package structure (B2).
- **Config system** → `@dataclass` + JSON `save_pretrained`/`from_pretrained` (B2); reject `configurator.py` exec pattern. (Will be locked in `docs/01_architecture.md` and ADR-001.)
- **KV cache** → gpt-fast pattern (`nn.Module` with pre-allocated static buffers + `update(input_pos, k, v)`). (Will be locked in `docs/01_architecture.md`.)
- **Logging** → wandb (already locked in CLAUDE.md §10, ADR-002).
- **Eval** → `lm-eval-harness` wrapper + perplexity on WikiText-103 + held-out FineWeb-Edu slice + per-parameter efficiency chart. (Will be locked in `docs/03_training_plan.md`.)
- **Differentiation deliverables (5 above)** → assigned to roadmap milestones M9, M10, M11, M12 and Phase H. (Will be reflected in `docs/04_roadmap.md`.)
