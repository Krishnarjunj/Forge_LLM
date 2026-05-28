# 02 — Correctness Plan

> Test contract. Every milestone in `docs/04_roadmap.md` lists which tests from this file must be green before the milestone exits. Tolerances per `CLAUDE.md` §4.
>
> Hard rule: oracles are imported only in `tests/` (never in `src/`). If an oracle can't be set up, the test is skipped with a `pytest.skip(reason=...)` referencing this file — values are never fabricated.

---

## 0. Global testing conventions

- Test runner: `pytest -q` from repo root. Marker `@pytest.mark.slow` for tests that load HF reference models (≥30s); CI runs them on a separate job, dev runs `pytest -q -m "not slow"`.
- Fixtures (in `tests/conftest.py`):
  - `seed_all(seed=0)` invoked autouse on every test.
  - `tiny_config` → `ForgeConfig` at `n_layer=2, d_model=64, n_head=4, n_kv_head=2, head_dim=16, vocab=128, max_seq=32`. Used for fast shape/grad tests.
  - `cpu_device`, `cuda_device` (skip if no CUDA).
  - `tmp_ckpt` → tempdir for checkpoint write/read.
- Coverage threshold: **≥85%** statement coverage in `src/forge_llm/` reported by `pytest-cov`. CI fails below that. The threshold is intentionally below 100% because `train.py` paths that require GPU + real data are exercised in Phase F preflight, not unit tests.
- All comparisons against reference oracles use `torch.testing.assert_close(actual, expected, rtol=..., atol=...)` not `torch.allclose` — `assert_close` produces useful error messages on mismatch.

---

## 1. Per-layer correctness tests

### 1.1 RMSNorm (`src/forge_llm/norm.py`)

| Test name | Input shape | Oracle | Tolerance | Bug it catches |
|-----------|-------------|--------|-----------|----------------|
| `test_rmsnorm_shape` | `(2, 8, 64)` fp32 | n/a — own forward | shape exact | Wrong axis (normalising over batch instead of features) |
| `test_rmsnorm_value_vs_llama` | `(2, 8, 64)` fp32 | `transformers.models.llama.modeling_llama.LlamaRMSNorm(hidden_size=64, eps=1e-5)` with weights copied from our module | rtol/atol = 1e-6 (fp32, exactly equivalent op) | Wrong ε placement (inside vs outside sqrt), wrong reduction dim, missing learnable γ |
| `test_rmsnorm_backward` | `(2, 8, 8)` fp64 + `gradcheck` | n/a — uses `torch.autograd.gradcheck` | gradcheck default | Bad gradient routing through `rsqrt` |
| `test_rmsnorm_determinism` | `(2, 8, 64)` × 2 same seed | n/a | byte-identical (`torch.equal`) | Hidden nondeterministic op |
| `test_rmsnorm_dtype_promotion` | `(2, 8, 64)` fp16 in → fp16 out, but internal compute fp32 | comparison: fp32-input vs fp16-input-then-cast | atol = 1e-3 | Loss of fp16 stability if compute is done in fp16 |

### 1.2 RoPE (`src/forge_llm/rope.py`)

| Test name | Input shape | Oracle | Tolerance | Bug it catches |
|-----------|-------------|--------|-----------|----------------|
| `test_rope_freqs_shape` | head_dim=16, max_seq=32, theta=10000 | n/a | shape `(32, 8)` complex or `(32, 16)` real (whichever convention) | Off-by-one on `head_dim // 2` |
| `test_rope_value_vs_llama` | q,k of shape `(2, 4, 8, 16)` fp32 | `LlamaRotaryEmbedding.apply_rotary_pos_emb(q, k, cos, sin)` with our `freqs_cis` converted to (cos, sin) | rtol/atol = 1e-5 | Wrong rotation convention (Llama interleaved vs original-paper paired) |
| `test_rope_rotation_identity` | Apply rotation at position 0 — should be identity | n/a (math identity) | atol = 1e-7 (fp32) | Wrong sign on sin term |
| `test_rope_relative_position_invariance` | Compute attention scores `q_m · k_n` and `q_{m+s} · k_{n+s}` for various s, m, n | should be equal (RoPE property: depends only on m−n) | rtol = 1e-5 | RoPE that depends on absolute position only — defeats the entire point of RoPE |
| `test_rope_long_context_extrapolation` | Apply at position 4096 (well beyond max_seq=1024 training) | shape preserved, no NaN | finite check + atol 1e-3 vs naive computation | Numeric blowup at long positions when theta is mis-set |
| `test_rope_determinism` | Same input twice | byte-identical | `torch.equal` | Hidden nondeterminism |

**RoPE convention note (ADR-007, amended 2026-05-28):** Forge-LLM uses the **HF Llama half-split convention** (rotate halves `(x[:d/2], x[d/2:])` via `y = x*cos + rotate_half(x)*sin`), to match `transformers.models.llama.modeling_llama.apply_rotary_pos_emb` bit-for-bit. The earlier version of this note prescribed Meta interleaved on the mistaken belief that HF used it; see ADR-007's Amendment block for the correction.

### 1.3 Vanilla MHA (`src/forge_llm/attention.py`, transitional)

Per the roadmap, we implement MHA first (M4) as a stepping stone before GQA (M5). MHA shares its forward path with GQA except `n_kv_head == n_head`.

| Test name | Input shape | Oracle | Tolerance | Bug it catches |
|-----------|-------------|--------|-----------|----------------|
| `test_mha_shape` | `(2, 8, 64)`, n_head=4, head_dim=16 | n/a | output shape `(2, 8, 64)` | Wrong reshape/permute order |
| `test_mha_value_vs_torch_reference` | same | `torch.nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True, bias=False)` configured to produce equivalent op, given identical weights | rtol=1e-5, atol=1e-5 | Wrong scaling (missing `1/sqrt(head_dim)`), wrong softmax dim, wrong head-permute |
| `test_mha_causal_no_leak` | mutate token T-1, check t<T-1 byte-identical | n/a | `torch.equal` on slice `[:, :T-1, :]` | Future-token leakage through mask |
| `test_mha_backward` | tiny fp64 + gradcheck | n/a | gradcheck default | Bad backward |
| `test_mha_determinism` | seeded forward twice | byte-identical | `torch.equal` | Hidden nondeterminism |

### 1.4 GQA (`src/forge_llm/attention.py`)

| Test name | Input shape | Oracle | Tolerance | Bug it catches |
|-----------|-------------|--------|-----------|----------------|
| `test_gqa_shape` | `(2, 8, 64)`, n_head=4, n_kv_head=2 | n/a | output `(2, 8, 64)` | Wrong KV repeat or projection shape |
| `test_gqa_reduces_to_mha_when_kv_eq_q` | configure n_kv_head = n_head | should equal our MHA forward | rtol=1e-6 | Different code path produces different result |
| `test_gqa_value_vs_llama_attention` | tiny config matching HF | `LlamaAttention` with weights copied across our (Wq, Wk, Wv, Wo) into theirs | rtol=1e-5 | Wrong K/V repeat scheme (`einops.repeat` vs `expand` vs explicit broadcast — only one is bitwise-correct under fp16) |
| `test_gqa_kv_head_grouping_count` | n_head=8, n_kv_head=2 | n/a | assert K shape after repeat is `(B, 8, T, head_dim)` not `(B, 2, T, head_dim)` | Forgetting to repeat KV to match Q head count |
| `test_gqa_causal_no_leak` | mutate token T-1 | n/a | byte-identical at t<T-1 | Mask leak |
| `test_gqa_with_rope_value_vs_llama` | full attention + RoPE applied | `LlamaAttention` with rope | rtol=1e-5 | RoPE applied at wrong stage (post-projection vs pre-projection) |
| `test_gqa_backward` | tiny fp64 + gradcheck | gradcheck default | Bad backward through head grouping |

### 1.5 SwiGLU (`src/forge_llm/mlp.py`)

| Test name | Input shape | Oracle | Tolerance | Bug it catches |
|-----------|-------------|--------|-----------|----------------|
| `test_swiglu_shape` | `(2, 8, 64)`, d_ff=176 | n/a | output `(2, 8, 64)` | Wrong projection dim |
| `test_swiglu_value_vs_llama_mlp` | weights copied to `LlamaMLP` | `LlamaMLP` | rtol=1e-5 | Wrong gate vs up assignment (swapping w_gate and w_up silently passes shape tests but fails value test) |
| `test_swiglu_uses_silu_not_gelu` | inject known-value input, compute expected `SiLU(gate) * up` by hand on one element | manual | atol=1e-7 | Using GELU when we say SwiGLU |
| `test_swiglu_backward` | gradcheck | gradcheck default | Bad backward |
| `test_swiglu_determinism` | byte-identical | `torch.equal` | Hidden nondeterminism |

### 1.6 KV cache (`src/forge_llm/cache.py`)

| Test name | Input shape | Oracle | Tolerance | Bug it catches |
|-----------|-------------|--------|-----------|----------------|
| `test_kvcache_shape` | max_batch=2, max_seq=16, n_kv_head=2, head_dim=16 | n/a | k_cache, v_cache shape `(2, 2, 16, 16)` | Wrong buffer allocation shape |
| `test_kvcache_update_correctly_overwrites_positions` | update at positions [0,1,2] then [3,4] | manually constructed expected buffer | atol=0 (integer-indexed, exact) | `index_copy_` arg order, off-by-one in `input_pos` |
| `test_kvcache_does_not_grow_unboundedly` | call `.update()` 16 times | n/a | assert `k_cache.shape[2] == 16` (the pre-allocated size, not growing) | Accidentally `cat`-ing instead of indexing — silent OOM on long generation |

### 1.7 Causal mask correctness (the adversarial test) — `tests/test_causal_mask.py`

This is the headline correctness test.

```python
def test_causal_mask_adversarial_no_future_leak():
    """
    Mutate the value of token T-1, assert that outputs at all positions t < T-1
    are byte-identical to the unmutated forward. Any single-bit difference proves
    future-token information leaked through the mask.
    """
    seed_all(0)
    model = ForgeForCausalLM(tiny_config).eval()
    tokens = torch.randint(0, tiny_config.vocab_size, (1, 8))
    out_a = model(tokens).logits

    tokens_b = tokens.clone()
    tokens_b[0, -1] = (tokens_b[0, -1].item() + 17) % tiny_config.vocab_size
    out_b = model(tokens_b).logits

    # Output at positions 0..T-2 must be byte-identical.
    torch.testing.assert_close(out_a[:, :-1, :], out_b[:, :-1, :], rtol=0, atol=0)
    # Position T-1 must differ (sanity: the mutation actually propagated).
    assert not torch.equal(out_a[:, -1, :], out_b[:, -1, :])
```

This catches: wrong mask direction (lower-triangular vs upper-triangular swapped); off-by-one in mask construction (allowing attention to *self* but ALSO to the next position); softmax normalisation bug that lets a tiny epsilon of future info leak. All three are silent bugs that pass the loss-decrease check and only show up under adversarial testing.

### 1.8 KV-cache equivalence test (the other headline) — `tests/test_cache.py::test_kvcache_equivalence`

```python
def test_kvcache_full_vs_token_by_token_equivalence():
    """
    Full-sequence forward (no cache) must produce the same logits as token-by-token
    forward with the KV cache, within fp32 numerical tolerance.
    """
    seed_all(0)
    model = ForgeForCausalLM(tiny_config).eval()
    tokens = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.max_seq))

    with torch.no_grad():
        logits_full = model(tokens, use_cache=False).logits

        cache = KVCache.allocate(tiny_config, max_batch=1)
        logits_cached = []
        for t in range(tiny_config.max_seq):
            out = model(tokens[:, t:t+1], use_cache=True,
                        cache=cache, input_pos=torch.tensor([t]))
            logits_cached.append(out.logits)
        logits_cached = torch.cat(logits_cached, dim=1)

    torch.testing.assert_close(logits_full, logits_cached, rtol=1e-5, atol=1e-5)
```

This catches: wrong `input_pos` indexing into the cache; RoPE applied with the wrong position when in cached vs uncached path; cache slicing returning stale zeros for unfilled positions.

### 1.9 Full block test (`tests/test_block.py`)

| Test name | Input shape | Oracle | Tolerance | Bug it catches |
|-----------|-------------|--------|-----------|----------------|
| `test_block_shape` | `(2, 8, 64)` | n/a | output `(2, 8, 64)` | |
| `test_block_value_vs_llama_decoder_layer` | weights copied to `LlamaDecoderLayer` | rtol=1e-5 | Wrong pre-norm vs post-norm ordering, wrong residual connection wiring |
| `test_block_backward` | gradcheck fp64 | gradcheck default | Bad backward through residuals |
| `test_block_residual_routing` | zero-out the attention submodule, assert output equals input + MLP(norm2(input)) | atol=1e-6 | Residual connected to wrong tensor |

### 1.10 Full model test (`tests/test_model.py`)

| Test name | Setup | Tolerance | Bug it catches |
|-----------|-------|-----------|----------------|
| `test_model_param_count` | construct `forge-30m`, count params | exact equality with the table in `docs/01_architecture.md` §6 | Silent architecture drift |
| `test_model_embeddings_are_tied` | `model.lm_head.weight is model.embed_tokens.weight` | identity check | Untying ruins generation quality and breaks our param count |
| `test_model_forward_shape` | random tokens `(2, 32)` | logits shape `(2, 32, 16384)` | Wrong head projection |
| `test_model_value_vs_llama_smallcfg` | identical small config, weights copied to `LlamaForCausalLM` | rtol=1e-4, atol=1e-4 (loosen slightly due to accumulated drift across 6 layers in fp32) | Any of the per-layer bugs that survived isolation |
| `test_model_dtype_inference` | fp16 input on CPU + fp32 master; ensure output is fp16, intermediate norm/softmax cast correctly | atol=5e-3 | Accidental dtype downgrade |

### 1.11 Tokenizer (`tests/test_tokenizer.py`)

| Test name | Setup | Tolerance | Bug it catches |
|-----------|-------|-----------|----------------|
| `test_bpe_roundtrip` | train on small corpus, encode→decode, exact string equality | exact | Lossy decode |
| `test_bpe_special_tokens` | `<bos>`, `<eos>`, `<pad>`, `<unk>` round-trip with `add_special_tokens=True` and `=False` | exact | Special-token mishandling |
| `test_bpe_vocab_size_matches_config` | `tokenizer.vocab_size == config.vocab_size` | exact | Mismatch causes embedding OOB |
| `test_bpe_against_tiktoken_sanity` (slow) | encode same string with `tiktoken.get_encoding("gpt2")` and our BPE on the *same* training corpus — token counts should be within ±20% | numerical sanity | BPE training fundamentally broken |
| `test_bpe_save_load` | train, save, load in new process, encode → identical tokens | exact | Lossy serialisation |

### 1.12 Sampling (`tests/test_sampling.py`)

| Test name | Setup | Tolerance | Bug it catches |
|-----------|-------|-----------|----------------|
| `test_top_k_keeps_only_k_logits` | known logits, k=3, check k values nonzero, rest -inf | exact | Off-by-one on k |
| `test_top_p_distribution_sanity` | tight-spike distribution + p=0.5 → should keep 1 token; flat distribution + p=0.5 → keeps half | manual | Cumulative-sum bug |
| `test_temperature_extremes` | T=0 → argmax (or epsilon-greedy); T=∞ → uniform | KL within 1e-2 | Division by zero on T=0; wrong scaling |
| `test_repetition_penalty_known_input` | logits with known token IDs in history, apply penalty=1.2, check those logits divided/multiplied correctly per the CTRL paper formula | exact | Wrong sign for negative logits |

### 1.13 Generation (`tests/test_generation.py`)

| Test name | Setup | Tolerance | Bug it catches |
|-----------|-------|-----------|----------------|
| `test_generate_deterministic_with_seed` | call `generate(prompt, seed=0, max_new=20)` twice | byte-identical token sequence | Hidden RNG |
| `test_generate_uses_cache_by_default` | mock the attention layer, assert it's called with `use_cache=True` | mock-assert | Cache silently disabled |
| `test_generate_stops_on_eos` | force-feed an EOS in the sampling stream, assert iterator halts | iteration count | EOS not respected |
| `test_generate_load_checkpoint_and_emit_100_tokens` (slow) | load a fixture checkpoint, generate 100 tokens deterministically | byte-identical | End-to-end regression |

### 1.14 Data pipeline (`tests/test_data.py`)

| Test name | Setup | Tolerance | Bug it catches |
|-----------|-------|-----------|----------------|
| `test_packed_dataset_yields_correct_seq_len` | mock stream of varying-length docs, ask for seq_len=32 | every yielded tensor is exactly len 32 | Packing leaks short sequences |
| `test_packed_dataset_iterator_step_preserves_position` | iterate N steps, save iterator state, build new iterator from saved state, next N steps should match steps N+1..2N of the original | exact token equality | Resume reads from the wrong offset (loss curve discontinuity at resume) |
| `test_packed_dataset_no_eos_at_arbitrary_position_leak` | when packing, EOS must be inserted between docs but not at arbitrary positions | exact | Pack boundary bug |

### 1.15 Resume safety (the third headline) — `tests/test_resume.py`

```python
@pytest.mark.slow
def test_resume_safety_loss_curve_indistinguishable():
    """
    Train for 200 steps uninterrupted; record loss curve A.
    Train for 100 steps, checkpoint, kill, resume from checkpoint, continue to 200; record loss curve B.
    Assert steps 101..200 of A and B are bitwise-identical on single-GPU fp32 (CPU acceptable in CI).
    """
    cfg = make_resume_test_config()  # small: tiny_config + tiny data
    
    # Run A: uninterrupted
    seed_all(0)
    losses_a = run_training(cfg, steps=200)
    
    # Run B: interrupted at step 100
    seed_all(0)
    trainer = Trainer(cfg)
    trainer.train_steps(100)
    ckpt_path = trainer.save_checkpoint()
    
    # Simulate process death + cold restart
    del trainer
    trainer2 = Trainer.load_checkpoint(ckpt_path)
    losses_b = trainer2.train_steps(100)  # steps 101..200
    
    # Compare step-by-step
    for i, (a, b) in enumerate(zip(losses_a[100:], losses_b)):
        torch.testing.assert_close(torch.tensor(a), torch.tensor(b),
                                    rtol=0, atol=1e-6,
                                    msg=f"Resume drift at step {101+i}: {a} vs {b}")
```

This catches: missing RNG state restore; data iterator position lost; optimizer momentum state lost; LR scheduler step counter lost; wandb logging drift (a soft fail — wandb is checked separately).

The test runs in fp32 on CPU to make the bitwise comparison meaningful (fp16 + autocast on CUDA introduces non-deterministic accumulation order that we'd have to loosen with `atol=1e-4`; doing it in fp32 keeps the test strict).

---

## 2. CI matrix

| Job | What it runs | When |
|-----|--------------|------|
| `lint` | `ruff check src/ tests/` | push, PR |
| `type-check` | `mypy src/` (strict mode on `src/`, lenient on `tests/`) | push, PR |
| `unit` | `pytest -q -m "not slow"` on Python 3.11 and 3.12 | push, PR |
| `import-audit` | `grep -rE "(MultiheadAttention\|scaled_dot_product\|xformers\|flash_attn\|^from transformers\|^import transformers)" src/` returns nothing | push, PR |
| `slow` | `pytest -q -m "slow"` — runs HF oracle comparisons, resume-safety, generation | nightly + on release branches |

CI fails the PR if any of `lint`, `type-check`, `unit`, `import-audit` fails. `slow` failures block release but not PRs.

---

## 3. Coverage thresholds

- `src/forge_llm/`: ≥85% statement coverage in unit tests.
- `src/forge_llm/train.py` and `src/forge_llm/data.py`: ≥60% — the remaining lines exercise real GPU + real data and are covered by Phase F preflight.
- New code added in Phase E PRs may not lower coverage by more than 2 percentage points; configured in `pyproject.toml` under `[tool.coverage.report]`.

---

## 4. What does NOT have a test (intentional)

- The wandb integration code path beyond the no-network fallback. Wandb-side rendering is not Forge-LLM's responsibility.
- The CLI's argument parsing minutiae — covered by `--help` smoke test in `cli.py`'s docstring example.
- HF Hub upload retries and rate-limit handling — exercised in Phase H manually.

All three are recorded here so reviewers don't add tests we explicitly decided not to write.
