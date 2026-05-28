"""Generation loop (M11).

``generate(model, tokenizer, prompt, ...) -> Iterator[str]`` streams the
generated tokens, one decoded fragment per yield. Uses ``KVCache`` by default
so token-by-token decoding is O(1) per step in attention cost.

The internal ``_sample_token`` indirection exists so the EOS-stop test can
monkey-patch the sampling call without poking at the rest of the loop.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from forge_llm.cache import KVCache
from forge_llm.model import ForgeForCausalLM
from forge_llm.sampling import apply_temperature, repetition_penalty
from forge_llm.sampling import top_k as _top_k
from forge_llm.sampling import top_p as _top_p
from forge_llm.tokenizer import BPETokenizer


def _sample_token(
    logits: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Multinomial sample from ``logits``; returns a ``(B,)`` int64 tensor.

    Factored out so the EOS-stop test can ``patch.object`` this name without
    reaching into ``generate``'s sampling block.
    """
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def generate(
    model: ForgeForCausalLM,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new: int = 50,
    top_p: float = 1.0,
    top_k: int | None = None,
    temperature: float = 1.0,
    rep_penalty: float = 1.0,
    seed: int | None = None,
    use_cache: bool = True,
    device: str | torch.device = "cpu",
) -> Iterator[str]:
    """Stream generated tokens decoded as strings.

    Yields one decoded-token-fragment per step, stops on EOS or after
    ``max_new`` new tokens. With ``seed`` set, two calls with the same seed
    produce byte-identical token sequences via a local ``torch.Generator``
    (no reliance on the global RNG).
    """
    model.eval()
    device_t = torch.device(device)

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=device_t)
        generator.manual_seed(seed)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_tokens:
        # An empty prompt seeds the cache with a single BOS so attention has
        # at least one position to read from.
        prompt_tokens = [tokenizer.bos_id]

    history: list[int] = list(prompt_tokens)
    input_ids = torch.tensor([prompt_tokens], device=device_t, dtype=torch.long)

    cache = None
    if use_cache:
        cache = KVCache.allocate(model.config, max_batch=1).to(device_t)

    with torch.no_grad():
        if use_cache:
            input_pos = torch.arange(len(prompt_tokens), device=device_t)
            logits_full = model(input_ids, cache=cache, input_pos=input_pos)
        else:
            logits_full = model(input_ids)
        next_logits = logits_full[:, -1, :]  # (1, V)

        max_pos = model.config.max_seq
        for _ in range(max_new):
            # The cache only has slots up to max_seq; stop cleanly rather than
            # OOB-indexing freqs_cis on long generations.
            if len(history) >= max_pos:
                return
            modified = next_logits[0].clone()
            if rep_penalty != 1.0:
                hist = torch.tensor(history, device=device_t, dtype=torch.long)
                modified = repetition_penalty(modified, hist, rep_penalty)
            modified = apply_temperature(modified, temperature)
            if top_k is not None:
                modified = _top_k(modified, top_k)
            if top_p < 1.0:
                modified = _top_p(modified, top_p)

            next_token = _sample_token(modified.unsqueeze(0), generator=generator)
            token_id = int(next_token.item())

            if token_id == tokenizer.eos_id:
                return

            yield tokenizer.decode([token_id])
            history.append(token_id)

            if use_cache:
                step_input = next_token.unsqueeze(0)  # (1, 1)
                step_pos = torch.tensor([len(history) - 1], device=device_t)
                logits_step = model(step_input, cache=cache, input_pos=step_pos)
            else:
                logits_step = model(torch.tensor([history], device=device_t, dtype=torch.long))
            next_logits = logits_step[:, -1, :]
