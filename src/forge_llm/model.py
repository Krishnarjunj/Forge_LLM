"""Forge-LLM full model (M8).

``ForgeModel`` is the decoder trunk (embedding + N pre-norm blocks + final
RMSNorm + RoPE cache). ``ForgeForCausalLM`` adds the LM head, tied to the
input embedding by default (per docs/01_architecture.md sec 7).

Initialisation follows the recipe in docs/01_architecture.md sec 7:

* Linear weights and embeddings ~ ``N(0, init_std)``; biases zero
  (most linears are bias-free per Llama convention).
* RMSNorm weight stays at ones (its own ``__init__`` default).
* GPT-2 residual scaling: ``attn.wo`` and ``mlp.w_down`` are scaled by
  ``1 / sqrt(2 * n_layer)`` so variance stays bounded across depth.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from forge_llm.block import TransformerBlock
from forge_llm.config import ForgeConfig
from forge_llm.norm import RMSNorm
from forge_llm.rope import precompute_freqs_cis


class ForgeModel(nn.Module):
    """Decoder trunk: embedding + ``n_layer`` blocks + final RMSNorm."""

    freqs_cis: Tensor  # registered buffer; declared for mypy strict.

    def __init__(self, config: ForgeConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_head=config.n_head,
                    n_kv_head=config.n_kv_head,
                    head_dim=config.head_dim,
                    d_ff=config.d_ff,
                    max_seq=config.max_seq,
                    eps=config.norm_eps,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        freqs = precompute_freqs_cis(
            config.head_dim, config.max_seq, config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Embed -> stack of blocks -> final RMSNorm.

        Shape: ``input_ids`` is ``(B, T)`` int; returns ``(B, T, d_model)``.
        """
        x = self.embed_tokens(input_ids)
        for block in self.layers:
            x = block(x, self.freqs_cis)
        return self.norm(x)


class ForgeForCausalLM(nn.Module):
    """``ForgeModel`` + LM head (tied to the input embedding by default)."""

    def __init__(self, config: ForgeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ForgeModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._init_weights()
        if config.tie_embeddings:
            # Share the embedding tensor with the LM head. The lm_head's
            # original .weight (allocated by nn.Linear's __init__ + our
            # _init_weights pass) is dropped here -- which is fine, we only
            # ever want the embedding tensor on the output side.
            self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.model.embed_tokens

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=self.config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.config.init_std)
        # GPT-2 residual scaling on the second linear in each residual branch.
        scale = (2 * self.config.n_layer) ** -0.5
        for block in self.model.layers:
            with torch.no_grad():
                block.attn.wo.weight.mul_(scale)
                block.mlp.w_down.weight.mul_(scale)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Run the trunk and project to vocab logits.

        Shape: ``input_ids`` is ``(B, T)`` int; returns ``(B, T, vocab_size)``.
        """
        hidden = self.model(input_ids)
        return self.lm_head(hidden)
