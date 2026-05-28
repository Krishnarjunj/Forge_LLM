"""Typed config (``ForgeConfig``) and named presets registry (``PRESETS``).

ADR-001 (CLOSED) picks a frozen ``@dataclass`` over Hydra / plain YAML so that
configs are typed (a fat-fingered ``n_kv_head: "2"`` does not propagate
silently) and serialise cleanly to JSON for HF Hub artifacts. ``__post_init__``
performs the shape validation listed in ADR-001's Consequences block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True)
class ForgeConfig:
    """Architecture + training-side hyperparameters for a Forge-LLM model.

    Frozen so configs can be hashed (useful for run caches) and so a downstream
    accidental mutation does not silently desync the model from the saved
    config JSON. Construct via ``ForgeConfig(**PRESETS["forge-30m"])`` or load
    from YAML.
    """

    name: str
    n_layer: int
    d_model: int
    n_head: int
    n_kv_head: int
    head_dim: int
    d_ff: int
    vocab_size: int
    max_seq: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")
        if self.n_kv_head <= 0:
            raise ValueError(f"n_kv_head must be positive, got {self.n_kv_head}")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by "
                f"n_kv_head ({self.n_kv_head})"
            )
        if self.head_dim * self.n_head != self.d_model:
            raise ValueError(
                f"head_dim * n_head ({self.head_dim} * {self.n_head} = "
                f"{self.head_dim * self.n_head}) must equal "
                f"d_model ({self.d_model})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even (for RoPE)"
            )
        if self.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {self.vocab_size}"
            )
        if self.max_seq <= 0:
            raise ValueError(f"max_seq must be positive, got {self.max_seq}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")


# Locked from configs/model_30m.yaml. Changes require an ADR.
PRESETS: Final[dict[str, dict[str, Any]]] = {
    "forge-30m": {
        "name": "forge-30m",
        "n_layer": 6,
        "d_model": 512,
        "n_head": 8,
        "n_kv_head": 2,
        "head_dim": 64,
        "d_ff": 1408,
        "vocab_size": 16384,
        "max_seq": 1024,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
        "init_std": 0.02,
        "dropout": 0.0,
    },
}
