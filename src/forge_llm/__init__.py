"""Forge-LLM: a from-scratch ~30M-param Llama-family decoder.

Public API: ``ForgeConfig``, ``PRESETS``, ``ForgeModel``, ``ForgeForCausalLM``
(M8) and ``generate`` (M11).
"""

from forge_llm.config import PRESETS, ForgeConfig
from forge_llm.generation import generate
from forge_llm.model import ForgeForCausalLM, ForgeModel

__version__ = "0.1.0a1"

__all__ = [
    "PRESETS",
    "ForgeConfig",
    "ForgeForCausalLM",
    "ForgeModel",
    "generate",
]
