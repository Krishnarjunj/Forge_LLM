"""Forge-LLM: a from-scratch ~30M-param Llama-family decoder.

Public API as of M8: ``ForgeConfig``, ``PRESETS``, ``ForgeModel``,
``ForgeForCausalLM``. Hub utilities (M9) and ``generate`` (M11) extend this
list as the milestones land.
"""

from forge_llm.config import PRESETS, ForgeConfig
from forge_llm.model import ForgeForCausalLM, ForgeModel

__version__ = "0.0.0"

__all__ = ["PRESETS", "ForgeConfig", "ForgeForCausalLM", "ForgeModel"]
