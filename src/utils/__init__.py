"""utils — a minimal, principled implementation of the irace racing algorithm.

Features:
    - Single-race racing with rank-based (Friedman + Nemenyi) or mean-based
      (paired t-test) elimination.
    - Pluggable candidate sampling and target evaluation hooks.
    - Budget-aware loop with configurable schedule (T^first, T^each).

Public API:
    - `Config`, `RaceState`: data structures used by the driver and tests.
"""

from .race import (
    elitist_race,
    print_iteration_header,
    print_elite_configs,
    print_markers_header,
)
from .config import Config, RaceState, ConfigAS
from .llm import (
    OpenRouterClient,
    OllamaClient,
    MistralClient,
    vLLMClient,
    GoogleClient,
    CachedLLM,
    parse_eoh_response,
)
from .logger import make_wandb_logger

__all__ = [
    "elitist_race",
    "print_iteration_header",
    "print_elite_configs",
    "print_markers_header",
    "Config",
    "RaceState",
    "ConfigAS",
    "OpenRouterClient",
    "OllamaClient",
    "MistralClient",
    "vLLMClient",
    "GoogleClient",
    "CachedLLM",
    "parse_eoh_response",
    "make_wandb_logger",
]
