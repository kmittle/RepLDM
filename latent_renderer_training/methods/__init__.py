"""Source-registered latent-renderer training methods."""

from .common import (
    ScoredRollout,
    dpo_objective,
    opd_objective,
    rl_objective,
    search_distill_objective,
)
from .opd import run_opd

__all__ = [
    "ScoredRollout",
    "dpo_objective",
    "opd_objective",
    "rl_objective",
    "search_distill_objective",
    "run_opd",
]
