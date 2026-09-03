"""Reference-anchored RL formal handler."""

from .runner import run_pair_training


def run_rl(binding):
    return run_pair_training(binding, "rl")


__all__ = ["run_rl"]
