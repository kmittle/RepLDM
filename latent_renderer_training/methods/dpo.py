"""Renderer-DPO formal handler."""

from .runner import run_pair_training


def run_dpo(binding):
    return run_pair_training(binding, "dpo")


__all__ = ["run_dpo"]
