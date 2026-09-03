"""Winner-only native-transition distillation handler."""

from .runner import run_pair_training


def run_search_distill(binding):
    return run_pair_training(binding, "search_distill")


__all__ = ["run_search_distill"]
