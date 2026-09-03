"""Source-registered external-teacher OPD handler."""

from .runner import run_opd_training


def run_opd(binding):
    """Run OPD only through the frozen external-teacher schedule."""
    return run_opd_training(binding)


__all__ = ["run_opd"]
