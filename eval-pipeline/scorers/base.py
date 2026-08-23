"""Scorer registry + base class for the RepLDM eval-pipeline.

Each metric is a self-contained `Scorer` that loads its own model(s) ONCE and is
fully DECOUPLED from Sana (uses upstream pip packages only; no Sana imports).
A config (configs/*.yaml) lists which metrics to run; score.py dispatches them.
Adding GenEval/DPG/FID later = drop a new module here + register it.
"""
from __future__ import annotations

REGISTRY: dict = {}


def register_metric(name):
    def deco(cls):
        cls.NAME = name
        REGISTRY[name] = cls
        return cls
    return deco


class Scorer:
    NAME = None
    # OUTPUT_KEYS: tuple of (column, direction); direction in {"higher", "lower", "witness"}
    OUTPUT_KEYS: tuple = ()

    def __init__(self, device="cuda", **params):
        self.device = device
        self.params = params

    @classmethod
    def weights_status(cls, **params):
        """Return (ready: bool, message: str). Override to validate OFFLINE weights
        before instantiation so score.py can skip a metric cleanly instead of crashing."""
        return True, ""

    def score_image(self, image, prompt):
        """Return {column: float, ...} for one PIL.Image + prompt string."""
        raise NotImplementedError
