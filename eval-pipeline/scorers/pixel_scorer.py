"""Weightless pixel witnesses (no model / no weights). Wraps metrics.py.

These are RepLDM-unique decorrelated reward-hacking witnesses (§13.5) — colorfulness,
Laplacian sharpness, mean saturation, clipped fraction, contrast std. Sana has no
low-level pixel metrics; these stay our own.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval-pipeline/ on path
import metrics  # noqa: E402

from .base import Scorer, register_metric


@register_metric("pixel")
class PixelScorer(Scorer):
    OUTPUT_KEYS = (("colorfulness", "witness"), ("laplacian_sharpness", "witness"),
                   ("mean_saturation", "witness"), ("clipped_fraction", "witness"),
                   ("contrast_std", "witness"))

    def score_image(self, image, prompt):
        return metrics.compute_all(image)
