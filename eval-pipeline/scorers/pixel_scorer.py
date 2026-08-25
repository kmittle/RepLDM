"""Weightless pixel witnesses (no model / no weights). Wraps metrics.py.

These are low-level reward-hacking witnesses (§13.5) — colorfulness,
Laplacian sharpness, mean saturation, clipped fraction, contrast std. Sana has no
low-level pixel metrics; these stay our own.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval-pipeline/ on path
import metrics  # noqa: E402

from scorer_provenance import source_file_record
from .base import Scorer, register_metric


@register_metric("pixel")
class PixelScorer(Scorer):
    OUTPUT_KEYS = (("colorfulness", "witness"), ("laplacian_sharpness", "witness"),
                   ("mean_saturation", "witness"), ("clipped_fraction", "witness"),
                   ("contrast_std", "witness"))
    PROVENANCE_PACKAGES = ("numpy", "Pillow")

    def provenance_metadata(self):
        return {
            "models": [],
            "checkpoint_files": [],
            "preprocessing": {
                "input": "PIL RGB",
                "array_dtype": "uint8",
                "spatial_resolution": "native",
                "clipped_fraction_thresholds": {"lo": 2, "hi": 253},
            },
            "parameters": {},
            "supporting_sources": [
                source_file_record(
                    metrics.__file__,
                    label="pixel_metric_implementation",
                    root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    module="metrics",
                )
            ],
        }

    def score_image(self, image, prompt):
        return metrics.compute_all(image)
