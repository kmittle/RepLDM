"""Scorer package — importing it registers every metric into base.REGISTRY."""
from .base import REGISTRY, Scorer, register_metric  # noqa: F401
from . import aesthetic_scorer, clip_scorer, hps_scorer, imagereward_scorer, iqa_scorer, pixel_scorer  # noqa: F401,E402
