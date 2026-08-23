"""Scorer package — importing it registers every metric into base.REGISTRY."""
from .base import REGISTRY, Scorer, register_metric  # noqa: F401
from . import imagereward_scorer, pixel_scorer, clip_scorer, hps_scorer, aesthetic_scorer  # noqa: F401,E402
