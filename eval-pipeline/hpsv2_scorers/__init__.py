"""Scorers isolated from legacy registrations for the full HPSv2 run."""

from .base import REGISTRY, StagedScorerMixin, register_metric  # noqa: F401
from . import aesthetic_scorer  # noqa: F401,E402
from . import clip_scorer  # noqa: F401,E402
from . import hps_scorer  # noqa: F401,E402
from . import imagereward_scorer  # noqa: F401,E402
from . import iqa_scorer  # noqa: F401,E402
from . import pixel_scorer  # noqa: F401,E402
