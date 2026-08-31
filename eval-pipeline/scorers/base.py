"""Scorer registry + base class for the RepLDM eval-pipeline.

Each metric is a self-contained `Scorer` that loads its own model(s) ONCE and is
fully DECOUPLED from Sana (uses upstream pip packages only; no Sana imports).
A config (configs/*.yaml) lists which metrics to run; score.py dispatches them.
Adding GenEval/DPG/FID later = drop a new module here + register it.
"""
from __future__ import annotations

import copy

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
    # Strict scoring requires subclasses to name every distribution that can
    # affect their result and to implement provenance_metadata().
    PROVENANCE_PACKAGES: tuple[str, ...] | None = None

    def __init__(
        self,
        device="cuda",
        scorer_assets=None,
        scorer_asset_revisions=None,
        scorer_asset_manifest=None,
        **params,
    ):
        self.device = device
        self.params = params
        self.scorer_assets = dict(scorer_assets or {})
        self.scorer_asset_revisions = dict(scorer_asset_revisions or {})
        self.scorer_asset_manifest = copy.deepcopy(scorer_asset_manifest)
        self.asset_loading_mode = (
            "pinned_private_stage_v1"
            if self.scorer_assets
            else "mutable_cache_legacy"
        )
        if self.scorer_assets and not isinstance(self.scorer_asset_manifest, dict):
            raise RuntimeError("staged scorer asset manifest is missing")

    @classmethod
    def asset_sources(cls, **params):
        """Return named local files needed to construct this scorer."""
        return {}

    def asset_path(self, key, fallback):
        if self.scorer_assets:
            path = self.scorer_assets.get(key)
            if not isinstance(path, str) or not path:
                raise RuntimeError(f"staged scorer asset {key!r} is missing")
            return path
        return fallback

    def asset_revision(self, key):
        value = self.scorer_asset_revisions.get(key)
        return value if isinstance(value, str) and value else None

    def asset_provenance_parameters(self):
        """Bind the loading mode and copied-file inventory into score provenance."""
        return {
            "asset_loading": self.asset_loading_mode,
            "asset_stage": copy.deepcopy(self.scorer_asset_manifest),
        }

    @classmethod
    def weights_status(cls, **params):
        """Return (ready: bool, message: str). Override to validate OFFLINE weights
        before instantiation so score.py can skip a metric cleanly instead of crashing."""
        return True, ""

    def score_image(self, image, prompt):
        """Return {column: float, ...} for one PIL.Image + prompt string."""
        raise NotImplementedError

    def provenance_metadata(self):
        """Describe models, assets, preprocessing, and effective parameters."""
        raise NotImplementedError
