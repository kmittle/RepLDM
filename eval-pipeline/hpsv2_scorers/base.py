"""Private-asset support used only by the formal HPSv2 scorer."""
from __future__ import annotations

import copy


REGISTRY: dict = {}
NETWORK_ISOLATION_SCHEMA = "linux_seccomp_network_deny_v2"


def register_metric(name):
    def decorator(cls):
        cls.NAME = name
        REGISTRY[name] = cls
        return cls

    return decorator


class StagedScorerMixin:
    """Add pinned private assets without changing the shared scorer package."""

    def _init_staged_scorer(
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
        if not self.scorer_assets:
            raise RuntimeError("formal HPSv2 scorer assets are missing")
        if not isinstance(self.scorer_asset_manifest, dict):
            raise RuntimeError("staged scorer asset manifest is missing")

    @classmethod
    def asset_sources(cls, **params):
        return {}

    def asset_path(self, key):
        path = self.scorer_assets.get(key)
        if not isinstance(path, str) or not path:
            raise RuntimeError(f"staged scorer asset {key!r} is missing")
        return path

    def asset_revision(self, key):
        value = self.scorer_asset_revisions.get(key)
        return value if isinstance(value, str) and value else None

    def asset_provenance_parameters(self):
        return {
            "asset_loading": "pinned_private_stage_v1",
            "asset_stage": copy.deepcopy(self.scorer_asset_manifest),
            "network_isolation": NETWORK_ISOLATION_SCHEMA,
        }
