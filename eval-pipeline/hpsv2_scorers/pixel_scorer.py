"""Private wrapper for the weightless pixel witnesses."""

import copy

from scorers.pixel_scorer import PixelScorer as LegacyPixelScorer

from .base import StagedScorerMixin, register_metric


@register_metric("pixel")
class PixelScorer(StagedScorerMixin, LegacyPixelScorer):
    """Bind the private registry and complete legacy implementation MRO."""

    def __init__(
        self,
        device="cuda",
        scorer_assets=None,
        scorer_asset_revisions=None,
        scorer_asset_manifest=None,
        **params,
    ):
        if scorer_assets or scorer_asset_revisions:
            raise RuntimeError("weightless pixel scorer received staged weights")
        if not isinstance(scorer_asset_manifest, dict):
            raise RuntimeError("staged pixel scorer manifest is missing")
        super().__init__(device=device, **params)
        self.scorer_assets = {}
        self.scorer_asset_revisions = {}
        self.scorer_asset_manifest = copy.deepcopy(scorer_asset_manifest)

    def provenance_metadata(self):
        payload = super().provenance_metadata()
        payload["parameters"].update(self.asset_provenance_parameters())
        return payload
