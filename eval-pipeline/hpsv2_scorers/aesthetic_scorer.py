"""Aesthetic scorer that consumes only copied private checkpoints."""
import os

import torch

from scorer_provenance import (
    checkpoint_file_record,
    describe_preprocess,
    package_python_source_tree_record,
    source_file_record,
)
import scorers.aesthetic_scorer as legacy_aesthetic
from scorers.aesthetic_scorer import (
    AESTHETIC_PATH,
    CLIP_CACHE,
    AestheticScorer as LegacyAestheticScorer,
    _AestheticMLP,
)
from .base import StagedScorerMixin, register_metric


@register_metric("aesthetic")
class AestheticScorer(StagedScorerMixin, LegacyAestheticScorer):
    @classmethod
    def asset_sources(cls, **params):
        return {
            "clip_checkpoint": {
                "path": os.path.join(CLIP_CACHE, "ViT-L-14.pt"),
                "staged_name": "ViT-L-14.pt",
                "revision": None,
            },
            "aesthetic_checkpoint": {
                "path": AESTHETIC_PATH,
                "staged_name": os.path.basename(AESTHETIC_PATH),
                "revision": None,
            },
        }

    def __init__(self, device="cuda", **params):
        self._init_staged_scorer(device=device, **params)
        import clip

        self.clip_path = self.asset_path("clip_checkpoint")
        self.aesthetic_path = self.asset_path("aesthetic_checkpoint")
        self.model, self.preprocess = clip.load(self.clip_path, device=device)
        self.model.eval()
        self.mlp = _AestheticMLP().to(device)
        self.mlp.load_state_dict(torch.load(self.aesthetic_path, map_location="cpu"))
        self.mlp.eval()

    def provenance_metadata(self):
        return {
            "models": [
                {
                    "identifier": "openai-clip:ViT-L/14",
                    "repository_id": "openai/CLIP",
                    "revision": None,
                },
                {
                    "identifier": "laion-aesthetic-predictor-v2",
                    "repository_id": "christophschuhmann/improved-aesthetic-predictor",
                    "revision": None,
                },
            ],
            "checkpoint_files": [
                checkpoint_file_record(
                    self.clip_path,
                    role="clip_checkpoint",
                    filename="ViT-L-14.pt",
                    repository_id="openai/CLIP",
                ),
                checkpoint_file_record(
                    self.aesthetic_path,
                    role="aesthetic_mlp_checkpoint",
                    filename=os.path.basename(AESTHETIC_PATH),
                    repository_id="christophschuhmann/improved-aesthetic-predictor",
                ),
            ],
            "preprocessing": {
                "image_transform": describe_preprocess(self.preprocess),
                "feature_normalization": "l2",
            },
            "parameters": {
                "clip_model": "ViT-L/14",
                "mlp_dimensions": [768, 1024, 128, 64, 16, 1],
                **self.asset_provenance_parameters(),
            },
            "supporting_sources": [
                package_python_source_tree_record(
                    "clip", label="openai_clip_python_source_tree"
                ),
                source_file_record(
                    legacy_aesthetic.__file__,
                    label="inherited_score_implementation",
                    root=os.path.dirname(os.path.dirname(__file__)),
                    module=legacy_aesthetic.__name__,
                )
            ],
        }
