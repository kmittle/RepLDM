"""LAION aesthetic predictor — CLIP ViT-L/14 image embedding -> MLP, score ~[1,10].
Decoupled (OpenAI `clip` + the public improved-aesthetic-predictor MLP weights in
~/.cache/aesthetic). §13.5: CLIP-L/14@224, correlated with the other CLIP-family
metrics; an aesthetic witness, NOT an independent detail witness.
"""
import os

import torch
import torch.nn as nn

from scorer_provenance import checkpoint_file_record, describe_preprocess
from .base import Scorer, register_metric

CLIP_CACHE = os.path.expanduser("~/.cache/clip")
AESTHETIC_PATH = os.path.expanduser("~/.cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth")


class _AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16), nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


@register_metric("aesthetic")
class AestheticScorer(Scorer):
    OUTPUT_KEYS = (("aesthetic", "higher"),)
    PROVENANCE_PACKAGES = ("openai-clip", "Pillow", "torch", "torchvision")

    @classmethod
    def asset_sources(cls, **p):
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

    def __init__(self, device="cuda", **p):
        super().__init__(device, **p)
        import clip
        self.clip_path = self.asset_path(
            "clip_checkpoint", os.path.join(CLIP_CACHE, "ViT-L-14.pt")
        )
        self.aesthetic_path = self.asset_path(
            "aesthetic_checkpoint", AESTHETIC_PATH
        )
        self.model, self.preprocess = clip.load(
            self.clip_path, device=device, download_root=CLIP_CACHE
        )
        self.model.eval()
        self.mlp = _AestheticMLP().to(device)
        self.mlp.load_state_dict(
            torch.load(self.aesthetic_path, map_location="cpu")
        )
        self.mlp.eval()

    @classmethod
    def weights_status(cls, **p):
        miss = [x for x in [os.path.join(CLIP_CACHE, "ViT-L-14.pt"), AESTHETIC_PATH] if not os.path.exists(x)]
        return (not miss), ("" if not miss else f"missing {miss}")

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
            "supporting_sources": [],
        }

    @torch.no_grad()
    def score_image(self, image, prompt):
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return {"aesthetic": float(self.mlp(feat).item())}
