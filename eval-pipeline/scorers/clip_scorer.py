"""CLIP-Score — alignment witness. Decoupled (OpenAI `clip` upstream pkg).

CANONICAL Hessel et al. 2021: clipscore = w * mean(max(cos(E_img, E_txt), 0)),
w=2.5 with ViT-B/32 — we implement it CORRECTLY (Sana drops the max(·,0) ReLU clamp
and uses a non-standard 100×cos@ViT-L/14, so its "CLIP-Score" is not paper-comparable).
We also emit the raw mean cosine, which is the cleanest scale-free trend for a
guidance sweep. §13.5: CLIP@224 is correlated with saturation — an alignment witness,
NOT an independent detail witness.
"""
import os

import torch

from scorer_provenance import checkpoint_file_record, describe_preprocess
from .base import Scorer, register_metric

CLIP_CACHE = os.path.expanduser("~/.cache/clip")
_MODEL_FILE = {"ViT-B/32": "ViT-B-32.pt", "ViT-L/14": "ViT-L-14.pt", "ViT-B/16": "ViT-B-16.pt"}


@register_metric("clip")
class ClipScorer(Scorer):
    OUTPUT_KEYS = (("clip_cosine", "higher"), ("clipscore", "higher"))
    PROVENANCE_PACKAGES = ("openai-clip", "Pillow", "torch", "torchvision")

    def __init__(self, device="cuda", clip_model="ViT-B/32", clipscore_w=2.5, **p):
        super().__init__(
            device, clip_model=clip_model, clipscore_w=clipscore_w, **p
        )
        import clip
        self.clip = clip
        self.clip_model = clip_model
        self.w = clipscore_w
        self.model, self.preprocess = clip.load(clip_model, device=device, download_root=CLIP_CACHE)
        self.model.eval()

    @classmethod
    def weights_status(cls, clip_model="ViT-B/32", **p):
        f = os.path.join(CLIP_CACHE, _MODEL_FILE.get(clip_model, "ViT-B-32.pt"))
        ok = os.path.exists(f)
        return ok, ("" if ok else f"missing {f}")

    def provenance_metadata(self):
        filename = _MODEL_FILE[self.clip_model]
        model_registry = getattr(
            getattr(self.clip, "clip", None), "_MODELS", {}
        )
        artifact_uri = model_registry.get(self.clip_model)
        return {
            "models": [
                {
                    "identifier": f"openai-clip:{self.clip_model}",
                    "repository_id": "openai/CLIP",
                    "revision": None,
                    "artifact_uri": artifact_uri,
                }
            ],
            "checkpoint_files": [
                checkpoint_file_record(
                    os.path.join(CLIP_CACHE, filename),
                    role="clip_checkpoint",
                    filename=filename,
                    repository_id="openai/CLIP",
                    artifact_uri=artifact_uri,
                )
            ],
            "preprocessing": {
                "image_transform": describe_preprocess(self.preprocess),
                "text_tokenizer": {
                    "model": self.clip_model,
                    "context_length": 77,
                    "truncate": True,
                },
                "feature_normalization": "l2",
                "clipscore_relu": True,
            },
            "parameters": {
                "clip_model": self.clip_model,
                "clipscore_w": self.w,
            },
            "supporting_sources": [],
        }

    @torch.no_grad()
    def score_image(self, image, prompt):
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        tok = self.clip.tokenize([prompt], truncate=True).to(self.device)
        fi = self.model.encode_image(img)
        ft = self.model.encode_text(tok)
        fi = fi / fi.norm(dim=-1, keepdim=True)
        ft = ft / ft.norm(dim=-1, keepdim=True)
        cos = float((fi * ft).sum(-1).item())
        return {"clip_cosine": cos, "clipscore": self.w * max(cos, 0.0)}
