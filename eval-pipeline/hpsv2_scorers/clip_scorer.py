"""CLIP scorer with an explicit checkpoint and tokenizer vocabulary."""
import importlib.util
import os
from pathlib import Path

import torch

from scorer_provenance import (
    checkpoint_file_record,
    describe_preprocess,
    package_python_source_tree_record,
)
from scorers.clip_scorer import (
    CLIP_CACHE,
    _MODEL_FILE,
    ClipScorer as LegacyClipScorer,
)
from .base import StagedScorerMixin, register_metric


def _clip_vocabulary_path():
    spec = importlib.util.find_spec("clip")
    if spec is None or spec.origin is None:
        raise RuntimeError("openai-clip package is unavailable")
    return str(Path(spec.origin).resolve().parent / "bpe_simple_vocab_16e6.txt.gz")


@register_metric("clip")
class ClipScorer(StagedScorerMixin, LegacyClipScorer):
    PROVENANCE_PACKAGES = (
        "ftfy",
        "openai-clip",
        "Pillow",
        "regex",
        "torch",
        "torchvision",
    )

    @classmethod
    def asset_sources(cls, clip_model="ViT-B/32", **params):
        filename = _MODEL_FILE.get(clip_model, "ViT-B-32.pt")
        return {
            "clip_checkpoint": {
                "path": os.path.join(CLIP_CACHE, filename),
                "staged_name": filename,
                "revision": None,
            },
            "tokenizer_vocabulary": {
                "path": _clip_vocabulary_path(),
                "staged_name": "bpe_simple_vocab_16e6.txt.gz",
                "revision": None,
            },
        }

    def __init__(self, device="cuda", clip_model="ViT-B/32", clipscore_w=2.5, **params):
        self._init_staged_scorer(
            device=device,
            clip_model=clip_model,
            clipscore_w=clipscore_w,
            **params,
        )
        import clip
        from clip.simple_tokenizer import SimpleTokenizer

        self.clip = clip
        self.clip_model = clip_model
        self.w = clipscore_w
        self.checkpoint_path = self.asset_path("clip_checkpoint")
        self.vocabulary_path = self.asset_path("tokenizer_vocabulary")
        self.model, self.preprocess = clip.load(self.checkpoint_path, device=device)
        self.tokenizer = SimpleTokenizer(bpe_path=self.vocabulary_path)
        self.model.eval()

    def _tokenize(self, text, context_length=77):
        start = self.tokenizer.encoder["<|startoftext|>"]
        end = self.tokenizer.encoder["<|endoftext|>"]
        tokens = [start, *self.tokenizer.encode(text), end]
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            tokens[-1] = end
        result = torch.zeros(1, context_length, dtype=torch.int)
        result[0, : len(tokens)] = torch.tensor(tokens)
        return result

    def provenance_metadata(self):
        filename = _MODEL_FILE[self.clip_model]
        model_registry = getattr(getattr(self.clip, "clip", None), "_MODELS", {})
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
                    self.checkpoint_path,
                    role="clip_checkpoint",
                    filename=filename,
                    repository_id="openai/CLIP",
                    artifact_uri=artifact_uri,
                ),
                checkpoint_file_record(
                    self.vocabulary_path,
                    role="clip_tokenizer_vocabulary",
                    filename="bpe_simple_vocab_16e6.txt.gz",
                    repository_id="openai/CLIP",
                ),
            ],
            "preprocessing": {
                "image_transform": describe_preprocess(self.preprocess),
                "text_tokenizer": {
                    "model": self.clip_model,
                    "context_length": 77,
                    "truncate": True,
                    "implementation": "explicit_staged_simple_tokenizer",
                },
                "feature_normalization": "l2",
                "clipscore_relu": True,
            },
            "parameters": {
                "clip_model": self.clip_model,
                "clipscore_w": self.w,
                **self.asset_provenance_parameters(),
            },
            "supporting_sources": [
                package_python_source_tree_record(
                    "clip", label="openai_clip_python_source_tree"
                )
            ],
        }

    @torch.no_grad()
    def score_image(self, image, prompt):
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tensor = self._tokenize(prompt).to(self.device)
        image_features = self.model.encode_image(image_tensor)
        text_features = self.model.encode_text(text_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cosine = float((image_features * text_features).sum(-1).item())
        return {
            "clip_cosine": cosine,
            "clipscore": self.w * max(cosine, 0.0),
        }
