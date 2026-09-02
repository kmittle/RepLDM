"""HPSv2 scorer with explicit model, architecture, and tokenizer assets."""
import importlib.util
import json
import os
from pathlib import Path

import torch

from scorer_provenance import (
    checkpoint_file_record,
    describe_preprocess,
    package_python_source_tree_record,
    resolved_hf_revision,
    source_file_record,
)
import scorers.hps_scorer as legacy_hps
from scorers.hps_scorer import HPSScorer as LegacyHPSScorer
from .base import StagedScorerMixin, register_metric


def _hps_package_root():
    spec = importlib.util.find_spec("hpsv2")
    if spec is None or spec.origin is None:
        raise RuntimeError("hpsv2 package is unavailable")
    return Path(spec.origin).resolve().parent


@register_metric("hps")
class HPSScorer(StagedScorerMixin, LegacyHPSScorer):
    TOKENIZER_TRUNCATE = True
    PROVENANCE_PACKAGES = (
        "ftfy",
        "hpsv2",
        "huggingface-hub",
        "Pillow",
        "regex",
        "torch",
        "torchvision",
    )

    @classmethod
    def asset_sources(cls, **params):
        from huggingface_hub import hf_hub_download

        hps_checkpoint = hf_hub_download(
            "xswu/HPSv2",
            "HPS_v2.1_compressed.pt",
            local_files_only=True,
        )
        backbone = hf_hub_download(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "open_clip_pytorch_model.bin",
            local_files_only=True,
        )
        open_clip_root = _hps_package_root() / "src" / "open_clip"
        return {
            "hpsv2_checkpoint": {
                "path": hps_checkpoint,
                "staged_name": "HPS_v2.1_compressed.pt",
                "revision": resolved_hf_revision(hps_checkpoint),
            },
            "open_clip_backbone": {
                "path": backbone,
                "staged_name": "open_clip_pytorch_model.bin",
                "revision": resolved_hf_revision(backbone),
            },
            "tokenizer_vocabulary": {
                "path": str(open_clip_root / "bpe_simple_vocab_16e6.txt.gz"),
                "staged_name": "bpe_simple_vocab_16e6.txt.gz",
                "revision": None,
            },
            "model_config": {
                "path": str(open_clip_root / "model_configs" / "ViT-H-14.json"),
                "staged_name": "ViT-H-14.json",
                "revision": None,
            },
        }

    def __init__(self, device="cuda", **params):
        self._init_staged_scorer(device=device, **params)
        from hpsv2.src.open_clip import factory
        from hpsv2.src.open_clip.tokenizer import SimpleTokenizer

        self.checkpoint_path = self.asset_path("hpsv2_checkpoint")
        self.backbone_path = self.asset_path("open_clip_backbone")
        self.vocabulary_path = self.asset_path("tokenizer_vocabulary")
        self.model_config_path = self.asset_path("model_config")
        with open(self.model_config_path, encoding="utf-8") as handle:
            model_config = json.load(handle)
        if not isinstance(model_config, dict) or not all(
            key in model_config for key in ("embed_dim", "vision_cfg", "text_cfg")
        ):
            raise RuntimeError("staged HPS ViT-H-14 model config is invalid")
        factory._MODEL_CONFIGS = {"ViT-H-14": model_config}
        self.model, _, self.preprocess = factory.create_model_and_transforms(
            "ViT-H-14",
            pretrained=self.backbone_path,
            precision="amp",
            device=device,
            output_dict=True,
        )
        state = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state["state_dict"])
        self.simple_tokenizer = SimpleTokenizer(bpe_path=self.vocabulary_path)
        self.tokenizer = self._tokenize
        self.model = self.model.to(device).eval()

    def _tokenize(self, texts, context_length=77):
        if isinstance(texts, str):
            texts = [texts]
        start = self.simple_tokenizer.encoder["<start_of_text>"]
        end = self.simple_tokenizer.encoder["<end_of_text>"]
        all_tokens = [
            [start, *self.simple_tokenizer.encode(text), end]
            for text in texts
        ]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for index, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = end
            result[index, : len(tokens)] = torch.tensor(tokens)
        return result

    def provenance_metadata(self):
        hps_checkpoint = checkpoint_file_record(
            self.checkpoint_path,
            role="hpsv2_checkpoint",
            filename="HPS_v2.1_compressed.pt",
            repository_id="xswu/HPSv2",
            revision=self.asset_revision("hpsv2_checkpoint")
            or resolved_hf_revision(self.checkpoint_path),
        )
        backbone = checkpoint_file_record(
            self.backbone_path,
            role="open_clip_backbone",
            filename="open_clip_pytorch_model.bin",
            repository_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            revision=self.asset_revision("open_clip_backbone")
            or resolved_hf_revision(self.backbone_path),
        )
        vocabulary = checkpoint_file_record(
            self.vocabulary_path,
            role="tokenizer_vocabulary",
            filename="bpe_simple_vocab_16e6.txt.gz",
            repository_id="hpsv2",
        )
        model_config = checkpoint_file_record(
            self.model_config_path,
            role="open_clip_model_config",
            filename="ViT-H-14.json",
            repository_id="hpsv2",
        )
        return {
            "models": [
                {
                    "identifier": "HPS_v2.1_compressed",
                    "repository_id": "xswu/HPSv2",
                    "revision": hps_checkpoint["revision"],
                },
                {
                    "identifier": "ViT-H-14:laion2B-s32B-b79K",
                    "repository_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                    "revision": backbone["revision"],
                },
            ],
            "checkpoint_files": [
                hps_checkpoint,
                backbone,
                vocabulary,
                model_config,
            ],
            "preprocessing": {
                "image_transform": describe_preprocess(self.preprocess),
                "text_tokenizer": {
                    "architecture": "ViT-H-14",
                    "context_length": 77,
                    "truncate": self.TOKENIZER_TRUNCATE,
                    "implementation": "explicit_staged_simple_tokenizer",
                },
                "feature_normalization": "model_native_normalized_features",
            },
            "parameters": {
                "architecture": "ViT-H-14",
                "pretrained": "laion2B-s32B-b79K",
                "hps_version": "v2.1",
                **self.asset_provenance_parameters(),
            },
            "supporting_sources": [
                package_python_source_tree_record(
                    "hpsv2", label="hpsv2_python_source_tree"
                ),
                source_file_record(
                    legacy_hps.__file__,
                    label="inherited_score_implementation",
                    root=Path(__file__).resolve().parents[1],
                    module=legacy_hps.__name__,
                )
            ],
        }
