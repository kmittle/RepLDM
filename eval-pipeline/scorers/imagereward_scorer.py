"""ImageReward scorer: global-IR (full image, 224 downsample) + patch-IR (native 224 crops).

Decoupled: imports ImageReward from the SOURCE checkout via the documented shim
(datasets stub + transformers symbol shim) — NOT the broken pip `image-reward`, and
NOT from Sana. patch-IR preserves native-resolution local crops as a diagnostic;
it is not treated as an independently validated perceptual-detail metric.
"""
import importlib.machinery
import os
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

from scorer_provenance import (
    checkpoint_file_record,
    describe_preprocess,
    git_revision,
    loaded_python_source_records,
    resolved_hf_revision,
)
from .base import Scorer, register_metric

IMAGEREWARD_SRC = "/mnt/miah204/bycao/ImageReward"
IR_CACHE = os.path.expanduser("~/.cache/ImageReward")


def _setup_imagereward_imports(source_parent=IMAGEREWARD_SRC):
    if "datasets" not in sys.modules:
        d = types.ModuleType("datasets")
        d.load_dataset = lambda *a, **k: None
        d.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
        sys.modules["datasets"] = d
    try:
        import transformers.modeling_utils as mu
        from transformers.pytorch_utils import (apply_chunking_to_forward,
                                                find_pruneable_heads_and_indices,
                                                prune_linear_layer)
        for n, f in [("apply_chunking_to_forward", apply_chunking_to_forward),
                     ("find_pruneable_heads_and_indices", find_pruneable_heads_and_indices),
                     ("prune_linear_layer", prune_linear_layer)]:
            if not hasattr(mu, n):
                setattr(mu, n, f)
    except Exception:
        pass
    if source_parent not in sys.path:
        sys.path.insert(0, source_parent)


def _purge_imagereward_modules():
    for name in list(sys.modules):
        if name == "ImageReward" or name.startswith("ImageReward."):
            del sys.modules[name]


def _native_crops(img, size=224):
    """Deterministic center + 4-corner crops at native resolution (no downsample)."""
    W, H = img.size
    if W < size or H < size:
        return [img.resize((size, size), Image.BICUBIC)]
    xs = [0, W - size, (W - size) // 2]
    ys = [0, H - size, (H - size) // 2]
    coords = [(xs[2], ys[2]), (xs[0], ys[0]), (xs[1], ys[0]), (xs[0], ys[1]), (xs[1], ys[1])]
    return [img.crop((x, y, x + size, y + size)) for (x, y) in coords]


@register_metric("imagereward")
class ImageRewardScorer(Scorer):
    OUTPUT_KEYS = (("imagereward", "higher"), ("patch_ir_mean", "higher"),
                   ("patch_ir_std", "witness"), ("patch_ir_n", "witness"))
    PROVENANCE_PACKAGES = (
        "fairscale",
        "numpy",
        "Pillow",
        "timm",
        "torch",
        "torchvision",
        "transformers",
    )

    @classmethod
    def asset_sources(cls, **p):
        from huggingface_hub import hf_hub_download

        revision = git_revision(IMAGEREWARD_SRC)
        package_root = os.path.join(IMAGEREWARD_SRC, "ImageReward")
        sources = {
            f"source::{os.path.relpath(path, package_root)}": {
                "path": path,
                "staged_name": os.path.join(
                    "ImageReward", os.path.relpath(path, package_root)
                ),
                "revision": revision,
            }
            for path in sorted(
                str(value)
                for value in Path(package_root).rglob("*.py")
            )
        }
        tokenizer_files = (
            "vocab.txt",
            "tokenizer_config.json",
            "tokenizer.json",
            "config.json",
        )
        for filename in tokenizer_files:
            path = hf_hub_download(
                "bert-base-uncased", filename, local_files_only=True
            )
            sources[f"bert_tokenizer::{filename}"] = {
                "path": path,
                "staged_name": os.path.join("bert-base-uncased", filename),
                "revision": resolved_hf_revision(path),
            }
        sources.update(
            {
                "imagereward_checkpoint": {
                    "path": os.path.join(IR_CACHE, "ImageReward.pt"),
                    "staged_name": "ImageReward.pt",
                    "revision": None,
                },
                "imagereward_med_config": {
                    "path": os.path.join(IR_CACHE, "med_config.json"),
                    "staged_name": "med_config.json",
                    "revision": None,
                },
            }
        )
        return sources

    def __init__(self, device="cuda", patch_crops=5, no_patch_ir=False, **p):
        super().__init__(
            device, patch_crops=patch_crops, no_patch_ir=no_patch_ir, **p
        )
        self.patch_crops = patch_crops
        self.no_patch_ir = no_patch_ir
        self.checkpoint_path = self.asset_path(
            "imagereward_checkpoint", os.path.join(IR_CACHE, "ImageReward.pt")
        )
        self.med_config_path = self.asset_path(
            "imagereward_med_config", os.path.join(IR_CACHE, "med_config.json")
        )
        if self.scorer_assets:
            package_init = self.asset_path(
                "source::__init__.py",
                os.path.join(IMAGEREWARD_SRC, "ImageReward", "__init__.py"),
            )
            self.imagereward_source_root = os.path.dirname(package_init)
            source_parent = os.path.dirname(self.imagereward_source_root)
            tokenizer_path = self.asset_path(
                "bert_tokenizer::vocab.txt", ""
            )
            self.tokenizer_root = os.path.dirname(tokenizer_path)
            self.imagereward_revision = self.asset_revision(
                "source::__init__.py"
            )
        else:
            self.imagereward_source_root = os.path.join(
                IMAGEREWARD_SRC, "ImageReward"
            )
            source_parent = IMAGEREWARD_SRC
            self.tokenizer_root = "bert-base-uncased"
            self.imagereward_revision = git_revision(IMAGEREWARD_SRC)
        if self.scorer_assets:
            _purge_imagereward_modules()
        _setup_imagereward_imports(source_parent)

        from transformers import BertTokenizer

        original_from_pretrained = BertTokenizer.from_pretrained
        had_override = "from_pretrained" in BertTokenizer.__dict__
        original_override = BertTokenizer.__dict__.get("from_pretrained")

        def staged_from_pretrained(_name, *args, **kwargs):
            kwargs["local_files_only"] = True
            return original_from_pretrained(self.tokenizer_root, *args, **kwargs)

        if self.scorer_assets:
            BertTokenizer.from_pretrained = staticmethod(staged_from_pretrained)
        try:
            import ImageReward

            if self.scorer_assets:
                loaded_root = Path(ImageReward.__file__).resolve().parent
                expected_root = Path(self.imagereward_source_root).resolve()
                if loaded_root != expected_root:
                    raise RuntimeError(
                        "ImageReward was not imported from the staged source"
                    )

            self.model = ImageReward.load(
                self.checkpoint_path,
                device=device,
                med_config=self.med_config_path,
            )
        finally:
            if self.scorer_assets:
                if had_override:
                    BertTokenizer.from_pretrained = original_override
                else:
                    delattr(BertTokenizer, "from_pretrained")

    @classmethod
    def weights_status(cls, **p):
        required = [
            os.path.join(IR_CACHE, "ImageReward.pt"),
            os.path.join(IR_CACHE, "med_config.json"),
        ]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            return False, f"missing {missing}"
        try:
            import fairscale  # noqa: F401
        except ImportError:
            return False, "fairscale is not installed"
        try:
            from transformers import BertTokenizer
            BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        except Exception as exc:
            return False, f"bert-base-uncased tokenizer is not cached ({exc})"
        if not os.path.isdir(IMAGEREWARD_SRC):
            return False, f"ImageReward source checkout is missing: {IMAGEREWARD_SRC}"
        return True, ""

    def provenance_metadata(self):
        checkpoint = checkpoint_file_record(
            self.checkpoint_path,
            role="imagereward_checkpoint",
            filename="ImageReward.pt",
            repository_id="THUDM/ImageReward",
            revision=self.asset_revision("imagereward_checkpoint"),
        )
        med_config = checkpoint_file_record(
            self.med_config_path,
            role="imagereward_med_config",
            filename="med_config.json",
            repository_id="THUDM/ImageReward",
            revision=self.asset_revision("imagereward_med_config"),
        )
        tokenizer_assets = []
        for filename, role in (
            ("vocab.txt", "vocabulary"),
            ("tokenizer_config.json", "config"),
            ("tokenizer.json", "tokenizer"),
            ("config.json", "model_config"),
        ):
            if self.scorer_assets:
                path = self.asset_path(f"bert_tokenizer::{filename}", "")
            else:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    "bert-base-uncased", filename, local_files_only=True
                )
            tokenizer_assets.append(
                checkpoint_file_record(
                    path,
                    role=f"bert_tokenizer_{role}",
                    filename=filename,
                    repository_id="bert-base-uncased",
                    revision=self.asset_revision(
                        f"bert_tokenizer::{filename}"
                    )
                    or resolved_hf_revision(path),
                )
            )
        tokenizer_revision = tokenizer_assets[0]["revision"]
        return {
            "models": [
                {
                    "identifier": "ImageReward-v1.0",
                    "repository_id": "THUDM/ImageReward",
                    "revision": self.imagereward_revision,
                },
                {
                    "identifier": "bert-base-uncased",
                    "repository_id": "bert-base-uncased",
                    "revision": tokenizer_revision,
                },
            ],
            "checkpoint_files": [checkpoint, med_config, *tokenizer_assets],
            "preprocessing": {
                "global_image_transform": describe_preprocess(
                    self.model.preprocess
                ),
                "text_tokenizer": {
                    "identifier": "bert-base-uncased",
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": 35,
                    "added_tokens": ["[DEC]", "[ENC]"],
                },
                "patches": {
                    "size": 224,
                    "locations": ["center", "top_left", "top_right", "bottom_left", "bottom_right"],
                    "small_image_resize": "PIL.Image.Resampling.BICUBIC",
                },
            },
            "parameters": {
                "patch_crops": self.patch_crops,
                "no_patch_ir": self.no_patch_ir,
                **self.asset_provenance_parameters(),
            },
            "supporting_sources": loaded_python_source_records(
                self.imagereward_source_root,
                label="imagereward_loaded_source",
            ),
        }

    def score_image(self, image, prompt):
        rec = {"imagereward": float(self.model.score(prompt, image))}
        if not self.no_patch_ir:
            crops = _native_crops(image, 224)
            if self.patch_crops:
                crops = crops[:self.patch_crops]
            ps = [float(self.model.score(prompt, c)) for c in crops]
            rec["patch_ir_mean"] = float(np.mean(ps))
            rec["patch_ir_std"] = float(np.std(ps))
            rec["patch_ir_n"] = len(crops)
        return rec
