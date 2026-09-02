"""ImageReward scorer loaded from copied source, weights, and tokenizer files."""
import importlib.machinery
import os
from pathlib import Path
import sys
import types

from scorer_provenance import (
    checkpoint_file_record,
    describe_preprocess,
    git_revision,
    loaded_python_source_records,
    resolved_hf_revision,
    source_file_record,
)
import scorers.imagereward_scorer as legacy_imagereward
from scorers.imagereward_scorer import (
    IMAGEREWARD_SRC,
    IR_CACHE,
    ImageRewardScorer as LegacyImageRewardScorer,
)
from .base import StagedScorerMixin, register_metric


BERT_REPOSITORY = "bert-base-uncased"
BERT_REQUIRED_FILES = (
    "vocab.txt",
    "tokenizer_config.json",
    "tokenizer.json",
    "config.json",
)


def _local_tokenizer_paths():
    """Return every BERT file used by the staged ImageReward import."""
    from huggingface_hub import hf_hub_download

    return {
        filename: hf_hub_download(
            BERT_REPOSITORY,
            filename,
            local_files_only=True,
        )
        for filename in BERT_REQUIRED_FILES
    }


def _is_nonempty_file(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _setup_imagereward_imports(source_parent):
    if "datasets" not in sys.modules:
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = lambda *args, **kwargs: None
        datasets.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
        sys.modules["datasets"] = datasets
    try:
        import transformers.modeling_utils as modeling_utils
        from transformers.pytorch_utils import (
            apply_chunking_to_forward,
            find_pruneable_heads_and_indices,
            prune_linear_layer,
        )

        for name, function in (
            ("apply_chunking_to_forward", apply_chunking_to_forward),
            ("find_pruneable_heads_and_indices", find_pruneable_heads_and_indices),
            ("prune_linear_layer", prune_linear_layer),
        ):
            if not hasattr(modeling_utils, name):
                setattr(modeling_utils, name, function)
    except Exception:
        pass
    if source_parent not in sys.path:
        sys.path.insert(0, source_parent)


def _purge_imagereward_modules():
    for name in list(sys.modules):
        if name == "ImageReward" or name.startswith("ImageReward."):
            del sys.modules[name]


@register_metric("imagereward")
class ImageRewardScorer(StagedScorerMixin, LegacyImageRewardScorer):
    @classmethod
    def asset_sources(cls, **params):
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
            for path in sorted(str(value) for value in Path(package_root).rglob("*.py"))
        }
        for filename, path in _local_tokenizer_paths().items():
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

    @classmethod
    def weights_status(cls, **params):
        """Validate the complete checkpoint, source, and tokenizer closure."""
        required = {
            "ImageReward checkpoint": os.path.join(IR_CACHE, "ImageReward.pt"),
            "ImageReward media config": os.path.join(IR_CACHE, "med_config.json"),
        }
        missing = [
            f"{label}: {path}"
            for label, path in required.items()
            if not _is_nonempty_file(path)
        ]
        if missing:
            return False, "missing " + "; ".join(missing)
        if not os.path.isdir(IMAGEREWARD_SRC):
            return False, f"ImageReward source checkout is missing: {IMAGEREWARD_SRC}"
        package_root = os.path.join(IMAGEREWARD_SRC, "ImageReward")
        if not os.path.isfile(os.path.join(package_root, "__init__.py")):
            return False, f"ImageReward package source is missing: {package_root}"
        try:
            import fairscale  # noqa: F401
            from transformers import BertTokenizer
        except Exception as exc:
            return False, f"fairscale or transformers is not installed ({exc})"
        try:
            tokenizer_paths = _local_tokenizer_paths()
        except Exception as exc:
            return False, f"bert-base-uncased tokenizer assets are not cached ({exc})"
        missing = [
            f"{filename}: {path}"
            for filename, path in tokenizer_paths.items()
            if not _is_nonempty_file(path)
        ]
        if missing:
            return False, "missing BERT tokenizer assets: " + "; ".join(missing)
        try:
            BertTokenizer.from_pretrained(
                BERT_REPOSITORY,
                local_files_only=True,
            )
        except Exception as exc:
            return False, f"bert-base-uncased tokenizer is invalid ({exc})"
        return True, ""

    def __init__(self, device="cuda", patch_crops=5, no_patch_ir=False, **params):
        self._init_staged_scorer(
            device=device,
            patch_crops=patch_crops,
            no_patch_ir=no_patch_ir,
            **params,
        )
        self.patch_crops = patch_crops
        self.no_patch_ir = no_patch_ir
        self.checkpoint_path = self.asset_path("imagereward_checkpoint")
        self.med_config_path = self.asset_path("imagereward_med_config")
        package_init = self.asset_path("source::__init__.py")
        self.imagereward_source_root = os.path.dirname(package_init)
        source_parent = os.path.dirname(self.imagereward_source_root)
        tokenizer_path = self.asset_path("bert_tokenizer::vocab.txt")
        self.tokenizer_root = os.path.dirname(tokenizer_path)
        self.imagereward_revision = self.asset_revision("source::__init__.py")
        _purge_imagereward_modules()
        _setup_imagereward_imports(source_parent)

        from transformers import BertTokenizer

        original_from_pretrained = BertTokenizer.from_pretrained
        had_override = "from_pretrained" in BertTokenizer.__dict__
        original_override = BertTokenizer.__dict__.get("from_pretrained")

        def staged_from_pretrained(_name, *args, **kwargs):
            kwargs["local_files_only"] = True
            return original_from_pretrained(self.tokenizer_root, *args, **kwargs)

        BertTokenizer.from_pretrained = staticmethod(staged_from_pretrained)
        try:
            import ImageReward

            loaded_root = Path(ImageReward.__file__).resolve().parent
            expected_root = Path(self.imagereward_source_root).resolve()
            if loaded_root != expected_root:
                raise RuntimeError("ImageReward was not imported from staged source")
            self.model = ImageReward.load(
                self.checkpoint_path,
                device=device,
                med_config=self.med_config_path,
            )
        finally:
            if had_override:
                BertTokenizer.from_pretrained = original_override
            else:
                delattr(BertTokenizer, "from_pretrained")

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
            path = self.asset_path(f"bert_tokenizer::{filename}")
            tokenizer_assets.append(
                checkpoint_file_record(
                    path,
                    role=f"bert_tokenizer_{role}",
                    filename=filename,
                    repository_id=BERT_REPOSITORY,
                    revision=self.asset_revision(f"bert_tokenizer::{filename}")
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
                    "repository_id": BERT_REPOSITORY,
                    "revision": tokenizer_revision,
                },
            ],
            "checkpoint_files": [checkpoint, med_config, *tokenizer_assets],
            "preprocessing": {
                "global_image_transform": describe_preprocess(self.model.preprocess),
                "text_tokenizer": {
                    "identifier": "bert-base-uncased",
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": 35,
                    "added_tokens": ["[DEC]", "[ENC]"],
                },
                "patches": {
                    "size": 224,
                    "locations": [
                        "center",
                        "top_left",
                        "top_right",
                        "bottom_left",
                        "bottom_right",
                    ],
                    "small_image_resize": "PIL.Image.Resampling.BICUBIC",
                },
            },
            "parameters": {
                "patch_crops": self.patch_crops,
                "no_patch_ir": self.no_patch_ir,
                **self.asset_provenance_parameters(),
            },
            "supporting_sources": [
                *loaded_python_source_records(
                    self.imagereward_source_root,
                    label="imagereward_loaded_source",
                ),
                source_file_record(
                    legacy_imagereward.__file__,
                    label="inherited_score_implementation",
                    root=Path(__file__).resolve().parents[1],
                    module=legacy_imagereward.__name__,
                ),
            ],
        }
