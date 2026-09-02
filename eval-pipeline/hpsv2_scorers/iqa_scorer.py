"""TOPIQ-NR scorer with explicit TOPIQ and ResNet-50 checkpoints."""
import os

from pathlib import Path

from scorer_provenance import (
    checkpoint_file_record,
    package_python_source_tree_record,
    resolved_hf_revision,
    source_file_record,
)
import scorers.iqa_scorer as legacy_iqa
from scorers.iqa_scorer import IQAScorer as LegacyIQAScorer, TOPIQ_CHECKPOINT
from .base import StagedScorerMixin, register_metric


TOPIQ_REPOSITORY = "chaofengc/IQA-PyTorch-Weights"
TOPIQ_FILENAME = os.path.basename(TOPIQ_CHECKPOINT)
TIMM_REPOSITORY = "timm/resnet50.a1_in1k"
TIMM_FILENAME = "model.safetensors"


def _local_backbone_path():
    """Resolve the cached timm file without permitting a network lookup."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        TIMM_REPOSITORY,
        TIMM_FILENAME,
        local_files_only=True,
    )


def _is_nonempty_file(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


@register_metric("iqa")
class IQAScorer(StagedScorerMixin, LegacyIQAScorer):
    PROVENANCE_PACKAGES = (*LegacyIQAScorer.PROVENANCE_PACKAGES, "safetensors")

    @classmethod
    def asset_sources(cls, **params):
        backbone = _local_backbone_path()
        return {
            "topiq_checkpoint": {
                "path": TOPIQ_CHECKPOINT,
                "staged_name": TOPIQ_FILENAME,
                "revision": None,
            },
            "resnet50_backbone": {
                "path": backbone,
                "staged_name": "resnet50.a1_in1k.safetensors",
                "revision": resolved_hf_revision(backbone),
            },
        }

    @classmethod
    def weights_status(cls, **params):
        """Validate every file consumed by the staged TOPIQ implementation."""
        try:
            import pyiqa  # noqa: F401
            import timm  # noqa: F401
            from huggingface_hub import hf_hub_download  # noqa: F401
        except Exception as exc:
            return False, f"pyiqa, timm, or huggingface_hub is not installed ({exc})"
        if not _is_nonempty_file(TOPIQ_CHECKPOINT):
            return False, f"missing TOPIQ-NR checkpoint {TOPIQ_CHECKPOINT}"
        try:
            backbone = _local_backbone_path()
        except Exception as exc:
            return False, f"TOPIQ ResNet-50 backbone is not cached ({exc})"
        if not _is_nonempty_file(backbone):
            return False, f"missing TOPIQ ResNet-50 backbone {backbone}"
        return True, ""

    def __init__(self, device="cuda", **params):
        self._init_staged_scorer(device=device, **params)
        import pyiqa
        import timm

        self.topiq_path = self.asset_path("topiq_checkpoint")
        self.backbone_path = self.asset_path("resnet50_backbone")
        original_create_model = timm.create_model

        def create_staged_backbone(model_name, *args, **kwargs):
            if model_name != "resnet50":
                raise RuntimeError(f"unexpected TOPIQ backbone {model_name!r}")
            kwargs["pretrained"] = False
            kwargs.pop("checkpoint_path", None)
            model = original_create_model(model_name, *args, **kwargs)
            incompatible = timm.models.load_checkpoint(
                model,
                self.backbone_path,
                strict=False,
            )
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            if missing or unexpected != {"fc.bias", "fc.weight"}:
                raise RuntimeError(
                    "staged TOPIQ backbone checkpoint keys differ: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            return model

        timm.create_model = create_staged_backbone
        try:
            self.metric = pyiqa.create_metric(
                "topiq_nr",
                device=device,
                pretrained_model_path=self.topiq_path,
            )
        finally:
            timm.create_model = original_create_model

    def provenance_metadata(self):
        topiq = checkpoint_file_record(
            self.topiq_path,
            role="topiq_nr_checkpoint",
            filename=TOPIQ_FILENAME,
            repository_id="chaofengc/IQA-PyTorch",
        )
        backbone = checkpoint_file_record(
            self.backbone_path,
            role="resnet50_backbone",
            filename="model.safetensors",
            repository_id=TIMM_REPOSITORY,
            revision=self.asset_revision("resnet50_backbone")
            or resolved_hf_revision(self.backbone_path),
        )
        return {
            "models": [
                {
                    "identifier": "pyiqa:topiq_nr",
                    "repository_id": "chaofengc/IQA-PyTorch",
                    "revision": None,
                },
                {
                    "identifier": "resnet50.a1_in1k",
                    "repository_id": TIMM_REPOSITORY,
                    "revision": backbone["revision"],
                },
            ],
            "checkpoint_files": [topiq, backbone],
            "preprocessing": {
                "input": "PIL RGB",
                "conversion": "torchvision.transforms.functional.pil_to_tensor",
                "value_range": [0.0, 1.0],
                "spatial_resolution": "native",
                "resize": None,
            },
            "parameters": {
                "metric": "topiq_nr",
                **self.asset_provenance_parameters(),
            },
            "supporting_sources": [
                package_python_source_tree_record(
                    "pyiqa", label="pyiqa_python_source_tree"
                ),
                package_python_source_tree_record(
                    "timm", label="timm_python_source_tree"
                ),
                source_file_record(
                    legacy_iqa.__file__,
                    label="inherited_score_implementation",
                    root=Path(__file__).resolve().parents[1],
                    module=legacy_iqa.__name__,
                )
            ],
        }
