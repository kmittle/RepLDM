"""Native-resolution no-reference image quality assessment."""
import os

import torch
from torchvision.transforms.functional import pil_to_tensor

from scorer_provenance import (
    checkpoint_file_record,
    resolved_hf_revision,
)
from .base import Scorer, register_metric


TOPIQ_CHECKPOINT = os.path.expanduser(
    "~/.cache/torch/hub/pyiqa/cfanet_nr_koniq_res50-9a73138b.pth"
)


@register_metric("iqa")
class IQAScorer(Scorer):
    """TOPIQ-NR on the complete input image without 224px downsampling."""

    OUTPUT_KEYS = (("topiq_nr", "higher"),)
    PROVENANCE_PACKAGES = (
        "huggingface-hub",
        "Pillow",
        "pyiqa",
        "timm",
        "torch",
        "torchvision",
    )

    @classmethod
    def asset_sources(cls, **params):
        from huggingface_hub import hf_hub_download

        backbone = hf_hub_download(
            "timm/resnet50.a1_in1k",
            "model.safetensors",
            local_files_only=True,
        )
        return {
            "topiq_checkpoint": {
                "path": TOPIQ_CHECKPOINT,
                "staged_name": os.path.basename(TOPIQ_CHECKPOINT),
                "revision": None,
            },
            "resnet50_backbone": {
                "path": backbone,
                "staged_name": "resnet50.a1_in1k.safetensors",
                "revision": resolved_hf_revision(backbone),
            },
        }

    def __init__(self, device="cuda", **params):
        super().__init__(device, **params)
        import pyiqa
        self.topiq_path = self.asset_path(
            "topiq_checkpoint", TOPIQ_CHECKPOINT
        )
        if self.scorer_assets:
            self.backbone_path = self.asset_path("resnet50_backbone", "")
            import timm

            original_create_model = timm.create_model

            def create_staged_backbone(model_name, *args, **kwargs):
                if model_name != "resnet50":
                    raise RuntimeError(
                        f"unexpected TOPIQ backbone {model_name!r}"
                    )
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
                        f"missing={sorted(missing)}, "
                        f"unexpected={sorted(unexpected)}"
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
        else:
            from huggingface_hub import hf_hub_download

            self.backbone_path = hf_hub_download(
                "timm/resnet50.a1_in1k",
                "model.safetensors",
                local_files_only=True,
            )
            self.metric = pyiqa.create_metric("topiq_nr", device=device)

    @classmethod
    def weights_status(cls, **params):
        try:
            import pyiqa  # noqa: F401
            from huggingface_hub import hf_hub_download
        except ImportError:
            return False, "pyiqa or huggingface_hub is not installed"
        if not os.path.exists(TOPIQ_CHECKPOINT):
            return False, f"missing {TOPIQ_CHECKPOINT}"
        try:
            hf_hub_download(
                "timm/resnet50.a1_in1k", "model.safetensors", local_files_only=True
            )
        except Exception as exc:
            return False, f"TOPIQ ResNet-50 backbone is not cached ({exc})"
        return True, ""

    def provenance_metadata(self):
        topiq = checkpoint_file_record(
            self.topiq_path,
            role="topiq_nr_checkpoint",
            filename=os.path.basename(TOPIQ_CHECKPOINT),
            repository_id="chaofengc/IQA-PyTorch",
        )
        backbone = checkpoint_file_record(
            self.backbone_path,
            role="resnet50_backbone",
            filename="model.safetensors",
            repository_id="timm/resnet50.a1_in1k",
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
                    "repository_id": "timm/resnet50.a1_in1k",
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
            "supporting_sources": [],
        }

    @torch.no_grad()
    def score_image(self, image, prompt):
        tensor = pil_to_tensor(image).float().div_(255).unsqueeze(0).to(self.device)
        return {"topiq_nr": float(self.metric(tensor).item())}
