"""Native-resolution no-reference image quality assessment."""
import os

import torch
from torchvision.transforms.functional import pil_to_tensor

from scorer_provenance import checkpoint_file_record, hf_checkpoint_file_record
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

    def __init__(self, device="cuda", **params):
        super().__init__(device, **params)
        import pyiqa

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
            TOPIQ_CHECKPOINT,
            role="topiq_nr_checkpoint",
            filename=os.path.basename(TOPIQ_CHECKPOINT),
            repository_id="chaofengc/IQA-PyTorch",
        )
        backbone = hf_checkpoint_file_record(
            "timm/resnet50.a1_in1k",
            "model.safetensors",
            role="resnet50_backbone",
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
            "parameters": {"metric": "topiq_nr"},
            "supporting_sources": [],
        }

    @torch.no_grad()
    def score_image(self, image, prompt):
        tensor = pil_to_tensor(image).float().div_(255).unsqueeze(0).to(self.device)
        return {"topiq_nr": float(self.metric(tensor).item())}
