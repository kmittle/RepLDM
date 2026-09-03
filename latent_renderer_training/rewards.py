"""Tensor-native frozen rewards for latent-renderer training."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import torch
from torch import Tensor, nn


IMAGE_REWARD_PREPROCESS = {
    "schema": "repldm.imagereward_tensor_preprocess.v1",
    "input": "rgb_nchw_float_[0,1]",
    "resize": {
        "shorter_side": 224,
        "interpolation": "bicubic",
        "antialias": True,
    },
    "center_crop": [224, 224],
    "normalize_mean": [0.48145466, 0.4578275, 0.40821073],
    "normalize_std": [0.26862954, 0.26130258, 0.27577711],
    "text": {
        "padding": "max_length",
        "truncation": True,
        "max_length": 35,
    },
    "model_entrypoint": "score_gard",
}
IMAGE_REWARD_PREPROCESS_SHA256 = hashlib.sha256(
    json.dumps(
        IMAGE_REWARD_PREPROCESS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()


def _first_floating_parameter(module: nn.Module) -> Tensor:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter
    raise ValueError("ImageReward model has no floating-point parameters")


class ImageRewardTensorAdapter(nn.Module):
    """Expose ImageReward v1.0 without PIL conversion or gradient detachment."""

    def __init__(self, model: nn.Module, *, tokenizer: Any = None) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not callable(getattr(model, "score_gard", None)):
            raise TypeError("ImageReward model must expose score_gard")
        resolved_tokenizer = tokenizer
        if resolved_tokenizer is None:
            resolved_tokenizer = getattr(getattr(model, "blip", None), "tokenizer", None)
        if not callable(resolved_tokenizer):
            raise TypeError("ImageReward tokenizer must be callable")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        self.model = model
        # Tokenizers are intentionally not registered as modules.
        self.tokenizer = resolved_tokenizer
        self.preprocess_sha256 = IMAGE_REWARD_PREPROCESS_SHA256

    def _preprocess(self, image: Tensor) -> Tensor:
        if (
            not isinstance(image, Tensor)
            or image.ndim != 4
            or image.shape[1] != 3
            or not image.is_floating_point()
        ):
            raise ValueError("ImageReward input must be a floating RGB NCHW tensor")
        if not torch.isfinite(image).all():
            raise ValueError("ImageReward input contains non-finite pixels")
        detached_range = image.detach()
        if torch.any(detached_range < 0.0) or torch.any(detached_range > 1.0):
            raise ValueError("ImageReward input must stay in [0, 1]")
        from torchvision.transforms import (
            CenterCrop,
            Compose,
            InterpolationMode,
            Normalize,
            Resize,
        )

        transform = Compose(
            [
                Resize(
                    224,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                CenterCrop(224),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        parameter = _first_floating_parameter(self.model)
        return transform(image).to(device=parameter.device, dtype=parameter.dtype)

    def score_tensor(self, images: Tensor, prompts: Sequence[str]) -> Tensor:
        """Return one differentiable score per image."""
        prompt_values = tuple(prompts)
        if len(prompt_values) != images.shape[0] or any(
            not isinstance(prompt, str) or not prompt for prompt in prompt_values
        ):
            raise ValueError("ImageReward prompts must align with the image batch")
        parameter = _first_floating_parameter(self.model)
        tokens = self.tokenizer(
            list(prompt_values),
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        input_ids = getattr(tokens, "input_ids", None)
        attention_mask = getattr(tokens, "attention_mask", None)
        if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
            if isinstance(tokens, dict):
                input_ids = tokens.get("input_ids")
                attention_mask = tokens.get("attention_mask")
        if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
            raise RuntimeError("ImageReward tokenizer omitted input_ids or attention_mask")
        input_ids = input_ids.to(parameter.device)
        attention_mask = attention_mask.to(parameter.device)
        prepared = self._preprocess(images)
        scores = self.model.score_gard(input_ids, attention_mask, prepared)
        if not isinstance(scores, Tensor):
            raise TypeError("ImageReward score_gard must return a Tensor")
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if scores.ndim != 1 or scores.shape[0] != images.shape[0]:
            raise ValueError("ImageReward score_gard returned an invalid score batch")
        if not torch.isfinite(scores).all():
            raise RuntimeError("ImageReward returned non-finite scores")
        return scores

    def forward(self, images: Tensor, prompts: Sequence[str]) -> Tensor:
        return self.score_tensor(images, prompts)


__all__ = [
    "IMAGE_REWARD_PREPROCESS",
    "IMAGE_REWARD_PREPROCESS_SHA256",
    "ImageRewardTensorAdapter",
]
