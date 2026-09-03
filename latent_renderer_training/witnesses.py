"""Tensor-native independent witnesses for formal F0 evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import torch
from torch import Tensor, nn

from .operations import (
    LedgeredOperationExecutor,
    OperationContext,
    OperationReceipt,
    operation_output_sha256,
)
from .renderer import tensor_sha256


TOPIQ_NR_MODEL_ID = "pyiqa:topiq_nr"
TOPIQ_NR_ROLE = "independent_f0_witness"
TOPIQ_NR_PREPROCESS = {
    "schema": "repldm.topiq_nr_tensor_preprocess.v1",
    "input": "rgb_nchw_float_[0,1]",
    "color_space": "RGB",
    "spatial_resolution": "native",
    "resize": None,
    "crop": None,
    "normalization": None,
    "model_entrypoint": "pyiqa:topiq_nr",
}
TOPIQ_NR_PREPROCESS_SHA256 = hashlib.sha256(
    json.dumps(
        TOPIQ_NR_PREPROCESS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()


def _placement(module: nn.Module, images: Tensor) -> tuple[torch.device, torch.dtype]:
    floating = tuple(
        value
        for value in (*tuple(module.parameters()), *tuple(module.buffers()))
        if value.is_floating_point()
    )
    if not floating:
        return images.device, images.dtype
    devices = {value.device for value in floating}
    dtypes = {value.dtype for value in floating}
    if len(devices) != 1 or len(dtypes) != 1:
        raise RuntimeError("TOPIQ-NR has mixed device or floating-point precision")
    return next(iter(devices)), next(iter(dtypes))


class TopiqNrTensorWitness(nn.Module):
    """Run frozen TOPIQ-NR on native-resolution RGB tensors.

    A formal instance owns the runtime's operation executor.  Its calls are
    charged to the shared ``reward_forward`` budget, but their receipts carry
    a distinct model and role so they cannot be confused with ImageReward.
    """

    model_id = TOPIQ_NR_MODEL_ID
    role = TOPIQ_NR_ROLE
    preprocess_sha256 = TOPIQ_NR_PREPROCESS_SHA256

    def __init__(
        self,
        metric: nn.Module,
        *,
        operation_executor: Optional[LedgeredOperationExecutor] = None,
    ) -> None:
        super().__init__()
        if not isinstance(metric, nn.Module):
            raise TypeError("TOPIQ-NR metric must be a torch.nn.Module")
        if not callable(getattr(metric, "forward", None)):
            raise TypeError("TOPIQ-NR metric must be callable")
        if operation_executor is not None and not isinstance(
            operation_executor, LedgeredOperationExecutor
        ):
            raise TypeError("TOPIQ-NR requires a LedgeredOperationExecutor")
        metric.eval()
        for parameter in metric.parameters():
            parameter.requires_grad_(False)
        self.metric = metric
        self.operation_executor = operation_executor
        super().train(False)

    def train(self, mode: bool = True) -> "TopiqNrTensorWitness":
        if mode:
            raise RuntimeError("TOPIQ-NR is a frozen evaluation-only witness")
        super().train(False)
        return self

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if (
            not isinstance(images, Tensor)
            or images.ndim != 4
            or images.shape[0] <= 0
            or images.shape[1] != 3
            or images.shape[2] <= 0
            or images.shape[3] <= 0
            or not images.is_floating_point()
        ):
            raise ValueError(
                "TOPIQ-NR input must be a non-empty floating RGB NCHW tensor"
            )
        detached = images.detach()
        if not torch.isfinite(detached).all():
            raise ValueError("TOPIQ-NR input contains non-finite pixels")
        if torch.any(detached < 0.0) or torch.any(detached > 1.0):
            raise ValueError("TOPIQ-NR input must stay in [0, 1]")

    def _score(self, images: Tensor) -> Tensor:
        if any(module.training for module in self.metric.modules()):
            raise RuntimeError("TOPIQ-NR metric left evaluation mode")
        if any(parameter.requires_grad for parameter in self.metric.parameters()):
            raise RuntimeError("TOPIQ-NR metric is no longer frozen")
        device, dtype = _placement(self.metric, images)
        with torch.no_grad():
            scores = self.metric(images.detach().to(device=device, dtype=dtype))
        if not isinstance(scores, Tensor):
            raise TypeError("TOPIQ-NR must return a Tensor")
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if scores.ndim != 1 or scores.shape[0] != images.shape[0]:
            raise ValueError("TOPIQ-NR must return one scalar per image")
        if not scores.is_floating_point() or not torch.isfinite(scores).all():
            raise RuntimeError("TOPIQ-NR returned non-finite or non-floating scores")
        return scores.detach()

    def _execute(
        self,
        images: Tensor,
        context: Optional[OperationContext],
        parent_receipt: Optional[OperationReceipt],
    ) -> tuple[Tensor, Optional[OperationReceipt]]:
        self._validate_images(images)
        if self.operation_executor is None:
            if context is not None or parent_receipt is not None:
                raise ValueError("an informal TOPIQ-NR call cannot claim ledger provenance")
            return self._score(images), None
        if not isinstance(context, OperationContext):
            raise TypeError("formal TOPIQ-NR scoring requires an OperationContext")
        if parent_receipt is None:
            raise TypeError("formal TOPIQ-NR scoring requires its VAE decode receipt")
        if images.shape[0] != 1:
            raise ValueError("formal TOPIQ-NR scoring uses one prompt-seed image per call")
        image_hash = tensor_sha256(images)
        if context.image_hash != image_hash:
            raise ValueError("TOPIQ-NR context does not bind the scored image tensor")
        parent = self.operation_executor.verify_receipt(parent_receipt)
        decoded_output_hash = operation_output_sha256(images)
        if (
            parent.kind != "vae_decode"
            or parent.model != "repldm-runtime"
            or parent.role != "vae_decode"
        ):
            raise ValueError("TOPIQ-NR parent is not a VAE decode receipt")
        if parent.output_hash != decoded_output_hash:
            raise ValueError("TOPIQ-NR parent receipt names a different decoded tensor")
        if context.cached_parent != decoded_output_hash:
            raise ValueError("TOPIQ-NR context does not bind its VAE decode parent")
        for field in (
            "split",
            "prompt",
            "seed",
            "step",
            "prefix",
            "branch",
            "action",
            "checkpoint_hash",
        ):
            if getattr(parent.context, field) != getattr(context, field):
                raise ValueError("TOPIQ-NR context differs from its VAE decode parent")
        return self.operation_executor.execute_with_receipt(
            "reward_forward",
            context,
            lambda: self._score(images),
            amount=images.shape[0],
            scalar_or_gradient="scalar",
            model=self.model_id,
            role=self.role,
            preprocess_hash=self.preprocess_sha256,
            parent_receipt=parent,
        )

    def score_with_receipt(
        self,
        images: Tensor,
        context: OperationContext,
        *,
        parent_receipt: OperationReceipt,
    ) -> tuple[Tensor, OperationReceipt]:
        """Score one formal image and return its durable witness receipt."""
        if self.operation_executor is None:
            raise RuntimeError("an informal TOPIQ-NR witness cannot issue receipts")
        scores, receipt = self._execute(images, context, parent_receipt)
        if receipt is None:  # guarded above; keeps the return type fail closed
            raise RuntimeError("formal TOPIQ-NR scoring omitted its operation receipt")
        return scores, receipt

    def score_tensor(
        self,
        images: Tensor,
        context: Optional[OperationContext] = None,
        *,
        parent_receipt: Optional[OperationReceipt] = None,
    ) -> Tensor:
        """Return scores, discarding a formal receipt only when supplied."""
        scores, _receipt = self._execute(images, context, parent_receipt)
        return scores

    def forward(
        self,
        images: Tensor,
        context: Optional[OperationContext] = None,
        *,
        parent_receipt: Optional[OperationReceipt] = None,
    ) -> Tensor:
        return self.score_tensor(images, context, parent_receipt=parent_receipt)


__all__ = [
    "TOPIQ_NR_MODEL_ID",
    "TOPIQ_NR_PREPROCESS",
    "TOPIQ_NR_PREPROCESS_SHA256",
    "TOPIQ_NR_ROLE",
    "TopiqNrTensorWitness",
]
