"""Frozen relational latent alignment primitives.

The module deliberately stops at a fixed, auditable operator.  It does not
contain a learned policy or an RL loop.  A caller supplies one detached U-Net
feature map and the scheduler's update; the operator returns a bounded clean
latent residual that can be injected after the scheduler step.
"""

from dataclasses import dataclass
import math
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .latent_renderer import (
    _fixed_moment_geodesic,
    _project_fixed_moment_tangent,
    cap_update_norm,
)


Lag = Tuple[int, int]
DEFAULT_LAGS: Tuple[Lag, ...] = (
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (0, 2),
)


@dataclass(frozen=True)
class FRLAConfig:
    """Frozen settings for one relational correction.

    ``projection_seed`` fixes the feature-to-four-channel reduction.  The
    projection is generated on CPU and then moved to the feature device, so
    the same action is reproducible across CUDA and CPU smoke tests.
    """

    grid_size: int = 16
    # Relative step size measured against the scheduler update norm.  The
    # gradient is normalized per sample before this scale is applied.
    eta: float = 0.02
    projection_seed: int = 17
    lags: Tuple[Lag, ...] = DEFAULT_LAGS
    max_update_ratio: float = 0.05
    preserve_moments: bool = True
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        try:
            grid_size = int(self.grid_size)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("grid_size must be an integer") from exc
        if grid_size < 2:
            raise ValueError("grid_size must be at least two")
        try:
            eta = float(self.eta)
            max_update_ratio = float(self.max_update_ratio)
            epsilon = float(self.epsilon)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("eta, max_update_ratio, and epsilon must be numeric") from exc
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        if int(self.projection_seed) < 0:
            raise ValueError("projection_seed must be non-negative")
        if not self.lags:
            raise ValueError("lags must not be empty")
        for lag in self.lags:
            if len(lag) != 2 or min(int(lag[0]), int(lag[1])) < 0:
                raise ValueError("lags must contain non-negative (dy, dx) pairs")
            if max(int(lag[0]), int(lag[1])) >= grid_size:
                raise ValueError("lag must fit inside grid_size")
        if (
            not math.isfinite(max_update_ratio) or max_update_ratio < 0
        ):
            raise ValueError("max_update_ratio must be finite and non-negative")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

    def to_record(self) -> Dict[str, object]:
        """Return the frozen operator settings in JSON-safe form."""
        return {
            "grid_size": int(self.grid_size),
            "eta": float(self.eta),
            "projection_seed": int(self.projection_seed),
            "lags": [[int(dy), int(dx)] for dy, dx in self.lags],
            "max_update_ratio": float(self.max_update_ratio),
            "preserve_moments": bool(self.preserve_moments),
            "epsilon": float(self.epsilon),
        }


@dataclass
class FRLAOutput:
    """Result and diagnostics from one fixed relational correction."""

    guided_x0: Tensor
    residual: Tensor
    descriptor_before: Tensor
    descriptor_after: Tensor
    loss_before: Tensor
    loss_after: Tensor
    gradient_norm: Tensor
    update_ratio: Tensor

    def to_record(self) -> Dict[str, object]:
        def encode(value: Tensor):
            return value.detach().float().cpu().tolist()

        return {
            "descriptor_before": encode(self.descriptor_before),
            "descriptor_after": encode(self.descriptor_after),
            "loss_before": encode(self.loss_before),
            "loss_after": encode(self.loss_after),
            "gradient_norm": encode(self.gradient_norm),
            "update_ratio": encode(self.update_ratio),
        }


def _check_nchw(value: Tensor, name: str) -> None:
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [B,C,H,W]")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")


def fixed_channel_projection(
    in_channels: int,
    out_channels: int = 4,
    *,
    seed: int = 17,
    device: torch.device,
) -> Tensor:
    """Return a deterministic, row-normalized Gaussian projection."""
    if int(in_channels) <= 0 or int(out_channels) <= 0:
        raise ValueError("projection dimensions must be positive")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        int(out_channels), int(in_channels), generator=generator, dtype=torch.float32
    )
    matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return matrix.to(device=device)


def _reduce_to_grid(
    value: Tensor, projection: Tensor, grid_size: int
) -> Tensor:
    """Resize and reduce a feature map to four channels."""
    work = F.interpolate(
        value.float(), size=(int(grid_size), int(grid_size)), mode="area"
    )
    if work.shape[1] != projection.shape[1]:
        raise ValueError("projection channel count does not match feature map")
    return torch.einsum("oc,bchw->bohw", projection.float(), work)


def local_cosine_descriptor(value: Tensor, lags: Sequence[Lag], epsilon: float = 1e-6) -> Tensor:
    """Compute mean local cosine similarity for each fixed spatial lag."""
    _check_nchw(value, "value")
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("epsilon must be finite and positive") from exc
    if not math.isfinite(epsilon_value) or epsilon_value <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not lags:
        raise ValueError("lags must not be empty")
    height, width = value.shape[-2:]
    descriptors = []
    work = value.float()
    for dy, dx in lags:
        try:
            dy, dx = int(dy), int(dx)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("lags must contain integer pairs") from exc
        if dy < 0 or dx < 0:
            raise ValueError("lags must contain non-negative pairs")
        if dy >= height or dx >= width:
            raise ValueError("lag does not fit inside value")
        first = work[..., : height - dy, : width - dx]
        second = work[..., dy:, dx:]
        numerator = (first * second).sum(dim=1)
        denominator = (
            first.square().sum(dim=1).sqrt()
            * second.square().sum(dim=1).sqrt()
        )
        cosine = numerator / denominator.clamp_min(epsilon_value)
        descriptors.append(cosine.mean(dim=(-2, -1)))
    return torch.stack(descriptors, dim=1)


def _per_sample_norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value.float().flatten(1), dim=1)


def apply_frla(
    pred_original_sample: Tensor,
    feature: Tensor,
    scheduler_update: Tensor,
    config: FRLAConfig = FRLAConfig(),
) -> FRLAOutput:
    """Apply one detached-feature relational correction.

    The feature map is detached before any operation requiring gradients.  A
    graph is created only for a cloned clean latent, so no U-Net parameter or
    activation receives a gradient.  The returned tensor keeps the input
    latent dtype for direct scheduler injection.
    """
    _check_nchw(pred_original_sample, "pred_original_sample")
    _check_nchw(feature, "feature")
    _check_nchw(scheduler_update, "scheduler_update")
    if pred_original_sample.shape != scheduler_update.shape:
        raise ValueError("pred_original_sample and scheduler_update must match")
    if feature.shape[0] != pred_original_sample.shape[0]:
        raise ValueError("feature and latent batch sizes must match")
    if not (
        torch.isfinite(pred_original_sample).all()
        and torch.isfinite(feature).all()
        and torch.isfinite(scheduler_update).all()
    ):
        raise ValueError("FRLA inputs must be finite")

    base_x0 = pred_original_sample.detach()
    base_scheduler_update = scheduler_update.detach()
    feature = feature.detach()
    projection = fixed_channel_projection(
        feature.shape[1], seed=config.projection_seed, device=feature.device
    )
    target = _reduce_to_grid(feature, projection, config.grid_size).detach()
    target_descriptor = local_cosine_descriptor(target, config.lags, config.epsilon)

    # The only differentiable object is a cloned, detached clean latent.
    with torch.enable_grad():
        latent = base_x0.float().clone().requires_grad_(True)
        latent_grid = F.interpolate(
            latent, size=(config.grid_size, config.grid_size), mode="area"
        )
        descriptor_before = local_cosine_descriptor(
            latent_grid, config.lags, config.epsilon
        )
        per_lag_loss = (descriptor_before - target_descriptor).square()
        loss_before = per_lag_loss.mean(dim=1)
        gradient = torch.autograd.grad(loss_before.sum(), latent)[0]
        gradient_norm = _per_sample_norm(gradient)
        scheduler_norm = _per_sample_norm(base_scheduler_update)
        # Normalize the direction so ``eta`` has the same meaning across
        # feature magnitudes and image resolutions.  The subsequent cap still
        # enforces the hard per-sample trust bound.
        direction = gradient / gradient_norm.reshape((-1, 1, 1, 1)).clamp_min(
            config.epsilon
        )
        raw_update = (
            -float(config.eta)
            * scheduler_norm.reshape((-1, 1, 1, 1))
            * direction
        )

    if config.preserve_moments:
        tangent = _project_fixed_moment_tangent(
            base_x0.float(), raw_update, config.epsilon
        )
    else:
        tangent = raw_update
    bounded = cap_update_norm(
        tangent,
        base_scheduler_update.float(),
        float(config.max_update_ratio),
        config.epsilon,
    )
    if config.preserve_moments:
        guided_float = _fixed_moment_geodesic(
            base_x0.float(), bounded, config.epsilon
        )
    else:
        guided_float = base_x0.float() + bounded

    with torch.no_grad():
        guided_grid = F.interpolate(
            guided_float, size=(config.grid_size, config.grid_size), mode="area"
        )
        descriptor_after = local_cosine_descriptor(
            guided_grid, config.lags, config.epsilon
        )
        loss_after = (descriptor_after - target_descriptor).square().mean(dim=1)
        # Measure the residual after the output cast.  This is the quantity
        # that will actually be injected into a low-precision pipeline.
        guided_x0 = guided_float.to(base_x0.dtype).detach()
        residual = guided_x0.float() - base_x0.float()
        update_ratio = _per_sample_norm(residual) / _per_sample_norm(
            base_scheduler_update
        ).clamp_min(config.epsilon)
    return FRLAOutput(
        guided_x0=guided_x0,
        residual=guided_x0 - base_x0,
        descriptor_before=descriptor_before.detach(),
        descriptor_after=descriptor_after.detach(),
        loss_before=loss_before.detach(),
        loss_after=loss_after.detach(),
        gradient_norm=gradient_norm.detach(),
        update_ratio=update_ratio.detach(),
    )
