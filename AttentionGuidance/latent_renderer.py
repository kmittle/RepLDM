"""Small, constrained latent renderers for a frozen diffusion backbone.

This module is deliberately independent of ``diffusers`` and of the SDXL
pipeline.  A caller supplies interpretable candidate residuals (for example,
spectral, semantic-transport, FreeU-style, or Laplacian bases) together with
the scheduler's predicted clean latent.  :class:`StructuralLatentRenderer`
only learns how to allocate bounded coefficients over those bases.  The
resulting clean-latent update can then be injected with
:func:`inject_rendered_clean_update` after the scheduler has performed its
ordinary step.

The zero-initialised head is intentional: a freshly constructed renderer is
exactly the identity, so training or loading a checkpoint is an explicit
choice.  Moment and trust-region constraints are applied in float32 for
half-precision latents and are kept differentiable for later distillation or
RL experiments.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
from typing import Any, Callable, Iterator, Optional, Protocol, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .semantic_transport import (
    QKCapture,
    affinity_from_qk,
    affinity_from_tokens,
    infer_token_grid,
)


@dataclass(frozen=True)
class LatentRendererConfig:
    """Architecture and safety settings for one renderer instance."""

    num_bases: int
    latent_channels: int = 4
    prompt_dim: int = 0
    state_dim: int = 0
    hidden_dim: int = 128
    depth: int = 2
    timestep_dim: int = 16
    coefficient_bound: float = 1.0
    spatial_hidden_dim: int = 0
    spatial_kernel_size: int = 3
    spatial_bound: float = 1.0
    max_update_ratio: Optional[float] = 0.05
    preserve_moments: bool = True
    normalize_bases: bool = True
    epsilon: float = 1e-6
    # Keep the historical pre-cast behavior by default.  When enabled, the
    # actual low-precision residual is rechecked and corrected per sample.
    enforce_post_cast_cap: bool = False

    def __post_init__(self) -> None:
        if int(self.num_bases) <= 0:
            raise ValueError("num_bases must be positive")
        if int(self.latent_channels) <= 0:
            raise ValueError("latent_channels must be positive")
        if int(self.prompt_dim) < 0 or int(self.state_dim) < 0:
            raise ValueError("prompt_dim and state_dim must be non-negative")
        if int(self.hidden_dim) <= 0 or int(self.depth) <= 0:
            raise ValueError("hidden_dim and depth must be positive")
        if int(self.spatial_hidden_dim) < 0:
            raise ValueError("spatial_hidden_dim must be non-negative")
        if int(self.spatial_kernel_size) < 1 or int(self.spatial_kernel_size) % 2 == 0:
            raise ValueError("spatial_kernel_size must be a positive odd integer")
        if int(self.timestep_dim) < 0:
            raise ValueError("timestep_dim must be non-negative")
        if (
            not math.isfinite(float(self.coefficient_bound))
            or self.coefficient_bound <= 0
        ):
            raise ValueError("coefficient_bound must be finite and positive")
        if not math.isfinite(float(self.spatial_bound)) or self.spatial_bound <= 0:
            raise ValueError("spatial_bound must be finite and positive")
        if self.max_update_ratio is not None:
            if not math.isfinite(float(self.max_update_ratio)) or self.max_update_ratio < 0:
                raise ValueError("max_update_ratio must be finite and non-negative")
        if not math.isfinite(float(self.epsilon)) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not isinstance(self.enforce_post_cast_cap, bool):
            raise ValueError("enforce_post_cast_cap must be a boolean")
        if self.enforce_post_cast_cap and self.max_update_ratio is None:
            raise ValueError(
                "enforce_post_cast_cap requires max_update_ratio to be configured"
            )


@dataclass(frozen=True)
class RendererDiagnostics:
    """Auditable quantities emitted by a renderer forward pass."""

    raw_update_norm: Tensor
    bounded_update_norm: Tensor
    scheduler_update_norm: Optional[Tensor]
    update_ratio: Optional[Tensor]
    mean_error: Tensor
    variance_error: Tensor
    # Optional fields keep positional construction of the original six-field
    # diagnostics object source-compatible for downstream audit utilities.
    precast_update_norm: Optional[Tensor] = None
    precast_update_ratio: Optional[Tensor] = None
    precast_overrun: Optional[Tensor] = None
    postcast_update_norm: Optional[Tensor] = None
    postcast_update_ratio: Optional[Tensor] = None
    postcast_overrun: Optional[Tensor] = None
    observed_postcast_overrun: Optional[Tensor] = None
    postcast_cap_applied: Optional[Tensor] = None
    postcast_noop_fallback: Optional[Tensor] = None

    def to_record(self) -> dict:
        """Return JSON-safe per-sample diagnostics for an experiment sidecar."""

        def encode(value: Optional[Tensor]):
            if value is None:
                return None
            if value.dtype == torch.bool:
                return value.detach().cpu().tolist()
            return value.detach().float().cpu().tolist()

        return {
            "raw_update_norm": encode(self.raw_update_norm),
            "bounded_update_norm": encode(self.bounded_update_norm),
            "scheduler_update_norm": encode(self.scheduler_update_norm),
            "update_ratio": encode(self.update_ratio),
            "mean_error": encode(self.mean_error),
            "variance_error": encode(self.variance_error),
            "precast_update_norm": encode(self.precast_update_norm),
            "precast_update_ratio": encode(self.precast_update_ratio),
            "precast_overrun": encode(self.precast_overrun),
            "postcast_update_norm": encode(self.postcast_update_norm),
            "postcast_update_ratio": encode(self.postcast_update_ratio),
            "postcast_overrun": encode(self.postcast_overrun),
            "observed_postcast_overrun": encode(self.observed_postcast_overrun),
            "postcast_cap_applied": encode(self.postcast_cap_applied),
            "postcast_noop_fallback": encode(self.postcast_noop_fallback),
        }


@dataclass(frozen=True)
class InjectionDiagnostics:
    """Auditable residuals for the final scheduler-sample injection."""

    scheduler_update_norm: Optional[Tensor]
    precast_residual_norm: Tensor
    precast_update_ratio: Optional[Tensor]
    precast_overrun: Optional[Tensor]
    postcast_residual_norm: Tensor
    postcast_update_ratio: Optional[Tensor]
    postcast_overrun: Optional[Tensor]
    observed_postcast_overrun: Optional[Tensor]
    postcast_cap_applied: Optional[Tensor]
    postcast_noop_fallback: Optional[Tensor]

    def to_record(self) -> dict:
        """Return JSON-safe per-sample injection diagnostics."""

        def encode(value: Optional[Tensor]):
            if value is None:
                return None
            if value.dtype == torch.bool:
                return value.detach().cpu().tolist()
            return value.detach().float().cpu().tolist()

        return {
            "scheduler_update_norm": encode(self.scheduler_update_norm),
            "precast_residual_norm": encode(self.precast_residual_norm),
            "precast_update_ratio": encode(self.precast_update_ratio),
            "precast_overrun": encode(self.precast_overrun),
            "postcast_residual_norm": encode(self.postcast_residual_norm),
            "postcast_update_ratio": encode(self.postcast_update_ratio),
            "postcast_overrun": encode(self.postcast_overrun),
            "observed_postcast_overrun": encode(self.observed_postcast_overrun),
            "postcast_cap_applied": encode(self.postcast_cap_applied),
            "postcast_noop_fallback": encode(self.postcast_noop_fallback),
        }


@dataclass(frozen=True)
class InjectionOutput:
    """Final injected sample and optional strict-cap diagnostics."""

    sample: Tensor
    diagnostics: InjectionDiagnostics


@dataclass(frozen=True)
class RendererOutput:
    """Rendered clean latent, residual, coefficients, and safety diagnostics."""

    guided_x0: Tensor
    residual: Tensor
    coefficients: Tensor
    diagnostics: RendererDiagnostics


@dataclass(frozen=True)
class RendererObservation:
    """One scheduler transition exposed to a basis provider."""

    latents_before_step: Tensor
    pred_original_sample: Tensor
    scheduler_update: Tensor
    step_index: int
    timestep: Tensor
    normalized_timestep: Tensor
    pooled_prompt_embeds: Optional[Tensor] = None


@dataclass(frozen=True)
class RendererCondition:
    """Candidate bases and optional compact conditioning for one renderer call."""

    bases: Tensor
    prompt_embedding: Optional[Tensor] = None
    state_features: Optional[Tensor] = None


class RendererBasisProvider(Protocol):
    """Build renderer bases from one ordinary frozen denoising transition."""

    def __call__(self, observation: RendererObservation) -> RendererCondition:
        ...


def _require_nchw(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, channels, height, width)")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _vector_norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value.float().flatten(1), dim=-1)


def _sinusoidal_embedding(timestep: Tensor, dimension: int) -> Tensor:
    """Encode normalized timesteps without assuming a scheduler implementation."""
    if dimension == 0:
        return timestep.new_empty((timestep.shape[0], 0), dtype=torch.float32)
    half = dimension // 2
    if half == 0:
        return timestep.float().unsqueeze(-1)
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = timestep.float().reshape(-1, 1) * frequency.reshape(1, -1)
    embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _coerce_batch_vector(
    value: Optional[Tensor], batch: int, name: str
) -> Optional[Tensor]:
    if value is None:
        return None
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        value = value.expand(batch)
    if value.ndim != 1 or value.shape[0] not in (1, batch):
        raise ValueError(f"{name} must be scalar or have one value per batch item")
    if value.shape[0] == 1:
        value = value.expand(batch)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _coerce_context(
    value: Optional[Tensor], batch: int, width: int, name: str, reference: Tensor
) -> Tensor:
    if width == 0:
        if value is not None and torch.as_tensor(value).numel() != 0:
            raise ValueError(f"{name} was provided but configured width is zero")
        return reference.new_empty((batch, 0), dtype=torch.float32)
    if value is not None and not isinstance(value, Tensor):
        value = torch.as_tensor(value)
    if value is None:
        raise ValueError(f"{name} is required when configured width is {width}")
    if value.ndim != 2 or value.shape[0] not in (1, batch) or value.shape[1] != width:
        raise ValueError(f"{name} must have shape (batch, {width})")
    if value.shape[0] == 1:
        value = value.expand(batch, -1)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value.to(device=reference.device, dtype=torch.float32)


def _basis_statistics(latent: Tensor, bases: Tensor, epsilon: float) -> Tensor:
    """Summarise each basis without introducing a resolution-dependent encoder."""
    batch, num_bases = bases.shape[:2]
    flat_basis = bases.float().flatten(2)
    flat_latent = latent.float().flatten(1)
    latent_norm = torch.linalg.vector_norm(flat_latent, dim=-1).clamp_min(epsilon)
    basis_norm = torch.linalg.vector_norm(flat_basis, dim=-1).clamp_min(epsilon)
    mean = flat_basis.mean(dim=-1)
    std = flat_basis.std(dim=-1, unbiased=False)
    norm_ratio = basis_norm / latent_norm[:, None]
    cosine = (flat_basis * flat_latent[:, None]).sum(dim=-1) / (
        basis_norm * latent_norm[:, None]
    )
    return torch.stack((mean, std, norm_ratio, cosine), dim=-1).reshape(
        batch, num_bases * 4
    )


def _spectral_summary(value: Tensor, epsilon: float) -> Tensor:
    """Return low/mid/high energy fractions as a compact state feature."""
    height, width = value.shape[-2:]
    spectrum = torch.fft.rfft2(value.float(), dim=(-2, -1), norm="ortho")
    fy = torch.fft.fftfreq(height, device=value.device, dtype=torch.float32)
    fx = torch.fft.rfftfreq(width, device=value.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    low = radius <= 0.08
    mid = (radius > 0.08) & (radius <= 0.25)
    high = radius > 0.25
    energy = spectrum.abs().square().mean(dim=1)
    parts = torch.stack(
        (
            (energy * low).sum(dim=(-2, -1)),
            (energy * mid).sum(dim=(-2, -1)),
            (energy * high).sum(dim=(-2, -1)),
        ),
        dim=-1,
    )
    return parts / parts.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def _project_fixed_moment_tangent(
    latent: Tensor, residual: Tensor, epsilon: float
) -> Tensor:
    work = latent.float()
    raw = residual.float()
    centered = work - work.mean(dim=(-2, -1), keepdim=True)
    centered_residual = raw - raw.mean(dim=(-2, -1), keepdim=True)
    energy = centered.square().sum(dim=(-2, -1), keepdim=True)
    inner = (centered_residual * centered).sum(dim=(-2, -1), keepdim=True)
    tangent = centered_residual - inner / energy.clamp_min(epsilon) * centered
    return torch.where(energy > epsilon, tangent, torch.zeros_like(tangent))


def _fixed_moment_geodesic(latent: Tensor, tangent: Tensor, epsilon: float) -> Tensor:
    """Map a tangent vector to the per-channel fixed-mean/norm sphere."""
    work = latent.float()
    tangent = tangent.float()
    mean = work.mean(dim=(-2, -1), keepdim=True)
    centered = work - mean
    latent_energy = centered.square().sum(dim=(-2, -1), keepdim=True)
    tangent_energy = tangent.square().sum(dim=(-2, -1), keepdim=True)
    radius = torch.sqrt(latent_energy.clamp_min(0.0))
    speed = torch.sqrt(tangent_energy.clamp_min(0.0))
    direction = tangent / speed.clamp_min(math.sqrt(epsilon))
    angle = speed / radius.clamp_min(math.sqrt(epsilon))
    moved = torch.cos(angle) * centered + torch.sin(angle) * radius * direction + mean
    active = (latent_energy > epsilon) & (tangent_energy > epsilon)
    return torch.where(active, moved, work)


def cap_update_norm(
    update: Tensor,
    reference_update: Tensor,
    max_update_ratio: float,
    epsilon: float = 1e-6,
) -> Tensor:
    """Cap each sample's update norm relative to a scheduler update."""
    _require_nchw(update, "update")
    _require_nchw(reference_update, "reference_update")
    if update.shape != reference_update.shape:
        raise ValueError("update and reference_update must have identical shapes")
    if max_update_ratio < 0 or not math.isfinite(float(max_update_ratio)):
        raise ValueError("max_update_ratio must be finite and non-negative")
    update_norm = _vector_norm(update)
    reference_norm = _vector_norm(reference_update)
    multiplier = torch.clamp(
        float(max_update_ratio) * reference_norm / (update_norm + epsilon), max=1.0
    )
    return update * multiplier.reshape((-1, 1, 1, 1)).to(update.dtype)


def _render_guided_x0(
    latent: Tensor, update: Tensor, preserve_moments: bool, epsilon: float
) -> Tensor:
    """Map a float32 update to a clean latent before dtype conversion."""
    if preserve_moments:
        return _fixed_moment_geodesic(latent, update, epsilon)
    return latent.float() + update.float()


def _ratio_against_reference(
    residual: Tensor, reference_update: Tensor, epsilon: float
) -> Tensor:
    """Return per-sample residual/reference norms in float32."""
    return _vector_norm(residual) / _vector_norm(reference_update).clamp_min(epsilon)


def _cast_guided_x0_with_cap(
    latent: Tensor,
    update: Tensor,
    scheduler_update: Tensor,
    max_update_ratio: float,
    preserve_moments: bool,
    epsilon: float,
    enforce_post_cast_cap: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Cast a candidate and optionally enforce the cap after quantization.

    The first return value is the float32 candidate corresponding to the final
    action, the second is its stored latent-dtype representation, and the third
    records any post-cast overrun observed before strict correction.  The fourth
    and fifth values indicate correction and exact no-op fallback per sample.
    Strict correction scales the tangent once, then falls back to an exact
    no-op for samples that remain over the bound after quantization.
    """
    candidate_float = _render_guided_x0(
        latent, update, preserve_moments, epsilon
    )
    candidate_cast = candidate_float.to(latent.dtype)
    reference_norm = _vector_norm(scheduler_update)
    candidate_finite = torch.isfinite(candidate_cast).flatten(1).all(dim=1)
    observed_ratio = _ratio_against_reference(
        candidate_cast.float() - latent.float(), scheduler_update, epsilon
    )
    observed_overrun = (observed_ratio - float(max_update_ratio)).clamp_min(0.0)
    observed_overrun = torch.where(
        candidate_finite,
        observed_overrun,
        torch.full_like(observed_overrun, float("inf")),
    )
    if not enforce_post_cast_cap or not torch.any(observed_overrun > 0):
        zeros = torch.zeros_like(observed_overrun, dtype=torch.bool)
        return candidate_float, candidate_cast, observed_overrun, zeros, zeros

    overrun = observed_overrun > 0
    cast_norm = _vector_norm(candidate_cast.float() - latent.float())
    scale = torch.clamp(
        float(max_update_ratio) * reference_norm / (cast_norm + epsilon), max=1.0
    )
    scaled_update = update * scale.reshape((-1, 1, 1, 1)).to(update.dtype)
    scaled_float = _render_guided_x0(
        latent, scaled_update, preserve_moments, epsilon
    )
    scaled_cast = scaled_float.to(latent.dtype)
    scaled_ratio = _ratio_against_reference(
        scaled_cast.float() - latent.float(), scheduler_update, epsilon
    )
    scaled_finite = torch.isfinite(scaled_cast).flatten(1).all(dim=1)
    fallback = overrun & (
        ~candidate_finite
        | ~scaled_finite
        | (scaled_ratio > float(max_update_ratio))
    )
    batch_mask = overrun.reshape((-1, 1, 1, 1))
    fallback_mask = fallback.reshape((-1, 1, 1, 1))
    final_float = torch.where(batch_mask, scaled_float, candidate_float)
    final_cast = torch.where(batch_mask, scaled_cast, candidate_cast)
    # Quantization can make a single representable step larger than the cap;
    # an exact latent fallback is the only strict and moment-preserving action.
    final_float = torch.where(fallback_mask, latent.float(), final_float)
    final_cast = torch.where(fallback_mask, latent, final_cast)
    return final_float, final_cast, observed_overrun, overrun, fallback


def build_spectral_bases(
    reference: Tensor, cutoffs: Tuple[float, float] = (0.08, 0.25)
) -> Tensor:
    """Decompose a latent residual into smooth low/mid/high Fourier bases."""
    _require_nchw(reference, "reference")
    low_cutoff, mid_cutoff = map(float, cutoffs)
    if not 0 < low_cutoff < mid_cutoff < 0.5:
        raise ValueError("cutoffs must satisfy 0 < low < mid < 0.5")
    height, width = reference.shape[-2:]
    work = reference.float()
    spectrum = torch.fft.rfft2(work, dim=(-2, -1), norm="ortho")
    fy = torch.fft.fftfreq(height, device=reference.device, dtype=torch.float32)
    fx = torch.fft.rfftfreq(width, device=reference.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    low_pass = torch.exp(-0.5 * (radius / low_cutoff).pow(4))
    mid_pass = torch.exp(-0.5 * (radius / mid_cutoff).pow(4))
    masks = torch.stack((low_pass, mid_pass - low_pass, 1.0 - mid_pass), dim=0)
    bands = torch.fft.irfft2(
        spectrum[:, None] * masks[None, :, None],
        s=(height, width),
        dim=(-2, -1),
        norm="ortho",
    )
    return bands.to(reference.dtype)


def build_graph_transport_basis(
    reference: Tensor, graph: Tensor, grid_height: int, grid_width: int
) -> Tensor:
    """Construct a semantic transport residual from a row-stochastic graph."""
    _require_nchw(reference, "reference")
    if graph.ndim != 3 or graph.shape[0] not in (1, reference.shape[0]):
        raise ValueError("graph must have shape (batch, tokens, tokens)")
    tokens = int(grid_height) * int(grid_width)
    if (
        grid_height <= 0
        or grid_width <= 0
        or graph.shape[-2:] != (tokens, tokens)
    ):
        raise ValueError("graph shape does not match grid_height/grid_width")
    if graph.shape[0] == 1:
        graph = graph.expand(reference.shape[0], -1, -1)
    if not torch.isfinite(graph).all() or torch.any(graph < 0):
        raise ValueError("graph must contain finite non-negative weights")
    row_sum = graph.float().sum(dim=-1)
    if not torch.allclose(
        row_sum, torch.ones_like(row_sum), atol=1e-4, rtol=1e-4
    ):
        raise ValueError("graph must be row-stochastic")
    work = reference.float()
    coarse = F.interpolate(work, size=(grid_height, grid_width), mode="area")
    tokens_value = coarse.flatten(2).transpose(1, 2)
    moved = torch.bmm(graph.to(work), tokens_value)
    coarse_residual = (moved - tokens_value).transpose(1, 2).reshape(
        reference.shape[0], reference.shape[1], grid_height, grid_width
    )
    residual = F.interpolate(
        coarse_residual, size=reference.shape[-2:], mode="bilinear", align_corners=False
    )
    return residual.to(reference.dtype).unsqueeze(1)


def build_laplacian_basis(reference: Tensor) -> Tensor:
    """Return a local edge/Laplacian residual at the latent resolution."""
    _require_nchw(reference, "reference")
    mode = "reflect" if min(reference.shape[-2:]) > 1 else "replicate"
    padded = F.pad(reference.float(), (1, 1, 1, 1), mode=mode)
    smooth = F.avg_pool2d(padded, kernel_size=3, stride=1)
    return (reference.float() - smooth).to(reference.dtype).unsqueeze(1)


def build_feature_difference_basis(
    backbone: Tensor,
    skip: Tensor,
    target_size: Tuple[int, int],
    projection: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Build a FreeU-style backbone-minus-skip basis.

    Feature tensors may come from different UNet resolutions.  They must have
    matching channels after the optional caller-owned projection; keeping that
    projection explicit makes its parameter count and compute auditable.
    """
    _require_nchw(backbone, "backbone")
    _require_nchw(skip, "skip")
    if projection is not None:
        backbone = projection(backbone)
        skip = projection(skip)
        _require_nchw(backbone, "projected backbone")
        _require_nchw(skip, "projected skip")
    if backbone.shape[1] != skip.shape[1]:
        raise ValueError("backbone and skip must have equal channels after projection")
    resized_backbone = F.interpolate(
        backbone.float(), size=target_size, mode="bilinear", align_corners=False
    )
    resized_skip = F.interpolate(
        skip.float(), size=target_size, mode="bilinear", align_corners=False
    )
    return (resized_backbone - resized_skip).to(backbone.dtype).unsqueeze(1)


def inject_rendered_clean_update(
    prev_sample: Tensor,
    pred_original_sample: Tensor,
    guided_x0: Tensor,
    *,
    scheduler_update: Optional[Tensor] = None,
    max_update_ratio: Optional[float] = None,
    enforce_post_cast_cap: bool = False,
    return_diagnostics: bool = False,
    epsilon: float = 1e-6,
) -> Union[Tensor, InjectionOutput]:
    """Inject a rendered ``x0`` while retaining the scheduler's own step.

    The three positional arguments intentionally retain the historical
    implementation byte-for-byte when no keyword options are supplied.  The
    optional contract measures the *actual* residual after conversion to
    ``prev_sample.dtype``.  With ``enforce_post_cast_cap=True``, samples that
    exceed ``max_update_ratio`` of ``scheduler_update`` are scaled and then
    checked again after conversion; an unrepresentable or still-over-limit
    action is replaced by the exact scheduler sample (a no-op).
    """
    # Keep the old validation and arithmetic on the compatibility path.  This
    # matters for callers that rely on low-precision operation ordering.  The
    # strict path permits a non-finite rendered candidate so it can turn that
    # action into an exact no-op instead of propagating it.
    if not isinstance(enforce_post_cast_cap, bool):
        raise ValueError("enforce_post_cast_cap must be a boolean")

    def _require_layout(value: Tensor, name: str) -> None:
        if not isinstance(value, Tensor) or value.ndim != 4:
            raise ValueError(f"{name} must have shape (batch, channels, height, width)")

    _require_nchw(prev_sample, "prev_sample")
    if enforce_post_cast_cap:
        _require_layout(pred_original_sample, "pred_original_sample")
        _require_layout(guided_x0, "guided_x0")
    else:
        _require_nchw(pred_original_sample, "pred_original_sample")
        _require_nchw(guided_x0, "guided_x0")
    if (
        prev_sample.shape != pred_original_sample.shape
        or guided_x0.shape != prev_sample.shape
    ):
        raise ValueError("scheduler samples and guided_x0 must have identical shapes")

    compatibility_path = (
        scheduler_update is None
        and max_update_ratio is None
        and not enforce_post_cast_cap
        and not return_diagnostics
    )
    if compatibility_path:
        return (prev_sample + guided_x0 - pred_original_sample).to(prev_sample.dtype)

    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("epsilon must be finite and positive") from exc
    if not math.isfinite(epsilon_value) or epsilon_value <= 0:
        raise ValueError("epsilon must be finite and positive")

    ratio_bound: Optional[float]
    if max_update_ratio is None:
        ratio_bound = None
    else:
        try:
            ratio_bound = float(max_update_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_update_ratio must be finite and non-negative") from exc
        if not math.isfinite(ratio_bound) or ratio_bound < 0:
            raise ValueError("max_update_ratio must be finite and non-negative")

    if ratio_bound is not None and scheduler_update is None:
        raise ValueError("scheduler_update is required with max_update_ratio")
    if enforce_post_cast_cap and (scheduler_update is None or ratio_bound is None):
        raise ValueError(
            "enforce_post_cast_cap requires scheduler_update and max_update_ratio"
        )

    if scheduler_update is not None:
        _require_nchw(scheduler_update, "scheduler_update")
        if scheduler_update.shape != prev_sample.shape:
            raise ValueError("scheduler_update must have the same shape as prev_sample")

    # The strict path is also a finite-value guard.  Inputs are normally
    # finite, but allowing a non-finite rendered candidate here lets the
    # exact-no-op fallback protect the scheduler transition instead of
    # propagating an invalid latent.  ``prev_sample`` and the reference update
    # themselves must remain finite so the fallback and ratio are meaningful.
    if enforce_post_cast_cap:
        if not torch.isfinite(prev_sample).all():
            raise ValueError("prev_sample contains non-finite values")
        if scheduler_update is not None and not torch.isfinite(scheduler_update).all():
            raise ValueError("scheduler_update contains non-finite values")

    # This is intentionally the historical expression.  In particular, the
    # pre-cast value may already be half precision; it is the quantity whose
    # final cast is audited below.  The compatibility path above returns before
    # any additional work is performed.
    candidate_precast = prev_sample + guided_x0 - pred_original_sample
    candidate_cast = candidate_precast.to(prev_sample.dtype)
    batch = prev_sample.shape[0]
    # Addition/subtraction in fp16 can round a mathematically zero rendered
    # delta (``guided_x0 == pred_original_sample``) away from the scheduler
    # sample.  Preserve the identity contract only on the strict opt-in path;
    # the extended diagnostics path with enforcement disabled must remain
    # byte-compatible with the historical three-argument expression.
    if enforce_post_cast_cap:
        zero_render_delta = (
            (guided_x0 == pred_original_sample).flatten(1).all(dim=1)
            & torch.isfinite(guided_x0).flatten(1).all(dim=1)
            & torch.isfinite(pred_original_sample).flatten(1).all(dim=1)
        )
        zero_render_mask = zero_render_delta.reshape((-1, 1, 1, 1))
        candidate_precast = torch.where(
            zero_render_mask,
            prev_sample,
            candidate_precast,
        )
        candidate_cast = torch.where(
            zero_render_mask,
            prev_sample,
            candidate_cast,
        )
    prev_float = prev_sample.float()
    precast_residual = candidate_precast.float() - prev_float
    postcast_residual = candidate_cast.float() - prev_float
    precast_norm = _vector_norm(precast_residual)
    postcast_norm_observed = _vector_norm(postcast_residual)

    scheduler_norm: Optional[Tensor] = None
    precast_ratio: Optional[Tensor] = None
    observed_postcast_ratio: Optional[Tensor] = None
    precast_overrun: Optional[Tensor] = None
    observed_postcast_overrun: Optional[Tensor] = None
    if scheduler_update is not None:
        scheduler_norm = _vector_norm(scheduler_update)
        denominator = scheduler_norm.clamp_min(epsilon_value)
        precast_ratio = precast_norm / denominator
        observed_postcast_ratio = postcast_norm_observed / denominator
        if ratio_bound is not None:
            precast_overrun = (precast_ratio - ratio_bound).clamp_min(0.0)
            observed_postcast_overrun = (
                observed_postcast_ratio - ratio_bound
            ).clamp_min(0.0)

    # Detect invalid values using the cast representation, because that is
    # what the scheduler will actually consume.  ``isfinite`` on the ratio is
    # included for the zero/overflow cases where a norm can become NaN.
    cast_finite = torch.isfinite(candidate_cast).flatten(1).all(dim=1)
    if ratio_bound is not None and observed_postcast_ratio is not None:
        over_limit = (
            ~cast_finite
            | ~torch.isfinite(observed_postcast_ratio)
            | (observed_postcast_ratio > ratio_bound)
        )
    else:
        over_limit = ~cast_finite if enforce_post_cast_cap else torch.zeros(
            batch, dtype=torch.bool, device=prev_sample.device
        )

    correction_applied = torch.zeros(
        batch, dtype=torch.bool, device=prev_sample.device
    )
    noop_fallback = torch.zeros_like(correction_applied)
    final_sample = candidate_cast
    if enforce_post_cast_cap and torch.any(over_limit):
        correction_applied = over_limit

        # Recompute in float32 for correction.  This recovers finite values
        # when the original half-precision sum overflowed before its final
        # cast, while still falling back when the rendered inputs themselves
        # are non-finite.
        safe_candidate = prev_float + guided_x0.float() - pred_original_sample.float()
        safe_residual = safe_candidate - prev_float
        safe_finite = torch.isfinite(safe_candidate).flatten(1).all(dim=1)
        safe_norm = _vector_norm(safe_residual)
        target_norm = scheduler_norm * ratio_bound
        scale = torch.clamp(
            target_norm / (safe_norm + epsilon_value), max=1.0
        )
        scaled_candidate = prev_float + safe_residual * scale.reshape(
            (-1, 1, 1, 1)
        )
        scaled_sample = scaled_candidate.to(prev_sample.dtype)
        scaled_finite = torch.isfinite(scaled_sample).flatten(1).all(dim=1)
        scaled_residual = scaled_sample.float() - prev_float
        scaled_ratio = _vector_norm(scaled_residual) / scheduler_norm.clamp_min(
            epsilon_value
        )
        still_invalid = (
            ~safe_finite
            | ~scaled_finite
            | ~torch.isfinite(scaled_ratio)
            | (scaled_ratio > ratio_bound)
        )
        correction_mask = over_limit.reshape((-1, 1, 1, 1))
        final_sample = torch.where(correction_mask, scaled_sample, candidate_cast)
        noop_fallback = over_limit & still_invalid
        fallback_mask = noop_fallback.reshape((-1, 1, 1, 1))
        final_sample = torch.where(fallback_mask, prev_sample, final_sample)

    final_residual = final_sample.float() - prev_float
    final_norm = _vector_norm(final_residual)
    final_ratio: Optional[Tensor] = None
    final_overrun: Optional[Tensor] = None
    if scheduler_norm is not None:
        final_ratio = final_norm / scheduler_norm.clamp_min(epsilon_value)
        if ratio_bound is not None:
            final_overrun = (final_ratio - ratio_bound).clamp_min(0.0)

    if not return_diagnostics:
        return final_sample
    diagnostics = InjectionDiagnostics(
        scheduler_update_norm=scheduler_norm,
        precast_residual_norm=precast_norm,
        precast_update_ratio=precast_ratio,
        precast_overrun=precast_overrun,
        postcast_residual_norm=final_norm,
        postcast_update_ratio=final_ratio,
        postcast_overrun=final_overrun,
        observed_postcast_overrun=observed_postcast_overrun,
        postcast_cap_applied=correction_applied,
        postcast_noop_fallback=noop_fallback,
    )
    return InjectionOutput(sample=final_sample, diagnostics=diagnostics)


class _D4DepthwiseConv2d(nn.Module):
    """Depthwise convolution whose kernel is symmetrized over flips/rotations."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(
            torch.empty(self.channels, 1, self.kernel_size, self.kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @staticmethod
    def _d4_variants(weight: Tensor) -> Tuple[Tensor, ...]:
        variants = []
        for transpose in (False, True):
            value = weight.transpose(-1, -2) if transpose else weight
            variants.extend(
                (
                    value,
                    torch.flip(value, dims=(-1,)),
                    torch.flip(value, dims=(-2,)),
                    torch.flip(value, dims=(-2, -1)),
                )
            )
        return tuple(variants)

    def forward(self, value: Tensor) -> Tensor:
        kernel = torch.stack(self._d4_variants(self.weight), dim=0).mean(dim=0)
        return F.conv2d(
            value,
            kernel,
            padding=self.kernel_size // 2,
            groups=self.channels,
        )


class StructuralLatentRenderer(nn.Module):
    """Allocate bounded coefficients over interpretable latent residual bases.

    ``timestep`` is expected to be normalized to ``[0, 1]`` by the caller.
    ``prompt_embedding`` and ``state_features`` are optional only when their
    configured dimensions are zero.  The final linear layer starts at zero,
    making an untrained renderer an exact no-op.  Set ``spatial_hidden_dim``
    above zero to add a depthwise-separable local latent head; zero keeps the
    coefficient-only control used by the fixed-basis ablation.  Set
    ``enforce_post_cast_cap`` to recheck the actual low-precision residual and
    shrink or disable samples that quantize beyond the scheduler trust cap.
    """

    def __init__(self, config: LatentRendererConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = (
            config.num_bases * 4
            + 3  # latent mean/std/norm
            + 3  # low/mid/high spectral fractions
            + config.timestep_dim
            + config.prompt_dim
            + config.state_dim
        )
        layers = []
        current_dim = input_dim
        for _ in range(config.depth):
            layers.extend((nn.Linear(current_dim, config.hidden_dim), nn.SiLU()))
            current_dim = config.hidden_dim
        layers.append(nn.Linear(current_dim, config.num_bases))
        self.policy = nn.Sequential(*layers)
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)
        self.spatial_head: Optional[nn.Module]
        if config.spatial_hidden_dim == 0:
            self.spatial_head = None
        else:
            input_channels = 2 * config.latent_channels
            self.spatial_head = nn.ModuleDict(
                {
                    "depthwise": _D4DepthwiseConv2d(
                        input_channels, config.spatial_kernel_size
                    ),
                    "pointwise": nn.Conv2d(
                        input_channels, config.spatial_hidden_dim, kernel_size=1
                    ),
                    "film": nn.Linear(
                        input_dim, 2 * config.spatial_hidden_dim
                    ),
                    "output": nn.Conv2d(
                        config.spatial_hidden_dim,
                        config.latent_channels,
                        kernel_size=1,
                    ),
                }
            )
            # Preserve the identity at initialization; the output projection
            # can learn a local residual after the first optimizer update.
            nn.init.zeros_(self.spatial_head["output"].weight)
            nn.init.zeros_(self.spatial_head["output"].bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _prepare_bases(self, latent: Tensor, bases: Tensor) -> Tensor:
        _require_nchw(latent, "latent")
        if latent.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"latent has {latent.shape[1]} channels; expected {self.config.latent_channels}"
            )
        if not isinstance(bases, Tensor) or bases.ndim != 5:
            raise ValueError(
                "bases must have shape (batch, num_bases, channels, height, width)"
            )
        if bases.device != latent.device:
            raise ValueError("bases and latent must be on the same device")
        if bases.shape[0] != latent.shape[0] or bases.shape[1] != self.config.num_bases:
            raise ValueError("bases batch or num_bases does not match the renderer")
        if bases.shape[2:] != latent.shape[1:]:
            raise ValueError("each basis must have the same channels and resolution as latent")
        if not torch.isfinite(bases).all():
            raise ValueError("bases contain non-finite values")
        work = bases.float()
        if not self.config.normalize_bases:
            return work
        basis_norm = torch.linalg.vector_norm(work.flatten(2), dim=-1)
        latent_rms = _vector_norm(latent).reshape(-1, 1, 1, 1, 1) / math.sqrt(
            latent[0].numel()
        )
        basis_scale = basis_norm.reshape(
            basis_norm.shape[0], basis_norm.shape[1], 1, 1, 1
        )
        normalized = work / basis_scale.clamp_min(self.config.epsilon)
        normalized = normalized * latent_rms
        active = basis_scale > self.config.epsilon
        return torch.where(active, normalized, torch.zeros_like(normalized))

    def _spatial_residual(self, latent: Tensor, bases: Tensor, context: Tensor) -> Tensor:
        if self.spatial_head is None:
            return torch.zeros_like(latent, dtype=torch.float32)
        feature = torch.cat((latent.float(), bases.mean(dim=1)), dim=1)
        feature = self.spatial_head["depthwise"](feature)
        feature = F.silu(self.spatial_head["pointwise"](feature))
        film_scale, film_shift = self.spatial_head["film"](context).chunk(2, dim=-1)
        film_scale = torch.tanh(film_scale).unsqueeze(-1).unsqueeze(-1)
        film_shift = film_shift.unsqueeze(-1).unsqueeze(-1)
        feature = feature * (1.0 + film_scale) + film_shift
        local = torch.tanh(self.spatial_head["output"](feature))
        latent_rms = _vector_norm(latent).reshape(-1, 1, 1, 1) / math.sqrt(
            latent[0].numel()
        )
        return local * latent_rms * float(self.config.spatial_bound)

    def forward(
        self,
        latent: Tensor,
        bases: Tensor,
        *,
        timestep: Optional[Tensor] = None,
        prompt_embedding: Optional[Tensor] = None,
        state_features: Optional[Tensor] = None,
        scheduler_update: Optional[Tensor] = None,
    ) -> RendererOutput:
        _require_nchw(latent, "latent")
        batch = latent.shape[0]
        if scheduler_update is not None:
            _require_nchw(scheduler_update, "scheduler_update")
            if scheduler_update.shape != latent.shape:
                raise ValueError("scheduler_update must match latent shape")
        prepared_bases = self._prepare_bases(latent, bases)
        basis_stats = _basis_statistics(latent, prepared_bases, self.config.epsilon)
        latent_work = latent.float()
        latent_flat = latent_work.flatten(1)
        latent_stats = torch.stack(
            (
                latent_flat.mean(dim=-1),
                latent_flat.std(dim=-1, unbiased=False),
                _vector_norm(latent),
            ),
            dim=-1,
        )
        spectral_stats = _spectral_summary(latent, self.config.epsilon)
        if timestep is None:
            timestep = latent.new_zeros(batch)
        timestep = _coerce_batch_vector(timestep, batch, "timestep")
        timestep_features = _sinusoidal_embedding(
            timestep.to(latent.device), self.config.timestep_dim
        )
        prompt_features = _coerce_context(
            prompt_embedding, batch, self.config.prompt_dim, "prompt_embedding", latent
        )
        state = _coerce_context(
            state_features, batch, self.config.state_dim, "state_features", latent
        )
        context = torch.cat(
            (
                basis_stats,
                latent_stats,
                spectral_stats,
                timestep_features,
                prompt_features,
                state,
            ),
            dim=-1,
        )
        coefficients = torch.tanh(self.policy(context)) * float(
            self.config.coefficient_bound
        )
        basis_update = (
            coefficients[:, :, None, None, None] * prepared_bases
        ).sum(dim=1)
        raw_update = basis_update + self._spatial_residual(
            latent, prepared_bases, context
        )
        if self.config.preserve_moments:
            update = _project_fixed_moment_tangent(
                latent, raw_update, self.config.epsilon
            )
        else:
            update = raw_update.float()
        if self.config.max_update_ratio is not None:
            if scheduler_update is None:
                raise ValueError(
                    "scheduler_update is required when max_update_ratio is configured"
                )
            update = cap_update_norm(
                update,
                scheduler_update,
                float(self.config.max_update_ratio),
                self.config.epsilon,
            )
        observed_postcast_overrun = None
        postcast_cap_applied = None
        postcast_noop_fallback = None
        if self.config.max_update_ratio is not None:
            # The helper preserves the historical path unless strict post-cast
            # enforcement is explicitly enabled in the action configuration.
            (
                guided_float,
                guided_x0,
                observed_postcast_overrun,
                postcast_cap_applied,
                postcast_noop_fallback,
            ) = (
                _cast_guided_x0_with_cap(
                    latent,
                    update,
                    scheduler_update,
                    float(self.config.max_update_ratio),
                    self.config.preserve_moments,
                    self.config.epsilon,
                    self.config.enforce_post_cast_cap,
                )
            )
        else:
            guided_float = _render_guided_x0(
                latent, update, self.config.preserve_moments, self.config.epsilon
            )
            guided_x0 = guided_float.to(latent.dtype)
        precast_residual = guided_float - latent.float()
        # Preserve the historical output dtype; diagnostics use float32 views.
        residual = guided_x0 - latent
        raw_norm = _vector_norm(raw_update)
        precast_norm = _vector_norm(precast_residual)
        postcast_norm = _vector_norm(residual)
        bounded_norm = postcast_norm
        scheduler_norm = (
            _vector_norm(scheduler_update) if scheduler_update is not None else None
        )
        precast_ratio = (
            precast_norm / scheduler_norm.clamp_min(self.config.epsilon)
            if scheduler_norm is not None
            else None
        )
        postcast_ratio = (
            postcast_norm / scheduler_norm.clamp_min(self.config.epsilon)
            if scheduler_norm is not None
            else None
        )
        precast_overrun = (
            (precast_ratio - float(self.config.max_update_ratio)).clamp_min(0.0)
            if precast_ratio is not None and self.config.max_update_ratio is not None
            else None
        )
        postcast_overrun = (
            (postcast_ratio - float(self.config.max_update_ratio)).clamp_min(0.0)
            if postcast_ratio is not None and self.config.max_update_ratio is not None
            else None
        )
        mean_error = (
            guided_x0.float().mean(dim=(-2, -1)) - latent.float().mean(dim=(-2, -1))
        ).abs().amax(dim=1)
        variance_error = (
            guided_x0.float().var(dim=(-2, -1), correction=0)
            - latent.float().var(dim=(-2, -1), correction=0)
        ).abs().amax(dim=1)
        diagnostics = RendererDiagnostics(
            raw_update_norm=raw_norm,
            bounded_update_norm=bounded_norm,
            scheduler_update_norm=scheduler_norm,
            # ``update_ratio`` remains the actual injected post-cast ratio for
            # compatibility with existing selection and audit consumers.
            update_ratio=postcast_ratio,
            mean_error=mean_error,
            variance_error=variance_error,
            precast_update_norm=precast_norm,
            precast_update_ratio=precast_ratio,
            precast_overrun=precast_overrun,
            postcast_update_norm=postcast_norm,
            postcast_update_ratio=postcast_ratio,
            postcast_overrun=postcast_overrun,
            observed_postcast_overrun=observed_postcast_overrun,
            postcast_cap_applied=postcast_cap_applied,
            postcast_noop_fallback=postcast_noop_fallback,
        )
        return RendererOutput(
            guided_x0=guided_x0,
            residual=residual,
            coefficients=coefficients,
            diagnostics=diagnostics,
        )


def build_fixed_coefficient_renderer(
    coefficients,
    *,
    latent_channels: int = 4,
    coefficient_bound: float = 1.0,
    max_update_ratio: Optional[float] = 0.05,
    preserve_moments: bool = True,
    enforce_post_cast_cap: bool = False,
) -> StructuralLatentRenderer:
    """Construct a parameter-free-in-behavior renderer for basis search.

    The returned module uses the same constrained geometry and diagnostics as
    :class:`StructuralLatentRenderer`, but all policy weights are zeroed and
    the final bias is set so every sample emits the supplied coefficients.
    Coefficients must be strictly inside the bound so the inverse ``atanh`` is
    finite.  Prompt/state conditioning is intentionally disabled for this
    fixed-action control.
    """
    values = torch.as_tensor(coefficients, dtype=torch.float32)
    if values.ndim != 1 or values.numel() <= 0:
        raise ValueError("coefficients must be a non-empty one-dimensional sequence")
    if not torch.isfinite(values).all():
        raise ValueError("coefficients must be finite")
    bound = float(coefficient_bound)
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("coefficient_bound must be finite and positive")
    if torch.any(values.abs() >= bound):
        raise ValueError("fixed coefficients must be strictly inside coefficient_bound")
    renderer = StructuralLatentRenderer(
        LatentRendererConfig(
            num_bases=int(values.numel()),
            latent_channels=int(latent_channels),
            prompt_dim=0,
            state_dim=0,
            hidden_dim=1,
            depth=1,
            timestep_dim=0,
            coefficient_bound=bound,
            spatial_hidden_dim=0,
            max_update_ratio=max_update_ratio,
            preserve_moments=preserve_moments,
            enforce_post_cast_cap=enforce_post_cast_cap,
        )
    )
    with torch.no_grad():
        for parameter in renderer.policy.parameters():
            parameter.zero_()
        renderer.policy[-1].bias.copy_(torch.atanh(values / bound))
    return renderer


def _resolve_module_path(root: nn.Module, path: str) -> nn.Module:
    """Resolve attributes and numeric ``ModuleList`` components explicitly."""
    module: Any = root
    for component in str(path).split("."):
        if component.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
            index = int(component)
            if index >= len(module):
                raise AttributeError(f"module path {path!r} index {index} is out of range")
            module = module[index]
        elif hasattr(module, component):
            module = getattr(module, component)
        else:
            raise AttributeError(
                f"module path {path!r} missing component {component!r}"
            )
    if not isinstance(module, nn.Module):
        raise TypeError(f"module path {path!r} does not resolve to a module")
    return module


class StructuralUNetFeatureCapture:
    """Capture one UNet backbone/skip pair and optional self-attention Q/K.

    Hooks are active only inside :meth:`capture_forward`; they observe the
    ordinary denoiser call and detach their outputs so a provider cannot keep
    the full UNet autograd graph alive.  SDXL's classifier-free guidance rows
    are reduced to the positive half when the provider is queried.
    """

    def __init__(
        self,
        unet: nn.Module,
        *,
        batch_size: int,
        do_classifier_free_guidance: bool,
        feature_block: str = "up_blocks.0",
        attention_layer: Optional[str] = (
            "up_blocks.0.attentions.0.transformer_blocks.0.attn1"
        ),
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.do_classifier_free_guidance = bool(do_classifier_free_guidance)
        self.feature_module = _resolve_module_path(unet, feature_block)
        self.attention_module = (
            _resolve_module_path(unet, attention_layer)
            if attention_layer is not None
            else None
        )
        self.qk_capture = (
            QKCapture(self.attention_module)
            if self.attention_module is not None
            else None
        )
        self.backbone: Optional[Tensor] = None
        self.skip: Optional[Tensor] = None

    @staticmethod
    def _as_tensor(value: Any, name: str) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 4:
            raise RuntimeError(f"captured {name} must be a four-dimensional tensor")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"captured {name} contains non-finite values")
        return value.detach()

    def _capture_feature_inputs(self, _module, args, kwargs) -> None:
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        hidden = self._as_tensor(hidden, "backbone feature")
        residuals = kwargs.get("res_hidden_states_tuple")
        if residuals is None and len(args) > 1:
            residuals = args[1]
        if residuals is None:
            raise RuntimeError("the selected UNet block did not expose skip features")
        candidates = [
            value
            for value in reversed(tuple(residuals))
            if isinstance(value, Tensor)
            and value.ndim == 4
            and tuple(value.shape[-2:]) == tuple(hidden.shape[-2:])
        ]
        if not candidates:
            raise RuntimeError(
                "the selected UNet block has no skip feature at the backbone resolution"
            )
        self.backbone = hidden
        self.skip = self._as_tensor(candidates[0], "skip feature")

    @contextmanager
    def capture_forward(self) -> Iterator["StructuralUNetFeatureCapture"]:
        self.backbone = None
        self.skip = None
        feature_handle = self.feature_module.register_forward_pre_hook(
            self._capture_feature_inputs, with_kwargs=True
        )
        qk_context = (
            self.qk_capture.forward() if self.qk_capture is not None else nullcontext()
        )
        try:
            with qk_context:
                yield self
        finally:
            feature_handle.remove()

    def _select_cfg_rows(self, value: Tensor, name: str) -> Tensor:
        if value.shape[0] == self.batch_size:
            return value
        if (
            self.do_classifier_free_guidance
            and value.shape[0] == 2 * self.batch_size
        ):
            return value[self.batch_size :]
        raise RuntimeError(
            f"captured {name} has batch {value.shape[0]}, expected "
            f"{self.batch_size} or {2 * self.batch_size} with CFG"
        )

    def conditional_features(self) -> Tuple[Tensor, Tensor]:
        if self.backbone is None or self.skip is None:
            raise RuntimeError(
                "no UNet structural features were captured; wrap the denoiser call "
                "in capture_forward()"
            )
        return (
            self._select_cfg_rows(self.backbone, "backbone feature"),
            self._select_cfg_rows(self.skip, "skip feature"),
        )

    def conditional_qk(self) -> Optional[Tuple[Tensor, Tensor]]:
        if self.qk_capture is None:
            return None
        query = self.qk_capture.query
        key = self.qk_capture.key
        if query is None or key is None:
            raise RuntimeError(
                "no self-attention Q/K tensors were captured; check attention_layer"
            )
        return (
            self._select_cfg_rows(query, "query"),
            self._select_cfg_rows(key, "key"),
        )


class StructuralUNetBasisProvider:
    """Build the registered six latent bases from one frozen UNet forward.

    The provider has no trainable parameters.  Channel reduction and prompt
    pooling are deterministic, while the semantic graph uses the selected
    self-attention Q/K tensors.  It is therefore suitable for fixed-action
    LR-1 audits before introducing a learned renderer.
    """

    def __init__(
        self,
        unet: nn.Module,
        *,
        batch_size: int,
        do_classifier_free_guidance: bool,
        latent_channels: int = 4,
        semantic_mode: str = "reciprocal_semantic",
        semantic_topk: int = 16,
        semantic_layer: Optional[str] = (
            "up_blocks.0.attentions.0.transformer_blocks.0.attn1"
        ),
        feature_block: str = "up_blocks.0",
        permutation_seed: int = 1729,
        prompt_dim: int = 32,
        state_dim: int = 16,
    ) -> None:
        if int(latent_channels) <= 0:
            raise ValueError("latent_channels must be positive")
        if int(semantic_topk) <= 0:
            raise ValueError("semantic_topk must be positive")
        if semantic_mode not in {
            "clean_tfsa",
            "reciprocal_latent",
            "reciprocal_semantic",
            "reciprocal_semantic_permuted",
        }:
            raise ValueError(f"unsupported semantic_mode {semantic_mode!r}")
        if int(prompt_dim) < 0 or int(state_dim) < 0:
            raise ValueError("prompt_dim and state_dim must be non-negative")
        self.latent_channels = int(latent_channels)
        self.semantic_mode = semantic_mode
        self.semantic_topk = int(semantic_topk)
        self.permutation_seed = int(permutation_seed)
        self.prompt_dim = int(prompt_dim)
        self.state_dim = int(state_dim)
        self.capture = StructuralUNetFeatureCapture(
            unet,
            batch_size=batch_size,
            do_classifier_free_guidance=do_classifier_free_guidance,
            feature_block=feature_block,
            attention_layer=semantic_layer,
        )
        self.last_diagnostics: Optional[dict] = None

    def capture_forward(self) -> Iterator[StructuralUNetFeatureCapture]:
        """Return the context manager used around the ordinary UNet call."""
        return self.capture.capture_forward()

    @staticmethod
    def _reduce_channels(value: Tensor, channels: int) -> Tensor:
        if value.shape[1] == channels:
            return value
        groups = int(math.ceil(value.shape[1] / channels))
        padded_channels = groups * channels
        if padded_channels != value.shape[1]:
            value = torch.cat(
                (
                    value,
                    value.new_zeros(
                        value.shape[0],
                        padded_channels - value.shape[1],
                        value.shape[2],
                        value.shape[3],
                    ),
                ),
                dim=1,
            )
        return value.reshape(
            value.shape[0], channels, groups, value.shape[2], value.shape[3]
        ).mean(dim=2)

    @staticmethod
    def _normalize_feature(value: Tensor, epsilon: float = 1e-6) -> Tensor:
        mean = value.float().mean(dim=(1, 2, 3), keepdim=True)
        centered = value.float() - mean
        rms = centered.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        return centered / rms.clamp_min(epsilon)

    @staticmethod
    def _rms(value: Tensor, epsilon: float = 1e-6) -> Tensor:
        return _vector_norm(value).div(math.sqrt(value[0].numel())).clamp_min(epsilon)

    def _prompt_features(
        self, pooled_prompt_embeds: Optional[Tensor], batch: int, reference: Tensor
    ) -> Optional[Tensor]:
        if self.prompt_dim == 0:
            if pooled_prompt_embeds is not None and pooled_prompt_embeds.numel() > 0:
                return None
            return reference.new_empty((batch, 0), dtype=torch.float32)
        if pooled_prompt_embeds is None:
            raise ValueError("pooled_prompt_embeds is required for prompt conditioning")
        value = pooled_prompt_embeds
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError("pooled_prompt_embeds must have shape (batch, features)")
        if value.shape[0] not in (1, batch):
            raise ValueError("pooled_prompt_embeds batch does not match latent batch")
        if value.shape[0] == 1:
            value = value.expand(batch, -1)
        value = value.to(device=reference.device, dtype=torch.float32)
        pooled = F.adaptive_avg_pool1d(
            value.unsqueeze(1), self.prompt_dim
        ).squeeze(1)
        pooled = pooled - pooled.mean(dim=-1, keepdim=True)
        return pooled / pooled.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)

    def _state_features(
        self,
        observation: RendererObservation,
        x0: Tensor,
        backbone: Tensor,
        skip: Tensor,
        freeu: Tensor,
        laplacian: Tensor,
        entropy: Tensor,
    ) -> Optional[Tensor]:
        if self.state_dim == 0:
            return x0.new_empty((x0.shape[0], 0), dtype=torch.float32)
        timestep = _coerce_batch_vector(
            observation.normalized_timestep, x0.shape[0], "normalized_timestep"
        ).to(device=x0.device, dtype=torch.float32)
        raw_timestep = _coerce_batch_vector(
            observation.timestep, x0.shape[0], "timestep"
        ).to(device=x0.device, dtype=torch.float32)
        x0_work = x0.float()
        scheduler_rms = self._rms(observation.scheduler_update)
        x0_rms = self._rms(x0)
        spectral = _spectral_summary(x0, 1e-6)
        confidence = (1.0 - entropy).clamp(0.0, 1.0)
        full = torch.stack(
            (
                timestep,
                (raw_timestep / 1000.0).clamp(0.0, 1.0),
                x0_work.mean(dim=(1, 2, 3)),
                x0_work.std(dim=(1, 2, 3), unbiased=False),
                x0_rms,
                scheduler_rms,
                scheduler_rms / x0_rms,
                entropy,
                confidence,
                spectral[:, 0],
                spectral[:, 1],
                spectral[:, 2],
                self._rms(backbone),
                self._rms(skip),
                self._rms(freeu),
                self._rms(laplacian),
            ),
            dim=-1,
        )
        if self.state_dim == full.shape[1]:
            return full
        return F.adaptive_avg_pool1d(full.unsqueeze(1), self.state_dim).squeeze(1)

    def __call__(self, observation: RendererObservation) -> RendererCondition:
        x0 = observation.pred_original_sample
        _require_nchw(x0, "observation.pred_original_sample")
        if x0.shape[1] != self.latent_channels:
            raise ValueError(
                f"predicted clean latent has {x0.shape[1]} channels; "
                f"expected {self.latent_channels}"
            )
        if observation.scheduler_update.shape != x0.shape:
            raise ValueError("scheduler_update must match predicted clean latent")
        backbone, skip = self.capture.conditional_features()
        qk = self.capture.conditional_qk()
        if qk is None:
            grid_height = min(x0.shape[-2], max(1, round(x0.shape[-2] / 4)))
            grid_width = min(x0.shape[-1], max(1, round(x0.shape[-1] / 4)))
            tokens = F.interpolate(
                backbone.float(), size=(grid_height, grid_width), mode="area"
            ).flatten(2).transpose(1, 2)
            graph, entropy = affinity_from_tokens(
                tokens,
                mode=self.semantic_mode,
                topk=self.semantic_topk,
                permutation_seed=self.permutation_seed,
            )
        else:
            query, key = qk
            grid_height, grid_width = infer_token_grid(
                query.shape[1], x0.shape[-2], x0.shape[-1]
            )
            graph, entropy = affinity_from_qk(
                query,
                key,
                attention_module=self.capture.attention_module,
                mode=self.semantic_mode,
                topk=self.semantic_topk,
                permutation_seed=self.permutation_seed,
            )
        semantic = build_graph_transport_basis(
            x0, graph, grid_height, grid_width
        )[:, 0]
        spectral = build_spectral_bases(x0)
        backbone_latent = self._reduce_channels(backbone, self.latent_channels)
        skip_latent = self._reduce_channels(skip, self.latent_channels)
        backbone_latent = self._normalize_feature(backbone_latent)
        skip_latent = self._normalize_feature(skip_latent)
        freeu = build_feature_difference_basis(
            backbone_latent,
            skip_latent,
            tuple(x0.shape[-2:]),
        )[:, 0]
        laplacian = build_laplacian_basis(x0)[:, 0]
        bases = torch.cat(
            (semantic[:, None], spectral, freeu[:, None], laplacian[:, None]), dim=1
        ).to(dtype=x0.dtype)
        prompt_features = self._prompt_features(
            observation.pooled_prompt_embeds, x0.shape[0], x0
        )
        state_features = self._state_features(
            observation,
            x0,
            backbone_latent,
            skip_latent,
            freeu,
            laplacian,
            entropy,
        )
        self.last_diagnostics = {
            "semantic_graph_mode": self.semantic_mode,
            "semantic_graph_topk": self.semantic_topk,
            "semantic_token_grid": [int(grid_height), int(grid_width)],
            "semantic_entropy": entropy.detach().float().cpu().tolist(),
            "backbone_shape": list(backbone.shape),
            "skip_shape": list(skip.shape),
            "basis_rms": self._rms(bases.reshape(x0.shape[0], -1, *x0.shape[1:])).detach()
            .float()
            .cpu()
            .tolist(),
        }
        return RendererCondition(
            bases=bases,
            prompt_embedding=prompt_features,
            state_features=state_features,
        )
