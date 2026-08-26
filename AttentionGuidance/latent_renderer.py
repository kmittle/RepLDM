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

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
import math
from typing import Any, Callable, Iterator, Optional, Protocol, Sequence, Tuple

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
    basis_normalization: str = "legacy_l2_to_rms"
    epsilon: float = 1e-6

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
        if self.basis_normalization not in {"legacy_l2_to_rms", "match_rms"}:
            raise ValueError(
                "basis_normalization must be legacy_l2_to_rms or match_rms"
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
    clean_update_gain: Optional[Tensor] = None
    applied_update_norm: Optional[Tensor] = None
    applied_update_ratio: Optional[Tensor] = None

    def to_record(self) -> dict:
        """Return JSON-safe per-sample diagnostics for an experiment sidecar."""

        def encode(value: Optional[Tensor]):
            if value is None:
                return None
            return value.detach().float().cpu().tolist()

        return {
            "raw_update_norm": encode(self.raw_update_norm),
            "bounded_update_norm": encode(self.bounded_update_norm),
            "scheduler_update_norm": encode(self.scheduler_update_norm),
            "update_ratio": encode(self.update_ratio),
            "mean_error": encode(self.mean_error),
            "variance_error": encode(self.variance_error),
            "clean_update_gain": encode(self.clean_update_gain),
            "applied_update_norm": encode(self.applied_update_norm),
            "applied_update_ratio": encode(self.applied_update_ratio),
        }


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


# Canonical names are part of the renderer action contract.  Keep the spectral
# bands separate so a caller can request a genuinely minimal basis without
# constructing semantic or decoder features.
LAZY_LATENT_STRUCTURE_BASIS_NAMES = (
    "semantic",
    "spectral_low",
    "spectral_mid",
    "spectral_high",
    "freeu",
    "laplacian",
)
LAZY_LATENT_STRUCTURE_BASIS_ALIASES = {
    "spectral": ("spectral_low", "spectral_mid", "spectral_high"),
    "semantic_transport": ("semantic",),
    "reciprocal_semantic_transport": ("semantic",),
    "freeu_backbone_minus_skip": ("freeu",),
    "freeu_style": ("freeu",),
    "laplacian_edge": ("laplacian",),
}
LAZY_LATENT_STRUCTURE_PROVIDER_ID = "lazy_latent_structure_basis_v1"
LATENT_STRUCTURE_PROVIDER_IMPLEMENTATION_ALIASES = {
    "lazy": LAZY_LATENT_STRUCTURE_PROVIDER_ID,
    "lazy_latent_structure": LAZY_LATENT_STRUCTURE_PROVIDER_ID,
    "lazy_latent_structure_basis_provider": LAZY_LATENT_STRUCTURE_PROVIDER_ID,
    "LazyLatentStructureBasisProvider": LAZY_LATENT_STRUCTURE_PROVIDER_ID,
    "structural": "structural_unet_basis_v1",
    "structural_unet": "structural_unet_basis_v1",
    "structural_unet_basis_provider": "structural_unet_basis_v1",
    "StructuralUNetBasisProvider": "structural_unet_basis_v1",
}
LATENT_RENDERER_SCHEDULER_MAPPINGS = frozenset(
    {"legacy_unit", "euler_clean_endpoint"}
)
LATENT_RENDERER_BASIS_NORMALIZATIONS = frozenset(
    {"legacy_l2_to_rms", "match_rms"}
)


def normalize_latent_structure_bases(
    requested_bases: Optional[Sequence[str] | str] = None,
) -> Tuple[str, ...]:
    """Normalize a requested basis list while preserving its explicit order."""
    if requested_bases is None:
        values = list(LAZY_LATENT_STRUCTURE_BASIS_NAMES)
    elif isinstance(requested_bases, str):
        values = [requested_bases]
    else:
        if not isinstance(requested_bases, Sequence):
            raise ValueError("requested_bases must be a sequence of basis names")
        values = list(requested_bases)
    expanded = []
    for raw_name in values:
        name = str(raw_name)
        if not name:
            raise ValueError("requested_bases cannot contain an empty basis name")
        names = LAZY_LATENT_STRUCTURE_BASIS_ALIASES.get(name, (name,))
        for canonical in names:
            if canonical not in LAZY_LATENT_STRUCTURE_BASIS_NAMES:
                raise ValueError(f"unsupported latent structure basis {name!r}")
            if canonical in expanded:
                raise ValueError(
                    f"requested_bases contains duplicate basis {canonical!r}"
                )
            expanded.append(canonical)
    return tuple(expanded)


def normalize_latent_structure_provider_implementation(value: Optional[str]) -> str:
    """Normalize public provider class names to a registered implementation id."""
    raw = LAZY_LATENT_STRUCTURE_PROVIDER_ID if value is None else str(value)
    return LATENT_STRUCTURE_PROVIDER_IMPLEMENTATION_ALIASES.get(raw, raw)


def latent_structure_required_hook_names(
    requested_bases: Optional[Sequence[str] | str] = None,
    *,
    semantic_layer: Optional[str] = (
        "up_blocks.0.attentions.0.transformer_blocks.0.attn1"
    ),
    feature_block: str = "up_blocks.0",
) -> Tuple[str, ...]:
    """Return the hooks a lazy provider is allowed to register for a request."""
    names = normalize_latent_structure_bases(requested_bases)
    hooks = []
    if "freeu" in names:
        # This is a mechanism-level name.  The actual PyTorch pre-hook handle
        # is intentionally private and can vary across implementations.
        hooks.append(f"{feature_block}_backbone_skip")
    if "semantic" in names and semantic_layer is not None:
        hooks.append(f"{semantic_layer}_qk")
    return tuple(hooks)


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
    # Keep both Hermitian halves so 90-degree rotations do not change the
    # multiplicity assigned to the horizontal and vertical frequency axes.
    spectrum = torch.fft.fft2(value.float(), dim=(-2, -1), norm="ortho")
    fy = torch.fft.fftfreq(height, device=value.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(width, device=value.device, dtype=torch.float32)
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


def build_spectral_basis(
    reference: Tensor,
    band: str,
    cutoffs: Tuple[float, float] = (0.08, 0.25),
) -> Tensor:
    """Construct exactly one requested Fourier band.

    This deliberately does not call :func:`build_spectral_bases`: a lazy
    provider can therefore prove that an action requesting one band did not
    materialize the other two bands.
    """
    _require_nchw(reference, "reference")
    low_cutoff, mid_cutoff = map(float, cutoffs)
    if not 0 < low_cutoff < mid_cutoff < 0.5:
        raise ValueError("cutoffs must satisfy 0 < low < mid < 0.5")
    band = str(band)
    if band not in {"spectral_low", "spectral_mid", "spectral_high"}:
        raise ValueError(f"unsupported spectral band {band!r}")
    height, width = reference.shape[-2:]
    work = reference.float()
    spectrum = torch.fft.rfft2(work, dim=(-2, -1), norm="ortho")
    fy = torch.fft.fftfreq(height, device=reference.device, dtype=torch.float32)
    fx = torch.fft.rfftfreq(width, device=reference.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    low_pass = torch.exp(-0.5 * (radius / low_cutoff).pow(4))
    mid_pass = torch.exp(-0.5 * (radius / mid_cutoff).pow(4))
    if band == "spectral_low":
        mask = low_pass
    elif band == "spectral_mid":
        mask = mid_pass - low_pass
    else:
        mask = 1.0 - mid_pass
    return torch.fft.irfft2(
        spectrum * mask[None, None],
        s=(height, width),
        dim=(-2, -1),
        norm="ortho",
    ).to(reference.dtype)


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
    prev_sample: Tensor, pred_original_sample: Tensor, guided_x0: Tensor
) -> Tensor:
    """Apply the legacy unit-gain clean-latent displacement after a step.

    This operation translates the scheduler output by the full clean-latent
    displacement. It is retained to reproduce the registered LR-1 runs. A
    coherent Euler transition from the unchanged current sample must instead
    use :func:`inject_euler_clean_update`, whose gain depends on both sigmas.
    """
    _require_nchw(prev_sample, "prev_sample")
    _require_nchw(pred_original_sample, "pred_original_sample")
    _require_nchw(guided_x0, "guided_x0")
    if (
        prev_sample.shape != pred_original_sample.shape
        or guided_x0.shape != prev_sample.shape
    ):
        raise ValueError("scheduler samples and guided_x0 must have identical shapes")
    return (prev_sample + guided_x0 - pred_original_sample).to(prev_sample.dtype)


def euler_clean_update_gain(
    sigma_from: Tensor | float,
    sigma_to: Tensor | float,
    *,
    sigma_hat: Optional[Tensor | float] = None,
) -> Tensor:
    """Return Euler's exact map from a clean-endpoint shift to the next sample.

    With the current noisy sample fixed, replacing ``x0`` by ``x0 + delta``
    changes the Euler derivative by ``-delta / sigma_hat``. The resulting
    next-sample displacement is therefore
    ``(1 - sigma_to / sigma_hat) * delta``. ``sigma_hat`` defaults to
    ``sigma_from``, which is exact when scheduler churn is disabled.
    """

    source = torch.as_tensor(sigma_from, dtype=torch.float64)
    target = torch.as_tensor(sigma_to, dtype=torch.float64, device=source.device)
    effective = (
        source
        if sigma_hat is None
        else torch.as_tensor(sigma_hat, dtype=torch.float64, device=source.device)
    )
    try:
        source, target, effective = torch.broadcast_tensors(source, target, effective)
    except RuntimeError as exc:
        raise ValueError("Euler sigmas must be broadcast-compatible") from exc
    if not (
        torch.isfinite(source).all()
        and torch.isfinite(target).all()
        and torch.isfinite(effective).all()
    ):
        raise ValueError("Euler sigmas must be finite")
    if torch.any(source <= 0) or torch.any(effective <= 0):
        raise ValueError("sigma_from and sigma_hat must be positive")
    if torch.any(target < 0) or torch.any(target > effective):
        raise ValueError("sigma_to must satisfy 0 <= sigma_to <= sigma_hat")
    return (effective - target) / effective


@dataclass(frozen=True)
class EulerCleanEndpoint:
    """Analytic Euler state needed by a pre-step clean-latent renderer."""

    pred_original_sample: Tensor
    nominal_update: Tensor
    clean_update_gain: Tensor
    sigma_from: Tensor
    sigma_to: Tensor
    prediction_type: str


@dataclass(frozen=True)
class EulerMappedIntervention:
    """Measured intervention after Euler output casting and ``step`` mapping."""

    native_prev_sample: Tensor
    intervention: Tensor
    nominal_update: Tensor
    intervention_norm: Tensor
    nominal_update_norm: Tensor
    ratio: Tensor


def measure_euler_mapped_intervention(
    sample: Tensor,
    native_prev_sample: Tensor,
    guided_prev_sample: Tensor,
) -> EulerMappedIntervention:
    """Measure a guided Euler step against its no-extra-call native baseline.

    The pinned Euler scheduler computes in float32 and casts ``prev_sample`` to
    its model-output dtype.  Reproducing that final cast yields the exact native
    comparator without invoking ``scheduler.step`` a second time.
    """

    for value, name in (
        (sample, "sample"),
        (native_prev_sample, "native_prev_sample"),
        (guided_prev_sample, "guided_prev_sample"),
    ):
        _require_nchw(value, name)
    if sample.shape != native_prev_sample.shape or sample.shape != guided_prev_sample.shape:
        raise ValueError("Euler mapped-intervention tensors must have identical shapes")
    if sample.device != native_prev_sample.device or sample.device != guided_prev_sample.device:
        raise ValueError("Euler mapped-intervention tensors must share one device")

    work = sample.float()
    mapped_nominal_update = native_prev_sample.float() - work
    intervention = guided_prev_sample.float() - native_prev_sample.float()
    nominal_norm = torch.linalg.vector_norm(
        mapped_nominal_update.flatten(1), dim=1
    )
    intervention_norm = torch.linalg.vector_norm(intervention.flatten(1), dim=1)
    if not (
        torch.isfinite(nominal_norm).all()
        and torch.isfinite(intervention_norm).all()
    ):
        raise RuntimeError("Euler mapped-intervention norms are non-finite")
    if torch.any(nominal_norm <= 1e-12):
        raise RuntimeError("Euler mapped native update norm is zero")
    ratio = intervention_norm / (nominal_norm + 1e-12)
    if not torch.isfinite(ratio).all():
        raise RuntimeError("Euler mapped-intervention ratio is non-finite")
    return EulerMappedIntervention(
        native_prev_sample=native_prev_sample.detach(),
        intervention=intervention.detach(),
        nominal_update=mapped_nominal_update.detach(),
        intervention_norm=intervention_norm.detach(),
        nominal_update_norm=nominal_norm.detach(),
        ratio=ratio.detach(),
    )


def prepare_euler_clean_endpoint(
    sample: Tensor,
    model_output: Tensor,
    *,
    sigma_from: Tensor | float,
    sigma_to: Tensor | float,
    prediction_type: str,
) -> EulerCleanEndpoint:
    """Recover Euler's nominal ``x0`` and transition before ``scheduler.step``.

    This adapter is intentionally limited to a no-churn Euler step. It mirrors
    diffusers' prediction conversions in float32 without importing or mutating
    a scheduler instance.
    """

    _require_nchw(sample, "sample")
    _require_nchw(model_output, "model_output")
    if sample.shape != model_output.shape:
        raise ValueError("sample and model_output must have identical shapes")
    source = torch.as_tensor(
        sigma_from, device=sample.device, dtype=torch.float32
    )
    target = torch.as_tensor(sigma_to, device=sample.device, dtype=torch.float32)
    if source.numel() != 1 or target.numel() != 1:
        raise ValueError("Euler endpoint preparation requires scalar sigmas")
    source = source.reshape(())
    target = target.reshape(())
    gain = euler_clean_update_gain(source, target).to(
        device=sample.device, dtype=torch.float32
    )
    work = sample.float()
    if prediction_type == "epsilon":
        x0 = work - source * model_output
    elif prediction_type in {"sample", "original_sample"}:
        x0 = model_output
    elif prediction_type == "v_prediction":
        x0 = (
            model_output * (-source / torch.sqrt(source.square() + 1.0))
            + work / (source.square() + 1.0)
        )
    else:
        raise ValueError(f"unsupported Euler prediction_type {prediction_type!r}")
    derivative = (work - x0) / source
    nominal_update = derivative * (target - source)
    return EulerCleanEndpoint(
        pred_original_sample=x0,
        nominal_update=nominal_update,
        clean_update_gain=gain.reshape(1),
        sigma_from=source.reshape(1),
        sigma_to=target.reshape(1),
        prediction_type=prediction_type,
    )


def euler_model_output_from_clean_sample(
    sample: Tensor,
    guided_x0: Tensor,
    *,
    sigma_from: Tensor | float,
    prediction_type: str,
    output_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Convert a guided clean endpoint back to Euler's native model output."""

    _require_nchw(sample, "sample")
    _require_nchw(guided_x0, "guided_x0")
    if sample.shape != guided_x0.shape:
        raise ValueError("sample and guided_x0 must have identical shapes")
    sigma = torch.as_tensor(
        sigma_from, device=sample.device, dtype=torch.float32
    )
    if sigma.numel() != 1 or not torch.isfinite(sigma).all() or torch.any(sigma <= 0):
        raise ValueError("sigma_from must be one finite positive scalar")
    work = sample.float()
    clean = guided_x0.float()
    if prediction_type == "epsilon":
        converted = (work - clean) / sigma
    elif prediction_type in {"sample", "original_sample"}:
        converted = clean
    elif prediction_type == "v_prediction":
        converted = (
            work - (1.0 + sigma.square()) * clean
        ) / (sigma * torch.sqrt(1.0 + sigma.square()))
    else:
        raise ValueError(f"unsupported Euler prediction_type {prediction_type!r}")
    return converted.to(output_dtype or sample.dtype)


def predict_euler_no_churn_prev_sample(
    sample: Tensor,
    model_output: Tensor,
    *,
    sigma_from: Tensor | float,
    sigma_to: Tensor | float,
    prediction_type: str,
) -> Tensor:
    """Mirror one pinned Euler ``step`` without advancing scheduler state."""

    _require_nchw(sample, "sample")
    _require_nchw(model_output, "model_output")
    if sample.shape != model_output.shape or sample.device != model_output.device:
        raise ValueError("Euler sample and model_output must share shape and device")
    source = torch.as_tensor(
        sigma_from, device=sample.device, dtype=torch.float32
    )
    target = torch.as_tensor(
        sigma_to, device=sample.device, dtype=torch.float32
    )
    if source.numel() != 1 or target.numel() != 1:
        raise ValueError("Euler no-churn prediction requires scalar sigmas")
    source = source.reshape(())
    target = target.reshape(())
    if (
        not torch.isfinite(source)
        or not torch.isfinite(target)
        or source <= 0
        or target < 0
        or target > source
    ):
        raise ValueError("Euler no-churn sigmas must satisfy 0 <= sigma_to <= sigma_from")
    work = sample.float()
    if prediction_type == "epsilon":
        predicted_x0 = work - source * model_output
    elif prediction_type in {"sample", "original_sample"}:
        predicted_x0 = model_output
    elif prediction_type == "v_prediction":
        predicted_x0 = (
            model_output * (-source / torch.sqrt(source.square() + 1.0))
            + work / (source.square() + 1.0)
        )
    else:
        raise ValueError(f"unsupported Euler prediction_type {prediction_type!r}")
    derivative = (work - predicted_x0) / source
    delta_sigma = target - source
    return (work + derivative * delta_sigma).to(model_output.dtype)


def inject_euler_clean_update(
    prev_sample: Tensor,
    pred_original_sample: Tensor,
    guided_x0: Tensor,
    *,
    sigma_from: Tensor | float,
    sigma_to: Tensor | float,
    sigma_hat: Optional[Tensor | float] = None,
) -> Tensor:
    """Closed-form parity helper for Euler clean-endpoint tests.

    Production integration should convert the guided endpoint with
    :func:`euler_model_output_from_clean_sample` and invoke the scheduler once.
    """

    _require_nchw(prev_sample, "prev_sample")
    _require_nchw(pred_original_sample, "pred_original_sample")
    _require_nchw(guided_x0, "guided_x0")
    if (
        prev_sample.shape != pred_original_sample.shape
        or guided_x0.shape != prev_sample.shape
    ):
        raise ValueError("scheduler samples and guided_x0 must have identical shapes")
    gain = euler_clean_update_gain(
        sigma_from,
        sigma_to,
        sigma_hat=sigma_hat,
    ).to(device=prev_sample.device, dtype=torch.float32)
    if gain.ndim == 0:
        gain = gain.reshape(1)
    if gain.ndim != 1 or gain.shape[0] not in (1, prev_sample.shape[0]):
        raise ValueError("Euler gain must be scalar or have one value per batch item")
    update = (guided_x0.float() - pred_original_sample.float()) * gain.reshape(
        -1, 1, 1, 1
    )
    return (prev_sample.float() + update).to(prev_sample.dtype)


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
    coefficient-only control used by the fixed-basis ablation.
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
        latent_norm = _vector_norm(latent).reshape(-1, 1, 1, 1, 1)
        if self.config.basis_normalization == "legacy_l2_to_rms":
            target_norm = latent_norm / math.sqrt(latent[0].numel())
        else:
            target_norm = latent_norm
        basis_scale = basis_norm.reshape(
            basis_norm.shape[0], basis_norm.shape[1], 1, 1, 1
        )
        normalized = work / basis_scale.clamp_min(self.config.epsilon)
        normalized = normalized * target_norm
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
        clean_update_gain: Optional[Tensor] = None,
    ) -> RendererOutput:
        _require_nchw(latent, "latent")
        batch = latent.shape[0]
        if scheduler_update is not None:
            _require_nchw(scheduler_update, "scheduler_update")
            if scheduler_update.shape != latent.shape:
                raise ValueError("scheduler_update must match latent shape")
        gain = None
        if clean_update_gain is not None:
            gain = _coerce_batch_vector(
                clean_update_gain, batch, "clean_update_gain"
            ).to(device=latent.device, dtype=torch.float32)
            if torch.any(gain <= 0) or torch.any(gain > 1):
                raise ValueError("clean_update_gain must satisfy 0 < gain <= 1")
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
            cap_reference = scheduler_update
            if gain is not None:
                cap_reference = scheduler_update.float() / gain.reshape(-1, 1, 1, 1)
            update = cap_update_norm(
                update,
                cap_reference,
                float(self.config.max_update_ratio),
                self.config.epsilon,
            )
        if self.config.preserve_moments:
            guided_x0 = _fixed_moment_geodesic(latent, update, self.config.epsilon)
        else:
            guided_x0 = latent.float() + update.float()
        guided_x0 = guided_x0.to(latent.dtype)
        residual = guided_x0 - latent
        raw_norm = _vector_norm(raw_update)
        bounded_norm = _vector_norm(residual)
        scheduler_norm = (
            _vector_norm(scheduler_update) if scheduler_update is not None else None
        )
        update_ratio = (
            bounded_norm / scheduler_norm.clamp_min(self.config.epsilon)
            if scheduler_norm is not None
            else None
        )
        applied_update_norm = None
        applied_update_ratio = None
        if gain is not None:
            applied_update_norm = _vector_norm(
                residual.float() * gain.reshape(-1, 1, 1, 1)
            )
            applied_update_ratio = (
                applied_update_norm / scheduler_norm.clamp_min(self.config.epsilon)
                if scheduler_norm is not None
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
            update_ratio=update_ratio,
            mean_error=mean_error,
            variance_error=variance_error,
            clean_update_gain=gain,
            applied_update_norm=applied_update_norm,
            applied_update_ratio=applied_update_ratio,
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
    basis_normalization: str = "legacy_l2_to_rms",
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
            basis_normalization=basis_normalization,
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
        requested_bases: Optional[Sequence[str] | str] = None,
        required_hook_names: Optional[Sequence[str]] = None,
        scheduler_mapping: str = "legacy_unit",
        basis_normalization: str = "legacy_l2_to_rms",
        provider_id: Optional[str] = None,
        provider_provenance_id: Optional[str] = None,
        provenance_id: Optional[str] = None,
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
        if scheduler_mapping not in LATENT_RENDERER_SCHEDULER_MAPPINGS:
            raise ValueError("scheduler_mapping is not registered")
        if basis_normalization not in LATENT_RENDERER_BASIS_NORMALIZATIONS:
            raise ValueError("basis_normalization is not registered")
        requested = normalize_latent_structure_bases(requested_bases)
        if requested != LAZY_LATENT_STRUCTURE_BASIS_NAMES:
            raise ValueError(
                "StructuralUNetBasisProvider constructs all six canonical bases; "
                "use LazyLatentStructureBasisProvider for a subset"
            )
        expected_hooks = latent_structure_required_hook_names(
            requested,
            semantic_layer=semantic_layer,
            feature_block=feature_block,
        )
        if required_hook_names is None:
            hooks = expected_hooks
        elif isinstance(required_hook_names, str):
            raise ValueError("required_hook_names must be a sequence of names")
        else:
            hooks = tuple(str(value) for value in required_hook_names)
            if hooks != expected_hooks:
                raise ValueError(
                    "required_hook_names do not match the requested basis mechanisms"
                )
        supplied_ids = [
            str(value)
            for value in (provider_id, provider_provenance_id, provenance_id)
            if value is not None
        ]
        if any(not value for value in supplied_ids):
            raise ValueError("provider provenance ids must be non-empty strings")
        if supplied_ids and any(value != supplied_ids[0] for value in supplied_ids[1:]):
            raise ValueError("provider_id and provider_provenance_id must agree")
        self.latent_channels = int(latent_channels)
        self.semantic_mode = semantic_mode
        self.semantic_topk = int(semantic_topk)
        self.permutation_seed = int(permutation_seed)
        self.prompt_dim = int(prompt_dim)
        self.state_dim = int(state_dim)
        self.feature_block = str(feature_block)
        self.semantic_layer = None if semantic_layer is None else str(semantic_layer)
        self.requested_bases = requested
        self.required_hook_names = hooks
        self.scheduler_mapping = scheduler_mapping
        self.basis_normalization = basis_normalization
        self.provider_id = supplied_ids[0] if supplied_ids else "structural_unet_basis_v1"
        self.provider_provenance_id = self.provider_id
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

    @staticmethod
    def _basis_rms(value: Tensor, epsilon: float = 1e-6) -> Tensor:
        """Return one RMS value per canonical basis slot (batch, basis)."""
        if value.ndim != 5:
            raise ValueError("basis tensor must have shape (batch, basis, channels, height, width)")
        # Do not clamp here: zero slots are part of the lazy-provider contract
        # and must remain auditable as exact zeroes.
        return value.float().square().mean(dim=(2, 3, 4)).sqrt()

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
            "implementation": "structural_unet_basis_v1",
            "provider_id": self.provider_id,
            "provider_provenance_id": self.provider_provenance_id,
            "requested_bases": list(self.requested_bases),
            "constructed_bases": list(self.requested_bases),
            "registered_hook_names": list(self.required_hook_names),
            "required_hook_names": list(self.required_hook_names),
            "scheduler_mapping": self.scheduler_mapping,
            "basis_normalization": self.basis_normalization,
            "semantic_graph_mode": self.semantic_mode,
            "semantic_graph_topk": self.semantic_topk,
            "semantic_token_grid": [int(grid_height), int(grid_width)],
            "semantic_entropy": entropy.detach().float().cpu().tolist(),
            "backbone_shape": list(backbone.shape),
            "skip_shape": list(skip.shape),
            "basis_rms": self._basis_rms(bases).detach().float().cpu().tolist(),
        }
        return RendererCondition(
            bases=bases,
            prompt_embedding=prompt_features,
            state_features=state_features,
        )


class LazyLatentStructureBasisProvider(StructuralUNetBasisProvider):
    """Construct only the requested latent-structure bases.

    Spectral and Laplacian bases depend only on the predicted clean latent and
    therefore install no UNet hooks.  The semantic basis optionally captures
    self-attention Q/K, while the FreeU-style basis optionally captures one
    decoder backbone/skip pair.  The ordinary denoiser call remains the sole
    UNet evaluation in every mode.
    """

    def __init__(
        self,
        unet: Optional[nn.Module] = None,
        *,
        batch_size: int,
        do_classifier_free_guidance: bool,
        latent_channels: int = 4,
        requested_bases: Optional[Sequence[str] | str] = None,
        required_hook_names: Optional[Sequence[str]] = None,
        semantic_mode: str = "reciprocal_semantic",
        semantic_topk: int = 16,
        semantic_layer: Optional[str] = (
            "up_blocks.0.attentions.0.transformer_blocks.0.attn1"
        ),
        feature_block: str = "up_blocks.0",
        permutation_seed: int = 1729,
        prompt_dim: int = 0,
        state_dim: int = 0,
        scheduler_mapping: str = "legacy_unit",
        basis_normalization: str = "legacy_l2_to_rms",
        provider_id: Optional[str] = None,
        provider_provenance_id: Optional[str] = None,
        provenance_id: Optional[str] = None,
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
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
        if scheduler_mapping not in LATENT_RENDERER_SCHEDULER_MAPPINGS:
            raise ValueError("scheduler_mapping is not registered")
        if basis_normalization not in LATENT_RENDERER_BASIS_NORMALIZATIONS:
            raise ValueError("basis_normalization is not registered")
        supplied_ids = [
            str(value)
            for value in (provider_id, provider_provenance_id, provenance_id)
            if value is not None
        ]
        if any(not value for value in supplied_ids):
            raise ValueError("provider provenance ids must be non-empty strings")
        if supplied_ids and any(value != supplied_ids[0] for value in supplied_ids[1:]):
            raise ValueError("provider_id and provider_provenance_id must agree")
        resolved_provider_id = supplied_ids[0] if supplied_ids else LAZY_LATENT_STRUCTURE_PROVIDER_ID

        names = normalize_latent_structure_bases(requested_bases)
        expected_hooks = latent_structure_required_hook_names(
            names,
            semantic_layer=semantic_layer,
            feature_block=feature_block,
        )
        if required_hook_names is None:
            hooks = expected_hooks
        else:
            if isinstance(required_hook_names, str):
                raise ValueError("required_hook_names must be a sequence of names")
            hooks = tuple(str(value) for value in required_hook_names)
            if hooks != expected_hooks:
                raise ValueError(
                    "required_hook_names do not match the requested basis mechanisms"
                )

        self.latent_channels = int(latent_channels)
        self.semantic_mode = semantic_mode
        self.semantic_topk = int(semantic_topk)
        self.permutation_seed = int(permutation_seed)
        self.prompt_dim = int(prompt_dim)
        self.state_dim = int(state_dim)
        self.batch_size = int(batch_size)
        self.do_classifier_free_guidance = bool(do_classifier_free_guidance)
        self.requested_bases = names
        self.required_hook_names = hooks
        self.provider_id = resolved_provider_id
        self.provider_provenance_id = resolved_provider_id
        self.scheduler_mapping = scheduler_mapping
        self.basis_normalization = basis_normalization
        self.feature_block = str(feature_block)
        self.semantic_layer = None if semantic_layer is None else str(semantic_layer)
        self._needs_semantic = "semantic" in names
        self._needs_freeu = "freeu" in names
        self._feature_capture: Optional[StructuralUNetFeatureCapture] = None
        self._semantic_capture: Optional[QKCapture] = None
        self._semantic_module: Optional[nn.Module] = None

        # Resolve only mechanisms that were requested.  In particular, this
        # leaves spectral/laplacian-only providers independent of any UNet.
        if self._needs_freeu:
            if unet is None:
                raise ValueError("freeu basis requires a UNet decoder")
            self._feature_capture = StructuralUNetFeatureCapture(
                unet,
                batch_size=self.batch_size,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                feature_block=self.feature_block,
                attention_layer=None,
            )
        if self._needs_semantic and self.semantic_layer is not None:
            if unet is None:
                raise ValueError("semantic Q/K basis requires a UNet attention layer")
            self._semantic_module = _resolve_module_path(unet, self.semantic_layer)
            self._semantic_capture = QKCapture(self._semantic_module)
        # ``capture`` is retained as a compatibility/introspection alias for
        # callers that previously inspected StructuralUNetBasisProvider.
        self.capture = self._feature_capture
        self.registered_hook_names: list[str] = []
        self.last_diagnostics: Optional[dict] = None

    @contextmanager
    def capture_forward(self) -> Iterator["LazyLatentStructureBasisProvider"]:
        """Register only the hooks implied by ``requested_bases``."""
        active_names = list(self.required_hook_names)
        self.registered_hook_names = active_names
        try:
            with ExitStack() as stack:
                if self._feature_capture is not None:
                    stack.enter_context(self._feature_capture.capture_forward())
                if self._semantic_capture is not None:
                    stack.enter_context(self._semantic_capture.forward())
                yield self
        finally:
            # Keep the names from the completed ordinary forward available for
            # the sidecar, while ensuring the actual handles are removed.
            self.registered_hook_names = active_names

    def _semantic_condition(
        self,
        x0: Tensor,
        backbone: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[int, int]]]:
        if self._semantic_capture is not None:
            query, key = self._semantic_capture.get_conditional(
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                batch_size=self.batch_size,
            )
            grid_height, grid_width = infer_token_grid(
                query.shape[1], x0.shape[-2], x0.shape[-1]
            )
            graph, entropy = affinity_from_qk(
                query,
                key,
                attention_module=self._semantic_module,
                mode=self.semantic_mode,
                topk=self.semantic_topk,
                permutation_seed=self.permutation_seed,
            )
            return (
                build_graph_transport_basis(
                    x0, graph, grid_height, grid_width
                )[:, 0],
                entropy,
                (grid_height, grid_width),
            )

        source = x0 if backbone is None else backbone
        grid_height = min(x0.shape[-2], max(1, round(x0.shape[-2] / 4)))
        grid_width = min(x0.shape[-1], max(1, round(x0.shape[-1] / 4)))
        tokens = F.interpolate(
            source.float(), size=(grid_height, grid_width), mode="area"
        ).flatten(2).transpose(1, 2)
        graph, entropy = affinity_from_tokens(
            tokens,
            mode=self.semantic_mode,
            topk=self.semantic_topk,
            permutation_seed=self.permutation_seed,
        )
        return (
            build_graph_transport_basis(x0, graph, grid_height, grid_width)[:, 0],
            entropy,
            (grid_height, grid_width),
        )

    def __call__(self, observation: RendererObservation) -> RendererCondition:
        x0 = observation.pred_original_sample
        _require_nchw(x0, "observation.pred_original_sample")
        if x0.shape[0] != self.batch_size:
            raise ValueError("observation batch does not match provider batch_size")
        if x0.shape[1] != self.latent_channels:
            raise ValueError(
                f"predicted clean latent has {x0.shape[1]} channels; "
                f"expected {self.latent_channels}"
            )
        if observation.scheduler_update.shape != x0.shape:
            raise ValueError("scheduler_update must match predicted clean latent")

        backbone = skip = None
        if self._feature_capture is not None:
            backbone, skip = self._feature_capture.conditional_features()

        semantic = None
        entropy = x0.new_zeros((x0.shape[0],), dtype=torch.float32)
        token_grid = None
        if self._needs_semantic:
            semantic, entropy, token_grid = self._semantic_condition(x0, backbone)

        spectral_map = {}
        for name in ("spectral_low", "spectral_mid", "spectral_high"):
            if name in self.requested_bases:
                # Build only the requested band.  In particular, a low-band
                # action must not materialize mid/high tensors as a side effect.
                spectral_map[name] = build_spectral_basis(x0, name)[:, None]

        freeu = None
        backbone_latent = x0.new_zeros(x0.shape)
        skip_latent = x0.new_zeros(x0.shape)
        if self._needs_freeu:
            if backbone is None or skip is None:
                raise RuntimeError("freeu basis requested without captured decoder features")
            backbone_latent = self._normalize_feature(
                self._reduce_channels(backbone, self.latent_channels)
            )
            skip_latent = self._normalize_feature(
                self._reduce_channels(skip, self.latent_channels)
            )
            freeu = build_feature_difference_basis(
                backbone_latent,
                skip_latent,
                tuple(x0.shape[-2:]),
            )[:, 0].to(dtype=x0.dtype)

        laplacian = None
        if "laplacian" in self.requested_bases:
            laplacian = build_laplacian_basis(x0)[:, 0]

        basis_map = {}
        if semantic is not None:
            basis_map["semantic"] = semantic[:, None]
        basis_map.update(spectral_map)
        if freeu is not None:
            basis_map["freeu"] = freeu[:, None]
        if laplacian is not None:
            basis_map["laplacian"] = laplacian[:, None]
        missing = [name for name in self.requested_bases if name not in basis_map]
        if missing:
            raise RuntimeError(f"lazy provider failed to construct bases: {missing}")
        # The renderer action vector is always indexed by the six canonical
        # names.  Lazy construction only controls which slots are populated;
        # every omitted slot remains an explicit zero so coefficient vectors,
        # checkpoints, and sidecars retain a stable shape.
        zero_basis = x0.new_zeros(
            (x0.shape[0], 1, x0.shape[1], x0.shape[2], x0.shape[3])
        )
        bases = torch.cat(
            [basis_map.get(name, zero_basis) for name in LAZY_LATENT_STRUCTURE_BASIS_NAMES],
            dim=1,
        )
        constructed_bases = [
            name for name in self.requested_bases if name in basis_map
        ]

        prompt_features = self._prompt_features(
            observation.pooled_prompt_embeds, x0.shape[0], x0
        )
        state_features = self._state_features(
            observation,
            x0,
            backbone_latent,
            skip_latent,
            freeu if freeu is not None else x0.new_zeros(x0.shape),
            laplacian if laplacian is not None else x0.new_zeros(x0.shape),
            entropy,
        )
        provider_record = {
            "implementation": LAZY_LATENT_STRUCTURE_PROVIDER_ID,
            "provider_id": self.provider_id,
            "provider_provenance_id": self.provider_provenance_id,
            "requested_bases": list(self.requested_bases),
            "constructed_bases": constructed_bases,
            "registered_hook_names": list(self.registered_hook_names),
            "required_hook_names": list(self.required_hook_names),
            "scheduler_mapping": self.scheduler_mapping,
            "basis_normalization": self.basis_normalization,
            "semantic_graph_mode": self.semantic_mode if self._needs_semantic else None,
            "semantic_graph_topk": self.semantic_topk if self._needs_semantic else None,
            "semantic_token_grid": (
                None if token_grid is None else [int(value) for value in token_grid]
            ),
            "semantic_entropy": entropy.detach().float().cpu().tolist(),
            "backbone_shape": None if backbone is None else list(backbone.shape),
            "skip_shape": None if skip is None else list(skip.shape),
            "basis_rms": self._basis_rms(bases).detach().float().cpu().tolist(),
        }
        self.last_diagnostics = provider_record
        return RendererCondition(
            bases=bases.to(dtype=x0.dtype),
            prompt_embedding=prompt_features,
            state_features=state_features,
        )
