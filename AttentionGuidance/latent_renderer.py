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

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


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


@dataclass(frozen=True)
class RendererDiagnostics:
    """Auditable quantities emitted by a renderer forward pass."""

    raw_update_norm: Tensor
    bounded_update_norm: Tensor
    scheduler_update_norm: Optional[Tensor]
    update_ratio: Optional[Tensor]
    mean_error: Tensor
    variance_error: Tensor


@dataclass(frozen=True)
class RendererOutput:
    """Rendered clean latent, residual, coefficients, and safety diagnostics."""

    guided_x0: Tensor
    residual: Tensor
    coefficients: Tensor
    diagnostics: RendererDiagnostics


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
    prev_sample: Tensor, pred_original_sample: Tensor, guided_x0: Tensor
) -> Tensor:
    """Inject a rendered ``x0`` while retaining the scheduler's own step."""
    _require_nchw(prev_sample, "prev_sample")
    _require_nchw(pred_original_sample, "pred_original_sample")
    _require_nchw(guided_x0, "guided_x0")
    if (
        prev_sample.shape != pred_original_sample.shape
        or guided_x0.shape != prev_sample.shape
    ):
        raise ValueError("scheduler samples and guided_x0 must have identical shapes")
    return (prev_sample + guided_x0 - pred_original_sample).to(prev_sample.dtype)


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
        )
        return RendererOutput(
            guided_x0=guided_x0,
            residual=residual,
            coefficients=coefficients,
            diagnostics=diagnostics,
        )
