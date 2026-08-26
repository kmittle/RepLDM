"""Local relational controls on a fixed predicted-clean latent grid.

This module is intentionally separate from the registered six-basis renderer
contract.  It constructs three local graph residuals using uniform,
predicted-clean, or caller-supplied non-attention affinities and does not
install hooks, run a denoiser, or make quality decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


LOCAL_RELATIONAL_OFFSET_ORBITS = (
    ("axis-r1", ((0, 1), (1, 0))),
    ("diagonal-r1", ((1, 1), (1, -1))),
    ("axis-r2", ((0, 2), (2, 0))),
)
LOCAL_RELATIONAL_ORBIT_NAMES = tuple(
    name for name, _offsets in LOCAL_RELATIONAL_OFFSET_ORBITS
)
LOCAL_RELATIONAL_AFFINITY_SOURCES = (
    "feature",
    "uniform_local",
    "predicted_clean",
)


@dataclass(frozen=True)
class LocalRelationalBasisDiagnostics:
    """Mechanism-level statistics for one local-basis construction."""

    orbit_names: Tuple[str, ...]
    affinity_source: str
    grid_size: int
    feature_norm_epsilon: float
    predicted_clean_norm_epsilon: float
    affinity_floor: float
    undirected_edge_counts: Tuple[int, ...]
    edge_weight_min: Tensor
    edge_weight_max: Tensor
    row_affinity_sum_min: Tensor
    row_affinity_sum_max: Tensor
    row_probability_sum_min: Tensor
    row_probability_sum_max: Tensor
    basis_rms: Tensor

    def to_record(self) -> Dict[str, object]:
        """Return detached, JSON-safe diagnostics."""

        def encode(value: Tensor) -> List[List[float]]:
            return value.detach().float().cpu().tolist()

        return {
            "orbit_names": list(self.orbit_names),
            "affinity_source": self.affinity_source,
            "grid_size": self.grid_size,
            "feature_norm_epsilon": self.feature_norm_epsilon,
            "predicted_clean_norm_epsilon": self.predicted_clean_norm_epsilon,
            "affinity_floor": self.affinity_floor,
            "undirected_edge_counts": list(self.undirected_edge_counts),
            "edge_weight_min": encode(self.edge_weight_min),
            "edge_weight_max": encode(self.edge_weight_max),
            "row_affinity_sum_min": encode(self.row_affinity_sum_min),
            "row_affinity_sum_max": encode(self.row_affinity_sum_max),
            "row_probability_sum_min": encode(self.row_probability_sum_min),
            "row_probability_sum_max": encode(self.row_probability_sum_max),
            "basis_rms": encode(self.basis_rms),
        }


class LocalRelationalBasisProvider:
    """Construct three D4-equivariant local graph controls.

    Clean latents and optional non-attention features are area-pooled to
    ``grid_size`` square cells.  Feature and predicted-clean controls use the
    symmetric affinity

    ``a[p, q] = affinity_floor + (1 + clip(cos(control[p], control[q]))) / 2``.

    Uniform-local controls instead assign exactly one to every legal edge.
    Each row is normalized over in-bounds neighbours.  All modes then use the
    same difference-form transport ``sum_q W[p, q] * (z[q] - z[p])``, exactly
    ``W z - z``, with no boundary padding or wrap.  Inputs and outputs are
    detached so the provider never retains a backbone graph.
    """

    def __init__(
        self,
        *,
        grid_size: int = 16,
        feature_norm_epsilon: float = 1e-6,
        predicted_clean_norm_epsilon: float = 1e-6,
        affinity_floor: float = 1e-6,
    ) -> None:
        if (
            isinstance(grid_size, bool)
            or not isinstance(grid_size, Integral)
            or int(grid_size) < 4
        ):
            raise ValueError("grid_size must be an integer of at least 4")
        self.grid_size = int(grid_size)
        self.feature_norm_epsilon = self._validate_epsilon(
            feature_norm_epsilon, "feature_norm_epsilon"
        )
        self.predicted_clean_norm_epsilon = self._validate_epsilon(
            predicted_clean_norm_epsilon, "predicted_clean_norm_epsilon"
        )
        self.affinity_floor = self._validate_epsilon(
            affinity_floor, "affinity_floor"
        )

    @staticmethod
    def _validate_epsilon(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        result = float(value)
        if not math.isfinite(result) or not 0 < result < 1:
            raise ValueError(f"{name} must be finite and between zero and one")
        if result < torch.finfo(torch.float32).tiny:
            raise ValueError(f"{name} must be representable as a positive float32 value")
        return result

    def _validate_tensor(self, value: Tensor, name: str) -> None:
        if not isinstance(value, Tensor) or value.ndim != 4:
            raise ValueError(
                f"{name} must have shape (batch, channels, height, width)"
            )
        if not torch.is_floating_point(value):
            raise ValueError(f"{name} must have a real floating-point dtype")
        if value.shape[0] <= 0 or value.shape[1] <= 0:
            raise ValueError(f"{name} must have positive batch and channel sizes")
        if min(value.shape[-2:]) < self.grid_size:
            raise ValueError(f"{name} spatial dimensions must be at least grid_size")
        if not torch.isfinite(value.detach()).all():
            raise ValueError(f"{name} contains non-finite values")

    def _validate_pred_original_sample(self, pred_original_sample: Tensor) -> None:
        self._validate_tensor(pred_original_sample, "pred_original_sample")
        if pred_original_sample.shape[1] != 4:
            raise ValueError("pred_original_sample must have exactly four channels")

    def _validate_inputs(
        self, pred_original_sample: Tensor, feature: Tensor
    ) -> None:
        self._validate_pred_original_sample(pred_original_sample)
        self._validate_tensor(feature, "feature")
        if pred_original_sample.shape[0] != feature.shape[0]:
            raise ValueError("feature batch must match pred_original_sample batch")
        if pred_original_sample.device != feature.device:
            raise ValueError("feature and pred_original_sample must use the same device")

    @staticmethod
    def _edge_indices(
        grid_size: int, dy: int, dx: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        if dy < 0 or dy >= grid_size or abs(dx) >= grid_size:
            raise RuntimeError("invalid internal offset")
        source_y = torch.arange(0, grid_size - dy, device=device)
        target_y = source_y + dy
        if dx >= 0:
            source_x = torch.arange(0, grid_size - dx, device=device)
            target_x = source_x + dx
        else:
            source_x = torch.arange(-dx, grid_size, device=device)
            target_x = source_x + dx
        source = (
            source_y[:, None] * grid_size + source_x[None, :]
        ).reshape(-1)
        target = (
            target_y[:, None] * grid_size + target_x[None, :]
        ).reshape(-1)
        return source, target

    def _orbit_basis(
        self,
        x0_tokens: Tensor,
        control_tokens: Optional[Tensor],
        offsets: Tuple[Tuple[int, int], ...],
        affinity_source: str = "feature",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        if affinity_source not in LOCAL_RELATIONAL_AFFINITY_SOURCES:
            raise ValueError(f"unsupported affinity_source {affinity_source!r}")
        if affinity_source == "uniform_local":
            if control_tokens is not None:
                raise ValueError("uniform_local affinity does not accept control tokens")
        elif control_tokens is None or control_tokens.ndim != 3:
            raise ValueError(
                f"{affinity_source} affinity requires rank-three control tokens"
            )
        batch, token_count, _channels = x0_tokens.shape
        if control_tokens is not None and control_tokens.shape[:2] != (
            batch,
            token_count,
        ):
            raise ValueError("control tokens must match x0 batch and token dimensions")
        numerator = x0_tokens.new_zeros(x0_tokens.shape)
        row_affinity_sum = x0_tokens.new_zeros((batch, token_count))
        weights = []
        edge_indices = []
        edge_count = 0

        for dy, dx in offsets:
            source, target = self._edge_indices(
                self.grid_size, dy, dx, x0_tokens.device
            )
            if affinity_source == "uniform_local":
                weight = x0_tokens.new_ones((batch, source.numel()))
            else:
                cosine = (
                    control_tokens.index_select(1, source)
                    * control_tokens.index_select(1, target)
                ).sum(dim=-1).clamp(-1.0, 1.0)
                weight = self.affinity_floor + 0.5 * (1.0 + cosine)
            delta = (
                x0_tokens.index_select(1, target)
                - x0_tokens.index_select(1, source)
            )
            weighted_delta = weight.unsqueeze(-1) * delta
            numerator = numerator.index_add(1, source, weighted_delta)
            numerator = numerator.index_add(1, target, -weighted_delta)
            row_affinity_sum = row_affinity_sum.index_add(1, source, weight)
            row_affinity_sum = row_affinity_sum.index_add(1, target, weight)
            weights.append(weight)
            edge_indices.append((source, target))
            edge_count += source.numel()

        all_weights = torch.cat(weights, dim=1)
        if torch.any(row_affinity_sum <= 0) or not torch.isfinite(row_affinity_sum).all():
            raise RuntimeError("local relational orbit produced invalid row weights")
        row_probability_sum = x0_tokens.new_zeros((batch, token_count))
        for (source, target), weight in zip(edge_indices, weights):
            source_probability = weight / row_affinity_sum.index_select(1, source)
            target_probability = weight / row_affinity_sum.index_select(1, target)
            row_probability_sum = row_probability_sum.index_add(
                1, source, source_probability
            )
            row_probability_sum = row_probability_sum.index_add(
                1, target, target_probability
            )
        if not torch.allclose(
            row_probability_sum,
            torch.ones_like(row_probability_sum),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("local relational orbit is not row-normalized")
        residual = numerator / row_affinity_sum.unsqueeze(-1)
        return (
            residual,
            all_weights,
            row_affinity_sum,
            row_probability_sum,
            edge_count,
        )

    def _build(
        self,
        pred_original_sample: Tensor,
        *,
        affinity_source: str,
        feature: Optional[Tensor] = None,
    ) -> Tuple[Tensor, LocalRelationalBasisDiagnostics]:
        if affinity_source == "feature":
            self._validate_inputs(pred_original_sample, feature)
        else:
            self._validate_pred_original_sample(pred_original_sample)
            if feature is not None:
                raise ValueError(f"{affinity_source} affinity does not accept a feature")
        target_size = tuple(pred_original_sample.shape[-2:])
        grid_shape = (self.grid_size, self.grid_size)

        x0_grid = F.adaptive_avg_pool2d(
            pred_original_sample.detach().float(), grid_shape
        )
        if not torch.isfinite(x0_grid.detach()).all():
            raise ValueError("pred_original_sample is not finite after float32 pooling")

        x0_tokens = x0_grid.flatten(2).transpose(1, 2)
        control_tokens = None
        if affinity_source == "feature":
            feature_grid = F.adaptive_avg_pool2d(
                feature.detach().float(), grid_shape
            )
            if not torch.isfinite(feature_grid).all():
                raise ValueError("feature is not finite after float32 pooling")
            control_grid = feature_grid
            norm_epsilon = self.feature_norm_epsilon
            norm_name = "feature"
        elif affinity_source == "predicted_clean":
            control_grid = x0_grid
            norm_epsilon = self.predicted_clean_norm_epsilon
            norm_name = "predicted clean"
        elif affinity_source != "uniform_local":
            raise ValueError(f"unsupported affinity_source {affinity_source!r}")
        if affinity_source != "uniform_local":
            control_norm = torch.linalg.vector_norm(
                control_grid, dim=1, keepdim=True
            )
            if not torch.isfinite(control_norm).all():
                raise ValueError(f"{norm_name} channel norms are not finite in float32")
            if torch.any(control_norm <= norm_epsilon):
                raise RuntimeError(
                    f"{norm_name} channel norm is below the registered threshold"
                )
            control_unit = control_grid / control_norm
            control_tokens = control_unit.flatten(2).transpose(1, 2)
        coarse_bases = []
        edge_weight_min = []
        edge_weight_max = []
        row_affinity_sum_min = []
        row_affinity_sum_max = []
        row_probability_sum_min = []
        row_probability_sum_max = []
        edge_counts = []
        for _name, offsets in LOCAL_RELATIONAL_OFFSET_ORBITS:
            residual, weights, affinity_sums, probability_sums, edge_count = (
                self._orbit_basis(
                    x0_tokens,
                    control_tokens,
                    offsets,
                    affinity_source=affinity_source,
                )
            )
            coarse_bases.append(
                residual.transpose(1, 2).reshape(
                    pred_original_sample.shape[0], 4, *grid_shape
                )
            )
            edge_weight_min.append(weights.amin(dim=1))
            edge_weight_max.append(weights.amax(dim=1))
            row_affinity_sum_min.append(affinity_sums.amin(dim=1))
            row_affinity_sum_max.append(affinity_sums.amax(dim=1))
            row_probability_sum_min.append(probability_sums.amin(dim=1))
            row_probability_sum_max.append(probability_sums.amax(dim=1))
            edge_counts.append(edge_count)

        bases = torch.stack(coarse_bases, dim=1)
        if target_size != grid_shape:
            bases = F.interpolate(
                bases.flatten(0, 1),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).reshape(pred_original_sample.shape[0], 3, 4, *target_size)
        if not torch.isfinite(bases.detach()).all():
            raise RuntimeError("local relational basis contains non-finite values")
        bases = bases.to(dtype=pred_original_sample.dtype).detach()
        if not torch.isfinite(bases.detach()).all():
            raise RuntimeError(
                "local relational basis is not finite after conversion to x0 dtype"
            )
        diagnostics = LocalRelationalBasisDiagnostics(
            orbit_names=LOCAL_RELATIONAL_ORBIT_NAMES,
            affinity_source=affinity_source,
            grid_size=self.grid_size,
            feature_norm_epsilon=self.feature_norm_epsilon,
            predicted_clean_norm_epsilon=self.predicted_clean_norm_epsilon,
            affinity_floor=self.affinity_floor,
            undirected_edge_counts=tuple(edge_counts),
            edge_weight_min=torch.stack(edge_weight_min, dim=1).detach(),
            edge_weight_max=torch.stack(edge_weight_max, dim=1).detach(),
            row_affinity_sum_min=torch.stack(row_affinity_sum_min, dim=1).detach(),
            row_affinity_sum_max=torch.stack(row_affinity_sum_max, dim=1).detach(),
            row_probability_sum_min=torch.stack(
                row_probability_sum_min, dim=1
            ).detach(),
            row_probability_sum_max=torch.stack(
                row_probability_sum_max, dim=1
            ).detach(),
            basis_rms=bases.float().square().mean(dim=(2, 3, 4)).sqrt().detach(),
        )
        return bases, diagnostics

    def __call__(
        self, pred_original_sample: Tensor, feature: Tensor
    ) -> Tuple[Tensor, LocalRelationalBasisDiagnostics]:
        """Build the backward-compatible non-attention feature control."""
        return self._build(
            pred_original_sample,
            affinity_source="feature",
            feature=feature,
        )

    def uniform_local(
        self, pred_original_sample: Tensor
    ) -> Tuple[Tensor, LocalRelationalBasisDiagnostics]:
        """Build a control with affinity exactly one on every legal edge."""
        return self._build(pred_original_sample, affinity_source="uniform_local")

    def predicted_clean(
        self, pred_original_sample: Tensor
    ) -> Tuple[Tensor, LocalRelationalBasisDiagnostics]:
        """Build affinities and transported values from pooled predicted clean latents."""
        return self._build(pred_original_sample, affinity_source="predicted_clean")


__all__ = [
    "LOCAL_RELATIONAL_AFFINITY_SOURCES",
    "LOCAL_RELATIONAL_OFFSET_ORBITS",
    "LOCAL_RELATIONAL_ORBIT_NAMES",
    "LocalRelationalBasisDiagnostics",
    "LocalRelationalBasisProvider",
]
