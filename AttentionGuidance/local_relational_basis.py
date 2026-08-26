"""Local relational controls on a fixed predicted-clean latent grid.

This module is intentionally separate from the registered six-basis renderer
contract.  It constructs three local graph residuals using uniform,
predicted-clean, counter-keyed random-edge, or caller-supplied non-attention
affinities and does not install hooks, run a denoiser, or make quality
decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import re
from typing import Dict, List, Optional, Set, Tuple

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
    "random_edge",
)
RANDOM_EDGE_COUNTER_SCHEMA = "ao-random-edge-counter-v1"
_RANDOM_EDGE_COUNTER_STRING = re.compile(r"[A-Za-z0-9._:-]+")
_RANDOM_EDGE_DENOMINATOR = 1 << 24
_RANDOM_EDGE_KEY = Tuple[str, str, str, int, int]
_ORBIT_OFFSETS_BY_NAME = dict(LOCAL_RELATIONAL_OFFSET_ORBITS)


def _counter_string(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or _RANDOM_EDGE_COUNTER_STRING.fullmatch(value) is None
    ):
        raise ValueError(f"{name} must match [A-Za-z0-9._:-]+")
    return value


def _counter_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def canonical_random_edge_nodes(
    grid_size: int,
    edge_low: int,
    edge_high: int,
) -> Tuple[int, int]:
    """Return the lexicographically first D4 image of one undirected edge."""
    size = _counter_integer(grid_size, "grid_size")
    if size < 2:
        raise ValueError("grid_size must be an integer of at least 2")
    low = _counter_integer(edge_low, "edge_low")
    high = _counter_integer(edge_high, "edge_high")
    if low >= high:
        raise ValueError("random edge node ids must satisfy edge_low < edge_high")
    if high >= size * size:
        raise ValueError("random edge node ids must be inside the registered grid")

    endpoints = (divmod(low, size), divmod(high, size))

    def images(y: int, x: int) -> Tuple[Tuple[int, int], ...]:
        last = size - 1
        return (
            (y, x),
            (x, last - y),
            (last - y, last - x),
            (last - x, y),
            (y, last - x),
            (last - x, last - y),
            (last - y, x),
            (x, y),
        )

    endpoint_images = tuple(images(*endpoint) for endpoint in endpoints)
    representatives = []
    for transform_index in range(8):
        transformed = sorted(
            y * size + x
            for y, x in (
                endpoint_images[0][transform_index],
                endpoint_images[1][transform_index],
            )
        )
        representatives.append((transformed[0], transformed[1]))
    return min(representatives)


def _random_edge_key(
    *,
    experiment_id: str,
    split_role: str,
    prompt_row_id: str,
    seed: int,
    step_index: int,
) -> _RANDOM_EDGE_KEY:
    return (
        _counter_string(experiment_id, "experiment_id"),
        _counter_string(split_role, "split_role"),
        _counter_string(prompt_row_id, "prompt_row_id"),
        _counter_integer(seed, "seed"),
        _counter_integer(step_index, "step_index"),
    )


def _validated_random_edge_counter_bytes(
    key: _RANDOM_EDGE_KEY,
    *,
    orbit_name: str,
    edge_low: int,
    edge_high: int,
) -> bytes:
    experiment_id, split_role, prompt_row_id, seed, step_index = key
    counter = {
        "schema": RANDOM_EDGE_COUNTER_SCHEMA,
        "experiment_id": experiment_id,
        "split_role": split_role,
        "prompt_row_id": prompt_row_id,
        "seed": seed,
        "step_index": step_index,
        "orbit_name": orbit_name,
        "edge_low": edge_low,
        "edge_high": edge_high,
    }
    return json.dumps(
        counter,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def random_edge_counter_bytes(
    *,
    experiment_id: str,
    split_role: str,
    prompt_row_id: str,
    seed: int,
    step_index: int,
    orbit_name: str,
    edge_low: int,
    edge_high: int,
) -> bytes:
    """Return the protocol's exact canonical counter bytes for one edge."""
    key = _random_edge_key(
        experiment_id=experiment_id,
        split_role=split_role,
        prompt_row_id=prompt_row_id,
        seed=seed,
        step_index=step_index,
    )
    orbit = _counter_string(orbit_name, "orbit_name")
    if orbit not in LOCAL_RELATIONAL_ORBIT_NAMES:
        raise ValueError(f"orbit_name must be one of {LOCAL_RELATIONAL_ORBIT_NAMES}")
    low = _counter_integer(edge_low, "edge_low")
    high = _counter_integer(edge_high, "edge_high")
    if low >= high:
        raise ValueError("random edge node ids must satisfy edge_low < edge_high")
    return _validated_random_edge_counter_bytes(
        key,
        orbit_name=orbit,
        edge_low=low,
        edge_high=high,
    )


def _validated_random_edge_uniform(
    key: _RANDOM_EDGE_KEY,
    *,
    orbit_name: str,
    edge_low: int,
    edge_high: int,
) -> float:
    payload = _validated_random_edge_counter_bytes(
        key,
        orbit_name=orbit_name,
        edge_low=edge_low,
        edge_high=edge_high,
    )
    truncated = int.from_bytes(hashlib.sha256(payload).digest()[:3], "big")
    return truncated / _RANDOM_EDGE_DENOMINATOR


def random_edge_uniform(
    *,
    experiment_id: str,
    split_role: str,
    prompt_row_id: str,
    seed: int,
    step_index: int,
    orbit_name: str,
    edge_low: int,
    edge_high: int,
) -> float:
    """Map one validated edge counter to ``k / 2**24`` using SHA-256."""
    payload = random_edge_counter_bytes(
        experiment_id=experiment_id,
        split_role=split_role,
        prompt_row_id=prompt_row_id,
        seed=seed,
        step_index=step_index,
        orbit_name=orbit_name,
        edge_low=edge_low,
        edge_high=edge_high,
    )
    truncated = int.from_bytes(hashlib.sha256(payload).digest()[:3], "big")
    return truncated / _RANDOM_EDGE_DENOMINATOR


def random_edge_counter_set_sha256(
    *,
    experiment_id: str,
    split_role: str,
    prompt_row_id: str,
    seed: int,
    step_index: int,
    orbit_name: str,
    grid_size: int = 16,
) -> str:
    """Hash the sorted D4-canonical counter set for one grid orbit."""

    key = _random_edge_key(
        experiment_id=experiment_id,
        split_role=split_role,
        prompt_row_id=prompt_row_id,
        seed=seed,
        step_index=step_index,
    )
    size = _counter_integer(grid_size, "grid_size")
    if size < 4:
        raise ValueError("grid_size must be an integer of at least 4")
    orbit = _counter_string(orbit_name, "orbit_name")
    offsets = _ORBIT_OFFSETS_BY_NAME.get(orbit)
    if offsets is None:
        raise ValueError(f"orbit_name must be one of {LOCAL_RELATIONAL_ORBIT_NAMES}")

    canonical_edges: Set[Tuple[int, int]] = set()
    for dy, dx in offsets:
        for source_y in range(0, size - dy):
            source_x_start = 0 if dx >= 0 else -dx
            source_x_stop = size - dx if dx >= 0 else size
            for source_x in range(source_x_start, source_x_stop):
                target_y = source_y + dy
                target_x = source_x + dx
                actual = sorted(
                    (
                        source_y * size + source_x,
                        target_y * size + target_x,
                    )
                )
                canonical_edges.add(
                    canonical_random_edge_nodes(size, actual[0], actual[1])
                )

    digest = hashlib.sha256()
    for edge_low, edge_high in sorted(canonical_edges):
        payload = _validated_random_edge_counter_bytes(
            key,
            orbit_name=orbit,
            edge_low=edge_low,
            edge_high=edge_high,
        )
        digest.update(len(payload).to_bytes(4, "big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


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
    random_edge_counter_schema: Optional[str] = None
    random_edge_actual_edge_counts: Optional[Tuple[int, ...]] = None
    random_edge_unique_canonical_key_counts: Optional[Tuple[int, ...]] = None
    random_edge_counter_set_sha256: Optional[Tuple[str, ...]] = None
    random_edge_actual_edges_unique: Optional[bool] = None

    def to_record(self) -> Dict[str, object]:
        """Return detached, JSON-safe diagnostics."""

        def encode(value: Tensor) -> List[List[float]]:
            return value.detach().float().cpu().tolist()

        record = {
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
        if self.affinity_source == "random_edge":
            if (
                self.random_edge_counter_schema is None
                or self.random_edge_actual_edge_counts is None
                or self.random_edge_unique_canonical_key_counts is None
                or self.random_edge_counter_set_sha256 is None
                or self.random_edge_actual_edges_unique is None
            ):
                raise RuntimeError("random-edge diagnostics are incomplete")
            record.update(
                {
                    "random_edge_counter_schema": self.random_edge_counter_schema,
                    "random_edge_actual_edge_counts": list(
                        self.random_edge_actual_edge_counts
                    ),
                    "random_edge_unique_canonical_key_counts": list(
                        self.random_edge_unique_canonical_key_counts
                    ),
                    "random_edge_counter_set_sha256": list(
                        self.random_edge_counter_set_sha256
                    ),
                    "random_edge_actual_edges_unique": (
                        self.random_edge_actual_edges_unique
                    ),
                }
            )
        return record


class LocalRelationalBasisProvider:
    """Construct three local graph controls on registered offset orbits.

    Clean latents and optional non-attention features are area-pooled to
    ``grid_size`` square cells.  Feature and predicted-clean controls use the
    symmetric affinity

    ``a[p, q] = affinity_floor + (1 + clip(cos(control[p], control[q]))) / 2``.

    Uniform-local controls instead assign exactly one to every legal edge.
    Random-edge controls use the first 24 bits of a canonical counter's SHA-256
    digest.  Each row is normalized over in-bounds neighbours.  All modes then
    use the same difference-form transport ``sum_q W[p, q] * (z[q] - z[p])``,
    exactly ``W z - z``, with no boundary padding or wrap.  Inputs and outputs
    are detached so the provider never retains a backbone graph.
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

    def _random_edge_weights(
        self,
        x0_tokens: Tensor,
        source: Tensor,
        target: Tensor,
        *,
        key: _RANDOM_EDGE_KEY,
        orbit_name: str,
        seen_actual_edges: Set[Tuple[int, int]],
        canonical_uniforms: Dict[Tuple[int, int], float],
    ) -> Tensor:
        values = []
        for source_id, target_id in zip(
            source.detach().cpu().tolist(),
            target.detach().cpu().tolist(),
        ):
            low, high = sorted((int(source_id), int(target_id)))
            actual_edge = (low, high)
            if actual_edge in seen_actual_edges:
                raise RuntimeError("random edge orbit produced a duplicate actual edge")
            seen_actual_edges.add(actual_edge)

            canonical_edge = canonical_random_edge_nodes(
                self.grid_size,
                low,
                high,
            )
            uniform = canonical_uniforms.get(canonical_edge)
            if uniform is None:
                uniform = _validated_random_edge_uniform(
                    key,
                    orbit_name=orbit_name,
                    edge_low=canonical_edge[0],
                    edge_high=canonical_edge[1],
                )
                canonical_uniforms[canonical_edge] = uniform
            values.append(uniform)
        uniforms = x0_tokens.new_tensor(values).unsqueeze(0).expand(
            x0_tokens.shape[0], -1
        )
        return self.affinity_floor + uniforms

    def _orbit_basis(
        self,
        x0_tokens: Tensor,
        control_tokens: Optional[Tensor],
        offsets: Tuple[Tuple[int, int], ...],
        affinity_source: str = "feature",
        *,
        random_edge_key: Optional[_RANDOM_EDGE_KEY] = None,
        orbit_name: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, Optional[int]]:
        if affinity_source not in LOCAL_RELATIONAL_AFFINITY_SOURCES:
            raise ValueError(f"unsupported affinity_source {affinity_source!r}")
        if affinity_source in ("uniform_local", "random_edge"):
            if control_tokens is not None:
                raise ValueError(
                    f"{affinity_source} affinity does not accept control tokens"
                )
        elif control_tokens is None or control_tokens.ndim != 3:
            raise ValueError(
                f"{affinity_source} affinity requires rank-three control tokens"
            )
        if affinity_source == "random_edge":
            if random_edge_key is None or orbit_name is None:
                raise ValueError(
                    "random_edge affinity requires a counter key and orbit name"
                )
            if (
                orbit_name not in _ORBIT_OFFSETS_BY_NAME
                or offsets != _ORBIT_OFFSETS_BY_NAME[orbit_name]
            ):
                raise ValueError("random_edge orbit name and offsets differ")
        elif random_edge_key is not None or orbit_name is not None:
            raise ValueError(
                f"{affinity_source} affinity does not accept random-edge counter fields"
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
        seen_actual_random_edges: Set[Tuple[int, int]] = set()
        canonical_random_uniforms: Dict[Tuple[int, int], float] = {}

        for dy, dx in offsets:
            source, target = self._edge_indices(
                self.grid_size, dy, dx, x0_tokens.device
            )
            if affinity_source == "uniform_local":
                weight = x0_tokens.new_ones((batch, source.numel()))
            elif affinity_source == "random_edge":
                weight = self._random_edge_weights(
                    x0_tokens,
                    source,
                    target,
                    key=random_edge_key,
                    orbit_name=orbit_name,
                    seen_actual_edges=seen_actual_random_edges,
                    canonical_uniforms=canonical_random_uniforms,
                )
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

        if (
            affinity_source == "random_edge"
            and len(seen_actual_random_edges) != edge_count
        ):
            raise RuntimeError("random edge orbit failed actual-edge uniqueness")

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
            (
                len(canonical_random_uniforms)
                if affinity_source == "random_edge"
                else None
            ),
        )

    def _build(
        self,
        pred_original_sample: Tensor,
        *,
        affinity_source: str,
        feature: Optional[Tensor] = None,
        random_edge_key: Optional[_RANDOM_EDGE_KEY] = None,
    ) -> Tuple[Tensor, LocalRelationalBasisDiagnostics]:
        if affinity_source == "feature":
            self._validate_inputs(pred_original_sample, feature)
        else:
            self._validate_pred_original_sample(pred_original_sample)
            if feature is not None:
                raise ValueError(f"{affinity_source} affinity does not accept a feature")
        if affinity_source == "random_edge":
            if random_edge_key is None:
                raise ValueError("random_edge affinity requires a counter key")
        elif random_edge_key is not None:
            raise ValueError(
                f"{affinity_source} affinity does not accept a random-edge counter key"
            )
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
        elif affinity_source not in ("uniform_local", "random_edge"):
            raise ValueError(f"unsupported affinity_source {affinity_source!r}")
        if affinity_source not in ("uniform_local", "random_edge"):
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
        canonical_key_counts = []
        counter_set_hashes = []
        for orbit_name, offsets in LOCAL_RELATIONAL_OFFSET_ORBITS:
            (
                residual,
                weights,
                affinity_sums,
                probability_sums,
                edge_count,
                canonical_key_count,
            ) = self._orbit_basis(
                x0_tokens,
                control_tokens,
                offsets,
                affinity_source=affinity_source,
                random_edge_key=random_edge_key,
                orbit_name=(
                    orbit_name if affinity_source == "random_edge" else None
                ),
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
            if affinity_source == "random_edge":
                if canonical_key_count is None:
                    raise RuntimeError("random edge orbit omitted canonical-key count")
                canonical_key_counts.append(canonical_key_count)
                counter_set_hashes.append(
                    random_edge_counter_set_sha256(
                        experiment_id=random_edge_key[0],
                        split_role=random_edge_key[1],
                        prompt_row_id=random_edge_key[2],
                        seed=random_edge_key[3],
                        step_index=random_edge_key[4],
                        orbit_name=orbit_name,
                        grid_size=self.grid_size,
                    )
                )
            elif canonical_key_count is not None:
                raise RuntimeError("non-random orbit reported canonical-key count")

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
            random_edge_counter_schema=(
                RANDOM_EDGE_COUNTER_SCHEMA
                if affinity_source == "random_edge"
                else None
            ),
            random_edge_actual_edge_counts=(
                tuple(edge_counts) if affinity_source == "random_edge" else None
            ),
            random_edge_unique_canonical_key_counts=(
                tuple(canonical_key_counts)
                if affinity_source == "random_edge"
                else None
            ),
            random_edge_counter_set_sha256=(
                tuple(counter_set_hashes)
                if affinity_source == "random_edge"
                else None
            ),
            random_edge_actual_edges_unique=(
                True if affinity_source == "random_edge" else None
            ),
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

    def random_edge(
        self,
        pred_original_sample: Tensor,
        *,
        experiment_id: str,
        split_role: str,
        prompt_row_id: str,
        seed: int,
        step_index: int,
    ) -> Tuple[Tensor, LocalRelationalBasisDiagnostics]:
        """Build one deterministic graph for upper-layer ``+/-`` reuse.

        The same keyed graph is broadcast across batch rows.  No action id or
        sign enters the counter, so antithetic actions must reuse these bases
        rather than request independently keyed graphs.
        """
        key = _random_edge_key(
            experiment_id=experiment_id,
            split_role=split_role,
            prompt_row_id=prompt_row_id,
            seed=seed,
            step_index=step_index,
        )
        return self._build(
            pred_original_sample,
            affinity_source="random_edge",
            random_edge_key=key,
        )


__all__ = [
    "LOCAL_RELATIONAL_AFFINITY_SOURCES",
    "LOCAL_RELATIONAL_OFFSET_ORBITS",
    "LOCAL_RELATIONAL_ORBIT_NAMES",
    "RANDOM_EDGE_COUNTER_SCHEMA",
    "LocalRelationalBasisDiagnostics",
    "LocalRelationalBasisProvider",
    "canonical_random_edge_nodes",
    "random_edge_counter_bytes",
    "random_edge_counter_set_sha256",
    "random_edge_uniform",
]
