"""Auditable runtime primitives for the adaptive local-relational oracle.

The classes in this module deliberately stop short of defining an executable
experiment.  They capture one exact non-attention SDXL feature and map one
registered relational basis to a fixed-ratio Euler clean endpoint.  Prompt
selection, action banks, generation authorization, and quality decisions live
outside this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
import re
from typing import Any, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn

from .latent_renderer import (
    EulerMappedIntervention,
    RendererCondition,
    RendererObservation,
    euler_model_output_from_clean_sample,
    measure_euler_mapped_intervention,
    prepare_euler_clean_endpoint,
    predict_euler_no_churn_prev_sample,
)
from .local_relational_basis import (
    LOCAL_RELATIONAL_AFFINITY_SOURCES,
    LOCAL_RELATIONAL_ORBIT_NAMES,
    RANDOM_EDGE_COUNTER_SCHEMA,
    LocalRelationalBasisProvider,
)


ADAPTIVE_ORACLE_FEATURE_BLOCK = "up_blocks.0"
ADAPTIVE_ORACLE_FEATURE_CHANNELS = 1280
ADAPTIVE_ORACLE_FEATURE_SIZE = (32, 32)
ADAPTIVE_ORACLE_TARGET_UPDATE_RATIO = 0.02
ADAPTIVE_ORACLE_HARD_UPDATE_CAP = 0.05
ADAPTIVE_ORACLE_BASIS_PROVIDER_ID = "adaptive_oracle_local_relational_basis_v1"

_RANDOM_CONTEXT_STRING = re.compile(r"[A-Za-z0-9._:-]+")


@dataclass(frozen=True)
class AdaptiveOracleRandomContext:
    """Immutable trajectory identity for counter-keyed random affinities."""

    experiment_id: str
    split_role: str
    prompt_row_id: str
    seed: int

    def __post_init__(self) -> None:
        for name in ("experiment_id", "split_role", "prompt_row_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or _RANDOM_CONTEXT_STRING.fullmatch(value) is None
            ):
                raise ValueError(
                    f"{name} must be a non-empty string matching "
                    "[A-Za-z0-9._:-]+"
                )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or int(self.seed) < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        object.__setattr__(self, "seed", int(self.seed))

    def to_record(self, *, step_index: int, orbit_name: str) -> dict:
        """Return the exact JSON-safe context shared by all orbit edges."""
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, Integral)
            or int(step_index) < 0
        ):
            raise ValueError("observation.step_index must be a non-negative integer")
        return {
            "schema": RANDOM_EDGE_COUNTER_SCHEMA,
            "experiment_id": self.experiment_id,
            "split_role": self.split_role,
            "prompt_row_id": self.prompt_row_id,
            "seed": self.seed,
            "step_index": int(step_index),
            "orbit_name": orbit_name,
        }


def _resolve_module_path(root: nn.Module, path: str) -> nn.Module:
    module: Any = root
    for component in path.split("."):
        if component.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
            module = module[int(component)]
        elif hasattr(module, component):
            module = getattr(module, component)
        else:
            raise AttributeError(
                f"module path {path!r} is missing component {component!r}"
            )
    if not isinstance(module, nn.Module):
        raise TypeError(f"module path {path!r} does not resolve to a module")
    return module


def _require_nchw(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, channels, height, width)")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must have a floating-point dtype")
    if min(value.shape) <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    if not torch.isfinite(value.detach()).all():
        raise ValueError(f"{name} contains non-finite values")


def _batch_vector(
    value: Tensor | float,
    *,
    batch: int,
    device: torch.device,
    name: str,
) -> Tensor:
    result = torch.as_tensor(value, device=device, dtype=torch.float32)
    if result.ndim == 0:
        result = result.expand(batch)
    elif result.ndim == 1 and result.shape[0] == 1:
        result = result.expand(batch)
    elif result.ndim != 1 or result.shape[0] != batch:
        raise ValueError(f"{name} must be scalar or have one value per batch item")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _vector_norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value.float().flatten(1), dim=1)


def _channel_covariance(value: Tensor) -> Tensor:
    centered = value.float() - value.float().mean(dim=(-2, -1), keepdim=True)
    flat = centered.flatten(2)
    return torch.bmm(flat, flat.transpose(1, 2)) / float(flat.shape[-1])


class AdaptiveOracleFeatureCapture:
    """Capture the conditional input to one exact SDXL ``up_blocks.0`` call.

    The hook accepts only the keyword ``hidden_states`` used by the pinned
    diffusers UNet.  It requires unconditional-then-conditional CFG rows and
    clones the conditional half at hook time, so later in-place mutation cannot
    alter the descriptor input.  One context must contain exactly one forward
    and its captured tensor may be consumed exactly once.
    """

    def __init__(
        self,
        unet: nn.Module,
        *,
        batch_size: int,
        feature_block: str = ADAPTIVE_ORACLE_FEATURE_BLOCK,
        expected_channels: int = ADAPTIVE_ORACLE_FEATURE_CHANNELS,
        expected_size: Tuple[int, int] = ADAPTIVE_ORACLE_FEATURE_SIZE,
    ) -> None:
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(expected_channels, bool) or int(expected_channels) <= 0:
            raise ValueError("expected_channels must be a positive integer")
        if (
            not isinstance(expected_size, tuple)
            or len(expected_size) != 2
            or any(isinstance(item, bool) or int(item) <= 0 for item in expected_size)
        ):
            raise ValueError("expected_size must contain two positive integers")
        self.batch_size = int(batch_size)
        self.feature_block = str(feature_block)
        self.expected_channels = int(expected_channels)
        self.expected_size = tuple(int(item) for item in expected_size)
        self.feature_module = _resolve_module_path(unet, self.feature_block)
        self._active = False
        self._capture_complete = False
        self._hook_calls = 0
        self._consume_calls = 0
        self._conditional: Optional[Tensor] = None

    def _capture(self, _module, _args, kwargs) -> None:
        self._hook_calls += 1
        if self._hook_calls != 1:
            raise RuntimeError("adaptive-oracle feature hook ran more than once")
        if not isinstance(kwargs, dict) or "hidden_states" not in kwargs:
            raise RuntimeError(
                "adaptive-oracle feature hook requires keyword hidden_states"
            )
        hidden = kwargs["hidden_states"]
        _require_nchw(hidden, "captured hidden_states")
        expected_shape = (
            2 * self.batch_size,
            self.expected_channels,
            *self.expected_size,
        )
        if tuple(hidden.shape) != expected_shape:
            raise RuntimeError(
                "adaptive-oracle hidden_states shape differs from the registered "
                f"CFG shape: expected {expected_shape}, got {tuple(hidden.shape)}"
            )
        self._conditional = hidden[self.batch_size :].detach().clone()

    @contextmanager
    def capture_forward(self) -> Iterator["AdaptiveOracleFeatureCapture"]:
        if self._active:
            raise RuntimeError("adaptive-oracle feature capture cannot be nested")
        self._active = True
        self._capture_complete = False
        self._hook_calls = 0
        self._consume_calls = 0
        self._conditional = None
        handle = self.feature_module.register_forward_pre_hook(
            self._capture, with_kwargs=True
        )
        try:
            yield self
        except BaseException:
            raise
        else:
            if self._hook_calls != 1 or self._conditional is None:
                raise RuntimeError(
                    "adaptive-oracle capture requires exactly one feature-block forward"
                )
            self._capture_complete = True
        finally:
            handle.remove()
            self._active = False

    def conditional_feature(self) -> Tensor:
        if not self._capture_complete or self._conditional is None:
            raise RuntimeError("no complete adaptive-oracle feature capture is available")
        self._consume_calls += 1
        if self._consume_calls != 1:
            raise RuntimeError("adaptive-oracle feature may be consumed only once")
        return self._conditional

    def to_record(self) -> dict:
        feature = self._conditional
        return {
            "feature_block": self.feature_block,
            "expected_cfg_shape": [
                2 * self.batch_size,
                self.expected_channels,
                *self.expected_size,
            ],
            "captured_shape": None if feature is None else list(feature.shape),
            "hook_calls": int(self._hook_calls),
            "consume_calls": int(self._consume_calls),
            "capture_complete": bool(self._capture_complete),
            "conditional_rows": "second_half",
            "detached": None if feature is None else not feature.requires_grad,
        }


class AdaptiveOracleBasisProvider:
    """Adapt one registered local-relation orbit to ``RendererBasisProvider``.

    Feature affinity is the only mode that owns a U-Net hook.  The other
    controls depend solely on the scheduler observation and therefore keep the
    ordinary forward completely uninstrumented.
    """

    def __init__(
        self,
        unet: Optional[nn.Module] = None,
        *,
        batch_size: int,
        affinity_source: str,
        orbit_name: str,
        random_context: Optional[AdaptiveOracleRandomContext] = None,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or int(batch_size) != 1
        ):
            raise ValueError(
                "registered adaptive-oracle provider requires batch_size=1"
            )
        if affinity_source not in LOCAL_RELATIONAL_AFFINITY_SOURCES:
            raise ValueError(
                f"affinity_source must be one of {LOCAL_RELATIONAL_AFFINITY_SOURCES}"
            )
        if orbit_name not in LOCAL_RELATIONAL_ORBIT_NAMES:
            raise ValueError(
                f"orbit_name must be one of {LOCAL_RELATIONAL_ORBIT_NAMES}"
            )
        if affinity_source == "random_edge":
            if not isinstance(random_context, AdaptiveOracleRandomContext):
                raise ValueError(
                    "random_edge affinity requires an immutable random_context"
                )
        elif random_context is not None:
            raise ValueError(
                f"{affinity_source} affinity does not accept a random_context"
            )
        if affinity_source == "feature" and unet is None:
            raise ValueError("feature affinity requires a U-Net")

        self.batch_size = 1
        self.affinity_source = affinity_source
        self.orbit_name = orbit_name
        self.orbit_index = LOCAL_RELATIONAL_ORBIT_NAMES.index(orbit_name)
        self._random_context = random_context
        self.local_basis_provider = LocalRelationalBasisProvider()
        self.capture: Optional[AdaptiveOracleFeatureCapture] = None
        if affinity_source == "feature":
            self.capture = AdaptiveOracleFeatureCapture(unet, batch_size=1)
        self.last_diagnostics: Optional[dict] = None

    @property
    def random_context(self) -> Optional[AdaptiveOracleRandomContext]:
        """Return the provider's immutable random counter identity, if any."""
        return self._random_context

    @contextmanager
    def capture_forward(self) -> Iterator["AdaptiveOracleBasisProvider"]:
        """Instrument exactly the one ordinary U-Net call needed by the mode."""
        self.last_diagnostics = None
        if self.capture is None:
            yield self
            return
        with self.capture.capture_forward():
            yield self

    @staticmethod
    def _basis_sha256(value: Tensor) -> str:
        byte_tensor = (
            value.detach().contiguous().cpu().view(torch.uint8).reshape(-1)
        )
        try:
            raw = byte_tensor.numpy().tobytes()
        except (AttributeError, RuntimeError, TypeError):
            raw = bytes(byte_tensor.tolist())
        return hashlib.sha256(raw).hexdigest()

    def __call__(self, observation: RendererObservation) -> RendererCondition:
        """Build and select one orbit basis from an ordinary transition."""
        self.last_diagnostics = None
        x0 = observation.pred_original_sample
        _require_nchw(x0, "observation.pred_original_sample")
        if x0.shape[0] != 1:
            raise ValueError(
                "registered adaptive-oracle observation requires batch_size=1"
            )

        random_counter_context = None
        if self.affinity_source == "feature":
            if self.capture is None:
                raise RuntimeError("feature affinity has no feature capture")
            feature = self.capture.conditional_feature()
            all_bases, local_diagnostics = self.local_basis_provider(x0, feature)
        elif self.affinity_source == "uniform_local":
            all_bases, local_diagnostics = self.local_basis_provider.uniform_local(
                x0
            )
        elif self.affinity_source == "predicted_clean":
            all_bases, local_diagnostics = self.local_basis_provider.predicted_clean(
                x0
            )
        elif self.affinity_source == "random_edge":
            if self.random_context is None:
                raise RuntimeError("random_edge affinity has no random_context")
            random_counter_context = self.random_context.to_record(
                step_index=observation.step_index,
                orbit_name=self.orbit_name,
            )
            all_bases, local_diagnostics = self.local_basis_provider.random_edge(
                x0,
                experiment_id=self.random_context.experiment_id,
                split_role=self.random_context.split_role,
                prompt_row_id=self.random_context.prompt_row_id,
                seed=self.random_context.seed,
                step_index=random_counter_context["step_index"],
            )
        else:  # Constructor validation makes this an internal-state failure.
            raise RuntimeError(f"unsupported affinity_source {self.affinity_source!r}")

        expected_shape = (
            1,
            len(LOCAL_RELATIONAL_ORBIT_NAMES),
            4,
            *x0.shape[-2:],
        )
        if tuple(all_bases.shape) != expected_shape:
            raise RuntimeError(
                "local relational provider returned an unexpected basis shape: "
                f"expected {expected_shape}, got {tuple(all_bases.shape)}"
            )
        if tuple(local_diagnostics.orbit_names) != LOCAL_RELATIONAL_ORBIT_NAMES:
            raise RuntimeError(
                "local relational diagnostics changed registered orbit order"
            )
        if local_diagnostics.affinity_source != self.affinity_source:
            raise RuntimeError("local relational diagnostics changed affinity source")

        selected = all_bases[
            :, self.orbit_index : self.orbit_index + 1
        ].contiguous()
        selected = selected.detach()
        record = {
            "implementation": ADAPTIVE_ORACLE_BASIS_PROVIDER_ID,
            "affinity_source": self.affinity_source,
            "selected_orbit": self.orbit_name,
            "selected_orbit_index": self.orbit_index,
            "selected_basis_shape": list(selected.shape),
            "selected_basis_dtype": str(selected.dtype).removeprefix("torch."),
            "selected_basis_sha256": self._basis_sha256(selected),
            "local_diagnostics": local_diagnostics.to_record(),
        }
        if self.affinity_source == "feature":
            record["capture_record"] = self.capture.to_record()
        if random_counter_context is not None:
            record["random_counter_context"] = random_counter_context
            counter_hashes = (
                local_diagnostics.random_edge_counter_set_sha256
            )
            if counter_hashes is None or len(counter_hashes) != len(
                LOCAL_RELATIONAL_ORBIT_NAMES
            ):
                raise RuntimeError("random-edge counter-set diagnostics are incomplete")
            record["random_counter_set_sha256"] = counter_hashes[
                self.orbit_index
            ]
        self.last_diagnostics = record
        return RendererCondition(bases=selected)


@dataclass(frozen=True)
class FixedRatioGeodesicDiagnostics:
    """JSON-safe mechanism diagnostics for one adaptive-oracle step."""

    raw_update_norm: Tensor
    bounded_update_norm: Tensor
    scheduler_update_norm: Tensor
    update_ratio: Tensor
    mean_error: Tensor
    variance_error: Tensor
    clean_update_gain: Tensor
    applied_update_norm: Tensor
    applied_update_ratio: Tensor
    target_update_ratio: Tensor
    target_ratio_error: Tensor
    angle: Tensor
    covariance_drift: Tensor
    tangent_norm_min: Tensor
    tangent_norm_max: Tensor
    cap_hit: Tensor
    sign: int

    def to_record(self) -> dict:
        def numbers(value: Tensor) -> list[float]:
            return value.detach().float().cpu().tolist()

        return {
            "raw_update_norm": numbers(self.raw_update_norm),
            "bounded_update_norm": numbers(self.bounded_update_norm),
            "scheduler_update_norm": numbers(self.scheduler_update_norm),
            "update_ratio": numbers(self.update_ratio),
            "mean_error": numbers(self.mean_error),
            "variance_error": numbers(self.variance_error),
            "clean_update_gain": numbers(self.clean_update_gain),
            "applied_update_norm": numbers(self.applied_update_norm),
            "applied_update_ratio": numbers(self.applied_update_ratio),
            "target_update_ratio": numbers(self.target_update_ratio),
            "target_ratio_error": numbers(self.target_ratio_error),
            "angle": numbers(self.angle),
            "covariance_drift": numbers(self.covariance_drift),
            "tangent_norm_min": numbers(self.tangent_norm_min),
            "tangent_norm_max": numbers(self.tangent_norm_max),
            "cap_hit": self.cap_hit.detach().cpu().tolist(),
            "sign": int(self.sign),
        }


@dataclass(frozen=True)
class FixedRatioGeodesicOutput:
    guided_x0: Tensor
    residual: Tensor
    coefficients: Tensor
    diagnostics: FixedRatioGeodesicDiagnostics


@dataclass(frozen=True)
class FixedRatioMappedGeodesicOutput:
    """Quantization-aware Euler result produced before the one scheduler call."""

    rendered: FixedRatioGeodesicOutput
    model_output: Tensor
    predicted_prev_sample: Tensor
    mapped_intervention: EulerMappedIntervention
    solver_target_update_ratio: float
    solver_evaluations: int


class FixedRatioMomentGeodesicRenderer(nn.Module):
    """Map one basis to an antithetic, moment-preserving Euler endpoint.

    A single common angle is solved analytically for each sample so the mapped
    scheduler displacement has the registered norm ratio.  The basis controls
    direction only; its magnitude cannot change the action strength.
    """

    requires_strict_scheduler_round_trip = True
    requires_strict_scheduler_mapped_ratio = True
    scheduler_pred_original_relative_l2_tolerance = 0.01
    scheduler_prev_sample_relative_l2_tolerance = 1e-3
    native_endpoint_input_relative_l2_tolerance = 1e-3
    native_gain_input_relative_tolerance = 1e-3

    def __init__(
        self,
        *,
        sign: int,
        target_update_ratio: float = ADAPTIVE_ORACLE_TARGET_UPDATE_RATIO,
        hard_update_cap: float = ADAPTIVE_ORACLE_HARD_UPDATE_CAP,
        epsilon: float = 1e-6,
        ratio_tolerance: float = 5e-4,
        mean_tolerance: float = 1e-4,
        relative_variance_tolerance: float = 1e-3,
        covariance_tolerance: float = 0.01,
    ) -> None:
        super().__init__()
        if isinstance(sign, bool) or int(sign) not in {-1, 1}:
            raise ValueError("sign must be exactly -1 or +1")
        values = {
            "target_update_ratio": target_update_ratio,
            "hard_update_cap": hard_update_cap,
            "epsilon": epsilon,
            "ratio_tolerance": ratio_tolerance,
            "mean_tolerance": mean_tolerance,
            "relative_variance_tolerance": relative_variance_tolerance,
            "covariance_tolerance": covariance_tolerance,
        }
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("adaptive-oracle renderer thresholds must be finite")
        if not 0 < float(target_update_ratio) < float(hard_update_cap) < 1:
            raise ValueError("target ratio and hard cap must satisfy 0 < target < cap < 1")
        if any(float(values[name]) <= 0 for name in values if name not in {"target_update_ratio", "hard_update_cap"}):
            raise ValueError("adaptive-oracle renderer tolerances must be positive")
        self.sign = int(sign)
        self.target_update_ratio = float(target_update_ratio)
        self.hard_update_cap = float(hard_update_cap)
        self.epsilon = float(epsilon)
        self.ratio_tolerance = float(ratio_tolerance)
        self.mean_tolerance = float(mean_tolerance)
        self.relative_variance_tolerance = float(relative_variance_tolerance)
        self.covariance_tolerance = float(covariance_tolerance)

    @property
    def parameter_count(self) -> int:
        return 0

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
        solver_target_update_ratio: Optional[float] = None,
    ) -> FixedRatioGeodesicOutput:
        del timestep
        _require_nchw(latent, "latent")
        if latent.shape[1] != 4:
            raise ValueError("adaptive-oracle latent must have exactly four channels")
        if (
            not isinstance(bases, Tensor)
            or bases.ndim != 5
            or bases.shape[:2] != (latent.shape[0], 1)
            or bases.shape[2:] != latent.shape[1:]
        ):
            raise ValueError(
                "adaptive-oracle bases must have shape (batch, 1, 4, height, width)"
            )
        if bases.device != latent.device or not torch.isfinite(bases.detach()).all():
            raise ValueError("adaptive-oracle basis must be finite and share latent device")
        for value, name in (
            (prompt_embedding, "prompt_embedding"),
            (state_features, "state_features"),
        ):
            if value is not None and value.numel() != 0:
                raise ValueError(f"fixed adaptive-oracle action cannot read {name}")
        if scheduler_update is None or clean_update_gain is None:
            raise ValueError(
                "scheduler_update and clean_update_gain are required for native Euler mapping"
            )
        _require_nchw(scheduler_update, "scheduler_update")
        if scheduler_update.shape != latent.shape:
            raise ValueError("scheduler_update must match latent shape")

        batch = latent.shape[0]
        gain = _batch_vector(
            clean_update_gain,
            batch=batch,
            device=latent.device,
            name="clean_update_gain",
        )
        if torch.any(gain <= 0) or torch.any(gain > 1):
            raise ValueError("clean_update_gain must satisfy 0 < gain <= 1")
        scheduler_norm = _vector_norm(scheduler_update)
        if torch.any(scheduler_norm <= self.epsilon):
            raise RuntimeError("scheduler update norm is below the registered threshold")
        scheduler_denominator = scheduler_norm + 1e-12

        work = latent.float()
        raw = bases[:, 0].float()
        mean = work.mean(dim=(-2, -1), keepdim=True)
        centered = work - mean
        centered_basis = raw - raw.mean(dim=(-2, -1), keepdim=True)
        centered_flat = centered.flatten(2)
        basis_flat = centered_basis.flatten(2)
        channel_gram = torch.bmm(centered_flat, centered_flat.transpose(1, 2))
        channel_energy = channel_gram.diagonal(dim1=1, dim2=2)
        if torch.any(channel_energy <= self.epsilon):
            raise RuntimeError("latent channel energy is below the registered threshold")
        channel_cholesky, channel_info = torch.linalg.cholesky_ex(channel_gram)
        if torch.any(channel_info != 0) or not torch.isfinite(channel_cholesky).all():
            raise RuntimeError("latent channel Gram matrix is singular or non-finite")

        basis_channel_inner = torch.bmm(
            basis_flat, centered_flat.transpose(1, 2)
        )
        projection_coefficients = torch.linalg.solve(
            channel_gram, basis_channel_inner.transpose(1, 2)
        ).transpose(1, 2)
        tangent_flat = basis_flat - torch.bmm(
            projection_coefficients, centered_flat
        )
        tangent_gram = torch.bmm(tangent_flat, tangent_flat.transpose(1, 2))
        tangent_cholesky, tangent_info = torch.linalg.cholesky_ex(tangent_gram)
        tangent_norm = torch.linalg.vector_norm(tangent_flat, dim=2)
        if not torch.isfinite(tangent_norm).all() or torch.any(
            tangent_norm <= self.epsilon
        ) or torch.any(tangent_info != 0) or not torch.isfinite(
            tangent_cholesky
        ).all():
            raise RuntimeError("adaptive-oracle tangent norm is zero or non-finite")

        whitened_tangent = torch.linalg.solve_triangular(
            tangent_cholesky, tangent_flat, upper=False
        )
        direction_flat = torch.bmm(channel_cholesky, whitened_tangent)
        direction = direction_flat.reshape_as(centered)

        centered_norm = _vector_norm(centered)
        if solver_target_update_ratio is None:
            solver_target = self.target_update_ratio
        else:
            if (
                isinstance(solver_target_update_ratio, bool)
                or not isinstance(solver_target_update_ratio, Real)
            ):
                raise ValueError("solver_target_update_ratio must be a real number")
            solver_target = float(solver_target_update_ratio)
            if not math.isfinite(solver_target) or not (
                0.0 <= solver_target < self.hard_update_cap
            ):
                raise ValueError(
                    "solver_target_update_ratio must satisfy 0 <= target < hard cap"
                )
        desired_applied_norm = solver_target * scheduler_denominator
        sine_half = desired_applied_norm / (
            2.0 * gain * centered_norm.clamp_min(self.epsilon)
        )
        if not torch.isfinite(sine_half).all() or torch.any(sine_half >= 1.0):
            raise RuntimeError("registered target ratio is outside geodesic reach")
        angle = 2.0 * torch.asin(sine_half)
        angle_bc = angle.reshape(batch, 1, 1, 1)
        moved = (
            mean
            + centered * torch.cos(angle_bc)
            + float(self.sign) * direction * torch.sin(angle_bc)
        )
        guided = moved.to(dtype=latent.dtype)
        if not torch.isfinite(guided.detach()).all():
            raise RuntimeError("adaptive-oracle endpoint is non-finite after dtype conversion")

        residual = guided.float() - work
        bounded_norm = _vector_norm(residual)
        applied_norm = _vector_norm(residual * gain.reshape(batch, 1, 1, 1))
        applied_ratio = applied_norm / scheduler_denominator
        target = applied_ratio.new_full((batch,), solver_target)
        target_error = (applied_ratio - target).abs()
        cap_hit = applied_ratio >= self.hard_update_cap

        guided_mean = guided.float().mean(dim=(-2, -1))
        latent_mean = work.mean(dim=(-2, -1))
        mean_error = (guided_mean - latent_mean).abs().amax(dim=1)
        latent_variance = work.var(dim=(-2, -1), correction=0)
        guided_variance = guided.float().var(dim=(-2, -1), correction=0)
        variance_error = (
            (guided_variance - latent_variance).abs()
            / latent_variance.abs().clamp_min(self.epsilon)
        ).amax(dim=1)
        latent_covariance = _channel_covariance(work)
        guided_covariance = _channel_covariance(guided)
        covariance_drift = torch.linalg.matrix_norm(
            guided_covariance - latent_covariance, ord="fro"
        ) / torch.linalg.matrix_norm(
            latent_covariance, ord="fro"
        ).clamp_min(self.epsilon)

        if torch.any(target_error > self.ratio_tolerance):
            raise RuntimeError("adaptive-oracle mapped ratio missed its registered target")
        if torch.any(cap_hit):
            raise RuntimeError("adaptive-oracle mapped ratio hit the hard cap")
        if torch.any(mean_error > self.mean_tolerance):
            raise RuntimeError("adaptive-oracle channel mean drift exceeds tolerance")
        if torch.any(variance_error > self.relative_variance_tolerance):
            raise RuntimeError("adaptive-oracle channel variance drift exceeds tolerance")
        if torch.any(covariance_drift > self.covariance_tolerance):
            raise RuntimeError("adaptive-oracle channel covariance drift exceeds tolerance")

        diagnostics = FixedRatioGeodesicDiagnostics(
            raw_update_norm=_vector_norm(raw),
            bounded_update_norm=bounded_norm,
            scheduler_update_norm=scheduler_norm,
            update_ratio=bounded_norm / scheduler_denominator,
            mean_error=mean_error,
            variance_error=variance_error,
            clean_update_gain=gain,
            applied_update_norm=applied_norm,
            applied_update_ratio=applied_ratio,
            target_update_ratio=target,
            target_ratio_error=target_error,
            angle=angle,
            covariance_drift=covariance_drift,
            tangent_norm_min=tangent_norm.amin(dim=1),
            tangent_norm_max=tangent_norm.amax(dim=1),
            cap_hit=cap_hit,
            sign=self.sign,
        )
        return FixedRatioGeodesicOutput(
            guided_x0=guided,
            residual=(guided.float() - work).to(dtype=latent.dtype),
            coefficients=guided.new_full((batch, 1), float(self.sign)),
            diagnostics=diagnostics,
        )

    def forward_euler_mapped(
        self,
        latent: Tensor,
        bases: Tensor,
        *,
        sample: Tensor,
        native_model_output: Tensor,
        sigma_from: Tensor | float,
        sigma_to: Tensor | float,
        prediction_type: str,
        scheduler_update: Tensor,
        clean_update_gain: Tensor | float,
        timestep: Optional[Tensor] = None,
        prompt_embedding: Optional[Tensor] = None,
        state_features: Optional[Tensor] = None,
    ) -> FixedRatioMappedGeodesicOutput:
        """Solve the registered ratio after fp16 Euler output quantization."""

        _require_nchw(latent, "latent")
        if latent.shape[0] != 1:
            raise ValueError("mapped adaptive-oracle solver requires batch_size=1")
        native_endpoint = prepare_euler_clean_endpoint(
            sample,
            native_model_output,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            prediction_type=prediction_type,
        )
        for supplied, reconstructed, name in (
            (latent, native_endpoint.pred_original_sample, "latent"),
            (scheduler_update, native_endpoint.nominal_update, "scheduler_update"),
        ):
            _require_nchw(supplied, name)
            if supplied.shape != reconstructed.shape or supplied.device != reconstructed.device:
                raise ValueError(
                    f"{name} must share shape and device with the reconstructed native endpoint"
                )
            relative_error = _vector_norm(supplied.float() - reconstructed.float()) / (
                _vector_norm(reconstructed) + 1e-12
            )
            if torch.any(
                relative_error > self.native_endpoint_input_relative_l2_tolerance
            ):
                raise ValueError(f"{name} differs from the reconstructed native endpoint")
        supplied_gain = _batch_vector(
            clean_update_gain,
            batch=latent.shape[0],
            device=latent.device,
            name="clean_update_gain",
        )
        reconstructed_gain = native_endpoint.clean_update_gain.to(
            device=latent.device, dtype=torch.float32
        )
        gain_relative_error = (supplied_gain - reconstructed_gain).abs() / (
            reconstructed_gain.abs() + 1e-12
        )
        if torch.any(
            gain_relative_error > self.native_gain_input_relative_tolerance
        ):
            raise ValueError(
                "clean_update_gain differs from the reconstructed native endpoint"
            )
        native_prev = predict_euler_no_churn_prev_sample(
            sample,
            native_model_output,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            prediction_type=prediction_type,
        )
        evaluations = 0

        def evaluate(solver_target: float):
            nonlocal evaluations
            rendered = self.forward(
                native_endpoint.pred_original_sample,
                bases,
                timestep=timestep,
                prompt_embedding=prompt_embedding,
                state_features=state_features,
                scheduler_update=native_endpoint.nominal_update,
                clean_update_gain=native_endpoint.clean_update_gain,
                solver_target_update_ratio=solver_target,
            )
            model_output = euler_model_output_from_clean_sample(
                sample,
                rendered.guided_x0,
                sigma_from=sigma_from,
                prediction_type=prediction_type,
                output_dtype=native_model_output.dtype,
            )
            predicted_prev = predict_euler_no_churn_prev_sample(
                sample,
                model_output,
                sigma_from=sigma_from,
                sigma_to=sigma_to,
                prediction_type=prediction_type,
            )
            measured = measure_euler_mapped_intervention(
                sample, native_prev, predicted_prev
            )
            evaluations += 1
            return rendered, model_output, predicted_prev, measured

        target = self.target_update_ratio
        initial = evaluate(target)
        initial_ratio = float(initial[3].ratio[0].detach().cpu().item())
        best = (abs(initial_ratio - target), target, initial)

        if best[0] > self.ratio_tolerance:
            if initial_ratio > target:
                lower_target = 0.0
                lower = evaluate(lower_target)
                upper_target = target
                upper = initial
            else:
                lower_target = target
                lower = initial
                upper_target = min(
                    target * 2.0,
                    self.hard_update_cap - max(self.ratio_tolerance, 1e-6),
                )
                upper = evaluate(upper_target)
            lower_ratio = float(lower[3].ratio[0].detach().cpu().item())
            upper_ratio = float(upper[3].ratio[0].detach().cpu().item())
            for solver_target, candidate, ratio in (
                (lower_target, lower, lower_ratio),
                (upper_target, upper, upper_ratio),
            ):
                candidate_key = (abs(ratio - target), solver_target, candidate)
                if candidate_key[:2] < best[:2]:
                    best = candidate_key
            if not lower_ratio <= target <= upper_ratio:
                raise RuntimeError(
                    "scheduler-mapped ratio solver failed to bracket the target"
                )

            for _ in range(12):
                midpoint = 0.5 * (lower_target + upper_target)
                candidate = evaluate(midpoint)
                ratio = float(candidate[3].ratio[0].detach().cpu().item())
                candidate_key = (abs(ratio - target), midpoint, candidate)
                if candidate_key[:2] < best[:2]:
                    best = candidate_key
                if ratio < target:
                    lower_target = midpoint
                else:
                    upper_target = midpoint

        error, solver_target, selected = best
        rendered, model_output, predicted_prev, measured = selected
        if error > self.ratio_tolerance:
            raise RuntimeError("scheduler-mapped ratio solver missed the target")
        if torch.any(measured.ratio >= self.hard_update_cap):
            raise RuntimeError("scheduler-mapped ratio solver hit the hard cap")
        return FixedRatioMappedGeodesicOutput(
            rendered=rendered,
            model_output=model_output,
            predicted_prev_sample=predicted_prev,
            mapped_intervention=measured,
            solver_target_update_ratio=float(solver_target),
            solver_evaluations=evaluations,
        )


__all__ = [
    "ADAPTIVE_ORACLE_BASIS_PROVIDER_ID",
    "ADAPTIVE_ORACLE_FEATURE_BLOCK",
    "ADAPTIVE_ORACLE_FEATURE_CHANNELS",
    "ADAPTIVE_ORACLE_FEATURE_SIZE",
    "ADAPTIVE_ORACLE_HARD_UPDATE_CAP",
    "ADAPTIVE_ORACLE_TARGET_UPDATE_RATIO",
    "AdaptiveOracleBasisProvider",
    "AdaptiveOracleFeatureCapture",
    "AdaptiveOracleRandomContext",
    "FixedRatioGeodesicDiagnostics",
    "FixedRatioGeodesicOutput",
    "FixedRatioMappedGeodesicOutput",
    "FixedRatioMomentGeodesicRenderer",
]
