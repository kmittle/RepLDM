"""Scheduler-native latent renderer and its auditable action frame.

The production SDXL pipeline supplies a frozen denoising observation and a
set of detached structural residuals.  This module turns those residuals into
one fixed, low-dimensional action space shared by OPD, DPO and RL.  It does
not import diffusers and can therefore be tested with small CPU tensors.

The important distinction from the historical renderer is that bases are
measured in the scheduler's actual Euler update coordinates before coefficients
are predicted.  A global rank mask is frozen once during calibration; it is
never recomputed per state or per method.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from AttentionGuidance.latent_renderer import (
    EulerMappedIntervention,
    RendererCondition,
    RendererObservation,
    _basis_statistics,
    _coerce_batch_vector,
    _coerce_context,
    _fixed_moment_geodesic,
    _sinusoidal_embedding,
    _spectral_summary,
    euler_model_output_from_clean_sample,
    predict_euler_no_churn_prev_sample,
)

from .contracts import ActionSpaceContract, contract_hash


FRAME_SCHEMA = "repldm.euler_native_frame.v1"
CALIBRATION_SCHEMA = FRAME_SCHEMA + ".mask.v2"
CALIBRATION_STATE_COUNT = 576
CALIBRATION_SLOTS = 6
CALIBRATION_NORM_THRESHOLD = 1e-8
CALIBRATION_RESIDUAL_THRESHOLD = 1e-6
CALIBRATION_EPSILON = 1e-12
CALIBRATION_DECISION_INDICES = (8, 24, 40)
CANONICAL_FRAME_SLOTS = (
    "semantic",
    "spectral_low",
    "spectral_mid",
    "spectral_high",
    "freeu",
    "laplacian",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def tensor_sha256(value: Tensor) -> str:
    """Hash a tensor after a deterministic CPU, contiguous conversion."""
    flat = value.detach().cpu().reshape(-1)
    # ``Tensor.contiguous()`` and ``clone()`` preserve a zero stride for an
    # expanded one-element tensor.  A fresh one-dimensional allocation is
    # required before viewing floats as bytes.
    packed = torch.empty(flat.numel(), dtype=flat.dtype, device="cpu")
    packed.copy_(flat)
    data = packed.view(torch.uint8).reshape(-1)
    try:
        payload = data.numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError):
        payload = bytes(data.tolist())
    return hashlib.sha256(payload).hexdigest()


def mask_sha256(mask: Sequence[bool]) -> str:
    try:
        values = tuple(mask)
    except TypeError as exc:
        raise ValueError("active mask must be a sequence of booleans") from exc
    if not values or any(not isinstance(item, bool) for item in values):
        raise ValueError("active mask must contain non-empty boolean values")
    return hashlib.sha256(_canonical(list(values))).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest") from exc
    if value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _nchw(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, channels, height, width)")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _raw_basis_shape(value: Tensor, reference: Tensor, slots: Optional[int] = None) -> None:
    _nchw(reference, "clean_latent")
    if not isinstance(value, Tensor) or value.ndim != 5:
        raise ValueError(
            "raw_bases must have shape (batch, slots, channels, height, width)"
        )
    if value.shape[0] != reference.shape[0] or value.shape[2:] != reference.shape[1:]:
        raise ValueError("raw_bases do not match clean_latent")
    if not value.is_floating_point():
        raise ValueError("raw_bases must use a floating-point dtype")
    if slots is not None and value.shape[1] != int(slots):
        raise ValueError(f"raw_bases must contain exactly {int(slots)} slots")
    if value.device != reference.device:
        raise ValueError("raw_bases and clean_latent must share one device")
    if not torch.isfinite(value).all():
        raise ValueError("raw_bases contain non-finite values")


def _sigma_pair(
    sigma_from: Tensor | float,
    sigma_to: Tensor | float,
    batch: int,
    reference: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    source = torch.as_tensor(sigma_from, device=reference.device, dtype=torch.float32)
    target = torch.as_tensor(sigma_to, device=reference.device, dtype=torch.float32)
    if source.ndim == 0:
        source = source.expand(batch)
    if target.ndim == 0:
        target = target.expand(batch)
    if source.ndim != 1 or target.ndim != 1 or source.numel() not in (1, batch) or target.numel() not in (1, batch):
        raise ValueError("Euler sigmas must be scalar or have one value per batch item")
    if source.numel() == 1:
        source = source.expand(batch)
    if target.numel() == 1:
        target = target.expand(batch)
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("Euler sigmas must be finite")
    if torch.any(source <= 0) or torch.any(target < 0) or torch.any(target > source):
        raise ValueError("Euler sigmas must satisfy 0 <= sigma_to <= sigma_from and sigma_from > 0")
    kappa = 1.0 - target / source
    return source, target, kappa


def _batch_euler_endpoint(
    sample: Tensor,
    model_output: Tensor,
    sigma_from: Tensor | float,
    sigma_to: Tensor | float,
    prediction_type: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Construct an Euler endpoint for scalar or per-row sigma values.

    The historical helpers in ``AttentionGuidance`` intentionally retain a
    scalar-only public contract.  The training renderer, however, receives a
    batch of states and must support a distinct scheduler sigma for each row.
    Keep this adapter local so the legacy pipeline's numerical path remains
    unchanged for scalar calls.
    """
    _nchw(sample, "sample")
    _nchw(model_output, "model_output")
    if sample.shape != model_output.shape or sample.device != model_output.device:
        raise ValueError("sample and model_output must share shape and device")
    source, target, _ = _sigma_pair(
        sigma_from, sigma_to, sample.shape[0], sample
    )
    source4 = source.reshape(-1, 1, 1, 1)
    target4 = target.reshape(-1, 1, 1, 1)
    work = sample.float()
    output = model_output.float()
    if prediction_type == "epsilon":
        clean = work - source4 * output
    elif prediction_type in {"sample", "original_sample"}:
        clean = output
    elif prediction_type == "v_prediction":
        clean = (
            output * (-source4 / torch.sqrt(source4.square() + 1.0))
            + work / (source4.square() + 1.0)
        )
    else:
        raise ValueError(f"unsupported Euler prediction_type {prediction_type!r}")
    derivative = (work - clean) / source4
    nominal = derivative * (target4 - source4)
    gain = (source - target) / source
    return clean, nominal, gain, source, target


def _batch_euler_model_output(
    sample: Tensor,
    clean: Tensor,
    sigma_from: Tensor | float,
    prediction_type: str,
    output_dtype: torch.dtype,
) -> Tensor:
    """Convert a clean endpoint back to Euler model-output coordinates."""
    _nchw(sample, "sample")
    _nchw(clean, "guided_x0")
    if sample.shape != clean.shape or sample.device != clean.device:
        raise ValueError("sample and guided_x0 must share shape and device")
    source = torch.as_tensor(sigma_from, device=sample.device, dtype=torch.float32)
    if source.ndim == 0 or source.numel() == 1:
        source = source.reshape(1).expand(sample.shape[0])
    elif source.ndim == 1 and source.numel() == sample.shape[0]:
        source = source.reshape(-1)
    else:
        raise ValueError("sigma_from must be scalar or have one value per batch item")
    if not torch.isfinite(source).all() or torch.any(source <= 0):
        raise ValueError("sigma_from must contain finite positive values")
    source4 = source.reshape(-1, 1, 1, 1)
    work = sample.float()
    endpoint = clean.float()
    if prediction_type == "epsilon":
        converted = (work - endpoint) / source4
    elif prediction_type in {"sample", "original_sample"}:
        converted = endpoint
    elif prediction_type == "v_prediction":
        converted = (
            work - (1.0 + source4.square()) * endpoint
        ) / (source4 * torch.sqrt(1.0 + source4.square()))
    else:
        raise ValueError(f"unsupported Euler prediction_type {prediction_type!r}")
    return converted.to(output_dtype)


def _batch_euler_prev_sample(
    sample: Tensor,
    model_output: Tensor,
    sigma_from: Tensor | float,
    sigma_to: Tensor | float,
    prediction_type: str,
) -> Tensor:
    """Mirror one Euler step when sigma values vary across batch rows."""
    clean, nominal, _gain, _source, _target = _batch_euler_endpoint(
        sample, model_output, sigma_from, sigma_to, prediction_type
    )
    # ``nominal`` is already the exact per-row Euler displacement in float32.
    return (sample.float() + nominal).to(model_output.dtype)


def project_fixed_moment_tangent(latent: Tensor, residual: Tensor, epsilon: float = 1e-12) -> Tensor:
    """Project a residual onto the per-channel fixed-mean/norm tangent."""
    _nchw(latent, "latent")
    _nchw(residual, "residual")
    if latent.shape != residual.shape:
        raise ValueError("latent and residual must have identical shapes")
    if latent.device != residual.device:
        raise ValueError("latent and residual must share one device")
    if epsilon <= 0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and positive")
    x = latent.float()
    v = residual.float()
    xc = x - x.mean(dim=(-2, -1), keepdim=True)
    vc = v - v.mean(dim=(-2, -1), keepdim=True)
    energy = xc.square().sum(dim=(-2, -1), keepdim=True)
    inner = (vc * xc).sum(dim=(-2, -1), keepdim=True)
    tangent = vc - inner / energy.clamp_min(float(epsilon)) * xc
    return torch.where(energy > float(epsilon), tangent, torch.zeros_like(tangent))


def _project_many(latent: Tensor, bases: Tensor, epsilon: float) -> Tensor:
    """Apply the tangent projection independently to every basis slot."""
    batch, slots = bases.shape[:2]
    flat = bases.reshape(batch * slots, *bases.shape[2:])
    repeated = latent[:, None].expand(batch, slots, *latent.shape[1:]).reshape_as(flat)
    return project_fixed_moment_tangent(repeated, flat, epsilon).reshape_as(bases)


@dataclass(frozen=True)
class FrameCalibrationSample:
    """One or more states from the pre-registered calibration table.

    The tensor batch may contain several rows, but every row must have an
    explicit immutable state id and state hash.  The hashes bind the prompt,
    seed, decision index, scheduler snapshot, and basis-provider provenance;
    the renderer never invents those identifiers from tensor bytes alone.
    """

    clean_latent: Tensor
    raw_bases: Tensor
    native_update: Tensor
    sigma_from: Tensor | float
    sigma_to: Tensor | float
    manifest_sha256: Optional[str] = None
    source_sha256: Optional[str] = None
    state_ids: Optional[Sequence[str]] = None
    state_hashes: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        if self.manifest_sha256 is not None:
            _require_sha256(self.manifest_sha256, "calibration manifest_sha256")
        if self.source_sha256 is not None:
            _require_sha256(self.source_sha256, "calibration source_sha256")
        for name in ("state_ids", "state_hashes"):
            value = getattr(self, name)
            if value is not None:
                try:
                    value = tuple(value)
                except TypeError as exc:
                    raise ValueError(f"{name} must be a sequence") from exc
                object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FrameCalibration:
    """The immutable result of global rank-mask calibration."""

    active_mask: Tuple[bool, ...]
    state_count: int
    minimum_norm_squared: Tuple[float, ...]
    minimum_residual_ratio: Tuple[float, ...]
    manifest_sha256: str = ""
    source_sha256: str = ""
    state_provenance_sha256: str = ""
    schema: str = CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        try:
            mask = tuple(self.active_mask)
        except TypeError as exc:
            raise ValueError("calibration active_mask must be a sequence") from exc
        if len(mask) != CALIBRATION_SLOTS or any(
            not isinstance(value, bool) for value in mask
        ):
            raise ValueError("calibration active_mask must contain exactly six booleans")
        object.__setattr__(self, "active_mask", mask)
        try:
            minimum_norm_squared = tuple(self.minimum_norm_squared)
            minimum_residual_ratio = tuple(self.minimum_residual_ratio)
        except TypeError as exc:
            raise ValueError("calibration arrays must be sequences") from exc
        if len(mask) != len(minimum_norm_squared):
            raise ValueError("calibration arrays must have one entry per slot")
        if len(mask) != len(minimum_residual_ratio):
            raise ValueError("calibration arrays must have one entry per slot")
        if (
            isinstance(self.state_count, bool)
            or not isinstance(self.state_count, int)
            or self.state_count != CALIBRATION_STATE_COUNT
        ):
            raise ValueError(
                f"state_count must equal the registered {CALIBRATION_STATE_COUNT} states"
            )
        for name, values in (
            ("minimum_norm_squared", minimum_norm_squared),
            ("minimum_residual_ratio", minimum_residual_ratio),
        ):
            try:
                values = tuple(values)
            except TypeError as exc:
                raise ValueError(f"{name} must be a sequence") from exc
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in values
            ):
                raise ValueError(f"{name} must contain finite non-negative numbers")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if self.schema != CALIBRATION_SCHEMA:
            raise ValueError("calibration schema is not the registered canonical schema")
        _require_sha256(self.manifest_sha256, "calibration manifest_sha256")
        _require_sha256(self.source_sha256, "calibration source_sha256")
        _require_sha256(
            self.state_provenance_sha256, "calibration state_provenance_sha256"
        )
        if sum(mask) < 2:
            raise ValueError("global frame rank must be at least two")

    @property
    def rank(self) -> int:
        return sum(self.active_mask)

    @property
    def mask_hash(self) -> str:
        return mask_sha256(self.active_mask)

    @property
    def calibration_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "active_mask": list(self.active_mask),
            "rank": self.rank,
            "state_count": self.state_count,
            "minimum_norm_squared": list(self.minimum_norm_squared),
            "minimum_residual_ratio": list(self.minimum_residual_ratio),
            "manifest_sha256": self.manifest_sha256,
            "source_sha256": self.source_sha256,
            "state_provenance_sha256": self.state_provenance_sha256,
            "decision_indices": list(CALIBRATION_DECISION_INDICES),
            }
        return hashlib.sha256(_canonical(payload)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "active_mask": list(self.active_mask),
            "rank": self.rank,
            "state_count": self.state_count,
            "minimum_norm_squared": list(self.minimum_norm_squared),
            "minimum_residual_ratio": list(self.minimum_residual_ratio),
            "manifest_sha256": self.manifest_sha256,
            "source_sha256": self.source_sha256,
            "state_provenance_sha256": self.state_provenance_sha256,
            "decision_indices": list(CALIBRATION_DECISION_INDICES),
            "mask_hash": self.mask_hash,
            "calibration_hash": self.calibration_hash,
        }


def _sample_parts(sample: FrameCalibrationSample, slots: int, epsilon: float) -> Tensor:
    _raw_basis_shape(sample.raw_bases, sample.clean_latent)
    if sample.raw_bases.shape[1] != slots:
        raise ValueError("calibration sample has the wrong number of slots")
    _nchw(sample.native_update, "native_update")
    if sample.native_update.shape != sample.clean_latent.shape:
        raise ValueError("native_update must match clean_latent")
    if sample.native_update.device != sample.clean_latent.device:
        raise ValueError("native_update and clean_latent must share one device")
    _, _, kappa = _sigma_pair(
        sample.sigma_from, sample.sigma_to, sample.clean_latent.shape[0], sample.clean_latent
    )
    if torch.any(kappa < 1e-6):
        raise ValueError("calibration contains an invalid Euler kappa")
    norm = torch.linalg.vector_norm(sample.native_update.float().flatten(1), dim=1)
    if torch.any(norm < 1e-6):
        raise ValueError("calibration contains a near-zero native update")
    q = _project_many(sample.clean_latent, sample.raw_bases, epsilon)
    d = q * kappa.reshape(-1, 1, 1, 1, 1)
    return d.float().flatten(2) / norm.reshape(-1, 1, 1)


def _sample_provenance(
    sample: FrameCalibrationSample,
    *,
    batch: int,
) -> tuple[str, str, tuple[dict[str, str], ...]]:
    """Validate and expand the immutable provenance attached to one batch."""
    if sample.manifest_sha256 is None or sample.source_sha256 is None:
        raise ValueError(
            "calibration samples require manifest_sha256 and source_sha256"
        )
    manifest = _require_sha256(sample.manifest_sha256, "calibration manifest_sha256")
    source = _require_sha256(sample.source_sha256, "calibration source_sha256")
    if sample.state_ids is None or sample.state_hashes is None:
        raise ValueError("calibration samples require state_ids and state_hashes")
    state_ids = tuple(sample.state_ids)
    state_hashes = tuple(sample.state_hashes)
    if len(state_ids) != batch or len(state_hashes) != batch:
        raise ValueError(
            "calibration state_ids and state_hashes must have one entry per row"
        )
    if any(not isinstance(value, str) or not value for value in state_ids):
        raise ValueError("calibration state_ids must be non-empty strings")
    for value in state_hashes:
        _require_sha256(value, "calibration state_hash")
    records: list[dict[str, str]] = []
    for row, (state_id, state_hash) in enumerate(zip(state_ids, state_hashes)):
        records.append(
            {
                "state_id": state_id,
                "state_sha256": state_hash,
                "clean_latent_sha256": tensor_sha256(sample.clean_latent[row]),
                "raw_bases_sha256": tensor_sha256(sample.raw_bases[row]),
                "native_update_sha256": tensor_sha256(sample.native_update[row]),
                "row": str(row),
            }
        )
    return manifest, source, tuple(records)


def calibrate_global_mask(
    samples: Iterable[FrameCalibrationSample],
    *,
    slots: int = 6,
    epsilon: float = 1e-12,
    norm_threshold: float = 1e-8,
    residual_threshold: float = 1e-6,
) -> FrameCalibration:
    """Freeze one rank mask using deterministic two-pass modified Gram-Schmidt.

    A slot is active only if it has sufficient norm and residual energy in
    every calibration state.  The caller is responsible for supplying only
    the pre-registered calibration split; validation rows must not be used.
    """
    if slots != CALIBRATION_SLOTS:
        raise ValueError(
            f"calibration must use exactly the registered {CALIBRATION_SLOTS} slots"
        )
    if float(epsilon) != CALIBRATION_EPSILON:
        raise ValueError("calibration epsilon differs from the registered value")
    if float(norm_threshold) != CALIBRATION_NORM_THRESHOLD:
        raise ValueError("calibration norm threshold differs from the registered value")
    if float(residual_threshold) != CALIBRATION_RESIDUAL_THRESHOLD:
        raise ValueError(
            "calibration residual threshold differs from the registered value"
        )
    samples = tuple(samples)
    if not samples:
        raise ValueError("at least one calibration sample is required")
    if any(not isinstance(sample, FrameCalibrationSample) for sample in samples):
        raise TypeError("calibration entries must be FrameCalibrationSample instances")
    minimum_norm = [float("inf")] * slots
    minimum_residual = [float("inf")] * slots
    active = [False] * slots
    state_vectors: list[Tensor] = []
    provenance: list[dict[str, str]] = []
    manifest_hash: Optional[str] = None
    source_hash: Optional[str] = None
    for sample in samples:
        d = _sample_parts(sample, slots, epsilon)
        current_manifest, current_source, records = _sample_provenance(
            sample, batch=d.shape[0]
        )
        if manifest_hash is None:
            manifest_hash = current_manifest
            source_hash = current_source
        elif (current_manifest, current_source) != (manifest_hash, source_hash):
            raise ValueError("calibration samples disagree on manifest/source provenance")
        # Calibration samples are allowed to have a batch dimension.  Each
        # row is a separate state in the global intersection.
        for row in range(d.shape[0]):
            state_vectors.append(d[row].double())
            provenance.append(records[row])
    if len(state_vectors) != CALIBRATION_STATE_COUNT:
        raise ValueError(
            f"calibration requires exactly {CALIBRATION_STATE_COUNT} registered states; "
            f"received {len(state_vectors)}"
        )
    state_ids = [record["state_id"] for record in provenance]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("calibration state_ids must be globally unique")
    state_provenance_hash = hashlib.sha256(_canonical(provenance)).hexdigest()
    # Decide slots in canonical order.  A slot that fails in one calibration
    # state is excluded globally, and is therefore not used to explain later
    # slots in any state.
    retained_slots: list[int] = []
    orthonormal: list[list[Tensor]] = [[] for _ in state_vectors]
    for slot in range(slots):
        slot_passes = True
        residuals_for_slot: list[Tensor] = []
        for state_index, vectors in enumerate(state_vectors):
            vector = vectors[slot]
            norm_sq = float(torch.dot(vector, vector).item())
            minimum_norm[slot] = min(minimum_norm[slot], norm_sq)
            residual = vector
            for unit in orthonormal[state_index]:
                residual = residual - torch.dot(residual, unit) * unit
            # A second pass removes the small projection error that can
            # otherwise make near-collinear columns appear independent.
            for unit in orthonormal[state_index]:
                residual = residual - torch.dot(residual, unit) * unit
            residual_sq = float(torch.dot(residual, residual).item())
            ratio = residual_sq / max(norm_sq, float(epsilon))
            minimum_residual[slot] = min(minimum_residual[slot], ratio)
            slot_passes = slot_passes and (
                norm_sq >= norm_threshold and ratio >= residual_threshold
            )
            residuals_for_slot.append(residual)
        active[slot] = bool(slot_passes)
        if slot_passes:
            retained_slots.append(slot)
            for state_index, residual in enumerate(residuals_for_slot):
                residual_norm = torch.linalg.vector_norm(residual).clamp_min(float(epsilon))
                orthonormal[state_index].append(residual / residual_norm)
    return FrameCalibration(
        tuple(bool(value) for value in active),
        len(state_vectors),
        tuple(float(value) for value in minimum_norm),
        tuple(float(value) for value in minimum_residual),
        manifest_sha256=str(manifest_hash),
        source_sha256=str(source_hash),
        state_provenance_sha256=state_provenance_hash,
    )


@dataclass(frozen=True)
class EulerFrameDiagnostics:
    valid: Tensor
    active_mask: Tuple[bool, ...]
    eigenvalues: Tensor
    condition_number: Tensor
    gram_error: Tensor
    angle: Tensor
    angle_cap_multiplier: Tensor
    scheduler_cap_multiplier: Tensor
    mapped_update_ratio: Tensor
    gram_hash: Tuple[str, ...]
    frame_hash: Tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        def values(value: Tensor) -> list[Any]:
            return value.detach().float().cpu().tolist()

        def flags(value: Tensor) -> list[Any]:
            return value.detach().bool().cpu().tolist()

        return {
            "schema": FRAME_SCHEMA + ".diagnostics",
            "valid": flags(self.valid),
            "active_mask": list(self.active_mask),
            "eigenvalues": values(self.eigenvalues),
            "condition_number": values(self.condition_number),
            "gram_error": values(self.gram_error),
            "realized_geodesic_angle": values(self.angle),
            "angle_cap_multiplier": values(self.angle_cap_multiplier),
            "scheduler_cap_multiplier": values(self.scheduler_cap_multiplier),
            "mapped_update_ratio": values(self.mapped_update_ratio),
            "gram_hash": list(self.gram_hash),
            "frame_hash": list(self.frame_hash),
        }


@dataclass(frozen=True)
class EulerFrameState:
    """Prepared state consumed by all three training objectives."""

    clean_latent: Tensor
    sample: Tensor
    native_model_output: Optional[Tensor]
    raw_bases: Tensor
    tangent_bases: Tensor
    mapped_bases: Tensor
    clean_bases: Tensor
    native_update: Tensor
    sigma_from: Tensor
    sigma_to: Tensor
    kappa: Tensor
    context: Tensor
    diagnostics: EulerFrameDiagnostics
    prediction_type: str = "epsilon"


@dataclass(frozen=True)
class EulerFrameOutput:
    guided_x0: Tensor
    residual: Tensor
    coefficients: Tensor
    diagnostics: EulerFrameDiagnostics


@dataclass(frozen=True)
class EulerMappedOutput:
    rendered: EulerFrameOutput
    model_output: Tensor
    predicted_prev_sample: Tensor
    mapped_intervention: Any
    solver_target_update_ratio: float = 0.0
    solver_evaluations: int = 1


def _whiten_frame(
    d: Tensor,
    active_mask: Tuple[bool, ...],
    *,
    epsilon: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[str], list[str]]:
    """Whiten active rows with the symmetric inverse square root of Gram."""
    batch, slots, dimensions = d.shape
    indices = [index for index, value in enumerate(active_mask) if value]
    rank = len(indices)
    mapped = torch.zeros_like(d)
    eigenvalues = d.new_zeros((batch, rank), dtype=torch.float32)
    condition = d.new_zeros((batch,), dtype=torch.float32)
    gram_error = d.new_zeros((batch,), dtype=torch.float32)
    valid = torch.ones(batch, device=d.device, dtype=torch.bool)
    gram_hash: list[str] = []
    frame_hash: list[str] = []
    for row in range(batch):
        active_d = d[row, indices].float()
        gram = active_d @ active_d.transpose(0, 1)
        gram = 0.5 * (gram + gram.transpose(0, 1))
        try:
            values, vectors = torch.linalg.eigh(gram.double())
        except RuntimeError:
            valid[row] = False
            gram_hash.append(tensor_sha256(gram))
            frame_hash.append(tensor_sha256(active_d))
            continue
        max_value = values.max().clamp_min(float(epsilon))
        floor = torch.maximum(
            torch.tensor(1e-12, device=d.device, dtype=torch.float64),
            1e-6 * max_value,
        )
        safe_values = values.clamp_min(floor)
        inv_sqrt = vectors @ torch.diag(safe_values.rsqrt()) @ vectors.transpose(0, 1)
        frame = inv_sqrt.float() @ active_d
        identity_error = torch.linalg.matrix_norm(
            frame @ frame.transpose(0, 1) - torch.eye(rank, device=d.device), ord=2
        )
        diagonal_alignment = torch.diag(frame @ active_d.transpose(0, 1))
        row_valid = (
            torch.isfinite(values).all()
            and torch.isfinite(frame).all()
            and bool(torch.all(diagonal_alignment > 0))
            and bool(identity_error <= 1e-3)
        )
        valid[row] = bool(row_valid)
        eigenvalues[row] = values.float()
        condition[row] = (safe_values.max() / safe_values.min()).float()
        gram_error[row] = identity_error.float()
        if row_valid:
            mapped[row, indices] = frame
        gram_hash.append(tensor_sha256(gram))
        frame_hash.append(tensor_sha256(frame))
    return mapped, eigenvalues, condition, gram_error, valid, gram_hash, frame_hash


def _frame_components(
    clean_latent: Tensor,
    raw_bases: Tensor,
    native_update: Tensor,
    sigma_from: Tensor | float,
    sigma_to: Tensor | float,
    active_mask: Tuple[bool, ...],
    *,
    epsilon: float,
    max_update_ratio: float,
    coefficient_bound: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, EulerFrameDiagnostics]:
    _raw_basis_shape(raw_bases, clean_latent, len(active_mask))
    _nchw(native_update, "native_update")
    if native_update.shape != clean_latent.shape:
        raise ValueError("native_update must match clean_latent")
    if native_update.device != clean_latent.device:
        raise ValueError("native_update and clean_latent must share one device")
    batch, slots = raw_bases.shape[:2]
    source, target, kappa = _sigma_pair(sigma_from, sigma_to, batch, clean_latent)
    native_norm = torch.linalg.vector_norm(native_update.float().flatten(1), dim=1)
    valid = (kappa >= 1e-6) & (native_norm >= 1e-6)
    tangent = _project_many(clean_latent, raw_bases, epsilon)
    d = tangent * kappa.reshape(-1, 1, 1, 1, 1)
    d_flat = d.flatten(2) / native_norm.reshape(-1, 1, 1).clamp_min(epsilon)
    whitened, eigenvalues, condition, gram_error, whitening_valid, gram_hash, frame_hash = _whiten_frame(
        d_flat, active_mask, epsilon=epsilon
    )
    valid = valid & whitening_valid
    beta = 0.999 * float(max_update_ratio) / (float(coefficient_bound) * math.sqrt(float(slots)))
    mapped = whitened.reshape_as(d) * (beta * native_norm.reshape(-1, 1, 1, 1, 1))
    clean = mapped / kappa.reshape(-1, 1, 1, 1, 1).clamp_min(epsilon)
    mapped = torch.where(valid.reshape(-1, 1, 1, 1, 1), mapped, torch.zeros_like(mapped))
    clean = torch.where(valid.reshape(-1, 1, 1, 1, 1), clean, torch.zeros_like(clean))
    # The diagnostics are completed by apply_coefficients for angle/cap values.
    diagnostics = EulerFrameDiagnostics(
        valid=valid,
        active_mask=active_mask,
        eigenvalues=eigenvalues,
        condition_number=condition,
        gram_error=gram_error,
        angle=torch.zeros(batch, device=clean_latent.device),
        angle_cap_multiplier=torch.ones(batch, device=clean_latent.device),
        scheduler_cap_multiplier=torch.ones(batch, device=clean_latent.device),
        mapped_update_ratio=torch.zeros(batch, device=clean_latent.device),
        gram_hash=tuple(gram_hash),
        frame_hash=tuple(frame_hash),
    )
    return tangent, mapped, clean, source, target, diagnostics


def _replace_diagnostics(
    base: EulerFrameDiagnostics,
    *,
    angle: Tensor,
    angle_cap_multiplier: Tensor,
    scheduler_cap_multiplier: Tensor,
    mapped_update_ratio: Tensor,
) -> EulerFrameDiagnostics:
    return EulerFrameDiagnostics(
        valid=base.valid,
        active_mask=base.active_mask,
        eigenvalues=base.eigenvalues,
        condition_number=base.condition_number,
        gram_error=base.gram_error,
        angle=angle,
        angle_cap_multiplier=angle_cap_multiplier,
        scheduler_cap_multiplier=scheduler_cap_multiplier,
        mapped_update_ratio=mapped_update_ratio,
        gram_hash=base.gram_hash,
        frame_hash=base.frame_hash,
    )


def _measure_mapped_intervention_fail_closed(
    sample: Tensor,
    native_prev_sample: Tensor,
    guided_prev_sample: Tensor,
) -> EulerMappedIntervention:
    """Measure mapped updates while representing zero-norm rows as no-ops."""
    work = sample.float()
    native = native_prev_sample.float()
    guided = guided_prev_sample.float()
    nominal_update = native - work
    intervention = guided - native
    nominal_norm = torch.linalg.vector_norm(nominal_update.flatten(1), dim=1)
    intervention_norm = torch.linalg.vector_norm(intervention.flatten(1), dim=1)
    if not (
        torch.isfinite(nominal_norm).all()
        and torch.isfinite(intervention_norm).all()
    ):
        raise RuntimeError("Euler mapped-intervention norms are non-finite")
    nonzero = nominal_norm > 1e-12
    ratio = torch.where(
        nonzero,
        intervention_norm / nominal_norm.clamp_min(1e-12),
        torch.zeros_like(intervention_norm),
    )
    return EulerMappedIntervention(
        native_prev_sample=native_prev_sample.detach(),
        intervention=intervention.detach(),
        nominal_update=nominal_update.detach(),
        intervention_norm=intervention_norm.detach(),
        nominal_update_norm=nominal_norm.detach(),
        ratio=ratio.detach(),
    )


class EulerNativeFrameV1(nn.Module):
    """The frozen six-slot frame and a 91,654-parameter coefficient policy."""

    # The production pipeline must not route this renderer through the
    # historical unit-gain post-step injection.  This identity is consumed by
    # the explicit Euler-native branch added to the pipeline adapter.
    scheduler_mapping = "euler_native_frame_v1"
    requires_euler_native_adapter = True
    requires_strict_scheduler_round_trip = True
    scheduler_pred_original_relative_l2_tolerance = 0.01
    scheduler_prev_sample_relative_l2_tolerance = 1e-3

    def __init__(
        self,
        *,
        active_mask: Optional[Sequence[bool]] = None,
        calibration: Optional[FrameCalibration] = None,
        latent_channels: int = 4,
        hidden_dim: int = 256,
        depth: int = 2,
        prompt_dim: int = 32,
        state_dim: int = 16,
        timestep_dim: int = 16,
        coefficient_bound: float = 1.0,
        max_update_ratio: float = 0.05,
        preserve_moments: bool = True,
        theta_max: float = 0.05,
        epsilon: float = 1e-12,
        pre_squash_sigma: float = 0.25,
    ) -> None:
        super().__init__()
        if calibration is None:
            raise ValueError(
                "a validated FrameCalibration is required; active_mask alone cannot "
                "construct a formal renderer"
            )
        if not isinstance(calibration, FrameCalibration):
            raise TypeError("calibration must be a FrameCalibration")
        if active_mask is not None:
            try:
                supplied_mask = tuple(active_mask)
            except TypeError as exc:
                raise ValueError("active_mask must be a sequence of booleans") from exc
            if any(not isinstance(value, bool) for value in supplied_mask):
                raise ValueError("active_mask must contain booleans")
            if supplied_mask != calibration.active_mask:
                raise ValueError("active_mask disagrees with calibration")
        mask = calibration.active_mask
        if len(mask) != len(CANONICAL_FRAME_SLOTS):
            raise ValueError("EulerNativeFrameV1 requires six canonical slots")
        if sum(mask) < 2:
            raise ValueError("EulerNativeFrameV1 requires global rank at least two")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (latent_channels, hidden_dim, depth)
        ) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (prompt_dim, state_dim, timestep_dim)
        ):
            raise ValueError("network dimensions must be positive; context widths may be zero")
        if (
            isinstance(coefficient_bound, bool)
            or not isinstance(coefficient_bound, (int, float))
            or not math.isfinite(float(coefficient_bound))
            or coefficient_bound != 1.0
            or isinstance(max_update_ratio, bool)
            or not isinstance(max_update_ratio, (int, float))
            or not math.isfinite(float(max_update_ratio))
            or max_update_ratio != 0.05
            or isinstance(theta_max, bool)
            or not isinstance(theta_max, (int, float))
            or not math.isfinite(float(theta_max))
            or theta_max <= 0
            or theta_max >= math.pi / 2
        ):
            raise ValueError(
                "bounds must use the registered coefficient and 0.05 update cap"
            )
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or epsilon <= 0
        ):
            raise ValueError("epsilon must be finite and positive")
        if not isinstance(preserve_moments, bool):
            raise TypeError("preserve_moments must be boolean")
        if (
            isinstance(pre_squash_sigma, bool)
            or not isinstance(pre_squash_sigma, (int, float))
            or not math.isfinite(float(pre_squash_sigma))
            or pre_squash_sigma <= 0
        ):
            raise ValueError("pre_squash_sigma must be finite and positive")
        self.active_mask = mask
        self.calibration = calibration
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.prompt_dim = int(prompt_dim)
        self.state_dim = int(state_dim)
        self.timestep_dim = int(timestep_dim)
        self.coefficient_bound = float(coefficient_bound)
        self.max_update_ratio = float(max_update_ratio)
        self.preserve_moments = bool(preserve_moments)
        self.theta_max = float(theta_max)
        self.epsilon = float(epsilon)
        self.pre_squash_sigma = float(pre_squash_sigma)
        self.contract = ActionSpaceContract(
            len(mask), mask, coefficient_bound, pre_squash_sigma
        )
        # This input layout is intentionally identical to the registered
        # coefficient-only StructuralLatentRenderer: 24 basis statistics,
        # 3 latent statistics, 3 spectral fractions, 16 time, 32 prompt and
        # 16 state features = 94 inputs.
        input_dim = len(mask) * 4 + 3 + 3 + timestep_dim + prompt_dim + state_dim
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(current, hidden_dim), nn.SiLU()))
            current = hidden_dim
        layers.append(nn.Linear(current, len(mask)))
        self.policy = nn.Sequential(*layers)
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)
        self.register_buffer(
            "active_mask_tensor", torch.tensor(mask, dtype=torch.bool), persistent=True
        )
        self.schema = FRAME_SCHEMA

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def frame_contract_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "canonical_slots": list(CANONICAL_FRAME_SLOTS),
            "action_space": self.contract.to_dict(),
            "calibration": self.calibration.to_dict(),
            "calibration_hash": self.calibration.calibration_hash,
            "latent_channels": self.latent_channels,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "prompt_dim": self.prompt_dim,
            "state_dim": self.state_dim,
            "timestep_dim": self.timestep_dim,
            "coefficient_bound": self.coefficient_bound,
            "max_update_ratio": self.max_update_ratio,
            "preserve_moments": self.preserve_moments,
            "theta_max": self.theta_max,
            "epsilon": self.epsilon,
            "pre_squash_sigma": self.pre_squash_sigma,
            "parameter_count": self.parameter_count,
        }
        return contract_hash(payload)

    @property
    def calibration_hash(self) -> str:
        return self.calibration.calibration_hash

    @property
    def mask_hash(self) -> str:
        return mask_sha256(self.active_mask)

    def _context(
        self,
        clean_latent: Tensor,
        raw_bases: Tensor,
        *,
        normalized_timestep: Optional[Tensor],
        prompt_embedding: Optional[Tensor],
        state_features: Optional[Tensor],
    ) -> Tensor:
        basis_stats = _basis_statistics(clean_latent, raw_bases, self.epsilon)
        flat = clean_latent.float().flatten(1)
        latent_stats = torch.stack(
            (flat.mean(dim=-1), flat.std(dim=-1, unbiased=False), torch.linalg.vector_norm(flat, dim=-1)),
            dim=-1,
        )
        spectral = _spectral_summary(clean_latent, self.epsilon)
        if normalized_timestep is None:
            normalized_timestep = clean_latent.new_zeros(clean_latent.shape[0])
        timestep = _coerce_batch_vector(
            normalized_timestep, clean_latent.shape[0], "normalized_timestep"
        ).to(device=clean_latent.device, dtype=torch.float32)
        time = _sinusoidal_embedding(timestep, self.timestep_dim)
        prompt = _coerce_context(
            prompt_embedding, clean_latent.shape[0], self.prompt_dim,
            "prompt_embedding", clean_latent
        )
        state = _coerce_context(
            state_features, clean_latent.shape[0], self.state_dim,
            "state_features", clean_latent
        )
        context = torch.cat((basis_stats, latent_stats, spectral, time, prompt, state), dim=-1)
        if context.shape[-1] != self.policy[0].in_features:
            raise RuntimeError("renderer context width does not match registered policy")
        return context.float()

    def _prepare_from_parts(
        self,
        *,
        clean_latent: Tensor,
        sample: Tensor,
        native_model_output: Optional[Tensor],
        raw_bases: Tensor,
        native_update: Tensor,
        sigma_from: Tensor | float,
        sigma_to: Tensor | float,
        normalized_timestep: Optional[Tensor],
        prompt_embedding: Optional[Tensor],
        state_features: Optional[Tensor],
        prediction_type: str,
    ) -> EulerFrameState:
        _nchw(clean_latent, "clean_latent")
        _nchw(sample, "sample")
        _nchw(native_update, "native_update")
        if clean_latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"clean_latent has {clean_latent.shape[1]} channels; "
                f"expected {self.latent_channels}"
            )
        if sample.shape != clean_latent.shape or sample.device != clean_latent.device:
            raise ValueError("sample must match clean_latent shape and device")
        if native_update.shape != clean_latent.shape or native_update.device != clean_latent.device:
            raise ValueError("native_update must match clean_latent shape and device")
        if native_model_output is not None:
            _nchw(native_model_output, "native_model_output")
            if (
                native_model_output.shape != clean_latent.shape
                or native_model_output.device != clean_latent.device
            ):
                raise ValueError(
                    "native_model_output must match clean_latent shape and device"
                )
        # Frame construction is a frozen state transform.  Detach it from
        # the backbone/decoder graph so objective gradients can only update
        # the policy (and the explicitly varied clean endpoint), never the
        # U-Net features or the state geometry.
        frame_latent = clean_latent.detach()
        frame_update = native_update.detach()
        tangent, mapped, clean, source, target, diagnostics = _frame_components(
            frame_latent,
            raw_bases.detach(),
            frame_update,
            sigma_from,
            sigma_to,
            self.active_mask,
            epsilon=self.epsilon,
            max_update_ratio=self.max_update_ratio,
            coefficient_bound=self.coefficient_bound,
        )
        context = self._context(
            frame_latent,
            raw_bases.detach(),
            normalized_timestep=normalized_timestep,
            prompt_embedding=prompt_embedding,
            state_features=state_features,
        )
        return EulerFrameState(
            clean_latent=clean_latent,
            sample=sample,
            native_model_output=(
                None if native_model_output is None else native_model_output.detach()
            ),
            raw_bases=raw_bases.detach(),
            tangent_bases=tangent,
            mapped_bases=mapped,
            clean_bases=clean,
            native_update=frame_update,
            sigma_from=source,
            sigma_to=target,
            kappa=1.0 - target / source,
            context=context,
            diagnostics=diagnostics,
            prediction_type=str(prediction_type),
        )

    def prepare_state(
        self,
        observation: RendererObservation,
        condition: RendererCondition,
        *,
        sigma_from: Optional[Tensor | float] = None,
        sigma_to: Optional[Tensor | float] = None,
        native_model_output: Optional[Tensor] = None,
        prediction_type: str = "epsilon",
    ) -> EulerFrameState:
        """Prepare one state without sampling or evaluating a reward model."""
        if not isinstance(observation, RendererObservation):
            raise TypeError("observation must be a RendererObservation")
        if not isinstance(condition, RendererCondition):
            raise TypeError("condition must be a RendererCondition")
        clean = observation.pred_original_sample
        sample = observation.latents_before_step
        _nchw(clean, "observation.pred_original_sample")
        _nchw(sample, "observation.latents_before_step")
        if not isinstance(observation.scheduler_update, Tensor):
            raise TypeError("observation.scheduler_update must be a Tensor")
        _nchw(observation.scheduler_update, "observation.scheduler_update")
        if (
            clean.shape != sample.shape
            or observation.scheduler_update.shape != clean.shape
            or clean.device != sample.device
            or observation.scheduler_update.device != clean.device
        ):
            raise ValueError("observation tensors must have identical shapes")
        if (sigma_from is None) != (sigma_to is None):
            raise ValueError("sigma_from and sigma_to must be supplied together")
        if sigma_from is None and sigma_to is None:
            raise ValueError(
                "explicit sigma_from and sigma_to are required; inferred Euler "
                "coordinates are not valid for a formal renderer state"
            )
        return self._prepare_from_parts(
            clean_latent=clean,
            sample=sample,
            native_model_output=native_model_output,
            raw_bases=condition.bases,
            native_update=observation.scheduler_update,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            normalized_timestep=observation.normalized_timestep,
            prompt_embedding=condition.prompt_embedding,
            state_features=condition.state_features,
            prediction_type=prediction_type,
        )

    def action_parameters(self, state: EulerFrameState) -> Tensor:
        """Return pre-squash means with exact zeros in inactive slots."""
        if not isinstance(state, EulerFrameState):
            raise TypeError("state must be an EulerFrameState")
        mean = self.policy(state.context.float())
        active = self.active_mask_tensor.to(device=mean.device)
        return torch.where(active, mean, torch.zeros_like(mean))

    def action_distribution(self, mean: Tensor):
        """Build the shared transformed-Gaussian distribution lazily."""
        from .distributions import SquashedGaussian

        return SquashedGaussian(mean, self.contract)

    def deterministic_action_from_mean(self, mean: Tensor) -> Tensor:
        """Map one policy mean to the exact deterministic deployment action."""
        if (
            not isinstance(mean, Tensor)
            or mean.ndim < 1
            or mean.shape[-1] != len(self.active_mask)
            or not mean.is_floating_point()
            or not torch.isfinite(mean).all()
        ):
            raise ValueError("mean must be a finite floating tensor with six slots")
        action = float(self.coefficient_bound) * torch.tanh(mean)
        # ``tanh`` can round to exactly one in reduced precision.  The action
        # contract uses an open interval so such a value has no finite inverse
        # transform density; keep deterministic deployment on the same
        # interior as stochastic sampling.
        bound = torch.tensor(
            self.coefficient_bound, device=action.device, dtype=action.dtype
        )
        try:
            interior = torch.nextafter(bound, torch.zeros_like(bound))
        except RuntimeError:
            interior = bound * (1.0 - 2.0 * torch.finfo(action.dtype).eps)
        action = action.clamp(-interior, interior)
        active = self.active_mask_tensor.to(device=action.device)
        return torch.where(active, action, torch.zeros_like(action))

    def deterministic_action(self, state: EulerFrameState) -> Tensor:
        return self.deterministic_action_from_mean(self.action_parameters(state))

    def apply_coefficients(
        self,
        state: EulerFrameState,
        action: Tensor,
        *,
        validate: bool = True,
    ) -> EulerFrameOutput:
        """Apply a coefficient action in the fixed frame and enforce all caps."""
        if not isinstance(state, EulerFrameState):
            raise TypeError("state must be an EulerFrameState")
        if not isinstance(action, Tensor) or action.shape != state.clean_latent.shape[:1] + (len(self.active_mask),):
            raise ValueError("action must have shape (batch, six)")
        if action.device != state.clean_latent.device:
            raise ValueError("action and renderer state must share one device")
        if not action.is_floating_point() or not torch.isfinite(action).all():
            raise ValueError("action must be finite and floating-point")
        if validate:
            self.contract.validate_action(action)
        calc_dtype = torch.float64 if action.dtype == torch.float64 else torch.float32
        action_work = action.to(dtype=calc_dtype)
        active = self.active_mask_tensor.to(device=action.device)
        action_work = torch.where(active, action_work, torch.zeros_like(action_work))
        basis = state.clean_bases
        # This is the first-order path of the renderer at the identity.  The
        # value is removed by ``candidate - candidate.detach()`` below, while
        # its derivative keeps a zero-initialised policy learnable under OPD.
        identity_tangent = (
            action_work[:, :, None, None, None] * basis
        ).sum(dim=1)
        identity_candidate = (
            state.clean_latent.float() + identity_tangent
        ).to(state.clean_latent.dtype)
        identity_with_grad = state.clean_latent + (
            identity_candidate - identity_candidate.detach()
        )
        # The explicit zero path is important: it preserves the clean endpoint
        # bytes and avoids creating a numerically different geodesic round trip.
        zero_row = torch.count_nonzero(action_work, dim=-1) == 0
        if bool(torch.all(zero_row)):
            zero = identity_with_grad - state.clean_latent
            diagnostics = _replace_diagnostics(
                state.diagnostics,
                angle=torch.zeros_like(state.diagnostics.angle),
                angle_cap_multiplier=torch.ones_like(
                    state.diagnostics.angle_cap_multiplier
                ),
                scheduler_cap_multiplier=torch.ones_like(
                    state.diagnostics.scheduler_cap_multiplier
                ),
                mapped_update_ratio=torch.zeros_like(
                    state.diagnostics.mapped_update_ratio
                ),
            )
            # Return the original tensor, rather than a float32 round trip, so
            # a no-op is exact for float16, float32, and float64 inputs alike.
            # ``identity_with_grad`` has the same value as the original tensor
            # but retains the renderer action's local derivative.
            return EulerFrameOutput(identity_with_grad, zero, action, diagnostics)
        tangent = (action_work[:, :, None, None, None] * basis).sum(dim=1)
        tangent = project_fixed_moment_tangent(state.clean_latent, tangent, self.epsilon)
        mapped = tangent * state.kappa.reshape(-1, 1, 1, 1)
        nominal = state.native_update.float()
        nominal_norm = torch.linalg.vector_norm(nominal.flatten(1), dim=1).clamp_min(self.epsilon)
        mapped_norm = torch.linalg.vector_norm(mapped.flatten(1), dim=1)
        scheduler_multiplier = torch.clamp(
            float(self.max_update_ratio) * nominal_norm / mapped_norm.clamp_min(self.epsilon), max=1.0
        )
        mapped = mapped * scheduler_multiplier.reshape(-1, 1, 1, 1)
        tangent = mapped / state.kappa.reshape(-1, 1, 1, 1).clamp_min(self.epsilon)
        centered = state.clean_latent.float() - state.clean_latent.float().mean(dim=(-2, -1), keepdim=True)
        centered_norm = torch.linalg.vector_norm(centered.flatten(1), dim=1).clamp_min(self.epsilon)
        tangent_norm = torch.linalg.vector_norm(tangent.flatten(1), dim=1)
        angle = tangent_norm / centered_norm
        angle_multiplier = torch.clamp(
            # Leave a small deterministic margin for float32 round-off: the
            # recorded angle must never cross the registered hard bound.
            math.tan(self.theta_max) * (1.0 - 1e-6) * centered_norm
            / tangent_norm.clamp_min(self.epsilon),
            max=1.0,
        )
        tangent = tangent * angle_multiplier.reshape(-1, 1, 1, 1)
        angle = torch.linalg.vector_norm(tangent.flatten(1), dim=1) / centered_norm
        angle = angle.clamp_max(float(self.theta_max))
        if self.preserve_moments:
            guided = _fixed_moment_geodesic(state.clean_latent, tangent, self.epsilon)
        else:
            guided = state.clean_latent.float() + tangent
        valid = state.diagnostics.valid
        valid_shape = valid.reshape(-1, 1, 1, 1)
        zero_shape = zero_row.reshape(-1, 1, 1, 1)
        guided = torch.where(valid_shape, guided, state.clean_latent.float())
        guided = guided.to(state.clean_latent.dtype)
        # Restore the original clean-latent bytes for zero rows after the
        # mixed-batch computation.  This also preserves their identity
        # gradient while allowing nonzero rows to remain differentiable.
        guided = torch.where(zero_shape, identity_with_grad, guided)
        realized = guided.float() - state.clean_latent.float()
        ratio = torch.linalg.vector_norm((realized * state.kappa.reshape(-1, 1, 1, 1)).flatten(1), dim=1) / nominal_norm
        diagnostics = _replace_diagnostics(
            state.diagnostics,
            angle=angle,
            angle_cap_multiplier=angle_multiplier,
            scheduler_cap_multiplier=scheduler_multiplier,
            mapped_update_ratio=ratio,
        )
        return EulerFrameOutput(guided, guided - state.clean_latent, action, diagnostics)

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
    ) -> EulerFrameOutput:
        """Compatibility call for the pipeline's clean-endpoint branch."""
        if scheduler_update is None:
            raise ValueError("scheduler_update is required for EulerNativeFrameV1")
        if clean_update_gain is None:
            clean_update_gain = latent.new_tensor(1.0)
        state = self._prepare_from_parts(
            clean_latent=latent,
            sample=latent,
            native_model_output=None,
            raw_bases=bases,
            native_update=scheduler_update,
            sigma_from=latent.new_tensor(1.0),
            sigma_to=latent.new_tensor(1.0) - torch.as_tensor(
                clean_update_gain, device=latent.device
            ),
            normalized_timestep=timestep,
            prompt_embedding=prompt_embedding,
            state_features=state_features,
            prediction_type="epsilon",
        )
        return self.apply_coefficients(state, self.deterministic_action(state))

    def forward_euler_mapped(
        self,
        pred_original_sample: Tensor,
        bases: Tensor,
        *,
        sample: Tensor,
        native_model_output: Tensor,
        sigma_from: Tensor | float,
        sigma_to: Tensor | float,
        prediction_type: str,
        scheduler_update: Tensor,
        clean_update_gain: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        prompt_embedding: Optional[Tensor] = None,
        state_features: Optional[Tensor] = None,
        action: Optional[Tensor] = None,
    ) -> EulerMappedOutput:
        _nchw(sample, "sample")
        _nchw(pred_original_sample, "pred_original_sample")
        if not isinstance(native_model_output, Tensor) or native_model_output.shape != sample.shape:
            raise ValueError("native_model_output must match sample")
        if native_model_output.device != sample.device:
            raise ValueError("native_model_output and sample must share one device")
        if not native_model_output.is_floating_point() or not torch.isfinite(native_model_output).all():
            raise ValueError("native_model_output must be finite and floating-point")
        # Bind every supplied native endpoint to the model output and sigma
        # pair.  Without this check a stale clean endpoint can be paired with
        # a fresh U-Net prediction and silently train on the wrong transition.
        native_endpoint = None
        sigma_from_values = torch.as_tensor(sigma_from, device=sample.device)
        sigma_to_values = torch.as_tensor(sigma_to, device=sample.device)
        if sigma_from_values.numel() > 1 or sigma_to_values.numel() > 1:
            try:
                endpoint_clean, endpoint_nominal, endpoint_gain, endpoint_source, endpoint_target = (
                    _batch_euler_endpoint(
                        sample,
                        native_model_output,
                        sigma_from,
                        sigma_to,
                        prediction_type,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid Euler native endpoint inputs") from exc
        else:
            try:
                from AttentionGuidance.latent_renderer import prepare_euler_clean_endpoint

                native_endpoint = prepare_euler_clean_endpoint(
                    sample,
                    native_model_output,
                    sigma_from=sigma_from,
                    sigma_to=sigma_to,
                    prediction_type=prediction_type,
                )
                endpoint_clean = native_endpoint.pred_original_sample
                endpoint_nominal = native_endpoint.nominal_update
                endpoint_gain = native_endpoint.clean_update_gain.reshape(-1)
                endpoint_source = native_endpoint.sigma_from.reshape(-1)
                endpoint_target = native_endpoint.sigma_to.reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid Euler native endpoint inputs") from exc
        for supplied, reconstructed, name in (
            (pred_original_sample, endpoint_clean, "pred_original_sample"),
            (scheduler_update, endpoint_nominal, "scheduler_update"),
        ):
            if supplied.shape != reconstructed.shape or supplied.device != reconstructed.device:
                raise ValueError(f"{name} must match the reconstructed native endpoint")
            relative = torch.linalg.vector_norm(
                (supplied.float() - reconstructed.float()).flatten(1), dim=1
            ) / (torch.linalg.vector_norm(reconstructed.float().flatten(1), dim=1) + 1e-12)
            if torch.any(relative > 1e-3):
                raise ValueError(f"{name} differs from the reconstructed native endpoint")
        if clean_update_gain is not None:
            supplied_gain = torch.as_tensor(
                clean_update_gain, device=sample.device, dtype=torch.float32
            )
            expected_gain = endpoint_gain.to(device=sample.device, dtype=torch.float32)
            if supplied_gain.numel() not in (1, sample.shape[0]):
                raise ValueError("clean_update_gain must be scalar or batch-shaped")
            supplied_gain = supplied_gain.reshape(-1)
            if supplied_gain.numel() == 1:
                supplied_gain = supplied_gain.expand_as(expected_gain.reshape(-1))
            relative_gain = (supplied_gain - expected_gain.reshape(-1)).abs() / (
                expected_gain.reshape(-1).abs() + 1e-12
            )
            if not torch.isfinite(supplied_gain).all() or torch.any(relative_gain > 1e-3):
                raise ValueError("clean_update_gain differs from the reconstructed endpoint")
        state = self._prepare_from_parts(
            clean_latent=pred_original_sample,
            sample=sample,
            native_model_output=native_model_output,
            raw_bases=bases,
            native_update=scheduler_update,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            normalized_timestep=timestep,
            prompt_embedding=prompt_embedding,
            state_features=state_features,
            prediction_type=prediction_type,
        )
        selected_action = (
            self.deterministic_action(state) if action is None else action
        )
        rendered = self.apply_coefficients(state, selected_action)
        if native_endpoint is None:
            native_prev = _batch_euler_prev_sample(
                sample,
                native_model_output,
                sigma_from,
                sigma_to,
                prediction_type,
            )
        else:
            native_prev = predict_euler_no_churn_prev_sample(
                sample,
                native_model_output,
                sigma_from=sigma_from,
                sigma_to=sigma_to,
                prediction_type=prediction_type,
            )
        zero_row = torch.count_nonzero(rendered.coefficients, dim=-1) == 0
        # Do not even invoke the clean-endpoint converter for a strict no-op.
        # Besides preserving bytes, this prevents an unnecessary numerical
        # path (and an unnecessary failure point) in the native baseline.
        if bool(torch.all(zero_row)):
            # Keep the native output/previous sample bytes while carrying the
            # renderer's zero-action endpoint derivative for native-transition
            # objectives.  No clean-endpoint converter is called here.
            endpoint_delta = rendered.guided_x0 - rendered.guided_x0.detach()
            source = endpoint_source.to(
                device=sample.device, dtype=native_model_output.dtype
            )
            target = endpoint_target.to(
                device=sample.device, dtype=native_model_output.dtype
            )
            source4 = source.reshape(-1, 1, 1, 1)
            gain = (source - target) / source
            if prediction_type == "epsilon":
                output_delta = -endpoint_delta / source4
            elif prediction_type in {"sample", "original_sample"}:
                output_delta = endpoint_delta
            elif prediction_type == "v_prediction":
                output_delta = -endpoint_delta * torch.sqrt(
                    1.0 + source4.square()
                ) / source4
            else:
                raise ValueError(f"unsupported Euler prediction_type {prediction_type!r}")
            model_with_grad = native_model_output + output_delta.to(
                dtype=native_model_output.dtype
            )
            prev_with_grad = native_prev + (
                gain.reshape(-1, 1, 1, 1)
                * endpoint_delta.to(dtype=native_prev.dtype)
            )
            measured = _measure_mapped_intervention_fail_closed(
                sample, native_prev, native_prev
            )
            return EulerMappedOutput(
                rendered=rendered,
                model_output=model_with_grad,
                predicted_prev_sample=prev_with_grad,
                mapped_intervention=measured,
                solver_target_update_ratio=0.0,
                solver_evaluations=1,
            )
        if native_endpoint is None:
            model_output = _batch_euler_model_output(
                sample,
                rendered.guided_x0,
                sigma_from,
                prediction_type,
                native_model_output.dtype,
            )
            predicted_prev = _batch_euler_prev_sample(
                sample,
                model_output,
                sigma_from,
                sigma_to,
                prediction_type,
            )
        else:
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
        # Reusing the frozen native bytes for an exact zero action is required
        # by the rollout contract.  Reconstructing epsilon from a quantized
        # clean endpoint can differ by one ULP (or more in fp16), which would
        # turn a strict no-op into a spurious intervention.
        zero_shape = zero_row.reshape(-1, 1, 1, 1)
        invalid_row = (~state.diagnostics.valid) | (
            torch.linalg.vector_norm(
                (native_prev.float() - sample.float()).flatten(1), dim=1
            ) <= 1e-12
        )
        endpoint_delta = rendered.guided_x0 - rendered.guided_x0.detach()
        source = endpoint_source.to(
            device=sample.device, dtype=native_model_output.dtype
        )
        target = endpoint_target.to(
            device=sample.device, dtype=native_model_output.dtype
        )
        source4 = source.reshape(-1, 1, 1, 1)
        gain = (source - target) / source
        if prediction_type == "epsilon":
            output_delta = -endpoint_delta / source4
        elif prediction_type in {"sample", "original_sample"}:
            output_delta = endpoint_delta
        elif prediction_type == "v_prediction":
            output_delta = -endpoint_delta * torch.sqrt(
                1.0 + source4.square()
            ) / source4
        else:
            raise ValueError(f"unsupported Euler prediction_type {prediction_type!r}")
        # Zero rows retain a straight-through identity derivative in mixed
        # batches as well; selecting a detached native tensor here would make
        # those rows invisible to OPD/RL gradients.
        zero_model_with_grad = native_model_output + output_delta.to(
            dtype=native_model_output.dtype
        )
        zero_prev_with_grad = native_prev + (
            gain.reshape(-1, 1, 1, 1)
            * endpoint_delta.to(dtype=native_prev.dtype)
        )
        fallback_shape = (zero_row | invalid_row).reshape(-1, 1, 1, 1)
        fallback_model = torch.where(
            zero_shape, zero_model_with_grad, native_model_output
        )
        fallback_prev = torch.where(zero_shape, zero_prev_with_grad, native_prev)
        model_output = torch.where(fallback_shape, fallback_model, model_output)
        predicted_prev = torch.where(fallback_shape, fallback_prev, predicted_prev)
        measured = _measure_mapped_intervention_fail_closed(
            sample, native_prev, predicted_prev
        )
        return EulerMappedOutput(
            rendered=rendered,
            model_output=model_output,
            predicted_prev_sample=predicted_prev,
            mapped_intervention=measured,
            solver_target_update_ratio=float(self.max_update_ratio),
            solver_evaluations=1,
        )


def _generator_from_key(key: str) -> torch.Generator:
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def haar_tangent_frame(
    clean_latent: Tensor,
    native_update: Tensor,
    *,
    active_mask: Sequence[bool],
    key: str,
    epsilon: float = 1e-12,
) -> Tensor:
    """Generate a deterministic equal-rank Haar tangent frame."""
    _nchw(clean_latent, "clean_latent")
    if native_update.shape != clean_latent.shape:
        raise ValueError("native_update must match clean_latent")
    if native_update.device != clean_latent.device:
        raise ValueError("native_update and clean_latent must share one device")
    try:
        mask = tuple(active_mask)
    except TypeError as exc:
        raise ValueError("active_mask must be a sequence of booleans") from exc
    if not mask or any(not isinstance(value, bool) for value in mask):
        raise ValueError("active_mask must contain non-empty boolean values")
    rank = sum(mask)
    if rank < 1:
        raise ValueError("at least one active slot is required")
    batch, channels, height, width = clean_latent.shape
    generator = _generator_from_key(key)
    random = torch.randn(
        batch, rank, channels, height, width, generator=generator, dtype=torch.float32
    ).to(device=clean_latent.device)
    random = _project_many(clean_latent, random, epsilon)
    result = torch.zeros(batch, len(mask), channels, height, width, device=clean_latent.device)
    for row in range(batch):
        matrix = random[row].reshape(rank, -1).double().transpose(0, 1)
        q, r = torch.linalg.qr(matrix, mode="reduced")
        signs = torch.sign(torch.diag(r)).where(torch.diag(r) != 0, torch.ones_like(torch.diag(r)))
        q = q * signs.reshape(1, -1)
        columns = q.transpose(0, 1).reshape(rank, channels, height, width).float()
        result[row, torch.as_tensor(mask, device=result.device, dtype=torch.bool)] = columns
    return result


def phase_matched_frame(mapped_bases: Tensor, *, key: str) -> Tensor:
    """Randomize only Fourier phase while preserving every rFFT magnitude."""
    if mapped_bases.ndim != 5 or not torch.isfinite(mapped_bases).all():
        raise ValueError("mapped_bases must be finite with shape (batch, slots, channels, height, width)")
    generator = _generator_from_key(key)
    spectrum = torch.fft.rfft2(mapped_bases.float(), dim=(-2, -1), norm="ortho")
    magnitude = spectrum.abs()
    phase = torch.rand(magnitude.shape, generator=generator, dtype=torch.float32).to(mapped_bases.device)
    randomized = magnitude * torch.exp(1j * (phase * (2.0 * math.pi) - math.pi))
    # For the omitted negative-rFFT half, the kx=0 and (when present) the
    # Nyquist columns must themselves be conjugate-symmetric along ky.  Keep
    # their magnitudes and mirror only their phases; all other columns already
    # acquire the conjugate counterpart through irfft2.
    height, width = mapped_bases.shape[-2:]
    self_conjugate_x = [0]
    if width % 2 == 0:
        self_conjugate_x.append(width // 2)
    for x_index in self_conjugate_x:
        for y_index in range(1, (height + 1) // 2):
            mirror = (-y_index) % height
            randomized[..., mirror, x_index] = randomized[..., y_index, x_index].conj()
        randomized[..., 0, x_index] = magnitude[..., 0, x_index] * torch.where(
            phase[..., 0, x_index] < 0.5,
            torch.ones_like(phase[..., 0, x_index]),
            -torch.ones_like(phase[..., 0, x_index]),
        )
        if height % 2 == 0:
            nyquist = height // 2
            randomized[..., nyquist, x_index] = magnitude[..., nyquist, x_index] * torch.where(
                phase[..., nyquist, x_index] < 0.5,
                torch.ones_like(phase[..., nyquist, x_index]),
                -torch.ones_like(phase[..., nyquist, x_index]),
            )
    return torch.fft.irfft2(
        randomized, s=mapped_bases.shape[-2:], dim=(-2, -1), norm="ortho"
    ).to(mapped_bases.dtype)


__all__ = [
    "CALIBRATION_DECISION_INDICES",
    "CALIBRATION_SCHEMA",
    "CALIBRATION_STATE_COUNT",
    "CANONICAL_FRAME_SLOTS",
    "FRAME_SCHEMA",
    "EulerFrameDiagnostics",
    "EulerFrameOutput",
    "EulerFrameState",
    "EulerMappedOutput",
    "EulerNativeFrameV1",
    "FrameCalibration",
    "FrameCalibrationSample",
    "calibrate_global_mask",
    "haar_tangent_frame",
    "mask_sha256",
    "phase_matched_frame",
    "project_fixed_moment_tangent",
    "tensor_sha256",
]
