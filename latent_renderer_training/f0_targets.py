"""Typed, detached targets and the exact F0 cross-fitting objective."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .objectives import normalized_transition_losses
from .renderer import (
    EulerFrameDiagnostics,
    EulerFrameState,
    EulerNativeFrameV1,
    tensor_sha256,
)


F0_TARGET_SCHEMA = "repldm.renderer_f0_target.v1"
F0_TRAINING_SEEDS = (2026090101, 2026090102)
F0_VALIDATION_SEEDS = (2026090191, 2026090192)
F0_DECISION_INDICES = (8, 24, 40)
F0_SOURCES = ("four_k_lsdb", "pixverve_95k")
F0_STRATA = (
    "nature",
    "urban",
    "people",
    "food",
    "artwork",
    "cgi",
    "animals",
    "architecture",
)


def f0_target_stable_id(
    *, prompt_id: str, generation_seed: int, decision_index: int, split: str
) -> str:
    """Return the canonical filename-safe identity for one logical F0 state."""
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError("F0 target prompt_id must be a non-empty string")
    if type(generation_seed) is not int or generation_seed < 0:
        raise ValueError("F0 target generation_seed must be non-negative")
    if type(decision_index) is not int or decision_index not in F0_DECISION_INDICES:
        raise ValueError("F0 target decision_index is not registered")
    if split not in {"train", "validation"}:
        raise ValueError("F0 target split must be train or validation")
    digest = hashlib.sha256(
        f"{prompt_id}\0{generation_seed}\0{decision_index}".encode("utf-8")
    ).hexdigest()[:20]
    return f"{split}-{generation_seed}-{decision_index:02d}-{digest}"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _copy_tensor(value: Tensor, *, device: torch.device | str | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("F0 state values must be tensors")
    result = value.detach().clone()
    if device is not None:
        result = result.to(device=device)
    return result


def _copy_diagnostics(
    value: EulerFrameDiagnostics, *, device: torch.device | str | None = None
) -> EulerFrameDiagnostics:
    if not isinstance(value, EulerFrameDiagnostics):
        raise TypeError("F0 target state has invalid frame diagnostics")
    return EulerFrameDiagnostics(
        valid=_copy_tensor(value.valid, device=device),
        active_mask=tuple(value.active_mask),
        eigenvalues=_copy_tensor(value.eigenvalues, device=device),
        condition_number=_copy_tensor(value.condition_number, device=device),
        gram_error=_copy_tensor(value.gram_error, device=device),
        angle=_copy_tensor(value.angle, device=device),
        angle_cap_multiplier=_copy_tensor(value.angle_cap_multiplier, device=device),
        scheduler_cap_multiplier=_copy_tensor(
            value.scheduler_cap_multiplier, device=device
        ),
        mapped_update_ratio=_copy_tensor(value.mapped_update_ratio, device=device),
        gram_hash=tuple(value.gram_hash),
        frame_hash=tuple(value.frame_hash),
    )


def detached_frame_state(
    value: EulerFrameState, *, device: torch.device | str | None = None
) -> EulerFrameState:
    """Return a non-aliasing state with no path to the U-Net or target graph."""
    if not isinstance(value, EulerFrameState):
        raise TypeError("F0 target state must be EulerFrameState")
    return EulerFrameState(
        clean_latent=_copy_tensor(value.clean_latent, device=device),
        sample=_copy_tensor(value.sample, device=device),
        native_model_output=(
            None
            if value.native_model_output is None
            else _copy_tensor(value.native_model_output, device=device)
        ),
        raw_bases=_copy_tensor(value.raw_bases, device=device),
        tangent_bases=_copy_tensor(value.tangent_bases, device=device),
        mapped_bases=_copy_tensor(value.mapped_bases, device=device),
        clean_bases=_copy_tensor(value.clean_bases, device=device),
        native_update=_copy_tensor(value.native_update, device=device),
        sigma_from=_copy_tensor(value.sigma_from, device=device),
        sigma_to=_copy_tensor(value.sigma_to, device=device),
        kappa=_copy_tensor(value.kappa, device=device),
        context=_copy_tensor(value.context, device=device),
        diagnostics=_copy_diagnostics(value.diagnostics, device=device),
        prediction_type=value.prediction_type,
    )


def _require_row_tensor(
    value: Tensor,
    *,
    name: str,
    shape: torch.Size,
    device: torch.device,
) -> Tensor:
    if not isinstance(value, Tensor) or value.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if value.device != device or not value.is_floating_point():
        raise ValueError(f"{name} must share the F0 state device and use floating point")
    if value.requires_grad or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be detached and finite")
    return value.detach().clone()


@dataclass(frozen=True)
class F0TargetRow:
    """One immutable logical F0 teacher state and its two native targets."""

    prompt_id: str
    generation_seed: int
    decision_index: int
    split: str
    source: str
    stratum: str
    fold: int | None
    state: EulerFrameState
    anchor_u: Tensor
    plus_u: Tensor
    minus_u: Tensor
    reward_gradient: Tensor
    plus_transition: Tensor
    minus_transition: Tensor
    reference_u: Tensor
    pair_weight: float
    valid: bool
    schema: str = F0_TARGET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != F0_TARGET_SCHEMA:
            raise ValueError("unsupported F0 target schema")
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("F0 prompt_id must be a non-empty string")
        if (
            isinstance(self.generation_seed, bool)
            or not isinstance(self.generation_seed, int)
            or self.generation_seed < 0
        ):
            raise ValueError("F0 generation_seed must be a non-negative integer")
        if self.decision_index not in F0_DECISION_INDICES or type(self.decision_index) is not int:
            raise ValueError("F0 decision_index differs from the registered schedule")
        if self.split not in {"train", "validation"}:
            raise ValueError("F0 split must be train or validation")
        expected_seeds = (
            F0_TRAINING_SEEDS if self.split == "train" else F0_VALIDATION_SEEDS
        )
        if self.generation_seed not in expected_seeds:
            raise ValueError("F0 generation seed differs from its split")
        if self.source not in F0_SOURCES or self.stratum not in F0_STRATA:
            raise ValueError("F0 source or stratum is not registered")
        if self.split == "train":
            if type(self.fold) is not int or self.fold not in range(4):
                raise ValueError("F0 training rows require fold 0..3")
        elif self.fold is not None:
            raise ValueError("F0 validation rows cannot carry a training fold")
        if not isinstance(self.valid, bool):
            raise TypeError("F0 target validity must be boolean")
        if (
            isinstance(self.pair_weight, bool)
            or not isinstance(self.pair_weight, (int, float))
            or not math.isfinite(float(self.pair_weight))
            or not 0.0 <= float(self.pair_weight) <= 1.0
        ):
            raise ValueError("F0 pair_weight must be finite and inside [0, 1]")

        state = detached_frame_state(self.state)
        batch = state.sample.shape[0]
        slots = len(state.diagnostics.active_mask)
        if batch != 1 or slots != 6:
            raise ValueError("F0 rows must contain one sample and six action slots")
        action_shape = torch.Size((batch, slots))
        latent_shape = state.native_update.shape
        if latent_shape != state.sample.shape:
            raise ValueError("F0 native update must match the scheduler sample")
        action_values = {
            "anchor_u": self.anchor_u,
            "plus_u": self.plus_u,
            "minus_u": self.minus_u,
            "reward_gradient": self.reward_gradient,
            "reference_u": self.reference_u,
        }
        copied_actions = {
            name: _require_row_tensor(
                value,
                name=name,
                shape=action_shape,
                device=state.sample.device,
            )
            for name, value in action_values.items()
        }
        copied_plus = _require_row_tensor(
            self.plus_transition,
            name="plus_transition",
            shape=latent_shape,
            device=state.sample.device,
        )
        copied_minus = _require_row_tensor(
            self.minus_transition,
            name="minus_transition",
            shape=latent_shape,
            device=state.sample.device,
        )
        active = torch.tensor(
            state.diagnostics.active_mask,
            device=state.sample.device,
            dtype=torch.bool,
        )
        for name, value in copied_actions.items():
            if value[..., ~active].numel() and torch.count_nonzero(value[..., ~active]):
                raise ValueError(f"{name} has a nonzero inactive coordinate")
        anchor = copied_actions["anchor_u"].double()
        for name in ("plus_u", "minus_u"):
            radius = torch.linalg.vector_norm(
                (copied_actions[name].double() - anchor)[..., active], dim=-1
            )
            if torch.any(radius > 0.5 + 1e-7):
                raise ValueError(f"{name} exceeds the frozen F0 trust radius")
        if self.valid and not bool(state.diagnostics.valid.all()):
            raise ValueError("a valid F0 target cannot contain an invalid frame")

        object.__setattr__(self, "state", state)
        for name, value in copied_actions.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "plus_transition", copied_plus)
        object.__setattr__(self, "minus_transition", copied_minus)
        object.__setattr__(self, "pair_weight", float(self.pair_weight))

    @property
    def stable_id(self) -> str:
        return f0_target_stable_id(
            prompt_id=self.prompt_id,
            generation_seed=self.generation_seed,
            decision_index=self.decision_index,
            split=self.split,
        )

    @property
    def record_sha256(self) -> str:
        tensor_fields = {
            "sample": self.state.sample,
            "clean_latent": self.state.clean_latent,
            "native_model_output": self.state.native_model_output,
            "raw_bases": self.state.raw_bases,
            "tangent_bases": self.state.tangent_bases,
            "mapped_bases": self.state.mapped_bases,
            "native_update": self.state.native_update,
            "sigma_from": self.state.sigma_from,
            "sigma_to": self.state.sigma_to,
            "kappa": self.state.kappa,
            "context": self.state.context,
            "clean_bases": self.state.clean_bases,
            "diagnostics_valid": self.state.diagnostics.valid,
            "eigenvalues": self.state.diagnostics.eigenvalues,
            "condition_number": self.state.diagnostics.condition_number,
            "gram_error": self.state.diagnostics.gram_error,
            "angle": self.state.diagnostics.angle,
            "angle_cap_multiplier": self.state.diagnostics.angle_cap_multiplier,
            "scheduler_cap_multiplier": self.state.diagnostics.scheduler_cap_multiplier,
            "mapped_update_ratio": self.state.diagnostics.mapped_update_ratio,
            "anchor_u": self.anchor_u,
            "plus_u": self.plus_u,
            "minus_u": self.minus_u,
            "reward_gradient": self.reward_gradient,
            "plus_transition": self.plus_transition,
            "minus_transition": self.minus_transition,
            "reference_u": self.reference_u,
        }
        payload = {
            "schema": self.schema,
            "prompt_id": self.prompt_id,
            "generation_seed": self.generation_seed,
            "decision_index": self.decision_index,
            "split": self.split,
            "source": self.source,
            "stratum": self.stratum,
            "fold": self.fold,
            "pair_weight": self.pair_weight,
            "valid": self.valid,
            "active_mask": list(self.state.diagnostics.active_mask),
            "gram_hash": list(self.state.diagnostics.gram_hash),
            "frame_hash": list(self.state.diagnostics.frame_hash),
            "prediction_type": self.state.prediction_type,
            "tensors": {
                name: None if value is None else tensor_sha256(value)
                for name, value in tensor_fields.items()
            },
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()

    def to(self, device: torch.device | str) -> "F0TargetRow":
        """Return an independent row on ``device`` while preserving tensor dtypes."""
        return replace(
            self,
            state=detached_frame_state(self.state, device=device),
            anchor_u=_copy_tensor(self.anchor_u, device=device),
            plus_u=_copy_tensor(self.plus_u, device=device),
            minus_u=_copy_tensor(self.minus_u, device=device),
            reward_gradient=_copy_tensor(self.reward_gradient, device=device),
            plus_transition=_copy_tensor(self.plus_transition, device=device),
            minus_transition=_copy_tensor(self.minus_transition, device=device),
            reference_u=_copy_tensor(self.reference_u, device=device),
        )


def f0_target_to_payload(value: F0TargetRow) -> dict[str, Any]:
    """Convert a target to a weights-only-safe tensor/primitive mapping."""
    if not isinstance(value, F0TargetRow):
        raise TypeError("value must be an F0TargetRow")
    state = value.state
    diagnostics = state.diagnostics
    return {
        "schema": F0_TARGET_SCHEMA,
        "record_sha256": value.record_sha256,
        "metadata": {
            "prompt_id": value.prompt_id,
            "generation_seed": value.generation_seed,
            "decision_index": value.decision_index,
            "split": value.split,
            "source": value.source,
            "stratum": value.stratum,
            "fold": value.fold,
            "pair_weight": value.pair_weight,
            "valid": value.valid,
        },
        "actions": {
            "anchor_u": value.anchor_u.detach().cpu(),
            "plus_u": value.plus_u.detach().cpu(),
            "minus_u": value.minus_u.detach().cpu(),
            "reward_gradient": value.reward_gradient.detach().cpu(),
            "reference_u": value.reference_u.detach().cpu(),
        },
        "targets": {
            "plus_transition": value.plus_transition.detach().cpu(),
            "minus_transition": value.minus_transition.detach().cpu(),
        },
        "state": {
            "clean_latent": state.clean_latent.detach().cpu(),
            "sample": state.sample.detach().cpu(),
            "native_model_output": (
                None
                if state.native_model_output is None
                else state.native_model_output.detach().cpu()
            ),
            "raw_bases": state.raw_bases.detach().cpu(),
            "tangent_bases": state.tangent_bases.detach().cpu(),
            "mapped_bases": state.mapped_bases.detach().cpu(),
            "clean_bases": state.clean_bases.detach().cpu(),
            "native_update": state.native_update.detach().cpu(),
            "sigma_from": state.sigma_from.detach().cpu(),
            "sigma_to": state.sigma_to.detach().cpu(),
            "kappa": state.kappa.detach().cpu(),
            "context": state.context.detach().cpu(),
            "prediction_type": state.prediction_type,
            "diagnostics": {
                "valid": diagnostics.valid.detach().cpu(),
                "active_mask": list(diagnostics.active_mask),
                "eigenvalues": diagnostics.eigenvalues.detach().cpu(),
                "condition_number": diagnostics.condition_number.detach().cpu(),
                "gram_error": diagnostics.gram_error.detach().cpu(),
                "angle": diagnostics.angle.detach().cpu(),
                "angle_cap_multiplier": diagnostics.angle_cap_multiplier.detach().cpu(),
                "scheduler_cap_multiplier": diagnostics.scheduler_cap_multiplier.detach().cpu(),
                "mapped_update_ratio": diagnostics.mapped_update_ratio.detach().cpu(),
                "gram_hash": list(diagnostics.gram_hash),
                "frame_hash": list(diagnostics.frame_hash),
            },
        },
    }


def _mapping(
    value: Any, *, fields: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields differ from the registered F0 schema")
    return value


def f0_target_from_payload(value: Mapping[str, Any]) -> F0TargetRow:
    """Rebuild and validate a target loaded with ``torch.load(weights_only=True)``."""
    root = _mapping(
        value,
        fields={"schema", "record_sha256", "metadata", "actions", "targets", "state"},
        label="F0 target payload",
    )
    if root["schema"] != F0_TARGET_SCHEMA:
        raise ValueError("unsupported F0 target payload schema")
    metadata = _mapping(
        root["metadata"],
        fields={
            "prompt_id",
            "generation_seed",
            "decision_index",
            "split",
            "source",
            "stratum",
            "fold",
            "pair_weight",
            "valid",
        },
        label="F0 target metadata",
    )
    actions = _mapping(
        root["actions"],
        fields={"anchor_u", "plus_u", "minus_u", "reward_gradient", "reference_u"},
        label="F0 target actions",
    )
    targets = _mapping(
        root["targets"],
        fields={"plus_transition", "minus_transition"},
        label="F0 target transitions",
    )
    state_value = _mapping(
        root["state"],
        fields={
            "clean_latent",
            "sample",
            "native_model_output",
            "raw_bases",
            "tangent_bases",
            "mapped_bases",
            "clean_bases",
            "native_update",
            "sigma_from",
            "sigma_to",
            "kappa",
            "context",
            "prediction_type",
            "diagnostics",
        },
        label="F0 frame state",
    )
    diagnostics_value = _mapping(
        state_value["diagnostics"],
        fields={
            "valid",
            "active_mask",
            "eigenvalues",
            "condition_number",
            "gram_error",
            "angle",
            "angle_cap_multiplier",
            "scheduler_cap_multiplier",
            "mapped_update_ratio",
            "gram_hash",
            "frame_hash",
        },
        label="F0 frame diagnostics",
    )
    try:
        diagnostics = EulerFrameDiagnostics(
            valid=diagnostics_value["valid"],
            active_mask=tuple(diagnostics_value["active_mask"]),
            eigenvalues=diagnostics_value["eigenvalues"],
            condition_number=diagnostics_value["condition_number"],
            gram_error=diagnostics_value["gram_error"],
            angle=diagnostics_value["angle"],
            angle_cap_multiplier=diagnostics_value["angle_cap_multiplier"],
            scheduler_cap_multiplier=diagnostics_value["scheduler_cap_multiplier"],
            mapped_update_ratio=diagnostics_value["mapped_update_ratio"],
            gram_hash=tuple(diagnostics_value["gram_hash"]),
            frame_hash=tuple(diagnostics_value["frame_hash"]),
        )
        state = EulerFrameState(
            clean_latent=state_value["clean_latent"],
            sample=state_value["sample"],
            native_model_output=state_value["native_model_output"],
            raw_bases=state_value["raw_bases"],
            tangent_bases=state_value["tangent_bases"],
            mapped_bases=state_value["mapped_bases"],
            clean_bases=state_value["clean_bases"],
            native_update=state_value["native_update"],
            sigma_from=state_value["sigma_from"],
            sigma_to=state_value["sigma_to"],
            kappa=state_value["kappa"],
            context=state_value["context"],
            diagnostics=diagnostics,
            prediction_type=state_value["prediction_type"],
        )
        row = F0TargetRow(
            prompt_id=metadata["prompt_id"],
            generation_seed=metadata["generation_seed"],
            decision_index=metadata["decision_index"],
            split=metadata["split"],
            source=metadata["source"],
            stratum=metadata["stratum"],
            fold=metadata["fold"],
            state=state,
            anchor_u=actions["anchor_u"],
            plus_u=actions["plus_u"],
            minus_u=actions["minus_u"],
            reward_gradient=actions["reward_gradient"],
            plus_transition=targets["plus_transition"],
            minus_transition=targets["minus_transition"],
            reference_u=actions["reference_u"],
            pair_weight=metadata["pair_weight"],
            valid=metadata["valid"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("F0 target payload contains invalid values") from exc
    if root["record_sha256"] != row.record_sha256:
        raise ValueError("F0 target payload differs from its record hash")
    return row


@dataclass(frozen=True)
class F0FoldSplit:
    held_out_fold: int
    fit_rows: tuple[F0TargetRow, ...]
    held_out_rows: tuple[F0TargetRow, ...]


def validate_f0_rows(
    values: Iterable[F0TargetRow], *, split: str
) -> tuple[F0TargetRow, ...]:
    """Validate the complete frozen train or validation state matrix."""
    rows = tuple(values)
    if split not in {"train", "validation"}:
        raise ValueError("F0 row validation requires train or validation")
    expected_prompts = 64 if split == "train" else 32
    expected_rows = expected_prompts * len(
        F0_TRAINING_SEEDS if split == "train" else F0_VALIDATION_SEEDS
    ) * len(F0_DECISION_INDICES)
    if len(rows) != expected_rows:
        raise ValueError(f"F0 {split} matrix must contain exactly {expected_rows} rows")
    if any(not isinstance(row, F0TargetRow) or row.split != split for row in rows):
        raise ValueError(f"F0 {split} matrix contains a wrong row type or split")
    keys = {
        (row.prompt_id, row.generation_seed, row.decision_index) for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("F0 state matrix contains duplicate logical states")

    prompts: dict[str, tuple[str, str, int | None]] = {}
    prompt_rows: dict[str, set[tuple[int, int]]] = {}
    for row in rows:
        identity = (row.source, row.stratum, row.fold)
        if row.prompt_id in prompts and prompts[row.prompt_id] != identity:
            raise ValueError("F0 prompt metadata changes across its logical states")
        prompts[row.prompt_id] = identity
        prompt_rows.setdefault(row.prompt_id, set()).add(
            (row.generation_seed, row.decision_index)
        )
    expected_grid = {
        (seed, decision)
        for seed in (
            F0_TRAINING_SEEDS if split == "train" else F0_VALIDATION_SEEDS
        )
        for decision in F0_DECISION_INDICES
    }
    if len(prompts) != expected_prompts or any(
        grid != expected_grid for grid in prompt_rows.values()
    ):
        raise ValueError("F0 prompts do not each contain the complete seed-decision grid")

    cells: dict[tuple[str, str, int | None], int] = {}
    for source, stratum, fold in prompts.values():
        key = (source, stratum, fold)
        cells[key] = cells.get(key, 0) + 1
    if split == "train":
        for source in F0_SOURCES:
            for stratum in F0_STRATA:
                for fold in range(4):
                    if cells.get((source, stratum, fold)) != 1:
                        raise ValueError(
                            "F0 train folds are not stratified one prompt per source-stratum cell"
                        )
    else:
        for source in F0_SOURCES:
            for stratum in F0_STRATA:
                if cells.get((source, stratum, None)) != 2:
                    raise ValueError(
                        "F0 validation prompts violate the source-stratum quota"
                    )
    return rows


def f0_crossfit_splits(values: Iterable[F0TargetRow]) -> tuple[F0FoldSplit, ...]:
    """Return the four fixed 288-row fit / 96-row held-out partitions."""
    rows = validate_f0_rows(values, split="train")
    result = []
    for fold in range(4):
        fit = tuple(row for row in rows if row.fold != fold)
        held_out = tuple(row for row in rows if row.fold == fold)
        if len(fit) != 288 or len(held_out) != 96:
            raise RuntimeError("F0 cross-fitting partition has an invalid size")
        result.append(F0FoldSplit(fold, fit, held_out))
    return tuple(result)


def _require_renderer_pair(
    renderer: EulerNativeFrameV1, reference: EulerNativeFrameV1
) -> None:
    if not isinstance(renderer, EulerNativeFrameV1) or not isinstance(
        reference, EulerNativeFrameV1
    ):
        raise TypeError("F0 objective requires EulerNativeFrameV1 renderers")
    if renderer is reference:
        raise ValueError("F0 behavior and reference renderers must be distinct")
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError("F0 reference renderer must be frozen")
    if (
        renderer.frame_contract_hash != reference.frame_contract_hash
        or renderer.calibration_hash != reference.calibration_hash
        or renderer.contract.to_dict() != reference.contract.to_dict()
    ):
        raise ValueError("F0 renderer and reference contracts differ")


def f0_objective(
    renderer: EulerNativeFrameV1,
    reference_renderer: EulerNativeFrameV1,
    values: Sequence[F0TargetRow],
    *,
    branch_coefficient: float = 1.0,
    anchor_weight: float = 0.10,
) -> Tensor:
    """Compute the frozen native-transition F0 objective without reward graphs."""
    _require_renderer_pair(renderer, reference_renderer)
    if branch_coefficient != 1.0:
        raise ValueError("F0 branch_coefficient must remain exactly 1.0")
    if anchor_weight != 0.10:
        raise ValueError("F0 anchor_weight must remain exactly 0.10")
    rows = tuple(values)
    if not rows or any(not isinstance(row, F0TargetRow) for row in rows):
        raise ValueError("F0 objective requires at least one typed target row")
    parameter = next(renderer.parameters())
    losses: list[Tensor] = []
    validity: list[Tensor] = []
    connected_zero: Tensor | None = None
    active = torch.as_tensor(
        renderer.active_mask, device=parameter.device, dtype=torch.bool
    )
    for row in rows:
        state = row.state
        if state.sample.device != parameter.device:
            raise ValueError("F0 target row is on a different device from the renderer")
        if tuple(state.diagnostics.active_mask) != tuple(renderer.active_mask):
            raise ValueError("F0 target row uses a different global action mask")
        mean = renderer.action_parameters(state)
        connected_zero = mean.sum() * 0.0 if connected_zero is None else connected_zero
        with torch.no_grad():
            reference = reference_renderer.action_parameters(state).detach()
        if reference.shape != row.reference_u.shape or not torch.equal(
            reference, row.reference_u
        ):
            raise ValueError("F0 target row reference mean differs from frozen C0")
        if mean.shape != row.anchor_u.shape:
            raise ValueError("F0 target anchor shape differs from renderer output")
        delta = float(branch_coefficient) * (mean - row.anchor_u)
        plus_u = row.anchor_u + delta
        minus_u = row.anchor_u - delta
        plus_action = renderer.deterministic_action_from_mean(plus_u)
        minus_action = renderer.deterministic_action_from_mean(minus_u)
        plus = renderer.apply_coefficients(state, plus_action)
        minus = renderer.apply_coefficients(state, minus_action)
        kappa = state.kappa.to(
            device=plus.residual.device, dtype=plus.residual.dtype
        ).reshape(-1, 1, 1, 1)
        nominal = state.native_update.to(dtype=plus.residual.dtype)
        plus_transition = nominal + kappa * plus.residual
        minus_transition = nominal + kappa * minus.residual
        valid_tensor = torch.tensor(
            [row.valid], device=parameter.device, dtype=torch.bool
        )
        valid_tensor = (
            valid_tensor
            & state.diagnostics.valid
            & plus.diagnostics.valid
            & minus.diagnostics.valid
        )
        plus_loss = normalized_transition_losses(
            plus_transition,
            row.plus_transition.detach(),
            nominal,
            valid_mask=valid_tensor,
        )
        minus_loss = normalized_transition_losses(
            minus_transition,
            row.minus_transition.detach(),
            nominal,
            valid_mask=valid_tensor,
        )
        anchor = (mean[..., active] - reference[..., active]).square().mean(dim=-1)
        row_loss = (
            float(row.pair_weight) * plus_loss
            + (1.0 - float(row.pair_weight)) * minus_loss
            + float(anchor_weight) * anchor
        )
        row_valid = (
            valid_tensor
            & torch.isfinite(row_loss)
            & torch.isfinite(anchor)
            & torch.isfinite(row.plus_transition).flatten(1).all(dim=1)
            & torch.isfinite(row.minus_transition).flatten(1).all(dim=1)
        )
        losses.append(torch.nan_to_num(row_loss, nan=0.0, posinf=0.0, neginf=0.0))
        validity.append(row_valid)
    stacked = torch.cat(losses)
    valid = torch.cat(validity)
    if not bool(valid.any()):
        if connected_zero is None:
            raise RuntimeError("F0 objective failed to construct a differentiable zero")
        return connected_zero
    weights = valid.to(dtype=stacked.dtype)
    return (stacked * weights).sum() / weights.sum()


__all__ = [
    "F0_DECISION_INDICES",
    "F0_SOURCES",
    "F0_STRATA",
    "F0_TARGET_SCHEMA",
    "F0_TRAINING_SEEDS",
    "F0_VALIDATION_SEEDS",
    "F0FoldSplit",
    "F0TargetRow",
    "detached_frame_state",
    "f0_crossfit_splits",
    "f0_target_stable_id",
    "f0_objective",
    "f0_target_from_payload",
    "f0_target_to_payload",
    "validate_f0_rows",
]
