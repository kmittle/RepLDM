"""Shared, tensor-only objectives for formal renderer method handlers.

This module deliberately does not load SDXL, data, or reward weights.  It turns
detached, already charged rollout records into one optimizer loss.  Keeping the
conversion here makes OPD, search distillation, DPO, and RL consume identical
trajectory semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from ..artifacts import module_state_sha256
from ..objectives import dpo_loss, opd_loss, per_decision_rl_loss, search_distill_loss
from ..operations import operation_output_sha256
from ..preferences import PreferenceLabelProvenance
from ..renderer import EulerFrameState, EulerNativeFrameV1, tensor_sha256
from ..rollout import BranchTrajectory, RolloutCollection, Transition


PAIR_BRANCHES = ("plus", "minus")
TIE_EPSILON = 1e-6
OPD_DECISION_INDICES = (8, 24, 40)


def _json_copy(value: Any, *, label: str) -> Any:
    """Copy and validate JSON provenance without retaining caller mutability."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc


def _hash_tensor_descriptor(digest: "hashlib._Hash", name: str, value: Tensor) -> None:
    """Bind a tensor's semantic shape/dtype as well as its raw bytes."""
    if not isinstance(value, Tensor):
        raise TypeError(f"anchor state field {name!r} must be a tensor")
    descriptor = {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "hash": tensor_sha256(value),
    }
    payload = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def opd_anchor_state_sha256(transition: Transition) -> str:
    """Hash the exact detached frame visited by an OPD student anchor.

    The raw sample, clean endpoint, scheduler update, and all renderer basis
    tensors are included with shape and dtype metadata.  This prevents a
    target charged for one frame from being silently reused for another frame
    whose byte representation happens to be compatible.
    """
    if not isinstance(transition, Transition):
        raise TypeError("OPD anchor state hash requires a Transition")
    state = transition.state
    if not isinstance(state, EulerFrameState):
        raise TypeError("OPD anchor transition state must be EulerFrameState")
    digest = hashlib.sha256(b"repldm.opd.anchor_state.v1\0")
    step = transition.step_index
    if type(step) is not int:
        raise ValueError("OPD anchor transition must have an integer step index")
    digest.update(step.to_bytes(8, "big", signed=True))
    for name in (
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
    ):
        value = getattr(state, name)
        digest.update(name.encode("utf-8") + b"\0")
        if value is None:
            digest.update(b"none\0")
        else:
            _hash_tensor_descriptor(digest, name, value)
    diagnostics = state.diagnostics
    _hash_tensor_descriptor(digest, "diagnostics.valid", diagnostics.valid)
    digest.update(
        json.dumps(
            {
                "active_mask": list(diagnostics.active_mask),
                "gram_hash": list(diagnostics.gram_hash),
                "frame_hash": list(diagnostics.frame_hash),
                "prediction_type": state.prediction_type,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    # The objective also consumes the recorded native transition as its
    # anchor baseline.  Bind it (and the deterministic action/result) so a
    # caller cannot mutate those detached fields while retaining the same
    # frame hash.
    for name in ("action", "native_transition", "rendered_transition"):
        _hash_tensor_descriptor(digest, f"transition.{name}", getattr(transition, name))
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    """Validate a lower-case SHA-256 digest used by an OPD label."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


@dataclass(frozen=True)
class OpdTeacherLabel:
    """One detached, ledgered teacher target for a student anchor state.

    The target is copied at construction and its tensor/output hashes are
    checked again before every objective evaluation.  This matters because a
    detached tensor remains mutable in Python; without the second check a
    caller could alter the in-memory target after the corresponding ``label``
    receipt had been written.
    """

    decision_index: int
    target: Tensor
    target_sha256: str
    receipt_output_sha256: str
    frame_valid: bool
    teacher_checkpoint_sha256: str
    student_state: str = "anchor"
    # ``teacher_checkpoint_sha256`` is the historical field name and denotes
    # the immutable renderer-state hash.  Keep an explicit alias so manifests
    # cannot confuse it with the checkpoint *file* hash.
    teacher_state_sha256: str | None = None
    teacher_checkpoint_file_sha256: str | None = None
    reservation_id: str | None = None
    operation_context: Mapping[str, Any] | None = None
    student_state_sha256: str | None = None
    raw_target_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, Tensor):
            raise TypeError("OPD teacher target must be a tensor")
        # Keep ownership of the detached bytes.  The formal target must never
        # retain an autograd edge back to the teacher or student rollout.
        object.__setattr__(self, "target", self.target.detach().clone())
        if self.teacher_state_sha256 is None:
            object.__setattr__(self, "teacher_state_sha256", self.teacher_checkpoint_sha256)
        if self.raw_target_sha256 is None:
            object.__setattr__(self, "raw_target_sha256", self.target_sha256)
        if self.operation_context is not None:
            object.__setattr__(
                self,
                "operation_context",
                _json_copy(self.operation_context, label="OPD operation_context"),
            )
        self.validate()

    def validate(self, *, strict: bool = False) -> None:
        """Recheck mutable target bytes and all receipt bindings.

        ``strict=False`` preserves compatibility with direct historical
        objective callers.  Formal runner/storage paths use ``strict=True``
        and require every receipt, checkpoint, and anchor-state binding.
        """
        if type(strict) is not bool:
            raise TypeError("strict must be boolean")
        if type(self.decision_index) is not int or self.decision_index not in OPD_DECISION_INDICES:
            raise ValueError("OPD teacher label has an invalid decision index")
        target = self.target
        if target.ndim != 4 or not target.is_floating_point():
            raise ValueError("OPD teacher target must be a floating NCHW tensor")
        if target.requires_grad or not torch.isfinite(target).all():
            raise ValueError("OPD teacher target must be detached and finite")
        _require_sha256(self.target_sha256, label="OPD teacher target hash")
        _require_sha256(
            self.receipt_output_sha256, label="OPD teacher label receipt hash"
        )
        _require_sha256(
            self.teacher_checkpoint_sha256,
            label="OPD teacher checkpoint hash",
        )
        _require_sha256(
            self.teacher_state_sha256,
            label="OPD teacher state hash",
        )
        if self.teacher_state_sha256 != self.teacher_checkpoint_sha256:
            raise ValueError("OPD teacher state/checkpoint hashes differ")
        if self.target_sha256 != tensor_sha256(target):
            raise ValueError("OPD teacher target bytes differ from the label hash")
        if self.receipt_output_sha256 != operation_output_sha256(target):
            raise ValueError("OPD teacher label receipt does not bind its target")
        if type(self.frame_valid) is not bool:
            raise TypeError("OPD teacher label frame_valid must be boolean")
        if self.student_state != "anchor":
            raise ValueError("OPD teacher labels must bind the student anchor state")
        _require_sha256(self.raw_target_sha256, label="OPD raw target hash")
        if self.raw_target_sha256 != self.target_sha256:
            raise ValueError("OPD raw target hash differs from target hash")
        if self.teacher_checkpoint_file_sha256 is not None:
            _require_sha256(
                self.teacher_checkpoint_file_sha256,
                label="OPD teacher checkpoint file hash",
            )
        if self.student_state_sha256 is not None:
            _require_sha256(
                self.student_state_sha256,
                label="OPD student state hash",
            )
        if self.reservation_id is not None and (
            not isinstance(self.reservation_id, str) or not self.reservation_id
        ):
            raise ValueError("OPD label reservation_id must be a non-empty string")
        if self.operation_context is not None:
            context = self.operation_context
            if not isinstance(context, Mapping):
                raise TypeError("OPD operation_context must be a mapping")
            required = {
                "split", "prompt", "seed", "step", "prefix", "branch", "action",
                "checkpoint_hash", "image_hash", "cached_parent",
            }
            if set(context) != required:
                raise ValueError("OPD operation_context has an unsupported shape")
            if context["step"] != self.decision_index:
                raise ValueError("OPD operation_context step differs from decision")
            if context["prefix"] != 8 or context["branch"] != "teacher":
                raise ValueError("OPD operation_context is not bound to the teacher decision")
            if context["checkpoint_hash"] != self.teacher_state_sha256:
                raise ValueError("OPD operation_context checkpoint differs from teacher state")
            action = context["action"]
            if not isinstance(action, Mapping):
                raise ValueError("OPD operation_context action must be a mapping")
            if action.get("target_sha256") != self.target_sha256:
                raise ValueError("OPD operation_context target differs from the label")
            if action.get("decision_index") != self.decision_index:
                raise ValueError("OPD operation_context decision differs from the label")
            for name in ("split", "prompt"):
                if not isinstance(context[name], str) or not context[name]:
                    raise ValueError(f"OPD operation_context {name} is invalid")
            for name in ("seed", "step", "prefix"):
                if type(context[name]) is not int or context[name] < 0:
                    raise ValueError(f"OPD operation_context {name} is invalid")
            _require_sha256(
                context["checkpoint_hash"], label="OPD operation_context checkpoint hash"
            )
        if strict:
            if self.teacher_checkpoint_file_sha256 is None:
                raise ValueError("formal OPD label is missing checkpoint file provenance")
            if self.student_state_sha256 is None:
                raise ValueError("formal OPD label is missing student state provenance")
            if self.reservation_id is None:
                raise ValueError("formal OPD label is missing operation reservation")
            if self.operation_context is None:
                raise ValueError("formal OPD label is missing operation context")

    @classmethod
    def from_target(
        cls,
        decision_index: int,
        target: Tensor,
        *,
        receipt_output_sha256: str,
        frame_valid: bool,
        teacher_checkpoint_sha256: str,
        teacher_checkpoint_file_sha256: str | None = None,
        reservation_id: str | None = None,
        operation_context: Mapping[str, Any] | None = None,
        student_state_sha256: str | None = None,
        teacher_state_sha256: str | None = None,
    ) -> "OpdTeacherLabel":
        """Construct a label while deriving the immutable target hash."""
        return cls(
            decision_index=decision_index,
            target=target,
            target_sha256=tensor_sha256(target.detach()),
            receipt_output_sha256=receipt_output_sha256,
            frame_valid=frame_valid,
            teacher_checkpoint_sha256=teacher_checkpoint_sha256,
            teacher_state_sha256=teacher_state_sha256,
            teacher_checkpoint_file_sha256=teacher_checkpoint_file_sha256,
            reservation_id=reservation_id,
            operation_context=operation_context,
            student_state_sha256=student_state_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe provenance portion (the tensor stays in memory)."""
        self.validate()
        value: dict[str, Any] = {
            "decision_index": self.decision_index,
            "target_sha256": self.target_sha256,
            "raw_target_sha256": self.raw_target_sha256,
            "target_shape": list(self.target.shape),
            "target_dtype": str(self.target.dtype),
            "label_receipt_output_sha256": self.receipt_output_sha256,
            "frame_valid": self.frame_valid,
            "student_state": self.student_state,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "teacher_state_sha256": self.teacher_state_sha256,
        }
        if self.teacher_checkpoint_file_sha256 is not None:
            value["teacher_checkpoint_file_sha256"] = self.teacher_checkpoint_file_sha256
        if self.reservation_id is not None:
            value["reservation_id"] = self.reservation_id
        if self.operation_context is not None:
            value["operation_context"] = _json_copy(
                self.operation_context, label="OPD operation_context"
            )
        if self.student_state_sha256 is not None:
            value["student_state_sha256"] = self.student_state_sha256
        return value


@dataclass(frozen=True)
class ScoredRollout:
    """One detached rollout and its three terminal scalar rewards."""

    collection: RolloutCollection
    rewards: Mapping[str, Tensor]
    preference_label: PreferenceLabelProvenance
    # Formal OPD attaches the already-charged teacher targets here.  Pair arms
    # leave this field as ``None``; direct legacy objective tests may likewise
    # omit it and use the historical recomputation fallback.
    opd_teacher_labels: tuple[OpdTeacherLabel, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.collection, RolloutCollection):
            raise TypeError("collection must be a RolloutCollection")
        if self.collection.preserve_graph:
            raise ValueError("training objectives require a detached rollout")
        if set(self.collection.branches) != {"plus", "minus", "anchor"}:
            raise ValueError("rollout branches must be plus, minus, and anchor")
        if set(self.rewards) != {"plus", "minus", "anchor"}:
            raise ValueError("terminal rewards must cover plus, minus, and anchor")
        for branch, reward in self.rewards.items():
            if (
                not isinstance(reward, Tensor)
                or reward.numel() != 1
                or not reward.is_floating_point()
                or not torch.isfinite(reward).all()
                or reward.requires_grad
            ):
                raise ValueError(
                    f"terminal reward for {branch!r} must be one finite detached tensor"
                )
        if not isinstance(self.preference_label, PreferenceLabelProvenance):
            raise TypeError("scored rollout requires an immutable preference label")
        self.preference_label.validate_rewards(
            float(self.rewards["plus"].detach().double().cpu()),
            float(self.rewards["minus"].detach().double().cpu()),
        )
        decision_indices = tuple(
            proposal.step_index for proposal in self.collection.proposals
        )
        if decision_indices != (8, 24, 40):
            raise ValueError("rollout decisions differ from the registered schedule")
        for branch, trajectory in self.collection.branches.items():
            _validate_trajectory(trajectory, branch=branch)
        if self.opd_teacher_labels is not None:
            labels = tuple(self.opd_teacher_labels)
            if len(labels) != len(OPD_DECISION_INDICES):
                raise ValueError("OPD rollouts must contain three teacher labels")
            anchor_transitions = self.collection.branches["anchor"].transitions
            for expected_index, label, transition in zip(
                OPD_DECISION_INDICES, labels, anchor_transitions
            ):
                if not isinstance(label, OpdTeacherLabel):
                    raise TypeError("OPD teacher labels must be OpdTeacherLabel values")
                label.validate()
                if label.decision_index != expected_index:
                    raise ValueError("OPD teacher labels differ from the decision schedule")
                if (
                    label.target.shape != transition.native_transition.shape
                    or label.target.device != transition.native_transition.device
                    or label.target.dtype != transition.native_transition.dtype
                ):
                    raise ValueError("OPD teacher target does not match its anchor transition")
            object.__setattr__(self, "opd_teacher_labels", labels)

    def preference(self) -> tuple[str, str, bool]:
        self.preference_label.validate_rewards(
            float(self.rewards["plus"].detach().double().cpu()),
            float(self.rewards["minus"].detach().double().cpu()),
        )
        return self.preference_label.preference()

    def signed_advantage(self) -> float:
        self.preference_label.validate_rewards(
            float(self.rewards["plus"].detach().double().cpu()),
            float(self.rewards["minus"].detach().double().cpu()),
        )
        return self.preference_label.signed_advantage


def _validate_renderer(value: Any, *, label: str) -> EulerNativeFrameV1:
    if not isinstance(value, EulerNativeFrameV1):
        raise TypeError(f"{label} must be EulerNativeFrameV1")
    return value


def _validate_renderer_pair(
    renderer: Any, reference_renderer: Any
) -> tuple[EulerNativeFrameV1, EulerNativeFrameV1]:
    renderer = _validate_renderer(renderer, label="renderer")
    reference = _validate_renderer(reference_renderer, label="reference_renderer")
    if renderer is reference:
        raise ValueError("renderer and reference_renderer must be distinct objects")
    if renderer.frame_contract_hash != reference.frame_contract_hash:
        raise ValueError("renderer and reference frame contracts differ")
    if renderer.calibration_hash != reference.calibration_hash:
        raise ValueError("renderer and reference calibrations differ")
    if renderer.contract.to_dict() != reference.contract.to_dict():
        raise ValueError("renderer and reference action contracts differ")
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError("reference_renderer must be frozen")
    return renderer, reference


def _state_tensor_mapping(module: Any, *, label: str) -> Mapping[str, Tensor]:
    """Return a complete state mapping for fail-closed teacher checks."""
    try:
        state = module.state_dict()
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"{label} must expose state_dict()") from exc
    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"{label} state_dict() must be a non-empty mapping")
    if any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise TypeError(f"{label} state_dict() must contain only named tensors")
    return state


def _tensor_storage_pointer(value: Tensor) -> int | None:
    """Get a storage identity without retaining a view of the tensor."""
    if value.numel() == 0:
        return None
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:  # pragma: no cover - compatibility with old torch
        return int(value.storage().data_ptr())


def _validate_opd_teacher(
    renderer: EulerNativeFrameV1,
    teacher: EulerNativeFrameV1,
    reference: EulerNativeFrameV1,
) -> None:
    """Validate the external OPD teacher at the objective boundary.

    Runner-time authentication is necessary for provenance, but callers can
    invoke this tensor objective directly.  Keep the safety checks here too so
    a teacher cannot silently become a trainable, aliased, or differently
    calibrated target network between collection and optimization.
    """
    if teacher is renderer or teacher is reference:
        raise ValueError("teacher_renderer must be a distinct frozen snapshot")
    if any(module.training for module in teacher.modules()):
        raise ValueError("teacher_renderer must remain in eval mode")
    teacher_parameters = tuple(teacher.parameters())
    if not teacher_parameters:
        raise ValueError("teacher_renderer must expose parameters")
    if any(parameter.requires_grad for parameter in teacher_parameters):
        raise ValueError("teacher_renderer must be frozen")

    # Frame hash alone is not enough: explicitly compare the underlying
    # calibration and action-space payloads so a malformed implementation that
    # reports a stale hash cannot pass this boundary.
    if teacher.calibration_hash != renderer.calibration_hash:
        raise ValueError("teacher and student calibration hashes differ")
    if teacher.calibration_hash != reference.calibration_hash:
        raise ValueError("teacher and reference calibration hashes differ")

    def contract_payload(value: Any, *, label: str) -> Mapping[str, Any]:
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(f"{label} action contract must expose to_dict()")
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError(f"{label} action contract payload must be a mapping")
        return payload

    student_contract = contract_payload(renderer.contract, label="student")
    teacher_contract = contract_payload(teacher.contract, label="teacher")
    reference_contract = contract_payload(reference.contract, label="reference")
    if teacher_contract != student_contract:
        raise ValueError("teacher and student action contracts differ")
    if teacher_contract != reference_contract:
        raise ValueError("teacher and reference action contracts differ")
    if teacher.frame_contract_hash != renderer.frame_contract_hash:
        raise ValueError("teacher and student frame contracts differ")
    if teacher.frame_contract_hash != reference.frame_contract_hash:
        raise ValueError("teacher and reference frame contracts differ")

    student_state = _state_tensor_mapping(renderer, label="renderer")
    teacher_state = _state_tensor_mapping(teacher, label="teacher_renderer")
    reference_state = _state_tensor_mapping(reference, label="reference_renderer")
    if set(teacher_state) != set(student_state) or set(reference_state) != set(student_state):
        raise ValueError("OPD renderer state schemas differ")
    for name, teacher_tensor in teacher_state.items():
        student_tensor = student_state[name]
        reference_tensor = reference_state[name]
        if (
            teacher_tensor.shape != student_tensor.shape
            or teacher_tensor.dtype != student_tensor.dtype
            or teacher_tensor.device != student_tensor.device
            or teacher_tensor.shape != reference_tensor.shape
            or teacher_tensor.dtype != reference_tensor.dtype
            or teacher_tensor.device != reference_tensor.device
        ):
            raise ValueError("OPD teacher state tensors do not match renderer state")
        for other in (student_tensor, reference_tensor):
            if teacher_tensor is other:
                raise ValueError("teacher_renderer state aliases another renderer")
            teacher_ptr = _tensor_storage_pointer(teacher_tensor)
            other_ptr = _tensor_storage_pointer(other)
            if teacher_ptr is not None and teacher_ptr == other_ptr:
                raise ValueError("teacher_renderer state storage aliases another renderer")

    if tuple(teacher.active_mask) != tuple(renderer.active_mask):
        raise ValueError("teacher and student active masks differ")
    if tuple(teacher.active_mask) != tuple(reference.active_mask):
        raise ValueError("teacher and reference active masks differ")


def _validate_trajectory(trajectory: Any, *, branch: str) -> None:
    if not isinstance(trajectory, BranchTrajectory) or trajectory.branch_id != branch:
        raise ValueError(f"invalid {branch!r} branch trajectory")
    if len(trajectory.transitions) != 3:
        raise ValueError("every branch must contain exactly three decisions")
    if tuple(item.step_index for item in trajectory.transitions) != (8, 24, 40):
        raise ValueError("branch decisions differ from the registered schedule")
    for transition in trajectory.transitions:
        if not isinstance(transition, Transition):
            raise TypeError("branch contains a non-Transition decision")
        if not isinstance(transition.state, EulerFrameState):
            raise TypeError("decision state must be EulerFrameState")
        for name in ("action", "native_transition", "rendered_transition"):
            value = getattr(transition, name)
            if not isinstance(value, Tensor) or value.requires_grad:
                raise ValueError(f"recorded transition {name} must be detached")
        if transition.pre_squash is None or transition.behavior_mean is None:
            raise ValueError("formal decisions must retain pre-squash and behavior means")
        if transition.pre_squash.requires_grad or transition.behavior_mean.requires_grad:
            raise ValueError("recorded policy coordinates must be detached")


def _rollouts(values: Iterable[ScoredRollout]) -> tuple[ScoredRollout, ...]:
    result = tuple(values)
    if not result or any(not isinstance(value, ScoredRollout) for value in result):
        raise ValueError("at least one ScoredRollout is required")
    return result


def _policy_transition(
    renderer: EulerNativeFrameV1, transition: Transition
) -> tuple[Tensor, Tensor, Tensor]:
    """Return exact Euler update, pre-squash mean, and row validity."""
    state = transition.state
    if not isinstance(state, EulerFrameState):
        raise TypeError("transition state must be EulerFrameState")
    mean = renderer.action_parameters(state)
    action = renderer.deterministic_action_from_mean(mean)
    rendered = renderer.apply_coefficients(state, action)
    kappa = state.kappa.to(
        device=rendered.residual.device, dtype=rendered.residual.dtype
    ).reshape(-1, 1, 1, 1)
    update = state.native_update.to(rendered.residual.dtype) + kappa * rendered.residual
    valid = state.diagnostics.valid & rendered.diagnostics.valid
    return update, mean, valid


def _reference_mean(
    reference: EulerNativeFrameV1, transition: Transition
) -> Tensor:
    state = transition.state
    if not isinstance(state, EulerFrameState):
        raise TypeError("transition state must be EulerFrameState")
    with torch.no_grad():
        mean = reference.action_parameters(state).detach()
    recorded = transition.reference_mean
    if recorded is None:
        raise ValueError("rollout omitted the frozen reference mean")
    if recorded.shape != mean.shape or not torch.equal(recorded.detach(), mean):
        raise ValueError("recorded reference mean differs from the frozen reference")
    return mean


def _flatten_transitions(
    scored: Sequence[ScoredRollout], branches: Sequence[str]
) -> tuple[Transition, ...]:
    if len(scored) != len(branches):
        raise ValueError("one branch selection is required per rollout")
    rows: list[Transition] = []
    for item, branch in zip(scored, branches):
        if branch not in item.collection.branches:
            raise ValueError(f"rollout has no branch {branch!r}")
        rows.extend(item.collection.branches[branch].transitions)
    return tuple(rows)


def _optional_state_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _validate_opd_label_binding(
    label: OpdTeacherLabel,
    transition: Transition,
    *,
    strict: bool,
    teacher_checkpoint_hash: str | None,
    teacher_checkpoint_file_hash: str | None,
) -> None:
    """Bind one charged target to the exact student transition it supervises."""
    label.validate(strict=strict)
    if label.decision_index != transition.step_index:
        raise ValueError("OPD teacher label decision differs from the anchor transition")
    if teacher_checkpoint_hash is not None and (
        label.teacher_checkpoint_sha256 != teacher_checkpoint_hash
        or label.teacher_state_sha256 != teacher_checkpoint_hash
    ):
        raise ValueError("OPD teacher label is from a different teacher state")
    if teacher_checkpoint_file_hash is not None and (
        label.teacher_checkpoint_file_sha256 != teacher_checkpoint_file_hash
    ):
        raise ValueError("OPD teacher label is from a different checkpoint file")
    expected_state_hash = opd_anchor_state_sha256(transition)
    if label.student_state_sha256 is not None and (
        label.student_state_sha256 != expected_state_hash
    ):
        raise ValueError("OPD teacher label is bound to a different student anchor state")
    if strict and label.student_state_sha256 != expected_state_hash:
        raise ValueError("formal OPD label is missing the exact student anchor binding")

    context = label.operation_context
    if context is None:
        if strict:
            raise ValueError("formal OPD label is missing operation context")
        return
    action = context.get("action") if isinstance(context, Mapping) else None
    if isinstance(action, Mapping):
        if action.get("student_state_sha256") is not None and action.get(
            "student_state_sha256"
        ) != expected_state_hash:
            raise ValueError("OPD operation context is bound to a different anchor state")
        if action.get("raw_target_sha256") is not None and action.get(
            "raw_target_sha256"
        ) != label.target_sha256:
            raise ValueError("OPD operation context raw target differs from the label")

    # SdxlEulerState is intentionally duck-typed here to keep this module
    # framework-neutral.  Formal SDXL states expose all three identity fields;
    # older synthetic unit-test states may omit them.
    next_state = transition.next_state
    conditioning = _optional_state_field(next_state, "conditioning")
    prompt_ids = _optional_state_field(conditioning, "prompt_ids")
    seeds = _optional_state_field(conditioning, "generation_seeds")
    actual_split = _optional_state_field(next_state, "split")
    if prompt_ids is not None:
        if not isinstance(prompt_ids, (tuple, list)) or not prompt_ids:
            raise ValueError("OPD anchor next state has invalid prompt identity")
        if prompt_ids[0] != context["prompt"]:
            raise ValueError("OPD operation context prompt differs from the anchor state")
    if seeds is not None:
        if not isinstance(seeds, (tuple, list)) or not seeds:
            raise ValueError("OPD anchor next state has invalid seed identity")
        if seeds[0] != context["seed"]:
            raise ValueError("OPD operation context seed differs from the anchor state")
    if actual_split is not None and actual_split != context["split"]:
        raise ValueError("OPD operation context split differs from the anchor state")
    if isinstance(action, Mapping):
        expected_action_fields = {
            "prompt_id": context["prompt"],
            "split": context["split"],
            "seed": context["seed"],
        }
        for name, expected in expected_action_fields.items():
            if name in action and action[name] != expected:
                raise ValueError(
                    f"OPD operation context action {name} differs from the anchor state"
                )
        if "prompt" in action:
            prompts = _optional_state_field(conditioning, "prompts")
            if prompts is not None and (not prompts or prompts[0] != action["prompt"]):
                raise ValueError(
                    "OPD operation context prompt text differs from the anchor state"
                )


def opd_objective(
    renderer: EulerNativeFrameV1,
    teacher_renderer: EulerNativeFrameV1,
    reference_renderer: EulerNativeFrameV1,
    rollouts: Iterable[ScoredRollout],
    *,
    require_stored_labels: bool = False,
    teacher_checkpoint_hash: str | None = None,
    teacher_checkpoint_file_hash: str | None = None,
) -> Tensor:
    """External-teacher OPD on states visited by the current student.

    Formal runner calls require the detached targets that were produced and
    charged during collection.  The optional recomputation path is retained
    only for older, non-formal callers that construct a ``ScoredRollout``
    without labels.
    """
    if type(require_stored_labels) is not bool:
        raise TypeError("require_stored_labels must be boolean")
    if teacher_checkpoint_hash is not None:
        _require_sha256(
            teacher_checkpoint_hash, label="OPD teacher checkpoint hash"
        )
    if teacher_checkpoint_file_hash is not None:
        _require_sha256(
            teacher_checkpoint_file_hash,
            label="OPD teacher checkpoint file hash",
        )
    if require_stored_labels and (
        teacher_checkpoint_hash is None or teacher_checkpoint_file_hash is None
    ):
        raise ValueError(
            "formal OPD requires both teacher state and checkpoint file hashes"
        )
    renderer, reference = _validate_renderer_pair(renderer, reference_renderer)
    teacher = _validate_renderer(teacher_renderer, label="teacher_renderer")
    _validate_opd_teacher(renderer, teacher, reference)
    scored = _rollouts(rollouts)
    if teacher_checkpoint_hash is not None:
        actual_hash = module_state_sha256(teacher)
        if actual_hash != teacher_checkpoint_hash:
            raise ValueError("OPD teacher state changed after label collection")
    # OPD follows the deterministic student branch.  These are precisely the
    # states visited by the behavior mean, never states generated by teacher.
    transitions = _flatten_transitions(scored, ("anchor",) * len(scored))
    stored_labels: list[OpdTeacherLabel | None] = []
    for item in scored:
        labels = item.opd_teacher_labels
        if labels is None:
            if require_stored_labels:
                raise ValueError("formal OPD rollout is missing charged teacher labels")
            stored_labels.extend([None] * len(OPD_DECISION_INDICES))
            continue
        if len(labels) != len(OPD_DECISION_INDICES):
            raise ValueError("OPD rollout must contain three teacher labels")
        item_transitions = item.collection.branches["anchor"].transitions
        for expected_index, label, transition in zip(
            OPD_DECISION_INDICES, labels, item_transitions
        ):
            if label.decision_index != expected_index:
                raise ValueError("OPD teacher label decision differs from the rollout")
            _validate_opd_label_binding(
                label,
                transition,
                strict=require_stored_labels,
                teacher_checkpoint_hash=teacher_checkpoint_hash,
                teacher_checkpoint_file_hash=teacher_checkpoint_file_hash,
            )
        stored_labels.extend(labels)
    predicted: list[Tensor] = []
    targets: list[Tensor] = []
    nominal: list[Tensor] = []
    means: list[Tensor] = []
    references: list[Tensor] = []
    valid: list[Tensor] = []
    for transition, label in zip(transitions, stored_labels):
        update, mean, row_valid = _policy_transition(renderer, transition)
        if label is None:
            # Compatibility for informal historical callers.  A formal run
            # always reaches the branch above and therefore never performs an
            # uncharged teacher recomputation in its optimizer objective.
            with torch.no_grad():
                target, _teacher_mean, teacher_valid = _policy_transition(
                    teacher, transition
                )
        else:
            target = label.target
            if (
                target.shape != update.shape
                or target.device != update.device
                or target.dtype != update.dtype
            ):
                raise ValueError("OPD teacher target does not match the predicted transition")
            teacher_valid = torch.full(
                (target.shape[0],),
                label.frame_valid,
                dtype=torch.bool,
                device=target.device,
            )
        predicted.append(update)
        targets.append(target.detach())
        nominal.append(transition.native_transition.detach())
        means.append(mean)
        references.append(_reference_mean(reference, transition))
        valid.append(row_valid & teacher_valid)
    return opd_loss(
        torch.cat(predicted),
        torch.cat(targets),
        torch.cat(nominal),
        torch.cat(means),
        torch.cat(references),
        renderer.contract,
        anchor_weight=0.10,
        valid_mask=torch.cat(valid),
    )


def search_distill_objective(
    renderer: EulerNativeFrameV1,
    reference_renderer: EulerNativeFrameV1,
    rollouts: Iterable[ScoredRollout],
) -> Tensor:
    """Regress native transitions from the better sampled terminal branch."""
    renderer, reference = _validate_renderer_pair(renderer, reference_renderer)
    scored = _rollouts(rollouts)
    selections = [item.preference() for item in scored]
    branches = [chosen for chosen, _rejected, _tie in selections]
    transitions = _flatten_transitions(scored, branches)
    predicted: list[Tensor] = []
    target: list[Tensor] = []
    nominal: list[Tensor] = []
    means: list[Tensor] = []
    references: list[Tensor] = []
    valid: list[Tensor] = []
    for transition in transitions:
        update, mean, row_valid = _policy_transition(renderer, transition)
        predicted.append(update)
        target.append(transition.rendered_transition.detach())
        nominal.append(transition.native_transition.detach())
        means.append(mean)
        references.append(_reference_mean(reference, transition))
        valid.append(row_valid)
    decision_weights = torch.cat(
        [
            torch.zeros(3, device=predicted[0].device)
            if tie
            else torch.ones(3, device=predicted[0].device)
            for _chosen, _rejected, tie in selections
        ]
    )
    return search_distill_loss(
        torch.cat(predicted),
        torch.cat(target),
        torch.cat(nominal),
        torch.cat(means),
        torch.cat(references),
        renderer.contract,
        chosen_weight=decision_weights,
        valid_mask=torch.cat(valid),
    )


def _trajectory_policy_rows(
    renderer: EulerNativeFrameV1,
    reference: EulerNativeFrameV1,
    trajectory: BranchTrajectory,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    actions: list[Tensor] = []
    pre_squash: list[Tensor] = []
    behavior: list[Tensor] = []
    policy: list[Tensor] = []
    reference_means: list[Tensor] = []
    for transition in trajectory.transitions:
        actions.append(transition.action.detach())
        if transition.pre_squash is None or transition.behavior_mean is None:
            raise ValueError("formal rollout omitted policy coordinates")
        pre_squash.append(transition.pre_squash.detach())
        behavior.append(transition.behavior_mean.detach())
        policy.append(renderer.action_parameters(transition.state))
        reference_means.append(_reference_mean(reference, transition))
    return tuple(
        torch.stack(values, dim=1)
        for values in (actions, pre_squash, behavior, policy, reference_means)
    )


def dpo_objective(
    renderer: EulerNativeFrameV1,
    reference_renderer: EulerNativeFrameV1,
    rollouts: Iterable[ScoredRollout],
) -> Tensor:
    """One reference-relative DPO margin per three-decision trajectory."""
    renderer, reference = _validate_renderer_pair(renderer, reference_renderer)
    scored = _rollouts(rollouts)
    chosen_rows = []
    rejected_rows = []
    ties = []
    for item in scored:
        chosen, rejected, tie = item.preference()
        # A tie has zero loss weight.  Use plus/minus for density shape rather
        # than the deterministic anchor, whose sample is not stochastic data.
        if tie:
            chosen, rejected = PAIR_BRANCHES
        chosen_rows.append(
            _trajectory_policy_rows(
                renderer, reference, item.collection.branches[chosen]
            )
        )
        rejected_rows.append(
            _trajectory_policy_rows(
                renderer, reference, item.collection.branches[rejected]
            )
        )
        ties.append(tie)

    def combine(rows: Sequence[tuple[Tensor, ...]], index: int) -> Tensor:
        return torch.cat([row[index] for row in rows], dim=0)

    return dpo_loss(
        combine(chosen_rows, 0),
        combine(rejected_rows, 0),
        combine(chosen_rows, 3),
        combine(rejected_rows, 3),
        combine(chosen_rows, 4),
        combine(rejected_rows, 4),
        renderer.contract,
        beta=0.10,
        tie_mask=torch.tensor(ties, device=combine(chosen_rows, 0).device),
        chosen_pre_squash=combine(chosen_rows, 1),
        rejected_pre_squash=combine(rejected_rows, 1),
        require_pre_squash=True,
    )


def rl_objective(
    renderer: EulerNativeFrameV1,
    reference_renderer: EulerNativeFrameV1,
    rollouts: Iterable[ScoredRollout],
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference-anchored DDPO loss over both antithetic branches."""
    renderer, reference = _validate_renderer_pair(renderer, reference_renderer)
    scored = _rollouts(rollouts)
    rows = []
    advantages = []
    for item in scored:
        value = item.signed_advantage()
        for branch, advantage in (("plus", value), ("minus", -value)):
            row = _trajectory_policy_rows(
                renderer, reference, item.collection.branches[branch]
            )
            rows.append(row)
            advantages.append(
                torch.full(
                    row[0].shape[:-1],
                    float(advantage),
                    device=row[0].device,
                    dtype=row[0].dtype,
                )
            )

    def combine(index: int) -> Tensor:
        return torch.cat([row[index] for row in rows], dim=0)

    return per_decision_rl_loss(
        combine(0),
        combine(2),
        combine(3),
        combine(4),
        torch.cat(advantages, dim=0),
        renderer.contract,
        clip_low=0.8,
        clip_high=1.2,
        kl_weight=0.01,
        pre_squash=combine(1),
        require_pre_squash=True,
    )


__all__ = [
    "OPD_DECISION_INDICES",
    "PAIR_BRANCHES",
    "OpdTeacherLabel",
    "ScoredRollout",
    "TIE_EPSILON",
    "dpo_objective",
    "opd_anchor_state_sha256",
    "opd_objective",
    "rl_objective",
    "search_distill_objective",
]
