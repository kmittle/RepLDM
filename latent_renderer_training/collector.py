"""A contract-bound Euler collector for latent-renderer training.

The production inference pipeline stays under ``no_grad``.  Training callers
provide two small callbacks instead: ``observe_fn`` performs one frozen U-Net
observation and ``transition_fn`` performs one scheduler transition.  The
collector owns the renderer action mapping, exact transformed-Gaussian sample,
shared-prefix branching, and the distinction between decision and auxiliary
rows.  This keeps OPD, renderer-DPO, and RL on one physical trajectory
implementation without copying the SDXL pipeline into the training package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from AttentionGuidance.latent_renderer import RendererCondition, RendererObservation

from .renderer import EulerFrameState, EulerNativeFrameV1
from .rollout import (
    BranchTrajectory,
    DecisionProposal,
    RolloutCollection,
    Transition,
    _clone_value,
)
from .run_contract import RUN_CONTRACT_SCHEMA, TrainingRunContract


COLLECTOR_ROUND_TRIP_RTOL = 2e-3
COLLECTOR_ROUND_TRIP_ATOL = 3e-5
F0_TOTAL_STEPS = 50
F0_DECISION_INDICES = (8, 24, 40)


@dataclass(frozen=True)
class EulerRolloutStep:
    """Detached output of one frozen U-Net/CFG observation.

    ``native_prev_sample`` is the result of the native, zero-churn scheduler
    step for ``native_model_output``.  The transition callback is still called
    for the actual step so a real scheduler can remain the single source of
    deployed bytes.  All observation tensors must be detached: gradients are
    allowed only through the renderer action and the explicitly differentiable
    scheduler/decoder path after this record is created.
    """

    observation: RendererObservation
    condition: RendererCondition
    native_model_output: Tensor
    native_prev_sample: Tensor
    sigma_from: Tensor | float
    sigma_to: Tensor | float
    prediction_type: str = "epsilon"
    clean_update_gain: Optional[Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, RendererObservation):
            raise TypeError("observation must be a RendererObservation")
        if not isinstance(self.condition, RendererCondition):
            raise TypeError("condition must be a RendererCondition")
        current = self.observation.latents_before_step
        if not isinstance(current, Tensor) or current.ndim != 4:
            raise ValueError("observation latents must be NCHW")
        for name, value in (
            ("native_model_output", self.native_model_output),
            ("native_prev_sample", self.native_prev_sample),
        ):
            if not isinstance(value, Tensor) or value.shape != current.shape:
                raise ValueError(f"{name} must match observation latents")
            if value.device != current.device or not value.is_floating_point():
                raise ValueError(f"{name} must share the latent device and use floating point")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            if value.requires_grad:
                raise ValueError(f"{name} must be detached from the frozen backbone")
        for name, value in (
            ("prompt_embedding", self.condition.prompt_embedding),
            ("state_features", self.condition.state_features),
        ):
            if value is not None:
                if not isinstance(value, Tensor) or not torch.isfinite(value).all():
                    raise ValueError(f"{name} must be finite when supplied")
                if value.requires_grad:
                    raise ValueError(f"{name} must be detached from the frozen backbone")
        if current.requires_grad:
            raise ValueError("observation latents must be detached from the frozen backbone")
        for name, value in (
            ("pred_original_sample", self.observation.pred_original_sample),
            ("scheduler_update", self.observation.scheduler_update),
            ("bases", self.condition.bases),
        ):
            if not isinstance(value, Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            if value.requires_grad:
                raise ValueError(f"{name} must be detached from the frozen backbone")
        if self.observation.pred_original_sample.shape != current.shape:
            raise ValueError("pred_original_sample must match observation latents")
        if self.observation.scheduler_update.shape != current.shape:
            raise ValueError("scheduler_update must match observation latents")
        if self.condition.bases.ndim != 5 or self.condition.bases.shape[0] != current.shape[0]:
            raise ValueError("condition bases must have shape (batch, slots, channels, height, width)")
        if self.condition.bases.shape[2:] != current.shape[1:]:
            raise ValueError("condition bases do not match observation latents")
        expected_update = self.native_prev_sample - current
        # The native step is supplied by the same scheduler snapshot as the
        # observation.  A small tolerance accommodates reduced-precision
        # scheduler output while rejecting stale or unrelated transitions.
        if not torch.allclose(
            self.observation.scheduler_update.float(), expected_update.float(),
            rtol=COLLECTOR_ROUND_TRIP_RTOL,
            atol=COLLECTOR_ROUND_TRIP_ATOL,
        ):
            raise ValueError("observation scheduler_update disagrees with native_prev_sample")
        if self.prediction_type not in {"epsilon", "sample", "original_sample", "v_prediction"}:
            raise ValueError(f"unsupported prediction_type {self.prediction_type!r}")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")


@dataclass(frozen=True)
class EulerTransitionResult:
    """Result returned by one real scheduler transition callback."""

    state: Any
    latent: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.latent, Tensor) or self.latent.ndim != 4:
            raise ValueError("transition latent must be an NCHW tensor")
        if not self.latent.is_floating_point() or not torch.isfinite(self.latent).all():
            raise ValueError("transition latent must be finite and floating point")


@dataclass(frozen=True)
class CollectorStats:
    """Callback and verified physical-operation counts for one collection."""

    observe_calls: int
    transition_calls: int
    decision_transitions: int
    auxiliary_transitions: int
    verified_unet_forwards: Optional[int] = None


@dataclass(frozen=True)
class TeacherRolloutResult:
    """One deterministic external-teacher branch sharing the student prefix."""

    branch: BranchTrajectory
    terminal_state: Any
    stats: CollectorStats

    def __post_init__(self) -> None:
        if not isinstance(self.branch, BranchTrajectory):
            raise TypeError("teacher branch must be a BranchTrajectory")
        if self.branch.branch_id != "teacher":
            raise ValueError("teacher branch must use the reserved teacher ID")
        if len(self.branch.transitions) != 3:
            raise ValueError("teacher branch must contain exactly three decisions")
        if tuple(item.step_index for item in self.branch.transitions) != F0_DECISION_INDICES:
            raise ValueError("teacher branch decisions differ from the registered schedule")
        if not isinstance(self.stats, CollectorStats):
            raise TypeError("teacher branch stats must be CollectorStats")


@dataclass(frozen=True)
class CachedEulerDecision:
    """Detached F0 decision snapshot that can seed independent suffixes."""

    decision_index: int
    state: Any
    step: EulerRolloutStep
    frame: EulerFrameState
    checkpoint_hash: Optional[str]
    frame_contract_hash: str
    calibration_hash: str
    run_contract: Optional[str]

    def __post_init__(self) -> None:
        if self.decision_index not in F0_DECISION_INDICES:
            raise ValueError("cached decision is outside the registered F0 schedule")
        if not isinstance(self.step, EulerRolloutStep):
            raise TypeError("cached step must be EulerRolloutStep")
        if not isinstance(self.frame, EulerFrameState):
            raise TypeError("cached frame must be EulerFrameState")
        if self.step.observation.step_index != self.decision_index:
            raise ValueError("cached observation has a mismatched decision index")
        for name in ("frame_contract_hash", "calibration_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.checkpoint_hash is not None and not isinstance(self.checkpoint_hash, str):
            raise TypeError("checkpoint_hash must be a string or None")
        if self.run_contract is not None and not isinstance(self.run_contract, str):
            raise TypeError("run_contract must be a string or None")


@dataclass(frozen=True)
class F0AnchorTrace:
    """One strict-C0 trajectory and its three reusable decision snapshots."""

    branch: str
    initial_state: Any
    terminal_state: Any
    cached_decisions: Mapping[int, CachedEulerDecision]
    transitions: tuple[Transition, ...]
    stats: CollectorStats
    checkpoint_hash: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.branch, str) or not self.branch:
            raise ValueError("anchor branch must be a non-empty string")
        cache = dict(self.cached_decisions)
        if set(cache) != set(F0_DECISION_INDICES):
            raise ValueError("F0 anchor cache must contain decisions 8, 24, and 40")
        if any(key != item.decision_index for key, item in cache.items()):
            raise ValueError("F0 anchor cache key differs from its decision record")
        # Keep the cache serializable for checkpoint/restart handoff.  The
        # decision records themselves are frozen and replay revalidates every
        # provenance field before consuming one.
        object.__setattr__(self, "cached_decisions", cache)
        object.__setattr__(self, "transitions", tuple(self.transitions))
        if len(self.transitions) != F0_TOTAL_STEPS:
            raise ValueError("F0 anchor trace must contain exactly 50 transitions")
        if self.stats.observe_calls != 50 or self.stats.transition_calls != 50:
            raise ValueError("F0 anchor trace must use exactly 50 observations and transitions")
        if self.stats.decision_transitions != 3 or self.stats.auxiliary_transitions != 47:
            raise ValueError("F0 anchor transition classes do not match the frozen schedule")

    @property
    def cache(self) -> Mapping[int, CachedEulerDecision]:
        """Short alias for callers that only need the replay cache."""

        return self.cached_decisions


@dataclass(frozen=True)
class CachedSuffixResult:
    """Auditable result of one action followed by a native F0 suffix."""

    decision_index: int
    branch: str
    initial_state: Any
    terminal_state: Any
    requested_action: Tensor
    applied_action: Tensor
    frame_valid: Tensor
    transitions: tuple[Transition, ...]
    stats: CollectorStats
    checkpoint_hash: Optional[str]

    def __post_init__(self) -> None:
        if self.decision_index not in F0_DECISION_INDICES:
            raise ValueError("suffix decision is outside the registered F0 schedule")
        if not isinstance(self.branch, str) or not self.branch:
            raise ValueError("suffix branch must be a non-empty string")
        if (
            not isinstance(self.requested_action, Tensor)
            or not isinstance(self.applied_action, Tensor)
            or self.requested_action.shape != self.applied_action.shape
        ):
            raise ValueError("requested and applied actions must be matching tensors")
        if (
            not isinstance(self.frame_valid, Tensor)
            or self.frame_valid.dtype != torch.bool
            or self.frame_valid.shape != self.requested_action.shape[:1]
        ):
            raise ValueError("frame_valid must contain one boolean per batch row")
        object.__setattr__(self, "transitions", tuple(self.transitions))
        expected_observations = F0_TOTAL_STEPS - self.decision_index - 1
        expected_transitions = F0_TOTAL_STEPS - self.decision_index
        expected_decisions = sum(
            index >= self.decision_index for index in F0_DECISION_INDICES
        )
        if len(self.transitions) != expected_transitions:
            raise ValueError("cached suffix has the wrong transition count")
        if (
            self.stats.observe_calls != expected_observations
            or self.stats.transition_calls != expected_transitions
            or self.stats.decision_transitions != expected_decisions
            or self.stats.auxiliary_transitions
            != expected_transitions - expected_decisions
        ):
            raise ValueError("cached suffix statistics differ from the frozen budget")

    @property
    def fallback_mask(self) -> Tensor:
        """Rows forced to native behavior because their cached frame was invalid."""

        return ~self.frame_valid

    @property
    def selected_transition(self) -> Transition:
        return self.transitions[0]


def _clone_state(value: Any, *, preserve_graph: bool, clone_fn: Optional[Callable[[Any], Any]]) -> Any:
    if clone_fn is not None:
        return clone_fn(value)
    return _clone_value(value, detach=not preserve_graph)


class EulerNativeRolloutCollector:
    """Collect the exact shared-prefix branches used by all training arms.

    ``observe_fn(state, step_index)`` must run the frozen U-Net/CFG path and
    return :class:`EulerRolloutStep`.  ``transition_fn(state, step_index,
    model_output, step)`` must call the real scheduler exactly once and return
    :class:`EulerTransitionResult`.  When ``physical_unet`` is supplied, the
    collector temporarily hooks that exact module around every observation and
    rejects zero or multiple successful forwards.  Formal collectors require
    this physical proof; informal CPU fixtures may omit it and expose ``None``
    in :attr:`last_stats` for the unverified count.
    """

    def __init__(
        self,
        renderer: EulerNativeFrameV1,
        *,
        decision_indices: Sequence[int],
        total_steps: int,
        registered_decision_indices: Sequence[int],
        adapter: Any = None,
        observe_fn: Optional[Callable[[Any, int], EulerRolloutStep]] = None,
        transition_fn: Optional[
            Callable[[Any, int, Tensor, EulerRolloutStep], EulerTransitionResult]
        ] = None,
        physical_unet: Optional[nn.Module] = None,
        branch_ids: tuple[str, str] = ("plus", "minus"),
        reference_renderer: Optional[EulerNativeFrameV1] = None,
        state_clone_fn: Optional[Callable[[Any], Any]] = None,
        preserve_graph: bool = False,
        round_trip_rtol: float = COLLECTOR_ROUND_TRIP_RTOL,
        round_trip_atol: float = COLLECTOR_ROUND_TRIP_ATOL,
        run_contract: Any = None,
        authorization_binding: Any = None,
        authorization: Any = None,
        require_authorization: bool = False,
    ) -> None:
        if not isinstance(renderer, EulerNativeFrameV1):
            raise TypeError("renderer must be EulerNativeFrameV1")
        if authorization_binding is not None and authorization is not None:
            raise TypeError("provide authorization_binding or authorization, not both")
        binding = authorization_binding if authorization_binding is not None else authorization
        if binding is not None:
            from .authorization import require_authorization_binding
            binding = require_authorization_binding(binding)
            if run_contract is not None:
                supplied = (
                    run_contract.sha256
                    if isinstance(run_contract, TrainingRunContract)
                    else TrainingRunContract.from_mapping(run_contract).sha256
                    if isinstance(run_contract, Mapping)
                    else str(run_contract)
                )
                if supplied != binding.contract_hash:
                    raise ValueError("collector run contract differs from authorization binding")
            # Validate before freezing or mutating a reference renderer.
            binding.validate_current(component=renderer)
        elif isinstance(run_contract, TrainingRunContract) or (
            isinstance(run_contract, Mapping)
            and run_contract.get("schema") == RUN_CONTRACT_SCHEMA
        ):
            raise RuntimeError(
                "a formal training run contract requires a validated authorization binding"
            )
        elif require_authorization:
            raise RuntimeError(
                "formal rollout collection requires a validated authorization binding"
            )
        self.authorization_binding = binding
        self.require_authorization = bool(require_authorization or binding is not None)
        if adapter is not None:
            from .sdxl_adapter import SdxlEulerTrainingAdapter

            if not isinstance(adapter, SdxlEulerTrainingAdapter):
                raise TypeError("adapter must be SdxlEulerTrainingAdapter")
            if observe_fn is not None or transition_fn is not None:
                raise ValueError("adapter cannot be combined with rollout callbacks")
            if physical_unet is not None and physical_unet is not adapter.unet:
                raise ValueError("physical_unet differs from the adapter U-Net")
            if state_clone_fn is not None:
                raise ValueError("adapter owns branch-safe state cloning")
            if adapter.total_steps != total_steps:
                raise ValueError("adapter total_steps differs from the collector")
            physical_unet = adapter.unet
            observe_fn = adapter.observe
            transition_fn = adapter.transition
        elif self.require_authorization:
            raise RuntimeError(
                "formal rollout collection requires SdxlEulerTrainingAdapter"
            )
        if self.require_authorization and physical_unet is None:
            raise RuntimeError(
                "formal rollout collection requires a physical_unet for forward verification"
            )
        self.run_contract = binding.contract_hash if binding is not None else (
            str(run_contract) if run_contract is not None else None
        )
        if reference_renderer is not None and not isinstance(reference_renderer, EulerNativeFrameV1):
            raise TypeError("reference_renderer must be EulerNativeFrameV1")
        if reference_renderer is renderer:
            raise ValueError("reference_renderer must be a distinct frozen snapshot")
        if reference_renderer is not None:
            if reference_renderer.frame_contract_hash != renderer.frame_contract_hash:
                raise ValueError("reference renderer frame contract does not match student")
            if reference_renderer.calibration_hash != renderer.calibration_hash:
                raise ValueError("reference renderer calibration does not match student")
            if reference_renderer.contract.to_dict() != renderer.contract.to_dict():
                raise ValueError("reference renderer action contract does not match student")
            # A reference is an inference-only snapshot.  Make the boundary
            # explicit even when the caller forgot to disable its gradients.
            for parameter in reference_renderer.parameters():
                parameter.requires_grad_(False)
            reference_renderer.eval()
        if len(branch_ids) != 2 or branch_ids[0] == branch_ids[1] or "anchor" in branch_ids:
            raise ValueError("branch_ids must contain two distinct non-anchor names")
        if isinstance(total_steps, bool) or type(total_steps) is not int or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        indices = tuple(decision_indices)
        registered = tuple(registered_decision_indices)
        if not indices or any(type(i) is not int for i in indices):
            raise ValueError("decision_indices must contain plain integers")
        if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
            raise ValueError("decision_indices must be strictly increasing")
        if any(i < 0 or i >= total_steps for i in indices):
            raise ValueError("decision_indices must be inside the trajectory")
        if not registered or any(type(i) is not int for i in registered) or indices != registered:
            raise ValueError("decision_indices differ from the registered schedule")
        if not callable(observe_fn) or not callable(transition_fn):
            raise TypeError("observe_fn and transition_fn must be callable")
        if physical_unet is not None and not isinstance(physical_unet, nn.Module):
            raise TypeError("physical_unet must be a torch.nn.Module or None")
        if (
            isinstance(round_trip_rtol, bool)
            or not isinstance(round_trip_rtol, (int, float))
            or not torch.isfinite(torch.tensor(float(round_trip_rtol)))
            or round_trip_rtol != COLLECTOR_ROUND_TRIP_RTOL
            or isinstance(round_trip_atol, bool)
            or not isinstance(round_trip_atol, (int, float))
            or not torch.isfinite(torch.tensor(float(round_trip_atol)))
            or round_trip_atol != COLLECTOR_ROUND_TRIP_ATOL
        ):
            raise ValueError("round-trip tolerances must use the registered constants")
        self.renderer = renderer
        self.adapter = adapter
        self.reference_renderer = reference_renderer
        self.decision_indices = indices
        self.total_steps = total_steps
        self.observe_fn = observe_fn
        self.transition_fn = transition_fn
        self.physical_unet = physical_unet
        self.registered_decision_indices = registered
        self.branch_ids = branch_ids
        self.state_clone_fn = state_clone_fn
        self.preserve_graph = bool(preserve_graph)
        self.round_trip_rtol = float(round_trip_rtol)
        self.round_trip_atol = float(round_trip_atol)
        self.last_stats = CollectorStats(
            0, 0, 0, 0, 0 if physical_unet is not None else None
        )
        self._verified_unet_forwards: Optional[int] = (
            0 if physical_unet is not None else None
        )
        if binding is not None:
            executor = getattr(adapter, "operation_executor", None)
            if executor is None or executor.authorization_binding is not binding:
                raise RuntimeError(
                    "formal rollout adapter must use the collector authorization binding"
                )

    def _preflight(self) -> None:
        if self.authorization_binding is None:
            if self.require_authorization:
                raise RuntimeError(
                    "rollout collection requires a validated authorization binding"
                )
            return
        self.authorization_binding.validate_current(component=self.renderer)

    def _observe(self, state: Any, index: int) -> EulerRolloutStep:
        # The callback owns the frozen backbone.  Enforce no-grad at the
        # boundary even if a caller forgot to wrap its U-Net invocation.
        physical_calls = 0

        def count_forward(_module: nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
            nonlocal physical_calls
            physical_calls += 1

        handle = (
            self.physical_unet.register_forward_hook(count_forward)
            if self.physical_unet is not None
            else None
        )
        try:
            with torch.no_grad():
                step = self.observe_fn(state, index)
        finally:
            if handle is not None:
                handle.remove()
        if self.physical_unet is not None:
            if physical_calls != 1:
                raise RuntimeError(
                    "observe_fn must execute exactly one physical U-Net forward; "
                    f"observed {physical_calls}"
                )
            self._verified_unet_forwards += physical_calls
        if not isinstance(step, EulerRolloutStep):
            raise TypeError("observe_fn must return EulerRolloutStep")
        self._observe_calls += 1
        return step

    def _clone_branch_state(self, value: Any, branch: str) -> Any:
        if self.adapter is not None:
            return self.adapter.clone_state(
                value,
                branch=branch,
                preserve_graph=self.preserve_graph,
            )
        return _clone_state(
            value,
            preserve_graph=self.preserve_graph,
            clone_fn=self.state_clone_fn,
        )

    @staticmethod
    def _state_field(value: Any, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    def _clone_f0_state(self, value: Any, branch: str) -> Any:
        """Create a detached replay snapshot and set its branch when supported."""

        if not isinstance(branch, str) or not branch:
            raise ValueError("branch must be a non-empty string")
        if self.adapter is not None:
            cloned = self.adapter.clone_state(
                value,
                branch=branch,
                preserve_graph=False,
            )
        else:
            cloned = _clone_state(
                value,
                preserve_graph=False,
                clone_fn=self.state_clone_fn,
            )
            # A caller-provided clone function is not trusted to detach or to
            # allocate fresh tensor storage.  Snapshot its result once more.
            cloned = _clone_value(cloned, detach=True)
            if isinstance(cloned, dict) and "branch" in cloned:
                cloned["branch"] = branch
            elif is_dataclass(cloned) and hasattr(cloned, "branch"):
                cloned = replace(cloned, branch=branch)
            elif hasattr(cloned, "branch"):
                try:
                    setattr(cloned, "branch", branch)
                except (AttributeError, TypeError) as exc:
                    raise TypeError("F0 state branch cannot be set") from exc
        actual_branch = self._state_field(cloned, "branch")
        if actual_branch is not None and actual_branch != branch:
            raise RuntimeError("F0 state clone did not preserve the requested branch")
        return cloned

    def _checkpoint_hash(self, state: Any) -> Optional[str]:
        value = self._state_field(state, "checkpoint_hash")
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError("state checkpoint_hash must be a non-empty string")
        return value

    def _check_f0_state(
        self,
        state: Any,
        *,
        checkpoint_hash: Optional[str],
        branch: str,
        step_index: Optional[int] = None,
    ) -> None:
        if self._checkpoint_hash(state) != checkpoint_hash:
            raise RuntimeError("F0 replay changed checkpoint provenance")
        actual_branch = self._state_field(state, "branch")
        if actual_branch is not None and actual_branch != branch:
            raise RuntimeError("F0 replay changed branch provenance")
        actual_step = self._state_field(state, "step_index")
        if step_index is not None and actual_step is not None and actual_step != step_index:
            raise RuntimeError("F0 state scheduler position is inconsistent")

    def _require_f0_schedule(self) -> None:
        if self.total_steps != F0_TOTAL_STEPS:
            raise ValueError("F0 collection requires exactly 50 scheduler steps")
        if self.decision_indices != F0_DECISION_INDICES:
            raise ValueError("F0 collection requires decision indices 8, 24, and 40")

    def _reset_stats(self) -> None:
        self._observe_calls = 0
        self._transition_calls = 0
        self._decision_count = 0
        self._auxiliary_count = 0
        self._verified_unet_forwards = 0 if self.physical_unet is not None else None

    def _finish_stats(self) -> CollectorStats:
        stats = CollectorStats(
            self._observe_calls,
            self._transition_calls,
            self._decision_count,
            self._auxiliary_count,
            self._verified_unet_forwards,
        )
        self.last_stats = stats
        return stats

    def _transition(
        self,
        state: Any,
        index: int,
        model_output: Tensor,
        step: EulerRolloutStep,
        *,
        action: Optional[Tensor],
    ) -> EulerTransitionResult:
        if self.adapter is not None:
            return self.adapter.transition(
                state,
                index,
                model_output,
                step,
                action=action,
            )
        return self.transition_fn(state, index, model_output, step)

    def _prepare(self, step: EulerRolloutStep, renderer: EulerNativeFrameV1) -> EulerFrameState:
        return renderer.prepare_state(
            step.observation,
            step.condition,
            sigma_from=step.sigma_from,
            sigma_to=step.sigma_to,
            native_model_output=step.native_model_output,
            prediction_type=step.prediction_type,
        )

    def _check_transition(self, result: EulerTransitionResult, step: EulerRolloutStep, expected: Tensor) -> None:
        if result.latent.shape != expected.shape or result.latent.device != expected.device:
            raise ValueError("transition callback returned a latent with the wrong shape or device")
        trajectory_dtype = step.observation.latents_before_step.dtype
        if result.latent.dtype != expected.dtype or result.latent.dtype != trajectory_dtype:
            raise ValueError("transition callback returned a latent with the wrong dtype")
        if not torch.allclose(
            result.latent.detach().float(), expected.detach().float(),
            rtol=self.round_trip_rtol, atol=self.round_trip_atol,
        ):
            raise ValueError("real scheduler transition disagrees with its registered native mapping")

    def _native_transition_from_step(
        self,
        state: Any,
        index: int,
        step: EulerRolloutStep,
        *,
        frame: Optional[EulerFrameState] = None,
        decision_action: Optional[Tensor] = None,
    ) -> tuple[Any, Transition]:
        """Advance with cached native bytes and an explicit decision no-op."""

        current = step.observation.latents_before_step
        zero_action = current.new_zeros(
            (current.shape[0], self.renderer.contract.num_slots)
        )
        is_decision = index in self.registered_decision_indices
        if decision_action is not None:
            if not is_decision:
                raise ValueError("decision_action is only valid at a registered decision")
            if (
                not isinstance(decision_action, Tensor)
                or decision_action.shape != zero_action.shape
                or decision_action.device != zero_action.device
                or not decision_action.is_floating_point()
                or not torch.isfinite(decision_action).all()
                or torch.count_nonzero(decision_action).item() != 0
            ):
                raise ValueError("native decision_action must be an exact finite zero tensor")
            zero_action = decision_action
        result = self._transition(
            state,
            index,
            step.native_model_output,
            step,
            action=zero_action if is_decision else None,
        )
        if not isinstance(result, EulerTransitionResult):
            raise TypeError("transition_fn must return EulerTransitionResult")
        self._transition_calls += 1
        self._check_transition(result, step, step.native_prev_sample)
        # The real scheduler is the source of deployed bytes.  The analytic
        # native_prev_sample above is a parity oracle, not the recorded output.
        native = result.latent - current
        return result.state, Transition(
            state=step.observation if frame is None else frame,
            action=zero_action,
            next_state=result.state,
            native_transition=native,
            rendered_transition=native,
            step_index=index,
        )

    def _native_advance(self, state: Any, index: int) -> tuple[Any, Transition]:
        step = self._observe(state, index)
        return self._native_transition_from_step(state, index, step)

    def _cached_action_transition(
        self,
        state: Any,
        index: int,
        step: EulerRolloutStep,
        frame: EulerFrameState,
        action: Tensor,
    ) -> tuple[Any, Transition]:
        """Apply one F0 action without re-running the cached observation."""

        mapped = self.renderer.forward_euler_mapped(
            step.observation.pred_original_sample,
            step.condition.bases,
            sample=step.observation.latents_before_step,
            native_model_output=step.native_model_output,
            sigma_from=step.sigma_from,
            sigma_to=step.sigma_to,
            prediction_type=step.prediction_type,
            scheduler_update=step.observation.scheduler_update,
            clean_update_gain=step.clean_update_gain,
            timestep=step.observation.normalized_timestep,
            prompt_embedding=step.condition.prompt_embedding,
            state_features=step.condition.state_features,
            action=action,
        )
        result = self._transition(
            state,
            index,
            mapped.model_output,
            step,
            action=action,
        )
        if not isinstance(result, EulerTransitionResult):
            raise TypeError("transition_fn must return EulerTransitionResult")
        self._transition_calls += 1
        self._check_transition(result, step, mapped.predicted_prev_sample)
        current = step.observation.latents_before_step
        return result.state, Transition(
            state=frame,
            action=action,
            next_state=result.state,
            native_transition=step.native_prev_sample - current,
            rendered_transition=result.latent - current,
            step_index=index,
        )

    def _apply_action(
        self,
        state: Any,
        index: int,
        step: EulerRolloutStep,
        frame: EulerFrameState,
        action: Tensor,
        pre_squash: Tensor,
        mean: Tensor,
        log_prob: Optional[Tensor],
        *,
        renderer: Optional[EulerNativeFrameV1] = None,
    ) -> tuple[Any, Transition]:
        active_renderer = self.renderer if renderer is None else renderer
        if not isinstance(active_renderer, EulerNativeFrameV1):
            raise TypeError("action renderer must be EulerNativeFrameV1")
        mapped = active_renderer.forward_euler_mapped(
            step.observation.pred_original_sample,
            step.condition.bases,
            sample=step.observation.latents_before_step,
            native_model_output=step.native_model_output,
            sigma_from=step.sigma_from,
            sigma_to=step.sigma_to,
            prediction_type=step.prediction_type,
            scheduler_update=step.observation.scheduler_update,
            clean_update_gain=step.clean_update_gain,
            timestep=step.observation.normalized_timestep,
            prompt_embedding=step.condition.prompt_embedding,
            state_features=step.condition.state_features,
            action=action,
        )
        result = self._transition(
            state,
            index,
            mapped.model_output,
            step,
            action=action,
        )
        if not isinstance(result, EulerTransitionResult):
            raise TypeError("transition_fn must return EulerTransitionResult")
        self._transition_calls += 1
        self._check_transition(result, step, mapped.predicted_prev_sample)
        current = step.observation.latents_before_step
        reference_mean = None
        if self.reference_renderer is not None:
            with torch.no_grad():
                reference_state = self.reference_renderer.prepare_state(
                    step.observation,
                    step.condition,
                    sigma_from=step.sigma_from,
                    sigma_to=step.sigma_to,
                    native_model_output=step.native_model_output,
                    prediction_type=step.prediction_type,
                )
                reference_mean = self.reference_renderer.action_parameters(reference_state)
        return result.state, Transition(
            state=frame,
            action=action,
            next_state=result.state,
            native_transition=step.native_prev_sample - current,
            rendered_transition=result.latent - current,
            log_prob=log_prob,
            step_index=index,
            pre_squash=pre_squash,
            behavior_mean=mean,
            reference_mean=reference_mean,
        )

    def _proposal(
        self,
        step: EulerRolloutStep,
        frames: Mapping[str, EulerFrameState],
        means: Mapping[str, Tensor],
        noise: Tensor,
    ) -> tuple[DecisionProposal, dict[str, tuple[Tensor, Tensor, Optional[Tensor]]]]:
        actions: dict[str, tuple[Tensor, Tensor, Optional[Tensor]]] = {}
        for branch, sign in ((self.branch_ids[0], 1.0), (self.branch_ids[1], -1.0)):
            mean = means[branch]
            if noise.shape != mean.shape:
                raise ValueError("noise manifest shape must match branch action means")
            distribution = self.renderer.action_distribution(mean)
            signed_noise = noise if sign > 0 else -noise
            action, pre_squash = distribution.rsample(signed_noise)
            log_prob = distribution.log_prob_from_pre_squash(pre_squash)
            actions[branch] = (action, pre_squash, log_prob)
        anchor_mean = means["anchor"]
        anchor_action = self.renderer.deterministic_action_from_mean(anchor_mean)
        # Deterministic deployment uses u=mu exactly; storing it makes the
        # anchor's behavior/reference provenance unambiguous.
        actions["anchor"] = (anchor_action, anchor_mean, None)
        proposal = DecisionProposal(
            step_index=int(step.observation.step_index),
            mean=_clone_value(anchor_mean, detach=not self.preserve_graph),
            noise=_clone_value(noise, detach=not self.preserve_graph),
            plus_action=_clone_value(actions[self.branch_ids[0]][0], detach=not self.preserve_graph),
            minus_action=_clone_value(actions[self.branch_ids[1]][0], detach=not self.preserve_graph),
            anchor_action=_clone_value(anchor_action, detach=not self.preserve_graph),
            plus_log_prob=_clone_value(actions[self.branch_ids[0]][2], detach=not self.preserve_graph),
            minus_log_prob=_clone_value(actions[self.branch_ids[1]][2], detach=not self.preserve_graph),
            plus_mean=_clone_value(means[self.branch_ids[0]], detach=not self.preserve_graph),
            minus_mean=_clone_value(means[self.branch_ids[1]], detach=not self.preserve_graph),
            anchor_mean=_clone_value(anchor_mean, detach=not self.preserve_graph),
            branch_ids=self.branch_ids,
            plus_pre_squash=_clone_value(actions[self.branch_ids[0]][1], detach=not self.preserve_graph),
            minus_pre_squash=_clone_value(actions[self.branch_ids[1]][1], detach=not self.preserve_graph),
            anchor_pre_squash=_clone_value(anchor_mean, detach=not self.preserve_graph),
        )
        return proposal, actions

    def _cached_frame_validity(self, cached: CachedEulerDecision) -> Tensor:
        """Return row validity, treating any cache inconsistency as invalid."""

        batch = cached.step.observation.latents_before_step.shape[0]
        valid = cached.frame.diagnostics.valid
        if (
            not isinstance(valid, Tensor)
            or valid.dtype != torch.bool
            or valid.shape != (batch,)
        ):
            return torch.zeros(
                batch,
                dtype=torch.bool,
                device=cached.step.observation.latents_before_step.device,
            )
        consistent = all(
            torch.equal(left.detach(), right.detach())
            for left, right in (
                (cached.frame.sample, cached.step.observation.latents_before_step),
                (cached.frame.clean_latent, cached.step.observation.pred_original_sample),
                (cached.frame.native_update, cached.step.observation.scheduler_update),
                (cached.frame.raw_bases, cached.step.condition.bases),
            )
        )
        if cached.frame.native_model_output is not None:
            consistent = consistent and torch.equal(
                cached.frame.native_model_output.detach(),
                cached.step.native_model_output.detach(),
            )
        if not consistent:
            return torch.zeros_like(valid)
        return valid.detach().clone()

    def _validate_f0_cache(self, cached: CachedEulerDecision) -> None:
        if not isinstance(cached, CachedEulerDecision):
            raise TypeError("cached decision must be CachedEulerDecision")
        if cached.decision_index not in self.decision_indices:
            raise ValueError("cached decision differs from the collector schedule")
        if cached.frame_contract_hash != self.renderer.frame_contract_hash:
            raise ValueError("cached renderer frame contract differs from the collector")
        if cached.calibration_hash != self.renderer.calibration_hash:
            raise ValueError("cached renderer calibration differs from the collector")
        if cached.run_contract != self.run_contract:
            raise ValueError("cached run contract differs from the collector")
        self._check_f0_state(
            cached.state,
            checkpoint_hash=cached.checkpoint_hash,
            branch=str(self._state_field(cached.state, "branch") or "anchor"),
            step_index=cached.decision_index,
        )
        state_latents = self._state_field(cached.state, "latents")
        if isinstance(state_latents, Tensor) and not torch.equal(
            state_latents.detach(),
            cached.step.observation.latents_before_step.detach(),
        ):
            raise ValueError("cached state and observation latents differ")

    def collect_f0_anchor_trace(
        self,
        initial_state: Any,
        *,
        branch: str = "anchor",
    ) -> F0AnchorTrace:
        """Collect one 50-step strict no-op trace and cache all F0 decisions."""

        self._preflight()
        self._require_f0_schedule()
        self._reset_stats()
        checkpoint_hash = self._checkpoint_hash(initial_state)
        state = self._clone_f0_state(initial_state, branch)
        self._check_f0_state(
            state,
            checkpoint_hash=checkpoint_hash,
            branch=branch,
            step_index=0,
        )
        initial_snapshot = self._clone_f0_state(state, branch)
        caches: dict[int, CachedEulerDecision] = {}
        transitions: list[Transition] = []

        for index in range(F0_TOTAL_STEPS):
            step = self._observe(state, index)
            if step.observation.step_index != index:
                raise ValueError("observation step_index differs from the F0 schedule")
            if index in F0_DECISION_INDICES:
                frame = self._prepare(step, self.renderer)
                caches[index] = CachedEulerDecision(
                    decision_index=index,
                    state=self._clone_f0_state(state, branch),
                    step=_clone_value(step, detach=True),
                    frame=_clone_value(frame, detach=True),
                    checkpoint_hash=checkpoint_hash,
                    frame_contract_hash=self.renderer.frame_contract_hash,
                    calibration_hash=self.renderer.calibration_hash,
                    run_contract=self.run_contract,
                )
                state, transition = self._native_transition_from_step(
                    state,
                    index,
                    step,
                    frame=frame,
                )
                self._decision_count += 1
            else:
                state, transition = self._native_transition_from_step(
                    state,
                    index,
                    step,
                )
                self._auxiliary_count += 1
            transitions.append(_clone_value(transition, detach=True))
            self._check_f0_state(
                state,
                checkpoint_hash=checkpoint_hash,
                branch=branch,
                step_index=index + 1,
            )

        self._preflight()
        stats = self._finish_stats()
        return F0AnchorTrace(
            branch=branch,
            initial_state=initial_snapshot,
            terminal_state=self._clone_f0_state(state, branch),
            cached_decisions=caches,
            transitions=tuple(transitions),
            stats=stats,
            checkpoint_hash=checkpoint_hash,
        )

    def collect_f0_anchor(
        self,
        initial_state: Any,
        *,
        branch: str = "anchor",
    ) -> F0AnchorTrace:
        """Compatibility alias for :meth:`collect_f0_anchor_trace`."""

        return self.collect_f0_anchor_trace(initial_state, branch=branch)

    def replay_f0_suffix(
        self,
        source: F0AnchorTrace | CachedEulerDecision,
        action: Tensor,
        *,
        branch: str,
        decision_index: Optional[int] = None,
    ) -> CachedSuffixResult:
        """Replay one cached F0 decision and its remaining native suffix."""

        self._preflight()
        self._require_f0_schedule()
        if isinstance(source, F0AnchorTrace):
            if decision_index is None:
                raise ValueError("decision_index is required when replaying an anchor trace")
            if decision_index not in source.cached_decisions:
                raise ValueError("anchor trace does not contain the requested decision")
            cached = source.cached_decisions[decision_index]
            if source.checkpoint_hash != cached.checkpoint_hash:
                raise ValueError("anchor trace and cached checkpoint provenance differ")
        elif isinstance(source, CachedEulerDecision):
            cached = source
            if decision_index is not None and decision_index != cached.decision_index:
                raise ValueError("requested decision differs from the cached decision")
            decision_index = cached.decision_index
        else:
            raise TypeError("source must be F0AnchorTrace or CachedEulerDecision")
        self._validate_f0_cache(cached)
        if not isinstance(action, Tensor):
            raise TypeError("F0 suffix action must be a tensor")
        current = cached.step.observation.latents_before_step
        expected_shape = (current.shape[0], self.renderer.contract.num_slots)
        if (
            action.shape != expected_shape
            or action.device != current.device
            or not action.is_floating_point()
            or not torch.isfinite(action).all()
        ):
            raise ValueError("F0 suffix action has an invalid shape, device, dtype, or value")
        self.renderer.contract.validate_action(action)
        self._reset_stats()

        checkpoint_hash = cached.checkpoint_hash
        state = self._clone_f0_state(cached.state, branch)
        self._check_f0_state(
            state,
            checkpoint_hash=checkpoint_hash,
            branch=branch,
            step_index=decision_index,
        )
        initial_snapshot = self._clone_f0_state(state, branch)
        step = _clone_value(cached.step, detach=True)
        frame = _clone_value(cached.frame, detach=True)
        frame_valid = self._cached_frame_validity(cached).to(device=action.device)
        applied_action = torch.where(
            frame_valid.reshape(-1, 1),
            action,
            torch.zeros_like(action),
        )
        if torch.count_nonzero(applied_action).item() == 0:
            state, transition = self._native_transition_from_step(
                state,
                decision_index,
                step,
                frame=frame,
                decision_action=applied_action,
            )
        else:
            state, transition = self._cached_action_transition(
                state,
                decision_index,
                step,
                frame,
                applied_action,
            )
        self._decision_count += 1
        transitions = [_clone_value(transition, detach=True)]
        self._check_f0_state(
            state,
            checkpoint_hash=checkpoint_hash,
            branch=branch,
            step_index=decision_index + 1,
        )

        for index in range(decision_index + 1, F0_TOTAL_STEPS):
            state, transition = self._native_advance(state, index)
            transitions.append(_clone_value(transition, detach=True))
            if index in F0_DECISION_INDICES:
                self._decision_count += 1
            else:
                self._auxiliary_count += 1
            self._check_f0_state(
                state,
                checkpoint_hash=checkpoint_hash,
                branch=branch,
                step_index=index + 1,
            )

        self._preflight()
        stats = self._finish_stats()
        return CachedSuffixResult(
            decision_index=decision_index,
            branch=branch,
            initial_state=initial_snapshot,
            terminal_state=self._clone_f0_state(state, branch),
            requested_action=action.detach().clone(),
            applied_action=applied_action.detach().clone(),
            frame_valid=frame_valid.detach().clone(),
            transitions=tuple(transitions),
            stats=stats,
            checkpoint_hash=checkpoint_hash,
        )

    def replay_cached_suffix(
        self,
        cached: CachedEulerDecision,
        action: Tensor,
        *,
        branch: str,
    ) -> CachedSuffixResult:
        """Compatibility alias for replaying one cached decision directly."""

        return self.replay_f0_suffix(cached, action, branch=branch)

    def collect(
        self,
        initial_state: Any,
        *,
        noise_by_decision: Mapping[int, Tensor],
    ) -> RolloutCollection:
        """Collect one prefix, three branches, and a common decision schedule."""
        # This must precede noise validation and, crucially, the first
        # observe_fn invocation.  A failed authorization therefore cannot
        # consume U-Net, scheduler, RNG, or reward work.
        self._preflight()
        if not isinstance(noise_by_decision, Mapping) or set(noise_by_decision) != set(self.decision_indices):
            raise ValueError("noise manifest keys must exactly match the registered schedule")
        if any(type(key) is not int for key in noise_by_decision):
            raise ValueError("noise manifest keys must be plain integers")
        self._observe_calls = 0
        self._transition_calls = 0
        self._decision_count = 0
        self._auxiliary_count = 0
        self._verified_unet_forwards = 0 if self.physical_unet is not None else None

        first = self.decision_indices[0]
        state = initial_state
        prefix = BranchTrajectory("prefix", preserve_graph=self.preserve_graph)
        for index in range(first):
            state, transition = self._native_advance(state, index)
            prefix.append_auxiliary(transition)
            self._auxiliary_count += 1

        branches = {
            branch: BranchTrajectory(branch, preserve_graph=self.preserve_graph)
            for branch in (*self.branch_ids, "anchor")
        }
        states = {
            branch: self._clone_branch_state(state, branch)
            for branch in (*self.branch_ids, "anchor")
        }
        proposals: list[DecisionProposal] = []
        previous = first

        for position, decision_index in enumerate(self.decision_indices):
            if position == 0:
                # The first decision is still on the shared prefix.  One
                # observation feeds all three action branches.
                shared_step = self._observe(state, decision_index)
                steps = {branch: shared_step for branch in states}
            else:
                for branch in states:
                    for index in range(previous, decision_index):
                        states[branch], transition = self._native_advance(states[branch], index)
                        branches[branch].append_auxiliary(transition)
                        self._auxiliary_count += 1
                steps = {
                    branch: self._observe(states[branch], decision_index)
                    for branch in states
                }

            frames = {
                branch: self._prepare(steps[branch], self.renderer)
                for branch in states
            }
            if self.preserve_graph:
                means = {
                    branch: self.renderer.action_parameters(frames[branch])
                    for branch in states
                }
            else:
                with torch.no_grad():
                    means = {
                        branch: self.renderer.action_parameters(frames[branch])
                        for branch in states
                    }
            noise = noise_by_decision[decision_index]
            if not isinstance(noise, Tensor) or not noise.is_floating_point() or not torch.isfinite(noise).all():
                raise ValueError("noise manifest values must be finite floating tensors")
            proposal, actions = self._proposal(steps["anchor"], frames, means, noise)
            if proposal.step_index != decision_index:
                raise ValueError("observation step_index differs from the registered decision")
            for branch in (*self.branch_ids, "anchor"):
                action, pre_squash, log_prob = actions[branch]
                states[branch], transition = self._apply_action(
                    states[branch], decision_index, steps[branch], frames[branch],
                    action, pre_squash, means[branch], log_prob,
                )
                branches[branch].append(transition)
                self._decision_count += 1
            proposals.append(proposal)
            previous = decision_index + 1

        for branch in states:
            for index in range(previous, self.total_steps):
                states[branch], transition = self._native_advance(states[branch], index)
                branches[branch].append_auxiliary(transition)
                self._auxiliary_count += 1

        # Revalidate once after the collection so a mid-rollout Git or data
        # change cannot be published as an authorized trajectory.  Individual
        # U-Net/scheduler callbacks deliberately do not rehash provenance.
        self._preflight()

        self.last_stats = CollectorStats(
            self._observe_calls,
            self._transition_calls,
            self._decision_count,
            self._auxiliary_count,
            self._verified_unet_forwards,
        )
        return RolloutCollection(
            branches=branches,
            proposals=proposals,
            prefix_steps=first,
            total_steps=self.total_steps,
            terminal_states={
                branch: self._clone_branch_state(value, branch)
                for branch, value in states.items()
            },
            preserve_graph=self.preserve_graph,
            prefix_transitions=prefix.auxiliary_transitions,
        )

    def collect_with_teacher(
        self,
        initial_state: Any,
        *,
        noise_by_decision: Mapping[int, Tensor],
        teacher_renderer: EulerNativeFrameV1,
        teacher_checkpoint_hash: Optional[str] = None,
    ) -> tuple[RolloutCollection, TeacherRolloutResult]:
        """Collect student branches and one deterministic teacher branch.

        The first decision observation is shared with the student prefix.  At
        later decisions the teacher follows its own state and therefore gets a
        fresh frozen-backbone observation.  This gives the protocol's
        ``9 + 4 * 41`` physical U-Net topology while keeping OPD labels tied
        to the student anchor frames (the labels are constructed by the
        runner, not by teacher-generated states).
        """
        self._preflight()
        if not isinstance(teacher_renderer, EulerNativeFrameV1):
            raise TypeError("teacher_renderer must be EulerNativeFrameV1")
        if teacher_renderer is self.renderer:
            raise ValueError("teacher_renderer must be distinct from the student")
        if teacher_renderer.frame_contract_hash != self.renderer.frame_contract_hash:
            raise ValueError("teacher frame contract differs from the student")
        if teacher_renderer.calibration_hash != self.renderer.calibration_hash:
            raise ValueError("teacher calibration differs from the student")
        def contract_payload(value: Any) -> Any:
            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                return to_dict()
            if isinstance(value, Mapping):
                return dict(value)
            raise TypeError("teacher and student expose an unsupported action contract")

        if contract_payload(teacher_renderer.contract) != contract_payload(self.renderer.contract):
            raise ValueError("teacher action contract differs from the student")
        if any(parameter.requires_grad for parameter in teacher_renderer.parameters()):
            raise ValueError("teacher_renderer must be frozen")
        teacher_renderer.eval()
        if teacher_checkpoint_hash is not None and (
            not isinstance(teacher_checkpoint_hash, str) or len(teacher_checkpoint_hash) != 64
        ):
            raise ValueError("teacher_checkpoint_hash must be a SHA-256 string")
        if not isinstance(noise_by_decision, Mapping) or set(noise_by_decision) != set(self.decision_indices):
            raise ValueError("noise manifest keys must exactly match the registered schedule")
        if any(type(key) is not int for key in noise_by_decision):
            raise ValueError("noise manifest keys must be plain integers")

        self._observe_calls = 0
        self._transition_calls = 0
        self._decision_count = 0
        self._auxiliary_count = 0
        self._verified_unet_forwards = 0 if self.physical_unet is not None else None

        first = self.decision_indices[0]
        state = initial_state
        prefix = BranchTrajectory("prefix", preserve_graph=self.preserve_graph)
        for index in range(first):
            state, transition = self._native_advance(state, index)
            prefix.append_auxiliary(transition)
            self._auxiliary_count += 1

        branches = {
            branch: BranchTrajectory(branch, preserve_graph=self.preserve_graph)
            for branch in (*self.branch_ids, "anchor")
        }
        states = {
            branch: self._clone_branch_state(state, branch)
            for branch in (*self.branch_ids, "anchor")
        }
        teacher_state = self._clone_branch_state(state, "teacher")
        if teacher_checkpoint_hash is not None:
            if is_dataclass(teacher_state):
                if not hasattr(teacher_state, "checkpoint_hash"):
                    raise TypeError("teacher state has no checkpoint provenance")
                teacher_state = replace(
                    teacher_state, checkpoint_hash=teacher_checkpoint_hash
                )
            elif isinstance(teacher_state, dict):
                teacher_state["checkpoint_hash"] = teacher_checkpoint_hash
            elif hasattr(teacher_state, "checkpoint_hash"):
                try:
                    setattr(teacher_state, "checkpoint_hash", teacher_checkpoint_hash)
                except (AttributeError, TypeError) as exc:
                    raise TypeError("teacher state checkpoint provenance is immutable") from exc
            else:
                raise TypeError("teacher state has no checkpoint provenance")
        teacher_branch = BranchTrajectory("teacher", preserve_graph=self.preserve_graph)
        proposals: list[DecisionProposal] = []
        previous = first

        for position, decision_index in enumerate(self.decision_indices):
            if position == 0:
                # One observation at the first decision is shared by all
                # student branches and the external teacher branch.
                shared_step = self._observe(state, decision_index)
                steps = {branch: shared_step for branch in states}
                teacher_step = shared_step
            else:
                for branch in states:
                    for index in range(previous, decision_index):
                        states[branch], transition = self._native_advance(states[branch], index)
                        branches[branch].append_auxiliary(transition)
                        self._auxiliary_count += 1
                for index in range(previous, decision_index):
                    teacher_state, transition = self._native_advance(teacher_state, index)
                    teacher_branch.append_auxiliary(transition)
                    self._auxiliary_count += 1
                steps = {
                    branch: self._observe(states[branch], decision_index)
                    for branch in states
                }
                teacher_step = self._observe(teacher_state, decision_index)

            frames = {
                branch: self._prepare(steps[branch], self.renderer)
                for branch in states
            }
            teacher_frame = self._prepare(teacher_step, teacher_renderer)
            if self.preserve_graph:
                means = {
                    branch: self.renderer.action_parameters(frames[branch])
                    for branch in states
                }
                teacher_mean = teacher_renderer.action_parameters(teacher_frame)
            else:
                with torch.no_grad():
                    means = {
                        branch: self.renderer.action_parameters(frames[branch])
                        for branch in states
                    }
                    teacher_mean = teacher_renderer.action_parameters(teacher_frame)
            noise = noise_by_decision[decision_index]
            if not isinstance(noise, Tensor) or not noise.is_floating_point() or not torch.isfinite(noise).all():
                raise ValueError("noise manifest values must be finite floating tensors")
            proposal, actions = self._proposal(steps["anchor"], frames, means, noise)
            if proposal.step_index != decision_index:
                raise ValueError("observation step_index differs from the registered decision")
            for branch in (*self.branch_ids, "anchor"):
                action, pre_squash, log_prob = actions[branch]
                states[branch], transition = self._apply_action(
                    states[branch], decision_index, steps[branch], frames[branch],
                    action, pre_squash, means[branch], log_prob,
                )
                branches[branch].append(transition)
                self._decision_count += 1

            teacher_action = teacher_renderer.deterministic_action_from_mean(teacher_mean)
            teacher_state, teacher_transition = self._apply_action(
                teacher_state,
                decision_index,
                teacher_step,
                teacher_frame,
                teacher_action,
                teacher_mean,
                teacher_mean,
                None,
                renderer=teacher_renderer,
            )
            teacher_branch.append(teacher_transition)
            self._decision_count += 1
            proposals.append(proposal)
            previous = decision_index + 1

        for branch in states:
            for index in range(previous, self.total_steps):
                states[branch], transition = self._native_advance(states[branch], index)
                branches[branch].append_auxiliary(transition)
                self._auxiliary_count += 1
        for index in range(previous, self.total_steps):
            teacher_state, transition = self._native_advance(teacher_state, index)
            teacher_branch.append_auxiliary(transition)
            self._auxiliary_count += 1

        self._preflight()
        stats = CollectorStats(
            self._observe_calls,
            self._transition_calls,
            self._decision_count,
            self._auxiliary_count,
            self._verified_unet_forwards,
        )
        self.last_stats = stats
        teacher_steps = self.total_steps - first
        teacher_stats = CollectorStats(
            teacher_steps,
            teacher_steps,
            len(self.decision_indices),
            teacher_steps - len(self.decision_indices),
            None
            if self.physical_unet is None
            else max(teacher_steps - 1, 0),
        )
        collection = RolloutCollection(
            branches=branches,
            proposals=proposals,
            prefix_steps=first,
            total_steps=self.total_steps,
            terminal_states={
                branch: self._clone_branch_state(value, branch)
                for branch, value in states.items()
            },
            preserve_graph=self.preserve_graph,
            prefix_transitions=prefix.auxiliary_transitions,
        )
        teacher_result = TeacherRolloutResult(
            branch=teacher_branch,
            terminal_state=self._clone_branch_state(teacher_state, "teacher"),
            stats=teacher_stats,
        )
        return collection, teacher_result


__all__ = [
    "CachedEulerDecision",
    "CachedSuffixResult",
    "CollectorStats",
    "EulerRolloutStep",
    "EulerTransitionResult",
    "EulerNativeRolloutCollector",
    "TeacherRolloutResult",
    "F0AnchorTrace",
    "F0_DECISION_INDICES",
    "F0_TOTAL_STEPS",
]
