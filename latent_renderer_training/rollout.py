"""Framework-neutral trajectory records and shared-prefix replay."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import copy
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class Transition:
    """One renderer decision and the native scheduler transition it changes."""

    state: Any
    action: Tensor
    next_state: Any
    native_transition: Tensor
    rendered_transition: Tensor
    log_prob: Optional[Tensor] = None
    step_index: Optional[int] = None
    # Training records keep the exact pre-squash sample.  Reconstructing it
    # from a rounded bounded action with ``atanh`` is not equivalent near the
    # open coefficient boundary and can change DPO/RL ratios.
    pre_squash: Optional[Tensor] = None
    behavior_mean: Optional[Tensor] = None
    reference_mean: Optional[Tensor] = None


def _clone_value(value: Any, *, detach: bool = True) -> Any:
    """Copy a rollout value, optionally retaining its autograd graph."""
    if isinstance(value, Tensor):
        return value.detach().clone() if detach else value.clone()
    if is_dataclass(value) and not isinstance(value, type):
        replacements = {
            item.name: _clone_value(getattr(value, item.name), detach=detach)
            for item in fields(value)
            if item.init
        }
        try:
            return replace(value, **replacements)
        except Exception as exc:
            raise TypeError("transition dataclass cannot be snapshotted") from exc
    if isinstance(value, dict):
        return {key: _clone_value(item, detach=detach) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item, detach=detach) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item, detach=detach) for item in value)
    if isinstance(value, set):
        return {_clone_value(item, detach=detach) for item in value}
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise TypeError("transition field cannot be snapshotted") from exc


def _snapshot(value: Any) -> Any:
    """Detach and copy rollout fields for the default audit collector."""
    return _clone_value(value, detach=True)


@dataclass
class BranchTrajectory:
    """A branch with decision transitions and separate native bookkeeping.

    ``transitions`` contains only rows at which an action was selected.  A
    caller may also return ordinary scheduler transitions while traversing a
    gap or suffix; those rows are retained in ``auxiliary_transitions`` so
    they cannot accidentally enter a per-decision DPO/RL loss.
    """

    branch_id: str
    transitions: list[Transition] = field(default_factory=list)
    terminal_reward: Optional[Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    preserve_graph: bool = False
    auxiliary_transitions: list[Transition] = field(default_factory=list)

    def _copy_transition(self, transition: Transition) -> Transition:
        detach = not self.preserve_graph
        return Transition(
            state=_clone_value(transition.state, detach=detach),
            action=_clone_value(transition.action, detach=detach),
            next_state=_clone_value(transition.next_state, detach=detach),
            native_transition=_clone_value(transition.native_transition, detach=detach),
            rendered_transition=_clone_value(transition.rendered_transition, detach=detach),
            log_prob=_clone_value(transition.log_prob, detach=detach),
            step_index=transition.step_index,
            pre_squash=_clone_value(transition.pre_squash, detach=detach),
            behavior_mean=_clone_value(transition.behavior_mean, detach=detach),
            reference_mean=_clone_value(transition.reference_mean, detach=detach),
        )

    def append(self, transition: Transition, *, auxiliary: bool = False) -> None:
        if not isinstance(transition, Transition):
            raise TypeError("trajectory transitions must be Transition instances")
        destination = self.auxiliary_transitions if auxiliary else self.transitions
        destination.append(self._copy_transition(transition))

    def append_auxiliary(self, transition: Transition) -> None:
        """Record a non-decision scheduler transition outside the action list."""
        self.append(transition, auxiliary=True)

    @property
    def all_transitions(self) -> tuple[Transition, ...]:
        """Return decision and auxiliary rows for audit/reporting purposes."""
        rows = list(self.transitions) + list(self.auxiliary_transitions)
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.step_index is None,
                    row.step_index if row.step_index is not None else 0,
                ),
            )
        )

    @property
    def actions(self) -> Tensor:
        if not self.transitions:
            raise ValueError("trajectory has no transitions")
        return torch.stack([item.action for item in self.transitions], dim=-2)


@dataclass(frozen=True)
class DecisionProposal:
    """One antithetic proposal shared by all method arms."""

    step_index: int
    mean: Tensor
    noise: Tensor
    plus_action: Tensor
    minus_action: Tensor
    anchor_action: Tensor
    plus_log_prob: Optional[Tensor] = None
    minus_log_prob: Optional[Tensor] = None
    # Keep all branch-conditioned means.  ``mean`` remains the historical
    # anchor alias so older readers can consume a proposal manifest, while
    # DPO/RL can now evaluate the density at the state each branch actually
    # visited.
    plus_mean: Optional[Tensor] = None
    minus_mean: Optional[Tensor] = None
    anchor_mean: Optional[Tensor] = None
    branch_ids: tuple[str, str] = ("plus", "minus")
    plus_pre_squash: Optional[Tensor] = None
    minus_pre_squash: Optional[Tensor] = None
    anchor_pre_squash: Optional[Tensor] = None

    def branch_means(self, *, allow_legacy_alias: bool = False) -> dict[str, Tensor]:
        """Return branch-conditioned means, rejecting incomplete new records.

        ``allow_legacy_alias`` is an explicit escape hatch for old manifests
        that only stored the historical anchor alias.  New OPD/DPO/RL code
        must leave it false so a missing branch mean cannot silently corrupt a
        density calculation.
        """
        if self.plus_mean is None or self.minus_mean is None or self.anchor_mean is None:
            if not allow_legacy_alias:
                raise ValueError("proposal is missing branch-conditioned means")
            anchor = self.mean
            return {
                self.branch_ids[0]: anchor,
                self.branch_ids[1]: anchor,
                "anchor": anchor,
            }
        return {
            self.branch_ids[0]: self.plus_mean,
            self.branch_ids[1]: self.minus_mean,
            "anchor": self.anchor_mean,
        }


@dataclass
class RolloutCollection:
    """A common-prefix collection with an auditable proposal manifest."""

    branches: dict[str, BranchTrajectory]
    proposals: list[DecisionProposal]
    prefix_steps: int
    total_steps: int
    terminal_states: dict[str, Any] = field(default_factory=dict)
    preserve_graph: bool = False
    prefix_transitions: list[Transition] = field(default_factory=list)
    # OPD persists the exact detached teacher targets alongside the generic
    # trajectory.  Defaults retain positional compatibility with older pair
    # rollouts and synthetic callers.
    opd_teacher_targets: tuple[Tensor, ...] | None = None
    opd_teacher_label_provenance: tuple[Mapping[str, Any], ...] | None = None

    @property
    def all_transitions(self) -> dict[str, tuple[Transition, ...]]:
        """Return the complete physical trajectory split by shared/branch rows."""
        return {
            "prefix": tuple(self.prefix_transitions),
            **{
                branch: trajectory.all_transitions
                for branch, trajectory in self.branches.items()
            },
        }


def _call_step(
    step_fn: Callable[[Any, Optional[Tensor], int], Any],
    state: Any,
    action: Optional[Tensor],
    index: int,
) -> tuple[Any, Optional[Transition]]:
    result = step_fn(state, action, index)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Transition):
        return result[0], result[1]
    return result, None


def _unpack_action_result(
    result: Any,
    *,
    label: str,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Normalize a two- or three-field action sampler result.

    The three-field form is ``(action, pre_squash, log_prob)``.  Keeping the
    exact pre-squash coordinate in the collection avoids an ``atanh`` round
    trip when a later DPO/RL objective evaluates the transformed density.
    """
    if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
        raise ValueError(
            f"action_from_mean must return (action, log_prob) or "
            f"(action, pre_squash, log_prob) for {label!r}"
        )
    if len(result) == 2:
        action, log_prob = result
        pre_squash = None
    else:
        action, pre_squash, log_prob = result
    return action, pre_squash, log_prob


def collect_antithetic_rollout(
    initial_state: Any,
    *,
    decision_indices: Sequence[int],
    total_steps: int,
    step_fn: Callable[[Any, Optional[Tensor], int], Any],
    mean_fn: Callable[[Any, int], Tensor],
    action_from_mean: Callable[[Tensor, Tensor], tuple[Tensor, Optional[Tensor]]],
    noise_by_decision: Mapping[int, Tensor],
    branch_ids: tuple[str, str] = ("plus", "minus"),
    registered_decision_indices: Sequence[int],
    state_clone_fn: Optional[Callable[[Any], Any]] = None,
    preserve_graph: bool = False,
    require_pre_squash: bool = False,
) -> RolloutCollection:
    """Collect plus/minus/anchor branches with one physical shared prefix.

    ``mean_fn`` is evaluated on the state actually visited by each branch.
    ``action_from_mean(mean, noise)`` must return a bounded coefficient action
    and its log probability (or ``None``).  The plus branch receives ``+noise``
    and the minus branch receives ``-noise``; the deterministic anchor uses a
    zero noise tensor.  ``registered_decision_indices`` is mandatory: a formal
    run must bind the schedule in its manifest instead of silently accepting a
    caller's ad-hoc decision positions.  By default every recorded field is
    detached for an audit collection.  Set ``preserve_graph=True`` explicitly
    for a small differentiable training collection and provide a state clone
    function that preserves the intended graph.
    """
    if len(branch_ids) != 2 or branch_ids[0] == branch_ids[1]:
        raise ValueError("branch_ids must contain two distinct names")
    if "anchor" in branch_ids:
        raise ValueError("branch_ids cannot reuse the reserved anchor name")
    if isinstance(total_steps, bool) or type(total_steps) is not int or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    raw_indices = tuple(decision_indices)
    if not raw_indices or any(type(value) is not int for value in raw_indices):
        raise ValueError("decision_indices must contain only integer values")
    indices = raw_indices
    if not indices or any(value < 0 for value in indices):
        raise ValueError("decision_indices must be a non-empty sequence of non-negative integers")
    if any(value >= total_steps for value in indices):
        raise ValueError("decision_indices must be smaller than total_steps")
    if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
        raise ValueError("decision_indices must be strictly increasing")
    if registered_decision_indices is None:
        raise ValueError("registered_decision_indices is required for a formal rollout")
    registered_raw = tuple(registered_decision_indices)
    if not registered_raw or any(type(value) is not int for value in registered_raw):
        raise ValueError("registered_decision_indices must contain only integers")
    if indices != registered_raw:
        raise ValueError("decision_indices differ from the registered schedule")
    if not isinstance(noise_by_decision, Mapping):
        raise ValueError("noise_by_decision must be a mapping")
    if any(type(key) is not int for key in noise_by_decision):
        raise ValueError("noise manifest keys must be plain integers")
    if set(noise_by_decision) != set(indices):
        raise ValueError("noise manifest keys must exactly match decision_indices")
    prefix_steps = indices[0]
    state = initial_state
    prefix_holder = BranchTrajectory(
        branch_id="prefix", preserve_graph=preserve_graph
    )
    for index in range(prefix_steps):
        state, transition = _call_step(step_fn, state, None, index)
        if transition is not None:
            prefix_holder.append_auxiliary(transition)

    def clone(value: Any) -> Any:
        if state_clone_fn is not None:
            return state_clone_fn(value)
        return _clone_value(value, detach=not preserve_graph)

    starts = {branch_ids[0]: clone(state), branch_ids[1]: clone(state)}
    # The anchor is collected as a third branch.  Its actions are deterministic
    # means and therefore do not consume a stochastic noise draw.
    starts["anchor"] = clone(state)
    trajectories = {
        branch: BranchTrajectory(branch_id=branch, preserve_graph=preserve_graph)
        for branch in starts
    }
    states = starts
    proposals: list[DecisionProposal] = []
    previous_index = prefix_steps
    for decision_index in indices:
        # Advance every branch through the gap since the previous decision.
        for branch in tuple(states):
            for index in range(previous_index, decision_index):
                states[branch], transition = _call_step(step_fn, states[branch], None, index)
                if transition is not None:
                    trajectories[branch].append_auxiliary(transition)
        means = {
            branch: mean_fn(states[branch], decision_index)
            for branch in (branch_ids[0], branch_ids[1], "anchor")
        }
        mean = means["anchor"]
        if not isinstance(mean, Tensor) or mean.ndim < 1 or not mean.is_floating_point():
            raise ValueError("mean_fn must return a floating tensor with a slot dimension")
        if not torch.isfinite(mean).all():
            raise ValueError("mean_fn returned a non-finite anchor mean")
        for branch, branch_mean in means.items():
            if (
                not isinstance(branch_mean, Tensor)
                or branch_mean.shape != mean.shape
                or not branch_mean.is_floating_point()
                or not torch.isfinite(branch_mean).all()
            ):
                raise ValueError(
                    f"mean_fn returned an invalid or mismatched mean for branch {branch!r}"
                )
        if decision_index not in noise_by_decision:
            raise ValueError(f"noise manifest is missing decision index {decision_index}")
        noise = noise_by_decision[decision_index]
        if not isinstance(noise, Tensor) or noise.shape != mean.shape:
            raise ValueError("noise manifest shape must match the anchor mean")
        if not torch.isfinite(noise).all():
            raise ValueError("noise manifest contains non-finite values")
        plus_action, plus_pre_squash, plus_log_prob = _unpack_action_result(
            action_from_mean(means[branch_ids[0]], noise), label=branch_ids[0]
        )
        minus_action, minus_pre_squash, minus_log_prob = _unpack_action_result(
            action_from_mean(means[branch_ids[1]], -noise), label=branch_ids[1]
        )
        anchor_action, anchor_pre_squash, _anchor_log_prob = _unpack_action_result(
            action_from_mean(means["anchor"], torch.zeros_like(noise)), label="anchor"
        )
        actions = {
            branch_ids[0]: (plus_action, plus_pre_squash, plus_log_prob),
            branch_ids[1]: (minus_action, minus_pre_squash, minus_log_prob),
            "anchor": (anchor_action, anchor_pre_squash, None),
        }
        for branch, (action, pre_squash, log_prob) in actions.items():
            branch_mean = means[branch]
            if (
                not isinstance(action, Tensor)
                or action.shape != branch_mean.shape
                or not action.is_floating_point()
                or not torch.isfinite(action).all()
            ):
                raise ValueError(f"action_from_mean returned an invalid action for {branch!r}")
            if pre_squash is not None:
                if (
                    not isinstance(pre_squash, Tensor)
                    or pre_squash.shape != branch_mean.shape
                    or not pre_squash.is_floating_point()
                    or not torch.isfinite(pre_squash).all()
                ):
                    raise ValueError(
                        f"action_from_mean returned an invalid pre_squash for {branch!r}"
                    )
            elif require_pre_squash:
                raise ValueError(
                    f"formal rollout requires a pre_squash sample for {branch!r}"
                )
            if log_prob is not None:
                if (
                    not isinstance(log_prob, Tensor)
                    or log_prob.shape != branch_mean.shape[:-1]
                    or not log_prob.is_floating_point()
                    or not torch.isfinite(log_prob).all()
                ):
                    raise ValueError(
                        f"action_from_mean returned an invalid log probability for {branch!r}"
                    )
        for branch, action, pre_squash, log_prob in (
            (branch_ids[0], plus_action, plus_pre_squash, plus_log_prob),
            (branch_ids[1], minus_action, minus_pre_squash, minus_log_prob),
            ("anchor", anchor_action, anchor_pre_squash, None),
        ):
            states[branch], transition = _call_step(
                step_fn, states[branch], action, decision_index
            )
            if transition is None:
                raise ValueError("step_fn must return a Transition at every decision")
            if (
                not isinstance(transition.action, Tensor)
                or transition.action.shape != action.shape
                or not torch.equal(
                    transition.action.detach(), action.detach()
                )
            ):
                raise ValueError(
                    f"step_fn recorded an action different from the proposed {branch!r} action"
                )
            if (
                transition.step_index is not None
                and int(transition.step_index) != decision_index
            ):
                raise ValueError("decision transition has a mismatched step_index")
            if type(transition.step_index) is not int:
                raise ValueError("decision transition must record an integer step_index")
            if transition.log_prob is None and log_prob is not None:
                transition = Transition(
                    transition.state,
                    transition.action,
                    transition.next_state,
                    transition.native_transition,
                    transition.rendered_transition,
                    log_prob=log_prob,
                    step_index=transition.step_index,
                    pre_squash=transition.pre_squash,
                    behavior_mean=transition.behavior_mean,
                    reference_mean=transition.reference_mean,
                )
            if pre_squash is not None:
                if transition.pre_squash is not None and not torch.equal(
                    transition.pre_squash.detach(), pre_squash.detach()
                ):
                    raise ValueError(
                        f"step_fn recorded a different pre_squash for {branch!r}"
                    )
                if transition.pre_squash is None:
                    transition = Transition(
                        transition.state,
                        transition.action,
                        transition.next_state,
                        transition.native_transition,
                        transition.rendered_transition,
                        log_prob=transition.log_prob,
                        step_index=transition.step_index,
                        pre_squash=pre_squash,
                        behavior_mean=transition.behavior_mean,
                        reference_mean=transition.reference_mean,
                    )
            trajectories[branch].append(transition)
        proposals.append(
            DecisionProposal(
                step_index=decision_index,
                mean=_clone_value(mean, detach=not preserve_graph),
                noise=_clone_value(noise, detach=not preserve_graph),
                plus_action=_clone_value(plus_action, detach=not preserve_graph),
                minus_action=_clone_value(minus_action, detach=not preserve_graph),
                anchor_action=_clone_value(anchor_action, detach=not preserve_graph),
                plus_log_prob=(
                    None
                    if plus_log_prob is None
                    else _clone_value(plus_log_prob, detach=not preserve_graph)
                ),
                minus_log_prob=(
                    None
                    if minus_log_prob is None
                    else _clone_value(minus_log_prob, detach=not preserve_graph)
                ),
                plus_mean=_clone_value(means[branch_ids[0]], detach=not preserve_graph),
                minus_mean=_clone_value(means[branch_ids[1]], detach=not preserve_graph),
                anchor_mean=_clone_value(means["anchor"], detach=not preserve_graph),
                branch_ids=branch_ids,
                plus_pre_squash=(
                    None
                    if plus_pre_squash is None
                    else _clone_value(plus_pre_squash, detach=not preserve_graph)
                ),
                minus_pre_squash=(
                    None
                    if minus_pre_squash is None
                    else _clone_value(minus_pre_squash, detach=not preserve_graph)
                ),
                anchor_pre_squash=(
                    None
                    if anchor_pre_squash is None
                    else _clone_value(anchor_pre_squash, detach=not preserve_graph)
                ),
            )
        )
        previous_index = decision_index + 1
    # Complete the common suffix after the last decision.  Every branch gets
    # exactly the same number of native scheduler calls, including the final
    # terminal state used by reward evaluation.
    for branch in tuple(states):
        for index in range(previous_index, total_steps):
            states[branch], transition = _call_step(
                step_fn, states[branch], None, index
            )
            if transition is not None:
                trajectories[branch].append_auxiliary(transition)
    terminal_states = {
        branch: _clone_value(value, detach=not preserve_graph)
        for branch, value in states.items()
    }
    return RolloutCollection(
        trajectories,
        proposals,
        prefix_steps,
        total_steps,
        terminal_states,
        preserve_graph,
        prefix_holder.auxiliary_transitions,
    )


def replay_shared_prefix(
    initial_state: Any,
    prefix_steps: int,
    step_fn: Callable[[Any, Optional[Tensor], int], Any],
    branch_actions: Mapping[str, Iterable[Optional[Tensor]]],
    *,
    state_clone_fn: Optional[Callable[[Any], Any]] = None,
) -> dict[str, BranchTrajectory]:
    """Replay one prefix once, then branch with registered actions.

    ``step_fn(state, action, index)`` returns a next state, or the legacy pair
    ``(next_state, Transition)``. A tuple is interpreted as that pair only when
    its second item is actually a :class:`Transition`; other tuples remain valid
    tuple-valued states. Prefix calls receive ``None`` and are executed once;
    each branch starts from the same prefix state. The helper is independent of
    diffusers so CPU tests can prove that branch code does not recompute the
    prefix.
    """
    if prefix_steps < 0:
        raise ValueError("prefix_steps must be non-negative")
    state = initial_state
    for index in range(prefix_steps):
        result = step_fn(state, None, index)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(
            result[1], Transition
        ):
            state = result[0]
        else:
            state = result
    def clone_state(value: Any) -> Any:
        if state_clone_fn is not None:
            return state_clone_fn(value)
        if isinstance(value, Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {key: clone_state(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone_state(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone_state(item) for item in value)
        if isinstance(value, set):
            return {clone_state(item) for item in value}
        if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
            return value
        try:
            return copy.deepcopy(value)
        except Exception as exc:
            # Never silently alias an unknown mutable state.  Callers with a
            # custom state object must provide an explicit clone hook.
            raise TypeError(
                "state cannot be isolated; provide state_clone_fn"
            ) from exc

    trajectories: dict[str, BranchTrajectory] = {}
    for branch_id, actions in branch_actions.items():
        branch_state = clone_state(state)
        trajectory = BranchTrajectory(branch_id=branch_id)
        for offset, action in enumerate(actions, start=prefix_steps):
            result = step_fn(branch_state, action, offset)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(
                result[1], Transition
            ):
                branch_state, transition = result
                trajectory.append(transition)
            else:
                branch_state = result
        trajectories[branch_id] = trajectory
    return trajectories
