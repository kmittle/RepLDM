"""Framework-neutral trajectory records and shared-prefix replay."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any, Callable, Iterable, Mapping, Optional

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


def _snapshot(value: Any) -> Any:
    """Detach and copy rollout fields so later state mutation cannot rewrite history."""
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, set):
        return {_snapshot(item) for item in value}
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise TypeError("transition field cannot be snapshotted") from exc


@dataclass
class BranchTrajectory:
    """A branch with auditable decisions and one terminal reward."""

    branch_id: str
    transitions: list[Transition] = field(default_factory=list)
    terminal_reward: Optional[Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise TypeError("trajectory transitions must be Transition instances")
        self.transitions.append(
            Transition(
                state=_snapshot(transition.state),
                action=_snapshot(transition.action),
                next_state=_snapshot(transition.next_state),
                native_transition=_snapshot(transition.native_transition),
                rendered_transition=_snapshot(transition.rendered_transition),
                log_prob=_snapshot(transition.log_prob),
                step_index=transition.step_index,
            )
        )

    @property
    def actions(self) -> Tensor:
        if not self.transitions:
            raise ValueError("trajectory has no transitions")
        return torch.stack([item.action for item in self.transitions], dim=-2)


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
