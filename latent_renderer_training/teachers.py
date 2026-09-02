"""Reward-gradient target construction for an OPSD-style teacher."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import Tensor

from .contracts import ActionSpaceContract


@dataclass(frozen=True)
class TargetStepConfig:
    """Frozen constants for one bounded positive/negative target pair."""

    eta_target: float = 0.25
    target_steps: int = 2
    trust_radius_u: float = 0.50
    epsilon_grad: float = 1e-12
    backtracking: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)

    def __post_init__(self) -> None:
        if isinstance(self.target_steps, bool) or not isinstance(self.target_steps, int):
            raise ValueError("target_steps must be a positive integer")
        if (
            not math.isfinite(float(self.eta_target))
            or self.eta_target <= 0
            or self.target_steps <= 0
            or not math.isfinite(float(self.trust_radius_u))
            or self.trust_radius_u <= 0
        ):
            raise ValueError("target step constants must be positive")
        if not math.isfinite(float(self.epsilon_grad)) or self.epsilon_grad <= 0:
            raise ValueError("epsilon_grad must be positive")
        scales = tuple(self.backtracking)
        if not scales or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in scales
        ):
            raise ValueError("backtracking scales must be finite and positive")
        object.__setattr__(self, "backtracking", scales)


@dataclass(frozen=True)
class RewardTargetPair:
    anchor_u: Tensor
    plus_u: Tensor
    minus_u: Tensor
    gradient_u: Tensor
    valid: Tensor | None = None


def _project_ball(value: Tensor, centre: Tensor, radius: float) -> Tensor:
    delta = value - centre
    norm = torch.linalg.vector_norm(delta.flatten(1), dim=-1, keepdim=True)
    shape = (norm.shape[0],) + (1,) * (delta.ndim - 1)
    scale = (radius / norm.clamp_min(radius)).clamp_max(1.0).reshape(shape)
    return centre + delta * scale


def construct_reward_targets(
    anchor_u: Tensor,
    clean_from_u: Callable[[Tensor], Tensor],
    reward: Callable[[Tensor], Tensor],
    *,
    config: TargetStepConfig = TargetStepConfig(),
    contract: ActionSpaceContract | None = None,
    candidate_validator: Callable[[Tensor], Tensor | bool] | None = None,
) -> RewardTargetPair:
    """Construct detached positive/negative targets from a clean-latent reward.

    The reward graph is used only while differentiating the detached anchor;
    returned targets and gradients are detached, so fitting a student cannot
    backpropagate through the reward model or decoder.
    """
    if (
        anchor_u.ndim < 2
        or not anchor_u.is_floating_point()
        or not torch.isfinite(anchor_u).all()
    ):
        raise ValueError("anchor_u must be finite and have batch and slot dimensions")
    if contract is not None:
        if anchor_u.shape[-1] != contract.num_slots:
            raise ValueError("anchor_u does not match the action contract")
        inactive = anchor_u[..., ~torch.as_tensor(
            contract.active_mask, device=anchor_u.device, dtype=torch.bool
        )]
        if inactive.numel() and not torch.all(inactive == 0):
            raise ValueError("anchor_u inactive coordinates must be exact zero")
    anchor = anchor_u.detach()
    active = None
    if contract is not None:
        active = torch.as_tensor(
            contract.active_mask, device=anchor.device, dtype=anchor.dtype
        )

    calc_dtype = torch.float64 if anchor.dtype == torch.float64 else torch.float32
    active_calc = None if active is None else active.to(dtype=calc_dtype)

    def walk(sign: float) -> tuple[Tensor, Tensor, Tensor]:
        current = anchor.clone()
        first_gradient = torch.zeros_like(anchor, dtype=calc_dtype)
        valid_rows = torch.ones(anchor.shape[0], device=anchor.device, dtype=torch.bool)
        for _ in range(config.target_steps):
            variable = current.detach().clone().requires_grad_(True)
            value = reward(clean_from_u(variable))
            if value.ndim == 0:
                if anchor.shape[0] != 1:
                    raise ValueError("reward must return one value per batch item")
                value_rows = value.reshape(1)
            else:
                if value.shape[0] != anchor.shape[0]:
                    raise ValueError("reward must return one value per batch item")
                value_rows = value.reshape(anchor.shape[0], -1).mean(dim=1)
            finite_value = torch.isfinite(value_rows)
            safe_rows = torch.nan_to_num(value_rows, nan=0.0, posinf=0.0, neginf=0.0)
            gradient = torch.zeros_like(variable, dtype=calc_dtype)
            gradient_valid = torch.zeros(anchor.shape[0], device=anchor.device, dtype=torch.bool)
            for row in range(anchor.shape[0]):
                if not bool(finite_value[row]) or not safe_rows[row].requires_grad:
                    continue
                row_gradient = torch.autograd.grad(
                    safe_rows[row],
                    variable,
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if row_gradient is None:
                    continue
                row_gradient = row_gradient.to(dtype=calc_dtype)
                if row_gradient.shape != variable.shape:
                    raise ValueError("reward gradient shape does not match the action batch")
                # A reward that differentiates one sample through another makes
                # teacher labels depend on batch composition; reject it.
                cross = row_gradient.detach().clone()
                cross[row] = 0
                if not torch.isfinite(cross).all() or torch.any(cross.abs() > 1e-12):
                    raise ValueError("reward must be batch-independent")
                if not torch.isfinite(row_gradient[row]).all():
                    continue
                gradient[row] = row_gradient[row]
                gradient_valid[row] = True
            finite_gradient = torch.isfinite(gradient).flatten(1).all(dim=1)
            safe_gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
            if active_calc is not None:
                safe_gradient = safe_gradient * active_calc
            if _ == 0:
                # The exported teacher direction is part of the action
                # contract too; inactive Dirac coordinates must stay exact
                # zero, not merely be masked during target construction.
                first_gradient = safe_gradient.detach()
            norm = torch.linalg.vector_norm(
                safe_gradient.flatten(1), dim=-1, keepdim=True
            )
            step_valid = (
                valid_rows
                & finite_value
                & gradient_valid
                & finite_gradient
                & (norm[:, 0] > config.epsilon_grad)
            )
            shape = (norm.shape[0],) + (1,) * (safe_gradient.ndim - 1)
            direction = safe_gradient / norm.clamp_min(config.epsilon_grad).reshape(shape)
            anchor_calc = anchor.to(dtype=calc_dtype)
            current_calc = current.to(dtype=calc_dtype)
            chosen = current_calc.clone()
            chosen_valid = torch.zeros_like(step_valid)
            for scale in config.backtracking:
                candidate = _project_ball(
                    current_calc
                    + sign
                    * direction
                    * (float(config.eta_target) / config.target_steps)
                    * float(scale),
                    anchor_calc,
                    float(config.trust_radius_u),
                )
                candidate_ok = torch.isfinite(candidate).flatten(1).all(dim=1)
                if candidate_validator is not None:
                    external = candidate_validator(candidate.to(dtype=anchor.dtype))
                    if isinstance(external, bool):
                        candidate_ok = candidate_ok & external
                    else:
                        if external.shape != candidate_ok.shape:
                            raise ValueError("candidate validator must return one flag per batch item")
                        candidate_ok = candidate_ok & external.to(device=anchor.device, dtype=torch.bool)
                take = candidate_ok & step_valid & ~chosen_valid
                take_shape = (take.shape[0],) + (1,) * (candidate.ndim - 1)
                chosen = torch.where(take.reshape(take_shape), candidate, chosen)
                chosen_valid = chosen_valid | take
            current = chosen.to(dtype=anchor.dtype).detach()
            if active is not None:
                current = current * active
            valid_rows = step_valid & chosen_valid & torch.isfinite(current).flatten(1).all(dim=1)
        return current, first_gradient.to(dtype=anchor.dtype), valid_rows

    plus, gradient, plus_valid = walk(+1.0)
    minus, _, minus_valid = walk(-1.0)
    valid = plus_valid & minus_valid
    return RewardTargetPair(anchor, plus, minus, gradient, valid)
