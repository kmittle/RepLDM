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
    reward_gradient: Callable[[Tensor, Tensor], Tensor],
    config: TargetStepConfig = TargetStepConfig(),
    contract: ActionSpaceContract | None = None,
    candidate_validator: Callable[[Tensor], Tensor | bool] | None = None,
) -> RewardTargetPair:
    """Construct detached positive/negative targets from one anchor gradient.

    ``reward_gradient`` is an injected, budgeted operation. Formal runs bind it
    to ``SdxlTrainingAdapter.reward_gradient``; isolated tests may inject local
    autograd. The reward and its gradient are evaluated exactly once at the
    detached anchor. Both signs and all deterministic projection substeps reuse
    that direction, so one logical state consumes one reward backward.
    """
    if (
        anchor_u.ndim < 2
        or not anchor_u.is_floating_point()
        or not torch.isfinite(anchor_u).all()
    ):
        raise ValueError("anchor_u must be finite and have batch and slot dimensions")
    if not callable(clean_from_u) or not callable(reward) or not callable(reward_gradient):
        raise TypeError("clean_from_u, reward, and reward_gradient must be callable")
    if candidate_validator is not None and not callable(candidate_validator):
        raise TypeError("candidate_validator must be callable or None")
    if contract is not None:
        if candidate_validator is None:
            raise ValueError(
                "an action contract requires an explicit candidate validator"
            )
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

    variable = anchor.detach().clone().requires_grad_(True)
    value = reward(clean_from_u(variable))
    if not isinstance(value, Tensor):
        raise TypeError("reward must return a Tensor")
    if value.ndim == 0:
        if anchor.shape[0] != 1:
            raise ValueError("reward must return one value per batch item")
        value_rows = value.reshape(1)
    elif value.ndim == 1:
        if value.shape[0] != anchor.shape[0]:
            raise ValueError("reward must return one value per batch item")
        # Preserve the exact tensor object returned by the ledgered reward
        # adapter. Its in-process receipt capability is consumed by the one
        # authorized backward call below.
        value_rows = value
    else:
        if value.shape[0] != anchor.shape[0]:
            raise ValueError("reward must return one value per batch item")
        value_rows = value.reshape(anchor.shape[0], -1).mean(dim=1)
    finite_value = torch.isfinite(value_rows)
    gradient = reward_gradient(value_rows, variable)
    if not isinstance(gradient, Tensor):
        raise TypeError("reward_gradient must return a Tensor")
    if gradient.shape != variable.shape:
        raise ValueError("reward gradient shape does not match the action batch")
    if not gradient.is_floating_point():
        raise ValueError("reward gradient must use floating point")
    if gradient.device != variable.device or gradient.dtype != variable.dtype:
        raise ValueError("reward gradient must match the action device and dtype")
    gradient = gradient.detach().to(dtype=calc_dtype)
    finite_gradient = torch.isfinite(gradient).flatten(1).all(dim=1)
    safe_gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    if active_calc is not None:
        safe_gradient = safe_gradient * active_calc
    norm = torch.linalg.vector_norm(safe_gradient.flatten(1), dim=-1, keepdim=True)
    gradient_valid = finite_value & finite_gradient & (norm[:, 0] > config.epsilon_grad)
    shape = (norm.shape[0],) + (1,) * (safe_gradient.ndim - 1)
    direction = safe_gradient / norm.clamp_min(config.epsilon_grad).reshape(shape)
    exported_gradient = safe_gradient.to(dtype=anchor.dtype).detach()

    def walk(sign: float) -> tuple[Tensor, Tensor]:
        current = anchor.clone()
        valid_rows = gradient_valid.clone()
        for _ in range(config.target_steps):
            anchor_calc = anchor.to(dtype=calc_dtype)
            current_calc = current.to(dtype=calc_dtype)
            chosen = current_calc.clone()
            chosen_valid = torch.zeros_like(valid_rows)
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
                        if not isinstance(external, Tensor) or external.dtype != torch.bool:
                            raise TypeError(
                                "candidate validator must return bool flags"
                            )
                        if external.shape != candidate_ok.shape:
                            raise ValueError("candidate validator must return one flag per batch item")
                        candidate_ok = candidate_ok & external.to(device=anchor.device, dtype=torch.bool)
                take = candidate_ok & valid_rows & ~chosen_valid
                take_shape = (take.shape[0],) + (1,) * (candidate.ndim - 1)
                chosen = torch.where(take.reshape(take_shape), candidate, chosen)
                chosen_valid = chosen_valid | take
            current = chosen.to(dtype=anchor.dtype).detach()
            if active is not None:
                current = current * active
            valid_rows = (
                valid_rows & chosen_valid & torch.isfinite(current).flatten(1).all(dim=1)
            )
        return current, valid_rows

    plus, plus_valid = walk(+1.0)
    minus, minus_valid = walk(-1.0)
    valid = plus_valid & minus_valid
    valid_shape = (valid.shape[0],) + (1,) * (anchor.ndim - 1)
    plus = torch.where(valid.reshape(valid_shape), plus, anchor)
    minus = torch.where(valid.reshape(valid_shape), minus, anchor)
    exported_gradient = torch.where(
        valid.reshape(valid_shape), exported_gradient, torch.zeros_like(exported_gradient)
    )
    return RewardTargetPair(anchor, plus, minus, exported_gradient, valid)
