"""Exact transformed-Gaussian probabilities on the fixed action space."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .contracts import ActionSpaceContract


class SquashedGaussian:
    """A diagonal Normal transformed by coefficient_bound times tanh."""

    def __init__(
        self, mean: Tensor, contract: ActionSpaceContract, *, strict: bool = True
    ) -> None:
        if mean.ndim < 1 or mean.shape[-1] != contract.num_slots:
            raise ValueError("mean has the wrong number of slots")
        if not mean.is_floating_point():
            raise ValueError("mean must use a floating-point dtype")
        if strict and not torch.isfinite(mean).all():
            raise ValueError("mean contains non-finite values")
        self.mean = mean
        self.contract = contract
        self.strict = bool(strict)

    def rsample(self, noise: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        active_bool = torch.as_tensor(
            self.contract.active_mask, device=self.mean.device, dtype=torch.bool
        )
        active_mean = self.mean[..., active_bool]
        if noise is None:
            # Inactive Dirac coordinates are not sampled and therefore do not
            # consume RNG draws.  This keeps paired methods bit-for-bit aligned.
            active_noise = torch.randn_like(active_mean)
        else:
            if not isinstance(noise, Tensor) or noise.shape != self.mean.shape:
                raise ValueError("noise shape must match mean")
            if not noise.is_floating_point() or not torch.isfinite(noise).all():
                raise ValueError("noise must be finite and floating-point")
            active_noise = noise[..., active_bool]
        calc_dtype = torch.float64 if self.mean.dtype == torch.float64 else torch.float32
        u_active = active_mean.to(calc_dtype) + float(
            self.contract.pre_squash_sigma
        ) * active_noise.to(calc_dtype)
        action_active = (float(self.contract.coefficient_bound) * torch.tanh(u_active)).to(
            dtype=self.mean.dtype
        )
        # Rounding in fp16/fp32 can turn a large finite tanh output into the
        # exact open-bound endpoint, which has no finite transformed density.
        bound = torch.tensor(
            self.contract.coefficient_bound, device=self.mean.device, dtype=self.mean.dtype
        )
        try:
            interior = torch.nextafter(bound, torch.zeros_like(bound))
        except RuntimeError:
            interior = bound * (1 - 2 * torch.finfo(self.mean.dtype).eps)
        action_active = action_active.clamp(-interior, interior)
        u = torch.zeros_like(self.mean)
        action = torch.zeros_like(self.mean)
        u[..., active_bool] = u_active.to(dtype=self.mean.dtype)
        action[..., active_bool] = action_active
        return action, u

    def log_prob(
        self,
        action: Tensor,
        *,
        eps: float = 1e-6,
        strict: Optional[bool] = None,
        return_valid: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        strict = self.strict if strict is None else bool(strict)
        if (
            not isinstance(action, Tensor)
            or action.shape != self.mean.shape
            or not action.is_floating_point()
        ):
            raise ValueError("action must have the same shape as mean")
        if not torch.isfinite(torch.tensor(eps)) or eps <= 0 or eps >= 1:
            raise ValueError("eps must be finite and lie in (0, 1)")
        valid = self.contract.action_valid_mask(action)
        valid = valid & torch.isfinite(self.mean).all(dim=-1)
        if strict and not bool(valid.all()):
            # Preserve the more useful contract-specific error for callers
            # that request strict validation.
            self.contract.validate_action(action, atol=eps)
            raise ValueError("mean contains non-finite values")
        active = torch.as_tensor(self.contract.active_mask, device=action.device, dtype=torch.bool)
        # Use float64 for the quadratic so finite fp32 inputs cannot overflow
        # merely because the action mean is large.
        safe_action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        selected = (
            safe_action[..., active].double() / float(self.contract.coefficient_bound)
        ).clamp(-1 + eps, 1 - eps)
        u = torch.atanh(selected)
        safe_mean = torch.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)
        mean = safe_mean[..., active].double()
        sigma = float(self.contract.pre_squash_sigma)
        normal = (
            -0.5 * ((u - mean) / sigma).square()
            - math.log(sigma)
            - 0.5 * math.log(2 * math.pi)
        )
        log_det = math.log(self.contract.coefficient_bound) + torch.log1p(-selected.square())
        value = (normal - log_det).sum(dim=-1)
        finite = torch.isfinite(value)
        valid = valid & finite
        value = torch.where(finite, value, torch.zeros_like(value))
        value = torch.where(valid, value, torch.zeros_like(value))
        # Non-strict callers need the wider intermediate dtype so a finite
        # fp64 log density is not turned into fp32 -inf before a mask can be
        # applied by the objective.
        if strict:
            output = value.to(dtype=self.mean.dtype)
            # A finite float64 density can still be outside the representable
            # range of the policy's dtype.  Treat that row as invalid instead
            # of leaking an Inf into a strict caller.
            valid = valid & torch.isfinite(output)
            output = torch.where(valid, output, torch.zeros_like(output))
            if not bool(valid.all()):
                raise ValueError("log probability is non-finite")
        else:
            output = value
        if return_valid:
            return output, valid
        return output

    def log_prob_u(self, pre_squash: Tensor) -> Tensor:
        if (
            pre_squash.ndim < 1
            or pre_squash.shape[-1] != self.contract.num_slots
            or not pre_squash.is_floating_point()
        ):
            raise ValueError("pre_squash has the wrong number of slots")
        active = torch.as_tensor(self.contract.active_mask, device=pre_squash.device, dtype=torch.bool)
        if not torch.isfinite(pre_squash).all():
            raise ValueError("pre_squash contains non-finite values")
        inactive = pre_squash[..., ~active] if (~active).any() else pre_squash[..., :0]
        if inactive.numel() and not torch.all(inactive == 0):
            raise ValueError("inactive pre_squash coordinates must be exact zero")
        sigma = float(self.contract.pre_squash_sigma)
        value = pre_squash[..., active].double()
        mean = self.mean[..., active].double()
        result = (
            -0.5 * ((value - mean) / sigma).square()
            - math.log(sigma)
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError("log probability is non-finite")
        output = result.to(dtype=self.mean.dtype)
        if not torch.isfinite(output).all():
            raise ValueError("log probability is outside the mean dtype range")
        return output


def transformed_gaussian_kl(
    mean: Tensor,
    reference_mean: Tensor,
    contract: ActionSpaceContract,
    *,
    reduction: str = "sum",
    strict: bool = True,
    return_valid: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """KL in shared pre-squash coordinates; the bijective transform cancels.

    Sum is the usual diagonal-Gaussian KL. Mean is provided for the
    protocol's per-active-coordinate reference penalty.
    """
    if (
        mean.shape != reference_mean.shape
        or mean.ndim < 1
        or mean.shape[-1] != contract.num_slots
        or not mean.is_floating_point()
        or not reference_mean.is_floating_point()
    ):
        raise ValueError("KL means must have equal shape and match the contract")
    if mean.device != reference_mean.device:
        raise ValueError("KL means must be on the same device")
    valid = torch.isfinite(mean).all(dim=-1) & torch.isfinite(reference_mean).all(dim=-1)
    if strict and not bool(valid.all()):
        raise ValueError("KL means must be finite")
    active = torch.as_tensor(contract.active_mask, device=mean.device, dtype=torch.bool)
    sigma = contract.pre_squash_sigma
    safe_mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    safe_reference = torch.nan_to_num(reference_mean, nan=0.0, posinf=0.0, neginf=0.0)
    values = 0.5 * (
        (safe_mean[..., active].double() - safe_reference[..., active].double()) / sigma
    ).square()
    finite = torch.isfinite(values).all(dim=-1)
    valid = valid & finite
    if strict and not bool(valid.all()):
        raise ValueError("KL is non-finite")
    if reduction == "sum":
        result = values.sum(dim=-1)
    elif reduction == "mean":
        result = values.mean(dim=-1)
    else:
        raise ValueError("reduction must be sum or mean")
    finite_result = torch.isfinite(result)
    valid = valid & finite_result
    result = torch.where(valid, result, torch.zeros_like(result))
    if strict:
        output = result.to(dtype=mean.dtype)
        valid = valid & torch.isfinite(output)
        output = torch.where(valid, output, torch.zeros_like(output))
        if not bool(valid.all()):
            raise ValueError("KL is outside the mean dtype range")
    else:
        output = result
    if return_valid:
        return output, valid
    return output
