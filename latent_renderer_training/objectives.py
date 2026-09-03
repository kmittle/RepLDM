"""Objective functions for OPD, search distillation, DPO, and RL."""

from __future__ import annotations

from typing import Optional
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .contracts import ActionSpaceContract
from .distributions import SquashedGaussian, transformed_gaussian_kl


def _finite_rows(value: Tensor) -> Tensor:
    """Return one finite-value flag for each leading batch row."""
    if value.ndim < 1:
        raise ValueError("objective tensors must have a batch dimension")
    if value.ndim == 1:
        return torch.isfinite(value)
    return torch.isfinite(value).flatten(1).all(dim=1)


def _require_same_batch(*values: Tensor, label: str) -> int:
    """Require tensor inputs to describe the same leading batch dimension."""
    if not values or any(not isinstance(value, Tensor) or value.ndim < 1 for value in values):
        raise ValueError(f"{label} tensors must have a batch dimension")
    batch = values[0].shape[0]
    if any(value.shape[0] != batch for value in values[1:]):
        raise ValueError(f"{label} tensors must have the same batch size")
    return batch


def _safe_zero(value: Tensor) -> Tensor:
    """Make a detached zero anchor without allowing NaN to poison gradients."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0


def _anchor_losses(
    mean: Tensor, reference_mean: Tensor, contract: ActionSpaceContract
) -> tuple[Tensor, Tensor]:
    """Return finite per-row reference penalties and their validity mask."""
    if mean.shape != reference_mean.shape or mean.ndim < 2:
        raise ValueError("reference means must have equal shape and a batch dimension")
    if mean.shape[-1] != contract.num_slots:
        raise ValueError("reference means must match the action contract")
    active = torch.as_tensor(contract.active_mask, device=mean.device, dtype=torch.bool)
    valid = _finite_rows(mean) & _finite_rows(reference_mean)
    safe_mean = torch.nan_to_num(mean.double(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_reference = torch.nan_to_num(
        reference_mean.detach().double(), nan=0.0, posinf=0.0, neginf=0.0
    )
    values = (safe_mean[..., active] - safe_reference[..., active]).square()
    values = values.flatten(1).mean(dim=1)
    valid = valid & torch.isfinite(values)
    if mean.dtype.is_floating_point and mean.dtype != torch.float64:
        valid = valid & (values <= torch.finfo(mean.dtype).max)
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values, valid


def _valid_weight(values: Tensor, valid: Tensor, weight: Optional[Tensor] = None) -> Tensor:
    if valid.ndim != 1 or valid.numel() != values.shape[0]:
        raise ValueError("validity mask must have one value per batch item")
    result = valid.to(device=values.device, dtype=values.dtype)
    if weight is not None:
        if weight.ndim != 1 or weight.numel() != values.shape[0]:
            raise ValueError("sample weight must have one value per batch item")
        weight = weight.to(device=values.device, dtype=values.dtype)
        if not torch.isfinite(weight).all() or torch.any(weight < 0):
            raise ValueError("sample weight must be finite and non-negative")
        result = result * weight
    return result


def _transition_valid_rows(
    predicted: Tensor,
    target: Tensor,
    nominal_update: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return the fail-closed validity mask shared by transition objectives.

    A transition with no measurable native scheduler update is not a valid
    decision.  Keeping this predicate in one place prevents OPD/search
    distillation from accidentally applying their reference anchor penalty to
    a row that the normalized transition metric already rejected.
    """
    row_valid = (
        torch.isfinite(predicted).flatten(1).all(dim=1)
        & torch.isfinite(target).flatten(1).all(dim=1)
        & torch.isfinite(nominal_update).flatten(1).all(dim=1)
    )
    native_norm = torch.linalg.vector_norm(nominal_update.float().flatten(1), dim=1)
    row_valid = row_valid & (native_norm >= 1e-6)
    if valid_mask is not None:
        if (
            not isinstance(valid_mask, Tensor)
            or valid_mask.ndim != 1
            or valid_mask.numel() != predicted.shape[0]
        ):
            raise ValueError("valid_mask must have one boolean value per transition row")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        row_valid = row_valid & valid_mask.to(device=predicted.device)
    return row_valid


def normalized_transition_loss(
    predicted: Tensor,
    target: Tensor,
    nominal_update: Tensor,
    *,
    eps: float = 1e-12,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Native scheduler-transition MSE normalized by nominal update energy."""
    return normalized_transition_losses(
        predicted, target, nominal_update, eps=eps, valid_mask=valid_mask
    ).mean()


def normalized_transition_losses(
    predicted: Tensor,
    target: Tensor,
    nominal_update: Tensor,
    *,
    eps: float = 1e-12,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return one normalized transition loss per leading batch item."""
    if predicted.shape != target.shape or predicted.shape != nominal_update.shape:
        raise ValueError("transition tensors must have equal shape")
    if predicted.ndim < 2:
        raise ValueError("transition tensors must have a batch dimension")
    if not torch.isfinite(torch.tensor(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    # A malformed rollout is charged but contributes zero loss.  Sanitize
    # before arithmetic so an invalid row cannot poison a whole batch through
    # NaN propagation or float64 overflow.
    # A near-zero native transition has no meaningful scheduler metric.  It
    # must be rejected as a decision rather than converted into a zero
    # denominator and allowed to contribute an anchor or policy loss.
    row_valid = _transition_valid_rows(
        predicted, target, nominal_update, valid_mask=valid_mask
    )
    pred = torch.nan_to_num(predicted.double(), nan=0.0, posinf=0.0, neginf=0.0)
    goal = torch.nan_to_num(target.double(), nan=0.0, posinf=0.0, neginf=0.0)
    native = torch.nan_to_num(
        nominal_update.double(), nan=0.0, posinf=0.0, neginf=0.0
    )
    denominator = native.square().flatten(1).mean(dim=1).clamp_min(float(eps))
    numerator = (pred - goal).square().flatten(1).mean(dim=1)
    result = numerator / denominator
    row_valid = row_valid & torch.isfinite(denominator) & torch.isfinite(numerator)
    row_valid = row_valid & torch.isfinite(result)
    output_dtype = predicted.dtype if predicted.dtype.is_floating_point else torch.float32
    if output_dtype != torch.float64:
        row_valid = row_valid & (result <= torch.finfo(output_dtype).max)
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(row_valid, result, torch.zeros_like(result)).to(dtype=output_dtype)


def _weighted_mean(values: Tensor, weight: Optional[Tensor]) -> Tensor:
    """Reduce per-sample values without allowing tied samples to leak loss."""
    if weight is None:
        return values.mean()
    if weight.ndim != 1 or weight.numel() != values.shape[0]:
        raise ValueError("sample weight must have one value per batch item")
    weight = weight.to(device=values.device, dtype=values.dtype)
    if not torch.isfinite(weight).all() or torch.any(weight < 0):
        raise ValueError("sample weight must be finite and non-negative")
    total = weight.sum()
    if total <= 0:
        return values.sum() * 0.0
    return (values * weight).sum() / total


def _resolve_pair_weight(
    batch: int,
    *,
    sample_weight: Optional[Tensor],
    tie_mask: Optional[Tensor],
    device: torch.device,
) -> Optional[Tensor]:
    if sample_weight is not None and tie_mask is not None:
        raise ValueError("provide sample_weight or tie_mask, not both")
    if tie_mask is not None:
        if tie_mask.ndim != 1 or tie_mask.numel() != batch:
            raise ValueError("tie_mask must have one value per batch item")
        if tie_mask.dtype != torch.bool:
            raise ValueError("tie_mask must be boolean")
        return (~tie_mask.to(device=device)).float()
    if sample_weight is None:
        return None
    if sample_weight.ndim != 1 or sample_weight.numel() != batch:
        raise ValueError("sample_weight must have one value per batch item")
    return sample_weight.to(device=device)


def _density(
    mean: Tensor,
    action: Tensor,
    contract: ActionSpaceContract,
    pre_squash: Optional[Tensor],
    *,
    require_pre_squash: bool,
) -> tuple[Tensor, Tensor]:
    """Evaluate a policy density, retaining the exact sampled coordinate.

    A bounded action can lose information when converted back with ``atanh``.
    Formal rollouts therefore pass the recorded pre-squash value.  The action
    consistency check prevents a caller from pairing a valid density with a
    different action tensor.
    """
    if pre_squash is None:
        if require_pre_squash:
            raise ValueError("formal density evaluation requires pre_squash samples")
        return SquashedGaussian(mean, contract, strict=False).log_prob(
            action, strict=False, return_valid=True
        )
    if not isinstance(pre_squash, Tensor) or pre_squash.shape != action.shape:
        raise ValueError("pre_squash must have the same shape as action")
    fixed_pre_squash = pre_squash.detach()
    distribution = SquashedGaussian(mean, contract, strict=False)
    density, valid = distribution.log_prob_from_pre_squash(
        fixed_pre_squash, strict=False, return_valid=True
    )
    active = torch.as_tensor(contract.active_mask, device=action.device, dtype=torch.bool)
    reconstructed = torch.zeros_like(action)
    calculated = (
        float(contract.coefficient_bound)
        * torch.tanh(fixed_pre_squash[..., active])
    ).to(dtype=action.dtype)
    bound = torch.tensor(
        contract.coefficient_bound, device=action.device, dtype=action.dtype
    )
    try:
        interior = torch.nextafter(bound, torch.zeros_like(bound))
    except RuntimeError:
        interior = bound * (1 - 2 * torch.finfo(action.dtype).eps)
    reconstructed[..., active] = calculated.clamp(-interior, interior)
    # The sampler and this reconstruction use the same retained ``u`` bytes.
    # A different bounded action is not a valid observation of this density.
    consistent = torch.isfinite(action).all(dim=-1) & torch.isfinite(reconstructed).all(dim=-1)
    consistent = consistent & torch.eq(action, reconstructed).all(dim=-1)
    valid = valid & consistent
    density = torch.where(valid, density, torch.zeros_like(density))
    return density, valid


def opd_loss(
    predicted: Tensor,
    teacher_target: Tensor,
    nominal_update: Tensor,
    mean: Tensor,
    reference_mean: Tensor,
    contract: ActionSpaceContract,
    *,
    anchor_weight: float = 0.10,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Detached native-transition OPD target plus a fixed reference anchor."""
    if not math.isfinite(float(anchor_weight)) or anchor_weight < 0:
        raise ValueError("anchor_weight must be finite and non-negative")
    _require_same_batch(
        predicted,
        teacher_target,
        nominal_update,
        mean,
        reference_mean,
        label="OPD",
    )
    losses = normalized_transition_losses(
        predicted, teacher_target.detach(), nominal_update, valid_mask=valid_mask
    ).double()
    transition_valid = _transition_valid_rows(
        predicted, teacher_target, nominal_update, valid_mask=valid_mask
    ) & torch.isfinite(losses)
    anchor_values, anchor_valid = _anchor_losses(mean, reference_mean, contract)
    joint_valid = transition_valid & anchor_valid
    transition = _weighted_mean(
        torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0),
        _valid_weight(losses, joint_valid),
    )
    anchor = _weighted_mean(
        torch.nan_to_num(anchor_values, nan=0.0, posinf=0.0, neginf=0.0),
        _valid_weight(anchor_values, joint_valid),
    )
    result = transition + float(anchor_weight) * anchor
    output_dtype = predicted.dtype if predicted.dtype.is_floating_point else torch.float32
    return result.to(dtype=output_dtype)


def search_distill_loss(
    predicted: Tensor,
    chosen_target: Tensor,
    nominal_update: Tensor,
    mean: Tensor,
    reference_mean: Tensor,
    contract: ActionSpaceContract,
    *,
    chosen_weight: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Winner-only branch distillation; ties have zero target weight."""
    _require_same_batch(
        predicted,
        chosen_target,
        nominal_update,
        mean,
        reference_mean,
        label="search distillation",
    )
    losses = normalized_transition_losses(
        predicted, chosen_target.detach(), nominal_update, valid_mask=valid_mask
    ).double()
    transition_valid = _transition_valid_rows(
        predicted, chosen_target, nominal_update, valid_mask=valid_mask
    ) & torch.isfinite(losses)
    weights = None if chosen_weight is None else chosen_weight.detach().float()
    anchor_values, anchor_valid = _anchor_losses(mean, reference_mean, contract)
    joint_valid = transition_valid & anchor_valid
    loss = _weighted_mean(
        torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0),
        _valid_weight(losses, joint_valid, weights),
    )
    anchor = _weighted_mean(
        torch.nan_to_num(anchor_values, nan=0.0, posinf=0.0, neginf=0.0),
        _valid_weight(anchor_values, joint_valid, weights),
    )
    result = loss + 0.10 * anchor
    output_dtype = predicted.dtype if predicted.dtype.is_floating_point else torch.float32
    return result.to(dtype=output_dtype)


def dpo_loss(
    chosen_action: Tensor,
    rejected_action: Tensor,
    chosen_mean: Tensor,
    rejected_mean: Tensor,
    reference_chosen_mean: Tensor,
    reference_rejected_mean: Tensor,
    contract: ActionSpaceContract,
    *,
    beta: float = 0.10,
    sample_weight: Optional[Tensor] = None,
    tie_mask: Optional[Tensor] = None,
    chosen_pre_squash: Optional[Tensor] = None,
    rejected_pre_squash: Optional[Tensor] = None,
    require_pre_squash: bool = False,
) -> Tensor:
    """Reference-relative preference loss over matched branch trajectories."""
    if not torch.isfinite(torch.tensor(beta)) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    _require_same_batch(
        chosen_action,
        rejected_action,
        chosen_mean,
        rejected_mean,
        reference_chosen_mean,
        reference_rejected_mean,
        label="DPO",
    )
    if not (
        chosen_action.shape
        == rejected_action.shape
        == chosen_mean.shape
        == rejected_mean.shape
        == reference_chosen_mean.shape
        == reference_rejected_mean.shape
    ):
        raise ValueError("DPO tensors must have equal shapes")
    # Rollout actions are observations collected by a behavior policy.  They
    # must not carry a pathwise gradient into either policy's density.
    fixed_chosen = chosen_action.detach()
    fixed_rejected = rejected_action.detach()
    chosen_logp, chosen_valid = _density(
        chosen_mean, fixed_chosen, contract, chosen_pre_squash,
        require_pre_squash=require_pre_squash,
    )
    rejected_logp, rejected_valid = _density(
        rejected_mean, fixed_rejected, contract, rejected_pre_squash,
        require_pre_squash=require_pre_squash,
    )
    # The reference is frozen, and trajectory preferences are one margin per
    # trajectory rather than one independent sigmoid per decision.
    ref_chosen, ref_chosen_valid = _density(
        reference_chosen_mean.detach(), fixed_chosen, contract, chosen_pre_squash,
        require_pre_squash=require_pre_squash,
    )
    ref_rejected, ref_rejected_valid = _density(
        reference_rejected_mean.detach(), fixed_rejected, contract, rejected_pre_squash,
        require_pre_squash=require_pre_squash,
    )
    policy_delta = chosen_logp - rejected_logp
    reference_delta = ref_chosen - ref_rejected
    if policy_delta.ndim > 1:
        # A trajectory may contain several decision axes (for example
        # scheduler steps and action groups).  They are all part of one
        # preference example; reduce them to one margin per leading row.
        policy_delta = policy_delta.flatten(1).mean(dim=1)
        reference_delta = reference_delta.flatten(1).mean(dim=1)
    margin = float(beta) * (policy_delta - reference_delta)
    valid = chosen_valid & rejected_valid & ref_chosen_valid & ref_rejected_valid
    if valid.ndim > 1:
        valid = valid.flatten(1).all(dim=1)
    valid = valid & torch.isfinite(margin)
    margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
    weights = _resolve_pair_weight(
        margin.shape[0],
        sample_weight=None if sample_weight is None else sample_weight.detach(),
        tie_mask=tie_mask,
        device=margin.device,
    )
    validity_weight = valid.to(dtype=margin.dtype)
    weights = validity_weight if weights is None else weights * validity_weight
    return _weighted_mean(-F.logsigmoid(margin), weights)


def per_decision_rl_loss(
    actions: Tensor,
    behavior_mean: Tensor,
    policy_mean: Tensor,
    reference_mean: Tensor,
    advantages: Tensor,
    contract: ActionSpaceContract,
    *,
    clip_low: float = 0.8,
    clip_high: float = 1.2,
    kl_weight: float = 0.01,
    pre_squash: Optional[Tensor] = None,
    require_pre_squash: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """DDPO-style per-decision clipped ratios and reference KL."""
    if advantages.ndim == 0:
        raise ValueError("advantages must have one value per decision")
    _require_same_batch(
        actions, behavior_mean, policy_mean, reference_mean, label="RL"
    )
    if not (
        actions.shape == behavior_mean.shape == policy_mean.shape == reference_mean.shape
    ):
        raise ValueError("RL tensors must have equal shapes")
    if (
        not torch.isfinite(torch.tensor(clip_low))
        or not torch.isfinite(torch.tensor(clip_high))
        or clip_low <= 0
        or clip_low > clip_high
    ):
        raise ValueError("clip bounds must be finite and ordered")
    if not torch.isfinite(torch.tensor(kl_weight)) or kl_weight < 0:
        raise ValueError("kl_weight must be finite and non-negative")
    # PPO/DDPO treats the sampled behavior action as data.  Detaching here
    # prevents a reparameterized rollout from adding an unintended pathwise
    # gradient to the likelihood-ratio objective.
    fixed_actions = actions.detach()
    behavior_logp, behavior_valid = _density(
        behavior_mean.detach(), fixed_actions, contract, pre_squash,
        require_pre_squash=require_pre_squash,
    )
    policy_logp, policy_valid = _density(
        policy_mean, fixed_actions, contract, pre_squash,
        require_pre_squash=require_pre_squash,
    )
    log_ratio = policy_logp - behavior_logp
    valid = behavior_valid & policy_valid & torch.isfinite(log_ratio)
    log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    max_log = math.log(torch.finfo(torch.float64).max)
    valid = valid & (log_ratio.abs() <= max_log)
    ratio = torch.exp(torch.where(valid, log_ratio, torch.zeros_like(log_ratio)))
    if advantages.shape != ratio.shape:
        raise ValueError("advantages must have one value per decision")
    valid = valid & torch.isfinite(ratio)
    fixed_advantages = advantages.detach()
    valid = valid & torch.isfinite(fixed_advantages)
    safe_advantages = torch.nan_to_num(
        fixed_advantages, nan=0.0, posinf=0.0, neginf=0.0
    )
    ratio = torch.where(valid, ratio, torch.zeros_like(ratio))
    clipped = ratio.clamp(clip_low, clip_high)
    surrogate = torch.minimum(ratio * safe_advantages, clipped * safe_advantages)
    safe_policy_mean = torch.nan_to_num(
        policy_mean, nan=0.0, posinf=0.0, neginf=0.0
    )
    safe_reference_mean = torch.nan_to_num(
        reference_mean.detach(), nan=0.0, posinf=0.0, neginf=0.0
    )
    kl, kl_valid = transformed_gaussian_kl(
        safe_policy_mean,
        safe_reference_mean,
        contract,
        reduction="mean",
        strict=False,
        return_valid=True,
    )
    # Keep the original finite-row checks in addition to the sanitized KL:
    # sanitization prevents NaN gradients, while the mask still charges the
    # malformed rollout as zero-weight data.
    # Keep the decision axis intact: one bad timestep must not invalidate all
    # other decisions in the same trajectory.
    reference_valid = torch.isfinite(reference_mean).all(dim=-1)
    valid = valid & reference_valid & kl_valid
    ratio = torch.where(valid, ratio, torch.zeros_like(ratio))
    weight = valid.to(dtype=surrogate.dtype)
    denominator = weight.sum()
    if denominator <= 0:
        loss = _safe_zero(policy_mean)
    else:
        loss = -(surrogate * weight).sum() / denominator
        loss = loss + float(kl_weight) * (torch.where(valid, kl, torch.zeros_like(kl)) * weight).sum() / denominator
    output_dtype = policy_mean.dtype if policy_mean.dtype.is_floating_point else torch.float32
    return (
        loss.to(dtype=output_dtype),
        ratio.detach().to(dtype=output_dtype),
        torch.where(valid, kl, torch.zeros_like(kl)).detach().to(dtype=output_dtype),
    )
