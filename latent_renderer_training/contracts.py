"""Immutable contracts shared by renderer training objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ActionSpaceContract:
    """Fixed transformed-Gaussian action space with inactive Dirac slots."""

    num_slots: int
    active_mask: tuple[bool, ...]
    coefficient_bound: float = 1.0
    pre_squash_sigma: float = 0.25
    schema: str = "repldm.action_space.v1"
    measure: str = "active_lebesgue_inactive_dirac"

    def __post_init__(self) -> None:
        if isinstance(self.num_slots, bool) or not isinstance(self.num_slots, int):
            raise ValueError("num_slots must be a positive integer")
        if self.num_slots <= 0:
            raise ValueError("active_mask must contain one entry per positive slot")
        try:
            mask = tuple(self.active_mask)
        except TypeError as exc:
            raise ValueError("active_mask must be an iterable of booleans") from exc
        if len(mask) != self.num_slots or any(not isinstance(value, bool) for value in mask):
            raise ValueError("active_mask must contain one boolean per positive slot")
        # Frozen dataclasses do not freeze a list supplied to the constructor.
        # Normalize it to an immutable tuple before exposing the contract.
        object.__setattr__(self, "active_mask", mask)
        if not any(mask):
            raise ValueError("at least one action slot must be active")
        if isinstance(self.coefficient_bound, bool) or not isinstance(
            self.coefficient_bound, (int, float)
        ):
            raise ValueError("coefficient_bound must be finite and positive")
        if not math.isfinite(self.coefficient_bound) or self.coefficient_bound <= 0:
            raise ValueError("coefficient_bound must be finite and positive")
        object.__setattr__(self, "coefficient_bound", float(self.coefficient_bound))
        if isinstance(self.pre_squash_sigma, bool) or not isinstance(
            self.pre_squash_sigma, (int, float)
        ):
            raise ValueError("pre_squash_sigma must be finite and positive")
        if not math.isfinite(self.pre_squash_sigma) or self.pre_squash_sigma <= 0:
            raise ValueError("pre_squash_sigma must be finite and positive")
        object.__setattr__(self, "pre_squash_sigma", float(self.pre_squash_sigma))
        if self.measure != "active_lebesgue_inactive_dirac":
            raise ValueError("unsupported action-space measure")

    @property
    def rank(self) -> int:
        return sum(self.active_mask)

    @property
    def inactive_mask(self) -> tuple[bool, ...]:
        return tuple(not value for value in self.active_mask)

    @property
    def mask_hash(self) -> str:
        return hashlib.sha256(_canonical(list(self.active_mask))).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_mask"] = list(self.active_mask)
        payload["rank"] = self.rank
        payload["mask_hash"] = self.mask_hash
        return payload

    def validate_action(self, action: Any, *, atol: float = 1e-7) -> None:
        import torch

        value = torch.as_tensor(action)
        if not value.is_floating_point():
            raise ValueError("action must use a floating-point dtype")
        if value.ndim < 1 or value.shape[-1] != self.num_slots:
            raise ValueError("action has the wrong number of slots")
        if not torch.isfinite(value).all():
            raise ValueError("action contains non-finite values")
        active = torch.as_tensor(self.active_mask, device=value.device, dtype=torch.bool)
        inactive = value[..., ~active] if (~active).any() else value[..., :0]
        # Inactive coordinates use a Dirac measure.  Approximate zeros would
        # silently create probability mass outside the registered action space.
        if inactive.numel() and not torch.all(inactive == 0):
            raise ValueError("inactive action slots must be exact zero")
        selected = value[..., active]
        if torch.any(selected.abs() >= self.coefficient_bound):
            raise ValueError("active actions must be strictly inside coefficient_bound")

    def action_valid_mask(self, action: Any) -> Any:
        """Return a per-row validity mask without raising for bad samples.

        The training objectives use this helper to fail closed on an individual
        malformed rollout while charging its query.  Shape errors remain hard
        errors because they indicate a programming bug rather than a bad row.
        """
        import torch

        value = torch.as_tensor(action)
        if not value.is_floating_point():
            raise ValueError("action must use a floating-point dtype")
        if value.ndim == 0 or value.shape[-1] != self.num_slots:
            raise ValueError("action has the wrong number of slots")
        valid = torch.isfinite(value).all(dim=-1)
        active = torch.as_tensor(self.active_mask, device=value.device, dtype=torch.bool)
        inactive = value[..., ~active] if (~active).any() else value[..., :0]
        if inactive.numel():
            valid = valid & torch.all(inactive == 0, dim=-1)
        selected = value[..., active]
        if selected.numel():
            valid = valid & torch.all(selected.abs() < self.coefficient_bound, dim=-1)
        return valid


def contract_hash(contract: ActionSpaceContract | dict[str, Any]) -> str:
    payload = contract.to_dict() if isinstance(contract, ActionSpaceContract) else contract
    return hashlib.sha256(_canonical(payload)).hexdigest()


def make_mask(values: Iterable[bool], *, coefficient_bound: float = 1.0, sigma: float = 0.25) -> ActionSpaceContract:
    mask = tuple(values)
    if any(not isinstance(value, bool) for value in mask):
        raise ValueError("mask values must be booleans")
    return ActionSpaceContract(len(mask), mask, coefficient_bound, sigma)
