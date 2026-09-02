"""Method-agnostic optimizer loop for registered renderer objectives."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import numbers
from typing import Any, Callable, Iterable, Mapping, Optional

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .authorization import TrainingAuthorization


@dataclass(frozen=True)
class UpdateRecord:
    step: int
    loss: float
    gradient_norm: float


class RendererTrainer:
    """Run explicit updates; collection and objective construction stay external."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, contract: Mapping[str, Any], authorization: TrainingAuthorization | None = None, grad_norm_cap: float = 1.0, ema_decay: float = 0.995, scheduler: Any | None = None) -> None:
        if grad_norm_cap <= 0 or not 0 < ema_decay < 1:
            raise ValueError("grad_norm_cap must be positive and ema_decay must be in (0,1)")
        self.model = model
        self.optimizer = optimizer
        self.contract = dict(contract)
        if authorization is not None and (
            type(authorization) is not TrainingAuthorization
            or not authorization.is_validated()
        ):
            raise TypeError("authorization must be loaded from a validated receipt")
        self.authorization = authorization
        self.grad_norm_cap = float(grad_norm_cap)
        self.ema_decay = float(ema_decay)
        self.scheduler = scheduler
        self.step = 0
        self.ema_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def update(self, loss_fn: Callable[[], torch.Tensor]) -> UpdateRecord:
        if (
            type(self.authorization) is not TrainingAuthorization
            or not self.authorization.is_validated()
        ):
            raise RuntimeError(
                "training requires a validated TrainingAuthorization receipt"
            )
        self.authorization.validate_current()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("objective must return one finite scalar")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_cap)
        if not torch.isfinite(torch.as_tensor(norm)):
            raise ValueError("gradient norm is non-finite")
        before_model = {
            name: value.detach().clone() for name, value in self.model.state_dict().items()
        }
        before_optimizer = copy.deepcopy(self.optimizer.state_dict())
        before_scheduler = (
            copy.deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else None
        )
        before_ema = {name: value.detach().clone() for name, value in self.ema_state.items()}

        def all_finite(value: Any) -> bool:
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
            if isinstance(value, Mapping):
                return all(all_finite(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return all(all_finite(item) for item in value)
            if isinstance(value, numbers.Real) and not isinstance(value, bool):
                return math.isfinite(float(value))
            return True

        try:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if not all_finite(self.model.state_dict()) or not all_finite(
                self.optimizer.state_dict()
            ) or (
                self.scheduler is not None and not all_finite(self.scheduler.state_dict())
            ):
                raise ValueError("optimizer step produced non-finite state")
            with torch.no_grad():
                for name, value in self.model.state_dict().items():
                    if self.ema_state[name].is_floating_point():
                        self.ema_state[name].mul_(self.ema_decay).add_(
                            value.detach(), alpha=1 - self.ema_decay
                        )
                    else:
                        self.ema_state[name].copy_(value.detach())
            if not all_finite(self.ema_state):
                raise ValueError("EMA update produced non-finite state")
        except Exception:
            try:
                self.model.load_state_dict(before_model, strict=True)
                self.optimizer.load_state_dict(before_optimizer)
                if self.scheduler is not None and before_scheduler is not None:
                    self.scheduler.load_state_dict(before_scheduler)
                self.ema_state = before_ema
            except Exception as rollback_error:
                raise RuntimeError("failed to roll back invalid optimizer update") from rollback_error
            raise
        self.step += 1
        return UpdateRecord(self.step, float(loss.detach()), float(norm))

    def save(self, path: str, *, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            contract=self.contract,
            extra={"ema_state": self.ema_state, **dict(extra or {})},
        )

    def load(self, path: str, *, expected_contract: Mapping[str, Any] | None = None, map_location: str | torch.device = "cpu", restore_rng: bool = True) -> dict[str, Any]:
        """Resume model, optimizer, scheduler, EMA, step, and RNG together."""
        if expected_contract is not None:
            if not isinstance(expected_contract, Mapping):
                raise ValueError("expected contract must be a mapping")
            if dict(expected_contract) != self.contract:
                raise ValueError("expected contract does not match trainer contract")
        return load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            trainer=self,
            expected_contract=self.contract,
            map_location=map_location,
            restore_rng=restore_rng,
        )


def run_updates(trainer: RendererTrainer, losses: Iterable[Callable[[], torch.Tensor]], *, checkpoint_path: Optional[str] = None) -> list[UpdateRecord]:
    records = []
    for loss_fn in losses:
        records.append(trainer.update(loss_fn))
        if checkpoint_path is not None:
            trainer.save(checkpoint_path)
    return records
