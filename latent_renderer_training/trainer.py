"""Method-agnostic optimizer loop for registered renderer objectives."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import numbers
from typing import Any, Callable, Iterable, Mapping, Optional

import torch

from .artifacts import module_state_sha256
from .checkpoint import load_checkpoint, save_checkpoint
from .authorization import require_authorization_binding
from .operations import LedgeredOperationExecutor, OperationContext


@dataclass(frozen=True)
class UpdateRecord:
    step: int
    loss: float
    gradient_norm: float


class RendererTrainer:
    """Run explicit updates; collection and objective construction stay external."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        contract: Mapping[str, Any],
        authorization: Any = None,
        authorization_binding: Any = None,
        run_binding: Any = None,
        grad_norm_cap: float = 1.0,
        ema_decay: float = 0.995,
        scheduler: Any | None = None,
        operation_executor: LedgeredOperationExecutor | None = None,
    ) -> None:
        if grad_norm_cap <= 0 or not 0 < ema_decay < 1:
            raise ValueError("grad_norm_cap must be positive and ema_decay must be in (0,1)")
        self.model = model
        self.optimizer = optimizer
        self._optimizer_parameter_ids = self._validate_optimizer_scope()
        self.contract = dict(contract)
        if authorization is not None:
            raise TypeError(
                "RendererTrainer requires an identity-registered AuthorizationBinding; "
                "raw TrainingAuthorization, including one loaded from a validated receipt, "
                "is not accepted"
            )
        if authorization_binding is not None and run_binding is not None:
            raise TypeError("provide authorization_binding or run_binding, not both")
        binding = authorization_binding if authorization_binding is not None else run_binding
        if binding is not None:
            binding = require_authorization_binding(binding)
            if binding.contract != self.contract:
                raise ValueError("trainer contract differs from authorization binding")
            binding.validate_initial_renderer(model)
        # Keep the construction-time identity separately.  The public field is
        # retained for integration compatibility, but clearing or replacing it
        # can never downgrade the trainer to a weaker authorization mode.
        self._original_authorization_binding = binding
        self.authorization_binding = binding
        self.grad_norm_cap = float(grad_norm_cap)
        self.ema_decay = float(ema_decay)
        self.scheduler = scheduler
        self.operation_executor = operation_executor
        if binding is not None:
            if not isinstance(operation_executor, LedgeredOperationExecutor):
                raise RuntimeError(
                    "formal training requires the shared LedgeredOperationExecutor"
                )
            if operation_executor.authorization_binding is not binding:
                raise ValueError(
                    "trainer and operation executor must share one authorization binding"
                )
            expected_ledger = self.contract["paths"]["ledger_path"]
            if str(operation_executor.ledger.path.resolve()) != expected_ledger:
                raise ValueError("optimizer ledger path differs from the run contract")
            if "optimizer_step" not in operation_executor.ledger.budget:
                raise ValueError("run query budget does not include optimizer_step")
            self._validate_registered_optimizer()
            if scheduler is not None:
                raise ValueError("the registered first-run optimizer has no LR scheduler")
        elif operation_executor is not None:
            raise ValueError("an operation executor requires a formal authorization binding")
        self.step = 0
        self.ema_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        self._round_two_started = False

    def _validate_registered_optimizer(self) -> None:
        """Match the live optimizer and trainer guards to the hashed contract."""
        registered = self.contract["optimizer"]
        if type(self.optimizer) is not torch.optim.AdamW:
            raise ValueError("the live optimizer must be exactly torch.optim.AdamW")
        for group in self.optimizer.param_groups:
            if float(group["lr"]) != float(registered["learning_rate"]):
                raise ValueError("optimizer learning rate differs from the run contract")
            if tuple(float(value) for value in group["betas"]) != tuple(
                float(value) for value in registered["betas"]
            ):
                raise ValueError("optimizer betas differ from the run contract")
            if float(group["weight_decay"]) != float(registered["weight_decay"]):
                raise ValueError("optimizer weight decay differs from the run contract")
        if self.grad_norm_cap != float(registered["gradient_norm_cap"]):
            raise ValueError("gradient norm cap differs from the run contract")
        if self.ema_decay != float(registered["ema_decay"]):
            raise ValueError("EMA decay differs from the run contract")

    def optimizer_context(
        self,
        *,
        split: str,
        batch_id: str,
        action: Mapping[str, Any],
        prefix: int = 0,
    ) -> OperationContext:
        """Build provenance for the next optimizer mutation from current state."""
        self._preflight()
        return OperationContext(
            split=split,
            prompt=batch_id,
            seed=int(self.contract["seed"]),
            step=self.step,
            prefix=prefix,
            branch=str(self.contract["method"]),
            action=dict(action),
            checkpoint_hash=module_state_sha256(self.model),
        )

    def start_round_two_from_ema(
        self, optimizer: torch.optim.Optimizer
    ) -> dict[str, Any]:
        """Swap round-one EMA into the policy and install a fresh AdamW state."""
        binding = self._preflight()
        if self.step != 32:
            raise RuntimeError("round two must start immediately after optimizer step 32")
        if self._round_two_started:
            raise RuntimeError("round two optimizer reset has already occurred")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("round-two optimizer must be a torch optimizer")
        if optimizer.state:
            raise ValueError("round-two optimizer must have an empty state")

        previous_optimizer = self.optimizer
        previous_optimizer_ids = self._optimizer_parameter_ids
        previous_model = {
            name: value.detach().clone() for name, value in self.model.state_dict().items()
        }
        previous_hash = module_state_sha256(self.model)
        previous_state_entries = len(previous_optimizer.state)
        try:
            self.model.load_state_dict(self.ema_state, strict=True)
            self.model.zero_grad(set_to_none=True)
            self.optimizer = optimizer
            self._optimizer_parameter_ids = self._validate_optimizer_scope()
            self._validate_registered_optimizer()
            binding.validate_current(component=self.model)
        except BaseException:
            self.model.load_state_dict(previous_model, strict=True)
            self.optimizer = previous_optimizer
            self._optimizer_parameter_ids = previous_optimizer_ids
            raise
        self._round_two_started = True
        return {
            "schema": "repldm.round_transition.v1",
            "step": self.step,
            "source": "round_1_ema",
            "raw_policy_state_sha256": previous_hash,
            "ema_policy_state_sha256": module_state_sha256(self.model),
            "discarded_optimizer_state_entries": previous_state_entries,
            "new_optimizer_state_entries": len(self.optimizer.state),
        }

    def _validate_optimizer_scope(self) -> tuple[int, ...]:
        """Require the optimizer to contain every and only trainable renderer parameter."""
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        expected = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        supplied = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group.get("params", ())
        ]
        supplied_ids = [id(parameter) for parameter in supplied]
        if len(supplied_ids) != len(set(supplied_ids)):
            raise ValueError("optimizer contains duplicate renderer parameters")
        if not expected or set(supplied_ids) != {id(parameter) for parameter in expected}:
            raise ValueError(
                "optimizer parameters must be exactly the trainable renderer parameters"
            )
        if any(not isinstance(parameter, torch.nn.Parameter) for parameter in supplied):
            raise TypeError("optimizer entries must be torch parameters")
        return tuple(supplied_ids)

    def _preflight(self) -> Any:
        """Validate the capability before model, reward, or file side effects."""
        binding = self._original_authorization_binding
        if binding is None:
            raise RuntimeError(
                "training requires an identity-registered AuthorizationBinding, "
                "not a raw TrainingAuthorization receipt"
            )
        if self.authorization_binding is not binding:
            raise RuntimeError(
                "trainer authorization binding was cleared or replaced after construction"
            )
        try:
            current_optimizer_ids = self._validate_optimizer_scope()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "trainer optimizer parameter scope changed after construction"
            ) from exc
        if current_optimizer_ids != self._optimizer_parameter_ids:
            raise RuntimeError("trainer optimizer parameter scope changed after construction")
        binding = require_authorization_binding(binding)
        if binding.contract != self.contract:
            raise RuntimeError("trainer run contract changed after construction")
        binding.validate_current(component=self.model)
        if self.operation_executor is None:
            raise RuntimeError("trainer operation executor was cleared after construction")
        if self.operation_executor.authorization_binding is not binding:
            raise RuntimeError("trainer operation executor binding was replaced")
        self.operation_executor._preflight()
        self._validate_registered_optimizer()
        return binding

    def update(
        self,
        loss_fn: Callable[[], torch.Tensor],
        *,
        operation_context: OperationContext | None = None,
    ) -> UpdateRecord:
        self._preflight()
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        if not isinstance(operation_context, OperationContext):
            raise TypeError("operation_context must be an OperationContext")
        if operation_context.step != self.step:
            raise ValueError("optimizer operation context has a stale step")
        if operation_context.branch != self.contract["method"]:
            raise ValueError("optimizer operation context has the wrong method branch")
        if operation_context.seed != self.contract["seed"]:
            raise ValueError("optimizer operation context has the wrong optimization seed")
        if operation_context.checkpoint_hash != module_state_sha256(self.model):
            raise ValueError("optimizer operation context has a stale checkpoint hash")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("objective must return one finite scalar")
        # A callback cannot revoke or swap the capability and still reach an
        # optimizer mutation.
        self._preflight()
        before_model = {
            name: value.detach().clone() for name, value in self.model.state_dict().items()
        }
        before_optimizer = copy.deepcopy(self.optimizer.state_dict())
        before_scheduler = (
            copy.deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else None
        )
        before_ema = {name: value.detach().clone() for name, value in self.ema_state.items()}
        before_step = self.step

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

        def optimizer_call() -> Mapping[str, Any]:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_norm_cap
            )
            if not torch.isfinite(torch.as_tensor(norm)):
                raise ValueError("gradient norm is non-finite")
            # Autograd hooks are user code too; gate immediately before state
            # mutation while the ledger reservation is already durable.
            self._preflight()
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
            # Optimizer implementations and parameter hooks are executable
            # code. Revalidate every selected-view and run dependency after
            # they finish, while the outer transaction can still restore all
            # mutable training state.
            self._preflight()
            self.step += 1
            return {
                "step": self.step,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "model_state_sha256": module_state_sha256(self.model),
            }

        try:
            result = self.operation_executor.execute(
                "optimizer_step",
                operation_context,
                optimizer_call,
                scalar_or_gradient="optimizer_state",
            )
        except BaseException:
            try:
                self.model.load_state_dict(before_model, strict=True)
                self.optimizer.load_state_dict(before_optimizer)
                if self.scheduler is not None and before_scheduler is not None:
                    self.scheduler.load_state_dict(before_scheduler)
                self.ema_state = before_ema
                self.step = before_step
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as rollback_error:
                raise RuntimeError("failed to roll back invalid optimizer update") from rollback_error
            raise
        return UpdateRecord(
            int(result["step"]),
            float(result["loss"]),
            float(result["gradient_norm"]),
        )

    def save(self, path: str, *, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        binding = self._preflight()
        extra_payload = dict(extra or {})
        if "ema_state" in extra_payload or "authorization_binding" in extra_payload:
            raise ValueError("reserved checkpoint provenance fields cannot be overridden")
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            contract=self.contract,
            extra={"ema_state": self.ema_state, **extra_payload},
            authorization_binding=binding,
            require_authorization=True,
        )

    def load(self, path: str, *, expected_contract: Mapping[str, Any] | None = None, map_location: str | torch.device = "cpu", restore_rng: bool = True) -> dict[str, Any]:
        """Resume model, optimizer, scheduler, EMA, step, and RNG together."""
        binding = self._preflight()
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
            authorization_binding=binding,
            require_authorization=True,
        )


def run_updates(
    trainer: RendererTrainer,
    losses: Iterable[tuple[Callable[[], torch.Tensor], OperationContext]],
    *,
    checkpoint_path: Optional[str] = None,
) -> list[UpdateRecord]:
    if not isinstance(trainer, RendererTrainer):
        raise TypeError("trainer must be a RendererTrainer")
    # Obtain no iterator before the gate.  Generators commonly perform data
    # reads, reward queries, or RNG draws from __iter__ itself.
    trainer._preflight()
    if not hasattr(losses, "__iter__"):
        raise TypeError("losses must be iterable")
    iterator = iter(losses)
    records = []
    while True:
        # ``next`` may execute generator code before yielding a loss callback.
        # Revalidate before every resume, including the final StopIteration.
        trainer._preflight()
        try:
            item = next(iterator)
        except StopIteration:
            break
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("each update must provide (loss_fn, operation_context)")
        loss_fn, operation_context = item
        records.append(
            trainer.update(loss_fn, operation_context=operation_context)
        )
        if checkpoint_path is not None:
            trainer.save(checkpoint_path)
    return records
