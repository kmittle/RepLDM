"""Atomic checkpoint and resume helpers with explicit provenance."""

from __future__ import annotations

import hashlib
import json
import copy
import os
import random
from pathlib import Path
from typing import Any, Mapping

import torch


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def capture_rng_state() -> dict[str, Any]:
    """Capture all available pseudo-random streams used by a training run."""
    state: dict[str, Any] = {
        "schema": "repldm.rng_state.v1",
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "python": random.getstate(),
        "numpy": None,
        "numpy_available": False,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
        state["numpy_available"] = True
    except ImportError:
        pass
    return state


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    """Validate every stream before any stream is mutated."""
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint RNG state is incomplete")
    required = {
        "schema", "torch_cpu", "torch_cuda", "python", "numpy", "numpy_available"
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"checkpoint RNG state is incomplete: missing {missing}")
    if state["schema"] != "repldm.rng_state.v1":
        raise ValueError("unsupported checkpoint RNG state schema")
    cpu_state = state["torch_cpu"]
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
        or cpu_state.numel() == 0
    ):
        raise ValueError("checkpoint CPU RNG state is invalid")
    # Validate Python's state on an isolated generator so a malformed state
    # cannot partially alter the process RNG.
    try:
        random.Random().setstate(state["python"])
    except Exception as exc:
        raise ValueError("checkpoint Python RNG state is invalid") from exc
    numpy_available = state["numpy_available"]
    if not isinstance(numpy_available, bool):
        raise ValueError("checkpoint NumPy availability flag is invalid")
    numpy_state = state["numpy"]
    if numpy_available:
        if numpy_state is None:
            raise ValueError("checkpoint NumPy RNG state is missing")
        try:
            import numpy as np

            np.random.RandomState().set_state(numpy_state)
        except ImportError as exc:
            raise RuntimeError("checkpoint requires NumPy to restore its RNG state") from exc
        except Exception as exc:
            raise ValueError("checkpoint NumPy RNG state is invalid") from exc
    elif numpy_state is not None:
        raise ValueError("checkpoint NumPy RNG availability does not match its state")
    cuda_state = state["torch_cuda"]
    if cuda_state is None:
        if torch.cuda.is_available():
            raise RuntimeError("checkpoint lacks CUDA RNG state")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if not isinstance(cuda_state, (list, tuple)) or len(cuda_state) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state is invalid")
        for item in cuda_state:
            if (
                not isinstance(item, torch.Tensor)
                or item.dtype != torch.uint8
                or item.ndim != 1
                or item.numel() == 0
            ):
                raise ValueError("checkpoint CUDA RNG state is invalid")


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`.

    CUDA state is restored only when CUDA is available; a CPU resume remains
    useful for audits and does not silently consume a different CPU stream.
    """
    _validate_rng_state(state)
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8, device="cpu"))
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    if state["numpy"] is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError as exc:
            raise RuntimeError("checkpoint requires NumPy to restore its RNG state") from exc


def _clone_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _validate_model_state(
    supplied: Any, expected: Mapping[str, Any], *, label: str
) -> None:
    if not isinstance(supplied, Mapping):
        raise ValueError(f"checkpoint {label} state is invalid")
    if set(supplied) != set(expected):
        raise ValueError(f"checkpoint {label} state keys do not match the model")
    for name, expected_value in expected.items():
        actual = supplied[name]
        if isinstance(expected_value, torch.Tensor):
            if not isinstance(actual, torch.Tensor):
                raise ValueError(f"checkpoint {label} tensor is invalid for {name}")
            if actual.shape != expected_value.shape or actual.dtype != expected_value.dtype:
                raise ValueError(f"checkpoint {label} tensor shape or dtype is invalid for {name}")
        elif actual != expected_value:
            raise ValueError(f"checkpoint {label} value is invalid for {name}")


def _optimizer_manifest(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    """Bind optimizer groups to parameter names, shapes, and dtypes."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[list[dict[str, Any]]] = []
    for group in optimizer.param_groups:
        entries: list[dict[str, Any]] = []
        for parameter in group.get("params", ()):
            name = names.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter outside the model")
            entries.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
            )
        groups.append(entries)
    return {"schema": "repldm.optimizer_manifest.v1", "groups": groups}


def _validate_optimizer_payload(
    supplied: Any,
    recorded_manifest: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    if not isinstance(recorded_manifest, Mapping):
        raise ValueError("checkpoint optimizer manifest is missing")
    expected_manifest = _optimizer_manifest(model, optimizer)
    if dict(recorded_manifest) != expected_manifest:
        raise ValueError("checkpoint optimizer parameter manifest does not match the optimizer")
    if not isinstance(supplied, Mapping):
        raise ValueError("checkpoint optimizer state is invalid")
    groups = supplied.get("param_groups")
    state = supplied.get("state")
    if not isinstance(groups, list) or not isinstance(state, Mapping):
        raise ValueError("checkpoint optimizer state is invalid")
    recorded_groups = recorded_manifest.get("groups")
    if not isinstance(recorded_groups, list) or len(groups) != len(recorded_groups):
        raise ValueError("checkpoint optimizer parameter groups do not match")
    id_specs: dict[Any, dict[str, Any]] = {}
    seen_ids: set[Any] = set()
    for group, specs in zip(groups, recorded_groups):
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("checkpoint optimizer parameter group is invalid")
        if not isinstance(specs, list) or len(group["params"]) != len(specs):
            raise ValueError("checkpoint optimizer parameter order does not match")
        for parameter_id, spec in zip(group["params"], specs):
            if parameter_id in seen_ids:
                raise ValueError("checkpoint optimizer contains duplicate parameter ids")
            seen_ids.add(parameter_id)
            if not isinstance(spec, Mapping):
                raise ValueError("checkpoint optimizer manifest entry is invalid")
            id_specs[parameter_id] = dict(spec)
    for parameter_id, state_entry in state.items():
        if parameter_id not in id_specs or not isinstance(state_entry, Mapping):
            raise ValueError("checkpoint optimizer state has an unknown parameter")
        shape = tuple(id_specs[parameter_id].get("shape", ()))
        for value in state_entry.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                # Optimizer counters are scalar tensors; moment tensors must
                # have exactly the parameter shape.
                if tuple(value.shape) != shape and value.numel() != 1:
                    raise ValueError("checkpoint optimizer state tensor shape does not match")


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int,
    contract: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a checkpoint through a fsync-and-replace transaction."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extra_payload = dict(extra or {})
    if scheduler is not None:
        extra_payload["scheduler_state"] = scheduler.state_dict()
    if optimizer is not None:
        extra_payload["optimizer_manifest"] = _optimizer_manifest(model, optimizer)
    supplied_rng = capture_rng_state() if rng_state is None else rng_state
    if not isinstance(supplied_rng, Mapping):
        raise ValueError("rng_state must be a complete mapping")
    rng_payload = dict(supplied_rng)
    _validate_rng_state(rng_payload)
    extra_payload["rng_state"] = rng_payload
    payload: dict[str, Any] = {
        "schema": "repldm.latent_renderer_checkpoint.v1",
        "step": int(step), "contract": dict(contract),
        "contract_hash": hashlib.sha256(_canonical(contract)).hexdigest(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra_payload,
    }
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    directory_fd = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {"path": str(destination), "step": int(step), "contract_hash": payload["contract_hash"], "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    trainer: Any | None = None,
    expected_contract: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and restore the complete trainer state.

    Validation happens before model mutation.  Passing ``trainer`` restores
    its step counter and EMA state; passing ``scheduler`` restores its state.
    Older/incomplete checkpoints are rejected when either state is requested.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "repldm.latent_renderer_checkpoint.v1":
        raise ValueError("unsupported latent renderer checkpoint schema")
    if not isinstance(payload.get("contract"), Mapping):
        raise ValueError("checkpoint contract is invalid")
    actual_hash = hashlib.sha256(_canonical(payload.get("contract", {}))).hexdigest()
    if actual_hash != payload.get("contract_hash"):
        raise ValueError("checkpoint contract hash is invalid")
    if trainer is not None:
        trainer_contract = getattr(trainer, "contract", None)
        if not isinstance(trainer_contract, Mapping):
            raise ValueError("trainer contract must be a mapping")
        if expected_contract is not None:
            if not isinstance(expected_contract, Mapping):
                raise ValueError("expected contract must be a mapping")
            if _canonical(expected_contract) != _canonical(trainer_contract):
                raise ValueError("expected contract does not match trainer contract")
        # A trainer-bound load must always validate against the contract that
        # owns the run; callers cannot substitute a different expectation.
        effective_expected_contract = trainer_contract
    else:
        effective_expected_contract = expected_contract
    if effective_expected_contract is not None and not isinstance(effective_expected_contract, Mapping):
        raise ValueError("expected contract must be a mapping")
    if effective_expected_contract is not None and hashlib.sha256(_canonical(effective_expected_contract)).hexdigest() != actual_hash:
        raise ValueError("checkpoint contract does not match the requested run")
    if isinstance(payload.get("step"), bool) or not isinstance(payload.get("step"), int) or payload["step"] < 0:
        raise ValueError("checkpoint step is invalid")
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("checkpoint extra state is invalid")
    current_model_state = model.state_dict()
    _validate_model_state(payload.get("model"), current_model_state, label="model")
    if trainer is not None and getattr(trainer, "model", None) is not model:
        raise ValueError("trainer must own the model passed to load_checkpoint")
    if optimizer is not None and payload.get("optimizer") is None:
        raise ValueError("checkpoint does not contain optimizer state")
    if optimizer is not None:
        _validate_optimizer_payload(
            payload.get("optimizer"),
            extra.get("optimizer_manifest"),
            model,
            optimizer,
        )
    if scheduler is not None and "scheduler_state" not in extra:
        raise ValueError("checkpoint does not contain scheduler state")
    if scheduler is not None and not isinstance(extra["scheduler_state"], Mapping):
        raise ValueError("checkpoint scheduler state is invalid")
    if trainer is not None:
        ema = extra.get("ema_state")
        if not isinstance(ema, dict) or set(ema) != set(model.state_dict()):
            raise ValueError("checkpoint EMA state is incomplete")
        for name, value in ema.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != current_model_state[name].shape
                or value.dtype != current_model_state[name].dtype
            ):
                raise ValueError(f"checkpoint EMA tensor is invalid for {name}")
    if restore_rng and "rng_state" not in extra:
        raise ValueError("checkpoint does not contain RNG state")
    if restore_rng:
        _validate_rng_state(extra["rng_state"])

    # All structural checks are complete.  Keep snapshots so an optimizer,
    # scheduler, custom module hook, or RNG failure cannot leave a half-loaded
    # trainer behind.
    before_model = {name: _clone_state(value) for name, value in current_model_state.items()}
    before_optimizer = _clone_state(optimizer.state_dict()) if optimizer is not None else None
    before_scheduler = _clone_state(scheduler.state_dict()) if scheduler is not None else None
    before_rng = capture_rng_state()
    before_step = getattr(trainer, "step", None) if trainer is not None else None
    before_ema = _clone_state(getattr(trainer, "ema_state", None)) if trainer is not None else None
    try:
        model.load_state_dict(payload["model"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(extra["scheduler_state"])
        if trainer is not None:
            trainer.step = int(payload["step"])
            state_by_name = model.state_dict()
            trainer.ema_state = {
                name: extra["ema_state"][name].detach().clone().to(device=state_by_name[name].device)
                for name in state_by_name
            }
        if restore_rng:
            restore_rng_state(extra["rng_state"])
    except Exception:
        try:
            model.load_state_dict(before_model, strict=True)
            if optimizer is not None and before_optimizer is not None:
                optimizer.load_state_dict(before_optimizer)
            if scheduler is not None and before_scheduler is not None:
                scheduler.load_state_dict(before_scheduler)
            if trainer is not None:
                trainer.step = before_step
                trainer.ema_state = before_ema
            restore_rng_state(before_rng)
        except Exception:
            # Preserve the original exception; the pre-validated snapshots
            # still make ordinary failures fully transactional.
            pass
        raise
    return payload
