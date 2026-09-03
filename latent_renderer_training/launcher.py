"""The single formal entry point for OPD/DPO/RL renderer training.

This module intentionally has no diffusers, dataset, or reward-model imports at
module load time.  The receipt and Git gate run first; only an authorized
callback is allowed to construct those expensive objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .authorization import TrainingAuthorization
from .run_contract import TrainingRunContract
from .runtime import dispatch_training


TRAINING_CONFIG_SCHEMA = "repldm.renderer_training_config.v1"


@dataclass(frozen=True)
class LaunchResult:
    """Auditable result returned by validation-only or executed launches."""

    run_id: str
    method: str
    run_contract_sha256: str
    config_sha256: str
    authorization_receipt: str
    executed: bool
    validation_only: bool
    value: Any = None


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    """Read a JSON/YAML config after authorization has already succeeded."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("training config must be an ordinary file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("training config is unreadable") from exc
    if not raw:
        raise ValueError("training config is empty")
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(raw.decode("utf-8"))
        else:
            # PyYAML is an optional experiment dependency.  Importing it here
            # is safe because the receipt/Git gate has already run.
            try:
                import yaml
            except ImportError as exc:
                raise ValueError("PyYAML is required to read this training config") from exc

            try:
                payload = yaml.safe_load(raw.decode("utf-8"))
            except yaml.YAMLError as exc:
                raise ValueError("training config contains invalid YAML") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("training config is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("training config must be a mapping")
    try:
        result = json.loads(
            json.dumps(
                dict(payload),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("training config must contain JSON values") from exc
    if not isinstance(result, dict) or set(result) != {"schema", "run_contract"}:
        raise ValueError(
            "training config must contain only schema and run_contract"
        )
    if result.get("schema") != TRAINING_CONFIG_SCHEMA:
        raise ValueError("unsupported renderer training config schema")
    return result, hashlib.sha256(raw).hexdigest()


def launch_training(
    *,
    receipt_path: str | Path,
    config_path: str | Path,
    repository_root: str | Path,
    validation_only: bool = False,
) -> LaunchResult:
    """Validate one formal run and, by default, execute its fixed handler."""
    if not isinstance(validation_only, bool):
        raise TypeError("validation_only must be boolean")
    # No config bytes, catalog builder, data rows, reward model, or GPU object
    # are touched before this call completes.
    authorization = TrainingAuthorization.load(
        receipt_path, repository_root=repository_root
    )
    config, config_hash = _load_config(Path(config_path).absolute())
    contract = TrainingRunContract.from_mapping(config["run_contract"])
    binding = authorization.bind_run_contract(contract)
    # Recheck immediately before handing control to the method dispatcher.  A
    # dispatcher may safely assume this is the last gate before model/data I/O.
    binding.validate_current()
    value = None if validation_only else dispatch_training(binding)
    executed = not validation_only
    payload = contract.payload
    return LaunchResult(
        run_id=str(payload["run_id"]),
        method=str(payload["method"]),
        run_contract_sha256=binding.contract_hash,
        config_sha256=config_hash,
        authorization_receipt=str(authorization.receipt_path),
        executed=executed,
        validation_only=validation_only,
        value=value,
    )


__all__ = ["TRAINING_CONFIG_SCHEMA", "LaunchResult", "launch_training"]
