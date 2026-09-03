"""Closed method registry for formal latent-renderer training runs."""

from __future__ import annotations

import importlib
import json
from typing import Any, Mapping

from .authorization import require_authorization_binding


METHOD_RESULT_SCHEMA = "repldm.renderer_training_result.v1"
F0_FAILURE_RESULT_SCHEMA = "repldm.renderer_f0_failure.v1"

# Module and symbol names are source-controlled instead of being accepted from
# YAML or the command line.  This prevents a reviewed authorization receipt
# from becoming a capability to execute an unrelated callback.
_METHOD_HANDLERS = {
    "f0": ("latent_renderer_training.methods.f0", "run_f0"),
    "opd": ("latent_renderer_training.methods.opd", "run_opd"),
    "search_distill": (
        "latent_renderer_training.methods.search_distill",
        "run_search_distill",
    ),
    "dpo": ("latent_renderer_training.methods.dpo", "run_dpo"),
    "rl": ("latent_renderer_training.methods.rl", "run_rl"),
}


def _validate_result(value: Any, *, binding: Any, method: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("training method must return an auditable result mapping")
    try:
        result = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("training method result must contain finite JSON values") from exc
    common = {
        "run_id": binding.contract["run_id"],
        "method": method,
        "run_contract_sha256": binding.contract_hash,
    }
    for field, expected in common.items():
        if result.get(field) != expected:
            raise RuntimeError(
                f"training method result field {field} differs from the authorized run"
            )
    if method == "f0" and result.get("status") == "screen_failed":
        required = {
            "schema": F0_FAILURE_RESULT_SCHEMA,
            "benchmark_status": "not_applicable",
        }
        if result.get("stage") not in {"train_gate", "validation_gate"}:
            raise RuntimeError(
                "F0 failure result field stage differs from the authorized screen"
            )
        if not isinstance(result.get("reason"), str) or not result["reason"]:
            raise RuntimeError("F0 failure result must contain a reason")
    else:
        required = {
            "schema": METHOD_RESULT_SCHEMA,
            "status": "training_complete",
            "benchmark_status": "pending",
        }
    for field, expected in required.items():
        if result.get(field) != expected:
            raise RuntimeError(
                f"training method result field {field} differs from the authorized run"
            )
    return result


def dispatch_training(binding: Any) -> dict[str, Any]:
    """Run exactly the source-registered handler for an authorized method."""
    binding = require_authorization_binding(binding)
    binding.validate_current()
    method = binding.contract.get("method")
    specification = _METHOD_HANDLERS.get(method)
    if specification is None:
        raise RuntimeError(f"no source-registered training handler for method {method!r}")
    module_name, symbol_name = specification
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if isinstance(exc.name, str) and (
            exc.name == module_name or module_name.startswith(exc.name + ".")
        ):
            raise RuntimeError(
                f"training method {method!r} is registered but not implemented"
            ) from exc
        raise
    handler = getattr(module, symbol_name, None)
    if not callable(handler):
        raise RuntimeError(
            f"training method {method!r} has no callable {symbol_name}"
        )
    result = _validate_result(handler(binding), binding=binding, method=str(method))
    # A handler cannot invalidate its source/data/reward inputs and still
    # publish a successful result.
    binding.validate_current()
    return result


__all__ = [
    "F0_FAILURE_RESULT_SCHEMA",
    "METHOD_RESULT_SCHEMA",
    "dispatch_training",
]
