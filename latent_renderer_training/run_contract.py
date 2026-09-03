"""Immutable, hash-addressed contracts for one renderer training run.

The optimizer, rollout collector, query ledger, and checkpoints must all see
the same contract.  Keeping this value in a small dependency-free module makes
that invariant testable without importing diffusers or touching a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


RUN_CONTRACT_SCHEMA = "repldm.training_run_contract.v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_RE = re.compile(r"^catalog-[0-9a-f]{20}$")
_SELECTED_RELEASE_RE = re.compile(r"^selected-view-[0-9a-f]{20}$")

# These fields are deliberately explicit.  A method name or a reward budget
# hidden in an un-hashed side file would make two apparently paired runs
# incomparable.
_REQUIRED_FIELDS = (
    "schema",
    "run_id",
    "method",
    "code_commit",
    "catalog_release_id",
    "catalog_manifest_sha256",
    "selected_view_release_id",
    "selected_view_manifest_sha256",
    "selected_view_config_sha256",
    "selected_view_id",
    "selected_payload_manifest_sha256",
    "selected_rows",
    "selected_splits",
    "data_manifest_sha256",
    "prompt_manifest_sha256",
    "renderer_frame_contract_hash",
    "renderer_frame_contract_artifact_sha256",
    "calibration_hash",
    "calibration_artifact_sha256",
    "action_contract_hash",
    "basis_provider_contract_hash",
    "basis_provider_config_sha256",
    "reward_config_sha256",
    "reward_preprocess_sha256",
    "reward_asset_manifest_sha256",
    "model_asset_manifest_sha256",
    "initial_renderer_state_sha256",
    "initial_renderer_state_manifest_sha256",
    "artifacts",
    "query_budget",
    "nfe",
    "scheduler_config_sha256",
    "scheduler_schedule_sha256",
    "prediction_type",
    "do_classifier_free_guidance",
    "guidance_scale",
    "guidance_rescale",
    "decision_indices",
    "optimizer",
    "method_hyperparameters",
    "runtime",
    "paths",
    "seed",
)

_REGISTERED_NFE = 50
_REGISTERED_DECISION_INDICES = (8, 24, 40)
_GATED_METHODS = frozenset({"opd", "search_distill", "dpo", "rl"})
_GATE_HASH_FIELDS = (
    "f0_gate_sha256",
    "opsd_teacher_state_manifest_sha256",
    "opsd_teacher_renderer_sha256",
    "reward_statistics_sha256",
    "cohort_manifest_sha256",
)
_F0_WITNESS_HASH_FIELDS = (
    "witness_config_sha256",
    "witness_preprocess_sha256",
    "witness_asset_manifest_sha256",
)
_PATH_FIELDS = (
    "run_dir",
    "output_dir",
    "checkpoint_dir",
    "ledger_path",
    "rollout_dir",
)
_RUNTIME_FIELDS = (
    "device",
    "batch_size",
    "model_dtype",
    "vae_dtype",
    "reward_dtype",
    "vae_scaling_factor",
    "height",
    "width",
)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("training run contract must contain JSON values") from exc


def _hash(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hash")


def _positive_int(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _nonnegative_int(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _finite_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _validate_optimizer(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("optimizer must be a mapping")
    required = {
        "name",
        "learning_rate",
        "betas",
        "weight_decay",
        "gradient_norm_cap",
        "ema_decay",
    }
    if set(value) != required:
        raise ValueError(f"optimizer must contain exactly {sorted(required)}")
    if value.get("name") != "AdamW":
        raise ValueError("the registered optimizer is AdamW")
    learning_rate = _finite_number(value.get("learning_rate"), "optimizer.learning_rate")
    weight_decay = _finite_number(value.get("weight_decay"), "optimizer.weight_decay")
    gradient_norm_cap = _finite_number(
        value.get("gradient_norm_cap"), "optimizer.gradient_norm_cap"
    )
    ema_decay = _finite_number(value.get("ema_decay"), "optimizer.ema_decay")
    betas = value.get("betas")
    if (
        not isinstance(betas, (list, tuple))
        or len(betas) != 2
        or any(isinstance(item, bool) for item in betas)
    ):
        raise ValueError("optimizer.betas must contain two finite numbers")
    beta_values = tuple(
        _finite_number(item, f"optimizer.betas[{index}]")
        for index, item in enumerate(betas)
    )
    expected = (1e-4, 0.01, 1.0, 0.995, (0.9, 0.999))
    actual = (
        learning_rate,
        weight_decay,
        gradient_norm_cap,
        ema_decay,
        beta_values,
    )
    if actual != expected:
        raise ValueError("optimizer settings differ from the registered first-run protocol")


def _validate_paths(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_PATH_FIELDS):
        raise ValueError(f"paths must contain exactly {list(_PATH_FIELDS)}")
    resolved: dict[str, Path] = {}
    for field in _PATH_FIELDS:
        raw = value.get(field)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"paths.{field} must be a non-empty absolute path")
        path = Path(raw)
        if not path.is_absolute() or path != Path(str(path.resolve(strict=False))):
            raise ValueError(f"paths.{field} must be a normalized absolute path")
        resolved[field] = path
    run_dir = resolved["run_dir"]
    for field in _PATH_FIELDS[1:]:
        try:
            resolved[field].relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(f"paths.{field} must be inside paths.run_dir") from exc
    if resolved["ledger_path"] == run_dir or resolved["ledger_path"].suffix != ".jsonl":
        raise ValueError("paths.ledger_path must name a JSONL file inside paths.run_dir")


def _validate_runtime(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_RUNTIME_FIELDS):
        raise ValueError(f"runtime must contain exactly {list(_RUNTIME_FIELDS)}")
    device = value.get("device")
    if (
        not isinstance(device, str)
        or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) is None
    ):
        raise ValueError("runtime.device must be an explicit CUDA device such as cuda:4")
    if value.get("batch_size") != 1 or type(value.get("batch_size")) is not int:
        raise ValueError("runtime.batch_size must be one prompt-seed block")
    expected_dtypes = {
        "model_dtype": "float16",
        "vae_dtype": "float32",
        "reward_dtype": "float32",
    }
    for field, expected in expected_dtypes.items():
        if value.get(field) != expected:
            raise ValueError(f"runtime.{field} must be {expected}")
    if _finite_number(
        value.get("vae_scaling_factor"), "runtime.vae_scaling_factor"
    ) != 0.13025:
        raise ValueError("runtime.vae_scaling_factor must be the frozen SDXL value 0.13025")
    for field in ("height", "width"):
        dimension = value.get(field)
        if type(dimension) is not int or dimension != 1024:
            raise ValueError(f"runtime.{field} must be the registered base resolution 1024")


def validate_run_contract_payload(
    value: Mapping[str, Any], *, require_complete: bool = True
) -> dict[str, Any]:
    """Return a canonical copy after validating the run-contract schema.

    ``require_complete=False`` is useful for read-only diagnostics.  Formal
    authorization always uses the complete form.
    """
    if not isinstance(value, Mapping):
        raise ValueError("training run contract must be a mapping")
    try:
        payload = json.loads(_canonical(dict(value)).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("training run contract is not canonical JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("training run contract must be a JSON object")
    if payload.get("schema") != RUN_CONTRACT_SCHEMA:
        raise ValueError("unsupported training run contract schema")
    if require_complete:
        missing = [field for field in _REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"training run contract is missing fields: {missing}")

    run_id = payload.get("run_id")
    if require_complete and (not isinstance(run_id, str) or not run_id):
        raise ValueError("run_id must be a non-empty string")
    method = payload.get("method")
    if require_complete and method not in {
        "f0",
        "opd",
        "search_distill",
        "dpo",
        "rl",
    }:
        raise ValueError("method is not a registered renderer-training method")
    if method in _GATED_METHODS:
        cohort_id = payload.get("cohort_id")
        if not isinstance(cohort_id, str) or not cohort_id:
            raise ValueError("gated training methods require a non-empty cohort_id")
        for field in _GATE_HASH_FIELDS:
            _hash(payload.get(field), field)
    elif require_complete and any(
        field in payload for field in ("cohort_id", *_GATE_HASH_FIELDS)
    ):
        raise ValueError("F0 contracts cannot claim outputs from their own gate")
    if method == "f0":
        for field in _F0_WITNESS_HASH_FIELDS:
            _hash(payload.get(field), field)
    elif require_complete and any(field in payload for field in _F0_WITNESS_HASH_FIELDS):
        raise ValueError("training contracts cannot consume the independent F0 witness")

    commit = payload.get("code_commit")
    if require_complete and (
        not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None
    ):
        raise ValueError("code_commit must be a full Git commit")
    release = payload.get("catalog_release_id")
    if require_complete and (
        not isinstance(release, str) or _RELEASE_RE.fullmatch(release) is None
    ):
        raise ValueError("catalog_release_id is not a content-addressed release")
    selected_release = payload.get("selected_view_release_id")
    if require_complete and (
        not isinstance(selected_release, str)
        or _SELECTED_RELEASE_RE.fullmatch(selected_release) is None
    ):
        raise ValueError("selected_view_release_id is not a content-addressed child release")
    selected_view_id = payload.get("selected_view_id")
    if require_complete and (
        not isinstance(selected_view_id, str) or not selected_view_id
    ):
        raise ValueError("selected_view_id must be a non-empty string")

    hash_fields = (
        "catalog_manifest_sha256",
        "selected_view_manifest_sha256",
        "selected_view_config_sha256",
        "selected_payload_manifest_sha256",
        "data_manifest_sha256",
        "prompt_manifest_sha256",
        "renderer_frame_contract_hash",
        "renderer_frame_contract_artifact_sha256",
        "calibration_hash",
        "calibration_artifact_sha256",
        "action_contract_hash",
        "basis_provider_contract_hash",
        "basis_provider_config_sha256",
        "reward_config_sha256",
        "reward_preprocess_sha256",
        "reward_asset_manifest_sha256",
        "model_asset_manifest_sha256",
        "initial_renderer_state_sha256",
        "initial_renderer_state_manifest_sha256",
    )
    for field in hash_fields:
        if require_complete or field in payload:
            _hash(payload.get(field), field)

    if require_complete:
        if payload.get("selected_rows") != 96 or type(payload.get("selected_rows")) is not int:
            raise ValueError("selected_rows must be the frozen 96-row selected view")
        if payload.get("selected_splits") != {"train": 64, "validation": 32}:
            raise ValueError("selected_splits must be the frozen 64 train + 32 validation split")
        _positive_int(payload.get("nfe"), "nfe")
        if payload["nfe"] != _REGISTERED_NFE:
            raise ValueError("nfe differs from the registered 50-step schedule")
        seed = payload.get("seed")
        _nonnegative_int(seed, "seed")
    elif "nfe" in payload:
        _positive_int(payload["nfe"], "nfe")

    for field in ("scheduler_config_sha256", "scheduler_schedule_sha256"):
        if require_complete or field in payload:
            _hash(payload.get(field), field)
    if require_complete:
        if payload.get("prediction_type") != "epsilon":
            raise ValueError("the registered SDXL scheduler prediction_type is epsilon")
        if payload.get("do_classifier_free_guidance") is not True:
            raise ValueError("the registered trajectory requires classifier-free guidance")
        guidance_scale = _finite_number(payload.get("guidance_scale"), "guidance_scale")
        guidance_rescale = _finite_number(
            payload.get("guidance_rescale"), "guidance_rescale"
        )
        if guidance_scale != 7.5 or guidance_rescale != 0.0:
            raise ValueError("CFG settings differ from the registered SDXL trajectory")
        indices = payload.get("decision_indices")
        if not isinstance(indices, (list, tuple)) or tuple(indices) != _REGISTERED_DECISION_INDICES:
            raise ValueError("decision_indices differ from the registered schedule")
        if any(type(index) is not int for index in indices):
            raise ValueError("decision_indices must contain plain integers")
        _validate_optimizer(payload.get("optimizer"))
        method_hyperparameters = payload.get("method_hyperparameters")
        if not isinstance(method_hyperparameters, Mapping) or not method_hyperparameters:
            raise ValueError("method_hyperparameters must be a non-empty mapping")
        if method_hyperparameters.get("method") != payload.get("method"):
            raise ValueError("method_hyperparameters.method differs from method")
        _validate_runtime(payload.get("runtime"))
        _validate_paths(payload.get("paths"))

    budget = payload.get("query_budget")
    if require_complete or budget is not None:
        if not isinstance(budget, Mapping) or not budget:
            raise ValueError("query_budget must be a non-empty mapping")
        for key, amount in budget.items():
            if not isinstance(key, str) or not key:
                raise ValueError("query_budget keys must be non-empty strings")
            _nonnegative_int(amount, f"query_budget[{key}]")

    reward_budget = payload.get("reward_budget")
    if reward_budget is not None:
        if not isinstance(reward_budget, Mapping) or not reward_budget:
            raise ValueError("reward_budget must be a non-empty mapping")
        for key, amount in reward_budget.items():
            if not isinstance(key, str) or not key:
                raise ValueError("reward_budget keys must be non-empty strings")
            _nonnegative_int(amount, f"reward_budget[{key}]")

    artifacts = payload.get("artifacts")
    if require_complete or artifacts is not None:
        from .artifacts import required_run_artifacts

        if not isinstance(artifacts, Mapping):
            raise ValueError("artifacts must be a mapping")
        if require_complete:
            expected_artifacts = set(required_run_artifacts(method))
            missing_artifacts = sorted(expected_artifacts.difference(artifacts))
            if missing_artifacts:
                raise ValueError(
                    "training run contract is missing artifact descriptors: "
                    f"{missing_artifacts}"
                )
        for name, descriptor in artifacts.items():
            if not isinstance(name, str) or not name:
                raise ValueError("artifact names must be non-empty strings")
            if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
                raise ValueError(
                    f"artifacts[{name}] must contain only path and sha256"
                )
            path = descriptor.get("path")
            if not isinstance(path, str) or not path.startswith("/"):
                raise ValueError(f"artifacts[{name}].path must be absolute")
            _hash(descriptor.get("sha256"), f"artifacts[{name}].sha256")

    # A contract must not hide a non-finite scalar in a nested method setting.
    # json.dumps(..., allow_nan=False) catches ordinary floats; this explicit
    # walk gives a clearer error for custom numeric values.
    def finite(value: Any) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, Mapping):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return True

    if not finite(payload):
        raise ValueError("training run contract contains a non-finite value")
    return payload


@dataclass(frozen=True)
class TrainingRunContract:
    """Immutable JSON contract and its SHA-256 identity."""

    _canonical_json: str
    sha256: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | "TrainingRunContract", *, require_complete: bool = True
    ) -> "TrainingRunContract":
        if isinstance(value, TrainingRunContract):
            if require_complete:
                validate_run_contract_payload(value.to_dict(), require_complete=True)
            return value
        payload = validate_run_contract_payload(value, require_complete=require_complete)
        encoded = _canonical(payload)
        return cls(
            encoded.decode("utf-8"), hashlib.sha256(encoded).hexdigest()
        )

    @property
    def payload(self) -> dict[str, Any]:
        """Return a defensive decoded copy of the contract."""
        return json.loads(self._canonical_json)

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    @property
    def contract_hash(self) -> str:
        """Descriptive alias used by ledgers and checkpoint metadata."""
        return self.sha256


__all__ = [
    "RUN_CONTRACT_SCHEMA",
    "TrainingRunContract",
    "validate_run_contract_payload",
]
