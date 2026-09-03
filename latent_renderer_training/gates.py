"""Machine-checkable F0 and cohort gates for formal renderer training."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

from .f0_metrics import (
    F0_QUERY_BUDGET,
    build_f0_metric_action,
    load_f0_phase_evidence,
    validate_f0_screen_registration_evidence,
)
from .ledger import QueryLedger
from .operations import tensor_operation_output_sha256
from .witnesses import TOPIQ_NR_MODEL_ID, TOPIQ_NR_PREPROCESS_SHA256, TOPIQ_NR_ROLE

F0_GATE_SCHEMA = "repldm.renderer_f0_gate.v5"
OPSD_TEACHER_STATE_SCHEMA = "repldm.renderer_opsd_teacher_state.v1"
REWARD_STATISTICS_SCHEMA = "repldm.renderer_reward_statistics.v1"
TRAINING_COHORT_SCHEMA = "repldm.renderer_training_cohort.v1"
MAIN_METHODS = ("opd", "search_distill", "dpo", "rl")
OPTIMIZATION_SEEDS = (202609011, 202609012, 202609013)
TRAIN_GENERATION_SEEDS = (2026090101, 2026090102)
VALIDATION_GENERATION_SEEDS = (2026090191, 2026090192)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CATALOG_RE = re.compile(r"^catalog-[0-9a-f]{20}$")
_SELECTED_RE = re.compile(r"^selected-view-[0-9a-f]{20}$")

# These values are duplicated deliberately as an independent gate oracle.  The
# runner is allowed to evolve only when the registered protocol and this oracle
# agree; importing ``methods.f0`` here would create a circular dependency.
_F0_TOTAL_STEPS = 50
_F0_DECISION_INDICES = (8, 24, 40)
_F0_FIT_BATCH_SIZE = 8
_F0_FIT_STEPS = 200
_F0_FOLD_OPTIMIZER_SEEDS = (2026090100, 2026090101, 2026090102, 2026090103)
_F0_FINAL_OPTIMIZER_SEED = 2026090104
_F0_OPERATION_KINDS = {
    "unet_forward",
    "scheduler_step",
    "vae_decode",
    "reward_forward",
    "reward_backward",
    "optimizer_step",
}
_F0_RESERVATION_FIELDS = {
    "code_hash", "data_hash", "checkpoint_hash", "method_allocation", "split",
    "prompt", "seed", "step", "prefix", "branch", "action", "model", "role",
    "preprocess_hash", "model_config_sha256", "model_asset_manifest_sha256",
    "run_contract_hash", "renderer_frame_contract_hash", "calibration_hash",
    "data_manifest_sha256", "reward_config_sha256",
}
_F0_RESULT_FIELDS = {
    "image_hash", "output_hash", "reward_preprocess_hash", "model", "role",
    "model_config_sha256", "model_asset_manifest_sha256", "scalar_or_gradient",
    "wall_seconds", "cached_parent", "parent_reservation_id", "failure",
    "run_contract_hash", "renderer_frame_contract_hash", "calibration_hash",
    "data_manifest_sha256", "reward_config_sha256", "result_hash",
}

_SHARED_FIELDS = (
    "code_commit",
    "catalog_release_id",
    "catalog_manifest_sha256",
    "selected_view_release_id",
    "selected_view_manifest_sha256",
    "selected_view_config_sha256",
    "selected_view_id",
    "selected_payload_manifest_sha256",
    "data_manifest_sha256",
    "prompt_manifest_sha256",
    "renderer_frame_contract_hash",
    "calibration_hash",
    "action_contract_hash",
    "basis_provider_contract_hash",
    "reward_config_sha256",
    "reward_preprocess_sha256",
    "model_asset_manifest_sha256",
    "initial_renderer_state_sha256",
    "f0_gate_sha256",
    "opsd_teacher_state_manifest_sha256",
    "opsd_teacher_renderer_sha256",
    "reward_statistics_sha256",
    "nfe",
    "scheduler_config_sha256",
    "scheduler_schedule_sha256",
    "prediction_type",
    "do_classifier_free_guidance",
    "guidance_scale",
    "guidance_rescale",
    "decision_indices",
)
_RUNTIME_INVARIANTS = (
    "batch_size",
    "model_dtype",
    "vae_dtype",
    "reward_dtype",
    "vae_scaling_factor",
    "height",
    "width",
)


def _copy(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
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
        raise ValueError(f"{label} must contain finite JSON values") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a JSON object")
    return result


def _hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def validate_file_binding(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
        raise ValueError(f"{label} file binding is incomplete")
    path = value.get("path")
    size = value.get("bytes")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label} byte count must be positive")
    return {"path": path, "sha256": _hash(value.get("sha256"), label=label), "bytes": size}


def validate_reward_statistics(
    value: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    result = _copy(value, label="reward statistics")
    required = {
        "schema", "status", "anchor_count", "estimator", "location", "scale",
        "initial_renderer_state_sha256", "reward_config_sha256",
        "reward_preprocess_sha256", "selected_view_release_id", "anchors",
    }
    if set(result) != required or result.get("schema") != REWARD_STATISTICS_SCHEMA:
        raise ValueError("reward statistics fields differ from the registered schema")
    if result.get("status") != "frozen" or result.get("anchor_count") != 128:
        raise ValueError("reward statistics must freeze exactly 128 C0 anchors")
    if result.get("estimator") != "median_iqr_over_1.349_floor_1e-6":
        raise ValueError("reward statistics estimator differs from the protocol")
    _finite(result.get("location"), label="reward statistics location")
    if _finite(result.get("scale"), label="reward statistics scale") < 1e-6:
        raise ValueError("reward statistics scale is below the frozen floor")
    for field in (
        "initial_renderer_state_sha256", "reward_config_sha256",
        "reward_preprocess_sha256", "selected_view_release_id",
    ):
        if result.get(field) != contract.get(field):
            raise ValueError(f"reward statistics {field} differs from the run contract")
    anchors = result.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 128:
        raise ValueError("reward statistics must contain 128 anchor rows")
    seen: set[tuple[str, int]] = set()
    prompts: set[str] = set()
    for index, row in enumerate(anchors):
        if not isinstance(row, Mapping) or set(row) != {
            "prompt_id", "generation_seed", "image_sha256", "reward"
        }:
            raise ValueError(f"reward anchor row {index} is invalid")
        prompt_id = row.get("prompt_id")
        seed = row.get("generation_seed")
        if not isinstance(prompt_id, str) or not prompt_id or seed not in TRAIN_GENERATION_SEEDS:
            raise ValueError(f"reward anchor row {index} has an invalid prompt or seed")
        key = (prompt_id, int(seed))
        if key in seen:
            raise ValueError("reward anchor prompt-seed pairs must be unique")
        seen.add(key)
        prompts.add(prompt_id)
        _hash(row.get("image_sha256"), label=f"reward anchor row {index} image")
        _finite(row.get("reward"), label=f"reward anchor row {index} value")
    if len(prompts) != 64 or any((prompt, seed) not in seen for prompt in prompts for seed in TRAIN_GENERATION_SEEDS):
        raise ValueError("reward anchors must cover 64 prompts under both registered seeds")
    return result


def validate_opsd_teacher_state(
    value: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _copy(value, label="OPSD teacher state")
    required = {
        "schema", "status", "role", "f0_run_contract_sha256",
        "renderer_state_sha256", "checkpoint",
    }
    if set(result) != required or result.get("schema") != OPSD_TEACHER_STATE_SCHEMA:
        raise ValueError("OPSD teacher state fields differ from the registered schema")
    if result.get("status") != "frozen" or result.get("role") != "T_OPSD":
        raise ValueError("OPSD teacher state is not a frozen T_OPSD checkpoint")
    _hash(result.get("f0_run_contract_sha256"), label="F0 run contract")
    if result.get("renderer_state_sha256") != contract.get("opsd_teacher_renderer_sha256"):
        raise ValueError("OPSD teacher renderer hash differs from the run contract")
    return result, validate_file_binding(result.get("checkpoint"), label="OPSD checkpoint")


def _reward_scale_from_contract(contract: Mapping[str, Any]) -> float:
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("run contract does not bind reward statistics")
    descriptor = artifacts.get("reward_statistics")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("run contract reward statistics binding is incomplete")
    path_value = descriptor.get("path")
    if not isinstance(path_value, str):
        raise ValueError("run contract reward statistics path is invalid")
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("run contract reward statistics must be an absolute ordinary file")
    expected_hash = _hash(
        descriptor.get("sha256"), label="run contract reward statistics"
    )
    if expected_hash != contract.get("reward_statistics_sha256"):
        raise ValueError("reward statistics artifact hash differs from the run contract")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError("reward statistics artifact is unreadable") from exc
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("reward statistics artifact changed after authorization")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("reward statistics artifact is invalid JSON") from exc
    statistics = validate_reward_statistics(value, contract=contract)
    return float(statistics["scale"])


def _read_bound_file(value: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    binding = validate_file_binding(value, label=label)
    path = Path(binding["path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute ordinary file")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if (
        len(payload) != binding["bytes"]
        or hashlib.sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise ValueError(f"{label} bytes differ from the binding")
    return path, binding


def _float32_tensor_output_sha256(value: Any, *, label: str) -> str:
    scalar = _finite(value, label=label)
    try:
        tensor_hash = hashlib.sha256(struct.pack("<f", scalar)).hexdigest()
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{label} is outside the float32 score range") from exc
    return tensor_operation_output_sha256(tensor_hash)


def _f0_metric_action(
    metadata: Mapping[str, Any],
    *,
    branch: str,
    target_record_sha256: str,
) -> None:
    action = metadata.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("F0 metric receipt action does not bind its evidence row")
    expected_reuse = "shared_prompt_seed" if branch == "anchor" else "forbidden"
    expected_target = None if branch == "anchor" else target_record_sha256
    try:
        expected = build_f0_metric_action(
            branch=branch,
            target_record_sha256=expected_target,
            renderer_action_history=action.get("renderer_action_history"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "F0 metric receipt action differs from its evidence row"
        ) from exc
    if dict(action) != expected or action.get("reuse_scope") != expected_reuse:
        raise ValueError("F0 metric receipt action differs from its evidence row")


def _receipt_result_has_identity(
    result: Mapping[str, Any], identity: Mapping[str, Any]
) -> bool:
    return all(
        result.get("reward_preprocess_hash" if key == "preprocess_hash" else key)
        == value
        for key, value in identity.items()
    )


def _f0_json_key(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("F0 operation action is not canonical JSON") from exc


def _f0_operation_key(kind: str, metadata: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        kind,
        metadata.get("split"),
        metadata.get("prompt"),
        metadata.get("seed"),
        metadata.get("step"),
        metadata.get("prefix"),
        metadata.get("branch"),
        metadata.get("checkpoint_hash"),
        _f0_json_key(metadata.get("action")),
    )


def _f0_zero_history(count: int) -> list[list[list[float]]]:
    if type(count) is not int or count < 0 or count > len(_F0_DECISION_INDICES):
        raise ValueError("invalid F0 action-history length")
    return [[[0.0] * 6] for _ in range(count)]


def _f0_stable_id(
    *, prompt_id: str, generation_seed: int, decision_index: int, split: str
) -> str:
    if split not in {"train", "validation"}:
        raise ValueError("invalid F0 stable-id split")
    digest = hashlib.sha256(
        f"{prompt_id}\0{generation_seed}\0{decision_index}".encode("utf-8")
    ).hexdigest()[:20]
    return f"{split}-{generation_seed}-{decision_index:02d}-{digest}"


def _f0_fit_batch_ids(
    rows: Sequence[tuple[str, int, int, int | None]],
    *,
    optimizer_seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Rebuild the registered deterministic fit order as an independent oracle."""
    if len(rows) not in {288, 384}:
        raise ValueError("F0 fit row count differs from the protocol")
    stable_rows = [
        (
            _f0_stable_id(
                prompt_id=prompt,
                generation_seed=seed,
                decision_index=decision,
                split="train",
            ),
            fold,
        )
        for prompt, seed, decision, fold in rows
    ]
    if len({stable_id for stable_id, _fold in stable_rows}) != len(stable_rows):
        raise ValueError("F0 fit rows contain duplicate stable IDs")
    batches: list[tuple[str, ...]] = []
    epoch = 0
    while len(batches) < _F0_FIT_STEPS:
        ordered = sorted(
            stable_rows,
            key=lambda item: hashlib.sha256(
                f"sha256_epoch_order_v1\0{optimizer_seed}\0{epoch}\0{item[0]}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        for offset in range(0, len(ordered), _F0_FIT_BATCH_SIZE):
            batch = tuple(item[0] for item in ordered[offset : offset + _F0_FIT_BATCH_SIZE])
            if len(batch) != _F0_FIT_BATCH_SIZE:
                raise ValueError("F0 fit batch is incomplete")
            batches.append(batch)
            if len(batches) == _F0_FIT_STEPS:
                break
        epoch += 1
    return tuple(batches)


def _f0_expected_identity(
    kind: str, metadata: Mapping[str, Any], *, gate: Mapping[str, Any]
) -> dict[str, str]:
    model = metadata.get("model")
    role = metadata.get("role")
    preprocess = metadata.get("preprocess_hash")
    if kind == "reward_forward" and model == TOPIQ_NR_MODEL_ID:
        expected = {
            "model": TOPIQ_NR_MODEL_ID,
            "role": TOPIQ_NR_ROLE,
            "preprocess_hash": gate["witness_preprocess_sha256"],
            "model_config_sha256": gate["witness_config_sha256"],
            "model_asset_manifest_sha256": gate["witness_asset_manifest_sha256"],
        }
    elif kind in {"reward_forward", "reward_backward"}:
        expected = {
            "model": "ImageReward-v1.0",
            "role": "training_reward",
            "preprocess_hash": gate["reward_preprocess_sha256"],
            "model_config_sha256": gate["reward_config_sha256"],
            "model_asset_manifest_sha256": gate["reward_asset_manifest_sha256"],
        }
    else:
        expected = {
            "model": "repldm-runtime",
            "role": kind,
            "preprocess_hash": gate["reward_preprocess_sha256"],
            "model_config_sha256": gate["reward_config_sha256"],
            "model_asset_manifest_sha256": gate["reward_asset_manifest_sha256"],
        }
    if (
        model != expected["model"]
        or role != expected["role"]
        or preprocess != expected["preprocess_hash"]
    ):
        raise ValueError("F0 operation identity differs from the registered operation")
    return expected


def _validate_f0_operation_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    gate: Mapping[str, Any],
    contract: Mapping[str, Any],
    f0_run_contract_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    """Validate every reservation/receipt envelope before semantic matching."""
    reservations = {
        row["reservation_id"]: row
        for row in records
        if row.get("type") == "reservation"
    }
    receipts = {
        row["reservation_id"]: row
        for row in records
        if row.get("type") == "receipt"
    }
    if set(reservations) != set(receipts):
        raise ValueError("F0 ledger must contain exactly one receipt for every reservation")
    code_commit = gate.get("code_commit", contract.get("code_commit"))
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        raise ValueError("F0 gate does not bind a valid code commit")
    expected_code_hash = hashlib.sha256(code_commit.encode("ascii")).hexdigest()
    selected_payload = gate.get("selected_payload_sha256")
    _hash(selected_payload, label="F0 selected payload")
    contract_data = contract.get("data_manifest_sha256", selected_payload)
    _hash(contract_data, label="F0 data manifest")
    if contract_data != selected_payload:
        raise ValueError("F0 data manifest differs from the selected payload")
    common = {
        "run_contract_hash": f0_run_contract_sha256,
        "renderer_frame_contract_hash": gate["renderer_frame_contract_hash"],
        "calibration_hash": gate["calibration_hash"],
        "data_manifest_sha256": contract_data,
        "reward_config_sha256": gate["reward_config_sha256"],
    }
    pairs: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for reservation_id, reservation in reservations.items():
        receipt = receipts[reservation_id]
        kind = reservation.get("kind")
        amount = reservation.get("amount")
        if kind not in _F0_OPERATION_KINDS or amount != 1:
            raise ValueError("F0 ledger contains an unsupported or aggregated operation")
        metadata = reservation.get("metadata")
        result = receipt.get("result")
        if not isinstance(metadata, Mapping) or not isinstance(result, Mapping):
            raise ValueError("F0 operation pair is missing provenance")
        if set(metadata) != _F0_RESERVATION_FIELDS:
            raise ValueError("F0 reservation provenance fields differ from the operation schema")
        if set(result) != _F0_RESULT_FIELDS:
            raise ValueError("F0 receipt provenance fields differ from the operation schema")
        allocation = metadata.get("method_allocation")
        if allocation != {"kind": kind, "amount": 1, "method": "f0"}:
            raise ValueError("F0 operation allocation is not one registered call")
        if any(metadata.get(key) != value for key, value in common.items()):
            raise ValueError("F0 reservation is bound to a different run")
        if metadata.get("code_hash") != expected_code_hash:
            raise ValueError("F0 reservation code provenance differs from the run")
        identity = _f0_expected_identity(kind, metadata, gate=gate)
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("F0 reservation model provenance differs from the run")
        if any(result.get(key) != value for key, value in common.items()):
            raise ValueError("F0 receipt is bound to a different run")
        if any(result.get(key) != metadata.get(key) for key in (
            "model", "role", "model_config_sha256", "model_asset_manifest_sha256"
        )):
            raise ValueError("F0 receipt model provenance differs from its reservation")
        if result.get("reward_preprocess_hash") != metadata.get("preprocess_hash"):
            raise ValueError("F0 receipt preprocessing differs from its reservation")
        for field in ("image_hash", "output_hash"):
            _hash(result.get(field), label=f"F0 {kind} receipt {field}")
        timing = result.get("wall_seconds")
        if (
            isinstance(timing, bool)
            or not isinstance(timing, (int, float))
            or not math.isfinite(float(timing))
            or float(timing) < 0
        ):
            raise ValueError("F0 receipt timing is invalid")
        if result.get("failure") is not None:
            raise ValueError("F0 successful ledger validation found a failed receipt")
        expected_mode = {
            "unet_forward": "tensor",
            "scheduler_step": "tensor",
            "vae_decode": "tensor",
            "reward_forward": "scalar",
            "reward_backward": "gradient",
            "optimizer_step": "optimizer_state",
        }[kind]
        if result.get("scalar_or_gradient") != expected_mode:
            raise ValueError("F0 receipt operation mode differs from its kind")
        if kind not in {"reward_forward", "reward_backward"} and (
            result.get("image_hash") != result.get("output_hash")
            or result.get("cached_parent") is not None
            or result.get("parent_reservation_id") is not None
        ):
            raise ValueError("F0 non-reward receipt has an invalid parent or image binding")
        if kind in {"reward_forward", "reward_backward"}:
            parent_id = result.get("parent_reservation_id")
            if not isinstance(parent_id, str) or parent_id not in reservations:
                raise ValueError("F0 reward receipt has no durable parent")
            if not isinstance(result.get("cached_parent"), str):
                raise ValueError("F0 reward receipt has no cached parent hash")
        pairs[reservation_id] = (reservation, receipt)

    # Parent chains are checked after all reservations are indexed so a forged
    # forward/backward receipt cannot point at an unvalidated operation.
    for reservation_id, (reservation, receipt) in pairs.items():
        kind = reservation["kind"]
        if kind not in {"reward_forward", "reward_backward"}:
            continue
        result = receipt["result"]
        parent_id = result["parent_reservation_id"]
        parent_reservation, parent_receipt = pairs[parent_id]
        parent_result = parent_receipt["result"]
        if kind == "reward_forward":
            if parent_reservation["kind"] != "vae_decode":
                raise ValueError("F0 reward forward does not descend from a VAE decode")
            if result["cached_parent"] != parent_result["output_hash"]:
                raise ValueError("F0 reward forward parent hash is inconsistent")
            if tensor_operation_output_sha256(result["image_hash"]) != parent_result["output_hash"]:
                raise ValueError("F0 reward image differs from its VAE parent")
        else:
            if parent_reservation["kind"] != "reward_forward":
                raise ValueError("F0 reward backward does not descend from a reward forward")
            if result["cached_parent"] != parent_result["output_hash"]:
                raise ValueError("F0 reward backward parent hash is inconsistent")
            if result["image_hash"] != parent_result["image_hash"]:
                raise ValueError("F0 reward backward image differs from its forward parent")
        for field in ("split", "prompt", "seed", "step", "prefix", "branch", "action", "checkpoint_hash"):
            if reservation["metadata"].get(field) != parent_reservation["metadata"].get(field):
                raise ValueError("F0 reward parent context differs from its child")
    return reservations, pairs


def _validate_f0_semantic_operations(
    rows: Sequence[Mapping[str, Any]],
    *,
    reservations: Mapping[str, Mapping[str, Any]],
    pair_by_id: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    metric_contexts: Mapping[tuple[str, str, int, str, int | None], Mapping[str, Any]],
    gate: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    """Match every non-optimizer receipt to one registered F0 operation."""
    expected: Counter[tuple[Any, ...]] = Counter()

    def add(kind: str, context: Mapping[str, Any]) -> None:
        expected[_f0_operation_key(kind, context)] += 1

    def history(value: Any, *, label: str) -> list[Any]:
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"{label} must contain three decisions")
        for position, decision in enumerate(value):
            if (
                not isinstance(decision, list)
                or len(decision) != 1
                or not isinstance(decision[0], list)
                or len(decision[0]) != 6
            ):
                raise ValueError(f"{label} has an invalid shape at decision {position}")
            for coefficient in decision[0]:
                _finite(coefficient, label=f"{label} coefficient")
        return value

    def replace_context(
        source: Mapping[str, Any],
        *,
        step: int,
        prefix: int,
        branch: str,
        checkpoint: str,
        action: Any,
    ) -> dict[str, Any]:
        return {
            "split": source["split"],
            "prompt": source["prompt"],
            "seed": source["seed"],
            "step": step,
            "prefix": prefix,
            "branch": branch,
            "checkpoint_hash": checkpoint,
            "action": action,
        }

    # The metric contexts are the durable roots for each branch trajectory.
    # They also provide the exact action history produced by the collector.
    blocks = sorted(
        {
            (phase, prompt, seed)
            for phase, prompt, seed, branch, decision in metric_contexts
            if branch == "anchor" and decision is None
        }
    )
    if len(blocks) != 192:
        raise ValueError("F0 ledger must cover exactly 192 prompt-seed blocks")
    for phase, prompt, seed in blocks:
        anchor_meta = metric_contexts[(phase, prompt, seed, "anchor", None)]
        if anchor_meta.get("step") != _F0_TOTAL_STEPS - 1 or anchor_meta.get("prefix") != 0:
            raise ValueError("F0 anchor metric context is not the terminal C0 state")
        anchor_action = anchor_meta.get("action")
        if not isinstance(anchor_action, Mapping):
            raise ValueError("F0 anchor metric action is not a mapping")
        anchor_history = history(
            anchor_action.get("renderer_action_history"),
            label="F0 anchor action history",
        )
        if any(
            coefficient != 0.0
            for decision in anchor_history
            for coefficient in decision[0]
        ):
            raise ValueError("F0 anchor trajectory is not strict no-op")
        initial_checkpoint = gate["initial_renderer_state_sha256"]
        anchor_context = replace_context(
            anchor_meta,
            step=_F0_TOTAL_STEPS - 1,
            prefix=0,
            branch="anchor",
            checkpoint=initial_checkpoint,
            action=anchor_action,
        )
        add("vae_decode", anchor_context)
        # The shared anchor image is scored by both ImageReward and TOPIQ-NR.
        add("reward_forward", anchor_context)
        add("reward_forward", anchor_context)
        for step in range(_F0_TOTAL_STEPS):
            prior_count = sum(decision < step for decision in _F0_DECISION_INDICES)
            add(
                "unet_forward",
                replace_context(
                    anchor_meta,
                    step=step,
                    prefix=0,
                    branch="anchor",
                    checkpoint=initial_checkpoint,
                    action=anchor_history[:prior_count],
                ),
            )
            current_count = prior_count + int(step in _F0_DECISION_INDICES)
            add(
                "scheduler_step",
                replace_context(
                    anchor_meta,
                    step=step,
                    prefix=0,
                    branch="anchor",
                    checkpoint=initial_checkpoint,
                    action=anchor_history[:current_count],
                ),
            )

        for decision in _F0_DECISION_INDICES:
            position = _F0_DECISION_INDICES.index(decision)
            target_action = _f0_zero_history(position + 1)
            target_context = replace_context(
                anchor_meta,
                step=decision,
                prefix=0,
                branch="anchor",
                checkpoint=initial_checkpoint,
                action=target_action,
            )
            add("vae_decode", target_context)
            add("reward_forward", target_context)
            add("reward_backward", target_context)

            for branch in ("plus", "minus"):
                branch_meta = metric_contexts.get(
                    (phase, prompt, seed, branch, decision)
                )
                if branch_meta is None:
                    raise ValueError("F0 metric ledger is missing a target branch")
                branch_action = branch_meta.get("action")
                if not isinstance(branch_action, Mapping):
                    raise ValueError("F0 target metric action is not a mapping")
                branch_history = history(
                    branch_action.get("renderer_action_history"),
                    label=f"F0 {branch} action history",
                )
                for index, value in enumerate(branch_history):
                    if index != position and any(coefficient != 0.0 for coefficient in value[0]):
                        raise ValueError("F0 suffix changed an unregistered decision")
                checkpoint = gate["initial_renderer_state_sha256"]
                metric_context = replace_context(
                    branch_meta,
                    step=_F0_TOTAL_STEPS - 1,
                    prefix=0,
                    branch=branch,
                    checkpoint=checkpoint,
                    action=branch_action,
                )
                add("vae_decode", metric_context)
                # Each terminal metric image is consumed by two independent
                # scorers (ImageReward and TOPIQ-NR).  They share the exact
                # operation context and VAE parent, but each is still a
                # separately budgeted reward-forward receipt.
                add("reward_forward", metric_context)
                add("reward_forward", metric_context)
                for step in range(decision, _F0_TOTAL_STEPS):
                    prior_count = sum(item < step for item in _F0_DECISION_INDICES)
                    if step == decision:
                        prior_count = position
                    add(
                        "scheduler_step",
                        replace_context(
                            branch_meta,
                            step=step,
                            prefix=0,
                            branch=branch,
                            checkpoint=checkpoint,
                            action=branch_history[
                                : prior_count + int(step in _F0_DECISION_INDICES)
                            ],
                        ),
                    )
                for step in range(decision + 1, _F0_TOTAL_STEPS):
                    prior_count = sum(item < step for item in _F0_DECISION_INDICES)
                    add(
                        "unet_forward",
                        replace_context(
                            branch_meta,
                            step=step,
                            prefix=0,
                            branch=branch,
                            checkpoint=checkpoint,
                            action=branch_history[:prior_count],
                        ),
                    )

            realization = "out_of_fold_student" if phase == "train" else "final_teacher"
            realization_meta = metric_contexts.get(
                (phase, prompt, seed, realization, decision)
            )
            if realization_meta is None:
                raise ValueError("F0 metric ledger is missing a realization branch")
            realization_action = realization_meta.get("action")
            if not isinstance(realization_action, Mapping):
                raise ValueError("F0 realization metric action is not a mapping")
            realization_history = history(
                realization_action.get("renderer_action_history"),
                label="F0 realization action history",
            )
            if any(
                index != position and any(coefficient != 0.0 for coefficient in value[0])
                for index, value in enumerate(realization_history)
            ):
                raise ValueError("F0 realization changed an unregistered decision")
            realization_checkpoint = (
                gate["crossfit_renderer_state_sha256"][str(
                    next(
                        row["provenance"]["held_out_fold"]
                        for row in rows
                        if row["phase"] == phase
                        and row["prompt_id"] == prompt
                        and row["generation_seed"] == seed
                        and row["decision_index"] == decision
                    )
                )]
                if phase == "train"
                else gate["opsd_teacher_renderer_sha256"]
            )
            realization_context = replace_context(
                realization_meta,
                step=_F0_TOTAL_STEPS - 1,
                prefix=0,
                branch=realization,
                checkpoint=realization_checkpoint,
                action=realization_action,
            )
            add("vae_decode", realization_context)
            # Realization metrics also have two independent scorer forwards.
            add("reward_forward", realization_context)
            add("reward_forward", realization_context)
            for step in range(decision, _F0_TOTAL_STEPS):
                prior_count = sum(item < step for item in _F0_DECISION_INDICES)
                if step == decision:
                    prior_count = position
                add(
                    "scheduler_step",
                    replace_context(
                        realization_meta,
                        step=step,
                        prefix=0,
                        branch=realization,
                        checkpoint=realization_checkpoint,
                        action=realization_history[
                            : prior_count + int(step in _F0_DECISION_INDICES)
                        ],
                    ),
                )
            for step in range(decision + 1, _F0_TOTAL_STEPS):
                prior_count = sum(item < step for item in _F0_DECISION_INDICES)
                add(
                    "unet_forward",
                    replace_context(
                        realization_meta,
                        step=step,
                        prefix=0,
                        branch=realization,
                        checkpoint=realization_checkpoint,
                        action=realization_history[:prior_count],
                    ),
                )

    observed: Counter[tuple[Any, ...]] = Counter()
    for reservation_id, (reservation, _receipt) in pair_by_id.items():
        if reservation["kind"] != "optimizer_step":
            observed[_f0_operation_key(reservation["kind"], reservation["metadata"])] += 1
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        raise ValueError(
            "F0 semantic operation coverage differs from the registered plan; "
            f"missing={sum(missing.values())}, extra={sum(extra.values())}"
        )

    # Optimizer receipts are checked against the exact five deterministic fit
    # schedules.  This prevents a valid-looking optimizer count from hiding a
    # different data order or a cross-fit leakage.
    train_rows = [
        (
            str(row["prompt_id"]),
            int(row["generation_seed"]),
            int(row["decision_index"]),
            int(row["provenance"]["held_out_fold"]),
        )
        for row in rows
        if row.get("phase") == "train"
    ]
    if len(train_rows) != 384:
        raise ValueError("F0 optimizer ledger requires 384 training target rows")
    expected_roles = {
        **{
            f"crossfit-fold-{fold}": seed
            for fold, seed in enumerate(_F0_FOLD_OPTIMIZER_SEEDS)
        },
        "T_OPSD": _F0_FINAL_OPTIMIZER_SEED,
    }
    optimizer_groups: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for reservation_id, pair in pair_by_id.items():
        reservation, receipt = pair
        if reservation["kind"] != "optimizer_step":
            continue
        action = reservation["metadata"].get("action")
        if not isinstance(action, Mapping) or set(action) != {
            "fit_role", "optimizer_seed", "target_stable_ids"
        }:
            raise ValueError("F0 optimizer action is not registered")
        role = action.get("fit_role")
        if role not in expected_roles:
            raise ValueError("F0 optimizer receipt names an unknown fit role")
        optimizer_groups.setdefault(str(role), []).append(pair)
    if set(optimizer_groups) != set(expected_roles):
        raise ValueError("F0 optimizer ledger does not contain exactly five fits")
    for role, optimizer_seed in expected_roles.items():
        group = optimizer_groups[role]
        ordered = sorted(group, key=lambda pair: pair[0]["metadata"]["step"])
        if len(group) != _F0_FIT_STEPS or [
            pair[0]["metadata"].get("step") for pair in ordered
        ] != list(range(_F0_FIT_STEPS)):
            raise ValueError("F0 optimizer receipts are not one contiguous 200-step fit")
        fit_rows = (
            train_rows
            if role == "T_OPSD"
            else [row for row in train_rows if row[3] != int(role.rsplit("-", 1)[1])]
        )
        expected_batches = _f0_fit_batch_ids(fit_rows, optimizer_seed=optimizer_seed)
        for index, (reservation, receipt) in enumerate(ordered):
            metadata = reservation["metadata"]
            action = metadata["action"]
            if (
                metadata.get("split") != "train"
                or metadata.get("prompt") != f"{role}-update-{index:03d}"
                or metadata.get("step") != index
                or metadata.get("prefix") != 8
                or metadata.get("branch") != "f0"
                or action.get("optimizer_seed") != optimizer_seed
                or action.get("target_stable_ids") != list(expected_batches[index])
                or metadata.get("checkpoint_hash") == ""
            ):
                raise ValueError("F0 optimizer receipt differs from its registered fit step")
            result = receipt["result"]
            if result.get("parent_reservation_id") is not None or result.get("cached_parent") is not None:
                raise ValueError("F0 optimizer receipt has an unexpected parent")


def validate_f0_run_outputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    ledger: Mapping[str, Any],
    ledger_seal: Mapping[str, Any],
    gate: Mapping[str, Any],
    contract: Mapping[str, Any],
    f0_run_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconcile raw F0 metric rows with a verified sealed operation ledger."""
    runtime = contract.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("reward_dtype") != "float32":
        raise ValueError("F0 metric receipts require the frozen float32 reward dtype")
    ledger_path, ledger_binding = _read_bound_file(ledger, label="F0 ledger")
    seal_path, seal_binding = _read_bound_file(ledger_seal, label="F0 ledger seal")
    query_ledger = QueryLedger(
        ledger_path,
        F0_QUERY_BUDGET,
        run_contract=f0_run_contract_sha256,
        strict_provenance=True,
    )
    if query_ledger.seal_path != seal_path:
        raise ValueError("F0 ledger seal path does not match the ledger")
    records = query_ledger.verified_records()
    ledger_summary = query_ledger.summary()
    if (
        ledger_summary.get("budget") != F0_QUERY_BUDGET
        or ledger_summary.get("reserved") != F0_QUERY_BUDGET
        or ledger_summary.get("successful_amount") != F0_QUERY_BUDGET
        or ledger_summary.get("unfinished_reservations") != 0
        or any(ledger_summary.get("failed_amount", {}).values())
        or any(ledger_summary.get("remaining", {}).values())
    ):
        raise ValueError("F0 sealed ledger does not contain one complete successful budget")
    reservations, pair_by_id = _validate_f0_operation_pairs(
        records,
        gate=gate,
        contract=contract,
        f0_run_contract_sha256=f0_run_contract_sha256,
    )

    reward_identity = {
        "model": "ImageReward-v1.0",
        "role": "training_reward",
        "preprocess_hash": gate["reward_preprocess_sha256"],
        "model_config_sha256": gate["reward_config_sha256"],
        "model_asset_manifest_sha256": gate["reward_asset_manifest_sha256"],
    }
    witness_identity = {
        "model": TOPIQ_NR_MODEL_ID,
        "role": TOPIQ_NR_ROLE,
        "preprocess_hash": gate["witness_preprocess_sha256"],
        "model_config_sha256": gate["witness_config_sha256"],
        "model_asset_manifest_sha256": gate["witness_asset_manifest_sha256"],
    }
    runtime_identity = {
        "model": "repldm-runtime",
        "role": "vae_decode",
        "preprocess_hash": gate["reward_preprocess_sha256"],
        "model_config_sha256": gate["reward_config_sha256"],
        "model_asset_manifest_sha256": gate["reward_asset_manifest_sha256"],
    }
    bound = {
        "run_contract_hash": f0_run_contract_sha256,
        "renderer_frame_contract_hash": gate["renderer_frame_contract_hash"],
        "calibration_hash": gate["calibration_hash"],
        "data_manifest_sha256": gate["selected_payload_sha256"],
        "reward_config_sha256": gate["reward_config_sha256"],
    }
    context_fields = (
        "split", "prompt", "seed", "step", "prefix", "branch", "action",
        "checkpoint_hash",
    )
    score_uses: dict[str, list[tuple[str, str, int, int, str, str]]] = {}
    parent_ids: set[str] = set()
    target_records: set[str] = set()
    metric_contexts: dict[tuple[str, str, int, str, int | None], Mapping[str, Any]] = {}
    crossfit_hashes = gate.get("crossfit_renderer_state_sha256")
    if not isinstance(crossfit_hashes, Mapping) or set(crossfit_hashes) != {
        "0", "1", "2", "3"
    }:
        raise ValueError("F0 gate cross-fit renderer map is incomplete")
    crossfit_hashes = {
        key: _hash(value, label=f"F0 cross-fit renderer {key}")
        for key, value in crossfit_hashes.items()
    }

    for row in rows:
        phase = str(row["phase"])
        prompt = str(row["prompt_id"])
        seed = int(row["generation_seed"])
        decision = int(row["decision_index"])
        provenance = row["provenance"]
        held_out_fold = provenance["held_out_fold"]
        target_record = _hash(
            provenance["target_record_sha256"], label="F0 target record"
        )
        if target_record in target_records:
            raise ValueError("F0 metric rows reuse a target record")
        target_records.add(target_record)
        realization = "out_of_fold_student" if phase == "train" else "final_teacher"
        branch_names = {
            "anchor": "anchor",
            "plus": "plus",
            "minus": "minus",
            "realized": realization,
        }
        for branch, ledger_branch in branch_names.items():
            branch_value = provenance["branches"][branch]
            if branch != "realized":
                expected_checkpoint = gate["initial_renderer_state_sha256"]
            elif phase == "train":
                expected_checkpoint = crossfit_hashes[str(held_out_fold)]
            else:
                expected_checkpoint = gate["opsd_teacher_renderer_sha256"]
            if branch_value["renderer_state_sha256"] != expected_checkpoint:
                raise ValueError("F0 metric branch uses the wrong renderer checkpoint")
            image_sha256 = _hash(
                branch_value["image_sha256"], label=f"F0 {branch} image"
            )
            reward_value = row["reward_select"][ledger_branch]
            witness_value = row["topiq_nr"][ledger_branch]
            expected_outputs = {
                "image_reward": _float32_tensor_output_sha256(
                    reward_value, label=f"F0 {branch} ImageReward score"
                ),
                "topiq_nr": _float32_tensor_output_sha256(
                    witness_value, label=f"F0 {branch} TOPIQ-NR score"
                ),
            }
            score_parent: str | None = None
            score_metadata: Mapping[str, Any] | None = None
            for scorer, identity in (
                ("image_reward", reward_identity),
                ("topiq_nr", witness_identity),
            ):
                reference = branch_value[scorer]
                reservation_id = reference["reservation_id"]
                pair = pair_by_id.get(reservation_id)
                if pair is None:
                    raise ValueError("F0 metric row names no successful ledger receipt")
                reservation, receipt = pair
                metadata = reservation.get("metadata")
                result = receipt.get("result")
                if not isinstance(metadata, Mapping) or not isinstance(result, Mapping):
                    raise RuntimeError("F0 metric receipt pair is malformed")
                expected_output = expected_outputs[scorer]
                if (
                    reference["output_sha256"] != expected_output
                    or result.get("output_hash") != expected_output
                    or reservation.get("kind") != "reward_forward"
                    or receipt.get("kind") != "reward_forward"
                    or reservation.get("amount") != 1
                    or receipt.get("amount") != 1
                    or any(metadata.get(key) != value for key, value in identity.items())
                    or not _receipt_result_has_identity(result, identity)
                    or any(metadata.get(key) != value for key, value in bound.items())
                    or any(result.get(key) != value for key, value in bound.items())
                    or result.get("image_hash") != image_sha256
                    or result.get("scalar_or_gradient") != "scalar"
                    or result.get("failure") is not None
                    or metadata.get("split") != phase
                    or metadata.get("prompt") != prompt
                    or metadata.get("seed") != seed
                    or metadata.get("branch") != ledger_branch
                    or metadata.get("checkpoint_hash") != expected_checkpoint
                ):
                    raise ValueError("F0 metric receipt differs from its raw evidence")
                allocation = metadata.get("method_allocation")
                if not isinstance(allocation, Mapping) or dict(allocation) != {
                    "kind": "reward_forward", "amount": 1, "method": "f0"
                }:
                    raise ValueError("F0 metric receipt is outside the F0 allocation")
                _f0_metric_action(
                    metadata,
                    branch=ledger_branch,
                    target_record_sha256=target_record,
                )
                parent_id = result.get("parent_reservation_id")
                if not isinstance(parent_id, str) or not parent_id:
                    raise ValueError("F0 metric receipt has no VAE parent")
                if score_parent is not None and score_parent != parent_id:
                    raise ValueError("ImageReward and TOPIQ-NR do not share one VAE parent")
                if score_metadata is not None and any(
                    score_metadata.get(field) != metadata.get(field)
                    for field in context_fields
                ):
                    raise ValueError("F0 scorers do not share one operation context")
                score_parent = parent_id
                score_metadata = metadata
                score_uses.setdefault(reservation_id, []).append(
                    (phase, prompt, seed, decision, branch, scorer)
                )

            if score_parent is None or score_metadata is None:
                raise RuntimeError("F0 metric branch omitted its score receipts")
            parent_pair = pair_by_id.get(score_parent)
            if parent_pair is None:
                raise ValueError("F0 metric score has no successful VAE receipt")
            parent_reservation, parent_receipt = parent_pair
            parent_metadata = parent_reservation.get("metadata")
            parent_result = parent_receipt.get("result")
            if not isinstance(parent_metadata, Mapping) or not isinstance(parent_result, Mapping):
                raise RuntimeError("F0 VAE receipt pair is malformed")
            expected_image_output = tensor_operation_output_sha256(image_sha256)
            if (
                parent_reservation.get("kind") != "vae_decode"
                or parent_receipt.get("kind") != "vae_decode"
                or parent_reservation.get("amount") != 1
                or parent_receipt.get("amount") != 1
                or any(parent_metadata.get(key) != value for key, value in runtime_identity.items())
                or not _receipt_result_has_identity(parent_result, runtime_identity)
                or any(parent_metadata.get(key) != value for key, value in bound.items())
                or any(parent_result.get(key) != value for key, value in bound.items())
                or parent_result.get("output_hash") != expected_image_output
                or parent_result.get("image_hash") != expected_image_output
                or parent_result.get("scalar_or_gradient") != "tensor"
                or parent_result.get("cached_parent") is not None
                or parent_result.get("parent_reservation_id") is not None
                or parent_result.get("failure") is not None
                or any(
                    parent_metadata.get(field) != score_metadata.get(field)
                    for field in context_fields
                )
            ):
                raise ValueError("F0 metric image differs from its VAE parent receipt")
            for scorer in ("image_reward", "topiq_nr"):
                result = pair_by_id[branch_value[scorer]["reservation_id"]][1]["result"]
                if (
                    result.get("cached_parent") != expected_image_output
                    or result.get("parent_reservation_id") != score_parent
                ):
                    raise ValueError("F0 metric scorer breaks the VAE parent chain")
            parent_ids.add(score_parent)
            if score_metadata is None:
                raise RuntimeError("F0 metric context was not retained")
            context_key = (
                phase,
                prompt,
                seed,
                ledger_branch,
                None if branch == "anchor" else decision,
            )
            previous_context = metric_contexts.get(context_key)
            if previous_context is not None and any(
                previous_context.get(field) != score_metadata.get(field)
                for field in (
                    "split", "prompt", "seed", "step", "prefix", "branch",
                    "action", "checkpoint_hash",
                )
            ):
                raise ValueError("F0 metric trajectory context is inconsistent")
            metric_contexts[context_key] = score_metadata

    for uses in score_uses.values():
        anchor = all(use[4] == "anchor" for use in uses)
        if anchor:
            stable = {(use[0], use[1], use[2], use[5]) for use in uses}
            decisions = {use[3] for use in uses}
            if len(uses) != 3 or len(stable) != 1 or decisions != {8, 24, 40}:
                raise ValueError("F0 shared anchor receipt reuse differs from the protocol")
        elif len(uses) != 1:
            raise ValueError("F0 state-specific score receipt was reused")

    _validate_f0_semantic_operations(
        rows,
        reservations=reservations,
        pair_by_id=pair_by_id,
        metric_contexts=metric_contexts,
        gate=gate,
        contract=contract,
    )

    metric_reservations = {
        row["reservation_id"]
        for row in records
        if row.get("type") == "reservation"
        and isinstance(row.get("metadata"), Mapping)
        and isinstance(row["metadata"].get("action"), Mapping)
        and "f0_metric_branch" in row["metadata"]["action"]
        and row.get("kind") == "reward_forward"
    }
    metric_parents = {
        row["reservation_id"]
        for row in records
        if row.get("type") == "reservation"
        and isinstance(row.get("metadata"), Mapping)
        and isinstance(row["metadata"].get("action"), Mapping)
        and "f0_metric_branch" in row["metadata"]["action"]
        and row.get("kind") == "vae_decode"
    }
    if metric_reservations != set(score_uses) or metric_parents != parent_ids:
        raise ValueError("F0 raw evidence does not cover the metric receipt ledger one-to-one")
    return ledger_binding, seal_binding


def _validate_phase(
    value: Any,
    *,
    name: str,
    reward_scale: float,
    reward_statistics_sha256: str,
    f0_run_contract_sha256: str,
    screen_registration_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]]:
    expected_counts = {"train": (64, 384), "validation": (32, 192)}
    if not isinstance(value, Mapping) or set(value) != {
        "passed", "prompt_count", "state_count", "safety_violations", "metrics"
    }:
        raise ValueError(f"F0 {name} gate is incomplete")
    if type(value.get("passed")) is not bool:
        raise ValueError(f"F0 {name} passed claim must be boolean")
    for field in ("prompt_count", "state_count", "safety_violations"):
        if type(value.get(field)) is not int or int(value[field]) < 0:
            raise ValueError(f"F0 {name} {field} claim must be a non-negative integer")
    summary, metrics_binding, rows = load_f0_phase_evidence(
        value.get("metrics"),
        phase=name,
        reward_scale=reward_scale,
        reward_statistics_sha256=reward_statistics_sha256,
        f0_run_contract_sha256=f0_run_contract_sha256,
        screen_registration_sha256=screen_registration_sha256,
    )
    prompts, states = expected_counts[name]
    if summary.get("prompt_count") != prompts or summary.get("state_count") != states:
        raise ValueError(f"F0 {name} evidence counts differ from the protocol")
    claimed = {
        "passed": value.get("passed"),
        "prompt_count": value.get("prompt_count"),
        "state_count": value.get("state_count"),
        "safety_violations": value.get("safety_violations"),
    }
    recomputed = {
        "passed": summary.get("passed"),
        "prompt_count": summary.get("prompt_count"),
        "state_count": summary.get("state_count"),
        "safety_violations": summary.get("safety_violations"),
    }
    if claimed != recomputed:
        raise ValueError(f"F0 {name} gate claims differ from recomputed evidence")
    if summary.get("passed") is not True or summary.get("safety_violations") != 0:
        raise ValueError(f"F0 {name} gate did not pass the frozen requirements")
    result = dict(value)
    result["metrics"] = metrics_binding
    return result, metrics_binding, rows


def validate_f0_gate(
    value: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    result = _copy(value, label="F0 gate")
    required = {
        "schema", "status", "code_commit", "f0_run_contract_sha256",
        "selected_view_release_id", "selected_view_manifest_sha256",
        "selected_payload_sha256", "renderer_frame_contract_hash",
        "calibration_hash", "action_contract_hash", "initial_renderer_state_sha256",
        "reward_config_sha256", "reward_preprocess_sha256",
        "reward_asset_manifest_sha256", "witness_config_sha256",
        "witness_preprocess_sha256", "witness_asset_manifest_sha256",
        "reward_statistics_sha256", "opsd_teacher_state_manifest_sha256",
        "opsd_teacher_renderer_sha256", "crossfit_renderer_state_sha256",
        "train_gate", "validation_gate",
        "screen_registration", "ledger", "ledger_seal",
    }
    if set(result) != required or result.get("schema") != F0_GATE_SCHEMA:
        raise ValueError("F0 gate fields differ from the registered schema")
    if result.get("status") != "passed":
        raise ValueError("F0 gate has not passed")
    mapping = {
        "code_commit": "code_commit",
        "selected_view_release_id": "selected_view_release_id",
        "selected_view_manifest_sha256": "selected_view_manifest_sha256",
        "selected_payload_sha256": "selected_payload_manifest_sha256",
        "renderer_frame_contract_hash": "renderer_frame_contract_hash",
        "calibration_hash": "calibration_hash",
        "action_contract_hash": "action_contract_hash",
        "initial_renderer_state_sha256": "initial_renderer_state_sha256",
        "reward_config_sha256": "reward_config_sha256",
        "reward_preprocess_sha256": "reward_preprocess_sha256",
        "reward_asset_manifest_sha256": "reward_asset_manifest_sha256",
        "reward_statistics_sha256": "reward_statistics_sha256",
        "opsd_teacher_state_manifest_sha256": "opsd_teacher_state_manifest_sha256",
        "opsd_teacher_renderer_sha256": "opsd_teacher_renderer_sha256",
    }
    for gate_field, contract_field in mapping.items():
        if result.get(gate_field) != contract.get(contract_field):
            raise ValueError(f"F0 gate {gate_field} differs from the run contract")
    for field in (
        "witness_config_sha256",
        "witness_preprocess_sha256",
        "witness_asset_manifest_sha256",
    ):
        _hash(result.get(field), label=f"F0 gate {field}")
    if result.get("witness_preprocess_sha256") != TOPIQ_NR_PREPROCESS_SHA256:
        raise ValueError("F0 gate witness preprocess differs from frozen TOPIQ-NR")
    if _COMMIT_RE.fullmatch(str(result.get("code_commit"))) is None:
        raise ValueError("F0 gate code commit is invalid")
    f0_run_contract_sha256 = _hash(
        result.get("f0_run_contract_sha256"), label="F0 run contract"
    )
    reward_statistics_sha256 = _hash(
        result.get("reward_statistics_sha256"), label="F0 reward statistics"
    )
    reward_scale = _reward_scale_from_contract(contract)
    _registration, registration_binding = validate_f0_screen_registration_evidence(
        result.get("screen_registration"),
        f0_run_contract_sha256=f0_run_contract_sha256,
    )
    train, train_metrics, train_rows = _validate_phase(
        result.get("train_gate"),
        name="train",
        reward_scale=reward_scale,
        reward_statistics_sha256=reward_statistics_sha256,
        f0_run_contract_sha256=f0_run_contract_sha256,
        screen_registration_sha256=registration_binding["sha256"],
    )
    validation, validation_metrics, validation_rows = _validate_phase(
        result.get("validation_gate"),
        name="validation",
        reward_scale=reward_scale,
        reward_statistics_sha256=reward_statistics_sha256,
        f0_run_contract_sha256=f0_run_contract_sha256,
        screen_registration_sha256=registration_binding["sha256"],
    )
    result["train_gate"] = train
    result["validation_gate"] = validation
    result["screen_registration"] = registration_binding
    ledger_binding, seal_binding = validate_f0_run_outputs(
        (*train_rows, *validation_rows),
        ledger=result.get("ledger"),
        ledger_seal=result.get("ledger_seal"),
        gate=result,
        contract=contract,
        f0_run_contract_sha256=f0_run_contract_sha256,
    )
    result["ledger"] = ledger_binding
    result["ledger_seal"] = seal_binding
    evidence = (
        train_metrics,
        validation_metrics,
        registration_binding,
        ledger_binding,
        seal_binding,
    )
    return result, evidence


def validate_training_cohort(
    value: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    result = _copy(value, label="training cohort")
    if set(result) != {"schema", "status", "cohort_id", "methods", "optimization_seeds", "shared", "runtime_invariants"}:
        raise ValueError("training cohort fields differ from the registered schema")
    if result.get("schema") != TRAINING_COHORT_SCHEMA or result.get("status") != "registered":
        raise ValueError("training cohort is not a registered cohort")
    if not isinstance(result.get("cohort_id"), str) or not result["cohort_id"]:
        raise ValueError("training cohort ID is missing")
    if result["cohort_id"] != contract.get("cohort_id"):
        raise ValueError("training cohort ID differs from the run contract")
    if result.get("methods") != list(MAIN_METHODS):
        raise ValueError("training cohort must register all four primary methods")
    if result.get("optimization_seeds") != list(OPTIMIZATION_SEEDS):
        raise ValueError("training cohort must register all three optimization seeds")
    shared = result.get("shared")
    if not isinstance(shared, Mapping) or set(shared) != set(_SHARED_FIELDS):
        raise ValueError("training cohort shared contract is incomplete")
    for field in _SHARED_FIELDS:
        if shared.get(field) != contract.get(field):
            raise ValueError(f"training cohort shared field {field} differs")
    runtime = contract.get("runtime")
    invariants = result.get("runtime_invariants")
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(invariants, Mapping)
        or set(invariants) != set(_RUNTIME_INVARIANTS)
        or any(invariants.get(field) != runtime.get(field) for field in _RUNTIME_INVARIANTS)
    ):
        raise ValueError("training cohort runtime invariants differ")
    return result


__all__ = [
    "F0_GATE_SCHEMA",
    "MAIN_METHODS",
    "OPSD_TEACHER_STATE_SCHEMA",
    "OPTIMIZATION_SEEDS",
    "REWARD_STATISTICS_SCHEMA",
    "TRAINING_COHORT_SCHEMA",
    "validate_f0_gate",
    "validate_f0_run_outputs",
    "validate_file_binding",
    "validate_opsd_teacher_state",
    "validate_reward_statistics",
    "validate_training_cohort",
]
