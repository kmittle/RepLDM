"""Pure-stdlib contract for the adaptive-oracle engineering smoke.

This module defines tasks and validates JSON records.  It performs no file I/O,
generation, scoring, model loading, or device work.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence


EXPERIMENT_ID = "adaptive_oracle_engineering_v1"
SPLIT_ROLE = "engineering"
DESIGN_SCHEMA = "adaptive_oracle_engineering_design_v1"
TASK_SCHEMA = "adaptive_oracle_engineering_task_v1"
SIDECAR_SCHEMA = "adaptive_oracle_engineering_sidecar_v1"
EVIDENCE_SCOPE_SCHEMA = "adaptive_oracle_engineering_evidence_scope_v1"
BLOCK_EVIDENCE_SCHEMA = "adaptive_oracle_engineering_block_evidence_v2"
PROMPT_MANIFEST_SCHEMA = "adaptive_oracle_prompt_manifest_v1"
EXCLUSION_INVENTORY_SCHEMA = "adaptive_oracle_exclusion_inventory_v1"
EXCLUSION_INVENTORY_PATH = (
    "eval-pipeline/prompts/adaptive_oracle_exclusion_inventory_v1.json"
)
RANDOM_EDGE_COUNTER_SCHEMA = "ao-random-edge-counter-v1"
RANDOM_EDGE_COUNTER_SET_HASH_ALGORITHM = (
    "sha256-length-prefixed-sorted-d4-canonical-counter-bytes-v1"
)
SEED_SELECTION_NAMESPACE = "repldm-adaptive-oracle-engineering-v1-seed"
SEED_SELECTION_HASH_ALGORITHM = "sha256-first-8-hex-and-0x7fffffff-v1"
_CONTRACT_INITIALIZED = False

PROMPT_COUNT = 11
TASKS_PER_PROMPT = 15
TOTAL_TASK_COUNT = PROMPT_COUNT * TASKS_PER_PROMPT
GRID_SIZE = 16
WIDTH = 1024
HEIGHT = 1024
NUM_INFERENCE_STEPS = 50
CFG_SCALE = 7.5
SCHEDULER_CLASS = "EulerDiscreteScheduler"
SCHEDULER_CHURN = 0.0
SCHEDULER_PREDICTION_TYPE = "epsilon"
SCHEDULER_CONFIG_SHA256_V2 = (
    "6bf0509f22d8d3a06d6493c2291af8655f6f74846b2d2eae3cf71b5cda000102"
)
SCHEDULER_SCHEDULE_SHA256 = (
    "302d2452f411bf3eea64f8dd3530e232b95c23aed7b818ed6697982a4428c144"
)
SCHEDULER_TIMESTEPS = tuple(float(value) for value in range(981, 0, -20))
SCHEDULER_SIGMAS = (
    13.120415687561035,
    11.676050186157227,
    10.425045013427734,
    9.338079452514648,
    8.390686988830566,
    7.562390327453613,
    6.835997581481934,
    6.197037696838379,
    5.633296489715576,
    5.134433269500732,
    4.691673278808594,
    4.297549724578857,
    3.945692777633667,
    3.6306545734405518,
    3.3477649688720703,
    3.093010187149048,
    2.862931728363037,
    2.6545450687408447,
    2.465266227722168,
    2.2928547859191895,
    2.135362386703491,
    1.9910918474197388,
    1.858561396598816,
    1.7364743947982788,
    1.6236929893493652,
    1.519217610359192,
    1.422167181968689,
    1.3317636251449585,
    1.2473174333572388,
    1.1682164669036865,
    1.0939152240753174,
    1.02392578125,
    0.9578100442886353,
    0.8951724171638489,
    0.8356532454490662,
    0.7789238095283508,
    0.7246790528297424,
    0.672633171081543,
    0.6225128769874573,
    0.5740507245063782,
    0.5269757509231567,
    0.4810032844543457,
    0.43581610918045044,
    0.39103835821151733,
    0.3461872935295105,
    0.3005809485912323,
    0.25313183665275574,
    0.20179356634616852,
    0.14146365225315094,
    0.04131447896361351,
    0.0,
)
SCHEDULER_CONSTRUCTION_INIT_NOISE_SIGMA = 14.648818969726562
SCHEDULER_EFFECTIVE_INIT_NOISE_SIGMA = 13.158469200134277
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
MODEL_SNAPSHOT_MANIFEST_SHA256 = (
    "53ebd71cceb73d660ce163b94bab48f4834e494ff60cbbabcb1feffa77eb8f28"
)
ACTIVE_TARGET_UPDATE_RATIO = 0.02
TARGET_RATIO_TOLERANCE = 5e-4
HARD_UPDATE_CAP = 0.05
CHANNEL_MEAN_ABS_ERROR_MAX = 1e-4
CHANNEL_VARIANCE_RELATIVE_ERROR_MAX = 1e-3
CHANNEL_COVARIANCE_DRIFT_MAX = 0.01
PRED_ORIGINAL_SAMPLE_RELATIVE_L2_MAX = 0.01
EXPECTED_PREV_SAMPLE_RELATIVE_L2_MAX = 1e-3

FEATURE_BLOCK = "up_blocks.0"
FEATURE_INPUT_NAME = "hidden_states"
FEATURE_CFG_SHAPE = (2, 1280, 32, 32)
FEATURE_CONDITIONAL_SHAPE = (1, 1280, 32, 32)
CFG_ROW_ORDER = "unconditional_then_conditional"

_ORBIT_OFFSETS = MappingProxyType({
    "axis-r1": ((0, 1), (1, 0)),
    "diagonal-r1": ((1, 1), (1, -1)),
    "axis-r2": ((0, 2), (2, 0)),
})
ORBIT_OFFSETS = _ORBIT_OFFSETS
_ORBIT_EDGE_COUNTS = MappingProxyType({
    "axis-r1": 480,
    "diagonal-r1": 450,
    "axis-r2": 448,
})
ORBIT_EDGE_COUNTS = _ORBIT_EDGE_COUNTS
SIGNED_ORBIT_CYCLE = (
    ("axis-r1", 1),
    ("axis-r1", -1),
    ("diagonal-r1", 1),
    ("diagonal-r1", -1),
    ("axis-r2", 1),
    ("axis-r2", -1),
)

PROMPT_CSV_FIELDS = (
    "index",
    "TEXT",
    "bucket",
    "source_row",
    "source_category",
    "source_challenge",
    "source_note",
    "split",
    "prompt_row_id",
    "seed",
)
PROMPT_ROW_FIELDS = frozenset(PROMPT_CSV_FIELDS)
ACTION_FIELDS = frozenset(
    (
        "action_id",
        "family",
        "affinity_source",
        "orbit_name",
        "sign",
        "d4_canonical_offsets",
        "physical_no_op",
        "feature_hook_required",
        "random_counter_required",
        "selected_control",
        "control_cycle_index",
        "matched_feature_action_id",
        "random_edge_counter_schema",
        "target_update_ratio",
        "hard_update_cap",
        "basis_recomputed_each_step",
    )
)
_TASK_FIELDS = frozenset(
    (
        "schema",
        "experiment_id",
        "split_role",
        "task_id",
        "prompt",
        "action",
        "execution",
        "provenance",
        "task_sha256",
    )
)
_TASK_PROMPT_FIELDS = frozenset(
    (
        "index",
        "prompt_row_id",
        "prompt_sha256",
        "normalized_prompt_sha256",
        "source_row",
        "source_category",
        "source_challenge",
        "seed",
    )
)
_TASK_PROVENANCE_FIELDS = frozenset(
    (
        "contract_sha256",
        "primary_action_bank_sha256",
        "signed_orbit_cycle_sha256",
        "execution_contract_sha256",
        "sidecar_shape_sha256",
        "prompt_rows_sha256",
        "exclusion_inventory_sha256",
        "prompt_manifest_canonical_sha256",
        "prompt_csv_sha256",
        "prompt_selection_digest",
    )
)
_MANIFEST_PROMPT_FIELDS = frozenset(
    (
        "category",
        "challenge",
        "normalized_prompt_sha256",
        "prompt_row_id",
        "prompt_sha256",
        "rank_within_challenge",
        "selection_digest",
        "source_row",
    )
)
_EXECUTION_CONTRACT = MappingProxyType({
    "split_role": SPLIT_ROLE,
    "width": WIDTH,
    "height": HEIGHT,
    "num_inference_steps": NUM_INFERENCE_STEPS,
    "scheduler_class": SCHEDULER_CLASS,
    "scheduler_churn": SCHEDULER_CHURN,
    "scheduler_prediction_type": SCHEDULER_PREDICTION_TYPE,
    "scheduler_config_sha256_v2": SCHEDULER_CONFIG_SHA256_V2,
    "scheduler_schedule_sha256": SCHEDULER_SCHEDULE_SHA256,
    "scheduler_construction_init_noise_sigma": (
        SCHEDULER_CONSTRUCTION_INIT_NOISE_SIGMA
    ),
    "scheduler_effective_init_noise_sigma": SCHEDULER_EFFECTIVE_INIT_NOISE_SIGMA,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "model_snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
    "cfg_scale": CFG_SCALE,
    "grid_size": GRID_SIZE,
    "active_target_update_ratio": ACTIVE_TARGET_UPDATE_RATIO,
    "target_ratio_tolerance": TARGET_RATIO_TOLERANCE,
    "hard_update_cap": HARD_UPDATE_CAP,
    "channel_mean_abs_error_max": CHANNEL_MEAN_ABS_ERROR_MAX,
    "channel_variance_relative_error_max": CHANNEL_VARIANCE_RELATIVE_ERROR_MAX,
    "channel_covariance_drift_max": CHANNEL_COVARIANCE_DRIFT_MAX,
    "pred_original_sample_relative_l2_max": (
        PRED_ORIGINAL_SAMPLE_RELATIVE_L2_MAX
    ),
    "expected_prev_sample_relative_l2_max": (
        EXPECTED_PREV_SAMPLE_RELATIVE_L2_MAX
    ),
    "native_round_trip_max_abs_policy": "finite_diagnostic_only_not_a_pass_fail_bound",
    "unet_calls_per_step": 1,
    "scheduler_calls_per_step": 1,
    "decode_policy": "final_only",
    "scoring_authorized": False,
})
EXECUTION_CONTRACT = _EXECUTION_CONTRACT

_TOP_LEVEL_SIDECAR_FIELDS = frozenset(
    (
        "schema",
        "experiment_id",
        "task_id",
        "prompt",
        "action",
        "trajectory",
        "call_totals",
        "step_ledger",
        "evidence_scope",
    )
)
_TRAJECTORY_FIELDS = frozenset(
    (
        "trajectory_id",
        "initial_latent_sha256",
        "final_latent_sha256",
        "png_sha256",
        "physical_no_op",
        "scheduler_class",
        "scheduler_churn",
        "num_inference_steps",
        "width",
        "height",
        "cfg_scale",
        "grid_size",
        "model_id",
        "model_revision",
        "model_snapshot_manifest_sha256",
        "scheduler_prediction_type",
        "scheduler_config_sha256_v2",
        "scheduler_schedule_sha256",
        "scheduler_timesteps",
        "scheduler_sigmas",
        "scheduler_construction_init_noise_sigma",
        "scheduler_effective_init_noise_sigma",
    )
)
_CALL_TOTAL_FIELDS = frozenset(
    (
        "unet_calls",
        "scheduler_calls",
        "extra_unet_calls",
        "extra_scheduler_calls",
        "backbone_backward_calls",
        "intermediate_decode_calls",
        "final_decode_calls",
        "attention_probability_reads",
        "qk_reads",
    )
)
_STEP_FIELDS = frozenset(
    (
        "step_index",
        "affinity_source",
        "orbit_name",
        "sign",
        "unet_calls",
        "scheduler_calls",
        "extra_unet_calls",
        "extra_scheduler_calls",
        "backbone_backward_calls",
        "intermediate_decode_calls",
        "feature_hook",
        "random_edge_counter",
        "basis_sha256",
        "latent_before_sha256",
        "latent_after_sha256",
        "applied_update_ratio",
        "target_ratio_error",
        "channel_mean_abs_error",
        "channel_variance_relative_error",
        "channel_covariance_drift",
        "pred_original_sample_relative_l2_error",
        "expected_prev_sample_relative_l2_error",
        "native_round_trip_max_abs_error",
        "solver_target_update_ratio",
        "mapped_ratio_solver_evaluations",
        "finite",
        "cap_hit",
    )
)
_HOOK_FIELDS = frozenset(
    (
        "module_path",
        "input_name",
        "hook_calls",
        "consume_calls",
        "expected_cfg_shape",
        "captured_conditional_shape",
        "cfg_row_order",
        "detached",
    )
)
_COUNTER_FIELDS = frozenset(
    (
        "schema",
        "experiment_id",
        "split_role",
        "prompt_row_id",
        "seed",
        "step_index",
        "orbit_name",
        "undirected_edge_count",
        "counter_set_sha256",
    )
)
_EVIDENCE_SCOPE_FIELDS = frozenset(
    (
        "schema",
        "split_role",
        "generation_only",
        "scoring_authorized",
        "quality_outcomes_present",
        "prompt_csv_sha256",
        "prompt_manifest_canonical_sha256",
        "prompt_rows_sha256",
        "contract_sha256",
        "task_sha256",
    )
)
_BLOCK_EVIDENCE_FIELDS = frozenset(
    (
        "schema",
        "prompt_row_id",
        "record_count",
        "initial_latent_sha256",
        "task_ids_sha256",
        "trajectory_ids_sha256",
        "active_final_latents_sha256",
        "active_pngs_sha256",
        "trajectory_chains_sha256",
        "antithetic_step0_basis_pairs_sha256",
        "random_counter_pairs_sha256",
        "sidecars_sha256",
        "contract_sha256",
        "block_evidence_sha256",
    )
)
_FORBIDDEN_STATE_KEY_FRAGMENTS = (
    "cache",
    "cached",
    "later_state",
    "later_step_state",
    "base_trajectory_state",
    "reused_state",
    "reused_latent",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COUNTER_STRING_PATTERN = re.compile(r"[A-Za-z0-9._:-]+")
_STDERR_WARNING_TOKEN = re.compile(
    r"(?:"
    r"\b[A-Za-z0-9_]*warnings?\b"
    r"|\bwarn\b"
    r"|\bW[0-9]{3,8}\b"
    r"|\[\s*W(?:ARN(?:ING)?)?(?=[0-9\s:\]])"
    r")",
    re.IGNORECASE,
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one JSON value with the canonical contract encoding."""

    def plain_json_value(child: Any) -> Any:
        if isinstance(child, Mapping):
            return {key: plain_json_value(item) for key, item in child.items()}
        if isinstance(child, (list, tuple)):
            return [plain_json_value(item) for item in child]
        return child

    try:
        return json.dumps(
            plain_json_value(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not finite canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stderr_warning_line_count(stderr_text: str) -> int:
    """Count stderr lines containing a warning token or category name."""

    if not isinstance(stderr_text, str):
        raise TypeError("stderr warning evidence must be a string")
    return sum(
        _STDERR_WARNING_TOKEN.search(line) is not None
        for line in stderr_text.splitlines()
    )


def _require_same_json_value(value: Any, expected: Any, name: str) -> None:
    """Require value and JSON scalar/container types to match exactly."""

    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValueError(f"{name} differs from the exact contract")


def normalize_prompt(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split()).casefold()


def _counter_string(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or _COUNTER_STRING_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{name} must match [A-Za-z0-9._:-]+")
    return value


def _canonical_random_edge_nodes(
    grid_size: int, edge_low: int, edge_high: int
) -> tuple[int, int]:
    endpoints = (divmod(edge_low, grid_size), divmod(edge_high, grid_size))

    def images(y: int, x: int) -> tuple[tuple[int, int], ...]:
        last = grid_size - 1
        return (
            (y, x),
            (x, last - y),
            (last - y, last - x),
            (last - x, y),
            (y, last - x),
            (last - x, last - y),
            (last - y, x),
            (x, y),
        )

    endpoint_images = tuple(images(*endpoint) for endpoint in endpoints)
    representatives = []
    for transform_index in range(8):
        transformed = sorted(
            y * grid_size + x
            for y, x in (
                endpoint_images[0][transform_index],
                endpoint_images[1][transform_index],
            )
        )
        representatives.append((transformed[0], transformed[1]))
    return min(representatives)


def random_edge_counter_set_sha256(
    *,
    experiment_id: str,
    split_role: str,
    prompt_row_id: str,
    seed: int,
    step_index: int,
    orbit_name: str,
    grid_size: int = GRID_SIZE,
) -> str:
    """Recompute the registered D4-canonical random-counter set digest."""

    experiment = _counter_string(experiment_id, "experiment_id")
    split = _counter_string(split_role, "split_role")
    prompt_id = _counter_string(prompt_row_id, "prompt_row_id")
    parsed_seed = _require_int(seed, "seed", minimum=0)
    parsed_step = _require_int(step_index, "step_index", minimum=0)
    size = _require_int(grid_size, "grid_size", minimum=4)
    orbit = _counter_string(orbit_name, "orbit_name")
    offsets = _ORBIT_OFFSETS.get(orbit)
    if offsets is None:
        raise ValueError(f"orbit_name must be one of {tuple(_ORBIT_OFFSETS)}")

    canonical_edges: set[tuple[int, int]] = set()
    for dy, dx in offsets:
        for source_y in range(0, size - dy):
            source_x_start = 0 if dx >= 0 else -dx
            source_x_stop = size - dx if dx >= 0 else size
            for source_x in range(source_x_start, source_x_stop):
                target_y = source_y + dy
                target_x = source_x + dx
                actual = sorted(
                    (
                        source_y * size + source_x,
                        target_y * size + target_x,
                    )
                )
                canonical_edges.add(
                    _canonical_random_edge_nodes(size, actual[0], actual[1])
                )

    digest = hashlib.sha256()
    for edge_low, edge_high in sorted(canonical_edges):
        payload = canonical_json_bytes(
            {
                "schema": RANDOM_EDGE_COUNTER_SCHEMA,
                "experiment_id": experiment,
                "split_role": split,
                "prompt_row_id": prompt_id,
                "seed": parsed_seed,
                "step_index": parsed_step,
                "orbit_name": orbit,
                "edge_low": edge_low,
                "edge_high": edge_high,
            }
        )
        digest.update(len(payload).to_bytes(4, "big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_exact_keys(
    value: Any, expected: frozenset[str], name: str
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name)
    observed = set(mapping)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise ValueError(f"{name} fields differ: missing={missing}, extra={extra}")
    return mapping


def _require_int(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        result = int(value.strip())
    else:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _require_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _require_number_list(
    value: Any, name: str, *, expected_length: int
) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{name}[{index}] must be a finite number")
        parsed = float(item)
        if not math.isfinite(parsed):
            raise ValueError(f"{name}[{index}] must be a finite number")
        result.append(parsed)
    return result


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_bool(value: Any, expected: bool, name: str) -> None:
    if value is not expected:
        raise ValueError(f"{name} must be {expected}")


def _static_contract_payload() -> dict[str, Any]:
    return {
        "schemas": {
            "design": DESIGN_SCHEMA,
            "task": TASK_SCHEMA,
            "sidecar": SIDECAR_SCHEMA,
            "evidence_scope": EVIDENCE_SCOPE_SCHEMA,
            "block_evidence": BLOCK_EVIDENCE_SCHEMA,
            "prompt_manifest": PROMPT_MANIFEST_SCHEMA,
            "exclusion_inventory": EXCLUSION_INVENTORY_SCHEMA,
            "random_edge_counter": RANDOM_EDGE_COUNTER_SCHEMA,
        },
        "experiment_id": EXPERIMENT_ID,
        "split_role": SPLIT_ROLE,
        "prompt_count": PROMPT_COUNT,
        "tasks_per_prompt": TASKS_PER_PROMPT,
        "total_task_count": TOTAL_TASK_COUNT,
        "execution": EXECUTION_CONTRACT,
        "feature": {
            "block": FEATURE_BLOCK,
            "input_name": FEATURE_INPUT_NAME,
            "cfg_shape": FEATURE_CFG_SHAPE,
            "conditional_shape": FEATURE_CONDITIONAL_SHAPE,
            "cfg_row_order": CFG_ROW_ORDER,
        },
        "orbit_offsets": ORBIT_OFFSETS,
        "orbit_edge_counts": ORBIT_EDGE_COUNTS,
        "signed_orbit_cycle": SIGNED_ORBIT_CYCLE,
        "random_edge_counter_set_hash_algorithm": (
            RANDOM_EDGE_COUNTER_SET_HASH_ALGORITHM
        ),
        "seed_selection": {
            "namespace": SEED_SELECTION_NAMESPACE,
            "hash_algorithm": SEED_SELECTION_HASH_ALGORITHM,
        },
        "fields": {
            "prompt_csv": sorted(PROMPT_ROW_FIELDS),
            "action": sorted(ACTION_FIELDS),
            "task": sorted(_TASK_FIELDS),
            "task_prompt": sorted(_TASK_PROMPT_FIELDS),
            "task_provenance": sorted(_TASK_PROVENANCE_FIELDS),
            "sidecar": sorted(_TOP_LEVEL_SIDECAR_FIELDS),
            "trajectory": sorted(_TRAJECTORY_FIELDS),
            "call_totals": sorted(_CALL_TOTAL_FIELDS),
            "step": sorted(_STEP_FIELDS),
            "hook": sorted(_HOOK_FIELDS),
            "counter": sorted(_COUNTER_FIELDS),
            "evidence_scope": sorted(_EVIDENCE_SCOPE_FIELDS),
            "block_evidence": sorted(_BLOCK_EVIDENCE_FIELDS),
        },
    }


def _require_static_contract() -> None:
    if not _CONTRACT_INITIALIZED:
        return
    if canonical_sha256(_static_contract_payload()) != _STATIC_CONTRACT_SHA256:
        raise RuntimeError("adaptive-oracle module constants drifted after import")


def _sign_label(sign: int) -> str:
    if sign == 1:
        return "pos"
    if sign == -1:
        return "neg"
    raise ValueError("active action sign must be -1 or +1")


def _orbit_slug(orbit_name: str) -> str:
    return orbit_name.replace("-", "_")


def _feature_action_id(orbit_name: str, sign: int) -> str:
    return f"p_{_orbit_slug(orbit_name)}_{_sign_label(sign)}"


def _active_action(
    *,
    family: str,
    affinity_source: str,
    orbit_name: str,
    sign: int,
    selected_control: bool,
    control_cycle_index: Optional[int],
) -> dict[str, Any]:
    prefix = family.casefold()
    action_id = f"{prefix}_{_orbit_slug(orbit_name)}_{_sign_label(sign)}"
    return {
        "action_id": action_id,
        "family": family,
        "affinity_source": affinity_source,
        "orbit_name": orbit_name,
        "sign": sign,
        "d4_canonical_offsets": [
            list(value) for value in _ORBIT_OFFSETS[orbit_name]
        ],
        "physical_no_op": False,
        "feature_hook_required": affinity_source == "feature",
        "random_counter_required": affinity_source == "random_edge",
        "selected_control": selected_control,
        "control_cycle_index": control_cycle_index,
        "matched_feature_action_id": _feature_action_id(orbit_name, sign),
        "random_edge_counter_schema": (
            RANDOM_EDGE_COUNTER_SCHEMA if affinity_source == "random_edge" else None
        ),
        "target_update_ratio": ACTIVE_TARGET_UPDATE_RATIO,
        "hard_update_cap": HARD_UPDATE_CAP,
        "basis_recomputed_each_step": True,
    }


def canonical_primary_action_bank() -> list[dict[str, Any]]:
    _require_static_contract()
    bank = [
        {
            "action_id": "p0",
            "family": "P",
            "affinity_source": "no_op",
            "orbit_name": None,
            "sign": 0,
            "d4_canonical_offsets": [],
            "physical_no_op": True,
            "feature_hook_required": False,
            "random_counter_required": False,
            "selected_control": False,
            "control_cycle_index": None,
            "matched_feature_action_id": "p0",
            "random_edge_counter_schema": None,
            "target_update_ratio": 0.0,
            "hard_update_cap": HARD_UPDATE_CAP,
            "basis_recomputed_each_step": False,
        }
    ]
    bank.extend(
        _active_action(
            family="P",
            affinity_source="feature",
            orbit_name=orbit_name,
            sign=sign,
            selected_control=False,
            control_cycle_index=None,
        )
        for orbit_name, sign in SIGNED_ORBIT_CYCLE
    )
    bank.extend(
        _active_action(
            family="R",
            affinity_source="random_edge",
            orbit_name=orbit_name,
            sign=sign,
            selected_control=False,
            control_cycle_index=None,
        )
        for orbit_name, sign in SIGNED_ORBIT_CYCLE
    )
    return [dict(action) for action in bank]


def canonical_selected_controls(prompt_index: int) -> list[dict[str, Any]]:
    _require_static_contract()
    prompt_index = _require_int(prompt_index, "prompt_index", minimum=0)
    cycle_index = prompt_index % len(SIGNED_ORBIT_CYCLE)
    orbit_name, sign = SIGNED_ORBIT_CYCLE[cycle_index]
    return [
        _active_action(
            family="U",
            affinity_source="uniform_local",
            orbit_name=orbit_name,
            sign=sign,
            selected_control=True,
            control_cycle_index=cycle_index,
        ),
        _active_action(
            family="X",
            affinity_source="predicted_clean",
            orbit_name=orbit_name,
            sign=sign,
            selected_control=True,
            control_cycle_index=cycle_index,
        ),
    ]


PRIMARY_ACTION_BANK_SHA256 = canonical_sha256(canonical_primary_action_bank())
SIGNED_ORBIT_CYCLE_SHA256 = canonical_sha256(SIGNED_ORBIT_CYCLE)
EXECUTION_CONTRACT_SHA256 = canonical_sha256(_EXECUTION_CONTRACT)
SIDECAR_SHAPE_SHA256 = canonical_sha256(
    {
        "top_level": sorted(_TOP_LEVEL_SIDECAR_FIELDS),
        "trajectory": sorted(_TRAJECTORY_FIELDS),
        "call_totals": sorted(_CALL_TOTAL_FIELDS),
        "step": sorted(_STEP_FIELDS),
        "hook": sorted(_HOOK_FIELDS),
        "counter": sorted(_COUNTER_FIELDS),
        "evidence_scope": sorted(_EVIDENCE_SCOPE_FIELDS),
        "block_evidence": sorted(_BLOCK_EVIDENCE_FIELDS),
    }
)
CONTRACT_SHA256 = canonical_sha256(
    {
        "schema": DESIGN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "prompt_count": PROMPT_COUNT,
        "tasks_per_prompt": TASKS_PER_PROMPT,
        "primary_action_bank_sha256": PRIMARY_ACTION_BANK_SHA256,
        "signed_orbit_cycle_sha256": SIGNED_ORBIT_CYCLE_SHA256,
        "execution_contract_sha256": EXECUTION_CONTRACT_SHA256,
        "sidecar_shape_sha256": SIDECAR_SHAPE_SHA256,
        "orbit_offsets": _ORBIT_OFFSETS,
        "orbit_edge_counts": _ORBIT_EDGE_COUNTS,
        "feature_contract": {
            "block": FEATURE_BLOCK,
            "input_name": FEATURE_INPUT_NAME,
            "cfg_shape": FEATURE_CFG_SHAPE,
            "conditional_shape": FEATURE_CONDITIONAL_SHAPE,
            "cfg_row_order": CFG_ROW_ORDER,
        },
        "random_edge_counter_schema": RANDOM_EDGE_COUNTER_SCHEMA,
        "random_edge_counter_set_hash_algorithm": (
            RANDOM_EDGE_COUNTER_SET_HASH_ALGORITHM
        ),
        "seed_selection_namespace": SEED_SELECTION_NAMESPACE,
        "seed_selection_hash_algorithm": SEED_SELECTION_HASH_ALGORITHM,
    }
)
_STATIC_CONTRACT_SHA256 = canonical_sha256(_static_contract_payload())
_CONTRACT_INITIALIZED = True


def canonical_hashes() -> dict[str, str]:
    _require_static_contract()
    return {
        "contract_sha256": CONTRACT_SHA256,
        "primary_action_bank_sha256": PRIMARY_ACTION_BANK_SHA256,
        "signed_orbit_cycle_sha256": SIGNED_ORBIT_CYCLE_SHA256,
        "execution_contract_sha256": EXECUTION_CONTRACT_SHA256,
        "sidecar_shape_sha256": SIDECAR_SHAPE_SHA256,
    }


def _bucket(challenge: str) -> str:
    return challenge.casefold().replace(" & ", "-and-").replace(" ", "-")


def _prompt_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=PROMPT_CSV_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def validate_prompt_assets(
    prompt_rows: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    _require_static_contract()
    rows = list(prompt_rows)
    if len(rows) != PROMPT_COUNT:
        raise ValueError(f"adaptive-oracle engineering prompts must contain {PROMPT_COUNT} rows")
    normalized_rows = []
    observed_prompts = set()
    observed_source_rows = set()
    observed_challenges = set()
    observed_seeds = set()
    for expected_index, raw_row in enumerate(rows):
        row = _require_exact_keys(raw_row, PROMPT_ROW_FIELDS, f"prompt row {expected_index}")
        index = _require_int(row["index"], "prompt index", minimum=0)
        if index != expected_index:
            raise ValueError("prompt indices must be contiguous and ordered from zero")
        prompt = row["TEXT"]
        challenge = row["source_challenge"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt TEXT must be non-empty")
        if not isinstance(challenge, str) or not challenge.strip():
            raise ValueError("source_challenge must be non-empty")
        if row["split"] != SPLIT_ROLE:
            raise ValueError("every prompt row must use the engineering split")
        if row["prompt_row_id"] != f"engineering-{expected_index + 1:04d}":
            raise ValueError("prompt_row_id differs from the canonical row identity")
        if row["bucket"] != _bucket(challenge):
            raise ValueError("prompt bucket differs from its source challenge")
        normalized_prompt = normalize_prompt(prompt)
        source_row = _require_int(row["source_row"], "source_row", minimum=0)
        seed = _require_int(row["seed"], "seed", minimum=1)
        if normalized_prompt in observed_prompts:
            raise ValueError("engineering prompts are not normalized-text unique")
        if source_row in observed_source_rows:
            raise ValueError("engineering prompt source rows are not unique")
        if challenge in observed_challenges:
            raise ValueError("engineering source challenges are not unique")
        observed_prompts.add(normalized_prompt)
        observed_source_rows.add(source_row)
        observed_challenges.add(challenge)
        observed_seeds.add(seed)
        normalized_rows.append(
            {
                "index": index,
                "prompt_row_id": row["prompt_row_id"],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "normalized_prompt_sha256": hashlib.sha256(
                    normalized_prompt.encode("utf-8")
                ).hexdigest(),
                "source_row": source_row,
                "source_category": str(row["source_category"]),
                "source_challenge": challenge,
                "seed": seed,
            }
        )
    if len(observed_seeds) != 1:
        raise ValueError("engineering prompts must share one global seed")
    global_seed = next(iter(observed_seeds))

    manifest = _require_mapping(manifest, "prompt manifest")
    if manifest.get("schema") != PROMPT_MANIFEST_SCHEMA:
        raise ValueError("prompt manifest schema differs from the adaptive-oracle contract")
    if manifest.get("status") != "registration_only_gpu_not_authorized":
        raise ValueError("prompt manifest must remain registration-only")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("prompt manifest experiment_id differs")
    exclusion_inventory = _require_exact_keys(
        manifest.get("exclusion_inventory"),
        frozenset({"schema", "path", "sha256"}),
        "prompt exclusion inventory binding",
    )
    if exclusion_inventory["schema"] != EXCLUSION_INVENTORY_SCHEMA:
        raise ValueError("prompt exclusion inventory schema differs")
    if exclusion_inventory["path"] != EXCLUSION_INVENTORY_PATH:
        raise ValueError("prompt exclusion inventory path differs")
    exclusion_inventory_sha256 = _require_sha256(
        exclusion_inventory["sha256"], "prompt exclusion inventory SHA-256"
    )
    collisions = _require_mapping(manifest.get("collisions"), "prompt collisions")
    if set(collisions) != {
        "normalized_prompts",
        "source_rows",
        "used_or_reserved_seeds",
    } or any(collisions[key] != [] for key in collisions):
        raise ValueError("prompt manifest collision lists must be present and empty")
    engineering = _require_mapping(manifest.get("engineering"), "engineering prompts")
    if engineering.get("prompt_count") != PROMPT_COUNT or engineering.get(
        "challenge_count"
    ) != PROMPT_COUNT:
        raise ValueError("prompt manifest engineering counts differ")
    if engineering.get("counts_per_challenge") != 1:
        raise ValueError("prompt manifest must select one row per challenge")
    prompt_csv_sha256 = _require_sha256(
        engineering.get("csv_sha256"), "prompt CSV SHA-256"
    )
    observed_prompt_csv_sha256 = hashlib.sha256(_prompt_csv_bytes(rows)).hexdigest()
    if observed_prompt_csv_sha256 != prompt_csv_sha256:
        raise ValueError("prompt rows do not match the registered CSV SHA-256")
    selection_digest = _require_sha256(
        engineering.get("selection_digest"), "prompt selection digest"
    )
    expected_normalized_digests = [
        row["normalized_prompt_sha256"] for row in normalized_rows
    ]
    if engineering.get("normalized_prompt_digests") != expected_normalized_digests:
        raise ValueError("prompt manifest normalized prompt digests differ")
    manifest_prompt_rows = engineering.get("prompts")
    if not isinstance(manifest_prompt_rows, list) or len(manifest_prompt_rows) != PROMPT_COUNT:
        raise ValueError("prompt manifest must contain 11 prompt identities")
    for normalized, manifest_row in zip(normalized_rows, manifest_prompt_rows):
        manifest_row = _require_exact_keys(
            manifest_row,
            _MANIFEST_PROMPT_FIELDS,
            "manifest prompt identity",
        )
        expected_identity = {
            "prompt_row_id": normalized["prompt_row_id"],
            "category": normalized["source_category"],
            "challenge": normalized["source_challenge"],
            "prompt_sha256": normalized["prompt_sha256"],
            "normalized_prompt_sha256": normalized["normalized_prompt_sha256"],
            "source_row": normalized["source_row"],
            "rank_within_challenge": 0,
        }
        for key, expected_value in expected_identity.items():
            _require_same_json_value(
                manifest_row.get(key),
                expected_value,
                f"manifest prompt identity field {key!r}",
            )
        _require_sha256(
            manifest_row.get("selection_digest"),
            "manifest prompt selection digest",
        )
    if canonical_sha256(manifest_prompt_rows) != selection_digest:
        raise ValueError("prompt manifest selection digest is inconsistent")

    seed_registration = _require_mapping(
        manifest.get("seed_registration"), "seed registration"
    )
    if seed_registration.get("namespace") != SEED_SELECTION_NAMESPACE:
        raise ValueError("seed selection namespace differs")
    if seed_registration.get("global_seed_count") != 1:
        raise ValueError("seed registration must contain exactly one global seed")
    if seed_registration.get("reuse_policy") != "forbidden_after_the_first_generation_attempt":
        raise ValueError("seed reuse policy differs from the retired-on-use contract")
    registered_seed = _require_mapping(
        seed_registration.get("engineering_seed"), "engineering seed"
    )
    registered_seed_value = _require_int(
        registered_seed.get("seed"), "registered seed", minimum=1
    )
    if registered_seed_value != global_seed:
        raise ValueError("prompt rows differ from the registered global seed")
    seed_counter_raw = registered_seed.get("counter")
    seed_counter = _require_int(seed_counter_raw, "engineering seed counter", minimum=0)
    if canonical_json_bytes(seed_counter_raw) != canonical_json_bytes(seed_counter):
        raise ValueError("engineering seed counter must be a canonical integer")
    seed_payload = f"{SEED_SELECTION_NAMESPACE}:engineering:{seed_counter}"
    expected_seed_digest = hashlib.sha256(seed_payload.encode("ascii")).hexdigest()
    expected_seed_value = int(expected_seed_digest[:8], 16) & 0x7FFFFFFF
    observed_seed_digest = _require_sha256(
        registered_seed.get("selection_digest"), "engineering seed selection digest"
    )
    if observed_seed_digest != expected_seed_digest:
        raise ValueError("engineering seed selection digest is inconsistent")
    if registered_seed_value != expected_seed_value:
        raise ValueError("engineering seed differs from its deterministic selection digest")
    _require_bool(registered_seed.get("retired_on_use"), True, "retired_on_use")
    if registered_seed.get("status") != "reserved_retired_on_use":
        raise ValueError("engineering seed is not marked reserved_retired_on_use")
    if registered_seed.get("retirement_trigger") != (
        "first_generation_attempt_regardless_of_outcome"
    ):
        raise ValueError("engineering seed retirement trigger differs")
    return {
        "rows": normalized_rows,
        "global_seed": global_seed,
        "prompt_rows_sha256": canonical_sha256(normalized_rows),
        "exclusion_inventory_sha256": exclusion_inventory_sha256,
        "prompt_manifest_canonical_sha256": canonical_sha256(manifest),
        "prompt_csv_sha256": prompt_csv_sha256,
        "selection_digest": selection_digest,
    }


def _task_without_hash(
    prompt: Mapping[str, Any],
    action: Mapping[str, Any],
    prompt_assets: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = f"{prompt['prompt_row_id']}--{action['action_id']}"
    return {
        "schema": TASK_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "split_role": SPLIT_ROLE,
        "task_id": task_id,
        "prompt": dict(prompt),
        "action": dict(action),
        "execution": dict(_EXECUTION_CONTRACT),
        "provenance": {
            "contract_sha256": CONTRACT_SHA256,
            "primary_action_bank_sha256": PRIMARY_ACTION_BANK_SHA256,
            "signed_orbit_cycle_sha256": SIGNED_ORBIT_CYCLE_SHA256,
            "execution_contract_sha256": EXECUTION_CONTRACT_SHA256,
            "sidecar_shape_sha256": SIDECAR_SHAPE_SHA256,
            "prompt_rows_sha256": prompt_assets["prompt_rows_sha256"],
            "exclusion_inventory_sha256": prompt_assets[
                "exclusion_inventory_sha256"
            ],
            "prompt_manifest_canonical_sha256": prompt_assets[
                "prompt_manifest_canonical_sha256"
            ],
            "prompt_csv_sha256": prompt_assets["prompt_csv_sha256"],
            "prompt_selection_digest": prompt_assets["selection_digest"],
        },
    }


def build_engineering_design(
    prompt_rows: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    prompt_assets = validate_prompt_assets(prompt_rows, manifest)
    primary = canonical_primary_action_bank()
    if len(primary) != 13 or len({row["action_id"] for row in primary}) != 13:
        raise RuntimeError("canonical primary action bank is invalid")
    tasks = []
    for prompt in prompt_assets["rows"]:
        actions = primary + canonical_selected_controls(prompt["index"])
        if len(actions) != TASKS_PER_PROMPT or len(
            {action["action_id"] for action in actions}
        ) != TASKS_PER_PROMPT:
            raise RuntimeError("canonical prompt action block is invalid")
        for action in actions:
            if set(action) != set(ACTION_FIELDS):
                raise RuntimeError("canonical action fields drifted")
            task = _task_without_hash(prompt, action, prompt_assets)
            task["task_sha256"] = canonical_sha256(task)
            tasks.append(task)
    if len(tasks) != TOTAL_TASK_COUNT or len({task["task_id"] for task in tasks}) != len(
        tasks
    ):
        raise RuntimeError("adaptive-oracle engineering design is not 165 unique tasks")
    observed_cycle = [
        (
            canonical_selected_controls(index)[0]["orbit_name"],
            canonical_selected_controls(index)[0]["sign"],
        )
        for index in range(PROMPT_COUNT)
    ]
    if set(observed_cycle) != set(SIGNED_ORBIT_CYCLE):
        raise RuntimeError("selected controls do not cover all signed orbits")
    design = {
        "schema": DESIGN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "contract_only_gpu_not_authorized",
        "contract_hashes": canonical_hashes(),
        "prompt_assets": {
            key: value
            for key, value in prompt_assets.items()
            if key not in {"rows", "global_seed"}
        }
        | {"global_seed": prompt_assets["global_seed"]},
        "task_count": len(tasks),
        "tasks_per_prompt": TASKS_PER_PROMPT,
        "tasks_sha256": canonical_sha256(tasks),
        "tasks": tasks,
    }
    design["design_sha256"] = canonical_sha256(design)
    return design


def validate_engineering_design(
    design: Mapping[str, Any],
    prompt_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_engineering_design(prompt_rows, manifest)
    if canonical_json_bytes(design) != canonical_json_bytes(expected):
        raise ValueError("engineering design differs from the canonical contract")
    return {
        "task_count": TOTAL_TASK_COUNT,
        "tasks_sha256": expected["tasks_sha256"],
        "design_sha256": expected["design_sha256"],
        **canonical_hashes(),
    }


def _validate_expected_task(value: Any) -> Mapping[str, Any]:
    task = _require_exact_keys(value, _TASK_FIELDS, "registered task")
    fixed_fields = {
        "schema": TASK_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "split_role": SPLIT_ROLE,
    }
    for key, expected_value in fixed_fields.items():
        _require_same_json_value(
            task.get(key), expected_value, f"registered task field {key!r}"
        )

    prompt = _require_exact_keys(
        task.get("prompt"), _TASK_PROMPT_FIELDS, "registered task prompt"
    )
    prompt_index = _require_int(prompt.get("index"), "task prompt index", minimum=0)
    if prompt_index >= PROMPT_COUNT:
        raise ValueError("task prompt index is outside the engineering design")
    _require_same_json_value(prompt.get("index"), prompt_index, "task prompt index")
    source_row = _require_int(
        prompt.get("source_row"), "task prompt source_row", minimum=0
    )
    _require_same_json_value(
        prompt.get("source_row"), source_row, "task prompt source_row"
    )
    seed = _require_int(prompt.get("seed"), "task prompt seed", minimum=1)
    _require_same_json_value(prompt.get("seed"), seed, "task prompt seed")
    expected_prompt_row_id = f"engineering-{prompt_index + 1:04d}"
    _require_same_json_value(
        prompt.get("prompt_row_id"), expected_prompt_row_id, "task prompt row id"
    )
    for key in ("prompt_sha256", "normalized_prompt_sha256"):
        _require_sha256(prompt.get(key), f"task prompt {key}")
    for key in ("source_category", "source_challenge"):
        if not isinstance(prompt.get(key), str) or not prompt[key].strip():
            raise ValueError(f"task prompt {key} must be a non-empty string")

    action = _require_exact_keys(
        task.get("action"), ACTION_FIELDS, "registered task action"
    )
    action_id = action.get("action_id")
    canonical_actions = {
        row["action_id"]: row
        for row in (
            canonical_primary_action_bank()
            + canonical_selected_controls(prompt_index)
        )
    }
    if action_id not in canonical_actions:
        raise ValueError("registered task action is not in its canonical prompt block")
    _require_same_json_value(
        action, canonical_actions[action_id], "registered task action"
    )
    expected_task_id = f"{expected_prompt_row_id}--{action_id}"
    _require_same_json_value(task.get("task_id"), expected_task_id, "registered task id")
    _require_same_json_value(
        task.get("execution"), _EXECUTION_CONTRACT, "registered task execution"
    )

    provenance = _require_exact_keys(
        task.get("provenance"),
        _TASK_PROVENANCE_FIELDS,
        "registered task provenance",
    )
    fixed_provenance = {
        "contract_sha256": CONTRACT_SHA256,
        "primary_action_bank_sha256": PRIMARY_ACTION_BANK_SHA256,
        "signed_orbit_cycle_sha256": SIGNED_ORBIT_CYCLE_SHA256,
        "execution_contract_sha256": EXECUTION_CONTRACT_SHA256,
        "sidecar_shape_sha256": SIDECAR_SHAPE_SHA256,
    }
    for key, expected_value in fixed_provenance.items():
        _require_same_json_value(
            provenance.get(key),
            expected_value,
            f"registered task provenance field {key!r}",
        )
    for key in (
        "prompt_rows_sha256",
        "exclusion_inventory_sha256",
        "prompt_manifest_canonical_sha256",
        "prompt_csv_sha256",
        "prompt_selection_digest",
    ):
        _require_sha256(provenance.get(key), f"registered task provenance {key}")

    task_sha256 = _require_sha256(
        task.get("task_sha256"), "registered task SHA-256"
    )
    unhashed_task = {key: child for key, child in task.items() if key != "task_sha256"}
    if canonical_sha256(unhashed_task) != task_sha256:
        raise ValueError("registered task SHA-256 is inconsistent")
    return task


def _reject_forbidden_state_keys(value: Any, path: str = "sidecar") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold().replace("-", "_")
            if any(fragment in key for fragment in _FORBIDDEN_STATE_KEY_FRAGMENTS):
                raise ValueError(f"{path} contains forbidden later-state/cache key {raw_key!r}")
            _reject_forbidden_state_keys(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_state_keys(child, f"{path}[{index}]")


def _validate_feature_hook(value: Any, step_index: int) -> None:
    hook = _require_exact_keys(value, _HOOK_FIELDS, f"step {step_index} feature hook")
    expected = {
        "module_path": FEATURE_BLOCK,
        "input_name": FEATURE_INPUT_NAME,
        "hook_calls": 1,
        "consume_calls": 1,
        "expected_cfg_shape": list(FEATURE_CFG_SHAPE),
        "captured_conditional_shape": list(FEATURE_CONDITIONAL_SHAPE),
        "cfg_row_order": CFG_ROW_ORDER,
        "detached": True,
    }
    _require_same_json_value(hook, expected, f"step {step_index} feature hook")


def _validate_random_counter(
    value: Any,
    *,
    step_index: int,
    prompt: Mapping[str, Any],
    action: Mapping[str, Any],
) -> str:
    counter = _require_exact_keys(
        value, _COUNTER_FIELDS, f"step {step_index} random counter"
    )
    expected = {
        "schema": RANDOM_EDGE_COUNTER_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "split_role": SPLIT_ROLE,
        "prompt_row_id": prompt["prompt_row_id"],
        "seed": prompt["seed"],
        "step_index": step_index,
        "orbit_name": action["orbit_name"],
        "undirected_edge_count": _ORBIT_EDGE_COUNTS[action["orbit_name"]],
    }
    for key, expected_value in expected.items():
        _require_same_json_value(
            counter.get(key),
            expected_value,
            f"step {step_index} random counter field {key!r}",
        )
    counter_set_sha256 = _require_sha256(
        counter.get("counter_set_sha256"),
        f"step {step_index} random counter set SHA-256",
    )
    expected_counter_set_sha256 = random_edge_counter_set_sha256(
        experiment_id=EXPERIMENT_ID,
        split_role=SPLIT_ROLE,
        prompt_row_id=prompt["prompt_row_id"],
        seed=prompt["seed"],
        step_index=step_index,
        orbit_name=action["orbit_name"],
        grid_size=GRID_SIZE,
    )
    if counter_set_sha256 != expected_counter_set_sha256:
        raise ValueError(
            f"step {step_index} random counter-set SHA-256 differs from "
            "the registered D4 counter graph"
        )
    return counter_set_sha256


def _validate_evidence_scope(
    value: Any, expected_task: Mapping[str, Any]
) -> None:
    scope = _require_exact_keys(value, _EVIDENCE_SCOPE_FIELDS, "evidence scope")
    provenance = expected_task["provenance"]
    expected = {
        "schema": EVIDENCE_SCOPE_SCHEMA,
        "split_role": SPLIT_ROLE,
        "generation_only": True,
        "scoring_authorized": False,
        "quality_outcomes_present": False,
        "prompt_csv_sha256": provenance["prompt_csv_sha256"],
        "prompt_manifest_canonical_sha256": provenance[
            "prompt_manifest_canonical_sha256"
        ],
        "prompt_rows_sha256": provenance["prompt_rows_sha256"],
        "contract_sha256": CONTRACT_SHA256,
        "task_sha256": expected_task["task_sha256"],
    }
    _require_same_json_value(scope, expected, "sidecar evidence scope")


def validate_sidecar(
    record: Mapping[str, Any], expected_task: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one in-memory JSON sidecar against its canonical task."""

    _require_static_contract()
    expected_task = _validate_expected_task(expected_task)
    canonical_json_bytes(record)
    _reject_forbidden_state_keys(record)
    sidecar = _require_exact_keys(record, _TOP_LEVEL_SIDECAR_FIELDS, "sidecar")
    if sidecar["schema"] != SIDECAR_SCHEMA:
        raise ValueError("sidecar schema differs")
    if sidecar["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("sidecar experiment_id differs")
    if sidecar["task_id"] != expected_task.get("task_id"):
        raise ValueError("sidecar task_id differs from the registered task")
    if canonical_json_bytes(sidecar["prompt"]) != canonical_json_bytes(
        expected_task.get("prompt")
    ):
        raise ValueError("sidecar prompt identity differs from the registered task")
    if canonical_json_bytes(sidecar["action"]) != canonical_json_bytes(
        expected_task.get("action")
    ):
        raise ValueError("sidecar action differs from the registered task")
    action = expected_task["action"]
    prompt = expected_task["prompt"]

    trajectory = _require_exact_keys(
        sidecar["trajectory"], _TRAJECTORY_FIELDS, "trajectory"
    )
    trajectory_id = trajectory.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id must be a non-empty string")
    initial_hash = _require_sha256(
        trajectory.get("initial_latent_sha256"), "initial latent SHA-256"
    )
    final_hash = _require_sha256(
        trajectory.get("final_latent_sha256"), "final latent SHA-256"
    )
    png_hash = _require_sha256(trajectory.get("png_sha256"), "PNG SHA-256")
    expected_trajectory = {
        "physical_no_op": action["physical_no_op"],
        "scheduler_class": SCHEDULER_CLASS,
        "scheduler_churn": SCHEDULER_CHURN,
        "scheduler_prediction_type": SCHEDULER_PREDICTION_TYPE,
        "scheduler_config_sha256_v2": SCHEDULER_CONFIG_SHA256_V2,
        "scheduler_schedule_sha256": SCHEDULER_SCHEDULE_SHA256,
        "scheduler_construction_init_noise_sigma": (
            SCHEDULER_CONSTRUCTION_INIT_NOISE_SIGMA
        ),
        "scheduler_effective_init_noise_sigma": (
            SCHEDULER_EFFECTIVE_INIT_NOISE_SIGMA
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "width": WIDTH,
        "height": HEIGHT,
        "cfg_scale": CFG_SCALE,
        "grid_size": GRID_SIZE,
    }
    for key, expected_value in expected_trajectory.items():
        _require_same_json_value(
            trajectory.get(key), expected_value, f"trajectory field {key!r}"
        )

    scheduler_timesteps = _require_number_list(
        trajectory.get("scheduler_timesteps"),
        "trajectory scheduler timesteps",
        expected_length=NUM_INFERENCE_STEPS,
    )
    scheduler_sigmas = _require_number_list(
        trajectory.get("scheduler_sigmas"),
        "trajectory scheduler sigmas",
        expected_length=NUM_INFERENCE_STEPS + 1,
    )
    observed_schedule_sha256 = canonical_sha256(
        {"timesteps": scheduler_timesteps, "sigmas": scheduler_sigmas}
    )
    if observed_schedule_sha256 != trajectory["scheduler_schedule_sha256"]:
        raise ValueError(
            "trajectory scheduler arrays differ from the frozen schedule hash"
        )
    _require_same_json_value(
        scheduler_timesteps,
        list(SCHEDULER_TIMESTEPS),
        "trajectory scheduler timesteps",
    )
    _require_same_json_value(
        scheduler_sigmas,
        list(SCHEDULER_SIGMAS),
        "trajectory scheduler sigmas",
    )

    totals = _require_exact_keys(sidecar["call_totals"], _CALL_TOTAL_FIELDS, "call totals")
    expected_totals = {
        "unet_calls": NUM_INFERENCE_STEPS,
        "scheduler_calls": NUM_INFERENCE_STEPS,
        "extra_unet_calls": 0,
        "extra_scheduler_calls": 0,
        "backbone_backward_calls": 0,
        "intermediate_decode_calls": 0,
        "final_decode_calls": 1,
        "attention_probability_reads": 0,
        "qk_reads": 0,
    }
    _require_same_json_value(
        totals, expected_totals, "sidecar 50x1 frozen-backbone call totals"
    )

    steps = sidecar["step_ledger"]
    if not isinstance(steps, list) or len(steps) != NUM_INFERENCE_STEPS:
        raise ValueError("sidecar must contain exactly 50 step-ledger records")
    basis_hashes = []
    counter_hashes = []
    trajectory_chain = []
    expected_latent_before = initial_hash
    for expected_index, raw_step in enumerate(steps):
        step = _require_exact_keys(raw_step, _STEP_FIELDS, f"step {expected_index}")
        if canonical_json_bytes(step["step_index"]) != canonical_json_bytes(
            expected_index
        ):
            raise ValueError("step indices must be contiguous and ordered")
        for key in (
            "affinity_source",
            "orbit_name",
            "sign",
        ):
            if canonical_json_bytes(step[key]) != canonical_json_bytes(action[key]):
                raise ValueError(f"step {expected_index} action field {key!r} differs")
        expected_step_calls = {
            "unet_calls": 1,
            "scheduler_calls": 1,
            "extra_unet_calls": 0,
            "extra_scheduler_calls": 0,
            "backbone_backward_calls": 0,
            "intermediate_decode_calls": 0,
        }
        for key, expected_value in expected_step_calls.items():
            if canonical_json_bytes(step[key]) != canonical_json_bytes(expected_value):
                raise ValueError(f"step {expected_index} call field {key!r} differs")
        _require_bool(step["finite"], True, f"step {expected_index} finite")
        _require_bool(step["cap_hit"], False, f"step {expected_index} cap_hit")
        latent_before = _require_sha256(
            step["latent_before_sha256"],
            f"step {expected_index} latent-before SHA-256",
        )
        latent_after = _require_sha256(
            step["latent_after_sha256"],
            f"step {expected_index} latent-after SHA-256",
        )
        if latent_before != expected_latent_before:
            raise ValueError(
                f"step {expected_index} latent-before hash breaks the trajectory chain"
            )
        if latent_after == latent_before:
            raise ValueError(
                f"step {expected_index} latent transition is an exact no-op"
            )
        trajectory_chain.append(
            {
                "step_index": expected_index,
                "latent_before_sha256": latent_before,
                "latent_after_sha256": latent_after,
            }
        )
        expected_latent_before = latent_after

        if action["feature_hook_required"]:
            _validate_feature_hook(step["feature_hook"], expected_index)
        elif step["feature_hook"] is not None:
            raise ValueError(f"step {expected_index} must not contain a feature hook")
        if action["random_counter_required"]:
            counter_hashes.append(
                _validate_random_counter(
                    step["random_edge_counter"],
                    step_index=expected_index,
                    prompt=prompt,
                    action=action,
                )
            )
        elif step["random_edge_counter"] is not None:
            raise ValueError(f"step {expected_index} must not contain a random counter")

        applied_ratio = _require_number(
            step["applied_update_ratio"], f"step {expected_index} applied ratio"
        )
        target_error = _require_number(
            step["target_ratio_error"], f"step {expected_index} target ratio error"
        )
        mean_error = _require_number(
            step["channel_mean_abs_error"], f"step {expected_index} mean error"
        )
        variance_error = _require_number(
            step["channel_variance_relative_error"],
            f"step {expected_index} variance error",
        )
        covariance_drift = _require_number(
            step["channel_covariance_drift"],
            f"step {expected_index} covariance drift",
        )
        pred_x0_l2 = _require_number(
            step["pred_original_sample_relative_l2_error"],
            f"step {expected_index} pred_original_sample round-trip error",
        )
        prev_l2 = _require_number(
            step["expected_prev_sample_relative_l2_error"],
            f"step {expected_index} expected prev_sample round-trip error",
        )
        native_max_abs = _require_number(
            step["native_round_trip_max_abs_error"],
            f"step {expected_index} native round-trip max-abs diagnostic",
        )
        solver_target = _require_number(
            step["solver_target_update_ratio"],
            f"step {expected_index} solver target update ratio",
        )
        solver_evaluations = _require_int(
            step["mapped_ratio_solver_evaluations"],
            f"step {expected_index} mapped-ratio solver evaluations",
            minimum=0,
        )

        if action["physical_no_op"]:
            if step["basis_sha256"] is not None:
                raise ValueError("physical P0 must not contain a basis hash")
            if any(
                value != 0.0
                for value in (
                    applied_ratio,
                    target_error,
                    mean_error,
                    variance_error,
                    covariance_drift,
                    pred_x0_l2,
                    prev_l2,
                    native_max_abs,
                    solver_target,
                    float(solver_evaluations),
                )
            ):
                raise ValueError("physical P0 diagnostics must be exact zero")
        else:
            basis_hashes.append(
                _require_sha256(
                    step["basis_sha256"], f"step {expected_index} basis SHA-256"
                )
            )
            expected_target_error = abs(applied_ratio - ACTIVE_TARGET_UPDATE_RATIO)
            if not math.isclose(target_error, expected_target_error, abs_tol=1e-12):
                raise ValueError(f"step {expected_index} target ratio error is inconsistent")
            if target_error > TARGET_RATIO_TOLERANCE:
                raise ValueError(f"step {expected_index} missed the target ratio")
            if applied_ratio >= HARD_UPDATE_CAP:
                raise ValueError(f"step {expected_index} reached the hard update cap")
            if mean_error > CHANNEL_MEAN_ABS_ERROR_MAX:
                raise ValueError(f"step {expected_index} exceeds the channel mean bound")
            if variance_error > CHANNEL_VARIANCE_RELATIVE_ERROR_MAX:
                raise ValueError(f"step {expected_index} exceeds the variance bound")
            if covariance_drift > CHANNEL_COVARIANCE_DRIFT_MAX:
                raise ValueError(f"step {expected_index} exceeds the covariance bound")
            if not 0.0 < solver_target < HARD_UPDATE_CAP:
                raise ValueError(
                    f"step {expected_index} has an invalid solver target ratio"
                )
            if solver_evaluations not in {1, 14}:
                raise ValueError(
                    f"step {expected_index} has an invalid mapped-ratio solver count"
                )
            if pred_x0_l2 > PRED_ORIGINAL_SAMPLE_RELATIVE_L2_MAX:
                raise ValueError(
                    f"step {expected_index} exceeds pred_original_sample round-trip bound"
                )
            if prev_l2 > EXPECTED_PREV_SAMPLE_RELATIVE_L2_MAX:
                raise ValueError(
                    f"step {expected_index} exceeds expected prev_sample round-trip bound"
                )

    if action["physical_no_op"]:
        if basis_hashes or counter_hashes:
            raise ValueError("physical P0 cannot contain active mechanism hashes")
    else:
        if len(set(basis_hashes)) != NUM_INFERENCE_STEPS:
            raise ValueError("active trajectory basis hashes must be unique at all 50 steps")
        if action["random_counter_required"] and len(set(counter_hashes)) != (
            NUM_INFERENCE_STEPS
        ):
            raise ValueError("random-edge counter-set hashes must be unique at all 50 steps")
    after_hashes = [row["latent_after_sha256"] for row in trajectory_chain]
    if len(set(after_hashes)) != NUM_INFERENCE_STEPS:
        raise ValueError("trajectory latent-after hashes must be unique at all 50 steps")
    if expected_latent_before != final_hash:
        raise ValueError("trajectory final latent hash differs from the last step")
    _validate_evidence_scope(sidecar["evidence_scope"], expected_task)
    return {
        "task_id": expected_task["task_id"],
        "action_id": action["action_id"],
        "prompt_row_id": prompt["prompt_row_id"],
        "physical_no_op": action["physical_no_op"],
        "trajectory_id": trajectory_id,
        "initial_latent_sha256": initial_hash,
        "final_latent_sha256": final_hash,
        "png_sha256": png_hash,
        "trajectory_chain_sha256": canonical_sha256(trajectory_chain),
        "sidecar_sha256": canonical_sha256(sidecar),
    }


def validate_prompt_block(
    records: Iterable[Mapping[str, Any]], expected_tasks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate one complete 15-trajectory prompt block in memory."""

    _require_static_contract()
    records = list(records)
    expected_tasks = list(expected_tasks)
    if len(records) != TASKS_PER_PROMPT or len(expected_tasks) != TASKS_PER_PROMPT:
        raise ValueError("one engineering prompt block must contain exactly 15 records")
    expected_tasks = [_validate_expected_task(task) for task in expected_tasks]
    prompt_row_ids = {task["prompt"]["prompt_row_id"] for task in expected_tasks}
    if len(prompt_row_ids) != 1:
        raise ValueError("expected tasks do not belong to one prompt block")
    if len({canonical_json_bytes(task["prompt"]) for task in expected_tasks}) != 1:
        raise ValueError("expected prompt-block tasks do not share one prompt identity")
    if len(
        {canonical_json_bytes(task["provenance"]) for task in expected_tasks}
    ) != 1:
        raise ValueError("expected prompt-block tasks do not share prompt provenance")
    tasks_by_id = {task["task_id"]: task for task in expected_tasks}
    if len(tasks_by_id) != TASKS_PER_PROMPT:
        raise ValueError("expected prompt block contains duplicate task ids")
    records_by_id = {}
    for record in records:
        task_id = record.get("task_id") if isinstance(record, Mapping) else None
        if not isinstance(task_id, str) or task_id in records_by_id:
            raise ValueError("prompt block contains an empty or duplicate task id")
        records_by_id[task_id] = record
    if set(records_by_id) != set(tasks_by_id):
        raise ValueError("prompt block task ids differ from the exact 15-task design")

    summaries = [
        validate_sidecar(records_by_id[task_id], tasks_by_id[task_id])
        for task_id in sorted(tasks_by_id)
    ]
    initial_hashes = {row["initial_latent_sha256"] for row in summaries}
    if len(initial_hashes) != 1:
        raise ValueError("prompt block trajectories do not share one initial latent hash")
    trajectory_ids = [row["trajectory_id"] for row in summaries]
    if len(set(trajectory_ids)) != TASKS_PER_PROMPT:
        raise ValueError("prompt block trajectory ids are not unique")
    no_ops = [row for row in summaries if row["physical_no_op"]]
    active = [row for row in summaries if not row["physical_no_op"]]
    if len(no_ops) != 1 or len(active) != TASKS_PER_PROMPT - 1:
        raise ValueError("prompt block must contain one physical P0 and 14 active trajectories")
    active_final_hashes = [row["final_latent_sha256"] for row in active]
    if len(set(active_final_hashes)) != len(active_final_hashes):
        raise ValueError("active prompt-block final latent hashes are not unique")
    if no_ops[0]["final_latent_sha256"] in set(active_final_hashes):
        raise ValueError("an active trajectory final latent equals physical P0")
    active_png_hashes = [row["png_sha256"] for row in active]
    if len(set(active_png_hashes)) != len(active_png_hashes):
        raise ValueError("active prompt-block PNG hashes are not unique")
    if no_ops[0]["png_sha256"] in set(active_png_hashes):
        raise ValueError("an active trajectory PNG equals physical P0")
    trajectory_chain_hashes = [row["trajectory_chain_sha256"] for row in summaries]
    if len(set(trajectory_chain_hashes)) != TASKS_PER_PROMPT:
        raise ValueError("prompt-block trajectory hash chains are not unique")
    action_counts = Counter(row["action_id"] for row in summaries)
    if set(action_counts.values()) != {1}:
        raise ValueError("prompt block action ids are not one-to-one")

    antithetic_step0_bases: dict[tuple[str, str], dict[int, str]] = {}
    random_counter_sets: dict[str, dict[int, tuple[str, ...]]] = {}
    for task_id, task in tasks_by_id.items():
        action = task["action"]
        if action["family"] in {"P", "R"} and not action["physical_no_op"]:
            pair_key = (action["family"], action["orbit_name"])
            signed_bases = antithetic_step0_bases.setdefault(pair_key, {})
            if action["sign"] in signed_bases:
                raise ValueError("prompt block contains duplicate signed basis controls")
            signed_bases[action["sign"]] = records_by_id[task_id]["step_ledger"][0][
                "basis_sha256"
            ]
        if action["family"] != "R":
            continue
        counter_hashes = tuple(
            step["random_edge_counter"]["counter_set_sha256"]
            for step in records_by_id[task_id]["step_ledger"]
        )
        orbit_pairs = random_counter_sets.setdefault(action["orbit_name"], {})
        if action["sign"] in orbit_pairs:
            raise ValueError("prompt block contains duplicate signed random controls")
        orbit_pairs[action["sign"]] = counter_hashes
    expected_antithetic_keys = {
        (family, orbit_name)
        for family in ("P", "R")
        for orbit_name in _ORBIT_OFFSETS
    }
    if set(antithetic_step0_bases) != expected_antithetic_keys:
        raise ValueError("prompt block does not contain all six antithetic basis pairs")
    antithetic_step0_basis_pairs = []
    for family, orbit_name in sorted(expected_antithetic_keys):
        signed_bases = antithetic_step0_bases[(family, orbit_name)]
        if set(signed_bases) != {-1, 1}:
            raise ValueError("antithetic basis orbit does not contain one +/- pair")
        if signed_bases[-1] != signed_bases[1]:
            raise ValueError(
                "antithetic +/- actions must share the same step-0 basis"
            )
        antithetic_step0_basis_pairs.append(
            {
                "family": family,
                "orbit_name": orbit_name,
                "basis_sha256": signed_bases[1],
            }
        )

    if set(random_counter_sets) != set(_ORBIT_OFFSETS):
        raise ValueError("prompt block does not contain all three random-edge orbits")
    random_counter_pairs = []
    for orbit_name in sorted(_ORBIT_OFFSETS):
        signed_sets = random_counter_sets[orbit_name]
        if set(signed_sets) != {-1, 1}:
            raise ValueError("random-edge orbit does not contain one +/- pair")
        if signed_sets[-1] != signed_sets[1]:
            raise ValueError(
                "random-edge +/- actions must reuse the same counter sets at every step"
            )
        random_counter_pairs.append(
            {
                "orbit_name": orbit_name,
                "counter_set_sha256_by_step": list(signed_sets[1]),
            }
        )
    evidence = {
        "schema": BLOCK_EVIDENCE_SCHEMA,
        "prompt_row_id": next(iter(prompt_row_ids)),
        "record_count": TASKS_PER_PROMPT,
        "initial_latent_sha256": next(iter(initial_hashes)),
        "task_ids_sha256": canonical_sha256(sorted(tasks_by_id)),
        "trajectory_ids_sha256": canonical_sha256(sorted(trajectory_ids)),
        "active_final_latents_sha256": canonical_sha256(sorted(active_final_hashes)),
        "active_pngs_sha256": canonical_sha256(sorted(active_png_hashes)),
        "trajectory_chains_sha256": canonical_sha256(
            sorted(trajectory_chain_hashes)
        ),
        "antithetic_step0_basis_pairs_sha256": canonical_sha256(
            antithetic_step0_basis_pairs
        ),
        "random_counter_pairs_sha256": canonical_sha256(random_counter_pairs),
        "sidecars_sha256": canonical_sha256(
            sorted(row["sidecar_sha256"] for row in summaries)
        ),
        "contract_sha256": CONTRACT_SHA256,
    }
    evidence["block_evidence_sha256"] = canonical_sha256(evidence)
    return evidence


__all__ = [
    "ACTION_FIELDS",
    "ACTIVE_TARGET_UPDATE_RATIO",
    "BLOCK_EVIDENCE_SCHEMA",
    "CFG_SCALE",
    "CHANNEL_COVARIANCE_DRIFT_MAX",
    "CHANNEL_MEAN_ABS_ERROR_MAX",
    "CHANNEL_VARIANCE_RELATIVE_ERROR_MAX",
    "CONTRACT_SHA256",
    "DESIGN_SCHEMA",
    "EVIDENCE_SCOPE_SCHEMA",
    "EXCLUSION_INVENTORY_PATH",
    "EXCLUSION_INVENTORY_SCHEMA",
    "EXPECTED_PREV_SAMPLE_RELATIVE_L2_MAX",
    "EXECUTION_CONTRACT",
    "EXPERIMENT_ID",
    "GRID_SIZE",
    "HARD_UPDATE_CAP",
    "HEIGHT",
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_SNAPSHOT_MANIFEST_SHA256",
    "NUM_INFERENCE_STEPS",
    "ORBIT_EDGE_COUNTS",
    "ORBIT_OFFSETS",
    "PRED_ORIGINAL_SAMPLE_RELATIVE_L2_MAX",
    "PRIMARY_ACTION_BANK_SHA256",
    "PROMPT_COUNT",
    "RANDOM_EDGE_COUNTER_SCHEMA",
    "SCHEDULER_CLASS",
    "SCHEDULER_CONFIG_SHA256_V2",
    "SCHEDULER_CONSTRUCTION_INIT_NOISE_SIGMA",
    "SCHEDULER_EFFECTIVE_INIT_NOISE_SIGMA",
    "SCHEDULER_PREDICTION_TYPE",
    "SCHEDULER_SCHEDULE_SHA256",
    "SCHEDULER_SIGMAS",
    "SCHEDULER_TIMESTEPS",
    "SIDECAR_SCHEMA",
    "SIGNED_ORBIT_CYCLE",
    "SIGNED_ORBIT_CYCLE_SHA256",
    "SPLIT_ROLE",
    "TARGET_RATIO_TOLERANCE",
    "TASKS_PER_PROMPT",
    "TASK_SCHEMA",
    "TOTAL_TASK_COUNT",
    "WIDTH",
    "build_engineering_design",
    "canonical_hashes",
    "canonical_json_bytes",
    "canonical_primary_action_bank",
    "canonical_selected_controls",
    "canonical_sha256",
    "normalize_prompt",
    "stderr_warning_line_count",
    "validate_engineering_design",
    "validate_prompt_assets",
    "validate_prompt_block",
    "validate_sidecar",
]
