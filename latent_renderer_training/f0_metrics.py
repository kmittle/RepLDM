"""Versioned, recomputable evidence for the F0 feasibility gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence


F0_PHASE_EVIDENCE_SCHEMA = "repldm.renderer_f0_phase_evidence.v2"
F0_METRIC_ROW_SCHEMA = "repldm.renderer_f0_metric_row.v2"
F0_METRICS_SUMMARY_SCHEMA = "repldm.renderer_f0_metrics_summary.v1"
F0_POWER_SCHEMA = "repldm.renderer_f0_power.v1"
F0_SCREEN_REGISTRATION_SCHEMA = "repldm.renderer_f0_screen_registration.v2"
F0_PIXEL_SUMMARY_SCHEMA = "repldm.renderer_f0_pixel_summary.v1"

BOOTSTRAP_SEED = 20260901
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_METHOD = "prompt_cluster_percentile_type7_v1"
DECISION_INDICES = (8, 24, 40)
TRAIN_GENERATION_SEEDS = (2026090101, 2026090102)
VALIDATION_GENERATION_SEEDS = (2026090191, 2026090192)
F0_QUERY_BUDGET = {
    "unet_forward": 52_800,
    "scheduler_step": 54_528,
    "vae_decode": 2_496,
    "reward_forward": 4_416,
    "reward_backward": 576,
    "optimizer_step": 1_000,
}

_EXPECTED_COUNTS = {"train": (64, 384), "validation": (32, 192)}
_PHASE_SEEDS = {
    "train": TRAIN_GENERATION_SEEDS,
    "validation": VALIDATION_GENERATION_SEEDS,
}
_VIOLATION_FIELDS = (
    "no_op_parity",
    "scheduler_parity",
    "finite_value",
    "moment",
    "hard_cap",
)
_POWER_EFFECTS = {
    "direction_accuracy": 0.55,
    "g_target": 0.10,
    "topiq_nr_delta": 0.005,
    "realization_ratio": 0.25,
}
_PIXEL_BRANCHES = (
    ("target", "plus"),
    ("minus", "minus"),
    ("realized", "realized"),
)
_REALIZATION_FIELDS = {
    "train": ("out_of_fold_student", "out_of_fold"),
    "validation": ("final_teacher", "final_teacher"),
}
_PROVENANCE_BRANCHES = ("anchor", "plus", "minus", "realized")
_SCORE_RECEIPTS = ("image_reward", "topiq_nr")
_PIXEL_GUARDS = (
    "clipped_fraction",
    "mean_saturation",
    "contrast_log_ratio",
)
_BOOTSTRAP_KEYS = (
    "direction_accuracy",
    "g_target",
    "g_realized",
    "topiq_nr_delta",
    "topiq_nr_minus_delta",
    "topiq_nr_realized_delta",
) + tuple(
    f"{branch}_{guard}"
    for branch, _raw_prefix in _PIXEL_BRANCHES
    for guard in _PIXEL_GUARDS
)
_PROTOCOL_QUERY_MAXIMA = {
    "train": {
        "unet_forwards": 35_200,
        "vae_decodes": 1_664,
        "reward_forwards": 2_944,
        "reward_backwards": 384,
        "image_reward_forwards": 1_664,
        "topiq_nr_forwards": 1_280,
    },
    "validation": {
        "unet_forwards": 17_600,
        "vae_decodes": 832,
        "reward_forwards": 1_472,
        "reward_backwards": 192,
        "image_reward_forwards": 832,
        "topiq_nr_forwards": 640,
    },
}
_F0_SCREEN_DESIGN = {
    "train": {
        "prompt_count": 64,
        "state_count": 384,
        "generation_seeds": list(TRAIN_GENERATION_SEEDS),
    },
    "validation": {
        "prompt_count": 32,
        "state_count": 192,
        "generation_seeds": list(VALIDATION_GENERATION_SEEDS),
    },
    "seeds_per_prompt": 2,
    "decision_indices": list(DECISION_INDICES),
    "aggregation_unit": "prompt",
    "within_prompt_average": ["generation_seed", "decision_index"],
    "bootstrap": {
        "method": BOOTSTRAP_METHOD,
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "interval_probabilities": [0.025, 0.975],
        "interval_role": "descriptive_stability_filter",
    },
}
_F0_SCREEN_THRESHOLDS = {
    "safety": {
        "maximum_total_violations": 0,
        "violation_fields": list(_VIOLATION_FIELDS),
    },
    "coverage": {"minimum_by_decision": 0.80},
    "direction_accuracy": {
        "minimum_mean": 0.55,
        "interval_lower_strictly_above": 0.50,
    },
    "g_target": {
        "minimum_mean": 0.10,
        "interval_lower_strictly_above": 0.0,
    },
    "topiq_nr": {
        "comparison": {"candidate": "plus", "reference": "anchor"},
        "minimum_mean_delta": 0.005,
        "interval_lower_strictly_above": 0.0,
    },
    "clipped_fraction": {
        "branches": [branch for branch, _raw_prefix in _PIXEL_BRANCHES],
        "maximum_interval_upper_delta": 0.001,
    },
    "mean_saturation": {
        "branches": [branch for branch, _raw_prefix in _PIXEL_BRANCHES],
        "maximum_interval_upper_delta": 0.005,
    },
    "contrast": {
        "branches": [branch for branch, _raw_prefix in _PIXEL_BRANCHES],
        "minimum_interval_lower_ratio": 0.95,
        "maximum_interval_upper_ratio": 1.05,
    },
    "target_cap": {"fraction_strictly_below": 0.01},
    "g_realized": {
        "interval_lower_strictly_above": 0.0,
        "minimum_mean_ratio_to_g_target": 0.25,
    },
}
_F0_PIXEL_SUMMARY_DEFINITION = {
    "schema": F0_PIXEL_SUMMARY_SCHEMA,
    "input": {
        "layout": "rgb_nchw",
        "batch_size": 1,
        "source": "finite_floating_tensor",
        "working_device": "cpu",
        "working_dtype": "float32",
    },
    "quantization": {
        "formula": "uint8(round_ties_to_even(clamp(x,0,1)*255))",
    },
    "clipped_fraction": {
        "formula": "mean(any_channel(rgb_uint8<=lo or rgb_uint8>=hi))",
        "lo": 2,
        "hi": 253,
    },
    "mean_saturation": {
        "color_model": "HSV",
        "formula": "mean(where(max>epsilon,(max-min)/(max+epsilon),0))",
        "rgb_scale": "rgb_uint8/255",
        "epsilon": 1e-6,
    },
    "contrast": {
        "formula": "population_std(0.299*R+0.587*G+0.114*B)",
        "rgb_scale": "uint8_0_255",
        "luma": "BT.601",
        "correction": 0,
    },
}
_F0_BENCHMARK_OBLIGATION = {
    "checkpoint": "T_OPSD",
    "hpsv2": {"complete": True, "image_count": 3_200},
    "geneval": {
        "complete": True,
        "official": True,
        "prompt_count": 553,
        "images_per_prompt": 4,
        "image_count": 2_212,
    },
    "required_before_performance_claim": True,
}
_F0_LIMITATIONS = {
    "no_power_claim": True,
    "no_significance_claim": True,
    "no_benchmark_substitution": True,
    "not_opd_dpo_rl_result": True,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
        raise ValueError("F0 evidence must contain finite JSON values") from exc


def _json_copy(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    try:
        result = json.loads(_canonical(dict(value)).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a JSON object")
    return result


def f0_screen_design() -> dict[str, Any]:
    """Return the fixed design to embed in a run contract without an artifact hash."""
    return _json_copy(_F0_SCREEN_DESIGN, label="F0 screen design")


def compute_f0_pixel_summary(images: Any) -> dict[str, float]:
    """Compute the frozen F0 pixel guards for one RGB NCHW image tensor."""
    import torch

    if (
        not isinstance(images, torch.Tensor)
        or images.ndim != 4
        or tuple(images.shape[:2]) != (1, 3)
        or images.shape[2] <= 0
        or images.shape[3] <= 0
        or not images.is_floating_point()
    ):
        raise ValueError("F0 pixel summary requires one floating RGB NCHW image")
    value = images.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("F0 pixel summary image contains non-finite values")
    rgb = torch.round(value.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    rgb_hwc = rgb[0].permute(1, 2, 0)
    clipped = torch.any((rgb_hwc <= 2) | (rgb_hwc >= 253), dim=-1)
    normalized = rgb_hwc.to(torch.float32) / 255.0
    maximum = normalized.amax(dim=-1)
    minimum = normalized.amin(dim=-1)
    saturation = torch.where(
        maximum > 1e-6,
        (maximum - minimum) / (maximum + 1e-6),
        torch.zeros_like(maximum),
    )
    rgb_float = rgb_hwc.to(torch.float32)
    luma = (
        0.299 * rgb_float[..., 0]
        + 0.587 * rgb_float[..., 1]
        + 0.114 * rgb_float[..., 2]
    )
    result = {
        "clipped_fraction": float(clipped.to(torch.float32).mean().item()),
        "mean_saturation": float(saturation.mean().item()),
        "contrast": float(torch.std(luma, unbiased=False).item()),
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise RuntimeError("F0 pixel summary produced a non-finite value")
    return result


def _hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def build_f0_metric_action(
    *,
    branch: str,
    target_record_sha256: str | None,
    renderer_action_history: Any,
) -> dict[str, Any]:
    """Build the single ledger action shape shared by F0 scoring and gates."""
    if branch not in {
        "anchor",
        "plus",
        "minus",
        "out_of_fold_student",
        "final_teacher",
    }:
        raise ValueError("F0 metric branch is not registered")
    if branch == "anchor":
        if target_record_sha256 is not None:
            raise ValueError("F0 anchor metric cannot bind a target record")
        reuse_scope = "shared_prompt_seed"
    else:
        _hash(target_record_sha256, label="F0 metric target record")
        reuse_scope = "forbidden"
    try:
        history = json.loads(_canonical(renderer_action_history).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("F0 renderer action history is not canonical JSON") from exc
    if not isinstance(history, list) or len(history) != 3:
        raise ValueError("F0 renderer action history must contain three decisions")
    flattened: list[float] = []
    for decision in history:
        if (
            not isinstance(decision, list)
            or len(decision) != 1
            or not isinstance(decision[0], list)
            or len(decision[0]) != 6
        ):
            raise ValueError("F0 renderer action history has an invalid shape")
        for coefficient in decision[0]:
            flattened.append(
                _finite(coefficient, label="F0 renderer action coefficient")
            )
    if branch == "anchor" and any(value != 0.0 for value in flattened):
        raise ValueError("F0 anchor metric is not a strict zero action")
    return {
        "f0_metric_branch": branch,
        "reuse_scope": reuse_scope,
        "target_record_sha256": target_record_sha256,
        "pixel_summary_schema": F0_PIXEL_SUMMARY_SCHEMA,
        "renderer_action_history": history,
    }


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _fraction(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _file_binding(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _read_bound_bytes(value: Any, *, label: str) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
        raise ValueError(f"{label} file binding is incomplete")
    path_value = value.get("path")
    size = value.get("bytes")
    if not isinstance(path_value, str):
        raise ValueError(f"{label} path must be absolute")
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute ordinary file")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label} byte count must be positive")
    expected_hash = _hash(value.get("sha256"), label=label)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"{label} bytes differ from the binding")
    return payload, {"path": str(path), "sha256": expected_hash, "bytes": size}


def _write_once(path: Path, payload: bytes) -> dict[str, Any]:
    if not path.is_absolute():
        path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace F0 evidence: {path}")
    else:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return _file_binding(path, payload)


def _phase_shape(phase: str) -> tuple[int, int, tuple[int, ...]]:
    if phase not in _EXPECTED_COUNTS:
        raise ValueError("F0 phase must be train or validation")
    prompts, states = _EXPECTED_COUNTS[phase]
    return prompts, states, _PHASE_SEEDS[phase]


def _validate_row(value: Any, *, phase: str, index: int) -> dict[str, Any]:
    row = _json_copy(value, label=f"F0 {phase} row {index}")
    required = {
        "schema",
        "phase",
        "prompt_id",
        "generation_seed",
        "decision_index",
        "provenance",
        "valid_nonzero_gradient",
        "violations",
        "reward_select",
        "topiq_nr",
        "pixel",
        "target_cap",
    }
    if set(row) != required or row.get("schema") != F0_METRIC_ROW_SCHEMA:
        raise ValueError(f"F0 {phase} row {index} fields differ from the schema")
    if row.get("phase") != phase:
        raise ValueError(f"F0 {phase} row {index} names a different phase")
    prompt_id = row.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError(f"F0 {phase} row {index} has an invalid prompt ID")
    seed = row.get("generation_seed")
    if type(seed) is not int or seed not in _PHASE_SEEDS[phase]:
        raise ValueError(f"F0 {phase} row {index} has an invalid generation seed")
    decision = row.get("decision_index")
    if type(decision) is not int or decision not in DECISION_INDICES:
        raise ValueError(f"F0 {phase} row {index} has an invalid decision index")

    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "target_record_sha256", "pixel_summary_schema", "held_out_fold", "branches"
    }:
        raise ValueError(f"F0 {phase} row {index} provenance is incomplete")
    _hash(
        provenance.get("target_record_sha256"),
        label=f"F0 {phase} row {index} target record",
    )
    if provenance.get("pixel_summary_schema") != F0_PIXEL_SUMMARY_SCHEMA:
        raise ValueError(f"F0 {phase} row {index} pixel summary schema is invalid")
    held_out_fold = provenance.get("held_out_fold")
    if phase == "train":
        if type(held_out_fold) is not int or held_out_fold not in range(4):
            raise ValueError(f"F0 {phase} row {index} held-out fold is invalid")
    elif held_out_fold is not None:
        raise ValueError(f"F0 {phase} row {index} cannot name a held-out fold")
    branches = provenance.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != set(_PROVENANCE_BRANCHES):
        raise ValueError(f"F0 {phase} row {index} provenance branches are incomplete")
    for branch in _PROVENANCE_BRANCHES:
        branch_value = branches.get(branch)
        if not isinstance(branch_value, Mapping) or set(branch_value) != {
            "reuse_scope", "renderer_state_sha256", "image_sha256", *_SCORE_RECEIPTS
        }:
            raise ValueError(
                f"F0 {phase} row {index} {branch} provenance is incomplete"
            )
        expected_reuse = "shared_prompt_seed" if branch == "anchor" else "forbidden"
        if branch_value.get("reuse_scope") != expected_reuse:
            raise ValueError(
                f"F0 {phase} row {index} {branch} receipt reuse differs from the protocol"
            )
        _hash(
            branch_value.get("renderer_state_sha256"),
            label=f"F0 {phase} row {index} {branch} renderer state",
        )
        _hash(
            branch_value.get("image_sha256"),
            label=f"F0 {phase} row {index} {branch} image",
        )
        for scorer in _SCORE_RECEIPTS:
            receipt = branch_value.get(scorer)
            if not isinstance(receipt, Mapping) or set(receipt) != {
                "reservation_id", "output_sha256"
            }:
                raise ValueError(
                    f"F0 {phase} row {index} {branch} {scorer} receipt is incomplete"
                )
            reservation_id = receipt.get("reservation_id")
            if not isinstance(reservation_id, str) or not reservation_id:
                raise ValueError(
                    f"F0 {phase} row {index} {branch} {scorer} reservation is invalid"
                )
            _hash(
                receipt.get("output_sha256"),
                label=f"F0 {phase} row {index} {branch} {scorer} output",
            )
    _boolean(
        row.get("valid_nonzero_gradient"),
        label=f"F0 {phase} row {index} gradient validity",
    )

    violations = row.get("violations")
    if not isinstance(violations, Mapping) or set(violations) != set(_VIOLATION_FIELDS):
        raise ValueError(f"F0 {phase} row {index} violations are incomplete")
    for field in _VIOLATION_FIELDS:
        _boolean(
            violations.get(field),
            label=f"F0 {phase} row {index} violation {field}",
        )

    realization_field, realization_pixel_prefix = _REALIZATION_FIELDS[phase]
    reward = row.get("reward_select")
    if not isinstance(reward, Mapping) or set(reward) != {
        "anchor", "plus", "minus", realization_field
    }:
        raise ValueError(f"F0 {phase} row {index} rewards are incomplete")
    for field in ("anchor", "plus", "minus", realization_field):
        _finite(reward.get(field), label=f"F0 {phase} row {index} reward {field}")

    witness = row.get("topiq_nr")
    if not isinstance(witness, Mapping) or set(witness) != {
        "anchor", "plus", "minus", realization_field
    }:
        raise ValueError(f"F0 {phase} row {index} TOPIQ values are incomplete")
    for field in ("anchor", "plus", "minus", realization_field):
        _finite(witness.get(field), label=f"F0 {phase} row {index} TOPIQ {field}")

    pixel = row.get("pixel")
    pixel_prefixes = ("anchor", "plus", "minus", realization_pixel_prefix)
    fraction_fields = {
        f"{prefix}_{metric}"
        for prefix in pixel_prefixes
        for metric in ("clipped_fraction", "mean_saturation")
    }
    contrast_fields = {f"{prefix}_contrast" for prefix in pixel_prefixes}
    pixel_fields = fraction_fields | contrast_fields
    if not isinstance(pixel, Mapping) or set(pixel) != pixel_fields:
        raise ValueError(f"F0 {phase} row {index} pixel guards are incomplete")
    for field in sorted(fraction_fields):
        _fraction(pixel.get(field), label=f"F0 {phase} row {index} pixel {field}")
    for field in sorted(contrast_fields):
        if _finite(pixel.get(field), label=f"F0 {phase} row {index} pixel {field}") <= 0:
            raise ValueError(f"F0 {phase} row {index} contrast must be positive")

    cap = row.get("target_cap")
    if not isinstance(cap, Mapping) or set(cap) != {"plus_at_98pct", "minus_at_98pct"}:
        raise ValueError(f"F0 {phase} row {index} target cap flags are incomplete")
    for field in ("plus_at_98pct", "minus_at_98pct"):
        _boolean(cap.get(field), label=f"F0 {phase} row {index} cap {field}")
    return row


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["prompt_id"]),
        int(row["generation_seed"]),
        int(row["decision_index"]),
    )


def _row_values(row: Mapping[str, Any], reward_scale: float) -> dict[str, float]:
    reward = row["reward_select"]
    witness = row["topiq_nr"]
    pixel = row["pixel"]
    plus = float(reward["plus"])
    minus = float(reward["minus"])
    if plus > minus + 1e-6:
        direction = 1.0
    elif plus < minus - 1e-6:
        direction = 0.0
    else:
        direction = 0.5
    anchor = float(reward["anchor"])
    witness_comparison = _F0_SCREEN_THRESHOLDS["topiq_nr"]["comparison"]
    witness_reference = float(witness[witness_comparison["reference"]])
    realization_field, realization_pixel_prefix = _REALIZATION_FIELDS[str(row["phase"])]
    values = {
        "direction_accuracy": direction,
        "g_target": (plus - anchor) / reward_scale,
        "g_realized": (float(reward[realization_field]) - anchor) / reward_scale,
        "topiq_nr_delta": float(witness[witness_comparison["candidate"]])
        - witness_reference,
        "topiq_nr_minus_delta": float(witness["minus"]) - witness_reference,
        "topiq_nr_realized_delta": float(witness[realization_field])
        - witness_reference,
    }
    for branch, raw_prefix in _PIXEL_BRANCHES:
        if branch == "realized":
            raw_prefix = realization_pixel_prefix
        values[f"{branch}_clipped_fraction"] = (
            float(pixel[f"{raw_prefix}_clipped_fraction"])
            - float(pixel["anchor_clipped_fraction"])
        )
        values[f"{branch}_mean_saturation"] = (
            float(pixel[f"{raw_prefix}_mean_saturation"])
            - float(pixel["anchor_mean_saturation"])
        )
        values[f"{branch}_contrast_log_ratio"] = math.log(
            float(pixel[f"{raw_prefix}_contrast"])
        ) - math.log(float(pixel["anchor_contrast"]))
    for key, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"F0 derived metric {key} must be finite")
    return values


def _validate_shared_anchors(rows: Sequence[Mapping[str, Any]], *, phase: str) -> None:
    anchors: dict[tuple[str, int], bytes] = {}
    for row in rows:
        key = (str(row["prompt_id"]), int(row["generation_seed"]))
        anchor = {
            "reward_select": row["reward_select"]["anchor"],
            "topiq_nr": row["topiq_nr"]["anchor"],
            "pixel": {
                field: value
                for field, value in row["pixel"].items()
                if field.startswith("anchor_")
            },
            "provenance": row["provenance"]["branches"]["anchor"],
        }
        encoded = _canonical(anchor)
        if key in anchors and anchors[key] != encoded:
            raise ValueError(
                f"F0 {phase} anchor differs across decisions for one prompt-seed"
            )
        anchors[key] = encoded


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average empty F0 evidence")
    try:
        result = math.fsum(values) / len(values)
    except OverflowError as exc:
        raise ValueError("F0 metric mean must be finite") from exc
    if not math.isfinite(result):
        raise ValueError("F0 metric mean must be finite")
    return result


def _percentile_type7(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of empty F0 evidence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    result = float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
    if not math.isfinite(result):
        raise ValueError("F0 percentile must be finite")
    return result


def _bootstrap(
    prompt_values: Sequence[Mapping[str, float]],
) -> dict[str, tuple[float, float]]:
    if not prompt_values:
        raise ValueError("F0 evidence contains no prompt clusters")
    columns = {
        key: tuple(float(row[key]) for row in prompt_values) for key in _BOOTSTRAP_KEYS
    }
    # Constant fixtures and genuinely constant experiments have a degenerate,
    # exact bootstrap distribution; avoid needlessly materializing 10,000 copies.
    if all(len(set(values)) == 1 for values in columns.values()):
        return {key: (values[0], values[0]) for key, values in columns.items()}
    rng = random.Random(BOOTSTRAP_SEED)
    samples = {key: [] for key in _BOOTSTRAP_KEYS}
    count = len(prompt_values)
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = [rng.randrange(count) for _ in range(count)]
        for key, values in columns.items():
            samples[key].append(_mean([values[index] for index in indices]))
    return {
        key: (
            _percentile_type7(values, 0.025),
            _percentile_type7(values, 0.975),
        )
        for key, values in samples.items()
    }


def compute_f0_phase_summary(
    rows: Iterable[Mapping[str, Any]], *, phase: str, reward_scale: float
) -> dict[str, Any]:
    """Recompute every F0 phase gate from raw logical-state rows."""
    expected_prompts, expected_states, seeds = _phase_shape(phase)
    scale = _finite(reward_scale, label="F0 reward scale")
    if scale < 1e-6:
        raise ValueError("F0 reward scale is below the frozen floor")
    validated = [_validate_row(row, phase=phase, index=index) for index, row in enumerate(rows)]
    if len(validated) != expected_states:
        raise ValueError(f"F0 {phase} evidence must contain exactly {expected_states} rows")
    ordered = sorted(validated, key=_row_key)
    if validated != ordered:
        raise ValueError(f"F0 {phase} rows must use canonical prompt/seed/decision order")
    keys = [_row_key(row) for row in validated]
    if len(set(keys)) != len(keys):
        raise ValueError(f"F0 {phase} logical-state rows must be unique")
    prompt_ids = sorted({str(row["prompt_id"]) for row in validated})
    if len(prompt_ids) != expected_prompts:
        raise ValueError(f"F0 {phase} evidence must contain exactly {expected_prompts} prompts")
    expected_keys = {
        (prompt_id, seed, decision)
        for prompt_id in prompt_ids
        for seed in seeds
        for decision in DECISION_INDICES
    }
    if set(keys) != expected_keys:
        raise ValueError(f"F0 {phase} evidence does not cover every prompt/seed/decision")
    _validate_shared_anchors(validated, phase=phase)

    prompt_metrics: list[dict[str, float]] = []
    for prompt_id in prompt_ids:
        values = [
            _row_values(row, scale)
            for row in validated
            if row["prompt_id"] == prompt_id
        ]
        prompt_metrics.append(
            {key: _mean([value[key] for value in values]) for key in _BOOTSTRAP_KEYS}
        )
    intervals = _bootstrap(prompt_metrics)
    means = {
        key: _mean([value[key] for value in prompt_metrics]) for key in _BOOTSTRAP_KEYS
    }
    coverage = {
        str(decision): _mean(
            [
                1.0 if row["valid_nonzero_gradient"] else 0.0
                for row in validated
                if row["decision_index"] == decision
            ]
        )
        for decision in DECISION_INDICES
    }
    safety_violations = sum(
        int(bool(row["violations"][field]))
        for row in validated
        for field in _VIOLATION_FIELDS
    )
    cap_fraction = _mean(
        [
            1.0 if row["target_cap"][field] else 0.0
            for row in validated
            for field in ("plus_at_98pct", "minus_at_98pct")
        ]
    )
    pixel_summaries: dict[str, dict[str, Any]] = {}
    for branch, _raw_prefix in _PIXEL_BRANCHES:
        contrast_key = f"{branch}_contrast_log_ratio"
        try:
            contrast_mean = math.exp(means[contrast_key])
            contrast_interval = [
                math.exp(intervals[contrast_key][0]),
                math.exp(intervals[contrast_key][1]),
            ]
        except OverflowError as exc:
            raise ValueError("F0 contrast geometric ratio must be finite") from exc
        pixel_summaries[branch] = {
            "clipped_fraction_delta": {
                "mean": means[f"{branch}_clipped_fraction"],
                "bootstrap_interval_95": list(
                    intervals[f"{branch}_clipped_fraction"]
                ),
            },
            "mean_saturation_delta": {
                "mean": means[f"{branch}_mean_saturation"],
                "bootstrap_interval_95": list(
                    intervals[f"{branch}_mean_saturation"]
                ),
            },
            "contrast_geometric_ratio": {
                "mean": contrast_mean,
                "bootstrap_interval_95": contrast_interval,
            },
        }
    target_mean = means["g_target"]
    realization_ratio = means["g_realized"] / target_mean if target_mean > 0 else 0.0
    if not math.isfinite(realization_ratio):
        raise ValueError("F0 realization ratio must be finite")
    thresholds = _F0_SCREEN_THRESHOLDS
    checks = {
        "safety": safety_violations
        <= thresholds["safety"]["maximum_total_violations"],
        "coverage": all(
            value >= thresholds["coverage"]["minimum_by_decision"]
            for value in coverage.values()
        ),
        "direction_accuracy": (
            means["direction_accuracy"]
            >= thresholds["direction_accuracy"]["minimum_mean"]
            and intervals["direction_accuracy"][0]
            > thresholds["direction_accuracy"]["interval_lower_strictly_above"]
        ),
        "g_target": (
            target_mean >= thresholds["g_target"]["minimum_mean"]
            and intervals["g_target"][0]
            > thresholds["g_target"]["interval_lower_strictly_above"]
        ),
        "topiq_nr": (
            means["topiq_nr_delta"]
            >= thresholds["topiq_nr"]["minimum_mean_delta"]
            and intervals["topiq_nr_delta"][0]
            > thresholds["topiq_nr"]["interval_lower_strictly_above"]
        ),
        "clipped_fraction": all(
            intervals[f"{branch}_clipped_fraction"][1]
            <= thresholds["clipped_fraction"]["maximum_interval_upper_delta"]
            for branch, _raw_prefix in _PIXEL_BRANCHES
        ),
        "mean_saturation": all(
            intervals[f"{branch}_mean_saturation"][1]
            <= thresholds["mean_saturation"]["maximum_interval_upper_delta"]
            for branch, _raw_prefix in _PIXEL_BRANCHES
        ),
        "contrast": all(
            pixel_summaries[branch]["contrast_geometric_ratio"][
                "bootstrap_interval_95"
            ][0]
            >= thresholds["contrast"]["minimum_interval_lower_ratio"]
            and pixel_summaries[branch]["contrast_geometric_ratio"][
                "bootstrap_interval_95"
            ][1]
            <= thresholds["contrast"]["maximum_interval_upper_ratio"]
            for branch, _raw_prefix in _PIXEL_BRANCHES
        ),
        "target_cap": cap_fraction
        < thresholds["target_cap"]["fraction_strictly_below"],
        "g_realized": (
            intervals["g_realized"][0]
            > thresholds["g_realized"]["interval_lower_strictly_above"]
            and realization_ratio
            >= thresholds["g_realized"]["minimum_mean_ratio_to_g_target"]
        ),
    }

    def metric(key: str) -> dict[str, Any]:
        return {
            "mean": means[key],
            "bootstrap_interval_95": list(intervals[key]),
        }

    return {
        "schema": F0_METRICS_SUMMARY_SCHEMA,
        "record_type": "summary",
        "phase": phase,
        "prompt_count": len(prompt_ids),
        "state_count": len(validated),
        "safety_violations": safety_violations,
        "coverage_by_decision": coverage,
        "direction_accuracy": metric("direction_accuracy"),
        "g_target": metric("g_target"),
        "g_realized": metric("g_realized"),
        "realization_ratio": realization_ratio,
        "topiq_nr_delta": metric("topiq_nr_delta"),
        "topiq_nr_minus_delta": metric("topiq_nr_minus_delta"),
        "topiq_nr_realized_delta": metric("topiq_nr_realized_delta"),
        "pixel_guards": pixel_summaries,
        "target_cap_fraction": cap_fraction,
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "prompt",
            "interval_probabilities": [0.025, 0.975],
            "interval_role": "descriptive_stability_filter",
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_f0_phase_evidence(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    phase: str,
    reward_scale: float,
    reward_statistics_sha256: str,
    f0_run_contract_sha256: str,
    screen_registration_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write canonical header/row/summary JSONL and return its binding and summary."""
    _phase_shape(phase)
    row_values = [_validate_row(row, phase=phase, index=index) for index, row in enumerate(rows)]
    summary = compute_f0_phase_summary(row_values, phase=phase, reward_scale=reward_scale)
    header = {
        "schema": F0_PHASE_EVIDENCE_SCHEMA,
        "record_type": "header",
        "phase": phase,
        "reward_scale": _finite(reward_scale, label="F0 reward scale"),
        "reward_statistics_sha256": _hash(
            reward_statistics_sha256, label="F0 reward statistics"
        ),
        "f0_run_contract_sha256": _hash(
            f0_run_contract_sha256, label="F0 run contract"
        ),
        "screen_registration_sha256": _hash(
            screen_registration_sha256, label="F0 screen registration"
        ),
        "row_count": len(row_values),
        "protocol_query_maxima": dict(_PROTOCOL_QUERY_MAXIMA[phase]),
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "prompt",
            "interval_probabilities": [0.025, 0.975],
            "interval_role": "descriptive_stability_filter",
        },
    }
    payload = b"\n".join(
        _canonical(item) for item in (header, *row_values, summary)
    ) + b"\n"
    binding = _write_once(Path(path), payload)
    return binding, summary


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} in F0 evidence")
        result[key] = value
    return result


def _parse_canonical_lines(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be newline-terminated canonical JSONL")
    lines = payload[:-1].split(b"\n")
    if not lines or any(not line for line in lines):
        raise ValueError(f"{label} contains an empty JSONL row")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(
                line.decode("ascii"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{label} row {index} is invalid JSON") from exc
        if not isinstance(value, dict) or _canonical(value) != line:
            raise ValueError(f"{label} row {index} is not canonical JSON")
        records.append(value)
    return records


def load_f0_phase_evidence(
    binding: Mapping[str, Any],
    *,
    phase: str,
    reward_scale: float,
    reward_statistics_sha256: str,
    f0_run_contract_sha256: str,
    screen_registration_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]]:
    """Read validated phase evidence and return its summary, binding, and raw rows."""
    expected_prompts, expected_states, _seeds = _phase_shape(phase)
    payload, normalized_binding = _read_bound_bytes(binding, label=f"F0 {phase} metrics")
    records = _parse_canonical_lines(payload, label=f"F0 {phase} metrics")
    if len(records) != expected_states + 2:
        raise ValueError(f"F0 {phase} metrics have the wrong JSONL row count")
    header = records[0]
    expected_header = {
        "schema": F0_PHASE_EVIDENCE_SCHEMA,
        "record_type": "header",
        "phase": phase,
        "reward_scale": _finite(reward_scale, label="F0 reward scale"),
        "reward_statistics_sha256": _hash(
            reward_statistics_sha256, label="F0 reward statistics"
        ),
        "f0_run_contract_sha256": _hash(
            f0_run_contract_sha256, label="F0 run contract"
        ),
        "screen_registration_sha256": _hash(
            screen_registration_sha256, label="F0 screen registration"
        ),
        "row_count": expected_states,
        "protocol_query_maxima": dict(_PROTOCOL_QUERY_MAXIMA[phase]),
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "prompt",
            "interval_probabilities": [0.025, 0.975],
            "interval_role": "descriptive_stability_filter",
        },
    }
    if _canonical(header) != _canonical(expected_header):
        raise ValueError(f"F0 {phase} metrics header differs from the frozen contract")
    rows = records[1:-1]
    computed = compute_f0_phase_summary(rows, phase=phase, reward_scale=reward_scale)
    if computed["prompt_count"] != expected_prompts:
        raise ValueError(f"F0 {phase} prompt count differs from the protocol")
    if _canonical(records[-1]) != _canonical(computed):
        raise ValueError(f"F0 {phase} stored summary differs from raw rows")
    return computed, normalized_binding, tuple(rows)


def validate_f0_phase_evidence(
    binding: Mapping[str, Any],
    *,
    phase: str,
    reward_scale: float,
    reward_statistics_sha256: str,
    f0_run_contract_sha256: str,
    screen_registration_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read bound JSONL, recompute its summary, and reject any claimed mismatch."""
    summary, normalized_binding, _rows = load_f0_phase_evidence(
        binding,
        phase=phase,
        reward_scale=reward_scale,
        reward_statistics_sha256=reward_statistics_sha256,
        f0_run_contract_sha256=f0_run_contract_sha256,
        screen_registration_sha256=screen_registration_sha256,
    )
    return summary, normalized_binding


def build_f0_screen_registration(
    *, f0_run_contract_sha256: str
) -> dict[str, Any]:
    """Build the fixed pre-output registration for the engineering F0 screen."""
    artifact = {
        "schema": F0_SCREEN_REGISTRATION_SCHEMA,
        "status": "frozen_before_output",
        "evidence_tier": "engineering_screen",
        "inferential_claim_allowed": False,
        "f0_run_contract_sha256": _hash(
            f0_run_contract_sha256, label="F0 run contract"
        ),
        "design": f0_screen_design(),
        "thresholds": _json_copy(
            _F0_SCREEN_THRESHOLDS, label="F0 screen thresholds"
        ),
        "pixel_summary": _json_copy(
            _F0_PIXEL_SUMMARY_DEFINITION, label="F0 pixel summary definition"
        ),
        "benchmark_obligation": _json_copy(
            _F0_BENCHMARK_OBLIGATION, label="F0 benchmark obligation"
        ),
        "limitations": _json_copy(_F0_LIMITATIONS, label="F0 limitations"),
    }
    return _validate_screen_registration_payload(
        artifact, f0_run_contract_sha256=f0_run_contract_sha256
    )


def _validate_screen_registration_payload(
    value: Any, *, f0_run_contract_sha256: str
) -> dict[str, Any]:
    artifact = _json_copy(value, label="F0 screen registration")
    required = {
        "schema",
        "status",
        "evidence_tier",
        "inferential_claim_allowed",
        "f0_run_contract_sha256",
        "design",
        "thresholds",
        "pixel_summary",
        "benchmark_obligation",
        "limitations",
    }
    if set(artifact) != required or artifact.get("schema") != F0_SCREEN_REGISTRATION_SCHEMA:
        raise ValueError(
            "F0 screen registration fields differ from the registered schema"
        )
    if artifact.get("status") != "frozen_before_output":
        raise ValueError("F0 screen registration was not frozen before output")
    if artifact.get("evidence_tier") != "engineering_screen":
        raise ValueError("F0 registration must identify an engineering screen")
    if artifact.get("inferential_claim_allowed") is not False:
        raise ValueError("F0 screen registration must forbid inferential claims")
    expected_hash = _hash(f0_run_contract_sha256, label="F0 run contract")
    if artifact.get("f0_run_contract_sha256") != expected_hash:
        raise ValueError("F0 screen registration names a different run contract")
    frozen_fields = {
        "design": _F0_SCREEN_DESIGN,
        "thresholds": _F0_SCREEN_THRESHOLDS,
        "pixel_summary": _F0_PIXEL_SUMMARY_DEFINITION,
        "benchmark_obligation": _F0_BENCHMARK_OBLIGATION,
        "limitations": _F0_LIMITATIONS,
    }
    for field, expected in frozen_fields.items():
        if _canonical(artifact.get(field)) != _canonical(expected):
            raise ValueError(f"F0 screen registration {field} differs from the protocol")
    return artifact


def write_f0_screen_registration_evidence(
    path: str | Path, artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and immutably write a canonical F0 screen registration."""
    contract_hash = (
        artifact.get("f0_run_contract_sha256")
        if isinstance(artifact, Mapping)
        else None
    )
    normalized = _validate_screen_registration_payload(
        artifact,
        f0_run_contract_sha256=_hash(contract_hash, label="F0 run contract"),
    )
    return _write_once(Path(path), _canonical(normalized) + b"\n")


def validate_f0_screen_registration_evidence(
    binding: Mapping[str, Any], *, f0_run_contract_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and fully validate the bound engineering-screen registration."""
    payload, normalized_binding = _read_bound_bytes(
        binding, label="F0 screen registration"
    )
    records = _parse_canonical_lines(payload, label="F0 screen registration")
    if len(records) != 1:
        raise ValueError("F0 screen registration must contain one canonical JSON object")
    artifact = _validate_screen_registration_payload(
        records[0], f0_run_contract_sha256=f0_run_contract_sha256
    )
    return artifact, normalized_binding


def build_f0_power_artifact(
    *,
    f0_run_contract_sha256: str,
    phase_power: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject the retired v1 self-reported power-plan format."""
    artifact = {
        "schema": F0_POWER_SCHEMA,
        "status": "frozen_before_output",
        "f0_run_contract_sha256": _hash(
            f0_run_contract_sha256, label="F0 run contract"
        ),
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "prompt",
        },
        "minimum_effects": dict(_POWER_EFFECTS),
        "phases": _json_copy(phase_power, label="F0 phase power"),
    }
    _validate_power_payload(artifact, f0_run_contract_sha256=f0_run_contract_sha256)
    return artifact


def _validate_power_payload(
    value: Any, *, f0_run_contract_sha256: str
) -> dict[str, Any]:
    artifact = _json_copy(value, label="F0 power artifact")
    if set(artifact) != {
        "schema", "status", "f0_run_contract_sha256", "bootstrap",
        "minimum_effects", "phases",
    } or artifact.get("schema") != F0_POWER_SCHEMA:
        raise ValueError("F0 power artifact fields differ from the registered schema")
    if artifact.get("status") != "frozen_before_output":
        raise ValueError("F0 power artifact was not frozen before output")
    expected_hash = _hash(f0_run_contract_sha256, label="F0 run contract")
    if artifact.get("f0_run_contract_sha256") != expected_hash:
        raise ValueError("F0 power artifact names a different run contract")
    expected_bootstrap = {
        "method": BOOTSTRAP_METHOD,
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "unit": "prompt",
    }
    if _canonical(artifact.get("bootstrap")) != _canonical(expected_bootstrap):
        raise ValueError("F0 power artifact bootstrap differs from the protocol")
    if _canonical(artifact.get("minimum_effects")) != _canonical(_POWER_EFFECTS):
        raise ValueError("F0 power artifact minimum effects differ from the protocol")
    phases = artifact.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != set(_EXPECTED_COUNTS):
        raise ValueError("F0 power artifact must cover train and validation")
    for phase, (prompt_count, _state_count) in _EXPECTED_COUNTS.items():
        phase_value = phases.get(phase)
        if not isinstance(phase_value, Mapping) or set(phase_value) != {
            "prompt_count", "metrics"
        }:
            raise ValueError(f"F0 {phase} power fields are incomplete")
        if (
            type(phase_value.get("prompt_count")) is not int
            or phase_value.get("prompt_count") != prompt_count
        ):
            raise ValueError(f"F0 {phase} power prompt count differs from the protocol")
        metrics = phase_value.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(_POWER_EFFECTS):
            raise ValueError(f"F0 {phase} power metrics are incomplete")
        for metric, values in metrics.items():
            if not isinstance(values, Mapping) or set(values) != {
                "required_power", "planned_power"
            }:
                raise ValueError(f"F0 {phase} power metric {metric} is incomplete")
            required = _finite(
                values.get("required_power"),
                label=f"F0 {phase} required power for {metric}",
            )
            planned = _finite(
                values.get("planned_power"),
                label=f"F0 {phase} planned power for {metric}",
            )
            if not (0.0 < required <= 1.0 and 0.0 <= planned <= 1.0):
                raise ValueError(f"F0 {phase} power for {metric} is outside (0, 1]")
            if planned < required:
                raise ValueError(f"F0 {phase} is underpowered for {metric}")
    raise ValueError(
        "F0 power schema v1 is a non-recomputable self-report and cannot "
        "authorize formal F0"
    )


def write_f0_power_evidence(
    path: str | Path, artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject writes of the retired v1 self-reported power artifact."""
    contract_hash = (
        artifact.get("f0_run_contract_sha256")
        if isinstance(artifact, Mapping)
        else None
    )
    normalized = _validate_power_payload(
        artifact,
        f0_run_contract_sha256=_hash(contract_hash, label="F0 run contract"),
    )
    payload = _canonical(normalized) + b"\n"
    return _write_once(Path(path), payload)


def validate_f0_power_evidence(
    binding: Mapping[str, Any], *, f0_run_contract_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the bound artifact and fail closed on non-recomputable v1 plans."""
    payload, normalized_binding = _read_bound_bytes(binding, label="F0 power artifact")
    records = _parse_canonical_lines(payload, label="F0 power artifact")
    if len(records) != 1:
        raise ValueError("F0 power artifact must contain one canonical JSON object")
    artifact = _validate_power_payload(
        records[0], f0_run_contract_sha256=f0_run_contract_sha256
    )
    return artifact, normalized_binding


__all__ = [
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "DECISION_INDICES",
    "F0_METRIC_ROW_SCHEMA",
    "F0_METRICS_SUMMARY_SCHEMA",
    "F0_PHASE_EVIDENCE_SCHEMA",
    "F0_PIXEL_SUMMARY_SCHEMA",
    "F0_POWER_SCHEMA",
    "F0_QUERY_BUDGET",
    "F0_SCREEN_REGISTRATION_SCHEMA",
    "TRAIN_GENERATION_SEEDS",
    "VALIDATION_GENERATION_SEEDS",
    "build_f0_power_artifact",
    "build_f0_metric_action",
    "build_f0_screen_registration",
    "compute_f0_pixel_summary",
    "compute_f0_phase_summary",
    "f0_screen_design",
    "load_f0_phase_evidence",
    "validate_f0_phase_evidence",
    "validate_f0_power_evidence",
    "validate_f0_screen_registration_evidence",
    "write_f0_phase_evidence",
    "write_f0_power_evidence",
    "write_f0_screen_registration_evidence",
]
