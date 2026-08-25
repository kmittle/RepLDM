"""Evaluate the frozen scheduler-native development headroom gate.

This evaluator is deliberately not a method selector. It applies the analysis
registered in the executable actions YAML as a quantitative screen and reports
whether any primary action requires independent review. The point threshold is
not a confidence bound on a practical effect, and the output never authorizes
validation, distillation, or RL. Mechanism ablations are descriptive and cannot
affect the route decision.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from compare_actions import (
    holm_adjust,
    merge_manifest_scores,
    prompt_sign_flip_pvalue,
    validate_expected_grid,
    validate_pairing,
)


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SCHEMA = "scheduler_native_fixed_headroom_actions_v1"
EVALUATION_SCHEMA = "scheduler_native_fixed_headroom_development_evaluation_v1"
AUTHORIZATION_SCHEMA = "scheduler_native_fixed_headroom_evaluation_authorization_v1"
EVALUATOR_VERSION = "native_fixed_headroom_evaluator_v1"
OUTPUT_JSON = "scheduler_native_fixed_headroom_evaluation.json"
OUTPUT_CSV = "scheduler_native_fixed_headroom_evaluation.csv"
AUDITOR_PATH = ROOT / "eval-pipeline" / "audit_latent_renderer_run.py"
METRICS = (
    "topiq_nr",
    "hpsv2",
    "clip_cosine",
    "clipped_fraction",
    "mean_saturation",
    "contrast_std",
)
EXPECTED_ANALYSIS_FIELDS = {
    "baseline_action",
    "identity_control",
    "primary_metric",
    "minimum_practical_mean_delta",
    "primary_holm_family",
    "mechanism_ablation_family",
    "inference",
    "noninferiority_guards",
    "pass_rule",
    "null_rule",
    "ablation_rule",
}
EXPECTED_PASS_RULE = (
    "lazy_zero_identity_must_be_byte_identical_to_no_op",
    "topiq_mean_delta_meets_practical_bound",
    "topiq_crossed_ci_lower_is_positive",
    "topiq_prompt_sign_flip_passes_primary_holm",
    "all_noninferiority_guards_pass",
    "every_step_and_score_record_passes_strict_audit",
)
EXPECTED_NULL_RULE = "close_fixed_renderer_distillation_and_rl_without_validation"
EXPECTED_ABLATION_RULE = (
    "semantic_or_freeu_success_cannot_authorize_a_non_attention_claim"
)
EXPECTED_INTERPRETATION = {
    "estimand": "registered_33_prompt_3_seed_grid_screen",
    "practical_threshold_inference": False,
    "topiq_multiplicity_scope": "four_action_zero_null_only",
    "guard_familywise_control_across_actions": False,
    "challenge_inference": "descriptive_only",
    "pass_decision": "independent_review_required",
    "null_scope": "registered_positive_0p02_primary_action_family_only",
    "method_selection": False,
}


@dataclass(frozen=True)
class Protocol:
    baseline: str
    identity_control: str
    action_order: tuple[str, ...]
    primary_actions: tuple[str, ...]
    ablation_actions: tuple[str, ...]
    minimum_topiq_delta: float
    holm_alpha: float
    ablation_holm_alpha: float
    bootstrap: int
    confidence_level: float
    randomizations: int
    seed: int
    hpsv2_ci_lower_min: float
    clip_ci_lower_min: float
    clipped_ci_upper_max: float
    saturation_ci_upper_max: float
    contrast_ratio_min: float
    contrast_ratio_max: float


@dataclass(frozen=True)
class VerifiedInputs:
    protocol: Protocol
    frame: pd.DataFrame
    actions_config: Mapping[str, Any]
    audit: Mapping[str, Any]
    prompt_indices: tuple[int, ...]
    seeds: tuple[int, ...]
    hashes: Mapping[str, str]
    paths: Mapping[str, str]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str, fields: set[str] | None = None) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if fields is not None and set(value) != fields:
        raise ValueError(f"{label} fields differ from the frozen v1 contract")
    return value


def _require_list(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_frozen_float(value: Any, expected: float, label: str) -> float:
    observed = _require_float(value, label)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} differs from the frozen v1 value {expected}")
    return observed


def load_protocol(actions_config: Mapping[str, Any]) -> Protocol:
    """Parse and freeze the only analysis contract accepted by this evaluator."""
    if actions_config.get("schema") != NATIVE_SCHEMA:
        raise ValueError(f"native evaluator requires schema {NATIVE_SCHEMA}")
    if actions_config.get("status") != "authorized_development":
        raise ValueError("native development actions are not authorized")
    authorization = _require_mapping(
        actions_config.get("authorization"), "authorization"
    )
    if authorization.get("method_selection") is not False:
        raise ValueError("native development evaluator forbids method selection")
    if actions_config.get("split_role") != "development":
        raise ValueError("native evaluator is development-only")

    analysis = _require_mapping(
        actions_config.get("analysis"), "analysis", EXPECTED_ANALYSIS_FIELDS
    )
    if tuple(_require_list(analysis.get("pass_rule"), "analysis.pass_rule")) != (
        EXPECTED_PASS_RULE
    ):
        raise ValueError("analysis.pass_rule differs from the frozen v1 gate")
    if analysis.get("null_rule") != EXPECTED_NULL_RULE:
        raise ValueError("analysis.null_rule differs from the frozen v1 stop rule")
    if analysis.get("ablation_rule") != EXPECTED_ABLATION_RULE:
        raise ValueError("analysis.ablation_rule differs from the frozen v1 rule")

    actions = _require_list(actions_config.get("actions"), "actions")
    if any(not isinstance(action, Mapping) for action in actions):
        raise ValueError("every action must be a mapping")
    action_order = [str(action.get("id", "")) for action in actions]
    if not all(action_order) or len(action_order) != len(set(action_order)):
        raise ValueError("actions contain empty or duplicate ids")
    by_id = {str(action["id"]): action for action in actions}

    baseline = str(analysis.get("baseline_action", ""))
    identity = str(analysis.get("identity_control", ""))
    if baseline != "no_op" or by_id.get(baseline, {}).get("type") != "none":
        raise ValueError("native development baseline must be no_op")
    if by_id.get(baseline, {}).get("role") != "nominal_scheduler_baseline":
        raise ValueError("no_op must be the nominal scheduler baseline")
    if by_id.get(identity, {}).get("role") != "implementation_identity_control":
        raise ValueError("analysis.identity_control differs from the registered role")

    primary_family = _require_mapping(
        analysis.get("primary_holm_family"),
        "analysis.primary_holm_family",
        {"alpha", "actions"},
    )
    primary_actions = tuple(
        str(value)
        for value in _require_list(
            primary_family.get("actions"), "analysis.primary_holm_family.actions"
        )
    )
    registered_primary = tuple(
        action_id
        for action_id in action_order
        if by_id[action_id].get("role") == "non_attention_primary"
    )
    if len(primary_actions) != 4 or primary_actions != registered_primary:
        raise ValueError("primary Holm family must contain exactly the four primary actions")
    if any(by_id[action_id].get("selection_eligible") is not True for action_id in primary_actions):
        raise ValueError("every primary headroom action must be registered as eligible")

    ablation_family = _require_mapping(
        analysis.get("mechanism_ablation_family"),
        "analysis.mechanism_ablation_family",
        {"confirmatory_selection", "holm_alpha", "actions"},
    )
    if ablation_family.get("confirmatory_selection") is not False:
        raise ValueError("mechanism ablations must remain descriptive")
    ablation_actions = tuple(
        str(value)
        for value in _require_list(
            ablation_family.get("actions"),
            "analysis.mechanism_ablation_family.actions",
        )
    )
    registered_ablations = tuple(
        action_id
        for action_id in action_order
        if by_id[action_id].get("role")
        in {"attention_mechanism_ablation", "decoder_feature_mechanism_ablation"}
    )
    if ablation_actions != registered_ablations or len(ablation_actions) != 2:
        raise ValueError("mechanism ablation family differs from registered action roles")
    if any(by_id[action_id].get("selection_eligible") is not False for action_id in ablation_actions):
        raise ValueError("mechanism ablations cannot be selection-eligible")
    expected_actions = {baseline, identity, *primary_actions, *ablation_actions}
    if set(action_order) != expected_actions:
        raise ValueError("native action grid contains an unclassified action")

    if analysis.get("primary_metric") != "topiq_nr":
        raise ValueError("native development primary metric must be topiq_nr")
    minimum_topiq = _require_frozen_float(
        analysis.get("minimum_practical_mean_delta"),
        0.005,
        "analysis.minimum_practical_mean_delta",
    )
    holm_alpha = _require_frozen_float(
        primary_family.get("alpha"), 0.05, "analysis.primary_holm_family.alpha"
    )
    ablation_holm_alpha = _require_frozen_float(
        ablation_family.get("holm_alpha"),
        0.05,
        "analysis.mechanism_ablation_family.holm_alpha",
    )

    inference = _require_mapping(
        analysis.get("inference"),
        "analysis.inference",
        {
            "crossed_prompt_seed_bootstrap",
            "confidence_level",
            "prompt_level_sign_flips",
            "random_seed",
        },
    )
    bootstrap = _require_int(
        inference.get("crossed_prompt_seed_bootstrap"),
        "analysis.inference.crossed_prompt_seed_bootstrap",
    )
    randomizations = _require_int(
        inference.get("prompt_level_sign_flips"),
        "analysis.inference.prompt_level_sign_flips",
    )
    seed = _require_int(inference.get("random_seed"), "analysis.inference.random_seed")
    confidence = _require_frozen_float(
        inference.get("confidence_level"),
        0.95,
        "analysis.inference.confidence_level",
    )
    if bootstrap != 10000 or randomizations != 100000 or seed != 2026:
        raise ValueError("analysis inference counts/seed differ from frozen v1")

    guards = _require_mapping(
        analysis.get("noninferiority_guards"),
        "analysis.noninferiority_guards",
        {
            "hpsv2_ci_lower_min_delta",
            "clip_cosine_ci_lower_min_delta",
            "clipped_fraction_ci_upper_max_delta",
            "mean_saturation_ci_upper_max_delta",
            "contrast_geometric_ratio_interval",
        },
    )
    contrast_interval = _require_list(
        guards.get("contrast_geometric_ratio_interval"),
        "analysis.noninferiority_guards.contrast_geometric_ratio_interval",
    )
    if len(contrast_interval) != 2:
        raise ValueError("contrast geometric-ratio interval must contain two bounds")

    return Protocol(
        baseline=baseline,
        identity_control=identity,
        action_order=tuple(action_order),
        primary_actions=primary_actions,
        ablation_actions=ablation_actions,
        minimum_topiq_delta=minimum_topiq,
        holm_alpha=holm_alpha,
        ablation_holm_alpha=ablation_holm_alpha,
        bootstrap=bootstrap,
        confidence_level=confidence,
        randomizations=randomizations,
        seed=seed,
        hpsv2_ci_lower_min=_require_frozen_float(
            guards.get("hpsv2_ci_lower_min_delta"),
            -0.005,
            "analysis.noninferiority_guards.hpsv2_ci_lower_min_delta",
        ),
        clip_ci_lower_min=_require_frozen_float(
            guards.get("clip_cosine_ci_lower_min_delta"),
            -0.005,
            "analysis.noninferiority_guards.clip_cosine_ci_lower_min_delta",
        ),
        clipped_ci_upper_max=_require_frozen_float(
            guards.get("clipped_fraction_ci_upper_max_delta"),
            0.001,
            "analysis.noninferiority_guards.clipped_fraction_ci_upper_max_delta",
        ),
        saturation_ci_upper_max=_require_frozen_float(
            guards.get("mean_saturation_ci_upper_max_delta"),
            0.005,
            "analysis.noninferiority_guards.mean_saturation_ci_upper_max_delta",
        ),
        contrast_ratio_min=_require_frozen_float(
            contrast_interval[0],
            0.95,
            "analysis.noninferiority_guards.contrast_geometric_ratio_interval[0]",
        ),
        contrast_ratio_max=_require_frozen_float(
            contrast_interval[1],
            1.05,
            "analysis.noninferiority_guards.contrast_geometric_ratio_interval[1]",
        ),
    )


def _crossed_bootstrap_ci(
    values: pd.Series, *, n_boot: int, confidence_level: float, seed: int
) -> tuple[float, float]:
    """Bootstrap prompt and seed axes independently, preserving paired cells."""
    matrix = values.unstack("seed")
    if matrix.isna().any().any():
        raise ValueError("crossed bootstrap requires a complete prompt x seed matrix")
    array = matrix.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("crossed bootstrap received non-finite values")
    rng = np.random.default_rng(seed)
    prompt_indices = rng.integers(0, array.shape[0], size=(n_boot, array.shape[0]))
    seed_indices = rng.integers(0, array.shape[1], size=(n_boot, array.shape[1]))
    sampled = array[prompt_indices[:, :, None], seed_indices[:, None, :]]
    means = sampled.mean(axis=(1, 2))
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _complete_pivots(
    frame: pd.DataFrame, action_order: Sequence[str]
) -> dict[str, pd.DataFrame]:
    pivots = {}
    for metric in METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"metric {metric!r} contains missing or non-finite values")
        pivot = frame.assign(**{metric: values}).pivot(
            index=["prompt_index", "seed"], columns="action_id", values=metric
        )
        pivot = pivot.reindex(columns=list(action_order))
        if pivot.isna().any().any():
            raise ValueError(f"metric {metric!r} is incomplete on the registered grid")
        pivots[metric] = pivot
    if (pivots["contrast_std"] <= 0.0).any().any():
        raise ValueError("contrast_std must be strictly positive for log-ratio inference")
    return pivots


def _action_statistics(
    action: str,
    baseline: str,
    pivots: Mapping[str, pd.DataFrame],
    protocol: Protocol,
    *,
    action_seed: int,
) -> dict[str, Any]:
    additive_metrics = METRICS[:-1]
    deltas = {
        metric: pivots[metric][action] - pivots[metric][baseline]
        for metric in additive_metrics
    }
    cis = {
        metric: _crossed_bootstrap_ci(
            delta,
            n_boot=protocol.bootstrap,
            confidence_level=protocol.confidence_level,
            seed=action_seed + metric_index,
        )
        for metric_index, (metric, delta) in enumerate(deltas.items())
    }
    log_contrast_ratio = np.log(
        pivots["contrast_std"][action] / pivots["contrast_std"][baseline]
    )
    contrast_ci_log = _crossed_bootstrap_ci(
        log_contrast_ratio,
        n_boot=protocol.bootstrap,
        confidence_level=protocol.confidence_level,
        seed=action_seed + len(deltas),
    )
    row: dict[str, Any] = {"action": action, "baseline": baseline}
    for metric in additive_metrics:
        row[f"{metric}_mean_delta"] = float(deltas[metric].mean())
        row[f"{metric}_ci_low"] = float(cis[metric][0])
        row[f"{metric}_ci_high"] = float(cis[metric][1])
    row.update(
        {
            "contrast_geometric_mean_ratio": float(
                math.exp(float(log_contrast_ratio.mean()))
            ),
            "contrast_ratio_ci_low": float(math.exp(contrast_ci_log[0])),
            "contrast_ratio_ci_high": float(math.exp(contrast_ci_log[1])),
        }
    )
    return row


def _apply_primary_gate(row: dict[str, Any], protocol: Protocol) -> None:
    checks = {
        "topiq_point_estimate_screen_pass": (
            row["topiq_nr_mean_delta"] >= protocol.minimum_topiq_delta
        ),
        "topiq_ci_pass": row["topiq_nr_ci_low"] > 0.0,
        "topiq_holm_pass": row["topiq_p_holm"] < protocol.holm_alpha,
        "hpsv2_noninferiority_pass": (
            row["hpsv2_ci_low"] >= protocol.hpsv2_ci_lower_min
        ),
        "clip_cosine_noninferiority_pass": (
            row["clip_cosine_ci_low"] >= protocol.clip_ci_lower_min
        ),
        "clipped_fraction_guard_pass": (
            row["clipped_fraction_ci_high"] <= protocol.clipped_ci_upper_max
        ),
        "mean_saturation_guard_pass": (
            row["mean_saturation_ci_high"] <= protocol.saturation_ci_upper_max
        ),
        "contrast_guard_pass": (
            row["contrast_ratio_ci_low"] >= protocol.contrast_ratio_min
            and row["contrast_ratio_ci_high"] <= protocol.contrast_ratio_max
        ),
    }
    row.update({key: bool(value) for key, value in checks.items()})
    row["passes_headroom_gate"] = bool(all(checks.values()))


def evaluate_frame(
    frame: pd.DataFrame,
    protocol: Protocol,
    *,
    expected_prompt_indices: Sequence[int],
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Apply the frozen route gate without selecting among passing actions."""
    required = {
        "action_id",
        "prompt_index",
        "seed",
        "device",
        "source_challenge",
        *METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"evaluation frame lacks required columns: {missing}")
    validate_expected_grid(
        frame,
        expected_prompt_indices,
        expected_seeds,
        protocol.action_order,
    )
    validate_pairing(frame)
    pivots = _complete_pivots(frame, protocol.action_order)
    prompt_challenges = frame[["prompt_index", "source_challenge"]].drop_duplicates()
    if prompt_challenges["prompt_index"].duplicated().any():
        raise ValueError("a prompt is assigned to multiple source challenges")
    challenge_counts = prompt_challenges.groupby("source_challenge")[
        "prompt_index"
    ].nunique()
    if len(challenge_counts) != 11 or set(challenge_counts.astype(int)) != {3}:
        raise ValueError("native development requires 11 challenges with 3 prompts each")
    challenge_by_prompt = {
        int(row["prompt_index"]): str(row["source_challenge"])
        for _, row in prompt_challenges.iterrows()
    }

    primary_rows = []
    primary_pvalues = []
    for action_index, action in enumerate(protocol.primary_actions):
        row = _action_statistics(
            action,
            protocol.baseline,
            pivots,
            protocol,
            action_seed=protocol.seed + action_index * 100,
        )
        row.update(
            {
                "family": "primary_headroom",
                "family_index": action_index,
                "inferential_use": "quantitative_screen_only",
            }
        )
        delta = pivots["topiq_nr"][action] - pivots["topiq_nr"][protocol.baseline]
        challenge_labels = pd.Series(
            [
                challenge_by_prompt[int(prompt_index)]
                for prompt_index in delta.index.get_level_values("prompt_index")
            ],
            index=delta.index,
        )
        row["topiq_mean_delta_by_challenge"] = {
            str(challenge): float(value)
            for challenge, value in delta.groupby(challenge_labels).mean().sort_index().items()
        }
        row["challenge_summary_inferential_use"] = "descriptive_only"
        pvalue = prompt_sign_flip_pvalue(
            delta,
            n_random=protocol.randomizations,
            seed=protocol.seed + action_index,
        )
        row["topiq_p_sign_flip"] = float(pvalue)
        primary_pvalues.append(float(pvalue))
        primary_rows.append(row)
    for row, adjusted in zip(primary_rows, holm_adjust(primary_pvalues)):
        row["topiq_p_holm"] = float(adjusted)
        _apply_primary_gate(row, protocol)

    ablation_rows = []
    ablation_pvalues = []
    for action_index, action in enumerate(protocol.ablation_actions):
        row = _action_statistics(
            action,
            protocol.baseline,
            pivots,
            protocol,
            action_seed=protocol.seed + 10000 + action_index * 100,
        )
        row.update(
            {
                "family": "mechanism_ablation",
                "family_index": action_index,
                "inferential_use": "descriptive_only",
                "passes_headroom_gate": None,
            }
        )
        delta = pivots["topiq_nr"][action] - pivots["topiq_nr"][protocol.baseline]
        pvalue = prompt_sign_flip_pvalue(
            delta,
            n_random=protocol.randomizations,
            seed=protocol.seed + 10000 + action_index,
        )
        row["topiq_p_sign_flip"] = float(pvalue)
        ablation_pvalues.append(float(pvalue))
        ablation_rows.append(row)
    for row, adjusted in zip(ablation_rows, holm_adjust(ablation_pvalues)):
        row["topiq_p_holm"] = float(adjusted)
        row["ablation_holm_alpha"] = protocol.ablation_holm_alpha

    quantitative_trigger = any(
        row["passes_headroom_gate"] for row in primary_rows
    )
    decision = "independent_review_required" if quantitative_trigger else "null_route"
    return {
        "schema": EVALUATION_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "decision": decision,
        "screen_only": True,
        "decision_scope": EXPECTED_INTERPRETATION["null_scope"],
        "inference_scope": EXPECTED_INTERPRETATION["estimand"],
        "qualitative_review_required": bool(quantitative_trigger),
        "qualitative_review_authorized": False,
        "qualitative_review_protocol_must_be_frozen_before_image_access": bool(
            quantitative_trigger
        ),
        "method_selection_performed": False,
        "validation_authorized": False,
        "distillation_authorized": False,
        "rl_authorized": False,
        "baseline_action": protocol.baseline,
        "identity_control": protocol.identity_control,
        "primary_holm_family": list(protocol.primary_actions),
        "descriptive_ablation_family": list(protocol.ablation_actions),
        "claims": {
            "topiq_threshold_is_point_estimate_screen": True,
            "effect_at_least_threshold_established": False,
            "population_generalization_established": False,
            "primary_topiq_zero_null_fwer_controlled": True,
            "guard_familywise_controlled_across_actions": False,
            "challenge_results_are_descriptive_only": True,
        },
        "rule": {
            "topiq_point_estimate_screen_threshold": protocol.minimum_topiq_delta,
            "practical_threshold_inference_performed": False,
            "require_positive_topiq_crossed_ci_lower": True,
            "topiq_holm_alpha": protocol.holm_alpha,
            "descriptive_ablation_holm_alpha": protocol.ablation_holm_alpha,
            "hpsv2_ci_lower_min_delta": protocol.hpsv2_ci_lower_min,
            "clip_cosine_ci_lower_min_delta": protocol.clip_ci_lower_min,
            "clipped_fraction_ci_upper_max_delta": protocol.clipped_ci_upper_max,
            "mean_saturation_ci_upper_max_delta": protocol.saturation_ci_upper_max,
            "contrast_geometric_ratio_interval": [
                protocol.contrast_ratio_min,
                protocol.contrast_ratio_max,
            ],
            "bootstrap": protocol.bootstrap,
            "confidence_level": protocol.confidence_level,
            "prompt_level_sign_flips": protocol.randomizations,
            "random_seed": protocol.seed,
        },
        "n_prompts": int(frame["prompt_index"].nunique()),
        "n_seeds": int(frame["seed"].nunique()),
        "n_cells_per_action": int(
            frame[["prompt_index", "seed"]].drop_duplicates().shape[0]
        ),
        "rows": primary_rows + ablation_rows,
    }


def _resolve_registered_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    return path


def _load_json(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    return _require_mapping(value, label)


def _load_jsonl(payload: bytes, label: str) -> list[dict]:
    try:
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSONL") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must contain non-empty JSON object rows")
    return rows


def _require_hash(mapping: Mapping[str, Any], key: str, observed: str, label: str) -> None:
    if mapping.get(key) != observed:
        raise ValueError(f"{label} hash mismatch for {key}")


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
    }


def _validate_evaluation_authorization(
    authorization: Mapping[str, Any],
    *,
    actions_file: Path,
    hashes: Mapping[str, str],
    actions_config: Mapping[str, Any],
) -> None:
    """Require the result-blind, source-controlled screen authorization."""
    if authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("evaluation authorization schema differs from frozen v1")
    if authorization.get("status") != "authorized_quantitative_screen":
        raise ValueError("quantitative screen evaluation is not authorized")
    if authorization.get("scope") != "development_only_quantitative_screen":
        raise ValueError("evaluation authorization has an invalid scope")

    permissions = _require_mapping(
        authorization.get("authorization"),
        "evaluation authorization.permissions",
        {
            "reviewers",
            "result_access_before_freeze",
            "method_selection",
            "validation",
            "distillation",
            "rl",
        },
    )
    reviewers = _require_list(
        permissions.get("reviewers"), "evaluation authorization.reviewers"
    )
    if reviewers != ["plan_progress_audit", "literature_survey"]:
        raise ValueError("evaluation authorization reviewers differ from the freeze")
    for key in (
        "result_access_before_freeze",
        "method_selection",
        "validation",
        "distillation",
        "rl",
    ):
        if permissions.get(key) is not False:
            raise ValueError(f"evaluation authorization must set {key} to false")

    bindings = _require_mapping(
        authorization.get("bindings"),
        "evaluation authorization.bindings",
        {
            "evaluator_path",
            "evaluator_sha256",
            "auditor_path",
            "auditor_sha256",
            "actions_path",
            "actions_sha256",
            "source_template_sha256",
            "prompts_sha256",
            "prompt_manifest_sha256",
            "scorer_provenance_sha256",
            "evaluation_schema",
        },
    )
    bound_paths = {
        "evaluator_path": Path(__file__).resolve(),
        "auditor_path": AUDITOR_PATH.resolve(),
        "actions_path": actions_file,
    }
    for key, expected in bound_paths.items():
        observed = _resolve_registered_path(bindings.get(key), key)
        if observed != expected:
            raise ValueError(f"evaluation authorization {key} differs from runtime")
    bound_hashes = {
        "evaluator_sha256": sha256_file(__file__),
        "auditor_sha256": sha256_file(AUDITOR_PATH),
        "actions_sha256": hashes["actions"],
        "source_template_sha256": hashes["source_template"],
        "prompts_sha256": hashes["prompts"],
        "prompt_manifest_sha256": hashes["prompt_manifest"],
    }
    scoring = _require_mapping(actions_config.get("scoring"), "scoring")
    bound_hashes["scorer_provenance_sha256"] = str(
        scoring.get("registered_scorer_provenance_sha256", "")
    )
    for key, expected in bound_hashes.items():
        if bindings.get(key) != expected:
            raise ValueError(f"evaluation authorization hash mismatch for {key}")
    if bindings.get("evaluation_schema") != EVALUATION_SCHEMA:
        raise ValueError("evaluation authorization binds an unexpected output schema")

    runtime = _require_mapping(
        authorization.get("runtime"),
        "evaluation authorization.runtime",
        {"python", "numpy", "pandas", "pyyaml"},
    )
    if dict(runtime) != _runtime_versions():
        raise ValueError("evaluation runtime versions differ from the authorization")
    interpretation = _require_mapping(
        authorization.get("interpretation"),
        "evaluation authorization.interpretation",
        set(EXPECTED_INTERPRETATION),
    )
    if dict(interpretation) != EXPECTED_INTERPRETATION:
        raise ValueError("evaluation interpretation differs from the screen-only freeze")


def load_verified_inputs(
    run_dir: str | os.PathLike[str],
    actions_path: str | os.PathLike[str],
    audit_path: str | os.PathLike[str],
    evaluation_authorization_path: str | os.PathLike[str],
) -> VerifiedInputs:
    """Bind immutable run snapshots to a passing result-blind run audit."""
    run_root = Path(run_dir).resolve()
    actions_file = Path(actions_path).resolve()
    audit_file = Path(audit_path).resolve()
    evaluation_authorization_file = Path(evaluation_authorization_path).resolve()
    paths = {
        "run_config": run_root / "config.json",
        "manifest": run_root / "manifest.jsonl",
        "scores": run_root / "scores.jsonl",
        "actions": actions_file,
        "audit": audit_file,
        "evaluation_authorization": evaluation_authorization_file,
    }
    for label, path in paths.items():
        if not Path(path).is_file():
            raise ValueError(f"required {label} file is missing: {path}")
    payloads = {label: Path(path).read_bytes() for label, path in paths.items()}
    hashes = {label: sha256_bytes(payload) for label, payload in payloads.items()}

    try:
        actions_config = yaml.safe_load(payloads["actions"]) or {}
    except yaml.YAMLError as exc:
        raise ValueError("actions YAML is invalid") from exc
    actions_config = _require_mapping(actions_config, "actions YAML")
    protocol = load_protocol(actions_config)
    source_manifest = _require_mapping(
        actions_config.get("source_manifest"), "source_manifest"
    )
    prompts_path = _resolve_registered_path(
        source_manifest.get("prompts"), "registered prompts"
    )
    prompt_manifest_path = _resolve_registered_path(
        source_manifest.get("path"), "prompt manifest"
    )
    registration = _require_mapping(
        actions_config.get("registration_source"), "registration_source"
    )
    source_template_path = _resolve_registered_path(
        registration.get("path"), "source template"
    )
    for label, path in (
        ("prompts", prompts_path),
        ("prompt_manifest", prompt_manifest_path),
        ("source_template", source_template_path),
    ):
        payloads[label] = path.read_bytes()
        hashes[label] = sha256_bytes(payloads[label])

    try:
        evaluation_authorization = yaml.safe_load(
            payloads["evaluation_authorization"]
        ) or {}
    except yaml.YAMLError as exc:
        raise ValueError("evaluation authorization YAML is invalid") from exc
    evaluation_authorization = _require_mapping(
        evaluation_authorization, "evaluation authorization"
    )
    _validate_evaluation_authorization(
        evaluation_authorization,
        actions_file=actions_file,
        hashes=hashes,
        actions_config=actions_config,
    )

    _require_hash(source_manifest, "prompts_sha256", hashes["prompts"], "source_manifest")
    _require_hash(source_manifest, "sha256", hashes["prompt_manifest"], "source_manifest")
    _require_hash(registration, "sha256", hashes["source_template"], "registration_source")

    run_config = _load_json(payloads["run_config"], "run config")
    audit = _load_json(payloads["audit"], "run audit")
    if audit.get("passed") is not True or audit.get("split_role") != "development":
        raise ValueError("a passing development run audit is required")
    audit_provenance = _require_mapping(audit.get("provenance"), "audit.provenance")
    for audit_key, payload_key in (
        ("config_sha256", "run_config"),
        ("manifest_sha256", "manifest"),
        ("scores_sha256", "scores"),
        ("prompts_sha256", "prompts"),
        ("source_actions_sha256", "actions"),
        ("source_template_sha256", "source_template"),
    ):
        _require_hash(audit_provenance, audit_key, hashes[payload_key], "run audit")

    if run_config.get("native_renderer_registered") is not True:
        raise ValueError("run config is not a registered native renderer run")
    if run_config.get("action_schema") != NATIVE_SCHEMA:
        raise ValueError("run config action schema differs from native v1")
    if run_config.get("split_role") != "development":
        raise ValueError("run config is not the development split")
    for key in ("actions_sha256", "native_renderer_executable_actions_sha256"):
        _require_hash(run_config, key, hashes["actions"], "run config")
    _require_hash(run_config, "prompts_sha256", hashes["prompts"], "run config")
    _require_hash(
        run_config,
        "native_renderer_source_template_sha256",
        hashes["source_template"],
        "run config",
    )
    if run_config.get("scorer_provenance_binding_required") is not True:
        raise ValueError("run config does not require scorer provenance binding")
    if Path(str(run_config.get("actions_yaml", ""))).resolve() != actions_file:
        raise ValueError("run config actions path differs from evaluator input")
    if Path(str(run_config.get("prompts_csv", ""))).resolve() != prompts_path:
        raise ValueError("run config prompts path differs from executable registration")

    split_seeds = _require_mapping(actions_config.get("split_seeds"), "split_seeds")
    seeds = tuple(int(value) for value in _require_list(split_seeds.get("development"), "development seeds"))
    if tuple(int(value) for value in run_config.get("seeds", [])) != seeds:
        raise ValueError("run config seeds differ from executable development seeds")
    if tuple(str(value) for value in audit.get("actions", [])) != protocol.action_order:
        raise ValueError("run audit action order differs from executable YAML")
    if tuple(int(value) for value in audit.get("seeds", [])) != seeds:
        raise ValueError("run audit seeds differ from executable YAML")

    scoring = _require_mapping(actions_config.get("scoring"), "scoring")
    scorer_hash = str(scoring.get("registered_scorer_provenance_sha256", ""))
    if audit.get("scorer_provenance_sha256") != scorer_hash:
        raise ValueError("run audit scorer provenance differs from executable YAML")
    if audit.get("registered_identity_pair") != [
        protocol.baseline,
        protocol.identity_control,
    ] or audit.get("identity_pair_png_hashes_equal") is not True:
        raise ValueError("run audit did not establish exact zero-identity parity")
    audited_score_keys = audit.get("required_score_keys")
    if not isinstance(audited_score_keys, list) or not set(METRICS).issubset(
        set(audited_score_keys)
    ):
        raise ValueError("run audit did not verify every evaluator score key")

    manifest = _load_jsonl(payloads["manifest"], "manifest.jsonl")
    scores = _load_jsonl(payloads["scores"], "scores.jsonl")
    if audit.get("records") != len(manifest) or len(scores) != len(manifest):
        raise ValueError("audited run record counts differ from immutable inputs")
    try:
        prompts = pd.read_csv(io.BytesIO(payloads["prompts"]))
    except Exception as exc:
        raise ValueError("registered prompts CSV is invalid") from exc
    if not {"index", "TEXT", "source_challenge", "split"}.issubset(prompts.columns):
        raise ValueError(
            "registered prompts require index, TEXT, source_challenge, and split columns"
        )
    if prompts["index"].duplicated().any() or set(prompts["split"].astype(str)) != {
        "development"
    }:
        raise ValueError("registered prompt CSV is not a unique development split")
    expected_prompt_count = int(source_manifest.get("expected_prompt_count", -1))
    if len(prompts) != expected_prompt_count or audit.get("prompts") != len(prompts):
        raise ValueError("prompt count differs from executable registration or run audit")
    prompt_indices = tuple(int(value) for value in prompts["index"])
    challenge_counts = prompts.groupby("source_challenge")["index"].nunique()
    if len(challenge_counts) != 11 or set(challenge_counts.astype(int)) != {3}:
        raise ValueError("registered prompts must contain 11 challenges with 3 prompts each")
    challenge_by_prompt = {
        int(row["index"]): str(row["source_challenge"])
        for _, row in prompts.iterrows()
    }

    manifest_frame = pd.DataFrame(manifest)
    scores_frame = pd.DataFrame(scores)
    frame = merge_manifest_scores(manifest_frame, scores_frame)
    frame["source_challenge"] = frame["prompt_index"].map(challenge_by_prompt)
    if frame["source_challenge"].isna().any():
        raise ValueError("manifest contains a prompt absent from the challenge registration")
    validate_expected_grid(
        frame, prompt_indices, seeds, protocol.action_order
    )
    validate_pairing(frame)
    resolved_paths = {
        key: str(Path(path).resolve()) for key, path in paths.items()
    }
    resolved_paths.update(
        {
            "prompts": str(prompts_path),
            "prompt_manifest": str(prompt_manifest_path),
            "source_template": str(source_template_path),
        }
    )
    return VerifiedInputs(
        protocol=protocol,
        frame=frame,
        actions_config=actions_config,
        audit=audit,
        prompt_indices=prompt_indices,
        seeds=seeds,
        hashes=hashes,
        paths=resolved_paths,
    )


@contextmanager
def evaluation_lock(run_dir: str | os.PathLike[str]):
    """Exclude generation, scoring, and concurrent evaluation."""
    run_root = Path(run_dir).resolve()
    run_handle = open(run_root / ".generate.lock", "a+")
    evaluator_handle = open(run_root / ".native_headroom_evaluation.lock", "a+")
    try:
        try:
            fcntl.flock(run_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(evaluator_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"generation, scoring, or native evaluation is active for {run_root}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(evaluator_handle.fileno(), fcntl.LOCK_UN)
            fcntl.flock(run_handle.fileno(), fcntl.LOCK_UN)
        finally:
            evaluator_handle.close()
            run_handle.close()


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _require_committed_inputs(paths: Sequence[str | os.PathLike[str]]) -> str:
    """Require every analysis-contract file to equal its blob at ``HEAD``."""
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    for value in paths:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"analysis input is outside the repository: {path}") from exc
        try:
            committed = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"analysis input is not committed at HEAD: {relative}") from exc
        if committed != path.read_bytes():
            raise RuntimeError(f"analysis input differs from HEAD: {relative}")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--evaluation-authorization", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    json_path = run_dir / OUTPUT_JSON
    csv_path = run_dir / OUTPUT_CSV
    with evaluation_lock(run_dir):
        if json_path.exists() or csv_path.exists():
            raise ValueError("native development evaluation is intentionally one-shot")
        inputs = load_verified_inputs(
            run_dir,
            args.actions,
            args.audit,
            args.evaluation_authorization,
        )
        evaluator_git_commit = _require_committed_inputs(
            (
                __file__,
                inputs.paths["evaluation_authorization"],
                AUDITOR_PATH,
                inputs.paths["actions"],
                inputs.paths["source_template"],
                inputs.paths["prompts"],
                inputs.paths["prompt_manifest"],
            )
        )
        result = evaluate_frame(
            inputs.frame,
            inputs.protocol,
            expected_prompt_indices=inputs.prompt_indices,
            expected_seeds=inputs.seeds,
        )
        result["provenance"] = {
            "run_dir": str(run_dir),
            "run_config_sha256": inputs.hashes["run_config"],
            "manifest_sha256": inputs.hashes["manifest"],
            "scores_sha256": inputs.hashes["scores"],
            "actions_path": inputs.paths["actions"],
            "actions_sha256": inputs.hashes["actions"],
            "audit_path": inputs.paths["audit"],
            "audit_sha256": inputs.hashes["audit"],
            "evaluation_authorization_path": inputs.paths[
                "evaluation_authorization"
            ],
            "evaluation_authorization_sha256": inputs.hashes[
                "evaluation_authorization"
            ],
            "prompts_path": inputs.paths["prompts"],
            "prompts_sha256": inputs.hashes["prompts"],
            "prompt_manifest_path": inputs.paths["prompt_manifest"],
            "prompt_manifest_sha256": inputs.hashes["prompt_manifest"],
            "source_template_path": inputs.paths["source_template"],
            "source_template_sha256": inputs.hashes["source_template"],
            "auditor_path": str(AUDITOR_PATH.resolve()),
            "auditor_script_sha256": sha256_file(AUDITOR_PATH),
            "evaluator_script_sha256": sha256_file(__file__),
            "evaluator_git_commit": evaluator_git_commit,
            "runtime_versions": _runtime_versions(),
        }
        json_payload = (
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        csv_payload = pd.DataFrame(result["rows"]).to_csv(index=False).encode("utf-8")
        _atomic_write(csv_path, csv_payload)
        _atomic_write(json_path, json_payload)
    print(json.dumps({"decision": result["decision"]}, indent=2))
    print(f"evaluation -> {json_path}")


if __name__ == "__main__":
    main()
