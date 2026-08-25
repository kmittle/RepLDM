"""Evaluate the registered structural-control development matrix.

This module reports every preregistered contrast and robustness diagnostic. It
does not rank actions, select a method, authorize validation, or authorize RL.
The executable action file and a passing result-blind run audit are mandatory;
the blocked registration template is deliberately rejected as an input run.
"""

from __future__ import annotations

import argparse
import copy
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
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

import audit_structural_control_run as structural_audit
from compare_actions import (
    holm_adjust,
    merge_manifest_scores,
    prompt_sign_flip_pvalue,
    validate_expected_grid,
    validate_pairing,
)


ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE_SCHEMA = "scheduler_native_structural_controls_actions_v1"
REGISTRATION_SCHEMA = "scheduler_native_structural_controls_v1"
REGISTRATION_FILE_SCHEMA = "structural_control_registration_v1"
EXPERIMENT_ID = "scheduler_native_structural_controls_development_v1"
AUTH_SCOPE = "development_only_baseline_calibration"
SCORER_SCHEMA = "repldm_scorer_provenance_v1"
EVALUATION_SCHEMA = "structural_control_development_evaluation_v1"
EVALUATOR_VERSION = "structural_control_evaluator_v1"
OUTPUT_JSON = "structural_control_evaluation.json"
OUTPUT_CSV = "structural_control_contrasts.csv"
AUDITOR_PATH = ROOT / "eval-pipeline" / "audit_structural_control_run.py"
BASE_AUDITOR_PATH = ROOT / "eval-pipeline" / "audit_latent_renderer_run.py"
COMPARE_ACTIONS_PATH = ROOT / "eval-pipeline" / "compare_actions.py"
AUDIT_SCHEMA = "scheduler_native_structural_control_audit_v1"
ANALYSIS_IMPLEMENTATION_SCHEMA = "structural_control_analysis_implementation_v1"
EXPECTED_ANALYSIS_PATHS = (
    "AttentionGuidance/__init__.py",
    "eval-pipeline/audit_latent_renderer_run.py",
    "eval-pipeline/audit_structural_control_run.py",
    "eval-pipeline/compare_actions.py",
    "eval-pipeline/evaluate_structural_control_run.py",
    "eval-pipeline/generate.py",
    "eval-pipeline/generation_environment.py",
    "eval-pipeline/s7_provenance.py",
    "eval-pipeline/scorer_provenance.py",
)

EXPECTED_PROMPTS = 33
EXPECTED_SEEDS = 3
EXPECTED_CHALLENGES = 11
EXPECTED_ACTIONS = (
    "no_op_cfg7p5",
    "cfg_only_5",
    "conference_tfsa",
    "freeu_diffusers_historical",
    "freeu_diffusers_paper_parameters",
    "freeu_paper_adaptive",
    "pladis_operator_port",
    "gag_eq13_reimplementation",
)
EXPECTED_TASKS = EXPECTED_PROMPTS * EXPECTED_SEEDS * len(EXPECTED_ACTIONS)
EXPECTED_ACTION_CONTRACT = {
    "no_op_cfg7p5": ("none", "nominal_reference", 7.5),
    "cfg_only_5": ("none", "matched_cfg_reference_for_attention_baselines", 5.0),
    "conference_tfsa": ("legacy", "conference_method_reference", 7.5),
    "freeu_diffusers_historical": (
        "freeu",
        "historical_surrogate_reference",
        7.5,
    ),
    "freeu_diffusers_paper_parameters": (
        "freeu",
        "matched_constant_paper_parameters",
        7.5,
    ),
    "freeu_paper_adaptive": ("freeu", "pinned_paper_operator", 7.5),
    "pladis_operator_port": (
        "attention_baseline",
        "pinned_operator_port_not_upstream_environment",
        5.0,
    ),
    "gag_eq13_reimplementation": (
        "attention_baseline",
        "paper_derived_reimplementation_not_official_code",
        5.0,
    ),
}

ADDITIVE_METRICS = (
    "topiq_nr",
    "hpsv2",
    "clip_cosine",
    "clipped_fraction",
    "mean_saturation",
    "colorfulness",
    "laplacian_sharpness",
)
RATIO_METRIC = "contrast_std"
REQUIRED_SCORE_METRICS = (*ADDITIVE_METRICS, RATIO_METRIC)
RUNTIME_COLUMNS = (
    "inference_seconds",
    "peak_gpu_memory_bytes",
    "num_inference_steps",
    "guidance_scale",
    "extra_unet_calls",
    "unet_calls_per_step",
)

EXPECTED_DESIGN = {
    "role": "development_only_baseline_calibration",
    "prompt_count": EXPECTED_PROMPTS,
    "seed_count": EXPECTED_SEEDS,
    "action_count": len(EXPECTED_ACTIONS),
    "expected_task_count": EXPECTED_TASKS,
    "paired_block_policy": "one_prompt_seed_block_per_gpu",
    "method_selection": False,
    "confirmation": False,
    "reuse_scope": "same_development_split_as_headroom_screen",
}
EXPECTED_ENGINEERING_SMOKE = {
    "role": "engineering_only",
    "engineering_only": True,
    "prompts": "eval-pipeline/prompts/scheduler_native_fixed_headroom_smoke.csv",
    "prompts_sha256": "60a3bf278689165bcb7a4bdf1f18c7ab91d8bf7a77eb287369a60fe40b6ef4e1",
    "expected_prompt_count": 11,
    "expected_challenges": 11,
    "seeds": [1798464083],
    "action_count": 8,
    "expected_task_count": 88,
    "require_all_actions_distinct_within_block": True,
    "quality_scoring": False,
    "formal_matrix_evidence": False,
    "quality_claim_allowed": False,
    "method_selection_allowed": False,
}
EXPECTED_FAMILY_SPECS = {
    "freeu_vs_no_op": {
        "reference": "no_op_cfg7p5",
        "treatments": [
            "freeu_diffusers_historical",
            "freeu_diffusers_paper_parameters",
            "freeu_paper_adaptive",
        ],
        "holm_alpha": 0.05,
        "point_estimate_screen_delta": 0.005,
    },
    "freeu_mechanism_contrasts": {
        "holm_alpha": 0.05,
        "point_estimate_screen_delta": 0.005,
        "contrasts": [
            {
                "id": "paper_parameter_delta_within_constant_operator",
                "treatment": "freeu_diffusers_paper_parameters",
                "reference": "freeu_diffusers_historical",
            },
            {
                "id": "adaptive_operator_delta_at_paper_parameters",
                "treatment": "freeu_paper_adaptive",
                "reference": "freeu_diffusers_paper_parameters",
            },
        ],
    },
    "attention_vs_cfg5": {
        "reference": "cfg_only_5",
        "treatments": ["pladis_operator_port", "gag_eq13_reimplementation"],
        "holm_alpha": 0.05,
        "point_estimate_screen_delta": 0.005,
    },
}
EXPECTED_DESCRIPTIVE_CONTRASTS = [
    {
        "id": "conference_method_vs_nominal_cfg7p5",
        "treatment": "conference_tfsa",
        "reference": "no_op_cfg7p5",
        "confirmatory": False,
    }
]
EXPECTED_REPORT = (
    "every_registered_action_including_failures",
    "paired_deltas_and_confidence_intervals",
    "hpsv2_and_clip_noninferiority",
    "clipping_saturation_contrast_colorfulness_and_sharpness",
    "latency_peak_memory_and_unet_calls",
    "equal_nfe_does_not_imply_equal_compute",
    "prompt_median_win_probability_and_sign_consistency",
    "per_seed_leave_one_seed_out_and_leave_one_prompt_out_effects",
    "every_guard_for_every_action_without_selective_omission",
)
EXPECTED_INTERPRETATION = (
    "no_action_can_authorize_method_selection_or_rl",
    "freeu_matched_contrasts_do_not_form_a_complete_factorial",
    "pladis_and_gag_are_only_operator_matched_against_cfg_only_5",
    "pladis_is_an_operator_port_not_an_upstream_environment_reproduction",
    "gag_is_a_paper_derived_reimplementation_without_verified_author_code",
    "point_estimate_screen_delta_is_not_a_lower_bound_on_true_effect",
)
EXPECTED_AUTH_FIELDS = {
    "reviewer",
    "reviewed_commit",
    "source_template",
    "source_template_sha256",
    "scope",
    "gpu_generation",
    "scoring",
    "method_selection",
    "result_access_before_freeze",
}


@dataclass(frozen=True)
class Contrast:
    id: str
    family: str
    treatment: str
    reference: str
    family_index: int
    holm_alpha: float | None
    inferential_use: str


@dataclass(frozen=True)
class Protocol:
    action_order: tuple[str, ...]
    action_cfg: Mapping[str, float]
    contrasts: tuple[Contrast, ...]
    descriptive_contrasts: tuple[Contrast, ...]
    point_estimate_screen_delta: float
    bootstrap: int
    confidence_level: float
    randomizations: int
    random_seed: int
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


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_exact_fields(
    value: Any, label: str, fields: set[str]
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, label)
    if set(mapping) != fields:
        raise ValueError(f"{label} fields differ from the frozen v1 contract")
    return mapping


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_exact_number(value: Any, expected: float, label: str) -> float:
    observed = _require_number(value, label)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} differs from the frozen value {expected}")
    return observed


def _registered_contrasts() -> tuple[Contrast, ...]:
    contrasts: list[Contrast] = []
    for family, spec in EXPECTED_FAMILY_SPECS.items():
        alpha = float(spec["holm_alpha"])
        if "treatments" in spec:
            reference = str(spec["reference"])
            members = [
                (f"{treatment}_vs_{reference}", str(treatment), reference)
                for treatment in spec["treatments"]
            ]
        else:
            members = [
                (
                    str(member["id"]),
                    str(member["treatment"]),
                    str(member["reference"]),
                )
                for member in spec["contrasts"]
            ]
        for index, (contrast_id, treatment, reference) in enumerate(members):
            contrasts.append(
                Contrast(
                    id=contrast_id,
                    family=family,
                    treatment=treatment,
                    reference=reference,
                    family_index=index,
                    holm_alpha=alpha,
                    inferential_use="development_screen_only",
                )
            )
    return tuple(contrasts)


def load_protocol(actions_config: Mapping[str, Any]) -> Protocol:
    """Parse the exact authorized development analysis contract."""
    if actions_config.get("schema") != EXECUTABLE_SCHEMA:
        raise ValueError(
            "structural-control evaluator requires the authorized executable schema"
        )
    if actions_config.get("registration_schema") != REGISTRATION_SCHEMA:
        raise ValueError("structural-control registration_schema differs")
    if actions_config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("structural-control experiment_id differs")
    if actions_config.get("status") != "authorized_development":
        raise ValueError("structural-control executable is not authorized_development")
    if actions_config.get("split_role") != "development":
        raise ValueError("structural-control evaluator is development-only")
    if actions_config.get("design") != EXPECTED_DESIGN:
        raise ValueError("structural-control design differs from the frozen v1 matrix")
    if actions_config.get("engineering_smoke") != EXPECTED_ENGINEERING_SMOKE:
        raise ValueError("structural-control engineering smoke profile differs")
    if actions_config.get("failure_policy") != "shared_abort_after_first_task_error":
        raise ValueError("structural-control failure policy differs")

    authorization = _require_exact_fields(
        actions_config.get("authorization"), "authorization", EXPECTED_AUTH_FIELDS
    )
    if not str(authorization.get("reviewer", "")).strip():
        raise ValueError("structural-control authorization reviewer is missing")
    if not re.fullmatch(r"[0-9a-f]{40}", str(authorization.get("reviewed_commit", ""))):
        raise ValueError("structural-control reviewed_commit must be full lowercase 40-hex")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(authorization.get("source_template_sha256", ""))
    ):
        raise ValueError("structural-control source-template hash is invalid")
    if authorization.get("scope") != AUTH_SCOPE:
        raise ValueError("structural-control authorization scope differs")
    if authorization.get("gpu_generation") is not True:
        raise ValueError("structural-control generation was not authorized")
    if authorization.get("scoring") is not True:
        raise ValueError("structural-control scoring was not authorized")
    if authorization.get("method_selection") is not False:
        raise ValueError("structural-control evaluator forbids method selection")
    if authorization.get("result_access_before_freeze") is not False:
        raise ValueError("structural-control result access was not frozen")

    actions = _require_list(actions_config.get("actions"), "actions")
    action_order = tuple(str(action.get("id", "")) for action in actions)
    if action_order != EXPECTED_ACTIONS:
        raise ValueError("structural-control action IDs/order differ from frozen v1")
    action_cfg: dict[str, float] = {}
    for action in actions:
        action_id = str(action["id"])
        expected_type, expected_role, expected_cfg = EXPECTED_ACTION_CONTRACT[action_id]
        if action.get("type") != expected_type or action.get("role") != expected_role:
            raise ValueError(f"{action_id}: type or role differs from frozen v1")
        action_cfg[action_id] = _require_exact_number(
            action.get("cfg_scale"), expected_cfg, f"{action_id}.cfg_scale"
        )

    analysis = _require_exact_fields(
        actions_config.get("analysis"),
        "analysis",
        {
            "primary_metric",
            "point_estimate_screen_delta",
            "inference",
            "multiplicity_families",
            "descriptive_contrasts",
            "noninferiority_guards",
            "report",
            "interpretation",
        },
    )
    if analysis.get("primary_metric") != "topiq_nr":
        raise ValueError("structural-control primary metric must be topiq_nr")
    point_delta = _require_exact_number(
        analysis.get("point_estimate_screen_delta"),
        0.005,
        "analysis.point_estimate_screen_delta",
    )
    if analysis.get("multiplicity_families") != EXPECTED_FAMILY_SPECS:
        raise ValueError("structural-control Holm families differ from frozen v1")
    if analysis.get("descriptive_contrasts") != EXPECTED_DESCRIPTIVE_CONTRASTS:
        raise ValueError("structural-control descriptive contrasts differ from frozen v1")
    if tuple(_require_list(analysis.get("report"), "analysis.report")) != EXPECTED_REPORT:
        raise ValueError("structural-control report contract differs from frozen v1")
    if tuple(
        _require_list(analysis.get("interpretation"), "analysis.interpretation")
    ) != EXPECTED_INTERPRETATION:
        raise ValueError("structural-control interpretation differs from frozen v1")

    inference = _require_exact_fields(
        analysis.get("inference"),
        "analysis.inference",
        {
            "crossed_prompt_seed_bootstrap",
            "confidence_level",
            "prompt_level_sign_flips",
            "random_seed",
            "generalization_unit",
        },
    )
    if inference.get("generalization_unit") != "prompt_after_seed_mean":
        raise ValueError("sign-flip generalization unit must be prompt_after_seed_mean")
    bootstrap = inference.get("crossed_prompt_seed_bootstrap")
    randomizations = inference.get("prompt_level_sign_flips")
    random_seed = inference.get("random_seed")
    if (
        isinstance(bootstrap, bool)
        or bootstrap != 10000
        or isinstance(randomizations, bool)
        or randomizations != 100000
        or isinstance(random_seed, bool)
        or random_seed != 2026
    ):
        raise ValueError("structural-control inference counts or random seed differ")
    confidence = _require_exact_number(
        inference.get("confidence_level"), 0.95, "analysis.inference.confidence_level"
    )

    guards = _require_exact_fields(
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

    descriptive = EXPECTED_DESCRIPTIVE_CONTRASTS[0]
    descriptive_contrasts = (
        Contrast(
            id=str(descriptive["id"]),
            family="descriptive_contrasts",
            treatment=str(descriptive["treatment"]),
            reference=str(descriptive["reference"]),
            family_index=0,
            holm_alpha=None,
            inferential_use="descriptive_only",
        ),
    )
    return Protocol(
        action_order=action_order,
        action_cfg=action_cfg,
        contrasts=_registered_contrasts(),
        descriptive_contrasts=descriptive_contrasts,
        point_estimate_screen_delta=point_delta,
        bootstrap=int(bootstrap),
        confidence_level=confidence,
        randomizations=int(randomizations),
        random_seed=int(random_seed),
        hpsv2_ci_lower_min=_require_exact_number(
            guards.get("hpsv2_ci_lower_min_delta"),
            -0.005,
            "hpsv2 noninferiority bound",
        ),
        clip_ci_lower_min=_require_exact_number(
            guards.get("clip_cosine_ci_lower_min_delta"),
            -0.005,
            "CLIP noninferiority bound",
        ),
        clipped_ci_upper_max=_require_exact_number(
            guards.get("clipped_fraction_ci_upper_max_delta"),
            0.001,
            "clipped-fraction guard",
        ),
        saturation_ci_upper_max=_require_exact_number(
            guards.get("mean_saturation_ci_upper_max_delta"),
            0.005,
            "saturation guard",
        ),
        contrast_ratio_min=_require_exact_number(
            contrast_interval[0], 0.95, "contrast ratio lower bound"
        ),
        contrast_ratio_max=_require_exact_number(
            contrast_interval[1], 1.05, "contrast ratio upper bound"
        ),
    )


def _crossed_bootstrap_ci(
    values: pd.Series, *, n_boot: int, confidence_level: float, seed: int
) -> tuple[float, float]:
    """Resample prompt and seed axes independently, preserving paired cells."""
    matrix = values.unstack("seed")
    if matrix.isna().any().any():
        raise ValueError("crossed bootstrap requires a complete prompt x seed matrix")
    array = matrix.to_numpy(dtype=float)
    if array.shape != (EXPECTED_PROMPTS, EXPECTED_SEEDS):
        raise ValueError("crossed bootstrap received an unexpected matrix shape")
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


def _numeric_pivots(
    frame: pd.DataFrame, protocol: Protocol
) -> dict[str, pd.DataFrame]:
    pivots: dict[str, pd.DataFrame] = {}
    for metric in (*REQUIRED_SCORE_METRICS, "inference_seconds", "peak_gpu_memory_bytes"):
        values = pd.to_numeric(frame[metric], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"metric {metric!r} contains missing or non-finite values")
        pivot = frame.assign(**{metric: values}).pivot(
            index=["prompt_index", "seed"], columns="action_id", values=metric
        )
        pivot = pivot.reindex(columns=list(protocol.action_order))
        if pivot.shape != (EXPECTED_PROMPTS * EXPECTED_SEEDS, len(EXPECTED_ACTIONS)):
            raise ValueError(f"metric {metric!r} has an unexpected registered-grid shape")
        if pivot.isna().any().any():
            raise ValueError(f"metric {metric!r} is incomplete on the registered grid")
        pivots[metric] = pivot
    if (pivots[RATIO_METRIC] <= 0.0).any().any():
        raise ValueError("contrast_std must be strictly positive for log-ratio inference")
    return pivots


def _validate_runtime_contract(frame: pd.DataFrame, protocol: Protocol) -> None:
    numeric = {}
    for column in RUNTIME_COLUMNS[:-1]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"runtime field {column!r} contains non-finite values")
        numeric[column] = values
    if (numeric["inference_seconds"] <= 0.0).any():
        raise ValueError("inference_seconds must be strictly positive")
    if (numeric["peak_gpu_memory_bytes"] < 0.0).any():
        raise ValueError("peak_gpu_memory_bytes must be non-negative")
    if not (numeric["num_inference_steps"] == 50).all():
        raise ValueError("every structural-control task must record 50 inference steps")
    if not (numeric["extra_unet_calls"] == 0).all():
        raise ValueError("structural controls require zero extra U-Net calls")
    expected_cfg = frame["action_id"].map(protocol.action_cfg).astype(float)
    if not np.array_equal(numeric["guidance_scale"].to_numpy(), expected_cfg.to_numpy()):
        raise ValueError("runtime guidance_scale differs from the registered action")
    for row_id, calls in zip(frame.get("id", frame.index), frame["unet_calls_per_step"]):
        if calls != [1] * 50:
            raise ValueError(f"{row_id}: expected exactly one U-Net call on each of 50 steps")


def _robustness_summary(
    delta: pd.Series, challenge_by_prompt: Mapping[int, str]
) -> dict[str, Any]:
    prompt_delta = delta.groupby(level="prompt_index", sort=True).mean()
    per_seed = delta.groupby(level="seed", sort=True).mean()
    overall = float(prompt_delta.mean())
    positive = int((prompt_delta > 0.0).sum())
    negative = int((prompt_delta < 0.0).sum())
    ties = int((prompt_delta == 0.0).sum())
    leave_seed = {
        str(int(seed)): float(
            delta[delta.index.get_level_values("seed") != seed].mean()
        )
        for seed in per_seed.index
    }
    leave_prompt = {
        str(int(prompt)): float(prompt_delta.drop(prompt).mean())
        for prompt in prompt_delta.index
    }
    challenge_labels = pd.Series(
        [challenge_by_prompt[int(prompt)] for prompt in prompt_delta.index],
        index=prompt_delta.index,
    )
    per_challenge = prompt_delta.groupby(challenge_labels).mean().sort_index()
    overall_sign = int(np.sign(overall))
    seed_signs = np.sign(per_seed.to_numpy(dtype=float)).astype(int)
    return {
        "prompt_median_delta": float(prompt_delta.median()),
        "prompt_win_probability": float((positive + 0.5 * ties) / len(prompt_delta)),
        "prompt_positive_count": positive,
        "prompt_negative_count": negative,
        "prompt_tie_count": ties,
        "per_seed_mean_delta": {
            str(int(seed)): float(value) for seed, value in per_seed.items()
        },
        "leave_one_seed_out_mean_delta": leave_seed,
        "leave_one_prompt_out_mean_delta": leave_prompt,
        "challenge_mean_delta": {
            str(challenge): float(value) for challenge, value in per_challenge.items()
        },
        "challenge_inferential_use": "descriptive_only",
        "all_seed_means_match_overall_sign": bool(
            overall_sign != 0 and np.all(seed_signs == overall_sign)
        ),
    }


def _contrast_statistics(
    contrast: Contrast,
    pivots: Mapping[str, pd.DataFrame],
    protocol: Protocol,
    challenge_by_prompt: Mapping[int, str],
    *,
    contrast_seed: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "contrast_id": contrast.id,
        "family": contrast.family,
        "family_index": contrast.family_index,
        "treatment": contrast.treatment,
        "reference": contrast.reference,
        "inferential_use": contrast.inferential_use,
        "n_prompts": EXPECTED_PROMPTS,
        "n_seeds": EXPECTED_SEEDS,
        "n_paired_cells": EXPECTED_PROMPTS * EXPECTED_SEEDS,
        "sign_flip_unit": "prompt_after_seed_mean",
    }
    deltas: dict[str, pd.Series] = {}
    for metric_index, metric in enumerate(ADDITIVE_METRICS):
        delta = pivots[metric][contrast.treatment] - pivots[metric][contrast.reference]
        deltas[metric] = delta
        ci_low, ci_high = _crossed_bootstrap_ci(
            delta,
            n_boot=protocol.bootstrap,
            confidence_level=protocol.confidence_level,
            seed=contrast_seed + metric_index,
        )
        row[f"{metric}_mean_delta"] = float(delta.mean())
        row[f"{metric}_ci_low"] = ci_low
        row[f"{metric}_ci_high"] = ci_high

    log_contrast_ratio = np.log(
        pivots[RATIO_METRIC][contrast.treatment]
        / pivots[RATIO_METRIC][contrast.reference]
    )
    contrast_ci = _crossed_bootstrap_ci(
        log_contrast_ratio,
        n_boot=protocol.bootstrap,
        confidence_level=protocol.confidence_level,
        seed=contrast_seed + len(ADDITIVE_METRICS),
    )
    row["contrast_geometric_mean_ratio"] = float(
        math.exp(float(log_contrast_ratio.mean()))
    )
    row["contrast_ratio_ci_low"] = float(math.exp(contrast_ci[0]))
    row["contrast_ratio_ci_high"] = float(math.exp(contrast_ci[1]))

    for metric in ("inference_seconds", "peak_gpu_memory_bytes"):
        delta = pivots[metric][contrast.treatment] - pivots[metric][contrast.reference]
        row[f"{metric}_mean_delta"] = float(delta.mean())
        row[f"{metric}_median_delta"] = float(
            delta.groupby(level="prompt_index").mean().median()
        )

    topiq_delta = deltas["topiq_nr"]
    row.update(_robustness_summary(topiq_delta, challenge_by_prompt))
    row["topiq_p_sign_flip"] = float(
        prompt_sign_flip_pvalue(
            topiq_delta,
            n_random=protocol.randomizations,
            seed=contrast_seed,
        )
    )
    row["quality_guards"] = {
        "hpsv2_noninferiority_pass": bool(
            row["hpsv2_ci_low"] >= protocol.hpsv2_ci_lower_min
        ),
        "clip_cosine_noninferiority_pass": bool(
            row["clip_cosine_ci_low"] >= protocol.clip_ci_lower_min
        ),
        "clipped_fraction_guard_pass": bool(
            row["clipped_fraction_ci_high"] <= protocol.clipped_ci_upper_max
        ),
        "mean_saturation_guard_pass": bool(
            row["mean_saturation_ci_high"] <= protocol.saturation_ci_upper_max
        ),
        "contrast_guard_pass": bool(
            row["contrast_ratio_ci_low"] >= protocol.contrast_ratio_min
            and row["contrast_ratio_ci_high"] <= protocol.contrast_ratio_max
        ),
    }
    row.update(row["quality_guards"])
    row["all_quality_guards_pass"] = bool(all(row["quality_guards"].values()))
    return row


def _apply_development_screen(row: dict[str, Any], protocol: Protocol) -> None:
    checks = {
        "topiq_point_estimate_screen_pass": bool(
            row["topiq_nr_mean_delta"] >= protocol.point_estimate_screen_delta
        ),
        "topiq_positive_ci_pass": bool(row["topiq_nr_ci_low"] > 0.0),
        "topiq_family_holm_pass": bool(
            row["topiq_p_holm"] < float(row["holm_alpha"])
        ),
        "all_quality_guards_pass": bool(row["all_quality_guards_pass"]),
    }
    row.update(checks)
    row["passes_registered_development_screen"] = bool(all(checks.values()))
    row["effect_at_least_point_threshold_established"] = False


def _action_summaries(frame: pd.DataFrame, protocol: Protocol) -> list[dict[str, Any]]:
    summaries = []
    for action_id in protocol.action_order:
        group = frame.loc[frame["action_id"] == action_id]
        if len(group) != EXPECTED_PROMPTS * EXPECTED_SEEDS:
            raise ValueError(f"{action_id}: action does not contain exactly 99 tasks")
        metric_summary = {}
        for metric in REQUIRED_SCORE_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            metric_summary[metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
            }
        latency = pd.to_numeric(group["inference_seconds"]).to_numpy(dtype=float)
        memory = pd.to_numeric(group["peak_gpu_memory_bytes"]).to_numpy(dtype=float)
        summaries.append(
            {
                "action_id": action_id,
                "registered_cfg_scale": float(protocol.action_cfg[action_id]),
                "expected_tasks": EXPECTED_PROMPTS * EXPECTED_SEEDS,
                "observed_tasks": int(len(group)),
                "audited_missing_or_corrupt_tasks": 0,
                "metrics": metric_summary,
                "inference_seconds_mean": float(np.mean(latency)),
                "inference_seconds_median": float(np.median(latency)),
                "inference_seconds_iqr": [
                    float(np.quantile(latency, 0.25)),
                    float(np.quantile(latency, 0.75)),
                ],
                "peak_gpu_memory_bytes_median": float(np.median(memory)),
                "peak_gpu_memory_bytes_max": int(np.max(memory)),
                "unet_forward_calls_per_task": 50,
                "cfg_branches_per_forward": 2,
                "branch_equivalent_nfe_per_task": 100,
            }
        )
    return summaries


def evaluate_frame(
    frame: pd.DataFrame,
    protocol: Protocol,
    *,
    expected_prompt_indices: Sequence[int],
    expected_seeds: Sequence[int],
    audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report the frozen development estimands without selecting an action."""
    required = {
        "id",
        "action_id",
        "prompt_index",
        "seed",
        "device",
        "source_challenge",
        *REQUIRED_SCORE_METRICS,
        *RUNTIME_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"evaluation frame lacks required columns: {missing}")
    if len(expected_prompt_indices) != EXPECTED_PROMPTS:
        raise ValueError("structural-control evaluator requires exactly 33 prompts")
    if len(expected_seeds) != EXPECTED_SEEDS:
        raise ValueError("structural-control evaluator requires exactly 3 seeds")
    validate_expected_grid(
        frame, expected_prompt_indices, expected_seeds, protocol.action_order
    )
    if len(frame) != EXPECTED_TASKS:
        raise ValueError(f"structural-control evaluation requires exactly {EXPECTED_TASKS} tasks")
    validate_pairing(frame)
    _validate_runtime_contract(frame, protocol)

    prompt_challenges = frame[["prompt_index", "source_challenge"]].drop_duplicates()
    if prompt_challenges["prompt_index"].duplicated().any():
        raise ValueError("a prompt is assigned to multiple source challenges")
    challenge_counts = prompt_challenges.groupby("source_challenge")[
        "prompt_index"
    ].nunique()
    if len(challenge_counts) != EXPECTED_CHALLENGES or set(
        challenge_counts.astype(int)
    ) != {3}:
        raise ValueError("development requires 11 challenges with 3 prompts each")
    challenge_by_prompt = {
        int(row["prompt_index"]): str(row["source_challenge"])
        for _, row in prompt_challenges.iterrows()
    }
    pivots = _numeric_pivots(frame, protocol)

    rows: list[dict[str, Any]] = []
    for contrast_index, contrast in enumerate(protocol.contrasts):
        rows.append(
            _contrast_statistics(
                contrast,
                pivots,
                protocol,
                challenge_by_prompt,
                contrast_seed=protocol.random_seed + contrast_index * 100,
            )
        )
    for family, spec in EXPECTED_FAMILY_SPECS.items():
        family_rows = [row for row in rows if row["family"] == family]
        adjusted = holm_adjust([row["topiq_p_sign_flip"] for row in family_rows])
        for row, adjusted_p in zip(family_rows, adjusted):
            row["topiq_p_holm"] = float(adjusted_p)
            row["holm_alpha"] = float(spec["holm_alpha"])
            row["multiplicity_scope"] = "within_registered_family_only"
            _apply_development_screen(row, protocol)

    descriptive_rows = []
    for offset, contrast in enumerate(protocol.descriptive_contrasts):
        row = _contrast_statistics(
            contrast,
            pivots,
            protocol,
            challenge_by_prompt,
            contrast_seed=protocol.random_seed + 10000 + offset * 100,
        )
        row.update(
            {
                "topiq_p_holm": None,
                "holm_alpha": None,
                "multiplicity_scope": "none_descriptive_only",
                "passes_registered_development_screen": None,
                "effect_at_least_point_threshold_established": False,
            }
        )
        descriptive_rows.append(row)
    all_rows = rows + descriptive_rows

    family_summary = {
        family: {
            "correction": "Holm",
            "alpha": float(spec["holm_alpha"]),
            "contrast_ids": [
                row["contrast_id"] for row in rows if row["family"] == family
            ],
        }
        for family, spec in EXPECTED_FAMILY_SPECS.items()
    }
    audit_warnings = [] if audit is None else list(audit.get("warnings") or [])
    return {
        "schema": EVALUATION_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "scope": "development_evidence_reporting_only",
        "screen_only": True,
        "method_selection_authorized": False,
        "method_selection_performed": False,
        "validation_authorized": False,
        "rl_authorized": False,
        "publication_superiority_established": False,
        "claims": {
            "point_estimate_threshold_is_not_effect_lower_bound": True,
            "population_generalization_established": False,
            "cross_cfg_rankings_are_causal": False,
            "equal_nfe_implies_equal_compute": False,
            "challenge_inference_is_descriptive_only": True,
            "global_multiplicity_across_families_controlled": False,
        },
        "task_accounting": {
            "expected_tasks": EXPECTED_TASKS,
            "observed_manifest_score_pairs": int(len(frame)),
            "missing_tasks": 0,
            "corrupt_or_nonfinite_tasks": 0,
            "expected_prompt_seed_blocks": EXPECTED_PROMPTS * EXPECTED_SEEDS,
            "audit_warnings": audit_warnings,
        },
        "analysis_contract": {
            "primary_metric": "topiq_nr",
            "point_estimate_screen_delta": protocol.point_estimate_screen_delta,
            "crossed_prompt_seed_bootstrap": protocol.bootstrap,
            "confidence_level": protocol.confidence_level,
            "prompt_level_sign_flips": protocol.randomizations,
            "generalization_unit": "prompt_after_seed_mean",
            "robustness_analyses_are_descriptive": True,
        },
        "multiplicity_families": family_summary,
        "descriptive_contrast_ids": [
            contrast.id for contrast in protocol.descriptive_contrasts
        ],
        "action_summaries": _action_summaries(frame, protocol),
        "contrasts": all_rows,
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


def _load_jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSONL") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must contain non-empty JSON-object rows")
    return rows


def _require_hash(
    mapping: Mapping[str, Any], key: str, observed: str, label: str
) -> None:
    if mapping.get(key) != observed:
        raise ValueError(f"{label} hash mismatch for {key}")


def _registration_design_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(value))
    for key in (
        "schema",
        "status",
        "authorization",
        "blocking_conditions",
        "registration_source",
        "implementation_source",
    ):
        body.pop(key, None)
    return body


def _validate_analysis_implementation(
    actions_config: Mapping[str, Any], registration_config: Mapping[str, Any]
) -> Mapping[str, Any]:
    implementation = _require_exact_fields(
        actions_config.get("analysis_implementation"),
        "analysis_implementation",
        {"schema", "files"},
    )
    if implementation != registration_config.get("analysis_implementation"):
        raise ValueError("analysis implementation differs from frozen registration")
    if implementation.get("schema") != ANALYSIS_IMPLEMENTATION_SCHEMA:
        raise ValueError("analysis implementation schema differs")
    files = _require_mapping(implementation.get("files"), "analysis files")
    if set(files) != set(EXPECTED_ANALYSIS_PATHS):
        raise ValueError("analysis implementation paths differ")
    for relative_path, expected_hash in files.items():
        path = (ROOT / str(relative_path)).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("analysis implementation path is outside the repository") from exc
        if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)):
            raise ValueError(f"analysis implementation entry is invalid: {relative_path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"analysis implementation hash differs: {relative_path}")
    return implementation


def load_verified_inputs(
    run_dir: str | os.PathLike[str],
    actions_path: str | os.PathLike[str],
    audit_path: str | os.PathLike[str],
) -> VerifiedInputs:
    """Bind a complete run, scores, executable design, and passing run audit."""
    run_root = Path(run_dir).resolve()
    paths: dict[str, Path] = {
        "run_config": run_root / "config.json",
        "manifest": run_root / "manifest.jsonl",
        "scores": run_root / "scores.jsonl",
        "actions": Path(actions_path).resolve(),
        "audit": Path(audit_path).resolve(),
    }
    if paths["audit"] != run_root / "run_audit.json":
        raise ValueError("structural evaluator requires the canonical run_audit.json")
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"required {label} file is missing: {path}")
    payloads = {label: path.read_bytes() for label, path in paths.items()}
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
    registration_source = _require_mapping(
        actions_config.get("registration_source"), "registration_source"
    )
    prompts_path = _resolve_registered_path(
        source_manifest.get("prompts"), "registered prompts"
    )
    prompt_manifest_path = _resolve_registered_path(
        source_manifest.get("path"), "prompt manifest"
    )
    registration_path = _resolve_registered_path(
        registration_source.get("path"), "registration source"
    )
    for label, path in (
        ("prompts", prompts_path),
        ("prompt_manifest", prompt_manifest_path),
        ("registration", registration_path),
    ):
        paths[label] = path
        payloads[label] = path.read_bytes()
        hashes[label] = sha256_bytes(payloads[label])
    _require_hash(source_manifest, "prompts_sha256", hashes["prompts"], "source_manifest")
    _require_hash(
        source_manifest, "sha256", hashes["prompt_manifest"], "source_manifest"
    )
    _require_hash(
        registration_source, "sha256", hashes["registration"], "registration_source"
    )
    authorization = _require_mapping(actions_config.get("authorization"), "authorization")
    if authorization.get("source_template") != registration_source.get("path"):
        raise ValueError("authorization source-template path differs from registration")
    if authorization.get("source_template_sha256") != hashes["registration"]:
        raise ValueError("authorization source-template hash differs from registration")
    try:
        registration_config = yaml.safe_load(payloads["registration"]) or {}
    except yaml.YAMLError as exc:
        raise ValueError("registration source is invalid YAML") from exc
    if registration_config.get("schema") != REGISTRATION_FILE_SCHEMA:
        raise ValueError("registration source has the wrong blocked schema")
    if registration_config.get("status") != "blocked_registration_only":
        raise ValueError("registration source is not the result-blind blocked template")
    registration_authorization = _require_mapping(
        registration_config.get("authorization"), "registration authorization"
    )
    for key in ("gpu_generation", "scoring", "method_selection"):
        if registration_authorization.get(key) is not False:
            raise ValueError(f"registration source must keep {key} false")
    if _registration_design_body(actions_config) != _registration_design_body(
        registration_config
    ):
        raise ValueError("authorized executable differs from the frozen registration")
    analysis_implementation = _validate_analysis_implementation(
        actions_config, registration_config
    )

    run_config = _load_json(payloads["run_config"], "run config")
    audit = _load_json(payloads["audit"], "run audit")
    recomputed_audit = structural_audit.audit_run(
        run_root,
        prompts_path,
        paths["actions"],
        registration_actions_path=registration_path,
    )
    if audit != recomputed_audit:
        raise ValueError("supplied run audit differs from an in-lock recomputation")
    if audit.get("passed") is not True or audit.get("split_role") != "development":
        raise ValueError("a passing development run audit is required")
    if audit.get("audit_schema") != AUDIT_SCHEMA:
        raise ValueError("dedicated structural-control audit schema is required")
    dedicated_checks = {
        "structural_control_contract_passed": True,
        "image_decode_verified": True,
        "quality_results_inspected": False,
        "expected_task_count": EXPECTED_TASKS,
        "matched_unet_calls": "50x1",
        "scheduler": "EulerDiscreteScheduler",
        "runtime_activation_ledgers_verified": True,
        "duplicate_action_pngs_are_failure": False,
        "isolated_duplicate_action_pngs_are_failure": False,
        "full_action_collapse_is_failure": True,
        "duplicate_action_png_policy": "reject_action_pair_equal_in_all_99_blocks",
        "fully_collapsed_action_pairs": [],
    }
    for key, expected in dedicated_checks.items():
        if audit.get(key) != expected:
            raise ValueError(f"dedicated structural-control audit field {key!r} differs")
    if audit.get("warnings") != []:
        raise ValueError("dedicated structural-control audit must be warning-free")
    audit_provenance = _require_mapping(audit.get("provenance"), "audit.provenance")
    for audit_key, payload_key in (
        ("config_sha256", "run_config"),
        ("manifest_sha256", "manifest"),
        ("scores_sha256", "scores"),
        ("prompts_sha256", "prompts"),
        ("source_actions_sha256", "actions"),
        ("source_template_sha256", "registration"),
    ):
        _require_hash(audit_provenance, audit_key, hashes[payload_key], "run audit")
    if audit_provenance.get("input_snapshot_stable") is not True:
        raise ValueError("run audit did not establish a stable input snapshot")
    _require_hash(
        audit_provenance,
        "audit_script_sha256",
        sha256_file(AUDITOR_PATH),
        "run audit",
    )
    _require_hash(
        audit, "executable_actions_sha256", hashes["actions"], "dedicated run audit"
    )
    _require_hash(
        audit, "registration_sha256", hashes["registration"], "dedicated run audit"
    )
    if audit.get("analysis_implementation") != analysis_implementation:
        raise ValueError("dedicated run audit analysis implementation differs")
    if audit.get("analysis_implementation_sha256") != structural_audit.json_sha256(
        analysis_implementation
    ):
        raise ValueError("dedicated run audit analysis implementation hash differs")

    if run_config.get("structural_control_registered") is not True:
        raise ValueError("run config is not a registered structural-control run")
    smoke_scope = {
        key: run_config.get(key)
        for key in (
            "engineering_only",
            "formal_matrix_evidence",
            "quality_claim_allowed",
            "method_selection_allowed",
        )
        if key in run_config
    }
    if smoke_scope:
        raise ValueError(
            "formal evaluator rejects engineering-smoke evidence-scope artifacts"
        )
    if run_config.get("action_schema") != EXECUTABLE_SCHEMA:
        raise ValueError("run config action schema differs from structural-control v1")
    if run_config.get("structural_control_registration_schema") != REGISTRATION_SCHEMA:
        raise ValueError("run config registration schema differs from structural-control v1")
    if run_config.get("split_role") != "development":
        raise ValueError("run config is not the development split")
    generation_commit = str(run_config.get("git_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", generation_commit):
        raise ValueError("run config generation commit is invalid")
    if audit.get("generation_commit") != generation_commit:
        raise ValueError("dedicated audit generation commit differs from run config")
    if run_config.get("scorer_provenance_binding_required") is not True:
        raise ValueError("run config does not require scorer provenance binding")
    if Path(str(run_config.get("actions_yaml", ""))).resolve() != paths["actions"]:
        raise ValueError("run config actions path differs from evaluator input")
    if Path(str(run_config.get("prompts_csv", ""))).resolve() != prompts_path:
        raise ValueError("run config prompts path differs from registration")
    for key in ("actions_sha256", "structural_control_executable_actions_sha256"):
        _require_hash(run_config, key, hashes["actions"], "run config")
    _require_hash(run_config, "prompts_sha256", hashes["prompts"], "run config")
    _require_hash(
        run_config,
        "structural_control_source_template_sha256",
        hashes["registration"],
        "run config",
    )
    if run_config.get("structural_control_authorization") != authorization:
        raise ValueError("run config authorization differs from executable YAML")
    if run_config.get("structural_control_implementation_source") != actions_config.get(
        "implementation_source"
    ):
        raise ValueError("run config implementation source differs from executable YAML")
    if run_config.get(
        "structural_control_analysis_implementation"
    ) != analysis_implementation:
        raise ValueError("run config analysis implementation differs from executable YAML")
    if run_config.get("scoring") != actions_config.get("scoring"):
        raise ValueError("run config scoring contract differs from executable YAML")
    if run_config.get("structural_control_failure_policy") != actions_config.get(
        "failure_policy"
    ):
        raise ValueError("run config failure policy differs from executable YAML")
    if run_config.get("structural_control_source_template") != registration_source.get(
        "path"
    ):
        raise ValueError("run config source-template path differs from registration")

    split_seeds = _require_mapping(actions_config.get("split_seeds"), "split_seeds")
    seeds = tuple(
        int(value)
        for value in _require_list(split_seeds.get("development"), "development seeds")
    )
    if len(seeds) != EXPECTED_SEEDS or tuple(run_config.get("seeds", ())) != seeds:
        raise ValueError("run config seeds differ from registered development seeds")
    if audit.get("records") != EXPECTED_TASKS:
        raise ValueError("run audit does not report exactly 792 records")
    if audit.get("prompts") != EXPECTED_PROMPTS or audit.get("blocks") != (
        EXPECTED_PROMPTS * EXPECTED_SEEDS
    ):
        raise ValueError("run audit prompt/block counts differ from the registered grid")
    if tuple(audit.get("seeds", ())) != seeds:
        raise ValueError("run audit seeds differ from the registered grid")
    if tuple(audit.get("actions", ())) != protocol.action_order:
        raise ValueError("run audit action order differs from the registered grid")
    required_keys = audit.get("required_score_keys")
    if not isinstance(required_keys, list) or not set(REQUIRED_SCORE_METRICS).issubset(
        required_keys
    ):
        raise ValueError("run audit did not verify every evaluator score key")
    scoring = _require_mapping(actions_config.get("scoring"), "scoring")
    if scoring.get("required_schema") != SCORER_SCHEMA:
        raise ValueError("structural-control scorer schema differs")
    scorer_hash = str(scoring.get("registered_scorer_provenance_sha256", ""))
    if audit.get("scorer_provenance_schema") != SCORER_SCHEMA:
        raise ValueError("run audit did not establish hardened scorer provenance")
    if audit.get("scorer_provenance_sha256") != scorer_hash:
        raise ValueError("run audit scorer provenance differs from executable YAML")
    if audit.get("all_action_png_hashes_distinct_within_block") is not False:
        raise ValueError("run audit used an outcome-dependent distinct-image gate")
    if audit.get("scheduler_schedule_sha256") != actions_config.get(
        "scheduler_runtime", {}
    ).get("schedule_sha256"):
        raise ValueError("dedicated audit scheduler schedule differs from registration")
    device_identities = audit.get("device_identities")
    if not isinstance(device_identities, dict) or not device_identities:
        raise ValueError("dedicated audit lacks verified worker device identities")
    if set(device_identities) != set(audit.get("devices", [])):
        raise ValueError("dedicated audit worker identities differ from audited devices")

    manifest = _load_jsonl(payloads["manifest"], "manifest.jsonl")
    scores = _load_jsonl(payloads["scores"], "scores.jsonl")
    if len(manifest) != EXPECTED_TASKS or len(scores) != EXPECTED_TASKS:
        raise ValueError("manifest and scores must each contain exactly 792 tasks")
    try:
        prompts = pd.read_csv(io.BytesIO(payloads["prompts"]))
    except Exception as exc:
        raise ValueError("registered prompts CSV is invalid") from exc
    required_prompt_columns = {"index", "TEXT", "source_challenge", "split"}
    if not required_prompt_columns.issubset(prompts.columns):
        raise ValueError("registered prompts lack required development columns")
    if len(prompts) != EXPECTED_PROMPTS or prompts["index"].duplicated().any():
        raise ValueError("registered prompts do not contain 33 unique indices")
    if set(prompts["split"].astype(str)) != {"development"}:
        raise ValueError("registered prompts are not the development split")
    challenge_counts = prompts.groupby("source_challenge")["index"].nunique()
    if len(challenge_counts) != EXPECTED_CHALLENGES or set(
        challenge_counts.astype(int)
    ) != {3}:
        raise ValueError("registered prompts do not contain the frozen challenge grid")
    prompt_indices = tuple(int(value) for value in prompts["index"])
    challenge_by_prompt = {
        int(row["index"]): str(row["source_challenge"])
        for _, row in prompts.iterrows()
    }
    frame = merge_manifest_scores(pd.DataFrame(manifest), pd.DataFrame(scores))
    frame["source_challenge"] = frame["prompt_index"].map(challenge_by_prompt)
    if frame["source_challenge"].isna().any():
        raise ValueError("manifest contains a prompt absent from the registration")
    validate_expected_grid(frame, prompt_indices, seeds, protocol.action_order)
    validate_pairing(frame)
    resolved_paths = {key: str(path.resolve()) for key, path in paths.items()}
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
    """Exclude generation, scoring, and concurrent structural evaluation."""
    run_root = Path(run_dir).resolve()
    generation_handle = open(run_root / ".generate.lock", "a+")
    evaluation_handle = open(run_root / ".structural_control_evaluation.lock", "a+")
    try:
        try:
            fcntl.flock(generation_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(evaluation_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"generation, scoring, or structural evaluation is active for {run_root}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(evaluation_handle.fileno(), fcntl.LOCK_UN)
            fcntl.flock(generation_handle.fileno(), fcntl.LOCK_UN)
        finally:
            evaluation_handle.close()
            generation_handle.close()


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


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
    }


def _csv_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    frame = pd.DataFrame(rows).copy()
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (dict, list))).any():
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
    return frame.to_csv(index=False).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    json_path = run_dir / OUTPUT_JSON
    csv_path = run_dir / OUTPUT_CSV
    with evaluation_lock(run_dir):
        if json_path.exists() or csv_path.exists():
            raise ValueError("structural-control development evaluation is one-shot")
        inputs = load_verified_inputs(run_dir, args.actions, args.audit)
        registered_analysis_files = inputs.actions_config[
            "analysis_implementation"
        ]["files"]
        evaluator_git_commit = _require_committed_inputs(
            (
                inputs.paths["actions"],
                inputs.paths["registration"],
                inputs.paths["prompts"],
                inputs.paths["prompt_manifest"],
                *(ROOT / relative_path for relative_path in registered_analysis_files),
            )
        )
        result = evaluate_frame(
            inputs.frame,
            inputs.protocol,
            expected_prompt_indices=inputs.prompt_indices,
            expected_seeds=inputs.seeds,
            audit=inputs.audit,
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
            "registration_path": inputs.paths["registration"],
            "registration_sha256": inputs.hashes["registration"],
            "prompts_path": inputs.paths["prompts"],
            "prompts_sha256": inputs.hashes["prompts"],
            "prompt_manifest_path": inputs.paths["prompt_manifest"],
            "prompt_manifest_sha256": inputs.hashes["prompt_manifest"],
            "dedicated_auditor_path": str(AUDITOR_PATH.resolve()),
            "dedicated_auditor_sha256": sha256_file(AUDITOR_PATH),
            "base_auditor_path": str(BASE_AUDITOR_PATH.resolve()),
            "base_auditor_sha256": sha256_file(BASE_AUDITOR_PATH),
            "compare_actions_path": str(COMPARE_ACTIONS_PATH.resolve()),
            "compare_actions_sha256": sha256_file(COMPARE_ACTIONS_PATH),
            "evaluator_script_sha256": sha256_file(__file__),
            "evaluator_git_commit": evaluator_git_commit,
            "analysis_implementation": inputs.actions_config[
                "analysis_implementation"
            ],
            "runtime_versions": _runtime_versions(),
        }
        json_payload = (
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _atomic_write(csv_path, _csv_payload(result["contrasts"]))
        _atomic_write(json_path, json_payload)
    print(
        json.dumps(
            {
                "scope": result["scope"],
                "method_selection_performed": False,
                "rl_authorized": False,
            },
            indent=2,
        )
    )
    print(f"evaluation -> {json_path}")


if __name__ == "__main__":
    main()
