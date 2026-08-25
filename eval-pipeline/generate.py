"""Generation stage of the RepLDM eval pipeline (env: `repldm`).

Generates an image for every (prompt x seed x guidance action) combination and
writes a lossless PNG + a per-image sidecar JSON manifest recording the *full*
guidance config. Scoring is a separate stage (score.py, env `repldm_eval`) that
reads these PNGs + manifests — see eval-pipeline/README.md.

Design notes (grounded in the actual pipeline, EXPERIMENT_PLAN §4/§13):
  * Stage 1 remains the default. High-resolution Stage 2 requires the explicit
    `--stage2` opt-in and uses task-seeded resampling noise for paired actions.
    Attention Guidance acts only in Stage 1 in both regimes.
  * `--scales` preserves the constant-scalar sweep. `--actions` accepts a YAML
    grid containing no-AG, legacy schedule, scalar, and three-band actions.
  * Attention Guidance schedules are indexed by t_index and are reentrant, so
    the same pipeline/controller can safely serve repeated rollouts.
  * Multi-GPU with one queue per device. Every action for a (prompt, seed)
    block stays on one device, including after an interrupted run. Complete
    PNG + JSON pairs are skipped.

Example:
  conda run -n diff_attn python eval-pipeline/generate.py \
      --devices 1 --prompts eval-pipeline/prompts/eval_v1.csv \
      --out_dir outputs/frequency_action_pilot \
      --actions eval-pipeline/configs/frequency_action_pilot.yaml \
      --seeds 0,42,123
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import glob
import hashlib
import json
import math
import os
import platform
import queue
import re
import subprocess
import sys
import time
import traceback

import pandas as pd
import torch
import torch.multiprocessing as tmp
import yaml
import diffusers
from diffusers import EulerAncestralDiscreteScheduler

# Keep optional classes lazy so legacy environments that predate UniPC still
# run non-scheduler actions and fail with a clear action-validation error only
# when a missing reference is requested.
DPMSolverMultistepScheduler = getattr(diffusers, "DPMSolverMultistepScheduler", None)
UniPCMultistepScheduler = getattr(diffusers, "UniPCMultistepScheduler", None)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AttentionGuidance import (
    ConstantGuidanceController,
    MOMENT_TANGENT_MODES,
    RESIDUAL_MODES,
    SEMANTIC_TRANSPORT_MODES,
    StructuralUNetBasisProvider,
    build_fixed_coefficient_renderer,
    FreeUSchedule,
    TrajectoryCorrectionConfig,
)
from AttentionGuidance.attention_baselines import installed_attention_baseline
from InferencePipelines import RepLDMSDXLPipeline
from s7_provenance import (
    PROVENANCE_SCHEMA,
    action_sha256,
    image_sha256,
    json_sha256,
    validate_design_rows,
    validate_run_contract,
    validate_sidecar,
)

DEFAULT_CACHE_DIR = "/mnt/miah204/bycao/RepLDM/pretrained_ckpts"
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_NEG = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
FRLA_SELECTION_SCHEMAS = {
    "frla_relational_actions_v1",
    "frla_relational_validation_v1",
}
TRAJECTORY_SCHEMAS = {
    "trajectory_correction_actions_v1",
    "trajectory_correction_validation_v1",
}

# Keep this map explicit: scheduler names come from experiment YAML and must
# never be resolved through eval/importlib. The kwargs are part of the action
# hash after load_actions normalizes them.
SCHEDULER_BASELINE_CLASSES = {
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
}
if DPMSolverMultistepScheduler is not None:
    SCHEDULER_BASELINE_CLASSES["DPMSolverMultistepScheduler"] = DPMSolverMultistepScheduler
if UniPCMultistepScheduler is not None:
    SCHEDULER_BASELINE_CLASSES["UniPCMultistepScheduler"] = UniPCMultistepScheduler
SCHEDULER_BASELINE_DEFAULT_KWARGS = {
    "EulerAncestralDiscreteScheduler": {},
    "DPMSolverMultistepScheduler": {
        "solver_order": 2,
        "algorithm_type": "dpmsolver++",
        "solver_type": "midpoint",
        "lower_order_final": True,
    },
    "UniPCMultistepScheduler": {
        "solver_order": 2,
        "predict_x0": True,
        "solver_type": "bh2",
        "lower_order_final": True,
    },
}
SCHEDULER_BASELINE_ALLOWED_KWARGS = {
    "EulerAncestralDiscreteScheduler": set(),
    "DPMSolverMultistepScheduler": {
        "solver_order",
        "algorithm_type",
        "solver_type",
        "lower_order_final",
    },
    "UniPCMultistepScheduler": {
        "solver_order",
        "predict_x0",
        "solver_type",
        "lower_order_final",
    },
}


def _require_int(value, *, action_id: str, field: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{action_id}: {field} must be integer {expected}")


def task_id(prompt_index: int, seed: int, action: dict, legacy_scale_id: bool = False) -> str:
    if legacy_scale_id:
        return f"p{prompt_index}_seed{seed}_s{action['scale']:.4f}"
    return f"p{prompt_index}_seed{seed}_a{action['id']}"


def build_tasks(prompts: pd.DataFrame, seeds, actions, legacy_scale_ids: bool = False) -> list:
    tasks = []
    for _, row in prompts.iterrows():
        pid = int(row["index"])
        for seed in seeds:
            for action in actions:
                task = {
                    "id": task_id(pid, seed, action, legacy_scale_ids),
                    "prompt_index": pid,
                    "prompt": str(row["TEXT"]),
                    "bucket": str(row.get("bucket", "")),
                    "seed": int(seed),
                    "action_id": action["id"],
                    "action_type": action["type"],
                    "action": action,
                    "action_sha256": action_sha256(action),
                }
                if "scale" in action:
                    task["scale"] = float(action["scale"])
                if "band_scales" in action:
                    task["band_scales"] = list(action["band_scales"])
                task["residual_mode"] = action.get("residual_mode", "raw")
                tasks.append(task)
    return tasks


def scale_actions(scales):
    return [
        {
            "id": f"scale_{scale:.4f}",
            "type": "none" if scale == 0 else "scalar",
            "scale": float(scale),
        }
        for scale in scales
    ]


def group_tasks_by_pair(tasks):
    """Keep every action for a (prompt, seed) pair on one worker/device."""
    groups = {}
    for task in tasks:
        key = (task["prompt_index"], task["seed"])
        groups.setdefault(key, []).append(task)
    return list(groups.values())


def task_is_complete(
    task: dict,
    img_dir: str,
    *,
    run_contract_sha256: str | None = None,
) -> bool:
    stem = os.path.join(img_dir, task["id"])
    if not (os.path.exists(stem + ".png") and os.path.exists(stem + ".json")):
        return False
    # Legacy, non-registered sweeps predate the S7 provenance contract.  Keep
    # their resume behavior unchanged; registered runs always pass a contract
    # hash and take the strict path below.
    if run_contract_sha256 is None:
        return True
    try:
        with open(stem + ".json") as handle:
            record = json.load(handle)
        validate_sidecar(
            record,
            os.path.dirname(img_dir),
            expected_task=task,
            expected_contract_sha256=run_contract_sha256,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def recorded_group_device(task_group: list, img_dir: str):
    """Return a block's recorded device and reject already-invalid pairing."""
    devices = set()
    for task in task_group:
        json_path = os.path.join(img_dir, task["id"] + ".json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path) as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read existing sidecar {json_path}: {exc}") from exc
        if record.get("device"):
            devices.add(str(record["device"]))

    if len(devices) > 1:
        key = (task_group[0]["prompt_index"], task_group[0]["seed"])
        raise ValueError(
            f"existing prompt/seed block {key} spans devices {sorted(devices)}; "
            "paired inference is invalid, so use a new output directory"
        )
    return next(iter(devices), None)


def assign_tasks_to_devices(
    tasks: list,
    devices: list,
    img_dir: str,
    *,
    run_contract_sha256: str | None = None,
) -> dict:
    """Assign pending tasks without changing block placement on resume.

    The fallback round-robin index is computed over all blocks, not only the
    unfinished subset. An existing sidecar takes precedence over that fallback.
    """
    assignments = {device: [] for device in devices}
    for group_index, task_group in enumerate(group_tasks_by_pair(tasks)):
        recorded_device = recorded_group_device(task_group, img_dir)
        ordered_group = sorted(
            task_group,
            key=lambda task: hashlib.sha256(
                (
                    f"action-order-v1:{task['prompt_index']}:{task['seed']}:"
                    f"{task['action_id']}"
                ).encode("utf-8")
            ).digest(),
        )
        for execution_rank, task in enumerate(ordered_group):
            task["execution_rank"] = execution_rank
        pending = [
            task
            for task in ordered_group
            if not task_is_complete(
                task, img_dir, run_contract_sha256=run_contract_sha256
            )
        ]
        if not pending:
            continue
        device = recorded_device or devices[group_index % len(devices)]
        if device not in assignments:
            key = (task_group[0]["prompt_index"], task_group[0]["seed"])
            raise ValueError(
                f"prompt/seed block {key} must resume on {device}, which is not in "
                f"--devices {','.join(devices)}"
            )
        assignments[device].extend(pending)
    return {device: pending for device, pending in assignments.items() if pending}


def load_actions(path: str, num_inference_steps: int):
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") == "latent_renderer_registration_v1":
        raise ValueError(
            "latent-renderer registration manifests are not generation action configs; "
            "run eval-pipeline/audit_latent_renderer.py first"
        )
    actions = config.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("action config must contain a non-empty 'actions' list")
    if config.get("schema") == "trajectory_correction_validation_v1":
        selected_action = str(config.get("selected_action", ""))
        if not selected_action:
            raise ValueError(
                "trajectory-correction validation requires selected_action frozen from development"
            )
        reference_ids = {
            str(action.get("id"))
            for action in actions
            if not bool(action.get("selection_eligible", True))
        }
        selected_ids = {"no_correction", selected_action} | reference_ids
        actions = [action for action in actions if str(action.get("id")) in selected_ids]
        if len(actions) < 2 or selected_action not in {
            str(action.get("id")) for action in actions
        }:
            raise ValueError(
                "selected_action must identify one registered non-baseline action"
            )

    seen = set()
    default_layer = str(
        config.get(
            "semantic_transport_layer",
            "up_blocks.0.attentions.0.transformer_blocks.0.attn1",
        )
    )
    default_topk = int(config.get("semantic_transport_topk", 16))
    if default_topk <= 0:
        raise ValueError("semantic_transport_topk must be positive")
    provider_defaults = config.get("latent_renderer_provider", {}) or {}
    if not isinstance(provider_defaults, dict):
        raise ValueError("latent_renderer_provider must be a mapping")
    normalized = []
    for raw in actions:
        action = dict(raw)
        action_id = str(action.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", action_id):
            raise ValueError(f"invalid action id {action_id!r}; use letters, digits, '.', '_' or '-'")
        if action_id in seen:
            raise ValueError(f"duplicate action id {action_id!r}")
        seen.add(action_id)
        action_type = action.get("type")
        if action_type not in {
            "none",
            "scalar",
            "legacy",
            "frequency_bands",
            "clean_transport",
            "attention_baseline",
            "latent_renderer_fixed",
            "freeu",
            "trajectory_correction",
            "scheduler_baseline",
        }:
            raise ValueError(f"unsupported action type {action_type!r} for {action_id}")

        if action_type == "none":
            action["scale"] = 0.0
        elif action_type in {"scalar", "legacy"}:
            action["scale"] = float(action["scale"])
            if action["scale"] < 0:
                raise ValueError(f"{action_id}: scale must be non-negative")
        elif action_type == "frequency_bands":
            action["band_scales"] = [float(value) for value in action["band_scales"]]
            if len(action["band_scales"]) != 3 or min(action["band_scales"]) < 0:
                raise ValueError(f"{action_id}: band_scales must be three non-negative values")
        elif action_type == "clean_transport":
            mode = str(action.get("transport_mode", ""))
            if mode not in SEMANTIC_TRANSPORT_MODES:
                raise ValueError(f"{action_id}: unsupported transport_mode {mode!r}")
            angle = float(action.get("angle", -1.0))
            if not math.isfinite(angle) or angle < 0:
                raise ValueError(f"{action_id}: angle must be finite and non-negative")
            topk = int(action.get("topk", default_topk))
            if topk <= 0:
                raise ValueError(f"{action_id}: topk must be positive")
            action["transport_mode"] = mode
            action["angle"] = angle
            action["topk"] = topk
            action["semantic_transport_layer"] = str(
                action.get("semantic_transport_layer", default_layer)
            )
            action["permutation_seed"] = int(action.get("permutation_seed", 1729))
            action["scale"] = 0.0
        elif action_type == "attention_baseline":
            baseline = str(action.get("attention_baseline", ""))
            if baseline not in {"pladis", "gag"}:
                raise ValueError(f"{action_id}: attention_baseline must be pladis or gag")
            alpha = float(action.get("alpha", 1.5))
            baseline_scale = float(action.get("baseline_scale", -1.0))
            if alpha != 1.5 or not math.isfinite(baseline_scale) or baseline_scale < 0:
                raise ValueError(
                    f"{action_id}: alpha must be 1.5 and baseline_scale non-negative"
                )
            action["attention_baseline"] = baseline
            action["alpha"] = alpha
            action["baseline_scale"] = baseline_scale
            action["eta"] = float(action.get("eta", 15.0))
            action["zeta"] = float(action.get("zeta", 0.0))
            if baseline == "gag" and (
                action["eta"] <= 0 or not 0 <= action["zeta"] <= 1
            ):
                raise ValueError(f"{action_id}: invalid GAG eta/zeta")
            action["scale"] = 0.0
        elif action_type == "latent_renderer_fixed":
            coefficients = [float(value) for value in action.get("coefficients", [])]
            if len(coefficients) != 6 or not all(math.isfinite(value) for value in coefficients):
                raise ValueError(
                    f"{action_id}: latent_renderer_fixed requires six finite coefficients"
                )
            coefficient_bound = float(
                action.get("coefficient_bound", provider_defaults.get("coefficient_bound", 1.0))
            )
            if not math.isfinite(coefficient_bound) or coefficient_bound <= 0:
                raise ValueError(f"{action_id}: coefficient_bound must be positive")
            if any(abs(value) >= coefficient_bound for value in coefficients):
                raise ValueError(
                    f"{action_id}: coefficients must be strictly inside coefficient_bound"
                )
            provider = dict(provider_defaults)
            provider.update(action.get("provider", {}) or {})
            semantic_mode = str(provider.get("semantic_mode", "reciprocal_semantic"))
            if semantic_mode not in {
                "clean_tfsa",
                "reciprocal_latent",
                "reciprocal_semantic",
                "reciprocal_semantic_permuted",
            }:
                raise ValueError(f"{action_id}: unsupported latent renderer semantic_mode")
            semantic_topk = int(provider.get("semantic_topk", 16))
            if semantic_topk <= 0:
                raise ValueError(f"{action_id}: semantic_topk must be positive")
            prompt_dim = int(provider.get("prompt_dim", 0))
            state_dim = int(provider.get("state_dim", 0))
            if prompt_dim != 0 or state_dim != 0:
                raise ValueError(
                    f"{action_id}: fixed renderer actions require prompt_dim=state_dim=0"
                )
            feature_block = str(provider.get("feature_block", "up_blocks.0"))
            semantic_layer = provider.get(
                "semantic_layer",
                "up_blocks.0.attentions.0.transformer_blocks.0.attn1",
            )
            action["coefficients"] = coefficients
            action["coefficient_bound"] = coefficient_bound
            action["max_update_ratio"] = float(
                action.get("max_update_ratio", 0.05)
            )
            if (
                not math.isfinite(action["max_update_ratio"])
                or action["max_update_ratio"] < 0
            ):
                raise ValueError(f"{action_id}: max_update_ratio must be non-negative")
            action["latent_renderer_provider"] = {
                "feature_block": feature_block,
                "semantic_layer": None if semantic_layer is None else str(semantic_layer),
                "semantic_mode": semantic_mode,
                "semantic_topk": semantic_topk,
                "permutation_seed": int(provider.get("permutation_seed", 1729)),
                "prompt_dim": prompt_dim,
                "state_dim": state_dim,
            }
            action["scale"] = 0.0
        elif action_type == "freeu":
            if "parameters" in action and "knots" in action:
                raise ValueError(f"{action_id}: provide either parameters or knots, not both")
            if "parameters" in action:
                knots = ((0.0, action["parameters"]), (1.0, action["parameters"]))
            else:
                raw_knots = action.get("knots")
                if raw_knots is None and isinstance(action.get("schedule"), dict):
                    raw_knots = action["schedule"].get("knots")
                if not isinstance(raw_knots, list):
                    raise ValueError(f"{action_id}: freeu requires parameters or a knots list")
                knots = []
                for raw_knot in raw_knots:
                    if not isinstance(raw_knot, dict):
                        raise ValueError(f"{action_id}: each FreeU knot must be a mapping")
                    knots.append((raw_knot.get("position"), raw_knot.get("parameters")))
            try:
                schedule = FreeUSchedule(knots)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{action_id}: invalid FreeU schedule: {exc}") from exc
            preserve_moments = action.get("preserve_moments", False)
            if not isinstance(preserve_moments, bool):
                raise ValueError(f"{action_id}: preserve_moments must be a boolean")
            action["freeu_schedule"] = schedule.to_record()
            action["freeu_preserve_moments"] = preserve_moments
            action.pop("parameters", None)
            action.pop("knots", None)
            action.pop("schedule", None)
            action.pop("preserve_moments", None)
            action["scale"] = 0.0
        elif action_type == "trajectory_correction":
            mix = float(action.get("mix", -1.0))
            if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
                raise ValueError(f"{action_id}: mix must be finite and in [0, 1]")
            noise_mode = str(action.get("noise_mode", "sqrt"))
            if noise_mode not in {"sqrt", "linear", "none"}:
                raise ValueError(
                    f"{action_id}: noise_mode must be sqrt, linear, or none"
                )
            max_ratio = action.get("max_correction_ratio")
            if max_ratio is not None:
                max_ratio = float(max_ratio)
                if not math.isfinite(max_ratio) or max_ratio < 0:
                    raise ValueError(
                        f"{action_id}: max_correction_ratio must be finite and non-negative"
                    )
            action["mix"] = mix
            action["noise_mode"] = noise_mode
            action["max_correction_ratio"] = max_ratio
            action["scale"] = 0.0
        elif action_type == "scheduler_baseline":
            scheduler_class = str(
                action.get("scheduler_class", "EulerAncestralDiscreteScheduler")
            )
            if scheduler_class not in SCHEDULER_BASELINE_CLASSES:
                raise ValueError(
                    f"{action_id}: unsupported scheduler baseline {scheduler_class!r}; "
                    f"choose one of {sorted(SCHEDULER_BASELINE_CLASSES)}"
                )
            supplied_kwargs = action.get("scheduler_kwargs", {}) or {}
            if not isinstance(supplied_kwargs, dict):
                raise ValueError(f"{action_id}: scheduler_kwargs must be a mapping")
            unknown_kwargs = set(supplied_kwargs) - SCHEDULER_BASELINE_ALLOWED_KWARGS[
                scheduler_class
            ]
            if unknown_kwargs:
                raise ValueError(
                    f"{action_id}: unsupported {scheduler_class} kwargs: "
                    f"{sorted(unknown_kwargs)}"
                )
            scheduler_kwargs = dict(SCHEDULER_BASELINE_DEFAULT_KWARGS[scheduler_class])
            scheduler_kwargs.update(supplied_kwargs)
            if scheduler_class == "DPMSolverMultistepScheduler":
                _require_int(
                    scheduler_kwargs["solver_order"],
                    action_id=action_id,
                    field="DPM++ solver_order",
                    expected=2,
                )
                if not isinstance(scheduler_kwargs["algorithm_type"], str):
                    raise ValueError(f"{action_id}: DPM++ algorithm_type must be a string")
                if scheduler_kwargs["algorithm_type"] != "dpmsolver++":
                    raise ValueError(f"{action_id}: only deterministic dpmsolver++ is registered")
                if not isinstance(scheduler_kwargs["solver_type"], str):
                    raise ValueError(f"{action_id}: DPM++ solver_type must be a string")
                if scheduler_kwargs["solver_type"] not in {"midpoint", "heun"}:
                    raise ValueError(f"{action_id}: DPM++ solver_type must be midpoint or heun")
                if not isinstance(scheduler_kwargs["lower_order_final"], bool):
                    raise ValueError(f"{action_id}: lower_order_final must be boolean")
            elif scheduler_class == "UniPCMultistepScheduler":
                _require_int(
                    scheduler_kwargs["solver_order"],
                    action_id=action_id,
                    field="UniPC solver_order",
                    expected=2,
                )
                if not isinstance(scheduler_kwargs["predict_x0"], bool):
                    raise ValueError(f"{action_id}: predict_x0 must be boolean")
                if not isinstance(scheduler_kwargs["solver_type"], str):
                    raise ValueError(f"{action_id}: UniPC solver_type must be a string")
                if scheduler_kwargs["solver_type"] not in {"bh1", "bh2"}:
                    raise ValueError(f"{action_id}: UniPC solver_type must be bh1 or bh2")
                if not isinstance(scheduler_kwargs["lower_order_final"], bool):
                    raise ValueError(f"{action_id}: lower_order_final must be boolean")
            action["scheduler_class"] = scheduler_class
            action["scheduler_kwargs"] = scheduler_kwargs
            # Reference actions are reported but never selected by the fixed
            # correction gate unless explicitly marked eligible.
            action["selection_eligible"] = bool(action.get("selection_eligible", False))
            action["scale"] = 0.0

        residual_mode = str(action.get("residual_mode", "raw"))
        if residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"{action_id}: unsupported residual_mode {residual_mode!r}")
        if action_type != "scalar" and residual_mode != "raw":
            raise ValueError(f"{action_id}: only scalar actions support non-raw residual modes")
        action["residual_mode"] = residual_mode

        if "cfg_scale" in action:
            cfg_scale = float(action["cfg_scale"])
            if not math.isfinite(cfg_scale) or cfg_scale < 0:
                raise ValueError(f"{action_id}: cfg_scale must be finite and non-negative")
            action["cfg_scale"] = cfg_scale

        delay_steps = int(action.get("delay_steps", 0))
        if not 0 <= delay_steps < num_inference_steps:
            raise ValueError(f"{action_id}: delay_steps must be in [0, {num_inference_steps})")
        action["delay_steps"] = delay_steps
        if "decay" in action and action["decay"] is not None:
            if len(action["decay"]) != 3:
                raise ValueError(f"{action_id}: decay must have three values")
            action["decay"] = list(action["decay"])
        if "max_update_ratio" in action and action["max_update_ratio"] is not None:
            action["max_update_ratio"] = float(action["max_update_ratio"])
            if action["max_update_ratio"] < 0:
                raise ValueError(f"{action_id}: max_update_ratio must be non-negative")
            if residual_mode in MOMENT_TANGENT_MODES:
                raise ValueError(
                    f"{action_id}: max_update_ratio is not defined for moment-tangent updates"
                )
        normalized.append(action)

    cutoffs = [float(value) for value in config.get("frequency_band_cutoffs", [0.08, 0.25])]
    if len(cutoffs) != 2 or not 0 < cutoffs[0] < cutoffs[1] < 0.5:
        raise ValueError("frequency_band_cutoffs must satisfy 0 < low < mid < 0.5")
    return normalized, cutoffs


def validate_split_seed_role(path: str, split_role, seeds) -> None:
    """Enforce registered search/validation/test seeds when a YAML defines them."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    registered = config.get("split_seeds")
    if registered is None:
        if split_role is not None:
            raise ValueError("--split_role requires an action config with split_seeds")
        return
    if not isinstance(registered, dict) or not registered:
        raise ValueError("split_seeds must be a non-empty mapping")
    if split_role is None:
        raise ValueError(
            "this action config registers split_seeds; pass --split_role explicitly"
        )
    if split_role not in registered:
        raise ValueError(
            f"unknown --split_role {split_role!r}; expected one of {sorted(registered)}"
        )
    expected = [int(value) for value in registered[split_role]]
    if list(seeds) != expected:
        raise ValueError(
            f"--seeds {list(seeds)} do not match {split_role} registered seeds {expected}"
        )


def validate_scheduler_baseline_authorization(path: str) -> None:
    """Require an explicit human authorization for scheduler reference runs."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") != "scheduler_baselines_v1":
        return
    if config.get("status") != "authorized":
        raise ValueError(
            "scheduler_baselines_v1 is CPU-audited but not authorized; "
            "a reviewer must freeze status: authorized before generation"
        )
    authorization = config.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError(
            "authorized scheduler baseline config requires an authorization mapping"
        )
    if not str(authorization.get("reviewer", "")).strip():
        raise ValueError("authorized scheduler baseline config requires authorization.reviewer")
    if not str(authorization.get("reviewed_commit", "")).strip():
        raise ValueError(
            "authorized scheduler baseline config requires authorization.reviewed_commit"
        )


def validate_scheduler_baseline_design(
    path: str,
    *,
    prompts_path: str,
    actions: list,
    seeds: list,
    model_name: str,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    stage2_enabled: bool,
) -> None:
    """Reject sampling, provenance, and design drift in a frozen manifest."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") != "scheduler_baselines_v1":
        return
    sampling = config.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("scheduler_baselines_v1 requires a sampling mapping")
    expected_sampling = {
        "model": model_name,
        "base_scheduler": "EulerDiscreteScheduler",
        "resolution": int(resolution),
        "num_inference_steps": int(num_inference_steps),
        "cfg_scale": float(guidance_scale),
        "stage2": bool(stage2_enabled),
        "extra_unet_calls": 0,
        "initialization": "scheduler_native_init_sigma",
    }
    for key, observed in expected_sampling.items():
        registered = sampling.get(key)
        if isinstance(observed, float):
            same = isinstance(registered, (int, float)) and float(registered) == observed
        else:
            same = registered == observed
        if not same:
            raise ValueError(
                f"scheduler baseline sampling.{key} differs: "
                f"registered {registered!r}, observed {observed!r}"
            )
    source = config.get("source_manifest")
    if not isinstance(source, dict):
        raise ValueError("scheduler_baselines_v1 requires a source_manifest mapping")
    registered_prompts = source.get("prompts")
    if not isinstance(registered_prompts, str) or not registered_prompts:
        raise ValueError("scheduler baseline source_manifest.prompts is required")
    registered_abs = os.path.abspath(os.path.join(ROOT, registered_prompts))
    actual_abs = os.path.abspath(prompts_path)
    if registered_abs != actual_abs:
        raise ValueError(
            "scheduler baseline prompt path differs: "
            f"registered {registered_abs!r}, observed {actual_abs!r}"
        )
    registered_prompt_hash = source.get("prompts_sha256")
    if registered_prompt_hash != sha256_file(prompts_path):
        raise ValueError("scheduler baseline prompt SHA-256 differs from registration")
    design = config.get("design")
    if not isinstance(design, dict):
        raise ValueError("scheduler_baselines_v1 requires a design count mapping")
    prompts = pd.read_csv(prompts_path)
    observed = {
        "prompt_count": len(prompts),
        "action_count": len(actions),
        "seed_count": len(seeds),
        "task_count": len(prompts) * len(actions) * len(seeds),
    }
    expected = {
        "prompt_count": int(design.get("expected_prompt_count", -1)),
        "action_count": int(design.get("expected_action_count", -1)),
        "seed_count": int(design.get("expected_seed_count", -1)),
        "task_count": int(design.get("expected_task_count", -1)),
    }
    for key in ("prompt_count", "action_count", "seed_count", "task_count"):
        if observed[key] != expected[key]:
            raise ValueError(
                f"scheduler baseline design {key} differs: "
                f"registered {expected[key]}, observed {observed[key]}"
            )
    expected_action_ids = design.get("action_ids")
    observed_action_ids = [str(action.get("id")) for action in actions]
    if expected_action_ids != observed_action_ids:
        raise ValueError(
            "scheduler baseline action IDs differ: "
            f"registered {expected_action_ids!r}, observed {observed_action_ids!r}"
        )
    if any(bool(action.get("selection_eligible", True)) for action in actions):
        raise ValueError("scheduler baseline references must all be selection-ineligible")
    expected_types = {
        "no_correction": ("none", None, {}),
        "euler_ancestral_reference": (
            "scheduler_baseline",
            "EulerAncestralDiscreteScheduler",
            {},
        ),
        "dpmpp_2m_reference": (
            "scheduler_baseline",
            "DPMSolverMultistepScheduler",
            SCHEDULER_BASELINE_DEFAULT_KWARGS["DPMSolverMultistepScheduler"],
        ),
        "unipc_2_reference": (
            "scheduler_baseline",
            "UniPCMultistepScheduler",
            SCHEDULER_BASELINE_DEFAULT_KWARGS["UniPCMultistepScheduler"],
        ),
    }
    for action in actions:
        action_id = str(action.get("id"))
        expected_type, expected_class, expected_kwargs = expected_types.get(
            action_id, (None, None, None)
        )
        if action.get("type") != expected_type:
            raise ValueError(f"scheduler baseline action {action_id!r} has wrong type")
        if expected_class is not None and action.get("scheduler_class") != expected_class:
            raise ValueError(f"scheduler baseline action {action_id!r} has wrong class")
        if dict(action.get("scheduler_kwargs", {})) != dict(expected_kwargs):
            raise ValueError(f"scheduler baseline action {action_id!r} has wrong kwargs")
    expected_hashes = design.get("action_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("scheduler baseline design requires action_sha256 mapping")
    for action in actions:
        action_id = str(action.get("id"))
        expected_hash = expected_hashes.get(action_id)
        if expected_hash != action_sha256(action):
            raise ValueError(
                f"scheduler baseline action hash differs for {action_id!r}"
            )


def validate_registered_trajectory_design(
    path: str,
    *,
    prompts_path: str,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    stage2_enabled: bool,
    model_name: str | None = None,
    scheduler_name: str | None = None,
    extra_unet_calls: int | None = None,
) -> None:
    """Reject accidental protocol drift for the frozen S7 development gate."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") not in {
        "trajectory_correction_actions_v1",
        "trajectory_correction_validation_v1",
    }:
        return
    if config.get("schema") == "trajectory_correction_validation_v1" and not config.get(
        "selected_action"
    ):
        raise ValueError(
            "trajectory-correction validation requires selected_action frozen from development"
        )
    sampling = config.get("sampling") or {}
    if not isinstance(sampling, dict):
        raise ValueError("trajectory-correction sampling must be a mapping")
    registered_model = sampling.get("model")
    registered_scheduler = sampling.get("scheduler")
    registered_extra_calls = sampling.get("extra_unet_calls")
    if not isinstance(registered_model, str) or not registered_model:
        raise ValueError("trajectory-correction registration must declare sampling.model")
    if not isinstance(registered_scheduler, str) or not registered_scheduler:
        raise ValueError(
            "trajectory-correction registration must declare sampling.scheduler"
        )
    if isinstance(registered_extra_calls, bool) or not isinstance(
        registered_extra_calls, int
    ):
        raise ValueError(
            "trajectory-correction registration must declare integer extra_unet_calls"
        )
    if registered_extra_calls != 0:
        raise ValueError(
            "trajectory-correction actions require extra_unet_calls == 0"
        )
    if model_name is not None and str(model_name) != registered_model:
        raise ValueError(
            f"trajectory-correction model is registered as {registered_model!r}, "
            f"got {model_name!r}"
        )
    if scheduler_name is not None and str(scheduler_name) != registered_scheduler:
        raise ValueError(
            f"trajectory-correction scheduler is registered as {registered_scheduler!r}, "
            f"got {scheduler_name!r}"
        )
    if extra_unet_calls is not None and int(extra_unet_calls) != registered_extra_calls:
        raise ValueError(
            "trajectory-correction extra_unet_calls differs from registration"
        )
    registered_noise_mode = sampling.get("noise_mode")
    if registered_noise_mode is not None and str(registered_noise_mode) not in {
        "sqrt",
        "linear",
        "none",
    }:
        raise ValueError("trajectory-correction sampling.noise_mode is invalid")
    for action in config.get("actions", []):
        if action.get("type") != "trajectory_correction":
            continue
        mode = str(action.get("noise_mode", "sqrt"))
        if mode not in {"sqrt", "linear", "none"}:
            raise ValueError(
                f"trajectory-correction action {action.get('id')!r} has invalid noise_mode"
            )
    expected = {
        "resolution": int(sampling.get("resolution", resolution)),
        "num_inference_steps": int(
            sampling.get("num_inference_steps", num_inference_steps)
        ),
        "guidance_scale": float(sampling.get("cfg_scale", guidance_scale)),
        "stage2": sampling.get("stage2", stage2_enabled),
    }
    actual = {
        "resolution": int(resolution),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "stage2": bool(stage2_enabled),
    }
    if not isinstance(expected["stage2"], bool):
        raise ValueError("trajectory-correction registration sampling.stage2 must be boolean")
    if expected != actual:
        raise ValueError(
            "trajectory-correction sampling is registered as "
            f"{expected}, got {actual}"
        )
    source_manifest = config.get("source_manifest") or {}
    registered_prompts = source_manifest.get("prompts")
    if registered_prompts is not None:
        expected_path = os.path.abspath(
            os.path.join(ROOT, str(registered_prompts))
        )
        if os.path.abspath(prompts_path) != expected_path:
            raise ValueError(
                "trajectory-correction action config requires prompts file "
                f"{expected_path}"
            )
    expected_hash = source_manifest.get("prompts_sha256")
    if expected_hash is not None and expected_hash != sha256_file(prompts_path):
        raise ValueError("trajectory-correction prompt hash does not match registration")


def generation_contract(
    cfg: dict,
    *,
    actions: list,
    seeds: list[int],
    prompts_sha256: str,
    actions_sha256: str | None,
) -> dict:
    """Return the immutable fields that define a resumable generation run."""
    return {
        "schema": PROVENANCE_SCHEMA,
        "action_schema": cfg.get("action_schema"),
        "actions_sha256": actions_sha256,
        "actions": actions,
        "prompts_sha256": prompts_sha256,
        "seeds": [int(value) for value in seeds],
        "model_name": cfg["model_name"],
        "resolution": int(cfg["resolution"]),
        "num_inference_steps": int(cfg["num_inference_steps"]),
        "guidance_scale": float(cfg["guidance_scale"]),
        "negative_prompt": cfg["negative_prompt"],
        "power_calibrate": int(cfg["power_calibrate"]),
        "stage_name": cfg["stage_name"],
        "stage2_enabled": bool(cfg["stage2_enabled"]),
        "models_to_cpu": bool(cfg["models_to_cpu"]),
        "multi_encoder": bool(cfg["multi_encoder"]),
        "multi_decoder": bool(cfg["multi_decoder"]),
        "num_resample_timesteps": int(cfg["num_resample_timesteps"]),
        "init_rates": list(cfg["init_rates"]),
        "frequency_band_cutoffs": list(cfg["frequency_band_cutoffs"]),
        "split_role": cfg.get("split_role"),
        "git_commit": cfg.get("git_commit"),
        "runtime_provenance": cfg.get("runtime_provenance", {}),
    }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scheduler_config_sha256(scheduler) -> str:
    """Hash the legacy JSON-normalized scheduler config.

    Keep this byte-for-byte compatible with existing S7 sidecars. New
    scheduler-baseline records additionally use ``scheduler_config_sha256_v2``
    below, which canonicalizes diffusers bookkeeping.
    """
    config = getattr(scheduler, "config", {})
    payload = json.dumps(dict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scheduler_config_sha256_v2(scheduler) -> str:
    """Hash effective scheduler config with stable default bookkeeping."""
    config = dict(getattr(scheduler, "config", {}))
    if isinstance(config.get("_use_default_values"), list):
        config["_use_default_values"] = sorted(config["_use_default_values"])
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scheduler_baseline_runtime(action: dict, base_scheduler):
    """Construct one registered scheduler baseline from the base config."""
    if action.get("type") != "scheduler_baseline":
        return None
    base_name = type(base_scheduler).__name__
    if base_name != "EulerDiscreteScheduler":
        raise ValueError(
            "scheduler baselines require an EulerDiscreteScheduler base, "
            f"got {base_name}"
        )
    scheduler_class_name = str(action.get("scheduler_class", ""))
    scheduler_class = SCHEDULER_BASELINE_CLASSES.get(scheduler_class_name)
    if scheduler_class is None:
        raise ValueError(f"unsupported scheduler baseline {scheduler_class_name!r}")
    scheduler_kwargs = dict(action.get("scheduler_kwargs", {}))
    return scheduler_class.from_config(base_scheduler.config, **scheduler_kwargs)


def scheduler_provenance_record(
    action: dict,
    *,
    include: bool,
    base_config_sha256_v2: str,
    active_config_sha256_v2: str,
    order: int,
    solver_order,
    init_noise_sigma: float,
) -> dict:
    """Return the versioned scheduler ledger for registered reference runs."""
    if not include:
        return {}
    return {
        "scheduler_config_sha256_v2": base_config_sha256_v2,
        "active_scheduler_config_sha256_v2": active_config_sha256_v2,
        "scheduler_kwargs": dict(action.get("scheduler_kwargs", {})),
        "scheduler_order": int(order),
        "scheduler_solver_order": solver_order,
        "scheduler_init_noise_sigma": float(init_noise_sigma),
    }


def validate_final_test_authorization(
    authorization_path: str, actions_path: str, seeds
) -> None:
    """Require an independently hashed LR-1 validation/review authorization."""
    with open(authorization_path) as handle:
        authorization = json.load(handle)
    if authorization.get("schema") != "latent_renderer_final_authorization_v1":
        raise ValueError("invalid final-test authorization schema")
    if authorization.get("status") != "authorized_final_test":
        raise ValueError("final-test authorization is not active")
    expected_seeds = [int(value) for value in authorization.get("test_seeds", [])]
    if list(seeds) != expected_seeds:
        raise ValueError(
            f"--seeds {list(seeds)} do not match authorized final-test seeds {expected_seeds}"
        )
    source_hash = authorization.get("source_actions_sha256")
    if source_hash != sha256_file(actions_path):
        raise ValueError("final-test authorization hash does not match --actions")
    summary = authorization.get("review_summary", {})
    if summary.get("passed") is not True:
        raise ValueError("final-test authorization lacks a passing blinded review")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def runtime_provenance() -> dict:
    """Return package/runtime versions that can affect generated pixels."""
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "diffusers_version": str(getattr(diffusers, "__version__", "unknown")),
        "cuda_runtime_version": torch.version.cuda,
    }


def validate_model_cache(model_name: str, cache_dir: str) -> None:
    if os.path.isdir(model_name):
        if not os.path.isfile(os.path.join(model_name, "model_index.json")):
            raise FileNotFoundError(f"local model directory lacks model_index.json: {model_name}")
        return
    repo_dir = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
    snapshots = glob.glob(os.path.join(repo_dir, "snapshots", "*", "model_index.json"))
    if not snapshots:
        raise FileNotFoundError(
            f"no local snapshot for {model_name!r} under {cache_dir!r}; "
            f"expected {repo_dir}/snapshots/*/model_index.json"
        )


def generation_stage_settings(
    stage2_enabled: bool,
    resolution: int,
    low_vram: bool,
) -> dict:
    """Validate the requested stage and return recorded pipeline settings."""
    if resolution <= 0 or resolution % 8:
        raise ValueError("resolution must be a positive multiple of 8")
    if stage2_enabled and resolution <= 1024:
        raise ValueError("--stage2 requires --resolution greater than 1024")
    if not stage2_enabled and resolution > 1024:
        raise ValueError("resolution above 1024 requires the explicit --stage2 opt-in")

    init_rates = [0.8]
    if stage2_enabled and resolution >= 4096:
        init_rates = [0.9, 0.8]
    return {
        "stage2_enabled": stage2_enabled,
        "stage_name": (
            f"stage2_{resolution}" if stage2_enabled else f"stage1_{resolution}"
        ),
        "models_to_cpu": bool(low_vram or stage2_enabled),
        "multi_encoder": bool(stage2_enabled),
        "multi_decoder": bool(stage2_enabled and resolution > 2048),
        "num_resample_timesteps": 50,
        "init_rates": init_rates,
        "stage2_noise_source": "task_generator" if stage2_enabled else None,
    }


def guidance_runtime(action: dict, num_inference_steps: int):
    """Translate a validated action record into pipeline guidance arguments."""
    action_type = action["type"]
    residual_mode = action.get("residual_mode", "raw")
    controller = None
    attn_scale = float(action.get("scale", 0.0))
    attn_density = "all"
    attn_decay = None
    if action_type == "legacy":
        delay_steps = action.get("delay_steps", 0)
        attn_density = tuple(
            [1] * (num_inference_steps - delay_steps) + [0] * delay_steps
        )
        attn_decay = tuple(action["decay"]) if action.get("decay") else None
    elif action_type == "frequency_bands":
        controller = ConstantGuidanceController(
            band_scales=tuple(action["band_scales"]),
            max_update_ratio=action.get("max_update_ratio"),
        )
    elif action_type == "scalar" and (
        residual_mode != "raw" or action.get("max_update_ratio") is not None
    ):
        controller = ConstantGuidanceController(
            scale=attn_scale,
            max_update_ratio=action.get("max_update_ratio"),
            residual_mode=residual_mode,
        )
        attn_scale = 0.0
    return controller, attn_scale, attn_density, attn_decay


def clean_transport_runtime(action: dict):
    """Translate a validated clean-transport action for the pipeline."""
    if action["type"] != "clean_transport":
        return None
    return {
        "mode": action["transport_mode"],
        "angle": action["angle"],
        "topk": action["topk"],
        "layer_name": action["semantic_transport_layer"],
        "permutation_seed": action["permutation_seed"],
    }


def attention_baseline_runtime(action: dict):
    """Translate a validated PLADIS/GAG action for the worker context."""
    if action["type"] != "attention_baseline":
        return None
    return {
        "kind": action["attention_baseline"],
        "attention_scale": action["baseline_scale"],
        "alpha": action["alpha"],
        "eta": action["eta"],
        "zeta": action["zeta"],
    }


def freeu_runtime(action: dict):
    """Translate a normalized FreeU action into a re-entrant schedule."""
    if action["type"] != "freeu":
        return None
    record = action.get("freeu_schedule")
    if not isinstance(record, dict) or not isinstance(record.get("knots"), list):
        raise ValueError("normalized FreeU action lacks a knots record")
    return FreeUSchedule(
        (item["position"], item["parameters"]) for item in record["knots"]
    )


def trajectory_correction_runtime(action: dict):
    """Translate a normalized trajectory-correction action."""
    if action["type"] != "trajectory_correction":
        return None
    return TrajectoryCorrectionConfig(
        mix=action["mix"],
        noise_mode=action["noise_mode"],
        max_correction_ratio=action.get("max_correction_ratio"),
    )


def latent_renderer_runtime(action: dict, pipe, device: str, guidance_scale: float):
    """Construct one fixed LR-1 renderer/provider pair for a worker device."""
    if action["type"] != "latent_renderer_fixed":
        return None, None
    renderer = build_fixed_coefficient_renderer(
        action["coefficients"],
        latent_channels=4,
        coefficient_bound=action["coefficient_bound"],
        max_update_ratio=action["max_update_ratio"],
        preserve_moments=True,
    ).to(device)
    provider_config = action["latent_renderer_provider"]
    provider = StructuralUNetBasisProvider(
        pipe.unet,
        batch_size=1,
        do_classifier_free_guidance=float(
            action.get("cfg_scale", guidance_scale)
        ) > 1.0,
        latent_channels=4,
        semantic_mode=provider_config["semantic_mode"],
        semantic_topk=provider_config["semantic_topk"],
        semantic_layer=provider_config["semantic_layer"],
        feature_block=provider_config["feature_block"],
        permutation_seed=provider_config["permutation_seed"],
        prompt_dim=provider_config["prompt_dim"],
        state_dim=provider_config["state_dim"],
    )
    renderer.eval()
    return renderer, provider


def worker_process(cfg: dict, device: str, task_queue, error_queue):
    torch.cuda.set_device(device)
    pipe = RepLDMSDXLPipeline.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float16, variant="fp16",
        cache_dir=cfg["cache_dir"], local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    base_scheduler = pipe.scheduler
    base_scheduler_name = type(base_scheduler).__name__
    base_scheduler_hash = scheduler_config_sha256(base_scheduler)
    base_scheduler_hash_v2 = scheduler_config_sha256_v2(base_scheduler)
    registered_sampling = cfg.get("registered_sampling") or {}
    if cfg.get("trajectory_registered"):
        expected_scheduler = str(registered_sampling.get("scheduler", ""))
        if base_scheduler_name != expected_scheduler:
            raise RuntimeError(
                "trajectory-correction run loaded the wrong scheduler: "
                f"registered {expected_scheduler!r}, got {base_scheduler_name!r}"
            )
        if str(cfg.get("model_name")) != str(registered_sampling.get("model")):
            raise RuntimeError("trajectory-correction model differs from registration")
        if int(registered_sampling.get("extra_unet_calls", -1)) != 0:
            raise RuntimeError("trajectory-correction requires zero extra U-Net calls")
    if cfg.get("scheduler_baseline_registered"):
        expected_scheduler = str(registered_sampling.get("base_scheduler", ""))
        if base_scheduler_name != expected_scheduler:
            raise RuntimeError(
                "scheduler-baseline run loaded the wrong base scheduler: "
                f"registered {expected_scheduler!r}, got {base_scheduler_name!r}"
            )
        if str(cfg.get("model_name")) != str(registered_sampling.get("model")):
            raise RuntimeError("scheduler-baseline model differs from registration")
        if int(registered_sampling.get("extra_unet_calls", -1)) != 0:
            raise RuntimeError("scheduler-baseline requires zero extra U-Net calls")

    img_dir = os.path.join(cfg["out_dir"], "images")
    commit = cfg["git_commit"]
    latent_runtime_cache = {}
    n_done = 0
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break
        if task is None:
            break
        png_path = os.path.join(img_dir, task["id"] + ".png")
        json_path = os.path.join(img_dir, task["id"] + ".json")
        if os.path.exists(png_path) and os.path.exists(json_path):
            continue
        try:
            generator = torch.Generator(device).manual_seed(task["seed"])
            action = task["action"]
            controller, attn_scale, attn_density, attn_decay = guidance_runtime(
                action, cfg["num_inference_steps"]
            )
            clean_config = clean_transport_runtime(action)
            baseline_config = attention_baseline_runtime(action)
            freeu_schedule = freeu_runtime(action)
            trajectory_correction = trajectory_correction_runtime(action)
            # The diffusers FreeU implementation stores mutable attributes on
            # every up block.  Clear them before every action to preserve exact
            # paired comparisons, including after a failed task.
            pipe.unet.disable_freeu()
            if cfg["stage2_enabled"] and action["type"] == "latent_renderer_fixed":
                raise ValueError("latent renderer is registered for Stage 1 only")
            if action["type"] == "latent_renderer_fixed":
                latent_renderer, latent_provider = latent_runtime_cache.get(action["id"], (None, None))
                if latent_renderer is None:
                    latent_renderer, latent_provider = latent_renderer_runtime(
                        action, pipe, device, cfg["guidance_scale"]
                    )
                    latent_runtime_cache[action["id"]] = (latent_renderer, latent_provider)
            else:
                latent_renderer, latent_provider = None, None
            if cfg["stage2_enabled"] and clean_config is not None:
                raise ValueError("clean transport is registered for Stage 1 only")
            cfg_scale = float(action.get("cfg_scale", cfg["guidance_scale"]))
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start_time = time.perf_counter()
            baseline_context = (
                installed_attention_baseline(pipe.unet, **baseline_config)
                if baseline_config is not None
                else nullcontext()
            )
            scheduler_reference = action["type"] == "scheduler_baseline"
            scheduler_name = base_scheduler_name
            active_scheduler_hash = base_scheduler_hash
            active_scheduler_hash_v2 = base_scheduler_hash_v2
            active_scheduler_order = int(getattr(base_scheduler, "order", 1))
            active_solver_order = base_scheduler.config.get("solver_order")
            active_init_noise_sigma = float(base_scheduler.init_noise_sigma)
            if scheduler_reference:
                pipe.scheduler = scheduler_baseline_runtime(action, base_scheduler)
                scheduler_name = type(pipe.scheduler).__name__
                active_scheduler_hash = scheduler_config_sha256(pipe.scheduler)
                active_scheduler_hash_v2 = scheduler_config_sha256_v2(pipe.scheduler)
                active_scheduler_order = int(getattr(pipe.scheduler, "order", 1))
                active_solver_order = pipe.scheduler.config.get("solver_order")
                active_init_noise_sigma = float(pipe.scheduler.init_noise_sigma)
            try:
                with baseline_context:
                    images = pipe(
                        task["prompt"],
                        negative_prompt=cfg["negative_prompt"],
                        generator=generator,
                        height=cfg["resolution"], width=cfg["resolution"],
                        num_inference_steps=cfg["num_inference_steps"],
                        guidance_scale=cfg_scale,
                        show_image=False,
                        multi_decoder=cfg["multi_decoder"],
                        multi_encoder=cfg["multi_encoder"],
                        models_to_cpu=cfg["models_to_cpu"],
                        num_resample_timesteps=cfg["num_resample_timesteps"],
                        init_rates=cfg["init_rates"],
                        attn_type="vanilla",
                        attn_guidance_scale=attn_scale,
                        attn_guidance_density=attn_density,
                        attn_guidance_decay=attn_decay,
                        power_calibrate=cfg["power_calibrate"],
                        attn_guidance_filter=None,
                        attn_guidance_controller=controller,
                        attn_guidance_band_cutoffs=tuple(cfg["frequency_band_cutoffs"]),
                        semantic_transport_config=clean_config,
                        latent_renderer=latent_renderer,
                        latent_renderer_basis_provider=latent_provider,
                        freeu_schedule=freeu_schedule,
                        freeu_preserve_moments=bool(action.get("freeu_preserve_moments", False)),
                        trajectory_correction=trajectory_correction,
                    )
            finally:
                if scheduler_reference:
                    pipe.scheduler = base_scheduler
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                peak_memory = int(torch.cuda.max_memory_allocated(device))
            else:
                peak_memory = None
            elapsed = time.perf_counter() - start_time
            observed_unet_calls = [
                int(value)
                for value in getattr(pipe, "_last_unet_calls_per_step", [])
            ]
            if trajectory_correction is not None:
                if len(observed_unet_calls) != cfg["num_inference_steps"] or any(
                    value != 1 for value in observed_unet_calls
                ):
                    raise RuntimeError(
                        "trajectory correction requires exactly one U-Net call per Stage-1 step"
                    )
            if cfg.get("scheduler_baseline_registered"):
                if len(observed_unet_calls) != cfg["num_inference_steps"] or any(
                    value != 1 for value in observed_unet_calls
                ):
                    raise RuntimeError(
                        "scheduler baselines require exactly one U-Net call per step"
                    )
            observed_extra_unet_calls = max(
                (value - 1 for value in observed_unet_calls), default=0
            )
            diagnostics = getattr(pipe, "_last_guidance_diagnostics", None)
            renderer_diagnostics = getattr(
                pipe, "_last_latent_renderer_diagnostics", None
            )
            renderer_provider_diagnostics = getattr(
                pipe, "_last_latent_renderer_provider_diagnostics", None
            )
            images[-1].save(png_path)  # lossless PNG
            generated_image_sha256 = image_sha256(png_path)
            scheduler_reference_provenance = scheduler_provenance_record(
                action,
                include=bool(
                    scheduler_reference or cfg.get("scheduler_baseline_registered")
                ),
                base_config_sha256_v2=base_scheduler_hash_v2,
                active_config_sha256_v2=active_scheduler_hash_v2,
                order=active_scheduler_order,
                solver_order=active_solver_order,
                init_noise_sigma=active_init_noise_sigma,
            )
            record = {
                **task,
                "provenance_schema": PROVENANCE_SCHEMA,
                "run_contract_sha256": cfg["run_contract_sha256"],
                "config_sha256": cfg["run_contract_sha256"],
                "image_sha256": generated_image_sha256,
                "image_path": os.path.relpath(png_path, cfg["out_dir"]),
                "height": cfg["resolution"], "width": cfg["resolution"],
                "num_inference_steps": cfg["num_inference_steps"],
                "guidance_scale": cfg_scale,
                "inference_seconds": elapsed,
                "peak_gpu_memory_bytes": peak_memory,
                "power_calibrate": cfg["power_calibrate"],
                "attn_guidance_density": attn_density,
                "attn_guidance_decay": attn_decay,
                "frequency_band_cutoffs": cfg["frequency_band_cutoffs"],
                "stage": cfg["stage_name"],
                "stage2_enabled": cfg["stage2_enabled"],
                "models_to_cpu": cfg["models_to_cpu"],
                "multi_encoder": cfg["multi_encoder"],
                "multi_decoder": cfg["multi_decoder"],
                "num_resample_timesteps": cfg["num_resample_timesteps"],
                "init_rates": cfg["init_rates"],
                "stage2_noise_source": cfg["stage2_noise_source"],
                "model_name": cfg["model_name"],
                "scheduler_name": scheduler_name,
                "base_scheduler_name": base_scheduler_name,
                "scheduler_config_sha256": base_scheduler_hash,
                "active_scheduler_config_sha256": active_scheduler_hash,
                "scheduler_reference": scheduler_reference,
                "git_commit": commit,
                "registered_sampling": registered_sampling or None,
                "extra_unet_calls": observed_extra_unet_calls,
                "unet_calls_per_step": observed_unet_calls,
                **cfg["runtime_provenance"],
                "device": device,
                "latent_renderer_diagnostics": (
                    renderer_diagnostics.to_record()
                    if renderer_diagnostics is not None
                    and hasattr(renderer_diagnostics, "to_record")
                    else None
                ),
                "latent_renderer_provider_diagnostics": renderer_provider_diagnostics,
                "freeu_schedule": getattr(pipe, "_last_freeu_schedule", None),
                "freeu_preserve_moments": getattr(pipe, "_last_freeu_preserve_moments", False),
                "trajectory_correction": getattr(
                    pipe, "_last_trajectory_correction", None
                ),
                "trajectory_correction_diagnostics": getattr(
                    pipe, "_last_trajectory_correction_diagnostics", None
                ),
                **scheduler_reference_provenance,
            }
            if diagnostics:
                record.update(diagnostics)
            else:
                record.update(
                    {
                        "semantic_transport_mode": None,
                        "semantic_transport_layer": None,
                        "semantic_transport_angle": None,
                        "semantic_transport_topk": None,
                        "semantic_transport_steps": 0,
                        "mean_normalized_affinity_entropy": None,
                        "mean_transport_confidence": None,
                        "mean_transport_scheduler_update_norm_ratio": None,
                        "mean_transport_update_norm": None,
                        "mean_scheduler_update_norm": None,
                    }
                )
            with open(json_path, "w") as f:
                json.dump(record, f)
            del images
            n_done += 1
        except Exception:
            error = traceback.format_exc()
            error_queue.put((task["id"], error))
            print(f"[{device}] FAILED task {task['id']}:\n{error}", flush=True)
    print(f"[{device}] done, generated {n_done} images", flush=True)


def consolidate_manifest(
    out_dir: str,
    expected_ids=None,
    *,
    expected_tasks=None,
    run_contract_sha256: str | None = None,
    strict: bool = False,
):
    img_dir = os.path.join(out_dir, "images")
    rows = []
    expected_task_map = {
        str(task["id"]): task for task in (expected_tasks or [])
    }
    expected_set = set(expected_ids or expected_task_map)
    observed_set = set()
    for fn in sorted(os.listdir(img_dir)):
        if fn.endswith(".json"):
            stem = fn[:-5]
            if expected_set and stem not in expected_set:
                if strict:
                    raise ValueError(f"unexpected sidecar for unregistered task: {fn}")
                continue
            with open(os.path.join(img_dir, fn)) as f:
                record = json.load(f)
            if record.get("id") != stem:
                raise ValueError(f"sidecar id does not match filename: {fn}")
            if run_contract_sha256 is not None:
                validate_sidecar(
                    record,
                    out_dir,
                    expected_task=expected_task_map.get(stem),
                    expected_contract_sha256=run_contract_sha256,
                )
            rows.append(record)
            observed_set.add(stem)
    if strict and expected_set and observed_set != expected_set:
        raise ValueError(
            f"incomplete manifest: {len(expected_set - observed_set)} missing sidecars"
        )
    if expected_tasks and run_contract_sha256 is not None:
        validate_design_rows(
            rows,
            expected_action_ids=sorted({task["action_id"] for task in expected_tasks}),
            expected_seeds=sorted({int(task["seed"]) for task in expected_tasks}),
        )
    with open(os.path.join(out_dir, "manifest.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="CSV with columns index,TEXT,bucket")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--devices", default="0", help="comma-separated GPU indices, e.g. 6,7")
    ap.add_argument("--scales", default=None,
                    help="comma-separated constant attn_guidance_scale values")
    ap.add_argument("--actions", default=None,
                    help="YAML action grid; mutually exclusive with --scales")
    ap.add_argument(
        "--split_role",
        default=None,
        help="registered split_seeds role, e.g. train_search or validation_confirmation",
    )
    ap.add_argument(
        "--authorization",
        default=None,
        help="final-test authorization JSON required for latent-renderer test_final runs",
    )
    ap.add_argument("--seeds", default="0,42,123", help="comma-separated seeds")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument(
        "--stage2",
        action="store_true",
        help="explicitly enable RepLDM high-resolution resampling for resolution > 1024",
    )
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--power_calibrate", type=int, default=0)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEG)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--low_vram", action="store_true",
                    help="offload models to CPU between phases (models_to_cpu=True); use on busy GPUs")
    args = ap.parse_args()

    try:
        stage_settings = generation_stage_settings(
            args.stage2, args.resolution, args.low_vram
        )
    except ValueError as exc:
        ap.error(str(exc))
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    if len(seeds) != len(set(seeds)):
        ap.error("--seeds must not contain duplicates")
    if args.actions and args.scales:
        ap.error("--actions and --scales are mutually exclusive")
    action_config = {}
    if args.actions:
        actions, band_cutoffs = load_actions(args.actions, args.num_inference_steps)
        try:
            with open(args.actions) as action_handle:
                action_config = yaml.safe_load(action_handle) or {}
            validate_scheduler_baseline_authorization(args.actions)
            validate_split_seed_role(args.actions, args.split_role, seeds)
            validate_scheduler_baseline_design(
                args.actions,
                prompts_path=args.prompts,
                actions=actions,
                seeds=seeds,
                model_name=args.model_name,
                resolution=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                stage2_enabled=args.stage2,
            )
            validate_registered_trajectory_design(
                args.actions,
                prompts_path=args.prompts,
                resolution=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                stage2_enabled=args.stage2,
                model_name=args.model_name,
            )
            if (
                args.split_role == "test_final"
                and action_config.get("schema") == "latent_renderer_actions_v1"
            ):
                if not args.authorization:
                    raise ValueError(
                        "latent-renderer test_final requires --authorization from the validation gate"
                    )
                validate_final_test_authorization(args.authorization, args.actions, seeds)
        except ValueError as exc:
            ap.error(str(exc))
        legacy_scale_ids = False
    else:
        if args.split_role is not None:
            ap.error("--split_role requires --actions")
        scale_values = args.scales or "0,0.001,0.002,0.003,0.005"
        scales = [float(s) for s in scale_values.split(",") if s != ""]
        actions = scale_actions(scales)
        band_cutoffs = [0.08, 0.25]
        legacy_scale_ids = True
    devices = [f"cuda:{d.strip()}" for d in args.devices.split(",") if d.strip() != ""]
    assert devices, "no devices"
    if len(devices) != len(set(devices)):
        ap.error("--devices must not contain duplicates")

    prompts = pd.read_csv(args.prompts)
    assert "index" in prompts.columns and "TEXT" in prompts.columns, "prompts CSV needs index,TEXT[,bucket]"
    if prompts["index"].duplicated().any():
        ap.error("prompt indices must be unique")
    validate_model_cache(args.model_name, args.cache_dir)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    cfg = {
        "out_dir": args.out_dir, "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps, "guidance_scale": args.guidance_scale,
        "power_calibrate": args.power_calibrate, "negative_prompt": args.negative_prompt,
        "cache_dir": args.cache_dir, "model_name": args.model_name, "low_vram": args.low_vram,
        "frequency_band_cutoffs": band_cutoffs,
        "git_commit": git_commit(),
        "split_role": args.split_role,
        "prompts_csv": os.path.abspath(args.prompts),
        "prompts_sha256": sha256_file(args.prompts),
        "actions_yaml": os.path.abspath(args.actions) if args.actions else None,
        "actions_sha256": sha256_file(args.actions) if args.actions else None,
        "runtime_provenance": runtime_provenance(),
        **stage_settings,
    }
    action_schema = action_config.get("schema") if args.actions else None
    trajectory_registered = action_schema in TRAJECTORY_SCHEMAS
    scheduler_baseline_registered = action_schema == "scheduler_baselines_v1"
    cfg["action_schema"] = action_schema
    cfg["trajectory_registered"] = trajectory_registered
    cfg["scheduler_baseline_registered"] = scheduler_baseline_registered
    cfg["registered_sampling"] = (
        dict(action_config.get("sampling") or {})
        if trajectory_registered or scheduler_baseline_registered
        else {}
    )
    cfg["run_contract"] = generation_contract(
        cfg,
        actions=actions,
        seeds=seeds,
        prompts_sha256=cfg["prompts_sha256"],
        actions_sha256=cfg["actions_sha256"],
    )
    cfg["run_contract_sha256"] = json_sha256(cfg["run_contract"])
    # Registered S7 and scheduler-reference runs require strict sidecar/
    # contract validation on resume. Existing exploratory sweeps predate this
    # contract and must keep their original PNG+JSON completion semantics.
    registered_run = trajectory_registered or scheduler_baseline_registered
    resume_contract_sha256 = cfg["run_contract_sha256"] if registered_run else None

    tasks = build_tasks(prompts, seeds, actions, legacy_scale_ids)
    expected_ids = {task["id"] for task in tasks}
    img_dir = os.path.join(args.out_dir, "images")
    config_path = os.path.join(args.out_dir, "config.json")
    if registered_run and os.path.exists(config_path):
        try:
            with open(config_path) as handle:
                previous_config = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            ap.error(f"cannot read existing run config: {exc}")
        try:
            validate_run_contract(previous_config)
        except ValueError as exc:
            ap.error(f"existing registered run config is invalid: {exc}")
        if previous_config.get("run_contract_sha256") != cfg["run_contract_sha256"]:
            ap.error(
                "existing trajectory-correction run config differs from the requested "
                "model/scheduler/action/sampling contract; use a new output directory"
            )
    elif registered_run and not os.path.exists(config_path) and os.listdir(img_dir):
        ap.error(
            "registered artifacts exist without config.json; use a new output directory"
        )
    todo = [
        task
        for task in tasks
        if not task_is_complete(
            task, img_dir, run_contract_sha256=resume_contract_sha256
        )
    ]
    device_tasks = assign_tasks_to_devices(
        tasks,
        devices,
        img_dir,
        run_contract_sha256=resume_contract_sha256,
    )
    worker_count = len(device_tasks)
    print(f"{len(prompts)} prompts x {len(seeds)} seeds x {len(actions)} actions = {len(tasks)} tasks; "
          f"{len(tasks) - len(todo)} already done, {len(todo)} to generate on {worker_count} GPU(s).", flush=True)

    if todo:
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump({**cfg, "seeds": seeds, "actions": actions,
                       "devices": devices}, f, indent=2)
        tmp.set_start_method("spawn", force=True)
        manager = tmp.Manager()
        error_queue = manager.Queue()
        active_devices = list(device_tasks)
        task_queues = [manager.Queue() for _ in active_devices]
        for device, queue in zip(active_devices, task_queues):
            for task in device_tasks[device]:
                queue.put(task)
        procs = []
        for device, task_queue in zip(active_devices, task_queues):
            p = tmp.Process(target=worker_process, args=(cfg, device, task_queue, error_queue))
            p.start()
            procs.append((device, p))
        for _, p in procs:
            p.join()

        worker_failures = [(device, p.exitcode) for device, p in procs if p.exitcode != 0]
        task_failures = []
        while True:
            try:
                task_failures.append(error_queue.get_nowait())
            except Exception:
                break
        if worker_failures or task_failures:
            n = consolidate_manifest(
                args.out_dir,
                expected_ids,
                expected_tasks=tasks,
                run_contract_sha256=resume_contract_sha256,
                strict=trajectory_registered,
            )
            examples = ", ".join(task_id for task_id, _ in task_failures[:5])
            raise RuntimeError(
                f"generation failed: workers={worker_failures}, "
                f"task_failures={len(task_failures)} ({examples}); "
                f"preserved {n} completed records for resume"
            )

    n = consolidate_manifest(
        args.out_dir,
        expected_ids,
        expected_tasks=tasks,
        run_contract_sha256=resume_contract_sha256,
        strict=trajectory_registered,
    )
    print(f"manifest.jsonl written with {n} records -> {os.path.join(args.out_dir, 'manifest.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
