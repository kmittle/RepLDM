"""Strict post-run validator for the registered tuned-CFG development sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping

import pandas as pd
import yaml

from s7_provenance import (
    PROVENANCE_SCHEMA,
    action_sha256,
    json_sha256,
    sha256_file,
    validate_design_rows,
    validate_run_contract,
    validate_scores_against_manifest,
    validate_sidecar,
)


CFG_SCALES = (2.5, 5.0, 7.5, 10.0, 15.0)
CFG_ACTION_IDS = ("cfg_2p5", "cfg_5p0", "cfg_7p5", "cfg_10p0", "cfg_15p0")
BASELINE_ACTION_ID = "cfg_7p5"
CFG_PROMPTS = "eval-pipeline/prompts/s5_development.csv"
CFG_PROMPTS_SHA256 = (
    "cf9ae37f2c066e5a35712e2aaf6ea637d85281dc7cd2b104fbc07fe1d8e5201e"
)
CFG_SPLIT_SEEDS = {"development": [0, 42, 123]}
CFG_SOURCE_TEMPLATE = "eval-pipeline/configs/cfg_baselines_v1.yaml"
CFG_AUTH_SCOPE = "development_only_cfg_selection"
CFG_AUTH_FIELDS = {
    "reviewer",
    "reviewed_commit",
    "source_template",
    "source_template_sha256",
    "scope",
}
CFG_SAMPLING = {
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "model_revision": "462165984030d82259a11f4367a4eed129e94a7b",
    "base_scheduler": "EulerDiscreteScheduler",
    "resolution": 1024,
    "num_inference_steps": 50,
    "default_cfg_scale": 7.5,
    "guidance_rescale": 0.0,
    "negative_prompt": "blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
    "power_calibrate": 0,
    "stage2": False,
    "extra_unet_calls": 0,
    "initialization": "scheduler_native_init_sigma",
    "cfg_source": "action.cfg_scale",
    "cfg_pipeline_argument": "guidance_scale",
}
CFG_DESIGN_COUNTS = {
    "expected_prompt_count": 12,
    "expected_action_count": 5,
    "expected_seed_count": 3,
    "expected_block_count": 36,
    "expected_task_count": 180,
}
CFG_SCHEDULER_RUNTIME = {
    "config_sha256_v2": "6bf0509f22d8d3a06d6493c2291af8655f6f74846b2d2eae3cf71b5cda000102",
    "num_inference_steps": 50,
    "schedule_sha256": "302d2452f411bf3eea64f8dd3530e232b95c23aed7b818ed6697982a4428c144",
    "construction_init_noise_sigma": 14.648818969726562,
    "effective_init_noise_sigma": 13.158469200134277,
}
CFG_SCORING = {
    "metrics": ["pixel", "clip", "hps", "iqa"],
    "strict": True,
    "params": {
        "patch_crops": 5,
        "clip_model": "ViT-B/32",
        "clipscore_w": 2.5,
    },
    "required_score_keys": [
        "colorfulness",
        "laplacian_sharpness",
        "clipped_fraction",
        "mean_saturation",
        "contrast_std",
        "clip_cosine",
        "clipscore",
        "hpsv2",
        "topiq_nr",
    ],
}
EXECUTION_ORDER = {
    "version": "action-order-v1",
    "grouping": ["prompt_index", "seed"],
    "digest": "sha256",
    "input": "action-order-v1:{prompt_index}:{seed}:{action_id}",
    "sort": "ascending_raw_digest",
    "single_device_per_group": True,
}
SAMPLING_KEYS = set(CFG_SAMPLING)
SCORE_KEYS = tuple(CFG_SCORING["required_score_keys"])
SCORING_SCHEMA = "cfg_scoring_contract_v1"


def load_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def require_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def require_uniform(rows: Iterable[Mapping[str, Any]], key: str, label: str) -> Any:
    values: Dict[str, Any] = {}
    for row in rows:
        value = row.get(key)
        values[json.dumps(value, sort_keys=True, default=str)] = value
    if len(values) != 1:
        raise ValueError(f"{label} has {len(values)} distinct {key} values")
    return next(iter(values.values()))


def expected_execution_ranks(
    prompt_index: int, seed: int, action_ids: Iterable[str]
) -> dict[str, int]:
    """Recompute generate.py's deterministic block-wise action permutation."""
    ordered = sorted(
        (str(action_id) for action_id in action_ids),
        key=lambda action_id: hashlib.sha256(
            f"action-order-v1:{prompt_index}:{seed}:{action_id}".encode("utf-8")
        ).digest(),
    )
    return {action_id: rank for rank, action_id in enumerate(ordered)}


def scheduler_config_sha256_v2(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_output(repository_root: str, *args: str, text: bool = True):
    try:
        return subprocess.check_output(
            ["git", "-C", repository_root, *args],
            stderr=subprocess.PIPE,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise ValueError(
            f"cannot verify CFG authorization with Git: {str(detail).strip() or exc}"
        ) from exc


def validate_cfg_authorization(
    actions_config: Mapping[str, Any], repository_root: str
) -> None:
    """Verify that authorization refers to immutable template bytes in Git."""
    if actions_config.get("status") != "authorized":
        raise ValueError("CFG baseline config is not authorized")
    authorization = actions_config.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("CFG baseline authorization is incomplete")
    if set(authorization) != CFG_AUTH_FIELDS:
        raise ValueError("CFG authorization fields differ from frozen v1")
    if not str(authorization.get("reviewer", "")).strip():
        raise ValueError("CFG baseline authorization reviewer is missing")
    if authorization.get("source_template") != CFG_SOURCE_TEMPLATE:
        raise ValueError("CFG authorization source_template differs from frozen v1")
    if authorization.get("scope") != CFG_AUTH_SCOPE:
        raise ValueError("CFG authorization scope differs from frozen v1")

    reviewed_commit = str(authorization.get("reviewed_commit", ""))
    template_sha256 = str(authorization.get("source_template_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", reviewed_commit):
        raise ValueError("CFG authorization reviewed_commit must be full 40-hex")
    if not re.fullmatch(r"[0-9a-f]{64}", template_sha256):
        raise ValueError("CFG authorization source_template_sha256 must be 64-hex")

    head = _git_output(repository_root, "rev-parse", "HEAD").strip()
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            repository_root,
            "merge-base",
            "--is-ancestor",
            reviewed_commit,
            head,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ancestry.returncode == 1:
        raise ValueError("CFG authorization reviewed_commit is not an ancestor of HEAD")
    if ancestry.returncode != 0:
        raise ValueError(
            "cannot verify CFG authorization ancestry: "
            + (ancestry.stderr.strip() or "git merge-base failed")
        )

    template_bytes = _git_output(
        repository_root,
        "show",
        f"{reviewed_commit}:{CFG_SOURCE_TEMPLATE}",
        text=False,
    )
    if hashlib.sha256(template_bytes).hexdigest() != template_sha256:
        raise ValueError("CFG authorization template hash differs from reviewed Git bytes")
    try:
        reviewed_template = yaml.safe_load(template_bytes) or {}
    except yaml.YAMLError as exc:
        raise ValueError("reviewed CFG source template is invalid YAML") from exc
    if not isinstance(reviewed_template, dict):
        raise ValueError("reviewed CFG source template is not a mapping")

    def unsigned_body(value: Mapping[str, Any]) -> dict:
        return {
            key: item
            for key, item in value.items()
            if key not in {"status", "authorization"}
        }

    if unsigned_body(actions_config) != unsigned_body(reviewed_template):
        raise ValueError("authorized CFG config differs from its reviewed template")


def validate_frozen_registration(
    actions_config: Mapping[str, Any], prompts_path: str, repository_root: str
) -> None:
    """Reject a self-consistent replacement YAML that is not the frozen v1 study."""
    if actions_config.get("schema") != "cfg_baselines_v1":
        raise ValueError("actions YAML is not cfg_baselines_v1")
    if actions_config.get("experiment_id") != "cfg_baselines_development_v1":
        raise ValueError("CFG experiment_id differs from frozen v1")
    if (
        actions_config.get("parent_principle")
        != "isolate_scalar_classifier_free_guidance"
    ):
        raise ValueError("CFG parent principle differs from frozen v1")

    source = actions_config.get("source_manifest")
    if not isinstance(source, dict):
        raise ValueError("CFG registration lacks source_manifest")
    if source.get("prompts") != CFG_PROMPTS:
        raise ValueError("CFG prompt path differs from frozen v1")
    registered_prompt_path = os.path.join(repository_root, CFG_PROMPTS)
    if os.path.realpath(prompts_path) != os.path.realpath(registered_prompt_path):
        raise ValueError("prompt path differs from frozen v1")
    if source.get("prompts_sha256") != CFG_PROMPTS_SHA256:
        raise ValueError("CFG prompt hash differs from frozen v1")
    if sha256_file(prompts_path) != CFG_PROMPTS_SHA256:
        raise ValueError("prompt file bytes differ from frozen v1")
    if actions_config.get("split_seeds") != CFG_SPLIT_SEEDS:
        raise ValueError("CFG split seeds differ from frozen v1")
    if actions_config.get("sampling") != CFG_SAMPLING:
        raise ValueError("CFG sampling differs from frozen v1")
    if actions_config.get("scheduler_runtime") != CFG_SCHEDULER_RUNTIME:
        raise ValueError("CFG Euler runtime differs from frozen v1")
    if actions_config.get("scoring") != CFG_SCORING:
        raise ValueError("CFG scoring differs from frozen v1")
    if actions_config.get("execution_order") != EXECUTION_ORDER:
        raise ValueError("CFG execution order differs from frozen v1")

    design = actions_config.get("design")
    if not isinstance(design, dict):
        raise ValueError("CFG registration lacks design")
    for key, expected in CFG_DESIGN_COUNTS.items():
        if design.get(key) != expected:
            raise ValueError(f"CFG design.{key} differs from frozen v1")
    if design.get("action_ids") != list(CFG_ACTION_IDS):
        raise ValueError("CFG action IDs differ from frozen v1")
    if design.get("cfg_scales") != list(CFG_SCALES):
        raise ValueError("CFG scales differ from frozen v1")
    if design.get("baseline_action_id") != BASELINE_ACTION_ID:
        raise ValueError("CFG baseline action differs from frozen v1")


def _validate_registered_actions(actions_config: Mapping[str, Any]) -> tuple[list[str], dict]:
    design = actions_config.get("design") or {}
    source_actions = actions_config.get("actions") or []
    if not isinstance(source_actions, list) or any(
        not isinstance(action, dict) for action in source_actions
    ):
        raise ValueError("registered CFG actions must be a list of mappings")
    action_ids = [str(value) for value in design.get("action_ids", [])]
    if action_ids != list(CFG_ACTION_IDS):
        raise ValueError("registered CFG action IDs differ from the frozen scale grid")
    if action_ids != [str(action.get("id")) for action in source_actions]:
        raise ValueError("registered action order differs from design.action_ids")
    registered_scales = design.get("cfg_scales", [])
    if (
        not isinstance(registered_scales, list)
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in registered_scales
        )
        or [float(value) for value in registered_scales] != list(CFG_SCALES)
    ):
        raise ValueError("registered CFG scales differ from the frozen scale grid")
    if design.get("baseline_action_id") != BASELINE_ACTION_ID:
        raise ValueError("registered CFG baseline must be cfg_7p5")
    source_by_id = {str(action.get("id")): action for action in source_actions}
    for action_id, scale in zip(CFG_ACTION_IDS, CFG_SCALES):
        action = source_by_id[action_id]
        cfg_scale = action.get("cfg_scale")
        if (
            action.get("type") != "none"
            or isinstance(cfg_scale, bool)
            or not isinstance(cfg_scale, (int, float))
            or not math.isfinite(float(cfg_scale))
            or float(cfg_scale) != scale
        ):
            raise ValueError(f"registered action {action_id!r} has the wrong CFG scale")
        if set(action) - {"id", "type", "cfg_scale"}:
            raise ValueError(f"registered action {action_id!r} has unregistered fields")
    return action_ids, source_by_id


def validate(
    run_dir: str,
    actions_path: str,
    prompts_path: str,
    kind: str,
) -> dict:
    """Validate the frozen grid, Euler ledger, images, and optional scores."""
    run_dir = os.path.abspath(run_dir)
    with open(os.path.join(run_dir, "config.json")) as handle:
        run_config = json.load(handle)
    with open(actions_path) as handle:
        actions_config = yaml.safe_load(handle) or {}
    prompts = pd.read_csv(prompts_path)

    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    validate_cfg_authorization(actions_config, repository_root)
    validate_frozen_registration(actions_config, prompts_path, repository_root)
    if run_config.get("action_schema") != "cfg_baselines_v1":
        raise ValueError("run config has the wrong action schema")
    if run_config.get("cfg_baseline_registered") is not True:
        raise ValueError("run is not marked as a registered CFG baseline")
    if run_config.get("scheduler_baseline_registered") is not False:
        raise ValueError("CFG run must not be marked as a scheduler baseline")
    if run_config.get("trajectory_registered") is not False:
        raise ValueError("CFG run must not be a trajectory-selection run")
    if not re.fullmatch(r"[0-9a-f]{40}", str(run_config.get("git_commit", ""))):
        raise ValueError("CFG run git_commit must be a full 40-hex commit")
    if run_config.get("scheduler_runtime") != actions_config.get("scheduler_runtime"):
        raise ValueError("run Euler runtime differs from frozen v1")
    if os.path.realpath(str(run_config.get("actions_yaml", ""))) != os.path.realpath(
        actions_path
    ):
        raise ValueError("run actions path differs from the authorized YAML")
    if os.path.realpath(str(run_config.get("prompts_csv", ""))) != os.path.realpath(
        prompts_path
    ):
        raise ValueError("run prompt path differs from the registered CSV")
    if run_config.get("actions_sha256") != sha256_file(actions_path):
        raise ValueError("run actions hash differs from the authorized YAML")
    if run_config.get("prompts_sha256") != sha256_file(prompts_path):
        raise ValueError("run prompts hash differs from the registered CSV")
    if prompts["index"].duplicated().any():
        raise ValueError("prompt CSV contains duplicate indices")

    sampling = actions_config.get("sampling") or {}
    if set(sampling) != SAMPLING_KEYS:
        raise ValueError("registered CFG sampling fields differ from the frozen schema")
    for key in (
        "model",
        "model_revision",
        "base_scheduler",
        "negative_prompt",
        "initialization",
    ):
        if not isinstance(sampling.get(key), str) or not sampling[key]:
            raise ValueError(f"registered sampling.{key} must be a non-empty string")
    for key in ("resolution", "num_inference_steps", "power_calibrate", "extra_unet_calls"):
        value = sampling.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"registered sampling.{key} must be an integer")
    if sampling["resolution"] <= 0 or sampling["num_inference_steps"] <= 0:
        raise ValueError("registered resolution and step count must be positive")
    if not isinstance(sampling.get("stage2"), bool):
        raise ValueError("registered sampling.stage2 must be boolean")
    default_cfg = sampling.get("default_cfg_scale")
    if (
        isinstance(default_cfg, bool)
        or not isinstance(default_cfg, (int, float))
        or not math.isfinite(float(default_cfg))
    ):
        raise ValueError("registered sampling.default_cfg_scale must be finite numeric")
    guidance_rescale = sampling.get("guidance_rescale")
    if (
        isinstance(guidance_rescale, bool)
        or not isinstance(guidance_rescale, (int, float))
        or not math.isfinite(float(guidance_rescale))
    ):
        raise ValueError("registered sampling.guidance_rescale must be finite numeric")
    if run_config.get("registered_sampling") != sampling:
        raise ValueError("run registered_sampling differs from the authorized YAML")
    expected_sampling = {
        "model_name": sampling.get("model"),
        "model_revision": sampling.get("model_revision"),
        "resolution": int(sampling.get("resolution")),
        "num_inference_steps": int(sampling.get("num_inference_steps")),
        "guidance_scale": float(sampling.get("default_cfg_scale")),
        "guidance_rescale": float(sampling.get("guidance_rescale")),
        "stage2_enabled": bool(sampling.get("stage2")),
        "negative_prompt": sampling.get("negative_prompt"),
        "power_calibrate": int(sampling.get("power_calibrate")),
    }
    for key, expected in expected_sampling.items():
        observed = run_config.get(key)
        same = (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and float(observed) == expected
            if isinstance(expected, float)
            else observed == expected
        )
        if not same:
            raise ValueError(f"run field {key!r} differs from registration")
    if sampling.get("base_scheduler") != "EulerDiscreteScheduler":
        raise ValueError("registered CFG sweep requires EulerDiscreteScheduler")
    if sampling.get("extra_unet_calls") != 0:
        raise ValueError("registered CFG sweep must require zero extra U-Net calls")
    if sampling.get("initialization") != "scheduler_native_init_sigma":
        raise ValueError("registered CFG sweep has the wrong initialization")
    if sampling.get("cfg_source") != "action.cfg_scale":
        raise ValueError("registered CFG sweep has the wrong CFG source")
    if sampling.get("cfg_pipeline_argument") != "guidance_scale":
        raise ValueError("registered CFG sweep has the wrong pipeline argument")
    if actions_config.get("execution_order") != EXECUTION_ORDER:
        raise ValueError("registered CFG execution order differs from action-order-v1")

    split_role = run_config.get("split_role")
    if split_role != "development":
        raise ValueError("CFG run split_role must be development")
    expected_seeds = (actions_config.get("split_seeds") or {}).get(split_role)
    if expected_seeds is None or run_config.get("seeds") != expected_seeds:
        raise ValueError("run seeds differ from the registered split role")
    action_ids, source_by_id = _validate_registered_actions(actions_config)
    design = actions_config.get("design") or {}
    run_actions = run_config.get("actions") or []
    if [str(action.get("id")) for action in run_actions] != action_ids:
        raise ValueError("normalized run actions differ from registered order")
    expected_action_hashes = design.get("action_sha256") or {}
    if set(expected_action_hashes) != set(action_ids):
        raise ValueError("registered action hash mapping is incomplete")
    for action in run_actions:
        action_id = str(action.get("id"))
        if action_sha256(action) != expected_action_hashes[action_id]:
            raise ValueError(f"normalized action hash differs for {action_id!r}")

    contract_hash = validate_run_contract(run_config)
    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    manifest = load_jsonl(manifest_path)
    expected_prompts = [int(value) for value in prompts["index"].tolist()]
    for row in manifest:
        prompt_index = require_integer(
            row.get("prompt_index"), f"{row.get('id', '')}: prompt_index"
        )
        seed = require_integer(row.get("seed"), f"{row.get('id', '')}: seed")
        action_id = row.get("action_id")
        if not isinstance(action_id, str) or action_id not in CFG_ACTION_IDS:
            raise ValueError(f"{row.get('id', '')}: action_id differs from frozen v1")
        expected_id = f"p{prompt_index}_seed{seed}_a{action_id}"
        if row.get("id") != expected_id:
            raise ValueError(f"{row.get('id', '')}: sidecar id is not canonical")
    validate_design_rows(
        manifest,
        expected_action_ids=action_ids,
        expected_seeds=expected_seeds,
        expected_prompt_indices=expected_prompts,
    )
    expected_count = int(design.get("expected_task_count", -1))
    observed_counts = {
        "prompt_count": len(expected_prompts),
        "action_count": len(action_ids),
        "seed_count": len(expected_seeds),
        "task_count": len(manifest),
    }
    registered_counts = {
        "prompt_count": int(design.get("expected_prompt_count", -1)),
        "action_count": int(design.get("expected_action_count", -1)),
        "seed_count": int(design.get("expected_seed_count", -1)),
        "task_count": expected_count,
    }
    if observed_counts != registered_counts:
        raise ValueError("observed CFG design counts differ from registration")
    if expected_count != len(expected_prompts) * len(expected_seeds) * len(action_ids):
        raise ValueError("registered CFG design counts are internally inconsistent")

    prompt_by_index = {int(row["index"]): row for _, row in prompts.iterrows()}
    run_action_by_id = {str(action["id"]): action for action in run_actions}
    rows_by_pair: Dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in manifest:
        row_id = str(row.get("id", ""))
        if row.get("provenance_schema") != PROVENANCE_SCHEMA:
            raise ValueError(f"{row_id}: missing registered provenance schema")
        validate_sidecar(row, run_dir, expected_contract_sha256=contract_hash)
        prompt = prompt_by_index[int(row["prompt_index"])]
        if str(row.get("prompt")) != str(prompt["TEXT"]):
            raise ValueError(f"{row_id}: prompt text differs from registration")
        if "bucket" in prompts and str(row.get("bucket", "")) != str(
            prompt.get("bucket", "")
        ):
            raise ValueError(f"{row_id}: prompt bucket differs from registration")
        action_id = str(row.get("action_id"))
        source_action = source_by_id[action_id]
        run_action = run_action_by_id[action_id]
        if row.get("action_type") != "none" or source_action.get("type") != "none":
            raise ValueError(f"{row_id}: CFG action type differs from registration")
        if row.get("action") != run_action:
            raise ValueError(f"{row_id}: normalized action differs from run contract")
        if row.get("action_sha256") != expected_action_hashes[action_id]:
            raise ValueError(f"{row_id}: action hash differs from registration")
        scale = float(source_action["cfg_scale"])
        if require_finite(row.get("guidance_scale"), f"{row_id}: guidance scale") != scale:
            raise ValueError(f"{row_id}: effective CFG scale differs from registration")
        if row.get("registered_sampling") != sampling:
            raise ValueError(f"{row_id}: registered sampling ledger drifted")
        if row.get("scheduler_runtime") != actions_config.get("scheduler_runtime"):
            raise ValueError(f"{row_id}: scheduler runtime ledger drifted")
        if row.get("model_name") != sampling.get("model"):
            raise ValueError(f"{row_id}: model differs from registration")
        if row.get("model_revision") != sampling.get("model_revision"):
            raise ValueError(f"{row_id}: model revision differs from registration")
        if require_finite(
            row.get("guidance_rescale"), f"{row_id}: guidance rescale"
        ) != float(sampling["guidance_rescale"]):
            raise ValueError(f"{row_id}: guidance rescale differs from registration")
        if row.get("width") != sampling.get("resolution") or row.get(
            "height"
        ) != sampling.get("resolution"):
            raise ValueError(f"{row_id}: image dimensions differ from registration")
        if row.get("stage2_enabled") != sampling.get("stage2"):
            raise ValueError(f"{row_id}: Stage-2 flag differs from registration")
        if row.get("power_calibrate") != sampling.get("power_calibrate"):
            raise ValueError(f"{row_id}: power calibration differs from registration")
        if row.get("base_scheduler_name") != sampling.get("base_scheduler"):
            raise ValueError(f"{row_id}: base scheduler differs from registration")
        if row.get("scheduler_name") != sampling.get("base_scheduler"):
            raise ValueError(f"{row_id}: active scheduler differs from registration")
        if row.get("scheduler_reference") is not False:
            raise ValueError(f"{row_id}: scheduler_reference flag is wrong")
        if row.get("scheduler_kwargs") != {}:
            raise ValueError(f"{row_id}: native Euler scheduler has unexpected kwargs")
        if row.get("num_inference_steps") != sampling.get("num_inference_steps"):
            raise ValueError(f"{row_id}: step count differs from registration")
        if row.get("unet_calls_per_step") != [1] * int(sampling["num_inference_steps"]):
            raise ValueError(f"{row_id}: U-Net call ledger is not exactly one per step")
        if row.get("extra_unet_calls") != 0:
            raise ValueError(f"{row_id}: extra U-Net calls are non-zero")
        if row.get("error") not in (None, ""):
            raise ValueError(f"{row_id}: sidecar contains an error")
        if not isinstance(row.get("device"), str) or not re.fullmatch(
            r"cuda:\d+", row["device"]
        ):
            raise ValueError(f"{row_id}: CUDA device provenance is invalid")
        if row.get("scheduler_solver_order") is not None:
            raise ValueError(f"{row_id}: native Euler scheduler has a solver order")
        if row.get("scheduler_order") != 1:
            raise ValueError(f"{row_id}: native Euler scheduler order differs")
        base_scheduler_config = row.get("scheduler_config")
        active_scheduler_config = row.get("active_scheduler_config")
        if not isinstance(base_scheduler_config, dict) or not base_scheduler_config:
            raise ValueError(f"{row_id}: complete base scheduler config is missing")
        if not isinstance(active_scheduler_config, dict) or not active_scheduler_config:
            raise ValueError(f"{row_id}: complete active scheduler config is missing")
        if row.get("scheduler_config_sha256_v2") != scheduler_config_sha256_v2(
            base_scheduler_config
        ):
            raise ValueError(f"{row_id}: base scheduler config hash is invalid")
        if row.get("active_scheduler_config_sha256_v2") != scheduler_config_sha256_v2(
            active_scheduler_config
        ):
            raise ValueError(f"{row_id}: active scheduler config hash is invalid")
        scheduler_runtime = actions_config["scheduler_runtime"]
        if (
            row.get("scheduler_config_sha256_v2")
            != scheduler_runtime["config_sha256_v2"]
            or row.get("active_scheduler_config_sha256_v2")
            != scheduler_runtime["config_sha256_v2"]
        ):
            raise ValueError(f"{row_id}: Euler config differs from frozen v1")

        construction = require_finite(
            row.get("scheduler_construction_init_noise_sigma"),
            f"{row_id}: construction init sigma",
        )
        effective = require_finite(
            row.get("scheduler_effective_init_noise_sigma"),
            f"{row_id}: effective init sigma",
        )
        if construction <= 0 or effective <= 0:
            raise ValueError(f"{row_id}: initial noise sigma must be positive")
        if construction != scheduler_runtime["construction_init_noise_sigma"]:
            raise ValueError(f"{row_id}: construction init sigma differs from frozen v1")
        if effective != scheduler_runtime["effective_init_noise_sigma"]:
            raise ValueError(f"{row_id}: effective init sigma differs from frozen v1")
        if row.get("scheduler_init_noise_sigma") != construction:
            raise ValueError(f"{row_id}: legacy init sigma alias is inconsistent")
        timesteps = row.get("scheduler_timesteps")
        sigmas = row.get("scheduler_sigmas")
        if not isinstance(timesteps, list) or len(timesteps) != int(
            sampling["num_inference_steps"]
        ):
            raise ValueError(f"{row_id}: timestep schedule length is invalid")
        if not isinstance(sigmas, list) or len(sigmas) != len(timesteps) + 1:
            raise ValueError(f"{row_id}: sigma schedule length is invalid")
        for index, value in enumerate(timesteps):
            require_finite(value, f"{row_id}: timestep[{index}]")
        for index, value in enumerate(sigmas):
            require_finite(value, f"{row_id}: sigma[{index}]")
        schedule_payload = {"timesteps": timesteps, "sigmas": sigmas}
        if row.get("scheduler_schedule_sha256") != json_sha256(schedule_payload):
            raise ValueError(f"{row_id}: scheduler schedule hash is invalid")
        if (
            row.get("scheduler_schedule_sha256")
            != scheduler_runtime["schedule_sha256"]
        ):
            raise ValueError(f"{row_id}: Euler schedule differs from frozen v1")
        rows_by_pair[(int(row["prompt_index"]), int(row["seed"]))].append(row)

    if len(rows_by_pair) != int(design.get("expected_block_count", -1)):
        raise ValueError("observed CFG block count differs from registration")
    duplicate_image_groups: list[dict] = []
    for pair, rows in rows_by_pair.items():
        if len({str(row.get("device")) for row in rows}) != 1:
            raise ValueError(f"prompt/seed block {pair} spans devices")
        expected_ranks = expected_execution_ranks(pair[0], pair[1], action_ids)
        for row in rows:
            if row.get("execution_rank") != expected_ranks[str(row["action_id"])]:
                raise ValueError(f"prompt/seed block {pair} has invalid execution ranks")
        by_image_hash: Dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_image_hash[str(row["image_sha256"])].append(row)
        for image_hash, duplicate_rows in by_image_hash.items():
            if len(duplicate_rows) > 1:
                duplicate_image_groups.append(
                    {
                        "prompt_index": pair[0],
                        "seed": pair[1],
                        "image_sha256": image_hash,
                        "ids": sorted(str(row["id"]) for row in duplicate_rows),
                        "action_ids": sorted(
                            str(row["action_id"]) for row in duplicate_rows
                        ),
                    }
                )

    uniform_scheduler_fields = (
        "scheduler_config_sha256_v2",
        "active_scheduler_config_sha256_v2",
        "scheduler_config",
        "active_scheduler_config",
        "scheduler_schedule_sha256",
        "scheduler_construction_init_noise_sigma",
        "scheduler_effective_init_noise_sigma",
        "scheduler_timesteps",
        "scheduler_sigmas",
    )
    scheduler_summary = {
        key: require_uniform(manifest, key, "CFG matrix")
        for key in uniform_scheduler_fields
    }
    if (
        scheduler_summary["scheduler_config_sha256_v2"]
        != scheduler_summary["active_scheduler_config_sha256_v2"]
    ):
        raise ValueError("CFG actions do not use the frozen base scheduler config")
    if scheduler_summary["scheduler_config"] != scheduler_summary[
        "active_scheduler_config"
    ]:
        raise ValueError("CFG actions do not preserve the complete base scheduler config")

    scores_path = os.path.join(run_dir, "scores.jsonl")
    scores_sha256 = None
    scoring_contract_sha256 = None
    if kind == "scores":
        scores = load_jsonl(scores_path)
        validate_scores_against_manifest(manifest, scores)
        manifest_by_id = {str(row["id"]): row for row in manifest}
        scores_by_id = {str(score["id"]): score for score in scores}
        scoring = actions_config["scoring"]
        scoring_contract = {
            "schema": SCORING_SCHEMA,
            "action_schema": "cfg_baselines_v1",
            "metrics": list(scoring["metrics"]),
            "strict": True,
            "params": dict(scoring["params"]),
            "required_score_keys": list(scoring["required_score_keys"]),
            "actions_sha256": run_config["actions_sha256"],
        }
        scoring_contract_sha256 = json_sha256(scoring_contract)
        for score in scores:
            row = manifest_by_id[str(score["id"])]
            if score.get("provenance_schema") != PROVENANCE_SCHEMA:
                raise ValueError(f"{score['id']}: score provenance schema is missing")
            for key in (
                "prompt_index",
                "seed",
                "action_id",
                "action_type",
                "action_sha256",
                "image_path",
            ):
                if score.get(key) != row.get(key):
                    raise ValueError(f"{score['id']}: score field {key!r} drifted")
            if score.get("scoring_contract") != scoring_contract:
                raise ValueError(f"{score['id']}: scoring contract drifted")
            if score.get("scoring_contract_sha256") != scoring_contract_sha256:
                raise ValueError(f"{score['id']}: scoring contract hash drifted")
            for key in SCORE_KEYS:
                require_finite(score.get(key), f"{score['id']}: score {key}")
        for group in duplicate_image_groups:
            reference = scores_by_id[group["ids"][0]]
            for row_id in group["ids"][1:]:
                candidate = scores_by_id[row_id]
                for key in SCORE_KEYS:
                    if candidate[key] != reference[key]:
                        raise ValueError(
                            f"duplicate image {group['image_sha256']} has inconsistent "
                            f"deterministic score {key!r}"
                        )
        scores_sha256 = sha256_file(scores_path)

    return {
        "schema": "cfg_baseline_run_audit_v1",
        "status": "pass",
        "kind": kind,
        "run_dir": run_dir,
        "git_commit": run_config.get("git_commit"),
        "run_contract_sha256": contract_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "scores_sha256": scores_sha256,
        "scoring_contract_sha256": scoring_contract_sha256,
        "row_count": len(manifest),
        "score_keys": list(SCORE_KEYS) if kind == "scores" else [],
        "duplicate_image_group_count": len(duplicate_image_groups),
        "duplicate_image_groups": duplicate_image_groups,
        "scheduler_summary": scheduler_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--kind", choices=("manifest", "scores"), required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = validate(args.run_dir, args.actions, args.prompts, args.kind)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        temporary = args.output + ".tmp"
        with open(temporary, "w") as handle:
            handle.write(payload + "\n")
        os.replace(temporary, args.output)
    print(payload)


if __name__ == "__main__":
    main()
