"""Strict post-run validator for registered matched-NFE scheduler controls."""

from __future__ import annotations

import argparse
import json
import math
import os
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


SCORE_KEYS = (
    "imagereward",
    "patch_ir_mean",
    "patch_ir_std",
    "patch_ir_n",
    "colorfulness",
    "laplacian_sharpness",
    "clipped_fraction",
    "mean_saturation",
    "contrast_std",
    "clip_cosine",
    "clipscore",
    "hpsv2",
    "aesthetic",
    "topiq_nr",
)


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


def require_uniform(rows: Iterable[Mapping[str, Any]], key: str, label: str) -> Any:
    values: Dict[str, Any] = {}
    for row in rows:
        value = row.get(key)
        values[json.dumps(value, sort_keys=True, default=str)] = value
    if len(values) != 1:
        raise ValueError(f"{label} has {len(values)} distinct {key} values")
    return next(iter(values.values()))


def validate(
    run_dir: str,
    actions_path: str,
    prompts_path: str,
    kind: str,
) -> dict:
    """Validate the frozen design, scheduler ledger, images, and optional scores."""
    run_dir = os.path.abspath(run_dir)
    with open(os.path.join(run_dir, "config.json")) as handle:
        run_config = json.load(handle)
    with open(actions_path) as handle:
        actions_config = yaml.safe_load(handle) or {}
    prompts = pd.read_csv(prompts_path)

    if actions_config.get("schema") != "scheduler_baselines_v1":
        raise ValueError("actions YAML is not scheduler_baselines_v1")
    if actions_config.get("status") != "authorized":
        raise ValueError("scheduler baseline config is not authorized")
    authorization = actions_config.get("authorization")
    if not isinstance(authorization, dict) or not authorization.get("reviewed_commit"):
        raise ValueError("scheduler baseline authorization is incomplete")
    if run_config.get("action_schema") != "scheduler_baselines_v1":
        raise ValueError("run config has the wrong action schema")
    if run_config.get("scheduler_baseline_registered") is not True:
        raise ValueError("run is not marked as a registered scheduler baseline")
    if run_config.get("trajectory_registered") is not False:
        raise ValueError("scheduler reference run must not be a trajectory-selection run")
    if run_config.get("actions_sha256") != sha256_file(actions_path):
        raise ValueError("run actions hash differs from the authorized YAML")
    if run_config.get("prompts_sha256") != sha256_file(prompts_path):
        raise ValueError("run prompts hash differs from the registered CSV")
    if prompts["index"].duplicated().any():
        raise ValueError("prompt CSV contains duplicate indices")

    sampling = actions_config.get("sampling") or {}
    if run_config.get("registered_sampling") != sampling:
        raise ValueError("run registered_sampling differs from the authorized YAML")
    expected_sampling = {
        "model_name": sampling.get("model"),
        "resolution": int(sampling.get("resolution")),
        "num_inference_steps": int(sampling.get("num_inference_steps")),
        "guidance_scale": float(sampling.get("cfg_scale")),
        "stage2_enabled": bool(sampling.get("stage2")),
    }
    for key, expected in expected_sampling.items():
        observed = run_config.get(key)
        if isinstance(expected, float):
            same = isinstance(observed, (int, float)) and float(observed) == expected
        else:
            same = observed == expected
        if not same:
            raise ValueError(f"run field {key!r} differs from registration")
    if sampling.get("extra_unet_calls") != 0:
        raise ValueError("registered scheduler matrix must require zero extra U-Net calls")

    split_role = run_config.get("split_role")
    expected_seeds = (actions_config.get("split_seeds") or {}).get(split_role)
    if expected_seeds is None or run_config.get("seeds") != expected_seeds:
        raise ValueError("run seeds differ from the registered split role")
    design = actions_config.get("design") or {}
    source_actions = actions_config.get("actions") or []
    source_by_id = {str(action.get("id")): action for action in source_actions}
    action_ids = [str(value) for value in design.get("action_ids", [])]
    if action_ids != [str(action.get("id")) for action in source_actions]:
        raise ValueError("registered action order differs from design.action_ids")
    if any(bool(action.get("selection_eligible", True)) for action in source_actions):
        raise ValueError("scheduler controls must all be selection-ineligible")
    run_actions = run_config.get("actions") or []
    if [str(action.get("id")) for action in run_actions] != action_ids:
        raise ValueError("normalized run actions differ from the registered action order")
    expected_action_hashes = design.get("action_sha256") or {}
    for action in run_actions:
        action_id = str(action.get("id"))
        if action_sha256(action) != expected_action_hashes.get(action_id):
            raise ValueError(f"normalized action hash differs for {action_id!r}")
        if bool(action.get("selection_eligible", True)):
            raise ValueError(f"normalized action {action_id!r} became selection-eligible")

    contract_hash = validate_run_contract(run_config)
    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    manifest = load_jsonl(manifest_path)
    expected_prompts = [int(value) for value in prompts["index"].tolist()]
    validate_design_rows(
        manifest,
        expected_action_ids=action_ids,
        expected_seeds=expected_seeds,
        expected_prompt_indices=expected_prompts,
    )
    expected_count = int(design.get("expected_task_count", -1))
    if len(manifest) != expected_count:
        raise ValueError("manifest row count differs from registered design")
    if expected_count != len(expected_prompts) * len(expected_seeds) * len(action_ids):
        raise ValueError("registered design counts are internally inconsistent")

    prompt_by_index = {
        int(row["index"]): row for _, row in prompts.iterrows()
    }
    rows_by_action: Dict[str, list[dict]] = defaultdict(list)
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
        if row.get("action_type") != source_action.get("type"):
            raise ValueError(f"{row_id}: action type differs from registration")
        if row.get("action_sha256") != expected_action_hashes[action_id]:
            raise ValueError(f"{row_id}: action hash differs from registration")
        if row.get("registered_sampling") != sampling:
            raise ValueError(f"{row_id}: registered sampling ledger drifted")
        if row.get("model_name") != sampling.get("model"):
            raise ValueError(f"{row_id}: model differs from registration")
        if row.get("base_scheduler_name") != sampling.get("base_scheduler"):
            raise ValueError(f"{row_id}: base scheduler differs from registration")
        if row.get("num_inference_steps") != sampling.get("num_inference_steps"):
            raise ValueError(f"{row_id}: NFE differs from registration")
        calls = row.get("unet_calls_per_step")
        if calls != [1] * int(sampling["num_inference_steps"]):
            raise ValueError(f"{row_id}: U-Net call ledger is not exactly one per step")
        if row.get("extra_unet_calls") != 0:
            raise ValueError(f"{row_id}: extra U-Net calls are non-zero")
        if row.get("error") not in (None, ""):
            raise ValueError(f"{row_id}: sidecar contains an error")

        reference = source_action.get("type") == "scheduler_baseline"
        expected_name = (
            source_action.get("scheduler_class")
            if reference
            else sampling.get("base_scheduler")
        )
        if row.get("scheduler_reference") is not reference:
            raise ValueError(f"{row_id}: scheduler_reference flag is wrong")
        if row.get("scheduler_name") != expected_name:
            raise ValueError(f"{row_id}: active scheduler class is wrong")
        expected_kwargs = source_action.get("scheduler_kwargs") or {}
        if row.get("scheduler_kwargs") != expected_kwargs:
            raise ValueError(f"{row_id}: scheduler kwargs differ from registration")
        if row.get("scheduler_solver_order") != expected_kwargs.get("solver_order"):
            raise ValueError(f"{row_id}: solver order differs from registration")
        if not isinstance(row.get("scheduler_order"), int) or row["scheduler_order"] <= 0:
            raise ValueError(f"{row_id}: scheduler order is invalid")

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
        rows_by_action[action_id].append(row)
        rows_by_pair[(int(row["prompt_index"]), int(row["seed"]))].append(row)

    for pair, rows in rows_by_pair.items():
        if len({str(row.get("device")) for row in rows}) != 1:
            raise ValueError(f"prompt/seed block {pair} spans devices")
        if sorted(int(row.get("execution_rank", -1)) for row in rows) != list(
            range(len(action_ids))
        ):
            raise ValueError(f"prompt/seed block {pair} has invalid execution ranks")

    base_hash = require_uniform(
        manifest, "scheduler_config_sha256_v2", "scheduler matrix"
    )
    schedule_summary = {}
    for action_id in action_ids:
        rows = rows_by_action[action_id]
        summary = {
            "count": len(rows),
            "scheduler_name": require_uniform(rows, "scheduler_name", action_id),
            "active_config_sha256_v2": require_uniform(
                rows, "active_scheduler_config_sha256_v2", action_id
            ),
            "schedule_sha256": require_uniform(
                rows, "scheduler_schedule_sha256", action_id
            ),
            "construction_init_noise_sigma": require_uniform(
                rows, "scheduler_construction_init_noise_sigma", action_id
            ),
            "effective_init_noise_sigma": require_uniform(
                rows, "scheduler_effective_init_noise_sigma", action_id
            ),
        }
        schedule_summary[action_id] = summary
    no_correction = schedule_summary.get("no_correction")
    if not no_correction or no_correction["active_config_sha256_v2"] != base_hash:
        raise ValueError("no_correction does not use the frozen base scheduler config")

    scores_path = os.path.join(run_dir, "scores.jsonl")
    scores_sha256 = None
    if kind == "scores":
        scores = load_jsonl(scores_path)
        validate_scores_against_manifest(manifest, scores)
        manifest_by_id = {str(row["id"]): row for row in manifest}
        for score in scores:
            row = manifest_by_id[str(score["id"])]
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
            for key in SCORE_KEYS:
                require_finite(score.get(key), f"{score['id']}: score {key}")
            if not isinstance(score.get("patch_ir_n"), int) or score["patch_ir_n"] <= 0:
                raise ValueError(f"{score['id']}: patch_ir_n must be a positive integer")
        scores_sha256 = sha256_file(scores_path)

    return {
        "schema": "scheduler_baseline_run_audit_v1",
        "status": "pass",
        "kind": kind,
        "run_dir": run_dir,
        "git_commit": run_config.get("git_commit"),
        "run_contract_sha256": contract_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "scores_sha256": scores_sha256,
        "row_count": len(manifest),
        "score_keys": list(SCORE_KEYS) if kind == "scores" else [],
        "schedule_summary": schedule_summary,
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
        tmp = args.output + ".tmp"
        with open(tmp, "w") as handle:
            handle.write(payload + "\n")
        os.replace(tmp, args.output)
    print(payload)


if __name__ == "__main__":
    main()
