"""Read-only completeness/provenance validator used by the S7 queue."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import yaml

from s7_provenance import (
    PROVENANCE_SCHEMA,
    action_sha256,
    sha256_file,
    validate_design_rows,
    validate_run_contract,
    validate_scores_against_manifest,
    validate_sidecar,
)


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(run_dir, actions_path, prompts_path, seeds, kind):
    run_dir = os.path.abspath(run_dir)
    with open(os.path.join(run_dir, "config.json")) as handle:
        config = json.load(handle)
    with open(actions_path) as handle:
        actions_config = yaml.safe_load(handle) or {}
    prompts = pd.read_csv(prompts_path)
    contract_hash = validate_run_contract(config)
    action_ids = [str(action.get("id", "")) for action in config.get("actions", [])]
    if not action_ids or len(action_ids) != len(set(action_ids)):
        raise ValueError("run config has no unique action list")
    if config.get("actions_sha256") != sha256_file(actions_path):
        raise ValueError("run config actions hash differs from registered YAML")
    if actions_config.get("schema") not in {
        "trajectory_correction_actions_v1",
        "trajectory_correction_validation_v1",
    }:
        raise ValueError("actions YAML has an invalid S7 schema")
    if config.get("action_schema") != actions_config.get("schema"):
        raise ValueError("run config action schema differs from actions YAML")
    if config.get("prompts_sha256") != sha256_file(prompts_path):
        raise ValueError("run config prompts hash differs from registered CSV")
    if prompts["index"].duplicated().any():
        raise ValueError("prompt CSV contains duplicate indices")
    sampling = actions_config.get("sampling") or {}
    if config.get("trajectory_registered"):
        if config.get("model_name") != sampling.get("model"):
            raise ValueError("run model differs from registered sampling.model")
        if config.get("registered_sampling", {}).get("scheduler") != sampling.get("scheduler"):
            raise ValueError("run scheduler differs from registered sampling.scheduler")
        if config.get("registered_sampling", {}).get("extra_unet_calls") != 0:
            raise ValueError("registered S7 run has non-zero extra_unet_calls")
        expected_sampling = {
            "resolution": int(sampling.get("resolution")),
            "num_inference_steps": int(sampling.get("num_inference_steps")),
            "cfg_scale": float(sampling.get("cfg_scale")),
            "stage2": sampling.get("stage2"),
        }
        observed_sampling = {
            "resolution": int(config.get("resolution")),
            "num_inference_steps": int(config.get("num_inference_steps")),
            "cfg_scale": float(config.get("guidance_scale")),
            "stage2": bool(config.get("stage2_enabled")),
        }
        if expected_sampling != observed_sampling:
            raise ValueError("run sampling fields differ from registration")
        if config.get("registered_sampling", {}).get("noise_mode") is not None:
            if config["registered_sampling"]["noise_mode"] not in {"sqrt", "linear", "none"}:
                raise ValueError("registered sampling noise_mode is invalid")
    expected_prompts = [int(value) for value in prompts["index"].tolist()]
    expected_seeds = [int(value) for value in seeds]
    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    manifest = load_jsonl(manifest_path)
    validate_design_rows(
        manifest,
        expected_action_ids=action_ids,
        expected_seeds=expected_seeds,
        expected_prompt_indices=expected_prompts,
    )
    prompt_by_index = {
        int(row["index"]): row for _, row in prompts.iterrows()
    }
    source_actions = {
        str(action.get("id")): action for action in actions_config.get("actions", [])
    }
    if not set(action_ids).issubset(set(source_actions)):
        raise ValueError("run action set contains an unregistered action")
    for record in manifest:
        if record.get("provenance_schema") != PROVENANCE_SCHEMA:
            raise ValueError(f"{record.get('id')}: missing provenance schema")
        validate_sidecar(record, run_dir, expected_contract_sha256=contract_hash)
        prompt = prompt_by_index[int(record["prompt_index"])]
        if str(record.get("prompt")) != str(prompt["TEXT"]):
            raise ValueError(f"{record['id']}: prompt text differs from registered CSV")
        if "bucket" in prompts.columns and str(record.get("bucket", "")) != str(prompt.get("bucket", "")):
            raise ValueError(f"{record['id']}: prompt bucket differs from registered CSV")
        action_id = str(record.get("action_id"))
        source_action = source_actions[action_id]
        observed_action = record.get("action")
        if record.get("action_type") != source_action.get("type"):
            raise ValueError(f"{record['id']}: action type differs from registration")
        if not isinstance(observed_action, dict) or observed_action.get("id") != action_id:
            raise ValueError(f"{record['id']}: normalized action provenance is invalid")
        if action_sha256(observed_action) != record.get("action_sha256"):
            raise ValueError(f"{record['id']}: action hash differs from normalized action")
        calls = record.get("unet_calls_per_step")
        if not isinstance(calls, list) or len(calls) != int(config["num_inference_steps"]):
            raise ValueError(f"{record['id']}: U-Net call ledger length is invalid")
        if any(int(value) != 1 for value in calls) or int(record.get("extra_unet_calls", -1)) != 0:
            raise ValueError(f"{record['id']}: unexpected extra U-Net call")
        if source_action.get("type") == "trajectory_correction":
            for key in ("mix", "noise_mode"):
                if str(observed_action.get(key)) != str(source_action.get(key)):
                    raise ValueError(f"{record['id']}: trajectory action field {key!r} drifted")
    if kind == "manifest":
        return len(manifest)
    scores = load_jsonl(os.path.join(run_dir, "scores.jsonl"))
    validate_scores_against_manifest(manifest, scores)
    return len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--kind", choices=("manifest", "scores"), required=True)
    args = parser.parse_args()
    count = validate(
        args.run_dir,
        args.actions,
        args.prompts,
        [int(value) for value in args.seeds.split(",") if value != ""],
        args.kind,
    )
    print(count)


if __name__ == "__main__":
    main()
