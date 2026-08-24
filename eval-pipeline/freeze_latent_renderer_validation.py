"""Freeze the one permitted LR-1 validation action and preregistered controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_validation_config(
    selection: Dict[str, Any], source: Dict[str, Any], template: Dict[str, Any]
) -> Dict[str, Any]:
    train_requirements = template.get("train_selection_requirements", {})
    for field in ("baseline", "selection_metric", "topiq_used_for_selection", "bootstrap", "seed"):
        if selection.get(field) != train_requirements.get(field):
            raise ValueError(f"train selection field {field!r} differs from registration")
    if selection.get("constraints") != train_requirements.get("constraints"):
        raise ValueError("train selection constraints differ from registration")
    registration = selection.get("registration")
    if not isinstance(registration, dict):
        raise ValueError("train selection lacks registration provenance")
    if registration.get("source_experiment_id") != train_requirements.get("source_experiment_id"):
        raise ValueError("train selection source experiment differs from registration")
    if source.get("experiment_id") != train_requirements.get("source_experiment_id"):
        raise ValueError("source YAML experiment differs from validation template")
    if registration.get("source_schema") != source.get("schema"):
        raise ValueError("train selection source schema differs from source YAML")
    if registration.get("split_role") != train_requirements.get("split_role"):
        raise ValueError("train selection split role differs from registration")
    selected_id = str(selection.get("selected_action", ""))
    if not selected_id or selected_id == str(selection.get("baseline", "no_ag")):
        raise ValueError("no non-baseline train action was selected; validation is closed")
    source_actions = {str(action.get("id")): dict(action) for action in source.get("actions", [])}
    source_action_ids = list(source_actions)
    expected_candidates = [
        action_id
        for action_id, action in source_actions.items()
        if action.get("type") == "latent_renderer_fixed"
    ]
    if registration.get("action_ids") != source_action_ids:
        raise ValueError("train run did not use the complete registered action set")
    if registration.get("candidate_actions") != expected_candidates:
        raise ValueError("train registration candidate set differs from source YAML")
    if selection.get("candidate_actions") != expected_candidates:
        raise ValueError("train selection omitted or reordered registered candidates")
    if selected_id not in source_actions:
        raise ValueError(f"selected action {selected_id!r} is absent from source YAML")
    selected = source_actions[selected_id]
    if selected.get("type") != "latent_renderer_fixed":
        raise ValueError("selected validation action must be latent_renderer_fixed")
    coefficients = [float(value) for value in selected.get("coefficients", [])]
    if len(coefficients) != 6 or not all(math.isfinite(value) for value in coefficients):
        raise ValueError("selected action must contain six finite coefficients")

    controls = template.get("controls", {})
    random_spec = controls.get("matched_random", {})
    signs = [int(value) for value in random_spec.get("signs", [])]
    if len(signs) != len(coefficients) or any(value not in (-1, 1) for value in signs):
        raise ValueError("matched-random signs must be one +/-1 value per coefficient")
    coefficient_norm = math.sqrt(sum(value * value for value in coefficients))
    random_scale = coefficient_norm / math.sqrt(len(signs))
    random_action = {
        "id": str(random_spec.get("id", "matched_random")),
        "type": "latent_renderer_fixed",
        "coefficients": [sign * random_scale for sign in signs],
        "matched_coefficient_l2": coefficient_norm,
        "random_seed_label": int(random_spec.get("seed_label", 20260824)),
    }
    no_ag = {"id": "no_ag", "type": "none"}
    expert = dict(controls.get("conference_expert", {}))
    if expert.get("type") != "legacy":
        raise ValueError("validation template must define the conference expert")
    if not controls.get("include_no_ag") or not controls.get("include_conference_expert"):
        raise ValueError("validation template must retain no-AG and conference expert")

    split_seeds = source.get("split_seeds")
    train_role = str(train_requirements.get("split_role", ""))
    train_seeds = [int(value) for value in registration.get("seeds", [])]
    if not isinstance(split_seeds, dict) or split_seeds.get(train_role) != train_seeds:
        raise ValueError("train selection seeds disagree with source split_seeds")
    requirements = template.get("validation_requirements", {})
    validation_role = str(requirements.get("split_role", ""))
    validation_seeds = [int(value) for value in requirements.get("seeds", [])]
    if split_seeds.get(validation_role) != validation_seeds:
        raise ValueError("template validation seeds disagree with source split_seeds")
    if int(template.get("selected_action_count", 0)) != 1:
        raise ValueError("validation template must authorize exactly one selected action")
    selected["selection_role"] = "train_search_winner"
    return {
        "schema": "latent_renderer_actions_v1",
        "experiment_id": str(template.get("experiment_id", "lr1_fixed_validation_v1")),
        "status": "frozen_validation",
        "selected_action": selected_id,
        "split_seeds": {validation_role: validation_seeds},
        "latent_renderer_provider": dict(source.get("latent_renderer_provider", {})),
        "frequency_band_cutoffs": list(source.get("frequency_band_cutoffs", [0.08, 0.25])),
        "validation_requirements": requirements,
        "train_selection": {
            "selected_action": selected_id,
            "requirements": train_requirements,
            "registration": registration,
        },
        "actions": [no_ag, selected, expert, random_action],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument(
        "--source_actions",
        default="eval-pipeline/configs/latent_renderer_fixed_lr1.yaml",
    )
    parser.add_argument(
        "--template",
        default="eval-pipeline/configs/latent_renderer_validation_template.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.selection) as handle:
        selection = json.load(handle)
    with open(args.source_actions) as handle:
        source = yaml.safe_load(handle) or {}
    with open(args.template) as handle:
        template = yaml.safe_load(handle) or {}
    source_hash = sha256_file(args.source_actions)
    selection_provenance = selection.get("provenance", {})
    if selection_provenance.get("source_actions_sha256") != source_hash:
        raise ValueError("selection provenance does not match the source action YAML")
    result = freeze_validation_config(selection, source, template)
    result["provenance"] = {
        "selection_path": os.path.abspath(args.selection),
        "selection_sha256": sha256_file(args.selection),
        "source_actions_sha256": source_hash,
        "template_sha256": sha256_file(args.template),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print(f"frozen validation action {result['selected_action']} -> {output}")


if __name__ == "__main__":
    main()
