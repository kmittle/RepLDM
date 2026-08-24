"""Freeze the S7 validation template from a completed development selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import yaml


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(selection_path: str, template_path: str, output_path: str) -> dict:
    with open(selection_path) as handle:
        selection = json.load(handle)
    selected = str(selection.get("selected_action", ""))
    if not selected or selected == "no_correction":
        raise ValueError("development gate did not authorize a non-baseline validation action")
    passing = {
        str(row.get("action"))
        for row in selection.get("rows", [])
        if row.get("passes_gate") is True
    }
    if selected not in passing:
        raise ValueError("selected action is not marked passes_gate in the development report")

    with open(template_path) as handle:
        template = yaml.safe_load(handle) or {}
    if template.get("schema") != "trajectory_correction_validation_v1":
        raise ValueError("template has the wrong validation schema")
    if template.get("selected_action") not in (None, ""):
        raise ValueError("validation template is already frozen")
    action_ids = [str(action.get("id", "")) for action in template.get("actions", [])]
    if selected not in action_ids or "no_correction" not in action_ids:
        raise ValueError("selected action is absent from the validation template")

    frozen = dict(template)
    frozen["selected_action"] = selected
    frozen["selection_provenance"] = {
        "selection_path": os.path.abspath(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selected_action": selected,
        "development_gate": selection.get("gate"),
    }
    if os.path.abspath(output_path) == os.path.abspath(template_path):
        raise ValueError("write a new validation file; do not overwrite the template")
    with open(output_path, "w") as handle:
        yaml.safe_dump(frozen, handle, sort_keys=False)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument(
        "--template",
        default="eval-pipeline/configs/trajectory_correction_validation_template.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen = freeze(args.selection, args.template, args.output)
    print(json.dumps({"selected_action": frozen["selected_action"], "output": os.path.abspath(args.output)}, indent=2))


if __name__ == "__main__":
    main()
