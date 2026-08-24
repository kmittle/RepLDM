"""Select a fixed S7 action using the preregistered development gate.

This selector reads only the development run.  It never optimizes a guard or
adds an action after seeing scores; a failed gate returns ``no_correction`` and
closes the route for the next stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from compare_actions import crossed_bootstrap_ci, holm_adjust, prompt_sign_flip_pvalue, validate_pairing


PRIMARY_MIN_DELTA = 0.005
NONINFERIORITY = -0.005
CLIPPED_MAX_DELTA = 0.001
SATURATION_MAX_DELTA = 0.005


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _complete_pivot(frame: pd.DataFrame, metric: str, actions: List[str]) -> pd.DataFrame:
    if frame.duplicated(["prompt_index", "seed", "action_id"]).any():
        raise ValueError("duplicate prompt/seed/action rows make selection ambiguous")
    pivot = frame.pivot(index=["prompt_index", "seed"], columns="action_id", values=metric)
    expected_index = pd.MultiIndex.from_product(
        [sorted(frame["prompt_index"].unique()), sorted(frame["seed"].unique())],
        names=["prompt_index", "seed"],
    )
    pivot = pivot.reindex(index=expected_index, columns=actions)
    if pivot.isna().any().any():
        raise ValueError(f"incomplete prompt x seed x action design for {metric}")
    if not np.isfinite(pivot.to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite values found for {metric}")
    return pivot


def _diagnostics_are_valid(frame: pd.DataFrame, action: str) -> bool:
    if "trajectory_correction_diagnostics" not in frame:
        return False
    rows = frame[frame["action_id"] == action]
    for value in rows["trajectory_correction_diagnostics"]:
        if not isinstance(value, list) or not value:
            return False
        expected_steps = None
        if "num_inference_steps" in rows:
            expected_steps = int(rows.loc[rows.index[0], "num_inference_steps"])
            if expected_steps <= 0 or len(value) != expected_steps:
                return False
        previous_sigma_from = None
        for expected_index, record in enumerate(value):
            if not isinstance(record, dict):
                return False
            for key in (
                "step_index",
                "sigma_from",
                "sigma_to",
                "sigma_up",
                "raw_correction_norm_ratio",
                "applied_correction_norm_ratio",
            ):
                if not math.isfinite(float(record.get(key))):
                    return False
            if int(record["step_index"]) != expected_index:
                return False
            sigma_from = float(record["sigma_from"])
            sigma_to = float(record["sigma_to"])
            sigma_up = float(record["sigma_up"])
            if sigma_from <= 0.0 or sigma_to < 0.0 or sigma_to > sigma_from:
                return False
            if sigma_up < 0.0 or sigma_up > sigma_to + 1e-6:
                return False
            if previous_sigma_from is not None and sigma_from > previous_sigma_from + 1e-5:
                return False
            previous_sigma_from = sigma_from
    return True


def select(
    frame: pd.DataFrame,
    *,
    action_order: List[str],
    selection_eligible: Optional[List[str]] = None,
    baseline: str = "no_correction",
    bootstrap: int = 10000,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Return a frozen action decision and the complete gate table."""
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    validate_pairing(frame)
    required = {
        "action_id",
        "prompt_index",
        "seed",
        "topiq_nr",
        "hpsv2",
        "clip_cosine",
        "clipped_fraction",
        "mean_saturation",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing selection columns: {missing}")
    available = set(str(value) for value in frame["action_id"])
    if baseline not in available:
        raise ValueError(f"baseline {baseline!r} is missing")
    ordered = [str(value) for value in action_order]
    if ordered != list(dict.fromkeys(ordered)):
        raise ValueError("registered action order contains duplicates")
    if set(ordered) != available:
        raise ValueError(
            f"observed actions {sorted(available)} differ from registered actions {ordered}"
        )
    metrics = ["topiq_nr", "hpsv2", "clip_cosine", "clipped_fraction", "mean_saturation"]
    pivots = {metric: _complete_pivot(frame, metric, ordered) for metric in metrics}
    eligible_set = (
        set(ordered) - {baseline}
        if selection_eligible is None
        else {str(action) for action in selection_eligible}
    )
    if baseline in eligible_set:
        raise ValueError("baseline cannot be a selectable candidate")
    if not eligible_set.issubset(set(ordered)):
        raise ValueError("selection_eligible contains an unregistered action")
    candidates = [
        action for action in ordered if action != baseline and action in eligible_set
    ]
    reference_actions = [
        action for action in ordered if action != baseline and action not in eligible_set
    ]
    rows = []
    topiq_p = []
    eligible_row_indices = []
    all_candidates = [action for action in ordered if action != baseline]
    for index, action in enumerate(all_candidates):
        deltas = {
            metric: pivots[metric][action] - pivots[metric][baseline]
            for metric in metrics
        }
        cis = {
            metric: crossed_bootstrap_ci(
                deltas[metric], n_boot=bootstrap, seed=seed + index * 100 + metric_index
            )
            for metric_index, metric in enumerate(metrics)
        }
        p = None
        if action in eligible_set:
            p = prompt_sign_flip_pvalue(
                deltas["topiq_nr"], n_random=max(10000, bootstrap), seed=seed + index
            )
            topiq_p.append(p)
            eligible_row_indices.append(len(rows))
        # Scheduler references have no correction diagnostics by design; their
        # finite scorer values are sufficient for the auditable comparison.
        valid_diagnostics = (
            True if action in reference_actions else _diagnostics_are_valid(frame, action)
        )
        row = {
            "action": action,
            "order": index,
            "selection_eligible": action in eligible_set,
            "valid_diagnostics": valid_diagnostics,
            "topiq_mean_delta": float(deltas["topiq_nr"].mean()),
            "topiq_ci_low": float(cis["topiq_nr"][0]),
            "topiq_ci_high": float(cis["topiq_nr"][1]),
            "topiq_p_sign_flip": None if p is None else float(p),
        }
        for metric in metrics[1:]:
            row[f"{metric}_mean_delta"] = float(deltas[metric].mean())
            row[f"{metric}_ci_low"] = float(cis[metric][0])
            row[f"{metric}_ci_high"] = float(cis[metric][1])
        rows.append(row)
    adjusted = holm_adjust(topiq_p) if topiq_p else []
    adjusted_by_row = dict(zip(eligible_row_indices, adjusted))
    for row_index, row in enumerate(rows):
        if row_index in eligible_row_indices:
            adjusted_p = adjusted_by_row[row_index]
            row["topiq_p_holm"] = float(adjusted_p)
        else:
            row["topiq_p_holm"] = None
        row["passes_gate"] = bool(
            row["selection_eligible"]
            and row["valid_diagnostics"]
            and row["topiq_mean_delta"] >= PRIMARY_MIN_DELTA
            and row["topiq_ci_low"] > 0.0
            and row["topiq_p_holm"] < 0.05
            and row["hpsv2_ci_low"] >= NONINFERIORITY
            and row["clip_cosine_ci_low"] >= NONINFERIORITY
            and row["clipped_fraction_ci_high"] <= CLIPPED_MAX_DELTA
            and row["mean_saturation_ci_high"] <= SATURATION_MAX_DELTA
        )
    eligible = [row for row in rows if row["passes_gate"]]
    selected = baseline
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (row["topiq_mean_delta"], -row["order"]),
        )["action"]
    return {
        "selected_action": selected,
        "baseline": baseline,
        "candidate_actions": candidates,
        "reference_actions": reference_actions,
        "selection_eligible": candidates,
        "gate": {
            "primary_metric": "topiq_nr",
            "minimum_mean_delta": PRIMARY_MIN_DELTA,
            "require_ci_low_positive": True,
            "holm_alpha": 0.05,
            "hpsv2_ci_low_min": NONINFERIORITY,
            "clip_cosine_ci_low_min": NONINFERIORITY,
            "clipped_fraction_ci_high_max": CLIPPED_MAX_DELTA,
            "mean_saturation_ci_high_max": SATURATION_MAX_DELTA,
        },
        "bootstrap": int(bootstrap),
        "seed": int(seed),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--baseline", default="no_correction")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    with open(args.actions) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") != "trajectory_correction_actions_v1":
        raise ValueError("S7 selector requires trajectory_correction_actions_v1")
    action_order = [str(action.get("id", "")) for action in config.get("actions", [])]
    if not action_order or any(not action for action in action_order):
        raise ValueError("actions YAML must contain non-empty action ids")
    selection_eligible = [
        str(action.get("id"))
        for action in config.get("actions", [])
        if bool(action.get("selection_eligible", True))
        and str(action.get("id")) != args.baseline
    ]
    manifest = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "manifest.jsonl")))
    scores = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "scores.jsonl")))
    if manifest.empty or scores.empty:
        raise ValueError("manifest and scores must be complete before selection")
    score_columns = [column for column in scores if column == "id" or column not in manifest]
    frame = manifest.merge(scores[score_columns], on="id", how="inner")
    result = select(
        frame,
        action_order=action_order,
        selection_eligible=selection_eligible,
        baseline=args.baseline,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    result["provenance"] = {
        "run_dir": os.path.abspath(args.run_dir),
        "actions_path": os.path.abspath(args.actions),
        "actions_sha256": sha256_file(args.actions),
        "manifest_sha256": sha256_file(os.path.join(args.run_dir, "manifest.jsonl")),
        "scores_sha256": sha256_file(os.path.join(args.run_dir, "scores.jsonl")),
    }
    output_path = os.path.join(args.run_dir, "trajectory_correction_selection.json")
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    pd.DataFrame(result["rows"]).to_csv(
        os.path.join(args.run_dir, "trajectory_correction_selection.csv"), index=False
    )
    print(json.dumps({key: result[key] for key in ("selected_action", "gate")}, indent=2))
    print(f"selection -> {output_path}")


if __name__ == "__main__":
    main()
