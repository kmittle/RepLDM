"""Apply the preregistered LR-1 validation statistics without retuning gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from compare_actions import (
    crossed_bootstrap_ci,
    holm_adjust,
    prompt_sign_flip_pvalue,
    validate_pairing,
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metric_delta(
    frame: pd.DataFrame, metric: str, selected: str, comparator: str
) -> pd.Series:
    if frame.duplicated(["prompt_index", "seed", "action_id"]).any():
        raise ValueError("validation manifest contains duplicate design cells")
    pivot = frame.pivot(
        index=["prompt_index", "seed"], columns="action_id", values=metric
    )
    missing = [action for action in (selected, comparator) if action not in pivot]
    if missing:
        raise ValueError(f"validation metric {metric!r} lacks actions {missing}")
    paired = pivot[[selected, comparator]]
    if paired.isna().any().any():
        raise ValueError(f"validation metric {metric!r} has incomplete pairs")
    values = paired.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"validation metric {metric!r} contains non-finite values")
    return paired[selected] - paired[comparator]


def paired_statistics(
    delta: pd.Series, *, n_boot: int, n_random: int, seed: int
) -> Dict[str, float]:
    ci_low, ci_high = crossed_bootstrap_ci(delta, n_boot=n_boot, seed=seed)
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_sign_flip": float(
            prompt_sign_flip_pvalue(delta, n_random=n_random, seed=seed)
        ),
        "n_prompts": int(delta.index.get_level_values("prompt_index").nunique()),
        "n_seeds": int(delta.index.get_level_values("seed").nunique()),
        "n_pairs": int(len(delta)),
    }


def evaluate_validation(
    frame: pd.DataFrame, frozen: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the statistical LR-1 decision; qualitative review stays separate."""
    validate_pairing(frame)
    requirements = frozen.get("validation_requirements", {})
    selected = str(frozen.get("selected_action", ""))
    if not selected or selected == "no_ag":
        raise ValueError("frozen validation config lacks a non-baseline winner")
    n_boot = int(requirements.get("bootstrap", 0))
    n_random = int(requirements.get("randomizations", 0))
    base_seed = int(requirements.get("seed", 0))
    if n_boot <= 0 or n_random <= 0:
        raise ValueError("validation resampling counts must be positive")

    primary_metric = str(requirements.get("primary_metric", ""))
    comparators = requirements.get("primary_comparators")
    if not isinstance(comparators, dict) or not comparators:
        raise ValueError("validation must register primary comparators")
    primary_rows = []
    for index, (comparator, minimum_delta) in enumerate(comparators.items()):
        delta = metric_delta(frame, primary_metric, selected, str(comparator))
        row = {
            "metric": primary_metric,
            "selected": selected,
            "comparator": str(comparator),
            "minimum_delta": float(minimum_delta),
            **paired_statistics(
                delta,
                n_boot=n_boot,
                n_random=n_random,
                seed=base_seed + index,
            ),
        }
        primary_rows.append(row)
    adjusted = holm_adjust([row["p_sign_flip"] for row in primary_rows])
    p_threshold = float(requirements.get("require_holm_p_below", 0.0))
    require_positive_ci = bool(
        requirements.get("require_positive_crossed_bootstrap_ci", False)
    )
    for row, adjusted_p in zip(primary_rows, adjusted):
        row["p_holm_within_primary"] = float(adjusted_p)
        row["passed"] = bool(
            row["mean_delta"] >= row["minimum_delta"]
            and (not require_positive_ci or row["ci95_low"] > 0.0)
            and row["p_holm_within_primary"] < p_threshold
        )

    reference = str(requirements.get("noninferiority_reference", "no_ag"))
    use_ci_low = requirements.get("noninferiority_uses_ci_low") is True
    if not use_ci_low:
        raise ValueError("validation non-inferiority must use the CI lower bound")
    noninferiority_rows = []
    for offset, (metric, margin_key) in enumerate(
        (("clip_cosine", "clip_cosine_min_delta"), ("hpsv2", "hpsv2_min_delta"))
    ):
        margin = float(requirements[margin_key])
        stats = paired_statistics(
            metric_delta(frame, metric, selected, reference),
            n_boot=n_boot,
            n_random=n_random,
            seed=base_seed + 100 + offset,
        )
        noninferiority_rows.append(
            {
                "metric": metric,
                "selected": selected,
                "comparator": reference,
                "margin": margin,
                **stats,
                "passed": bool(stats["ci95_low"] >= margin),
            }
        )

    guard_rows = []
    for metric, threshold_key in (
        ("clipped_fraction", "clipped_fraction_max_delta"),
        ("mean_saturation", "mean_saturation_max_delta"),
    ):
        threshold = float(requirements[threshold_key])
        delta = metric_delta(frame, metric, selected, reference)
        mean_delta = float(delta.mean())
        guard_rows.append(
            {
                "metric": metric,
                "selected": selected,
                "comparator": reference,
                "maximum_delta": threshold,
                "mean_delta": mean_delta,
                "passed": bool(math.isfinite(mean_delta) and mean_delta <= threshold),
            }
        )

    statistical_pass = bool(
        all(row["passed"] for row in primary_rows)
        and all(row["passed"] for row in noninferiority_rows)
        and all(row["passed"] for row in guard_rows)
    )
    return {
        "schema": "latent_renderer_validation_gate_v1",
        "selected_action": selected,
        "statistical_pass": statistical_pass,
        "qualitative_review_required": True,
        "validation_pass": False,
        "decision": "qualitative_review_required" if statistical_pass else "close_lr1",
        "primary": primary_rows,
        "noninferiority": noninferiority_rows,
        "guards": guard_rows,
        "qualitative_montage": requirements.get("qualitative_montage", {}),
    }


def validate_audit_provenance(
    audit: Dict[str, Any], run_dir: str | os.PathLike[str], frozen_path: str
) -> None:
    if audit.get("passed") is not True or audit.get("split_role") != "validation_confirmation":
        raise ValueError("a passing validation-confirmation run audit is required")
    provenance = audit.get("provenance", {})
    expected = {
        "manifest_sha256": sha256_file(Path(run_dir) / "manifest.jsonl"),
        "scores_sha256": sha256_file(Path(run_dir) / "scores.jsonl"),
        "source_actions_sha256": sha256_file(frozen_path),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"run audit provenance mismatch for {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--frozen_actions", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    with open(args.frozen_actions) as handle:
        frozen = yaml.safe_load(handle) or {}
    with open(args.audit) as handle:
        audit = json.load(handle)
    validate_audit_provenance(audit, args.run_dir, args.frozen_actions)
    manifest = pd.DataFrame(load_jsonl(Path(args.run_dir) / "manifest.jsonl"))
    scores = pd.DataFrame(load_jsonl(Path(args.run_dir) / "scores.jsonl"))
    score_columns = [column for column in scores if column == "id" or column not in manifest]
    frame = manifest.merge(scores[score_columns], on="id", how="inner")
    if len(frame) != len(manifest):
        raise ValueError("validation scores are incomplete")
    result = evaluate_validation(frame, frozen)
    result["provenance"] = {
        "audit_sha256": sha256_file(args.audit),
        "frozen_actions_sha256": sha256_file(args.frozen_actions),
        "manifest_sha256": sha256_file(Path(args.run_dir) / "manifest.jsonl"),
        "scores_sha256": sha256_file(Path(args.run_dir) / "scores.jsonl"),
    }
    output = Path(args.output) if args.output else Path(args.run_dir) / "validation_gate.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"validation gate -> {output}")


if __name__ == "__main__":
    main()
