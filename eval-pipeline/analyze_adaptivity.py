"""Estimate action-selection headroom without reusing the evaluation seed.

For every held-out seed, actions are selected on all remaining seeds. The
script compares a globally selected action, a per-prompt selected action, and
their difference on the held-out samples. This tests cross-seed consistency;
it does not estimate generalization to unseen prompts.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from compare_actions import (
    action_column,
    crossed_bootstrap_ci,
    holm_adjust,
    prompt_sign_flip_pvalue,
    validate_pairing,
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _complete_pivot(
    frame: pd.DataFrame,
    action_col: str,
    metric: str,
    candidates: Sequence[str],
) -> pd.DataFrame:
    if frame.duplicated(["prompt_index", "seed", action_col]).any():
        raise ValueError("duplicate prompt/seed/action rows make selection ambiguous")
    pivot = frame.pivot(
        index=["prompt_index", "seed"], columns=action_col, values=metric
    )
    expected_index = pd.MultiIndex.from_product(
        [sorted(frame["prompt_index"].unique()), sorted(frame["seed"].unique())],
        names=["prompt_index", "seed"],
    )
    pivot = pivot.reindex(index=expected_index, columns=candidates)
    if pivot.isna().any().any():
        missing = int(pivot.isna().sum().sum())
        raise ValueError(
            f"seed-CV requires a complete prompt x seed x action design; "
            f"found {missing} missing values for {metric}"
        )
    return pivot


def seed_cv_headroom(
    frame: pd.DataFrame,
    baseline: str,
    selection_metric: str,
    metrics: Sequence[str],
    candidates: Optional[Sequence[str]] = None,
    higher_is_better: bool = True,
    n_boot: int = 10000,
    n_random: int = 100000,
    seed: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return out-of-seed comparisons and the actions selected in each fold."""
    frame = frame.copy()
    validate_pairing(frame)
    action_col = action_column(frame)
    available = sorted(frame[action_col].unique())
    if baseline not in available:
        raise ValueError(f"baseline {baseline!r} is missing")

    if candidates:
        candidates = list(dict.fromkeys(candidates))
        missing = sorted(set(candidates) - set(available))
        if missing:
            raise ValueError(f"candidate actions are missing: {missing}")
    else:
        candidates = available
    if baseline not in candidates:
        candidates.append(baseline)
    candidates = sorted(candidates)
    candidate_spec = "|".join(candidates)
    selection_direction = "higher" if higher_is_better else "lower"

    metrics = list(dict.fromkeys(metrics))
    if not metrics:
        raise ValueError("at least one evaluation metric is required")
    requested_metrics = list(dict.fromkeys([selection_metric, *metrics]))
    missing_metrics = [metric for metric in requested_metrics if metric not in frame]
    if missing_metrics:
        raise ValueError(f"missing score columns: {missing_metrics}")

    seeds = sorted(frame["seed"].unique())
    if len(seeds) < 2:
        raise ValueError("seed-CV requires at least two seeds")

    selection_pivot = _complete_pivot(
        frame, action_col, selection_metric, candidates
    )
    objective = selection_pivot if higher_is_better else -selection_pivot
    selection_rows = []
    for held_out_seed in seeds:
        train = objective[
            objective.index.get_level_values("seed") != held_out_seed
        ]
        global_action = train.mean(axis=0).idxmax()
        prompt_actions = train.groupby(level="prompt_index").mean().idxmax(axis=1)
        for prompt_index in sorted(frame["prompt_index"].unique()):
            selection_rows.append(
                {
                    "held_out_seed": held_out_seed,
                    "prompt_index": prompt_index,
                    "baseline": baseline,
                    "selection_metric": selection_metric,
                    "selection_direction": selection_direction,
                    "candidate_actions": candidate_spec,
                    "global_action": global_action,
                    "per_prompt_action": prompt_actions.loc[prompt_index],
                }
            )
    selections = pd.DataFrame(selection_rows)

    result_rows = []
    for metric_index, metric in enumerate(metrics):
        pivot = _complete_pivot(frame, action_col, metric, candidates)
        indices = pd.MultiIndex.from_arrays(
            [selections["prompt_index"], selections["held_out_seed"]],
            names=["prompt_index", "seed"],
        )
        baseline_values = np.array(
            [pivot.loc[index, baseline] for index in indices], dtype=float
        )
        global_values = np.array(
            [
                pivot.loc[index, action]
                for index, action in zip(indices, selections["global_action"])
            ],
            dtype=float,
        )
        prompt_values = np.array(
            [
                pivot.loc[index, action]
                for index, action in zip(indices, selections["per_prompt_action"])
            ],
            dtype=float,
        )
        comparisons = {
            "global_static_vs_baseline": global_values - baseline_values,
            "per_prompt_vs_baseline": prompt_values - baseline_values,
            "per_prompt_vs_global": prompt_values - global_values,
        }
        metric_rows = []
        inference_seed = seed + metric_index * 1000
        for comparison, values in comparisons.items():
            delta = pd.Series(values, index=indices)
            ci_low, ci_high = crossed_bootstrap_ci(
                delta, n_boot=n_boot, seed=inference_seed
            )
            metric_rows.append(
                {
                    "baseline": baseline,
                    "selection_metric": selection_metric,
                    "selection_direction": selection_direction,
                    "candidate_actions": candidate_spec,
                    "metric": metric,
                    "comparison": comparison,
                    "mean_delta": float(delta.mean()),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "p_sign_flip": prompt_sign_flip_pvalue(
                        delta, n_random=n_random, seed=inference_seed
                    ),
                    "n_prompts": int(
                        delta.index.get_level_values("prompt_index").nunique()
                    ),
                    "n_seeds": int(delta.index.get_level_values("seed").nunique()),
                    "n_pairs": int(len(delta)),
                    "n_candidates": int(len(candidates)),
                    "n_bootstrap": int(n_boot),
                    "n_randomizations": int(n_random),
                    "inference_seed": int(inference_seed),
                }
            )
        adjusted = holm_adjust([row["p_sign_flip"] for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["p_holm_within_metric"] = float(adjusted_p)
            result_rows.append(row)

    result = pd.DataFrame(result_rows)
    result["p_holm"] = holm_adjust(result["p_sign_flip"].to_numpy())
    return result, selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--baseline", default="no_ag")
    parser.add_argument("--selection_metric", default="topiq_nr")
    parser.add_argument(
        "--metrics",
        default="topiq_nr,hpsv2,imagereward,clipped_fraction,mean_saturation",
    )
    parser.add_argument(
        "--actions",
        default="",
        help="comma-separated candidate subset; the baseline is always included",
    )
    parser.add_argument("--lower_is_better", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--randomizations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    manifest = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "manifest.jsonl")))
    scores = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "scores.jsonl")))
    if manifest["id"].duplicated().any() or scores["id"].duplicated().any():
        raise ValueError("manifest and scores must contain unique ids")
    score_columns = [
        column for column in scores if column == "id" or column not in manifest
    ]
    frame = manifest.merge(scores[score_columns], on="id", how="inner")
    if len(frame) != len(manifest):
        raise ValueError("scores are incomplete for the manifest")

    metrics = [metric for metric in args.metrics.split(",") if metric]
    candidates = [action for action in args.actions.split(",") if action] or None
    result, selections = seed_cv_headroom(
        frame,
        args.baseline,
        args.selection_metric,
        metrics,
        candidates=candidates,
        higher_is_better=not args.lower_is_better,
        n_boot=args.bootstrap,
        n_random=args.randomizations,
        seed=args.seed,
    )

    comparison_path = os.path.join(args.run_dir, "adaptivity_comparisons.csv")
    selection_path = os.path.join(args.run_dir, "adaptivity_selections.csv")
    result.to_csv(comparison_path, index=False)
    selections.to_csv(selection_path, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nglobal action by held-out seed:")
    print(
        selections[["held_out_seed", "global_action"]]
        .drop_duplicates()
        .to_string(index=False)
    )
    print("\nper-prompt action frequency:")
    print(selections["per_prompt_action"].value_counts().to_string())
    print(f"\ncomparisons -> {comparison_path}")
    print(f"selections  -> {selection_path}")


if __name__ == "__main__":
    main()
