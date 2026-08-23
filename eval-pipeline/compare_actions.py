"""Paired action comparison with crossed prompt/seed bootstrap inference."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def crossed_bootstrap_ci(delta, n_boot=10000, seed=2026):
    """Resample prompt and seed clusters independently while preserving pairing."""
    matrix = delta.unstack("seed")
    if matrix.isna().any().any():
        raise ValueError("crossed bootstrap requires a complete prompt x seed matrix")
    values = matrix.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    prompt_indices = rng.integers(0, values.shape[0], size=(n_boot, values.shape[0]))
    seed_indices = rng.integers(0, values.shape[1], size=(n_boot, values.shape[1]))
    boot = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        boot[index] = values[np.ix_(prompt_indices[index], seed_indices[index])].mean()
    return tuple(np.quantile(boot, [0.025, 0.975]))


def prompt_sign_flip_pvalue(delta, n_random=100000, seed=2026):
    """Two-sided randomization test on prompt-level paired mean differences."""
    prompt_delta = delta.groupby(level="prompt_index").mean().to_numpy(dtype=float)
    observed = abs(prompt_delta.mean())
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    batch_size = 10000
    while completed < n_random:
        count = min(batch_size, n_random - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, len(prompt_delta)))
        exceed += int(np.sum(np.abs((signs * prompt_delta).mean(axis=1)) >= observed))
        completed += count
    return (exceed + 1) / (n_random + 1)


def holm_adjust(pvalues):
    """Holm family-wise-error correction in the original order."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def action_column(frame):
    if "action_id" in frame.columns and frame["action_id"].notna().all():
        return "action_id"
    frame["_action"] = frame["scale"].map(lambda value: f"scale_{value:.4f}")
    return "_action"


def validate_pairing(frame):
    """Reject comparisons whose actions were generated on different devices."""
    if "device" not in frame:
        raise ValueError("manifest lacks device metadata required for paired inference")
    if frame["device"].isna().any():
        raise ValueError("manifest has missing device metadata required for paired inference")
    device_counts = frame.groupby(["prompt_index", "seed"])["device"].nunique()
    invalid = device_counts[device_counts > 1]
    if not invalid.empty:
        raise ValueError(
            f"{len(invalid)} prompt/seed blocks span multiple devices; paired inference is invalid. "
            "Regenerate with grouped action assignment."
        )


def compare(frame, baseline, metrics, n_boot=10000, n_random=100000, seed=2026):
    validate_pairing(frame)
    acol = action_column(frame)
    actions = [action for action in sorted(frame[acol].unique()) if action != baseline]
    rows = []
    for metric_index, metric in enumerate(metrics):
        pivot = frame.pivot_table(
            index=["prompt_index", "seed"], columns=acol, values=metric, aggfunc="mean"
        )
        if baseline not in pivot:
            raise ValueError(f"baseline {baseline!r} is missing for metric {metric}")
        metric_rows = []
        for action_index, action in enumerate(actions):
            paired = pivot[[baseline, action]].dropna()
            delta = paired[action] - paired[baseline]
            ci_low, ci_high = crossed_bootstrap_ci(
                delta, n_boot=n_boot, seed=seed + metric_index * 1000 + action_index
            )
            metric_rows.append(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "action": action,
                    "mean_delta": float(delta.mean()),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "p_sign_flip": prompt_sign_flip_pvalue(
                        delta,
                        n_random=n_random,
                        seed=seed + metric_index * 1000 + action_index,
                    ),
                    "n_prompts": int(delta.index.get_level_values("prompt_index").nunique()),
                    "n_seeds": int(delta.index.get_level_values("seed").nunique()),
                    "n_pairs": int(len(delta)),
                }
            )
        adjusted = holm_adjust([row["p_sign_flip"] for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["p_holm_within_metric"] = float(adjusted_p)
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no non-baseline actions are available for comparison")
    result["p_holm"] = holm_adjust(result["p_sign_flip"].to_numpy())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--baseline", default="no_ag")
    parser.add_argument(
        "--metrics",
        default="imagereward,topiq_nr,clipped_fraction,mean_saturation",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--randomizations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    manifest = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "manifest.jsonl")))
    scores = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "scores.jsonl")))
    score_columns = [column for column in scores if column == "id" or column not in manifest]
    frame = manifest.merge(scores[score_columns], on="id", how="inner")
    metrics = [metric for metric in args.metrics.split(",") if metric]
    missing = [metric for metric in metrics if metric not in frame]
    if missing:
        raise ValueError(f"missing score columns: {missing}")

    result = compare(
        frame,
        args.baseline,
        metrics,
        n_boot=args.bootstrap,
        n_random=args.randomizations,
        seed=args.seed,
    )
    output_path = os.path.join(args.run_dir, "action_comparisons.csv")
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\ncomparisons -> {output_path}")


if __name__ == "__main__":
    main()
