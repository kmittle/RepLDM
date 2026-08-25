"""Paired action comparison with crossed prompt/seed bootstrap inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import product
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unique_registered_values(values: Sequence[Any], label: str) -> list[Any]:
    result = list(values)
    if not result:
        raise ValueError(f"registered {label} must not be empty")
    if any(pd.isna(value) for value in result):
        raise ValueError(f"registered {label} contain missing values")
    if len(result) != len(set(result)):
        raise ValueError(f"registered {label} contain duplicates")
    return result


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected_grid(
    run_dir: str, prompts_path: Optional[str] = None
) -> tuple[list[Any], list[Any], list[str]]:
    """Load the preregistered prompt, seed, and action axes for one run."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError("run config.json is required to establish the expected grid")
    with open(config_path) as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("run config.json must contain a mapping")

    registered_prompts_path = prompts_path or config.get("prompts_csv")
    if not isinstance(registered_prompts_path, str) or not registered_prompts_path:
        raise ValueError("run config lacks a registered prompts_csv")
    registered_prompts_path = os.path.abspath(registered_prompts_path)
    if not os.path.isfile(registered_prompts_path):
        raise ValueError(
            "registered prompt CSV is unavailable; pass --prompts with the same file"
        )
    prompt_hash = config.get("prompts_sha256")
    if prompt_hash is not None and _sha256_file(registered_prompts_path) != prompt_hash:
        raise ValueError("prompt CSV hash differs from the run registration")
    prompts = pd.read_csv(registered_prompts_path)
    if "index" not in prompts:
        raise ValueError("registered prompt CSV lacks the index column")
    prompt_indices = _unique_registered_values(
        prompts["index"].tolist(), "prompt indices"
    )

    contract = config.get("run_contract")
    contract = contract if isinstance(contract, dict) else {}
    seeds = config.get("seeds")
    if seeds is None:
        seeds = contract.get("seeds")
    if not isinstance(seeds, list):
        raise ValueError("run config lacks registered seeds")
    seeds = _unique_registered_values(seeds, "seeds")

    actions = config.get("actions")
    if actions is None:
        actions = contract.get("actions")
    if actions is not None:
        if not isinstance(actions, list) or any(
            not isinstance(action, Mapping) for action in actions
        ):
            raise ValueError("registered actions must be a list of mappings")
        action_ids = [action.get("id") for action in actions]
        if any(
            not isinstance(action_id, str) or not action_id
            for action_id in action_ids
        ):
            raise ValueError("every registered action must have a non-empty string id")
    else:
        scales = config.get("scales")
        if not isinstance(scales, list):
            raise ValueError("run config lacks registered actions or scales")
        action_ids = [f"scale_{float(value):.4f}" for value in scales]
    action_ids = _unique_registered_values(action_ids, "action ids")
    return prompt_indices, seeds, action_ids


def merge_manifest_scores(
    manifest: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    """Require exact score IDs before constructing a one-to-one analysis frame."""
    for frame, label in ((manifest, "manifest"), (scores, "scores")):
        if "id" not in frame:
            raise ValueError(f"{label} lacks id")
        if frame["id"].isna().any() or (frame["id"].astype(str) == "").any():
            raise ValueError(f"{label} contains empty score IDs")
        if frame["id"].duplicated().any():
            raise ValueError(f"{label} contains duplicate score IDs")
    manifest_ids = set(manifest["id"])
    score_ids = set(scores["id"])
    if manifest_ids != score_ids:
        missing = sorted(str(value) for value in manifest_ids - score_ids)
        extra = sorted(str(value) for value in score_ids - manifest_ids)
        raise ValueError(
            "score IDs do not exactly match manifest IDs: "
            f"{len(missing)} missing and {len(extra)} extra"
        )
    score_columns = [
        column for column in scores if column == "id" or column not in manifest
    ]
    return manifest.merge(
        scores[score_columns], on="id", how="inner", validate="one_to_one"
    )


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
    if "action_id" in frame.columns:
        if frame["action_id"].isna().any():
            raise ValueError("manifest contains missing action_id values")
        return "action_id"
    if "scale" not in frame.columns or frame["scale"].isna().any():
        raise ValueError("manifest lacks complete action_id or scale metadata")
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


def validate_expected_grid(
    frame: pd.DataFrame,
    expected_prompt_indices: Sequence[Any],
    expected_seeds: Sequence[Any],
    expected_actions: Sequence[str],
) -> str:
    """Require the exact preregistered prompt x seed x action product."""
    required = {"prompt_index", "seed"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"analysis frame lacks grid columns: {missing_columns}")
    if frame[["prompt_index", "seed"]].isna().any().any():
        raise ValueError("analysis frame contains missing prompt or seed values")
    action_col = action_column(frame)
    prompts = _unique_registered_values(expected_prompt_indices, "prompt indices")
    seeds = _unique_registered_values(expected_seeds, "seeds")
    actions = _unique_registered_values(expected_actions, "action ids")
    cell_columns = ["prompt_index", "seed", action_col]
    if frame.duplicated(cell_columns).any():
        raise ValueError("duplicate prompt_index/seed/action rows are not allowed")
    observed = set(frame[cell_columns].itertuples(index=False, name=None))
    expected = set(product(prompts, seeds, actions))
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        raise ValueError(
            "analysis frame differs from the registered prompt x seed x action grid: "
            f"{len(missing)} missing and {len(extra)} extra cells"
        )
    return action_col


def compare(
    frame: pd.DataFrame,
    baseline: str,
    metrics: Sequence[str],
    n_boot: int = 10000,
    n_random: int = 100000,
    seed: int = 2026,
    *,
    expected_prompt_indices: Sequence[Any],
    expected_seeds: Sequence[Any],
    expected_actions: Sequence[str],
) -> pd.DataFrame:
    acol = validate_expected_grid(
        frame, expected_prompt_indices, expected_seeds, expected_actions
    )
    validate_pairing(frame)
    if baseline not in expected_actions:
        raise ValueError(f"baseline {baseline!r} is absent from the registered actions")
    actions = [action for action in sorted(expected_actions) if action != baseline]
    expected_index = pd.MultiIndex.from_product(
        [expected_prompt_indices, expected_seeds], names=["prompt_index", "seed"]
    )
    rows = []
    for metric_index, metric in enumerate(metrics):
        if metric not in frame:
            raise ValueError(f"missing score column: {metric}")
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"metric {metric!r} contains missing or non-finite values")
        pivot = frame.assign(**{metric: numeric}).pivot(
            index=["prompt_index", "seed"], columns=acol, values=metric
        )
        pivot = pivot.reindex(index=expected_index, columns=expected_actions)
        if pivot.isna().any().any():
            raise ValueError(f"metric {metric!r} is incomplete on the registered grid")
        metric_rows = []
        for action_index, action in enumerate(actions):
            delta = pivot[action] - pivot[baseline]
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
    parser.add_argument(
        "--prompts",
        default=None,
        help="registered prompt CSV override when config.json's path is unavailable",
    )
    args = parser.parse_args()

    manifest = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "manifest.jsonl")))
    scores = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "scores.jsonl")))
    prompt_indices, seeds, actions = load_expected_grid(args.run_dir, args.prompts)
    frame = merge_manifest_scores(manifest, scores)
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
        expected_prompt_indices=prompt_indices,
        expected_seeds=seeds,
        expected_actions=actions,
    )
    output_path = os.path.join(args.run_dir, "action_comparisons.csv")
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\ncomparisons -> {output_path}")


if __name__ == "__main__":
    main()
