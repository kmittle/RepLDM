"""Apply the preregistered LR-1 train-split action-selection rule.

The selector deliberately does not read TOPIQ-NR. It is a small audit tool for
the fixed-action search, not a general hyper-parameter optimizer. Validation and
test runs must be scored with a new invocation after this file emits one frozen
action.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from compare_actions import crossed_bootstrap_ci, validate_pairing


DEFAULT_CLIP_FLOOR = -0.005
DEFAULT_CLIP_MAX = 0.001
DEFAULT_SATURATION_MAX = 0.005
DEFAULT_MEAN_ERROR_MAX = 1e-4
DEFAULT_VARIANCE_ERROR_MAX = 1e-3


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_train_design(
    prompts: pd.DataFrame, frame: pd.DataFrame, forbidden_seeds: Sequence[int]
) -> None:
    """Reject split or seed leakage before any action score is selected."""
    if "split" not in prompts or set(prompts["split"].astype(str)) != {"train"}:
        raise ValueError("action selection requires a prompt CSV explicitly marked split=train")
    prompt_ids = set(int(value) for value in prompts["index"])
    observed_ids = set(int(value) for value in frame["prompt_index"])
    if observed_ids != prompt_ids:
        raise ValueError("manifest prompt ids do not exactly match the train split")
    if "prompt" not in frame:
        raise ValueError("manifest lacks prompt text required for split verification")
    expected_text = {
        int(row["index"]): str(row["TEXT"]) for _, row in prompts.iterrows()
    }
    observed_text = frame.groupby("prompt_index")["prompt"].agg(
        lambda values: set(map(str, values))
    )
    if any(observed_text.get(index, set()) != {text} for index, text in expected_text.items()):
        raise ValueError("manifest prompt text does not exactly match the train split")
    leaked = sorted(set(int(value) for value in frame["seed"]) & set(forbidden_seeds))
    if leaked:
        raise ValueError(
            f"train selection contains final-test seeds {leaked}; regenerate with search-only seeds"
        )


def _action_order(run_dir: str, frame: pd.DataFrame) -> List[str]:
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as handle:
            config = json.load(handle)
        ordered = [str(action["id"]) for action in config.get("actions", [])]
        ordered.extend(action for action in frame["action_id"].unique() if action not in ordered)
        return ordered
    return list(frame["action_id"].drop_duplicates())


def _complete_metric_pivot(
    frame: pd.DataFrame, metric: str, actions: Sequence[str]
) -> pd.DataFrame:
    if frame.duplicated(["prompt_index", "seed", "action_id"]).any():
        raise ValueError("duplicate prompt/seed/action rows make selection ambiguous")
    pivot = frame.pivot(
        index=["prompt_index", "seed"], columns="action_id", values=metric
    ).reindex(columns=list(actions))
    expected = pd.MultiIndex.from_product(
        [sorted(frame["prompt_index"].unique()), sorted(frame["seed"].unique())],
        names=["prompt_index", "seed"],
    )
    pivot = pivot.reindex(expected)
    if pivot.isna().any().any():
        raise ValueError(f"incomplete prompt x seed x action design for {metric}")
    if not np.isfinite(pivot.to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite values found for {metric}")
    return pivot


def _finite_renderer_diagnostics(frame: pd.DataFrame, action: str) -> bool:
    rows = frame[frame["action_id"] == action]
    for _, row in rows.iterrows():
        diagnostics = row.get("latent_renderer_diagnostics")
        if not isinstance(diagnostics, dict):
            return False
        for key in ("update_ratio", "mean_error", "variance_error"):
            values = diagnostics.get(key)
            if not isinstance(values, list) or not values:
                return False
            if not all(math.isfinite(float(value)) for value in values):
                return False
        if max(map(abs, map(float, diagnostics["mean_error"]))) > DEFAULT_MEAN_ERROR_MAX:
            return False
        if max(map(abs, map(float, diagnostics["variance_error"]))) > DEFAULT_VARIANCE_ERROR_MAX:
            return False
        action_record = row.get("action")
        if isinstance(action_record, dict):
            bound = action_record.get("max_update_ratio")
            if bound is not None and any(
                float(value) > float(bound) + 1e-6
                for value in diagnostics["update_ratio"]
            ):
                return False
    return True


def _mean_update_ratio(frame: pd.DataFrame, action: str) -> float:
    values: List[float] = []
    for diagnostics in frame.loc[
        frame["action_id"] == action, "latent_renderer_diagnostics"
    ]:
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("update_ratio"), list):
            values.extend(float(value) for value in diagnostics["update_ratio"])
    return float(np.mean(values)) if values else math.inf


def select_fixed_action(
    frame: pd.DataFrame,
    *,
    action_order: Optional[Sequence[str]] = None,
    baseline: str = "no_ag",
    candidates: Optional[Iterable[str]] = None,
    clip_floor: float = DEFAULT_CLIP_FLOOR,
    clip_max: float = DEFAULT_CLIP_MAX,
    saturation_max: float = DEFAULT_SATURATION_MAX,
    bootstrap: int = 10000,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Return the one action allowed to proceed to validation."""
    if int(bootstrap) <= 0:
        raise ValueError("bootstrap must be positive")
    validate_pairing(frame)
    required = {"action_id", "prompt_index", "seed", "hpsv2", "clip_cosine",
                "clipped_fraction", "mean_saturation"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing selection columns: {missing}")
    available = list(frame["action_id"].drop_duplicates())
    if baseline not in available:
        raise ValueError(f"baseline {baseline!r} is missing")
    order = list(action_order or available)
    order.extend(action for action in available if action not in order)
    if candidates is None:
        # Controls such as the conference expert may be added to a resumed
        # manifest, but they are never candidates for this fixed-basis search.
        if "action_type" in frame:
            fixed_actions = set(
                frame.loc[frame["action_type"] == "latent_renderer_fixed", "action_id"]
            )
        else:
            fixed_actions = set(order)
        candidates = [
            action for action in order if action != baseline and action in fixed_actions
        ]
    candidates = list(dict.fromkeys(candidates))
    unknown = sorted(set(candidates + [baseline]) - set(available))
    if unknown:
        raise ValueError(f"unknown actions: {unknown}")
    actions = [baseline] + [action for action in candidates if action != baseline]
    pivots = {
        metric: _complete_metric_pivot(frame, metric, actions)
        for metric in ("hpsv2", "clip_cosine", "clipped_fraction", "mean_saturation")
    }
    rows = []
    for action in candidates:
        deltas = {metric: pivots[metric][action] - pivots[metric][baseline]
                  for metric in pivots}
        hps_ci = crossed_bootstrap_ci(deltas["hpsv2"], n_boot=bootstrap, seed=seed)
        mean = {metric: float(delta.mean()) for metric, delta in deltas.items()}
        finite = _finite_renderer_diagnostics(frame, action)
        eligible = bool(
            finite
            and mean["clip_cosine"] >= clip_floor
            and mean["clipped_fraction"] <= clip_max
            and mean["mean_saturation"] <= saturation_max
        )
        rows.append({
            "action": action,
            **{f"delta_{metric}": value for metric, value in mean.items()},
            "hpsv2_ci_low": float(hps_ci[0]),
            "hpsv2_ci_high": float(hps_ci[1]),
            "finite_diagnostics": finite,
            "eligible": eligible,
            "delta_update_ratio": _mean_update_ratio(frame, action),
            "order": order.index(action),
        })

    table = pd.DataFrame(rows)
    eligible = table[table["eligible"]].copy()
    if eligible.empty:
        selected = baseline
    else:
        # Pick the largest mean, then treat every overlapping HPS interval as
        # an unresolved tie and prefer the smaller update ratio.
        winner = eligible.loc[eligible["delta_hpsv2"].idxmax()]
        tied = eligible[
            (eligible["hpsv2_ci_low"] <= winner["hpsv2_ci_high"])
            & (eligible["hpsv2_ci_high"] >= winner["hpsv2_ci_low"])
        ]
        selected = str(
            tied.sort_values(["delta_update_ratio", "order"], na_position="last")
            .iloc[0]["action"]
        )
    return {
        "selected_action": selected,
        "baseline": baseline,
        "candidate_actions": candidates,
        "selection_metric": "hpsv2",
        "topiq_used_for_selection": False,
        "constraints": {
            "clip_cosine_min_delta": clip_floor,
            "clipped_fraction_max_delta": clip_max,
            "mean_saturation_max_delta": saturation_max,
            "mean_error_max": DEFAULT_MEAN_ERROR_MAX,
            "variance_error_max": DEFAULT_VARIANCE_ERROR_MAX,
        },
        "bootstrap": bootstrap,
        "seed": seed,
        "rows": table.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prompts", required=True, help="train-split CSV; never pass test here")
    parser.add_argument("--baseline", default="no_ag")
    parser.add_argument("--candidates", default="", help="comma-separated fixed candidates")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--forbidden_seeds",
        default="0,42,123",
        help="comma-separated final-test seeds that must not occur in train selection",
    )
    args = parser.parse_args()

    prompts = pd.read_csv(args.prompts)
    manifest = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "manifest.jsonl")))
    scores = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "scores.jsonl")))
    if manifest.empty or scores.empty:
        raise ValueError("manifest and scores must be complete before selection")
    prompt_ids = set(int(value) for value in prompts["index"])
    frame = manifest[manifest["prompt_index"].isin(prompt_ids)].copy()
    score_columns = [column for column in scores if column == "id" or column not in manifest]
    frame = frame.merge(scores[score_columns], on="id", how="inner")
    if len(frame) != len(manifest[manifest["prompt_index"].isin(prompt_ids)]):
        raise ValueError("scores are incomplete for the requested split")
    forbidden_seeds = [
        int(value) for value in args.forbidden_seeds.split(",") if value
    ]
    validate_train_design(prompts, frame, forbidden_seeds)
    config_path = os.path.join(args.run_dir, "config.json")
    action_order = None
    if os.path.exists(config_path):
        with open(config_path) as handle:
            action_order = [str(action["id"]) for action in json.load(handle).get("actions", [])]
    result = select_fixed_action(
        frame,
        action_order=action_order,
        baseline=args.baseline,
        candidates=[value for value in args.candidates.split(",") if value] or None,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    output_path = os.path.join(args.run_dir, "fixed_action_selection.json")
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    pd.DataFrame(result["rows"]).to_csv(
        os.path.join(args.run_dir, "fixed_action_selection.csv"), index=False
    )
    print(json.dumps({key: result[key] for key in ("selected_action", "candidate_actions", "constraints")}, indent=2))
    print(f"selection -> {output_path}")


if __name__ == "__main__":
    main()
