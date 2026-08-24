"""Apply the preregistered LR-1 train-split action-selection rule.

The selector deliberately does not read TOPIQ-NR. It is a small audit tool for
the fixed-action search, not a general hyper-parameter optimizer. Validation and
test runs must be scored with a new invocation after this file emits one frozen
action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from compare_actions import crossed_bootstrap_ci, validate_pairing


DEFAULT_CLIP_FLOOR = -0.005
DEFAULT_CLIP_MAX = 0.001
DEFAULT_SATURATION_MAX = 0.005
DEFAULT_MEAN_ERROR_MAX = 1e-4
DEFAULT_VARIANCE_ERROR_MAX = 1e-3


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_registered_train_run(
    frame: pd.DataFrame,
    run_config: Dict[str, Any],
    source: Dict[str, Any],
    *,
    split_role: str = "train_search",
) -> Dict[str, Any]:
    """Verify that selection uses the complete preregistered action/seed grid."""
    if split_role != "train_search":
        raise ValueError("fixed-action selection is authorized only for train_search")
    split_seeds = source.get("split_seeds")
    if not isinstance(split_seeds, dict) or split_role not in split_seeds:
        raise ValueError("source YAML does not register train_search seeds")
    expected_seeds = [int(value) for value in split_seeds[split_role]]
    configured_seeds = [int(value) for value in run_config.get("seeds", [])]
    observed_seeds = sorted(set(int(value) for value in frame["seed"]))
    if configured_seeds != expected_seeds or observed_seeds != sorted(expected_seeds):
        raise ValueError(
            "run must use exactly the registered train_search seeds "
            f"{expected_seeds}; configured={configured_seeds}, observed={observed_seeds}"
        )

    source_actions = source.get("actions")
    run_actions = run_config.get("actions")
    if not isinstance(source_actions, list) or not source_actions:
        raise ValueError("source YAML must contain registered actions")
    if not isinstance(run_actions, list) or not run_actions:
        raise ValueError("run config must contain generated actions")
    source_ids = [str(action.get("id", "")) for action in source_actions]
    run_ids = [str(action.get("id", "")) for action in run_actions]
    if len(source_ids) != len(set(source_ids)) or not all(source_ids):
        raise ValueError("source YAML contains empty or duplicate action ids")
    if run_ids != source_ids:
        raise ValueError("run action order/set differs from the registered source YAML")
    observed_ids = set(map(str, frame["action_id"]))
    if observed_ids != set(source_ids):
        raise ValueError("manifest action set differs from the registered source YAML")

    provider_defaults = dict(source.get("latent_renderer_provider", {}) or {})
    for registered, generated in zip(source_actions, run_actions):
        action_id = str(registered["id"])
        if generated.get("type") != registered.get("type"):
            raise ValueError(f"{action_id}: generated action type differs from registration")
        if registered.get("type") != "latent_renderer_fixed":
            continue
        registered_coefficients = [float(value) for value in registered.get("coefficients", [])]
        generated_coefficients = [float(value) for value in generated.get("coefficients", [])]
        if generated_coefficients != registered_coefficients:
            raise ValueError(f"{action_id}: generated coefficients differ from registration")
        expected_provider = dict(provider_defaults)
        expected_provider.update(registered.get("provider", {}) or {})
        generated_provider = generated.get("latent_renderer_provider", {}) or {}
        for key, value in expected_provider.items():
            if generated_provider.get(key) != value:
                raise ValueError(
                    f"{action_id}: generated provider field {key!r} differs from registration"
                )

    expected_cutoffs = [float(value) for value in source.get("frequency_band_cutoffs", [])]
    configured_cutoffs = [float(value) for value in run_config.get("frequency_band_cutoffs", [])]
    if configured_cutoffs != expected_cutoffs:
        raise ValueError("run frequency-band cutoffs differ from registration")
    candidate_actions = [
        str(action["id"])
        for action in source_actions
        if action.get("type") == "latent_renderer_fixed"
    ]
    return {
        "source_schema": str(source.get("schema", "")),
        "source_experiment_id": str(source.get("experiment_id", "")),
        "split_role": split_role,
        "seeds": expected_seeds,
        "action_ids": source_ids,
        "candidate_actions": candidate_actions,
    }


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
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--source_actions",
        default="eval-pipeline/configs/latent_renderer_fixed_lr1.yaml",
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
    with open(args.source_actions) as handle:
        source = yaml.safe_load(handle) or {}
    forbidden_seeds = [int(value) for value in source.get("split_seeds", {}).get("test_final", [])]
    validate_train_design(prompts, frame, forbidden_seeds)
    config_path = os.path.join(args.run_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError("run config.json is required for registered action selection")
    with open(config_path) as handle:
        run_config = json.load(handle)
    registration = validate_registered_train_run(frame, run_config, source)
    recorded_prompts = run_config.get("prompts_csv")
    if recorded_prompts and os.path.realpath(recorded_prompts) != os.path.realpath(args.prompts):
        raise ValueError("run config prompts_csv differs from --prompts")
    recorded_action_hash = run_config.get("actions_sha256")
    source_hash = sha256_file(args.source_actions)
    if recorded_action_hash is not None and recorded_action_hash != source_hash:
        raise ValueError("run config action YAML hash differs from --source_actions")
    recorded_prompt_hash = run_config.get("prompts_sha256")
    prompt_hash = sha256_file(args.prompts)
    if recorded_prompt_hash is not None and recorded_prompt_hash != prompt_hash:
        raise ValueError("run config prompt CSV hash differs from --prompts")
    result = select_fixed_action(
        frame,
        action_order=registration["action_ids"],
        baseline=args.baseline,
        candidates=registration["candidate_actions"],
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    result["registration"] = registration
    result["provenance"] = {
        "run_config_path": os.path.abspath(config_path),
        "run_config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(os.path.join(args.run_dir, "manifest.jsonl")),
        "scores_sha256": sha256_file(os.path.join(args.run_dir, "scores.jsonl")),
        "prompts_path": os.path.abspath(args.prompts),
        "prompts_sha256": prompt_hash,
        "source_actions_path": os.path.abspath(args.source_actions),
        "source_actions_sha256": source_hash,
    }
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
