"""Apply the preregistered tuned-CFG development selector exactly once."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from compare_actions import holm_adjust, prompt_sign_flip_pvalue, validate_pairing
from s7_provenance import (
    sha256_file,
    validate_run_contract,
    validate_scores_against_manifest,
)
from validate_cfg_baseline_run import (
    BASELINE_ACTION_ID,
    CFG_ACTION_IDS,
    CFG_SCALES,
    validate as validate_cfg_run,
)


SELECTOR_VERSION = "tuned_cfg_selector_v1"
FROZEN_RULE = {
    "baseline_action_id": BASELINE_ACTION_ID,
    "primary_metric": "topiq_nr",
    "minimum_mean_delta": 0.005,
    "topiq_ci_level": 0.95,
    "topiq_holm_alpha": 0.05,
    "bootstrap": 10000,
    "randomizations": 100000,
    "seed": 2026,
    "guard_familywise_alpha": 0.05,
    "hpsv2_lower_min_delta": -0.005,
    "clip_cosine_lower_min_delta": -0.005,
    "clipped_fraction_upper_max_delta": 0.001,
    "mean_saturation_upper_max_delta": 0.005,
    "contrast_ratio_min": 0.95,
    "contrast_ratio_max": 1.05,
    "practical_tie_delta": 0.005,
    "tie_break": [
        "closest_to_cfg_7p5",
        "lower_cfg_scale",
        "registered_action_order",
    ],
}
METRICS = (
    "topiq_nr",
    "hpsv2",
    "clip_cosine",
    "clipped_fraction",
    "mean_saturation",
    "contrast_std",
)


@contextmanager
def _selection_lock(run_dir: str):
    """Exclude generation, scoring, and concurrent selection for one run."""
    run_lock_path = os.path.join(run_dir, ".generate.lock")
    selection_lock_path = os.path.join(run_dir, ".cfg_baseline_selection.lock")
    run_handle = open(run_lock_path, "a+")
    selection_handle = open(selection_lock_path, "a+")
    try:
        try:
            fcntl.flock(run_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(
                selection_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                f"generation, scoring, or CFG selection is already running for {run_dir}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(selection_handle.fileno(), fcntl.LOCK_UN)
            fcntl.flock(run_handle.fileno(), fcntl.LOCK_UN)
        finally:
            selection_handle.close()
            run_handle.close()


def _prepare_selection_outputs(output_path: str, csv_path: str) -> None:
    """Reject a completed selection and remove one interrupted half-result."""
    output_exists = os.path.exists(output_path)
    csv_exists = os.path.exists(csv_path)
    if output_exists and csv_exists:
        raise ValueError("CFG selection already exists; selection is intentionally one-shot")
    if output_exists:
        os.unlink(output_path)
    if csv_exists:
        os.unlink(csv_path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_verified_bytes(path: str, expected_sha256: str, label: str) -> bytes:
    """Read one immutable snapshot and bind it to the validator's audit hash."""
    with open(path, "rb") as handle:
        payload = handle.read()
    if _sha256_bytes(payload) != expected_sha256:
        raise RuntimeError(f"{label} changed after CFG run validation")
    return payload


def _require_unchanged_snapshot(before: bytes, after: bytes, label: str) -> None:
    if before != after:
        raise RuntimeError(f"{label} changed during CFG run validation")


def _load_jsonl_bytes(payload: bytes, label: str) -> list[dict]:
    try:
        return [json.loads(line) for line in payload.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSONL: {exc}") from exc


def _atomic_write_bytes(path: str, payload: bytes) -> None:
    """Publish bytes through a unique temporary file in the destination directory."""
    directory = os.path.dirname(path) or "."
    prefix = f".{os.path.basename(path)}."
    descriptor, temporary = tempfile.mkstemp(
        dir=directory, prefix=prefix, suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _complete_pivot(
    frame: pd.DataFrame, metric: str, actions: Sequence[str]
) -> pd.DataFrame:
    if frame.duplicated(["prompt_index", "seed", "action_id"]).any():
        raise ValueError("duplicate prompt/seed/action rows make CFG selection ambiguous")
    pivot = frame.pivot(
        index=["prompt_index", "seed"], columns="action_id", values=metric
    )
    expected_index = pd.MultiIndex.from_product(
        [sorted(frame["prompt_index"].unique()), sorted(frame["seed"].unique())],
        names=["prompt_index", "seed"],
    )
    pivot = pivot.reindex(index=expected_index, columns=list(actions))
    if pivot.isna().any().any():
        raise ValueError(f"incomplete prompt x seed x action design for {metric}")
    values = pivot.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values found for {metric}")
    return pivot


def _crossed_bootstrap_means(
    values: pd.Series, *, n_boot: int, seed: int
) -> np.ndarray:
    matrix = values.unstack("seed")
    if matrix.isna().any().any():
        raise ValueError("crossed bootstrap requires a complete prompt x seed matrix")
    array = matrix.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("crossed bootstrap received non-finite values")
    rng = np.random.default_rng(seed)
    prompt_indices = rng.integers(0, array.shape[0], size=(n_boot, array.shape[0]))
    seed_indices = rng.integers(0, array.shape[1], size=(n_boot, array.shape[1]))
    result = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        result[index] = array[
            np.ix_(prompt_indices[index], seed_indices[index])
        ].mean()
    if not np.isfinite(result).all():
        raise ValueError("crossed bootstrap produced non-finite values")
    return result


def _validate_action_provenance(
    frame: pd.DataFrame,
    registered_actions: Mapping[str, Mapping[str, Any]],
) -> None:
    if "action" not in frame:
        raise ValueError("manifest lacks normalized action provenance")
    for action_id, expected in registered_actions.items():
        rows = frame[frame["action_id"].astype(str) == action_id]
        if rows.empty:
            raise ValueError(f"registered action {action_id!r} is absent")
        for observed in rows["action"]:
            if observed != expected:
                raise ValueError(f"{action_id}: normalized action provenance drifted")


def select(
    frame: pd.DataFrame,
    *,
    action_order: Sequence[str] = CFG_ACTION_IDS,
    action_scales: Mapping[str, float] | None = None,
    registered_actions: Mapping[str, Mapping[str, Any]] | None = None,
    bootstrap: int = 10000,
    randomizations: int = 100000,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Return the frozen scale decision and complete candidate gate table."""
    if bootstrap <= 0 or randomizations <= 0:
        raise ValueError("bootstrap and randomizations must be positive")
    validate_pairing(frame)
    required = {"action_id", "prompt_index", "seed", "device", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing CFG selection columns: {missing}")
    ordered = [str(value) for value in action_order]
    if ordered != list(CFG_ACTION_IDS):
        raise ValueError("CFG selector requires the frozen ordered action grid")
    available = set(frame["action_id"].astype(str))
    if available != set(ordered):
        raise ValueError(
            f"observed CFG actions {sorted(available)} differ from {ordered}"
        )
    scales = dict(action_scales or zip(CFG_ACTION_IDS, CFG_SCALES))
    if list(scales) != ordered or [float(scales[key]) for key in ordered] != list(
        CFG_SCALES
    ):
        raise ValueError("CFG selector scale mapping differs from registration")
    if registered_actions is not None:
        _validate_action_provenance(frame, registered_actions)

    pivots = {metric: _complete_pivot(frame, metric, ordered) for metric in METRICS}
    if (pivots["contrast_std"] <= 0).any().any():
        raise ValueError("contrast_std must be strictly positive for log-ratio guards")

    candidates = [action for action in ordered if action != BASELINE_ACTION_ID]
    guard_alpha = FROZEN_RULE["guard_familywise_alpha"] / len(candidates)
    rows = []
    topiq_pvalues = []
    for action_index, action in enumerate(candidates):
        deltas = {
            metric: pivots[metric][action] - pivots[metric][BASELINE_ACTION_ID]
            for metric in METRICS
            if metric != "contrast_std"
        }
        log_contrast_ratio = np.log(
            pivots["contrast_std"][action]
            / pivots["contrast_std"][BASELINE_ACTION_ID]
        )
        distributions = {}
        for metric_index, (metric, delta) in enumerate(deltas.items()):
            distributions[metric] = _crossed_bootstrap_means(
                delta,
                n_boot=bootstrap,
                seed=seed + action_index * 100 + metric_index,
            )
        contrast_distribution = _crossed_bootstrap_means(
            log_contrast_ratio,
            n_boot=bootstrap,
            seed=seed + action_index * 100 + len(deltas),
        )
        topiq_ci = np.quantile(distributions["topiq_nr"], [0.025, 0.975])
        pvalue = prompt_sign_flip_pvalue(
            deltas["topiq_nr"],
            n_random=randomizations,
            seed=seed + action_index,
        )
        topiq_pvalues.append(pvalue)
        row = {
            "action": action,
            "cfg_scale": float(scales[action]),
            "order": ordered.index(action),
            "topiq_mean_delta": float(deltas["topiq_nr"].mean()),
            "topiq_ci_low": float(topiq_ci[0]),
            "topiq_ci_high": float(topiq_ci[1]),
            "topiq_p_sign_flip": float(pvalue),
            "hpsv2_mean_delta": float(deltas["hpsv2"].mean()),
            "hpsv2_lower_bound": float(
                np.quantile(distributions["hpsv2"], guard_alpha)
            ),
            "clip_cosine_mean_delta": float(deltas["clip_cosine"].mean()),
            "clip_cosine_lower_bound": float(
                np.quantile(distributions["clip_cosine"], guard_alpha)
            ),
            "clipped_fraction_mean_delta": float(
                deltas["clipped_fraction"].mean()
            ),
            "clipped_fraction_upper_bound": float(
                np.quantile(distributions["clipped_fraction"], 1.0 - guard_alpha)
            ),
            "mean_saturation_mean_delta": float(
                deltas["mean_saturation"].mean()
            ),
            "mean_saturation_upper_bound": float(
                np.quantile(distributions["mean_saturation"], 1.0 - guard_alpha)
            ),
            "contrast_geometric_mean_ratio": float(
                math.exp(float(log_contrast_ratio.mean()))
            ),
            "contrast_ratio_lower_bound": float(
                math.exp(float(np.quantile(contrast_distribution, guard_alpha)))
            ),
            "contrast_ratio_upper_bound": float(
                math.exp(
                    float(np.quantile(contrast_distribution, 1.0 - guard_alpha))
                )
            ),
        }
        rows.append(row)

    adjusted = holm_adjust(topiq_pvalues)
    for row, adjusted_p in zip(rows, adjusted):
        row["topiq_p_holm"] = float(adjusted_p)
        row["passes_gate"] = bool(
            row["topiq_mean_delta"] >= FROZEN_RULE["minimum_mean_delta"]
            and row["topiq_ci_low"] > 0.0
            and row["topiq_p_holm"] < FROZEN_RULE["topiq_holm_alpha"]
            and row["hpsv2_lower_bound"]
            >= FROZEN_RULE["hpsv2_lower_min_delta"]
            and row["clip_cosine_lower_bound"]
            >= FROZEN_RULE["clip_cosine_lower_min_delta"]
            and row["clipped_fraction_upper_bound"]
            <= FROZEN_RULE["clipped_fraction_upper_max_delta"]
            and row["mean_saturation_upper_bound"]
            <= FROZEN_RULE["mean_saturation_upper_max_delta"]
            and row["contrast_ratio_lower_bound"]
            >= FROZEN_RULE["contrast_ratio_min"]
            and row["contrast_ratio_upper_bound"]
            <= FROZEN_RULE["contrast_ratio_max"]
        )

    passing = [row for row in rows if row["passes_gate"]]
    selected = BASELINE_ACTION_ID
    decision = "null_route"
    fallback_reason = "no_nondefault_scale_passed_the_frozen_gate"
    if passing:
        best_mean = max(row["topiq_mean_delta"] for row in passing)
        tied = [
            row
            for row in passing
            if best_mean - row["topiq_mean_delta"]
            <= FROZEN_RULE["practical_tie_delta"]
        ]
        winner = min(
            tied,
            key=lambda row: (
                abs(row["cfg_scale"] - float(scales[BASELINE_ACTION_ID])),
                row["cfg_scale"],
                row["order"],
            ),
        )
        selected = str(winner["action"])
        decision = "selected_nondefault_scale"
        fallback_reason = None

    return {
        "schema": "tuned_cfg_selection_v1",
        "selector_version": SELECTOR_VERSION,
        "selected_action": selected,
        "selected_cfg_scale": float(scales[selected]),
        "decision": decision,
        "fallback_reason": fallback_reason,
        "baseline_action": BASELINE_ACTION_ID,
        "action_order": ordered,
        "candidate_actions": candidates,
        "rule": dict(FROZEN_RULE),
        "guard_one_sided_alpha_per_candidate": float(guard_alpha),
        "bootstrap": int(bootstrap),
        "randomizations": int(randomizations),
        "seed": int(seed),
        "n_prompts": int(frame["prompt_index"].nunique()),
        "n_seeds": int(frame["seed"].nunique()),
        "n_pairs": int(
            frame[["prompt_index", "seed"]].drop_duplicates().shape[0]
        ),
        "rows": rows,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _require_frozen_rule(config: Mapping[str, Any]) -> None:
    if config.get("selection") != FROZEN_RULE:
        raise ValueError("CFG selection rule differs from tuned_cfg_selector_v1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--prompts", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output_path = os.path.join(run_dir, "cfg_baseline_selection.json")
    csv_path = os.path.join(run_dir, "cfg_baseline_selection.csv")
    with _selection_lock(run_dir):
        _prepare_selection_outputs(output_path, csv_path)
        config_path = os.path.join(run_dir, "config.json")
        manifest_path = os.path.join(run_dir, "manifest.jsonl")
        scores_path = os.path.join(run_dir, "scores.jsonl")
        with open(manifest_path, "rb") as handle:
            manifest_before_validation = handle.read()
        with open(scores_path, "rb") as handle:
            scores_before_validation = handle.read()

        audit = validate_cfg_run(run_dir, args.actions, args.prompts, "scores")

        with open(config_path, "rb") as handle:
            config_bytes = handle.read()
        with open(args.actions, "rb") as handle:
            actions_bytes = handle.read()
        with open(args.prompts, "rb") as handle:
            prompts_bytes = handle.read()
        manifest_bytes = _read_verified_bytes(
            manifest_path, audit["manifest_sha256"], "manifest.jsonl"
        )
        scores_bytes = _read_verified_bytes(
            scores_path, audit["scores_sha256"], "scores.jsonl"
        )
        _require_unchanged_snapshot(
            manifest_before_validation, manifest_bytes, "manifest.jsonl"
        )
        _require_unchanged_snapshot(
            scores_before_validation, scores_bytes, "scores.jsonl"
        )

        run_config = json.loads(config_bytes)
        if validate_run_contract(run_config) != audit["run_contract_sha256"]:
            raise RuntimeError("run config changed after CFG run validation")
        if _sha256_bytes(actions_bytes) != run_config.get("actions_sha256"):
            raise RuntimeError("actions YAML changed after CFG run validation")
        if _sha256_bytes(prompts_bytes) != run_config.get("prompts_sha256"):
            raise RuntimeError("prompts CSV changed after CFG run validation")
        actions_config = yaml.safe_load(actions_bytes) or {}
        _require_frozen_rule(actions_config)

        manifest = _load_jsonl_bytes(manifest_bytes, "manifest.jsonl")
        scores = _load_jsonl_bytes(scores_bytes, "scores.jsonl")
        validate_scores_against_manifest(manifest, scores)
        manifest_frame = pd.DataFrame(manifest)
        scores_frame = pd.DataFrame(scores)
        score_columns = [
            column
            for column in scores_frame
            if column == "id" or column not in manifest_frame
        ]
        frame = manifest_frame.merge(
            scores_frame[score_columns], on="id", how="inner", validate="one_to_one"
        )
        if len(frame) != len(manifest):
            raise ValueError("inner join dropped registered CFG rows")
        run_actions = {
            str(action["id"]): action for action in run_config.get("actions", [])
        }
        action_scales = {
            action_id: float(run_actions[action_id]["cfg_scale"])
            for action_id in CFG_ACTION_IDS
        }
        result = select(
            frame,
            action_order=CFG_ACTION_IDS,
            action_scales=action_scales,
            registered_actions=run_actions,
            bootstrap=FROZEN_RULE["bootstrap"],
            randomizations=FROZEN_RULE["randomizations"],
            seed=FROZEN_RULE["seed"],
        )
        result["provenance"] = {
            "run_dir": run_dir,
            "run_config_sha256": _sha256_bytes(config_bytes),
            "run_contract_sha256": audit["run_contract_sha256"],
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "scores_sha256": _sha256_bytes(scores_bytes),
            "actions_path": os.path.abspath(args.actions),
            "actions_sha256": _sha256_bytes(actions_bytes),
            "prompts_path": os.path.abspath(args.prompts),
            "prompts_sha256": _sha256_bytes(prompts_bytes),
            "validator_script_sha256": sha256_file(
                os.path.join(os.path.dirname(__file__), "validate_cfg_baseline_run.py")
            ),
            "selector_script_sha256": sha256_file(__file__),
            "selector_git_commit": _git_commit(),
        }
        json_payload = (
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        csv_payload = pd.DataFrame(result["rows"]).to_csv(index=False).encode("utf-8")
        _atomic_write_bytes(output_path, json_payload)
        _atomic_write_bytes(csv_path, csv_payload)
    print(
        json.dumps(
            {
                "selected_action": result["selected_action"],
                "selected_cfg_scale": result["selected_cfg_scale"],
                "decision": result["decision"],
            },
            indent=2,
        )
    )
    print(f"selection -> {output_path}")


if __name__ == "__main__":
    main()
