"""Turn a passed LR-1 validation plus blinded review into final-test authority."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


RESPONSE_COLUMNS = ("overall", "structure", "text", "counting", "position", "detail")
VALID_RESPONSES = {"a", "b", "tie", "na", "n/a", ""}
FINAL_TEST_SEEDS = [0, 42, 123]


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson_lower(wins: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return float("nan")
    rate = wins / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return (center - radius) / denominator


def _read_form(path: str | os.PathLike[str]) -> tuple[str, List[Dict[str, str]]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"reviewer_id", "pair_id", *RESPONSE_COLUMNS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"review form {path} lacks required columns")
        rows = [{key: str(value or "").strip().lower() for key, value in row.items()} for row in reader]
    reviewer_ids = {row["reviewer_id"] for row in rows}
    if len(reviewer_ids) != 1 or not next(iter(reviewer_ids), ""):
        raise ValueError(f"review form {path} must contain one non-empty reviewer_id")
    reviewer_id = next(iter(reviewer_ids))
    for row in rows:
        for column in RESPONSE_COLUMNS:
            if row[column] not in VALID_RESPONSES:
                raise ValueError(
                    f"review form {path}: invalid {column} response {row[column]!r}"
                )
    return reviewer_id, rows


def summarize_reviews(
    key: Dict[str, Any],
    review_paths: Iterable[str | os.PathLike[str]],
    montage_spec: Dict[str, Any],
) -> Dict[str, Any]:
    pairs = key.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("review key contains no pairs")
    key_by_id = {str(pair.get("pair_id")): pair for pair in pairs}
    if len(key_by_id) != len(pairs) or not all(key_by_id):
        raise ValueError("review key contains duplicate or empty pair ids")

    forms = []
    reviewer_ids = set()
    for path in review_paths:
        reviewer_id, rows = _read_form(path)
        if reviewer_id in reviewer_ids:
            raise ValueError(f"duplicate reviewer_id {reviewer_id!r}")
        reviewer_ids.add(reviewer_id)
        row_by_pair = {row["pair_id"]: row for row in rows}
        if len(row_by_pair) != len(rows) or set(row_by_pair) != set(key_by_id):
            raise ValueError(f"reviewer {reviewer_id!r} does not review exactly the frozen pairs")
        forms.append((reviewer_id, row_by_pair))

    min_reviewers = int(montage_spec.get("minimum_reviewers", 0))
    if len(forms) < min_reviewers:
        raise ValueError(f"need at least {min_reviewers} independent reviewers")
    selected = str(key.get("selected_action", ""))
    if not selected:
        raise ValueError("review key lacks selected action")

    def outcome(response: str, pair: Dict[str, Any]) -> int:
        if response not in {"a", "b"}:
            return 0
        selected_side = "a" if pair.get("left_action") == selected else "b"
        return 1 if response == selected_side else -1

    def summarize_column(column: str) -> Dict[str, Any]:
        wins = losses = ties = 0
        for _, row_by_pair in forms:
            for pair_id, pair in key_by_id.items():
                value = outcome(row_by_pair[pair_id][column], pair)
                if value > 0:
                    wins += 1
                elif value < 0:
                    losses += 1
                else:
                    ties += 1
        total = wins + losses
        return {
            "wins": wins,
            "losses": losses,
            "ties_or_na": ties,
            "decisive": total,
            "win_rate": wins / total if total else float("nan"),
            "wilson_ci_low": _wilson_lower(wins, total),
        }

    overall = summarize_column("overall")
    minimum_rate = float(montage_spec.get("minimum_overall_preference_rate", 0.55))
    ci_floor = float(montage_spec.get("overall_wilson_ci_low", 0.5))
    dimension_rate = float(montage_spec.get("dimension_minimum_win_rate", 0.5))
    dimensions = {
        column: summarize_column(column)
        for column in RESPONSE_COLUMNS
        if column != "overall"
    }
    positive_dimensions = sum(
        1
        for summary in dimensions.values()
        if math.isfinite(summary["win_rate"]) and summary["win_rate"] >= dimension_rate
    )
    minimum_dimensions = int(montage_spec.get("minimum_positive_dimensions", 0))
    passed = bool(
        overall["decisive"] > 0
        and overall["win_rate"] >= minimum_rate
        and overall["wilson_ci_low"] > ci_floor
        and positive_dimensions >= minimum_dimensions
    )
    return {
        "reviewers": sorted(reviewer_ids),
        "n_reviewers": len(reviewer_ids),
        "n_pairs": len(key_by_id),
        "selected_action": selected,
        "overall": overall,
        "dimensions": dimensions,
        "positive_dimensions": positive_dimensions,
        "minimum_positive_dimensions": minimum_dimensions,
        "passed": passed,
    }


def finalize(
    frozen: Dict[str, Any],
    gate: Dict[str, Any],
    key: Dict[str, Any],
    review_summary: Dict[str, Any],
    *,
    frozen_path: str,
    gate_path: str,
    key_path: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if gate.get("statistical_pass") is not True:
        raise ValueError("statistical validation gate did not pass")
    if review_summary.get("passed") is not True:
        raise ValueError("blinded review gate did not pass")
    if key.get("selected_action") != frozen.get("selected_action"):
        raise ValueError("review key winner differs from frozen validation winner")
    selected = str(frozen["selected_action"])
    actions = [dict(action) for action in frozen.get("actions", [])]
    if [str(action.get("id")) for action in actions] != [
        "no_ag",
        selected,
        "conference_expert",
        "matched_random",
    ]:
        raise ValueError("frozen validation actions do not match the registered four-action set")
    final_config = {
        "schema": "latent_renderer_actions_v1",
        "experiment_id": "lr1_fixed_final_test_v1",
        "status": "frozen_final_test",
        "selected_action": selected,
        "split_seeds": {"test_final": FINAL_TEST_SEEDS},
        "latent_renderer_provider": dict(frozen.get("latent_renderer_provider", {})),
        "frequency_band_cutoffs": list(frozen.get("frequency_band_cutoffs", [0.08, 0.25])),
        "validation_provenance": {
            "frozen_validation_sha256": sha256_file(frozen_path),
            "validation_gate_sha256": sha256_file(gate_path),
            "review_key_sha256": sha256_file(key_path),
            "review_summary": review_summary,
        },
        "actions": actions,
    }
    return final_config, {
        "schema": "latent_renderer_final_authorization_v1",
        "status": "authorized_final_test",
        "test_seeds": FINAL_TEST_SEEDS,
        "selected_action": selected,
        "validation_config_sha256": sha256_file(frozen_path),
        "validation_gate_sha256": sha256_file(gate_path),
        "review_key_sha256": sha256_file(key_path),
        "review_summary": review_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_actions", required=True)
    parser.add_argument("--validation_gate", required=True)
    parser.add_argument("--review_key", required=True)
    parser.add_argument("--review_forms", nargs="+", required=True)
    parser.add_argument("--output_actions", required=True)
    parser.add_argument("--output_authorization", required=True)
    args = parser.parse_args()
    with open(args.frozen_actions) as handle:
        frozen = yaml.safe_load(handle) or {}
    with open(args.validation_gate) as handle:
        gate = json.load(handle)
    with open(args.review_key) as handle:
        key = json.load(handle)
    montage_spec = frozen.get("validation_requirements", {}).get("qualitative_montage", {})
    review_summary = summarize_reviews(key, args.review_forms, montage_spec)
    final_config, authorization = finalize(
        frozen,
        gate,
        key,
        review_summary,
        frozen_path=args.frozen_actions,
        gate_path=args.validation_gate,
        key_path=args.review_key,
    )
    output_actions = Path(args.output_actions)
    output_actions.parent.mkdir(parents=True, exist_ok=True)
    output_actions.write_text(yaml.safe_dump(final_config, sort_keys=False))
    authorization["source_actions_sha256"] = sha256_file(output_actions)
    output_authorization = Path(args.output_authorization)
    output_authorization.parent.mkdir(parents=True, exist_ok=True)
    output_authorization.write_text(json.dumps(authorization, indent=2) + "\n")
    print(json.dumps(authorization, indent=2))


if __name__ == "__main__":
    main()
