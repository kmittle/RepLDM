"""Shared, dependency-light provenance checks for the registered S7 runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


PROVENANCE_SCHEMA = "s7_trajectory_provenance_v1"


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values deterministically for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_sha256(path: str | os.PathLike[str]) -> str:
    return sha256_file(path)


def json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def action_sha256(action: Mapping[str, Any]) -> str:
    return json_sha256(dict(action))


def safe_child_path(root: str | os.PathLike[str], relative: str) -> Path:
    root_path = Path(root).resolve()
    path = (root_path / relative).resolve()
    if os.path.commonpath((str(root_path), str(path))) != str(root_path):
        raise ValueError(f"path escapes run directory: {relative!r}")
    return path


def validate_design_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_action_ids: Optional[Iterable[str]] = None,
    expected_seeds: Optional[Iterable[int]] = None,
    expected_prompt_indices: Optional[Iterable[int]] = None,
) -> Dict[tuple, Mapping[str, Any]]:
    """Validate unique IDs and the complete crossed prompt/seed/action grid."""
    records = list(rows)
    by_id: Dict[str, Mapping[str, Any]] = {}
    by_design: Dict[tuple, Mapping[str, Any]] = {}
    for row in records:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in by_id:
            raise ValueError(f"duplicate or empty record id: {row_id!r}")
        by_id[row_id] = row
        try:
            key = (
                int(row["prompt_index"]),
                int(row["seed"]),
                str(row["action_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"record lacks a valid design key: {row_id!r}") from exc
        if key in by_design:
            raise ValueError(f"duplicate design cell: {key}")
        by_design[key] = row

    action_ids = sorted({key[2] for key in by_design})
    if expected_action_ids is not None:
        action_ids = [str(value) for value in expected_action_ids]
        observed_actions = {key[2] for key in by_design}
        if observed_actions != set(action_ids):
            raise ValueError(
                f"observed action IDs {sorted(observed_actions)} differ from "
                f"registered IDs {action_ids}"
            )
    seed_values = sorted({key[1] for key in by_design})
    if expected_seeds is not None:
        seed_values = [int(value) for value in expected_seeds]
        observed_seeds = {key[1] for key in by_design}
        if observed_seeds != set(seed_values):
            raise ValueError(
                f"observed seeds {sorted(observed_seeds)} differ from registered seeds "
                f"{seed_values}"
            )
    prompt_values = sorted({key[0] for key in by_design})
    if expected_prompt_indices is not None:
        prompt_values = [int(value) for value in expected_prompt_indices]
        observed_prompts = {key[0] for key in by_design}
        if observed_prompts != set(prompt_values):
            raise ValueError(
                f"observed prompt indices {sorted(observed_prompts)} differ from "
                f"registered indices {prompt_values}"
            )
    expected = {
        (prompt, seed, action)
        for prompt in prompt_values
        for seed in seed_values
        for action in action_ids
    }
    if set(by_design) != expected:
        missing = len(expected - set(by_design))
        extra = len(set(by_design) - expected)
        raise ValueError(f"incomplete design: {missing} missing and {extra} extra cells")
    return by_design


def validate_sidecar(
    record: Mapping[str, Any],
    run_dir: str | os.PathLike[str],
    *,
    expected_task: Optional[Mapping[str, Any]] = None,
    expected_contract_sha256: Optional[str] = None,
) -> str:
    """Validate a generated sidecar and return the verified image hash."""
    row_id = str(record.get("id", ""))
    if not row_id:
        raise ValueError("sidecar has no id")
    image_path_value = record.get("image_path")
    if not isinstance(image_path_value, str) or not image_path_value:
        raise ValueError(f"{row_id}: image_path is missing")
    image_path = safe_child_path(run_dir, image_path_value)
    if not image_path.is_file():
        raise ValueError(f"{row_id}: image is missing: {image_path}")
    observed_hash = record.get("image_sha256")
    if not isinstance(observed_hash, str) or len(observed_hash) != 64:
        raise ValueError(f"{row_id}: image_sha256 is missing")
    actual_hash = image_sha256(image_path)
    if actual_hash != observed_hash:
        raise ValueError(f"{row_id}: image hash does not match sidecar")
    contract = record.get("run_contract_sha256")
    if expected_contract_sha256 is not None and contract != expected_contract_sha256:
        raise ValueError(f"{row_id}: run contract hash differs from the active run")
    action_hash = record.get("action_sha256")
    action = record.get("action")
    if not isinstance(action, dict) or not isinstance(action_hash, str):
        raise ValueError(f"{row_id}: normalized action provenance is missing")
    if action_sha256(action) != action_hash:
        raise ValueError(f"{row_id}: action hash does not match normalized action")
    if expected_task is not None:
        for key in ("id", "prompt_index", "prompt", "seed", "action_id", "action_type"):
            if record.get(key) != expected_task.get(key):
                raise ValueError(f"{row_id}: sidecar field {key!r} differs from task")
        if action != expected_task.get("action"):
            raise ValueError(f"{row_id}: sidecar action differs from registered task")
    return actual_hash


def validate_scores_against_manifest(
    manifest: Iterable[Mapping[str, Any]], scores: Iterable[Mapping[str, Any]]
) -> None:
    """Ensure scores are one-to-one with the current, hashed manifest."""
    manifest_by_id: Dict[str, Mapping[str, Any]] = {}
    for row in manifest:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in manifest_by_id:
            raise ValueError(f"manifest contains duplicate or empty id: {row_id!r}")
        manifest_by_id[row_id] = row
    score_rows = list(scores)
    score_by_id: Dict[str, Mapping[str, Any]] = {}
    for score in score_rows:
        row_id = str(score.get("id", ""))
        if not row_id or row_id in score_by_id:
            raise ValueError(f"duplicate or empty score id: {row_id!r}")
        score_by_id[row_id] = score
    if set(score_by_id) != set(manifest_by_id):
        raise ValueError("score IDs do not exactly match the complete manifest")
    for row_id, manifest_row in manifest_by_id.items():
        if score_by_id[row_id].get("image_sha256") != manifest_row.get("image_sha256"):
            raise ValueError(f"{row_id}: score provenance does not match image hash")
        if score_by_id[row_id].get("run_contract_sha256") != manifest_row.get(
            "run_contract_sha256"
        ):
            raise ValueError(f"{row_id}: score provenance does not match run contract")


__all__ = [
    "PROVENANCE_SCHEMA",
    "action_sha256",
    "canonical_json",
    "image_sha256",
    "json_sha256",
    "safe_child_path",
    "sha256_bytes",
    "sha256_file",
    "validate_design_rows",
    "validate_scores_against_manifest",
    "validate_sidecar",
]
