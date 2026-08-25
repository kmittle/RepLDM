"""Shared, dependency-light provenance checks for the registered S7 runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


PROVENANCE_SCHEMA = "s7_trajectory_provenance_v1"
DEFAULT_FREQUENCY_BAND_CUTOFFS = (0.08, 0.25)

# Keep this list synchronized with generate.generation_contract().  The
# contract intentionally excludes operational paths and device placement while
# binding every field that can change generated pixels or the crossed design.
RUN_CONTRACT_KEYS = (
    "schema",
    "action_schema",
    "actions_sha256",
    "actions",
    "prompts_sha256",
    "seeds",
    "model_name",
    "resolution",
    "num_inference_steps",
    "guidance_scale",
    "negative_prompt",
    "power_calibrate",
    "stage_name",
    "stage2_enabled",
    "models_to_cpu",
    "multi_encoder",
    "multi_decoder",
    "num_resample_timesteps",
    "init_rates",
    "frequency_band_cutoffs",
    "split_role",
    "git_commit",
    "runtime_provenance",
)


def resolve_frequency_band_cutoffs(source: Mapping[str, Any]) -> list[float]:
    """Resolve and validate the shared action-schema frequency cutoffs."""
    raw = source.get("frequency_band_cutoffs", DEFAULT_FREQUENCY_BAND_CUTOFFS)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ValueError("frequency_band_cutoffs must contain exactly two values")
    if len(raw) != 2:
        raise ValueError("frequency_band_cutoffs must contain exactly two values")
    try:
        cutoffs = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("frequency_band_cutoffs must be numeric") from exc
    if not all(math.isfinite(value) for value in cutoffs) or not (
        0 < cutoffs[0] < cutoffs[1] < 0.5
    ):
        raise ValueError("frequency_band_cutoffs must satisfy 0 < low < mid < 0.5")
    return cutoffs


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


def _contract_fields(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the expected immutable contract fields from a run config."""
    try:
        return {
            "schema": PROVENANCE_SCHEMA,
            "action_schema": config["action_schema"],
            "actions_sha256": config["actions_sha256"],
            "actions": config["actions"],
            "prompts_sha256": config["prompts_sha256"],
            "seeds": config["seeds"],
            "model_name": config["model_name"],
            "resolution": config["resolution"],
            "num_inference_steps": config["num_inference_steps"],
            "guidance_scale": config["guidance_scale"],
            "negative_prompt": config["negative_prompt"],
            "power_calibrate": config["power_calibrate"],
            "stage_name": config["stage_name"],
            "stage2_enabled": config["stage2_enabled"],
            "models_to_cpu": config["models_to_cpu"],
            "multi_encoder": config["multi_encoder"],
            "multi_decoder": config["multi_decoder"],
            "num_resample_timesteps": config["num_resample_timesteps"],
            "init_rates": list(config["init_rates"]),
            "frequency_band_cutoffs": list(config["frequency_band_cutoffs"]),
            "split_role": config.get("split_role"),
            "git_commit": config.get("git_commit"),
            "runtime_provenance": config.get("runtime_provenance", {}),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"run config lacks valid contract fields: {exc}") from exc


def validate_run_contract(config: Mapping[str, Any]) -> str:
    """Recompute and field-bind a registered run's immutable contract.

    The hash alone is insufficient when both the payload and digest are
    edited.  Binding the payload to the duplicated top-level config fields
    catches that class of drift while keeping the helper independent of
    generation/scoring dependencies.
    """
    contract = config.get("run_contract")
    if not isinstance(contract, dict):
        raise ValueError("run config lacks a run_contract object")
    for key in ("stage2_enabled", "models_to_cpu", "multi_encoder", "multi_decoder"):
        if not isinstance(config.get(key), bool):
            raise ValueError(f"run config field {key!r} must be boolean")
    for key in ("resolution", "num_inference_steps", "power_calibrate", "num_resample_timesteps"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"run config field {key!r} must be an integer")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in seeds
    ):
        raise ValueError("run config field 'seeds' must be a list of integers")
    guidance_scale = config.get("guidance_scale")
    if (
        isinstance(guidance_scale, bool)
        or not isinstance(guidance_scale, (int, float))
        or not math.isfinite(float(guidance_scale))
    ):
        raise ValueError("run config field 'guidance_scale' must be finite numeric")
    expected = _contract_fields(config)
    if set(contract) != set(RUN_CONTRACT_KEYS):
        raise ValueError("run contract fields differ from the registered schema")
    for key in RUN_CONTRACT_KEYS:
        if contract.get(key) != expected[key]:
            raise ValueError(f"run contract field {key!r} differs from run config")
    observed_hash = config.get("run_contract_sha256")
    recomputed_hash = json_sha256(contract)
    if observed_hash != recomputed_hash:
        raise ValueError("run_contract_sha256 does not match the run_contract payload")
    return recomputed_hash


def safe_child_path(root: str | os.PathLike[str], relative: str) -> Path:
    root_path = Path(root).resolve()
    path = (root_path / relative).resolve()
    if os.path.commonpath((str(root_path), str(path))) != str(root_path):
        raise ValueError(f"path escapes run directory: {relative!r}")
    return path


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read and validate the PNG signature/IHDR without decoding the image."""
    with path.open("rb") as handle:
        header = handle.read(33)
    signature = b"\x89PNG\r\n\x1a\n"
    if len(header) < 33 or header[:8] != signature:
        raise ValueError(f"image is not a valid PNG: {path}")
    chunk_length = struct.unpack(">I", header[8:12])[0]
    if header[12:16] != b"IHDR" or chunk_length != 13:
        raise ValueError(f"PNG lacks a valid IHDR: {path}")
    expected_crc = struct.unpack(">I", header[29:33])[0]
    if zlib.crc32(header[12:29]) & 0xFFFFFFFF != expected_crc:
        raise ValueError(f"PNG IHDR checksum is invalid: {path}")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", header[16:29]
    )
    if width <= 0 or height <= 0:
        raise ValueError(f"PNG has invalid dimensions: {path}")
    if bit_depth not in {1, 2, 4, 8, 16} or color_type not in {0, 2, 3, 4, 6}:
        raise ValueError(f"PNG has an invalid pixel format: {path}")
    if compression != 0 or filtering != 0 or interlace not in {0, 1}:
        raise ValueError(f"PNG has an invalid IHDR format: {path}")
    return width, height


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
    strict_s7 = expected_contract_sha256 is not None
    if strict_s7 and image_path_value != f"images/{row_id}.png":
        raise ValueError(f"{row_id}: image_path must be images/<id>.png")
    image_path = safe_child_path(run_dir, image_path_value)
    if not image_path.is_file():
        raise ValueError(f"{row_id}: image is missing: {image_path}")
    if strict_s7:
        width, height = _png_dimensions(image_path)
        has_width = "width" in record
        has_height = "height" in record
        if has_width != has_height:
            raise ValueError(f"{row_id}: PNG width/height metadata must be paired")
        if has_width:
            for key, observed in (("width", record["width"]), ("height", record["height"])):
                if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
                    raise ValueError(f"{row_id}: {key} metadata is invalid")
            if (record["width"], record["height"]) != (width, height):
                raise ValueError(f"{row_id}: PNG dimensions differ from sidecar")
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
    "RUN_CONTRACT_KEYS",
    "action_sha256",
    "canonical_json",
    "image_sha256",
    "json_sha256",
    "safe_child_path",
    "sha256_bytes",
    "sha256_file",
    "validate_design_rows",
    "validate_run_contract",
    "validate_scores_against_manifest",
    "validate_sidecar",
]
