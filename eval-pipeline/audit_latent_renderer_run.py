"""Audit a latent-renderer run without inspecting action quality rankings."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml
from PIL import Image


DEFAULT_SCORE_KEYS = (
    "colorfulness",
    "laplacian_sharpness",
    "mean_saturation",
    "clipped_fraction",
    "contrast_std",
    "clip_cosine",
    "clipscore",
    "hpsv2",
    "imagereward",
    "patch_ir_mean",
    "patch_ir_std",
    "patch_ir_n",
    "aesthetic",
    "topiq_nr",
)
SPLIT_NAMES = {
    "train_search": "train",
    "validation_confirmation": "validation",
    "test_final": "test",
}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unique_by_id(rows: Iterable[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in result:
            raise ValueError(f"{label} contains an empty or duplicate id {row_id!r}")
        result[row_id] = row
    return result


def _finite_values(record: Dict[str, Any], key: str) -> List[float]:
    values = record.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"renderer diagnostic {key!r} must be a non-empty list")
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"renderer diagnostic {key!r} contains non-finite values")
    return converted


def _audit_injection_trajectory(
    record: Dict[str, Any], bound: float
) -> Dict[str, float]:
    """Validate the strict post-cast injection trajectory in one sidecar."""
    trajectory = record.get("latent_renderer_injection_diagnostics")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("strict renderer injection diagnostics are missing")

    max_ratio = 0.0
    max_overrun = 0.0
    for expected_step, step in enumerate(trajectory):
        if not isinstance(step, dict):
            raise ValueError("strict renderer injection step is not an object")
        step_index = step.get("step_index")
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or step_index != expected_step
        ):
            raise ValueError("strict renderer injection step indices are not contiguous")
        ratio = _finite_values(step, "postcast_update_ratio")
        overrun = _finite_values(step, "postcast_overrun")
        observed_overrun = _finite_values(step, "observed_postcast_overrun")
        scheduler_norm = _finite_values(step, "scheduler_update_norm")
        precast_ratio = _finite_values(step, "precast_update_ratio")
        for key in ("postcast_cap_applied", "postcast_noop_fallback"):
            flags = step.get(key)
            if (
                not isinstance(flags, list)
                or not flags
                or not all(isinstance(value, bool) for value in flags)
            ):
                raise ValueError(f"strict renderer injection {key!r} is invalid")
        lengths = {
            len(ratio),
            len(overrun),
            len(observed_overrun),
            len(scheduler_norm),
            len(precast_ratio),
            len(step["postcast_cap_applied"]),
            len(step["postcast_noop_fallback"]),
        }
        if len(lengths) != 1:
            raise ValueError("strict renderer injection batch lengths differ")
        if min(ratio) < 0 or max(ratio) > bound + 1e-6:
            raise ValueError("strict renderer post-cast trust bound is violated")
        if min(overrun) < 0 or max(overrun) > 1e-7:
            raise ValueError("strict renderer post-cast overrun is non-zero")
        max_ratio = max(max_ratio, max(ratio))
        max_overrun = max(max_overrun, max(overrun))

    return {
        "max_postcast_injection_ratio": max_ratio,
        "max_postcast_injection_overrun": max_overrun,
    }


def audit_run(
    run_dir: str | os.PathLike[str],
    prompts_path: str | os.PathLike[str],
    source_actions_path: str | os.PathLike[str],
    *,
    split_role: str,
    required_score_keys: Iterable[str] = DEFAULT_SCORE_KEYS,
    verify_images: bool = True,
    require_distinct_actions: bool = True,
) -> Dict[str, Any]:
    """Reject incomplete, unpaired, malformed, or numerically unsafe runs."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    manifest_path = run_dir / "manifest.jsonl"
    scores_path = run_dir / "scores.jsonl"
    for path in (config_path, manifest_path, scores_path):
        if not path.is_file():
            raise ValueError(f"required run file is missing: {path}")

    with config_path.open() as handle:
        config = json.load(handle)
    with open(source_actions_path) as handle:
        source = yaml.safe_load(handle) or {}
    prompts = pd.read_csv(prompts_path)
    manifest = load_jsonl(manifest_path)
    scores = load_jsonl(scores_path)
    manifest_by_id = _unique_by_id(manifest, "manifest")
    scores_by_id = _unique_by_id(scores, "scores")

    split_seeds = source.get("split_seeds")
    if not isinstance(split_seeds, dict) or split_role not in split_seeds:
        raise ValueError(f"source actions do not register split role {split_role!r}")
    seeds = [int(value) for value in split_seeds[split_role]]
    if [int(value) for value in config.get("seeds", [])] != seeds:
        raise ValueError("run config seeds differ from the registered split role")
    if config.get("split_role") not in (None, split_role):
        raise ValueError("run config split_role differs from the requested audit role")
    recorded_prompt_hash = config.get("prompts_sha256")
    if recorded_prompt_hash is not None and recorded_prompt_hash != sha256_file(prompts_path):
        raise ValueError("run config prompt hash differs from the audited prompt CSV")
    recorded_action_hash = config.get("actions_sha256")
    if (
        recorded_action_hash is not None
        and recorded_action_hash != sha256_file(source_actions_path)
    ):
        raise ValueError("run config action hash differs from the audited source YAML")

    expected_split = SPLIT_NAMES.get(split_role)
    if not {"index", "TEXT"}.issubset(prompts.columns):
        raise ValueError("prompt CSV must contain index and TEXT columns")
    if expected_split and (
        "split" not in prompts or set(prompts["split"].astype(str)) != {expected_split}
    ):
        raise ValueError(f"prompt CSV is not explicitly marked split={expected_split}")
    if prompts["index"].duplicated().any():
        raise ValueError("prompt CSV contains duplicate indices")
    prompt_text = {
        int(row["index"]): str(row["TEXT"]) for _, row in prompts.iterrows()
    }

    source_actions = source.get("actions")
    configured_actions = config.get("actions")
    if not isinstance(source_actions, list) or not isinstance(configured_actions, list):
        raise ValueError("source and run config must both contain action lists")
    action_ids = [str(action.get("id", "")) for action in source_actions]
    configured_ids = [str(action.get("id", "")) for action in configured_actions]
    if not all(action_ids) or len(action_ids) != len(set(action_ids)):
        raise ValueError("source actions contain empty or duplicate ids")
    if configured_ids != action_ids:
        raise ValueError("run action order/set differs from source actions")
    source_by_id = {str(action["id"]): action for action in source_actions}
    provider_defaults = dict(source.get("latent_renderer_provider", {}) or {})
    for registered, configured in zip(source_actions, configured_actions):
        action_id = str(registered["id"])
        if configured.get("type") != registered.get("type"):
            raise ValueError(f"{action_id}: run action type differs from source")
        if registered.get("type") == "latent_renderer_fixed":
            expected = [float(value) for value in registered.get("coefficients", [])]
            observed = [float(value) for value in configured.get("coefficients", [])]
            if observed != expected:
                raise ValueError(f"{action_id}: run coefficients differ from source")
            expected_provider = dict(provider_defaults)
            expected_provider.update(registered.get("provider", {}) or {})
            configured_provider = configured.get("latent_renderer_provider", {}) or {}
            for key, value in expected_provider.items():
                if configured_provider.get(key) != value:
                    raise ValueError(
                        f"{action_id}: run provider field {key!r} differs from source"
                    )
    expected_cutoffs = [float(value) for value in source.get("frequency_band_cutoffs", [])]
    observed_cutoffs = [float(value) for value in config.get("frequency_band_cutoffs", [])]
    if observed_cutoffs != expected_cutoffs:
        raise ValueError("run frequency-band cutoffs differ from source")

    expected_design = {
        (prompt_index, seed, action_id)
        for prompt_index in prompt_text
        for seed in seeds
        for action_id in action_ids
    }
    records_by_design: Dict[tuple, Dict[str, Any]] = {}
    for record in manifest:
        key = (
            int(record.get("prompt_index", -1)),
            int(record.get("seed", -1)),
            str(record.get("action_id", "")),
        )
        if key in records_by_design:
            raise ValueError(f"manifest duplicates design cell {key}")
        records_by_design[key] = record
    observed_design = set(records_by_design)
    if observed_design != expected_design:
        missing = len(expected_design - observed_design)
        extra = len(observed_design - expected_design)
        raise ValueError(f"incomplete design: {missing} missing and {extra} extra cells")
    if set(scores_by_id) != set(manifest_by_id):
        raise ValueError("score ids do not exactly match the complete manifest")

    required_score_keys = tuple(required_score_keys)
    max_update_ratio = 0.0
    max_postcast_overrun = 0.0
    max_postcast_injection_ratio = 0.0
    max_postcast_injection_overrun = 0.0
    max_mean_error = 0.0
    max_variance_error = 0.0
    image_hashes: Dict[tuple, str] = {}
    block_devices: Dict[tuple, set] = {}
    block_ranks: Dict[tuple, set] = {}
    run_root = run_dir.resolve()
    resolution = int(config.get("resolution", 0))
    for design, record in records_by_design.items():
        prompt_index, seed, action_id = design
        expected_id = f"p{prompt_index}_seed{seed}_a{action_id}"
        if record.get("id") != expected_id:
            raise ValueError(f"record id differs from design cell: {record.get('id')!r}")
        if str(record.get("prompt")) != prompt_text[prompt_index]:
            raise ValueError(f"prompt text drift at prompt index {prompt_index}")
        if record.get("action_type") != source_by_id[action_id].get("type"):
            raise ValueError(f"{expected_id}: record action type differs from source")

        block = (prompt_index, seed)
        block_devices.setdefault(block, set()).add(str(record.get("device", "")))
        block_ranks.setdefault(block, set()).add(int(record.get("execution_rank", -1)))
        image_path = (run_dir / str(record.get("image_path", ""))).resolve()
        if os.path.commonpath((run_root, image_path)) != str(run_root):
            raise ValueError(f"{expected_id}: image path escapes the run directory")
        if not image_path.is_file():
            raise ValueError(f"{expected_id}: image is missing")
        image_hashes[design] = sha256_file(image_path)
        if verify_images:
            with Image.open(image_path) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError(f"{expected_id}: expected an RGB PNG")
                if image.size != (resolution, resolution):
                    raise ValueError(f"{expected_id}: image dimensions differ from config")
                image.verify()

        score = scores_by_id[expected_id]
        for key in required_score_keys:
            try:
                value = float(score[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{expected_id}: missing/non-numeric score {key!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{expected_id}: non-finite score {key!r}")

        if source_by_id[action_id].get("type") == "latent_renderer_fixed":
            diagnostics = record.get("latent_renderer_diagnostics")
            if not isinstance(diagnostics, dict):
                raise ValueError(f"{expected_id}: renderer diagnostics are missing")
            update_ratio = _finite_values(diagnostics, "update_ratio")
            mean_error = _finite_values(diagnostics, "mean_error")
            variance_error = _finite_values(diagnostics, "variance_error")
            bound = float(record.get("action", {}).get("max_update_ratio", 0.05))
            if not math.isfinite(bound) or bound < 0:
                raise ValueError(f"{expected_id}: renderer trust bound is invalid")
            if min(update_ratio) < 0 or max(update_ratio) > bound + 1e-6:
                raise ValueError(f"{expected_id}: renderer trust bound is violated")
            if max(map(abs, mean_error)) > 1e-4:
                raise ValueError(f"{expected_id}: renderer mean preservation is violated")
            if max(map(abs, variance_error)) > 1e-3:
                raise ValueError(f"{expected_id}: renderer variance preservation is violated")
            max_update_ratio = max(max_update_ratio, max(update_ratio))
            action_record = record.get("action", {})
            if bool(action_record.get("enforce_post_cast_cap", False)):
                precast_ratio = _finite_values(diagnostics, "precast_update_ratio")
                postcast_ratio = _finite_values(diagnostics, "postcast_update_ratio")
                observed_overrun = _finite_values(
                    diagnostics, "observed_postcast_overrun"
                )
                postcast_overrun = _finite_values(diagnostics, "postcast_overrun")
                if len(postcast_ratio) != len(update_ratio):
                    raise ValueError(f"{expected_id}: post-cast ratio length differs")
                if any(
                    abs(left - right) > 1e-6
                    for left, right in zip(postcast_ratio, update_ratio)
                ):
                    raise ValueError(f"{expected_id}: update_ratio is not post-cast ratio")
                if max(postcast_overrun) > 1e-7:
                    raise ValueError(f"{expected_id}: strict post-cast cap is violated")
                if (
                    len(precast_ratio) != len(update_ratio)
                    or len(observed_overrun) != len(update_ratio)
                ):
                    raise ValueError(f"{expected_id}: strict renderer diagnostic lengths differ")
                max_postcast_overrun = max(max_postcast_overrun, max(postcast_overrun))
                injection_summary = _audit_injection_trajectory(record, bound)
                max_postcast_injection_ratio = max(
                    max_postcast_injection_ratio,
                    injection_summary["max_postcast_injection_ratio"],
                )
                max_postcast_injection_overrun = max(
                    max_postcast_injection_overrun,
                    injection_summary["max_postcast_injection_overrun"],
                )
            max_mean_error = max(max_mean_error, max(map(abs, mean_error)))
            max_variance_error = max(
                max_variance_error, max(map(abs, variance_error))
            )
            provider_diagnostics = record.get("latent_renderer_provider_diagnostics")
            if not isinstance(provider_diagnostics, dict):
                raise ValueError(f"{expected_id}: provider diagnostics are missing")
            _finite_values(provider_diagnostics, "semantic_entropy")
            _finite_values(provider_diagnostics, "basis_rms")
        elif (
            record.get("latent_renderer_diagnostics") is not None
            or record.get("latent_renderer_provider_diagnostics") is not None
        ):
            raise ValueError(f"{expected_id}: non-renderer action has stale diagnostics")

    expected_ranks = set(range(len(action_ids)))
    for block, devices in block_devices.items():
        if len(devices) != 1 or "" in devices:
            raise ValueError(f"prompt/seed block {block} spans or lacks devices")
        if block_ranks[block] != expected_ranks:
            raise ValueError(f"prompt/seed block {block} has invalid execution ranks")
        if require_distinct_actions:
            hashes = {
                image_hashes[(block[0], block[1], action_id)]
                for action_id in action_ids
            }
            if len(hashes) != len(action_ids):
                raise ValueError(f"prompt/seed block {block} has identical PNG hashes")

    warnings = []
    if config.get("split_role") is None:
        warnings.append("run config predates explicit split_role recording")
    return {
        "passed": True,
        "split_role": split_role,
        "records": len(manifest),
        "prompts": len(prompt_text),
        "seeds": seeds,
        "actions": action_ids,
        "blocks": len(block_devices),
        "devices": sorted({next(iter(values)) for values in block_devices.values()}),
        "required_score_keys": list(required_score_keys),
        "max_update_ratio": max_update_ratio,
        "max_postcast_overrun": max_postcast_overrun,
        "max_postcast_injection_ratio": max_postcast_injection_ratio,
        "max_postcast_injection_overrun": max_postcast_injection_overrun,
        "max_abs_mean_error": max_mean_error,
        "max_abs_variance_error": max_variance_error,
        "all_action_png_hashes_distinct_within_block": require_distinct_actions,
        "warnings": warnings,
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest_path),
            "scores_sha256": sha256_file(scores_path),
            "prompts_sha256": sha256_file(prompts_path),
            "source_actions_sha256": sha256_file(source_actions_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--source_actions", required=True)
    parser.add_argument("--split_role", required=True, choices=tuple(SPLIT_NAMES))
    parser.add_argument("--output", default="")
    parser.add_argument("--skip_image_verify", action="store_true")
    parser.add_argument("--allow_duplicate_action_hashes", action="store_true")
    args = parser.parse_args()

    report = audit_run(
        args.run_dir,
        args.prompts,
        args.source_actions,
        split_role=args.split_role,
        verify_images=not args.skip_image_verify,
        require_distinct_actions=not args.allow_duplicate_action_hashes,
    )
    output = Path(args.output) if args.output else Path(args.run_dir) / "run_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"audit -> {output}")


if __name__ == "__main__":
    main()
