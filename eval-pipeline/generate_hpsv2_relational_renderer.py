"""Generate the frozen 3,200-prompt HPSv2 relational-renderer matrix.

This is intentionally separate from ``generate.py``.  It exposes no prompt,
seed, action, or output-directory overrides: a launch is either the complete
registered 3,200 x 4 matrix or it is rejected.  Prompt blocks are assigned to
physical GPUs deterministically, and all four settings for one prompt stay on
that GPU across resume.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Optional, Sequence
import uuid

import yaml


ROOT = Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))
import adaptive_oracle_model_snapshot as model_snapshot  # noqa: E402
DEFAULT_CONFIG = ROOT / "eval-pipeline/configs/hpsv2_relational_renderer_full_v1.yaml"
RUNNER_RELATIVE_PATH = "eval-pipeline/generate_hpsv2_relational_renderer.py"
SIDECAR_SCHEMA = "hpsv2_relational_renderer_sidecar_v4"
MANIFEST_SCHEMA = "hpsv2_relational_renderer_manifest_v3"
RUN_CONFIG_SCHEMA = "hpsv2_relational_renderer_run_v2"
GENERATION_ATTEMPT_SCHEMA = "hpsv2_generation_attempt_v1"
GENERATION_ATTEMPT_TERMINAL_SCHEMA = "hpsv2_generation_attempt_terminal_v1"
GENERATION_ATTEMPTS_DIR_NAME = "generation_attempts"
GENERATION_PROMPT_BLOCKS_PER_DEVICE_PER_ATTEMPT = 50
PARTIAL_MANIFEST_NAME = "partial_manifest.jsonl"
FINAL_MANIFEST_NAME = "manifest.jsonl"
EXPECTED_CONFIG_SCHEMA = "hpsv2_relational_renderer_full_v1"
EXPECTED_EXPERIMENT_ID = "hpsv2_relational_renderer_full_v1"
EXPECTED_PROMPT_COUNT = 3200
EXPECTED_SETTING_COUNT = 4
EXPECTED_TASK_COUNT = EXPECTED_PROMPT_COUNT * EXPECTED_SETTING_COUNT
EXPECTED_BASE_SEED = 20260831
EXPECTED_SEED_POLICY = "base_seed_plus_official_prompt_index"
EXPECTED_UNIQUE_PROMPT_COUNT = 3172
EXPECTED_DEVICES = (4, 5, 6, 7)
EXPECTED_STYLES = ("anime", "concept-art", "paintings", "photo")
EXPECTED_SETTINGS = (
    ("no_ag", "none", "baseline"),
    ("feature_axis_r1_pos", "feature", "proposed_structural_direction"),
    ("uniform_axis_r1_pos", "uniform_local", "local_smoothing_control"),
    ("random_axis_r1_pos", "random_edge", "matched_random_direction_control"),
)
ACTION_ORDER_NAMESPACE = "hpsv2-setting-order-v1"
DEVICE_ASSIGNMENT = "prompt-index-round-robin-v1"
ORBIT_NAME = "axis-r1"
TARGET_UPDATE_RATIO = 0.02
HARD_UPDATE_CAP = 0.05
NUM_INFERENCE_STEPS = 50
RESOLUTION = 1024
RANDOM_EDGE_COUNTER_SCHEMA = "ao-random-edge-counter-v1"
GENERATION_ENVIRONMENT_SCHEMA = "hpsv2_generation_environment_v1"
WORKER_MODEL_SNAPSHOT_SCHEMA = "hpsv2_worker_model_snapshot_v1"
WORKER_READY_SCHEMA = "hpsv2_generation_worker_ready_v2"
WORKER_COMPLETION_SCHEMA = "hpsv2_generation_worker_completion_v1"
GENERATION_PACKAGES = (
    "accelerate",
    "diffusers",
    "einops",
    "entmax",
    "huggingface-hub",
    "numpy",
    "pillow",
    "pyyaml",
    "safetensors",
    "scipy",
    "tokenizers",
    "torch",
    "torchvision",
    "tqdm",
    "transformers",
)
WORKER_DETERMINISM = {
    "deterministic_algorithms": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ATTEMPT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SCORE_MANIFEST_FIELDS = (
    "id",
    "prompt_index",
    "prompt",
    "bucket",
    "seed",
    "action_id",
    "action_type",
    "image_path",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise ValueError(f"{label} fields differ (missing={missing}, extra={extra})")


def _repository_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError(f"{label} must be a canonical repository-relative path")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} leaves the repository") from exc
    if path != (ROOT / relative).absolute():
        raise ValueError(f"{label} may not traverse a symlink")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return _require_mapping(value, label)


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return _require_mapping(value, label)


def _load_prompt_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_columns = [
                "index",
                "TEXT",
                "bucket",
                "benchmark_style",
                "style_index",
                "source_row_id",
            ]
            if reader.fieldnames != expected_columns:
                raise ValueError("HPSv2 CSV columns or order differ from the frozen schema")
            raw_rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot read frozen HPSv2 CSV: {path}") from exc

    if len(raw_rows) != EXPECTED_PROMPT_COUNT:
        raise ValueError("HPSv2 CSV must contain exactly 3,200 prompt records")
    rows: list[dict[str, Any]] = []
    style_counts = {style: 0 for style in EXPECTED_STYLES}
    source_ids: set[str] = set()
    for expected_index, raw in enumerate(raw_rows):
        try:
            index = int(raw["index"])
            style_index = int(raw["style_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError("HPSv2 CSV indices must be decimal integers") from exc
        style = raw["benchmark_style"]
        source_row_id = raw["source_row_id"]
        if index != expected_index:
            raise ValueError("HPSv2 CSV index must be contiguous 0..3199")
        if style not in EXPECTED_STYLES or raw["bucket"] != style:
            raise ValueError("HPSv2 CSV style/bucket differs from the official schema")
        if style_index != style_counts[style] or not 0 <= style_index < 800:
            raise ValueError("HPSv2 style indices must be contiguous 0..799")
        if source_row_id != f"{style}:{style_index:04d}":
            raise ValueError("HPSv2 source_row_id differs from style/style_index")
        if source_row_id in source_ids:
            raise ValueError("HPSv2 source_row_id must be unique")
        prompt = raw["TEXT"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("HPSv2 prompts must be non-empty strings")
        source_ids.add(source_row_id)
        style_counts[style] += 1
        rows.append(
            {
                "index": index,
                "prompt": prompt,
                "bucket": style,
                "benchmark_style": style,
                "style_index": style_index,
                "source_row_id": source_row_id,
            }
        )
    if style_counts != {style: 800 for style in EXPECTED_STYLES}:
        raise ValueError("HPSv2 CSV must contain exactly 800 rows per official style")
    prompt_counts = Counter(row["prompt"] for row in rows)
    unique_prompt_count = len(prompt_counts)
    if unique_prompt_count != EXPECTED_UNIQUE_PROMPT_COUNT:
        raise ValueError("HPSv2 CSV must preserve exactly 3,172 unique prompt strings")
    if len(rows) - unique_prompt_count != 28:
        raise ValueError("HPSv2 CSV must preserve exactly 28 official duplicate rows")
    if sum(count > 1 for count in prompt_counts.values()) != 23:
        raise ValueError("HPSv2 CSV duplicate texts must form exactly 23 clusters")
    if max(prompt_counts.values()) != 4:
        raise ValueError("HPSv2 CSV maximum exact-prompt multiplicity must be four")
    return rows


def _validate_prompt_manifest(
    manifest: Mapping[str, Any], csv_path: Path, csv_sha256: str
) -> None:
    if manifest.get("schema") != "hpsv2_official_prompt_manifest_v1":
        raise ValueError("HPSv2 prompt manifest schema differs")
    if manifest.get("benchmark") != "HPSv2":
        raise ValueError("HPSv2 prompt manifest benchmark differs")
    if manifest.get("official_prompt_count") != EXPECTED_PROMPT_COUNT:
        raise ValueError("HPSv2 prompt manifest must register exactly 3,200 rows")
    if manifest.get("exact_unique_prompt_count") != EXPECTED_UNIQUE_PROMPT_COUNT:
        raise ValueError("HPSv2 prompt manifest unique-prompt count differs")
    if manifest.get("official_duplicate_row_count") != 28:
        raise ValueError("HPSv2 prompt manifest duplicate-row count differs")
    if manifest.get("duplicate_policy") != (
        "preserve_official_rows_and_identify_by_style_and_style_index"
    ):
        raise ValueError("HPSv2 prompt manifest duplicate policy differs")
    if manifest.get("style_order") != list(EXPECTED_STYLES):
        raise ValueError("HPSv2 prompt manifest style order differs")
    if manifest.get("style_counts") != {style: 800 for style in EXPECTED_STYLES}:
        raise ValueError("HPSv2 prompt manifest style counts differ")
    csv_record = _require_mapping(manifest.get("csv"), "prompt manifest csv")
    expected_relative = csv_path.relative_to(ROOT).as_posix()
    if csv_record.get("path") != expected_relative:
        raise ValueError("HPSv2 prompt manifest CSV path differs")
    if csv_record.get("sha256") != csv_sha256:
        raise ValueError("HPSv2 prompt manifest CSV hash differs")
    if csv_record.get("size_bytes") != csv_path.stat().st_size:
        raise ValueError("HPSv2 prompt manifest CSV byte count differs")
    if csv_record.get("index_range") != [0, 3199]:
        raise ValueError("HPSv2 prompt manifest index range differs")


def _validate_settings(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    renderer = _require_mapping(config.get("renderer"), "renderer")
    expected_renderer = {
        "feature_block": "up_blocks.0",
        "feature_shape": [2, 1280, 32, 32],
        "pooled_grid": [16, 16],
        "orbit": ORBIT_NAME,
        "sign": 1,
        "target_update_ratio": TARGET_UPDATE_RATIO,
        "hard_update_cap": HARD_UPDATE_CAP,
        "scheduler_mapping": "euler_clean_endpoint",
        "preserve_channel_mean": True,
        "preserve_channel_covariance": True,
    }
    if renderer != expected_renderer:
        raise ValueError("renderer contract differs from the frozen axis-r1 action")

    raw_settings = config.get("settings")
    if not isinstance(raw_settings, list) or len(raw_settings) != EXPECTED_SETTING_COUNT:
        raise ValueError("settings must contain exactly four frozen entries")
    expected_rows = [
        {"id": action_id, "affinity_source": source, "role": role}
        for action_id, source, role in EXPECTED_SETTINGS
    ]
    if raw_settings != expected_rows:
        raise ValueError("settings or setting order differ from the frozen matrix")
    normalized = []
    for setting in expected_rows:
        physical_no_op = setting["id"] == "no_ag"
        normalized.append(
            {
                **setting,
                "physical_no_op": physical_no_op,
                "orbit_name": None if physical_no_op else ORBIT_NAME,
                "sign": 0 if physical_no_op else 1,
                "target_update_ratio": 0.0
                if physical_no_op
                else TARGET_UPDATE_RATIO,
                "hard_update_cap": 0.0 if physical_no_op else HARD_UPDATE_CAP,
                "scheduler_mapping": None
                if physical_no_op
                else "euler_clean_endpoint",
            }
        )
    return normalized


def _validate_scoring_and_analysis(config: Mapping[str, Any]) -> None:
    scoring = _require_mapping(config.get("scoring"), "scoring")
    required_outputs = [
        "hpsv2",
        "topiq_nr",
        "imagereward",
        "patch_ir_mean",
        "patch_ir_std",
        "patch_ir_n",
        "clip_cosine",
        "clipscore",
        "aesthetic",
        "colorfulness",
        "laplacian_sharpness",
        "clipped_fraction",
        "mean_saturation",
        "contrast_std",
    ]
    scoring_path = _repository_path(scoring.get("config"), "scoring.config")
    scoring_sha256 = sha256_file(scoring_path)
    expected_scoring = {
        "config": "eval-pipeline/configs/hpsv2_full_scoring_v1.yaml",
        "config_sha256": scoring_sha256,
        "strict": True,
        "scorer_provenance_required": True,
        "registered_scorer_provenance_sha256": (
            "f2734daafd1040ad95ceb1295d5bcee35ac3882e66bf5bb32a9576ae8311ed42"
        ),
        "hps_version": "v2.1",
        "hpsv2_stored_scale": "raw_cosine",
        "hpsv2_official_report_multiplier": 100.0,
        "metrics": ["imagereward", "pixel", "clip", "hps", "aesthetic", "iqa"],
        "required_outputs": required_outputs,
    }
    if scoring != expected_scoring:
        raise ValueError("scoring contract differs from the frozen HPSv2 recipe")
    if scoring.get("config_sha256") != scoring_sha256:
        raise ValueError("HPSv2 scoring config hash differs")

    scoring_config = _read_yaml(scoring_path, "HPSv2 scoring config")
    expected_scoring_config = {
        "schema": "hpsv2_full_scoring_v1",
        "benchmark": "HPSv2",
        "prompt_count": EXPECTED_PROMPT_COUNT,
        "subset_evaluation_allowed": False,
        "exclusive_cuda_process": True,
        "hps_version": "v2.1",
        "hpsv2_stored_scale": "raw_cosine",
        "hpsv2_official_report_multiplier": 100.0,
        "metrics": ["imagereward", "pixel", "clip", "hps", "aesthetic", "iqa"],
        "params": {
            "patch_crops": 5,
            "clip_model": "ViT-B/32",
            "clipscore_w": 2.5,
        },
        "scorer_provenance": {
            "required_schema": "repldm_scorer_provenance_v1",
        },
        "registered_scorer_provenance_sha256": (
            "f2734daafd1040ad95ceb1295d5bcee35ac3882e66bf5bb32a9576ae8311ed42"
        ),
        "metric_meta": {
            "imagereward": "higher",
            "patch_ir_mean": "higher",
            "clip_cosine": "higher",
            "clipscore": "higher",
            "hpsv2": "higher",
            "aesthetic": "higher",
            "topiq_nr": "higher",
        },
    }
    if scoring_config != expected_scoring_config:
        raise ValueError("HPSv2 scoring YAML differs from its frozen v1 contract")

    expected_analysis = {
        "statistical_unit": "prompt_row",
        "paired_on": ["benchmark_style", "style_index", "seed"],
        "primary_contrast": "feature_axis_r1_pos-minus-no_ag",
        "mechanism_contrasts": [
            "feature_axis_r1_pos-minus-uniform_axis_r1_pos",
            "feature_axis_r1_pos-minus-random_axis_r1_pos",
        ],
        "report_by_style": True,
        "report_official_80_prompt_groups": True,
        "paired_bootstrap_samples": 10000,
        "confidence_level": 0.95,
        "bootstrap_unit": "exact_prompt_text_cluster",
        "bootstrap_cluster_count": EXPECTED_UNIQUE_PROMPT_COUNT,
        "official_duplicate_rows_in_point_estimates": True,
        "bootstrap_seed": EXPECTED_BASE_SEED,
        "guards": {
            "clip_cosine_ci_lower_min_delta": -0.005,
            "topiq_nr_ci_lower_min_delta": -0.005,
            "clipped_fraction_ci_upper_max_delta": 0.001,
            "mean_saturation_ci_upper_max_delta": 0.005,
            "contrast_geometric_ratio_interval": [0.95, 1.05],
        },
        "interpretation_rule": [
            "feature_hpsv2_ci_lower_must_exceed_no_ag",
            "feature_hpsv2_ci_lower_must_exceed_uniform",
            "feature_hpsv2_ci_lower_must_exceed_random",
            "all_quality_and_pixel_guards_must_pass",
            "otherwise_do_not_use_this_direction_for_rl_or_distillation",
        ],
    }
    if config.get("analysis") != expected_analysis:
        raise ValueError("analysis contrasts or guards differ from the frozen contract")


def load_contract(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("only the frozen HPSv2 full-matrix config may be used")
    config_raw = config_path.read_bytes()
    try:
        config = yaml.safe_load(config_raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("frozen HPSv2 config is unreadable") from exc
    config = _require_mapping(config, "HPSv2 generation config")
    if config.get("schema") != EXPECTED_CONFIG_SCHEMA:
        raise ValueError("HPSv2 generation config schema differs")
    if config.get("status") != "frozen_full_benchmark":
        raise ValueError("HPSv2 generation config is not frozen")
    if config.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError("HPSv2 experiment id differs")

    benchmark = _require_mapping(config.get("benchmark"), "benchmark")
    expected_benchmark_fields = {
        "name": "HPSv2",
        "protocol": "official_complete_prompt_set",
        "prompt_count": EXPECTED_PROMPT_COUNT,
        "style_order": list(EXPECTED_STYLES),
        "prompts_per_style": 800,
        "images_per_prompt_per_setting": 1,
        "base_seed": EXPECTED_BASE_SEED,
        "seed_policy": EXPECTED_SEED_POLICY,
        "subset_evaluation_allowed": False,
    }
    for key, value in expected_benchmark_fields.items():
        if benchmark.get(key) != value:
            raise ValueError(f"benchmark.{key} differs from the frozen contract")
    prompt_csv = _repository_path(benchmark.get("prompt_csv"), "benchmark.prompt_csv")
    prompt_manifest = _repository_path(
        benchmark.get("prompt_manifest"), "benchmark.prompt_manifest"
    )
    csv_sha256 = sha256_file(prompt_csv)
    manifest_sha256 = sha256_file(prompt_manifest)
    if benchmark.get("prompt_csv_sha256") != csv_sha256:
        raise ValueError("frozen HPSv2 CSV hash differs from the config")
    if benchmark.get("prompt_manifest_sha256") != manifest_sha256:
        raise ValueError("frozen HPSv2 prompt-manifest hash differs from the config")
    prompt_manifest_value = _read_json(prompt_manifest, "HPSv2 prompt manifest")
    _validate_prompt_manifest(prompt_manifest_value, prompt_csv, csv_sha256)
    prompts = _load_prompt_rows(prompt_csv)

    sampling = _require_mapping(config.get("sampling"), "sampling")
    expected_sampling = {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_revision": "462165984030d82259a11f4367a4eed129e94a7b",
        "model_variant": "fp16",
        "model_snapshot_loaded_file_count": 18,
        "model_snapshot_manifest_sha256": (
            model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256
        ),
        "pipeline": "RepLDMSDXLPipeline",
        "resolution": RESOLUTION,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "cfg_scale": 7.5,
        "guidance_rescale": 0.0,
        "negative_prompt": "blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
        "scheduler": "EulerDiscreteScheduler",
        "prediction_type": "epsilon",
        "scheduler_churn": 0.0,
        "stage2": False,
        "extra_unet_calls": 0,
        "batch_size": 1,
        "torch_dtype": "float16",
        "local_files_only": True,
    }
    if sampling != expected_sampling:
        raise ValueError("sampling contract differs from the frozen SDXL Euler setup")

    execution = _require_mapping(config.get("execution"), "execution")
    expected_execution = {
        "requested_physical_devices": list(EXPECTED_DEVICES),
        "grouping": "prompt",
        "same_device_for_all_settings_of_one_prompt": True,
        "deterministic_setting_order": "sha256",
        "resumable": True,
        "prompt_blocks_per_device_per_attempt": (
            GENERATION_PROMPT_BLOCKS_PER_DEVICE_PER_ATTEMPT
        ),
        "expected_setting_count": EXPECTED_SETTING_COUNT,
        "expected_image_count": EXPECTED_TASK_COUNT,
        "require_clean_git_worktree": True,
        "require_head_pushed_to_origin": True,
        "generation_environment_schema": GENERATION_ENVIRONMENT_SCHEMA,
        "generation_packages": list(GENERATION_PACKAGES),
        "worker_determinism": dict(WORKER_DETERMINISM),
    }
    if execution != expected_execution:
        raise ValueError("execution contract differs from the frozen full matrix")
    settings = _validate_settings(config)
    _validate_scoring_and_analysis(config)
    output_dir = _repository_path(config.get("output_dir"), "output_dir")
    if output_dir != (ROOT / "outputs/hpsv2_relational_renderer/full_v1").resolve():
        raise ValueError("HPSv2 output directory differs from the frozen path")

    return {
        "config": config,
        "config_path": config_path,
        "config_sha256": sha256_bytes(config_raw),
        "prompt_csv": prompt_csv,
        "prompt_csv_sha256": csv_sha256,
        "prompt_manifest": prompt_manifest,
        "prompt_manifest_sha256": manifest_sha256,
        "prompt_manifest_value": prompt_manifest_value,
        "prompts": prompts,
        "sampling": sampling,
        "settings": settings,
        "output_dir": output_dir,
    }


def _seed_for_prompt(prompt_index: int) -> int:
    if not 0 <= prompt_index < EXPECTED_PROMPT_COUNT:
        raise ValueError("prompt index is outside the frozen HPSv2 range")
    return EXPECTED_BASE_SEED + prompt_index


def _setting_execution_order(
    prompt_index: int, source_row_id: str, row_seed: int
) -> list[str]:
    rows = []
    for action_id, _source, _role in EXPECTED_SETTINGS:
        payload = (
            f"{ACTION_ORDER_NAMESPACE}:{prompt_index}:{source_row_id}:"
            f"{row_seed}:{action_id}"
        ).encode("utf-8")
        rows.append((hashlib.sha256(payload).digest(), action_id))
    return [action_id for _digest, action_id in sorted(rows)]


def build_tasks(contract: Mapping[str, Any], devices: Sequence[int]) -> list[dict[str, Any]]:
    if tuple(devices) != EXPECTED_DEVICES:
        raise ValueError("--devices must be exactly 4,5,6,7 in that order")
    settings = {setting["id"]: setting for setting in contract["settings"]}
    tasks: list[dict[str, Any]] = []
    for prompt in contract["prompts"]:
        prompt_index = int(prompt["index"])
        row_seed = _seed_for_prompt(prompt_index)
        physical_index = int(devices[prompt_index % len(devices)])
        order = _setting_execution_order(
            prompt_index, prompt["source_row_id"], row_seed
        )
        for execution_rank, action_id in enumerate(order):
            task_id = f"p{prompt_index:04d}_seed{row_seed}_a{action_id}"
            task = {
                "id": task_id,
                "prompt": prompt["prompt"],
                "prompt_index": prompt_index,
                "prompt_row_id": prompt["source_row_id"],
                "source_row_id": prompt["source_row_id"],
                "bucket": prompt["bucket"],
                "benchmark_style": prompt["benchmark_style"],
                "style_index": int(prompt["style_index"]),
                "seed": row_seed,
                "action_id": action_id,
                "action_type": "none"
                if settings[action_id]["physical_no_op"]
                else "adaptive_oracle_fixed",
                "action": copy.deepcopy(settings[action_id]),
                "execution_rank": execution_rank,
                "device": f"cuda:{physical_index}",
                "physical_device_index": physical_index,
            }
            task["task_sha256"] = canonical_sha256(task)
            tasks.append(task)
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError("full HPSv2 matrix must contain exactly 12,800 tasks")
    ids = {task["id"] for task in tasks}
    if len(ids) != EXPECTED_TASK_COUNT:
        raise RuntimeError("full HPSv2 task ids must be unique")
    return tasks


def _run_command(arguments: Sequence[str], timeout_seconds: float = 10.0) -> str:
    try:
        result = subprocess.run(
            list(arguments),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"command timed out after {timeout_seconds:g}s "
            f"({' '.join(arguments)})"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({' '.join(arguments)}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _configure_worker_determinism(torch: Any) -> None:
    """Apply the exact numerical controls shared by every generation worker."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _observed_worker_determinism(torch: Any) -> dict[str, bool]:
    return {
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _generation_environment(
    devices: Sequence[int], torch: Any
) -> dict[str, Any]:
    """Bind package, CUDA, driver, hardware, and determinism state for resume."""
    if tuple(int(value) for value in devices) != EXPECTED_DEVICES:
        raise ValueError("generation environment requires the frozen GPU set")
    packages: dict[str, str] = {}
    for name in GENERATION_PACKAGES:
        try:
            value = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"generation package is missing: {name}") from exc
        if not value:
            raise RuntimeError(f"generation package has an empty version: {name}")
        packages[name] = str(value)

    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    inventory = []
    requested = set(EXPECTED_DEVICES)
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 5)]
        if len(fields) != 6:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        if index not in requested:
            continue
        try:
            total_memory_mib = int(fields[4])
        except ValueError as exc:
            raise RuntimeError("nvidia-smi returned invalid GPU memory") from exc
        inventory.append(
            {
                "index": index,
                "uuid": fields[1],
                "pci_bus_id": fields[2],
                "name": fields[3],
                "total_memory_mib": total_memory_mib,
                "driver_version": fields[5],
            }
        )
    inventory.sort(key=lambda row: row["index"])
    if [row["index"] for row in inventory] != list(EXPECTED_DEVICES):
        raise RuntimeError("nvidia-smi did not return the frozen GPU inventory")
    if len({row["uuid"] for row in inventory}) != len(EXPECTED_DEVICES) or len(
        {row["pci_bus_id"] for row in inventory}
    ) != len(EXPECTED_DEVICES):
        raise RuntimeError("frozen GPU identities are empty or duplicated")

    payload = {
        "schema": GENERATION_ENVIRONMENT_SCHEMA,
        "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": os.path.realpath(sys.executable),
        "packages": packages,
        "torch_build_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "worker_determinism": _observed_worker_determinism(torch),
        "gpu_inventory": inventory,
    }
    if payload["worker_determinism"] != WORKER_DETERMINISM:
        raise RuntimeError("generation determinism controls differ from the frozen values")
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _validate_generation_environment(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("recorded generation environment must be a mapping")
    expected_keys = {
        "schema",
        "platform",
        "python_version",
        "python_implementation",
        "python_executable",
        "packages",
        "torch_build_version",
        "cuda_runtime_version",
        "cudnn_version",
        "worker_determinism",
        "gpu_inventory",
        "sha256",
    }
    _require_exact_keys(record, expected_keys, "generation environment")
    if record.get("schema") != GENERATION_ENVIRONMENT_SCHEMA:
        raise ValueError("recorded generation environment schema differs")
    payload = dict(record)
    digest = payload.pop("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError("recorded generation environment hash is invalid")
    if canonical_sha256(payload) != digest:
        raise ValueError("recorded generation environment hash differs")
    packages = record.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != set(GENERATION_PACKAGES):
        raise ValueError("recorded generation package set differs")
    if any(not isinstance(value, str) or not value for value in packages.values()):
        raise ValueError("recorded generation package versions are invalid")
    if record.get("worker_determinism") != WORKER_DETERMINISM:
        raise ValueError("recorded generation determinism controls differ")
    inventory = record.get("gpu_inventory")
    if not isinstance(inventory, list) or [
        row.get("index") for row in inventory if isinstance(row, Mapping)
    ] != list(EXPECTED_DEVICES):
        raise ValueError("recorded generation GPU inventory differs")
    for row in inventory:
        if not isinstance(row, Mapping) or set(row) != {
            "index",
            "uuid",
            "pci_bus_id",
            "name",
            "total_memory_mib",
            "driver_version",
        }:
            raise ValueError("recorded generation GPU identity fields differ")
        if any(
            not isinstance(row[field], str) or not row[field]
            for field in ("uuid", "pci_bus_id", "name", "driver_version")
        ) or isinstance(row["total_memory_mib"], bool) or not isinstance(
            row["total_memory_mib"], int
        ):
            raise ValueError("recorded generation GPU identity is invalid")
    if len({row["uuid"] for row in inventory}) != len(EXPECTED_DEVICES) or len(
        {row["pci_bus_id"] for row in inventory}
    ) != len(EXPECTED_DEVICES):
        raise ValueError("recorded generation GPU identities are duplicated")


def _validate_model_snapshot_record(record: Any) -> None:
    expected = {
        **model_snapshot.expected_model_manifest(),
        "manifest_sha256": model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256,
    }
    if record != expected:
        raise ValueError("recorded SDXL loaded-file manifest differs")


def _worker_model_snapshot_evidence(
    stage_record: Mapping[str, Any],
    verifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if stage_record.get("manifest_sha256") != model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256:
        raise RuntimeError("worker staged-model manifest differs")
    if stage_record.get("loaded_file_count") != 18:
        raise RuntimeError("worker staged-model file count differs")
    if len(verifications) < 3:
        raise RuntimeError("worker staged-model load lacks pre/post verification")
    tree_hashes = {row.get("tree_sha256") for row in verifications}
    if None in tree_hashes or len(tree_hashes) != 1:
        raise RuntimeError("worker staged-model tree changed during bound load")
    for row in verifications:
        if (
            row.get("schema") != model_snapshot.MODEL_STAGE_VERIFICATION_SCHEMA
            or row.get("status") != "verified_unchanged"
            or row.get("manifest_sha256")
            != model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256
            or row.get("loaded_file_count") != 18
        ):
            raise RuntimeError("worker staged-model verification record differs")
    return {
        "schema": WORKER_MODEL_SNAPSHOT_SCHEMA,
        "loader": "pinned_proc_fd_with_pre_post_tree_verification",
        "model_id": model_snapshot.MODEL_ID,
        "revision": model_snapshot.MODEL_REVISION,
        "variant": "fp16",
        "manifest_sha256": model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256,
        "loaded_file_count": 18,
        "bound_load_verification_count": len(verifications),
        "tree_unchanged_during_load": True,
    }


def _validate_worker_model_snapshot_evidence(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("worker model-snapshot evidence must be a mapping")
    expected = {
        "schema": WORKER_MODEL_SNAPSHOT_SCHEMA,
        "loader": "pinned_proc_fd_with_pre_post_tree_verification",
        "model_id": model_snapshot.MODEL_ID,
        "revision": model_snapshot.MODEL_REVISION,
        "variant": "fp16",
        "manifest_sha256": model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256,
        "loaded_file_count": 18,
        "bound_load_verification_count": 3,
        "tree_unchanged_during_load": True,
    }
    if dict(record) != expected:
        raise ValueError("worker model-snapshot evidence differs")


def _generation_compute_conflicts(
    generation_environment: Mapping[str, Any],
    *,
    expected_workers: Optional[Mapping[int, Mapping[str, Any]]] = None,
    required_pids: Sequence[int] = (),
    expected_worker_start_ticks: Optional[Mapping[int, int]] = None,
) -> list[dict[str, Any]]:
    """Audit frozen GPU identity and bind each observed worker PID to one GPU."""
    _validate_generation_environment(generation_environment)
    frozen_inventory = [dict(row) for row in generation_environment["gpu_inventory"]]
    inventory_output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    requested_indices = {int(row["index"]) for row in frozen_inventory}
    observed_inventory = []
    for line in inventory_output.splitlines():
        fields = [field.strip() for field in line.split(",", 5)]
        if len(fields) != 6:
            raise RuntimeError("nvidia-smi returned a malformed GPU inventory row")
        try:
            index = int(fields[0])
            total_memory_mib = int(fields[4])
        except ValueError as exc:
            raise RuntimeError("nvidia-smi returned invalid GPU inventory values") from exc
        if index in requested_indices:
            observed_inventory.append(
                {
                    "index": index,
                    "uuid": fields[1],
                    "pci_bus_id": fields[2],
                    "name": fields[3],
                    "total_memory_mib": total_memory_mib,
                    "driver_version": fields[5],
                }
            )
    observed_inventory.sort(key=lambda row: row["index"])
    if observed_inventory != frozen_inventory:
        raise RuntimeError("generation GPU inventory drifted from the frozen run contract")

    targets = {str(row["uuid"]).lower(): row for row in frozen_inventory}
    workers: dict[int, dict[str, Any]] = {}
    for raw_pid, raw_identity in (expected_workers or {}).items():
        if isinstance(raw_pid, bool):
            raise ValueError("expected generation worker PID is invalid")
        pid = int(raw_pid)
        if pid <= 0 or pid in workers:
            raise ValueError("expected generation worker PID is invalid or duplicated")
        identity = dict(raw_identity)
        frozen = targets.get(str(identity.get("uuid", "")).lower())
        if frozen is None or identity != frozen:
            raise ValueError("expected generation worker identity is not frozen")
        workers[pid] = identity
    if len({row["uuid"].lower() for row in workers.values()}) != len(workers):
        raise ValueError("multiple generation workers target the same frozen GPU")
    required = {int(pid) for pid in required_pids}
    if not required.issubset(workers):
        raise ValueError("required generation worker is not registered")
    start_ticks = {
        int(pid): value
        for pid, value in (expected_worker_start_ticks or {}).items()
    }
    if required and expected_worker_start_ticks is None:
        raise ValueError("required generation worker start times are not registered")
    for pid in required:
        value = start_ticks.get(pid)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("required generation worker start time is invalid")

    output = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    conflicts: list[dict[str, Any]] = []
    worker_observations: dict[int, list[dict[str, Any]]] = {
        pid: [] for pid in workers
    }
    observed_start_ticks = {}
    for pid in sorted(required):
        observed = _process_start_ticks(pid)
        observed_start_ticks[pid] = observed
        if observed is None:
            conflicts.append(
                {
                    "kind": "required_worker_process_missing",
                    "pid": pid,
                    "expected_process_start_ticks": start_ticks[pid],
                }
            )
        elif observed != start_ticks[pid]:
            conflicts.append(
                {
                    "kind": "worker_pid_reused",
                    "pid": pid,
                    "expected_process_start_ticks": start_ticks[pid],
                    "observed_process_start_ticks": observed,
                }
            )
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError("nvidia-smi returned a malformed compute-process row")
        try:
            pid = int(fields[1])
        except ValueError as exc:
            raise RuntimeError("nvidia-smi returned an invalid compute-process PID") from exc
        gpu_uuid = fields[0]
        target = targets.get(gpu_uuid.lower())
        expected = workers.get(pid)
        if target is None and expected is None:
            continue
        observation = {
            "device": None if target is None else int(target["index"]),
            "gpu_uuid": gpu_uuid,
            "pid": pid,
            "process_name": fields[2],
            "used_memory_mib": fields[3],
        }
        if expected is not None:
            worker_observations[pid].append(observation)
            if gpu_uuid.lower() != str(expected["uuid"]).lower():
                conflicts.append(
                    {
                        "kind": "worker_on_wrong_gpu",
                        "expected_device": int(expected["index"]),
                        "expected_gpu_uuid": expected["uuid"],
                        **observation,
                    }
                )
        elif target is not None:
            conflicts.append(
                {
                    "kind": "foreign_compute_process",
                    **observation,
                }
            )
    for pid, observations in worker_observations.items():
        if len(observations) > 1:
            conflicts.append(
                {
                    "kind": "worker_pid_observed_multiple_times",
                    "pid": pid,
                    "expected_device": int(workers[pid]["index"]),
                    "observed_gpu_uuids": [row["gpu_uuid"] for row in observations],
                }
            )
        if pid in required and not observations:
            conflicts.append(
                {
                    "kind": "required_worker_missing",
                    "pid": pid,
                    "expected_device": int(workers[pid]["index"]),
                    "expected_gpu_uuid": workers[pid]["uuid"],
                }
            )
    for pid in sorted(required):
        observed = _process_start_ticks(pid)
        if observed != start_ticks[pid]:
            conflicts.append(
                {
                    "kind": "worker_process_changed_during_gpu_query",
                    "pid": pid,
                    "expected_process_start_ticks": start_ticks[pid],
                    "observed_process_start_ticks": observed,
                }
            )
    return conflicts


def _process_start_ticks(pid: int) -> Optional[int]:
    """Return Linux /proc start ticks, or None when a worker has exited."""
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="ascii") as handle:
            value = handle.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot inspect generation worker process identity") from exc
    closing = value.rfind(")")
    fields = value[closing + 2 :].split() if closing >= 0 else []
    if len(fields) <= 19:
        raise RuntimeError("generation worker process record is invalid")
    try:
        ticks = int(fields[19])
    except ValueError as exc:
        raise RuntimeError("generation worker process start time is invalid") from exc
    return ticks if ticks >= 0 else None


def _worker_ready_record(
    device_index: int, device_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the one-shot IPC record emitted after CUDA context initialization."""
    process_start_ticks = _process_start_ticks(os.getpid())
    if process_start_ticks is None:
        raise RuntimeError("generation worker process start time is unavailable")
    return {
        "schema": WORKER_READY_SCHEMA,
        "pid": os.getpid(),
        "process_start_ticks": process_start_ticks,
        "physical_device_index": int(device_index),
        "gpu_uuid": device_identity["gpu_uuid"],
        "torch_device_uuid": device_identity["torch_device_uuid"],
        "pci_bus_id": device_identity["pci_bus_id"],
        "gpu": device_identity["gpu"],
    }


def _validate_worker_ready_record(
    record: Any,
    expected_workers: Mapping[int, Mapping[str, Any]],
    expected_worker_start_ticks: Mapping[int, int],
) -> int:
    if not isinstance(record, Mapping) or set(record) != {
        "schema",
        "pid",
        "process_start_ticks",
        "physical_device_index",
        "gpu_uuid",
        "torch_device_uuid",
        "pci_bus_id",
        "gpu",
    }:
        raise ValueError("generation worker-ready record fields differ")
    if record.get("schema") != WORKER_READY_SCHEMA:
        raise ValueError("generation worker-ready schema differs")
    pid = record.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid not in expected_workers:
        raise ValueError("generation worker-ready PID is not registered")
    expected_start_ticks = expected_worker_start_ticks.get(pid)
    if (
        not isinstance(expected_start_ticks, int)
        or isinstance(expected_start_ticks, bool)
        or expected_start_ticks < 0
        or record.get("process_start_ticks") != expected_start_ticks
    ):
        raise ValueError("generation worker-ready process start time differs")
    if _process_start_ticks(pid) != expected_start_ticks:
        raise ValueError("generation worker-ready PID was reused")
    expected = expected_workers[pid]
    observed = {
        "index": record.get("physical_device_index"),
        "uuid": record.get("gpu_uuid"),
        "pci_bus_id": record.get("pci_bus_id"),
        "name": record.get("gpu"),
    }
    frozen = {
        "index": expected.get("index"),
        "uuid": expected.get("uuid"),
        "pci_bus_id": expected.get("pci_bus_id"),
        "name": expected.get("name"),
    }
    if observed != frozen:
        raise ValueError("generation worker-ready GPU identity differs")
    if str(record.get("torch_device_uuid", "")).lower() != str(
        record.get("gpu_uuid", "")
    ).lower():
        raise ValueError("generation worker-ready Torch UUID differs")
    return pid


def _worker_completion_record(
    device_index: int,
    device_identity: Mapping[str, Any],
    completed_task_count: int,
) -> dict[str, Any]:
    """Create the completion handshake held until the parent audits the GPU."""
    process_start_ticks = _process_start_ticks(os.getpid())
    if process_start_ticks is None:
        raise RuntimeError("generation worker process start time is unavailable")
    return {
        "schema": WORKER_COMPLETION_SCHEMA,
        "pid": os.getpid(),
        "process_start_ticks": process_start_ticks,
        "physical_device_index": int(device_index),
        "gpu_uuid": device_identity["gpu_uuid"],
        "torch_device_uuid": device_identity["torch_device_uuid"],
        "pci_bus_id": device_identity["pci_bus_id"],
        "gpu": device_identity["gpu"],
        "completed_task_count": int(completed_task_count),
    }


def _worker_handshake_failures(
    expected_workers: Mapping[int, Mapping[str, Any]],
    ready_pids: Sequence[int],
    completed_pids: Sequence[int],
    gpu_verified_pids: Sequence[int],
) -> list[tuple[str, str, str]]:
    """Return terminal failures for missing worker lifecycle receipts."""
    expected = {int(pid) for pid in expected_workers}
    ready = {int(pid) for pid in ready_pids}
    completed = {int(pid) for pid in completed_pids}
    verified = {int(pid) for pid in gpu_verified_pids}
    failures: list[tuple[str, str, str]] = []
    missing_ready = sorted(expected - ready)
    if missing_ready:
        failures.append(
            (
                "gpu-monitor",
                "worker-ready-missing",
                json.dumps({"missing_pids": missing_ready}, sort_keys=True),
            )
        )
    missing_completion = sorted(expected - completed)
    if missing_completion:
        failures.append(
            (
                "gpu-monitor",
                "worker-completion-missing",
                json.dumps({"missing_pids": missing_completion}, sort_keys=True),
            )
        )
    missing_gpu_verification = sorted(expected - verified)
    if missing_gpu_verification:
        failures.append(
            (
                "gpu-monitor",
                "worker-gpu-verification-missing",
                json.dumps(
                    {"missing_pids": missing_gpu_verification}, sort_keys=True
                ),
            )
        )
    return failures


def _validate_worker_completion_record(
    record: Any,
    expected_workers: Mapping[int, Mapping[str, Any]],
    expected_worker_start_ticks: Mapping[int, int],
    expected_task_counts: Mapping[int, int],
    *,
    require_live_process: bool = True,
) -> int:
    if not isinstance(record, Mapping) or set(record) != {
        "schema",
        "pid",
        "process_start_ticks",
        "physical_device_index",
        "gpu_uuid",
        "torch_device_uuid",
        "pci_bus_id",
        "gpu",
        "completed_task_count",
    }:
        raise ValueError("generation worker-completion record fields differ")
    if record.get("schema") != WORKER_COMPLETION_SCHEMA:
        raise ValueError("generation worker-completion schema differs")
    pid = record.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid not in expected_workers:
        raise ValueError("generation worker-completion PID is not registered")
    expected_start_ticks = expected_worker_start_ticks.get(pid)
    if (
        not isinstance(expected_start_ticks, int)
        or isinstance(expected_start_ticks, bool)
        or expected_start_ticks < 0
        or record.get("process_start_ticks") != expected_start_ticks
    ):
        raise ValueError("generation worker-completion process start time differs")
    if require_live_process and _process_start_ticks(pid) != expected_start_ticks:
        raise ValueError("generation worker-completion PID was reused")
    expected = expected_workers[pid]
    observed = {
        "index": record.get("physical_device_index"),
        "uuid": record.get("gpu_uuid"),
        "pci_bus_id": record.get("pci_bus_id"),
        "name": record.get("gpu"),
    }
    frozen = {
        "index": expected.get("index"),
        "uuid": expected.get("uuid"),
        "pci_bus_id": expected.get("pci_bus_id"),
        "name": expected.get("name"),
    }
    if observed != frozen:
        raise ValueError("generation worker-completion GPU identity differs")
    if str(record.get("torch_device_uuid", "")).lower() != str(
        record.get("gpu_uuid", "")
    ).lower():
        raise ValueError("generation worker-completion Torch UUID differs")
    expected_count = expected_task_counts.get(pid)
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or record.get("completed_task_count") != expected_count
    ):
        raise ValueError("generation worker-completion task count differs")
    return pid


def _require_generation_devices_process_free(
    generation_environment: Mapping[str, Any],
) -> None:
    """Fail if another compute process appeared after the queue's final poll."""
    conflicts = _generation_compute_conflicts(generation_environment)
    if conflicts:
        raise RuntimeError(
            "generation GPUs gained compute processes after the queue check: "
            + json.dumps(conflicts, sort_keys=True)
        )


def validate_git_launch_contract() -> str:
    status = _run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"]
    )
    if status:
        raise RuntimeError("HPSv2 generation requires a clean Git worktree")
    head = _run_command(["git", "rev-parse", "HEAD"])
    remote = _run_command(["git", "rev-parse", "origin/rl-version"])
    if not re.fullmatch(r"[0-9a-f]{40}", head) or head != remote:
        raise RuntimeError(
            "HPSv2 generation requires HEAD to equal pushed origin/rl-version"
        )
    for relative in (
        RUNNER_RELATIVE_PATH,
        "eval-pipeline/adaptive_oracle_model_snapshot.py",
        "eval-pipeline/configs/hpsv2_relational_renderer_full_v1.yaml",
        "eval-pipeline/prompts/hpsv2_official_3200.csv",
        "eval-pipeline/prompts/hpsv2_official_3200_manifest.json",
    ):
        tracked = _run_command(["git", "ls-files", "--error-unmatch", relative])
        if tracked != relative:
            raise RuntimeError(f"launch input is not tracked at HEAD: {relative}")
    return head


def _parse_devices(value: str) -> tuple[int, ...]:
    if value != "4,5,6,7":
        raise argparse.ArgumentTypeError("devices must be exactly 4,5,6,7")
    devices = tuple(int(item) for item in value.split(","))
    if devices != EXPECTED_DEVICES:
        raise argparse.ArgumentTypeError("devices must be exactly 4,5,6,7")
    return devices


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the frozen complete HPSv2 relational-renderer matrix",
        allow_abbrev=False,
    )
    parser.add_argument("--devices", required=True, type=_parse_devices)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the frozen 3,200-prompt contract without touching CUDA or outputs",
    )
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="read-only validation of every final PNG, sidecar, and manifest row",
    )
    return parser.parse_args(argv)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("atomic artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(temporary):
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_write_bytes(path, payload)


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _png_dimensions(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError as exc:
        raise ValueError(f"cannot read PNG header: {path}") from exc
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"image is not a canonical PNG: {path}")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width, height


def _image_paths(output_dir: Path, task_id: str) -> tuple[Path, Path]:
    images = output_dir / "images"
    return images / f"{task_id}.png", images / f"{task_id}.json"


def _attempt_paths(output_dir: Path, attempt_id: str) -> dict[str, Path]:
    if not ATTEMPT_ID_RE.fullmatch(attempt_id):
        raise ValueError("generation attempt id is invalid")
    root = output_dir / GENERATION_ATTEMPTS_DIR_NAME / attempt_id
    return {
        "root": root,
        "record": root / "attempt.json",
        "accepted": root / "accepted.json",
        "poisoned": root / "poisoned.json",
        "staging": root / "staging",
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _begin_generation_attempt(
    output_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
) -> tuple[dict[str, Any], str, Path]:
    task_ids = [str(task["id"]) for task in tasks]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("generation attempt requires distinct pending task ids")
    attempt_id = uuid.uuid4().hex
    record = {
        "schema": GENERATION_ATTEMPT_SCHEMA,
        "status": "running",
        "attempt_id": attempt_id,
        "run_contract_sha256": str(run_config["run_contract_sha256"]),
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "task_ids_sha256": canonical_sha256(task_ids),
    }
    attempt_sha256 = canonical_sha256(record)
    paths = _attempt_paths(output_dir, attempt_id)
    attempts_root = paths["root"].parent
    attempts_root.mkdir(parents=True, exist_ok=True)
    try:
        paths["root"].mkdir()
        (paths["staging"] / "images").mkdir(parents=True)
        _atomic_write_json(paths["record"], record)
        _fsync_directory(attempts_root)
    except BaseException:
        shutil.rmtree(paths["root"], ignore_errors=True)
        raise
    return record, attempt_sha256, paths["staging"]


def _validate_generation_attempt_record(
    record: Mapping[str, Any],
    *,
    attempt_id: str,
    run_contract_sha256: str,
    known_task_ids: set[str],
) -> dict[str, Any]:
    record = _require_mapping(record, "generation attempt record")
    _require_exact_keys(
        record,
        {
            "schema",
            "status",
            "attempt_id",
            "run_contract_sha256",
            "task_count",
            "task_ids",
            "task_ids_sha256",
        },
        "generation attempt record",
    )
    task_ids = record.get("task_ids")
    if (
        record.get("schema") != GENERATION_ATTEMPT_SCHEMA
        or record.get("status") != "running"
        or record.get("attempt_id") != attempt_id
        or record.get("run_contract_sha256") != run_contract_sha256
        or not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
        or set(task_ids) - known_task_ids
        or record.get("task_count") != len(task_ids)
        or record.get("task_ids_sha256") != canonical_sha256(task_ids)
    ):
        raise ValueError("generation attempt record differs from the frozen run")
    return record


def _validate_generation_terminal(
    terminal: Mapping[str, Any],
    *,
    status: str,
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = _require_mapping(terminal, f"generation {status} receipt")
    common = {
        "schema",
        "status",
        "attempt_id",
        "attempt_sha256",
        "run_contract_sha256",
    }
    expected_keys = common | ({"error"} if status == "poisoned" else {
        "task_count",
        "artifacts",
        "artifacts_sha256",
    })
    _require_exact_keys(terminal, expected_keys, f"generation {status} receipt")
    if (
        terminal.get("schema") != GENERATION_ATTEMPT_TERMINAL_SCHEMA
        or terminal.get("status") != status
        or terminal.get("attempt_id") != attempt["attempt_id"]
        or terminal.get("attempt_sha256") != canonical_sha256(attempt)
        or terminal.get("run_contract_sha256") != attempt["run_contract_sha256"]
    ):
        raise ValueError(f"generation {status} receipt binding differs")
    if status == "poisoned":
        error = _require_mapping(terminal.get("error"), "generation poison error")
        _require_exact_keys(error, {"type", "message"}, "generation poison error")
        if any(not isinstance(error.get(key), str) or not error[key] for key in error):
            raise ValueError("generation poison error is invalid")
        return terminal

    artifacts = terminal.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("generation accepted artifacts must be a list")
    for artifact in artifacts:
        artifact = _require_mapping(artifact, "generation accepted artifact")
        _require_exact_keys(
            artifact,
            {
                "task_id",
                "image_path",
                "image_sha256",
                "sidecar_path",
                "sidecar_sha256",
            },
            "generation accepted artifact",
        )
        task_id = artifact.get("task_id")
        if (
            not isinstance(task_id, str)
            or artifact.get("image_path") != f"images/{task_id}.png"
            or artifact.get("sidecar_path") != f"images/{task_id}.json"
            or any(
                not isinstance(artifact.get(key), str)
                or not SHA256_RE.fullmatch(artifact[key])
                for key in ("image_sha256", "sidecar_sha256")
            )
        ):
            raise ValueError("generation accepted artifact is invalid")
    if (
        [artifact["task_id"] for artifact in artifacts] != attempt["task_ids"]
        or terminal.get("task_count") != len(artifacts)
        or terminal.get("artifacts_sha256") != canonical_sha256(artifacts)
    ):
        raise ValueError("generation accepted artifact set differs from its attempt")
    return terminal


def _load_generation_attempt_index(
    output_dir: Path,
    run_config: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[tuple[dict[str, Any], Path]]]:
    """Return accepted task bindings and unterminated attempts."""
    attempts_root = output_dir / GENERATION_ATTEMPTS_DIR_NAME
    if not os.path.lexists(attempts_root):
        return {}, []
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise ValueError("generation attempts directory is unsafe")
    known_task_ids = {str(task["id"]) for task in tasks}
    accepted_index: dict[str, dict[str, Any]] = {}
    running: list[tuple[dict[str, Any], Path]] = []
    for attempt_dir in sorted(attempts_root.iterdir(), key=lambda path: path.name):
        if (
            not ATTEMPT_ID_RE.fullmatch(attempt_dir.name)
            or attempt_dir.is_symlink()
            or not attempt_dir.is_dir()
        ):
            raise ValueError(f"generation attempt entry is unsafe: {attempt_dir}")
        paths = _attempt_paths(output_dir, attempt_dir.name)
        attempt = _validate_generation_attempt_record(
            _read_json(paths["record"], "generation attempt record"),
            attempt_id=attempt_dir.name,
            run_contract_sha256=str(run_config["run_contract_sha256"]),
            known_task_ids=known_task_ids,
        )
        has_accepted = os.path.lexists(paths["accepted"])
        has_poisoned = os.path.lexists(paths["poisoned"])
        if has_accepted and has_poisoned:
            raise ValueError("generation attempt has two terminal states")
        if not has_accepted and not has_poisoned:
            running.append((attempt, attempt_dir))
            continue
        terminal_path = paths["accepted"] if has_accepted else paths["poisoned"]
        if not _regular_file(terminal_path):
            raise ValueError("generation attempt terminal receipt is unsafe")
        status = "accepted" if has_accepted else "poisoned"
        terminal = _validate_generation_terminal(
            _read_json(terminal_path, f"generation {status} receipt"),
            status=status,
            attempt=attempt,
        )
        if status == "poisoned":
            continue
        receipt_sha256 = sha256_file(terminal_path)
        for artifact in terminal["artifacts"]:
            task_id = str(artifact["task_id"])
            if task_id in accepted_index:
                raise ValueError("task is covered by multiple accepted attempts")
            accepted_index[task_id] = {
                **copy.deepcopy(dict(artifact)),
                "generation_attempt_id": attempt["attempt_id"],
                "generation_attempt_sha256": canonical_sha256(attempt),
                "accepted_receipt_sha256": receipt_sha256,
            }
    for attempt, _ in running:
        if set(attempt["task_ids"]) & set(accepted_index):
            raise ValueError("unterminated attempt overlaps accepted task artifacts")
    return accepted_index, running


def _quarantine_paths(output_dir: Path, paths: Sequence[Path], label: str) -> int:
    existing = [path for path in paths if os.path.lexists(path)]
    if not existing:
        return 0
    quarantine = output_dir / ".incomplete" / label
    quarantine.mkdir(parents=True, exist_ok=True)
    for path in existing:
        target = quarantine / f"{path.name}.{time.time_ns()}-{uuid.uuid4().hex}"
        os.replace(path, target)
    _fsync_directory(quarantine)
    _fsync_directory(output_dir / "images")
    return len(existing)


def _poison_generation_attempt(
    output_dir: Path,
    attempt: Mapping[str, Any],
    error: BaseException,
) -> int:
    paths = _attempt_paths(output_dir, str(attempt["attempt_id"]))
    if os.path.lexists(paths["accepted"]):
        raise RuntimeError("cannot poison an accepted generation attempt")
    poison = {
        "schema": GENERATION_ATTEMPT_TERMINAL_SCHEMA,
        "status": "poisoned",
        "attempt_id": attempt["attempt_id"],
        "attempt_sha256": canonical_sha256(attempt),
        "run_contract_sha256": attempt["run_contract_sha256"],
        "error": {
            "type": type(error).__name__,
            "message": str(error)[:2000] or "generation attempt failed",
        },
    }
    if os.path.lexists(paths["poisoned"]):
        _validate_generation_terminal(
            _read_json(paths["poisoned"], "generation poisoned receipt"),
            status="poisoned",
            attempt=attempt,
        )
    else:
        _atomic_write_json(paths["poisoned"], poison)
    canonical_paths = []
    for task_id in attempt["task_ids"]:
        canonical_paths.extend(_image_paths(output_dir, str(task_id)))
    return _quarantine_paths(
        output_dir,
        canonical_paths,
        f"poisoned-{attempt['attempt_id']}",
    )


def _quarantine_unaccepted_artifacts(
    output_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    accepted_index: Mapping[str, Mapping[str, Any]],
) -> int:
    paths = []
    for task in tasks:
        task_id = str(task["id"])
        if task_id in accepted_index:
            continue
        paths.extend(_image_paths(output_dir, task_id))
    return _quarantine_paths(
        output_dir,
        paths,
        f"unaccepted-{time.time_ns()}",
    )


def _validate_staged_generation_attempt(
    staging_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    attempt_sha256 = canonical_sha256(attempt)
    artifacts = []
    expected_names: set[str] = set()
    for task in tasks:
        task_id = str(task["id"])
        png_path, json_path = _image_paths(staging_dir, task_id)
        expected_names.update((png_path.name, json_path.name))
        if not _regular_file(png_path) or not _regular_file(json_path):
            raise ValueError(f"staged attempt is missing task {task_id}")
        sidecar = _read_json(json_path, f"staged sidecar {task_id}")
        validate_sidecar(
            sidecar,
            task,
            run_contract_sha256=str(run_config["run_contract_sha256"]),
            config_sha256=str(run_config["config"]["sha256"]),
            git_commit=str(run_config["git_commit"]),
            generation_environment_sha256=str(
                run_config["generation_environment"]["sha256"]
            ),
            generation_attempt_id=str(attempt["attempt_id"]),
            generation_attempt_sha256=attempt_sha256,
            output_dir=staging_dir,
        )
        artifacts.append(
            {
                "task_id": task_id,
                "image_path": f"images/{task_id}.png",
                "image_sha256": sha256_file(png_path),
                "sidecar_path": f"images/{task_id}.json",
                "sidecar_sha256": sha256_file(json_path),
            }
        )
    images_dir = staging_dir / "images"
    observed_names = {path.name for path in images_dir.iterdir()}
    if observed_names != expected_names:
        raise ValueError("staged generation attempt contains unexpected artifacts")
    return artifacts


def _promote_and_accept_generation_attempt(
    output_dir: Path,
    staging_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = _validate_staged_generation_attempt(
        staging_dir, tasks, run_config, attempt
    )
    for task in tasks:
        task_id = str(task["id"])
        canonical_png, canonical_json = _image_paths(output_dir, task_id)
        if os.path.lexists(canonical_png) or os.path.lexists(canonical_json):
            raise RuntimeError(f"canonical task appeared before promotion: {task_id}")
        staged_png, staged_json = _image_paths(staging_dir, task_id)
        os.replace(staged_png, canonical_png)
        os.replace(staged_json, canonical_json)
    _fsync_directory(output_dir / "images")
    receipt = {
        "schema": GENERATION_ATTEMPT_TERMINAL_SCHEMA,
        "status": "accepted",
        "attempt_id": attempt["attempt_id"],
        "attempt_sha256": canonical_sha256(attempt),
        "run_contract_sha256": attempt["run_contract_sha256"],
        "task_count": len(artifacts),
        "artifacts": artifacts,
        "artifacts_sha256": canonical_sha256(artifacts),
    }
    paths = _attempt_paths(output_dir, str(attempt["attempt_id"]))
    _atomic_write_json(paths["accepted"], receipt)
    _validate_generation_terminal(receipt, status="accepted", attempt=attempt)
    return receipt


def _quarantine_orphans(output_dir: Path, tasks: Sequence[Mapping[str, Any]]) -> int:
    quarantine = output_dir / ".incomplete"
    moved = 0
    for task in tasks:
        png_path, json_path = _image_paths(output_dir, str(task["id"]))
        png_exists = os.path.lexists(png_path)
        json_exists = os.path.lexists(json_path)
        if png_exists == json_exists:
            continue
        quarantine.mkdir(parents=True, exist_ok=True)
        orphan = png_path if png_exists else json_path
        target = quarantine / (
            f"{orphan.name}.orphan-{time.time_ns()}-{os.getpid()}"
        )
        os.replace(orphan, target)
        moved += 1
    if moved:
        directory_descriptor = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    return moved


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _one_number(value: Any, label: str) -> float:
    if not isinstance(value, list) or len(value) != 1 or not _finite_number(value[0]):
        raise ValueError(f"{label} must contain one finite number")
    return float(value[0])


def _validate_provider_step(
    provider: Any, task: Mapping[str, Any], step_index: int
) -> None:
    provider = _require_mapping(provider, "provider diagnostics")
    action = task["action"]
    if provider.get("implementation") != "adaptive_oracle_local_relational_basis_v1":
        raise ValueError("provider implementation differs")
    if provider.get("affinity_source") != action["affinity_source"]:
        raise ValueError("provider affinity source differs")
    if provider.get("selected_orbit") != ORBIT_NAME:
        raise ValueError("provider selected orbit differs")
    if provider.get("selected_orbit_index") != 0:
        raise ValueError("provider selected orbit index differs")
    if provider.get("selected_basis_shape") != [1, 1, 4, 128, 128]:
        raise ValueError("provider selected basis shape differs")
    if provider.get("selected_basis_dtype") != "float32":
        raise ValueError("provider selected basis dtype differs")
    if not isinstance(provider.get("selected_basis_sha256"), str) or not SHA256_RE.fullmatch(
        provider["selected_basis_sha256"]
    ):
        raise ValueError("provider basis hash is invalid")
    local = _require_mapping(provider.get("local_diagnostics"), "local diagnostics")
    if local.get("affinity_source") != action["affinity_source"]:
        raise ValueError("local diagnostics affinity source differs")
    if local.get("orbit_names") != ["axis-r1", "diagonal-r1", "axis-r2"]:
        raise ValueError("local diagnostics orbit order differs")
    if local.get("grid_size") != 16:
        raise ValueError("local diagnostics grid size differs")

    capture = provider.get("capture_record")
    if action["affinity_source"] == "feature":
        capture = _require_mapping(capture, "feature capture")
        expected_capture = {
            "feature_block": "up_blocks.0",
            "expected_cfg_shape": [2, 1280, 32, 32],
            "captured_shape": [2, 1280, 32, 32],
            "hook_calls": 1,
            "consume_calls": 1,
            "capture_complete": True,
            "conditional_rows": "second_half",
            "detached": True,
        }
        for key, expected in expected_capture.items():
            if capture.get(key) != expected:
                raise ValueError(f"feature capture {key} differs")
    elif capture is not None:
        raise ValueError("non-feature control unexpectedly recorded a feature capture")

    random_context = provider.get("random_counter_context")
    if action["affinity_source"] == "random_edge":
        random_context = _require_mapping(random_context, "random counter context")
        expected_context = {
            "schema": RANDOM_EDGE_COUNTER_SCHEMA,
            "experiment_id": EXPECTED_EXPERIMENT_ID,
            "split_role": "hpsv2_full",
            "prompt_row_id": task["prompt_row_id"],
            "seed": task["seed"],
            "step_index": step_index,
            "orbit_name": ORBIT_NAME,
        }
        if random_context != expected_context:
            raise ValueError("random-edge counter context differs")
        counter_hash = provider.get("random_counter_set_sha256")
        if not isinstance(counter_hash, str) or not SHA256_RE.fullmatch(counter_hash):
            raise ValueError("random-edge counter-set hash is invalid")
        if local.get("random_edge_counter_schema") != RANDOM_EDGE_COUNTER_SCHEMA:
            raise ValueError("random-edge local schema differs")
        if local.get("random_edge_actual_edge_counts") != [480, 450, 448]:
            raise ValueError("random-edge actual edge counts differ")
        if local.get("random_edge_unique_canonical_key_counts") != [64, 64, 56]:
            raise ValueError("random-edge canonical key counts differ")
        local_hashes = local.get("random_edge_counter_set_sha256")
        if (
            not isinstance(local_hashes, list)
            or len(local_hashes) != 3
            or any(
                not isinstance(value, str) or not SHA256_RE.fullmatch(value)
                for value in local_hashes
            )
            or local_hashes[0] != counter_hash
        ):
            raise ValueError("random-edge local counter hashes differ")
        if local.get("random_edge_actual_edges_unique") is not True:
            raise ValueError("random-edge actual edges are not unique")
    elif random_context is not None or provider.get("random_counter_set_sha256") is not None:
        raise ValueError("non-random control recorded random-edge state")


def _validate_renderer_steps(steps: Any, task: Mapping[str, Any]) -> None:
    action = task["action"]
    if action["physical_no_op"]:
        if steps not in (None, []):
            raise ValueError("no_ag must not contain renderer diagnostics")
        return
    if not isinstance(steps, list) or len(steps) != NUM_INFERENCE_STEPS:
        raise ValueError("active action must contain exactly 50 renderer steps")
    for index, raw in enumerate(steps):
        raw = _require_mapping(raw, f"renderer step {index}")
        if raw.get("step_index") != index or raw.get("scheduler_step_index") != index:
            raise ValueError("renderer step indices are not contiguous")
        if raw.get("prediction_type") != "epsilon":
            raise ValueError("renderer prediction type differs")
        if raw.get("sign") != 1:
            raise ValueError("renderer sign differs")
        ratio = _one_number(raw.get("applied_update_ratio"), "applied_update_ratio")
        error = _one_number(raw.get("target_ratio_error"), "target_ratio_error")
        target = _one_number(raw.get("target_update_ratio"), "target_update_ratio")
        if not 0.0 <= target < HARD_UPDATE_CAP:
            raise ValueError("renderer solver target ratio is outside its bounds")
        if not 0.0 <= ratio < HARD_UPDATE_CAP or not 0.0 <= error <= 5e-4:
            raise ValueError("renderer update ratio violates its frozen bounds")
        covariance = _one_number(raw.get("covariance_drift"), "covariance_drift")
        if not 0.0 <= covariance <= 0.01:
            raise ValueError("renderer covariance drift exceeds its frozen guard")
        raw_cap_hit = raw.get("cap_hit")
        if not isinstance(raw_cap_hit, list) or raw_cap_hit != [False]:
            raise ValueError("renderer unexpectedly hit the raw hard cap")
        round_trip = _require_mapping(
            raw.get("native_scheduler_round_trip"),
            "native scheduler round trip",
        )
        pred_error = _one_number(
            round_trip.get("pred_original_sample_relative_l2_error"),
            "pred_original_sample_relative_l2_error",
        )
        prev_error = _one_number(
            round_trip.get("expected_prev_sample_relative_l2_error"),
            "expected_prev_sample_relative_l2_error",
        )
        max_abs_error = _one_number(
            round_trip.get("native_round_trip_max_abs_error"),
            "native_round_trip_max_abs_error",
        )
        if not 0.0 <= pred_error <= 0.01 or not 0.0 <= prev_error <= 1e-3:
            raise ValueError("native Euler round trip exceeds its frozen bounds")
        if max_abs_error < 0.0:
            raise ValueError("native Euler round-trip absolute error is negative")
        mapped = _require_mapping(
            raw.get("scheduler_mapped_intervention"),
            "scheduler-mapped intervention",
        )
        mapped_ratio = _one_number(
            mapped.get("applied_update_ratio"), "mapped applied_update_ratio"
        )
        mapped_error = _one_number(
            mapped.get("target_ratio_error"), "mapped target_ratio_error"
        )
        if (
            not 0.0 <= mapped_ratio < HARD_UPDATE_CAP
            or not 0.0 <= mapped_error <= 5e-4
            or abs(mapped_ratio - TARGET_UPDATE_RATIO) > 5e-4
            or abs(mapped_error - abs(mapped_ratio - TARGET_UPDATE_RATIO)) > 1e-6
        ):
            raise ValueError("mapped renderer ratio violates its frozen bounds")
        cap_hit = mapped.get("cap_hit")
        if not isinstance(cap_hit, list) or cap_hit != [False]:
            raise ValueError("mapped renderer unexpectedly hit the hard cap")
        solver_target = _one_number(
            mapped.get("solver_target_update_ratio"),
            "solver_target_update_ratio",
        )
        if abs(solver_target - target) > 1e-6:
            raise ValueError("mapped solver target differs from renderer diagnostics")
        solver_evaluations = mapped.get("solver_evaluations")
        if (
            isinstance(solver_evaluations, bool)
            or not isinstance(solver_evaluations, int)
            or not 1 <= solver_evaluations <= 15
        ):
            raise ValueError("mapped solver evaluation count is invalid")
        _validate_provider_step(raw.get("provider_diagnostics"), task, index)


def validate_sidecar(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    run_contract_sha256: str,
    config_sha256: str,
    git_commit: str,
    generation_environment_sha256: str,
    generation_attempt_id: str,
    generation_attempt_sha256: str,
    output_dir: Path,
) -> None:
    if record.get("schema") != SIDECAR_SCHEMA:
        raise ValueError("sidecar schema differs")
    if record.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError("sidecar experiment id differs")
    for key in (
        "id",
        "prompt",
        "prompt_index",
        "prompt_row_id",
        "source_row_id",
        "bucket",
        "benchmark_style",
        "style_index",
        "seed",
        "action_id",
        "action_type",
        "execution_rank",
        "device",
        "physical_device_index",
        "task_sha256",
    ):
        if record.get(key) != task.get(key):
            raise ValueError(f"sidecar {key} differs from the frozen task")
    if record.get("action") != task.get("action"):
        raise ValueError("sidecar action differs from the frozen task")
    if record.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("sidecar run-contract hash differs")
    if record.get("config_sha256") != config_sha256:
        raise ValueError("sidecar frozen-config hash differs")
    if record.get("git_commit") != git_commit:
        raise ValueError("sidecar Git commit differs")
    if record.get("generation_environment_sha256") != generation_environment_sha256:
        raise ValueError("sidecar generation-environment hash differs")
    if (
        not isinstance(generation_attempt_id, str)
        or not ATTEMPT_ID_RE.fullmatch(generation_attempt_id)
        or record.get("generation_attempt_id") != generation_attempt_id
    ):
        raise ValueError("sidecar generation-attempt id differs")
    if (
        not isinstance(generation_attempt_sha256, str)
        or not SHA256_RE.fullmatch(generation_attempt_sha256)
        or record.get("generation_attempt_sha256") != generation_attempt_sha256
    ):
        raise ValueError("sidecar generation-attempt hash differs")
    image_path = f"images/{task['id']}.png"
    if record.get("image_path") != image_path:
        raise ValueError("sidecar image path differs")
    image_sha256 = record.get("image_sha256")
    if not isinstance(image_sha256, str) or not SHA256_RE.fullmatch(image_sha256):
        raise ValueError("sidecar image hash is invalid")
    png_path = output_dir / image_path
    if not _regular_file(png_path) or sha256_file(png_path) != image_sha256:
        raise ValueError("sidecar image is missing, unsafe, or changed")
    if record.get("height") != RESOLUTION or record.get("width") != RESOLUTION:
        raise ValueError("sidecar image dimensions differ")
    if _png_dimensions(png_path) != (RESOLUTION, RESOLUTION):
        raise ValueError("actual PNG dimensions differ from the frozen resolution")
    if record.get("num_inference_steps") != NUM_INFERENCE_STEPS:
        raise ValueError("sidecar NFE differs")
    if record.get("guidance_scale") != 7.5 or record.get("guidance_rescale") != 0.0:
        raise ValueError("sidecar CFG settings differ")
    if record.get("unet_calls_per_step") != [1] * NUM_INFERENCE_STEPS:
        raise ValueError("sidecar violates one U-Net call per step")
    if record.get("scheduler_calls_per_step") != [1] * NUM_INFERENCE_STEPS:
        raise ValueError("sidecar violates one scheduler call per step")
    call_totals = _require_mapping(record.get("call_totals"), "call totals")
    if call_totals.get("unet_calls") != 50 or call_totals.get("scheduler_calls") != 50:
        raise ValueError("sidecar call totals differ")
    if call_totals.get("extra_unet_calls") != 0:
        raise ValueError("sidecar recorded an extra U-Net call")
    if call_totals.get("extra_scheduler_calls") != 0:
        raise ValueError("sidecar recorded an extra scheduler call")
    if call_totals.get("backbone_backward_calls") != 0:
        raise ValueError("sidecar recorded a frozen-backbone backward call")
    if call_totals.get("intermediate_decode_calls") != 0:
        raise ValueError("sidecar recorded an intermediate decode")
    if call_totals.get("final_decode_calls") != 1:
        raise ValueError("sidecar final decode count differs")
    schedule = _require_mapping(record.get("scheduler_schedule"), "scheduler schedule")
    if len(schedule.get("timesteps", [])) != 50 or len(schedule.get("sigmas", [])) != 51:
        raise ValueError("sidecar scheduler schedule is incomplete")
    schedule_payload = {
        "timesteps": schedule["timesteps"],
        "sigmas": schedule["sigmas"],
    }
    if schedule.get("schedule_sha256") != canonical_sha256(schedule_payload):
        raise ValueError("sidecar scheduler schedule hash differs")
    for key in ("construction_init_noise_sigma", "effective_init_noise_sigma"):
        if not _finite_number(schedule.get(key)) or float(schedule[key]) <= 0.0:
            raise ValueError(f"sidecar scheduler {key} is invalid")
    if record.get("latent_renderer_scheduler_mapping") != task["action"][
        "scheduler_mapping"
    ]:
        raise ValueError("sidecar renderer scheduler mapping differs")
    _validate_renderer_steps(record.get("latent_renderer_step_diagnostics"), task)
    worker = _require_mapping(record.get("worker_device_provenance"), "worker device")
    if worker.get("requested_device") != task["device"]:
        raise ValueError("worker requested device differs from the task")
    if worker.get("physical_device_index") != task["physical_device_index"]:
        raise ValueError("worker physical device differs from the task")
    if worker.get("cuda_visible_devices") is not None:
        raise ValueError("worker used CUDA_VISIBLE_DEVICES remapping")
    for key in ("gpu_uuid", "torch_device_uuid", "pci_bus_id", "gpu"):
        if not isinstance(worker.get(key), str) or not worker[key]:
            raise ValueError(f"worker device identity lacks {key}")
    worker_pid = worker.get("worker_pid")
    worker_start_ticks = worker.get("worker_process_start_ticks")
    if (
        isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid <= 0
        or isinstance(worker_start_ticks, bool)
        or not isinstance(worker_start_ticks, int)
        or worker_start_ticks < 0
    ):
        raise ValueError("worker process provenance is invalid")
    if worker["torch_device_uuid"].lower() != worker["gpu_uuid"].lower():
        raise ValueError("worker Torch and NVIDIA GPU UUIDs differ")
    worker_model = record.get("worker_model_snapshot_provenance")
    _validate_worker_model_snapshot_evidence(worker_model)
    if record.get("worker_model_snapshot_sha256") != canonical_sha256(worker_model):
        raise ValueError("worker model-snapshot evidence hash differs")
    for key in (
        "initial_latent_sha256",
        "final_latent_sha256",
        "image_sha256",
        "task_sha256",
        "run_contract_sha256",
    ):
        value = record.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(f"sidecar {key} is not a SHA-256 digest")
    before = record.get("latents_before_step_sha256")
    after = record.get("latents_after_step_sha256")
    for values, label in ((before, "before-step"), (after, "after-step")):
        if (
            not isinstance(values, list)
            or len(values) != NUM_INFERENCE_STEPS
            or any(
                not isinstance(value, str) or not SHA256_RE.fullmatch(value)
                for value in values
            )
        ):
            raise ValueError(f"sidecar {label} latent hash ledger is invalid")
    if before[0] != record["initial_latent_sha256"]:
        raise ValueError("sidecar initial latent hash differs from its step ledger")
    if after[-1] != record["final_latent_sha256"] or before[1:] != after[:-1]:
        raise ValueError("sidecar latent hash chain is broken")


def _load_complete_record(
    output_dir: Path,
    task: Mapping[str, Any],
    run_config: Mapping[str, Any],
    accepted_index: Mapping[str, Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    png_path, json_path = _image_paths(output_dir, str(task["id"]))
    accepted = accepted_index.get(str(task["id"]))
    if not os.path.lexists(png_path) and not os.path.lexists(json_path):
        if accepted is not None:
            raise ValueError(f"{task['id']}: accepted output pair is missing")
        return None
    if not _regular_file(png_path) or not _regular_file(json_path):
        raise ValueError(f"{task['id']}: output pair is incomplete or unsafe")
    if accepted is None:
        raise ValueError(f"{task['id']}: output is not covered by an accepted attempt")
    record = _read_json(json_path, f"sidecar {task['id']}")
    validate_sidecar(
        record,
        task,
        run_contract_sha256=str(run_config["run_contract_sha256"]),
        config_sha256=str(run_config["config"]["sha256"]),
        git_commit=str(run_config["git_commit"]),
        generation_environment_sha256=str(
            run_config["generation_environment"]["sha256"]
        ),
        generation_attempt_id=str(accepted["generation_attempt_id"]),
        generation_attempt_sha256=str(accepted["generation_attempt_sha256"]),
        output_dir=output_dir,
    )
    if (
        record.get("image_sha256") != accepted.get("image_sha256")
        or sha256_file(json_path) != accepted.get("sidecar_sha256")
    ):
        raise ValueError(f"{task['id']}: accepted artifact hashes differ")
    return _manifest_row(
        record,
        json_path,
        accepted_receipt_sha256=str(accepted["accepted_receipt_sha256"]),
    )


def _manifest_row(
    record: Mapping[str, Any],
    sidecar_path: Path,
    *,
    accepted_receipt_sha256: str,
) -> dict[str, Any]:
    """Build the compact scorer index while keeping step ledgers in sidecars."""
    copied_fields = (
        "experiment_id",
        "id",
        "prompt",
        "prompt_index",
        "prompt_row_id",
        "source_row_id",
        "bucket",
        "benchmark_style",
        "style_index",
        "seed",
        "action_id",
        "action_type",
        "action",
        "execution_rank",
        "device",
        "physical_device_index",
        "task_sha256",
        "run_contract_sha256",
        "config_sha256",
        "git_commit",
        "generation_environment_sha256",
        "generation_attempt_id",
        "generation_attempt_sha256",
        "worker_model_snapshot_sha256",
        "image_path",
        "image_sha256",
        "height",
        "width",
        "num_inference_steps",
        "initial_latent_sha256",
        "final_latent_sha256",
        "latent_renderer_scheduler_mapping",
    )
    missing = [field for field in copied_fields if field not in record]
    if missing:
        raise ValueError(f"sidecar lacks compact-manifest fields: {missing}")
    if not SHA256_RE.fullmatch(accepted_receipt_sha256):
        raise ValueError("accepted generation receipt hash is invalid")
    worker = _require_mapping(record.get("worker_device_provenance"), "worker device")
    schedule = _require_mapping(record.get("scheduler_schedule"), "scheduler schedule")
    steps = record.get("latent_renderer_step_diagnostics")
    if not isinstance(steps, list):
        raise ValueError("sidecar renderer step ledger must be a list")
    row = {
        "schema": MANIFEST_SCHEMA,
        **{field: copy.deepcopy(record[field]) for field in copied_fields},
        "scheduler_schedule_sha256": schedule.get("schedule_sha256"),
        "renderer_active": len(steps) > 0,
        "renderer_step_count": len(steps),
        "worker_gpu_uuid": worker.get("gpu_uuid"),
        "worker_pci_bus_id": worker.get("pci_bus_id"),
        "sidecar_path": f"images/{record['id']}.json",
        "sidecar_sha256": sha256_file(sidecar_path),
        "generation_attempt_receipt_sha256": accepted_receipt_sha256,
    }
    for field in SCORE_MANIFEST_FIELDS:
        if field not in row:
            raise ValueError(f"compact manifest lacks score.py field {field}")
    return row


def validate_complete_matrix(
    records: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]]
) -> None:
    if len(tasks) != EXPECTED_TASK_COUNT or len(records) != EXPECTED_TASK_COUNT:
        raise ValueError("final HPSv2 manifest requires exactly 12,800 records")
    expected = {str(task["id"]): task for task in tasks}
    observed: dict[str, Mapping[str, Any]] = {}
    blocks: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        task_id = record.get("id")
        if not isinstance(task_id, str) or task_id not in expected:
            raise ValueError("final manifest contains an unregistered task id")
        if task_id in observed:
            raise ValueError("final manifest contains a duplicate task id")
        task = expected[task_id]
        for field in (
            "prompt",
            "prompt_index",
            "prompt_row_id",
            "source_row_id",
            "bucket",
            "benchmark_style",
            "style_index",
            "seed",
            "action_id",
            "action_type",
            "action",
            "execution_rank",
            "device",
            "physical_device_index",
            "task_sha256",
        ):
            if record.get(field) != task.get(field):
                raise ValueError(f"final manifest {task_id} field {field} differs")
        if record.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("final manifest row schema differs")
        if not isinstance(
            record.get("generation_attempt_receipt_sha256"), str
        ) or not SHA256_RE.fullmatch(record["generation_attempt_receipt_sha256"]):
            raise ValueError("final manifest accepted-attempt receipt hash is invalid")
        for field in SCORE_MANIFEST_FIELDS:
            if field not in record:
                raise ValueError(f"final manifest lacks score.py field {field}")
        physical_no_op = bool(task["action"]["physical_no_op"])
        expected_step_count = 0 if physical_no_op else NUM_INFERENCE_STEPS
        if record.get("renderer_active") is not (not physical_no_op):
            raise ValueError("final manifest renderer activation differs")
        if record.get("renderer_step_count") != expected_step_count:
            raise ValueError("final manifest renderer step count differs")
        if record.get("latent_renderer_scheduler_mapping") != task["action"][
            "scheduler_mapping"
        ]:
            raise ValueError("final manifest renderer mapping differs")
        observed[task_id] = record
        blocks.setdefault(int(record["prompt_index"]), []).append(record)
    if set(observed) != set(expected) or len(blocks) != EXPECTED_PROMPT_COUNT:
        raise ValueError("final manifest is not the complete frozen Cartesian product")
    expected_actions = {row[0] for row in EXPECTED_SETTINGS}
    for prompt_index in range(EXPECTED_PROMPT_COUNT):
        rows = blocks.get(prompt_index, [])
        if len(rows) != EXPECTED_SETTING_COUNT:
            raise ValueError("each prompt must contain exactly four paired settings")
        if {row["action_id"] for row in rows} != expected_actions:
            raise ValueError("prompt block action set differs")
        if len({row["device"] for row in rows}) != 1:
            raise ValueError("paired settings for one prompt crossed GPUs")
        gpu_uuids = {row.get("worker_gpu_uuid") for row in rows}
        if None in gpu_uuids or "" in gpu_uuids or len(gpu_uuids) != 1:
            raise ValueError("paired settings for one prompt crossed physical GPU UUIDs")
        pci_bus_ids = {row.get("worker_pci_bus_id") for row in rows}
        if None in pci_bus_ids or "" in pci_bus_ids or len(pci_bus_ids) != 1:
            raise ValueError("paired settings for one prompt crossed physical PCI devices")
        if len({row["initial_latent_sha256"] for row in rows}) != 1:
            raise ValueError("paired settings for one prompt used different initial noise")
        if len({row["scheduler_schedule_sha256"] for row in rows}) != 1:
            raise ValueError("paired settings for one prompt used different schedules")
        if len({row["sidecar_path"] for row in rows}) != EXPECTED_SETTING_COUNT:
            raise ValueError("paired settings do not have distinct sidecars")
        if len({row["sidecar_sha256"] for row in rows}) != EXPECTED_SETTING_COUNT:
            raise ValueError("paired setting sidecars are byte-identical")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not _regular_file(path):
        raise ValueError(f"{label} is missing or unsafe: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                rows.append(_require_mapping(value, f"{label} line {line_number}"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return rows


def _validate_recorded_run_config(
    run_config: Mapping[str, Any],
    contract: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> None:
    if run_config.get("schema") != RUN_CONFIG_SCHEMA:
        raise ValueError("recorded run config schema differs")
    git_commit = run_config.get("git_commit")
    if not isinstance(git_commit, str) or not GIT_COMMIT_RE.fullmatch(git_commit):
        raise ValueError("recorded run config Git commit is invalid")
    generation_environment = run_config.get("generation_environment")
    model_snapshot_record = run_config.get("model_snapshot")
    _validate_generation_environment(generation_environment)
    _validate_model_snapshot_record(model_snapshot_record)
    expected = _run_config(
        contract,
        EXPECTED_DEVICES,
        tasks,
        git_commit,
        generation_environment=generation_environment,
        model_snapshot_record=model_snapshot_record,
    )
    if dict(run_config) != expected:
        raise ValueError("recorded run config differs from current frozen inputs")


def validate_complete_run(
    run_dir: Path,
    config: Path = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    """Read-only, fail-closed audit of the complete frozen HPSv2 run."""
    contract = load_contract(Path(config))
    tasks = build_tasks(contract, EXPECTED_DEVICES)
    output_dir = Path(run_dir).resolve()
    run_config = _read_json(output_dir / "config.json", "recorded run config")
    _validate_recorded_run_config(run_config, contract, tasks)
    accepted_index, running = _load_generation_attempt_index(
        output_dir, run_config, tasks
    )
    if running:
        raise ValueError("complete-run audit found an unterminated generation attempt")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        row = _load_complete_record(
            output_dir,
            task,
            run_config,
            accepted_index,
        )
        if row is None:
            raise ValueError(f"complete-run audit is missing task {task['id']}")
        rows.append(row)
    validate_complete_matrix(rows, tasks)
    final_rows = _read_jsonl(output_dir / FINAL_MANIFEST_NAME, "final manifest")
    if final_rows != rows:
        raise ValueError("final manifest differs from validated sidecars")
    partial_rows = _read_jsonl(
        output_dir / PARTIAL_MANIFEST_NAME,
        "partial manifest",
    )
    if partial_rows != rows:
        raise ValueError("partial manifest differs from the complete final manifest")
    return rows


def _consolidate(
    contract: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
    accepted_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        record = _load_complete_record(
            contract["output_dir"], task, run_config, accepted_index
        )
        if record is None:
            pending.append(dict(task))
        else:
            complete.append(record)
    _atomic_write_jsonl(
        contract["output_dir"] / PARTIAL_MANIFEST_NAME,
        complete,
    )
    final_path = contract["output_dir"] / FINAL_MANIFEST_NAME
    if pending:
        if os.path.lexists(final_path):
            raise ValueError("final manifest exists while the matrix is incomplete")
    else:
        validate_complete_matrix(complete, tasks)
        _atomic_write_jsonl(final_path, complete)
    return complete, pending


def _run_config(
    contract: Mapping[str, Any],
    devices: Sequence[int],
    tasks: Sequence[Mapping[str, Any]],
    git_commit: str,
    *,
    generation_environment: Mapping[str, Any],
    model_snapshot_record: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_generation_environment(generation_environment)
    _validate_model_snapshot_record(model_snapshot_record)
    runner_sha256 = sha256_file(ROOT / RUNNER_RELATIVE_PATH)
    payload = {
        "schema": RUN_CONFIG_SCHEMA,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "git_commit": git_commit,
        "generation_environment": copy.deepcopy(dict(generation_environment)),
        "model_snapshot": copy.deepcopy(dict(model_snapshot_record)),
        "runner": {"path": RUNNER_RELATIVE_PATH, "sha256": runner_sha256},
        "config": {
            "path": contract["config_path"].relative_to(ROOT).as_posix(),
            "sha256": contract["config_sha256"],
        },
        "prompt_csv": {
            "path": contract["prompt_csv"].relative_to(ROOT).as_posix(),
            "sha256": contract["prompt_csv_sha256"],
            "record_count": EXPECTED_PROMPT_COUNT,
        },
        "prompt_manifest": {
            "path": contract["prompt_manifest"].relative_to(ROOT).as_posix(),
            "sha256": contract["prompt_manifest_sha256"],
        },
        "devices": [int(value) for value in devices],
        "device_assignment": DEVICE_ASSIGNMENT,
        "setting_order": ACTION_ORDER_NAMESPACE,
        "seed_policy": EXPECTED_SEED_POLICY,
        "base_seed": EXPECTED_BASE_SEED,
        "seed_range": [
            EXPECTED_BASE_SEED,
            EXPECTED_BASE_SEED + EXPECTED_PROMPT_COUNT - 1,
        ],
        "prompt_count": EXPECTED_PROMPT_COUNT,
        "setting_count": EXPECTED_SETTING_COUNT,
        "expected_image_count": EXPECTED_TASK_COUNT,
        "task_matrix_sha256": canonical_sha256(tasks),
        "settings": copy.deepcopy(contract["settings"]),
        "sampling": copy.deepcopy(contract["sampling"]),
        "scoring": copy.deepcopy(contract["config"]["scoring"]),
    }
    payload["run_contract_sha256"] = canonical_sha256(payload)
    return payload


def _prepare_output(
    contract: Mapping[str, Any], run_config: Mapping[str, Any]
) -> None:
    output_dir = contract["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if os.path.lexists(images_dir):
        if images_dir.is_symlink() or not images_dir.is_dir():
            raise ValueError("run images directory is unsafe")
    else:
        images_dir.mkdir()
    config_path = output_dir / "config.json"
    if os.path.lexists(config_path):
        if not _regular_file(config_path):
            raise ValueError("run config is unsafe")
        if _read_json(config_path, "run config") != dict(run_config):
            raise ValueError("existing run config differs; resume is forbidden")
    else:
        _atomic_write_json(config_path, run_config)


def _tensor_sha256(value: Any) -> str:
    raw = value.detach().contiguous().cpu().view(sys.modules["torch"].uint8)
    raw = raw.reshape(-1)
    try:
        payload = raw.numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError):
        payload = bytes(raw.tolist())
    return sha256_bytes(payload)


def _torch_cuda_uuid(properties: Any) -> str:
    value = getattr(properties, "uuid", None)
    if value is None:
        raise RuntimeError("Torch CUDA properties do not expose a GPU UUID")
    raw = str(value).strip().lower()
    if raw.startswith("gpu-"):
        raw = raw[4:]
    if re.fullmatch(
        r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", raw
    ) is None:
        raise RuntimeError("Torch CUDA GPU UUID is invalid")
    return f"GPU-{raw}"


def _cuda_device_identity(
    torch: Any,
    device_index: int,
    *,
    expected_identity: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if "CUDA_VISIBLE_DEVICES" in os.environ or "CUDA_DEVICE_ORDER" in os.environ:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER must be unset; device ids are physical"
        )
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise RuntimeError(f"physical cuda:{device_index} is unavailable")
    torch.cuda.set_device(device_index)
    if torch.cuda.current_device() != device_index:
        raise RuntimeError("CUDA current device differs from the assigned physical GPU")
    properties = torch.cuda.get_device_properties(device_index)
    torch_uuid = _torch_cuda_uuid(properties)
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    matches = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) == 5 and fields[1].lower() == torch_uuid.lower():
            matches.append(fields)
    if len(matches) != 1:
        raise RuntimeError("Torch GPU UUID is absent or ambiguous in nvidia-smi")
    physical, gpu_uuid, pci_bus_id, gpu_name, total_memory_mib = matches[0]
    if int(physical) != int(device_index):
        raise RuntimeError("Torch GPU UUID maps to a different physical GPU index")
    if gpu_uuid.lower() != torch_uuid.lower():
        raise RuntimeError("Torch and nvidia-smi disagree on the assigned GPU UUID")
    if gpu_name != str(properties.name):
        raise RuntimeError("Torch and nvidia-smi disagree on the assigned GPU name")
    identity = {
        "requested_device": f"cuda:{device_index}",
        "logical_device_index": int(torch.cuda.current_device()),
        "physical_device_index": int(physical),
        "gpu_uuid": gpu_uuid,
        "torch_device_uuid": torch_uuid,
        "pci_bus_id": pci_bus_id,
        "gpu": gpu_name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "nvidia_smi_total_memory_mib": int(total_memory_mib),
        "cuda_visible_devices": None,
    }
    if expected_identity is not None:
        expected = {
            "index": identity["physical_device_index"],
            "uuid": identity["gpu_uuid"],
            "pci_bus_id": identity["pci_bus_id"],
            "name": identity["gpu"],
            "total_memory_mib": identity["nvidia_smi_total_memory_mib"],
        }
        observed = {
            key: expected_identity.get(key)
            for key in (
                "index",
                "uuid",
                "pci_bus_id",
                "name",
                "total_memory_mib",
            )
        }
        if observed != expected:
            raise RuntimeError(
                "worker CUDA identity differs from the frozen run inventory"
            )
    return identity


def _reset_pipeline_ledgers(pipe: Any) -> None:
    for name, value in (
        ("_last_unet_calls_total", 0),
        ("_last_unet_calls_per_step", []),
        ("_last_scheduler_calls_total", 0),
        ("_last_scheduler_calls_per_step", []),
        ("_last_final_decode_calls", 0),
        ("_last_intermediate_decode_calls", 0),
        ("_last_latent_renderer_scheduler_mapping", None),
        ("_last_latent_renderer_diagnostics", None),
        ("_last_latent_renderer_provider_diagnostics", None),
        ("_last_latent_renderer_step_diagnostics", []),
        ("_last_scheduler_schedule_record", None),
        ("_last_prepared_initial_latent_sha256", None),
        ("_last_latents_before_step_sha256", []),
        ("_last_latents_after_step_sha256", []),
    ):
        setattr(pipe, name, value)


def _hook_signature(unet: Any) -> tuple[tuple[int, int], ...]:
    hooks = unet.up_blocks[0]._forward_pre_hooks
    return tuple((int(key), id(value)) for key, value in hooks.items())


def _png_bytes(image: Any) -> bytes:
    if getattr(image, "size", None) != (RESOLUTION, RESOLUTION):
        raise RuntimeError("pipeline returned an image with non-1024 dimensions")
    if getattr(image, "mode", None) != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    payload = buffer.getvalue()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("pipeline output did not encode as PNG")
    return payload


def _runtime_action(task: Mapping[str, Any], pipe: Any) -> tuple[Any, Any]:
    action = task["action"]
    if action["physical_no_op"]:
        return None, None
    from AttentionGuidance.adaptive_oracle import (
        AdaptiveOracleBasisProvider,
        AdaptiveOracleRandomContext,
        FixedRatioMomentGeodesicRenderer,
    )

    random_context = None
    if action["affinity_source"] == "random_edge":
        random_context = AdaptiveOracleRandomContext(
            experiment_id=EXPECTED_EXPERIMENT_ID,
            split_role="hpsv2_full",
            prompt_row_id=task["prompt_row_id"],
            seed=int(task["seed"]),
        )
    provider = AdaptiveOracleBasisProvider(
        pipe.unet if action["affinity_source"] == "feature" else None,
        batch_size=1,
        affinity_source=action["affinity_source"],
        orbit_name=ORBIT_NAME,
        random_context=random_context,
    )
    renderer = FixedRatioMomentGeodesicRenderer(
        sign=1,
        target_update_ratio=TARGET_UPDATE_RATIO,
        hard_update_cap=HARD_UPDATE_CAP,
    )
    return provider, renderer


def _sidecar_from_pipeline(
    *,
    task: Mapping[str, Any],
    pipe: Any,
    device_identity: Mapping[str, Any],
    model_snapshot_evidence: Mapping[str, Any],
    png_sha256: str,
    run_config: Mapping[str, Any],
    generation_attempt_id: str,
    generation_attempt_sha256: str,
    elapsed: float,
    peak_memory: int,
) -> dict[str, Any]:
    unet_steps = [int(value) for value in pipe._last_unet_calls_per_step]
    scheduler_steps = [int(value) for value in pipe._last_scheduler_calls_per_step]
    if unet_steps != [1] * NUM_INFERENCE_STEPS:
        raise RuntimeError("pipeline violated one ordinary U-Net call per step")
    if scheduler_steps != [1] * NUM_INFERENCE_STEPS:
        raise RuntimeError("pipeline violated one scheduler call per step")
    if pipe._last_unet_calls_total != 50 or pipe._last_scheduler_calls_total != 50:
        raise RuntimeError("pipeline call totals differ from the 50-step contract")
    schedule = copy.deepcopy(pipe._last_scheduler_schedule_record)
    if not isinstance(schedule, dict):
        raise RuntimeError("pipeline omitted scheduler schedule diagnostics")
    if len(schedule.get("timesteps", [])) != 50 or len(schedule.get("sigmas", [])) != 51:
        raise RuntimeError("pipeline scheduler schedule diagnostics are incomplete")
    before = list(pipe._last_latents_before_step_sha256)
    after = list(pipe._last_latents_after_step_sha256)
    initial = pipe._last_prepared_initial_latent_sha256
    if len(before) != 50 or len(after) != 50 or before[0] != initial:
        raise RuntimeError("pipeline latent hash ledger is incomplete")
    if before[1:] != after[:-1]:
        raise RuntimeError("pipeline latent hash chain is broken")
    action = task["action"]
    renderer_steps = copy.deepcopy(pipe._last_latent_renderer_step_diagnostics)
    expected_mapping = action["scheduler_mapping"]
    if pipe._last_latent_renderer_scheduler_mapping != expected_mapping:
        raise RuntimeError("pipeline renderer scheduler mapping differs")
    _validate_renderer_steps(renderer_steps, task)
    last_renderer = None
    last_provider = None
    if not action["physical_no_op"]:
        last_renderer = pipe._last_latent_renderer_diagnostics.to_record()
        last_provider = copy.deepcopy(pipe._last_latent_renderer_provider_diagnostics)
        if renderer_steps[-1]["provider_diagnostics"] != last_provider:
            raise RuntimeError("last provider diagnostics differ from the step ledger")
    _validate_worker_model_snapshot_evidence(model_snapshot_evidence)
    model_snapshot_evidence = copy.deepcopy(dict(model_snapshot_evidence))
    record = {
        "schema": SIDECAR_SCHEMA,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        **copy.deepcopy(dict(task)),
        "run_contract_sha256": run_config["run_contract_sha256"],
        "config_sha256": run_config["config"]["sha256"],
        "git_commit": run_config["git_commit"],
        "generation_environment_sha256": run_config["generation_environment"][
            "sha256"
        ],
        "generation_attempt_id": generation_attempt_id,
        "generation_attempt_sha256": generation_attempt_sha256,
        "image_path": f"images/{task['id']}.png",
        "image_sha256": png_sha256,
        "height": RESOLUTION,
        "width": RESOLUTION,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": 7.5,
        "guidance_rescale": 0.0,
        "inference_seconds": float(elapsed),
        "peak_gpu_memory_bytes": int(peak_memory),
        "worker_device_provenance": copy.deepcopy(dict(device_identity)),
        "worker_model_snapshot_provenance": model_snapshot_evidence,
        "worker_model_snapshot_sha256": canonical_sha256(model_snapshot_evidence),
        "unet_calls_per_step": unet_steps,
        "scheduler_calls_per_step": scheduler_steps,
        "call_totals": {
            "unet_calls": 50,
            "scheduler_calls": 50,
            "extra_unet_calls": 0,
            "extra_scheduler_calls": 0,
            "backbone_backward_calls": 0,
            "intermediate_decode_calls": int(pipe._last_intermediate_decode_calls),
            "final_decode_calls": int(pipe._last_final_decode_calls),
        },
        "scheduler_schedule": schedule,
        "initial_latent_sha256": initial,
        "final_latent_sha256": after[-1],
        "latents_before_step_sha256": before,
        "latents_after_step_sha256": after,
        "latent_renderer_scheduler_mapping": expected_mapping,
        "latent_renderer_diagnostics": last_renderer,
        "latent_renderer_provider_diagnostics": last_provider,
        "latent_renderer_step_diagnostics": renderer_steps,
    }
    return record


def _execute_task(
    *,
    task: Mapping[str, Any],
    pipe: Any,
    scheduler_class: Any,
    base_scheduler_config: Mapping[str, Any],
    raw_noise: Any,
    device_identity: Mapping[str, Any],
    model_snapshot_evidence: Mapping[str, Any],
    run_config: Mapping[str, Any],
    generation_attempt_id: str,
    generation_attempt_sha256: str,
    torch: Any,
) -> tuple[dict[str, Any], bytes]:
    _reset_pipeline_ledgers(pipe)
    baseline_hooks = _hook_signature(pipe.unet)
    pipe.scheduler = scheduler_class.from_config(copy.deepcopy(dict(base_scheduler_config)))
    if type(pipe.scheduler).__name__ != "EulerDiscreteScheduler":
        raise RuntimeError("fresh task scheduler is not EulerDiscreteScheduler")
    if pipe.scheduler.config.prediction_type != "epsilon":
        raise RuntimeError("fresh task scheduler prediction type is not epsilon")
    provider = renderer = None
    raw_hash = _tensor_sha256(raw_noise)
    try:
        provider, renderer = _runtime_action(task, pipe)
        task_generator = torch.Generator(device=task["device"]).manual_seed(
            int(task["seed"])
        )
        torch.cuda.synchronize(task["physical_device_index"])
        torch.cuda.reset_peak_memory_stats(task["physical_device_index"])
        started = time.perf_counter()
        images = pipe(
            prompt=task["prompt"],
            negative_prompt=(
                "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
            ),
            generator=task_generator,
            latents=raw_noise.clone(),
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=7.5,
            guidance_rescale=0.0,
            eta=0.0,
            output_type="pil",
            record_latent_audit=True,
            show_image=False,
            image_lr=None,
            lowvram=False,
            models_to_cpu=False,
            multi_encoder=False,
            multi_decoder=False,
            num_resample_timesteps=50,
            init_rates=[0.8],
            attn_guidance_scale=0.0,
            attn_guidance_controller=None,
            power_calibrate=0,
            semantic_transport_config=None,
            freeu_schedule=None,
            trajectory_correction=None,
            latent_renderer=renderer,
            latent_renderer_basis_provider=provider,
            latent_renderer_scheduler_mapping=(
                "legacy_unit" if renderer is None else "euler_clean_endpoint"
            ),
        )
        torch.cuda.synchronize(task["physical_device_index"])
        elapsed = time.perf_counter() - started
        peak_memory = torch.cuda.max_memory_allocated(task["physical_device_index"])
        if not isinstance(images, list) or len(images) != 1:
            raise RuntimeError("pipeline must return exactly one final image")
        if _tensor_sha256(raw_noise) != raw_hash:
            raise RuntimeError("pipeline mutated the paired initial noise")
        if _hook_signature(pipe.unet) != baseline_hooks:
            raise RuntimeError("adaptive-oracle feature hook leaked after generation")
        png = _png_bytes(images[0])
        sidecar = _sidecar_from_pipeline(
            task=task,
            pipe=pipe,
            device_identity=device_identity,
            model_snapshot_evidence=model_snapshot_evidence,
            png_sha256=sha256_bytes(png),
            run_config=run_config,
            generation_attempt_id=generation_attempt_id,
            generation_attempt_sha256=generation_attempt_sha256,
            elapsed=elapsed,
            peak_memory=peak_memory,
        )
        return sidecar, png
    finally:
        if provider is not None:
            provider.last_diagnostics = None
        provider = renderer = None
        _reset_pipeline_ledgers(pipe)


def _publish_generated_artifacts(
    *,
    png_path: Path,
    json_path: Path,
    png: bytes,
    sidecar: Mapping[str, Any],
    abort_event: Any,
    publication_lock: Any,
) -> None:
    """Publish one pair only while no monitor or peer has requested abort."""
    with publication_lock:
        if abort_event.is_set():
            raise RuntimeError("generation aborted before artifact publication")
        _atomic_write_bytes(png_path, png)
        _atomic_write_json(json_path, sidecar)


def _request_generation_abort(abort_event: Any) -> None:
    """Signal rejection without waiting on a lock a dead worker may own."""
    abort_event.set()


def _worker(
    device_index: int,
    blocks: Sequence[Sequence[Mapping[str, Any]]],
    contract_payload: Mapping[str, Any],
    run_config: Mapping[str, Any],
    abort_event: Any,
    publication_lock: Any,
    error_queue: Any,
    ready_queue: Any,
    completion_queue: Any,
    release_event: Any,
) -> None:
    current_task = "worker-startup"
    pipe = None
    stage_record = None
    torch_module = None
    try:
        worker_error = None
        try:
            import torch
            from diffusers import EulerDiscreteScheduler
            from InferencePipelines.RepLDM.pipeline_repldm_sdxl import (
                RepLDMSDXLPipeline,
            )

            torch_module = torch
            device = f"cuda:{device_index}"
            torch.set_grad_enabled(False)
            _configure_worker_determinism(torch)
            observed_environment = _generation_environment(run_config["devices"], torch)
            if observed_environment != run_config["generation_environment"]:
                raise RuntimeError("worker generation environment differs from run contract")
            expected_identities = [
                row
                for row in run_config["generation_environment"]["gpu_inventory"]
                if int(row["index"]) == int(device_index)
            ]
            if len(expected_identities) != 1:
                raise RuntimeError(
                    "worker assigned GPU is absent or duplicated in run inventory"
                )
            identity = _cuda_device_identity(
                torch,
                device_index,
                expected_identity=expected_identities[0],
            )
            worker_pid = os.getpid()
            worker_process_start_ticks = _process_start_ticks(worker_pid)
            if worker_process_start_ticks is None:
                raise RuntimeError("generation worker process start time is unavailable")
            identity["worker_pid"] = worker_pid
            identity["worker_process_start_ticks"] = worker_process_start_ticks
            context_probe = torch.empty((1,), device=device)
            torch.cuda.synchronize(device_index)
            del context_probe
            ready_queue.put(_worker_ready_record(device_index, identity))
            _validate_model_snapshot_record(run_config["model_snapshot"])
            try:
                stage_record = model_snapshot.stage_model_snapshot(
                    ROOT,
                    expected_manifest_sha256=run_config["model_snapshot"][
                        "manifest_sha256"
                    ],
                )
            except model_snapshot.ModelStageCreationError as exc:
                stage_record = exc.cleanup_record
                raise

            stage_verifications: list[dict[str, Any]] = []

            def load_pipeline(pinned_stage_path: str) -> Any:
                if not pinned_stage_path.startswith("/proc/self/fd/"):
                    raise RuntimeError("model loader did not receive a pinned stage root")
                return RepLDMSDXLPipeline.from_pretrained(
                    pinned_stage_path,
                    local_files_only=True,
                    variant=contract_payload["sampling"]["model_variant"],
                    torch_dtype=torch.float16,
                )

            pipe = model_snapshot.load_from_verified_staged_model_snapshot(
                stage_record,
                load_pipeline,
                expected_manifest_sha256=run_config["model_snapshot"][
                    "manifest_sha256"
                ],
                verification_log=stage_verifications,
            ).to(device)
            stage_verifications.append(
                model_snapshot.verify_staged_model_snapshot(
                    stage_record,
                    expected_manifest_sha256=run_config["model_snapshot"][
                        "manifest_sha256"
                    ],
                )
            )
            model_stage_evidence = _worker_model_snapshot_evidence(
                stage_record, stage_verifications
            )
            pipe.set_progress_bar_config(disable=True)
            if type(pipe).__name__ != "RepLDMSDXLPipeline":
                raise RuntimeError("loaded pipeline class differs")
            base_scheduler_config = dict(pipe.scheduler.config)
            if type(pipe.scheduler).__name__ != "EulerDiscreteScheduler":
                raise RuntimeError("loaded base scheduler differs from EulerDiscreteScheduler")
            if base_scheduler_config.get("prediction_type") != "epsilon":
                raise RuntimeError("loaded base scheduler prediction type differs")
            completed = 0
            for block in blocks:
                if abort_event.is_set():
                    break
                row_seed = int(block[0]["seed"])
                if any(int(task["seed"]) != row_seed for task in block):
                    raise RuntimeError("prompt block contains inconsistent row seeds")
                raw_generator = torch.Generator(device=device).manual_seed(row_seed)
                raw_noise = torch.randn(
                    (1, 4, RESOLUTION // 8, RESOLUTION // 8),
                    generator=raw_generator,
                    device=device,
                    dtype=torch.float16,
                )
                for task in block:
                    if abort_event.is_set():
                        break
                    current_task = str(task["id"])
                    png_path, json_path = _image_paths(
                        Path(contract_payload["output_dir"]), current_task
                    )
                    if os.path.lexists(png_path) or os.path.lexists(json_path):
                        raise RuntimeError("worker received an already-materialized task")
                    sidecar, png = _execute_task(
                        task=task,
                        pipe=pipe,
                        scheduler_class=EulerDiscreteScheduler,
                        base_scheduler_config=base_scheduler_config,
                        raw_noise=raw_noise,
                        device_identity=identity,
                        model_snapshot_evidence=model_stage_evidence,
                        run_config=run_config,
                        generation_attempt_id=str(contract_payload["attempt_id"]),
                        generation_attempt_sha256=str(
                            contract_payload["attempt_sha256"]
                        ),
                        torch=torch,
                    )
                    _publish_generated_artifacts(
                        png_path=png_path,
                        json_path=json_path,
                        png=png,
                        sidecar=sidecar,
                        abort_event=abort_event,
                        publication_lock=publication_lock,
                    )
                    completed += 1
                    print(
                        f"[cuda:{device_index}] {completed} generated; {current_task}",
                        flush=True,
                    )
            final_stage_verification = model_snapshot.verify_staged_model_snapshot(
                stage_record,
                expected_manifest_sha256=run_config["model_snapshot"][
                    "manifest_sha256"
                ],
            )
            if final_stage_verification.get("tree_sha256") != stage_verifications[0].get(
                "tree_sha256"
            ):
                raise RuntimeError("worker staged-model tree changed during generation")
            if not abort_event.is_set():
                completion_queue.put(
                    _worker_completion_record(
                        device_index, identity, completed
                    )
                )
                while not release_event.wait(0.25):
                    if abort_event.is_set():
                        break
        except BaseException as exc:
            worker_error = exc

        cleanup_error = None
        pipe = None
        if torch_module is not None:
            try:
                torch_module.cuda.empty_cache()
            except BaseException as exc:
                cleanup_error = exc
        if stage_record is not None:
            try:
                model_snapshot.cleanup_staged_model_snapshot(stage_record)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        if worker_error is not None and cleanup_error is not None:
            raise RuntimeError(
                "generation worker failed and staged-model cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            ) from worker_error
        if worker_error is not None:
            raise worker_error.with_traceback(worker_error.__traceback__)
        if cleanup_error is not None:
            raise cleanup_error.with_traceback(cleanup_error.__traceback__)
    except BaseException:
        _request_generation_abort(abort_event)
        error_queue.put((f"cuda:{device_index}", current_task, traceback.format_exc()))
        raise


def _pending_blocks(
    pending: Sequence[Mapping[str, Any]], devices: Sequence[int]
) -> dict[int, list[list[dict[str, Any]]]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = {
        int(device): {} for device in devices
    }
    for task in pending:
        physical = int(task["physical_device_index"])
        if physical not in grouped:
            raise ValueError("pending task requires a GPU absent from --devices")
        grouped[physical].setdefault(int(task["prompt_index"]), []).append(dict(task))
    result: dict[int, list[list[dict[str, Any]]]] = {}
    for device, prompts in grouped.items():
        blocks = []
        for prompt_index in sorted(prompts):
            block = sorted(prompts[prompt_index], key=lambda row: row["execution_rank"])
            blocks.append(block)
        result[device] = blocks
    return result


def _generation_attempt_batches(
    pending: Sequence[Mapping[str, Any]], devices: Sequence[int]
) -> list[list[dict[str, Any]]]:
    """Bound restart loss while keeping every prompt's settings together."""
    blocks = _pending_blocks(pending, devices)
    max_blocks = max(
        (len(device_blocks) for device_blocks in blocks.values()), default=0
    )
    batches = []
    for start in range(
        0, max_blocks, GENERATION_PROMPT_BLOCKS_PER_DEVICE_PER_ATTEMPT
    ):
        batch = []
        stop = start + GENERATION_PROMPT_BLOCKS_PER_DEVICE_PER_ATTEMPT
        for device in devices:
            for prompt_block in blocks[int(device)][start:stop]:
                batch.extend(prompt_block)
        if batch:
            batches.append(batch)
    observed_ids = [str(task["id"]) for batch in batches for task in batch]
    expected_ids = [str(task["id"]) for task in pending]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(
        expected_ids
    ):
        raise RuntimeError("generation attempt batches differ from pending tasks")
    return batches


def _run_workers(
    pending: Sequence[Mapping[str, Any]],
    devices: Sequence[int],
    contract: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> None:
    blocks = _pending_blocks(pending, devices)
    context = multiprocessing.get_context("spawn")
    abort_event = context.Event()
    publication_lock = context.Lock()
    error_queue = context.Queue()
    ready_queue = context.Queue()
    completion_queue = context.Queue()
    release_event = context.Event()
    processes = []
    process_assignments = []
    expected_workers: dict[int, dict[str, Any]] = {}
    process_by_pid = {}
    ready_pids: set[int] = set()
    completed_pids: set[int] = set()
    gpu_verified_pids: set[int] = set()

    expected_worker_start_ticks: dict[int, int] = {}
    expected_task_counts: dict[int, int] = {}

    def drain_ready_queue() -> None:
        while True:
            try:
                record = ready_queue.get_nowait()
            except queue.Empty:
                return
            pid = _validate_worker_ready_record(
                record, expected_workers, expected_worker_start_ticks
            )
            if pid in ready_pids:
                raise ValueError("generation worker emitted readiness more than once")
            ready_pids.add(pid)

    def drain_completion_queue(*, require_live_process: bool = True) -> None:
        while True:
            try:
                record = completion_queue.get_nowait()
            except queue.Empty:
                return
            pid = _validate_worker_completion_record(
                record,
                expected_workers,
                expected_worker_start_ticks,
                expected_task_counts,
                require_live_process=require_live_process,
            )
            if pid in completed_pids:
                raise ValueError("generation worker emitted completion more than once")
            completed_pids.add(pid)

    contract_payload = {
        "sampling": copy.deepcopy(contract["sampling"]),
        "output_dir": str(contract["output_dir"]),
        "attempt_id": str(contract["attempt_id"]),
        "attempt_sha256": str(contract["attempt_sha256"]),
    }
    failures = []
    lifecycle_error: Optional[BaseException] = None
    try:
        for device in devices:
            if not blocks[int(device)]:
                continue
            process = context.Process(
                target=_worker,
                args=(
                    int(device),
                    blocks[int(device)],
                    contract_payload,
                    dict(run_config),
                    abort_event,
                    publication_lock,
                    error_queue,
                    ready_queue,
                    completion_queue,
                    release_event,
                ),
                name=f"hpsv2-gpu-{device}",
            )
            try:
                process.start()
            except BaseException:
                if process.pid is not None:
                    processes.append(process)
                raise
            processes.append(process)
            process_assignments.append((process, int(device)))
        frozen_by_device = {
            int(row["index"]): dict(row)
            for row in run_config["generation_environment"]["gpu_inventory"]
        }
        for process, device in process_assignments:
            if process.pid is None or int(process.pid) <= 0:
                raise RuntimeError("generation worker started without a valid PID")
            pid = int(process.pid)
            if pid in expected_workers or device not in frozen_by_device:
                raise RuntimeError("generation worker PID or GPU assignment is invalid")
            process_start_ticks = _process_start_ticks(pid)
            if process_start_ticks is None:
                raise RuntimeError("generation worker process exited before registration")
            expected_workers[pid] = frozen_by_device[device]
            expected_worker_start_ticks[pid] = process_start_ticks
            expected_task_counts[pid] = sum(
                len(block) for block in blocks[device]
            )
            process_by_pid[pid] = process

        next_gpu_monitor = 0.0
        while any(process.is_alive() for process in processes):
            try:
                failures.append(error_queue.get(timeout=0.5))
                _request_generation_abort(abort_event)
            except queue.Empty:
                pass
            ready_count = len(ready_pids)
            try:
                drain_ready_queue()
            except BaseException:
                failures.append(("gpu-monitor", "worker-ready", traceback.format_exc()))
                _request_generation_abort(abort_event)
            if len(ready_pids) != ready_count:
                next_gpu_monitor = 0.0
            try:
                drain_completion_queue()
            except BaseException:
                failures.append(("gpu-monitor", "worker-completion", traceback.format_exc()))
                _request_generation_abort(abort_event)
            for process in processes:
                if process.exitcode not in (0, None) and not failures:
                    failures.append(
                        (process.name, "unknown", f"worker exited {process.exitcode}")
                    )
                    _request_generation_abort(abort_event)
            now = time.monotonic()
            if now >= next_gpu_monitor and not failures:
                required_worker_pids = [
                    pid
                    for pid in ready_pids | completed_pids
                    if process_by_pid[pid].is_alive()
                ]
                try:
                    conflicts = _generation_compute_conflicts(
                        run_config["generation_environment"],
                        expected_workers=expected_workers,
                        required_pids=required_worker_pids,
                        expected_worker_start_ticks=expected_worker_start_ticks,
                    )
                except BaseException:
                    failures.append(
                        ("gpu-monitor", "nvidia-smi", traceback.format_exc())
                    )
                    _request_generation_abort(abort_event)
                else:
                    if conflicts:
                        failures.append(
                            (
                                "gpu-monitor",
                                "foreign-compute-process",
                                json.dumps(conflicts, sort_keys=True),
                            )
                        )
                        _request_generation_abort(abort_event)
                    else:
                        gpu_verified_pids.update(required_worker_pids)
                        if (
                            expected_workers
                            and set(expected_workers) <= completed_pids
                            and set(expected_workers) <= gpu_verified_pids
                        ):
                            release_event.set()
                next_gpu_monitor = now + 2.0
            if failures:
                break
    except KeyboardInterrupt as exc:
        _request_generation_abort(abort_event)
        failures.append(("parent", "signal", "generation interrupted"))
        lifecycle_error = exc
    except BaseException as exc:
        _request_generation_abort(abort_event)
        lifecycle_error = exc
    finally:
        release_event.set()
        join_deadline = time.monotonic() + 60.0
        for process in processes:
            try:
                process.join(timeout=max(0.0, join_deadline - time.monotonic()))
            except BaseException:
                failures.append(
                    (process.name, "join", traceback.format_exc())
                )
        for process in processes:
            try:
                alive = process.is_alive()
            except BaseException:
                failures.append(
                    (process.name, "is-alive", traceback.format_exc())
                )
                alive = True
            if alive:
                try:
                    process.terminate()
                except BaseException:
                    failures.append(
                        (process.name, "terminate", traceback.format_exc())
                    )
        terminate_deadline = time.monotonic() + 10.0
        for process in processes:
            try:
                process.join(
                    timeout=max(0.0, terminate_deadline - time.monotonic())
                )
            except BaseException:
                failures.append(
                    (process.name, "post-terminate-join", traceback.format_exc())
                )
        for process in processes:
            try:
                alive = process.is_alive()
            except BaseException:
                alive = True
            if alive:
                try:
                    process.kill()
                    process.join(timeout=10.0)
                except BaseException:
                    failures.append(
                        (process.name, "kill", traceback.format_exc())
                    )
            try:
                if process.is_alive():
                    failures.append(
                        (process.name, "survivor", "worker survived terminate and kill")
                    )
            except BaseException:
                failures.append(
                    (process.name, "final-is-alive", traceback.format_exc())
                )
        while True:
            try:
                failures.append(error_queue.get_nowait())
            except queue.Empty:
                break
        try:
            drain_ready_queue()
        except BaseException:
            failures.append(("gpu-monitor", "worker-ready", traceback.format_exc()))
        try:
            drain_completion_queue(require_live_process=False)
        except BaseException:
            failures.append(("gpu-monitor", "worker-completion", traceback.format_exc()))
        failures.extend(
            _worker_handshake_failures(
                expected_workers,
                ready_pids,
                completed_pids,
                gpu_verified_pids,
            )
        )
        for process in processes:
            if process.exitcode not in (0, None) and not failures:
                failures.append(
                    (process.name, "unknown", f"worker exited {process.exitcode}")
                )
    if lifecycle_error is not None and not isinstance(lifecycle_error, KeyboardInterrupt):
        raise RuntimeError("HPSv2 worker lifecycle failed") from lifecycle_error
    if failures:
        summary = "\n".join(
            f"{device} {task}:\n{detail}" for device, task, detail in failures
        )
        raise RuntimeError(f"HPSv2 generation worker failed:\n{summary}")


def _run_workers_with_model_postaudit(
    pending: Sequence[Mapping[str, Any]],
    devices: Sequence[int],
    contract: Mapping[str, Any],
    run_config: Mapping[str, Any],
    expected_model_snapshot: Mapping[str, Any],
) -> None:
    """Re-audit the shared source snapshot even when a worker fails."""
    worker_error = None
    try:
        _run_workers(pending, devices, contract, run_config)
    except BaseException as exc:
        worker_error = exc
    model_error = None
    try:
        observed = model_snapshot.validate_model_snapshot(ROOT)
        if observed != dict(expected_model_snapshot):
            raise RuntimeError("SDXL source snapshot changed during generation")
    except BaseException as exc:
        model_error = exc
    if worker_error is not None and model_error is not None:
        raise RuntimeError(
            "generation workers and the post-generation model audit both failed: "
            f"{type(model_error).__name__}: {model_error}"
        ) from worker_error
    if worker_error is not None:
        raise worker_error.with_traceback(worker_error.__traceback__)
    if model_error is not None:
        raise model_error.with_traceback(model_error.__traceback__)


def _run_generation_attempt_batch(
    *,
    output_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    devices: Sequence[int],
    contract: Mapping[str, Any],
    run_config: Mapping[str, Any],
    generation_environment: Mapping[str, Any],
    model_snapshot_record: Mapping[str, Any],
) -> dict[str, Any]:
    attempt, attempt_sha256, staging_dir = _begin_generation_attempt(
        output_dir, tasks, run_config
    )
    attempt_contract = dict(contract)
    attempt_contract["output_dir"] = staging_dir
    attempt_contract["attempt_id"] = attempt["attempt_id"]
    attempt_contract["attempt_sha256"] = attempt_sha256
    try:
        _require_generation_devices_process_free(generation_environment)
        _run_workers_with_model_postaudit(
            tasks,
            devices,
            attempt_contract,
            run_config,
            model_snapshot_record,
        )
        _require_generation_devices_process_free(generation_environment)
        return _promote_and_accept_generation_attempt(
            output_dir,
            staging_dir,
            tasks,
            run_config,
            attempt,
        )
    except BaseException as generation_error:
        try:
            _poison_generation_attempt(output_dir, attempt, generation_error)
        except BaseException as poison_error:
            raise RuntimeError(
                "generation failed and its attempt could not be poisoned: "
                f"{type(poison_error).__name__}: {poison_error}"
            ) from generation_error
        raise


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    contract = load_contract()
    tasks = build_tasks(contract, args.devices)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "prompt_count": EXPECTED_PROMPT_COUNT,
                    "setting_count": EXPECTED_SETTING_COUNT,
                    "task_count": EXPECTED_TASK_COUNT,
                    "devices": list(args.devices),
                    "prompt_csv_sha256": contract["prompt_csv_sha256"],
                    "prompt_manifest_sha256": contract["prompt_manifest_sha256"],
                    "config_sha256": contract["config_sha256"],
                    "task_matrix_sha256": canonical_sha256(tasks),
                    "generation_environment_schema": GENERATION_ENVIRONMENT_SCHEMA,
                    "model_snapshot_manifest_sha256": (
                        model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256
                    ),
                },
                sort_keys=True,
            )
        )
        return 0

    if args.audit_only:
        output_dir = contract["output_dir"]
        if not output_dir.is_dir():
            raise ValueError(f"HPSv2 run directory is missing: {output_dir}")
        lock_path = output_dir / ".generate.lock"
        with lock_path.open("a+b") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "generation or scoring owns the HPSv2 run lock"
                ) from exc
            rows = validate_complete_run(output_dir)
        print(
            json.dumps(
                {
                    "status": "complete_and_valid",
                    "record_count": len(rows),
                    "manifest_sha256": sha256_file(output_dir / FINAL_MANIFEST_NAME),
                    "run_dir": str(output_dir),
                },
                sort_keys=True,
            )
        )
        return 0

    git_commit = validate_git_launch_contract()
    import torch

    _configure_worker_determinism(torch)
    generation_environment = _generation_environment(args.devices, torch)
    model_snapshot_record = model_snapshot.validate_model_snapshot(ROOT)
    _validate_model_snapshot_record(model_snapshot_record)
    run_config = _run_config(
        contract,
        args.devices,
        tasks,
        git_commit,
        generation_environment=generation_environment,
        model_snapshot_record=model_snapshot_record,
    )
    output_dir = contract["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".generate.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another HPSv2 generation owner holds the run lock") from exc
        _prepare_output(contract, run_config)
        accepted_index, running_attempts = _load_generation_attempt_index(
            output_dir, run_config, tasks
        )
        for stale_attempt, _ in running_attempts:
            _poison_generation_attempt(
                output_dir,
                stale_attempt,
                RuntimeError("previous process ended before a terminal attempt state"),
            )
        accepted_index, running_attempts = _load_generation_attempt_index(
            output_dir, run_config, tasks
        )
        if running_attempts:
            raise RuntimeError("generation attempt reconciliation left a running attempt")
        moved = _quarantine_unaccepted_artifacts(
            output_dir, tasks, accepted_index
        )
        complete, pending = _consolidate(
            contract, tasks, run_config, accepted_index
        )
        print(
            f"validated {len(complete)}/{EXPECTED_TASK_COUNT}; "
            f"pending {len(pending)}; quarantined {moved}",
            flush=True,
        )
        batches = _generation_attempt_batches(pending, args.devices)
        for batch_index, batch in enumerate(batches, 1):
            print(
                f"generation attempt batch {batch_index}/{len(batches)}: "
                f"{len(batch)} tasks",
                flush=True,
            )
            _run_generation_attempt_batch(
                output_dir=output_dir,
                tasks=batch,
                devices=args.devices,
                contract=contract,
                run_config=run_config,
                generation_environment=generation_environment,
                model_snapshot_record=model_snapshot_record,
            )
            accepted_index, running_attempts = _load_generation_attempt_index(
                output_dir, run_config, tasks
            )
            if running_attempts:
                raise RuntimeError("accepted generation left an unterminated attempt")
            complete, pending = _consolidate(
                contract, tasks, run_config, accepted_index
            )
            print(
                f"accepted batch {batch_index}/{len(batches)}; "
                f"{len(complete)}/{EXPECTED_TASK_COUNT} tasks durable",
                flush=True,
            )
        if pending:
            raise RuntimeError(
                f"HPSv2 matrix remains incomplete: {len(pending)} tasks pending"
            )
        print(
            f"complete: {len(complete)} records -> {output_dir / FINAL_MANIFEST_NAME}",
            flush=True,
        )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
