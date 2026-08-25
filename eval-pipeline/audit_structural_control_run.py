"""Result-blind provenance audit for the registered structural-control matrix."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml
from PIL import Image

import audit_latent_renderer_run as base_audit
import generate
from scorer_provenance import validate_hardened_score_rows
from s7_provenance import (
    json_sha256,
    validate_design_rows,
    validate_run_contract,
    validate_scores_against_manifest,
    validate_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRATION = ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE
GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def _load_json(path: Path, *, label: str) -> dict:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_yaml(path: Path, *, label: str) -> dict:
    try:
        with path.open() as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{label} is not readable YAML") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict]:
    try:
        rows = base_audit.load_jsonl(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSONL") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} rows must be JSON objects")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
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


@contextmanager
def audit_lock(run_dir: str | os.PathLike[str]):
    """Exclude generation, scoring, and concurrent structural audits."""
    run_root = Path(run_dir).resolve()
    generation_handle = open(run_root / ".generate.lock", "a+")
    audit_handle = open(run_root / ".structural_control_audit.lock", "a+")
    try:
        try:
            fcntl.flock(generation_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(audit_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"generation, scoring, or structural audit is active for {run_root}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(audit_handle.fileno(), fcntl.LOCK_UN)
            fcntl.flock(generation_handle.fileno(), fcntl.LOCK_UN)
        finally:
            audit_handle.close()
            generation_handle.close()


def _require_finite_number(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite{' and positive' if positive else ''}")
    return number


def _git_bytes(commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{commit}:{relative_path}"],
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"cannot read {relative_path!r} from generation commit {commit}"
        ) from exc


def _validate_generation_commit(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    source_path: Path,
    registration_path: Path,
) -> str:
    generation_commit = str(config.get("git_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", generation_commit):
        raise ValueError("run config git_commit must be full lowercase 40-hex")
    authorization = source["authorization"]
    reviewed_commit = str(authorization["reviewed_commit"])
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            reviewed_commit,
            generation_commit,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ancestry.returncode != 0:
        raise ValueError("reviewed implementation commit is not an ancestor of generation")

    source_relative = os.path.relpath(source_path.resolve(), ROOT.resolve())
    if source_relative.startswith(".." + os.sep) or source_relative == "..":
        raise ValueError("audited executable actions must remain inside the repository")
    committed_actions = _git_bytes(generation_commit, source_relative)
    current_actions = source_path.read_bytes()
    if committed_actions != current_actions:
        raise ValueError("executable action bytes differ from the generation commit")

    registration_relative = os.path.relpath(registration_path.resolve(), ROOT.resolve())
    if registration_relative.startswith(".." + os.sep) or registration_relative == "..":
        raise ValueError("structural registration must remain inside the repository")
    committed_registration = _git_bytes(generation_commit, registration_relative)
    if committed_registration != registration_path.read_bytes():
        raise ValueError("structural registration bytes differ from generation commit")

    implementation = source.get("implementation_source")
    if not isinstance(implementation, dict):
        raise ValueError("structural executable lacks implementation_source")
    files = implementation.get("files")
    if not isinstance(files, dict):
        raise ValueError("structural executable lacks reviewed implementation files")
    for relative_path, expected_hash in files.items():
        observed_hash = hashlib.sha256(
            _git_bytes(generation_commit, str(relative_path))
        ).hexdigest()
        if observed_hash != expected_hash:
            raise ValueError(
                f"generation commit implementation hash differs for {relative_path}"
            )
    return generation_commit


def _validate_analysis_implementation(
    source: Mapping[str, Any],
    registration: Mapping[str, Any],
    generation_commit: str,
) -> dict:
    registered = source.get("analysis_implementation")
    if registered != registration.get("analysis_implementation"):
        raise ValueError("executable analysis implementation differs from registration")
    if not isinstance(registered, dict) or set(registered) != {"schema", "files"}:
        raise ValueError("structural analysis implementation is invalid")
    if registered.get("schema") != generate.STRUCTURAL_CONTROL_ANALYSIS_SCHEMA:
        raise ValueError("structural analysis implementation schema differs")
    files = registered.get("files")
    if not isinstance(files, dict) or set(files) != set(
        generate.STRUCTURAL_CONTROL_ANALYSIS_PATHS
    ):
        raise ValueError("structural analysis implementation paths differ")
    for relative_path, expected_hash in files.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)):
            raise ValueError("structural analysis implementation hash is invalid")
        current_path = (ROOT / str(relative_path)).resolve()
        try:
            current_path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("structural analysis source is outside the repository") from exc
        if not current_path.is_file():
            raise ValueError(f"structural analysis source is missing: {relative_path}")
        current_hash = hashlib.sha256(current_path.read_bytes()).hexdigest()
        committed_hash = hashlib.sha256(
            _git_bytes(generation_commit, str(relative_path))
        ).hexdigest()
        if current_hash != expected_hash or committed_hash != expected_hash:
            raise ValueError(
                f"structural analysis source hash differs for {relative_path}"
            )
    return dict(registered)


def _validate_environment_contract(
    config: Mapping[str, Any], source: Mapping[str, Any], generation_commit: str
) -> None:
    environment = source.get("environment_lock")
    if not isinstance(environment, dict) or set(environment) != {"path", "sha256"}:
        raise ValueError("source generation environment lock is invalid")
    relative_path = str(environment.get("path", ""))
    lock_path = (ROOT / relative_path).resolve()
    try:
        lock_path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("generation environment lock is outside the repository") from exc
    if not lock_path.is_file():
        raise ValueError("generation environment lock file is missing")
    current_bytes = lock_path.read_bytes()
    expected_hash = str(environment.get("sha256", ""))
    if hashlib.sha256(current_bytes).hexdigest() != expected_hash or hashlib.sha256(
        _git_bytes(generation_commit, relative_path)
    ).hexdigest() != expected_hash:
        raise ValueError("generation environment lock bytes differ")
    lock = _load_yaml(lock_path, label="generation environment lock")
    runtime = config.get("runtime_provenance")
    if not isinstance(runtime, dict):
        raise ValueError("run config lacks runtime provenance")
    expected_runtime = {
        "python_version": str(lock["runtime"]["python"]),
        "torch_version": str(lock["packages"]["torch"]),
        "diffusers_version": str(lock["packages"]["diffusers"]),
        "cuda_runtime_version": str(lock["runtime"]["cuda"]),
        "cudnn_version": lock["runtime"]["cudnn"],
        "generation_environment_lock_id": lock["lock_id"],
        "generation_environment_lock_path": relative_path,
        "generation_environment_lock_sha256": expected_hash,
        "generation_environment_packages": lock["packages"],
        "generation_environment_platform": lock["platform"],
        "generation_environment_hardware": lock["reference_hardware"],
        "generation_environment_determinism": lock["determinism"],
    }
    if runtime != expected_runtime:
        raise ValueError("run runtime provenance differs from the full environment lock")


def _validate_config_contract(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    normalized_actions: Sequence[Mapping[str, Any]],
    source_path: Path,
    registration_path: Path,
) -> None:
    if config.get("structural_control_registered") is not True:
        raise ValueError("run is not marked as a registered structural-control matrix")
    if config.get("native_renderer_registered") is not False:
        raise ValueError("structural-control run must not be marked as a native renderer run")
    if config.get("action_schema") != generate.STRUCTURAL_CONTROL_SCHEMA:
        raise ValueError("run action schema differs from structural-control v1")
    if config.get("structural_control_registration_schema") != source.get(
        "registration_schema"
    ):
        raise ValueError("run structural registration schema differs")
    expected_bindings = {
        "structural_control_authorization": source.get("authorization"),
        "structural_control_source_template": source.get("authorization", {}).get(
            "source_template"
        ),
        "structural_control_source_template_sha256": source.get(
            "authorization", {}
        ).get("source_template_sha256"),
        "structural_control_implementation_source": source.get(
            "implementation_source"
        ),
        "structural_control_analysis_implementation": source.get(
            "analysis_implementation"
        ),
        "structural_control_executable_actions_sha256": base_audit.sha256_file(
            source_path
        ),
        "registered_sampling": source.get("sampling"),
        "scheduler_runtime": source.get("scheduler_runtime"),
        "scoring": source.get("scoring"),
        "structural_control_failure_policy": source.get("failure_policy"),
    }
    for key, expected in expected_bindings.items():
        if config.get(key) != expected:
            raise ValueError(f"run config field {key!r} differs from executable actions")
    if source.get("failure_policy") != "shared_abort_after_first_task_error":
        raise ValueError("structural-control failure policy differs")
    if config.get("actions") != list(normalized_actions):
        raise ValueError("run normalized actions differ from the executable design")
    if config.get("scorer_provenance_binding_required") is not True:
        raise ValueError("structural-control scorer provenance binding is disabled")

    sampling = source.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("structural executable lacks sampling registration")
    resolution = int(sampling["resolution"])
    stage2_enabled = bool(sampling["stage2"])
    low_vram = bool(sampling["low_vram"])
    expected_run_fields = {
        "model_name": sampling["model"],
        "model_revision": sampling["model_revision"],
        "resolution": resolution,
        "num_inference_steps": int(sampling["num_inference_steps"]),
        "guidance_scale": float(sampling["default_cfg_scale"]),
        "guidance_rescale": float(sampling["guidance_rescale"]),
        "negative_prompt": sampling["negative_prompt"],
        "power_calibrate": int(sampling["power_calibrate"]),
        "low_vram": low_vram,
        "stage2_enabled": stage2_enabled,
        "stage_name": (
            f"stage2_{resolution}" if stage2_enabled else f"stage1_{resolution}"
        ),
        "models_to_cpu": bool(low_vram or stage2_enabled),
        "multi_encoder": stage2_enabled,
        "multi_decoder": bool(stage2_enabled and resolution > 2048),
        "num_resample_timesteps": 50,
        "init_rates": [0.9, 0.8]
        if stage2_enabled and resolution >= 4096
        else [0.8],
        "stage2_noise_source": "task_generator" if stage2_enabled else None,
        "frequency_band_cutoffs": source.get("frequency_band_cutoffs"),
        "trajectory_registered": False,
        "scheduler_baseline_registered": False,
        "cfg_baseline_registered": False,
    }
    for key, expected in expected_run_fields.items():
        if config.get(key) != expected:
            raise ValueError(f"run config field {key!r} differs from registration")

    registration_hash = base_audit.sha256_file(registration_path)
    registration_source = source.get("registration_source")
    if registration_source != {
        "path": generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
        "sha256": registration_hash,
    }:
        raise ValueError("executable registration_source differs from frozen bytes")
    if source.get("authorization", {}).get("source_template_sha256") != registration_hash:
        raise ValueError("authorization source-template hash differs from frozen bytes")

    runtime = config.get("runtime_provenance")
    if not isinstance(runtime, dict):
        raise ValueError("run config lacks runtime provenance")
    environment = source.get("environment_lock")
    if not isinstance(environment, dict):
        raise ValueError("source lacks generation environment lock")
    if runtime.get("generation_environment_lock_path") != environment.get("path"):
        raise ValueError("runtime environment-lock path differs from registration")
    if runtime.get("generation_environment_lock_sha256") != environment.get("sha256"):
        raise ValueError("runtime environment-lock hash differs from registration")
    for key in ("generation_environment_hardware", "generation_environment_determinism"):
        if not isinstance(runtime.get(key), dict):
            raise ValueError(f"runtime provenance lacks {key}")
    if runtime.get("diffusers_version") != "0.32.1":
        raise ValueError("structural-control generation did not use diffusers 0.32.1")


def _validate_scheduler_ledger(
    record: Mapping[str, Any], runtime: Mapping[str, Any], *, record_id: str
) -> None:
    expected_steps = int(runtime["num_inference_steps"])
    if record.get("scheduler_name") != "EulerDiscreteScheduler" or record.get(
        "base_scheduler_name"
    ) != "EulerDiscreteScheduler":
        raise ValueError(f"{record_id}: scheduler class differs from Euler")
    expected_config_hash = runtime["config_sha256_v2"]
    for payload_key, digest_key in (
        ("scheduler_config", "scheduler_config_sha256_v2"),
        ("active_scheduler_config", "active_scheduler_config_sha256_v2"),
    ):
        payload = record.get(payload_key)
        if not isinstance(payload, dict):
            raise ValueError(f"{record_id}: {payload_key} payload is missing")
        observed_hash = generate.scheduler_config_payload_sha256(payload)
        if record.get(digest_key) != observed_hash or observed_hash != expected_config_hash:
            raise ValueError(
                f"{record_id}: {digest_key} differs from registered Euler config"
            )
    if record.get("scheduler_config") != record.get("active_scheduler_config"):
        raise ValueError(f"{record_id}: active Euler config differs from base config")
    if record.get("scheduler_schedule_sha256") != runtime["schedule_sha256"]:
        raise ValueError(f"{record_id}: scheduler schedule hash differs")
    timesteps = record.get("scheduler_timesteps")
    sigmas = record.get("scheduler_sigmas")
    if not isinstance(timesteps, list) or len(timesteps) != expected_steps:
        raise ValueError(f"{record_id}: scheduler timestep ledger is incomplete")
    if not isinstance(sigmas, list) or len(sigmas) != expected_steps + 1:
        raise ValueError(f"{record_id}: scheduler sigma ledger is incomplete")
    values = timesteps + sigmas
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"{record_id}: scheduler ledger contains non-finite values")
    if json_sha256({"timesteps": timesteps, "sigmas": sigmas}) != runtime[
        "schedule_sha256"
    ]:
        raise ValueError(f"{record_id}: scheduler ledger bytes do not match its hash")
    exact_values = {
        "scheduler_construction_init_noise_sigma": runtime[
            "construction_init_noise_sigma"
        ],
        "scheduler_effective_init_noise_sigma": runtime[
            "effective_init_noise_sigma"
        ],
        "scheduler_init_noise_sigma": runtime["construction_init_noise_sigma"],
        "scheduler_order": 1,
    }
    for key, expected in exact_values.items():
        if record.get(key) != expected:
            raise ValueError(f"{record_id}: {key} differs from registered Euler runtime")
    if record.get("scheduler_kwargs") != {}:
        raise ValueError(f"{record_id}: Euler scheduler kwargs are not empty")


def _validate_worker_provenance(
    record: Mapping[str, Any], config: Mapping[str, Any], *, record_id: str
) -> None:
    device = str(record.get("device", ""))
    if not re.fullmatch(r"cuda:\d+", device):
        raise ValueError(f"{record_id}: generation device is not an explicit CUDA device")
    devices = config.get("devices")
    if not isinstance(devices, list) or device not in devices:
        raise ValueError(f"{record_id}: generation device is absent from run config")
    provenance = record.get("worker_device_provenance")
    required_fields = {
        "gpu",
        "compute_capability",
        "total_memory_bytes",
        "requested_device",
        "logical_device_index",
        "physical_device_index",
        "gpu_uuid",
        "pci_bus_id",
        "cuda_visible_devices",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_fields:
        raise ValueError(f"{record_id}: worker device provenance fields differ")
    logical_index = int(device.split(":", 1)[1])
    if provenance.get("requested_device") != device or provenance.get(
        "logical_device_index"
    ) != logical_index:
        raise ValueError(f"{record_id}: worker logical device identity differs")
    if (
        isinstance(provenance.get("physical_device_index"), bool)
        or not isinstance(provenance.get("physical_device_index"), int)
        or provenance["physical_device_index"] < 0
        or not GPU_UUID_PATTERN.fullmatch(str(provenance.get("gpu_uuid", "")))
        or not str(provenance.get("pci_bus_id", "")).strip()
        or isinstance(provenance.get("total_memory_bytes"), bool)
        or not isinstance(provenance.get("total_memory_bytes"), int)
        or provenance["total_memory_bytes"] <= 0
    ):
        raise ValueError(f"{record_id}: worker physical GPU identity is invalid")
    expected_hardware = config["runtime_provenance"][
        "generation_environment_hardware"
    ]
    for key in ("gpu", "compute_capability"):
        if provenance.get(key) != expected_hardware.get(key):
            raise ValueError(f"{record_id}: worker {key} differs from environment lock")
    expected_determinism = config["runtime_provenance"][
        "generation_environment_determinism"
    ]
    if record.get("worker_determinism_provenance") != expected_determinism:
        raise ValueError(f"{record_id}: worker determinism differs from environment lock")
    sampling = config.get("registered_sampling")
    if not isinstance(sampling, dict):
        raise ValueError(f"{record_id}: registered sampling is missing")
    expected_model_load = {
        "torch_dtype": sampling.get("torch_dtype"),
        "variant": sampling.get("variant"),
        "local_files_only": sampling.get("local_files_only"),
        "revision": sampling.get("model_revision"),
    }
    if record.get("model_load_provenance") != expected_model_load:
        raise ValueError(f"{record_id}: model-load provenance differs")
    for key, expected in config["runtime_provenance"].items():
        if record.get(key) != expected:
            raise ValueError(f"{record_id}: runtime provenance field {key!r} differs")


def _validate_action_provenance(
    record: Mapping[str, Any], action: Mapping[str, Any], *, record_id: str
) -> None:
    action_type = action["type"]
    freeu_fields = (
        "freeu_schedule",
        "freeu_operator_runtime",
        "freeu_implementation",
        "freeu_source_commit",
        "freeu_implementation_diffusers_version",
    )
    attention_fields = (
        "attention_baseline_implementation",
        "attention_baseline_source_commit",
        "attention_baseline_paper_id",
        "attention_baseline_topology",
    )
    if action_type == "freeu":
        expected = {
            "freeu_schedule": action.get("freeu_schedule"),
            "freeu_operator_runtime": {
                "implementation": action.get("implementation"),
                "operator_calls_total": action.get(
                    "expected_operator_calls_per_step"
                )
                * 50,
                "resolution_idx_call_counts": {
                    str(index): count * 50
                    for index, count in enumerate(
                        action.get(
                            "expected_resolution_idx_call_counts_per_step"
                        )
                    )
                },
                "hidden_channel_call_counts": {
                    str(channel): count * 50
                    for channel, count in action.get(
                        "expected_hidden_channel_call_counts_per_step"
                    ).items()
                },
                "resolution_channel_call_counts": {
                    str(key): count * 50
                    for key, count in action.get(
                        "expected_resolution_channel_call_counts_per_step"
                    ).items()
                },
                "operator_effect_call_counts": {
                    str(effect): count * 50
                    for effect, count in action.get(
                        "expected_operator_effect_call_counts_per_step"
                    ).items()
                },
            },
            "freeu_implementation": action.get("implementation"),
            "freeu_source_commit": action.get("source_commit"),
            "freeu_implementation_diffusers_version": action.get(
                "implementation_diffusers_version"
            ),
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"{record_id}: {key} differs from registered FreeU")
        if record.get("freeu_preserve_moments") is not False:
            raise ValueError(f"{record_id}: unexpected FreeU moment preservation")
        expected_runtime = []
        schedule = generate.freeu_runtime(dict(action))
        for step_index in range(50):
            expected_runtime.append(
                {
                    "step_index": step_index,
                    "parameters": list(
                        schedule.at(step_index / 49.0).as_tuple()
                    ),
                }
            )
        if record.get("freeu_runtime") != expected_runtime:
            raise ValueError(f"{record_id}: FreeU activation ledger differs")
        if any(record.get(key) is not None for key in attention_fields):
            raise ValueError(f"{record_id}: FreeU record has stale attention provenance")
        return
    if any(record.get(key) is not None for key in freeu_fields) or record.get(
        "freeu_preserve_moments"
    ) is not False:
        raise ValueError(f"{record_id}: non-FreeU record has stale FreeU provenance")
    if record.get("freeu_runtime") != []:
        raise ValueError(f"{record_id}: non-FreeU record has stale activation ledger")

    if action_type == "attention_baseline":
        expected = {
            "attention_baseline_implementation": action.get("implementation"),
            "attention_baseline_source_commit": action.get("source_commit"),
            "attention_baseline_paper_id": action.get("paper_id"),
            "attention_baseline_topology": {
                "group_counts": action.get("expected_processor_group_counts"),
                "processor_count": action.get("expected_processor_count"),
                "processor_names_sha256": action.get(
                    "expected_processor_names_sha256"
                ),
                "processors_called": action.get("expected_processor_count"),
                "processor_calls_total": action.get("expected_processor_count") * 50,
                "processor_call_count_min": 50,
                "processor_call_count_max": 50,
            },
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"{record_id}: {key} differs from registered baseline")
    elif any(record.get(key) is not None for key in attention_fields):
        raise ValueError(f"{record_id}: non-attention record has stale baseline provenance")


def audit_structural_records(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    normalized_actions: Sequence[Mapping[str, Any]],
    *,
    expected_tasks: int | None = None,
) -> dict:
    """Audit structural sidecars without reading any quality metric values."""
    if expected_tasks is None:
        expected_tasks = int(source.get("design", {}).get("expected_task_count", -1))
    if len(manifest) != expected_tasks:
        raise ValueError(
            f"structural manifest requires {expected_tasks} records, got {len(manifest)}"
        )
    actions = {str(action["id"]): action for action in normalized_actions}
    runtime = source.get("scheduler_runtime")
    sampling = source.get("sampling")
    if not isinstance(runtime, dict) or not isinstance(sampling, dict):
        raise ValueError("structural source lacks scheduler/sampling registration")
    per_device_identity: dict[str, tuple[Any, ...]] = {}
    for record in manifest:
        record_id = str(record.get("id", ""))
        action_id = str(record.get("action_id", ""))
        if action_id not in actions:
            raise ValueError(f"{record_id}: action is absent from structural registration")
        action = actions[action_id]
        if record.get("registered_sampling") != sampling or record.get(
            "scheduler_runtime"
        ) != runtime:
            raise ValueError(f"{record_id}: sidecar sampling registration differs")
        if record.get("num_inference_steps") != 50:
            raise ValueError(f"{record_id}: structural run requires 50 denoising steps")
        if record.get("unet_calls_per_step") != [1] * 50 or record.get(
            "extra_unet_calls"
        ) != 0:
            raise ValueError(f"{record_id}: structural run violates matched 50x1 U-Net calls")
        expected_cfg = float(action["cfg_scale"])
        if record.get("guidance_scale") != expected_cfg or record.get(
            "guidance_rescale"
        ) != 0.0:
            raise ValueError(f"{record_id}: CFG runtime differs from registered action")
        expected_run_fields = {
            "height": config["resolution"],
            "width": config["resolution"],
            "power_calibrate": config["power_calibrate"],
            "frequency_band_cutoffs": config["frequency_band_cutoffs"],
            "stage": config["stage_name"],
            "stage2_enabled": config["stage2_enabled"],
            "models_to_cpu": config["models_to_cpu"],
            "multi_encoder": config["multi_encoder"],
            "multi_decoder": config["multi_decoder"],
            "num_resample_timesteps": config["num_resample_timesteps"],
            "init_rates": config["init_rates"],
            "stage2_noise_source": config["stage2_noise_source"],
            "model_name": config["model_name"],
            "model_revision": config["model_revision"],
        }
        if any(
            record.get(key) != expected for key, expected in expected_run_fields.items()
        ):
            raise ValueError(f"{record_id}: sidecar run settings differ")
        _require_finite_number(
            record.get("inference_seconds"),
            label=f"{record_id}: inference_seconds",
            positive=True,
        )
        peak_memory = record.get("peak_gpu_memory_bytes")
        if isinstance(peak_memory, bool) or not isinstance(peak_memory, int) or peak_memory <= 0:
            raise ValueError(f"{record_id}: peak GPU memory provenance is invalid")
        _validate_scheduler_ledger(record, runtime, record_id=record_id)
        _validate_worker_provenance(record, config, record_id=record_id)
        generate.validate_structural_intervention_runtime(
            dict(record), dict(action), num_inference_steps=50
        )
        _validate_action_provenance(record, action, record_id=record_id)

        worker = record["worker_device_provenance"]
        identity = (
            worker["logical_device_index"],
            worker["physical_device_index"],
            worker["gpu_uuid"],
            worker["pci_bus_id"],
            worker["cuda_visible_devices"],
        )
        prior = per_device_identity.setdefault(str(record["device"]), identity)
        if prior != identity:
            raise ValueError(f"{record_id}: device identity differs across sidecars")
    return {
        "structural_control_contract_passed": True,
        "expected_task_count": expected_tasks,
        "matched_unet_calls": "50x1",
        "scheduler": "EulerDiscreteScheduler",
        "scheduler_schedule_sha256": runtime["schedule_sha256"],
        "device_identities": {
            device: {
                "logical_device_index": identity[0],
                "physical_device_index": identity[1],
                "gpu_uuid": identity[2],
                "pci_bus_id": identity[3],
                "cuda_visible_devices": identity[4],
            }
            for device, identity in sorted(per_device_identity.items())
        },
        "quality_results_inspected": False,
        "runtime_activation_ledgers_verified": True,
    }


def audit_action_png_pair_counts(
    manifest: Sequence[Mapping[str, Any]],
    action_ids: Sequence[str],
    *,
    expected_blocks: int,
) -> list[dict]:
    """Report isolated equality and reject an action pair collapsed everywhere."""
    by_block: dict[tuple[int, int], dict[str, str]] = {}
    expected_actions = {str(action_id) for action_id in action_ids}
    for record in manifest:
        block = (int(record["prompt_index"]), int(record["seed"]))
        action_id = str(record["action_id"])
        image_hash = str(record.get("image_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", image_hash):
            raise ValueError(f"{record.get('id')}: image SHA-256 is invalid")
        block_hashes = by_block.setdefault(block, {})
        if action_id in block_hashes:
            raise ValueError(f"duplicate action {action_id!r} in block {block}")
        block_hashes[action_id] = image_hash
    if len(by_block) != expected_blocks:
        raise ValueError(
            f"structural duplicate audit requires {expected_blocks} blocks, "
            f"got {len(by_block)}"
        )
    for block, hashes in by_block.items():
        if set(hashes) != expected_actions:
            raise ValueError(f"structural block {block} has an incomplete action set")

    pair_counts = []
    for left, right in combinations(action_ids, 2):
        matching_blocks = sum(
            hashes[str(left)] == hashes[str(right)] for hashes in by_block.values()
        )
        if matching_blocks == expected_blocks:
            raise ValueError(
                f"structural actions {left!r} and {right!r} produced identical "
                f"PNGs in all {expected_blocks} blocks"
            )
        if matching_blocks:
            pair_counts.append(
                {
                    "actions": [str(left), str(right)],
                    "matching_blocks": matching_blocks,
                    "total_blocks": expected_blocks,
                }
            )
    return pair_counts


def expected_execution_ranks(
    prompt_index: int, seed: int, action_ids: Sequence[str]
) -> dict[str, int]:
    """Recompute the frozen action-order-v1 rank mapping for one block."""
    ordered = sorted(
        (str(action_id) for action_id in action_ids),
        key=lambda action_id: hashlib.sha256(
            f"action-order-v1:{prompt_index}:{seed}:{action_id}".encode("utf-8")
        ).digest(),
    )
    return {action_id: rank for rank, action_id in enumerate(ordered)}


def validate_execution_ranks(
    manifest: Sequence[Mapping[str, Any]], action_ids: Sequence[str]
) -> None:
    """Require every sidecar action to carry its deterministic block rank."""
    expected_actions = {str(action_id) for action_id in action_ids}
    per_block_actions: dict[tuple[int, int], set[str]] = {}
    for record in manifest:
        prompt_index = int(record["prompt_index"])
        seed = int(record["seed"])
        action_id = str(record["action_id"])
        rank = record.get("execution_rank")
        expected = expected_execution_ranks(prompt_index, seed, action_ids)
        if (
            action_id not in expected_actions
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank != expected[action_id]
        ):
            raise ValueError(
                f"{record.get('id')}: execution_rank differs from action-order-v1"
            )
        block = (prompt_index, seed)
        block_actions = per_block_actions.setdefault(block, set())
        if action_id in block_actions:
            raise ValueError(f"duplicate action {action_id!r} in block {block}")
        block_actions.add(action_id)
    if any(actions != expected_actions for actions in per_block_actions.values()):
        raise ValueError("structural execution-order block has an incomplete action set")


def validate_manifest_sidecars(
    run_path: Path, manifest: Sequence[Mapping[str, Any]]
) -> None:
    """Require every published manifest row to equal its on-disk sidecar."""
    image_dir = (run_path / "images").resolve()
    for record in manifest:
        record_id = str(record.get("id", ""))
        sidecar_path = (image_dir / f"{record_id}.json").resolve()
        if os.path.commonpath((str(image_dir), str(sidecar_path))) != str(image_dir):
            raise ValueError(f"{record_id}: sidecar path escapes the image directory")
        sidecar = _load_json(sidecar_path, label=f"{record_id} sidecar")
        if sidecar != dict(record):
            raise ValueError(f"{record_id}: disk sidecar differs from manifest row")


def _validate_registered_artifact_bindings(
    run_path: Path,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    normalized_actions: Sequence[Mapping[str, Any]],
) -> str:
    """Apply structural-only score, contract, action, and image bindings."""
    contract_hash = validate_run_contract(config)
    scores = _load_jsonl(run_path / "scores.jsonl", label="scores")
    scoring = source.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("structural executable lacks scoring provenance")
    scorer_hash = validate_hardened_score_rows(
        scores,
        required_schema=str(scoring.get("required_schema", "")),
        expected_sha256=str(scoring.get("registered_scorer_provenance_sha256", "")),
    )
    for score in scores:
        if score.get("run_contract_sha256") != contract_hash:
            raise ValueError(
                f"{score.get('id')}: score is not bound to the structural run contract"
            )
    validate_scores_against_manifest(manifest, scores)

    actions = {str(action["id"]): dict(action) for action in normalized_actions}
    for record in manifest:
        action_id = str(record.get("action_id", ""))
        action = actions.get(action_id)
        if action is None:
            raise ValueError(f"{record.get('id')}: action is not registered")
        expected_task = {
            "id": str(record.get("id", "")),
            "prompt_index": int(record.get("prompt_index", -1)),
            "prompt": str(record.get("prompt", "")),
            "seed": int(record.get("seed", -1)),
            "action_id": action_id,
            "action_type": action.get("type"),
            "action": action,
        }
        validate_sidecar(
            record,
            run_path,
            expected_task=expected_task,
            expected_contract_sha256=contract_hash,
        )
    return scorer_hash


def audit_engineering_smoke(
    run_dir: str | os.PathLike[str],
    prompts_path: str | os.PathLike[str],
    source_actions_path: str | os.PathLike[str],
    *,
    registration_actions_path: str | os.PathLike[str] = DEFAULT_REGISTRATION,
) -> dict:
    """Audit the exact 88-task, no-scoring structural engineering profile."""
    run_path = Path(run_dir).resolve()
    source_path = Path(source_actions_path).resolve()
    registration_path = Path(registration_actions_path).resolve()
    if (run_path / "scores.jsonl").exists():
        raise ValueError("engineering smoke must not contain quality scores")
    input_paths = {
        "config_sha256": run_path / "config.json",
        "manifest_sha256": run_path / "manifest.jsonl",
        "prompts_sha256": Path(prompts_path).resolve(),
        "source_actions_sha256": source_path,
        "source_template_sha256": registration_path,
    }
    for label, path in input_paths.items():
        if not path.is_file():
            raise ValueError(f"engineering smoke input for {label} is missing: {path}")
    initial_hashes = {
        label: base_audit.sha256_file(path) for label, path in input_paths.items()
    }
    source = _load_yaml(source_path, label="structural executable actions")
    registration = _load_yaml(registration_path, label="frozen structural registration")
    if registration.get("schema") != "structural_control_registration_v1":
        raise ValueError("frozen structural registration schema differs")
    generate.validate_structural_control_authorization(
        str(source_path), require_clean=False, verify_current_source=False
    )
    normalized_actions, band_cutoffs = generate.load_actions(str(source_path), 50)
    if band_cutoffs != [0.08, 0.25]:
        raise ValueError("normalized structural frequency cutoffs differ")
    smoke_seeds = generate.STRUCTURAL_CONTROL_SPLIT_SEEDS["engineering_smoke"]
    generate.validate_structural_control_design(
        str(source_path),
        prompts_path=str(Path(prompts_path).resolve()),
        actions=normalized_actions,
        seeds=smoke_seeds,
        model_name=str(source["sampling"]["model"]),
        resolution=1024,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=generate.DEFAULT_NEG,
        power_calibrate=0,
        stage2_enabled=False,
        split_role="engineering_smoke",
    )
    config = _load_json(run_path / "config.json", label="run config")
    if config.get("split_role") != "engineering_smoke":
        raise ValueError("run config is not the engineering smoke profile")
    if config.get("seeds") != smoke_seeds:
        raise ValueError("engineering smoke seeds differ")
    evidence_scope = {
        key: config.get(key)
        for key in generate.STRUCTURAL_CONTROL_EVIDENCE_SCOPE_KEYS
    }
    if evidence_scope != generate.STRUCTURAL_CONTROL_SMOKE_EVIDENCE_SCOPE:
        raise ValueError("engineering smoke evidence scope differs")
    _validate_config_contract(
        config, source, normalized_actions, source_path, registration_path
    )
    generation_commit = _validate_generation_commit(
        config, source, source_path, registration_path
    )
    analysis_implementation = _validate_analysis_implementation(
        source, registration, generation_commit
    )
    _validate_environment_contract(config, source, generation_commit)
    contract_hash = validate_run_contract(config)

    prompts = pd.read_csv(prompts_path)
    prompt_indices = [int(value) for value in prompts["index"]]
    prompt_text = {
        int(row["index"]): str(row["TEXT"]) for _, row in prompts.iterrows()
    }
    manifest = _load_jsonl(run_path / "manifest.jsonl", label="manifest")
    action_ids = [str(action["id"]) for action in normalized_actions]
    validate_manifest_sidecars(run_path, manifest)
    validate_design_rows(
        manifest,
        expected_action_ids=action_ids,
        expected_seeds=smoke_seeds,
        expected_prompt_indices=prompt_indices,
    )
    validate_execution_ranks(manifest, action_ids)
    expected_stems = {str(record["id"]) for record in manifest}
    image_dir = run_path / "images"
    observed_pngs = {path.stem for path in image_dir.glob("*.png")}
    observed_sidecars = {path.stem for path in image_dir.glob("*.json")}
    if observed_pngs != expected_stems or observed_sidecars != expected_stems:
        raise ValueError("engineering smoke image/sidecar file set differs from its grid")
    actions = {str(action["id"]): dict(action) for action in normalized_actions}
    for record in manifest:
        action_id = str(record["action_id"])
        prompt_index = int(record["prompt_index"])
        expected_task = {
            "id": f"p{prompt_index}_seed{int(record['seed'])}_a{action_id}",
            "prompt_index": prompt_index,
            "prompt": prompt_text[prompt_index],
            "seed": int(record["seed"]),
            "action_id": action_id,
            "action_type": actions[action_id]["type"],
            "action": actions[action_id],
        }
        validate_sidecar(
            record,
            run_path,
            expected_task=expected_task,
            expected_contract_sha256=contract_hash,
        )
        generate.validate_structural_control_sidecar(
            dict(record), config, expected_task=expected_task
        )
        image_path = (run_path / str(record["image_path"])).resolve()
        with Image.open(image_path) as image:
            if image.size != (1024, 1024) or image.mode != "RGB":
                raise ValueError(f"{record['id']}: smoke PNG is not 1024px RGB")
            image.verify()
    structural_report = audit_structural_records(
        config,
        source,
        manifest,
        normalized_actions,
        expected_tasks=88,
    )
    duplicate_pairs = audit_action_png_pair_counts(
        manifest, action_ids, expected_blocks=11
    )
    if duplicate_pairs:
        raise ValueError("engineering smoke requires all eight PNGs to differ per block")
    final_hashes = {
        label: base_audit.sha256_file(path) for label, path in input_paths.items()
    }
    if final_hashes != initial_hashes:
        raise ValueError("engineering smoke inputs changed during verification")
    return {
        **structural_report,
        "passed": True,
        "audit_schema": "scheduler_native_structural_control_smoke_audit_v1",
        "split_role": "engineering_smoke",
        "records": 88,
        "prompts": 11,
        "seeds": smoke_seeds,
        "actions": action_ids,
        "blocks": 11,
        "generation_commit": generation_commit,
        **evidence_scope,
        "quality_scoring_performed": False,
        "all_actions_distinct_within_every_block": True,
        "analysis_implementation": analysis_implementation,
        "analysis_implementation_sha256": json_sha256(analysis_implementation),
        "warnings": [],
        "provenance": {
            **final_hashes,
            "audit_script_sha256": base_audit.sha256_file(__file__),
            "input_snapshot_stable": True,
        },
    }


def audit_run(
    run_dir: str | os.PathLike[str],
    prompts_path: str | os.PathLike[str],
    source_actions_path: str | os.PathLike[str],
    *,
    registration_actions_path: str | os.PathLike[str] = DEFAULT_REGISTRATION,
) -> dict:
    """Run the generic S7 audit and all structural-control-specific checks."""
    run_path = Path(run_dir)
    source_path = Path(source_actions_path).resolve()
    registration_path = Path(registration_actions_path).resolve()
    input_paths = {
        "config_sha256": run_path / "config.json",
        "manifest_sha256": run_path / "manifest.jsonl",
        "scores_sha256": run_path / "scores.jsonl",
        "prompts_sha256": Path(prompts_path).resolve(),
        "source_actions_sha256": source_path,
        "source_template_sha256": registration_path,
    }
    for label, path in input_paths.items():
        if not path.is_file():
            raise ValueError(f"audit input for {label} is missing: {path}")
    initial_hashes = {
        label: base_audit.sha256_file(path) for label, path in input_paths.items()
    }
    source = _load_yaml(source_path, label="structural executable actions")
    registration = _load_yaml(registration_path, label="frozen structural registration")
    if registration.get("schema") != "structural_control_registration_v1":
        raise ValueError("frozen structural registration schema differs")
    generate.validate_structural_control_authorization(
        str(source_path), require_clean=False, verify_current_source=False
    )
    normalized_actions, band_cutoffs = generate.load_actions(str(source_path), 50)
    if band_cutoffs != [0.08, 0.25]:
        raise ValueError("normalized structural frequency cutoffs differ")
    generate.validate_structural_control_design(
        str(source_path),
        prompts_path=str(Path(prompts_path).resolve()),
        actions=normalized_actions,
        seeds=generate.STRUCTURAL_CONTROL_SPLIT_SEEDS["development"],
        model_name=str(source["sampling"]["model"]),
        resolution=1024,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=generate.DEFAULT_NEG,
        power_calibrate=0,
        stage2_enabled=False,
        split_role="development",
    )
    config = _load_json(run_path / "config.json", label="run config")
    _validate_config_contract(
        config, source, normalized_actions, source_path, registration_path
    )
    generation_commit = _validate_generation_commit(
        config, source, source_path, registration_path
    )
    analysis_implementation = _validate_analysis_implementation(
        source, registration, generation_commit
    )
    _validate_environment_contract(config, source, generation_commit)
    base_report = base_audit.audit_run(
        run_path,
        prompts_path,
        source_path,
        split_role="development",
        verify_images=True,
        require_distinct_actions=False,
        registration_actions_path=registration_path,
    )
    manifest = _load_jsonl(run_path / "manifest.jsonl", label="manifest")
    validate_manifest_sidecars(run_path.resolve(), manifest)
    validate_execution_ranks(
        manifest, [str(action["id"]) for action in normalized_actions]
    )
    scorer_provenance_sha256 = _validate_registered_artifact_bindings(
        run_path, config, source, manifest, normalized_actions
    )
    structural_report = audit_structural_records(
        config, source, manifest, normalized_actions
    )
    duplicate_pair_counts = audit_action_png_pair_counts(
        manifest,
        [str(action["id"]) for action in normalized_actions],
        expected_blocks=99,
    )
    final_hashes = {
        label: base_audit.sha256_file(path) for label, path in input_paths.items()
    }
    if final_hashes != initial_hashes:
        raise ValueError("structural audit inputs changed during verification")
    report = dict(base_report)
    report.update(structural_report)
    report.update(
        {
            "audit_schema": "scheduler_native_structural_control_audit_v1",
            "generation_commit": generation_commit,
            "executable_actions_sha256": base_audit.sha256_file(source_path),
            "registration_sha256": base_audit.sha256_file(registration_path),
            "analysis_implementation": analysis_implementation,
            "analysis_implementation_sha256": json_sha256(
                analysis_implementation
            ),
            "duplicate_action_pngs_are_failure": False,
            "isolated_duplicate_action_pngs_are_failure": False,
            "full_action_collapse_is_failure": True,
            "duplicate_action_png_policy": "reject_action_pair_equal_in_all_99_blocks",
            "duplicate_action_png_pair_counts": duplicate_pair_counts,
            "fully_collapsed_action_pairs": [],
            "image_decode_verified": True,
            "scorer_provenance_schema": str(source["scoring"]["required_schema"]),
            "scorer_provenance_sha256": scorer_provenance_sha256,
            "warnings": [],
            "provenance": {
                **final_hashes,
                "audit_script_sha256": base_audit.sha256_file(__file__),
                "input_snapshot_stable": True,
            },
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--registration", default=str(DEFAULT_REGISTRATION))
    parser.add_argument("--output")
    parser.add_argument("--engineering_smoke", action="store_true")
    args = parser.parse_args()
    default_name = "engineering_smoke_audit.json" if args.engineering_smoke else "run_audit.json"
    output_path = Path(args.output).resolve() if args.output else (
        Path(args.run_dir).resolve() / default_name
    )
    with audit_lock(args.run_dir):
        if output_path.exists():
            raise ValueError(f"structural audit is one-shot; output exists: {output_path}")
        audit_function = audit_engineering_smoke if args.engineering_smoke else audit_run
        report = audit_function(
            args.run_dir,
            args.prompts,
            args.actions,
            registration_actions_path=args.registration,
        )
        payload = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _atomic_write(output_path, payload)
    print(json.dumps({"passed": True, "output": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
