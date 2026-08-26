"""One-shot generator for the adaptive-oracle engineering smoke.

This executable is intentionally independent of ``generate.py``.  It accepts
only a separately reviewed authorization, never resumes an output directory,
and exposes no scoring or quality-inspection path.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stderr
import csv
import hashlib
import importlib
import io
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import select
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence
import uuid
import warnings

import yaml


ROOT = Path(__file__).resolve().parents[1]

import adaptive_oracle_contract as contract
import adaptive_oracle_model_snapshot as model_snapshot
import audit_adaptive_oracle_cpu as cpu_audit_contract


AUTHORIZATION_SCHEMA = "adaptive_oracle_engineering_authorized_v1"
AUTHORIZATION_STATUS = "authorized_engineering_only"
REGISTRATION_SCHEMA = "adaptive_oracle_engineering_registration_v1"
REGISTRATION_STATUS = "registration_only_gpu_not_authorized"
ATTEMPT_SCHEMA = "adaptive_oracle_engineering_attempt_v1"
FAILURE_SCHEMA = "adaptive_oracle_engineering_failure_v1"
SUCCESS_SCHEMA = "adaptive_oracle_engineering_success_v1"
RUN_CONFIG_SCHEMA = "adaptive_oracle_engineering_run_config_v2"
RUN_MANIFEST_SCHEMA = "adaptive_oracle_engineering_run_manifest_v1"
RUNTIME_EVIDENCE_SCHEMA = "adaptive_oracle_engineering_runtime_evidence_v1"
MODEL_STAGE_EVIDENCE_SCHEMA = "adaptive_oracle_model_stage_evidence_v2"
ENVIRONMENT_LOCK_SCHEMA = "repldm_generation_environment_lock_v2"
EXCLUSION_INVENTORY_SCHEMA = "adaptive_oracle_exclusion_inventory_v1"
LAUNCH_SCHEMA = "adaptive_oracle_explicit_git_blob_launch_v2"
LAUNCH_TRUST_ROOT = "cpython_explicit_commit_launcher_v2"
LAUNCH_LOADER_ID = "adaptive_oracle_reviewed_git_blob_loader_v2"
COMMIT_TOPOLOGY_SCHEMA = "adaptive_oracle_commit_topology_v1"

PIPELINE_CLASS = "RepLDMSDXLPipeline"
ADAPTIVE_ORACLE_PROVIDER_ID = "adaptive_oracle_local_relational_basis_v1"
OUTPUT_DIR = "outputs/adaptive_oracle/engineering_v1"
REGISTRATION_PATH = (
    "eval-pipeline/configs/adaptive_oracle_engineering_registration_v1.yaml"
)
CPU_AUDIT_PATH = (
    "eval-pipeline/configs/adaptive_oracle_engineering_cpu_audit_v1.json"
)
AUTHORIZATION_PATH = (
    "eval-pipeline/configs/adaptive_oracle_engineering_authorization_v1.yaml"
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
)
ALLOWED_CUDA_DEVICES = ("cuda:1", "cuda:2", "cuda:3", "cuda:4")

SOURCE_PATHS = {
    "prompt_csv": "eval-pipeline/prompts/adaptive_oracle_engineering.csv",
    "prompt_manifest": (
        "eval-pipeline/prompts/adaptive_oracle_prompt_manifest_v1.json"
    ),
    "exclusion_inventory": (
        "eval-pipeline/prompts/adaptive_oracle_exclusion_inventory_v1.json"
    ),
    "prompt_builder": "eval-pipeline/build_adaptive_oracle_prompts.py",
    "contract": "eval-pipeline/adaptive_oracle_contract.py",
    "generator": "eval-pipeline/generate_adaptive_oracle_engineering.py",
    "model_snapshot_validator": (
        "eval-pipeline/adaptive_oracle_model_snapshot.py"
    ),
    "cpu_auditor": "eval-pipeline/audit_adaptive_oracle_cpu.py",
    "engineering_auditor": (
        "eval-pipeline/audit_adaptive_oracle_engineering.py"
    ),
    "adaptive_oracle": "AttentionGuidance/adaptive_oracle.py",
    "attention_guidance": "AttentionGuidance/attention_guidance.py",
    "controller": "AttentionGuidance/controller.py",
    "guidance_types": "AttentionGuidance/types.py",
    "semantic_transport": "AttentionGuidance/semantic_transport.py",
    "freeu": "AttentionGuidance/freeu.py",
    "ancestral_correction": "AttentionGuidance/ancestral_correction.py",
    "local_relational_basis": "AttentionGuidance/local_relational_basis.py",
    "latent_renderer": "AttentionGuidance/latent_renderer.py",
    "pipeline": "InferencePipelines/RepLDM/pipeline_repldm_sdxl.py",
    "attention_guidance_init": "AttentionGuidance/__init__.py",
    "inference_pipelines_init": "InferencePipelines/__init__.py",
    "cfg_batch": "InferencePipelines/cfg_batch.py",
    "environment_validator": (
        "eval-pipeline/adaptive_oracle_generation_environment.py"
    ),
    "environment_lock": (
        "eval-pipeline/configs/generation_environment_adaptive_oracle_v2.yaml"
    ),
    "registration": REGISTRATION_PATH,
    "launcher": "eval-pipeline/launch_adaptive_oracle_tool.py",
}
_LAUNCH_INPUT_PATHS = {
    "registration": REGISTRATION_PATH,
    "cpu_audit": CPU_AUDIT_PATH,
    "prompt_csv": SOURCE_PATHS["prompt_csv"],
    "prompt_manifest": SOURCE_PATHS["prompt_manifest"],
    "exclusion_inventory": SOURCE_PATHS["exclusion_inventory"],
    "environment_lock": SOURCE_PATHS["environment_lock"],
}
_IMPLEMENTATION_SOURCE_NAMES = frozenset(
    {
        "contract",
        "generator",
        "adaptive_oracle",
        "attention_guidance",
        "controller",
        "guidance_types",
        "semantic_transport",
        "freeu",
        "ancestral_correction",
        "local_relational_basis",
        "latent_renderer",
        "pipeline",
        "attention_guidance_init",
        "inference_pipelines_init",
        "cfg_batch",
        "environment_validator",
        "model_snapshot_validator",
        "cpu_auditor",
        "engineering_auditor",
        "prompt_builder",
        "launcher",
    }
)
_VERIFIED_RUNTIME_MODULE_SOURCES = {
    "adaptive_oracle_contract": "contract",
    "adaptive_oracle_model_snapshot": "model_snapshot_validator",
    "audit_adaptive_oracle_cpu": "cpu_auditor",
    "adaptive_oracle_generation_environment": "environment_validator",
    "AttentionGuidance": "attention_guidance_init",
    "AttentionGuidance.attention_guidance": "attention_guidance",
    "AttentionGuidance.controller": "controller",
    "AttentionGuidance.types": "guidance_types",
    "AttentionGuidance.semantic_transport": "semantic_transport",
    "AttentionGuidance.latent_renderer": "latent_renderer",
    "AttentionGuidance.freeu": "freeu",
    "AttentionGuidance.ancestral_correction": "ancestral_correction",
    "AttentionGuidance.local_relational_basis": "local_relational_basis",
    "AttentionGuidance.adaptive_oracle": "adaptive_oracle",
    "InferencePipelines": "inference_pipelines_init",
    "InferencePipelines.cfg_batch": "cfg_batch",
    "InferencePipelines.RepLDM.pipeline_repldm_sdxl": "pipeline",
}
_RUNTIME_MODULE_SOURCE_NAMES = frozenset(
    _VERIFIED_RUNTIME_MODULE_SOURCES.values()
)
_PAIR_FIELDS = frozenset({"path", "sha256"})
_AUTHORIZATION_FIELDS = frozenset(
    {
        "reviewer",
        "reviewed_commit",
        "source_registration",
        "cpu_audit",
        "gpu_generation",
        "scoring",
        "quality_inspection",
        "method_selection",
        "renderer_training",
        "rl",
        "one_shot",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "model_id",
        "revision",
        "cache_dir",
        "local_files_only",
        "torch_dtype",
        "variant",
        "pipeline_class",
        "snapshot_manifest_schema",
        "snapshot_manifest_sha256",
        "snapshot_loaded_file_count",
    }
)
_EVIDENCE_SCOPE = {
    "generation_only": True,
    "scoring_authorized": False,
    "quality_outcomes_present": False,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_CUDA_DEVICE = re.compile(r"cuda:(0|[1-9][0-9]*)")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"authorization YAML contains duplicate key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class _EvidenceLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(
            {
                "level": record.levelname,
                "level_number": int(record.levelno),
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


class _RuntimeEvidenceCapture:
    """Capture process-local warning, logging, and stderr evidence."""

    def __init__(self) -> None:
        self._stderr_chunks: list[bytes] = []
        self._stderr_reader_error: Optional[BaseException] = None
        self._stderr_read_fd: Optional[int] = None
        self._stderr_saved_fd: Optional[int] = None
        self._stderr_stop = threading.Event()
        self._stderr_stream: Optional[io.TextIOWrapper] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_context: Optional[Any] = None
        self._log_handler = _EvidenceLogHandler()
        self._warning_context = warnings.catch_warnings(record=True)
        self._root_logger = logging.getLogger()
        self._previous_root_level = self._root_logger.level
        self._observed_warnings: list[Any] = []
        self._entered = False

    def _drain_stderr(self) -> None:
        descriptor = self._stderr_read_fd
        if descriptor is None:
            return
        eof_deadline: Optional[float] = None
        try:
            while True:
                if self._stderr_stop.is_set():
                    if eof_deadline is None:
                        eof_deadline = time.monotonic() + 0.5
                    elif time.monotonic() >= eof_deadline:
                        raise RuntimeError(
                            "OS stderr pipe did not reach EOF after parent writers closed"
                        )
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    timeout = 0.1
                    if eof_deadline is not None:
                        timeout = max(
                            0.0, min(timeout, eof_deadline - time.monotonic())
                        )
                    select.select((descriptor,), (), (), timeout)
                    continue
                if not chunk:
                    break
                self._stderr_chunks.append(chunk)
        except BaseException as exc:
            self._stderr_reader_error = exc
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _start_stderr_capture(self) -> None:
        saved_fd = os.dup(2)
        read_fd: Optional[int] = None
        write_fd: Optional[int] = None
        try:
            read_fd, write_fd = os.pipe()
            os.set_blocking(read_fd, False)
            self._stderr_saved_fd = saved_fd
            self._stderr_read_fd = read_fd
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                name="adaptive-oracle-stderr-capture",
                daemon=False,
            )
            self._stderr_thread.start()
            os.dup2(write_fd, 2, inheritable=True)
            os.close(write_fd)
            write_fd = None
            self._stderr_stream = os.fdopen(
                os.dup(2),
                "w",
                encoding="utf-8",
                errors="backslashreplace",
                buffering=1,
                closefd=True,
            )
            self._stderr_context = redirect_stderr(self._stderr_stream)
            self._stderr_context.__enter__()
        except BaseException:
            if self._stderr_saved_fd is not None:
                try:
                    os.dup2(self._stderr_saved_fd, 2, inheritable=True)
                finally:
                    os.close(self._stderr_saved_fd)
                    self._stderr_saved_fd = None
            else:
                os.close(saved_fd)
            if write_fd is not None:
                os.close(write_fd)
            if self._stderr_stream is not None:
                self._stderr_stream.close()
                self._stderr_stream = None
            self._stderr_stop.set()
            thread = self._stderr_thread
            if thread is not None:
                thread.join()
                self._stderr_thread = None
            elif read_fd is not None:
                os.close(read_fd)
            self._stderr_read_fd = None
            raise

    def _stop_stderr_capture(self) -> None:
        context = self._stderr_context
        stream = self._stderr_stream
        saved_fd = self._stderr_saved_fd
        thread = self._stderr_thread
        errors: list[BaseException] = []
        try:
            if stream is not None:
                stream.flush()
        except BaseException as exc:
            errors.append(exc)
        try:
            if context is not None:
                context.__exit__(None, None, None)
        except BaseException as exc:
            errors.append(exc)
        try:
            if saved_fd is not None:
                os.dup2(saved_fd, 2, inheritable=True)
        except BaseException as exc:
            errors.append(exc)
            try:
                os.close(2)
            except OSError:
                pass
        finally:
            if saved_fd is not None:
                try:
                    os.close(saved_fd)
                except BaseException as exc:
                    errors.append(exc)
        try:
            if stream is not None:
                stream.close()
        except BaseException as exc:
            errors.append(exc)
        self._stderr_stop.set()
        if thread is not None:
            thread.join()
        if self._stderr_reader_error is not None:
            errors.append(self._stderr_reader_error)
        self._stderr_context = None
        self._stderr_stream = None
        self._stderr_saved_fd = None
        self._stderr_read_fd = None
        self._stderr_thread = None
        if errors:
            raise RuntimeError("OS stderr evidence capture failed") from errors[0]

    def __enter__(self) -> "_RuntimeEvidenceCapture":
        if self._entered:
            raise RuntimeError("runtime evidence capture may be entered only once")
        self._observed_warnings = self._warning_context.__enter__()
        try:
            warnings.simplefilter("always")
            self._start_stderr_capture()
            self._root_logger.setLevel(logging.NOTSET)
            self._root_logger.addHandler(self._log_handler)
            self._entered = True
            return self
        except BaseException:
            if self._stderr_saved_fd is not None:
                self._stop_stderr_capture()
            self._warning_context.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc, traceback_value) -> bool:
        if not self._entered:
            return False
        cleanup_errors: list[BaseException] = []
        try:
            self._root_logger.removeHandler(self._log_handler)
            self._root_logger.setLevel(self._previous_root_level)
        except BaseException as cleanup_exc:
            cleanup_errors.append(cleanup_exc)
        try:
            self._stop_stderr_capture()
        except BaseException as cleanup_exc:
            cleanup_errors.append(cleanup_exc)
        try:
            self._warning_context.__exit__(exc_type, exc, traceback_value)
        except BaseException as cleanup_exc:
            cleanup_errors.append(cleanup_exc)
        finally:
            self._entered = False
        if cleanup_errors:
            raise RuntimeError("runtime evidence capture finalization failed") from (
                cleanup_errors[0]
            )
        return False

    def record(self) -> dict[str, Any]:
        if self._entered:
            raise RuntimeError("runtime evidence cannot be read while capture is active")
        warning_rows = [
            {
                "category": row.category.__name__,
                "filename": str(row.filename),
                "lineno": int(row.lineno),
                "message": str(row.message),
            }
            for row in self._observed_warnings
        ]
        log_rows = copy.deepcopy(self._log_handler.records)
        stderr_bytes = b"".join(self._stderr_chunks)
        stderr_text = stderr_bytes.decode("utf-8", errors="surrogateescape")
        warning_log_count = sum(
            row["level_number"] >= logging.WARNING for row in log_rows
        )
        stderr_warning_line_count = contract.stderr_warning_line_count(stderr_text)
        warning_count = (
            len(warning_rows) + warning_log_count + stderr_warning_line_count
        )
        return {
            "schema": RUNTIME_EVIDENCE_SCHEMA,
            "capture_scope": "runtime_load_environment_pipeline_and_generation",
            "python_warnings": warning_rows,
            "python_warnings_sha256": contract.canonical_sha256(warning_rows),
            "logging_records": log_rows,
            "logging_records_sha256": contract.canonical_sha256(log_rows),
            "stderr": stderr_text,
            "stderr_byte_count": len(stderr_bytes),
            "stderr_sha256": sha256_bytes(stderr_bytes),
            "warnings": {
                "count": warning_count,
                "python_warning_count": len(warning_rows),
                "logging_warning_or_higher_count": warning_log_count,
                "stderr_warning_line_count": stderr_warning_line_count,
            },
        }


def _require_warning_free_runtime_evidence(evidence: Mapping[str, Any]) -> None:
    warning_record = evidence.get("warnings")
    if not isinstance(warning_record, Mapping) or warning_record.get("count") != 0:
        raise RuntimeError("engineering generation emitted runtime warnings")


def _require_exact_fields(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise ValueError(
            f"{label} fields differ: missing={missing}, extra={extra}"
        )
    return value


def _canonical_equal(observed: Any, expected: Any, label: str) -> None:
    if contract.canonical_json_bytes(observed) != contract.canonical_json_bytes(
        expected
    ):
        raise ValueError(f"{label} differs from the exact executable contract")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_unique_yaml_bytes(raw: bytes, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        parsed = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 YAML") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a mapping")
    contract.canonical_json_bytes(parsed)
    return parsed, sha256_bytes(raw)


def _cleanup_runtime_resources(runtime: Optional[SimpleNamespace] = None) -> None:
    """Close Torch's import-time temporary directory before evidence capture ends."""

    module = sys.modules.get("torch.distributed.nn.jit.instantiator")
    temporary = None if module is None else getattr(module, "_TEMP_DIR", None)
    if temporary is not None:
        temporary.cleanup()
    del runtime


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value!r}")


def _load_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    contract.canonical_json_bytes(value)
    return value


def _read_prompt_assets_bytes(
    csv_bytes: bytes, manifest_bytes: bytes
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    try:
        text = csv_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("adaptive-oracle prompt CSV is not UTF-8") from exc
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    manifest = _load_json_bytes(
        manifest_bytes, label="adaptive-oracle prompt manifest"
    )
    return rows, manifest


def _validate_exclusion_inventory_binding(
    prompt_manifest: Mapping[str, Any],
    inventory_bytes: bytes,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate the prompt manifest's exact binding to the held inventory bytes."""

    if not isinstance(inventory_bytes, bytes):
        raise TypeError("adaptive-oracle exclusion inventory must be immutable bytes")
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(
        expected_sha256
    ) is None:
        raise ValueError("exclusion inventory expected SHA-256 is malformed")
    binding = _require_exact_fields(
        prompt_manifest.get("exclusion_inventory"),
        frozenset({"schema", "path", "sha256"}),
        "prompt manifest exclusion inventory binding",
    )
    observed_sha256 = sha256_bytes(inventory_bytes)
    expected_binding = {
        "schema": EXCLUSION_INVENTORY_SCHEMA,
        "path": SOURCE_PATHS["exclusion_inventory"],
        "sha256": observed_sha256,
    }
    _canonical_equal(
        binding,
        expected_binding,
        "prompt manifest exclusion inventory binding",
    )
    if observed_sha256 != expected_sha256:
        raise ValueError("exclusion inventory bytes differ from the authorized source")
    inventory = _load_json_bytes(
        inventory_bytes, label="adaptive-oracle exclusion inventory"
    )
    if inventory.get("schema") != EXCLUSION_INVENTORY_SCHEMA:
        raise ValueError("adaptive-oracle exclusion inventory schema differs")
    return inventory


def _expected_model() -> dict[str, Any]:
    return {
        "model_id": model_snapshot.MODEL_ID,
        "revision": model_snapshot.MODEL_REVISION,
        "cache_dir": model_snapshot.MODEL_CACHE,
        "local_files_only": True,
        "torch_dtype": "float16",
        "variant": "fp16",
        "pipeline_class": PIPELINE_CLASS,
        "snapshot_manifest_schema": model_snapshot.MODEL_SNAPSHOT_SCHEMA,
        "snapshot_manifest_sha256": (
            model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256
        ),
        "snapshot_loaded_file_count": len(
            model_snapshot.expected_model_manifest()["files"]
        ),
    }


def _expected_execution(
    *,
    device: str,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": contract.EXPERIMENT_ID,
        "split_role": contract.SPLIT_ROLE,
        "output_dir": OUTPUT_DIR,
        "cuda_device": device,
        "width": contract.WIDTH,
        "height": contract.HEIGHT,
        "num_inference_steps": contract.NUM_INFERENCE_STEPS,
        "cfg_scale": contract.CFG_SCALE,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "guidance_rescale": 0.0,
        "scheduler_class": contract.SCHEDULER_CLASS,
        "prediction_type": contract.SCHEDULER_PREDICTION_TYPE,
        "scheduler_churn": contract.SCHEDULER_CHURN,
        "scheduler_config_sha256_v2": contract.SCHEDULER_CONFIG_SHA256_V2,
        "scheduler_schedule_sha256": contract.SCHEDULER_SCHEDULE_SHA256,
        "scheduler_construction_init_noise_sigma": (
            contract.SCHEDULER_CONSTRUCTION_INIT_NOISE_SIGMA
        ),
        "scheduler_effective_init_noise_sigma": (
            contract.SCHEDULER_EFFECTIVE_INIT_NOISE_SIGMA
        ),
        "model_snapshot_manifest_sha256": (
            contract.MODEL_SNAPSHOT_MANIFEST_SHA256
        ),
        "eta": 0.0,
        "stage": "stage1",
        "stage2_enabled": False,
        "power_calibrate": 0,
        "batch_size": 1,
        "num_images_per_prompt": 1,
        "prompt_count": contract.PROMPT_COUNT,
        "tasks_per_prompt": contract.TASKS_PER_PROMPT,
        "total_task_count": contract.TOTAL_TASK_COUNT,
        "contract_sha256": contract.CONTRACT_SHA256,
        "design_sha256": design["design_sha256"],
        "tasks_sha256": design["tasks_sha256"],
        "evidence_scope": dict(_EVIDENCE_SCOPE),
    }


def _expected_registration(
    *,
    design: Mapping[str, Any],
    prompt_csv_sha256: str,
    prompt_manifest_sha256: str,
    exclusion_inventory_sha256: str,
    environment_lock_sha256: str,
) -> dict[str, Any]:
    """Return the exact non-executable engineering registration."""

    return {
        "schema": REGISTRATION_SCHEMA,
        "status": REGISTRATION_STATUS,
        "experiment_id": contract.EXPERIMENT_ID,
        "scope": {
            "gpu_generation": False,
            "scoring": False,
            "quality_inspection": False,
            "method_selection": False,
            "renderer_training": False,
            "rl": False,
            "executable_authorization_required": True,
        },
        "authorization_boundary": {
            "allowed_cuda_devices": list(ALLOWED_CUDA_DEVICES),
            "cpu_audit_path": CPU_AUDIT_PATH,
            "executable_authorization_path": AUTHORIZATION_PATH,
            "output_dir": OUTPUT_DIR,
            "one_shot": True,
            "resume": False,
        },
        "inputs": {
            "prompt_csv": {
                "path": SOURCE_PATHS["prompt_csv"],
                "sha256": prompt_csv_sha256,
            },
            "prompt_manifest": {
                "path": SOURCE_PATHS["prompt_manifest"],
                "sha256": prompt_manifest_sha256,
            },
            "exclusion_inventory": {
                "path": SOURCE_PATHS["exclusion_inventory"],
                "sha256": exclusion_inventory_sha256,
            },
            "environment_lock": {
                "path": SOURCE_PATHS["environment_lock"],
                "sha256": environment_lock_sha256,
            },
        },
        "model": _expected_model(),
        "execution": {
            key: value
            for key, value in _expected_execution(
                device=ALLOWED_CUDA_DEVICES[0], design=design
            ).items()
            if key != "cuda_device"
        },
        "design": {
            "prompt_count": contract.PROMPT_COUNT,
            "tasks_per_prompt": contract.TASKS_PER_PROMPT,
            "total_task_count": contract.TOTAL_TASK_COUNT,
            "primary_action_count": 13,
            "selected_control_count_per_prompt": 2,
            "contract_sha256": contract.CONTRACT_SHA256,
            "design_sha256": design["design_sha256"],
            "tasks_sha256": design["tasks_sha256"],
            "primary_action_bank_sha256": contract.PRIMARY_ACTION_BANK_SHA256,
            "signed_orbit_cycle_sha256": contract.SIGNED_ORBIT_CYCLE_SHA256,
            "target_update_ratio": contract.ACTIVE_TARGET_UPDATE_RATIO,
            "target_ratio_tolerance": contract.TARGET_RATIO_TOLERANCE,
            "hard_update_cap": contract.HARD_UPDATE_CAP,
        },
        "interpretation": {
            "generation_integrity_only": True,
            "quality_claim_allowed": False,
            "formal_search_authorized": False,
            "distillation_authorized": False,
            "rl_authorized": False,
        },
    }


def _validated_pair(value: Any, *, expected_path: str, label: str) -> dict[str, str]:
    pair = _require_exact_fields(value, _PAIR_FIELDS, label)
    if pair["path"] != expected_path:
        raise ValueError(f"{label} path differs from {expected_path}")
    digest = pair["sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hex characters")
    return {"path": expected_path, "sha256": digest}


def _validate_commit_topology(
    value: Any,
    *,
    reviewed_commit: str,
    authorization_binding: Mapping[str, Any],
) -> dict[str, str]:
    fields = frozenset(
        {
            "schema",
            "implementation_commit",
            "cpu_audit_commit",
            "authorization_commit",
            "head_commit",
            "cpu_audit_parent",
            "authorization_parent",
        }
    )
    topology = dict(_require_exact_fields(value, fields, "launcher commit topology"))
    if topology["schema"] != COMMIT_TOPOLOGY_SCHEMA:
        raise ValueError("launcher commit topology schema differs")
    commit_fields = (
        "implementation_commit",
        "cpu_audit_commit",
        "authorization_commit",
        "head_commit",
        "cpu_audit_parent",
        "authorization_parent",
    )
    for name in commit_fields:
        commit = topology[name]
        if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
            raise ValueError(f"launcher commit topology {name} is not a full Git SHA")
    implementation_commit = topology["implementation_commit"]
    cpu_audit_commit = topology["cpu_audit_commit"]
    authorization_commit = topology["authorization_commit"]
    if len({implementation_commit, cpu_audit_commit, authorization_commit}) != 3:
        raise ValueError("launcher implementation/audit/authorization commits must be distinct")
    if (
        cpu_audit_commit != reviewed_commit
        or topology["cpu_audit_parent"] != implementation_commit
        or topology["authorization_parent"] != cpu_audit_commit
        or topology["head_commit"] != authorization_commit
        or authorization_binding.get("commit") != authorization_commit
    ):
        raise ValueError("launcher A/B/C commit topology binding differs")
    return topology


def _path_inside(path_value: str, directory: Path) -> bool:
    candidate = Path(path_value or os.getcwd()).resolve(strict=False)
    return candidate == directory or directory in candidate.parents


def _validate_launcher_evidence(value: Any) -> tuple[dict[str, Any], Path]:
    fields = frozenset(
        {
            "schema",
            "trust_root",
            "repo_root",
            "authorization",
            "authorization_raw_sha256",
            "authorization_binding",
            "commit_topology",
            "reviewed_commit",
            "python",
            "git",
            "sys_path",
            "repository_sys_path_entries",
            "cwd_sys_path_entries",
            "launcher",
            "inputs",
            "module_execution",
        }
    )
    evidence = dict(_require_exact_fields(value, fields, "launcher evidence"))
    if evidence["schema"] != LAUNCH_SCHEMA or evidence["trust_root"] != LAUNCH_TRUST_ROOT:
        raise ValueError("launcher evidence does not use the frozen trust root")
    reviewed_commit = evidence["reviewed_commit"]
    if not isinstance(reviewed_commit, str) or _COMMIT.fullmatch(reviewed_commit) is None:
        raise ValueError("launcher reviewed commit must be full lowercase Git SHA")
    root = Path(str(evidence["repo_root"])).resolve(strict=True)
    if Path.cwd().resolve(strict=True) != root:
        raise ValueError("launcher repository root differs from the process cwd")
    if evidence["repository_sys_path_entries"] != [] or evidence["cwd_sys_path_entries"] != []:
        raise ValueError("launcher retained repository or cwd import paths")
    if evidence["sys_path"] != list(sys.path):
        raise ValueError("launcher sys.path evidence differs from the live process")
    if any(_path_inside(str(entry), root) for entry in sys.path):
        raise ValueError("repository remains importable through live sys.path")
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.dont_write_bytecode
        or sys.flags.optimize != 0
        or sys.warnoptions != ["error"]
    ):
        raise ValueError("generator process does not satisfy -I -B -S -W error")

    binding = _require_exact_fields(
        evidence["authorization_binding"],
        frozenset({"path", "sha256", "commit"}),
        "launcher authorization binding",
    )
    if binding["path"] != AUTHORIZATION_PATH:
        raise ValueError("launcher authorization path differs")
    if (
        not isinstance(binding["sha256"], str)
        or _SHA256.fullmatch(binding["sha256"]) is None
        or binding["sha256"] != evidence["authorization_raw_sha256"]
        or not isinstance(binding["commit"], str)
        or _COMMIT.fullmatch(binding["commit"]) is None
    ):
        raise ValueError("launcher authorization binding is malformed")
    _validate_commit_topology(
        evidence["commit_topology"],
        reviewed_commit=reviewed_commit,
        authorization_binding=binding,
    )

    expected_origin = (
        f"<git-blob:{reviewed_commit}:{SOURCE_PATHS['generator']}>"
    )
    if __name__ != "generate_adaptive_oracle_engineering" or __file__ != expected_origin:
        raise RuntimeError("generator was not executed from its reviewed Git blob")
    if globals().get("__cached__") is not None:
        raise RuntimeError("reviewed generator unexpectedly has bytecode cache provenance")
    module_execution = evidence["module_execution"]
    if not isinstance(module_execution, Mapping):
        raise ValueError("launcher module execution evidence must be a mapping")
    generator_row = _require_exact_fields(
        module_execution.get(__name__),
        frozenset(
            {
                "source_name",
                "path",
                "origin",
                "sha256",
                "loader_id",
                "execution_count",
                "cached",
            }
        ),
        "launcher generator execution",
    )
    source_pair = evidence["authorization"].get("sources", {}).get("generator", {})
    if (
        generator_row["source_name"] != "generator"
        or generator_row["path"] != SOURCE_PATHS["generator"]
        or generator_row["origin"] != expected_origin
        or generator_row["sha256"] != source_pair.get("sha256")
        or generator_row["loader_id"] != LAUNCH_LOADER_ID
        or generator_row["execution_count"] != 1
        or generator_row["cached"] is not None
    ):
        raise RuntimeError("reviewed generator execution evidence differs")
    return copy.deepcopy(evidence), root


def _validate_launcher_bundle(
    launcher_evidence: Mapping[str, Any],
    input_bytes: Mapping[str, bytes],
    *,
    requested_device: str,
    requested_output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate only authenticated launcher evidence and reviewed Git-blob inputs."""

    evidence, root = _validate_launcher_evidence(launcher_evidence)
    commit_topology = _validate_commit_topology(
        evidence["commit_topology"],
        reviewed_commit=evidence["reviewed_commit"],
        authorization_binding=evidence["authorization_binding"],
    )
    expected_input_paths = _LAUNCH_INPUT_PATHS
    if set(input_bytes) != set(expected_input_paths):
        raise ValueError("launcher input-byte inventory differs from the exact allowlist")
    input_inventory = _require_exact_fields(
        evidence["inputs"], frozenset(expected_input_paths), "launcher input inventory"
    )
    normalized_inputs: dict[str, dict[str, Any]] = {}
    for name, expected_path in expected_input_paths.items():
        raw = input_bytes[name]
        if not isinstance(raw, bytes):
            raise TypeError(f"launcher input {name} must be immutable bytes")
        row = _require_exact_fields(
            input_inventory[name],
            frozenset({"path", "sha256", "byte_count", "commit"}),
            f"launcher input {name}",
        )
        if (
            row["path"] != expected_path
            or row["sha256"] != sha256_bytes(raw)
            or row["byte_count"] != len(raw)
            or row["commit"] != evidence["reviewed_commit"]
        ):
            raise ValueError(f"launcher input binding differs for {name}")
        normalized_inputs[name] = dict(row)

    config = evidence["authorization"]
    top = _require_exact_fields(
        config,
        frozenset({"schema", "status", "authorization", "model", "execution", "sources"}),
        "executable authorization",
    )
    if top["schema"] != AUTHORIZATION_SCHEMA:
        raise ValueError(
            f"executable authorization schema must be {AUTHORIZATION_SCHEMA}"
        )
    if top["status"] != AUTHORIZATION_STATUS:
        raise ValueError(
            "registration-only YAML is not an authorized engineering executable"
        )
    if not isinstance(requested_device, str) or _CUDA_DEVICE.fullmatch(
        requested_device
    ) is None:
        raise ValueError("exactly one explicit CUDA device such as cuda:1 is required")
    if requested_device not in ALLOWED_CUDA_DEVICES:
        raise ValueError("CUDA device is outside the registered cuda:1..cuda:4 set")

    expected_output = root.joinpath(*PurePosixPath(OUTPUT_DIR).parts)
    if Path(requested_output_dir).resolve(strict=False) != expected_output:
        raise ValueError(f"output directory must be exactly {expected_output}")

    authorization = _require_exact_fields(
        top["authorization"], _AUTHORIZATION_FIELDS, "authorization block"
    )
    reviewer = authorization["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("authorization reviewer must be non-empty")
    reviewed_commit = authorization["reviewed_commit"]
    if not isinstance(reviewed_commit, str) or _COMMIT.fullmatch(
        reviewed_commit
    ) is None:
        raise ValueError("reviewed_commit must be a full lowercase Git commit")
    if reviewed_commit != evidence["reviewed_commit"]:
        raise ValueError("authorization reviewed commit differs from launcher evidence")
    exact_authority = {
        "gpu_generation": True,
        "scoring": False,
        "quality_inspection": False,
        "method_selection": False,
        "renderer_training": False,
        "rl": False,
        "one_shot": True,
    }
    for key, expected in exact_authority.items():
        if authorization[key] is not expected:
            raise ValueError(f"authorization.{key} must be {expected}")

    source_registration = _validated_pair(
        authorization["source_registration"],
        expected_path=REGISTRATION_PATH,
        label="authorization source registration",
    )
    cpu_audit_pair = _validated_pair(
        authorization["cpu_audit"],
        expected_path=CPU_AUDIT_PATH,
        label="authorization CPU audit",
    )

    model = _require_exact_fields(top["model"], _MODEL_FIELDS, "model block")
    _canonical_equal(model, _expected_model(), "model block")
    if (
        contract.MODEL_ID != model_snapshot.MODEL_ID
        or contract.MODEL_REVISION != model_snapshot.MODEL_REVISION
        or contract.MODEL_SNAPSHOT_MANIFEST_SHA256
        != model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256
    ):
        raise RuntimeError("contract and model snapshot provenance drifted")
    sources = _require_exact_fields(
        top["sources"], frozenset(SOURCE_PATHS), "source block"
    )
    normalized_sources = {}
    for name, expected_path in SOURCE_PATHS.items():
        pair = _validated_pair(
            sources[name],
            expected_path=expected_path,
            label=f"source {name}",
        )
        normalized_sources[name] = pair
    _canonical_equal(
        normalized_sources["registration"],
        source_registration,
        "authorization/source registration binding",
    )
    for public_name, source_name in {
        "registration": "registration",
        "prompt_csv": "prompt_csv",
        "prompt_manifest": "prompt_manifest",
        "exclusion_inventory": "exclusion_inventory",
        "environment_lock": "environment_lock",
    }.items():
        if normalized_inputs[public_name]["sha256"] != normalized_sources[source_name]["sha256"]:
            raise ValueError(f"launcher input/source hash differs for {public_name}")
    if normalized_inputs["cpu_audit"]["sha256"] != cpu_audit_pair["sha256"]:
        raise ValueError("launcher CPU-audit bytes differ from authorization")

    prompt_rows, prompt_manifest = _read_prompt_assets_bytes(
        input_bytes["prompt_csv"], input_bytes["prompt_manifest"]
    )
    exclusion_inventory = _validate_exclusion_inventory_binding(
        prompt_manifest,
        input_bytes["exclusion_inventory"],
        expected_sha256=normalized_sources["exclusion_inventory"]["sha256"],
    )
    design = contract.build_engineering_design(prompt_rows, prompt_manifest)
    contract.validate_engineering_design(design, prompt_rows, prompt_manifest)
    registration, registration_sha256 = _load_unique_yaml_bytes(
        input_bytes["registration"], label="non-executable registration"
    )
    _canonical_equal(
        registration,
        _expected_registration(
            design=design,
            prompt_csv_sha256=normalized_sources["prompt_csv"]["sha256"],
            prompt_manifest_sha256=normalized_sources["prompt_manifest"][
                "sha256"
            ],
            exclusion_inventory_sha256=normalized_sources[
                "exclusion_inventory"
            ]["sha256"],
            environment_lock_sha256=normalized_sources["environment_lock"][
                "sha256"
            ],
        ),
        "non-executable registration",
    )
    if registration_sha256 != source_registration["sha256"]:
        raise ValueError("registration source hash differs from its parsed bytes")
    execution = top["execution"]
    _canonical_equal(
        execution,
        _expected_execution(device=requested_device, design=design),
        "execution block",
    )

    implementation_hashes = {
        pair["path"]: pair["sha256"]
        for name, pair in normalized_sources.items()
        if name in _IMPLEMENTATION_SOURCE_NAMES
    }
    if set(implementation_hashes) != set(
        cpu_audit_contract.IMPLEMENTATION_SOURCE_PATHS
    ):
        raise RuntimeError(
            "authorization and CPU-audit implementation allowlists differ"
        )
    cpu_audit_value = _load_json_bytes(
        input_bytes["cpu_audit"], label="adaptive-oracle CPU audit"
    )
    cpu_audit_summary = dict(
        cpu_audit_contract._validate_cpu_audit_mapping(
            cpu_audit_value,
            expected_implementation_commit=commit_topology[
                "implementation_commit"
            ],
            expected_implementation_hashes=implementation_hashes,
            expected_environment_lock_sha256=normalized_sources[
                "environment_lock"
            ]["sha256"],
            expected_contract_sha256=contract.CONTRACT_SHA256,
            expected_design_sha256=design["design_sha256"],
            expected_tasks_sha256=design["tasks_sha256"],
            expected_prompt_csv_sha256=normalized_sources["prompt_csv"][
                "sha256"
            ],
            expected_exclusion_inventory_sha256=normalized_sources[
                "exclusion_inventory"
            ]["sha256"],
            expected_prompt_manifest_sha256=normalized_sources[
                "prompt_manifest"
            ]["sha256"],
            expected_registration_sha256=normalized_sources["registration"][
                "sha256"
            ],
        )
    )
    return {
        "config": copy.deepcopy(config),
        "authorization_sha256": evidence["authorization_raw_sha256"],
        "authorization_binding": copy.deepcopy(evidence["authorization_binding"]),
        "commit_topology": commit_topology,
        "reviewed_commit": reviewed_commit,
        "reviewer": reviewer,
        "source_paths": {
            name: pair["path"] for name, pair in normalized_sources.items()
        },
        "source_hashes": normalized_sources,
        "cpu_audit_sha256": cpu_audit_pair["sha256"],
        "cpu_audit_summary": cpu_audit_summary,
        "prompt_rows": prompt_rows,
        "prompt_manifest": prompt_manifest,
        "exclusion_inventory": exclusion_inventory,
        "design": design,
        "model_snapshot_manifest": model_snapshot.expected_model_manifest(),
        "launcher_evidence": evidence,
        "launcher_inputs": normalized_inputs,
        "authorized_input_bytes": {
            name: bytes(raw) for name, raw in input_bytes.items()
        },
        "environment_lock_bytes": bytes(input_bytes["environment_lock"]),
        "device": requested_device,
        "output_dir": str(expected_output),
        "repo_root": str(root),
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o640)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_create_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish bytes without ever replacing an existing path."""

    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("atomic output parent must be a regular directory")
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    _write_exclusive(temporary, payload)
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(parent)
    except BaseException:
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_create_json(path: Path, value: Any) -> None:
    atomic_create_bytes(path, contract.canonical_json_bytes(value) + b"\n")


def _initialize_attempt_directory(
    output_dir: Path, attempt: Mapping[str, Any]
) -> Path:
    marker = output_dir / "attempt.json"
    atomic_create_json(marker, attempt)
    records = output_dir / "records"
    os.mkdir(records, mode=0o750)
    _fsync_directory(output_dir)
    return records


def _verified_module_execution(
    module_name: str,
    source_name: str,
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    module = sys.modules.get(module_name)
    if module is None:
        raise RuntimeError(f"reviewed runtime module was not loaded: {module_name}")
    loader = getattr(module, "__loader__", None)
    snapshot = getattr(loader, "snapshot", None)
    execution_count = getattr(loader, "execution_count", None)
    pair = validated["source_hashes"][source_name]
    expected_origin = (
        f"<git-blob:{validated['reviewed_commit']}:{pair['path']}>"
    )
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("source_name") != source_name
        or snapshot.get("relative_path") != pair["path"]
        or snapshot.get("origin") != expected_origin
        or snapshot.get("sha256") != pair["sha256"]
        or sha256_bytes(snapshot.get("bytes", b"")) != pair["sha256"]
        or execution_count != 1
        or getattr(module, "__file__", None) != expected_origin
        or getattr(module, "__cached__", None) is not None
    ):
        raise RuntimeError(f"reviewed runtime module provenance differs: {module_name}")
    return {
        "source_name": source_name,
        "path": pair["path"],
        "origin": expected_origin,
        "sha256": pair["sha256"],
        "loader_id": LAUNCH_LOADER_ID,
        "execution_count": 1,
        "cached": None,
    }


def _load_verified_runtime_components(
    validated: Mapping[str, Any]
) -> SimpleNamespace:
    """Import the protected runtime graph through the launcher's Git-blob finder."""

    import_order = (
        "adaptive_oracle_contract",
        "adaptive_oracle_model_snapshot",
        "audit_adaptive_oracle_cpu",
        "adaptive_oracle_generation_environment",
        "AttentionGuidance",
        "InferencePipelines.RepLDM.pipeline_repldm_sdxl",
    )
    modules = {name: importlib.import_module(name) for name in import_order}
    for name in _VERIFIED_RUNTIME_MODULE_SOURCES:
        modules[name] = importlib.import_module(name)

    import torch
    from diffusers import EulerDiscreteScheduler

    if (
        modules["adaptive_oracle_contract"] is not contract
        or modules["adaptive_oracle_model_snapshot"] is not model_snapshot
        or modules["audit_adaptive_oracle_cpu"] is not cpu_audit_contract
    ):
        raise RuntimeError("bootstrap module identity differs from reviewed imports")
    adaptive_oracle_module = modules["AttentionGuidance.adaptive_oracle"]
    pipeline_module = modules["InferencePipelines.RepLDM.pipeline_repldm_sdxl"]
    environment_module = modules["adaptive_oracle_generation_environment"]
    execution = {
        module_name: _verified_module_execution(
            module_name, source_name, validated
        )
        for module_name, source_name in _VERIFIED_RUNTIME_MODULE_SOURCES.items()
    }
    runtime = SimpleNamespace(
        torch=torch,
        Scheduler=EulerDiscreteScheduler,
        Pipeline=pipeline_module.RepLDMSDXLPipeline,
        BasisProvider=adaptive_oracle_module.AdaptiveOracleBasisProvider,
        RandomContext=adaptive_oracle_module.AdaptiveOracleRandomContext,
        Renderer=adaptive_oracle_module.FixedRatioMomentGeodesicRenderer,
        validate_environment_lock_bytes=(
            environment_module.validate_environment_lock_bytes
        ),
        verified_source_execution=execution,
    )
    if (
        runtime.Pipeline is not pipeline_module.RepLDMSDXLPipeline
        or runtime.BasisProvider
        is not adaptive_oracle_module.AdaptiveOracleBasisProvider
        or runtime.RandomContext
        is not adaptive_oracle_module.AdaptiveOracleRandomContext
        or runtime.Renderer
        is not adaptive_oracle_module.FixedRatioMomentGeodesicRenderer
        or runtime.validate_environment_lock_bytes
        is not environment_module.validate_environment_lock_bytes
    ):
        raise RuntimeError("reviewed runtime export identity differs")
    return runtime


def _revalidate_immutable_authorized_bundle(
    validated: Mapping[str, Any],
    runtime: SimpleNamespace,
    *,
    phase: str,
) -> dict[str, Any]:
    """Recheck held input bytes and reviewed runtime sources at a phase boundary."""

    if phase not in {"after_runtime_import", "after_generation"}:
        raise ValueError("authorized-bundle revalidation phase is not registered")
    topology = _validate_commit_topology(
        validated.get("commit_topology"),
        reviewed_commit=validated.get("reviewed_commit"),
        authorization_binding=validated.get("authorization_binding", {}),
    )
    launcher_evidence = validated.get("launcher_evidence")
    cpu_audit_summary = validated.get("cpu_audit_summary")
    if (
        not isinstance(launcher_evidence, Mapping)
        or launcher_evidence.get("commit_topology") != topology
        or launcher_evidence.get("reviewed_commit") != validated.get("reviewed_commit")
        or launcher_evidence.get("authorization_binding")
        != validated.get("authorization_binding")
        or not isinstance(cpu_audit_summary, Mapping)
        or cpu_audit_summary.get("implementation_commit")
        != topology["implementation_commit"]
    ):
        raise RuntimeError(f"held launcher commit topology drifted {phase}")

    expected_names = frozenset(_LAUNCH_INPUT_PATHS)
    held = _require_exact_fields(
        validated.get("authorized_input_bytes"),
        expected_names,
        "held authorized input bytes",
    )
    inventory = _require_exact_fields(
        validated.get("launcher_inputs"),
        expected_names,
        "held launcher input inventory",
    )
    input_digests: dict[str, str] = {}
    for name in sorted(expected_names):
        raw = held[name]
        if not isinstance(raw, bytes):
            raise RuntimeError(f"held authorized input {name} is no longer immutable bytes")
        digest = sha256_bytes(raw)
        row = inventory[name]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "byte_count", "commit"}
            or row.get("path") != _LAUNCH_INPUT_PATHS[name]
            or row.get("sha256") != digest
            or row.get("byte_count") != len(raw)
            or row.get("commit") != validated["reviewed_commit"]
        ):
            raise RuntimeError(
                f"held authorized input binding drifted {phase}: {name}"
            )
        input_digests[name] = digest

    for input_name, source_name in {
        "registration": "registration",
        "prompt_csv": "prompt_csv",
        "prompt_manifest": "prompt_manifest",
        "exclusion_inventory": "exclusion_inventory",
        "environment_lock": "environment_lock",
    }.items():
        if input_digests[input_name] != validated["source_hashes"][source_name][
            "sha256"
        ]:
            raise RuntimeError(
                f"held authorized input/source binding drifted {phase}: {input_name}"
            )
    if input_digests["cpu_audit"] != validated["cpu_audit_sha256"]:
        raise RuntimeError(f"held authorized CPU-audit binding drifted {phase}")
    prompt_manifest = _load_json_bytes(
        held["prompt_manifest"], label="held adaptive-oracle prompt manifest"
    )
    exclusion_inventory = _validate_exclusion_inventory_binding(
        prompt_manifest,
        held["exclusion_inventory"],
        expected_sha256=input_digests["exclusion_inventory"],
    )
    _canonical_equal(
        exclusion_inventory,
        validated["exclusion_inventory"],
        f"held exclusion inventory {phase}",
    )
    if held["environment_lock"] != validated["environment_lock_bytes"]:
        raise RuntimeError(f"held environment-lock bytes drifted {phase}")

    execution = {
        module_name: _verified_module_execution(module_name, source_name, validated)
        for module_name, source_name in _VERIFIED_RUNTIME_MODULE_SOURCES.items()
    }
    _canonical_equal(
        execution,
        runtime.verified_source_execution,
        f"reviewed runtime source execution {phase}",
    )
    return {
        "phase": phase,
        "input_sha256": input_digests,
        "runtime_source_execution": copy.deepcopy(execution),
    }


def _validate_cuda_device(runtime: SimpleNamespace, device: str) -> None:
    torch = runtime.torch
    if device not in ALLOWED_CUDA_DEVICES:
        raise RuntimeError("CUDA device is outside the registered cuda:1..cuda:4 set")
    if "CUDA_VISIBLE_DEVICES" in os.environ or "CUDA_DEVICE_ORDER" in os.environ:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER must be unset so cuda:N "
            "is a physical GPU index"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("authorized engineering generation requires CUDA")
    index = int(device.split(":", 1)[1])
    if index >= int(torch.cuda.device_count()):
        raise RuntimeError("authorized CUDA device index is unavailable")
    torch.cuda.set_device(index)
    if int(torch.cuda.current_device()) != index:
        raise RuntimeError("CUDA current device differs from explicit authorization")


def _scheduler_config_record(scheduler: Any) -> dict[str, Any]:
    config = dict(getattr(scheduler, "config", {}))
    if isinstance(config.get("_use_default_values"), list):
        config["_use_default_values"] = sorted(config["_use_default_values"])
    return json.loads(json.dumps(config, sort_keys=True, default=str))


def scheduler_config_sha256_v2(config: Mapping[str, Any]) -> str:
    """Hash the effective scheduler config with the registered v2 encoding."""

    payload = json.dumps(dict(config), sort_keys=True, default=str).encode("utf-8")
    return sha256_bytes(payload)


def scheduler_schedule_sha256(scheduler: Any) -> str:
    """Hash the exact active timestep/sigma arrays."""

    def values(value: Any) -> Optional[list[float]]:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1).tolist()
        return [float(item) for item in value]

    payload = {
        "timesteps": values(getattr(scheduler, "timesteps", None)),
        "sigmas": values(getattr(scheduler, "sigmas", None)),
    }
    return contract.canonical_sha256(payload)


def _validated_pipeline_schedule_record(pipe: Any) -> dict[str, Any]:
    raw = getattr(pipe, "_last_scheduler_schedule_record", None)
    record = _require_exact_fields(
        raw,
        frozenset(
            {
                "timesteps",
                "sigmas",
                "schedule_sha256",
                "construction_init_noise_sigma",
                "effective_init_noise_sigma",
            }
        ),
        "pipeline scheduler schedule record",
    )
    timesteps = record["timesteps"]
    sigmas = record["sigmas"]
    if not isinstance(timesteps, list) or len(timesteps) != contract.NUM_INFERENCE_STEPS:
        raise RuntimeError("pipeline scheduler did not expose exactly 50 timesteps")
    if not isinstance(sigmas, list) or len(sigmas) != contract.NUM_INFERENCE_STEPS + 1:
        raise RuntimeError("pipeline scheduler did not expose exactly 51 sigmas")
    for label, values in (("timestep", timesteps), ("sigma", sigmas)):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise RuntimeError(f"pipeline scheduler exposed a non-finite {label}")
    observed_hash = contract.canonical_sha256(
        {"timesteps": timesteps, "sigmas": sigmas}
    )
    if record["schedule_sha256"] != observed_hash:
        raise RuntimeError("pipeline scheduler schedule record has an invalid hash")
    if observed_hash != contract.SCHEDULER_SCHEDULE_SHA256:
        raise RuntimeError("pipeline scheduler schedule differs from the frozen hash")
    if (
        record["construction_init_noise_sigma"]
        != contract.SCHEDULER_CONSTRUCTION_INIT_NOISE_SIGMA
    ):
        raise RuntimeError("pipeline scheduler construction init-noise sigma differs")
    if (
        record["effective_init_noise_sigma"]
        != contract.SCHEDULER_EFFECTIVE_INIT_NOISE_SIGMA
    ):
        raise RuntimeError("pipeline scheduler effective init-noise sigma differs")
    return copy.deepcopy(dict(record))


def _validate_base_scheduler(scheduler: Any) -> dict[str, Any]:
    if type(scheduler).__name__ != contract.SCHEDULER_CLASS:
        raise RuntimeError("base scheduler must be exactly EulerDiscreteScheduler")
    config = _scheduler_config_record(scheduler)
    if config.get("prediction_type") != contract.SCHEDULER_PREDICTION_TYPE:
        raise RuntimeError("base Euler scheduler prediction_type must be epsilon")
    if scheduler_config_sha256_v2(config) != contract.SCHEDULER_CONFIG_SHA256_V2:
        raise RuntimeError("base Euler scheduler config differs from the frozen hash")
    return config


def _validate_scheduler_schedule(
    scheduler: Any, *, device: str
) -> None:
    construction_sigma = float(scheduler.init_noise_sigma)
    scheduler.set_timesteps(contract.NUM_INFERENCE_STEPS, device=device)
    effective_sigma = float(scheduler.init_noise_sigma)
    if construction_sigma != contract.SCHEDULER_CONSTRUCTION_INIT_NOISE_SIGMA:
        raise RuntimeError("Euler construction init-noise sigma differs")
    if effective_sigma != contract.SCHEDULER_EFFECTIVE_INIT_NOISE_SIGMA:
        raise RuntimeError("Euler effective init-noise sigma differs")
    if scheduler_schedule_sha256(scheduler) != contract.SCHEDULER_SCHEDULE_SHA256:
        raise RuntimeError("Euler 50-step schedule differs from the frozen hash")


def _fresh_scheduler(runtime: SimpleNamespace, base_config: Mapping[str, Any]):
    scheduler = runtime.Scheduler.from_config(copy.deepcopy(dict(base_config)))
    if type(scheduler).__name__ != contract.SCHEDULER_CLASS:
        raise RuntimeError("fresh scheduler class drifted from EulerDiscreteScheduler")
    observed = _scheduler_config_record(scheduler)
    if contract.canonical_json_bytes(observed) != contract.canonical_json_bytes(
        dict(base_config)
    ):
        raise RuntimeError("fresh scheduler config drifted from frozen base config")
    if observed.get("prediction_type") != "epsilon":
        raise RuntimeError("fresh scheduler prediction_type drifted from epsilon")
    return scheduler


def _load_pipeline(
    validated: Mapping[str, Any],
    runtime: SimpleNamespace,
    model_stage: Mapping[str, Any],
    *,
    verification_log: Optional[list[dict[str, Any]]] = None,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    model = validated["config"]["model"]
    verifications = [] if verification_log is None else verification_log

    def load_pipeline(pinned_stage_path: str) -> Any:
        if not pinned_stage_path.startswith("/proc/self/fd/"):
            raise RuntimeError("model loader did not receive a pinned stage root")
        return runtime.Pipeline.from_pretrained(
            pinned_stage_path,
            torch_dtype=runtime.torch.float16,
            variant=model["variant"],
            local_files_only=True,
        )

    pipe = model_snapshot.load_from_verified_staged_model_snapshot(
        model_stage,
        load_pipeline,
        expected_manifest_sha256=model["snapshot_manifest_sha256"],
        verification_log=verifications,
    )
    pipe = pipe.to(validated["device"])
    verifications.append(
        model_snapshot.verify_staged_model_snapshot(
            model_stage,
            expected_manifest_sha256=model["snapshot_manifest_sha256"],
        )
    )
    if type(pipe).__name__ != PIPELINE_CLASS:
        raise RuntimeError("loaded pipeline class differs from RepLDMSDXLPipeline")
    try:
        unet_device = str(next(pipe.unet.parameters()).device)
    except (AttributeError, StopIteration) as exc:
        raise RuntimeError("loaded pipeline U-Net has no device-bearing parameters") from exc
    if unet_device != validated["device"]:
        raise RuntimeError("loaded U-Net is not on the one explicit CUDA device")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    base_config = _validate_base_scheduler(pipe.scheduler)
    preview_scheduler = runtime.Scheduler.from_config(copy.deepcopy(base_config))
    _validate_scheduler_schedule(preview_scheduler, device=validated["device"])
    return pipe, base_config, verifications


def tensor_sha256(value: Any) -> str:
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        raise RuntimeError("torch must be loaded before hashing a tensor")
    byte_tensor = (
        value.detach()
        .contiguous()
        .cpu()
        .view(torch_module.uint8)
        .reshape(-1)
    )
    try:
        raw = byte_tensor.numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError):
        raw = bytes(byte_tensor.tolist())
    return sha256_bytes(raw)


def _single_number(value: Any, label: str) -> float:
    if not isinstance(value, list) or len(value) != 1:
        raise RuntimeError(f"{label} must contain exactly one batch value")
    result = float(value[0])
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is non-finite")
    return result


def _hook_signature(unet: Any) -> tuple[tuple[int, int], ...]:
    try:
        hooks = unet.up_blocks[0]._forward_pre_hooks
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("U-Net lacks the registered up_blocks.0 hook surface") from exc
    return tuple((int(key), id(value)) for key, value in hooks.items())


def _reset_pipeline_ledgers(pipe: Any) -> None:
    pipe._last_unet_calls_total = 0
    pipe._last_unet_calls_per_step = []
    pipe._last_scheduler_calls_total = 0
    pipe._last_scheduler_calls_per_step = []
    pipe._last_final_decode_calls = 0
    pipe._last_intermediate_decode_calls = 0
    pipe._last_latent_renderer_scheduler_mapping = None
    pipe._last_latent_renderer_diagnostics = None
    pipe._last_latent_renderer_provider_diagnostics = None
    pipe._last_latent_renderer_step_diagnostics = []
    pipe._last_scheduler_schedule_record = None
    pipe._last_prepared_initial_latent_sha256 = None
    pipe._last_latents_before_step_sha256 = []
    pipe._last_latents_after_step_sha256 = []


def _action_runtime(
    task: Mapping[str, Any], pipe: Any, runtime: SimpleNamespace
) -> tuple[Any, Any]:
    action = task["action"]
    if action["physical_no_op"]:
        return None, None
    random_context = None
    if action["affinity_source"] == "random_edge":
        random_context = runtime.RandomContext(
            experiment_id=contract.EXPERIMENT_ID,
            split_role=contract.SPLIT_ROLE,
            prompt_row_id=task["prompt"]["prompt_row_id"],
            seed=task["prompt"]["seed"],
        )
    provider = runtime.BasisProvider(
        pipe.unet if action["affinity_source"] == "feature" else None,
        batch_size=1,
        affinity_source=action["affinity_source"],
        orbit_name=action["orbit_name"],
        random_context=random_context,
    )
    renderer = runtime.Renderer(sign=action["sign"])
    return provider, renderer


def _feature_hook_record(provider_record: Mapping[str, Any]) -> dict[str, Any]:
    capture = provider_record.get("capture_record")
    if not isinstance(capture, Mapping):
        raise RuntimeError("feature action omitted capture diagnostics")
    if capture.get("capture_complete") is not True:
        raise RuntimeError("feature action capture was not complete")
    if capture.get("conditional_rows") != "second_half":
        raise RuntimeError("feature action did not consume the conditional CFG half")
    return {
        "module_path": capture.get("feature_block"),
        "input_name": contract.FEATURE_INPUT_NAME,
        "hook_calls": capture.get("hook_calls"),
        "consume_calls": capture.get("consume_calls"),
        "expected_cfg_shape": capture.get("expected_cfg_shape"),
        "captured_conditional_shape": capture.get("captured_shape"),
        "cfg_row_order": contract.CFG_ROW_ORDER,
        "detached": capture.get("detached"),
    }


def _random_counter_record(
    provider_record: Mapping[str, Any], task: Mapping[str, Any], step_index: int
) -> dict[str, Any]:
    context_record = provider_record.get("random_counter_context")
    if not isinstance(context_record, Mapping):
        raise RuntimeError("random action omitted counter context")
    expected_context = {
        "schema": contract.RANDOM_EDGE_COUNTER_SCHEMA,
        "experiment_id": contract.EXPERIMENT_ID,
        "split_role": contract.SPLIT_ROLE,
        "prompt_row_id": task["prompt"]["prompt_row_id"],
        "seed": task["prompt"]["seed"],
        "step_index": step_index,
        "orbit_name": task["action"]["orbit_name"],
    }
    _canonical_equal(context_record, expected_context, "random counter context")
    counter_hash = provider_record.get("random_counter_set_sha256")
    if not isinstance(counter_hash, str) or _SHA256.fullmatch(counter_hash) is None:
        raise RuntimeError("random action omitted its counter-set SHA-256")
    return {
        **expected_context,
        "undirected_edge_count": contract.ORBIT_EDGE_COUNTS[
            task["action"]["orbit_name"]
        ],
        "counter_set_sha256": counter_hash,
    }


def _validate_provider_record(
    provider: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    action = task["action"]
    expected_keys = {
        "implementation",
        "affinity_source",
        "selected_orbit",
        "selected_orbit_index",
        "selected_basis_shape",
        "selected_basis_dtype",
        "selected_basis_sha256",
        "local_diagnostics",
    }
    if action["feature_hook_required"]:
        expected_keys.add("capture_record")
    if action["random_counter_required"]:
        expected_keys.update(
            {"random_counter_context", "random_counter_set_sha256"}
        )
    if set(provider) != expected_keys:
        raise RuntimeError("provider diagnostics fields differ from the runtime contract")
    expected_orbit_index = list(contract.ORBIT_OFFSETS).index(action["orbit_name"])
    expected = {
        "implementation": ADAPTIVE_ORACLE_PROVIDER_ID,
        "affinity_source": action["affinity_source"],
        "selected_orbit": action["orbit_name"],
        "selected_orbit_index": expected_orbit_index,
        "selected_basis_shape": [1, 1, 4, contract.HEIGHT // 8, contract.WIDTH // 8],
        "selected_basis_dtype": "float32",
    }
    for key, value in expected.items():
        _canonical_equal(provider.get(key), value, f"provider diagnostic {key}")
    basis_hash = provider.get("selected_basis_sha256")
    if not isinstance(basis_hash, str) or _SHA256.fullmatch(basis_hash) is None:
        raise RuntimeError("provider selected basis SHA-256 is invalid")
    local = provider.get("local_diagnostics")
    if not isinstance(local, Mapping):
        raise RuntimeError("provider omitted local relational diagnostics")
    local_expected = {
        "orbit_names": list(contract.ORBIT_OFFSETS),
        "affinity_source": action["affinity_source"],
        "grid_size": contract.GRID_SIZE,
        "feature_norm_epsilon": 1e-6,
        "predicted_clean_norm_epsilon": 1e-6,
        "affinity_floor": 1e-6,
        "undirected_edge_counts": [
            contract.ORBIT_EDGE_COUNTS[name] for name in contract.ORBIT_OFFSETS
        ],
    }
    for key, value in local_expected.items():
        _canonical_equal(local.get(key), value, f"local diagnostic {key}")
    if action["random_counter_required"]:
        random_expected = {
            "random_edge_counter_schema": contract.RANDOM_EDGE_COUNTER_SCHEMA,
            "random_edge_actual_edge_counts": local_expected[
                "undirected_edge_counts"
            ],
            "random_edge_actual_edges_unique": True,
        }
        for key, value in random_expected.items():
            _canonical_equal(local.get(key), value, f"random local diagnostic {key}")


def _active_step_record(
    raw: Mapping[str, Any],
    task: Mapping[str, Any],
    step_index: int,
    *,
    latent_before_sha256: str,
    latent_after_sha256: str,
) -> dict[str, Any]:
    if raw.get("step_index") != step_index:
        raise RuntimeError("pipeline renderer step ledger is not contiguous")
    provider = raw.get("provider_diagnostics")
    if not isinstance(provider, Mapping):
        raise RuntimeError("active pipeline step omitted provider diagnostics")
    _validate_provider_record(provider, task)
    action = task["action"]
    if provider.get("affinity_source") != action["affinity_source"]:
        raise RuntimeError("provider affinity source differs from task")
    if provider.get("selected_orbit") != action["orbit_name"]:
        raise RuntimeError("provider orbit differs from task")
    round_trip = raw.get("native_scheduler_round_trip")
    if not isinstance(round_trip, Mapping):
        raise RuntimeError("active step omitted native scheduler round-trip diagnostics")
    mapped = raw.get("scheduler_mapped_intervention")
    if not isinstance(mapped, Mapping):
        raise RuntimeError("active step omitted scheduler-mapped intervention diagnostics")
    mapped = _require_exact_fields(
        mapped,
        frozenset(
            {
                "nominal_update_norm",
                "applied_update_norm",
                "applied_update_ratio",
                "target_ratio_error",
                "cap_hit",
                "solver_target_update_ratio",
                "solver_evaluations",
            }
        ),
        "scheduler-mapped intervention diagnostics",
    )
    solver_evaluations = mapped["solver_evaluations"]
    if (
        isinstance(solver_evaluations, bool)
        or not isinstance(solver_evaluations, int)
        or solver_evaluations not in {1, 14}
    ):
        raise RuntimeError("mapped-ratio solver evaluation count is invalid")
    feature_hook = (
        _feature_hook_record(provider) if action["feature_hook_required"] else None
    )
    random_counter = (
        _random_counter_record(provider, task, step_index)
        if action["random_counter_required"]
        else None
    )
    return {
        "step_index": step_index,
        "affinity_source": action["affinity_source"],
        "orbit_name": action["orbit_name"],
        "sign": action["sign"],
        "unet_calls": 1,
        "scheduler_calls": 1,
        "extra_unet_calls": 0,
        "extra_scheduler_calls": 0,
        "backbone_backward_calls": 0,
        "intermediate_decode_calls": 0,
        "feature_hook": feature_hook,
        "random_edge_counter": random_counter,
        "basis_sha256": provider.get("selected_basis_sha256"),
        "latent_before_sha256": latent_before_sha256,
        "latent_after_sha256": latent_after_sha256,
        "applied_update_ratio": _single_number(
            mapped.get("applied_update_ratio"), "scheduler-mapped applied update ratio"
        ),
        "target_ratio_error": _single_number(
            mapped.get("target_ratio_error"), "scheduler-mapped target ratio error"
        ),
        "channel_mean_abs_error": _single_number(raw.get("mean_error"), "mean error"),
        "channel_variance_relative_error": _single_number(
            raw.get("variance_error"), "variance error"
        ),
        "channel_covariance_drift": _single_number(
            raw.get("covariance_drift"), "covariance drift"
        ),
        "pred_original_sample_relative_l2_error": _single_number(
            round_trip.get("pred_original_sample_relative_l2_error"),
            "pred_original_sample round-trip error",
        ),
        "expected_prev_sample_relative_l2_error": _single_number(
            round_trip.get("expected_prev_sample_relative_l2_error"),
            "expected prev_sample round-trip error",
        ),
        "native_round_trip_max_abs_error": _single_number(
            round_trip.get("native_round_trip_max_abs_error"),
            "native round-trip max-abs error",
        ),
        "solver_target_update_ratio": _single_number(
            mapped.get("solver_target_update_ratio"),
            "mapped-ratio solver target",
        ),
        "mapped_ratio_solver_evaluations": solver_evaluations,
        "finite": True,
        "cap_hit": _single_bool(mapped.get("cap_hit"), "scheduler-mapped cap_hit"),
    }


def _single_value(value: Any, label: str) -> Any:
    if not isinstance(value, list) or len(value) != 1:
        raise RuntimeError(f"{label} must contain exactly one batch value")
    return value[0]


def _single_bool(value: Any, label: str) -> bool:
    result = _single_value(value, label)
    if not isinstance(result, bool):
        raise RuntimeError(f"{label} must contain one JSON boolean")
    return result


def _p0_step_record(
    task: Mapping[str, Any],
    step_index: int,
    unet_calls: int,
    scheduler_calls: int,
    *,
    latent_before_sha256: str,
    latent_after_sha256: str,
) -> dict[str, Any]:
    action = task["action"]
    return {
        "step_index": step_index,
        "affinity_source": action["affinity_source"],
        "orbit_name": action["orbit_name"],
        "sign": action["sign"],
        "unet_calls": unet_calls,
        "scheduler_calls": scheduler_calls,
        "extra_unet_calls": 0,
        "extra_scheduler_calls": 0,
        "backbone_backward_calls": 0,
        "intermediate_decode_calls": 0,
        "feature_hook": None,
        "random_edge_counter": None,
        "basis_sha256": None,
        "latent_before_sha256": latent_before_sha256,
        "latent_after_sha256": latent_after_sha256,
        "applied_update_ratio": 0.0,
        "target_ratio_error": 0.0,
        "channel_mean_abs_error": 0.0,
        "channel_variance_relative_error": 0.0,
        "channel_covariance_drift": 0.0,
        "pred_original_sample_relative_l2_error": 0.0,
        "expected_prev_sample_relative_l2_error": 0.0,
        "native_round_trip_max_abs_error": 0.0,
        "solver_target_update_ratio": 0.0,
        "mapped_ratio_solver_evaluations": 0,
        "finite": True,
        "cap_hit": False,
    }


def _png_bytes(image: Any) -> bytes:
    if getattr(image, "size", None) != (contract.WIDTH, contract.HEIGHT):
        raise RuntimeError("pipeline returned an image with non-1024 dimensions")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    payload = buffer.getvalue()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("pipeline image did not encode as PNG")
    return payload


def _assemble_sidecar(
    *,
    task: Mapping[str, Any],
    pipe: Any,
    png_sha256: str,
) -> dict[str, Any]:
    unet_steps = list(pipe._last_unet_calls_per_step)
    scheduler_steps = list(pipe._last_scheduler_calls_per_step)
    if len(unet_steps) != contract.NUM_INFERENCE_STEPS or len(
        scheduler_steps
    ) != contract.NUM_INFERENCE_STEPS:
        raise RuntimeError("pipeline did not emit exactly 50 call-ledger entries")
    if pipe._last_unet_calls_total != sum(unet_steps) or (
        pipe._last_scheduler_calls_total != sum(scheduler_steps)
    ):
        raise RuntimeError("pipeline call totals differ from per-step ledgers")
    if any(value != 1 for value in unet_steps + scheduler_steps):
        raise RuntimeError("pipeline violated the one-U-Net/one-scheduler step contract")
    schedule = _validated_pipeline_schedule_record(pipe)
    initial_latent_sha256 = getattr(
        pipe, "_last_prepared_initial_latent_sha256", None
    )
    latent_before_sha256_by_step = list(
        getattr(pipe, "_last_latents_before_step_sha256", [])
    )
    latent_after_sha256_by_step = list(
        getattr(pipe, "_last_latents_after_step_sha256", [])
    )
    if (
        not isinstance(initial_latent_sha256, str)
        or _SHA256.fullmatch(initial_latent_sha256) is None
    ):
        raise RuntimeError("pipeline did not hash its prepared initial latent")
    if len(latent_before_sha256_by_step) != contract.NUM_INFERENCE_STEPS:
        raise RuntimeError("pipeline did not expose exactly 50 latent-before hashes")
    if len(latent_after_sha256_by_step) != contract.NUM_INFERENCE_STEPS:
        raise RuntimeError("pipeline did not expose exactly 50 latent-after hashes")
    if latent_before_sha256_by_step[0] != initial_latent_sha256:
        raise RuntimeError("pipeline initial latent differs from step zero input")
    if latent_before_sha256_by_step[1:] != latent_after_sha256_by_step[:-1]:
        raise RuntimeError("pipeline directly observed a broken latent chain")
    final_latent_sha256 = latent_after_sha256_by_step[-1]

    action = task["action"]
    if action["physical_no_op"]:
        if pipe._last_latent_renderer_scheduler_mapping is not None:
            raise RuntimeError("physical P0 retained a renderer scheduler mapping")
        if pipe._last_latent_renderer_step_diagnostics:
            raise RuntimeError("physical P0 retained active renderer diagnostics")
        steps = [
            _p0_step_record(
                task,
                index,
                unet_steps[index],
                scheduler_steps[index],
                latent_before_sha256=latent_before_sha256_by_step[index],
                latent_after_sha256=latent_after_sha256_by_step[index],
            )
            for index in range(contract.NUM_INFERENCE_STEPS)
        ]
    else:
        if pipe._last_latent_renderer_scheduler_mapping != "euler_clean_endpoint":
            raise RuntimeError("active trajectory did not use native Euler mapping")
        raw_steps = copy.deepcopy(pipe._last_latent_renderer_step_diagnostics)
        if len(raw_steps) != contract.NUM_INFERENCE_STEPS:
            raise RuntimeError("active trajectory omitted renderer step diagnostics")
        steps = [
            _active_step_record(
                raw,
                task,
                index,
                latent_before_sha256=latent_before_sha256_by_step[index],
                latent_after_sha256=latent_after_sha256_by_step[index],
            )
            for index, raw in enumerate(raw_steps)
        ]

    provenance = task["provenance"]
    return {
        "schema": contract.SIDECAR_SCHEMA,
        "experiment_id": contract.EXPERIMENT_ID,
        "task_id": task["task_id"],
        "prompt": copy.deepcopy(task["prompt"]),
        "action": copy.deepcopy(action),
        "trajectory": {
            "trajectory_id": f"trajectory:{task['task_id']}",
            "initial_latent_sha256": initial_latent_sha256,
            "final_latent_sha256": final_latent_sha256,
            "png_sha256": png_sha256,
            "physical_no_op": action["physical_no_op"],
            "scheduler_class": contract.SCHEDULER_CLASS,
            "scheduler_churn": contract.SCHEDULER_CHURN,
            "scheduler_prediction_type": contract.SCHEDULER_PREDICTION_TYPE,
            "scheduler_config_sha256_v2": contract.SCHEDULER_CONFIG_SHA256_V2,
            "scheduler_schedule_sha256": schedule["schedule_sha256"],
            "scheduler_timesteps": schedule["timesteps"],
            "scheduler_sigmas": schedule["sigmas"],
            "scheduler_construction_init_noise_sigma": (
                schedule["construction_init_noise_sigma"]
            ),
            "scheduler_effective_init_noise_sigma": (
                schedule["effective_init_noise_sigma"]
            ),
            "model_id": contract.MODEL_ID,
            "model_revision": contract.MODEL_REVISION,
            "model_snapshot_manifest_sha256": (
                contract.MODEL_SNAPSHOT_MANIFEST_SHA256
            ),
            "num_inference_steps": contract.NUM_INFERENCE_STEPS,
            "width": contract.WIDTH,
            "height": contract.HEIGHT,
            "cfg_scale": contract.CFG_SCALE,
            "grid_size": contract.GRID_SIZE,
        },
        "call_totals": {
            "unet_calls": pipe._last_unet_calls_total,
            "scheduler_calls": pipe._last_scheduler_calls_total,
            "extra_unet_calls": 0,
            "extra_scheduler_calls": 0,
            "backbone_backward_calls": 0,
            "intermediate_decode_calls": pipe._last_intermediate_decode_calls,
            "final_decode_calls": pipe._last_final_decode_calls,
            "attention_probability_reads": 0,
            "qk_reads": 0,
        },
        "step_ledger": steps,
        "evidence_scope": {
            "schema": contract.EVIDENCE_SCOPE_SCHEMA,
            "split_role": contract.SPLIT_ROLE,
            "generation_only": True,
            "scoring_authorized": False,
            "quality_outcomes_present": False,
            "prompt_csv_sha256": provenance["prompt_csv_sha256"],
            "prompt_manifest_canonical_sha256": provenance[
                "prompt_manifest_canonical_sha256"
            ],
            "prompt_rows_sha256": provenance["prompt_rows_sha256"],
            "contract_sha256": contract.CONTRACT_SHA256,
            "task_sha256": task["task_sha256"],
        },
    }


def _execute_task(
    *,
    task: Mapping[str, Any],
    prompt_text: str,
    raw_noise: Any,
    pipe: Any,
    base_scheduler_config: Mapping[str, Any],
    runtime: SimpleNamespace,
    baseline_hooks: tuple[tuple[int, int], ...],
) -> tuple[dict[str, Any], bytes]:
    _reset_pipeline_ledgers(pipe)
    if _hook_signature(pipe.unet) != baseline_hooks:
        raise RuntimeError("U-Net hook state leaked into the next task")
    scheduler = _fresh_scheduler(runtime, base_scheduler_config)
    pipe.scheduler = scheduler
    provider = renderer = None
    final_latent = None
    callback_indices = []
    latent_after_hashes = []
    raw_hash_before = tensor_sha256(raw_noise)
    if not bool(runtime.torch.isfinite(raw_noise.detach()).all().item()):
        raise RuntimeError("task-paired raw initial noise is non-finite")

    def capture_final(step_index, _timestep, latents):
        nonlocal final_latent
        callback_indices.append(int(step_index))
        latent_after_hashes.append(tensor_sha256(latents))
        if int(step_index) == contract.NUM_INFERENCE_STEPS - 1:
            if final_latent is not None:
                raise RuntimeError("final latent callback ran more than once")
            final_latent = latents.detach().clone()

    try:
        provider, renderer = _action_runtime(task, pipe, runtime)
        if task["action"]["physical_no_op"] and (
            provider is not None or renderer is not None
        ):
            raise RuntimeError("physical P0 instantiated an active runtime")
        task_generator = runtime.torch.Generator(
            device=runtime.torch.device(str(raw_noise.device))
        ).manual_seed(task["prompt"]["seed"])
        images = pipe(
            prompt=prompt_text,
            height=contract.HEIGHT,
            width=contract.WIDTH,
            num_inference_steps=contract.NUM_INFERENCE_STEPS,
            guidance_scale=contract.CFG_SCALE,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_images_per_prompt=1,
            eta=0.0,
            generator=task_generator,
            latents=raw_noise.clone(),
            output_type="pil",
            callback=capture_final,
            callback_steps=1,
            record_latent_audit=True,
            guidance_rescale=0.0,
            image_lr=None,
            multi_decoder=False,
            show_image=False,
            lowvram=False,
            models_to_cpu=False,
            multi_encoder=False,
            num_resample_timesteps=50,
            init_rates=[0.8],
            attn_guidance_scale=0.0,
            attn_guidance_controller=None,
            power_calibrate=0,
            semantic_transport_config=None,
            latent_renderer=renderer,
            latent_renderer_basis_provider=provider,
            latent_renderer_scheduler_mapping=(
                "legacy_unit" if renderer is None else "euler_clean_endpoint"
            ),
            freeu_schedule=None,
            trajectory_correction=None,
        )
        if not isinstance(images, list) or len(images) != 1:
            raise RuntimeError("pipeline must return exactly one final image")
        if callback_indices != list(range(contract.NUM_INFERENCE_STEPS)):
            raise RuntimeError("pipeline callback did not expose all 50 ordered steps")
        if final_latent is None:
            raise RuntimeError("pipeline did not expose the final latent")
        if len(latent_after_hashes) != contract.NUM_INFERENCE_STEPS:
            raise RuntimeError("pipeline did not expose all latent transitions")
        if not bool(runtime.torch.isfinite(final_latent.detach()).all().item()):
            raise RuntimeError("pipeline returned a non-finite final latent")
        final_latent_hash = tensor_sha256(final_latent)
        if latent_after_hashes[-1] != final_latent_hash:
            raise RuntimeError("final callback latent differs from the last step hash")
        pipeline_after_hashes = list(pipe._last_latents_after_step_sha256)
        if latent_after_hashes != pipeline_after_hashes:
            raise RuntimeError(
                "callback latent hashes differ from direct post-step observations"
            )
        if tensor_sha256(raw_noise) != raw_hash_before:
            raise RuntimeError("pipeline mutated the task-paired raw initial noise")
        if _hook_signature(pipe.unet) != baseline_hooks:
            raise RuntimeError("adaptive-oracle hook leaked after pipeline invocation")
        png = _png_bytes(images[0])
        record = _assemble_sidecar(
            task=task,
            pipe=pipe,
            png_sha256=sha256_bytes(png),
        )
        if record["trajectory"]["final_latent_sha256"] != final_latent_hash:
            raise RuntimeError("sidecar final latent differs from pipeline return")
        contract.validate_sidecar(record, task)
        return record, png
    finally:
        if provider is not None:
            provider.last_diagnostics = None
        provider = None
        renderer = None
        pipe.scheduler = None
        scheduler = None
        final_latent = None
        _reset_pipeline_ledgers(pipe)


def _raw_initial_noise(runtime: SimpleNamespace, device: str, seed: int):
    generator = runtime.torch.Generator(device=device).manual_seed(int(seed))
    noise = runtime.torch.randn(
        (1, 4, contract.HEIGHT // 8, contract.WIDTH // 8),
        generator=generator,
        device=device,
        dtype=runtime.torch.float16,
    )
    expected_shape = (1, 4, contract.HEIGHT // 8, contract.WIDTH // 8)
    if tuple(noise.shape) != expected_shape or noise.dtype != runtime.torch.float16:
        raise RuntimeError("raw initial noise shape or dtype differs from registration")
    if str(noise.device) != device:
        raise RuntimeError("raw initial noise was not created on the explicit CUDA device")
    if not bool(runtime.torch.isfinite(noise).all().item()):
        raise RuntimeError("raw initial noise is non-finite")
    return noise


def _attempt_record(validated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": ATTEMPT_SCHEMA,
        "status": "started_one_shot_no_resume",
        "one_shot": True,
        "experiment_id": contract.EXPERIMENT_ID,
        "split_role": contract.SPLIT_ROLE,
        "authorization_sha256": validated["authorization_sha256"],
        "authorization_binding": copy.deepcopy(
            validated["authorization_binding"]
        ),
        "launcher_evidence": copy.deepcopy(validated["launcher_evidence"]),
        "reviewed_commit": validated["reviewed_commit"],
        "design_sha256": validated["design"]["design_sha256"],
        "tasks_sha256": validated["design"]["tasks_sha256"],
        "cuda_device": validated["device"],
        "output_dir": OUTPUT_DIR,
    }


def _failure_record(
    validated: Mapping[str, Any],
    exc: BaseException,
    completed_records: int,
    runtime_evidence: Optional[Mapping[str, Any]],
    model_stage: Optional[Mapping[str, Any]],
    model_stage_verifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trace = traceback.format_exc().encode("utf-8", errors="replace")
    return {
        "schema": FAILURE_SCHEMA,
        "status": "failed_terminal_no_resume_or_retry",
        "experiment_id": contract.EXPERIMENT_ID,
        "authorization_sha256": validated["authorization_sha256"],
        "authorization_binding": copy.deepcopy(
            validated["authorization_binding"]
        ),
        "launcher_evidence": copy.deepcopy(validated["launcher_evidence"]),
        "design_sha256": validated["design"]["design_sha256"],
        "completed_records": int(completed_records),
        "expected_records": contract.TOTAL_TASK_COUNT,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback_sha256": sha256_bytes(trace),
        "runtime_evidence": (
            None if runtime_evidence is None else copy.deepcopy(runtime_evidence)
        ),
        "model_stage": None if model_stage is None else copy.deepcopy(model_stage),
        "model_stage_verifications": copy.deepcopy(
            list(model_stage_verifications)
        ),
    }


def _model_stage_evidence(
    model_stage: Mapping[str, Any],
    verifications: Sequence[Mapping[str, Any]],
    *,
    cleanup: Optional[Mapping[str, Any]],
    cleanup_error: Optional[BaseException] = None,
) -> dict[str, Any]:
    if (cleanup is None) == (cleanup_error is None):
        raise ValueError("model-stage evidence requires exactly one cleanup outcome")
    record = {
        "schema": MODEL_STAGE_EVIDENCE_SCHEMA,
        "status": "removed" if cleanup_error is None else "cleanup_failed_terminal",
        "trust_boundary": {
            "same_uid_noninterference_required_between_verification_and_open": False,
            "loader_root_pinned_by_procfs_fd": True,
            "pre_post_content_and_object_identity_binding": True,
            "network_access_used": False,
            "load_source": "pinned_procfs_fd_private_regular_file_stage",
        },
        "stage": copy.deepcopy(model_stage),
        "verifications": copy.deepcopy(list(verifications)),
        "cleanup": None if cleanup is None else copy.deepcopy(cleanup),
        "cleanup_failure": None,
    }
    if cleanup_error is not None:
        record["cleanup_failure"] = {
            "exception_type": type(cleanup_error).__name__,
            "exception_message": str(cleanup_error),
        }
    return record


def _run_authorized_engineering_generation(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the authenticated 165-task generation-only matrix exactly once."""

    output = Path(validated["output_dir"])
    if os.path.lexists(output):
        raise FileExistsError(
            "adaptive-oracle engineering output already exists; resume/retry is forbidden"
        )
    output_parent = output.parent
    if not output_parent.is_dir() or output_parent.is_symlink():
        raise ValueError("authorized output parent must already be a regular directory")

    completed = 0
    owned_output = False
    runtime_capture: Optional[_RuntimeEvidenceCapture] = None
    runtime_evidence: Optional[dict[str, Any]] = None
    runtime: Optional[SimpleNamespace] = None
    runtime_cleanup_attempted = False
    model_stage: Optional[Mapping[str, Any]] = None
    model_stage_verifications: list[dict[str, Any]] = []
    model_stage_cleanup: Optional[Mapping[str, Any]] = None
    model_stage_cleanup_error: Optional[BaseException] = None
    model_stage_cleanup_attempted = False
    try:
        try:
            os.mkdir(output, mode=0o750)
        except FileExistsError as exc:
            raise FileExistsError(
                "adaptive-oracle engineering output already exists; "
                "resume/retry is forbidden"
            ) from exc
        owned_output = True
        records_dir = _initialize_attempt_directory(
            output, _attempt_record(validated)
        )
        runtime_capture = _RuntimeEvidenceCapture()
        runtime_capture.__enter__()
        runtime = _load_verified_runtime_components(validated)
        _revalidate_immutable_authorized_bundle(
            validated, runtime, phase="after_runtime_import"
        )
        runtime_environment = runtime.validate_environment_lock_bytes(
            validated["environment_lock_bytes"],
            source_path=validated["source_paths"]["environment_lock"],
            expected_sha256=validated["source_hashes"]["environment_lock"][
                "sha256"
            ],
            cuda_device=validated["device"],
            require_unmasked_cuda=True,
        )
        if not isinstance(runtime_environment, Mapping):
            raise RuntimeError("environment validator returned a non-mapping record")
        _validate_cuda_device(runtime, validated["device"])
        try:
            model_stage = model_snapshot.stage_model_snapshot(
                Path(validated["repo_root"]),
                expected_manifest_sha256=validated["config"]["model"][
                    "snapshot_manifest_sha256"
                ],
            )
        except model_snapshot.ModelStageCreationError as stage_exc:
            model_stage = stage_exc.cleanup_record
            raise
        pipe, base_scheduler_config, _ = _load_pipeline(
            validated,
            runtime,
            model_stage,
            verification_log=model_stage_verifications,
        )
        baseline_hooks = _hook_signature(pipe.unet)
        design = validated["design"]
        rows_by_id = {
            row["prompt_row_id"]: row for row in validated["prompt_rows"]
        }
        summaries = []
        block_evidence = []

        model_stage_verifications.append(
            model_snapshot.verify_staged_model_snapshot(
                model_stage,
                expected_manifest_sha256=validated["config"]["model"][
                    "snapshot_manifest_sha256"
                ],
            )
        )

        for prompt_index in range(contract.PROMPT_COUNT):
            start = prompt_index * contract.TASKS_PER_PROMPT
            tasks = design["tasks"][start : start + contract.TASKS_PER_PROMPT]
            prompt_row_id = tasks[0]["prompt"]["prompt_row_id"]
            if any(
                task["prompt"]["prompt_row_id"] != prompt_row_id for task in tasks
            ):
                raise RuntimeError("design prompt block is not contiguous")
            prompt_row = rows_by_id.get(prompt_row_id)
            if prompt_row is None:
                raise RuntimeError("design prompt identity is absent from prompt CSV")
            raw_noise = _raw_initial_noise(
                runtime, validated["device"], tasks[0]["prompt"]["seed"]
            )
            block_records = []
            for task in tasks:
                record, png = _execute_task(
                    task=task,
                    prompt_text=prompt_row["TEXT"],
                    raw_noise=raw_noise,
                    pipe=pipe,
                    base_scheduler_config=base_scheduler_config,
                    runtime=runtime,
                    baseline_hooks=baseline_hooks,
                )
                contract.validate_sidecar(record, task)
                png_path = records_dir / f"{task['task_id']}.png"
                json_path = records_dir / f"{task['task_id']}.json"
                atomic_create_bytes(png_path, png)
                atomic_create_json(json_path, record)
                summary = contract.validate_sidecar(record, task)
                summaries.append(
                    {
                        **summary,
                        "png_path": str(png_path.relative_to(output)),
                        "sidecar_path": str(json_path.relative_to(output)),
                    }
                )
                block_records.append(record)
                completed += 1
            block_evidence.append(contract.validate_prompt_block(block_records, tasks))
            raw_noise = None

        _revalidate_immutable_authorized_bundle(
            validated, runtime, phase="after_generation"
        )
        runtime_cleanup_attempted = True
        _cleanup_runtime_resources(runtime)
        runtime_capture.__exit__(None, None, None)
        runtime_evidence = runtime_capture.record()
        runtime_capture = None
        _require_warning_free_runtime_evidence(runtime_evidence)

        if completed != contract.TOTAL_TASK_COUNT or len(summaries) != completed:
            raise RuntimeError("engineering run did not complete exactly 165 records")
        runtime_evidence_sha256 = contract.canonical_sha256(runtime_evidence)
        manifest = {
            "schema": RUN_MANIFEST_SCHEMA,
            "status": "complete_generation_only_unscored",
            "experiment_id": contract.EXPERIMENT_ID,
            "record_count": completed,
            "design_sha256": design["design_sha256"],
            "tasks_sha256": design["tasks_sha256"],
            "contract_sha256": contract.CONTRACT_SHA256,
            "runtime_evidence_sha256": runtime_evidence_sha256,
            "block_evidence": block_evidence,
            "records": summaries,
            "evidence_scope": dict(_EVIDENCE_SCOPE),
        }
        model_stage_cleanup_attempted = True
        try:
            model_stage_cleanup = model_snapshot.cleanup_staged_model_snapshot(
                model_stage
            )
        except BaseException as cleanup_exc:
            model_stage_cleanup_error = cleanup_exc
            raise
        model_stage_evidence = _model_stage_evidence(
            model_stage,
            model_stage_verifications,
            cleanup=model_stage_cleanup,
        )
        model_stage_evidence_sha256 = contract.canonical_sha256(
            model_stage_evidence
        )
        model_stage_evidence_file_sha256 = sha256_bytes(
            contract.canonical_json_bytes(model_stage_evidence) + b"\n"
        )
        run_config = {
            "schema": RUN_CONFIG_SCHEMA,
            "authorization_sha256": validated["authorization_sha256"],
            "authorization_binding": copy.deepcopy(
                validated["authorization_binding"]
            ),
            "launcher_evidence": copy.deepcopy(validated["launcher_evidence"]),
            "reviewed_commit": validated["reviewed_commit"],
            "authorization": validated["config"],
            "runtime_environment": copy.deepcopy(dict(runtime_environment)),
            "verified_source_execution": copy.deepcopy(
                dict(runtime.verified_source_execution)
            ),
            "design_sha256": design["design_sha256"],
            "contract_sha256": contract.CONTRACT_SHA256,
            "runtime_evidence_sha256": runtime_evidence_sha256,
            "model_stage_evidence_sha256": model_stage_evidence_sha256,
            "model_stage_evidence_file_sha256": (
                model_stage_evidence_file_sha256
            ),
        }
        manifest_sha256 = contract.canonical_sha256(manifest)
        run_config_sha256 = contract.canonical_sha256(run_config)
        success = {
            "schema": SUCCESS_SCHEMA,
            "status": "complete_generation_only_unscored",
            "one_shot": True,
            "experiment_id": contract.EXPERIMENT_ID,
            "record_count": completed,
            "authorization_binding": copy.deepcopy(
                validated["authorization_binding"]
            ),
            "launcher_evidence": copy.deepcopy(validated["launcher_evidence"]),
            "manifest_sha256": manifest_sha256,
            "run_config_sha256": run_config_sha256,
            "design_sha256": design["design_sha256"],
            "contract_sha256": contract.CONTRACT_SHA256,
            "runtime_evidence_sha256": runtime_evidence_sha256,
            "runtime_evidence_file_sha256": sha256_bytes(
                contract.canonical_json_bytes(runtime_evidence) + b"\n"
            ),
            "model_stage_evidence_sha256": model_stage_evidence_sha256,
            "model_stage_evidence_file_sha256": (
                model_stage_evidence_file_sha256
            ),
            "scoring_authorized": False,
            "quality_inspection_authorized": False,
        }
        atomic_create_json(output / "runtime_evidence.json", runtime_evidence)
        atomic_create_json(output / "manifest.json", manifest)
        atomic_create_json(
            output / "model_stage_evidence.json", model_stage_evidence
        )
        atomic_create_json(output / "config.json", run_config)
        atomic_create_json(output / "success.json", success)
        return success
    except BaseException as exc:
        original_exc_info = (type(exc), exc, exc.__traceback__)
        secondary_errors: list[tuple[str, BaseException]] = []
        runtime_cleanup_error: Optional[BaseException] = None
        if runtime_capture is not None:
            try:
                runtime_capture.__exit__(*original_exc_info)
                runtime_evidence = runtime_capture.record()
            except BaseException as capture_exc:
                secondary_errors.append(
                    ("runtime evidence finalization also failed", capture_exc)
                )
            finally:
                runtime_capture = None
        if owned_output:
            failure_path = output / "failure.json"
            if not os.path.lexists(failure_path):
                try:
                    atomic_create_json(
                        failure_path,
                        _failure_record(
                            validated,
                            exc,
                            completed,
                            runtime_evidence,
                            model_stage,
                            model_stage_verifications,
                        ),
                    )
                except BaseException as receipt_exc:
                    secondary_errors.append(
                        ("failure receipt creation also failed", receipt_exc)
                    )

        if not runtime_cleanup_attempted:
            runtime_cleanup_attempted = True
            try:
                _cleanup_runtime_resources(runtime)
            except BaseException as cleanup_exc:
                runtime_cleanup_error = cleanup_exc
                secondary_errors.append(
                    ("runtime cleanup also failed", cleanup_exc)
                )

        if model_stage is not None and not model_stage_cleanup_attempted:
            model_stage_cleanup_attempted = True
            try:
                model_stage_cleanup = model_snapshot.cleanup_staged_model_snapshot(
                    model_stage
                )
            except BaseException as cleanup_exc:
                model_stage_cleanup_error = cleanup_exc
        if model_stage_cleanup_error is not None:
            secondary_errors.append(
                ("model-stage cleanup also failed", model_stage_cleanup_error)
            )
        if model_stage is not None and owned_output:
            model_stage_evidence_path = output / "model_stage_evidence.json"
            if not os.path.lexists(model_stage_evidence_path):
                try:
                    evidence = _model_stage_evidence(
                        model_stage,
                        model_stage_verifications,
                        cleanup=model_stage_cleanup,
                        cleanup_error=model_stage_cleanup_error,
                    )
                    atomic_create_json(model_stage_evidence_path, evidence)
                except BaseException as evidence_exc:
                    secondary_errors.append(
                        ("model-stage evidence creation also failed", evidence_exc)
                    )

        if secondary_errors:
            detail = "; ".join(label for label, _ in secondary_errors)
            finalization_error = RuntimeError(
                f"engineering run failed and {detail}"
            )
            finalization_error.original_error = exc
            finalization_error.secondary_errors = tuple(secondary_errors)
            for label, secondary in secondary_errors:
                finalization_error.add_note(
                    f"{label}: {type(secondary).__name__}: {secondary}"
                )
            raise finalization_error from exc
        raise


def _run_from_verified_launcher(
    argv: Sequence[str], *, launcher_context: Any
) -> int:
    """Consume one authenticated launcher context and run generation."""

    parser = argparse.ArgumentParser(
        prog="generate_adaptive_oracle_engineering",
        description="Run the authorized one-shot adaptive-oracle engineering smoke",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--device", required=True, help="one registered device, cuda:1 through cuda:4"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv))
    launcher_evidence = launcher_context.claim()
    input_bytes = {
        name: launcher_context.read_input(name)
        for name in (
            "registration",
            "cpu_audit",
            "prompt_csv",
            "prompt_manifest",
            "exclusion_inventory",
            "environment_lock",
        )
    }
    validated = _validate_launcher_bundle(
        launcher_evidence,
        input_bytes,
        requested_device=args.device,
        requested_output_dir=args.output_dir,
    )
    success = _run_authorized_engineering_generation(validated)
    print(json.dumps(success, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    raise RuntimeError(
        "engineering generation must be invoked by the verified execution-root launcher"
    )


if __name__ == "__main__":
    raise SystemExit(main())
