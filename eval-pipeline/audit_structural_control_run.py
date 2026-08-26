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
import secrets
import stat
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

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
STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT = (
    "e0b323fab88c0cc9e24256d7bf7d466a049d2675"
)
STRUCTURAL_CONTROL_ACTIONS_PATH = (
    "eval-pipeline/configs/"
    "scheduler_native_structural_controls_development_authorized_v1.yaml"
)
STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_SCHEMA = (
    "structural_control_analysis_amendment_v1"
)
STRUCTURAL_CONTROL_AMENDMENT_STATUS = "authorized_pre_score"
STRUCTURAL_CONTROL_AMENDMENT_CANDIDATE_STATUS = (
    "blocked_pending_independent_review"
)
STRUCTURAL_CONTROL_REVIEWER_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._@+-]{0,127}"
)
STRUCTURAL_CONTROL_UNCHANGED_INVARIANTS = frozenset(
    {
        "generation",
        "scorer_recipe",
        "metrics",
        "contrasts",
        "multiplicity",
        "decision_rules",
    }
)
STRUCTURAL_CONTROL_UNAUTHORIZED_USES = frozenset(
    {"method_selection", "validation", "publication_claims", "rl"}
)
STRUCTURAL_CONTROL_AUDITOR_SCOPE = "formal_development_only"
STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "reviewer",
        "reviewed_commit",
        "base_commit",
        "result_access_before_authorization",
        "scoring_started_before_authorization",
        "unchanged_invariants",
        "authorizations",
        "replacements",
    }
)
STRUCTURAL_CONTROL_PRE_SCORE_SEAL_SCHEMA = "structural_control_pre_score_seal_v1"
STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME = "structural_control_pre_score_seal.json"
STRUCTURAL_CONTROL_SCORING_ATTEMPT_SCHEMA = "structural_control_scoring_attempt_v1"
STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME = "structural_control_scoring_attempt.json"
STRUCTURAL_CONTROL_SCORING_SUCCESS_SCHEMA = "structural_control_scoring_success_v1"
STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME = "structural_control_scoring_success.json"
STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME = "run_audit.json"
STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME = "structural_control_audit_attempt.json"
STRUCTURAL_CONTROL_AUDIT_ATTEMPT_SCHEMA = "structural_control_audit_attempt_v1"
STRUCTURAL_CONTROL_EVALUATION_ATTEMPT_NAME = (
    "structural_control_evaluation_attempt.json"
)
STRUCTURAL_CONTROL_EXPECTED_TASKS = 792
STRUCTURAL_CONTROL_FORMAL_RUN_PATH = "outputs/structural_controls/development_v1"
STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH = (
    "eval-pipeline/configs/"
    "scheduler_native_structural_controls_analysis_amendment_v1.yaml"
)
STRUCTURAL_CONTROL_SCORE_SCRIPT_PATH = "eval-pipeline/score.py"
STRUCTURAL_CONTROL_SCORE_SCRIPT_SHA256 = (
    "849201edcf21d6c4f2828699b2b79cd87efb83f79f3ff2722822d5088ebf86af"
)
STRUCTURAL_CONTROL_SCORE_CONFIG_PATH = "eval-pipeline/configs/eval_common.yaml"
STRUCTURAL_CONTROL_SCORE_CONFIG_SHA256 = (
    "4901d660383f30a4e65aaceb9180eecf72aea951c3ea0abb8dd845cb65fbe932"
)
STRUCTURAL_CONTROL_SCORER_PROVENANCE_SHA256 = (
    "9c6858b1825c599a60be33282d99ffd4a8f75ac0ecf0fc8071bb97ad16eb61b9"
)
STRUCTURAL_CONTROL_SCORE_METRICS = (
    "imagereward",
    "pixel",
    "clip",
    "hps",
    "aesthetic",
    "iqa",
)
STRUCTURAL_CONTROL_SCORE_OUTPUT_KEYS = frozenset(
    {
        "imagereward",
        "patch_ir_mean",
        "patch_ir_std",
        "patch_ir_n",
        "colorfulness",
        "laplacian_sharpness",
        "mean_saturation",
        "clipped_fraction",
        "contrast_std",
        "clip_cosine",
        "clipscore",
        "hpsv2",
        "aesthetic",
        "topiq_nr",
    }
)
STRUCTURAL_CONTROL_SCORE_DEVICE = "cuda:7"
STRUCTURAL_CONTROL_SCORING_CAPABILITY_ENV = (
    "REPLDM_STRUCTURAL_SCORING_CAPABILITY"
)
STRUCTURAL_CONTROL_SCORING_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
}
STRUCTURAL_CONTROL_ANALYSIS_REPLACEMENTS = frozenset(
    {
        "eval-pipeline/audit_structural_control_run.py",
        "eval-pipeline/evaluate_structural_control_run.py",
        "eval-pipeline/scorer_provenance.py",
    }
)
STRUCTURAL_CONTROL_EVALUATION_BUNDLE = "structural_control_evaluation_bundle"
STRUCTURAL_CONTROL_LEGACY_EVALUATION_OUTPUTS = (
    "structural_control_evaluation.json",
    "structural_control_contrasts.csv",
)
STRUCTURAL_CONTROL_PRE_SCORE_FORBIDDEN_NAMES = frozenset(
    {
        "scores.jsonl",
        STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME,
        STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME,
        STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME,
        STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME,
        STRUCTURAL_CONTROL_EVALUATION_BUNDLE,
        STRUCTURAL_CONTROL_EVALUATION_ATTEMPT_NAME,
        *STRUCTURAL_CONTROL_LEGACY_EVALUATION_OUTPUTS,
    }
)
STRUCTURAL_CONTROL_PRE_SCORE_FORBIDDEN_PATTERNS = (
    re.compile(r"\.scores\.jsonl\..+\.tmp"),
    re.compile(
        r"\.structural_control_scoring_attempt\.json\."
        r".+\.tmp"
    ),
    re.compile(
        r"\.structural_control_scoring_success\.json\."
        r".+\.tmp"
    ),
    re.compile(r"\.run_audit\.json\..+\.tmp"),
    re.compile(r"\.structural_control_evaluation_bundle\..+"),
    re.compile(
        r"\.structural_control_evaluation\.json\..+\.tmp"
    ),
    re.compile(
        r"\.structural_control_contrasts\.csv\..+\.tmp"
    ),
    re.compile(
        r"\.structural_control_pre_score_seal\.json\."
        r".+\.tmp"
    ),
)
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _audit_attempt_marker_payload() -> bytes:
    return (
        json.dumps(
            {
                "schema": STRUCTURAL_CONTROL_AUDIT_ATTEMPT_SCHEMA,
                "scope": STRUCTURAL_CONTROL_AUDITOR_SCOPE,
                "one_shot": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _audit_attempt_artifacts(run_root: Path, output_path: Path) -> tuple[Path, ...]:
    return (
        run_root / STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME,
        run_root / STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME,
        output_path,
    )


def _require_canonical_audit_output(
    run_root: Path, output_path: str | os.PathLike[str]
) -> Path:
    raw_output = Path(output_path)
    output = raw_output.absolute()
    canonical_output = run_root / STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME
    if raw_output.is_symlink() or output != canonical_output:
        raise ValueError("structural audit requires the canonical output path")
    return output


def create_audit_attempt_marker(
    run_dir: str | os.PathLike[str], output_path: str | os.PathLike[str]
) -> Path:
    """Durably consume the formal auditor's one allowed result-reading attempt."""
    run_root = _require_canonical_formal_run(
        run_dir, label="structural audit attempt"
    )
    output = _require_canonical_audit_output(run_root, output_path)
    marker_path = run_root / STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME
    if any(
        os.path.lexists(path) for path in _audit_attempt_artifacts(run_root, output)
    ):
        raise ValueError("structural audit is one-shot; attempt or output exists")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(marker_path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError("structural audit is one-shot; attempt or output exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(_audit_attempt_marker_payload())
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(run_root)
    return marker_path


def require_audit_attempt_marker(run_dir: str | os.PathLike[str]) -> Path:
    """Validate the durable canonical marker without consuming another attempt."""
    run_root = Path(run_dir).resolve()
    marker_path = run_root / STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME
    descriptor = -1
    try:
        descriptor = os.open(
            marker_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("canonical structural audit attempt marker is not regular")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
    except OSError as exc:
        raise ValueError("canonical structural audit attempt marker is missing") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if payload != _audit_attempt_marker_payload():
        raise ValueError("canonical structural audit attempt marker is invalid")
    return marker_path


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    """Read one regular file without following links or blocking on a FIFO."""
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} is not regular")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _linux_process_start_ticks(pid: int) -> int:
    """Return Linux /proc field 22 without trusting the process name layout."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("scoring launcher PID is invalid")
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = payload[payload.rfind(") ") + 2 :].split()
        ticks = int(fields[19])
    except (OSError, UnicodeError, IndexError, ValueError) as exc:
        raise ValueError("cannot verify scoring launcher process start time") from exc
    if ticks <= 0:
        raise ValueError("scoring launcher process start time is invalid")
    return ticks


def _linux_boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError("cannot verify Linux boot identity for scoring") from exc
    if not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        value,
    ):
        raise ValueError("Linux boot identity for scoring is invalid")
    return value


def _canonical_scoring_argv(run_root: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str((ROOT / STRUCTURAL_CONTROL_SCORE_SCRIPT_PATH).resolve()),
        "--run_dir",
        str(run_root.resolve()),
        "--config",
        str((ROOT / STRUCTURAL_CONTROL_SCORE_CONFIG_PATH).resolve()),
        "--device",
        STRUCTURAL_CONTROL_SCORE_DEVICE,
        "--strict",
        "--require-scorer-provenance",
    ]


def _scoring_command_binding(run_root: Path) -> dict[str, Any]:
    return {
        "argv": _canonical_scoring_argv(run_root),
        "cwd": str(ROOT.resolve()),
        "environment": dict(STRUCTURAL_CONTROL_SCORING_ENVIRONMENT),
    }


def _scoring_attempt_marker_payload(
    run_dir: str | os.PathLike[str],
    analysis_amendment_path: str | os.PathLike[str],
    pre_score_seal_path: str | os.PathLike[str],
    *,
    base_commit: str = STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
    analysis_commit: str | None = None,
    launcher_pid: int | None = None,
    launcher_start_ticks: int | None = None,
    boot_id: str | None = None,
    capability_sha256: str | None = None,
) -> bytes:
    """Build canonical marker bytes; optional identities support CPU fixtures."""
    run_root = Path(run_dir).resolve()
    amendment_path = Path(analysis_amendment_path).resolve()
    seal_path = Path(pre_score_seal_path).resolve()
    if analysis_commit is None:
        analysis_commit = _current_head_commit()
    if launcher_pid is None:
        launcher_pid = os.getpid()
    if launcher_start_ticks is None:
        launcher_start_ticks = _linux_process_start_ticks(launcher_pid)
    if boot_id is None:
        boot_id = _linux_boot_id()
    if capability_sha256 is None:
        capability_sha256 = _sha256_bytes(
            b"structural-control-scoring-fixture-capability"
        )
    payload = {
        "schema": STRUCTURAL_CONTROL_SCORING_ATTEMPT_SCHEMA,
        "scope": STRUCTURAL_CONTROL_AUDITOR_SCOPE,
        "one_shot": True,
        "base_commit": base_commit,
        "analysis_commit": analysis_commit,
        "expected_score_rows": STRUCTURAL_CONTROL_EXPECTED_TASKS,
        "command": _scoring_command_binding(run_root),
        "launcher": {
            "pid": launcher_pid,
            "proc_start_ticks": launcher_start_ticks,
            "boot_id": boot_id,
            "capability_sha256": capability_sha256,
        },
        "bindings": {
            "analysis_amendment": _seal_binding(
                STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH,
                base_audit.sha256_file(amendment_path),
            ),
            "pre_score_seal": _seal_binding(
                STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME,
                base_audit.sha256_file(seal_path),
            ),
            "score_script": _seal_binding(
                STRUCTURAL_CONTROL_SCORE_SCRIPT_PATH,
                STRUCTURAL_CONTROL_SCORE_SCRIPT_SHA256,
            ),
            "score_config": _seal_binding(
                STRUCTURAL_CONTROL_SCORE_CONFIG_PATH,
                STRUCTURAL_CONTROL_SCORE_CONFIG_SHA256,
            ),
        },
    }
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _scoring_success_receipt_payload(
    run_dir: str | os.PathLike[str],
    analysis_amendment_path: str | os.PathLike[str],
    pre_score_seal_path: str | os.PathLike[str],
    *,
    base_commit: str = STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
    analysis_commit: str | None = None,
    child_pid: int | None = None,
    scorer_provenance_sha256: str = STRUCTURAL_CONTROL_SCORER_PROVENANCE_SHA256,
) -> bytes:
    """Build the deterministic receipt that proves sealed scoring completed."""
    run_root = Path(run_dir).resolve()
    amendment_path = Path(analysis_amendment_path).resolve()
    seal_path = Path(pre_score_seal_path).resolve()
    if analysis_commit is None:
        analysis_commit = _current_head_commit()
    if child_pid is None:
        child_pid = os.getpid()
    payload = {
        "schema": STRUCTURAL_CONTROL_SCORING_SUCCESS_SCHEMA,
        "scope": STRUCTURAL_CONTROL_AUDITOR_SCOPE,
        "base_commit": base_commit,
        "analysis_commit": analysis_commit,
        "score_rows": STRUCTURAL_CONTROL_EXPECTED_TASKS,
        "child": {"pid": child_pid, "returncode": 0},
        "scorer_provenance_sha256": scorer_provenance_sha256,
        "command": _scoring_command_binding(run_root),
        "bindings": {
            "scoring_attempt": _seal_binding(
                STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME,
                base_audit.sha256_file(
                    run_root / STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
                ),
            ),
            "scores": _seal_binding(
                "scores.jsonl", base_audit.sha256_file(run_root / "scores.jsonl")
            ),
            "manifest": _seal_binding(
                "manifest.jsonl",
                base_audit.sha256_file(run_root / "manifest.jsonl"),
            ),
            "analysis_amendment": _seal_binding(
                STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH,
                base_audit.sha256_file(amendment_path),
            ),
            "pre_score_seal": _seal_binding(
                STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME,
                base_audit.sha256_file(seal_path),
            ),
            "score_script": _seal_binding(
                STRUCTURAL_CONTROL_SCORE_SCRIPT_PATH,
                STRUCTURAL_CONTROL_SCORE_SCRIPT_SHA256,
            ),
            "score_config": _seal_binding(
                STRUCTURAL_CONTROL_SCORE_CONFIG_PATH,
                STRUCTURAL_CONTROL_SCORE_CONFIG_SHA256,
            ),
        },
    }
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _pre_score_forbidden_artifact_exists(run_root: Path) -> bool:
    for entry in run_root.iterdir():
        if entry.name in STRUCTURAL_CONTROL_PRE_SCORE_FORBIDDEN_NAMES or any(
            pattern.fullmatch(entry.name)
            for pattern in STRUCTURAL_CONTROL_PRE_SCORE_FORBIDDEN_PATTERNS
        ):
            return True
    return False


def _require_exact_fields(
    value: Any, *, label: str, fields: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return value


def _repository_relative(path: Path, *, label: str) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the repository") from exc


def _require_canonical_formal_run(
    run_dir: str | os.PathLike[str], *, label: str
) -> Path:
    raw_run = Path(run_dir)
    lexical_run = raw_run.absolute()
    run_root = raw_run.resolve()
    canonical_run = (ROOT / STRUCTURAL_CONTROL_FORMAL_RUN_PATH).resolve()
    if raw_run.is_symlink() or lexical_run != run_root or run_root != canonical_run:
        raise ValueError(f"{label} requires the canonical non-symlink formal run")
    return run_root


def _require_canonical_regular_input(
    path: str | os.PathLike[str],
    expected_path: Path,
    *,
    label: str,
) -> Path:
    """Validate path identity and file type without reading input bytes."""
    raw_path = Path(path)
    lexical_path = raw_path.absolute()
    resolved_path = raw_path.resolve()
    canonical_path = expected_path.resolve()
    try:
        mode = os.lstat(raw_path).st_mode
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc
    if (
        raw_path.is_symlink()
        or lexical_path != resolved_path
        or resolved_path != canonical_path
        or not stat.S_ISREG(mode)
    ):
        raise ValueError(f"{label} requires the canonical non-symlink regular file")
    return resolved_path


def _preflight_formal_audit_paths(
    run_dir: str | os.PathLike[str],
    prompts_path: str | os.PathLike[str],
    source_actions_path: str | os.PathLike[str],
    registration_actions_path: str | os.PathLike[str],
    analysis_amendment_path: str | os.PathLike[str],
    pre_score_seal_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None,
) -> dict[str, Path]:
    """Reject path mistakes before consuming the one formal audit attempt."""
    run_root = _require_canonical_formal_run(
        run_dir, label="structural audit CLI"
    )
    requested_output = (
        output_path
        if output_path is not None
        else run_root / STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME
    )
    return {
        "run": run_root,
        "output": _require_canonical_audit_output(run_root, requested_output),
        "prompts": _require_canonical_regular_input(
            prompts_path,
            ROOT / generate.STRUCTURAL_CONTROL_PROMPTS,
            label="formal audit development prompts",
        ),
        "actions": _require_canonical_regular_input(
            source_actions_path,
            ROOT / STRUCTURAL_CONTROL_ACTIONS_PATH,
            label="formal audit structural actions",
        ),
        "registration": _require_canonical_regular_input(
            registration_actions_path,
            ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
            label="formal audit structural registration",
        ),
        "analysis_amendment": _require_canonical_regular_input(
            analysis_amendment_path,
            ROOT / STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH,
            label="formal audit analysis amendment",
        ),
        "pre_score_seal": _require_canonical_regular_input(
            pre_score_seal_path,
            run_root / STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME,
            label="formal audit pre-score seal",
        ),
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


@contextmanager
def pre_score_seal_lock(run_dir: str | os.PathLike[str]):
    """Exclude generation and scoring while publishing the pre-score seal."""
    run_root = Path(run_dir).resolve()
    generation_handle = open(run_root / ".generate.lock", "a+")
    try:
        try:
            fcntl.flock(generation_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"generation or scoring is active for {run_root}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(generation_handle.fileno(), fcntl.LOCK_UN)
        finally:
            generation_handle.close()


def _seal_binding(path: str, digest: str) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def _validate_binding(
    value: Any, *, expected_path: str, observed_path: Path, label: str
) -> None:
    binding = _require_exact_fields(
        value, label=f"pre-score seal {label} binding", fields={"path", "sha256"}
    )
    if binding.get("path") != expected_path:
        raise ValueError(f"pre-score seal {label} path differs")
    digest = binding.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"pre-score seal {label} hash is invalid")
    if (
        observed_path.is_symlink()
        or not observed_path.is_file()
        or base_audit.sha256_file(observed_path) != digest
    ):
        raise ValueError(f"pre-score seal {label} bytes differ")


def validate_pre_score_seal(
    path: str | os.PathLike[str],
    *,
    analysis_amendment_path: str | os.PathLike[str],
    base_commit: str,
) -> Mapping[str, Any]:
    """Validate the immutable pre-score bindings without inspecting outcomes."""
    if base_commit != STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT:
        raise ValueError("pre-score seal base commit differs from the frozen run")
    supplied_seal_path = Path(path)
    if supplied_seal_path.is_symlink():
        raise ValueError("pre-score seal must not be a symbolic link")
    seal_path = supplied_seal_path.resolve()
    if seal_path.name != STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME:
        raise ValueError("pre-score seal must use its canonical filename")
    run_root = _require_canonical_formal_run(
        seal_path.parent, label="pre-score seal validation"
    )
    seal = _load_json(seal_path, label="pre-score seal")
    _require_exact_fields(
        seal,
        label="pre-score seal",
        fields={
            "schema",
            "base_commit",
            "expected_manifest_rows",
            "outcome_artifacts_absent_at_creation",
            "bindings",
        },
    )
    if seal.get("schema") != STRUCTURAL_CONTROL_PRE_SCORE_SEAL_SCHEMA:
        raise ValueError("pre-score seal schema differs")
    if seal.get("base_commit") != base_commit:
        raise ValueError("pre-score seal does not bind the frozen base commit")
    expected_rows = seal.get("expected_manifest_rows")
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows != STRUCTURAL_CONTROL_EXPECTED_TASKS
    ):
        raise ValueError("pre-score seal manifest row declaration differs")
    if seal.get("outcome_artifacts_absent_at_creation") is not True:
        raise ValueError("pre-score seal lacks the outcome-absence declaration")
    bindings = _require_exact_fields(
        seal.get("bindings"),
        label="pre-score seal bindings",
        fields={"config", "manifest", "actions", "registration", "analysis_amendment"},
    )

    config_path = run_root / "config.json"
    manifest_path = run_root / "manifest.jsonl"
    actions_path = (ROOT / STRUCTURAL_CONTROL_ACTIONS_PATH).resolve()
    registration_path = (
        ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE
    ).resolve()
    supplied_amendment_path = Path(analysis_amendment_path)
    if supplied_amendment_path.is_symlink():
        raise ValueError("analysis amendment must not be a symbolic link")
    amendment_path = supplied_amendment_path.resolve()
    canonical_amendment_path = (
        ROOT / STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH
    ).resolve()
    if amendment_path != canonical_amendment_path:
        raise ValueError("pre-score seal requires the canonical analysis amendment")
    amendment_relative = _repository_relative(
        amendment_path, label="analysis amendment"
    )
    for value, expected_path, observed_path, label in (
        (bindings.get("config"), "config.json", config_path, "config"),
        (bindings.get("manifest"), "manifest.jsonl", manifest_path, "manifest"),
        (
            bindings.get("actions"),
            STRUCTURAL_CONTROL_ACTIONS_PATH,
            actions_path,
            "actions",
        ),
        (
            bindings.get("registration"),
            generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
            registration_path,
            "registration",
        ),
        (
            bindings.get("analysis_amendment"),
            amendment_relative,
            amendment_path,
            "analysis amendment",
        ),
    ):
        _validate_binding(
            value,
            expected_path=expected_path,
            observed_path=observed_path,
            label=label,
        )

    config = _load_json(config_path, label="sealed run config")
    if config.get("git_commit") != base_commit:
        raise ValueError("sealed run config generation commit differs")
    manifest = _load_jsonl(manifest_path, label="sealed manifest")
    if len(manifest) != STRUCTURAL_CONTROL_EXPECTED_TASKS:
        raise ValueError(
            f"pre-score seal requires {STRUCTURAL_CONTROL_EXPECTED_TASKS} manifest rows"
        )
    for relative_path, observed_path in (
        (STRUCTURAL_CONTROL_ACTIONS_PATH, actions_path),
        (generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE, registration_path),
    ):
        if observed_path.read_bytes() != _git_bytes(base_commit, relative_path):
            raise ValueError(f"sealed frozen source bytes differ: {relative_path}")
    validate_analysis_amendment(amendment_path, base_commit=base_commit)
    return dict(seal)


def create_pre_score_seal(
    run_dir: str | os.PathLike[str],
    source_actions_path: str | os.PathLike[str],
    registration_actions_path: str | os.PathLike[str],
    analysis_amendment_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    base_commit: str = STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
) -> Mapping[str, Any]:
    """Atomically create the one-shot seal while generation/scoring are excluded."""
    run_root = _require_canonical_formal_run(
        run_dir, label="pre-score seal creation"
    )
    actions_path = Path(source_actions_path).resolve()
    registration_path = Path(registration_actions_path).resolve()
    supplied_amendment_path = Path(analysis_amendment_path)
    if supplied_amendment_path.is_symlink():
        raise ValueError("analysis amendment must not be a symbolic link")
    amendment_path = supplied_amendment_path.resolve()
    canonical_amendment_path = (
        ROOT / STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH
    ).resolve()
    if amendment_path != canonical_amendment_path:
        raise ValueError("pre-score seal requires the canonical analysis amendment")
    seal_path = (
        Path(output_path).resolve()
        if output_path is not None
        else run_root / STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
    )
    if seal_path != run_root / STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME:
        raise ValueError("pre-score seal output must use the canonical run path")
    if actions_path != (ROOT / STRUCTURAL_CONTROL_ACTIONS_PATH).resolve():
        raise ValueError("pre-score seal actions path differs from the frozen executable")
    if registration_path != (
        ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE
    ).resolve():
        raise ValueError("pre-score seal registration path differs from the frozen source")
    amendment_relative = _repository_relative(
        amendment_path, label="analysis amendment"
    )

    with pre_score_seal_lock(run_root):
        if os.path.lexists(seal_path):
            raise ValueError("pre-score seal is one-shot; output already exists")
        if _pre_score_forbidden_artifact_exists(run_root):
            raise ValueError(
                "pre-score seal requires all score, audit, and evaluation artifacts absent"
            )
        config_path = run_root / "config.json"
        manifest_path = run_root / "manifest.jsonl"
        for label, input_path in (
            ("config", config_path),
            ("manifest", manifest_path),
            ("actions", actions_path),
            ("registration", registration_path),
            ("analysis amendment", amendment_path),
        ):
            if input_path.is_symlink() or not input_path.is_file():
                raise ValueError(f"pre-score seal {label} input is missing")
        config = _load_json(config_path, label="run config")
        if config.get("git_commit") != base_commit:
            raise ValueError("run config generation commit differs from seal base")
        manifest = _load_jsonl(manifest_path, label="manifest")
        if len(manifest) != STRUCTURAL_CONTROL_EXPECTED_TASKS:
            raise ValueError(
                f"pre-score seal requires {STRUCTURAL_CONTROL_EXPECTED_TASKS} manifest rows"
            )
        for relative_path, input_path in (
            (STRUCTURAL_CONTROL_ACTIONS_PATH, actions_path),
            (generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE, registration_path),
        ):
            if input_path.read_bytes() != _git_bytes(base_commit, relative_path):
                raise ValueError(f"pre-score frozen source bytes differ: {relative_path}")
        validate_analysis_amendment(amendment_path, base_commit=base_commit)
        seal = {
            "schema": STRUCTURAL_CONTROL_PRE_SCORE_SEAL_SCHEMA,
            "base_commit": base_commit,
            "expected_manifest_rows": STRUCTURAL_CONTROL_EXPECTED_TASKS,
            "outcome_artifacts_absent_at_creation": True,
            "bindings": {
                "config": _seal_binding(
                    "config.json", base_audit.sha256_file(config_path)
                ),
                "manifest": _seal_binding(
                    "manifest.jsonl", base_audit.sha256_file(manifest_path)
                ),
                "actions": _seal_binding(
                    STRUCTURAL_CONTROL_ACTIONS_PATH,
                    base_audit.sha256_file(actions_path),
                ),
                "registration": _seal_binding(
                    generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                    base_audit.sha256_file(registration_path),
                ),
                "analysis_amendment": _seal_binding(
                    amendment_relative, base_audit.sha256_file(amendment_path)
                ),
            },
        }
        payload = (
            json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _atomic_write(seal_path, payload)
        _fsync_directory(run_root)
        return validate_pre_score_seal(
            seal_path,
            analysis_amendment_path=amendment_path,
            base_commit=base_commit,
        )


def _exclusive_durable_write(path: Path, payload: bytes, *, label: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"{label} already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _scoring_temporary_artifact_exists(run_root: Path) -> bool:
    for entry in run_root.iterdir():
        name = entry.name
        if (
            name.startswith(".scores.jsonl.")
            and name.endswith(".tmp")
        ) or (
            name.startswith(f".{STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME}.")
            and name.endswith(".tmp")
        ) or (
            name.startswith(f".{STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME}.")
            and name.endswith(".tmp")
        ):
            return True
    return False


def _require_initial_scoring_artifacts_absent(run_root: Path) -> None:
    for name in (
        "scores.jsonl",
        STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME,
        STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME,
    ):
        if os.path.lexists(run_root / name):
            raise ValueError("sealed scoring is one-shot; score or receipt exists")
    if _scoring_temporary_artifact_exists(run_root):
        raise ValueError("sealed scoring rejects score or receipt temporary debris")
    if _pre_score_forbidden_artifact_exists(run_root):
        raise ValueError(
            "sealed scoring requires all score, audit, and evaluation artifacts absent"
        )


def _validate_scoring_sources(
    *, base_commit: str = STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT
) -> str:
    if base_commit != STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT:
        raise ValueError("sealed scoring base commit differs from the frozen run")
    head_commit = _current_head_commit()
    for relative_path, expected_sha256 in (
        (STRUCTURAL_CONTROL_SCORE_SCRIPT_PATH, STRUCTURAL_CONTROL_SCORE_SCRIPT_SHA256),
        (STRUCTURAL_CONTROL_SCORE_CONFIG_PATH, STRUCTURAL_CONTROL_SCORE_CONFIG_SHA256),
    ):
        path = ROOT / relative_path
        current = _read_regular_bytes(path, label=f"sealed scoring {relative_path}")
        if _sha256_bytes(current) != expected_sha256:
            raise ValueError(f"sealed scoring source hash differs: {relative_path}")
        if current != _git_bytes(base_commit, relative_path):
            raise ValueError(f"sealed scoring source differs from base: {relative_path}")
        if current != _git_bytes(head_commit, relative_path):
            raise ValueError(f"sealed scoring source differs from analysis HEAD: {relative_path}")
    return head_commit


def _validate_scoring_inputs(
    run_root: Path,
    analysis_amendment_path: str | os.PathLike[str],
    pre_score_seal_path: str | os.PathLike[str],
    *,
    base_commit: str = STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
    require_canonical_amendment: bool,
) -> tuple[Path, Path, str]:
    raw_amendment = Path(analysis_amendment_path)
    raw_seal = Path(pre_score_seal_path)
    if raw_amendment.is_symlink():
        raise ValueError("sealed scoring analysis amendment must not be a symlink")
    amendment_path = raw_amendment.resolve()
    canonical_amendment = (ROOT / STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH).resolve()
    if require_canonical_amendment and amendment_path != canonical_amendment:
        raise ValueError("sealed scoring requires the canonical analysis amendment")
    if raw_seal.is_symlink() or raw_seal.resolve() != (
        run_root / STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
    ):
        raise ValueError("sealed scoring requires the canonical pre-score seal")
    seal_path = raw_seal.resolve()
    head_commit = _validate_scoring_sources(base_commit=base_commit)
    validate_analysis_amendment(amendment_path, base_commit=base_commit)
    validate_pre_score_seal(
        seal_path,
        analysis_amendment_path=amendment_path,
        base_commit=base_commit,
    )
    return amendment_path, seal_path, head_commit


def _parse_scoring_attempt_payload(payload: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical structural scoring attempt marker is invalid") from exc
    attempt = _require_exact_fields(
        value,
        label="structural scoring attempt marker",
        fields={
            "schema",
            "scope",
            "one_shot",
            "base_commit",
            "analysis_commit",
            "expected_score_rows",
            "command",
            "launcher",
            "bindings",
        },
    )
    launcher = _require_exact_fields(
        attempt.get("launcher"),
        label="structural scoring attempt launcher",
        fields={"pid", "proc_start_ticks", "boot_id", "capability_sha256"},
    )
    if (
        attempt.get("schema") != STRUCTURAL_CONTROL_SCORING_ATTEMPT_SCHEMA
        or attempt.get("scope") != STRUCTURAL_CONTROL_AUDITOR_SCOPE
        or attempt.get("one_shot") is not True
        or attempt.get("base_commit") != STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT
        or attempt.get("expected_score_rows") != STRUCTURAL_CONTROL_EXPECTED_TASKS
        or not isinstance(launcher.get("pid"), int)
        or isinstance(launcher.get("pid"), bool)
        or launcher["pid"] <= 0
        or not isinstance(launcher.get("proc_start_ticks"), int)
        or isinstance(launcher.get("proc_start_ticks"), bool)
        or launcher["proc_start_ticks"] <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", str(launcher.get("capability_sha256", "")))
        or launcher.get("capability_sha256") == "0" * 64
        or not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            str(launcher.get("boot_id", "")),
        )
        or not re.fullmatch(r"[0-9a-f]{40}", str(attempt.get("analysis_commit", "")))
    ):
        raise ValueError("canonical structural scoring attempt marker is invalid")
    return attempt


def require_scoring_attempt_marker(
    run_dir: str | os.PathLike[str],
    *,
    analysis_amendment_path: str | os.PathLike[str],
    pre_score_seal_path: str | os.PathLike[str],
) -> Path:
    """Validate the sealed-scoring attempt without consuming another attempt."""
    run_root = _require_canonical_formal_run(
        run_dir, label="structural scoring attempt validation"
    )
    marker_path = run_root / STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
    marker_payload = _read_regular_bytes(
        marker_path, label="canonical structural scoring attempt marker"
    )
    attempt = _parse_scoring_attempt_payload(marker_payload)
    amendment_path, seal_path, head_commit = _validate_scoring_inputs(
        run_root,
        analysis_amendment_path,
        pre_score_seal_path,
        require_canonical_amendment=True,
    )
    if attempt.get("analysis_commit") != head_commit:
        raise ValueError("canonical structural scoring attempt analysis commit differs")
    launcher = attempt["launcher"]
    expected = _scoring_attempt_marker_payload(
        run_root,
        amendment_path,
        seal_path,
        analysis_commit=head_commit,
        launcher_pid=int(launcher["pid"]),
        launcher_start_ticks=int(launcher["proc_start_ticks"]),
        boot_id=str(launcher["boot_id"]),
        capability_sha256=str(launcher["capability_sha256"]),
    )
    if marker_payload != expected:
        raise ValueError("canonical structural scoring attempt marker is invalid")
    return marker_path


def _parse_scoring_success_payload(payload: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical structural scoring success receipt is invalid") from exc
    receipt = _require_exact_fields(
        value,
        label="structural scoring success receipt",
        fields={
            "schema",
            "scope",
            "base_commit",
            "analysis_commit",
            "score_rows",
            "child",
            "scorer_provenance_sha256",
            "command",
            "bindings",
        },
    )
    child = _require_exact_fields(
        receipt.get("child"),
        label="structural scoring success child",
        fields={"pid", "returncode"},
    )
    if (
        receipt.get("schema") != STRUCTURAL_CONTROL_SCORING_SUCCESS_SCHEMA
        or receipt.get("scope") != STRUCTURAL_CONTROL_AUDITOR_SCOPE
        or receipt.get("base_commit") != STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT
        or receipt.get("score_rows") != STRUCTURAL_CONTROL_EXPECTED_TASKS
        or not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("analysis_commit", "")))
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(receipt.get("scorer_provenance_sha256", ""))
        )
        or not isinstance(child.get("pid"), int)
        or isinstance(child.get("pid"), bool)
        or child["pid"] <= 0
        or child.get("returncode") != 0
    ):
        raise ValueError("canonical structural scoring success receipt is invalid")
    return receipt


def require_scoring_success_receipt(
    run_dir: str | os.PathLike[str],
    *,
    analysis_amendment_path: str | os.PathLike[str],
    pre_score_seal_path: str | os.PathLike[str],
) -> Path:
    """Validate the zero-exit scoring receipt before parsing score outcomes."""
    run_root = _require_canonical_formal_run(
        run_dir, label="structural scoring success validation"
    )
    receipt_path = run_root / STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME
    receipt_payload = _read_regular_bytes(
        receipt_path, label="canonical structural scoring success receipt"
    )
    receipt = _parse_scoring_success_payload(receipt_payload)
    attempt_path = require_scoring_attempt_marker(
        run_root,
        analysis_amendment_path=analysis_amendment_path,
        pre_score_seal_path=pre_score_seal_path,
    )
    attempt = _parse_scoring_attempt_payload(
        _read_regular_bytes(
            attempt_path, label="canonical structural scoring attempt marker"
        )
    )
    source = _load_yaml(
        ROOT / STRUCTURAL_CONTROL_ACTIONS_PATH,
        label="structural executable actions",
    )
    registered_scorer_hash = (source.get("scoring") or {}).get(
        "registered_scorer_provenance_sha256"
    )
    if (
        registered_scorer_hash != STRUCTURAL_CONTROL_SCORER_PROVENANCE_SHA256
        or receipt.get("scorer_provenance_sha256") != registered_scorer_hash
    ):
        raise ValueError("canonical structural scoring success scorer hash differs")
    scores_path = run_root / "scores.jsonl"
    _read_regular_bytes(scores_path, label="canonical structural scores")
    if _scoring_temporary_artifact_exists(run_root):
        raise ValueError("canonical structural scoring has temporary debris")
    head_commit = _validate_scoring_sources()
    if (
        receipt.get("analysis_commit") != head_commit
        or receipt.get("analysis_commit") != attempt.get("analysis_commit")
    ):
        raise ValueError("canonical structural scoring success analysis commit differs")
    child = receipt["child"]
    expected = _scoring_success_receipt_payload(
        run_root,
        analysis_amendment_path,
        pre_score_seal_path,
        analysis_commit=head_commit,
        child_pid=int(child["pid"]),
        scorer_provenance_sha256=str(receipt["scorer_provenance_sha256"]),
    )
    if receipt_payload != expected:
        raise ValueError("canonical structural scoring success receipt is invalid")
    return receipt_path


def require_scoring_child_authorization(
    run_dir: str | os.PathLike[str],
    *,
    analysis_amendment_path: str | os.PathLike[str],
    pre_score_seal_path: str | os.PathLike[str],
) -> Path:
    """Authenticate the one child allowed to enter the formal metric loop."""
    run_root = _require_canonical_formal_run(
        run_dir, label="structural scoring child authorization"
    )
    marker_path = require_scoring_attempt_marker(
        run_root,
        analysis_amendment_path=analysis_amendment_path,
        pre_score_seal_path=pre_score_seal_path,
    )
    attempt = _parse_scoring_attempt_payload(
        _read_regular_bytes(
            marker_path, label="canonical structural scoring attempt marker"
        )
    )
    launcher = attempt["launcher"]
    if _linux_boot_id() != launcher["boot_id"]:
        raise ValueError("sealed scoring launcher boot identity differs")
    parent_pid = os.getppid()
    if parent_pid != launcher["pid"] or _linux_process_start_ticks(parent_pid) != (
        launcher["proc_start_ticks"]
    ):
        raise ValueError("sealed scoring child does not match its launcher process")
    if [str(Path(sys.executable).resolve()), *sys.argv] != attempt["command"]["argv"]:
        raise ValueError("sealed scoring child argv differs from the registered command")
    if str(Path.cwd().resolve()) != attempt["command"]["cwd"]:
        raise ValueError("sealed scoring child working directory differs")
    for key, expected_value in attempt["command"]["environment"].items():
        if os.environ.get(key) != expected_value:
            raise ValueError("sealed scoring child environment differs")
    capability = os.environ.pop(STRUCTURAL_CONTROL_SCORING_CAPABILITY_ENV, None)
    if (
        launcher["capability_sha256"] == "0" * 64
        or not isinstance(capability, str)
        or _sha256_bytes(capability.encode("ascii", errors="strict"))
        != launcher["capability_sha256"]
    ):
        raise ValueError("sealed scoring child capability is invalid")
    if os.path.lexists(run_root / "scores.jsonl") or os.path.lexists(
        run_root / STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME
    ):
        raise ValueError("sealed scoring child requires scores and receipt absent")
    if _scoring_temporary_artifact_exists(run_root):
        raise ValueError("sealed scoring child rejects temporary debris")
    return marker_path


def _validate_completed_scoring(run_root: Path) -> str:
    scores_path = run_root / "scores.jsonl"
    _read_regular_bytes(scores_path, label="canonical structural scores")
    manifest = _load_jsonl(run_root / "manifest.jsonl", label="manifest")
    scores = _load_jsonl(scores_path, label="scores")
    if len(manifest) != STRUCTURAL_CONTROL_EXPECTED_TASKS or len(scores) != (
        STRUCTURAL_CONTROL_EXPECTED_TASKS
    ):
        raise ValueError("sealed scoring did not produce exactly 792 bound rows")
    source_path = (ROOT / STRUCTURAL_CONTROL_ACTIONS_PATH).resolve()
    source = _load_yaml(source_path, label="structural executable actions")
    config = _load_json(run_root / "config.json", label="run config")
    normalized_actions, _ = generate.load_actions(str(source_path), 50)
    scorer_hash = _validate_registered_artifact_bindings(
        run_root, config, source, manifest, normalized_actions
    )
    if scorer_hash != STRUCTURAL_CONTROL_SCORER_PROVENANCE_SHA256:
        raise ValueError("sealed scoring scorer provenance differs from the frozen contract")
    provenance = scores[0].get("scorer_provenance")
    scorer_records = provenance.get("scorers") if isinstance(provenance, Mapping) else None
    if not isinstance(scorer_records, list):
        raise ValueError("sealed scoring lacks a unified scorer provenance contract")
    score_config = _load_yaml(
        ROOT / STRUCTURAL_CONTROL_SCORE_CONFIG_PATH,
        label="sealed scoring config",
    )
    configured_metrics = score_config.get("metrics")
    if (
        not isinstance(configured_metrics, list)
        or not configured_metrics
        or any(not isinstance(name, str) for name in configured_metrics)
        or len(configured_metrics) != len(set(configured_metrics))
        or configured_metrics != list(STRUCTURAL_CONTROL_SCORE_METRICS)
        or provenance.get("metrics") != configured_metrics
    ):
        raise ValueError("sealed scoring metric order differs from the frozen recipe")
    output_names: list[str] = []
    for scorer in scorer_records:
        outputs = scorer.get("output_keys") if isinstance(scorer, Mapping) else None
        if not isinstance(outputs, list):
            raise ValueError("sealed scoring scorer output contract is invalid")
        for output in outputs:
            if not isinstance(output, Mapping) or not isinstance(output.get("name"), str):
                raise ValueError("sealed scoring scorer output key is invalid")
            output_names.append(str(output["name"]))
    required_outputs = set(output_names)
    if (
        required_outputs != set(STRUCTURAL_CONTROL_SCORE_OUTPUT_KEYS)
        or len(required_outputs) != len(output_names)
    ):
        raise ValueError("sealed scoring scorer output keys differ from the frozen recipe")
    for score in scores:
        if score.get("scorer_provenance") != provenance:
            raise ValueError(f"{score.get('id')}: scorer provenance contract drifted")
        for key in required_outputs:
            value = score.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"{score.get('id')}: required score {key!r} is missing or non-finite"
                )
    return scorer_hash


def _scoring_child_environment(capability: str) -> dict[str, str]:
    environment = dict(os.environ)
    for key in (
        STRUCTURAL_CONTROL_SCORING_CAPABILITY_ENV,
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONBREAKPOINT",
        "LD_PRELOAD",
    ):
        environment.pop(key, None)
    environment.update(STRUCTURAL_CONTROL_SCORING_ENVIRONMENT)
    environment[STRUCTURAL_CONTROL_SCORING_CAPABILITY_ENV] = capability
    return environment


def launch_sealed_scoring(
    run_dir: str | os.PathLike[str],
    source_actions_path: str | os.PathLike[str],
    registration_actions_path: str | os.PathLike[str],
    analysis_amendment_path: str | os.PathLike[str],
    pre_score_seal_path: str | os.PathLike[str],
) -> Path:
    """Consume the one scoring attempt, run the fixed child, and seal success."""
    run_root = _require_canonical_formal_run(
        run_dir, label="sealed scoring launcher"
    )
    actions_path = Path(source_actions_path)
    registration_path = Path(registration_actions_path)
    if actions_path.is_symlink() or actions_path.resolve() != (
        ROOT / STRUCTURAL_CONTROL_ACTIONS_PATH
    ).resolve():
        raise ValueError("sealed scoring actions path differs from the canonical source")
    if registration_path.is_symlink() or registration_path.resolve() != (
        ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE
    ).resolve():
        raise ValueError(
            "sealed scoring registration path differs from the canonical source"
        )

    capability = secrets.token_urlsafe(32)
    capability_sha256 = _sha256_bytes(capability.encode("ascii"))
    launcher_pid = os.getpid()
    launcher_start_ticks = _linux_process_start_ticks(launcher_pid)
    boot_id = _linux_boot_id()
    with pre_score_seal_lock(run_root):
        amendment_path, seal_path, head_commit = _validate_scoring_inputs(
            run_root,
            analysis_amendment_path,
            pre_score_seal_path,
            require_canonical_amendment=True,
        )
        _require_initial_scoring_artifacts_absent(run_root)
        marker_path = run_root / STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
        marker_payload = _scoring_attempt_marker_payload(
            run_root,
            amendment_path,
            seal_path,
            analysis_commit=head_commit,
            launcher_pid=launcher_pid,
            launcher_start_ticks=launcher_start_ticks,
            boot_id=boot_id,
            capability_sha256=capability_sha256,
        )
        _exclusive_durable_write(
            marker_path, marker_payload, label="sealed scoring attempt marker"
        )

    argv = _canonical_scoring_argv(run_root)
    child = subprocess.Popen(
        argv,
        cwd=str(ROOT.resolve()),
        env=_scoring_child_environment(capability),
        shell=False,
    )
    return_code = child.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, argv)

    with pre_score_seal_lock(run_root):
        require_scoring_attempt_marker(
            run_root,
            analysis_amendment_path=amendment_path,
            pre_score_seal_path=seal_path,
        )
        if os.path.lexists(run_root / STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME):
            raise ValueError("sealed scoring success receipt already exists")
        if _scoring_temporary_artifact_exists(run_root):
            raise ValueError("sealed scoring completion has temporary debris")
        scorer_provenance_sha256 = _validate_completed_scoring(run_root)
        receipt_path = run_root / STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME
        receipt_payload = _scoring_success_receipt_payload(
            run_root,
            amendment_path,
            seal_path,
            analysis_commit=head_commit,
            child_pid=int(child.pid),
            scorer_provenance_sha256=scorer_provenance_sha256,
        )
        _exclusive_durable_write(
            receipt_path, receipt_payload, label="sealed scoring success receipt"
        )
        require_scoring_success_receipt(
            run_root,
            analysis_amendment_path=amendment_path,
            pre_score_seal_path=seal_path,
        )
    return receipt_path


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


def _current_head_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.PIPE,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot resolve current analysis HEAD") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("current analysis HEAD is not full lowercase 40-hex")
    return commit


def _require_git_ancestor(
    ancestor: str, descendant: str, *, label: str, strict: bool
) -> None:
    if strict and ancestor == descendant:
        raise ValueError(f"{label} must use a strict two-commit authorization flow")
    try:
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"{label} ancestry verification failed") from exc
    if ancestry.returncode != 0:
        raise ValueError(f"{label} ancestry verification failed")


def _git_command_bytes(arguments: Sequence[str], *, label: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *arguments], stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"{label} verification failed") from exc


def _validate_analysis_authorization_topology(
    reviewed_commit: str, head_commit: str, amendment_relative: str
) -> None:
    """Require one direct authorization commit changing only the amendment blob."""
    parent_row = _git_command_bytes(
        ["rev-list", "--parents", "--max-count=1", head_commit],
        label="analysis amendment authorization parent",
    )
    if parent_row.split() != [
        head_commit.encode("ascii"),
        reviewed_commit.encode("ascii"),
    ]:
        raise ValueError(
            "analysis amendment authorization commit must have reviewed_commit "
            "as its sole parent"
        )

    raw_diff = _git_command_bytes(
        [
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "-r",
            "-z",
            "--abbrev=40",
            "-M",
            "-C",
            reviewed_commit,
            head_commit,
        ],
        label="analysis amendment authorization diff",
    )
    fields = raw_diff.split(b"\0")
    if len(fields) != 3 or fields[-1] != b"":
        raise ValueError(
            "analysis amendment authorization commit must modify only the amendment file"
        )
    metadata, changed_path = fields[:2]
    try:
        metadata_fields = metadata.decode("ascii").split()
    except UnicodeDecodeError as exc:
        raise ValueError(
            "analysis amendment authorization commit has an invalid raw diff"
        ) from exc
    if (
        len(metadata_fields) != 5
        or metadata_fields[0] != ":100644"
        or metadata_fields[1] != "100644"
        or not re.fullmatch(r"[0-9a-f]{40}", metadata_fields[2])
        or not re.fullmatch(r"[0-9a-f]{40}", metadata_fields[3])
        or metadata_fields[2] == metadata_fields[3]
        or metadata_fields[4] != "M"
        or changed_path != amendment_relative.encode("utf-8")
    ):
        raise ValueError(
            "analysis amendment authorization commit must be one ordinary 100644 "
            "amendment modification"
        )


def _reconstruct_analysis_authorization(
    candidate_bytes: bytes, *, reviewer: str, reviewed_commit: str
) -> bytes:
    """Apply exactly the three permitted scalar-line authorization edits."""
    replacements = (
        (
            b"status: blocked_pending_independent_review",
            b"status: authorized_pre_score",
        ),
        (b"reviewer: null", f"reviewer: {reviewer}".encode("ascii")),
        (
            b"reviewed_commit: null",
            f"reviewed_commit: {reviewed_commit}".encode("ascii"),
        ),
    )
    lines = candidate_bytes.splitlines(keepends=True)

    def line_body(line: bytes) -> bytes:
        if line.endswith(b"\r\n"):
            return line[:-2]
        if line.endswith((b"\n", b"\r")):
            return line[:-1]
        return line

    for blocked_line, authorized_line in replacements:
        matches = [
            index for index, line in enumerate(lines) if line_body(line) == blocked_line
        ]
        if len(matches) != 1:
            raise ValueError(
                "reviewed amendment candidate must contain each authorization line "
                "exactly once"
            )
        index = matches[0]
        ending = lines[index][len(blocked_line) :]
        lines[index] = authorized_line + ending
    return b"".join(lines)


def _load_yaml_bytes(payload: bytes, *, label: str) -> dict:
    try:
        value = yaml.safe_load(payload) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} is not readable YAML") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return value


def _base_analysis_files(base_commit: str) -> Mapping[str, str]:
    actions = _load_yaml_bytes(
        _git_bytes(base_commit, STRUCTURAL_CONTROL_ACTIONS_PATH),
        label="base structural actions",
    )
    registration = _load_yaml_bytes(
        _git_bytes(base_commit, generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE),
        label="base structural registration",
    )
    actions_implementation = actions.get("analysis_implementation")
    if actions_implementation != registration.get("analysis_implementation"):
        raise ValueError("base analysis implementation differs from registration")
    implementation = _require_exact_fields(
        actions_implementation,
        label="base analysis implementation",
        fields={"schema", "files"},
    )
    if implementation.get("schema") != generate.STRUCTURAL_CONTROL_ANALYSIS_SCHEMA:
        raise ValueError("base analysis implementation schema differs")
    files = implementation.get("files")
    if not isinstance(files, dict) or set(files) != set(
        generate.STRUCTURAL_CONTROL_ANALYSIS_PATHS
    ):
        raise ValueError("base analysis implementation paths differ")
    return {str(path): str(digest) for path, digest in files.items()}


def validate_analysis_amendment(
    path: str | os.PathLike[str], *, base_commit: str
) -> Mapping[str, Any]:
    """Validate a reviewed, result-blind, two-commit analysis authorization."""
    if base_commit != STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT:
        raise ValueError("analysis amendment base commit differs from the frozen run")
    supplied_amendment_path = Path(path)
    if supplied_amendment_path.is_symlink():
        raise ValueError("analysis amendment must not be a symbolic link")
    amendment_path = supplied_amendment_path.resolve()
    amendment_relative = _repository_relative(
        amendment_path, label="analysis amendment"
    )
    try:
        amendment_bytes = amendment_path.read_bytes()
    except OSError as exc:
        raise ValueError("analysis amendment is not readable YAML") from exc
    amendment = _load_yaml_bytes(amendment_bytes, label="analysis amendment")
    _require_exact_fields(
        amendment,
        label="analysis amendment",
        fields=set(STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_FIELDS),
    )
    if amendment.get("schema") != STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_SCHEMA:
        raise ValueError("analysis amendment schema differs")
    if amendment.get("status") != STRUCTURAL_CONTROL_AMENDMENT_STATUS:
        raise ValueError("analysis amendment is not authorized_pre_score")
    reviewer = amendment.get("reviewer")
    if not isinstance(reviewer, str) or not STRUCTURAL_CONTROL_REVIEWER_PATTERN.fullmatch(
        reviewer
    ):
        raise ValueError("analysis amendment reviewer must be a safe ASCII identity")
    reviewed_commit = amendment.get("reviewed_commit")
    if not isinstance(reviewed_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", reviewed_commit
    ):
        raise ValueError("analysis amendment reviewed_commit must be full lowercase 40-hex")
    if amendment.get("base_commit") != base_commit:
        raise ValueError("analysis amendment does not bind the frozen base commit")
    if amendment.get("result_access_before_authorization") is not False:
        raise ValueError("analysis amendment authorization was not result-blind")
    if amendment.get("scoring_started_before_authorization") is not False:
        raise ValueError("analysis amendment was not authorized before scoring")
    unchanged = _require_exact_fields(
        amendment.get("unchanged_invariants"),
        label="analysis amendment unchanged invariants",
        fields=set(STRUCTURAL_CONTROL_UNCHANGED_INVARIANTS),
    )
    if any(unchanged[key] is not True for key in STRUCTURAL_CONTROL_UNCHANGED_INVARIANTS):
        raise ValueError("analysis amendment changes a frozen study invariant")
    authorizations = _require_exact_fields(
        amendment.get("authorizations"),
        label="analysis amendment authorizations",
        fields=set(STRUCTURAL_CONTROL_UNAUTHORIZED_USES),
    )
    if any(authorizations[key] is not False for key in STRUCTURAL_CONTROL_UNAUTHORIZED_USES):
        raise ValueError("analysis amendment grants an unauthorized downstream use")
    replacements = amendment.get("replacements")
    if not isinstance(replacements, dict) or set(replacements) != set(
        STRUCTURAL_CONTROL_ANALYSIS_REPLACEMENTS
    ):
        raise ValueError("analysis amendment replacement paths differ")

    head_commit = _current_head_commit()
    _require_git_ancestor(
        base_commit,
        reviewed_commit,
        label="analysis amendment base-to-review",
        strict=False,
    )
    _validate_analysis_authorization_topology(
        reviewed_commit, head_commit, amendment_relative
    )
    try:
        candidate_bytes = _git_bytes(reviewed_commit, amendment_relative)
        candidate = _load_yaml_bytes(
            candidate_bytes,
            label="reviewed blocked amendment candidate",
        )
    except ValueError as exc:
        raise ValueError("reviewed blocked amendment candidate is missing") from exc
    _require_exact_fields(
        candidate,
        label="reviewed blocked amendment candidate",
        fields=set(STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_FIELDS),
    )
    if (
        candidate.get("status") != STRUCTURAL_CONTROL_AMENDMENT_CANDIDATE_STATUS
        or candidate.get("reviewer") is not None
        or candidate.get("reviewed_commit") is not None
    ):
        raise ValueError("reviewed amendment candidate was not blocked and unauthored")
    authorization_fields = {"status", "reviewer", "reviewed_commit"}
    if any(
        candidate.get(field) != amendment.get(field)
        for field in STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_FIELDS
        - authorization_fields
    ):
        raise ValueError("reviewed amendment candidate content differs from authorization")
    committed_amendment_bytes = _git_bytes(head_commit, amendment_relative)
    if committed_amendment_bytes != amendment_bytes:
        raise ValueError("analysis amendment bytes differ from current HEAD")
    reconstructed_amendment = _reconstruct_analysis_authorization(
        candidate_bytes, reviewer=reviewer, reviewed_commit=reviewed_commit
    )
    if reconstructed_amendment != committed_amendment_bytes:
        raise ValueError(
            "analysis amendment authorization changes bytes outside the three "
            "permitted lines"
        )

    for relative_path in (
        STRUCTURAL_CONTROL_ACTIONS_PATH,
        generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
    ):
        current = (ROOT / relative_path).read_bytes()
        if current != _git_bytes(base_commit, relative_path) or current != _git_bytes(
            head_commit, relative_path
        ):
            raise ValueError(f"frozen structural source bytes differ: {relative_path}")

    registered_files = _base_analysis_files(base_commit)
    for relative_path, registered_hash in registered_files.items():
        if not re.fullmatch(r"[0-9a-f]{64}", registered_hash):
            raise ValueError("base analysis implementation hash is invalid")
        base_payload = _git_bytes(base_commit, relative_path)
        base_hash = _sha256_bytes(base_payload)
        if base_hash != registered_hash:
            raise ValueError(
                f"base analysis implementation hash differs: {relative_path}"
            )
        current_path = (ROOT / relative_path).resolve()
        _repository_relative(current_path, label="analysis implementation")
        if not current_path.is_file():
            raise ValueError(f"analysis implementation is missing: {relative_path}")
        current_payload = current_path.read_bytes()
        current_hash = _sha256_bytes(current_payload)
        if _git_bytes(head_commit, relative_path) != current_payload:
            raise ValueError(
                f"effective analysis implementation differs from current HEAD: {relative_path}"
            )
        if relative_path not in STRUCTURAL_CONTROL_ANALYSIS_REPLACEMENTS:
            if current_hash != base_hash:
                raise ValueError(
                    f"unamended analysis implementation differs: {relative_path}"
                )
            continue
        replacement = _require_exact_fields(
            replacements.get(relative_path),
            label="analysis replacement",
            fields={"base_sha256", "amended_sha256"},
        )
        if replacement.get("base_sha256") != base_hash:
            raise ValueError(f"analysis replacement base hash differs: {relative_path}")
        amended_hash = replacement.get("amended_sha256")
        if not isinstance(amended_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", amended_hash
        ):
            raise ValueError("analysis replacement amended hash is invalid")
        if _sha256_bytes(_git_bytes(reviewed_commit, relative_path)) != amended_hash:
            raise ValueError(
                f"reviewed analysis replacement blob differs: {relative_path}"
            )
        if amended_hash == base_hash or current_hash != amended_hash:
            raise ValueError(f"analysis replacement bytes differ: {relative_path}")
    return dict(amendment)


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
    *,
    replacements: Mapping[str, Any] | None = None,
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
    effective_files = dict(files)
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
        if committed_hash != expected_hash:
            raise ValueError(
                f"committed structural analysis source hash differs for {relative_path}"
            )
        replacement = None if replacements is None else replacements.get(relative_path)
        if replacement is None:
            if current_hash != expected_hash:
                raise ValueError(
                    f"structural analysis source hash differs for {relative_path}"
                )
        elif current_hash != replacement.get("amended_sha256"):
            raise ValueError(
                f"amended structural analysis source hash differs for {relative_path}"
            )
        else:
            effective_files[relative_path] = current_hash
    return {"schema": registered["schema"], "files": effective_files}


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


def validate_action_png_collapse(
    manifest: Sequence[Mapping[str, Any]],
    action_ids: Sequence[str],
    *,
    expected_blocks: int,
    reject_partial_equality: bool = False,
) -> bool:
    """Validate image-hash collapse without disclosing method labels or counts."""
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

    for left, right in combinations(action_ids, 2):
        matching_blocks = sum(
            hashes[str(left)] == hashes[str(right)] for hashes in by_block.values()
        )
        if matching_blocks == expected_blocks:
            raise ValueError("structural outcome collapse validation failed")
        if reject_partial_equality and matching_blocks:
            raise ValueError("structural outcome distinctness validation failed")
    return True


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
    """Reject smoke reuse: the amended v2 auditor is formal-development only."""
    raise ValueError(
        "analysis-amended structural-control auditor v2 is formal-only; "
        "engineering smoke remains frozen pre-amendment evidence"
    )


def audit_run(
    run_dir: str | os.PathLike[str],
    prompts_path: str | os.PathLike[str],
    source_actions_path: str | os.PathLike[str],
    *,
    registration_actions_path: str | os.PathLike[str] = DEFAULT_REGISTRATION,
    analysis_amendment_path: str | os.PathLike[str],
    pre_score_seal_path: str | os.PathLike[str],
) -> dict:
    """Run the generic S7 audit and all structural-control-specific checks."""
    run_path = _require_canonical_formal_run(
        run_dir, label="formal structural audit"
    )
    audit_attempt_path = require_audit_attempt_marker(run_path)
    supplied_seal_path = Path(pre_score_seal_path)
    if supplied_seal_path.is_symlink() or supplied_seal_path.resolve() != (
        run_path / STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
    ):
        raise ValueError("formal audit requires the canonical pre-score seal path")
    scoring_attempt_path = require_scoring_attempt_marker(
        run_path,
        analysis_amendment_path=analysis_amendment_path,
        pre_score_seal_path=supplied_seal_path,
    )
    scoring_success_path = require_scoring_success_receipt(
        run_path,
        analysis_amendment_path=analysis_amendment_path,
        pre_score_seal_path=supplied_seal_path,
    )
    source_path = Path(source_actions_path).resolve()
    registration_path = Path(registration_actions_path).resolve()
    input_paths = {
        "audit_attempt_sha256": audit_attempt_path,
        "scoring_attempt_sha256": scoring_attempt_path,
        "scoring_success_sha256": scoring_success_path,
        "config_sha256": run_path / "config.json",
        "manifest_sha256": run_path / "manifest.jsonl",
        "scores_sha256": run_path / "scores.jsonl",
        "prompts_sha256": Path(prompts_path).resolve(),
        "source_actions_sha256": source_path,
        "source_template_sha256": registration_path,
        "analysis_amendment_sha256": Path(analysis_amendment_path).resolve(),
        "pre_score_seal_sha256": supplied_seal_path.resolve(),
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
    amendment = validate_analysis_amendment(
        analysis_amendment_path, base_commit=generation_commit
    )
    validate_pre_score_seal(
        pre_score_seal_path,
        analysis_amendment_path=analysis_amendment_path,
        base_commit=generation_commit,
    )
    analysis_implementation = _validate_analysis_implementation(
        source,
        registration,
        generation_commit,
        replacements=amendment["replacements"],
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
    validate_action_png_collapse(
        manifest,
        [str(action["id"]) for action in normalized_actions],
        expected_blocks=99,
    )
    require_audit_attempt_marker(run_path)
    require_scoring_attempt_marker(
        run_path,
        analysis_amendment_path=analysis_amendment_path,
        pre_score_seal_path=supplied_seal_path,
    )
    require_scoring_success_receipt(
        run_path,
        analysis_amendment_path=analysis_amendment_path,
        pre_score_seal_path=supplied_seal_path,
    )
    final_hashes = {
        label: base_audit.sha256_file(path) for label, path in input_paths.items()
    }
    if final_hashes != initial_hashes:
        raise ValueError("structural audit inputs changed during verification")
    report = dict(base_report)
    for outcome_field in (
        "all_action_png_hashes_distinct_within_block",
        "allowed_identity_pairs",
        "observed_identity_pairs",
        "registered_identity_pair",
        "identity_pair_png_hashes_equal",
    ):
        report.pop(outcome_field, None)
    report.update(structural_report)
    report.update(
        {
            "audit_schema": "scheduler_native_structural_control_audit_v2",
            "auditor_scope": STRUCTURAL_CONTROL_AUDITOR_SCOPE,
            "generation_commit": generation_commit,
            "executable_actions_sha256": base_audit.sha256_file(source_path),
            "registration_sha256": base_audit.sha256_file(registration_path),
            "analysis_implementation": analysis_implementation,
            "analysis_implementation_sha256": json_sha256(
                analysis_implementation
            ),
            "analysis_amendment_sha256": base_audit.sha256_file(
                analysis_amendment_path
            ),
            "scoring_attempt_sha256": final_hashes["scoring_attempt_sha256"],
            "scoring_success_sha256": final_hashes["scoring_success_sha256"],
            "audit_attempt_sha256": final_hashes["audit_attempt_sha256"],
            "pre_score_seal_sha256": base_audit.sha256_file(pre_score_seal_path),
            "duplicate_action_pngs_are_failure": False,
            "isolated_duplicate_action_pngs_are_failure": False,
            "full_action_collapse_is_failure": True,
            "duplicate_action_png_policy": (
                "reject_any_action_pair_equal_in_all_registered_blocks"
            ),
            "full_action_collapse_check_passed": True,
            "outcome_details_disclosed": False,
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
    parser.add_argument("--prompts")
    parser.add_argument("--actions", required=True)
    parser.add_argument("--registration", default=str(DEFAULT_REGISTRATION))
    parser.add_argument("--output")
    parser.add_argument("--engineering_smoke", action="store_true")
    parser.add_argument("--score-sealed", action="store_true")
    parser.add_argument(
        "--create-pre-score-seal",
        "--create-seal",
        dest="create_pre_score_seal",
        action="store_true",
    )
    parser.add_argument("--analysis-amendment")
    parser.add_argument("--pre-score-seal")
    args = parser.parse_args()
    if args.engineering_smoke:
        parser.error(
            "analysis-amended structural-control auditor v2 is formal-only; "
            "engineering smoke remains frozen pre-amendment evidence"
        )
    if args.score_sealed:
        if args.create_pre_score_seal:
            parser.error("sealed scoring conflicts with pre-score seal creation")
        if args.output or args.prompts:
            parser.error("sealed scoring does not accept audit input/output options")
        if args.analysis_amendment is None or args.pre_score_seal is None:
            parser.error(
                "--analysis-amendment and --pre-score-seal are required for sealed scoring"
            )
        receipt_path = launch_sealed_scoring(
            args.run_dir,
            args.actions,
            args.registration,
            args.analysis_amendment,
            args.pre_score_seal,
        )
        print(
            json.dumps(
                {"scored": True, "success_receipt": str(receipt_path)},
                sort_keys=True,
            )
        )
        return
    if args.create_pre_score_seal:
        if args.output:
            parser.error("seal creation does not accept an audit output path")
        if args.analysis_amendment is None:
            parser.error("--analysis-amendment is required for seal creation")
        seal = create_pre_score_seal(
            args.run_dir,
            args.actions,
            args.registration,
            args.analysis_amendment,
            output_path=args.pre_score_seal,
        )
        print(
            json.dumps(
                {
                    "created": True,
                    "schema": seal["schema"],
                    "output": str(
                        Path(args.run_dir).resolve()
                        / STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
                    ),
                },
                sort_keys=True,
            )
        )
        return
    if args.prompts is None:
        parser.error("--prompts is required for audit")
    if args.analysis_amendment is None or args.pre_score_seal is None:
        parser.error(
            "--analysis-amendment and --pre-score-seal are required for the formal audit"
        )
    paths = _preflight_formal_audit_paths(
        args.run_dir,
        args.prompts,
        args.actions,
        args.registration,
        args.analysis_amendment,
        args.pre_score_seal,
        args.output,
    )
    run_root = paths["run"]
    output_path = paths["output"]
    with audit_lock(run_root):
        create_audit_attempt_marker(run_root, output_path)
        report = audit_run(
            run_root,
            paths["prompts"],
            paths["actions"],
            registration_actions_path=paths["registration"],
            analysis_amendment_path=paths["analysis_amendment"],
            pre_score_seal_path=paths["pre_score_seal"],
        )
        payload = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _atomic_write(output_path, payload)
    print(json.dumps({"passed": True, "output": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
