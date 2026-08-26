"""Result-blind integrity auditor for the adaptive-oracle engineering smoke.

The auditor verifies image SHA-256 digests, PNG chunks, and the decompressed
filter-prefixed scanline layout.  It never unfilters, reconstructs, displays, or
scores pixels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import traceback
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
import zlib


ROOT = Path(__file__).resolve().parents[1]

import adaptive_oracle_contract as contract
import generate_adaptive_oracle_engineering as generation


AUDIT_SCOPE = "result_blind_generation_integrity_only"
AUDIT_ATTEMPT_NAME = "engineering_audit_attempt.json"
AUDIT_SUCCESS_NAME = "engineering_audit.json"
AUDIT_FAILURE_NAME = "engineering_audit_failure.json"
AUDIT_ATTEMPT_SCHEMA = "adaptive_oracle_engineering_audit_attempt_v1"
AUDIT_SUCCESS_SCHEMA = "adaptive_oracle_engineering_audit_v1"
AUDIT_FAILURE_SCHEMA = "adaptive_oracle_engineering_audit_failure_v1"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_CRITICAL_CHUNKS = frozenset({b"IHDR", b"PLTE", b"IDAT", b"IEND"})
_PNG_BIT_DEPTH = 8
_PNG_COLOR_TYPE_RGB = 2
_PNG_RGB_CHANNELS = 3
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_PCI_BUS_ID = re.compile(r"[0-9A-F]{8}:[0-9A-F]{2}:[0-9A-F]{2}\.[0-7]")
_AUDIT_NAMES = frozenset(
    {AUDIT_ATTEMPT_NAME, AUDIT_SUCCESS_NAME, AUDIT_FAILURE_NAME}
)
_NEGATIVE_OUTCOME_FIELDS = frozenset(
    {
        "scoring",
        "scoring_authorized",
        "quality_inspection",
        "quality_inspection_authorized",
        "quality_outcomes_present",
    }
)
_FORBIDDEN_FIELD_FRAGMENTS = (
    "score",
    "scorer",
    "quality",
    "metric",
    "ranking",
    "reward",
    "preference",
    "aesthetic",
)
_FORBIDDEN_FILE_FRAGMENTS = _FORBIDDEN_FIELD_FRAGMENTS + (
    "clip",
    "topiq",
    "ocr",
)
_SUMMARY_FIELDS = frozenset(
    {
        "task_id",
        "action_id",
        "prompt_row_id",
        "physical_no_op",
        "trajectory_id",
        "initial_latent_sha256",
        "final_latent_sha256",
        "png_sha256",
        "trajectory_chain_sha256",
        "sidecar_sha256",
    }
)
_SUCCESS_TOP_LEVEL = frozenset(
    {
        "attempt.json",
        "config.json",
        "manifest.json",
        "model_stage_evidence.json",
        "runtime_evidence.json",
        "success.json",
        "records",
    }
)
_RUNTIME_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "capture_scope",
        "python_warnings",
        "python_warnings_sha256",
        "logging_records",
        "logging_records_sha256",
        "stderr",
        "stderr_byte_count",
        "stderr_sha256",
        "warnings",
    }
)


class GenerationFailedTerminally(RuntimeError):
    """Raised after a valid terminal generation-failure receipt is observed."""


class _JsonRecord:
    __slots__ = ("value", "raw", "file_sha256")

    def __init__(self, value: dict[str, Any], raw: bytes, file_sha256: str) -> None:
        self.value = value
        self.raw = raw
        self.file_sha256 = file_sha256


class _AuditSession:
    __slots__ = (
        "terminal",
        "generation_attempt",
        "generation_terminal",
        "audit_attempt",
    )

    def __init__(
        self,
        terminal: str,
        generation_attempt: _JsonRecord,
        generation_terminal: _JsonRecord,
        audit_attempt: Optional[dict[str, Any]] = None,
    ) -> None:
        self.terminal = terminal
        self.generation_attempt = generation_attempt
        self.generation_terminal = generation_terminal
        self.audit_attempt = audit_attempt


def _stable_file_metadata(value: os.stat_result) -> tuple[int, ...]:
    # External staging cleanup can make NFS refresh nlink and ctime after the
    # receipt is pinned. Path identity and parsed raw bytes are bound separately.
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
    )


class _PinnedArtifacts:
    """Hold every validated file descriptor through receipt publication."""

    def __init__(self) -> None:
        self._rows: list[tuple[int, int, str, tuple[int, ...], str]] = []

    def add(
        self,
        descriptor: int,
        directory_descriptor: int,
        name: str,
        metadata: os.stat_result,
        label: str,
    ) -> None:
        self._rows.append(
            (
                os.dup(descriptor),
                directory_descriptor,
                name,
                _stable_file_metadata(metadata),
                label,
            )
        )

    def verify(self) -> None:
        for descriptor, directory_descriptor, name, expected, label in self._rows:
            opened = os.fstat(descriptor)
            current = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or _stable_file_metadata(opened) != expected
                or _stable_file_metadata(current) != expected
            ):
                raise ValueError(f"{label} changed before audit publication")

    def close(self) -> list[BaseException]:
        errors = []
        while self._rows:
            descriptor, _, _, _, _ = self._rows.pop()
            try:
                os.close(descriptor)
            except BaseException as exc:
                errors.append(exc)
        return errors


def _consume_regular_file(
    path: Path,
    label: str,
    consumer: Callable[[int], Any],
    *,
    directory_descriptor: Optional[int] = None,
    pins: Optional[_PinnedArtifacts] = None,
) -> Any:
    """Consume one stable regular file through a single non-following descriptor."""

    descriptor = -1
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        target: str | os.PathLike[str] = (
            path.name if directory_descriptor is not None else path
        )
        descriptor = os.open(target, flags, dir_fd=directory_descriptor)
        before = os.fstat(descriptor)
        path_before = os.stat(
            target,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(path_before.st_mode)
            or (before.st_dev, before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
        ):
            raise ValueError(f"{label} must be a regular non-symlink file")

        result = consumer(descriptor)

        after = os.fstat(descriptor)
        path_after = os.stat(
            target,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _stable_file_metadata(after) != _stable_file_metadata(before)
            or not stat.S_ISREG(path_after.st_mode)
            or (path_after.st_dev, path_after.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"{label} changed while it was read")
        if pins is not None:
            if directory_descriptor is None:
                raise ValueError("pinned artifact reads require a pinned directory")
            pins.add(
                descriptor,
                directory_descriptor,
                path.name,
                after,
                label,
            )
        return result
    except OSError as exc:
        raise ValueError(
            f"{label} must be a stable regular non-symlink file"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def sha256_file(
    path: Path,
    *,
    directory_descriptor: Optional[int] = None,
    pins: Optional[_PinnedArtifacts] = None,
) -> str:
    def consume(descriptor: int) -> str:
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)

    return _consume_regular_file(
        path,
        f"hashed file {path}",
        consume,
        directory_descriptor=directory_descriptor,
        pins=pins,
    )


def _require_regular_file(
    path: Path,
    label: str,
    *,
    directory_descriptor: Optional[int] = None,
    pins: Optional[_PinnedArtifacts] = None,
) -> None:
    _consume_regular_file(
        path,
        label,
        lambda descriptor: None,
        directory_descriptor=directory_descriptor,
        pins=pins,
    )


def _read_regular_bytes(
    path: Path,
    label: str,
    *,
    directory_descriptor: Optional[int] = None,
    pins: Optional[_PinnedArtifacts] = None,
) -> bytes:
    def consume(descriptor: int) -> bytes:
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    return _consume_regular_file(
        path,
        label,
        consume,
        directory_descriptor=directory_descriptor,
        pins=pins,
    )


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value!r}")


def _parse_canonical_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain one JSON object")
    if raw != contract.canonical_json_bytes(parsed) + b"\n":
        raise ValueError(f"{label} is not canonical JSON")
    return parsed


def load_canonical_json_record(
    path: Path,
    *,
    label: str,
    directory_descriptor: Optional[int] = None,
    pins: Optional[_PinnedArtifacts] = None,
) -> tuple[dict[str, Any], bytes]:
    """Load canonical JSON and retain the exact bytes used for validation."""

    raw = _read_regular_bytes(
        path,
        label,
        directory_descriptor=directory_descriptor,
        pins=pins,
    )
    return _parse_canonical_json(raw, label), raw


def load_canonical_json(
    path: Path,
    *,
    label: str,
    directory_descriptor: Optional[int] = None,
    pins: Optional[_PinnedArtifacts] = None,
) -> dict[str, Any]:
    return load_canonical_json_record(
        path,
        label=label,
        directory_descriptor=directory_descriptor,
        pins=pins,
    )[0]


def _load_json_record(
    path: Path,
    *,
    label: str,
    directory_descriptor: int,
    pins: _PinnedArtifacts,
) -> _JsonRecord:
    value, raw = load_canonical_json_record(
        path,
        label=label,
        directory_descriptor=directory_descriptor,
        pins=pins,
    )
    return _JsonRecord(
        value=value,
        raw=raw,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _json_record_from_raw(raw: bytes, label: str) -> _JsonRecord:
    return _JsonRecord(
        value=_parse_canonical_json(raw, label),
        raw=raw,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _canonical_equal(observed: Any, expected: Any, label: str) -> None:
    if contract.canonical_json_bytes(observed) != contract.canonical_json_bytes(
        expected
    ):
        raise ValueError(f"{label} differs from the exact expected value")


def _require_exact_fields(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"{label} fields differ: expected {sorted(expected)}, got {observed}"
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _validate_runtime_evidence(
    value: Any, *, require_warning_free: bool
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(_RUNTIME_EVIDENCE_FIELDS):
        raise ValueError("runtime evidence fields differ")
    if value["schema"] != generation.RUNTIME_EVIDENCE_SCHEMA:
        raise ValueError("runtime evidence schema differs")
    if value["capture_scope"] != (
        "runtime_load_environment_pipeline_and_generation"
    ):
        raise ValueError("runtime evidence capture scope differs")

    warning_rows = value["python_warnings"]
    if not isinstance(warning_rows, list):
        raise ValueError("runtime Python warnings must be a list")
    for row in warning_rows:
        if not isinstance(row, dict) or set(row) != {
            "category",
            "filename",
            "lineno",
            "message",
        }:
            raise ValueError("runtime Python warning record fields differ")
        if (
            not isinstance(row["category"], str)
            or not isinstance(row["filename"], str)
            or isinstance(row["lineno"], bool)
            or not isinstance(row["lineno"], int)
            or row["lineno"] < 0
            or not isinstance(row["message"], str)
        ):
            raise ValueError("runtime Python warning record is invalid")
    expected_warning_hash = contract.canonical_sha256(warning_rows)
    if value["python_warnings_sha256"] != expected_warning_hash:
        raise ValueError("runtime Python warning hash differs")

    log_rows = value["logging_records"]
    if not isinstance(log_rows, list):
        raise ValueError("runtime logging records must be a list")
    for row in log_rows:
        if not isinstance(row, dict) or set(row) != {
            "level",
            "level_number",
            "logger",
            "message",
        }:
            raise ValueError("runtime logging record fields differ")
        if (
            not isinstance(row["level"], str)
            or isinstance(row["level_number"], bool)
            or not isinstance(row["level_number"], int)
            or not isinstance(row["logger"], str)
            or not isinstance(row["message"], str)
        ):
            raise ValueError("runtime logging record is invalid")
    expected_log_hash = contract.canonical_sha256(log_rows)
    if value["logging_records_sha256"] != expected_log_hash:
        raise ValueError("runtime logging record hash differs")

    stderr_text = value["stderr"]
    if not isinstance(stderr_text, str):
        raise ValueError("runtime stderr evidence must be a string")
    stderr_bytes = stderr_text.encode("utf-8", errors="surrogateescape")
    if value["stderr_byte_count"] != len(stderr_bytes):
        raise ValueError("runtime stderr byte count differs")
    if value["stderr_sha256"] != hashlib.sha256(stderr_bytes).hexdigest():
        raise ValueError("runtime stderr hash differs")

    warning_record = value["warnings"]
    expected_warning_fields = {
        "count",
        "python_warning_count",
        "logging_warning_or_higher_count",
        "stderr_warning_line_count",
    }
    if not isinstance(warning_record, dict) or set(warning_record) != expected_warning_fields:
        raise ValueError("runtime warning-count fields differ")
    warning_log_count = sum(row["level_number"] >= 30 for row in log_rows)
    stderr_warning_line_count = contract.stderr_warning_line_count(stderr_text)
    expected_counts = {
        "count": len(warning_rows) + warning_log_count + stderr_warning_line_count,
        "python_warning_count": len(warning_rows),
        "logging_warning_or_higher_count": warning_log_count,
        "stderr_warning_line_count": stderr_warning_line_count,
    }
    if warning_record != expected_counts:
        raise ValueError("runtime warning counts differ from captured evidence")
    if require_warning_free and warning_record["count"] != 0:
        raise ValueError("engineering generation runtime was not warning-free")
    return copy.deepcopy(value)


def _validate_model_stage_evidence(
    value: Any,
    *,
    expected_stage: Optional[Mapping[str, Any]] = None,
    expected_verifications: Optional[Sequence[Mapping[str, Any]]] = None,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate a private-model staging, verification, and cleanup transcript."""

    record = _require_exact_fields(
        value,
        frozenset(
            {
                "schema",
                "status",
                "trust_boundary",
                "stage",
                "verifications",
                "cleanup",
                "cleanup_failure",
            }
        ),
        "model-stage evidence",
    )
    if record["schema"] != generation.MODEL_STAGE_EVIDENCE_SCHEMA:
        raise ValueError("model-stage evidence schema differs")
    if record["status"] not in {"removed", "cleanup_failed_terminal"}:
        raise ValueError("model-stage evidence status differs")
    if record["trust_boundary"] != {
        "same_uid_noninterference_required_between_verification_and_open": False,
        "loader_root_pinned_by_procfs_fd": True,
        "pre_post_content_and_object_identity_binding": True,
        "network_access_used": False,
        "load_source": "pinned_procfs_fd_private_regular_file_stage",
    }:
        raise ValueError("model-stage trust boundary differs")

    raw_stage = record["stage"]
    partial_stage = (
        isinstance(raw_stage, Mapping)
        and raw_stage.get("status") == "staging_failed_cleanup_pending"
    )
    if partial_stage:
        stage = _require_exact_fields(
            raw_stage,
            frozenset({"schema", "status", "path", "parent", "root_identity"}),
            "partial model stage",
        )
        if stage["schema"] != generation.model_snapshot.MODEL_STAGE_SCHEMA:
            raise ValueError("partial model stage schema differs")
    else:
        stage = _require_exact_fields(
            raw_stage,
            frozenset(
                {
                    "schema",
                    "status",
                    "path",
                    "parent",
                    "manifest",
                    "manifest_sha256",
                    "loaded_file_count",
                    "tree_sha256",
                    "root_identity",
                    "source_snapshot",
                    "source_snapshot_sha256",
                }
            ),
            "model stage",
        )
        model = generation._expected_model()
        if (
            stage["schema"] != generation.model_snapshot.MODEL_STAGE_SCHEMA
            or stage["status"] != "staged_verified_read_only"
            or stage["manifest"]
            != generation.model_snapshot.expected_model_manifest()
            or stage["manifest_sha256"] != model["snapshot_manifest_sha256"]
            or stage["loaded_file_count"] != model["snapshot_loaded_file_count"]
        ):
            raise ValueError("model stage differs from the frozen snapshot")
        for field in ("tree_sha256", "source_snapshot_sha256"):
            _require_sha256(stage[field], f"model stage {field}")
        if generation.model_snapshot.canonical_sha256(
            stage["source_snapshot"]
        ) != stage["source_snapshot_sha256"]:
            raise ValueError("model stage source-snapshot hash differs")

    verifications = record["verifications"]
    if not isinstance(verifications, list) or not 0 <= len(verifications) <= 4:
        raise ValueError("model stage has an invalid verification transcript")
    if partial_stage and verifications:
        raise ValueError("partial model stage cannot have verification records")
    if require_complete and (partial_stage or len(verifications) != 4):
        raise ValueError("model stage must have exactly four verification records")
    verification_fields = frozenset(
        {
            "schema",
            "status",
            "path",
            "manifest_sha256",
            "loaded_file_count",
            "tree_sha256",
            "root_identity",
        }
    )
    for index, raw in enumerate(verifications):
        verification = _require_exact_fields(
            raw, verification_fields, f"model-stage verification {index}"
        )
        if (
            verification["schema"]
            != generation.model_snapshot.MODEL_STAGE_VERIFICATION_SCHEMA
            or verification["status"] != "verified_unchanged"
            or verification["path"] != stage["path"]
            or verification["manifest_sha256"] != stage["manifest_sha256"]
            or verification["loaded_file_count"] != stage["loaded_file_count"]
            or verification["tree_sha256"] != stage["tree_sha256"]
            or verification["root_identity"] != stage["root_identity"]
        ):
            raise ValueError(f"model-stage verification {index} differs")

    if expected_stage is not None:
        _canonical_equal(stage, expected_stage, "embedded/file model stage")
    if expected_verifications is not None:
        _canonical_equal(
            verifications,
            list(expected_verifications),
            "embedded/file model-stage verifications",
        )

    cleanup = record["cleanup"]
    cleanup_failure = record["cleanup_failure"]
    if record["status"] == "removed":
        cleanup_record = _require_exact_fields(
            cleanup,
            frozenset(
                {
                    "schema",
                    "status",
                    "path",
                    "manifest_sha256",
                    "loaded_file_count",
                    "root_identity",
                }
            ),
            "model-stage cleanup",
        )
        if cleanup_failure is not None:
            raise ValueError("removed model stage contains a cleanup failure")
        expected_manifest_sha256 = (
            None if partial_stage else stage["manifest_sha256"]
        )
        expected_loaded_file_count = (
            None if partial_stage else stage["loaded_file_count"]
        )
        if (
            cleanup_record["schema"]
            != generation.model_snapshot.MODEL_STAGE_CLEANUP_SCHEMA
            or cleanup_record["status"] != "removed"
            or cleanup_record["path"] != stage["path"]
            or cleanup_record["manifest_sha256"] != expected_manifest_sha256
            or cleanup_record["loaded_file_count"]
            != expected_loaded_file_count
            or cleanup_record["root_identity"] != stage["root_identity"]
        ):
            raise ValueError("model-stage cleanup differs")
    else:
        if cleanup is not None:
            raise ValueError("failed model-stage cleanup unexpectedly has a receipt")
        failure = _require_exact_fields(
            cleanup_failure,
            frozenset({"exception_type", "exception_message"}),
            "model-stage cleanup failure",
        )
        if not isinstance(failure["exception_type"], str) or not failure[
            "exception_type"
        ]:
            raise ValueError("model-stage cleanup exception type is empty")
        if not isinstance(failure["exception_message"], str):
            raise ValueError("model-stage cleanup exception message is invalid")
    return copy.deepcopy(dict(record))


def _validate_auditor_launcher_execution(validated: Mapping[str, Any]) -> None:
    evidence = validated.get("launcher_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("auditor launch omitted launcher evidence")
    reviewed_commit = validated.get("reviewed_commit")
    source_name = "engineering_auditor"
    source_path = generation.SOURCE_PATHS[source_name]
    expected_origin = f"<git-blob:{reviewed_commit}:{source_path}>"
    if (
        __name__ != "audit_adaptive_oracle_engineering"
        or __file__ != expected_origin
        or globals().get("__cached__") is not None
    ):
        raise RuntimeError("engineering auditor was not executed from its reviewed Git blob")
    module_execution = evidence.get("module_execution")
    row = _require_exact_fields(
        None if not isinstance(module_execution, Mapping) else module_execution.get(__name__),
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
        "launcher engineering-auditor execution",
    )
    source_hashes = validated.get("source_hashes")
    pair = None if not isinstance(source_hashes, Mapping) else source_hashes.get(source_name)
    if (
        not isinstance(pair, Mapping)
        or row["source_name"] != source_name
        or row["path"] != source_path
        or row["origin"] != expected_origin
        or row["sha256"] != pair.get("sha256")
        or row["loader_id"] != generation.LAUNCH_LOADER_ID
        or row["execution_count"] != 1
        or row["cached"] is not None
    ):
        raise RuntimeError("reviewed engineering-auditor execution evidence differs")


def _scan_forbidden_fields(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold().replace("-", "_")
            if key in _NEGATIVE_OUTCOME_FIELDS:
                if child is not False:
                    raise ValueError(f"{path}.{raw_key} must remain false")
            elif any(fragment in key for fragment in _FORBIDDEN_FIELD_FRAGMENTS):
                raise ValueError(
                    f"{path} contains forbidden score/quality field {raw_key!r}"
                )
            _scan_forbidden_fields(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_forbidden_fields(child, f"{path}[{index}]")


def _directory_names(descriptor: int, label: str) -> frozenset[str]:
    try:
        names = os.listdir(descriptor)
    except OSError as exc:
        raise ValueError(f"cannot enumerate {label}") from exc
    if any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise ValueError(f"{label} contains an invalid basename")
    return frozenset(names)


def _require_entry_kind(
    directory_descriptor: int,
    name: str,
    *,
    label: str,
    directory: bool,
) -> None:
    try:
        observed = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError(f"{label} is missing or unstable") from exc
    expected = stat.S_ISDIR(observed.st_mode) if directory else stat.S_ISREG(
        observed.st_mode
    )
    if not expected:
        kind = "directory" if directory else "regular non-symlink file"
        raise ValueError(f"{label} must be a {kind}")


def _scan_forbidden_artifacts(tree: _PinnedRunTree) -> None:
    inventories = [(tree.run_descriptor, "")]
    if tree.records_descriptor is not None:
        inventories.append((tree.records_descriptor, "records/"))
    for descriptor, prefix in inventories:
        for name in _directory_names(descriptor, f"{prefix or 'run '}directory"):
            observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = f"{prefix}{name}"
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"run tree contains a symlink: {relative}")
            folded = name.casefold().replace("-", "_")
            if any(fragment in folded for fragment in _FORBIDDEN_FILE_FRAGMENTS):
                raise ValueError(
                    "run tree contains a forbidden score/quality artifact: "
                    f"{relative}"
                )


def _validate_png_scanlines(path: Path, compressed: bytes) -> int:
    row_size = 1 + contract.WIDTH * _PNG_RGB_CHANNELS
    expected_size = contract.HEIGHT * row_size
    decoder = zlib.decompressobj()
    try:
        scanlines = decoder.decompress(compressed, expected_size + 1)
        if decoder.unconsumed_tail:
            raise ValueError(
                f"{path.name}: PNG decompressed scanline byte length differs"
            )
        scanlines += decoder.flush()
    except zlib.error as exc:
        raise ValueError(f"{path.name}: PNG IDAT zlib stream is invalid") from exc
    if decoder.unused_data:
        raise ValueError(f"{path.name}: PNG IDAT zlib stream has trailing data")
    if not decoder.eof:
        raise ValueError(f"{path.name}: PNG IDAT zlib stream is truncated")
    if len(scanlines) != expected_size:
        raise ValueError(f"{path.name}: PNG decompressed scanline byte length differs")
    for row_index in range(contract.HEIGHT):
        filter_byte = scanlines[row_index * row_size]
        if filter_byte > 4:
            raise ValueError(
                f"{path.name}: PNG row {row_index} has an invalid filter byte"
            )
    return len(scanlines)


def inspect_png_container(
    path: Path,
    *,
    directory_descriptor: Optional[int] = None,
    pins: Optional[_PinnedArtifacts] = None,
) -> dict[str, Any]:
    """Validate an exact RGB PNG without reconstructing or inspecting pixels."""

    raw = _read_regular_bytes(
        path,
        "PNG record",
        directory_descriptor=directory_descriptor,
        pins=pins,
    )
    if len(raw) < len(_PNG_SIGNATURE) or raw[:8] != _PNG_SIGNATURE:
        raise ValueError(f"{path.name}: invalid PNG signature")

    offset = len(_PNG_SIGNATURE)
    chunk_count = 0
    seen_ihdr = False
    seen_plte = False
    seen_idat = False
    idat_closed = False
    seen_iend = False
    idat_payloads = []
    width = height = bit_depth = color_type = None

    while offset < len(raw):
        if seen_iend:
            raise ValueError(f"{path.name}: PNG has trailing bytes after IEND")
        if len(raw) - offset < 8:
            raise ValueError(f"{path.name}: truncated PNG chunk header")

        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        if length > 0x7FFFFFFF:
            raise ValueError(f"{path.name}: PNG chunk length exceeds the format limit")
        if any(
            value not in range(ord("A"), ord("Z") + 1)
            and value not in range(ord("a"), ord("z") + 1)
            for value in chunk_type
        ):
            raise ValueError(f"{path.name}: PNG chunk type is invalid")
        if chunk_type[2] & 0x20:
            raise ValueError(f"{path.name}: PNG chunk reserved bit is invalid")

        data_start = offset + 8
        data_end = data_start + length
        chunk_end = data_end + 4
        if chunk_end > len(raw):
            raise ValueError(f"{path.name}: truncated PNG chunk")
        payload = raw[data_start:data_end]
        observed_crc = struct.unpack(">I", raw[data_end:chunk_end])[0]
        expected_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        chunk_label = chunk_type.decode("ascii")
        if observed_crc != expected_crc:
            raise ValueError(f"{path.name}: PNG {chunk_label} CRC differs")

        if chunk_count == 0 and chunk_type != b"IHDR":
            raise ValueError(f"{path.name}: PNG must begin with one 13-byte IHDR")
        if chunk_type == b"IHDR":
            if seen_ihdr or chunk_count != 0 or length != 13:
                raise ValueError(f"{path.name}: PNG must begin with one 13-byte IHDR")
            seen_ihdr = True
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if (width, height) != (contract.WIDTH, contract.HEIGHT):
                raise ValueError(
                    f"{path.name}: PNG IHDR is {width}x{height}, expected "
                    f"{contract.WIDTH}x{contract.HEIGHT}"
                )
            if compression != 0 or filtering != 0:
                raise ValueError(f"{path.name}: PNG IHDR methods are invalid")
            if bit_depth != _PNG_BIT_DEPTH or color_type != _PNG_COLOR_TYPE_RGB:
                raise ValueError(f"{path.name}: PNG must use exact 8-bit RGB encoding")
            if interlace != 0:
                raise ValueError(f"{path.name}: PNG must be non-interlaced")
        elif not seen_ihdr:
            raise ValueError(f"{path.name}: PNG omitted IHDR")
        elif chunk_type == b"PLTE":
            if seen_plte or seen_idat or length == 0 or length % 3 or length > 768:
                raise ValueError(f"{path.name}: PNG PLTE structure is invalid")
            seen_plte = True
        elif chunk_type == b"IDAT":
            if idat_closed:
                raise ValueError(f"{path.name}: PNG IDAT chunks are not consecutive")
            seen_idat = True
            idat_payloads.append(payload)
        elif chunk_type == b"IEND":
            if length != 0 or not seen_idat:
                raise ValueError(f"{path.name}: PNG IEND structure is invalid")
            seen_iend = True
        elif chunk_type[0] & 0x20 == 0 and chunk_type not in _PNG_CRITICAL_CHUNKS:
            raise ValueError(f"{path.name}: PNG contains an unknown critical chunk")

        if seen_idat and chunk_type not in {b"IDAT", b"IEND"}:
            idat_closed = True
        offset = chunk_end
        chunk_count += 1

    if not seen_iend:
        raise ValueError(f"{path.name}: PNG omitted IEND")
    if offset != len(raw):
        raise ValueError(f"{path.name}: PNG has trailing bytes after IEND")
    decompressed_size = _validate_png_scanlines(path, b"".join(idat_payloads))
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "width": width,
        "height": height,
        "chunk_count": chunk_count,
        "idat_chunk_count": len(idat_payloads),
        "decompressed_scanline_byte_count": decompressed_size,
    }


def atomic_create_json(
    path: Path,
    value: Any,
    *,
    directory_descriptor: Optional[int] = None,
    staging_directory_descriptor: Optional[int] = None,
    ownership_candidates: Optional[list[tuple[int, int]]] = None,
) -> tuple[int, int]:
    """Publish canonical JSON without replacing an existing receipt."""

    return generation.atomic_create_bytes(
        path,
        contract.canonical_json_bytes(value) + b"\n",
        directory_descriptor=directory_descriptor,
        staging_directory_descriptor=staging_directory_descriptor,
        ownership_candidates=ownership_candidates,
    )


class _PinnedRunTree:
    def __init__(
        self,
        *,
        root: Path,
        run: Path,
        root_descriptor: int,
        directory_links: Sequence[tuple[int, str, int, str]],
        run_parent_descriptor: int,
        run_descriptor: int,
        records_descriptor: Optional[int],
    ) -> None:
        self.root = root
        self.run = run
        self.root_descriptor = root_descriptor
        self.directory_links = list(directory_links)
        self.run_parent_descriptor = run_parent_descriptor
        self.run_descriptor = run_descriptor
        self.records_descriptor = records_descriptor
        self.pins = _PinnedArtifacts()
        self.run_inventory: Optional[frozenset[str]] = None
        self.records_inventory: Optional[frozenset[str]] = None
        self.published_identities: dict[str, tuple[int, int]] = {}

    def snapshot_inventories(self) -> None:
        self.run_inventory = _directory_names(
            self.run_descriptor, "engineering run directory"
        )
        self.records_inventory = (
            None
            if self.records_descriptor is None
            else _directory_names(
                self.records_descriptor, "engineering records directory"
            )
        )

    def add_published_entry(
        self, name: str, identity: tuple[int, int]
    ) -> None:
        if self.run_inventory is None:
            raise RuntimeError("run inventory must be pinned before publication")
        if name in self.run_inventory or name in self.published_identities:
            raise FileExistsError(f"audit receipt already exists: {name}")
        self.run_inventory = frozenset((*self.run_inventory, name))
        self.published_identities[name] = identity

    def verify_directories(self) -> None:
        opened_root = os.fstat(self.root_descriptor)
        current_root = os.stat(self.root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened_root.st_mode)
            or not stat.S_ISDIR(current_root.st_mode)
            or generation._object_identity(opened_root)
            != generation._object_identity(current_root)
        ):
            raise ValueError("engineering repository root identity changed")
        for parent, name, child, label in self.directory_links:
            generation._verify_directory_entry(parent, name, child, label)
        generation._verify_directory_entry(
            self.run_parent_descriptor,
            self.run.name,
            self.run_descriptor,
            "engineering run directory",
        )
        if self.records_descriptor is not None:
            generation._verify_directory_entry(
                self.run_descriptor,
                "records",
                self.records_descriptor,
                "engineering records directory",
            )

    def verify(self) -> None:
        self.verify_directories()
        if self.run_inventory is not None:
            observed_run = _directory_names(
                self.run_descriptor, "engineering run directory"
            )
            if observed_run != self.run_inventory:
                raise ValueError("engineering run directory inventory changed")
        if self.records_inventory is not None:
            if self.records_descriptor is None:
                raise ValueError("engineering records directory disappeared")
            observed_records = _directory_names(
                self.records_descriptor, "engineering records directory"
            )
            if observed_records != self.records_inventory:
                raise ValueError("engineering records directory inventory changed")
        for name, expected in self.published_identities.items():
            observed = os.stat(
                name,
                dir_fd=self.run_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(observed.st_mode)
                or generation._object_identity(observed) != expected
            ):
                raise ValueError(f"published audit receipt identity changed: {name}")
        self.pins.verify()

    def close(self) -> None:
        self.pins.close()
        descriptors = [
            self.records_descriptor,
            self.run_descriptor,
            *(child for _, _, child, _ in reversed(self.directory_links)),
            self.root_descriptor,
        ]
        closed: set[int] = set()
        for descriptor in descriptors:
            if descriptor is not None and descriptor not in closed:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                closed.add(descriptor)


def _pin_run_tree(
    root: Path,
    relative: Path,
) -> _PinnedRunTree:
    root = Path(os.path.abspath(root))
    root_descriptor = generation._open_pinned_directory(
        root, "engineering repository root"
    )
    directory_links: list[tuple[int, str, int, str]] = []
    current_descriptor = root_descriptor
    run_descriptor = -1
    records_descriptor: Optional[int] = None
    try:
        for index, part in enumerate(relative.parts[:-1]):
            child = generation._open_pinned_directory_at(
                current_descriptor,
                part,
                f"engineering run parent component {index}",
            )
            directory_links.append(
                (
                    current_descriptor,
                    part,
                    child,
                    f"engineering run parent component {index}",
                )
            )
            current_descriptor = child
        run_descriptor = generation._open_pinned_directory_at(
            current_descriptor,
            relative.name,
            "engineering run directory",
        )
        try:
            records_descriptor = generation._open_pinned_directory_at(
                run_descriptor,
                "records",
                "engineering records directory",
            )
        except ValueError:
            try:
                os.stat("records", dir_fd=run_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                records_descriptor = None
            else:
                raise
        return _PinnedRunTree(
            root=root,
            run=root.joinpath(*relative.parts),
            root_descriptor=root_descriptor,
            directory_links=directory_links,
            run_parent_descriptor=current_descriptor,
            run_descriptor=run_descriptor,
            records_descriptor=records_descriptor,
        )
    except BaseException:
        if records_descriptor is not None:
            os.close(records_descriptor)
        if run_descriptor >= 0:
            os.close(run_descriptor)
        for _, _, descriptor, _ in reversed(directory_links):
            os.close(descriptor)
        os.close(root_descriptor)
        raise


def _resolve_run_dir(
    run_dir: str | os.PathLike[str], repo_root: str | os.PathLike[str]
) -> tuple[Path, _PinnedRunTree]:
    root = Path(repo_root).resolve(strict=True)
    expected = root.joinpath(*Path(generation.OUTPUT_DIR).parts)
    supplied = Path(os.path.abspath(run_dir))
    if supplied != expected:
        raise ValueError(f"engineering auditor requires the exact run {expected}")
    return root, _pin_run_tree(root, Path(generation.OUTPUT_DIR))


def _terminal_state(tree: _PinnedRunTree) -> str:
    names = _directory_names(tree.run_descriptor, "engineering run directory")
    success = "success.json" in names
    failure = "failure.json" in names
    if success == failure:
        raise ValueError(
            "generation must have exactly one atomic success or failure receipt"
        )
    return "success" if success else "failure"


def _attempt_payload(
    terminal: str,
    *,
    generation_attempt_file_sha256: str,
    generation_terminal_file_sha256: str,
    auditor_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": AUDIT_ATTEMPT_SCHEMA,
        "status": "started_one_shot_no_resume",
        "one_shot": True,
        "scope": AUDIT_SCOPE,
        "experiment_id": contract.EXPERIMENT_ID,
        "run_dir": generation.OUTPUT_DIR,
        "generation_terminal": terminal,
        "generation_attempt_file_sha256": _require_sha256(
            generation_attempt_file_sha256,
            "generation attempt file SHA-256",
        ),
        "generation_terminal_file_sha256": _require_sha256(
            generation_terminal_file_sha256,
            "generation terminal file SHA-256",
        ),
        "auditor_sha256": _require_sha256(auditor_sha256, "auditor SHA-256"),
}


def _validate_attempt(
    attempt: Mapping[str, Any], validated: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(attempt))
    _canonical_equal(
        normalized,
        generation._attempt_record(validated),
        "generation attempt",
    )
    _scan_forbidden_fields(normalized, "generation attempt")
    return normalized


def _read_generation_session(
    tree: _PinnedRunTree,
    validated: Mapping[str, Any],
    *,
    terminal: Optional[str] = None,
) -> _AuditSession:
    observed_terminal = _terminal_state(tree)
    if terminal is not None and terminal != observed_terminal:
        raise ValueError("generation terminal state changed before audit")
    attempt = _load_json_record(
        tree.run / "attempt.json",
        label="generation attempt",
        directory_descriptor=tree.run_descriptor,
        pins=tree.pins,
    )
    _validate_attempt(attempt.value, validated)
    terminal_record = _load_json_record(
        tree.run / f"{observed_terminal}.json",
        label=f"generation {observed_terminal} receipt",
        directory_descriptor=tree.run_descriptor,
        pins=tree.pins,
    )
    return _AuditSession(
        terminal=observed_terminal,
        generation_attempt=attempt,
        generation_terminal=terminal_record,
    )


def _publish_audit_json(
    tree: _PinnedRunTree,
    name: str,
    value: Mapping[str, Any],
    *,
    ownership: Optional[list[tuple[int, int]]] = None,
) -> tuple[int, int]:
    payload = contract.canonical_json_bytes(value) + b"\n"
    tree.verify()
    if generation._entry_exists(tree.run_descriptor, name):
        raise FileExistsError(f"audit receipt already exists: {name}")
    identity = atomic_create_json(
        tree.run / name,
        value,
        directory_descriptor=tree.run_descriptor,
        staging_directory_descriptor=tree.run_parent_descriptor,
        ownership_candidates=ownership,
    )
    tree.add_published_entry(name, identity)
    observed = _read_regular_bytes(
        tree.run / name,
        f"published audit receipt {name}",
        directory_descriptor=tree.run_descriptor,
        pins=tree.pins,
    )
    if observed != payload:
        raise RuntimeError(f"published audit receipt bytes differ: {name}")
    tree.verify()
    return identity


def _publish_audit_success(
    tree: _PinnedRunTree,
    value: Mapping[str, Any],
    *,
    ownership: list[tuple[int, int]],
) -> tuple[int, int]:
    tree.verify()
    for name in (AUDIT_SUCCESS_NAME, AUDIT_FAILURE_NAME):
        if generation._entry_exists(tree.run_descriptor, name):
            raise FileExistsError(f"audit terminal receipt already exists: {name}")
    return atomic_create_json(
        tree.run / AUDIT_SUCCESS_NAME,
        value,
        directory_descriptor=tree.run_descriptor,
        staging_directory_descriptor=tree.run_parent_descriptor,
        ownership_candidates=ownership,
    )


def _publish_audit_failure(
    tree: _PinnedRunTree,
    value: Mapping[str, Any],
) -> None:
    """Publish a non-pass terminal even when evidence verification has failed."""

    payload = contract.canonical_json_bytes(value) + b"\n"
    tree.verify_directories()
    if generation._entry_exists(tree.run_descriptor, AUDIT_FAILURE_NAME):
        return
    identity = atomic_create_json(
        tree.run / AUDIT_FAILURE_NAME,
        value,
        directory_descriptor=tree.run_descriptor,
        staging_directory_descriptor=tree.run_parent_descriptor,
    )
    tree.add_published_entry(AUDIT_FAILURE_NAME, identity)
    observed = _read_regular_bytes(
        tree.run / AUDIT_FAILURE_NAME,
        "published audit failure receipt",
        directory_descriptor=tree.run_descriptor,
    )
    if observed != payload:
        raise RuntimeError("published audit failure receipt bytes differ")
    tree.verify_directories()


def _preflight_audit(
    tree: _PinnedRunTree,
    validated: Mapping[str, Any],
    *,
    auditor_sha256: str,
) -> _AuditSession:
    tree.snapshot_inventories()
    if tree.run_inventory is None:
        raise RuntimeError("engineering run inventory was not pinned")
    if tree.run_inventory.intersection(_AUDIT_NAMES):
        raise FileExistsError("engineering audit is one-shot and cannot resume or retry")
    terminal = _terminal_state(tree)
    generation_attempt_raw = _read_regular_bytes(
        tree.run / "attempt.json",
        "generation attempt",
        directory_descriptor=tree.run_descriptor,
        pins=tree.pins,
    )
    generation_terminal_raw = _read_regular_bytes(
        tree.run / f"{terminal}.json",
        f"generation {terminal} receipt",
        directory_descriptor=tree.run_descriptor,
        pins=tree.pins,
    )
    attempt = _attempt_payload(
        terminal,
        generation_attempt_file_sha256=hashlib.sha256(
            generation_attempt_raw
        ).hexdigest(),
        generation_terminal_file_sha256=hashlib.sha256(
            generation_terminal_raw
        ).hexdigest(),
        auditor_sha256=auditor_sha256,
    )
    attempt_ownership: list[tuple[int, int]] = []
    try:
        _publish_audit_json(
            tree,
            AUDIT_ATTEMPT_NAME,
            attempt,
            ownership=attempt_ownership,
        )
        generation_attempt = _json_record_from_raw(
            generation_attempt_raw,
            "generation attempt",
        )
        _validate_attempt(generation_attempt.value, validated)
        generation_terminal = _json_record_from_raw(
            generation_terminal_raw,
            f"generation {terminal} receipt",
        )
    except BaseException as exc:
        try:
            observed_attempt = generation._entry_metadata(
                tree.run_descriptor, AUDIT_ATTEMPT_NAME
            )
        except BaseException as existence_error:
            generation._add_exception_note(
                exc,
                "audit-attempt reconciliation failed: "
                f"{type(existence_error).__name__}: {existence_error}",
            )
            observed_attempt = None
        attempt_owned = (
            len(attempt_ownership) == 1
            and observed_attempt is not None
            and stat.S_ISREG(observed_attempt.st_mode)
            and generation._object_identity(observed_attempt)
            == attempt_ownership[0]
        )
        if attempt_owned:
            try:
                _publish_audit_failure(tree, _failure_payload(attempt, exc))
            except BaseException as receipt_error:
                generation._add_exception_note(
                    exc,
                    "audit preflight failure receipt also failed: "
                    f"{type(receipt_error).__name__}: {receipt_error}",
                )
        raise
    return _AuditSession(
        terminal=terminal,
        generation_attempt=generation_attempt,
        generation_terminal=generation_terminal,
        audit_attempt=attempt,
    )


def _validated_design(validated: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    design = validated.get("design")
    if not isinstance(design, dict):
        raise ValueError("authorization validator did not return a design")
    tasks = design.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != contract.TOTAL_TASK_COUNT:
        raise ValueError("authorized design must contain exactly 165 tasks")
    if design.get("task_count") != contract.TOTAL_TASK_COUNT:
        raise ValueError("authorized design task_count differs")
    if design.get("tasks_per_prompt") != contract.TASKS_PER_PROMPT:
        raise ValueError("authorized design tasks_per_prompt differs")
    if design.get("tasks_sha256") != contract.canonical_sha256(tasks):
        raise ValueError("authorized design task-list hash differs")
    _require_sha256(design.get("design_sha256"), "authorized design SHA-256")
    task_ids = [task.get("task_id") for task in tasks if isinstance(task, Mapping)]
    if len(task_ids) != len(tasks) or any(
        not isinstance(task_id, str) or not task_id for task_id in task_ids
    ):
        raise ValueError("authorized design task ids are empty or duplicated")
    if len(set(task_ids)) != len(tasks):
        raise ValueError("authorized design task ids are empty or duplicated")
    for task_id in task_ids:
        if Path(task_id).name != task_id:
            raise ValueError("authorized task id is not a safe record basename")
    return design, tasks


def _expected_run_config(
    validated: Mapping[str, Any],
    design: Mapping[str, Any],
    runtime_evidence_sha256: str,
    runtime_environment: Mapping[str, Any],
    verified_source_execution: Mapping[str, Any],
    model_stage_evidence: Mapping[str, Any],
    model_stage_evidence_file_sha256: str,
) -> dict[str, Any]:
    model_stage_evidence_sha256 = contract.canonical_sha256(model_stage_evidence)
    return {
        "schema": generation.RUN_CONFIG_SCHEMA,
        "authorization_sha256": validated["authorization_sha256"],
        "authorization_binding": copy.deepcopy(
            validated["authorization_binding"]
        ),
        "launcher_evidence": copy.deepcopy(validated["launcher_evidence"]),
        "reviewed_commit": validated["reviewed_commit"],
        "authorization": validated["config"],
        "runtime_environment": copy.deepcopy(dict(runtime_environment)),
        "verified_source_execution": copy.deepcopy(
            dict(verified_source_execution)
        ),
        "design_sha256": design["design_sha256"],
        "contract_sha256": contract.CONTRACT_SHA256,
        "runtime_evidence_sha256": runtime_evidence_sha256,
        "model_stage_evidence_sha256": model_stage_evidence_sha256,
        "model_stage_evidence_file_sha256": _require_sha256(
            model_stage_evidence_file_sha256,
            "model-stage evidence file SHA-256",
        ),
    }


def _validate_verified_source_execution(
    value: Any, validated: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        generation._VERIFIED_RUNTIME_MODULE_SOURCES
    ):
        raise ValueError("verified source execution inventory differs")
    source_paths = validated.get("source_paths")
    source_hashes = validated.get("source_hashes")
    if not isinstance(source_paths, Mapping) or not isinstance(
        source_hashes, Mapping
    ):
        raise ValueError("authorization omitted runtime source bindings")
    normalized = {}
    for module_name, source_name in generation._VERIFIED_RUNTIME_MODULE_SOURCES.items():
        row = _require_exact_fields(
            value[module_name],
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
            f"verified source execution {module_name}",
        )
        expected_hash = source_hashes.get(source_name)
        expected_path = source_paths.get(source_name)
        expected_origin = (
            f"<git-blob:{validated.get('reviewed_commit')}:{expected_path}>"
        )
        if (
            row["source_name"] != source_name
            or row["path"] != expected_path
            or row["origin"] != expected_origin
            or not isinstance(expected_hash, Mapping)
            or row["sha256"] != expected_hash.get("sha256")
            or row["loader_id"] != generation.LAUNCH_LOADER_ID
            or row["execution_count"] != 1
            or row["cached"] is not None
        ):
            raise ValueError(
                f"verified source execution differs for {module_name}"
            )
        normalized[module_name] = copy.deepcopy(dict(row))
    return normalized


def _validate_runtime_environment(
    value: Any, validated: Mapping[str, Any]
) -> dict[str, Any]:
    record = _require_exact_fields(
        value,
        frozenset({"schema", "lock_id", "path", "sha256", "observed"}),
        "runtime environment",
    )
    if record["schema"] != generation.ENVIRONMENT_LOCK_SCHEMA:
        raise ValueError("runtime environment schema differs")
    if not isinstance(record["lock_id"], str) or not record["lock_id"]:
        raise ValueError("runtime environment lock id is missing")
    source_paths = validated.get("source_paths")
    source_hashes = validated.get("source_hashes")
    if not isinstance(source_paths, Mapping) or not isinstance(
        source_hashes, Mapping
    ):
        raise ValueError("authorization omitted environment source bindings")
    if record["path"] != source_paths.get("environment_lock"):
        raise ValueError("runtime environment lock path differs")
    expected_lock = source_hashes.get("environment_lock")
    if not isinstance(expected_lock, Mapping) or record["sha256"] != expected_lock.get(
        "sha256"
    ):
        raise ValueError("runtime environment lock SHA-256 differs")

    observed = _require_exact_fields(
        record["observed"],
        frozenset(
            {
                "platform",
                "runtime",
                "packages",
                "hardware",
                "cuda_device",
                "determinism",
            }
        ),
        "runtime environment observation",
    )
    for key in ("runtime", "packages", "hardware", "determinism"):
        if not isinstance(observed[key], Mapping):
            raise ValueError(f"runtime environment {key} must be a mapping")
    device = _require_exact_fields(
        observed["cuda_device"],
        frozenset(
            {
                "requested",
                "logical_index",
                "cuda_visible_devices",
                "cuda_device_order",
                "torch",
                "nvidia_smi",
                "binding",
            }
        ),
        "runtime CUDA device",
    )
    requested = validated["device"]
    expected_index = int(requested.split(":", 1)[1])
    if (
        device["requested"] != requested
        or device["logical_index"] != expected_index
        or device["cuda_visible_devices"] is not None
        or device["cuda_device_order"] is not None
    ):
        raise ValueError("runtime CUDA logical/physical identity differs")
    torch_device = _require_exact_fields(
        device["torch"],
        frozenset({"index", "name", "uuid"}),
        "runtime PyTorch CUDA identity",
    )
    smi_device = _require_exact_fields(
        device["nvidia_smi"],
        frozenset({"index", "name", "uuid", "pci_bus_id"}),
        "runtime nvidia-smi CUDA identity",
    )
    binding = _require_exact_fields(
        device["binding"],
        frozenset(
            {"uuid_match", "name_match", "unmasked_physical_index_match"}
        ),
        "runtime CUDA identity binding",
    )
    if (
        torch_device["index"] != expected_index
        or smi_device["index"] != expected_index
        or torch_device["name"] != smi_device["name"]
        or torch_device["uuid"] != smi_device["uuid"]
        or _GPU_UUID.fullmatch(str(torch_device["uuid"])) is None
        or _PCI_BUS_ID.fullmatch(str(smi_device["pci_bus_id"])) is None
        or binding
        != {
            "uuid_match": True,
            "name_match": True,
            "unmasked_physical_index_match": True,
        }
    ):
        raise ValueError("runtime CUDA UUID/PCI physical identity differs")
    return copy.deepcopy(dict(record))


def _validate_summary(
    summary: Any,
    *,
    task: Mapping[str, Any],
    record: Mapping[str, Any],
    png_sha256: str,
) -> dict[str, Any]:
    if not isinstance(summary, dict) or set(summary) != set(_SUMMARY_FIELDS):
        raise ValueError("sidecar validator returned a non-canonical summary shape")
    if summary["task_id"] != task["task_id"]:
        raise ValueError("sidecar validator summary task id differs")
    if summary["png_sha256"] != png_sha256:
        raise ValueError(f"{task['task_id']}: PNG bytes differ from sidecar SHA-256")
    if summary["sidecar_sha256"] != contract.canonical_sha256(record):
        raise ValueError(f"{task['task_id']}: sidecar canonical SHA-256 differs")
    for key in (
        "initial_latent_sha256",
        "final_latent_sha256",
        "png_sha256",
        "trajectory_chain_sha256",
        "sidecar_sha256",
    ):
        _require_sha256(summary[key], f"{task['task_id']} summary {key}")
    return summary


def _expected_manifest(
    design: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    block_evidence: Sequence[Mapping[str, Any]],
    runtime_evidence_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": generation.RUN_MANIFEST_SCHEMA,
        "status": "complete_generation_only_unscored",
        "experiment_id": contract.EXPERIMENT_ID,
        "record_count": contract.TOTAL_TASK_COUNT,
        "design_sha256": design["design_sha256"],
        "tasks_sha256": design["tasks_sha256"],
        "contract_sha256": contract.CONTRACT_SHA256,
        "runtime_evidence_sha256": runtime_evidence_sha256,
        "block_evidence": list(block_evidence),
        "records": list(summaries),
        "evidence_scope": {
            "generation_only": True,
            "scoring_authorized": False,
            "quality_outcomes_present": False,
        },
    }


def _expected_success(
    validated: Mapping[str, Any],
    design: Mapping[str, Any],
    manifest: Mapping[str, Any],
    run_config: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
    runtime_evidence_file_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": generation.SUCCESS_SCHEMA,
        "status": "complete_generation_only_unscored",
        "one_shot": True,
        "experiment_id": contract.EXPERIMENT_ID,
        "record_count": contract.TOTAL_TASK_COUNT,
        "authorization_binding": copy.deepcopy(
            validated["authorization_binding"]
        ),
        "launcher_evidence": copy.deepcopy(validated["launcher_evidence"]),
        "manifest_sha256": contract.canonical_sha256(manifest),
        "run_config_sha256": contract.canonical_sha256(run_config),
        "design_sha256": design["design_sha256"],
        "contract_sha256": contract.CONTRACT_SHA256,
        "runtime_evidence_sha256": contract.canonical_sha256(runtime_evidence),
        "runtime_evidence_file_sha256": _require_sha256(
            runtime_evidence_file_sha256,
            "runtime evidence file SHA-256",
        ),
        "model_stage_evidence_sha256": run_config[
            "model_stage_evidence_sha256"
        ],
        "model_stage_evidence_file_sha256": run_config[
            "model_stage_evidence_file_sha256"
        ],
        "scoring_authorized": False,
        "quality_inspection_authorized": False,
    }


def _require_exact_success_tree(
    tree: _PinnedRunTree,
    tasks: Sequence[Mapping[str, Any]],
    marker: bool,
) -> None:
    expected_top = set(_SUCCESS_TOP_LEVEL)
    if marker:
        expected_top.add(AUDIT_ATTEMPT_NAME)
    observed_top = _directory_names(
        tree.run_descriptor, "engineering run directory"
    )
    if observed_top != expected_top:
        raise ValueError(
            "engineering success tree differs: "
            f"missing={sorted(expected_top - observed_top)}, "
            f"extra={sorted(observed_top - expected_top)}"
        )
    if tree.records_descriptor is None:
        raise ValueError("records must be a regular non-symlink directory")
    _require_entry_kind(
        tree.run_descriptor,
        "records",
        label="records",
        directory=True,
    )
    expected_records = {
        f"{task['task_id']}{suffix}" for task in tasks for suffix in (".json", ".png")
    }
    observed_records = _directory_names(
        tree.records_descriptor, "engineering records directory"
    )
    if observed_records != expected_records:
        raise ValueError(
            "record tree differs from the 165-task design: "
            f"missing={sorted(expected_records - observed_records)[:5]}, "
            f"extra={sorted(observed_records - expected_records)[:5]}"
        )
    for name in observed_top - {"records"}:
        _require_entry_kind(
            tree.run_descriptor,
            name,
            label=f"run artifact {name}",
            directory=False,
        )
    for name in observed_records:
        _require_entry_kind(
            tree.records_descriptor,
            name,
            label=f"record {name}",
            directory=False,
        )


def _audit_success_run(
    tree: _PinnedRunTree,
    validated: Mapping[str, Any],
    session: _AuditSession,
    *,
    sidecar_validator: Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
    block_validator: Callable[
        [Iterable[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        dict[str, Any],
    ],
    marker_present: bool,
) -> dict[str, Any]:
    design, tasks = _validated_design(validated)
    _scan_forbidden_artifacts(tree)
    _require_exact_success_tree(tree, tasks, marker_present)
    _validate_attempt(session.generation_attempt.value, validated)

    runtime_record = _load_json_record(
        tree.run / "runtime_evidence.json",
        label="runtime evidence",
        directory_descriptor=tree.run_descriptor,
        pins=tree.pins,
    )
    runtime_evidence = _validate_runtime_evidence(
        runtime_record.value,
        require_warning_free=True,
    )
    runtime_evidence_sha256 = contract.canonical_sha256(runtime_evidence)
    model_stage_record = _load_json_record(
        tree.run / "model_stage_evidence.json",
        label="model-stage evidence",
        directory_descriptor=tree.run_descriptor,
        pins=tree.pins,
    )
    model_stage_evidence = _validate_model_stage_evidence(
        model_stage_record.value,
        require_complete=True,
    )
    run_config_record = _load_json_record(
        tree.run / "config.json",
        label="run config",
        directory_descriptor=tree.run_descriptor,
        pins=tree.pins,
    )
    run_config = run_config_record.value
    runtime_environment = _validate_runtime_environment(
        run_config.get("runtime_environment"), validated
    )
    verified_source_execution = _validate_verified_source_execution(
        run_config.get("verified_source_execution"), validated
    )
    expected_config = _expected_run_config(
        validated,
        design,
        runtime_evidence_sha256,
        runtime_environment,
        verified_source_execution,
        model_stage_evidence,
        model_stage_record.file_sha256,
    )
    _canonical_equal(run_config, expected_config, "run config")
    _scan_forbidden_fields(run_config, "run config")

    if tree.records_descriptor is None:
        raise ValueError("engineering records directory is missing")
    records_dir = tree.run / "records"
    summaries: list[dict[str, Any]] = []
    block_evidence: list[dict[str, Any]] = []
    png_bindings: list[dict[str, str]] = []
    sidecar_bindings: list[dict[str, str]] = []
    for block_index in range(contract.PROMPT_COUNT):
        start = block_index * contract.TASKS_PER_PROMPT
        block_tasks = tasks[start : start + contract.TASKS_PER_PROMPT]
        block_records = []
        for task in block_tasks:
            task_id = task["task_id"]
            sidecar_path = records_dir / f"{task_id}.json"
            png_path = records_dir / f"{task_id}.png"
            sidecar_record = _load_json_record(
                sidecar_path,
                label=f"sidecar {task_id}",
                directory_descriptor=tree.records_descriptor,
                pins=tree.pins,
            )
            record = sidecar_record.value
            _scan_forbidden_fields(record, f"sidecar {task_id}")
            png = inspect_png_container(
                png_path,
                directory_descriptor=tree.records_descriptor,
                pins=tree.pins,
            )
            summary = _validate_summary(
                sidecar_validator(record, task),
                task=task,
                record=record,
                png_sha256=png["sha256"],
            )
            summaries.append(
                {
                    **summary,
                    "png_path": f"records/{task_id}.png",
                    "sidecar_path": f"records/{task_id}.json",
                }
            )
            png_bindings.append(
                {"path": f"records/{task_id}.png", "sha256": png["sha256"]}
            )
            sidecar_bindings.append(
                {
                    "path": f"records/{task_id}.json",
                    "sha256": sidecar_record.file_sha256,
                }
            )
            block_records.append(record)
        evidence = block_validator(block_records, block_tasks)
        if not isinstance(evidence, dict):
            raise ValueError("prompt-block validator did not return an evidence object")
        block_evidence.append(evidence)

    if len(summaries) != contract.TOTAL_TASK_COUNT:
        raise RuntimeError("auditor did not validate exactly 165 sidecars")
    if len(block_evidence) != contract.PROMPT_COUNT:
        raise RuntimeError("auditor did not validate exactly 11 prompt blocks")

    expected_manifest = _expected_manifest(
        design, summaries, block_evidence, runtime_evidence_sha256
    )
    manifest_record = _load_json_record(
        tree.run / "manifest.json",
        label="run manifest",
        directory_descriptor=tree.run_descriptor,
        pins=tree.pins,
    )
    manifest = manifest_record.value
    _canonical_equal(manifest, expected_manifest, "run manifest")
    _scan_forbidden_fields(manifest, "run manifest")

    expected_success = _expected_success(
        validated,
        design,
        manifest,
        run_config,
        runtime_evidence,
        runtime_record.file_sha256,
    )
    success = session.generation_terminal.value
    _canonical_equal(success, expected_success, "success receipt")
    _scan_forbidden_fields(success, "success receipt")

    return {
        "schema": AUDIT_SUCCESS_SCHEMA,
        "status": "passed_generation_integrity_only_unscored",
        "one_shot": True,
        "scope": AUDIT_SCOPE,
        "experiment_id": contract.EXPERIMENT_ID,
        "record_count": contract.TOTAL_TASK_COUNT,
        "prompt_block_count": contract.PROMPT_COUNT,
        "authorization_sha256": validated["authorization_sha256"],
        "design_sha256": design["design_sha256"],
        "contract_sha256": contract.CONTRACT_SHA256,
        "generation_attempt_file_sha256": session.generation_attempt.file_sha256,
        "run_config_file_sha256": run_config_record.file_sha256,
        "manifest_file_sha256": manifest_record.file_sha256,
        "generation_success_file_sha256": session.generation_terminal.file_sha256,
        "runtime_evidence_file_sha256": runtime_record.file_sha256,
        "model_stage_evidence_file_sha256": model_stage_record.file_sha256,
        "png_files_sha256": contract.canonical_sha256(png_bindings),
        "sidecar_files_sha256": contract.canonical_sha256(sidecar_bindings),
        "scoring_authorized": False,
        "quality_inspection_authorized": False,
        "warnings": copy.deepcopy(runtime_evidence["warnings"]),
    }


def _validate_failure_receipt(
    tree: _PinnedRunTree,
    validated: Mapping[str, Any],
    session: _AuditSession,
    *,
    sidecar_validator: Callable[
        [Mapping[str, Any], Mapping[str, Any]], dict[str, Any]
    ],
    block_validator: Callable[
        [Iterable[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        dict[str, Any],
    ],
    marker_present: bool,
) -> None:
    design, tasks = _validated_design(validated)
    run_names = _directory_names(tree.run_descriptor, "engineering run directory")
    allowed = {"attempt.json", "failure.json"}
    if tree.records_descriptor is not None:
        allowed.add("records")
    optional_tail = {
        name
        for name in (
            "runtime_evidence.json",
            "manifest.json",
            "model_stage_evidence.json",
            "config.json",
        )
        if name in run_names
    }
    non_stage_tail = optional_tail - {"model_stage_evidence.json"}
    if non_stage_tail not in (
        set(),
        {"runtime_evidence.json"},
        {"runtime_evidence.json", "manifest.json"},
        {"runtime_evidence.json", "manifest.json", "config.json"},
    ):
        raise ValueError("failed generation contains an impossible terminal-write prefix")
    allowed.update(optional_tail)
    if marker_present:
        allowed.add(AUDIT_ATTEMPT_NAME)
    if run_names != allowed:
        raise ValueError("failed generation tree contains unexpected artifacts")
    _scan_forbidden_artifacts(tree)
    for name in run_names - {"records"}:
        _require_entry_kind(
            tree.run_descriptor,
            name,
            label=f"failed-run artifact {name}",
            directory=False,
        )
    _validate_attempt(session.generation_attempt.value, validated)
    failure = session.generation_terminal.value
    expected_fields = {
        "schema",
        "status",
        "experiment_id",
        "authorization_sha256",
        "authorization_binding",
        "launcher_evidence",
        "design_sha256",
        "completed_records",
        "expected_records",
        "exception_type",
        "exception_message",
        "traceback_sha256",
        "runtime_evidence",
        "model_stage",
        "model_stage_verifications",
    }
    if set(failure) != expected_fields:
        raise ValueError("generation failure receipt fields differ")
    fixed = {
        "schema": generation.FAILURE_SCHEMA,
        "status": "failed_terminal_no_resume_or_retry",
        "experiment_id": contract.EXPERIMENT_ID,
        "authorization_sha256": validated["authorization_sha256"],
        "authorization_binding": validated["authorization_binding"],
        "launcher_evidence": validated["launcher_evidence"],
        "design_sha256": design["design_sha256"],
        "expected_records": contract.TOTAL_TASK_COUNT,
    }
    for key, expected in fixed.items():
        if failure.get(key) != expected:
            raise ValueError(f"generation failure receipt {key} differs")
    completed = failure["completed_records"]
    if isinstance(completed, bool) or not isinstance(completed, int) or not (
        0 <= completed <= contract.TOTAL_TASK_COUNT
    ):
        raise ValueError("generation failure completed_records is invalid")
    if not isinstance(failure["exception_type"], str) or not failure["exception_type"]:
        raise ValueError("generation failure exception_type is empty")
    if not isinstance(failure["exception_message"], str):
        raise ValueError("generation failure exception_message must be a string")
    _require_sha256(failure["traceback_sha256"], "generation failure traceback")
    embedded_runtime_evidence = failure["runtime_evidence"]
    if embedded_runtime_evidence is not None:
        embedded_runtime_evidence = _validate_runtime_evidence(
            embedded_runtime_evidence, require_warning_free=False
        )
    runtime_evidence = None
    if "runtime_evidence.json" in optional_tail:
        runtime_record = _load_json_record(
            tree.run / "runtime_evidence.json",
            label="failed-run runtime evidence",
            directory_descriptor=tree.run_descriptor,
            pins=tree.pins,
        )
        runtime_evidence = _validate_runtime_evidence(
            runtime_record.value,
            require_warning_free=True,
        )
        _canonical_equal(
            embedded_runtime_evidence,
            runtime_evidence,
            "embedded/file runtime evidence",
        )

    embedded_model_stage = failure["model_stage"]
    embedded_model_stage_verifications = failure["model_stage_verifications"]
    model_stage_record: Optional[_JsonRecord] = None
    if embedded_model_stage is None:
        if embedded_model_stage_verifications != []:
            raise ValueError(
                "generation failure has model-stage verifications without a stage"
            )
        if "model_stage_evidence.json" in optional_tail:
            raise ValueError(
                "generation failure has model-stage evidence without a stage"
            )
        model_stage_evidence = None
    else:
        if not isinstance(embedded_model_stage, Mapping):
            raise ValueError("generation failure model stage must be a mapping")
        if not isinstance(embedded_model_stage_verifications, list):
            raise ValueError(
                "generation failure model-stage verifications must be a list"
            )
        if "model_stage_evidence.json" not in optional_tail:
            raise ValueError(
                "generation failure omitted terminal model-stage evidence"
            )
        model_stage_record = _load_json_record(
            tree.run / "model_stage_evidence.json",
            label="failed-run model-stage evidence",
            directory_descriptor=tree.run_descriptor,
            pins=tree.pins,
        )
        model_stage_evidence = _validate_model_stage_evidence(
            model_stage_record.value,
            expected_stage=embedded_model_stage,
            expected_verifications=embedded_model_stage_verifications,
            require_complete=False,
        )

    expected_ids = [task["task_id"] for task in tasks]
    json_ids: set[str] = set()
    png_ids: set[str] = set()
    if tree.records_descriptor is not None:
        _require_entry_kind(
            tree.run_descriptor,
            "records",
            label="failed generation records directory",
            directory=True,
        )
        for name in _directory_names(
            tree.records_descriptor, "failed generation records directory"
        ):
            path = Path(name)
            _require_entry_kind(
                tree.records_descriptor,
                name,
                label=f"failed-run record {name}",
                directory=False,
            )
            if path.suffix not in {".json", ".png"} or path.stem not in set(
                expected_ids
            ):
                raise ValueError("failed generation contains an unknown record artifact")
            (json_ids if path.suffix == ".json" else png_ids).add(path.stem)
    elif completed != 0:
        raise ValueError("failed generation omitted records after completing tasks")
    if not json_ids.issubset(png_ids) or len(png_ids - json_ids) > 1:
        raise ValueError("failed generation has an impossible atomic record prefix")
    if len(json_ids) not in {completed, min(completed + 1, contract.TOTAL_TASK_COUNT)}:
        raise ValueError("failure completed_records differs from the record prefix")
    expected_json_prefix = set(expected_ids[: len(json_ids)])
    expected_png_prefix = set(expected_ids[: len(png_ids)])
    if json_ids != expected_json_prefix or png_ids != expected_png_prefix:
        raise ValueError("failed generation record artifacts are not a design prefix")
    tasks_by_id = {task["task_id"]: task for task in tasks}
    records_by_id: dict[str, dict[str, Any]] = {}
    summaries_by_id: dict[str, dict[str, Any]] = {}
    png_by_id: dict[str, dict[str, Any]] = {}
    if tree.records_descriptor is not None:
        for task_id in expected_ids[: len(png_ids)]:
            png_by_id[task_id] = inspect_png_container(
                tree.run / "records" / f"{task_id}.png",
                directory_descriptor=tree.records_descriptor,
                pins=tree.pins,
            )
    for task_id in expected_ids[: len(json_ids)]:
        if tree.records_descriptor is None:
            raise RuntimeError("failed generation records directory disappeared")
        record = _load_json_record(
            tree.run / "records" / f"{task_id}.json",
            label=f"failed-run sidecar {task_id}",
            directory_descriptor=tree.records_descriptor,
            pins=tree.pins,
        ).value
        _scan_forbidden_fields(record, f"failed-run sidecar {task_id}")
        png = png_by_id[task_id]
        summary = _validate_summary(
            sidecar_validator(record, tasks_by_id[task_id]),
            task=tasks_by_id[task_id],
            record=record,
            png_sha256=png["sha256"],
        )
        records_by_id[task_id] = record
        summaries_by_id[task_id] = {
            **summary,
            "png_path": f"records/{task_id}.png",
            "sidecar_path": f"records/{task_id}.json",
        }

    complete_blocks = len(json_ids) // contract.TASKS_PER_PROMPT
    block_evidence = []
    for block_index in range(complete_blocks):
        start = block_index * contract.TASKS_PER_PROMPT
        block_tasks = tasks[start : start + contract.TASKS_PER_PROMPT]
        block_records = [records_by_id[task["task_id"]] for task in block_tasks]
        evidence = block_validator(block_records, block_tasks)
        if not isinstance(evidence, dict):
            raise ValueError("prompt-block validator did not return an evidence object")
        block_evidence.append(evidence)

    if "manifest.json" in optional_tail:
        if len(json_ids) != contract.TOTAL_TASK_COUNT:
            raise ValueError("failed generation published a manifest before 165 records")
        summaries = [summaries_by_id[task_id] for task_id in expected_ids]
        manifest = _load_json_record(
            tree.run / "manifest.json",
            label="failed-run manifest",
            directory_descriptor=tree.run_descriptor,
            pins=tree.pins,
        ).value
        _canonical_equal(
            manifest,
            _expected_manifest(
                design,
                summaries,
                block_evidence,
                contract.canonical_sha256(runtime_evidence),
            ),
            "failed-run manifest",
        )
        _scan_forbidden_fields(manifest, "failed-run manifest")
    if "config.json" in optional_tail:
        if model_stage_evidence is None or model_stage_record is None:
            raise ValueError("failed-run config omitted its model-stage evidence")
        run_config = _load_json_record(
            tree.run / "config.json",
            label="failed-run config",
            directory_descriptor=tree.run_descriptor,
            pins=tree.pins,
        ).value
        runtime_environment = _validate_runtime_environment(
            run_config.get("runtime_environment"), validated
        )
        verified_source_execution = _validate_verified_source_execution(
            run_config.get("verified_source_execution"), validated
        )
        _canonical_equal(
            run_config,
            _expected_run_config(
                validated,
                design,
                contract.canonical_sha256(runtime_evidence),
                runtime_environment,
                verified_source_execution,
                model_stage_evidence,
                model_stage_record.file_sha256,
            ),
            "failed-run config",
        )
        _scan_forbidden_fields(run_config, "failed-run config")


def _failure_payload(attempt: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    trace = traceback.format_exc().encode("utf-8", errors="replace")
    return {
        "schema": AUDIT_FAILURE_SCHEMA,
        "status": "failed_terminal_no_resume_or_retry",
        "one_shot": True,
        "scope": AUDIT_SCOPE,
        "experiment_id": contract.EXPERIMENT_ID,
        "audit_attempt_sha256": contract.canonical_sha256(attempt),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback_sha256": hashlib.sha256(trace).hexdigest(),
    }


def audit_engineering_run(
    authorization_path: str | os.PathLike[str],
    *,
    device: str,
    run_dir: str | os.PathLike[str],
    repo_root: str | os.PathLike[str] = ROOT,
) -> dict[str, Any]:
    """Reject direct audit publication outside the verified launcher."""

    del authorization_path, device, run_dir, repo_root
    raise RuntimeError(
        "engineering audit must be invoked by the verified execution-root launcher"
    )


def _audit_launcher_validated_run(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish an audit after the launcher's complete trust bundle is validated."""

    if not isinstance(validated, Mapping):
        raise ValueError("launcher validator did not return a mapping")
    _validate_auditor_launcher_execution(validated)
    _root, tree = _resolve_run_dir(
        str(validated.get("output_dir", "")),
        str(validated.get("repo_root", "")),
    )
    try:
        generation._require_fd_headroom(
            generation._AUDIT_PINNED_FILE_COUNT
            + generation._FD_SAFETY_RESERVE,
            "adaptive-oracle engineering audit",
        )
        if validated.get("device") not in generation.ALLOWED_CUDA_DEVICES:
            raise ValueError("launcher validator returned an invalid CUDA device")
        source_hashes = validated.get("source_hashes")
        auditor_pair = (
            None
            if not isinstance(source_hashes, Mapping)
            else source_hashes.get("engineering_auditor")
        )
        if not isinstance(auditor_pair, Mapping):
            raise ValueError(
                "launcher validation omitted the engineering-auditor binding"
            )
        auditor_sha256 = _require_sha256(
            auditor_pair.get("sha256"), "engineering auditor SHA-256"
        )
        session = _preflight_audit(
            tree,
            validated,
            auditor_sha256=auditor_sha256,
        )
        if session.audit_attempt is None:
            raise RuntimeError("audit preflight omitted its one-shot attempt")
        success_ownership: list[tuple[int, int]] = []
        try:
            result = _audit_validated_run(
                tree,
                validated,
                terminal=session.terminal,
                sidecar_validator=contract.validate_sidecar,
                block_validator=contract.validate_prompt_block,
                marker_present=True,
                session=session,
            )
            result["audit_attempt_sha256"] = contract.canonical_sha256(
                session.audit_attempt
            )
            _publish_audit_success(
                tree,
                result,
                ownership=success_ownership,
            )
            return result
        except BaseException as exc:
            if len(success_ownership) == 1:
                try:
                    generation._remove_matching_entry(
                        tree.run_descriptor,
                        AUDIT_SUCCESS_NAME,
                        success_ownership[0],
                    )
                except BaseException as rollback_exc:
                    generation._add_exception_note(
                        exc,
                        "audit success rollback also failed: "
                        f"{type(rollback_exc).__name__}: {rollback_exc}",
                    )
            try:
                final_names = _directory_names(
                    tree.run_descriptor, "engineering run directory"
                ).intersection({AUDIT_SUCCESS_NAME, AUDIT_FAILURE_NAME})
            except BaseException:
                final_names = frozenset({"untrusted-directory-state"})
            if not final_names:
                try:
                    _publish_audit_failure(
                        tree,
                        _failure_payload(session.audit_attempt, exc),
                    )
                except BaseException as receipt_exc:
                    raise RuntimeError(
                        "engineering audit failed and failure receipt creation also "
                        f"failed: {type(receipt_exc).__name__}: {receipt_exc}"
                    ) from exc
            raise
    finally:
        tree.close()


def _audit_validated_run(
    run: Path | _PinnedRunTree,
    validated: Mapping[str, Any],
    *,
    terminal: Optional[str] = None,
    sidecar_validator: Optional[
        Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]
    ] = None,
    block_validator: Optional[
        Callable[
            [Iterable[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
            dict[str, Any],
        ]
    ] = None,
    marker_present: bool = False,
    session: Optional[_AuditSession] = None,
) -> dict[str, Any]:
    """Audit an authorized run without creating or replacing any receipt."""

    own_tree = not isinstance(run, _PinnedRunTree)
    if own_tree:
        _root, tree = _resolve_run_dir(
            str(run), str(validated.get("repo_root", ""))
        )
    else:
        tree = run
    try:
        if tree.run_inventory is None:
            tree.snapshot_inventories()
        active_session = session or _read_generation_session(
            tree,
            validated,
            terminal=terminal,
        )
        observed_terminal = _terminal_state(tree)
        if (
            active_session.terminal not in {"success", "failure"}
            or observed_terminal != active_session.terminal
            or (terminal is not None and terminal != active_session.terminal)
        ):
            raise ValueError("engineering generation terminal state is invalid")
        selected_sidecar_validator = sidecar_validator or contract.validate_sidecar
        selected_block_validator = block_validator or contract.validate_prompt_block
        if observed_terminal == "failure":
            _validate_failure_receipt(
                tree,
                validated,
                active_session,
                sidecar_validator=selected_sidecar_validator,
                block_validator=selected_block_validator,
                marker_present=marker_present,
            )
            tree.verify()
            raise GenerationFailedTerminally(
                "engineering generation has a valid terminal failure receipt; "
                "it cannot pass audit"
            )
        result = _audit_success_run(
            tree,
            validated,
            active_session,
            sidecar_validator=selected_sidecar_validator,
            block_validator=selected_block_validator,
            marker_present=marker_present,
        )
        tree.verify()
        return result
    finally:
        if own_tree:
            tree.close()


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    execution_context: Optional[Mapping[str, Any]] = None,
) -> int:
    del argv, execution_context
    raise RuntimeError(
        "engineering audit must be invoked by the verified execution-root launcher"
    )


def _run_from_verified_launcher(
    argv: Sequence[str], *, launcher_context: Any
) -> int:
    """Consume one authenticated launcher context and audit its bound run."""

    parser = argparse.ArgumentParser(
        prog="audit_adaptive_oracle_engineering",
        description="Result-blind audit of adaptive-oracle engineering generation",
        allow_abbrev=False,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(list(argv))
    launcher_evidence = launcher_context.claim()
    input_bytes = {
        name: launcher_context.read_input(name)
        for name in (
            "registration",
            "cpu_audit",
            "prompt_csv",
            "exclusion_inventory",
            "prompt_manifest",
            "environment_lock",
        )
    }
    validated = generation._validate_launcher_bundle(
        launcher_evidence,
        input_bytes,
        requested_device=args.device,
        requested_output_dir=args.run_dir,
    )
    result = _audit_launcher_validated_run(validated)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
