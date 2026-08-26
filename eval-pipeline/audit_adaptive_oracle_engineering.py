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
import struct
import traceback
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
import uuid
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")


def _read_regular_bytes(path: Path, label: str) -> bytes:
    _require_regular_file(path, label)
    return path.read_bytes()


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value!r}")


def load_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    """Load one exact canonical JSON object with one trailing newline."""

    raw = _read_regular_bytes(path, label)
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


def _scan_forbidden_artifacts(run_dir: Path) -> None:
    for path in run_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"run tree contains a symlink: {path.relative_to(run_dir)}")
        folded = path.name.casefold().replace("-", "_")
        if any(fragment in folded for fragment in _FORBIDDEN_FILE_FRAGMENTS):
            raise ValueError(
                "run tree contains a forbidden score/quality artifact: "
                f"{path.relative_to(run_dir)}"
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


def inspect_png_container(path: Path) -> dict[str, Any]:
    """Validate an exact RGB PNG without reconstructing or inspecting pixels."""

    raw = _read_regular_bytes(path, "PNG record")
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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


def atomic_create_json(path: Path, value: Any) -> None:
    """Publish canonical JSON without replacing an existing receipt."""

    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("audit receipt parent must be a regular directory")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    _write_exclusive(temporary, contract.canonical_json_bytes(value) + b"\n")
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _resolve_run_dir(
    run_dir: str | os.PathLike[str], repo_root: str | os.PathLike[str]
) -> tuple[Path, Path]:
    root = Path(repo_root).resolve(strict=True)
    raw_run = Path(run_dir)
    if raw_run.is_symlink():
        raise ValueError("engineering run directory cannot be a symlink")
    resolved = raw_run.resolve(strict=True)
    expected = (root / generation.OUTPUT_DIR).resolve(strict=False)
    if resolved != expected or not resolved.is_dir():
        raise ValueError(f"engineering auditor requires the exact run {expected}")
    relative = Path(generation.OUTPUT_DIR)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("engineering run path contains a symlink component")
    return root, resolved


def _terminal_state(run_dir: Path) -> str:
    success = os.path.lexists(run_dir / "success.json")
    failure = os.path.lexists(run_dir / "failure.json")
    if success == failure:
        raise ValueError(
            "generation must have exactly one atomic success or failure receipt"
        )
    return "success" if success else "failure"


def _attempt_payload(
    run_dir: Path, terminal: str, *, auditor_sha256: str
) -> dict[str, Any]:
    generation_attempt = run_dir / "attempt.json"
    terminal_path = run_dir / f"{terminal}.json"
    return {
        "schema": AUDIT_ATTEMPT_SCHEMA,
        "status": "started_one_shot_no_resume",
        "one_shot": True,
        "scope": AUDIT_SCOPE,
        "experiment_id": contract.EXPERIMENT_ID,
        "run_dir": generation.OUTPUT_DIR,
        "generation_terminal": terminal,
        "generation_attempt_file_sha256": sha256_file(generation_attempt),
        "generation_terminal_file_sha256": sha256_file(terminal_path),
        "auditor_sha256": _require_sha256(auditor_sha256, "auditor SHA-256"),
    }


def _preflight_audit(
    run_dir: Path, *, auditor_sha256: str
) -> tuple[str, dict[str, Any]]:
    _require_regular_file(run_dir / "attempt.json", "generation attempt receipt")
    terminal = _terminal_state(run_dir)
    _require_regular_file(
        run_dir / f"{terminal}.json", f"generation {terminal} receipt"
    )
    if any(os.path.lexists(run_dir / name) for name in _AUDIT_NAMES):
        raise FileExistsError("engineering audit is one-shot and cannot resume or retry")
    attempt = _attempt_payload(
        run_dir, terminal, auditor_sha256=auditor_sha256
    )
    atomic_create_json(run_dir / AUDIT_ATTEMPT_NAME, attempt)
    return terminal, attempt


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


def _validate_attempt(
    run_dir: Path, validated: Mapping[str, Any]
) -> dict[str, Any]:
    attempt = load_canonical_json(run_dir / "attempt.json", label="generation attempt")
    _canonical_equal(attempt, generation._attempt_record(validated), "generation attempt")
    _scan_forbidden_fields(attempt, "generation attempt")
    return attempt


def _expected_run_config(
    validated: Mapping[str, Any],
    design: Mapping[str, Any],
    runtime_evidence_sha256: str,
    runtime_environment: Mapping[str, Any],
    verified_source_execution: Mapping[str, Any],
    model_stage_evidence: Mapping[str, Any],
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
        "model_stage_evidence_file_sha256": hashlib.sha256(
            contract.canonical_json_bytes(model_stage_evidence) + b"\n"
        ).hexdigest(),
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
        "runtime_evidence_file_sha256": hashlib.sha256(
            contract.canonical_json_bytes(runtime_evidence) + b"\n"
        ).hexdigest(),
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
    run_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    marker: bool,
) -> None:
    expected_top = set(_SUCCESS_TOP_LEVEL)
    if marker:
        expected_top.add(AUDIT_ATTEMPT_NAME)
    observed_top = {path.name for path in run_dir.iterdir()}
    if observed_top != expected_top:
        raise ValueError(
            "engineering success tree differs: "
            f"missing={sorted(expected_top - observed_top)}, "
            f"extra={sorted(observed_top - expected_top)}"
        )
    records = run_dir / "records"
    if records.is_symlink() or not records.is_dir():
        raise ValueError("records must be a regular non-symlink directory")
    expected_records = {
        f"{task['task_id']}{suffix}" for task in tasks for suffix in (".json", ".png")
    }
    observed_records = {path.name for path in records.iterdir()}
    if observed_records != expected_records:
        raise ValueError(
            "record tree differs from the 165-task design: "
            f"missing={sorted(expected_records - observed_records)[:5]}, "
            f"extra={sorted(observed_records - expected_records)[:5]}"
        )
    for path in records.iterdir():
        _require_regular_file(path, f"record {path.name}")


def _audit_success_run(
    run_dir: Path,
    validated: Mapping[str, Any],
    *,
    sidecar_validator: Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
    block_validator: Callable[
        [Iterable[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        dict[str, Any],
    ],
    marker_present: bool,
) -> dict[str, Any]:
    design, tasks = _validated_design(validated)
    _scan_forbidden_artifacts(run_dir)
    _require_exact_success_tree(run_dir, tasks, marker_present)
    _validate_attempt(run_dir, validated)

    runtime_evidence = _validate_runtime_evidence(
        load_canonical_json(
            run_dir / "runtime_evidence.json", label="runtime evidence"
        ),
        require_warning_free=True,
    )
    runtime_evidence_sha256 = contract.canonical_sha256(runtime_evidence)
    model_stage_evidence = _validate_model_stage_evidence(
        load_canonical_json(
            run_dir / "model_stage_evidence.json",
            label="model-stage evidence",
        ),
        require_complete=True,
    )
    run_config = load_canonical_json(run_dir / "config.json", label="run config")
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
    )
    _canonical_equal(run_config, expected_config, "run config")
    _scan_forbidden_fields(run_config, "run config")

    records_dir = run_dir / "records"
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
            record = load_canonical_json(sidecar_path, label=f"sidecar {task_id}")
            _scan_forbidden_fields(record, f"sidecar {task_id}")
            png = inspect_png_container(png_path)
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
                    "sha256": sha256_file(sidecar_path),
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
    manifest = load_canonical_json(run_dir / "manifest.json", label="run manifest")
    _canonical_equal(manifest, expected_manifest, "run manifest")
    _scan_forbidden_fields(manifest, "run manifest")

    expected_success = _expected_success(
        validated, design, manifest, run_config, runtime_evidence
    )
    success = load_canonical_json(run_dir / "success.json", label="success receipt")
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
        "generation_attempt_file_sha256": sha256_file(run_dir / "attempt.json"),
        "run_config_file_sha256": sha256_file(run_dir / "config.json"),
        "manifest_file_sha256": sha256_file(run_dir / "manifest.json"),
        "generation_success_file_sha256": sha256_file(run_dir / "success.json"),
        "runtime_evidence_file_sha256": sha256_file(
            run_dir / "runtime_evidence.json"
        ),
        "model_stage_evidence_file_sha256": sha256_file(
            run_dir / "model_stage_evidence.json"
        ),
        "png_files_sha256": contract.canonical_sha256(png_bindings),
        "sidecar_files_sha256": contract.canonical_sha256(sidecar_bindings),
        "scoring_authorized": False,
        "quality_inspection_authorized": False,
        "warnings": copy.deepcopy(runtime_evidence["warnings"]),
    }


def _validate_failure_receipt(
    run_dir: Path,
    validated: Mapping[str, Any],
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
    records_dir = run_dir / "records"
    allowed = {"attempt.json", "failure.json"}
    if records_dir.exists():
        allowed.add("records")
    optional_tail = {
        name
        for name in (
            "runtime_evidence.json",
            "manifest.json",
            "model_stage_evidence.json",
            "config.json",
        )
        if (run_dir / name).exists()
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
    observed = {path.name for path in run_dir.iterdir()}
    if observed != allowed:
        raise ValueError("failed generation tree contains unexpected artifacts")
    _scan_forbidden_artifacts(run_dir)
    _validate_attempt(run_dir, validated)
    failure = load_canonical_json(run_dir / "failure.json", label="failure receipt")
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
        runtime_evidence = _validate_runtime_evidence(
            load_canonical_json(
                run_dir / "runtime_evidence.json",
                label="failed-run runtime evidence",
            ),
            require_warning_free=True,
        )
        _canonical_equal(
            embedded_runtime_evidence,
            runtime_evidence,
            "embedded/file runtime evidence",
        )

    embedded_model_stage = failure["model_stage"]
    embedded_model_stage_verifications = failure["model_stage_verifications"]
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
        model_stage_evidence = _validate_model_stage_evidence(
            load_canonical_json(
                run_dir / "model_stage_evidence.json",
                label="failed-run model-stage evidence",
            ),
            expected_stage=embedded_model_stage,
            expected_verifications=embedded_model_stage_verifications,
            require_complete=False,
        )

    expected_ids = [task["task_id"] for task in tasks]
    json_ids: set[str] = set()
    png_ids: set[str] = set()
    if records_dir.exists():
        if records_dir.is_symlink() or not records_dir.is_dir():
            raise ValueError("failed generation records directory is invalid")
        for path in records_dir.iterdir():
            _require_regular_file(path, f"failed-run record {path.name}")
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
    for task_id in expected_ids[: len(json_ids)]:
        record = load_canonical_json(
            records_dir / f"{task_id}.json", label=f"failed-run sidecar {task_id}"
        )
        _scan_forbidden_fields(record, f"failed-run sidecar {task_id}")
        png = inspect_png_container(records_dir / f"{task_id}.png")
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
        manifest = load_canonical_json(
            run_dir / "manifest.json", label="failed-run manifest"
        )
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
        if model_stage_evidence is None:
            raise ValueError("failed-run config omitted its model-stage evidence")
        run_config = load_canonical_json(
            run_dir / "config.json", label="failed-run config"
        )
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
            ),
            "failed-run config",
        )
        _scan_forbidden_fields(run_config, "failed-run config")
    raise GenerationFailedTerminally(
        "engineering generation has a valid terminal failure receipt; it cannot pass audit"
    )


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
    _root, run = _resolve_run_dir(
        str(validated.get("output_dir", "")),
        str(validated.get("repo_root", "")),
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
        raise ValueError("launcher validation omitted the engineering-auditor binding")
    auditor_sha256 = _require_sha256(
        auditor_pair.get("sha256"), "engineering auditor SHA-256"
    )
    terminal, audit_attempt = _preflight_audit(
        run, auditor_sha256=auditor_sha256
    )
    try:
        result = _audit_validated_run(
            run,
            validated,
            terminal=terminal,
            sidecar_validator=contract.validate_sidecar,
            block_validator=contract.validate_prompt_block,
            marker_present=True,
        )
        result["audit_attempt_sha256"] = contract.canonical_sha256(audit_attempt)
        atomic_create_json(run / AUDIT_SUCCESS_NAME, result)
        return result
    except BaseException as exc:
        if not os.path.lexists(run / AUDIT_FAILURE_NAME):
            atomic_create_json(run / AUDIT_FAILURE_NAME, _failure_payload(audit_attempt, exc))
        raise


def _audit_validated_run(
    run: Path,
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
) -> dict[str, Any]:
    """Audit an authorized run without creating or replacing any receipt."""

    run = Path(run).resolve(strict=True)
    observed_terminal = _terminal_state(run) if terminal is None else terminal
    if observed_terminal not in {"success", "failure"}:
        raise ValueError("engineering generation terminal state is invalid")
    selected_sidecar_validator = sidecar_validator or contract.validate_sidecar
    selected_block_validator = block_validator or contract.validate_prompt_block
    if observed_terminal == "failure":
        _validate_failure_receipt(
            run,
            validated,
            sidecar_validator=selected_sidecar_validator,
            block_validator=selected_block_validator,
            marker_present=marker_present,
        )
    return _audit_success_run(
        run,
        validated,
        sidecar_validator=selected_sidecar_validator,
        block_validator=selected_block_validator,
        marker_present=marker_present,
    )


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
