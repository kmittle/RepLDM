"""Canonical record schema and benchmark-firewall helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence


DATA_RECORD_SCHEMA = "repldm.data_record.v1"
REQUIRED_FIELDS = (
    "schema",
    "id",
    "source",
    "source_roots",
    "split",
    "prompt",
    "image_path",
    "payload_integrity",
    "width",
    "height",
    "license",
    "license_status",
    "modality",
    "intended_use",
    "training_eligible",
    "exclusion_reason",
    "benchmark_exact_match",
    "source_record",
)


def normalize_prompt(value: str) -> str:
    """Normalize prompt text for conservative, exact leakage detection."""
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def stable_record_id(source: str, stable_key: str) -> str:
    """Return a compact deterministic ID without embedding local paths."""
    digest = hashlib.sha256(f"{source}\0{stable_key}".encode("utf-8")).hexdigest()
    return f"{source}:{digest[:24]}"


def _dimension(value: Any, name: str) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer or null")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def make_record(
    *,
    source: str,
    stable_key: str,
    source_roots: Sequence[str],
    split: str,
    prompt: str | None,
    image_path: str | Path | None,
    width: Any = None,
    height: Any = None,
    license_name: str,
    license_status: str,
    modality: str,
    intended_use: Sequence[str],
    training_eligible: bool,
    exclusion_reason: str | None = None,
    benchmark_exact_match: Sequence[str] = (),
    source_record: Mapping[str, Any] | None = None,
    payload_integrity: str | None = None,
) -> dict[str, Any]:
    """Create and validate one normalized catalog record."""
    clean_prompt = prompt.strip() if isinstance(prompt, str) else None
    clean_prompt = clean_prompt or None
    clean_path_object = Path(image_path) if image_path else None
    if clean_path_object is not None and not clean_path_object.is_absolute():
        raise ValueError("image_path must resolve to an absolute path")
    clean_path = str(clean_path_object) if clean_path_object is not None else None
    integrity = payload_integrity or (
        "unbound_path_no_checksum" if clean_path_object is not None else "not_applicable"
    )
    matches = sorted(set(benchmark_exact_match))
    reason = exclusion_reason
    if not isinstance(training_eligible, bool):
        raise ValueError("training_eligible must be boolean")
    eligible = training_eligible
    if matches and eligible:
        eligible = False
        reason = "benchmark_prompt_exact_match"
    if not eligible and not reason:
        raise ValueError("ineligible records require exclusion_reason")
    record = {
        "schema": DATA_RECORD_SCHEMA,
        "id": stable_record_id(source, stable_key),
        "source": source,
        "source_roots": list(source_roots),
        "split": split,
        "prompt": clean_prompt,
        "image_path": clean_path,
        "payload_integrity": integrity,
        "width": _dimension(width, "width"),
        "height": _dimension(height, "height"),
        "license": license_name,
        "license_status": license_status,
        "modality": modality,
        "intended_use": list(dict.fromkeys(intended_use)),
        "training_eligible": eligible,
        "exclusion_reason": reason,
        "benchmark_exact_match": matches,
        "source_record": dict(source_record or {}),
    }
    validate_record(record)
    return record


def validate_record(record: Mapping[str, Any]) -> None:
    """Fail closed when a generated row violates the shared contract."""
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"record is missing fields: {missing}")
    if record["schema"] != DATA_RECORD_SCHEMA:
        raise ValueError(f"unexpected record schema: {record['schema']!r}")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError("record id must be a non-empty string")
    for field in ("source", "split", "license", "license_status", "modality"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"{field} must be a non-empty string")
    if (
        not isinstance(record["source_roots"], list)
        or not record["source_roots"]
        or any(not isinstance(value, str) or not value for value in record["source_roots"])
    ):
        raise ValueError("source_roots must be a non-empty list of strings")
    if (
        not isinstance(record["intended_use"], list)
        or not record["intended_use"]
        or any(not isinstance(value, str) or not value for value in record["intended_use"])
    ):
        raise ValueError("intended_use must be a non-empty list of strings")
    if record["image_path"] is not None and not Path(record["image_path"]).is_absolute():
        raise ValueError("image_path must be absolute")
    if not isinstance(record["payload_integrity"], str) or record[
        "payload_integrity"
    ] not in {
        "not_applicable",
        "missing_reference",
        "unbound_auxiliary_payload",
        "unbound_path_no_checksum",
    }:
        raise ValueError("payload_integrity has an unsupported value")
    if record["prompt"] is not None and not isinstance(record["prompt"], str):
        raise ValueError("prompt must be a string or null")
    if not isinstance(record["training_eligible"], bool):
        raise ValueError("training_eligible must be boolean")
    if not record["training_eligible"] and not record["exclusion_reason"]:
        raise ValueError("ineligible records require exclusion_reason")
    if not isinstance(record["benchmark_exact_match"], list):
        raise ValueError("benchmark_exact_match must be a list")
    if any(
        not isinstance(value, str) or not value for value in record["benchmark_exact_match"]
    ):
        raise ValueError("benchmark_exact_match entries must be non-empty strings")
    for field in ("width", "height"):
        value = record[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{field} must be a positive integer or null")
    if not isinstance(record["source_record"], dict):
        raise ValueError("source_record must be an object")


def benchmark_matches(
    prompt: str | None,
    protected_prompts: Mapping[str, Collection[str]],
) -> list[str]:
    """Return protected sources containing the normalized prompt."""
    if not prompt:
        return []
    normalized = normalize_prompt(prompt)
    return sorted(protected_prompts.get(normalized, ()))
