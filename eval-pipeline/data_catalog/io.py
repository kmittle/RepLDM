"""Streaming JSONL I/O with deterministic hashing and atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .schema import DATA_RECORD_SCHEMA, validate_record


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


class ArtifactWriter:
    """Write one normalized data artifact and collect validation statistics."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("xb")
        self._digest = hashlib.sha256()
        self._id_fingerprints: set[bytes] = set()
        self.rows = 0
        self.eligible = 0
        self.with_prompt = 0
        self.with_image = 0
        self.missing_images = 0
        self.leakage_matches = 0
        self.modalities: Counter[str] = Counter()
        self.splits: Counter[str] = Counter()

    def write(self, record: Mapping[str, Any]) -> None:
        validate_record(record)
        fingerprint = hashlib.blake2b(record["id"].encode("utf-8"), digest_size=16).digest()
        if fingerprint in self._id_fingerprints:
            raise ValueError(f"duplicate record id in {self.path.name}: {record['id']}")
        self._id_fingerprints.add(fingerprint)
        image_path = record["image_path"]
        if image_path:
            self.with_image += 1
        if record["source_record"].get("image_exists") is False:
            self.missing_images += 1
        if record["prompt"]:
            self.with_prompt += 1
        self.eligible += int(record["training_eligible"])
        self.leakage_matches += int(bool(record["benchmark_exact_match"]))
        self.modalities[record["modality"]] += 1
        self.splits[record["split"]] += 1
        payload = canonical_json_bytes(record)
        self._handle.write(payload)
        self._digest.update(payload)
        self.rows += 1
        if self.rows % 500_000 == 0:
            print(f"[{self.path.name}] wrote {self.rows:,} rows", flush=True)

    def close(self) -> dict[str, Any]:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        return {
            "path": self.path.name,
            "schema": DATA_RECORD_SCHEMA,
            "rows": self.rows,
            "training_eligible_rows": self.eligible,
            "rows_with_prompt": self.with_prompt,
            "rows_with_image": self.with_image,
            "missing_images": self.missing_images,
            "benchmark_exact_match_rows": self.leakage_matches,
            "modalities": dict(sorted(self.modalities.items())),
            "splits": dict(sorted(self.splits.items())),
            "bytes": self.path.stat().st_size,
            "sha256": self._digest.hexdigest(),
        }

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        self.path.unlink(missing_ok=True)


def write_records(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    verify_paths: bool,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    writer = ArtifactWriter(path)
    try:
        iterator = iter(records)
        if verify_paths:
            with ThreadPoolExecutor(max_workers=32) as executor:
                while batch := list(islice(iterator, 4096)):
                    verify_image_paths(batch, executor=executor)
                    for record in batch:
                        writer.write(record)
        else:
            for record in iterator:
                writer.write(record)
        result = writer.close()
    except BaseException:
        writer.abort()
        raise
    if expected_rows is not None and result["rows"] != expected_rows:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"{path.name} has {result['rows']:,} rows, expected {expected_rows:,}"
        )
    return result


def verify_image_paths(
    records: Iterable[Mapping[str, Any]],
    *,
    executor: ThreadPoolExecutor | None = None,
) -> None:
    """Check absolute image references with bounded parallel filesystem I/O."""
    paths = [str(row["image_path"]) for row in records if row.get("image_path")]
    if not paths:
        return
    owns_executor = executor is None
    pool = executor or ThreadPoolExecutor(max_workers=32)
    try:
        exists = pool.map(os.path.isfile, paths)
        for image_path, is_file in zip(paths, exists):
            if not is_file:
                raise FileNotFoundError(f"catalog image is missing: {image_path}")
    finally:
        if owns_executor:
            pool.shutdown(wait=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
