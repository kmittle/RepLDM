"""Parallel, content-addressed construction of the selected image index.

The selected-view builder must inspect every protected image, but decoding and
CLIP inference are embarrassingly parallel.  This module splits the frozen
global row order into deterministic shards and validates the merged result
before the ordinary selected-view builder consumes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import canonical_json_bytes, iter_jsonl, sha256_file
from .selected import SELECTED_IMAGE_MAX_PIXELS, decode_image_payload, dct_phash_v1
from .selected_assets import (
    _absolute_path,
    _encode_images,
    _load_bound_clip,
    _load_rows,
    _manifest_artifact_path,
    _model_descriptor,
    _ordinary_file,
    _preflight_selected_asset_destinations,
    _reject_directory_overlap,
    _validate_parent_release,
    _validate_directory,
    file_binding,
    unique_protected_prompts,
    unique_protected_images,
)


IMAGE_SHARD_SCHEMA = "repldm.selected_image_index_shard.v2"
IMAGE_BUNDLE_SCHEMA = "repldm.selected_image_index_bundle.v2"
_SHARD_FIELDS = frozenset(
    {
        "schema",
        "parent_release_id",
        "parent_manifest_sha256",
        "clip_checkpoint",
        "decoder",
        "shard_index",
        "shard_count",
        "total_count",
        "positions",
        "embedding_dim",
        "artifacts",
    }
)
_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "parent_release_id",
        "parent_manifest_sha256",
        "clip_checkpoint",
        "decoder",
        "total_count",
        "embedding_dim",
        "shard_count",
        "shards",
        "artifacts",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PHASH_RE = re.compile(r"^[0-9a-f]{16}$")


def decoder_contract() -> dict[str, Any]:
    """Return the decoder contract used by the selected-view builder."""
    from PIL import __version__ as pillow_version
    from PIL import features

    return {
        "library": "Pillow",
        "version": pillow_version,
        "littlecms_version": features.version("littlecms2"),
        "max_image_pixels": SELECTED_IMAGE_MAX_PIXELS,
        "exif_transpose": True,
        "icc_to_srgb": True,
        "output_mode": "RGB",
        "pixel_hash": "sha256_rgb_u64be_width_height_bytes_v1",
    }


def shard_positions(total: int, shard_index: int, shard_count: int) -> tuple[int, ...]:
    """Return the stable global positions assigned to one shard."""
    if type(total) is not int or total < 0:
        raise ValueError("total must be a non-negative integer")
    if type(shard_index) is not int or shard_index < 0:
        raise ValueError("shard_index must be a non-negative integer")
    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    if shard_index >= shard_count:
        raise ValueError("shard_index must be smaller than shard_count")
    return tuple(range(shard_index, total, shard_count))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_bytes(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _parent_rows(
    parent_release: Path, *, validate_files: bool
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    parent_release = _validate_parent_release(parent_release)
    manifest_path = _ordinary_file(parent_release / "manifest.json", label="parent manifest")
    manifest = _read_json(manifest_path, label="parent manifest")
    if manifest.get("release_id") != parent_release.name:
        raise ValueError("parent manifest directory and release_id differ")
    if manifest.get("candidate_catalog_complete") is not True:
        raise ValueError("parent catalog is not a complete candidate release")
    holdout_path = _manifest_artifact_path(
        parent_release,
        manifest,
        "benchmark_holdouts.jsonl",
        label="protected holdout catalog",
    )
    holdout_artifacts = [
        value
        for value in manifest.get("artifacts", [])
        if isinstance(value, Mapping) and value.get("path") == "benchmark_holdouts.jsonl"
    ]
    rows = _load_rows(holdout_path)
    if len(rows) != 49_393:
        raise ValueError("protected holdout catalog must contain 49,393 rows")
    if len(holdout_artifacts) != 1 or holdout_artifacts[0].get("rows") != len(rows):
        raise ValueError("protected holdout artifact row count differs from its contents")
    prompts = unique_protected_prompts(rows)
    expected_prompts = manifest.get("protected_normalized_unique_prompts")
    if type(expected_prompts) is not int or len(prompts) != expected_prompts:
        raise ValueError("protected prompt count differs from parent manifest")
    images = unique_protected_images(rows, validate_files=validate_files)
    expected_images = manifest.get("protected_unique_images")
    if type(expected_images) is not int or len(images) != expected_images:
        raise ValueError("protected image count differs from parent manifest")
    return manifest, sha256_file(manifest_path), images


def _checkpoint_descriptor(checkpoint: Path) -> dict[str, Any]:
    # This checks bytes and identity without deserializing the model.
    return _model_descriptor(_ordinary_file(Path(checkpoint), label="CLIP checkpoint"))


def _build_image_index_shard_in_directory(
    *,
    parent_release: Path,
    output_dir: Path,
    clip_checkpoint: Path,
    shard_index: int,
    shard_count: int,
    device: str = "cuda",
    batch_size: int = 32,
    decode_workers: int = 8,
    published_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Encode one deterministic shard of the protected image cohort."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    published_output_dir = _validate_directory(
        Path(published_output_dir or output_dir),
        label="published image shard directory",
    )
    parent_release = _absolute_path(Path(parent_release))
    output_dir = _validate_directory(output_dir, label="image shard output directory")
    # The parent manifest is already hash-bound.  Each selected image is still
    # opened and decoded below, so repeating 37,160 network ``stat`` calls in
    # every worker only creates avoidable I/O contention.
    parent_manifest, parent_hash, image_rows = _parent_rows(
        parent_release, validate_files=False
    )
    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    if shard_count > len(image_rows):
        raise ValueError("shard_count cannot exceed the protected image count")
    positions = shard_positions(len(image_rows), shard_index, shard_count)
    selected_rows = [image_rows[position] for position in positions]
    decoder = decoder_contract()
    checkpoint = _ordinary_file(Path(clip_checkpoint), label="CLIP checkpoint")
    _reject_directory_overlap(
        published_output_dir,
        [parent_release, checkpoint.parent],
        label="image shard output directory",
    )
    stem = f"shard-{shard_index:04d}"
    npz_path = output_dir / f"{stem}.npz"
    records_path = output_dir / f"{stem}.jsonl"
    manifest_path = output_dir / f"{stem}.json"
    _preflight_selected_asset_destinations(
        (("arrays", npz_path), ("records", records_path), ("manifest", manifest_path)),
        [
            parent_release / "manifest.json",
            parent_release / "benchmark_holdouts.jsonl",
            checkpoint,
            *(Path(str(row["image_path"])) for row in image_rows),
        ],
        input_dirs=[parent_release, checkpoint.parent],
    )
    descriptor, model, preprocess = _load_bound_clip(checkpoint, device)
    if descriptor != _checkpoint_descriptor(checkpoint):
        raise RuntimeError("CLIP descriptor changed while loading the shard")
    embeddings, phashes, evidence = _encode_images(
        model,
        preprocess,
        selected_rows,
        decoder=decoder,
        device=device,
        batch_size=batch_size,
        with_hashes=True,
        decode_workers=decode_workers,
    )
    if len(embeddings) != len(positions) or len(phashes) != len(positions):
        raise RuntimeError("image encoder returned an incomplete shard")
    embedding_dim = _validate_embedding_matrix(
        np.asarray(embeddings), label="image encoder output"
    )
    records: list[dict[str, Any]] = []
    for position, row, phash, item in zip(positions, selected_rows, phashes, evidence):
        record = {
            "position": position,
            "id": str(row["id"]),
            "image_path": str(row["image_path"]),
            "phash": str(phash),
            "raw_file_sha256": str(item["raw_file_sha256"]),
            "decoded_pixel_sha256": str(item["decoded_pixel_sha256"]),
            "decoded_width": int(item["decoded_width"]),
            "decoded_height": int(item["decoded_height"]),
        }
        records.append(record)
    def published_path(path: Path) -> Path:
        return published_output_dir / path.relative_to(output_dir)

    _atomic_npz(
        npz_path,
        positions=np.asarray(positions, dtype=np.int64),
        ids=np.asarray([row["id"] for row in records], dtype=str),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    _atomic_jsonl(records_path, records)
    shard = {
        "schema": IMAGE_SHARD_SCHEMA,
        "parent_release_id": parent_manifest["release_id"],
        "parent_manifest_sha256": parent_hash,
        "clip_checkpoint": descriptor,
        "decoder": decoder,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "total_count": len(image_rows),
        "positions": {"first": positions[0] if positions else None, "count": len(positions)},
        "embedding_dim": embedding_dim,
        "artifacts": {
            "arrays": file_binding(npz_path, published_path=published_path(npz_path)),
            "records": file_binding(records_path, published_path=published_path(records_path)),
        },
    }
    _atomic_json(manifest_path, shard)
    return shard


def _fsync_tree(root: Path) -> None:
    """Flush a completed shard staging directory before installation."""
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_image_index_shard(
    *,
    parent_release: Path,
    output_dir: Path,
    clip_checkpoint: Path,
    shard_index: int,
    shard_count: int,
    device: str = "cuda",
    batch_size: int = 32,
    decode_workers: int = 8,
) -> dict[str, Any]:
    """Build one shard and atomically install its complete artifact directory."""
    output_root = _validate_directory(Path(output_dir), label="image shard output root")
    if type(shard_index) is not int or shard_index < 0:
        raise ValueError("shard_index must be a non-negative integer")
    parent_path = _absolute_path(Path(parent_release))
    checkpoint_path = _ordinary_file(Path(clip_checkpoint), label="CLIP checkpoint")
    _reject_directory_overlap(
        output_root,
        [parent_path, checkpoint_path.parent],
        label="image shard output root",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = _validate_directory(output_root, label="image shard output root")
    stem = f"shard-{shard_index:04d}"
    final_shard_dir = output_root / stem
    if final_shard_dir.exists():
        raise ValueError(f"image shard output already exists: {final_shard_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{stem}.staging-", dir=str(output_root)))
    try:
        manifest = _build_image_index_shard_in_directory(
            parent_release=parent_release,
            output_dir=staging,
            clip_checkpoint=clip_checkpoint,
            shard_index=shard_index,
            shard_count=shard_count,
            device=device,
            batch_size=batch_size,
            decode_workers=decode_workers,
            published_output_dir=final_shard_dir,
        )
        _fsync_tree(staging)
        os.replace(staging, final_shard_dir)
        descriptor = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return manifest
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_binding(value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} binding is incomplete")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{label} path is invalid")
    canonical_path = Path(os.path.abspath(raw_path))
    if raw_path != str(canonical_path):
        raise ValueError(f"{label} path is not canonical")
    path = _ordinary_file(canonical_path, label=label)
    if type(value.get("bytes")) is not int or value["bytes"] != path.stat().st_size:
        raise ValueError(f"{label} byte count differs")
    digest = value.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} hash is invalid")
    if sha256_file(path) != digest:
        raise ValueError(f"{label} bytes differ from its binding")
    return path


def bound_image_index_paths(bundle_path: Path) -> tuple[Path, ...]:
    """Return every file bound by a bundle, after validating its descriptors.

    The selected-view builder uses this read-only preflight to ensure that no
    generated artifact can overwrite the bundle or one of its source files.
    Full parent/provenance and array validation remains in
    :func:`load_image_index_bundle`.
    """
    bundle_path = _ordinary_file(Path(bundle_path), label="image index bundle")
    bundle = _read_json(bundle_path, label="image index bundle")
    if set(bundle) != _BUNDLE_FIELDS or bundle.get("schema") != IMAGE_BUNDLE_SCHEMA:
        raise ValueError("unsupported image index bundle schema")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"arrays", "records"}:
        raise ValueError("image index bundle artifacts are missing")
    shard_bindings = bundle.get("shards")
    shard_count = bundle.get("shard_count")
    if (
        type(shard_count) is not int
        or shard_count <= 0
        or not isinstance(shard_bindings, list)
        or len(shard_bindings) != shard_count
    ):
        raise ValueError("image index bundle shard set is incomplete")
    paths = [
        bundle_path,
        _validate_binding(artifacts["arrays"], label="merged image arrays"),
        _validate_binding(artifacts["records"], label="merged image records"),
    ]
    for index, binding in enumerate(shard_bindings):
        if not isinstance(binding, Mapping) or set(binding) != {
            "manifest",
            "arrays",
            "records",
        }:
            raise ValueError(f"image index bundle shard {index} binding is incomplete")
        manifest_path = _validate_binding(binding["manifest"], label="image shard manifest")
        arrays_path = _validate_binding(binding["arrays"], label="image shard arrays")
        records_path = _validate_binding(binding["records"], label="image shard records")
        expected_stem = f"shard-{index:04d}"
        if (
            manifest_path.name != f"{expected_stem}.json"
            or manifest_path.parent.name != expected_stem
            or arrays_path != manifest_path.with_suffix(".npz")
            or records_path != manifest_path.with_suffix(".jsonl")
        ):
            raise ValueError("image index bundle shard artifact paths are not canonical")
        paths.extend((manifest_path, arrays_path, records_path))
    if len({path for path in paths}) != len(paths):
        raise ValueError("image index bundle binds the same path more than once")
    return tuple(paths)


def _validate_record(record: Mapping[str, Any], *, position: int, expected: Mapping[str, Any]) -> None:
    required = {
        "position",
        "id",
        "image_path",
        "phash",
        "raw_file_sha256",
        "decoded_pixel_sha256",
        "decoded_width",
        "decoded_height",
    }
    if set(record) != required:
        raise ValueError("image shard record fields are incomplete")
    if record.get("position") != position:
        raise ValueError("image shard record position differs from its array")
    if record.get("id") != expected.get("id") or record.get("image_path") != expected.get("image_path"):
        raise ValueError("image shard record differs from the parent cohort")
    if not isinstance(record.get("phash"), str) or _PHASH_RE.fullmatch(record["phash"]) is None:
        raise ValueError("image shard record pHash is invalid")
    for key in ("raw_file_sha256", "decoded_pixel_sha256"):
        if not isinstance(record.get(key), str) or _SHA256_RE.fullmatch(record[key]) is None:
            raise ValueError(f"image shard record {key} is invalid")
    for key in ("decoded_width", "decoded_height"):
        if type(record.get(key)) is not int or record[key] <= 0:
            raise ValueError(f"image shard record {key} is invalid")


def _validate_embedding_matrix(
    values: np.ndarray, *, label: str, expected_dim: int | None = None
) -> int:
    """Validate the finite, row-normalized embedding contract."""
    if values.ndim != 2 or values.shape[1] <= 0:
        raise ValueError(f"{label} embeddings have an invalid shape")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{label} embeddings must be floating point")
    if expected_dim is not None and values.shape[1] != expected_dim:
        raise ValueError(f"{label} embedding dimension differs from its manifest")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} embeddings are not finite floats")
    norms = np.linalg.norm(values.astype(np.float64, copy=False), axis=1)
    if np.any(norms <= 1e-12) or not np.allclose(norms, 1.0, rtol=0.0, atol=5e-3):
        raise ValueError(f"{label} embeddings must be row-normalized")
    return int(values.shape[1])


def _validate_record_image(
    record: Mapping[str, Any], *, expected: Mapping[str, Any], decoder: Mapping[str, Any]
) -> None:
    """Recompute image evidence so shard records cannot forge a pHash."""
    image_path = _ordinary_file(Path(str(expected["image_path"])), label="protected image")
    decoded = decode_image_payload(image_path, decoder)
    if sha256_file(image_path) != record["raw_file_sha256"]:
        raise ValueError("image shard raw image hash differs from the source")
    if (
        decoded.pixel_sha256 != record["decoded_pixel_sha256"]
        or decoded.width != record["decoded_width"]
        or decoded.height != record["decoded_height"]
    ):
        raise ValueError("image shard decoded image evidence differs from the source")
    if dct_phash_v1(decoded) != record["phash"]:
        raise ValueError("image shard pHash differs from decoded pixels")


def _load_shard_manifest(path: Path) -> dict[str, Any]:
    path = _ordinary_file(path, label="image shard manifest")
    manifest = _read_json(path, label="image shard manifest")
    if set(manifest) != _SHARD_FIELDS or manifest.get("schema") != IMAGE_SHARD_SCHEMA:
        raise ValueError("unsupported image shard schema")
    return manifest


def _validate_bundle_shard_binding(
    binding: Mapping[str, Any],
    *,
    index: int,
    shard_count: int,
    total_count: int,
    parent_release_id: str,
    parent_hash: str,
    expected_checkpoint: Mapping[str, Any],
    expected_decoder: Mapping[str, Any],
    embedding_dim: int,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Validate one bundle shard binding and return its exact source paths."""
    if set(binding) != {"manifest", "arrays", "records"}:
        raise ValueError(f"image index bundle shard {index} binding is incomplete")
    manifest_path = _validate_binding(binding["manifest"], label="image shard manifest")
    if (
        manifest_path.name != f"shard-{index:04d}.json"
        or manifest_path.parent.name != f"shard-{index:04d}"
    ):
        raise ValueError("image index bundle shard filename differs from its index")
    manifest = _load_shard_manifest(manifest_path)
    if (
        manifest.get("parent_release_id") != parent_release_id
        or manifest.get("parent_manifest_sha256") != parent_hash
        or manifest.get("clip_checkpoint") != expected_checkpoint
        or manifest.get("decoder") != expected_decoder
        or manifest.get("shard_index") != index
        or manifest.get("shard_count") != shard_count
        or manifest.get("total_count") != total_count
        or manifest.get("embedding_dim") != embedding_dim
    ):
        raise ValueError("image index bundle shard provenance differs from the bundle")
    expected_positions = shard_positions(total_count, index, shard_count)
    expected_first = expected_positions[0] if expected_positions else None
    positions_meta = manifest.get("positions")
    if (
        not isinstance(positions_meta, Mapping)
        or set(positions_meta) != {"first", "count"}
        or positions_meta.get("first") != expected_first
        or positions_meta.get("count") != len(expected_positions)
    ):
        raise ValueError("image index bundle shard positions differ from its index")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, Mapping) or set(manifest_artifacts) != {
        "arrays",
        "records",
    }:
        raise ValueError("image index bundle shard artifacts are incomplete")
    arrays_path = _validate_binding(binding["arrays"], label="image shard arrays")
    records_path = _validate_binding(binding["records"], label="image shard records")
    if (
        arrays_path != manifest_path.with_suffix(".npz")
        or records_path != manifest_path.with_suffix(".jsonl")
    ):
        raise ValueError("image index bundle shard artifact paths are not canonical")
    if (
        manifest_artifacts["arrays"] != binding["arrays"]
        or manifest_artifacts["records"] != binding["records"]
    ):
        raise ValueError("image index bundle shard artifact bindings differ")
    return manifest_path, arrays_path, records_path, manifest


def _merge_image_index_shards_in_directory(
    *,
    parent_release: Path,
    shard_dir: Path,
    output_dir: Path,
    clip_checkpoint: Path,
    shard_count: int | None = None,
    published_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate all shards and publish one merged image-index bundle."""
    published_output_dir = _validate_directory(
        Path(published_output_dir or output_dir),
        label="published merged image output directory",
    )
    parent_release = _absolute_path(Path(parent_release))
    shard_dir = _validate_directory(shard_dir, label="image shard input directory")
    output_dir = _validate_directory(output_dir, label="merged image output directory")
    _reject_directory_overlap(
        published_output_dir,
        [parent_release, shard_dir, Path(clip_checkpoint).absolute().parent],
        label="merged image output directory",
    )
    parent_manifest, parent_hash, image_rows = _parent_rows(
        parent_release, validate_files=False
    )
    expected_checkpoint = _checkpoint_descriptor(Path(clip_checkpoint))
    expected_decoder = decoder_contract()
    manifests = sorted(shard_dir.glob("shard-*/shard-*.json"))
    if not manifests:
        raise ValueError("no image shard manifests found")
    loaded = [_load_shard_manifest(path) for path in manifests]
    raw_counts = [item.get("shard_count") for item in loaded]
    if any(type(value) is not int or value <= 0 for value in raw_counts):
        raise ValueError("image shard manifests have an invalid shard_count")
    counts = set(raw_counts)
    if len(counts) != 1:
        raise ValueError("image shards disagree on shard_count")
    observed_count = counts.pop()
    if shard_count is not None and (
        type(shard_count) is not int or shard_count <= 0 or shard_count != observed_count
    ):
        raise ValueError("requested shard_count differs from shard manifests")
    if len(loaded) != observed_count:
        raise ValueError("image shard set is incomplete")
    if observed_count <= 0 or observed_count > len(image_rows):
        raise ValueError("image shard count is outside the protected image range")
    by_index: dict[int, dict[str, Any]] = {}
    for manifest_path, item in zip(manifests, loaded):
        index = item.get("shard_index")
        if type(index) is not int or index < 0 or index >= observed_count or index in by_index:
            raise ValueError("image shard indices are incomplete or duplicated")
        if (
            manifest_path.name != f"shard-{index:04d}.json"
            or manifest_path.parent.name != f"shard-{index:04d}"
        ):
            raise ValueError("image shard filename differs from its shard_index")
        if (
            item.get("parent_release_id") != parent_manifest["release_id"]
            or item.get("parent_manifest_sha256") != parent_hash
            or item.get("clip_checkpoint") != expected_checkpoint
            or item.get("decoder") != expected_decoder
            or item.get("shard_count") != observed_count
            or item.get("total_count") != len(image_rows)
        ):
            raise ValueError("image shard provenance differs from the current parent")
        by_index[index] = item
    if set(by_index) != set(range(observed_count)):
        raise ValueError("image shard indices do not cover the registered range")

    embeddings_by_position: dict[int, np.ndarray] = {}
    records_by_position: dict[int, dict[str, Any]] = {}
    shard_bindings: list[dict[str, Any]] = []
    embedding_dim: int | None = None
    for index in range(observed_count):
        item = by_index[index]
        item_dim = item.get("embedding_dim")
        if type(item_dim) is not int or item_dim <= 0:
            raise ValueError("image shard embedding_dim is invalid")
        if embedding_dim is None:
            embedding_dim = item_dim
        elif item_dim != embedding_dim:
            raise ValueError("image shards disagree on embedding_dim")
        artifacts = item.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {"arrays", "records"}:
            raise ValueError("image shard artifacts are missing")
        arrays_path = _validate_binding(artifacts.get("arrays"), label="image shard arrays")
        records_path = _validate_binding(artifacts.get("records"), label="image shard records")
        manifest_path = (
            shard_dir / f"shard-{index:04d}" / f"shard-{index:04d}.json"
        )
        if (
            arrays_path != manifest_path.with_suffix(".npz")
            or records_path != manifest_path.with_suffix(".jsonl")
        ):
            raise ValueError("image shard artifact paths are not canonical")
        with np.load(arrays_path, allow_pickle=False) as loaded_arrays:
            if set(loaded_arrays.files) != {"positions", "ids", "embeddings"}:
                raise ValueError("image shard arrays have unexpected fields")
            positions = np.asarray(loaded_arrays["positions"])
            ids = np.asarray(loaded_arrays["ids"])
            values = np.asarray(loaded_arrays["embeddings"])
        if positions.ndim != 1 or ids.ndim != 1:
            raise ValueError("image shard arrays have invalid dimensions")
        _validate_embedding_matrix(
            values,
            label="image shard",
            expected_dim=embedding_dim,
        )
        if len(positions) != len(ids) or len(ids) != len(values):
            raise ValueError("image shard arrays have inconsistent lengths")
        if positions.dtype.kind not in "iu" or ids.dtype.kind not in "SU":
            raise ValueError("image shard arrays have invalid dtypes")
        rows = list(iter_jsonl(records_path))
        if len(rows) != len(positions):
            raise ValueError("image shard record count differs from its arrays")
        expected_positions = shard_positions(len(image_rows), index, observed_count)
        positions_meta = item.get("positions")
        if (
            not isinstance(positions_meta, Mapping)
            or set(positions_meta) != {"first", "count"}
            or positions_meta.get("first")
            != (expected_positions[0] if expected_positions else None)
            or positions_meta.get("count") != len(expected_positions)
        ):
            raise ValueError("image shard manifest positions differ from its index")
        if tuple(int(value) for value in positions.tolist()) != expected_positions:
            raise ValueError("image shard positions are not in canonical order")
        for offset, (position, row) in enumerate(zip(expected_positions, rows)):
            expected = image_rows[position]
            _validate_record(row, position=position, expected=expected)
            _validate_record_image(row, expected=expected, decoder=expected_decoder)
            if str(ids[offset]) != str(expected["id"]):
                raise ValueError("image shard ID array differs from the parent cohort")
            embeddings_by_position[position] = values[offset].astype(np.float32, copy=True)
            records_by_position[position] = dict(row)
        shard_bindings.append(
            {
                "manifest": file_binding(manifest_path),
                "arrays": file_binding(arrays_path),
                "records": file_binding(records_path),
            }
        )
    if set(embeddings_by_position) != set(range(len(image_rows))):
        raise ValueError("image shards do not cover every protected image")
    merged_embeddings = np.stack(
        [embeddings_by_position[position] for position in range(len(image_rows))], axis=0
    )
    if embedding_dim is None:
        raise ValueError("merged image embeddings have an invalid dimension")
    _validate_embedding_matrix(
        merged_embeddings,
        label="merged image",
        expected_dim=embedding_dim,
    )
    merged_records = [records_by_position[position] for position in range(len(image_rows))]
    input_paths = [
        parent_release / "manifest.json",
        parent_release / "benchmark_holdouts.jsonl",
        Path(clip_checkpoint),
        *(Path(str(row["image_path"])) for row in image_rows),
        *(path for binding in shard_bindings for path in (
            Path(str(binding["manifest"]["path"])),
            Path(str(binding["arrays"]["path"])),
            Path(str(binding["records"]["path"])),
        )),
    ]
    arrays_path = output_dir / "protected_image_embedding.npz"
    records_path = output_dir / "protected_image_records.jsonl"
    bundle_path = output_dir / "image_index_bundle.json"
    _preflight_selected_asset_destinations(
        (("arrays", arrays_path), ("records", records_path), ("bundle", bundle_path)),
        input_paths,
        input_dirs=[parent_release, shard_dir, Path(clip_checkpoint).absolute().parent],
    )

    def published_path(path: Path) -> Path:
        return published_output_dir / path.relative_to(output_dir)

    _atomic_npz(
        arrays_path,
        ids=np.asarray([row["id"] for row in merged_records], dtype=str),
        embeddings=merged_embeddings,
    )
    _atomic_jsonl(records_path, merged_records)
    bundle = {
        "schema": IMAGE_BUNDLE_SCHEMA,
        "parent_release_id": parent_manifest["release_id"],
        "parent_manifest_sha256": parent_hash,
        "clip_checkpoint": expected_checkpoint,
        "decoder": expected_decoder,
        "total_count": len(image_rows),
        "embedding_dim": int(merged_embeddings.shape[1]),
        "shard_count": observed_count,
        "shards": shard_bindings,
        "artifacts": {
            "arrays": file_binding(arrays_path, published_path=published_path(arrays_path)),
            "records": file_binding(records_path, published_path=published_path(records_path)),
        },
    }
    _atomic_json(bundle_path, bundle)
    return bundle


def _fsync_tree(root: Path) -> None:
    """Flush a completed merged-index staging directory before installation."""
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def merge_image_index_shards(
    *,
    parent_release: Path,
    shard_dir: Path,
    output_dir: Path,
    clip_checkpoint: Path,
    shard_count: int | None = None,
) -> dict[str, Any]:
    """Merge shards and atomically install one complete bundle directory."""
    output_root = _validate_directory(Path(output_dir), label="merged image output directory")
    parent_path = _absolute_path(Path(parent_release))
    shard_path = _validate_directory(Path(shard_dir), label="image shard input directory")
    checkpoint_path = _ordinary_file(Path(clip_checkpoint), label="CLIP checkpoint")
    _reject_directory_overlap(
        output_root,
        [parent_path, shard_path, checkpoint_path.parent],
        label="merged image output directory",
    )
    if output_root.exists():
        raise ValueError("merged image output directory must be a new, absent directory")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    _validate_directory(output_root.parent, label="merged image output parent")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=str(output_root.parent),
        )
    )
    try:
        bundle = _merge_image_index_shards_in_directory(
            parent_release=parent_release,
            shard_dir=shard_dir,
            output_dir=staging,
            clip_checkpoint=clip_checkpoint,
            shard_count=shard_count,
            published_output_dir=output_root,
        )
        _fsync_tree(staging)
        os.replace(staging, output_root)
        descriptor = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return bundle
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_image_index_bundle(
    bundle_path: Path,
    *,
    parent_release: Path,
    clip_checkpoint: Path,
    decoder: Mapping[str, Any],
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Load a merged bundle and recheck its parent/order before consumption."""
    bundle_path = _ordinary_file(Path(bundle_path), label="image index bundle")
    bundle = _read_json(bundle_path, label="image index bundle")
    if set(bundle) != _BUNDLE_FIELDS or bundle.get("schema") != IMAGE_BUNDLE_SCHEMA:
        raise ValueError("unsupported image index bundle schema")
    parent_manifest, parent_hash, image_rows = _parent_rows(
        parent_release, validate_files=False
    )
    expected_checkpoint = _checkpoint_descriptor(Path(clip_checkpoint))
    if (
        bundle.get("parent_release_id") != parent_manifest["release_id"]
        or bundle.get("parent_manifest_sha256") != parent_hash
        or bundle.get("clip_checkpoint") != expected_checkpoint
        or bundle.get("decoder") != dict(decoder)
        or bundle.get("total_count") != len(image_rows)
    ):
        raise ValueError("image index bundle provenance differs from the current build")
    embedding_dim = bundle.get("embedding_dim")
    if type(embedding_dim) is not int or embedding_dim <= 0:
        raise ValueError("image index bundle embedding_dim is invalid")
    bundle_shard_count = bundle.get("shard_count")
    if type(bundle_shard_count) is not int or bundle_shard_count <= 0:
        raise ValueError("image index bundle shard_count is invalid")
    if bundle_shard_count > len(image_rows):
        raise ValueError("image index bundle shard_count exceeds the image count")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"arrays", "records"}:
        raise ValueError("image index bundle artifacts are missing")
    shard_bindings = bundle.get("shards")
    if (
        not isinstance(shard_bindings, list)
        or len(shard_bindings) != bundle_shard_count
        or not shard_bindings
    ):
        raise ValueError("image index bundle shard set is incomplete")
    arrays_path = _validate_binding(artifacts.get("arrays"), label="merged image arrays")
    records_path = _validate_binding(artifacts.get("records"), label="merged image records")
    with np.load(arrays_path, allow_pickle=False) as loaded_arrays:
        if set(loaded_arrays.files) != {"ids", "embeddings"}:
            raise ValueError("merged image arrays have unexpected fields")
        ids = np.asarray(loaded_arrays["ids"])
        embeddings = np.asarray(loaded_arrays["embeddings"])
    rows = list(iter_jsonl(records_path))
    if (
        embeddings.ndim != 2
        or len(embeddings) != len(rows)
        or len(rows) != len(image_rows)
    ):
        raise ValueError("merged image bundle has inconsistent dimensions")
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "SU"
        or len(ids) != len(rows)
        or len(set(str(value) for value in ids.tolist())) != len(ids)
    ):
        raise ValueError("merged image bundle has inconsistent dimensions")
    _validate_embedding_matrix(
        embeddings,
        label="merged image",
        expected_dim=embedding_dim,
    )
    validated_shards: list[tuple[Path, Path, Path, dict[str, Any]]] = []
    for index, binding in enumerate(shard_bindings):
        if not isinstance(binding, Mapping):
            raise ValueError(f"image index bundle shard {index} binding is incomplete")
        validated_shards.append(
            _validate_bundle_shard_binding(
                binding,
                index=index,
                shard_count=bundle_shard_count,
                total_count=len(image_rows),
                parent_release_id=str(parent_manifest["release_id"]),
                parent_hash=parent_hash,
                expected_checkpoint=expected_checkpoint,
                expected_decoder=decoder,
                embedding_dim=embedding_dim,
            )
        )
    for index, (_manifest_path, shard_arrays_path, shard_records_path, _manifest) in enumerate(
        validated_shards
    ):
        with np.load(shard_arrays_path, allow_pickle=False) as shard_arrays:
            if set(shard_arrays.files) != {"positions", "ids", "embeddings"}:
                raise ValueError("image index bundle shard arrays have unexpected fields")
            positions = np.asarray(shard_arrays["positions"])
            shard_ids = np.asarray(shard_arrays["ids"])
            shard_values = np.asarray(shard_arrays["embeddings"])
        if (
            positions.ndim != 1
            or shard_ids.ndim != 1
            or positions.dtype.kind not in "iu"
            or shard_ids.dtype.kind not in "SU"
        ):
            raise ValueError("image index bundle shard arrays have invalid dimensions")
        _validate_embedding_matrix(
            shard_values,
            label="image index bundle shard",
            expected_dim=embedding_dim,
        )
        expected_positions = shard_positions(len(image_rows), index, bundle_shard_count)
        if tuple(int(value) for value in positions.tolist()) != expected_positions:
            raise ValueError("image index bundle shard positions are not canonical")
        if len(shard_ids) != len(expected_positions) or len(shard_values) != len(expected_positions):
            raise ValueError("image index bundle shard lengths are inconsistent")
        if not np.array_equal(shard_values, embeddings[list(expected_positions)]):
            raise ValueError("image index bundle shard embeddings differ from the merged index")
        shard_rows = list(iter_jsonl(shard_records_path))
        if len(shard_rows) != len(expected_positions):
            raise ValueError("image index bundle shard records have an invalid count")
        for offset, position in enumerate(expected_positions):
            if str(shard_ids[offset]) != str(ids[position]):
                raise ValueError("image index bundle shard IDs differ from the merged index")
            if shard_rows[offset] != rows[position]:
                raise ValueError("image index bundle shard records differ from the merged index")
    phashes: list[str] = []
    evidence: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        expected = image_rows[position]
        _validate_record(row, position=position, expected=expected)
        if str(ids[position]) != str(expected["id"]):
            raise ValueError("merged image IDs differ from the parent cohort")
        _validate_record_image(row, expected=expected, decoder=decoder)
        phashes.append(str(row["phash"]))
        evidence.append(
            {
                "id": str(row["id"]),
                "image_path": str(row["image_path"]),
                "raw_file_sha256": str(row["raw_file_sha256"]),
                "decoded_pixel_sha256": str(row["decoded_pixel_sha256"]),
                "decoded_width": int(row["decoded_width"]),
                "decoded_height": int(row["decoded_height"]),
            }
        )
    return embeddings.astype(np.float32, copy=True), phashes, evidence


__all__ = [
    "IMAGE_BUNDLE_SCHEMA",
    "IMAGE_SHARD_SCHEMA",
    "bound_image_index_paths",
    "build_image_index_shard",
    "decoder_contract",
    "load_image_index_bundle",
    "merge_image_index_shards",
    "shard_positions",
]
