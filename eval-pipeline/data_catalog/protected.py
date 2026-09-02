"""Load held-out prompts into the training-data firewall."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .io import sha256_file
from .schema import make_record, normalize_prompt


@dataclass(frozen=True)
class ProtectedPrompt:
    source: str
    stable_key: str
    source_roots: tuple[str, ...]
    prompt: str
    image_path: Path | None
    license_name: str
    license_status: str
    source_record: Mapping[str, Any]


def _path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _within_root(path: str | Path, root: str | Path, *, label: str) -> Path:
    root_lexical = Path(os.path.abspath(root))
    candidate_lexical = Path(os.path.abspath(path))
    try:
        relative = candidate_lexical.relative_to(root_lexical)
    except ValueError:
        relative = None
    if root_lexical.is_symlink():
        raise ValueError(f"{label} uses a symlinked root: {root}")
    if relative is not None:
        cursor = root_lexical
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise ValueError(f"{label} uses a symlinked path: {path}")
    candidate = Path(path).resolve(strict=False)
    root_path = Path(root).resolve(strict=True)
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root: {path}") from exc
    return candidate


def _reject_symlink_path(path: str | Path, *, label: str) -> None:
    """Reject a path or any existing parent component that is a symlink."""
    lexical = Path(os.path.abspath(path))
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{label} uses a symlinked path: {path}")


def _bind_declared_root(
    root: Path,
    source_roots: Iterable[str],
    physical_roots: Mapping[str, Path] | None,
    *,
    label: str,
) -> Path:
    names = tuple(str(name) for name in source_roots)
    _reject_symlink_path(root, label=label)
    candidate = root.resolve(strict=True)
    if not physical_roots:
        return candidate
    unknown = [name for name in names if name not in physical_roots]
    if unknown or not names:
        raise ValueError(f"no physical root is registered for {label}")
    allowed = [physical_roots[name] for name in names]
    for physical in allowed:
        try:
            candidate.relative_to(Path(physical).resolve(strict=True))
            return candidate
        except ValueError:
            continue
    raise ValueError(f"{label} is outside its declared source roots: {root}")


def _iter_source_rows(
    spec: Mapping[str, Any], repository_root: Path,
    physical_roots: Mapping[str, Path] | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    path = _bind_declared_root(
        _path(spec["path"], repository_root),
        spec["source_roots"],
        physical_roots,
        label=f"protected metadata source {spec.get('id', '')}",
    )
    source_format = spec["format"]
    if source_format == "text":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                prompt = line.strip()
                if prompt:
                    yield str(line_number), {"prompt": prompt, "line_number": line_number}
        return
    if source_format == "csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), 2):
                payload = dict(row)
                payload["line_number"] = line_number
                yield str(line_number), payload
        return
    if source_format == "jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected object at {path}:{line_number}")
                yield str(line_number), {**row, "line_number": line_number}
        return
    if source_format == "json_map":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected object map in {path}")
        for key, row in payload.items():
            if not isinstance(row, dict):
                raise ValueError(f"expected object for key {key!r} in {path}")
            yield str(key), dict(row)
        return
    raise ValueError(f"unsupported protected prompt format: {source_format!r}")


def _image_path(
    spec: Mapping[str, Any],
    row_key: str,
    row: Mapping[str, Any],
    repository_root: Path,
    *,
    sharded_index: Mapping[str, Path] | None = None,
    physical_roots: Mapping[str, Path] | None = None,
) -> Path | None:
    mode = spec.get("image_path_mode")
    if not mode:
        return None
    image_root = _bind_declared_root(
        _path(spec["image_root"], repository_root),
        spec["source_roots"],
        physical_roots,
        label="protected image root",
    )
    if mode == "flat_file_name":
        return _within_root(
            image_root / str(row["file_name"]), image_root, label="protected image"
        )
    if mode == "category_key_jpg":
        return _within_root(
            image_root / str(row["category"]) / f"{row_key}.jpg",
            image_root,
            label="protected image",
        )
    if mode == "sharded_id_jpg":
        if sharded_index is None:
            raise RuntimeError("sharded image mode requires a prebuilt image index")
        image_id = _metadata_image_id(spec, row)
        try:
            return _within_root(
                sharded_index[image_id], image_root, label="protected image"
            )
        except KeyError as exc:
            raise FileNotFoundError(
                f"protected metadata image id {image_id!r} has no JPG under {image_root}"
            ) from exc
    raise ValueError(f"unsupported image_path_mode: {mode!r}")


def _metadata_image_id(spec: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    key = str(spec["image_id_key"])
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"protected image id field {key!r} must be a string or integer")
    image_id = str(value).strip()
    if not image_id:
        raise ValueError(f"protected image id field {key!r} must not be empty")
    return image_id


def _sharded_id_jpg_index(
    spec: Mapping[str, Any], repository_root: Path,
    physical_roots: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    image_root = _bind_declared_root(
        _path(spec["image_root"], repository_root),
        spec["source_roots"],
        physical_roots,
        label="protected image root",
    )
    shard_glob = str(spec["image_shard_glob"])
    shard_dirs = sorted(path for path in image_root.glob(shard_glob) if path.is_dir())
    expected_shards = int(spec["expected_image_shards"])
    if len(shard_dirs) != expected_shards:
        raise ValueError(
            f"protected source {spec['id']} found {len(shard_dirs)} image shards, "
            f"expected {expected_shards}"
        )

    image_subdir = str(spec["image_subdir"])
    image_paths: dict[str, Path] = {}
    for shard_dir in shard_dirs:
        payload_dir = shard_dir / image_subdir
        if not payload_dir.is_dir():
            raise FileNotFoundError(
                f"protected image payload directory is missing: {payload_dir}"
            )
        for image_path in sorted(payload_dir.glob("*.jpg")):
            image_path = _within_root(
                image_path, image_root, label="protected shard image"
            )
            image_id = image_path.stem
            previous = image_paths.setdefault(image_id, image_path)
            if previous != image_path:
                raise ValueError(
                    f"duplicate protected image id {image_id!r}: {previous} and {image_path}"
                )

    expected_images = int(spec["expected_unique_images"])
    if len(image_paths) != expected_images:
        raise ValueError(
            f"protected source {spec['id']} indexed {len(image_paths):,} unique images, "
            f"expected {expected_images:,}"
        )
    return image_paths


def load_protected_prompts(
    specs: Iterable[Mapping[str, Any]],
    repository_root: Path,
    physical_roots: Mapping[str, Path] | None = None,
) -> tuple[dict[str, set[str]], list[ProtectedPrompt], dict[str, int]]:
    """Return normalized prompt membership, auditable rows, and source counts."""
    protected: dict[str, set[str]] = {}
    records: list[ProtectedPrompt] = []
    counts: dict[str, int] = {}
    for spec in specs:
        source = str(spec["id"])
        source_path = _bind_declared_root(
            _path(spec["path"], repository_root),
            spec["source_roots"],
            physical_roots,
            label=f"protected metadata source {source}",
        )
        source_file = str(source_path)
        if source_path.stat().st_size != int(spec["expected_bytes"]):
            raise ValueError(f"protected source byte count changed: {source_path}")
        if sha256_file(source_path) != spec["expected_sha256"]:
            raise ValueError(f"protected source hash changed: {source_path}")
        sharded_index = (
            _sharded_id_jpg_index(spec, repository_root, physical_roots)
            if spec.get("image_path_mode") == "sharded_id_jpg"
            else None
        )
        referenced_image_ids: set[str] = set()
        emitted = 0
        for row_key, row in _iter_source_rows(spec, repository_root, physical_roots):
            image_path = _image_path(
                spec,
                row_key,
                row,
                repository_root,
                sharded_index=sharded_index,
                physical_roots=physical_roots,
            )
            if sharded_index is not None:
                image_id = _metadata_image_id(spec, row)
                if image_id in referenced_image_ids:
                    raise ValueError(
                        f"protected source {source} repeats metadata image id {image_id!r}"
                    )
                referenced_image_ids.add(image_id)
            for prompt_key in spec["prompt_keys"]:
                value = row.get(prompt_key)
                if not isinstance(value, str) or not value.strip():
                    continue
                prompt = value.strip()
                protected.setdefault(normalize_prompt(prompt), set()).add(source)
                records.append(
                    ProtectedPrompt(
                        source=source,
                        stable_key=f"{row_key}:{prompt_key}",
                        source_roots=tuple(spec["source_roots"]),
                        prompt=prompt,
                        image_path=image_path,
                        license_name=str(spec["license"]),
                        license_status=str(spec["license_status"]),
                        source_record={
                            "source_file": source_file,
                            "row_key": row_key,
                            "prompt_key": prompt_key,
                            "metadata": row,
                        },
                    )
                )
                emitted += 1
        if sharded_index is not None:
            unreferenced = sorted(set(sharded_index).difference(referenced_image_ids))
            if unreferenced:
                raise ValueError(
                    f"protected source {source} has {len(unreferenced):,} images without "
                    f"metadata; examples: {unreferenced[:3]}"
                )
        expected_prompts = int(spec["expected_prompts"])
        if emitted != expected_prompts:
            raise ValueError(
                f"protected source {source} emitted {emitted:,} prompts, "
                f"expected {expected_prompts:,}"
            )
        counts[source] = emitted
    return protected, records, counts


def unique_protected_image_count(records: Iterable[ProtectedPrompt]) -> int:
    """Count physical benchmark images independently of per-image prompt variants."""
    return len(
        {
            record.image_path.resolve()
            for record in records
            if record.image_path is not None
        }
    )


def iter_holdout_records(records: Iterable[ProtectedPrompt]) -> Iterator[dict[str, Any]]:
    for record in records:
        yield make_record(
            source=f"holdout_{record.source}",
            stable_key=record.stable_key,
            source_roots=record.source_roots,
            split="heldout",
            prompt=record.prompt,
            image_path=record.image_path,
            license_name=record.license_name,
            license_status=record.license_status,
            modality="benchmark_image_text" if record.image_path else "benchmark_prompt",
            intended_use=("benchmark_only",),
            training_eligible=False,
            exclusion_reason="heldout_or_benchmark_prompt",
            benchmark_exact_match=(record.source,),
            source_record=record.source_record,
        )
