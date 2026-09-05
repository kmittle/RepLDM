#!/usr/bin/env python3
"""Build an auditable, ordinary-file ImageReward runtime closure.

The formal latent-renderer runtime loads ImageReward from a staged source tree
and never resolves a cache path or a model name.  This utility copies the
already downloaded checkpoint, BERT tokenizer files, and source checkout into
one immutable asset directory, then writes the exact config and file manifest
consumed by ``latent_renderer_training.runtime_factory``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Union


PREPROCESS_SHA256 = (
    "a29b943538f5ec4cd920800601cf30400c9f31b583c199a845849a8529764dba"
)
BERT_FILES = ("vocab.txt", "tokenizer_config.json", "tokenizer.json", "config.json")


def _absolute_path(raw: str, *, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


def _output_path(raw: Union[str, Path], *, label: str) -> Path:
    """Normalize an output path and reject symlinked existing components."""
    path = _absolute_path(os.fspath(raw), label=label)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            status = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} cannot contain a symlink: {current}")
    return path


def _ordinary_file(raw: Union[str, Path], *, label: str) -> Path:
    path = _absolute_path(os.fspath(raw), label=label)
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    if not path.is_file() or path.is_symlink() or status.st_size <= 0:
        raise ValueError(f"{label} must be a non-empty ordinary file: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} cannot contain a symlink: {current}")
    return path


def _source_file(raw: Union[str, Path], *, label: str) -> Path:
    """Resolve a read-only cache link before copying it into the closure."""
    path = _absolute_path(os.fspath(raw), label=label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} must resolve to a non-empty regular file: {path}")
    return resolved


def _ordinary_directory(raw: Union[str, Path], *, label: str) -> Path:
    path = _absolute_path(os.fspath(raw), label=label)
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be an ordinary directory: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} cannot contain a symlink: {current}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(root: Path) -> list[tuple[Path, str]]:
    """Return package Python files with paths relative to the checkout root."""
    package = root / "ImageReward"
    _ordinary_directory(package, label="ImageReward package")
    result: list[tuple[Path, str]] = []
    pending = [package]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise ValueError(f"cannot inspect ImageReward package: {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"cannot inspect ImageReward source {relative}") from exc
            if stat.S_ISLNK(status.st_mode):
                raise ValueError(f"ImageReward source cannot contain a symlink: {relative}")
            if stat.S_ISDIR(status.st_mode):
                pending.append(path)
            elif stat.S_ISREG(status.st_mode) and path.suffix == ".py":
                result.append(
                    (_ordinary_file(path, label=f"ImageReward source {relative}"), relative)
                )
            elif not stat.S_ISREG(status.st_mode):
                raise ValueError(f"ImageReward source is not an ordinary file: {relative}")
    result.sort(key=lambda row: row[1])
    if not any(relative == "ImageReward/__init__.py" for _path, relative in result):
        raise ValueError("ImageReward source closure omits ImageReward/__init__.py")
    if not result:
        raise ValueError("ImageReward source closure is empty")
    return result


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite asset: {destination}")
    source_descriptor = os.open(
        os.fspath(source),
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    destination_descriptor = -1
    destination_created = False
    copy_succeeded = False
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError(f"source must be a non-empty ordinary file: {source}")
        destination_descriptor = os.open(
            os.fspath(destination),
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        destination_created = True
        os.fchmod(destination_descriptor, 0o400)
        while True:
            chunk = os.read(source_descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("short write while copying ImageReward asset")
                view = view[written:]
        after = os.fstat(source_descriptor)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
            raise RuntimeError(f"source changed while copying ImageReward asset: {source}")
        os.fsync(destination_descriptor)
        copy_succeeded = True
    finally:
        if destination_descriptor != -1:
            os.close(destination_descriptor)
        os.close(source_descriptor)
        if destination_created and not copy_succeeded:
            destination.unlink(missing_ok=True)


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _reject_path_overlap(paths: list[tuple[str, Path]]) -> None:
    for index, (left_label, left) in enumerate(paths):
        for right_label, right in paths[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    f"{left_label} and {right_label} paths must not overlap"
                )


def _unlink_file_if_present(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)


def _missing_ancestors(path: Path) -> tuple[Path, ...]:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return tuple(missing)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            file_descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if file_descriptor != -1:
            os.close(file_descriptor)
        temporary.unlink(missing_ok=True)


def _copy_sources(source_root: Path, destination_root: Path) -> list[str]:
    rows = _source_files(source_root)
    relative_names: list[str] = []
    for source, relative in rows:
        destination = destination_root / relative
        _copy(source, destination)
        relative_names.append(relative)
    return relative_names


def build(
    *,
    checkpoint: str,
    med_config: str,
    source_root: str,
    tokenizer_root: str,
    asset_root: Path,
    output_config: Path,
    output_manifest: Path,
) -> None:
    checkpoint_path = _ordinary_file(checkpoint, label="ImageReward checkpoint")
    med_config_path = _ordinary_file(med_config, label="ImageReward med_config")
    source_path = _ordinary_directory(source_root, label="ImageReward source root")
    tokenizer_path = _ordinary_directory(tokenizer_root, label="BERT tokenizer root")
    if checkpoint_path == med_config_path:
        raise ValueError("ImageReward checkpoint and med_config must differ")

    asset_root = _output_path(asset_root, label="asset root")
    output_config = _output_path(output_config, label="output config")
    output_manifest = _output_path(output_manifest, label="output manifest")
    if output_config == output_manifest:
        raise ValueError("output config and manifest must differ")
    _reject_path_overlap(
        [
            ("asset root", asset_root),
            ("output config", output_config),
            ("output manifest", output_manifest),
        ]
    )
    if output_config.exists() or output_manifest.exists() or asset_root.exists():
        raise FileExistsError(
            "refusing to overwrite existing ImageReward asset output; choose a new directory"
        )

    tokenizer_sources = [
        _source_file(tokenizer_path / filename, label=f"BERT tokenizer {filename}")
        for filename in BERT_FILES
    ]
    _reject_path_overlap(
        [
            ("asset root", asset_root),
            ("output config", output_config),
            ("output manifest", output_manifest),
            ("ImageReward source root", source_path),
            ("BERT tokenizer root", tokenizer_path),
            ("ImageReward checkpoint", checkpoint_path),
            ("ImageReward med_config", med_config_path),
        ]
        + [
            (f"BERT tokenizer {filename}", source)
            for filename, source in zip(BERT_FILES, tokenizer_sources)
        ]
    )
    source_destination = asset_root / "source"
    tokenizer_destination = asset_root / "tokenizer"
    checkpoint_destination = asset_root / "checkpoint" / "ImageReward.pt"
    med_destination = asset_root / "med_config.json"
    missing_asset_ancestors = _missing_ancestors(asset_root)
    asset_root_created = False
    try:
        asset_root.mkdir(mode=0o700, parents=True)
        asset_root_created = True
        _copy(checkpoint_path, checkpoint_destination)
        _copy(med_config_path, med_destination)
        source_files = _copy_sources(source_path, source_destination)
        for source, filename in zip(tokenizer_sources, BERT_FILES):
            _copy(source, tokenizer_destination / filename)

        manifest_files = [checkpoint_destination, med_destination]
        manifest_files.extend(source_destination / relative for relative in source_files)
        manifest_files.extend(tokenizer_destination / filename for filename in BERT_FILES)
        manifest = {
            "schema": "repldm.file_manifest.v1",
            "files": [_binding(path) for path in manifest_files],
        }
        config = {
            "schema": "repldm.imagereward_runtime.v1",
            "implementation": "ImageReward-v1.0",
            "dtype": "float32",
            "preprocess_sha256": PREPROCESS_SHA256,
            "checkpoint": str(checkpoint_destination),
            "med_config": str(med_destination),
            "source_root": str(source_destination),
            "source_files": source_files,
            "tokenizer_root": str(tokenizer_destination),
            "tokenizer_files": list(BERT_FILES),
        }
        _write_json(output_manifest, manifest)
        _write_json(output_config, config)
        print(
            json.dumps(
                {
                    "config": str(output_config),
                    "manifest": str(output_manifest),
                    "asset_count": len(manifest_files),
                },
                sort_keys=True,
            )
        )
    except BaseException:
        if (asset_root_created or asset_root in missing_asset_ancestors) and asset_root.is_dir():
            shutil.rmtree(asset_root, ignore_errors=True)
        _unlink_file_if_present(output_config)
        _unlink_file_if_present(output_manifest)
        for directory in missing_asset_ancestors:
            if directory == asset_root or directory.is_symlink():
                continue
            try:
                directory.rmdir()
            except OSError:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--med-config", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--tokenizer-root", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    build(
        checkpoint=args.checkpoint,
        med_config=args.med_config,
        source_root=args.source_root,
        tokenizer_root=args.tokenizer_root,
        asset_root=args.asset_root,
        output_config=args.output_config,
        output_manifest=args.output_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
