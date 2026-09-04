#!/usr/bin/env python3
"""Promote an already verified candidate catalog to the current formal commit.

The ordinary catalog builder is intentionally conservative and may rescan very
large image manifests.  This command is a bounded alternative for an existing
candidate whose artifact bytes are already frozen: it rebinds those bytes to a
clean, pushed commit and runs the same formal validator before installation.
It never changes the source release and never marks the metadata-only catalog
as training-ready.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

from data_catalog.builder import (
    CATALOG_SCHEMA,
    CONFIG_SNAPSHOT_NAME,
    CONFIG_SNAPSHOT_SCHEMA,
    DATA_RECORD_SCHEMA,
    REPOSITORY_ROOT,
    _artifact_contract,
    _builder_provenance,
    _release_id,
    _source_provenance,
    _tracked_config_path,
    enforce_git_gate,
    load_config,
    resolve_path,
    validate_release,
)
from data_catalog.io import sha256_file
from data_catalog.sources import four_k_lsdb_ids_sha256


def _absolute_without_symlinks(path: Path, *, label: str) -> Path:
    """Normalize a path while rejecting symlinks in every existing component."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    normalized = Path(os.path.abspath(os.fspath(candidate)))
    current = Path(normalized.anchor)
    for part in normalized.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} cannot contain a symlink: {current}")
    return normalized


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} must be a non-empty ordinary file: {path}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
    ):
        raise ValueError(f"{label} must be a non-empty ordinary file: {path}")
    return path


def _link_or_copy(source: Path, destination: Path) -> None:
    """Copy one artifact into the staging tree as an independent inode."""
    source = _regular_file(source, label="source artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
        target_handle.flush()
        os.fsync(target_handle.fileno())


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    """Write a new file and make its bytes durable before publication."""
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Make directory entries durable on filesystems that support fsync."""
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextlib.contextmanager
def _publication_lock(catalogs_root: Path):
    """Serialize formal promotions that share one catalogs directory."""
    lock_path = catalogs_root / ".promote.lock"
    flags = os.O_CREAT | os.O_RDWR
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags | nofollow, 0o600)
    except OSError as exc:
        raise ValueError(f"cannot open catalog publication lock: {lock_path}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise ValueError(f"cannot acquire catalog publication lock: {lock_path}") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_source_artifacts(
    source_release: Path, config: Mapping[str, Any]
) -> None:
    """Check the source release against the currently frozen artifact contract."""
    source_manifest = _json_object(source_release / "manifest.json", label="source manifest")
    if source_manifest.get("schema") != CATALOG_SCHEMA:
        raise ValueError("source release is not a catalog v1 release")
    if source_manifest.get("release_id") != source_release.name:
        raise ValueError("source manifest release_id does not match its directory")
    artifacts = source_manifest.get("artifacts")
    expected = _artifact_contract(config)
    if not isinstance(artifacts, list) or any(
        not isinstance(row, Mapping) for row in artifacts
    ):
        raise ValueError("source artifact descriptor is malformed")
    if [row.get("path") for row in artifacts] != [row["path"] for row in expected]:
        raise ValueError("source release does not contain the frozen 15-artifact order")
    for observed, contract in zip(artifacts, expected):
        if not isinstance(observed, Mapping):
            raise ValueError("source artifact descriptor is malformed")
        for key in ("schema", "rows", "bytes", "sha256"):
            if observed.get(key) != contract.get(key):
                raise ValueError(
                    f"source artifact {contract['path']} differs from the frozen contract"
                )
        _regular_file(source_release / contract["path"], label=contract["path"])
        if sha256_file(source_release / contract["path"]) != contract["sha256"]:
            raise ValueError(f"source artifact hash differs: {contract['path']}")


def _cross_checks(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = resolve_path(
        config["cross_checks"]["four_k_lsdb_manifest"], REPOSITORY_ROOT
    )
    cache_path = resolve_path(config["cross_checks"]["sana_cache_meta"], REPOSITORY_ROOT)
    cache = _json_object(cache_path, label="Sana cache metadata")
    ids_hash = four_k_lsdb_ids_sha256(manifest_path)
    if cache.get("ids_sha256") != ids_hash:
        raise ValueError("Sana text cache does not match the 4KLSDB manifest")
    return {
        "four_k_lsdb_ids_sha256": ids_hash,
        "sana_cache_ids_sha256": cache["ids_sha256"],
        "sana_cache_matches_four_k_lsdb": True,
    }


def _manifest_core(
    *, config: Mapping[str, Any], config_path: Path, config_bytes: bytes,
    git: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    config_repo_path = _tracked_config_path(
        config_path, REPOSITORY_ROOT, commit=str(git["commit"]), required=True
    )
    if config_repo_path is None:  # pragma: no cover - guarded by required=True
        raise ValueError("catalog config is not tracked")
    snapshot = {
        "path": CONFIG_SNAPSHOT_NAME,
        "schema": CONFIG_SNAPSHOT_SCHEMA,
        "bytes": len(config_bytes),
        "sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    return {
        "complete": False,
        "candidate_catalog_complete": True,
        "development_build": False,
        "record_schema": DATA_RECORD_SCHEMA,
        "config_repo_path": config_repo_path,
        "config_snapshot": snapshot,
        "git": dict(git),
        "verify_paths": True,
        "physical_source_count": len(config["physical_sources"]),
        "training_ready": False,
        "payload_integrity_policy": config["payload_integrity_policy"],
        "protected_prompt_source_counts": {
            row["id"]: row["expected_prompts"]
            for row in config["protected_prompt_sources"]
        },
        "protected_normalized_unique_prompts": config[
            "expected_protected_normalized_unique_prompts"
        ],
        "protected_unique_images": config["expected_protected_unique_images"],
        "source_provenance": _source_provenance(config, REPOSITORY_ROOT),
        "builder_provenance": _builder_provenance(REPOSITORY_ROOT),
        "cross_checks": _cross_checks(config),
        "artifacts": [dict(row) for row in artifacts],
    }


def promote_catalog(
    *, source_release: Path, config_path: Path, output_dir: Path | None = None
) -> Path:
    """Promote one frozen candidate catalog and return its new release path."""
    source_release = _absolute_without_symlinks(source_release, label="source release")
    config_path = _absolute_without_symlinks(config_path, label="catalog config")
    output_dir = _absolute_without_symlinks(
        output_dir or REPOSITORY_ROOT / "DATA", label="output directory"
    )
    catalogs_root = output_dir / "catalogs"
    if catalogs_root.is_symlink():
        raise ValueError(f"catalog directory cannot contain a symlink: {catalogs_root}")
    if catalogs_root.exists() and not catalogs_root.is_dir():
        raise ValueError(f"catalog directory must be a directory: {catalogs_root}")
    expected_output_dir = _absolute_without_symlinks(
        REPOSITORY_ROOT / "DATA", label="repository DATA directory"
    )
    if output_dir != expected_output_dir:
        raise ValueError(
            "formal promotion output directory must be the repository DATA directory"
        )
    if source_release.parent != catalogs_root:
        raise ValueError(
            f"source release must be a direct child of {catalogs_root}"
        )
    if source_release.name.startswith("."):
        raise ValueError("source release must be an installed catalog")
    if not source_release.is_dir():
        raise ValueError(f"source release must be a directory: {source_release}")
    config = load_config(config_path, require_frozen_contract=True)
    git = enforce_git_gate(REPOSITORY_ROOT, allow_dirty=False)
    _validate_source_artifacts(source_release, config)
    config_bytes = config_path.read_bytes()
    artifacts = _artifact_contract(config)
    core = _manifest_core(
        config=config,
        config_path=config_path,
        config_bytes=config_bytes,
        git=git,
        artifacts=artifacts,
    )
    release_id = _release_id(core)
    destination = catalogs_root / release_id
    catalogs_root.mkdir(parents=True, exist_ok=True)
    if catalogs_root.is_symlink() or not catalogs_root.is_dir():
        raise ValueError(f"catalog directory must be an ordinary directory: {catalogs_root}")
    with _publication_lock(catalogs_root):
        if destination.is_symlink() or destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError(f"destination release must be an ordinary directory: {destination}")
            existing = validate_release(
                destination,
                verify_paths=True,
                require_formal_catalog=True,
                require_training_ready=False,
            )
            if existing.get("release_id") != release_id:
                raise ValueError("existing release has an unexpected identity")
            return destination

        staging = Path(tempfile.mkdtemp(prefix=".catalog-promote-", dir=catalogs_root))
        installed = False
        try:
            for artifact in artifacts:
                _link_or_copy(
                    source_release / artifact["path"], staging / artifact["path"]
                )
            snapshot_path = staging / CONFIG_SNAPSHOT_NAME
            _write_bytes_fsync(snapshot_path, config_bytes)
            manifest = {"schema": CATALOG_SCHEMA, "release_id": release_id, **core}
            _write_bytes_fsync(
                staging / "manifest.json",
                (
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8"),
            )
            _fsync_directory(staging)
            os.replace(staging, destination)
            installed = True
            _fsync_directory(catalogs_root)
            validated = validate_release(
                destination,
                verify_paths=True,
                require_formal_catalog=True,
                require_training_ready=False,
            )
            if validated != manifest:
                raise ValueError("promoted manifest changed during validation")
            return destination
        except BaseException as primary_error:
            rollback_error = None
            try:
                if installed:
                    shutil.rmtree(destination)
                    _fsync_directory(catalogs_root)
                else:
                    shutil.rmtree(staging)
                    _fsync_directory(catalogs_root)
            except BaseException as error:
                rollback_error = error
            if rollback_error is not None:
                raise primary_error from rollback_error
            raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "eval-pipeline/configs/data_catalog_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="must be the repository DATA directory (kept for API compatibility)",
    )
    args = parser.parse_args(argv)
    release = promote_catalog(
        source_release=args.source_release,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    manifest = _json_object(release / "manifest.json", label="promoted manifest")
    print(
        json.dumps(
            {
                "release_dir": str(release),
                "release_id": manifest["release_id"],
                "candidate_catalog_complete": manifest["candidate_catalog_complete"],
                "training_ready": manifest["training_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
