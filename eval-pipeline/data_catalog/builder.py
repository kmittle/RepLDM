"""Orchestrate a versioned, auditable catalog build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .io import (
    canonical_json_bytes,
    iter_jsonl,
    sha256_file,
    verify_image_paths,
    write_records,
)
from .protected import (
    iter_holdout_records,
    load_protected_prompts,
    unique_protected_image_count,
)
from .schema import DATA_RECORD_SCHEMA, validate_record
from .sources import SourceContext, four_k_lsdb_ids_sha256, iter_dataset_records, resolve_path


CATALOG_SCHEMA = "repldm.data_catalog.v1"
CONFIG_SNAPSHOT_SCHEMA = "repldm.data_catalog_config_snapshot.v1"
CONFIG_SNAPSHOT_NAME = "catalog-config.yaml"
TRAINING_VIEW_SCHEMA = "repldm.training_view.v1"
PHYSICAL_SOURCE_SCHEMA = "repldm.physical_source.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "eval-pipeline/configs/data_catalog_v1.yaml"
REQUIRED_SOURCE_ROOTS = {
    "pixel_render_miah204": Path("/mnt/miah204/bycao/Pixel-Render"),
    "sana_miah204": Path("/mnt/miah204/bycao/Sana"),
    "pixel_render_miah209": Path("/mnt/miah209/bycao/Pixel-Render"),
    "sana_data_miah209": Path("/mnt/miah209/bycao/sana_data"),
    "sana_runs_miah209": Path("/mnt/miah209/bycao/sana_runs"),
}
REQUIRED_SOURCE_ROOT_IDS = set(REQUIRED_SOURCE_ROOTS)
DATA_STAT_KEYS = {
    "training_eligible_rows",
    "rows_with_prompt",
    "rows_with_image",
    "missing_images",
    "benchmark_exact_match_rows",
    "modalities",
    "splits",
}
_DATA_STAT_COUNT_KEYS = {
    "rows",
    "training_eligible_rows",
    "rows_with_prompt",
    "rows_with_image",
    "missing_images",
    "benchmark_exact_match_rows",
}


def _nonnegative_int(value: object, *, label: str) -> bool:
    """Return whether ``value`` is an actual integer count, not YAML ``true``/``false``."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _strict_bool(value: object) -> bool:
    """Return whether a config flag is a real YAML boolean, not a truthy surrogate."""
    return isinstance(value, bool)


def _validate_stats_contract(
    value: object, *, label: str, include_rows: bool
) -> None:
    """Validate the typed aggregate statistics embedded in the catalog config."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must pin all semantic artifact statistics")
    expected_keys = DATA_STAT_KEYS | ({"rows"} if include_rows else set())
    if set(value) != expected_keys:
        raise ValueError(f"{label} must pin all semantic artifact statistics")
    for key in _DATA_STAT_COUNT_KEYS:
        if key == "rows" and not include_rows:
            continue
        if not _nonnegative_int(value.get(key), label=f"{label}.{key}"):
            raise ValueError(f"{label}.{key} must be a non-negative integer")
    for key in ("modalities", "splits"):
        mapping = value.get(key)
        if not isinstance(mapping, dict) or any(
            not isinstance(name, str)
            or not name
            or not _nonnegative_int(count, label=f"{label}.{key}.{name}")
            for name, count in (mapping.items() if isinstance(mapping, dict) else ())
        ):
            raise ValueError(f"{label}.{key} must map names to non-negative integers")


def _physical_root_map(
    config: Mapping[str, Any], repository_root: Path
) -> dict[str, Path]:
    """Resolve configured physical roots for source-payload containment."""
    roots = {
        str(row["id"]): resolve_path(row["path"], repository_root).resolve()
        for row in config["physical_sources"]
    }
    roots.setdefault("repldm", repository_root.resolve())
    return roots


def _run_git(arguments: Sequence[str], repository_root: Path) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _run_git_bytes(arguments: Sequence[str], repository_root: Path) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def git_state(repository_root: Path) -> dict[str, Any]:
    commit = _run_git(("rev-parse", "HEAD"), repository_root)
    status = _run_git(("status", "--porcelain=v1", "--untracked-files=all"), repository_root)
    try:
        upstream = _run_git(("rev-parse", "@{upstream}"), repository_root)
        upstream_ref = _run_git(
            ("rev-parse", "--symbolic-full-name", "@{upstream}"),
            repository_root,
        )
    except subprocess.CalledProcessError:
        upstream = None
        upstream_ref = None
    return {
        "commit": commit,
        "branch": _run_git(("branch", "--show-current"), repository_root),
        "dirty": bool(status),
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "upstream_ref": upstream_ref,
        "upstream_commit": upstream,
        "pushed": upstream == commit if upstream else False,
    }


def enforce_git_gate(repository_root: Path, *, allow_dirty: bool) -> dict[str, Any]:
    state = git_state(repository_root)
    if not allow_dirty and (state["dirty"] or not state["pushed"]):
        raise RuntimeError(
            "formal catalog builds require a clean worktree at the pushed upstream commit; "
            "use --allow-dirty only for development smoke builds"
        )
    return state


def _validate_recorded_upstream_ancestry(
    recorded_commit: object,
    upstream_ref: object,
    *,
    repository_root: Path,
) -> None:
    """Require a published catalog commit to remain on its recorded upstream history."""
    if not isinstance(recorded_commit, str) or not recorded_commit:
        raise ValueError("formal release has an invalid recorded Git commit")
    if not isinstance(upstream_ref, str) or not upstream_ref:
        raise ValueError("formal release has an invalid recorded upstream ref")
    if not upstream_ref.startswith("refs/remotes/"):
        raise ValueError("formal release upstream ref must be fully qualified")
    try:
        _run_git(("check-ref-format", upstream_ref), repository_root)
        upstream_commit = _run_git(
            ("rev-parse", "--verify", f"{upstream_ref}^{{commit}}"),
            repository_root,
        )
        _run_git(
            ("merge-base", "--is-ancestor", recorded_commit, upstream_commit),
            repository_root,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "recorded catalog commit is no longer reachable from the recorded upstream ref"
        ) from exc


def _validate_formal_git_record(git: Mapping[str, Any]) -> None:
    commit = git.get("commit")
    if (
        not isinstance(commit, str)
        or not commit
        or git.get("dirty") is not False
        or git.get("pushed") is not True
        or git.get("upstream_commit") != commit
        or not isinstance(git.get("upstream_ref"), str)
        or not git["upstream_ref"].startswith("refs/remotes/")
    ):
        raise ValueError("formal release Git provenance is internally inconsistent")


def _load_config_bytes(payload_bytes: bytes, *, source: str) -> dict[str, Any]:
    payload = yaml.safe_load(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {source}")
    if payload.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"unexpected catalog config schema: {payload.get('schema')!r}")
    roots = payload.get("physical_sources")
    if not isinstance(roots, list):
        raise ValueError("physical_sources must be a list")
    if len(roots) != len(REQUIRED_SOURCE_ROOT_IDS) or not all(
        isinstance(row, dict) for row in roots
    ):
        raise ValueError("physical_sources must contain exactly five mappings")
    root_ids = [row.get("id") for row in roots]
    if set(root_ids) != REQUIRED_SOURCE_ROOT_IDS or len(root_ids) != len(set(root_ids)):
        raise ValueError(
            "physical_sources must name exactly the five requested directories; "
            f"got {sorted(str(value) for value in root_ids)}"
        )
    observed_roots = {
        row["id"]: Path(str(row.get("path", ""))).resolve() for row in roots
    }
    expected_roots = {key: value.resolve() for key, value in REQUIRED_SOURCE_ROOTS.items()}
    if observed_roots != expected_roots:
        raise ValueError("physical_sources paths must match the five requested directories")

    declared_roots = dict(expected_roots)
    declared_roots["repldm"] = REPOSITORY_ROOT.resolve()

    def validate_declared_path(value: object, source_ids: object, label: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty path")
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError(f"{label} must declare at least one physical source root")
        names = [str(name) for name in source_ids]
        if any(name not in declared_roots for name in names):
            raise ValueError(f"{label} declares an unknown physical source root")
        candidate = resolve_path(value, REPOSITORY_ROOT).resolve(strict=False)
        if not any(
            candidate == root or root in candidate.parents for root in (declared_roots[name] for name in names)
        ):
            raise ValueError(f"{label} is outside its declared physical source roots")

    license_policy = payload.get("training_license_policy")
    if not isinstance(license_policy, dict) or license_policy.get("mode") != "fail_closed":
        raise ValueError("training_license_policy must use fail_closed mode")
    allowed_statuses = license_policy.get("allowed_statuses")
    if (
        not isinstance(allowed_statuses, list)
        or not allowed_statuses
        or any(not isinstance(value, str) or not value for value in allowed_statuses)
        or len(allowed_statuses) != len(set(allowed_statuses))
    ):
        raise ValueError("training license statuses must be unique non-empty strings")

    protected_sources = payload.get("protected_prompt_sources")
    if not isinstance(protected_sources, list) or not protected_sources:
        raise ValueError("protected_prompt_sources must be a non-empty list")
    protected_ids = [row.get("id") for row in protected_sources if isinstance(row, dict)]
    if len(protected_ids) != len(protected_sources) or len(protected_ids) != len(
        set(protected_ids)
    ):
        raise ValueError("protected prompt source ids must be unique")
    for row in protected_sources:
        if (
            not _nonnegative_int(row.get("expected_prompts"), label="expected_prompts")
            or not _nonnegative_int(row.get("expected_bytes"), label="expected_bytes")
            or not isinstance(row.get("expected_sha256"), str)
            or len(row["expected_sha256"]) != 64
        ):
            raise ValueError(f"protected source {row.get('id')} lacks a pinned count/hash")
        if row.get("image_path_mode") == "sharded_id_jpg":
            shard_glob = row.get("image_shard_glob")
            image_subdir = row.get("image_subdir")
            if (
                not isinstance(row.get("image_root"), str)
                or not row["image_root"]
                or not isinstance(row.get("image_id_key"), str)
                or not row["image_id_key"]
                or not isinstance(shard_glob, str)
                or not shard_glob
                or Path(shard_glob).is_absolute()
                or Path(shard_glob).name != shard_glob
                or not isinstance(image_subdir, str)
                or not image_subdir
                or Path(image_subdir).is_absolute()
                or Path(image_subdir).name != image_subdir
                or not _nonnegative_int(
                    row.get("expected_image_shards"), label="expected_image_shards"
                )
                or row["expected_image_shards"] <= 0
                or not _nonnegative_int(
                    row.get("expected_unique_images"), label="expected_unique_images"
                )
                or row["expected_unique_images"] <= 0
            ):
                raise ValueError(
                    f"protected source {row.get('id')} has an invalid sharded image contract"
                )
    for protected in protected_sources:
        validate_declared_path(
            protected.get("path"), protected.get("source_roots"),
            f"protected source {protected.get('id')} metadata path",
        )
        if protected.get("image_root") is not None:
            validate_declared_path(
                protected.get("image_root"), protected.get("source_roots"),
                f"protected source {protected.get('id')} image root",
            )
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    if not all(isinstance(row, dict) for row in datasets):
        raise ValueError("every dataset entry must be a mapping")
    dataset_ids = [row.get("id") for row in datasets]
    if any(not isinstance(value, str) or not value for value in dataset_ids):
        raise ValueError("every dataset must have a non-empty string id")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("dataset ids must be unique")
    output_names = [row.get("output") for row in datasets]
    if any(
        not isinstance(value, str)
        or Path(value).name != value
        or not value.endswith(".jsonl")
        for value in output_names
    ):
        raise ValueError("dataset outputs must be JSONL basenames")
    if len(output_names) != len(set(output_names)):
        raise ValueError("dataset output names must be unique")
    reserved_outputs = {
        "benchmark_holdouts.jsonl",
        "source_inventory.jsonl",
        "training_candidates.jsonl",
        "training_views.jsonl",
    }
    if reserved_outputs.intersection(output_names):
        raise ValueError("dataset outputs must not use reserved catalog artifact names")
    for dataset in datasets:
        if not _nonnegative_int(dataset.get("expected_rows"), label="expected_rows"):
            raise ValueError(f"dataset {dataset.get('id')} lacks expected_rows")
        if not _strict_bool(dataset.get("include_in_training_candidates")):
            raise ValueError(
                f"dataset {dataset.get('id')} include_in_training_candidates must be boolean"
            )
        expected_stats = dataset.get("expected_stats")
        _validate_stats_contract(
            expected_stats,
            label=f"dataset {dataset.get('id')} expected_stats",
            include_rows=False,
        )
        source_ids = dataset.get("source_roots")
        for key in ("root", "manifest", "style_mapper", "data_root", "image_root"):
            if dataset.get(key) is not None:
                validate_declared_path(
                    dataset.get(key), source_ids,
                    f"dataset {dataset.get('id')} {key}",
                )
        splits = dataset.get("splits", [])
        if not isinstance(splits, list):
            raise ValueError(f"dataset {dataset.get('id')} splits must be a list")
        for split in splits:
            if not isinstance(split, dict):
                raise ValueError(f"dataset {dataset.get('id')} split must be a mapping")
            if not _strict_bool(split.get("training_eligible")):
                raise ValueError(
                    f"dataset {dataset.get('id')} split {split.get('name')} "
                    "training_eligible must be boolean"
                )
            if "exclusion_reason" in split and split["exclusion_reason"] is not None and (
                not isinstance(split["exclusion_reason"], str)
                or not split["exclusion_reason"]
            ):
                raise ValueError(
                    f"dataset {dataset.get('id')} split {split.get('name')} "
                    "exclusion_reason must be a non-empty string or null"
                )
            validate_declared_path(
                split.get("path"), source_ids,
                f"dataset {dataset.get('id')} split {split.get('name')} path",
            )

    for value in payload.get("provenance_files", ()):
        candidate = resolve_path(value, REPOSITORY_ROOT).resolve(strict=False)
        if not any(candidate == root or root in candidate.parents for root in declared_roots.values()):
            raise ValueError(f"provenance file is outside all declared physical roots: {value}")
    for name, value in payload.get("cross_checks", {}).items():
        candidate = resolve_path(value, REPOSITORY_ROOT).resolve(strict=False)
        if not any(candidate == root or root in candidate.parents for root in declared_roots.values()):
            raise ValueError(f"cross-check {name} is outside all declared physical roots")

    expected_order = payload.get("expected_artifact_order")
    derived_order = [
        "benchmark_holdouts.jsonl",
        *output_names,
        "training_candidates.jsonl",
        "training_views.jsonl",
        "source_inventory.jsonl",
    ]
    if expected_order != derived_order:
        raise ValueError("expected_artifact_order must exactly match the configured build order")
    expected_training = payload.get("expected_training_candidate_stats")
    _validate_stats_contract(
        expected_training, label="expected_training_candidate_stats", include_rows=True
    )
    expected_holdout = payload.get("expected_holdout_stats")
    _validate_stats_contract(
        expected_holdout, label="expected_holdout_stats", include_rows=True
    )
    if expected_holdout["rows"] != sum(
        row["expected_prompts"] for row in protected_sources
    ):
        raise ValueError("held-out aggregate count does not match protected-source counts")
    if not _nonnegative_int(
        payload.get("expected_protected_normalized_unique_prompts"),
        label="expected_protected_normalized_unique_prompts",
    ):
        raise ValueError("expected protected unique-prompt count is missing")
    expected_unique_images = payload.get("expected_protected_unique_images")
    has_sharded_images = any(
        row.get("image_path_mode") == "sharded_id_jpg" for row in protected_sources
    )
    if expected_unique_images is not None and (
        not isinstance(expected_unique_images, int)
        or isinstance(expected_unique_images, bool)
        or expected_unique_images < 0
    ):
        raise ValueError("expected protected unique-image count is invalid")
    if has_sharded_images and expected_unique_images is None:
        raise ValueError("sharded protected sources require a unique-image count")

    views = payload.get("training_views")
    if not isinstance(views, list) or not views:
        raise ValueError("training_views must be a non-empty list")
    view_ids = [row.get("id") for row in views if isinstance(row, dict)]
    if len(view_ids) != len(views) or len(view_ids) != len(set(view_ids)):
        raise ValueError("training view ids must be unique")
    for view in views:
        if (
            view.get("schema") != TRAINING_VIEW_SCHEMA
            or view.get("artifact") != "training_candidates.jsonl"
            or not _nonnegative_int(view.get("expected_rows"), label="view.expected_rows")
        ):
            raise ValueError(f"invalid training view contract: {view.get('id')}")
        filters = view.get("filter")
        if not isinstance(filters, dict) or set(filters) not in (
            {"prompt_is_not_null"},
            {"modality_in"},
        ):
            raise ValueError(f"unsupported training view filter: {view.get('id')}")
        if "prompt_is_not_null" in filters and not _strict_bool(
            filters["prompt_is_not_null"]
        ):
            raise ValueError(
                f"training view filter must use a boolean prompt_is_not_null: {view.get('id')}"
            )
        if "modality_in" in filters:
            modalities = filters["modality_in"]
            if (
                not isinstance(modalities, list)
                or not modalities
                or any(not isinstance(value, str) or not value for value in modalities)
                or len(modalities) != len(set(modalities))
            ):
                raise ValueError(
                    f"training view modality_in must be unique non-empty strings: {view.get('id')}"
                )

    payload_policy = payload.get("payload_integrity_policy")
    if (
        not isinstance(payload_policy, dict)
        or payload_policy.get("catalog_role") != "candidate_inventory"
        or payload_policy.get("training_ready") is not False
    ):
        raise ValueError("v1 is a candidate inventory and must not authorize training")

    expected_hashes = payload.get("expected_artifact_hashes")
    if not isinstance(expected_hashes, dict):
        raise ValueError("expected_artifact_hashes must be a mapping")
    if expected_hashes:
        if list(expected_hashes) != expected_order:
            raise ValueError("expected_artifact_hashes must follow the exact artifact order")
        for name, values in expected_hashes.items():
            if (
                not isinstance(values, dict)
                or set(values) != {"bytes", "sha256"}
                or not _nonnegative_int(values["bytes"], label=f"{name}.bytes")
                or not isinstance(values["sha256"], str)
                or len(values["sha256"]) != 64
            ):
                raise ValueError(f"invalid pinned artifact hash for {name}")
    return payload


def load_config(path: Path, *, require_frozen_contract: bool = False) -> dict[str, Any]:
    payload = _load_config_bytes(path.read_bytes(), source=str(path))
    if require_frozen_contract and not payload["expected_artifact_hashes"]:
        raise ValueError("formal catalogs require pinned bytes and SHA-256 for every artifact")
    return payload


def _resolve_output_dir(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _source_provenance(
    config: Mapping[str, Any], repository_root: Path
) -> list[dict[str, Any]]:
    configured_paths = list(config["provenance_files"])
    configured_paths.extend(spec["path"] for spec in config["protected_prompt_sources"])
    configured_paths.extend(config.get("cross_checks", {}).values())
    for dataset in config["datasets"]:
        for key in ("manifest", "style_mapper"):
            if key in dataset:
                configured_paths.append(dataset[key])
        configured_paths.extend(split["path"] for split in dataset.get("splits", ()))

    paths = {
        resolve_path(value, repository_root).resolve() for value in configured_paths
    }
    result = []
    for path in sorted(paths, key=str):
        if not path.is_file():
            raise FileNotFoundError(f"provenance file is missing: {path}")
        result.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside the repository: {path}") from exc


def _tracked_config_path(
    config_path: Path,
    repository_root: Path,
    *,
    commit: str,
    required: bool,
) -> str | None:
    try:
        relative = _repository_relative(config_path, repository_root)
    except ValueError:
        if required:
            raise ValueError("formal catalog config must be inside the repository")
        return None
    try:
        committed = _run_git_bytes(("show", f"{commit}:{relative}"), repository_root)
    except subprocess.CalledProcessError as exc:
        if required:
            raise ValueError("formal catalog config must be tracked at the recorded commit") from exc
        return None
    if committed != config_path.read_bytes():
        if required:
            raise ValueError("formal catalog config differs from the recorded commit")
        return None
    return relative


def _builder_paths(repository_root: Path) -> list[Path]:
    package_root = repository_root / "eval-pipeline/data_catalog"
    paths = sorted(package_root.rglob("*.py"))
    paths += [repository_root / "eval-pipeline/build_data_catalog.py"]
    return paths


def _builder_provenance(repository_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "repo_path": _repository_relative(path, repository_root),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _builder_paths(repository_root)
    ]


def _builder_repo_paths_at_commit(
    repository_root: Path,
    *,
    commit: str,
) -> list[str]:
    package_prefix = "eval-pipeline/data_catalog/"
    entrypoint = "eval-pipeline/build_data_catalog.py"
    tracked = _run_git(
        (
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            package_prefix.rstrip("/"),
            entrypoint,
        ),
        repository_root,
    ).splitlines()
    package_paths = sorted(
        path
        for path in tracked
        if path.startswith(package_prefix) and path.endswith(".py")
    )
    if entrypoint not in tracked:
        raise ValueError("recorded commit lacks the catalog builder entrypoint")
    return [*package_paths, entrypoint]


def _validate_builder_provenance(
    recorded: Any,
    *,
    repository_root: Path,
    commit: str,
) -> None:
    if not isinstance(recorded, list):
        raise ValueError("builder provenance must be a list")
    expected_paths = _builder_repo_paths_at_commit(repository_root, commit=commit)
    if [row.get("repo_path") for row in recorded if isinstance(row, dict)] != expected_paths:
        raise ValueError("builder provenance does not contain the exact builder file set")
    for row in recorded:
        blob = _run_git_bytes(("show", f"{commit}:{row['repo_path']}"), repository_root)
        if len(blob) != row.get("bytes") or hashlib.sha256(blob).hexdigest() != row.get(
            "sha256"
        ):
            raise ValueError(f"builder provenance differs from Git: {row['repo_path']}")


def _release_id(manifest_core: Mapping[str, Any]) -> str:
    """Bind the release name to metadata and every generated artifact hash."""
    digest = hashlib.sha256(canonical_json_bytes(manifest_core)).hexdigest()
    return f"catalog-{digest[:20]}"


def _generic_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    schema: str,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    with path.open("xb") as handle:
        for row in rows:
            payload = canonical_json_bytes(row)
            handle.write(payload)
            digest.update(payload)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "schema": schema,
        "rows": count,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _physical_source_inventory(
    config: Mapping[str, Any], repository_root: Path
) -> list[dict[str, Any]]:
    rows = []
    for spec in config["physical_sources"]:
        path = resolve_path(spec["path"], repository_root).resolve()
        row = {
            "schema": PHYSICAL_SOURCE_SCHEMA,
            "id": spec["id"],
            "path": str(path),
            "exists": path.is_dir(),
            "role": spec["role"],
            "logical_datasets": list(spec["logical_datasets"]),
            "notes": spec["notes"],
        }
        if not row["exists"]:
            raise FileNotFoundError(f"physical source directory is missing: {path}")
        rows.append(row)
    if len(rows) != 5:
        raise AssertionError("physical source inventory must contain exactly five rows")
    return rows


def _publish_current(output_dir: Path, release_dir: Path) -> None:
    current = output_dir / "current"
    temporary = output_dir / ".current.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(Path("catalogs") / release_dir.name, target_is_directory=True)
    os.replace(temporary, current)
    directory_fd = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _install_catalog_release(
    staging: Path,
    release_dir: Path,
    catalogs_dir: Path,
    manifest: Mapping[str, Any],
    *,
    verify_paths: bool,
    require_formal_catalog: bool,
    require_training_ready: bool = False,
) -> None:
    """Install a staged release and remove it if post-rename validation fails."""
    if release_dir.exists():
        existing = validate_release(
            release_dir,
            verify_paths=verify_paths,
            require_formal_catalog=require_formal_catalog,
            require_training_ready=require_training_ready,
        )
        if existing != dict(manifest):
            raise ValueError(
                f"release id collision with different manifest: {release_dir}"
            )
        shutil.rmtree(staging)
        return

    installed = False
    try:
        os.replace(staging, release_dir)
        installed = True
        _fsync_directory(catalogs_dir)
        validate_release(
            release_dir,
            verify_paths=verify_paths,
            require_formal_catalog=require_formal_catalog,
            require_training_ready=require_training_ready,
        )
    except BaseException as primary_error:
        rollback_error = None
        if installed:
            try:
                shutil.rmtree(release_dir)
                _fsync_directory(catalogs_dir)
            except BaseException as error:
                rollback_error = error
        if rollback_error is not None:
            raise primary_error from rollback_error
        raise


def _artifact_contract(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = [
        {
            "path": "benchmark_holdouts.jsonl",
            "schema": DATA_RECORD_SCHEMA,
            **config["expected_holdout_stats"],
        }
    ]
    for dataset in config["datasets"]:
        contracts.append(
            {
                "path": dataset["output"],
                "schema": DATA_RECORD_SCHEMA,
                "rows": dataset["expected_rows"],
                **dataset["expected_stats"],
            }
        )
    contracts.extend(
        (
            {
                "path": "training_candidates.jsonl",
                "schema": DATA_RECORD_SCHEMA,
                **config["expected_training_candidate_stats"],
            },
            {
                "path": "training_views.jsonl",
                "schema": TRAINING_VIEW_SCHEMA,
                "rows": len(config["training_views"]),
            },
            {
                "path": "source_inventory.jsonl",
                "schema": PHYSICAL_SOURCE_SCHEMA,
                "rows": len(config["physical_sources"]),
            },
        )
    )
    hashes = config["expected_artifact_hashes"]
    if hashes:
        for contract in contracts:
            contract.update(hashes[contract["path"]])
    if [row["path"] for row in contracts] != config["expected_artifact_order"]:
        raise AssertionError("internal artifact contract order mismatch")
    return contracts


def _assert_stats_match(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for key, expected_value in expected.items():
        if observed.get(key) != expected_value:
            raise ValueError(
                f"{context} has {key}={observed.get(key)!r}, expected {expected_value!r}"
            )


def _assert_artifact_contract(
    artifacts: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    contracts = _artifact_contract(config)
    if len(artifacts) != len(contracts):
        raise ValueError("artifact inventory length differs from the frozen config")
    for observed, expected in zip(artifacts, contracts):
        _assert_stats_match(observed, expected, context=str(expected["path"]))


def _write_config_snapshot(path: Path, config_path: Path) -> dict[str, Any]:
    payload = config_path.read_bytes()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "schema": CONFIG_SNAPSHOT_SCHEMA,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _compare_jsonl_derivation(
    actual_path: Path,
    expected_rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    with actual_path.open("rb") as actual:
        for line_number, expected_row in enumerate(expected_rows, 1):
            observed = actual.readline()
            expected = canonical_json_bytes(expected_row)
            if observed != expected:
                raise ValueError(f"{label} differs from its source derivation at row {line_number}")
        if actual.readline():
            raise ValueError(f"{label} contains rows beyond its source derivation")


def _compare_training_candidate_derivation(
    release_dir: Path,
    config: Mapping[str, Any],
) -> None:
    allowed = set(config["training_license_policy"]["allowed_statuses"])
    candidate_path = release_dir / "training_candidates.jsonl"
    emitted = 0
    with candidate_path.open("rb") as candidates:
        for dataset in config["datasets"]:
            if not dataset["include_in_training_candidates"]:
                continue
            with (release_dir / dataset["output"]).open("rb") as source:
                for source_line in source:
                    row = json.loads(source_line)
                    if not row["training_eligible"]:
                        continue
                    if row["license_status"] not in allowed:
                        raise ValueError(
                            f"training candidate uses a disallowed license status: {row['id']}"
                        )
                    emitted += 1
                    if candidates.readline() != source_line:
                        raise ValueError(
                            "training_candidates is not the byte-for-byte eligible-row derivation"
                        )
        if candidates.readline():
            raise ValueError("training_candidates has rows beyond the configured derivation")
    expected = config["expected_training_candidate_stats"]["rows"]
    if emitted != expected:
        raise ValueError(
            f"training candidate derivation emitted {emitted:,} rows, expected {expected:,}"
        )


def _view_matches(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    if "prompt_is_not_null" in filters:
        return bool(row["prompt"]) is bool(filters["prompt_is_not_null"])
    if "modality_in" in filters:
        return row["modality"] in filters["modality_in"]
    raise ValueError(f"unsupported training view filter: {filters}")


def _validate_training_views(release_dir: Path, config: Mapping[str, Any]) -> None:
    _compare_jsonl_derivation(
        release_dir / "training_views.jsonl",
        config["training_views"],
        label="training_views.jsonl",
    )
    counts = {view["id"]: 0 for view in config["training_views"]}
    for row in iter_jsonl(release_dir / "training_candidates.jsonl"):
        for view in config["training_views"]:
            counts[view["id"]] += int(_view_matches(row, view["filter"]))
    for view in config["training_views"]:
        if counts[view["id"]] != view["expected_rows"]:
            raise ValueError(f"training view count mismatch: {view['id']}")


def _validate_source_derivations(
    release_dir: Path,
    config: Mapping[str, Any],
    repository_root: Path,
) -> tuple[dict[str, int], int, int]:
    protected, protected_rows, protected_counts = load_protected_prompts(
        config["protected_prompt_sources"],
        repository_root,
        _physical_root_map(config, repository_root),
    )
    if protected_counts != {
        row["id"]: row["expected_prompts"] for row in config["protected_prompt_sources"]
    }:
        raise ValueError("protected prompt counts differ from the frozen config")
    unique_count = len(protected)
    if unique_count != config["expected_protected_normalized_unique_prompts"]:
        raise ValueError("protected normalized unique-prompt count changed")
    unique_image_count = unique_protected_image_count(protected_rows)
    if (
        config.get("expected_protected_unique_images") is not None
        and unique_image_count != config["expected_protected_unique_images"]
    ):
        raise ValueError("protected unique-image count changed")
    _compare_jsonl_derivation(
        release_dir / "benchmark_holdouts.jsonl",
        iter_holdout_records(protected_rows),
        label="benchmark_holdouts.jsonl",
    )
    context = SourceContext(
        repository_root=repository_root,
        protected_prompts=protected,
        allowed_training_license_statuses=frozenset(
            config["training_license_policy"]["allowed_statuses"]
        ),
        physical_roots=_physical_root_map(config, repository_root),
    )
    for dataset in config["datasets"]:
        _compare_jsonl_derivation(
            release_dir / dataset["output"],
            iter_dataset_records(dataset, context),
            label=dataset["output"],
        )
    _compare_training_candidate_derivation(release_dir, config)
    _validate_training_views(release_dir, config)
    _compare_jsonl_derivation(
        release_dir / "source_inventory.jsonl",
        _physical_source_inventory(config, repository_root),
        label="source_inventory.jsonl",
    )
    return protected_counts, unique_count, unique_image_count


def _artifact_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return the filesystem identity used to pin one artifact read."""
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _sha256_descriptor(descriptor: int) -> str:
    """Hash bytes from a pinned descriptor without changing its file offset."""
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 8 * 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _sha256_path_snapshot(path: Path) -> tuple[str, os.stat_result]:
    """Hash the bytes currently reachable through a fresh, pinned pathname."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"catalog artifact changed while it was read: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"catalog artifact changed while it was read: {path}")
        path_before = _artifact_path_stat(path)
        if _artifact_identity(path_before) != _artifact_identity(before):
            raise ValueError(f"catalog artifact changed while it was read: {path}")
        actual_sha256 = _sha256_descriptor(descriptor)
        after = os.fstat(descriptor)
        path_after = _artifact_path_stat(path)
        if (
            _artifact_identity(after) != _artifact_identity(before)
            or _artifact_identity(path_after) != _artifact_identity(before)
        ):
            raise ValueError(f"catalog artifact changed while it was read: {path}")
        return actual_sha256, after
    finally:
        os.close(descriptor)


def _artifact_path_stat(path: Path) -> os.stat_result:
    """Read the pathname identity without following a final symlink."""
    try:
        value = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"catalog artifact changed while it was read: {path}") from exc
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"catalog artifact changed while it was read: {path}")
    return value


def _open_artifact_snapshot(
    path: Path,
    expected: Mapping[str, Any],
    *,
    pinned_descriptor: int | None = None,
) -> tuple[int, os.stat_result, str]:
    """Open, hash, and pin one artifact before any row parsing begins."""
    # A selected-view parent may already have been opened through a directory
    # descriptor.  Duplicate that descriptor so this validator never follows a
    # pathname after the parent directory is replaced.  The unpinned path
    # branch remains for ordinary catalog validation and publication.
    if pinned_descriptor is not None:
        try:
            descriptor = os.dup(pinned_descriptor)
        except OSError as exc:
            raise FileNotFoundError(
                f"catalog artifact pinned descriptor is unavailable: {path}"
            ) from exc
    else:
        # Reject a non-regular pathname before ``open`` so a FIFO cannot block
        # the validator.  The descriptor and pathname identities are checked
        # again below because this preflight is inherently racy.
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"catalog artifact is missing or not a regular file: {path}"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise FileNotFoundError(
                f"catalog artifact is missing or not a regular file: {path}"
            ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FileNotFoundError(
                f"catalog artifact is missing or not a regular file: {path}"
            )
        if before.st_size != expected["bytes"]:
            raise ValueError(f"artifact byte count mismatch: {path}")
        current = _artifact_path_stat(path)
        if _artifact_identity(current) != _artifact_identity(before):
            raise ValueError(f"catalog artifact changed while it was read: {path}")
        actual_sha256 = _sha256_descriptor(descriptor)
        if actual_sha256 != expected["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
        after = os.fstat(descriptor)
        current = _artifact_path_stat(path)
        if (
            _artifact_identity(after) != _artifact_identity(before)
            or _artifact_identity(current) != _artifact_identity(before)
        ):
            raise ValueError(f"catalog artifact changed while it was read: {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, before, actual_sha256
    except BaseException:
        os.close(descriptor)
        raise


def _assert_artifact_snapshot_unchanged(
    path: Path,
    descriptor: int,
    snapshot: os.stat_result,
    expected: Mapping[str, Any],
) -> tuple[str, int]:
    """Hash the pinned bytes and reject replacement during or after parsing."""
    before_hash = os.fstat(descriptor)
    path_stat = _artifact_path_stat(path)
    if (
        _artifact_identity(before_hash) != _artifact_identity(snapshot)
        or _artifact_identity(path_stat) != _artifact_identity(snapshot)
    ):
        raise ValueError(f"catalog artifact changed while it was read: {path}")
    actual_sha256 = _sha256_descriptor(descriptor)
    if actual_sha256 != expected["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    after_hash = os.fstat(descriptor)
    after_path = _artifact_path_stat(path)
    if (
        _artifact_identity(after_hash) != _artifact_identity(snapshot)
        or _artifact_identity(after_path) != _artifact_identity(snapshot)
    ):
        raise ValueError(f"catalog artifact changed while it was read: {path}")
    path_sha256, _path_snapshot = _sha256_path_snapshot(path)
    if path_sha256 != actual_sha256 or path_sha256 != expected["sha256"]:
        raise ValueError(f"catalog artifact changed while it was read: {path}")
    return actual_sha256, after_hash.st_size


def _read_artifact_snapshot(
    path: Path,
    expected: Mapping[str, Any],
    *,
    pinned_descriptor: int | None = None,
) -> bytes:
    """Read one small artifact from its pinned descriptor and verify its bytes."""
    descriptor, snapshot, _initial_sha256 = _open_artifact_snapshot(
        path, expected, pinned_descriptor=pinned_descriptor
    )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        actual_sha256, actual_bytes = _assert_artifact_snapshot_unchanged(
            path, descriptor, snapshot, expected
        )
        if len(payload) != actual_bytes or hashlib.sha256(payload).hexdigest() != actual_sha256:
            raise ValueError(f"catalog artifact changed while it was read: {path}")
        return payload
    finally:
        os.close(descriptor)


def _iter_jsonl_descriptor(
    descriptor: int,
    path: Path,
    *,
    digest: hashlib._Hash | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield JSONL rows while optionally hashing the exact bytes parsed."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    duplicate = os.dup(descriptor)
    try:
        handle = os.fdopen(duplicate, "rb")
    except BaseException:
        # ``fdopen`` does not take ownership when construction fails.
        os.close(duplicate)
        raise
    with handle:
        for line_number, raw_line in enumerate(handle, 1):
            if digest is not None:
                digest.update(raw_line)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"invalid UTF-8 at {path}:{line_number}: {exc}"
                ) from exc
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


def _read_pinned_descriptor(
    path: Path,
    descriptor: int,
    *,
    expected_bytes: int | None = None,
) -> bytes:
    """Read bytes from a caller-owned descriptor without following ``path``.

    The descriptor is duplicated so the caller retains ownership and file
    position.  When a pathname is supplied, its identity is checked at both
    boundaries; a directory swap therefore either fails closed or leaves the
    read pinned to the original inode.
    """
    try:
        duplicate = os.dup(descriptor)
    except OSError as exc:
        raise ValueError(f"catalog artifact descriptor is unavailable: {path}") from exc
    try:
        before = os.fstat(duplicate)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"catalog artifact is not a regular file: {path}")
        path_before = _artifact_path_stat(path)
        if _artifact_identity(path_before) != _artifact_identity(before):
            raise ValueError(f"catalog artifact changed while it was read: {path}")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise ValueError(f"catalog artifact byte count mismatch: {path}")
        os.lseek(duplicate, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(duplicate, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(duplicate)
        path_after = _artifact_path_stat(path)
        if (
            _artifact_identity(after) != _artifact_identity(before)
            or _artifact_identity(path_after) != _artifact_identity(before)
        ):
            raise ValueError(f"catalog artifact changed while it was read: {path}")
        if expected_bytes is not None and len(payload) != expected_bytes:
            raise ValueError(f"catalog artifact byte count mismatch: {path}")
        return payload
    finally:
        os.close(duplicate)


def _validate_artifact_file(
    path: Path,
    expected: Mapping[str, Any],
    *,
    pinned_descriptor: int | None = None,
) -> str:
    """Validate an immutable artifact's file binding without parsing its rows."""
    descriptor, snapshot, initial_sha256 = _open_artifact_snapshot(
        path, expected, pinned_descriptor=pinned_descriptor
    )
    try:
        # Rehash the pinned descriptor after the initial check.  This is
        # intentional even though no rows are parsed here: NFS attribute
        # caching can leave size and timestamps unchanged after an in-place
        # rewrite, so metadata alone cannot close the read window.
        actual_sha256, actual_bytes = _assert_artifact_snapshot_unchanged(
            path, descriptor, snapshot, expected
        )
        if actual_sha256 != initial_sha256 or actual_bytes != expected["bytes"]:
            raise ValueError(f"catalog artifact changed while it was read: {path}")
        return actual_sha256
    finally:
        os.close(descriptor)


def _validate_artifact(
    path: Path,
    expected: Mapping[str, Any],
    *,
    verify_paths: bool,
    pinned_descriptor: int | None = None,
) -> dict[str, Any]:
    descriptor, snapshot, _initial_sha256 = _open_artifact_snapshot(
        path, expected, pinned_descriptor=pinned_descriptor
    )
    count = 0
    eligible = 0
    with_prompt = 0
    with_image = 0
    leakage_matches = 0
    missing_images = 0
    modalities: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    id_fingerprints: set[bytes] = set()
    path_check_batch: list[Mapping[str, Any]] = []
    parsed_digest = hashlib.sha256()
    executor: ThreadPoolExecutor | None = None
    try:
        executor = ThreadPoolExecutor(max_workers=32) if verify_paths else None
        try:
            for row in _iter_jsonl_descriptor(
                descriptor, path, digest=parsed_digest
            ):
                row_id = row.get("id")
                if not isinstance(row_id, str) or not row_id:
                    raise ValueError(f"artifact row lacks a non-empty id: {path}")
                fingerprint = hashlib.blake2b(row_id.encode("utf-8"), digest_size=16).digest()
                if fingerprint in id_fingerprints:
                    raise ValueError(f"duplicate record id in {path}: {row_id}")
                id_fingerprints.add(fingerprint)
                if expected.get("schema") == DATA_RECORD_SCHEMA:
                    validate_record(row)
                    eligible += int(row["training_eligible"])
                    with_prompt += int(bool(row["prompt"]))
                    with_image += int(bool(row["image_path"]))
                    leakage_matches += int(bool(row["benchmark_exact_match"]))
                    missing_images += int(row["source_record"].get("image_exists") is False)
                    modalities[row["modality"]] += 1
                    splits[row["split"]] += 1
                    if verify_paths and row["image_path"]:
                        path_check_batch.append(row)
                        if len(path_check_batch) == 4096:
                            verify_image_paths(path_check_batch, executor=executor)
                            path_check_batch.clear()
                elif row.get("schema") != expected.get("schema"):
                    raise ValueError(f"unexpected row schema in {path}")
                count += 1
                if count % 500_000 == 0:
                    print(f"[{path.name}] validated {count:,} rows", flush=True)
            if path_check_batch:
                verify_image_paths(path_check_batch, executor=executor)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        # The artifact may be replaced while rows are being parsed.  Revalidate
        # the pinned descriptor and pathname so the returned statistics and
        # hash describe the same published bytes.
        actual_sha256, actual_bytes = _assert_artifact_snapshot_unchanged(
            path, descriptor, snapshot, expected
        )
        if parsed_digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"catalog artifact changed while it was read: {path}")
    finally:
        os.close(descriptor)
    observed: dict[str, Any] = {
        "path": path.name,
        "schema": expected["schema"],
        "rows": count,
    }
    if expected.get("schema") == DATA_RECORD_SCHEMA:
        observed.update(
            {
                "training_eligible_rows": eligible,
                "rows_with_prompt": with_prompt,
                "rows_with_image": with_image,
                "benchmark_exact_match_rows": leakage_matches,
                "missing_images": missing_images,
                "modalities": dict(sorted(modalities.items())),
                "splits": dict(sorted(splits.items())),
            }
        )
    observed.update({"bytes": actual_bytes, "sha256": actual_sha256})
    if dict(expected) != observed:
        raise ValueError(f"manifest artifact metadata does not match observed content: {path}")
    return observed


def validate_release(
    release_dir: Path,
    *,
    verify_paths: bool,
    require_formal_catalog: bool = False,
    require_training_ready: bool = False,
    validate_records: bool = True,
    pinned_descriptors: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if type(validate_records) is not bool:
        raise TypeError("validate_records must be a boolean")
    if not validate_records and (
        not require_formal_catalog or verify_paths is not True
    ):
        raise ValueError(
            "record-skipping validation requires a path-verified formal catalog"
        )
    if require_training_ready and not require_formal_catalog:
        raise ValueError(
            "training-ready validation requires a formal catalog from a clean pushed commit"
        )
    manifest_path = release_dir / "manifest.json"
    pinned = dict(pinned_descriptors or {})
    if pinned_descriptors is not None and "manifest.json" not in pinned:
        raise ValueError("pinned catalog validation requires a manifest descriptor")
    manifest_descriptor = pinned.get("manifest.json")
    if manifest_descriptor is None:
        manifest_bytes = manifest_path.read_bytes()
    else:
        manifest_bytes = _read_pinned_descriptor(manifest_path, manifest_descriptor)
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"release manifest is not readable JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"release manifest must contain one JSON object: {manifest_path}")
    if manifest.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"unexpected release schema in {manifest_path}")
    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or release_dir.name != release_id:
        raise ValueError("release directory does not match manifest release_id")
    manifest_core = {
        key: value for key, value in manifest.items() if key not in {"schema", "release_id"}
    }
    if _release_id(manifest_core) != release_id:
        raise ValueError("release_id does not match the manifest and artifact hashes")
    if manifest.get("record_schema") != DATA_RECORD_SCHEMA:
        raise ValueError("release has an unexpected record schema")

    snapshot = manifest.get("config_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("path") != CONFIG_SNAPSHOT_NAME:
        raise ValueError("release lacks the required catalog config snapshot")
    if set(snapshot) != {"path", "schema", "bytes", "sha256"} or snapshot.get(
        "schema"
    ) != CONFIG_SNAPSHOT_SCHEMA:
        raise ValueError("catalog config snapshot metadata is invalid")
    snapshot_path = release_dir / CONFIG_SNAPSHOT_NAME
    if pinned_descriptors is not None and CONFIG_SNAPSHOT_NAME not in pinned:
        raise ValueError("pinned catalog validation requires a config descriptor")
    snapshot_bytes = _read_artifact_snapshot(
        snapshot_path,
        snapshot,
        pinned_descriptor=pinned.get(CONFIG_SNAPSHOT_NAME),
    )
    config = _load_config_bytes(snapshot_bytes, source=str(snapshot_path))
    if manifest.get("physical_source_count") != len(config["physical_sources"]):
        raise ValueError("release does not inventory exactly the configured five sources")
    if manifest.get("payload_integrity_policy") != config["payload_integrity_policy"]:
        raise ValueError("release payload-integrity policy differs from the frozen config")
    if manifest.get("training_ready") is not config["payload_integrity_policy"][
        "training_ready"
    ]:
        raise ValueError("release training-ready state differs from the frozen config")
    if manifest.get("complete") is not (
        manifest.get("candidate_catalog_complete") is True
        and manifest.get("training_ready") is True
    ):
        raise ValueError("release complete flag is inconsistent with payload integrity")

    git = manifest.get("git")
    if not isinstance(git, dict):
        raise ValueError("release lacks Git provenance")
    if require_formal_catalog:
        if (
            manifest.get("candidate_catalog_complete") is not True
            or manifest.get("verify_paths") is not True
            or verify_paths is not True
            or manifest.get("development_build") is not False
            or git.get("dirty") is not False
            or git.get("pushed") is not True
        ):
            raise ValueError(
                "formal validation requires a complete candidate catalog from a clean, "
                "path-verified, pushed commit"
            )
        if not config["expected_artifact_hashes"]:
            raise ValueError("formal catalog validation requires frozen artifact hashes")
        _validate_formal_git_record(git)
        config_repo_path = manifest.get("config_repo_path")
        if not isinstance(config_repo_path, str) or not config_repo_path:
            raise ValueError("formal release config is not bound to a repository path")
        committed_config = _run_git_bytes(
            ("show", f"{git['commit']}:{config_repo_path}"), REPOSITORY_ROOT
        )
        if committed_config != snapshot_bytes:
            raise ValueError("catalog config snapshot differs from the recorded Git commit")
        _validate_recorded_upstream_ancestry(
            git.get("commit"),
            git.get("upstream_ref"),
            repository_root=REPOSITORY_ROOT,
        )
        _validate_builder_provenance(
            manifest.get("builder_provenance"),
            repository_root=REPOSITORY_ROOT,
            commit=git["commit"],
        )
    elif manifest.get("builder_provenance") != _builder_provenance(REPOSITORY_ROOT):
        raise ValueError("development catalog builder files changed after publication")

    if require_training_ready and (
        manifest.get("complete") is not True or manifest.get("training_ready") is not True
    ):
        raise ValueError(
            "catalog is a metadata-only candidate inventory; build a complete selected-payload "
            "SHA-256 manifest before training"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("release has no artifact inventory")
    artifact_names = [artifact.get("path") for artifact in artifacts]
    if any(
        not isinstance(name, str) or Path(name).name != name for name in artifact_names
    ) or len(artifact_names) != len(set(artifact_names)):
        raise ValueError("artifact paths must be unique basenames")
    if artifact_names != config["expected_artifact_order"]:
        raise ValueError("release artifact inventory differs from the frozen config")
    if pinned_descriptors is not None:
        expected_pinned = {"manifest.json", CONFIG_SNAPSHOT_NAME, *artifact_names}
        if set(pinned) != expected_pinned:
            missing = sorted(expected_pinned - set(pinned))
            extra = sorted(set(pinned) - expected_pinned)
            raise ValueError(
                "pinned catalog descriptor set is incomplete"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else "")
            )
    for artifact in artifacts:
        path = release_dir / artifact["path"]
        pinned_descriptor = pinned.get(str(artifact["path"]))
        if validate_records:
            _validate_artifact(
                path,
                artifact,
                verify_paths=verify_paths,
                pinned_descriptor=pinned_descriptor,
            )
        else:
            _validate_artifact_file(
                path,
                artifact,
                pinned_descriptor=pinned_descriptor,
            )
    _assert_artifact_contract(artifacts, config)
    training = artifacts[artifact_names.index("training_candidates.jsonl")]
    if (
        training.get("training_eligible_rows") != training.get("rows")
        or training.get("benchmark_exact_match_rows") != 0
    ):
        raise ValueError("training_candidates contains an ineligible or protected row")

    if not validate_records:
        expected_protected_counts = {
            row["id"]: row["expected_prompts"]
            for row in config["protected_prompt_sources"]
        }
        if manifest.get("protected_prompt_source_counts") != expected_protected_counts:
            raise ValueError("manifest protected-source counts differ from the frozen config")
        if manifest.get("protected_normalized_unique_prompts") != config[
            "expected_protected_normalized_unique_prompts"
        ]:
            raise ValueError("manifest protected unique-prompt count differs from the frozen config")
        if (
            config.get("expected_protected_unique_images") is not None
            and manifest.get("protected_unique_images")
            != config["expected_protected_unique_images"]
        ):
            raise ValueError("manifest protected unique-image count differs from the frozen config")
        cross_checks = manifest.get("cross_checks")
        if (
            not isinstance(cross_checks, Mapping)
            or cross_checks.get("sana_cache_matches_four_k_lsdb") is not True
        ):
            raise ValueError("formal catalog cross-checks are incomplete")
        return manifest

    current_source_provenance = _source_provenance(config, REPOSITORY_ROOT)
    if manifest.get("source_provenance") != current_source_provenance:
        raise ValueError("source metadata changed after catalog publication")
    protected_counts, unique_count, unique_image_count = _validate_source_derivations(
        release_dir, config, REPOSITORY_ROOT
    )
    if manifest.get("protected_prompt_source_counts") != protected_counts:
        raise ValueError("manifest protected-source counts differ from the frozen derivation")
    if manifest.get("protected_normalized_unique_prompts") != unique_count:
        raise ValueError("manifest protected unique-prompt count differs from the derivation")
    if (
        config.get("expected_protected_unique_images") is not None
        and manifest.get("protected_unique_images") != unique_image_count
    ):
        raise ValueError("manifest protected unique-image count differs from the derivation")
    if (
        "protected_unique_images" in manifest
        and manifest["protected_unique_images"] != unique_image_count
    ):
        raise ValueError("manifest recorded an incorrect protected unique-image count")

    four_k_manifest = resolve_path(
        config["cross_checks"]["four_k_lsdb_manifest"], REPOSITORY_ROOT
    )
    cache_meta_path = resolve_path(
        config["cross_checks"]["sana_cache_meta"], REPOSITORY_ROOT
    )
    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    ids_sha256 = four_k_lsdb_ids_sha256(four_k_manifest)
    expected_cross_checks = {
        "four_k_lsdb_ids_sha256": ids_sha256,
        "sana_cache_ids_sha256": cache_meta["ids_sha256"],
        "sana_cache_matches_four_k_lsdb": ids_sha256 == cache_meta["ids_sha256"],
    }
    if not expected_cross_checks["sana_cache_matches_four_k_lsdb"]:
        raise ValueError("Sana text cache does not match the 4KLSDB manifest order")
    if manifest.get("cross_checks") != expected_cross_checks:
        raise ValueError("manifest cross-checks differ from current source derivation")
    return manifest


def validate_release_artifact_closure(
    release_dir: Path,
    *,
    require_training_ready: bool = False,
    pinned_descriptors: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate a published catalog's immutable closure for downstream consumers.

    Catalog publication performs the expensive row-by-row derivation and path
    checks.  Consumers still verify the formal Git provenance, manifest
    identity, artifact statistics, and every artifact hash, then validate the
    rows they actually consume.  This avoids repeating millions of unrelated
    image-path checks in every parallel selected-view worker.
    """
    return validate_release(
        release_dir,
        verify_paths=True,
        require_formal_catalog=True,
        require_training_ready=require_training_ready,
        validate_records=False,
        pinned_descriptors=pinned_descriptors,
    )


def _assert_build_inputs_unchanged(
    *,
    initial_git: Mapping[str, Any],
    initial_source_provenance: Sequence[Mapping[str, Any]],
    initial_builder_provenance: Sequence[Mapping[str, Any]],
    config_path: Path,
    config_sha256: str,
    config_repo_path: str | None,
    formal: bool,
) -> None:
    if git_state(REPOSITORY_ROOT) != dict(initial_git):
        raise RuntimeError("repository state changed during catalog construction")
    config = load_config(config_path, require_frozen_contract=formal)
    if sha256_file(config_path) != config_sha256:
        raise RuntimeError("catalog config changed during construction")
    if _source_provenance(config, REPOSITORY_ROOT) != list(initial_source_provenance):
        raise RuntimeError("source metadata changed during catalog construction")
    if _builder_provenance(REPOSITORY_ROOT) != list(initial_builder_provenance):
        raise RuntimeError("catalog builder changed during construction")
    observed_repo_path = _tracked_config_path(
        config_path,
        REPOSITORY_ROOT,
        commit=str(initial_git["commit"]),
        required=formal,
    )
    if observed_repo_path != config_repo_path:
        raise RuntimeError("catalog config Git binding changed during construction")


def build_catalog(
    *,
    config_path: Path,
    output_dir: Path | None,
    verify_paths: bool,
    allow_dirty: bool,
    require_training_ready: bool = False,
) -> Path:
    repository_root = REPOSITORY_ROOT
    if not verify_paths and not allow_dirty:
        raise RuntimeError("--no-verify-paths requires --allow-dirty")
    if require_training_ready and allow_dirty:
        raise RuntimeError(
            "training-ready catalogs cannot be built from a dirty or unpushed worktree"
        )
    config = load_config(config_path, require_frozen_contract=not allow_dirty)
    if require_training_ready and config["payload_integrity_policy"].get(
        "training_ready"
    ) is not True:
        raise ValueError(
            "catalog is a metadata-only candidate inventory; build a complete "
            "selected-payload SHA-256 manifest before training"
        )
    git = enforce_git_gate(repository_root, allow_dirty=allow_dirty)
    config_repo_path = _tracked_config_path(
        config_path,
        repository_root,
        commit=git["commit"],
        required=not allow_dirty,
    )
    config_sha256 = sha256_file(config_path)
    destination = output_dir or _resolve_output_dir(config["output_dir"], repository_root)
    destination.mkdir(parents=True, exist_ok=True)
    catalogs_dir = destination / "catalogs"
    catalogs_dir.mkdir(exist_ok=True)

    print("hashing source metadata", flush=True)
    source_provenance = _source_provenance(config, repository_root)
    builder_provenance = _builder_provenance(repository_root)
    staging = Path(tempfile.mkdtemp(prefix=".catalog-build-", dir=catalogs_dir))
    artifacts: list[dict[str, Any]] = []
    try:
        protected, protected_rows, protected_counts = load_protected_prompts(
            config["protected_prompt_sources"],
            repository_root,
            _physical_root_map(config, repository_root),
        )
        if protected_counts != {
            row["id"]: row["expected_prompts"]
            for row in config["protected_prompt_sources"]
        }:
            raise ValueError("protected prompt source counts changed")
        if len(protected) != config["expected_protected_normalized_unique_prompts"]:
            raise ValueError("protected normalized unique-prompt count changed")
        protected_unique_images = unique_protected_image_count(protected_rows)
        if (
            config.get("expected_protected_unique_images") is not None
            and protected_unique_images != config["expected_protected_unique_images"]
        ):
            raise ValueError("protected unique-image count changed")
        context = SourceContext(
            repository_root=repository_root,
            protected_prompts=protected,
            allowed_training_license_statuses=frozenset(
                config["training_license_policy"]["allowed_statuses"]
            ),
            physical_roots=_physical_root_map(config, repository_root),
        )
        holdout_stats = write_records(
            staging / "benchmark_holdouts.jsonl",
            iter_holdout_records(protected_rows),
            verify_paths=verify_paths,
            expected_rows=config["expected_holdout_stats"]["rows"],
        )
        _assert_stats_match(
            holdout_stats,
            {
                "path": "benchmark_holdouts.jsonl",
                "schema": DATA_RECORD_SCHEMA,
                **config["expected_holdout_stats"],
            },
            context="benchmark_holdouts.jsonl",
        )
        artifacts.append(holdout_stats)
        print(
            f"[benchmark_holdouts.jsonl] complete: {holdout_stats['rows']:,} rows",
            flush=True,
        )

        training_outputs: list[Path] = []
        for dataset in config["datasets"]:
            output_path = staging / dataset["output"]
            print(f"[{output_path.name}] building", flush=True)
            stats = write_records(
                output_path,
                iter_dataset_records(dataset, context),
                verify_paths=verify_paths,
                expected_rows=int(dataset["expected_rows"]),
            )
            _assert_stats_match(
                stats,
                {
                    "path": dataset["output"],
                    "schema": DATA_RECORD_SCHEMA,
                    "rows": dataset["expected_rows"],
                    **dataset["expected_stats"],
                },
                context=output_path.name,
            )
            artifacts.append(stats)
            print(f"[{output_path.name}] complete: {stats['rows']:,} rows", flush=True)
            if dataset.get("include_in_training_candidates", False):
                training_outputs.append(output_path)

        def training_records() -> Iterable[Mapping[str, Any]]:
            for path in training_outputs:
                for row in iter_jsonl(path):
                    if row["training_eligible"]:
                        yield row

        training_stats = write_records(
            staging / "training_candidates.jsonl",
            training_records(),
            verify_paths=False,
            expected_rows=config["expected_training_candidate_stats"]["rows"],
        )
        _assert_stats_match(
            training_stats,
            {
                "path": "training_candidates.jsonl",
                "schema": DATA_RECORD_SCHEMA,
                **config["expected_training_candidate_stats"],
            },
            context="training_candidates.jsonl",
        )
        artifacts.append(training_stats)
        print(
            f"[training_candidates.jsonl] complete: {training_stats['rows']:,} rows",
            flush=True,
        )

        artifacts.append(
            _generic_jsonl(
                staging / "training_views.jsonl",
                config["training_views"],
                schema=TRAINING_VIEW_SCHEMA,
            )
        )

        inventory_stats = _generic_jsonl(
            staging / "source_inventory.jsonl",
            _physical_source_inventory(config, repository_root),
            schema=PHYSICAL_SOURCE_SCHEMA,
        )
        artifacts.append(inventory_stats)
        _assert_artifact_contract(artifacts, config)

        config_snapshot = _write_config_snapshot(
            staging / CONFIG_SNAPSHOT_NAME, config_path
        )

        four_k_manifest = resolve_path(config["cross_checks"]["four_k_lsdb_manifest"], repository_root)
        cache_meta_path = resolve_path(config["cross_checks"]["sana_cache_meta"], repository_root)
        cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        computed_ids_sha256 = four_k_lsdb_ids_sha256(four_k_manifest)
        if computed_ids_sha256 != cache_meta["ids_sha256"]:
            raise ValueError("Sana text cache does not match the 4KLSDB manifest order")

        candidate_catalog_complete = (
            not allow_dirty and verify_paths and bool(config["expected_artifact_hashes"])
        )
        training_ready = bool(config["payload_integrity_policy"]["training_ready"])
        manifest_core = {
            "complete": candidate_catalog_complete and training_ready,
            "candidate_catalog_complete": candidate_catalog_complete,
            "development_build": allow_dirty,
            "record_schema": DATA_RECORD_SCHEMA,
            "config_repo_path": config_repo_path,
            "config_snapshot": config_snapshot,
            "git": git,
            "verify_paths": verify_paths,
            "physical_source_count": len(config["physical_sources"]),
            "training_ready": training_ready,
            "payload_integrity_policy": config["payload_integrity_policy"],
            "protected_prompt_source_counts": protected_counts,
            "protected_normalized_unique_prompts": len(protected),
            "protected_unique_images": protected_unique_images,
            "source_provenance": source_provenance,
            "builder_provenance": builder_provenance,
            "cross_checks": {
                "four_k_lsdb_ids_sha256": computed_ids_sha256,
                "sana_cache_ids_sha256": cache_meta["ids_sha256"],
                "sana_cache_matches_four_k_lsdb": True,
            },
            "artifacts": artifacts,
        }
        release_id = _release_id(manifest_core)
        manifest = {"schema": CATALOG_SCHEMA, "release_id": release_id, **manifest_core}
        manifest_path = staging / "manifest.json"
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        _assert_build_inputs_unchanged(
            initial_git=git,
            initial_source_provenance=source_provenance,
            initial_builder_provenance=builder_provenance,
            config_path=config_path,
            config_sha256=config_sha256,
            config_repo_path=config_repo_path,
            formal=not allow_dirty,
        )
        release_dir = catalogs_dir / release_id
        _install_catalog_release(
            staging,
            release_dir,
            catalogs_dir,
            manifest,
            verify_paths=verify_paths,
            require_formal_catalog=candidate_catalog_complete,
            require_training_ready=require_training_ready,
        )
        if candidate_catalog_complete:
            _publish_current(destination, release_dir)
        return release_dir
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-verify-paths",
        action="store_true",
        help="development only: skip image existence checks",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development only: allow an uncommitted or unpushed code state",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate DATA/current instead of building",
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        help="release to validate; defaults to the current catalog",
    )
    parser.add_argument(
        "--require-training-ready",
        action="store_true",
        help="reject metadata-only candidate catalogs that lack payload checksums",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _resolve_output_dir(config["output_dir"], REPOSITORY_ROOT)
    else:
        output_dir = output_dir.resolve()
    verify_paths = not args.no_verify_paths
    if not verify_paths and not args.allow_dirty:
        raise RuntimeError("--no-verify-paths requires --allow-dirty")
    if (
        args.require_training_ready
        and args.allow_dirty
    ):
        raise RuntimeError(
            "--require-training-ready cannot be combined with --allow-dirty"
        )
    if (
        args.require_training_ready
        and not args.validate_only
        and config["payload_integrity_policy"].get("training_ready") is not True
    ):
        raise ValueError(
            "catalog is a metadata-only candidate inventory; build a complete "
            "selected-payload SHA-256 manifest before training"
        )
    if args.validate_only:
        release_dir = (
            args.release_dir.resolve()
            if args.release_dir is not None
            else (output_dir / "current").resolve()
        )
        manifest = validate_release(
            release_dir,
            verify_paths=verify_paths,
            require_formal_catalog=not args.allow_dirty,
            require_training_ready=args.require_training_ready,
        )
        print(json.dumps({"validated": str(release_dir), "release_id": manifest["release_id"]}))
        return 0
    release_dir = build_catalog(
        config_path=config_path,
        output_dir=output_dir,
        verify_paths=verify_paths,
        allow_dirty=args.allow_dirty,
        require_training_ready=args.require_training_ready,
    )
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "release_dir": str(release_dir),
                "release_id": manifest["release_id"],
                "complete": manifest["complete"],
                "candidate_catalog_complete": manifest["candidate_catalog_complete"],
                "training_ready": manifest["training_ready"],
                "artifacts": {row["path"]: row["rows"] for row in manifest["artifacts"]},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0
