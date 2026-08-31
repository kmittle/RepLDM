"""Transactional scorer for the frozen HPSv2 relational-renderer experiment.

Reads a run's manifest, runs the configured Scorers (self-contained metric modules
under scorers/, each decoupled from Sana), and writes scores.jsonl. Each metric is
declared in a yaml config (configs/eval_common.yaml). Weights are validated up front
so a missing metric is SKIPPED with a warning instead of crashing the run.

Resume is additive when enabled metrics and their execution provenance still match.
Strict runs bind source, package/runtime, model asset, and preprocessing metadata;
any drift invalidates the row before old columns can be reused. Private progress and
its hash-bound receipt survive ordinary interruption; canonical scores and their
receipt are published only after complete scoring and a clean watchdog exit.

  /home/bycao/miniforge3/envs/repldm_eval/bin/python \
      eval-pipeline/score_hpsv2_relational_renderer.py \
      --run_dir outputs/hpsv2_relational_renderer/full_v1 \
      --config eval-pipeline/configs/hpsv2_full_scoring_v1.yaml \
      --device cuda:2 --strict --require-scorer-provenance \
      --require-exclusive-gpu
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import urllib.request
import uuid

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

import yaml  # noqa: E402
import scorers  # noqa: E402,F401  (import registers all metrics)
import scorers.base as scorer_base  # noqa: E402
from scorers.base import REGISTRY  # noqa: E402
from scorer_provenance import (  # noqa: E402
    SCORER_PROVENANCE_SCHEMA,
    build_scorer_provenance,
    registered_scorer_provenance_contract,
    validate_hardened_score_rows,
)
from s7_provenance import (  # noqa: E402
    PROVENANCE_SCHEMA,
    json_sha256,
    validate_design_rows,
    validate_run_contract,
    validate_scores_against_manifest,
    validate_sidecar,
)


EXCLUSIVE_CUDA_EXECUTION_SCHEMA = "repldm_exclusive_cuda_execution_v1"
EXCLUSIVE_CUDA_POLL_SECONDS = 2.0
SCORING_SUCCESS_SCHEMA = "repldm_scoring_success_v1"
SCORING_SUCCESS_NAME = "scoring_success.json"
SCORING_PROGRESS_SCHEMA = "repldm_scoring_progress_v1"
SCORING_PROGRESS_RECEIPT_NAME = ".scoring_progress.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SCORING_PROGRESS_PAYLOAD_RE = re.compile(
    r"^\.scoring_progress\.([0-9a-f]{32})\.([0-9a-f]{64})\.jsonl$"
)
SCORER_ASSET_STAGE_PREFIX = "repldm-scorer-assets-"
SCORER_ASSET_LOADING_MODE = "pinned_private_stage_v1"


class ExclusiveCudaWatchdogError(RuntimeError):
    """The formal scoring GPU could not be proven exclusive."""


class ScorerAssetStageError(RuntimeError):
    """A copied scorer asset changed or could not be handled safely."""


def _file_identity(value):
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _safe_stage_relative_path(value):
    path = Path(str(value))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError("scorer asset staged name is unsafe")
    return path


def _copy_scorer_asset(source, destination):
    resolved = Path(source).expanduser().resolve(strict=True)
    source_descriptor = os.open(
        resolved,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    destination_descriptor = -1
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("scorer asset source is not a regular file")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        if _file_identity(before) != _file_identity(after) or copied != before.st_size:
            raise ScorerAssetStageError(
                "scorer asset changed while it was staged"
            )
        os.fchmod(destination_descriptor, 0o400)
        destination_identity = _file_identity(os.fstat(destination_descriptor))
        return {
            "source_path": str(resolved),
            "source_identity": _file_identity(before),
            "destination_identity": destination_identity,
            "size_bytes": copied,
            "sha256": digest.hexdigest(),
        }
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        os.close(source_descriptor)


def _sha256_descriptor(descriptor):
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 8 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _scan_stage_directory(directory_descriptor, prefix=Path()):
    files = {}
    directories = {}
    for name in sorted(os.listdir(directory_descriptor)):
        status = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        relative = prefix / name
        relative_name = relative.as_posix()
        if stat.S_ISREG(status.st_mode):
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(descriptor)
                if _file_identity(opened) != _file_identity(status):
                    raise ScorerAssetStageError(
                        "scorer asset changed while it was opened for verification"
                    )
                files[relative_name] = {
                    "identity": _file_identity(opened),
                    "size_bytes": int(opened.st_size),
                    "sha256": _sha256_descriptor(descriptor),
                }
            finally:
                os.close(descriptor)
        elif stat.S_ISDIR(status.st_mode):
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(descriptor)
                if _file_identity(opened) != _file_identity(status):
                    raise ScorerAssetStageError(
                        "scorer asset directory changed during verification"
                    )
                directories[relative_name] = _file_identity(opened)
                child_files, child_directories = _scan_stage_directory(
                    descriptor, relative
                )
                files.update(child_files)
                directories.update(child_directories)
            finally:
                os.close(descriptor)
        else:
            raise ScorerAssetStageError(
                "scorer asset stage contains a non-regular entry"
            )
    return files, directories


def _clear_stage_directory(directory_descriptor):
    os.fchmod(directory_descriptor, 0o700)
    for name in sorted(os.listdir(directory_descriptor)):
        status = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(status.st_mode):
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            try:
                _clear_stage_directory(descriptor)
            finally:
                os.close(descriptor)
            os.rmdir(name, dir_fd=directory_descriptor)
        else:
            os.unlink(name, dir_fd=directory_descriptor)


class ScorerAssetStage:
    """Private copied scorer files loaded through one pinned directory FD."""

    def __init__(self, metric_names, params):
        self.parent = None
        self.path = None
        self.proc_root = None
        self.parent_descriptor = -1
        self.root_descriptor = -1
        self.root_identity = None
        self.rows = []
        self.asset_paths = {}
        self.asset_revisions = {}
        self.asset_manifests = {}
        self.file_identities = {}
        self.directory_identities = {}
        self.closed = False
        try:
            raw_parent = Path(tempfile.gettempdir())
            if raw_parent.is_symlink():
                raise ScorerAssetStageError(
                    "scorer asset staging parent must not be a symlink"
                )
            parent = raw_parent.resolve(strict=True)
            parent_stat = parent.stat()
            if parent_stat.st_mode & 0o022 and not (
                parent_stat.st_mode & stat.S_ISVTX
            ):
                raise ScorerAssetStageError(
                    "scorer asset staging parent is unsafe"
                )
            self.parent = parent
            self.parent_descriptor = os.open(
                parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            if _file_identity(os.fstat(self.parent_descriptor)) != _file_identity(
                parent_stat
            ):
                raise ScorerAssetStageError(
                    "scorer asset staging parent changed while it was opened"
                )
            self.path = Path(
                tempfile.mkdtemp(prefix=SCORER_ASSET_STAGE_PREFIX, dir=parent)
            )
            self.root_descriptor = os.open(
                self.path,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            self.proc_root = Path("/proc/self/fd") / str(self.root_descriptor)
            staged_names = set()
            for metric_name in metric_names:
                metric_path = _safe_stage_relative_path(metric_name)
                if len(metric_path.parts) != 1:
                    raise ValueError("scorer metric name is unsafe for staging")
                cls = REGISTRY.get(metric_name)
                builder = getattr(cls, "asset_sources", None)
                specs = builder(**params) if callable(builder) else {}
                if not isinstance(specs, dict):
                    raise ScorerAssetStageError(
                        "scorer asset source contract is not a mapping"
                    )
                self.asset_paths[metric_name] = {}
                self.asset_revisions[metric_name] = {}
                for key, raw_spec in sorted(specs.items()):
                    if not isinstance(key, str) or not key or not isinstance(raw_spec, dict):
                        raise ScorerAssetStageError(
                            "scorer asset source entry is invalid"
                        )
                    if set(raw_spec) != {"path", "staged_name", "revision"}:
                        raise ScorerAssetStageError(
                            "scorer asset source fields differ"
                        )
                    relative = metric_path / _safe_stage_relative_path(
                        raw_spec["staged_name"]
                    )
                    relative_name = relative.as_posix()
                    if relative_name in staged_names:
                        raise ScorerAssetStageError(
                            "scorer asset staged names collide"
                        )
                    staged_names.add(relative_name)
                    destination = self.path / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    revision = raw_spec.get("revision")
                    if revision is not None and (
                        not isinstance(revision, str) or not revision
                    ):
                        raise ScorerAssetStageError(
                            "scorer asset revision is invalid"
                        )
                    copy_record = _copy_scorer_asset(
                        raw_spec["path"], destination
                    )
                    self.rows.append(
                        {
                            "metric": metric_name,
                            "key": key,
                            "path": relative_name,
                            "revision": revision,
                            "size_bytes": copy_record["size_bytes"],
                            "sha256": copy_record["sha256"],
                        }
                    )
                    self.file_identities[relative_name] = copy_record[
                        "destination_identity"
                    ]
                    self.asset_revisions[metric_name][key] = revision
            for directory, subdirectories, _ in os.walk(
                self.path, topdown=False, followlinks=False
            ):
                for name in subdirectories:
                    child = Path(directory) / name
                    os.chmod(child, 0o500)
                    descriptor = os.open(
                        child,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                    )
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            os.fchmod(self.root_descriptor, 0o500)
            os.fsync(self.root_descriptor)
            self.root_identity = _file_identity(os.fstat(self.root_descriptor))
            if not self.proc_root.is_dir():
                raise ScorerAssetStageError(
                    "pinned scorer asset loading requires procfs"
                )
            _, self.directory_identities = _scan_stage_directory(
                self.root_descriptor
            )
            for row in self.rows:
                self.asset_paths[row["metric"]][row["key"]] = str(
                    self.proc_root / row["path"]
                )
            for metric_name in metric_names:
                files = [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "metric"
                    }
                    for row in self.rows
                    if row["metric"] == metric_name
                ]
                self.asset_manifests[metric_name] = {
                    "schema": "repldm_scorer_asset_stage_v1",
                    "loading_mode": SCORER_ASSET_LOADING_MODE,
                    "files": files,
                    "files_sha256": json_sha256(files),
                }
            self.verify()
        except BaseException:
            self.cleanup(verify=False)
            raise

    def verify(self):
        if self.closed:
            raise ScorerAssetStageError(
                "scorer asset stage is already closed"
            )
        if _file_identity(os.fstat(self.root_descriptor)) != self.root_identity:
            raise ScorerAssetStageError(
                "scorer asset stage root identity changed"
            )
        try:
            public = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ScorerAssetStageError(
                "scorer asset stage path disappeared"
            ) from exc
        if _file_identity(public) != self.root_identity:
            raise ScorerAssetStageError(
                "scorer asset stage path identity changed"
            )
        observed, observed_directories = _scan_stage_directory(
            self.root_descriptor
        )
        if observed_directories != self.directory_identities:
            raise ScorerAssetStageError(
                "scorer asset stage directory identity changed"
            )
        observed_files = sorted(observed)
        expected_files = sorted(row["path"] for row in self.rows)
        if observed_files != expected_files:
            raise ScorerAssetStageError(
                "scorer asset stage inventory changed"
            )
        for row in self.rows:
            record = observed[row["path"]]
            if record["identity"] != self.file_identities[row["path"]]:
                raise ScorerAssetStageError(
                    "scorer asset stage file identity changed"
                )
            if record["size_bytes"] != row["size_bytes"]:
                raise ScorerAssetStageError(
                    "scorer asset stage size changed"
                )
            if record["sha256"] != row["sha256"]:
                raise ScorerAssetStageError(
                    "scorer asset stage hash changed"
                )
        return {
            "schema": "repldm_scorer_asset_stage_v1",
            "status": "verified_unchanged",
            "files": list(self.rows),
            "files_sha256": json_sha256(self.rows),
        }

    def cleanup(self, *, verify=True):
        if self.closed:
            return
        cleanup_error = None
        if verify:
            try:
                self.verify()
            except BaseException as exc:
                cleanup_error = exc
        proc_prefix = str(self.proc_root) if self.proc_root is not None else None
        public_prefix = str(self.path) if self.path is not None else None

        def belongs_to_stage(value):
            if not value or public_prefix is None:
                return False
            raw = str(value)
            if proc_prefix is not None and (
                raw == proc_prefix or raw.startswith(proc_prefix + os.sep)
            ):
                return True
            try:
                resolved = str(Path(raw).resolve())
            except OSError:
                return False
            return resolved == public_prefix or resolved.startswith(
                public_prefix + os.sep
            )

        for name, module in list(sys.modules.items()):
            if belongs_to_stage(getattr(module, "__file__", None)):
                del sys.modules[name]
        sys.path[:] = [value for value in sys.path if not belongs_to_stage(value)]
        try:
            if self.root_descriptor >= 0:
                _clear_stage_directory(self.root_descriptor)
            if self.path is not None and self.parent_descriptor >= 0:
                try:
                    public = os.stat(
                        self.path.name,
                        dir_fd=self.parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    public = None
                if public is not None:
                    current_root = (
                        os.fstat(self.root_descriptor)
                        if self.root_descriptor >= 0
                        else None
                    )
                    if current_root is None or _file_identity(
                        public
                    ) != _file_identity(current_root):
                        if cleanup_error is None:
                            cleanup_error = ScorerAssetStageError(
                                "scorer asset stage path identity changed"
                            )
                    else:
                        os.rmdir(
                            self.path.name,
                            dir_fd=self.parent_descriptor,
                        )
                        os.fsync(self.parent_descriptor)
        finally:
            if self.root_descriptor >= 0:
                os.close(self.root_descriptor)
                self.root_descriptor = -1
            if self.parent_descriptor >= 0:
                os.close(self.parent_descriptor)
                self.parent_descriptor = -1
            self.closed = True
        if cleanup_error is not None:
            raise cleanup_error


@contextmanager
def staged_scorer_assets(metric_names, params, *, enabled):
    if not enabled:
        yield None
        return
    stage = ScorerAssetStage(metric_names, params)
    try:
        yield stage
    except BaseException as primary:
        try:
            stage.cleanup()
        except BaseException as cleanup_error:
            primary.add_note(f"scorer asset cleanup also failed: {cleanup_error}")
        raise
    else:
        stage.cleanup()


@contextmanager
def formal_offline_network_guard(*, enabled):
    if not enabled:
        yield
        return
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) != "1":
        raise RuntimeError("registered scoring requires offline environment flags")
    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_urlopen = urllib.request.urlopen

    def blocked_network(*_args, **_kwargs):
        raise RuntimeError("network access is disabled for registered scoring")

    class OfflineSocket(original_socket):
        def connect(self, *_args, **_kwargs):
            return blocked_network()

        def connect_ex(self, *_args, **_kwargs):
            return blocked_network()

    socket.socket = OfflineSocket
    socket.create_connection = blocked_network
    urllib.request.urlopen = blocked_network
    try:
        yield
    finally:
        urllib.request.urlopen = original_urlopen
        socket.create_connection = original_create_connection
        socket.socket = original_socket


def exclusive_cuda_execution_contract(device):
    """Describe the fail-closed process monitor used by one scoring run."""
    normalized_device = str(device)
    if re.fullmatch(r"cuda:\d+", normalized_device) is None:
        raise ValueError("exclusive CUDA execution requires an explicit cuda:N device")
    return {
        "schema": EXCLUSIVE_CUDA_EXECUTION_SCHEMA,
        "exclusive_cuda_process": True,
        "device": normalized_device,
        "poll_seconds": EXCLUSIVE_CUDA_POLL_SECONDS,
        "query_backend": "nvidia-smi",
        "allowed_process_policy": "scorer_pid_only",
    }


def validate_exclusive_cuda_execution_contract(
    contract, observed_sha256, *, expected_device
):
    """Require the exact watchdog contract and its canonical hash."""
    expected = exclusive_cuda_execution_contract(expected_device)
    if contract != expected:
        raise ValueError("exclusive CUDA scoring execution contract differs")
    digest = json_sha256(expected)
    if observed_sha256 != digest:
        raise ValueError("exclusive CUDA scoring execution hash differs")
    return digest


def _task_id_binding(rows, *, label):
    """Bind row count and ordered task identities without copying all IDs."""
    task_ids = []
    observed = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} contains a non-mapping row")
        task_id = row.get("id")
        if not isinstance(task_id, str) or not task_id or task_id in observed:
            raise ValueError(f"{label} contains an empty or duplicate task id")
        observed.add(task_id)
        task_ids.append(task_id)
    return {
        "row_count": len(task_ids),
        "task_ids_sha256": json_sha256(task_ids),
    }


def _require_sha256(value, *, label, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")


def build_scoring_success_receipt(
    *,
    scores_sha256,
    score_rows,
    manifest_sha256,
    manifest_rows,
    run_config_sha256,
    run_contract_sha256,
    scoring_config_sha256,
    scoring_config,
    metric_names,
    params,
    strict,
    required_scorer_provenance_schema,
    scorer_provenance,
    scorer_provenance_sha256,
    cuda_execution_provenance,
    cuda_execution_provenance_sha256,
):
    """Build the deterministic receipt for one fully published score table."""
    _require_sha256(scores_sha256, label="scores SHA-256")
    _require_sha256(manifest_sha256, label="manifest SHA-256")
    _require_sha256(
        run_config_sha256, label="run config SHA-256", optional=True
    )
    _require_sha256(
        run_contract_sha256, label="run contract SHA-256", optional=True
    )
    _require_sha256(scoring_config_sha256, label="scoring config SHA-256")
    if type(strict) is not bool:
        raise ValueError("strict scoring flag must be boolean")
    if not isinstance(scoring_config, dict) or not isinstance(params, dict):
        raise ValueError("scoring config and params must be mappings")
    metrics = list(metric_names)
    if any(not isinstance(name, str) or not name for name in metrics):
        raise ValueError("metric names must be non-empty strings")

    scorer_contract = (
        dict(scorer_provenance) if scorer_provenance is not None else None
    )
    if (scorer_contract is None) != (scorer_provenance_sha256 is None):
        raise ValueError("scorer provenance payload and hash must appear together")
    if scorer_contract is not None:
        _require_sha256(
            scorer_provenance_sha256, label="scorer provenance SHA-256"
        )
        if json_sha256(scorer_contract) != scorer_provenance_sha256:
            raise ValueError("scorer provenance receipt binding differs")

    cuda_contract = (
        dict(cuda_execution_provenance)
        if cuda_execution_provenance is not None
        else None
    )
    if (cuda_contract is None) != (cuda_execution_provenance_sha256 is None):
        raise ValueError("CUDA execution payload and hash must appear together")
    if cuda_contract is not None:
        _require_sha256(
            cuda_execution_provenance_sha256,
            label="CUDA execution provenance SHA-256",
        )
        if json_sha256(cuda_contract) != cuda_execution_provenance_sha256:
            raise ValueError("CUDA execution receipt binding differs")

    scores_binding = _task_id_binding(score_rows, label="scores")
    manifest_binding = _task_id_binding(manifest_rows, label="manifest")
    return {
        "schema": SCORING_SUCCESS_SCHEMA,
        "scores": {"sha256": scores_sha256, **scores_binding},
        "manifest": {"sha256": manifest_sha256, **manifest_binding},
        "run_contract": {
            "config_sha256": run_config_sha256,
            "run_contract_sha256": run_contract_sha256,
        },
        "scoring_config": {
            "sha256": scoring_config_sha256,
            "payload": dict(scoring_config),
            "payload_sha256": json_sha256(dict(scoring_config)),
            "effective": {
                "metrics": metrics,
                "params": dict(params),
                "strict": strict,
                "required_scorer_provenance_schema": (
                    required_scorer_provenance_schema
                ),
            },
        },
        "scorer_provenance": {
            "contract": scorer_contract,
            "sha256": scorer_provenance_sha256,
        },
        "cuda_execution_provenance": {
            "contract": cuda_contract,
            "sha256": cuda_execution_provenance_sha256,
        },
    }


def scoring_success_receipt_bytes(receipt):
    """Serialize a receipt canonically for byte-for-byte recomputation."""
    return (
        json.dumps(
            dict(receipt),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def validate_scoring_success_receipt(receipt, **expected_inputs):
    """Rebuild and compare a scoring receipt from current artifacts."""
    if not isinstance(receipt, dict):
        raise ValueError("scoring success receipt must be a mapping")
    expected = build_scoring_success_receipt(**expected_inputs)
    if dict(receipt) != expected:
        raise ValueError("scoring success receipt differs from current artifacts")
    return expected


CFG_SCORING_SCHEMA = "cfg_scoring_contract_v1"
CFG_ACTION_SCHEMA = "cfg_baselines_v1"
CFG_SCORING_METRICS = ("pixel", "clip", "hps", "iqa")
CFG_SCORING_PARAMS = {
    "patch_crops": 5,
    "clip_model": "ViT-B/32",
    "clipscore_w": 2.5,
}
CFG_REQUIRED_SCORE_KEYS = (
    "colorfulness",
    "laplacian_sharpness",
    "clipped_fraction",
    "mean_saturation",
    "contrast_std",
    "clip_cosine",
    "clipscore",
    "hpsv2",
    "topiq_nr",
)
CFG_SCORING_FIELDS = {"metrics", "strict", "params", "required_score_keys"}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _unique_rows(rows, label):
    result = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in result:
            raise ValueError(f"{label} contains duplicate or empty id {row_id!r}")
        result[row_id] = row
    return result


def resolve_device(device, cuda_available):
    """Normalize CLI GPU indices to the Torch ``cuda:N`` device form."""
    value = str(device).strip()
    if value.isdecimal():
        value = f"cuda:{value}"
    if value != "cpu" and not cuda_available:
        return "cpu"
    return value


def finite_numeric(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def manifest_uses_hash_binding(manifest):
    """Require all-or-none image/run hashes and return whether they are present."""
    required = ("image_sha256", "run_contract_sha256")
    presence = [any(key in row for key in required) for row in manifest]
    if any(presence) and not all(presence):
        raise RuntimeError("manifest mixes hash-bound and unbound rows")
    if not any(presence):
        return False
    for row in manifest:
        for key in required:
            value = row.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RuntimeError(
                    f"{row.get('id')}: manifest {key} is not a lowercase SHA-256"
                )
    return True


def load_hash_bound_image_bytes(run_dir, row):
    """Read one regular image once and bind the returned bytes to its manifest hash."""
    root = os.path.realpath(os.path.abspath(run_dir))
    relative = row.get("image_path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{row.get('id')}: hash-bound manifest lacks image_path")
    candidate = os.path.abspath(os.path.join(root, relative))
    try:
        contained = os.path.commonpath((root, candidate)) == root
    except ValueError:
        contained = False
    if not contained or os.path.realpath(candidate) != candidate:
        raise RuntimeError(f"{row.get('id')}: hash-bound image path is unsafe")

    descriptor = -1
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{row.get('id')}: hash-bound image path is unsafe")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeError(f"{row.get('id')}: hash-bound image path is unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise RuntimeError(f"{row.get('id')}: image changed while it was read")
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != row.get("image_sha256"):
        raise RuntimeError(f"{row.get('id')}: image changed after manifest validation")
    return payload


def validate_hash_bound_images(run_dir, manifest):
    """Verify every bound image before score reuse and again after scoring."""
    for row in manifest:
        load_hash_bound_image_bytes(run_dir, row)


def validate_partial_score_bindings(manifest, scores):
    """Bind any resumable subset to current manifest images and run contract."""
    manifest_by_id = _unique_rows(manifest, "manifest")
    for score_id, score in _unique_rows(scores, "scoring progress").items():
        row = manifest_by_id.get(score_id)
        if row is None:
            raise ValueError(f"scoring progress contains unknown id {score_id!r}")
        if score.get("image_sha256") != row.get("image_sha256"):
            raise ValueError(f"{score_id}: scoring progress image hash differs")
        if score.get("run_contract_sha256") != row.get("run_contract_sha256"):
            raise ValueError(f"{score_id}: scoring progress run contract differs")


def _nvidia_smi(arguments):
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("cannot query NVIDIA compute-process state") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "cannot query NVIDIA compute-process state: " + result.stderr.strip()
        )
    return result.stdout


def _exclusive_cuda_conflicts(device, allowed_pid):
    if "CUDA_VISIBLE_DEVICES" in os.environ or "CUDA_DEVICE_ORDER" in os.environ:
        raise RuntimeError("exclusive GPU scoring requires unmasked physical CUDA ids")
    matched = re.fullmatch(r"cuda:(\d+)", str(device))
    if matched is None:
        raise RuntimeError("exclusive GPU scoring requires an explicit cuda:N device")
    physical_index = int(matched.group(1))
    inventory = _nvidia_smi(
        ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    matches = []
    for line in inventory.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) == 2 and fields[0] == str(physical_index):
            matches.append(fields[1])
    if len(matches) != 1:
        raise RuntimeError("exclusive scoring GPU is absent or ambiguous")
    target_uuid = matches[0]
    processes = _nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    conflicts = []
    for line in processes.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4 or fields[0] != target_uuid:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            pid = -1
        if pid != int(allowed_pid):
            conflicts.append(
                {
                    "gpu_uuid": fields[0],
                    "pid": pid,
                    "process_name": fields[2],
                    "used_memory_mib": fields[3],
                }
            )
    return conflicts


class ExclusiveCudaWatchdog:
    """Continuously fail scoring if a foreign process appears on its GPU."""

    def __init__(
        self, device, enabled, poll_seconds=EXCLUSIVE_CUDA_POLL_SECONDS
    ):
        self.device = device
        self.enabled = bool(enabled)
        self.poll_seconds = float(poll_seconds)
        self.pid = os.getpid()
        self.stop_event = threading.Event()
        self.failures = []
        self.thread = None

    def _check_now(self):
        try:
            conflicts = _exclusive_cuda_conflicts(self.device, self.pid)
        except ExclusiveCudaWatchdogError:
            raise
        except BaseException as exc:
            raise ExclusiveCudaWatchdogError(
                f"exclusive GPU scoring query failed: {exc}"
            ) from exc
        if conflicts:
            raise ExclusiveCudaWatchdogError(
                "foreign compute process appeared during scoring: "
                + json.dumps(conflicts, sort_keys=True)
            )

    def _monitor(self):
        while not self.stop_event.wait(self.poll_seconds):
            try:
                self._check_now()
            except BaseException as exc:
                self.failures.append(exc)
                self.stop_event.set()
                return

    def assert_healthy(self):
        if self.failures:
            raise ExclusiveCudaWatchdogError(
                "exclusive GPU scoring monitor failed"
            ) from self.failures[0]

    def __enter__(self):
        if self.enabled:
            self._check_now()
            self.thread = threading.Thread(
                target=self._monitor,
                name="exclusive-cuda-watchdog",
                daemon=True,
            )
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        if not self.enabled:
            return False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.poll_seconds * 2.0))
            if self.thread.is_alive():
                raise ExclusiveCudaWatchdogError(
                    "exclusive GPU scoring monitor did not stop"
                )
        self.assert_healthy()
        self._check_now()
        return False


def exclusive_cuda_required(cfg, cli_required=False):
    """Resolve the watchdog requirement from config and the CLI."""
    configured = cfg.get("exclusive_cuda_process", False)
    if type(configured) is not bool:
        raise RuntimeError("exclusive_cuda_process must be true or false")
    return configured or bool(cli_required)


def required_scorer_provenance_schema(cfg, run_config, cli_required=False):
    """Resolve an optional fail-closed provenance requirement."""
    requested = []
    provenance_config = cfg.get("scorer_provenance")
    if provenance_config is not None:
        if not isinstance(provenance_config, dict) or set(provenance_config) != {
            "required_schema"
        }:
            raise RuntimeError(
                "scorer_provenance config must contain only required_schema"
            )
        requested.append(provenance_config.get("required_schema"))
    run_requirement = run_config.get("required_scorer_provenance_schema")
    if run_requirement is not None:
        requested.append(run_requirement)
    if cli_required:
        requested.append(SCORER_PROVENANCE_SCHEMA)
    if not requested:
        return None
    if any(value != SCORER_PROVENANCE_SCHEMA for value in requested):
        raise RuntimeError(
            f"unsupported scorer provenance requirement; expected "
            f"{SCORER_PROVENANCE_SCHEMA!r}"
        )
    return SCORER_PROVENANCE_SCHEMA


@contextmanager
def scoring_output_lock(run_dir):
    """Exclude generation and concurrent scoring from one run directory."""
    # Generation uses the same lock file, so scoring cannot consume a manifest
    # while a resumed generator may still replace images or sidecars.
    lock_path = os.path.join(os.path.abspath(run_dir), ".generate.lock")
    handle = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "generation or scoring is already running for "
                f"{os.path.abspath(run_dir)}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def atomic_text_writer(path):
    """Publish a text file through a unique same-directory temporary file."""
    destination = os.path.abspath(path)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=os.path.dirname(destination),
    )
    handle = None
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if handle is None:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path):
    """Durably publish directory-entry changes."""
    descriptor = os.open(os.path.abspath(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_bytes(path, label):
    """Read one regular file through a no-follow descriptor."""
    descriptor = -1
    try:
        descriptor = os.open(
            os.path.abspath(path),
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeError(f"{label} is missing or unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise RuntimeError(f"{label} changed while it was read")
    return b"".join(chunks)


def _jsonl_from_bytes(payload, label):
    rows = []
    try:
        text = payload.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{label} line {line_number} is not a mapping")
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSONL") from exc
    return rows


def _scoring_receipt_context(
    *,
    run_dir,
    scoring_config_path,
    scoring_config,
    manifest,
    metric_names,
    params,
    strict,
    required_scorer_provenance_schema,
    scorer_provenance,
    scorer_provenance_sha256,
    cuda_execution_provenance,
    cuda_execution_provenance_sha256,
):
    """Capture every non-score input required to rebuild a success receipt."""
    manifest_payload = _read_regular_bytes(
        os.path.join(run_dir, "manifest.jsonl"), "manifest"
    )
    scoring_config_payload = _read_regular_bytes(
        scoring_config_path, "scoring config"
    )
    run_config_path = os.path.join(run_dir, "config.json")
    run_config_sha256 = None
    run_contract_sha256 = None
    if os.path.lexists(run_config_path):
        run_config_payload = _read_regular_bytes(run_config_path, "run config")
        run_config_sha256 = hashlib.sha256(run_config_payload).hexdigest()
        try:
            current_run_config = json.loads(run_config_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("run config is not valid JSON") from exc
        if not isinstance(current_run_config, dict):
            raise RuntimeError("run config must contain a mapping")
        run_contract_sha256 = current_run_config.get("run_contract_sha256")
    return {
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_rows": list(manifest),
        "run_config_sha256": run_config_sha256,
        "run_contract_sha256": run_contract_sha256,
        "scoring_config_sha256": hashlib.sha256(
            scoring_config_payload
        ).hexdigest(),
        "scoring_config": dict(scoring_config),
        "metric_names": list(metric_names),
        "params": dict(params),
        "strict": bool(strict),
        "required_scorer_provenance_schema": (
            required_scorer_provenance_schema
        ),
        "scorer_provenance": scorer_provenance,
        "scorer_provenance_sha256": scorer_provenance_sha256,
        "cuda_execution_provenance": cuda_execution_provenance,
        "cuda_execution_provenance_sha256": (
            cuda_execution_provenance_sha256
        ),
    }


def load_verified_score_publication(run_dir, receipt_context):
    """Load canonical rows only when their recomputed receipt is exact."""
    scores_path = os.path.join(run_dir, "scores.jsonl")
    receipt_path = os.path.join(run_dir, SCORING_SUCCESS_NAME)
    if not os.path.lexists(scores_path) and not os.path.lexists(receipt_path):
        return []
    if not os.path.lexists(scores_path) or not os.path.lexists(receipt_path):
        raise ValueError("canonical scores and success receipt must appear together")
    scores_payload = _read_regular_bytes(scores_path, "canonical scores")
    receipt_payload = _read_regular_bytes(receipt_path, "scoring success receipt")
    score_rows = _jsonl_from_bytes(scores_payload, "canonical scores")
    try:
        receipt = json.loads(receipt_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("scoring success receipt is not valid JSON") from exc
    expected = validate_scoring_success_receipt(
        receipt,
        scores_sha256=hashlib.sha256(scores_payload).hexdigest(),
        score_rows=score_rows,
        **receipt_context,
    )
    if receipt_payload != scoring_success_receipt_bytes(expected):
        raise ValueError("scoring success receipt is not canonical")
    return score_rows


def _scoring_progress_receipt_path(run_dir):
    return os.path.join(
        os.path.abspath(run_dir), SCORING_PROGRESS_RECEIPT_NAME
    )


def _scoring_progress_payload_paths(run_dir):
    root = os.path.abspath(run_dir)
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        return []
    return [
        os.path.join(root, name)
        for name in sorted(names)
        if _SCORING_PROGRESS_PAYLOAD_RE.fullmatch(name) is not None
    ]


def _scoring_progress_payload_name(attempt_id, scores_sha256):
    if _ATTEMPT_ID_RE.fullmatch(str(attempt_id)) is None:
        raise ValueError("scoring progress attempt id is invalid")
    _require_sha256(scores_sha256, label="scoring progress SHA-256")
    return f".scoring_progress.{attempt_id}.{scores_sha256}.jsonl"


def _scoring_progress_binding(receipt_context):
    template = build_scoring_success_receipt(
        scores_sha256="0" * 64,
        score_rows=[],
        **receipt_context,
    )
    template.pop("scores")
    template["schema"] = SCORING_PROGRESS_SCHEMA
    return template


def _progress_rows_follow_manifest(score_rows, manifest_rows):
    score_ids = [row.get("id") for row in score_rows]
    score_id_set = set(score_ids)
    manifest_ids = [row.get("id") for row in manifest_rows]
    return score_ids == [task_id for task_id in manifest_ids if task_id in score_id_set]


def build_scoring_progress_receipt(
    *, attempt_id, scores_payload, score_rows, receipt_context
):
    if not isinstance(attempt_id, str) or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        raise ValueError("scoring progress attempt id is invalid")
    if not _progress_rows_follow_manifest(score_rows, receipt_context["manifest_rows"]):
        raise ValueError("scoring progress task order differs from the manifest")
    scores_sha256 = hashlib.sha256(scores_payload).hexdigest()
    scores_name = _scoring_progress_payload_name(attempt_id, scores_sha256)
    binding = _scoring_progress_binding(receipt_context)
    return {
        "schema": SCORING_PROGRESS_SCHEMA,
        "status": "in_progress",
        "attempt_id": attempt_id,
        "binding": binding,
        "binding_sha256": json_sha256(binding),
        "scores": {
            "path": scores_name,
            "sha256": scores_sha256,
            **_task_id_binding(score_rows, label="scoring progress"),
        },
    }


def load_verified_scoring_progress(run_dir, receipt_context):
    receipt_path = _scoring_progress_receipt_path(run_dir)
    if not os.path.lexists(receipt_path):
        return [], None, None
    receipt_payload = _read_regular_bytes(receipt_path, "scoring progress receipt")
    try:
        receipt = json.loads(receipt_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("scoring progress receipt is not valid JSON") from exc
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "status",
        "attempt_id",
        "binding",
        "binding_sha256",
        "scores",
    }:
        raise ValueError("scoring progress receipt fields differ")
    scores = receipt.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("scoring progress scores binding is invalid")
    scores_name = scores.get("path")
    match = (
        _SCORING_PROGRESS_PAYLOAD_RE.fullmatch(scores_name)
        if isinstance(scores_name, str)
        else None
    )
    if match is None or match.group(1) != receipt.get("attempt_id"):
        raise ValueError("scoring progress payload path is invalid")
    if match.group(2) != scores.get("sha256"):
        raise ValueError("scoring progress payload name differs from its hash")
    scores_path = os.path.join(os.path.abspath(run_dir), scores_name)
    scores_payload = _read_regular_bytes(scores_path, "scoring progress")
    score_rows = _jsonl_from_bytes(scores_payload, "scoring progress")
    expected = build_scoring_progress_receipt(
        attempt_id=receipt.get("attempt_id"),
        scores_payload=scores_payload,
        score_rows=score_rows,
        receipt_context=receipt_context,
    )
    if receipt != expected:
        raise ValueError("scoring progress receipt differs from current inputs")
    if receipt_payload != scoring_success_receipt_bytes(expected):
        raise ValueError("scoring progress receipt is not canonical")
    return score_rows, str(receipt["attempt_id"]), scores_path


def _write_scoring_progress(
    run_dir, attempt_id, score_rows, receipt_context, *, allow_nan
):
    root = os.path.abspath(run_dir)
    descriptor, staged_path = tempfile.mkstemp(
        prefix=".scoring_progress.stage.", suffix=".tmp", dir=root
    )
    handle = None
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        with handle:
            for row in score_rows:
                handle.write(json.dumps(row, allow_nan=allow_nan) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        scores_payload = _read_regular_bytes(staged_path, "staged scoring progress")
        scores_sha256 = hashlib.sha256(scores_payload).hexdigest()
        scores_name = _scoring_progress_payload_name(attempt_id, scores_sha256)
        scores_path = os.path.join(root, scores_name)
        os.replace(staged_path, scores_path)
    except BaseException:
        if handle is None:
            os.close(descriptor)
        try:
            os.unlink(staged_path)
        except FileNotFoundError:
            pass
        raise
    _fsync_directory(root)
    receipt = build_scoring_progress_receipt(
        attempt_id=attempt_id,
        scores_payload=scores_payload,
        score_rows=score_rows,
        receipt_context=receipt_context,
    )
    receipt_path = _scoring_progress_receipt_path(root)
    with atomic_text_writer(receipt_path) as handle:
        handle.write(scoring_success_receipt_bytes(receipt).decode("ascii"))
    _fsync_directory(root)
    _, _, verified_path = load_verified_scoring_progress(root, receipt_context)
    for path in _scoring_progress_payload_paths(root):
        if path != verified_path:
            os.unlink(path)
    _fsync_directory(root)


def _clear_scoring_progress(run_dir):
    changed = False
    receipt_path = _scoring_progress_receipt_path(run_dir)
    paths = [receipt_path, *_scoring_progress_payload_paths(run_dir)]
    for path in paths:
        if not os.path.lexists(path):
            continue
        try:
            os.unlink(path)
        except IsADirectoryError as exc:
            raise RuntimeError(f"scoring progress artifact is unsafe: {path}") from exc
        changed = True
    if changed:
        _fsync_directory(run_dir)


def _clear_canonical_scoring_outputs(run_dir):
    """Invalidate the receipt first, then remove both canonical artifacts."""
    changed = False
    for name in (SCORING_SUCCESS_NAME, "scores.jsonl"):
        path = os.path.join(run_dir, name)
        if not os.path.lexists(path):
            continue
        try:
            os.unlink(path)
        except IsADirectoryError as exc:
            raise RuntimeError(f"canonical scoring artifact is unsafe: {path}") from exc
        changed = True
    if changed:
        _fsync_directory(run_dir)


def _canonical_scoring_backup(run_dir):
    scores_path = os.path.join(run_dir, "scores.jsonl")
    receipt_path = os.path.join(run_dir, SCORING_SUCCESS_NAME)
    if not os.path.lexists(scores_path) and not os.path.lexists(receipt_path):
        return None
    if not os.path.lexists(scores_path) or not os.path.lexists(receipt_path):
        return None
    try:
        return (
            _read_regular_bytes(scores_path, "previous canonical scores"),
            _read_regular_bytes(receipt_path, "previous scoring receipt"),
        )
    except (RuntimeError, ValueError):
        return None


def _restore_canonical_scoring_backup(run_dir, backup):
    scores_payload, receipt_payload = backup
    scores_path = os.path.join(run_dir, "scores.jsonl")
    receipt_path = os.path.join(run_dir, SCORING_SUCCESS_NAME)
    try:
        with atomic_text_writer(scores_path) as handle:
            handle.write(scores_payload.decode("utf-8"))
        with atomic_text_writer(receipt_path) as handle:
            handle.write(receipt_payload.decode("utf-8"))
        _fsync_directory(run_dir)
        if _read_regular_bytes(
            scores_path, "restored canonical scores"
        ) != scores_payload or _read_regular_bytes(
            receipt_path, "restored scoring receipt"
        ) != receipt_payload:
            raise RuntimeError("restored canonical scoring bytes differ")
    except BaseException:
        _clear_canonical_scoring_outputs(run_dir)
        raise


def _publish_scoring_attempt(run_dir, manifest, receipt_context):
    """Atomically publish scores, then their receipt, or leave neither resumable."""
    scores_path = os.path.join(run_dir, "scores.jsonl")
    receipt_path = os.path.join(run_dir, SCORING_SUCCESS_NAME)
    verified_rows, _, attempt_path = load_verified_scoring_progress(
        run_dir, receipt_context
    )
    if attempt_path is None:
        raise RuntimeError("verified scoring progress is missing before publication")
    attempt_payload = _read_regular_bytes(attempt_path, "scoring attempt")
    score_rows = _jsonl_from_bytes(attempt_payload, "scoring attempt")
    if score_rows != verified_rows:
        raise RuntimeError("scoring progress changed before final publication")
    if [row.get("id") for row in score_rows] != [row.get("id") for row in manifest]:
        raise RuntimeError("scoring attempt task order differs from the manifest")
    receipt = build_scoring_success_receipt(
        scores_sha256=hashlib.sha256(attempt_payload).hexdigest(),
        score_rows=score_rows,
        **receipt_context,
    )
    previous_publication = _canonical_scoring_backup(run_dir)
    if previous_publication is None:
        _clear_canonical_scoring_outputs(run_dir)
    try:
        with atomic_text_writer(scores_path) as handle:
            handle.write(attempt_payload.decode("utf-8"))
        _fsync_directory(run_dir)
        if _read_regular_bytes(scores_path, "canonical scores") != attempt_payload:
            raise RuntimeError("canonical scores changed during publication")
        receipt_payload = scoring_success_receipt_bytes(receipt)
        with atomic_text_writer(receipt_path) as handle:
            handle.write(receipt_payload.decode("ascii"))
        _fsync_directory(run_dir)
        load_verified_score_publication(run_dir, receipt_context)
    except BaseException:
        if previous_publication is None:
            _clear_canonical_scoring_outputs(run_dir)
        else:
            _restore_canonical_scoring_backup(
                run_dir, previous_publication
            )
        raise
    try:
        _clear_scoring_progress(run_dir)
    except RuntimeError as exc:
        print(f"[warn] could not remove completed scoring progress -> {exc}", flush=True)
    return score_rows


def cfg_scoring_contract(run_config, metric_names, params, strict):
    """Load and enforce the scoring recipe bound by a registered CFG run."""
    if run_config.get("cfg_baseline_registered") is not True:
        return None
    actions_path = run_config.get("actions_yaml")
    actions_hash = run_config.get("actions_sha256")
    if not isinstance(actions_path, str) or not os.path.isfile(actions_path):
        raise RuntimeError("registered CFG run actions YAML is unavailable for scoring")
    with open(actions_path, "rb") as handle:
        actions_bytes = handle.read()
    if hashlib.sha256(actions_bytes).hexdigest() != actions_hash:
        raise RuntimeError("registered CFG run actions YAML changed before scoring")
    actions_config = yaml.safe_load(actions_bytes) or {}
    if not isinstance(actions_config, dict):
        raise RuntimeError("registered CFG actions YAML must contain a mapping")
    if actions_config.get("schema") != CFG_ACTION_SCHEMA:
        raise RuntimeError("registered CFG actions YAML has the wrong schema")
    scoring = actions_config.get("scoring")
    if not isinstance(scoring, dict):
        raise RuntimeError("registered CFG actions lack a scoring contract")
    if set(scoring) != CFG_SCORING_FIELDS:
        raise RuntimeError("registered CFG scoring fields differ from the v1 contract")
    registered_metrics = scoring.get("metrics")
    registered_params = scoring.get("params")
    if registered_metrics != list(CFG_SCORING_METRICS):
        raise RuntimeError("registered CFG YAML metrics differ from the v1 contract")
    if list(metric_names) != list(CFG_SCORING_METRICS):
        raise RuntimeError(
            f"registered CFG scoring requires metrics {list(CFG_SCORING_METRICS)}, "
            f"got {list(metric_names)}"
        )
    if strict is not True or scoring.get("strict") is not True:
        raise RuntimeError("registered CFG scoring requires --strict")
    if json_sha256(registered_params) != json_sha256(CFG_SCORING_PARAMS):
        raise RuntimeError("registered CFG YAML params differ from the v1 contract")
    if json_sha256(params) != json_sha256(CFG_SCORING_PARAMS):
        raise RuntimeError("scoring config params differ from CFG registration")
    if scoring.get("required_score_keys") != list(CFG_REQUIRED_SCORE_KEYS):
        raise RuntimeError(
            "registered CFG required score keys differ from the v1 contract"
        )
    return {
        "schema": CFG_SCORING_SCHEMA,
        "action_schema": CFG_ACTION_SCHEMA,
        "metrics": list(CFG_SCORING_METRICS),
        "strict": True,
        "params": dict(CFG_SCORING_PARAMS),
        "required_score_keys": list(CFG_REQUIRED_SCORE_KEYS),
        "actions_sha256": actions_hash,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default=os.path.join(THIS, "configs", "eval_common.yaml"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metrics", default=None, help="comma-separated override of config 'metrics'")
    ap.add_argument(
        "--strict", action="store_true",
        help="fail if any requested scorer cannot load or score an image",
    )
    ap.add_argument(
        "--require-scorer-provenance",
        action="store_true",
        help="require the hardened scorer provenance schema (implies --strict)",
    )
    ap.add_argument(
        "--require-exclusive-gpu",
        action="store_true",
        help="stop if another compute PID appears on the explicit physical cuda:N",
    )
    args = ap.parse_args()
    if args.require_scorer_provenance:
        args.strict = True

    with scoring_output_lock(args.run_dir):
        return _score_run(args, ap)


def _score_run(args, ap):
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError("scoring config must contain a YAML mapping")
    import torch
    device = resolve_device(args.device, torch.cuda.is_available())
    try:
        torch.device(device)
    except (RuntimeError, TypeError) as exc:
        ap.error(f"invalid --device {args.device!r}: {exc}")
    watchdog = ExclusiveCudaWatchdog(
        device,
        exclusive_cuda_required(
            cfg, getattr(args, "require_exclusive_gpu", False)
        ),
    )
    metric_names = args.metrics.split(",") if args.metrics else cfg.get("metrics", [])
    params = dict(cfg.get("params", {}))
    registrations = [cfg]
    run_config_path = os.path.join(args.run_dir, "config.json")
    if os.path.isfile(run_config_path):
        with open(run_config_path) as handle:
            candidate_run_config = json.load(handle)
        if isinstance(candidate_run_config, dict):
            registrations.append(candidate_run_config)
    registered_hashes = []
    for registration in registrations:
        _, candidate_hash = registered_scorer_provenance_contract(registration)
        if candidate_hash is not None:
            registered_hashes.append(candidate_hash)
    if len(set(registered_hashes)) > 1:
        raise RuntimeError("registered scorer provenance hashes disagree")
    formal_registered_scoring = bool(registered_hashes)
    try:
        with formal_offline_network_guard(
            enabled=formal_registered_scoring
        ), staged_scorer_assets(
            metric_names,
            params,
            enabled=formal_registered_scoring,
        ) as asset_stage:
            with watchdog:
                publication = _score_run_impl(
                    args,
                    ap,
                    cfg,
                    torch,
                    device,
                    watchdog,
                    asset_stage,
                )
            if asset_stage is not None:
                asset_stage.verify()
        watchdog.assert_healthy()
        if publication is None:
            return None
        current_context = _scoring_receipt_context(
            **publication["receipt_context_args"]
        )
        if current_context != publication["receipt_context"]:
            raise RuntimeError("scoring inputs changed before publication")
        _publish_scoring_attempt(
            args.run_dir,
            publication["manifest"],
            current_context,
        )
        print(
            f"scores -> {os.path.join(args.run_dir, 'scores.jsonl')}",
            flush=True,
        )
        print(
            f"receipt -> {os.path.join(args.run_dir, SCORING_SUCCESS_NAME)}",
            flush=True,
        )
        return None
    except BaseException as exc:
        if isinstance(
            exc, (ExclusiveCudaWatchdogError, ScorerAssetStageError)
        ):
            _clear_scoring_progress(args.run_dir)
        raise


def _score_run_impl(args, ap, cfg, torch, device, watchdog, asset_stage=None):
    metric_names = args.metrics.split(",") if args.metrics else cfg.get("metrics", [])
    params = dict(cfg.get("params", {}))
    exclusive_execution = (
        exclusive_cuda_execution_contract(device) if watchdog.enabled else None
    )
    exclusive_execution_sha256 = (
        json_sha256(exclusive_execution) if exclusive_execution is not None else None
    )

    manifest = load_jsonl(os.path.join(args.run_dir, "manifest.jsonl"))
    if not manifest:
        raise RuntimeError("manifest.jsonl is empty or missing")
    manifest_by_id = _unique_rows(manifest, "manifest")
    run_config_path = os.path.join(args.run_dir, "config.json")
    run_config = {}
    if os.path.isfile(run_config_path):
        with open(run_config_path) as handle:
            run_config = json.load(handle)
    if (
        run_config.get("split_role") == "engineering_smoke"
        or run_config.get("engineering_only") is True
    ):
        raise RuntimeError(
            "structural engineering smoke forbids quality scoring"
        )
    required_provenance_schema = required_scorer_provenance_schema(
        cfg,
        run_config,
        getattr(args, "require_scorer_provenance", False),
    )
    if required_provenance_schema is not None and not args.strict:
        raise RuntimeError("hardened scorer provenance requires --strict")
    s7_run = any(
        run_config.get(flag) is True
        for flag in (
            "trajectory_registered",
            "scheduler_baseline_registered",
            "cfg_baseline_registered",
            "native_renderer_registered",
        )
    ) or any(
        row.get("provenance_schema") == PROVENANCE_SCHEMA for row in manifest
    )
    if s7_run:
        contract_hash = validate_run_contract(run_config)
        action_ids = [str(action.get("id")) for action in run_config.get("actions", [])]
        seeds = [int(value) for value in run_config.get("seeds", [])]
        validate_design_rows(
            manifest,
            expected_action_ids=action_ids or None,
            expected_seeds=seeds or None,
        )
        for row in manifest:
            if row.get("provenance_schema") != PROVENANCE_SCHEMA:
                raise RuntimeError(f"{row.get('id')}: missing S7 provenance schema")
            validate_sidecar(
                row,
                args.run_dir,
                expected_contract_sha256=contract_hash,
            )
    hash_bound_run = manifest_uses_hash_binding(manifest)
    if hash_bound_run:
        validate_hash_bound_images(args.run_dir, manifest)
    registered_scoring = cfg_scoring_contract(
        run_config, metric_names, params, args.strict
    )
    registered_scoring_sha256 = (
        json_sha256(registered_scoring) if registered_scoring is not None else None
    )
    # instantiate scorers (validate weights first; skip cleanly if missing/broken)
    active = []
    unavailable = []
    for name in metric_names:
        if name not in REGISTRY:
            message = f"unknown metric '{name}'"
            print(f"[warn] {message}, skipping", flush=True)
            unavailable.append(message)
            continue
        cls = REGISTRY[name]
        ready, msg = cls.weights_status(**params)
        if not ready:
            message = f"'{name}' weights missing -> {msg}"
            print(f"[skip] {message}", flush=True)
            unavailable.append(message)
            continue
        try:
            staged_paths = (
                asset_stage.asset_paths.get(name, {})
                if asset_stage is not None
                else {}
            )
            staged_revisions = (
                asset_stage.asset_revisions.get(name, {})
                if asset_stage is not None
                else {}
            )
            staged_manifest = (
                asset_stage.asset_manifests.get(name)
                if asset_stage is not None and staged_paths
                else None
            )
            active.append(
                (
                    name,
                    cls(
                        device=device,
                        scorer_assets=staged_paths,
                        scorer_asset_revisions=staged_revisions,
                        scorer_asset_manifest=staged_manifest,
                        **params,
                    ),
                )
            )
            print(f"[ok] loaded scorer '{name}' on {device}", flush=True)
            if asset_stage is not None:
                asset_stage.verify()
            watchdog.assert_healthy()
        except Exception as e:
            message = f"'{name}' failed to init -> {e}"
            print(f"[skip] {message}", flush=True)
            unavailable.append(message)
    if args.strict and unavailable:
        raise RuntimeError("requested scorers unavailable: " + "; ".join(unavailable))
    if not active:
        print("no active scorers; nothing to do", flush=True)
        return None

    scorer_provenance = None
    scorer_provenance_sha256 = None
    if args.strict:
        scorer_provenance, scorer_provenance_sha256 = build_scorer_provenance(
            active,
            params=params,
            device=device,
            runner_path=__file__,
            base_path=scorer_base.__file__,
            source_root=THIS,
        )
    registered_scorer_contract = None
    registered_scorer_hash = None
    for registration in (cfg, run_config):
        candidate_contract, candidate_hash = registered_scorer_provenance_contract(
            registration
        )
        if candidate_contract is not None:
            if (
                registered_scorer_contract is not None
                and candidate_contract != registered_scorer_contract
            ):
                raise RuntimeError("registered scorer provenance contracts disagree")
            registered_scorer_contract = candidate_contract
        if candidate_hash is not None:
            if (
                registered_scorer_hash is not None
                and candidate_hash != registered_scorer_hash
            ):
                raise RuntimeError("registered scorer provenance hashes disagree")
            registered_scorer_hash = candidate_hash
    if registered_scorer_hash is not None:
        if scorer_provenance is None:
            raise RuntimeError(
                "a registered scorer provenance contract requires --strict"
            )
        if scorer_provenance_sha256 != registered_scorer_hash:
            raise RuntimeError(
                "loaded scorer provenance differs from the registered contract"
            )
        if registered_scorer_contract is not None and (
            scorer_provenance != registered_scorer_contract
        ):
            raise RuntimeError(
                "loaded scorer provenance differs from the registered payload"
            )

    receipt_context_args = {
        "run_dir": args.run_dir,
        "scoring_config_path": args.config,
        "scoring_config": cfg,
        "manifest": manifest,
        "metric_names": metric_names,
        "params": params,
        "strict": args.strict,
        "required_scorer_provenance_schema": required_provenance_schema,
        "scorer_provenance": scorer_provenance,
        "scorer_provenance_sha256": scorer_provenance_sha256,
        "cuda_execution_provenance": exclusive_execution,
        "cuda_execution_provenance_sha256": exclusive_execution_sha256,
    }
    receipt_context = _scoring_receipt_context(**receipt_context_args)
    canonical_publication_loaded = False
    try:
        existing_rows = load_verified_score_publication(
            args.run_dir, receipt_context
        )
    except (RuntimeError, ValueError) as exc:
        print(
            f"[warn] canonical scores are not resumable -> {exc}",
            flush=True,
        )
        existing_rows = []
    else:
        canonical_publication_loaded = bool(
            os.path.lexists(os.path.join(args.run_dir, "scores.jsonl"))
        )
    progress_attempt_id = uuid.uuid4().hex
    loaded_from_progress = False
    if not canonical_publication_loaded:
        try:
            progress_rows, resumed_attempt_id, _ = load_verified_scoring_progress(
                args.run_dir, receipt_context
            )
        except (RuntimeError, ValueError) as exc:
            print(
                f"[warn] private scoring progress is not resumable -> {exc}",
                flush=True,
            )
            _clear_scoring_progress(args.run_dir)
        else:
            if resumed_attempt_id is not None:
                existing_rows = progress_rows
                progress_attempt_id = resumed_attempt_id
                loaded_from_progress = True
    existing = _unique_rows(existing_rows, "scores")
    if hash_bound_run and existing_rows:
        try:
            validate_partial_score_bindings(manifest, existing_rows)
        except ValueError:
            # A private receipt cannot make rows for replaced images reusable.
            existing = {}
            if loaded_from_progress:
                _clear_scoring_progress(args.run_dir)
                progress_attempt_id = uuid.uuid4().hex

    output_keys = [key for _, scorer in active for key, _ in scorer.OUTPUT_KEYS]
    if len(output_keys) != len(set(output_keys)):
        raise RuntimeError("active scorers declare duplicate output keys")
    need_keys = set(output_keys)
    if registered_scoring is not None and need_keys != set(CFG_REQUIRED_SCORE_KEYS):
        raise RuntimeError(
            "registered CFG scorer outputs differ from required_score_keys"
        )

    def score_provenance_is_current(row, score):
        if (
            score.get("scoring_execution_provenance") != exclusive_execution
            or score.get("scoring_execution_provenance_sha256")
            != exclusive_execution_sha256
        ):
            return False
        if hash_bound_run:
            if (
                score.get("image_sha256") != row.get("image_sha256")
                or score.get("run_contract_sha256")
                != row.get("run_contract_sha256")
            ):
                return False
        if s7_run:
            if registered_scoring is not None and not (
                score.get("scoring_contract") == registered_scoring
                and score.get("scoring_contract_sha256")
                == registered_scoring_sha256
            ):
                return False
        if scorer_provenance is not None:
            return (
                score.get("scorer_provenance") == scorer_provenance
                and score.get("scorer_provenance_sha256")
                == scorer_provenance_sha256
            )
        return True

    def score_is_current(row):
        if row["id"] not in existing or not need_keys.issubset(existing[row["id"]].keys()):
            return False
        score = existing[row["id"]]
        if not score_provenance_is_current(row, score):
            return False
        return not (args.strict or hash_bound_run) or all(
            finite_numeric(score.get(key)) for key in need_keys
        )

    todo = [r for r in manifest if not score_is_current(r)]
    print(f"{len(manifest)} images; {len(todo)} to (re)score with {[n for n, _ in active]}", flush=True)

    def flush():
        rows = [existing[r["id"]] for r in manifest if r["id"] in existing]
        _write_scoring_progress(
            args.run_dir,
            progress_attempt_id,
            rows,
            receipt_context,
            allow_nan=not args.strict and registered_scoring is None,
        )

    if canonical_publication_loaded and not todo:
        print("canonical scores are complete and current; nothing to do", flush=True)
        return None
    from PIL import Image
    for i, r in enumerate(todo):
        watchdog.assert_healthy()
        if hash_bound_run:
            image_bytes = load_hash_bound_image_bytes(args.run_dir, r)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            image_path = os.path.join(args.run_dir, r["image_path"])
            img = Image.open(image_path).convert("RGB")
        metadata_keys = (
            "id", "prompt_index", "bucket", "seed", "scale", "action_id",
            "action_type", "band_scales", "image_path",
        )
        prior = existing.get(r["id"])
        rec = (
            prior
            if prior is not None and score_provenance_is_current(r, prior)
            else {key: r[key] for key in metadata_keys if key in r}
        )
        if hash_bound_run:
            rec.update(
                {
                    "image_sha256": r["image_sha256"],
                    "run_contract_sha256": r["run_contract_sha256"],
                }
            )
        if s7_run:
            rec.update(
                {
                    "provenance_schema": PROVENANCE_SCHEMA,
                    "action_sha256": r.get("action_sha256"),
                }
            )
            if registered_scoring is not None:
                rec.update(
                    {
                        "scoring_contract": registered_scoring,
                        "scoring_contract_sha256": registered_scoring_sha256,
                    }
                )
        if scorer_provenance is not None:
            rec.update(
                {
                    "scorer_provenance": scorer_provenance,
                    "scorer_provenance_sha256": scorer_provenance_sha256,
                }
            )
        if exclusive_execution is not None:
            rec.update(
                {
                    "scoring_execution_provenance": exclusive_execution,
                    "scoring_execution_provenance_sha256": (
                        exclusive_execution_sha256
                    ),
                }
            )
        for name, sc in active:
            scorer_keys = {key for key, _ in sc.OUTPUT_KEYS}
            if scorer_keys.issubset(rec) and (
                not (args.strict or s7_run)
                or all(finite_numeric(rec.get(key)) for key in scorer_keys)
            ):
                continue
            for key in scorer_keys:
                rec.pop(key, None)
            try:
                scored = sc.score_image(img, r["prompt"])
                if registered_scoring is not None and (
                    not isinstance(scored, dict) or set(scored) != scorer_keys
                ):
                    raise ValueError(
                        f"{name} returned fields outside its registered output contract"
                    )
                if (args.strict or hash_bound_run) and any(
                    not finite_numeric(scored.get(key)) for key in scorer_keys
                ):
                    raise ValueError(f"{name} returned non-finite strict scores")
                rec.update(scored)
            except Exception as e:
                print(f"[warn] {name} failed on {r['id']}: {e}", flush=True)
                if args.strict:
                    existing[r["id"]] = rec
                    flush()
                    raise RuntimeError(f"{name} failed on {r['id']}") from e
            watchdog.assert_healthy()
        existing[r["id"]] = rec
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            flush()
            print(f"  scored {i + 1}/{len(todo)}", flush=True)
    flush()
    watchdog.assert_healthy()
    if hash_bound_run:
        validate_hash_bound_images(args.run_dir, manifest)
        validate_scores_against_manifest(manifest, [existing[row["id"]] for row in manifest])
    if scorer_provenance is not None:
        validate_hardened_score_rows(
            [existing[row["id"]] for row in manifest],
            required_schema=required_provenance_schema
            or SCORER_PROVENANCE_SCHEMA,
            expected_sha256=registered_scorer_hash,
            expected_contract=registered_scorer_contract,
        )
    if asset_stage is not None:
        asset_stage.verify()
    return {
        "manifest": manifest,
        "receipt_context_args": receipt_context_args,
        "receipt_context": receipt_context,
    }


if __name__ == "__main__":
    main()
