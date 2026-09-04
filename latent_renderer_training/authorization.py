"""Fail-closed authorization for latent-renderer training data.

Training is authorized by a formal selected-view child release.  Its v1
candidate parent remains non-training-ready; the child supplies the frozen
selection policy, complete Data Gate evidence, and exactly 64 train plus 32
validation rows.  The loader runs the canonical child validator and snapshots
every bound byte so later optimizer updates fail if any dependency drifts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AUTHORIZATION_SCHEMA = "repldm.training_authorization.v2"
SELECTED_VIEW_SCHEMA = "repldm.selected_view_release.v1"
SELECTED_CONFIG_SCHEMA = "repldm.selected_view_config.v2"
SELECTED_ROW_SCHEMA = "repldm.selected_data_record.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CATALOG_ID_RE = re.compile(r"^catalog-[0-9a-f]{20}$")
_SELECTED_ID_RE = re.compile(r"^selected-view-[0-9a-f]{20}$")
_PACKAGE_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# The formal gate must not resolve Git through a caller-controlled PATH.  Keep
# these absolute candidates aligned with the pinned executable used by the
# trusted production launcher; an unavailable or redirected binary fails
# closed in ``_trusted_git_executable``.
_GIT_EXECUTABLE = "/usr/bin/git"
_GIT_FALLBACK_EXECUTABLE = "/bin/git"
_FORBIDDEN_GIT_ENVIRONMENT = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_WORK_TREE",
    }
)
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "LC_ALL": "C",
}
_GIT_CONFIG_OVERRIDES = (
    ("core.fsmonitor", "false"),
    ("core.untrackedCache", "false"),
    ("core.splitIndex", "false"),
    ("core.sparseCheckout", "false"),
    ("index.sparse", "false"),
    ("status.aheadBehind", "false"),
)
_PARENT_ARTIFACT_ORDER = (
    "benchmark_holdouts.jsonl",
    "four_k_lsdb.jsonl",
    "pixverve_95k.jsonl",
    "sana_pickscore_prompts.jsonl",
    "sana_ocr_prompts.jsonl",
    "sana_geneval_prompts.jsonl",
    "sana_drawbench_prompts.jsonl",
    "aesthetic_4k.jsonl",
    "style30k.jsonl",
    "omnistyle_available.jsonl",
    "pixel_render_image_only.jsonl",
    "sana_4klsdb_text_features.jsonl",
    "training_candidates.jsonl",
    "training_views.jsonl",
    "source_inventory.jsonl",
)

# A copied dataclass must not inherit authorization.  Keep an identity-based
# registry whose weak references disappear with the validated object.
_VALIDATED_IDENTITIES: dict[int, Any] = {}
_VALIDATED_BINDINGS: dict[int, Any] = {}


def _register_validated(value: "TrainingAuthorization") -> None:
    identity = id(value)

    def remove(reference: Any, *, identity: int = identity) -> None:
        if _VALIDATED_IDENTITIES.get(identity) is reference:
            _VALIDATED_IDENTITIES.pop(identity, None)

    import weakref

    _VALIDATED_IDENTITIES[identity] = weakref.ref(value, remove)


def _is_registered(value: "TrainingAuthorization") -> bool:
    reference = _VALIDATED_IDENTITIES.get(id(value))
    return reference is not None and reference() is value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(f"cannot open protected file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not os.path.isfile(path) or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"protected path is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)

        def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if identity(before) != identity(after):
            raise RuntimeError(f"protected file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _absolute_without_symlinks(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    normalized = Path(os.path.abspath(os.fspath(path)))
    current = Path(normalized.anchor)
    for component in normalized.parts[1:]:
        current /= component
        try:
            status = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} cannot contain a symlink: {current}")
    return normalized


def _regular_file(path: Path, *, label: str) -> Path:
    path = _absolute_without_symlinks(path, label=label)
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        raise ValueError(f"{label} must be a non-empty ordinary file: {path}")
    return path


def _inside(path: Path, root: Path, *, label: str) -> None:
    path = _absolute_without_symlinks(path, label=label)
    root = _absolute_without_symlinks(root, label=f"{label} root")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root") from exc


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _git_environment() -> dict[str, str]:
    """Return a minimal Git environment bound to this checkout's metadata."""
    # Do not inherit any caller-controlled GIT_* variables (or PATH/PYTHONPATH
    # when this environment is used for the validator subprocess).  Git's
    # local checkout config remains available because only global/system and
    # command-injection config sources are disabled above.
    return dict(_GIT_ENVIRONMENT)


def _trusted_git_executable() -> str:
    """Return a fixed, non-symlink regular Git executable."""
    for candidate_raw in (_GIT_EXECUTABLE, _GIT_FALLBACK_EXECUTABLE):
        candidate = Path(candidate_raw)
        try:
            resolved = candidate.resolve(strict=True)
            status = resolved.stat()
        except (OSError, RuntimeError):
            continue
        # Requiring the candidate itself to be its real path prevents a caller
        # from replacing an absolute launcher path with an arbitrary symlink.
        if (
            resolved != candidate
            or not stat.S_ISREG(status.st_mode)
            or not (status.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
            or not os.access(resolved, os.X_OK)
        ):
            continue
        return str(resolved)
    raise ValueError("trusted Git executable is unavailable or redirected")


def _git_command(repository_root: Path, *arguments: str) -> list[str]:
    """Build a Git command explicitly bound to this checkout.

    Local Git config is still read for ordinary repository metadata, but the
    options that can redirect the work tree or execute a helper are overridden
    on the command line.  Explicit ``--git-dir``/``--work-tree`` also prevents
    inherited environment or ``core.worktree`` values from changing the
    checkout being inspected.
    """
    root = _absolute_without_symlinks(repository_root, label="repository root")
    git_dir = root / ".git"
    try:
        git_dir_status = git_dir.lstat()
    except OSError as exc:
        raise ValueError("training repository Git directory is unavailable") from exc
    if not stat.S_ISDIR(git_dir_status.st_mode):
        raise ValueError("training repository Git directory must be an ordinary directory")
    command = [
        _trusted_git_executable(),
        "--no-optional-locks",
        f"--git-dir={git_dir}",
        f"--work-tree={root}",
    ]
    for key, value in _GIT_CONFIG_OVERRIDES:
        command.extend(("-c", f"{key}={value}"))
    command.extend(arguments)
    return command


def _git_state(repository_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            _git_command(repository_root, "rev-parse", "HEAD"),
            cwd=repository_root,
            env=_git_environment(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        status = subprocess.run(
            _git_command(
                repository_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--no-renames",
            ),
            cwd=repository_root,
            env=_git_environment(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        # ``git status`` deliberately trusts the index's assume-unchanged and
        # skip-worktree bits.  Either bit can hide edits to a tracked source
        # file, so a clean porcelain result is not sufficient for a formal
        # checkout gate.  ``ls-files -v`` is read-only and exposes both flags
        # (`h` for assume-unchanged and `S` for skip-worktree).
        index_entries = subprocess.run(
            _git_command(repository_root, "ls-files", "-v", "-z"),
            cwd=repository_root,
            env=_git_environment(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot inspect the training repository Git state") from exc
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("current Git HEAD is not a full commit")
    suppressed = False
    for entry in index_entries.split(b"\0"):
        if not entry:
            continue
        tag = entry[:1]
        if len(entry) < 2 or entry[1:2] != b" ":
            raise ValueError("cannot inspect the training repository Git index")
        if tag.islower() or tag == b"S":
            suppressed = True
            break
    return commit, not bool(status) and not suppressed


def _git_upstream(repository_root: Path, *, expected_commit: str) -> None:
    try:
        upstream_ref = subprocess.run(
            _git_command(
                repository_root,
                "rev-parse",
                "--symbolic-full-name",
                "@{upstream}",
            ),
            cwd=repository_root,
            env=_git_environment(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        upstream = subprocess.run(
            _git_command(repository_root, "rev-parse", "@{upstream}"),
            cwd=repository_root,
            env=_git_environment(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("training repository has no reachable upstream commit") from exc
    if not upstream_ref.startswith("refs/remotes/"):
        raise ValueError("training repository upstream is not a remote-tracking ref")
    if upstream != expected_commit:
        raise ValueError("training repository is not at the pushed authorized commit")


def _require_repository_root(value: str | os.PathLike[str] | None) -> Path:
    if value is None:
        raise ValueError("repository_root is required for training authorization")
    root = _absolute_without_symlinks(Path(value), label="repository root")
    if not root.is_dir():
        raise ValueError("training repository root is not a directory")
    if root != _PACKAGE_REPOSITORY_ROOT:
        raise ValueError("repository_root must be the repository containing this package")
    return root


def _validate_selected_view_release(release_dir: Path, repository_root: Path) -> None:
    """Run the canonical selected-view validator in a separate process."""
    selected_root = repository_root / "DATA" / "selected-views"
    _inside(release_dir, selected_root, label="selected-view release")
    if release_dir.parent != selected_root:
        raise ValueError("selected-view release must be a direct child of DATA/selected-views")
    if _SELECTED_ID_RE.fullmatch(release_dir.name) is None:
        raise ValueError("selected-view release has an invalid content-addressed ID")
    validator = _regular_file(
        repository_root / "eval-pipeline" / "validate_selected_view_release.py",
        label="selected-view validator",
    )
    try:
        result = subprocess.run(
            [sys.executable, str(validator), "--release-dir", str(release_dir)],
            cwd=repository_root,
            env=_git_environment(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("formal selected-view validation could not be completed") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[-1] if detail else "validator returned a non-zero status"
        raise ValueError(f"formal selected-view validation failed: {reason}")


def _require_tracked_source(
    repository_root: Path, path: Path, *, label: str
) -> Path:
    """Require a fixed validator source to be tracked by the authorized commit."""
    path = _regular_file(path, label=label)
    try:
        relative = path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} escapes the training repository") from exc
    try:
        result = subprocess.run(
            _git_command(repository_root, "ls-files", "--error-unmatch", "--", relative),
            cwd=repository_root,
            env=_git_environment(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot verify tracked {label}") from exc
    if result.returncode != 0 or result.stdout.strip() != relative:
        raise ValueError(f"{label} is not tracked by the authorized commit")
    return path


def _validate_selected_view_runtime(release_dir: Path, repository_root: Path) -> None:
    """Re-run learned/indexed gates through the fixed production runtime.

    The builder's ``runtime_factory`` is deliberately not accepted here.  A
    separate, source-controlled helper imports only
    ``data_catalog.selected_runtime:build_runtime_v1``.  That registry in turn
    permits one repository-local backend and fails closed when it is absent.
    """
    selected_root = repository_root / "DATA" / "selected-views"
    _inside(release_dir, selected_root, label="selected-view release")
    if release_dir.parent != selected_root:
        raise ValueError("selected-view release must be a direct child of DATA/selected-views")
    if _SELECTED_ID_RE.fullmatch(release_dir.name) is None:
        raise ValueError("selected-view release has an invalid content-addressed ID")

    runtime_script = _require_tracked_source(
        repository_root,
        repository_root / "eval-pipeline" / "revalidate_selected_view_runtime.py",
        label="selected-view runtime revalidator",
    )
    _require_tracked_source(
        repository_root,
        repository_root / "eval-pipeline" / "data_catalog" / "selected_runtime.py",
        label="selected-view runtime registry",
    )
    backend = repository_root / "eval-pipeline" / "data_catalog" / "selected_runtime_backend.py"
    # A missing backend is reported by the fixed registry with a specific
    # fail-closed error.  If a file exists, it must also be source-controlled;
    # ignored or injected modules must never be imported by authorization.
    if backend.exists():
        _require_tracked_source(
            repository_root, backend, label="selected-view runtime backend"
        )

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(runtime_script),
                "--release-dir",
                str(release_dir),
                "--repository-root",
                str(repository_root),
            ],
            cwd=repository_root,
            env=_git_environment(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(
            "formal selected-view runtime revalidation could not be completed"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[-1] if detail else "runtime revalidator returned a non-zero status"
        raise ValueError(f"formal selected-view runtime revalidation failed: {reason}")


def _receipt_file(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    return _regular_file(Path(value), label=label)


def _release_file(
    release_dir: Path,
    descriptor: object,
    *,
    label: str,
    expected_keys: set[str],
) -> Path:
    if not isinstance(descriptor, Mapping) or set(descriptor) != expected_keys:
        raise ValueError(f"{label} descriptor is incomplete")
    name = descriptor.get("path")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"{label} must be one release-local filename")
    path = _regular_file(release_dir / name, label=label)
    _verify_binding(path, descriptor, label=label)
    return path


def _verify_binding(path: Path, binding: Mapping[str, Any], *, label: str) -> None:
    size = binding.get("bytes")
    digest = binding.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or path.stat().st_size != size
        or _sha256_file(path) != digest
    ):
        raise ValueError(f"{label} differs from its binding")


def _external_binding(binding: object, *, label: str) -> Path:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} file binding is incomplete")
    raw = binding.get("path")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = _regular_file(Path(raw), label=label)
    _verify_binding(path, binding, label=label)
    return path


def _collect_external_assets(config: Mapping[str, Any]) -> tuple[tuple[str, Path], ...]:
    found: dict[Path, str] = {}

    def visit(value: object, label: str) -> None:
        if isinstance(value, Mapping):
            if set(value) == {"path", "bytes", "sha256"}:
                path = _external_binding(value, label=label)
                found.setdefault(path, label)
                return
            for key, item in value.items():
                visit(item, f"{label}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{label}[{index}]")

    visit(config, "selected-view config")
    if not found:
        raise ValueError("selected-view config binds no external assets")
    return tuple((f"external asset {label}", path) for path, label in found.items())


def _parent_artifact_paths(
    parent: Mapping[str, Any],
    child_binding: object,
    *,
    parent_dir: Path,
) -> tuple[tuple[str, Path], ...]:
    parent_artifacts = parent.get("artifacts")
    if not isinstance(parent_artifacts, list) or not isinstance(child_binding, list):
        raise ValueError("candidate parent artifact inventory is missing")
    parent_paths = tuple(
        item.get("path") for item in parent_artifacts if isinstance(item, Mapping)
    )
    child_paths = tuple(item.get("path") for item in child_binding if isinstance(item, Mapping))
    if (
        len(parent_artifacts) != 15
        or len(child_binding) != 15
        or parent_paths != _PARENT_ARTIFACT_ORDER
        or child_paths != _PARENT_ARTIFACT_ORDER
    ):
        raise ValueError("authorization requires exactly the 15 frozen parent artifacts")
    projected = [
        {"path": item.get("path"), "bytes": item.get("bytes"), "sha256": item.get("sha256")}
        for item in parent_artifacts
        if isinstance(item, Mapping)
    ]
    if child_binding != projected:
        raise ValueError("candidate parent artifacts differ from the selected child binding")
    paths = []
    for binding in child_binding:
        if not isinstance(binding, Mapping) or set(binding) != {"path", "bytes", "sha256"}:
            raise ValueError("candidate parent artifact binding is incomplete")
        name = binding.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("candidate parent artifact path is invalid")
        path = _regular_file(parent_dir / name, label=f"candidate parent artifact {name}")
        _verify_binding(path, binding, label=f"candidate parent artifact {name}")
        paths.append((f"candidate parent artifact {name}", path))
    return tuple(paths)


def _selected_image_paths(payload_path: Path) -> tuple[tuple[str, Path], ...]:
    rows = []
    try:
        with payload_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError("selected payload contains a blank row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"selected payload row {line_number} is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("selected payload is unreadable") from exc
    if len(rows) != 96:
        raise ValueError(f"training authorization requires exactly 96 selected rows, got {len(rows)}")
    splits = {"train": 0, "validation": 0}
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    result = []
    for row in rows:
        row_id = row.get("id")
        split = row.get("selected_split")
        if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
            raise ValueError("selected payload IDs must be unique non-empty strings")
        if split not in splits:
            raise ValueError(f"selected row {row_id} has an invalid split")
        splits[split] += 1
        raw = row.get("image_path")
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise ValueError(f"selected row {row_id} image path must be absolute")
        image = _regular_file(Path(raw), label=f"selected raw image {row_id}")
        if image in seen_paths:
            raise ValueError("selected raw image paths must be unique")
        expected = row.get("raw_file_sha256")
        if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
            raise ValueError(f"selected row {row_id} raw image hash is invalid")
        if _sha256_file(image) != expected:
            raise ValueError(f"selected raw image {row_id} differs from its row binding")
        result.append((f"selected raw image {row_id}", image))
        seen_ids.add(row_id)
        seen_paths.add(image)
    if splits != {"train": 64, "validation": 32}:
        raise ValueError("training authorization requires exactly 64 train and 32 validation rows")
    return tuple(result)


def _file_snapshot(label: str, path: Path) -> tuple[str, str, int, int, int, int, int, str]:
    path = _regular_file(path, label=label)
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    def fields(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    if fields(before) != fields(after):
        raise RuntimeError(f"{label} changed while it was snapshotted")
    return (label, str(path), *fields(after), digest)


@dataclass(frozen=True)
class TrainingAuthorization:
    """Validated immutable authorization used by :class:`RendererTrainer`."""

    receipt_path: Path
    selected_view_manifest_path: Path
    candidate_parent_manifest_path: Path
    selected_config_path: Path
    selected_payload_path: Path
    selected_view_manifest_sha256: str
    candidate_parent_manifest_sha256: str
    selected_config_sha256: str
    selected_payload_sha256: str
    selected_view_release_id: str
    candidate_parent_release_id: str
    selected_view_id: str
    selected_rows: int
    code_commit: str
    repository_root: Path
    _bound_files: tuple[tuple[str, str, int, int, int, int, int, str], ...] = ()
    _receipt_sha256: str = ""
    _seal: object = field(default=None, repr=False, compare=False)

    def is_validated(self) -> bool:
        """Return whether this exact object came from a successful ``load``."""
        return _is_registered(self)

    @property
    def catalog_manifest_path(self) -> Path:
        """Compatibility alias for the candidate-parent manifest."""
        return self.candidate_parent_manifest_path

    @property
    def selected_manifest_path(self) -> Path:
        """Compatibility alias for the selected 96-row payload."""
        return self.selected_payload_path

    @property
    def catalog_manifest_sha256(self) -> str:
        """Compatibility alias for the candidate-parent manifest hash."""
        return self.candidate_parent_manifest_sha256

    @property
    def selected_manifest_sha256(self) -> str:
        """Compatibility alias for the selected 96-row payload hash."""
        return self.selected_payload_sha256

    @property
    def catalog_release_id(self) -> str:
        """Compatibility alias for the candidate-parent release ID."""
        return self.candidate_parent_release_id

    def bind_run_contract(
        self,
        contract: Mapping[str, Any] | Any,
    ) -> "AuthorizationBinding":
        """Bind one complete run contract to this validated child release."""
        if type(self) is not TrainingAuthorization or not self.is_validated():
            raise TypeError("run binding requires a validated TrainingAuthorization")
        from .run_contract import TrainingRunContract

        normalized = TrainingRunContract.from_mapping(contract, require_complete=True)
        self.validate_current()
        payload = normalized.payload
        expected = {
            "code_commit": self.code_commit,
            "catalog_release_id": self.candidate_parent_release_id,
            "catalog_manifest_sha256": self.candidate_parent_manifest_sha256,
            "selected_view_release_id": self.selected_view_release_id,
            "selected_view_manifest_sha256": self.selected_view_manifest_sha256,
            "selected_view_config_sha256": self.selected_config_sha256,
            "selected_payload_manifest_sha256": self.selected_payload_sha256,
            "selected_view_id": self.selected_view_id,
            "selected_rows": self.selected_rows,
        }
        for field_name, expected_value in expected.items():
            if payload.get(field_name) != expected_value:
                raise ValueError(
                    f"run contract {field_name} differs from authorization"
                )
        from .artifacts import verify_run_artifacts

        verified_artifacts = verify_run_artifacts(
            payload,
            contract_sha256=normalized.sha256,
            selected_manifest_path=self.selected_payload_path,
            selected_manifest_sha256=self.selected_payload_sha256,
        )
        result = AuthorizationBinding(
            authorization=self,
            run_contract=normalized,
            run_artifacts=verified_artifacts,
            _seal=object(),
        )
        _register_binding(result)
        return result

    @classmethod
    def load(
        cls,
        receipt_path: str | os.PathLike[str],
        *,
        repository_root: str | os.PathLike[str] | None,
    ) -> "TrainingAuthorization":
        repo = _require_repository_root(repository_root)
        receipt = _regular_file(Path(receipt_path).absolute(), label="training receipt")
        payload = _json_object(receipt, label="training authorization receipt")
        if payload.get("schema") != AUTHORIZATION_SCHEMA:
            raise ValueError("unsupported training authorization schema; v2 child release required")
        expected = {
            "schema",
            "selected_view_manifest",
            "candidate_parent_manifest",
            "selected_view_id",
            "selected_rows",
            "selected_splits",
            "code_commit",
        }
        if set(payload) != expected:
            raise ValueError("training authorization receipt fields differ")

        child_info = payload.get("selected_view_manifest")
        parent_info = payload.get("candidate_parent_manifest")
        if not isinstance(child_info, Mapping) or set(child_info) != {
            "path",
            "release_id",
            "sha256",
            "complete",
            "training_ready",
            "git_commit",
        }:
            raise ValueError("selected-view receipt binding is incomplete")
        if not isinstance(parent_info, Mapping) or set(parent_info) != {
            "path",
            "release_id",
            "sha256",
            "training_ready",
        }:
            raise ValueError("candidate parent receipt binding is incomplete")
        child_id = child_info.get("release_id")
        parent_id = parent_info.get("release_id")
        child_hash = child_info.get("sha256")
        parent_hash = parent_info.get("sha256")
        code_commit = payload.get("code_commit")
        selected_view_id = payload.get("selected_view_id")
        if (
            not isinstance(child_id, str)
            or _SELECTED_ID_RE.fullmatch(child_id) is None
            or not isinstance(parent_id, str)
            or _CATALOG_ID_RE.fullmatch(parent_id) is None
            or not isinstance(child_hash, str)
            or _SHA256_RE.fullmatch(child_hash) is None
            or not isinstance(parent_hash, str)
            or _SHA256_RE.fullmatch(parent_hash) is None
            or not isinstance(code_commit, str)
            or _COMMIT_RE.fullmatch(code_commit) is None
            or not isinstance(selected_view_id, str)
            or not selected_view_id
            or child_info.get("complete") is not True
            or child_info.get("training_ready") is not True
            or child_info.get("git_commit") != code_commit
            or payload.get("selected_rows") != 96
            or payload.get("selected_splits") != {"train": 64, "validation": 32}
        ):
            raise ValueError("training authorization values are invalid")
        if parent_info.get("training_ready") is not False:
            raise ValueError("candidate parent must remain non-training-ready")

        # Reject dirty or unpushed code before executing the source-controlled
        # child validator or scanning any large release artifact.
        current_commit, clean = _git_state(repo)
        if current_commit != code_commit or not clean:
            raise ValueError("training repository differs from the authorized commit")
        _git_upstream(repo, expected_commit=code_commit)

        child_path = _receipt_file(child_info.get("path"), label="selected-view manifest")
        child_dir = child_path.parent
        if child_path.name != "manifest.json" or child_dir.name != child_id:
            raise ValueError("selected-view manifest path does not match its release ID")
        if _sha256_file(child_path) != child_hash:
            raise ValueError("selected-view manifest hash does not match the receipt")
        _validate_selected_view_release(child_dir, repo)

        child = _json_object(child_path, label="selected-view manifest")
        if (
            child.get("schema") != SELECTED_VIEW_SCHEMA
            or child.get("release_id") != child_id
            or child.get("complete") is not True
            or child.get("training_ready") is not True
            or child.get("development_build") is not False
        ):
            raise ValueError("selected child is not a complete formal training release")
        git = child.get("git")
        if (
            not isinstance(git, Mapping)
            or git.get("commit") != code_commit
            or git.get("dirty") is not False
            or git.get("pushed") is not True
            or git.get("upstream_commit") != code_commit
        ):
            raise ValueError("selected child Git provenance is not clean and pushed")

        child_parent = child.get("parent_catalog")
        if not isinstance(child_parent, Mapping) or set(child_parent) != {
            "path",
            "release_id",
            "manifest_sha256",
            "artifacts",
        }:
            raise ValueError("selected child candidate-parent binding is incomplete")
        parent_path = _receipt_file(parent_info.get("path"), label="candidate parent manifest")
        if (
            parent_path.name != "manifest.json"
            or parent_path.parent.name != parent_id
            or child_parent.get("path") != str(parent_path)
            or child_parent.get("release_id") != parent_id
            or child_parent.get("manifest_sha256") != parent_hash
            or _sha256_file(parent_path) != parent_hash
        ):
            raise ValueError("candidate parent manifest differs from the selected child binding")
        parent = _json_object(parent_path, label="candidate parent manifest")
        if (
            parent.get("schema") != "repldm.data_catalog.v1"
            or parent.get("release_id") != parent_id
            or parent.get("candidate_catalog_complete") is not True
            or parent.get("development_build") is not False
            or parent.get("verify_paths") is not True
        ):
            raise ValueError("candidate parent is not a complete formal inventory")
        if parent.get("complete") is not False or parent.get("training_ready") is not False:
            raise ValueError("candidate parent must remain non-training-ready")

        config_path = _release_file(
            child_dir,
            child.get("config"),
            label="selected-view config",
            expected_keys={"path", "bytes", "sha256", "schema"},
        )
        selected_descriptor = child.get("selected_payload")
        payload_path = _release_file(
            child_dir,
            selected_descriptor,
            label="selected payload",
            expected_keys={"path", "bytes", "sha256", "schema", "rows", "splits"},
        )
        if (
            child["config"].get("schema") != SELECTED_CONFIG_SCHEMA
            or not isinstance(selected_descriptor, Mapping)
            or selected_descriptor.get("schema") != SELECTED_ROW_SCHEMA
            or selected_descriptor.get("rows") != 96
            or selected_descriptor.get("splits") != {"train": 64, "validation": 32}
        ):
            raise ValueError("selected child descriptors are not the frozen 64+32 contract")
        config = _json_object(config_path, label="selected-view config")
        if config.get("view_id") != selected_view_id:
            raise ValueError("selected-view ID differs from the frozen config")

        parent_artifacts = _parent_artifact_paths(
            parent,
            child_parent.get("artifacts"),
            parent_dir=parent_path.parent,
        )
        external_assets = _collect_external_assets(config)
        selected_images = _selected_image_paths(payload_path)

        # Static evidence is not sufficient for a formal authorization: the
        # learned and indexed gates must be recomputed by the fixed registry.
        _validate_selected_view_runtime(child_dir, repo)

        current_commit, clean = _git_state(repo)
        if current_commit != code_commit or not clean:
            raise ValueError("training repository differs from the authorized commit")
        _git_upstream(repo, expected_commit=code_commit)

        bound_paths = (
            ("selected-view manifest", child_path),
            ("candidate parent manifest", parent_path),
            ("selected-view config", config_path),
            ("selected payload", payload_path),
            *parent_artifacts,
            *external_assets,
            *selected_images,
        )
        snapshots = tuple(_file_snapshot(label, path) for label, path in bound_paths)
        result = cls(
            receipt_path=receipt,
            selected_view_manifest_path=child_path,
            candidate_parent_manifest_path=parent_path,
            selected_config_path=config_path,
            selected_payload_path=payload_path,
            selected_view_manifest_sha256=child_hash,
            candidate_parent_manifest_sha256=parent_hash,
            selected_config_sha256=_sha256_file(config_path),
            selected_payload_sha256=_sha256_file(payload_path),
            selected_view_release_id=child_id,
            candidate_parent_release_id=parent_id,
            selected_view_id=selected_view_id,
            selected_rows=96,
            code_commit=code_commit,
            repository_root=repo,
            _bound_files=snapshots,
            _receipt_sha256=_sha256_file(receipt),
            _seal=object(),
        )
        _register_validated(result)
        return result

    def validate_current(self) -> None:
        """Reject receipt, release dependency, payload, or Git drift."""
        if not _is_registered(self):
            raise RuntimeError("training authorization is not a validated loader object")
        if _sha256_file(self.receipt_path) != self._receipt_sha256:
            raise RuntimeError("training authorization receipt changed during training")
        for expected in self._bound_files:
            label = expected[0]
            try:
                observed = _file_snapshot(label, Path(expected[1]))
            except (OSError, ValueError, RuntimeError) as exc:
                raise RuntimeError(f"authorized {label} changed during training") from exc
            if observed != expected:
                raise RuntimeError(f"authorized {label} changed during training")
        current_commit, clean = _git_state(self.repository_root)
        if current_commit != self.code_commit or not clean:
            raise RuntimeError("training repository changed after authorization")
        try:
            _git_upstream(self.repository_root, expected_commit=self.code_commit)
        except ValueError as exc:
            raise RuntimeError("training repository is no longer at the pushed commit") from exc


@dataclass(frozen=True)
class AuthorizationBinding:
    """Identity-checked capability for one complete selected-view training run."""

    authorization: TrainingAuthorization
    run_contract: Any
    run_artifacts: Any = None
    _seal: object = field(default=None, repr=False, compare=False)

    @property
    def contract_hash(self) -> str:
        return self.run_contract.sha256

    @property
    def contract(self) -> dict[str, Any]:
        return self.run_contract.to_dict()

    def is_validated(self) -> bool:
        return _is_registered_binding(self)

    def validate_current(self, *, component: Any | None = None) -> None:
        """Revalidate the selected release, run artifacts, and renderer contract."""
        if not _is_registered_binding(self):
            raise RuntimeError("training run binding is not a validated loader object")
        authorization = self.authorization
        if (
            type(authorization) is not TrainingAuthorization
            or not authorization.is_validated()
        ):
            raise RuntimeError("training run binding lost its authorization")
        authorization.validate_current()
        from .artifacts import require_verified_run_artifacts

        artifacts = require_verified_run_artifacts(self.run_artifacts)
        if artifacts.contract_sha256 != self.contract_hash:
            raise RuntimeError("verified run artifacts have a different contract")
        artifacts.validate_current()
        from .run_contract import TrainingRunContract

        current = TrainingRunContract.from_mapping(self.contract, require_complete=True)
        if current.sha256 != self.run_contract.sha256:
            raise RuntimeError("training run contract changed after authorization")
        payload = current.payload
        expected = {
            "code_commit": authorization.code_commit,
            "catalog_release_id": authorization.candidate_parent_release_id,
            "catalog_manifest_sha256": authorization.candidate_parent_manifest_sha256,
            "selected_view_release_id": authorization.selected_view_release_id,
            "selected_view_manifest_sha256": authorization.selected_view_manifest_sha256,
            "selected_view_config_sha256": authorization.selected_config_sha256,
            "selected_payload_manifest_sha256": authorization.selected_payload_sha256,
            "selected_view_id": authorization.selected_view_id,
            "selected_rows": authorization.selected_rows,
        }
        if any(payload.get(field) != value for field, value in expected.items()):
            raise RuntimeError("training run contract no longer matches its authorization")
        if component is not None:
            self.validate_component(component)

    def validate_initial_renderer(self, component: Any) -> None:
        self.validate_current(component=component)
        from .artifacts import require_verified_run_artifacts

        require_verified_run_artifacts(self.run_artifacts).validate_initial_renderer(
            component
        )

    def validate_component(self, component: Any) -> None:
        if not _is_registered_binding(self):
            raise RuntimeError("training run binding is not validated")
        payload = self.contract
        frame_hash = getattr(component, "frame_contract_hash", None)
        calibration_hash = getattr(component, "calibration_hash", None)
        action_contract = getattr(component, "contract", None)
        if not isinstance(frame_hash, str):
            raise TypeError("bound component must expose renderer frame contract hash")
        if not isinstance(calibration_hash, str):
            raise TypeError("bound component must expose calibration hash")
        if action_contract is None:
            raise TypeError("bound component must expose its action contract")
        if payload.get("renderer_frame_contract_hash") != frame_hash:
            raise ValueError("renderer frame contract hash differs from the run contract")
        if payload.get("calibration_hash") != calibration_hash:
            raise ValueError("renderer calibration hash differs from the run contract")
        from .contracts import contract_hash

        if payload.get("action_contract_hash") != contract_hash(action_contract):
            raise ValueError("renderer action contract hash differs from the run contract")

    def provenance(self) -> dict[str, Any]:
        payload = self.contract
        authorization = self.authorization
        from .artifacts import require_verified_run_artifacts

        artifacts = require_verified_run_artifacts(self.run_artifacts)
        return {
            "schema": "repldm.training_authorization_binding.v2",
            "authorization_receipt_sha256": authorization._receipt_sha256,
            "code_commit": authorization.code_commit,
            "candidate_parent_release_id": authorization.candidate_parent_release_id,
            "candidate_parent_manifest_sha256": authorization.candidate_parent_manifest_sha256,
            "selected_view_release_id": authorization.selected_view_release_id,
            "selected_view_manifest_sha256": authorization.selected_view_manifest_sha256,
            "selected_view_config_sha256": authorization.selected_config_sha256,
            "selected_payload_sha256": authorization.selected_payload_sha256,
            "selected_view_id": authorization.selected_view_id,
            "run_contract_sha256": self.contract_hash,
            "renderer_frame_contract_hash": payload["renderer_frame_contract_hash"],
            "calibration_hash": payload["calibration_hash"],
            "action_contract_hash": payload["action_contract_hash"],
            "data_manifest_sha256": payload["data_manifest_sha256"],
            "reward_config_sha256": payload["reward_config_sha256"],
            "verified_run_artifacts": artifacts.provenance(),
        }


def _register_binding(value: AuthorizationBinding) -> None:
    import weakref

    identity = id(value)

    def remove(reference: Any, *, identity: int = identity) -> None:
        if _VALIDATED_BINDINGS.get(identity) is reference:
            _VALIDATED_BINDINGS.pop(identity, None)

    _VALIDATED_BINDINGS[identity] = weakref.ref(value, remove)


def _is_registered_binding(value: AuthorizationBinding) -> bool:
    reference = _VALIDATED_BINDINGS.get(id(value))
    return reference is not None and reference() is value


def require_authorization_binding(value: Any) -> AuthorizationBinding:
    """Validate a capability without accepting forged subclasses or copies."""
    if type(value) is not AuthorizationBinding or not _is_registered_binding(value):
        raise TypeError("a validated AuthorizationBinding is required")
    from .artifacts import require_verified_run_artifacts

    require_verified_run_artifacts(value.run_artifacts)
    return value


def write_authorization_receipt(
    path: str | os.PathLike[str],
    *,
    selected_view_manifest: str | os.PathLike[str],
    code_commit: str,
    repository_root: str | os.PathLike[str],
) -> TrainingAuthorization:
    """Write and validate a v2 receipt bound to one selected-view child."""
    repo = _require_repository_root(repository_root)
    if not isinstance(code_commit, str) or _COMMIT_RE.fullmatch(code_commit) is None:
        raise ValueError("code_commit must be a full Git commit")
    current_commit, clean = _git_state(repo)
    if current_commit != code_commit or not clean:
        raise ValueError("training repository differs from the authorized commit")
    _git_upstream(repo, expected_commit=code_commit)
    child_path = _regular_file(
        Path(selected_view_manifest).absolute(), label="selected-view manifest"
    )
    child = _json_object(child_path, label="selected-view manifest")
    parent_binding = child.get("parent_catalog")
    if not isinstance(parent_binding, Mapping):
        raise ValueError("selected child candidate-parent binding is incomplete")
    parent_path = _receipt_file(
        parent_binding.get("path"), label="candidate parent manifest"
    )
    parent = _json_object(parent_path, label="candidate parent manifest")
    config_descriptor = child.get("config")
    if not isinstance(config_descriptor, Mapping):
        raise ValueError("selected-view config descriptor is incomplete")
    config_name = config_descriptor.get("path")
    if not isinstance(config_name, str) or Path(config_name).name != config_name:
        raise ValueError("selected-view config must be one release-local filename")
    config = _json_object(child_path.parent / config_name, label="selected-view config")
    receipt_payload = {
        "schema": AUTHORIZATION_SCHEMA,
        "selected_view_manifest": {
            "path": str(child_path),
            "release_id": child.get("release_id"),
            "sha256": _sha256_file(child_path),
            "complete": child.get("complete"),
            "training_ready": child.get("training_ready"),
            "git_commit": (child.get("git") or {}).get("commit"),
        },
        "candidate_parent_manifest": {
            "path": str(parent_path),
            "release_id": parent.get("release_id"),
            "sha256": _sha256_file(parent_path),
            "training_ready": parent.get("training_ready"),
        },
        "selected_view_id": config.get("view_id"),
        "selected_rows": 96,
        "selected_splits": {"train": 64, "validation": 32},
        "code_commit": str(code_commit),
    }
    destination = _absolute_without_symlinks(
        Path(path).absolute(), label="training authorization destination"
    )
    if destination.exists() and destination.is_symlink():
        raise ValueError("training authorization destination must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _absolute_without_symlinks(
        destination.parent, label="training authorization destination parent"
    )
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(receipt_payload, ensure_ascii=True, indent=2, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        _absolute_without_symlinks(
            destination, label="training authorization destination"
        )
        if destination.exists() and destination.is_symlink():
            raise ValueError("training authorization destination must not be a symlink")
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return TrainingAuthorization.load(destination, repository_root=repo)


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "AuthorizationBinding",
    "TrainingAuthorization",
    "require_authorization_binding",
    "write_authorization_receipt",
]
