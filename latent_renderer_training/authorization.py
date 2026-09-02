"""Fail-closed authorization for latent-renderer training data.

The training package deliberately does not import the evaluation package (the
directory name contains a dash).  Instead, the catalog builder emits a small
JSON receipt after it has validated a formal release and a selected-payload
manifest.  This module verifies that receipt and the referenced bytes before a
trainer is allowed to perform an optimizer update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping


AUTHORIZATION_SCHEMA = "repldm.training_authorization.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_ID_RE = re.compile(r"^catalog-[0-9a-f]{20}$")
_PACKAGE_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# WeakSet uses dataclass equality when checking membership.  A copied
# dataclass would therefore look authorized.  Keep an identity registry
# instead: the weak reference callback removes entries after collection.
_VALIDATED_IDENTITIES: dict[int, Any] = {}


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
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"protected file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _absolute_without_symlinks(path: Path, *, label: str) -> Path:
    """Normalize a path lexically and reject symlinks in every component.

    ``Path.resolve`` is deliberately not used here: it would turn a symlink
    into an apparently safe path before we had a chance to reject it.
    """
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    normalized = Path(os.path.abspath(os.fspath(path)))
    current = Path(normalized.anchor)
    for component in normalized.parts[1:]:
        current /= component
        try:
            status = current.lstat()
        except FileNotFoundError:
            # Let the caller report the more useful missing-file error.
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
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be an absolute ordinary file: {path}")
    if status.st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    return path


def _inside(path: Path, root: Path, *, label: str) -> None:
    path = _absolute_without_symlinks(path, label=label)
    root = _absolute_without_symlinks(root, label=f"{label} root")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root") from exc


def _git_state(repository_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot inspect the training repository Git state") from exc
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("current Git HEAD is not a full commit")
    return commit, not bool(status)


def _git_upstream(repository_root: Path, *, expected_commit: str) -> None:
    """Require an existing upstream ref whose tip is the authorized commit."""
    try:
        upstream = subprocess.run(
            ["git", "rev-parse", "@{upstream}"],
            cwd=repository_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("training repository has no reachable upstream commit") from exc
    if upstream != expected_commit:
        raise ValueError("training repository is not at the pushed authorized commit")


def _require_repository_root(value: str | os.PathLike[str] | None) -> Path:
    """Return the one repository whose builder and training package are trusted."""
    if value is None:
        raise ValueError("repository_root is required for training authorization")
    root = _absolute_without_symlinks(Path(value), label="repository root")
    if not root.is_dir():
        raise ValueError("training repository root is not a directory")
    # A receipt must not authorize a different checkout whose builder has not
    # undergone this repository's review and push gate.
    if root != _PACKAGE_REPOSITORY_ROOT:
        raise ValueError("repository_root must be the repository containing this package")
    return root


def _validate_formal_release(release_dir: Path, repository_root: Path) -> None:
    """Run the canonical catalog validator in its own process.

    The catalog package lives below ``eval-pipeline`` (a directory name that
    cannot be imported as a normal Python package).  Invoking its CLI avoids a
    second, potentially divergent implementation of the release contract.
    """
    catalogs_root = repository_root / "DATA" / "catalogs"
    _inside(release_dir, catalogs_root, label="catalog release")
    if release_dir.parent != catalogs_root:
        raise ValueError("training catalog must be a direct child of DATA/catalogs")
    if _RELEASE_ID_RE.fullmatch(release_dir.name) is None:
        raise ValueError("training catalog has an invalid content-addressed release ID")
    builder = _regular_file(
        repository_root / "eval-pipeline" / "build_data_catalog.py",
        label="catalog builder",
    )
    command = [
        sys.executable,
        str(builder),
        "--validate-only",
        "--release-dir",
        str(release_dir),
        "--require-training-ready",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=repository_root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("formal catalog validation could not be completed") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[-1] if detail else "builder returned a non-zero status"
        raise ValueError(f"formal catalog validation failed: {reason}")


def _validate_selected_rows(
    path: Path, *, expected_rows: int, manifest_sha256: str
) -> tuple[dict[str, Any], ...]:
    actual_hash = _sha256_file(path)
    if actual_hash != manifest_sha256:
        raise ValueError("selected payload manifest hash does not match the receipt")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    paths: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"selected payload manifest has invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError("selected payload rows must be objects")
            required = {
                "id",
                "image_path",
                "payload_sha256",
                "training_eligible",
                "benchmark_exact_match",
            }
            if not required.issubset(row):
                raise ValueError("selected payload row is missing required fields")
            row_id = row["id"]
            if not isinstance(row_id, str) or not row_id or row_id in ids:
                raise ValueError("selected payload IDs must be unique non-empty strings")
            image_path_raw = row["image_path"]
            if not isinstance(image_path_raw, str) or not image_path_raw:
                raise ValueError("selected payload image_path must be a string")
            image_path = _regular_file(Path(image_path_raw), label="selected image")
            image_key = str(image_path)
            if image_key in paths:
                raise ValueError("selected payload image paths must be unique")
            payload_hash = row["payload_sha256"]
            if not isinstance(payload_hash, str) or _SHA256_RE.fullmatch(payload_hash) is None:
                raise ValueError("selected payload hash is invalid")
            if _sha256_file(image_path) != payload_hash:
                raise ValueError(f"selected image hash does not match row {row_id}")
            if row["training_eligible"] is not True:
                raise ValueError(f"selected row {row_id} is not training eligible")
            if row["benchmark_exact_match"] != []:
                raise ValueError(f"selected row {row_id} matches a protected benchmark")
            rows.append(dict(row))
            ids.add(row_id)
            paths.add(image_key)
    if not rows or len(rows) != expected_rows:
        raise ValueError(
            f"selected payload row count differs: expected {expected_rows}, got {len(rows)}"
        )
    return tuple(rows)


def _catalog_artifact(
    catalog: Mapping[str, Any], release_dir: Path, name: str
) -> Path:
    """Resolve and hash one catalog artifact referenced by its formal manifest."""
    release_dir = _absolute_without_symlinks(release_dir, label="catalog release")
    if not release_dir.is_dir():
        raise ValueError("catalog release is not a directory")
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("catalog artifact name must be a single filename")
    artifacts = catalog.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("catalog artifact inventory is missing")
    matches = [row for row in artifacts if isinstance(row, Mapping) and row.get("path") == name]
    if len(matches) != 1:
        raise ValueError(f"catalog artifact {name!r} is not uniquely declared")
    record = matches[0]
    path = _regular_file(release_dir / name, label=f"catalog artifact {name}")
    expected_size = record.get("bytes")
    expected_hash = record.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or not isinstance(expected_hash, str)
        or _SHA256_RE.fullmatch(expected_hash) is None
        or path.stat().st_size != expected_size
        or _sha256_file(path) != expected_hash
    ):
        raise ValueError(f"catalog artifact {name!r} does not match its manifest")
    return path


def _bind_selected_rows_to_catalog(
    catalog: Mapping[str, Any],
    release_dir: Path,
    *,
    selected_view_id: str,
    selected_rows: tuple[dict[str, Any], ...],
) -> None:
    """Prove that selected rows are an exact, declared view of the catalog."""
    candidate_path = _catalog_artifact(catalog, release_dir, "training_candidates.jsonl")
    views_path = _catalog_artifact(catalog, release_dir, "training_views.jsonl")
    view_records = []
    view_ids: set[str] = set()
    with views_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("training view records must be objects")
            view_id = row.get("id")
            if not isinstance(view_id, str) or not view_id or view_id in view_ids:
                raise ValueError("training view IDs must be unique non-empty strings")
            view_ids.add(view_id)
            view_records.append(row)
    views = [row for row in view_records if row.get("id") == selected_view_id]
    if len(views) != 1:
        raise ValueError("selected_view_id is not declared exactly once in the catalog")
    view = views[0]
    if (
        view.get("schema") != "repldm.training_view.v1"
        or view.get("artifact") != "training_candidates.jsonl"
        or view.get("expected_rows") != len(selected_rows)
    ):
        raise ValueError("selected view contract does not match selected payload rows")
    filters = view.get("filter")
    if not isinstance(filters, Mapping) or len(filters) != 1:
        raise ValueError("selected view filter is invalid")
    selected_by_id = {row["id"]: row for row in selected_rows}
    matched: dict[str, Mapping[str, Any]] = {}
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("training candidate records must be objects")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError("training candidate IDs must be non-empty strings")
            if row_id in matched:
                raise ValueError("training candidate IDs are not unique")
            matched[row_id] = row
    if set(matched) != set(selected_by_id):
        if not set(selected_by_id).issubset(matched):
            raise ValueError("selected payload contains an ID outside training_candidates")
        # Extra candidates are allowed, but every selected ID must be bound to
        # exactly one canonical candidate row.
    for row_id, selected in selected_by_id.items():
        candidate = matched[row_id]
        if set(selected) < set(candidate):
            raise ValueError(f"selected row {row_id} omits catalog fields")
        # Selection metadata may add fields, but it may never rewrite any
        # field emitted by the catalog (especially prompt, license, source,
        # split, or leakage results).
        for key, value in candidate.items():
            if selected.get(key) != value:
                raise ValueError(f"selected row {row_id} differs from catalog candidate")
        if "prompt_is_not_null" in filters:
            expected = filters["prompt_is_not_null"]
            if not isinstance(expected, bool) or bool(candidate.get("prompt")) is not expected:
                raise ValueError(f"selected row {row_id} does not satisfy its view filter")
        elif "modality_in" in filters:
            allowed = filters["modality_in"]
            if (
                not isinstance(allowed, list)
                or any(not isinstance(value, str) for value in allowed)
                or candidate.get("modality") not in allowed
            ):
                raise ValueError(f"selected row {row_id} does not satisfy its view filter")
        else:
            raise ValueError("selected view filter is unsupported")


@dataclass(frozen=True)
class TrainingAuthorization:
    """Validated immutable authorization used by :class:`RendererTrainer`."""

    receipt_path: Path
    catalog_manifest_path: Path
    selected_manifest_path: Path
    catalog_manifest_sha256: str
    selected_manifest_sha256: str
    catalog_release_id: str
    selected_view_id: str
    selected_rows: int
    code_commit: str
    repository_root: Path
    _payload_identities: tuple[tuple[str, int, int, int, int], ...] = ()
    _catalog_artifacts: tuple[tuple[str, int, int, int, int, str], ...] = ()
    _receipt_sha256: str = ""
    _seal: object = field(default=None, repr=False, compare=False)

    def is_validated(self) -> bool:
        """Return whether this exact object came from a successful ``load``."""
        return _is_registered(self)

    @classmethod
    def load(
        cls,
        receipt_path: str | os.PathLike[str],
        *,
        repository_root: str | os.PathLike[str] | None,
    ) -> "TrainingAuthorization":
        repo = _require_repository_root(repository_root)
        receipt = _regular_file(Path(receipt_path).absolute(), label="training receipt")
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("training authorization receipt is unreadable") from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != AUTHORIZATION_SCHEMA:
            raise ValueError("unsupported training authorization schema")
        expected_keys = {
            "schema",
            "catalog_manifest",
            "selected_payload_manifest",
            "selected_view_id",
            "selected_rows",
            "code_commit",
        }
        if set(payload) != expected_keys:
            raise ValueError("training authorization receipt fields differ")

        catalog_info = payload["catalog_manifest"]
        selected_info = payload["selected_payload_manifest"]
        if not isinstance(catalog_info, Mapping) or set(catalog_info) != {
            "path",
            "release_id",
            "sha256",
            "complete",
            "training_ready",
            "git_commit",
        }:
            raise ValueError("training catalog receipt is incomplete")
        if not isinstance(selected_info, Mapping) or set(selected_info) != {
            "path",
            "sha256",
        }:
            raise ValueError("selected payload receipt is incomplete")
        release_id = catalog_info.get("release_id")
        catalog_hash = catalog_info.get("sha256")
        selected_hash = selected_info.get("sha256")
        code_commit = payload.get("code_commit")
        selected_view_id = payload.get("selected_view_id")
        selected_rows = payload.get("selected_rows")
        if (
            not isinstance(release_id, str)
            or not release_id
            or _RELEASE_ID_RE.fullmatch(release_id) is None
            or _SHA256_RE.fullmatch(str(catalog_hash)) is None
            or _SHA256_RE.fullmatch(str(selected_hash)) is None
            or not isinstance(code_commit, str)
            or _COMMIT_RE.fullmatch(code_commit) is None
            or not isinstance(catalog_info.get("git_commit"), str)
            or catalog_info["git_commit"] != code_commit
            or not isinstance(selected_view_id, str)
            or not selected_view_id
            or isinstance(selected_rows, bool)
            or not isinstance(selected_rows, int)
            or selected_rows <= 0
            or catalog_info.get("complete") is not True
            or catalog_info.get("training_ready") is not True
        ):
            raise ValueError("training authorization values are invalid")

        catalog_path = _regular_file(
            Path(str(catalog_info["path"])).absolute(), label="catalog manifest"
        )
        selected_path = _regular_file(
            Path(str(selected_info["path"])).absolute(),
            label="selected payload manifest",
        )
        release_dir = catalog_path.parent
        if release_dir.name != release_id:
            raise ValueError("catalog release directory does not match its ID")
        _validate_formal_release(release_dir, repo)
        _inside(selected_path, release_dir, label="selected payload manifest")
        if _sha256_file(catalog_path) != catalog_hash:
            raise ValueError("catalog manifest hash does not match the receipt")
        try:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("catalog manifest is unreadable") from exc
        if not isinstance(catalog, Mapping):
            raise ValueError("catalog manifest is not an object")
        if (
            catalog.get("schema") != "repldm.data_catalog.v1"
            or catalog.get("release_id") != release_id
            or catalog.get("complete") is not True
            or catalog.get("training_ready") is not True
            or catalog.get("candidate_catalog_complete") is not True
            or catalog.get("development_build") is not False
            or catalog.get("verify_paths") is not True
        ):
            raise ValueError("catalog is not a complete formal training release")
        policy = catalog.get("payload_integrity_policy")
        if not isinstance(policy, Mapping) or policy.get("training_ready") is not True:
            raise ValueError("catalog payload-integrity policy is not training-ready")
        git = catalog.get("git")
        if (
            not isinstance(git, Mapping)
            or git.get("commit") != code_commit
            or git.get("dirty") is not False
            or git.get("pushed") is not True
        ):
            raise ValueError("catalog Git provenance is not clean and pushed")
        current_commit, clean = _git_state(repo)
        if current_commit != code_commit or not clean:
            raise ValueError("training repository differs from the authorized commit")
        _git_upstream(repo, expected_commit=code_commit)
        selected_records = _validate_selected_rows(
            selected_path,
            expected_rows=selected_rows,
            manifest_sha256=str(selected_hash),
        )
        _bind_selected_rows_to_catalog(
            catalog,
            release_dir,
            selected_view_id=selected_view_id,
            selected_rows=selected_records,
        )

        identities = []
        with selected_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                image = Path(row["image_path"])
                status = image.stat()
                identities.append(
                    (str(image), int(status.st_dev), int(status.st_ino),
                     int(status.st_size), int(status.st_mtime_ns))
                )
        artifact_snapshots = []
        artifacts = catalog.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("catalog artifact inventory is missing")
        seen_artifacts: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
                raise ValueError("catalog artifact inventory is invalid")
            name = artifact["path"]
            if name in seen_artifacts:
                raise ValueError("catalog artifact inventory contains duplicate paths")
            seen_artifacts.add(name)
            artifact_path = _catalog_artifact(catalog, release_dir, name)
            status = artifact_path.stat()
            artifact_snapshots.append(
                (
                    name,
                    int(status.st_dev),
                    int(status.st_ino),
                    int(status.st_size),
                    int(status.st_mtime_ns),
                    _sha256_file(artifact_path),
                )
            )
        result = cls(
            receipt_path=receipt,
            catalog_manifest_path=catalog_path,
            selected_manifest_path=selected_path,
            catalog_manifest_sha256=str(catalog_hash),
            selected_manifest_sha256=str(selected_hash),
            catalog_release_id=release_id,
            selected_view_id=selected_view_id,
            selected_rows=selected_rows,
            code_commit=code_commit,
            repository_root=repo,
            _payload_identities=tuple(identities),
            _catalog_artifacts=tuple(artifact_snapshots),
            _receipt_sha256=_sha256_file(receipt),
            _seal=object(),
        )
        _register_validated(result)
        return result

    def validate_current(self) -> None:
        """Reject catalog, selected manifest, payload, or Git drift."""
        if not _is_registered(self):
            raise RuntimeError("training authorization is not a validated loader object")
        if _sha256_file(self.receipt_path) != self._receipt_sha256:
            raise RuntimeError("training authorization receipt changed during training")
        if _sha256_file(self.catalog_manifest_path) != self.catalog_manifest_sha256:
            raise RuntimeError("authorized catalog manifest changed during training")
        if _sha256_file(self.selected_manifest_path) != self.selected_manifest_sha256:
            raise RuntimeError("authorized selected payload manifest changed during training")
        # Re-check every artifact, not just the two files needed to resolve the
        # selected view.  The release ID binds the complete inventory, so a
        # mutation of an unrelated source shard must also stop training.
        try:
            catalog = json.loads(
                self.catalog_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("authorized catalog manifest became unreadable") from exc
        if not isinstance(catalog, Mapping):
            raise RuntimeError("authorized catalog manifest is not an object")
        snapshots = []
        artifacts = catalog.get("artifacts")
        if not isinstance(artifacts, list):
            raise RuntimeError("authorized catalog artifact inventory is missing")
        try:
            for artifact in artifacts:
                if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
                    raise ValueError("artifact inventory is invalid")
                name = artifact["path"]
                artifact_path = _catalog_artifact(catalog, self.catalog_manifest_path.parent, name)
                status = artifact_path.stat()
                snapshots.append(
                    (
                        name,
                        int(status.st_dev),
                        int(status.st_ino),
                        int(status.st_size),
                        int(status.st_mtime_ns),
                        _sha256_file(artifact_path),
                    )
                )
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError("authorized catalog artifact changed during training") from exc
        if tuple(snapshots) != self._catalog_artifacts:
            raise RuntimeError("authorized catalog artifact changed during training")
        try:
            selected_records = _validate_selected_rows(
                self.selected_manifest_path,
                expected_rows=self.selected_rows,
                manifest_sha256=self.selected_manifest_sha256,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            detail = str(exc)
            if "selected image hash does not match" in detail or "protected file changed" in detail:
                raise RuntimeError(
                    "authorized selected payload bytes changed during training"
                ) from exc
            raise RuntimeError("authorized selected view changed during training") from exc
        try:
            _bind_selected_rows_to_catalog(
                catalog,
                self.catalog_manifest_path.parent,
                selected_view_id=self.selected_view_id,
                selected_rows=selected_records,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("authorized selected view changed during training") from exc
        current_identities = []
        with self.selected_manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                status = Path(row["image_path"]).stat()
                current_identities.append(
                    (str(Path(row["image_path"])), int(status.st_dev), int(status.st_ino),
                     int(status.st_size), int(status.st_mtime_ns))
                )
                if _sha256_file(Path(row["image_path"])) != row["payload_sha256"]:
                    raise RuntimeError("authorized selected payload bytes changed during training")
        if tuple(current_identities) != self._payload_identities:
            raise RuntimeError("authorized selected payload changed during training")
        current_commit, clean = _git_state(self.repository_root)
        if current_commit != self.code_commit or not clean:
            raise RuntimeError("training repository changed after authorization")
        try:
            _git_upstream(self.repository_root, expected_commit=self.code_commit)
        except ValueError as exc:
            raise RuntimeError("training repository is no longer at the pushed commit") from exc


def write_authorization_receipt(
    path: str | os.PathLike[str],
    *,
    catalog_manifest: str | os.PathLike[str],
    selected_payload_manifest: str | os.PathLike[str],
    selected_view_id: str,
    code_commit: str,
    repository_root: str | os.PathLike[str],
) -> TrainingAuthorization:
    """Create and validate one receipt after a formal catalog check."""
    catalog = _regular_file(
        Path(catalog_manifest).absolute(), label="catalog manifest"
    )
    selected = _regular_file(
        Path(selected_payload_manifest).absolute(), label="selected payload manifest"
    )
    catalog_payload = json.loads(catalog.read_text(encoding="utf-8"))
    if not isinstance(catalog_payload, Mapping):
        raise ValueError("catalog manifest is not an object")
    rows = sum(1 for line in selected.read_text(encoding="utf-8").splitlines() if line.strip())
    receipt_payload = {
        "schema": AUTHORIZATION_SCHEMA,
        "catalog_manifest": {
            "path": str(catalog),
            "release_id": catalog_payload.get("release_id"),
            "sha256": _sha256_file(catalog),
            "complete": catalog_payload.get("complete"),
            "training_ready": catalog_payload.get("training_ready"),
            "git_commit": (catalog_payload.get("git") or {}).get("commit"),
        },
        "selected_payload_manifest": {
            "path": str(selected),
            "sha256": _sha256_file(selected),
        },
        "selected_view_id": str(selected_view_id),
        "selected_rows": rows,
        "code_commit": str(code_commit),
    }
    destination = Path(path).absolute()
    if destination.exists() and destination.is_symlink():
        raise ValueError("training authorization destination must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(receipt_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return TrainingAuthorization.load(destination, repository_root=repository_root)


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "TrainingAuthorization",
    "write_authorization_receipt",
]
