"""Exact local SDXL files loaded by the adaptive-oracle engineering smoke."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Mapping, Optional


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
MODEL_CACHE = "pretrained_ckpts"
MODEL_SNAPSHOT_SCHEMA = "adaptive_oracle_model_snapshot_manifest_v1"
MODEL_SNAPSHOT_HASH_ALGORITHM = "sha256-canonical-loaded-file-manifest-v1"
MODEL_STAGE_SCHEMA = "adaptive_oracle_model_stage_v2"
MODEL_STAGE_VERIFICATION_SCHEMA = "adaptive_oracle_model_stage_verification_v2"
MODEL_STAGE_CLEANUP_SCHEMA = "adaptive_oracle_model_stage_cleanup_v1"
MODEL_STAGE_PREFIX = "adaptive-oracle-model-"
MODEL_REPOSITORY_DIR = (
    "models--stabilityai--stable-diffusion-xl-base-1.0"
)
MODEL_SNAPSHOT_RELATIVE = (
    MODEL_REPOSITORY_DIR + "/snapshots/" + MODEL_REVISION
)

_LOADED_FILES = {
    "model_index.json": (609, "6d7b93508390ab91ac5bfbe4aeb4dc2d83f7bb1b05fb069d714b5b0c75f70d44"),
    "scheduler/scheduler_config.json": (479, "af3e45a949aff8b8341ab8b811429ec03fee857a700a1d9477363e4fff9666e2"),
    "text_encoder/config.json": (565, "39b8b2e4b1949e36969caa425b6c81c68bace99198dd9078ce05d16ad401fe7f"),
    "text_encoder/model.fp16.safetensors": (246144152, "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"),
    "text_encoder_2/config.json": (575, "a892d1c3a69a7e9247a24de2bc1d5891e3109a54696e53be20093af671072c34"),
    "text_encoder_2/model.fp16.safetensors": (1389382176, "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4"),
    "tokenizer/merges.txt": (524619, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
    "tokenizer/special_tokens_map.json": (472, "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac"),
    "tokenizer/tokenizer_config.json": (737, "19d7b034cb0cc3ce9766c2231373ab8aa8991fc72e2c8f76558bfaae3de0d563"),
    "tokenizer/vocab.json": (1059962, "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"),
    "tokenizer_2/merges.txt": (524619, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
    "tokenizer_2/special_tokens_map.json": (460, "f118ab3a983206e4f32583448de6bd6aae4ee21869135cef1f5848a753cdaab6"),
    "tokenizer_2/tokenizer_config.json": (725, "c9d23941f76a41cbd50eda9290f57be7828f0a7a677939e9ef181f7e12bd1bdf"),
    "tokenizer_2/vocab.json": (1059962, "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"),
    "unet/config.json": (1680, "30ebc70750223e59006f7f2b4e1e6c102570aa19a9c4ae3e1fbe7591332dbae6"),
    "unet/diffusion_pytorch_model.fp16.safetensors": (5135149760, "83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139"),
    "vae/config.json": (642, "0b331c8ac22ded5f9997a144a575c1d113d6169aff262c353f39015bd24a6264"),
    "vae/diffusion_pytorch_model.fp16.safetensors": (167335342, "bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68"),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def expected_model_manifest(
    files: Optional[Mapping[str, tuple[int, str]]] = None,
) -> dict[str, Any]:
    """Return the frozen manifest for files actually selected by fp16 loading."""

    selected = dict(_LOADED_FILES if files is None else files)
    rows = []
    for relative_path, (size, digest) in sorted(selected.items()):
        pure = PurePosixPath(relative_path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != relative_path
            or not relative_path
            or not pure.parts
        ):
            raise ValueError("model file path must be canonical and snapshot-relative")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("model file size must be a positive integer")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("model file digest must be lowercase SHA-256")
        rows.append({"path": relative_path, "size": size, "sha256": digest})
    if not rows:
        raise ValueError("model snapshot manifest cannot be empty")
    return {
        "schema": MODEL_SNAPSHOT_SCHEMA,
        "hash_algorithm": MODEL_SNAPSHOT_HASH_ALGORITHM,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "variant": "fp16",
        "torch_dtype": "float16",
        "loaded_file_policy": "exact_files_selected_by_diffusers_fp16_variant",
        "loaded_file_count": len(rows),
        "files": rows,
    }


MODEL_SNAPSHOT_MANIFEST_SHA256 = canonical_sha256(expected_model_manifest())


_COPY_CHUNK_SIZE = 8 * 1024 * 1024
_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_IDENTITY_RECORD_FIELDS = frozenset(field[3:] for field in _IDENTITY_FIELDS)
_STAGE_RECORD_FIELDS = frozenset(
    {
        "schema",
        "status",
        "path",
        "parent",
        "manifest",
        "manifest_sha256",
        "loaded_file_count",
        "tree_sha256",
        "root_identity",
        "source_snapshot",
        "source_snapshot_sha256",
    }
)
_PARTIAL_STAGE_RECORD_FIELDS = frozenset(
    {"schema", "status", "path", "parent", "root_identity"}
)
_ACTIVE_STAGES: dict[str, dict[str, Any]] = {}


class _LiveModelStageRecord(dict[str, Any]):
    """JSON-compatible evidence carrying a process-local cleanup capability."""

    cleanup_token: object


class _LivePartialModelStageRecord(dict[str, Any]):
    """Process-local capability for cleaning an incompletely built stage."""

    cleanup_token: object


class ModelStageCreationError(RuntimeError):
    """A staging error whose first cleanup failed but remains retryable."""

    def __init__(
        self,
        original_error: BaseException,
        cleanup_error: BaseException,
        cleanup_record: Mapping[str, Any],
    ) -> None:
        super().__init__(
            "model staging failed and initial cleanup also failed: "
            f"{type(original_error).__name__}: {original_error}; "
            f"cleanup {type(cleanup_error).__name__}: {cleanup_error}; "
            f"retryable stage={cleanup_record.get('path')}"
        )
        self.original_error = original_error
        self.cleanup_error = cleanup_error
        self.cleanup_record = cleanup_record


def _require_live_stage_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(
        record, (_LiveModelStageRecord, _LivePartialModelStageRecord)
    ) or not hasattr(record, "cleanup_token"):
        raise ValueError("model stage operation requires its live creation record")
    state = _ACTIVE_STAGES.get(str(record["path"]))
    if state is None or state["token"] is not record.cleanup_token:
        raise ValueError("model stage capability is absent or stale")
    if state["record_sha256"] != canonical_sha256(record):
        raise ValueError("model stage record changed after creation")
    return state


def _require_fd_safety() -> None:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("model staging requires O_NOFOLLOW and O_DIRECTORY")


def _read_flags(*, directory: bool = False) -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= os.O_DIRECTORY
    return flags


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(value, field)) for field in _IDENTITY_FIELDS)


def _identity_record(value: os.stat_result) -> dict[str, int]:
    return {field[3:]: int(getattr(value, field)) for field in _IDENTITY_FIELDS}


def _record_matches_object(
    recorded: Any, observed: os.stat_result
) -> bool:
    if not isinstance(recorded, Mapping):
        return False
    stable_fields = ("dev", "ino", "uid", "gid")
    return all(
        recorded.get(field) == int(getattr(observed, "st_" + field))
        for field in stable_fields
    ) and stat.S_ISDIR(observed.st_mode)


def _require_stage_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != _STAGE_RECORD_FIELDS:
        raise ValueError("model stage record fields differ")
    if record["schema"] != MODEL_STAGE_SCHEMA:
        raise ValueError("model stage record schema differs")
    if record["status"] != "staged_verified_read_only":
        raise ValueError("model stage record status differs")
    manifest = record.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("model stage record manifest is invalid")
    if canonical_sha256(manifest) != record.get("manifest_sha256"):
        raise ValueError("model stage record manifest digest differs")
    if manifest.get("loaded_file_count") != record.get("loaded_file_count"):
        raise ValueError("model stage record file count differs")
    source_snapshot = record.get("source_snapshot")
    if canonical_sha256(source_snapshot) != record.get("source_snapshot_sha256"):
        raise ValueError("model stage source evidence digest differs")
    _validate_source_snapshot_record(source_snapshot, manifest)


def _require_partial_stage_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != _PARTIAL_STAGE_RECORD_FIELDS:
        raise ValueError("partial model stage cleanup record fields differ")
    if record["schema"] != MODEL_STAGE_SCHEMA:
        raise ValueError("partial model stage cleanup record schema differs")
    if record["status"] != "staging_failed_cleanup_pending":
        raise ValueError("partial model stage cleanup record status differs")
    _require_identity_record(record["root_identity"], "partial model stage root")


def _require_identity_record(value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_RECORD_FIELDS:
        raise ValueError(f"{label} identity record differs")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value.values()):
        raise ValueError(f"{label} identity values must be integers")


def _validate_source_snapshot_record(
    value: Any, manifest: Mapping[str, Any]
) -> None:
    expected_top = frozenset(
        {
            "path",
            "identity",
            "model_repository_path",
            "model_repository_identity",
            "files",
        }
    )
    if not isinstance(value, Mapping) or set(value) != expected_top:
        raise ValueError("model stage source evidence fields differ")
    for field in ("path", "model_repository_path"):
        path = Path(str(value[field]))
        if not path.is_absolute() or Path(os.path.abspath(path)) != path:
            raise ValueError(f"model stage source {field} is not canonical")
    _require_identity_record(value["identity"], "model source snapshot")
    _require_identity_record(
        value["model_repository_identity"], "model source repository"
    )
    rows = value["files"]
    if not isinstance(rows, list) or len(rows) != manifest["loaded_file_count"]:
        raise ValueError("model stage source file evidence count differs")
    expected_file_fields = frozenset(
        {
            "path",
            "size",
            "sha256",
            "kind",
            "snapshot_entry_identity",
            "link_target",
            "storage_path",
            "storage_relative_path",
            "storage_identity",
        }
    )
    for source_row, manifest_row in zip(rows, manifest["files"]):
        if not isinstance(source_row, Mapping) or set(source_row) != expected_file_fields:
            raise ValueError("model stage source file evidence fields differ")
        if any(
            source_row[field] != manifest_row[field]
            for field in ("path", "size", "sha256")
        ):
            raise ValueError("model stage source file evidence differs from manifest")
        _require_identity_record(
            source_row["snapshot_entry_identity"], "model source entry"
        )
        _require_identity_record(source_row["storage_identity"], "model source storage")
        storage_path = Path(str(source_row["storage_path"]))
        if not storage_path.is_absolute() or Path(os.path.abspath(storage_path)) != storage_path:
            raise ValueError("model source storage path is not canonical")
        kind = source_row["kind"]
        if kind == "snapshot_regular_file":
            if (
                source_row["link_target"] is not None
                or source_row["storage_relative_path"] is not None
            ):
                raise ValueError("regular model source has symlink evidence")
        elif kind == "huggingface_blob_symlink":
            if not isinstance(source_row["link_target"], str) or not isinstance(
                source_row["storage_relative_path"], str
            ):
                raise ValueError("HF model source omitted symlink evidence")
            blob_parts = PurePosixPath(source_row["storage_relative_path"]).parts
            if len(blob_parts) != 2 or blob_parts[0] != "blobs":
                raise ValueError("HF model source storage evidence escapes blobs")
            if _normalized_blob_parts(
                str(source_row["path"]), source_row["link_target"]
            ) != tuple(blob_parts):
                raise ValueError("HF model source link evidence differs")
        else:
            raise ValueError("model source kind differs")


def _require_regular_file(value: os.stat_result, label: str) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    if value.st_nlink != 1:
        raise ValueError(f"{label} must not be hard-linked")


def _require_directory(value: os.stat_result, label: str) -> None:
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink directory")


def _expected_manifest(
    files: Optional[Mapping[str, tuple[int, str]]],
    expected_manifest_sha256: Optional[str],
) -> tuple[dict[str, Any], str]:
    manifest = expected_model_manifest(files)
    expected_digest = (
        MODEL_SNAPSHOT_MANIFEST_SHA256
        if expected_manifest_sha256 is None
        else expected_manifest_sha256
    )
    if canonical_sha256(manifest) != expected_digest:
        raise ValueError("expected model snapshot manifest SHA-256 is inconsistent")
    return manifest, expected_digest


def _source_snapshot(
    repo_root: Path,
) -> tuple[Path, int, os.stat_result, Path, int]:
    _require_fd_safety()
    root = Path(repo_root).resolve(strict=True)
    cache = (root / MODEL_CACHE).resolve(strict=True)
    model_repository = (cache / MODEL_REPOSITORY_DIR).resolve(strict=True)
    snapshot = (model_repository / "snapshots" / MODEL_REVISION).resolve(strict=True)
    if model_repository not in snapshot.parents:
        raise ValueError("model snapshot escapes its model repository")
    try:
        model_repository_descriptor = os.open(
            model_repository, _read_flags(directory=True)
        )
        descriptor = os.open(snapshot, _read_flags(directory=True))
    except OSError as exc:
        try:
            os.close(model_repository_descriptor)
        except UnboundLocalError:
            pass
        raise ValueError("model snapshot must be a regular non-symlink directory") from exc
    observed = os.fstat(descriptor)
    try:
        repository_identity = os.fstat(model_repository_descriptor)
        _require_directory(repository_identity, "model repository")
        repository_path_identity = os.stat(model_repository, follow_symlinks=False)
        if _identity(repository_path_identity) != _identity(repository_identity):
            raise ValueError("model repository path changed while it was opened")
        _require_directory(observed, "model snapshot")
        path_identity = os.stat(snapshot, follow_symlinks=False)
        if _identity(path_identity) != _identity(observed):
            raise ValueError("model snapshot path changed while it was opened")
    except BaseException:
        os.close(descriptor)
        os.close(model_repository_descriptor)
        raise
    return (
        snapshot,
        descriptor,
        observed,
        model_repository,
        model_repository_descriptor,
    )


def _open_relative_directory(
    root_descriptor: int,
    parts: tuple[str, ...],
    *,
    create: bool,
) -> int:
    current = os.dup(root_descriptor)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                    os.fsync(current)
                except FileExistsError:
                    pass
            try:
                child = os.open(part, _read_flags(directory=True), dir_fd=current)
            except OSError as exc:
                raise ValueError(
                    f"model snapshot directory is missing, linked, or invalid: {part}"
                ) from exc
            child_identity = os.fstat(child)
            try:
                _require_directory(child_identity, f"model directory {part}")
            except BaseException:
                os.close(child)
                raise
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("model staging write made no progress")
        view = view[written:]


def _normalized_blob_parts(row_path: str, link_target: str) -> tuple[str, str]:
    target = PurePosixPath(link_target)
    if (
        not link_target
        or target.is_absolute()
        or str(target) != link_target
        or "\\" in link_target
    ):
        raise ValueError(f"model source symlink target is invalid: {row_path}")
    parent_depth = len(PurePosixPath(row_path).parts) - 1
    expected_prefix = ("..",) * (2 + parent_depth) + ("blobs",)
    if len(target.parts) != len(expected_prefix) + 1 or tuple(
        target.parts[:-1]
    ) != expected_prefix:
        raise ValueError(
            f"model source symlink must resolve inside repository blobs: {row_path}"
        )
    blob_name = target.parts[-1]
    if len(blob_name) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in blob_name
    ):
        raise ValueError(f"model source blob name is invalid: {row_path}")
    return "blobs", blob_name


def _open_registered_source(
    *,
    source_parent: int,
    source_name: str,
    row_path: str,
    source_snapshot_path: Path,
    model_repository_descriptor: int,
    model_repository_path: Path,
) -> tuple[int, os.stat_result, dict[str, Any]]:
    entry_before = os.stat(
        source_name, dir_fd=source_parent, follow_symlinks=False
    )
    if stat.S_ISREG(entry_before.st_mode):
        _require_regular_file(entry_before, f"model source {row_path}")
        try:
            descriptor = os.open(source_name, _read_flags(), dir_fd=source_parent)
        except OSError as exc:
            raise ValueError(f"model source changed while opened: {row_path}") from exc
        opened = os.fstat(descriptor)
        if _identity(entry_before) != _identity(opened):
            os.close(descriptor)
            raise ValueError(f"model source path changed while opened: {row_path}")
        return descriptor, entry_before, {
            "kind": "snapshot_regular_file",
            "snapshot_entry_identity": _identity_record(entry_before),
            "link_target": None,
            "storage_path": str(
                source_snapshot_path.joinpath(*PurePosixPath(row_path).parts)
            ),
            "storage_relative_path": None,
        }
    if not stat.S_ISLNK(entry_before.st_mode):
        raise ValueError(f"model source is not a regular file or HF blob link: {row_path}")
    if entry_before.st_nlink != 1:
        raise ValueError(f"model source symlink must not be hard-linked: {row_path}")
    try:
        link_target = os.readlink(source_name, dir_fd=source_parent)
    except OSError as exc:
        raise ValueError(f"model source symlink changed: {row_path}") from exc
    blob_directory, blob_name = _normalized_blob_parts(row_path, link_target)
    blob_parent = _open_relative_directory(
        model_repository_descriptor, (blob_directory,), create=False
    )
    try:
        try:
            descriptor = os.open(blob_name, _read_flags(), dir_fd=blob_parent)
        except OSError as exc:
            raise ValueError(
                f"model source blob is missing, linked, or invalid: {row_path}"
            ) from exc
        opened = os.fstat(descriptor)
        try:
            _require_regular_file(opened, f"model source blob {row_path}")
            blob_path_identity = os.stat(
                blob_name, dir_fd=blob_parent, follow_symlinks=False
            )
            if _identity(blob_path_identity) != _identity(opened):
                raise ValueError(f"model source blob path changed: {row_path}")
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        os.close(blob_parent)
    return descriptor, entry_before, {
        "kind": "huggingface_blob_symlink",
        "snapshot_entry_identity": _identity_record(entry_before),
        "link_target": link_target,
        "storage_path": str(model_repository_path / blob_directory / blob_name),
        "storage_relative_path": str(PurePosixPath(blob_directory, blob_name)),
    }


def _copy_manifest_file(
    source_root: int,
    source_snapshot_path: Path,
    model_repository_descriptor: int,
    model_repository_path: Path,
    destination_root: Optional[int],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    parts = PurePosixPath(str(row["path"])).parts
    source_parent = _open_relative_directory(
        source_root, tuple(parts[:-1]), create=False
    )
    destination_parent: Optional[int] = None
    source_descriptor: Optional[int] = None
    destination_descriptor: Optional[int] = None
    try:
        try:
            source_descriptor, entry_before, source_record = _open_registered_source(
                source_parent=source_parent,
                source_name=parts[-1],
                row_path=str(row["path"]),
                source_snapshot_path=source_snapshot_path,
                model_repository_descriptor=model_repository_descriptor,
                model_repository_path=model_repository_path,
            )
        except (OSError, FileNotFoundError) as exc:
            raise ValueError(
                f"model source is missing, linked, or invalid: {row['path']}"
            ) from exc
        before = os.fstat(source_descriptor)
        _require_regular_file(before, f"model source {row['path']}")
        if before.st_size != row["size"]:
            raise ValueError(f"model file size differs: {row['path']}")

        if destination_root is not None:
            destination_parent = _open_relative_directory(
                destination_root, tuple(parts[:-1]), create=True
            )
            create_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                destination_descriptor = os.open(
                    parts[-1], create_flags, 0o600, dir_fd=destination_parent
                )
            except OSError as exc:
                raise ValueError(
                    f"staged model file already exists or is invalid: {row['path']}"
                ) from exc

        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_descriptor, _COPY_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            if destination_descriptor is not None:
                _write_all(destination_descriptor, chunk)

        after = os.fstat(source_descriptor)
        if _identity(before) != _identity(after):
            raise ValueError(f"model source changed while copied: {row['path']}")
        entry_after = os.stat(
            parts[-1], dir_fd=source_parent, follow_symlinks=False
        )
        if _identity(entry_before) != _identity(entry_after):
            raise ValueError(f"model source path changed while copied: {row['path']}")
        if source_record["kind"] == "snapshot_regular_file":
            if _identity(before) != _identity(entry_after):
                raise ValueError(
                    f"model source path changed while copied: {row['path']}"
                )
        else:
            try:
                link_after = os.readlink(parts[-1], dir_fd=source_parent)
            except OSError as exc:
                raise ValueError(
                    f"model source symlink changed while copied: {row['path']}"
                ) from exc
            if link_after != source_record["link_target"]:
                raise ValueError(
                    f"model source symlink target changed while copied: {row['path']}"
                )
            storage_parts = PurePosixPath(
                source_record["storage_relative_path"]
            ).parts
            blob_parent = _open_relative_directory(
                model_repository_descriptor, tuple(storage_parts[:-1]), create=False
            )
            try:
                blob_after = os.stat(
                    storage_parts[-1], dir_fd=blob_parent, follow_symlinks=False
                )
            finally:
                os.close(blob_parent)
            if _identity(before) != _identity(blob_after):
                raise ValueError(
                    f"model source blob path changed while copied: {row['path']}"
                )
        observed_digest = digest.hexdigest()
        if copied != row["size"] or observed_digest != row["sha256"]:
            raise ValueError(f"model file SHA-256 differs: {row['path']}")

        if destination_descriptor is not None:
            os.fchmod(destination_descriptor, 0o400)
            os.fsync(destination_descriptor)
            staged = os.fstat(destination_descriptor)
            _require_regular_file(staged, f"staged model file {row['path']}")
            if staged.st_size != copied or stat.S_IMODE(staged.st_mode) != 0o400:
                raise ValueError(f"staged model file identity differs: {row['path']}")
            os.fsync(destination_parent)
        return {
            "path": row["path"],
            "size": copied,
            "sha256": observed_digest,
            **source_record,
            "storage_identity": _identity_record(before),
        }
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)
        if destination_parent is not None:
            os.close(destination_parent)
        os.close(source_parent)


def _expected_tree(manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    files = {str(row["path"]) for row in manifest["files"]}
    directories: set[str] = set()
    for relative_path in files:
        parts = PurePosixPath(relative_path).parts[:-1]
        for end in range(1, len(parts) + 1):
            directories.add(str(PurePosixPath(*parts[:end])))
    return files, directories


def _seal_stage_tree(root_descriptor: int, manifest: Mapping[str, Any]) -> None:
    _, directories = _expected_tree(manifest)
    for relative_path in sorted(
        directories, key=lambda value: len(PurePosixPath(value).parts), reverse=True
    ):
        descriptor = _open_relative_directory(
            root_descriptor, PurePosixPath(relative_path).parts, create=False
        )
        try:
            os.fchmod(descriptor, 0o500)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.fchmod(root_descriptor, 0o500)
    os.fsync(root_descriptor)


def _inspect_stage_tree(
    root_descriptor: int, manifest: Mapping[str, Any]
) -> tuple[dict[str, int], list[dict[str, Any]], str]:
    expected_files, expected_directories = _expected_tree(manifest)
    expected_rows = {str(row["path"]): row for row in manifest["files"]}
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    observed_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    held_files: list[tuple[int, tuple[int, ...], str]] = []
    held_directories: list[tuple[int, tuple[int, ...], list[str], str]] = []

    def visit(directory_descriptor: int, prefix: PurePosixPath) -> None:
        directory_before = os.fstat(directory_descriptor)
        _require_directory(directory_before, "staged model directory")
        if stat.S_IMODE(directory_before.st_mode) != 0o500:
            raise ValueError("staged model directories must be read-only")
        if directory_before.st_uid != os.geteuid():
            raise ValueError("staged model directory owner differs")
        identity_rows.append(
            {
                "path": str(prefix) or ".",
                "kind": "directory",
                "identity": _identity_record(directory_before),
            }
        )
        names_before = sorted(os.listdir(directory_descriptor))
        for name in names_before:
            relative = prefix / name
            relative_text = str(relative)
            entry = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if stat.S_ISLNK(entry.st_mode):
                raise ValueError(f"staged model tree contains a symlink: {relative_text}")
            if stat.S_ISDIR(entry.st_mode):
                actual_directories.add(relative_text)
                try:
                    child = os.open(
                        name, _read_flags(directory=True), dir_fd=directory_descriptor
                    )
                except OSError as exc:
                    raise ValueError(
                        f"staged model directory changed: {relative_text}"
                    ) from exc
                try:
                    if _identity(entry) != _identity(os.fstat(child)):
                        raise ValueError(
                            f"staged model directory path changed: {relative_text}"
                        )
                    visit(child, relative)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(entry.st_mode):
                raise ValueError(
                    f"staged model tree contains a non-regular entry: {relative_text}"
                )
            actual_files.add(relative_text)
            row = expected_rows.get(relative_text)
            if row is None:
                continue
            try:
                descriptor = os.open(name, _read_flags(), dir_fd=directory_descriptor)
            except OSError as exc:
                raise ValueError(f"staged model file changed: {relative_text}") from exc
            try:
                before = os.fstat(descriptor)
                _require_regular_file(before, f"staged model file {relative_text}")
                if stat.S_IMODE(before.st_mode) != 0o400:
                    raise ValueError(f"staged model file is not read-only: {relative_text}")
                if before.st_uid != os.geteuid():
                    raise ValueError(f"staged model file owner differs: {relative_text}")
                digest = hashlib.sha256()
                size = 0
                while True:
                    chunk = os.read(descriptor, _COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                after = os.fstat(descriptor)
                path_after = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
                if _identity(before) != _identity(after) or _identity(before) != _identity(
                    path_after
                ):
                    raise ValueError(f"staged model file changed: {relative_text}")
                observed_digest = digest.hexdigest()
                if size != row["size"]:
                    raise ValueError(f"staged model file size differs: {relative_text}")
                if observed_digest != row["sha256"]:
                    raise ValueError(f"staged model file SHA-256 differs: {relative_text}")
                observed_rows.append(
                    {"path": relative_text, "size": size, "sha256": observed_digest}
                )
                identity_rows.append(
                    {
                        "path": relative_text,
                        "kind": "file",
                        "identity": _identity_record(before),
                    }
                )
                held_files.append((descriptor, _identity(before), relative_text))
                descriptor = None
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        names_after = sorted(os.listdir(directory_descriptor))
        directory_after = os.fstat(directory_descriptor)
        if names_before != names_after or _identity(directory_before) != _identity(
            directory_after
        ):
            raise ValueError("staged model directory changed while verified")
        held_directories.append(
            (
                os.dup(directory_descriptor),
                _identity(directory_after),
                names_after,
                str(prefix),
            )
        )

    try:
        visit(root_descriptor, PurePosixPath())
        if actual_files != expected_files or actual_directories != expected_directories:
            raise ValueError("staged model tree inventory differs from the exact manifest")
        for descriptor, expected_identity, relative_text in held_files:
            if _identity(os.fstat(descriptor)) != expected_identity:
                raise ValueError(
                    f"staged model file changed after verification: {relative_text}"
                )
        for descriptor, expected_identity, expected_names, relative_text in held_directories:
            if _identity(os.fstat(descriptor)) != expected_identity or sorted(
                os.listdir(descriptor)
            ) != expected_names:
                raise ValueError(
                    f"staged model directory changed after verification: {relative_text}"
                )
        root_identity = os.fstat(root_descriptor)
        rows = sorted(observed_rows, key=lambda row: row["path"])
        tree_sha256 = canonical_sha256(
            {
                "files": rows,
                "object_identities": sorted(
                    identity_rows, key=lambda row: (row["path"], row["kind"])
                ),
            }
        )
        return _identity_record(root_identity), rows, tree_sha256
    finally:
        for descriptor, _, _ in held_files:
            os.close(descriptor)
        for descriptor, _, _, _ in held_directories:
            os.close(descriptor)


def _open_recorded_stage_root(
    record: Mapping[str, Any],
) -> tuple[Path, int, int, os.stat_result]:
    path = Path(str(record.get("path", "")))
    parent = Path(str(record.get("parent", "")))
    if (
        not path.is_absolute()
        or not parent.is_absolute()
        or path.parent != parent
        or not path.name.startswith(MODEL_STAGE_PREFIX)
        or Path(os.path.abspath(path)) != path
        or Path(os.path.abspath(parent)) != parent
    ):
        raise ValueError("model stage path record is not canonical")
    try:
        parent_descriptor = os.open(parent, _read_flags(directory=True))
        path_identity = os.stat(
            path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        root_descriptor = os.open(
            path.name, _read_flags(directory=True), dir_fd=parent_descriptor
        )
    except OSError as exc:
        try:
            os.close(parent_descriptor)
        except UnboundLocalError:
            pass
        raise ValueError("model stage root is missing, linked, or invalid") from exc
    observed = os.fstat(root_descriptor)
    try:
        _require_directory(observed, "model stage root")
        if _identity(path_identity) != _identity(observed):
            raise ValueError("model stage root path identity changed")
        recorded_identity = record.get("root_identity")
        if not _record_matches_object(recorded_identity, observed):
            raise ValueError("model stage root differs from its creation record")
    except BaseException:
        os.close(root_descriptor)
        os.close(parent_descriptor)
        raise
    return path, parent_descriptor, root_descriptor, observed


def _remove_directory_contents(directory_descriptor: int) -> None:
    os.fchmod(directory_descriptor, 0o700)
    for name in sorted(os.listdir(directory_descriptor)):
        entry = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISDIR(entry.st_mode):
            child = os.open(
                name, _read_flags(directory=True), dir_fd=directory_descriptor
            )
            try:
                if _identity(entry) != _identity(os.fstat(child)):
                    raise ValueError("model stage directory changed during cleanup")
                _remove_directory_contents(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=directory_descriptor)
        else:
            os.unlink(name, dir_fd=directory_descriptor)
    os.fsync(directory_descriptor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_snapshot(
    repo_root: Path,
    *,
    files: Optional[Mapping[str, tuple[int, str]]] = None,
    expected_manifest_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Hash every registered source through stable, non-following descriptors."""

    manifest, expected_digest = _expected_manifest(files, expected_manifest_sha256)
    (
        source_path,
        source_descriptor,
        _,
        model_repository_path,
        model_repository_descriptor,
    ) = _source_snapshot(Path(repo_root))
    try:
        for row in manifest["files"]:
            _copy_manifest_file(
                source_descriptor,
                source_path,
                model_repository_descriptor,
                model_repository_path,
                None,
                row,
            )
    finally:
        os.close(source_descriptor)
        os.close(model_repository_descriptor)
    return {**manifest, "manifest_sha256": expected_digest}


def stage_model_snapshot(
    repo_root: Path,
    *,
    staging_parent: Optional[Path] = None,
    files: Optional[Mapping[str, tuple[int, str]]] = None,
    expected_manifest_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Copy the exact registered model bytes into a private immutable load tree."""

    manifest, expected_digest = _expected_manifest(files, expected_manifest_sha256)
    (
        source_path,
        source_descriptor,
        source_identity,
        model_repository_path,
        model_repository_descriptor,
    ) = _source_snapshot(Path(repo_root))
    parent = Path(tempfile.gettempdir() if staging_parent is None else staging_parent)
    stage_path: Optional[Path] = None
    stage_descriptor: Optional[int] = None
    creation_parent_descriptor: Optional[int] = None
    cleanup_record: Optional[Mapping[str, Any]] = None
    source_files: list[dict[str, Any]] = []
    try:
        if parent.is_symlink():
            raise ValueError("model staging parent must not be a symlink")
        parent = parent.resolve(strict=True)
        if not parent.is_dir():
            raise ValueError("model staging parent must be a directory")
        creation_parent_descriptor = os.open(parent, _read_flags(directory=True))
        parent_identity = os.fstat(creation_parent_descriptor)
        if parent_identity.st_uid != os.geteuid() and not (
            parent_identity.st_mode & stat.S_ISVTX
        ):
            raise ValueError("model staging parent is not owned or sticky")
        if parent_identity.st_mode & 0o022 and not (
            parent_identity.st_mode & stat.S_ISVTX
        ):
            raise ValueError("model staging parent is writable by other users")
        stage_path = Path(tempfile.mkdtemp(prefix=MODEL_STAGE_PREFIX, dir=str(parent)))
        try:
            created_path_identity = os.stat(
                stage_path.name,
                dir_fd=creation_parent_descriptor,
                follow_symlinks=False,
            )
            stage_descriptor = os.open(
                stage_path.name,
                _read_flags(directory=True),
                dir_fd=creation_parent_descriptor,
            )
        except OSError as exc:
            raise ValueError("model stage root changed before it was opened") from exc
        created = os.fstat(stage_descriptor)
        _require_directory(created, "model stage root")
        if _identity(created_path_identity) != _identity(created):
            raise ValueError("model stage root changed while it was opened")
        if created.st_uid != os.geteuid() or stat.S_IMODE(created.st_mode) != 0o700:
            raise ValueError("model stage root is not private to the current user")
        partial_record = _LivePartialModelStageRecord(
            {
                "schema": MODEL_STAGE_SCHEMA,
                "status": "staging_failed_cleanup_pending",
                "path": str(stage_path),
                "parent": str(parent),
                "root_identity": _identity_record(created),
            }
        )
        partial_record.cleanup_token = object()
        _ACTIVE_STAGES[str(stage_path)] = {
            "token": partial_record.cleanup_token,
            "record_sha256": canonical_sha256(partial_record),
        }
        cleanup_record = partial_record
        os.fsync(creation_parent_descriptor)
        os.close(creation_parent_descriptor)
        creation_parent_descriptor = None

        for row in manifest["files"]:
            source_files.append(
                _copy_manifest_file(
                    source_descriptor,
                    source_path,
                    model_repository_descriptor,
                    model_repository_path,
                    stage_descriptor,
                    row,
                )
            )
        _seal_stage_tree(stage_descriptor, manifest)
        root_identity, observed_rows, tree_sha256 = _inspect_stage_tree(
            stage_descriptor, manifest
        )
        if observed_rows != manifest["files"]:
            raise ValueError("staged model bytes differ from the exact manifest")
        source_snapshot_record = {
            "path": str(source_path),
            "identity": _identity_record(source_identity),
            "model_repository_path": str(model_repository_path),
            "model_repository_identity": _identity_record(
                os.fstat(model_repository_descriptor)
            ),
            "files": source_files,
        }
        record = _LiveModelStageRecord({
            "schema": MODEL_STAGE_SCHEMA,
            "status": "staged_verified_read_only",
            "path": str(stage_path),
            "parent": str(parent),
            "manifest": manifest,
            "manifest_sha256": expected_digest,
            "loaded_file_count": manifest["loaded_file_count"],
            "tree_sha256": tree_sha256,
            "root_identity": root_identity,
            "source_snapshot": source_snapshot_record,
            "source_snapshot_sha256": canonical_sha256(source_snapshot_record),
        })
        record.cleanup_token = partial_record.cleanup_token
        _ACTIVE_STAGES[str(stage_path)] = {
            "token": record.cleanup_token,
            "record_sha256": canonical_sha256(record),
        }
        cleanup_record = record
        verify_staged_model_snapshot(
            record,
            files=files,
            expected_manifest_sha256=expected_digest,
        )
        return record
    except BaseException as stage_exc:
        if cleanup_record is not None:
            if stage_descriptor is not None:
                os.close(stage_descriptor)
                stage_descriptor = None
            try:
                cleanup_staged_model_snapshot(cleanup_record)
            except BaseException as cleanup_exc:
                raise ModelStageCreationError(
                    stage_exc, cleanup_exc, cleanup_record
                ) from stage_exc
        elif stage_path is not None and creation_parent_descriptor is not None:
            try:
                entry = os.stat(
                    stage_path.name,
                    dir_fd=creation_parent_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISDIR(entry.st_mode):
                    os.rmdir(stage_path.name, dir_fd=creation_parent_descriptor)
                else:
                    os.unlink(stage_path.name, dir_fd=creation_parent_descriptor)
                os.fsync(creation_parent_descriptor)
            except BaseException as cleanup_exc:
                raise RuntimeError(
                    "model staging failed before cleanup capability creation and "
                    "initial cleanup also failed"
                ) from cleanup_exc
        raise
    finally:
        if stage_descriptor is not None:
            os.close(stage_descriptor)
        if creation_parent_descriptor is not None:
            os.close(creation_parent_descriptor)
        os.close(source_descriptor)
        os.close(model_repository_descriptor)


def verify_staged_model_snapshot(
    record: Mapping[str, Any],
    *,
    files: Optional[Mapping[str, tuple[int, str]]] = None,
    expected_manifest_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Rehash a staged tree and reject any identity, link, or inventory drift."""

    _require_stage_record(record)
    _require_live_stage_record(record)
    manifest, expected_digest = _expected_manifest(files, expected_manifest_sha256)
    if record["manifest"] != manifest:
        raise ValueError("model stage manifest differs")
    if (
        record["manifest_sha256"] != expected_digest
        or record["loaded_file_count"] != manifest["loaded_file_count"]
    ):
        raise ValueError("model stage manifest binding differs")

    path, parent_descriptor, root_descriptor, _ = _open_recorded_stage_root(record)
    try:
        root_identity, observed_rows, tree_sha256 = _inspect_stage_tree(
            root_descriptor, manifest
        )
        if (
            root_identity != record["root_identity"]
            or observed_rows != manifest["files"]
            or tree_sha256 != record["tree_sha256"]
        ):
            raise ValueError("staged model tree digest differs")
        return {
            "schema": MODEL_STAGE_VERIFICATION_SCHEMA,
            "status": "verified_unchanged",
            "path": str(path),
            "manifest_sha256": expected_digest,
            "loaded_file_count": manifest["loaded_file_count"],
            "tree_sha256": tree_sha256,
            "root_identity": root_identity,
        }
    finally:
        os.close(root_descriptor)
        os.close(parent_descriptor)


def load_from_verified_staged_model_snapshot(
    record: Mapping[str, Any],
    loader: Any,
    *,
    files: Optional[Mapping[str, tuple[int, str]]] = None,
    expected_manifest_sha256: Optional[str] = None,
    verification_log: Optional[list[dict[str, Any]]] = None,
) -> Any:
    """Load through a pinned root FD and bind acceptance to pre/post tree signatures."""

    if not callable(loader):
        raise TypeError("model loader must be callable")
    _require_stage_record(record)
    _require_live_stage_record(record)
    manifest, expected_digest = _expected_manifest(files, expected_manifest_sha256)
    if record["manifest"] != manifest or record["manifest_sha256"] != expected_digest:
        raise ValueError("model stage manifest binding differs")
    if not Path("/proc/self/fd").is_dir():
        raise RuntimeError("pinned model loading requires Linux procfs file descriptors")

    path, parent_descriptor, root_descriptor, _ = _open_recorded_stage_root(record)
    verifications = [] if verification_log is None else verification_log

    def verify_open_root() -> dict[str, Any]:
        root_identity, observed_rows, tree_sha256 = _inspect_stage_tree(
            root_descriptor, manifest
        )
        if (
            root_identity != record["root_identity"]
            or observed_rows != manifest["files"]
            or tree_sha256 != record["tree_sha256"]
        ):
            raise ValueError("staged model tree digest differs during bound load")
        return {
            "schema": MODEL_STAGE_VERIFICATION_SCHEMA,
            "status": "verified_unchanged",
            "path": str(path),
            "manifest_sha256": expected_digest,
            "loaded_file_count": manifest["loaded_file_count"],
            "tree_sha256": tree_sha256,
            "root_identity": root_identity,
        }

    try:
        before = verify_open_root()
        verifications.append(before)
        proc_root = Path("/proc/self/fd") / str(root_descriptor)
        proc_identity = os.stat(proc_root, follow_symlinks=True)
        if _identity(proc_identity) != _identity(os.fstat(root_descriptor)):
            raise RuntimeError("pinned model load root does not resolve to the held FD")
        loaded = loader(str(proc_root))
        after = verify_open_root()
        verifications.append(after)
        public_identity = os.stat(
            path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if _identity(public_identity) != _identity(os.fstat(root_descriptor)):
            raise ValueError("public model stage path changed during bound load")
        return loaded
    finally:
        os.close(root_descriptor)
        os.close(parent_descriptor)


def cleanup_staged_model_snapshot(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the recorded staging inode without following tree links."""

    partial = isinstance(record, _LivePartialModelStageRecord)
    if partial:
        _require_partial_stage_record(record)
    else:
        _require_stage_record(record)
    _require_live_stage_record(record)
    path_key = str(record["path"])
    path, parent_descriptor, root_descriptor, root_identity = (
        _open_recorded_stage_root(record)
    )
    try:
        _remove_directory_contents(root_descriptor)
        path_after = os.stat(
            path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if _identity(path_after) != _identity(os.fstat(root_descriptor)):
            raise ValueError("model stage root path changed during cleanup")
        os.rmdir(path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(root_descriptor)
        os.close(parent_descriptor)
    del _ACTIVE_STAGES[path_key]
    return {
        "schema": MODEL_STAGE_CLEANUP_SCHEMA,
        "status": "removed",
        "path": str(path),
        "manifest_sha256": None if partial else record["manifest_sha256"],
        "loaded_file_count": None if partial else record["loaded_file_count"],
        "root_identity": _identity_record(root_identity),
    }


__all__ = [
    "MODEL_CACHE",
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_REPOSITORY_DIR",
    "MODEL_SNAPSHOT_HASH_ALGORITHM",
    "MODEL_SNAPSHOT_MANIFEST_SHA256",
    "MODEL_SNAPSHOT_RELATIVE",
    "MODEL_SNAPSHOT_SCHEMA",
    "MODEL_STAGE_CLEANUP_SCHEMA",
    "MODEL_STAGE_SCHEMA",
    "MODEL_STAGE_VERIFICATION_SCHEMA",
    "ModelStageCreationError",
    "canonical_json_bytes",
    "canonical_sha256",
    "cleanup_staged_model_snapshot",
    "expected_model_manifest",
    "sha256_file",
    "stage_model_snapshot",
    "validate_model_snapshot",
    "verify_staged_model_snapshot",
    "load_from_verified_staged_model_snapshot",
]
