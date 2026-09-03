"""Verification of every external byte source used by a formal run.

The run contract stores absolute paths because training is executed on one
shared cluster filesystem.  Paths are never trusted by name alone: the
launcher hashes each descriptor and every file listed by the model/reward
manifests before constructing a renderer, dataset, or reward model.  Hot-path
checks compare inode and timestamp snapshots so callbacks cannot silently swap
an already verified artifact during a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
import weakref

import torch


FILE_MANIFEST_SCHEMA = "repldm.file_manifest.v1"
INITIAL_STATE_SCHEMA = "repldm.renderer_initial_state.v1"

REQUIRED_RUN_ARTIFACTS = (
    "data_manifest",
    "prompt_manifest",
    "reward_config",
    "reward_assets_manifest",
    "model_assets_manifest",
    "basis_provider_config",
    "calibration",
    "renderer_frame_contract",
    "initial_renderer_state",
)
GATED_RUN_ARTIFACTS = (
    "f0_gate",
    "opsd_teacher_state",
    "reward_statistics",
    "cohort_manifest",
)
F0_WITNESS_RUN_ARTIFACTS = (
    "witness_config",
    "witness_assets_manifest",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_VERIFIED: dict[int, Any] = {}


def required_run_artifacts(method: Any) -> tuple[str, ...]:
    """Return the exact descriptor inventory for one registered method."""
    names = list(REQUIRED_RUN_ARTIFACTS)
    if method == "f0":
        names.extend(F0_WITNESS_RUN_ARTIFACTS)
    if method in {"opd", "search_distill", "dpo", "rl"}:
        names.extend(GATED_RUN_ARTIFACTS)
    return tuple(names)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("run artifacts must contain canonical JSON values") from exc


def module_state_sha256(module: Any) -> str:
    """Hash a module state without depending on nondeterministic pickle bytes."""
    try:
        state = module.state_dict()
    except (AttributeError, TypeError) as exc:
        raise TypeError("renderer must expose state_dict()") from exc
    if not isinstance(state, Mapping):
        raise TypeError("renderer state_dict() must return a mapping")
    if any(not isinstance(name, str) for name in state):
        raise ValueError("renderer state keys must be strings")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"renderer state {name} is not a tensor")
        try:
            tensor = value.detach().contiguous().cpu()
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise ValueError(f"renderer state {name} is not a tensor") from exc
        metadata = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        digest.update(len(_canonical(metadata)).to_bytes(8, "big"))
        digest.update(_canonical(metadata))
        try:
            payload = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
        except (AttributeError, RuntimeError, TypeError):
            payload = bytes(tensor.view(torch.uint8).reshape(-1).tolist())
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _sha256_file(path: Path) -> tuple[str, os.stat_result]:
    digest = hashlib.sha256()
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(f"cannot open run artifact: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError(f"run artifact must be a non-empty regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError(f"run artifact changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), after


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _ordinary_absolute_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty string")
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = Path(os.path.abspath(os.fspath(raw)))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            status = current.lstat()
        except OSError as exc:
            raise ValueError(f"{label} path is unavailable: {current}") from exc
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} path cannot contain a symlink: {current}")
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        raise ValueError(f"{label} must be a non-empty ordinary file")
    return path


def _json_file(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


@dataclass(frozen=True)
class ArtifactSnapshot:
    """Immutable identity captured after a complete SHA-256 pass."""

    label: str
    path: str
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return self.device, self.inode, self.size, self.mtime_ns, self.ctime_ns


@dataclass(frozen=True)
class VerifiedRunArtifacts:
    """Identity-registered artifact capability attached to a run binding."""

    contract_sha256: str
    initial_renderer_state_sha256: str
    descriptors: tuple[ArtifactSnapshot, ...]
    payload_files: tuple[ArtifactSnapshot, ...]
    _seal: object = field(default=None, repr=False, compare=False)

    def is_verified(self) -> bool:
        reference = _VERIFIED.get(id(self))
        return reference is not None and reference() is self

    def validate_current(self) -> None:
        if not self.is_verified():
            raise RuntimeError("run artifact capability is not verified")
        for snapshot in (*self.descriptors, *self.payload_files):
            try:
                path = _ordinary_absolute_file(snapshot.path, label=snapshot.label)
                current = _identity(path.stat())
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"verified run artifact disappeared: {snapshot.label}") from exc
            if current != snapshot.identity:
                raise RuntimeError(f"verified run artifact changed: {snapshot.label}")

    def validate_initial_renderer(self, module: Any) -> None:
        """Require the loaded renderer to match the registered initial state."""
        self.validate_current()
        if module_state_sha256(module) != self.initial_renderer_state_sha256:
            raise ValueError("loaded renderer state differs from the run contract")

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": "repldm.verified_run_artifacts.v1",
            "contract_sha256": self.contract_sha256,
            "initial_renderer_state_sha256": self.initial_renderer_state_sha256,
            "descriptors": [
                {"label": item.label, "path": item.path, "sha256": item.sha256}
                for item in self.descriptors
            ],
            "payload_files": [
                {"label": item.label, "path": item.path, "sha256": item.sha256}
                for item in self.payload_files
            ],
        }


def _snapshot(label: str, path: Path, expected_sha256: str) -> ArtifactSnapshot:
    if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} sha256 must be a lowercase SHA-256 digest")
    actual, status = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs from the run contract")
    identity = _identity(status)
    return ArtifactSnapshot(label, str(path), actual, *identity)


def _verify_file_manifest(
    descriptor: ArtifactSnapshot,
    *,
    label: str,
) -> tuple[ArtifactSnapshot, ...]:
    payload = _json_file(Path(descriptor.path), label=label)
    if payload.get("schema") != FILE_MANIFEST_SCHEMA or set(payload) != {"schema", "files"}:
        raise ValueError(f"{label} has an unsupported file-manifest schema")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label} must list at least one file")
    result: list[ArtifactSnapshot] = []
    seen: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError(f"{label} file entry {index} is invalid")
        path = _ordinary_absolute_file(item.get("path"), label=f"{label}[{index}]")
        if str(path) in seen:
            raise ValueError(f"{label} contains duplicate file paths")
        seen.add(str(path))
        size = item.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"{label} file entry {index} has an invalid byte count")
        snap = _snapshot(f"{label}[{index}]", path, item.get("sha256"))
        if snap.size != size:
            raise ValueError(f"{label} file entry {index} byte count differs")
        result.append(snap)
    return tuple(result)


def _verify_initial_state(
    descriptor: ArtifactSnapshot,
    *,
    expected_state_hash: str,
) -> tuple[ArtifactSnapshot, ...]:
    payload = _json_file(Path(descriptor.path), label="initial_renderer_state")
    required = {"schema", "renderer_state_sha256", "checkpoint"}
    if payload.get("schema") != INITIAL_STATE_SCHEMA or set(payload) != required:
        raise ValueError("initial_renderer_state manifest is invalid")
    if payload.get("renderer_state_sha256") != expected_state_hash:
        raise ValueError("initial renderer state hash differs from the run contract")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"path", "sha256", "bytes"}:
        raise ValueError("initial renderer checkpoint entry is invalid")
    path = _ordinary_absolute_file(
        checkpoint.get("path"), label="initial renderer checkpoint"
    )
    size = checkpoint.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("initial renderer checkpoint byte count is invalid")
    snap = _snapshot(
        "initial renderer checkpoint", path, checkpoint.get("sha256")
    )
    if snap.size != size:
        raise ValueError("initial renderer checkpoint byte count differs")
    return (snap,)


def _verify_bound_file(
    value: Mapping[str, Any], *, label: str
) -> ArtifactSnapshot:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
        raise ValueError(f"{label} file binding is invalid")
    path = _ordinary_absolute_file(value.get("path"), label=label)
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label} byte count is invalid")
    snapshot = _snapshot(label, path, value.get("sha256"))
    if snapshot.size != size:
        raise ValueError(f"{label} byte count differs")
    return snapshot


def _register(value: VerifiedRunArtifacts) -> None:
    identity = id(value)

    def remove(reference: Any, *, identity: int = identity) -> None:
        if _VERIFIED.get(identity) is reference:
            _VERIFIED.pop(identity, None)

    _VERIFIED[identity] = weakref.ref(value, remove)


def verify_run_artifacts(
    contract: Mapping[str, Any],
    *,
    contract_sha256: str,
    selected_manifest_path: Path,
    selected_manifest_sha256: str,
) -> VerifiedRunArtifacts:
    """Resolve and hash every artifact declared by a complete run contract."""
    if not isinstance(contract_sha256, str) or _SHA256_RE.fullmatch(contract_sha256) is None:
        raise ValueError("contract_sha256 must be a lowercase SHA-256 digest")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("run contract artifacts must be a mapping")
    method = contract.get("method")
    gated = method in {"opd", "search_distill", "dpo", "rl"}
    f0_witness = method == "f0"
    expected_names = set(required_run_artifacts(method))
    supplied_names = set(artifacts)
    if supplied_names != expected_names:
        missing = sorted(expected_names.difference(supplied_names))
        extra = sorted(supplied_names.difference(expected_names))
        raise ValueError(
            "run artifact descriptors differ from the registered schema: "
            f"missing={missing}, extra={extra}"
        )
    descriptors: dict[str, ArtifactSnapshot] = {}
    seen_paths: set[str] = set()
    for name, entry in artifacts.items():
        if not isinstance(name, str) or _ARTIFACT_NAME_RE.fullmatch(name) is None:
            raise ValueError("run artifact names must be stable lowercase identifiers")
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"run artifact {name} must contain only path and sha256")
        path = _ordinary_absolute_file(entry.get("path"), label=f"run artifact {name}")
        if str(path) in seen_paths:
            raise ValueError("run artifact descriptors contain duplicate paths")
        seen_paths.add(str(path))
        descriptors[name] = _snapshot(name, path, entry.get("sha256"))

    direct_hashes = {
        "data_manifest": "data_manifest_sha256",
        "prompt_manifest": "prompt_manifest_sha256",
        "reward_config": "reward_config_sha256",
        "reward_assets_manifest": "reward_asset_manifest_sha256",
        "model_assets_manifest": "model_asset_manifest_sha256",
        "basis_provider_config": "basis_provider_config_sha256",
        "calibration": "calibration_artifact_sha256",
        "renderer_frame_contract": "renderer_frame_contract_artifact_sha256",
        "initial_renderer_state": "initial_renderer_state_manifest_sha256",
    }
    if gated:
        direct_hashes.update(
            {
                "f0_gate": "f0_gate_sha256",
                "opsd_teacher_state": "opsd_teacher_state_manifest_sha256",
                "reward_statistics": "reward_statistics_sha256",
                "cohort_manifest": "cohort_manifest_sha256",
            }
        )
    if f0_witness:
        direct_hashes.update(
            {
                "witness_config": "witness_config_sha256",
                "witness_assets_manifest": "witness_asset_manifest_sha256",
            }
        )
    for name, field_name in direct_hashes.items():
        if descriptors[name].sha256 != contract.get(field_name):
            raise ValueError(f"{name} hash does not match {field_name}")

    selected = Path(selected_manifest_path)
    if (
        Path(descriptors["data_manifest"].path) != selected
        or descriptors["data_manifest"].sha256 != selected_manifest_sha256
        or contract.get("data_manifest_sha256") != selected_manifest_sha256
    ):
        raise ValueError("data_manifest must be the authorized selected-payload manifest")

    calibration = _json_file(
        Path(descriptors["calibration"].path), label="calibration artifact"
    )
    if calibration.get("calibration_hash") != contract.get("calibration_hash"):
        raise ValueError("calibration artifact does not match calibration_hash")
    frame = _json_file(
        Path(descriptors["renderer_frame_contract"].path),
        label="renderer frame contract artifact",
    )
    if (
        frame.get("renderer_frame_contract_hash")
        != contract.get("renderer_frame_contract_hash")
        or frame.get("action_contract_hash") != contract.get("action_contract_hash")
    ):
        raise ValueError("renderer frame artifact differs from the run contract")
    provider = _json_file(
        Path(descriptors["basis_provider_config"].path),
        label="basis provider config",
    )
    if provider.get("basis_provider_contract_hash") != contract.get(
        "basis_provider_contract_hash"
    ):
        raise ValueError("basis provider config differs from the run contract")

    payload_files: list[ArtifactSnapshot] = []
    payload_files.extend(
        _verify_file_manifest(
            descriptors["model_assets_manifest"], label="model_assets_manifest"
        )
    )
    if gated:
        from .gates import (
            validate_f0_gate,
            validate_opsd_teacher_state,
            validate_reward_statistics,
            validate_training_cohort,
        )

        reward_statistics = _json_file(
            Path(descriptors["reward_statistics"].path),
            label="reward_statistics",
        )
        validate_reward_statistics(reward_statistics, contract=contract)
        teacher_state = _json_file(
            Path(descriptors["opsd_teacher_state"].path),
            label="opsd_teacher_state",
        )
        _teacher, teacher_checkpoint = validate_opsd_teacher_state(
            teacher_state, contract=contract
        )
        payload_files.append(
            _verify_bound_file(teacher_checkpoint, label="OPSD teacher checkpoint")
        )
        f0_gate = _json_file(Path(descriptors["f0_gate"].path), label="f0_gate")
        _gate, evidence = validate_f0_gate(f0_gate, contract=contract)
        payload_files.extend(
            _verify_bound_file(item, label=f"F0 evidence {index}")
            for index, item in enumerate(evidence)
        )
        cohort = _json_file(
            Path(descriptors["cohort_manifest"].path), label="cohort_manifest"
        )
        validate_training_cohort(cohort, contract=contract)
    payload_files.extend(
        _verify_file_manifest(
            descriptors["reward_assets_manifest"], label="reward_assets_manifest"
        )
    )
    if f0_witness:
        payload_files.extend(
            _verify_file_manifest(
                descriptors["witness_assets_manifest"],
                label="witness_assets_manifest",
            )
        )
    payload_files.extend(
        _verify_initial_state(
            descriptors["initial_renderer_state"],
            expected_state_hash=str(contract.get("initial_renderer_state_sha256")),
        )
    )
    payload_paths = [item.path for item in payload_files]
    if len(payload_paths) != len(set(payload_paths)):
        raise ValueError("nested run manifests contain duplicate payload paths")
    descriptor_paths = {item.path for item in descriptors.values()}
    if descriptor_paths.intersection(payload_paths):
        raise ValueError("run descriptor and nested payload paths must be distinct")

    result = VerifiedRunArtifacts(
        contract_sha256=contract_sha256,
        initial_renderer_state_sha256=str(
            contract.get("initial_renderer_state_sha256")
        ),
        descriptors=tuple(descriptors[name] for name in sorted(descriptors)),
        payload_files=tuple(sorted(payload_files, key=lambda item: (item.label, item.path))),
        _seal=object(),
    )
    _register(result)
    return result


def require_verified_run_artifacts(value: Any) -> VerifiedRunArtifacts:
    if type(value) is not VerifiedRunArtifacts or not value.is_verified():
        raise TypeError("verified run artifacts are required")
    return value


__all__ = [
    "FILE_MANIFEST_SCHEMA",
    "F0_WITNESS_RUN_ARTIFACTS",
    "INITIAL_STATE_SCHEMA",
    "GATED_RUN_ARTIFACTS",
    "REQUIRED_RUN_ARTIFACTS",
    "ArtifactSnapshot",
    "VerifiedRunArtifacts",
    "module_state_sha256",
    "require_verified_run_artifacts",
    "required_run_artifacts",
    "verify_run_artifacts",
]
