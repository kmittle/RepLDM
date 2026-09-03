"""One fail-closed construction path for latent-renderer training runtimes.

Training methods must receive their SDXL, VAE, scheduler, basis provider, and
reward model from this module.  The factory consumes an already validated run
binding, resolves only files covered by its artifact capability, and performs
all third-party construction while network access and credential variables are
disabled.  Heavy loaders are dependency-injected so the security and lifecycle
contract can be tested on CPU without loading SDXL.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import io
import importlib
from importlib import metadata as importlib_metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import stat
import sys
import tempfile
from types import ModuleType
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence
from unittest import mock

import torch
from torch import nn


REWARD_RUNTIME_SCHEMA = "repldm.imagereward_runtime.v1"
WITNESS_RUNTIME_SCHEMA = "repldm.topiq_nr_tensor_witness_runtime.v1"
BASIS_RUNTIME_SCHEMA = "repldm.latent_renderer_basis_provider.v1"
RENDERER_RUNTIME_SCHEMA = "repldm.euler_native_frame_runtime.v1"
FILE_MANIFEST_SCHEMA = "repldm.file_manifest.v1"

_OFFLINE_FLAGS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
}
_PRIVATE_ENVIRONMENT_KEYS = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "WANDB_API_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)


@dataclass(frozen=True)
class BoundAsset:
    """A file identity copied from ``VerifiedRunArtifacts``."""

    label: str
    path: Path
    sha256: str
    size: int
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ModelStage:
    """Live capability returned by the pinned SDXL snapshot backend."""

    backend: Any
    record: Mapping[str, Any]
    manifest_sha256: str


@dataclass(frozen=True)
class RewardStage:
    """Private, immutable copy of the complete ImageReward load closure."""

    root: Path
    checkpoint: Path
    med_config: Path
    source_root: Path
    tokenizer_root: Path


@dataclass(frozen=True)
class WitnessStage:
    """Private copy of the complete TOPIQ-NR load closure."""

    root: Path
    checkpoint: Path
    backbone: Path
    source_root: Path
    package_versions: Mapping[str, str]


@dataclass(frozen=True)
class RendererBundle:
    """Independent renderer roles and their verified checkpoint identities."""

    renderer: nn.Module
    reference_renderer: nn.Module
    initial_checkpoint_provenance: Any
    teacher_renderer: Optional[nn.Module] = None
    teacher_checkpoint_provenance: Any = None


@dataclass(frozen=True)
class RuntimeInfrastructure:
    """The one ledger/executor/store object graph owned by a runtime."""

    ledger: Any
    operation_executor: Any
    rollout_store: Any


@dataclass(frozen=True)
class RuntimeLoadSpec:
    """Runtime-only placement; precision is intentionally fixed for run one."""

    device: str | torch.device
    batch_size: int
    model_dtype: torch.dtype = torch.float16
    vae_dtype: torch.dtype = torch.float32
    reward_dtype: torch.dtype = torch.float32
    vae_scaling_factor: float = 0.13025
    height: int = 1024
    width: int = 1024
    staging_parent: Optional[Path] = None

    def __post_init__(self) -> None:
        try:
            normalized_device = str(torch.device(self.device))
        except (TypeError, RuntimeError) as exc:
            raise ValueError("device must be a valid torch device") from exc
        object.__setattr__(self, "device", normalized_device)
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be a plain integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.model_dtype is not torch.float16:
            raise ValueError("the pinned SDXL runtime requires float16")
        if self.vae_dtype is not torch.float32:
            raise ValueError("the force-upcast VAE must remain float32")
        if self.reward_dtype is not torch.float32:
            raise ValueError("the frozen ImageReward runtime requires float32")
        if self.vae_scaling_factor != 0.13025:
            raise ValueError("the frozen SDXL VAE scaling factor is 0.13025")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value % 8
            for value in (self.height, self.width)
        ):
            raise ValueError("height and width must be positive multiples of eight")
        if self.staging_parent is not None:
            parent = Path(self.staging_parent)
            if not parent.is_absolute() or not parent.is_dir() or parent.is_symlink():
                raise ValueError("staging_parent must be an existing absolute directory")

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "RuntimeLoadSpec":
        if not isinstance(contract, Mapping):
            raise TypeError("contract must be a mapping")
        runtime = contract.get("runtime")
        required = {
            "device",
            "batch_size",
            "model_dtype",
            "vae_dtype",
            "reward_dtype",
            "vae_scaling_factor",
            "height",
            "width",
        }
        if not isinstance(runtime, Mapping) or set(runtime) != required:
            raise ValueError("run contract runtime fields differ from the registered schema")
        dtypes = {
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            model_dtype = dtypes[runtime["model_dtype"]]
            vae_dtype = dtypes[runtime["vae_dtype"]]
            reward_dtype = dtypes[runtime["reward_dtype"]]
        except (KeyError, TypeError) as exc:
            raise ValueError("run contract runtime has an unsupported precision") from exc
        return cls(
            device=runtime["device"],
            batch_size=runtime["batch_size"],
            model_dtype=model_dtype,
            vae_dtype=vae_dtype,
            reward_dtype=reward_dtype,
            vae_scaling_factor=runtime["vae_scaling_factor"],
            height=runtime["height"],
            width=runtime["width"],
        )

    def contract_record(self) -> dict[str, Any]:
        names = {
            torch.float16: "float16",
            torch.float32: "float32",
        }
        return {
            "device": str(torch.device(self.device)),
            "batch_size": self.batch_size,
            "model_dtype": names[self.model_dtype],
            "vae_dtype": names[self.vae_dtype],
            "reward_dtype": names[self.reward_dtype],
            "vae_scaling_factor": self.vae_scaling_factor,
            "height": self.height,
            "width": self.width,
        }


@dataclass(frozen=True)
class RuntimeFactoryDependencies:
    """Injectable loaders; production callers normally use ``defaults()``."""

    validate_binding: Callable[[Any], Any]
    stage_model: Callable[[Any, Sequence[BoundAsset], Optional[Path]], ModelStage]
    load_pipeline: Callable[[ModelStage, torch.device, torch.dtype, torch.dtype], Any]
    cleanup_model: Callable[[ModelStage], None]
    stage_reward: Callable[[Mapping[str, Any], Sequence[BoundAsset], Optional[Path]], RewardStage]
    load_reward: Callable[[RewardStage, torch.device, torch.dtype], nn.Module]
    cleanup_reward: Callable[[RewardStage], None]
    load_renderers: Callable[
        [Any, Mapping[str, BoundAsset], Sequence[BoundAsset], torch.device],
        RendererBundle,
    ]
    make_infrastructure: Callable[[Any], RuntimeInfrastructure]
    make_basis_provider: Callable[[Any, Mapping[str, Any], int, bool], Any]
    wrap_reward: Callable[[nn.Module], nn.Module]
    make_adapter: Callable[[Any, Any, nn.Module, Any, Mapping[str, Any]], Any]
    reward_preprocess_sha256: str
    stage_witness: Optional[
        Callable[
            [Mapping[str, Any], Sequence[BoundAsset], Optional[Path]],
            WitnessStage,
        ]
    ] = None
    load_witness: Optional[
        Callable[[WitnessStage, torch.device, torch.dtype], nn.Module]
    ] = None
    cleanup_witness: Optional[Callable[[WitnessStage], None]] = None
    wrap_witness: Optional[Callable[[nn.Module, Any], nn.Module]] = None
    witness_preprocess_sha256: Optional[str] = None

    @classmethod
    def defaults(cls) -> "RuntimeFactoryDependencies":
        from .authorization import require_authorization_binding
        from .rewards import (
            IMAGE_REWARD_PREPROCESS_SHA256,
            ImageRewardTensorAdapter,
        )
        from .witnesses import (
            TOPIQ_NR_PREPROCESS_SHA256,
            TopiqNrTensorWitness,
        )

        return cls(
            validate_binding=require_authorization_binding,
            stage_model=_stage_default_model,
            load_pipeline=_load_default_pipeline,
            cleanup_model=_cleanup_default_model,
            stage_reward=_stage_default_reward,
            load_reward=_load_default_reward,
            cleanup_reward=_cleanup_default_reward,
            load_renderers=_load_default_renderers,
            make_infrastructure=_make_default_infrastructure,
            make_basis_provider=_make_default_basis_provider,
            wrap_reward=ImageRewardTensorAdapter,
            make_adapter=_make_default_adapter,
            reward_preprocess_sha256=IMAGE_REWARD_PREPROCESS_SHA256,
            stage_witness=_stage_default_witness,
            load_witness=_load_default_witness,
            cleanup_witness=_cleanup_default_witness,
            wrap_witness=lambda model, executor: TopiqNrTensorWitness(
                model, operation_executor=executor
            ),
            witness_preprocess_sha256=TOPIQ_NR_PREPROCESS_SHA256,
        )


@dataclass
class TrainingRuntime:
    """Loaded formal runtime with deterministic reverse-order cleanup."""

    binding: Any
    renderer: nn.Module
    reference_renderer: nn.Module
    teacher_renderer: Optional[nn.Module]
    pipeline: Any
    vae: nn.Module
    scheduler: Any
    basis_provider: Any
    reward_model: nn.Module
    witness_model: Optional[nn.Module]
    adapter: Any
    ledger: Any
    operation_executor: Any
    rollout_store: Any
    selected_records: tuple[Mapping[str, Any], ...]
    reward_statistics: Optional[Mapping[str, Any]]
    f0_gate: Optional[Mapping[str, Any]]
    opsd_teacher_state: Optional[Mapping[str, Any]]
    training_cohort: Optional[Mapping[str, Any]]
    initial_checkpoint_provenance: Any
    teacher_checkpoint_provenance: Any
    provenance: Mapping[str, Any]
    _model_stage: ModelStage = field(repr=False)
    _reward_stage: RewardStage = field(repr=False)
    _witness_stage: Optional[WitnessStage] = field(repr=False)
    _cleanup_model: Callable[[ModelStage], None] = field(repr=False)
    _cleanup_reward: Callable[[RewardStage], None] = field(repr=False)
    _cleanup_witness: Optional[Callable[[WitnessStage], None]] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for callback, value in (
            (self._cleanup_reward, self._reward_stage),
            (self._cleanup_model, self._model_stage),
            (self._cleanup_witness, self._witness_stage),
        ):
            if callback is None or value is None:
                continue
            try:
                callback(value)
            except BaseException as exc:  # cleanup must attempt both stages
                errors.append(exc)
        if errors:
            failure = RuntimeError("runtime cleanup failed")
            failure.cleanup_errors = tuple(errors)  # type: ignore[attr-defined]
            raise failure from errors[0]

    def __enter__(self) -> "TrainingRuntime":
        if self._closed:
            raise RuntimeError("runtime is already closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        """Close private stages without hiding an exception from the body."""
        if exc is None:
            self.close()
            return False
        try:
            self.close()
        except BaseException as cleanup_error:
            # The body failure is the actionable cause.  Keep cleanup details
            # available to callers without replacing that original exception.
            try:
                setattr(exc, "_runtime_cleanup_error", cleanup_error)
            except BaseException:
                pass
        return False


def _snapshot_identity(value: Any) -> tuple[int, int, int, int, int]:
    identity = getattr(value, "identity", None)
    if not isinstance(identity, tuple) or len(identity) != 5:
        raise TypeError("verified artifact snapshot has no stable identity")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in identity):
        raise TypeError("verified artifact identity must contain integers")
    return identity


def _bound_asset(value: Any) -> BoundAsset:
    label = getattr(value, "label", None)
    path = getattr(value, "path", None)
    digest = getattr(value, "sha256", None)
    size = getattr(value, "size", None)
    if not isinstance(label, str) or not label:
        raise TypeError("verified artifact label is invalid")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise TypeError("verified artifact path is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise TypeError("verified artifact digest is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise TypeError("verified artifact size is invalid")
    return BoundAsset(label, Path(path), digest, size, _snapshot_identity(value))


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_asset(asset: BoundAsset) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(asset.path, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open verified runtime asset: {asset.label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _stat_identity(before) != asset.identity:
            raise RuntimeError(f"verified runtime asset identity changed: {asset.label}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(after) != asset.identity or digest.hexdigest() != asset.sha256:
            raise RuntimeError(f"verified runtime asset bytes changed: {asset.label}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_asset(asset: BoundAsset) -> dict[str, Any]:
    try:
        value = json.loads(_read_asset(asset).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{asset.label} must contain one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{asset.label} must contain one JSON object")
    return value


def _artifact_inventory(binding: Any) -> tuple[dict[str, BoundAsset], tuple[BoundAsset, ...]]:
    artifacts = getattr(binding, "run_artifacts", None)
    if artifacts is None or not callable(getattr(artifacts, "validate_current", None)):
        raise TypeError("run binding has no verified artifact capability")
    artifacts.validate_current()
    descriptors = tuple(_bound_asset(value) for value in artifacts.descriptors)
    payload = tuple(_bound_asset(value) for value in artifacts.payload_files)
    by_label = {item.label: item for item in descriptors}
    if len(by_label) != len(descriptors):
        raise RuntimeError("verified artifact descriptor labels are not unique")
    contract = getattr(binding, "contract", None)
    if not isinstance(contract, Mapping):
        raise TypeError("run binding has no canonical contract mapping")
    required = {
        "data_manifest",
        "model_assets_manifest",
        "reward_assets_manifest",
        "reward_config",
        "basis_provider_config",
        "calibration",
        "renderer_frame_contract",
        "initial_renderer_state",
    }
    if contract.get("method") == "f0":
        required.update({"witness_config", "witness_assets_manifest"})
    if contract.get("method") in {"opd", "search_distill", "dpo", "rl"}:
        required.update(
            {
                "f0_gate",
                "opsd_teacher_state",
                "reward_statistics",
                "cohort_manifest",
            }
        )
    if not required.issubset(by_label):
        raise RuntimeError("runtime artifact descriptors are incomplete")
    return by_label, payload


def _manifest_assets(
    descriptor: BoundAsset,
    payload: Sequence[BoundAsset],
) -> tuple[BoundAsset, ...]:
    value = _json_asset(descriptor)
    if value.get("schema") != FILE_MANIFEST_SCHEMA or set(value) != {"schema", "files"}:
        raise ValueError(f"{descriptor.label} has an unsupported schema")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{descriptor.label} has no files")
    available = {str(item.path): item for item in payload}
    selected: list[BoundAsset] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "bytes"}:
            raise ValueError(f"{descriptor.label} file row {index} is invalid")
        path = row.get("path")
        if not isinstance(path, str) or path in seen:
            raise ValueError(f"{descriptor.label} contains a duplicate or invalid path")
        seen.add(path)
        asset = available.get(path)
        if asset is None:
            raise RuntimeError(f"{descriptor.label} references an unverified payload")
        if row.get("sha256") != asset.sha256 or row.get("bytes") != asset.size:
            raise RuntimeError(f"{descriptor.label} payload binding changed")
        selected.append(asset)
    prefix = descriptor.label + "["
    labelled = {str(item.path) for item in payload if item.label.startswith(prefix)}
    if labelled != seen:
        raise RuntimeError(f"{descriptor.label} verified payload inventory differs")
    return tuple(selected)


def _require_contract(binding: Any) -> dict[str, Any]:
    value = getattr(binding, "contract", None)
    if not isinstance(value, Mapping):
        raise TypeError("run binding has no canonical contract mapping")
    contract = dict(value)
    required = {
        "nfe",
        "prediction_type",
        "do_classifier_free_guidance",
        "guidance_scale",
        "guidance_rescale",
        "decision_indices",
        "basis_provider_contract_hash",
        "selected_rows",
        "runtime",
    }
    if not required.issubset(contract):
        raise RuntimeError("run contract lacks mandatory runtime fields")
    return contract


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _load_selected_records(
    descriptor: BoundAsset,
    *,
    expected_rows: int,
) -> tuple[Mapping[str, Any], ...]:
    if expected_rows <= 0:
        raise ValueError("selected_rows must be positive")
    try:
        lines = _read_asset(descriptor).decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("selected payload manifest is not UTF-8") from exc
    records: list[Mapping[str, Any]] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError("selected payload manifest contains a blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"selected payload manifest has invalid JSON at row {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError("selected payload rows must be objects")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("selected payload IDs must be unique non-empty strings")
        identifiers.add(identifier)
        records.append(_freeze_json(row))
    if len(records) != expected_rows:
        raise ValueError("selected payload row count differs from the run contract")
    return tuple(records)


def _blocked_network(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("network access is disabled during formal runtime loading")


@contextmanager
def _offline_loading() -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in (*_OFFLINE_FLAGS, *_PRIVATE_ENVIRONMENT_KEYS)}
    try:
        for key in _PRIVATE_ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ.update(_OFFLINE_FLAGS)
        with mock.patch.object(socket, "create_connection", _blocked_network), mock.patch.object(
            socket.socket, "connect", _blocked_network
        ):
            yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _module_dtype(module: nn.Module, label: str) -> torch.dtype:
    dtypes = {
        value.dtype
        for value in (*tuple(module.parameters()), *tuple(module.buffers()))
        if value.is_floating_point()
    }
    if not dtypes:
        raise ValueError(f"{label} has no floating-point state")
    if len(dtypes) != 1:
        raise ValueError(f"{label} has mixed floating-point precision")
    return next(iter(dtypes))


def _validate_loaded_components(
    pipeline: Any,
    reward: nn.Module,
    witness: Optional[nn.Module],
    spec: RuntimeLoadSpec,
    contract: Mapping[str, Any],
) -> None:
    for name in ("unet", "vae", "scheduler"):
        if not hasattr(pipeline, name):
            raise TypeError(f"loaded SDXL pipeline has no {name}")
    if not isinstance(pipeline.unet, nn.Module) or not isinstance(pipeline.vae, nn.Module):
        raise TypeError("loaded SDXL pipeline modules are invalid")
    if _module_dtype(pipeline.unet, "SDXL U-Net") is not spec.model_dtype:
        raise ValueError("SDXL U-Net precision differs from the runtime spec")
    if _module_dtype(pipeline.vae, "SDXL VAE") is not torch.float32:
        raise ValueError("SDXL VAE was not kept permanently in float32")
    if bool(getattr(getattr(pipeline.vae, "config", None), "force_upcast", False)) is not True:
        raise ValueError("pinned SDXL VAE must declare force_upcast=true")
    scheduler = pipeline.scheduler
    if type(scheduler).__name__ != "EulerDiscreteScheduler":
        raise ValueError("pinned SDXL scheduler must be exactly EulerDiscreteScheduler")
    prediction_type = getattr(getattr(scheduler, "config", None), "prediction_type", None)
    if prediction_type != contract["prediction_type"] or prediction_type != "epsilon":
        raise ValueError("Euler prediction_type differs from the run contract")
    scheduler.set_timesteps(int(contract["nfe"]), device=torch.device(spec.device))
    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(timesteps, torch.Tensor) or len(timesteps) != int(contract["nfe"]):
        raise ValueError("Euler scheduler did not expose the registered timesteps")
    if not isinstance(sigmas, torch.Tensor) or len(sigmas) != int(contract["nfe"]) + 1:
        raise ValueError("Euler scheduler did not expose the registered sigma pairs")
    if not isinstance(reward, nn.Module):
        raise TypeError("ImageReward loader did not return a module")
    if _module_dtype(reward, "ImageReward") is not spec.reward_dtype:
        raise ValueError("ImageReward precision differs from the runtime spec")
    frozen_modules = [
        (pipeline.unet, "SDXL U-Net"),
        (pipeline.vae, "SDXL VAE"),
        (reward, "ImageReward"),
    ]
    if contract.get("method") == "f0":
        from .witnesses import TOPIQ_NR_MODEL_ID, TOPIQ_NR_ROLE

        if not isinstance(witness, nn.Module):
            raise TypeError("F0 runtime omitted its TOPIQ-NR witness")
        if _module_dtype(witness, "TOPIQ-NR") is not torch.float32:
            raise ValueError("TOPIQ-NR precision must remain float32")
        if any(module.training for module in witness.modules()):
            raise ValueError("TOPIQ-NR must remain in eval mode")
        if getattr(witness, "model_id", None) != TOPIQ_NR_MODEL_ID:
            raise ValueError("TOPIQ-NR model identity differs from the registered witness")
        if getattr(witness, "role", None) != TOPIQ_NR_ROLE:
            raise ValueError("TOPIQ-NR role differs from the registered witness")
        if getattr(witness, "preprocess_sha256", None) != contract.get(
            "witness_preprocess_sha256"
        ):
            raise ValueError("TOPIQ-NR preprocessing differs from the run contract")
        frozen_modules.append((witness, "TOPIQ-NR"))
    elif witness is not None:
        raise RuntimeError("training runtime cannot expose the independent F0 witness")
    for module, label in frozen_modules:
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError(f"{label} must be frozen")


def _require_independent_matching_renderers(
    renderer: nn.Module,
    reference: nn.Module,
    *,
    device: torch.device,
) -> None:
    if not isinstance(renderer, nn.Module) or not isinstance(reference, nn.Module):
        raise TypeError("renderer loaders must return torch modules")
    if renderer is reference:
        raise RuntimeError("behavior and reference renderers must be distinct objects")
    renderer_parameters = tuple(renderer.parameters())
    reference_parameters = tuple(reference.parameters())
    if not renderer_parameters or not any(
        parameter.requires_grad for parameter in renderer_parameters
    ):
        raise ValueError("behavior renderer has no trainable parameters")
    if any(parameter.requires_grad for parameter in reference_parameters):
        raise ValueError("reference renderer must be frozen")
    for module, label in ((renderer, "behavior"), (reference, "reference")):
        if _module_dtype(module, f"{label} renderer") is not torch.float32:
            raise ValueError(f"{label} renderer must use float32")
        if any(parameter.device != device for parameter in module.parameters()):
            raise ValueError(f"{label} renderer is on the wrong device")
    left = renderer.state_dict()
    right = reference.state_dict()
    if set(left) != set(right):
        raise ValueError("behavior and reference renderer states have different keys")
    for name in left:
        first = left[name]
        second = right[name]
        if (
            not isinstance(first, torch.Tensor)
            or not isinstance(second, torch.Tensor)
            or first.shape != second.shape
            or first.dtype != second.dtype
            or not torch.equal(first.detach().cpu(), second.detach().cpu())
        ):
            raise ValueError("behavior and reference renderer states differ at C0")
    for first, second in zip(renderer.parameters(), reference.parameters()):
        if first is second:
            raise RuntimeError("behavior and reference renderer parameters alias")
        if first.numel() and second.numel():
            first_storage = first.untyped_storage().data_ptr()
            second_storage = second.untyped_storage().data_ptr()
            if first.device == second.device and first_storage == second_storage:
                raise RuntimeError("behavior and reference renderer storage aliases")


def _module_state_tensors(module: nn.Module) -> tuple[torch.Tensor, ...]:
    return (*tuple(module.parameters()), *tuple(module.buffers()))


def _require_independent_frozen_teacher(
    teacher: nn.Module,
    renderer: nn.Module,
    reference: nn.Module,
    *,
    binding: Any,
    contract: Mapping[str, Any],
    device: torch.device,
) -> str:
    """Validate the immutable T_OPSD role after every loader has completed."""
    from .artifacts import module_state_sha256
    from .contracts import contract_hash

    if not all(isinstance(value, nn.Module) for value in (teacher, renderer, reference)):
        raise TypeError("T_OPSD and both runtime renderers must be torch modules")
    if teacher is renderer or teacher is reference:
        raise RuntimeError("T_OPSD must be a distinct renderer object")
    if type(teacher) is not type(renderer) or type(teacher) is not type(reference):
        raise TypeError("T_OPSD must use the same renderer implementation as the run")
    if any(module.training for module in teacher.modules()):
        raise ValueError("T_OPSD must remain in eval mode")
    parameters = tuple(teacher.parameters())
    if not parameters or any(parameter.requires_grad for parameter in parameters):
        raise ValueError("T_OPSD must be completely frozen")
    if _module_dtype(teacher, "T_OPSD") is not torch.float32:
        raise ValueError("T_OPSD must use float32")
    if any(value.device != device for value in _module_state_tensors(teacher)):
        raise ValueError("T_OPSD is on the wrong device")

    binding.validate_component(teacher)
    frame_hash = getattr(teacher, "frame_contract_hash", None)
    calibration_hash = getattr(teacher, "calibration_hash", None)
    action_contract = getattr(teacher, "contract", None)
    if frame_hash != getattr(renderer, "frame_contract_hash", None) or frame_hash != getattr(
        reference, "frame_contract_hash", None
    ):
        raise ValueError("T_OPSD renderer frame differs from student/reference")
    if calibration_hash != getattr(
        renderer, "calibration_hash", None
    ) or calibration_hash != getattr(reference, "calibration_hash", None):
        raise ValueError("T_OPSD calibration differs from student/reference")
    if action_contract is None or contract_hash(action_contract) != contract.get(
        "action_contract_hash"
    ):
        raise ValueError("T_OPSD action contract differs from the run contract")
    for other in (renderer, reference):
        for teacher_tensor in _module_state_tensors(teacher):
            for other_tensor in _module_state_tensors(other):
                if teacher_tensor is other_tensor:
                    raise RuntimeError("T_OPSD renderer state aliases another renderer")
                if (
                    teacher_tensor.numel()
                    and other_tensor.numel()
                    and teacher_tensor.device == other_tensor.device
                    and teacher_tensor.untyped_storage().data_ptr()
                    == other_tensor.untyped_storage().data_ptr()
                ):
                    raise RuntimeError("T_OPSD renderer storage aliases another renderer")

    state_hash = module_state_sha256(teacher)
    if state_hash != contract.get("opsd_teacher_renderer_sha256"):
        raise ValueError("loaded T_OPSD state differs from the run contract")
    return state_hash


def _require_teacher_checkpoint_provenance(
    value: Any, *, expected_state_hash: str
) -> Any:
    from .storage import CheckpointProvenance

    if not isinstance(value, CheckpointProvenance):
        raise TypeError("T_OPSD checkpoint provenance is missing or invalid")
    if value.renderer_state_sha256 != expected_state_hash:
        raise ValueError("T_OPSD checkpoint provenance has the wrong renderer state")
    value.validate_current()
    return value


class TrainingRuntimeFactory:
    """Build all shared runtime components from one validated binding."""

    def __init__(
        self,
        binding: Any,
        *,
        dependencies: Optional[RuntimeFactoryDependencies] = None,
    ) -> None:
        self.dependencies = dependencies or RuntimeFactoryDependencies.defaults()
        self.binding = self.dependencies.validate_binding(binding)
        if self.binding is not binding:
            raise RuntimeError("binding validator replaced the authorization capability")

    def load(
        self,
        spec: Optional[RuntimeLoadSpec] = None,
    ) -> TrainingRuntime:
        self.binding.validate_current()
        contract = _require_contract(self.binding)
        registered_spec = RuntimeLoadSpec.from_contract(contract)
        if spec is None:
            spec = registered_spec
        elif not isinstance(spec, RuntimeLoadSpec):
            raise TypeError("spec must be RuntimeLoadSpec or None")
        elif spec.contract_record() != registered_spec.contract_record():
            raise ValueError("explicit runtime spec differs from the run contract")
        descriptors, payload = _artifact_inventory(self.binding)
        model_assets = _manifest_assets(descriptors["model_assets_manifest"], payload)
        reward_assets = _manifest_assets(descriptors["reward_assets_manifest"], payload)
        reward_config = _json_asset(descriptors["reward_config"])
        witness_config: Optional[dict[str, Any]] = None
        witness_assets: tuple[BoundAsset, ...] = ()
        if contract.get("method") == "f0":
            witness_assets = _manifest_assets(
                descriptors["witness_assets_manifest"], payload
            )
            witness_config = _json_asset(descriptors["witness_config"])
        basis_config = _json_asset(descriptors["basis_provider_config"])
        selected_records = _load_selected_records(
            descriptors["data_manifest"], expected_rows=int(contract.get("selected_rows", 0))
        )
        reward_statistics = None
        f0_gate = None
        opsd_teacher_state = None
        training_cohort = None
        if contract.get("method") in {"opd", "search_distill", "dpo", "rl"}:
            from .gates import (
                validate_f0_gate,
                validate_opsd_teacher_state,
                validate_reward_statistics,
                validate_training_cohort,
            )

            reward_statistics = validate_reward_statistics(
                _json_asset(descriptors["reward_statistics"]), contract=contract
            )
            f0_gate, _f0_evidence = validate_f0_gate(
                _json_asset(descriptors["f0_gate"]), contract=contract
            )
            opsd_teacher_state, _teacher_checkpoint = validate_opsd_teacher_state(
                _json_asset(descriptors["opsd_teacher_state"]), contract=contract
            )
            training_cohort = validate_training_cohort(
                _json_asset(descriptors["cohort_manifest"]), contract=contract
            )
            if (
                f0_gate["f0_run_contract_sha256"]
                != opsd_teacher_state["f0_run_contract_sha256"]
            ):
                raise ValueError("F0 gate and OPSD teacher state name different F0 runs")
        if reward_config.get("schema") != REWARD_RUNTIME_SCHEMA:
            raise ValueError("reward_config has an unsupported runtime schema")
        if reward_config.get("implementation") != "ImageReward-v1.0":
            raise ValueError("the first formal runtime requires ImageReward-v1.0")
        if reward_config.get("dtype") != "float32":
            raise ValueError("ImageReward config must bind float32 precision")
        if reward_config.get("preprocess_sha256") != self.dependencies.reward_preprocess_sha256:
            raise ValueError("ImageReward preprocessing differs from the tensor adapter")
        if witness_config is not None:
            from .witnesses import TOPIQ_NR_MODEL_ID, TOPIQ_NR_ROLE

            required_witness_dependencies = (
                self.dependencies.stage_witness,
                self.dependencies.load_witness,
                self.dependencies.cleanup_witness,
                self.dependencies.wrap_witness,
            )
            if any(not callable(value) for value in required_witness_dependencies):
                raise RuntimeError("F0 runtime dependencies omit the TOPIQ-NR witness")
            if witness_config.get("schema") != WITNESS_RUNTIME_SCHEMA:
                raise ValueError("witness_config has an unsupported runtime schema")
            if witness_config.get("implementation") != TOPIQ_NR_MODEL_ID:
                raise ValueError("the F0 witness must be TOPIQ-NR")
            if witness_config.get("role") != TOPIQ_NR_ROLE:
                raise ValueError("TOPIQ-NR has the wrong runtime role")
            if witness_config.get("dtype") != "float32":
                raise ValueError("TOPIQ-NR config must bind float32 precision")
            preprocess_hash = witness_config.get("preprocess_sha256")
            if (
                preprocess_hash != contract.get("witness_preprocess_sha256")
                or preprocess_hash != self.dependencies.witness_preprocess_sha256
            ):
                raise ValueError("TOPIQ-NR preprocessing differs from the tensor witness")
        if basis_config.get("schema") != BASIS_RUNTIME_SCHEMA:
            raise ValueError("basis_provider_config has an unsupported runtime schema")
        if basis_config.get("basis_provider_contract_hash") != contract[
            "basis_provider_contract_hash"
        ]:
            raise ValueError("basis-provider config differs from the run contract")

        infrastructure = self.dependencies.make_infrastructure(self.binding)
        if infrastructure.operation_executor.ledger is not infrastructure.ledger:
            raise RuntimeError("operation executor does not own the runtime ledger")
        if getattr(infrastructure.operation_executor, "authorization_binding", None) is not self.binding:
            raise RuntimeError("operation executor does not share the runtime binding")
        if getattr(infrastructure.rollout_store, "authorization_binding", None) is not self.binding:
            raise RuntimeError("rollout store does not share the runtime binding")

        model_stage: Optional[ModelStage] = None
        reward_stage: Optional[RewardStage] = None
        witness_stage: Optional[WitnessStage] = None
        witness: Optional[nn.Module] = None
        try:
            with _offline_loading():
                if witness_config is not None:
                    stage_witness = self.dependencies.stage_witness
                    load_witness = self.dependencies.load_witness
                    wrap_witness = self.dependencies.wrap_witness
                    assert callable(stage_witness)
                    assert callable(load_witness)
                    assert callable(wrap_witness)
                    witness_stage = stage_witness(
                        witness_config, witness_assets, spec.staging_parent
                    )
                    raw_witness = load_witness(
                        witness_stage, torch.device(spec.device), torch.float32
                    )
                    witness = wrap_witness(
                        raw_witness, infrastructure.operation_executor
                    )
                    if (
                        getattr(witness, "operation_executor", None)
                        is not infrastructure.operation_executor
                    ):
                        raise RuntimeError(
                            "TOPIQ-NR does not share the runtime operation executor"
                        )
                model_stage = self.dependencies.stage_model(
                    self.binding, model_assets, spec.staging_parent
                )
                pipeline = self.dependencies.load_pipeline(
                    model_stage,
                    torch.device(spec.device),
                    spec.model_dtype,
                    spec.vae_dtype,
                )
                reward_stage = self.dependencies.stage_reward(
                    reward_config, reward_assets, spec.staging_parent
                )
                raw_reward = self.dependencies.load_reward(
                    reward_stage, torch.device(spec.device), spec.reward_dtype
                )
                reward = self.dependencies.wrap_reward(raw_reward)
                basis = self.dependencies.make_basis_provider(
                    pipeline.unet,
                    basis_config,
                    spec.batch_size,
                    bool(contract["do_classifier_free_guidance"]),
                )
                renderers = self.dependencies.load_renderers(
                    self.binding, descriptors, payload, torch.device(spec.device)
                )
                _require_independent_matching_renderers(
                    renderers.renderer,
                    renderers.reference_renderer,
                    device=torch.device(spec.device),
                )
                if opsd_teacher_state is None:
                    if (
                        renderers.teacher_renderer is not None
                        or renderers.teacher_checkpoint_provenance is not None
                    ):
                        raise RuntimeError("F0 runtime cannot expose a T_OPSD teacher")
                    teacher_renderer = None
                    teacher_checkpoint_provenance = None
                else:
                    if renderers.teacher_renderer is None:
                        raise RuntimeError("gated runtime omitted the frozen T_OPSD teacher")
                    teacher_state_hash = _require_independent_frozen_teacher(
                        renderers.teacher_renderer,
                        renderers.renderer,
                        renderers.reference_renderer,
                        binding=self.binding,
                        contract=contract,
                        device=torch.device(spec.device),
                    )
                    teacher_checkpoint_provenance = (
                        _require_teacher_checkpoint_provenance(
                            renderers.teacher_checkpoint_provenance,
                            expected_state_hash=teacher_state_hash,
                        )
                    )
                    teacher_renderer = renderers.teacher_renderer
                adapter = self.dependencies.make_adapter(
                    pipeline,
                    basis,
                    reward,
                    infrastructure.operation_executor,
                    contract,
                )
                if (
                    getattr(adapter, "operation_executor", None)
                    is not infrastructure.operation_executor
                ):
                    raise RuntimeError("adapter does not share the runtime operation executor")
                _validate_loaded_components(pipeline, reward, witness, spec, contract)
                self.binding.validate_initial_renderer(renderers.renderer)
                self.binding.validate_initial_renderer(renderers.reference_renderer)
                self.binding.validate_current()
        except BaseException as original_error:
            cleanup_errors: list[BaseException] = []
            for callback, value in (
                (self.dependencies.cleanup_reward, reward_stage),
                (self.dependencies.cleanup_model, model_stage),
                (self.dependencies.cleanup_witness, witness_stage),
            ):
                if callback is None or value is None:
                    continue
                try:
                    callback(value)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                failure = RuntimeError(
                    "runtime construction failed and private-stage cleanup also failed"
                )
                failure.original_error = original_error  # type: ignore[attr-defined]
                failure.cleanup_errors = tuple(cleanup_errors)  # type: ignore[attr-defined]
                raise failure from original_error
            raise

        contract_hash = getattr(self.binding, "contract_hash", None)
        provenance = {
            "schema": "repldm.loaded_training_runtime.v1",
            "run_contract_sha256": contract_hash,
            "model_asset_manifest_sha256": descriptors["model_assets_manifest"].sha256,
            "reward_asset_manifest_sha256": descriptors["reward_assets_manifest"].sha256,
            "reward_config_sha256": descriptors["reward_config"].sha256,
            "basis_provider_config_sha256": descriptors["basis_provider_config"].sha256,
            "model_dtype": str(spec.model_dtype),
            "vae_dtype": str(spec.vae_dtype),
            "reward_dtype": str(spec.reward_dtype),
            "vae_scaling_factor": spec.vae_scaling_factor,
            "device": str(torch.device(spec.device)),
            "batch_size": spec.batch_size,
            "height": spec.height,
            "width": spec.width,
        }
        if witness_config is not None:
            provenance.update(
                {
                    "witness_model": witness_config["implementation"],
                    "witness_role": witness_config["role"],
                    "witness_config_sha256": descriptors["witness_config"].sha256,
                    "witness_asset_manifest_sha256": descriptors[
                        "witness_assets_manifest"
                    ].sha256,
                    "witness_preprocess_sha256": witness_config[
                        "preprocess_sha256"
                    ],
                    "witness_package_versions": dict(
                        witness_config["package_versions"]
                    ),
                }
            )
        if f0_gate is not None:
            provenance.update(
                {
                    "f0_run_contract_sha256": f0_gate["f0_run_contract_sha256"],
                    "f0_gate_sha256": contract["f0_gate_sha256"],
                    "opsd_teacher_renderer_sha256": contract[
                        "opsd_teacher_renderer_sha256"
                    ],
                    "reward_statistics_sha256": contract["reward_statistics_sha256"],
                    "cohort_id": contract["cohort_id"],
                    "cohort_manifest_sha256": contract["cohort_manifest_sha256"],
                    "opsd_teacher_checkpoint_sha256": (
                        teacher_checkpoint_provenance.sha256
                    ),
                    "opsd_teacher_checkpoint_bytes": (
                        teacher_checkpoint_provenance.bytes
                    ),
                }
            )
        return TrainingRuntime(
            binding=self.binding,
            renderer=renderers.renderer,
            reference_renderer=renderers.reference_renderer,
            teacher_renderer=teacher_renderer,
            pipeline=pipeline,
            vae=pipeline.vae,
            scheduler=pipeline.scheduler,
            basis_provider=basis,
            reward_model=reward,
            witness_model=witness,
            adapter=adapter,
            ledger=infrastructure.ledger,
            operation_executor=infrastructure.operation_executor,
            rollout_store=infrastructure.rollout_store,
            selected_records=selected_records,
            reward_statistics=reward_statistics,
            f0_gate=f0_gate,
            opsd_teacher_state=opsd_teacher_state,
            training_cohort=training_cohort,
            initial_checkpoint_provenance=renderers.initial_checkpoint_provenance,
            teacher_checkpoint_provenance=teacher_checkpoint_provenance,
            provenance=provenance,
            _model_stage=model_stage,
            _reward_stage=reward_stage,
            _witness_stage=witness_stage,
            _cleanup_model=self.dependencies.cleanup_model,
            _cleanup_reward=self.dependencies.cleanup_reward,
            _cleanup_witness=self.dependencies.cleanup_witness,
        )


def build_training_runtime(
    binding: Any,
    spec: Optional[RuntimeLoadSpec] = None,
    *,
    dependencies: Optional[RuntimeFactoryDependencies] = None,
) -> TrainingRuntime:
    """Convenience entrypoint shared by F0, OPD, DPO, and RL handlers."""
    return TrainingRuntimeFactory(binding, dependencies=dependencies).load(spec)


def _load_snapshot_backend(repository_root: Path) -> ModuleType:
    path = repository_root / "eval-pipeline" / "adaptive_oracle_model_snapshot.py"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("pinned model-snapshot backend is unavailable")
    name = "_repldm_training_model_snapshot"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct the pinned model-snapshot backend")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_default_model(
    binding: Any,
    assets: Sequence[BoundAsset],
    staging_parent: Optional[Path],
) -> ModelStage:
    repository_root = Path(binding.authorization.repository_root)
    backend = _load_snapshot_backend(repository_root)
    expected = backend.expected_model_manifest()
    rows = expected["files"]
    if len(rows) != len(assets):
        raise ValueError("model artifact count differs from the pinned SDXL manifest")
    remaining = {str(asset.path): asset for asset in assets}
    snapshot_roots: set[Path] = set()
    for row in rows:
        relative = PurePosixPath(row["path"])
        matches = [
            asset
            for asset in remaining.values()
            if tuple(asset.path.parts[-len(relative.parts) :]) == relative.parts
        ]
        if len(matches) != 1:
            raise ValueError(f"pinned SDXL asset path is ambiguous or absent: {relative}")
        asset = matches[0]
        if asset.size != row["size"] or asset.sha256 != row["sha256"]:
            raise ValueError(f"pinned SDXL asset bytes differ: {relative}")
        snapshot_roots.add(asset.path.parents[len(relative.parts) - 1])
        remaining.pop(str(asset.path))
    if remaining or len(snapshot_roots) != 1:
        raise ValueError("model artifacts do not form one pinned SDXL snapshot")
    snapshot_root = next(iter(snapshot_roots))
    expected_root = repository_root / backend.MODEL_CACHE / backend.MODEL_SNAPSHOT_RELATIVE
    if snapshot_root != expected_root:
        raise ValueError("model artifacts are outside the registered SDXL snapshot root")
    record = backend.stage_model_snapshot(
        repository_root,
        staging_parent=staging_parent,
        expected_manifest_sha256=backend.MODEL_SNAPSHOT_MANIFEST_SHA256,
    )
    return ModelStage(backend, record, backend.MODEL_SNAPSHOT_MANIFEST_SHA256)


def _load_default_pipeline(
    stage: ModelStage,
    device: torch.device,
    model_dtype: torch.dtype,
    vae_dtype: torch.dtype,
) -> Any:
    from diffusers import AutoencoderKL
    from InferencePipelines.RepLDM.pipeline_repldm_sdxl import RepLDMSDXLPipeline

    def loader(root: str) -> Any:
        if not root.startswith("/proc/self/fd/"):
            raise RuntimeError("SDXL loader did not receive a pinned stage descriptor")
        vae = AutoencoderKL.from_pretrained(
            root,
            subfolder="vae",
            variant="fp16",
            torch_dtype=vae_dtype,
            local_files_only=True,
            use_safetensors=True,
        )
        return RepLDMSDXLPipeline.from_pretrained(
            root,
            vae=vae,
            variant="fp16",
            torch_dtype=model_dtype,
            local_files_only=True,
            use_safetensors=True,
        )

    pipeline = stage.backend.load_from_verified_staged_model_snapshot(
        stage.record,
        loader,
        expected_manifest_sha256=stage.manifest_sha256,
    )
    # Supplying no dtype preserves the separately loaded float32 VAE.
    pipeline.to(device)
    pipeline.vae.to(device=device, dtype=torch.float32)
    for module in (
        pipeline.unet,
        pipeline.vae,
        getattr(pipeline, "text_encoder", None),
        getattr(pipeline, "text_encoder_2", None),
    ):
        if isinstance(module, nn.Module):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return pipeline


def _cleanup_default_model(stage: ModelStage) -> None:
    stage.backend.cleanup_staged_model_snapshot(stage.record)


def _relative_file(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be a canonical relative path")
    return path


def _copy_asset(asset: BoundAsset, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    source = os.open(asset.path, source_flags)
    target: Optional[int] = None
    try:
        before = os.fstat(source)
        if not stat.S_ISREG(before.st_mode) or _stat_identity(before) != asset.identity:
            raise RuntimeError(f"verified reward asset identity changed: {asset.label}")
        target = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target, view)
                if written <= 0:
                    raise OSError("short write while staging reward asset")
                view = view[written:]
        os.fsync(target)
        after = os.fstat(source)
        if (
            _stat_identity(after) != asset.identity
            or copied != asset.size
            or digest.hexdigest() != asset.sha256
        ):
            raise RuntimeError(f"verified reward asset bytes changed: {asset.label}")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if target is not None:
            os.close(target)
        os.close(source)


def _stage_default_reward(
    config: Mapping[str, Any],
    assets: Sequence[BoundAsset],
    staging_parent: Optional[Path],
) -> RewardStage:
    required = {
        "schema",
        "implementation",
        "dtype",
        "preprocess_sha256",
        "checkpoint",
        "med_config",
        "source_root",
        "source_files",
        "tokenizer_root",
        "tokenizer_files",
    }
    if set(config) != required:
        raise ValueError("ImageReward runtime config fields differ from the registered schema")
    available = {str(asset.path): asset for asset in assets}
    checkpoint = available.get(config["checkpoint"])
    med_config = available.get(config["med_config"])
    if checkpoint is None or med_config is None:
        raise ValueError("ImageReward checkpoint or med_config is not a verified reward asset")
    source_root = Path(config["source_root"])
    tokenizer_root = Path(config["tokenizer_root"])
    if not source_root.is_absolute() or not tokenizer_root.is_absolute():
        raise ValueError("ImageReward source and tokenizer roots must be absolute")
    source_rows = config["source_files"]
    tokenizer_rows = config["tokenizer_files"]
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("ImageReward source_files must be a non-empty list")
    if not isinstance(tokenizer_rows, list) or not tokenizer_rows:
        raise ValueError("ImageReward tokenizer_files must be a non-empty list")

    mappings: list[tuple[BoundAsset, Path]] = []
    used = {str(checkpoint.path), str(med_config.path)}
    for label, root, rows, prefix in (
        ("source_files", source_root, source_rows, Path("source")),
        ("tokenizer_files", tokenizer_root, tokenizer_rows, Path("tokenizer")),
    ):
        for index, raw in enumerate(rows):
            relative = _relative_file(raw, f"{label}[{index}]")
            source = root.joinpath(*relative.parts)
            asset = available.get(str(source))
            if asset is None or str(source) in used:
                raise ValueError(f"{label} references an absent or duplicate reward asset")
            used.add(str(source))
            mappings.append((asset, prefix.joinpath(*relative.parts)))
    if used != set(available):
        raise ValueError("reward asset manifest contains unassigned files")

    parent = Path(tempfile.gettempdir()) if staging_parent is None else staging_parent
    root = Path(tempfile.mkdtemp(prefix="repldm-reward-", dir=str(parent)))
    root.chmod(0o700)
    try:
        staged_checkpoint = root / "checkpoint" / "ImageReward.pt"
        staged_med_config = root / "med_config.json"
        _copy_asset(checkpoint, staged_checkpoint)
        _copy_asset(med_config, staged_med_config)
        for asset, relative in mappings:
            _copy_asset(asset, root / relative)
        return RewardStage(
            root=root,
            checkpoint=staged_checkpoint,
            med_config=staged_med_config,
            source_root=root / "source",
            tokenizer_root=root / "tokenizer",
        )
    except BaseException:
        shutil.rmtree(root)
        raise


@contextmanager
def _isolated_image_reward_source(source_root: Path) -> Iterator[None]:
    existing = {name for name in sys.modules if name == "ImageReward" or name.startswith("ImageReward.")}
    if existing:
        raise RuntimeError("ImageReward was imported before its pinned source was staged")
    sys.path.insert(0, str(source_root))
    try:
        yield
    finally:
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)
        else:
            try:
                sys.path.remove(str(source_root))
            except ValueError:
                pass
        for name in tuple(sys.modules):
            if name == "ImageReward" or name.startswith("ImageReward."):
                sys.modules.pop(name, None)


def _load_default_reward(
    stage: RewardStage,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        str(stage.tokenizer_root), local_files_only=True
    )
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    with _isolated_image_reward_source(stage.source_root):
        reward_module = importlib.import_module("ImageReward.ImageReward")
        blip_module = importlib.import_module("ImageReward.models.BLIP.blip_pretrain")
        original_tokenizer = blip_module.init_tokenizer
        blip_module.init_tokenizer = lambda: tokenizer
        try:
            model = reward_module.ImageReward(
                device=str(device), med_config=str(stage.med_config)
            )
        finally:
            blip_module.init_tokenizer = original_tokenizer
        if "weights_only" not in inspect.signature(torch.load).parameters:
            raise RuntimeError("safe ImageReward loading requires torch.load(weights_only=True)")
        state = torch.load(stage.checkpoint, map_location="cpu", weights_only=True)
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("ImageReward checkpoint state differs from the pinned model")
        model.to(device=device, dtype=dtype)
        model.device = device
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model


def _cleanup_default_reward(stage: RewardStage) -> None:
    if not stage.root.name.startswith("repldm-reward-") or stage.root.is_symlink():
        raise RuntimeError("refusing to clean an invalid reward stage")
    shutil.rmtree(stage.root)


_TOPIQ_PACKAGE_NAMES = frozenset(
    {"pyiqa", "timm", "torch", "torchvision", "safetensors"}
)
_TOPIQ_SOURCE_PACKAGES = frozenset({"pyiqa", "timm"})


def _stage_default_witness(
    config: Mapping[str, Any],
    assets: Sequence[BoundAsset],
    staging_parent: Optional[Path],
) -> WitnessStage:
    """Copy the checkpoint, backbone, and pinned Python sources privately."""
    required = {
        "schema",
        "implementation",
        "role",
        "dtype",
        "preprocess_sha256",
        "checkpoint",
        "backbone",
        "package_versions",
        "source_packages",
    }
    if set(config) != required:
        raise ValueError("TOPIQ-NR runtime config fields differ from the registered schema")
    from .witnesses import (
        TOPIQ_NR_MODEL_ID,
        TOPIQ_NR_PREPROCESS_SHA256,
        TOPIQ_NR_ROLE,
    )

    if config.get("schema") != WITNESS_RUNTIME_SCHEMA:
        raise ValueError("TOPIQ-NR runtime config has an unsupported schema")
    if config.get("implementation") != TOPIQ_NR_MODEL_ID:
        raise ValueError("TOPIQ-NR runtime config has an unsupported implementation")
    if config.get("role") != TOPIQ_NR_ROLE:
        raise ValueError("TOPIQ-NR runtime config has an unsupported role")
    if config.get("dtype") != "float32":
        raise ValueError("TOPIQ-NR runtime config must bind float32")
    if config.get("preprocess_sha256") != TOPIQ_NR_PREPROCESS_SHA256:
        raise ValueError("TOPIQ-NR runtime config has the wrong preprocessing hash")
    versions = config.get("package_versions")
    if not isinstance(versions, Mapping) or set(versions) != _TOPIQ_PACKAGE_NAMES:
        raise ValueError("TOPIQ-NR package versions are incomplete")
    if any(not isinstance(value, str) or not value for value in versions.values()):
        raise ValueError("TOPIQ-NR package versions must be non-empty strings")
    packages = config.get("source_packages")
    if not isinstance(packages, Mapping) or set(packages) != _TOPIQ_SOURCE_PACKAGES:
        raise ValueError("TOPIQ-NR source package inventory is incomplete")

    available = {str(asset.path): asset for asset in assets}
    if len(available) != len(assets):
        raise ValueError("TOPIQ-NR asset manifest contains duplicate paths")
    checkpoint_path = config.get("checkpoint")
    backbone_path = config.get("backbone")
    checkpoint = available.get(checkpoint_path) if isinstance(checkpoint_path, str) else None
    backbone = available.get(backbone_path) if isinstance(backbone_path, str) else None
    if checkpoint is None or backbone is None or checkpoint is backbone:
        raise ValueError("TOPIQ-NR checkpoint or ResNet-50 backbone is not verified")
    used = {str(checkpoint.path), str(backbone.path)}
    mappings: list[tuple[BoundAsset, Path]] = []
    for package_name in sorted(_TOPIQ_SOURCE_PACKAGES):
        package = packages.get(package_name)
        if not isinstance(package, Mapping) or set(package) != {"root", "files"}:
            raise ValueError(f"TOPIQ-NR {package_name} source binding is invalid")
        raw_root = package.get("root")
        if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
            raise ValueError(f"TOPIQ-NR {package_name} source root must be absolute")
        source_root = Path(raw_root)
        rows = package.get("files")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"TOPIQ-NR {package_name} source files are empty")
        seen_relative: set[str] = set()
        for index, raw_relative in enumerate(rows):
            relative = _relative_file(
                raw_relative, f"source_packages.{package_name}.files[{index}]"
            )
            if relative.suffix != ".py" or str(relative) in seen_relative:
                raise ValueError(
                    f"TOPIQ-NR {package_name} sources must be unique Python files"
                )
            seen_relative.add(str(relative))
            source = source_root.joinpath(*relative.parts)
            asset = available.get(str(source))
            if asset is None or str(source) in used:
                raise ValueError(
                    f"TOPIQ-NR {package_name} references an absent or duplicate source"
                )
            used.add(str(source))
            mappings.append(
                (asset, Path("source") / package_name / Path(*relative.parts))
            )
        if "__init__.py" not in seen_relative:
            raise ValueError(f"TOPIQ-NR {package_name} sources omit __init__.py")
    if used != set(available):
        raise ValueError("TOPIQ-NR asset manifest contains unassigned files")

    parent = Path(tempfile.gettempdir()) if staging_parent is None else staging_parent
    root = Path(tempfile.mkdtemp(prefix="repldm-witness-", dir=str(parent)))
    root.chmod(0o700)
    try:
        staged_checkpoint = root / "checkpoint" / "topiq_nr.pth"
        staged_backbone = root / "backbone" / "resnet50.a1_in1k.safetensors"
        _copy_asset(checkpoint, staged_checkpoint)
        _copy_asset(backbone, staged_backbone)
        for asset, relative in mappings:
            _copy_asset(asset, root / relative)
        return WitnessStage(
            root=root,
            checkpoint=staged_checkpoint,
            backbone=staged_backbone,
            source_root=root / "source",
            package_versions=MappingProxyType(
                {str(key): str(value) for key, value in sorted(versions.items())}
            ),
        )
    except BaseException:
        shutil.rmtree(root)
        raise


@contextmanager
def _isolated_topiq_source(source_root: Path) -> Iterator[None]:
    prefixes = ("pyiqa", "timm")
    existing = {
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }
    if existing:
        raise RuntimeError("pyiqa or timm was imported before pinned TOPIQ sources")
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)
        else:
            try:
                sys.path.remove(str(source_root))
            except ValueError:
                pass
        for name in tuple(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                sys.modules.pop(name, None)
        importlib.invalidate_caches()


def _load_default_witness(
    stage: WitnessStage,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    if dtype is not torch.float32:
        raise ValueError("TOPIQ-NR must load in float32")
    for distribution, expected in stage.package_versions.items():
        try:
            actual = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"TOPIQ-NR dependency {distribution} is not installed"
            ) from exc
        if actual != expected:
            raise ValueError(
                f"TOPIQ-NR dependency {distribution} version differs from its config"
            )

    with _offline_loading(), _isolated_topiq_source(stage.source_root):
        pyiqa = importlib.import_module("pyiqa")
        timm = importlib.import_module("timm")
        for module, package_name in ((pyiqa, "pyiqa"), (timm, "timm")):
            module_file = getattr(module, "__file__", None)
            expected_root = stage.source_root / package_name
            if not isinstance(module_file, str):
                raise RuntimeError(f"staged {package_name} has no source file")
            try:
                Path(module_file).resolve().relative_to(expected_root.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    f"{package_name} was not imported from the verified witness stage"
                ) from exc

        original_create_model = timm.create_model
        original_torch_load = torch.load

        def create_staged_backbone(model_name: str, *args: Any, **kwargs: Any) -> Any:
            if model_name != "resnet50":
                raise RuntimeError(f"unexpected TOPIQ-NR backbone {model_name!r}")
            kwargs["pretrained"] = False
            kwargs.pop("checkpoint_path", None)
            model = original_create_model(model_name, *args, **kwargs)
            incompatible = timm.models.load_checkpoint(
                model, str(stage.backbone), strict=False
            )
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            if missing or unexpected != {"fc.bias", "fc.weight"}:
                raise RuntimeError(
                    "staged TOPIQ-NR backbone checkpoint keys differ: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            return model

        timm.create_model = create_staged_backbone
        allowed_torch_loads = {stage.checkpoint.resolve(), stage.backbone.resolve()}

        def safe_torch_load(source: Any, *args: Any, **kwargs: Any) -> Any:
            if "weights_only" not in inspect.signature(original_torch_load).parameters:
                raise RuntimeError("safe TOPIQ-NR loading requires weights_only support")
            if not isinstance(source, (str, os.PathLike)):
                raise RuntimeError("TOPIQ-NR attempted to load an unbound checkpoint stream")
            path = Path(source).resolve()
            if path not in allowed_torch_loads:
                raise RuntimeError("TOPIQ-NR attempted to load an unbound checkpoint")
            kwargs["weights_only"] = True
            return original_torch_load(path, *args, **kwargs)

        try:
            with mock.patch.object(torch, "load", safe_torch_load):
                metric = pyiqa.create_metric(
                    "topiq_nr",
                    device=device,
                    pretrained_model_path=str(stage.checkpoint),
                )
        finally:
            timm.create_model = original_create_model
    if not isinstance(metric, nn.Module):
        raise TypeError("pyiqa TOPIQ-NR loader did not return a torch module")
    metric.to(device=device, dtype=dtype)
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    if _module_dtype(metric, "TOPIQ-NR") is not torch.float32:
        raise ValueError("loaded TOPIQ-NR did not remain float32")
    return metric


def _cleanup_default_witness(stage: WitnessStage) -> None:
    if not stage.root.name.startswith("repldm-witness-") or stage.root.is_symlink():
        raise RuntimeError("refusing to clean an invalid TOPIQ-NR witness stage")
    shutil.rmtree(stage.root)


def _initial_checkpoint_asset(
    descriptor: BoundAsset,
    payload: Sequence[BoundAsset],
) -> tuple[BoundAsset, str]:
    value = _json_asset(descriptor)
    required = {"schema", "renderer_state_sha256", "checkpoint"}
    if value.get("schema") != "repldm.renderer_initial_state.v1" or set(value) != required:
        raise ValueError("initial renderer state manifest is invalid")
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"path", "sha256", "bytes"}:
        raise ValueError("initial renderer checkpoint descriptor is invalid")
    matches = [item for item in payload if str(item.path) == checkpoint.get("path")]
    if len(matches) != 1:
        raise ValueError("initial renderer checkpoint is not one verified payload")
    asset = matches[0]
    if asset.sha256 != checkpoint.get("sha256") or asset.size != checkpoint.get("bytes"):
        raise ValueError("initial renderer checkpoint binding changed")
    state_hash = value.get("renderer_state_sha256")
    if not isinstance(state_hash, str) or len(state_hash) != 64:
        raise ValueError("initial renderer state hash is invalid")
    return asset, state_hash


def _frame_calibration(value: Mapping[str, Any]) -> Any:
    from .renderer import FrameCalibration

    required = {
        "schema",
        "active_mask",
        "rank",
        "state_count",
        "minimum_norm_squared",
        "minimum_residual_ratio",
        "manifest_sha256",
        "source_sha256",
        "state_provenance_sha256",
        "decision_indices",
        "mask_hash",
        "calibration_hash",
    }
    if set(value) != required:
        raise ValueError("calibration artifact fields differ from FrameCalibration")
    calibration = FrameCalibration(
        active_mask=tuple(value["active_mask"]),
        state_count=value["state_count"],
        minimum_norm_squared=tuple(value["minimum_norm_squared"]),
        minimum_residual_ratio=tuple(value["minimum_residual_ratio"]),
        manifest_sha256=value["manifest_sha256"],
        source_sha256=value["source_sha256"],
        state_provenance_sha256=value["state_provenance_sha256"],
        schema=value["schema"],
    )
    if calibration.to_dict() != dict(value):
        raise ValueError("calibration artifact is not canonical")
    return calibration


def _renderer_from_descriptors(
    binding: Any,
    descriptors: Mapping[str, BoundAsset],
) -> nn.Module:
    from .renderer import EulerNativeFrameV1

    calibration = _frame_calibration(_json_asset(descriptors["calibration"]))
    frame = _json_asset(descriptors["renderer_frame_contract"])
    required = {
        "schema",
        "renderer_frame_contract_hash",
        "action_contract_hash",
        "renderer_config",
    }
    if frame.get("schema") != RENDERER_RUNTIME_SCHEMA or set(frame) != required:
        raise ValueError("renderer frame artifact has an unsupported runtime schema")
    config = frame.get("renderer_config")
    config_fields = {
        "latent_channels",
        "hidden_dim",
        "depth",
        "prompt_dim",
        "state_dim",
        "timestep_dim",
        "coefficient_bound",
        "max_update_ratio",
        "preserve_moments",
        "theta_max",
        "epsilon",
        "pre_squash_sigma",
    }
    if not isinstance(config, Mapping) or set(config) != config_fields:
        raise ValueError("renderer constructor config fields differ")
    renderer = EulerNativeFrameV1(calibration=calibration, **dict(config))
    if renderer.frame_contract_hash != frame["renderer_frame_contract_hash"]:
        raise ValueError("constructed renderer frame hash differs from its artifact")
    binding.validate_component(renderer)
    return renderer


def _safe_state_dict(
    asset: BoundAsset, *, label: str
) -> Mapping[str, torch.Tensor]:
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError("safe renderer loading requires torch.load(weights_only=True)")
    payload = torch.load(
        io.BytesIO(_read_asset(asset)), map_location="cpu", weights_only=True
    )
    if (
        isinstance(payload, Mapping)
        and payload.get("schema") == "repldm.latent_renderer_checkpoint.v1"
        and isinstance(payload.get("model"), Mapping)
    ):
        payload = payload["model"]
    if (
        not isinstance(payload, Mapping)
        or not payload
        or any(not isinstance(key, str) for key in payload)
        or any(not isinstance(value, torch.Tensor) for value in payload.values())
    ):
        raise ValueError(f"{label} is not a tensor state_dict")
    return payload


def _opsd_teacher_checkpoint_asset(
    descriptor: BoundAsset,
    payload: Sequence[BoundAsset],
    *,
    contract: Mapping[str, Any],
) -> tuple[BoundAsset, dict[str, Any]]:
    from .gates import validate_opsd_teacher_state

    teacher_state, checkpoint = validate_opsd_teacher_state(
        _json_asset(descriptor), contract=contract
    )
    matches = [
        item
        for item in payload
        if item.label == "OPSD teacher checkpoint"
        and str(item.path) == checkpoint.get("path")
    ]
    if len(matches) != 1:
        raise ValueError("OPSD teacher checkpoint is not one verified payload")
    asset = matches[0]
    if asset.sha256 != checkpoint.get("sha256") or asset.size != checkpoint.get(
        "bytes"
    ):
        raise ValueError("OPSD teacher checkpoint binding changed")
    return asset, teacher_state


def _load_default_renderers(
    binding: Any,
    descriptors: Mapping[str, BoundAsset],
    payload: Sequence[BoundAsset],
    device: torch.device,
) -> RendererBundle:
    from .storage import CheckpointProvenance

    checkpoint, state_hash = _initial_checkpoint_asset(
        descriptors["initial_renderer_state"], payload
    )
    state = _safe_state_dict(checkpoint, label="initial renderer checkpoint")
    renderer = _renderer_from_descriptors(binding, descriptors)
    reference = _renderer_from_descriptors(binding, descriptors)
    renderer.load_state_dict(state, strict=True)
    reference.load_state_dict(state, strict=True)
    renderer.to(device=device, dtype=torch.float32).train()
    reference.to(device=device, dtype=torch.float32).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    if renderer is reference or any(
        left is right for left, right in zip(renderer.parameters(), reference.parameters())
    ):
        raise RuntimeError("behavior and reference renderer parameters alias")
    binding.validate_initial_renderer(renderer)
    binding.validate_initial_renderer(reference)
    provenance = CheckpointProvenance.capture(
        checkpoint.path, renderer_state_sha256=state_hash
    )
    teacher = None
    teacher_provenance = None
    if "opsd_teacher_state" in descriptors:
        teacher_checkpoint, teacher_state = _opsd_teacher_checkpoint_asset(
            descriptors["opsd_teacher_state"], payload, contract=binding.contract
        )
        teacher_weights = _safe_state_dict(
            teacher_checkpoint, label="OPSD teacher checkpoint"
        )
        teacher = _renderer_from_descriptors(binding, descriptors)
        teacher.load_state_dict(teacher_weights, strict=True)
        teacher.to(device=device, dtype=torch.float32).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        from .artifacts import module_state_sha256

        teacher_hash = module_state_sha256(teacher)
        if (
            teacher_hash != teacher_state["renderer_state_sha256"]
            or teacher_hash != binding.contract.get("opsd_teacher_renderer_sha256")
        ):
            raise ValueError("loaded OPSD teacher renderer state differs from its contract")
        binding.validate_component(teacher)
        teacher_provenance = CheckpointProvenance.capture(
            teacher_checkpoint.path, renderer_state_sha256=teacher_hash
        )
    return RendererBundle(
        renderer,
        reference,
        provenance,
        teacher,
        teacher_provenance,
    )


def _make_default_infrastructure(binding: Any) -> RuntimeInfrastructure:
    from .ledger import QueryLedger
    from .operations import LedgeredOperationExecutor
    from .storage import AtomicRolloutStore

    contract = binding.contract
    ledger = QueryLedger(
        contract["paths"]["ledger_path"],
        contract["query_budget"],
        run_contract=binding.run_contract,
        strict_provenance=True,
        authorization_binding=binding,
    )
    executor = LedgeredOperationExecutor(ledger, authorization_binding=binding)
    store = AtomicRolloutStore(binding)
    return RuntimeInfrastructure(ledger, executor, store)


def _make_default_basis_provider(
    unet: Any,
    config: Mapping[str, Any],
    batch_size: int,
    do_classifier_free_guidance: bool,
) -> Any:
    from AttentionGuidance.latent_renderer import LazyLatentStructureBasisProvider

    required = {
        "schema",
        "implementation",
        "basis_provider_contract_hash",
        "provider_id",
        "requested_bases",
        "required_hook_names",
        "latent_channels",
        "semantic_mode",
        "semantic_topk",
        "semantic_layer",
        "feature_block",
        "permutation_seed",
        "prompt_dim",
        "state_dim",
        "scheduler_mapping",
        "basis_normalization",
    }
    if set(config) != required:
        raise ValueError("basis-provider runtime config fields differ")
    if config["implementation"] != "lazy_latent_structure_basis_provider":
        raise ValueError("basis-provider implementation is not registered")
    return LazyLatentStructureBasisProvider(
        unet,
        batch_size=batch_size,
        do_classifier_free_guidance=do_classifier_free_guidance,
        latent_channels=int(config["latent_channels"]),
        requested_bases=config["requested_bases"],
        required_hook_names=config["required_hook_names"],
        semantic_mode=str(config["semantic_mode"]),
        semantic_topk=int(config["semantic_topk"]),
        semantic_layer=config["semantic_layer"],
        feature_block=str(config["feature_block"]),
        permutation_seed=int(config["permutation_seed"]),
        prompt_dim=int(config["prompt_dim"]),
        state_dim=int(config["state_dim"]),
        scheduler_mapping=str(config["scheduler_mapping"]),
        basis_normalization=str(config["basis_normalization"]),
        provider_id=str(config["provider_id"]),
    )


def _make_default_adapter(
    pipeline: Any,
    basis_provider: Any,
    reward: nn.Module,
    operation_executor: Any,
    contract: Mapping[str, Any],
) -> Any:
    from .sdxl_adapter import SdxlEulerTrainingAdapter

    return SdxlEulerTrainingAdapter(
        pipeline,
        basis_provider,
        total_steps=int(contract["nfe"]),
        guidance_scale=float(contract["guidance_scale"]),
        guidance_rescale=float(contract["guidance_rescale"]),
        vae_scaling_factor=float(contract["runtime"]["vae_scaling_factor"]),
        decision_indices=tuple(contract["decision_indices"]),
        operation_executor=operation_executor,
        reward_model=reward,
    )


__all__ = [
    "BASIS_RUNTIME_SCHEMA",
    "RENDERER_RUNTIME_SCHEMA",
    "REWARD_RUNTIME_SCHEMA",
    "WITNESS_RUNTIME_SCHEMA",
    "BoundAsset",
    "ModelStage",
    "RendererBundle",
    "RewardStage",
    "WitnessStage",
    "RuntimeFactoryDependencies",
    "RuntimeInfrastructure",
    "RuntimeLoadSpec",
    "TrainingRuntime",
    "TrainingRuntimeFactory",
    "build_training_runtime",
    "_cleanup_default_witness",
    "_load_default_witness",
    "_stage_default_witness",
]
