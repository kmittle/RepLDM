"""Offline OpenAI-CLIP runtime for the formal selected-view gate.

This module is intentionally the only production implementation behind the
``selected_view_runtime_v1`` registry.  It accepts paths that are already
bound by the selected-view config, verifies every byte again, and then loads
the local OpenAI CLIP checkpoint and Transformers CLIP tokenizer.  No model or
index is resolved by a model id, cache lookup, or network request.

The protected indexes use deliberately small, explicit formats:

* semantic and image indexes are NumPy ``.npz`` files containing exactly
  ``ids`` (string array) and ``embeddings`` (finite, row-normalized float
  matrix);
* the pHash index is UTF-8 JSONL with exactly ``{"id": ..., "phash": ...}``
  per row.

The surrounding selected-view validator checks the public contract.  This
backend repeats the security-sensitive checks because it is also callable by
the isolated authorization revalidator.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import socket
import stat
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .selected import (
    DecodedImage,
    ParentArtifactSnapshot,
    _validate_protected_index_binding,
    model_binding_sha256,
)
from .selected_builder import (
    ClassifierGateResult,
    DistanceGateResult,
    SimilarityGateResult,
    TokenizerGateResult,
)

REGISTRY_ID = "selected_view_runtime_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PHASH_RE = re.compile(r"^[0-9a-f]{16}$")
_DEVICE_RE = re.compile(r"^(?:cpu|cuda(?::[0-9]+)?)$")
_CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".bin", ".ckpt"})
_INDEX_KEYS = frozenset({"ids", "embeddings"})


def _absolute_without_symlinks(path: Path, *, label: str) -> Path:
    """Normalize an absolute path while rejecting symlinks in every component."""
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be an absolute path")
    normalized = Path(os.path.abspath(os.fspath(path)))
    current = Path(normalized.anchor)
    for part in normalized.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{label} cannot contain a symlink: {current}")
    return normalized


def _ordinary_file(path: Path, *, label: str) -> Path:
    path = _absolute_without_symlinks(path, label=label)
    try:
        mode = path.lstat().st_mode
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(mode) or size <= 0:
        raise RuntimeError(f"{label} must be a non-empty ordinary file: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"cannot hash protected file: {path}") from exc
    return digest.hexdigest()


def _verify_binding(value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{label} binding is incomplete")
    raw_path = value.get("path")
    if not isinstance(raw_path, str):
        raise RuntimeError(f"{label} path is invalid")
    path = _ordinary_file(Path(raw_path), label=label)
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
    ):
        raise RuntimeError(f"{label} binding has invalid size or SHA-256")
    try:
        observed_size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"cannot stat {label}: {path}") from exc
    if observed_size != size or _sha256(path) != digest:
        raise RuntimeError(f"{label} differs from its config binding")
    return path


def _model_descriptor(
    value: object, *, label: str
) -> tuple[dict[str, Any], str, tuple[Path, ...]]:
    if not isinstance(value, Mapping) or set(value) != {"id", "revision", "files"}:
        raise RuntimeError(f"{label} model descriptor is incomplete")
    model_id = value.get("id")
    revision = value.get("revision")
    files = value.get("files")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"{label} model id is missing")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{7,64}", revision):
        raise RuntimeError(f"{label} model revision is not pinned")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"{label} model has no files")
    paths = tuple(
        _verify_binding(item, label=f"{label} model file {index}")
        for index, item in enumerate(files)
    )
    if len(set(paths)) != len(paths):
        raise RuntimeError(f"{label} model repeats a file")
    descriptor = dict(value)
    return descriptor, model_binding_sha256(descriptor), paths


def _checkpoint_path(paths: Sequence[Path], *, label: str) -> Path:
    candidates = [path for path in paths if path.suffix.lower() in _CHECKPOINT_SUFFIXES]
    if len(candidates) != 1:
        # A local OpenAI CLIP checkpoint is conventionally a .pt file.  Do not
        # guess when a descriptor contains two checkpoints or only HF files.
        raise RuntimeError(
            f"{label} must bind exactly one OpenAI CLIP .pt/.pth/.bin/.ckpt checkpoint"
        )
    return candidates[0]


def _tokenizer_root(paths: Sequence[Path], *, label: str) -> Path:
    if not paths:
        raise RuntimeError(f"{label} tokenizer has no files")
    roots = {path.parent for path in paths}
    if len(roots) != 1:
        raise RuntimeError(f"{label} tokenizer files must share one local directory")
    root = next(iter(roots))
    names = {path.name for path in paths}
    # The slow Transformers CLIPTokenizer is intentionally constructed from
    # the OpenAI BPE pair.  Requiring both files avoids silently switching to
    # an unrelated fast tokenizer or an unbound cache artifact.
    if not {"vocab.json", "merges.txt"}.issubset(names):
        raise RuntimeError(f"{label} tokenizer must bind vocab.json and merges.txt")
    known_files = {
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    }
    try:
        present = {item.name for item in root.iterdir() if item.name in known_files}
    except OSError as exc:
        raise RuntimeError(f"cannot inspect {label} tokenizer directory") from exc
    if present - names:
        raise RuntimeError(
            f"{label} tokenizer has unbound auxiliary files: "
            + ", ".join(sorted(present - names))
        )
    return root


@contextlib.contextmanager
def _offline_environment():
    """Make accidental network use fail instead of silently downloading assets."""
    old_env = {
        key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    old_create_connection = socket.create_connection
    old_socket_connect = socket.socket.connect
    old_urlopen = urllib.request.urlopen

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("selected-view runtime network access is disabled")

    socket.create_connection = blocked
    socket.socket.connect = blocked
    urllib.request.urlopen = blocked
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        socket.create_connection = old_create_connection
        socket.socket.connect = old_socket_connect
        urllib.request.urlopen = old_urlopen
        for key, old in old_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _runtime_device() -> str:
    raw = os.environ.get("REPLDM_SELECTED_RUNTIME_DEVICE", "cpu").strip().lower()
    if not raw:
        raw = "cpu"
    if _DEVICE_RE.fullmatch(raw) is None:
        raise RuntimeError("REPLDM_SELECTED_RUNTIME_DEVICE must be cpu or cuda[:index]")
    if raw.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "selected-view runtime requested CUDA but CUDA is unavailable"
            )
        if ":" in raw:
            index = int(raw.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise RuntimeError("selected-view runtime CUDA index is unavailable")
    return raw


def _as_tensor(value: Any, *, label: str):
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        try:
            tensor = torch.as_tensor(value)
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise RuntimeError(f"{label} is not tensor-like") from exc
    if tensor.numel() == 0 or not tensor.is_floating_point() and label != "token ids":
        raise RuntimeError(f"{label} has an invalid dtype or is empty")
    if label != "token ids" and not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"{label} contains non-finite values")
    return tensor


def _normalize_tensor(value: Any, *, label: str):
    import torch

    tensor = _as_tensor(value, label=label).float()
    if tensor.ndim != 2 or tensor.shape[1] <= 0:
        raise RuntimeError(f"{label} must have shape (batch, dimension)")
    norms = torch.linalg.vector_norm(tensor, dim=1)
    if bool((norms <= 1e-12).any()):
        raise RuntimeError(f"{label} contains a zero vector")
    return tensor / norms[:, None]


def _tensor_to_ids(value: Any, *, expected_batch: int | None = None):
    import torch

    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if value is None:
        raise RuntimeError("tokenizer returned no input_ids")
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] <= 0:
        raise RuntimeError("tokenizer input_ids have an invalid shape")
    if expected_batch is not None and tensor.shape[0] != expected_batch:
        raise RuntimeError("tokenizer returned an unexpected batch size")
    if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise RuntimeError("tokenizer input_ids must be integral")
    return tensor.to(dtype=torch.long)


@dataclass
class _ClipBundle:
    model: Any
    preprocess: Any
    descriptor_hash: str


class OpenAIClipSelectedViewRuntime:
    """Concrete gate runtime backed by local OpenAI CLIP assets."""

    def __init__(
        self,
        config: Mapping[str, Any],
        parent_dir: Path,
        repository_root: Path,
        parent_snapshot: ParentArtifactSnapshot | None = None,
    ):
        if not isinstance(config, Mapping):
            raise RuntimeError("selected-view config must be a mapping")
        self._device_name = _runtime_device()
        self._device = self._torch_device(self._device_name)
        self._bundles: dict[str, _ClipBundle] = {}
        self._tokenizers: dict[str, Any] = {}
        self._template_cache: dict[tuple[str, ...], Any] = {}
        self._text_index: tuple[tuple[str, ...], Any] | None = None
        self._image_index: tuple[tuple[str, ...], Any] | None = None
        self._phash_ids: tuple[str, ...] = ()
        self._phash_values: tuple[int, ...] = ()

        classifier = config.get("classifier")
        semantic = config.get("semantic_text")
        image_embedding = config.get("image_embedding")
        tokenizers = config.get("tokenizers")
        clip_tokenizer_id = config.get("clip_tokenizer_id")
        protected = config.get("protected_index")
        if not isinstance(classifier, Mapping) or not isinstance(semantic, Mapping):
            raise RuntimeError(
                "classifier and semantic_text model descriptors are required"
            )
        if not isinstance(image_embedding, Mapping) or not isinstance(tokenizers, list):
            raise RuntimeError("image_embedding and tokenizer descriptors are required")
        if not isinstance(protected, Mapping):
            raise RuntimeError("protected index descriptor is required")
        if len(tokenizers) != 2:
            raise RuntimeError("exactly two SDXL tokenizer descriptors are required")
        if not isinstance(clip_tokenizer_id, str) or not clip_tokenizer_id:
            raise RuntimeError("selected-view CLIP tokenizer binding is invalid")

        self._config = config
        self._repository_root = _absolute_without_symlinks(
            Path(repository_root), label="repository root"
        )
        self._parent_dir = _absolute_without_symlinks(
            Path(parent_dir), label="parent directory"
        )

        # A formal candidate parent must carry the immutable protected cohort
        # manifest.  Historical unit-test fixtures have no parent manifest at
        # all, so they retain the minimal runtime contract below.
        parent_manifest: Mapping[str, Any] | None = None
        parent_manifest_file: Path | None = None
        parent_manifest_path = self._parent_dir / "manifest.json"
        try:
            parent_manifest_path.lstat()
        except FileNotFoundError:
            pass
        else:
            parent_manifest_file = _ordinary_file(
                parent_manifest_path, label="parent manifest"
            )
            try:
                loaded_parent = json.loads(
                    (
                        parent_snapshot.read_bytes("manifest.json")
                        if parent_snapshot is not None
                        else parent_manifest_file.read_bytes()
                    ).decode("utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("parent manifest is not readable JSON") from exc
            if not isinstance(loaded_parent, Mapping):
                raise RuntimeError("parent manifest is not an object")
            parent_manifest = loaded_parent

        formal_parent = False
        if parent_manifest is not None:
            artifacts = parent_manifest.get("artifacts")
            holdout: Mapping[str, Any] | None = None
            if isinstance(artifacts, list):
                holdout = next(
                    (
                        row
                        for row in artifacts
                        if isinstance(row, Mapping)
                        and row.get("path") == "benchmark_holdouts.jsonl"
                    ),
                    None,
                )
            holdout_rows = parent_manifest.get("holdout_rows")
            if holdout_rows is None and isinstance(holdout, Mapping):
                holdout_rows = holdout.get("rows")
            formal_parent = (
                parent_manifest.get("candidate_catalog_complete") is True
                and parent_manifest.get("development_build") is False
                and parent_manifest.get("verify_paths") is True
                and parent_manifest.get("training_ready") is False
                and parent_manifest.get("complete") is False
                and parent_manifest.get("protected_normalized_unique_prompts")
                == 46619
                and parent_manifest.get("protected_unique_images") == 37160
                and holdout_rows == 49393
            )
        if "manifest" not in protected and formal_parent:
            raise RuntimeError(
                "formal candidate parent requires a protected index manifest"
            )

        # Validate the logical protected cohort before loading any model.  A
        # same-sized replacement index must not be able to weaken leakage
        # checks merely by preserving its row count and file binding.
        protected_manifest_info: dict[str, Any] | None = None
        if "manifest" in protected:
            try:
                if parent_manifest is None or parent_manifest_file is None:
                    raise ValueError("protected index manifest requires a parent manifest")
                parent_ref = config.get("parent_catalog")
                if not isinstance(parent_ref, Mapping):
                    raise ValueError("selected config lacks parent catalog binding")
                observed_parent_hash = (
                    parent_snapshot.manifest_sha256
                    if parent_snapshot is not None
                    else _sha256(parent_manifest_file)
                )
                if (
                    parent_ref.get("release_id") != parent_manifest.get("release_id")
                    or parent_ref.get("manifest_sha256") != observed_parent_hash
                ):
                    raise ValueError("selected config binds a different parent manifest")
                protected_manifest_info = _validate_protected_index_binding(
                    protected,
                    parent=parent_manifest,
                    parent_dir=self._parent_dir,
                    parent_manifest_sha256=observed_parent_hash,
                    decoder=config.get("decoder")
                    if isinstance(config.get("decoder"), Mapping)
                    else {},
                    parent_snapshot=parent_snapshot,
                )
            except (
                OSError,
                KeyError,
                TypeError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                raise RuntimeError(
                    f"protected index manifest validation failed: {exc}"
                ) from exc

        # Load one bundle per distinct pinned model.  The three gate roles may
        # share an OpenAI checkpoint, but their config bindings remain separate.
        model_descriptors = {
            "classifier": classifier.get("model"),
            "semantic_text": semantic.get("model"),
            "image_embedding": image_embedding.get("model"),
        }
        descriptor_hashes: dict[str, str] = {}
        for role, descriptor in model_descriptors.items():
            normalized, descriptor_hash, paths = _model_descriptor(
                descriptor, label=role
            )
            model_id = str(normalized["id"]).lower()
            if not model_id.startswith("openai/"):
                raise RuntimeError(f"{role} must use an OpenAI CLIP model descriptor")
            descriptor_hashes[role] = descriptor_hash
            checkpoint = _checkpoint_path(paths, label=role)
            if descriptor_hash not in self._bundles:
                self._bundles[descriptor_hash] = self._load_clip(
                    checkpoint, descriptor_hash, device=self._device_name, label=role
                )
        if len(set(descriptor_hashes.values())) != 1:
            raise RuntimeError(
                "classifier, semantic_text, and image_embedding must share one "
                "pinned OpenAI CLIP checkpoint"
            )

        seen_tokenizer_ids: set[str] = set()
        for index, tokenizer_descriptor in enumerate(tokenizers):
            normalized, descriptor_hash, paths = _model_descriptor(
                tokenizer_descriptor.get("model")
                if isinstance(tokenizer_descriptor, Mapping)
                else None,
                label=f"tokenizer {index}",
            )
            tokenizer_id = (
                tokenizer_descriptor.get("id")
                if isinstance(tokenizer_descriptor, Mapping)
                else None
            )
            if not isinstance(tokenizer_id, str) or not tokenizer_id:
                raise RuntimeError(f"tokenizer {index} has no id")
            if tokenizer_id in seen_tokenizer_ids:
                raise RuntimeError("tokenizer IDs must be unique")
            seen_tokenizer_ids.add(tokenizer_id)
            root = _tokenizer_root(paths, label=f"tokenizer {tokenizer_id}")
            self._tokenizers[tokenizer_id] = self._load_tokenizer(
                root, label=f"tokenizer {tokenizer_id}"
            )
        if clip_tokenizer_id not in self._tokenizers:
            raise RuntimeError(
                "selected-view CLIP tokenizer binding is not declared"
            )
        self._clip_tokenizer_id = clip_tokenizer_id

        self._classifier_hash = model_binding_sha256(dict(classifier["model"]))
        self._semantic_hash = model_binding_sha256(dict(semantic["model"]))
        self._image_hash = model_binding_sha256(dict(image_embedding["model"]))
        self._tokenizer_hashes = {
            str(row["id"]): model_binding_sha256(dict(row["model"]))
            for row in tokenizers
            if isinstance(row, Mapping)
        }

        index_specs = {
            "semantic_text": protected.get("semantic_text"),
            "phash": protected.get("phash"),
            "image_embedding": protected.get("image_embedding"),
        }
        expected_counts = {
            "semantic_text": protected.get("normalized_unique_prompts"),
            "phash": protected.get("unique_images"),
            "image_embedding": protected.get("unique_images"),
        }
        for name, binding in index_specs.items():
            path = _verify_binding(binding, label=f"protected {name} index")
            count = expected_counts[name]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise RuntimeError(f"protected {name} index count is invalid")
            if name == "semantic_text":
                self._text_index = self._load_embedding_index(path, count, label=name)
            elif name == "image_embedding":
                self._image_index = self._load_embedding_index(path, count, label=name)
            else:
                self._phash_ids, self._phash_values = self._load_phash_index(
                    path, count
                )

        if protected_manifest_info is not None:
            expected_ids = {
                "semantic_text": tuple(protected_manifest_info["prompt_ids"]),
                "phash": tuple(protected_manifest_info["image_ids"]),
                "image_embedding": tuple(protected_manifest_info["image_ids"]),
            }
            observed_ids = {
                "semantic_text": self._text_index[0] if self._text_index else (),
                "phash": self._phash_ids,
                "image_embedding": self._image_index[0]
                if self._image_index
                else (),
            }
            for name in expected_ids:
                if observed_ids[name] != expected_ids[name]:
                    raise RuntimeError(
                        f"protected {name} index IDs differ from the parent manifest"
                    )

        binding_values: dict[str, str] = {
            "classifier": self._classifier_hash,
            "image_embedding": self._image_hash,
            "protected:image_embedding": str(protected["image_embedding"]["sha256"]),
            "protected:phash": str(protected["phash"]["sha256"]),
            "protected:semantic_text": str(protected["semantic_text"]["sha256"]),
            "semantic_text": self._semantic_hash,
            **{
                f"tokenizer:{key}": value
                for key, value in self._tokenizer_hashes.items()
            },
        }
        if protected_manifest_info is not None:
            binding_values["protected:manifest"] = str(
                protected_manifest_info["manifest_sha256"]
            )
        self._bindings = MappingProxyType(binding_values)
        self._index_counts = MappingProxyType(
            {
                "semantic_text": len(self._text_index[0]) if self._text_index else 0,
                "phash": len(self._phash_ids),
                "image_embedding": len(self._image_index[0])
                if self._image_index
                else 0,
            }
        )
        expected_bindings = config.get("runtime_bindings")
        expected_index_counts = config.get("runtime_index_counts")
        if isinstance(expected_bindings, Mapping) and dict(expected_bindings) != dict(
            self._bindings
        ):
            raise RuntimeError(
                "loaded runtime bindings differ from the selected config"
            )
        if isinstance(expected_index_counts, Mapping) and dict(
            expected_index_counts
        ) != dict(self._index_counts):
            raise RuntimeError(
                "loaded protected-index counts differ from the selected config"
            )

        self._role_hashes = {
            "classifier": self._classifier_hash,
            "semantic_text": self._semantic_hash,
            "image_embedding": self._image_hash,
        }

    @staticmethod
    def _torch_device(name: str):
        import torch

        return torch.device(name)

    @staticmethod
    def _load_clip(
        checkpoint: Path,
        descriptor_hash: str,
        *,
        device: str,
        label: str,
    ) -> _ClipBundle:
        try:
            import clip
        except ImportError as exc:
            raise RuntimeError("OpenAI clip package is unavailable") from exc
        if not callable(getattr(clip, "load", None)):
            raise RuntimeError("OpenAI clip package lacks clip.load")
        # ``clip.load`` accepts a filesystem checkpoint and otherwise attempts
        # to resolve a named model.  Passing the verified absolute path and a
        # blocked download root makes that fallback impossible in practice.
        try:
            model, preprocess = clip.load(
                str(checkpoint),
                device=device,
                jit=False,
                download_root=str(checkpoint.parent),
            )
        except Exception as exc:
            raise RuntimeError(
                f"cannot load local OpenAI CLIP checkpoint for {label}"
            ) from exc
        if not callable(getattr(model, "encode_text", None)) or not callable(
            getattr(model, "encode_image", None)
        ):
            raise RuntimeError(f"{label} CLIP model lacks encode_text/encode_image")
        if not callable(preprocess):
            raise RuntimeError(f"{label} CLIP preprocess is not callable")
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()
        return _ClipBundle(
            model=model, preprocess=preprocess, descriptor_hash=descriptor_hash
        )

    @staticmethod
    def _load_tokenizer(root: Path, *, label: str) -> Any:
        try:
            from transformers import CLIPTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers.CLIPTokenizer is unavailable") from exc
        try:
            tokenizer = CLIPTokenizer.from_pretrained(str(root), local_files_only=True)
        except Exception as exc:
            raise RuntimeError(f"cannot load local CLIP tokenizer for {label}") from exc
        if not callable(tokenizer):
            raise RuntimeError(f"{label} tokenizer is not callable")
        return tokenizer

    @staticmethod
    def _load_embedding_index(
        path: Path, expected_count: int, *, label: str
    ) -> tuple[tuple[str, ...], Any]:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != _INDEX_KEYS:
                    raise RuntimeError(
                        f"protected {label} index must contain exactly ids and "
                        "embeddings"
                    )
                ids_array = np.asarray(archive["ids"])
                embeddings = np.asarray(archive["embeddings"])
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"cannot read protected {label} npz index") from exc
        if ids_array.ndim != 1 or len(ids_array) != expected_count:
            raise RuntimeError(f"protected {label} index row count differs from config")
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != expected_count
            or embeddings.shape[1] <= 0
        ):
            raise RuntimeError(f"protected {label} embeddings have an invalid shape")
        if not np.issubdtype(embeddings.dtype, np.floating):
            raise RuntimeError(f"protected {label} embeddings must be floating point")
        if not np.isfinite(embeddings).all():
            raise RuntimeError(
                f"protected {label} embeddings contain non-finite values"
            )
        norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
        if np.any(norms <= 1e-12) or not np.allclose(norms, 1.0, rtol=0.0, atol=5e-3):
            raise RuntimeError(f"protected {label} embeddings must be row-normalized")
        ids: list[str] = []
        for index, raw in enumerate(ids_array.tolist()):
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise RuntimeError(
                        f"protected {label} id {index} is not UTF-8"
                    ) from exc
            if not isinstance(raw, str) or not raw:
                raise RuntimeError(f"protected {label} id {index} is invalid")
            ids.append(raw)
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"protected {label} index IDs are not unique")
        return tuple(ids), embeddings.astype(np.float32, copy=True)

    @staticmethod
    def _load_phash_index(
        path: Path, expected_count: int
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        ids: list[str] = []
        values: list[int] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise RuntimeError(
                            f"protected phash index has a blank row at {line_number}"
                        )
                    row = json.loads(line)
                    if not isinstance(row, Mapping) or set(row) != {"id", "phash"}:
                        raise RuntimeError(
                            f"protected phash row {line_number} is incomplete"
                        )
                    row_id = row.get("id")
                    phash = row.get("phash")
                    if not isinstance(row_id, str) or not row_id:
                        raise RuntimeError(
                            f"protected phash row {line_number} has an invalid id"
                        )
                    if not isinstance(phash, str) or _PHASH_RE.fullmatch(phash) is None:
                        raise RuntimeError(
                            f"protected phash row {line_number} has an invalid hash"
                        )
                    ids.append(row_id)
                    values.append(int(phash, 16))
        except RuntimeError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("cannot read protected phash JSONL index") from exc
        if len(ids) != expected_count:
            raise RuntimeError("protected phash index row count differs from config")
        if len(set(ids)) != len(ids):
            raise RuntimeError("protected phash index IDs are not unique")
        return tuple(ids), tuple(values)

    @property
    def bindings(self) -> Mapping[str, str]:
        return self._bindings

    @property
    def index_counts(self) -> Mapping[str, int]:
        return self._index_counts

    def _tokenizer(self, tokenizer_id: str) -> Any:
        try:
            return self._tokenizers[tokenizer_id]
        except KeyError as exc:
            raise RuntimeError(
                f"unknown selected-view tokenizer: {tokenizer_id}"
            ) from exc

    def tokenize(
        self,
        tokenizer_id: str,
        prompt: str,
        *,
        max_tokens: int,
        add_special_tokens: bool,
        truncation: bool,
    ) -> TokenizerGateResult:
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("tokenizer prompt must be a non-empty string")
        if max_tokens <= 0 or add_special_tokens is not True or truncation is not False:
            raise RuntimeError(
                "selected tokenizer flags differ from the frozen contract"
            )
        tokenizer = self._tokenizer(tokenizer_id)
        try:
            encoded = tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=False,
                max_length=max_tokens,
                return_attention_mask=True,
            )
        except Exception as exc:
            raise RuntimeError(f"tokenizer {tokenizer_id} failed") from exc
        ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
        mask = encoded.get("attention_mask") if isinstance(encoded, Mapping) else None
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if isinstance(mask, list) and mask and isinstance(mask[0], list):
            mask = mask[0]
        if not isinstance(ids, (list, tuple)):
            try:
                ids = list(ids)
            except Exception as exc:
                raise RuntimeError("tokenizer returned malformed input_ids") from exc
        count = len(ids)
        if mask is not None:
            try:
                count = sum(int(value) for value in mask)
            except Exception as exc:
                raise RuntimeError(
                    "tokenizer returned malformed attention_mask"
                ) from exc
        if count <= 0:
            raise RuntimeError("tokenizer returned an empty sequence")
        return TokenizerGateResult(
            token_count=int(count), truncated=bool(len(ids) > max_tokens)
        )

    def _encode_text(
        self, bundle: _ClipBundle, prompt_list: Sequence[str], tokenizer: Any
    ):
        import torch

        if not prompt_list:
            raise RuntimeError("cannot encode an empty text batch")
        try:
            encoded = tokenizer(
                list(prompt_list),
                add_special_tokens=True,
                padding="max_length",
                max_length=77,
                # Protected benchmark captions can exceed CLIP's fixed
                # context window.  Match the OpenAI CLIP tokenizer's
                # deterministic 77-token truncation for index/query text.
                # Candidate rows use ``tokenize`` above with truncation off.
                truncation=True,
                return_tensors="pt",
            )
        except Exception as exc:
            raise RuntimeError("CLIP tokenizer failed while encoding text") from exc
        ids = _tensor_to_ids(encoded, expected_batch=len(prompt_list))
        if ids.shape[0] != len(prompt_list) or ids.shape[1] != 77:
            raise RuntimeError("CLIP tokenizer did not produce a [batch,77] tensor")
        ids = ids.to(self._device)
        with torch.inference_mode():
            features = bundle.model.encode_text(ids)
        normalized = _normalize_tensor(features, label="CLIP text features")
        return normalized

    def _image_tensor(self, bundle: _ClipBundle, image: DecodedImage):
        from PIL import Image

        pil = Image.frombytes("RGB", (image.width, image.height), image.rgb_bytes)
        try:
            transformed = bundle.preprocess(pil)
        except Exception as exc:
            raise RuntimeError("CLIP image preprocessing failed") from exc
        tensor = _as_tensor(transformed, label="CLIP image tensor")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise RuntimeError("CLIP image preprocessing must return [1,3,H,W]")
        if tensor.shape[1] != 3:
            raise RuntimeError("CLIP image preprocessing must return RGB channels")
        return tensor.to(self._device)

    def _encode_image(self, bundle: _ClipBundle, image: DecodedImage):
        import torch

        tensor = self._image_tensor(bundle, image)
        with torch.inference_mode():
            features = bundle.model.encode_image(tensor)
        return _normalize_tensor(features, label="CLIP image features")

    def _bundle_for(self, role: str) -> _ClipBundle:
        descriptor = self._config[role]["model"]
        descriptor_hash = model_binding_sha256(dict(descriptor))
        return self._bundles[descriptor_hash]

    def classify(
        self,
        record_id: str,
        image: DecodedImage,
        class_templates: Mapping[str, Sequence[str]],
    ) -> ClassifierGateResult:
        del record_id
        if not isinstance(class_templates, Mapping) or not class_templates:
            raise RuntimeError("classifier templates are empty")
        bundle = self._bundle_for("classifier")
        tokenizer = self._tokenizer(self._clip_tokenizer_id)
        flattened: list[str] = []
        spans: list[tuple[str, int, int]] = []
        for name, templates in class_templates.items():
            if (
                not isinstance(name, str)
                or not isinstance(templates, Sequence)
                or not templates
            ):
                raise RuntimeError("classifier template group is invalid")
            start = len(flattened)
            flattened.extend(str(template) for template in templates)
            spans.append((name, start, len(flattened)))
        cache_key = tuple(flattened)
        text_features = self._template_cache.get(cache_key)
        if text_features is None:
            text_features = self._encode_text(bundle, flattened, tokenizer)
            self._template_cache[cache_key] = text_features
        image_features = self._encode_image(bundle, image)
        scores = (image_features @ text_features.T).detach().cpu().numpy()[0]
        class_scores = [float(np.max(scores[start:end])) for _, start, end in spans]
        if not all(math.isfinite(value) for value in class_scores):
            raise RuntimeError("classifier produced non-finite scores")
        order = sorted(
            range(len(spans)), key=lambda index: (-class_scores[index], index)
        )
        top_index = order[0]
        runner_index = order[1] if len(order) > 1 else top_index
        return ClassifierGateResult(
            str(spans[top_index][0]),
            class_scores[top_index],
            class_scores[runner_index],
        )

    def nearest_protected_text(self, prompt: str) -> SimilarityGateResult:
        if self._text_index is None:
            raise RuntimeError("semantic text index is unavailable")
        bundle = self._bundle_for("semantic_text")
        tokenizer = self._tokenizer(self._clip_tokenizer_id)
        query = self._encode_text(bundle, [prompt], tokenizer).detach().cpu().numpy()[0]
        ids, matrix = self._text_index
        similarities = matrix @ query
        index = int(np.argmax(similarities))
        value = float(similarities[index])
        if not math.isfinite(value):
            raise RuntimeError("semantic text similarity is non-finite")
        return SimilarityGateResult(ids[index], value)

    def nearest_protected_phash(self, phash: str) -> DistanceGateResult:
        if not isinstance(phash, str) or _PHASH_RE.fullmatch(phash) is None:
            raise RuntimeError("query pHash is invalid")
        query = int(phash, 16)
        distances = [int((query ^ value).bit_count()) for value in self._phash_values]
        if not distances:
            raise RuntimeError("pHash index is empty")
        index = min(
            range(len(distances)), key=lambda position: (distances[position], position)
        )
        return DistanceGateResult(self._phash_ids[index], distances[index])

    def nearest_protected_image(self, image: DecodedImage) -> SimilarityGateResult:
        if self._image_index is None:
            raise RuntimeError("image embedding index is unavailable")
        bundle = self._bundle_for("image_embedding")
        query = self._encode_image(bundle, image).detach().cpu().numpy()[0]
        ids, matrix = self._image_index
        similarities = matrix @ query
        index = int(np.argmax(similarities))
        value = float(similarities[index])
        if not math.isfinite(value):
            raise RuntimeError("image embedding similarity is non-finite")
        return SimilarityGateResult(ids[index], value)

    def close(self) -> None:
        """Release model/index references so a subsequent run can reclaim memory."""
        self._template_cache.clear()
        self._bundles.clear()
        self._tokenizers.clear()
        self._text_index = None
        self._image_index = None
        self._phash_ids = ()
        self._phash_values = ()


def build_runtime_v1(
    config: Mapping[str, Any],
    parent_dir: Path,
    repository_root: Path,
    parent_snapshot: ParentArtifactSnapshot | None = None,
) -> OpenAIClipSelectedViewRuntime:
    """Build the fixed, local-only selected-view runtime."""
    with _offline_environment():
        return OpenAIClipSelectedViewRuntime(
            config,
            Path(parent_dir),
            Path(repository_root),
            parent_snapshot=parent_snapshot,
        )


__all__ = ["REGISTRY_ID", "OpenAIClipSelectedViewRuntime", "build_runtime_v1"]
