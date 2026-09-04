"""Deterministically build the first 64+32 latent-renderer selected view."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .builder import REPOSITORY_ROOT, _tracked_config_path, enforce_git_gate
from .io import canonical_json_bytes, iter_jsonl, sha256_file
from .selected import (
    SELECTED_CONFIG_SCHEMA,
    SELECTED_GATE_REPORT_SCHEMA,
    SELECTED_ROW_SCHEMA,
    SELECTED_VIEW_SCHEMA,
    SOURCES,
    STRATA,
    DecodedImage,
    _artifact_projection,
    _field,
    _json_object,
    _load_protected_exact_index,
    _validate_config,
    _validate_parent_binding,
    dct_phash_v1,
    decode_image_payload,
    selected_release_id,
    validate_selected_view_release,
)
from .schema import normalize_prompt


@dataclass(frozen=True)
class TokenizerGateResult:
    token_count: int
    truncated: bool


@dataclass(frozen=True)
class ClassifierGateResult:
    stratum: str
    top_score: float
    runner_up_score: float


@dataclass(frozen=True)
class SimilarityGateResult:
    nearest_id: str
    similarity: float


@dataclass(frozen=True)
class DistanceGateResult:
    nearest_id: str
    distance: int


class SelectedViewGateRuntime(Protocol):
    """CPU-testable interface for every learned/indexed selected-view gate."""

    @property
    def bindings(self) -> Mapping[str, str]: ...

    @property
    def index_counts(self) -> Mapping[str, int]: ...

    def tokenize(
        self,
        tokenizer_id: str,
        prompt: str,
        *,
        max_tokens: int,
        add_special_tokens: bool,
        truncation: bool,
    ) -> TokenizerGateResult: ...

    def classify(
        self,
        record_id: str,
        image: DecodedImage,
        class_templates: Mapping[str, Sequence[str]],
    ) -> ClassifierGateResult: ...

    def nearest_protected_text(
        self, prompt: str
    ) -> SimilarityGateResult: ...

    def nearest_protected_phash(self, phash: str) -> DistanceGateResult: ...

    def nearest_protected_image(
        self, image: DecodedImage
    ) -> SimilarityGateResult: ...


RuntimeFactory = Callable[
    [Mapping[str, Any], Path, Path], SelectedViewGateRuntime
]


class _CandidateRejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be a finite number")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


@contextmanager
def _offline_runtime() -> Any:
    """Prevent runtime factories and gates from downloading implicit assets."""
    original_create_connection = socket.create_connection
    original_socket_connect = socket.socket.connect
    original_urlopen = urllib.request.urlopen
    environment = {
        name: os.environ.get(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }

    def disabled(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("selected-view runtime network access is disabled")

    socket.create_connection = disabled
    socket.socket.connect = disabled
    urllib.request.urlopen = disabled
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        socket.create_connection = original_create_connection
        socket.socket.connect = original_socket_connect
        urllib.request.urlopen = original_urlopen
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _runtime_bindings(
    runtime: SelectedViewGateRuntime, expected: Mapping[str, str]
) -> dict[str, str]:
    observed = getattr(runtime, "bindings", None)
    if not isinstance(observed, Mapping) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(digest, str)
        or len(digest) != 64
        for name, digest in observed.items()
    ):
        raise RuntimeError("selected-view runtime has no valid asset bindings")
    result = dict(observed)
    if result != dict(expected):
        raise RuntimeError("selected-view runtime bindings differ from the frozen config")
    return result


def _runtime_index_counts(
    runtime: SelectedViewGateRuntime, expected: Mapping[str, int]
) -> dict[str, int]:
    observed = getattr(runtime, "index_counts", None)
    if not isinstance(observed, Mapping) or any(
        not isinstance(name, str)
        or not name
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        for name, count in observed.items()
    ):
        raise RuntimeError("selected-view runtime has no valid protected-index counts")
    result = dict(observed)
    if result != dict(expected):
        raise RuntimeError("selected-view runtime protected indexes are incomplete")
    return result


def create_selected_view_runtime(
    factory: RuntimeFactory,
    *,
    config: Mapping[str, Any],
    parent_dir: Path,
    repository_root: Path,
) -> SelectedViewGateRuntime:
    """Instantiate a runtime with network access disabled."""
    with _offline_runtime():
        return factory(config, parent_dir, repository_root)


def _selection_digest(seed: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{record_id}".encode("utf-8")).hexdigest()


def _exact_check(
    prompt: str, protected_exact: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    normalized = normalize_prompt(prompt)
    return {
        "normalized_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "protected_matches": list(protected_exact.get(normalized, ())),
    }


def _gate_candidate(
    candidate: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    runtime: SelectedViewGateRuntime,
    protected_exact: Mapping[str, Sequence[str]],
) -> tuple[str, dict[str, Any], str, str]:
    record_id = candidate.get("id")
    source = candidate.get("source")
    if not isinstance(record_id, str) or not record_id:
        raise _CandidateRejected("invalid_record_id")
    if source not in SOURCES:
        raise _CandidateRejected("source_not_selected")
    if candidate.get("training_eligible") is not True or candidate.get(
        "benchmark_exact_match"
    ) != []:
        raise _CandidateRejected("parent_candidate_ineligible")

    source_config = frozen["sources"][source]
    try:
        model_prompt = _field(candidate, source_config["model_prompt_field"])
        raw_prompt = _field(candidate, source_config["raw_prompt_field"])
    except ValueError as exc:
        raise _CandidateRejected("missing_prompt_field") from exc
    if not isinstance(model_prompt, str) or not model_prompt.strip():
        raise _CandidateRejected("missing_model_prompt")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise _CandidateRejected("missing_raw_prompt")
    if (
        candidate.get("license") != source_config["license"]
        or candidate.get("license_status") != source_config["license_status"]
    ):
        raise _CandidateRejected("license_evidence_mismatch")

    exact_checks = {
        "model_prompt": _exact_check(model_prompt, protected_exact),
        "raw_prompt": _exact_check(raw_prompt, protected_exact),
    }
    if any(check["protected_matches"] for check in exact_checks.values()):
        raise _CandidateRejected("exact_text_firewall")

    token_counts: dict[str, int] = {}
    for tokenizer in config["tokenizers"]:
        result = runtime.tokenize(
            tokenizer["id"],
            model_prompt,
            max_tokens=tokenizer["max_tokens"],
            add_special_tokens=tokenizer["add_special_tokens"],
            truncation=tokenizer["truncation"],
        )
        if not isinstance(result, TokenizerGateResult):
            raise RuntimeError("tokenizer gate returned an unsupported result")
        count = _positive_int(result.token_count, label="tokenizer token count")
        if result.truncated is not False or count > tokenizer["max_tokens"]:
            raise _CandidateRejected("tokenizer_truncation")
        token_counts[tokenizer["id"]] = count

    image_raw = candidate.get("image_path")
    if not isinstance(image_raw, str):
        raise _CandidateRejected("missing_image")
    image_path = Path(image_raw)
    try:
        raw_hash = sha256_file(image_path)
        decoded = decode_image_payload(image_path, frozen["decoder"])
    except (OSError, ValueError) as exc:
        raise _CandidateRejected("image_decode_failure") from exc

    classification = runtime.classify(
        record_id,
        decoded,
        config["classifier"]["class_templates"],
    )
    if not isinstance(classification, ClassifierGateResult):
        raise RuntimeError("classifier gate returned an unsupported result")
    if classification.stratum not in STRATA:
        raise RuntimeError("classifier gate returned an unknown stratum")
    top_score = _finite_number(classification.top_score, label="classifier top score")
    runner_up = _finite_number(
        classification.runner_up_score, label="classifier runner-up score"
    )
    if top_score < runner_up:
        raise RuntimeError("classifier top score is below its runner-up")
    confidence_margin = top_score - runner_up
    required_margin = float(config["classifier"]["confidence_margin"])
    if confidence_margin < required_margin or confidence_margin == 0:
        raise _CandidateRejected("classifier_margin_or_tie")
    classifier_check = {
        "stratum": classification.stratum,
        "top_score": top_score,
        "runner_up_score": runner_up,
        "confidence_margin": confidence_margin,
        "required_margin": required_margin,
        "model_binding_sha256": frozen["classifier_model_sha256"],
    }

    semantic_checks: dict[str, dict[str, Any]] = {}
    for name, prompt in (("model_prompt", model_prompt), ("raw_prompt", raw_prompt)):
        result = runtime.nearest_protected_text(prompt)
        if not isinstance(result, SimilarityGateResult) or not result.nearest_id:
            raise RuntimeError("semantic text gate returned an unsupported result")
        similarity = _finite_number(result.similarity, label="semantic text similarity")
        if similarity < -1 or similarity > 1:
            raise RuntimeError("semantic text similarity is outside [-1, 1]")
        if similarity >= frozen["semantic_threshold"]:
            raise _CandidateRejected("semantic_text_firewall")
        semantic_checks[name] = {
            "nearest_protected_id": result.nearest_id,
            "similarity": similarity,
            "threshold": frozen["semantic_threshold"],
            "model_binding_sha256": frozen["semantic_model_sha256"],
        }

    phash = dct_phash_v1(decoded)
    phash_neighbor = runtime.nearest_protected_phash(phash)
    if not isinstance(phash_neighbor, DistanceGateResult) or not phash_neighbor.nearest_id:
        raise RuntimeError("pHash gate returned an unsupported result")
    if (
        isinstance(phash_neighbor.distance, bool)
        or not isinstance(phash_neighbor.distance, int)
        or phash_neighbor.distance < 0
    ):
        raise RuntimeError("pHash gate returned an invalid distance")
    if phash_neighbor.distance <= frozen["phash_threshold"]:
        raise _CandidateRejected("protected_phash_near_duplicate")

    image_neighbor = runtime.nearest_protected_image(decoded)
    if not isinstance(image_neighbor, SimilarityGateResult) or not image_neighbor.nearest_id:
        raise RuntimeError("image embedding gate returned an unsupported result")
    image_similarity = _finite_number(
        image_neighbor.similarity, label="image embedding similarity"
    )
    if image_similarity < -1 or image_similarity > 1:
        raise RuntimeError("image embedding similarity is outside [-1, 1]")
    if image_similarity >= frozen["image_threshold"]:
        raise _CandidateRejected("protected_image_embedding_near_duplicate")

    evidence = {
        "model_prompt": model_prompt,
        "raw_prompt": raw_prompt,
        "raw_file_sha256": raw_hash,
        "decoded_pixel_sha256": decoded.pixel_sha256,
        "decoded_width": decoded.width,
        "decoded_height": decoded.height,
        "phash": phash,
        "token_counts": token_counts,
        "classifier_check": classifier_check,
        "exact_text_checks": exact_checks,
        "semantic_text_checks": semantic_checks,
        "nearest_protected_image": {
            "phash": {
                "nearest_protected_id": phash_neighbor.nearest_id,
                "distance": phash_neighbor.distance,
                "threshold": frozen["phash_threshold"],
                "definition_sha256": frozen["phash_definition_sha256"],
            },
            "embedding": {
                "nearest_protected_id": image_neighbor.nearest_id,
                "similarity": image_similarity,
                "threshold": frozen["image_threshold"],
                "model_binding_sha256": frozen["image_model_sha256"],
            },
        },
    }
    return classification.stratum, evidence, raw_hash, decoded.pixel_sha256


def _empty_cells() -> dict[str, int]:
    return {f"{source}/{stratum}": 0 for source in SOURCES for stratum in STRATA}


def _select_rows(
    *,
    parent_dir: Path,
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    runtime: SelectedViewGateRuntime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protected_exact = _load_protected_exact_index(parent_dir)
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    ids: set[str] = set()
    digests: set[str] = set()
    seed = frozen["selection_seed"]
    for candidate in iter_jsonl(parent_dir / "training_candidates.jsonl"):
        if candidate.get("source") not in SOURCES:
            continue
        record_id = candidate.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("selected-source candidate lacks a stable id")
        if record_id in ids:
            raise ValueError(f"selected-source candidate id is duplicated: {record_id}")
        ids.add(record_id)
        digest = _selection_digest(seed, record_id)
        if digest in digests:
            raise ValueError("selection digest collision rejects the selected view")
        digests.add(digest)
        candidates.append((digest, record_id, candidate))
    candidates.sort(key=lambda value: (value[0], value[1]))

    accepted: dict[tuple[str, str], list[dict[str, Any]]] = {
        (source, stratum): [] for source in SOURCES for stratum in STRATA
    }
    rejection_counts: Counter[str] = Counter()
    examined = 0
    raw_hashes: set[str] = set()
    pixel_hashes: set[str] = set()
    for digest, record_id, candidate in candidates:
        if all(len(rows) == 6 for rows in accepted.values()):
            break
        examined += 1
        try:
            stratum, evidence, raw_hash, pixel_hash = _gate_candidate(
                candidate,
                config=config,
                frozen=frozen,
                runtime=runtime,
                protected_exact=protected_exact,
            )
        except _CandidateRejected as exc:
            rejection_counts[exc.reason] += 1
            continue
        cell = (candidate["source"], stratum)
        if len(accepted[cell]) == 6:
            rejection_counts["cell_already_full"] += 1
            continue
        if raw_hash in raw_hashes or pixel_hash in pixel_hashes:
            rejection_counts["selected_content_duplicate"] += 1
            continue
        raw_hashes.add(raw_hash)
        pixel_hashes.add(pixel_hash)
        accepted[cell].append(
            {
                **candidate,
                "schema": SELECTED_ROW_SCHEMA,
                "stratum": stratum,
                "selection_digest": digest,
                **evidence,
            }
        )

    complete = all(len(rows) == 6 for rows in accepted.values())
    selected: list[dict[str, Any]] = []
    if complete:
        for source in SOURCES:
            for stratum in STRATA:
                rows = accepted[(source, stratum)]
                for rank, row in enumerate(rows, 1):
                    selected.append(
                        {
                            **row,
                            "selected_split": "train" if rank <= 4 else "validation",
                            "fold": rank - 1 if rank <= 4 else None,
                            "selection_rank": rank,
                        }
                    )
    stats = {
        "candidates_examined": examined,
        "accepted_by_cell": {
            f"{source}/{stratum}": len(accepted[(source, stratum)])
            for source in SOURCES
            for stratum in STRATA
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    return selected, stats


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _descriptor(path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _failure(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _build_report(
    *,
    config_valid: bool,
    runtime_ready: bool,
    selection_complete: bool,
    training_ready: bool,
    failures: Sequence[Mapping[str, str]],
    runtime_bindings: Mapping[str, str],
    runtime_index_counts: Mapping[str, int],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(
        ({"code": row["code"], "detail": row["detail"]} for row in failures),
        key=lambda row: (row["code"], row["detail"]),
    )
    if len({row["code"] for row in ordered}) != len(ordered):
        raise ValueError("selected-view failure codes must be unique")
    return {
        "schema": SELECTED_GATE_REPORT_SCHEMA,
        "config_valid": config_valid,
        "runtime_ready": runtime_ready,
        "selection_complete": selection_complete,
        "training_ready": training_ready,
        "failures": ordered,
        "runtime_bindings": dict(sorted(runtime_bindings.items())),
        "runtime_index_counts": dict(sorted(runtime_index_counts.items())),
        "candidates_examined": stats["candidates_examined"],
        "accepted_by_cell": stats["accepted_by_cell"],
        "rejection_counts": stats["rejection_counts"],
        "selected_splits": (
            {"train": 64, "validation": 32}
            if selection_complete
            else {"train": 0, "validation": 0}
        ),
    }


def _parent_manifest_binding(
    parent_release: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
    manifest_path = parent_release / "manifest.json"
    parent_manifest = _json_object(manifest_path, label="candidate parent manifest")
    artifacts = parent_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("candidate parent lacks an artifact inventory")
    binding = {
        "path": str(manifest_path.absolute()),
        "release_id": parent_manifest.get("release_id"),
        "manifest_sha256": sha256_file(manifest_path),
        "artifacts": [_artifact_projection(row) for row in artifacts],
    }
    parent, parent_dir, parent_hash = _validate_parent_binding(
        binding, repository_root=repository_root
    )
    return parent, parent_dir, parent_hash, binding


def _install_release(staging: Path, destination: Path) -> bool:
    if destination.exists():
        existing = (destination / "manifest.json").read_bytes()
        proposed = (staging / "manifest.json").read_bytes()
        if existing != proposed:
            raise ValueError(f"selected-view release id collision: {destination.name}")
        shutil.rmtree(staging)
        return False
    installed = False
    try:
        os.replace(staging, destination)
        installed = True
        _fsync_directory(destination.parent)
    except BaseException:
        if installed:
            shutil.rmtree(destination, ignore_errors=True)
            _fsync_directory(destination.parent)
        raise
    return True


def build_selected_view_release(
    *,
    config_path: Path,
    parent_release: Path,
    output_root: Path | None = None,
    runtime_factory: RuntimeFactory | None,
    repository_root: Path = REPOSITORY_ROOT,
    allow_dirty: bool = False,
) -> Path:
    """Build, atomically install, and revalidate one selected-view release."""
    repository_root = Path(repository_root).absolute()
    config_path = Path(config_path).absolute()
    parent_release = Path(parent_release).absolute()
    output_root = Path(
        output_root or repository_root / "DATA" / "selected-views"
    ).absolute()
    git = enforce_git_gate(repository_root, allow_dirty=allow_dirty)
    config_repo_path = _tracked_config_path(
        config_path,
        repository_root,
        commit=git["commit"],
        required=not allow_dirty,
    )
    parent, parent_dir, parent_hash, parent_binding = _parent_manifest_binding(
        parent_release, repository_root=repository_root
    )
    config_bytes = config_path.read_bytes()
    try:
        config = json.loads(config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("selected-view config is not readable JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("selected-view config must contain one JSON object")

    failures: list[dict[str, str]] = []
    config_valid = False
    runtime_ready = False
    selection_complete = False
    runtime_bindings: dict[str, str] = {}
    runtime_index_counts: dict[str, int] = {}
    selected_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "candidates_examined": 0,
        "accepted_by_cell": _empty_cells(),
        "rejection_counts": {},
    }
    frozen: dict[str, Any] | None = None
    runtime: SelectedViewGateRuntime | None = None
    try:
        frozen = _validate_config(
            config,
            parent=parent,
            parent_dir=parent_dir,
            parent_manifest_sha256=parent_hash,
        )
        config_valid = True
    except (OSError, ValueError) as exc:
        failures.append(_failure("config_invalid", str(exc)))

    if config_valid and frozen is not None:
        if runtime_factory is None:
            failures.append(
                _failure(
                    "runtime_unavailable",
                    "no selected-view runtime factory was provided",
                )
            )
        else:
            try:
                runtime = create_selected_view_runtime(
                    runtime_factory,
                    config=config,
                    parent_dir=parent_dir,
                    repository_root=repository_root,
                )
                runtime_bindings = _runtime_bindings(
                    runtime, frozen["runtime_bindings"]
                )
                runtime_index_counts = _runtime_index_counts(
                    runtime, frozen["runtime_index_counts"]
                )
                runtime_ready = True
            except Exception as exc:
                runtime_bindings = {}
                runtime_index_counts = {}
                failures.append(
                    _failure(
                        "runtime_initialization_failed",
                        f"{type(exc).__name__}: {exc}",
                    )
                )

    if runtime_ready and runtime is not None and frozen is not None:
        try:
            with _offline_runtime():
                selected_rows, stats = _select_rows(
                    parent_dir=parent_dir,
                    config=config,
                    frozen=frozen,
                    runtime=runtime,
                )
            selection_complete = len(selected_rows) == 96
            if not selection_complete:
                insufficient = [
                    f"{cell}={count}"
                    for cell, count in stats["accepted_by_cell"].items()
                    if count != 6
                ]
                failures.append(
                    _failure(
                        "insufficient_source_stratum_quota",
                        "expected six accepted rows per cell; got "
                        + ", ".join(insufficient),
                    )
                )
        except Exception as exc:
            selected_rows = []
            selection_complete = False
            failures.append(
                _failure(
                    "gate_execution_failed", f"{type(exc).__name__}: {exc}"
                )
            )

    if allow_dirty:
        failures.append(
            _failure(
                "development_build",
                "dirty or unpushed selected-view builds cannot authorize training",
            )
        )
    training_ready = (
        config_valid
        and runtime_ready
        and selection_complete
        and not allow_dirty
        and not failures
    )
    report = _build_report(
        config_valid=config_valid,
        runtime_ready=runtime_ready,
        selection_complete=selection_complete,
        training_ready=training_ready,
        failures=failures,
        runtime_bindings=runtime_bindings,
        runtime_index_counts=runtime_index_counts,
        stats=stats,
    )

    final_git = enforce_git_gate(repository_root, allow_dirty=allow_dirty)
    if final_git != git:
        raise RuntimeError("repository state changed during selected-view construction")
    if config_path.read_bytes() != config_bytes:
        raise RuntimeError("selected-view config changed during construction")
    if sha256_file(parent_dir / "manifest.json") != parent_hash:
        raise RuntimeError("candidate parent changed during selected-view construction")

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".selected-view-build-", dir=output_root))
    try:
        config_output = staging / "selection-config.json"
        report_output = staging / "gate-report.json"
        _write_bytes(config_output, config_bytes)
        _write_bytes(report_output, canonical_json_bytes(report))
        core: dict[str, Any] = {
            "complete": training_ready,
            "training_ready": training_ready,
            "development_build": allow_dirty,
            "git": git,
            "config_repo_path": config_repo_path,
            "parent_catalog": parent_binding,
            "config": _descriptor(config_output, schema=SELECTED_CONFIG_SCHEMA),
            "gate_report": _descriptor(
                report_output, schema=SELECTED_GATE_REPORT_SCHEMA
            ),
        }
        if selection_complete:
            payload_output = staging / "selected-payload.jsonl"
            _write_bytes(
                payload_output,
                b"".join(canonical_json_bytes(row) for row in selected_rows),
            )
            core["selected_payload"] = _descriptor(
                payload_output,
                schema=SELECTED_ROW_SCHEMA,
                rows=96,
                splits={"train": 64, "validation": 32},
            )
        release_id = selected_release_id(core)
        manifest = {
            "schema": SELECTED_VIEW_SCHEMA,
            "release_id": release_id,
            **core,
        }
        _write_bytes(
            staging / "manifest.json",
            (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        _fsync_directory(staging)
        release_dir = output_root / release_id
        installed = _install_release(staging, release_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        validate_selected_view_release(
            release_dir,
            repository_root=repository_root,
            require_formal=not allow_dirty,
            require_training_ready=False,
        )
        if selection_complete and runtime is not None:
            verify_selected_view_runtime(
                release_dir,
                runtime=runtime,
                repository_root=repository_root,
                require_formal=not allow_dirty,
            )
    except BaseException:
        if installed:
            shutil.rmtree(release_dir, ignore_errors=True)
            _fsync_directory(output_root)
        raise
    return release_dir


def verify_selected_view_runtime(
    release_dir: Path,
    *,
    runtime: SelectedViewGateRuntime,
    repository_root: Path = REPOSITORY_ROOT,
    require_formal: bool = True,
) -> dict[str, Any]:
    """Re-run every learned/indexed gate for an installed 96-row payload."""
    manifest = validate_selected_view_release(
        release_dir,
        repository_root=repository_root,
        require_formal=require_formal,
        require_training_ready=False,
    )
    if "selected_payload" not in manifest:
        raise ValueError("selected-view release has no payload to reverify")
    parent, parent_dir, parent_hash = _validate_parent_binding(
        manifest["parent_catalog"], repository_root=Path(repository_root).absolute()
    )
    config_path = Path(release_dir) / manifest["config"]["path"]
    config = _json_object(config_path, label="selected-view config")
    frozen = _validate_config(
        config,
        parent=parent,
        parent_dir=parent_dir,
        parent_manifest_sha256=parent_hash,
    )
    _runtime_bindings(runtime, frozen["runtime_bindings"])
    _runtime_index_counts(runtime, frozen["runtime_index_counts"])
    selected = list(
        iter_jsonl(Path(release_dir) / manifest["selected_payload"]["path"])
    )
    ids = {row["id"] for row in selected}
    candidates = {
        row["id"]: row
        for row in iter_jsonl(parent_dir / "training_candidates.jsonl")
        if row.get("id") in ids
    }
    if set(candidates) != ids:
        raise ValueError("selected payload cannot be derived from its candidate parent")
    protected_exact = _load_protected_exact_index(parent_dir)
    evidence_keys = {
        "model_prompt",
        "raw_prompt",
        "raw_file_sha256",
        "decoded_pixel_sha256",
        "decoded_width",
        "decoded_height",
        "phash",
        "token_counts",
        "classifier_check",
        "exact_text_checks",
        "semantic_text_checks",
        "nearest_protected_image",
    }
    with _offline_runtime():
        for row in selected:
            stratum, evidence, _, _ = _gate_candidate(
                candidates[row["id"]],
                config=config,
                frozen=frozen,
                runtime=runtime,
                protected_exact=protected_exact,
            )
            if row.get("stratum") != stratum or {
                key: row.get(key) for key in evidence_keys
            } != evidence:
                raise ValueError(
                    f"selected row gate evidence does not reproduce: {row['id']}"
                )
    return manifest


__all__ = [
    "ClassifierGateResult",
    "DistanceGateResult",
    "RuntimeFactory",
    "SelectedViewGateRuntime",
    "SimilarityGateResult",
    "TokenizerGateResult",
    "build_selected_view_release",
    "create_selected_view_runtime",
    "verify_selected_view_runtime",
]
