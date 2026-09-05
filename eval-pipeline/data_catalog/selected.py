"""Validate a content-addressed child selected-view release.

The parent v1 catalog remains a non-training-ready candidate inventory.  A
child release may authorize training only after it binds and executes the
complete parent inventory, frozen gate configuration, 96 selected payloads,
and every calibrated model/index dependency used by the Data Gate. Failed
attempts remain revalidatable but cannot authorize training.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import threading
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .builder import (
    CATALOG_SCHEMA,
    REPOSITORY_ROOT,
    _run_git_bytes,
    _validate_formal_git_record,
    _validate_recorded_upstream_ancestry,
    validate_release,
)
from .io import canonical_json_bytes, iter_jsonl, sha256_file
from .schema import normalize_prompt

SELECTED_VIEW_SCHEMA = "repldm.selected_view_release.v1"
SELECTED_CONFIG_SCHEMA = "repldm.selected_view_config.v2"
SELECTED_ROW_SCHEMA = "repldm.selected_data_record.v1"
SELECTED_GATE_REPORT_SCHEMA = "repldm.selected_view_gate_report.v1"
THRESHOLD_CALIBRATION_SCHEMA = "repldm.threshold_calibration.v1"
SELECTED_IMAGE_MAX_PIXELS = 400_000_000
_IMAGE_DECODE_LIMIT_LOCK = threading.Lock()

STRATA = (
    "nature",
    "urban",
    "people",
    "food",
    "artwork",
    "cgi",
    "animals",
    "architecture",
)
SOURCES = ("four_k_lsdb", "pixverve_95k")
PARENT_ARTIFACT_ORDER = (
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

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_CATALOG_ID_RE = re.compile(r"^catalog-[0-9a-f]{20}$")
_SELECTED_ID_RE = re.compile(r"^selected-view-[0-9a-f]{20}$")
_SELECTED_EXTRA_FIELDS = {
    "selected_split",
    "stratum",
    "fold",
    "selection_rank",
    "selection_digest",
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


def selected_release_id(manifest_core: Mapping[str, Any]) -> str:
    """Return the content-addressed directory name for one child release."""
    digest = hashlib.sha256(canonical_json_bytes(manifest_core)).hexdigest()
    return f"selected-view-{digest[:20]}"


def model_binding_sha256(value: Mapping[str, Any]) -> str:
    """Hash a model identity independently of where it is referenced."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class DecodedImage:
    """Canonical RGB payload used by the selected-view gates."""

    width: int
    height: int
    rgb_bytes: bytes
    pixel_sha256: str


def decode_image_payload(path: Path, decoder: Mapping[str, Any]) -> DecodedImage:
    """Decode one image under the frozen Pillow/EXIF/ICC contract."""
    from PIL import Image, ImageCms, ImageOps, __version__ as pillow_version, features

    image_path = _regular_file(Path(path), label="selected image")
    if (
        decoder.get("library") != "Pillow"
        or decoder.get("version") != pillow_version
        or decoder.get("littlecms_version") != features.version("littlecms2")
        or decoder.get("max_image_pixels") != SELECTED_IMAGE_MAX_PIXELS
    ):
        raise ValueError("installed Pillow/LittleCMS stack differs from the selected-view config")
    # Pillow stores this guard and warning filters as process-global values.
    # Hold the lock until every lazy decode, EXIF transform, and ICC conversion
    # has completed; several plugins re-check the pixel limit during ``load``.
    with _IMAGE_DECODE_LIMIT_LOCK:
        previous_max_image_pixels = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = SELECTED_IMAGE_MAX_PIXELS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(image_path) as opened:
                    opened_width, opened_height = opened.size
                    if opened_width * opened_height > SELECTED_IMAGE_MAX_PIXELS:
                        raise ValueError(
                            f"decoded image exceeds the frozen pixel limit: {image_path}"
                        )
                    image = ImageOps.exif_transpose(opened)
                    width, height = image.size
                    if width * height > SELECTED_IMAGE_MAX_PIXELS:
                        raise ValueError(
                            f"decoded image exceeds the frozen pixel limit: {image_path}"
                        )
                    icc_payload = image.info.get("icc_profile")
                    if icc_payload:
                        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_payload))
                        target_profile = ImageCms.createProfile("sRGB")
                        # LittleCMS cannot build an RGB-profile transform directly
                        # for L/LA/P inputs.  Expand only those first.
                        profile_color_space = getattr(
                            source_profile.profile, "xcolor_space", ""
                        ).strip()
                        if image.mode in {"L", "LA", "P"} and profile_color_space == "RGB":
                            image = image.convert("RGB")
                        image = ImageCms.profileToProfile(
                            image,
                            source_profile,
                            target_profile,
                            outputMode="RGB",
                        )
                    else:
                        image = image.convert("RGB")
                    image.load()
                    width, height = image.size
                    pixels = image.tobytes()
        except (
            OSError,
            ValueError,
            ImageCms.PyCMSError,
            Image.DecompressionBombError,
        ) as exc:
            raise ValueError(f"cannot decode selected image: {image_path}") from exc
        finally:
            Image.MAX_IMAGE_PIXELS = previous_max_image_pixels
    if width <= 0 or height <= 0 or len(pixels) != width * height * 3:
        raise ValueError(f"decoded image has an invalid RGB payload: {image_path}")
    digest = hashlib.sha256(
        width.to_bytes(8, "big") + height.to_bytes(8, "big") + pixels
    ).hexdigest()
    return DecodedImage(width, height, pixels, digest)


@lru_cache(maxsize=1)
def _phash_cosines() -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(math.cos(math.pi * (2 * position + 1) * frequency / 64) for position in range(32))
        for frequency in range(8)
    )


def dct_phash_v1(image: DecodedImage) -> str:
    """Return the frozen 64-bit row-major DCT pHash."""
    from PIL import Image

    resized = Image.frombytes("RGB", (image.width, image.height), image.rgb_bytes).convert(
        "L"
    ).resize((32, 32), resample=Image.Resampling.LANCZOS)
    pixels = tuple(resized.getdata())
    cosines = _phash_cosines()
    coefficients: list[float] = []
    for vertical in range(8):
        vertical_cosines = cosines[vertical]
        for horizontal in range(8):
            horizontal_cosines = cosines[horizontal]
            coefficients.append(
                math.fsum(
                    pixels[y * 32 + x]
                    * vertical_cosines[y]
                    * horizontal_cosines[x]
                    for y in range(32)
                    for x in range(32)
                )
            )
    median_values = sorted(coefficients[1:])
    median = median_values[len(median_values) // 2]
    value = 0
    for coefficient in coefficients:
        value = (value << 1) | int(coefficient > median)
    return f"{value:016x}"


def _is_int(value: object, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _absolute_without_symlinks(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    normalized = Path(os.path.abspath(os.fspath(path)))
    current = Path(normalized.anchor)
    for part in normalized.parts[1:]:
        current /= part
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
        raise ValueError(f"{label} must be a non-empty ordinary file")
    return path


def _inside(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root") from exc


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _resolve_release_file(value: object, release_dir: Path, *, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{label} must be one release-local filename")
    return _regular_file(release_dir / value, label=label)


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_file_binding(
    value: object,
    *,
    label: str,
    release_dir: Path | None = None,
    extra_keys: Sequence[str] = (),
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "bytes",
        "sha256",
        *extra_keys,
    }:
        raise ValueError(f"{label} file binding is incomplete")
    if release_dir is None:
        raw_path = value.get("path")
        if not isinstance(raw_path, str):
            raise ValueError(f"{label} path must be absolute")
        path = _regular_file(Path(raw_path), label=label)
    else:
        path = _resolve_release_file(value.get("path"), release_dir, label=label)
    size = value.get("bytes")
    digest = _validate_sha256(value.get("sha256"), label=f"{label}.sha256")
    if not _is_int(size, minimum=1) or path.stat().st_size != size:
        raise ValueError(f"{label} byte count differs from its binding")
    if sha256_file(path) != digest:
        raise ValueError(f"{label} content differs from its binding")
    return path


def _validate_model(value: object, *, label: str) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping) or set(value) != {"id", "revision", "files"}:
        raise ValueError(f"{label} model binding is incomplete")
    model_id = value.get("id")
    revision = value.get("revision")
    files = value.get("files")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"{label} model id is missing")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError(f"{label} model revision must be a pinned hexadecimal revision")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label} model has no bound files")
    paths = [
        _validate_file_binding(item, label=f"{label} model file {index}")
        for index, item in enumerate(files)
    ]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} model repeats a file binding")
    frozen = dict(value)
    return frozen, model_binding_sha256(frozen)


def _validate_calibration(
    binding: object,
    *,
    label: str,
    metric: str,
    selected_value: float | int,
    comparison: str,
    model_hash: str,
    protected_sample_binding: Mapping[str, Any] | None = None,
    sample_ids: Sequence[str] | None = None,
) -> None:
    path = _validate_file_binding(binding, label=f"{label} calibration")
    value = _json_object(path, label=f"{label} calibration")
    expected_keys = {
        "schema",
        "metric",
        "selected_value",
        "comparison",
        "sample_count",
        "positive_count",
        "negative_count",
        "source",
        "model_binding_sha256",
    }
    if protected_sample_binding is not None:
        expected_keys.add("protected_sample_binding")
    if sample_ids is not None:
        expected_keys.update({"sample_ids", "sample_ids_sha256"})
    if set(value) != expected_keys or value.get("schema") != THRESHOLD_CALIBRATION_SCHEMA:
        raise ValueError(f"{label} calibration schema is incomplete")
    if value.get("metric") != metric or value.get("comparison") != comparison:
        raise ValueError(f"{label} calibration metric differs from the gate")
    observed = value.get("selected_value")
    if not _is_number(observed) or float(observed) != float(selected_value):
        raise ValueError(f"{label} calibration selected value differs from the gate")
    positive = value.get("positive_count")
    negative = value.get("negative_count")
    total = value.get("sample_count")
    if (
        not _is_int(positive, minimum=1)
        or not _is_int(negative, minimum=1)
        or not _is_int(total, minimum=2)
        or positive + negative != total
    ):
        raise ValueError(f"{label} calibration must contain positive and negative examples")
    source = value.get("source")
    source_file = _validate_file_binding(source, label=f"{label} calibration source")
    if value.get("model_binding_sha256") != model_hash:
        raise ValueError(f"{label} calibration is bound to a different model or definition")
    if sample_ids is not None:
        expected_ids = list(sample_ids)
        if (
            value.get("sample_ids") != expected_ids
            or value.get("sample_ids_sha256")
            != hashlib.sha256(canonical_json_bytes(expected_ids)).hexdigest()
        ):
            raise ValueError(f"{label} calibration sample IDs differ from its source")
        try:
            source_rows = list(iter_jsonl(source_file))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"{label} calibration source is not readable JSONL") from exc
        observed_ids = [row.get("id") for row in source_rows]
        if observed_ids != expected_ids:
            raise ValueError(f"{label} calibration source IDs differ from its artifact")
    if protected_sample_binding is not None:
        observed_binding = value.get("protected_sample_binding")
        expected_binding = {
            "manifest_sha256": protected_sample_binding.get("manifest_sha256"),
            "index_sha256": protected_sample_binding.get("index_sha256"),
            "sample_ids": [list(row) for row in protected_sample_binding.get("sample_ids", ())],
        }
        if not isinstance(observed_binding, Mapping) or set(observed_binding) != {
            "manifest_sha256",
            "index_sha256",
            "sample_ids",
            "sample_ids_sha256",
        }:
            raise ValueError(f"{label} calibration protected binding is incomplete")
        observed_sample_ids = observed_binding.get("sample_ids")
        if observed_sample_ids != expected_binding["sample_ids"]:
            raise ValueError(f"{label} calibration protected IDs differ from the manifest")
        if (
            observed_binding.get("manifest_sha256")
            != expected_binding["manifest_sha256"]
            or observed_binding.get("index_sha256")
            != expected_binding["index_sha256"]
            or observed_binding.get("sample_ids_sha256")
            != hashlib.sha256(
                canonical_json_bytes(expected_binding["sample_ids"])
            ).hexdigest()
        ):
            raise ValueError(f"{label} calibration protected binding differs from the index")
        try:
            source_rows = list(iter_jsonl(source_file))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"{label} calibration source is not readable JSONL") from exc
        source_ids = [row.get("protected_ids") for row in source_rows]
        if source_ids != expected_binding["sample_ids"]:
            raise ValueError(f"{label} calibration samples are not bound to the protected IDs")


def _protected_calibration_sample_ids(ids: Sequence[str], count: int = 128) -> list[list[str]]:
    """Mirror the producer's positive/cyclic-neighbor calibration cohort."""
    base = list(ids[: min(count, len(ids))])
    if len(base) < 2:
        raise ValueError("protected calibration cohort must contain at least two IDs")
    return [[value] for value in base] + [
        [value, base[(index + 1) % len(base)]]
        for index, value in enumerate(base)
    ]


def _classifier_calibration_sample_ids(parent_dir: Path) -> list[str]:
    """Reproduce the producer's balanced, ID-sorted classifier cohort."""
    from .selected_assets import STRATA as ASSET_STRATA
    from .selected_assets import _keyword_class

    by_class: dict[str, list[tuple[str, str]]] = {name: [] for name in ASSET_STRATA}
    seen: set[str] = set()
    for row in iter_jsonl(parent_dir / "training_candidates.jsonl"):
        if row.get("modality") != "image_text" or not row.get("image_path"):
            continue
        expected = _keyword_class(str(row.get("prompt", "")))
        if expected is None:
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("classifier calibration candidate lacks a stable id")
        # Parent publication already verifies candidate image paths.  Keep
        # validator-side calibration-ID derivation lexical so repeated config
        # validation does not issue one network-filesystem stat per row.
        raw_path = Path(str(row["image_path"]))
        image_path = str(Path(os.path.abspath(os.fspath(raw_path))))
        if image_path in seen:
            continue
        seen.add(image_path)
        by_class[expected].append((row_id, image_path))
    minimum_per_class = 32
    missing = [
        f"{name}={len(by_class[name])}"
        for name in ASSET_STRATA
        if len(by_class[name]) < minimum_per_class
    ]
    if missing:
        raise ValueError(
            "classifier calibration lacks the required per-stratum cohort: "
            + ", ".join(missing)
        )
    return [
        row_id
        for name in ASSET_STRATA
        for row_id, _ in sorted(by_class[name])[:minimum_per_class]
    ]


def _validate_source_config(value: object, *, source: str) -> dict[str, Any]:
    expected = {
        "model_prompt_field",
        "raw_prompt_field",
        "license",
        "license_status",
        "license_evidence",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"source field map is incomplete for {source}")
    for key in ("model_prompt_field", "raw_prompt_field", "license", "license_status"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"source {source} has an invalid {key}")
    evidence = value.get("license_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"source {source} lacks license evidence")
    for index, binding in enumerate(evidence):
        _validate_file_binding(binding, label=f"{source} license evidence {index}")
    return dict(value)


def _index_ids(path: Path, *, kind: str) -> tuple[str, ...]:
    """Read only the ordered IDs from one protected index.

    Embeddings are deliberately not loaded here; the runtime performs the
    finite/normalization checks.  Reading IDs at config validation time closes
    the same-count-but-unrelated-index gap before model initialization.
    """
    if kind in {"semantic_text", "image_embedding"}:
        try:
            import numpy as np

            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"ids", "embeddings"}:
                    raise ValueError(f"protected {kind} index fields are incomplete")
                values = np.asarray(archive["ids"])
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"protected {kind} index is not readable") from exc
        if values.ndim != 1:
            raise ValueError(f"protected {kind} index IDs are not one-dimensional")
        result: list[str] = []
        for index, value in enumerate(values.tolist()):
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"protected {kind} index ID {index} is not UTF-8") from exc
            if not isinstance(value, str) or not value:
                raise ValueError(f"protected {kind} index ID {index} is invalid")
            result.append(value)
    elif kind == "phash":
        result = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise ValueError(f"protected phash index has a blank row at {line_number}")
                    row = json.loads(line)
                    if not isinstance(row, Mapping) or set(row) != {"id", "phash"}:
                        raise ValueError(f"protected phash row {line_number} is incomplete")
                    value = row.get("id")
                    if not isinstance(value, str) or not value:
                        raise ValueError(f"protected phash row {line_number} has an invalid id")
                    result.append(value)
        except ValueError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("protected phash index is not readable") from exc
    else:  # pragma: no cover - internal caller guard
        raise ValueError(f"unknown protected index kind: {kind}")
    if len(set(result)) != len(result):
        raise ValueError(f"protected {kind} index IDs are not unique")
    return tuple(result)


def _validate_protected_index_binding(
    protected: object,
    *,
    parent: Mapping[str, Any],
    parent_dir: Path,
    parent_manifest_sha256: str,
    decoder: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate frozen cohort evidence and every index's ordered IDs.

    The producer's evidence is not trusted as a substitute for the current
    files: formal validation re-reads and hashes every protected image, then
    rebinds that evidence to the current holdout/decoder contract.  This
    closes the gap where an image could be replaced after asset construction.
    """
    if not isinstance(protected, Mapping) or set(protected) != {
        "holdout_rows",
        "normalized_unique_prompts",
        "unique_images",
        "manifest",
        "manifest_sha256",
        "semantic_text",
        "phash",
        "image_embedding",
        "index_bindings",
    }:
        raise ValueError("protected index binding is incomplete")
    manifest_path = _validate_file_binding(protected["manifest"], label="protected index manifest")
    observed_manifest = _json_object(manifest_path, label="protected index manifest")
    try:
        from .selected_assets import (
            derive_protected_index_manifest,
            protected_ids_sha256,
        )

        # Re-verify the external image bytes at authorization time.  The
        # manifest is an expected-value record, not a trust boundary.
        frozen_images = observed_manifest.get("images")
        expected_manifest = derive_protected_index_manifest(
            parent_dir,
            parent_release_id=str(parent.get("release_id")),
            parent_manifest_sha256=parent_manifest_sha256,
            decoder=decoder,
            image_evidence=frozen_images
            if isinstance(frozen_images, list)
            else None,
            verify_image_evidence=True,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError("cannot derive the protected index manifest from the parent") from exc
    if observed_manifest != expected_manifest:
        raise ValueError("protected index manifest differs from the parent holdout catalog")
    manifest_hash = expected_manifest["manifest_sha256"]
    if protected.get("manifest_sha256") != manifest_hash:
        raise ValueError("protected index manifest hash differs from its binding")

    counts = expected_manifest["counts"]
    if (
        protected.get("holdout_rows") != counts["holdout_rows"]
        or protected.get("normalized_unique_prompts") != counts["normalized_unique_prompts"]
        or protected.get("unique_images") != counts["unique_images"]
        or protected.get("holdout_rows") != 49393
        or protected.get("normalized_unique_prompts")
        != parent.get("protected_normalized_unique_prompts")
        or protected.get("unique_images") != parent.get("protected_unique_images")
    ):
        raise ValueError("protected index counts differ from the candidate parent")

    prompt_ids = tuple(row["id"] for row in expected_manifest["prompts"])
    image_ids = tuple(row["id"] for row in expected_manifest["images"])
    if expected_manifest["prompt_ids_sha256"] != protected_ids_sha256(prompt_ids):
        raise ValueError("protected prompt ID manifest hash is invalid")
    if expected_manifest["image_ids_sha256"] != protected_ids_sha256(image_ids):
        raise ValueError("protected image ID manifest hash is invalid")
    index_bindings = protected.get("index_bindings")
    if not isinstance(index_bindings, Mapping) or set(index_bindings) != {
        "semantic_text",
        "phash",
        "image_embedding",
    }:
        raise ValueError("protected index ID bindings are incomplete")
    expected_ids = {
        "semantic_text": (prompt_ids, expected_manifest["prompt_ids_sha256"]),
        "phash": (image_ids, expected_manifest["image_ids_sha256"]),
        "image_embedding": (image_ids, expected_manifest["image_ids_sha256"]),
    }
    index_paths: dict[str, Path] = {}
    index_hashes: dict[str, str] = {}
    for name in ("semantic_text", "phash", "image_embedding"):
        index_paths[name] = _validate_file_binding(
            protected[name], label=f"protected {name} index"
        )
        metadata = index_bindings[name]
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "manifest_sha256",
            "ids_sha256",
        }:
            raise ValueError(f"protected {name} ID binding is incomplete")
        expected_index_ids, expected_ids_hash = expected_ids[name]
        if (
            metadata.get("manifest_sha256") != manifest_hash
            or metadata.get("ids_sha256") != expected_ids_hash
        ):
            raise ValueError(f"protected {name} index is bound to a different manifest")
        observed_ids = _index_ids(index_paths[name], kind=name)
        if observed_ids != expected_index_ids:
            raise ValueError(f"protected {name} index IDs differ from the parent manifest")
        index_hashes[name] = str(protected[name]["sha256"])

    return {
        "manifest_sha256": manifest_hash,
        "manifest_file_sha256": str(protected["manifest"]["sha256"]),
        "prompt_ids": prompt_ids,
        "image_ids": image_ids,
        "prompt_ids_sha256": expected_manifest["prompt_ids_sha256"],
        "image_ids_sha256": expected_manifest["image_ids_sha256"],
        "index_sha256": index_hashes,
    }


def _legacy_protected_index_info(
    protected: Mapping[str, Any], *, parent: Mapping[str, Any], parent_dir: Path
) -> dict[str, Any] | None:
    """Read the pre-manifest fixture contract when the parent is not formal.

    The repository's historical unit-test fixture intentionally contains one
    holdout row while carrying the production counts.  It is not a valid
    candidate release and is already rejected by the parent validator.  Keep
    those tests useful without allowing a correctly-sized formal parent to
    omit the identity manifest.
    """
    try:
        row_count = sum(1 for _ in iter_jsonl(parent_dir / "benchmark_holdouts.jsonl"))
    except (OSError, ValueError):
        return None
    if row_count == 49393:
        return None
    if (
        protected.get("holdout_rows") != 49393
        or protected.get("normalized_unique_prompts")
        != parent.get("protected_normalized_unique_prompts")
        or protected.get("unique_images") != parent.get("protected_unique_images")
    ):
        return None
    protected_hashes: dict[str, str] = {}
    for name in ("semantic_text", "phash", "image_embedding"):
        binding = protected.get(name)
        if not isinstance(binding, Mapping):
            return None
        _validate_file_binding(binding, label=f"protected {name} index")
        digest = binding.get("sha256")
        if not isinstance(digest, str):
            return None
        protected_hashes[name] = digest
    return {
        "legacy": True,
        "index_sha256": protected_hashes,
        "prompt_ids": (),
        "image_ids": (),
        "prompt_ids_sha256": "",
        "image_ids_sha256": "",
        "manifest_sha256": "",
        "manifest_file_sha256": "",
    }


def _validate_config(
    config: object,
    *,
    parent: Mapping[str, Any],
    parent_dir: Path,
    parent_manifest_sha256: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "view_id",
        "parent_catalog",
        "sources",
        "strata",
        "quotas",
        "selection",
        "classifier",
        "tokenizers",
        "clip_tokenizer_id",
        "decoder",
        "semantic_text",
        "phash",
        "image_embedding",
        "protected_index",
    }
    if not isinstance(config, Mapping) or set(config) != expected_keys:
        raise ValueError("selected-view config fields are incomplete")
    if config.get("schema") != SELECTED_CONFIG_SCHEMA:
        raise ValueError("selected-view config has an unsupported schema")
    view_id = config.get("view_id")
    if not isinstance(view_id, str) or not view_id:
        raise ValueError("selected-view config has no view id")

    parent_ref = config.get("parent_catalog")
    if not isinstance(parent_ref, Mapping) or set(parent_ref) != {
        "release_id",
        "manifest_sha256",
    }:
        raise ValueError("selected-view config parent binding is incomplete")
    if (
        parent_ref.get("release_id") != parent.get("release_id")
        or parent_ref.get("manifest_sha256") != parent_manifest_sha256
    ):
        raise ValueError("selected-view config binds a different candidate parent")

    sources = config.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCES):
        raise ValueError("selected-view config must bind exactly the two frozen sources")
    source_config = {
        source: _validate_source_config(sources[source], source=source) for source in SOURCES
    }
    if tuple(config.get("strata", ())) != STRATA:
        raise ValueError("selected-view config strata differ from the frozen eight strata")

    quotas = config.get("quotas")
    expected_quotas = {
        "train_per_source_stratum": 4,
        "validation_per_source_stratum": 2,
        "train_total": 64,
        "validation_total": 32,
    }
    if quotas != expected_quotas:
        raise ValueError("selected-view quotas must remain exactly 64 train and 32 validation")

    selection = config.get("selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "seed",
        "algorithm",
        "tie_rule",
    }:
        raise ValueError("selected-view selection contract is incomplete")
    if (
        not isinstance(selection.get("seed"), str)
        or not selection["seed"]
        or selection.get("algorithm") != "sha256_seed_nul_record_id_v1"
        or selection.get("tie_rule") != "reject"
    ):
        raise ValueError("selected-view selection seed, algorithm, or tie rule is invalid")

    classifier = config.get("classifier")
    if not isinstance(classifier, Mapping) or set(classifier) != {
        "model",
        "class_templates",
        "confidence_margin",
        "tie_rule",
        "calibration",
    }:
        raise ValueError("classifier gate is incomplete")
    _, classifier_hash = _validate_model(classifier["model"], label="classifier")
    templates = classifier.get("class_templates")
    if not isinstance(templates, Mapping) or set(templates) != set(STRATA):
        raise ValueError("classifier templates must cover exactly the frozen strata")
    for stratum in STRATA:
        rows = templates[stratum]
        if (
            not isinstance(rows, list)
            or not rows
            or any(not isinstance(row, str) or not row for row in rows)
        ):
            raise ValueError(f"classifier templates are invalid for {stratum}")
    margin = classifier.get("confidence_margin")
    if not _is_number(margin) or not 0 < float(margin) <= 2:
        raise ValueError("classifier confidence margin is invalid")
    if classifier.get("tie_rule") != "reject":
        raise ValueError("classifier ties must reject the candidate")
    tokenizers = config.get("tokenizers")
    if not isinstance(tokenizers, list) or len(tokenizers) != 2:
        raise ValueError("both SDXL tokenizer manifests are required")
    tokenizer_ids = []
    tokenizer_hashes: dict[str, str] = {}
    for index, tokenizer in enumerate(tokenizers):
        if not isinstance(tokenizer, Mapping) or set(tokenizer) != {
            "id",
            "model",
            "max_tokens",
            "add_special_tokens",
            "truncation",
        }:
            raise ValueError(f"SDXL tokenizer {index} contract is incomplete")
        tokenizer_id = tokenizer.get("id")
        if not isinstance(tokenizer_id, str) or not tokenizer_id:
            raise ValueError("SDXL tokenizer id is invalid")
        tokenizer_ids.append(tokenizer_id)
        _, tokenizer_hash = _validate_model(
            tokenizer.get("model"), label=f"tokenizer {tokenizer_id}"
        )
        tokenizer_hashes[tokenizer_id] = tokenizer_hash
        if (
            tokenizer.get("max_tokens") != 77
            or tokenizer.get("add_special_tokens") is not True
            or tokenizer.get("truncation") is not False
        ):
            raise ValueError("SDXL tokenizer must use special tokens, max 77, without truncation")
    if len(set(tokenizer_ids)) != 2:
        raise ValueError("SDXL tokenizer ids must be unique")
    first_files = tokenizers[0]["model"]["files"]
    second_files = tokenizers[1]["model"]["files"]
    if first_files == second_files:
        raise ValueError("SDXL tokenizers must bind distinct frozen file manifests")

    clip_tokenizer_id = config.get("clip_tokenizer_id")
    if not isinstance(clip_tokenizer_id, str) or not clip_tokenizer_id:
        raise ValueError("selected-view CLIP tokenizer binding is invalid")
    if clip_tokenizer_id not in tokenizer_ids:
        raise ValueError("selected-view CLIP tokenizer binding is not declared")

    decoder = config.get("decoder")
    if not isinstance(decoder, Mapping) or set(decoder) != {
        "library",
        "version",
        "littlecms_version",
        "max_image_pixels",
        "exif_transpose",
        "icc_to_srgb",
        "output_mode",
        "pixel_hash",
    }:
        raise ValueError("decoder contract is incomplete")
    if (
        decoder.get("library") != "Pillow"
        or not isinstance(decoder.get("version"), str)
        or not decoder["version"]
        or not isinstance(decoder.get("littlecms_version"), str)
        or not decoder["littlecms_version"]
        or decoder.get("max_image_pixels") != SELECTED_IMAGE_MAX_PIXELS
        or decoder.get("exif_transpose") is not True
        or decoder.get("icc_to_srgb") is not True
        or decoder.get("output_mode") != "RGB"
        or decoder.get("pixel_hash") != "sha256_rgb_u64be_width_height_bytes_v1"
    ):
        raise ValueError("decoder/EXIF/ICC behavior is not fully frozen")

    protected = config.get("protected_index")
    if (
        isinstance(protected, Mapping)
        and "manifest" not in protected
        and isinstance(parent, Mapping)
    ):
        protected_info = _legacy_protected_index_info(
            protected, parent=parent, parent_dir=parent_dir
        )
        if protected_info is None:
            raise ValueError("protected index manifest is required for a formal parent")
    else:
        protected_info = _validate_protected_index_binding(
            protected,
            parent=parent,
            parent_dir=parent_dir,
            parent_manifest_sha256=parent_manifest_sha256,
            decoder=decoder,
        )
    protected_calibration_enabled = not bool(protected_info.get("legacy"))
    semantic_calibration_binding = (
        {
            "manifest_sha256": protected_info["manifest_sha256"],
            "index_sha256": protected_info["index_sha256"]["semantic_text"],
            "sample_ids": _protected_calibration_sample_ids(protected_info["prompt_ids"]),
        }
        if protected_calibration_enabled
        else None
    )
    image_calibration_binding = (
        {
            "manifest_sha256": protected_info["manifest_sha256"],
            "index_sha256": protected_info["index_sha256"]["image_embedding"],
            "sample_ids": _protected_calibration_sample_ids(protected_info["image_ids"]),
        }
        if protected_calibration_enabled
        else None
    )
    phash_calibration_binding = (
        {
            "manifest_sha256": protected_info["manifest_sha256"],
            "index_sha256": protected_info["index_sha256"]["phash"],
            "sample_ids": _protected_calibration_sample_ids(protected_info["image_ids"]),
        }
        if protected_calibration_enabled
        else None
    )

    classifier_sample_ids = (
        _classifier_calibration_sample_ids(parent_dir)
        if protected_calibration_enabled
        else None
    )
    _validate_calibration(
        classifier.get("calibration"),
        label="classifier",
        metric="classifier_confidence_margin",
        selected_value=margin,
        comparison="reject_below_margin",
        model_hash=classifier_hash,
        sample_ids=classifier_sample_ids,
    )

    semantic = config.get("semantic_text")
    if not isinstance(semantic, Mapping) or set(semantic) != {
        "model",
        "threshold",
        "comparison",
        "calibration",
    }:
        raise ValueError("semantic text gate is incomplete")
    _, semantic_hash = _validate_model(semantic["model"], label="semantic text")
    semantic_threshold = semantic.get("threshold")
    if (
        not _is_number(semantic_threshold)
        or not -1 <= float(semantic_threshold) <= 1
        or semantic.get("comparison") != "reject_at_or_above"
    ):
        raise ValueError("semantic text threshold is invalid")
    _validate_calibration(
        semantic.get("calibration"),
        label="semantic text",
        metric="cosine_similarity",
        selected_value=semantic_threshold,
        comparison="reject_at_or_above",
        model_hash=semantic_hash,
        protected_sample_binding=semantic_calibration_binding,
    )

    phash = config.get("phash")
    phash_definition_keys = {
        "implementation",
        "hash_bits",
        "resize",
        "low_frequency_size",
        "exclude_dc",
    }
    if not isinstance(phash, Mapping) or set(phash) != phash_definition_keys | {
        "threshold",
        "comparison",
        "calibration",
    }:
        raise ValueError("pHash gate is incomplete")
    phash_definition = {key: phash[key] for key in phash_definition_keys}
    if phash_definition != {
        "implementation": "dct_phash_v1",
        "hash_bits": 64,
        "resize": 32,
        "low_frequency_size": 8,
        "exclude_dc": True,
    }:
        raise ValueError("pHash definition differs from the frozen v1 definition")
    phash_hash = hashlib.sha256(canonical_json_bytes(phash_definition)).hexdigest()
    phash_threshold = phash.get("threshold")
    if (
        not _is_int(phash_threshold)
        or phash_threshold >= phash["hash_bits"]
        or phash.get("comparison") != "reject_at_or_below"
    ):
        raise ValueError("pHash threshold is invalid")
    _validate_calibration(
        phash.get("calibration"),
        label="pHash",
        metric="hamming_distance",
        selected_value=phash_threshold,
        comparison="reject_at_or_below",
        model_hash=phash_hash,
        protected_sample_binding=phash_calibration_binding,
    )

    image_embedding = config.get("image_embedding")
    if not isinstance(image_embedding, Mapping) or set(image_embedding) != {
        "model",
        "threshold",
        "comparison",
        "calibration",
    }:
        raise ValueError("image embedding gate is incomplete")
    _, image_hash = _validate_model(image_embedding["model"], label="image embedding")
    image_threshold = image_embedding.get("threshold")
    if (
        not _is_number(image_threshold)
        or not -1 <= float(image_threshold) <= 1
        or image_embedding.get("comparison") != "reject_at_or_above"
    ):
        raise ValueError("image embedding threshold is invalid")
    _validate_calibration(
        image_embedding.get("calibration"),
        label="image embedding",
        metric="cosine_similarity",
        selected_value=image_threshold,
        comparison="reject_at_or_above",
        model_hash=image_hash,
        protected_sample_binding=image_calibration_binding,
    )

    protected_hashes = protected_info["index_sha256"]

    runtime_bindings = {
        "classifier": classifier_hash,
        "image_embedding": image_hash,
        "protected:image_embedding": protected_hashes["image_embedding"],
        "protected:phash": protected_hashes["phash"],
        "protected:semantic_text": protected_hashes["semantic_text"],
        "semantic_text": semantic_hash,
        **{
            f"tokenizer:{tokenizer_id}": tokenizer_hashes[tokenizer_id]
            for tokenizer_id in tokenizer_ids
        },
    }
    if protected_calibration_enabled:
        runtime_bindings["protected:manifest"] = protected_info["manifest_sha256"]

    return {
        "view_id": view_id,
        "sources": source_config,
        "selection_seed": selection["seed"],
        "classifier_model_sha256": classifier_hash,
        "classifier_confidence_margin": float(margin),
        "tokenizer_ids": tuple(tokenizer_ids),
        "tokenizer_model_sha256": tokenizer_hashes,
        "clip_tokenizer_id": clip_tokenizer_id,
        "decoder": dict(decoder),
        "semantic_model_sha256": semantic_hash,
        "semantic_threshold": float(semantic_threshold),
        "phash_definition_sha256": phash_hash,
        "phash_definition": phash_definition,
        "phash_threshold": int(phash_threshold),
        "image_model_sha256": image_hash,
        "image_threshold": float(image_threshold),
        "protected_index_sha256": protected_hashes,
        "protected_manifest_sha256": protected_info["manifest_sha256"],
        "protected_manifest_file_sha256": protected_info["manifest_file_sha256"],
        "protected_index_ids_sha256": {
            "semantic_text": protected_info["prompt_ids_sha256"],
            "phash": protected_info["image_ids_sha256"],
            "image_embedding": protected_info["image_ids_sha256"],
        },
        "runtime_bindings": runtime_bindings,
        "runtime_index_counts": {
            "semantic_text": protected["normalized_unique_prompts"],
            "phash": protected["unique_images"],
            "image_embedding": protected["unique_images"],
        },
    }


def _field(record: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = record
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"candidate row lacks configured field {dotted_path}")
        value = value[part]
    return value


def _validate_text_check(
    value: object,
    *,
    label: str,
    prompt: str,
    protected_matches: Sequence[str],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "normalized_sha256",
        "protected_matches",
    }:
        raise ValueError(f"{label} exact text check is incomplete")
    expected = hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()
    if (
        value.get("normalized_sha256") != expected
        or value.get("protected_matches") != list(protected_matches)
        or protected_matches
    ):
        raise ValueError(f"{label} has an exact protected-prompt match")


def _load_protected_exact_index(parent_dir: Path) -> dict[str, tuple[str, ...]]:
    matches: dict[str, set[str]] = {}
    for row_number, row in enumerate(
        iter_jsonl(parent_dir / "benchmark_holdouts.jsonl"), 1
    ):
        prompt = row.get("prompt")
        row_id = row.get("id")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"protected prompt row {row_number} lacks a stable id")
        matches.setdefault(normalize_prompt(prompt), set()).add(row_id)
    if not matches:
        raise ValueError("protected exact-prompt index is empty")
    return {key: tuple(sorted(value)) for key, value in matches.items()}


def _validate_semantic_check(
    value: object,
    *,
    label: str,
    threshold: float,
    model_hash: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "nearest_protected_id",
        "similarity",
        "threshold",
        "model_binding_sha256",
    }:
        raise ValueError(f"{label} semantic text check is incomplete")
    if not isinstance(value.get("nearest_protected_id"), str) or not value[
        "nearest_protected_id"
    ]:
        raise ValueError(f"{label} has no nearest protected text neighbor")
    similarity = value.get("similarity")
    if (
        not _is_number(similarity)
        or not -1 <= float(similarity) <= 1
        or float(similarity) >= threshold
        or float(value.get("threshold", math.nan)) != threshold
        or value.get("model_binding_sha256") != model_hash
    ):
        raise ValueError(f"{label} fails the calibrated semantic text gate")


def _validate_classifier_check(
    value: object,
    *,
    stratum: str,
    required_margin: float,
    model_hash: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "stratum",
        "top_score",
        "runner_up_score",
        "confidence_margin",
        "required_margin",
        "model_binding_sha256",
    }:
        raise ValueError("selected classifier evidence is incomplete")
    top_score = value.get("top_score")
    runner_up = value.get("runner_up_score")
    margin = value.get("confidence_margin")
    if (
        value.get("stratum") != stratum
        or not _is_number(top_score)
        or not _is_number(runner_up)
        or not _is_number(margin)
        or float(top_score) < float(runner_up)
        or float(margin) != float(top_score) - float(runner_up)
        or float(margin) < required_margin
        or float(value.get("required_margin", math.nan)) != required_margin
        or value.get("model_binding_sha256") != model_hash
    ):
        raise ValueError("selected row fails the frozen classifier gate")


def _validate_image_checks(
    value: object,
    *,
    phash_threshold: int,
    phash_definition_hash: str,
    image_threshold: float,
    image_model_hash: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"phash", "embedding"}:
        raise ValueError("nearest protected image evidence is incomplete")
    phash = value["phash"]
    if not isinstance(phash, Mapping) or set(phash) != {
        "nearest_protected_id",
        "distance",
        "threshold",
        "definition_sha256",
    }:
        raise ValueError("nearest protected pHash evidence is incomplete")
    if (
        not isinstance(phash.get("nearest_protected_id"), str)
        or not phash["nearest_protected_id"]
        or not _is_int(phash.get("distance"))
        or phash["distance"] <= phash_threshold
        or phash.get("threshold") != phash_threshold
        or phash.get("definition_sha256") != phash_definition_hash
    ):
        raise ValueError("selected image fails the calibrated pHash gate")
    embedding = value["embedding"]
    if not isinstance(embedding, Mapping) or set(embedding) != {
        "nearest_protected_id",
        "similarity",
        "threshold",
        "model_binding_sha256",
    }:
        raise ValueError("nearest protected image embedding evidence is incomplete")
    similarity = embedding.get("similarity")
    if (
        not isinstance(embedding.get("nearest_protected_id"), str)
        or not embedding["nearest_protected_id"]
        or not _is_number(similarity)
        or not -1 <= float(similarity) <= 1
        or float(similarity) >= image_threshold
        or float(embedding.get("threshold", math.nan)) != image_threshold
        or embedding.get("model_binding_sha256") != image_model_hash
    ):
        raise ValueError("selected image fails the calibrated image-embedding gate")


def _validate_selected_rows(
    payload_path: Path,
    *,
    parent_dir: Path,
    frozen: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    rows = list(iter_jsonl(payload_path))
    if len(rows) != 96:
        raise ValueError(f"selected payload must contain exactly 96 rows, got {len(rows)}")
    selected_ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in selected_ids):
        raise ValueError("selected row id is invalid")
    if len(set(selected_ids)) != 96:
        raise ValueError("selected row ids must be unique")

    candidates: dict[str, dict[str, Any]] = {}
    selected_set = set(selected_ids)
    for candidate in iter_jsonl(parent_dir / "training_candidates.jsonl"):
        row_id = candidate.get("id")
        if row_id in selected_set:
            if row_id in candidates:
                raise ValueError("candidate parent repeats a selected id")
            candidates[row_id] = candidate
    if set(candidates) != selected_set:
        raise ValueError("selected payload contains an id outside the candidate parent")

    quotas: Counter[tuple[str, str, str]] = Counter()
    ranks: dict[tuple[str, str], set[int]] = {}
    image_paths: set[Path] = set()
    raw_hashes: set[str] = set()
    pixel_hashes: set[str] = set()
    protected_exact = _load_protected_exact_index(parent_dir)
    source_order = {name: index for index, name in enumerate(SOURCES)}
    stratum_order = {name: index for index, name in enumerate(STRATA)}
    observed_order = []
    for row in rows:
        row_id = row["id"]
        candidate = candidates[row_id]
        expected_fields = set(candidate) | _SELECTED_EXTRA_FIELDS
        if set(row) != expected_fields:
            raise ValueError(f"selected row fields differ from the contract: {row_id}")
        if row.get("schema") != SELECTED_ROW_SCHEMA:
            raise ValueError(f"selected row has an unsupported schema: {row_id}")
        for key, candidate_value in candidate.items():
            if key == "schema":
                continue
            if row.get(key) != candidate_value:
                raise ValueError(f"selected row rewrites candidate field {key}: {row_id}")
        if candidate.get("training_eligible") is not True or candidate.get(
            "benchmark_exact_match"
        ) != []:
            raise ValueError(f"selected row is not eligible in the candidate parent: {row_id}")

        source = row.get("source")
        stratum = row.get("stratum")
        split = row.get("selected_split")
        if source not in SOURCES or stratum not in STRATA or split not in {
            "train",
            "validation",
        }:
            raise ValueError(f"selected row has an invalid source, stratum, or split: {row_id}")
        quotas[(source, stratum, split)] += 1
        fold = row.get("fold")
        if split == "train" and (not _is_int(fold) or fold > 3):
            raise ValueError(f"training row has an invalid cross-fitting fold: {row_id}")
        if split == "validation" and fold is not None:
            raise ValueError(f"validation row must not have a training fold: {row_id}")

        rank = row.get("selection_rank")
        if not _is_int(rank, minimum=1):
            raise ValueError(f"selected row has an invalid selection rank: {row_id}")
        cell_ranks = ranks.setdefault((source, stratum), set())
        if rank in cell_ranks:
            raise ValueError(f"selected source x stratum cell repeats a rank: {row_id}")
        cell_ranks.add(rank)
        expected_digest = hashlib.sha256(
            f"{frozen['selection_seed']}\0{row_id}".encode("utf-8")
        ).hexdigest()
        if row.get("selection_digest") != expected_digest:
            raise ValueError(f"selected row rank digest differs from the frozen seed: {row_id}")

        source_config = frozen["sources"][source]
        model_prompt = _field(candidate, source_config["model_prompt_field"])
        raw_prompt = _field(candidate, source_config["raw_prompt_field"])
        if (
            not isinstance(model_prompt, str)
            or not model_prompt
            or not isinstance(raw_prompt, str)
            or not raw_prompt
            or row.get("model_prompt") != model_prompt
            or row.get("raw_prompt") != raw_prompt
        ):
            raise ValueError(f"selected row prompt field mapping differs: {row_id}")
        if (
            row.get("license") != source_config["license"]
            or row.get("license_status") != source_config["license_status"]
        ):
            raise ValueError(f"selected row license differs from the frozen evidence: {row_id}")

        image_raw = row.get("image_path")
        if not isinstance(image_raw, str):
            raise ValueError(f"selected row has no image path: {row_id}")
        image_path = _regular_file(Path(image_raw), label=f"selected image {row_id}")
        if image_path in image_paths:
            raise ValueError("selected image paths must be unique")
        image_paths.add(image_path)
        raw_hash = sha256_file(image_path)
        if row.get("raw_file_sha256") != raw_hash:
            raise ValueError(f"selected raw image hash differs: {row_id}")
        decoded = decode_image_payload(image_path, frozen["decoder"])
        if (
            row.get("decoded_pixel_sha256") != decoded.pixel_sha256
            or row.get("decoded_width") != decoded.width
            or row.get("decoded_height") != decoded.height
        ):
            raise ValueError(f"selected decoded image evidence differs: {row_id}")
        phash = dct_phash_v1(decoded)
        if row.get("phash") != phash:
            raise ValueError(f"selected pHash differs from decoded pixels: {row_id}")
        if raw_hash in raw_hashes or decoded.pixel_sha256 in pixel_hashes:
            raise ValueError("selected payload repeats raw or decoded image content")
        raw_hashes.add(raw_hash)
        pixel_hashes.add(decoded.pixel_sha256)

        token_counts = row.get("token_counts")
        if not isinstance(token_counts, Mapping) or set(token_counts) != set(
            frozen["tokenizer_ids"]
        ):
            raise ValueError(f"selected tokenizer evidence is incomplete: {row_id}")
        if any(not _is_int(count, minimum=1) or count > 77 for count in token_counts.values()):
            raise ValueError(f"selected prompt exceeds a frozen SDXL tokenizer: {row_id}")
        _validate_classifier_check(
            row.get("classifier_check"),
            stratum=stratum,
            required_margin=frozen["classifier_confidence_margin"],
            model_hash=frozen["classifier_model_sha256"],
        )

        exact = row.get("exact_text_checks")
        semantic = row.get("semantic_text_checks")
        prompt_fields = {"model_prompt", "raw_prompt"}
        if not isinstance(exact, Mapping) or set(exact) != prompt_fields:
            raise ValueError(f"selected exact text evidence is incomplete: {row_id}")
        if not isinstance(semantic, Mapping) or set(semantic) != prompt_fields:
            raise ValueError(f"selected semantic text evidence is incomplete: {row_id}")
        for name, prompt in (("model_prompt", model_prompt), ("raw_prompt", raw_prompt)):
            matches = protected_exact.get(normalize_prompt(prompt), ())
            _validate_text_check(
                exact[name],
                label=name,
                prompt=prompt,
                protected_matches=matches,
            )
            _validate_semantic_check(
                semantic[name],
                label=name,
                threshold=frozen["semantic_threshold"],
                model_hash=frozen["semantic_model_sha256"],
            )
        _validate_image_checks(
            row.get("nearest_protected_image"),
            phash_threshold=frozen["phash_threshold"],
            phash_definition_hash=frozen["phash_definition_sha256"],
            image_threshold=frozen["image_threshold"],
            image_model_hash=frozen["image_model_sha256"],
        )
        observed_order.append(
            (
                source_order[source],
                stratum_order[stratum],
                0 if split == "train" else 1,
                rank,
                row_id,
            )
        )

    for source in SOURCES:
        for stratum in STRATA:
            if quotas[(source, stratum, "train")] != 4 or quotas[
                (source, stratum, "validation")
            ] != 2:
                raise ValueError(
                    "every source x stratum quota must be four train and two validation"
                )
            cell = [
                row
                for row in rows
                if row["source"] == source and row["stratum"] == stratum
            ]
            by_rank = sorted(cell, key=lambda row: row["selection_rank"])
            if [row["selection_rank"] for row in by_rank] != list(range(1, 7)):
                raise ValueError("every source x stratum cell must contain ranks one through six")
            if [row["selected_split"] for row in by_rank] != [
                "train",
                "train",
                "train",
                "train",
                "validation",
                "validation",
            ]:
                raise ValueError("selection ranks one-four train and five-six validate")
            digest_order = [(row["selection_digest"], row["id"]) for row in by_rank]
            if len({digest for digest, _ in digest_order}) != 6:
                raise ValueError("selection digest ties reject the source x stratum cell")
            if digest_order != sorted(digest_order):
                raise ValueError("selection ranks do not follow the frozen SHA-256 digest order")
            if any(
                row["fold"] != row["selection_rank"] - 1 for row in by_rank[:4]
            ):
                raise ValueError("every source x stratum cell must map one train row to each fold")
    if observed_order != sorted(observed_order):
        raise ValueError("selected rows are not in deterministic source/stratum/split/rank order")
    return tuple(rows)


def _artifact_projection(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": artifact.get("path"),
        "bytes": artifact.get("bytes"),
        "sha256": artifact.get("sha256"),
    }


def _validate_candidate_parent(parent_dir: Path, repository_root: Path) -> dict[str, Any]:
    """Run the existing canonical validator without asking v1 to be training-ready."""
    return validate_release(
        parent_dir,
        verify_paths=True,
        require_formal_catalog=True,
        require_training_ready=False,
    )


def _validate_parent_binding(
    value: object,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], Path, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "release_id",
        "manifest_sha256",
        "artifacts",
    }:
        raise ValueError("selected child parent binding is incomplete")
    path_raw = value.get("path")
    if not isinstance(path_raw, str):
        raise ValueError("selected child parent manifest path is invalid")
    parent_manifest_path = _regular_file(Path(path_raw), label="candidate parent manifest")
    catalogs_root = _absolute_without_symlinks(
        repository_root / "DATA" / "catalogs", label="candidate catalog root"
    )
    _inside(parent_manifest_path, catalogs_root, label="candidate parent manifest")
    parent_dir = parent_manifest_path.parent
    if parent_dir.parent != catalogs_root:
        raise ValueError("candidate parent must be a direct child of DATA/catalogs")
    parent_hash = sha256_file(parent_manifest_path)
    if value.get("manifest_sha256") != parent_hash:
        raise ValueError("candidate parent manifest hash differs from the child binding")
    parent = _validate_candidate_parent(parent_dir, repository_root)
    if (
        parent.get("schema") != CATALOG_SCHEMA
        or parent.get("release_id") != value.get("release_id")
        or _CATALOG_ID_RE.fullmatch(str(parent.get("release_id"))) is None
        or parent.get("candidate_catalog_complete") is not True
        or parent.get("development_build") is not False
        or parent.get("verify_paths") is not True
    ):
        raise ValueError("candidate parent is not a complete formal inventory")
    if (
        parent.get("training_ready") is not False
        or parent.get("complete") is not False
    ):
        raise ValueError("candidate parent must remain non-training-ready")
    artifacts = parent.get("artifacts")
    bindings = value.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or not isinstance(bindings, list)
        or len(artifacts) != 15
        or len(bindings) != 15
        or tuple(row.get("path") for row in artifacts if isinstance(row, Mapping))
        != PARENT_ARTIFACT_ORDER
        or tuple(row.get("path") for row in bindings if isinstance(row, Mapping))
        != PARENT_ARTIFACT_ORDER
    ):
        raise ValueError("selected child must bind exactly the 15 parent artifacts")
    expected = [_artifact_projection(row) for row in artifacts]
    if bindings != expected:
        raise ValueError("selected child parent artifact hashes differ from the parent manifest")
    for binding in bindings:
        _validate_file_binding(
            binding,
            label=f"candidate parent artifact {binding['path']}",
            release_dir=parent_dir,
        )
    return parent, parent_dir, parent_hash


def _validate_selected_git(
    manifest: Mapping[str, Any],
    *,
    config_bytes: bytes,
    repository_root: Path,
) -> None:
    git = manifest.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("selected child lacks Git provenance")
    _validate_formal_git_record(git)
    _validate_recorded_upstream_ancestry(
        git.get("commit"),
        git.get("upstream_ref"),
        repository_root=repository_root,
    )
    config_repo_path = manifest.get("config_repo_path")
    if not isinstance(config_repo_path, str) or not config_repo_path:
        raise ValueError("selected child config is not bound to a repository path")
    committed = _run_git_bytes(
        ("show", f"{git['commit']}:{config_repo_path}"), repository_root
    )
    if committed != config_bytes:
        raise ValueError("selected child config differs from its recorded Git commit")


def _validate_gate_report(
    descriptor: object,
    *,
    release_dir: Path,
) -> dict[str, Any]:
    path = _validate_file_binding(
        descriptor,
        label="selected-view gate report",
        release_dir=release_dir,
        extra_keys=("schema",),
    )
    if descriptor["schema"] != SELECTED_GATE_REPORT_SCHEMA:  # type: ignore[index]
        raise ValueError("selected-view gate report descriptor has an invalid schema")
    report = _json_object(path, label="selected-view gate report")
    expected = {
        "schema",
        "config_valid",
        "runtime_ready",
        "selection_complete",
        "training_ready",
        "failures",
        "runtime_bindings",
        "runtime_index_counts",
        "candidates_examined",
        "accepted_by_cell",
        "rejection_counts",
        "selected_splits",
    }
    if set(report) != expected or report.get("schema") != SELECTED_GATE_REPORT_SCHEMA:
        raise ValueError("selected-view gate report fields or schema are incomplete")
    for name in (
        "config_valid",
        "runtime_ready",
        "selection_complete",
        "training_ready",
    ):
        if not isinstance(report.get(name), bool):
            raise ValueError(f"selected-view gate report {name} must be boolean")
    failures = report.get("failures")
    if not isinstance(failures, list) or any(
        not isinstance(row, Mapping)
        or set(row) != {"code", "detail"}
        or not isinstance(row.get("code"), str)
        or not row["code"]
        or not isinstance(row.get("detail"), str)
        or not row["detail"]
        for row in failures
    ):
        raise ValueError("selected-view gate report failures are malformed")
    ordered_failures = sorted(failures, key=lambda row: (row["code"], row["detail"]))
    if failures != ordered_failures or len({row["code"] for row in failures}) != len(
        failures
    ):
        raise ValueError("selected-view gate report failures must be sorted and unique")
    if report["training_ready"] is bool(failures):
        raise ValueError("selected-view gate report readiness disagrees with its failures")

    bindings = report.get("runtime_bindings")
    if not isinstance(bindings, Mapping) or any(
        not isinstance(name, str)
        or not name
        or _SHA256_RE.fullmatch(str(digest)) is None
        for name, digest in bindings.items()
    ):
        raise ValueError("selected-view gate report runtime bindings are invalid")
    if report["runtime_ready"] is not bool(bindings):
        raise ValueError("selected-view runtime readiness disagrees with its bindings")
    index_counts = report.get("runtime_index_counts")
    if not isinstance(index_counts, Mapping) or any(
        not isinstance(name, str) or not name or not _is_int(count, minimum=1)
        for name, count in index_counts.items()
    ):
        raise ValueError("selected-view runtime index counts are invalid")
    if report["runtime_ready"] is not bool(index_counts):
        raise ValueError("selected-view runtime readiness disagrees with its index counts")
    if not _is_int(report.get("candidates_examined")):
        raise ValueError("selected-view candidate count is invalid")

    expected_cells = {f"{source}/{stratum}" for source in SOURCES for stratum in STRATA}
    cells = report.get("accepted_by_cell")
    if not isinstance(cells, Mapping) or set(cells) != expected_cells or any(
        not _is_int(count) or count > 6 for count in cells.values()
    ):
        raise ValueError("selected-view accepted cell counts are invalid")
    rejection_counts = report.get("rejection_counts")
    if not isinstance(rejection_counts, Mapping) or any(
        not isinstance(name, str) or not name or not _is_int(count, minimum=1)
        for name, count in rejection_counts.items()
    ):
        raise ValueError("selected-view rejection counts are invalid")
    splits = report.get("selected_splits")
    expected_splits = (
        {"train": 64, "validation": 32}
        if report["selection_complete"]
        else {"train": 0, "validation": 0}
    )
    if splits != expected_splits:
        raise ValueError("selected-view gate report split counts are inconsistent")
    if report["selection_complete"] and any(count != 6 for count in cells.values()):
        raise ValueError("selected-view gate report claims incomplete cell quotas")
    if report["training_ready"] and not all(
        report[name] for name in ("config_valid", "runtime_ready", "selection_complete")
    ):
        raise ValueError("selected-view gate report authorizes an incomplete build")
    return report


def validate_selected_view_release(
    release_dir: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    require_formal: bool = True,
    require_training_ready: bool = True,
    require_gate_report: bool = False,
) -> dict[str, Any]:
    """Validate one content-addressed child release or fail closed."""
    repository_root = _absolute_without_symlinks(
        Path(repository_root).absolute(), label="repository root"
    )
    release_dir = _absolute_without_symlinks(
        Path(release_dir).absolute(), label="selected-view release"
    )
    if require_formal:
        selected_root = _absolute_without_symlinks(
            repository_root / "DATA" / "selected-views", label="selected-view root"
        )
        _inside(release_dir, selected_root, label="selected-view release")
        if release_dir.parent != selected_root:
            raise ValueError("selected-view release must be a direct child of DATA/selected-views")
    manifest_path = _regular_file(release_dir / "manifest.json", label="selected manifest")
    manifest = _json_object(manifest_path, label="selected manifest")
    legacy_manifest_keys = {
        "schema",
        "release_id",
        "complete",
        "training_ready",
        "development_build",
        "git",
        "config_repo_path",
        "parent_catalog",
        "config",
        "selected_payload",
    }
    modern_base_keys = (legacy_manifest_keys - {"selected_payload"}) | {"gate_report"}
    modern_manifest_keys = (
        modern_base_keys | {"selected_payload"}
        if "selected_payload" in manifest
        else modern_base_keys
    )
    if (
        set(manifest) not in (legacy_manifest_keys, modern_manifest_keys)
        or manifest.get("schema") != SELECTED_VIEW_SCHEMA
    ):
        raise ValueError("selected child manifest fields or schema are invalid")
    release_id = manifest.get("release_id")
    if (
        not isinstance(release_id, str)
        or _SELECTED_ID_RE.fullmatch(release_id) is None
        or release_dir.name != release_id
    ):
        raise ValueError("selected child directory does not match its release id")
    core = {
        key: value for key, value in manifest.items() if key not in {"schema", "release_id"}
    }
    if selected_release_id(core) != release_id:
        raise ValueError("selected child release id does not bind its manifest core")
    training_ready = manifest.get("training_ready")
    complete = manifest.get("complete")
    development = manifest.get("development_build")
    if not isinstance(training_ready, bool) or not isinstance(complete, bool) or not isinstance(
        development, bool
    ):
        raise ValueError("selected child readiness flags must be boolean")
    if training_ready:
        if complete is not True or development is not False or "selected_payload" not in manifest:
            raise ValueError("selected child is not a complete formal training release")
    elif complete is not False:
        raise ValueError("non-training selected child cannot be marked complete")
    if require_training_ready and not training_ready:
        raise ValueError("selected child is not training-ready")

    report = (
        _validate_gate_report(manifest["gate_report"], release_dir=release_dir)
        if "gate_report" in manifest
        else None
    )
    if report is None and not training_ready:
        raise ValueError("non-training selected child lacks a gate report")
    if report is None and training_ready and require_gate_report:
        raise ValueError("training-ready selected child lacks a complete gate report")
    if report is not None and report["training_ready"] is not training_ready:
        raise ValueError("selected child readiness differs from its gate report")

    parent, parent_dir, parent_hash = _validate_parent_binding(
        manifest.get("parent_catalog"), repository_root=repository_root
    )
    config_path = _validate_file_binding(
        manifest.get("config"),
        label="selected-view config",
        release_dir=release_dir,
        extra_keys=("schema",),
    )
    if manifest["config"].get("schema") != SELECTED_CONFIG_SCHEMA:
        raise ValueError("selected-view config descriptor has an invalid schema")
    config_bytes = config_path.read_bytes()
    config = _json_object(config_path, label="selected-view config")
    frozen: dict[str, Any] | None
    try:
        frozen = _validate_config(
            config,
            parent=parent,
            parent_dir=parent_dir,
            parent_manifest_sha256=parent_hash,
        )
    except (OSError, ValueError):
        failure_codes = (
            {row["code"] for row in report["failures"]} if report is not None else set()
        )
        if (
            report is None
            or report["config_valid"] is not False
            or training_ready
            or "config_invalid" not in failure_codes
        ):
            raise
        frozen = None
    else:
        if report is not None and report["config_valid"] is not True:
            raise ValueError("selected-view config is valid but gate report claims otherwise")
        if report is not None and any(
            row["code"] == "config_invalid" for row in report["failures"]
        ):
            raise ValueError("selected-view gate report falsely claims an invalid config")
        if report is not None and report["runtime_ready"]:
            if dict(report["runtime_bindings"]) != frozen["runtime_bindings"]:
                raise ValueError("selected-view runtime bindings differ from the frozen config")
            if dict(report["runtime_index_counts"]) != frozen["runtime_index_counts"]:
                raise ValueError("selected-view runtime index counts are incomplete")
    if require_formal:
        _validate_selected_git(
            manifest, config_bytes=config_bytes, repository_root=repository_root
        )

    if "selected_payload" in manifest:
        if frozen is None:
            raise ValueError("selected payload cannot accompany an invalid config")
        payload_path = _validate_file_binding(
            manifest.get("selected_payload"),
            label="selected payload",
            release_dir=release_dir,
            extra_keys=("schema", "rows", "splits"),
        )
        payload_descriptor = manifest["selected_payload"]
        if (
            payload_descriptor.get("schema") != SELECTED_ROW_SCHEMA
            or payload_descriptor.get("rows") != 96
            or payload_descriptor.get("splits") != {"train": 64, "validation": 32}
        ):
            raise ValueError(
                "selected payload descriptor must bind exactly 64 train and 32 validation"
            )
        rows = _validate_selected_rows(
            payload_path, parent_dir=parent_dir, frozen=frozen
        )
        observed_splits = Counter(row["selected_split"] for row in rows)
        if dict(observed_splits) != {"train": 64, "validation": 32}:
            raise ValueError("selected payload split counts differ from its descriptor")
        if report is not None and report["selection_complete"] is not True:
            raise ValueError("selected payload exists despite an incomplete gate report")
    elif report is None or report["selection_complete"] is not False:
        raise ValueError("selected child omits a completed selected payload")
    return manifest


__all__ = [
    "PARENT_ARTIFACT_ORDER",
    "SELECTED_CONFIG_SCHEMA",
    "SELECTED_GATE_REPORT_SCHEMA",
    "SELECTED_IMAGE_MAX_PIXELS",
    "SELECTED_ROW_SCHEMA",
    "SELECTED_VIEW_SCHEMA",
    "STRATA",
    "THRESHOLD_CALIBRATION_SCHEMA",
    "DecodedImage",
    "decode_image_payload",
    "dct_phash_v1",
    "model_binding_sha256",
    "selected_release_id",
    "validate_selected_view_release",
]
