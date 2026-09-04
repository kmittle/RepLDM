"""Build the immutable model and protected-index assets for a selected view.

The selected-view validator deliberately does not create model assets.  This
module is the reproducible, offline producer for those assets: it derives the
protected rows from one frozen parent catalog, encodes them with one local
OpenAI CLIP checkpoint, and writes threshold calibration records before a
selected-view build is attempted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import canonical_json_bytes, iter_jsonl, sha256_file
from .schema import normalize_prompt
from .selected import _field, dct_phash_v1, decode_image_payload

ASSET_SCHEMA = "repldm.selected_view_assets.v1"
CALIBRATION_SCHEMA = "repldm.threshold_calibration.v1"
PROTECTED_INDEX_MANIFEST_SCHEMA = "repldm.protected_index_manifest.v1"
CLIP_MODEL_ID = "openai/ViT-L-14"
CLIP_CHECKPOINT_REVISION = (
    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
)
SDXL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
TOKENIZER_FILE_SHA256: dict[str, dict[str, str]] = {
    "sdxl_tokenizer_1": {
        "merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
        "special_tokens_map.json": "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
        "tokenizer_config.json": "19d7b034cb0cc3ce9766c2231373ab8aa8991fc72e2c8f76558bfaae3de0d563",
        "vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    },
    "sdxl_tokenizer_2": {
        "merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
        "special_tokens_map.json": "f118ab3a983206e4f32583448de6bd6aae4ee21869135cef1f5848a753cdaab6",
        "tokenizer_config.json": "c9d23941f76a41cbd50eda9290f57be7828f0a7a677939e9ef181f7e12bd1bdf",
        "vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    },
}
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_PROMPT_FIELDS: dict[str, dict[str, str]] = {
    "four_k_lsdb": {
        "model_prompt_field": "prompt",
        "raw_prompt_field": "prompt",
    },
    # The PixVerve adapter preserves both captions in source_record.  Binding
    # these nested fields keeps the selected view tied to the normalized
    # adapter contract instead of an upstream metadata spelling.
    "pixverve_95k": {
        "model_prompt_field": "source_record.model_prompt",
        "raw_prompt_field": "source_record.raw_prompt",
    },
}
SOURCE_EVIDENCE_PATHS: dict[str, Path] = {
    "four_k_lsdb": Path("/mnt/miah204/bycao/Sana/data/4klsdb/index.jsonl"),
    "pixverve_95k": Path(
        "/mnt/miah209/bycao/Pixel-Render/data/PixVerve-95K/"
        "PixVerve-95K-metadata/data.jsonl"
    ),
}
CLASS_TEMPLATES: dict[str, list[str]] = {
    "nature": ["a photo of nature", "a nature scene", "a landscape photograph"],
    "urban": [
        "a photo of an urban city scene",
        "a city street photograph",
        "an urban scene",
    ],
    "people": [
        "a photo of people",
        "a portrait photograph of a person",
        "people in a scene",
    ],
    "food": ["a photo of food", "a food photograph", "a meal"],
    "artwork": [
        "a painting or artwork",
        "a fine art image",
        "an artistic illustration",
    ],
    "cgi": [
        "a 3D rendered CGI image",
        "a computer generated render",
        "a 3D scene",
    ],
    "animals": ["a photo of animals", "an animal photograph", "wildlife"],
    "architecture": [
        "a photo of architecture",
        "a building photograph",
        "an architectural scene",
    ],
}


def _ordinary_file(path: Path, *, label: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty ordinary file: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} cannot contain a symlink: {current}")
    return path


def file_binding(path: Path) -> dict[str, Any]:
    """Return the absolute, byte-bound descriptor used by the config schema."""
    path = _ordinary_file(Path(path), label="asset")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(raw):
            os.unlink(raw)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    _atomic_bytes(path, payload)


def _atomic_npz(path: Path, *, ids: Sequence[str], embeddings: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, ids=np.asarray(ids, dtype=str), embeddings=embeddings)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(raw):
            os.unlink(raw)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if not isinstance(row, dict):
            raise TypeError(f"catalog row is not an object: {path}")
        rows.append(row)
    return rows


def _validate_source_prompt_fields(parent: Path) -> None:
    """Check configured prompt paths against real eligible parent rows."""
    observed: set[str] = set()
    for row in iter_jsonl(parent / "training_candidates.jsonl"):
        source = row.get("source")
        if source not in SOURCE_PROMPT_FIELDS or source in observed:
            continue
        if row.get("training_eligible") is not True or row.get("benchmark_exact_match") != []:
            continue
        fields = SOURCE_PROMPT_FIELDS[source]
        try:
            values = [_field(row, fields[key]) for key in ("model_prompt_field", "raw_prompt_field")]
        except ValueError as exc:
            raise ValueError(f"{source} prompt field mapping does not match the parent rows") from exc
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{source} prompt field mapping resolves to an empty value")
        observed.add(source)
        if observed == set(SOURCE_PROMPT_FIELDS):
            return
    missing = sorted(set(SOURCE_PROMPT_FIELDS) - observed)
    raise ValueError("parent has no eligible rows for configured sources: " + ", ".join(missing))


def _source_evidence_from_parent(parent_manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Bind source metadata only when it matches the frozen parent manifest."""
    provenance = parent_manifest.get("source_provenance")
    if not isinstance(provenance, list):
        raise TypeError("parent manifest lacks source provenance")
    by_path = {
        str(row.get("path")): row
        for row in provenance
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for source, path in SOURCE_EVIDENCE_PATHS.items():
        observed = file_binding(path)
        expected = by_path.get(str(path))
        if (
            not isinstance(expected, Mapping)
            or expected.get("bytes") != observed["bytes"]
            or expected.get("sha256") != observed["sha256"]
        ):
            raise ValueError(f"{source} source metadata differs from the parent manifest")
        result[source] = [observed]
    return result


def unique_protected_prompts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Choose one stable record for every normalized protected prompt."""
    chosen: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("protected prompt row lacks a stable id")
        normalized = normalize_prompt(prompt)
        previous = chosen.get(normalized)
        if previous is None or row_id < str(previous["id"]):
            chosen[normalized] = row
    result = []
    for normalized, row in sorted(
        chosen.items(), key=lambda item: (item[0], str(item[1]["id"]))
    ):
        result.append(
            {
                "id": str(row["id"]),
                "prompt": str(row["prompt"]),
                "normalized": normalized,
            }
        )
    return result


def unique_protected_images(
    rows: Sequence[Mapping[str, Any]], *, validate_files: bool = True
) -> list[dict[str, Any]]:
    """Choose one stable record for every protected physical image path.

    ``validate_files=False`` is reserved for validation against an already
    frozen protected manifest.  The parent holdout bytes and manifest image
    paths remain bound in that mode, while avoiding one network filesystem
    ``stat`` per image.  Asset production always keeps the default strict
    file checks.
    """
    chosen: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        raw = row.get("image_path")
        if not isinstance(raw, str) or not raw:
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("protected image row lacks a stable id")
        if validate_files:
            try:
                path = _ordinary_file(Path(raw), label="protected image")
            except ValueError as exc:
                if not Path(raw).exists():
                    raise FileNotFoundError(f"protected image is missing: {raw}") from exc
                raise
        else:
            # This is deliberately a pure lexical normalization.  The parent
            # holdout file is hash-bound and the resulting path is compared
            # with the frozen manifest before any index is accepted.
            path = Path(os.path.abspath(os.fspath(Path(raw))))
        key = str(path)
        previous = chosen.get(key)
        if previous is None or row_id < str(previous["id"]):
            chosen[key] = row
    result = []
    for image_path, row in sorted(
        chosen.items(), key=lambda item: (item[0], str(item[1]["id"]))
    ):
        result.append({"id": str(row["id"]), "image_path": image_path})
    return result


def protected_ids_sha256(ids: Sequence[object]) -> str:
    """Hash an ordered protected-ID sequence without lossy string joining."""
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("protected IDs must be non-empty strings")
    return hashlib.sha256(canonical_json_bytes(list(ids))).hexdigest()


def _protected_sample_ids(ids: Sequence[str], count: int = 128) -> list[list[str]]:
    """Return the deterministic positive/cyclic-neighbor ID pairs used by calibration."""
    if len(ids) < 2:
        raise ValueError("at least two protected IDs are required for calibration")
    base = list(ids[: min(count, len(ids))])
    return [[value] for value in base] + [
        [value, base[(index + 1) % len(base)]]
        for index, value in enumerate(base)
    ]


def _protected_image_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    decoder: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Collect byte and decoded-pixel identities for the canonical image cohort."""
    evidence: list[dict[str, Any]] = []
    for row in rows:
        image_path = _ordinary_file(Path(str(row["image_path"])), label="protected image")
        decoded = decode_image_payload(image_path, decoder)
        evidence.append(
            {
                "id": str(row["id"]),
                "image_path": str(image_path),
                "raw_file_sha256": sha256_file(image_path),
                "decoded_pixel_sha256": decoded.pixel_sha256,
                "decoded_width": decoded.width,
                "decoded_height": decoded.height,
            }
        )
    return evidence


def derive_protected_index_manifest(
    parent_release: Path,
    *,
    parent_release_id: str,
    parent_manifest_sha256: str,
    decoder: Mapping[str, Any],
    holdout_rows: Sequence[Mapping[str, Any]] | None = None,
    prompt_rows: Sequence[Mapping[str, Any]] | None = None,
    image_rows: Sequence[Mapping[str, Any]] | None = None,
    image_evidence: Sequence[Mapping[str, Any]] | None = None,
    verify_image_evidence: bool = True,
) -> dict[str, Any]:
    """Derive the immutable protected cohort identity from one parent catalog.

    ``manifest_sha256`` is a logical hash of the canonical manifest core.  It
    intentionally differs from the byte hash used for the manifest file
    binding, so changing indentation or JSON key order cannot change cohort
    identity while changing an ID/path/hash always does.
    """
    if not isinstance(verify_image_evidence, bool):
        raise TypeError("verify_image_evidence must be a boolean")
    parent_release = Path(parent_release).resolve()
    holdout_path = _ordinary_file(
        parent_release / "benchmark_holdouts.jsonl", label="protected holdout catalog"
    )
    canonical_rows = _load_rows(holdout_path)
    if holdout_rows is None:
        rows = canonical_rows
    else:
        try:
            rows = [dict(row) for row in holdout_rows]
        except (TypeError, ValueError) as exc:
            raise ValueError("holdout_rows must contain mapping rows") from exc
        if rows != canonical_rows:
            raise ValueError("holdout_rows differ from the current holdout catalog")

    canonical_prompts = unique_protected_prompts(rows)
    if prompt_rows is None:
        prompts = canonical_prompts
    else:
        try:
            prompts = [dict(row) for row in prompt_rows]
        except (TypeError, ValueError) as exc:
            raise ValueError("prompt_rows must contain mapping rows") from exc
        if prompts != canonical_prompts:
            raise ValueError("prompt_rows differ from the current holdout cohort")

    canonical_images = unique_protected_images(
        rows, validate_files=verify_image_evidence
    )
    if image_rows is None:
        images = canonical_images
    else:
        try:
            images = [dict(row) for row in image_rows]
        except (TypeError, ValueError) as exc:
            raise ValueError("image_rows must contain mapping rows") from exc
        if images != canonical_images:
            raise ValueError("image_rows differ from the current holdout cohort")

    if image_evidence is None:
        if not verify_image_evidence:
            raise ValueError(
                "metadata-only manifest derivation requires frozen image_evidence"
            )
        image_records = _protected_image_evidence(images, decoder=decoder)
    else:
        try:
            image_records = [dict(value) for value in image_evidence]
        except (TypeError, ValueError) as exc:
            raise ValueError("image_evidence must contain mapping rows") from exc
    expected_image_ids = [str(row["id"]) for row in images]
    expected_image_paths = [
        str(Path(os.path.abspath(os.fspath(Path(str(row["image_path"]))))))
        for row in images
    ]
    if len(image_records) != len(images):
        raise ValueError("protected image evidence count differs from the parent cohort")
    for index, record in enumerate(image_records):
        if set(record) != {
            "id",
            "image_path",
            "raw_file_sha256",
            "decoded_pixel_sha256",
            "decoded_width",
            "decoded_height",
        }:
            raise ValueError("protected image evidence fields are incomplete")
        expected = {
            "id": expected_image_ids[index],
            "image_path": expected_image_paths[index],
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("protected image evidence order differs from the parent cohort")
        for key in ("raw_file_sha256", "decoded_pixel_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"protected image evidence has an invalid {key}")
        for key in ("decoded_width", "decoded_height"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"protected image evidence has an invalid {key}")
        if verify_image_evidence:
            # Re-read the bound bytes when a caller supplies cached evidence.
            # This closes a time-of-check gap between CLIP encoding and
            # manifest write.  Runtime/config validation intentionally uses
            # the frozen evidence-only mode below.
            image_path = _ordinary_file(
                Path(str(record["image_path"])), label="protected image"
            )
            if sha256_file(image_path) != record["raw_file_sha256"]:
                raise ValueError("protected image raw bytes changed during manifest build")
            decoded = decode_image_payload(image_path, decoder)
            if (
                decoded.pixel_sha256 != record["decoded_pixel_sha256"]
                or decoded.width != record["decoded_width"]
                or decoded.height != record["decoded_height"]
            ):
                raise ValueError("protected image decoded pixels changed during manifest build")

    prompt_records = [
        {
            "id": str(row["id"]),
            "normalized_prompt_sha256": hashlib.sha256(
                str(row["normalized"]).encode("utf-8")
            ).hexdigest(),
        }
        for row in prompts
    ]
    image_records = [
        {
            "id": str(record["id"]),
            "image_path": str(record["image_path"]),
            "raw_file_sha256": str(record["raw_file_sha256"]),
            "decoded_pixel_sha256": str(record["decoded_pixel_sha256"]),
            "decoded_width": int(record["decoded_width"]),
            "decoded_height": int(record["decoded_height"]),
        }
        for record in image_records
    ]
    prompt_ids = [row["id"] for row in prompt_records]
    image_ids = [row["id"] for row in image_records]
    core: dict[str, Any] = {
        "schema": PROTECTED_INDEX_MANIFEST_SCHEMA,
        "parent_catalog": {
            "release_id": str(parent_release_id),
            "manifest_sha256": str(parent_manifest_sha256),
        },
        "holdout": {
            "path": holdout_path.name,
            "bytes": holdout_path.stat().st_size,
            "sha256": sha256_file(holdout_path),
        },
        "decoder_sha256": hashlib.sha256(canonical_json_bytes(dict(decoder))).hexdigest(),
        "counts": {
            "holdout_rows": len(rows),
            "normalized_unique_prompts": len(prompt_records),
            "unique_images": len(image_records),
        },
        "prompt_ids_sha256": protected_ids_sha256(prompt_ids),
        "image_ids_sha256": protected_ids_sha256(image_ids),
        "prompts": prompt_records,
        "images": image_records,
    }
    return {
        **core,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest(),
    }


def _load_clip(checkpoint: Path, device: str):
    import clip

    model, preprocess = clip.load(
        str(_ordinary_file(checkpoint, label="CLIP checkpoint")),
        device=device,
        jit=False,
        download_root=str(checkpoint.parent),
    )
    model.eval()
    return model, preprocess


def _load_bound_clip(checkpoint: Path, device: str) -> tuple[dict[str, Any], Any, Any]:
    """Verify the pinned checkpoint before allowing deserialization."""
    descriptor = _model_descriptor(checkpoint)
    model, preprocess = _load_clip(checkpoint, device)
    return descriptor, model, preprocess


def _load_tokenizer(root: Path):
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained(
        str(Path(root).resolve()), local_files_only=True
    )


def _encode_texts(model: Any, tokenizer: Any, texts: Sequence[str], *, device: str, batch_size: int) -> np.ndarray:
    import torch

    outputs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        with torch.inference_mode():
            features = model.encode_text(ids).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        outputs.append(features.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def _decode_clip_tensor(path: Path, preprocess: Any, decoder: Mapping[str, Any]):
    from PIL import Image

    decoded = decode_image_payload(path, decoder)
    image = Image.frombytes("RGB", (decoded.width, decoded.height), decoded.rgb_bytes)
    return decoded, preprocess(image)


def _encode_images(
    model: Any,
    preprocess: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    decoder: Mapping[str, Any],
    device: str,
    batch_size: int,
    with_hashes: bool = False,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    import torch

    outputs: list[np.ndarray] = []
    phashes: list[str] = []
    evidence: list[dict[str, Any]] = []
    tensors: list[Any] = []
    for index, row in enumerate(rows):
        decoded, tensor = _decode_clip_tensor(Path(str(row["image_path"])), preprocess, decoder)
        tensors.append(tensor)
        if with_hashes:
            image_path = _ordinary_file(
                Path(str(row["image_path"])), label="protected image"
            )
            phashes.append(dct_phash_v1(decoded))
            evidence.append(
                {
                    "id": str(row["id"]),
                    "image_path": str(image_path),
                    "raw_file_sha256": sha256_file(image_path),
                    "decoded_pixel_sha256": decoded.pixel_sha256,
                    "decoded_width": decoded.width,
                    "decoded_height": decoded.height,
                }
            )
        if len(tensors) == batch_size or index == len(rows) - 1:
            batch = torch.stack(tensors).to(device)
            with torch.inference_mode():
                features = model.encode_image(batch).float()
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            outputs.append(features.cpu().numpy().astype(np.float32, copy=False))
            tensors.clear()
        if (index + 1) % 256 == 0:
            print(f"encoded protected images: {index + 1:,}/{len(rows):,}", flush=True)
    return np.concatenate(outputs, axis=0), phashes, evidence


def _descriptor(path: Path, **extra: Any) -> dict[str, Any]:
    return {**file_binding(path), **extra}


def _model_descriptor(checkpoint: Path) -> dict[str, Any]:
    binding = _descriptor(checkpoint)
    if binding["sha256"] != CLIP_CHECKPOINT_REVISION:
        raise ValueError(
            "CLIP checkpoint bytes do not match the pinned selected-view revision"
        )
    return {
        "id": CLIP_MODEL_ID,
        "revision": CLIP_CHECKPOINT_REVISION,
        "files": [binding],
    }


def _tokenizer_descriptor(root: Path, tokenizer_id: str) -> dict[str, Any]:
    expected_hashes = TOKENIZER_FILE_SHA256.get(tokenizer_id)
    if expected_hashes is None:
        raise ValueError(f"no frozen tokenizer registration exists for {tokenizer_id}")
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
        raise ValueError(f"cannot inspect tokenizer directory: {root}") from exc
    if present != set(expected_hashes):
        raise ValueError(
            f"tokenizer files differ from the frozen registration for {tokenizer_id}"
        )
    files = []
    for name in sorted(expected_hashes):
        binding = _descriptor(root / name)
        if binding["sha256"] != expected_hashes[name]:
            raise ValueError(
                f"tokenizer file hash differs from the frozen registration: {name}"
            )
        files.append(binding)
    return {
        "id": tokenizer_id,
        "model": {"id": "stabilityai/stable-diffusion-xl-base-1.0", "revision": SDXL_REVISION, "files": files},
        "max_tokens": 77,
        "add_special_tokens": True,
        "truncation": False,
    }


def _similarity_rows(
    values: Sequence[float],
    *,
    label: str,
    protected_ids: Sequence[Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    if protected_ids is not None and len(values) != len(protected_ids):
        raise ValueError("calibration values and protected IDs have different lengths")
    rows = []
    for index, value in enumerate(values):
        row: dict[str, Any] = {
            "pair": index,
            "label": label,
            "similarity": float(value),
        }
        if protected_ids is not None:
            row["protected_ids"] = list(protected_ids[index])
        rows.append(row)
    return rows


def _cyclic_hamming_distances(values: Sequence[int]) -> list[int]:
    """Compare adjacent 64-bit hashes without coercing them through float64."""
    if len(values) < 2:
        raise ValueError("at least two pHashes are required for calibration")
    neighbors = [*values[1:], values[0]]
    return [(int(left) ^ int(right)).bit_count() for left, right in zip(values, neighbors)]


def _calibration_artifact(
    path: Path,
    *,
    metric: str,
    selected_value: float,
    comparison: str,
    source_path: Path,
    positive_count: int,
    negative_count: int,
    model_hash: str,
    protected_manifest_sha256: str | None = None,
    protected_index_sha256: str | None = None,
    protected_sample_ids: Sequence[Sequence[str]] | None = None,
    sample_ids: Sequence[str] | None = None,
) -> None:
    total = positive_count + negative_count
    payload: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "metric": metric,
        "selected_value": selected_value,
        "comparison": comparison,
        "sample_count": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "source": file_binding(source_path),
        "model_binding_sha256": model_hash,
    }
    if (
        protected_manifest_sha256 is None
        or protected_index_sha256 is None
        or protected_sample_ids is None
    ):
        if any(
            value is not None
            for value in (
                protected_manifest_sha256,
                protected_index_sha256,
                protected_sample_ids,
            )
        ):
            raise ValueError("protected calibration binding must be complete")
    else:
        protected_rows = [list(row) for row in protected_sample_ids]
        payload["protected_sample_binding"] = {
            "manifest_sha256": protected_manifest_sha256,
            "index_sha256": protected_index_sha256,
            "sample_ids": protected_rows,
            "sample_ids_sha256": hashlib.sha256(
                canonical_json_bytes(protected_rows)
            ).hexdigest(),
        }
    if sample_ids is not None:
        if any(not isinstance(value, str) or not value for value in sample_ids):
            raise ValueError("calibration sample IDs must be non-empty strings")
        ordered_ids = list(sample_ids)
        payload["sample_ids"] = ordered_ids
        payload["sample_ids_sha256"] = hashlib.sha256(
            canonical_json_bytes(ordered_ids)
        ).hexdigest()
    _atomic_json(path, payload)


def _keyword_class(prompt: str) -> str | None:
    lower = prompt.lower()
    keywords = {
        "nature": ("landscape", "forest", "mountain", "beach", "ocean", "river", "sunset", "tree"),
        "urban": ("city", "urban", "street", "downtown", "skyscraper", "subway"),
        "people": ("woman", "man", "person", "people", "portrait", "child", "family"),
        "food": ("food", "apple", "cake", "meal", "dish", "restaurant", "fruit"),
        "artwork": ("painting", "illustration", "artwork", "drawing", "sketch", "sculpture"),
        "cgi": ("3d", "cgi", "render", "unreal", "blender", "digital render"),
        "animals": ("dog", "cat", "bird", "horse", "animal", "wildlife", "bear"),
        "architecture": ("architecture", "building", "house", "room", "interior", "castle", "bridge"),
    }
    for name in STRATA:
        if any(token in lower for token in keywords[name]):
            return name
    return None


def _build_classifier_calibration(
    parent: Path,
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    *,
    decoder: Mapping[str, Any],
    device: str,
    batch_size: int,
    source_path: Path,
    artifact_path: Path,
    model_hash: str,
) -> float:
    by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATA}
    seen: set[str] = set()
    for row in iter_jsonl(parent / "training_candidates.jsonl"):
        if row.get("modality") != "image_text" or not row.get("image_path"):
            continue
        expected = _keyword_class(str(row.get("prompt", "")))
        if expected is None:
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise RuntimeError("classifier calibration candidate lacks a stable id")
        path = str(Path(str(row["image_path"])).resolve())
        if path in seen:
            continue
        seen.add(path)
        by_class[expected].append(
            {
                "id": row_id,
                "image_path": path,
                "expected": expected,
            }
        )
    minimum_per_class = 32
    if any(len(by_class[name]) < minimum_per_class for name in STRATA):
        missing = ", ".join(
            f"{name}={len(by_class[name])}"
            for name in STRATA
            if len(by_class[name]) < minimum_per_class
        )
        raise RuntimeError(
            "not enough keyword-labelled image-text candidates per stratum: " + missing
        )
    candidates = []
    for name in STRATA:
        candidates.extend(
            sorted(by_class[name], key=lambda row: (row["id"], row["image_path"]))[
                :minimum_per_class
            ]
        )
    embeddings, _, _ = _encode_images(
        model,
        preprocess,
        candidates,
        decoder=decoder,
        device=device,
        batch_size=batch_size,
    )
    text_rows = [template for name in STRATA for template in CLASS_TEMPLATES[name]]
    text_features = _encode_texts(model, tokenizer, text_rows, device=device, batch_size=64)
    scores = embeddings @ text_features.T
    margins = []
    predicted_correct = []
    for index, row in enumerate(candidates):
        grouped = []
        offset = 0
        for name in STRATA:
            width = len(CLASS_TEMPLATES[name])
            grouped.append(float(scores[index, offset : offset + width].max()))
            offset += width
        order = np.argsort(np.asarray(grouped))
        margin = grouped[int(order[-1])] - grouped[int(order[-2])]
        margins.append(margin)
        predicted_correct.append(STRATA[int(order[-1])] == row["expected"])
    threshold = float(np.median(np.asarray(margins)))
    positive_count = sum(value >= threshold for value in margins)
    negative_count = len(margins) - positive_count
    if positive_count == 0 or negative_count == 0:
        threshold = float(np.mean(np.asarray(margins)))
        positive_count = sum(value >= threshold for value in margins)
        negative_count = len(margins) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise RuntimeError("classifier calibration did not produce two classes")
    _atomic_jsonl(
        source_path,
        [
            {
                "id": row["id"],
                "expected_class": row["expected"],
                "margin": float(value),
                "predicted_class_correct": bool(correct),
                "label": "positive" if value >= threshold else "negative",
            }
            for row, value, correct in zip(candidates, margins, predicted_correct)
        ],
    )
    _calibration_artifact(
        artifact_path,
        metric="classifier_confidence_margin",
        selected_value=threshold,
        comparison="reject_below_margin",
        source_path=source_path,
        positive_count=positive_count,
        negative_count=negative_count,
        model_hash=model_hash,
        sample_ids=[str(row["id"]) for row in candidates],
    )
    return threshold


def build_selected_assets(
    *,
    parent_release: Path,
    output_dir: Path,
    config_output: Path,
    clip_checkpoint: Path,
    tokenizer_root: Path,
    tokenizer_root_2: Path,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Build all local selected-view dependencies and the frozen config."""
    from PIL import __version__ as pillow_version
    from PIL import features

    parent_release = Path(parent_release).resolve()
    output_dir = Path(output_dir).resolve()
    config_output = Path(config_output).resolve()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    manifest_path = _ordinary_file(parent_release / "manifest.json", label="parent manifest")
    parent_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_hash = sha256_file(manifest_path)
    if parent_manifest.get("release_id") != parent_release.name:
        raise ValueError("parent manifest directory and release_id differ")
    holdout_rows = _load_rows(parent_release / "benchmark_holdouts.jsonl")
    prompt_rows = unique_protected_prompts(holdout_rows)
    image_rows = unique_protected_images(holdout_rows)
    if len(prompt_rows) != int(parent_manifest["protected_normalized_unique_prompts"]):
        raise ValueError("protected prompt count differs from parent manifest")
    if len(image_rows) != int(parent_manifest["protected_unique_images"]):
        raise ValueError("protected image count differs from parent manifest")
    _validate_source_prompt_fields(parent_release)
    source_evidence = _source_evidence_from_parent(parent_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    decoder = {
        "library": "Pillow",
        "version": pillow_version,
        "littlecms_version": features.version("littlecms2"),
        "exif_transpose": True,
        "icc_to_srgb": True,
        "output_mode": "RGB",
        "pixel_hash": "sha256_rgb_u64be_width_height_bytes_v1",
    }
    checkpoint = _ordinary_file(Path(clip_checkpoint), label="CLIP checkpoint")
    tokenizer_root = Path(os.path.abspath(os.fspath(tokenizer_root)))
    tokenizer_root_2 = Path(os.path.abspath(os.fspath(tokenizer_root_2)))
    model_descriptor, model, preprocess = _load_bound_clip(checkpoint, device)
    model_hash = hashlib.sha256(canonical_json_bytes(model_descriptor)).hexdigest()
    tokenizer_descriptor = _tokenizer_descriptor(tokenizer_root, "sdxl_tokenizer_1")
    tokenizer_descriptor_2 = _tokenizer_descriptor(tokenizer_root_2, "sdxl_tokenizer_2")
    tokenizer = _load_tokenizer(tokenizer_root)
    _load_tokenizer(tokenizer_root_2)

    text_embeddings = _encode_texts(
        model,
        tokenizer,
        [row["prompt"] for row in prompt_rows],
        device=device,
        batch_size=max(1, batch_size * 4),
    )
    text_index = output_dir / "protected_semantic_text.npz"
    _atomic_npz(text_index, ids=[row["id"] for row in prompt_rows], embeddings=text_embeddings)

    image_embeddings, phashes, image_evidence = _encode_images(
        model,
        preprocess,
        image_rows,
        decoder=decoder,
        device=device,
        batch_size=batch_size,
        with_hashes=True,
    )
    image_index = output_dir / "protected_image_embedding.npz"
    _atomic_npz(image_index, ids=[row["id"] for row in image_rows], embeddings=image_embeddings)
    phash_index = output_dir / "protected_phash.jsonl"
    _atomic_jsonl(
        phash_index,
        ({"id": row["id"], "phash": value} for row, value in zip(image_rows, phashes)),
    )

    protected_manifest = derive_protected_index_manifest(
        parent_release,
        parent_release_id=str(parent_manifest["release_id"]),
        parent_manifest_sha256=parent_hash,
        decoder=decoder,
        holdout_rows=holdout_rows,
        prompt_rows=prompt_rows,
        image_rows=image_rows,
        image_evidence=image_evidence,
    )
    protected_manifest_path = output_dir / "protected_index_manifest.json"
    _atomic_json(protected_manifest_path, protected_manifest)
    protected_manifest_hash = str(protected_manifest["manifest_sha256"])
    protected_manifest_file = file_binding(protected_manifest_path)
    semantic_index_file = file_binding(text_index)
    phash_index_file = file_binding(phash_index)
    image_index_file = file_binding(image_index)
    protected_prompt_ids = [str(row["id"]) for row in prompt_rows]
    protected_image_ids = [str(row["id"]) for row in image_rows]
    protected_prompt_samples = _protected_sample_ids(protected_prompt_ids)
    protected_image_samples = _protected_sample_ids(protected_image_ids)

    calibration_dir = output_dir / "calibration"
    # Positive examples are deterministic exact/near transformations; the
    # negative examples are fixed cyclic neighbors from the protected cohort.
    semantic_positive = _encode_texts(
        model,
        tokenizer,
        [row["prompt"] + " ." for row in prompt_rows[:128]],
        device=device,
        batch_size=128,
    )
    semantic_base = text_embeddings[: len(semantic_positive)]
    semantic_pos_values = np.sum(semantic_positive * semantic_base, axis=1)
    semantic_neg_values = np.sum(
        text_embeddings[:128] * np.roll(text_embeddings[:128], -1, axis=0), axis=1
    )
    semantic_threshold = max(0.80, min(0.92, float(np.quantile(semantic_neg_values, 0.99) + 0.03)))
    semantic_source = calibration_dir / "semantic_text.jsonl"
    _atomic_jsonl(
        semantic_source,
        _similarity_rows(
            semantic_pos_values,
            label="positive",
            protected_ids=protected_prompt_samples[: len(semantic_pos_values)],
        )
        + _similarity_rows(
            semantic_neg_values,
            label="negative",
            protected_ids=protected_prompt_samples[len(semantic_pos_values) :],
        ),
    )
    semantic_artifact = calibration_dir / "semantic_text_calibration.json"
    _calibration_artifact(
        semantic_artifact,
        metric="cosine_similarity",
        selected_value=semantic_threshold,
        comparison="reject_at_or_above",
        source_path=semantic_source,
        positive_count=len(semantic_pos_values),
        negative_count=len(semantic_neg_values),
        model_hash=model_hash,
        protected_manifest_sha256=protected_manifest_hash,
        protected_index_sha256=semantic_index_file["sha256"],
        protected_sample_ids=protected_prompt_samples,
    )

    phash_values = [int(value, 16) for value in phashes[:128]]
    phash_positive = [0 for _ in phash_values]
    phash_negative = _cyclic_hamming_distances(phash_values)
    phash_threshold = int(max(4, min(12, np.quantile(phash_negative, 0.01))))
    if all(value <= phash_threshold for value in phash_negative):
        phash_threshold = max(0, int(np.median(phash_negative)) - 1)
    phash_source = calibration_dir / "phash.jsonl"
    _atomic_jsonl(
        phash_source,
        [
            *(
                {
                    "pair": i,
                    "label": "positive",
                    "distance": value,
                    "protected_ids": protected_image_samples[i],
                }
                for i, value in enumerate(phash_positive)
            ),
            *(
                {
                    "pair": i,
                    "label": "negative",
                    "distance": value,
                    "protected_ids": protected_image_samples[len(phash_positive) + i],
                }
                for i, value in enumerate(phash_negative)
            ),
        ],
    )
    phash_definition = {
        "implementation": "dct_phash_v1",
        "hash_bits": 64,
        "resize": 32,
        "low_frequency_size": 8,
        "exclude_dc": True,
    }
    phash_definition_hash = hashlib.sha256(canonical_json_bytes(phash_definition)).hexdigest()
    phash_artifact = calibration_dir / "phash_calibration.json"
    _calibration_artifact(
        phash_artifact,
        metric="hamming_distance",
        selected_value=phash_threshold,
        comparison="reject_at_or_below",
        source_path=phash_source,
        positive_count=len(phash_positive),
        negative_count=len(phash_negative),
        model_hash=phash_definition_hash,
        protected_manifest_sha256=protected_manifest_hash,
        protected_index_sha256=phash_index_file["sha256"],
        protected_sample_ids=protected_image_samples,
    )

    # Exact-image positives are a conservative lower bound for the embedding
    # gate.  Cyclic neighbors provide a reproducible negative distribution.
    image_positive = np.ones(min(128, len(image_embeddings)), dtype=np.float32)
    image_negative = np.sum(
        image_embeddings[: len(image_positive)]
        * np.roll(image_embeddings[: len(image_positive)], -1, axis=0),
        axis=1,
    )
    image_threshold = max(0.90, min(0.985, float(np.quantile(image_negative, 0.99) + 0.02)))
    image_source = calibration_dir / "image_embedding.jsonl"
    _atomic_jsonl(
        image_source,
        _similarity_rows(
            image_positive,
            label="positive",
            protected_ids=protected_image_samples[: len(image_positive)],
        )
        + _similarity_rows(
            image_negative,
            label="negative",
            protected_ids=protected_image_samples[len(image_positive) :],
        ),
    )
    image_artifact = calibration_dir / "image_embedding_calibration.json"
    _calibration_artifact(
        image_artifact,
        metric="cosine_similarity",
        selected_value=image_threshold,
        comparison="reject_at_or_above",
        source_path=image_source,
        positive_count=len(image_positive),
        negative_count=len(image_negative),
        model_hash=model_hash,
        protected_manifest_sha256=protected_manifest_hash,
        protected_index_sha256=image_index_file["sha256"],
        protected_sample_ids=protected_image_samples,
    )

    classifier_source = calibration_dir / "classifier.jsonl"
    classifier_artifact = calibration_dir / "classifier_calibration.json"
    classifier_threshold = _build_classifier_calibration(
        parent_release,
        model,
        preprocess,
        tokenizer,
        decoder=decoder,
        device=device,
        batch_size=batch_size,
        source_path=classifier_source,
        artifact_path=classifier_artifact,
        model_hash=model_hash,
    )

    config = {
        "schema": "repldm.selected_view_config.v1",
        "view_id": "opd_dpo_rl_selected_view_v1",
        "parent_catalog": {"release_id": parent_manifest["release_id"], "manifest_sha256": parent_hash},
        "sources": {
            "four_k_lsdb": {
                **SOURCE_PROMPT_FIELDS["four_k_lsdb"],
                "license": "CC-BY-4.0",
                "license_status": "verified_from_dataset_card",
                "license_evidence": source_evidence["four_k_lsdb"],
            },
            "pixverve_95k": {
                **SOURCE_PROMPT_FIELDS["pixverve_95k"],
                "license": "Apache-2.0",
                "license_status": "verified_from_dataset_card",
                "license_evidence": source_evidence["pixverve_95k"],
            },
        },
        "strata": list(STRATA),
        "quotas": {
            "train_per_source_stratum": 4,
            "validation_per_source_stratum": 2,
            "train_total": 64,
            "validation_total": 32,
        },
        "selection": {
            "seed": "2026090401",
            "algorithm": "sha256_seed_nul_record_id_v1",
            "tie_rule": "reject",
        },
        "classifier": {
            "model": model_descriptor,
            "class_templates": CLASS_TEMPLATES,
            "confidence_margin": classifier_threshold,
            "tie_rule": "reject",
            "calibration": file_binding(classifier_artifact),
        },
        "tokenizers": [tokenizer_descriptor, tokenizer_descriptor_2],
        "clip_tokenizer_id": "sdxl_tokenizer_1",
        "decoder": decoder,
        "semantic_text": {
            "model": model_descriptor,
            "threshold": semantic_threshold,
            "comparison": "reject_at_or_above",
            "calibration": file_binding(semantic_artifact),
        },
        "phash": {
            **phash_definition,
            "threshold": phash_threshold,
            "comparison": "reject_at_or_below",
            "calibration": file_binding(phash_artifact),
        },
        "image_embedding": {
            "model": model_descriptor,
            "threshold": image_threshold,
            "comparison": "reject_at_or_above",
            "calibration": file_binding(image_artifact),
        },
        "protected_index": {
            "holdout_rows": len(holdout_rows),
            "normalized_unique_prompts": len(prompt_rows),
            "unique_images": len(image_rows),
            "manifest": protected_manifest_file,
            "manifest_sha256": protected_manifest_hash,
            "semantic_text": semantic_index_file,
            "phash": phash_index_file,
            "image_embedding": image_index_file,
            "index_bindings": {
                "semantic_text": {
                    "manifest_sha256": protected_manifest_hash,
                    "ids_sha256": protected_ids_sha256(protected_prompt_ids),
                },
                "phash": {
                    "manifest_sha256": protected_manifest_hash,
                    "ids_sha256": protected_ids_sha256(protected_image_ids),
                },
                "image_embedding": {
                    "manifest_sha256": protected_manifest_hash,
                    "ids_sha256": protected_ids_sha256(protected_image_ids),
                },
            },
        },
    }
    _atomic_json(config_output, config)
    asset_manifest = {
        "schema": ASSET_SCHEMA,
        "parent_release_id": parent_manifest["release_id"],
        "parent_manifest_sha256": parent_hash,
        "clip_checkpoint": file_binding(checkpoint),
        "tokenizer_roots": [str(tokenizer_root), str(tokenizer_root_2)],
        "device": device,
        "batch_size": batch_size,
        "protected_prompt_count": len(prompt_rows),
        "protected_image_count": len(image_rows),
        "artifacts": {
            name: file_binding(path)
            for name, path in {
                "semantic_text": text_index,
                "phash": phash_index,
                "image_embedding": image_index,
                "protected_index_manifest": protected_manifest_path,
                "classifier_calibration": classifier_artifact,
                "semantic_calibration": semantic_artifact,
                "phash_calibration": phash_artifact,
                "image_calibration": image_artifact,
                "config": config_output,
            }.items()
        },
    }
    _atomic_json(output_dir / "asset_manifest.json", asset_manifest)
    return config


__all__ = [
    "ASSET_SCHEMA",
    "CLASS_TEMPLATES",
    "PROTECTED_INDEX_MANIFEST_SCHEMA",
    "SOURCE_EVIDENCE_PATHS",
    "SOURCE_PROMPT_FIELDS",
    "TOKENIZER_FILE_SHA256",
    "build_selected_assets",
    "derive_protected_index_manifest",
    "file_binding",
    "protected_ids_sha256",
    "unique_protected_images",
    "unique_protected_prompts",
]
