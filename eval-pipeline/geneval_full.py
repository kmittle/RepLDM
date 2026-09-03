"""Fail-closed, resumable GenEval evaluation for renderer checkpoints.

The upstream GenEval evaluator writes one loose JSONL file and does not know
anything about a RepLDM checkpoint, seed manifest, or paired experiment.  This
module supplies that missing boundary.  It validates the frozen 553-prompt
source, binds exactly four samples to every prompt, records one shared seed
cohort for all methods, materializes the directory layout expected by the
upstream evaluator, and aggregates only a complete 2,212-image result.

The object detector itself is deliberately not copied into this repository.
``run_official_evaluator`` invokes the reviewed local Sana/GenEval checkout in
an explicitly selected environment; all resulting files are still checked and
hashed here.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = Path(
    "/mnt/miah204/bycao/Sana/diffusion/post_training/dataset/geneval/"
    "test_metadata.jsonl"
)
DEFAULT_METADATA_SHA256 = (
    "eb9d9f624e76c2efb1bf8495494121d30e86d73358022502b071bebcfc8a2ad7"
)
DEFAULT_CONFIG_PATH = ROOT / "eval-pipeline/configs/geneval_full_v1.yaml"

GENEVAL_SCHEMA = "repldm.geneval_full_evaluation.v1"
PROMPT_MANIFEST_SCHEMA = "repldm.geneval_prompt_manifest.v1"
INPUT_ROW_SCHEMA = "repldm.geneval_input_row.v1"
SCORE_ROW_SCHEMA = "repldm.geneval_score_row.v1"
SUMMARY_SCHEMA = "repldm.geneval_summary.v1"
LAYOUT_SCHEMA = "repldm.geneval_layout.v1"
SEED_COHORT_SCHEMA = "repldm.geneval_seed_cohort.v1"
DEFAULT_SEED_COHORT_ID = "geneval_shared_v1"
DEFAULT_SAMPLE_SEEDS = (2026090301, 2026090302, 2026090303, 2026090304)
DEFAULT_SEED_COHORT_SHA256 = (
    "d6250b326e44e2e11b97a77c1fd8d1c4867486735007fedb02fb8beaf18f947b"
)

# A formal run may select only a cohort that has been registered in a reviewed
# repository revision.  Registering a new cohort requires a new ID and commit;
# a method cannot introduce a private seed list through its CLI invocation.
REGISTERED_SEED_COHORTS = {
    DEFAULT_SEED_COHORT_ID: {
        "schema": SEED_COHORT_SCHEMA,
        "id": DEFAULT_SEED_COHORT_ID,
        "seeds": list(DEFAULT_SAMPLE_SEEDS),
        "sha256": DEFAULT_SEED_COHORT_SHA256,
    }
}

EXPECTED_PROMPT_COUNT = 553
EXPECTED_SAMPLES_PER_PROMPT = 4
EXPECTED_IMAGE_COUNT = EXPECTED_PROMPT_COUNT * EXPECTED_SAMPLES_PER_PROMPT
EXPECTED_TAG_COUNTS = {
    "single_object": 80,
    "two_object": 99,
    "counting": 80,
    "colors": 94,
    "position": 100,
    "color_attr": 100,
}
DEFAULT_BOOTSTRAP_SEED = 20260901
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYOUT_NAME_RE = re.compile(r"(?:^|/)(\d{5})/samples/(\d{4,5})\.png$")
DEFAULT_INPUT_MANIFEST = Path("geneval/input_manifest.jsonl")
DEFAULT_LAYOUT_DIR = Path("geneval/layout")
DEFAULT_RAW_RESULTS = Path("geneval/raw_results.jsonl")
DEFAULT_SCORES = Path("geneval/scores.jsonl")
DEFAULT_SUMMARY = Path("geneval/summary.json")
EVALUATOR_SCHEMA = "repldm.geneval_evaluator.v1"

# The formal CLI is intentionally tied to the reviewed Sana evaluator bundle.
# Programmatic callers may still use ``run_official_evaluator`` for isolated
# tests, but a published benchmark result must use these exact bytes.
GENEVAL_EVALUATOR_REGISTRY = {
    "official_v1": {
        # Pin the actual interpreter file, rather than /usr/bin/python3
        # (which is a symlink and is rejected by the provenance validator).
        "python_path": "/home/bycao/miniforge3/envs/repldm_eval/bin/python3.11",
        "python_sha256": "7532e040c16c142e1fef0fd8f2f3ef7ff994607efceebc070b281d40c7087c1a",
        "script_path": "/mnt/miah204/bycao/Sana/tools/metrics/geneval/evaluation/evaluate_images.py",
        "script_sha256": "0571f83138a940bb867c7d0bad07da2bc41626741d8ce8f9f8ab9be4e7225ad1",
        "model_path": "/mnt/miah204/bycao/Sana/output/pretrained_models/geneval",
        "model_tree_sha256": "202d76f03f94b93ee0fd4797927084fadd31bbff7cba1150a1fdb7a28e28a7e8",
    }
}
DEFAULT_EVALUATOR_ID = "official_v1"


def canonical_json(value: Any) -> bytes:
    """Serialize JSON values without platform-dependent whitespace."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("GenEval payload contains non-finite or non-JSON values") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sample_seeds(value: Any, *, label: str = "sample_seeds") -> tuple[int, ...]:
    """Validate the ordered four-seed sample axis used by formal GenEval."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must contain exactly four distinct integers")
    seeds = tuple(value)
    if (
        len(seeds) != EXPECTED_SAMPLES_PER_PROMPT
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(f"{label} must contain exactly four distinct non-negative integers")
    return seeds


def seed_cohort_sha256(cohort_id: str, seeds: Sequence[int]) -> str:
    """Return the stable digest for one shared, ordered seed cohort."""
    if not isinstance(cohort_id, str) or not cohort_id.strip():
        raise ValueError("seed cohort id must be non-empty")
    normalized_id = cohort_id.strip()
    normalized_seeds = _validate_sample_seeds(seeds, label="seed cohort seeds")
    payload = {
        "schema": SEED_COHORT_SCHEMA,
        "cohort_id": normalized_id,
        "seeds": list(normalized_seeds),
    }
    return sha256_bytes(canonical_json(payload))


def _normalize_seed_cohort(
    value: Mapping[str, Any], *, expected_seeds: Sequence[int] | None = None
) -> dict[str, Any]:
    """Validate and canonicalize a registered seed cohort descriptor."""
    if not isinstance(value, Mapping):
        raise ValueError("seed_cohort must be a mapping")
    if set(value) != {"schema", "id", "seeds", "sha256"}:
        raise ValueError("seed_cohort must contain exactly schema, id, seeds, and sha256")
    if value.get("schema") != SEED_COHORT_SCHEMA:
        raise ValueError("seed_cohort schema is invalid")
    cohort_id = value.get("id")
    if not isinstance(cohort_id, str) or not cohort_id.strip():
        raise ValueError("seed_cohort.id must be a non-empty string")
    cohort_id = cohort_id.strip()
    seeds = _validate_sample_seeds(value.get("seeds"), label="seed_cohort.seeds")
    if expected_seeds is not None and seeds != _validate_sample_seeds(
        expected_seeds, label="expected seed cohort"
    ):
        raise ValueError("seed cohort seeds differ from the expected shared cohort")
    digest = value.get("sha256")
    _require_hash(digest, label="seed_cohort.sha256")
    expected_digest = seed_cohort_sha256(cohort_id, seeds)
    if digest != expected_digest:
        raise ValueError("seed_cohort.sha256 does not match its id and seeds")
    return {
        "schema": SEED_COHORT_SCHEMA,
        "id": cohort_id,
        "seeds": list(seeds),
        "sha256": digest,
    }


def validate_shared_seed_cohort(
    cohorts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require every formal comparison arm to use one registered cohort."""
    normalized = [_registered_seed_cohort(value) for value in cohorts]
    if not normalized:
        raise ValueError("at least one seed cohort is required")
    first = normalized[0]
    if any(canonical_json(value) != canonical_json(first) for value in normalized[1:]):
        raise ValueError("compared GenEval methods use different seed cohorts")
    return first


def _registered_seed_cohort(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one cohort against the reviewed formal registry."""
    normalized = _normalize_seed_cohort(value)
    registered = REGISTERED_SEED_COHORTS.get(normalized["id"])
    if registered is None:
        raise ValueError(
            "seed cohort is not registered for formal GenEval; add a reviewed cohort version"
        )
    if canonical_json(normalized) != canonical_json(registered):
        raise ValueError("seed cohort differs from the reviewed registry bytes")
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash a JSONL sequence using the exact order supplied by the caller."""
    payload = b"".join(canonical_json(dict(row)) + b"\n" for row in rows)
    return sha256_bytes(payload)


def _require_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_plain_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{label} must be an integer{suffix}")
    return value


def _json_file(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _jsonl_file_sha256(path: Path, *, label: str) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    return sha256_file(path)


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    result: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{label} line {line_number} is not an object")
                result.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return result


def _normalise_metadata(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    required = {"tag", "include", "prompt"}
    if not required.issubset(row):
        raise ValueError(f"GenEval metadata row {index} lacks {sorted(required - set(row))}")
    tag = row.get("tag")
    prompt = row.get("prompt")
    include = row.get("include")
    exclude = row.get("exclude", [])
    if not isinstance(tag, str) or not tag:
        raise ValueError(f"GenEval metadata row {index} has an invalid tag")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"GenEval metadata row {index} has an invalid prompt")
    if not isinstance(include, list) or not isinstance(exclude, list):
        raise ValueError(f"GenEval metadata row {index} include/exclude must be lists")
    # Keep the upstream schema and key names, but make the optional exclude
    # field explicit so duplicate rows compare byte-for-byte after parsing.
    normalised = dict(row)
    normalised["include"] = include
    normalised["exclude"] = exclude
    normalised["prompt"] = prompt
    normalised["tag"] = tag
    return normalised


def load_prompt_manifest(
    path: Path = DEFAULT_METADATA_PATH,
    *,
    expected_sha256: str | None = DEFAULT_METADATA_SHA256,
    expected_prompt_count: int = EXPECTED_PROMPT_COUNT,
    expected_samples_per_prompt: int = EXPECTED_SAMPLES_PER_PROMPT,
    strict_official: bool = True,
) -> dict[str, Any]:
    """Load and validate the official four-block GenEval metadata file.

    The official file contains 2,212 rows, with each of its 553 prompt
    specifications repeated four times.  The returned manifest collapses only
    this known sample repetition while retaining the source row indices.
    """
    _require_plain_int(expected_prompt_count, label="expected_prompt_count", minimum=1)
    _require_plain_int(
        expected_samples_per_prompt,
        label="expected_samples_per_prompt",
        minimum=1,
    )
    if expected_sha256 is not None:
        _require_hash(expected_sha256, label="expected_sha256")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"GenEval metadata source is missing or unsafe: {path}")
    source_sha256 = sha256_file(path)
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise ValueError(
            "GenEval metadata source hash differs from the frozen official source"
        )
    raw_rows = _read_jsonl(path, label="GenEval metadata")
    expected_rows = expected_prompt_count * expected_samples_per_prompt
    if len(raw_rows) != expected_rows:
        raise ValueError(
            f"GenEval metadata must contain {expected_rows} rows, got {len(raw_rows)}"
        )

    groups: "OrderedDict[str, list[tuple[int, dict[str, Any]]]]" = OrderedDict()
    for index, raw in enumerate(raw_rows):
        row = _normalise_metadata(raw, index=index)
        groups.setdefault(row["prompt"], []).append((index, row))
    if len(groups) != expected_prompt_count:
        raise ValueError(
            f"GenEval metadata must contain {expected_prompt_count} unique prompts, "
            f"got {len(groups)}"
        )

    prompts: list[dict[str, Any]] = []
    for prompt_index, (prompt, members) in enumerate(groups.items()):
        if len(members) != expected_samples_per_prompt:
            raise ValueError(
                f"GenEval prompt {prompt_index} must have exactly "
                f"{expected_samples_per_prompt} rows"
            )
        reference = members[0][1]
        if any(member[1] != reference for member in members[1:]):
            raise ValueError(f"GenEval duplicate metadata differs for prompt {prompt_index}")
        indices = [member[0] for member in members]
        if strict_official:
            expected_indices = [
                prompt_index + block * expected_prompt_count
                for block in range(expected_samples_per_prompt)
            ]
            if indices != expected_indices:
                raise ValueError(
                    f"GenEval source row order differs for prompt {prompt_index}"
                )
        prompts.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "tag": reference["tag"],
                "metadata": reference,
                "source_row_indices": indices,
            }
        )

    tag_counts = dict(Counter(prompt["tag"] for prompt in prompts))
    if strict_official and tag_counts != EXPECTED_TAG_COUNTS:
        raise ValueError(
            f"GenEval task counts differ from the official contract: {tag_counts}"
        )
    manifest_core = {
        "schema": PROMPT_MANIFEST_SCHEMA,
        "source_path": str(path.resolve()),
        "source_sha256": source_sha256,
        "source_rows": len(raw_rows),
        "prompt_count": expected_prompt_count,
        "samples_per_prompt": expected_samples_per_prompt,
        "tag_counts": tag_counts,
        "prompts": prompts,
    }
    return {
        **manifest_core,
        "manifest_sha256": sha256_bytes(canonical_json(manifest_core)),
    }


def _validate_prompt_manifest(
    prompt_manifest: Mapping[str, Any], *, strict_official: bool = True
) -> dict[int, Mapping[str, Any]]:
    """Validate the collapsed manifest before it is used by any later stage.

    Loading the official metadata once is not enough: callers may deserialize a
    JSON manifest from disk between stages.  Rechecking the content hash and the
    complete prompt/tag contract prevents a hand-edited manifest from silently
    changing the benchmark while retaining the same schema string.
    """
    if not isinstance(prompt_manifest, Mapping):
        raise ValueError("GenEval prompt manifest must be an object")
    if prompt_manifest.get("schema") != PROMPT_MANIFEST_SCHEMA:
        raise ValueError("GenEval prompt manifest schema differs from the registered contract")
    for field in ("source_sha256", "manifest_sha256"):
        _require_hash(prompt_manifest.get(field), label=f"prompt_manifest.{field}")
    core = {
        key: value for key, value in prompt_manifest.items() if key != "manifest_sha256"
    }
    if sha256_bytes(canonical_json(core)) != prompt_manifest["manifest_sha256"]:
        raise ValueError("GenEval prompt manifest hash does not match its payload")
    prompt_count = _require_plain_int(
        prompt_manifest.get("prompt_count"), label="prompt_manifest.prompt_count", minimum=1
    )
    samples_per_prompt = _require_plain_int(
        prompt_manifest.get("samples_per_prompt"),
        label="prompt_manifest.samples_per_prompt",
        minimum=1,
    )
    source_rows = _require_plain_int(
        prompt_manifest.get("source_rows"), label="prompt_manifest.source_rows", minimum=1
    )
    if source_rows != prompt_count * samples_per_prompt:
        raise ValueError("GenEval prompt manifest source row count is inconsistent")
    prompts = prompt_manifest.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != prompt_count:
        raise ValueError("GenEval prompt manifest prompt count is inconsistent")
    result: dict[int, Mapping[str, Any]] = {}
    observed_tags: Counter[str] = Counter()
    for row_number, row in enumerate(prompts):
        if not isinstance(row, Mapping):
            raise ValueError(f"GenEval prompt manifest row {row_number} is not an object")
        index = row.get("prompt_index")
        if type(index) is not int or index in result:
            raise ValueError("GenEval prompt indices are invalid or duplicated")
        prompt = row.get("prompt")
        tag = row.get("tag")
        metadata = row.get("metadata")
        source_indices = row.get("source_row_indices")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"GenEval prompt manifest row {row_number} has an invalid prompt")
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"GenEval prompt manifest row {row_number} has an invalid tag")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"GenEval prompt manifest row {row_number} lacks metadata")
        if metadata.get("prompt") != prompt or metadata.get("tag") != tag:
            raise ValueError(f"GenEval prompt manifest row {row_number} metadata disagrees")
        if (
            not isinstance(source_indices, list)
            or len(source_indices) != samples_per_prompt
            or any(type(value) is not int or value < 0 for value in source_indices)
            or len(set(source_indices)) != len(source_indices)
        ):
            raise ValueError(f"GenEval prompt manifest row {row_number} source indices are invalid")
        if strict_official:
            expected_indices = [
                index + block * prompt_count for block in range(samples_per_prompt)
            ]
            if source_indices != expected_indices:
                raise ValueError("GenEval prompt manifest source row order is invalid")
        result[index] = row
        observed_tags[tag] += 1
    if set(result) != set(range(prompt_count)):
        raise ValueError("GenEval prompt indices are not a complete contiguous range")
    all_source_indices = [
        source_index
        for row in result.values()
        for source_index in row["source_row_indices"]
    ]
    if set(all_source_indices) != set(range(source_rows)):
        raise ValueError("GenEval prompt manifest source rows are incomplete or duplicated")
    declared_tags = prompt_manifest.get("tag_counts")
    if declared_tags != dict(observed_tags):
        raise ValueError("GenEval prompt manifest tag counts are inconsistent")
    if strict_official:
        if prompt_count != EXPECTED_PROMPT_COUNT or samples_per_prompt != EXPECTED_SAMPLES_PER_PROMPT:
            raise ValueError("the formal GenEval manifest requires 553 prompts and four samples")
        if dict(observed_tags) != EXPECTED_TAG_COUNTS:
            raise ValueError("GenEval prompt manifest task counts differ from the official contract")
    return result


def _safe_relative_path(run_dir: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside the run directory")
    root = run_dir.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the run directory") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink: {current}")
    return candidate


def _validate_png(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    with path.open("rb") as handle:
        signature = handle.read(8)
    if signature != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{label} is not a PNG: {path}")
    return sha256_file(path)


def _prompt_by_index(prompt_manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return _validate_prompt_manifest(prompt_manifest, strict_official=True)


def build_input_manifest(
    records: Iterable[Mapping[str, Any]],
    prompt_manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    checkpoint_id: str,
    checkpoint_sha256: str,
    method: str,
    run_contract_sha256: str,
    sample_seeds: Sequence[int],
    seed_cohort: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate generation records and return the canonical 2,212-row manifest."""
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("checkpoint_id must be a non-empty string")
    _require_hash(checkpoint_sha256, label="checkpoint_sha256")
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a non-empty string")
    _require_hash(run_contract_sha256, label="run_contract_sha256")
    seeds = _validate_sample_seeds(sample_seeds)
    if seed_cohort is None:
        # Keep programmatic smoke callers usable, but label their cohort as
        # unregistered so it cannot be mistaken for a formal shared cohort.
        cohort = {
            "schema": SEED_COHORT_SCHEMA,
            "id": "unregistered",
            "seeds": list(seeds),
            "sha256": seed_cohort_sha256("unregistered", seeds),
        }
    else:
        cohort = _registered_seed_cohort(
            _normalize_seed_cohort(seed_cohort, expected_seeds=seeds)
        )
    prompts = _prompt_by_index(prompt_manifest)
    if len(prompts) != EXPECTED_PROMPT_COUNT:
        raise ValueError("the formal GenEval input requires 553 prompts")

    observed: dict[tuple[int, int], dict[str, Any]] = {}
    paths: set[str] = set()
    for source_index, source in enumerate(records):
        if not isinstance(source, Mapping):
            raise ValueError(f"generation record {source_index} is not an object")
        prompt_index = source.get("prompt_index", source.get("benchmark_prompt_index"))
        if type(prompt_index) is not int or prompt_index not in prompts:
            raise ValueError(f"generation record {source_index} has an invalid prompt_index")
        sample_index = source.get("sample_index")
        if sample_index is None:
            seed = source.get("seed")
            sample_index = seeds.index(seed) if seed in seeds else None
        if type(sample_index) is not int or sample_index not in range(len(seeds)):
            raise ValueError(f"generation record {source_index} has an invalid sample_index")
        seed = source.get("seed", seeds[sample_index])
        if type(seed) is not int or seed != seeds[sample_index]:
            raise ValueError(f"generation record {source_index} seed differs from the frozen list")
        key = (prompt_index, sample_index)
        if key in observed:
            raise ValueError(f"duplicate GenEval input cell: {key}")

        expected_prompt = prompts[prompt_index]
        if source.get("prompt") != expected_prompt.get("prompt"):
            raise ValueError(f"GenEval prompt text differs at index {prompt_index}")
        if source.get("tag", expected_prompt.get("tag")) != expected_prompt.get("tag"):
            raise ValueError(f"GenEval tag differs at index {prompt_index}")
        if source.get("checkpoint_id", checkpoint_id) != checkpoint_id:
            raise ValueError("generation records contain mixed checkpoints")
        if source.get("checkpoint_sha256", checkpoint_sha256) != checkpoint_sha256:
            raise ValueError("generation records contain a stale checkpoint hash")
        if source.get("method", method) != method:
            raise ValueError("generation records contain mixed methods")
        if source.get("run_contract_sha256", run_contract_sha256) != run_contract_sha256:
            raise ValueError("generation records contain mixed run contracts")

        image_path = _safe_relative_path(run_dir, source.get("image_path"), label="image_path")
        image_hash = _validate_png(image_path, label=f"GenEval image {key}")
        declared_hash = source.get("image_sha256", image_hash)
        _require_hash(declared_hash, label=f"image_sha256 for {key}")
        if declared_hash != image_hash:
            raise ValueError(f"GenEval image hash differs at {key}")
        path_key = str(image_path)
        if path_key in paths:
            raise ValueError("one image path is assigned to multiple GenEval cells")
        paths.add(path_key)
        row = {
            "schema": INPUT_ROW_SCHEMA,
            "benchmark": "GenEval",
            "checkpoint_id": checkpoint_id,
            "checkpoint_sha256": checkpoint_sha256,
            "method": method,
            "run_contract_sha256": run_contract_sha256,
            "prompt_index": prompt_index,
            "sample_index": sample_index,
            "seed": seed,
            "prompt": expected_prompt["prompt"],
            "tag": expected_prompt["tag"],
            "image_path": str(image_path.relative_to(run_dir.resolve())),
            "image_sha256": image_hash,
            "source_record_id": source.get("id"),
            "optimization_seed": source.get("optimization_seed"),
        }
        row["seed_cohort_id"] = cohort["id"]
        row["seed_cohort_sha256"] = cohort["sha256"]
        observed[key] = row

    expected_cells = {
        (prompt_index, sample_index)
        for prompt_index in prompts
        for sample_index in range(len(seeds))
    }
    if set(observed) != expected_cells:
        missing = len(expected_cells - set(observed))
        extra = len(set(observed) - expected_cells)
        raise ValueError(f"GenEval input is incomplete: {missing} missing, {extra} extra")
    by_prompt: dict[int, list[dict[str, Any]]] = {}
    for (prompt_index, _sample_index), row in observed.items():
        by_prompt.setdefault(prompt_index, []).append(row)
    for prompt_index, rows in by_prompt.items():
        if len({row["image_sha256"] for row in rows}) != len(rows):
            raise ValueError(
                f"GenEval prompt {prompt_index} has duplicate images across sample seeds"
            )
    return [observed[key] for key in sorted(observed)]


def _validate_input_rows(
    rows: Sequence[Mapping[str, Any]],
    prompt_manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    expected_seed_cohort: Mapping[str, Any] | None = None,
) -> dict[tuple[int, int], Mapping[str, Any]]:
    """Revalidate a persisted input manifest at every downstream boundary."""
    prompts = _prompt_by_index(prompt_manifest)
    expected_cohort = (
        _normalize_seed_cohort(expected_seed_cohort)
        if expected_seed_cohort is not None
        else None
    )
    if len(rows) != EXPECTED_IMAGE_COUNT:
        raise ValueError("formal GenEval input manifest must contain exactly 2,212 rows")
    expected_cells = {
        (prompt_index, sample_index)
        for prompt_index in prompts
        for sample_index in range(EXPECTED_SAMPLES_PER_PROMPT)
    }
    observed: dict[tuple[int, int], Mapping[str, Any]] = {}
    seed_by_sample: dict[int, int] = {}
    cohort_ids: set[str] = set()
    cohort_hashes: set[str] = set()
    cohort_presence: set[bool] = set()
    identity_fields = (
        "checkpoint_id",
        "checkpoint_sha256",
        "method",
        "run_contract_sha256",
    )
    for row_number, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("schema") != INPUT_ROW_SCHEMA:
            raise ValueError(f"GenEval input row {row_number} has an invalid schema")
        prompt_index = row.get("prompt_index")
        sample_index = row.get("sample_index")
        cell = (prompt_index, sample_index)
        if (
            type(prompt_index) is not int
            or type(sample_index) is not int
            or cell not in expected_cells
            or cell in observed
        ):
            raise ValueError(f"GenEval input row {row_number} has a duplicate or invalid cell")
        prompt_spec = prompts[prompt_index]
        if row.get("prompt") != prompt_spec["prompt"] or row.get("tag") != prompt_spec["tag"]:
            raise ValueError(f"GenEval input row {row_number} prompt metadata differs")
        _require_nonempty_string(row.get("checkpoint_id"), label="checkpoint_id")
        _require_hash(row.get("checkpoint_sha256"), label="checkpoint_sha256")
        _require_nonempty_string(row.get("method"), label="method")
        _require_hash(row.get("run_contract_sha256"), label="run_contract_sha256")
        if type(row.get("seed")) is not int:
            raise ValueError(f"GenEval input row {row_number} seed is invalid")
        cohort_id = row.get("seed_cohort_id")
        cohort_hash = row.get("seed_cohort_sha256")
        cohort_presence.add(cohort_id is not None)
        if (cohort_id is None) != (cohort_hash is None):
            raise ValueError(
                f"GenEval input row {row_number} has an incomplete seed cohort binding"
            )
        if cohort_id is not None:
            if not isinstance(cohort_id, str) or not cohort_id.strip():
                raise ValueError(
                    f"GenEval input row {row_number} seed_cohort_id is invalid"
                )
            _require_hash(
                cohort_hash,
                label=f"seed_cohort_sha256 for input row {row_number}",
            )
            cohort_ids.add(cohort_id)
            cohort_hashes.add(cohort_hash)
        elif expected_cohort is not None:
            raise ValueError(
                f"GenEval input row {row_number} lacks the expected seed cohort binding"
            )
        previous_seed = seed_by_sample.setdefault(sample_index, row["seed"])
        if previous_seed != row["seed"]:
            raise ValueError("GenEval input uses inconsistent seeds for one sample index")
        image_path = _safe_relative_path(run_dir, row.get("image_path"), label="image_path")
        actual_hash = _validate_png(image_path, label=f"GenEval input image {cell}")
        _require_hash(row.get("image_sha256"), label=f"image_sha256 for {cell}")
        if row["image_sha256"] != actual_hash:
            raise ValueError(f"GenEval input image hash differs at {cell}")
        observed[cell] = row
    if set(observed) != expected_cells:
        missing = len(expected_cells - set(observed))
        extra = len(set(observed) - expected_cells)
        raise ValueError(f"GenEval input is incomplete: {missing} missing, {extra} extra")
    if len(seed_by_sample) != EXPECTED_SAMPLES_PER_PROMPT or len(set(seed_by_sample.values())) != EXPECTED_SAMPLES_PER_PROMPT:
        raise ValueError("GenEval input must bind four distinct seeds to the sample indices")
    observed_seeds = tuple(seed_by_sample[index] for index in range(EXPECTED_SAMPLES_PER_PROMPT))
    _validate_sample_seeds(observed_seeds, label="GenEval input seeds")
    if len(cohort_ids) > 1 or len(cohort_hashes) > 1:
        raise ValueError("GenEval input contains mixed seed cohorts")
    if len(cohort_presence) > 1:
        raise ValueError("GenEval input contains incomplete seed cohort bindings")
    observed_cohort: dict[str, Any] | None = None
    if cohort_ids:
        observed_cohort = _normalize_seed_cohort(
            {
                "schema": SEED_COHORT_SCHEMA,
                "id": next(iter(cohort_ids)),
                "seeds": list(observed_seeds),
                "sha256": next(iter(cohort_hashes)),
            },
            expected_seeds=observed_seeds,
        )
    if expected_cohort is not None:
        if observed_cohort is None or canonical_json(observed_cohort) != canonical_json(expected_cohort):
            raise ValueError("GenEval input seed cohort differs from the registered shared cohort")
    first = observed[min(observed)]
    for field in identity_fields:
        if any(row.get(field) != first.get(field) for row in observed.values()):
            raise ValueError(f"GenEval input contains mixed {field}")
    paths = [str(_safe_relative_path(run_dir, row["image_path"], label="image_path")) for row in observed.values()]
    if len(set(paths)) != len(paths):
        raise ValueError("GenEval input assigns one image path to multiple cells")
    return observed


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Atomically publish JSONL and return its content hash."""
    payload = b"".join(canonical_json(dict(row)) + b"\n" for row in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing to write through symlink: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_bytes(payload)


def write_input_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if len(rows) != EXPECTED_IMAGE_COUNT:
        raise ValueError("formal GenEval input manifest must contain exactly 2,212 rows")
    return write_jsonl(path, rows)


def _write_new_or_verify(path: Path, payload: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing {label} differs")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize_official_layout(
    rows: Sequence[Mapping[str, Any]],
    prompt_manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    layout_dir: Path,
    input_manifest_sha256: str,
    expected_seed_cohort: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the upstream ``00000/samples/0000.png`` layout by hardlinking."""
    _require_hash(input_manifest_sha256, label="input_manifest_sha256")
    run_dir = Path(run_dir).resolve()
    layout_dir = Path(layout_dir)
    if not layout_dir.is_absolute():
        layout_dir = (run_dir / layout_dir).resolve()
    else:
        layout_dir = layout_dir.resolve()
    try:
        layout_dir.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("GenEval layout must be inside the run directory") from exc
    if layout_dir.is_symlink():
        raise ValueError("GenEval layout cannot be a symlink")
    layout_dir.mkdir(parents=True, exist_ok=True)
    prompts = _prompt_by_index(prompt_manifest)
    by_cell = _validate_input_rows(
        rows,
        prompt_manifest,
        run_dir=run_dir,
        expected_seed_cohort=expected_seed_cohort,
    )
    unexpected_directories = [
        child.name
        for child in layout_dir.iterdir()
        if child.is_dir() and child.name.isdigit() and child.name != ""
        and not (len(child.name) == 5 and 0 <= int(child.name) < EXPECTED_PROMPT_COUNT)
    ]
    if unexpected_directories:
        raise ValueError(
            "GenEval layout contains unexpected numeric prompt directories: "
            + ", ".join(sorted(unexpected_directories))
        )
    layout_rows: list[dict[str, Any]] = []
    for prompt_index in range(EXPECTED_PROMPT_COUNT):
        prompt_dir = layout_dir / f"{prompt_index:05d}"
        sample_dir = prompt_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        metadata_payload = canonical_json(dict(prompts[prompt_index]["metadata"]))
        _write_new_or_verify(
            prompt_dir / "metadata.jsonl",
            metadata_payload + b"\n",
            label=f"metadata for prompt {prompt_index}",
        )
        for sample_index in range(EXPECTED_SAMPLES_PER_PROMPT):
            row = by_cell[(prompt_index, sample_index)]
            source = _safe_relative_path(run_dir, row["image_path"], label="image_path")
            destination = sample_dir / f"{sample_index:04d}.png"
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() or _validate_png(destination, label="layout image") != row["image_sha256"]:
                    raise ValueError(f"existing GenEval layout image differs: {destination}")
            else:
                try:
                    os.link(source, destination, follow_symlinks=False)
                except OSError:
                    # Cross-device run/layout paths are allowed, but the copy
                    # remains atomic and is checked against the source hash.
                    descriptor, temporary = tempfile.mkstemp(
                        prefix=f".{destination.name}.", suffix=".tmp", dir=str(sample_dir)
                    )
                    try:
                        with source.open("rb") as src, os.fdopen(descriptor, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                            dst.flush()
                            os.fsync(dst.fileno())
                        os.link(temporary, destination, follow_symlinks=False)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
                if _validate_png(destination, label="materialized GenEval image") != row["image_sha256"]:
                    raise ValueError(f"materialized GenEval image hash differs: {destination}")
            layout_rows.append(
                {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "source_image_path": row["image_path"],
                    "layout_image_path": str(destination.relative_to(layout_dir)),
                    "image_sha256": row["image_sha256"],
                }
            )
    descriptor = {
        "schema": LAYOUT_SCHEMA,
        "benchmark": "GenEval",
        "prompt_manifest_sha256": prompt_manifest["manifest_sha256"],
        "input_manifest_sha256": input_manifest_sha256,
        "prompt_count": EXPECTED_PROMPT_COUNT,
        "samples_per_prompt": EXPECTED_SAMPLES_PER_PROMPT,
        "image_count": EXPECTED_IMAGE_COUNT,
        "rows": layout_rows,
    }
    layout_cohort = (
        _normalize_seed_cohort(expected_seed_cohort)
        if expected_seed_cohort is not None
        else None
    )
    if layout_cohort is None:
        first_row = by_cell[(0, 0)]
        if "seed_cohort_id" in first_row and "seed_cohort_sha256" in first_row:
            observed_seeds = tuple(
                by_cell[(0, index)]["seed"]
                for index in range(EXPECTED_SAMPLES_PER_PROMPT)
            )
            layout_cohort = _normalize_seed_cohort(
                {
                    "schema": SEED_COHORT_SCHEMA,
                    "id": first_row["seed_cohort_id"],
                    "seeds": list(observed_seeds),
                    "sha256": first_row["seed_cohort_sha256"],
                },
                expected_seeds=observed_seeds,
            )
    if layout_cohort is not None:
        descriptor["seed_cohort"] = layout_cohort
    descriptor["layout_sha256"] = sha256_bytes(canonical_json(descriptor))
    _write_new_or_verify(
        layout_dir / "layout.json",
        canonical_json(descriptor) + b"\n",
        label="GenEval layout descriptor",
    )
    return descriptor


def _resolve_result_filename(filename: Any, layout_dir: Path) -> tuple[int, int] | None:
    if not isinstance(filename, str) or not filename:
        return None
    layout_root = Path(layout_dir).resolve()
    candidate = Path(filename)
    if not candidate.is_absolute():
        candidate = layout_root / candidate
    try:
        relative = candidate.resolve(strict=False).relative_to(layout_root)
    except ValueError:
        return None
    match = _LAYOUT_NAME_RE.fullmatch(relative.as_posix())
    if not match:
        return None
    prompt_index = int(match.group(1))
    sample_index = int(match.group(2))
    # The official layout uses exactly four sample digits.  Reject aliases
    # such as ``00000.png`` even though they could be mapped to the same cell.
    if relative.as_posix() != f"{prompt_index:05d}/samples/{sample_index:04d}.png":
        return None
    return prompt_index, sample_index


def normalize_results(
    raw_rows: Iterable[Mapping[str, Any]],
    input_rows: Sequence[Mapping[str, Any]],
    prompt_manifest: Mapping[str, Any],
    *,
    layout_dir: Path,
    evaluator: Mapping[str, Any],
    run_dir: Path | None = None,
    expected_seed_cohort: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Bind upstream result rows to the canonical input cells."""
    if len(input_rows) != EXPECTED_IMAGE_COUNT:
        raise ValueError("GenEval input must be complete before result normalization")
    layout_dir = Path(layout_dir).resolve()
    input_root = (
        Path(run_dir).resolve()
        if run_dir is not None
        else layout_dir.parent
    )
    expected = _validate_input_rows(
        input_rows,
        prompt_manifest,
        run_dir=input_root,
        expected_seed_cohort=expected_seed_cohort,
    )
    output: dict[tuple[int, int], dict[str, Any]] = {}
    prompts = _prompt_by_index(prompt_manifest)
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"GenEval evaluator row {index} is not an object")
        cell = _resolve_result_filename(raw.get("filename"), Path(layout_dir))
        if cell not in expected:
            raise ValueError(f"GenEval evaluator row {index} points to an unknown image")
        if cell in output:
            raise ValueError(f"duplicate GenEval evaluator result for {cell}")
        if type(raw.get("correct")) is not bool:
            raise ValueError(f"GenEval evaluator row {index} has a non-boolean correct field")
        source = expected[cell]
        prompt_spec = prompts[cell[0]]
        layout_image = layout_dir / f"{cell[0]:05d}" / "samples" / f"{cell[1]:04d}.png"
        if _validate_png(layout_image, label=f"GenEval layout image {cell}") != source["image_sha256"]:
            raise ValueError(f"GenEval layout image differs from input at {cell}")
        if raw.get("prompt") != prompt_spec["prompt"]:
            raise ValueError(f"GenEval evaluator prompt differs at {cell}")
        if raw.get("tag") != prompt_spec["tag"]:
            raise ValueError(f"GenEval evaluator tag differs at {cell}")
        metadata_value = raw.get("metadata")
        if isinstance(metadata_value, str):
            try:
                metadata_value = json.loads(metadata_value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"GenEval evaluator metadata is invalid at {cell}") from exc
        if metadata_value != prompt_spec["metadata"]:
            raise ValueError(f"GenEval evaluator metadata differs at {cell}")
        output[cell] = {
            "schema": SCORE_ROW_SCHEMA,
            "benchmark": "GenEval",
            "checkpoint_id": source["checkpoint_id"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "method": source["method"],
            "run_contract_sha256": source["run_contract_sha256"],
            "prompt_index": cell[0],
            "sample_index": cell[1],
            "seed": source["seed"],
            "prompt": source["prompt"],
            "tag": source["tag"],
            "image_path": source["image_path"],
            "image_sha256": source["image_sha256"],
            "correct": raw["correct"],
            "reason": raw.get("reason", ""),
            "details": raw.get("details", "{}"),
            "evaluator": dict(evaluator),
        }
        if "seed_cohort_id" in source:
            output[cell]["seed_cohort_id"] = source["seed_cohort_id"]
            output[cell]["seed_cohort_sha256"] = source["seed_cohort_sha256"]
    if set(output) != set(expected):
        missing = len(set(expected) - set(output))
        raise ValueError(f"GenEval evaluator output is incomplete: {missing} rows missing")
    return [output[key] for key in sorted(output)]


def _percentile_type7(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _bootstrap_scores(
    prompt_scores: Mapping[int, float],
    prompt_tags: Mapping[int, str],
    *,
    seed: int,
    resamples: int,
) -> dict[str, dict[str, float]]:
    _require_plain_int(seed, label="bootstrap_seed")
    _require_plain_int(resamples, label="bootstrap_resamples", minimum=1)
    indices = sorted(prompt_scores)
    by_tag = {
        tag: [index for index in indices if prompt_tags[index] == tag]
        for tag in EXPECTED_TAG_COUNTS
    }
    if not indices or any(not values for values in by_tag.values()):
        raise ValueError("bootstrap requires every GenEval task to have prompts")
    distributions: dict[str, list[float]] = {
        "overall_score": [],
        "prompt_mean": [],
        **{f"task:{tag}": [] for tag in by_tag},
    }
    rng = random.Random(seed)
    for _ in range(resamples):
        sampled = [indices[rng.randrange(len(indices))] for _ in indices]
        task_values: list[float] = []
        for tag in by_tag:
            # Resample within each task stratum.  A global bootstrap can omit a
            # small task by chance and gives it a different weight on every
            # replicate, which is inconsistent with the registered macro mean.
            task_indices = by_tag[tag]
            sampled_task = [
                task_indices[rng.randrange(len(task_indices))]
                for _ in task_indices
            ]
            values = [prompt_scores[index] for index in sampled_task]
            value = sum(values) / len(values)
            distributions[f"task:{tag}"].append(value)
            task_values.append(value)
        distributions["overall_score"].append(sum(task_values) / len(task_values))
        distributions["prompt_mean"].append(sum(prompt_scores[index] for index in sampled) / len(sampled))
    return {
        key: {
            "lower": _percentile_type7(values, 0.025),
            "upper": _percentile_type7(values, 0.975),
        }
        for key, values in distributions.items()
    }


def aggregate_scores(
    rows: Sequence[Mapping[str, Any]],
    prompt_manifest: Mapping[str, Any],
    *,
    input_manifest_sha256: str,
    raw_results_sha256: str,
    config_sha256: str,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    expected_seed_cohort: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute official task-mean and prompt-cluster statistics."""
    _require_hash(input_manifest_sha256, label="input_manifest_sha256")
    _require_hash(raw_results_sha256, label="raw_results_sha256")
    _require_hash(config_sha256, label="config_sha256")
    if len(rows) != EXPECTED_IMAGE_COUNT:
        raise ValueError("GenEval aggregation requires exactly 2,212 score rows")
    prompts = _prompt_by_index(prompt_manifest)
    by_prompt: dict[int, list[Mapping[str, Any]]] = {}
    observed_cells: set[tuple[int, int]] = set()
    cohort_ids: set[str] = set()
    cohort_hashes: set[str] = set()
    cohort_presence: set[bool] = set()
    for row in rows:
        if row.get("schema") != SCORE_ROW_SCHEMA or type(row.get("correct")) is not bool:
            raise ValueError("GenEval score row schema or correct field is invalid")
        p = row.get("prompt_index")
        s = row.get("sample_index")
        if type(p) is not int or type(s) is not int or p not in prompts or s not in range(4):
            raise ValueError("GenEval score row has an invalid cell")
        if (p, s) in observed_cells:
            raise ValueError("GenEval score rows contain a duplicate cell")
        observed_cells.add((p, s))
        if row.get("prompt") != prompts[p]["prompt"] or row.get("tag") != prompts[p]["tag"]:
            raise ValueError("GenEval score row prompt metadata differs")
        _require_nonempty_string(row.get("checkpoint_id"), label="score.checkpoint_id")
        _require_hash(row.get("checkpoint_sha256"), label="score.checkpoint_sha256")
        _require_nonempty_string(row.get("method"), label="score.method")
        _require_hash(row.get("run_contract_sha256"), label="score.run_contract_sha256")
        _require_plain_int(row.get("seed"), label="score.seed")
        cohort_id = row.get("seed_cohort_id")
        cohort_hash = row.get("seed_cohort_sha256")
        cohort_presence.add(cohort_id is not None)
        if (cohort_id is None) != (cohort_hash is None):
            raise ValueError("GenEval score rows contain an incomplete seed cohort binding")
        if cohort_id is not None:
            if not isinstance(cohort_id, str) or not cohort_id.strip():
                raise ValueError("GenEval score row seed_cohort_id is invalid")
            _require_hash(cohort_hash, label="score.seed_cohort_sha256")
            cohort_ids.add(cohort_id)
            cohort_hashes.add(cohort_hash)
        elif expected_seed_cohort is not None:
            raise ValueError("GenEval score rows lack the expected seed cohort binding")
        _require_hash(row.get("image_sha256"), label=f"score.image_sha256 for {(p, s)}")
        by_prompt.setdefault(p, []).append(row)
    expected_cells = {
        (prompt_index, sample_index)
        for prompt_index in prompts
        for sample_index in range(EXPECTED_SAMPLES_PER_PROMPT)
    }
    if observed_cells != expected_cells:
        raise ValueError("GenEval score rows do not contain the complete cell grid")
    if set(by_prompt) != set(prompts) or any(
        len(value) != EXPECTED_SAMPLES_PER_PROMPT for value in by_prompt.values()
    ):
        raise ValueError("GenEval scores do not contain exactly four samples per prompt")
    seed_by_sample: dict[int, int] = {}
    for row in rows:
        sample_index = int(row["sample_index"])
        seed = int(row["seed"])
        previous = seed_by_sample.setdefault(sample_index, seed)
        if previous != seed:
            raise ValueError("GenEval score rows use inconsistent seeds for one sample index")
    observed_seeds = tuple(seed_by_sample[index] for index in range(EXPECTED_SAMPLES_PER_PROMPT))
    _validate_sample_seeds(observed_seeds, label="GenEval score seeds")
    if len(cohort_ids) > 1 or len(cohort_hashes) > 1:
        raise ValueError("GenEval score rows contain mixed seed cohorts")
    if len(cohort_presence) > 1:
        raise ValueError("GenEval score rows contain incomplete seed cohort bindings")
    observed_cohort: dict[str, Any] | None = None
    if cohort_ids:
        observed_cohort = _normalize_seed_cohort(
            {
                "schema": SEED_COHORT_SCHEMA,
                "id": next(iter(cohort_ids)),
                "seeds": list(observed_seeds),
                "sha256": next(iter(cohort_hashes)),
            },
            expected_seeds=observed_seeds,
        )
    if expected_seed_cohort is not None:
        normalized_expected = _normalize_seed_cohort(expected_seed_cohort)
        if observed_cohort is None or canonical_json(observed_cohort) != canonical_json(normalized_expected):
            raise ValueError("GenEval scores use a different seed cohort from the registered one")
    prompt_scores = {
        p: sum(1.0 if row["correct"] else 0.0 for row in rows_for_prompt) / 4.0
        for p, rows_for_prompt in by_prompt.items()
    }
    prompt_tags = {p: str(prompts[p]["tag"]) for p in prompts}
    task_scores = {}
    task_counts = {}
    for tag in EXPECTED_TAG_COUNTS:
        values = [prompt_scores[p] for p in prompts if prompt_tags[p] == tag]
        task_scores[tag] = sum(values) / len(values)
        task_counts[tag] = len(values)
    if any(
        not isinstance(value, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in task_scores.values()
    ):
        raise ValueError("GenEval task scores are not finite probabilities")
    overall_score = sum(task_scores.values()) / len(task_scores)
    prompt_mean = sum(prompt_scores.values()) / len(prompt_scores)
    image_accuracy = sum(prompt_scores.values()) / EXPECTED_PROMPT_COUNT
    prompt_pass_rate = sum(value > 0.0 for value in prompt_scores.values()) / EXPECTED_PROMPT_COUNT
    bootstrap = _bootstrap_scores(
        prompt_scores,
        prompt_tags,
        seed=int(bootstrap_seed),
        resamples=int(bootstrap_resamples),
    )
    first = rows[0]
    identity_fields = ("checkpoint_id", "checkpoint_sha256", "method", "run_contract_sha256")
    for field in identity_fields:
        if field not in first:
            raise ValueError(f"GenEval score rows lack {field}")
        if any(row.get(field) != first[field] for row in rows):
            raise ValueError(f"GenEval score rows contain mixed {field}")
    evaluators = [row.get("evaluator") for row in rows]
    if any(value is not None and not isinstance(value, Mapping) for value in evaluators):
        raise ValueError("GenEval score rows contain an invalid evaluator descriptor")
    if any(value != evaluators[0] for value in evaluators):
        raise ValueError("GenEval score rows contain mixed evaluator descriptors")
    summary_core = {
        "schema": SUMMARY_SCHEMA,
        "benchmark": "GenEval",
        "status": "complete",
        "subset_evaluation_allowed": False,
        "checkpoint_id": first["checkpoint_id"],
        "checkpoint_sha256": first["checkpoint_sha256"],
        "method": first["method"],
        "run_contract_sha256": first["run_contract_sha256"],
        "prompt_manifest_sha256": prompt_manifest["manifest_sha256"],
        "input_manifest_sha256": input_manifest_sha256,
        "raw_results_sha256": raw_results_sha256,
        "config_sha256": config_sha256,
        "counts": {
            "prompts": EXPECTED_PROMPT_COUNT,
            "samples_per_prompt": EXPECTED_SAMPLES_PER_PROMPT,
            "images": EXPECTED_IMAGE_COUNT,
            "correct_images": int(sum(1 for row in rows if row["correct"])),
        },
        "image_accuracy": image_accuracy,
        "prompt_mean": prompt_mean,
        "prompt_pass_rate": prompt_pass_rate,
        "task_scores": task_scores,
        "task_prompt_counts": task_counts,
        "overall_score": overall_score,
        "score_definition": "unweighted_mean_of_six_task_image_accuracies",
        "bootstrap": {
            "unit": "prompt",
            "seed": int(bootstrap_seed),
            "resamples": int(bootstrap_resamples),
            "interval": [0.025, 0.975],
            "method": "prompt_cluster_percentile_type7_v1",
            "confidence_intervals": bootstrap,
        },
    }
    if observed_cohort is not None:
        summary_core["seed_cohort"] = observed_cohort
    summary = {
        **summary_core,
        "summary_sha256": sha256_bytes(canonical_json(summary_core)),
    }
    return summary


def validate_summary(
    summary: Mapping[str, Any],
    *,
    require_sealed: bool = False,
    expected_seed_cohort: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a previously published summary before treating it as complete."""
    if not isinstance(summary, Mapping) or summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("GenEval summary schema is invalid")
    if (
        summary.get("benchmark") != "GenEval"
        or summary.get("status") != "complete"
        or summary.get("subset_evaluation_allowed") is not False
    ):
        raise ValueError("GenEval summary is not a complete non-subset result")
    _require_nonempty_string(summary.get("checkpoint_id"), label="summary.checkpoint_id")
    _require_nonempty_string(summary.get("method"), label="summary.method")
    for field in (
        "checkpoint_sha256",
        "run_contract_sha256",
        "prompt_manifest_sha256",
        "input_manifest_sha256",
        "raw_results_sha256",
        "config_sha256",
        "summary_sha256",
    ):
        _require_hash(summary.get(field), label=f"summary.{field}")
    summary_cohort_value = summary.get("seed_cohort")
    if summary_cohort_value is None:
        if require_sealed or expected_seed_cohort is not None:
            raise ValueError("GenEval summary lacks its shared seed cohort binding")
        summary_cohort = None
    else:
        summary_cohort = _normalize_seed_cohort(summary_cohort_value)
        if require_sealed:
            summary_cohort = _registered_seed_cohort(summary_cohort)
    if expected_seed_cohort is not None:
        expected_cohort = _normalize_seed_cohort(expected_seed_cohort)
        if summary_cohort is None or canonical_json(summary_cohort) != canonical_json(expected_cohort):
            raise ValueError("GenEval summary seed cohort differs from the registered shared cohort")
    counts = summary.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("GenEval summary counts are incomplete")
    if (
        counts.get("prompts") != EXPECTED_PROMPT_COUNT
        or counts.get("samples_per_prompt") != EXPECTED_SAMPLES_PER_PROMPT
        or counts.get("images") != EXPECTED_IMAGE_COUNT
        or type(counts.get("correct_images")) is not int
        or not 0 <= counts["correct_images"] <= EXPECTED_IMAGE_COUNT
    ):
        raise ValueError("GenEval summary counts are invalid")
    for field in ("image_accuracy", "prompt_mean", "prompt_pass_rate", "overall_score"):
        value = summary.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"GenEval summary.{field} is not finite")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"GenEval summary.{field} is outside [0, 1]")
    if summary.get("score_definition") != "unweighted_mean_of_six_task_image_accuracies":
        raise ValueError("GenEval summary score definition is invalid")
    task_scores = summary.get("task_scores")
    task_counts = summary.get("task_prompt_counts")
    if not isinstance(task_scores, Mapping) or set(task_scores) != set(EXPECTED_TAG_COUNTS):
        raise ValueError("GenEval summary task scores are incomplete")
    if not isinstance(task_counts, Mapping) or task_counts != EXPECTED_TAG_COUNTS:
        raise ValueError("GenEval summary task counts are invalid")
    for tag, value in task_scores.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"GenEval task score is invalid: {tag}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"GenEval task score is outside [0, 1]: {tag}")
    bootstrap = summary.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("GenEval summary bootstrap contract is missing")
    if (
        bootstrap.get("unit") != "prompt"
        or bootstrap.get("method") != "prompt_cluster_percentile_type7_v1"
        or bootstrap.get("interval") != [0.025, 0.975]
    ):
        raise ValueError("GenEval summary bootstrap contract is invalid")
    _require_plain_int(bootstrap.get("seed"), label="summary.bootstrap.seed")
    _require_plain_int(bootstrap.get("resamples"), label="summary.bootstrap.resamples", minimum=1)
    intervals = bootstrap.get("confidence_intervals")
    if not isinstance(intervals, Mapping):
        raise ValueError("GenEval summary confidence intervals are missing")
    expected_interval_keys = {"overall_score", "prompt_mean"} | {
        f"task:{tag}" for tag in EXPECTED_TAG_COUNTS
    }
    if set(intervals) != expected_interval_keys:
        raise ValueError("GenEval summary confidence intervals are incomplete")
    for key, interval in intervals.items():
        if (
            not isinstance(interval, Mapping)
            or set(interval) != {"lower", "upper"}
            or any(
                not isinstance(interval[name], (int, float))
                or isinstance(interval[name], bool)
                or not math.isfinite(float(interval[name]))
                or not 0.0 <= float(interval[name]) <= 1.0
                for name in ("lower", "upper")
            )
            or interval["lower"] > interval["upper"]
        ):
            raise ValueError(f"GenEval confidence interval is invalid: {key}")
    optional_sealed = ("scores_sha256", "layout_sha256")
    present_sealed = [field for field in optional_sealed if field in summary]
    if require_sealed and len(present_sealed) != len(optional_sealed):
        raise ValueError("GenEval summary is not sealed to scores and layout")
    for field in present_sealed:
        _require_hash(summary.get(field), label=f"summary.{field}")
    if "evaluator" in summary:
        _validate_evaluator_descriptor(summary["evaluator"])
    elif require_sealed:
        raise ValueError("GenEval summary lacks evaluator provenance")
    core = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if sha256_bytes(canonical_json(core)) != summary["summary_sha256"]:
        raise ValueError("GenEval summary hash does not match its payload")
    return dict(summary)


def validate_summary_files(
    summary: Mapping[str, Any],
    *,
    run_dir: Path,
    expected_seed_cohort: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a sealed summary and rehash the files it claims to bind."""
    validated = validate_summary(
        summary,
        require_sealed=True,
        expected_seed_cohort=expected_seed_cohort,
    )
    root = _run_dir(Path(run_dir))
    file_bindings = {
        "input_manifest_sha256": root / DEFAULT_INPUT_MANIFEST,
        "raw_results_sha256": root / DEFAULT_RAW_RESULTS,
        "scores_sha256": root / DEFAULT_SCORES,
    }
    for field, path in file_bindings.items():
        if sha256_file(path) != validated[field]:
            raise ValueError(f"GenEval {field} does not match the published file")
    _read_input_manifest(
        root / DEFAULT_INPUT_MANIFEST,
        prompt_manifest=load_prompt_manifest(),
        run_dir=root,
        expected_sha256=validated["input_manifest_sha256"],
        expected_seed_cohort=validated["seed_cohort"],
    )
    layout_path = root / DEFAULT_LAYOUT_DIR
    if _layout_hash(
        layout_path,
        expected_seed_cohort=validated["seed_cohort"],
    ) != validated["layout_sha256"]:
        raise ValueError("GenEval layout hash does not match the published summary")
    evaluator = _validate_evaluator_descriptor(validated["evaluator"])
    registry_id = evaluator.get("registry_id")
    if not isinstance(registry_id, str):
        raise ValueError("GenEval summary evaluator is not bound to the registry")
    registered = _registered_evaluator(registry_id)
    if canonical_json(evaluator) != canonical_json(registered):
        raise ValueError("GenEval summary evaluator differs from the registered binding")
    script = Path(evaluator["script_path"])
    python = Path(evaluator["python_path"])
    model = Path(evaluator["model_path"])
    if script.is_symlink() or not script.is_file() or sha256_file(script) != evaluator["script_sha256"]:
        raise ValueError("GenEval evaluator script changed after scoring")
    if python.is_symlink() or not python.is_file() or sha256_file(python) != evaluator["python_sha256"]:
        raise ValueError("GenEval evaluator Python changed after scoring")
    if _tree_sha256(model) != evaluator["model_tree_sha256"]:
        raise ValueError("GenEval evaluator model assets changed after scoring")
    if "model_config_path" in evaluator:
        config_path = Path(evaluator["model_config_path"])
        if config_path.is_symlink() or not config_path.is_file() or sha256_file(config_path) != evaluator["model_config_sha256"]:
            raise ValueError("GenEval evaluator model config changed after scoring")
    return validated


def _tree_sha256(path: Path) -> str:
    """Hash a local evaluator/model tree without following symlink files."""
    path = Path(path)
    if path.is_file() and not path.is_symlink():
        return sha256_file(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"evaluator asset path is not a regular file/tree: {path}")
    entries = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError(f"evaluator asset tree contains a symlink: {child}")
        if child.is_file():
            relative = child.relative_to(path).as_posix()
            entries.append((relative, child.stat().st_size, sha256_file(child)))
    return sha256_bytes(canonical_json(entries))


def _evaluator_descriptor(
    *,
    evaluator_python: Path,
    evaluator_script: Path,
    model_path: Path,
    model_config: Path | None,
    gpu_id: int,
) -> dict[str, Any]:
    evaluator_python = Path(evaluator_python)
    evaluator_script = Path(evaluator_script)
    model_path = Path(model_path)
    if evaluator_python.is_symlink() or not evaluator_python.is_file():
        raise ValueError("evaluator_python must be an ordinary executable file")
    if not os.access(evaluator_python, os.X_OK):
        raise ValueError("evaluator_python is not executable")
    if evaluator_script.is_symlink() or not evaluator_script.is_file():
        raise ValueError("evaluator_script must be an ordinary file")
    if model_config is not None:
        model_config = Path(model_config)
        if model_config.is_symlink() or not model_config.is_file():
            raise ValueError("model_config must be an ordinary file")
    _require_plain_int(gpu_id, label="gpu_id", minimum=0)
    descriptor: dict[str, Any] = {
        "schema": EVALUATOR_SCHEMA,
        "script_path": str(evaluator_script.resolve()),
        "script_sha256": sha256_file(evaluator_script),
        "python_path": str(evaluator_python.resolve()),
        "python_sha256": sha256_file(evaluator_python),
        "model_path": str(model_path.resolve()),
        "model_tree_sha256": _tree_sha256(model_path),
        "gpu_id": gpu_id,
    }
    if model_config is not None:
        descriptor["model_config_path"] = str(model_config.resolve())
        descriptor["model_config_sha256"] = sha256_file(model_config)
    return descriptor


def _registered_evaluator(evaluator_id: str = DEFAULT_EVALUATOR_ID) -> dict[str, Any]:
    """Resolve and rehash one immutable evaluator registry entry."""
    entry = GENEVAL_EVALUATOR_REGISTRY.get(evaluator_id)
    if not isinstance(entry, Mapping):
        raise ValueError(f"unknown GenEval evaluator registry id: {evaluator_id!r}")
    descriptor = _evaluator_descriptor(
        evaluator_python=Path(entry["python_path"]),
        evaluator_script=Path(entry["script_path"]),
        model_path=Path(entry["model_path"]),
        model_config=None,
        gpu_id=0,
    )
    for field in ("python_sha256", "script_sha256", "model_tree_sha256"):
        if descriptor[field] != entry[field]:
            raise ValueError(f"registered GenEval evaluator {field} differs from frozen bytes")
    descriptor["registry_id"] = evaluator_id
    return descriptor


def _validate_evaluator_descriptor(evaluator: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(evaluator, Mapping) or evaluator.get("schema") != EVALUATOR_SCHEMA:
        raise ValueError("GenEval evaluator descriptor schema is invalid")
    required = {
        "schema",
        "script_path",
        "script_sha256",
        "python_path",
        "python_sha256",
        "model_path",
        "model_tree_sha256",
        "gpu_id",
    }
    if not required.issubset(evaluator):
        raise ValueError("GenEval evaluator descriptor is incomplete")
    for field in ("script_sha256", "python_sha256", "model_tree_sha256"):
        _require_hash(evaluator.get(field), label=f"evaluator.{field}")
    if ("model_config_path" in evaluator) != ("model_config_sha256" in evaluator):
        raise ValueError("evaluator model_config binding is incomplete")
    if "model_config_sha256" in evaluator:
        _require_hash(evaluator.get("model_config_sha256"), label="evaluator.model_config_sha256")
        _require_nonempty_string(evaluator.get("model_config_path"), label="evaluator.model_config_path")
    for field in ("script_path", "python_path", "model_path"):
        _require_nonempty_string(evaluator.get(field), label=f"evaluator.{field}")
    _require_plain_int(evaluator.get("gpu_id"), label="evaluator.gpu_id", minimum=0)
    return dict(evaluator)


def run_official_evaluator(
    *,
    input_rows: Sequence[Mapping[str, Any]],
    prompt_manifest: Mapping[str, Any],
    run_dir: Path,
    layout_dir: Path,
    input_manifest_sha256: str,
    config_sha256: str,
    evaluator_python: Path,
    evaluator_script: Path,
    model_path: Path,
    model_config: Path | None = None,
    gpu_id: int = 0,
    timeout_seconds: float = 86_400.0,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    expected_seed_cohort: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the reviewed upstream evaluator and publish normalized results."""
    layout = materialize_official_layout(
        input_rows,
        prompt_manifest,
        run_dir=run_dir,
        layout_dir=layout_dir,
        input_manifest_sha256=input_manifest_sha256,
        expected_seed_cohort=expected_seed_cohort,
    )
    evaluator_python = Path(evaluator_python)
    evaluator_script = Path(evaluator_script)
    model_path = Path(model_path)
    evaluator = _evaluator_descriptor(
        evaluator_python=evaluator_python,
        evaluator_script=evaluator_script,
        model_path=model_path,
        model_config=model_config,
        gpu_id=gpu_id,
    )
    # Preserve the registry identity in published summaries when the formal
    # CLI supplied the registered byte set.  Unregistered programmatic test
    # callers remain supported, but their summaries are not formal-sealed.
    for candidate_id in GENEVAL_EVALUATOR_REGISTRY:
        try:
            candidate = _registered_evaluator(candidate_id)
        except (OSError, ValueError):
            continue
        if all(evaluator.get(field) == candidate.get(field) for field in (
            "python_path", "python_sha256", "script_path", "script_sha256",
            "model_path", "model_tree_sha256",
        )):
            evaluator = candidate
            break
    raw_path = Path(run_dir) / "geneval" / "raw_results.jsonl"
    binding_path = Path(run_dir) / "geneval" / "evaluator_binding.json"
    upstream_path = Path(f"{layout_dir}_geneval.jsonl")
    binding = {
        "schema": "repldm.geneval_evaluator_binding.v1",
        "layout_sha256": layout["layout_sha256"],
        "input_manifest_sha256": input_manifest_sha256,
        "evaluator": evaluator,
    }
    binding_payload = canonical_json(binding) + b"\n"
    if upstream_path.is_file() or upstream_path.is_symlink():
        if not binding_path.is_file() or binding_path.is_symlink():
            raise RuntimeError(
                "an existing GenEval evaluator output lacks a provenance binding; "
                "remove it only after an explicit audit"
            )
        try:
            existing_binding = _json_file(binding_path, label="GenEval evaluator binding")
        except ValueError:
            raise
        if canonical_json(existing_binding) != canonical_json(binding):
            raise RuntimeError("existing GenEval evaluator output is bound to a different run")
    else:
        _write_new_or_verify(binding_path, binding_payload, label="GenEval evaluator binding")
    if not upstream_path.is_file():
        command = [
            str(evaluator_python),
            str(evaluator_script),
            "--img_path",
            str(Path(layout_dir).parent),
            "--exp_name",
            Path(layout_dir).name,
            "--model-path",
            str(model_path),
            "--gpu_id",
            # CUDA_VISIBLE_DEVICES remaps the selected physical card to index 0.
            "0",
        ]
        if model_config is not None:
            command.extend(["--model-config", str(model_config)])
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_DISABLED": "true",
            }
        )
        log_path = Path(run_dir) / "geneval" / "evaluator.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                command,
                cwd=str(evaluator_script.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("GenEval evaluator timed out; partial output is not accepted") from exc
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"GenEval evaluator failed with exit code {completed.returncode}; see {log_path}"
            )
    if not upstream_path.is_file():
        raise RuntimeError("GenEval evaluator completed without its JSONL output")
    raw_rows = _read_jsonl(upstream_path, label="raw GenEval evaluator results")
    raw_payload = upstream_path.read_bytes()
    raw_hash = sha256_bytes(raw_payload)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{raw_path.name}.", suffix=".tmp", dir=str(raw_path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as destination, upstream_path.open("rb") as source:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, raw_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    summary = aggregate_raw_results(
        raw_rows,
        input_rows,
        prompt_manifest,
        run_dir=run_dir,
        layout_dir=layout_dir,
        input_manifest_sha256=input_manifest_sha256,
        raw_results_sha256=raw_hash,
        config_sha256=config_sha256,
        evaluator=evaluator,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        expected_seed_cohort=expected_seed_cohort,
    )
    if summary.get("layout_sha256") != layout["layout_sha256"]:
        raise RuntimeError("GenEval summary layout hash differs from materialized layout")
    summary_path = Path(run_dir) / "geneval" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new_or_verify(
        summary_path,
        canonical_json(summary) + b"\n",
        label="GenEval summary",
    )
    return summary


def aggregate_raw_results(
    raw_rows: Iterable[Mapping[str, Any]],
    input_rows: Sequence[Mapping[str, Any]],
    prompt_manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    layout_dir: Path,
    input_manifest_sha256: str,
    raw_results_sha256: str,
    config_sha256: str,
    evaluator: Mapping[str, Any],
    scores_path: Path | None = None,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    expected_seed_cohort: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize raw evaluator rows, write scores, and publish a sealed summary."""
    evaluator = _validate_evaluator_descriptor(evaluator)
    normalized = normalize_results(
        raw_rows,
        input_rows,
        prompt_manifest,
        layout_dir=layout_dir,
        evaluator=evaluator,
        run_dir=run_dir,
        expected_seed_cohort=expected_seed_cohort,
    )
    destination = scores_path or (Path(run_dir) / "geneval" / "scores.jsonl")
    scores_hash = write_jsonl(destination, normalized)
    summary = aggregate_scores(
        normalized,
        prompt_manifest,
        input_manifest_sha256=input_manifest_sha256,
        raw_results_sha256=raw_results_sha256,
        config_sha256=config_sha256,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        expected_seed_cohort=expected_seed_cohort,
    )
    summary_core = {
        key: value
        for key, value in summary.items()
        if key != "summary_sha256"
    }
    summary_core.update(
        {
            "scores_sha256": scores_hash,
            "layout_sha256": _layout_hash(
                Path(layout_dir),
                expected_seed_cohort=expected_seed_cohort,
            ),
            "evaluator": evaluator,
        }
    )
    return {
        **summary_core,
        "summary_sha256": sha256_bytes(canonical_json(summary_core)),
    }


def _layout_hash(
    layout_dir: Path,
    *,
    expected_seed_cohort: Mapping[str, Any] | None = None,
) -> str:
    descriptor_path = Path(layout_dir) / "layout.json"
    if descriptor_path.is_symlink() or not descriptor_path.is_file():
        raise ValueError("GenEval layout descriptor is missing")
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("GenEval layout descriptor is unreadable") from exc
    if not isinstance(descriptor, Mapping) or descriptor.get("schema") != LAYOUT_SCHEMA:
        raise ValueError("GenEval layout descriptor schema is invalid")
    if (
        descriptor.get("benchmark") != "GenEval"
        or descriptor.get("prompt_count") != EXPECTED_PROMPT_COUNT
        or descriptor.get("samples_per_prompt") != EXPECTED_SAMPLES_PER_PROMPT
        or descriptor.get("image_count") != EXPECTED_IMAGE_COUNT
    ):
        raise ValueError("GenEval layout descriptor counts are invalid")
    layout_cohort_value = descriptor.get("seed_cohort")
    if layout_cohort_value is None:
        if expected_seed_cohort is not None:
            raise ValueError("GenEval layout lacks its shared seed cohort binding")
    else:
        layout_cohort = _normalize_seed_cohort(layout_cohort_value)
        if expected_seed_cohort is not None:
            expected_cohort = _normalize_seed_cohort(expected_seed_cohort)
            if canonical_json(layout_cohort) != canonical_json(expected_cohort):
                raise ValueError("GenEval layout seed cohort differs from the registered one")
    rows = descriptor.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_IMAGE_COUNT:
        raise ValueError("GenEval layout descriptor rows are incomplete")
    cells: set[tuple[int, int]] = set()
    for row_number, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"GenEval layout descriptor row {row_number} is invalid")
        prompt_index = row.get("prompt_index")
        sample_index = row.get("sample_index")
        cell = (prompt_index, sample_index)
        if (
            type(prompt_index) is not int
            or type(sample_index) is not int
            or not 0 <= prompt_index < EXPECTED_PROMPT_COUNT
            or not 0 <= sample_index < EXPECTED_SAMPLES_PER_PROMPT
            or cell in cells
        ):
            raise ValueError("GenEval layout descriptor contains an invalid cell")
        relative = row.get("layout_image_path")
        if not isinstance(relative, str):
            raise ValueError("GenEval layout descriptor lacks image paths")
        expected_relative = f"{prompt_index:05d}/samples/{sample_index:04d}.png"
        if relative != expected_relative:
            raise ValueError("GenEval layout descriptor image path is not canonical")
        image_path = _safe_relative_path(layout_dir, relative, label="layout image path")
        actual_hash = _validate_png(image_path, label=f"GenEval layout image {cell}")
        _require_hash(row.get("image_sha256"), label=f"layout image_sha256 for {cell}")
        if row["image_sha256"] != actual_hash:
            raise ValueError(f"GenEval layout image hash differs at {cell}")
        cells.add(cell)
    if len(cells) != EXPECTED_IMAGE_COUNT:
        raise ValueError("GenEval layout descriptor cells are incomplete")
    expected = descriptor.get("layout_sha256")
    core = {key: value for key, value in descriptor.items() if key != "layout_sha256"}
    if expected != sha256_bytes(canonical_json(core)):
        raise ValueError("GenEval layout descriptor hash is invalid")
    _require_hash(expected, label="layout_sha256")
    return expected


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    try:
        import yaml

        raw = path.read_bytes()
        value = yaml.safe_load(raw.decode("utf-8")) or {}
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read GenEval config: {path}") from exc
    except Exception as exc:  # yaml.YAMLError without importing it at module load
        raise ValueError(f"GenEval config is invalid: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != GENEVAL_SCHEMA:
        raise ValueError("GenEval config schema differs from the registered contract")
    return value, sha256_bytes(raw)


def _config_seed_cohort(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one seed cohort shared by every formal GenEval arm."""
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise ValueError("GenEval config lacks benchmark contract")
    sample_seeds = _validate_sample_seeds(
        benchmark.get("sample_seeds"), label="benchmark.sample_seeds"
    )
    cohort = _normalize_seed_cohort(benchmark.get("seed_cohort"), expected_seeds=sample_seeds)
    return _registered_seed_cohort(cohort)


def _validate_config_contract(config: Mapping[str, Any]) -> None:
    if config.get("status") != "frozen_official_benchmark":
        raise ValueError("GenEval config is not the frozen official benchmark")
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, Mapping) or benchmark.get("name") != "GenEval":
        raise ValueError("GenEval config lacks the benchmark contract")
    expected_benchmark = {
        "prompt_count": EXPECTED_PROMPT_COUNT,
        "samples_per_prompt": EXPECTED_SAMPLES_PER_PROMPT,
        "image_count": EXPECTED_IMAGE_COUNT,
        "subset_evaluation_allowed": False,
    }
    for field, expected in expected_benchmark.items():
        if benchmark.get(field) != expected:
            raise ValueError(f"GenEval config benchmark.{field} differs from the contract")
    metadata_path = benchmark.get("metadata_path")
    if metadata_path != str(DEFAULT_METADATA_PATH):
        raise ValueError("GenEval config must bind the frozen official metadata path")
    if benchmark.get("metadata_sha256") != DEFAULT_METADATA_SHA256:
        raise ValueError("GenEval config must bind the frozen official metadata hash")
    _config_seed_cohort(config)
    tags = benchmark.get("expected_tags")
    if tags != EXPECTED_TAG_COUNTS:
        raise ValueError("GenEval config task counts differ from the official contract")
    aggregation = config.get("aggregation")
    if not isinstance(aggregation, Mapping):
        raise ValueError("GenEval config lacks aggregation contract")
    if aggregation.get("unit") != "prompt":
        raise ValueError("GenEval aggregation unit must be prompt")
    if aggregation.get("score_definition") != "unweighted_mean_of_six_task_image_accuracies":
        raise ValueError("GenEval aggregation score definition is invalid")
    if aggregation.get("bootstrap_method") != "prompt_cluster_percentile_type7_v1":
        raise ValueError("GenEval bootstrap method is invalid")
    _require_plain_int(aggregation.get("bootstrap_seed"), label="aggregation.bootstrap_seed")
    _require_plain_int(
        aggregation.get("bootstrap_resamples"),
        label="aggregation.bootstrap_resamples",
        minimum=1,
    )
    if aggregation.get("confidence_interval") != [0.025, 0.975]:
        raise ValueError("GenEval confidence interval differs from the contract")
    evaluator = config.get("evaluator")
    if not isinstance(evaluator, Mapping) or evaluator.get("expected_result_rows") != EXPECTED_IMAGE_COUNT:
        raise ValueError("GenEval evaluator contract has the wrong result count")
    if evaluator.get("network_required") is not False:
        raise ValueError("GenEval evaluator must be explicitly offline")


def _config_prompt_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _validate_config_contract(config)
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise ValueError("GenEval config lacks benchmark contract")
    path = Path(str(benchmark.get("metadata_path", DEFAULT_METADATA_PATH)))
    expected_hash = benchmark.get("metadata_sha256")
    return load_prompt_manifest(
        path,
        expected_sha256=expected_hash,
        expected_prompt_count=int(benchmark.get("prompt_count", EXPECTED_PROMPT_COUNT)),
        expected_samples_per_prompt=int(
            benchmark.get("samples_per_prompt", EXPECTED_SAMPLES_PER_PROMPT)
        ),
    )


def _run_dir(path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"GenEval run directory is missing or unsafe: {path}")
    return path


def _run_child(run_dir: Path, value: Path | None, default: Path, *, label: str) -> Path:
    root = _run_dir(run_dir)
    candidate = Path(value if value is not None else default)
    if candidate.is_absolute():
        candidate = candidate.absolute()
    else:
        candidate = (root / candidate).absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the GenEval run directory") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink: {current}")
    return candidate


def _parse_seed_list(value: str | None, config: Mapping[str, Any]) -> tuple[int, ...]:
    registered = tuple(_config_seed_cohort(config)["seeds"])
    if value is None:
        seeds = registered
    else:
        try:
            seeds = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
        except ValueError as exc:
            raise ValueError("--seeds must be a comma-separated list of integers") from exc
    _validate_sample_seeds(seeds)
    if tuple(seeds) != registered:
        raise ValueError(
            "--seeds must exactly match the four seeds in the shared GenEval seed cohort"
        )
    return registered


def _read_input_manifest(
    path: Path,
    *,
    prompt_manifest: Mapping[str, Any],
    run_dir: Path,
    expected_sha256: str | None = None,
    expected_seed_cohort: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    rows = _read_jsonl(path, label="GenEval input manifest")
    actual_sha256 = _jsonl_file_sha256(path, label="GenEval input manifest")
    if expected_sha256 is not None and actual_sha256 != _require_hash(
        expected_sha256, label="input_manifest_sha256"
    ):
        raise ValueError("GenEval input manifest hash differs from the registered value")
    _validate_input_rows(
        rows,
        prompt_manifest,
        run_dir=run_dir,
        expected_seed_cohort=expected_seed_cohort,
    )
    return rows, actual_sha256


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    validated = validate_summary(summary, require_sealed=True)
    _write_new_or_verify(path, canonical_json(validated) + b"\n", label="GenEval summary")


def _evaluator_from_args(args: argparse.Namespace) -> dict[str, Any]:
    evaluator_id = getattr(args, "evaluator_id", DEFAULT_EVALUATOR_ID)
    registered = _registered_evaluator(evaluator_id)
    # Keep legacy flags parseable for scripts, but never allow them to select
    # arbitrary code/assets for a formal CLI invocation.
    supplied = {
        "python_path": str(Path(args.evaluator_python).absolute()),
        "script_path": str(Path(args.evaluator_script).absolute()),
        "model_path": str(Path(args.model_path).absolute()),
    }
    for field, value in supplied.items():
        if value != registered[field]:
            raise ValueError(f"{field} is not the registered GenEval evaluator")
    if args.model_config is not None:
        raise ValueError("--model-config is not supported by the registered GenEval evaluator")
    return registered


def _add_evaluator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluator-id", choices=tuple(GENEVAL_EVALUATOR_REGISTRY), default=DEFAULT_EVALUATOR_ID)
    parser.add_argument("--evaluator-python", type=Path, default=Path(GENEVAL_EVALUATOR_REGISTRY[DEFAULT_EVALUATOR_ID]["python_path"]))
    parser.add_argument("--evaluator-script", type=Path, default=Path(GENEVAL_EVALUATOR_REGISTRY[DEFAULT_EVALUATOR_ID]["script_path"]))
    parser.add_argument("--model-path", type=Path, default=Path(GENEVAL_EVALUATOR_REGISTRY[DEFAULT_EVALUATOR_ID]["model_path"]))
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and aggregate full GenEval runs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--print-manifest", action="store_true")
    commands = parser.add_subparsers(dest="command")

    validate_input = commands.add_parser(
        "validate-input", help="validate generation records and publish 2,212 input rows"
    )
    validate_input.add_argument("--run-dir", type=Path, required=True)
    validate_input.add_argument("--records", type=Path, required=True)
    validate_input.add_argument("--output", type=Path, default=None)
    validate_input.add_argument("--checkpoint-id", required=True)
    validate_input.add_argument("--checkpoint-sha256", required=True)
    validate_input.add_argument("--method", required=True)
    validate_input.add_argument("--run-contract-sha256", required=True)
    validate_input.add_argument("--seeds", default=None)

    prepare_layout = commands.add_parser(
        "prepare-layout", help="materialize the official GenEval directory layout"
    )
    prepare_layout.add_argument("--run-dir", type=Path, required=True)
    prepare_layout.add_argument("--input-manifest", type=Path, default=None)
    prepare_layout.add_argument("--layout-dir", type=Path, default=None)
    prepare_layout.add_argument("--input-manifest-sha256", default=None)

    aggregate = commands.add_parser(
        "aggregate", help="normalize raw evaluator output and publish a sealed summary"
    )
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.add_argument("--input-manifest", type=Path, default=None)
    aggregate.add_argument("--layout-dir", type=Path, default=None)
    aggregate.add_argument("--raw-results", type=Path, required=True)
    aggregate.add_argument("--scores", type=Path, default=None)
    aggregate.add_argument("--summary", type=Path, default=None)
    aggregate.add_argument("--input-manifest-sha256", default=None)
    _add_evaluator_arguments(aggregate)

    run = commands.add_parser(
        "run", help="materialize, run the reviewed evaluator, and publish a summary"
    )
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--input-manifest", type=Path, default=None)
    run.add_argument("--layout-dir", type=Path, default=None)
    run.add_argument("--input-manifest-sha256", default=None)
    run.add_argument("--timeout-seconds", type=float, default=86_400.0)
    _add_evaluator_arguments(run)

    validate = commands.add_parser("validate-summary", help="validate a sealed GenEval summary")
    validate.add_argument("--summary", type=Path, required=True)
    validate.add_argument("--run-dir", type=Path, default=None)
    validate.add_argument("--allow-unsealed", action="store_true")
    return parser


def _load_cli_contract(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Any]]:
    config, config_hash = _load_config(args.config)
    if args.metadata is not None:
        supplied = Path(args.metadata).expanduser().absolute()
        if supplied != DEFAULT_METADATA_PATH:
            raise ValueError(
                "--metadata cannot replace the frozen official GenEval metadata source"
            )
    manifest = _config_prompt_manifest(config)
    return config, config_hash, manifest


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config, config_hash, prompt_manifest = _load_cli_contract(args)
    seed_cohort = _config_seed_cohort(config)
    if args.command is None:
        if args.print_manifest:
            print(json.dumps(prompt_manifest, sort_keys=True, ensure_ascii=True))
        else:
            print(
                json.dumps(
                    {
                        "config_sha256": config_hash,
                        "prompt_manifest_sha256": prompt_manifest["manifest_sha256"],
                        "prompts": prompt_manifest["prompt_count"],
                        "images": prompt_manifest["prompt_count"] * prompt_manifest["samples_per_prompt"],
                        "tag_counts": prompt_manifest["tag_counts"],
                        "seed_cohort": seed_cohort,
                    },
                    sort_keys=True,
                )
            )
        return 0

    if args.command == "validate-input":
        run_dir = _run_dir(args.run_dir)
        records_path = Path(args.records).absolute()
        records = _read_jsonl(records_path, label="generation records")
        rows = build_input_manifest(
            records,
            prompt_manifest,
            run_dir=run_dir,
            checkpoint_id=args.checkpoint_id,
            checkpoint_sha256=args.checkpoint_sha256,
            method=args.method,
            run_contract_sha256=args.run_contract_sha256,
            sample_seeds=_parse_seed_list(args.seeds, config),
            seed_cohort=seed_cohort,
        )
        output = _run_child(run_dir, args.output, DEFAULT_INPUT_MANIFEST, label="input manifest")
        input_hash = write_input_manifest(output, rows)
        print(json.dumps({
            "input_manifest": str(output),
            "input_manifest_sha256": input_hash,
            "prompt_manifest_sha256": prompt_manifest["manifest_sha256"],
            "images": len(rows),
        }, sort_keys=True))
        return 0

    if args.command == "prepare-layout":
        run_dir = _run_dir(args.run_dir)
        input_path = _run_child(run_dir, args.input_manifest, DEFAULT_INPUT_MANIFEST, label="input manifest")
        layout_path = _run_child(run_dir, args.layout_dir, DEFAULT_LAYOUT_DIR, label="layout directory")
        rows, input_hash = _read_input_manifest(
            input_path,
            prompt_manifest=prompt_manifest,
            run_dir=run_dir,
            expected_sha256=args.input_manifest_sha256,
            expected_seed_cohort=seed_cohort,
        )
        descriptor = materialize_official_layout(
            rows,
            prompt_manifest,
            run_dir=run_dir,
            layout_dir=layout_path,
            input_manifest_sha256=input_hash,
            expected_seed_cohort=seed_cohort,
        )
        print(json.dumps({
            "layout_dir": str(layout_path),
            "layout_sha256": descriptor["layout_sha256"],
            "input_manifest_sha256": input_hash,
            "images": descriptor["image_count"],
        }, sort_keys=True))
        return 0

    if args.command == "aggregate":
        run_dir = _run_dir(args.run_dir)
        input_path = _run_child(run_dir, args.input_manifest, DEFAULT_INPUT_MANIFEST, label="input manifest")
        layout_path = _run_child(run_dir, args.layout_dir, DEFAULT_LAYOUT_DIR, label="layout directory")
        raw_path = _run_child(run_dir, args.raw_results, DEFAULT_RAW_RESULTS, label="raw results")
        rows, input_hash = _read_input_manifest(
            input_path,
            prompt_manifest=prompt_manifest,
            run_dir=run_dir,
            expected_sha256=args.input_manifest_sha256,
            expected_seed_cohort=seed_cohort,
        )
        evaluator = _evaluator_from_args(args)
        raw_rows = _read_jsonl(raw_path, label="raw GenEval evaluator results")
        summary = aggregate_raw_results(
            raw_rows,
            rows,
            prompt_manifest,
            run_dir=run_dir,
            layout_dir=layout_path,
            input_manifest_sha256=input_hash,
            raw_results_sha256=sha256_file(raw_path),
            config_sha256=config_hash,
            evaluator=evaluator,
            scores_path=_run_child(run_dir, args.scores, DEFAULT_SCORES, label="scores"),
            bootstrap_seed=int(config["aggregation"]["bootstrap_seed"]),
            bootstrap_resamples=int(config["aggregation"]["bootstrap_resamples"]),
            expected_seed_cohort=seed_cohort,
        )
        summary_path = _run_child(run_dir, args.summary, DEFAULT_SUMMARY, label="summary")
        _write_summary(summary_path, summary)
        print(json.dumps({**summary, "summary_path": str(summary_path)}, sort_keys=True))
        return 0

    if args.command == "run":
        run_dir = _run_dir(args.run_dir)
        input_path = _run_child(run_dir, args.input_manifest, DEFAULT_INPUT_MANIFEST, label="input manifest")
        layout_path = _run_child(run_dir, args.layout_dir, DEFAULT_LAYOUT_DIR, label="layout directory")
        rows, input_hash = _read_input_manifest(
            input_path,
            prompt_manifest=prompt_manifest,
            run_dir=run_dir,
            expected_sha256=args.input_manifest_sha256,
            expected_seed_cohort=seed_cohort,
        )
        evaluator = _evaluator_from_args(args)
        summary = run_official_evaluator(
            input_rows=rows,
            prompt_manifest=prompt_manifest,
            run_dir=run_dir,
            layout_dir=layout_path,
            input_manifest_sha256=input_hash,
            config_sha256=config_hash,
            evaluator_python=Path(evaluator["python_path"]),
            evaluator_script=Path(evaluator["script_path"]),
            model_path=Path(evaluator["model_path"]),
            model_config=None,
            gpu_id=args.gpu_id,
            timeout_seconds=args.timeout_seconds,
            bootstrap_seed=int(config["aggregation"]["bootstrap_seed"]),
            bootstrap_resamples=int(config["aggregation"]["bootstrap_resamples"]),
            expected_seed_cohort=seed_cohort,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0

    summary = _json_file(args.summary, label="GenEval summary")
    if args.run_dir is not None:
        if args.allow_unsealed:
            raise ValueError("--allow-unsealed cannot be combined with --run-dir")
        validate_summary_files(
            summary,
            run_dir=args.run_dir,
            expected_seed_cohort=seed_cohort,
        )
    else:
        validate_summary(
            summary,
            require_sealed=not args.allow_unsealed,
            expected_seed_cohort=seed_cohort,
        )
    print(json.dumps({"summary": str(Path(args.summary).absolute()), "validated": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_SAMPLE_SEEDS",
    "DEFAULT_SEED_COHORT_SHA256",
    "DEFAULT_METADATA_PATH",
    "DEFAULT_METADATA_SHA256",
    "DEFAULT_INPUT_MANIFEST",
    "DEFAULT_LAYOUT_DIR",
    "DEFAULT_RAW_RESULTS",
    "DEFAULT_SCORES",
    "DEFAULT_SUMMARY",
    "DEFAULT_SEED_COHORT_ID",
    "EXPECTED_IMAGE_COUNT",
    "EXPECTED_PROMPT_COUNT",
    "EXPECTED_SAMPLES_PER_PROMPT",
    "GENEVAL_SCHEMA",
    "SEED_COHORT_SCHEMA",
    "REGISTERED_SEED_COHORTS",
    "aggregate_scores",
    "aggregate_raw_results",
    "build_input_manifest",
    "canonical_json",
    "load_prompt_manifest",
    "materialize_official_layout",
    "normalize_results",
    "rows_sha256",
    "run_official_evaluator",
    "sha256_file",
    "seed_cohort_sha256",
    "validate_shared_seed_cohort",
    "validate_summary",
    "validate_summary_files",
    "write_input_manifest",
    "write_jsonl",
]
