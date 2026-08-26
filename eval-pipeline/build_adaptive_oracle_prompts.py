"""Build fresh, registration-only prompts for the adaptive-oracle smoke."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import pathlib
import re
import stat
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Optional

import yaml


SOURCE_REPOSITORY = "https://github.com/google-research/parti"
SOURCE_REVISION = "5a657978134374ce28973948331b319adef164bd"
SOURCE_SHA256 = "fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb"
SELECTION_NAMESPACE = "repldm-adaptive-oracle-engineering-v1"
SEED_NAMESPACE = f"{SELECTION_NAMESPACE}-seed"
EXPECTED_CHALLENGE_COUNT = 11
OUTPUT_CSV = "adaptive_oracle_engineering.csv"
OUTPUT_MANIFEST = "adaptive_oracle_prompt_manifest_v1.json"
OUTPUT_EXCLUSION_INVENTORY = "adaptive_oracle_exclusion_inventory_v1.json"
EXCLUSION_INVENTORY_SCHEMA = "adaptive_oracle_exclusion_inventory_v1"
OUTPUT_NAMES = frozenset(
    (OUTPUT_CSV, OUTPUT_MANIFEST, OUTPUT_EXCLUSION_INVENTORY)
)
SELF_METADATA_NAMES = frozenset(
    (
        "adaptive_oracle_engineering_cpu_audit_v1.json",
        "adaptive_oracle_engineering_authorization_v1.yaml",
        "adaptive_oracle_engineering_registration_v1.yaml",
        "generation_environment_adaptive_oracle_v2.yaml",
    )
)
SELF_OUTPUT_PREFIXES = ("adaptive_oracle/engineering_v1",)
CSV_FIELDS = (
    "index",
    "TEXT",
    "bucket",
    "source_row",
    "source_category",
    "source_challenge",
    "source_note",
    "split",
    "prompt_row_id",
    "seed",
)

_PROMPT_KEYS = frozenset(("prompt", "prompts", "text", "positive_prompt"))
_SOURCE_ROW_KEYS = frozenset(("source_row", "source_rows", "source_row_id"))
_SEED_KEY_EXCLUSIONS = (
    "count",
    "counter",
    "digest",
    "hash",
    "namespace",
    "schema",
    "sha256",
)
_OUTCOME_KEY_FRAGMENTS = (
    "aggregate",
    "analysis",
    "aesthetic",
    "audit",
    "evaluation",
    "hps",
    "imagereward",
    "metric",
    "outcome",
    "preference",
    "quality",
    "rank",
    "report",
    "result",
    "reward",
    "score",
    "statistic",
    "topiq",
)
_OUTCOME_PATH_FRAGMENTS = ("quality", "score")
_METADATA_SUFFIXES = frozenset((".json", ".jsonl", ".yaml", ".yml"))
_INVENTORY_ROLES = (
    "repository_prompt_csv",
    "repository_prompt_manifest",
    "repository_config",
    "output_config",
    "output_manifest",
    "output_sidecar",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTITY_FIELD_ALLOWLIST = (
    "prompt",
    "prompts",
    "text",
    "positive_prompt",
    "source_row",
    "source_rows",
    "source_row_id",
    "seed",
    "seeds",
    "*_seed",
    "*_seeds",
    "used_seed*",
    "reserved_seed*",
    "retired_seed*",
    "observed_seed*",
    "prior_seed*",
    "generated_seed*",
    "seed_list",
    "seed_values",
    "seeds_by_split",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(payload)


def normalize_prompt(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split()).casefold()


def _read_csv(path: pathlib.Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader)


def _source_rows(source_path: pathlib.Path) -> list[dict[str, Any]]:
    source_bytes = source_path.read_bytes()
    if sha256_bytes(source_bytes) != SOURCE_SHA256:
        raise ValueError("PartiPrompts.tsv differs from the frozen source hash")
    rows = _read_csv(source_path, delimiter="\t")
    required = {"Prompt", "Category", "Challenge", "Note"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("PartiPrompts.tsv has an unexpected schema")
    source_rows = []
    for index, row in enumerate(rows):
        prompt = row["Prompt"].strip()
        challenge = row["Challenge"].strip()
        if not prompt or not challenge:
            raise ValueError("PartiPrompts.tsv contains an empty prompt or challenge")
        source_rows.append(
            {
                "source_row": index,
                "prompt": prompt,
                "category": row["Category"].strip(),
                "challenge": challenge,
                "note": row["Note"].strip(),
            }
        )
    return source_rows


def _verify_source_revision(source_repo: pathlib.Path) -> None:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot verify the Parti source Git revision") from exc
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"Parti source revision {revision!r} differs from {SOURCE_REVISION!r}"
        )


def _normalized_key(value: Any) -> str:
    text = str(value)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _is_outcome_key(key: str) -> bool:
    return any(fragment in key for fragment in _OUTCOME_KEY_FRAGMENTS)


def _is_seed_key(key: str) -> bool:
    if "seed" not in key or any(key.endswith(f"_{value}") for value in _SEED_KEY_EXCLUSIONS):
        return False
    return (
        key in {"seed", "seeds"}
        or key.endswith("_seed")
        or key.endswith("_seeds")
        or key.startswith("used_seed")
        or key.startswith("reserved_seed")
        or key.startswith("retired_seed")
        or key.startswith("observed_seed")
        or key.startswith("prior_seed")
        or key.startswith("generated_seed")
        or key in {"seed_list", "seed_values", "seeds_by_split"}
    )


def _integer_values(value: Any) -> Iterable[int]:
    if isinstance(value, bool):
        return
    if isinstance(value, Integral):
        yield int(value)
        return
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            yield int(number)
        return
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            yield int(stripped)
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _integer_values(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _integer_values(child)


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _string_values(child)


def _empty_projection() -> dict[str, Any]:
    return {
        "normalized_prompts": [],
        "source_rows": [],
        "seeds": [],
        "forbidden_fields_skipped": 0,
    }


def _walk_identity_fields(
    value: Any,
    projection: dict[str, Any],
    *,
    collect_prompts: bool,
) -> None:
    """Traverse an outcome container without consuming its outcome scalars."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _normalized_key(raw_key)
            if collect_prompts and key in _PROMPT_KEYS:
                projection["normalized_prompts"].extend(
                    normalized
                    for item in _string_values(child)
                    if (normalized := normalize_prompt(item))
                )
            if key in _SOURCE_ROW_KEYS:
                projection["source_rows"].extend(_integer_values(child))
            if _is_seed_key(key):
                projection["seeds"].extend(_integer_values(child))
            if isinstance(child, (Mapping, list, tuple)):
                _walk_identity_fields(
                    child,
                    projection,
                    collect_prompts=collect_prompts,
                )
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, (Mapping, list, tuple)):
                _walk_identity_fields(
                    child,
                    projection,
                    collect_prompts=collect_prompts,
                )


def _walk_metadata(
    value: Any,
    projection: dict[str, Any],
    *,
    collect_prompts: bool,
    skip_outcome_fields: bool,
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _normalized_key(raw_key)
            if skip_outcome_fields and _is_outcome_key(key):
                projection["forbidden_fields_skipped"] += 1
                _walk_identity_fields(
                    child,
                    projection,
                    collect_prompts=collect_prompts,
                )
                continue
            if collect_prompts and key in _PROMPT_KEYS:
                projection["normalized_prompts"].extend(
                    normalized
                    for item in _string_values(child)
                    if (normalized := normalize_prompt(item))
                )
            if key in _SOURCE_ROW_KEYS:
                projection["source_rows"].extend(_integer_values(child))
            if _is_seed_key(key):
                projection["seeds"].extend(_integer_values(child))
            _walk_metadata(
                child,
                projection,
                collect_prompts=collect_prompts,
                skip_outcome_fields=skip_outcome_fields,
            )
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _walk_metadata(
                child,
                projection,
                collect_prompts=collect_prompts,
                skip_outcome_fields=skip_outcome_fields,
            )


def _stable_regular_bytes(path: pathlib.Path, *, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(f"cannot open {label} as a non-symlink file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"{label} changed while it was read: {path}")
    value = b"".join(chunks)
    if len(value) != before.st_size:
        raise RuntimeError(f"{label} byte count differs from its file identity: {path}")
    return value


def _decode_inventory_text(path: pathlib.Path, value: bytes) -> str:
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"metadata inventory file is not strict UTF-8: {path}") from exc


def _load_documents(path: pathlib.Path, value: bytes) -> list[Any]:
    suffix = path.suffix.casefold()
    text = _decode_inventory_text(path, value)

    def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, child in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = child
        return result

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_unique_mapping(
        loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise ValueError("YAML metadata contains an unhashable key") from exc
            if duplicate:
                raise ValueError(f"duplicate YAML key {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_unique_mapping,
    )
    try:
        if suffix == ".json":
            return (
                []
                if not text.strip()
                else [json.loads(text, object_pairs_hook=unique_json_object)]
            )
        if suffix == ".jsonl":
            return [
                json.loads(line, object_pairs_hook=unique_json_object)
                for line in text.splitlines()
                if line.strip()
            ]
        if suffix in {".yaml", ".yml"}:
            return [
                document
                for document in yaml.load_all(text, Loader=UniqueKeySafeLoader)
                if document is not None
            ]
    except (json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"cannot parse metadata inventory file {path}") from exc
    raise ValueError(f"unsupported metadata inventory file {path}")


def _path_label(path: pathlib.Path, repo_root: pathlib.Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"metadata inventory path {path} is outside repository") from exc


def _metadata_paths(
    repo_root: pathlib.Path,
    prompt_dir: pathlib.Path,
    outputs_root: pathlib.Path,
) -> tuple[list[tuple[str, pathlib.Path]], list[str]]:
    config_dir = repo_root / "eval-pipeline" / "configs"
    for required in (prompt_dir, config_dir, outputs_root):
        if not required.is_dir():
            raise ValueError(f"required inventory directory is missing: {required}")

    self_prompt_paths = {prompt_dir / name for name in OUTPUT_NAMES}
    self_config_paths = {config_dir / name for name in SELF_METADATA_NAMES}
    paths: list[tuple[str, pathlib.Path]] = []
    for path in sorted(prompt_dir.rglob("*.csv")):
        if path not in self_prompt_paths:
            paths.append(("repository_prompt_csv", path))
    for path in sorted(prompt_dir.rglob("*")):
        if (
            path.is_file()
            and path not in self_prompt_paths
            and path.suffix.casefold() in {".json", ".jsonl"}
        ):
            paths.append(("repository_prompt_manifest", path))
    for path in sorted(config_dir.rglob("*")):
        if (
            path.is_file()
            and path not in self_config_paths
            and path.suffix.casefold() in _METADATA_SUFFIXES
        ):
            paths.append(("repository_config", path))

    forbidden_output_paths = []
    for path in sorted(outputs_root.rglob("*")):
        relative_output = path.relative_to(outputs_root).as_posix()
        if any(
            relative_output == prefix or relative_output.startswith(prefix + "/")
            for prefix in SELF_OUTPUT_PREFIXES
        ):
            continue
        if not path.is_file() or path.suffix.casefold() not in _METADATA_SUFFIXES:
            continue
        lowered_name = path.name.casefold()
        if any(fragment in lowered_name for fragment in _OUTCOME_PATH_FRAGMENTS):
            forbidden_output_paths.append(_path_label(path, repo_root))
            continue
        if "manifest" in lowered_name:
            role = "output_manifest"
        elif "config" in lowered_name or path.suffix.casefold() in {".yaml", ".yml"}:
            role = "output_config"
        elif lowered_name == "attempt.json":
            role = "output_sidecar"
        elif (
            "sidecar" in lowered_name
            or path.parent.name.casefold() in {"images", "records", "sidecars"}
        ):
            role = "output_sidecar"
        else:
            continue
        paths.append((role, path))

    unique_paths = {(role, path) for role, path in paths}
    return (
        sorted(unique_paths, key=lambda item: (item[0], item[1].as_posix())),
        sorted(forbidden_output_paths),
    )


def _scan_prompt_csv(
    path: pathlib.Path, value: bytes
) -> tuple[int, dict[str, Any]]:
    text = _decode_inventory_text(path, value)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise ValueError(f"{path} has no CSV header")
    rows = list(reader)
    projection = _empty_projection()
    if rows:
        fieldnames = {_normalized_key(key): key for key in rows[0]}
        prompt_key = next(
            (fieldnames[key] for key in ("text", "prompt") if key in fieldnames),
            None,
        )
        if prompt_key is None:
            raise ValueError(f"repository prompt CSV {path} has no TEXT or prompt column")
        source_key = fieldnames.get("source_row")
        seed_keys = [original for key, original in fieldnames.items() if _is_seed_key(key)]
        for row in rows:
            normalized = normalize_prompt(row.get(prompt_key, ""))
            if normalized:
                projection["normalized_prompts"].append(normalized)
            if source_key is not None:
                projection["source_rows"].extend(
                    _integer_values(row.get(source_key, ""))
                )
            for seed_key in seed_keys:
                projection["seeds"].extend(_integer_values(row.get(seed_key, "")))
    return len(rows), projection


def _inventory_from_file_records(
    file_records: list[dict[str, Any]], forbidden_output_paths: list[str]
) -> dict[str, Any]:
    role_counts = {
        role: {
            "file_count": 0,
            "record_count": 0,
            "prompt_occurrence_count": 0,
            "explicit_source_row_occurrence_count": 0,
            "seed_occurrence_count": 0,
            "forbidden_fields_skipped": 0,
        }
        for role in _INVENTORY_ROLES
    }
    digest_records = []
    normalized_prompts: set[str] = set()
    source_rows: set[int] = set()
    used_reserved_seeds: set[int] = set()

    for record in file_records:
        role = record["role"]
        record_count = record["record_count"]
        projection = record["projection"]
        occurrence_counts = record["occurrence_counts"]
        occurrence_digests = record["occurrence_digests"]
        prompts = list(projection["normalized_prompts"])
        rows = list(projection["source_rows"])
        seeds = list(projection["used_reserved_seeds"])
        normalized_prompts.update(prompts)
        source_rows.update(rows)
        used_reserved_seeds.update(seeds)
        counts = role_counts[role]
        counts["file_count"] += 1
        counts["record_count"] += record_count
        counts["prompt_occurrence_count"] += occurrence_counts[
            "normalized_prompts"
        ]
        counts["explicit_source_row_occurrence_count"] += occurrence_counts[
            "source_rows"
        ]
        counts["seed_occurrence_count"] += occurrence_counts[
            "used_reserved_seeds"
        ]
        counts["forbidden_fields_skipped"] += int(
            projection["forbidden_fields_skipped"]
        )
        digest_records.append(
            {
                "path": record["path"],
                "role": role,
                "record_count": record_count,
                "normalized_prompts_sha256": occurrence_digests[
                    "normalized_prompts"
                ],
                "source_rows_sha256": occurrence_digests["source_rows"],
                "seeds_sha256": occurrence_digests["used_reserved_seeds"],
                "forbidden_fields_skipped": int(
                    projection["forbidden_fields_skipped"]
                ),
            }
        )

    public_inventory = {
        "roles": role_counts,
        "scanned_file_count": len(digest_records),
        "scanned_record_count": sum(
            counts["record_count"] for counts in role_counts.values()
        ),
        "file_path_digest": canonical_sha256(
            sorted(record["path"] for record in digest_records)
        ),
        "metadata_projection_digest": canonical_sha256(digest_records),
        "forbidden_output_path_count": len(forbidden_output_paths),
        "outcome_container_count_identity_only": sum(
            counts["forbidden_fields_skipped"] for counts in role_counts.values()
        ),
        "repository_metadata_files_parsed_for_identity_inventory": sum(
            role_counts[role]["file_count"]
            for role in (
                "repository_prompt_csv",
                "repository_prompt_manifest",
                "repository_config",
            )
        ),
        "output_metadata_files_parsed_for_identity_inventory": sum(
            role_counts[role]["file_count"]
            for role in ("output_config", "output_manifest", "output_sidecar")
        ),
        "parsed_identity_field_allowlist": list(_IDENTITY_FIELD_ALLOWLIST),
        "dedicated_score_or_quality_files_read": 0,
        "outcome_field_values_consumed": 0,
        "unique_normalized_prompt_count": len(normalized_prompts),
        "normalized_prompt_inventory_digest": canonical_sha256(
            sorted(normalized_prompts)
        ),
        "unique_explicit_source_row_count": len(source_rows),
        "explicit_source_row_inventory_digest": canonical_sha256(sorted(source_rows)),
        "unique_used_reserved_seed_count": len(used_reserved_seeds),
        "used_reserved_seed_digest": canonical_sha256(sorted(used_reserved_seeds)),
    }
    return {
        "normalized_prompts": normalized_prompts,
        "source_rows": source_rows,
        "used_reserved_seeds": used_reserved_seeds,
        "manifest": public_inventory,
        "files": file_records,
        "forbidden_output_paths": forbidden_output_paths,
    }


def scan_inventory(
    repo_root: pathlib.Path,
    prompt_dir: pathlib.Path,
    outputs_root: pathlib.Path,
) -> dict[str, Any]:
    """Project only prompts, source rows, and seeds from registered metadata."""

    repo_root = repo_root.resolve()
    prompt_dir = prompt_dir.resolve()
    outputs_root = outputs_root.resolve()
    metadata_paths, forbidden_output_paths = _metadata_paths(
        repo_root, prompt_dir, outputs_root
    )
    file_records = []
    for role, path in metadata_paths:
        value = _stable_regular_bytes(path, label="metadata inventory file")
        if role == "repository_prompt_csv":
            record_count, projection = _scan_prompt_csv(path, value)
        else:
            documents = _load_documents(path, value)
            record_count = len(documents)
            projection = _empty_projection()
            collect_prompts = True
            skip_outcome_fields = role in {"output_manifest", "output_sidecar"}
            for document in documents:
                _walk_metadata(
                    document,
                    projection,
                    collect_prompts=collect_prompts,
                    skip_outcome_fields=skip_outcome_fields,
                )
        prompt_occurrences = sorted(
            item for item in projection["normalized_prompts"] if item
        )
        row_occurrences = sorted(
            int(item) for item in projection["source_rows"] if int(item) >= 0
        )
        seed_occurrences = sorted(int(item) for item in projection["seeds"])
        file_records.append(
            {
                "path": _path_label(path, repo_root),
                "role": role,
                "sha256": sha256_bytes(value),
                "byte_count": len(value),
                "record_count": record_count,
                "projection": {
                    "normalized_prompts": sorted(set(prompt_occurrences)),
                    "source_rows": sorted(set(row_occurrences)),
                    "used_reserved_seeds": sorted(set(seed_occurrences)),
                    "forbidden_fields_skipped": int(
                        projection["forbidden_fields_skipped"]
                    ),
                },
                "occurrence_counts": {
                    "normalized_prompts": len(prompt_occurrences),
                    "source_rows": len(row_occurrences),
                    "used_reserved_seeds": len(seed_occurrences),
                },
                "occurrence_digests": {
                    "normalized_prompts": canonical_sha256(prompt_occurrences),
                    "source_rows": canonical_sha256(row_occurrences),
                    "used_reserved_seeds": canonical_sha256(seed_occurrences),
                },
            }
        )
    return _inventory_from_file_records(file_records, forbidden_output_paths)


def _selection_digest(row: Mapping[str, Any]) -> str:
    payload = ":".join(
        (
            SELECTION_NAMESPACE,
            "prompt",
            str(row["challenge"]),
            str(row["source_row"]),
            normalize_prompt(str(row["prompt"])),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_engineering_prompts(
    source_rows: Iterable[Mapping[str, Any]],
    excluded_texts: set[str],
    excluded_source_rows: set[int],
) -> list[dict[str, Any]]:
    canonical_by_text: dict[str, Mapping[str, Any]] = {}
    source_challenges = set()
    for row in source_rows:
        challenge = str(row["challenge"])
        source_challenges.add(challenge)
        normalized = normalize_prompt(str(row["prompt"]))
        source_row = int(row["source_row"])
        if normalized in excluded_texts or source_row in excluded_source_rows:
            continue
        previous = canonical_by_text.get(normalized)
        if previous is None or source_row < int(previous["source_row"]):
            canonical_by_text[normalized] = row
    if len(source_challenges) != EXPECTED_CHALLENGE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_CHALLENGE_COUNT} Parti challenges, "
            f"found {len(source_challenges)}"
        )

    by_challenge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_by_text.values():
        selected = dict(row)
        selected["selection_digest"] = _selection_digest(row)
        by_challenge[str(row["challenge"])].append(selected)
    if set(by_challenge) != source_challenges:
        missing = sorted(source_challenges - set(by_challenge))
        raise ValueError(f"Parti challenges lack unused prompts: {missing}")

    selected_rows = []
    for challenge in sorted(source_challenges):
        ranked = sorted(
            by_challenge[challenge],
            key=lambda row: (row["selection_digest"], int(row["source_row"])),
        )
        selected = dict(ranked[0])
        selected["rank_within_challenge"] = 0
        selected_rows.append(selected)
    return selected_rows


def _seed_candidate(counter: int) -> tuple[int, str]:
    if isinstance(counter, bool) or not isinstance(counter, Integral) or counter < 0:
        raise ValueError("seed counter must be a non-negative integer")
    payload = f"{SEED_NAMESPACE}:engineering:{int(counter)}"
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF, digest


def select_engineering_seed(used_reserved_seeds: set[int]) -> dict[str, Any]:
    counter = 0
    while True:
        seed, digest = _seed_candidate(counter)
        if seed > 0 and seed not in used_reserved_seeds:
            return {
                "seed": seed,
                "counter": counter,
                "selection_digest": digest,
                "status": "reserved_retired_on_use",
                "retired_on_use": True,
                "retirement_trigger": "first_generation_attempt_regardless_of_outcome",
            }
        counter += 1


def _bucket(challenge: str) -> str:
    return challenge.casefold().replace(" & ", "-and-").replace(" ", "-")


def _csv_bytes(rows: Iterable[Mapping[str, Any]], seed: int) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for index, row in enumerate(rows):
        writer.writerow(
            {
                "index": index,
                "TEXT": row["prompt"],
                "bucket": _bucket(str(row["challenge"])),
                "source_row": row["source_row"],
                "source_category": row["category"],
                "source_challenge": row["challenge"],
                "source_note": row["note"],
                "split": "engineering",
                "prompt_row_id": f"engineering-{index + 1:04d}",
                "seed": seed,
            }
        )
    return buffer.getvalue().encode("utf-8")


def _json_asset_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _text_derived_source_rows(
    source_rows: Iterable[Mapping[str, Any]], normalized_prompts: Iterable[str]
) -> set[int]:
    source_by_text: dict[str, set[int]] = defaultdict(set)
    for row in source_rows:
        source_by_text[normalize_prompt(str(row["prompt"]))].add(
            int(row["source_row"])
        )
    return {
        source_row
        for prompt in normalized_prompts
        for source_row in source_by_text.get(prompt, set())
    }


def _exclusion_inventory_value(
    inventory: Mapping[str, Any], source_rows: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    text_derived_rows = _text_derived_source_rows(
        source_rows, inventory["normalized_prompts"]
    )
    explicit_rows = set(inventory["source_rows"])
    excluded = {
        "normalized_prompts": sorted(inventory["normalized_prompts"]),
        "explicit_source_rows": sorted(explicit_rows),
        "text_derived_source_rows": sorted(text_derived_rows),
        "source_rows": sorted(explicit_rows | text_derived_rows),
        "used_reserved_seeds": sorted(inventory["used_reserved_seeds"]),
    }
    files = list(inventory["files"])
    forbidden_paths = list(inventory["forbidden_output_paths"])
    return {
        "schema": EXCLUSION_INVENTORY_SCHEMA,
        "status": "frozen_identity_projection_only_no_outcomes",
        "selection_namespace": SELECTION_NAMESPACE,
        "normalization_rule": "Unicode NFKC, collapse whitespace, then casefold",
        "policy": {
            "roles": list(_INVENTORY_ROLES),
            "parsed_identity_field_allowlist": list(_IDENTITY_FIELD_ALLOWLIST),
            "forbidden_field_fragments": list(_OUTCOME_KEY_FRAGMENTS),
            "forbidden_output_path_fragments": list(_OUTCOME_PATH_FRAGMENTS),
            "self_metadata_excluded_names": sorted(SELF_METADATA_NAMES),
            "self_output_excluded_prefixes": list(SELF_OUTPUT_PREFIXES),
            "dedicated_score_or_quality_files_read": 0,
            "outcome_field_values_consumed": 0,
        },
        "files": files,
        "forbidden_output_paths": forbidden_paths,
        "excluded": excluded,
        "summary": inventory["manifest"],
        "integrity": {
            "files_sha256": canonical_sha256(files),
            "forbidden_output_paths_sha256": canonical_sha256(forbidden_paths),
            "excluded_sha256": canonical_sha256(excluded),
        },
    }


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"exclusion inventory contains duplicate key {key!r}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"exclusion inventory contains non-finite constant {value!r}")


def _require_exact_fields(
    value: Any, fields: Iterable[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{label} fields differ from the frozen schema")
    return value


def _canonical_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    pure = pathlib.PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        raise ValueError(f"{label} is not a canonical repository-relative path")
    return value


def _validated_sorted_ints(
    value: Any, label: str, *, nonnegative: bool
) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or (nonnegative and item < 0)
        for item in value
    ):
        raise ValueError(f"{label} must contain only valid integers")
    if value != sorted(set(value)):
        raise ValueError(f"{label} is not sorted and unique")
    return list(value)


def _validated_frozen_inventory(value: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        decoded = value.decode("utf-8", errors="strict")
        artifact = json.loads(
            decoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("exclusion inventory is not strict canonical JSON") from exc
    if value != _json_asset_bytes(artifact):
        raise ValueError("exclusion inventory bytes are not canonical")
    top = _require_exact_fields(
        artifact,
        frozenset(
            {
                "schema",
                "status",
                "selection_namespace",
                "normalization_rule",
                "policy",
                "files",
                "forbidden_output_paths",
                "excluded",
                "summary",
                "integrity",
            }
        ),
        "exclusion inventory",
    )
    expected_policy = _exclusion_inventory_value(
        {
            "normalized_prompts": set(),
            "source_rows": set(),
            "used_reserved_seeds": set(),
            "files": [],
            "forbidden_output_paths": [],
            "manifest": {},
        },
        [],
    )["policy"]
    if (
        top["schema"] != EXCLUSION_INVENTORY_SCHEMA
        or top["status"] != "frozen_identity_projection_only_no_outcomes"
        or top["selection_namespace"] != SELECTION_NAMESPACE
        or top["normalization_rule"]
        != "Unicode NFKC, collapse whitespace, then casefold"
        or top["policy"] != expected_policy
    ):
        raise ValueError("exclusion inventory policy differs from the builder")
    raw_files = top["files"]
    if not isinstance(raw_files, list):
        raise ValueError("exclusion inventory files must be a list")
    files = []
    for index, raw_record in enumerate(raw_files):
        record = _require_exact_fields(
            raw_record,
            frozenset(
                {
                    "path",
                    "role",
                    "sha256",
                    "byte_count",
                    "record_count",
                    "projection",
                    "occurrence_counts",
                    "occurrence_digests",
                }
            ),
            f"exclusion inventory file {index}",
        )
        path = _canonical_relative_path(record["path"], f"inventory file {index}")
        role = record["role"]
        digest = record["sha256"]
        if role not in _INVENTORY_ROLES:
            raise ValueError(f"inventory file {index} has an invalid role")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"inventory file {index} has an invalid SHA-256")
        for field in ("byte_count", "record_count"):
            item = record[field]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"inventory file {index} has an invalid {field}")
        projection = _require_exact_fields(
            record["projection"],
            frozenset(
                {
                    "normalized_prompts",
                    "source_rows",
                    "used_reserved_seeds",
                    "forbidden_fields_skipped",
                }
            ),
            f"inventory file {index} projection",
        )
        prompts = projection["normalized_prompts"]
        if (
            not isinstance(prompts, list)
            or any(
                not isinstance(item, str)
                or not item
                or normalize_prompt(item) != item
                for item in prompts
            )
            or prompts != sorted(set(prompts))
        ):
            raise ValueError(
                f"inventory file {index} normalized prompts are invalid"
            )
        rows = _validated_sorted_ints(
            projection["source_rows"],
            f"inventory file {index} source rows",
            nonnegative=True,
        )
        seeds = _validated_sorted_ints(
            projection["used_reserved_seeds"],
            f"inventory file {index} seeds",
            nonnegative=False,
        )
        skipped = projection["forbidden_fields_skipped"]
        if isinstance(skipped, bool) or not isinstance(skipped, int) or skipped < 0:
            raise ValueError(
                f"inventory file {index} has an invalid skipped-field count"
            )
        occurrence_counts = _require_exact_fields(
            record["occurrence_counts"],
            frozenset(
                {"normalized_prompts", "source_rows", "used_reserved_seeds"}
            ),
            f"inventory file {index} occurrence counts",
        )
        occurrence_digests = _require_exact_fields(
            record["occurrence_digests"],
            frozenset(
                {"normalized_prompts", "source_rows", "used_reserved_seeds"}
            ),
            f"inventory file {index} occurrence digests",
        )
        minimum_counts = {
            "normalized_prompts": len(prompts),
            "source_rows": len(rows),
            "used_reserved_seeds": len(seeds),
        }
        for name, minimum in minimum_counts.items():
            count = occurrence_counts[name]
            digest_value = occurrence_digests[name]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < minimum
                or not isinstance(digest_value, str)
                or _SHA256.fullmatch(digest_value) is None
            ):
                raise ValueError(
                    f"inventory file {index} has invalid {name} occurrences"
                )
            if count == 0 and digest_value != canonical_sha256([]):
                raise ValueError(
                    f"inventory file {index} empty {name} digest differs"
                )
        files.append(
            {
                "path": path,
                "role": role,
                "sha256": digest,
                "byte_count": record["byte_count"],
                "record_count": record["record_count"],
                "projection": {
                    "normalized_prompts": list(prompts),
                    "source_rows": rows,
                    "used_reserved_seeds": seeds,
                    "forbidden_fields_skipped": skipped,
                },
                "occurrence_counts": dict(occurrence_counts),
                "occurrence_digests": dict(occurrence_digests),
            }
        )
    if files != sorted(files, key=lambda item: (item["role"], item["path"])):
        raise ValueError("exclusion inventory file records are not canonically ordered")
    if len({item["path"] for item in files}) != len(files):
        raise ValueError("exclusion inventory contains duplicate file paths")
    forbidden_paths = top["forbidden_output_paths"]
    if not isinstance(forbidden_paths, list):
        raise ValueError("forbidden output paths must be a list")
    normalized_forbidden_paths = [
        _canonical_relative_path(item, "forbidden output path")
        for item in forbidden_paths
    ]
    if normalized_forbidden_paths != sorted(set(normalized_forbidden_paths)):
        raise ValueError("forbidden output paths are not sorted and unique")
    inventory = _inventory_from_file_records(files, normalized_forbidden_paths)
    excluded = _require_exact_fields(
        top["excluded"],
        frozenset(
            {
                "normalized_prompts",
                "explicit_source_rows",
                "text_derived_source_rows",
                "source_rows",
                "used_reserved_seeds",
            }
        ),
        "excluded identities",
    )
    frozen_text_derived = _validated_sorted_ints(
        excluded["text_derived_source_rows"],
        "excluded text-derived source rows",
        nonnegative=True,
    )
    frozen_source_rows = _validated_sorted_ints(
        excluded["source_rows"], "excluded source rows", nonnegative=True
    )
    expected_excluded = {
        "normalized_prompts": sorted(inventory["normalized_prompts"]),
        "explicit_source_rows": sorted(inventory["source_rows"]),
        "text_derived_source_rows": frozen_text_derived,
        "source_rows": frozen_source_rows,
        "used_reserved_seeds": sorted(inventory["used_reserved_seeds"]),
    }
    if frozen_source_rows != sorted(
        set(inventory["source_rows"]) | set(frozen_text_derived)
    ):
        raise ValueError("excluded source-row union is incomplete")
    if excluded != expected_excluded or top["summary"] != inventory["manifest"]:
        raise ValueError("exclusion inventory projections or summary are incomplete")
    expected_integrity = {
        "files_sha256": canonical_sha256(files),
        "forbidden_output_paths_sha256": canonical_sha256(
            normalized_forbidden_paths
        ),
        "excluded_sha256": canonical_sha256(expected_excluded),
    }
    if top["integrity"] != expected_integrity:
        raise ValueError("exclusion inventory integrity digests differ")
    inventory["frozen_text_derived_source_rows"] = set(frozen_text_derived)
    inventory["frozen_excluded_source_rows"] = set(frozen_source_rows)
    return dict(artifact), inventory


def _build_assets_from_inventory(
    source_rows: list[dict[str, Any]],
    inventory: Mapping[str, Any],
    inventory_bytes: bytes,
) -> dict[str, bytes]:

    text_derived_source_rows = _text_derived_source_rows(
        source_rows, inventory["normalized_prompts"]
    )
    excluded_source_rows = set(inventory["source_rows"]) | text_derived_source_rows
    if (
        text_derived_source_rows
        != set(inventory["frozen_text_derived_source_rows"])
        or excluded_source_rows != set(inventory["frozen_excluded_source_rows"])
    ):
        raise ValueError(
            "frozen exclusion inventory source-row derivation differs from Parti"
        )
    selected_rows = select_engineering_prompts(
        source_rows,
        set(inventory["normalized_prompts"]),
        excluded_source_rows,
    )
    selected_seed = select_engineering_seed(set(inventory["used_reserved_seeds"]))
    csv_value = _csv_bytes(selected_rows, selected_seed["seed"])

    prompt_collisions = sorted(
        {
            normalize_prompt(str(row["prompt"]))
            for row in selected_rows
        }
        & set(inventory["normalized_prompts"])
    )
    source_row_collisions = sorted(
        {int(row["source_row"]) for row in selected_rows} & excluded_source_rows
    )
    seed_collisions = sorted(
        {int(selected_seed["seed"])} & set(inventory["used_reserved_seeds"])
    )
    if prompt_collisions or source_row_collisions or seed_collisions:
        raise RuntimeError("adaptive-oracle selection contains an inventory collision")

    selected_manifest_rows = []
    for index, row in enumerate(selected_rows):
        normalized = normalize_prompt(str(row["prompt"]))
        selected_manifest_rows.append(
            {
                "prompt_row_id": f"engineering-{index + 1:04d}",
                "category": row["category"],
                "challenge": row["challenge"],
                "source_row": int(row["source_row"]),
                "rank_within_challenge": int(row["rank_within_challenge"]),
                "prompt_sha256": sha256_bytes(str(row["prompt"]).encode("utf-8")),
                "normalized_prompt_sha256": sha256_bytes(normalized.encode("utf-8")),
                "selection_digest": row["selection_digest"],
            }
        )
    normalized_prompt_digests = [
        row["normalized_prompt_sha256"] for row in selected_manifest_rows
    ]
    selection_digest = canonical_sha256(selected_manifest_rows)
    challenges = sorted({str(row["challenge"]) for row in source_rows})
    manifest = {
        "schema": "adaptive_oracle_prompt_manifest_v1",
        "status": "registration_only_gpu_not_authorized",
        "experiment_id": "adaptive_oracle_engineering_v1",
        "selection_namespace": SELECTION_NAMESPACE,
        "normalization_rule": "Unicode NFKC, collapse whitespace, then casefold",
        "selection_rule": (
            "exclude every normalized prompt and explicit or text-derived source_row "
            "from repository prompt CSV/JSON metadata and historical output "
            "manifest/sidecar metadata; globally deduplicate the frozen source by "
            "normalized text using the lowest source_row; within each challenge sort "
            "SHA256(namespace:prompt:challenge:source_row:normalized_prompt) and take "
            "rank zero"
        ),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "file": "PartiPrompts.tsv",
            "file_sha256": SOURCE_SHA256,
            "revision_verified_by_builder": True,
            "row_count": len(source_rows),
            "challenge_count": len(challenges),
            "challenges": challenges,
        },
        "exclusion_inventory": {
            "schema": EXCLUSION_INVENTORY_SCHEMA,
            "path": f"eval-pipeline/prompts/{OUTPUT_EXCLUSION_INVENTORY}",
            "sha256": sha256_bytes(inventory_bytes),
        },
        "inventory_policy": {
            "repository_prompt_scope": "eval-pipeline/prompts/**/*.csv|json|jsonl",
            "repository_seed_scope": "eval-pipeline/configs/**/*.{yaml,yml,json,jsonl} plus prompt manifests",
            "historical_output_scope": "outputs/** config/manifest/sidecar metadata",
            "allowed_projection": ["prompt", "source_row", "used_or_reserved_seed"],
            "parsed_identity_field_allowlist": list(_IDENTITY_FIELD_ALLOWLIST),
            "forbidden_field_fragments": list(_OUTCOME_KEY_FRAGMENTS),
            "forbidden_output_path_fragments": list(_OUTCOME_PATH_FRAGMENTS),
            "outcome_container_policy": (
                "traverse mappings only to project allowlisted identity fields; "
                "do not consume score, quality, reward, metric, or result scalars"
            ),
            "dedicated_score_or_quality_files_read": 0,
            "outcome_field_values_consumed": 0,
            "self_metadata_excluded_names": sorted(SELF_METADATA_NAMES),
            "self_output_excluded_prefixes": list(SELF_OUTPUT_PREFIXES),
        },
        "inventory": inventory["manifest"],
        "excluded_normalized_prompt_count": len(inventory["normalized_prompts"]),
        "excluded_normalized_prompt_digest": canonical_sha256(
            sorted(inventory["normalized_prompts"])
        ),
        "excluded_explicit_source_row_count": len(inventory["source_rows"]),
        "excluded_explicit_source_row_digest": canonical_sha256(
            sorted(inventory["source_rows"])
        ),
        "excluded_text_derived_source_row_count": len(text_derived_source_rows),
        "excluded_text_derived_source_row_digest": canonical_sha256(
            sorted(text_derived_source_rows)
        ),
        "excluded_source_row_count": len(excluded_source_rows),
        "excluded_source_row_digest": canonical_sha256(sorted(excluded_source_rows)),
        "engineering": {
            "prompt_count": len(selected_rows),
            "challenge_count": len(challenges),
            "counts_per_challenge": 1,
            "csv": f"eval-pipeline/prompts/{OUTPUT_CSV}",
            "csv_sha256": sha256_bytes(csv_value),
            "selection_digest": selection_digest,
            "normalized_prompt_digests": normalized_prompt_digests,
            "prompts": selected_manifest_rows,
        },
        "seed_registration": {
            "namespace": SEED_NAMESPACE,
            "selection_rule": (
                "for counter starting at zero compute SHA256(namespace:engineering:counter), "
                "map the first eight hex digits with & 0x7fffffff, and take the first "
                "positive value absent from the frozen used/reserved seed inventory"
            ),
            "used_reserved_seed_count": len(inventory["used_reserved_seeds"]),
            "used_reserved_seed_digest": canonical_sha256(
                sorted(inventory["used_reserved_seeds"])
            ),
            "engineering_seed": selected_seed,
            "global_seed_count": 1,
            "reuse_policy": "forbidden_after_the_first_generation_attempt",
        },
        "collisions": {
            "normalized_prompts": prompt_collisions,
            "source_rows": source_row_collisions,
            "used_or_reserved_seeds": seed_collisions,
        },
    }
    manifest_value = _json_asset_bytes(manifest)
    return {
        OUTPUT_CSV: csv_value,
        OUTPUT_MANIFEST: manifest_value,
        OUTPUT_EXCLUSION_INVENTORY: inventory_bytes,
    }


def build_assets(
    source_path: pathlib.Path,
    prompt_dir: pathlib.Path,
    source_repo: Optional[pathlib.Path] = None,
    repo_root: Optional[pathlib.Path] = None,
    outputs_root: Optional[pathlib.Path] = None,
) -> dict[str, bytes]:
    """Build prompt assets from the current repository exclusion inventory."""

    source_repo = source_repo or source_path.parent
    repo_root = (repo_root or pathlib.Path(__file__).resolve().parents[1]).resolve()
    prompt_dir = prompt_dir.resolve()
    outputs_root = (outputs_root or repo_root / "outputs").resolve()
    _verify_source_revision(source_repo)
    source_rows = _source_rows(source_path)
    live_inventory = scan_inventory(repo_root, prompt_dir, outputs_root)
    inventory_bytes = _json_asset_bytes(
        _exclusion_inventory_value(live_inventory, source_rows)
    )
    _artifact, frozen_inventory = _validated_frozen_inventory(inventory_bytes)
    return _build_assets_from_inventory(
        source_rows, frozen_inventory, inventory_bytes
    )


def build_assets_from_frozen_inventory(
    source_path: pathlib.Path,
    inventory_path: pathlib.Path,
    source_repo: Optional[pathlib.Path] = None,
) -> dict[str, bytes]:
    """Rebuild prompt assets without repository configs or historical outputs."""

    source_repo = source_repo or source_path.parent
    _verify_source_revision(source_repo)
    source_rows = _source_rows(source_path)
    inventory_bytes = _stable_regular_bytes(
        inventory_path, label="frozen exclusion inventory"
    )
    _artifact, inventory = _validated_frozen_inventory(inventory_bytes)
    return _build_assets_from_inventory(source_rows, inventory, inventory_bytes)


def asset_mismatches(assets: Mapping[str, bytes], prompt_dir: pathlib.Path) -> list[str]:
    return sorted(
        name
        for name, expected in assets.items()
        if not (prompt_dir / name).is_file()
        or (prompt_dir / name).read_bytes() != expected
    )


def write_assets_once(assets: Mapping[str, bytes], prompt_dir: pathlib.Path) -> None:
    """Atomically create a complete asset set without replacing existing files."""

    prompt_dir.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(prompt_dir, os.O_RDONLY)
    temporary_paths: list[pathlib.Path] = []
    created_paths: list[pathlib.Path] = []
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        existing = sorted(name for name in assets if (prompt_dir / name).exists())
        if existing:
            raise FileExistsError(
                "adaptive-oracle assets already exist: " + ", ".join(existing)
            )
        for name, value in sorted(assets.items()):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{name}.", suffix=".tmp", dir=prompt_dir
            )
            temporary_path = pathlib.Path(temporary_name)
            temporary_paths.append(temporary_path)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        for (name, _value), temporary_path in zip(
            sorted(assets.items()), temporary_paths
        ):
            destination = prompt_dir / name
            os.link(temporary_path, destination)
            created_paths.append(destination)
        os.fsync(directory_fd)
    except BaseException:
        for path in reversed(created_paths):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/tmp/parti/PartiPrompts.tsv")
    parser.add_argument("--source-repo", default="/tmp/parti")
    parser.add_argument(
        "--repo-root", default=str(pathlib.Path(__file__).resolve().parents[1])
    )
    parser.add_argument(
        "--prompt-dir",
        default=str(pathlib.Path(__file__).resolve().parent / "prompts"),
    )
    parser.add_argument("--outputs-root", default=None)
    parser.add_argument(
        "--frozen-inventory",
        help="replay from a canonical exclusion inventory without live outputs",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write and args.frozen_inventory:
        parser.error("--write cannot be combined with --frozen-inventory")

    repo_root = pathlib.Path(args.repo_root)
    prompt_dir = pathlib.Path(args.prompt_dir)
    outputs_root = (
        pathlib.Path(args.outputs_root) if args.outputs_root else repo_root / "outputs"
    )
    if args.frozen_inventory:
        assets = build_assets_from_frozen_inventory(
            pathlib.Path(args.source),
            pathlib.Path(args.frozen_inventory),
            pathlib.Path(args.source_repo),
        )
    else:
        assets = build_assets(
            pathlib.Path(args.source),
            prompt_dir,
            pathlib.Path(args.source_repo),
            repo_root,
            outputs_root,
        )
    if args.write:
        write_assets_once(assets, prompt_dir)
        mismatches = []
    else:
        mismatches = asset_mismatches(assets, prompt_dir)
    if mismatches:
        raise SystemExit("adaptive-oracle assets differ: " + ", ".join(mismatches))
    print(
        json.dumps(
            {
                "mode": "write" if args.write else "check",
                "assets": sorted(assets),
                "status": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
