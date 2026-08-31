"""Build the registration-only scheduler-native fixed-headroom prompt splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pathlib
import subprocess
import unicodedata
from collections import defaultdict
from typing import Any, Iterable, Mapping, Optional


SOURCE_REVISION = "5a657978134374ce28973948331b319adef164bd"
SOURCE_SHA256 = "fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb"
SELECTION_NAMESPACE = "repldm-scheduler-native-fixed-headroom-v1"
SEED_NAMESPACE = f"{SELECTION_NAMESPACE}-seeds"
SPLIT_COUNTS_PER_CHALLENGE = {
    "smoke": 1,
    "development": 3,
    "validation": 2,
    "final": 2,
}
SEED_COUNTS = {"smoke": 1, "development": 3, "validation": 3, "final": 3}
OBSERVED_GENERATED_SEEDS = (0, 7, 19, 42, 73, 123)
PRIOR_REGISTERED_SEEDS = (11, 29, 101)
OUTPUT_PREFIX = "scheduler_native_fixed_headroom_"
PRIOR_PROMPT_CSV_NAMES = (
    "eval_v1.csv",
    "latent_renderer_test.csv",
    "latent_renderer_train.csv",
    "latent_renderer_validation.csv",
    "s5_development.csv",
    "s5_smoke.csv",
    "smoke.csv",
    "stage2_smoke.csv",
    "trajectory_correction_heldout_v1.csv",
    "trajectory_correction_validation_v1.csv",
)
CSV_FIELDS = (
    "index",
    "TEXT",
    "bucket",
    "source_row",
    "source_category",
    "source_challenge",
    "source_note",
    "split",
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


def is_prior_prompt_csv(path: pathlib.Path) -> bool:
    """Return whether a CSV belongs to the fixed-headroom exclusion universe."""

    return path.name in PRIOR_PROMPT_CSV_NAMES


def _read_csv(path: pathlib.Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _source_rows(source_path: pathlib.Path) -> list[dict[str, Any]]:
    source_bytes = source_path.read_bytes()
    if sha256_bytes(source_bytes) != SOURCE_SHA256:
        raise ValueError("PartiPrompts.tsv differs from the frozen source hash")
    rows = _read_csv(source_path, delimiter="\t")
    required = {"Prompt", "Category", "Challenge", "Note"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("PartiPrompts.tsv has an unexpected schema")
    return [
        {
            "source_row": index,
            "prompt": row["Prompt"].strip(),
            "category": row["Category"].strip(),
            "challenge": row["Challenge"].strip(),
            "note": row["Note"].strip(),
        }
        for index, row in enumerate(rows)
    ]


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


def _existing_prompt_inventory(
    prompt_dir: pathlib.Path, source_rows: Iterable[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], set[str], set[int]]:
    source_by_text: dict[str, set[int]] = defaultdict(set)
    for row in source_rows:
        source_by_text[normalize_prompt(str(row["prompt"]))].add(int(row["source_row"]))

    file_inventory = []
    excluded_texts: set[str] = set()
    explicit_source_rows: set[int] = set()
    for name in PRIOR_PROMPT_CSV_NAMES:
        path = prompt_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"registered historical prompt CSV is missing or unsafe: {path}")
        rows = _read_csv(path)
        if rows and "TEXT" not in rows[0]:
            raise ValueError(f"{path} lacks a TEXT column")
        local_texts = {normalize_prompt(row["TEXT"]) for row in rows}
        local_source_rows = {
            int(row["source_row"])
            for row in rows
            if str(row.get("source_row", "")).strip()
        }
        excluded_texts.update(local_texts)
        explicit_source_rows.update(local_source_rows)
        file_inventory.append(
            {
                "path": f"eval-pipeline/prompts/{path.name}",
                "sha256": sha256_bytes(path.read_bytes()),
                "row_count": len(rows),
                "unique_normalized_text_count": len(local_texts),
                "explicit_source_row_count": len(local_source_rows),
            }
        )

    derived_source_rows = {
        source_row
        for prompt in excluded_texts
        for source_row in source_by_text.get(prompt, set())
    }
    return (
        file_inventory,
        excluded_texts,
        explicit_source_rows | derived_source_rows,
    )


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


def _select_splits(
    source_rows: Iterable[Mapping[str, Any]],
    excluded_texts: set[str],
    excluded_source_rows: set[int],
) -> dict[str, list[dict[str, Any]]]:
    canonical_by_text: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        normalized = normalize_prompt(str(row["prompt"]))
        source_row = int(row["source_row"])
        if normalized in excluded_texts or source_row in excluded_source_rows:
            continue
        previous = canonical_by_text.get(normalized)
        if previous is None or source_row < int(previous["source_row"]):
            canonical_by_text[normalized] = row

    by_challenge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_by_text.values():
        selected = dict(row)
        selected["selection_digest"] = _selection_digest(row)
        by_challenge[str(row["challenge"])].append(selected)

    required_per_challenge = sum(SPLIT_COUNTS_PER_CHALLENGE.values())
    if len(by_challenge) != 11:
        raise ValueError(f"expected 11 Parti challenges, found {len(by_challenge)}")
    split_rows = {split: [] for split in SPLIT_COUNTS_PER_CHALLENGE}
    for challenge in sorted(by_challenge):
        ranked = sorted(
            by_challenge[challenge],
            key=lambda row: (row["selection_digest"], int(row["source_row"])),
        )
        if len(ranked) < required_per_challenge:
            raise ValueError(f"challenge {challenge!r} lacks enough unused prompts")
        offset = 0
        for split, count in SPLIT_COUNTS_PER_CHALLENGE.items():
            for rank, row in enumerate(ranked[offset : offset + count], start=offset):
                selected = dict(row)
                selected["rank_within_challenge"] = rank
                split_rows[split].append(selected)
            offset += count
    return split_rows


def _seed_registration() -> dict[str, Any]:
    retired = set(OBSERVED_GENERATED_SEEDS) | set(PRIOR_REGISTERED_SEEDS)
    used = set(retired)
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, count in SEED_COUNTS.items():
        selected = []
        counter = 0
        while len(selected) < count:
            payload = f"{SEED_NAMESPACE}:{split}:{counter}"
            digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
            seed = int(digest[:8], 16) & 0x7FFFFFFF
            if seed not in used:
                used.add(seed)
                selected.append(
                    {"seed": seed, "counter": counter, "selection_digest": digest}
                )
            counter += 1
        splits[split] = selected
    return {
        "namespace": SEED_NAMESPACE,
        "selection_rule": (
            "for each split and counter starting at zero, compute SHA256("
            "namespace:split:counter), map the first 8 hex digits with "
            "& 0x7fffffff, and take the first non-retired globally unique seeds"
        ),
        "observed_generated_seeds": list(OBSERVED_GENERATED_SEEDS),
        "observed_seed_scan": (
            "outputs/**/config.json top-level seeds, manifest.jsonl seed fields, "
            "and images/*.json seed fields, audited 2026-08-25"
        ),
        "prior_registered_or_reserved_seeds": list(PRIOR_REGISTERED_SEEDS),
        "retired_seeds": sorted(retired),
        "retired_seed_inventory_sha256": canonical_sha256(sorted(retired)),
        "splits": splits,
    }


def _bucket(challenge: str) -> str:
    return (
        challenge.casefold()
        .replace(" & ", "-and-")
        .replace(" ", "-")
    )


def _csv_bytes(split: str, rows: Iterable[Mapping[str, Any]]) -> bytes:
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
                "split": split,
            }
        )
    return buffer.getvalue().encode("utf-8")


def build_assets(
    source_path: pathlib.Path,
    prompt_dir: pathlib.Path,
    source_repo: Optional[pathlib.Path] = None,
) -> dict[str, bytes]:
    source_repo = source_repo or source_path.parent
    _verify_source_revision(source_repo)
    source_rows = _source_rows(source_path)
    inventory, excluded_texts, excluded_source_rows = _existing_prompt_inventory(
        prompt_dir, source_rows
    )
    splits = _select_splits(source_rows, excluded_texts, excluded_source_rows)
    assets = {
        f"{OUTPUT_PREFIX}{split}.csv": _csv_bytes(split, rows)
        for split, rows in splits.items()
    }
    challenges = sorted({str(row["challenge"]) for row in source_rows})
    manifest_splits = {}
    for split, rows in splits.items():
        manifest_splits[split] = [
            {
                "category": row["category"],
                "challenge": row["challenge"],
                "prompt_sha256": sha256_bytes(str(row["prompt"]).encode("utf-8")),
                "rank_within_challenge": row["rank_within_challenge"],
                "selection_digest": row["selection_digest"],
                "source_row": row["source_row"],
            }
            for row in rows
        ]
    manifest = {
        "schema": "scheduler_native_fixed_headroom_prompt_manifest_v1",
        "status": "registration_only_not_authorized",
        "selection_namespace": SELECTION_NAMESPACE,
        "normalization_rule": "Unicode NFKC, collapse whitespace, then casefold",
        "selection_rule": (
            "exclude every normalized TEXT and explicit or text-derived source_row "
            "from the listed repository CSV snapshot; globally deduplicate source "
            "prompts by normalized text using the lowest source_row; within each "
            "Challenge sort SHA256(namespace:prompt:Challenge:source_row:normalized_"
            "prompt), then assign ranks 0 smoke, 1-3 development, 4-5 validation, "
            "and 6-7 final"
        ),
        "source": {
            "repository": "https://github.com/google-research/parti",
            "revision": SOURCE_REVISION,
            "file": "PartiPrompts.tsv",
            "file_sha256": SOURCE_SHA256,
            "revision_verified_by_builder": True,
            "row_count": len(source_rows),
            "challenge_count": len(challenges),
            "challenges": challenges,
        },
        "excluded_repository_prompt_files": inventory,
        "excluded_normalized_text_count": len(excluded_texts),
        "excluded_normalized_texts_sha256": canonical_sha256(sorted(excluded_texts)),
        "excluded_source_rows": sorted(excluded_source_rows),
        "excluded_source_rows_sha256": canonical_sha256(sorted(excluded_source_rows)),
        "counts_per_challenge": dict(SPLIT_COUNTS_PER_CHALLENGE),
        "counts": {split: len(rows) for split, rows in splits.items()},
        "csv_sha256": {name: sha256_bytes(value) for name, value in assets.items()},
        "splits": manifest_splits,
        "seed_registration": _seed_registration(),
    }
    assets[f"{OUTPUT_PREFIX}manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/tmp/parti/PartiPrompts.tsv")
    parser.add_argument("--source-repo", default="/tmp/parti")
    parser.add_argument(
        "--prompt-dir",
        default=str(pathlib.Path(__file__).resolve().parent / "prompts"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    prompt_dir = pathlib.Path(args.prompt_dir)
    assets = build_assets(
        pathlib.Path(args.source), prompt_dir, pathlib.Path(args.source_repo)
    )
    mismatches = []
    for name, expected in assets.items():
        path = prompt_dir / name
        if args.write:
            path.write_bytes(expected)
        elif not path.is_file() or path.read_bytes() != expected:
            mismatches.append(name)
    if mismatches:
        raise SystemExit("registration assets differ: " + ", ".join(mismatches))
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
