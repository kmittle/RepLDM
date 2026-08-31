"""Freeze the official 3,200-row HPSv2 benchmark prompt manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = ROOT / "eval-pipeline/prompts/hpsv2_official_3200.csv"
DEFAULT_OUTPUT_MANIFEST = (
    ROOT / "eval-pipeline/prompts/hpsv2_official_3200_manifest.json"
)
SOURCE_REPOSITORY = "zhwang/HPDv2"
SOURCE_REVISION = "b9517430f34b1080d4f741118d2a440155a165cd"
STYLE_ORDER = ("anime", "concept-art", "paintings", "photo")
EXPECTED_SOURCE = {
    "anime": {
        "size_bytes": 62816,
        "sha256": "6b38a479cc6f899866e9b7a8bf91f096b4afc6883b95275a61891327182b14ae",
        "git_blob_oid": "b042bd6096b1337f4a7eef8c9b1233f71910f9c7",
    },
    "concept-art": {
        "size_bytes": 85787,
        "sha256": "530d057deafcb82c581e9d1be2ffcb98828fe99f0b54257414d2ab8a5cf2b83e",
        "git_blob_oid": "62f8ba6d263251840631eda54010ad2f91b1d5de",
    },
    "paintings": {
        "size_bytes": 81338,
        "sha256": "6570d39467e08b24b64e5b8a4177b0f61c525631121dc5a91fe2ab956eab4d5f",
        "git_blob_oid": "303e19b56e4c1c336d396a9cf252cade5baaaaf9",
    },
    "photo": {
        "size_bytes": 44606,
        "sha256": "92a3c9a0f743f66eb87bd112cdda830d6bdb4c073c86021d53b74001fcb18052",
        "git_blob_oid": "a114fe4a4738353438c5f95dc3e23b85882a1ca4",
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_source(source_dir: Path, style: str) -> tuple[list[str], dict]:
    path = source_dir / f"{style}.json"
    raw = path.read_bytes()
    expected = EXPECTED_SOURCE[style]
    if len(raw) != expected["size_bytes"]:
        raise ValueError(f"{path} has the wrong byte count")
    if sha256_bytes(raw) != expected["sha256"]:
        raise ValueError(f"{path} differs from the pinned HPDv2 source")
    prompts = json.loads(raw)
    if not isinstance(prompts, list) or len(prompts) != 800:
        raise ValueError(f"{path} must contain exactly 800 prompts")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError(f"{path} contains an empty or non-string prompt")
    return prompts, {
        "path": f"benchmark/{style}.json",
        "size_bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "git_blob_oid": expected["git_blob_oid"],
    }


def build(source_dir: Path, output_csv: Path, output_manifest: Path) -> None:
    rows = []
    sources = {}
    for style in STYLE_ORDER:
        prompts, source_record = load_source(source_dir, style)
        sources[style] = source_record
        for style_index, prompt in enumerate(prompts):
            rows.append(
                {
                    "index": len(rows),
                    "TEXT": prompt,
                    "bucket": style,
                    "benchmark_style": style,
                    "style_index": style_index,
                    "source_row_id": f"{style}:{style_index:04d}",
                }
            )
    if len(rows) != 3200:
        raise RuntimeError("the official HPSv2 benchmark must contain 3,200 rows")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "index",
                "TEXT",
                "bucket",
                "benchmark_style",
                "style_index",
                "source_row_id",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    csv_bytes = output_csv.read_bytes()
    unique_prompts = len({row["TEXT"] for row in rows})
    manifest = {
        "schema": "hpsv2_official_prompt_manifest_v1",
        "benchmark": "HPSv2",
        "official_prompt_count": len(rows),
        "exact_unique_prompt_count": unique_prompts,
        "official_duplicate_row_count": len(rows) - unique_prompts,
        "duplicate_policy": "preserve_official_rows_and_identify_by_style_and_style_index",
        "style_order": list(STYLE_ORDER),
        "style_counts": {style: 800 for style in STYLE_ORDER},
        "source": {
            "repository": SOURCE_REPOSITORY,
            "repository_type": "huggingface_dataset",
            "revision": SOURCE_REVISION,
            "canonical_url": f"https://huggingface.co/datasets/{SOURCE_REPOSITORY}",
            "files": sources,
        },
        "csv": {
            "path": relative_to_root(output_csv),
            "size_bytes": len(csv_bytes),
            "sha256": sha256_bytes(csv_bytes),
            "encoding": "utf-8",
            "line_ending": "LF",
            "columns": list(rows[0]),
            "index_range": [0, 3199],
        },
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST
    )
    args = parser.parse_args()
    build(args.source_dir, args.output_csv, args.output_manifest)


if __name__ == "__main__":
    main()
