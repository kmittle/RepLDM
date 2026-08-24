"""Create a fixed, action-blinded LR-1 validation review package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _font(size: int, bold: bool = False):
    names = ("DejaVuSans-Bold.ttf",) if bold else ("DejaVuSans.ttf",)
    for directory in ("/usr/share/fonts/truetype/dejavu",):
        for name in names:
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _left_action(pair_id: str, blinding_seed: int) -> bool:
    digest = hashlib.sha256(
        f"latent-renderer-blind-v1:{blinding_seed}:{pair_id}".encode("utf-8")
    ).digest()
    return bool(digest[0] & 1)


def _pair_id(prompt_index: int, seed: int) -> str:
    return f"pair_p{prompt_index}_s{seed}"


def build_blind_package(
    run_dir: str | os.PathLike[str],
    prompts_path: str | os.PathLike[str],
    frozen_actions_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    with open(frozen_actions_path) as handle:
        frozen = yaml.safe_load(handle) or {}
    requirements = frozen.get("validation_requirements", {})
    montage_spec = requirements.get("qualitative_montage", {}) or {}
    selected = str(frozen.get("selected_action", ""))
    actions = [str(value) for value in montage_spec.get("actions", [])]
    actions = [selected if value == "selected_action" else value for value in actions]
    if not selected or actions != ["no_ag", selected] or len(set(actions)) != 2:
        raise ValueError("frozen montage must contain exactly no_ag and selected_action")
    prompt_indices = [int(value) for value in montage_spec.get("prompt_indices", [])]
    if not prompt_indices or len(prompt_indices) != len(set(prompt_indices)):
        raise ValueError("frozen montage must contain unique prompt indices")
    seed = int(montage_spec.get("seed", -1))
    blinding_seed = int(montage_spec.get("blinding_seed", -1))
    if seed < 0 or blinding_seed < 0:
        raise ValueError("frozen montage seed and blinding_seed must be non-negative")

    prompts = pd.read_csv(prompts_path)
    if not {"index", "TEXT"}.issubset(prompts.columns):
        raise ValueError("prompt CSV must contain index and TEXT")
    prompt_rows = {
        int(row["index"]): row for _, row in prompts.iterrows()
    }
    missing_prompts = sorted(set(prompt_indices) - set(prompt_rows))
    if missing_prompts:
        raise ValueError(f"montage prompts are missing from CSV: {missing_prompts}")

    records = load_jsonl(run_dir / "manifest.jsonl")
    by_key: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for record in records:
        key = (
            int(record.get("prompt_index", -1)),
            int(record.get("seed", -1)),
            str(record.get("action_id", "")),
        )
        if key in by_key:
            raise ValueError(f"duplicate validation record {key}")
        by_key[key] = record

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_size = 256
    pairs: List[Dict[str, Any]] = []
    for prompt_index in prompt_indices:
        pair_id = _pair_id(prompt_index, seed)
        records_for_pair = []
        for action in actions:
            record = by_key.get((prompt_index, seed, action))
            if record is None:
                raise ValueError(f"missing validation image for {pair_id}, action {action}")
            image_path = (run_dir / str(record.get("image_path", ""))).resolve()
            if not image_path.is_file():
                raise ValueError(f"missing validation image: {image_path}")
            with Image.open(image_path) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError(f"{image_path}: expected RGB PNG")
                image.load()
                records_for_pair.append((action, image.copy()))

        left_selected = _left_action(pair_id, blinding_seed)
        if left_selected:
            left, right = records_for_pair[1], records_for_pair[0]
        else:
            left, right = records_for_pair[0], records_for_pair[1]
        canvas = Image.new("RGB", (tile_size * 2, tile_size + 34), "white")
        draw = ImageDraw.Draw(canvas)
        canvas.paste(left[1].resize((tile_size, tile_size)), (0, 34))
        canvas.paste(right[1].resize((tile_size, tile_size)), (tile_size, 34))
        label_font = _font(20, bold=True)
        draw.text((tile_size // 2 - 10, 7), "A", fill="black", font=label_font)
        draw.text((tile_size + tile_size // 2 - 10, 7), "B", fill="black", font=label_font)
        pair_path = output_dir / f"{pair_id}.png"
        canvas.save(pair_path)
        pairs.append(
            {
                "pair_id": pair_id,
                "prompt_index": prompt_index,
                "seed": seed,
                "prompt": str(prompt_rows[prompt_index]["TEXT"]),
                "challenge": str(prompt_rows[prompt_index].get("source_challenge", "")),
                "image_path": pair_path.name,
            }
        )

    columns = 4
    rows = (len(pairs) + columns - 1) // columns
    gap = 12
    header = 28
    montage = Image.new(
        "RGB",
        (columns * tile_size * 2 + (columns + 1) * gap, rows * (tile_size + 34) + (rows + 1) * gap + header),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    draw.text((gap, 5), "Blinded validation pairs", fill="black", font=_font(18, bold=True))
    for index, pair in enumerate(pairs):
        with Image.open(output_dir / pair["image_path"]) as image:
            x = gap + (index % columns) * (tile_size * 2 + gap)
            y = header + gap + (index // columns) * (tile_size + 34 + gap)
            montage.paste(image, (x, y))
            draw.text((x, y + tile_size + 35), pair["pair_id"], fill=(60, 60, 60), font=_font(12))
    montage.save(output_dir / "montage.png")

    with (output_dir / "review_prompts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "prompt_index",
                "seed",
                "challenge",
                "prompt",
                "image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(pairs)
    with (output_dir / "review_form_template.csv").open("w", newline="") as handle:
        fieldnames = ["reviewer_id", "pair_id", "overall", "structure", "text", "counting", "position", "detail"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "reviewer_id": "",
                    "pair_id": pair["pair_id"],
                    "overall": "",
                    "structure": "",
                    "text": "",
                    "counting": "",
                    "position": "",
                    "detail": "",
                }
            )

    # Keep the mapping separate from the shareable review images and prompts.
    key = {
        "schema": "latent_renderer_blind_key_v1",
        "selected_action": selected,
        "baseline_action": "no_ag",
        "blinding_seed": blinding_seed,
        "pairs": [
            {
                "pair_id": pair["pair_id"],
                "left_action": selected if _left_action(pair["pair_id"], blinding_seed) else "no_ag",
                "right_action": "no_ag" if _left_action(pair["pair_id"], blinding_seed) else selected,
            }
            for pair in pairs
        ],
        "provenance": {
            "run_manifest_sha256": sha256_file(run_dir / "manifest.jsonl"),
            "frozen_actions_sha256": sha256_file(frozen_actions_path),
            "prompts_sha256": sha256_file(prompts_path),
        },
    }
    (output_dir / "review_key.json").write_text(json.dumps(key, indent=2) + "\n")
    return {
        "schema": "latent_renderer_blind_package_v1",
        "pairs": len(pairs),
        "seed": seed,
        "selected_action": selected,
        "shareable_files": ["montage.png", "review_prompts.csv", "review_form_template.csv"]
        + [pair["image_path"] for pair in pairs],
        "private_file": "review_key.json",
        "provenance": key["provenance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--frozen_actions", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    result = build_blind_package(
        args.run_dir, args.prompts, args.frozen_actions, args.output_dir
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
