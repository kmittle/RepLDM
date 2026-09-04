#!/usr/bin/env python3
"""Build local CLIP indexes and calibration artifacts for selected-view v1."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

# Keep the script runnable from the repository root without requiring an
# editable install of the hyphenated ``eval-pipeline`` directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVAL_PIPELINE_ROOT = REPOSITORY_ROOT / "eval-pipeline"
if str(EVAL_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE_ROOT))

from data_catalog.selected_assets import build_selected_assets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root-2", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    config = build_selected_assets(
        parent_release=args.parent_release,
        output_dir=args.output_dir,
        config_output=args.config_output,
        clip_checkpoint=args.clip_checkpoint,
        tokenizer_root=args.tokenizer_root,
        tokenizer_root_2=args.tokenizer_root_2,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "schema": "repldm.selected_view_assets_build.v1",
                "config": str(args.config_output.resolve()),
                "protected_index": config["protected_index"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
