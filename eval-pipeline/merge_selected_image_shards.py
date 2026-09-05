#!/usr/bin/env python3
"""Validate and merge selected-view protected-image index shards."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_catalog.image_index_shards import merge_image_index_shards


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-release", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=None)
    args = parser.parse_args(argv)
    manifest = merge_image_index_shards(
        parent_release=args.parent_release,
        shard_dir=args.shard_dir,
        output_dir=args.output_dir,
        clip_checkpoint=args.clip_checkpoint,
        shard_count=args.shard_count,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
