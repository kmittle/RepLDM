#!/usr/bin/env python3
"""Encode one deterministic shard of the selected-view protected images."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_catalog.image_index_shards import build_image_index_shard


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=8)
    args = parser.parse_args(argv)
    manifest = build_image_index_shard(
        parent_release=args.parent_release,
        output_dir=args.output_dir,
        clip_checkpoint=args.clip_checkpoint,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        device=args.device,
        batch_size=args.batch_size,
        decode_workers=args.decode_workers,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
