#!/usr/bin/env python3
"""Revalidate a selected-view release with the fixed production runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# ``python -I`` intentionally removes the script directory from the import
# path.  Add this repository-local directory explicitly so the isolated
# revalidator can import the pinned ``data_catalog`` package without accepting
# caller-controlled ``PYTHONPATH`` entries.
_EVAL_PIPELINE_ROOT = Path(__file__).absolute().parent
if str(_EVAL_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_PIPELINE_ROOT))

from data_catalog.selected_runtime import revalidate_release_v1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = revalidate_release_v1(
            args.release_dir.resolve(), repository_root=args.repository_root.resolve()
        )
    except Exception as exc:
        print(f"selected-view runtime revalidation failed: {type(exc).__name__}: {exc}")
        return 1
    print(
        json.dumps(
            {
                "release_id": manifest.get("release_id"),
                "runtime_reverified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
