#!/usr/bin/env python3
"""Validate one formal latent-renderer selected-view child release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_catalog.selected import validate_selected_view_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-non-training-ready",
        action="store_true",
        help="validate a fail-closed build report without authorizing training",
    )
    args = parser.parse_args()
    manifest = validate_selected_view_release(
        args.release_dir.resolve(),
        require_training_ready=not args.allow_non_training_ready,
        require_gate_report=not args.allow_non_training_ready,
    )
    print(
        json.dumps(
            {
                "validated": str(args.release_dir.resolve()),
                "release_id": manifest["release_id"],
                "training_ready": manifest["training_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
