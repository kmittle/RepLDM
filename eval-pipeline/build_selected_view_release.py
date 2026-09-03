#!/usr/bin/env python3
"""Build or runtime-reverify a content-addressed selected-view release."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path

from data_catalog.builder import REPOSITORY_ROOT
from data_catalog.selected import validate_selected_view_release
from data_catalog.selected_builder import (
    RuntimeFactory,
    build_selected_view_release,
    create_selected_view_runtime,
    verify_selected_view_runtime,
)


def _runtime_factory(value: str) -> RuntimeFactory:
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("runtime factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("selected-view runtime factory is not callable")
    return factory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root", type=Path, default=REPOSITORY_ROOT
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build and immediately reverify a child release")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--parent-release", type=Path, required=True)
    build.add_argument("--output-root", type=Path)
    build.add_argument(
        "--runtime-factory",
        help="local model/index factory as module.path:callable; omission fails closed",
    )
    build.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development only; the resulting release cannot authorize training",
    )

    verify = commands.add_parser(
        "verify", help="re-run every learned and indexed gate for an installed payload"
    )
    verify.add_argument("--release-dir", type=Path, required=True)
    verify.add_argument("--runtime-factory", required=True)
    verify.add_argument("--allow-development", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    if args.command == "build":
        factory = (
            _runtime_factory(args.runtime_factory) if args.runtime_factory else None
        )
        release = build_selected_view_release(
            config_path=args.config.resolve(),
            parent_release=args.parent_release.resolve(),
            output_root=(args.output_root.resolve() if args.output_root else None),
            runtime_factory=factory,
            repository_root=repository_root,
            allow_dirty=args.allow_dirty,
        )
        manifest = validate_selected_view_release(
            release,
            repository_root=repository_root,
            require_formal=not args.allow_dirty,
            require_training_ready=False,
        )
        print(
            json.dumps(
                {
                    "release_dir": str(release),
                    "release_id": manifest["release_id"],
                    "training_ready": manifest["training_ready"],
                    "runtime_reverified": "selected_payload" in manifest and factory is not None,
                },
                sort_keys=True,
            )
        )
        return 0

    release = args.release_dir.resolve()
    manifest = validate_selected_view_release(
        release,
        repository_root=repository_root,
        require_formal=not args.allow_development,
        require_training_ready=False,
    )
    if "selected_payload" not in manifest:
        raise ValueError("selected-view release has no payload to runtime-reverify")
    config_path = release / manifest["config"]["path"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_dir = Path(manifest["parent_catalog"]["path"]).parent
    factory = _runtime_factory(args.runtime_factory)
    runtime = create_selected_view_runtime(
        factory,
        config=config,
        parent_dir=parent_dir,
        repository_root=repository_root,
    )
    verify_selected_view_runtime(
        release,
        runtime=runtime,
        repository_root=repository_root,
        require_formal=not args.allow_development,
    )
    print(
        json.dumps(
            {
                "release_dir": str(release),
                "release_id": manifest["release_id"],
                "runtime_reverified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
