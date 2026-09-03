"""Build auditable TOPIQ-NR runtime config and file manifest.

This utility performs no model loading and never downloads assets.  It is
intended for preparing a local, ordinary-file staging directory before a
formal F0 run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PREPROCESS_SHA256 = "4be16e500c78ebddd1fd75f8bd1385a84e301d5628598a084bababe0802535f4"


def _ordinary_file(raw: str, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    path = Path(path)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} cannot contain symlinks: {current}")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an ordinary regular file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} must be non-empty: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(root_raw: str, package: str) -> tuple[Path, list[str]]:
    root = Path(root_raw)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError(f"{package} source root must be an ordinary directory")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{package} source root cannot contain symlinks: {current}")
    files: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{package} source file is not ordinary: {path}")
        files.append(relative.as_posix())
    if "__init__.py" not in files:
        raise ValueError(f"{package} source closure omits __init__.py")
    return root, files


def _versions(python: str) -> dict[str, str]:
    code = (
        "import importlib.metadata as m; "
        "names=('pyiqa','timm','torch','torchvision','safetensors'); "
        "print('\\n'.join(n+'\\t'+m.version(n) for n in names))"
    )
    output = subprocess.check_output([python, "-c", code], text=True)
    result: dict[str, str] = {}
    for line in output.splitlines():
        name, separator, version = line.partition("\t")
        if not separator or not version:
            raise ValueError(f"version probe returned malformed output: {line!r}")
        result[name] = version
    expected = {"pyiqa", "timm", "torch", "torchvision", "safetensors"}
    if set(result) != expected:
        raise ValueError(f"version probe did not return {sorted(expected)}")
    return result


def _binding(label: str, path: Path) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def build(args: argparse.Namespace) -> None:
    checkpoint = _ordinary_file(args.checkpoint, "checkpoint")
    backbone = _ordinary_file(args.backbone, "backbone")
    if checkpoint == backbone:
        raise ValueError("checkpoint and backbone must be distinct files")
    pyiqa_root, pyiqa_files = _source_files(args.pyiqa_root, "pyiqa")
    timm_root, timm_files = _source_files(args.timm_root, "timm")
    versions = _versions(args.python)

    source_paths = [pyiqa_root / name for name in pyiqa_files]
    source_paths.extend(timm_root / name for name in timm_files)
    all_paths = [checkpoint, backbone, *source_paths]
    if len({str(path) for path in all_paths}) != len(all_paths):
        raise ValueError("witness assets contain duplicate paths")
    rows = [_binding(f"witness_assets_manifest[{i}]", path) for i, path in enumerate(all_paths)]
    manifest = {
        "schema": "repldm.file_manifest.v1",
        "files": [{key: row[key] for key in ("path", "sha256", "bytes")} for row in rows],
    }
    config = {
        "schema": "repldm.topiq_nr_tensor_witness_runtime.v1",
        "implementation": "pyiqa:topiq_nr",
        "role": "independent_f0_witness",
        "dtype": "float32",
        "preprocess_sha256": PREPROCESS_SHA256,
        "checkpoint": str(checkpoint),
        "backbone": str(backbone),
        "package_versions": versions,
        "source_packages": {
            "pyiqa": {"root": str(pyiqa_root), "files": pyiqa_files},
            "timm": {"root": str(timm_root), "files": timm_files},
        },
    }
    if not args.force:
        existing = [path for path in (args.output_config, args.output_manifest) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing output(s): "
                + ", ".join(str(path) for path in existing)
            )
    _write_json(args.output_config, config, force=args.force)
    _write_json(args.output_manifest, manifest, force=args.force)
    print(json.dumps({"config": str(args.output_config), "manifest": str(args.output_manifest),
                      "asset_count": len(rows), "package_versions": versions}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="ordinary CFANet TOPIQ-NR .pth")
    parser.add_argument("--backbone", required=True, help="ordinary timm ResNet-50 safetensors")
    parser.add_argument("--pyiqa-root", required=True)
    parser.add_argument("--timm-root", required=True)
    parser.add_argument("--python", default="python3", help="interpreter providing pinned packages")
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="allow replacing existing outputs")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
