from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "eval-pipeline"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from data_catalog import image_index_shards
from data_catalog.image_index_shards import (
    _parent_rows,
    _validate_binding,
    _validate_embedding_matrix,
    bound_image_index_paths,
    shard_positions,
)
from data_catalog.selected_assets import (
    _candidate_image_parent_directories,
    _image_parent_directories,
)


def test_shard_positions_partition_global_order() -> None:
    shards = [shard_positions(11, index, 4) for index in range(4)]
    assert shards == [(0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7)]
    assert sorted(position for shard in shards for position in shard) == list(range(11))


def test_image_parent_directories_are_lexical_and_deduplicated(tmp_path: Path) -> None:
    first = tmp_path / "images" / "a.jpg"
    second = tmp_path / "images" / "sub" / "b.jpg"
    directories = _image_parent_directories(
        [
            {"image_path": str(first)},
            {"image_path": str(tmp_path / "images" / "." / "a.jpg")},
            {"image_path": str(second)},
            {"image_path": ""},
        ]
    )
    assert directories == (tmp_path / "images", tmp_path / "images" / "sub")


def test_candidate_image_parent_directories_stream_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "training_candidates.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps({"id": str(index), "image_path": str(tmp_path / "images" / f"{index}.jpg")})
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    assert _candidate_image_parent_directories(tmp_path) == (tmp_path / "images",)


@pytest.mark.parametrize(
    ("total", "index", "count"),
    [(1, -1, 2), (1, 2, 2), (1, 0, 0), (-1, 0, 1)],
)
def test_shard_positions_reject_invalid_partition(
    total: int, index: int, count: int
) -> None:
    with pytest.raises(ValueError):
        shard_positions(total, index, count)


@pytest.mark.parametrize(
    "values",
    [np.zeros((2, 3), dtype=np.float32), np.full((2, 3), 2.0, dtype=np.float32)],
)
def test_embedding_matrix_requires_nonzero_normalized_rows(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _validate_embedding_matrix(values, label="test")


def test_embedding_matrix_accepts_normalized_rows() -> None:
    values = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    assert _validate_embedding_matrix(values, label="test") == 2


def test_build_rejects_more_shards_than_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        image_index_shards,
        "_parent_rows",
        lambda *args, **kwargs: ({"release_id": "parent"}, "hash", [{"id": "image"}]),
    )
    clip = tmp_path / "clip.pt"
    clip.write_bytes(b"fixture checkpoint")
    with pytest.raises(ValueError, match="shard_count cannot exceed"):
        image_index_shards.build_image_index_shard(
            parent_release=Path("/tmp/parent"),
            output_dir=Path("/tmp/output"),
            clip_checkpoint=clip,
            shard_index=0,
            shard_count=2,
        )


def _binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_binding_rejects_noncanonical_absolute_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    binding = _binding(artifact)
    binding["path"] = str(tmp_path / "nested" / ".." / artifact.name)

    with pytest.raises(ValueError, match="not canonical"):
        _validate_binding(binding, label="artifact")


def test_bundle_paths_require_exact_top_level_schema(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": image_index_shards.IMAGE_BUNDLE_SCHEMA,
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported image index bundle schema"):
        bound_image_index_paths(bundle)


def test_bundle_paths_reject_symlink_bundle(tmp_path: Path) -> None:
    target = tmp_path / "bundle.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "bundle-link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        bound_image_index_paths(link)


def test_selected_output_rejects_symlink_parent(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        image_index_shards._preflight_selected_asset_destinations(
            (("artifact", link_dir / "artifact.bin"),), []
        )


def test_build_shard_rejects_source_directory_output_before_loading_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    checkpoint = tmp_path / "clip.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        image_index_shards,
        "_parent_rows",
        lambda *args, **kwargs: (
            {"release_id": parent.name},
            "a" * 64,
            [{"id": "image", "image_path": str(tmp_path / "image.png")}],
        ),
    )
    monkeypatch.setattr(
        image_index_shards,
        "_load_bound_clip",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CLIP must not load for an aliased output directory")
        ),
    )

    with pytest.raises(ValueError, match="overlaps an input directory"):
        image_index_shards.build_image_index_shard(
            parent_release=parent,
            output_dir=parent,
            clip_checkpoint=checkpoint,
            shard_index=0,
            shard_count=1,
        )


def test_merge_rejects_shard_directory_as_output(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    clip = tmp_path / "clip.pt"
    clip.write_bytes(b"fixture checkpoint")

    with pytest.raises(ValueError, match="overlaps an input directory"):
        image_index_shards.merge_image_index_shards(
            parent_release=tmp_path / "parent",
            shard_dir=shard_dir,
            output_dir=shard_dir,
            clip_checkpoint=clip,
        )


def _formal_parent_fixture(tmp_path: Path, *, holdout_rows: int = 49393) -> Path:
    parent = tmp_path / "formal-parent"
    parent.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    holdout = parent / "benchmark_holdouts.jsonl"
    row = {"id": "protected", "prompt": "protected prompt", "image_path": str(image)}
    holdout.write_bytes(
        b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for _ in range(holdout_rows))
    )
    digest = hashlib.sha256(holdout.read_bytes()).hexdigest()
    manifest = {
        "release_id": parent.name,
        "candidate_catalog_complete": True,
        "protected_normalized_unique_prompts": 1,
        "protected_unique_images": 1,
        "artifacts": [
            {
                "path": "benchmark_holdouts.jsonl",
                "bytes": holdout.stat().st_size,
                "sha256": digest,
                "rows": holdout_rows,
            }
        ],
    }
    (parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return parent


def test_parent_rows_requires_frozen_holdout_artifact_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _formal_parent_fixture(tmp_path)
    monkeypatch.setattr(
        image_index_shards,
        "_validate_parent_release",
        lambda path: Path(path).absolute(),
    )
    manifest_path = parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact bytes differ"):
        _parent_rows(parent, validate_files=False)


def test_parent_rows_rejects_incomplete_holdout_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _formal_parent_fixture(tmp_path, holdout_rows=1)
    monkeypatch.setattr(
        image_index_shards,
        "_validate_parent_release",
        lambda path: Path(path).absolute(),
    )

    with pytest.raises(ValueError, match="49,393"):
        _parent_rows(parent, validate_files=False)
