from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "eval-pipeline"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from data_catalog import selected_assets
import data_catalog.selected as selected_module
from data_catalog.selected import (
    SELECTED_IMAGE_MAX_PIXELS,
    _validate_calibration,
    _validate_protected_index_binding,
)
from data_catalog.selected_assets import (
    SOURCE_PROMPT_FIELDS,
    TOKENIZER_FILE_SHA256,
    _calibration_artifact,
    _assert_parent_artifact_snapshot,
    _capture_parent_artifact_snapshot,
    _cyclic_hamming_distances,
    _preflight_selected_asset_destinations,
    _selected_asset_destinations,
    _protected_sample_ids,
    _revalidate_parent_release_after_reads,
    _source_evidence_from_parent,
    _validate_source_prompt_fields,
    derive_protected_index_manifest,
    protected_ids_sha256,
    unique_protected_images,
    unique_protected_prompts,
)


def _fixture_decoder() -> dict[str, object]:
    from PIL import __version__ as pillow_version
    from PIL import features

    return {
        "library": "Pillow",
        "version": pillow_version,
        "littlecms_version": features.version("littlecms2"),
        "max_image_pixels": SELECTED_IMAGE_MAX_PIXELS,
        "exif_transpose": True,
        "icc_to_srgb": True,
        "output_mode": "RGB",
        "pixel_hash": "sha256_rgb_u64be_width_height_bytes_v1",
    }


def test_decode_image_payload_expands_grayscale_with_rgb_icc_profile(
    tmp_path: Path,
) -> None:
    from PIL import ImageCms

    image_path = tmp_path / "gray-rgb-profile.png"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("L", (2, 1), color=96).save(image_path, icc_profile=profile)

    decoded = selected_assets.decode_image_payload(image_path, _fixture_decoder())

    assert (decoded.width, decoded.height) == (2, 1)
    assert decoded.rgb_bytes == bytes((96, 96, 96, 96, 96, 96))


def test_decode_image_payload_temporarily_overrides_pillow_pixel_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "small-image.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(image_path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    decoded = selected_assets.decode_image_payload(image_path, _fixture_decoder())

    assert (decoded.width, decoded.height) == (2, 2)
    assert Image.MAX_IMAGE_PIXELS == 1


def test_decode_image_payload_rejects_oversized_header_before_exif_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import ImageOps

    image_path = tmp_path / "oversized-header.bin"
    image_path.write_bytes(b"fixture")

    class _Opened:
        size = (SELECTED_IMAGE_MAX_PIXELS + 1, 1)

        def __enter__(self) -> "_Opened":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(Image, "open", lambda _path: _Opened())

    def unexpected_exif(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("EXIF transpose must not load an oversized image")

    monkeypatch.setattr(ImageOps, "exif_transpose", unexpected_exif)
    with pytest.raises(ValueError, match="cannot decode selected image"):
        selected_assets.decode_image_payload(image_path, _fixture_decoder())


def test_decode_image_payload_rejects_incompatible_cmyk_rgb_profile(
    tmp_path: Path,
) -> None:
    from PIL import ImageCms

    image_path = tmp_path / "cmyk-rgb-profile.jpg"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("CMYK", (1, 1), color=(0, 128, 255, 0)).save(
        image_path, format="JPEG", icc_profile=profile
    )

    with pytest.raises(ValueError, match="cannot decode selected image"):
        selected_assets.decode_image_payload(image_path, _fixture_decoder())


def test_calibration_validation_rejects_non_utf8_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid-calibration.jsonl"
    source.write_bytes(b'{"id":"ok"}\n\xff\n')
    artifact = tmp_path / "calibration.json"
    _calibration_artifact(
        artifact,
        metric="cosine_similarity",
        selected_value=0.8,
        comparison="reject_at_or_above",
        source_path=source,
        positive_count=1,
        negative_count=1,
        model_hash="a" * 64,
        sample_ids=["ok"],
    )
    with pytest.raises(ValueError, match="calibration source is not readable JSONL"):
        _validate_calibration(
            _file_binding(artifact),
            label="semantic text",
            metric="cosine_similarity",
            selected_value=0.8,
            comparison="reject_at_or_above",
            model_hash="a" * 64,
            sample_ids=["ok"],
        )


def _file_binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _small_protected_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    parent = tmp_path / "catalog-test"
    parent.mkdir()
    image_paths = []
    for index, color in enumerate(((20, 30, 40), (80, 90, 100))):
        image = parent / f"image-{index}.png"
        Image.new("RGB", (2, 2), color=color).save(image)
        image_paths.append(image)
    holdout = parent / "benchmark_holdouts.jsonl"
    rows = [
        {"id": "prompt-0", "prompt": "first protected prompt", "image_path": str(image_paths[0])},
        {"id": "prompt-1", "prompt": "second protected prompt", "image_path": str(image_paths[1])},
    ]
    # The formal contract has a fixed 49,393-row parent.  Repeating two
    # stable rows keeps this adversarial unit test small in semantic content
    # while exercising the same count gate.
    holdout.write_bytes(
        b"".join((json.dumps(rows[index % 2], sort_keys=True) + "\n").encode() for index in range(49393))
    )
    parent_manifest_sha = "a" * 64
    parent_manifest = {
        "release_id": parent.name,
        "protected_normalized_unique_prompts": 2,
        "protected_unique_images": 2,
    }
    decoder = _fixture_decoder()
    manifest = derive_protected_index_manifest(
        parent,
        parent_release_id=parent.name,
        parent_manifest_sha256=parent_manifest_sha,
        decoder=decoder,
    )
    manifest_path = parent / "protected-index-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    semantic = parent / "semantic.npz"
    image = parent / "image.npz"
    ids = np.asarray(["prompt-0", "prompt-1"])
    embeddings = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (2, 1))
    np.savez(semantic, ids=ids, embeddings=embeddings)
    np.savez(image, ids=ids, embeddings=embeddings)
    phash = parent / "phash.jsonl"
    phash.write_text(
        "".join(
            json.dumps({"id": value, "phash": f"{index + 1:016x}"}) + "\n"
            for index, value in enumerate(ids.tolist())
        ),
        encoding="utf-8",
    )
    index_bindings = {
        "semantic_text": {
            "manifest_sha256": manifest["manifest_sha256"],
            "ids_sha256": protected_ids_sha256(ids.tolist()),
        },
        "phash": {
            "manifest_sha256": manifest["manifest_sha256"],
            "ids_sha256": protected_ids_sha256(ids.tolist()),
        },
        "image_embedding": {
            "manifest_sha256": manifest["manifest_sha256"],
            "ids_sha256": protected_ids_sha256(ids.tolist()),
        },
    }
    protected = {
        "holdout_rows": 49393,
        "normalized_unique_prompts": 2,
        "unique_images": 2,
        "manifest": _file_binding(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "semantic_text": _file_binding(semantic),
        "phash": _file_binding(phash),
        "image_embedding": _file_binding(image),
        "index_bindings": index_bindings,
    }
    return parent_manifest, parent, {"decoder": decoder, "protected": protected, "manifest": manifest}


def test_unique_protected_prompts_is_stable_and_deduplicates_normalized_text() -> None:
    rows = [
        {"id": "prompt-z", "prompt": "  A shared prompt  "},
        {"id": "prompt-a", "prompt": "a   shared prompt"},
        {"id": "prompt-b", "prompt": "A different prompt"},
        {"id": "ignored", "prompt": ""},
        {"id": "ignored-none", "prompt": None},
    ]

    expected = [
        {
            "id": "prompt-b",
            "prompt": "A different prompt",
            "normalized": "a different prompt",
        },
        {
            "id": "prompt-a",
            "prompt": "a   shared prompt",
            "normalized": "a shared prompt",
        },
    ]
    assert unique_protected_prompts(rows) == expected
    assert unique_protected_prompts(list(reversed(rows))) == expected


def test_unique_protected_prompts_rejects_a_relevant_row_without_id() -> None:
    with pytest.raises(ValueError, match="stable id"):
        unique_protected_prompts([{"prompt": "usable prompt"}])


def test_unique_protected_images_is_stable_and_deduplicates_resolved_paths(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    rows = [
        {"id": "image-z", "image_path": str(first)},
        {"id": "image-a", "image_path": str(tmp_path / "." / "first.bin")},
        {"id": "image-b", "image_path": str(second)},
        {"id": "ignored", "image_path": ""},
    ]

    expected = [
        {"id": "image-a", "image_path": str(first.resolve())},
        {"id": "image-b", "image_path": str(second.resolve())},
    ]
    assert unique_protected_images(rows) == expected
    assert unique_protected_images(list(reversed(rows))) == expected


def test_unique_protected_images_rejects_missing_relevant_assets(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError, match="protected image is missing"):
        unique_protected_images([{"id": "image-1", "image_path": str(missing)}])


def test_unique_protected_images_rejects_symlink_paths(tmp_path: Path) -> None:
    target = tmp_path / "target.jpg"
    target.write_bytes(b"image")
    link = tmp_path / "link.jpg"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        unique_protected_images([{"id": "image-1", "image_path": str(link)}])


def test_selected_asset_preflight_rejects_input_output_collision(tmp_path: Path) -> None:
    output_dir = tmp_path / "selected"
    config_output = tmp_path / "config.json"
    destinations = _selected_asset_destinations(output_dir, config_output)

    with pytest.raises(ValueError, match="output .*collides"):
        _preflight_selected_asset_destinations(
            destinations, [output_dir / "protected_image_embedding.npz"]
        )

    assert not output_dir.exists()


def test_selected_asset_preflight_rejects_duplicate_destinations(tmp_path: Path) -> None:
    output_dir = tmp_path / "selected"
    duplicate = output_dir / "asset_manifest.json"

    with pytest.raises(ValueError, match="output paths collide"):
        _preflight_selected_asset_destinations(
            (("config", duplicate), ("asset_manifest", duplicate)), []
        )


def test_parent_revalidation_rejects_manifest_changed_after_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    manifest = parent / "manifest.json"
    manifest.write_text('{"release_id":"before"}\n', encoding="utf-8")

    def mutate_after_validation(_path: Path) -> Path:
        manifest.write_text('{"release_id":"after"}\n', encoding="utf-8")
        return parent

    monkeypatch.setattr(
        selected_assets, "_validate_parent_release_fast", mutate_after_validation
    )
    with pytest.raises(RuntimeError, match="parent catalog changed"):
        _revalidate_parent_release_after_reads(
            parent,
            expected_manifest_sha256=hashlib.sha256(
                b'{"release_id":"before"}\n'
            ).hexdigest(),
        )


def test_parent_snapshot_rejects_atomic_artifact_replacement(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    manifest = parent / "manifest.json"
    artifact = parent / "training_candidates.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": artifact.name,
                        "bytes": len(b"original\n"),
                        "sha256": hashlib.sha256(b"original\n").hexdigest(),
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact.write_bytes(b"original\n")
    snapshot = _capture_parent_artifact_snapshot(
        parent,
        json.loads(manifest.read_text(encoding="utf-8")),
    )
    replacement = parent / "replacement.jsonl"
    replacement.write_bytes(b"original\n")
    replacement.replace(artifact)
    with pytest.raises(RuntimeError, match="parent catalog changed"):
        _assert_parent_artifact_snapshot(snapshot)


def test_parent_snapshot_reads_pinned_bytes_after_directory_swap(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    alternate = tmp_path / "alternate"
    parent.mkdir()
    alternate.mkdir()
    original = json.dumps({"id": "original"}, sort_keys=True).encode() + b"\n"
    replacement = json.dumps({"id": "replacement"}, sort_keys=True).encode() + b"\n"
    for directory, payload in ((parent, original), (alternate, replacement)):
        artifact = directory / "training_candidates.jsonl"
        artifact.write_bytes(payload)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "artifacts": [
                        {
                            "path": artifact.name,
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
    snapshot = _capture_parent_artifact_snapshot(parent)
    saved = tmp_path / "saved"
    os.replace(parent, saved)
    os.replace(alternate, parent)
    try:
        consumed = list(snapshot.iter_jsonl("training_candidates.jsonl"))
    finally:
        os.replace(parent, alternate)
        os.replace(saved, parent)
        snapshot.close()
    assert consumed == [{"id": "original"}]


def test_parent_snapshot_rejects_transient_jsonl_rewrite_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    original = b'{"id":"original"}\n'
    replacement = b'{"id":"replacement"}\n'
    artifact = parent / "training_candidates.jsonl"
    artifact.write_bytes(original)
    (parent / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": artifact.name,
                        "bytes": len(original),
                        "sha256": hashlib.sha256(original).hexdigest(),
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    real_iter = selected_module._iter_jsonl_descriptor

    # Model a filesystem whose metadata can remain stale while bytes are
    # rewritten, as the catalog validator's race tests do.
    with monkeypatch.context() as patch:
        patch.setattr(selected_module, "_artifact_identity", lambda _value: (1, 2, 3, 4, 5))
        snapshot = _capture_parent_artifact_snapshot(parent)

        def rewrite_then_restore(descriptor, path, **kwargs):
            path.write_bytes(replacement)
            iterator = real_iter(descriptor, path, **kwargs)
            try:
                row = next(iterator)
            finally:
                path.write_bytes(original)
            yield row
            yield from iterator

        patch.setattr(
            selected_module,
            "_iter_jsonl_descriptor",
            rewrite_then_restore,
        )
        try:
            with pytest.raises(RuntimeError, match="parent catalog changed"):
                list(snapshot.iter_jsonl("training_candidates.jsonl"))
        finally:
            snapshot.close()


def test_parent_snapshot_rejects_transient_read_bytes_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    original = b"original\n"
    replacement = b"replaced!\n"
    artifact = parent / "training_candidates.jsonl"
    artifact.write_bytes(original)
    (parent / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": artifact.name,
                        "bytes": len(original),
                        "sha256": hashlib.sha256(original).hexdigest(),
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    real_read = selected_module._read_descriptor_bytes
    with monkeypatch.context() as patch:
        patch.setattr(selected_module, "_artifact_identity", lambda _value: (1, 2, 3, 4, 5))
        snapshot = _capture_parent_artifact_snapshot(parent)

        def rewrite_then_restore(descriptor):
            artifact.write_bytes(replacement)
            try:
                return real_read(descriptor)
            finally:
                artifact.write_bytes(original)

        patch.setattr(selected_module, "_read_descriptor_bytes", rewrite_then_restore)
        try:
            with pytest.raises(RuntimeError, match="parent catalog changed"):
                snapshot.read_bytes("training_candidates.jsonl")
        finally:
            snapshot.close()


def test_build_selected_assets_closes_snapshot_when_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []

    class _Snapshot:
        def close(self) -> None:
            closed.append(True)

    snapshot = _Snapshot()

    def capture(_parent: Path) -> _Snapshot:
        # The real capture helper registers snapshots in this scope.  Mirror
        # that ownership contract while keeping this test model-free.
        selected_module._register_parent_snapshot(snapshot)
        return snapshot

    monkeypatch.setattr(selected_assets, "_capture_parent_artifact_snapshot", capture)
    real_validate_directory = selected_assets._validate_directory

    def fail_output_parent(path: Path, *, label: str) -> Path:
        if label == "selected-view output parent":
            raise RuntimeError("setup failure")
        return real_validate_directory(path, label=label)

    monkeypatch.setattr(selected_assets, "_validate_directory", fail_output_parent)
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    output_dir = tmp_path / "outputs" / "selected"
    with pytest.raises(RuntimeError, match="setup failure"):
        selected_assets.build_selected_assets(
            parent_release=input_root / "parent",
            output_dir=output_dir,
            config_output=output_dir / "config.json",
            clip_checkpoint=input_root / "clip.pt",
            tokenizer_root=input_root / "tokenizer",
            tokenizer_root_2=input_root / "tokenizer-2",
        )

    assert closed == [True]


def test_build_selected_assets_removes_publication_when_parent_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []

    class _Snapshot:
        def close(self) -> None:
            closed.append(True)

    snapshot = _Snapshot()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    output_dir = tmp_path / "outputs" / "selected"

    monkeypatch.setattr(
        selected_assets,
        "_capture_parent_artifact_snapshot",
        lambda _path: snapshot,
    )
    monkeypatch.setattr(
        selected_assets,
        "_build_selected_assets_in_directory",
        lambda **_kwargs: {"stub": True},
    )
    monkeypatch.setattr(selected_assets, "_fsync_tree", lambda _path: None)

    def parent_changed(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("parent changed after publication")

    monkeypatch.setattr(
        selected_assets, "_assert_parent_artifact_snapshot", parent_changed
    )
    with pytest.raises(RuntimeError, match="parent changed after publication"):
        selected_assets.build_selected_assets(
            parent_release=input_root / "parent",
            output_dir=output_dir,
            config_output=output_dir / "config.json",
            clip_checkpoint=input_root / "clip.pt",
            tokenizer_root=input_root / "tokenizer",
            tokenizer_root_2=input_root / "tokenizer-2",
        )

    assert not output_dir.exists()
    assert closed == [True]


def test_source_prompt_fields_match_normalized_catalog_contract() -> None:
    assert SOURCE_PROMPT_FIELDS == {
        "four_k_lsdb": {
            "model_prompt_field": "prompt",
            "raw_prompt_field": "prompt",
        },
        "pixverve_95k": {
            "model_prompt_field": "source_record.model_prompt",
            "raw_prompt_field": "source_record.raw_prompt",
        },
    }


def test_classifier_calibration_forwards_decode_worker_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates_path = tmp_path / "training_candidates.jsonl"
    rows = []
    for class_index, name in enumerate(selected_assets.STRATA):
        for item_index in range(32):
            image_path = tmp_path / f"{class_index:02d}-{item_index:02d}.bin"
            image_path.write_bytes(b"fixture")
            prompt = f"a {name} scene"
            if name == "nature":
                prompt = "a landscape scene"
            rows.append(
                {
                    "id": f"candidate-{class_index:02d}-{item_index:02d}",
                    "modality": "image_text",
                    "prompt": prompt,
                    "image_path": str(image_path),
                }
            )
    candidates_path.write_bytes(
        b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    )
    observed_workers: list[int] = []

    def fake_encode_images(*_args: object, **kwargs: object) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
        observed_workers.append(int(kwargs["decode_workers"]))
        embeddings = np.zeros((len(rows), len(selected_assets.STRATA) * 3), dtype=np.float32)
        for index in range(len(rows)):
            embeddings[index, 0] = 1.0
            embeddings[index, 3] = 0.1 + index / (len(rows) * 2.0)
        return embeddings, [], []

    monkeypatch.setattr(selected_assets, "_encode_images", fake_encode_images)
    monkeypatch.setattr(
        selected_assets,
        "_encode_texts",
        lambda *_args, **_kwargs: np.eye(len(selected_assets.STRATA) * 3, dtype=np.float32),
    )

    selected_assets._build_classifier_calibration(
        tmp_path,
        object(),
        object(),
        object(),
        decoder=_fixture_decoder(),
        device="cpu",
        batch_size=8,
        decode_workers=3,
        source_path=tmp_path / "classifier.jsonl",
        artifact_path=tmp_path / "classifier.json",
        model_hash="a" * 64,
    )

    assert observed_workers == [3]


def test_cyclic_hamming_distances_keep_64_bit_hashes_as_integers() -> None:
    assert _cyclic_hamming_distances([0xFFFFFFFFFFFFFFFF, 0, 1]) == [64, 1, 63]


def test_source_prompt_paths_are_checked_against_real_candidate_rows(tmp_path: Path) -> None:
    candidates = tmp_path / "training_candidates.jsonl"
    rows = [
        {
            "id": "four-k-1",
            "source": "four_k_lsdb",
            "prompt": "a four k prompt",
            "training_eligible": True,
            "benchmark_exact_match": [],
        },
        {
            "id": "pix-1",
            "source": "pixverve_95k",
            "prompt": "a short prompt",
            "source_record": {
                "model_prompt": "a short prompt",
                "raw_prompt": "a long prompt",
            },
            "training_eligible": True,
            "benchmark_exact_match": [],
        },
    ]
    candidates.write_bytes(
        b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
        )
    )
    _validate_source_prompt_fields(tmp_path)

    rows[1]["source_record"].pop("raw_prompt")
    candidates.write_bytes(
        b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
        )
    )
    with pytest.raises(ValueError, match="prompt field mapping"):
        _validate_source_prompt_fields(tmp_path)


def test_source_evidence_must_match_parent_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    four_k = tmp_path / "four_k.jsonl"
    pixverve = tmp_path / "pixverve.jsonl"
    four_k.write_bytes(b"four k source")
    pixverve.write_bytes(b"pixverve source")
    paths = {"four_k_lsdb": four_k, "pixverve_95k": pixverve}
    monkeypatch.setattr(selected_assets, "SOURCE_EVIDENCE_PATHS", paths)
    provenance = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": selected_assets.file_binding(path)["sha256"],
        }
        for path in paths.values()
    ]
    assert set(_source_evidence_from_parent({"source_provenance": provenance})) == set(paths)
    pixverve.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="differs from the parent manifest"):
        _source_evidence_from_parent({"source_provenance": provenance})


def test_checkpoint_hash_is_checked_before_clip_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"not-the-pinned-checkpoint")
    called = False

    def unexpected_loader(*args: object, **kwargs: object) -> tuple[object, object]:
        nonlocal called
        called = True
        raise AssertionError("the CLIP loader must not see an unbound checkpoint")

    monkeypatch.setattr(selected_assets, "_load_clip", unexpected_loader)
    with pytest.raises(ValueError, match="pinned selected-view revision"):
        selected_assets._load_bound_clip(checkpoint, "cpu")
    assert called is False


def test_tokenizer_registration_rejects_unbound_file_bytes(tmp_path: Path) -> None:
    root = tmp_path / "tokenizer"
    root.mkdir()
    for name in TOKENIZER_FILE_SHA256["sdxl_tokenizer_1"]:
        (root / name).write_bytes(b"fixture tokenizer file")
    with pytest.raises(ValueError, match="frozen registration"):
        selected_assets._tokenizer_descriptor(root, "sdxl_tokenizer_1")


def test_same_count_but_wrong_protected_index_ids_are_rejected(tmp_path: Path) -> None:
    parent_manifest, parent, fixture = _small_protected_fixture(tmp_path)
    protected = fixture["protected"]
    selected_path = Path(protected["semantic_text"]["path"])
    np.savez(
        selected_path,
        ids=np.asarray(["prompt-1", "prompt-0"]),
        embeddings=np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (2, 1)),
    )
    protected["semantic_text"] = _file_binding(selected_path)
    with pytest.raises(ValueError, match="IDs differ from the parent manifest"):
        _validate_protected_index_binding(
            protected,
            parent=parent_manifest,
            parent_dir=parent,
            parent_manifest_sha256="a" * 64,
            decoder=fixture["decoder"],
        )


@pytest.mark.parametrize("override", ["holdout_rows", "prompt_rows", "image_rows"])
def test_manifest_derivation_rejects_rows_not_from_current_holdout(
    tmp_path: Path, override: str
) -> None:
    _parent_manifest, parent, fixture = _small_protected_fixture(tmp_path)
    values: dict[str, object] = {
        "holdout_rows": [{"id": "forged", "prompt": "forged"}],
        "prompt_rows": [
            {"id": "forged", "prompt": "forged", "normalized": "forged"}
        ],
        "image_rows": [{"id": "forged", "image_path": "/tmp/forged.png"}],
    }
    with pytest.raises(ValueError, match="current (holdout|holdout cohort)"):
        derive_protected_index_manifest(
            parent,
            parent_release_id=parent.name,
            parent_manifest_sha256="a" * 64,
            decoder=fixture["decoder"],
            **{override: values[override]},
        )


def test_metadata_only_manifest_validation_uses_frozen_image_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent_manifest, parent, fixture = _small_protected_fixture(tmp_path)

    def unexpected_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("metadata-only validation must not decode images")

    monkeypatch.setattr(selected_assets, "decode_image_payload", unexpected_decode)
    observed = derive_protected_index_manifest(
        parent,
        parent_release_id=parent.name,
        parent_manifest_sha256="a" * 64,
        decoder=fixture["decoder"],
        image_evidence=fixture["manifest"]["images"],
        verify_image_evidence=False,
    )
    assert observed == fixture["manifest"]


def test_metadata_only_manifest_still_binds_holdout_bytes(
    tmp_path: Path,
) -> None:
    parent_manifest, parent, fixture = _small_protected_fixture(tmp_path)
    holdout = parent / "benchmark_holdouts.jsonl"
    with holdout.open("ab") as handle:
        handle.write(b"{\"id\":\"changed\",\"prompt\":\"changed\"}\n")
    with pytest.raises(ValueError, match="protected index manifest differs"):
        _validate_protected_index_binding(
            fixture["protected"],
            parent=parent_manifest,
            parent_dir=parent,
            parent_manifest_sha256="a" * 64,
            decoder=fixture["decoder"],
        )


def test_formal_manifest_validation_rechecks_current_protected_image_bytes(
    tmp_path: Path,
) -> None:
    parent_manifest, parent, fixture = _small_protected_fixture(tmp_path)
    image_path = Path(fixture["manifest"]["images"][0]["image_path"])
    image_path.write_bytes(b"replaced-after-index-build")

    with pytest.raises(ValueError, match="cannot derive the protected index manifest"):
        _validate_protected_index_binding(
            fixture["protected"],
            parent=parent_manifest,
            parent_dir=parent,
            parent_manifest_sha256="a" * 64,
            decoder=fixture["decoder"],
        )


def test_protected_calibration_rejects_misbinding_to_another_index(
    tmp_path: Path,
) -> None:
    _parent_manifest, _parent, fixture = _small_protected_fixture(tmp_path)
    source = tmp_path / "calibration.jsonl"
    sample_ids = _protected_sample_ids(["prompt-0", "prompt-1"])
    source.write_text(
        "".join(
            json.dumps(
                {"pair": index, "label": "positive", "protected_ids": ids}
            )
            + "\n"
            for index, ids in enumerate(sample_ids)
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "calibration.json"
    _calibration_artifact(
        artifact,
        metric="cosine_similarity",
        selected_value=0.8,
        comparison="reject_at_or_above",
        source_path=source,
        positive_count=2,
        negative_count=2,
        model_hash="b" * 64,
        protected_manifest_sha256=fixture["manifest"]["manifest_sha256"],
        protected_index_sha256="c" * 64,
        protected_sample_ids=sample_ids,
    )
    binding = _file_binding(artifact)
    with pytest.raises(ValueError, match="protected binding differs from the index"):
        _validate_calibration(
            binding,
            label="semantic text",
            metric="cosine_similarity",
            selected_value=0.8,
            comparison="reject_at_or_above",
            model_hash="b" * 64,
            protected_sample_binding={
                "manifest_sha256": fixture["manifest"]["manifest_sha256"],
                "index_sha256": "d" * 64,
                "sample_ids": sample_ids,
            },
        )
