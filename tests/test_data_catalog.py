from __future__ import annotations

import hashlib
import json
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "eval-pipeline"
sys.path.insert(0, str(EVAL_ROOT))

from data_catalog.builder import (
    CATALOG_SCHEMA,
    CONFIG_SNAPSHOT_NAME,
    CONFIG_SNAPSHOT_SCHEMA,
    REQUIRED_SOURCE_ROOT_IDS,
    REPOSITORY_ROOT,
    _artifact_contract,
    _assert_artifact_contract,
    _builder_provenance,
    _compare_training_candidate_derivation,
    _load_config_bytes,
    _release_id,
    _validate_builder_provenance,
    _validate_formal_git_record,
    _validate_recorded_upstream_ancestry,
    build_catalog,
    enforce_git_gate,
    git_state,
    load_config,
    validate_release,
)
import data_catalog.builder as catalog_builder
from data_catalog.io import canonical_json_bytes, iter_jsonl, write_records
from data_catalog.protected import (
    iter_holdout_records,
    load_protected_prompts,
    unique_protected_image_count,
)
from data_catalog.schema import (
    DATA_RECORD_SCHEMA,
    benchmark_matches,
    make_record,
    normalize_prompt,
)
from data_catalog.sources import (
    SourceContext,
    iter_pixel_render_images,
    iter_pixverve,
    iter_sana_prompts,
)


class SchemaTest(unittest.TestCase):
    def test_prompt_normalization_is_conservative_and_unicode_aware(self):
        self.assertEqual(normalize_prompt("  A\tPHOTO  of Cat\n"), "a photo of cat")
        self.assertEqual(normalize_prompt("Ａ cat"), "a cat")

    def test_benchmark_match_makes_training_row_ineligible(self):
        protected = {normalize_prompt("A photo of a cat"): {"heldout"}}
        record = make_record(
            source="fixture",
            stable_key="1",
            source_roots=("fixture_root",),
            split="train",
            prompt=" a  photo of a CAT ",
            image_path=None,
            license_name="test",
            license_status="test",
            modality="prompt",
            intended_use=("rl_prompt_pool",),
            training_eligible=True,
            benchmark_exact_match=benchmark_matches(
                " a  photo of a CAT ", protected
            ),
        )
        self.assertFalse(record["training_eligible"])
        self.assertEqual(record["exclusion_reason"], "benchmark_prompt_exact_match")
        self.assertEqual(record["benchmark_exact_match"], ["heldout"])

    def test_boolean_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "width must be a positive integer"):
            make_record(
                source="fixture",
                stable_key="1",
                source_roots=("fixture_root",),
                split="train",
                prompt="prompt",
                image_path=None,
                width=True,
                license_name="test",
                license_status="test",
                modality="prompt",
                intended_use=("rl_prompt_pool",),
                training_eligible=True,
            )

    def test_dimension_coercion_is_rejected(self):
        for value in (2.0, 2.5, "2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "width must be a positive integer"):
                    make_record(
                        source="fixture",
                        stable_key=f"bad-dimension-{value}",
                        source_roots=("fixture_root",),
                        split="train",
                        prompt="prompt",
                        image_path=None,
                        width=value,
                        license_name="test",
                        license_status="test",
                        modality="prompt",
                        intended_use=("rl_prompt_pool",),
                        training_eligible=True,
                    )

    def test_source_context_fails_closed_for_unapproved_license_status(self):
        context = SourceContext(
            repository_root=ROOT,
            protected_prompts={},
            allowed_training_license_statuses=frozenset({"approved"}),
        )
        record = context.record(
            source="fixture",
            stable_key="1",
            source_roots=("fixture",),
            split="train",
            prompt="prompt",
            image_path=None,
            license_name="unknown",
            license_status="review_required",
            modality="prompt",
            intended_use=("rl_prompt_pool",),
            training_eligible=True,
        )
        self.assertFalse(record["training_eligible"])
        self.assertEqual(
            record["exclusion_reason"], "license_status_not_allowed_for_training"
        )

    def test_make_record_rejects_non_boolean_training_eligibility(self):
        with self.assertRaisesRegex(ValueError, "training_eligible must be boolean"):
            make_record(
                source="fixture",
                stable_key="bad-bool",
                source_roots=("fixture",),
                split="train",
                prompt="prompt",
                image_path=None,
                license_name="test",
                license_status="test",
                modality="prompt",
                intended_use=("rl_prompt_pool",),
                training_eligible="false",
            )


class AdapterTest(unittest.TestCase):
    def test_payload_path_rejects_symlink_even_when_target_is_inside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = root / "physical"
            physical.mkdir()
            target = physical / "target.jpg"
            target.write_bytes(b"fixture")
            link = root / "link.jpg"
            link.symlink_to(target)
            context = SourceContext(
                repository_root=root,
                protected_prompts={},
                allowed_training_license_statuses=frozenset({"test"}),
                physical_roots={"fixture": physical},
            )
            with self.assertRaisesRegex(ValueError, "symlink"):
                context.payload_path(
                    link, source_roots=("fixture",), label="fixture payload"
                )

    def _sharded_protected_spec(
        self,
        root: Path,
        rows: list[tuple[str, str, str]],
        shard_images: list[list[str]],
        *,
        expected_shards: int | None = None,
        expected_images: int | None = None,
    ) -> dict:
        metadata = root / "metadata.csv"
        metadata.write_text(
            "id,caption,cogvlm_caption\n"
            + "".join(
                f"{image_id},{caption},{extra}\n"
                for image_id, caption, extra in rows
            ),
            encoding="utf-8",
        )
        image_root = root / "images"
        for shard_index, image_ids in enumerate(shard_images):
            payload = image_root / f"split-{shard_index:02d}" / "hr"
            payload.mkdir(parents=True)
            for image_id in image_ids:
                (payload / f"{image_id}.jpg").write_bytes(image_id.encode("ascii"))
        return {
            "id": "sharded_benchmark",
            "path": str(metadata),
            "format": "csv",
            "prompt_keys": ["caption", "cogvlm_caption"],
            "source_roots": ["fixture"],
            "image_root": str(image_root),
            "image_path_mode": "sharded_id_jpg",
            "image_shard_glob": "split-*",
            "image_subdir": "hr",
            "image_id_key": "id",
            "expected_image_shards": (
                len(shard_images) if expected_shards is None else expected_shards
            ),
            "expected_unique_images": (
                len({value for shard in shard_images for value in shard})
                if expected_images is None
                else expected_images
            ),
            "license": "test",
            "license_status": "benchmark_only",
            "expected_bytes": metadata.stat().st_size,
            "expected_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            "expected_prompts": sum(bool(caption) + bool(extra) for _, caption, extra in rows),
        }

    def test_prompt_adapter_preserves_split_and_applies_firewall(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.txt"
            test = root / "test.txt"
            train.write_text("safe prompt\nheld out prompt\n", encoding="utf-8")
            test.write_text("held out prompt\n", encoding="utf-8")
            context = SourceContext(
                repository_root=root,
                protected_prompts={normalize_prompt("held out prompt"): {"test"}},
                allowed_training_license_statuses=frozenset({"test"}),
            )
            spec = {
                "id": "fixture_prompts",
                "source_roots": ["fixture"],
                "license": "test",
                "license_status": "test",
                "intended_use": ["rl_prompt_pool"],
                "splits": [
                    {
                        "name": "train",
                        "path": str(train),
                        "format": "text",
                        "training_eligible": True,
                    },
                    {
                        "name": "test",
                        "path": str(test),
                        "format": "text",
                        "training_eligible": False,
                        "exclusion_reason": "heldout_prompt_split",
                    },
                ],
            }
            records = list(iter_sana_prompts(spec, context))
        self.assertEqual(len(records), 3)
        self.assertTrue(records[0]["training_eligible"])
        self.assertFalse(records[1]["training_eligible"])
        self.assertEqual(records[1]["exclusion_reason"], "benchmark_prompt_exact_match")
        self.assertFalse(records[2]["training_eligible"])
        self.assertEqual(records[2]["exclusion_reason"], "heldout_prompt_split")

    def test_pixverve_firewall_checks_long_and_short_captions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            for name in ("long.png", "short.png"):
                (image_root / name).write_bytes(b"fixture")
            manifest = root / "pixverve.jsonl"
            rows = [
                {
                    "file_name": "long.png",
                    "long_caption": "protected long caption",
                    "short_caption": "safe short caption",
                },
                {
                    "file_name": "short.png",
                    "long_caption": "safe long caption",
                    "short_caption": "protected short caption",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            context = SourceContext(
                repository_root=root,
                protected_prompts={
                    normalize_prompt("protected long caption"): {"long_holdout"},
                    normalize_prompt("protected short caption"): {"short_holdout"},
                },
                allowed_training_license_statuses=frozenset(
                    {"verified_from_dataset_card"}
                ),
            )
            records = list(
                iter_pixverve(
                    {
                        "id": "pixverve_fixture",
                        "source_roots": ["fixture"],
                        "manifest": str(manifest),
                        "image_root": str(image_root),
                    },
                    context,
                )
            )
        self.assertEqual(len(records), 2)
        self.assertFalse(records[0]["training_eligible"])
        self.assertEqual(records[0]["benchmark_exact_match"], ["long_holdout"])
        self.assertFalse(records[1]["training_eligible"])
        self.assertEqual(records[1]["benchmark_exact_match"], ["short_holdout"])
        self.assertEqual(records[0]["prompt"], "safe short caption")
        self.assertEqual(records[0]["source_record"]["model_prompt"], "safe short caption")
        self.assertEqual(records[0]["source_record"]["raw_prompt"], "protected long caption")

    def test_protected_json_map_resolves_reference_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images/animals/key.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"fixture")
            metadata = root / "metadata.json"
            metadata.write_text(
                json.dumps({"key": {"prompt": "held out", "category": "animals"}}),
                encoding="utf-8",
            )
            specs = [
                {
                    "id": "benchmark",
                    "path": str(metadata),
                    "format": "json_map",
                    "prompt_keys": ["prompt"],
                    "source_roots": ["fixture"],
                    "image_root": str(root / "images"),
                    "image_path_mode": "category_key_jpg",
                    "license": "test",
                    "license_status": "benchmark_only",
                    "expected_bytes": metadata.stat().st_size,
                    "expected_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    "expected_prompts": 1,
                }
            ]
            protected, rows, counts = load_protected_prompts(specs, root)
            output = list(iter_holdout_records(rows))
        self.assertEqual(counts, {"benchmark": 1})
        self.assertEqual(protected["held out"], {"benchmark"})
        self.assertEqual(output[0]["image_path"], str(image.resolve()))
        self.assertFalse(output[0]["training_eligible"])

    def test_protected_source_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("held out\n", encoding="utf-8")
            spec = {
                "id": "benchmark",
                "path": str(path),
                "format": "text",
                "prompt_keys": ["prompt"],
                "source_roots": ["fixture"],
                "license": "test",
                "license_status": "benchmark_only",
                "expected_bytes": path.stat().st_size,
                "expected_sha256": "0" * 64,
                "expected_prompts": 1,
            }
            with self.assertRaisesRegex(ValueError, "hash changed"):
                load_protected_prompts([spec], ROOT)

    def test_sharded_protected_source_resolves_each_metadata_id_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self._sharded_protected_spec(
                root,
                [("1", "first", "detailed first"), ("2", "second", "")],
                [["1"], ["2"]],
            )
            protected, rows, counts = load_protected_prompts([spec], root)
        self.assertEqual(counts, {"sharded_benchmark": 3})
        self.assertEqual(len(protected), 3)
        self.assertEqual(unique_protected_image_count(rows), 2)
        self.assertEqual(rows[0].image_path, rows[1].image_path)
        self.assertNotEqual(rows[1].image_path, rows[2].image_path)

    def test_sharded_protected_source_rejects_duplicate_image_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self._sharded_protected_spec(
                root,
                [("1", "first", "")],
                [["1"], ["1"]],
                expected_images=1,
            )
            with self.assertRaisesRegex(ValueError, "duplicate protected image id"):
                load_protected_prompts([spec], root)

    def test_sharded_protected_source_rejects_missing_metadata_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self._sharded_protected_spec(
                root,
                [("1", "first", "")],
                [["2"]],
            )
            with self.assertRaisesRegex(FileNotFoundError, "has no JPG"):
                load_protected_prompts([spec], root)

    def test_sharded_protected_source_rejects_image_without_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self._sharded_protected_spec(
                root,
                [("1", "first", "")],
                [["1", "2"]],
            )
            with self.assertRaisesRegex(ValueError, "images without metadata"):
                load_protected_prompts([spec], root)

    def test_sharded_protected_source_rejects_duplicate_metadata_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self._sharded_protected_spec(
                root,
                [("1", "first", ""), ("1", "second", "")],
                [["1"]],
            )
            with self.assertRaisesRegex(ValueError, "repeats metadata image id"):
                load_protected_prompts([spec], root)

    def test_sharded_protected_source_rejects_wrong_shard_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self._sharded_protected_spec(
                root,
                [("1", "first", "")],
                [["1"]],
                expected_shards=2,
            )
            with self.assertRaisesRegex(ValueError, "found 1 image shards, expected 2"):
                load_protected_prompts([spec], root)

    def test_pixel_render_keeps_missing_manifest_rows_but_excludes_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            present = root / "data/unsplash-full/present.jpg"
            present.parent.mkdir(parents=True)
            present.write_bytes(b"fixture")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "path": "data/unsplash-full/present.jpg",
                            "width": 2,
                            "height": 2,
                        },
                        {
                            "path": "data/unsplash-full/missing.jpg",
                            "width": 2,
                            "height": 2,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            spec = {
                "id": "pixel_render_image_only",
                "source_roots": ["pixel_render_fixture"],
                "manifest": str(manifest),
                "data_root": str(root),
            }
            context = SourceContext(
                repository_root=root,
                protected_prompts={},
                allowed_training_license_statuses=frozenset(
                    {"verified_from_dataset_card"}
                ),
            )
            records = list(iter_pixel_render_images(spec, context))
            stats = write_records(
                root / "catalog.jsonl",
                records,
                verify_paths=True,
                expected_rows=2,
            )
        self.assertFalse(records[0]["training_eligible"])
        self.assertEqual(records[0]["image_path"], str(present))
        self.assertEqual(
            records[0]["exclusion_reason"], "source_not_approved_for_training"
        )
        self.assertFalse(records[1]["training_eligible"])
        self.assertIsNone(records[1]["image_path"])
        self.assertEqual(records[1]["exclusion_reason"], "source_image_missing")
        self.assertEqual(
            records[1]["source_record"]["declared_image_path"],
            str(present.parent / "missing.jpg"),
        )
        self.assertEqual(stats["rows_with_image"], 1)
        self.assertEqual(stats["missing_images"], 1)


class ArtifactTest(unittest.TestCase):
    def _record(self, key):
        return make_record(
            source="fixture",
            stable_key=key,
            source_roots=("fixture",),
            split="train",
            prompt="prompt",
            image_path=None,
            license_name="test",
            license_status="test",
            modality="prompt",
            intended_use=("rl_prompt_pool",),
            training_eligible=True,
        )

    def test_writer_hashes_and_counts_deterministic_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            stats = write_records(
                path,
                [self._record("a"), self._record("b")],
                verify_paths=True,
                expected_rows=2,
            )
            rows = list(iter_jsonl(path))
        self.assertEqual(stats["rows"], 2)
        self.assertEqual(stats["training_eligible_rows"], 2)
        self.assertEqual(len(stats["sha256"]), 64)
        self.assertEqual(len(rows), 2)

    def test_writer_rejects_duplicate_ids_and_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            record = self._record("same")
            with self.assertRaisesRegex(ValueError, "duplicate record id"):
                write_records(path, [record, record], verify_paths=False)
            self.assertFalse(path.exists())

    def test_writer_rejects_missing_image_and_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            record = make_record(
                source="fixture",
                stable_key="missing",
                source_roots=("fixture",),
                split="train",
                prompt="prompt",
                image_path=Path(directory) / "missing.png",
                license_name="test",
                license_status="test",
                modality="image_text",
                intended_use=("opd_teacher_targets",),
                training_eligible=True,
            )
            with self.assertRaises(FileNotFoundError):
                write_records(path, [record], verify_paths=True)
            self.assertFalse(path.exists())

    def test_validator_reuses_verified_artifact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.jsonl"
            path.write_bytes(b'{"id":"one","schema":"fixture.schema.v1"}\n')
            expected = {
                "path": path.name,
                "schema": "fixture.schema.v1",
                "rows": 1,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

            observed = catalog_builder._validate_artifact(
                path, expected, verify_paths=False
            )

        self.assertEqual(observed, expected)

    def test_validator_rechecks_hash_after_row_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.jsonl"
            original = b'{"id":"one","schema":"fixture.schema.v1"}\n'
            replacement = b'{"id":"two","schema":"fixture.schema.v1"}\n'
            path.write_bytes(original)
            expected = {
                "path": path.name,
                "schema": "fixture.schema.v1",
                "rows": 1,
                "bytes": len(original),
                "sha256": hashlib.sha256(original).hexdigest(),
            }
            real_hash = catalog_builder._sha256_descriptor
            calls = 0

            def rewrite_after_first_hash(value):
                nonlocal calls
                digest = real_hash(value)
                calls += 1
                if calls == 2:
                    path.write_bytes(replacement)
                return digest

            with mock.patch.object(
                catalog_builder,
                "_sha256_descriptor",
                side_effect=rewrite_after_first_hash,
            ):
                with self.assertRaisesRegex(
                    ValueError, "catalog artifact changed while it was read"
                ):
                    catalog_builder._validate_artifact(
                        path, expected, verify_paths=False
                    )

        self.assertEqual(calls, 2)

    def test_artifact_file_rechecks_pinned_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.jsonl"
            original = b'{"id":"one","schema":"fixture.schema.v1"}\n'
            replacement = b'{"id":"two","schema":"fixture.schema.v1"}\n'
            path.write_bytes(original)
            expected = {
                "path": path.name,
                "schema": "fixture.schema.v1",
                "rows": 1,
                "bytes": len(original),
                "sha256": hashlib.sha256(original).hexdigest(),
            }
            real_hash = catalog_builder._sha256_descriptor
            calls = 0

            def rewrite_after_snapshot_hash(value):
                nonlocal calls
                digest = real_hash(value)
                calls += 1
                if calls == 1:
                    path.write_bytes(replacement)
                return digest

            with mock.patch.object(
                catalog_builder,
                "_sha256_descriptor",
                side_effect=rewrite_after_snapshot_hash,
            ):
                with self.assertRaisesRegex(
                    ValueError, "catalog artifact changed while it was read"
                ):
                    catalog_builder._validate_artifact_file(path, expected)

        self.assertEqual(calls, 1)

    def test_artifact_file_rehash_rejects_same_metadata_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.jsonl"
            original = b'{"id":"one","schema":"fixture.schema.v1"}\n'
            replacement = b'{"id":"two","schema":"fixture.schema.v1"}\n'
            path.write_bytes(original)
            expected = {
                "path": path.name,
                "schema": "fixture.schema.v1",
                "rows": 1,
                "bytes": len(original),
                "sha256": hashlib.sha256(original).hexdigest(),
            }
            real_hash = catalog_builder._sha256_descriptor
            calls = 0

            def rewrite_after_initial_hash(value):
                nonlocal calls
                digest = real_hash(value)
                calls += 1
                if calls == 1:
                    path.write_bytes(replacement)
                return digest

            with mock.patch.object(
                catalog_builder,
                "_sha256_descriptor",
                side_effect=rewrite_after_initial_hash,
            ), mock.patch.object(
                catalog_builder, "_artifact_identity", return_value=(1, 2, 3, 4, 5)
            ):
                with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                    catalog_builder._validate_artifact_file(path, expected)

        self.assertEqual(calls, 2)

    def test_artifact_file_rejects_path_replacement_with_stale_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.jsonl"
            replacement_path = Path(directory) / "replacement.jsonl"
            original = b'{"id":"one","schema":"fixture.schema.v1"}\n'
            replacement = b'{"id":"two","schema":"fixture.schema.v1"}\n'
            path.write_bytes(original)
            replacement_path.write_bytes(replacement)
            expected = {
                "path": path.name,
                "schema": "fixture.schema.v1",
                "rows": 1,
                "bytes": len(original),
                "sha256": hashlib.sha256(original).hexdigest(),
            }
            real_hash = catalog_builder._sha256_descriptor
            calls = 0

            def replace_after_initial_hash(value):
                nonlocal calls
                digest = real_hash(value)
                calls += 1
                if calls == 1:
                    os.replace(replacement_path, path)
                return digest

            with mock.patch.object(
                catalog_builder,
                "_sha256_descriptor",
                side_effect=replace_after_initial_hash,
            ), mock.patch.object(
                catalog_builder, "_artifact_identity", return_value=(1, 2, 3, 4, 5)
            ):
                with self.assertRaisesRegex(
                    ValueError, "catalog artifact changed while it was read"
                ):
                    catalog_builder._validate_artifact_file(path, expected)

        self.assertEqual(calls, 3)

    def test_validator_hashes_bytes_seen_during_transient_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.jsonl"
            original = b'{"id":"one","schema":"fixture.schema.v1"}\n'
            replacement = b'{"id":"two","schema":"fixture.schema.v1"}\n'
            path.write_bytes(original)
            expected = {
                "path": path.name,
                "schema": "fixture.schema.v1",
                "rows": 1,
                "bytes": len(original),
                "sha256": hashlib.sha256(original).hexdigest(),
            }
            real_iter = catalog_builder._iter_jsonl_descriptor

            def rewrite_then_restore(descriptor, value, **kwargs):
                path.write_bytes(replacement)
                iterator = real_iter(descriptor, value, **kwargs)
                try:
                    row = next(iterator)
                finally:
                    path.write_bytes(original)
                yield row
                yield from iterator

            with mock.patch.object(
                catalog_builder,
                "_iter_jsonl_descriptor",
                side_effect=rewrite_then_restore,
            ), mock.patch.object(
                catalog_builder, "_artifact_identity", return_value=(1, 2, 3, 4, 5)
            ):
                with self.assertRaisesRegex(
                    ValueError, "catalog artifact changed while it was read"
                ):
                    catalog_builder._validate_artifact(
                        path, expected, verify_paths=False
                    )

    def test_read_artifact_snapshot_returns_pinned_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            payload = b"schema: fixture.schema.v1\n"
            path.write_bytes(payload)
            expected = {
                "path": path.name,
                "schema": "fixture.schema.v1",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            self.assertEqual(
                catalog_builder._read_artifact_snapshot(path, expected), payload
            )

    def test_jsonl_iterator_closes_duplicate_when_fdopen_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.jsonl"
            path.write_bytes(b'{"id":"one"}\n')
            descriptor = path.open("rb")
            duplicate = descriptor.fileno()
            leaked = os.dup(duplicate)
            try:
                with mock.patch.object(
                    catalog_builder.os, "dup", return_value=leaked
                ), mock.patch.object(
                    catalog_builder.os,
                    "fdopen",
                    side_effect=OSError("fdopen fixture failure"),
                ):
                    with self.assertRaisesRegex(OSError, "fdopen fixture failure"):
                        list(catalog_builder._iter_jsonl_descriptor(duplicate, path))
                with self.assertRaises(OSError):
                    os.fstat(leaked)
            finally:
                try:
                    os.close(leaked)
                except OSError:
                    pass
                descriptor.close()


class ConfigTest(unittest.TestCase):
    def test_config_rejects_boolean_counts_in_frozen_contract(self):
        config_path = ROOT / "eval-pipeline/configs/data_catalog_v1.yaml"
        config = load_config(config_path)
        mutations = (
            ("protected expected_prompts", lambda value: value["protected_prompt_sources"][0].__setitem__("expected_prompts", True)),
            ("protected expected_bytes", lambda value: value["protected_prompt_sources"][0].__setitem__("expected_bytes", True)),
            ("dataset expected_rows", lambda value: value["datasets"][0].__setitem__("expected_rows", True)),
            ("dataset statistic", lambda value: value["datasets"][0]["expected_stats"].__setitem__("rows_with_prompt", True)),
            ("training aggregate rows", lambda value: value["expected_training_candidate_stats"].__setitem__("rows", True)),
            ("protected unique prompts", lambda value: value.__setitem__("expected_protected_normalized_unique_prompts", True)),
            ("training view rows", lambda value: value["training_views"][0].__setitem__("expected_rows", True)),
            ("dataset candidate flag string", lambda value: value["datasets"][0].__setitem__("include_in_training_candidates", "false")),
            ("dataset candidate flag integer", lambda value: value["datasets"][0].__setitem__("include_in_training_candidates", 1)),
            ("split eligibility string", lambda value: value["datasets"][2]["splits"][0].__setitem__("training_eligible", "false")),
            ("view filter string", lambda value: value["training_views"][0]["filter"].__setitem__("prompt_is_not_null", "false")),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                altered = copy.deepcopy(config)
                mutate(altered)
                with self.assertRaises(ValueError):
                    _load_config_bytes(
                        yaml.safe_dump(altered).encode("utf-8"), source=label
                    )

    def test_versioned_config_names_exactly_five_requested_directories(self):
        config = load_config(ROOT / "eval-pipeline/configs/data_catalog_v1.yaml")
        self.assertEqual(
            {row["id"] for row in config["physical_sources"]},
            REQUIRED_SOURCE_ROOT_IDS,
        )
        self.assertEqual(len(config["physical_sources"]), 5)
        self.assertNotIn(
            "training_candidates.jsonl",
            {row["output"] for row in config["datasets"]},
        )
        pixel_render = next(
            row for row in config["datasets"] if row["id"] == "pixel_render_image_only"
        )
        self.assertEqual(pixel_render["expected_stats"]["missing_images"], 1_770_989)
        self.assertFalse(pixel_render["include_in_training_candidates"])
        self.assertEqual(config["expected_training_candidate_stats"]["rows"], 313_094)
        self.assertEqual(
            {
                row["id"]
                for row in config["protected_prompt_sources"]
                if row["id"].startswith("aesthetic_4k_eval_")
            },
            {"aesthetic_4k_eval_2048", "aesthetic_4k_eval_4096"},
        )
        protected_sources = {
            row["id"]: row for row in config["protected_prompt_sources"]
        }
        self.assertEqual(
            {
                source_id: (
                    protected_sources[source_id]["expected_prompts"],
                    protected_sources[source_id]["expected_unique_images"],
                    protected_sources[source_id]["expected_image_shards"],
                )
                for source_id in ("four_k_lsdb_validation", "four_k_lsdb_test")
            },
            {
                "four_k_lsdb_validation": (3_174, 2_000, 4),
                "four_k_lsdb_test": (3_162, 1_984, 4),
            },
        )
        self.assertEqual(config["expected_holdout_stats"]["rows"], 49_393)
        self.assertEqual(config["expected_protected_unique_images"], 37_160)
        for source_id in ("sana_pickscore_prompts", "sana_ocr_prompts"):
            source = next(row for row in config["datasets"] if row["id"] == source_id)
            self.assertFalse(source["include_in_training_candidates"])
            self.assertEqual(source["expected_stats"]["training_eligible_rows"], 0)

    def test_artifact_contract_rejects_missing_or_self_declared_inventory(self):
        config = load_config(ROOT / "eval-pipeline/configs/data_catalog_v1.yaml")
        contract = _artifact_contract(config)
        with self.assertRaisesRegex(ValueError, "inventory length"):
            _assert_artifact_contract(contract[:-1], config)
        altered = [dict(row) for row in contract]
        altered[-1]["rows"] = 0
        with self.assertRaisesRegex(ValueError, "expected 5"):
            _assert_artifact_contract(altered, config)

    def test_unverified_build_cannot_be_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "requires --allow-dirty"):
                build_catalog(
                    config_path=ROOT / "eval-pipeline/configs/data_catalog_v1.yaml",
                    output_dir=Path(directory),
                    verify_paths=False,
                    allow_dirty=False,
                )

    def test_formal_build_still_requires_head_to_equal_upstream_tip(self):
        state = {
            "commit": "a" * 40,
            "dirty": False,
            "pushed": False,
            "upstream_commit": "b" * 40,
        }
        with mock.patch.object(catalog_builder, "git_state", return_value=state):
            with self.assertRaisesRegex(RuntimeError, "pushed upstream commit"):
                enforce_git_gate(ROOT, allow_dirty=False)


class GitProvenanceTest(unittest.TestCase):
    def _git(self, repository: Path, *arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()

    def _commit(self, repository: Path, content: str, message: str) -> str:
        path = repository / "payload.txt"
        path.write_text(content, encoding="utf-8")
        self._git(repository, "add", path.name)
        self._git(repository, "commit", "-m", message)
        return self._git(repository, "rev-parse", "HEAD")

    def _repository_with_recorded_commit(self, root: Path) -> tuple[Path, str, str]:
        repository = root / "repository"
        repository.mkdir()
        self._git(repository, "init")
        self._git(repository, "config", "user.email", "catalog-test@example.invalid")
        self._git(repository, "config", "user.name", "Catalog Test")
        base = self._commit(repository, "base\n", "base")
        recorded = self._commit(repository, "recorded\n", "recorded")
        self._git(repository, "update-ref", "refs/remotes/origin/catalog", recorded)
        return repository, base, recorded

    def test_recorded_commit_at_upstream_tip_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _, recorded = self._repository_with_recorded_commit(
                Path(directory)
            )
            _validate_recorded_upstream_ancestry(
                recorded,
                "refs/remotes/origin/catalog",
                repository_root=repository,
            )

    def test_recorded_commit_survives_normal_upstream_fast_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _, recorded = self._repository_with_recorded_commit(
                Path(directory)
            )
            descendant = self._commit(repository, "descendant\n", "descendant")
            self._git(repository, "update-ref", "refs/remotes/origin/catalog", descendant)
            _validate_recorded_upstream_ancestry(
                recorded,
                "refs/remotes/origin/catalog",
                repository_root=repository,
            )

    def test_recorded_commit_fails_after_upstream_history_diverges(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, base, recorded = self._repository_with_recorded_commit(
                Path(directory)
            )
            self._git(repository, "switch", "--detach", base)
            divergent = self._commit(repository, "divergent\n", "divergent")
            self._git(repository, "update-ref", "refs/remotes/origin/catalog", divergent)
            with self.assertRaisesRegex(ValueError, "no longer reachable"):
                _validate_recorded_upstream_ancestry(
                    recorded,
                    "refs/remotes/origin/catalog",
                    repository_root=repository,
                )

    def test_recorded_commit_fails_when_upstream_ref_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _, recorded = self._repository_with_recorded_commit(
                Path(directory)
            )
            with self.assertRaisesRegex(ValueError, "no longer reachable"):
                _validate_recorded_upstream_ancestry(
                    recorded,
                    "refs/remotes/origin/missing",
                    repository_root=repository,
                )

    def test_recorded_commit_rejects_ambiguous_abbreviated_upstream_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _, recorded = self._repository_with_recorded_commit(
                Path(directory)
            )
            self._git(repository, "update-ref", "refs/heads/origin/catalog", recorded)
            with self.assertRaisesRegex(ValueError, "fully qualified"):
                _validate_recorded_upstream_ancestry(
                    recorded,
                    "origin/catalog",
                    repository_root=repository,
                )

    def test_builder_file_set_is_resolved_at_the_recorded_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            package = repository / "eval-pipeline/data_catalog"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            entrypoint = repository / "eval-pipeline/build_data_catalog.py"
            entrypoint.write_text("# entrypoint\n", encoding="utf-8")
            self._git(repository, "init")
            self._git(repository, "config", "user.email", "catalog-test@example.invalid")
            self._git(repository, "config", "user.name", "Catalog Test")
            self._git(repository, "add", "eval-pipeline")
            self._git(repository, "commit", "-m", "recorded builder")
            recorded_commit = self._git(repository, "rev-parse", "HEAD")
            recorded = _builder_provenance(repository)

            future_builder = package / "future/nested_builder.py"
            future_builder.parent.mkdir()
            future_builder.write_text("# future\n", encoding="utf-8")
            self._git(repository, "add", "eval-pipeline")
            self._git(repository, "commit", "-m", "future builder")
            current_commit = self._git(repository, "rev-parse", "HEAD")

            _validate_builder_provenance(
                _builder_provenance(repository),
                repository_root=repository,
                commit=current_commit,
            )

            _validate_builder_provenance(
                recorded,
                repository_root=repository,
                commit=recorded_commit,
            )
            with self.assertRaisesRegex(ValueError, "exact builder file set"):
                _validate_builder_provenance(
                    recorded[:-1],
                    repository_root=repository,
                    commit=recorded_commit,
                )

    def test_formal_git_record_requires_recorded_upstream_tip_equality(self):
        commit = "a" * 40
        record = {
            "commit": commit,
            "dirty": False,
            "pushed": True,
            "upstream_commit": commit,
            "upstream_ref": "refs/remotes/origin/catalog",
        }
        _validate_formal_git_record(record)
        abbreviated = {**record, "upstream_ref": "origin/catalog"}
        with self.assertRaisesRegex(ValueError, "internally inconsistent"):
            _validate_formal_git_record(abbreviated)
        inconsistent = {**record, "upstream_commit": "b" * 40}
        with self.assertRaisesRegex(ValueError, "internally inconsistent"):
            _validate_formal_git_record(inconsistent)

    def test_git_gate_uses_real_tracking_branch_for_equal_ahead_and_behind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote.git"
            repository = root / "repository"
            peer = root / "peer"
            self._git(root, "init", "--bare", str(remote))
            repository.mkdir()
            self._git(repository, "init")
            self._git(repository, "config", "user.email", "catalog-test@example.invalid")
            self._git(repository, "config", "user.name", "Catalog Test")
            self._git(repository, "branch", "-M", "catalog")
            self._git(repository, "remote", "add", "origin", str(remote))
            self._commit(repository, "equal\n", "equal")
            self._git(repository, "push", "-u", "origin", "catalog")

            self.assertTrue(git_state(repository)["pushed"])
            self.assertEqual(
                git_state(repository)["upstream_ref"],
                "refs/remotes/origin/catalog",
            )
            enforce_git_gate(repository, allow_dirty=False)

            self._commit(repository, "ahead\n", "ahead")
            self.assertFalse(git_state(repository)["pushed"])
            with self.assertRaisesRegex(RuntimeError, "pushed upstream commit"):
                enforce_git_gate(repository, allow_dirty=False)
            self._git(repository, "push", "origin", "catalog")

            self._git(root, "clone", "--branch", "catalog", str(remote), str(peer))
            self._git(peer, "config", "user.email", "catalog-test@example.invalid")
            self._git(peer, "config", "user.name", "Catalog Test")
            self._commit(peer, "behind\n", "behind")
            self._git(peer, "push", "origin", "catalog")
            self._git(repository, "fetch", "origin", "catalog")

            self.assertFalse(git_state(repository)["pushed"])
            with self.assertRaisesRegex(RuntimeError, "pushed upstream commit"):
                enforce_git_gate(repository, allow_dirty=False)


class ReleaseValidationTest(unittest.TestCase):
    def _write_manifest(self, root: Path, core: dict) -> Path:
        release_id = _release_id(core)
        release = root / release_id
        release.mkdir()
        (release / "manifest.json").write_text(
            json.dumps(
                {"schema": CATALOG_SCHEMA, "release_id": release_id, **core},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return release

    def _write_candidate_shell(self, root: Path) -> Path:
        config_bytes = (ROOT / "eval-pipeline/configs/data_catalog_v1.yaml").read_bytes()
        snapshot = {
            "path": CONFIG_SNAPSHOT_NAME,
            "schema": CONFIG_SNAPSHOT_SCHEMA,
            "bytes": len(config_bytes),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        }
        config = load_config(ROOT / "eval-pipeline/configs/data_catalog_v1.yaml")
        core = {
            "complete": False,
            "candidate_catalog_complete": False,
            "development_build": True,
            "record_schema": DATA_RECORD_SCHEMA,
            "config_repo_path": None,
            "config_snapshot": snapshot,
            "git": {"dirty": True, "pushed": False},
            "verify_paths": False,
            "physical_source_count": 5,
            "training_ready": False,
            "payload_integrity_policy": config["payload_integrity_policy"],
            "builder_provenance": _builder_provenance(REPOSITORY_ROOT),
            "artifacts": [],
        }
        release = self._write_manifest(root, core)
        (release / CONFIG_SNAPSHOT_NAME).write_bytes(config_bytes)
        return release

    def test_release_id_binds_artifact_hashes(self):
        first = {"artifacts": [{"path": "rows.jsonl", "sha256": "a" * 64}]}
        second = {"artifacts": [{"path": "rows.jsonl", "sha256": "b" * 64}]}
        self.assertNotEqual(_release_id(first), _release_id(second))

    def test_new_release_is_removed_when_post_rename_validation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogs = Path(directory) / "catalogs"
            staging = catalogs / ".catalog-build-fixture"
            release = catalogs / "catalog-fixture"
            staging.mkdir(parents=True)
            (staging / "manifest.json").write_text("{}\n", encoding="utf-8")

            with mock.patch.object(
                catalog_builder,
                "validate_release",
                side_effect=ValueError("post-rename validation failed"),
            ):
                with self.assertRaisesRegex(ValueError, "post-rename"):
                    catalog_builder._install_catalog_release(
                        staging,
                        release,
                        catalogs,
                        {},
                        verify_paths=False,
                        require_formal_catalog=False,
                    )

            self.assertFalse(staging.exists())
            self.assertFalse(release.exists())

    def test_new_release_is_removed_when_install_fsync_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogs = Path(directory) / "catalogs"
            staging = catalogs / ".catalog-build-fixture"
            release = catalogs / "catalog-fixture"
            staging.mkdir(parents=True)
            (staging / "manifest.json").write_text("{}\n", encoding="utf-8")

            with (
                mock.patch.object(
                    catalog_builder,
                    "_fsync_directory",
                    side_effect=[OSError("install fsync failed"), None],
                ),
                mock.patch.object(catalog_builder, "validate_release") as validate,
            ):
                with self.assertRaisesRegex(OSError, "install fsync failed"):
                    catalog_builder._install_catalog_release(
                        staging,
                        release,
                        catalogs,
                        {},
                        verify_paths=False,
                        require_formal_catalog=False,
                    )

            validate.assert_not_called()
            self.assertFalse(staging.exists())
            self.assertFalse(release.exists())

    def test_cleanup_fsync_does_not_replace_primary_validation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogs = Path(directory) / "catalogs"
            staging = catalogs / ".catalog-build-fixture"
            release = catalogs / "catalog-fixture"
            staging.mkdir(parents=True)
            (staging / "manifest.json").write_text("{}\n", encoding="utf-8")

            with (
                mock.patch.object(
                    catalog_builder,
                    "_fsync_directory",
                    side_effect=[None, OSError("rollback fsync failed")],
                ),
                mock.patch.object(
                    catalog_builder,
                    "validate_release",
                    side_effect=ValueError("post-rename validation failed"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "post-rename") as caught:
                    catalog_builder._install_catalog_release(
                        staging,
                        release,
                        catalogs,
                        {},
                        verify_paths=False,
                        require_formal_catalog=False,
                    )

            self.assertIsInstance(caught.exception.__cause__, OSError)
            self.assertIn("rollback fsync failed", str(caught.exception.__cause__))
            self.assertFalse(staging.exists())
            self.assertFalse(release.exists())

    def test_validation_rejects_release_without_frozen_config_snapshot(self):
        core = {
            "complete": False,
            "verify_paths": False,
            "git": {"dirty": True, "pushed": False},
            "record_schema": DATA_RECORD_SCHEMA,
            "physical_source_count": 5,
            "artifacts": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            release = self._write_manifest(Path(directory), core)
            with self.assertRaisesRegex(ValueError, "config snapshot"):
                validate_release(
                    release,
                    verify_paths=False,
                    require_formal_catalog=True,
                )

    def test_validation_rejects_manifest_changed_after_release_naming(self):
        core = {
            "complete": False,
            "verify_paths": False,
            "git": {"dirty": True, "pushed": False},
            "record_schema": DATA_RECORD_SCHEMA,
            "physical_source_count": 5,
            "artifacts": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            release = self._write_manifest(Path(directory), core)
            path = release / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["physical_source_count"] = 4
            path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "release_id does not match"):
                validate_release(release, verify_paths=False)

    def test_development_release_cannot_self_declare_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            release = self._write_candidate_shell(Path(directory))
            with self.assertRaisesRegex(ValueError, "complete candidate catalog"):
                validate_release(
                    release,
                    verify_paths=True,
                    require_formal_catalog=True,
                )

    def test_candidate_catalog_cannot_authorize_training(self):
        with tempfile.TemporaryDirectory() as directory:
            release = self._write_candidate_shell(Path(directory))
            with self.assertRaisesRegex(
                ValueError, "training-ready validation requires a formal catalog"
            ):
                validate_release(
                    release,
                    verify_paths=False,
                    require_training_ready=True,
                )

    def test_guarded_build_rejects_metadata_only_config_before_build(self):
        with mock.patch.object(catalog_builder, "build_catalog") as build:
            with self.assertRaisesRegex(
                RuntimeError, "cannot be combined with --allow-dirty"
            ):
                catalog_builder.main(
                    [
                        "--allow-dirty",
                        "--require-training-ready",
                        "--config",
                        str(ROOT / "eval-pipeline/configs/data_catalog_v1.yaml"),
                    ]
                )
            build.assert_not_called()

    def test_training_candidates_must_be_byte_for_byte_derivation(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            rows = [
                make_record(
                    source="fixture",
                    stable_key=str(index),
                    source_roots=("fixture",),
                    split="train",
                    prompt=f"prompt {index}",
                    image_path=None,
                    license_name="test",
                    license_status="approved",
                    modality="prompt",
                    intended_use=("rl_prompt_pool",),
                    training_eligible=True,
                )
                for index in range(2)
            ]
            source = release / "source.jsonl"
            candidates = release / "training_candidates.jsonl"
            source.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
            candidates.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
            config = {
                "training_license_policy": {"allowed_statuses": ["approved"]},
                "expected_training_candidate_stats": {"rows": 2},
                "datasets": [
                    {
                        "output": source.name,
                        "include_in_training_candidates": True,
                    }
                ],
            }
            _compare_training_candidate_derivation(release, config)
            candidates.write_bytes(
                canonical_json_bytes(rows[1]) + canonical_json_bytes(rows[0])
            )
            with self.assertRaisesRegex(ValueError, "byte-for-byte"):
                _compare_training_candidate_derivation(release, config)


if __name__ == "__main__":
    unittest.main()
