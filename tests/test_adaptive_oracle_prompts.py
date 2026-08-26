import csv
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
PROMPT_DIR = EVAL_PIPELINE / "prompts"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "adaptive_oracle_prompt_builder_test",
    EVAL_PIPELINE / "build_adaptive_oracle_prompts.py",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class AdaptiveOraclePromptBuilderTest(unittest.TestCase):
    manifest_path = PROMPT_DIR / builder.OUTPUT_MANIFEST
    csv_path = PROMPT_DIR / builder.OUTPUT_CSV
    inventory_path = PROMPT_DIR / builder.OUTPUT_EXCLUSION_INVENTORY

    def test_nfkc_whitespace_and_casefold_normalization(self):
        self.assertEqual(
            builder.normalize_prompt("  ＣＡＦＥ\u0301\tOn\nA  STRASSE  "),
            "café on a strasse",
        )
        self.assertEqual(
            builder.normalize_prompt("Straße"),
            builder.normalize_prompt("STRASSE"),
        )

    def test_source_bytes_and_revision_drift_fail_closed(self):
        source_bytes = (
            b"Prompt\tCategory\tChallenge\tNote\n"
            b"a prompt\tcategory\tchallenge\tnote\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "PartiPrompts.tsv"
            source.write_bytes(source_bytes)
            with mock.patch.object(
                builder, "SOURCE_SHA256", hashlib.sha256(source_bytes).hexdigest()
            ):
                self.assertEqual(builder._source_rows(source)[0]["prompt"], "a prompt")
                source.write_bytes(source_bytes + b"drift")
                with self.assertRaisesRegex(ValueError, "frozen source hash"):
                    builder._source_rows(source)

        with mock.patch.object(
            builder.subprocess,
            "check_output",
            return_value="wrong-revision\n",
        ):
            with self.assertRaisesRegex(ValueError, "differs from"):
                builder._verify_source_revision(pathlib.Path("/unused"))

    def test_prompt_collision_fails_when_a_challenge_is_exhausted(self):
        source_rows = [
            {
                "source_row": index,
                "prompt": f"prompt {index}",
                "category": "category",
                "challenge": f"challenge-{index:02d}",
                "note": "",
            }
            for index in range(builder.EXPECTED_CHALLENGE_COUNT)
        ]
        with self.assertRaisesRegex(ValueError, "lack unused prompts"):
            builder.select_engineering_prompts(
                source_rows,
                {builder.normalize_prompt(source_rows[0]["prompt"])},
                set(),
            )

    def test_seed_overlap_advances_to_the_first_unused_candidate(self):
        first_seed, first_digest = builder._seed_candidate(0)
        self.assertGreater(first_seed, 0)
        selected = builder.select_engineering_seed({first_seed})
        self.assertNotEqual(selected["seed"], first_seed)
        self.assertGreater(selected["counter"], 0)
        self.assertNotEqual(selected["selection_digest"], first_digest)
        self.assertTrue(selected["retired_on_use"])
        self.assertEqual(selected["status"], "reserved_retired_on_use")

    def test_inventory_projects_metadata_and_skips_outcome_values(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            prompt_dir = repo / "eval-pipeline" / "prompts"
            config_dir = repo / "eval-pipeline" / "configs"
            outputs = repo / "outputs"
            sidecars = outputs / "run" / "images"
            records = outputs / "run" / "records"
            for path in (prompt_dir, config_dir, sidecars, records):
                path.mkdir(parents=True, exist_ok=True)

            (prompt_dir / "prior.csv").write_text(
                "index,TEXT,source_row,seed\n0,ＣＡＦＥ́,7,11\n",
                encoding="utf-8",
            )
            (prompt_dir / "prior_manifest.json").write_text(
                json.dumps(
                    {
                        "prompt": "manifest prompt",
                        "source_row": 8,
                        "reserved_seeds": [101],
                    }
                ),
                encoding="utf-8",
            )
            (config_dir / "registration.yaml").write_text(
                "analysis:\n  bootstrap_seed: 202\n",
                encoding="utf-8",
            )
            (outputs / "run" / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "prompt": "output manifest prompt",
                        "seed": 303,
                        "scores": {"prompt": "hidden score prompt", "seed": 999},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (sidecars / "record.json").write_text(
                json.dumps(
                    {
                        "prompt": "sidecar prompt",
                        "seed": 404,
                        "quality": {"prompt": "hidden quality prompt", "seed": 998},
                    }
                ),
                encoding="utf-8",
            )
            (records / "task.json").write_text(
                json.dumps(
                    {
                        "prompt": "records sidecar prompt",
                        "source_row": 9,
                        "seed": 505,
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "run" / "attempt.json").write_text(
                json.dumps(
                    {
                        "prompt": "attempt prompt",
                        "source_row": 10,
                        "seed": 606,
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "run" / "scores.jsonl").write_text(
                json.dumps({"prompt": "forbidden file prompt", "seed": 997}) + "\n",
                encoding="utf-8",
            )

            inventory = builder.scan_inventory(repo, prompt_dir, outputs)

        self.assertEqual(
            inventory["normalized_prompts"],
            {
                "café",
                "hidden quality prompt",
                "hidden score prompt",
                "manifest prompt",
                "output manifest prompt",
                "records sidecar prompt",
                "sidecar prompt",
                "attempt prompt",
            },
        )
        self.assertEqual(inventory["source_rows"], {7, 8, 9, 10})
        self.assertEqual(
            inventory["used_reserved_seeds"],
            {11, 101, 202, 303, 404, 505, 606, 998, 999},
        )
        self.assertEqual(inventory["manifest"]["forbidden_output_path_count"], 1)
        self.assertEqual(
            inventory["manifest"]["outcome_container_count_identity_only"], 2
        )
        self.assertEqual(
            inventory["manifest"]["output_metadata_files_parsed_for_identity_inventory"],
            4,
        )
        self.assertEqual(
            inventory["manifest"]["dedicated_score_or_quality_files_read"], 0
        )
        self.assertEqual(inventory["manifest"]["outcome_field_values_consumed"], 0)

    def test_inventory_excludes_only_registered_self_metadata_names(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            prompt_dir = repo / "eval-pipeline" / "prompts"
            config_dir = repo / "eval-pipeline" / "configs"
            outputs = repo / "outputs"
            for path in (prompt_dir, config_dir, outputs):
                path.mkdir(parents=True, exist_ok=True)

            for name in builder.SELF_METADATA_NAMES:
                (config_dir / name).write_text(
                    "seed: 120129238\nsource_row: 991\n",
                    encoding="utf-8",
                )
            (config_dir / "unrelated_registration.yaml").write_text(
                "seed: 73\nsource_row: 992\n",
                encoding="utf-8",
            )

            inventory = builder.scan_inventory(repo, prompt_dir, outputs)

        self.assertNotIn(120129238, inventory["used_reserved_seeds"])
        self.assertNotIn(991, inventory["source_rows"])
        self.assertIn(73, inventory["used_reserved_seeds"])
        self.assertIn(992, inventory["source_rows"])
        self.assertEqual(
            inventory["manifest"]["roles"]["repository_config"]["file_count"],
            1,
        )

    def test_nested_self_names_are_scanned_as_historical_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            prompt_dir = repo / "eval-pipeline" / "prompts"
            config_dir = repo / "eval-pipeline" / "configs"
            outputs = repo / "outputs"
            nested_prompt_dir = prompt_dir / "historical"
            nested_config_dir = config_dir / "historical"
            for path in (
                prompt_dir,
                config_dir,
                outputs,
                nested_prompt_dir,
                nested_config_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)

            (prompt_dir / builder.OUTPUT_CSV).write_text(
                "index,TEXT,source_row,seed\n0,own prompt,991,120129238\n",
                encoding="utf-8",
            )
            (nested_prompt_dir / builder.OUTPUT_CSV).write_text(
                "index,TEXT,source_row,seed\n0,historical prompt,992,73\n",
                encoding="utf-8",
            )
            self_name = sorted(builder.SELF_METADATA_NAMES)[0]
            (config_dir / self_name).write_text(
                "seed: 120129238\nsource_row: 991\n",
                encoding="utf-8",
            )
            (nested_config_dir / self_name).write_text(
                "seed: 101\nsource_row: 993\n",
                encoding="utf-8",
            )

            inventory = builder.scan_inventory(repo, prompt_dir, outputs)

        self.assertNotIn("own prompt", inventory["normalized_prompts"])
        self.assertNotIn(991, inventory["source_rows"])
        self.assertNotIn(120129238, inventory["used_reserved_seeds"])
        self.assertIn("historical prompt", inventory["normalized_prompts"])
        self.assertIn(992, inventory["source_rows"])
        self.assertIn(993, inventory["source_rows"])
        self.assertIn(73, inventory["used_reserved_seeds"])
        self.assertIn(101, inventory["used_reserved_seeds"])

    def test_duplicate_metadata_keys_fail_closed(self):
        for name, payload in (
            ("duplicate.json", '{"seed": 7, "seed": 9}\n'),
            ("duplicate.yaml", "seed: 7\nseed: 9\n"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                repo = pathlib.Path(directory)
                prompt_dir = repo / "eval-pipeline" / "prompts"
                config_dir = repo / "eval-pipeline" / "configs"
                outputs = repo / "outputs"
                for path in (prompt_dir, config_dir, outputs):
                    path.mkdir(parents=True, exist_ok=True)
                (config_dir / name).write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "cannot parse metadata"):
                    builder.scan_inventory(repo, prompt_dir, outputs)

    def test_camel_case_identity_keys_are_projected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            prompt_dir = repo / "eval-pipeline" / "prompts"
            config_dir = repo / "eval-pipeline" / "configs"
            outputs = repo / "outputs"
            for path in (prompt_dir, config_dir, outputs):
                path.mkdir(parents=True, exist_ok=True)
            (config_dir / "camel.json").write_text(
                json.dumps(
                    {
                        "positivePrompt": "camel prompt",
                        "sourceRow": 77,
                        "reservedSeeds": [101, 202],
                    }
                ),
                encoding="utf-8",
            )

            inventory = builder.scan_inventory(repo, prompt_dir, outputs)

        self.assertIn("camel prompt", inventory["normalized_prompts"])
        self.assertIn(77, inventory["source_rows"])
        self.assertTrue({101, 202}.issubset(inventory["used_reserved_seeds"]))

    def test_inventory_excludes_only_the_registered_self_output_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            prompt_dir = repo / "eval-pipeline" / "prompts"
            config_dir = repo / "eval-pipeline" / "configs"
            outputs = repo / "outputs"
            own_records = outputs / "adaptive_oracle" / "engineering_v1" / "records"
            other_records = outputs / "other" / "records"
            for path in (prompt_dir, config_dir, own_records, other_records):
                path.mkdir(parents=True, exist_ok=True)
            (own_records / "own.json").write_text(
                '{"prompt":"own prompt","source_row":991,"seed":120129238}\n',
                encoding="utf-8",
            )
            (other_records / "other.json").write_text(
                '{"prompt":"other prompt","source_row":992,"seed":73}\n',
                encoding="utf-8",
            )

            inventory = builder.scan_inventory(repo, prompt_dir, outputs)

        self.assertNotIn("own prompt", inventory["normalized_prompts"])
        self.assertNotIn(991, inventory["source_rows"])
        self.assertNotIn(120129238, inventory["used_reserved_seeds"])
        self.assertIn("other prompt", inventory["normalized_prompts"])
        self.assertIn(992, inventory["source_rows"])
        self.assertIn(73, inventory["used_reserved_seeds"])

    def test_asset_tampering_is_detected_byte_for_byte(self):
        assets = {
            builder.OUTPUT_CSV: b"csv-bytes\n",
            builder.OUTPUT_MANIFEST: b"manifest-bytes\n",
            builder.OUTPUT_EXCLUSION_INVENTORY: b"inventory-bytes\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            prompt_dir = pathlib.Path(directory)
            for name, value in assets.items():
                (prompt_dir / name).write_bytes(value)
            self.assertEqual(builder.asset_mismatches(assets, prompt_dir), [])
            (prompt_dir / builder.OUTPUT_CSV).write_bytes(b"tampered\n")
            self.assertEqual(
                builder.asset_mismatches(assets, prompt_dir),
                [builder.OUTPUT_CSV],
            )

    def test_asset_writer_is_one_shot_and_never_replaces_existing_bytes(self):
        assets = {
            builder.OUTPUT_CSV: b"csv-bytes\n",
            builder.OUTPUT_MANIFEST: b"manifest-bytes\n",
            builder.OUTPUT_EXCLUSION_INVENTORY: b"inventory-bytes\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            prompt_dir = pathlib.Path(directory)
            builder.write_assets_once(assets, prompt_dir)
            self.assertEqual(
                {name: (prompt_dir / name).read_bytes() for name in assets},
                assets,
            )
            with self.assertRaisesRegex(FileExistsError, "already exist"):
                builder.write_assets_once(
                    {name: b"replacement\n" for name in assets}, prompt_dir
                )
            self.assertEqual(
                {name: (prompt_dir / name).read_bytes() for name in assets},
                assets,
            )
            self.assertEqual(list(prompt_dir.glob(".*.tmp")), [])

    def test_frozen_manifest_and_csv_contract(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        inventory = json.loads(self.inventory_path.read_text(encoding="utf-8"))
        rows = read_csv(self.csv_path)
        self.assertEqual(manifest["schema"], "adaptive_oracle_prompt_manifest_v1")
        self.assertEqual(manifest["status"], "registration_only_gpu_not_authorized")
        self.assertEqual(manifest["source"]["revision"], builder.SOURCE_REVISION)
        self.assertEqual(manifest["source"]["file_sha256"], builder.SOURCE_SHA256)
        self.assertEqual(inventory["schema"], builder.EXCLUSION_INVENTORY_SCHEMA)
        self.assertEqual(
            manifest["exclusion_inventory"],
            {
                "schema": builder.EXCLUSION_INVENTORY_SCHEMA,
                "path": (
                    "eval-pipeline/prompts/"
                    + builder.OUTPUT_EXCLUSION_INVENTORY
                ),
                "sha256": hashlib.sha256(self.inventory_path.read_bytes()).hexdigest(),
            },
        )
        self.assertEqual(len(rows), builder.EXPECTED_CHALLENGE_COUNT)
        self.assertEqual(
            len({row["source_challenge"] for row in rows}),
            builder.EXPECTED_CHALLENGE_COUNT,
        )
        self.assertEqual([int(row["index"]) for row in rows], list(range(len(rows))))
        self.assertTrue(all(row["split"] == "engineering" for row in rows))
        self.assertEqual(
            hashlib.sha256(self.csv_path.read_bytes()).hexdigest(),
            manifest["engineering"]["csv_sha256"],
        )
        selected_seed = manifest["seed_registration"]["engineering_seed"]
        self.assertGreater(selected_seed["seed"], 0)
        self.assertTrue(selected_seed["retired_on_use"])
        self.assertEqual({int(row["seed"]) for row in rows}, {selected_seed["seed"]})
        normalized_digests = [
            hashlib.sha256(builder.normalize_prompt(row["TEXT"]).encode("utf-8")).hexdigest()
            for row in rows
        ]
        self.assertEqual(
            normalized_digests,
            manifest["engineering"]["normalized_prompt_digests"],
        )
        self.assertEqual(
            manifest["collisions"],
            {
                "normalized_prompts": [],
                "source_rows": [],
                "used_or_reserved_seeds": [],
            },
        )
        self.assertEqual(
            manifest["inventory"]["dedicated_score_or_quality_files_read"], 0
        )
        self.assertEqual(manifest["inventory"]["outcome_field_values_consumed"], 0)
        self.assertEqual(
            manifest["inventory"]["parsed_identity_field_allowlist"],
            list(builder._IDENTITY_FIELD_ALLOWLIST),
        )
        self.assertEqual(
            manifest["inventory_policy"]["outcome_field_values_consumed"], 0
        )
        self.assertEqual(
            manifest["inventory_policy"]["self_metadata_excluded_names"],
            sorted(builder.SELF_METADATA_NAMES),
        )
        self.assertEqual(
            manifest["inventory_policy"]["self_output_excluded_prefixes"],
            list(builder.SELF_OUTPUT_PREFIXES),
        )
        self.assertEqual(inventory["summary"], manifest["inventory"])
        self.assertEqual(
            inventory["excluded"]["normalized_prompts"],
            sorted(set(inventory["excluded"]["normalized_prompts"])),
        )
        for name in (
            "explicit_source_rows",
            "text_derived_source_rows",
            "source_rows",
        ):
            self.assertEqual(
                inventory["excluded"][name],
                sorted(set(inventory["excluded"][name])),
            )
        self.assertEqual(
            inventory["excluded"]["source_rows"],
            sorted(
                set(inventory["excluded"]["explicit_source_rows"])
                | set(inventory["excluded"]["text_derived_source_rows"])
            ),
        )
        self.assertEqual(
            inventory["excluded"]["used_reserved_seeds"],
            sorted(set(inventory["excluded"]["used_reserved_seeds"])),
        )
        for record in inventory["files"]:
            projection = record["projection"]
            for name in (
                "normalized_prompts",
                "source_rows",
                "used_reserved_seeds",
            ):
                self.assertEqual(projection[name], sorted(set(projection[name])))

    def test_builder_reproduces_frozen_assets_in_private_snapshot(self):
        source = pathlib.Path("/tmp/parti/PartiPrompts.tsv")
        source_repo = pathlib.Path("/tmp/parti")
        self.assertTrue(source.is_file(), "frozen Parti source is required")
        self.assertTrue((source_repo / ".git").exists(), "Parti Git metadata is required")
        with tempfile.TemporaryDirectory() as directory:
            private_prompt_dir = pathlib.Path(directory) / "eval-pipeline" / "prompts"
            private_prompt_dir.mkdir(parents=True)
            for name in builder.OUTPUT_NAMES:
                (private_prompt_dir / name).write_bytes((PROMPT_DIR / name).read_bytes())
            self.assertFalse((pathlib.Path(directory) / "outputs").exists())
            assets = builder.build_assets_from_frozen_inventory(
                source,
                private_prompt_dir / builder.OUTPUT_EXCLUSION_INVENTORY,
                source_repo,
            )
            self.assertEqual(
                builder.asset_mismatches(assets, private_prompt_dir), []
            )

    def test_private_snapshot_inventory_missing_incomplete_or_duplicate_fails(self):
        source = pathlib.Path("/tmp/parti/PartiPrompts.tsv")
        source_repo = pathlib.Path("/tmp/parti")
        self.assertTrue(source.is_file(), "frozen Parti source is required")
        with tempfile.TemporaryDirectory() as directory:
            inventory_path = pathlib.Path(directory) / builder.OUTPUT_EXCLUSION_INVENTORY
            with self.assertRaisesRegex(ValueError, "frozen exclusion inventory"):
                builder.build_assets_from_frozen_inventory(
                    source, inventory_path, source_repo
                )

            original = json.loads(self.inventory_path.read_text(encoding="utf-8"))
            incomplete = json.loads(json.dumps(original))
            incomplete["files"].pop()
            inventory_path.write_bytes(builder._json_asset_bytes(incomplete))
            with self.assertRaisesRegex(ValueError, "incomplete|integrity"):
                builder.build_assets_from_frozen_inventory(
                    source, inventory_path, source_repo
                )

            duplicate = json.loads(json.dumps(original))
            record = next(
                item
                for item in duplicate["files"]
                if item["projection"]["normalized_prompts"]
            )
            prompts = record["projection"]["normalized_prompts"]
            prompts.append(prompts[0])
            prompts.sort()
            duplicate["integrity"]["files_sha256"] = builder.canonical_sha256(
                duplicate["files"]
            )
            inventory_path.write_bytes(builder._json_asset_bytes(duplicate))
            with self.assertRaisesRegex(ValueError, "normalized prompts"):
                builder.build_assets_from_frozen_inventory(
                    source, inventory_path, source_repo
                )

            for projection_name, error_label in (
                ("source_rows", "source rows"),
                ("used_reserved_seeds", "seeds"),
            ):
                duplicate_integer = json.loads(json.dumps(original))
                record = next(
                    item
                    for item in duplicate_integer["files"]
                    if item["projection"][projection_name]
                )
                values = record["projection"][projection_name]
                values.append(values[0])
                values.sort()
                duplicate_integer["integrity"]["files_sha256"] = (
                    builder.canonical_sha256(duplicate_integer["files"])
                )
                inventory_path.write_bytes(
                    builder._json_asset_bytes(duplicate_integer)
                )
                with self.assertRaisesRegex(ValueError, error_label):
                    builder.build_assets_from_frozen_inventory(
                        source, inventory_path, source_repo
                    )

            tampered = json.loads(json.dumps(original))
            tampered["files"][0]["sha256"] = "0" * 64
            tampered["integrity"]["files_sha256"] = builder.canonical_sha256(
                tampered["files"]
            )
            inventory_path.write_bytes(builder._json_asset_bytes(tampered))
            tampered_assets = builder.build_assets_from_frozen_inventory(
                source, inventory_path, source_repo
            )
            self.assertEqual(
                builder.asset_mismatches(tampered_assets, PROMPT_DIR),
                [builder.OUTPUT_EXCLUSION_INVENTORY, builder.OUTPUT_MANIFEST],
            )


if __name__ == "__main__":
    unittest.main()
