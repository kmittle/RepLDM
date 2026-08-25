import csv
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "eval-pipeline" / "prompts"
CONFIG_DIR = ROOT / "eval-pipeline" / "configs"
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "scheduler_native_fixed_headroom_builder_test",
    EVAL_PIPELINE / "build_scheduler_native_fixed_headroom_prompts.py",
)
generate = load_module(
    "scheduler_native_fixed_headroom_generate_test",
    EVAL_PIPELINE / "generate.py",
)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class SchedulerNativeFixedHeadroomRegistrationTest(unittest.TestCase):
    manifest_path = PROMPT_DIR / "scheduler_native_fixed_headroom_manifest.json"

    def setUp(self):
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    @unittest.skipUnless(
        pathlib.Path("/tmp/parti/PartiPrompts.tsv").is_file(),
        "frozen Parti checkout is unavailable",
    )
    def test_builder_reproduces_frozen_assets_and_source_revision(self):
        revision = subprocess.check_output(
            ["git", "-C", "/tmp/parti", "rev-parse", "HEAD"], text=True
        ).strip()
        self.assertEqual(revision, builder.SOURCE_REVISION)
        assets = builder.build_assets(
            pathlib.Path("/tmp/parti/PartiPrompts.tsv"),
            PROMPT_DIR,
            pathlib.Path("/tmp/parti"),
        )
        for name, expected in assets.items():
            with self.subTest(name=name):
                self.assertEqual((PROMPT_DIR / name).read_bytes(), expected)

    def test_prompt_splits_match_manifest_and_exclude_every_prior_csv(self):
        manifest_files = {
            entry["path"]: entry
            for entry in self.manifest["excluded_repository_prompt_files"]
        }
        observed_prior_files = {
            f"eval-pipeline/prompts/{path.name}"
            for path in PROMPT_DIR.glob("*.csv")
            if not path.name.startswith(builder.OUTPUT_PREFIX)
        }
        self.assertEqual(set(manifest_files), observed_prior_files)

        old_texts = set()
        old_source_rows = set()
        for relative_path, entry in manifest_files.items():
            path = ROOT / relative_path
            rows = read_csv(path)
            self.assertEqual(sha256_file(path), entry["sha256"])
            self.assertEqual(len(rows), entry["row_count"])
            old_texts.update(builder.normalize_prompt(row["TEXT"]) for row in rows)
            old_source_rows.update(
                int(row["source_row"])
                for row in rows
                if str(row.get("source_row", "")).strip()
            )
        self.assertEqual(len(old_texts), self.manifest["excluded_normalized_text_count"])

        expected_challenges = set(self.manifest["source"]["challenges"])
        seen_texts = set()
        seen_source_rows = set()
        for split, expected_count in self.manifest["counts"].items():
            path = PROMPT_DIR / f"{builder.OUTPUT_PREFIX}{split}.csv"
            rows = read_csv(path)
            self.assertEqual(len(rows), expected_count)
            self.assertEqual(
                sha256_file(path), self.manifest["csv_sha256"][path.name]
            )
            self.assertEqual(
                [int(row["index"]) for row in rows], list(range(expected_count))
            )
            self.assertTrue(all(row["split"] == split for row in rows))
            challenge_counts = {
                challenge: sum(
                    row["source_challenge"] == challenge for row in rows
                )
                for challenge in expected_challenges
            }
            self.assertEqual(set(challenge_counts), expected_challenges)
            self.assertEqual(
                set(challenge_counts.values()),
                {self.manifest["counts_per_challenge"][split]},
            )

            observed_entries = set()
            for row in rows:
                prompt = builder.normalize_prompt(row["TEXT"])
                source_row = int(row["source_row"])
                self.assertNotIn(prompt, old_texts)
                self.assertNotIn(source_row, self.manifest["excluded_source_rows"])
                self.assertNotIn(prompt, seen_texts)
                self.assertNotIn(source_row, seen_source_rows)
                seen_texts.add(prompt)
                seen_source_rows.add(source_row)
                observed_entries.add(
                    (
                        source_row,
                        hashlib.sha256(row["TEXT"].encode("utf-8")).hexdigest(),
                    )
                )
            registered_entries = {
                (entry["source_row"], entry["prompt_sha256"])
                for entry in self.manifest["splits"][split]
            }
            self.assertEqual(observed_entries, registered_entries)

        self.assertTrue(old_source_rows.issubset(set(self.manifest["excluded_source_rows"])))

    def test_seed_registration_is_reproducible_disjoint_and_unused(self):
        registration = self.manifest["seed_registration"]
        self.assertEqual(registration, builder._seed_registration())
        self.assertEqual(
            registration["observed_generated_seeds"], [0, 7, 19, 42, 73, 123]
        )
        self.assertEqual(
            registration["prior_registered_or_reserved_seeds"], [11, 29, 101]
        )
        retired = set(registration["retired_seeds"])
        selected = []
        for split, entries in registration["splits"].items():
            self.assertEqual(len(entries), builder.SEED_COUNTS[split])
            split_seeds = [entry["seed"] for entry in entries]
            self.assertTrue(retired.isdisjoint(split_seeds))
            selected.extend(split_seeds)
        self.assertEqual(len(selected), len(set(selected)))

    def test_action_registrations_are_fail_closed_and_match_manifest(self):
        expected_hashes = {
            "scheduler_native_fixed_headroom_smoke.yaml": (
                "609fbd984cb7ab17b49fc4793d6ddfbb020ce276ff3bde6deb1f1f83b7269a0a"
            ),
            "scheduler_native_fixed_headroom_development.yaml": (
                "aa42ab90e9dfc7993d0d8e5e0dcbcdfd80e3426cb3400c33dd3d2db1122fdb61"
            ),
        }
        for name, expected_hash in expected_hashes.items():
            path = CONFIG_DIR / name
            config = yaml.safe_load(path.read_text())
            self.assertEqual(sha256_file(path), expected_hash)
            self.assertEqual(config["schema"], "latent_renderer_registration_v1")
            self.assertFalse(config["authorization"]["gpu_generation"])
            self.assertEqual(
                config["source_manifest"]["sha256"], sha256_file(self.manifest_path)
            )
            provider = config.get("latent_renderer_provider") or config.get(
                "required_provider"
            )
            self.assertEqual(provider["scheduler_mapping"], "euler_clean_endpoint")
            self.assertEqual(provider["basis_normalization"], "match_rms")
            with self.assertRaisesRegex(ValueError, "registration manifests"):
                generate.load_actions(path, 50)

        development = yaml.safe_load(
            (CONFIG_DIR / "scheduler_native_fixed_headroom_development.yaml").read_text()
        )
        actions = development["actions"]
        self.assertEqual(len(actions), 8)
        primaries = [action for action in actions if action["role"] == "non_attention_primary"]
        self.assertEqual(len(primaries), 4)
        for action in primaries:
            self.assertEqual(action["required_hook_names"], [])
            self.assertEqual(len(action["requested_bases"]), 1)
            self.assertIn(
                action["requested_bases"][0],
                {"spectral_low", "spectral_mid", "spectral_high", "laplacian"},
            )
        ablations = [action for action in actions if "ablation" in action["role"]]
        self.assertEqual(len(ablations), 2)
        self.assertTrue(all(not action["selection_eligible"] for action in ablations))
        self.assertEqual(
            set(development["analysis"]["primary_holm_family"]["actions"]),
            {action["id"] for action in primaries},
        )
        self.assertEqual(
            development["required_provider"]["implementation_status"],
            "missing_blocker",
        )


if __name__ == "__main__":
    unittest.main()
