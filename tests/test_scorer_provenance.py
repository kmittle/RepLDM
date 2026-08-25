import copy
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))

import scorer_provenance as provenance  # noqa: E402


class _FixtureScorer:
    OUTPUT_KEYS = (("fixture_score", "higher"),)
    PROVENANCE_PACKAGES = ("fixture-dist",)

    def __init__(self, checkpoint, preprocessing):
        self.checkpoint = checkpoint
        self.preprocessing = preprocessing

    def provenance_metadata(self):
        return {
            "models": [
                {
                    "identifier": "fixture/model",
                    "repository_id": "fixture/model",
                    "revision": "fixture-revision",
                }
            ],
            "checkpoint_files": [
                provenance.checkpoint_file_record(
                    self.checkpoint,
                    role="fixture_checkpoint",
                    filename="fixture.bin",
                    repository_id="fixture/model",
                    revision="fixture-revision",
                )
            ],
            "preprocessing": copy.deepcopy(self.preprocessing),
            "parameters": {"scale": 1.0},
            "supporting_sources": [],
        }


class ScorerProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.fixture_version = "1.0"
        self.real_version = provenance.metadata.version
        self.version_patch = mock.patch.object(
            provenance.metadata,
            "version",
            side_effect=self._package_version,
        )
        self.version_patch.start()

    def tearDown(self):
        self.version_patch.stop()

    def _package_version(self, name):
        if name == "fixture-dist":
            return self.fixture_version
        return self.real_version(name)

    def _contract(self, root, scorer):
        return provenance.build_scorer_provenance(
            [("fixture", scorer)],
            params={"registered": True},
            device="cpu",
            runner_path=root / "runner.py",
            base_path=root / "base.py",
            source_root=root,
        )

    def test_contract_detects_source_checkpoint_version_and_preprocess_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            plugin = root / "fixture_plugin.py"
            runner = root / "runner.py"
            base = root / "base.py"
            checkpoint = root / "fixture.bin"
            for path, content in (
                (plugin, "PLUGIN = 1\n"),
                (runner, "RUNNER = 1\n"),
                (base, "BASE = 1\n"),
                (checkpoint, "weights-v1"),
            ):
                path.write_text(content)

            module_name = "scorer_provenance_fixture_module"
            module = types.ModuleType(module_name)
            module.__file__ = str(plugin)
            sys.modules[module_name] = module
            original_module = _FixtureScorer.__module__
            _FixtureScorer.__module__ = module_name
            try:
                scorer = _FixtureScorer(
                    checkpoint,
                    {"resize": 224, "interpolation": "bicubic"},
                )
                baseline, baseline_hash = self._contract(root, scorer)

                plugin.write_text("PLUGIN = 2\n")
                source_changed, source_hash = self._contract(root, scorer)
                self.assertNotEqual(source_hash, baseline_hash)
                self.assertNotEqual(
                    source_changed["scorers"][0]["plugin_source"]["sha256"],
                    baseline["scorers"][0]["plugin_source"]["sha256"],
                )

                plugin.write_text("PLUGIN = 1\n")
                checkpoint.write_text("weights-v2")
                checkpoint_changed, checkpoint_hash = self._contract(root, scorer)
                self.assertNotEqual(checkpoint_hash, baseline_hash)
                self.assertNotEqual(
                    checkpoint_changed["scorers"][0]["checkpoints"]["files_sha256"],
                    baseline["scorers"][0]["checkpoints"]["files_sha256"],
                )

                checkpoint.write_text("weights-v1")
                self.fixture_version = "2.0"
                version_changed, version_hash = self._contract(root, scorer)
                self.assertNotEqual(version_hash, baseline_hash)
                self.assertEqual(
                    version_changed["scorers"][0]["package_versions"]["fixture-dist"],
                    "2.0",
                )

                self.fixture_version = "1.0"
                scorer.preprocessing["resize"] = 384
                preprocess_changed, preprocess_hash = self._contract(root, scorer)
                self.assertNotEqual(preprocess_hash, baseline_hash)
                self.assertEqual(
                    preprocess_changed["scorers"][0]["preprocessing"]["resize"],
                    384,
                )
            finally:
                _FixtureScorer.__module__ = original_module
                sys.modules.pop(module_name, None)

    def test_hf_revision_survives_snapshot_symlink_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            revision = "a" * 40
            blob = root / "models--owner--model" / "blobs" / ("b" * 64)
            snapshot = (
                root
                / "models--owner--model"
                / "snapshots"
                / revision
                / "model.safetensors"
            )
            blob.parent.mkdir(parents=True)
            snapshot.parent.mkdir(parents=True)
            blob.write_bytes(b"checkpoint")
            os.symlink(blob, snapshot)

            self.assertEqual(provenance.resolved_hf_revision(snapshot), revision)
            record = provenance.checkpoint_file_record(
                snapshot,
                role="model",
                repository_id="owner/model",
            )
            self.assertEqual(record["revision"], revision)
            self.assertEqual(record["sha256"], provenance.sha256_file(blob))

    def test_runtime_manifest_is_json_safe_and_binds_cpu_runtime(self):
        runtime = provenance.runtime_manifest("cpu")
        json.dumps(runtime)
        self.assertEqual(runtime["device"], "cpu")
        self.assertIsNone(runtime["cuda_device"])
        for key in (
            "python_implementation",
            "python_version",
            "torch_version",
            "cuda_runtime_version",
            "cudnn_version",
            "runner_package_versions",
        ):
            self.assertIn(key, runtime)

    def test_hardened_rows_reject_self_inconsistent_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            plugin = root / "fixture_plugin.py"
            runner = root / "runner.py"
            base = root / "base.py"
            checkpoint = root / "fixture.bin"
            for path in (plugin, runner, base, checkpoint):
                path.write_text(path.name)
            module_name = "scorer_provenance_validation_fixture"
            module = types.ModuleType(module_name)
            module.__file__ = str(plugin)
            sys.modules[module_name] = module
            original_module = _FixtureScorer.__module__
            _FixtureScorer.__module__ = module_name
            try:
                contract, digest = self._contract(
                    root, _FixtureScorer(checkpoint, {"resize": 224})
                )
            finally:
                _FixtureScorer.__module__ = original_module
                sys.modules.pop(module_name, None)
            rows = [
                {
                    "id": "fixture",
                    "scorer_provenance": contract,
                    "scorer_provenance_sha256": digest,
                }
            ]
            self.assertEqual(provenance.validate_hardened_score_rows(rows), digest)

            rows[0]["scorer_provenance"]["scorers"][0]["preprocessing"][
                "resize"
            ] = 384
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                provenance.validate_hardened_score_rows(rows)


if __name__ == "__main__":
    unittest.main()
