from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
sys.path.insert(0, str(EVAL_PIPELINE))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = load_module(
    "hpsv2_relational_analyzer_test",
    EVAL_PIPELINE / "analyze_hpsv2_relational_renderer.py",
)
score = load_module(
    "hpsv2_score_hash_test",
    EVAL_PIPELINE / "score_hpsv2_relational_renderer.py",
)


class ScoreHashBindingTest(unittest.TestCase):
    def test_hash_binding_is_all_or_none_and_requires_sha256(self) -> None:
        self.assertFalse(score.manifest_uses_hash_binding([{"id": "plain"}]))
        bound = [
            {
                "id": "bound",
                "image_sha256": "a" * 64,
                "run_contract_sha256": "b" * 64,
            }
        ]
        self.assertTrue(score.manifest_uses_hash_binding(bound))
        with self.assertRaisesRegex(RuntimeError, "mixes"):
            score.manifest_uses_hash_binding([{"id": "plain"}, *bound])
        with self.assertRaisesRegex(RuntimeError, "lowercase SHA-256"):
            score.manifest_uses_hash_binding(
                [
                    {
                        "id": "bad",
                        "image_sha256": "A" * 64,
                        "run_contract_sha256": "b" * 64,
                    }
                ]
            )

    def test_hash_bound_images_are_verified_before_score_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            image = run_dir / "images/example.png"
            image.parent.mkdir()
            image.write_bytes(b"frozen-image")
            row = {
                "id": "example",
                "image_path": "images/example.png",
                "image_sha256": hashlib.sha256(b"frozen-image").hexdigest(),
                "run_contract_sha256": "b" * 64,
            }
            self.assertEqual(
                score.load_hash_bound_image_bytes(run_dir, row), b"frozen-image"
            )
            score.validate_hash_bound_images(run_dir, [row])
            image.write_bytes(b"changed-image")
            with self.assertRaisesRegex(RuntimeError, "image changed"):
                score.validate_hash_bound_images(run_dir, [row])
            row["image_path"] = "../outside.png"
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                score.validate_hash_bound_images(run_dir, [row])


class ExclusiveCudaWatchdogTest(unittest.TestCase):
    def query(self, process_rows: str):
        return mock.patch.object(
            score,
            "_nvidia_smi",
            side_effect=["5, GPU-target\n", process_rows],
        )

    def test_allows_only_the_scorer_pid_on_the_target_gpu(self) -> None:
        scorer_pid = 1234
        processes = (
            f"GPU-target, {scorer_pid}, python, 1024\n"
            "GPU-other, 9999, python, 2048\n"
        )
        with mock.patch.dict(score.os.environ, {}, clear=True), self.query(processes):
            self.assertEqual(
                score._exclusive_cuda_conflicts("cuda:5", scorer_pid), []
            )

    def test_rejects_a_foreign_pid_on_the_target_gpu(self) -> None:
        processes = "GPU-target, 9999, other-python, 2048\n"
        with mock.patch.dict(score.os.environ, {}, clear=True), self.query(processes):
            watchdog = score.ExclusiveCudaWatchdog("cuda:5", enabled=True)
            watchdog.pid = 1234
            with self.assertRaisesRegex(RuntimeError, "foreign compute process"):
                watchdog._check_now()

    def test_gpu_query_failure_is_fatal(self) -> None:
        with (
            mock.patch.dict(score.os.environ, {}, clear=True),
            mock.patch.object(
                score,
                "_nvidia_smi",
                side_effect=RuntimeError("cannot query NVIDIA compute-process state"),
            ),
        ):
            watchdog = score.ExclusiveCudaWatchdog("cuda:5", enabled=True)
            with self.assertRaisesRegex(RuntimeError, "cannot query"):
                watchdog.__enter__()

    def test_frozen_config_enables_monitor_without_cli_flag(self) -> None:
        self.assertTrue(
            score.exclusive_cuda_required(
                {"exclusive_cuda_process": True}, cli_required=False
            )
        )
        self.assertTrue(score.exclusive_cuda_required({}, cli_required=True))
        self.assertFalse(score.exclusive_cuda_required({}, cli_required=False))
        with self.assertRaisesRegex(RuntimeError, "must be true or false"):
            score.exclusive_cuda_required(
                {"exclusive_cuda_process": "true"}, cli_required=False
            )

    def test_analyzer_requires_hash_bound_monitor_provenance(self) -> None:
        contract = score.exclusive_cuda_execution_contract("cuda:2")
        digest = score.json_sha256(contract)
        self.assertEqual(
            analyzer.validate_exclusive_cuda_execution_contract(
                contract, digest, expected_device="cuda:2"
            ),
            digest,
        )
        tampered = dict(contract)
        tampered["exclusive_cuda_process"] = False
        with self.assertRaisesRegex(ValueError, "contract differs"):
            analyzer.validate_exclusive_cuda_execution_contract(
                tampered, digest, expected_device="cuda:2"
            )


class ScoringReceiptTest(unittest.TestCase):
    def make_publication(self, root: Path):
        run_dir = root / "run"
        run_dir.mkdir()
        scoring_config_path = EVAL_PIPELINE / "configs/hpsv2_full_scoring_v1.yaml"
        scoring_config_payload = scoring_config_path.read_bytes()
        scoring_config = yaml.safe_load(scoring_config_payload)
        run_contract_sha256 = "b" * 64
        run_config_payload = (
            json.dumps({"run_contract_sha256": run_contract_sha256}) + "\n"
        ).encode("utf-8")
        manifest = [{"id": "task-0", "run_contract_sha256": run_contract_sha256}]
        manifest_payload = (json.dumps(manifest[0]) + "\n").encode("utf-8")
        scorer_contract = {"schema": "test-scorer-provenance"}
        scorer_hash = score.json_sha256(scorer_contract)
        cuda_contract = score.exclusive_cuda_execution_contract("cuda:2")
        cuda_hash = score.json_sha256(cuda_contract)
        scores = [
            {
                "id": "task-0",
                "run_contract_sha256": run_contract_sha256,
                "hpsv2": 0.25,
                "scorer_provenance": scorer_contract,
                "scorer_provenance_sha256": scorer_hash,
                "scoring_execution_provenance": cuda_contract,
                "scoring_execution_provenance_sha256": cuda_hash,
            }
        ]
        scores_payload = (json.dumps(scores[0]) + "\n").encode("utf-8")
        (run_dir / "config.json").write_bytes(run_config_payload)
        (run_dir / "manifest.jsonl").write_bytes(manifest_payload)
        (run_dir / "scores.jsonl").write_bytes(scores_payload)
        receipt = score.build_scoring_success_receipt(
            scores_sha256=hashlib.sha256(scores_payload).hexdigest(),
            score_rows=scores,
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
            manifest_rows=manifest,
            run_config_sha256=hashlib.sha256(run_config_payload).hexdigest(),
            run_contract_sha256=run_contract_sha256,
            scoring_config_sha256=hashlib.sha256(
                scoring_config_payload
            ).hexdigest(),
            scoring_config=scoring_config,
            metric_names=scoring_config["metrics"],
            params=scoring_config["params"],
            strict=True,
            required_scorer_provenance_schema=scoring_config[
                "scorer_provenance"
            ]["required_schema"],
            scorer_provenance=scorer_contract,
            scorer_provenance_sha256=scorer_hash,
            cuda_execution_provenance=cuda_contract,
            cuda_execution_provenance_sha256=cuda_hash,
        )
        (run_dir / score.SCORING_SUCCESS_NAME).write_bytes(
            score.scoring_success_receipt_bytes(receipt)
        )
        contract = {
            "config": {
                "scoring": {
                    "config": "eval-pipeline/configs/hpsv2_full_scoring_v1.yaml"
                }
            }
        }
        return run_dir, manifest, scores, contract, scorer_hash, receipt

    def test_analyzer_rejects_numeric_score_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest, scores, contract, scorer_hash, _ = (
                self.make_publication(Path(temporary))
            )
            analyzer.validate_scoring_publication(
                run_dir, manifest, scores, contract, scorer_hash
            )
            scores[0]["hpsv2"] = 0.99
            (run_dir / "scores.jsonl").write_text(json.dumps(scores[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                analyzer.validate_scoring_publication(
                    run_dir, manifest, scores, contract, scorer_hash
                )

    def test_analyzer_requires_a_success_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest, scores, contract, scorer_hash, _ = (
                self.make_publication(Path(temporary))
            )
            (run_dir / score.SCORING_SUCCESS_NAME).unlink()
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                analyzer.validate_scoring_publication(
                    run_dir, manifest, scores, contract, scorer_hash
                )

    def test_analyzer_rejects_receipt_binding_mismatches(self) -> None:
        mutations = (
            ("scores", "task_ids_sha256"),
            ("scorer_provenance", "sha256"),
            ("scoring_config", "sha256"),
            ("cuda_execution_provenance", "sha256"),
        )
        for section, field in mutations:
            with self.subTest(section=section, field=field), tempfile.TemporaryDirectory() as temporary:
                run_dir, manifest, scores, contract, scorer_hash, receipt = (
                    self.make_publication(Path(temporary))
                )
                receipt[section][field] = "0" * 64
                (run_dir / score.SCORING_SUCCESS_NAME).write_bytes(
                    score.scoring_success_receipt_bytes(receipt)
                )
                with self.assertRaisesRegex(ValueError, "receipt differs"):
                    analyzer.validate_scoring_publication(
                        run_dir, manifest, scores, contract, scorer_hash
                    )


class TransactionalScoringTest(unittest.TestCase):
    @staticmethod
    def make_run(root: Path):
        run_dir = root / "run"
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True)
        image_path = image_dir / "task-0.png"
        Image.new("RGB", (4, 4), color=(20, 30, 40)).save(image_path)
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        run_contract_sha256 = "b" * 64
        manifest = [
            {
                "id": "task-0",
                "prompt": "a prompt",
                "image_path": "images/task-0.png",
                "image_sha256": image_hash,
                "run_contract_sha256": run_contract_sha256,
            }
        ]
        (run_dir / "manifest.jsonl").write_text(
            json.dumps(manifest[0]) + "\n", encoding="utf-8"
        )
        (run_dir / "config.json").write_text(
            json.dumps({"run_contract_sha256": run_contract_sha256}),
            encoding="utf-8",
        )
        scoring_config_path = root / "scoring.yaml"
        scoring_config_path.write_text(
            yaml.safe_dump(
                {
                    "metrics": ["fixture"],
                    "params": {},
                    "scorer_provenance": {
                        "required_schema": score.SCORER_PROVENANCE_SCHEMA
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return run_dir, scoring_config_path, manifest

    @staticmethod
    def registry(calls, *, value=0.75):
        class DummyScorer:
            OUTPUT_KEYS = (("fixture_score", "higher"),)
            PROVENANCE_PACKAGES = ()

            @classmethod
            def weights_status(cls, **_params):
                return True, ""

            def __init__(self, device="cpu", **_params):
                self.device = device

            def provenance_metadata(self):
                return {
                    "models": [],
                    "checkpoint_files": [],
                    "preprocessing": {"fixture": "identity"},
                    "parameters": {},
                    "supporting_sources": [],
                }

            def score_image(self, _image, _prompt):
                calls["fixture"] += 1
                return {"fixture_score": value}

        return {"fixture": DummyScorer}

    @staticmethod
    def invoke(run_dir: Path, scoring_config: Path, registry) -> None:
        argv = [
            "score_hpsv2_relational_renderer.py",
            "--run_dir",
            str(run_dir),
            "--config",
            str(scoring_config),
            "--device",
            "cpu",
            "--strict",
            "--require-scorer-provenance",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict(score.REGISTRY, registry, clear=True),
        ):
            score.main()

    def test_success_pair_is_receipted_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(Path(temporary))
            calls = {"fixture": 0}
            registry = self.registry(calls)

            self.invoke(run_dir, scoring_config, registry)

            self.assertEqual(calls["fixture"], 1)
            receipt = json.loads(
                (run_dir / score.SCORING_SUCCESS_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["scores"]["row_count"], 1)
            self.assertEqual(receipt["manifest"]["row_count"], 1)
            self.assertEqual(
                receipt["manifest"]["sha256"],
                hashlib.sha256((run_dir / "manifest.jsonl").read_bytes()).hexdigest(),
            )

            calls["fixture"] = 0
            self.invoke(run_dir, scoring_config, registry)
            self.assertEqual(calls["fixture"], 0)

    def test_unreceipted_scores_are_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(Path(temporary))
            (run_dir / "scores.jsonl").write_text(
                json.dumps({"id": "task-0", "fixture_score": 9.0}) + "\n",
                encoding="utf-8",
            )
            calls = {"fixture": 0}

            self.invoke(run_dir, scoring_config, self.registry(calls))

            self.assertEqual(calls["fixture"], 1)
            published = json.loads(
                (run_dir / "scores.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(published["fixture_score"], 0.75)

    def test_strict_failure_leaves_no_canonical_or_attempt_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(Path(temporary))
            calls = {"fixture": 0}
            with self.assertRaisesRegex(RuntimeError, "fixture failed"):
                self.invoke(
                    run_dir,
                    scoring_config,
                    self.registry(calls, value=float("nan")),
                )

            self.assertFalse((run_dir / "scores.jsonl").exists())
            self.assertFalse((run_dir / score.SCORING_SUCCESS_NAME).exists())
            self.assertEqual(list(run_dir.glob(".scores.attempt-*.jsonl")), [])

    def test_watchdog_exit_failure_after_progress_publishes_nothing(self) -> None:
        class ExitFailureWatchdog:
            enabled = False

            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, _exc_value, _traceback):
                if exc_type is None:
                    raise RuntimeError("exclusive GPU scoring monitor failed")
                return False

            def assert_healthy(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(Path(temporary))
            calls = {"fixture": 0}
            with mock.patch.object(
                score, "ExclusiveCudaWatchdog", ExitFailureWatchdog
            ), self.assertRaisesRegex(RuntimeError, "monitor failed"):
                self.invoke(run_dir, scoring_config, self.registry(calls))

            self.assertEqual(calls["fixture"], 1)
            self.assertFalse((run_dir / "scores.jsonl").exists())
            self.assertFalse((run_dir / score.SCORING_SUCCESS_NAME).exists())
            self.assertEqual(list(run_dir.glob(".scores.attempt-*.jsonl")), [])

    def test_nvidia_smi_failure_removes_stale_canonical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(Path(temporary))
            config = yaml.safe_load(scoring_config.read_text(encoding="utf-8"))
            config["exclusive_cuda_process"] = True
            scoring_config.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            (run_dir / "scores.jsonl").write_text("stale\n", encoding="utf-8")
            (run_dir / score.SCORING_SUCCESS_NAME).write_text(
                "{}\n", encoding="utf-8"
            )
            argv = [
                "score_hpsv2_relational_renderer.py",
                "--run_dir",
                str(run_dir),
                "--config",
                str(scoring_config),
                "--device",
                "cuda:2",
                "--strict",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch.object(
                    score,
                    "_nvidia_smi",
                    side_effect=RuntimeError(
                        "cannot query NVIDIA compute-process state"
                    ),
                ),
                self.assertRaisesRegex(RuntimeError, "cannot query"),
            ):
                score.main()

            self.assertFalse((run_dir / "scores.jsonl").exists())
            self.assertFalse((run_dir / score.SCORING_SUCCESS_NAME).exists())




if __name__ == "__main__":
    unittest.main()
