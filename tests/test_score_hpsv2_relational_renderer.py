from __future__ import annotations

import contextlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock
import urllib.request

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
    def test_frozen_config_registers_the_complete_scorer_contract_hash(self) -> None:
        config = yaml.safe_load(
            (EVAL_PIPELINE / "configs/hpsv2_full_scoring_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        contract, digest = score.registered_scorer_provenance_contract(config)
        self.assertIsNone(contract)
        self.assertEqual(
            digest,
            "4ae13c86588d4d1c23cf99e04bc178130ceb160288ed30c6df93641409447926",
        )

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
                    "config": "eval-pipeline/configs/hpsv2_full_scoring_v1.yaml",
                    "config_sha256": hashlib.sha256(
                        scoring_config_payload
                    ).hexdigest(),
                    "registered_scorer_provenance_sha256": scoring_config[
                        "registered_scorer_provenance_sha256"
                    ],
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
    def assert_private_progress_count(self, run_dir: Path, expected: int) -> None:
        self.assertEqual(
            len(score._scoring_progress_payload_paths(run_dir)), expected
        )

    @staticmethod
    def make_run(root: Path, row_count: int = 1):
        run_dir = root / "run"
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True)
        run_contract_sha256 = "b" * 64
        manifest = []
        for index in range(row_count):
            image_path = image_dir / f"task-{index}.png"
            Image.new(
                "RGB", (4, 4), color=(20 + index, 30, 40)
            ).save(image_path)
            manifest.append(
                {
                    "id": f"task-{index}",
                    "prompt": f"prompt {index}",
                    "image_path": f"images/task-{index}.png",
                    "image_sha256": hashlib.sha256(
                        image_path.read_bytes()
                    ).hexdigest(),
                    "run_contract_sha256": run_contract_sha256,
                }
            )
        (run_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in manifest),
            encoding="utf-8",
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
    def registry(calls, *, value=0.75, fail_on_call=None):
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
                if calls["fixture"] == fail_on_call:
                    raise RuntimeError("planned scorer interruption")
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
            scores_bytes = (run_dir / "scores.jsonl").read_bytes()
            receipt_bytes = (run_dir / score.SCORING_SUCCESS_NAME).read_bytes()
            scores_inode = (run_dir / "scores.jsonl").stat().st_ino
            receipt_inode = (run_dir / score.SCORING_SUCCESS_NAME).stat().st_ino

            calls["fixture"] = 0
            self.invoke(run_dir, scoring_config, registry)
            self.assertEqual(calls["fixture"], 0)
            self.assertEqual((run_dir / "scores.jsonl").read_bytes(), scores_bytes)
            self.assertEqual(
                (run_dir / score.SCORING_SUCCESS_NAME).read_bytes(), receipt_bytes
            )
            self.assertEqual((run_dir / "scores.jsonl").stat().st_ino, scores_inode)
            self.assertEqual(
                (run_dir / score.SCORING_SUCCESS_NAME).stat().st_ino,
                receipt_inode,
            )

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

    def test_strict_failure_keeps_only_bound_private_progress(self) -> None:
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
            self.assert_private_progress_count(run_dir, 1)
            self.assertTrue(
                (run_dir / score.SCORING_PROGRESS_RECEIPT_NAME).is_file()
            )

            calls["fixture"] = 0
            self.invoke(run_dir, scoring_config, self.registry(calls))
            self.assertEqual(calls["fixture"], 1)
            self.assert_private_progress_count(run_dir, 0)
            self.assertFalse(
                (run_dir / score.SCORING_PROGRESS_RECEIPT_NAME).exists()
            )

    def test_partial_progress_resumes_only_unfinished_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(
                Path(temporary), row_count=2
            )
            calls = {"fixture": 0}
            with self.assertRaisesRegex(RuntimeError, "fixture failed"):
                self.invoke(
                    run_dir,
                    scoring_config,
                    self.registry(calls, fail_on_call=2),
                )
            self.assertEqual(calls["fixture"], 2)
            self.assert_private_progress_count(run_dir, 1)

            calls["fixture"] = 0
            self.invoke(run_dir, scoring_config, self.registry(calls))
            self.assertEqual(calls["fixture"], 1)
            self.assertTrue((run_dir / score.SCORING_SUCCESS_NAME).is_file())
            self.assert_private_progress_count(run_dir, 0)

    def test_tampered_private_progress_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(
                Path(temporary), row_count=2
            )
            calls = {"fixture": 0}
            with self.assertRaisesRegex(RuntimeError, "fixture failed"):
                self.invoke(
                    run_dir,
                    scoring_config,
                    self.registry(calls, fail_on_call=2),
                )
            progress_path = Path(
                score._scoring_progress_payload_paths(run_dir)[0]
            )
            progress_path.write_text(
                progress_path.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )

            calls["fixture"] = 0
            self.invoke(run_dir, scoring_config, self.registry(calls))
            self.assertEqual(calls["fixture"], 2)
            self.assertTrue((run_dir / score.SCORING_SUCCESS_NAME).is_file())

    def test_interrupted_checkpoint_keeps_previous_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(
                Path(temporary), row_count=52
            )
            calls = {"fixture": 0}
            receipt_writes = 0
            real_atomic_writer = score.atomic_text_writer

            @contextlib.contextmanager
            def interrupt_second_receipt(path):
                nonlocal receipt_writes
                if Path(path).name == score.SCORING_PROGRESS_RECEIPT_NAME:
                    receipt_writes += 1
                    if receipt_writes == 2:
                        raise OSError("planned checkpoint interruption")
                with real_atomic_writer(path) as handle:
                    yield handle

            with (
                mock.patch.object(
                    score, "atomic_text_writer", interrupt_second_receipt
                ),
                self.assertRaisesRegex(OSError, "checkpoint interruption"),
            ):
                self.invoke(
                    run_dir,
                    scoring_config,
                    self.registry(calls, fail_on_call=52),
                )
            self.assertEqual(calls["fixture"], 52)
            receipt = json.loads(
                (run_dir / score.SCORING_PROGRESS_RECEIPT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            progress_path = run_dir / receipt["scores"]["path"]
            progress_rows = score.load_jsonl(progress_path)
            self.assertEqual(len(progress_rows), 50)
            self.assertTrue(progress_path.is_file())

            calls["fixture"] = 0
            self.invoke(run_dir, scoring_config, self.registry(calls))
            self.assertEqual(calls["fixture"], 2)
            self.assertTrue((run_dir / score.SCORING_SUCCESS_NAME).is_file())

    def test_watchdog_exit_failure_after_progress_publishes_nothing(self) -> None:
        class ExitFailureWatchdog:
            enabled = False

            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, _exc_value, _traceback):
                if exc_type is None:
                    raise score.ExclusiveCudaWatchdogError(
                        "exclusive GPU scoring monitor failed"
                    )
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
            self.assert_private_progress_count(run_dir, 0)
            self.assertFalse(
                (run_dir / score.SCORING_PROGRESS_RECEIPT_NAME).exists()
            )

    def test_watchdog_entry_failure_preserves_valid_canonical_publication(self) -> None:
        class EntryFailureWatchdog:
            enabled = False

            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                raise score.ExclusiveCudaWatchdogError(
                    "exclusive GPU scoring monitor failed at startup"
                )

            def __exit__(self, *_args):
                return False

            def assert_healthy(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(Path(temporary))
            calls = {"fixture": 0}
            registry = self.registry(calls)
            self.invoke(run_dir, scoring_config, registry)
            scores_bytes = (run_dir / "scores.jsonl").read_bytes()
            receipt_bytes = (run_dir / score.SCORING_SUCCESS_NAME).read_bytes()

            with mock.patch.object(
                score, "ExclusiveCudaWatchdog", EntryFailureWatchdog
            ), self.assertRaisesRegex(RuntimeError, "startup"):
                self.invoke(run_dir, scoring_config, registry)

            self.assertEqual((run_dir / "scores.jsonl").read_bytes(), scores_bytes)
            self.assertEqual(
                (run_dir / score.SCORING_SUCCESS_NAME).read_bytes(), receipt_bytes
            )

    def test_watchdog_exit_failure_preserves_valid_canonical_publication(self) -> None:
        class ExitFailureWatchdog:
            enabled = False

            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, _exc_value, _traceback):
                if exc_type is None:
                    raise score.ExclusiveCudaWatchdogError(
                        "exclusive GPU scoring monitor failed at exit"
                    )
                return False

            def assert_healthy(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(Path(temporary))
            calls = {"fixture": 0}
            registry = self.registry(calls)
            self.invoke(run_dir, scoring_config, registry)
            scores_bytes = (run_dir / "scores.jsonl").read_bytes()
            receipt_bytes = (run_dir / score.SCORING_SUCCESS_NAME).read_bytes()

            with mock.patch.object(
                score, "ExclusiveCudaWatchdog", ExitFailureWatchdog
            ), self.assertRaisesRegex(RuntimeError, "at exit"):
                self.invoke(run_dir, scoring_config, registry)

            self.assertEqual(calls["fixture"], 1)
            self.assertEqual((run_dir / "scores.jsonl").read_bytes(), scores_bytes)
            self.assertEqual(
                (run_dir / score.SCORING_SUCCESS_NAME).read_bytes(), receipt_bytes
            )

    def test_failed_republication_restores_previous_canonical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, scoring_config, _ = self.make_run(root)
            calls = {"fixture": 0}
            self.invoke(run_dir, scoring_config, self.registry(calls, value=0.75))
            scores_bytes = (run_dir / "scores.jsonl").read_bytes()
            receipt_bytes = (run_dir / score.SCORING_SUCCESS_NAME).read_bytes()

            config = yaml.safe_load(scoring_config.read_text(encoding="utf-8"))
            config["params"] = {"revision": 2}
            scoring_config.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            real_atomic_writer = score.atomic_text_writer
            failed_once = False

            @contextlib.contextmanager
            def fail_first_success_receipt(path):
                nonlocal failed_once
                if (
                    Path(path).name == score.SCORING_SUCCESS_NAME
                    and not failed_once
                ):
                    failed_once = True
                    raise OSError("planned final receipt interruption")
                with real_atomic_writer(path) as handle:
                    yield handle

            calls["fixture"] = 0
            with mock.patch.object(
                score, "atomic_text_writer", fail_first_success_receipt
            ), self.assertRaisesRegex(OSError, "final receipt interruption"):
                self.invoke(run_dir, scoring_config, self.registry(calls, value=0.9))

            self.assertEqual(calls["fixture"], 1)
            self.assertEqual((run_dir / "scores.jsonl").read_bytes(), scores_bytes)
            self.assertEqual(
                (run_dir / score.SCORING_SUCCESS_NAME).read_bytes(), receipt_bytes
            )
            self.assert_private_progress_count(run_dir, 1)

    def test_nvidia_smi_failure_preserves_existing_canonical_artifacts(self) -> None:
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

            self.assertEqual(
                (run_dir / "scores.jsonl").read_text(encoding="utf-8"),
                "stale\n",
            )
            self.assertEqual(
                (run_dir / score.SCORING_SUCCESS_NAME).read_text(
                    encoding="utf-8"
                ),
                "{}\n",
            )


class ScorerAssetStageTest(unittest.TestCase):
    @staticmethod
    def asset_class(source, staged_name="nested/weights.bin"):
        class AssetScorer:
            @classmethod
            def asset_sources(cls, **_params):
                return {
                    "weights": {
                        "path": str(source),
                        "staged_name": staged_name,
                        "revision": "fixture-revision",
                    }
                }

        return AssetScorer

    def test_stage_namespaces_assets_binds_manifest_and_cleans_nested_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "weights.bin"
            source.write_bytes(b"frozen-weights")
            registry = {
                "first": self.asset_class(source),
                "second": self.asset_class(source),
            }
            with mock.patch.dict(score.REGISTRY, registry, clear=True):
                stage = score.ScorerAssetStage(["first", "second"], {})
            stage_root = stage.path
            first = Path(stage.asset_paths["first"]["weights"])
            second = Path(stage.asset_paths["second"]["weights"])
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"frozen-weights")
            self.assertEqual(second.read_bytes(), b"frozen-weights")
            self.assertEqual(
                stage.asset_manifests["first"]["files_sha256"],
                score.json_sha256(stage.asset_manifests["first"]["files"]),
            )

            source.write_bytes(b"mutable-cache-changed")
            self.assertEqual(first.read_bytes(), b"frozen-weights")
            stage.verify()
            stage.cleanup()
            self.assertFalse(stage_root.exists())

    def test_stage_detects_same_byte_rewrite_and_still_cleans(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "weights.bin"
            source.write_bytes(b"same-bytes")
            with mock.patch.dict(
                score.REGISTRY,
                {"fixture": self.asset_class(source)},
                clear=True,
            ):
                stage = score.ScorerAssetStage(["fixture"], {})
            stage_root = stage.path
            staged = Path(stage.asset_paths["fixture"]["weights"])
            os.chmod(staged, 0o600)
            staged.write_bytes(b"same-bytes")
            os.chmod(staged, 0o400)
            with self.assertRaisesRegex(
                score.ScorerAssetStageError, "identity changed"
            ):
                stage.verify()
            stage.cleanup(verify=False)
            self.assertFalse(stage_root.exists())

    def test_copy_rejects_source_changed_during_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"source-bytes")
            real_read = os.read
            changed = False

            def read_then_touch(descriptor, size):
                nonlocal changed
                payload = real_read(descriptor, size)
                if payload and not changed:
                    changed = True
                    source.write_bytes(b"source-bytes")
                return payload

            with mock.patch.object(os, "read", side_effect=read_then_touch):
                with self.assertRaisesRegex(
                    score.ScorerAssetStageError, "changed while it was staged"
                ):
                    score._copy_scorer_asset(source, destination)

    def test_formal_network_guard_blocks_socket_and_urlopen(self):
        original_socket = socket.socket
        original_urlopen = urllib.request.urlopen
        with mock.patch.dict(
            os.environ,
            {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            clear=False,
        ):
            with score.formal_offline_network_guard(enabled=True):
                with self.assertRaisesRegex(RuntimeError, "network access"):
                    urllib.request.urlopen("https://example.invalid")
                guarded_socket = socket.socket()
                try:
                    with self.assertRaisesRegex(RuntimeError, "network access"):
                        guarded_socket.connect(("127.0.0.1", 9))
                finally:
                    guarded_socket.close()
        self.assertIs(socket.socket, original_socket)
        self.assertIs(urllib.request.urlopen, original_urlopen)

    @staticmethod
    def fixture_manifest(path):
        files = [
            {
                "key": "fixture",
                "path": Path(path).name,
                "revision": None,
                "size_bytes": Path(path).stat().st_size,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
        ]
        return {
            "schema": "repldm_scorer_asset_stage_v1",
            "loading_mode": score.SCORER_ASSET_LOADING_MODE,
            "files": files,
            "files_sha256": score.json_sha256(files),
        }

    def test_openai_clip_scorers_receive_explicit_local_paths(self):
        import clip

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip_b = root / "ViT-B-32.pt"
            clip_l = root / "ViT-L-14.pt"
            aesthetic = root / "aesthetic.pth"
            for path in (clip_b, clip_l, aesthetic):
                path.write_bytes(path.name.encode("ascii"))
            model = mock.Mock()

            def preprocess(image):
                return image

            with mock.patch.object(
                clip, "load", return_value=(model, preprocess)
            ) as clip_load:
                scorer = score.REGISTRY["clip"](
                    device="cpu",
                    clip_model="ViT-B/32",
                    scorer_assets={"clip_checkpoint": str(clip_b)},
                    scorer_asset_manifest=self.fixture_manifest(clip_b),
                )
            self.assertEqual(clip_load.call_args.args[0], str(clip_b))
            self.assertEqual(
                scorer.provenance_metadata()["parameters"]["asset_stage"][
                    "schema"
                ],
                "repldm_scorer_asset_stage_v1",
            )

            aesthetic_module = sys.modules[
                score.REGISTRY["aesthetic"].__module__
            ]
            with (
                mock.patch.object(
                    clip, "load", return_value=(model, preprocess)
                ) as aesthetic_clip_load,
                mock.patch.object(
                    aesthetic_module.torch,
                    "load",
                    return_value={},
                ) as torch_load,
                mock.patch.object(
                    aesthetic_module._AestheticMLP,
                    "load_state_dict",
                ),
            ):
                score.REGISTRY["aesthetic"](
                    device="cpu",
                    scorer_assets={
                        "clip_checkpoint": str(clip_l),
                        "aesthetic_checkpoint": str(aesthetic),
                    },
                    scorer_asset_manifest=self.fixture_manifest(clip_l),
                )
            self.assertEqual(aesthetic_clip_load.call_args.args[0], str(clip_l))
            self.assertEqual(torch_load.call_args.args[0], str(aesthetic))

    def test_hps_and_topiq_receive_explicit_local_paths(self):
        import hpsv2.src.open_clip as hps_open_clip
        import hpsv2.src.open_clip.tokenizer as hps_tokenizer
        import pyiqa
        import timm

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hps_checkpoint = root / "HPS_v2.1_compressed.pt"
            hps_backbone = root / "open_clip_pytorch_model.bin"
            vocabulary = root / "bpe.txt.gz"
            topiq = root / "topiq.pth"
            resnet = root / "resnet.safetensors"
            for path in (
                hps_checkpoint,
                hps_backbone,
                vocabulary,
                topiq,
                resnet,
            ):
                path.write_bytes(path.name.encode("ascii"))

            model = mock.Mock()
            model.to.return_value = model
            model.eval.return_value = model
            hps_module = sys.modules[score.REGISTRY["hps"].__module__]
            with (
                mock.patch.object(
                    hps_open_clip,
                    "create_model_and_transforms",
                    return_value=(model, None, mock.Mock()),
                ) as create_hps,
                mock.patch.object(
                    hps_tokenizer, "SimpleTokenizer", return_value=mock.Mock()
                ) as tokenizer,
                mock.patch.object(
                    hps_module.torch,
                    "load",
                    return_value={"state_dict": {}},
                ) as torch_load,
            ):
                score.REGISTRY["hps"](
                    device="cpu",
                    scorer_assets={
                        "hpsv2_checkpoint": str(hps_checkpoint),
                        "open_clip_backbone": str(hps_backbone),
                        "tokenizer_vocabulary": str(vocabulary),
                    },
                    scorer_asset_manifest=self.fixture_manifest(hps_checkpoint),
                )
            self.assertEqual(
                create_hps.call_args.kwargs["pretrained"], str(hps_backbone)
            )
            self.assertEqual(torch_load.call_args.args[0], str(hps_checkpoint))
            self.assertEqual(tokenizer.call_args.kwargs["bpe_path"], str(vocabulary))

            original_create_model = mock.Mock(return_value=mock.Mock())

            def create_metric(_name, **_kwargs):
                timm.create_model(
                    "resnet50", pretrained=True, features_only=True
                )
                return mock.Mock()

            incompatible = mock.Mock(
                missing_keys=[], unexpected_keys=["fc.bias", "fc.weight"]
            )
            with (
                mock.patch.object(timm, "create_model", original_create_model),
                mock.patch.object(
                    timm.models,
                    "load_checkpoint",
                    return_value=incompatible,
                ) as load_checkpoint,
                mock.patch.object(pyiqa, "create_metric", side_effect=create_metric),
            ):
                score.REGISTRY["iqa"](
                    device="cpu",
                    scorer_assets={
                        "topiq_checkpoint": str(topiq),
                        "resnet50_backbone": str(resnet),
                    },
                    scorer_asset_manifest=self.fixture_manifest(topiq),
                )
            self.assertFalse(
                original_create_model.call_args.kwargs["pretrained"]
            )
            self.assertEqual(load_checkpoint.call_args.args[1], str(resnet))
            self.assertFalse(load_checkpoint.call_args.kwargs["strict"])




if __name__ == "__main__":
    unittest.main()
