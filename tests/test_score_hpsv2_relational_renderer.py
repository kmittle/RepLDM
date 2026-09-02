from __future__ import annotations

import contextlib
import copy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
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
import scorer_provenance as provenance  # noqa: E402


def cuda_identity(index):
    gpu_uuid = f"GPU-{int(index):08x}-0000-0000-0000-000000000000"
    return {
        "physical_device_index": int(index),
        "gpu_uuid": gpu_uuid,
        "pci_bus_id": f"00000000:{int(index):02X}:00.0",
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "torch_device_uuid": gpu_uuid,
    }


class ScoreHashBindingTest(unittest.TestCase):
    def test_legacy_shared_scorer_sources_remain_frozen(self) -> None:
        expected = {
            "base.py": "6b1c2cbd36577a514fc8ced2b1b5955a315cf2bae49ab30c7d26f1ff71096dc0",
            "aesthetic_scorer.py": "f165a62a87f1c10b23b00c701931676ce3d99cb1e2bb526fdee77c2d3b05f245",
            "clip_scorer.py": "e2fd82e9cc9c315bd5a4e84f57d90b3013fc005debaa8d676d34cab705d035fd",
            "hps_scorer.py": "cd22a92b3e7c8e17c637a3c26d1c25674c4df7e2fa938ee9f86d229bcbe75aaa",
            "imagereward_scorer.py": "7e37d87156fbad635cad29888cc53ed44a0ee5db4a7b0c8c4ccb659340495633",
            "iqa_scorer.py": "9ecb76347c20c0f1e0acefb72788dcdd97b07cf8179d205083bde9395d572926",
        }
        scorer_root = EVAL_PIPELINE / "scorers"
        observed = {
            name: hashlib.sha256((scorer_root / name).read_bytes()).hexdigest()
            for name in expected
        }
        self.assertEqual(observed, expected)

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
            "f2734daafd1040ad95ceb1295d5bcee35ac3882e66bf5bb32a9576ae8311ed42",
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

    def test_formal_scoring_config_is_bound_to_recorded_path_and_bytes(self) -> None:
        config_path = EVAL_PIPELINE / "configs/hpsv2_full_scoring_v1.yaml"
        payload = config_path.read_bytes()
        run_config = {
            "scoring": {
                "config": "eval-pipeline/configs/hpsv2_full_scoring_v1.yaml",
                "config_sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
        score._validate_formal_scoring_config_binding(
            config_path, payload, run_config
        )
        with self.assertRaisesRegex(RuntimeError, "path differs"):
            score._validate_formal_scoring_config_binding(
                config_path,
                payload,
                {
                    "scoring": {
                        "config": "eval-pipeline/configs/other.yaml",
                        "config_sha256": run_config["scoring"]["config_sha256"],
                    }
                },
            )
        with self.assertRaisesRegex(RuntimeError, "bytes differ"):
            score._validate_formal_scoring_config_binding(
                config_path,
                payload,
                {
                    "scoring": {
                        "config": run_config["scoring"]["config"],
                        "config_sha256": "b" * 64,
                    }
                },
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
    GPU_UUID = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    OTHER_UUID = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    PCI_BUS_ID = "00000000:89:00.0"

    @property
    def identity(self):
        return {
            "physical_device_index": 5,
            "gpu_uuid": self.GPU_UUID,
            "pci_bus_id": self.PCI_BUS_ID,
            "gpu_name": "NVIDIA GeForce RTX 3090",
            "torch_device_uuid": self.GPU_UUID,
        }

    def query(self, process_rows: str):
        return mock.patch.object(
            score,
            "_nvidia_smi",
            side_effect=[
                (
                    f"5, {self.GPU_UUID}, {self.PCI_BUS_ID}, "
                    "NVIDIA GeForce RTX 3090\n"
                ),
                process_rows,
            ],
        )

    def patch_start_ticks(self, value=5678):
        return mock.patch.object(score, "_process_start_ticks", return_value=value)

    def test_allows_only_the_scorer_pid_on_the_target_gpu(self) -> None:
        scorer_pid = 1234
        processes = (
            f"{self.GPU_UUID}, {scorer_pid}, python, 1024\n"
            f"{self.OTHER_UUID}, 9999, python, 2048\n"
        )
        with mock.patch.dict(score.os.environ, {}, clear=True), self.query(processes), self.patch_start_ticks():
            self.assertEqual(
                score._exclusive_cuda_conflicts(
                    "cuda:5", scorer_pid, target_identity=self.identity
                ),
                [],
            )

    def test_rejects_a_foreign_pid_on_the_target_gpu(self) -> None:
        processes = (
            f"{self.GPU_UUID}, 1234, python, 1024\n"
            f"{self.GPU_UUID}, 9999, other-python, 2048\n"
        )
        with mock.patch.dict(score.os.environ, {}, clear=True), self.query(processes), self.patch_start_ticks():
            watchdog = score.ExclusiveCudaWatchdog(
                "cuda:5", enabled=True, target_identity=self.identity
            )
            watchdog.pid = 1234
            with self.assertRaisesRegex(RuntimeError, "foreign compute process"):
                watchdog._check_now()

    def test_rejects_missing_scorer_pid(self) -> None:
        processes = f"{self.OTHER_UUID}, 9999, python, 2048\n"
        with mock.patch.dict(score.os.environ, {}, clear=True), self.query(processes), self.patch_start_ticks():
            watchdog = score.ExclusiveCudaWatchdog(
                "cuda:5", enabled=True, target_identity=self.identity
            )
            watchdog.pid = 1234
            with self.assertRaisesRegex(RuntimeError, "scorer PID is missing"):
                watchdog._check_now()

    def test_rejects_scorer_pid_on_another_gpu(self) -> None:
        processes = f"{self.OTHER_UUID}, 1234, python, 1024\n"
        with mock.patch.dict(score.os.environ, {}, clear=True), self.query(processes), self.patch_start_ticks():
            watchdog = score.ExclusiveCudaWatchdog(
                "cuda:5", enabled=True, target_identity=self.identity
            )
            watchdog.pid = 1234
            with self.assertRaisesRegex(RuntimeError, "different physical GPU"):
                watchdog._check_now()

    def test_rejects_scorer_pid_reuse_during_gpu_query(self) -> None:
        processes = f"{self.GPU_UUID}, 1234, python, 1024\n"
        inventory = (
            f"5, {self.GPU_UUID}, {self.PCI_BUS_ID}, "
            "NVIDIA GeForce RTX 3090\n"
        )
        with (
            mock.patch.dict(score.os.environ, {}, clear=True),
            mock.patch.object(
                score,
                "_nvidia_smi",
                side_effect=[inventory, processes],
            ),
            mock.patch.object(
                score, "_process_start_ticks", side_effect=[5678, 5678, 1234]
            ),
        ):
            watchdog = score.ExclusiveCudaWatchdog(
                "cuda:5", enabled=True, target_identity=self.identity
            )
            watchdog.pid = 1234
            watchdog.process_start_ticks = 5678
            with self.assertRaisesRegex(RuntimeError, "changed while GPU state was queried"):
                watchdog._check_now()

    def test_rejects_gpu_pci_identity_drift(self) -> None:
        inventory = (
            f"5, {self.GPU_UUID}, 00000000:88:00.0, "
            "NVIDIA GeForce RTX 3090\n"
        )
        with (
            mock.patch.dict(score.os.environ, {}, clear=True),
            mock.patch.object(score, "_nvidia_smi", return_value=inventory),
            self.patch_start_ticks(),
        ):
            watchdog = score.ExclusiveCudaWatchdog(
                "cuda:5", enabled=True, target_identity=self.identity
            )
            with self.assertRaisesRegex(RuntimeError, "identity drifted"):
                watchdog._check_now()

    def test_resolves_physical_identity_by_torch_uuid(self) -> None:
        properties = type(
            "FixtureCudaProperties",
            (),
            {
                "uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "name": "NVIDIA GeForce RTX 3090",
            },
        )()
        torch_module = mock.Mock()
        torch_module.cuda.current_device.return_value = 5
        torch_module.cuda.get_device_properties.return_value = properties
        inventory = (
            f"5, {self.GPU_UUID}, {self.PCI_BUS_ID}, NVIDIA GeForce RTX 3090\n"
        )
        with (
            mock.patch.dict(score.os.environ, {}, clear=True),
            mock.patch.object(score, "_nvidia_smi", return_value=inventory),
        ):
            self.assertEqual(
                score.resolve_scoring_cuda_identity(torch_module, "cuda:5"),
                self.identity,
            )

    def test_watchdog_requires_entry_and_final_pid_observations(self) -> None:
        scorer_pid = 1234
        processes = f"{self.GPU_UUID}, {scorer_pid}, python, 1024\n"
        inventory = (
            f"5, {self.GPU_UUID}, {self.PCI_BUS_ID}, "
            "NVIDIA GeForce RTX 3090\n"
        )
        with (
            mock.patch.dict(score.os.environ, {}, clear=True),
            mock.patch.object(
                score,
                "_nvidia_smi",
                side_effect=[inventory, processes, inventory, processes],
            ),
            self.patch_start_ticks(),
        ):
            watchdog = score.ExclusiveCudaWatchdog(
                "cuda:5",
                enabled=True,
                poll_seconds=3600,
                target_identity=self.identity,
            )
            watchdog.pid = scorer_pid
            watchdog.process_start_ticks = 5678
            with watchdog:
                pass
        self.assertTrue(watchdog.finalized)
        self.assertEqual(len(watchdog.observations), 2)
        execution = score.exclusive_cuda_execution_contract(
            "cuda:5", target_identity=self.identity
        )
        execution_hash = score.json_sha256(execution)
        observation = score.exclusive_cuda_observation_contract(
            execution_hash,
            scorer_pid=watchdog.pid,
            process_start_ticks=watchdog.process_start_ticks,
        )
        score.validate_exclusive_cuda_observation_contract(
            observation, execution_sha256=execution_hash
        )
        tampered = copy.deepcopy(observation)
        tampered["final_observation"]["scorer_pid_present_on_target_only"] = False
        with self.assertRaisesRegex(ValueError, "observation differs"):
            score.validate_exclusive_cuda_observation_contract(
                tampered, execution_sha256=execution_hash
            )

    def test_gpu_query_failure_is_fatal(self) -> None:
        with (
            mock.patch.dict(score.os.environ, {}, clear=True),
            mock.patch.object(
                score,
                "_nvidia_smi",
                side_effect=RuntimeError("cannot query NVIDIA compute-process state"),
            ),
        ):
            watchdog = score.ExclusiveCudaWatchdog(
                "cuda:5", enabled=True, target_identity=self.identity
            )
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
        contract = score.exclusive_cuda_execution_contract(
            "cuda:2", target_identity=cuda_identity(2)
        )
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
        cuda_contract = score.exclusive_cuda_execution_contract(
            "cuda:4", target_identity=cuda_identity(4)
        )
        cuda_hash = score.json_sha256(cuda_contract)
        cuda_observation = score.exclusive_cuda_observation_contract(
            cuda_hash, scorer_pid=1234, process_start_ticks=5678
        )
        scores = [
            {
                "id": "task-0",
                "run_contract_sha256": run_contract_sha256,
                "hpsv2": 0.25,
                "scorer_provenance": scorer_contract,
                "scorer_provenance_sha256": scorer_hash,
                "scoring_execution_provenance": cuda_contract,
                "scoring_execution_provenance_sha256": cuda_hash,
                "scoring_execution_observation": cuda_observation,
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

    def test_dead_poison_recovery_clears_progress_and_all_temporary_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            attempt_id = "a" * 32
            poison = {
                "schema": score.SCORING_PROGRESS_POISON_SCHEMA,
                "status": "in_progress",
                "attempt_id": attempt_id,
                "pid": 2_147_483_647,
                "process_start_ticks": 1,
                "boot_id": score._boot_id(),
            }
            (run_dir / score.SCORING_PROGRESS_POISON_NAME).write_bytes(
                score._scoring_poison_bytes(poison)
            )
            progress_name = (
                f".scoring_progress.{attempt_id}.{'b' * 64}.jsonl"
            )
            for name in (
                progress_name,
                score.SCORING_PROGRESS_RECEIPT_NAME,
                ".scoring_progress.stage.orphan.tmp",
                "..scoring_progress.json.orphan.tmp",
                "..scoring_progress.poison.orphan.tmp",
            ):
                (run_dir / name).write_text("stale\n", encoding="utf-8")
            (run_dir / "scores.jsonl").write_text(
                "canonical\n", encoding="utf-8"
            )
            (run_dir / score.SCORING_SUCCESS_NAME).write_text(
                "{}\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                score.ScoringAttemptPoisonedError, "not resumable"
            ):
                score.load_verified_scoring_progress(run_dir, {})
            self.assertEqual(
                score._recover_poisoned_scoring_progress(run_dir), attempt_id
            )

            self.assertFalse(
                (run_dir / score.SCORING_PROGRESS_POISON_NAME).exists()
            )
            self.assertFalse(
                (run_dir / score.SCORING_PROGRESS_RECEIPT_NAME).exists()
            )
            self.assertEqual(score._scoring_progress_payload_paths(run_dir), [])
            self.assertEqual(score._scoring_progress_temporary_paths(run_dir), [])
            self.assertEqual(
                (run_dir / "scores.jsonl").read_text(encoding="utf-8"),
                "canonical\n",
            )
            self.assertEqual(
                (run_dir / score.SCORING_SUCCESS_NAME).read_text(
                    encoding="utf-8"
                ),
                "{}\n",
            )

    def test_live_poison_cannot_be_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            attempt_id = "c" * 32
            score._begin_scoring_progress_poison(
                run_dir, attempt_id=attempt_id
            )
            with self.assertRaisesRegex(
                score.ScoringAttemptPoisonedError, "owned by a live process"
            ):
                score._recover_poisoned_scoring_progress(run_dir)
            self.assertTrue(
                (run_dir / score.SCORING_PROGRESS_POISON_NAME).is_file()
            )

    def test_asset_primary_and_cleanup_errors_are_both_preserved(self):
        class BrokenStage:
            def cleanup(self):
                raise OSError("cleanup-failed")

        primary = ValueError("score-failed")
        with (
            mock.patch.object(score, "ScorerAssetStage", return_value=BrokenStage()),
            self.assertRaises(score.ScoringAttemptPoisonedError) as raised,
        ):
            with score.staged_scorer_assets(
                [], {}, enabled=True, run_dir="ignored"
            ):
                raise primary
        self.assertIs(raised.exception.primary_error, primary)
        self.assertIsInstance(raised.exception.cleanup_error, OSError)
        self.assertIn("cleanup-failed", str(raised.exception.cleanup_error))

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
            self.assert_private_progress_count(run_dir, 1)
            self.assertTrue(
                (run_dir / score.SCORING_PROGRESS_RECEIPT_NAME).is_file()
            )
            self.assertTrue(
                (run_dir / score.SCORING_PROGRESS_POISON_NAME).is_file()
            )
            with self.assertRaisesRegex(
                score.ScoringAttemptPoisonedError, "not resumable"
            ):
                score.load_verified_scoring_progress(run_dir, {})

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
            failed_once = False

            def fail_after_scores(state):
                nonlocal failed_once
                if state == "scores_replaced" and not failed_once:
                    failed_once = True
                    raise OSError("planned final receipt interruption")

            calls["fixture"] = 0
            with mock.patch.object(
                score, "_publication_fault_point", fail_after_scores
            ), self.assertRaisesRegex(OSError, "final receipt interruption"):
                self.invoke(run_dir, scoring_config, self.registry(calls, value=0.9))

            self.assertEqual(calls["fixture"], 1)
            self.assertEqual((run_dir / "scores.jsonl").read_bytes(), scores_bytes)
            self.assertEqual(
                (run_dir / score.SCORING_SUCCESS_NAME).read_bytes(), receipt_bytes
            )
            self.assert_private_progress_count(run_dir, 1)

    def test_publication_and_recovery_failures_are_both_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, scoring_config, _ = self.make_run(root)
            calls = {"fixture": 0}
            self.invoke(run_dir, scoring_config, self.registry(calls, value=0.75))
            config = yaml.safe_load(scoring_config.read_text(encoding="utf-8"))
            config["params"] = {"revision": 2}
            scoring_config.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            real_reconcile = score._reconcile_scoring_publication

            def fail_after_scores(state):
                if state == "scores_replaced":
                    raise OSError("planned publication failure")

            def fail_durable_recovery(path):
                if (Path(path) / score.SCORING_PUBLICATION_JOURNAL_NAME).exists():
                    raise RuntimeError("planned durable recovery failure")
                return real_reconcile(path)

            calls["fixture"] = 0
            with (
                mock.patch.object(
                    score, "_publication_fault_point", fail_after_scores
                ),
                mock.patch.object(
                    score,
                    "_reconcile_scoring_publication",
                    side_effect=fail_durable_recovery,
                ),
                self.assertRaises(score.ScoringAttemptPoisonedError) as raised,
            ):
                self.invoke(run_dir, scoring_config, self.registry(calls, value=0.9))

            self.assertIsInstance(raised.exception.primary_error, OSError)
            self.assertIn("publication failure", str(raised.exception.primary_error))
            self.assertIsInstance(raised.exception.cleanup_error, RuntimeError)
            self.assertIn("recovery failure", str(raised.exception.cleanup_error))
            self.assertIs(raised.exception.__cause__, raised.exception.primary_error)

    def test_process_death_recovery_after_every_publication_transition(self) -> None:
        class SimulatedProcessDeath(BaseException):
            pass

        transitions = (
            "backups_persisted",
            "journal_persisted",
            "scores_replaced",
            "receipt_replaced",
            "new_pair_verified",
        )
        for transition in transitions:
            with self.subTest(transition=transition), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run_dir, scoring_config, _ = self.make_run(root)
                calls = {"fixture": 0}
                self.invoke(
                    run_dir,
                    scoring_config,
                    self.registry(calls, value=0.75),
                )
                old_scores = (run_dir / "scores.jsonl").read_bytes()
                old_receipt = (
                    run_dir / score.SCORING_SUCCESS_NAME
                ).read_bytes()
                config = yaml.safe_load(
                    scoring_config.read_text(encoding="utf-8")
                )
                config["params"] = {"revision": 2}
                scoring_config.write_text(
                    yaml.safe_dump(config, sort_keys=False),
                    encoding="utf-8",
                )

                def terminate_at(state):
                    if state == transition:
                        raise SimulatedProcessDeath(state)

                calls["fixture"] = 0
                with (
                    mock.patch.object(
                        score, "_publication_fault_point", terminate_at
                    ),
                    self.assertRaises(SimulatedProcessDeath),
                ):
                    self.invoke(
                        run_dir,
                        scoring_config,
                        self.registry(calls, value=0.9),
                    )

                recovery = score._reconcile_scoring_publication(run_dir)
                if transition in ("receipt_replaced", "new_pair_verified"):
                    self.assertEqual(recovery, "committed")
                    row = json.loads(
                        (run_dir / "scores.jsonl").read_text(encoding="utf-8")
                    )
                    self.assertEqual(row["fixture_score"], 0.9)
                else:
                    self.assertIn(recovery, ("clean", "restored"))
                    self.assertEqual(
                        (run_dir / "scores.jsonl").read_bytes(), old_scores
                    )
                    self.assertEqual(
                        (run_dir / score.SCORING_SUCCESS_NAME).read_bytes(),
                        old_receipt,
                    )
                self.assertFalse(
                    (run_dir / score.SCORING_PUBLICATION_JOURNAL_NAME).exists()
                )
                self.assertFalse(
                    any(
                        path.name.startswith(".scoring_publication")
                        for path in run_dir.iterdir()
                    )
                )

    def test_process_death_without_previous_pair_removes_partial_output(self) -> None:
        class SimulatedProcessDeath(BaseException):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            run_dir, scoring_config, _ = self.make_run(Path(temporary))
            calls = {"fixture": 0}

            def terminate_after_scores(state):
                if state == "scores_replaced":
                    raise SimulatedProcessDeath(state)

            with (
                mock.patch.object(
                    score, "_publication_fault_point", terminate_after_scores
                ),
                self.assertRaises(SimulatedProcessDeath),
            ):
                self.invoke(run_dir, scoring_config, self.registry(calls))
            self.assertTrue((run_dir / "scores.jsonl").exists())
            self.assertEqual(
                score._reconcile_scoring_publication(run_dir), "restored"
            )
            self.assertFalse((run_dir / "scores.jsonl").exists())
            self.assertFalse(
                (run_dir / score.SCORING_SUCCESS_NAME).exists()
            )

    def test_changed_durable_backup_blocks_unsafe_recovery(self) -> None:
        class SimulatedProcessDeath(BaseException):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, scoring_config, _ = self.make_run(root)
            calls = {"fixture": 0}
            self.invoke(run_dir, scoring_config, self.registry(calls))
            config = yaml.safe_load(scoring_config.read_text(encoding="utf-8"))
            config["params"] = {"revision": 2}
            scoring_config.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )

            def terminate_after_scores(state):
                if state == "scores_replaced":
                    raise SimulatedProcessDeath(state)

            calls["fixture"] = 0
            with (
                mock.patch.object(
                    score, "_publication_fault_point", terminate_after_scores
                ),
                self.assertRaises(SimulatedProcessDeath),
            ):
                self.invoke(
                    run_dir,
                    scoring_config,
                    self.registry(calls, value=0.9),
                )
            journal = json.loads(
                (run_dir / score.SCORING_PUBLICATION_JOURNAL_NAME).read_text(
                    encoding="utf-8"
                )
            )
            backup_path = run_dir / journal["old_publication"]["scores"]["path"]
            backup_path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "backup changed"):
                score._reconcile_scoring_publication(run_dir)
            self.assertTrue(
                (run_dir / score.SCORING_PUBLICATION_JOURNAL_NAME).is_file()
            )

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
                mock.patch("torch.cuda.current_device", return_value=2),
                mock.patch(
                    "torch.cuda.get_device_properties",
                    return_value=type(
                        "FixtureCudaProperties",
                        (),
                        {
                            "uuid": (
                                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
                            ),
                            "name": "NVIDIA GeForce RTX 3090",
                        },
                    )(),
                ),
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

    @staticmethod
    def write_stage_owner(run_dir, *, pid, process_start_ticks, run_identity=None):
        stage_dir = run_dir / score.SCORER_ASSET_STAGE_NAME
        stage_dir.mkdir()
        owner = {
            "schema": score.SCORER_ASSET_OWNER_SCHEMA,
            "attempt_id": "a" * 32,
            "pid": pid,
            "process_start_ticks": process_start_ticks,
            "boot_id": score._boot_id(),
            "run_directory_identity": list(
                run_identity
                or score._directory_owner_identity(run_dir.stat())
            ),
            "stage_directory_identity": list(
                score._directory_owner_identity(stage_dir.stat())
            ),
        }
        (stage_dir / score.SCORER_ASSET_OWNER_NAME).write_bytes(
            score._scorer_stage_owner_bytes(owner)
        )
        return stage_dir

    def test_stage_namespaces_assets_binds_manifest_and_cleans_nested_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "weights.bin"
            source.write_bytes(b"frozen-weights")
            registry = {
                "first": self.asset_class(source),
                "second": self.asset_class(source),
            }
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            with mock.patch.dict(score.REGISTRY, registry, clear=True):
                stage = score.ScorerAssetStage(
                    ["first", "second"], {}, run_dir=run_dir
                )
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
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            with mock.patch.dict(
                score.REGISTRY,
                {"fixture": self.asset_class(source)},
                clear=True,
            ):
                stage = score.ScorerAssetStage(
                    ["fixture"], {}, run_dir=run_dir
                )
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

    def test_stage_recovers_a_dead_owner_after_process_death(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            stale = self.write_stage_owner(
                run_dir, pid=2_147_483_647, process_start_ticks=1
            )
            (stale / "left-by-sigkill.bin").write_bytes(b"stale")
            source = root / "weights.bin"
            source.write_bytes(b"fresh")
            registry = {"fixture": self.asset_class(source)}
            stage = score.ScorerAssetStage(
                ["fixture"], {}, run_dir=run_dir, registry=registry
            )
            try:
                self.assertEqual(stage.path, stale)
                self.assertFalse((stale / "left-by-sigkill.bin").exists())
                stage.verify()
            finally:
                stage.cleanup()

    def test_stage_does_not_take_over_a_live_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            stage_dir = self.write_stage_owner(
                run_dir,
                pid=os.getpid(),
                process_start_ticks=score._process_start_ticks(os.getpid()),
            )
            with self.assertRaisesRegex(
                score.ScorerAssetStageError, "owned by a live process"
            ):
                score.ScorerAssetStage([], {}, run_dir=run_dir, registry={})
            self.assertTrue(stage_dir.is_dir())

    def test_stage_recovery_rejects_an_owner_inode_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            stage_dir = self.write_stage_owner(
                run_dir,
                pid=2_147_483_647,
                process_start_ticks=1,
                run_identity=(0, 0),
            )
            with self.assertRaisesRegex(
                score.ScorerAssetStageError, "directory identity differs"
            ):
                score.ScorerAssetStage([], {}, run_dir=run_dir, registry={})
            self.assertTrue(stage_dir.is_dir())

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

    def test_seccomp_blocks_cached_native_and_child_network_access(self):
        program = f"""
import array
import ctypes
import errno
import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, {str(EVAL_PIPELINE)!r})
spec = importlib.util.spec_from_file_location(
    'isolated_hps_score', {str(EVAL_PIPELINE / 'score_hpsv2_relational_renderer.py')!r}
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'hpsv2_scorers' not in sys.modules
import numpy
cached_socket = socket.socket
attempt = threading.Event()
result_ready = threading.Event()
release = threading.Event()
sibling_results = []
def sibling_socket_attempt():
    attempt.wait()
    try:
        sibling_socket = cached_socket()
    except OSError as exc:
        sibling_results.append(exc.errno)
    else:
        sibling_socket.close()
        sibling_results.append(None)
    finally:
        result_ready.set()
        release.wait()
sibling = threading.Thread(target=sibling_socket_attempt, daemon=True)
sibling.start()
assert module.install_network_seccomp_filter() == module.NETWORK_ISOLATION_SCHEMA
attempt.set()
assert result_ready.wait(10)
assert sibling_results == [errno.EPERM]
modes = module._process_seccomp_modes()
assert modes
assert all(mode == 2 for mode in modes.values())
try:
    cached_socket()
except OSError as exc:
    assert exc.errno == errno.EPERM
else:
    raise AssertionError('cached socket constructor bypassed seccomp')
libc = ctypes.CDLL(None, use_errno=True)
for syscall_number in module._NETWORK_DENIED_SYSCALLS_X86_64:
    ctypes.set_errno(0)
    assert libc.syscall(syscall_number, -1, -1, -1, -1, -1, -1) == -1
    assert ctypes.get_errno() == errno.EPERM, syscall_number
try:
    socket.socketpair(socket.AF_INET, socket.SOCK_STREAM)
except OSError as exc:
    assert exc.errno == errno.EPERM
else:
    raise AssertionError('non-UNIX socketpair bypassed seccomp')
left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    assert os.write(left.fileno(), b'x') == 1
    assert os.read(right.fileno(), 1) == b'x'
    descriptor, path = tempfile.mkstemp()
    try:
        rights = array.array('i', [descriptor])
        try:
            left.sendmsg([b'f'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)])
        except OSError as exc:
            assert exc.errno == errno.EPERM
        else:
            raise AssertionError('SCM_RIGHTS bypassed seccomp')
    finally:
        os.close(descriptor)
        os.unlink(path)
finally:
    left.close()
    right.close()
descriptor, path = tempfile.mkstemp()
try:
    assert os.write(descriptor, b'file-ok') == 7
finally:
    os.close(descriptor)
with open(path, 'rb') as handle:
    assert handle.read() == b'file-ok'
os.unlink(path)
ordinary_child = subprocess.run(
    [sys.executable, '-c', 'print("child-ok")'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    check=True,
)
assert ordinary_child.stdout.strip() == 'child-ok'
child = subprocess.run(
    [sys.executable, '-c', 'import socket; socket.socket()'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
assert child.returncode != 0
registry, _ = module._load_scorer_registry(formal_registered_scoring=True)
assert set(registry) == {{'imagereward', 'pixel', 'clip', 'hps', 'aesthetic', 'iqa'}}
release.set()
sibling.join(10)
assert not sibling.is_alive()
print('seccomp-ok')
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertIn("seccomp-ok", result.stdout)

    def test_seccomp_rejects_an_inherited_unix_socket(self):
        program = f"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    'inherited_socket_hps_score',
    {str(EVAL_PIPELINE / 'score_hpsv2_relational_renderer.py')!r},
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.install_network_seccomp_filter()
except RuntimeError as exc:
    assert 'inherited a socket descriptor' in str(exc)
else:
    raise AssertionError('inherited UNIX socket was accepted')
print('inherited-socket-rejected')
"""
        parent, child = socket.socketpair()
        try:
            result = subprocess.run(
                [sys.executable, "-c", program],
                pass_fds=(child.fileno(),),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        finally:
            child.close()
            parent.close()
        self.assertIn("inherited-socket-rejected", result.stdout)

    def test_formal_asset_inventory_closes_tokenizer_and_model_config_inputs(self):
        registry, _ = score._load_scorer_registry(
            formal_registered_scoring=True
        )
        params = {
            "patch_crops": 5,
            "clip_model": "ViT-B/32",
            "clipscore_w": 2.5,
        }
        inventories = {
            name: getattr(registry[name], "asset_sources", lambda **values: {})(
                **params
            )
            for name in ("imagereward", "pixel", "clip", "hps", "aesthetic", "iqa")
        }
        self.assertEqual(sum(map(len, inventories.values())), 31)
        self.assertIn("tokenizer_vocabulary", inventories["clip"])
        self.assertIn("model_config", inventories["hps"])
        for metric_inventory in inventories.values():
            for record in metric_inventory.values():
                self.assertTrue(Path(record["path"]).is_file())
        for name in ("clip", "hps"):
            packages = set(registry[name].PROVENANCE_PACKAGES)
            self.assertIn("ftfy", packages)
            self.assertIn("regex", packages)
        self.assertIn("safetensors", set(registry["iqa"].PROVENANCE_PACKAGES))

        closures = {}
        for name, scorer_class in registry.items():
            probe = scorer_class.__new__(scorer_class)
            self.assertEqual(
                provenance.scorer_framework_source_records(
                    probe, root=EVAL_PIPELINE
                ),
                [],
            )
            closures[name] = provenance.scorer_framework_source_records(
                probe,
                root=EVAL_PIPELINE,
                source_closure=provenance.HPSV2_PRIVATE_SOURCE_CLOSURE,
            )
            closure_paths = {record["path"] for record in closures[name]}
            self.assertIn("hpsv2_scorers/__init__.py", closure_paths)
            self.assertIn("hpsv2_scorers/base.py", closure_paths)
            self.assertIn(f"hpsv2_scorers/{name}_scorer.py", closure_paths)
            self.assertIn("scorers/__init__.py", closure_paths)
            self.assertIn("scorers/base.py", closure_paths)
            self.assertIn(f"scorers/{name}_scorer.py", closure_paths)
            self.assertIn("scorer_provenance.py", closure_paths)

        framework_sources = closures["clip"]
        sources_by_label = {record["label"]: record for record in framework_sources}
        self.assertEqual(
            sources_by_label["scorer_package_initializer"]["path"],
            "hpsv2_scorers/__init__.py",
        )
        self.assertEqual(
            sources_by_label["scorer_provenance_builder"]["path"],
            "scorer_provenance.py",
        )
        self.assertEqual(
            {record["path"] for record in framework_sources},
            {
                "hpsv2_scorers/__init__.py",
                "hpsv2_scorers/base.py",
                "hpsv2_scorers/clip_scorer.py",
                "scorer_provenance.py",
                "scorers/__init__.py",
                "scorers/base.py",
                "scorers/clip_scorer.py",
            },
        )
        pixel_manifest = {
            "schema": "repldm_scorer_asset_stage_v1",
            "loading_mode": score.SCORER_ASSET_LOADING_MODE,
            "files": [],
            "files_sha256": score.json_sha256([]),
        }
        pixel = registry["pixel"](
            device="cpu", scorer_asset_manifest=pixel_manifest
        )
        self.assertEqual(
            pixel.provenance_metadata()["parameters"]["network_isolation"],
            score.NETWORK_ISOLATION_SCHEMA,
        )

    def test_explicit_tokenizers_and_hps_config_match_legacy_inputs(self):
        import clip
        import torch
        from clip.simple_tokenizer import SimpleTokenizer as ClipTokenizer
        from hpsv2.src.open_clip import factory as hps_factory
        from hpsv2.src.open_clip import get_tokenizer as get_hps_tokenizer
        from hpsv2.src.open_clip.tokenizer import SimpleTokenizer as HPSTokenizer

        registry, _ = score._load_scorer_registry(
            formal_registered_scoring=True
        )
        clip_assets = registry["clip"].asset_sources(clip_model="ViT-B/32")
        clip_scorer = registry["clip"].__new__(registry["clip"])
        clip_scorer.tokenizer = ClipTokenizer(
            bpe_path=clip_assets["tokenizer_vocabulary"]["path"]
        )
        prompts = (
            "a detailed red bicycle beside a tree",
            "unicode cafe and a deliberately long " + "word " * 120,
        )
        for prompt in prompts:
            self.assertTrue(
                torch.equal(
                    clip_scorer._tokenize(prompt),
                    clip.tokenize([prompt], truncate=True),
                )
            )

        hps_assets = registry["hps"].asset_sources()
        staged_tokenizer = HPSTokenizer(
            bpe_path=hps_assets["tokenizer_vocabulary"]["path"]
        )
        hps_scorer = registry["hps"].__new__(registry["hps"])
        hps_scorer.simple_tokenizer = staged_tokenizer
        legacy_tokenizer = get_hps_tokenizer("ViT-H-14")
        self.assertIs(registry["hps"].TOKENIZER_TRUNCATE, True)
        for prompt in prompts:
            self.assertTrue(
                torch.equal(
                    hps_scorer._tokenize([prompt]),
                    legacy_tokenizer([prompt]),
                )
            )
        staged_config = json.loads(
            Path(hps_assets["model_config"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(staged_config, hps_factory.get_model_config("ViT-H-14"))

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
        import clip.simple_tokenizer

        registry, _ = score._load_scorer_registry(
            formal_registered_scoring=True
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip_b = root / "ViT-B-32.pt"
            clip_l = root / "ViT-L-14.pt"
            aesthetic = root / "aesthetic.pth"
            vocabulary = root / "bpe_simple_vocab_16e6.txt.gz"
            for path in (clip_b, clip_l, aesthetic, vocabulary):
                path.write_bytes(path.name.encode("ascii"))
            model = mock.Mock()

            def preprocess(image):
                return image

            with (
                mock.patch.object(
                    clip, "load", return_value=(model, preprocess)
                ) as clip_load,
                mock.patch.object(
                    clip.simple_tokenizer,
                    "SimpleTokenizer",
                    return_value=mock.Mock(),
                ) as tokenizer,
            ):
                scorer = registry["clip"](
                    device="cpu",
                    clip_model="ViT-B/32",
                    scorer_assets={
                        "clip_checkpoint": str(clip_b),
                        "tokenizer_vocabulary": str(vocabulary),
                    },
                    scorer_asset_manifest=self.fixture_manifest(clip_b),
                )
            self.assertEqual(clip_load.call_args.args[0], str(clip_b))
            self.assertEqual(
                tokenizer.call_args.kwargs["bpe_path"], str(vocabulary)
            )
            self.assertEqual(
                scorer.provenance_metadata()["parameters"]["asset_stage"][
                    "schema"
                ],
                "repldm_scorer_asset_stage_v1",
            )

            aesthetic_module = sys.modules[
                registry["aesthetic"].__module__
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
                registry["aesthetic"](
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
        import hpsv2.src.open_clip.factory as hps_factory
        import hpsv2.src.open_clip.tokenizer as hps_tokenizer
        import pyiqa
        import timm

        registry, _ = score._load_scorer_registry(
            formal_registered_scoring=True
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hps_checkpoint = root / "HPS_v2.1_compressed.pt"
            hps_backbone = root / "open_clip_pytorch_model.bin"
            vocabulary = root / "bpe.txt.gz"
            model_config = root / "ViT-H-14.json"
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
            model_config.write_text(
                json.dumps(
                    {"embed_dim": 8, "vision_cfg": {}, "text_cfg": {}}
                ),
                encoding="utf-8",
            )

            model = mock.Mock()
            model.to.return_value = model
            model.eval.return_value = model
            def hps_preprocess(image):
                return image

            hps_module = sys.modules[registry["hps"].__module__]
            with (
                mock.patch.object(
                    hps_factory,
                    "create_model_and_transforms",
                    return_value=(model, None, hps_preprocess),
                ) as create_hps,
                mock.patch.object(
                    hps_tokenizer, "SimpleTokenizer", return_value=mock.Mock()
                ) as tokenizer,
                mock.patch.object(
                    hps_module.torch,
                    "load",
                    return_value={"state_dict": {}},
                ) as torch_load,
                mock.patch.object(
                    hps_factory,
                    "_MODEL_CONFIGS",
                    dict(hps_factory._MODEL_CONFIGS),
                ),
            ):
                hps_instance = registry["hps"](
                    device="cpu",
                    scorer_assets={
                        "hpsv2_checkpoint": str(hps_checkpoint),
                        "open_clip_backbone": str(hps_backbone),
                        "tokenizer_vocabulary": str(vocabulary),
                        "model_config": str(model_config),
                    },
                    scorer_asset_manifest=self.fixture_manifest(hps_checkpoint),
                )
            self.assertEqual(
                create_hps.call_args.kwargs["pretrained"], str(hps_backbone)
            )
            self.assertEqual(torch_load.call_args.args[0], str(hps_checkpoint))
            self.assertEqual(tokenizer.call_args.kwargs["bpe_path"], str(vocabulary))
            self.assertIs(
                hps_instance.provenance_metadata()["preprocessing"][
                    "text_tokenizer"
                ]["truncate"],
                True,
            )

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
                registry["iqa"](
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
