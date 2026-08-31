import contextlib
import copy
from collections import Counter, defaultdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "eval-pipeline/generate_hpsv2_relational_renderer.py"
SPEC = importlib.util.spec_from_file_location("hpsv2_full_generator", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def png_header(width=1024, height=1024):
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def generation_environment_fixture():
    environment = {
        "schema": runner.GENERATION_ENVIRONMENT_SCHEMA,
        "platform": "linux-x86_64",
        "python_version": "3.11.10",
        "python_implementation": "CPython",
        "python_executable": "/test/python",
        "packages": {name: "test" for name in runner.GENERATION_PACKAGES},
        "torch_build_version": "test",
        "cuda_runtime_version": "test",
        "cudnn_version": 1,
        "worker_determinism": dict(runner.WORKER_DETERMINISM),
        "gpu_inventory": [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "pci_bus_id": f"0000:{index:02d}:00.0",
                "name": "NVIDIA GeForce RTX 3090",
                "total_memory_mib": 24576,
                "driver_version": "test-driver",
            }
            for index in runner.EXPECTED_DEVICES
        ],
    }
    environment["sha256"] = runner.canonical_sha256(environment)
    return environment


class FrozenMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = runner.load_contract()
        cls.tasks = runner.build_tasks(cls.contract, runner.EXPECTED_DEVICES)

    def test_frozen_assets_and_scoring_contract(self):
        self.assertEqual(len(self.contract["prompts"]), 3200)
        self.assertEqual(
            self.contract["prompt_csv_sha256"],
            "bf51e0eb6659dcbf5666df1f6813334fb10f65bc40018f22c83abc6a801463c7",
        )
        self.assertEqual(
            self.contract["prompt_manifest_sha256"],
            "8df43332f69e46a5457a10aad16da168cc7db365064e7e8cdc8c2fc7a7102c91",
        )
        manifest = self.contract["prompt_manifest_value"]
        self.assertEqual(manifest["exact_unique_prompt_count"], 3172)
        self.assertEqual(manifest["official_duplicate_row_count"], 28)
        self.assertEqual(
            self.contract["config"]["scoring"]["config_sha256"],
            "22a162c7fc2151bbcc8ab08e5a4e3604de6fd56a955e37078577f8e0bd59dfac",
        )
        self.assertEqual(
            self.contract["config"]["scoring"][
                "registered_scorer_provenance_sha256"
            ],
            "af8ec22593bb551b0af06c56a692211ba48a9e52f40e13e5557a04fd53f747e3",
        )

    def test_analysis_and_duplicate_contracts_fail_closed(self):
        altered = copy.deepcopy(self.contract["config"])
        altered["analysis"]["guards"]["topiq_nr_ci_lower_min_delta"] = -1.0
        with self.assertRaisesRegex(ValueError, "analysis"):
            runner._validate_scoring_and_analysis(altered)
        manifest = copy.deepcopy(self.contract["prompt_manifest_value"])
        manifest["official_duplicate_row_count"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate-row"):
            runner._validate_prompt_manifest(
                manifest,
                self.contract["prompt_csv"],
                self.contract["prompt_csv_sha256"],
            )

    def test_exact_cartesian_product_and_balanced_pairing(self):
        self.assertEqual(len(self.tasks), 12800)
        self.assertEqual(len({task["id"] for task in self.tasks}), 12800)
        self.assertEqual(len({task["seed"] for task in self.tasks}), 3200)
        for task in self.tasks:
            self.assertEqual(task["seed"], 20260831 + task["prompt_index"])
        device_counts = Counter(task["physical_device_index"] for task in self.tasks)
        self.assertEqual(device_counts, {2: 3200, 5: 3200, 6: 3200, 7: 3200})
        blocks = defaultdict(list)
        for task in self.tasks:
            blocks[task["prompt_index"]].append(task)
        self.assertEqual(len(blocks), 3200)
        expected_settings = {row[0] for row in runner.EXPECTED_SETTINGS}
        for rows in blocks.values():
            self.assertEqual({row["action_id"] for row in rows}, expected_settings)
            self.assertEqual(len({row["physical_device_index"] for row in rows}), 1)
            self.assertEqual(sorted(row["execution_rank"] for row in rows), [0, 1, 2, 3])

    def test_setting_order_is_deterministic(self):
        repeated = runner.build_tasks(self.contract, runner.EXPECTED_DEVICES)
        self.assertEqual(
            [task["id"] for task in self.tasks],
            [task["id"] for task in repeated],
        )
        self.assertEqual(runner.canonical_sha256(self.tasks), runner.canonical_sha256(repeated))

    def test_cli_rejects_device_or_subset_overrides(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.parse_args(["--devices", "4,5,6,7"])
            with self.assertRaises(SystemExit):
                runner.parse_args(["--devices", "2,5,6,7", "--seeds", "1"])
            with self.assertRaises(SystemExit):
                runner.parse_args(
                    ["--devices", "2,5,6,7", "--validate-only", "--audit-only"]
                )

    def _matrix_rows(self):
        same_image_hash = "a" * 64
        rows = []
        for task in self.tasks:
            active = not task["action"]["physical_no_op"]
            rows.append(
                {
                    "schema": runner.MANIFEST_SCHEMA,
                    **task,
                    "image_path": f"images/{task['id']}.png",
                    "image_sha256": same_image_hash,
                    "initial_latent_sha256": digest(task["prompt_index"]),
                    "final_latent_sha256": digest(f"final:{task['id']}"),
                    "scheduler_schedule_sha256": "b" * 64,
                    "latent_renderer_scheduler_mapping": task["action"][
                        "scheduler_mapping"
                    ],
                    "renderer_active": active,
                    "renderer_step_count": 50 if active else 0,
                    "worker_gpu_uuid": f"GPU-{task['physical_device_index']}",
                    "worker_pci_bus_id": f"0000:{task['physical_device_index']:02d}:00.0",
                    "sidecar_path": f"images/{task['id']}.json",
                    "sidecar_sha256": digest(f"sidecar:{task['id']}"),
                    "generation_attempt_receipt_sha256": digest(
                        f"receipt:{task['id']}"
                    ),
                }
            )
        return rows

    def test_complete_matrix_accepts_byte_identical_images(self):
        rows = self._matrix_rows()
        runner.validate_complete_matrix(rows, self.tasks)
        self.assertEqual(len({row["image_sha256"] for row in rows}), 1)

    def test_complete_matrix_rejects_missing_or_inactive_renderer(self):
        rows = self._matrix_rows()
        with self.assertRaisesRegex(ValueError, "12,800"):
            runner.validate_complete_matrix(rows[:-1], self.tasks)
        active_index = next(
            index for index, row in enumerate(rows) if row["action_id"] != "no_ag"
        )
        rows[active_index] = {**rows[active_index], "renderer_active": False}
        with self.assertRaisesRegex(ValueError, "activation"):
            runner.validate_complete_matrix(rows, self.tasks)

    def test_complete_matrix_rejects_physical_gpu_drift_within_prompt(self):
        rows = self._matrix_rows()
        prompt_rows = [row for row in rows if row["prompt_index"] == 0]
        prompt_rows[0]["worker_gpu_uuid"] = "GPU-replaced"
        with self.assertRaisesRegex(ValueError, "physical GPU UUID"):
            runner.validate_complete_matrix(rows, self.tasks)

    def test_run_contract_binds_runtime_and_loaded_model_files(self):
        environment = generation_environment_fixture()
        model = {
            **runner.model_snapshot.expected_model_manifest(),
            "manifest_sha256": runner.model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256,
        }
        first = runner._run_config(
            self.contract,
            runner.EXPECTED_DEVICES,
            self.tasks,
            "e" * 40,
            generation_environment=environment,
            model_snapshot_record=model,
        )
        runner._validate_recorded_run_config(first, self.contract, self.tasks)
        changed = copy.deepcopy(environment)
        changed["packages"]["torch"] = "changed"
        changed.pop("sha256")
        changed["sha256"] = runner.canonical_sha256(changed)
        second = runner._run_config(
            self.contract,
            runner.EXPECTED_DEVICES,
            self.tasks,
            "e" * 40,
            generation_environment=changed,
            model_snapshot_record=model,
        )
        self.assertNotEqual(
            first["run_contract_sha256"], second["run_contract_sha256"]
        )

    def test_worker_model_stage_evidence_requires_bound_pre_post_verification(self):
        stage = {
            "manifest_sha256": runner.model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256,
            "loaded_file_count": 18,
        }
        verification = {
            "schema": runner.model_snapshot.MODEL_STAGE_VERIFICATION_SCHEMA,
            "status": "verified_unchanged",
            "manifest_sha256": runner.model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256,
            "loaded_file_count": 18,
            "tree_sha256": "a" * 64,
        }
        evidence = runner._worker_model_snapshot_evidence(
            stage, [dict(verification) for _ in range(3)]
        )
        runner._validate_worker_model_snapshot_evidence(evidence)
        with self.assertRaisesRegex(RuntimeError, "pre/post"):
            runner._worker_model_snapshot_evidence(stage, [verification, verification])

    def test_worker_failure_still_runs_source_model_postaudit(self):
        expected = {
            **runner.model_snapshot.expected_model_manifest(),
            "manifest_sha256": runner.model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256,
        }
        with mock.patch.object(
            runner, "_run_workers", side_effect=RuntimeError("worker failed")
        ), mock.patch.object(
            runner.model_snapshot,
            "validate_model_snapshot",
            return_value=expected,
        ) as post_audit:
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                runner._run_workers_with_model_postaudit(
                    [], runner.EXPECTED_DEVICES, self.contract, {}, expected
                )
        post_audit.assert_called_once_with(runner.ROOT)

    def test_gpu_monitor_allows_only_registered_worker_pids(self):
        environment = generation_environment_fixture()
        process_rows = (
            "GPU-2, 111, worker, 100\n"
            "GPU-5, 999, foreign, 100\n"
            "GPU-0, 888, unrelated, 100\n"
        )
        with mock.patch.object(runner, "_run_command", return_value=process_rows):
            conflicts = runner._generation_compute_conflicts(
                environment, allowed_pids=[111]
            )
        self.assertEqual([row["pid"] for row in conflicts], [999])

    def test_monitor_command_timeout_fails_closed(self):
        timeout = runner.subprocess.TimeoutExpired(["nvidia-smi"], 10)
        with mock.patch.object(runner.subprocess, "run", side_effect=timeout):
            with self.assertRaisesRegex(RuntimeError, "timed out after 10s"):
                runner._run_command(["nvidia-smi"])

    def test_worker_device_identity_uses_timed_command_wrapper(self):
        torch = mock.Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 8
        torch.cuda.current_device.return_value = 2
        class Properties:
            name = "NVIDIA GeForce RTX 3090"
            major = 8
            minor = 6
            total_memory = 24 * 1024**3

        torch.cuda.get_device_properties.return_value = Properties()
        inventory = "2, GPU-2, 0000:02:00.0, NVIDIA GeForce RTX 3090, 24576"
        with mock.patch.dict(runner.os.environ, {}, clear=True), mock.patch.object(
            runner, "_run_command", return_value=inventory
        ) as command:
            identity = runner._cuda_device_identity(torch, 2)
        command.assert_called_once_with(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        self.assertEqual(identity["physical_device_index"], 2)
        self.assertEqual(identity["gpu_uuid"], "GPU-2")

    def test_abort_prevents_artifact_publication(self):
        abort_event = mock.Mock()
        abort_event.is_set.return_value = True
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runner, "_atomic_write_bytes"
        ) as write_png, mock.patch.object(
            runner, "_atomic_write_json"
        ) as write_json:
            output = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "aborted before artifact"):
                runner._publish_generated_artifacts(
                    png_path=output / "image.png",
                    json_path=output / "image.json",
                    png=b"image",
                    sidecar={"id": "test"},
                    abort_event=abort_event,
                    publication_lock=contextlib.nullcontext(),
                )
        write_png.assert_not_called()
        write_json.assert_not_called()

    def test_partial_worker_start_failure_reaps_started_worker(self):
        class FakeEvent:
            def __init__(self):
                self.was_set = False

            def set(self):
                self.was_set = True

            def is_set(self):
                return self.was_set

        class FakeQueue:
            def get(self, timeout=None):
                raise runner.queue.Empty

            def get_nowait(self):
                raise runner.queue.Empty

        class FakeProcess:
            def __init__(self, fail_start=False, **kwargs):
                self.name = kwargs["name"]
                self.pid = None
                self.exitcode = None
                self.alive = False
                self.fail_start = fail_start
                self.terminated = False

            def start(self):
                if self.fail_start:
                    raise OSError("spawn failed")
                self.pid = 111
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                return None

            def terminate(self):
                self.terminated = True
                self.alive = False
                self.exitcode = -15

        event = FakeEvent()
        created = []

        class FakeContext:
            def Event(self):
                return event

            def Lock(self):
                return contextlib.nullcontext()

            def Queue(self):
                return FakeQueue()

            def Process(self, **kwargs):
                process = FakeProcess(fail_start=len(created) == 1, **kwargs)
                created.append(process)
                return process

        pending = [
            {"physical_device_index": 2, "prompt_index": 0, "execution_rank": 0},
            {"physical_device_index": 5, "prompt_index": 1, "execution_rank": 0},
        ]
        contract = {
            "sampling": {},
            "output_dir": "/tmp/staging",
            "attempt_id": "1" * 32,
            "attempt_sha256": "2" * 64,
        }
        run_config = {"generation_environment": generation_environment_fixture()}
        with mock.patch.object(
            runner.multiprocessing, "get_context", return_value=FakeContext()
        ):
            with self.assertRaisesRegex(RuntimeError, "worker lifecycle"):
                runner._run_workers(pending, (2, 5), contract, run_config)
        self.assertTrue(event.was_set)
        self.assertTrue(created[0].terminated)


class ArtifactAndDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contract = runner.load_contract()
        cls.tasks = runner.build_tasks(contract, runner.EXPECTED_DEVICES)

    def test_atomic_write_and_orphan_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / "nested/value.json"
            runner._atomic_write_json(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            self.assertFalse(any(path.parent.glob(".*.tmp-*")))

            task = {"id": "orphan"}
            png_path, json_path = runner._image_paths(output, "orphan")
            runner._atomic_write_bytes(png_path, png_header())
            self.assertEqual(runner._quarantine_orphans(output, [task]), 1)
            self.assertFalse(png_path.exists())
            self.assertFalse(json_path.exists())
            self.assertEqual(len(list((output / ".incomplete").iterdir())), 1)

    def _baseline_sidecar(self, task, output):
        png_path, _ = runner._image_paths(output, task["id"])
        payload = png_header()
        runner._atomic_write_bytes(png_path, payload)
        states = [digest(f"latent:{index}") for index in range(51)]
        schedule_payload = {
            "timesteps": [float(index) for index in range(50)],
            "sigmas": [float(index + 1) for index in range(51)],
        }
        model_evidence = {
            "schema": runner.WORKER_MODEL_SNAPSHOT_SCHEMA,
            "loader": "pinned_proc_fd_with_pre_post_tree_verification",
            "model_id": runner.model_snapshot.MODEL_ID,
            "revision": runner.model_snapshot.MODEL_REVISION,
            "variant": "fp16",
            "manifest_sha256": runner.model_snapshot.MODEL_SNAPSHOT_MANIFEST_SHA256,
            "loaded_file_count": 18,
            "bound_load_verification_count": 3,
            "tree_unchanged_during_load": True,
        }
        return {
            "schema": runner.SIDECAR_SCHEMA,
            "experiment_id": runner.EXPECTED_EXPERIMENT_ID,
            **task,
            "run_contract_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "git_commit": "e" * 40,
            "generation_environment_sha256": "f" * 64,
            "generation_attempt_id": "1" * 32,
            "generation_attempt_sha256": "2" * 64,
            "image_path": f"images/{task['id']}.png",
            "image_sha256": runner.sha256_bytes(payload),
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "guidance_rescale": 0.0,
            "unet_calls_per_step": [1] * 50,
            "scheduler_calls_per_step": [1] * 50,
            "call_totals": {
                "unet_calls": 50,
                "scheduler_calls": 50,
                "extra_unet_calls": 0,
                "extra_scheduler_calls": 0,
                "backbone_backward_calls": 0,
                "intermediate_decode_calls": 0,
                "final_decode_calls": 1,
            },
            "scheduler_schedule": {
                **schedule_payload,
                "schedule_sha256": runner.canonical_sha256(schedule_payload),
                "construction_init_noise_sigma": 1.0,
                "effective_init_noise_sigma": 1.0,
            },
            "initial_latent_sha256": states[0],
            "final_latent_sha256": states[-1],
            "latents_before_step_sha256": states[:-1],
            "latents_after_step_sha256": states[1:],
            "latent_renderer_scheduler_mapping": None,
            "latent_renderer_diagnostics": None,
            "latent_renderer_provider_diagnostics": None,
            "latent_renderer_step_diagnostics": [],
            "worker_device_provenance": {
                "requested_device": task["device"],
                "logical_device_index": task["physical_device_index"],
                "physical_device_index": task["physical_device_index"],
                "gpu_uuid": "GPU-test",
                "pci_bus_id": "0000:01:00.0",
                "gpu": "test-gpu",
                "cuda_visible_devices": None,
            },
            "worker_model_snapshot_provenance": model_evidence,
            "worker_model_snapshot_sha256": runner.canonical_sha256(model_evidence),
        }

    @staticmethod
    def _attempt_run_config():
        return {
            "run_contract_sha256": "c" * 64,
            "config": {"sha256": "d" * 64},
            "git_commit": "e" * 40,
            "generation_environment": {"sha256": "f" * 64},
        }

    def _stage_single_task_attempt(self, output, task):
        run_config = self._attempt_run_config()
        (output / "images").mkdir(parents=True)
        attempt, attempt_sha256, staging = runner._begin_generation_attempt(
            output, [task], run_config
        )
        sidecar = self._baseline_sidecar(task, staging)
        sidecar["generation_attempt_id"] = attempt["attempt_id"]
        sidecar["generation_attempt_sha256"] = attempt_sha256
        _, sidecar_path = runner._image_paths(staging, task["id"])
        runner._atomic_write_json(sidecar_path, sidecar)
        return run_config, attempt, staging

    def test_only_accepted_attempt_artifacts_can_resume(self):
        task = next(task for task in self.tasks if task["action_id"] == "no_ag")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_config, attempt, staging = self._stage_single_task_attempt(
                output, task
            )
            receipt = runner._promote_and_accept_generation_attempt(
                output, staging, [task], run_config, attempt
            )
            accepted, running = runner._load_generation_attempt_index(
                output, run_config, [task]
            )
            self.assertFalse(running)
            row = runner._load_complete_record(
                output, task, run_config, accepted
            )
            self.assertEqual(row["generation_attempt_id"], attempt["attempt_id"])
            receipt_path = runner._attempt_paths(
                output, attempt["attempt_id"]
            )["accepted"]
            self.assertEqual(
                row["generation_attempt_receipt_sha256"],
                runner.sha256_file(receipt_path),
            )
            self.assertEqual(receipt["task_count"], 1)

    def test_complete_pair_without_accepted_receipt_is_quarantined(self):
        task = next(task for task in self.tasks if task["action_id"] == "no_ag")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "images").mkdir()
            sidecar = self._baseline_sidecar(task, output)
            _, sidecar_path = runner._image_paths(output, task["id"])
            runner._atomic_write_json(sidecar_path, sidecar)
            moved = runner._quarantine_unaccepted_artifacts(
                output, [task], {}
            )
            self.assertEqual(moved, 2)
            png_path, sidecar_path = runner._image_paths(output, task["id"])
            self.assertFalse(png_path.exists())
            self.assertFalse(sidecar_path.exists())

    def test_failed_promotion_is_poisoned_and_cannot_resume(self):
        task = next(task for task in self.tasks if task["action_id"] == "no_ag")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_config, attempt, staging = self._stage_single_task_attempt(
                output, task
            )
            staged_png, staged_json = runner._image_paths(staging, task["id"])
            real_replace = runner.os.replace

            def fail_on_sidecar(source, target):
                if Path(source) == staged_json:
                    raise OSError("promotion failed")
                return real_replace(source, target)

            with mock.patch.object(runner.os, "replace", side_effect=fail_on_sidecar):
                with self.assertRaisesRegex(OSError, "promotion failed"):
                    runner._promote_and_accept_generation_attempt(
                        output, staging, [task], run_config, attempt
                    )
            runner._poison_generation_attempt(
                output, attempt, RuntimeError("promotion failed")
            )
            accepted, running = runner._load_generation_attempt_index(
                output, run_config, [task]
            )
            self.assertEqual(accepted, {})
            self.assertFalse(running)
            canonical_png, canonical_json = runner._image_paths(output, task["id"])
            self.assertFalse(canonical_png.exists())
            self.assertFalse(canonical_json.exists())
            self.assertTrue(staged_png.exists() or staged_json.exists())

    def test_accepted_receipt_and_artifact_tampering_fail_closed(self):
        task = next(task for task in self.tasks if task["action_id"] == "no_ag")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_config, attempt, staging = self._stage_single_task_attempt(
                output, task
            )
            runner._promote_and_accept_generation_attempt(
                output, staging, [task], run_config, attempt
            )
            receipt_path = runner._attempt_paths(
                output, attempt["attempt_id"]
            )["accepted"]
            receipt = json.loads(receipt_path.read_text())
            receipt["artifacts"] = []
            receipt["task_count"] = 0
            receipt["artifacts_sha256"] = runner.canonical_sha256([])
            runner._atomic_write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "artifact set"):
                runner._load_generation_attempt_index(output, run_config, [task])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_config, attempt, staging = self._stage_single_task_attempt(
                output, task
            )
            runner._promote_and_accept_generation_attempt(
                output, staging, [task], run_config, attempt
            )
            accepted, _ = runner._load_generation_attempt_index(
                output, run_config, [task]
            )
            png_path, _ = runner._image_paths(output, task["id"])
            png_path.write_bytes(png_header() + b"tampered")
            with self.assertRaisesRegex(ValueError, "missing, unsafe, or changed"):
                runner._load_complete_record(output, task, run_config, accepted)

    def test_accepted_receipt_with_missing_canonical_pair_fails_closed(self):
        task = next(task for task in self.tasks if task["action_id"] == "no_ag")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_config, attempt, staging = self._stage_single_task_attempt(
                output, task
            )
            runner._promote_and_accept_generation_attempt(
                output, staging, [task], run_config, attempt
            )
            accepted, _ = runner._load_generation_attempt_index(
                output, run_config, [task]
            )
            png_path, sidecar_path = runner._image_paths(output, task["id"])
            png_path.unlink()
            sidecar_path.unlink()
            with self.assertRaisesRegex(ValueError, "accepted output pair is missing"):
                runner._load_complete_record(output, task, run_config, accepted)

    def test_sidecar_resume_validation_and_compact_score_manifest(self):
        task = next(task for task in self.tasks if task["action_id"] == "no_ag")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            record = self._baseline_sidecar(task, output)
            runner.validate_sidecar(
                record,
                task,
                run_contract_sha256="c" * 64,
                config_sha256="d" * 64,
                git_commit="e" * 40,
                generation_environment_sha256="f" * 64,
                generation_attempt_id="1" * 32,
                generation_attempt_sha256="2" * 64,
                output_dir=output,
            )
            _, json_path = runner._image_paths(output, task["id"])
            runner._atomic_write_json(json_path, record)
            row = runner._manifest_row(
                record, json_path, accepted_receipt_sha256="3" * 64
            )
            self.assertTrue(set(runner.SCORE_MANIFEST_FIELDS).issubset(row))
            self.assertNotIn("latent_renderer_step_diagnostics", row)
            self.assertEqual(row["renderer_step_count"], 0)

            broken = copy.deepcopy(record)
            broken["latents_after_step_sha256"][0] = "f" * 64
            with self.assertRaisesRegex(ValueError, "hash chain"):
                runner.validate_sidecar(
                    broken,
                    task,
                    run_contract_sha256="c" * 64,
                    config_sha256="d" * 64,
                    git_commit="e" * 40,
                    generation_environment_sha256="f" * 64,
                    generation_attempt_id="1" * 32,
                    generation_attempt_sha256="2" * 64,
                    output_dir=output,
                )

    def _random_provider(self, task, step_index):
        hashes = [digest(f"counter:{index}") for index in range(3)]
        return {
            "implementation": "adaptive_oracle_local_relational_basis_v1",
            "affinity_source": "random_edge",
            "selected_orbit": "axis-r1",
            "selected_orbit_index": 0,
            "selected_basis_shape": [1, 1, 4, 128, 128],
            "selected_basis_dtype": "float32",
            "selected_basis_sha256": digest(f"basis:{step_index}"),
            "local_diagnostics": {
                "affinity_source": "random_edge",
                "orbit_names": ["axis-r1", "diagonal-r1", "axis-r2"],
                "grid_size": 16,
                "random_edge_counter_schema": runner.RANDOM_EDGE_COUNTER_SCHEMA,
                "random_edge_actual_edge_counts": [480, 450, 448],
                "random_edge_unique_canonical_key_counts": [64, 64, 56],
                "random_edge_counter_set_sha256": hashes,
                "random_edge_actual_edges_unique": True,
            },
            "random_counter_context": {
                "schema": runner.RANDOM_EDGE_COUNTER_SCHEMA,
                "experiment_id": runner.EXPECTED_EXPERIMENT_ID,
                "split_role": "hpsv2_full",
                "prompt_row_id": task["prompt_row_id"],
                "seed": task["seed"],
                "step_index": step_index,
                "orbit_name": "axis-r1",
            },
            "random_counter_set_sha256": hashes[0],
        }

    def test_real_random_schema_and_solver_target_are_accepted(self):
        task = next(
            task for task in self.tasks if task["action_id"] == "random_axis_r1_pos"
        )
        steps = []
        for index in range(50):
            steps.append(
                {
                    "step_index": index,
                    "scheduler_step_index": index,
                    "prediction_type": "epsilon",
                    "sign": 1,
                    "applied_update_ratio": [0.0195],
                    "target_ratio_error": [0.0],
                    "target_update_ratio": [0.0195],
                    "covariance_drift": [0.001],
                    "cap_hit": [False],
                    "native_scheduler_round_trip": {
                        "pred_original_sample_relative_l2_error": [0.001],
                        "expected_prev_sample_relative_l2_error": [0.0001],
                        "native_round_trip_max_abs_error": [0.01],
                    },
                    "scheduler_mapped_intervention": {
                        "applied_update_ratio": [0.02],
                        "target_ratio_error": [0.0],
                        "cap_hit": [False],
                        "solver_target_update_ratio": [0.0195],
                        "solver_evaluations": 15,
                    },
                    "provider_diagnostics": self._random_provider(task, index),
                }
            )
        runner._validate_renderer_steps(steps, task)
        broken = copy.deepcopy(steps)
        broken[0]["provider_diagnostics"]["random_counter_context"]["schema"] = "old"
        with self.assertRaisesRegex(ValueError, "counter context"):
            runner._validate_renderer_steps(broken, task)


class QueueContractTest(unittest.TestCase):
    def test_queue_waits_for_all_frozen_devices_then_audits_and_scores(self):
        source = (ROOT / "eval-pipeline/run_hpsv2_relational_renderer_queue.sh").read_text()
        self.assertIn("DEVICES=2,5,6,7", source)
        self.assertIn("waiting_for_all_gpus", source)
        self.assertIn("--query-compute-apps", source)
        self.assertIn("waiting_for_scoring_gpu", source)
        self.assertLess(source.index("waiting_for_scoring_gpu"), source.index("--device cuda:2"))
        self.assertIn("--audit-only", source)
        self.assertIn("--require-scorer-provenance", source)
        self.assertIn("--require-exclusive-gpu", source)
        self.assertIn("--device cuda:2", source)
        self.assertIn("nvidia_smi_gpu_error", source)
        self.assertIn("unexpected_queue_error", source)
        self.assertIn('timeout --foreground "$GPU_QUERY_TIMEOUT_SECONDS"', source)
        self.assertIn("validate_repository_snapshot", source)
        self.assertIn('git -C "$ROOT" status --porcelain=v1 --untracked-files=all', source)
        self.assertIn("refs/remotes/origin/rl-version", source)
        self.assertIn("pre_scoring_repository_snapshot", source)
        self.assertGreaterEqual(
            source.count("require_repository_snapshot_or_fail"),
            6,
        )
        self.assertIn("load_from_verified_staged_model_snapshot", RUNNER_PATH.read_text())
        self.assertNotIn(".generation.lock", source)


if __name__ == "__main__":
    unittest.main()
