import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib


ROOT = Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module(
    "audit_adaptive_oracle_engineering_test",
    EVAL_PIPELINE / "audit_adaptive_oracle_engineering.py",
)
contract = audit.contract
generation = audit.generation


def digest(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def png_chunk(kind, payload):
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def rgb_scanlines(width=contract.WIDTH, height=contract.HEIGHT, filter_byte=0):
    return (bytes([filter_byte]) + b"\x00" * (width * 3)) * height


def png_bytes(
    width=contract.WIDTH,
    height=contract.HEIGHT,
    *,
    bit_depth=8,
    color_type=2,
    interlace=0,
    scanlines=None,
    compressed_idat=None,
    idat_chunk_count=1,
):
    ihdr = struct.pack(
        ">IIBBBBB", width, height, bit_depth, color_type, 0, 0, interlace
    )
    if scanlines is None:
        scanlines = rgb_scanlines(width, height)
    if compressed_idat is None:
        compressed_idat = zlib.compress(scanlines)
    if idat_chunk_count <= 0:
        raise ValueError("test PNG must contain at least one IDAT chunk")
    boundaries = [
        len(compressed_idat) * index // idat_chunk_count
        for index in range(idat_chunk_count + 1)
    ]
    idat_chunks = b"".join(
        png_chunk(b"IDAT", compressed_idat[boundaries[index] : boundaries[index + 1]])
        for index in range(idat_chunk_count)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + idat_chunks
        + png_chunk(b"IEND", b"")
    )


class StableRegularFileReadTest(unittest.TestCase):
    def test_symlink_is_rejected_without_following_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_bytes(b"target\n")
            link = root / "link.json"
            link.symlink_to(target)

            with self.assertRaisesRegex(ValueError, "non-symlink"):
                audit._read_regular_bytes(link, "fixture input")
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                audit.sha256_file(link)

    def test_path_replacement_during_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_bytes(b"original\n")
            replacement = root / "replacement.json"
            replacement.write_bytes(b"replacement\n")
            original_read = os.read
            replaced = False

            def replace_after_read(descriptor, count):
                nonlocal replaced
                chunk = original_read(descriptor, count)
                if not replaced:
                    replaced = True
                    source.unlink()
                    source.symlink_to(replacement)
                return chunk

            with mock.patch.object(audit.os, "read", side_effect=replace_after_read):
                with self.assertRaisesRegex(ValueError, "changed while it was read"):
                    audit._read_regular_bytes(source, "fixture input")

    def test_in_place_mutation_during_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            source.write_bytes(b"original")
            original_read = os.read
            mutated = False

            def mutate_after_read(descriptor, count):
                nonlocal mutated
                chunk = original_read(descriptor, count)
                if not mutated:
                    mutated = True
                    source.write_bytes(b"mutated-and-longer")
                return chunk

            with mock.patch.object(audit.os, "read", side_effect=mutate_after_read):
                with self.assertRaisesRegex(ValueError, "changed while it was read"):
                    audit.sha256_file(source)


class EngineeringRunFixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.run = self.repo / generation.OUTPUT_DIR
        (self.run / "records").mkdir(parents=True)
        self.authorization = self.repo / "authorization.yaml"
        self.authorization.write_text("fixture\n", encoding="utf-8")
        self.png = png_bytes()
        self.tasks = []
        for prompt_index in range(contract.PROMPT_COUNT):
            prompt_row_id = f"engineering-{prompt_index + 1:04d}"
            for action_index in range(contract.TASKS_PER_PROMPT):
                action_id = f"A{action_index:02d}"
                self.tasks.append(
                    {
                        "task_id": f"{prompt_row_id}--{action_id}",
                        "prompt": {"prompt_row_id": prompt_row_id},
                        "action": {
                            "action_id": action_id,
                            "physical_no_op": action_index == 0,
                        },
                    }
                )
        self.design = {
            "schema": "fixture_design_v1",
            "task_count": contract.TOTAL_TASK_COUNT,
            "tasks_per_prompt": contract.TASKS_PER_PROMPT,
            "tasks_sha256": contract.canonical_sha256(self.tasks),
            "tasks": self.tasks,
            "design_sha256": digest("fixture-design"),
        }
        self.validated = {
            "authorization_sha256": digest("fixture-authorization"),
            "authorization_binding": {
                "path": generation.AUTHORIZATION_PATH,
                "sha256": digest("fixture-authorization"),
                "commit": "c" * 40,
            },
            "reviewed_commit": "a" * 40,
            "config": {
                "schema": generation.AUTHORIZATION_SCHEMA,
                "authorization": {
                    "scoring": False,
                    "quality_inspection": False,
                },
            },
            "design": self.design,
            "device": "cuda:1",
            "output_dir": str(self.run),
            "repo_root": str(self.repo),
            "launcher_evidence": {
                "schema": generation.LAUNCH_SCHEMA,
                "trust_root": generation.LAUNCH_TRUST_ROOT,
                "reviewed_commit": "a" * 40,
                "module_execution": {},
            },
        }
        environment_lock = self.repo / "environment-lock.yaml"
        self.validated["source_paths"] = {"environment_lock": str(environment_lock)}
        self.validated["source_hashes"] = {
            "environment_lock": {"sha256": digest("environment-lock")}
        }
        for source_name in set(
            generation._VERIFIED_RUNTIME_MODULE_SOURCES.values()
        ) | {"engineering_auditor"}:
            source_path = self.repo / f"source-{source_name}.py"
            self.validated["source_paths"][source_name] = str(source_path)
            self.validated["source_hashes"][source_name] = {
                "sha256": digest(f"source-{source_name}")
            }
        self.verified_source_execution = {
            module_name: {
                "source_name": source_name,
                "path": self.validated["source_paths"][source_name],
                "origin": (
                    f"<git-blob:{self.validated['reviewed_commit']}:"
                    f"{self.validated['source_paths'][source_name]}>"
                ),
                "sha256": self.validated["source_hashes"][source_name]["sha256"],
                "loader_id": generation.LAUNCH_LOADER_ID,
                "execution_count": 1,
                "cached": None,
            }
            for module_name, source_name in generation._VERIFIED_RUNTIME_MODULE_SOURCES.items()
        }
        self.runtime_environment = {
            "schema": generation.ENVIRONMENT_LOCK_SCHEMA,
            "lock_id": "fixture-lock",
            "path": str(environment_lock),
            "sha256": digest("environment-lock"),
            "observed": {
                "platform": "linux-x86_64",
                "runtime": {"python": "3.11.10"},
                "packages": {"torch": "2.5.1"},
                "hardware": {
                    "gpu": "NVIDIA GeForce RTX 3090",
                    "driver": "580.126.09",
                    "compute_capability": "8.6",
                },
                "cuda_device": {
                    "requested": "cuda:1",
                    "logical_index": 1,
                    "cuda_visible_devices": None,
                    "cuda_device_order": None,
                    "torch": {
                        "index": 1,
                        "name": "NVIDIA GeForce RTX 3090",
                        "uuid": "GPU-11111111-2222-3333-4444-555555555555",
                    },
                    "nvidia_smi": {
                        "index": 1,
                        "name": "NVIDIA GeForce RTX 3090",
                        "uuid": "GPU-11111111-2222-3333-4444-555555555555",
                        "pci_bus_id": "00000000:01:00.0",
                    },
                    "binding": {
                        "uuid_match": True,
                        "name_match": True,
                        "unmasked_physical_index_match": True,
                    },
                },
                "determinism": {"cudnn_benchmark": False},
            },
        }
        self.sidecar_calls = []
        self.block_calls = []
        self.model_stage_evidence = self._model_stage_evidence()
        self._write_complete_run()

    def close(self):
        self.temp.cleanup()

    def sidecar_summary(self, record, task):
        self.sidecar_calls.append(task["task_id"])
        task_id = task["task_id"]
        prompt_row_id = task["prompt"]["prompt_row_id"]
        return {
            "task_id": task_id,
            "action_id": task["action"]["action_id"],
            "prompt_row_id": prompt_row_id,
            "physical_no_op": task["action"]["physical_no_op"],
            "trajectory_id": f"trajectory:{task_id}",
            "initial_latent_sha256": digest(f"initial:{prompt_row_id}"),
            "final_latent_sha256": digest(f"final:{task_id}"),
            "png_sha256": digest(self.png),
            "trajectory_chain_sha256": digest(f"chain:{task_id}"),
            "sidecar_sha256": contract.canonical_sha256(record),
        }

    def block_evidence(self, records, tasks):
        self.block_calls.append([record["task_id"] for record in records])
        return {
            "schema": "fixture_block_evidence_v1",
            "prompt_row_id": tasks[0]["prompt"]["prompt_row_id"],
            "record_count": contract.TASKS_PER_PROMPT,
            "block_evidence_sha256": digest(tasks[0]["task_id"]),
        }

    def authorization_validator(self, *_args, **_kwargs):
        return copy.deepcopy(self.validated)

    def _write_json(self, path, value):
        path.write_bytes(contract.canonical_json_bytes(value) + b"\n")

    def _model_stage_evidence(self):
        manifest = generation.model_snapshot.expected_model_manifest()
        source_snapshot = {
            "path": str(self.repo / "model-source"),
            "identity": {"st_dev": 1, "st_ino": 2},
            "model_repository_path": str(self.repo / "model-repository"),
            "model_repository_identity": {"st_dev": 1, "st_ino": 3},
            "files": [],
        }
        root_identity = {"st_dev": 1, "st_ino": 4}
        stage = {
            "schema": generation.model_snapshot.MODEL_STAGE_SCHEMA,
            "status": "staged_verified_read_only",
            "path": str(self.repo / "private-model-stage"),
            "parent": str(self.repo),
            "manifest": manifest,
            "manifest_sha256": generation._expected_model()[
                "snapshot_manifest_sha256"
            ],
            "loaded_file_count": generation._expected_model()[
                "snapshot_loaded_file_count"
            ],
            "tree_sha256": digest("fixture-model-tree"),
            "root_identity": root_identity,
            "source_snapshot": source_snapshot,
            "source_snapshot_sha256": contract.canonical_sha256(source_snapshot),
        }
        verification = {
            "schema": generation.model_snapshot.MODEL_STAGE_VERIFICATION_SCHEMA,
            "status": "verified_unchanged",
            "path": stage["path"],
            "manifest_sha256": stage["manifest_sha256"],
            "loaded_file_count": stage["loaded_file_count"],
            "tree_sha256": stage["tree_sha256"],
            "root_identity": root_identity,
        }
        cleanup = {
            "schema": generation.model_snapshot.MODEL_STAGE_CLEANUP_SCHEMA,
            "status": "removed",
            "path": stage["path"],
            "manifest_sha256": stage["manifest_sha256"],
            "loaded_file_count": stage["loaded_file_count"],
            "root_identity": root_identity,
        }
        return {
            "schema": generation.MODEL_STAGE_EVIDENCE_SCHEMA,
            "status": "removed",
            "trust_boundary": {
                "same_uid_noninterference_required_between_verification_and_open": False,
                "loader_root_pinned_by_procfs_fd": True,
                "pre_post_content_and_object_identity_binding": True,
                "network_access_used": False,
                "load_source": "pinned_procfs_fd_private_regular_file_stage",
            },
            "stage": stage,
            "verifications": [copy.deepcopy(verification) for _ in range(4)],
            "cleanup": cleanup,
            "cleanup_failure": None,
        }

    def _write_complete_run(self):
        summaries = []
        block_evidence = []
        for block_index in range(contract.PROMPT_COUNT):
            start = block_index * contract.TASKS_PER_PROMPT
            block_tasks = self.tasks[start : start + contract.TASKS_PER_PROMPT]
            block_records = []
            for task in block_tasks:
                task_id = task["task_id"]
                record = {
                    "schema": "fixture_sidecar_v1",
                    "task_id": task_id,
                    "trajectory": {"png_sha256": digest(self.png)},
                }
                (self.run / "records" / f"{task_id}.png").write_bytes(self.png)
                self._write_json(self.run / "records" / f"{task_id}.json", record)
                summary = self.sidecar_summary(record, task)
                summaries.append(
                    {
                        **summary,
                        "png_path": f"records/{task_id}.png",
                        "sidecar_path": f"records/{task_id}.json",
                    }
                )
                block_records.append(record)
            block_evidence.append(self.block_evidence(block_records, block_tasks))
        self.sidecar_calls.clear()
        self.block_calls.clear()
        runtime_evidence = {
            "schema": generation.RUNTIME_EVIDENCE_SCHEMA,
            "capture_scope": "runtime_load_environment_pipeline_and_generation",
            "python_warnings": [],
            "python_warnings_sha256": contract.canonical_sha256([]),
            "logging_records": [],
            "logging_records_sha256": contract.canonical_sha256([]),
            "stderr": "",
            "stderr_byte_count": 0,
            "stderr_sha256": digest(b""),
            "warnings": {
                "count": 0,
                "python_warning_count": 0,
                "logging_warning_or_higher_count": 0,
                "stderr_warning_line_count": 0,
            },
        }
        runtime_evidence_sha256 = contract.canonical_sha256(runtime_evidence)
        attempt = generation._attempt_record(self.validated)
        run_config = audit._expected_run_config(
            self.validated,
            self.design,
            runtime_evidence_sha256,
            self.runtime_environment,
            self.verified_source_execution,
            self.model_stage_evidence,
            digest(
                contract.canonical_json_bytes(self.model_stage_evidence) + b"\n"
            ),
        )
        manifest = audit._expected_manifest(
            self.design, summaries, block_evidence, runtime_evidence_sha256
        )
        success = audit._expected_success(
            self.validated,
            self.design,
            manifest,
            run_config,
            runtime_evidence,
            digest(contract.canonical_json_bytes(runtime_evidence) + b"\n"),
        )
        self._write_json(self.run / "attempt.json", attempt)
        self._write_json(self.run / "runtime_evidence.json", runtime_evidence)
        self._write_json(
            self.run / "model_stage_evidence.json", self.model_stage_evidence
        )
        self._write_json(self.run / "config.json", run_config)
        self._write_json(self.run / "manifest.json", manifest)
        self._write_json(self.run / "success.json", success)

    def run_audit(self, **overrides):
        kwargs = {
            "sidecar_validator": self.sidecar_summary,
            "block_validator": self.block_evidence,
        }
        kwargs.update(overrides)
        return audit._audit_validated_run(
            self.run, copy.deepcopy(self.validated), **kwargs
        )

    def replace_with_failure(self, *, staged=True):
        for name in (
            "success.json",
            "manifest.json",
            "config.json",
            "runtime_evidence.json",
        ):
            (self.run / name).unlink()
        shutil.rmtree(self.run / "records")
        (self.run / "records").mkdir()
        if not staged:
            (self.run / "model_stage_evidence.json").unlink()
        failure = {
            "schema": generation.FAILURE_SCHEMA,
            "status": "failed_terminal_no_resume_or_retry",
            "experiment_id": contract.EXPERIMENT_ID,
            "authorization_sha256": self.validated["authorization_sha256"],
            "authorization_binding": copy.deepcopy(
                self.validated["authorization_binding"]
            ),
            "launcher_evidence": copy.deepcopy(self.validated["launcher_evidence"]),
            "design_sha256": self.design["design_sha256"],
            "completed_records": 0,
            "expected_records": contract.TOTAL_TASK_COUNT,
            "exception_type": "RuntimeError",
            "exception_message": "fixture runtime failure",
            "traceback_sha256": digest("fixture traceback"),
            "runtime_evidence": None,
            "model_stage": (
                copy.deepcopy(self.model_stage_evidence["stage"])
                if staged
                else None
            ),
            "model_stage_verifications": (
                copy.deepcopy(self.model_stage_evidence["verifications"])
                if staged
                else []
            ),
        }
        self._write_json(self.run / "failure.json", failure)
        return failure

    def run_production_audit(self):
        with (
            mock.patch.object(
                audit,
                "_validate_auditor_launcher_execution",
            ),
            mock.patch.object(
                audit.contract,
                "validate_sidecar",
                side_effect=self.sidecar_summary,
            ),
            mock.patch.object(
                audit.contract,
                "validate_prompt_block",
                side_effect=self.block_evidence,
            ),
        ):
            return audit._audit_launcher_validated_run(
                copy.deepcopy(self.validated)
            )


class AdaptiveOracleEngineeringAuditTest(unittest.TestCase):
    def setUp(self):
        self.fixture = EngineeringRunFixture()

    def tearDown(self):
        self.fixture.close()

    def test_production_signature_exposes_no_injection_or_publish_hooks(self):
        for entrypoint in (
            audit.audit_engineering_run,
            audit._audit_launcher_validated_run,
            audit._run_from_verified_launcher,
        ):
            parameters = inspect.signature(entrypoint).parameters
            for forbidden in (
                "authorization_validator",
                "sidecar_validator",
                "block_validator",
                "git_blob_reader",
                "git_ancestor_checker",
                "git_lineage_checker",
                "git_head_resolver",
                "publish_receipts",
            ):
                self.assertNotIn(forbidden, parameters)

    def test_receipt_free_core_cannot_publish_official_audit_receipts(self):
        result = self.fixture.run_audit()
        self.assertEqual(
            result["status"], "passed_generation_integrity_only_unscored"
        )
        for name in (
            audit.AUDIT_ATTEMPT_NAME,
            audit.AUDIT_SUCCESS_NAME,
            audit.AUDIT_FAILURE_NAME,
        ):
            self.assertFalse((self.fixture.run / name).exists())

    def test_direct_main_requires_verified_launcher_context(self):
        with self.assertRaisesRegex(RuntimeError, "verified execution-root launcher"):
            audit.main([])
        with self.assertRaisesRegex(RuntimeError, "verified execution-root launcher"):
            audit.main([], execution_context={"forged": True})
        with self.assertRaisesRegex(RuntimeError, "verified execution-root launcher"):
            audit.audit_engineering_run(
                self.fixture.authorization,
                device="cuda:1",
                run_dir=self.fixture.run,
                repo_root=self.fixture.repo,
            )
        self.assertFalse((self.fixture.run / audit.AUDIT_ATTEMPT_NAME).exists())

    def test_launcher_entrypoint_consumes_exact_trusted_input_bundle(self):
        names = (
            "registration",
            "cpu_audit",
            "prompt_csv",
            "exclusion_inventory",
            "prompt_manifest",
            "environment_lock",
        )
        input_bytes = {name: f"fixture:{name}".encode("ascii") for name in names}

        class Context:
            def __init__(self, evidence, inputs):
                self.evidence = evidence
                self.inputs = inputs
                self.claim_count = 0
                self.read_names = []

            def claim(self):
                self.claim_count += 1
                return copy.deepcopy(self.evidence)

            def read_input(self, name):
                self.read_names.append(name)
                return self.inputs[name]

        context = Context(self.fixture.validated["launcher_evidence"], input_bytes)
        with (
            mock.patch.object(
                generation,
                "_validate_launcher_bundle",
                return_value=copy.deepcopy(self.fixture.validated),
            ) as validate_bundle,
            mock.patch.object(
                audit,
                "_audit_launcher_validated_run",
                return_value={"status": "fixture-audited"},
            ) as run_audit,
            mock.patch("builtins.print"),
        ):
            result = audit._run_from_verified_launcher(
                [
                    "--device",
                    "cuda:1",
                    "--run-dir",
                    str(self.fixture.run),
                ],
                launcher_context=context,
            )
        self.assertEqual(result, 0)
        self.assertEqual(context.claim_count, 1)
        self.assertEqual(tuple(context.read_names), names)
        validate_bundle.assert_called_once_with(
            self.fixture.validated["launcher_evidence"],
            input_bytes,
            requested_device="cuda:1",
            requested_output_dir=str(self.fixture.run),
        )
        run_audit.assert_called_once()

    def test_launcher_execution_record_binds_reviewed_auditor_blob(self):
        validated = copy.deepcopy(self.fixture.validated)
        module_name = "audit_adaptive_oracle_engineering"
        source_name = "engineering_auditor"
        source_path = generation.SOURCE_PATHS[source_name]
        origin = (
            f"<git-blob:{validated['reviewed_commit']}:{source_path}>"
        )
        validated["launcher_evidence"]["module_execution"][module_name] = {
            "source_name": source_name,
            "path": source_path,
            "origin": origin,
            "sha256": validated["source_hashes"][source_name]["sha256"],
            "loader_id": generation.LAUNCH_LOADER_ID,
            "execution_count": 1,
            "cached": None,
        }
        with (
            mock.patch.object(audit, "__name__", module_name),
            mock.patch.object(audit, "__file__", origin),
            mock.patch.object(audit, "__cached__", None, create=True),
        ):
            audit._validate_auditor_launcher_execution(validated)
            validated["launcher_evidence"]["module_execution"][module_name][
                "loader_id"
            ] = "forged-loader"
            with self.assertRaisesRegex(RuntimeError, "execution evidence differs"):
                audit._validate_auditor_launcher_execution(validated)

    def test_complete_run_validates_165_records_and_11_blocks(self):
        result = self.fixture.run_audit()

        self.assertEqual(result["record_count"], contract.TOTAL_TASK_COUNT)
        self.assertEqual(result["prompt_block_count"], contract.PROMPT_COUNT)
        self.assertEqual(
            len(self.fixture.sidecar_calls), contract.TOTAL_TASK_COUNT
        )
        self.assertEqual(len(self.fixture.block_calls), contract.PROMPT_COUNT)
        self.assertEqual(result["warnings"]["count"], 0)
        self.assertFalse(result["scoring_authorized"])
        self.assertFalse(result["quality_inspection_authorized"])
        self.assertFalse((self.fixture.run / audit.AUDIT_ATTEMPT_NAME).exists())
        self.assertFalse((self.fixture.run / audit.AUDIT_SUCCESS_NAME).exists())
        self.assertFalse((self.fixture.run / audit.AUDIT_FAILURE_NAME).exists())

    def test_audit_file_digests_reuse_the_validated_reads(self):
        with mock.patch.object(
            audit,
            "sha256_file",
            side_effect=AssertionError("audited paths must not be reopened"),
        ):
            result = self.fixture.run_audit()
        self.assertEqual(
            result["generation_attempt_file_sha256"],
            digest((self.fixture.run / "attempt.json").read_bytes()),
        )

    def test_validated_sidecar_inode_replacement_is_rejected_before_return(self):
        first_task = self.fixture.tasks[0]["task_id"]
        target = self.fixture.run / "records" / f"{first_task}.json"
        replacement = self.fixture.repo / "replacement-sidecar.json"
        replacement.write_bytes(target.read_bytes())
        block_count = 0

        def replace_after_all_sidecars(records, tasks):
            nonlocal block_count
            result = self.fixture.block_evidence(records, tasks)
            block_count += 1
            if block_count == contract.PROMPT_COUNT:
                os.replace(replacement, target)
            return result

        with self.assertRaisesRegex(
            ValueError, "sidecar .* changed before audit publication"
        ):
            self.fixture.run_audit(block_validator=replace_after_all_sidecars)

    def test_validated_top_level_inode_replacement_is_rejected(self):
        target = self.fixture.run / "config.json"
        replacement = self.fixture.repo / "replacement-config.json"
        replacement.write_bytes(target.read_bytes())
        replaced = False

        def replace_after_config_read(record, task):
            nonlocal replaced
            if not replaced:
                replaced = True
                os.replace(replacement, target)
            return self.fixture.sidecar_summary(record, task)

        with self.assertRaisesRegex(
            ValueError, "run config changed before audit publication"
        ):
            self.fixture.run_audit(sidecar_validator=replace_after_config_read)

    def test_run_directory_replacement_is_rejected(self):
        displaced = self.fixture.repo / "displaced-run"
        replaced = False

        def replace_run(record, task):
            nonlocal replaced
            if not replaced:
                replaced = True
                self.fixture.run.rename(displaced)
                self.fixture.run.mkdir()
                (self.fixture.run / "records").mkdir()
            return self.fixture.sidecar_summary(record, task)

        with self.assertRaisesRegex(RuntimeError, "run directory path identity"):
            self.fixture.run_audit(sidecar_validator=replace_run)

    def test_records_directory_replacement_is_rejected(self):
        records = self.fixture.run / "records"
        displaced = self.fixture.repo / "displaced-records"
        replaced = False

        def replace_records(record, task):
            nonlocal replaced
            if not replaced:
                replaced = True
                records.rename(displaced)
                records.mkdir()
            return self.fixture.sidecar_summary(record, task)

        with self.assertRaisesRegex(
            RuntimeError, "records directory path identity"
        ):
            self.fixture.run_audit(sidecar_validator=replace_records)

    def test_run_parent_replacement_is_rejected(self):
        output_parent = self.fixture.repo / "outputs"
        displaced = self.fixture.repo / "displaced-outputs"
        replaced = False

        def replace_parent(record, task):
            nonlocal replaced
            if not replaced:
                replaced = True
                output_parent.rename(displaced)
                replacement_records = (
                    output_parent / "adaptive_oracle" / "engineering_v1" / "records"
                )
                replacement_records.mkdir(parents=True)
            return self.fixture.sidecar_summary(record, task)

        with self.assertRaisesRegex(RuntimeError, "component 0 path identity"):
            self.fixture.run_audit(sidecar_validator=replace_parent)

    def test_default_contract_validator_routes_are_used(self):
        with (
            mock.patch.object(
                audit.contract,
                "validate_sidecar",
                side_effect=self.fixture.sidecar_summary,
            ) as sidecar_validator,
            mock.patch.object(
                audit.contract,
                "validate_prompt_block",
                side_effect=self.fixture.block_evidence,
            ) as block_validator,
        ):
            self.fixture.run_audit(
                sidecar_validator=None,
                block_validator=None,
            )
        self.assertEqual(sidecar_validator.call_count, contract.TOTAL_TASK_COUNT)
        self.assertEqual(block_validator.call_count, contract.PROMPT_COUNT)

    def test_contract_binds_latent_chain_and_scheduler_model_provenance(self):
        contract_tests = load_module(
            "test_adaptive_oracle_contract_for_engineering_audit",
            ROOT / "tests" / "test_adaptive_oracle_contract.py",
        )
        AdaptiveOracleContractTest = contract_tests.AdaptiveOracleContractTest

        AdaptiveOracleContractTest.setUpClass()
        helper = AdaptiveOracleContractTest(
            methodName="test_p0_p_r_u_and_x_sidecars_are_accepted"
        )
        task = helper.design["tasks"][0]
        record = helper.make_sidecar(task)
        summary = audit.contract.validate_sidecar(record, task)
        audit._validate_summary(
            summary,
            task=task,
            record=record,
            png_sha256=record["trajectory"]["png_sha256"],
        )
        self.assertIn("trajectory_chain_sha256", summary)

        broken_chain = copy.deepcopy(record)
        broken_chain["step_ledger"][1]["latent_before_sha256"] = digest(
            "unrelated-latent"
        )
        with self.assertRaisesRegex(ValueError, "trajectory chain"):
            audit.contract.validate_sidecar(broken_chain, task)

        stale_provenance = copy.deepcopy(record)
        stale_provenance["trajectory"]["model_revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "model_revision"):
            audit.contract.validate_sidecar(stale_provenance, task)

    def test_audit_is_one_shot_and_does_not_replace_receipts(self):
        self.fixture.run_production_audit()
        success_before = (self.fixture.run / audit.AUDIT_SUCCESS_NAME).read_bytes()
        with self.assertRaisesRegex(FileExistsError, "one-shot"):
            self.fixture.run_production_audit()
        self.assertEqual(
            (self.fixture.run / audit.AUDIT_SUCCESS_NAME).read_bytes(), success_before
        )

    def test_malformed_generation_receipt_is_consumed_after_audit_attempt(self):
        (self.fixture.run / "success.json").write_bytes(b"not-json\n")

        with self.assertRaisesRegex(ValueError, "strict UTF-8 JSON"):
            self.fixture.run_production_audit()

        self.assertTrue((self.fixture.run / audit.AUDIT_ATTEMPT_NAME).is_file())
        self.assertTrue((self.fixture.run / audit.AUDIT_FAILURE_NAME).is_file())
        self.assertFalse((self.fixture.run / audit.AUDIT_SUCCESS_NAME).exists())

    def test_fd_headroom_failure_precedes_the_one_shot_audit_attempt(self):
        with (
            mock.patch.object(
                generation,
                "_require_fd_headroom",
                side_effect=RuntimeError("insufficient descriptor reserve"),
            ),
            self.assertRaisesRegex(RuntimeError, "descriptor reserve"),
        ):
            self.fixture.run_production_audit()

        self.assertFalse((self.fixture.run / audit.AUDIT_ATTEMPT_NAME).exists())
        self.assertFalse((self.fixture.run / audit.AUDIT_FAILURE_NAME).exists())
        self.assertFalse((self.fixture.run / audit.AUDIT_SUCCESS_NAME).exists())

    def test_production_rechecks_pins_after_core_before_success_publication(self):
        task_id = self.fixture.tasks[0]["task_id"]
        target = self.fixture.run / "records" / f"{task_id}.json"
        replacement = self.fixture.repo / "pre-publication-sidecar.json"
        replacement.write_bytes(target.read_bytes())
        real_publish = audit._publish_audit_json
        replaced = False

        def replace_before_success(tree, name, value):
            nonlocal replaced
            if name == audit.AUDIT_SUCCESS_NAME and not replaced:
                replaced = True
                os.replace(replacement, target)
            return real_publish(tree, name, value)

        with (
            mock.patch.object(
                audit,
                "_publish_audit_json",
                side_effect=replace_before_success,
            ),
            self.assertRaisesRegex(ValueError, "changed before audit publication"),
        ):
            self.fixture.run_production_audit()

        self.assertTrue(replaced)
        self.assertTrue((self.fixture.run / audit.AUDIT_ATTEMPT_NAME).is_file())
        self.assertFalse((self.fixture.run / audit.AUDIT_SUCCESS_NAME).exists())
        self.assertTrue((self.fixture.run / audit.AUDIT_FAILURE_NAME).is_file())

    def test_post_success_integrity_failure_publishes_audit_invalidation(self):
        task_id = self.fixture.tasks[0]["task_id"]
        target = self.fixture.run / "records" / f"{task_id}.json"
        replacement = self.fixture.repo / "post-success-sidecar.json"
        replacement.write_bytes(target.read_bytes())
        real_atomic = audit.atomic_create_json
        replaced = False

        def publish_then_replace(path, value, *, directory_descriptor=None):
            nonlocal replaced
            identity = real_atomic(
                path,
                value,
                directory_descriptor=directory_descriptor,
            )
            if Path(path).name == audit.AUDIT_SUCCESS_NAME and not replaced:
                replaced = True
                os.replace(replacement, target)
            return identity

        with (
            mock.patch.object(
                audit,
                "atomic_create_json",
                side_effect=publish_then_replace,
            ),
            self.assertRaisesRegex(ValueError, "changed before audit publication"),
        ):
            self.fixture.run_production_audit()

        self.assertTrue(replaced)
        self.assertTrue((self.fixture.run / audit.AUDIT_SUCCESS_NAME).is_file())
        self.assertTrue((self.fixture.run / audit.AUDIT_FAILURE_NAME).is_file())

    def test_png_inspector_validates_exact_rgb_scanlines_without_pillow(self):
        path = self.fixture.repo / "complete.png"
        path.write_bytes(png_bytes())
        with mock.patch("builtins.__import__", wraps=__import__) as importer:
            observed = audit.inspect_png_container(path)
        self.assertEqual(observed["width"], contract.WIDTH)
        self.assertEqual(observed["height"], contract.HEIGHT)
        self.assertEqual(observed["idat_chunk_count"], 1)
        self.assertEqual(
            observed["decompressed_scanline_byte_count"],
            contract.HEIGHT * (1 + contract.WIDTH * 3),
        )
        imported_names = [call.args[0] for call in importer.call_args_list]
        self.assertNotIn("PIL", imported_names)

        invalid_encodings = (
            ({"width": 512}, "expected 1024x1024"),
            ({"bit_depth": 16}, "8-bit RGB"),
            ({"color_type": 6}, "8-bit RGB"),
            ({"interlace": 1}, "non-interlaced"),
        )
        for kwargs, message in invalid_encodings:
            with self.subTest(kwargs=kwargs):
                path.write_bytes(png_bytes(**kwargs))
                with self.assertRaisesRegex(ValueError, message):
                    audit.inspect_png_container(path)

        corrupted = bytearray(png_bytes())
        iend_offset = corrupted.rfind(b"IEND") - 4
        corrupted[iend_offset - 1] ^= 1
        path.write_bytes(bytes(corrupted))
        with self.assertRaisesRegex(ValueError, "CRC"):
            audit.inspect_png_container(path)

    def test_png_inspector_accepts_one_zlib_stream_split_across_idat_chunks(self):
        path = self.fixture.repo / "multiple-idat.png"
        path.write_bytes(png_bytes(idat_chunk_count=3))
        observed = audit.inspect_png_container(path)
        self.assertEqual(observed["idat_chunk_count"], 3)

    def test_png_inspector_rejects_invalid_truncated_or_trailed_zlib_stream(self):
        path = self.fixture.repo / "invalid-zlib.png"
        scanlines = rgb_scanlines()
        compressed = zlib.compress(scanlines)

        cases = (
            (b"not-a-zlib-stream", "zlib stream is invalid"),
            (compressed[:-2], "zlib stream is truncated"),
            (compressed + b"trailing-zlib-data", "zlib stream has trailing data"),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                path.write_bytes(png_bytes(compressed_idat=payload))
                with self.assertRaisesRegex(ValueError, message):
                    audit.inspect_png_container(path)

    def test_png_inspector_rejects_scanline_length_and_filter_drift(self):
        path = self.fixture.repo / "invalid-scanlines.png"
        scanlines = rgb_scanlines()

        path.write_bytes(png_bytes(scanlines=scanlines[:-1]))
        with self.assertRaisesRegex(ValueError, "scanline byte length"):
            audit.inspect_png_container(path)

        invalid_filter = bytearray(scanlines)
        invalid_filter[0] = 5
        path.write_bytes(png_bytes(scanlines=bytes(invalid_filter)))
        with self.assertRaisesRegex(ValueError, "invalid filter byte"):
            audit.inspect_png_container(path)

    def test_png_inspector_rejects_truncation_missing_iend_and_trailing_bytes(self):
        path = self.fixture.repo / "broken.png"
        complete = png_bytes()

        path.write_bytes(complete[:-1])
        with self.assertRaisesRegex(ValueError, "truncated"):
            audit.inspect_png_container(path)

        path.write_bytes(complete[:-12])
        with self.assertRaisesRegex(ValueError, "omitted IEND"):
            audit.inspect_png_container(path)

        path.write_bytes(complete + b"opaque-trailing-bytes")
        with self.assertRaisesRegex(ValueError, "trailing bytes"):
            audit.inspect_png_container(path)

    def test_noncanonical_or_duplicate_sidecar_is_rejected(self):
        task_id = self.fixture.tasks[0]["task_id"]
        path = self.fixture.run / "records" / f"{task_id}.json"
        path.write_text(
            '{"schema":"fixture_sidecar_v1", "task_id":"x", '
            '"task_id":"x","trajectory":{}}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            self.fixture.run_audit()
        self.assertFalse((self.fixture.run / audit.AUDIT_ATTEMPT_NAME).exists())

    def test_png_hash_must_match_sidecar_summary(self):
        task_id = self.fixture.tasks[0]["task_id"]
        path = self.fixture.run / "records" / f"{task_id}.png"
        replacement = self.fixture.png[:-12] + png_chunk(
            b"tEXt", b"fixture\x00different"
        ) + self.fixture.png[-12:]
        path.write_bytes(replacement)
        with self.assertRaisesRegex(ValueError, "PNG bytes differ"):
            self.fixture.run_audit()

    def test_score_file_and_quality_field_are_rejected(self):
        score_path = self.fixture.run / "scores.jsonl"
        score_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "forbidden score/quality artifact"):
            self.fixture.run_audit()
        score_path.unlink()

        task_id = self.fixture.tasks[0]["task_id"]
        sidecar_path = self.fixture.run / "records" / f"{task_id}.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["quality_score"] = 0.5
        self.fixture._write_json(sidecar_path, sidecar)
        with self.assertRaisesRegex(ValueError, "score/quality field"):
            self.fixture.run_audit()

    def test_manifest_and_success_bindings_are_exact(self):
        manifest_path = self.fixture.run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["record_count"] -= 1
        self.fixture._write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "run manifest differs"):
            self.fixture.run_audit()

    def test_success_artifacts_bind_the_same_launcher_evidence(self):
        cases = (
            ("attempt.json", "generation attempt differs"),
            ("config.json", "run config differs"),
            ("success.json", "success receipt differs"),
        )
        for name, message in cases:
            with self.subTest(name=name):
                path = self.fixture.run / name
                original = path.read_bytes()
                value = json.loads(original.decode("utf-8"))
                value["launcher_evidence"]["reviewed_commit"] = "b" * 40
                self.fixture._write_json(path, value)
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        self.fixture.run_audit()
                finally:
                    path.write_bytes(original)

    def test_success_binds_complete_model_stage_evidence_and_hashes(self):
        evidence_path = self.fixture.run / "model_stage_evidence.json"
        original = evidence_path.read_bytes()
        evidence = json.loads(original.decode("utf-8"))
        evidence["verifications"][0]["tree_sha256"] = "0" * 64
        self.fixture._write_json(evidence_path, evidence)
        with self.assertRaisesRegex(ValueError, "verification 0 differs"):
            self.fixture.run_audit()
        evidence_path.write_bytes(original)

        config_path = self.fixture.run / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model_stage_evidence_sha256"] = "0" * 64
        self.fixture._write_json(config_path, config)
        with self.assertRaisesRegex(ValueError, "run config differs"):
            self.fixture.run_audit()

    def test_failure_binds_launcher_and_model_stage_transcript(self):
        failure = self.fixture.replace_with_failure()
        failure_path = self.fixture.run / "failure.json"
        failure["launcher_evidence"]["reviewed_commit"] = "b" * 40
        self.fixture._write_json(failure_path, failure)
        with self.assertRaisesRegex(ValueError, "launcher_evidence differs"):
            self.fixture.run_audit()

        failure["launcher_evidence"] = copy.deepcopy(
            self.fixture.validated["launcher_evidence"]
        )
        self.fixture._write_json(failure_path, failure)
        (self.fixture.run / "model_stage_evidence.json").unlink()
        with self.assertRaisesRegex(ValueError, "omitted terminal model-stage evidence"):
            self.fixture.run_audit()

    def test_failure_rejects_model_stage_transcript_drift(self):
        failure = self.fixture.replace_with_failure()
        failure["model_stage_verifications"][0]["tree_sha256"] = "0" * 64
        self.fixture._write_json(self.fixture.run / "failure.json", failure)
        with self.assertRaisesRegex(
            ValueError, "embedded/file model-stage verifications"
        ):
            self.fixture.run_audit()

    def test_failure_accepts_every_full_stage_verification_prefix(self):
        failure = self.fixture.replace_with_failure()
        evidence_path = self.fixture.run / "model_stage_evidence.json"
        full_verifications = copy.deepcopy(
            self.fixture.model_stage_evidence["verifications"]
        )
        for count in range(5):
            with self.subTest(count=count):
                prefix = copy.deepcopy(full_verifications[:count])
                failure["model_stage_verifications"] = prefix
                self.fixture._write_json(self.fixture.run / "failure.json", failure)
                evidence = copy.deepcopy(self.fixture.model_stage_evidence)
                evidence["verifications"] = prefix
                self.fixture._write_json(evidence_path, evidence)
                with self.assertRaises(audit.GenerationFailedTerminally):
                    self.fixture.run_audit()

    def test_failure_accepts_partial_stage_cleanup_success_and_failure(self):
        failure = self.fixture.replace_with_failure()
        full_stage = self.fixture.model_stage_evidence["stage"]
        partial_stage = {
            "schema": generation.model_snapshot.MODEL_STAGE_SCHEMA,
            "status": "staging_failed_cleanup_pending",
            "path": full_stage["path"],
            "parent": full_stage["parent"],
            "root_identity": copy.deepcopy(full_stage["root_identity"]),
        }
        failure["model_stage"] = copy.deepcopy(partial_stage)
        failure["model_stage_verifications"] = []
        self.fixture._write_json(self.fixture.run / "failure.json", failure)

        for cleanup_failed in (False, True):
            with self.subTest(cleanup_failed=cleanup_failed):
                evidence = {
                    "schema": generation.MODEL_STAGE_EVIDENCE_SCHEMA,
                    "status": (
                        "cleanup_failed_terminal" if cleanup_failed else "removed"
                    ),
                    "trust_boundary": {
                        "same_uid_noninterference_required_between_verification_and_open": False,
                        "loader_root_pinned_by_procfs_fd": True,
                        "pre_post_content_and_object_identity_binding": True,
                        "network_access_used": False,
                        "load_source": "pinned_procfs_fd_private_regular_file_stage",
                    },
                    "stage": copy.deepcopy(partial_stage),
                    "verifications": [],
                    "cleanup": (
                        None
                        if cleanup_failed
                        else {
                            "schema": generation.model_snapshot.MODEL_STAGE_CLEANUP_SCHEMA,
                            "status": "removed",
                            "path": partial_stage["path"],
                            "manifest_sha256": None,
                            "loaded_file_count": None,
                            "root_identity": copy.deepcopy(
                                partial_stage["root_identity"]
                            ),
                        }
                    ),
                    "cleanup_failure": (
                        {
                            "exception_type": "OSError",
                            "exception_message": "cleanup denied",
                        }
                        if cleanup_failed
                        else None
                    ),
                }
                self.fixture._write_json(
                    self.fixture.run / "model_stage_evidence.json", evidence
                )
                with self.assertRaises(audit.GenerationFailedTerminally):
                    self.fixture.run_audit()

    def test_partial_stage_rejects_impossible_transcript_and_cleanup_binding(self):
        full = self.fixture.model_stage_evidence
        partial_stage = {
            "schema": generation.model_snapshot.MODEL_STAGE_SCHEMA,
            "status": "staging_failed_cleanup_pending",
            "path": full["stage"]["path"],
            "parent": full["stage"]["parent"],
            "root_identity": copy.deepcopy(full["stage"]["root_identity"]),
        }
        evidence = {
            "schema": generation.MODEL_STAGE_EVIDENCE_SCHEMA,
            "status": "removed",
            "trust_boundary": copy.deepcopy(full["trust_boundary"]),
            "stage": partial_stage,
            "verifications": [copy.deepcopy(full["verifications"][0])],
            "cleanup": {
                "schema": generation.model_snapshot.MODEL_STAGE_CLEANUP_SCHEMA,
                "status": "removed",
                "path": partial_stage["path"],
                "manifest_sha256": None,
                "loaded_file_count": None,
                "root_identity": copy.deepcopy(partial_stage["root_identity"]),
            },
            "cleanup_failure": None,
        }
        with self.assertRaisesRegex(ValueError, "cannot have verification"):
            audit._validate_model_stage_evidence(evidence, require_complete=False)

        evidence["verifications"] = []
        evidence["cleanup"]["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "cleanup differs"):
            audit._validate_model_stage_evidence(evidence, require_complete=False)

    def test_failure_rejects_more_than_four_full_stage_verifications(self):
        evidence = copy.deepcopy(self.fixture.model_stage_evidence)
        evidence["verifications"].append(
            copy.deepcopy(evidence["verifications"][-1])
        )
        with self.assertRaisesRegex(ValueError, "invalid verification transcript"):
            audit._validate_model_stage_evidence(evidence, require_complete=False)

    def test_failure_before_model_staging_has_no_stage_evidence(self):
        self.fixture.replace_with_failure(staged=False)
        with self.assertRaises(audit.GenerationFailedTerminally):
            self.fixture.run_audit()

    def test_failure_pins_the_unpaired_atomic_png_tail(self):
        failure = self.fixture.replace_with_failure()
        task_id = self.fixture.tasks[0]["task_id"]
        target = self.fixture.run / "records" / f"{task_id}.png"
        target.write_bytes(self.fixture.png)
        replacement = self.fixture.repo / "replacement-tail.png"
        replacement.write_bytes(self.fixture.png)
        self.fixture._write_json(self.fixture.run / "failure.json", failure)
        real_inspect = audit.inspect_png_container

        def replace_after_inspection(path, **kwargs):
            result = real_inspect(path, **kwargs)
            if Path(path).name == target.name and replacement.exists():
                os.replace(replacement, target)
            return result

        with (
            mock.patch.object(
                audit,
                "inspect_png_container",
                side_effect=replace_after_inspection,
            ),
            self.assertRaisesRegex(
                ValueError, "PNG record changed before audit publication"
            ),
        ):
            self.fixture.run_audit()

    def test_runtime_environment_binds_the_unmasked_physical_device(self):
        run_config_path = self.fixture.run / "config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config["runtime_environment"]["observed"]["cuda_device"][
            "nvidia_smi"
        ]["index"] = 0
        self.fixture._write_json(run_config_path, run_config)
        with self.assertRaisesRegex(ValueError, "UUID/PCI physical identity"):
            self.fixture.run_audit()

    def test_runtime_environment_rejects_same_name_with_different_uuid(self):
        run_config_path = self.fixture.run / "config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config["runtime_environment"]["observed"]["cuda_device"][
            "nvidia_smi"
        ]["uuid"] = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self.fixture._write_json(run_config_path, run_config)
        with self.assertRaisesRegex(ValueError, "UUID/PCI"):
            self.fixture.run_audit()

    def test_runtime_execution_bytes_are_bound_to_authorized_sources(self):
        run_config_path = self.fixture.run / "config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        module_name = "AttentionGuidance.adaptive_oracle"
        run_config["verified_source_execution"][module_name]["sha256"] = "0" * 64
        self.fixture._write_json(run_config_path, run_config)
        with self.assertRaisesRegex(ValueError, "verified source execution differs"):
            self.fixture.run_audit()

    def test_runtime_warning_evidence_is_recomputed_and_must_be_warning_free(self):
        evidence_path = self.fixture.run / "runtime_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["warnings"]["count"] = 1
        self.fixture._write_json(evidence_path, evidence)
        with self.assertRaisesRegex(ValueError, "warning counts differ"):
            self.fixture.run_audit()

    def test_raw_stderr_warning_category_is_recomputed(self):
        evidence_path = self.fixture.run / "runtime_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        stderr_text = "[W0827 12:34:56 file.cc:1] raw subprocess warning\n"
        stderr_bytes = stderr_text.encode("utf-8")
        evidence["stderr"] = stderr_text
        evidence["stderr_byte_count"] = len(stderr_bytes)
        evidence["stderr_sha256"] = hashlib.sha256(stderr_bytes).hexdigest()
        self.fixture._write_json(evidence_path, evidence)
        with self.assertRaisesRegex(ValueError, "warning counts differ"):
            self.fixture.run_audit()

    def test_success_and_failure_receipts_are_mutually_exclusive(self):
        self.fixture._write_json(
            self.fixture.run / "failure.json",
            {"schema": generation.FAILURE_SCHEMA},
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.fixture.run_audit()
        self.assertFalse((self.fixture.run / audit.AUDIT_ATTEMPT_NAME).exists())

    def test_generation_failure_gets_terminal_audit_failure_receipt(self):
        self.fixture.replace_with_failure()

        with self.assertRaises(audit.GenerationFailedTerminally):
            self.fixture.run_production_audit()
        self.assertTrue((self.fixture.run / audit.AUDIT_ATTEMPT_NAME).is_file())
        self.assertTrue((self.fixture.run / audit.AUDIT_FAILURE_NAME).is_file())
        self.assertFalse((self.fixture.run / audit.AUDIT_SUCCESS_NAME).exists())
        audit_failure = audit.load_canonical_json(
            self.fixture.run / audit.AUDIT_FAILURE_NAME,
            label="audit failure",
        )
        self.assertEqual(
            audit_failure["status"], "failed_terminal_no_resume_or_retry"
        )

    def test_symlinked_record_is_rejected(self):
        task_id = self.fixture.tasks[0]["task_id"]
        png_path = self.fixture.run / "records" / f"{task_id}.png"
        target = self.fixture.repo / "target.png"
        target.write_bytes(self.fixture.png)
        png_path.unlink()
        png_path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.fixture.run_audit()


if __name__ == "__main__":
    unittest.main()
