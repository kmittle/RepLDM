import copy
import hashlib
import importlib.util
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "eval-pipeline" / "audit_adaptive_oracle_cpu.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "adaptive_oracle_cpu_audit_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module()


class AdaptiveOracleCpuAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        for relative in audit.AUDITED_TRACKED_PATHS:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"frozen:{relative}\n".encode("ascii"))
        (self.repo / "eval-pipeline/configs").mkdir(parents=True, exist_ok=True)
        self.provenance = {
            "commit": "a" * 40,
            "clean": True,
            "implementation_sources": audit._file_records(
                self.repo, audit.IMPLEMENTATION_SOURCE_PATHS
            ),
            "test_sources": audit._file_records(
                self.repo, audit.TEST_SOURCE_PATHS
            ),
            "registered_inputs": audit._file_records(
                self.repo, audit.REGISTERED_INPUT_PATHS
            ),
        }
        inputs = {
            row["path"]: row for row in self.provenance["registered_inputs"]
        }
        self.test_record = {
            "status": "passed_warning_free",
            "allowlist": list(audit.TEST_MODULES),
            "arguments": [
                "-I",
                "-B",
                "-S",
                "-W",
                "error",
                "-c",
                audit.TEST_BOOTSTRAP,
                *audit.TEST_MODULES,
            ],
            "environment_overrides": dict(audit.TEST_ENVIRONMENT_OVERRIDES),
            "interpreter": {
                "executable": sys.executable,
                "implementation": "CPython",
                "version": "3.11.10",
            },
            "return_code": 0,
            "test_count": 73,
            "warnings_as_errors": True,
            "warning_count": 0,
            "stdout_byte_count": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_byte_count": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }
        self.prompt_record = {
            "status": "passed_byte_exact",
            "upstream_repository": "https://github.com/google-research/parti",
            "source_repository": "/tmp/parti",
            "source_revision": "b" * 40,
            "source_path": "/tmp/parti/PartiPrompts.tsv",
            "source_sha256": "c" * 64,
            "assets": [
                inputs[audit.PROMPT_CSV_PATH],
                inputs[audit.EXCLUSION_INVENTORY_PATH],
                inputs[audit.PROMPT_MANIFEST_PATH],
            ],
            "mismatch_count": 0,
        }
        self.design_record = {
            "status": "passed",
            "prompt_count": 11,
            "tasks_per_prompt": 15,
            "task_count": 165,
            "tasks_sha256": "1" * 64,
            "design_sha256": "2" * 64,
            "contract_sha256": "3" * 64,
            "primary_action_bank_sha256": "4" * 64,
            "signed_orbit_cycle_sha256": "5" * 64,
            "execution_contract_sha256": "6" * 64,
            "sidecar_shape_sha256": "7" * 64,
        }
        lock_hash = inputs[audit.ENVIRONMENT_LOCK_PATH]["sha256"]
        self.registration_record = {
            "status": "passed_exact_non_executable",
            "schema": "adaptive_oracle_engineering_registration_v1",
            "sha256": inputs[audit.REGISTRATION_PATH]["sha256"],
            "environment_lock_sha256": lock_hash,
            "exclusion_inventory_sha256": inputs[
                audit.EXCLUSION_INVENTORY_PATH
            ]["sha256"],
            "design_sha256": self.design_record["design_sha256"],
        }
        self.environment_record = {
            "status": "passed",
            "lock_path": audit.ENVIRONMENT_LOCK_PATH,
            "lock_sha256": lock_hash,
            "helper_schema": "repldm_generation_environment_lock_v2",
            "lock_id": "test-lock",
            "observed": {"runtime": {"python": "3.11.10"}},
        }
        self.output = self.repo / audit.OUTPUT_RELATIVE_PATH

    def tearDown(self):
        self.temp.cleanup()

    def create(self, **overrides):
        callbacks = {
            "provenance_collector": lambda _root: copy.deepcopy(self.provenance),
            "test_runner": lambda _root: copy.deepcopy(self.test_record),
            "prompt_replayer": lambda _root, **_kwargs: copy.deepcopy(
                self.prompt_record
            ),
            "design_validator": lambda _root: copy.deepcopy(self.design_record),
            "registration_validator": lambda _root: copy.deepcopy(
                self.registration_record
            ),
            "environment_validator": lambda _root, **_kwargs: copy.deepcopy(
                self.environment_record
            ),
        }
        callbacks.update(overrides)
        return audit._build_cpu_audit_record(self.repo, **callbacks)

    def publish(self):
        with (
            mock.patch.object(
                audit,
                "run_exact_test_suite",
                side_effect=lambda _root: copy.deepcopy(self.test_record),
            ),
            mock.patch.object(
                audit,
                "validate_reviewed_prompt_assets",
                side_effect=lambda _root, **_kwargs: copy.deepcopy(
                    self.prompt_record
                ),
            ),
            mock.patch.object(
                audit,
                "validate_engineering_design",
                side_effect=lambda _root: copy.deepcopy(self.design_record),
            ),
            mock.patch.object(
                audit,
                "validate_engineering_registration",
                side_effect=lambda _root: copy.deepcopy(self.registration_record),
            ),
            mock.patch.object(
                audit,
                "validate_pinned_environment",
                side_effect=lambda _root, **_kwargs: copy.deepcopy(
                    self.environment_record
                ),
            ),
        ):
            records = {
                row["path"]: row
                for group in (
                    self.provenance["implementation_sources"],
                    self.provenance["test_sources"],
                    self.provenance["registered_inputs"],
                )
                for row in group
            }
            return audit._create_cpu_audit_from_reviewed_snapshot(
                audit.OUTPUT_RELATIVE_PATH,
                output_repo_root=self.repo,
                execution_root=self.repo,
                reviewed_commit=self.provenance["commit"],
                records=records,
            )

    def expected_bindings(self):
        inputs = {
            row["path"]: row for row in self.provenance["registered_inputs"]
        }
        return {
            "expected_implementation_commit": self.provenance["commit"],
            "expected_implementation_hashes": {
                row["path"]: row["sha256"]
                for row in self.provenance["implementation_sources"]
            },
            "expected_environment_lock_sha256": inputs[
                audit.ENVIRONMENT_LOCK_PATH
            ]["sha256"],
            "expected_contract_sha256": self.design_record["contract_sha256"],
            "expected_design_sha256": self.design_record["design_sha256"],
            "expected_tasks_sha256": self.design_record["tasks_sha256"],
            "expected_prompt_csv_sha256": inputs[audit.PROMPT_CSV_PATH][
                "sha256"
            ],
            "expected_exclusion_inventory_sha256": inputs[
                audit.EXCLUSION_INVENTORY_PATH
            ]["sha256"],
            "expected_prompt_manifest_sha256": inputs[
                audit.PROMPT_MANIFEST_PATH
            ]["sha256"],
            "expected_registration_sha256": inputs[audit.REGISTRATION_PATH][
                "sha256"
            ],
        }

    def test_fixed_production_gate_creates_and_validates_one_warning_free_record(self):
        record = self.publish()
        self.assertTrue(self.output.is_file())
        self.assertEqual(
            self.output.read_bytes(), audit.canonical_json_bytes(record)
        )
        summary = audit.validate_cpu_audit_record(
            self.output, **self.expected_bindings()
        )
        self.assertEqual(summary["status"], audit.AUDIT_STATUS)
        self.assertEqual(summary["implementation_commit"], "a" * 40)
        self.assertEqual(summary["test_count"], 73)
        self.assertEqual(summary["warning_count"], 0)

    def test_create_is_one_shot_and_never_invokes_gates_after_output_exists(self):
        original = self.publish()
        original_bytes = self.output.read_bytes()
        forbidden = mock.Mock(side_effect=AssertionError("gate should not run"))
        with mock.patch.object(audit, "_build_cpu_audit_record", forbidden):
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                audit._create_cpu_audit_from_reviewed_snapshot(
                    audit.OUTPUT_RELATIVE_PATH,
                    output_repo_root=self.repo,
                    execution_root=self.repo,
                    reviewed_commit=self.provenance["commit"],
                    records={},
                )
        forbidden.assert_not_called()
        self.assertEqual(self.output.read_bytes(), original_bytes)
        self.assertEqual(json.loads(original_bytes), original)

    def test_create_rejects_every_noncanonical_output_path(self):
        alternate = self.repo / "cpu_audit.json"
        forbidden = mock.Mock(side_effect=AssertionError("gate should not run"))
        with mock.patch.object(audit, "_build_cpu_audit_record", forbidden):
            with self.assertRaisesRegex(ValueError, "output must be exactly"):
                audit._create_cpu_audit_from_reviewed_snapshot(
                    alternate,
                    output_repo_root=self.repo,
                    execution_root=self.repo,
                    reviewed_commit=self.provenance["commit"],
                    records={},
                )
        forbidden.assert_not_called()
        self.assertFalse(alternate.exists())

    def test_production_signature_has_no_gate_or_provenance_injection(self):
        parameters = inspect.signature(audit.create_cpu_audit).parameters
        for forbidden in (
            "source_path",
            "source_repository",
            "provenance_collector",
            "test_runner",
            "prompt_replayer",
            "design_validator",
            "registration_validator",
            "environment_validator",
        ):
            self.assertNotIn(forbidden, parameters)
        with self.assertRaisesRegex(RuntimeError, "reviewed-commit launcher"):
            audit.create_cpu_audit(
                audit.OUTPUT_RELATIVE_PATH, repo_root=self.repo
            )

    def test_receipt_free_builder_cannot_publish_the_official_audit(self):
        record = self.create()
        self.assertEqual(record["status"], audit.AUDIT_STATUS)
        self.assertFalse(self.output.exists())

    def test_builder_rejects_fd_level_stderr_from_validator(self):
        def noisy_design(_root):
            os.write(2, b"native stderr evidence\n")
            return copy.deepcopy(self.design_record)

        with self.assertRaisesRegex(RuntimeError, "stderr_bytes"):
            self.create(design_validator=noisy_design)

    def test_builder_rejects_inherited_subprocess_stderr(self):
        def noisy_design(_root):
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os; os.write(2, b'inherited stderr evidence')",
                ],
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            return copy.deepcopy(self.design_record)

        with self.assertRaisesRegex(RuntimeError, "stderr_bytes"):
            self.create(design_validator=noisy_design)

    def test_builder_captures_multi_megabyte_stderr_without_deadlock(self):
        payload = b"x" * (4 * 1024 * 1024)

        def noisy_design(_root):
            remaining = memoryview(payload)
            while remaining:
                remaining = remaining[os.write(2, remaining) :]
            return copy.deepcopy(self.design_record)

        with self.assertRaisesRegex(RuntimeError, r"stderr_bytes=4194304"):
            self.create(design_validator=noisy_design)

    def test_builder_rejects_same_process_logging(self):
        def noisy_design(_root):
            logging.getLogger("adaptive_oracle.cpu.audit.test").warning(
                "logging evidence"
            )
            return copy.deepcopy(self.design_record)

        with self.assertRaisesRegex(RuntimeError, "logging_records"):
            self.create(design_validator=noisy_design)

    def test_builder_rejects_python_warning(self):
        def noisy_design(_root):
            warnings.warn("warning evidence", UserWarning)
            return copy.deepcopy(self.design_record)

        with self.assertRaisesRegex(RuntimeError, "warnings=UserWarning"):
            self.create(design_validator=noisy_design)

    def test_builder_fails_closed_on_cleanup_error(self):
        class FailingFlushHandler(logging.Handler):
            def emit(self, _record):
                pass

            def flush(self):
                raise RuntimeError("flush failed")

        root_logger = logging.getLogger()
        handler = FailingFlushHandler()
        root_logger.addHandler(handler)
        try:
            with self.assertRaisesRegex(RuntimeError, "cleanup=logging handler flush"):
                self.create()
            self.assertIn(handler, root_logger.handlers)
        finally:
            root_logger.removeHandler(handler)

    def test_validator_exception_restores_fd_and_root_logging_state(self):
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handlers = tuple(root_logger.handlers)
        original_stderr = os.fstat(2)
        original_stderr_inheritable = os.get_inheritable(2)

        def failed_design(_root):
            raise ValueError("validator failed")

        with self.assertRaisesRegex(ValueError, "validator failed"):
            self.create(design_validator=failed_design)

        restored_stderr = os.fstat(2)
        self.assertEqual(
            (restored_stderr.st_dev, restored_stderr.st_ino),
            (original_stderr.st_dev, original_stderr.st_ino),
        )
        self.assertEqual(os.get_inheritable(2), original_stderr_inheritable)
        self.assertEqual(root_logger.level, original_level)
        self.assertEqual(tuple(root_logger.handlers), original_handlers)

    def test_builder_is_reusable_across_successful_calls(self):
        first = self.create()
        second = self.create()
        self.assertEqual(first, second)

    def test_failed_capture_does_not_poison_following_success(self):
        def noisy_design(_root):
            os.write(2, b"one failed call")
            return copy.deepcopy(self.design_record)

        with self.assertRaisesRegex(RuntimeError, "stderr_bytes"):
            self.create(design_validator=noisy_design)
        self.assertEqual(self.create()["status"], audit.AUDIT_STATUS)

    def test_direct_main_requires_verified_launcher_context(self):
        with self.assertRaisesRegex(RuntimeError, "verified execution-root launcher"):
            audit.main([])

    def test_validator_rejects_warning_hash_design_and_scope_tampering(self):
        record = self.create()
        cases = {
            "warning": lambda value: value["warnings"].update(
                {"count": 1, "same_process_count": 1}
            ),
            "source hash": lambda value: value["implementation"]["source_files"][
                0
            ].update({"sha256": "f" * 64}),
            "design": lambda value: value["design"].update({"task_count": 164}),
            "scope": lambda value: value["scope"].update({"scoring": True}),
            "prompt replay": lambda value: value["prompt_replay"].update(
                {"status": "failed"}
            ),
            "registration": lambda value: value["registration"].update(
                {"design_sha256": "0" * 64}
            ),
            "stderr bytes": lambda value: value["tests"].update(
                {"stderr_byte_count": 1}
            ),
            "stderr hash": lambda value: value["tests"].update(
                {"stderr_sha256": hashlib.sha256(b"evidence").hexdigest()}
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(record)
                mutate(changed)
                path = self.repo / f"tampered-{label.replace(' ', '-')}.json"
                path.write_bytes(audit.canonical_json_bytes(changed))
                with self.assertRaises(ValueError):
                    audit.validate_cpu_audit_record(
                        path, **self.expected_bindings()
                    )

    def test_validator_rejects_duplicate_json_keys(self):
        path = self.repo / "duplicate.json"
        path.write_text(
            '{"schema":"first","schema":"second"}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            audit.validate_cpu_audit_record(path)

    def test_atomic_writer_refuses_existing_bytes_and_cleans_temporaries(self):
        path = self.repo / "atomic.json"
        audit.atomic_create_json(path, {"value": 1})
        original = path.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            audit.atomic_create_json(path, {"value": 2})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.repo.glob(".atomic.json.*.tmp")), [])

    def test_test_runner_uses_current_interpreter_cpu_and_warnings_as_errors(self):
        success = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=b"Ran 8 tests in 0.100s\n\nOK\n",
            stderr=b"",
        )
        with mock.patch.object(audit.subprocess, "run", return_value=success) as run:
            record = audit.run_exact_test_suite(self.repo)
        argv = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[1:7], ["-I", "-B", "-S", "-W", "error", "-c"])
        self.assertEqual(argv[7], audit.TEST_BOOTSTRAP)
        self.assertEqual(argv[8:], list(audit.TEST_MODULES))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("PATH", environment)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual(record["test_count"], 8)
        self.assertEqual(record["warning_count"], 0)
        self.assertEqual(record["stderr_byte_count"], 0)
        self.assertEqual(record["stderr_sha256"], hashlib.sha256(b"").hexdigest())

        warned = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                b"file.py:1: UserWarning: warning\n"
                b"Ran 8 tests in 0.100s\n\nOK\n"
            ),
            stderr=b"",
        )
        with mock.patch.object(audit.subprocess, "run", return_value=warned):
            with self.assertRaisesRegex(RuntimeError, "emitted warnings"):
                audit.run_exact_test_suite(self.repo)

    def test_test_runner_rejects_any_success_stderr_bytes(self):
        for label, stderr in (
            ("arbitrary", b"non-warning diagnostic\n"),
            ("large", b"z" * (4 * 1024 * 1024)),
        ):
            with self.subTest(label=label):
                result = subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=b"Ran 8 tests in 0.100s\n\nOK\n",
                    stderr=stderr,
                )
                with mock.patch.object(
                    audit.subprocess, "run", return_value=result
                ):
                    with self.assertRaisesRegex(RuntimeError, "stderr bytes"):
                        audit.run_exact_test_suite(self.repo)

    def test_module_import_does_not_load_project_or_third_party_packages(self):
        program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('isolated_cpu_audit', {str(MODULE_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for forbidden in ('torch', 'yaml', 'diffusers', 'AttentionGuidance', 'InferencePipelines'):
    if forbidden in sys.modules:
        raise SystemExit('loaded:' + forbidden)
print('stdlib-only')
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", program],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "stdlib-only")

    def test_production_source_has_no_head_or_worktree_provenance_path(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn('"HEAD"', source)
        self.assertNotIn("HEAD^{commit}", source)
        self.assertNotIn("collect_clean_provenance", source)

    def test_repository_registration_matches_the_exact_non_executable_schema(self):
        record = audit.validate_engineering_registration(ROOT)
        self.assertEqual(record["status"], "passed_exact_non_executable")
        self.assertEqual(
            record["sha256"],
            audit.sha256_file(ROOT / audit.REGISTRATION_PATH),
        )

    def _reviewed_launch_fixture(self):
        values = {
            path: (self.repo / path).read_bytes()
            for path in audit.AUDITED_TRACKED_PATHS
        }
        commit = "d" * 40
        inventory = {
            path: {
                "sha256": hashlib.sha256(value).hexdigest(),
                "byte_count": len(value),
                "commit": commit,
            }
            for path, value in values.items()
        }
        inputs = {
            name: {
                "path": path,
                **inventory[path],
            }
            for name, path in {
                "registration": audit.REGISTRATION_PATH,
                "prompt_csv": audit.PROMPT_CSV_PATH,
                "exclusion_inventory": audit.EXCLUSION_INVENTORY_PATH,
                "prompt_manifest": audit.PROMPT_MANIFEST_PATH,
                "environment_lock": audit.ENVIRONMENT_LOCK_PATH,
            }.items()
        }
        evidence = {
            "schema": audit.CPU_LAUNCH_SCHEMA,
            "trust_root": "cpython_explicit_commit_launcher_v2",
            "repo_root": str(ROOT),
            "mode": "cpu_audit_reviewed_commit",
            "reviewed_files": inventory,
            "reviewed_commit": commit,
            "python": {},
            "git": {},
            "sys_path": [],
            "repository_sys_path_entries": [],
            "cwd_sys_path_entries": [],
            "launcher": {},
            "inputs": inputs,
            "module_execution": {
                "audit_adaptive_oracle_cpu": {
                    "path": "eval-pipeline/audit_adaptive_oracle_cpu.py",
                    "sha256": inventory[
                        "eval-pipeline/audit_adaptive_oracle_cpu.py"
                    ]["sha256"],
                    "execution_count": 1,
                    "cached": None,
                    "origin": (
                        f"<git-blob:{commit}:"
                        "eval-pipeline/audit_adaptive_oracle_cpu.py>"
                    ),
                }
            },
        }

        class Context:
            def __init__(self):
                self.values = dict(values)
                self.read = set()

            def read_reviewed_file(self, path):
                if path in self.read:
                    raise RuntimeError("duplicate read")
                self.read.add(path)
                return self.values[path]

        return Context(), evidence, values

    def test_reviewed_launch_snapshot_consumes_exact_allowlist_once(self):
        context, evidence, values = self._reviewed_launch_fixture()
        root, commit, observed, records = audit._validated_reviewed_snapshot(
            context, evidence
        )
        self.assertEqual(root, ROOT)
        self.assertEqual(commit, "d" * 40)
        self.assertEqual(observed, values)
        self.assertEqual(context.read, set(audit.AUDITED_TRACKED_PATHS))
        self.assertEqual(set(records), set(audit.AUDITED_TRACKED_PATHS))

    def test_reviewed_launch_snapshot_rejects_hash_and_inventory_drift(self):
        context, evidence, _values = self._reviewed_launch_fixture()
        path = audit.REGISTRATION_PATH
        evidence["reviewed_files"][path]["sha256"] = "0" * 64
        evidence["inputs"]["registration"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "bytes differ"):
            audit._validated_reviewed_snapshot(context, evidence)

        context, evidence, _values = self._reviewed_launch_fixture()
        evidence["reviewed_files"].pop(path)
        with self.assertRaisesRegex(RuntimeError, "inventory differs"):
            audit._validated_reviewed_snapshot(context, evidence)

    def test_materialized_snapshot_contains_only_reviewed_paths(self):
        _context, _evidence, values = self._reviewed_launch_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit._materialize_reviewed_snapshot(root, values)
            observed = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(observed, set(audit.AUDITED_TRACKED_PATHS))
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
