import csv
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import torch


ROOT = Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generation = load_module(
    "generate_adaptive_oracle_engineering_test",
    EVAL_PIPELINE / "generate_adaptive_oracle_engineering.py",
)
environment = load_module(
    "adaptive_oracle_generation_environment_test",
    EVAL_PIPELINE / "adaptive_oracle_generation_environment.py",
)
contract = generation.contract


def digest(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class _HookBlock:
    def __init__(self):
        self._forward_pre_hooks = {}


class _FakeUNet:
    def __init__(self, device="cuda:1"):
        self.up_blocks = [_HookBlock()]
        self._parameter = SimpleNamespace(device=device)

    def parameters(self):
        yield self._parameter


class GenerationEnvironmentDeviceTest(unittest.TestCase):
    def _properties(self, uuid="11111111-2222-3333-4444-555555555555"):
        return SimpleNamespace(
            name="NVIDIA GeForce RTX 3090", major=8, minor=6, uuid=uuid
        )

    def test_requested_unmasked_physical_device_is_observed(self):
        smi_output = (
            "3, GPU-11111111-2222-3333-4444-555555555555, "
            "00000000:03:00.0, NVIDIA GeForce RTX 3090, 580.126.09\n"
        )
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(
                environment.subprocess, "check_output", return_value=smi_output
            ),
            mock.patch.object(environment.torch.cuda, "is_available", return_value=True),
            mock.patch.object(environment.torch.cuda, "device_count", return_value=8),
            mock.patch.object(
                environment.torch.cuda,
                "get_device_properties",
                return_value=self._properties(),
            ),
        ):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("CUDA_DEVICE_ORDER", None)
            observed = environment.observed_environment(
                [], cuda_device="cuda:3", require_unmasked_cuda=True
            )

        self.assertEqual(observed["cuda_device"]["logical_index"], 3)
        self.assertEqual(
            observed["cuda_device"]["torch"]["uuid"],
            "GPU-11111111-2222-3333-4444-555555555555",
        )
        self.assertEqual(observed["cuda_device"]["binding"], {
            "uuid_match": True,
            "name_match": True,
            "unmasked_physical_index_match": True,
        })

    def test_cuda_remapping_variables_are_rejected(self):
        for name, value in (
            ("CUDA_VISIBLE_DEVICES", "3"),
            ("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
        ):
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: value}):
                with self.assertRaisesRegex(ValueError, "must be unset"):
                    environment.observed_environment(
                        [], cuda_device="cuda:1", require_unmasked_cuda=True
                    )

    def test_same_name_with_different_uuid_is_rejected(self):
        smi_output = (
            "3, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, "
            "00000000:03:00.0, NVIDIA GeForce RTX 3090, 580.126.09\n"
        )
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(
                environment.subprocess, "check_output", return_value=smi_output
            ),
            mock.patch.object(environment.torch.cuda, "is_available", return_value=True),
            mock.patch.object(environment.torch.cuda, "device_count", return_value=8),
            mock.patch.object(
                environment.torch.cuda,
                "get_device_properties",
                return_value=self._properties(),
            ),
        ):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("CUDA_DEVICE_ORDER", None)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                environment.observed_environment(
                    [], cuda_device="cuda:3", require_unmasked_cuda=True
                )

    def test_matching_uuid_at_different_physical_index_is_rejected(self):
        smi_output = (
            "2, GPU-11111111-2222-3333-4444-555555555555, "
            "00000000:02:00.0, NVIDIA GeForce RTX 3090, 580.126.09\n"
        )
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(
                environment.subprocess, "check_output", return_value=smi_output
            ),
            mock.patch.object(environment.torch.cuda, "is_available", return_value=True),
            mock.patch.object(environment.torch.cuda, "device_count", return_value=8),
            mock.patch.object(
                environment.torch.cuda,
                "get_device_properties",
                return_value=self._properties(),
            ),
        ):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("CUDA_DEVICE_ORDER", None)
            with self.assertRaisesRegex(ValueError, "physical index"):
                environment.observed_environment(
                    [], cuda_device="cuda:3", require_unmasked_cuda=True
                )

    def test_missing_torch_uuid_is_rejected(self):
        with (
            mock.patch.object(environment.torch.cuda, "is_available", return_value=True),
            mock.patch.object(environment.torch.cuda, "device_count", return_value=8),
            mock.patch.object(
                environment.torch.cuda,
                "get_device_properties",
                return_value=self._properties(uuid=None),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "PyTorch.*UUID"):
                environment.observed_environment([], cuda_device="cuda:3")

    def test_duplicate_and_malformed_smi_identities_are_rejected(self):
        duplicate = (
            "2, GPU-11111111-2222-3333-4444-555555555555, "
            "00000000:02:00.0, NVIDIA GeForce RTX 3090, 580.126.09\n"
            "2, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, "
            "00000000:03:00.0, NVIDIA GeForce RTX 3090, 580.126.09\n"
        )
        malformed = (
            "2, GPU-11111111-2222-3333-4444-555555555555, invalid-pci, "
            "NVIDIA GeForce RTX 3090, 580.126.09\n"
        )
        for value, message in ((duplicate, "duplicate GPU index"), (malformed, "invalid PCI")):
            with self.subTest(message=message), mock.patch.object(
                environment.subprocess, "check_output", return_value=value
            ):
                with self.assertRaisesRegex(ValueError, message):
                    environment._nvidia_smi_inventory()


class LauncherEntrypointTest(unittest.TestCase):
    class Context:
        def __init__(self):
            self.events = []

        def claim(self):
            self.events.append("claim")
            return {"launcher": "evidence"}

        def read_input(self, name):
            self.events.append(f"read:{name}")
            return f"bytes:{name}".encode("ascii")

    def test_only_launcher_entrypoint_can_reach_generation(self):
        self.assertFalse(hasattr(generation, "run_engineering_generation"))
        self.assertFalse(hasattr(generation, "_run_engineering_generation_testable"))
        self.assertFalse(hasattr(generation, "validate_executable_authorization"))
        self.assertEqual(
            set(generation._run_from_verified_launcher.__annotations__),
            {"argv", "launcher_context", "return"},
        )
        with self.assertRaisesRegex(RuntimeError, "verified execution-root launcher"):
            generation.main(["--device", "cuda:1"])

    def test_launcher_context_is_claimed_and_all_inputs_are_consumed_once(self):
        context = self.Context()
        validated = {"validated": True}
        success = {"status": "complete"}
        with (
            mock.patch.object(
                generation, "_validate_launcher_bundle", return_value=validated
            ) as validate,
            mock.patch.object(
                generation,
                "_run_authorized_engineering_generation",
                return_value=success,
            ) as run,
            mock.patch("builtins.print") as output,
        ):
            result = generation._run_from_verified_launcher(
                [
                    "--device",
                    "cuda:2",
                    "--output-dir",
                    str(ROOT / generation.OUTPUT_DIR),
                ],
                launcher_context=context,
            )

        self.assertEqual(result, 0)
        self.assertEqual(context.events, [
            "claim",
            "read:registration",
            "read:cpu_audit",
            "read:prompt_csv",
            "read:prompt_manifest",
            "read:exclusion_inventory",
            "read:environment_lock",
        ])
        supplied = validate.call_args.args
        self.assertEqual(supplied[0], {"launcher": "evidence"})
        self.assertEqual(set(supplied[1]), {
            "registration",
            "cpu_audit",
            "prompt_csv",
            "prompt_manifest",
            "exclusion_inventory",
            "environment_lock",
        })
        self.assertEqual(validate.call_args.kwargs["requested_device"], "cuda:2")
        run.assert_called_once_with(validated)
        output.assert_called_once_with(json.dumps(success, sort_keys=True))

    def test_argument_failure_does_not_claim_launcher_context(self):
        context = self.Context()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            generation._run_from_verified_launcher([], launcher_context=context)
        self.assertEqual(context.events, [])


class CommitTopologyEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.implementation_commit = "a" * 40
        self.cpu_audit_commit = "b" * 40
        self.authorization_commit = "c" * 40
        self.binding = {
            "path": generation.AUTHORIZATION_PATH,
            "sha256": "d" * 64,
            "commit": self.authorization_commit,
        }
        self.topology = {
            "schema": generation.COMMIT_TOPOLOGY_SCHEMA,
            "implementation_commit": self.implementation_commit,
            "cpu_audit_commit": self.cpu_audit_commit,
            "authorization_commit": self.authorization_commit,
            "head_commit": self.authorization_commit,
            "cpu_audit_parent": self.implementation_commit,
            "authorization_parent": self.cpu_audit_commit,
        }

    def validate(self, topology=None):
        return generation._validate_commit_topology(
            self.topology if topology is None else topology,
            reviewed_commit=self.cpu_audit_commit,
            authorization_binding=self.binding,
        )

    def test_accepts_exact_implementation_audit_authorization_chain(self):
        self.assertEqual(self.validate(), self.topology)

    def test_rejects_single_field_topology_forgery(self):
        mutations = {
            "implementation_commit": "e" * 40,
            "cpu_audit_commit": "e" * 40,
            "authorization_commit": "e" * 40,
            "head_commit": "e" * 40,
            "cpu_audit_parent": "e" * 40,
            "authorization_parent": "e" * 40,
        }
        for name, replacement in mutations.items():
            with self.subTest(name=name):
                forged = dict(self.topology)
                forged[name] = replacement
                with self.assertRaisesRegex(ValueError, "topology binding differs"):
                    self.validate(forged)

    def test_rejects_collapsed_or_malformed_commit_identities(self):
        collapsed = dict(self.topology)
        collapsed["implementation_commit"] = self.cpu_audit_commit
        collapsed["cpu_audit_parent"] = self.cpu_audit_commit
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            self.validate(collapsed)

        malformed = dict(self.topology)
        malformed["implementation_commit"] = "A" * 40
        with self.assertRaisesRegex(ValueError, "not a full Git SHA"):
            self.validate(malformed)

    def test_rejects_topology_not_bound_to_authorization_commit(self):
        binding = dict(self.binding)
        binding["commit"] = "e" * 40
        with self.assertRaisesRegex(ValueError, "topology binding differs"):
            generation._validate_commit_topology(
                self.topology,
                reviewed_commit=self.cpu_audit_commit,
                authorization_binding=binding,
            )


class ImmutableLauncherInputTest(unittest.TestCase):
    def test_revalidation_rejects_a_tampered_canonical_input_path(self):
        implementation_commit = "a" * 40
        reviewed_commit = "b" * 40
        authorization_commit = "c" * 40
        topology = {
            "schema": generation.COMMIT_TOPOLOGY_SCHEMA,
            "implementation_commit": implementation_commit,
            "cpu_audit_commit": reviewed_commit,
            "authorization_commit": authorization_commit,
            "head_commit": authorization_commit,
            "cpu_audit_parent": implementation_commit,
            "authorization_parent": reviewed_commit,
        }
        binding = {
            "path": generation.AUTHORIZATION_PATH,
            "sha256": "d" * 64,
            "commit": authorization_commit,
        }
        held = {
            name: f"held:{name}".encode("ascii")
            for name in generation._LAUNCH_INPUT_PATHS
        }
        inventory = {
            name: {
                "path": path,
                "sha256": digest(held[name]),
                "byte_count": len(held[name]),
                "commit": reviewed_commit,
            }
            for name, path in generation._LAUNCH_INPUT_PATHS.items()
        }
        inventory["registration"] = dict(inventory["registration"])
        inventory["registration"]["path"] = "eval-pipeline/configs/forged.json"
        validated = {
            "commit_topology": topology,
            "reviewed_commit": reviewed_commit,
            "authorization_binding": binding,
            "launcher_evidence": {
                "commit_topology": dict(topology),
                "reviewed_commit": reviewed_commit,
                "authorization_binding": dict(binding),
            },
            "cpu_audit_summary": {"implementation_commit": implementation_commit},
            "authorized_input_bytes": held,
            "launcher_inputs": inventory,
        }
        with self.assertRaisesRegex(
            RuntimeError, "held authorized input binding drifted.*registration"
        ):
            generation._revalidate_immutable_authorized_bundle(
                validated,
                SimpleNamespace(),
                phase="after_runtime_import",
            )


class RuntimeEvidenceCaptureTest(unittest.TestCase):
    @staticmethod
    def _fd_identity(descriptor):
        value = os.fstat(descriptor)
        return (value.st_dev, value.st_ino, value.st_mode, value.st_rdev)

    def test_captures_native_and_large_subprocess_fd2_without_truncation(self):
        capture = generation._RuntimeEvidenceCapture()
        native = b"warning: native-fd2-test\n"
        child_marker = "warning: child-fd2-test"
        child_chunk_count = 2048
        child_chunk = b"x" * 128
        with capture:
            os.write(2, native)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os;"
                        f"chunk={child_chunk!r};"
                        f"[os.write(2,chunk) for _ in range({child_chunk_count})];"
                        f"os.write(2,{(chr(10) + child_marker + chr(10)).encode()!r})"
                    ),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                check=True,
            )

        record = capture.record()
        self.assertTrue(record["stderr"].startswith(native.decode("ascii")))
        self.assertTrue(record["stderr"].endswith(child_marker + "\n"))
        self.assertEqual(
            record["stderr_byte_count"],
            len(native) + child_chunk_count * len(child_chunk) + len(child_marker) + 2,
        )
        self.assertEqual(record["warnings"]["stderr_warning_line_count"], 2)
        self.assertEqual(record["warnings"]["count"], 2)
        self.assertIsNone(capture._stderr_thread)

    def test_exception_path_restores_fd2_and_releases_capture_resources(self):
        before = self._fd_identity(2)
        capture = generation._RuntimeEvidenceCapture()
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with capture:
                os.write(2, b"warning: before-exception\n")
                raise RuntimeError("fixture failure")

        self.assertEqual(self._fd_identity(2), before)
        self.assertIsNone(capture._stderr_saved_fd)
        self.assertIsNone(capture._stderr_read_fd)
        self.assertIsNone(capture._stderr_stream)
        self.assertIsNone(capture._stderr_thread)
        self.assertEqual(capture.record()["warnings"]["count"], 1)

    def test_raw_stderr_warning_categories_are_counted(self):
        capture = generation._RuntimeEvidenceCapture()
        raw = (
            b"worker.py:1: UserWarning: user detail\n"
            b"FutureWarning: future detail\n"
            b"RuntimeWarning: runtime detail\n"
            b"WARN: native fallback\n"
            b"[W0827 12:34:56 file.cc:1] native fallback\n"
        )
        with capture:
            os.write(2, raw)

        record = capture.record()
        self.assertEqual(record["warnings"]["stderr_warning_line_count"], 5)
        self.assertEqual(record["warnings"]["count"], 5)
        with self.assertRaisesRegex(RuntimeError, "runtime warnings"):
            generation._require_warning_free_runtime_evidence(record)

        self.assertEqual(
            generation.contract.stderr_warning_line_count(
                "worker finished\nreward model loaded\n"
            ),
            0,
        )

    def test_non_utf8_fd2_evidence_is_losslessly_hashed(self):
        capture = generation._RuntimeEvidenceCapture()
        raw = b"native-byte:\xff\xfe\n"
        with capture:
            os.write(2, raw)

        record = capture.record()
        recovered = record["stderr"].encode("utf-8", errors="surrogateescape")
        self.assertEqual(recovered, raw)
        self.assertEqual(record["stderr_byte_count"], len(raw))
        self.assertEqual(record["stderr_sha256"], digest(raw))
        contract.canonical_json_bytes(record)

    def test_capture_exit_fails_for_inheriting_live_child(self):
        capture = generation._RuntimeEvidenceCapture()
        child = None
        try:
            started = time.monotonic()
            with self.assertRaisesRegex(
                RuntimeError, "runtime evidence capture finalization failed"
            ):
                with capture:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            "import sys; sys.stdin.buffer.read(1)",
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                    )
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertIsNone(child.poll())
            self.assertIsNone(capture._stderr_thread)
        finally:
            if child is not None:
                if child.stdin is not None:
                    child.stdin.close()
                child.wait(timeout=5)


class ExclusionInventoryBindingTest(unittest.TestCase):
    def setUp(self):
        self.inventory_bytes = (
            EVAL_PIPELINE
            / "prompts"
            / "adaptive_oracle_exclusion_inventory_v1.json"
        ).read_bytes()
        self.manifest = json.loads(
            (
                EVAL_PIPELINE
                / "prompts"
                / "adaptive_oracle_prompt_manifest_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.inventory_sha256 = digest(self.inventory_bytes)

    def test_accepts_exact_manifest_and_inventory_binding(self):
        inventory = generation._validate_exclusion_inventory_binding(
            self.manifest,
            self.inventory_bytes,
            expected_sha256=self.inventory_sha256,
        )
        self.assertEqual(
            inventory["schema"], generation.EXCLUSION_INVENTORY_SCHEMA
        )

    def test_rejects_manifest_binding_field_and_value_tampering(self):
        mutations = {}
        for field in ("schema", "path", "sha256"):
            missing = json.loads(json.dumps(self.manifest))
            del missing["exclusion_inventory"][field]
            mutations[f"missing_{field}"] = missing
        extra = json.loads(json.dumps(self.manifest))
        extra["exclusion_inventory"]["extra"] = True
        mutations["extra_field"] = extra
        for field, value in (
            ("schema", "wrong_schema"),
            ("path", "eval-pipeline/prompts/wrong.json"),
            ("sha256", "0" * 64),
        ):
            changed = json.loads(json.dumps(self.manifest))
            changed["exclusion_inventory"][field] = value
            mutations[f"wrong_{field}"] = changed

        for name, manifest in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                generation._validate_exclusion_inventory_binding(
                    manifest,
                    self.inventory_bytes,
                    expected_sha256=self.inventory_sha256,
                )

    def test_rejects_inventory_raw_byte_and_authorized_hash_tampering(self):
        cases = (
            (self.inventory_bytes + b"\n", self.inventory_sha256),
            (self.inventory_bytes, "0" * 64),
        )
        for inventory_bytes, expected_sha256 in cases:
            with self.subTest(expected_sha256=expected_sha256), self.assertRaises(
                ValueError
            ):
                generation._validate_exclusion_inventory_binding(
                    self.manifest,
                    inventory_bytes,
                    expected_sha256=expected_sha256,
                )

    def test_rejects_inventory_payload_with_wrong_schema(self):
        inventory = json.loads(self.inventory_bytes)
        inventory["schema"] = "wrong_schema"
        raw = generation.contract.canonical_json_bytes(inventory)
        manifest = json.loads(json.dumps(self.manifest))
        manifest["exclusion_inventory"]["sha256"] = digest(raw)
        with self.assertRaisesRegex(ValueError, "schema differs"):
            generation._validate_exclusion_inventory_binding(
                manifest,
                raw,
                expected_sha256=digest(raw),
            )


class PipelineStagingTest(unittest.TestCase):
    def _runtime_and_pipe(self, events):
        class Pipe:
            def __init__(self):
                self.unet = _FakeUNet()
                self.scheduler = object()

            def to(self, device):
                events.append(f"to:{device}")
                return self

            def set_progress_bar_config(self, *, disable):
                self.progress_disabled = disable

        pipe = Pipe()

        class Pipeline:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                events.append(f"load:{path}")
                cls.call = (path, kwargs)
                return pipe

        preview = object()
        scheduler_factory = SimpleNamespace(
            from_config=mock.Mock(return_value=preview)
        )
        runtime = SimpleNamespace(
            Pipeline=Pipeline,
            Scheduler=scheduler_factory,
            torch=SimpleNamespace(float16=torch.float16),
        )
        return runtime, pipe, Pipeline, preview

    def test_pipeline_load_uses_pinned_stage_and_verifies_three_times(self):
        events = []
        runtime, pipe, pipeline_class, preview = self._runtime_and_pipe(events)
        stage = {"path": "/private/adaptive-oracle-model-test"}
        validated = {
            "device": "cuda:1",
            "config": {"model": generation._expected_model()},
        }

        def verify(*_args, **_kwargs):
            index = sum(item.startswith("verify") for item in events) + 1
            events.append(f"verify:{index}")
            return {"verification": index}

        def bound_load(_record, loader, **_kwargs):
            log = _kwargs["verification_log"]
            log.append(verify())
            loaded = loader("/proc/self/fd/99")
            log.append(verify())
            return loaded

        with (
            mock.patch.object(
                generation.model_snapshot,
                "load_from_verified_staged_model_snapshot",
                side_effect=bound_load,
            ),
            mock.patch.object(
                generation.model_snapshot,
                "verify_staged_model_snapshot",
                side_effect=verify,
            ),
            mock.patch.object(
                generation,
                "_validate_base_scheduler",
                return_value={"prediction_type": "epsilon"},
            ),
            mock.patch.object(generation, "PIPELINE_CLASS", "Pipe"),
            mock.patch.object(generation, "_validate_scheduler_schedule") as schedule,
        ):
            loaded, base, verifications = generation._load_pipeline(
                validated, runtime, stage
            )

        self.assertIs(loaded, pipe)
        self.assertEqual(base, {"prediction_type": "epsilon"})
        self.assertEqual(events, [
            "verify:1",
            "load:/proc/self/fd/99",
            "verify:2",
            "to:cuda:1",
            "verify:3",
        ])
        self.assertEqual(len(verifications), 3)
        path, kwargs = pipeline_class.call
        self.assertEqual(path, "/proc/self/fd/99")
        self.assertEqual(kwargs, {
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "local_files_only": True,
        })
        runtime.Scheduler.from_config.assert_called_once_with(
            {"prediction_type": "epsilon"}
        )
        schedule.assert_called_once_with(preview, device="cuda:1")

    def test_each_pipeline_load_verification_failure_stops_later_steps(self):
        for failing_call in (1, 2, 3):
            with self.subTest(failing_call=failing_call):
                events = []
                verification_log = []
                runtime, _pipe, _pipeline_class, _preview = self._runtime_and_pipe(events)
                calls = 0

                def verify(*_args, **_kwargs):
                    nonlocal calls
                    calls += 1
                    events.append(f"verify:{calls}")
                    if calls == failing_call:
                        raise ValueError(f"verification {calls} failed")
                    return {"verification": calls}

                def bound_load(_record, loader, **_kwargs):
                    log = _kwargs["verification_log"]
                    log.append(verify())
                    loaded = loader("/proc/self/fd/99")
                    log.append(verify())
                    return loaded

                with (
                    mock.patch.object(
                        generation.model_snapshot,
                        "load_from_verified_staged_model_snapshot",
                        side_effect=bound_load,
                    ),
                    mock.patch.object(
                        generation.model_snapshot,
                        "verify_staged_model_snapshot",
                        side_effect=verify,
                    ),
                    mock.patch.object(
                        generation,
                        "_validate_base_scheduler",
                        return_value={"prediction_type": "epsilon"},
                    ),
                    mock.patch.object(generation, "_validate_scheduler_schedule"),
                ):
                    with self.assertRaisesRegex(ValueError, "verification"):
                        generation._load_pipeline(
                            {
                                "device": "cuda:1",
                                "config": {"model": generation._expected_model()},
                            },
                            runtime,
                            {"path": "/private/stage"},
                            verification_log=verification_log,
                        )
                self.assertEqual(len(verification_log), failing_call - 1)
                if failing_call == 1:
                    self.assertEqual(events, ["verify:1"])
                elif failing_call == 2:
                    self.assertNotIn("to:cuda:1", events)
                else:
                    self.assertIn("to:cuda:1", events)


class AuthorizedGenerationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "outputs" / "adaptive_oracle").mkdir(parents=True)
        self.output = self.root / generation.OUTPUT_DIR
        with (EVAL_PIPELINE / "prompts" / "adaptive_oracle_engineering.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            self.rows = list(csv.DictReader(handle))
        self.prompt_manifest = json.loads(
            (
                EVAL_PIPELINE
                / "prompts"
                / "adaptive_oracle_prompt_manifest_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.design = contract.build_engineering_design(
            self.rows, self.prompt_manifest
        )
        self.launcher_evidence = {
            "schema": generation.LAUNCH_SCHEMA,
            "fixture": "reviewed-launcher-evidence",
        }
        self.validated = {
            "config": {
                "model": generation._expected_model(),
                "fixture": "authorized-config",
            },
            "authorization_sha256": "a" * 64,
            "authorization_binding": {
                "path": generation.AUTHORIZATION_PATH,
                "sha256": "a" * 64,
                "commit": "b" * 40,
            },
            "reviewed_commit": "c" * 40,
            "source_paths": {
                "environment_lock": generation.SOURCE_PATHS["environment_lock"]
            },
            "source_hashes": {
                "environment_lock": {
                    "path": generation.SOURCE_PATHS["environment_lock"],
                    "sha256": digest(b"environment-lock"),
                }
            },
            "prompt_rows": self.rows,
            "prompt_manifest": self.prompt_manifest,
            "design": self.design,
            "launcher_evidence": self.launcher_evidence,
            "environment_lock_bytes": b"environment-lock",
            "device": "cuda:1",
            "output_dir": str(self.output),
            "repo_root": str(self.root),
        }
        self.stage = {
            "schema": generation.model_snapshot.MODEL_STAGE_SCHEMA,
            "status": "staged_verified_read_only",
            "path": str(self.root / "private-model-stage"),
            "parent": str(self.root),
            "manifest_sha256": generation._expected_model()[
                "snapshot_manifest_sha256"
            ],
            "loaded_file_count": generation._expected_model()[
                "snapshot_loaded_file_count"
            ],
            "tree_sha256": "d" * 64,
            "root_identity": {"st_dev": 1, "st_ino": 2},
            "source_snapshot": {"fixture": True},
            "source_snapshot_sha256": "e" * 64,
            "manifest": generation.model_snapshot.expected_model_manifest(),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _runtime(self):
        environment_record = {
            "schema": generation.ENVIRONMENT_LOCK_SCHEMA,
            "lock_id": "fixture",
            "path": generation.SOURCE_PATHS["environment_lock"],
            "sha256": digest(b"environment-lock"),
            "observed": {"fixture": True},
        }
        return SimpleNamespace(
            validate_environment_lock_bytes=mock.Mock(
                return_value=environment_record
            ),
            verified_source_execution={"fixture": {"origin": "reviewed"}},
        )

    @staticmethod
    def _verification(index):
        return {
            "schema": generation.model_snapshot.MODEL_STAGE_VERIFICATION_SCHEMA,
            "status": "verified_unchanged",
            "path": "/private-model-stage",
            "manifest_sha256": "f" * 64,
            "loaded_file_count": 17,
            "tree_sha256": digest(f"tree:{index}"),
            "root_identity": {"st_dev": 1, "st_ino": 2},
            "index": index,
        }

    def _run_patches(
        self,
        events,
        *,
        task_error=None,
        stage_error=None,
        fourth_verification_error=None,
        cleanup_error=None,
        trace_atomic_json=False,
    ):
        runtime = self._runtime()
        pipe = SimpleNamespace(unet=_FakeUNet())
        initial_verifications = [self._verification(i) for i in (1, 2, 3)]
        real_atomic_json = generation.atomic_create_json

        def stage_model(*_args, **_kwargs):
            events.append("stage")
            if stage_error is not None:
                raise stage_error
            return self.stage

        def load_pipeline(*_args, **kwargs):
            events.extend(("verify:1", "load", "verify:2", "to", "verify:3"))
            kwargs["verification_log"].extend(initial_verifications)
            return pipe, {"prediction_type": "epsilon"}, initial_verifications

        def fourth_verify(*_args, **_kwargs):
            events.append("verify:4")
            if fourth_verification_error is not None:
                raise fourth_verification_error
            return self._verification(4)

        def execute_task(**kwargs):
            events.append("task")
            if task_error is not None:
                raise task_error
            return {"task_id": kwargs["task"]["task_id"]}, b"fixture-png"

        def validate_sidecar(_record, task):
            return {
                "task_id": task["task_id"],
                "action_id": task["action"]["action_id"],
                "prompt_row_id": task["prompt"]["prompt_row_id"],
                "physical_no_op": task["action"]["physical_no_op"],
                "png_sha256": digest(task["task_id"]),
            }

        def validate_block(_records, tasks):
            return {
                "prompt_row_id": tasks[0]["prompt"]["prompt_row_id"],
                "block_sha256": digest(tasks[0]["task_id"]),
            }

        def cleanup_stage(record, *_args, **_kwargs):
            events.append("model_cleanup")
            if cleanup_error is not None:
                raise cleanup_error
            return {
                "schema": "adaptive_oracle_model_stage_cleanup_v1",
                "status": "removed",
                "path": record["path"],
                "manifest_sha256": record.get("manifest_sha256"),
                "loaded_file_count": record.get("loaded_file_count"),
                "root_identity": record["root_identity"],
            }

        def cleanup_runtime(*_args, **_kwargs):
            events.append("runtime_cleanup")

        def revalidate(_validated, _runtime, *, phase):
            events.append(f"revalidate:{phase}")
            return {"phase": phase}

        def atomic_json(path, value):
            if trace_atomic_json:
                events.append(f"write:{Path(path).name}")
            return real_atomic_json(path, value)

        patches = (
            mock.patch.object(
                generation, "_load_verified_runtime_components", return_value=runtime
            ),
            mock.patch.object(
                generation,
                "_revalidate_immutable_authorized_bundle",
                side_effect=revalidate,
            ),
            mock.patch.object(generation, "_validate_cuda_device"),
            mock.patch.object(
                generation.model_snapshot,
                "stage_model_snapshot",
                side_effect=stage_model,
            ),
            mock.patch.object(
                generation, "_load_pipeline", side_effect=load_pipeline
            ),
            mock.patch.object(
                generation.model_snapshot,
                "verify_staged_model_snapshot",
                side_effect=fourth_verify,
            ),
            mock.patch.object(
                generation,
                "_raw_initial_noise",
                side_effect=[object() for _ in range(contract.PROMPT_COUNT)],
            ),
            mock.patch.object(generation, "_execute_task", side_effect=execute_task),
            mock.patch.object(
                generation.contract,
                "validate_sidecar",
                side_effect=validate_sidecar,
            ),
            mock.patch.object(
                generation.contract,
                "validate_prompt_block",
                side_effect=validate_block,
            ),
            mock.patch.object(
                generation.model_snapshot,
                "cleanup_staged_model_snapshot",
                side_effect=cleanup_stage,
            ),
            mock.patch.object(
                generation,
                "_cleanup_runtime_resources",
                side_effect=cleanup_runtime,
            ),
            mock.patch.object(
                generation, "atomic_create_json", side_effect=atomic_json
            ),
        )
        return runtime, patches

    def _run(self, events, **kwargs):
        runtime, patches = self._run_patches(events, **kwargs)
        started = []
        try:
            for patcher in patches:
                patcher.start()
                started.append(patcher)
            result = generation._run_authorized_engineering_generation(
                self.validated
            )
        finally:
            for patcher in reversed(started):
                patcher.stop()
        return result, runtime

    def test_success_binds_launcher_and_model_stage_evidence(self):
        events = []
        success, runtime = self._run(events)

        self.assertEqual(success["record_count"], contract.TOTAL_TASK_COUNT)
        self.assertEqual(events[:7], [
            "revalidate:after_runtime_import",
            "stage",
            "verify:1",
            "load",
            "verify:2",
            "to",
            "verify:3",
        ])
        self.assertIn("verify:4", events)
        self.assertEqual(events.count("revalidate:after_runtime_import"), 1)
        self.assertEqual(events.count("revalidate:after_generation"), 1)
        self.assertLess(
            events.index("task"), events.index("revalidate:after_generation")
        )
        self.assertLess(
            events.index("revalidate:after_generation"),
            events.index("runtime_cleanup"),
        )
        self.assertEqual(events.count("task"), contract.TOTAL_TASK_COUNT)
        self.assertLess(events.index("verify:4"), events.index("task"))
        self.assertLess(events.index("runtime_cleanup"), events.index("model_cleanup"))
        expected_files = {
            "attempt.json",
            "config.json",
            "manifest.json",
            "model_stage_evidence.json",
            "records",
            "runtime_evidence.json",
            "success.json",
        }
        self.assertEqual({path.name for path in self.output.iterdir()}, expected_files)
        self.assertFalse((self.output / "failure.json").exists())

        attempt = json.loads((self.output / "attempt.json").read_text())
        config = json.loads((self.output / "config.json").read_text())
        persisted_success = json.loads((self.output / "success.json").read_text())
        stage_evidence = json.loads(
            (self.output / "model_stage_evidence.json").read_text()
        )
        for record in (attempt, config, persisted_success):
            self.assertEqual(record["launcher_evidence"], self.launcher_evidence)
        self.assertEqual(stage_evidence["status"], "removed")
        self.assertEqual(len(stage_evidence["verifications"]), 4)
        self.assertEqual(
            config["model_stage_evidence_sha256"],
            contract.canonical_sha256(stage_evidence),
        )
        self.assertEqual(
            persisted_success["model_stage_evidence_file_sha256"],
            digest((self.output / "model_stage_evidence.json").read_bytes()),
        )
        runtime.validate_environment_lock_bytes.assert_called_once_with(
            b"environment-lock",
            source_path=generation.SOURCE_PATHS["environment_lock"],
            expected_sha256=digest(b"environment-lock"),
            cuda_device="cuda:1",
            require_unmasked_cuda=True,
        )

    def test_pre_generation_verification_failure_runs_no_task(self):
        events = []
        with self.assertRaisesRegex(ValueError, "pre-generation mutation"):
            self._run(
                events,
                fourth_verification_error=ValueError("pre-generation mutation"),
            )
        self.assertNotIn("task", events)
        self.assertTrue((self.output / "failure.json").is_file())
        self.assertFalse((self.output / "success.json").exists())

    def test_failure_receipt_precedes_runtime_and_model_cleanup(self):
        events = []
        with self.assertRaisesRegex(RuntimeError, "task failed"):
            self._run(
                events,
                task_error=RuntimeError("task failed"),
                trace_atomic_json=True,
            )
        self.assertLess(events.index("write:failure.json"), events.index("runtime_cleanup"))
        self.assertLess(events.index("write:failure.json"), events.index("model_cleanup"))
        self.assertLess(
            events.index("model_cleanup"),
            events.index("write:model_stage_evidence.json"),
        )
        failure = json.loads((self.output / "failure.json").read_text())
        self.assertEqual(failure["launcher_evidence"], self.launcher_evidence)
        self.assertEqual(len(failure["model_stage_verifications"]), 4)
        self.assertFalse((self.output / "success.json").exists())

    def test_cleanup_failure_preserves_receipt_and_records_stage_failure(self):
        events = []
        with self.assertRaisesRegex(RuntimeError, "model-stage cleanup also failed"):
            self._run(
                events,
                task_error=ValueError("generation failed"),
                cleanup_error=OSError("cleanup denied"),
                trace_atomic_json=True,
            )
        self.assertLess(events.index("write:failure.json"), events.index("model_cleanup"))
        evidence = json.loads(
            (self.output / "model_stage_evidence.json").read_text()
        )
        self.assertEqual(evidence["status"], "cleanup_failed_terminal")
        self.assertEqual(
            evidence["cleanup_failure"]["exception_message"], "cleanup denied"
        )
        self.assertTrue((self.output / "failure.json").is_file())

    def test_partial_stage_creation_failure_is_receipted_then_retried_and_removed(self):
        events = []
        partial_stage = {
            "schema": generation.model_snapshot.MODEL_STAGE_SCHEMA,
            "status": "staging_failed_cleanup_pending",
            "path": str(self.root / "partial-model-stage"),
            "parent": str(self.root),
            "root_identity": {"st_dev": 7, "st_ino": 11},
        }
        stage_error = generation.model_snapshot.ModelStageCreationError(
            ValueError("copy failed"),
            OSError("initial cleanup failed"),
            partial_stage,
        )
        with self.assertRaises(generation.model_snapshot.ModelStageCreationError):
            self._run(
                events,
                stage_error=stage_error,
                trace_atomic_json=True,
            )

        self.assertLess(events.index("write:failure.json"), events.index("model_cleanup"))
        failure = json.loads((self.output / "failure.json").read_text())
        self.assertEqual(failure["model_stage"], partial_stage)
        self.assertEqual(failure["model_stage_verifications"], [])
        evidence = json.loads(
            (self.output / "model_stage_evidence.json").read_text()
        )
        self.assertEqual(evidence["stage"], partial_stage)
        self.assertEqual(evidence["status"], "removed")
        self.assertIsNone(evidence["cleanup"]["manifest_sha256"])
        self.assertIsNone(evidence["cleanup"]["loaded_file_count"])

    def test_partial_stage_retry_failure_preserves_terminal_evidence(self):
        events = []
        partial_stage = {
            "schema": generation.model_snapshot.MODEL_STAGE_SCHEMA,
            "status": "staging_failed_cleanup_pending",
            "path": str(self.root / "partial-model-stage"),
            "parent": str(self.root),
            "root_identity": {"st_dev": 7, "st_ino": 11},
        }
        stage_error = generation.model_snapshot.ModelStageCreationError(
            ValueError("copy failed"),
            OSError("initial cleanup failed"),
            partial_stage,
        )
        with self.assertRaisesRegex(RuntimeError, "model-stage cleanup also failed"):
            self._run(
                events,
                stage_error=stage_error,
                cleanup_error=OSError("retry cleanup failed"),
            )

        failure = json.loads((self.output / "failure.json").read_text())
        self.assertEqual(failure["model_stage"], partial_stage)
        evidence = json.loads(
            (self.output / "model_stage_evidence.json").read_text()
        )
        self.assertEqual(evidence["stage"], partial_stage)
        self.assertEqual(evidence["status"], "cleanup_failed_terminal")
        self.assertEqual(
            evidence["cleanup_failure"]["exception_message"],
            "retry cleanup failed",
        )

    def test_existing_output_is_terminal_before_runtime_load(self):
        self.output.mkdir()
        with mock.patch.object(
            generation,
            "_load_verified_runtime_components",
            side_effect=AssertionError("runtime loaded"),
        ) as loader:
            with self.assertRaisesRegex(FileExistsError, "resume/retry"):
                generation._run_authorized_engineering_generation(self.validated)
        loader.assert_not_called()

    def test_concurrent_output_claim_does_not_write_into_foreign_tree(self):
        real_mkdir = generation.os.mkdir

        def competing_mkdir(path, mode=0o777, *, dir_fd=None):
            if Path(path) == self.output:
                real_mkdir(path, mode=mode, dir_fd=dir_fd)
                (self.output / "foreign.txt").write_text(
                    "foreign claimant\n", encoding="utf-8"
                )
                raise FileExistsError("claim lost")
            return real_mkdir(path, mode=mode, dir_fd=dir_fd)

        with mock.patch.object(generation.os, "mkdir", side_effect=competing_mkdir):
            with self.assertRaisesRegex(FileExistsError, "resume/retry"):
                generation._run_authorized_engineering_generation(self.validated)
        self.assertTrue((self.output / "foreign.txt").is_file())
        self.assertFalse((self.output / "attempt.json").exists())
        self.assertFalse((self.output / "failure.json").exists())

    def test_runtime_warning_log_and_stderr_are_captured_on_failure(self):
        def noisy_runtime_load(_validated):
            warnings.warn("captured warning", RuntimeWarning)
            logging.getLogger("adaptive-oracle-test").warning("captured log")
            print("warning: captured stderr", file=sys.stderr)
            raise RuntimeError("stop after capture")

        with mock.patch.object(
            generation,
            "_load_verified_runtime_components",
            side_effect=noisy_runtime_load,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after capture"):
                generation._run_authorized_engineering_generation(self.validated)
        failure = json.loads((self.output / "failure.json").read_text())
        evidence = failure["runtime_evidence"]
        self.assertEqual(evidence["warnings"], {
            "count": 3,
            "python_warning_count": 1,
            "logging_warning_or_higher_count": 1,
            "stderr_warning_line_count": 1,
        })


class GeneratorMechanismHelperTest(unittest.TestCase):
    def test_fresh_scheduler_config_drift_fails_closed(self):
        class EulerDiscreteScheduler:
            @classmethod
            def from_config(cls, config):
                instance = cls()
                instance.config = {**config, "prediction_type": "v_prediction"}
                return instance

        runtime = SimpleNamespace(Scheduler=EulerDiscreteScheduler)
        with mock.patch.object(
            generation.contract,
            "SCHEDULER_CLASS",
            "EulerDiscreteScheduler",
        ):
            with self.assertRaisesRegex(RuntimeError, "config drifted|prediction_type"):
                generation._fresh_scheduler(
                    runtime, {"prediction_type": "epsilon"}
                )

    def test_physical_noop_constructs_no_provider_or_renderer(self):
        with (EVAL_PIPELINE / "prompts" / "adaptive_oracle_engineering.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        manifest = json.loads(
            (
                EVAL_PIPELINE
                / "prompts"
                / "adaptive_oracle_prompt_manifest_v1.json"
            ).read_text(encoding="utf-8")
        )
        design = contract.build_engineering_design(rows, manifest)
        task = next(
            value
            for value in design["tasks"]
            if value["action"]["physical_no_op"]
        )

        class Runtime:
            @staticmethod
            def BasisProvider(*_args, **_kwargs):
                raise AssertionError("P0 constructed a provider")

            @staticmethod
            def Renderer(*_args, **_kwargs):
                raise AssertionError("P0 constructed a renderer")

        provider, renderer = generation._action_runtime(
            task, SimpleNamespace(unet=_FakeUNet()), Runtime
        )
        self.assertIsNone(provider)
        self.assertIsNone(renderer)


if __name__ == "__main__":
    unittest.main()
