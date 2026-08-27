import csv
from contextlib import redirect_stderr
import errno
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

    def test_stderr_writer_close_error_does_not_close_a_reused_descriptor(self):
        capture = generation._RuntimeEvidenceCapture()
        real_pipe = generation.os.pipe
        real_close = generation.os.close
        write_descriptor = -1
        reused_descriptor = -1
        reported = False

        def record_pipe():
            nonlocal write_descriptor
            read_descriptor, write_descriptor = real_pipe()
            return read_descriptor, write_descriptor

        def close_then_reuse(descriptor):
            nonlocal reported, reused_descriptor
            if descriptor == write_descriptor and not reported:
                reported = True
                real_close(descriptor)
                reused_descriptor = os.open("/dev/null", os.O_RDONLY)
                self.assertEqual(reused_descriptor, descriptor)
                raise OSError(errno.EIO, "simulated stderr writer close error")
            return real_close(descriptor)

        try:
            with (
                mock.patch.object(
                    generation.os,
                    "pipe",
                    side_effect=record_pipe,
                ),
                mock.patch.object(
                    generation.os,
                    "close",
                    side_effect=close_then_reuse,
                ),
                self.assertRaisesRegex(OSError, "stderr writer close error"),
            ):
                capture._start_stderr_capture()
            self.assertTrue(reported)
            os.fstat(reused_descriptor)
            self.assertIsNone(capture._stderr_saved_fd)
            self.assertIsNone(capture._stderr_read_fd)
            self.assertIsNone(capture._stderr_stream)
            self.assertIsNone(capture._stderr_thread)
        finally:
            if reused_descriptor >= 0:
                os.close(reused_descriptor)

    def test_fdopen_return_cancellation_keeps_stream_guard_single_owned(self):
        capture = generation._RuntimeEvidenceCapture()
        real_fdopen = generation.os.fdopen
        stream_descriptor = -1
        sentinel_descriptor = -1

        def fdopen_then_cancel(descriptor, *args, **kwargs):
            nonlocal stream_descriptor, sentinel_descriptor
            stream_descriptor = descriptor
            stream = real_fdopen(descriptor, *args, **kwargs)
            stream.close()
            sentinel_descriptor = os.open("/dev/null", os.O_RDONLY)
            raise KeyboardInterrupt("simulated fdopen return cancellation")

        try:
            with (
                mock.patch.object(
                    generation.os,
                    "fdopen",
                    side_effect=fdopen_then_cancel,
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "fdopen return"),
            ):
                capture._start_stderr_capture()

            self.assertGreaterEqual(stream_descriptor, 0)
            self.assertGreaterEqual(sentinel_descriptor, 0)
            self.assertNotEqual(stream_descriptor, sentinel_descriptor)
            with self.assertRaises(OSError) as closed:
                os.fstat(stream_descriptor)
            self.assertEqual(closed.exception.errno, errno.EBADF)
            os.fstat(sentinel_descriptor)
            self.assertIsNone(capture._stderr_stream)
            self.assertIsNone(capture._stderr_stream_fd)
        finally:
            if sentinel_descriptor >= 0:
                os.close(sentinel_descriptor)

    def test_startup_cleanup_cancellation_does_not_skip_fd_and_thread_cleanup(self):
        class CloseCancellingStream:
            def __init__(self, stream):
                self._stream = stream

            def close(self):
                self._stream.close()
                raise KeyboardInterrupt("simulated stream cleanup cancellation")

            def __getattr__(self, name):
                return getattr(self._stream, name)

        class FailingRedirect:
            def __enter__(self):
                raise RuntimeError("simulated redirect startup failure")

            def __exit__(self, *_args):
                return False

        capture = generation._RuntimeEvidenceCapture()
        before = self._fd_identity(2)
        real_fdopen = generation.os.fdopen
        stream_descriptors = []

        def wrap_stream(descriptor, *args, **kwargs):
            stream_descriptors.append(descriptor)
            return CloseCancellingStream(real_fdopen(descriptor, *args, **kwargs))

        with (
            mock.patch.object(
                generation.os,
                "fdopen",
                side_effect=wrap_stream,
            ),
            mock.patch.object(
                generation,
                "redirect_stderr",
                return_value=FailingRedirect(),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "redirect startup failure",
            ) as raised,
        ):
            capture._start_stderr_capture()

        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertEqual(self._fd_identity(2), before)
        self.assertEqual(len(stream_descriptors), 1)
        with self.assertRaises(OSError) as closed:
            os.fstat(stream_descriptors[0])
        self.assertEqual(closed.exception.errno, errno.EBADF)
        self.assertIsNone(capture._stderr_stream)
        self.assertIsNone(capture._stderr_stream_fd)
        self.assertIsNone(capture._stderr_read_fd)
        self.assertIsNone(capture._stderr_thread)

    def test_thread_start_failure_closes_pipe_and_restores_enter_state(self):
        capture = generation._RuntimeEvidenceCapture()
        before_fd2 = self._fd_identity(2)
        filters_before = warnings.filters
        pipe_descriptors = []
        real_pipe = generation.os.pipe

        def record_pipe():
            descriptors = real_pipe()
            pipe_descriptors.extend(descriptors)
            return descriptors

        with (
            mock.patch.object(generation.os, "pipe", side_effect=record_pipe),
            mock.patch.object(
                generation.threading.Thread,
                "start",
                side_effect=RuntimeError("simulated thread start failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "thread start failure"),
        ):
            capture.__enter__()

        self.assertEqual(self._fd_identity(2), before_fd2)
        self.assertIs(warnings.filters, filters_before)
        self.assertEqual(len(pipe_descriptors), 2)
        for descriptor in pipe_descriptors:
            with self.assertRaises(OSError) as closed:
                os.fstat(descriptor)
            self.assertEqual(closed.exception.errno, errno.EBADF)
        self.assertIsNone(capture._stderr_saved_fd)
        self.assertIsNone(capture._stderr_read_fd)
        self.assertIsNone(capture._stderr_thread)
        self.assertIsNone(capture._stderr_handoff)

    def test_ambiguous_thread_start_never_transfers_the_read_descriptor(self):
        capture = generation._RuntimeEvidenceCapture()
        before_fd2 = self._fd_identity(2)
        filters_before = warnings.filters
        real_pipe = generation.os.pipe
        real_start = generation.threading.Thread.start
        release_delayed_reader = generation.threading.Event()
        pipe_descriptors = []
        delayed_workers = []

        def record_pipe():
            descriptors = real_pipe()
            pipe_descriptors.extend(descriptors)
            return descriptors

        def launch_without_publishing_start(thread):
            target = thread._target
            args = thread._args
            kwargs = thread._kwargs

            def delayed_target():
                release_delayed_reader.wait()
                target(*args, **kwargs)

            worker = generation.threading.Thread(
                target=delayed_target,
                name="delayed-stderr-reader-fixture",
                daemon=False,
            )
            real_start(worker)
            delayed_workers.append(worker)
            raise KeyboardInterrupt("simulated ambiguous thread start")

        sentinels = []
        try:
            with (
                mock.patch.object(generation.os, "pipe", side_effect=record_pipe),
                mock.patch.object(
                    generation.threading.Thread,
                    "start",
                    autospec=True,
                    side_effect=launch_without_publishing_start,
                ),
                self.assertRaisesRegex(
                    KeyboardInterrupt,
                    "ambiguous thread start",
                ) as raised,
            ):
                capture.__enter__()

            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertEqual(len(pipe_descriptors), 2)
            read_descriptor = pipe_descriptors[0]
            while read_descriptor not in sentinels:
                sentinels.append(os.open("/dev/null", os.O_RDONLY))
            release_delayed_reader.set()
            for worker in delayed_workers:
                worker.join()
            os.fstat(read_descriptor)
            self.assertEqual(self._fd_identity(2), before_fd2)
            self.assertIs(warnings.filters, filters_before)
            self.assertIsNone(capture._stderr_read_fd)
            self.assertIsNone(capture._stderr_thread)
            self.assertIsNone(capture._stderr_handoff)
        finally:
            release_delayed_reader.set()
            for worker in delayed_workers:
                worker.join()
            for descriptor in sentinels:
                os.close(descriptor)

    def test_enter_rollback_survives_join_cancellation_and_restores_globals(self):
        capture = generation._RuntimeEvidenceCapture()
        root_logger = logging.getLogger()
        before_level = root_logger.level
        before_fd2 = self._fd_identity(2)
        filters_before = warnings.filters
        real_add_handler = root_logger.addHandler
        real_join = generation.threading.Thread.join
        join_cancelled = False

        def add_then_fail(handler):
            real_add_handler(handler)
            raise RuntimeError("simulated logger handoff failure")

        def cancel_first_capture_join(thread, *args, **kwargs):
            nonlocal join_cancelled
            if thread.name == "adaptive-oracle-stderr-capture" and not join_cancelled:
                join_cancelled = True
                raise KeyboardInterrupt("simulated enter rollback join cancellation")
            return real_join(thread, *args, **kwargs)

        with (
            mock.patch.object(
                root_logger,
                "addHandler",
                side_effect=add_then_fail,
            ),
            mock.patch.object(
                generation.threading.Thread,
                "join",
                autospec=True,
                side_effect=cancel_first_capture_join,
            ),
            self.assertRaisesRegex(RuntimeError, "logger handoff failure") as raised,
        ):
            capture.__enter__()

        self.assertTrue(join_cancelled)
        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertEqual(root_logger.level, before_level)
        self.assertNotIn(capture._log_handler, root_logger.handlers)
        self.assertEqual(self._fd_identity(2), before_fd2)
        self.assertIs(warnings.filters, filters_before)
        self.assertFalse(capture._entered)
        self.assertIsNone(capture._stderr_saved_fd)
        self.assertIsNone(capture._stderr_stream_fd)
        self.assertIsNone(capture._stderr_thread)
        self.assertIsNone(capture._stderr_handoff)

    def test_stop_join_cancellation_invalidates_state_before_fd_reuse(self):
        capture = generation._RuntimeEvidenceCapture()
        capture.__enter__()
        thread = capture._stderr_thread
        self.assertIsNotNone(thread)
        owned_descriptors = {
            capture._stderr_saved_fd,
            capture._stderr_stream_fd,
            capture._stderr_read_fd,
        }
        real_join = thread.join
        join_cancelled = False

        def cancel_once(*args, **kwargs):
            nonlocal join_cancelled
            if not join_cancelled:
                join_cancelled = True
                raise KeyboardInterrupt("simulated stop join cancellation")
            return real_join(*args, **kwargs)

        with (
            mock.patch.object(thread, "join", side_effect=cancel_once),
            self.assertRaisesRegex(
                RuntimeError,
                "runtime evidence capture finalization failed",
            ) as raised,
        ):
            capture.__exit__(None, None, None)

        self.assertTrue(join_cancelled)
        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertFalse(capture._entered)
        self.assertIsNone(capture._stderr_saved_fd)
        self.assertIsNone(capture._stderr_stream_fd)
        self.assertIsNone(capture._stderr_read_fd)
        self.assertIsNone(capture._stderr_thread)
        self.assertIsNone(capture._stderr_handoff)

        sentinel = os.open("/dev/null", os.O_RDONLY)
        try:
            self.assertIn(sentinel, owned_descriptors)
            capture._stop_stderr_capture()
            os.fstat(sentinel)
        finally:
            os.close(sentinel)

    def test_two_join_cancellations_retain_thread_ownership_for_retry(self):
        capture = generation._RuntimeEvidenceCapture()
        capture.__enter__()
        thread = capture._stderr_thread
        handoff = capture._stderr_handoff
        read_descriptor = capture._stderr_read_fd
        self.assertIsNotNone(thread)
        self.assertIsNotNone(handoff)
        self.assertIsNotNone(read_descriptor)
        held_writer = os.dup(2)
        sentinels = []
        join_calls = 0

        def cancel_join(*_args, **_kwargs):
            nonlocal join_calls
            join_calls += 1
            raise KeyboardInterrupt("simulated repeated join cancellation")

        try:
            with (
                mock.patch.object(thread, "join", side_effect=cancel_join),
                self.assertRaisesRegex(
                    RuntimeError,
                    "runtime evidence capture finalization failed",
                ) as raised,
            ):
                capture.__exit__(None, None, None)

            self.assertEqual(join_calls, 2)
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertTrue(capture._entered)
            self.assertTrue(capture._cleanup_pending)
            self.assertIs(capture._stderr_thread, thread)
            self.assertIs(capture._stderr_handoff, handoff)
            self.assertEqual(capture._stderr_read_fd, read_descriptor)
            with self.assertRaisesRegex(RuntimeError, "cannot be read"):
                capture.record()

            os.close(held_writer)
            held_writer = -1
            thread.join()
            while read_descriptor not in sentinels:
                sentinels.append(os.open("/dev/null", os.O_RDONLY))
            capture.__exit__(None, None, None)
            os.fstat(read_descriptor)
            self.assertFalse(capture._entered)
            self.assertFalse(capture._cleanup_pending)
            self.assertIsNone(capture._stderr_thread)
            self.assertIsNone(capture._stderr_handoff)
        finally:
            if held_writer >= 0:
                os.close(held_writer)
            if thread is not None and thread.is_alive():
                thread.join()
            if capture._cleanup_pending:
                capture.__exit__(None, None, None)
            for descriptor in sentinels:
                os.close(descriptor)

    def test_python_stderr_restore_failure_retains_open_retry_state(self):
        class InterruptingSys:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self.restore_attempts = 0

            @property
            def stderr(self):
                return self._wrapped.stderr

            @stderr.setter
            def stderr(self, _value):
                self.restore_attempts += 1
                raise KeyboardInterrupt("simulated persistent sys.stderr cancellation")

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        capture = generation._RuntimeEvidenceCapture()
        original_stderr = sys.stderr
        capture.__enter__()
        captured_stream = capture._stderr_stream
        proxy = InterruptingSys(generation.sys)
        try:
            with (
                mock.patch.object(generation, "sys", proxy),
                mock.patch.object(
                    capture,
                    "_restore_python_stderr_direct",
                    return_value=(
                        [KeyboardInterrupt("simulated direct restore cancellation")],
                        False,
                    ),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "runtime evidence capture finalization failed",
                ) as raised,
            ):
                capture.__exit__(None, None, None)

            self.assertEqual(proxy.restore_attempts, 3)
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertIs(sys.stderr, captured_stream)
            self.assertFalse(captured_stream.closed)
            self.assertTrue(capture._entered)
            self.assertTrue(capture._cleanup_pending)
            self.assertTrue(capture._stderr_python_restore_pending)
            self.assertIsNotNone(capture._stderr_context)
            with self.assertRaisesRegex(RuntimeError, "cannot be read"):
                capture.record()

            capture.__exit__(None, None, None)
            self.assertIs(sys.stderr, original_stderr)
            self.assertTrue(captured_stream.closed)
            self.assertFalse(capture._entered)
            self.assertFalse(capture._cleanup_pending)
            self.assertFalse(capture._stderr_python_restore_pending)
        finally:
            if sys.stderr is not original_stderr:
                sys.stderr = original_stderr
            if capture._cleanup_pending:
                capture.__exit__(None, None, None)

    def test_direct_python_stderr_fallback_releases_after_normal_exhaustion(self):
        class InterruptingSys:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self.restore_attempts = 0

            @property
            def stderr(self):
                return self._wrapped.stderr

            @stderr.setter
            def stderr(self, _value):
                self.restore_attempts += 1
                raise KeyboardInterrupt("simulated persistent ordinary restore failure")

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        capture = generation._RuntimeEvidenceCapture()
        original_stderr = sys.stderr
        capture.__enter__()
        captured_stream = capture._stderr_stream
        thread = capture._stderr_thread
        proxy = InterruptingSys(generation.sys)
        try:
            with (
                mock.patch.object(generation, "sys", proxy),
                self.assertRaisesRegex(
                    RuntimeError,
                    "runtime evidence capture finalization failed",
                ) as raised,
            ):
                capture.__exit__(None, None, None)

            self.assertEqual(proxy.restore_attempts, 3)
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertIs(sys.stderr, original_stderr)
            self.assertTrue(captured_stream.closed)
            self.assertTrue(thread.daemon)
            self.assertFalse(thread.is_alive())
            self.assertFalse(capture._entered)
            self.assertFalse(capture._cleanup_pending)
            self.assertIsNone(capture._stderr_thread)
            capture.record()
        finally:
            if sys.stderr is not original_stderr:
                sys.stderr = original_stderr
            if capture._cleanup_pending:
                capture.__exit__(None, None, None)

    def test_python_stderr_post_restore_cancellation_confirms_safe_state(self):
        class PostRestoreInterruptingSys:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self.restore_attempts = 0

            @property
            def stderr(self):
                return self._wrapped.stderr

            @stderr.setter
            def stderr(self, value):
                self.restore_attempts += 1
                self._wrapped.stderr = value
                raise KeyboardInterrupt("simulated post-restore cancellation")

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        capture = generation._RuntimeEvidenceCapture()
        original_stderr = sys.stderr
        capture.__enter__()
        captured_stream = capture._stderr_stream
        proxy = PostRestoreInterruptingSys(generation.sys)
        try:
            with (
                mock.patch.object(generation, "sys", proxy),
                self.assertRaisesRegex(
                    RuntimeError,
                    "runtime evidence capture finalization failed",
                ) as raised,
            ):
                capture.__exit__(None, None, None)

            self.assertEqual(proxy.restore_attempts, 1)
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertIs(sys.stderr, original_stderr)
            self.assertTrue(captured_stream.closed)
            self.assertFalse(capture._entered)
            self.assertFalse(capture._cleanup_pending)
            self.assertFalse(capture._stderr_python_restore_pending)
            capture.record()
        finally:
            if sys.stderr is not original_stderr:
                sys.stderr = original_stderr
            if capture._cleanup_pending:
                capture.__exit__(None, None, None)

    def test_stderr_restore_post_dup2_cancellation_keeps_fd2_restored(self):
        capture = generation._RuntimeEvidenceCapture()
        before_fd2 = self._fd_identity(2)
        capture.__enter__()
        saved_descriptor = capture._stderr_saved_fd
        self.assertIsNotNone(saved_descriptor)
        real_dup2 = generation.os.dup2
        cancelled = False

        def restore_then_cancel(source, destination, *args, **kwargs):
            nonlocal cancelled
            result = real_dup2(source, destination, *args, **kwargs)
            if source == saved_descriptor and destination == 2 and not cancelled:
                cancelled = True
                raise KeyboardInterrupt("simulated post-dup2 restore cancellation")
            return result

        with (
            mock.patch.object(
                generation.os,
                "dup2",
                side_effect=restore_then_cancel,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "runtime evidence capture finalization failed",
            ) as raised,
        ):
            capture.__exit__(None, None, None)

        self.assertTrue(cancelled)
        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertEqual(self._fd_identity(2), before_fd2)
        self.assertFalse(capture._entered)
        self.assertFalse(capture._cleanup_pending)

    def test_stderr_restore_exhaustion_retains_saved_fd_for_explicit_retry(self):
        capture = generation._RuntimeEvidenceCapture()
        before_fd2 = self._fd_identity(2)
        capture.__enter__()

        with (
            mock.patch.object(
                generation.os,
                "dup2",
                side_effect=KeyboardInterrupt(
                    "simulated persistent stderr restore cancellation"
                ),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "runtime evidence capture finalization failed",
            ) as raised,
        ):
            capture.__exit__(None, None, None)

        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertTrue(capture._entered)
        self.assertTrue(capture._cleanup_pending)
        self.assertIsNotNone(capture._stderr_saved_fd)
        os.fstat(capture._stderr_saved_fd)

        capture.__exit__(None, None, None)
        self.assertEqual(self._fd_identity(2), before_fd2)
        self.assertFalse(capture._entered)
        self.assertFalse(capture._cleanup_pending)
        self.assertIsNone(capture._stderr_saved_fd)

    def test_owned_stderr_fds_retain_pre_close_cancellation_for_retry(self):
        for descriptor_attribute in (
            "_stderr_saved_fd",
            "_stderr_stream_fd",
        ):
            with self.subTest(descriptor_attribute=descriptor_attribute):
                capture = generation._RuntimeEvidenceCapture()
                before_fd2 = self._fd_identity(2)
                capture.__enter__()
                descriptor = getattr(capture, descriptor_attribute)
                self.assertIsNotNone(descriptor)
                real_close = generation.os.close
                cancelled = False

                def cancel_before_close(candidate):
                    nonlocal cancelled
                    if candidate == descriptor and not cancelled:
                        cancelled = True
                        raise KeyboardInterrupt(
                            f"simulated {descriptor_attribute} pre-close cancellation"
                        )
                    return real_close(candidate)

                try:
                    with (
                        mock.patch.object(
                            generation.os,
                            "close",
                            side_effect=cancel_before_close,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError,
                            "runtime evidence capture finalization failed",
                        ) as raised,
                    ):
                        capture.__exit__(None, None, None)

                    self.assertTrue(cancelled)
                    self.assertTrue(
                        generation._exception_contains_cancellation(raised.exception)
                    )
                    self.assertTrue(capture._entered)
                    self.assertTrue(capture._cleanup_pending)
                    self.assertEqual(getattr(capture, descriptor_attribute), descriptor)
                    os.fstat(descriptor)

                    capture.__exit__(None, None, None)
                    self.assertEqual(self._fd_identity(2), before_fd2)
                    self.assertFalse(capture._entered)
                    self.assertFalse(capture._cleanup_pending)
                    self.assertIsNone(getattr(capture, descriptor_attribute))
                    with self.assertRaises(OSError) as closed:
                        os.fstat(descriptor)
                    self.assertEqual(closed.exception.errno, errno.EBADF)
                finally:
                    if capture._cleanup_pending:
                        capture.__exit__(None, None, None)

    def test_owned_stderr_fds_release_post_close_reused_numbers(self):
        for descriptor_attribute in (
            "_stderr_saved_fd",
            "_stderr_stream_fd",
        ):
            with self.subTest(descriptor_attribute=descriptor_attribute):
                capture = generation._RuntimeEvidenceCapture()
                before_fd2 = self._fd_identity(2)
                capture.__enter__()
                descriptor = getattr(capture, descriptor_attribute)
                self.assertIsNotNone(descriptor)
                real_close = generation.os.close
                reused_descriptors = []
                cancelled = False

                def close_reuse_then_cancel(candidate):
                    nonlocal cancelled
                    if candidate == descriptor and not cancelled:
                        cancelled = True
                        real_close(candidate)
                        while descriptor not in reused_descriptors:
                            reused_descriptors.append(
                                os.open("/dev/null", os.O_RDONLY)
                            )
                        raise KeyboardInterrupt(
                            f"simulated {descriptor_attribute} post-close cancellation"
                        )
                    return real_close(candidate)

                try:
                    with (
                        mock.patch.object(
                            generation.os,
                            "close",
                            side_effect=close_reuse_then_cancel,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError,
                            "runtime evidence capture finalization failed",
                        ) as raised,
                    ):
                        capture.__exit__(None, None, None)

                    self.assertTrue(cancelled)
                    self.assertTrue(
                        generation._exception_contains_cancellation(raised.exception)
                    )
                    self.assertEqual(self._fd_identity(2), before_fd2)
                    self.assertFalse(capture._entered)
                    self.assertFalse(capture._cleanup_pending)
                    self.assertIsNone(getattr(capture, descriptor_attribute))
                    os.fstat(descriptor)
                    capture._stop_stderr_capture()
                    os.fstat(descriptor)
                finally:
                    if capture._cleanup_pending:
                        capture.__exit__(None, None, None)
                    for reused_descriptor in reused_descriptors:
                        real_close(reused_descriptor)

    def test_stderr_reader_pre_close_cancellation_is_reconciled(self):
        capture = generation._RuntimeEvidenceCapture()
        capture.__enter__()
        read_descriptor = capture._stderr_read_fd
        self.assertIsNotNone(read_descriptor)
        real_close = generation.os.close
        cancelled = False

        def cancel_before_reader_close(descriptor):
            nonlocal cancelled
            if descriptor == read_descriptor and not cancelled:
                cancelled = True
                raise KeyboardInterrupt("simulated reader pre-close cancellation")
            return real_close(descriptor)

        with (
            mock.patch.object(
                generation.os,
                "close",
                side_effect=cancel_before_reader_close,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "runtime evidence capture finalization failed",
            ) as raised,
        ):
            capture.__exit__(None, None, None)

        self.assertTrue(cancelled)
        self.assertTrue(generation._exception_contains_cancellation(raised.exception))
        self.assertFalse(capture._entered)
        self.assertFalse(capture._cleanup_pending)
        self.assertIsNone(capture._stderr_read_fd)
        with self.assertRaises(OSError) as closed:
            os.fstat(read_descriptor)
        self.assertEqual(closed.exception.errno, errno.EBADF)

    def test_saved_fd_retry_never_restores_from_a_reused_descriptor(self):
        capture = generation._RuntimeEvidenceCapture()
        before_fd2 = self._fd_identity(2)
        capture.__enter__()
        saved_descriptor = capture._stderr_saved_fd
        self.assertIsNotNone(saved_descriptor)
        real_binding = capture._stderr_descriptor_binding
        real_close = generation.os.close
        saved_binding_calls = 0
        reused_descriptors = []
        close_cancelled = False

        def interrupt_third_saved_binding(descriptor, expected_identity):
            nonlocal saved_binding_calls
            if descriptor == saved_descriptor:
                saved_binding_calls += 1
                if saved_binding_calls == 3:
                    return (
                        "unknown",
                        [KeyboardInterrupt("simulated close reconciliation cancellation")],
                    )
            return real_binding(descriptor, expected_identity)

        def close_saved_reuse_then_cancel(descriptor):
            nonlocal close_cancelled
            if descriptor == saved_descriptor and not close_cancelled:
                close_cancelled = True
                real_close(descriptor)
                while saved_descriptor not in reused_descriptors:
                    reused_descriptors.append(os.open("/dev/null", os.O_RDONLY))
                raise KeyboardInterrupt("simulated saved fd post-close cancellation")
            return real_close(descriptor)

        try:
            with (
                mock.patch.object(
                    capture,
                    "_stderr_descriptor_binding",
                    side_effect=interrupt_third_saved_binding,
                ),
                mock.patch.object(
                    generation.os,
                    "close",
                    side_effect=close_saved_reuse_then_cancel,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "runtime evidence capture finalization failed",
                ) as raised,
            ):
                capture.__exit__(None, None, None)

            self.assertTrue(close_cancelled)
            self.assertEqual(saved_binding_calls, 3)
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertFalse(capture._stderr_fd_restore_pending)
            self.assertTrue(capture._cleanup_pending)
            self.assertEqual(capture._stderr_saved_fd, saved_descriptor)
            self.assertEqual(self._fd_identity(2), before_fd2)
            os.fstat(saved_descriptor)

            with self.assertRaisesRegex(
                RuntimeError,
                "runtime evidence capture finalization failed",
            ):
                capture.__exit__(None, None, None)
            self.assertFalse(capture._entered)
            self.assertFalse(capture._cleanup_pending)
            self.assertIsNone(capture._stderr_saved_fd)
            self.assertEqual(self._fd_identity(2), before_fd2)
            os.fstat(saved_descriptor)
            capture._stop_stderr_capture()
            os.fstat(saved_descriptor)
        finally:
            if capture._cleanup_pending:
                capture.__exit__(None, None, None)
            for reused_descriptor in reused_descriptors:
                real_close(reused_descriptor)

    def test_stderr_reader_post_close_reuse_is_not_closed_by_joiner(self):
        capture = generation._RuntimeEvidenceCapture()
        capture.__enter__()
        read_descriptor = capture._stderr_read_fd
        self.assertIsNotNone(read_descriptor)
        real_close = generation.os.close
        reused_descriptors = []
        cancelled = False

        def close_reader_reuse_then_cancel(descriptor):
            nonlocal cancelled
            if descriptor == read_descriptor and not cancelled:
                cancelled = True
                real_close(descriptor)
                while read_descriptor not in reused_descriptors:
                    reused_descriptors.append(os.open("/dev/null", os.O_RDONLY))
                raise KeyboardInterrupt("simulated reader post-close cancellation")
            return real_close(descriptor)

        try:
            with (
                mock.patch.object(
                    generation.os,
                    "close",
                    side_effect=close_reader_reuse_then_cancel,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "runtime evidence capture finalization failed",
                ) as raised,
            ):
                capture.__exit__(None, None, None)

            self.assertTrue(cancelled)
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertFalse(capture._entered)
            self.assertFalse(capture._cleanup_pending)
            self.assertIsNone(capture._stderr_read_fd)
            os.fstat(read_descriptor)
            capture._stop_stderr_capture()
            os.fstat(read_descriptor)
        finally:
            if capture._cleanup_pending:
                capture.__exit__(None, None, None)
            for reused_descriptor in reused_descriptors:
                real_close(reused_descriptor)

    def test_logger_handler_restore_retries_pre_remove_cancellation(self):
        capture = generation._RuntimeEvidenceCapture()
        root_logger = logging.getLogger()
        before_level = root_logger.level
        capture.__enter__()
        real_remove = root_logger.removeHandler
        cancelled = False

        def cancel_then_remove(handler):
            nonlocal cancelled
            if handler is capture._log_handler and not cancelled:
                cancelled = True
                raise KeyboardInterrupt("simulated pre-remove cancellation")
            return real_remove(handler)

        with (
            mock.patch.object(
                root_logger,
                "removeHandler",
                side_effect=cancel_then_remove,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "runtime evidence capture finalization failed",
            ) as raised,
        ):
            capture.__exit__(None, None, None)

        self.assertTrue(cancelled)
        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertNotIn(capture._log_handler, root_logger.handlers)
        self.assertEqual(root_logger.level, before_level)
        self.assertFalse(capture._entered)
        self.assertFalse(capture._cleanup_pending)

    def test_logger_level_restore_retries_pre_setlevel_cancellation(self):
        capture = generation._RuntimeEvidenceCapture()
        root_logger = logging.getLogger()
        before_level = root_logger.level
        capture.__enter__()
        real_set_level = root_logger.setLevel
        cancelled = False

        def cancel_then_set(level):
            nonlocal cancelled
            if level == before_level and not cancelled:
                cancelled = True
                raise KeyboardInterrupt("simulated pre-setLevel cancellation")
            return real_set_level(level)

        with (
            mock.patch.object(
                root_logger,
                "setLevel",
                side_effect=cancel_then_set,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "runtime evidence capture finalization failed",
            ) as raised,
        ):
            capture.__exit__(None, None, None)

        self.assertTrue(cancelled)
        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertNotIn(capture._log_handler, root_logger.handlers)
        self.assertEqual(root_logger.level, before_level)
        self.assertFalse(capture._entered)
        self.assertFalse(capture._cleanup_pending)

    def test_warning_restore_retries_pre_exit_cancellation(self):
        capture = generation._RuntimeEvidenceCapture()
        filters_before = warnings.filters
        showwarning_before = warnings.showwarning
        showwarnmsg_before = warnings._showwarnmsg_impl
        capture.__enter__()
        real_filters_mutated = warnings._filters_mutated
        cancelled = False

        def cancel_then_mutate():
            nonlocal cancelled
            if not cancelled:
                cancelled = True
                raise KeyboardInterrupt("simulated pre-warning-exit cancellation")
            return real_filters_mutated()

        with (
            mock.patch.object(
                warnings,
                "_filters_mutated",
                side_effect=cancel_then_mutate,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "runtime evidence capture finalization failed",
            ) as raised,
        ):
            capture.__exit__(None, None, None)

        self.assertTrue(cancelled)
        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertIs(warnings.filters, filters_before)
        self.assertIs(warnings.showwarning, showwarning_before)
        self.assertIs(warnings._showwarnmsg_impl, showwarnmsg_before)
        self.assertFalse(capture._entered)
        self.assertFalse(capture._cleanup_pending)

    def test_warning_enter_post_mutation_cancellation_restores_globals(self):
        capture = generation._RuntimeEvidenceCapture()
        filters_before = warnings.filters
        showwarning_before = warnings.showwarning
        showwarnmsg_before = warnings._showwarnmsg_impl
        context_type = type(warnings.catch_warnings())
        real_enter = context_type.__enter__
        cancelled = False

        def enter_then_cancel(context):
            nonlocal cancelled
            result = real_enter(context)
            if context is capture._warning_context and not cancelled:
                cancelled = True
                raise KeyboardInterrupt("simulated post-warning-enter cancellation")
            return result

        with (
            mock.patch.object(
                context_type,
                "__enter__",
                autospec=True,
                side_effect=enter_then_cancel,
            ),
            self.assertRaisesRegex(
                KeyboardInterrupt,
                "post-warning-enter cancellation",
            ) as raised,
        ):
            capture.__enter__()

        self.assertTrue(cancelled)
        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertIs(warnings.filters, filters_before)
        self.assertIs(warnings.showwarning, showwarning_before)
        self.assertIs(warnings._showwarnmsg_impl, showwarnmsg_before)
        self.assertFalse(capture._entered)
        self.assertFalse(capture._cleanup_pending)

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


class DescriptorBoundPublicationTest(unittest.TestCase):
    def test_fd_headroom_rejects_low_limit_before_publication(self):
        with (
            mock.patch.object(
                generation.resource,
                "getrlimit",
                return_value=(256, 256),
            ),
            mock.patch.object(
                generation,
                "_open_descriptor_count",
                return_value=10,
            ),
            self.assertRaisesRegex(RuntimeError, "free file descriptors"),
        ):
            generation._require_fd_headroom(300, "fixture workflow")

    def test_exception_notes_have_a_python_39_compatible_fallback(self):
        class LegacyError(RuntimeError):
            add_note = None

        error = LegacyError("primary")
        generation._add_exception_note(error, "secondary")
        self.assertEqual(error.__notes__, ["secondary"])

    def test_directory_chain_cleanup_preserves_primary_error_and_closes_all(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "present").mkdir()
            real_close = generation.os.close
            closed = []
            reported = False

            def close_then_report_once(descriptor):
                nonlocal reported
                real_close(descriptor)
                closed.append(descriptor)
                if not reported:
                    reported = True
                    raise OSError(errno.EIO, "simulated cleanup close failure")

            with (
                mock.patch.object(
                    generation.os,
                    "close",
                    side_effect=close_then_report_once,
                ),
                self.assertRaisesRegex(ValueError, "component 1") as raised,
            ):
                generation._open_pinned_directory_chain(
                    root,
                    ("present", "missing"),
                    "fixture chain",
                )

            self.assertTrue(reported)
            self.assertGreaterEqual(len(closed), 2)
            self.assertTrue(
                any("cleanup close failure" in note for note in raised.exception.__notes__)
            )

    def test_directory_chain_root_handoff_cancellation_closes_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_open = generation._open_pinned_directory
            opened = []

            def open_then_cancel(*args, **kwargs):
                descriptor = real_open(*args, **kwargs)
                opened.append(descriptor)
                raise KeyboardInterrupt("simulated root handoff cancellation")

            with (
                mock.patch.object(
                    generation,
                    "_open_pinned_directory",
                    side_effect=open_then_cancel,
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "root handoff"),
            ):
                generation._open_pinned_directory_chain(
                    root,
                    (),
                    "fixture chain",
                    descriptor_guard=[],
                )

            self.assertEqual(len(opened), 1)
            with self.assertRaises(OSError) as raised:
                os.fstat(opened[0])
            self.assertEqual(raised.exception.errno, errno.EBADF)

    def test_directory_chain_child_handoff_cancellation_closes_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "child").mkdir()
            real_open = generation._open_pinned_directory_at
            opened = []

            def open_then_cancel(*args, **kwargs):
                descriptor = real_open(*args, **kwargs)
                opened.append(descriptor)
                raise KeyboardInterrupt("simulated child handoff cancellation")

            with (
                mock.patch.object(
                    generation,
                    "_open_pinned_directory_at",
                    side_effect=open_then_cancel,
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "child handoff"),
            ):
                generation._open_pinned_directory_chain(
                    root,
                    ("child",),
                    "fixture chain",
                    descriptor_guard=[],
                )

            self.assertEqual(len(opened), 1)
            with self.assertRaises(OSError) as raised:
                os.fstat(opened[0])
            self.assertEqual(raised.exception.errno, errno.EBADF)

    def test_directory_chain_external_guard_owns_every_returned_descriptor(self):
        class InterruptingGuard(list):
            def extend(self, values):
                super().extend(values)
                raise KeyboardInterrupt("simulated chain return cancellation")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parent").mkdir()
            guard = InterruptingGuard()

            with self.assertRaisesRegex(KeyboardInterrupt, "chain return"):
                generation._open_pinned_directory_chain(
                    root,
                    ("parent",),
                    "fixture chain",
                    descriptor_guard=guard,
                )

            self.assertEqual(len(guard), 2)
            try:
                for descriptor in guard:
                    os.fstat(descriptor)
            finally:
                for descriptor in reversed(guard):
                    os.close(descriptor)

    def test_link_error_is_reconciled_when_held_inode_was_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            payload = b"committed-before-rpc-error\n"
            real_link = generation._link_descriptor

            def link_then_report_error(descriptor, directory_descriptor, name):
                real_link(descriptor, directory_descriptor, name)
                raise OSError(errno.EIO, "simulated NFS reply loss")

            with mock.patch.object(
                generation,
                "_link_descriptor",
                side_effect=link_then_report_error,
            ):
                identity = generation.atomic_create_bytes(target, payload)

            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(identity, generation._object_identity(target.stat()))

    def test_link_eexist_is_reconciled_when_nfs_already_committed_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            payload = b"committed-before-eexist\n"
            real_link = generation._link_descriptor

            def link_then_report_eexist(descriptor, directory_descriptor, name):
                real_link(descriptor, directory_descriptor, name)
                raise FileExistsError(errno.EEXIST, "simulated replayed link")

            with mock.patch.object(
                generation,
                "_link_descriptor",
                side_effect=link_then_report_eexist,
            ):
                identity = generation.atomic_create_bytes(target, payload)

            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(identity, generation._object_identity(target.stat()))

    def test_stale_enoent_after_ambiguous_link_retries_to_positive_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            payload = b"committed-before-stale-negative\n"
            real_link = generation._link_descriptor
            real_stat = generation.os.stat
            link_calls = 0
            stale_negative = True

            def link_then_report_error(descriptor, directory_descriptor, name):
                nonlocal link_calls
                link_calls += 1
                if link_calls == 1:
                    real_link(descriptor, directory_descriptor, name)
                    raise OSError(errno.EIO, "simulated NFS reply loss")
                return real_link(descriptor, directory_descriptor, name)

            def hide_first_target_lookup(path, *args, **kwargs):
                nonlocal stale_negative
                if (
                    stale_negative
                    and path == target.name
                    and kwargs.get("dir_fd") is not None
                    and kwargs.get("follow_symlinks") is False
                ):
                    stale_negative = False
                    raise FileNotFoundError(errno.ENOENT, "stale negative lookup")
                return real_stat(path, *args, **kwargs)

            with (
                mock.patch.object(
                    generation,
                    "_link_descriptor",
                    side_effect=link_then_report_error,
                ),
                mock.patch.object(
                    generation.os,
                    "stat",
                    side_effect=hide_first_target_lookup,
                ),
            ):
                identity = generation.atomic_create_bytes(target, payload)

            self.assertFalse(stale_negative)
            self.assertGreaterEqual(link_calls, 2)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(identity, generation._object_identity(target.stat()))

    def test_cancellation_during_link_reconciliation_is_deferred(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            real_link = generation._link_descriptor
            real_stat = generation.os.stat
            link_reported = False
            stat_cancelled = False

            def link_then_report_error(descriptor, directory_descriptor, name):
                nonlocal link_reported
                if not link_reported:
                    link_reported = True
                    real_link(descriptor, directory_descriptor, name)
                    raise OSError(errno.EIO, "simulated NFS reply loss")
                return real_link(descriptor, directory_descriptor, name)

            def cancel_first_target_lookup(path, *args, **kwargs):
                nonlocal stat_cancelled
                if (
                    not stat_cancelled
                    and path == target.name
                    and kwargs.get("dir_fd") is not None
                ):
                    stat_cancelled = True
                    raise KeyboardInterrupt("simulated reconciliation cancellation")
                return real_stat(path, *args, **kwargs)

            with (
                mock.patch.object(
                    generation,
                    "_link_descriptor",
                    side_effect=link_then_report_error,
                ),
                mock.patch.object(
                    generation.os,
                    "stat",
                    side_effect=cancel_first_target_lookup,
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "reconciliation"),
            ):
                generation.atomic_create_bytes(target, b"cancelled\n")

            self.assertTrue(link_reported)
            self.assertTrue(stat_cancelled)
            self.assertFalse(target.exists())

    def test_later_reconciliation_cancellation_is_retained_structurally(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            real_link = generation._link_descriptor
            real_stat = generation.os.stat
            linked = False
            cancelled = False

            def link_then_raise_runtime(descriptor, directory_descriptor, name):
                nonlocal linked
                real_link(descriptor, directory_descriptor, name)
                linked = True
                raise RuntimeError("simulated ordinary reconciliation error")

            def cancel_first_observation(path, *args, **kwargs):
                nonlocal cancelled
                if linked and not cancelled and path == target.name:
                    cancelled = True
                    raise KeyboardInterrupt("simulated later cancellation")
                return real_stat(path, *args, **kwargs)

            with (
                mock.patch.object(
                    generation,
                    "_link_descriptor",
                    side_effect=link_then_raise_runtime,
                ),
                mock.patch.object(
                    generation.os,
                    "stat",
                    side_effect=cancel_first_observation,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "ordinary reconciliation error",
                ) as raised,
            ):
                generation.atomic_create_bytes(target, b"cancelled\n")

            self.assertTrue(cancelled)
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertFalse(target.exists())

    def test_atomic_cleanup_cancellation_remains_visible_after_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            real_remove = generation._remove_matching_entry
            cancelled = False

            def remove_then_cancel(directory_descriptor, name, identity):
                nonlocal cancelled
                result = real_remove(directory_descriptor, name, identity)
                if name.startswith(".receipt.json.") and not cancelled:
                    cancelled = True
                    raise KeyboardInterrupt("simulated staging cleanup cancellation")
                return result

            with (
                mock.patch.object(
                    generation,
                    "_remove_matching_entry",
                    side_effect=remove_then_cancel,
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "staging cleanup"),
            ):
                generation.atomic_create_bytes(target, b"published\n")

            self.assertTrue(cancelled)
            self.assertEqual(target.read_bytes(), b"published\n")
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in target.parent.iterdir())
            )

    def test_terminal_commit_wins_over_reconciliation_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "target"
            staging_dir = root / "staging"
            target_dir.mkdir()
            staging_dir.mkdir()
            target_fd = generation._open_pinned_directory(target_dir, "target")
            staging_fd = generation._open_pinned_directory(staging_dir, "staging")
            real_link = generation._link_descriptor
            real_stat = generation.os.stat
            link_reported = False
            stat_cancelled = False
            prepared = generation._prepare_terminal_json(
                target_dir / "success.json",
                {"status": "complete"},
                directory_descriptor=target_fd,
                staging_directory_descriptor=staging_fd,
            )

            def link_then_report_error(descriptor, directory_descriptor, name):
                nonlocal link_reported
                if name == "success.json" and not link_reported:
                    link_reported = True
                    real_link(descriptor, directory_descriptor, name)
                    raise OSError(errno.EIO, "simulated NFS reply loss")
                return real_link(descriptor, directory_descriptor, name)

            def cancel_first_target_lookup(path, *args, **kwargs):
                nonlocal stat_cancelled
                if (
                    not stat_cancelled
                    and path == "success.json"
                    and kwargs.get("dir_fd") == target_fd
                ):
                    stat_cancelled = True
                    raise KeyboardInterrupt("simulated reconciliation cancellation")
                return real_stat(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_link_descriptor",
                        side_effect=link_then_report_error,
                    ),
                    mock.patch.object(
                        generation.os,
                        "stat",
                        side_effect=cancel_first_target_lookup,
                    ),
                    self.assertRaisesRegex(
                        KeyboardInterrupt,
                        "reconciliation cancellation",
                    ),
                ):
                    generation._commit_prepared_terminal(prepared)

                self.assertTrue(link_reported)
                self.assertTrue(stat_cancelled)
                self.assertTrue((target_dir / "success.json").is_file())
            finally:
                os.close(staging_fd)
                os.close(target_fd)

    def test_prepare_failure_retains_cleanup_cancellation_and_closes_fd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "target"
            staging_dir = root / "staging"
            target_dir.mkdir()
            staging_dir.mkdir()
            target_fd = generation._open_pinned_directory(target_dir, "target")
            staging_fd = generation._open_pinned_directory(staging_dir, "staging")
            real_open = generation._open_unique_regular_at
            opened = []

            def track_open(*args, **kwargs):
                descriptor = real_open(*args, **kwargs)
                opened.append(descriptor)
                return descriptor

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_open_unique_regular_at",
                        side_effect=track_open,
                    ),
                    mock.patch.object(
                        generation,
                        "_read_descriptor_bytes",
                        side_effect=RuntimeError("simulated prepare validation error"),
                    ),
                    mock.patch.object(
                        generation,
                        "_remove_matching_entry",
                        side_effect=KeyboardInterrupt(
                            "simulated prepare cleanup cancellation"
                        ),
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "prepare validation error",
                    ) as raised,
                ):
                    generation._prepare_terminal_json(
                        target_dir / "success.json",
                        {"status": "complete"},
                        directory_descriptor=target_fd,
                        staging_directory_descriptor=staging_fd,
                    )

                self.assertTrue(
                    generation._exception_contains_cancellation(raised.exception)
                )
                self.assertEqual(len(opened), 1)
                with self.assertRaises(OSError) as closed:
                    os.fstat(opened[0])
                self.assertEqual(closed.exception.errno, errno.EBADF)
            finally:
                os.close(staging_fd)
                os.close(target_fd)

    def test_commit_start_handoff_cancellation_discards_unpublished_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "target"
            staging_dir = root / "staging"
            target_dir.mkdir()
            staging_dir.mkdir()
            target_fd = generation._open_pinned_directory(target_dir, "target")
            staging_fd = generation._open_pinned_directory(staging_dir, "staging")
            prepared = generation._prepare_terminal_json(
                target_dir / "success.json",
                {"status": "complete"},
                directory_descriptor=target_fd,
                staging_directory_descriptor=staging_fd,
            )
            prepared_descriptor = prepared.descriptor
            real_mark = generation._mark_prepared_terminal_commit_started

            def mark_then_cancel(value):
                real_mark(value)
                raise KeyboardInterrupt("simulated commit-start cancellation")

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_mark_prepared_terminal_commit_started",
                        side_effect=mark_then_cancel,
                    ),
                    self.assertRaisesRegex(KeyboardInterrupt, "commit-start"),
                ):
                    generation._commit_prepared_terminal(prepared)

                self.assertFalse(prepared.commit_started)
                self.assertFalse((target_dir / "success.json").exists())
                self.assertFalse(
                    any(path.name.endswith(".tmp") for path in staging_dir.iterdir())
                )
                with self.assertRaises(OSError) as closed:
                    os.fstat(prepared_descriptor)
                self.assertEqual(closed.exception.errno, errno.EBADF)
            finally:
                os.close(staging_fd)
                os.close(target_fd)

    def test_committed_state_handoff_cancellation_closes_prepared_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "target"
            staging_dir = root / "staging"
            target_dir.mkdir()
            staging_dir.mkdir()
            target_fd = generation._open_pinned_directory(target_dir, "target")
            staging_fd = generation._open_pinned_directory(staging_dir, "staging")
            prepared = generation._prepare_terminal_json(
                target_dir / "success.json",
                {"status": "complete"},
                directory_descriptor=target_fd,
                staging_directory_descriptor=staging_fd,
            )
            prepared_descriptor = prepared.descriptor
            real_mark = generation._mark_prepared_terminal_committed

            def mark_then_cancel(value):
                real_mark(value)
                raise KeyboardInterrupt("simulated committed-state cancellation")

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_mark_prepared_terminal_committed",
                        side_effect=mark_then_cancel,
                    ),
                    self.assertRaisesRegex(KeyboardInterrupt, "committed-state"),
                ):
                    generation._commit_prepared_terminal(prepared)

                self.assertTrue(prepared.commit_started)
                self.assertTrue(prepared.committed)
                self.assertTrue(prepared.indeterminate)
                self.assertTrue((target_dir / "success.json").is_file())
                with self.assertRaises(OSError) as closed:
                    os.fstat(prepared_descriptor)
                self.assertEqual(closed.exception.errno, errno.EBADF)
            finally:
                os.close(staging_fd)
                os.close(target_fd)

    def test_prepared_close_cancellation_never_closes_reused_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "target"
            staging_dir = root / "staging"
            target_dir.mkdir()
            staging_dir.mkdir()
            target_fd = generation._open_pinned_directory(target_dir, "target")
            staging_fd = generation._open_pinned_directory(staging_dir, "staging")
            prepared = generation._prepare_terminal_json(
                target_dir / "success.json",
                {"status": "complete"},
                directory_descriptor=target_fd,
                staging_directory_descriptor=staging_fd,
            )
            prepared_descriptor = prepared.descriptor
            real_close = generation.os.close
            sentinel_descriptor = -1
            cancelled = False

            def close_then_reuse(descriptor):
                nonlocal sentinel_descriptor, cancelled
                if descriptor == prepared_descriptor and not cancelled:
                    cancelled = True
                    real_close(descriptor)
                    sentinel_descriptor = os.open("/dev/null", os.O_RDONLY)
                    self.assertEqual(sentinel_descriptor, descriptor)
                    raise KeyboardInterrupt("simulated prepared close cancellation")
                return real_close(descriptor)

            try:
                with (
                    mock.patch.object(
                        generation.os,
                        "close",
                        side_effect=close_then_reuse,
                    ),
                    self.assertRaisesRegex(KeyboardInterrupt, "prepared close"),
                ):
                    generation._commit_prepared_terminal(prepared)

                self.assertTrue(cancelled)
                self.assertEqual(prepared.descriptor, -1)
                self.assertTrue((target_dir / "success.json").is_file())
                os.fstat(sentinel_descriptor)
            finally:
                if sentinel_descriptor >= 0:
                    os.close(sentinel_descriptor)
                os.close(staging_fd)
                os.close(target_fd)

    def test_positive_link_helper_cancellation_closes_prepared_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "target"
            staging_dir = root / "staging"
            target_dir.mkdir()
            staging_dir.mkdir()
            target_fd = generation._open_pinned_directory(target_dir, "target")
            staging_fd = generation._open_pinned_directory(staging_dir, "staging")
            prepared = generation._prepare_terminal_json(
                target_dir / "success.json",
                {"status": "complete"},
                directory_descriptor=target_fd,
                staging_directory_descriptor=staging_fd,
            )
            prepared_descriptor = prepared.descriptor
            real_link = generation._link_with_positive_reconciliation

            def link_then_cancel(*args, **kwargs):
                real_link(*args, **kwargs)
                raise KeyboardInterrupt("simulated link helper return cancellation")

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_link_with_positive_reconciliation",
                        side_effect=link_then_cancel,
                    ),
                    self.assertRaisesRegex(KeyboardInterrupt, "helper return"),
                ):
                    generation._commit_prepared_terminal(prepared)

                self.assertTrue(prepared.indeterminate)
                self.assertTrue((target_dir / "success.json").is_file())
                with self.assertRaises(OSError) as closed:
                    os.fstat(prepared_descriptor)
                self.assertEqual(closed.exception.errno, errno.EBADF)
            finally:
                os.close(staging_fd)
                os.close(target_fd)

    def test_reconciliation_cancellation_forbids_another_link_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "target"
            staging_dir = root / "staging"
            target_dir.mkdir()
            staging_dir.mkdir()
            target_fd = generation._open_pinned_directory(target_dir, "target")
            staging_fd = generation._open_pinned_directory(staging_dir, "staging")
            prepared = generation._prepare_terminal_json(
                target_dir / "success.json",
                {"status": "complete"},
                directory_descriptor=target_fd,
                staging_directory_descriptor=staging_fd,
            )
            link_calls = 0

            def fail_link_before_commit(*_args, **_kwargs):
                nonlocal link_calls
                link_calls += 1
                raise OSError(errno.EIO, "simulated ambiguous link failure")

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_link_descriptor",
                        side_effect=fail_link_before_commit,
                    ),
                    mock.patch.object(
                        generation.time,
                        "sleep",
                        side_effect=KeyboardInterrupt(
                            "simulated reconciliation cancellation"
                        ),
                    ),
                    self.assertRaises(generation.AtomicPublicationIndeterminate),
                ):
                    generation._commit_prepared_terminal(prepared)
                self.assertEqual(link_calls, 1)
                self.assertTrue(prepared.commit_started)
                self.assertTrue(prepared.indeterminate)
                self.assertFalse((target_dir / "success.json").exists())
                self.assertTrue(
                    any(path.name.endswith(".tmp") for path in staging_dir.iterdir())
                )
            finally:
                os.close(staging_fd)
                os.close(target_fd)

    def test_post_link_cancellation_rolls_back_the_owned_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            candidates = []
            real_link = generation._link_descriptor

            def link_then_cancel(descriptor, directory_descriptor, name):
                real_link(descriptor, directory_descriptor, name)
                raise KeyboardInterrupt("simulated cancellation after link")

            with (
                mock.patch.object(
                    generation,
                    "_link_descriptor",
                    side_effect=link_then_cancel,
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "after link"),
            ):
                generation.atomic_create_bytes(
                    target,
                    b"cancelled\n",
                    ownership_candidates=candidates,
                )

            self.assertEqual(len(candidates), 1)
            self.assertFalse(target.exists())
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in target.parent.iterdir())
            )

    def test_uncertain_temporary_create_never_adopts_a_foreign_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "receipt.json"
            foreign = parent / "foreign.tmp"
            foreign.write_bytes(b"")
            real_open = generation.os.open
            injected = False

            def replace_created_temporary(path, flags, *args, **kwargs):
                nonlocal injected
                if not injected and flags & os.O_EXCL and flags & os.O_CREAT:
                    injected = True
                    descriptor = real_open(path, flags, *args, **kwargs)
                    os.close(descriptor)
                    os.unlink(path, dir_fd=kwargs.get("dir_fd"))
                    os.link(
                        foreign.name,
                        path,
                        src_dir_fd=kwargs.get("dir_fd"),
                        dst_dir_fd=kwargs.get("dir_fd"),
                        follow_symlinks=False,
                    )
                    raise OSError(errno.EIO, "simulated ambiguous exclusive create")
                return real_open(path, flags, *args, **kwargs)

            with (
                mock.patch.object(
                    generation.os,
                    "open",
                    side_effect=replace_created_temporary,
                ),
                self.assertRaises(OSError),
            ):
                generation.atomic_create_bytes(target, b"must-not-reach-foreign\n")

            self.assertTrue(injected)
            self.assertFalse(target.exists())
            self.assertEqual(foreign.read_bytes(), b"")
            temporary = [path for path in parent.iterdir() if path.name.endswith(".tmp")]
            self.assertEqual(len(temporary), 2)
            self.assertEqual(
                generation._object_identity(temporary[0].stat()),
                generation._object_identity(temporary[1].stat()),
            )

    def test_external_staging_cleanup_failure_does_not_pollute_target(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target_directory = parent / "target"
            staging_directory = parent / "staging"
            target_directory.mkdir()
            staging_directory.mkdir()
            target_descriptor = generation._open_pinned_directory(
                target_directory, "fixture target"
            )
            staging_descriptor = generation._open_pinned_directory(
                staging_directory, "fixture staging"
            )
            real_remove = generation._remove_matching_entry

            def fail_temporary_cleanup(directory_descriptor, name, identity):
                if directory_descriptor == staging_descriptor and name.endswith(".tmp"):
                    raise OSError(errno.EIO, "simulated external staging cleanup failure")
                return real_remove(directory_descriptor, name, identity)

            try:
                with mock.patch.object(
                    generation,
                    "_remove_matching_entry",
                    side_effect=fail_temporary_cleanup,
                ):
                    identity = generation.atomic_create_bytes(
                        target_directory / "receipt.json",
                        b"committed\n",
                        directory_descriptor=target_descriptor,
                        staging_directory_descriptor=staging_descriptor,
                    )
            finally:
                os.close(staging_descriptor)
                os.close(target_descriptor)

            self.assertEqual(
                {path.name for path in target_directory.iterdir()}, {"receipt.json"}
            )
            self.assertEqual(
                identity,
                generation._object_identity((target_directory / "receipt.json").stat()),
            )
            staging_entries = list(staging_directory.iterdir())
            self.assertEqual(len(staging_entries), 1)
            self.assertTrue(staging_entries[0].name.endswith(".tmp"))

    def test_target_replacement_during_staging_cleanup_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target_directory = parent / "target"
            staging_directory = parent / "staging"
            target_directory.mkdir()
            staging_directory.mkdir()
            target = target_directory / "receipt.json"
            target_descriptor = generation._open_pinned_directory(
                target_directory, "fixture target"
            )
            staging_descriptor = generation._open_pinned_directory(
                staging_directory, "fixture staging"
            )
            real_remove = generation._remove_matching_entry
            replaced = False

            def replace_target_after_cleanup(directory_descriptor, name, identity):
                nonlocal replaced
                result = real_remove(directory_descriptor, name, identity)
                if (
                    not replaced
                    and directory_descriptor == staging_descriptor
                    and name.endswith(".tmp")
                ):
                    replaced = True
                    target.unlink()
                    target.write_bytes(b"foreign\n")
                return result

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_remove_matching_entry",
                        side_effect=replace_target_after_cleanup,
                    ),
                    self.assertRaisesRegex(RuntimeError, "staging cleanup"),
                ):
                    generation.atomic_create_bytes(
                        target,
                        b"trusted\n",
                        directory_descriptor=target_descriptor,
                        staging_directory_descriptor=staging_descriptor,
                    )
            finally:
                os.close(staging_descriptor)
                os.close(target_descriptor)

            self.assertTrue(replaced)
            self.assertEqual(target.read_bytes(), b"foreign\n")

    def test_published_artifact_detects_same_size_rewrite_with_restored_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "record.bin"
            target.write_bytes(b"AAAA")
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            publications = generation._PublishedArtifacts()
            try:
                publications.add_exact(
                    descriptor,
                    target.name,
                    generation._object_identity(target.stat()),
                    b"AAAA",
                    "fixture record",
                )
                before = target.stat()
                target.write_bytes(b"BBBB")
                os.utime(
                    target,
                    ns=(before.st_atime_ns, before.st_mtime_ns),
                )
                with self.assertRaisesRegex(RuntimeError, "changed before terminal"):
                    publications.verify()
            finally:
                publications.close()
                os.close(descriptor)

    def test_published_artifact_row_owns_descriptor_after_append_cancellation(self):
        class AppendThenCancel(list):
            def append(self, value):
                super().append(value)
                raise KeyboardInterrupt("simulated publication row cancellation")

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "record.bin"
            payload = b"AAAA"
            target.write_bytes(payload)
            directory_descriptor = generation._open_pinned_directory(
                parent,
                "fixture parent",
            )
            publications = generation._PublishedArtifacts()
            publications._rows = AppendThenCancel()
            try:
                with self.assertRaisesRegex(KeyboardInterrupt, "row cancellation"):
                    publications.add_exact(
                        directory_descriptor,
                        target.name,
                        generation._object_identity(target.stat()),
                        payload,
                        "fixture record",
                    )
                self.assertEqual(len(publications._rows), 1)
                held_descriptor = publications._rows[0][0]
                os.fstat(held_descriptor)
                publications.close()
                with self.assertRaises(OSError) as closed:
                    os.fstat(held_descriptor)
                self.assertEqual(closed.exception.errno, errno.EBADF)
            finally:
                publications.close()
                os.close(directory_descriptor)

    def test_stable_snapshot_ignores_nfs_link_cleanup_attribute_lag(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "record.bin"
            target.write_bytes(b"stable")
            directory_descriptor = generation._open_pinned_directory(
                parent, "fixture parent"
            )
            descriptor = os.open(
                target.name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            observed = os.fstat(descriptor)

            def metadata(*, ctime_ns, nlink):
                return SimpleNamespace(
                    st_dev=observed.st_dev,
                    st_ino=observed.st_ino,
                    st_mode=observed.st_mode,
                    st_uid=observed.st_uid,
                    st_gid=observed.st_gid,
                    st_size=observed.st_size,
                    st_mtime_ns=observed.st_mtime_ns,
                    st_ctime_ns=ctime_ns,
                    st_nlink=nlink,
                )

            descriptor_view = metadata(ctime_ns=100, nlink=2)
            path_view = metadata(ctime_ns=200, nlink=1)
            try:
                with (
                    mock.patch.object(
                        generation.os,
                        "fstat",
                        return_value=descriptor_view,
                    ),
                    mock.patch.object(
                        generation.os,
                        "stat",
                        return_value=path_view,
                    ),
                ):
                    raw, stable = generation._stable_regular_snapshot(
                        descriptor,
                        directory_descriptor,
                        target.name,
                        "fixture record",
                    )
                self.assertEqual(raw, b"stable")
                self.assertEqual(stable, generation._stable_file_metadata(path_view))
            finally:
                os.close(descriptor)
                os.close(directory_descriptor)

    def test_directory_open_close_error_does_not_mask_validation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            real_stat = generation.os.stat
            real_close = generation.os.close
            close_failed = False

            def report_regular_path(target, *args, **kwargs):
                value = real_stat(target, *args, **kwargs)
                return SimpleNamespace(
                    st_dev=value.st_dev,
                    st_ino=value.st_ino,
                    st_mode=0o100600,
                )

            def close_then_report_error(descriptor):
                nonlocal close_failed
                real_close(descriptor)
                if not close_failed:
                    close_failed = True
                    raise OSError(errno.EIO, "simulated directory close failure")

            with (
                mock.patch.object(
                    generation.os,
                    "stat",
                    side_effect=report_regular_path,
                ),
                mock.patch.object(
                    generation.os,
                    "close",
                    side_effect=close_then_report_error,
                ),
                self.assertRaisesRegex(ValueError, "stable non-symlink") as raised,
            ):
                generation._open_pinned_directory(path, "fixture directory")

            self.assertTrue(close_failed)
            self.assertTrue(
                any("directory close failure" in note for note in raised.exception.__notes__)
            )

    def test_published_artifact_detects_rewrite_during_descriptor_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "record.bin"
            target.write_bytes(b"AAAA")
            directory_descriptor = generation._open_pinned_directory(
                parent, "fixture parent"
            )
            publications = generation._PublishedArtifacts()
            real_pread = generation.os.pread
            mutated = False
            try:
                publications.add_exact(
                    directory_descriptor,
                    target.name,
                    generation._object_identity(target.stat()),
                    b"AAAA",
                    "fixture record",
                )
                before = target.stat()

                def mutate_after_first_chunk(descriptor, count, offset):
                    nonlocal mutated
                    chunk = real_pread(descriptor, count, offset)
                    if chunk and not mutated:
                        mutated = True
                        os.chmod(target, 0o640)
                        target.write_bytes(b"BBBB")
                        os.chmod(target, before.st_mode & 0o777)
                        os.utime(
                            target,
                            ns=(before.st_atime_ns, before.st_mtime_ns),
                        )
                    return chunk

                with (
                    mock.patch.object(
                        generation.os,
                        "pread",
                        side_effect=mutate_after_first_chunk,
                    ),
                    self.assertRaisesRegex(RuntimeError, "changed before terminal"),
                ):
                    publications.verify()
                self.assertTrue(mutated)
            finally:
                publications.close()
                os.close(directory_descriptor)

    def test_global_snapshot_detects_earlier_file_changed_during_later_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            first = parent / "first.bin"
            second = parent / "second.bin"
            first.write_bytes(b"AAAA")
            second.write_bytes(b"CCCC")
            directory_descriptor = generation._open_pinned_directory(
                parent, "fixture parent"
            )
            publications = generation._PublishedArtifacts()
            real_pread = generation.os.pread
            mutated = False
            try:
                for path, payload in ((first, b"AAAA"), (second, b"CCCC")):
                    publications.add_exact(
                        directory_descriptor,
                        path.name,
                        generation._object_identity(path.stat()),
                        payload,
                        f"fixture {path.name}",
                    )
                second_descriptor = publications._rows[1][0]
                first_before = first.stat()

                def mutate_first_while_hashing_second(descriptor, count, offset):
                    nonlocal mutated
                    chunk = real_pread(descriptor, count, offset)
                    if descriptor == second_descriptor and chunk and not mutated:
                        mutated = True
                        os.chmod(first, 0o640)
                        first.write_bytes(b"BBBB")
                        os.chmod(first, first_before.st_mode & 0o777)
                        os.utime(
                            first,
                            ns=(first_before.st_atime_ns, first_before.st_mtime_ns),
                        )
                    return chunk

                with (
                    mock.patch.object(
                        generation.os,
                        "pread",
                        side_effect=mutate_first_while_hashing_second,
                    ),
                    self.assertRaisesRegex(RuntimeError, "changed before terminal"),
                ):
                    publications.verify()
                self.assertTrue(mutated)
            finally:
                publications.close()
                os.close(directory_descriptor)

    def test_committed_output_is_not_masked_by_source_close_error(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            real_close = generation.os.close
            failed = False

            def close_then_report_error(descriptor):
                nonlocal failed
                if target.exists() and not failed and not os.path.isdir(
                    f"/proc/self/fd/{descriptor}"
                ):
                    failed = True
                    real_close(descriptor)
                    raise OSError(errno.EIO, "simulated source close error")
                return real_close(descriptor)

            with mock.patch.object(
                generation.os,
                "close",
                side_effect=close_then_report_error,
            ):
                identity = generation.atomic_create_bytes(target, b"committed\n")

            self.assertTrue(failed)
            self.assertEqual(target.read_bytes(), b"committed\n")
            self.assertEqual(identity, generation._object_identity(target.stat()))

    def test_committed_output_is_not_masked_by_parent_close_error(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            real_close = generation.os.close
            failed = False

            def close_then_report_error(descriptor):
                nonlocal failed
                if target.exists() and not failed and os.path.isdir(
                    f"/proc/self/fd/{descriptor}"
                ):
                    failed = True
                    real_close(descriptor)
                    raise OSError(errno.EIO, "simulated parent close error")
                return real_close(descriptor)

            with mock.patch.object(
                generation.os,
                "close",
                side_effect=close_then_report_error,
            ):
                identity = generation.atomic_create_bytes(target, b"committed\n")

            self.assertTrue(failed)
            self.assertEqual(target.read_bytes(), b"committed\n")
            self.assertEqual(identity, generation._object_identity(target.stat()))

    def test_temporary_name_replacement_never_publishes_foreign_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            payload = b"descriptor-bound-payload\n"
            forged_payload = b"forged-path-bytes"
            real_link = generation._link_descriptor
            source_identity = None
            foreign_identity = None

            def replace_temporary_then_link(
                descriptor, directory_descriptor, name
            ):
                nonlocal source_identity, foreign_identity
                source_identity = generation._object_identity(os.fstat(descriptor))
                temporary_names = [
                    entry
                    for entry in os.listdir(directory_descriptor)
                    if entry.endswith(".tmp")
                ]
                self.assertEqual(len(temporary_names), 1)
                temporary_name = temporary_names[0]
                os.unlink(temporary_name, dir_fd=directory_descriptor)
                replacement = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_descriptor,
                )
                try:
                    os.write(replacement, forged_payload)
                    foreign_identity = generation._object_identity(
                        os.fstat(replacement)
                    )
                finally:
                    os.close(replacement)
                real_link(descriptor, directory_descriptor, name)

            identity = None
            try:
                with mock.patch.object(
                    generation,
                    "_link_descriptor",
                    side_effect=replace_temporary_then_link,
                ):
                    identity = generation.atomic_create_bytes(target, payload)
            except (OSError, RuntimeError):
                pass

            self.assertIsNotNone(source_identity)
            self.assertIsNotNone(foreign_identity)
            if identity is not None:
                self.assertTrue(target.is_file())
            if target.exists():
                self.assertEqual(target.read_bytes(), payload)
                self.assertEqual(
                    generation._object_identity(target.stat()),
                    source_identity,
                )
                self.assertNotEqual(
                    generation._object_identity(target.stat()),
                    foreign_identity,
                )
            foreign_entries = [
                entry
                for entry in target.parent.iterdir()
                if generation._object_identity(entry.stat()) == foreign_identity
            ]
            self.assertEqual(len(foreign_entries), 1)
            self.assertEqual(foreign_entries[0].read_bytes(), forged_payload)

    def test_temporary_name_replacement_after_commit_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            real_link = generation._link_descriptor

            def link_then_replace_temporary(
                descriptor, directory_descriptor, name
            ):
                real_link(descriptor, directory_descriptor, name)
                temporary_name = next(
                    entry
                    for entry in os.listdir(directory_descriptor)
                    if entry.endswith(".tmp")
                )
                os.unlink(temporary_name, dir_fd=directory_descriptor)
                replacement = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_descriptor,
                )
                os.close(replacement)

            with mock.patch.object(
                generation,
                "_link_descriptor",
                side_effect=link_then_replace_temporary,
            ):
                identity = generation.atomic_create_bytes(target, b"trusted\n")
            self.assertEqual(target.read_bytes(), b"trusted\n")
            self.assertEqual(identity, generation._object_identity(target.stat()))
            temporary = [entry for entry in target.parent.iterdir() if entry.name.endswith(".tmp")]
            self.assertEqual(len(temporary), 1)
            self.assertEqual(temporary[0].read_bytes(), b"")

    def test_directory_fsync_error_rolls_back_committed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            real_fsync = generation.os.fsync
            failed = False

            def fail_first_directory_fsync(descriptor):
                nonlocal failed
                if not failed and os.path.isdir(f"/proc/self/fd/{descriptor}"):
                    failed = True
                    raise OSError(errno.EIO, "simulated directory fsync failure")
                return real_fsync(descriptor)

            with mock.patch.object(
                generation.os,
                "fsync",
                side_effect=fail_first_directory_fsync,
            ):
                with self.assertRaises(OSError):
                    generation.atomic_create_bytes(target, b"not-durable\n")
            self.assertTrue(failed)
            self.assertFalse(target.exists())

    def test_post_link_metadata_error_rolls_back_committed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            real_link = generation._link_descriptor
            real_stat = generation.os.stat
            linked = False
            failed = False

            def publish(descriptor, directory_descriptor, name):
                nonlocal linked
                real_link(descriptor, directory_descriptor, name)
                linked = True

            def fail_first_target_stat(path, *args, **kwargs):
                nonlocal failed
                if linked and not failed and path == target.name:
                    failed = True
                    raise OSError(errno.EIO, "simulated NFS metadata failure")
                return real_stat(path, *args, **kwargs)

            with (
                mock.patch.object(
                    generation,
                    "_link_descriptor",
                    side_effect=publish,
                ),
                mock.patch.object(
                    generation.os,
                    "stat",
                    side_effect=fail_first_target_stat,
                ),
                self.assertRaises(OSError),
            ):
                generation.atomic_create_bytes(target, b"linked-before-error\n")

            self.assertTrue(failed)
            self.assertFalse(target.exists())

    def test_final_link_and_temporary_cleanup_are_fsynced_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            real_link = generation._link_descriptor
            real_fsync = generation.os.fsync
            real_remove = generation._remove_matching_entry
            linked = False
            post_link_directory_fsyncs = 0

            def publish(descriptor, directory_descriptor, name):
                nonlocal linked
                real_link(descriptor, directory_descriptor, name)
                linked = True

            def count_fsync(descriptor):
                nonlocal post_link_directory_fsyncs
                if linked and os.path.isdir(f"/proc/self/fd/{descriptor}"):
                    post_link_directory_fsyncs += 1
                return real_fsync(descriptor)

            def remove_without_implicit_fsync(directory_descriptor, name, identity):
                if name.endswith(".tmp"):
                    os.unlink(name, dir_fd=directory_descriptor)
                    return
                return real_remove(directory_descriptor, name, identity)

            with (
                mock.patch.object(
                    generation,
                    "_link_descriptor",
                    side_effect=publish,
                ),
                mock.patch.object(
                    generation.os,
                    "fsync",
                    side_effect=count_fsync,
                ),
                mock.patch.object(
                    generation,
                    "_remove_matching_entry",
                    side_effect=remove_without_implicit_fsync,
                ),
            ):
                generation.atomic_create_bytes(target, b"durable-name\n")

            self.assertGreaterEqual(post_link_directory_fsyncs, 2)
            self.assertEqual(target.read_bytes(), b"durable-name\n")

    def test_matching_rollback_never_deletes_a_racing_foreign_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "receipt.json"
            foreign = parent / "foreign.json"
            target.write_bytes(b"owned\n")
            foreign.write_bytes(b"foreign\n")
            identity = generation._object_identity(target.stat())
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            real_rename = generation.os.rename
            raced = False

            def replace_before_move(
                source,
                destination,
                *,
                src_dir_fd=None,
                dst_dir_fd=None,
            ):
                nonlocal raced
                if source == target.name and not raced:
                    raced = True
                    os.unlink(source, dir_fd=src_dir_fd)
                    real_rename(
                        foreign.name,
                        source,
                        src_dir_fd=src_dir_fd,
                        dst_dir_fd=src_dir_fd,
                    )
                return real_rename(
                    source,
                    destination,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                )

            try:
                with (
                    mock.patch.object(
                        generation.os,
                        "rename",
                        side_effect=replace_before_move,
                    ),
                    self.assertRaisesRegex(RuntimeError, "preserved"),
                ):
                    generation._remove_matching_entry(
                        descriptor,
                        target.name,
                        identity,
                    )
            finally:
                os.close(descriptor)

            preserved = list(parent.glob(".*.rollback"))
            self.assertTrue(raced)
            self.assertEqual(len(preserved), 1)
            self.assertEqual(preserved[0].read_bytes(), b"foreign\n")

    def test_unsupported_noreplace_uses_pinned_staging_and_claim_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            claimed = -1
            try:
                with mock.patch.object(
                    generation,
                    "_rename_directory_noreplace",
                    side_effect=OSError(errno.EOPNOTSUPP, "unsupported"),
                ):
                    claimed = generation._claim_pinned_directory_at(
                        descriptor,
                        "claimed",
                        "fixture claim",
                    )
                generation._verify_directory_entry(
                    descriptor,
                    "claimed",
                    claimed,
                    "fixture claim",
                )
                self.assertEqual(os.listdir(claimed), [])
                self.assertTrue((parent / ".claimed.claim").is_file())
            finally:
                if claimed >= 0:
                    os.close(claimed)
                os.close(descriptor)

    def test_fallback_exclusive_mkdir_preserves_a_competing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            real_mkdir = generation.os.mkdir
            competed = False

            def compete_before_final_mkdir(path, *args, **kwargs):
                nonlocal competed
                if path == "claimed" and not competed:
                    competed = True
                    real_mkdir(path, *args, **kwargs)
                    foreign_descriptor = os.open(
                        path,
                        generation._directory_flags(),
                        dir_fd=kwargs.get("dir_fd"),
                    )
                    try:
                        child = os.open(
                            "foreign.txt",
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600,
                            dir_fd=foreign_descriptor,
                        )
                        os.close(child)
                    finally:
                        os.close(foreign_descriptor)
                return real_mkdir(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=OSError(errno.EOPNOTSUPP, "unsupported"),
                    ),
                    mock.patch.object(
                        generation.os,
                        "mkdir",
                        side_effect=compete_before_final_mkdir,
                    ),
                    self.assertRaises(FileExistsError),
                ):
                    generation._claim_pinned_directory_at(
                        descriptor,
                        "claimed",
                        "fixture claim",
                    )
                self.assertTrue(competed)
                self.assertEqual(
                    {path.name for path in (parent / "claimed").iterdir()},
                    {"foreign.txt"},
                )
            finally:
                os.close(descriptor)

    def test_fallback_does_not_adopt_an_ambiguous_final_mkdir(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            real_mkdir = generation.os.mkdir
            reported = False
            ownership = []

            def create_final_then_report_error(path, *args, **kwargs):
                nonlocal reported
                real_mkdir(path, *args, **kwargs)
                if path == "claimed" and not reported:
                    reported = True
                    raise OSError(errno.EIO, "simulated final mkdir reply loss")

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=OSError(errno.EOPNOTSUPP, "unsupported"),
                    ),
                    mock.patch.object(
                        generation.os,
                        "mkdir",
                        side_effect=create_final_then_report_error,
                    ),
                    self.assertRaises(FileExistsError),
                ):
                    generation._claim_pinned_directory_at(
                        descriptor,
                        "claimed",
                        "fixture claim",
                        ownership_candidates=ownership,
                    )
                self.assertTrue(reported)
                self.assertEqual(ownership, [])
                self.assertTrue((parent / "claimed").is_dir())
                self.assertTrue((parent / ".claimed.claim").is_file())
                self.assertEqual(
                    len(
                        [
                            path
                            for path in parent.iterdir()
                            if path.name.startswith(".claimed.")
                            and path.name.endswith(".claim")
                            and path.is_dir()
                        ]
                    ),
                    1,
                )
            finally:
                os.close(descriptor)

    def test_fallback_reconciles_an_ambiguous_marker_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            real_atomic = generation.atomic_create_bytes
            reported = False
            claimed = -1

            def publish_marker_then_report_error(path, payload, **kwargs):
                nonlocal reported
                identity = real_atomic(path, payload, **kwargs)
                if Path(path).name == ".claimed.claim" and not reported:
                    reported = True
                    raise OSError(errno.EIO, "simulated marker publication reply loss")
                return identity

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=OSError(errno.EOPNOTSUPP, "unsupported"),
                    ),
                    mock.patch.object(
                        generation,
                        "atomic_create_bytes",
                        side_effect=publish_marker_then_report_error,
                    ),
                ):
                    claimed = generation._claim_pinned_directory_at(
                        descriptor, "claimed", "fixture claim"
                    )
                self.assertTrue(reported)
                generation._verify_directory_entry(
                    descriptor, "claimed", claimed, "fixture claim"
                )
                self.assertTrue((parent / ".claimed.claim").is_file())
            finally:
                if claimed >= 0:
                    os.close(claimed)
                os.close(descriptor)

    def test_interrupted_marker_observation_preserves_staging_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            real_atomic = generation.atomic_create_bytes
            real_entry = generation._entry_metadata
            marker_reported = False

            def publish_marker_then_report_error(path, payload, **kwargs):
                nonlocal marker_reported
                identity = real_atomic(path, payload, **kwargs)
                if Path(path).name == ".claimed.claim" and not marker_reported:
                    marker_reported = True
                    raise OSError(errno.EIO, "simulated marker reply loss")
                return identity

            def cancel_marker_observation(directory_descriptor, name):
                if marker_reported and name == ".claimed.claim":
                    raise KeyboardInterrupt(
                        "simulated marker reconciliation cancellation"
                    )
                return real_entry(directory_descriptor, name)

            try:
                with (
                    mock.patch.object(
                        generation,
                        "atomic_create_bytes",
                        side_effect=publish_marker_then_report_error,
                    ),
                    mock.patch.object(
                        generation,
                        "_entry_metadata",
                        side_effect=cancel_marker_observation,
                    ),
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                    ) as rename,
                    self.assertRaisesRegex(
                        generation.DirectoryClaimIndeterminate,
                        "observation was interrupted",
                    ) as raised,
                ):
                    generation._claim_pinned_directory_at(
                        descriptor, "claimed", "fixture claim"
                    )
                rename.assert_not_called()
                self.assertTrue(
                    generation._exception_contains_cancellation(raised.exception)
                )
                self.assertTrue(marker_reported)
                self.assertFalse((parent / "claimed").exists())
                self.assertTrue((parent / ".claimed.claim").is_file())
                staging = [
                    path
                    for path in parent.iterdir()
                    if path.name.startswith(".claimed.")
                    and path.name.endswith(".claim")
                    and path.is_dir()
                ]
                self.assertEqual(len(staging), 1)
            finally:
                os.close(descriptor)

    def test_marker_cancellation_prevents_claim_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            real_atomic = generation.atomic_create_bytes
            candidates = []
            cancelled = False

            def publish_marker_then_cancel(path, payload, **kwargs):
                nonlocal cancelled
                identity = real_atomic(path, payload, **kwargs)
                if Path(path).name == ".claimed.claim" and not cancelled:
                    cancelled = True
                    raise KeyboardInterrupt("simulated cancellation after marker")
                return identity

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                    ) as rename,
                    mock.patch.object(
                        generation,
                        "atomic_create_bytes",
                        side_effect=publish_marker_then_cancel,
                    ),
                    self.assertRaisesRegex(KeyboardInterrupt, "after marker"),
                ):
                    generation._claim_pinned_directory_at(
                        descriptor,
                        "claimed",
                        "fixture claim",
                        ownership_candidates=candidates,
                    )

                self.assertTrue(cancelled)
                rename.assert_not_called()
                self.assertEqual(candidates, [])
                self.assertFalse((parent / "claimed").exists())
                self.assertTrue((parent / ".claimed.claim").is_file())
                staging = [
                    path
                    for path in parent.iterdir()
                    if path.name.startswith(".claimed.")
                    and path.name.endswith(".claim")
                    and path.is_dir()
                ]
                self.assertEqual(len(staging), 1)
            finally:
                os.close(descriptor)

    def test_marker_helper_return_cancellation_preserves_staging_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            real_publish = generation._publish_directory_claim_marker
            candidates = []

            def publish_then_cancel(**kwargs):
                real_publish(**kwargs)
                raise KeyboardInterrupt(
                    "simulated cancellation after marker helper return"
                )

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_publish_directory_claim_marker",
                        side_effect=publish_then_cancel,
                    ),
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                    ) as rename,
                    self.assertRaisesRegex(KeyboardInterrupt, "helper return"),
                ):
                    generation._claim_pinned_directory_at(
                        descriptor,
                        "claimed",
                        "fixture claim",
                        ownership_candidates=candidates,
                    )
                rename.assert_not_called()
                self.assertEqual(candidates, [])
                self.assertFalse((parent / "claimed").exists())
                self.assertTrue((parent / ".claimed.claim").is_file())
                staging = [
                    path
                    for path in parent.iterdir()
                    if path.name.startswith(".claimed.")
                    and path.name.endswith(".claim")
                    and path.is_dir()
                ]
                self.assertEqual(len(staging), 1)
            finally:
                os.close(descriptor)

    def test_preexisting_foreign_marker_is_a_conflict_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            marker = parent / ".claimed.claim"
            marker.write_bytes(b"foreign-marker\n")
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            candidates = []
            try:
                with self.assertRaises(FileExistsError):
                    generation._claim_pinned_directory_at(
                        descriptor,
                        "claimed",
                        "fixture claim",
                        ownership_candidates=candidates,
                    )
                self.assertEqual(candidates, [])
                self.assertEqual(marker.read_bytes(), b"foreign-marker\n")
                self.assertFalse((parent / "claimed").exists())
                self.assertEqual(
                    [path.name for path in parent.iterdir()],
                    [".claimed.claim"],
                )
            finally:
                os.close(descriptor)

    def test_fallback_final_mkdir_cancellation_is_indeterminate(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            ownership = []
            real_mkdir = generation.os.mkdir
            cancelled = False

            def create_final_then_cancel(name, *args, **kwargs):
                nonlocal cancelled
                result = real_mkdir(name, *args, **kwargs)
                if name == "claimed" and not cancelled:
                    cancelled = True
                    raise KeyboardInterrupt("simulated cancellation after final mkdir")
                return result

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=OSError(errno.ENOSYS, "fallback required"),
                    ),
                    mock.patch.object(
                        generation.os,
                        "mkdir",
                        side_effect=create_final_then_cancel,
                    ),
                    self.assertRaisesRegex(
                        generation.DirectoryClaimIndeterminate,
                        "unknown outcome",
                    ),
                ):
                    generation._claim_pinned_directory_at(
                        descriptor,
                        "claimed",
                        "fixture claim",
                        fallback_marker_descriptor=descriptor,
                        fallback_marker_name=".claimed.claim",
                        ownership_candidates=ownership,
                    )
                self.assertTrue(cancelled)
                self.assertEqual(ownership, [])
                self.assertTrue((parent / "claimed").is_dir())
                self.assertTrue((parent / ".claimed.claim").is_file())
                staging = [
                    path
                    for path in parent.iterdir()
                    if path.name.startswith(".claimed.")
                    and path.name.endswith(".claim")
                    and path.is_dir()
                ]
                self.assertEqual(len(staging), 1)
            finally:
                os.close(descriptor)

    def test_fallback_retries_ambiguous_mkdir_until_positive_success(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_fd = generation._open_pinned_directory(parent, "fixture parent")
            ownership = []
            real_mkdir = generation.os.mkdir
            attempts = 0
            claimed = -1

            def fail_without_side_effect_then_create(name, *args, **kwargs):
                nonlocal attempts
                if name == "claimed":
                    attempts += 1
                    if attempts == 1:
                        raise OSError(errno.EIO, "simulated mkdir request loss")
                return real_mkdir(name, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=OSError(errno.ENOSYS, "fallback required"),
                    ),
                    mock.patch.object(
                        generation.os,
                        "mkdir",
                        side_effect=fail_without_side_effect_then_create,
                    ),
                    mock.patch.object(generation.time, "sleep", return_value=None),
                ):
                    claimed = generation._claim_pinned_directory_at(
                        parent_fd,
                        "claimed",
                        "fixture claim",
                        fallback_marker_descriptor=parent_fd,
                        fallback_marker_name=".claimed.claim",
                        ownership_candidates=ownership,
                    )
                self.assertEqual(attempts, 2)
                self.assertEqual(
                    ownership,
                    [generation._object_identity((parent / "claimed").stat())],
                )
                generation._verify_directory_entry(
                    parent_fd, "claimed", claimed, "fixture claim"
                )
            finally:
                if claimed >= 0:
                    os.close(claimed)
                os.close(parent_fd)

    def test_fallback_ambiguous_mkdir_preserves_claim_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_fd = generation._open_pinned_directory(parent, "fixture parent")
            real_mkdir = generation.os.mkdir
            ownership = []

            def create_final_then_report_error(name, *args, **kwargs):
                result = real_mkdir(name, *args, **kwargs)
                if name == "claimed":
                    raise OSError(errno.EIO, "simulated final mkdir reply loss")
                return result

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=OSError(errno.ENOSYS, "fallback required"),
                    ),
                    mock.patch.object(
                        generation.os,
                        "mkdir",
                        side_effect=create_final_then_report_error,
                    ),
                    mock.patch.object(generation.time, "sleep", return_value=None),
                    self.assertRaises(FileExistsError),
                ):
                    generation._claim_pinned_directory_at(
                        parent_fd,
                        "claimed",
                        "fixture claim",
                        fallback_marker_descriptor=parent_fd,
                        fallback_marker_name=".claimed.claim",
                        ownership_candidates=ownership,
                    )
                self.assertEqual(ownership, [])
                self.assertTrue((parent / "claimed").is_dir())
                self.assertTrue((parent / ".claimed.claim").is_file())
                self.assertEqual(
                    len([path for path in parent.iterdir() if path.name.endswith(".claim") and path.is_dir()]),
                    1,
                )
            finally:
                os.close(parent_fd)

    def test_fallback_eexist_after_ambiguity_never_adopts_empty_foreign_directory(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_fd = generation._open_pinned_directory(parent, "fixture parent")
            ownership = []
            real_mkdir = generation.os.mkdir
            real_open = generation._open_pinned_directory_at
            attempts = 0
            target_opened = False

            def lose_first_request_then_compete(name, *args, **kwargs):
                nonlocal attempts
                if name != "claimed":
                    return real_mkdir(name, *args, **kwargs)
                attempts += 1
                if attempts == 1:
                    raise OSError(errno.EIO, "simulated lost mkdir request")
                real_mkdir(name, *args, **kwargs)
                return real_mkdir(name, *args, **kwargs)

            def track_target_open(parent_descriptor, name, label, **kwargs):
                nonlocal target_opened
                if name == "claimed":
                    target_opened = True
                return real_open(parent_descriptor, name, label, **kwargs)

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=OSError(errno.ENOSYS, "fallback required"),
                    ),
                    mock.patch.object(
                        generation.os,
                        "mkdir",
                        side_effect=lose_first_request_then_compete,
                    ),
                    mock.patch.object(
                        generation,
                        "_open_pinned_directory_at",
                        side_effect=track_target_open,
                    ),
                    mock.patch.object(generation.time, "sleep", return_value=None),
                    self.assertRaises(FileExistsError),
                ):
                    generation._claim_pinned_directory_at(
                        parent_fd,
                        "claimed",
                        "fixture claim",
                        ownership_candidates=ownership,
                    )
                self.assertEqual(attempts, 2)
                self.assertFalse(target_opened)
                self.assertEqual(ownership, [])
                self.assertEqual(list((parent / "claimed").iterdir()), [])
                self.assertTrue((parent / ".claimed.claim").is_file())
            finally:
                os.close(parent_fd)

    def test_fallback_close_error_does_not_close_a_reused_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_fd = generation._open_pinned_directory(parent, "fixture parent")
            real_close = generation.os.close
            staging_descriptor = -1
            reused_descriptor = -1
            reported = False
            claimed = -1

            def require_fallback(
                _source_parent,
                _target_parent,
                _source_name,
                _target_name,
                source_descriptor,
            ):
                nonlocal staging_descriptor
                staging_descriptor = source_descriptor
                raise OSError(errno.ENOSYS, "fallback required")

            def close_then_reuse(descriptor):
                nonlocal reported, reused_descriptor
                if descriptor == staging_descriptor and not reported:
                    reported = True
                    real_close(descriptor)
                    reused_descriptor = os.open("/dev/null", os.O_RDONLY)
                    self.assertEqual(reused_descriptor, descriptor)
                    raise OSError(errno.EIO, "simulated close reply error")
                return real_close(descriptor)

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=require_fallback,
                    ),
                    mock.patch.object(
                        generation.os,
                        "close",
                        side_effect=close_then_reuse,
                    ),
                ):
                    claimed = generation._claim_pinned_directory_at(
                        parent_fd, "claimed", "fixture claim"
                    )
                self.assertTrue(reported)
                os.fstat(reused_descriptor)
            finally:
                if claimed >= 0:
                    os.close(claimed)
                if reused_descriptor >= 0:
                    os.close(reused_descriptor)
                os.close(parent_fd)

    def test_missing_libc_renameat2_is_reported_as_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source"
            source.mkdir()
            parent_descriptor = generation._open_pinned_directory(
                parent, "fixture parent"
            )
            source_descriptor = generation._open_pinned_directory(
                source, "fixture source"
            )
            try:
                with (
                    mock.patch.object(
                        generation.ctypes,
                        "CDLL",
                        return_value=SimpleNamespace(),
                    ),
                    self.assertRaises(OSError) as raised,
                ):
                    generation._rename_directory_noreplace(
                        parent_descriptor,
                        parent_descriptor,
                        source.name,
                        "target",
                        source_descriptor,
                    )
                self.assertEqual(raised.exception.errno, errno.ENOSYS)
            finally:
                os.close(source_descriptor)
                os.close(parent_descriptor)

    def test_ambiguous_rename_retries_a_stale_target_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source"
            source.mkdir()
            parent_fd = generation._open_pinned_directory(parent, "fixture parent")
            source_fd = generation._open_pinned_directory(source, "fixture source")
            real_stat = generation.os.stat
            stale = False

            class RenameThenReportError:
                argtypes = None
                restype = None

                def __call__(self, *_args):
                    os.rename(
                        "source",
                        "target",
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                    generation.ctypes.set_errno(errno.EIO)
                    return -1

            def stale_once(name, *args, **kwargs):
                nonlocal stale
                if name == "target" and not stale:
                    stale = True
                    raise FileNotFoundError(errno.ENOENT, "stale target lookup")
                return real_stat(name, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        generation.ctypes,
                        "CDLL",
                        return_value=SimpleNamespace(
                            renameat2=RenameThenReportError()
                        ),
                    ),
                    mock.patch.object(
                        generation.os,
                        "stat",
                        side_effect=stale_once,
                    ),
                    mock.patch.object(generation.time, "sleep", return_value=None),
                ):
                    generation._rename_directory_noreplace(
                        parent_fd,
                        parent_fd,
                        "source",
                        "target",
                        source_fd,
                    )
                self.assertTrue(stale)
                generation._verify_directory_entry(
                    parent_fd, "target", source_fd, "fixture target"
                )
            finally:
                os.close(source_fd)
                os.close(parent_fd)

    def test_production_rename_eexist_is_a_definite_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source"
            target = parent / "target"
            source.mkdir()
            target.mkdir()
            parent_fd = generation._open_pinned_directory(parent, "fixture parent")
            source_fd = generation._open_pinned_directory(source, "fixture source")

            class ReturnEexist:
                argtypes = None
                restype = None

                def __call__(self, *_args):
                    generation.ctypes.set_errno(errno.EEXIST)
                    return -1

            try:
                with (
                    mock.patch.object(
                        generation.ctypes,
                        "CDLL",
                        return_value=SimpleNamespace(renameat2=ReturnEexist()),
                    ),
                    self.assertRaises(FileExistsError),
                ):
                    generation._rename_directory_noreplace(
                        parent_fd,
                        parent_fd,
                        source.name,
                        target.name,
                        source_fd,
                    )
                self.assertTrue(source.is_dir())
                self.assertTrue(target.is_dir())
            finally:
                os.close(source_fd)
                os.close(parent_fd)

    def test_ambiguous_committed_rename_with_persistent_stale_target_never_falls_back(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_fd = generation._open_pinned_directory(parent, "fixture parent")
            real_stat = generation.os.stat
            ownership = []

            class RenameThenReportEio:
                argtypes = None
                restype = None

                def __call__(
                    self,
                    source_parent,
                    source_pointer,
                    target_parent,
                    target_pointer,
                    _flags,
                ):
                    os.rename(
                        os.fsdecode(source_pointer.value),
                        os.fsdecode(target_pointer.value),
                        src_dir_fd=source_parent,
                        dst_dir_fd=target_parent,
                    )
                    generation.ctypes.set_errno(errno.EIO)
                    return -1

            def persistently_hide_target(path, *args, **kwargs):
                if path == "claimed" and kwargs.get("dir_fd") == parent_fd:
                    raise FileNotFoundError(
                        errno.ENOENT, "simulated persistent stale target"
                    )
                return real_stat(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        generation.ctypes,
                        "CDLL",
                        return_value=SimpleNamespace(
                            renameat2=RenameThenReportEio()
                        ),
                    ),
                    mock.patch.object(
                        generation.os,
                        "stat",
                        side_effect=persistently_hide_target,
                    ),
                    mock.patch.object(generation.time, "sleep", return_value=None),
                    mock.patch.object(
                        generation,
                        "_claim_pinned_directory_with_marker",
                        side_effect=AssertionError("fallback claim attempted"),
                    ) as fallback,
                    self.assertRaises(generation.DirectoryClaimIndeterminate),
                ):
                    generation._claim_pinned_directory_at(
                        parent_fd,
                        "claimed",
                        "fixture claim",
                        ownership_candidates=ownership,
                    )
                fallback.assert_not_called()
                self.assertEqual(
                    ownership,
                    [generation._object_identity((parent / "claimed").stat())],
                )
                self.assertTrue((parent / ".claimed.claim").is_file())
                self.assertEqual(
                    [
                        path
                        for path in parent.iterdir()
                        if path.name.startswith(".claimed.")
                        and path.name.endswith(".claim")
                        and path.is_dir()
                    ],
                    [],
                )
            finally:
                os.close(parent_fd)

    def test_successful_rename_with_persistent_stale_target_is_committed_unopened(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source"
            source.mkdir()
            parent_fd = generation._open_pinned_directory(parent, "fixture parent")
            source_fd = generation._open_pinned_directory(source, "fixture source")

            class RenameThenReturnSuccess:
                argtypes = None
                restype = None

                def __call__(self, *_args):
                    os.rename(
                        "source",
                        "target",
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                    return 0

            try:
                with (
                    mock.patch.object(
                        generation.ctypes,
                        "CDLL",
                        return_value=SimpleNamespace(
                            renameat2=RenameThenReturnSuccess()
                        ),
                    ),
                    mock.patch.object(
                        generation.os,
                        "stat",
                        side_effect=FileNotFoundError(
                            errno.ENOENT, "persistently stale target lookup"
                        ),
                    ),
                    mock.patch.object(generation.time, "sleep", return_value=None),
                    self.assertRaises(
                        generation.DirectoryClaimCommittedUnopened
                    ),
                ):
                    generation._rename_directory_noreplace(
                        parent_fd,
                        parent_fd,
                        "source",
                        "target",
                        source_fd,
                    )
                self.assertTrue((parent / "target").is_dir())
            finally:
                os.close(source_fd)
                os.close(parent_fd)

    def test_committed_rename_with_persistent_stale_target_keeps_marker_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_fd = generation._open_pinned_directory(parent, "fixture parent")
            ownership = []

            def commit_then_remain_stale(
                source_parent,
                target_parent,
                source_name,
                target_name,
                source_descriptor,
            ):
                identity = generation._object_identity(os.fstat(source_descriptor))
                os.rename(
                    source_name,
                    target_name,
                    src_dir_fd=source_parent,
                    dst_dir_fd=target_parent,
                )
                error = generation.DirectoryClaimCommittedUnopened(
                    "simulated persistent stale target"
                )
                error.target_name = target_name
                error.expected_identity = identity
                error.namespace_committed = True
                raise error

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=commit_then_remain_stale,
                    ),
                    mock.patch.object(
                        generation,
                        "_claim_pinned_directory_with_marker",
                        side_effect=AssertionError("fallback claim attempted"),
                    ) as fallback,
                    self.assertRaises(
                        generation.DirectoryClaimCommittedUnopened
                    ),
                ):
                    generation._claim_pinned_directory_at(
                        parent_fd,
                        "claimed",
                        "fixture claim",
                        ownership_candidates=ownership,
                    )
                fallback.assert_not_called()
                self.assertEqual(
                    ownership,
                    [generation._object_identity((parent / "claimed").stat())],
                )
                self.assertTrue((parent / ".claimed.claim").is_file())
                self.assertEqual(
                    [
                        path
                        for path in parent.iterdir()
                        if path.name.startswith(".claimed.")
                        and path.name.endswith(".claim")
                        and path.is_dir()
                    ],
                    [],
                )
            finally:
                os.close(parent_fd)

    def test_rename_eexist_is_a_conflict_without_an_ownership_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_fd = generation._open_pinned_directory(parent, "fixture parent")
            ownership = []

            def competing_target(
                _source_parent,
                target_parent,
                _source_name,
                target_name,
                _source_descriptor,
            ):
                os.mkdir(target_name, mode=0o750, dir_fd=target_parent)
                raise FileExistsError(errno.EEXIST, "simulated conflict")

            try:
                with (
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=competing_target,
                    ),
                    self.assertRaises(FileExistsError),
                ):
                    generation._claim_pinned_directory_at(
                        parent_fd,
                        "claimed",
                        "fixture claim",
                        ownership_candidates=ownership,
                    )
                self.assertEqual(ownership, [])
                self.assertEqual(list((parent / "claimed").iterdir()), [])
                self.assertTrue((parent / ".claimed.claim").is_file())
            finally:
                os.close(parent_fd)

    def test_uncertain_staging_mkdir_is_not_adopted_by_random_name(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            descriptor = generation._open_pinned_directory(parent, "fixture parent")
            real_mkdir = generation.os.mkdir
            reported = False

            def create_then_report_error(path, *args, **kwargs):
                nonlocal reported
                real_mkdir(path, *args, **kwargs)
                if not reported and str(path).startswith(".claimed."):
                    reported = True
                    raise OSError(errno.EIO, "simulated NFS reply loss")

            try:
                with (
                    mock.patch.object(
                        generation.os,
                        "mkdir",
                        side_effect=create_then_report_error,
                    ),
                    self.assertRaises(OSError),
                ):
                    generation._claim_pinned_directory_at(
                        descriptor, "claimed", "fixture claim"
                    )
                self.assertTrue(reported)
                self.assertFalse((parent / "claimed").exists())
                staging = list(parent.glob(".claimed.*.claim"))
                self.assertEqual(len(staging), 1)
                self.assertTrue(staging[0].is_dir())
            finally:
                os.close(descriptor)

    def test_directory_claim_rejects_replaced_staging_name(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_descriptor = generation._open_pinned_directory(
                parent, "fixture parent"
            )
            real_rename = generation._rename_directory_noreplace

            class RenameCurrentName:
                argtypes = None
                restype = None

                def __call__(
                    self,
                    source_parent,
                    source_pointer,
                    target_parent,
                    target_pointer,
                    _flags,
                ):
                    os.rename(
                        os.fsdecode(source_pointer.value),
                        os.fsdecode(target_pointer.value),
                        src_dir_fd=source_parent,
                        dst_dir_fd=target_parent,
                    )
                    return 0

            def replace_staging_before_claim(
                source_parent,
                target_parent,
                source_name,
                target_name,
                source_descriptor,
            ):
                displaced = f"{source_name}.displaced"
                os.rename(
                    source_name,
                    displaced,
                    src_dir_fd=source_parent,
                    dst_dir_fd=source_parent,
                )
                os.mkdir(source_name, mode=0o750, dir_fd=source_parent)
                foreign_directory = os.open(
                    source_name,
                    generation._directory_flags(),
                    dir_fd=source_parent,
                )
                try:
                    foreign = os.open(
                        "foreign.txt",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=foreign_directory,
                    )
                    os.close(foreign)
                finally:
                    os.close(foreign_directory)
                real_rename(
                    source_parent,
                    target_parent,
                    source_name,
                    target_name,
                    source_descriptor,
                )

            try:
                with (
                    mock.patch.object(
                        generation.ctypes,
                        "CDLL",
                        return_value=SimpleNamespace(
                            renameat2=RenameCurrentName()
                        ),
                    ),
                    mock.patch.object(
                        generation,
                        "_rename_directory_noreplace",
                        side_effect=replace_staging_before_claim,
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "path identity changed"):
                        generation._claim_pinned_directory_at(
                            parent_descriptor,
                            "claimed",
                            "fixture claim",
                        )
                self.assertEqual(
                    {path.name for path in (parent / "claimed").iterdir()},
                    {"foreign.txt"},
                )
            finally:
                os.close(parent_descriptor)


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
        after_generation_hook=None,
        after_atomic_json_hook=None,
    ):
        runtime = self._runtime()
        pipe = SimpleNamespace(unet=_FakeUNet())
        initial_verifications = [self._verification(i) for i in (1, 2, 3)]
        real_atomic_json = generation.atomic_create_json
        real_commit_terminal = generation._commit_prepared_terminal

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
            if phase == "after_generation" and after_generation_hook is not None:
                after_generation_hook()
            return {"phase": phase}

        def atomic_json(
            path,
            value,
            *,
            directory_descriptor=None,
            staging_directory_descriptor=None,
            ownership_candidates=None,
        ):
            if trace_atomic_json:
                events.append(f"write:{Path(path).name}")
            identity = real_atomic_json(
                path,
                value,
                directory_descriptor=directory_descriptor,
                staging_directory_descriptor=staging_directory_descriptor,
                ownership_candidates=ownership_candidates,
            )
            if after_atomic_json_hook is not None:
                after_atomic_json_hook(path, value, directory_descriptor, identity)
            return identity

        def commit_terminal(prepared):
            identity = real_commit_terminal(prepared)
            if trace_atomic_json:
                events.append(f"write:{prepared.target_name}")
            return identity

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
            mock.patch.object(
                generation,
                "_commit_prepared_terminal",
                side_effect=commit_terminal,
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

    def test_parent_chain_helper_return_cancellation_closes_all_descriptors(self):
        real_chain = generation._open_pinned_directory_chain
        captured = []

        def open_then_cancel(*args, **kwargs):
            result = real_chain(*args, **kwargs)
            captured.extend(
                [
                    result[0],
                    *(descriptor for _, _, descriptor, _ in result[1]),
                ]
            )
            raise KeyboardInterrupt("simulated parent-chain return cancellation")

        with (
            mock.patch.object(
                generation,
                "_open_pinned_directory_chain",
                side_effect=open_then_cancel,
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "parent-chain return"),
        ):
            generation._run_authorized_engineering_generation(self.validated)

        self.assertGreaterEqual(len(captured), 2)
        for descriptor in captured:
            with self.assertRaises(OSError) as closed:
                os.fstat(descriptor)
            self.assertEqual(closed.exception.errno, errno.EBADF)
        self.assertFalse(self.output.exists())

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

    def test_failure_receipt_follows_runtime_and_model_cleanup(self):
        events = []
        with self.assertRaisesRegex(RuntimeError, "task failed"):
            self._run(
                events,
                task_error=RuntimeError("task failed"),
                trace_atomic_json=True,
            )
        self.assertLess(events.index("runtime_cleanup"), events.index("write:failure.json"))
        self.assertLess(events.index("model_cleanup"), events.index("write:failure.json"))
        self.assertLess(
            events.index("model_cleanup"),
            events.index("write:model_stage_evidence.json"),
        )
        self.assertLess(
            events.index("write:model_stage_evidence.json"),
            events.index("write:failure.json"),
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
        self.assertLess(events.index("model_cleanup"), events.index("write:failure.json"))
        self.assertLess(
            events.index("write:model_stage_evidence.json"),
            events.index("write:failure.json"),
        )
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

        self.assertLess(events.index("model_cleanup"), events.index("write:failure.json"))
        self.assertLess(
            events.index("write:model_stage_evidence.json"),
            events.index("write:failure.json"),
        )
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

    def test_labeled_secondary_cancellation_blocks_both_generation_terminals(self):
        events = []
        task_error = RuntimeError("generation failed before finalization")
        task_error.secondary_errors = (
            (
                "simulated labeled cleanup",
                KeyboardInterrupt("simulated labeled cleanup cancellation"),
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "generation failed before finalization",
        ) as raised:
            self._run(events, task_error=task_error)

        self.assertIs(raised.exception, task_error)
        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertIn("runtime_cleanup", events)
        self.assertIn("model_cleanup", events)
        self.assertFalse((self.output / "success.json").exists())
        self.assertFalse((self.output / "failure.json").exists())

    def test_model_stage_cleanup_cancellation_blocks_both_generation_terminals(self):
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
            KeyboardInterrupt("simulated initial stage cleanup cancellation"),
            partial_stage,
        )

        with self.assertRaises(
            generation.model_snapshot.ModelStageCreationError
        ) as raised:
            self._run(events, stage_error=stage_error)

        self.assertIs(raised.exception, stage_error)
        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertIn("model_cleanup", events)
        self.assertFalse((self.output / "success.json").exists())
        self.assertFalse((self.output / "failure.json").exists())

    def test_core_failure_with_persistent_normal_restore_releases_wrapper_capture(self):
        events = []
        capture = generation._RuntimeEvidenceCapture()
        original_stderr = sys.stderr
        restore_error = KeyboardInterrupt(
            "simulated persistent wrapper stderr restoration cancellation"
        )
        try:
            with (
                mock.patch.object(
                    generation,
                    "_RuntimeEvidenceCapture",
                    return_value=capture,
                ),
                mock.patch.object(
                    capture,
                    "_restore_python_stderr",
                    return_value=([restore_error], False),
                ) as restore,
                self.assertRaisesRegex(
                    RuntimeError,
                    "runtime evidence finalization also failed",
                ) as raised,
            ):
                self._run(
                    events,
                    task_error=RuntimeError("simulated generation core failure"),
                )

            restore.assert_called_once_with()
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertIs(sys.stderr, original_stderr)
            self.assertFalse(capture._entered)
            self.assertFalse(capture._cleanup_pending)
            self.assertIsNone(capture._stderr_stream)
            self.assertIsNone(capture._stderr_thread)
            self.assertFalse((self.output / "success.json").exists())
            self.assertFalse((self.output / "failure.json").exists())
        finally:
            if sys.stderr is not original_stderr:
                sys.stderr = original_stderr
            if capture._cleanup_pending:
                capture.__exit__(None, None, None)

    def test_success_cleanup_retries_pending_wrapper_capture_before_release(self):
        events = []
        capture = generation._RuntimeEvidenceCapture()
        original_stderr = sys.stderr
        real_restore = capture._restore_python_stderr
        real_direct_restore = capture._restore_python_stderr_direct
        restore_calls = 0
        direct_restore_calls = 0

        def fail_twice_then_restore():
            nonlocal restore_calls
            restore_calls += 1
            if restore_calls <= 2:
                return (
                    [KeyboardInterrupt("simulated repeated wrapper restore cancellation")],
                    False,
                )
            return real_restore()

        def fail_twice_then_direct_restore():
            nonlocal direct_restore_calls
            direct_restore_calls += 1
            if direct_restore_calls <= 2:
                return (
                    [KeyboardInterrupt("simulated repeated direct restore cancellation")],
                    False,
                )
            return real_direct_restore()

        try:
            with (
                mock.patch.object(
                    generation,
                    "_RuntimeEvidenceCapture",
                    return_value=capture,
                ),
                mock.patch.object(
                    capture,
                    "_restore_python_stderr",
                    side_effect=fail_twice_then_restore,
                ),
                mock.patch.object(
                    capture,
                    "_restore_python_stderr_direct",
                    side_effect=fail_twice_then_direct_restore,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "runtime evidence finalization also failed",
                ) as raised,
            ):
                self._run(events)

            self.assertEqual(restore_calls, 3)
            self.assertEqual(direct_restore_calls, 2)
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertIs(sys.stderr, original_stderr)
            self.assertFalse(capture._entered)
            self.assertFalse(capture._cleanup_pending)
            self.assertIsNone(capture._stderr_stream)
            self.assertIsNone(capture._stderr_thread)
            self.assertFalse((self.output / "success.json").exists())
            self.assertFalse((self.output / "failure.json").exists())
        finally:
            if sys.stderr is not original_stderr:
                sys.stderr = original_stderr
            if capture._cleanup_pending:
                capture.__exit__(None, None, None)

    def test_persistent_wrapper_cleanup_failure_hands_off_daemon_capture(self):
        events = []
        capture = generation._RuntimeEvidenceCapture()
        original_stderr = sys.stderr
        captured_stream = None
        try:
            with (
                mock.patch.object(
                    generation,
                    "_RuntimeEvidenceCapture",
                    return_value=capture,
                ),
                mock.patch.object(
                    capture,
                    "_restore_python_stderr",
                    return_value=(
                        [KeyboardInterrupt("simulated persistent normal failure")],
                        False,
                    ),
                ),
                mock.patch.object(
                    capture,
                    "_restore_python_stderr_direct",
                    return_value=(
                        [KeyboardInterrupt("simulated persistent direct failure")],
                        False,
                    ),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "runtime evidence finalization also failed",
                ) as raised,
            ):
                self._run(
                    events,
                    task_error=RuntimeError("simulated generation core failure"),
                )

            pending_errors = [
                error
                for _label, error in raised.exception.secondary_errors
                if hasattr(error, "runtime_capture")
            ]
            self.assertEqual(len(pending_errors), 1)
            self.assertIs(pending_errors[0].runtime_capture, capture)
            captured_stream = capture._stderr_stream
            thread = capture._stderr_thread
            self.assertTrue(capture._cleanup_pending)
            self.assertIs(sys.stderr, captured_stream)
            self.assertFalse(captured_stream.closed)
            self.assertIsNotNone(thread)
            self.assertTrue(thread.daemon)
            self.assertTrue(thread.is_alive())
            self.assertFalse((self.output / "success.json").exists())
            self.assertFalse((self.output / "failure.json").exists())

            capture.__exit__(None, None, None)
            self.assertIs(sys.stderr, original_stderr)
            self.assertTrue(captured_stream.closed)
            self.assertFalse(thread.is_alive())
            self.assertFalse(capture._cleanup_pending)
        finally:
            if sys.stderr is not original_stderr:
                sys.stderr = original_stderr
            if capture._cleanup_pending:
                capture.__exit__(None, None, None)

    def test_enter_rollback_retries_pending_wrapper_capture_before_release(self):
        events = []
        capture = generation._RuntimeEvidenceCapture()
        original_stderr = sys.stderr
        real_restore = capture._restore_python_stderr
        real_direct_restore = capture._restore_python_stderr_direct
        root_logger = logging.getLogger()
        real_add_handler = root_logger.addHandler
        restore_calls = 0
        direct_restore_calls = 0

        def add_then_fail(handler):
            real_add_handler(handler)
            raise RuntimeError("simulated wrapper capture enter failure")

        def fail_twice_then_restore():
            nonlocal restore_calls
            restore_calls += 1
            if restore_calls <= 2:
                return (
                    [KeyboardInterrupt("simulated enter rollback cancellation")],
                    False,
                )
            return real_restore()

        def fail_twice_then_direct_restore():
            nonlocal direct_restore_calls
            direct_restore_calls += 1
            if direct_restore_calls <= 2:
                return (
                    [KeyboardInterrupt("simulated direct enter rollback cancellation")],
                    False,
                )
            return real_direct_restore()

        try:
            with (
                mock.patch.object(
                    generation,
                    "_RuntimeEvidenceCapture",
                    return_value=capture,
                ),
                mock.patch.object(
                    root_logger,
                    "addHandler",
                    side_effect=add_then_fail,
                ),
                mock.patch.object(
                    capture,
                    "_restore_python_stderr",
                    side_effect=fail_twice_then_restore,
                ),
                mock.patch.object(
                    capture,
                    "_restore_python_stderr_direct",
                    side_effect=fail_twice_then_direct_restore,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "runtime evidence finalization also failed",
                ) as raised,
            ):
                self._run(events)

            self.assertEqual(restore_calls, 3)
            self.assertEqual(direct_restore_calls, 2)
            self.assertIn(
                "wrapper capture enter failure",
                str(raised.exception.original_error),
            )
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            self.assertIs(sys.stderr, original_stderr)
            self.assertNotIn(capture._log_handler, root_logger.handlers)
            self.assertFalse(capture._entered)
            self.assertFalse(capture._cleanup_pending)
            self.assertIsNone(capture._stderr_stream)
            self.assertIsNone(capture._stderr_thread)
            self.assertFalse((self.output / "success.json").exists())
            self.assertFalse((self.output / "failure.json").exists())
        finally:
            if sys.stderr is not original_stderr:
                sys.stderr = original_stderr
            if capture._log_handler in root_logger.handlers:
                root_logger.removeHandler(capture._log_handler)
            if capture._cleanup_pending:
                capture.__exit__(None, None, None)

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

    def test_fd_headroom_failure_does_not_claim_output(self):
        with (
            mock.patch.object(
                generation,
                "_require_fd_headroom",
                side_effect=RuntimeError("insufficient descriptor reserve"),
            ),
            self.assertRaisesRegex(RuntimeError, "descriptor reserve"),
        ):
            generation._run_authorized_engineering_generation(self.validated)
        self.assertFalse(self.output.exists())

    def test_concurrent_output_claim_does_not_write_into_foreign_tree(self):
        def competing_claim(
            source_parent_descriptor,
            target_parent_descriptor,
            source_name,
            target_name,
            source_descriptor,
        ):
            del source_parent_descriptor, source_name, source_descriptor
            if target_name == self.output.name:
                os.mkdir(target_name, mode=0o750, dir_fd=target_parent_descriptor)
                (self.output / "foreign.txt").write_text(
                    "foreign claimant\n", encoding="utf-8"
                )
                raise FileExistsError("claim lost")
            raise AssertionError("unexpected directory claim")

        with mock.patch.object(
            generation,
            "_rename_directory_noreplace",
            side_effect=competing_claim,
        ):
            with self.assertRaisesRegex(FileExistsError, "resume/retry"):
                generation._run_authorized_engineering_generation(self.validated)
        self.assertTrue((self.output / "foreign.txt").is_file())
        self.assertFalse((self.output / "attempt.json").exists())
        self.assertFalse((self.output / "failure.json").exists())

    def test_claim_validation_error_is_reconciled_to_attempt_and_failure(self):
        real_claim = generation._claim_pinned_directory_at
        failed = False

        def fail_claimed_output_validation(*args, **kwargs):
            nonlocal failed
            descriptor = real_claim(*args, **kwargs)
            if args[1] == self.output.name and not failed:
                failed = True
                os.close(descriptor)
                raise OSError(errno.EIO, "simulated claimed-directory validation error")
            return descriptor

        with (
            mock.patch.object(
                generation,
                "_rename_directory_noreplace",
                side_effect=OSError(errno.EOPNOTSUPP, "unsupported"),
            ),
            mock.patch.object(
                generation,
                "_claim_pinned_directory_at",
                side_effect=fail_claimed_output_validation,
            ),
            self.assertRaisesRegex(OSError, "claimed-directory validation"),
        ):
            generation._run_authorized_engineering_generation(self.validated)

        self.assertTrue(failed)
        self.assertTrue((self.output / "attempt.json").is_file())
        self.assertTrue((self.output / "failure.json").is_file())
        self.assertFalse((self.output / "success.json").exists())

    def test_interrupted_output_recovery_handoff_closes_its_descriptor(self):
        handed_off = []

        def create_then_report_indeterminate(
            parent_descriptor,
            name,
            _label,
            *,
            ownership_candidates,
            **_kwargs,
        ):
            os.mkdir(name, mode=0o750, dir_fd=parent_descriptor)
            observed = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            ownership_candidates.append(generation._object_identity(observed))
            raise generation.DirectoryClaimIndeterminate(
                "simulated output claim uncertainty"
            )

        def interrupt_recovery(
            parent_descriptor,
            name,
            label,
            *,
            descriptor_guard,
            **_kwargs,
        ):
            descriptor = generation._open_pinned_directory_at(
                parent_descriptor, name, label
            )
            descriptor_guard.append(descriptor)
            handed_off.append(descriptor)
            raise KeyboardInterrupt("simulated recovery handoff interruption")

        with (
            mock.patch.object(
                generation,
                "_claim_pinned_directory_at",
                side_effect=create_then_report_indeterminate,
            ),
            mock.patch.object(
                generation,
                "_open_claimed_directory_with_positive_reconciliation",
                side_effect=interrupt_recovery,
            ),
            self.assertRaisesRegex(
                generation.DirectoryClaimIndeterminate,
                "output claim uncertainty",
            ),
        ):
            generation._run_authorized_engineering_generation(self.validated)

        self.assertEqual(len(handed_off), 1)
        with self.assertRaises(OSError) as raised:
            os.fstat(handed_off[0])
        self.assertEqual(raised.exception.errno, errno.EBADF)
        self.assertFalse((self.output / "attempt.json").exists())
        self.assertFalse((self.output / "failure.json").exists())

    def test_deferred_recovery_cancellation_prevents_attempt_and_failure(self):
        handed_off = []
        deferred = KeyboardInterrupt("simulated deferred recovery cancellation")

        def create_then_report_indeterminate(
            parent_descriptor,
            name,
            _label,
            *,
            ownership_candidates,
            **_kwargs,
        ):
            os.mkdir(name, mode=0o750, dir_fd=parent_descriptor)
            observed = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            ownership_candidates.append(generation._object_identity(observed))
            raise generation.DirectoryClaimIndeterminate(
                "simulated output claim uncertainty"
            )

        def recover_with_deferred_cancellation(
            parent_descriptor,
            name,
            label,
            *,
            descriptor_guard,
            **_kwargs,
        ):
            descriptor = generation._open_pinned_directory_at(
                parent_descriptor,
                name,
                label,
                descriptor_guard=descriptor_guard,
            )
            handed_off.append(descriptor)
            return descriptor, (deferred, deferred.__traceback__)

        with (
            mock.patch.object(
                generation,
                "_claim_pinned_directory_at",
                side_effect=create_then_report_indeterminate,
            ),
            mock.patch.object(
                generation,
                "_open_claimed_directory_with_positive_reconciliation",
                side_effect=recover_with_deferred_cancellation,
            ),
            self.assertRaises(
                generation.DirectoryClaimIndeterminate
            ) as raised,
        ):
            generation._run_authorized_engineering_generation(self.validated)

        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertEqual(len(handed_off), 1)
        with self.assertRaises(OSError) as closed:
            os.fstat(handed_off[0])
        self.assertEqual(closed.exception.errno, errno.EBADF)
        self.assertFalse((self.output / "attempt.json").exists())
        self.assertFalse((self.output / "failure.json").exists())

    def test_interrupted_initial_output_claim_handoff_closes_its_descriptor(self):
        real_claim = generation._claim_pinned_directory_at
        handed_off = []

        def cancel_after_output_claim(*args, **kwargs):
            descriptor = real_claim(*args, **kwargs)
            if args[1] == self.output.name:
                handed_off.append(descriptor)
                raise KeyboardInterrupt(
                    "simulated initial output claim handoff interruption"
                )
            return descriptor

        with (
            mock.patch.object(
                generation,
                "_claim_pinned_directory_at",
                side_effect=cancel_after_output_claim,
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "initial output claim"),
        ):
            generation._run_authorized_engineering_generation(self.validated)

        self.assertEqual(len(handed_off), 1)
        with self.assertRaises(OSError) as raised:
            os.fstat(handed_off[0])
        self.assertEqual(raised.exception.errno, errno.EBADF)
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "attempt.json").exists())
        self.assertFalse((self.output / "failure.json").exists())
        self.assertFalse((self.output / "success.json").exists())

    def test_interrupted_records_claim_handoff_closes_without_failure_terminal(self):
        real_claim = generation._claim_pinned_directory_at
        handed_off = []

        def cancel_after_records_claim(*args, **kwargs):
            descriptor = real_claim(*args, **kwargs)
            if args[1] == "records":
                handed_off.append(descriptor)
                raise KeyboardInterrupt(
                    "simulated records claim handoff interruption"
                )
            return descriptor

        with (
            mock.patch.object(
                generation,
                "_claim_pinned_directory_at",
                side_effect=cancel_after_records_claim,
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "records claim handoff"),
        ):
            generation._run_authorized_engineering_generation(self.validated)

        self.assertEqual(len(handed_off), 1)
        with self.assertRaises(OSError) as raised:
            os.fstat(handed_off[0])
        self.assertEqual(raised.exception.errno, errno.EBADF)
        self.assertTrue((self.output / "attempt.json").is_file())
        self.assertTrue((self.output / "records").is_dir())
        self.assertFalse((self.output / "failure.json").exists())
        self.assertFalse((self.output / "success.json").exists())

    def test_attempt_cleanup_cancellation_stops_before_records_claim(self):
        real_remove = generation._remove_matching_entry
        cancelled = False

        def remove_then_cancel(directory_descriptor, name, identity):
            nonlocal cancelled
            result = real_remove(directory_descriptor, name, identity)
            if name.startswith(".attempt.json.") and not cancelled:
                cancelled = True
                raise KeyboardInterrupt("simulated attempt cleanup cancellation")
            return result

        with (
            mock.patch.object(
                generation,
                "_remove_matching_entry",
                side_effect=remove_then_cancel,
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "attempt cleanup"),
        ):
            self._run([])

        self.assertTrue(cancelled)
        self.assertTrue((self.output / "attempt.json").is_file())
        self.assertFalse((self.output / "records").exists())
        self.assertFalse((self.output / "failure.json").exists())
        self.assertFalse((self.output / "success.json").exists())

    def test_final_mkdir_cancellation_remains_unowned_and_indeterminate(self):
        real_mkdir = generation.os.mkdir
        cancelled = False

        def create_output_then_cancel(path, *args, **kwargs):
            nonlocal cancelled
            result = real_mkdir(path, *args, **kwargs)
            if path == self.output.name and not cancelled:
                cancelled = True
                raise KeyboardInterrupt(
                    "simulated cancellation after output directory mkdir"
                )
            return result

        with (
            mock.patch.object(
                generation,
                "_rename_directory_noreplace",
                side_effect=OSError(errno.EOPNOTSUPP, "unsupported"),
            ),
            mock.patch.object(
                generation.os,
                "mkdir",
                side_effect=create_output_then_cancel,
            ),
            self.assertRaisesRegex(
                generation.DirectoryClaimIndeterminate,
                "unknown outcome",
            ),
        ):
            generation._run_authorized_engineering_generation(self.validated)

        self.assertTrue(cancelled)
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "attempt.json").exists())
        self.assertFalse((self.output / "failure.json").exists())
        self.assertFalse((self.output / "success.json").exists())
        self.assertTrue((self.output.parent / f".{self.output.name}.claim").is_file())
        staging = [
            path
            for path in self.output.parent.iterdir()
            if path.name.startswith(f".{self.output.name}.")
            and path.name.endswith(".claim")
            and path.is_dir()
        ]
        self.assertEqual(len(staging), 1)

    def test_output_directory_replacement_blocks_terminal_publication(self):
        events = []
        displaced = self.root / "displaced-engineering-run"

        def replace_output_directory():
            self.output.rename(displaced)
            self.output.mkdir()
            (self.output / "foreign.txt").write_text(
                "foreign replacement\n", encoding="utf-8"
            )

        with self.assertRaises(RuntimeError) as raised:
            self._run(
                events,
                after_generation_hook=replace_output_directory,
            )

        secondary = getattr(raised.exception, "secondary_errors", ())
        self.assertTrue(
            any("path identity changed" in str(error) for _, error in secondary)
        )
        self.assertEqual(
            {path.name for path in self.output.iterdir()}, {"foreign.txt"}
        )
        self.assertFalse((displaced / "success.json").exists())
        self.assertFalse((displaced / "failure.json").exists())

    def test_output_parent_replacement_blocks_terminal_publication(self):
        events = []
        output_parent = self.output.parent
        displaced_parent = self.root / "displaced-adaptive-oracle"

        def replace_output_parent():
            output_parent.rename(displaced_parent)
            output_parent.mkdir()
            foreign_output = output_parent / self.output.name
            foreign_output.mkdir()
            (foreign_output / "foreign.txt").write_text(
                "foreign replacement\n", encoding="utf-8"
            )

        with self.assertRaises(RuntimeError) as raised:
            self._run(
                events,
                after_generation_hook=replace_output_parent,
            )

        secondary = getattr(raised.exception, "secondary_errors", ())
        self.assertTrue(
            any("path identity changed" in str(error) for _, error in secondary)
        )
        self.assertEqual(
            {path.name for path in self.output.iterdir()}, {"foreign.txt"}
        )
        displaced_output = displaced_parent / self.output.name
        self.assertFalse((displaced_output / "success.json").exists())
        self.assertFalse((displaced_output / "failure.json").exists())

    def test_record_inode_replacement_blocks_terminal_publication(self):
        first_task = self.design["tasks"][0]["task_id"]

        def replace_record_inode():
            target = self.output / "records" / f"{first_task}.json"
            replacement = self.root / "replacement-record.json"
            replacement.write_bytes(target.read_bytes())
            os.replace(replacement, target)

        with self.assertRaises(RuntimeError) as raised:
            self._run(
                [],
                after_generation_hook=replace_record_inode,
            )

        secondary = getattr(raised.exception, "secondary_errors", ())
        observed_errors = [
            getattr(raised.exception, "original_error", raised.exception),
            *(error for _, error in secondary),
        ]
        self.assertTrue(
            any(
                "changed" in str(error) or "descriptor snapshot" in str(error)
                for error in observed_errors
            )
        )
        self.assertFalse((self.output / "success.json").exists())
        self.assertTrue((self.output / "failure.json").is_file())

    def test_success_fsync_error_does_not_reverse_committed_terminal(self):
        real_fsync = generation.os.fsync
        failed = False

        def fail_success_directory_fsync(descriptor):
            nonlocal failed
            if (
                not failed
                and os.path.isdir(f"/proc/self/fd/{descriptor}")
                and (self.output / "success.json").exists()
            ):
                failed = True
                raise OSError(errno.EIO, "simulated terminal fsync failure")
            return real_fsync(descriptor)

        with (
            mock.patch.object(
                generation.os,
                "fsync",
                side_effect=fail_success_directory_fsync,
            ),
            self.assertRaisesRegex(OSError, "terminal fsync failure"),
        ):
            self._run([])

        self.assertTrue(failed)
        self.assertTrue((self.output / "success.json").is_file())
        self.assertFalse((self.output / "failure.json").exists())
        staging_intents = list(self.output.parent.glob(".success.json.*.tmp"))
        self.assertEqual(len(staging_intents), 1)
        self.assertTrue(os.path.samefile(staging_intents[0], self.output / "success.json"))

    def test_nonterminal_publication_indeterminate_still_gets_failure_terminal(self):
        real_reconcile = generation._link_with_positive_reconciliation
        failed = False

        def fail_first_png(descriptor, directory_descriptor, name, identity):
            nonlocal failed
            if name.endswith(".png") and not failed:
                failed = True
                error = generation.AtomicPublicationIndeterminate(
                    "simulated nonterminal publication uncertainty"
                )
                error.target_name = name
                error.source_identity = identity
                raise error
            return real_reconcile(
                descriptor,
                directory_descriptor,
                name,
                identity,
            )

        with (
            mock.patch.object(
                generation,
                "_link_with_positive_reconciliation",
                side_effect=fail_first_png,
            ),
            self.assertRaisesRegex(
                generation.AtomicPublicationIndeterminate,
                "nonterminal publication uncertainty",
            ),
        ):
            self._run([])

        self.assertTrue(failed)
        self.assertTrue((self.output / "failure.json").is_file())
        self.assertFalse((self.output / "success.json").exists())

    def test_success_prepare_return_cancellation_discards_private_output(self):
        real_prepare = generation._prepare_terminal_json
        captured = []

        def prepare_then_cancel(path, *args, **kwargs):
            prepared = real_prepare(path, *args, **kwargs)
            if Path(path).name == "success.json":
                captured.append((prepared, prepared.descriptor))
                raise KeyboardInterrupt("simulated success prepare return cancellation")
            return prepared

        with (
            mock.patch.object(
                generation,
                "_prepare_terminal_json",
                side_effect=prepare_then_cancel,
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "prepare return"),
        ):
            self._run([])

        self.assertEqual(len(captured), 1)
        prepared, descriptor = captured[0]
        self.assertEqual(prepared.descriptor, -1)
        with self.assertRaises(OSError) as closed:
            os.fstat(descriptor)
        self.assertEqual(closed.exception.errno, errno.EBADF)
        self.assertFalse((self.output / "success.json").exists())
        self.assertFalse((self.output / "failure.json").exists())
        self.assertFalse(
            any(
                path.name.startswith(".success.json.")
                and path.name.endswith(".tmp")
                for path in self.output.parent.iterdir()
            )
        )

    def test_success_discard_cleanup_cancellation_blocks_failure_terminal(self):
        real_prepare = generation._prepare_terminal_json
        real_inventory = generation._verify_directory_inventory
        real_remove = generation._remove_matching_entry
        captured = []
        cleanup_cancelled = False

        def capture_prepare(path, *args, **kwargs):
            prepared = real_prepare(path, *args, **kwargs)
            if Path(path).name == "success.json":
                captured.append((prepared, prepared.descriptor))
            return prepared

        def fail_precommit_inventory(descriptor, expected, label):
            if label == "pre-success engineering output":
                raise RuntimeError("simulated success pre-commit error")
            return real_inventory(descriptor, expected, label)

        def remove_then_cancel(directory_descriptor, name, identity):
            nonlocal cleanup_cancelled
            result = real_remove(directory_descriptor, name, identity)
            if name.startswith(".success.json.") and not cleanup_cancelled:
                cleanup_cancelled = True
                raise KeyboardInterrupt("simulated success discard cancellation")
            return result

        with (
            mock.patch.object(
                generation,
                "_prepare_terminal_json",
                side_effect=capture_prepare,
            ),
            mock.patch.object(
                generation,
                "_verify_directory_inventory",
                side_effect=fail_precommit_inventory,
            ),
            mock.patch.object(
                generation,
                "_remove_matching_entry",
                side_effect=remove_then_cancel,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "success pre-commit error",
            ) as raised,
        ):
            self._run([])

        self.assertTrue(cleanup_cancelled)
        self.assertTrue(
            generation._exception_contains_cancellation(raised.exception)
        )
        self.assertEqual(len(captured), 1)
        prepared, descriptor = captured[0]
        self.assertEqual(prepared.descriptor, -1)
        with self.assertRaises(OSError) as closed:
            os.fstat(descriptor)
        self.assertEqual(closed.exception.errno, errno.EBADF)
        self.assertFalse((self.output / "success.json").exists())
        self.assertFalse((self.output / "failure.json").exists())

    def test_commit_start_cancellation_discards_generation_success_prepare(self):
        real_mark = generation._mark_prepared_terminal_commit_started
        captured = []

        def mark_then_cancel(prepared):
            captured.append((prepared, prepared.descriptor))
            real_mark(prepared)
            raise KeyboardInterrupt("simulated generation commit-start cancellation")

        with (
            mock.patch.object(
                generation,
                "_mark_prepared_terminal_commit_started",
                side_effect=mark_then_cancel,
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "commit-start"),
        ):
            self._run([])

        self.assertEqual(len(captured), 1)
        prepared, descriptor = captured[0]
        self.assertFalse(prepared.commit_started)
        self.assertEqual(prepared.descriptor, -1)
        with self.assertRaises(OSError) as closed:
            os.fstat(descriptor)
        self.assertEqual(closed.exception.errno, errno.EBADF)
        self.assertFalse((self.output / "success.json").exists())
        self.assertFalse((self.output / "failure.json").exists())

    def test_committed_state_cancellation_closes_generation_success_prepare(self):
        real_mark = generation._mark_prepared_terminal_committed
        captured = []

        def mark_then_cancel(prepared):
            captured.append((prepared, prepared.descriptor))
            real_mark(prepared)
            raise KeyboardInterrupt("simulated generation committed-state cancellation")

        with (
            mock.patch.object(
                generation,
                "_mark_prepared_terminal_committed",
                side_effect=mark_then_cancel,
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "committed-state"),
        ):
            self._run([])

        self.assertEqual(len(captured), 1)
        prepared, descriptor = captured[0]
        self.assertTrue(prepared.committed)
        self.assertTrue(prepared.indeterminate)
        self.assertEqual(prepared.descriptor, -1)
        with self.assertRaises(OSError) as closed:
            os.fstat(descriptor)
        self.assertEqual(closed.exception.errno, errno.EBADF)
        self.assertTrue((self.output / "success.json").is_file())
        self.assertFalse((self.output / "failure.json").exists())

    def test_post_commit_snapshot_error_ignores_stale_success_lookup(self):
        real_snapshot = generation._stable_regular_snapshot
        real_exists = generation._entry_exists
        snapshot_failed = False
        success_lookups = 0

        def fail_committed_success_snapshot(
            descriptor, directory_descriptor, name, label
        ):
            nonlocal snapshot_failed
            if name == "success.json" and not snapshot_failed:
                snapshot_failed = True
                raise RuntimeError("simulated post-commit snapshot failure")
            return real_snapshot(descriptor, directory_descriptor, name, label)

        def stale_success_lookup(directory_descriptor, name):
            nonlocal success_lookups
            if name == "success.json":
                success_lookups += 1
                return False
            return real_exists(directory_descriptor, name)

        with (
            mock.patch.object(
                generation,
                "_stable_regular_snapshot",
                side_effect=fail_committed_success_snapshot,
            ),
            mock.patch.object(
                generation,
                "_entry_exists",
                side_effect=stale_success_lookup,
            ),
            self.assertRaisesRegex(RuntimeError, "post-commit snapshot failure"),
        ):
            self._run([])

        self.assertTrue(snapshot_failed)
        self.assertEqual(success_lookups, 0)
        self.assertTrue((self.output / "success.json").is_file())
        self.assertFalse((self.output / "failure.json").exists())

    def test_positive_link_then_helper_cancellation_cannot_publish_failure(self):
        real_reconcile = generation._link_with_positive_reconciliation
        cancelled = False

        def cancel_after_positive_reconciliation(
            descriptor, directory_descriptor, name, identity
        ):
            nonlocal cancelled
            result = real_reconcile(
                descriptor,
                directory_descriptor,
                name,
                identity,
            )
            if name == "success.json" and not cancelled:
                cancelled = True
                raise KeyboardInterrupt(
                    "simulated cancellation after positive terminal reconciliation"
                )
            return result

        with (
            mock.patch.object(
                generation,
                "_link_with_positive_reconciliation",
                side_effect=cancel_after_positive_reconciliation,
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "positive terminal"),
        ):
            self._run([])

        self.assertTrue(cancelled)
        self.assertTrue((self.output / "success.json").is_file())
        self.assertFalse((self.output / "failure.json").exists())

    def test_ambiguous_success_commit_wins_without_failure_receipt(self):
        real_link = generation._link_descriptor
        reported = False

        def link_success_then_report_error(descriptor, directory_descriptor, name):
            nonlocal reported
            real_link(descriptor, directory_descriptor, name)
            if name == "success.json" and not reported:
                reported = True
                raise OSError(errno.EIO, "simulated committed success error")

        with mock.patch.object(
            generation,
            "_link_descriptor",
            side_effect=link_success_then_report_error,
        ):
            result, _runtime = self._run([])

        self.assertTrue(reported)
        self.assertEqual(result["status"], "complete_generation_only_unscored")
        self.assertTrue((self.output / "success.json").is_file())
        self.assertFalse((self.output / "failure.json").exists())

    def test_post_link_cancellation_cannot_reverse_generation_success(self):
        real_link = generation._link_descriptor
        cancelled = False

        def link_success_then_cancel(descriptor, directory_descriptor, name):
            nonlocal cancelled
            real_link(descriptor, directory_descriptor, name)
            if name == "success.json" and not cancelled:
                cancelled = True
                raise KeyboardInterrupt("simulated cancellation after success link")

        with (
            mock.patch.object(
                generation,
                "_link_descriptor",
                side_effect=link_success_then_cancel,
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "after success link"),
        ):
            self._run([])

        self.assertTrue(cancelled)
        self.assertTrue((self.output / "success.json").is_file())
        self.assertFalse((self.output / "failure.json").exists())

    def test_success_publication_close_cancellation_is_propagated(self):
        real_close = generation._PublishedArtifacts.close
        real_os_close = generation.os.close
        pinned_descriptors = []
        target_descriptor = -1
        cancelled = False

        def close_with_inventory(publications):
            nonlocal target_descriptor
            pinned_descriptors.extend(row[0] for row in publications._rows)
            target_descriptor = publications._rows[-1][0]
            return real_close(publications)

        def cancel_before_close(descriptor):
            nonlocal cancelled
            if descriptor == target_descriptor and not cancelled:
                cancelled = True
                raise KeyboardInterrupt(
                    "simulated success publication pre-close cancellation"
                )
            return real_os_close(descriptor)

        try:
            with (
                mock.patch.object(
                    generation._PublishedArtifacts,
                    "close",
                    autospec=True,
                    side_effect=close_with_inventory,
                ),
                mock.patch.object(
                    generation.os,
                    "close",
                    side_effect=cancel_before_close,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "generation descriptor finalization",
                ) as raised,
            ):
                self._run([])

            self.assertTrue(cancelled)
            self.assertGreater(len(pinned_descriptors), 1)
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            os.fstat(target_descriptor)
            for descriptor in pinned_descriptors:
                if descriptor == target_descriptor:
                    continue
                with self.assertRaises(OSError) as closed:
                    os.fstat(descriptor)
                self.assertEqual(closed.exception.errno, errno.EBADF)
            self.assertTrue((self.output / "success.json").is_file())
            self.assertFalse((self.output / "failure.json").exists())
        finally:
            if target_descriptor >= 0:
                real_os_close(target_descriptor)

    def test_failure_publication_close_cancellation_attaches_to_primary(self):
        real_close = generation._PublishedArtifacts.close
        real_os_close = generation.os.close
        target_descriptor = -1
        sentinel_descriptor = -1
        sentinel_descriptors = []
        cancelled = False

        def close_with_inventory(publications):
            nonlocal target_descriptor
            target_descriptor = publications._rows[-1][0]
            return real_close(publications)

        def close_reuse_then_cancel(descriptor):
            nonlocal cancelled, sentinel_descriptor
            if descriptor == target_descriptor and not cancelled:
                cancelled = True
                real_os_close(descriptor)
                while descriptor not in sentinel_descriptors:
                    sentinel_descriptors.append(os.open("/dev/null", os.O_RDONLY))
                sentinel_descriptor = descriptor
                raise KeyboardInterrupt(
                    "simulated failure publication post-close cancellation"
                )
            return real_os_close(descriptor)

        try:
            with (
                mock.patch.object(
                    generation._PublishedArtifacts,
                    "close",
                    autospec=True,
                    side_effect=close_with_inventory,
                ),
                mock.patch.object(
                    generation.os,
                    "close",
                    side_effect=close_reuse_then_cancel,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "generation core failure",
                ) as raised,
            ):
                self._run(
                    [],
                    task_error=RuntimeError("simulated generation core failure"),
                )

            self.assertTrue(cancelled)
            self.assertGreaterEqual(target_descriptor, 0)
            self.assertEqual(sentinel_descriptor, target_descriptor)
            self.assertTrue(
                generation._exception_contains_cancellation(raised.exception)
            )
            os.fstat(sentinel_descriptor)
            self.assertFalse((self.output / "success.json").exists())
            self.assertTrue((self.output / "failure.json").is_file())
        finally:
            for descriptor in sentinel_descriptors:
                real_os_close(descriptor)

    def test_pre_success_same_inode_record_rewrite_blocks_success(self):
        first_task = self.design["tasks"][0]["task_id"]
        target = self.output / "records" / f"{first_task}.json"
        mutated = False

        def rewrite_record_before_success():
            nonlocal mutated
            if mutated:
                return
            mutated = True
            before = target.stat()
            payload = target.read_bytes()
            replacement = (b"X" if payload[:1] != b"X" else b"Y") + payload[1:]
            os.chmod(target, 0o640)
            target.write_bytes(replacement)
            os.chmod(target, before.st_mode & 0o777)
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))

        with self.assertRaises(RuntimeError):
            self._run([], after_generation_hook=rewrite_record_before_success)

        self.assertTrue(mutated)
        self.assertFalse((self.output / "success.json").exists())
        self.assertTrue((self.output / "failure.json").is_file())

    def test_success_receipt_is_the_final_publication(self):
        events = []
        success, _runtime = self._run(events, trace_atomic_json=True)
        self.assertEqual(events[-1], "write:success.json")
        self.assertEqual(success["status"], "complete_generation_only_unscored")
        self.assertTrue((self.output / "success.json").is_file())
        self.assertFalse((self.output / "failure.json").exists())

    def test_failure_receipt_rewrite_during_commit_is_detected(self):
        real_link = generation._link_descriptor
        mutated = False

        def link_then_rewrite_failure(descriptor, directory_descriptor, name):
            nonlocal mutated
            real_link(descriptor, directory_descriptor, name)
            if name != "failure.json" or mutated:
                return
            mutated = True
            target = self.output / name
            before = target.stat()
            payload = target.read_bytes()
            os.chmod(target, 0o640)
            target.write_bytes((b"X" if payload[:1] != b"X" else b"Y") + payload[1:])
            os.chmod(target, before.st_mode & 0o777)
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))

        with (
            mock.patch.object(
                generation,
                "_link_descriptor",
                side_effect=link_then_rewrite_failure,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "failure receipt creation also failed",
            ),
        ):
            self._run([], task_error=RuntimeError("fixture task failure"))

        self.assertTrue(mutated)
        self.assertTrue((self.output / "failure.json").is_file())
        self.assertFalse((self.output / "success.json").exists())

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
