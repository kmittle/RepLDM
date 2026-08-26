from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "eval-pipeline" / "launch_adaptive_oracle_tool.py"


def _load_launcher():
    name = f"launch_adaptive_oracle_tool_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load launcher test module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


launcher = _load_launcher()


def _run_git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


class GitAuthorizationFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        _run_git(root, "init", "--quiet")
        _run_git(root, "config", "user.name", "Launcher Tests")
        _run_git(root, "config", "user.email", "launcher@example.invalid")
        self.source_bytes = {}
        for index, (name, relative_path) in enumerate(launcher.SOURCE_PATHS.items()):
            value = f"trusted {index} {name}\n".encode("ascii")
            self.source_bytes[name] = value
            _write(root / relative_path, value)
        for relative_path in launcher.CPU_TEST_SOURCES:
            if not (root / relative_path).exists():
                _write(root / relative_path, f"reviewed:{relative_path}\n".encode("ascii"))
        _run_git(root, "add", ".")
        _run_git(root, "commit", "--quiet", "-m", "reviewed implementation")
        self.implementation_commit = _run_git(root, "rev-parse", "HEAD")
        self.cpu_audit_bytes = launcher._canonical_json_bytes(
            {
                "implementation": {"commit": self.implementation_commit},
                "status": "cpu-audited",
            }
        )
        _write(root / launcher.CPU_AUDIT_PATH, self.cpu_audit_bytes)
        _run_git(root, "add", launcher.CPU_AUDIT_PATH)
        _run_git(root, "commit", "--quiet", "-m", "record CPU audit")
        self.cpu_audit_commit = _run_git(root, "rev-parse", "HEAD")
        self.reviewed_commit = self.cpu_audit_commit
        self.authorization = self.make_authorization(self.reviewed_commit)
        self._commit_authorization(self.authorization)

    def _commit_authorization(self, value: dict, *, extra_file: bool = False) -> None:
        self.authorization = value
        self.authorization_raw = launcher._canonical_json_bytes(self.authorization)
        _write(self.root / launcher.AUTHORIZATION_PATH, self.authorization_raw)
        _run_git(self.root, "add", launcher.AUTHORIZATION_PATH)
        if extra_file:
            _write(self.root / "extra.txt", b"not authorized\n")
            _run_git(self.root, "add", "extra.txt")
        _run_git(self.root, "commit", "--quiet", "-m", "authorize")
        self.authorization_commit = _run_git(self.root, "rev-parse", "HEAD")
        self.authorization_sha256 = hashlib.sha256(self.authorization_raw).hexdigest()

    def make_authorization(self, reviewed_commit: str) -> dict:
        return {
            "authorization": {
                "reviewed_commit": reviewed_commit,
                "source_registration": {
                    "path": launcher.SOURCE_PATHS["registration"],
                    "sha256": hashlib.sha256(
                        self.source_bytes["registration"]
                    ).hexdigest(),
                },
                "cpu_audit": {
                    "path": launcher.CPU_AUDIT_PATH,
                    "sha256": hashlib.sha256(self.cpu_audit_bytes).hexdigest(),
                },
            },
            "sources": {
                name: {
                    "path": path,
                    "sha256": hashlib.sha256(self.source_bytes[name]).hexdigest(),
                }
                for name, path in launcher.SOURCE_PATHS.items()
            },
        }

    def amend_authorization(self, value: dict, *, extra_file: bool = False) -> None:
        _run_git(self.root, "reset", "--hard", self.reviewed_commit)
        self._commit_authorization(value, extra_file=extra_file)

    def verify(self):
        return launcher._verify_authorization(
            self.root,
            authorization_commit=self.authorization_commit,
            authorization_sha256=self.authorization_sha256,
        )


class AuthorizationTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        self.fixture = GitAuthorizationFixture(self.repo)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_explicit_canonical_three_commit_authorization(self):
        implementation, reviewed, authorization, raw, pairs, inputs = (
            self.fixture.verify()
        )
        self.assertEqual(implementation, self.fixture.implementation_commit)
        self.assertEqual(reviewed, self.fixture.reviewed_commit)
        self.assertEqual(authorization, self.fixture.authorization)
        self.assertEqual(raw, self.fixture.authorization_raw)
        self.assertEqual(set(pairs), set(launcher.SOURCE_PATHS))
        self.assertEqual(inputs["cpu_audit"]["path"], launcher.CPU_AUDIT_PATH)
        snapshots, input_bytes = launcher._snapshots(
            self.repo, reviewed, pairs, inputs
        )
        self.assertEqual(input_bytes["cpu_audit"], self.fixture.cpu_audit_bytes)
        self.assertEqual(
            input_bytes["exclusion_inventory"],
            self.fixture.source_bytes["exclusion_inventory"],
        )
        self.assertEqual(
            snapshots["generate_adaptive_oracle_engineering"]["bytes"],
            self.fixture.source_bytes["generator"],
        )

    def test_rejects_cpu_audit_commit_with_an_extra_path(self):
        _run_git(self.repo, "reset", "--hard", self.fixture.implementation_commit)
        _write(self.repo / launcher.CPU_AUDIT_PATH, self.fixture.cpu_audit_bytes)
        _write(self.repo / "extra-audit-input.txt", b"not reviewed\n")
        _run_git(
            self.repo,
            "add",
            launcher.CPU_AUDIT_PATH,
            "extra-audit-input.txt",
        )
        _run_git(self.repo, "commit", "--quiet", "-m", "invalid CPU audit")
        cpu_audit_commit = _run_git(self.repo, "rev-parse", "HEAD")
        self.fixture._commit_authorization(
            self.fixture.make_authorization(cpu_audit_commit)
        )
        with self.assertRaisesRegex(RuntimeError, "CPU-audit.*exactly one path"):
            self.fixture.verify()

    def test_rejects_commit_that_adds_the_wrong_cpu_audit_path(self):
        _run_git(self.repo, "reset", "--hard", self.fixture.implementation_commit)
        _write(self.repo / "not-the-cpu-audit.json", self.fixture.cpu_audit_bytes)
        _run_git(self.repo, "add", "not-the-cpu-audit.json")
        _run_git(self.repo, "commit", "--quiet", "-m", "wrong CPU audit path")
        cpu_audit_commit = _run_git(self.repo, "rev-parse", "HEAD")
        self.fixture._commit_authorization(
            self.fixture.make_authorization(cpu_audit_commit)
        )
        with self.assertRaisesRegex(RuntimeError, "CPU-audit.*must only add"):
            self.fixture.verify()

    def test_rejects_cpu_audit_merge_commit(self):
        _run_git(self.repo, "reset", "--hard", self.fixture.implementation_commit)
        _run_git(self.repo, "checkout", "--quiet", "-b", "audit-side")
        _write(self.repo / "side.txt", b"independent side\n")
        _run_git(self.repo, "add", "side.txt")
        _run_git(self.repo, "commit", "--quiet", "-m", "side parent")
        _run_git(
            self.repo,
            "checkout",
            "--quiet",
            "-b",
            "audit-main",
            self.fixture.implementation_commit,
        )
        _write(self.repo / launcher.CPU_AUDIT_PATH, self.fixture.cpu_audit_bytes)
        _run_git(self.repo, "add", launcher.CPU_AUDIT_PATH)
        _run_git(self.repo, "commit", "--quiet", "-m", "CPU audit parent")
        _run_git(
            self.repo,
            "merge",
            "--quiet",
            "--no-ff",
            "audit-side",
            "-m",
            "invalid audit merge",
        )
        cpu_audit_commit = _run_git(self.repo, "rev-parse", "HEAD")
        self.fixture._commit_authorization(
            self.fixture.make_authorization(cpu_audit_commit)
        )
        with self.assertRaisesRegex(RuntimeError, "CPU-audit.*exactly one parent"):
            self.fixture.verify()

    def test_rejects_authorization_merge_commit(self):
        _run_git(self.repo, "reset", "--hard", self.fixture.reviewed_commit)
        _run_git(self.repo, "checkout", "--quiet", "-b", "authorization-side")
        _write(self.repo / "authorization-side.txt", b"independent side\n")
        _run_git(self.repo, "add", "authorization-side.txt")
        _run_git(self.repo, "commit", "--quiet", "-m", "authorization side")
        _run_git(
            self.repo,
            "checkout",
            "--quiet",
            "-b",
            "authorization-main",
            self.fixture.reviewed_commit,
        )
        value = self.fixture.make_authorization(self.fixture.reviewed_commit)
        self.fixture._commit_authorization(value)
        _run_git(
            self.repo,
            "merge",
            "--quiet",
            "--no-ff",
            "authorization-side",
            "-m",
            "invalid authorization merge",
        )
        self.fixture.authorization_commit = _run_git(self.repo, "rev-parse", "HEAD")
        with self.assertRaisesRegex(RuntimeError, "authorization.*exactly one parent"):
            self.fixture.verify()

    def test_rejects_external_authorization_hash_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "external value"):
            launcher._verify_authorization(
                self.repo,
                authorization_commit=self.fixture.authorization_commit,
                authorization_sha256="0" * 64,
            )

    def test_rejects_authorization_parent_mismatch(self):
        value = self.fixture.make_authorization("0" * 40)
        self.fixture.amend_authorization(value)
        with self.assertRaisesRegex(RuntimeError, "sole parent"):
            self.fixture.verify()

    def test_rejects_noncanonical_authorization_json(self):
        value = self.fixture.make_authorization(self.fixture.reviewed_commit)
        raw = json.dumps(value, indent=2, sort_keys=True).encode("ascii")
        _run_git(self.repo, "reset", "--hard", self.fixture.reviewed_commit)
        _write(self.repo / launcher.AUTHORIZATION_PATH, raw)
        _run_git(self.repo, "add", launcher.AUTHORIZATION_PATH)
        _run_git(self.repo, "commit", "--quiet", "-m", "noncanonical")
        commit = _run_git(self.repo, "rev-parse", "HEAD")
        with self.assertRaisesRegex(RuntimeError, "strict canonical JSON"):
            launcher._verify_authorization(
                self.repo,
                authorization_commit=commit,
                authorization_sha256=hashlib.sha256(raw).hexdigest(),
            )

    def test_rejects_authorization_commit_with_extra_file(self):
        self.fixture.amend_authorization(
            self.fixture.make_authorization(self.fixture.reviewed_commit),
            extra_file=True,
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one path"):
            self.fixture.verify()

    def test_rejects_authorization_commit_that_modifies_existing_file(self):
        _run_git(self.repo, "reset", "--hard", self.fixture.reviewed_commit)
        _write(self.repo / "existing.txt", b"base\n")
        _run_git(self.repo, "add", "existing.txt")
        _run_git(self.repo, "commit", "--quiet", "-m", "new reviewed base")
        reviewed = _run_git(self.repo, "rev-parse", "HEAD")
        value = self.fixture.make_authorization(reviewed)
        _write(self.repo / launcher.AUTHORIZATION_PATH, launcher._canonical_json_bytes(value))
        _write(self.repo / "existing.txt", b"modified\n")
        _run_git(self.repo, "add", launcher.AUTHORIZATION_PATH, "existing.txt")
        _run_git(self.repo, "commit", "--quiet", "-m", "invalid authorization")
        commit = _run_git(self.repo, "rev-parse", "HEAD")
        raw = launcher._canonical_json_bytes(value)
        with self.assertRaisesRegex(RuntimeError, "exactly one path"):
            launcher._verify_authorization(
                self.repo,
                authorization_commit=commit,
                authorization_sha256=hashlib.sha256(raw).hexdigest(),
            )

    def test_rejects_authorization_commit_that_is_not_current_head(self):
        self.fixture.verify()
        _write(self.repo / "later.txt", b"later unrelated commit\n")
        _run_git(self.repo, "add", "later.txt")
        _run_git(self.repo, "commit", "--quiet", "-m", "move current branch")
        with self.assertRaisesRegex(RuntimeError, "current authorization-carrying HEAD"):
            self.fixture.verify()

    def test_rejects_tag_object_as_authorization_commit(self):
        _run_git(
            self.repo,
            "tag",
            "-a",
            "authorization-tag",
            self.fixture.authorization_commit,
            "-m",
            "tag",
        )
        tag_object = _run_git(self.repo, "rev-parse", "authorization-tag^{tag}")
        with self.assertRaisesRegex(RuntimeError, "not exactly a commit"):
            launcher._verify_authorization(
                self.repo,
                authorization_commit=tag_object,
                authorization_sha256=self.fixture.authorization_sha256,
            )


class GitHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        self.fixture = GitAuthorizationFixture(self.repo)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rejects_loose_replacement_ref(self):
        replacement = self.repo / ".git" / "refs" / "replace" / self.fixture.reviewed_commit
        _write(replacement, (self.fixture.authorization_commit + "\n").encode("ascii"))
        with self.assertRaisesRegex(RuntimeError, "replacement refs"):
            launcher._require_repository_contract(self.repo)

    def test_rejects_packed_replacement_ref(self):
        _run_git(
            self.repo,
            "replace",
            self.fixture.reviewed_commit,
            self.fixture.authorization_commit,
        )
        _run_git(self.repo, "pack-refs", "--all", "--prune")
        with self.assertRaisesRegex(RuntimeError, "replacement refs"):
            launcher._require_repository_contract(self.repo)

    def test_rejects_object_alternates_file(self):
        alternates = self.repo / ".git" / "objects" / "info" / "alternates"
        _write(alternates, b"/tmp/untrusted-objects\n")
        with self.assertRaisesRegex(RuntimeError, "alternates"):
            launcher._require_repository_contract(self.repo)

    def test_rejects_python_git_config_and_loader_environment_injection(self):
        environment = {
            "PYTHONPATH": "/tmp/shadow",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.replaceRefsBase",
            "GIT_OBJECT_DIRECTORY": "/tmp/objects",
            "LD_PRELOAD": "/tmp/inject.so",
            "PATH": "/tmp/bin",
        }
        self.assertEqual(
            launcher._forbidden_environment_names(environment),
            [
                "GIT_CONFIG_COUNT",
                "GIT_CONFIG_KEY_0",
                "GIT_OBJECT_DIRECTORY",
                "LD_PRELOAD",
                "PYTHONPATH",
            ],
        )

    def test_git_subprocess_uses_fixed_minimal_environment(self):
        original = os.environ.get("GIT_CONFIG_COUNT")
        os.environ["GIT_CONFIG_COUNT"] = "malicious"
        try:
            observed = launcher._git_output(
                self.repo, ["rev-parse", "--show-object-format"], "object format"
            )
        finally:
            if original is None:
                os.environ.pop("GIT_CONFIG_COUNT", None)
            else:
                os.environ["GIT_CONFIG_COUNT"] = original
        self.assertEqual(observed, b"sha1\n")

    def test_cpu_snapshot_uses_explicit_commit_and_excludes_its_output(self):
        snapshots, reviewed = launcher._cpu_snapshots(
            self.repo, self.fixture.reviewed_commit
        )
        self.assertEqual(set(reviewed), set(launcher.CPU_REVIEWED_PATHS))
        self.assertNotIn(launcher.CPU_AUDIT_PATH, reviewed)
        self.assertEqual(
            snapshots["audit_adaptive_oracle_cpu"]["bytes"],
            self.fixture.source_bytes["cpu_auditor"],
        )
        _write(self.repo / "worktree-shadow.py", b"malicious\n")
        later = launcher._cpu_snapshots(self.repo, self.fixture.reviewed_commit)[1]
        self.assertEqual(reviewed, later)

    def test_cpu_mode_runs_end_to_end_without_authorization_or_worktree_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            _run_git(repo, "init", "--quiet")
            _run_git(repo, "config", "user.name", "Launcher Tests")
            _run_git(repo, "config", "user.email", "launcher@example.invalid")
            launcher_bytes = MODULE_PATH.read_bytes()
            stub = (
                "def _run_from_verified_launcher(argv, *, launcher_context):\n"
                "    evidence = launcher_context.claim()\n"
                "    assert evidence['mode'] == 'cpu_audit_reviewed_commit'\n"
                "    for path in evidence['reviewed_files']:\n"
                "        launcher_context.read_reviewed_file(path)\n"
                "    return 0\n"
            ).encode("ascii")
            for relative_path in launcher.CPU_REVIEWED_PATHS:
                if relative_path == launcher.LAUNCHER_PATH:
                    value = launcher_bytes
                elif relative_path == launcher.MODULE_SOURCES[
                    "audit_adaptive_oracle_cpu"
                ]:
                    value = stub
                else:
                    value = b"REVIEWED_VALUE = 1\n"
                _write(repo / relative_path, value)
            _run_git(repo, "add", ".")
            _run_git(repo, "commit", "--quiet", "-m", "reviewed CPU audit")
            reviewed_commit = _run_git(repo, "rev-parse", "HEAD")
            _write(repo / "adaptive_oracle_contract.py", b"raise RuntimeError('shadow')\n")
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("PYTHON")
                and not key.startswith("GIT_CONFIG")
                and key
                not in launcher._FORBIDDEN_GIT_ENVIRONMENT
                | launcher._FORBIDDEN_PROCESS_ENVIRONMENT
            }
            result = subprocess.run(
                [
                    launcher.PYTHON_EXECUTABLE,
                    "-I",
                    "-B",
                    "-S",
                    "-W",
                    "error",
                    str(repo / launcher.LAUNCHER_PATH),
                    "--reviewed-commit",
                    reviewed_commit,
                    "cpu-audit",
                ],
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_engineering_mode_exposes_verified_three_commit_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            _run_git(repo, "init", "--quiet")
            _run_git(repo, "config", "user.name", "Launcher Tests")
            _run_git(repo, "config", "user.email", "launcher@example.invalid")
            launcher_bytes = MODULE_PATH.read_bytes()
            stub = (
                "def _run_from_verified_launcher(argv, *, launcher_context):\n"
                "    evidence = launcher_context.claim()\n"
                "    implementation, cpu_audit, authorization = argv\n"
                "    assert evidence['commit_topology'] == {\n"
                f"        'schema': {launcher.COMMIT_TOPOLOGY_SCHEMA!r},\n"
                "        'implementation_commit': implementation,\n"
                "        'cpu_audit_commit': cpu_audit,\n"
                "        'authorization_commit': authorization,\n"
                "        'head_commit': authorization,\n"
                "        'cpu_audit_parent': implementation,\n"
                "        'authorization_parent': cpu_audit,\n"
                "    }\n"
                "    assert evidence['reviewed_commit'] == cpu_audit\n"
                "    assert evidence['authorization_binding']['commit'] == authorization\n"
                "    for name in evidence['inputs']:\n"
                "        launcher_context.read_input(name)\n"
                "    return 0\n"
            ).encode("ascii")
            source_bytes = {}
            for index, (name, relative_path) in enumerate(
                launcher.SOURCE_PATHS.items()
            ):
                if relative_path == launcher.LAUNCHER_PATH:
                    value = launcher_bytes
                elif relative_path == launcher.MODULE_SOURCES[
                    "generate_adaptive_oracle_engineering"
                ]:
                    value = stub
                else:
                    value = f"REVIEWED_VALUE = {index}\n".encode("ascii")
                source_bytes[name] = value
                _write(repo / relative_path, value)
            _run_git(repo, "add", ".")
            _run_git(repo, "commit", "--quiet", "-m", "implementation A")
            implementation_commit = _run_git(repo, "rev-parse", "HEAD")

            cpu_audit_bytes = launcher._canonical_json_bytes(
                {"implementation": {"commit": implementation_commit}}
            )
            _write(repo / launcher.CPU_AUDIT_PATH, cpu_audit_bytes)
            _run_git(repo, "add", launcher.CPU_AUDIT_PATH)
            _run_git(repo, "commit", "--quiet", "-m", "CPU audit B")
            cpu_audit_commit = _run_git(repo, "rev-parse", "HEAD")

            authorization = {
                "authorization": {
                    "reviewed_commit": cpu_audit_commit,
                    "source_registration": {
                        "path": launcher.SOURCE_PATHS["registration"],
                        "sha256": hashlib.sha256(
                            source_bytes["registration"]
                        ).hexdigest(),
                    },
                    "cpu_audit": {
                        "path": launcher.CPU_AUDIT_PATH,
                        "sha256": hashlib.sha256(cpu_audit_bytes).hexdigest(),
                    },
                },
                "sources": {
                    name: {
                        "path": relative_path,
                        "sha256": hashlib.sha256(source_bytes[name]).hexdigest(),
                    }
                    for name, relative_path in launcher.SOURCE_PATHS.items()
                },
            }
            authorization_bytes = launcher._canonical_json_bytes(authorization)
            _write(repo / launcher.AUTHORIZATION_PATH, authorization_bytes)
            _run_git(repo, "add", launcher.AUTHORIZATION_PATH)
            _run_git(repo, "commit", "--quiet", "-m", "authorization C")
            authorization_commit = _run_git(repo, "rev-parse", "HEAD")

            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("PYTHON")
                and not key.startswith("GIT_CONFIG")
                and key
                not in launcher._FORBIDDEN_GIT_ENVIRONMENT
                | launcher._FORBIDDEN_PROCESS_ENVIRONMENT
            }
            result = subprocess.run(
                [
                    launcher.PYTHON_EXECUTABLE,
                    "-I",
                    "-B",
                    "-S",
                    "-W",
                    "error",
                    str(repo / launcher.LAUNCHER_PATH),
                    "--authorization-commit",
                    authorization_commit,
                    "--authorization-sha256",
                    hashlib.sha256(authorization_bytes).hexdigest(),
                    "generate",
                    implementation_commit,
                    cpu_audit_commit,
                    authorization_commit,
                ],
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


class ImportIsolationTests(unittest.TestCase):
    def _snapshot(self, module_name: str, value: bytes) -> dict:
        return {
            module_name: {
                "source_name": "contract",
                "relative_path": "eval-pipeline/adaptive_oracle_contract.py",
                "origin": "<git-blob:test:adaptive_oracle_contract.py>",
                "sha256": hashlib.sha256(value).hexdigest(),
                "bytes": value,
            }
        }

    @contextlib.contextmanager
    def _finder(self, snapshots):
        finder = launcher._GitBlobFinder(snapshots)
        previous_modules = {
            name: sys.modules.pop(name)
            for name in snapshots
            if name in sys.modules
        }
        sys.meta_path.insert(0, finder)
        try:
            yield finder
        finally:
            while finder in sys.meta_path:
                sys.meta_path.remove(finder)
            for name in snapshots:
                sys.modules.pop(name, None)
            sys.modules.update(previous_modules)

    def test_worktree_shadow_module_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            shadow = Path(temporary)
            _write(shadow / "adaptive_oracle_contract.py", b"VALUE = 'shadow'\n")
            original_path = list(sys.path)
            sys.path.insert(0, str(shadow))
            try:
                with self._finder(
                    self._snapshot("adaptive_oracle_contract", b"VALUE = 'reviewed'\n")
                ):
                    imported = importlib.import_module("adaptive_oracle_contract")
                    self.assertEqual(imported.VALUE, "reviewed")
                    self.assertTrue(imported.__file__.startswith("<git-blob:"))
            finally:
                sys.path[:] = original_path

    def test_malicious_bytecode_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            shadow = Path(temporary)
            source = shadow / "adaptive_oracle_contract.py"
            _write(source, b"VALUE = 'bytecode-shadow'\n")
            py_compile.compile(str(source), doraise=True)
            source.unlink()
            original_path = list(sys.path)
            sys.path.insert(0, str(shadow))
            try:
                with self._finder(
                    self._snapshot("adaptive_oracle_contract", b"VALUE = 'reviewed'\n")
                ):
                    imported = importlib.import_module("adaptive_oracle_contract")
                    self.assertEqual(imported.VALUE, "reviewed")
                    self.assertIsNone(imported.__cached__)
            finally:
                sys.path[:] = original_path

    def test_unregistered_protected_import_is_rejected(self):
        finder = launcher._GitBlobFinder({})
        with self.assertRaisesRegex(ImportError, "unregistered protected module"):
            finder.find_spec("AttentionGuidance.worktree_only")

    def test_package_and_namespace_paths_are_empty_and_synthetic(self):
        snapshots = self._snapshot("AttentionGuidance", b"PACKAGE_VALUE = 1\n")
        with self._finder(snapshots) as finder:
            package = importlib.import_module("AttentionGuidance")
            self.assertEqual(list(package.__path__), [])
            self.assertTrue(package.__file__.startswith("<git-blob:"))
            namespace_spec = finder.find_spec("InferencePipelines.RepLDM")
            self.assertIsNotNone(namespace_spec)
            self.assertEqual(list(namespace_spec.submodule_search_locations), [])
            self.assertTrue(namespace_spec.origin.startswith("<git-namespace:"))

    def test_sanitize_sys_path_removes_repository_and_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            repo = base / "repo"
            cwd = base / "working"
            trusted = base / "trusted-site-packages"
            repo.mkdir()
            cwd.mkdir()
            trusted.mkdir()
            original_path = list(sys.path)
            original_site = launcher.PYTHON_SITE_PACKAGES
            launcher.PYTHON_SITE_PACKAGES = str(trusted.resolve())
            sys.path[:] = ["", str(repo), str(repo / "child"), str(cwd), "/stdlib"]
            old_cwd = Path.cwd()
            os.chdir(cwd)
            try:
                observed = launcher._sanitize_sys_path(repo, cwd)
                self.assertNotIn("", observed)
                self.assertNotIn(str(repo), observed)
                self.assertNotIn(str(cwd), observed)
                self.assertIn("/stdlib", observed)
                self.assertIn(str(trusted), observed)
            finally:
                os.chdir(old_cwd)
                launcher.PYTHON_SITE_PACKAGES = original_site
                sys.path[:] = original_path

    def test_isolated_no_site_process_ignores_sitecustomize_and_pythonpath(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            marker = directory / "sitecustomize-ran"
            _write(
                directory / "sitecustomize.py",
                f"open({str(marker)!r}, 'w').write('ran')\n".encode("ascii"),
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(directory)
            result = subprocess.run(
                [
                    launcher.PYTHON_EXECUTABLE,
                    "-I",
                    "-B",
                    "-S",
                    "-W",
                    "error",
                    "-c",
                    "import sys; print('sitecustomize' in sys.modules); print(sys.path)",
                ],
                env=environment,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertTrue(result.stdout.startswith("False\n"))
            self.assertNotIn(str(directory), result.stdout)
            self.assertFalse(marker.exists())


class LauncherContextTests(unittest.TestCase):
    def _module(self):
        module = ModuleType("verified_launcher_target")
        exec(
            "def entry(context):\n"
            "    return context.claim()\n"
            "def read(context, name):\n"
            "    return context.read_input(name)\n",
            module.__dict__,
        )
        return module

    def _context(self, module):
        return launcher.LauncherContext(
            module=module,
            entrypoint=module.entry,
            evidence_factory=lambda: {"nested": {"values": [1]}},
            input_bytes={"registration": b"registered", "prompts": b"prompts"},
        )

    def test_context_is_one_use_caller_bound_and_requires_all_inputs(self):
        module = self._module()
        context = self._context(module)
        evidence = module.entry(context)
        evidence["nested"]["values"].append(2)
        self.assertFalse(context.consumed)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            module.entry(context)
        self.assertEqual(module.read(context, "registration"), b"registered")
        with self.assertRaisesRegex(RuntimeError, "only once"):
            module.read(context, "registration")
        self.assertFalse(context.consumed)
        self.assertEqual(module.read(context, "prompts"), b"prompts")
        self.assertTrue(context.consumed)

    def test_context_rejects_an_unbound_module_and_direct_caller(self):
        module = self._module()
        context = self._context(module)
        with self.assertRaisesRegex(RuntimeError, "bound verified module"):
            context.claim()
        attacker = ModuleType("attacker")
        exec("def steal(context):\n    return context.claim()\n", attacker.__dict__)
        with self.assertRaisesRegex(RuntimeError, "bound verified module"):
            attacker.steal(context)

    def test_context_returns_a_fresh_json_copy(self):
        module = self._module()
        context = self._context(module)
        observed = module.entry(context)
        observed["nested"]["values"].append(2)
        self.assertEqual(context._evidence_factory(), {"nested": {"values": [1]}})

    def test_reviewed_files_are_allowlisted_one_use_and_required(self):
        module = self._module()
        exec(
            "def read_file(context, path):\n"
            "    return context.read_reviewed_file(path)\n",
            module.__dict__,
        )
        context = launcher.LauncherContext(
            module=module,
            entrypoint=module.entry,
            evidence_factory=lambda: {"mode": "cpu"},
            input_bytes={},
            reviewed_file_bytes={"reviewed.py": b"reviewed"},
        )
        module.entry(context)
        self.assertFalse(context.consumed)
        self.assertEqual(module.read_file(context, "reviewed.py"), b"reviewed")
        self.assertTrue(context.consumed)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            module.read_file(context, "reviewed.py")
        with self.assertRaisesRegex(RuntimeError, "no reviewed file"):
            module.read_file(context, "shadow.py")


class SourceContractTests(unittest.TestCase):
    def test_launcher_binds_head_without_a_caller_token_trust_channel(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("HEAD^{commit}", source)
        self.assertNotIn("expected_token", source)
        self.assertNotIn("token=", source)


if __name__ == "__main__":
    unittest.main()
