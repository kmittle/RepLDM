"""Trusted, stdlib-only launcher for Adaptive Oracle production tools.

The launcher authenticates an explicit implementation/audit/authorization
commit chain and executes repository modules only from the reviewed commit's
Git blobs.  It must itself be invoked directly with the frozen CPython
executable and isolation flags.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = "eval-pipeline/launch_adaptive_oracle_tool.py"
AUTHORIZATION_PATH = (
    "eval-pipeline/configs/adaptive_oracle_engineering_authorization_v1.yaml"
)
CPU_AUDIT_PATH = (
    "eval-pipeline/configs/adaptive_oracle_engineering_cpu_audit_v1.json"
)

PYTHON_EXECUTABLE = "/home/bycao/miniforge3/envs/diff_attn/bin/python3.11"
PYTHON_SHA256 = "cf53a920b9d888d338692a6128e866246a0689251894ae877cb8148b995c9f35"
PYTHON_VERSION = (3, 11, 10)
PYTHON_SITE_PACKAGES = (
    "/home/bycao/miniforge3/envs/diff_attn/lib/python3.11/site-packages"
)
PYTHON_STDLIB_PATHS = (
    "/home/bycao/miniforge3/envs/diff_attn/lib/python311.zip",
    "/home/bycao/miniforge3/envs/diff_attn/lib/python3.11",
    "/home/bycao/miniforge3/envs/diff_attn/lib/python3.11/lib-dynload",
)
GIT = "/usr/bin/git"
GIT_SHA256 = "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
GIT_VERSION = "git version 2.43.0"

LAUNCH_SCHEMA = "adaptive_oracle_explicit_git_blob_launch_v2"
CPU_LAUNCH_SCHEMA = "adaptive_oracle_explicit_reviewed_cpu_launch_v1"
COMMIT_TOPOLOGY_SCHEMA = "adaptive_oracle_commit_topology_v1"
LOADER_ID = "adaptive_oracle_reviewed_git_blob_loader_v2"
TRUST_ROOT_ID = "cpython_explicit_commit_launcher_v2"

SOURCE_PATHS = {
    "launcher": LAUNCHER_PATH,
    "prompt_csv": "eval-pipeline/prompts/adaptive_oracle_engineering.csv",
    "prompt_manifest": "eval-pipeline/prompts/adaptive_oracle_prompt_manifest_v1.json",
    "exclusion_inventory": "eval-pipeline/prompts/adaptive_oracle_exclusion_inventory_v1.json",
    "prompt_builder": "eval-pipeline/build_adaptive_oracle_prompts.py",
    "contract": "eval-pipeline/adaptive_oracle_contract.py",
    "generator": "eval-pipeline/generate_adaptive_oracle_engineering.py",
    "model_snapshot_validator": "eval-pipeline/adaptive_oracle_model_snapshot.py",
    "cpu_auditor": "eval-pipeline/audit_adaptive_oracle_cpu.py",
    "engineering_auditor": "eval-pipeline/audit_adaptive_oracle_engineering.py",
    "adaptive_oracle": "AttentionGuidance/adaptive_oracle.py",
    "attention_guidance": "AttentionGuidance/attention_guidance.py",
    "controller": "AttentionGuidance/controller.py",
    "guidance_types": "AttentionGuidance/types.py",
    "semantic_transport": "AttentionGuidance/semantic_transport.py",
    "freeu": "AttentionGuidance/freeu.py",
    "ancestral_correction": "AttentionGuidance/ancestral_correction.py",
    "local_relational_basis": "AttentionGuidance/local_relational_basis.py",
    "latent_renderer": "AttentionGuidance/latent_renderer.py",
    "pipeline": "InferencePipelines/RepLDM/pipeline_repldm_sdxl.py",
    "attention_guidance_init": "AttentionGuidance/__init__.py",
    "inference_pipelines_init": "InferencePipelines/__init__.py",
    "cfg_batch": "InferencePipelines/cfg_batch.py",
    "environment_validator": (
        "eval-pipeline/adaptive_oracle_generation_environment.py"
    ),
    "environment_lock": "eval-pipeline/configs/generation_environment_adaptive_oracle_v2.yaml",
    "registration": "eval-pipeline/configs/adaptive_oracle_engineering_registration_v1.yaml",
}

MODULE_SOURCE_NAMES = {
    "adaptive_oracle_contract": "contract",
    "adaptive_oracle_model_snapshot": "model_snapshot_validator",
    "audit_adaptive_oracle_cpu": "cpu_auditor",
    "audit_adaptive_oracle_engineering": "engineering_auditor",
    "build_adaptive_oracle_prompts": "prompt_builder",
    "generate_adaptive_oracle_engineering": "generator",
    "adaptive_oracle_generation_environment": "environment_validator",
    "AttentionGuidance": "attention_guidance_init",
    "AttentionGuidance.attention_guidance": "attention_guidance",
    "AttentionGuidance.controller": "controller",
    "AttentionGuidance.types": "guidance_types",
    "AttentionGuidance.semantic_transport": "semantic_transport",
    "AttentionGuidance.latent_renderer": "latent_renderer",
    "AttentionGuidance.freeu": "freeu",
    "AttentionGuidance.ancestral_correction": "ancestral_correction",
    "AttentionGuidance.local_relational_basis": "local_relational_basis",
    "AttentionGuidance.adaptive_oracle": "adaptive_oracle",
    "InferencePipelines": "inference_pipelines_init",
    "InferencePipelines.cfg_batch": "cfg_batch",
    "InferencePipelines.RepLDM.pipeline_repldm_sdxl": "pipeline",
}
MODULE_SOURCES = {
    module_name: SOURCE_PATHS[source_name]
    for module_name, source_name in MODULE_SOURCE_NAMES.items()
}
INPUT_PUBLIC_NAMES = {
    "registration": "registration",
    "prompt_csv": "prompt_csv",
    "prompt_manifest": "prompt_manifest",
    "exclusion_inventory": "exclusion_inventory",
    "environment_lock": "environment_lock",
}
CPU_TEST_SOURCES = (
    "tests/test_adaptive_oracle.py",
    "tests/test_adaptive_oracle_contract.py",
    "tests/test_adaptive_oracle_model_snapshot.py",
    "tests/test_adaptive_oracle_prompts.py",
    "tests/test_audit_adaptive_oracle_cpu.py",
    "tests/test_audit_adaptive_oracle_engineering.py",
    "tests/test_generate_adaptive_oracle_engineering.py",
    "tests/test_launch_adaptive_oracle_tool.py",
    "tests/test_local_relational_basis.py",
)
CPU_REVIEWED_PATHS = tuple(sorted(set(SOURCE_PATHS.values()) | set(CPU_TEST_SOURCES)))

PACKAGE_MODULES = frozenset({"AttentionGuidance", "InferencePipelines"})
NAMESPACE_PACKAGES = frozenset({"InferencePipelines.RepLDM"})
PROTECTED_PREFIXES = ("AttentionGuidance", "InferencePipelines")
PROTECTED_TOP_LEVEL = frozenset(
    {
        "adaptive_oracle_contract",
        "adaptive_oracle_model_snapshot",
        "audit_adaptive_oracle_cpu",
        "audit_adaptive_oracle_engineering",
        "build_adaptive_oracle_prompts",
        "generate_adaptive_oracle_engineering",
        "adaptive_oracle_generation_environment",
    }
)
TOOLS = {
    "generate": "generate_adaptive_oracle_engineering",
    "audit": "audit_adaptive_oracle_engineering",
    "cpu-audit": "audit_adaptive_oracle_cpu",
}

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_GIT_ENVIRONMENT = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_WORK_TREE",
    }
)
_FORBIDDEN_PROCESS_ENVIRONMENT = frozenset(
    {"LD_AUDIT", "LD_LIBRARY_PATH", "LD_PRELOAD"}
)
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "LC_ALL": "C",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise RuntimeError(f"cannot open {label} as a non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"{label} changed while its identity was verified")
    value = b"".join(chunks)
    if len(value) != before.st_size:
        raise RuntimeError(f"{label} byte count differs from its file identity")
    return value


def _completed(argv: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(_GIT_ENVIRONMENT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _git_output(root: Path, arguments: Sequence[str], label: str) -> bytes:
    result = _completed(
        [GIT, "--no-replace-objects", "-C", str(root), *arguments], cwd=root
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"{label} failed: {detail}")
    return bytes(result.stdout)


def _forbidden_environment_names(
    environment: Optional[Mapping[str, str]] = None,
) -> list[str]:
    values = os.environ if environment is None else environment
    return sorted(
        key
        for key in values
        if key.startswith("PYTHON")
        or key.startswith("GIT_CONFIG")
        or key in _FORBIDDEN_GIT_ENVIRONMENT
        or key in _FORBIDDEN_PROCESS_ENVIRONMENT
    )


def _require_process_contract() -> dict[str, Any]:
    if sys.implementation.name != "cpython" or sys.version_info[:3] != PYTHON_VERSION:
        raise RuntimeError("launcher requires CPython 3.11.10")
    executable = Path(sys.executable).resolve(strict=True)
    if str(executable) != PYTHON_EXECUTABLE:
        raise RuntimeError("launcher is running under an untrusted Python executable")
    executable_hash = _sha256(
        _stable_regular_bytes(executable, label="Python executable")
    )
    if executable_hash != PYTHON_SHA256:
        raise RuntimeError("Python executable SHA-256 differs from the frozen identity")
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.dont_write_bytecode
        or sys.flags.optimize != 0
        or sys.warnoptions != ["error"]
    ):
        raise RuntimeError(
            "launcher requires python -I -B -S -W error with optimization disabled"
        )
    if tuple(sys.path) != PYTHON_STDLIB_PATHS:
        raise RuntimeError("initial isolated sys.path differs from the frozen stdlib paths")
    forbidden = _forbidden_environment_names()
    if forbidden:
        raise RuntimeError(
            "launcher forbids process or repository injection variables: "
            + ", ".join(forbidden)
        )
    git_path = Path(GIT).resolve(strict=True)
    if str(git_path) != GIT:
        raise RuntimeError("Git executable does not resolve to the frozen path")
    git_hash = _sha256(_stable_regular_bytes(git_path, label="Git executable"))
    if git_hash != GIT_SHA256:
        raise RuntimeError("Git executable SHA-256 differs from the frozen identity")
    version_result = _completed([GIT, "--version"], cwd=ROOT)
    if version_result.returncode != 0:
        raise RuntimeError("cannot execute the frozen Git identity")
    version = version_result.stdout.decode("ascii", errors="strict").strip()
    if version != GIT_VERSION:
        raise RuntimeError("Git version differs from the frozen identity")
    return {
        "implementation": "CPython",
        "version": ".".join(str(value) for value in PYTHON_VERSION),
        "executable": PYTHON_EXECUTABLE,
        "sha256": executable_hash,
        "flags": ["-I", "-B", "-S", "-W error"],
    }


def _require_direct_execution(root: Path) -> None:
    expected_launcher = (root / LAUNCHER_PATH).resolve(strict=True)
    main_module = sys.modules.get("__main__")
    main_file = None if main_module is None else getattr(main_module, "__file__", None)
    if (
        not isinstance(main_file, str)
        or Path(main_file).resolve(strict=True) != expected_launcher
        or Path(sys.argv[0]).resolve(strict=True) != expected_launcher
    ):
        raise RuntimeError("launcher must be executed directly as the main Python script")
    original = list(getattr(sys, "orig_argv", ()))
    if (
        len(original) < 7
        or Path(original[0]).resolve(strict=True) != Path(PYTHON_EXECUTABLE)
        or original[1:6] != ["-I", "-B", "-S", "-W", "error"]
        or Path(original[6]).resolve(strict=True) != expected_launcher
    ):
        raise RuntimeError(
            "launcher invocation must begin with exact -I -B -S -W error flags"
        )


def _require_repository_contract(root: Path) -> None:
    top = _git_output(
        root, ["rev-parse", "--path-format=absolute", "--show-toplevel"], "Git root"
    ).decode("utf-8", errors="strict").strip()
    if Path(top).resolve(strict=True) != root:
        raise RuntimeError("launcher path is not in the expected Git repository root")
    object_format = _git_output(
        root, ["rev-parse", "--show-object-format"], "Git object format"
    ).decode("ascii", errors="strict").strip()
    if object_format != "sha1":
        raise RuntimeError("launcher requires a SHA-1 object-format repository")
    replace_refs = _git_output(
        root,
        ["for-each-ref", "--format=%(refname)", "refs/replace"],
        "Git replacement-ref inventory",
    )
    if replace_refs:
        raise RuntimeError("Git replacement refs are forbidden")
    common_value = _git_output(
        root, ["rev-parse", "--git-common-dir"], "Git common directory"
    ).decode("utf-8", errors="strict").strip()
    common_candidate = Path(common_value)
    if not common_candidate.is_absolute():
        common_candidate = root / common_candidate
    common = common_candidate.resolve(strict=True)
    if os.path.lexists(common / "objects" / "info" / "alternates"):
        raise RuntimeError("Git object alternates are forbidden")


def _git_object(root: Path, object_id: str, object_type: str) -> bytes:
    if _COMMIT.fullmatch(object_id) is None:
        raise RuntimeError("Git object id must be full lowercase 40-hex")
    observed_type = _git_output(
        root, ["cat-file", "-t", object_id], f"Git object type {object_id}"
    ).decode("ascii", errors="strict").strip()
    if observed_type != object_type:
        raise RuntimeError(f"Git object {object_id} is not exactly a {object_type}")
    return _git_output(
        root, ["cat-file", object_type, object_id], f"Git {object_type} {object_id}"
    )


def _git_blob(root: Path, commit: str, relative_path: str) -> bytes:
    if _COMMIT.fullmatch(commit) is None:
        raise RuntimeError("Git commit id must be full lowercase 40-hex")
    return _git_output(
        root,
        ["cat-file", "blob", f"{commit}:{relative_path}"],
        f"Git blob {commit}:{relative_path}",
    )


def _sole_parent(commit_bytes: bytes, *, label: str = "authorization") -> str:
    header, separator, _message = commit_bytes.partition(b"\n\n")
    if not separator:
        raise RuntimeError(f"{label} commit object has no message separator")
    parents = []
    for line in header.splitlines():
        if line.startswith(b"parent "):
            value = line[len(b"parent ") :]
            try:
                parent = value.decode("ascii", errors="strict")
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"{label} commit parent is not ASCII") from exc
            if _COMMIT.fullmatch(parent) is None:
                raise RuntimeError(
                    f"{label} commit parent is not full lowercase 40-hex"
                )
            parents.append(parent)
    if len(parents) != 1:
        raise RuntimeError(f"{label} commit must have exactly one parent")
    return parents[0]


def _require_only_ordinary_file_addition(
    root: Path,
    *,
    parent: str,
    commit: str,
    expected_path: str,
    label: str,
) -> None:
    raw = _git_output(
        root,
        [
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "-r",
            "-z",
            "--no-renames",
            parent,
            commit,
        ],
        f"{label} commit diff",
    )
    parts = raw.split(b"\0")
    if len(parts) != 3 or parts[-1] != b"":
        raise RuntimeError(f"{label} commit must change exactly one path")
    metadata, path_bytes = parts[:2]
    fields = metadata.split()
    if len(fields) != 5 or not fields[0].startswith(b":"):
        raise RuntimeError(f"{label} commit has a malformed raw diff")
    old_mode = fields[0][1:]
    new_mode, old_oid, new_oid, status_value = fields[1:]
    try:
        path = path_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{label} commit path is not UTF-8") from exc
    if (
        old_mode != b"000000"
        or new_mode != b"100644"
        or old_oid != b"0" * 40
        or _COMMIT.fullmatch(new_oid.decode("ascii", errors="strict")) is None
        or status_value != b"A"
        or path != expected_path
    ):
        raise RuntimeError(
            f"{label} commit must only add the ordinary 100644 file {expected_path}"
        )


def _require_authorization_only_addition(
    root: Path, *, parent: str, authorization_commit: str
) -> None:
    _require_only_ordinary_file_addition(
        root,
        parent=parent,
        commit=authorization_commit,
        expected_path=AUTHORIZATION_PATH,
        label="authorization",
    )


def _require_cpu_audit_only_addition(
    root: Path, *, parent: str, cpu_audit_commit: str
) -> None:
    _require_only_ordinary_file_addition(
        root,
        parent=parent,
        commit=cpu_audit_commit,
        expected_path=CPU_AUDIT_PATH,
        label="CPU-audit",
    )


def _commit_topology(
    *,
    implementation_commit: str,
    cpu_audit_commit: str,
    authorization_commit: str,
) -> dict[str, Any]:
    return {
        "schema": COMMIT_TOPOLOGY_SCHEMA,
        "implementation_commit": implementation_commit,
        "cpu_audit_commit": cpu_audit_commit,
        "authorization_commit": authorization_commit,
        "head_commit": authorization_commit,
        "cpu_audit_parent": implementation_commit,
        "authorization_parent": cpu_audit_commit,
    }


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"authorization JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"authorization JSON contains non-finite constant {value!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_canonical_authorization(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"authorization is not strict canonical JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("authorization must contain exactly one JSON object")
    if raw != _canonical_json_bytes(value):
        raise RuntimeError("authorization bytes are not strict canonical JSON")
    return value


def _validated_source_pairs(
    authorization: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    raw_sources = authorization.get("sources")
    if not isinstance(raw_sources, dict) or set(raw_sources) != set(SOURCE_PATHS):
        raise RuntimeError("authorization source inventory differs from the launcher allowlist")
    rows: dict[str, dict[str, str]] = {}
    for name, expected_path in SOURCE_PATHS.items():
        pair = raw_sources.get(name)
        if not isinstance(pair, dict) or set(pair) != {"path", "sha256"}:
            raise RuntimeError(f"authorization source pair differs for {name}")
        path = pair.get("path")
        digest = pair.get("sha256")
        if path != expected_path or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise RuntimeError(f"authorization source identity differs for {name}")
        normalized = PurePosixPath(path)
        if normalized.is_absolute() or ".." in normalized.parts or str(normalized) != path:
            raise RuntimeError(f"authorization source path is not canonical for {name}")
        rows[name] = {"path": path, "sha256": digest}
    return rows


def _validated_file_pair(
    value: Any, *, expected_path: str, label: str
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise RuntimeError(f"{label} file pair differs")
    path = value.get("path")
    digest = value.get("sha256")
    if (
        path != expected_path
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise RuntimeError(f"{label} file identity differs")
    return {"path": path, "sha256": digest}


def _verify_authorization(
    root: Path, *, authorization_commit: str, authorization_sha256: str
) -> tuple[
    str,
    str,
    dict[str, Any],
    bytes,
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    if _COMMIT.fullmatch(authorization_commit) is None:
        raise RuntimeError("authorization commit must be full lowercase 40-hex")
    if _SHA256.fullmatch(authorization_sha256) is None:
        raise RuntimeError("authorization SHA-256 must be 64 lowercase hex")
    commit_bytes = _git_object(root, authorization_commit, "commit")
    head = _git_output(
        root,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        "authorization-carrying HEAD",
    ).decode("ascii", errors="strict").strip()
    if _COMMIT.fullmatch(head) is None:
        raise RuntimeError("authorization-carrying HEAD is not a full Git commit")
    if authorization_commit != head:
        raise RuntimeError(
            "authorization commit must be the current authorization-carrying HEAD"
        )
    reviewed_commit = _sole_parent(commit_bytes, label="authorization")
    reviewed_commit_bytes = _git_object(root, reviewed_commit, "commit")
    _require_authorization_only_addition(
        root, parent=reviewed_commit, authorization_commit=authorization_commit
    )
    implementation_commit = _sole_parent(
        reviewed_commit_bytes, label="CPU-audit"
    )
    _git_object(root, implementation_commit, "commit")
    _require_cpu_audit_only_addition(
        root,
        parent=implementation_commit,
        cpu_audit_commit=reviewed_commit,
    )
    raw = _git_blob(root, authorization_commit, AUTHORIZATION_PATH)
    if _sha256(raw) != authorization_sha256:
        raise RuntimeError("authorization blob SHA-256 differs from the external value")
    authorization = _load_canonical_authorization(raw)
    inner = authorization.get("authorization")
    if not isinstance(inner, dict) or inner.get("reviewed_commit") != reviewed_commit:
        raise RuntimeError(
            "authorization reviewed_commit must equal the authorization commit's sole parent"
        )
    source_pairs = _validated_source_pairs(authorization)
    registration_pair = _validated_file_pair(
        inner.get("source_registration"),
        expected_path=SOURCE_PATHS["registration"],
        label="authorization source registration",
    )
    if registration_pair != source_pairs["registration"]:
        raise RuntimeError(
            "authorization source registration differs from the source inventory"
        )
    input_pairs = {
        public_name: dict(source_pairs[source_name])
        for public_name, source_name in INPUT_PUBLIC_NAMES.items()
    }
    input_pairs["cpu_audit"] = _validated_file_pair(
        inner.get("cpu_audit"),
        expected_path=CPU_AUDIT_PATH,
        label="authorization CPU audit",
    )
    return (
        implementation_commit,
        reviewed_commit,
        authorization,
        raw,
        source_pairs,
        input_pairs,
    )


def _path_inside(path_value: str, directory: Path) -> bool:
    candidate = Path(path_value or os.getcwd()).resolve(strict=False)
    return candidate == directory or directory in candidate.parents


def _sanitize_sys_path(root: Path, cwd: Path) -> list[str]:
    sys.path[:] = [
        entry
        for entry in sys.path
        if not _path_inside(entry, root) and not _path_inside(entry, cwd)
    ]
    site_packages = Path(PYTHON_SITE_PACKAGES).resolve(strict=True)
    if not site_packages.is_dir() or str(site_packages) != PYTHON_SITE_PACKAGES:
        raise RuntimeError("trusted site-packages directory identity differs")
    if PYTHON_SITE_PACKAGES not in sys.path:
        sys.path.append(PYTHON_SITE_PACKAGES)
    if any(_path_inside(entry, root) or _path_inside(entry, cwd) for entry in sys.path):
        raise RuntimeError("repository or cwd remained importable through sys.path")
    return list(sys.path)


def _module_origin(module: Any) -> Optional[Path]:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str) or raw.startswith("<"):
        return None
    return Path(raw).resolve(strict=False)


def _reject_preloaded_repository_modules(root: Path) -> None:
    observed = []
    for name, module in sys.modules.items():
        origin = _module_origin(module)
        if origin is not None and (origin == root or root in origin.parents):
            if name != "__main__":
                observed.append(name)
    protected = [
        name
        for name in sys.modules
        if name in PROTECTED_TOP_LEVEL
        or any(name == prefix or name.startswith(prefix + ".") for prefix in PROTECTED_PREFIXES)
    ]
    if observed or protected:
        raise RuntimeError(
            "repository or protected modules were preloaded before Git-blob launch: "
            + ", ".join(sorted(set(observed + protected)))
        )


class _GitBlobLoader(importlib.abc.Loader):
    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        self.snapshot = dict(snapshot)
        self.execution_count = 0

    def create_module(self, spec):
        del spec
        return None

    def exec_module(self, module: ModuleType) -> None:
        if self.execution_count != 0:
            raise ImportError("a protected Git-blob module may execute only once")
        self.execution_count += 1
        module.__file__ = self.snapshot["origin"]
        module.__cached__ = None
        code = compile(
            self.snapshot["bytes"],
            self.snapshot["origin"],
            "exec",
            dont_inherit=True,
            optimize=0,
        )
        exec(code, module.__dict__)
        module.__cached__ = None


class _GitBlobFinder(importlib.abc.MetaPathFinder):
    def __init__(self, snapshots: Mapping[str, Mapping[str, Any]]) -> None:
        self.snapshots = {name: dict(value) for name, value in snapshots.items()}
        self.loaders = {
            name: _GitBlobLoader(snapshot) for name, snapshot in self.snapshots.items()
        }

    @staticmethod
    def is_protected(fullname: str) -> bool:
        return fullname in PROTECTED_TOP_LEVEL or any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in PROTECTED_PREFIXES
        )

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname in self.loaders:
            snapshot = self.snapshots[fullname]
            spec = importlib.util.spec_from_loader(
                fullname,
                self.loaders[fullname],
                origin=snapshot["origin"],
                is_package=fullname in PACKAGE_MODULES,
            )
            if spec is None:
                raise ImportError(f"cannot create a Git-blob spec for {fullname}")
            spec.cached = None
            if fullname in PACKAGE_MODULES:
                spec.submodule_search_locations = []
            return spec
        if fullname in NAMESPACE_PACKAGES:
            spec = importlib.util.spec_from_loader(fullname, loader=None, is_package=True)
            if spec is None:
                raise ImportError(f"cannot create namespace spec for {fullname}")
            spec.origin = f"<git-namespace:{fullname}>"
            spec.submodule_search_locations = []
            return spec
        if self.is_protected(fullname):
            raise ImportError(f"unregistered protected module {fullname}")
        return None

    def execution_record(self) -> dict[str, dict[str, Any]]:
        rows = {}
        for name, snapshot in sorted(self.snapshots.items()):
            module = sys.modules.get(name)
            rows[name] = {
                "source_name": snapshot["source_name"],
                "path": snapshot["relative_path"],
                "origin": snapshot["origin"],
                "sha256": snapshot["sha256"],
                "loader_id": LOADER_ID,
                "execution_count": self.loaders[name].execution_count,
                "cached": None if module is None else getattr(module, "__cached__", None),
            }
        return rows


def _snapshots(
    root: Path,
    reviewed_commit: str,
    source_pairs: Mapping[str, Mapping[str, str]],
    input_pairs: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    source_bytes = {}
    for source_name, pair in source_pairs.items():
        value = _git_blob(root, reviewed_commit, pair["path"])
        if _sha256(value) != pair["sha256"]:
            raise RuntimeError(
                f"reviewed Git blob SHA-256 differs for source {source_name}"
            )
        source_bytes[source_name] = value
    snapshots = {}
    for module_name, source_name in MODULE_SOURCE_NAMES.items():
        relative_path = SOURCE_PATHS[source_name]
        value = source_bytes[source_name]
        snapshots[module_name] = {
            "source_name": source_name,
            "relative_path": relative_path,
            "origin": f"<git-blob:{reviewed_commit}:{relative_path}>",
            "sha256": _sha256(value),
            "bytes": value,
        }
    inputs = {}
    for public_name, pair in input_pairs.items():
        source_name = INPUT_PUBLIC_NAMES.get(public_name)
        if source_name is None:
            value = _git_blob(root, reviewed_commit, pair["path"])
        else:
            value = source_bytes[source_name]
        if _sha256(value) != pair["sha256"]:
            raise RuntimeError(
                f"reviewed Git blob SHA-256 differs for input {public_name}"
            )
        inputs[public_name] = bytes(value)
    return snapshots, inputs


def _cpu_snapshots(
    root: Path, reviewed_commit: str
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    _git_object(root, reviewed_commit, "commit")
    reviewed_files = {
        relative_path: _git_blob(root, reviewed_commit, relative_path)
        for relative_path in CPU_REVIEWED_PATHS
    }
    snapshots = {}
    for module_name, source_name in MODULE_SOURCE_NAMES.items():
        relative_path = SOURCE_PATHS[source_name]
        value = reviewed_files[relative_path]
        snapshots[module_name] = {
            "source_name": source_name,
            "relative_path": relative_path,
            "origin": f"<git-blob:{reviewed_commit}:{relative_path}>",
            "sha256": _sha256(value),
            "bytes": value,
        }
    return snapshots, reviewed_files


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


class LauncherContext:
    """One-use, module-bound access to authenticated launch evidence and inputs."""

    __slots__ = (
        "_claimed",
        "_entrypoint_code",
        "_evidence_factory",
        "_input_bytes",
        "_module_globals",
        "_module_name",
        "_read_inputs",
        "_read_reviewed_files",
        "_reviewed_file_bytes",
    )

    def __init__(
        self,
        *,
        module: ModuleType,
        entrypoint: Any,
        evidence_factory: Any,
        input_bytes: Mapping[str, bytes],
        reviewed_file_bytes: Optional[Mapping[str, bytes]] = None,
    ) -> None:
        code = getattr(entrypoint, "__code__", None)
        if code is None:
            raise RuntimeError("verified launcher entrypoint must be a Python function")
        self._module_name = module.__name__
        self._module_globals = module.__dict__
        self._entrypoint_code = code
        self._evidence_factory = evidence_factory
        self._input_bytes = {name: bytes(value) for name, value in input_bytes.items()}
        self._reviewed_file_bytes = {
            name: bytes(value)
            for name, value in (reviewed_file_bytes or {}).items()
        }
        self._claimed = False
        self._read_inputs: set[str] = set()
        self._read_reviewed_files: set[str] = set()

    def _require_caller(self, *, exact_entrypoint: bool) -> None:
        try:
            frame = sys._getframe(2)
        except (AttributeError, ValueError) as exc:
            raise RuntimeError("launcher context cannot authenticate its caller") from exc
        if (
            frame.f_globals is not self._module_globals
            or frame.f_globals.get("__name__") != self._module_name
            or (exact_entrypoint and frame.f_code is not self._entrypoint_code)
        ):
            raise RuntimeError("launcher context caller is not the bound verified module")

    def claim(self) -> dict[str, Any]:
        self._require_caller(exact_entrypoint=True)
        if self._claimed:
            raise RuntimeError("launcher context may be claimed only once")
        self._claimed = True
        return _json_copy(self._evidence_factory())

    def read_input(self, name: str) -> bytes:
        self._require_caller(exact_entrypoint=False)
        if not self._claimed:
            raise RuntimeError("launcher context must be claimed before reading inputs")
        if name not in self._input_bytes:
            raise RuntimeError(f"launcher context has no registered input {name!r}")
        if name in self._read_inputs:
            raise RuntimeError(f"launcher input {name!r} may be read only once")
        self._read_inputs.add(name)
        return bytes(self._input_bytes[name])

    def read_reviewed_file(self, relative_path: str) -> bytes:
        self._require_caller(exact_entrypoint=False)
        if not self._claimed:
            raise RuntimeError("launcher context must be claimed before reading files")
        if relative_path not in self._reviewed_file_bytes:
            raise RuntimeError(
                f"launcher context has no reviewed file {relative_path!r}"
            )
        if relative_path in self._read_reviewed_files:
            raise RuntimeError(
                f"launcher reviewed file {relative_path!r} may be read only once"
            )
        self._read_reviewed_files.add(relative_path)
        return bytes(self._reviewed_file_bytes[relative_path])

    @property
    def consumed(self) -> bool:
        return (
            self._claimed
            and self._read_inputs == set(self._input_bytes)
            and self._read_reviewed_files == set(self._reviewed_file_bytes)
        )

    @property
    def unread_inputs(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._input_bytes) - self._read_inputs))

    @property
    def unread_reviewed_files(self) -> tuple[str, ...]:
        return tuple(
            sorted(set(self._reviewed_file_bytes) - self._read_reviewed_files)
        )


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="launch_adaptive_oracle_tool.py", allow_abbrev=False
    )
    parser.add_argument("--authorization-commit")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--reviewed-commit")
    parser.add_argument("tool", choices=tuple(TOOLS))
    parser.add_argument("tool_arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(list(arguments))
    if parsed.tool == "cpu-audit":
        if parsed.reviewed_commit is None:
            parser.error("cpu-audit requires --reviewed-commit")
        if parsed.authorization_commit is not None or parsed.authorization_sha256 is not None:
            parser.error("cpu-audit forbids engineering authorization arguments")
    else:
        if parsed.authorization_commit is None or parsed.authorization_sha256 is None:
            parser.error(
                "generate and audit require --authorization-commit and --authorization-sha256"
            )
        if parsed.reviewed_commit is not None:
            parser.error("engineering tools forbid --reviewed-commit")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parsed = _parse_arguments(sys.argv[1:] if argv is None else argv)
    python_identity = _require_process_contract()
    root = ROOT.resolve(strict=True)
    _require_direct_execution(root)
    cwd = Path.cwd().resolve(strict=True)
    if cwd != root:
        raise RuntimeError("Adaptive Oracle launcher must run from the repository root")
    _require_repository_contract(root)
    cpu_mode = parsed.tool == "cpu-audit"
    if cpu_mode:
        reviewed_commit = parsed.reviewed_commit
        if not isinstance(reviewed_commit, str) or _COMMIT.fullmatch(reviewed_commit) is None:
            raise RuntimeError("reviewed commit must be full lowercase 40-hex")
        authorization = None
        authorization_raw = None
        implementation_commit = None
        source_pairs = None
        input_pairs = None
    else:
        (
            implementation_commit,
            reviewed_commit,
            authorization,
            authorization_raw,
            source_pairs,
            input_pairs,
        ) = _verify_authorization(
            root,
            authorization_commit=parsed.authorization_commit,
            authorization_sha256=parsed.authorization_sha256,
        )
    launcher_bytes = _stable_regular_bytes(
        root / LAUNCHER_PATH, label="executed launcher"
    )
    reviewed_launcher_bytes = _git_blob(root, reviewed_commit, LAUNCHER_PATH)
    if launcher_bytes != reviewed_launcher_bytes:
        raise RuntimeError("executed launcher bytes differ from the reviewed Git blob")
    sanitized_path = _sanitize_sys_path(root, cwd)
    _reject_preloaded_repository_modules(root)
    if cpu_mode:
        snapshots, reviewed_files = _cpu_snapshots(root, reviewed_commit)
        inputs = {}
    else:
        assert source_pairs is not None and input_pairs is not None
        snapshots, inputs = _snapshots(
            root, reviewed_commit, source_pairs, input_pairs
        )
        reviewed_files = {}
    finder = _GitBlobFinder(snapshots)
    sys.meta_path.insert(0, finder)
    try:
        target = importlib.import_module(TOOLS[parsed.tool])
        entrypoint = getattr(target, "_run_from_verified_launcher", None)
        if not callable(entrypoint):
            raise RuntimeError("verified tool does not expose its launcher-only entrypoint")

        def evidence_factory() -> dict[str, Any]:
            if cpu_mode:
                input_inventory = {
                    public_name: {
                        "path": SOURCE_PATHS[source_name],
                        "sha256": _sha256(reviewed_files[SOURCE_PATHS[source_name]]),
                        "byte_count": len(reviewed_files[SOURCE_PATHS[source_name]]),
                        "commit": reviewed_commit,
                    }
                    for public_name, source_name in INPUT_PUBLIC_NAMES.items()
                }
                reviewed_inventory = {
                    path: {
                        "sha256": _sha256(value),
                        "byte_count": len(value),
                        "commit": reviewed_commit,
                    }
                    for path, value in sorted(reviewed_files.items())
                }
                authorization_fields = {
                    "mode": "cpu_audit_reviewed_commit",
                    "reviewed_files": reviewed_inventory,
                }
            else:
                assert input_pairs is not None
                input_inventory = {}
                for public_name, pair in input_pairs.items():
                    input_inventory[public_name] = {
                        "path": pair["path"],
                        "sha256": pair["sha256"],
                        "byte_count": len(inputs[public_name]),
                        "commit": reviewed_commit,
                    }
                assert authorization is not None and authorization_raw is not None
                assert implementation_commit is not None
                authorization_fields = {
                    "authorization": authorization,
                    "authorization_raw_sha256": _sha256(authorization_raw),
                    "authorization_binding": {
                        "path": AUTHORIZATION_PATH,
                        "sha256": _sha256(authorization_raw),
                        "commit": parsed.authorization_commit,
                    },
                    "commit_topology": _commit_topology(
                        implementation_commit=implementation_commit,
                        cpu_audit_commit=reviewed_commit,
                        authorization_commit=parsed.authorization_commit,
                    ),
                }
            return {
                "schema": CPU_LAUNCH_SCHEMA if cpu_mode else LAUNCH_SCHEMA,
                "trust_root": TRUST_ROOT_ID,
                "repo_root": str(root),
                **authorization_fields,
                "reviewed_commit": reviewed_commit,
                "python": python_identity,
                "git": {
                    "executable": GIT,
                    "sha256": GIT_SHA256,
                    "version": GIT_VERSION,
                    "replace_objects": False,
                    "config_isolated": True,
                    "alternates": False,
                },
                "sys_path": list(sanitized_path),
                "repository_sys_path_entries": [],
                "cwd_sys_path_entries": [],
                "launcher": {
                    "path": LAUNCHER_PATH,
                    "sha256": _sha256(reviewed_launcher_bytes),
                    "origin": str(root / LAUNCHER_PATH),
                    "loader_id": TRUST_ROOT_ID,
                    "execution_count": 1,
                    "cached": None,
                },
                "inputs": input_inventory,
                "module_execution": finder.execution_record(),
            }

        context = LauncherContext(
            module=target,
            entrypoint=entrypoint,
            evidence_factory=evidence_factory,
            input_bytes=inputs,
            reviewed_file_bytes=reviewed_files,
        )
        result = int(
            entrypoint(
                parsed.tool_arguments,
                launcher_context=context,
            )
        )
        if not context.consumed:
            unread = ", ".join(context.unread_inputs)
            unread_files = ", ".join(context.unread_reviewed_files)
            details = ", ".join(value for value in (unread, unread_files) if value)
            raise RuntimeError(
                "verified tool did not consume its launcher context and inputs"
                + (f": {details}" if details else "")
            )
        return result
    finally:
        while finder in sys.meta_path:
            sys.meta_path.remove(finder)


if __name__ == "__main__":
    raise SystemExit(main())
