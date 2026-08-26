"""One-shot, generation-free CPU gate for the adaptive-oracle smoke.

Importing this module loads only the Python standard library.  Heavy project
dependencies are imported lazily by the individual validation steps.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import logging
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCHEMA = "adaptive_oracle_engineering_cpu_audit_v1"
AUDIT_STATUS = "passed_warning_free"
OUTPUT_RELATIVE_PATH = (
    "eval-pipeline/configs/adaptive_oracle_engineering_cpu_audit_v1.json"
)
ENVIRONMENT_LOCK_PATH = (
    "eval-pipeline/configs/generation_environment_adaptive_oracle_v2.yaml"
)
PROMPT_CSV_PATH = "eval-pipeline/prompts/adaptive_oracle_engineering.csv"
EXCLUSION_INVENTORY_PATH = (
    "eval-pipeline/prompts/adaptive_oracle_exclusion_inventory_v1.json"
)
PROMPT_MANIFEST_PATH = (
    "eval-pipeline/prompts/adaptive_oracle_prompt_manifest_v1.json"
)
REGISTRATION_PATH = (
    "eval-pipeline/configs/adaptive_oracle_engineering_registration_v1.yaml"
)
PARTI_SOURCE_PATH = "/tmp/parti/PartiPrompts.tsv"
PARTI_SOURCE_REPOSITORY = "/tmp/parti"

IMPLEMENTATION_SOURCE_PATHS = (
    "AttentionGuidance/__init__.py",
    "AttentionGuidance/adaptive_oracle.py",
    "AttentionGuidance/ancestral_correction.py",
    "AttentionGuidance/attention_guidance.py",
    "AttentionGuidance/controller.py",
    "AttentionGuidance/freeu.py",
    "AttentionGuidance/latent_renderer.py",
    "AttentionGuidance/local_relational_basis.py",
    "AttentionGuidance/semantic_transport.py",
    "AttentionGuidance/types.py",
    "InferencePipelines/__init__.py",
    "InferencePipelines/RepLDM/pipeline_repldm_sdxl.py",
    "InferencePipelines/cfg_batch.py",
    "eval-pipeline/adaptive_oracle_contract.py",
    "eval-pipeline/adaptive_oracle_model_snapshot.py",
    "eval-pipeline/audit_adaptive_oracle_cpu.py",
    "eval-pipeline/audit_adaptive_oracle_engineering.py",
    "eval-pipeline/build_adaptive_oracle_prompts.py",
    "eval-pipeline/generate_adaptive_oracle_engineering.py",
    "eval-pipeline/adaptive_oracle_generation_environment.py",
    "eval-pipeline/launch_adaptive_oracle_tool.py",
)
TEST_SOURCE_PATHS = (
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
TEST_MODULES = tuple(
    path[:-3].replace("/", ".") for path in TEST_SOURCE_PATHS
)
REGISTERED_INPUT_PATHS = (
    ENVIRONMENT_LOCK_PATH,
    PROMPT_CSV_PATH,
    EXCLUSION_INVENTORY_PATH,
    PROMPT_MANIFEST_PATH,
    REGISTRATION_PATH,
)
AUDITED_TRACKED_PATHS = tuple(
    sorted(
        set(IMPLEMENTATION_SOURCE_PATHS)
        | set(TEST_SOURCE_PATHS)
        | set(REGISTERED_INPUT_PATHS)
    )
)
TEST_ENVIRONMENT_OVERRIDES = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
CPU_LAUNCH_SCHEMA = "adaptive_oracle_explicit_reviewed_cpu_launch_v1"
TRUSTED_SITE_PACKAGES = (
    "/home/bycao/miniforge3/envs/diff_attn/lib/python3.11/site-packages"
)
TEST_BOOTSTRAP = """\
import pathlib
import sys
import unittest

root = pathlib.Path.cwd().resolve(strict=True)
sys.path.extend([
    "/home/bycao/miniforge3/envs/diff_attn/lib/python3.11/site-packages",
    str(root),
    str(root / "eval-pipeline"),
])
suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
AUDIT_SCOPE = {
    "cpu_validation_only": True,
    "gpu_generation": False,
    "scoring": False,
    "quality_inspection": False,
    "method_selection": False,
    "renderer_training": False,
    "rl": False,
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_WARNING_LINE = re.compile(
    r"(?im)^(?:[^\r\n]*:\s*)?[A-Z][A-Za-z]*Warning:\s*"
)
_TEST_SUMMARY = re.compile(r"(?m)^Ran ([1-9][0-9]*) tests? in [^\r\n]+$")
_TEST_OK = re.compile(r"(?m)^OK(?: \([^\r\n]*\))?$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "status",
        "one_shot",
        "scope",
        "implementation",
        "registered_inputs",
        "tests",
        "prompt_replay",
        "design",
        "registration",
        "environment",
        "warnings",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_fields(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"{label} fields differ: expected {sorted(expected)}, got {observed}"
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _canonical_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        raise ValueError(f"{label} must be a canonical repository-relative path")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    search_entry = str(path.parent.resolve(strict=True))
    sys.path.insert(0, search_entry)
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == search_entry:
            del sys.path[0]
        else:
            sys.path.remove(search_entry)
    return module


def _file_record(root: Path, relative_path: str) -> dict[str, Any]:
    relative = _canonical_relative_path(relative_path, "audited file path")
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"audited file is missing: {relative}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"audited file must be regular and non-symlink: {relative}")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size": metadata.st_size,
    }


def _file_records(root: Path, paths: Sequence[str]) -> list[dict[str, Any]]:
    return [_file_record(root, path) for path in paths]


def run_exact_test_suite(repo_root: Path) -> dict[str, Any]:
    """Run only the frozen unit-test module allowlist on CPU."""

    root = Path(repo_root).resolve(strict=True)
    arguments = [
        "-I",
        "-B",
        "-S",
        "-W",
        "error",
        "-c",
        TEST_BOOTSTRAP,
        *TEST_MODULES,
    ]
    environment = {
        **TEST_ENVIRONMENT_OVERRIDES,
        "LC_ALL": "C",
    }
    try:
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("adaptive-oracle CPU test allowlist timed out") from exc
    stdout = bytes(result.stdout)
    stderr = bytes(result.stderr)
    combined = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            "adaptive-oracle CPU test allowlist failed:\n" + combined[-12000:]
        )
    if stderr:
        raise RuntimeError("adaptive-oracle CPU tests emitted stderr bytes")
    stdout_text = stdout.decode("utf-8", errors="replace")
    summaries = _TEST_SUMMARY.findall(stdout_text)
    if len(summaries) != 1 or _TEST_OK.search(stdout_text) is None:
        raise RuntimeError("unit-test runner did not emit one successful test summary")
    warning_count = len(_WARNING_LINE.findall(stdout_text))
    if warning_count:
        raise RuntimeError("adaptive-oracle CPU tests emitted warnings")
    return {
        "status": "passed_warning_free",
        "allowlist": list(TEST_MODULES),
        "arguments": arguments,
        "environment_overrides": dict(TEST_ENVIRONMENT_OVERRIDES),
        "interpreter": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "return_code": 0,
        "test_count": int(summaries[0]),
        "warnings_as_errors": True,
        "warning_count": 0,
        "stdout_byte_count": len(stdout),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_byte_count": len(stderr),
        "stderr_sha256": sha256_bytes(stderr),
    }


def replay_prompt_assets(
    repo_root: Path,
    *,
    source_path: Path = Path(PARTI_SOURCE_PATH),
    source_repository: Path = Path(PARTI_SOURCE_REPOSITORY),
) -> dict[str, Any]:
    """Rebuild prompt assets in memory and require byte-for-byte identity."""

    root = Path(repo_root).resolve(strict=True)
    builder = _load_module(
        root / "eval-pipeline/build_adaptive_oracle_prompts.py",
        "adaptive_oracle_prompt_builder_cpu_audit",
    )
    prompt_dir = root / "eval-pipeline/prompts"
    assets = builder.build_assets(
        Path(source_path),
        prompt_dir,
        Path(source_repository),
        root,
        root / "outputs",
    )
    if set(assets) != set(builder.OUTPUT_NAMES) or any(
        not isinstance(value, bytes) for value in assets.values()
    ):
        raise RuntimeError("prompt builder returned an unexpected asset set")
    mismatches = builder.asset_mismatches(assets, prompt_dir)
    if mismatches:
        raise ValueError("prompt byte replay differs: " + ", ".join(mismatches))
    asset_records = []
    for name in sorted(assets):
        relative = f"eval-pipeline/prompts/{name}"
        asset_records.append(
            {
                "path": relative,
                "sha256": sha256_bytes(assets[name]),
                "size": len(assets[name]),
            }
        )
    return {
        "status": "passed_byte_exact",
        "upstream_repository": builder.SOURCE_REPOSITORY,
        "source_repository": str(Path(source_repository).resolve(strict=True)),
        "source_revision": builder.SOURCE_REVISION,
        "source_path": str(Path(source_path).resolve(strict=True)),
        "source_sha256": builder.SOURCE_SHA256,
        "assets": asset_records,
        "mismatch_count": 0,
    }


def validate_reviewed_prompt_assets(
    repo_root: Path,
    *,
    source_path: Path = Path(PARTI_SOURCE_PATH),
    source_repository: Path = Path(PARTI_SOURCE_REPOSITORY),
) -> dict[str, Any]:
    """Validate reviewed prompt blobs against the frozen upstream source."""

    root = Path(repo_root).resolve(strict=True)
    builder = _load_module(
        root / "eval-pipeline/build_adaptive_oracle_prompts.py",
        "adaptive_oracle_prompt_builder_reviewed_cpu_audit",
    )
    source_candidate = Path(source_path)
    descriptor = os.open(
        source_candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Parti source must be a regular non-symlink file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("Parti source changed while it was verified")
    source_bytes = b"".join(chunks)
    if len(source_bytes) != before.st_size or sha256_bytes(source_bytes) != builder.SOURCE_SHA256:
        raise ValueError("Parti source bytes differ from the frozen SHA-256")
    git_result = subprocess.run(
        [
            "/usr/bin/git",
            "--no-replace-objects",
            "-C",
            str(Path(source_repository).resolve(strict=True)),
            "cat-file",
            "blob",
            f"{builder.SOURCE_REVISION}:PartiPrompts.tsv",
        ],
        env={
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if git_result.returncode != 0 or bytes(git_result.stdout) != source_bytes:
        raise ValueError("Parti source differs from the frozen reviewed Git blob")

    inventory_path = root / EXCLUSION_INVENTORY_PATH
    prompt_dir = root / "eval-pipeline/prompts"
    assets = builder.build_assets_from_frozen_inventory(
        source_candidate,
        inventory_path,
        Path(source_repository),
    )
    if set(assets) != set(builder.OUTPUT_NAMES) or any(
        not isinstance(value, bytes) for value in assets.values()
    ):
        raise RuntimeError("frozen prompt replay returned an unexpected asset set")
    mismatches = builder.asset_mismatches(assets, prompt_dir)
    if mismatches:
        raise ValueError("reviewed prompt byte replay differs: " + ", ".join(mismatches))

    try:
        source_text = source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("Parti source is not strict UTF-8") from exc
    reader = csv.DictReader(io.StringIO(source_text, newline=""), delimiter="\t")
    required = {"Prompt", "Category", "Challenge", "Note"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("Parti source has an unexpected schema")
    upstream_rows = []
    for index, row in enumerate(reader):
        prompt = row["Prompt"].strip()
        challenge = row["Challenge"].strip()
        if not prompt or not challenge:
            raise ValueError("Parti source contains an empty prompt or challenge")
        upstream_rows.append(
            {
                "source_row": index,
                "prompt": prompt,
                "category": row["Category"].strip(),
                "challenge": challenge,
                "note": row["Note"].strip(),
            }
        )
    upstream_by_index = {
        int(row["source_row"]): row for row in upstream_rows
    }
    csv_path = root / PROMPT_CSV_PATH
    manifest_path = root / PROMPT_MANIFEST_PATH
    csv_bytes = csv_path.read_bytes()
    inventory_bytes = inventory_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    prompt_rows = _read_prompt_rows(csv_path)
    manifest = _load_json(manifest_path, label="reviewed prompt manifest")
    source = manifest.get("source")
    exclusion_inventory = manifest.get("exclusion_inventory")
    engineering = manifest.get("engineering")
    seed_registration = manifest.get("seed_registration")
    if not isinstance(source, Mapping) or not isinstance(engineering, Mapping):
        raise ValueError("reviewed prompt manifest source or engineering block is missing")
    if not isinstance(seed_registration, Mapping):
        raise ValueError("reviewed prompt manifest seed registration is missing")
    if exclusion_inventory != {
        "schema": builder.EXCLUSION_INVENTORY_SCHEMA,
        "path": EXCLUSION_INVENTORY_PATH,
        "sha256": sha256_bytes(inventory_bytes),
    }:
        raise ValueError("reviewed prompt manifest exclusion inventory binding differs")
    if (
        source.get("repository") != builder.SOURCE_REPOSITORY
        or source.get("revision") != builder.SOURCE_REVISION
        or source.get("file_sha256") != builder.SOURCE_SHA256
        or engineering.get("csv") != PROMPT_CSV_PATH
        or engineering.get("csv_sha256") != sha256_bytes(csv_bytes)
    ):
        raise ValueError("reviewed prompt assets differ from their frozen source binding")
    manifest_rows = engineering.get("prompts")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != len(prompt_rows):
        raise ValueError("reviewed prompt CSV and manifest row counts differ")
    manifest_by_id = {
        row.get("prompt_row_id"): row
        for row in manifest_rows
        if isinstance(row, Mapping) and isinstance(row.get("prompt_row_id"), str)
    }
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("reviewed prompt manifest row identities are invalid")
    registered_seed = seed_registration.get("engineering_seed")
    if not isinstance(registered_seed, Mapping):
        raise ValueError("reviewed engineering seed registration is missing")
    expected_seed = registered_seed.get("seed")
    for row in prompt_rows:
        row_id = row.get("prompt_row_id")
        manifest_row = manifest_by_id.get(row_id)
        try:
            source_row = int(row.get("source_row", ""))
            observed_seed = int(row.get("seed", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError("reviewed prompt CSV identity fields are invalid") from exc
        upstream = upstream_by_index.get(source_row)
        if upstream is None or manifest_row is None:
            raise ValueError("reviewed prompt row is absent from its frozen source")
        prompt = str(row.get("TEXT", ""))
        normalized = builder.normalize_prompt(prompt)
        if (
            prompt != upstream["prompt"]
            or row.get("source_category") != upstream["category"]
            or row.get("source_challenge") != upstream["challenge"]
            or row.get("source_note") != upstream["note"]
            or observed_seed != expected_seed
            or manifest_row.get("source_row") != source_row
            or manifest_row.get("prompt_sha256")
            != sha256_bytes(prompt.encode("utf-8"))
            or manifest_row.get("normalized_prompt_sha256")
            != sha256_bytes(normalized.encode("utf-8"))
            or manifest_row.get("selection_digest")
            != builder._selection_digest(upstream)
        ):
            raise ValueError("reviewed prompt row differs from its frozen source lineage")
    return {
        "status": "passed_byte_exact",
        "upstream_repository": builder.SOURCE_REPOSITORY,
        "source_repository": str(Path(source_repository).resolve(strict=True)),
        "source_revision": builder.SOURCE_REVISION,
        "source_path": str(Path(source_path).resolve(strict=True)),
        "source_sha256": builder.SOURCE_SHA256,
        "assets": [
            {
                "path": f"eval-pipeline/prompts/{name}",
                "sha256": sha256_bytes(assets[name]),
                "size": len(assets[name]),
            }
            for name in sorted(assets)
        ],
        "mismatch_count": 0,
    }


def _read_prompt_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("adaptive-oracle prompt CSV has no header")
        return list(reader)


def validate_engineering_design(repo_root: Path) -> dict[str, Any]:
    """Construct and strictly replay the canonical 11 by 15 task design."""

    root = Path(repo_root).resolve(strict=True)
    contract = _load_module(
        root / "eval-pipeline/adaptive_oracle_contract.py",
        "adaptive_oracle_contract_cpu_audit",
    )
    prompt_rows = _read_prompt_rows(root / PROMPT_CSV_PATH)
    prompt_manifest = _load_json(
        root / PROMPT_MANIFEST_PATH, label="adaptive-oracle prompt manifest"
    )
    design = contract.build_engineering_design(prompt_rows, prompt_manifest)
    validated = contract.validate_engineering_design(
        design, prompt_rows, prompt_manifest
    )
    if validated.get("task_count") != 165 or design.get("tasks_per_prompt") != 15:
        raise RuntimeError("adaptive-oracle engineering design is not exactly 165 tasks")
    return {
        "status": "passed",
        "prompt_count": 11,
        "tasks_per_prompt": 15,
        **validated,
    }


def validate_engineering_registration(repo_root: Path) -> dict[str, Any]:
    """Rebuild and compare the exact non-executable registration semantics."""

    root = Path(repo_root).resolve(strict=True)
    generator = _load_module(
        root / "eval-pipeline/generate_adaptive_oracle_engineering.py",
        "adaptive_oracle_generator_registration_cpu_audit",
    )
    prompt_csv = root / PROMPT_CSV_PATH
    exclusion_inventory_path = root / EXCLUSION_INVENTORY_PATH
    prompt_manifest_path = root / PROMPT_MANIFEST_PATH
    prompt_rows = _read_prompt_rows(prompt_csv)
    prompt_manifest = _load_json(
        prompt_manifest_path, label="adaptive-oracle prompt manifest"
    )
    design = generator.contract.build_engineering_design(
        prompt_rows, prompt_manifest
    )
    registration_path = root / REGISTRATION_PATH
    registration, registration_sha256 = generator._load_unique_yaml_bytes(
        registration_path.read_bytes(), label="adaptive-oracle registration"
    )
    environment_lock_sha256 = sha256_file(root / ENVIRONMENT_LOCK_PATH)
    expected = generator._expected_registration(
        design=design,
        prompt_csv_sha256=sha256_file(prompt_csv),
        exclusion_inventory_sha256=sha256_file(exclusion_inventory_path),
        prompt_manifest_sha256=sha256_file(prompt_manifest_path),
        environment_lock_sha256=environment_lock_sha256,
    )
    if canonical_json_bytes(registration) != canonical_json_bytes(expected):
        raise ValueError("adaptive-oracle registration differs from the exact schema")
    return {
        "status": "passed_exact_non_executable",
        "schema": generator.REGISTRATION_SCHEMA,
        "sha256": registration_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "exclusion_inventory_sha256": sha256_file(exclusion_inventory_path),
        "design_sha256": design["design_sha256"],
    }


def validate_pinned_environment(
    repo_root: Path, *, environment_lock_sha256: str
) -> dict[str, Any]:
    """Call the generator's existing environment-lock validator without loading a model."""

    root = Path(repo_root).resolve(strict=True)
    helper = _load_module(
        root / "eval-pipeline/adaptive_oracle_generation_environment.py",
        "adaptive_oracle_generation_environment_cpu_audit",
    )
    lock_path = root / ENVIRONMENT_LOCK_PATH
    validated = helper.validate_environment_lock(
        str(lock_path),
        expected_sha256=environment_lock_sha256,
        cuda_device="cuda:1",
        require_unmasked_cuda=True,
    )
    if not isinstance(validated, Mapping):
        raise RuntimeError("environment helper returned a non-mapping record")
    if Path(str(validated.get("path", ""))).resolve(strict=True) != lock_path:
        raise RuntimeError("environment helper validated an unexpected lock path")
    if validated.get("sha256") != environment_lock_sha256:
        raise RuntimeError("environment helper returned a stale lock hash")
    normalized = json.loads(json.dumps(validated, allow_nan=False, sort_keys=True))
    return {
        "status": "passed",
        "lock_path": ENVIRONMENT_LOCK_PATH,
        "lock_sha256": environment_lock_sha256,
        "helper_schema": normalized.get("schema"),
        "lock_id": normalized.get("lock_id"),
        "observed": normalized.get("observed"),
    }


def _records_by_path(
    value: Any, expected_paths: Sequence[str], label: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(expected_paths):
        raise ValueError(f"{label} must contain the exact path allowlist")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        row = _require_exact_fields(
            raw, frozenset({"path", "sha256", "size"}), f"{label}[{index}]"
        )
        path = _canonical_relative_path(row["path"], f"{label}[{index}].path")
        if path in result:
            raise ValueError(f"{label} contains duplicate path {path!r}")
        result[path] = {
            "path": path,
            "sha256": _require_sha256(row["sha256"], f"{label}[{index}].sha256"),
            "size": _require_nonnegative_int(row["size"], f"{label}[{index}].size"),
        }
    if list(result) != list(expected_paths):
        raise ValueError(f"{label} order or membership differs from the allowlist")
    return result


def _validate_cpu_audit_mapping(
    value: Mapping[str, Any],
    *,
    expected_implementation_commit: Optional[str] = None,
    expected_implementation_hashes: Optional[Mapping[str, str]] = None,
    expected_environment_lock_sha256: Optional[str] = None,
    expected_contract_sha256: Optional[str] = None,
    expected_design_sha256: Optional[str] = None,
    expected_tasks_sha256: Optional[str] = None,
    expected_prompt_csv_sha256: Optional[str] = None,
    expected_exclusion_inventory_sha256: Optional[str] = None,
    expected_prompt_manifest_sha256: Optional[str] = None,
    expected_registration_sha256: Optional[str] = None,
) -> dict[str, Any]:
    top = _require_exact_fields(value, _TOP_LEVEL_FIELDS, "CPU audit")
    if top["schema"] != AUDIT_SCHEMA or top["status"] != AUDIT_STATUS:
        raise ValueError("CPU audit is not a warning-free v1 pass")
    if top["one_shot"] is not True or top["scope"] != AUDIT_SCOPE:
        raise ValueError("CPU audit scope or one-shot marker differs")

    implementation = _require_exact_fields(
        top["implementation"],
        frozenset({"commit", "clean", "source_files"}),
        "CPU audit implementation",
    )
    commit = implementation["commit"]
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("CPU audit implementation commit must be full lowercase Git SHA")
    if implementation["clean"] is not True:
        raise ValueError("CPU audit implementation sources were not clean")
    implementation_rows = _records_by_path(
        implementation["source_files"],
        IMPLEMENTATION_SOURCE_PATHS,
        "CPU audit implementation sources",
    )
    implementation_hashes = {
        path: row["sha256"] for path, row in implementation_rows.items()
    }
    if expected_implementation_commit is not None and commit != expected_implementation_commit:
        raise ValueError("CPU audit implementation commit differs from authorization")
    if expected_implementation_hashes is not None:
        normalized_expected = {
            _canonical_relative_path(path, "expected implementation path"): _require_sha256(
                digest, f"expected implementation hash for {path}"
            )
            for path, digest in expected_implementation_hashes.items()
        }
        if implementation_hashes != normalized_expected:
            raise ValueError("CPU audit implementation source hashes differ")

    input_rows = _records_by_path(
        top["registered_inputs"], REGISTERED_INPUT_PATHS, "CPU audit registered inputs"
    )
    test = _require_exact_fields(
        top["tests"],
        frozenset(
            {
                "status",
                "allowlist",
                "arguments",
                "environment_overrides",
                "interpreter",
                "return_code",
                "test_count",
                "warnings_as_errors",
                "warning_count",
                "stdout_byte_count",
                "stdout_sha256",
                "stderr_byte_count",
                "stderr_sha256",
                "source_files",
            }
        ),
        "CPU audit tests",
    )
    if test["status"] != "passed_warning_free":
        raise ValueError("CPU audit tests did not pass warning-free")
    if test["allowlist"] != list(TEST_MODULES):
        raise ValueError("CPU audit test module allowlist differs")
    expected_arguments = [
        "-I",
        "-B",
        "-S",
        "-W",
        "error",
        "-c",
        TEST_BOOTSTRAP,
        *TEST_MODULES,
    ]
    if test["arguments"] != expected_arguments:
        raise ValueError("CPU audit unittest arguments differ")
    if test["environment_overrides"] != TEST_ENVIRONMENT_OVERRIDES:
        raise ValueError("CPU audit test environment is not the frozen CPU-only environment")
    interpreter = _require_exact_fields(
        test["interpreter"],
        frozenset({"executable", "implementation", "version"}),
        "CPU audit interpreter",
    )
    if any(not isinstance(interpreter[key], str) or not interpreter[key] for key in interpreter):
        raise ValueError("CPU audit interpreter fields must be non-empty strings")
    if (
        test["return_code"] != 0
        or isinstance(test["return_code"], bool)
        or test["warnings_as_errors"] is not True
        or test["warning_count"] != 0
        or isinstance(test["warning_count"], bool)
    ):
        raise ValueError("CPU audit test exit or warning contract failed")
    if (
        isinstance(test["test_count"], bool)
        or not isinstance(test["test_count"], int)
        or test["test_count"] <= 0
    ):
        raise ValueError("CPU audit must report a positive test count")
    for stream in ("stdout", "stderr"):
        _require_nonnegative_int(test[f"{stream}_byte_count"], f"{stream} byte count")
        _require_sha256(test[f"{stream}_sha256"], f"{stream} SHA-256")
    if (
        test["stderr_byte_count"] != 0
        or test["stderr_sha256"] != sha256_bytes(b"")
    ):
        raise ValueError("CPU audit test stderr evidence must be empty")
    _records_by_path(
        test["source_files"], TEST_SOURCE_PATHS, "CPU audit test sources"
    )

    replay = _require_exact_fields(
        top["prompt_replay"],
        frozenset(
            {
                "status",
                "upstream_repository",
                "source_repository",
                "source_revision",
                "source_path",
                "source_sha256",
                "assets",
                "mismatch_count",
            }
        ),
        "CPU audit prompt replay",
    )
    if replay["status"] != "passed_byte_exact" or replay["mismatch_count"] != 0:
        raise ValueError("CPU audit prompt assets did not replay byte-for-byte")
    for key in (
        "upstream_repository",
        "source_repository",
        "source_revision",
        "source_path",
    ):
        if not isinstance(replay[key], str) or not replay[key]:
            raise ValueError(f"CPU audit prompt replay {key} must be non-empty")
    _require_sha256(replay["source_sha256"], "Parti source SHA-256")
    asset_rows = _records_by_path(
        replay["assets"],
        (PROMPT_CSV_PATH, EXCLUSION_INVENTORY_PATH, PROMPT_MANIFEST_PATH),
        "CPU audit replayed prompt assets",
    )
    if asset_rows[PROMPT_CSV_PATH]["sha256"] != input_rows[PROMPT_CSV_PATH]["sha256"]:
        raise ValueError("CPU audit replayed prompt CSV hash differs from registered input")
    if asset_rows[PROMPT_MANIFEST_PATH]["sha256"] != input_rows[PROMPT_MANIFEST_PATH]["sha256"]:
        raise ValueError("CPU audit replayed prompt manifest hash differs from registered input")
    if asset_rows[EXCLUSION_INVENTORY_PATH]["sha256"] != input_rows[EXCLUSION_INVENTORY_PATH]["sha256"]:
        raise ValueError("CPU audit replayed exclusion inventory hash differs from registered input")

    design = _require_exact_fields(
        top["design"],
        frozenset(
            {
                "status",
                "prompt_count",
                "tasks_per_prompt",
                "task_count",
                "tasks_sha256",
                "design_sha256",
                "contract_sha256",
                "primary_action_bank_sha256",
                "signed_orbit_cycle_sha256",
                "execution_contract_sha256",
                "sidecar_shape_sha256",
            }
        ),
        "CPU audit design",
    )
    if (
        design["status"] != "passed"
        or design["prompt_count"] != 11
        or design["tasks_per_prompt"] != 15
        or design["task_count"] != 165
    ):
        raise ValueError("CPU audit did not validate the exact 11 by 15 design")
    for key in (
        "tasks_sha256",
        "design_sha256",
        "contract_sha256",
        "primary_action_bank_sha256",
        "signed_orbit_cycle_sha256",
        "execution_contract_sha256",
        "sidecar_shape_sha256",
    ):
        _require_sha256(design[key], f"CPU audit design {key}")

    registration = _require_exact_fields(
        top["registration"],
        frozenset(
            {
                "status",
                "schema",
                "sha256",
                "environment_lock_sha256",
                "exclusion_inventory_sha256",
                "design_sha256",
            }
        ),
        "CPU audit registration",
    )
    if (
        registration["status"] != "passed_exact_non_executable"
        or registration["schema"]
        != "adaptive_oracle_engineering_registration_v1"
    ):
        raise ValueError("CPU audit did not validate the non-executable registration")
    registration_sha256 = _require_sha256(
        registration["sha256"], "CPU audit registration SHA-256"
    )
    if registration_sha256 != input_rows[REGISTRATION_PATH]["sha256"]:
        raise ValueError("CPU audit registration hash differs from registered input")
    if (
        registration["environment_lock_sha256"]
        != input_rows[ENVIRONMENT_LOCK_PATH]["sha256"]
    ):
        raise ValueError("CPU audit registration environment lock hash differs")
    if registration["design_sha256"] != design["design_sha256"]:
        raise ValueError("CPU audit registration design hash differs")
    registration_inventory_sha256 = _require_sha256(
        registration["exclusion_inventory_sha256"],
        "registration exclusion inventory SHA-256",
    )
    if registration_inventory_sha256 != input_rows[EXCLUSION_INVENTORY_PATH]["sha256"]:
        raise ValueError("CPU audit registration exclusion inventory hash differs")

    environment = _require_exact_fields(
        top["environment"],
        frozenset(
            {"status", "lock_path", "lock_sha256", "helper_schema", "lock_id", "observed"}
        ),
        "CPU audit environment",
    )
    if environment["status"] != "passed" or environment["lock_path"] != ENVIRONMENT_LOCK_PATH:
        raise ValueError("CPU audit environment lock did not pass")
    lock_sha256 = _require_sha256(environment["lock_sha256"], "environment lock SHA-256")
    if lock_sha256 != input_rows[ENVIRONMENT_LOCK_PATH]["sha256"]:
        raise ValueError("CPU audit environment lock hash differs from registered input")
    if environment["helper_schema"] != "repldm_generation_environment_lock_v2":
        raise ValueError("CPU audit environment helper schema differs")
    if not isinstance(environment["lock_id"], str) or not environment["lock_id"]:
        raise ValueError("CPU audit environment lock_id is missing")
    if not isinstance(environment["observed"], Mapping):
        raise ValueError("CPU audit observed environment must be a mapping")

    warning_record = _require_exact_fields(
        top["warnings"],
        frozenset({"count", "same_process_count", "test_subprocess_count"}),
        "CPU audit warnings",
    )
    if warning_record != {
        "count": 0,
        "same_process_count": 0,
        "test_subprocess_count": 0,
    }:
        raise ValueError("CPU audit is not warning-free")

    expectations = (
        (expected_environment_lock_sha256, lock_sha256, "environment lock"),
        (expected_contract_sha256, design["contract_sha256"], "contract"),
        (expected_design_sha256, design["design_sha256"], "design"),
        (expected_tasks_sha256, design["tasks_sha256"], "tasks"),
        (
            expected_prompt_csv_sha256,
            asset_rows[PROMPT_CSV_PATH]["sha256"],
            "prompt CSV",
        ),
        (
            expected_exclusion_inventory_sha256,
            asset_rows[EXCLUSION_INVENTORY_PATH]["sha256"],
            "exclusion inventory",
        ),
        (
            expected_prompt_manifest_sha256,
            asset_rows[PROMPT_MANIFEST_PATH]["sha256"],
            "prompt manifest",
        ),
        (
            expected_registration_sha256,
            registration_sha256,
            "registration",
        ),
    )
    for expected, observed, label in expectations:
        if expected is not None:
            _require_sha256(expected, f"expected {label} SHA-256")
            if observed != expected:
                raise ValueError(f"CPU audit {label} SHA-256 differs from authorization")

    return {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "implementation_commit": commit,
        "implementation_hashes": implementation_hashes,
        "environment_lock_sha256": lock_sha256,
        "contract_sha256": design["contract_sha256"],
        "design_sha256": design["design_sha256"],
        "tasks_sha256": design["tasks_sha256"],
        "prompt_csv_sha256": asset_rows[PROMPT_CSV_PATH]["sha256"],
        "exclusion_inventory_sha256": asset_rows[EXCLUSION_INVENTORY_PATH]["sha256"],
        "prompt_manifest_sha256": asset_rows[PROMPT_MANIFEST_PATH]["sha256"],
        "registration_sha256": input_rows[REGISTRATION_PATH]["sha256"],
        "test_count": test["test_count"],
        "warning_count": 0,
    }


def validate_cpu_audit_record(
    path: str | os.PathLike[str],
    *,
    expected_implementation_commit: Optional[str] = None,
    expected_implementation_hashes: Optional[Mapping[str, str]] = None,
    expected_environment_lock_sha256: Optional[str] = None,
    expected_contract_sha256: Optional[str] = None,
    expected_design_sha256: Optional[str] = None,
    expected_tasks_sha256: Optional[str] = None,
    expected_prompt_csv_sha256: Optional[str] = None,
    expected_exclusion_inventory_sha256: Optional[str] = None,
    expected_prompt_manifest_sha256: Optional[str] = None,
    expected_registration_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Purely validate a CPU audit and optional authorization bindings."""

    value = _load_json(Path(path), label="adaptive-oracle CPU audit")
    return _validate_cpu_audit_mapping(
        value,
        expected_implementation_commit=expected_implementation_commit,
        expected_implementation_hashes=expected_implementation_hashes,
        expected_environment_lock_sha256=expected_environment_lock_sha256,
        expected_contract_sha256=expected_contract_sha256,
        expected_design_sha256=expected_design_sha256,
        expected_tasks_sha256=expected_tasks_sha256,
        expected_prompt_csv_sha256=expected_prompt_csv_sha256,
        expected_exclusion_inventory_sha256=expected_exclusion_inventory_sha256,
        expected_prompt_manifest_sha256=expected_prompt_manifest_sha256,
        expected_registration_sha256=expected_registration_sha256,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON atomically and refuse every overwrite race."""

    payload = canonical_json_bytes(value)
    parent = path.parent
    directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    temporary: Optional[str] = None
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        if os.path.lexists(path):
            raise FileExistsError(f"CPU audit already exists: {path}")
        descriptor, temporary = tempfile.mkstemp(
            dir=parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"CPU audit already exists: {path}") from exc
        _fsync_directory(parent)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


def _canonical_output_path(repo_root: Path, output: str | os.PathLike[str]) -> Path:
    root = Path(repo_root).resolve(strict=True)
    expected = root.joinpath(*PurePosixPath(OUTPUT_RELATIVE_PATH).parts)
    raw = Path(output)
    candidate = raw if raw.is_absolute() else root / raw
    if Path(os.path.abspath(candidate)) != expected:
        raise ValueError(f"CPU audit output must be exactly {expected}")
    parent = expected.parent.resolve(strict=True)
    if parent != expected.parent or root not in parent.parents:
        raise ValueError("CPU audit output parent must be a non-symlink repository directory")
    return expected


def create_cpu_audit(
    output: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str] = ROOT,
) -> dict[str, Any]:
    """Reject publication that bypasses the explicit reviewed-commit launcher."""

    del output, repo_root
    raise RuntimeError(
        "CPU-audit publication requires the verified reviewed-commit launcher"
    )


class _CpuAuditLoggingCapture(logging.Handler):
    """Collect log records without formatting them or writing another stream."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _run_warning_free_cpu_gate(callback: Callable[[], Any]) -> Any:
    """Run gate callbacks while capturing every in-process stderr channel."""

    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = tuple(root_logger.handlers)
    log_capture = _CpuAuditLoggingCapture()
    observed_warnings: list[warnings.WarningMessage] = []
    cleanup_errors: list[tuple[str, BaseException]] = []
    callback_error: Optional[tuple[BaseException, Any]] = None
    result: Any = None
    saved_stderr: Optional[int] = None
    stderr_inheritable: Optional[bool] = None
    stderr_redirected = False
    warning_context: Any = None
    stderr_bytes = b""
    capture_file = tempfile.TemporaryFile(mode="w+b")

    try:
        sys.stderr.flush()
        stderr_inheritable = os.get_inheritable(2)
        saved_stderr = os.dup(2)
        os.dup2(capture_file.fileno(), 2, inheritable=True)
        stderr_redirected = True
        root_logger.addHandler(log_capture)
        root_logger.setLevel(logging.NOTSET)
        warning_context = warnings.catch_warnings(record=True)
        observed_warnings = warning_context.__enter__()
        warnings.simplefilter("always")
        result = callback()
    except BaseException as exc:
        callback_error = (exc, exc.__traceback__)
    finally:
        for handler in tuple(root_logger.handlers):
            try:
                handler.flush()
            except BaseException as exc:
                cleanup_errors.append(("logging handler flush", exc))
        try:
            sys.stderr.flush()
        except BaseException as exc:
            cleanup_errors.append(("sys.stderr flush", exc))
        try:
            root_logger.handlers[:] = list(original_handlers)
        except BaseException as exc:
            cleanup_errors.append(("logging handlers restore", exc))
        try:
            log_capture.close()
        except BaseException as exc:
            cleanup_errors.append(("logging capture close", exc))
        try:
            root_logger.setLevel(original_level)
        except BaseException as exc:
            cleanup_errors.append(("logging level restore", exc))
        if warning_context is not None:
            try:
                warning_context.__exit__(None, None, None)
            except BaseException as exc:
                cleanup_errors.append(("warnings state restore", exc))
        if stderr_redirected and saved_stderr is not None:
            try:
                os.dup2(
                    saved_stderr,
                    2,
                    inheritable=(
                        stderr_inheritable
                        if stderr_inheritable is not None
                        else True
                    ),
                )
            except BaseException as exc:
                cleanup_errors.append(("stderr fd restore", exc))
        if saved_stderr is not None:
            try:
                os.close(saved_stderr)
            except BaseException as exc:
                cleanup_errors.append(("saved stderr fd close", exc))
        try:
            capture_file.flush()
            capture_file.seek(0)
            stderr_bytes = capture_file.read()
        except BaseException as exc:
            cleanup_errors.append(("stderr evidence read", exc))
        try:
            capture_file.close()
        except BaseException as exc:
            cleanup_errors.append(("stderr evidence close", exc))

    evidence = []
    if observed_warnings:
        labels = sorted({type(row.message).__name__ for row in observed_warnings})
        evidence.append("warnings=" + ",".join(labels))
    if log_capture.records:
        levels = sorted({row.levelname for row in log_capture.records})
        evidence.append(
            f"logging_records={len(log_capture.records)}[{','.join(levels)}]"
        )
    if stderr_bytes:
        evidence.append(f"stderr_bytes={len(stderr_bytes)}")
    if cleanup_errors:
        labels = [f"{label}:{type(exc).__name__}" for label, exc in cleanup_errors]
        evidence.append("cleanup=" + ",".join(labels))
    if evidence:
        cause = callback_error[0] if callback_error is not None else None
        raise RuntimeError("CPU audit gate failed closed: " + "; ".join(evidence)) from cause
    if callback_error is not None:
        error, traceback = callback_error
        raise error.with_traceback(traceback)
    return result


def _build_cpu_audit_record(
    root: Path,
    *,
    source_path: Path = Path(PARTI_SOURCE_PATH),
    source_repository: Path = Path(PARTI_SOURCE_REPOSITORY),
    provenance_collector: Optional[Callable[[Path], Mapping[str, Any]]] = None,
    test_runner: Callable[[Path], Mapping[str, Any]] = run_exact_test_suite,
    prompt_replayer: Optional[Callable[[Path], Mapping[str, Any]]] = None,
    design_validator: Callable[[Path], Mapping[str, Any]] = validate_engineering_design,
    registration_validator: Callable[[Path], Mapping[str, Any]] = validate_engineering_registration,
    environment_validator: Optional[
        Callable[[Path, str], Mapping[str, Any]]
    ] = None,
) -> dict[str, Any]:
    """Assemble and validate a CPU-audit record without publishing any file."""

    root = Path(root).resolve(strict=True)
    if provenance_collector is None:
        raise RuntimeError(
            "CPU audit record construction requires reviewed launcher provenance"
        )

    def build_record() -> dict[str, Any]:
        before = dict(provenance_collector(root))
        required_provenance = frozenset(
            {
                "commit",
                "clean",
                "implementation_sources",
                "test_sources",
                "registered_inputs",
            }
        )
        _require_exact_fields(before, required_provenance, "CPU audit provenance")
        if before["clean"] is not True:
            raise ValueError("audited sources must be clean")
        input_rows = _records_by_path(
            before["registered_inputs"],
            REGISTERED_INPUT_PATHS,
            "pre-audit registered inputs",
        )
        environment_lock_sha256 = input_rows[ENVIRONMENT_LOCK_PATH]["sha256"]

        replay_prompts = prompt_replayer or (
            lambda checked_root: replay_prompt_assets(
                checked_root,
                source_path=source_path,
                source_repository=source_repository,
            )
        )
        validate_environment = environment_validator or (
            lambda checked_root, *, environment_lock_sha256: validate_pinned_environment(
                checked_root, environment_lock_sha256=environment_lock_sha256
            )
        )

        if prompt_replayer is None:
            prompt_record = dict(replay_prompts(root))
        else:
            prompt_record = dict(
                prompt_replayer(
                    root,
                    source_path=source_path,
                    source_repository=source_repository,
                )
            )
        design_record = dict(design_validator(root))
        registration_record = dict(registration_validator(root))
        environment_record = dict(
            validate_environment(
                root, environment_lock_sha256=environment_lock_sha256
            )
        )
        test_record = dict(test_runner(root))

        after = dict(provenance_collector(root))
        if before != after:
            raise RuntimeError("audited commit or file bytes changed while the CPU gate ran")
        record = {
            "schema": AUDIT_SCHEMA,
            "status": AUDIT_STATUS,
            "one_shot": True,
            "scope": dict(AUDIT_SCOPE),
            "implementation": {
                "commit": before["commit"],
                "clean": True,
                "source_files": before["implementation_sources"],
            },
            "registered_inputs": before["registered_inputs"],
            "tests": {**test_record, "source_files": before["test_sources"]},
            "prompt_replay": prompt_record,
            "design": design_record,
            "registration": registration_record,
            "environment": environment_record,
            "warnings": {
                "count": 0,
                "same_process_count": 0,
                "test_subprocess_count": 0,
            },
        }
        _validate_cpu_audit_mapping(
            record,
            expected_implementation_commit=before["commit"],
            expected_implementation_hashes={
                row["path"]: row["sha256"]
                for row in before["implementation_sources"]
            },
            expected_environment_lock_sha256=environment_lock_sha256,
            expected_contract_sha256=design_record.get("contract_sha256"),
            expected_design_sha256=design_record.get("design_sha256"),
            expected_tasks_sha256=design_record.get("tasks_sha256"),
            expected_prompt_csv_sha256=input_rows[PROMPT_CSV_PATH]["sha256"],
            expected_exclusion_inventory_sha256=input_rows[
                EXCLUSION_INVENTORY_PATH
            ]["sha256"],
            expected_prompt_manifest_sha256=input_rows[PROMPT_MANIFEST_PATH][
                "sha256"
            ],
            expected_registration_sha256=input_rows[REGISTRATION_PATH]["sha256"],
        )
        return record

    return _run_warning_free_cpu_gate(build_record)


def _validated_reviewed_snapshot(
    launcher_context: Any, launch_evidence: Mapping[str, Any]
) -> tuple[Path, str, dict[str, bytes], dict[str, Any]]:
    expected_fields = frozenset(
        {
            "schema",
            "trust_root",
            "repo_root",
            "mode",
            "reviewed_files",
            "reviewed_commit",
            "python",
            "git",
            "sys_path",
            "repository_sys_path_entries",
            "cwd_sys_path_entries",
            "launcher",
            "inputs",
            "module_execution",
        }
    )
    top = _require_exact_fields(
        launch_evidence, expected_fields, "CPU launcher evidence"
    )
    if (
        top["schema"] != CPU_LAUNCH_SCHEMA
        or top["mode"] != "cpu_audit_reviewed_commit"
        or top["repository_sys_path_entries"] != []
        or top["cwd_sys_path_entries"] != []
    ):
        raise RuntimeError("CPU launcher isolation evidence differs")
    reviewed_commit = top["reviewed_commit"]
    if not isinstance(reviewed_commit, str) or _COMMIT.fullmatch(reviewed_commit) is None:
        raise RuntimeError("CPU launcher reviewed commit is invalid")
    repo_root = Path(str(top["repo_root"])).resolve(strict=True)
    if repo_root != ROOT.resolve(strict=True):
        raise RuntimeError("CPU launcher repository root differs from the bound module")
    reviewed_inventory = top["reviewed_files"]
    if not isinstance(reviewed_inventory, Mapping) or set(reviewed_inventory) != set(
        AUDITED_TRACKED_PATHS
    ):
        raise RuntimeError("CPU launcher reviewed-file inventory differs")
    reviewed_bytes = {}
    records = {}
    for relative_path in AUDITED_TRACKED_PATHS:
        identity = _require_exact_fields(
            reviewed_inventory[relative_path],
            frozenset({"sha256", "byte_count", "commit"}),
            f"CPU launcher reviewed file {relative_path}",
        )
        expected_hash = _require_sha256(
            identity["sha256"], f"CPU launcher reviewed file {relative_path}"
        )
        expected_size = _require_nonnegative_int(
            identity["byte_count"], f"CPU launcher reviewed file {relative_path}"
        )
        if identity["commit"] != reviewed_commit:
            raise RuntimeError("CPU launcher reviewed-file commit differs")
        value = launcher_context.read_reviewed_file(relative_path)
        if not isinstance(value, bytes):
            raise RuntimeError("CPU launcher returned non-byte reviewed file data")
        if len(value) != expected_size or sha256_bytes(value) != expected_hash:
            raise RuntimeError(
                f"CPU launcher reviewed file bytes differ for {relative_path}"
            )
        reviewed_bytes[relative_path] = value
        records[relative_path] = {
            "path": relative_path,
            "sha256": expected_hash,
            "size": expected_size,
        }
    inputs = top["inputs"]
    expected_inputs = {
        "registration": REGISTRATION_PATH,
        "prompt_csv": PROMPT_CSV_PATH,
        "exclusion_inventory": EXCLUSION_INVENTORY_PATH,
        "prompt_manifest": PROMPT_MANIFEST_PATH,
        "environment_lock": ENVIRONMENT_LOCK_PATH,
    }
    if not isinstance(inputs, Mapping) or set(inputs) != set(expected_inputs):
        raise RuntimeError("CPU launcher input inventory differs")
    for name, relative_path in expected_inputs.items():
        identity = _require_exact_fields(
            inputs[name],
            frozenset({"path", "sha256", "byte_count", "commit"}),
            f"CPU launcher input {name}",
        )
        record = records[relative_path]
        if (
            identity["path"] != relative_path
            or identity["sha256"] != record["sha256"]
            or identity["byte_count"] != record["size"]
            or identity["commit"] != reviewed_commit
        ):
            raise RuntimeError(f"CPU launcher input binding differs for {name}")
    module_execution = top["module_execution"]
    row = (
        module_execution.get("audit_adaptive_oracle_cpu")
        if isinstance(module_execution, Mapping)
        else None
    )
    if not isinstance(row, Mapping) or (
        row.get("path") != "eval-pipeline/audit_adaptive_oracle_cpu.py"
        or row.get("sha256")
        != records["eval-pipeline/audit_adaptive_oracle_cpu.py"]["sha256"]
        or row.get("execution_count") != 1
        or row.get("cached") is not None
        or not str(row.get("origin", "")).startswith(
            f"<git-blob:{reviewed_commit}:"
        )
    ):
        raise RuntimeError("CPU auditor was not executed from its reviewed Git blob")
    return repo_root, reviewed_commit, reviewed_bytes, records


def _materialize_reviewed_snapshot(
    execution_root: Path, reviewed_bytes: Mapping[str, bytes]
) -> None:
    root = execution_root.resolve(strict=True)
    for relative_path in AUDITED_TRACKED_PATHS:
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
            raise RuntimeError("CPU reviewed snapshot path is not canonical")
        path = root.joinpath(*pure.parts)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            value = reviewed_bytes[relative_path]
            offset = 0
            while offset < len(value):
                offset += os.write(descriptor, value[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _create_cpu_audit_from_reviewed_snapshot(
    output: str | os.PathLike[str],
    *,
    output_repo_root: Path,
    execution_root: Path,
    reviewed_commit: str,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output_path = _canonical_output_path(output_repo_root, output)
    if os.path.lexists(output_path):
        raise FileExistsError(f"CPU audit already exists: {output_path}")
    provenance = {
        "commit": reviewed_commit,
        "clean": True,
        "implementation_sources": [
            dict(records[path]) for path in IMPLEMENTATION_SOURCE_PATHS
        ],
        "test_sources": [dict(records[path]) for path in TEST_SOURCE_PATHS],
        "registered_inputs": [
            dict(records[path]) for path in REGISTERED_INPUT_PATHS
        ],
    }
    record = _build_cpu_audit_record(
        execution_root,
        source_path=Path(PARTI_SOURCE_PATH),
        source_repository=Path(PARTI_SOURCE_REPOSITORY),
        provenance_collector=lambda _root: json.loads(
            json.dumps(provenance, allow_nan=False, sort_keys=True)
        ),
        test_runner=run_exact_test_suite,
        prompt_replayer=validate_reviewed_prompt_assets,
        design_validator=validate_engineering_design,
        registration_validator=validate_engineering_registration,
        environment_validator=validate_pinned_environment,
    )
    atomic_create_json(output_path, record)
    return record


def _run_from_verified_launcher(
    argv: Sequence[str], *, launcher_context: Any
) -> int:
    launch_evidence = launcher_context.claim()
    if not isinstance(launch_evidence, Mapping):
        raise RuntimeError("CPU launcher returned non-mapping evidence")
    repo_root, reviewed_commit, reviewed_bytes, records = (
        _validated_reviewed_snapshot(launcher_context, launch_evidence)
    )
    parser = argparse.ArgumentParser(
        description="Create the one-shot adaptive-oracle CPU gate audit."
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv))
    output_path = _canonical_output_path(repo_root, args.output)
    if os.path.lexists(output_path):
        raise FileExistsError(f"CPU audit already exists: {output_path}")
    with tempfile.TemporaryDirectory(prefix="adaptive-oracle-reviewed-cpu-") as temporary:
        execution_root = Path(temporary)
        _materialize_reviewed_snapshot(execution_root, reviewed_bytes)
        record = _create_cpu_audit_from_reviewed_snapshot(
            args.output,
            output_repo_root=repo_root,
            execution_root=execution_root,
            reviewed_commit=reviewed_commit,
            records=records,
        )
    print(
        json.dumps(
            {
                "schema": record["schema"],
                "status": record["status"],
                "output": str(output_path),
                "reviewed_commit": reviewed_commit,
            },
            sort_keys=True,
        )
    )
    return 0


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    execution_context: Optional[Mapping[str, Any]] = None,
) -> int:
    if execution_context is None:
        raise RuntimeError(
            "CPU-audit publication must be invoked by the verified execution-root launcher"
        )
    parser = argparse.ArgumentParser(
        description="Create the one-shot adaptive-oracle CPU gate audit."
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    record = create_cpu_audit(args.output)
    print(
        json.dumps(
            {
                "schema": record["schema"],
                "status": record["status"],
                "output": str(ROOT / OUTPUT_RELATIVE_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
