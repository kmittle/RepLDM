"""Validate the exact package contract used by registered generation runs."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import subprocess
from typing import Mapping, Optional

import torch
import yaml


LOCK_SCHEMA = "repldm_generation_environment_lock_v1"
REQUIRED_RUNTIME_FIELDS = frozenset({"python", "cuda", "cudnn"})
REQUIRED_HARDWARE_FIELDS = frozenset({"gpu", "driver", "compute_capability"})
REQUIRED_DETERMINISM_FIELDS = frozenset(
    {
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    }
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_environment(package_names) -> dict:
    packages = {}
    for name in package_names:
        try:
            packages[str(name)] = version(str(name))
        except PackageNotFoundError:
            packages[str(name)] = None
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = f"{major}.{minor}"
    else:
        gpu = None
        compute_capability = None
    return {
        "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        "runtime": {
            "python": platform.python_version(),
            "cuda": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
        },
        "packages": packages,
        "hardware": {
            "gpu": gpu,
            "driver": driver,
            "compute_capability": compute_capability,
        },
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def validate_environment_lock(
    path: str, *, expected_sha256: Optional[str] = None
) -> dict:
    """Fail closed when the current interpreter differs from a pinned lock."""

    absolute_path = os.path.abspath(path)
    actual_sha256 = sha256_file(absolute_path)
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256):
        raise ValueError(
            "generation environment lock SHA-256 differs: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    with open(absolute_path) as handle:
        lock = yaml.safe_load(handle) or {}
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError(f"generation environment lock must use schema {LOCK_SCHEMA}")
    lock_id = lock.get("lock_id")
    if not isinstance(lock_id, str) or not lock_id:
        raise ValueError("generation environment lock requires a non-empty lock_id")
    expected_runtime = lock.get("runtime")
    expected_packages = lock.get("packages")
    expected_platform = lock.get("platform")
    expected_hardware = lock.get("reference_hardware")
    expected_determinism = lock.get("determinism")
    if not isinstance(expected_runtime, Mapping) or set(expected_runtime) != REQUIRED_RUNTIME_FIELDS:
        raise ValueError(
            "generation environment lock runtime must exactly define "
            f"{sorted(REQUIRED_RUNTIME_FIELDS)}"
        )
    if not isinstance(expected_packages, Mapping) or not expected_packages:
        raise ValueError("generation environment lock requires pinned packages")
    if not isinstance(expected_platform, str) or not expected_platform:
        raise ValueError("generation environment lock requires a platform")
    if not isinstance(expected_hardware, Mapping) or set(
        expected_hardware
    ) != REQUIRED_HARDWARE_FIELDS:
        raise ValueError(
            "generation environment lock reference_hardware must exactly define "
            f"{sorted(REQUIRED_HARDWARE_FIELDS)}"
        )
    if not isinstance(expected_determinism, Mapping) or set(
        expected_determinism
    ) != REQUIRED_DETERMINISM_FIELDS:
        raise ValueError(
            "generation environment lock determinism must exactly define "
            f"{sorted(REQUIRED_DETERMINISM_FIELDS)}"
        )
    if any(not isinstance(value, bool) for value in expected_determinism.values()):
        raise ValueError("generation environment lock determinism values must be boolean")

    observed = observed_environment(expected_packages)
    mismatches = []
    if observed["platform"] != expected_platform:
        mismatches.append(
            f"platform: expected {expected_platform}, got {observed['platform']}"
        )
    for field in sorted(REQUIRED_RUNTIME_FIELDS):
        expected = expected_runtime[field]
        actual = observed["runtime"][field]
        if str(actual) != str(expected):
            mismatches.append(f"runtime.{field}: expected {expected}, got {actual}")
    for package_name, expected in sorted(expected_packages.items()):
        actual = observed["packages"][package_name]
        if str(actual) != str(expected):
            mismatches.append(
                f"packages.{package_name}: expected {expected}, got {actual}"
            )
    for field in sorted(REQUIRED_HARDWARE_FIELDS):
        expected = expected_hardware[field]
        actual = observed["hardware"][field]
        if str(actual) != str(expected):
            mismatches.append(
                f"reference_hardware.{field}: expected {expected}, got {actual}"
            )
    for field in sorted(REQUIRED_DETERMINISM_FIELDS):
        expected = expected_determinism[field]
        actual = observed["determinism"][field]
        if actual is not expected:
            mismatches.append(
                f"determinism.{field}: expected {expected}, got {actual}"
            )
    if mismatches:
        raise ValueError("generation environment mismatch: " + "; ".join(mismatches))

    return {
        "schema": LOCK_SCHEMA,
        "lock_id": lock_id,
        "path": absolute_path,
        "sha256": actual_sha256,
        "observed": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--expected_sha256")
    args = parser.parse_args()
    record = validate_environment_lock(
        args.lock, expected_sha256=args.expected_sha256
    )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
