"""Validate the physical GPU contract used by Adaptive Oracle runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import re
import subprocess
from typing import Mapping, Optional

import torch
import yaml


LOCK_SCHEMA = "repldm_generation_environment_lock_v1"
PHYSICAL_DEVICE_LOCK_SCHEMA = "repldm_generation_environment_lock_v2"
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
CUDA_DEVICE = re.compile(r"cuda:([0-9]+)")
GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
PCI_BUS_ID = re.compile(r"[0-9A-F]{8}:[0-9A-F]{2}:[0-9A-F]{2}\.[0-7]")
NVIDIA_SMI = "/usr/bin/nvidia-smi"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cuda_device_index(cuda_device: str) -> int:
    if not isinstance(cuda_device, str):
        raise ValueError("cuda_device must be an explicit cuda:N string")
    match = CUDA_DEVICE.fullmatch(cuda_device)
    if match is None:
        raise ValueError("cuda_device must be an explicit cuda:N string")
    return int(match.group(1))


def _canonical_gpu_uuid(value, *, source: str) -> str:
    raw = str(value).strip().lower()
    if not raw.startswith("gpu-"):
        raw = "gpu-" + raw
    canonical = "GPU-" + raw[4:]
    if GPU_UUID.fullmatch(canonical) is None:
        raise ValueError(f"{source} returned an invalid GPU UUID")
    return canonical


def _nvidia_smi_inventory() -> list[dict]:
    try:
        output = subprocess.check_output(
            [
                NVIDIA_SMI,
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("nvidia-smi physical GPU inventory is unavailable") from exc

    rows = []
    try:
        parsed_rows = list(csv.reader(output.splitlines()))
        if not parsed_rows:
            raise ValueError("nvidia-smi returned an empty GPU inventory")
        for parsed in parsed_rows:
            values = [value.strip() for value in parsed]
            if len(values) != 5:
                raise ValueError("nvidia-smi returned a malformed GPU identity row")
            index_text, uuid_text, pci_text, name, driver = values
            index = int(index_text)
            if index < 0 or not name or not driver:
                raise ValueError("nvidia-smi returned an incomplete GPU identity row")
            uuid_value = _canonical_gpu_uuid(uuid_text, source="nvidia-smi")
            pci_bus_id = pci_text.upper()
            if PCI_BUS_ID.fullmatch(pci_bus_id) is None:
                raise ValueError("nvidia-smi returned an invalid PCI bus id")
            rows.append(
                {
                    "index": index,
                    "uuid": uuid_value,
                    "pci_bus_id": pci_bus_id,
                    "name": name,
                    "driver": driver,
                }
            )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("nvidia-smi"):
            raise
        raise ValueError("nvidia-smi returned a malformed GPU inventory") from exc

    for key in ("index", "uuid", "pci_bus_id"):
        values = [row[key] for row in rows]
        if len(set(values)) != len(values):
            raise ValueError(f"nvidia-smi returned duplicate GPU {key} values")
    return rows


def observed_environment(
    package_names,
    *,
    cuda_device: str = "cuda:0",
    require_unmasked_cuda: bool = False,
) -> dict:
    device_index = _cuda_device_index(cuda_device)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_device_order = os.environ.get("CUDA_DEVICE_ORDER")
    if require_unmasked_cuda and (
        cuda_visible_devices is not None or cuda_device_order is not None
    ):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER must be unset when "
            "cuda:N denotes a physical GPU"
        )
    packages = {}
    for name in package_names:
        try:
            packages[str(name)] = version(str(name))
        except PackageNotFoundError:
            packages[str(name)] = None
    if not torch.cuda.is_available() or torch.cuda.device_count() <= device_index:
        raise ValueError("requested PyTorch CUDA device is unavailable")
    properties = torch.cuda.get_device_properties(device_index)
    gpu = str(properties.name)
    compute_capability = f"{int(properties.major)}.{int(properties.minor)}"
    torch_uuid = _canonical_gpu_uuid(
        getattr(properties, "uuid", None), source="PyTorch"
    )

    inventory = _nvidia_smi_inventory()
    matches = [row for row in inventory if row["uuid"] == torch_uuid]
    if len(matches) != 1:
        raise ValueError(
            "PyTorch GPU UUID must match exactly one nvidia-smi inventory row"
        )
    smi = matches[0]
    if smi["name"] != gpu:
        raise ValueError(
            "PyTorch and nvidia-smi disagree on the requested CUDA device name"
        )
    physical_index_match = smi["index"] == device_index
    if require_unmasked_cuda and not physical_index_match:
        raise ValueError(
            "PyTorch CUDA index does not match the unmasked nvidia-smi physical index"
        )
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
            "driver": smi["driver"],
            "compute_capability": compute_capability,
        },
        "cuda_device": {
            "requested": cuda_device,
            "logical_index": device_index,
            "cuda_visible_devices": cuda_visible_devices,
            "cuda_device_order": cuda_device_order,
            "torch": {
                "index": device_index,
                "name": gpu,
                "uuid": torch_uuid,
            },
            "nvidia_smi": {
                "index": smi["index"],
                "name": smi["name"],
                "uuid": smi["uuid"],
                "pci_bus_id": smi["pci_bus_id"],
            },
            "binding": {
                "uuid_match": smi["uuid"] == torch_uuid,
                "name_match": smi["name"] == gpu,
                "unmasked_physical_index_match": physical_index_match,
            },
        },
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def _validate_environment_lock_mapping(
    lock: Mapping,
    *,
    actual_sha256: str,
    source_path: str,
    expected_sha256: Optional[str] = None,
    cuda_device: str = "cuda:0",
    require_unmasked_cuda: bool = False,
) -> dict:
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256):
        raise ValueError(
            "generation environment lock SHA-256 differs: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    lock_schema = lock.get("schema")
    if lock_schema not in {LOCK_SCHEMA, PHYSICAL_DEVICE_LOCK_SCHEMA}:
        raise ValueError(
            "generation environment lock must use schema "
            f"{LOCK_SCHEMA} or {PHYSICAL_DEVICE_LOCK_SCHEMA}"
        )
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

    observed = observed_environment(
        expected_packages,
        cuda_device=cuda_device,
        require_unmasked_cuda=require_unmasked_cuda,
    )
    if lock_schema == LOCK_SCHEMA:
        observed = dict(observed)
        observed.pop("cuda_device")
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
        "schema": lock_schema,
        "lock_id": lock_id,
        "path": source_path,
        "sha256": actual_sha256,
        "observed": observed,
    }


def validate_environment_lock_bytes(
    raw: bytes,
    *,
    source_path: str,
    expected_sha256: Optional[str] = None,
    cuda_device: str = "cuda:0",
    require_unmasked_cuda: bool = False,
) -> dict:
    """Validate an immutable environment-lock snapshot supplied by a trusted caller."""

    if not isinstance(raw, bytes):
        raise TypeError("generation environment lock snapshot must be bytes")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("generation environment lock source path must be non-empty")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        lock = yaml.safe_load(raw.decode("utf-8", errors="strict")) or {}
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("generation environment lock is invalid UTF-8 YAML") from exc
    if not isinstance(lock, Mapping):
        raise ValueError("generation environment lock must be a mapping")
    return _validate_environment_lock_mapping(
        lock,
        actual_sha256=actual_sha256,
        source_path=source_path,
        expected_sha256=expected_sha256,
        cuda_device=cuda_device,
        require_unmasked_cuda=require_unmasked_cuda,
    )


def validate_environment_lock(
    path: str,
    *,
    expected_sha256: Optional[str] = None,
    cuda_device: str = "cuda:0",
    require_unmasked_cuda: bool = False,
) -> dict:
    """Fail closed when the current interpreter differs from a pinned lock file."""

    absolute_path = os.path.abspath(path)
    with open(absolute_path, "rb") as handle:
        raw = handle.read()
    return validate_environment_lock_bytes(
        raw,
        source_path=absolute_path,
        expected_sha256=expected_sha256,
        cuda_device=cuda_device,
        require_unmasked_cuda=require_unmasked_cuda,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--expected_sha256")
    parser.add_argument("--cuda_device", default="cuda:0")
    parser.add_argument("--require_unmasked_cuda", action="store_true")
    args = parser.parse_args()
    record = validate_environment_lock(
        args.lock,
        expected_sha256=args.expected_sha256,
        cuda_device=args.cuda_device,
        require_unmasked_cuda=args.require_unmasked_cuda,
    )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
