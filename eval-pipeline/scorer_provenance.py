"""Deterministic provenance contracts for metric execution.

The score values alone do not identify the metric that produced them.  This
module binds each strict scoring run to its implementation, installed packages,
model assets, and preprocessing so a resumed run cannot silently mix results
from different scorer definitions.
"""
from __future__ import annotations

from enum import Enum
import hashlib
import importlib.util
from importlib import metadata
import inspect
import json
import os
import platform
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


SCORER_PROVENANCE_SCHEMA = "repldm_scorer_provenance_v1"
CHECKPOINT_MANIFEST_SCHEMA = "repldm_scorer_assets_v1"
HPSV2_PRIVATE_SOURCE_CLOSURE = "hpsv2_private_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_TREE_SCHEMA = "python_source_tree_sha256_v1"
_OUTPUT_DIRECTIONS = {"higher", "lower", "witness"}
_TRANSFORM_ATTRIBUTES = (
    "size",
    "interpolation",
    "max_size",
    "antialias",
    "mean",
    "std",
    "inplace",
    "crop_pct",
)
_STRUCTURAL_ACTION_SCHEMA = "scheduler_native_structural_controls_actions_v1"
_STRUCTURAL_REGISTRATION_SCHEMA = "scheduler_native_structural_controls_v1"
_STRUCTURAL_SPLIT_ROLE = "development"
_STRUCTURAL_FORMAL_RUN_PATH = "outputs/structural_controls/development_v1"
_STRUCTURAL_SCORE_CONFIG_PATH = "configs/eval_common.yaml"
_STRUCTURAL_SCORE_CONFIG_SHA256 = (
    "4901d660383f30a4e65aaceb9180eecf72aea951c3ea0abb8dd845cb65fbe932"
)
_STRUCTURAL_ACTIONS_CONFIG_PATH = (
    "configs/scheduler_native_structural_controls_development_authorized_v1.yaml"
)
_STRUCTURAL_ACTIONS_CONFIG_SHA256 = (
    "c18524d99933d0444edaa0469f8937bb57bdd3834cab798f0685f0f6741898ea"
)


def _read_structural_gate_bytes(path: Path, *, label: str) -> bytes:
    descriptor = -1
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} is not regular")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _score_cli_targets_run(score_path: Path, canonical_run_root: Path) -> bool:
    if Path(sys.argv[0]).resolve() != score_path:
        return False
    candidates: list[str] = []
    for index, argument in enumerate(sys.argv):
        if argument == "--run_dir" and index + 1 < len(sys.argv):
            candidates.append(sys.argv[index + 1])
        elif argument.startswith("--run_dir="):
            candidates.append(argument.split("=", 1)[1])
    return any(
        candidate and Path(candidate).resolve() == canonical_run_root
        for candidate in candidates
    )


def _is_frozen_auxiliary_config(
    registration: Mapping[str, Any],
    *,
    pipeline_root: Path,
    run_root: Path,
    relative_path: str,
    expected_sha256: str,
    label: str,
) -> bool:
    """Identify one frozen first-pass config without exempting run config."""
    auxiliary_path = pipeline_root / relative_path
    auxiliary_bytes = _read_structural_gate_bytes(
        auxiliary_path, label=label
    )
    if hashlib.sha256(auxiliary_bytes).hexdigest() != expected_sha256:
        raise ValueError(f"{label} hash differs")
    try:
        import yaml

        auxiliary_config = yaml.safe_load(auxiliary_bytes.decode("utf-8")) or {}
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(auxiliary_config, Mapping):
        raise ValueError(f"{label} is not a mapping")

    run_config_bytes = _read_structural_gate_bytes(
        run_root / "config.json", label="formal structural scoring config"
    )
    try:
        run_config = json.loads(run_config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("formal structural scoring config is unreadable") from exc
    if not isinstance(run_config, Mapping):
        raise ValueError("formal structural scoring config is not a mapping")
    current = dict(registration)
    return current == dict(auxiliary_config) and current != dict(run_config)


def _structural_control_scoring_gate(
    registration: Mapping[str, Any],
) -> None:
    """Gate only the canonical formal score/audit processes for this protocol."""
    pipeline_root = Path(__file__).resolve().parent
    repository_root = pipeline_root.parent
    process_path = Path(sys.argv[0]).resolve()
    score_path = (pipeline_root / "score.py").resolve()
    audit_paths = {
        (pipeline_root / "audit_structural_control_run.py").resolve(),
        (pipeline_root / "evaluate_structural_control_run.py").resolve(),
    }
    canonical_run_root = (
        repository_root / _STRUCTURAL_FORMAL_RUN_PATH
    ).resolve()
    out_dir = registration.get("out_dir")
    out_dir_is_canonical = False
    if isinstance(out_dir, str) and out_dir:
        raw_out_dir = Path(out_dir)
        candidate = (
            raw_out_dir
            if raw_out_dir.is_absolute()
            else repository_root / raw_out_dir
        ).resolve()
        out_dir_is_canonical = candidate == canonical_run_root
    cli_run_is_canonical = _score_cli_targets_run(score_path, canonical_run_root)
    run_contract = registration.get("run_contract")
    intrinsic_structural_signals = (
        registration.get("structural_control_registered") is True,
        registration.get("action_schema") == _STRUCTURAL_ACTION_SCHEMA,
        registration.get("schema") == _STRUCTURAL_ACTION_SCHEMA,
        registration.get("structural_control_registration_schema")
        == _STRUCTURAL_REGISTRATION_SCHEMA,
        registration.get("registration_schema")
        == _STRUCTURAL_REGISTRATION_SCHEMA,
        isinstance(run_contract, Mapping)
        and run_contract.get("action_schema") == _STRUCTURAL_ACTION_SCHEMA,
        out_dir_is_canonical,
    )
    if not any(intrinsic_structural_signals) and not cli_run_is_canonical:
        return
    if process_path != score_path and process_path not in audit_paths:
        raise ValueError("formal structural scorer contract rejects this process context")
    if (
        process_path == score_path
        and cli_run_is_canonical
        and _is_frozen_auxiliary_config(
            registration,
            pipeline_root=pipeline_root,
            run_root=canonical_run_root,
            relative_path=_STRUCTURAL_SCORE_CONFIG_PATH,
            expected_sha256=_STRUCTURAL_SCORE_CONFIG_SHA256,
            label="formal structural shared score config",
        )
    ):
        return
    if process_path in audit_paths and _is_frozen_auxiliary_config(
        registration,
        pipeline_root=pipeline_root,
        run_root=canonical_run_root,
        relative_path=_STRUCTURAL_ACTIONS_CONFIG_PATH,
        expected_sha256=_STRUCTURAL_ACTIONS_CONFIG_SHA256,
        label="formal structural executable actions",
    ):
        return

    expected_signature = {
        "structural_control_registered": True,
        "action_schema": _STRUCTURAL_ACTION_SCHEMA,
        "structural_control_registration_schema": _STRUCTURAL_REGISTRATION_SCHEMA,
        "split_role": _STRUCTURAL_SPLIT_ROLE,
        "scorer_provenance_binding_required": True,
    }
    if any(registration.get(key) != value for key, value in expected_signature.items()):
        raise ValueError("formal structural scoring signature is incomplete or inconsistent")
    if not isinstance(run_contract, Mapping) or run_contract.get(
        "action_schema"
    ) != _STRUCTURAL_ACTION_SCHEMA:
        raise ValueError("formal structural run contract signature is inconsistent")

    if not isinstance(out_dir, str) or not out_dir:
        raise ValueError("formal structural scoring run config lacks out_dir")
    raw_out_dir = Path(out_dir)
    if not raw_out_dir.is_absolute() and (
        raw_out_dir == Path(".") or ".." in raw_out_dir.parts
    ):
        raise ValueError("formal structural scoring out_dir is not canonical")
    lexical_run_root = (
        raw_out_dir if raw_out_dir.is_absolute() else repository_root / raw_out_dir
    ).absolute()
    run_root = lexical_run_root.resolve()
    try:
        run_root.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError("formal structural scoring out_dir leaves the repository") from exc
    if lexical_run_root != run_root or run_root != canonical_run_root:
        raise ValueError(
            "formal structural scoring out_dir is not the canonical non-symlink run"
        )
    config_path = run_root / "config.json"
    try:
        disk_config = json.loads(
            _read_structural_gate_bytes(
                config_path, label="formal structural scoring config"
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("formal structural scoring config is unreadable") from exc
    if not isinstance(disk_config, Mapping):
        raise ValueError("formal structural scoring config is not a mapping")
    if disk_config != dict(registration):
        raise ValueError("formal structural scoring config changed after loading")

    # Lazy import avoids the audit -> scorer_provenance import cycle.
    import audit_structural_control_run as structural_audit

    amendment_path = (
        repository_root
        / structural_audit.STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH
    )
    seal_path = run_root / structural_audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
    if process_path == score_path:
        structural_audit.require_scoring_child_authorization(
            run_root,
            analysis_amendment_path=amendment_path,
            pre_score_seal_path=seal_path,
        )
        return
    structural_audit.require_scoring_attempt_marker(
        run_root,
        analysis_amendment_path=amendment_path,
        pre_score_seal_path=seal_path,
    )
    structural_audit.require_scoring_success_receipt(
        run_root,
        analysis_amendment_path=amendment_path,
        pre_score_seal_path=seal_path,
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_name(path: Path, root: str | os.PathLike[str] | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def _reject_symlink_components(path: Path, *, label: str) -> None:
    """Reject symlinks in an executable-source path before resolution."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            status = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(status.st_mode):
            raise RuntimeError(f"{label} cannot contain a symlink: {current}")


def source_file_record(
    path: str | os.PathLike[str],
    *,
    label: str,
    root: str | os.PathLike[str] | None = None,
    module: str | None = None,
) -> dict:
    source = Path(path).resolve()
    if not source.is_file():
        raise RuntimeError(f"provenance source file is missing: {source}")
    record = {
        "label": str(label),
        "path": _relative_name(source, root),
        "sha256": sha256_file(source),
    }
    if module is not None:
        record["module"] = str(module)
    return record


def python_source_tree_record(
    root: str | os.PathLike[str],
    *,
    label: str,
    module: str | None = None,
) -> dict:
    """Return one compact digest for every Python source file below ``root``."""
    raw_root = Path(root)
    try:
        root_status = raw_root.lstat()
    except OSError as exc:
        raise RuntimeError(f"Python source tree is missing: {raw_root}") from exc
    if stat.S_ISLNK(root_status.st_mode):
        raise RuntimeError(f"Python source tree root cannot be a symlink: {raw_root}")
    root_path = raw_root.resolve()
    if not root_path.is_dir():
        raise RuntimeError(f"Python source tree is missing: {root_path}")
    entries = []
    for candidate in sorted(root_path.rglob("*.py")):
        _reject_symlink_components(candidate, label="Python source tree file")
        try:
            candidate_status = candidate.lstat()
        except OSError as exc:
            raise RuntimeError(f"Python source tree contains an unreadable file: {candidate}") from exc
        if stat.S_ISLNK(candidate_status.st_mode):
            raise RuntimeError(f"Python source tree contains a symlink: {candidate}")
        if not candidate.is_file():
            continue
        source = candidate.resolve()
        try:
            relative_path = source.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"Python source tree contains an external link: {candidate}"
            ) from exc
        entries.append(
            {
                "path": relative_path,
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    if not entries:
        raise RuntimeError(f"Python source tree has no .py files: {root_path}")
    record = {
        "label": str(label),
        "path": str(module or root_path.name),
        "tree_schema": "python_source_tree_sha256_v1",
        "file_count": len(entries),
        "sha256": json_sha256(entries),
    }
    if module is not None:
        record["module"] = str(module)
    return record


def package_python_source_tree_record(module: str, *, label: str) -> dict:
    """Resolve an installed package and bind its complete Python source tree."""
    spec = importlib.util.find_spec(module)
    locations = tuple(spec.submodule_search_locations or ()) if spec else ()
    if len(locations) != 1:
        raise RuntimeError(
            f"cannot resolve one Python package source root for {module!r}"
        )
    return python_source_tree_record(locations[0], label=label, module=module)


def plugin_source_record(
    scorer: object,
    *,
    root: str | os.PathLike[str],
    require_containment: bool = True,
) -> dict:
    cls = scorer.__class__
    module_name = str(getattr(cls, "__module__", ""))
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    if path is None:
        try:
            path = inspect.getsourcefile(cls)
        except (OSError, TypeError):
            path = None
    if path is None:
        raise RuntimeError(
            f"cannot resolve scorer plugin source for {module_name}.{cls.__qualname__}"
        )
    # A scorer plugin is executable code and must be an ordinary repository
    # file.  Do not use the permissive source_file_record path here: resolving
    # a symlink before containment would allow code outside the provenance root
    # to be hashed under a misleading basename.
    return _repository_source_file_record(
        path,
        label="scorer_plugin",
        root=root,
        module=module_name,
        require_containment=require_containment,
    )


def _repository_source_file_record(
    path: str | os.PathLike[str],
    *,
    label: str,
    root: str | os.PathLike[str],
    module: str,
    require_containment: bool = True,
) -> dict:
    """Bind one ordinary source without following a symlink.

    Scorer plugins are executable code and must be bound to the declared source
    root. Keeping this check here (before ``source_file_record`` resolves the
    path) prevents an outside file from being represented by an in-root
    basename.
    """
    raw_root = Path(root)
    raw_path = Path(path)
    _reject_symlink_components(raw_root, label="provenance source root")
    _reject_symlink_components(raw_path, label="provenance source")
    root_path = raw_root.resolve(strict=True)
    try:
        raw_status = raw_path.lstat()
        source = raw_path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"provenance source is unavailable: {raw_path}"
        ) from exc
    if stat.S_ISLNK(raw_status.st_mode) or not stat.S_ISREG(raw_status.st_mode):
        raise RuntimeError(f"provenance source is not an ordinary file: {raw_path}")
    if not stat.S_ISREG(source.stat().st_mode):
        raise RuntimeError(f"provenance source is not a regular file: {source}")
    if require_containment:
        try:
            source.relative_to(root_path)
        except ValueError as exc:
            raise RuntimeError(
                f"provenance source is outside the source root: {raw_path}"
            ) from exc
    return source_file_record(
        source,
        label=label,
        root=root_path,
        module=module,
    )


def _scorer_mro_source_records(
    scorer: object, *, root: str | os.PathLike[str]
) -> list[dict]:
    records: dict[str, dict] = {}
    for cls in scorer.__class__.__mro__:
        if cls is object:
            continue
        module_name = str(getattr(cls, "__module__", ""))
        module = sys.modules.get(module_name)
        path = getattr(module, "__file__", None)
        if path is None:
            try:
                path = inspect.getsourcefile(cls)
            except (OSError, TypeError):
                path = None
        if path is None:
            raise RuntimeError(
                f"cannot resolve scorer MRO source for {module_name}.{cls.__qualname__}"
            )
        record = _repository_source_file_record(
            path,
            label="scorer_mro_source",
            root=root,
            module=module_name,
            require_containment=True,
        )
        records.setdefault(record["path"], record)
        if "." in module_name:
            package_name = module_name.split(".", 1)[0]
            package = sys.modules.get(package_name)
            package_path = getattr(package, "__file__", None)
            if package_path is None:
                raise RuntimeError(
                    f"cannot bind scorer MRO package initializer for {module_name!r}"
                )
            package_record = _repository_source_file_record(
                package_path,
                label="scorer_mro_package_initializer",
                root=root,
                module=package_name,
                require_containment=True,
            )
            records.setdefault(package_record["path"], package_record)
    return [records[path] for path in sorted(records)]


def scorer_framework_source_records(
    scorer: object,
    *,
    root: str | os.PathLike[str],
    source_closure: str | None = None,
) -> list[dict]:
    """Return an explicitly selected scorer framework source closure.

    The default is intentionally empty to preserve the frozen provenance bytes
    used by legacy ``score.py`` registrations. Formal HPSv2 scoring opts into a
    complete repository-local closure explicitly.
    """
    if source_closure is None:
        return []
    if source_closure != HPSV2_PRIVATE_SOURCE_CLOSURE:
        raise ValueError(f"unsupported scorer source closure: {source_closure!r}")

    records = [
        _repository_source_file_record(
            __file__,
            label="scorer_provenance_builder",
            root=root,
            module=__name__,
            require_containment=True,
        )
    ]
    module_name = str(getattr(scorer.__class__, "__module__", ""))
    if "." in module_name:
        package_name = module_name.split(".", 1)[0]
        package = sys.modules.get(package_name)
        package_path = getattr(package, "__file__", None)
        if package_path is None:
            raise RuntimeError(
                f"cannot bind scorer package initializer for {module_name!r}"
            )
        records.append(
            _repository_source_file_record(
                package_path,
                label="scorer_package_initializer",
                root=root,
                module=package_name,
                require_containment=True,
            )
        )
    records.extend(_scorer_mro_source_records(scorer, root=root))
    unique = {}
    for record in records:
        unique.setdefault(record["path"], record)
    return [unique[path] for path in sorted(unique)]


def package_version_manifest(distributions: Iterable[str]) -> dict[str, str]:
    versions = {}
    for distribution in sorted({str(value) for value in distributions}):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"cannot bind scorer provenance: package {distribution!r} has no version"
            ) from exc
    return versions


def resolved_hf_revision(path: str | os.PathLike[str]) -> str | None:
    # Do not resolve the final symlink: Hugging Face snapshot entries point at
    # blobs, and resolving first discards the snapshots/<commit> provenance.
    parts = Path(path).absolute().parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    revision = parts[index + 1]
    return revision if revision else None


def checkpoint_file_record(
    path: str | os.PathLike[str],
    *,
    role: str,
    filename: str | None = None,
    repository_id: str | None = None,
    revision: str | None = None,
    artifact_uri: str | None = None,
) -> dict:
    logical_path = Path(path).absolute()
    resolved_revision = revision or resolved_hf_revision(logical_path)
    asset = logical_path.resolve()
    if not asset.is_file():
        raise RuntimeError(f"scorer asset is missing: {asset}")
    return {
        "role": str(role),
        "filename": str(filename or asset.name),
        "repository_id": repository_id,
        "revision": resolved_revision,
        "artifact_uri": artifact_uri,
        "size_bytes": asset.stat().st_size,
        "sha256": sha256_file(asset),
    }


def hf_checkpoint_file_record(
    repository_id: str,
    filename: str,
    *,
    role: str,
) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repository_id,
        filename,
        local_files_only=True,
    )
    return checkpoint_file_record(
        path,
        role=role,
        filename=filename,
        repository_id=repository_id,
    )


def checkpoint_manifest(records: Iterable[Mapping[str, Any]]) -> dict:
    files = [dict(record) for record in records]
    files.sort(key=canonical_json)
    for record in files:
        if set(record) != {
            "role",
            "filename",
            "repository_id",
            "revision",
            "artifact_uri",
            "size_bytes",
            "sha256",
        }:
            raise RuntimeError("scorer asset record fields differ from the v1 schema")
        if not _SHA256_RE.fullmatch(str(record.get("sha256", ""))):
            raise RuntimeError("scorer asset record has an invalid SHA-256")
        for key in ("role", "filename"):
            value = record.get(key)
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"scorer asset record has an invalid {key}")
        if Path(record["filename"]).name != record["filename"]:
            raise RuntimeError("scorer asset filename must be a basename")
        for key in ("repository_id", "revision", "artifact_uri"):
            value = record.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                raise RuntimeError(f"scorer asset record has an invalid {key}")
        size = record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeError("scorer asset record has an invalid byte size")
    identities = [
        (record["role"], record.get("repository_id"), record["filename"])
        for record in files
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("scorer provenance contains duplicate asset identities")
    return {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "files": files,
        "files_sha256": json_sha256(files),
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return {
            "enum": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "name": value.name,
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if inspect.isfunction(value) or inspect.ismethod(value):
        return {
            "callable": f"{value.__module__}.{value.__qualname__}",
        }
    raise TypeError(f"preprocessing value is not deterministic JSON: {type(value)!r}")


def describe_preprocess(transform: object) -> dict:
    """Serialize torchvision/open_clip transforms without address-bearing reprs."""
    if inspect.isfunction(transform) or inspect.ismethod(transform):
        return _json_value(transform)
    result = {
        "type": f"{transform.__class__.__module__}.{transform.__class__.__qualname__}",
    }
    children = getattr(transform, "transforms", None)
    if isinstance(children, (list, tuple)):
        result["transforms"] = [describe_preprocess(child) for child in children]
    parameters = {}
    for name in _TRANSFORM_ATTRIBUTES:
        if hasattr(transform, name):
            parameters[name] = _json_value(getattr(transform, name))
    if parameters:
        result["parameters"] = parameters
    return result


def loaded_python_source_records(
    root: str | os.PathLike[str], *, label: str
) -> list[dict]:
    root_path = Path(root).resolve()
    records = {}
    for module_name, module in list(sys.modules.items()):
        path_value = getattr(module, "__file__", None)
        if not path_value:
            continue
        path = Path(path_value).resolve()
        try:
            path.relative_to(root_path)
        except ValueError:
            continue
        if path.suffix != ".py" or not path.is_file():
            continue
        records[str(path)] = source_file_record(
            path,
            label=label,
            root=root_path,
            module=module_name,
        )
    return sorted(records.values(), key=canonical_json)


def git_revision(root: str | os.PathLike[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision if re.fullmatch(r"[0-9a-fA-F]{40}", revision) else None


def runtime_manifest(device: str) -> dict:
    import torch

    torch_device = torch.device(device)
    cuda_device = None
    if torch_device.type == "cuda":
        index = torch_device.index
        if index is None:
            index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(index)
        cuda_device = {
            "index": int(index),
            "name": str(torch.cuda.get_device_name(index)),
            "capability": [int(capability[0]), int(capability[1])],
        }
    cudnn_version = torch.backends.cudnn.version()
    return {
        "device": str(torch_device),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": (
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        "cudnn_version": int(cudnn_version) if cudnn_version is not None else None,
        "cuda_device": cuda_device,
        "runner_package_versions": package_version_manifest(("Pillow", "PyYAML")),
    }


def _validate_model_records(models: object) -> list[dict]:
    if not isinstance(models, list):
        raise RuntimeError("scorer provenance models must be a list")
    result = []
    for model in models:
        if not _valid_model_record(model):
            raise RuntimeError("each scorer model must have a string identifier")
        result.append(dict(model))
    identifiers = [model["identifier"] for model in result]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("scorer provenance model identifiers are duplicated")
    canonical_json(result)
    return result


def _valid_model_record(value: object) -> bool:
    """Validate one persisted model identity without accepting lookalikes."""
    if not isinstance(value, Mapping):
        return False
    identifier = value.get("identifier")
    if not isinstance(identifier, str) or not identifier:
        return False
    if not set(value).issubset(
        {"identifier", "repository_id", "revision", "artifact_uri"}
    ):
        return False
    for key in ("repository_id", "revision", "artifact_uri"):
        field = value.get(key)
        if field is not None and (not isinstance(field, str) or not field):
            return False
    try:
        canonical_json(dict(value))
    except (TypeError, ValueError):
        return False
    return True


def _valid_source_record(
    value: object, *, allow_tree: bool, allow_legacy_hash_only: bool = False
) -> bool:
    """Validate a source digest record used by a persisted contract."""
    if not isinstance(value, Mapping):
        return False
    # Older score fixtures predate structured source metadata and persist only
    # the digest. Keep accepting that exact shape for implementation/plugin
    # fields, while rejecting partial or malformed structured records.
    if allow_legacy_hash_only and set(value) == {"sha256"}:
        digest = value.get("sha256")
        return isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None
    label = value.get("label")
    path = value.get("path")
    digest = value.get("sha256")
    if (
        not isinstance(label, str)
        or not label
        or not isinstance(path, str)
        or not path
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
    ):
        return False
    source_path = Path(path)
    if source_path.is_absolute() or ".." in source_path.parts:
        return False
    module = value.get("module")
    if module is not None and (not isinstance(module, str) or not module):
        return False
    if "tree_schema" in value or "file_count" in value:
        if not allow_tree:
            return False
        if value.get("tree_schema") != _SOURCE_TREE_SCHEMA:
            return False
        count = value.get("file_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            return False
    allowed = {"label", "path", "sha256", "module"}
    if value.get("tree_schema") is not None:
        allowed.update(("tree_schema", "file_count"))
    return set(value).issubset(allowed)


def _valid_runtime_manifest(runtime: object) -> bool:
    """Validate runtime identity fields in a persisted contract."""
    if not isinstance(runtime, Mapping):
        return False
    for key in ("device", "python_implementation", "python_version", "torch_version"):
        value = runtime.get(key)
        if not isinstance(value, str) or not value:
            return False
    cuda_runtime = runtime.get("cuda_runtime_version")
    if cuda_runtime is not None and (not isinstance(cuda_runtime, str) or not cuda_runtime):
        return False
    cudnn = runtime.get("cudnn_version")
    if cudnn is not None and (isinstance(cudnn, bool) or not isinstance(cudnn, int) or cudnn < 0):
        return False
    cuda_device = runtime.get("cuda_device")
    if cuda_device is not None:
        if not isinstance(cuda_device, Mapping) or set(cuda_device) != {"index", "name", "capability"}:
            return False
        index = cuda_device.get("index")
        name = cuda_device.get("name")
        capability = cuda_device.get("capability")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            return False
        if not isinstance(name, str) or not name:
            return False
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in capability)
        ):
            return False
    versions = runtime.get("runner_package_versions")
    if not isinstance(versions, Mapping) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in versions.items()
    ):
        return False
    try:
        canonical_json(dict(runtime))
    except (TypeError, ValueError):
        return False
    return True


def build_scorer_record(
    name: str,
    scorer: object,
    *,
    source_root: str | os.PathLike[str],
    source_closure: str | None = None,
) -> dict:
    distributions = getattr(scorer, "PROVENANCE_PACKAGES", None)
    metadata_builder = getattr(scorer, "provenance_metadata", None)
    if distributions is None or not callable(metadata_builder):
        raise RuntimeError(
            f"strict scorer {name!r} does not implement the provenance contract"
        )
    payload = metadata_builder()
    required = {
        "models",
        "checkpoint_files",
        "preprocessing",
        "parameters",
        "supporting_sources",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise RuntimeError(
            f"strict scorer {name!r} provenance fields differ from the v1 schema"
        )
    if not isinstance(payload["preprocessing"], Mapping):
        raise RuntimeError(f"strict scorer {name!r} preprocessing must be a mapping")
    if not isinstance(payload["parameters"], Mapping):
        raise RuntimeError(f"strict scorer {name!r} parameters must be a mapping")
    supporting_sources = payload["supporting_sources"]
    if not isinstance(supporting_sources, list):
        raise RuntimeError(f"strict scorer {name!r} supporting_sources must be a list")
    if any(
        not _valid_source_record(source, allow_tree=True)
        for source in supporting_sources
    ):
        raise RuntimeError(f"strict scorer {name!r} supporting_sources are invalid")
    try:
        output_keys = [
            {"name": key, "direction": direction}
            for key, direction in getattr(scorer, "OUTPUT_KEYS", ())
        ]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"strict scorer {name!r} output keys are invalid") from exc
    if any(
        not isinstance(item["name"], str)
        or not item["name"]
        or not isinstance(item["direction"], str)
        or item["direction"] not in _OUTPUT_DIRECTIONS
        for item in output_keys
    ) or len({item["name"] for item in output_keys}) != len(output_keys):
        raise RuntimeError(f"strict scorer {name!r} output keys are invalid")
    all_supporting_sources = [
        *[dict(value) for value in supporting_sources],
        *scorer_framework_source_records(
            scorer,
            root=source_root,
            source_closure=source_closure,
        ),
    ]
    # A closure can rediscover a file already declared by a scorer's metadata
    # (for example, an inherited implementation). Keep one canonical record per
    # path so persisted validation can unambiguously identify each source.
    unique_supporting_sources = {}
    for value in all_supporting_sources:
        unique_supporting_sources.setdefault(value["path"], value)
    record = {
        "name": str(name),
        "class": f"{scorer.__class__.__module__}.{scorer.__class__.__qualname__}",
        "plugin_source": plugin_source_record(
            scorer,
            root=source_root,
            require_containment=source_closure == HPSV2_PRIVATE_SOURCE_CLOSURE,
        ),
        "supporting_sources": sorted(
            unique_supporting_sources.values(), key=canonical_json
        ),
        "package_versions": package_version_manifest(distributions),
        "models": _validate_model_records(payload["models"]),
        "checkpoints": checkpoint_manifest(payload["checkpoint_files"]),
        "preprocessing": dict(payload["preprocessing"]),
        "parameters": dict(payload["parameters"]),
        "output_keys": output_keys,
    }
    canonical_json(record)
    return record


def build_scorer_provenance(
    active: Sequence[tuple[str, object]],
    *,
    params: Mapping[str, Any],
    device: str,
    runner_path: str | os.PathLike[str],
    base_path: str | os.PathLike[str],
    source_root: str | os.PathLike[str],
    source_closure: str | None = None,
) -> tuple[dict, str]:
    scorers = [
        build_scorer_record(
            name,
            scorer,
            source_root=source_root,
            source_closure=source_closure,
        )
        for name, scorer in active
    ]
    contract = {
        "schema": SCORER_PROVENANCE_SCHEMA,
        "metrics": [str(name) for name, _ in active],
        "config_params": dict(params),
        "runtime": runtime_manifest(device),
        "shared_preprocessing": {
            "decoder": "PIL.Image.open",
            "input_color_mode": "RGB",
        },
        "implementation": {
            "score_runner": source_file_record(
                runner_path, label="score_runner", root=source_root
            ),
            "scorer_base": source_file_record(
                base_path, label="scorer_base", root=source_root
            ),
        },
        "scorers": scorers,
    }
    digest = json_sha256(contract)
    validate_scorer_provenance(contract, digest)
    return contract, digest


def validate_scorer_provenance(contract: object, observed_sha256: object) -> str:
    if not isinstance(contract, Mapping):
        raise ValueError("scorer provenance contract must be a mapping")
    if contract.get("schema") != SCORER_PROVENANCE_SCHEMA:
        raise ValueError("scorer provenance contract has the wrong schema")
    expected_keys = {
        "schema",
        "metrics",
        "config_params",
        "runtime",
        "shared_preprocessing",
        "implementation",
        "scorers",
    }
    if set(contract) != expected_keys:
        raise ValueError("scorer provenance contract fields differ from the v1 schema")
    implementation = contract.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "score_runner",
        "scorer_base",
    }:
        raise ValueError("scorer provenance implementation sources are invalid")
    for source in implementation.values():
        if not _valid_source_record(
            source, allow_tree=False, allow_legacy_hash_only=True
        ):
            raise ValueError("scorer provenance implementation source hash is invalid")
    runtime = contract.get("runtime")
    runtime_keys = {
        "device",
        "python_implementation",
        "python_version",
        "torch_version",
        "cuda_runtime_version",
        "cudnn_version",
        "cuda_device",
        "runner_package_versions",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_keys:
        raise ValueError("scorer provenance runtime fields are invalid")
    if not _valid_runtime_manifest(runtime):
        raise ValueError("scorer provenance runtime identity is invalid")
    scorers = contract.get("scorers")
    metrics = contract.get("metrics")
    if not isinstance(scorers, list) or not isinstance(metrics, list):
        raise ValueError("scorer provenance metrics/scorers must be lists")
    if (
        any(not isinstance(name, str) or not name for name in metrics)
        or len(metrics) != len(set(metrics))
        or not metrics
    ):
        raise ValueError("scorer provenance metric names are invalid")
    if metrics != [record.get("name") for record in scorers if isinstance(record, Mapping)]:
        raise ValueError("scorer provenance metric order differs from scorer records")
    for record in scorers:
        if not isinstance(record, Mapping):
            raise ValueError("scorer provenance record must be a mapping")
        if set(record) != {
            "name",
            "class",
            "plugin_source",
            "supporting_sources",
            "package_versions",
            "models",
            "checkpoints",
            "preprocessing",
            "parameters",
            "output_keys",
        }:
            raise ValueError("scorer provenance record fields differ from v1")
        if (
            not isinstance(record.get("name"), str)
            or not record["name"]
            or not isinstance(record.get("class"), str)
            or not record["class"]
        ):
            raise ValueError("scorer provenance scorer identity is invalid")
        source = record.get("plugin_source")
        if not _valid_source_record(
            source, allow_tree=False, allow_legacy_hash_only=True
        ):
            raise ValueError("scorer provenance plugin source hash is invalid")
        versions = record.get("package_versions")
        if not isinstance(versions, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in versions.items()
        ):
            raise ValueError("scorer provenance package versions are invalid")
        models = record.get("models")
        if not isinstance(models, list) or any(
            not _valid_model_record(model) for model in models
        ):
            raise ValueError("scorer provenance model identifiers are invalid")
        model_ids = [model["identifier"] for model in models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("scorer provenance model identifiers are duplicated")
        supporting_sources = record.get("supporting_sources")
        if not isinstance(supporting_sources, list) or any(
            not _valid_source_record(item, allow_tree=True)
            for item in supporting_sources
        ):
            raise ValueError("scorer provenance supporting source hashes are invalid")
        source_paths = [item["path"] for item in supporting_sources]
        if len(source_paths) != len(set(source_paths)):
            raise ValueError("scorer provenance supporting sources are duplicated")
        if supporting_sources != sorted(supporting_sources, key=canonical_json):
            raise ValueError("scorer provenance supporting sources are not canonical")
        checkpoints = record.get("checkpoints")
        if not isinstance(checkpoints, Mapping):
            raise ValueError("scorer provenance checkpoint manifest is missing")
        if checkpoints.get("schema") != CHECKPOINT_MANIFEST_SCHEMA:
            raise ValueError("scorer provenance checkpoint schema is invalid")
        files = checkpoints.get("files")
        if not isinstance(files, list) or checkpoints.get("files_sha256") != json_sha256(files):
            raise ValueError("scorer provenance checkpoint manifest hash is invalid")
        if files != sorted(files, key=canonical_json):
            raise ValueError("scorer provenance checkpoint files are not canonical")
        identities = []
        for asset in files:
            if not isinstance(asset, Mapping) or set(asset) != {
                "role",
                "filename",
                "repository_id",
                "revision",
                "artifact_uri",
                "size_bytes",
                "sha256",
            }:
                raise ValueError("scorer provenance asset fields are invalid")
            if (
                not isinstance(asset.get("role"), str)
                or not asset["role"]
                or not isinstance(asset.get("filename"), str)
                or not asset["filename"]
                or any(
                    asset.get(key) is not None
                    and (not isinstance(asset.get(key), str) or not asset.get(key))
                    for key in ("repository_id", "revision", "artifact_uri")
                )
                or isinstance(asset.get("size_bytes"), bool)
                or not isinstance(asset.get("size_bytes"), int)
                or asset["size_bytes"] < 0
                or not isinstance(asset.get("sha256"), str)
                or not _SHA256_RE.fullmatch(asset["sha256"])
            ):
                raise ValueError("scorer provenance asset SHA-256 is invalid")
            identity = (
                asset["role"],
                asset.get("repository_id"),
                asset["filename"],
            )
            identities.append(identity)
        if len(identities) != len(set(identities)):
            raise ValueError("scorer provenance checkpoint identities are duplicated")
        if not isinstance(record.get("preprocessing"), Mapping):
            raise ValueError("scorer provenance preprocessing is invalid")
        if not isinstance(record.get("parameters"), Mapping):
            raise ValueError("scorer provenance parameters are invalid")
        output_keys = record.get("output_keys")
        if not isinstance(output_keys, list) or any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "direction"}
            or not isinstance(item["name"], str)
            or not isinstance(item["direction"], str)
            or not item["name"]
            or not item["direction"]
            or item["direction"] not in _OUTPUT_DIRECTIONS
            for item in output_keys
        ):
            raise ValueError("scorer provenance output key contract is invalid")
        output_names = [item["name"] for item in output_keys]
        if len(output_names) != len(set(output_names)):
            raise ValueError("scorer provenance output keys are duplicated")
    recomputed = json_sha256(contract)
    if observed_sha256 != recomputed:
        raise ValueError("scorer provenance SHA-256 does not match its payload")
    return recomputed


def validate_hardened_score_rows(
    scores: Iterable[Mapping[str, Any]],
    *,
    required_schema: str = SCORER_PROVENANCE_SCHEMA,
    expected_sha256: str | None = None,
    expected_contract: Mapping[str, Any] | None = None,
) -> str:
    rows = list(scores)
    if not rows:
        raise ValueError("hardened score validation requires at least one row")
    reference = rows[0].get("scorer_provenance")
    reference_hash = rows[0].get("scorer_provenance_sha256")
    if required_schema != SCORER_PROVENANCE_SCHEMA:
        raise ValueError(f"unsupported required scorer provenance schema {required_schema!r}")
    validate_scorer_provenance(reference, reference_hash)
    if expected_sha256 is not None and str(reference_hash) != str(expected_sha256):
        raise ValueError("score rows do not match the registered scorer provenance hash")
    if expected_contract is not None and reference != dict(expected_contract):
        raise ValueError("score rows do not match the registered scorer provenance contract")
    for row in rows:
        if row.get("scorer_provenance") != reference:
            raise ValueError(f"{row.get('id')}: scorer provenance contract drifted")
        if row.get("scorer_provenance_sha256") != reference_hash:
            raise ValueError(f"{row.get('id')}: scorer provenance hash drifted")
    return str(reference_hash)


def registered_scorer_provenance_contract(
    registration: Mapping[str, Any] | None,
) -> tuple[dict | None, str | None]:
    """Resolve an explicitly registered scorer contract from a run/config map.

    The generation and audit protocols use a deliberately narrow set of field
    names.  Conflicting aliases fail closed instead of silently choosing one.
    """
    if not isinstance(registration, Mapping):
        return None, None
    _structural_control_scoring_gate(registration)
    mappings = [registration]
    for key in ("scoring", "scoring_contract", "executable_scoring"):
        value = registration.get(key)
        if isinstance(value, Mapping):
            mappings.append(value)
    contracts = []
    hashes = []
    for value in mappings:
        for key in (
            "registered_scorer_provenance",
            "expected_scorer_provenance",
            "scorer_provenance_contract",
            "registered_scoring_contract",
        ):
            contract = value.get(key)
            if isinstance(contract, Mapping):
                contracts.append(dict(contract))
        for key in (
            "registered_scorer_provenance_sha256",
            "expected_scorer_provenance_sha256",
            "required_scorer_provenance_sha256",
            "scorer_provenance_sha256",
            "registered_scorer_provenance_hash",
            "expected_scorer_provenance_hash",
            "scorer_provenance_hash",
            "scoring_contract_hash",
        ):
            candidate = value.get(key)
            if candidate is not None:
                hashes.append(str(candidate))
    if contracts:
        first = contracts[0]
        if any(contract != first for contract in contracts[1:]):
            raise ValueError("registered scorer provenance contracts disagree")
        computed = json_sha256(first)
        # Validate the payload before allowing it to become a run binding.
        validate_scorer_provenance(first, computed)
        hashes.append(computed)
    if hashes and any(value != hashes[0] for value in hashes[1:]):
        raise ValueError("registered scorer provenance hashes disagree")
    contract = contracts[0] if contracts else None
    return contract, (hashes[0] if hashes else None)


__all__ = [
    "CHECKPOINT_MANIFEST_SCHEMA",
    "HPSV2_PRIVATE_SOURCE_CLOSURE",
    "SCORER_PROVENANCE_SCHEMA",
    "build_scorer_provenance",
    "checkpoint_file_record",
    "describe_preprocess",
    "git_revision",
    "hf_checkpoint_file_record",
    "json_sha256",
    "loaded_python_source_records",
    "resolved_hf_revision",
    "runtime_manifest",
    "source_file_record",
    "validate_hardened_score_rows",
    "registered_scorer_provenance_contract",
    "validate_scorer_provenance",
]
