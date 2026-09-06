"""Fixed registration point for the formal selected-view gate runtime.

The selected-view builder intentionally accepts a runtime factory so that CPU
fixtures can exercise the selection protocol.  Training authorization cannot
trust that caller-supplied callable.  This module is the only runtime entry
point used by the authorization revalidator; a production backend must be
implemented at the fixed, source-controlled location below.

Keeping the adapter separate from the gate protocol makes the eventual local
model/index implementation replaceable without weakening the authorization
boundary.  Until that backend exists, every formal authorization fails closed.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .selected import ParentArtifactSnapshot

RUNTIME_REGISTRY_SCHEMA = "repldm.selected_view_runtime_registry.v1"
RUNTIME_REGISTRY_ID = "selected_view_runtime_v1"
_BACKEND_MODULE = "data_catalog.selected_runtime_backend"
_BACKEND_FILE = "selected_runtime_backend.py"


def _load_backend(repository_root: Path) -> Any:
    """Load only the pre-registered backend from this repository."""
    try:
        backend = importlib.import_module(_BACKEND_MODULE)
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "no production selected-view runtime backend is registered; "
            "add eval-pipeline/data_catalog/selected_runtime_backend.py"
        ) from exc
    module_file = getattr(backend, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError("selected-view runtime backend has no source file")
    expected = (Path(repository_root) / "eval-pipeline" / "data_catalog" / _BACKEND_FILE).absolute()
    observed = Path(module_file).absolute()
    if observed != expected or observed.is_symlink() or not observed.is_file():
        raise RuntimeError("selected-view runtime backend is not the pinned repository file")
    if getattr(backend, "REGISTRY_ID", None) != RUNTIME_REGISTRY_ID:
        raise RuntimeError("selected-view runtime backend has the wrong registry identity")
    return backend


def build_runtime_v1(
    config: Mapping[str, Any],
    parent_dir: Path,
    repository_root: Path,
    parent_snapshot: "ParentArtifactSnapshot | None" = None,
) -> Any:
    """Build the one runtime permitted for formal selected-view validation."""
    backend = _load_backend(Path(repository_root))
    factory = getattr(backend, "build_runtime_v1", None)
    if not callable(factory):
        raise RuntimeError("selected-view runtime backend lacks build_runtime_v1")
    if parent_snapshot is None:
        # Keep the registry compatible with development fixtures and older
        # backends that predate the pinned-parent contract.
        return factory(config, Path(parent_dir), Path(repository_root))
    return factory(
        config,
        Path(parent_dir),
        Path(repository_root),
        parent_snapshot=parent_snapshot,
    )


def revalidate_release_v1(
    release_dir: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Re-run all learned/indexed gates through the fixed runtime entrypoint."""
    # Imports stay local so importing this registry remains cheap and does not
    # initialize model libraries before the caller has passed its own gates.
    from .selected import (
        _SHA256_RE,
        _capture_parent_artifact_snapshot,
        validate_selected_view_release,
    )
    from .selected_builder import (
        create_selected_view_runtime,
        verify_selected_view_runtime,
    )

    root = Path(repository_root).absolute()
    release = Path(release_dir).absolute()
    manifest = validate_selected_view_release(
        release,
        repository_root=root,
        require_formal=True,
        require_training_ready=True,
        require_gate_report=True,
    )
    config_descriptor = manifest.get("config")
    if not isinstance(config_descriptor, Mapping):
        raise RuntimeError("selected-view manifest has no config descriptor")
    config_path = release / str(config_descriptor.get("path", ""))
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("selected-view config cannot be loaded for revalidation") from exc
    if not isinstance(config, Mapping):
        raise RuntimeError("selected-view config is not an object")
    parent_descriptor = manifest.get("parent_catalog")
    if not isinstance(parent_descriptor, Mapping):
        raise RuntimeError("selected-view manifest has no parent catalog binding")
    parent_path = parent_descriptor.get("path")
    if not isinstance(parent_path, str) or not Path(parent_path).is_absolute():
        raise RuntimeError("selected-view parent catalog path is invalid")
    parent_hash = parent_descriptor.get("manifest_sha256")
    if not isinstance(parent_hash, str) or _SHA256_RE.fullmatch(parent_hash) is None:
        raise RuntimeError("selected-view parent catalog hash is invalid")

    # The first validation owns and closes its internal snapshot.  Capture a
    # fresh one before loading the production runtime, then keep that same
    # descriptor set through runtime construction and the final revalidation.
    parent_snapshot = _capture_parent_artifact_snapshot(
        Path(parent_path).parent,
        manifest_sha256=parent_hash,
    )
    try:
        runtime = create_selected_view_runtime(
            build_runtime_v1,
            config=config,
            parent_dir=Path(parent_path).parent,
            repository_root=root,
            parent_snapshot=parent_snapshot,
        )
        try:
            return verify_selected_view_runtime(
                release,
                runtime=runtime,
                repository_root=root,
                require_formal=True,
                parent_snapshot=parent_snapshot,
            )
        finally:
            close = getattr(runtime, "close", None)
            if callable(close):
                close()
    finally:
        parent_snapshot.close()


__all__ = [
    "RUNTIME_REGISTRY_ID",
    "RUNTIME_REGISTRY_SCHEMA",
    "build_runtime_v1",
    "revalidate_release_v1",
]
