from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval-pipeline"))

import data_catalog.selected_runtime as runtime_module
import data_catalog.selected as selected_module
import data_catalog.selected_builder as selected_builder_module


def _backend_path(root: Path) -> Path:
    path = root / "eval-pipeline" / "data_catalog" / "selected_runtime_backend.py"
    path.parent.mkdir(parents=True)
    path.write_text("# fixture backend\n", encoding="utf-8")
    return path


def test_registry_fails_closed_when_backend_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(_name: str):
        raise ModuleNotFoundError("fixture backend is absent")

    monkeypatch.setattr(runtime_module.importlib, "import_module", missing)
    with pytest.raises(
        RuntimeError, match="no production selected-view runtime backend"
    ):
        runtime_module.build_runtime_v1({}, tmp_path, tmp_path)


def test_registry_calls_only_the_pinned_backend_module(tmp_path: Path, monkeypatch):
    backend_path = _backend_path(tmp_path)
    calls: list[str] = []
    sentinel = object()
    backend = SimpleNamespace(
        __file__=str(backend_path),
        REGISTRY_ID=runtime_module.RUNTIME_REGISTRY_ID,
        build_runtime_v1=lambda config, parent, repository: sentinel,
    )

    def load(name: str):
        calls.append(name)
        return backend

    monkeypatch.setattr(runtime_module.importlib, "import_module", load)
    result = runtime_module.build_runtime_v1({}, tmp_path, tmp_path)

    assert result is sentinel
    assert calls == ["data_catalog.selected_runtime_backend"]


def test_registry_forwards_parent_snapshot(tmp_path: Path, monkeypatch):
    backend_path = _backend_path(tmp_path)
    sentinel = object()
    snapshot = object()
    seen: dict[str, object] = {}

    def build(config, parent, repository, *, parent_snapshot=None):
        seen.update(
            config=config,
            parent=parent,
            repository=repository,
            parent_snapshot=parent_snapshot,
        )
        return sentinel

    backend = SimpleNamespace(
        __file__=str(backend_path),
        REGISTRY_ID=runtime_module.RUNTIME_REGISTRY_ID,
        build_runtime_v1=build,
    )
    monkeypatch.setattr(runtime_module.importlib, "import_module", lambda _name: backend)

    result = runtime_module.build_runtime_v1({}, tmp_path, tmp_path, snapshot)

    assert result is sentinel
    assert seen == {
        "config": {},
        "parent": tmp_path,
        "repository": tmp_path,
        "parent_snapshot": snapshot,
    }


def test_revalidator_keeps_one_parent_snapshot_for_runtime_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "selection-config.json").write_text("{}", encoding="utf-8")
    parent_manifest = tmp_path / "parent" / "manifest.json"
    parent_manifest.parent.mkdir()
    parent_manifest.write_text("{}", encoding="utf-8")
    snapshot = SimpleNamespace(closed=False)

    def close_snapshot() -> None:
        snapshot.closed = True

    snapshot.close = close_snapshot
    manifest = {
        "config": {"path": "selection-config.json"},
        "parent_catalog": {
            "path": str(parent_manifest),
            "manifest_sha256": "a" * 64,
        },
    }
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        selected_module,
        "validate_selected_view_release",
        lambda *args, **kwargs: manifest,
    )

    def capture(path: Path, *, manifest_sha256: str):
        calls["capture"] = (path, manifest_sha256)
        return snapshot

    monkeypatch.setattr(selected_module, "_capture_parent_artifact_snapshot", capture)

    runtime = SimpleNamespace(close=lambda: calls.setdefault("runtime_closed", True))

    def create(*args, **kwargs):
        calls["create_snapshot"] = kwargs["parent_snapshot"]
        return runtime

    def verify(*args, **kwargs):
        calls["verify_snapshot"] = kwargs["parent_snapshot"]
        return {"verified": True}

    monkeypatch.setattr(selected_builder_module, "create_selected_view_runtime", create)
    monkeypatch.setattr(selected_builder_module, "verify_selected_view_runtime", verify)

    result = runtime_module.revalidate_release_v1(
        release, repository_root=tmp_path
    )

    assert result == {"verified": True}
    assert calls["capture"] == (parent_manifest.parent, "a" * 64)
    assert calls["create_snapshot"] is snapshot
    assert calls["verify_snapshot"] is snapshot
    assert calls["runtime_closed"] is True
    assert snapshot.closed is True


def test_registry_rejects_backend_from_another_path(tmp_path: Path, monkeypatch):
    expected = _backend_path(tmp_path)
    foreign = tmp_path / "foreign_backend.py"
    foreign.write_text("# foreign\n", encoding="utf-8")
    backend = SimpleNamespace(
        __file__=str(foreign),
        REGISTRY_ID=runtime_module.RUNTIME_REGISTRY_ID,
        build_runtime_v1=lambda *_args: object(),
    )
    monkeypatch.setattr(
        runtime_module.importlib, "import_module", lambda _name: backend
    )

    with pytest.raises(RuntimeError, match="not the pinned repository file"):
        runtime_module.build_runtime_v1({}, tmp_path, tmp_path)
    assert expected.exists()


def test_registry_rejects_wrong_identity(tmp_path: Path, monkeypatch):
    backend_path = _backend_path(tmp_path)
    backend = SimpleNamespace(
        __file__=str(backend_path),
        REGISTRY_ID="unapproved-runtime",
        build_runtime_v1=lambda *_args: object(),
    )
    monkeypatch.setattr(
        runtime_module.importlib, "import_module", lambda _name: backend
    )

    with pytest.raises(RuntimeError, match="wrong registry identity"):
        runtime_module.build_runtime_v1({}, tmp_path, tmp_path)


def test_isolated_revalidator_can_import_repository_local_package():
    script = ROOT / "eval-pipeline" / "revalidate_selected_view_runtime.py"
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--help"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    assert "--release-dir" in result.stdout
