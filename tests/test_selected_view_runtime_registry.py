from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval-pipeline"))

import data_catalog.selected_runtime as runtime_module


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
