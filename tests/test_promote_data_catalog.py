from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval-pipeline"))

import promote_data_catalog as promote  # noqa: E402


def _promotion_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repository = tmp_path / "repository"
    catalogs = repository / "DATA" / "catalogs"
    catalogs.mkdir(parents=True)
    source = catalogs / "catalog-source"
    source.mkdir()
    payload = source / "artifact.jsonl"
    payload.write_bytes(b"frozen artifact\n")
    artifact = {
        "path": payload.name,
        "schema": "fixture.schema.v1",
        "rows": 1,
        "bytes": payload.stat().st_size,
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema": promote.CATALOG_SCHEMA,
                "release_id": source.name,
                "artifacts": [artifact],
            }
        ),
        encoding="utf-8",
    )
    config_path = repository / "config.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")

    config = {"physical_sources": [], "payload_integrity_policy": {}}
    git = {
        "commit": "a" * 40,
        "branch": "fixture",
        "dirty": False,
        "pushed": True,
        "upstream_commit": "a" * 40,
        "upstream_ref": "refs/remotes/origin/fixture",
    }
    core = {"complete": False, "candidate_catalog_complete": True}

    monkeypatch.setattr(promote, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(promote, "load_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(promote, "enforce_git_gate", lambda *_args, **_kwargs: git)
    monkeypatch.setattr(promote, "_artifact_contract", lambda _config: [artifact])
    monkeypatch.setattr(promote, "_manifest_core", lambda **_kwargs: core)
    monkeypatch.setattr(promote, "_release_id", lambda _core: "catalog-promoted")
    return repository, catalogs, source, config_path, artifact


def test_copy_is_an_independent_inode(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "staging" / "copy.jsonl"
    source.write_bytes(b"before")

    promote._link_or_copy(source, destination)

    assert source.stat().st_ino != destination.stat().st_ino
    source.write_bytes(b"after")
    assert destination.read_bytes() == b"before"


def test_source_manifest_rejects_non_mapping_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema": promote.CATALOG_SCHEMA,
                "release_id": source.name,
                "artifacts": [1],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(promote, "_artifact_contract", lambda _config: [])

    with pytest.raises(ValueError, match="artifact descriptor is malformed"):
        promote._validate_source_artifacts(source, {})


def test_promotion_copies_and_fsyncs_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, catalogs, source, config_path, _artifact = _promotion_fixture(
        tmp_path, monkeypatch
    )
    real_fsync_directory = promote._fsync_directory
    fsynced: list[Path] = []

    def record_fsync(path: Path) -> None:
        fsynced.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(promote, "_fsync_directory", record_fsync)
    monkeypatch.setattr(
        promote,
        "validate_release",
        lambda path, **_kwargs: json.loads((path / "manifest.json").read_text()),
    )

    destination = promote.promote_catalog(
        source_release=source,
        config_path=config_path,
        output_dir=repository / "DATA",
    )

    assert destination == catalogs / "catalog-promoted"
    assert destination.is_dir()
    assert (destination / "artifact.jsonl").stat().st_ino != (
        source / "artifact.jsonl"
    ).stat().st_ino
    assert any(path == catalogs for path in fsynced)
    assert any(path.name.startswith(".catalog-promote-") for path in fsynced)
    (source / "artifact.jsonl").write_bytes(b"changed")
    assert (destination / "artifact.jsonl").read_bytes() == b"frozen artifact\n"


def test_promotion_removes_destination_after_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository, catalogs, source, config_path, _artifact = _promotion_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        promote,
        "validate_release",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("post-publish validation failed")
        ),
    )

    with pytest.raises(ValueError, match="post-publish"):
        promote.promote_catalog(
            source_release=source,
            config_path=config_path,
            output_dir=catalogs.parent,
        )

    assert not (catalogs / "catalog-promoted").exists()
    assert not list(catalogs.glob(".catalog-promote-*"))


def test_promotion_rejects_symlink_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    real_catalogs = repository / "real" / "catalogs"
    real_catalogs.mkdir(parents=True)
    source = real_catalogs / "source"
    source.mkdir()
    config = repository / "config.yaml"
    config.write_text("fixture: true\n", encoding="utf-8")

    output = repository / "DATA"
    output.mkdir()
    output_link = repository / "DATA-link"
    output_link.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="output directory.*symlink"):
        promote.promote_catalog(source_release=source, config_path=config, output_dir=output_link)

    catalogs_link = output / "catalogs"
    catalogs_link.symlink_to(real_catalogs, target_is_directory=True)
    with pytest.raises(ValueError, match="catalog directory.*symlink"):
        promote.promote_catalog(source_release=source, config_path=config, output_dir=output)

    source_link = real_catalogs / "source-link"
    source_link.symlink_to(source, target_is_directory=True)
    catalogs_link.unlink()
    (output / "catalogs").mkdir()
    with pytest.raises(ValueError, match="source release.*symlink"):
        promote.promote_catalog(
            source_release=source_link, config_path=config, output_dir=output
        )


def test_promotion_rejects_dangling_catalog_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "DATA"
    output.mkdir()
    catalogs = output / "catalogs"
    catalogs.symlink_to(tmp_path / "missing", target_is_directory=True)
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("fixture: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="catalog directory.*symlink"):
        promote.promote_catalog(source_release=source, config_path=config, output_dir=output)


def test_promotion_rejects_output_outside_repository_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    (repository / "DATA" / "catalogs").mkdir(parents=True)
    monkeypatch.setattr(promote, "REPOSITORY_ROOT", repository)
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    with pytest.raises(ValueError, match="repository DATA directory"):
        promote.promote_catalog(
            source_release=repository / "DATA" / "catalogs" / "source",
            config_path=repository / "config.yaml",
            output_dir=outside,
        )


def test_promotion_rejects_existing_destination_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository, catalogs, source, config_path, _artifact = _promotion_fixture(
        tmp_path, monkeypatch
    )
    destination_target = catalogs / "target"
    destination_target.mkdir()
    (catalogs / "catalog-promoted").symlink_to(
        destination_target, target_is_directory=True
    )

    with pytest.raises(ValueError, match="destination release.*directory"):
        promote.promote_catalog(
            source_release=source,
            config_path=config_path,
            output_dir=catalogs.parent,
        )
