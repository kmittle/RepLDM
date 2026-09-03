from __future__ import annotations

import json
import socket
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "eval-pipeline"
sys.path.insert(0, str(EVAL_ROOT))

import data_catalog.selected as selected_module
import data_catalog.selected_builder as builder_module
from data_catalog.io import iter_jsonl
from data_catalog.selected import model_binding_sha256, validate_selected_view_release
from data_catalog.selected_builder import (
    ClassifierGateResult,
    DistanceGateResult,
    SimilarityGateResult,
    TokenizerGateResult,
    build_selected_view_release,
    verify_selected_view_runtime,
)
from tests.test_selected_view_release import COMMIT, SelectedReleaseFixture


class _Runtime:
    def __init__(
        self,
        fixture: SelectedReleaseFixture,
        *,
        truncate_id: str | None = None,
        wrong_bindings: bool = False,
        incomplete_index: bool = False,
    ) -> None:
        config = fixture.config
        classifier_hash = model_binding_sha256(config["classifier"]["model"])
        self.bindings = {
            "classifier": classifier_hash,
            "image_embedding": model_binding_sha256(
                config["image_embedding"]["model"]
            ),
            "protected:image_embedding": config["protected_index"][
                "image_embedding"
            ]["sha256"],
            "protected:phash": config["protected_index"]["phash"]["sha256"],
            "protected:semantic_text": config["protected_index"]["semantic_text"][
                "sha256"
            ],
            "semantic_text": model_binding_sha256(config["semantic_text"]["model"]),
            **{
                f"tokenizer:{row['id']}": model_binding_sha256(row["model"])
                for row in config["tokenizers"]
            },
        }
        if wrong_bindings:
            self.bindings["classifier"] = "0" * 64
        self.index_counts = {
            "semantic_text": 46619,
            "phash": 37160,
            "image_embedding": 37160,
        }
        if incomplete_index:
            self.index_counts["semantic_text"] -= 1
        self.strata = {row["id"]: row["stratum"] for row in fixture.rows}
        self.truncate_prompt = next(
            (row["model_prompt"] for row in fixture.rows if row["id"] == truncate_id),
            None,
        )
        self.calls: Counter[str] = Counter()

    def tokenize(
        self,
        tokenizer_id: str,
        prompt: str,
        *,
        max_tokens: int,
        add_special_tokens: bool,
        truncation: bool,
    ) -> TokenizerGateResult:
        self.calls[f"tokenize:{tokenizer_id}"] += 1
        assert prompt
        assert max_tokens == 77
        assert add_special_tokens is True
        assert truncation is False
        truncated = prompt == self.truncate_prompt
        return TokenizerGateResult(
            token_count=78 if truncated else 12,
            truncated=truncated,
        )

    def classify(self, record_id, image, class_templates):
        self.calls["classify"] += 1
        assert image.width == 8 and image.height == 8
        assert set(class_templates) == set(selected_module.STRATA)
        return ClassifierGateResult(self.strata[record_id], 0.9, 0.8)

    def nearest_protected_text(self, prompt: str) -> SimilarityGateResult:
        self.calls["semantic_text"] += 1
        return SimilarityGateResult("protected-0", 0.7)

    def nearest_protected_phash(self, phash: str) -> DistanceGateResult:
        self.calls["phash"] += 1
        assert len(phash) == 16
        return DistanceGateResult("protected-0", 5)

    def nearest_protected_image(self, image) -> SimilarityGateResult:
        self.calls["image_embedding"] += 1
        return SimilarityGateResult("protected-0", 0.8)


def _formal_patches(
    monkeypatch: pytest.MonkeyPatch, fixture: SelectedReleaseFixture
) -> None:
    git = {
        "commit": COMMIT,
        "branch": "fixture",
        "dirty": False,
        "worktree_status_sha256": "0" * 64,
        "upstream_ref": "refs/remotes/origin/fixture",
        "upstream_commit": COMMIT,
        "pushed": True,
    }
    monkeypatch.setattr(builder_module, "enforce_git_gate", lambda *args, **kwargs: git)
    monkeypatch.setattr(
        builder_module,
        "_tracked_config_path",
        lambda *args, **kwargs: "eval-pipeline/configs/selected_view_v1.json",
    )
    monkeypatch.setattr(
        selected_module,
        "_validate_candidate_parent",
        lambda *args, **kwargs: json.loads(
            (fixture.parent / "manifest.json").read_text(encoding="utf-8")
        ),
    )
    monkeypatch.setattr(selected_module, "_validate_selected_git", lambda *args, **kwargs: None)


def _build(
    fixture: SelectedReleaseFixture,
    monkeypatch: pytest.MonkeyPatch,
    runtime: _Runtime | None,
) -> Path:
    _formal_patches(monkeypatch, fixture)
    return build_selected_view_release(
        config_path=fixture.release / "selection-config.json",
        parent_release=fixture.parent,
        output_root=fixture.selected_root,
        runtime_factory=(
            (lambda config, parent, repository: runtime) if runtime is not None else None
        ),
        repository_root=fixture.root,
        allow_dirty=False,
    )


def test_builder_deterministically_builds_and_reverifies_64_plus_32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    runtime = _Runtime(fixture)
    network_blocked = False

    def factory(config, parent, repository):
        nonlocal network_blocked
        try:
            socket.create_connection(("example.invalid", 443))
        except RuntimeError as exc:
            network_blocked = "network access is disabled" in str(exc)
        else:
            raise AssertionError("runtime factory unexpectedly reached the network")
        return runtime

    _formal_patches(monkeypatch, fixture)
    release = build_selected_view_release(
        config_path=fixture.release / "selection-config.json",
        parent_release=fixture.parent,
        output_root=fixture.selected_root,
        runtime_factory=factory,
        repository_root=fixture.root,
    )
    manifest_bytes = (release / "manifest.json").read_bytes()
    payload_bytes = (release / "selected-payload.jsonl").read_bytes()
    repeated = build_selected_view_release(
        config_path=fixture.release / "selection-config.json",
        parent_release=fixture.parent,
        output_root=fixture.selected_root,
        runtime_factory=factory,
        repository_root=fixture.root,
    )

    assert network_blocked is True
    assert repeated == release
    assert (repeated / "manifest.json").read_bytes() == manifest_bytes
    assert (repeated / "selected-payload.jsonl").read_bytes() == payload_bytes
    manifest = validate_selected_view_release(
        release, repository_root=fixture.root, require_formal=True
    )
    assert manifest["training_ready"] is True
    rows = list(iter_jsonl(release / "selected-payload.jsonl"))
    assert Counter(row["selected_split"] for row in rows) == {
        "train": 64,
        "validation": 32,
    }
    assert {row["fold"] for row in rows if row["selected_split"] == "train"} == {
        0,
        1,
        2,
        3,
    }
    assert runtime.calls["classify"] >= 96 * 4
    assert runtime.calls["semantic_text"] >= 96 * 2 * 4
    verify_selected_view_runtime(
        release,
        runtime=runtime,
        repository_root=fixture.root,
        require_formal=True,
    )


def test_missing_runtime_emits_revalidatable_non_training_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    release = _build(fixture, monkeypatch, None)
    manifest = validate_selected_view_release(
        release,
        repository_root=fixture.root,
        require_formal=True,
        require_training_ready=False,
    )
    report = json.loads((release / "gate-report.json").read_text(encoding="utf-8"))
    assert manifest["training_ready"] is False
    assert "selected_payload" not in manifest
    assert [row["code"] for row in report["failures"]] == ["runtime_unavailable"]
    with pytest.raises(ValueError, match="not training-ready"):
        validate_selected_view_release(
            release, repository_root=fixture.root, require_formal=True
        )


@pytest.mark.parametrize("missing_asset", ["model", "calibration", "protected_index"])
def test_missing_bound_dependency_fails_before_runtime_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_asset: str
) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    paths = {
        "model": fixture.config["classifier"]["model"]["files"][0]["path"],
        "calibration": fixture.config["semantic_text"]["calibration"]["path"],
        "protected_index": fixture.config["protected_index"]["semantic_text"]["path"],
    }
    Path(paths[missing_asset]).unlink()
    runtime = _Runtime(fixture)
    release = _build(fixture, monkeypatch, runtime)
    report = json.loads((release / "gate-report.json").read_text(encoding="utf-8"))
    assert report["config_valid"] is False
    assert report["runtime_ready"] is False
    assert report["failures"][0]["code"] == "config_invalid"
    assert not runtime.calls


@pytest.mark.parametrize("runtime_error", ["wrong_binding", "incomplete_index"])
def test_wrong_runtime_contract_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime_error: str
) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    runtime = _Runtime(
        fixture,
        wrong_bindings=runtime_error == "wrong_binding",
        incomplete_index=runtime_error == "incomplete_index",
    )
    release = _build(fixture, monkeypatch, runtime)
    report = json.loads((release / "gate-report.json").read_text(encoding="utf-8"))
    assert report["runtime_ready"] is False
    assert report["failures"][0]["code"] == "runtime_initialization_failed"
    assert "selected_payload" not in json.loads(
        (release / "manifest.json").read_text(encoding="utf-8")
    )


def test_one_truncated_prompt_rejects_the_whole_insufficient_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    rejected_id = fixture.rows[0]["id"]
    runtime = _Runtime(fixture, truncate_id=rejected_id)
    release = _build(fixture, monkeypatch, runtime)
    report = json.loads((release / "gate-report.json").read_text(encoding="utf-8"))
    assert report["selection_complete"] is False
    assert report["rejection_counts"]["tokenizer_truncation"] == 1
    assert any(
        row["code"] == "insufficient_source_stratum_quota"
        for row in report["failures"]
    )
    assert "selected_payload" not in json.loads(
        (release / "manifest.json").read_text(encoding="utf-8")
    )
