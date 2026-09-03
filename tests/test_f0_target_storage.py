from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from latent_renderer_training.storage import AtomicF0TargetStore
from tests.test_f0_targets import _row
from tests.test_latent_renderer_training import (
    _complete_run_contract,
    _training_authorization,
)
from tests.test_training_method_objectives import _renderer


def _parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(authorization, method="f0")
    binding = authorization.bind_run_contract(contract)
    return authorization, contract, binding, _row(_renderer())


def test_f0_target_store_round_trip_is_content_addressed_and_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, row = _parts(tmp_path, monkeypatch)
    store = AtomicF0TargetStore(binding)
    manifest = store.save(row)

    payload = Path(manifest["payload"]["path"])
    assert payload.parent == store.root
    assert payload.name == (
        f"{row.stable_id}-{manifest['payload']['sha256'][:16]}.pt"
    )
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == manifest["payload"][
        "sha256"
    ]
    manifest_core = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    expected_manifest_hash = hashlib.sha256(
        json.dumps(
            manifest_core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["manifest_sha256"] == expected_manifest_hash

    original_load = torch.load
    weights_only_calls = []

    def observed_load(source, map_location=None, *, weights_only=False):
        weights_only_calls.append(weights_only)
        return original_load(
            source, map_location=map_location, weights_only=weights_only
        )

    monkeypatch.setattr(torch, "load", observed_load)
    restored = store.load(row.stable_id)

    assert weights_only_calls == [True]
    assert restored.stable_id == row.stable_id
    assert restored.record_sha256 == row.record_sha256
    assert torch.equal(restored.state.raw_bases, row.state.raw_bases)
    assert torch.equal(restored.plus_transition, row.plus_transition)
    assert restored.state.raw_bases.data_ptr() != row.state.raw_bases.data_ptr()


def test_f0_target_store_load_or_none_only_skips_absent_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, row = _parts(tmp_path, monkeypatch)
    store = AtomicF0TargetStore(binding)

    assert store.load_or_none(row.stable_id) is None
    assert not store.has(row.stable_id)
    store.save(row)
    assert store.load_or_none(row.stable_id).record_sha256 == row.record_sha256
    assert store.has(row.stable_id)


def test_f0_target_store_never_overwrites_duplicate_stable_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, row = _parts(tmp_path, monkeypatch)
    store = AtomicF0TargetStore(binding)
    first = store.save(row)
    changed = replace(row, pair_weight=0.25)
    assert changed.stable_id == row.stable_id
    assert changed.record_sha256 != row.record_sha256

    with pytest.raises(FileExistsError, match="stable_id already exists"):
        store.save(changed)

    assert json.loads((store.root / f"{row.stable_id}.json").read_text()) == first
    assert len(list(store.root.glob("*.pt"))) == 1


def test_f0_target_store_rejects_payload_and_manifest_tampering_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, row = _parts(tmp_path, monkeypatch)
    store = AtomicF0TargetStore(binding)
    manifest = store.save(row)
    payload = Path(manifest["payload"]["path"])
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="payload bytes differ"):
        store.load(row.stable_id)

    second = replace(row, prompt_id="prompt-manifest-tamper")
    store.save(second)
    manifest_path = store.root / f"{second.stable_id}.json"
    changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed["record_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="manifest SHA-256"):
        store.load(second.stable_id)


def test_f0_target_store_rejects_symlinks_for_publish_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, row = _parts(tmp_path, monkeypatch)
    store = AtomicF0TargetStore(binding)
    store.root.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("do not replace", encoding="utf-8")
    (store.root / f"{row.stable_id}.json").symlink_to(outside)

    with pytest.raises(FileExistsError, match="stable_id already exists"):
        store.save(row)
    assert outside.read_text(encoding="utf-8") == "do not replace"
    assert not list(store.root.glob("*.pt"))

    linked_row = replace(row, prompt_id="prompt-payload-link")
    manifest_linked = store.save(linked_row)
    payload = Path(manifest_linked["payload"]["path"])
    moved = store.root / "moved-target.pt"
    payload.rename(moved)
    payload.symlink_to(moved)

    with pytest.raises(ValueError, match="not an ordinary file"):
        store.load(linked_row.stable_id)


def test_f0_target_store_rejects_manifest_from_another_run_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, contract, binding, row = _parts(tmp_path, monkeypatch)
    first = AtomicF0TargetStore(binding)
    first.save(row)
    other_contract = _complete_run_contract(
        authorization,
        method="f0",
        run_id="other-f0-run",
        paths=contract["paths"],
    )
    other_binding = authorization.bind_run_contract(other_contract)
    other = AtomicF0TargetStore(other_binding)

    with pytest.raises(ValueError, match="run_contract_sha256"):
        other.load(row.stable_id)


def test_f0_target_store_rejects_non_f0_authorized_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _training_authorization(tmp_path, monkeypatch)
    wrong_contract = _complete_run_contract(authorization, method="rl")
    wrong_binding = authorization.bind_run_contract(wrong_contract)

    with pytest.raises(ValueError, match="requires an F0 run contract"):
        AtomicF0TargetStore(wrong_binding)


def test_f0_target_store_revalidates_binding_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, row = _parts(tmp_path, monkeypatch)
    store = AtomicF0TargetStore(binding)
    binding_type = type(binding)
    original_validate = binding_type.validate_current
    calls = 0

    def revoke_during_save(self, *args, **kwargs):
        nonlocal calls
        if self is binding:
            calls += 1
            if calls == 5:
                raise RuntimeError("planned authorization revocation")
        return original_validate(self, *args, **kwargs)

    monkeypatch.setattr(binding_type, "validate_current", revoke_during_save)
    with pytest.raises(RuntimeError, match="planned authorization revocation"):
        store.save(row)

    assert calls == 5
    assert store.root.is_dir()
    assert not list(store.root.iterdir())


def test_f0_target_store_revalidates_binding_after_safe_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, row = _parts(tmp_path, monkeypatch)
    store = AtomicF0TargetStore(binding)
    store.save(row)
    binding_type = type(binding)
    original_validate = binding_type.validate_current
    calls = 0

    def revoke_after_decode(self, *args, **kwargs):
        nonlocal calls
        if self is binding:
            calls += 1
            if calls == 2:
                raise RuntimeError("planned post-load revocation")
        return original_validate(self, *args, **kwargs)

    monkeypatch.setattr(binding_type, "validate_current", revoke_after_decode)
    with pytest.raises(RuntimeError, match="planned post-load revocation"):
        store.load(row.stable_id)
    assert calls == 2
