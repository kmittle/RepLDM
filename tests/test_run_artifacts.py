import copy
import json
from pathlib import Path

import pytest
import torch

from latent_renderer_training.artifacts import module_state_sha256
from latent_renderer_training.contracts import contract_hash
from latent_renderer_training.ledger import QueryLedger
from tests.test_latent_renderer_training import (
    _BoundLinear,
    _complete_run_contract,
    _training_authorization,
)


def _bound_parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    authorization = _training_authorization(tmp_path, monkeypatch)
    renderer = _BoundLinear()
    contract = _complete_run_contract(
        authorization,
        renderer_frame_contract_hash=renderer.frame_contract_hash,
        calibration_hash=renderer.calibration_hash,
        action_contract_hash=contract_hash(renderer.contract),
        initial_renderer_state_sha256=module_state_sha256(renderer),
    )
    return authorization, renderer, contract


def test_binding_verifies_every_registered_artifact_and_initial_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization, renderer, contract = _bound_parts(tmp_path, monkeypatch)
    binding = authorization.bind_run_contract(contract)

    binding.validate_current(component=renderer)
    binding.validate_initial_renderer(renderer)
    provenance = binding.provenance()["verified_run_artifacts"]
    assert provenance["contract_sha256"] == binding.contract_hash
    assert provenance["initial_renderer_state_sha256"] == module_state_sha256(
        renderer
    )
    assert len(provenance["descriptors"]) == 13
    assert {item["label"] for item in provenance["payload_files"]} == {
        "initial renderer checkpoint",
        "model_assets_manifest[0]",
        "reward_assets_manifest[0]",
        "OPSD teacher checkpoint",
        "F0 evidence 0",
        "F0 evidence 1",
        "F0 evidence 2",
        "F0 evidence 3",
        "F0 evidence 4",
    }


@pytest.mark.parametrize("target", ("descriptor", "nested_payload"))
def test_binding_rejects_tampered_run_artifact_before_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
):
    authorization, _renderer, contract = _bound_parts(tmp_path, monkeypatch)
    if target == "descriptor":
        path = Path(contract["artifacts"]["reward_config"]["path"])
    else:
        manifest = Path(contract["artifacts"]["model_assets_manifest"]["path"])
        path = Path(json.loads(manifest.read_text(encoding="utf-8"))["files"][0]["path"])
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        authorization.bind_run_contract(contract)


def test_binding_detects_run_artifact_drift_after_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization, _renderer, contract = _bound_parts(tmp_path, monkeypatch)
    binding = authorization.bind_run_contract(contract)
    path = Path(contract["artifacts"]["basis_provider_config"]["path"])
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="verified run artifact changed"):
        binding.validate_current()


def test_initial_renderer_state_is_checked_independently_of_static_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization, renderer, contract = _bound_parts(tmp_path, monkeypatch)
    binding = authorization.bind_run_contract(contract)
    changed = copy.deepcopy(renderer)
    with torch.no_grad():
        changed.weight.add_(1.0)

    binding.validate_current(component=changed)
    with pytest.raises(ValueError, match="renderer state"):
        binding.validate_initial_renderer(changed)


def test_artifact_schema_rejects_unregistered_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization, _renderer, contract = _bound_parts(tmp_path, monkeypatch)
    contract["artifacts"]["unused"] = dict(contract["artifacts"]["reward_config"])

    with pytest.raises(ValueError, match="registered schema"):
        authorization.bind_run_contract(contract)


def test_formal_ledger_cannot_disable_binding_or_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization, _renderer, contract = _bound_parts(tmp_path, monkeypatch)
    binding = authorization.bind_run_contract(contract)

    with pytest.raises(ValueError, match="strict provenance"):
        QueryLedger(
            tmp_path / "relaxed.jsonl",
            contract["query_budget"],
            run_contract=contract,
            strict_provenance=False,
            authorization_binding=binding,
        )
    with pytest.raises(RuntimeError, match="formal query ledgers"):
        QueryLedger(
            tmp_path / "unbound.jsonl",
            contract["query_budget"],
            run_contract=contract,
        )
    with pytest.raises(TypeError, match="authorization_binding"):
        QueryLedger(
            tmp_path / "raw.jsonl",
            contract["query_budget"],
            run_contract=contract,
            authorization=authorization,
        )
    assert not (tmp_path / "relaxed.jsonl").exists()
    assert not (tmp_path / "unbound.jsonl").exists()
    assert not (tmp_path / "raw.jsonl").exists()
