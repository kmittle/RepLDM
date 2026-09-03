from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latent_renderer_training.artifacts import module_state_sha256
from latent_renderer_training.contracts import contract_hash
from latent_renderer_training.ledger import QueryLedger
import latent_renderer_training.methods.f0 as f0_module
from latent_renderer_training.operations import LedgeredOperationExecutor
from latent_renderer_training.storage import CheckpointProvenance
from tests.test_latent_renderer_training import (
    _BoundLinear,
    _complete_run_contract,
    _training_authorization,
)


@dataclass(frozen=True)
class _FitRow:
    stable_id: str

    def to(self, _device: torch.device) -> "_FitRow":
        return self


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    renderer = _BoundLinear()
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(
        authorization,
        method="f0",
        renderer_frame_contract_hash=renderer.frame_contract_hash,
        calibration_hash=renderer.calibration_hash,
        action_contract_hash=contract_hash(renderer.contract),
        initial_renderer_state_sha256=module_state_sha256(renderer),
        query_budget={"optimizer_step": 3},
    )
    binding = authorization.bind_run_contract(contract)
    initial_path = tmp_path / "initial-renderer.bin"
    initial_path.write_bytes(b"immutable C0 checkpoint identity")
    initial = CheckpointProvenance.capture(
        initial_path,
        renderer_state_sha256=module_state_sha256(renderer),
    )
    ledger = QueryLedger(
        contract["paths"]["ledger_path"],
        contract["query_budget"],
        run_contract=contract,
        authorization_binding=binding,
    )
    executor = LedgeredOperationExecutor(ledger, authorization_binding=binding)
    return SimpleNamespace(
        renderer=renderer,
        reference_renderer=copy.deepcopy(renderer),
        binding=binding,
        initial_checkpoint_provenance=initial,
        operation_executor=executor,
        ledger=ledger,
    )


def test_f0_fit_resumes_complete_state_and_repairs_latest_update_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    rows = tuple(_FitRow(f"target-{index}") for index in range(8))
    batches = (rows, rows, rows)
    monkeypatch.setattr(f0_module, "F0_FIT_STEPS", 3)
    monkeypatch.setattr(
        f0_module,
        "f0_fit_batches",
        lambda _values, *, optimizer_seed: batches,
    )

    def objective(renderer, _reference, _batch, **_kwargs):
        return renderer(torch.ones(1, 1)).square().mean()

    monkeypatch.setattr(f0_module, "f0_objective", objective)
    first = f0_module._train_fit(
        runtime,
        rows,
        optimizer_seed=f0_module.F0_FOLD_OPTIMIZATION_SEEDS[0],
        role="crossfit-fold-0",
        held_out_fold=0,
    )
    record_path = (
        Path(runtime.binding.contract["paths"]["output_dir"])
        / "f0-updates"
        / "crossfit-fold-0"
        / "update-002.json"
    )
    original_record = record_path.read_bytes()
    record_path.unlink()
    records_before = tuple(runtime.ledger.verified_records())

    resumed = f0_module._train_fit(
        runtime,
        rows,
        optimizer_seed=f0_module.F0_FOLD_OPTIMIZATION_SEEDS[0],
        role="crossfit-fold-0",
        held_out_fold=0,
    )

    assert record_path.read_bytes() == original_record
    assert tuple(runtime.ledger.verified_records()) == records_before
    assert resumed.raw_renderer_state_sha256 == first.raw_renderer_state_sha256
    assert resumed.ema_renderer_state_sha256 == first.ema_renderer_state_sha256
    assert resumed.checkpoint == first.checkpoint
    assert len(resumed.updates) == 3
