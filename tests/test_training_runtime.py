from pathlib import Path
from types import SimpleNamespace

import pytest

import latent_renderer_training.runtime as runtime
from latent_renderer_training.runtime import (
    F0_FAILURE_RESULT_SCHEMA,
    METHOD_RESULT_SCHEMA,
    _validate_result,
    dispatch_training,
)
from latent_renderer_training.methods.runner import COMMON_HYPERPARAMETERS, OPD_QUERY_BUDGET
from tests.test_latent_renderer_training import (
    _complete_run_contract,
    _training_authorization,
)


def _binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(authorization, method="rl")
    return authorization.bind_run_contract(contract)


def _result(binding, **overrides):
    value = {
        "schema": METHOD_RESULT_SCHEMA,
        "run_id": binding.contract["run_id"],
        "method": binding.contract["method"],
        "run_contract_sha256": binding.contract_hash,
        "status": "training_complete",
        "benchmark_status": "pending",
    }
    value.update(overrides)
    return value


def test_runtime_dispatches_only_the_source_registered_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    binding = _binding(tmp_path, monkeypatch)
    imports = []

    def import_module(name):
        imports.append(name)
        return SimpleNamespace(run_rl=lambda supplied: _result(supplied))

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    assert dispatch_training(binding)["status"] == "training_complete"
    assert imports == ["latent_renderer_training.methods.rl"]


def test_runtime_dispatches_the_source_registered_opd_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(
        authorization,
        method="opd",
        query_budget=OPD_QUERY_BUDGET,
        method_hyperparameters={"method": "opd", **COMMON_HYPERPARAMETERS},
    )
    binding = authorization.bind_run_contract(contract)
    imports = []

    def import_module(name):
        imports.append(name)
        return SimpleNamespace(run_opd=lambda supplied: _result(supplied))

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    assert dispatch_training(binding)["method"] == "opd"
    assert imports == ["latent_renderer_training.methods.opd"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("method", "dpo"),
        ("run_id", "other-run"),
        ("run_contract_sha256", "0" * 64),
        ("status", "dry-run"),
    ),
)
def test_runtime_rejects_a_fabricated_success_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
):
    binding = _binding(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            run_rl=lambda supplied: _result(supplied, **{field: value})
        ),
    )

    with pytest.raises(RuntimeError, match=field):
        dispatch_training(binding)


def test_runtime_handler_rejects_a_contract_outside_the_frozen_method_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    binding = _binding(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="method hyperparameters"):
        dispatch_training(binding)


def test_runtime_accepts_a_terminal_f0_screen_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    binding = authorization.bind_run_contract(
        _complete_run_contract(authorization, method="f0")
    )
    result = {
        "schema": F0_FAILURE_RESULT_SCHEMA,
        "run_id": binding.contract["run_id"],
        "method": "f0",
        "run_contract_sha256": binding.contract_hash,
        "status": "screen_failed",
        "stage": "train_gate",
        "reason": "the registered screen did not pass",
        "benchmark_status": "not_applicable",
    }

    assert _validate_result(result, binding=binding, method="f0") == result


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("method", "rl"),
        ("schema", METHOD_RESULT_SCHEMA),
        ("stage", "post_hoc_retry"),
        ("reason", ""),
        ("benchmark_status", "pending"),
    ),
)
def test_runtime_rejects_a_fabricated_f0_screen_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    binding = authorization.bind_run_contract(
        _complete_run_contract(authorization, method="f0")
    )
    result = {
        "schema": F0_FAILURE_RESULT_SCHEMA,
        "run_id": binding.contract["run_id"],
        "method": "f0",
        "run_contract_sha256": binding.contract_hash,
        "status": "screen_failed",
        "stage": "train_gate",
        "reason": "the registered screen did not pass",
        "benchmark_status": "not_applicable",
    }
    result[field] = value

    with pytest.raises(RuntimeError):
        _validate_result(result, binding=binding, method="f0")
