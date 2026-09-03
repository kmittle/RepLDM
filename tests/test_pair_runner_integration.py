import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import latent_renderer_training.methods.runner as runner
from latent_renderer_training.artifacts import module_state_sha256
from latent_renderer_training.contracts import contract_hash
from latent_renderer_training.methods.runner import (
    COMMON_HYPERPARAMETERS,
    OPTIMIZATION_SEEDS,
    OPD_QUERY_BUDGET,
    PAIR_QUERY_BUDGET,
    run_opd_training,
    run_pair_training,
)
from latent_renderer_training.storage import CheckpointProvenance
from latent_renderer_training.trainer import UpdateRecord
from tests.test_formal_training_hardening import _BoundLinear
from tests.test_latent_renderer_training import (
    _complete_run_contract,
    _training_authorization,
)


def _json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _descriptor(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


class _FakeLedger:
    def __init__(self, root: Path) -> None:
        self.path = root / "fake-ledger.jsonl"
        self.seal_path = root / "fake-ledger.jsonl.seal"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('{"fake":"ledger"}\n', encoding="utf-8")
        self.seal_path.write_text('{"fake":"seal"}\n', encoding="utf-8")
        self.label_hashes: list[str] = []

    def summary(self):
        return {
            "budget": dict(PAIR_QUERY_BUDGET),
            "reserved": dict(PAIR_QUERY_BUDGET),
            "remaining": {key: 0 for key in PAIR_QUERY_BUDGET},
            "completed_receipts": sum(PAIR_QUERY_BUDGET.values()),
            "successful_amount": dict(PAIR_QUERY_BUDGET),
            "failed_amount": {key: 0 for key in PAIR_QUERY_BUDGET},
            "wall_seconds": {key: 0.0 for key in PAIR_QUERY_BUDGET},
            "unfinished_reservations": 0,
            "record_count": 2 * sum(PAIR_QUERY_BUDGET.values()),
            "root_record_hash": "1" * 64,
            "tip_record_hash": "2" * 64,
        }

    def successful_output_hashes(self, kind: str):
        assert kind == "label"
        return tuple(self.label_hashes)


class _FakeOpdLedger(_FakeLedger):
    def summary(self):
        return {
            "budget": dict(OPD_QUERY_BUDGET),
            "reserved": dict(OPD_QUERY_BUDGET),
            "remaining": {key: 0 for key in OPD_QUERY_BUDGET},
            "completed_receipts": sum(OPD_QUERY_BUDGET.values()),
            "successful_amount": dict(OPD_QUERY_BUDGET),
            "failed_amount": {key: 0 for key in OPD_QUERY_BUDGET},
            "wall_seconds": {key: 0.0 for key in OPD_QUERY_BUDGET},
            "unfinished_reservations": 0,
            "record_count": 2 * sum(OPD_QUERY_BUDGET.values()),
            "root_record_hash": "1" * 64,
            "tip_record_hash": "2" * 64,
        }


class _FakeTrainer:
    instances = []

    def __init__(self, model, optimizer, **_kwargs) -> None:
        self.model = model
        self.optimizer = optimizer
        self.step = 0
        self.ema_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        self.round_two_resets = 0
        type(self).instances.append(self)

    def optimizer_context(self, **kwargs):
        return dict(kwargs)

    def update(self, loss_fn, *, operation_context):
        loss = loss_fn()
        assert isinstance(operation_context, dict)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(0.001)
        self.step += 1
        self.ema_state = {
            name: value.detach().clone() for name, value in self.model.state_dict().items()
        }
        return UpdateRecord(self.step, float(loss.detach()), 0.0)

    def start_round_two_from_ema(self, optimizer):
        assert self.step == 32
        assert optimizer.state == {}
        self.model.load_state_dict(self.ema_state, strict=True)
        self.optimizer = optimizer
        self.round_two_resets += 1
        return {
            "schema": "repldm.round_transition.v1",
            "step": self.step,
            "source": "round_1_ema",
            "raw_policy_state_sha256": module_state_sha256(self.model),
            "ema_policy_state_sha256": module_state_sha256(self.model),
            "discarded_optimizer_state_entries": 1,
            "new_optimizer_state_entries": 0,
        }

    def save(self, path: str, *, extra=None):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            json.dumps({"step": self.step, "extra": extra}, sort_keys=True).encode()
        )
        return {"path": str(destination), "step": self.step}


class _FakeRuntime:
    def __init__(self, binding, model, records, checkpoint, ledger) -> None:
        self.binding = binding
        self.renderer = model
        self.reference_renderer = copy.deepcopy(model)
        for parameter in self.reference_renderer.parameters():
            parameter.requires_grad_(False)
        operation_executor = object()
        self.operation_executor = operation_executor
        self.adapter = SimpleNamespace(operation_executor=operation_executor)
        self.selected_records = tuple(records)
        self.initial_checkpoint_provenance = checkpoint
        self.ledger = ledger
        self.reward_statistics = _json(
            binding.contract["artifacts"]["reward_statistics"]["path"]
        )
        self.f0_gate = _json(binding.contract["artifacts"]["f0_gate"]["path"])
        self.opsd_teacher_state = _json(
            binding.contract["artifacts"]["opsd_teacher_state"]["path"]
        )
        self.training_cohort = _json(
            binding.contract["artifacts"]["cohort_manifest"]["path"]
        )
        self.provenance = {"schema": "fake.runtime.v1"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class _FakeOpdRuntime(_FakeRuntime):
    def __init__(self, binding, model, records, checkpoint, ledger) -> None:
        super().__init__(binding, model, records, checkpoint, ledger)
        self.teacher_renderer = copy.deepcopy(model)
        for parameter in self.teacher_renderer.parameters():
            parameter.requires_grad_(False)
        self.teacher_renderer.eval()
        self.teacher_checkpoint_provenance = checkpoint


def test_pair_runner_executes_and_persists_the_complete_frozen_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    model = _BoundLinear()
    contract = _complete_run_contract(
        authorization,
        query_budget=PAIR_QUERY_BUDGET,
        method="rl",
        method_hyperparameters={
            "method": "rl",
            **COMMON_HYPERPARAMETERS,
        },
        seed=OPTIMIZATION_SEEDS[0],
        renderer_frame_contract_hash=model.frame_contract_hash,
        calibration_hash=model.calibration_hash,
        action_contract_hash=contract_hash(model.contract),
        initial_renderer_state_sha256=module_state_sha256(model),
    )
    binding = authorization.bind_run_contract(contract)
    records = [
        json.loads(line)
        for line in authorization.selected_manifest_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    initial_file = tmp_path / "fake-initial.pt"
    initial_file.write_bytes(b"fake initial renderer")
    initial_checkpoint = CheckpointProvenance.capture(
        initial_file, renderer_state_sha256=module_state_sha256(model)
    )
    fake_ledger = _FakeLedger(tmp_path / "ledger")
    runtime = _FakeRuntime(binding, model, records, initial_checkpoint, fake_ledger)
    collected: list[tuple[int, int | None, str]] = []

    def fake_collect(
        runtime_value,
        behavior_renderer,
        block,
        *,
        storage_id,
        round_index,
        update_index,
        **_kwargs,
    ):
        assert runtime_value is runtime
        assert module_state_sha256(behavior_renderer) == module_state_sha256(
            runtime.renderer
        )
        collected.append((round_index, update_index, storage_id))
        label_hash = hashlib.sha256(storage_id.encode("utf-8")).hexdigest()
        fake_ledger.label_hashes.append(label_hash)
        manifest_path = tmp_path / "fake-rollouts" / f"{storage_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"storage_id": storage_id}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "terminal_rewards": {"plus": 1.0, "minus": 0.0, "anchor": 0.5},
            "label_receipt_output_sha256": label_hash,
        }
        return object(), metadata, _descriptor(manifest_path)

    def fake_save_checkpoint(runtime_value, renderer, *, filename, step, role):
        assert runtime_value is runtime
        path = tmp_path / "fake-checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{step}:{role}".encode("ascii"))
        return CheckpointProvenance.capture(
            path, renderer_state_sha256=module_state_sha256(renderer)
        )

    _FakeTrainer.instances.clear()
    monkeypatch.setattr(runner, "RendererTrainer", _FakeTrainer)
    monkeypatch.setattr(runner, "_collect_block", fake_collect)
    monkeypatch.setattr(runner, "_save_renderer_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(
        runner,
        "_objective",
        lambda _method, runtime_value, _values: runtime_value.renderer.weight.sum() * 0 + 1,
    )

    result = run_pair_training(binding, "rl", runtime_builder=lambda _binding: runtime)

    assert result["status"] == "training_complete"
    assert result["benchmark_status"] == "pending"
    assert len(collected) == 320
    assert sum(update is not None for _round, update, _storage in collected) == 256
    assert len(_FakeTrainer.instances) == 1
    assert _FakeTrainer.instances[0].step == 64
    assert _FakeTrainer.instances[0].round_two_resets == 1
    assert [len(item["update_records"]) for item in result["rounds"]] == [32, 32]

    rollout_index = _json(result["rollout_index"]["path"])
    assert rollout_index["rollout_count"] == 320
    assert rollout_index["label_count"] == 320
    assert rollout_index["label_receipt_output_sha256"] == fake_ledger.label_hashes
    validation = _json(result["validation_no_update"]["path"])
    assert validation["rollout_count"] == 64
    assert validation["optimizer_step_before"] == 64
    assert validation["optimizer_step_after"] == 64
    assert validation["weights_unchanged"] is True
    result_path = Path(result["result_path"])
    assert result_path.is_file()
    persisted = _json(str(result_path))
    assert persisted == result
    digest = persisted["result_sha256"]
    core = dict(persisted)
    core.pop("result_sha256")
    canonical = json.dumps(
        core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"
    assert digest == hashlib.sha256(canonical).hexdigest()


def test_opd_runner_uses_fresh_blocks_and_three_teacher_labels_per_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The OPD arm keeps its external-teacher allocation separate from pairs."""
    import tests.test_latent_renderer_training as training_fixtures
    import latent_renderer_training.gates as gates_module

    # The shared contract helper normally materializes the full F0 operation
    # ledger (hundreds of MB).  Gate parsing is outside this scheduling test;
    # retain the artifact descriptors while keeping the fixture lightweight.
    monkeypatch.setattr(
        training_fixtures,
        "_sealed_metric_ledger",
        lambda *_args, **_kwargs: (b'{"fixture":"ledger"}\n', b'{"fixture":"seal"}\n'),
    )
    monkeypatch.setattr(
        gates_module,
        "validate_f0_gate",
        lambda gate, **_kwargs: (gate, ()),
    )
    authorization = _training_authorization(tmp_path, monkeypatch)
    model = _BoundLinear()
    contract = _complete_run_contract(
        authorization,
        query_budget=OPD_QUERY_BUDGET,
        method="opd",
        method_hyperparameters={
            "method": "opd",
            **COMMON_HYPERPARAMETERS,
        },
        seed=OPTIMIZATION_SEEDS[0],
        renderer_frame_contract_hash=model.frame_contract_hash,
        calibration_hash=model.calibration_hash,
        action_contract_hash=contract_hash(model.contract),
        initial_renderer_state_sha256=module_state_sha256(model),
    )
    binding = authorization.bind_run_contract(contract)
    # The authorization fixture contains a large synthetic F0 ledger.  The
    # runner's schedule is the subject of this test, so avoid re-hashing that
    # fixture before every published JSON artifact (the binding/gate contract
    # itself is exercised by the authorization tests).
    monkeypatch.setattr(
        type(binding),
        "validate_current",
        lambda _self, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_require_runtime_gates", lambda *_args: None)
    records = [
        json.loads(line)
        for line in authorization.selected_manifest_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    initial_file = tmp_path / "fake-initial.pt"
    initial_file.write_bytes(b"fake initial renderer")
    initial_checkpoint = CheckpointProvenance.capture(
        initial_file, renderer_state_sha256=module_state_sha256(model)
    )
    fake_ledger = _FakeOpdLedger(tmp_path / "ledger")
    runtime = _FakeOpdRuntime(
        binding, model, records, initial_checkpoint, fake_ledger
    )
    collected: list[tuple[int, int | None, str]] = []

    def fake_collect_opd(
        runtime_value,
        behavior_renderer,
        block,
        *,
        storage_id,
        round_index,
        update_index,
        **_kwargs,
    ):
        assert runtime_value is runtime
        assert module_state_sha256(behavior_renderer) == module_state_sha256(
            runtime.renderer
        )
        collected.append((round_index, update_index, storage_id))
        manifest_path = tmp_path / "fake-rollouts" / f"{storage_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"storage_id": storage_id}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        labels = []
        for decision_index in runner.DECISION_INDICES:
            label_hash = hashlib.sha256(
                f"{storage_id}:{decision_index}".encode("utf-8")
            ).hexdigest()
            fake_ledger.label_hashes.append(label_hash)
            labels.append(label_hash)
        return (
            object(),
            {
                "terminal_rewards": {
                    "plus": 1.0,
                    "minus": 0.0,
                    "anchor": 0.5,
                    "teacher": 1.25,
                },
                "teacher_label_receipt_output_sha256": labels,
            },
            _descriptor(manifest_path),
        )

    def fake_save_checkpoint(runtime_value, renderer, *, filename, step, role):
        assert runtime_value is runtime
        path = tmp_path / "fake-checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{step}:{role}".encode("ascii"))
        return CheckpointProvenance.capture(
            path, renderer_state_sha256=module_state_sha256(renderer)
        )

    _FakeTrainer.instances.clear()
    monkeypatch.setattr(runner, "RendererTrainer", _FakeTrainer)
    monkeypatch.setattr(runner, "_collect_opd_block", fake_collect_opd)
    monkeypatch.setattr(runner, "_save_renderer_checkpoint", fake_save_checkpoint)
    # The synthetic fixture intentionally has a placeholder authorized hash;
    # teacher authentication itself is covered by the production/runtime tests.
    monkeypatch.setattr(
        runner,
        "_require_opd_teacher",
        lambda runtime_value, _contract: runtime_value.teacher_renderer,
    )
    monkeypatch.setattr(
        runner,
        "_objective",
        lambda _method, runtime_value, _values: runtime_value.renderer.weight.sum()
        * 0
        + 1,
    )

    result = run_opd_training(binding, runtime_builder=lambda _binding: runtime)

    assert result["status"] == "training_complete"
    assert result["method"] == "opd"
    assert len(collected) == 320
    assert sum(update is not None for _round, update, _storage in collected) == 256
    assert len(fake_ledger.label_hashes) == 960
    assert len(_FakeTrainer.instances) == 1
    assert _FakeTrainer.instances[0].step == 64
    rollout_index = _json(result["rollout_index"]["path"])
    assert rollout_index["rollout_count"] == 320
    assert rollout_index["label_count"] == 960
    assert rollout_index["label_receipt_output_sha256"] == fake_ledger.label_hashes
    validation = _json(result["validation_no_update"]["path"])
    assert validation["rollout_count"] == 64
    assert validation["weights_unchanged"] is True

    # OPD cannot be smuggled through the sampled-pair entry point.
    with pytest.raises(ValueError, match="only accepts sampled-pair methods"):
        run_pair_training(binding, "opd", runtime_builder=lambda _binding: runtime)
