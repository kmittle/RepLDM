import json
import hashlib
import random
from pathlib import Path

import pytest
import torch

from latent_renderer_training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
import latent_renderer_training.authorization as authorization_module
from latent_renderer_training.authorization import TrainingAuthorization, write_authorization_receipt
from latent_renderer_training.cli import main as training_probe
from latent_renderer_training.contracts import ActionSpaceContract, contract_hash
from latent_renderer_training.distributions import SquashedGaussian, transformed_gaussian_kl
from latent_renderer_training.ledger import QueryLedger
from latent_renderer_training.objectives import (
    dpo_loss,
    normalized_transition_loss,
    normalized_transition_losses,
    opd_loss,
    per_decision_rl_loss,
    search_distill_loss,
)
from latent_renderer_training.rollout import Transition, replay_shared_prefix
from latent_renderer_training.trainer import RendererTrainer
from latent_renderer_training.teachers import TargetStepConfig, construct_reward_targets


def _ledger_metadata(kind="reward_forward", amount=1):
    digest = "0" * 64
    return {
        "code_hash": digest,
        "data_hash": digest,
        "checkpoint_hash": digest,
        "method_allocation": {"kind": kind, "amount": amount},
        "split": "train",
        "prompt": "prompt-0",
        "seed": 0,
        "step": 0,
        "prefix": 0,
        "branch": "anchor",
        "action": [0.0, 0.0],
    }


def _ledger_result():
    return {
        "image_hash": "1" * 64,
        "reward_preprocess_hash": "2" * 64,
        "scalar_or_gradient": "scalar",
        "cached_parent": None,
        "value": 0.5,
    }


def _training_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TrainingAuthorization:
    """Create a tiny on-disk formal release for trainer contract tests."""
    # The production loader invokes the canonical catalog builder and Git
    # gate.  This fixture keeps the authorization semantics under test while
    # replacing those expensive external checks with deterministic stubs.
    monkeypatch.setattr(authorization_module, "_validate_formal_release", lambda *args: None)
    commit = "0" * 40
    monkeypatch.setattr(authorization_module, "_git_state", lambda *args: (commit, True))
    monkeypatch.setattr(authorization_module, "_git_upstream", lambda *args, **kwargs: None)
    release_id = "catalog-" + "a" * 20
    release = tmp_path / release_id
    release.mkdir()
    image = release / "payload.bin"
    image.write_bytes(b"selected-payload")
    image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    selected = release / "selected_payload.jsonl"
    selected.write_text(
        json.dumps(
            {
                "id": "source:row-0",
                "prompt": "unit test prompt",
                "image_path": str(image),
                "payload_sha256": image_hash,
                "training_eligible": True,
                "benchmark_exact_match": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = release / "training_candidates.jsonl"
    candidate.write_text(selected.read_text(encoding="utf-8"), encoding="utf-8")
    views = release / "training_views.jsonl"
    views.write_text(
        json.dumps(
            {
                "schema": "repldm.training_view.v1",
                "id": "unit_test_view",
                "artifact": "training_candidates.jsonl",
                "filter": {"prompt_is_not_null": True},
                "expected_rows": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def artifact(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    catalog = {
        "schema": "repldm.data_catalog.v1",
        "release_id": release_id,
        "complete": True,
        "training_ready": True,
        "candidate_catalog_complete": True,
        "development_build": False,
        "verify_paths": True,
        "payload_integrity_policy": {"training_ready": True},
        "git": {"commit": commit, "dirty": False, "pushed": True},
        "artifacts": [artifact(candidate), artifact(views)],
    }
    catalog_path = release / "manifest.json"
    catalog_path.write_text(
        json.dumps(catalog, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = tmp_path / "training_authorization.json"
    return write_authorization_receipt(
        receipt,
        catalog_manifest=catalog_path,
        selected_payload_manifest=selected,
        selected_view_id="unit_test_view",
        code_commit=commit,
        repository_root=Path(__file__).resolve().parents[1],
    )


def test_action_contract_has_fixed_measure_and_rejects_inactive_values():
    contract = ActionSpaceContract(4, (True, False, True, False))
    assert contract.rank == 2
    assert contract_hash(contract) == contract_hash(contract.to_dict())
    contract.validate_action(torch.tensor([[0.1, 0.0, -0.2, 0.0]]))
    with pytest.raises(ValueError, match="inactive"):
        contract.validate_action(torch.tensor([[0.1, 0.01, -0.2, 0.0]]))
    with pytest.raises(ValueError, match="inactive"):
        contract.validate_action(torch.tensor([[0.1, 1e-9, -0.2, 0.0]]))


def test_squashed_gaussian_round_trip_and_inactive_dirac():
    contract = ActionSpaceContract(3, (True, False, True), pre_squash_sigma=0.25)
    mean = torch.tensor([[0.2, 9.0, -0.1]], requires_grad=True)
    distribution = SquashedGaussian(mean, contract)
    action, pre_squash = distribution.rsample(torch.zeros_like(mean))
    assert action[0, 1].item() == 0.0
    selected = pre_squash[..., torch.tensor(contract.active_mask, dtype=torch.bool)]
    jacobian = torch.log(torch.tensor(contract.coefficient_bound)) + torch.log1p(-torch.tanh(selected).square())
    torch.testing.assert_close(distribution.log_prob(action), distribution.log_prob_u(pre_squash) - jacobian.sum(dim=-1))
    distribution.log_prob(action).sum().backward()
    assert torch.isfinite(mean.grad).all()
    with pytest.raises(ValueError, match="inactive pre_squash"):
        distribution.log_prob_u(torch.tensor([[0.0, 99.0, 0.0]]))


def test_transformed_gaussian_kl_ignores_inactive_slots():
    contract = ActionSpaceContract(3, (True, False, True))
    first = torch.tensor([[0.1, 100.0, 0.3]])
    second = torch.tensor([[0.0, -100.0, 0.1]])
    expected = 0.5 * ((torch.tensor([0.1, 0.2]) / 0.25) ** 2).sum()
    torch.testing.assert_close(transformed_gaussian_kl(first, second, contract).squeeze(), expected)
    torch.testing.assert_close(
        transformed_gaussian_kl(first, second, contract, reduction="mean").squeeze(),
        expected / 2,
    )
    with pytest.raises(ValueError, match="finite"):
        transformed_gaussian_kl(torch.tensor([[float("nan"), 0.0, 0.0]]), second, contract)


def test_native_transition_loss_detaches_teacher():
    predicted = torch.zeros(2, 4, requires_grad=True)
    target = torch.ones(2, 4, requires_grad=True)
    nominal = torch.ones(2, 4)
    loss = normalized_transition_loss(predicted, target, nominal)
    loss.backward()
    assert predicted.grad is not None
    assert target.grad is not None


def test_opd_target_is_detached_and_dpo_has_gradient():
    contract = ActionSpaceContract(2, (True, True))
    predicted = torch.zeros(2, 4, requires_grad=True)
    target = torch.ones(2, 4, requires_grad=True)
    nominal = torch.ones(2, 4)
    mean = torch.zeros(2, 2, requires_grad=True)
    reference = torch.zeros(2, 2)
    loss = opd_loss(predicted, target, nominal, mean, reference, contract)
    loss.backward()
    assert target.grad is None
    chosen_mean = torch.zeros(2, 2, requires_grad=True)
    rejected_mean = torch.ones(2, 2)
    actions = torch.tensor([[0.1, -0.1], [0.2, -0.2]])
    dpo = dpo_loss(actions, -actions, chosen_mean, rejected_mean, torch.zeros_like(chosen_mean), torch.zeros_like(rejected_mean), contract)
    dpo.backward()
    assert torch.isfinite(chosen_mean.grad).all()


def test_search_distill_weights_are_applied_per_sample():
    contract = ActionSpaceContract(2, (True, True))
    predicted = torch.zeros(2, 2, requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [10.0, 0.0]])
    nominal = torch.ones(2, 2)
    mean = torch.zeros(2, 2, requires_grad=True)
    reference = torch.zeros(2, 2)
    per_sample = normalized_transition_losses(predicted, target, nominal)
    loss = search_distill_loss(
        predicted, target, nominal, mean, reference, contract,
        chosen_weight=torch.tensor([1.0, 0.0]),
    )
    expected = per_sample[0] + 0.10 * (mean.square().mean())
    torch.testing.assert_close(loss, expected)


def test_dpo_uses_one_trajectory_margin_and_freezes_reference():
    contract = ActionSpaceContract(2, (True, True))
    chosen_mean = torch.zeros(1, 2, 2, requires_grad=True)
    rejected_mean = torch.zeros(1, 2, 2, requires_grad=True)
    reference_chosen = torch.zeros(1, 2, 2, requires_grad=True)
    reference_rejected = torch.zeros(1, 2, 2, requires_grad=True)
    chosen = torch.full((1, 2, 2), 0.1)
    rejected = torch.full((1, 2, 2), -0.1)
    loss = dpo_loss(
        chosen, rejected, chosen_mean, rejected_mean,
        reference_chosen, reference_rejected, contract,
    )
    loss.backward()
    assert reference_chosen.grad is None
    assert reference_rejected.grad is None
    assert torch.isfinite(chosen_mean.grad).all()
    assert torch.isfinite(rejected_mean.grad).all()
    tied = dpo_loss(
        chosen, rejected, chosen_mean.detach(), rejected_mean.detach(),
        reference_chosen.detach(), reference_rejected.detach(), contract,
        tie_mask=torch.tensor([True]),
    )
    assert tied.item() == 0.0


def test_dpo_flattens_all_decision_axes_into_one_trajectory_margin():
    contract = ActionSpaceContract(2, (True, True))
    chosen_mean = torch.zeros(1, 2, 3, 2, requires_grad=True)
    rejected_mean = torch.zeros(1, 2, 3, 2, requires_grad=True)
    reference_chosen = torch.zeros_like(chosen_mean)
    reference_rejected = torch.zeros_like(rejected_mean)
    chosen = torch.full_like(chosen_mean, 0.1)
    rejected = torch.full_like(rejected_mean, -0.1)
    loss = dpo_loss(
        chosen,
        rejected,
        chosen_mean,
        rejected_mean,
        reference_chosen,
        reference_rejected,
        contract,
    )
    assert loss.ndim == 0
    loss.backward()
    assert torch.isfinite(chosen_mean.grad).all()
    assert torch.isfinite(rejected_mean.grad).all()


def test_dpo_treats_sampled_actions_as_detached_data():
    contract = ActionSpaceContract(2, (True, True))
    chosen = torch.full((1, 2), 0.1, requires_grad=True)
    rejected = torch.full((1, 2), -0.1, requires_grad=True)
    chosen_mean = torch.zeros(1, 2, requires_grad=True)
    rejected_mean = torch.zeros(1, 2, requires_grad=True)
    loss = dpo_loss(
        chosen,
        rejected,
        chosen_mean,
        rejected_mean,
        torch.zeros_like(chosen_mean),
        torch.zeros_like(rejected_mean),
        contract,
    )
    loss.backward()
    assert chosen.grad is None or torch.equal(chosen.grad, torch.zeros_like(chosen))
    assert rejected.grad is None or torch.equal(rejected.grad, torch.zeros_like(rejected))


def test_rl_uses_one_ratio_per_decision():
    contract = ActionSpaceContract(2, (True, True))
    actions = torch.zeros(3, 2)
    behavior = torch.zeros(3, 2)
    policy = torch.tensor([[0.0, 0.0], [0.3, 0.0], [-0.3, 0.0]], requires_grad=True)
    reference = torch.zeros(3, 2)
    advantages = torch.tensor([1.0, -1.0, 0.0])
    loss, ratio, kl = per_decision_rl_loss(actions, behavior, policy, reference, advantages, contract)
    assert ratio.shape == (3,)
    assert kl.shape == (3,)
    loss.backward()
    assert torch.isfinite(policy.grad).all()


def test_rl_masks_nonfinite_density_rows():
    contract = ActionSpaceContract(2, (True, True))
    actions = torch.zeros(2, 2)
    behavior = torch.zeros(2, 2)
    policy = torch.tensor([[0.0, 0.0], [1e20, 0.0]], requires_grad=True)
    reference = torch.zeros(2, 2)
    advantages = torch.ones(2)
    loss, ratio, kl = per_decision_rl_loss(
        actions, behavior, policy, reference, advantages, contract
    )
    assert torch.isfinite(loss)
    assert ratio[1].item() == 0.0
    assert kl[1].item() == 0.0


def test_transition_and_dpo_fail_closed_on_invalid_or_overflow_rows():
    nominal = torch.ones(2, 2)
    losses = normalized_transition_losses(
        torch.tensor([[0.0, 0.0], [1e20, 0.0]]),
        torch.zeros(2, 2), nominal,
    )
    assert losses[0].item() == 0.0
    assert losses[1].item() == 0.0
    contract = ActionSpaceContract(2, (True, True))
    chosen_mean = torch.tensor([[1e20, 0.0]], requires_grad=True)
    rejected_mean = torch.tensor([[-1e20, 0.0]])
    loss = dpo_loss(
        torch.zeros(1, 2), torch.zeros(1, 2), chosen_mean, rejected_mean,
        torch.zeros(1, 2), torch.zeros(1, 2), contract,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(chosen_mean.grad).all()


def test_rl_invalid_action_is_zero_weight():
    contract = ActionSpaceContract(2, (True, True))
    actions = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    policy = torch.zeros(2, 2, requires_grad=True)
    loss, ratio, kl = per_decision_rl_loss(
        actions, torch.zeros(2, 2), policy, torch.zeros(2, 2),
        torch.ones(2), contract,
    )
    assert torch.isfinite(loss)
    assert ratio[1].item() == 0.0
    assert kl[1].item() == 0.0


def test_rl_masks_nonfinite_reference_and_advantage_rows():
    contract = ActionSpaceContract(2, (True, True))
    actions = torch.zeros(2, 2)
    policy = torch.zeros(2, 2, requires_grad=True)
    reference = torch.tensor([[0.0, 0.0], [float("nan"), 0.0]])
    advantages = torch.tensor([1.0, 3.0])
    loss, ratio, kl = per_decision_rl_loss(
        actions,
        torch.zeros_like(actions),
        policy,
        reference,
        advantages,
        contract,
        kl_weight=0.0,
    )
    assert torch.isfinite(loss)
    assert ratio.tolist() == [1.0, 0.0]
    assert kl.tolist() == [0.0, 0.0]
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    loss.backward()
    assert torch.isfinite(policy.grad).all()


def test_rl_all_invalid_policy_returns_finite_zero_loss():
    contract = ActionSpaceContract(2, (True, True))
    policy = torch.full((2, 2), float("nan"), requires_grad=True)
    loss, ratio, kl = per_decision_rl_loss(
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        policy,
        torch.zeros(2, 2),
        torch.ones(2),
        contract,
    )
    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()


def test_shared_prefix_is_replayed_once():
    calls = []

    def step(state, action, index):
        calls.append((state, action, index))
        next_state = state + 1
        if action is None:
            return next_state
        transition = Transition(state, action, next_state, torch.tensor([1.0]), torch.tensor([2.0]), step_index=index)
        return next_state, transition

    trajectories = replay_shared_prefix(0, 2, step, {"plus": [torch.tensor([1.0])], "minus": [torch.tensor([-1.0])]})
    assert len([call for call in calls if call[1] is None]) == 2
    assert [item.step_index for item in trajectories["plus"].transitions] == [2]
    assert [item.step_index for item in trajectories["minus"].transitions] == [2]


def test_shared_prefix_preserves_tuple_valued_state_in_prefix_and_branches():
    calls = []

    def step(state, action, index):
        calls.append((state, action, index))
        return (state[0] + 1, state[1] + 1)

    replay_shared_prefix(
        (0, 10),
        1,
        step,
        {"left": [None], "right": [None]},
    )

    assert calls == [
        ((0, 10), None, 0),
        ((1, 11), None, 1),
        ((1, 11), None, 1),
    ]


def test_shared_prefix_branches_do_not_alias_mutable_state():
    starts = []

    def step(state, action, index):
        if action is None:
            state["value"] += 1
            return state
        starts.append(state["value"])
        state["value"] += int(action.item())
        transition = Transition(state["value"], action, state["value"], torch.ones(1), torch.ones(1), step_index=index)
        return state, transition

    replay_shared_prefix(
        {"value": 0}, 1, step,
        {"plus": [torch.tensor([1])], "minus": [torch.tensor([0])]}
    )
    assert starts == [1, 1]


def test_shared_prefix_requires_clone_hook_for_uncloneable_state():
    class Uncloneable:
        def __deepcopy__(self, memo):
            raise RuntimeError("no copy")

    with pytest.raises(TypeError, match="state_clone_fn"):
        replay_shared_prefix(Uncloneable(), 0, lambda state, action, index: state, {"a": []})


def test_query_ledger_conservatively_charges_unfinished_reservation(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 2}, run_contract="contract-a")
    reservation = ledger.reserve(
        "reward_forward", 1, metadata=_ledger_metadata()
    )
    assert ledger.remaining("reward_forward") == 1
    assert ledger.summary()["unfinished_reservations"] == 1
    with pytest.raises(ValueError, match="does not match"):
        ledger.receipt(
            type(reservation)(reservation.reservation_id, "reward_forward", 99, reservation.metadata),
            result=_ledger_result(), success=False,
        )
    ledger.receipt(reservation, result=_ledger_result(), success=True)
    assert ledger.summary()["unfinished_reservations"] == 0
    with pytest.raises(RuntimeError, match="budget exceeded"):
        ledger.reserve("reward_forward", 2)
    assert len(path.read_text().splitlines()) == 2
    assert all(json.loads(line)["run_contract"] == "contract-a" for line in path.read_text().splitlines())


def test_query_ledger_rejects_tampering(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 2}, run_contract="contract-a")
    ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    row = json.loads(path.read_text())
    row["amount"] = 2
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="hash"):
        ledger.remaining("reward_forward")


def test_query_ledger_requires_provenance_and_detects_deleted_pair(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 4}, run_contract="contract-a")
    with pytest.raises(ValueError, match="provenance"):
        ledger.reserve("reward_forward", 1, metadata={"prompt": "p"})
    first = ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    ledger.receipt(first, result=_ledger_result(), success=True)
    second = ledger.reserve(
        "reward_forward", 1,
        metadata={**_ledger_metadata(), "prompt": "prompt-1", "step": 1},
    )
    ledger.receipt(second, result=_ledger_result(), success=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[2:]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="seal|sequence|chain"):
        ledger.summary()


def test_query_ledger_binds_budget_and_strictness_on_reopen(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 1}, run_contract="contract-a")
    ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    with pytest.raises(RuntimeError, match="contract|budget"):
        QueryLedger(path, {"reward_forward": 2}, run_contract="contract-a").summary()
    with pytest.raises(RuntimeError, match="contract|strict"):
        QueryLedger(path, {"reward_forward": 1}, run_contract="contract-a", strict_provenance=False).summary()


def test_query_ledger_rejects_deleted_primary_file_with_surviving_seal(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 2}, run_contract="contract-a")
    ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    path.unlink()
    with pytest.raises(RuntimeError, match="seal|ledger"):
        ledger.summary()


def test_query_ledger_recovers_after_append_before_seal_crash(tmp_path: Path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 2}, run_contract="contract-a")
    original_replace = Path.replace
    failed = {"value": False}

    def fail_seal_replace(source, destination):
        if destination == ledger.seal_path and not failed["value"]:
            failed["value"] = True
            raise OSError("simulated crash before seal publication")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_seal_replace)
    with pytest.raises(OSError, match="simulated"):
        ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    monkeypatch.setattr(Path, "replace", original_replace)
    summary = ledger.summary()
    assert summary["reserved"]["reward_forward"] == 1
    assert summary["remaining"]["reward_forward"] == 1


def test_query_ledger_computes_and_checks_result_hash(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 1}, run_contract="contract-a")
    reservation = ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    result = _ledger_result()
    ledger.receipt(reservation, result=result, success=True)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    assert len(row["result"]["result_hash"]) == 64
    assert row["result"]["result_hash"] == hashlib.sha256(
        json.dumps(
            {key: value for key, value in row["result"].items() if key != "result_hash"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode()
    ).hexdigest()


def test_training_probe_is_read_only_for_missing_ledger(tmp_path: Path, capsys):
    ledger_path = tmp_path / "nested" / "probe.jsonl"
    assert training_probe(["--ledger", str(ledger_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ledger"] == {
        "path": str(ledger_path),
        "exists": False,
        "read_only": True,
    }
    assert not ledger_path.exists()
    assert not ledger_path.parent.exists()


def test_checkpoint_round_trip_rejects_contract_drift(tmp_path: Path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    contract = {"mask_hash": "abc", "rank": 2}
    path = tmp_path / "checkpoint.pt"
    info = save_checkpoint(path, model=model, optimizer=optimizer, step=3, contract=contract)
    assert info["sha256"]
    restored = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(path, model=restored, optimizer=restored_optimizer, expected_contract=contract)
    assert payload["step"] == 3
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path, model=restored, expected_contract={"mask_hash": "different"})


def test_checkpoint_rejects_fractional_step(tmp_path: Path):
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="non-negative integer"):
        save_checkpoint(
            tmp_path / "fractional.pt", model=model, step=1.5,
            contract={"schema": "test"},
        )


def test_checkpoint_rejects_boolean_step_on_load(tmp_path: Path):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model=model, step=0, contract={"schema": "test"})
    payload = torch.load(path, weights_only=False)
    payload["step"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="step"):
        load_checkpoint(path, model=torch.nn.Linear(1, 1), restore_rng=False)


def test_checkpoint_requires_complete_rng_state():
    state = capture_rng_state()
    incomplete = dict(state)
    incomplete.pop("python")
    with pytest.raises(ValueError, match="RNG"):
        restore_rng_state(incomplete)
    incomplete = dict(state)
    incomplete.pop("numpy")
    with pytest.raises(ValueError, match="RNG"):
        restore_rng_state(incomplete)


def test_checkpoint_rejects_shape_mismatch_without_mutating_model(tmp_path: Path):
    source = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    path = tmp_path / "corrupt.pt"
    save_checkpoint(path, model=source, step=0, contract={"schema": "test"})
    payload = torch.load(path, weights_only=False)
    key = "0.weight"
    payload["model"][key] = payload["model"][key][:-1]
    torch.save(payload, path)
    target = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    before = {name: value.detach().clone() for name, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="shape|state"):
        load_checkpoint(path, model=target, restore_rng=False)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_checkpoint_resume_defaults_to_trainer_contract_and_requires_optimizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    authorization = _training_authorization(tmp_path, monkeypatch)
    trainer = RendererTrainer(
        model, optimizer, contract={"mask_hash": "A"}, authorization=authorization
    )
    path = tmp_path / "model.pt"
    trainer.save(str(path))
    other = torch.nn.Linear(1, 1)
    other_optimizer = torch.optim.AdamW(other.parameters(), lr=1e-3)
    other_trainer = RendererTrainer(
        other, other_optimizer, contract={"mask_hash": "B"}, authorization=authorization
    )
    with pytest.raises(ValueError, match="does not match"):
        other_trainer.load(str(path))
    with pytest.raises(ValueError, match="trainer contract"):
        other_trainer.load(str(path), expected_contract={"mask_hash": "A"})
    with pytest.raises(ValueError, match="trainer contract"):
        load_checkpoint(
            path,
            model=other,
            optimizer=other_optimizer,
            trainer=other_trainer,
            expected_contract={"mask_hash": "A"},
            restore_rng=False,
        )
    payload = torch.load(path, weights_only=False)
    payload["optimizer"] = None
    torch.save(payload, path)
    with pytest.raises(ValueError, match="optimizer state"):
        load_checkpoint(path, model=other, optimizer=other_optimizer, restore_rng=False)


def test_trainer_updates_and_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    authorization = _training_authorization(tmp_path, monkeypatch)
    trainer = RendererTrainer(
        model, optimizer, contract={"schema": "test"}, authorization=authorization
    )
    record = trainer.update(lambda: (model(torch.ones(2, 1)) - 1).square().mean())
    assert record.step == 1
    assert torch.isfinite(torch.tensor(record.loss))
    info = trainer.save(str(tmp_path / "model.pt"))
    assert info["step"] == 1
    restored = torch.nn.Linear(1, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-2)
    restored_trainer = RendererTrainer(
        restored,
        restored_optimizer,
        contract={"schema": "test"},
        authorization=authorization,
    )
    restored_trainer.load(str(tmp_path / "model.pt"), expected_contract={"schema": "test"})
    assert restored_trainer.step == 1
    assert set(restored_trainer.ema_state) == set(restored.state_dict())


def test_trainer_update_requires_catalog_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = RendererTrainer(model, optimizer, contract={"schema": "test"})
    with pytest.raises(RuntimeError, match="TrainingAuthorization"):
        trainer.update(lambda: model(torch.ones(1, 1)).square().sum())

    authorized = _training_authorization(tmp_path, monkeypatch)
    trainer = RendererTrainer(
        model, optimizer, contract={"schema": "test"}, authorization=authorized
    )
    image_path = authorized.selected_manifest_path.parent / "payload.bin"
    image_path.write_bytes(b"changed-payload")
    with pytest.raises(RuntimeError, match="payload bytes changed"):
        trainer.update(lambda: model(torch.ones(1, 1)).square().sum())


def test_training_authorization_requires_explicit_repository_root(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="repository_root is required"):
        TrainingAuthorization.load(receipt, repository_root=None)


def test_training_authorization_rejects_manually_constructed_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    authorized = _training_authorization(tmp_path, monkeypatch)
    forged = TrainingAuthorization(
        receipt_path=authorized.receipt_path,
        catalog_manifest_path=authorized.catalog_manifest_path,
        selected_manifest_path=authorized.selected_manifest_path,
        catalog_manifest_sha256=authorized.catalog_manifest_sha256,
        selected_manifest_sha256=authorized.selected_manifest_sha256,
        catalog_release_id=authorized.catalog_release_id,
        selected_view_id=authorized.selected_view_id,
        selected_rows=authorized.selected_rows,
        code_commit=authorized.code_commit,
        repository_root=authorized.repository_root,
    )
    assert forged.is_validated() is False
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    with pytest.raises(TypeError, match="validated receipt"):
        RendererTrainer(
            model, optimizer, contract={"schema": "test"}, authorization=forged
        )


def test_trainer_rejects_authorization_replacement_after_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorized = _training_authorization(tmp_path, monkeypatch)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = RendererTrainer(
        model, optimizer, contract={"schema": "test"}, authorization=authorized
    )
    trainer.authorization = object()
    with pytest.raises(RuntimeError, match="validated TrainingAuthorization"):
        trainer.update(lambda: model(torch.ones(1, 1)).square().sum())


def test_trainer_rejects_training_authorization_subclass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorized = _training_authorization(tmp_path, monkeypatch)

    class ForgedAuthorization(TrainingAuthorization):
        def validate_current(self):
            raise AssertionError("forged validation should never run")

    forged = ForgedAuthorization(
        receipt_path=authorized.receipt_path,
        catalog_manifest_path=authorized.catalog_manifest_path,
        selected_manifest_path=authorized.selected_manifest_path,
        catalog_manifest_sha256=authorized.catalog_manifest_sha256,
        selected_manifest_sha256=authorized.selected_manifest_sha256,
        catalog_release_id=authorized.catalog_release_id,
        selected_view_id=authorized.selected_view_id,
        selected_rows=authorized.selected_rows,
        code_commit=authorized.code_commit,
        repository_root=authorized.repository_root,
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    with pytest.raises(TypeError, match="validated receipt"):
        RendererTrainer(
            model, optimizer, contract={"schema": "test"}, authorization=forged
        )


def test_training_authorization_rechecks_unrelated_catalog_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    authorized = _training_authorization(tmp_path, monkeypatch)
    # The second artifact is unrelated to the selected payload but remains
    # part of the content-addressed catalog contract.
    views = authorized.selected_manifest_path.parent / "training_views.jsonl"
    views.write_text(views.read_text(encoding="utf-8") + " \n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="catalog artifact changed"):
        authorized.validate_current()


def test_target_config_rejects_nonfinite_constants():
    with pytest.raises(ValueError):
        TargetStepConfig(eta_target=float("nan"))

    with pytest.raises(ValueError, match="integer"):
        TargetStepConfig(target_steps=2.5)


def test_reward_targets_mark_zero_gradient_rows_invalid():
    anchor = torch.zeros(2, 3)
    pair = construct_reward_targets(
        anchor,
        lambda value: value,
        lambda clean: clean.square().sum(dim=-1),
        config=TargetStepConfig(target_steps=1),
    )
    assert pair.valid is not None
    assert not bool(pair.valid[0])


def test_reward_targets_are_detached_and_trust_region_bounded():
    anchor = torch.zeros(2, 3, requires_grad=True)
    calls = []
    pair = construct_reward_targets(
        anchor,
        lambda value: value.square(),
        lambda clean: (calls.append(clean) or clean[..., 0].square()),
        config=TargetStepConfig(eta_target=2.0, target_steps=2, trust_radius_u=0.25),
    )
    assert not pair.plus_u.requires_grad
    assert not pair.minus_u.requires_grad
    assert torch.all(torch.linalg.vector_norm(pair.plus_u - pair.anchor_u, dim=-1) <= 0.250001)
    assert torch.all(torch.linalg.vector_norm(pair.minus_u - pair.anchor_u, dim=-1) <= 0.250001)
    assert len(calls) == 4


def test_reward_target_gradient_respects_inactive_action_slots():
    contract = ActionSpaceContract(3, (True, False, True))
    anchor = torch.tensor([[0.2, 0.0, -0.3]])
    pair = construct_reward_targets(
        anchor,
        lambda value: value,
        lambda clean: clean.square().sum(dim=-1),
        contract=contract,
        config=TargetStepConfig(target_steps=1),
    )
    assert pair.gradient_u[0, 1].item() == 0.0
    assert pair.plus_u[0, 1].item() == 0.0
    assert pair.minus_u[0, 1].item() == 0.0
