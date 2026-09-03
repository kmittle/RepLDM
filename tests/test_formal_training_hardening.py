import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from latent_renderer_training.checkpoint import load_checkpoint, save_checkpoint
from latent_renderer_training.artifacts import (
    FILE_MANIFEST_SCHEMA,
    INITIAL_STATE_SCHEMA,
    module_state_sha256,
)
from latent_renderer_training.contracts import ActionSpaceContract, contract_hash
from latent_renderer_training.trainer import RendererTrainer, run_updates
from tests.test_latent_renderer_training import (
    _complete_run_contract,
    _operation_executor,
    _optimizer_context,
    _training_authorization,
)


class _BoundLinear(torch.nn.Linear):
    """Small CPU model exposing the renderer contract checked by the binding."""

    def __init__(self) -> None:
        super().__init__(1, 1)
        self.frame_contract_hash = "f" * 64
        self.calibration_hash = "c" * 64
        self.contract = ActionSpaceContract(1, (True,))


def _formal_parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    authorization = _training_authorization(tmp_path, monkeypatch)
    model = _BoundLinear()
    artifact_dir = tmp_path / "run-artifacts"
    artifact_dir.mkdir()

    def write_bytes(name: str, payload: bytes) -> tuple[Path, str]:
        path = artifact_dir / name
        path.write_bytes(payload)
        return path, hashlib.sha256(payload).hexdigest()

    def write_json(name: str, payload) -> tuple[Path, str]:
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        return write_bytes(name, encoded)

    model_asset, model_asset_hash = write_bytes("model.bin", b"model-asset")
    reward_asset, reward_asset_hash = write_bytes("reward.bin", b"reward-asset")
    initial_checkpoint, initial_checkpoint_hash = write_bytes(
        "initial.pt", b"initial-renderer-checkpoint"
    )
    model_manifest, model_manifest_hash = write_json(
        "model-assets.json",
        {
            "schema": FILE_MANIFEST_SCHEMA,
            "files": [
                {
                    "path": str(model_asset),
                    "sha256": model_asset_hash,
                    "bytes": model_asset.stat().st_size,
                }
            ],
        },
    )
    reward_manifest, reward_manifest_hash = write_json(
        "reward-assets.json",
        {
            "schema": FILE_MANIFEST_SCHEMA,
            "files": [
                {
                    "path": str(reward_asset),
                    "sha256": reward_asset_hash,
                    "bytes": reward_asset.stat().st_size,
                }
            ],
        },
    )
    action_hash = contract_hash(model.contract)
    prompt_manifest, prompt_manifest_hash = write_json(
        "prompts.json", {"schema": "test.prompts.v1", "prompts": ["prompt"]}
    )
    reward_config, reward_config_hash = write_json(
        "reward.json", {"schema": "test.reward.v1", "name": "reward"}
    )
    calibration, calibration_artifact_hash = write_json(
        "calibration.json", {"calibration_hash": model.calibration_hash}
    )
    frame_contract, frame_artifact_hash = write_json(
        "frame.json",
        {
            "renderer_frame_contract_hash": model.frame_contract_hash,
            "action_contract_hash": action_hash,
        },
    )
    provider_hash = "3" * 64
    provider_config, provider_config_hash = write_json(
        "basis-provider.json", {"basis_provider_contract_hash": provider_hash}
    )
    initial_state_hash = module_state_sha256(model)
    initial_manifest, initial_manifest_hash = write_json(
        "initial-state.json",
        {
            "schema": INITIAL_STATE_SCHEMA,
            "renderer_state_sha256": initial_state_hash,
            "checkpoint": {
                "path": str(initial_checkpoint),
                "sha256": initial_checkpoint_hash,
                "bytes": initial_checkpoint.stat().st_size,
            },
        },
    )

    def descriptor(path: Path, sha256: str) -> dict[str, str]:
        return {"path": str(path), "sha256": sha256}

    contract = _complete_run_contract(
        authorization,
        data_manifest_sha256=authorization.selected_manifest_sha256,
        prompt_manifest_sha256=prompt_manifest_hash,
        renderer_frame_contract_hash=model.frame_contract_hash,
        renderer_frame_contract_artifact_sha256=frame_artifact_hash,
        calibration_hash=model.calibration_hash,
        calibration_artifact_sha256=calibration_artifact_hash,
        action_contract_hash=action_hash,
        basis_provider_contract_hash=provider_hash,
        basis_provider_config_sha256=provider_config_hash,
        reward_config_sha256=reward_config_hash,
        reward_asset_manifest_sha256=reward_manifest_hash,
        model_asset_manifest_sha256=model_manifest_hash,
        initial_renderer_state_sha256=initial_state_hash,
        initial_renderer_state_manifest_sha256=initial_manifest_hash,
    )
    contract["artifacts"].update(
        {
            "data_manifest": descriptor(
                authorization.selected_manifest_path,
                authorization.selected_manifest_sha256,
            ),
            "prompt_manifest": descriptor(prompt_manifest, prompt_manifest_hash),
            "reward_config": descriptor(reward_config, reward_config_hash),
            "reward_assets_manifest": descriptor(
                reward_manifest, reward_manifest_hash
            ),
            "model_assets_manifest": descriptor(model_manifest, model_manifest_hash),
            "basis_provider_config": descriptor(provider_config, provider_config_hash),
            "calibration": descriptor(calibration, calibration_artifact_hash),
            "renderer_frame_contract": descriptor(
                frame_contract, frame_artifact_hash
            ),
            "initial_renderer_state": descriptor(
                initial_manifest, initial_manifest_hash
            ),
        }
    )
    binding = authorization.bind_run_contract(contract)
    return authorization, model, contract, binding


def _trainer(model, contract, binding, *, trainer_type=RendererTrainer):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    return trainer_type(
        model,
        optimizer,
        contract=contract,
        authorization_binding=binding,
        operation_executor=_operation_executor(contract, binding),
    )


def test_renderer_trainer_rejects_raw_training_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization, model, contract, _binding = _formal_parts(tmp_path, monkeypatch)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    with pytest.raises(TypeError, match="AuthorizationBinding"):
        RendererTrainer(
            model,
            optimizer,
            contract=contract,
            authorization=authorization,
        )


@pytest.mark.parametrize("scope", ("missing", "external"))
def test_renderer_trainer_rejects_optimizer_outside_exact_model_scope(scope: str):
    model = _BoundLinear()
    if scope == "missing":
        optimizer = torch.optim.SGD([model.weight], lr=1e-2)
    else:
        external = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([*model.parameters(), external], lr=1e-2)

    with pytest.raises(ValueError, match="exactly the trainable renderer"):
        RendererTrainer(model, optimizer, contract={"schema": "cpu-test"})


def test_optimizer_scope_cannot_change_after_trainer_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _authorization, model, contract, binding = _formal_parts(tmp_path, monkeypatch)
    trainer = _trainer(model, contract, binding)
    trainer.optimizer.param_groups[0]["params"].append(
        torch.nn.Parameter(torch.ones(()))
    )
    called = []

    with pytest.raises(RuntimeError, match="optimizer parameter scope changed"):
        trainer.update(
            lambda: called.append(True)
            or model(torch.ones(1, 1)).square().mean(),
            operation_context=_optimizer_context(trainer),
        )
    assert called == []


def test_round_two_loads_ema_and_resets_adamw_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _authorization, model, contract, binding = _formal_parts(tmp_path, monkeypatch)
    trainer = _trainer(model, contract, binding)
    trainer.step = 32
    with torch.no_grad():
        for value in trainer.ema_state.values():
            if value.is_floating_point():
                value.add_(0.5)

    model(torch.ones(1, 1)).square().mean().backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    assert trainer.optimizer.state
    expected_ema = {
        name: value.detach().clone() for name, value in trainer.ema_state.items()
    }
    fresh = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    transition = trainer.start_round_two_from_ema(fresh)

    assert trainer.optimizer is fresh
    assert trainer.optimizer.state == {}
    assert transition["source"] == "round_1_ema"
    assert transition["discarded_optimizer_state_entries"] > 0
    assert transition["new_optimizer_state_entries"] == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_ema[name])
    with pytest.raises(RuntimeError, match="already occurred"):
        trainer.start_round_two_from_ema(
            torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        )


def test_trainer_binding_removal_or_replacement_blocks_update_save_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _authorization, model, contract, binding = _formal_parts(tmp_path, monkeypatch)
    source = _trainer(model, contract, binding)
    checkpoint = tmp_path / "bound.pt"
    source.save(str(checkpoint))

    called = []
    source.authorization_binding = None
    with pytest.raises(RuntimeError, match="cleared or replaced"):
        source.update(
            lambda: called.append("loss")
            or model(torch.ones(1, 1)).square().mean()
        )
    with pytest.raises(RuntimeError, match="cleared or replaced"):
        source.save(str(tmp_path / "must-not-exist.pt"))
    assert called == []
    assert not (tmp_path / "must-not-exist.pt").exists()

    restored_model = _BoundLinear()
    restored_model.load_state_dict(model.state_dict())
    restored = _trainer(restored_model, contract, binding)
    restored.authorization_binding = copy.copy(binding)
    with pytest.raises(RuntimeError, match="cleared or replaced"):
        restored.load(str(checkpoint), restore_rng=False)


def test_loss_callback_cannot_revoke_binding_before_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _authorization, model, contract, binding = _formal_parts(tmp_path, monkeypatch)
    trainer = _trainer(model, contract, binding)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def revoke_then_compute():
        trainer.authorization_binding = None
        return model(torch.ones(1, 1)).square().mean()

    with pytest.raises(RuntimeError, match="cleared or replaced"):
        trainer.update(
            revoke_then_compute,
            operation_context=_optimizer_context(trainer),
        )
    assert trainer.step == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_trainer_rechecks_selected_data_after_optimizer_step_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization, model, contract, binding = _formal_parts(tmp_path, monkeypatch)
    trainer = _trainer(model, contract, binding)
    selected = json.loads(
        authorization.selected_payload_path.read_text(encoding="utf-8").splitlines()[0]
    )
    image_path = Path(selected["image_path"])
    before_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    before_ema = {
        name: value.detach().clone() for name, value in trainer.ema_state.items()
    }
    original_step = trainer.optimizer.step

    def mutate_after_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        image_path.write_bytes(b"changed-after-optimizer-step")
        return result

    monkeypatch.setattr(trainer.optimizer, "step", mutate_after_step)
    context = _optimizer_context(trainer)
    with pytest.raises(RuntimeError, match="selected raw image .* changed"):
        trainer.update(
            lambda: model(torch.ones(1, 1)).square().mean(),
            operation_context=context,
        )

    assert trainer.step == 0
    assert trainer.optimizer.state_dict()["state"] == {}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_model[name])
        torch.testing.assert_close(trainer.ema_state[name], before_ema[name])


def test_formal_checkpoint_requires_binding_for_save_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _authorization, _renderer, contract, binding = _formal_parts(tmp_path, monkeypatch)
    model = torch.nn.Linear(1, 1)
    rejected = tmp_path / "unbound-formal.pt"

    with pytest.raises(RuntimeError, match="authorization binding"):
        save_checkpoint(rejected, model=model, step=0, contract=contract)
    assert not rejected.exists()

    checkpoint = tmp_path / "formal.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        step=0,
        contract=contract,
        authorization_binding=binding,
    )
    with pytest.raises(RuntimeError, match="authorization binding"):
        load_checkpoint(
            checkpoint,
            model=torch.nn.Linear(1, 1),
            expected_contract=contract,
            restore_rng=False,
        )
    with pytest.raises(RuntimeError, match="authorization binding"):
        load_checkpoint(
            checkpoint,
            model=torch.nn.Linear(1, 1),
            restore_rng=False,
        )

    payload = load_checkpoint(
        checkpoint,
        model=torch.nn.Linear(1, 1),
        expected_contract=contract,
        authorization_binding=binding,
        restore_rng=False,
    )
    assert payload["extra"]["authorization_binding"] == binding.provenance()


def test_checkpoint_provenance_is_reserved_and_cannot_be_downgraded(
    tmp_path: Path,
):
    contract = {"schema": "cpu-test"}
    injected = tmp_path / "nested" / "injected.pt"
    with pytest.raises(ValueError, match="reserved"):
        save_checkpoint(
            injected,
            model=torch.nn.Linear(1, 1),
            step=0,
            contract=contract,
            extra={"authorization_binding": {"schema": "forged"}},
        )
    assert not injected.parent.exists()

    checkpoint = tmp_path / "tampered.pt"
    save_checkpoint(
        checkpoint,
        model=torch.nn.Linear(1, 1),
        step=0,
        contract=contract,
    )
    payload = torch.load(checkpoint, weights_only=False)
    payload["extra"]["authorization_binding"] = {"schema": "forged"}
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="records authorization provenance"):
        load_checkpoint(
            checkpoint,
            model=torch.nn.Linear(1, 1),
            expected_contract=contract,
            restore_rng=False,
        )


def test_informal_checkpoint_round_trip_remains_available(tmp_path: Path):
    contract = {"schema": "cpu-test", "case": "round-trip"}
    checkpoint = tmp_path / "informal.pt"
    save_checkpoint(
        checkpoint,
        model=torch.nn.Linear(1, 1),
        step=2,
        contract=contract,
    )
    payload = load_checkpoint(
        checkpoint,
        model=torch.nn.Linear(1, 1),
        expected_contract=contract,
        restore_rng=False,
    )
    assert payload["step"] == 2


def test_checkpoint_publication_is_exclusive_and_preserves_existing_bytes(
    tmp_path: Path,
):
    checkpoint = tmp_path / "exclusive.pt"
    model = torch.nn.Linear(1, 1)
    save_checkpoint(
        checkpoint,
        model=model,
        step=0,
        contract={"schema": "cpu-test"},
    )
    original = checkpoint.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to replace"):
        save_checkpoint(
            checkpoint,
            model=model,
            step=1,
            contract={"schema": "cpu-test"},
        )

    assert checkpoint.read_bytes() == original
    assert not tuple(tmp_path.glob(".exclusive.pt.*.tmp"))


def test_checkpoint_io_rejects_symlinked_parent(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    model = torch.nn.Linear(1, 1)

    with pytest.raises(ValueError, match="symbolic links"):
        save_checkpoint(
            linked / "model.pt",
            model=model,
            step=0,
            contract={"schema": "cpu-test"},
        )
    assert not (real / "model.pt").exists()

    checkpoint = real / "model.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        step=0,
        contract={"schema": "cpu-test"},
    )
    with pytest.raises(ValueError, match="non-symlinked"):
        load_checkpoint(
            linked / "model.pt",
            model=torch.nn.Linear(1, 1),
            expected_contract={"schema": "cpu-test"},
            restore_rng=False,
        )


def test_run_updates_preflights_before_every_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _authorization, model, contract, binding = _formal_parts(tmp_path, monkeypatch)

    class _InvalidatingTrainer(RendererTrainer):
        def update(self, loss_fn, *, operation_context=None):
            record = super().update(
                loss_fn, operation_context=operation_context
            )
            self.authorization_binding = None
            return record

    trainer = _trainer(
        model,
        contract,
        binding,
        trainer_type=_InvalidatingTrainer,
    )

    class _SideEffectingIterator:
        def __init__(self) -> None:
            self.next_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.next_calls += 1
            if self.next_calls <= 2:
                return (
                    lambda: model(torch.ones(1, 1)).square().mean(),
                    _optimizer_context(trainer),
                )
            raise StopIteration

    losses = _SideEffectingIterator()
    with pytest.raises(RuntimeError, match="cleared or replaced"):
        run_updates(trainer, losses)
    assert trainer.step == 1
    assert losses.next_calls == 1
