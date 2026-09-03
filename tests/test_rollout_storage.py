import json
from pathlib import Path

import pytest
import torch

from AttentionGuidance.latent_renderer import RendererCondition, RendererObservation
from latent_renderer_training.methods.common import opd_anchor_state_sha256
from latent_renderer_training.operations import operation_output_sha256
from latent_renderer_training.renderer import (
    CALIBRATION_STATE_COUNT,
    EulerNativeFrameV1,
    FrameCalibration,
    tensor_sha256,
)
from latent_renderer_training.rollout import (
    BranchTrajectory,
    DecisionProposal,
    RolloutCollection,
    Transition,
)
from latent_renderer_training.storage import AtomicRolloutStore, CheckpointProvenance
from tests.test_latent_renderer_training import (
    _complete_run_contract,
    _training_authorization,
)


def _collection() -> RolloutCollection:
    proposals = []
    for index in (8, 24, 40):
        value = torch.zeros(1, 6)
        proposals.append(
            DecisionProposal(
                step_index=index,
                mean=value.clone(),
                noise=value.clone(),
                plus_action=value.clone(),
                minus_action=value.clone(),
                anchor_action=value.clone(),
                plus_mean=value.clone(),
                minus_mean=value.clone(),
                anchor_mean=value.clone(),
                plus_pre_squash=value.clone(),
                minus_pre_squash=value.clone(),
                anchor_pre_squash=value.clone(),
            )
        )
    return RolloutCollection(
        branches={
            branch: BranchTrajectory(branch)
            for branch in ("plus", "minus", "anchor")
        },
        proposals=proposals,
        prefix_steps=8,
        total_steps=50,
        terminal_states={
            branch: torch.full((1,), offset)
            for offset, branch in enumerate(("plus", "minus", "anchor"))
        },
    )


def _formal_opd_payload() -> tuple[
    RolloutCollection,
    list[dict[str, object]],
    dict[str, tuple[dict[str, object], dict[str, object]]],
]:
    """Build a small valid formal OPD payload for provenance regression tests."""
    calibration = FrameCalibration(
        (True,) * 6,
        CALIBRATION_STATE_COUNT,
        (1.0,) * 6,
        (1.0,) * 6,
        manifest_sha256="a" * 64,
        source_sha256="b" * 64,
        state_provenance_sha256="c" * 64,
    )
    renderer = EulerNativeFrameV1(calibration=calibration, latent_channels=2)
    collection = _collection()
    records: list[dict[str, object]] = []
    receipts: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    teacher_state_hash = "d" * 64
    teacher_file_hash = "e" * 64
    for decision_index in (8, 24, 40):
        sample = torch.full((1, 2, 2, 2), 0.25 + decision_index / 1000.0)
        clean = sample * 0.9
        native_update = torch.full_like(sample, 0.1)
        bases = torch.arange(48, dtype=torch.float32).reshape(1, 6, 2, 2, 2) / 50.0
        observation = RendererObservation(
            latents_before_step=sample,
            pred_original_sample=clean,
            scheduler_update=native_update,
            step_index=decision_index,
            timestep=torch.tensor([float(decision_index)]),
            normalized_timestep=torch.tensor([decision_index / 50.0]),
        )
        condition = RendererCondition(
            bases=bases,
            prompt_embedding=torch.zeros(1, renderer.prompt_dim),
            state_features=torch.zeros(1, renderer.state_dim),
        )
        frame = renderer.prepare_state(
            observation,
            condition,
            sigma_from=1.0,
            sigma_to=0.5,
            native_model_output=torch.zeros_like(sample),
            prediction_type="epsilon",
        )
        transition = Transition(
            state=frame,
            action=torch.zeros(1, 6),
            next_state={},
            native_transition=native_update,
            rendered_transition=native_update.clone(),
            step_index=decision_index,
            pre_squash=torch.zeros(1, 6),
            behavior_mean=torch.zeros(1, 6),
        )
        collection.branches["anchor"].transitions.append(transition)
        target = torch.full_like(sample, 0.01 * decision_index)
        target_hash = tensor_sha256(target)
        receipt_hash = operation_output_sha256(target)
        reservation_id = f"label-{decision_index}"
        student_state_hash = opd_anchor_state_sha256(transition)
        context = {
            "split": "train",
            "prompt": "source:row-0",
            "seed": 0,
            "step": decision_index,
            "prefix": 8,
            "branch": "teacher",
            "action": {
                "kind": "opd_teacher_transition",
                "decision_index": decision_index,
                "target_sha256": target_hash,
                "raw_target_sha256": target_hash,
                "student_state_sha256": student_state_hash,
                "prompt_id": "source:row-0",
                "prompt": "unit test prompt",
                "split": "train",
                "seed": 0,
            },
            "checkpoint_hash": teacher_state_hash,
            "image_hash": None,
            "cached_parent": None,
        }
        record = {
            "decision_index": decision_index,
            "target_sha256": target_hash,
            "raw_target_sha256": target_hash,
            "target_shape": list(target.shape),
            "target_dtype": str(target.dtype),
            "label_receipt_output_sha256": receipt_hash,
            "frame_valid": True,
            "teacher_checkpoint_sha256": teacher_state_hash,
            "teacher_state_sha256": teacher_state_hash,
            "teacher_checkpoint_file_sha256": teacher_file_hash,
            "reservation_id": reservation_id,
            "operation_context": context,
            "student_state_sha256": student_state_hash,
        }
        reservation_metadata = {
            "code_hash": "f" * 64,
            "data_hash": "1" * 64,
            "checkpoint_hash": teacher_state_hash,
            "method_allocation": {"kind": "label", "amount": 1, "method": "rl"},
            "split": "train",
            "prompt": "source:row-0",
            "seed": 0,
            "step": decision_index,
            "prefix": 8,
            "branch": "teacher",
            "action": context["action"],
        }
        receipt = {
            "kind": "label",
            "amount": 1,
            "success": True,
            "metadata": reservation_metadata,
            "result": {
                "output_hash": receipt_hash,
                "scalar_or_gradient": "teacher_transition",
                "parent_reservation_id": None,
            },
        }
        records.append(record)
        receipts[reservation_id] = (
            {
                "reservation_id": reservation_id,
                "kind": "label",
                "amount": 1,
                "metadata": reservation_metadata,
            },
            receipt,
        )
        collection.opd_teacher_targets = tuple(
            list(collection.opd_teacher_targets or ()) + [target]
        )
        collection.opd_teacher_label_provenance = tuple(
            list(collection.opd_teacher_label_provenance or ()) + [record]
        )
    return collection, records, receipts


def _store_parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(authorization)
    binding = authorization.bind_run_contract(contract)
    checkpoint_dir = Path(contract["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "c0.pt"
    checkpoint.write_bytes(b"frozen-renderer-checkpoint")
    provenance = CheckpointProvenance.capture(
        checkpoint,
        renderer_state_sha256=contract["initial_renderer_state_sha256"],
    )
    return contract, binding, provenance


def test_rollout_store_round_trip_binds_behavior_reference_and_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, binding, checkpoint = _store_parts(tmp_path, monkeypatch)
    store = AtomicRolloutStore(binding)
    manifest = store.save(
        "round-0-block-000",
        _collection(),
        behavior_checkpoint=checkpoint,
        reference_checkpoint=checkpoint,
        metadata={"round": 0, "prompt_ids": ["source:row-0"]},
    )

    assert manifest["run_contract_sha256"] == binding.contract_hash
    assert manifest["behavior_checkpoint"]["sha256"] == checkpoint.sha256
    assert manifest["reference_checkpoint"]["renderer_state_sha256"] == contract[
        "initial_renderer_state_sha256"
    ]
    restored = store.load(
        "round-0-block-000",
        behavior_checkpoint=checkpoint,
        reference_checkpoint=checkpoint,
    )
    assert [proposal.step_index for proposal in restored.proposals] == [8, 24, 40]
    assert set(restored.terminal_states) == {"plus", "minus", "anchor"}
    with pytest.raises(FileExistsError, match="already exists"):
        store.save(
            "round-0-block-000",
            _collection(),
            behavior_checkpoint=checkpoint,
            reference_checkpoint=checkpoint,
            metadata={"round": 0},
        )


def test_formal_opd_payload_rejects_unknown_label_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tests.test_latent_renderer_training as training_fixtures
    import latent_renderer_training.gates as gates_module

    monkeypatch.setattr(
        training_fixtures,
        "_sealed_metric_ledger",
        lambda *_args, **_kwargs: (b'{"fixture":"ledger"}\n', b'{"fixture":"seal"}\n'),
    )
    monkeypatch.setattr(gates_module, "validate_f0_gate", lambda gate, **_kwargs: (gate, ()))
    contract, binding, _checkpoint = _store_parts(tmp_path, monkeypatch)
    store = AtomicRolloutStore(binding)
    collection, records, receipts = _formal_opd_payload()
    monkeypatch.setattr(store, "_successful_opd_label_receipts", lambda: receipts)
    metadata = {
        "schema": "repldm.scored_opd_rollout.v1",
        "teacher_labels": records,
    }

    store._validate_opd_payload(collection, metadata)
    tampered = json.loads(json.dumps(records))
    tampered[0]["reservation_id"] = "label-not-in-this-ledger"
    tampered_metadata = {**metadata, "teacher_labels": tampered}
    tampered_collection = _formal_opd_payload()[0]
    tampered_collection.opd_teacher_targets = collection.opd_teacher_targets
    tampered_collection.opd_teacher_label_provenance = tuple(tampered)
    with pytest.raises(ValueError, match="no successful ledger receipt"):
        store._validate_opd_payload(tampered_collection, tampered_metadata)

    assert not Path(contract["paths"]["rollout_dir"]).exists()


def test_rollout_store_detects_payload_and_checkpoint_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, binding, checkpoint = _store_parts(tmp_path, monkeypatch)
    store = AtomicRolloutStore(binding)
    manifest = store.save(
        "round-0-block-001",
        _collection(),
        behavior_checkpoint=checkpoint,
        reference_checkpoint=checkpoint,
        metadata={"round": 0},
    )
    payload = Path(manifest["payload"]["path"])
    payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="payload bytes"):
        store.load(
            "round-0-block-001",
            behavior_checkpoint=checkpoint,
            reference_checkpoint=checkpoint,
        )

    checkpoint_path = Path(checkpoint.path)
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="checkpoint bytes changed"):
        checkpoint.validate_current()


def test_rollout_store_rejects_wrong_reference_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, binding, checkpoint = _store_parts(tmp_path, monkeypatch)
    wrong = CheckpointProvenance(
        path=checkpoint.path,
        sha256=checkpoint.sha256,
        bytes=checkpoint.bytes,
        renderer_state_sha256="f" * 64,
    )
    store = AtomicRolloutStore(binding)
    with pytest.raises(ValueError, match="frozen initial renderer"):
        store.save(
            "round-0-block-002",
            _collection(),
            behavior_checkpoint=checkpoint,
            reference_checkpoint=wrong,
            metadata={"round": 0},
        )
    assert not Path(contract["paths"]["rollout_dir"]).exists()


def test_rollout_manifest_tampering_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, binding, checkpoint = _store_parts(tmp_path, monkeypatch)
    store = AtomicRolloutStore(binding)
    store.save(
        "round-0-block-003",
        _collection(),
        behavior_checkpoint=checkpoint,
        reference_checkpoint=checkpoint,
        metadata={"round": 0},
    )
    manifest_path = Path(contract["paths"]["rollout_dir"]) / "round-0-block-003.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_contract_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="run_contract_sha256"):
        store.load(
            "round-0-block-003",
            behavior_checkpoint=checkpoint,
            reference_checkpoint=checkpoint,
        )
