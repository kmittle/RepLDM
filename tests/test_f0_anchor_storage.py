from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from AttentionGuidance.latent_renderer import RendererCondition, RendererObservation
from latent_renderer_training.collector import (
    CachedEulerDecision,
    CollectorStats,
    EulerRolloutStep,
    F0AnchorTrace,
)
from latent_renderer_training.f0_metrics import build_f0_metric_action
from latent_renderer_training.ledger import QueryLedger
from latent_renderer_training.methods.f0 import F0ScoredImage, _restore_anchor_score
from latent_renderer_training.methods.runner import PromptRecord, PromptSeedBlock
from latent_renderer_training.operations import (
    LedgeredOperationExecutor,
    OperationContext,
    OperationReceipt,
    operation_output_sha256,
)
from latent_renderer_training.renderer import (
    EulerFrameDiagnostics,
    EulerFrameState,
    tensor_sha256,
)
from latent_renderer_training.rollout import Transition
from latent_renderer_training.sdxl_adapter import (
    SdxlEulerState,
    SdxlPromptConditioning,
)
from latent_renderer_training.storage import (
    AtomicF0AnchorStore,
    F0AnchorScoreRecovery,
    _f0_tensor_relation,
)
from latent_renderer_training.witnesses import (
    TOPIQ_NR_MODEL_ID,
    TOPIQ_NR_PREPROCESS_SHA256,
    TOPIQ_NR_ROLE,
)
from tests.test_latent_renderer_training import (
    _complete_run_contract,
    _training_authorization,
)


CHECKPOINT_HASH = "9" * 64
FRAME_CONTRACT_HASH = "f" * 64
CALIBRATION_HASH = "c" * 64
DECISIONS = (8, 24, 40)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _block(record_id: str = "anchor-prompt") -> PromptSeedBlock:
    return PromptSeedBlock(
        PromptRecord(
            record_id=record_id,
            prompt=f"render {record_id}",
            split="train",
            source="storage-test",
            stratum="synthetic",
            fold=0,
        ),
        2026090101,
    )


def _conditioning(block: PromptSeedBlock) -> SdxlPromptConditioning:
    return SdxlPromptConditioning(
        prompt_ids=(block.record.record_id,),
        prompts=(block.record.prompt,),
        generation_seeds=(block.generation_seed,),
        prompt_embeds=torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        pooled_prompt_embeds=torch.arange(4, dtype=torch.float32).reshape(1, 4),
        add_text_embeds=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        add_time_ids=torch.arange(12, dtype=torch.float32).reshape(2, 6),
        do_classifier_free_guidance=True,
    )


def _action_history(step_index: int) -> tuple[object, ...]:
    count = sum(index < step_index for index in DECISIONS)
    return tuple([([0.0] * 6)] for _ in range(count))


def _state(
    block: PromptSeedBlock,
    conditioning: SdxlPromptConditioning,
    step_index: int,
) -> SdxlEulerState:
    base = torch.linspace(-0.5, 0.5, 8, dtype=torch.float32).reshape(1, 2, 2, 2)
    return SdxlEulerState(
        latents=base + 0.125 * step_index,
        conditioning=conditioning,
        step_index=step_index,
        total_steps=50,
        split=block.record.split,
        branch="anchor",
        prefix=0,
        checkpoint_hash=CHECKPOINT_HASH,
        action_history=_action_history(step_index),
    )


def _step_and_frame(
    current: SdxlEulerState,
    following: SdxlEulerState,
    index: int,
) -> tuple[EulerRolloutStep, EulerFrameState]:
    update = following.latents - current.latents
    clean = current.latents - 0.25
    bases = torch.arange(48, dtype=torch.float32).reshape(1, 6, 2, 2, 2) / 50
    observation = RendererObservation(
        latents_before_step=current.latents.clone(),
        pred_original_sample=clean,
        scheduler_update=update,
        step_index=index,
        timestep=torch.tensor(float(49 - index)),
        normalized_timestep=torch.tensor(index / 49.0),
        pooled_prompt_embeds=current.conditioning.pooled_prompt_embeds.clone(),
    )
    condition = RendererCondition(
        bases=bases,
        prompt_embedding=torch.ones(1, 4),
        state_features=torch.zeros(1, 3),
    )
    native_model_output=torch.full_like(current.latents, 0.01 * (index + 1))
    step = EulerRolloutStep(
        observation=observation,
        condition=condition,
        native_model_output=native_model_output,
        native_prev_sample=following.latents.clone(),
        sigma_from=torch.tensor(float(50 - index)),
        sigma_to=torch.tensor(float(49 - index)),
        prediction_type="epsilon",
        clean_update_gain=torch.tensor(1.0),
        metadata={
            "schema": "repldm.sdxl_euler_observation.v1",
            "schedule_sha256": "e" * 64,
            "prompt_ids": [current.conditioning.prompt_ids[0]],
            "branch": "anchor",
            "step_index": index,
            "physical_unet_forwards": 1,
        },
    )
    diagnostics = EulerFrameDiagnostics(
        valid=torch.tensor([True]),
        active_mask=(True,) * 6,
        eigenvalues=torch.ones(1, 6),
        condition_number=torch.ones(1),
        gram_error=torch.zeros(1),
        angle=torch.zeros(1),
        angle_cap_multiplier=torch.ones(1),
        scheduler_cap_multiplier=torch.ones(1),
        mapped_update_ratio=torch.ones(1),
        gram_hash=("a" * 64,),
        frame_hash=("b" * 64,),
    )
    frame = EulerFrameState(
        clean_latent=clean.clone(),
        sample=current.latents.clone(),
        native_model_output=native_model_output.clone(),
        raw_bases=bases.clone(),
        tangent_bases=bases.clone(),
        mapped_bases=bases.clone(),
        clean_bases=bases.clone(),
        native_update=update.clone(),
        sigma_from=torch.tensor(float(50 - index)),
        sigma_to=torch.tensor(float(49 - index)),
        kappa=torch.ones(1),
        context=torch.ones(1, 4),
        diagnostics=diagnostics,
        prediction_type="epsilon",
    )
    return step, frame


def _trace(block: PromptSeedBlock, contract_hash: str) -> F0AnchorTrace:
    conditioning = _conditioning(block)
    states = [_state(block, conditioning, index) for index in range(51)]
    transitions = []
    caches = {}
    for index in range(50):
        step, frame = _step_and_frame(states[index], states[index + 1], index)
        transition_state = frame if index in DECISIONS else step.observation
        delta = states[index + 1].latents - states[index].latents
        transitions.append(
            Transition(
                state=transition_state,
                action=torch.zeros(1, 6),
                next_state=states[index + 1],
                native_transition=delta,
                rendered_transition=delta.clone(),
                step_index=index,
            )
        )
        if index in DECISIONS:
            caches[index] = CachedEulerDecision(
                decision_index=index,
                state=states[index],
                step=step,
                frame=frame,
                checkpoint_hash=CHECKPOINT_HASH,
                frame_contract_hash=FRAME_CONTRACT_HASH,
                calibration_hash=CALIBRATION_HASH,
                run_contract=contract_hash,
            )
    return F0AnchorTrace(
        branch="anchor",
        initial_state=states[0],
        terminal_state=states[-1],
        cached_decisions=caches,
        transitions=tuple(transitions),
        stats=CollectorStats(50, 50, 3, 47, 50),
        checkpoint_hash=CHECKPOINT_HASH,
    )


def _receipt(
    block: PromptSeedBlock,
    *,
    reservation_id: str,
    kind: str,
    seal: object,
) -> OperationReceipt:
    context = OperationContext(
        split="train",
        prompt=block.record.record_id,
        seed=block.generation_seed,
        step=49,
        prefix=0,
        branch="anchor",
        action={"f0_metric_branch": "anchor"},
        checkpoint_hash=CHECKPOINT_HASH,
        image_hash=None if kind == "vae_decode" else "d" * 64,
        cached_parent=None if kind == "vae_decode" else "e" * 64,
    )
    return OperationReceipt(
        reservation_id=reservation_id,
        kind=kind,
        output_hash="1" * 64,
        image_hash="d" * 64,
        model="SDXL-VAE" if kind == "vae_decode" else "metric",
        role="decoder" if kind == "vae_decode" else "reward",
        preprocess_hash="2" * 64,
        model_config_sha256="3" * 64,
        model_asset_manifest_sha256="4" * 64,
        context=context,
        _executor_seal=seal,
    )


def _score(block: PromptSeedBlock) -> F0ScoredImage:
    seal = object()
    return F0ScoredImage(
        image_sha256="d" * 64,
        reward=torch.tensor(1.234567, dtype=torch.float64),
        topiq_nr=torch.tensor(0.8125, dtype=torch.float16),
        pixel={
            "clipped_fraction": 0.01,
            "mean_saturation": 0.22,
            "contrast": 0.33,
        },
        decode_receipt=_receipt(
            block,
            reservation_id="00000000-0000-0000-0000-000000000001",
            kind="vae_decode",
            seal=seal,
        ),
        reward_receipt=_receipt(
            block,
            reservation_id="00000000-0000-0000-0000-000000000002",
            kind="reward_forward",
            seal=seal,
        ),
        witness_receipt=_receipt(
            block,
            reservation_id="00000000-0000-0000-0000-000000000003",
            kind="reward_forward",
            seal=seal,
        ),
    )


def _parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_id="anchor-prompt"):
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(authorization, method="f0")
    binding = authorization.bind_run_contract(contract)
    block = _block(record_id)
    return authorization, contract, binding, block, _trace(block, binding.contract_hash)


def _rewrite_payload(
    store: AtomicF0AnchorStore,
    block: PromptSeedBlock,
    envelope: dict[str, object],
) -> None:
    manifest_path = store.root / f"{block.stable_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    buffer = io.BytesIO()
    torch.save(envelope, buffer)
    payload_bytes = buffer.getvalue()
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    payload_path = store.root / f"{block.stable_id}-{payload_hash[:16]}.pt"
    payload_path.write_bytes(payload_bytes)
    manifest["payload"] = {
        "path": str(payload_path),
        "sha256": payload_hash,
        "bytes": len(payload_bytes),
    }
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    manifest_path.write_bytes(_canonical(manifest) + b"\n")


def test_f0_anchor_store_round_trip_is_safe_detached_and_content_addressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, block, trace = _parts(tmp_path, monkeypatch)
    store = AtomicF0AnchorStore(binding)
    assert store.load_or_none(block) is None
    assert not store.has(block)

    manifest = store.save(block, trace, score=_score(block))
    payload = Path(manifest["payload"]["path"])
    assert payload.name == (
        f"{block.stable_id}-{manifest['payload']['sha256'][:16]}.pt"
    )
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == manifest["payload"][
        "sha256"
    ]

    original_load = torch.load
    weights_only_calls = []

    def observed_load(source, map_location=None, *, weights_only=False):
        weights_only_calls.append(weights_only)
        return original_load(
            source, map_location=map_location, weights_only=weights_only
        )

    monkeypatch.setattr(torch, "load", observed_load)
    restored = store.load(block)
    assert weights_only_calls == [True]
    assert restored.block == block
    assert isinstance(restored.trace, F0AnchorTrace)
    assert restored.trace.initial_state.latents.data_ptr() != trace.initial_state.latents.data_ptr()
    assert restored.trace.cached_decisions[24].frame.sample.data_ptr() != (
        trace.cached_decisions[24].frame.sample.data_ptr()
    )
    assert restored.score.image_reward.dtype == torch.float32
    assert restored.score.topiq_nr.dtype == torch.float32
    assert restored.score.vae_decode_reservation_id.endswith("1")
    assert restored.score.image_reward_reservation_id.endswith("2")
    assert restored.score.topiq_nr_reservation_id.endswith("3")
    assert store.has(block)
    assert weights_only_calls == [True, True]


def test_f0_anchor_accepts_bounded_fp16_native_scheduler_round_trip() -> None:
    """Native reduced-precision scheduler bytes may round-trip through fp16."""
    analytic = torch.tensor([0.125, -0.5, 1.0], dtype=torch.float32)
    native = analytic.to(torch.float16)
    _f0_tensor_relation(
        analytic,
        native,
        label="synthetic fp16 scheduler round trip",
    )


def test_f0_anchor_rejects_fp16_scheduler_round_trip_outside_tolerance() -> None:
    analytic = torch.tensor([0.125, -0.5, 1.0], dtype=torch.float32)
    forged = analytic.to(torch.float16).to(torch.float32)
    forged[0] += 0.1
    with pytest.raises(ValueError, match="synthetic fp16 scheduler round trip values differ"):
        _f0_tensor_relation(
            analytic,
            forged.to(torch.float16),
            label="synthetic fp16 scheduler round trip",
        )


def test_f0_anchor_payload_contains_no_receipt_or_executor_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, block, trace = _parts(tmp_path, monkeypatch)
    store = AtomicF0AnchorStore(binding)
    manifest = store.save(block, trace, score=_score(block))
    envelope = torch.load(
        manifest["payload"]["path"], map_location="cpu", weights_only=True
    )

    def inspect_value(value: object) -> None:
        assert not isinstance(value, OperationReceipt)
        if isinstance(value, dict):
            assert "_executor_seal" not in value
            for item in value.values():
                inspect_value(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect_value(item)

    inspect_value(envelope)
    assert set(envelope["score_recovery"]) == {
        "image_sha256",
        "image_reward",
        "topiq_nr",
        "pixel_summary",
        "receipt_reservation_ids",
    }


def test_f0_anchor_store_refuses_duplicate_stable_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, block, trace = _parts(tmp_path, monkeypatch)
    store = AtomicF0AnchorStore(binding)
    first = store.save(block, trace, score=_score(block))

    with pytest.raises(FileExistsError, match="stable_id already exists"):
        store.save(block, trace, score=_score(block))

    assert json.loads((store.root / f"{block.stable_id}.json").read_text()) == first
    assert len(list(store.root.glob("*.pt"))) == 1


def test_f0_anchor_store_rejects_payload_and_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, block, trace = _parts(tmp_path, monkeypatch)
    store = AtomicF0AnchorStore(binding)
    manifest = store.save(block, trace, score=_score(block))
    payload = Path(manifest["payload"]["path"])
    payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="payload bytes differ"):
        store.load(block)

    other = _block("manifest-tamper")
    other_trace = _trace(other, binding.contract_hash)
    store.save(other, other_trace, score=_score(other))
    manifest_path = store.root / f"{other.stable_id}.json"
    changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed["stable_id"] = block.stable_id
    manifest_path.write_bytes(_canonical(changed) + b"\n")
    with pytest.raises(RuntimeError, match="manifest SHA-256"):
        store.load(other)


def test_f0_anchor_store_rejects_wrong_block_and_run_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, contract, binding, block, trace = _parts(tmp_path, monkeypatch)
    store = AtomicF0AnchorStore(binding)
    store.save(block, trace, score=_score(block))
    wrong_block = replace(
        block,
        record=replace(block.record, prompt="a different prompt with the same stable id"),
    )
    assert wrong_block.stable_id == block.stable_id
    with pytest.raises(ValueError, match="manifest field block"):
        store.load(wrong_block)

    other_contract = _complete_run_contract(
        authorization,
        method="f0",
        run_id="other-anchor-run",
        paths=contract["paths"],
    )
    other_binding = authorization.bind_run_contract(other_contract)
    with pytest.raises(ValueError, match="run_contract_sha256"):
        AtomicF0AnchorStore(other_binding).load(block)


def test_f0_anchor_store_rejects_symlinks_and_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, block, trace = _parts(tmp_path, monkeypatch)
    store = AtomicF0AnchorStore(binding)
    store.root.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("do not replace", encoding="utf-8")
    (store.root / f"{block.stable_id}.json").symlink_to(outside)
    with pytest.raises(FileExistsError, match="stable_id already exists"):
        store.save(block, trace, score=_score(block))
    assert outside.read_text(encoding="utf-8") == "do not replace"

    linked = _block("payload-link")
    linked_trace = _trace(linked, binding.contract_hash)
    manifest = store.save(linked, linked_trace, score=_score(linked))
    payload = Path(manifest["payload"]["path"])
    moved = store.root / "moved-anchor.pt"
    payload.rename(moved)
    payload.symlink_to(moved)
    with pytest.raises(ValueError, match="not an ordinary file"):
        store.load(linked)

    escaped = _block("payload-escape")
    escaped_trace = _trace(escaped, binding.contract_hash)
    store.save(escaped, escaped_trace, score=_score(escaped))
    manifest_path = store.root / f"{escaped.stable_id}.json"
    changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed["payload"]["path"] = str(tmp_path / "outside.pt")
    core = {key: value for key, value in changed.items() if key != "manifest_sha256"}
    changed["manifest_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    manifest_path.write_bytes(_canonical(changed) + b"\n")
    with pytest.raises(ValueError, match="not content-addressed"):
        store.load(escaped)


def test_f0_anchor_store_rejects_self_consistent_malformed_trace_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, block, trace = _parts(tmp_path, monkeypatch)
    store = AtomicF0AnchorStore(binding)
    manifest = store.save(block, trace, score=_score(block))
    envelope = torch.load(
        manifest["payload"]["path"], map_location="cpu", weights_only=True
    )
    del envelope["trace"]["cached_decisions"]["24"]["frame"]["diagnostics"][
        "valid"
    ]
    _rewrite_payload(store, block, envelope)
    with pytest.raises(ValueError, match="diagnostics payload"):
        store.load(block)


def test_f0_anchor_store_rejects_trace_update_not_bound_to_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace cannot hide a scheduler update unrelated to its state record."""
    _authorization, _contract, binding, block, trace = _parts(tmp_path, monkeypatch)
    first = trace.transitions[0]
    observation = replace(
        first.state,
        scheduler_update=first.state.scheduler_update + 1.0,
    )
    malformed = replace(
        trace,
        transitions=(replace(first, state=observation), *trace.transitions[1:]),
    )
    store = AtomicF0AnchorStore(binding)
    with pytest.raises(ValueError, match="native transition differs"):
        store.save(block, malformed, score=_score(block))


def test_f0_anchor_store_rejects_unknown_state_dataclass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization, _contract, binding, block, trace = _parts(tmp_path, monkeypatch)
    malformed = replace(trace, initial_state=block)
    with pytest.raises(TypeError, match="endpoints must be SdxlEulerState"):
        AtomicF0AnchorStore(binding).save(
            block,
            malformed,
            score_recovery=F0AnchorScoreRecovery.from_scored_image(_score(block)),
        )


def test_f0_anchor_recovery_reseals_and_revalidates_all_score_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(
        authorization,
        method="f0",
        query_budget={"vae_decode": 1, "reward_forward": 2},
        initial_renderer_state_sha256=CHECKPOINT_HASH,
    )
    binding = authorization.bind_run_contract(contract)
    block = _block()
    trace = _trace(block, binding.contract_hash)
    ledger = QueryLedger(
        contract["paths"]["ledger_path"],
        contract["query_budget"],
        run_contract=contract,
        authorization_binding=binding,
    )
    executor = LedgeredOperationExecutor(ledger, authorization_binding=binding)
    state = trace.terminal_state
    action_metadata = build_f0_metric_action(
        branch="anchor",
        target_record_sha256=None,
        renderer_action_history=state.action_history,
    )
    image = torch.full((1, 3, 2, 2), 0.5, dtype=torch.float32)
    image_hash = tensor_sha256(image)
    image_output = operation_output_sha256(image)
    _decoded, decode_receipt = executor.execute_with_receipt(
        "vae_decode",
        state.operation_context(action_metadata=action_metadata),
        lambda: image,
    )
    score_context = state.operation_context(
        action_metadata=action_metadata,
        image_hash=image_hash,
        cached_parent=image_output,
    )
    reward = torch.tensor([0.25], dtype=torch.float32)
    _reward, reward_receipt = executor.execute_with_receipt(
        "reward_forward",
        score_context,
        lambda: reward,
        scalar_or_gradient="scalar",
        parent_receipt=decode_receipt,
    )
    topiq = torch.tensor([0.75], dtype=torch.float32)
    _topiq, topiq_receipt = executor.execute_with_receipt(
        "reward_forward",
        score_context,
        lambda: topiq,
        scalar_or_gradient="scalar",
        model=TOPIQ_NR_MODEL_ID,
        role=TOPIQ_NR_ROLE,
        preprocess_hash=TOPIQ_NR_PREPROCESS_SHA256,
        parent_receipt=decode_receipt,
    )
    score = F0ScoredImage(
        image_sha256=image_hash,
        reward=reward,
        topiq_nr=topiq,
        pixel={
            "clipped_fraction": 0.0,
            "mean_saturation": 0.0,
            "contrast": 0.0,
        },
        decode_receipt=decode_receipt,
        reward_receipt=reward_receipt,
        witness_receipt=topiq_receipt,
    )
    store = AtomicF0AnchorStore(binding)
    store.save(block, trace, score=score)
    persisted = store.load(block)

    resumed_ledger = QueryLedger(
        contract["paths"]["ledger_path"],
        contract["query_budget"],
        run_contract=contract,
        authorization_binding=binding,
    )
    resumed = LedgeredOperationExecutor(
        resumed_ledger, authorization_binding=binding
    )
    restored = _restore_anchor_score(
        SimpleNamespace(operation_executor=resumed),
        persisted.trace,
        persisted.score_recovery,
    )

    assert torch.equal(restored.reward, reward)
    assert torch.equal(restored.topiq_nr, topiq)
    resumed.verify_receipt(restored.decode_receipt)
    resumed.verify_receipt(restored.reward_receipt)
    resumed.verify_receipt(restored.witness_receipt)
