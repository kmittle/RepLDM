import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from AttentionGuidance.latent_renderer import RendererCondition, RendererObservation
from latent_renderer_training.artifacts import module_state_sha256
from latent_renderer_training.collector import CachedEulerDecision, EulerRolloutStep
from latent_renderer_training.contracts import contract_hash
from latent_renderer_training.f0_targets import (
    F0_SOURCES,
    F0_STRATA,
    F0_TRAINING_SEEDS,
    F0_VALIDATION_SEEDS,
)
from latent_renderer_training.f0_teacher import (
    TerminalAnchorRewardReceipt,
    construct_f0_target,
)
import latent_renderer_training.methods.f0 as f0_method
from latent_renderer_training.ledger import QueryLedger
from latent_renderer_training.operations import (
    LedgeredOperationExecutor,
    OperationContext,
    OperationReceipt,
    operation_output_sha256,
    tensor_operation_output_sha256,
)
from latent_renderer_training.renderer import (
    CALIBRATION_STATE_COUNT,
    EulerNativeFrameV1,
    FrameCalibration,
    tensor_sha256,
)
from latent_renderer_training.sdxl_adapter import (
    SdxlEulerState,
    SdxlEulerTrainingAdapter,
    SdxlPromptConditioning,
)
from tests.test_latent_renderer_training import (
    _complete_run_contract,
    _training_authorization,
)


def _renderer(mask=(True,) * 6):
    calibration = FrameCalibration(
        tuple(mask),
        CALIBRATION_STATE_COUNT,
        (1.0,) * 6,
        (1.0,) * 6,
        manifest_sha256="a" * 64,
        source_sha256="b" * 64,
        state_provenance_sha256="c" * 64,
    )
    return EulerNativeFrameV1(calibration=calibration, latent_channels=2)


def _reference(renderer):
    result = copy.deepcopy(renderer).eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def _cached(
    renderer,
    *,
    decision_index=8,
    prompt_id="prompt-0",
    seed=F0_TRAINING_SEEDS[0],
    split="train",
    run_contract=None,
):
    generator = torch.Generator().manual_seed(901 + decision_index)
    sample = torch.randn(1, 2, 5, 5, generator=generator)
    clean = torch.randn(1, 2, 5, 5, generator=generator)
    native = torch.randn(1, 2, 5, 5, generator=generator)
    model_output = torch.zeros_like(sample)
    observation = RendererObservation(
        latents_before_step=sample,
        pred_original_sample=clean,
        scheduler_update=native,
        step_index=decision_index,
        timestep=torch.tensor([float(49 - decision_index)]),
        normalized_timestep=torch.tensor([decision_index / 49.0]),
    )
    condition = RendererCondition(
        bases=torch.randn(1, 6, 2, 5, 5, generator=generator),
        prompt_embedding=torch.randn(1, 32, generator=generator),
        state_features=torch.randn(1, 16, generator=generator),
    )
    frame = renderer.prepare_state(
        observation,
        condition,
        sigma_from=1.0,
        sigma_to=0.8,
        native_model_output=model_output,
    )
    assert bool(frame.diagnostics.valid.all())
    step = EulerRolloutStep(
        observation=observation,
        condition=condition,
        native_model_output=model_output,
        native_prev_sample=sample + native,
        sigma_from=1.0,
        sigma_to=0.8,
    )
    conditioning = SdxlPromptConditioning(
        prompt_ids=(prompt_id,),
        prompts=("a test prompt",),
        generation_seeds=(seed,),
        prompt_embeds=torch.zeros(2, 4),
        pooled_prompt_embeds=torch.zeros(1, 4),
        add_text_embeds=torch.zeros(2, 4),
        add_time_ids=torch.zeros(2, 6),
        do_classifier_free_guidance=True,
    )
    checkpoint_hash = module_state_sha256(renderer)
    previous = sum(index < decision_index for index in (8, 24, 40))
    state = SdxlEulerState(
        latents=sample.detach().clone(),
        conditioning=conditioning,
        step_index=decision_index,
        total_steps=50,
        split=split,
        branch="anchor",
        prefix=0,
        checkpoint_hash=checkpoint_hash,
        action_history=tuple(torch.zeros(1, 6).tolist() for _ in range(previous)),
    )
    return CachedEulerDecision(
        decision_index=decision_index,
        state=state,
        step=step,
        frame=frame,
        checkpoint_hash=checkpoint_hash,
        frame_contract_hash=renderer.frame_contract_hash,
        calibration_hash=renderer.calibration_hash,
        run_contract=run_contract,
    )


@pytest.fixture
def formal_executor(tmp_path, monkeypatch):
    next_executor = 0

    def build(renderer):
        nonlocal next_executor
        root = tmp_path / f"executor-{next_executor}"
        next_executor += 1
        root.mkdir()
        authorization = _training_authorization(root, monkeypatch)
        contract = _complete_run_contract(
            authorization,
            method="f0",
            renderer_frame_contract_hash=renderer.frame_contract_hash,
            calibration_hash=renderer.calibration_hash,
            action_contract_hash=contract_hash(renderer.contract),
            initial_renderer_state_sha256=module_state_sha256(renderer),
            query_budget={
                "vae_decode": 64,
                "reward_forward": 64,
                "reward_backward": 64,
            },
        )
        binding = authorization.bind_run_contract(contract)
        ledger = QueryLedger(
            contract["paths"]["ledger_path"],
            contract["query_budget"],
            run_contract=contract,
            authorization_binding=binding,
        )
        return LedgeredOperationExecutor(ledger, authorization_binding=binding)

    return build


class _Vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("placement", torch.tensor(1.0))

    def decode(self, latent, return_dict=False):
        assert return_dict is False
        # The production VAE maps SDXL's latent channels to an RGB image.
        # Keep the fixture's small latent width while honoring that public
        # decode contract.
        return (latent[:, :1].expand(-1, 3, -1, -1),)


class _TerminalReward(nn.Module):
    def __init__(self, value=12.0):
        super().__init__()
        self.register_buffer("value", torch.tensor(float(value)))

    def score_tensor(self, image, prompts):
        assert len(prompts) == image.shape[0]
        return image.flatten(1).mean(dim=1) * 0.0 + self.value


class _Adapter(SdxlEulerTrainingAdapter):

    def __init__(self, gradient, operation_executor, reward_value=12.0):
        self.operation_executor = operation_executor
        self.vae = _Vae().eval()
        self.vae_scaling_factor = 1.0
        self.reward_model = _TerminalReward(reward_value).eval()
        self._reward_receipts = {}
        self.gradient = torch.as_tensor(gradient, dtype=torch.float32)
        self.decode_calls = 0
        self.reward_calls = 0
        self.gradient_calls = 0
        self.reward_state = None
        self.image_hash = None

    def decode(self, state):
        self.decode_calls += 1
        self.reward_state = state
        return super().decode(state)

    def reward(self, state, image):
        self.reward_calls += 1
        assert state is self.reward_state
        return super().reward(state, image)

    def reward_gradient(self, state, reward, inputs, *, image_hash=None):
        self.gradient_calls += 1
        assert state is self.reward_state
        assert inputs.requires_grad
        assert reward.shape == (1,)
        assert isinstance(image_hash, str) and len(image_hash) == 64
        self.image_hash = image_hash
        entry = self._reward_receipts.pop(id(reward), None)
        assert entry is not None and entry[0]() is reward
        parent_receipt = entry[1]
        reward_hash = operation_output_sha256(reward.detach())
        self.operation_executor.validate_receipt(
            parent_receipt,
            kind="reward_forward",
            output_hash=reward_hash,
            context=state.operation_context(
                image_hash=image_hash,
                cached_parent=tensor_operation_output_sha256(image_hash),
            ),
            scalar_or_gradient="scalar",
        )

        def backward():
            return self.gradient.to(device=inputs.device, dtype=inputs.dtype).clone()

        value, _receipt = self.operation_executor.execute_with_receipt(
            "reward_backward",
            state.operation_context(
                image_hash=image_hash,
                cached_parent=reward_hash,
            ),
            backward,
            scalar_or_gradient="gradient",
            parent_receipt=parent_receipt,
        )
        return value

    def set_reward(self, value):
        with torch.no_grad():
            self.reward_model.value.fill_(float(value))


@pytest.fixture
def formal_case(formal_executor):
    def build(renderer, gradient, *, reward_value=12.0, **cached_options):
        executor = formal_executor(renderer)
        adapter = _Adapter(gradient, executor, reward_value=reward_value)
        cached = _cached(
            renderer,
            run_contract=executor.authorization_binding.contract_hash,
            **cached_options,
        )
        return cached, adapter

    return build


def _terminal_state(cached, **overrides):
    state = replace(
        cached.state,
        step_index=50,
        action_history=tuple(
            torch.zeros(1, 6).tolist() for _ in (8, 24, 40)
        ),
    )
    return replace(state, **overrides)


def _terminal_receipt(adapter, cached, *, value=12.0, state=None):
    terminal = _terminal_state(cached) if state is None else state
    action_metadata = {
        "f0_metric_branch": "anchor",
        "reuse_scope": "shared_prompt_seed",
        "target_record_sha256": None,
        "pixel_summary_schema": "repldm.renderer_f0_pixel_summary.v1",
        "renderer_action_history": list(terminal.action_history),
    }
    previous = float(adapter.reward_model.value.detach())
    adapter.set_reward(value)
    try:
        image, decode_receipt = adapter.decode_with_receipt(
            terminal, action_metadata=action_metadata
        )
        reward, reward_receipt = adapter.reward_with_receipt(
            terminal,
            image,
            parent_receipt=decode_receipt,
            action_metadata=action_metadata,
        )
    finally:
        adapter.set_reward(previous)
    return TerminalAnchorRewardReceipt(
        reward=reward.detach(),
        decode_receipt=decode_receipt,
        reward_receipt=reward_receipt,
    )


def _changed_terminal_context(state, field):
    if field == "prompt":
        return replace(
            state,
            conditioning=replace(
                state.conditioning,
                prompt_ids=("other-prompt",),
            ),
        )
    if field == "seed":
        return replace(
            state,
            conditioning=replace(
                state.conditioning,
                generation_seeds=(state.conditioning.generation_seeds[0] + 1,),
            ),
        )
    if field == "split":
        return replace(state, split="validation")
    if field == "branch":
        return replace(state, branch="plus")
    if field == "step":
        return replace(state, step_index=48)
    if field == "checkpoint":
        return replace(state, checkpoint_hash="d" * 64)
    raise AssertionError(field)


def _restart_executor(executor):
    binding = executor.authorization_binding
    ledger = QueryLedger(
        executor.ledger.path,
        executor.ledger.budget,
        run_contract=binding.contract,
        authorization_binding=binding,
    )
    return LedgeredOperationExecutor(
        ledger,
        authorization_binding=binding,
    )


def _construct(cached, renderer, reference, adapter, **overrides):
    terminal_reward_value = overrides.pop("terminal_reward_value", 12.0)
    terminal_receipt = overrides.pop("terminal_anchor_reward", None)
    if terminal_receipt is None:
        terminal_receipt = _terminal_receipt(
            adapter,
            cached,
            value=terminal_reward_value,
        )
    arguments = {
        "prompt_id": cached.state.conditioning.prompt_ids[0],
        "source": F0_SOURCES[0],
        "stratum": F0_STRATA[0],
        "fold": 0,
        "terminal_anchor_reward": terminal_receipt,
        "reward_location": 10.0,
        "reward_scale": 4.0,
    }
    arguments.update(overrides)
    return construct_f0_target(
        cached,
        renderer,
        reference,
        adapter,
        **arguments,
    )


def test_construct_f0_target_uses_one_reward_backward_and_normalizes_active_gradient(
    formal_case,
):
    renderer = _renderer((True, False, True, True, False, True))
    reference = _reference(renderer)
    cached, adapter = formal_case(
        renderer,
        [[1.0, 20.0, 2.0, 2.0, 30.0, 3.0]],
    )

    row = _construct(cached, renderer, reference, adapter)

    assert (adapter.decode_calls, adapter.reward_calls, adapter.gradient_calls) == (1, 1, 1)
    expected = torch.tensor([[1.0, 0.0, 2.0, 2.0, 0.0, 3.0]])
    expected = expected / torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
    torch.testing.assert_close(row.reward_gradient, expected)
    assert row.valid
    assert row.pair_weight == pytest.approx(0.75)
    assert len(adapter.reward_state.action_history) == 1
    assert torch.count_nonzero(torch.as_tensor(adapter.reward_state.action_history[-1])) == 0

    for value, target in (
        (row.plus_u, row.plus_transition),
        (row.minus_u, row.minus_transition),
    ):
        rendered = renderer.apply_coefficients(
            cached.frame, renderer.deterministic_action_from_mean(value)
        )
        expected_transition = cached.frame.native_update + cached.frame.kappa.reshape(
            -1, 1, 1, 1
        ) * rendered.residual
        assert torch.equal(target, expected_transition)

    assert row.state.sample.data_ptr() != cached.frame.sample.data_ptr()
    assert row.anchor_u.data_ptr() != renderer.action_parameters(cached.frame).data_ptr()
    for value in (
        row.anchor_u,
        row.plus_u,
        row.minus_u,
        row.reward_gradient,
        row.plus_transition,
        row.minus_transition,
        row.reference_u,
        row.state.sample,
    ):
        assert not value.requires_grad


@pytest.mark.parametrize(
    ("decision_index", "split", "seed", "fold"),
    [
        (8, "train", F0_TRAINING_SEEDS[0], 0),
        (24, "train", F0_TRAINING_SEEDS[1], 1),
        (40, "validation", F0_VALIDATION_SEEDS[0], None),
    ],
)
def test_construct_f0_target_derives_registered_state_metadata(
    decision_index, split, seed, fold, formal_case
):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(
        renderer,
        [[1.0] * 6],
        decision_index=decision_index,
        split=split,
        seed=seed,
    )
    row = _construct(
        cached,
        renderer,
        reference,
        adapter,
        fold=fold,
    )

    assert row.decision_index == decision_index
    assert row.split == split
    assert row.generation_seed == seed
    assert row.fold == fold


def test_construct_f0_target_backtracks_in_frozen_order_without_more_rewards(
    monkeypatch, formal_case
):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    original = renderer.apply_coefficients
    candidates = []

    def apply(state, action, *, validate=True):
        output = original(state, action, validate=validate)
        if not action.requires_grad and len(candidates) < 16:
            index = len(candidates)
            candidates.append(action.detach().clone())
            if index % 4 == 0:
                output = replace(
                    output,
                    diagnostics=replace(
                        output.diagnostics,
                        valid=torch.zeros_like(output.diagnostics.valid),
                    ),
                )
        return output

    monkeypatch.setattr(renderer, "apply_coefficients", apply)
    row = _construct(cached, renderer, reference, adapter)

    assert row.valid
    assert len(candidates) == 16
    assert (adapter.decode_calls, adapter.reward_calls, adapter.gradient_calls) == (1, 1, 1)
    candidate_norms = [
        float(torch.linalg.vector_norm(torch.atanh(action), dim=-1))
        for action in candidates
    ]
    expected = [
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.1875,
        0.125,
        0.09375,
        0.078125,
    ] * 2
    assert candidate_norms == pytest.approx(expected, abs=2e-6)
    assert float(torch.linalg.vector_norm(row.plus_u - row.anchor_u)) == pytest.approx(
        0.125, abs=2e-6
    )
    assert float(torch.linalg.vector_norm(row.minus_u - row.anchor_u)) == pytest.approx(
        0.125, abs=2e-6
    )


def test_pair_weight_uses_terminal_anchor_reward_not_local_gradient_reward(formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6], reward_value=-100.0)

    low_local = _construct(
        cached,
        renderer,
        reference,
        adapter,
        terminal_reward_value=12.0,
    )
    adapter.set_reward(100.0)
    high_local = _construct(
        cached,
        renderer,
        reference,
        adapter,
        terminal_reward_value=12.0,
    )
    adapter.set_reward(-100.0)
    changed_terminal = _construct(
        cached,
        renderer,
        reference,
        adapter,
        terminal_reward_value=8.0,
    )

    assert low_local.pair_weight == pytest.approx(0.75)
    assert high_local.pair_weight == pytest.approx(0.75)
    assert changed_terminal.pair_weight == pytest.approx(0.25)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), torch.tensor([1.0], requires_grad=True), torch.ones(2)],
)
def test_pair_weight_rejects_unfrozen_or_nonfinite_terminal_anchor(value):
    with pytest.raises(ValueError, match="terminal anchor reward"):
        TerminalAnchorRewardReceipt(
            reward=value,
            decode_receipt=object(),
            reward_receipt=object(),
        )


@pytest.mark.parametrize("gradient", [torch.zeros(1, 6), torch.ones(1, 6)])
def test_construct_f0_target_falls_back_for_zero_gradient_or_invalid_candidates(
    gradient, monkeypatch, formal_case
):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, gradient)
    if torch.count_nonzero(gradient):
        original = renderer.apply_coefficients
        candidate_calls = 0

        def reject(state, action, *, validate=True):
            nonlocal candidate_calls
            output = original(state, action, validate=validate)
            if not action.requires_grad and candidate_calls < 16:
                candidate_calls += 1
                output = replace(
                    output,
                    diagnostics=replace(
                        output.diagnostics,
                        valid=torch.zeros_like(output.diagnostics.valid),
                    ),
                )
            return output

        monkeypatch.setattr(renderer, "apply_coefficients", reject)

    row = _construct(cached, renderer, reference, adapter)

    assert not row.valid
    assert torch.equal(row.plus_u, row.anchor_u)
    assert torch.equal(row.minus_u, row.anchor_u)
    assert torch.equal(row.reward_gradient, torch.zeros_like(row.reward_gradient))
    assert torch.equal(row.plus_transition, cached.frame.native_update)
    assert torch.equal(row.minus_transition, cached.frame.native_update)
    assert (adapter.decode_calls, adapter.reward_calls, adapter.gradient_calls) == (1, 1, 1)


def test_construct_f0_target_does_not_call_autograd_directly(monkeypatch, formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("formal F0 construction called torch.autograd.grad")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    row = _construct(cached, renderer, reference, adapter)
    assert row.valid


def test_construct_f0_target_rejects_contract_state_and_prompt_drift_before_reward(
    formal_case,
):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])

    cases = []
    cases.append((replace(cached, frame_contract_hash="changed"), renderer, reference, {}))
    cases.append(
        (
            replace(cached, frame=replace(cached.frame, context=cached.frame.context + 1.0)),
            renderer,
            reference,
            {},
        )
    )
    changed_reference = _reference(renderer)
    with torch.no_grad():
        changed_reference.policy[-1].bias.add_(0.1)
    cases.append((cached, renderer, changed_reference, {}))
    cases.append((cached, renderer, reference, {"prompt_id": "other-prompt"}))

    for changed_cache, behavior, frozen, arguments in cases:
        with pytest.raises(ValueError):
            _construct(changed_cache, behavior, frozen, adapter, **arguments)
        assert (adapter.decode_calls, adapter.reward_calls, adapter.gradient_calls) == (0, 0, 0)


def test_construct_f0_target_rejects_nonzero_c0_even_when_hashes_match(formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    with torch.no_grad():
        renderer.policy[-1].bias.fill_(0.1)
        reference.policy[-1].bias.fill_(0.1)
    cached, adapter = formal_case(renderer, [[1.0] * 6])

    with pytest.raises(ValueError, match="strict zero-action C0"):
        _construct(cached, renderer, reference, adapter)
    assert (adapter.decode_calls, adapter.reward_calls, adapter.gradient_calls) == (0, 0, 0)


def test_construct_f0_target_rejects_unwrapped_terminal_reward(formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])

    with pytest.raises(TypeError, match="TerminalAnchorRewardReceipt"):
        _construct(
            cached,
            renderer,
            reference,
            adapter,
            terminal_anchor_reward=12.0,
        )
    assert (adapter.decode_calls, adapter.reward_calls, adapter.gradient_calls) == (0, 0, 0)


def test_terminal_receipt_rejects_forged_executor_seal(formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    receipt = _terminal_receipt(adapter, cached)
    forged_decode = replace(receipt.decode_receipt, _executor_seal=object())
    forged = replace(receipt, decode_receipt=forged_decode)

    with pytest.raises(TypeError, match="not issued by this operation executor"):
        _construct(
            cached,
            renderer,
            reference,
            adapter,
            terminal_anchor_reward=forged,
        )


@pytest.mark.parametrize(
    "field", ["split", "prompt", "seed", "branch", "step", "checkpoint"]
)
def test_terminal_receipt_rejects_authentic_wrong_context(field, formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    terminal = _changed_terminal_context(_terminal_state(cached), field)
    receipt = _terminal_receipt(adapter, cached, state=terminal)

    with pytest.raises(ValueError, match="differs from the expected call"):
        _construct(
            cached,
            renderer,
            reference,
            adapter,
            terminal_anchor_reward=receipt,
        )
    assert (adapter.decode_calls, adapter.reward_calls, adapter.gradient_calls) == (0, 0, 0)


def test_terminal_receipt_rejects_other_executor_and_run_contract(
    formal_case, formal_executor
):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    other_executor = formal_executor(renderer)
    other_adapter = _Adapter([[1.0] * 6], other_executor)
    other_receipt = _terminal_receipt(other_adapter, cached)

    with pytest.raises(TypeError, match="not issued by this operation executor"):
        _construct(
            cached,
            renderer,
            reference,
            adapter,
            terminal_anchor_reward=other_receipt,
        )
    with pytest.raises(ValueError, match="cached run contract differs"):
        _construct(
            cached,
            renderer,
            reference,
            other_adapter,
            terminal_anchor_reward=other_receipt,
        )


def test_terminal_receipt_rejects_altered_reward_tensor(formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    receipt = _terminal_receipt(adapter, cached)
    altered = replace(receipt, reward=receipt.reward + 1.0)

    with pytest.raises(ValueError, match="differs from the expected call"):
        _construct(
            cached,
            renderer,
            reference,
            adapter,
            terminal_anchor_reward=altered,
        )


def test_terminal_receipt_rejects_altered_reward_output_hash(formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    receipt = _terminal_receipt(adapter, cached)
    altered = replace(
        receipt,
        reward_receipt=replace(receipt.reward_receipt, output_hash="6" * 64),
    )

    with pytest.raises(ValueError, match="differs from the expected call"):
        _construct(
            cached,
            renderer,
            reference,
            adapter,
            terminal_anchor_reward=altered,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "OtherReward"),
        ("role", "other_role"),
        ("preprocess_hash", "1" * 64),
        ("model_config_sha256", "2" * 64),
        ("model_asset_manifest_sha256", "3" * 64),
    ],
)
def test_terminal_receipt_rejects_wrong_reward_identity(field, value, formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    receipt = _terminal_receipt(adapter, cached)
    altered_reward = replace(receipt.reward_receipt, **{field: value})
    altered = replace(receipt, reward_receipt=altered_reward)

    with pytest.raises(ValueError, match="differs from the expected call"):
        _construct(
            cached,
            renderer,
            reference,
            adapter,
            terminal_anchor_reward=altered,
        )


def test_terminal_receipt_rejects_altered_image_and_missing_parent(formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    receipt = _terminal_receipt(adapter, cached)
    wrong_image = replace(
        receipt.reward_receipt,
        context=replace(receipt.reward_receipt.context, image_hash="4" * 64),
    )
    missing_parent = replace(
        receipt.reward_receipt,
        context=replace(receipt.reward_receipt.context, cached_parent=None),
    )

    for changed in (wrong_image, missing_parent):
        with pytest.raises(ValueError, match="terminal reward"):
            _construct(
                cached,
                renderer,
                reference,
                adapter,
                terminal_anchor_reward=replace(
                    receipt,
                    reward_receipt=changed,
                ),
            )


def test_terminal_receipt_rejects_mismatched_authentic_vae_parent(formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    receipt = _terminal_receipt(adapter, cached)
    changed_state = replace(
        _terminal_state(cached),
        latents=_terminal_state(cached).latents + 1.0,
    )
    changed_chain = _terminal_receipt(adapter, cached, state=changed_state)
    mixed = TerminalAnchorRewardReceipt(
        reward=changed_chain.reward,
        decode_receipt=receipt.decode_receipt,
        reward_receipt=changed_chain.reward_receipt,
    )

    with pytest.raises(ValueError, match="differs from its VAE decode output"):
        _construct(
            cached,
            renderer,
            reference,
            adapter,
            terminal_anchor_reward=mixed,
        )


def test_terminal_receipt_rejects_nonzero_terminal_action_history(formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    terminal = _terminal_state(cached)
    actions = list(terminal.action_history)
    actions[-1] = torch.full((1, 6), 0.25).tolist()
    receipt = _terminal_receipt(
        adapter,
        cached,
        state=replace(terminal, action_history=tuple(actions)),
    )

    with pytest.raises(ValueError, match="differs from the expected call"):
        _construct(
            cached,
            renderer,
            reference,
            adapter,
            terminal_anchor_reward=receipt,
        )


def test_decode_and_reward_legacy_interfaces_remain_tensor_only(formal_case):
    renderer = _renderer()
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    terminal = _terminal_state(cached)

    image = adapter.decode(terminal)
    reward = adapter.reward(terminal, image)

    assert isinstance(image, torch.Tensor) and image.shape == (
        terminal.latents.shape[0],
        3,
        terminal.latents.shape[2],
        terminal.latents.shape[3],
    )
    assert isinstance(reward, torch.Tensor) and reward.shape == (1,)
    assert (adapter.decode_calls, adapter.reward_calls, adapter.gradient_calls) == (1, 1, 0)
    summary = adapter.operation_executor.ledger.summary()
    assert summary["successful_amount"]["vae_decode"] == 1
    assert summary["successful_amount"]["reward_forward"] == 1


def test_restore_receipt_reseals_capability_and_supports_f0_resume(formal_case):
    renderer = _renderer()
    reference = _reference(renderer)
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    original = _terminal_receipt(adapter, cached)
    executor = adapter.operation_executor
    resumed = _restart_executor(executor)

    decode_receipt = resumed.restore_receipt(original.decode_receipt.reservation_id)
    reward_receipt = resumed.restore_receipt(original.reward_receipt.reservation_id)
    resumed.verify_receipt(decode_receipt)
    resumed.verify_receipt(reward_receipt)
    with pytest.raises(TypeError, match="not issued"):
        resumed.verify_receipt(original.decode_receipt)
    with pytest.raises(TypeError, match="not issued"):
        executor.verify_receipt(decode_receipt)

    resumed_adapter = _Adapter([[1.0] * 6], resumed)
    row = _construct(
        cached,
        renderer,
        reference,
        resumed_adapter,
        terminal_anchor_reward=TerminalAnchorRewardReceipt(
            reward=original.reward,
            decode_receipt=decode_receipt,
            reward_receipt=reward_receipt,
        ),
    )
    assert row.valid


def test_restore_receipt_rejects_failed_unfinished_and_unknown_records(formal_case):
    renderer = _renderer()
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    terminal = _terminal_state(cached)
    legitimate = _terminal_receipt(adapter, cached)
    executor = adapter.operation_executor

    with pytest.raises(RuntimeError, match="planned decode failure"):
        executor.execute_with_receipt(
            "vae_decode",
            terminal.operation_context(),
            lambda: (_ for _ in ()).throw(RuntimeError("planned decode failure")),
        )
    records = executor.ledger.verified_records()
    failed_id = next(
        row["reservation_id"]
        for row in reversed(records)
        if row["type"] == "receipt" and row["success"] is False
    )
    source_metadata = next(
        row["metadata"]
        for row in records
        if row["type"] == "reservation"
        and row["reservation_id"] == legitimate.decode_receipt.reservation_id
    )
    unfinished = executor.ledger.reserve(
        "vae_decode",
        1,
        metadata=source_metadata,
    )
    resumed = _restart_executor(executor)

    for reservation_id in (failed_id, unfinished.reservation_id, "not-a-reservation"):
        with pytest.raises(RuntimeError, match="no unique successful"):
            resumed.restore_receipt(reservation_id)


def test_restore_receipt_rejects_wrong_parent_kind_hash_and_context(formal_case):
    renderer = _renderer()
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    terminal = _terminal_state(cached)
    legitimate = _terminal_receipt(adapter, cached)
    executor = adapter.operation_executor

    _, wrong_kind = executor.execute_with_receipt(
        "reward_backward",
        terminal.operation_context(
            image_hash=legitimate.reward_receipt.image_hash,
            cached_parent=legitimate.decode_receipt.output_hash,
        ),
        lambda: torch.ones(1, 6),
        scalar_or_gradient="gradient",
        parent_receipt=legitimate.decode_receipt,
    )
    _, wrong_hash = executor.execute_with_receipt(
        "reward_backward",
        terminal.operation_context(
            image_hash=legitimate.reward_receipt.image_hash,
            cached_parent="5" * 64,
        ),
        lambda: torch.ones(1, 6),
        scalar_or_gradient="gradient",
        parent_receipt=legitimate.reward_receipt,
    )
    _, wrong_context = executor.execute_with_receipt(
        "reward_backward",
        replace(terminal, branch="minus").operation_context(
            image_hash=legitimate.reward_receipt.image_hash,
            cached_parent=legitimate.reward_receipt.output_hash,
        ),
        lambda: torch.ones(1, 6),
        scalar_or_gradient="gradient",
        parent_receipt=legitimate.reward_receipt,
    )
    resumed = _restart_executor(executor)

    for receipt in (wrong_kind, wrong_hash, wrong_context):
        with pytest.raises(ValueError, match="durable operation"):
            resumed.restore_receipt(receipt.reservation_id)


def test_restore_receipt_rejects_duplicate_and_inconsistent_pairs(
    formal_case, monkeypatch
):
    renderer = _renderer()
    cached, adapter = formal_case(renderer, [[1.0] * 6])
    receipt = _terminal_receipt(adapter, cached).decode_receipt
    executor = adapter.operation_executor
    pair = next(
        item
        for item in executor.ledger.successful_receipt_pairs()
        if item[0]["reservation_id"] == receipt.reservation_id
    )
    resumed = _restart_executor(executor)

    monkeypatch.setattr(
        resumed.ledger,
        "successful_receipt_pairs",
        lambda kind=None: (pair, pair),
    )
    with pytest.raises(RuntimeError, match="no unique successful"):
        resumed.restore_receipt(receipt.reservation_id)

    changed_durable = dict(pair[1])
    changed_durable["kind"] = "reward_forward"
    monkeypatch.setattr(
        resumed.ledger,
        "successful_receipt_pairs",
        lambda kind=None: ((pair[0], changed_durable),),
    )
    with pytest.raises(ValueError, match="kind or metadata is inconsistent"):
        resumed.restore_receipt(receipt.reservation_id)


def test_run_f0_publishes_bound_artifact_for_unexpected_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs"
    contract = {
        "run_id": "unexpected-f0",
        "method": "f0",
        "code_commit": "a" * 40,
        "paths": {
            "run_dir": str(tmp_path / "run"),
            "output_dir": str(output_dir),
            "ledger_path": str(tmp_path / "run" / "ledger.jsonl"),
        },
    }
    binding = SimpleNamespace(contract=contract, contract_hash="b" * 64)
    registration = {"path": "registration", "sha256": "c" * 64, "bytes": 1}
    monkeypatch.setattr(f0_method, "_require_f0_contract", lambda _binding: contract)
    monkeypatch.setattr(
        f0_method,
        "_publish_screen_registration",
        lambda _binding, _contract: registration,
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic runtime failure")

    monkeypatch.setattr(f0_method, "_run_f0_impl", fail)
    with pytest.raises(RuntimeError, match="synthetic runtime failure"):
        f0_method.run_f0(binding)

    artifact_path = output_dir / f0_method.F0_FAILURE_FILENAME
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "repldm.renderer_f0_failure.v1"
    assert payload["status"] == "screen_failed"
    assert payload["run_contract_sha256"] == binding.contract_hash
    assert payload["code_commit"] == contract["code_commit"]
    assert payload["stage"] == "execution"
    assert payload["error"]["message"] == "synthetic runtime failure"


def test_run_f0_does_not_reuse_a_different_failure_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs"
    contract = {
        "run_id": "unexpected-f0-retry",
        "method": "f0",
        "code_commit": "a" * 40,
        "paths": {
            "run_dir": str(tmp_path / "run"),
            "output_dir": str(output_dir),
            "ledger_path": str(tmp_path / "run" / "ledger.jsonl"),
        },
    }
    binding = SimpleNamespace(contract=contract, contract_hash="b" * 64)
    registration = {"path": "registration", "sha256": "c" * 64, "bytes": 1}
    monkeypatch.setattr(f0_method, "_require_f0_contract", lambda _binding: contract)
    monkeypatch.setattr(
        f0_method,
        "_publish_screen_registration",
        lambda _binding, _contract: registration,
    )
    failures = iter(("first failure", "second failure"))

    def fail(*_args, **_kwargs):
        raise RuntimeError(next(failures))

    monkeypatch.setattr(f0_method, "_run_f0_impl", fail)
    with pytest.raises(RuntimeError, match="first failure"):
        f0_method.run_f0(binding)
    with pytest.raises(RuntimeError, match="second failure") as caught:
        f0_method.run_f0(binding)

    assert isinstance(
        getattr(caught.value, "_f0_failure_artifact_error", None), FileExistsError
    )
    payload = json.loads(
        (output_dir / f0_method.F0_FAILURE_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["error"]["message"] == "first failure"
