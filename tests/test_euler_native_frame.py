import math

import pytest
import torch
import latent_renderer_training.renderer as renderer_module

from AttentionGuidance.latent_renderer import (
    RendererCondition,
    RendererObservation,
    prepare_euler_clean_endpoint,
    predict_euler_no_churn_prev_sample,
)
from latent_renderer_training.distributions import SquashedGaussian
from latent_renderer_training.collector import (
    EulerNativeRolloutCollector,
    EulerRolloutStep,
    EulerTransitionResult,
)
from latent_renderer_training.rollout import Transition, collect_antithetic_rollout
from latent_renderer_training.renderer import (
    CALIBRATION_STATE_COUNT,
    EulerNativeFrameV1,
    FrameCalibration,
    FrameCalibrationSample,
    calibrate_global_mask,
    haar_tangent_frame,
    phase_matched_frame,
)


def _calibration(mask=(True,) * 6):
    return FrameCalibration(
        tuple(mask),
        CALIBRATION_STATE_COUNT,
        (1.0,) * 6,
        (1.0,) * 6,
        manifest_sha256="a" * 64,
        source_sha256="b" * 64,
        state_provenance_sha256="c" * 64,
    )


def _renderer(mask=(True,) * 6, **kwargs):
    return EulerNativeFrameV1(
        calibration=_calibration(mask), latent_channels=2, **kwargs
    )


def _state(renderer, *, batch=2, channels=2, height=6, width=6):
    torch.manual_seed(123)
    clean = torch.randn(batch, channels, height, width)
    raw = torch.randn(batch, 6, channels, height, width)
    native = torch.randn_like(clean)
    observation = RendererObservation(
        latents_before_step=clean,
        pred_original_sample=clean,
        scheduler_update=native,
        step_index=8,
        timestep=torch.full((batch,), 400.0),
        normalized_timestep=torch.full((batch,), 8.0 / 49.0),
    )
    condition = RendererCondition(
        bases=raw,
        prompt_embedding=torch.randn(batch, 32),
        state_features=torch.randn(batch, 16),
    )
    return clean, raw, native, renderer.prepare_state(
        observation, condition, sigma_from=1.0, sigma_to=0.8
    )


def test_registered_parameter_count_and_zero_initialization():
    renderer = _renderer()
    assert renderer.parameter_count == 91654
    assert renderer.scheduler_mapping == "euler_native_frame_v1"
    assert renderer.requires_euler_native_adapter is True
    assert all(torch.equal(parameter, torch.zeros_like(parameter)) for parameter in [renderer.policy[-1].weight, renderer.policy[-1].bias])


def test_renderer_rejects_zero_cap_and_registered_channel_drift():
    with pytest.raises(ValueError, match="bounds"):
        EulerNativeFrameV1(
            calibration=_calibration(),
            latent_channels=2,
            max_update_ratio=0.0,
        )
    renderer = EulerNativeFrameV1(calibration=_calibration(), latent_channels=4)
    clean = torch.randn(1, 2, 4, 4)
    observation = RendererObservation(
        clean,
        clean,
        torch.randn_like(clean),
        0,
        torch.ones(1),
        torch.zeros(1),
    )
    condition = RendererCondition(
        torch.randn(1, 6, 2, 4, 4),
        torch.zeros(1, 32),
        torch.zeros(1, 16),
    )
    with pytest.raises(ValueError, match="expected 4"):
        renderer.prepare_state(
            observation, condition, sigma_from=1.0, sigma_to=0.8
        )


def test_prepare_state_rejects_non_tensor_scheduler_update_cleanly():
    renderer = _renderer()
    clean = torch.randn(1, 2, 4, 4)
    observation = RendererObservation(
        clean,
        clean,
        None,
        0,
        torch.ones(1),
        torch.zeros(1),
    )
    condition = RendererCondition(
        torch.randn(1, 6, 2, 4, 4),
        torch.zeros(1, 32),
        torch.zeros(1, 16),
    )
    with pytest.raises(TypeError, match="scheduler_update"):
        renderer.prepare_state(
            observation, condition, sigma_from=1.0, sigma_to=0.8
        )


def test_deterministic_action_stays_inside_open_coefficient_interval():
    renderer = _renderer()
    _clean, _raw, _native, state = _state(renderer)
    with torch.no_grad():
        renderer.policy[-1].bias.fill_(1e6)
    action = renderer.deterministic_action(state)
    assert bool(torch.all(action.abs() < renderer.coefficient_bound))
    renderer.contract.validate_action(action)


def test_frame_rejects_unregistered_slot_count():
    renderer = _renderer()
    clean = torch.randn(1, 2, 4, 4)
    native = torch.randn_like(clean)
    observation = RendererObservation(clean, clean, native, 0, torch.ones(1), torch.zeros(1))
    condition = RendererCondition(torch.randn(1, 5, 2, 4, 4), torch.zeros(1, 32), torch.zeros(1, 16))
    with pytest.raises(ValueError, match="exactly 6 slots"):
        renderer.prepare_state(observation, condition, sigma_from=1.0, sigma_to=0.8)


def test_frame_whitening_and_zero_action_are_exact_noops():
    renderer = _renderer()
    clean, _raw, native, state = _state(renderer)
    assert bool(state.diagnostics.valid.all())
    active = state.mapped_bases[:, torch.as_tensor(renderer.active_mask)]
    gram = active.flatten(2) @ active.flatten(2).transpose(1, 2)
    # The physical scale is beta*s; after undoing that scale the rows are
    # orthonormal in the native scheduler metric.
    scale = 0.999 * renderer.max_update_ratio / math.sqrt(6)
    normalized = active.flatten(2) / (scale * torch.linalg.vector_norm(native.flatten(1), dim=1).reshape(-1, 1, 1))
    torch.testing.assert_close(
        normalized @ normalized.transpose(1, 2),
        torch.eye(6).expand(clean.shape[0], -1, -1),
        atol=2e-4,
        rtol=2e-4,
    )
    output = renderer.apply_coefficients(state, torch.zeros(clean.shape[0], 6))
    assert torch.equal(output.guided_x0, clean)
    assert torch.equal(output.residual, torch.zeros_like(clean))
    assert torch.equal(output.diagnostics.mapped_update_ratio, torch.zeros(clean.shape[0]))


def test_all_action_box_corners_respect_scheduler_and_angle_caps():
    renderer = _renderer()
    clean, _raw, native, state = _state(renderer, batch=1)
    bound = 1.0 - 1e-4
    for bits in range(64):
        action = torch.tensor(
            [[bound if bits & (1 << index) else -bound for index in range(6)]]
        )
        output = renderer.apply_coefficients(state, action)
        assert bool(torch.isfinite(output.guided_x0).all())
        assert float(output.diagnostics.mapped_update_ratio[0]) <= 0.05 + 1e-5
        assert float(output.diagnostics.angle[0]) <= renderer.theta_max + 1e-5
        mean_delta = (
            output.guided_x0.float().mean(dim=(-2, -1))
            - clean.float().mean(dim=(-2, -1))
        ).abs().max()
        assert float(mean_delta) < 2e-5


def test_calibration_removes_duplicate_slot_globally():
    torch.manual_seed(99)
    batch = CALIBRATION_STATE_COUNT
    clean = torch.randn(batch, 2, 2, 2)
    raw = torch.randn(batch, 6, 2, 2, 2)
    raw[:, 1] = raw[:, 0]
    sample = FrameCalibrationSample(
        clean,
        raw,
        torch.randn_like(clean),
        1.0,
        0.8,
        manifest_sha256="a" * 64,
        source_sha256="b" * 64,
        state_ids=[f"state-{index}" for index in range(batch)],
        state_hashes=[f"{index + 1:064x}" for index in range(batch)],
    )
    result = calibrate_global_mask([sample], slots=6)
    assert result.active_mask[0]
    assert not result.active_mask[1]
    assert result.rank >= 2


def test_calibration_rejects_near_zero_scheduler_state():
    clean = torch.randn(1, 2, 4, 4)
    raw = torch.randn(1, 6, 2, 4, 4)
    with pytest.raises(ValueError, match="near-zero native update"):
        calibrate_global_mask(
            [
                FrameCalibrationSample(
                    clean,
                    raw,
                    torch.zeros_like(clean),
                    1.0,
                    0.8,
                    manifest_sha256="a" * 64,
                    source_sha256="b" * 64,
                    state_ids=["state-0"],
                    state_hashes=["1" * 64],
                )
            ],
            slots=6,
        )


def test_calibration_requires_the_complete_registered_state_table():
    clean = torch.randn(1, 2, 2, 2)
    raw = torch.randn(1, 6, 2, 2, 2)
    sample = FrameCalibrationSample(
        clean,
        raw,
        torch.randn_like(clean),
        1.0,
        0.8,
        manifest_sha256="a" * 64,
        source_sha256="b" * 64,
        state_ids=["state-0"],
        state_hashes=["1" * 64],
    )
    with pytest.raises(ValueError, match="exactly 576"):
        calibrate_global_mask([sample])


def test_renderer_cannot_bypass_calibration_with_an_active_mask():
    with pytest.raises(ValueError, match="FrameCalibration"):
        EulerNativeFrameV1(active_mask=(True,) * 6, latent_channels=2)


def test_inactive_coordinates_are_removed_from_probability_measure():
    renderer = _renderer((True, False, True, False, True, False))
    _clean, _raw, _native, state = _state(renderer)
    mean = renderer.action_parameters(state)
    assert torch.equal(mean[:, 1::2], torch.zeros_like(mean[:, 1::2]))
    action, _u = renderer.action_distribution(mean).rsample(
        torch.zeros_like(mean)
    )
    assert torch.equal(action[:, 1::2], torch.zeros_like(action[:, 1::2]))
    logp = renderer.action_distribution(mean).log_prob(action)
    manual = SquashedGaussian(mean, renderer.contract).log_prob(action)
    torch.testing.assert_close(logp, manual)


def test_haar_and_phase_matched_controls_are_deterministic():
    torch.manual_seed(7)
    clean = torch.randn(1, 2, 8, 8)
    native = torch.randn_like(clean)
    first = haar_tangent_frame(clean, native, active_mask=(True,) * 6, key="frame")
    second = haar_tangent_frame(clean, native, active_mask=(True,) * 6, key="frame")
    assert torch.equal(first, second)
    mapped = torch.randn(1, 6, 2, 8, 8)
    randomized = phase_matched_frame(mapped, key="phase")
    randomized_again = phase_matched_frame(mapped, key="phase")
    assert torch.equal(randomized, randomized_again)
    before = torch.fft.rfft2(mapped.float(), dim=(-2, -1), norm="ortho").abs()
    after = torch.fft.rfft2(randomized.float(), dim=(-2, -1), norm="ortho").abs()
    torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-5)


def test_invalid_kappa_fails_closed_before_frame_whitening():
    renderer = _renderer()
    clean = torch.randn(1, 2, 4, 4)
    raw = torch.randn(1, 6, 2, 4, 4)
    native = torch.randn_like(clean)
    observation = RendererObservation(clean, clean, native, 0, torch.ones(1), torch.zeros(1))
    condition = RendererCondition(raw, torch.zeros(1, 32), torch.zeros(1, 16))
    state = renderer.prepare_state(
        observation, condition, sigma_from=1.0, sigma_to=1.0
    )
    assert not bool(state.diagnostics.valid[0])
    output = renderer.apply_coefficients(state, torch.full((1, 6), 0.2))
    assert torch.equal(output.guided_x0, clean)


def test_euler_mapped_invalid_kappa_returns_zero_ratio_fallback():
    renderer = _renderer()
    sample = torch.randn(1, 2, 4, 4)
    native_output = torch.randn_like(sample)
    endpoint = prepare_euler_clean_endpoint(
        sample,
        native_output,
        sigma_from=1.0,
        sigma_to=1.0,
        prediction_type="epsilon",
    )
    bases = torch.randn(1, 6, 2, 4, 4)
    mapped = renderer.forward_euler_mapped(
        endpoint.pred_original_sample,
        bases,
        sample=sample,
        native_model_output=native_output,
        sigma_from=1.0,
        sigma_to=1.0,
        prediction_type="epsilon",
        scheduler_update=endpoint.nominal_update,
        prompt_embedding=torch.zeros(1, 32),
        state_features=torch.zeros(1, 16),
        action=torch.full((1, 6), 0.2),
    )
    assert not bool(mapped.rendered.diagnostics.valid[0])
    assert torch.equal(mapped.predicted_prev_sample, sample)
    assert torch.equal(mapped.model_output, native_output)
    assert torch.equal(mapped.mapped_intervention.ratio, torch.zeros(1))


def test_antithetic_collector_reuses_prefix_and_recomputes_later_means():
    calls = []

    def step(state, action, index):
        calls.append((state, action, index))
        value = state + (0 if action is None else float(action[..., 0].item())) + 1
        transition = Transition(
            state, torch.zeros(1) if action is None else action,
            value, torch.ones(1), torch.ones(1), step_index=index
        )
        return value, transition

    def mean_fn(state, index):
        return torch.tensor([[float(state) / 10.0, float(index) / 100.0]])

    def action_from_mean(mean, noise):
        action = mean + noise
        return action, action.sum(dim=-1)

    collection = collect_antithetic_rollout(
        0.0,
        decision_indices=(2, 4),
        total_steps=6,
        step_fn=step,
        mean_fn=mean_fn,
        action_from_mean=action_from_mean,
        noise_by_decision={2: torch.tensor([[0.1, 0.0]]), 4: torch.tensor([[0.2, 0.0]])},
        registered_decision_indices=(2, 4),
    )
    assert len([item for item in calls if item[2] == 0]) == 1
    assert len([item for item in calls if item[2] == 1]) == 1
    assert len(collection.proposals) == 2
    assert collection.proposals[1].plus_action.flatten()[0].item() != collection.proposals[0].plus_action.flatten()[0].item()
    # Later decisions visit branch-specific states, so their density means
    # must be retained separately rather than silently replaced by anchor.
    assert collection.proposals[1].plus_mean is not None
    assert collection.proposals[1].minus_mean is not None
    assert collection.proposals[1].anchor_mean is not None
    assert collection.proposals[1].plus_mean.flatten()[0].item() != collection.proposals[1].minus_mean.flatten()[0].item()
    assert [len(collection.branches[name].transitions) for name in ("plus", "minus", "anchor")] == [2, 2, 2]
    assert [len(collection.branches[name].auxiliary_transitions) for name in ("plus", "minus", "anchor")] == [2, 2, 2]
    assert collection.total_steps == 6
    assert set(collection.terminal_states) == {"plus", "minus", "anchor"}
    assert [row.step_index for row in collection.prefix_transitions] == [0, 1]


def test_euler_native_collector_counts_one_observation_per_physical_state():
    renderer = _renderer()
    calls = {"observe": 0, "transition": 0}

    def observe(state, index):
        calls["observe"] += 1
        sigma_from = float(6 - index)
        sigma_to = float(5 - index)
        native_output = 0.05 * state + 0.01 * (index + 1)
        endpoint = prepare_euler_clean_endpoint(
            state,
            native_output,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            prediction_type="epsilon",
        )
        generator = torch.Generator(device="cpu").manual_seed(1000 + calls["observe"])
        bases = torch.randn(1, 6, 2, 4, 4, generator=generator)
        observation = RendererObservation(
            latents_before_step=state.detach(),
            pred_original_sample=endpoint.pred_original_sample.detach(),
            scheduler_update=endpoint.nominal_update.detach(),
            step_index=index,
            timestep=torch.tensor([float(index)]),
            normalized_timestep=torch.tensor([index / 4]),
        )
        condition = RendererCondition(
            bases=bases,
            prompt_embedding=torch.zeros(1, 32),
            state_features=torch.zeros(1, 16),
        )
        return EulerRolloutStep(
            observation=observation,
            condition=condition,
            native_model_output=native_output.detach(),
            native_prev_sample=(state + endpoint.nominal_update).detach(),
            sigma_from=sigma_from,
            sigma_to=sigma_to,
        )

    def transition(state, _index, model_output, step):
        calls["transition"] += 1
        latent = predict_euler_no_churn_prev_sample(
            state,
            model_output,
            sigma_from=step.sigma_from,
            sigma_to=step.sigma_to,
            prediction_type=step.prediction_type,
        )
        return EulerTransitionResult(state=latent, latent=latent)

    collector = EulerNativeRolloutCollector(
        renderer,
        decision_indices=(1, 3),
        registered_decision_indices=(1, 3),
        total_steps=5,
        observe_fn=observe,
        transition_fn=transition,
    )
    collection = collector.collect(
        torch.zeros(1, 2, 4, 4),
        noise_by_decision={
            1: torch.full((1, 6), 0.25),
            3: torch.full((1, 6), -0.5),
        },
    )
    assert collector.last_stats.observe_calls == 11
    assert collector.last_stats.transition_calls == 13
    assert collector.last_stats.decision_transitions == 6
    assert collector.last_stats.auxiliary_transitions == 7
    assert calls == {"observe": 11, "transition": 13}
    assert len(collection.prefix_transitions) == 1
    for branch in ("plus", "minus", "anchor"):
        assert len(collection.branches[branch].transitions) == 2
        assert len(collection.branches[branch].auxiliary_transitions) == 2
    assert collection.proposals[0].plus_pre_squash is not None
    assert collection.branches["plus"].transitions[0].pre_squash is not None


def test_antithetic_collector_keeps_gap_and_suffix_rows_out_of_decisions():
    def step(state, action, index):
        value = state + 1
        transition = Transition(
            state,
            torch.zeros(1) if action is None else action,
            value,
            torch.ones(1),
            torch.ones(1),
            step_index=index,
        )
        return value, transition

    collection = collect_antithetic_rollout(
        0.0,
        decision_indices=(2, 4),
        total_steps=6,
        step_fn=step,
        mean_fn=lambda _state, _index: torch.zeros(1, 1),
        action_from_mean=lambda mean, noise: (mean + noise, None),
        noise_by_decision={2: torch.zeros(1, 1), 4: torch.zeros(1, 1)},
        registered_decision_indices=(2, 4),
    )
    branch = collection.branches["plus"]
    assert [row.step_index for row in branch.transitions] == [2, 4]
    assert [row.step_index for row in branch.auxiliary_transitions] == [3, 5]
    assert [row.step_index for row in branch.all_transitions] == [2, 3, 4, 5]
    assert branch.actions.shape[-2] == 2


def test_antithetic_collector_requires_registered_schedule_and_can_preserve_graph():
    with pytest.raises(ValueError, match="registered_decision_indices"):
        collect_antithetic_rollout(
            0.0,
            decision_indices=(1,),
            total_steps=2,
            step_fn=lambda state, action, index: state + 1,
            mean_fn=lambda _state, _index: torch.zeros(1, 1),
            action_from_mean=lambda mean, noise: (mean + noise, None),
            noise_by_decision={1: torch.zeros(1, 1)},
            registered_decision_indices=None,
        )

    initial = torch.zeros(1, requires_grad=True)

    def differentiable_step(state, action, index):
        del index
        value = state + (torch.zeros_like(state) if action is None else action[..., :1])
        if action is None:
            return value
        return value, Transition(state, action, value, value, value, step_index=1)

    collection = collect_antithetic_rollout(
        initial,
        decision_indices=(1,),
        total_steps=2,
        step_fn=differentiable_step,
        mean_fn=lambda state, _index: state.reshape(1, 1),
        action_from_mean=lambda mean, noise: (mean + noise, None),
        noise_by_decision={1: torch.ones(1, 1)},
        registered_decision_indices=(1,),
        preserve_graph=True,
    )
    assert collection.preserve_graph is True
    assert collection.branches["plus"].transitions[0].next_state.requires_grad
    assert collection.proposals[0].plus_action.requires_grad


@pytest.mark.parametrize("bad_indices", [(2.9,), (True,), ("2",), (2, 6)])
def test_antithetic_collector_rejects_unregistered_decision_indices(bad_indices):
    def step(state, action, index):
        value = state + 1
        return value, None

    with pytest.raises(ValueError):
        collect_antithetic_rollout(
            0.0,
            decision_indices=bad_indices,
            total_steps=6,
            step_fn=step,
            mean_fn=lambda _state, _index: torch.zeros(1, 2),
            action_from_mean=lambda mean, noise: (mean + noise, None),
            noise_by_decision={2: torch.zeros(1, 2)}
            if bad_indices == (2.9,) or bad_indices == (True,) or bad_indices == ("2",)
            else {2: torch.zeros(1, 2), 6: torch.zeros(1, 2)},
            registered_decision_indices=bad_indices,
        )


def test_antithetic_collector_rejects_non_integer_noise_keys_and_unregistered_schedule():
    common = dict(
        initial_state=0.0,
        decision_indices=(2, 4),
        total_steps=6,
        step_fn=lambda state, _action, _index: state + 1,
        mean_fn=lambda _state, _index: torch.zeros(1, 2),
        action_from_mean=lambda mean, noise: (mean + noise, None),
        registered_decision_indices=(2, 4),
    )
    with pytest.raises(ValueError, match="plain integers"):
        collect_antithetic_rollout(
            **common,
            noise_by_decision={True: torch.zeros(1, 2), 4: torch.zeros(1, 2)},
        )
    mismatched = dict(common)
    mismatched["registered_decision_indices"] = (8, 24, 40)
    with pytest.raises(ValueError, match="registered schedule"):
        collect_antithetic_rollout(
            **mismatched,
            noise_by_decision={2: torch.zeros(1, 2), 4: torch.zeros(1, 2)},
        )


def test_antithetic_collector_requires_decision_step_index():
    def step(state, action, index):
        transition = Transition(
            state,
            torch.zeros(1, 2) if action is None else action,
            state + 1,
            torch.ones(1),
            torch.ones(1),
            step_index=None if action is not None else index,
        )
        return state + 1, transition

    with pytest.raises(ValueError, match="integer step_index"):
        collect_antithetic_rollout(
            0.0,
            decision_indices=(1,),
            total_steps=2,
            step_fn=step,
            mean_fn=lambda _state, _index: torch.zeros(1, 2),
            action_from_mean=lambda mean, noise: (mean + noise, None),
            noise_by_decision={1: torch.zeros(1, 2)},
            registered_decision_indices=(1,),
        )


def test_proposal_requires_explicit_branch_means():
    from latent_renderer_training.rollout import DecisionProposal

    proposal = DecisionProposal(
        step_index=2,
        mean=torch.zeros(1, 2),
        noise=torch.zeros(1, 2),
        plus_action=torch.zeros(1, 2),
        minus_action=torch.zeros(1, 2),
        anchor_action=torch.zeros(1, 2),
    )
    with pytest.raises(ValueError, match="branch-conditioned means"):
        proposal.branch_means()
    assert set(proposal.branch_means(allow_legacy_alias=True)) == {
        "plus", "minus", "anchor"
    }


def test_forward_euler_mapped_reuses_native_bytes_for_zero_action():
    renderer = _renderer()
    sample = torch.randn(1, 2, 4, 4, dtype=torch.float16)
    native_output = torch.randn_like(sample)
    endpoint = prepare_euler_clean_endpoint(
        sample,
        native_output,
        sigma_from=3.0,
        sigma_to=2.0,
        prediction_type="epsilon",
    )
    bases = torch.randn(1, 6, 2, 4, 4, dtype=sample.dtype)
    mapped = renderer.forward_euler_mapped(
        endpoint.pred_original_sample,
        bases,
        sample=sample,
        native_model_output=native_output,
        sigma_from=3.0,
        sigma_to=2.0,
        prediction_type="epsilon",
        scheduler_update=endpoint.nominal_update,
        prompt_embedding=torch.zeros(1, 32),
        state_features=torch.zeros(1, 16),
    )
    native_prev = predict_euler_no_churn_prev_sample(
        sample,
        native_output,
        sigma_from=3.0,
        sigma_to=2.0,
        prediction_type="epsilon",
    )
    assert torch.equal(mapped.model_output, native_output)
    assert torch.equal(mapped.predicted_prev_sample, native_prev)
    assert torch.equal(mapped.mapped_intervention.ratio, torch.zeros(1))


@pytest.mark.parametrize("prediction_type", ("epsilon", "sample", "v_prediction"))
def test_forward_euler_mapped_supports_per_batch_sigmas(prediction_type):
    """A batched collector may carry one Euler interval per prompt row."""
    renderer = _renderer()
    torch.manual_seed(812)
    sample = torch.randn(2, 2, 4, 4, dtype=torch.float32)
    native_output = torch.randn_like(sample)
    sigma_from = torch.tensor([3.0, 2.0])
    sigma_to = torch.tensor([2.0, 1.0])
    source = sigma_from.reshape(-1, 1, 1, 1)
    target = sigma_to.reshape(-1, 1, 1, 1)
    if prediction_type == "epsilon":
        clean = sample - source * native_output
    elif prediction_type in {"sample", "original_sample"}:
        clean = native_output
    else:
        clean = (
            native_output * (-source / torch.sqrt(source.square() + 1.0))
            + sample / (source.square() + 1.0)
        )
    nominal = ((sample - clean) / source) * (target - source)
    bases = torch.randn(2, 6, 2, 4, 4)
    mapped = renderer.forward_euler_mapped(
        clean,
        bases,
        sample=sample,
        native_model_output=native_output,
        sigma_from=sigma_from,
        sigma_to=sigma_to,
        prediction_type=prediction_type,
        scheduler_update=nominal,
        prompt_embedding=torch.zeros(2, 32),
        state_features=torch.zeros(2, 16),
        action=torch.zeros(2, 6),
    )
    assert torch.equal(mapped.model_output, native_output)
    torch.testing.assert_close(mapped.predicted_prev_sample, sample + nominal)
    assert mapped.predicted_prev_sample.shape == sample.shape
    assert torch.equal(mapped.mapped_intervention.ratio, torch.zeros(2))


def test_zero_action_does_not_enter_clean_endpoint_converter(monkeypatch):
    renderer = _renderer()
    sample = torch.randn(1, 2, 4, 4, dtype=torch.float64)
    native_output = torch.randn_like(sample)
    endpoint = prepare_euler_clean_endpoint(
        sample,
        native_output,
        sigma_from=3.0,
        sigma_to=2.0,
        prediction_type="epsilon",
    )
    bases = torch.randn(1, 6, 2, 4, 4, dtype=sample.dtype)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("zero action entered clean endpoint conversion")

    monkeypatch.setattr(
        renderer_module, "euler_model_output_from_clean_sample", fail_if_called
    )
    mapped = renderer.forward_euler_mapped(
        endpoint.pred_original_sample,
        bases,
        sample=sample,
        native_model_output=native_output,
        sigma_from=3.0,
        sigma_to=2.0,
        prediction_type="epsilon",
        scheduler_update=endpoint.nominal_update,
        prompt_embedding=torch.zeros(1, 32, dtype=sample.dtype),
        state_features=torch.zeros(1, 16, dtype=sample.dtype),
    )
    assert torch.equal(mapped.model_output, native_output)
    assert torch.equal(mapped.rendered.guided_x0, endpoint.pred_original_sample)
    assert mapped.rendered.guided_x0.dtype == torch.float64


def test_zero_action_keeps_a_native_transition_gradient_for_opd():
    renderer = _renderer()
    clean, raw, native, state = _state(renderer, batch=1)
    rendered = renderer.apply_coefficients(state, renderer.deterministic_action(state))
    target = clean + torch.randn_like(clean)
    loss = (rendered.guided_x0 - target).square().mean()
    loss.backward()
    assert rendered.guided_x0.requires_grad
    assert renderer.policy[-1].bias.grad is not None
    assert bool(torch.isfinite(renderer.policy[-1].bias.grad).all())
    assert float(renderer.policy[-1].bias.grad.abs().sum()) > 0.0


def test_mixed_batch_zero_rows_keep_dtype_and_exact_clean_bytes():
    renderer = _renderer()
    clean, raw, native, state = _state(renderer, batch=2)
    clean64 = clean.double()
    raw64 = raw.double()
    native64 = native.double()
    observation = RendererObservation(
        latents_before_step=clean64,
        pred_original_sample=clean64,
        scheduler_update=native64,
        step_index=8,
        timestep=torch.full((2,), 400.0, dtype=torch.float64),
        normalized_timestep=torch.full((2,), 8.0 / 49.0, dtype=torch.float64),
    )
    condition = RendererCondition(
        bases=raw64,
        prompt_embedding=torch.zeros(2, 32, dtype=torch.float64),
        state_features=torch.zeros(2, 16, dtype=torch.float64),
    )
    state64 = renderer.prepare_state(
        observation, condition, sigma_from=1.0, sigma_to=0.8
    )
    action = torch.zeros(2, 6, dtype=torch.float64)
    action[1, 0] = 0.2
    output = renderer.apply_coefficients(state64, action)
    assert output.guided_x0.dtype == torch.float64
    assert torch.equal(output.guided_x0[0], clean64[0])
    assert torch.equal(output.residual[0], torch.zeros_like(clean64[0]))


def test_euler_mapped_accepts_sampled_action_without_float32_zero_alias(monkeypatch):
    renderer = _renderer()
    sample = torch.randn(1, 2, 4, 4, dtype=torch.float64)
    native_output = torch.randn_like(sample)
    endpoint = prepare_euler_clean_endpoint(
        sample,
        native_output,
        sigma_from=3.0,
        sigma_to=2.0,
        prediction_type="epsilon",
    )
    bases = torch.randn(1, 6, 2, 4, 4, dtype=sample.dtype)
    calls = []
    original = renderer_module.euler_model_output_from_clean_sample

    def record_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        renderer_module, "euler_model_output_from_clean_sample", record_call
    )
    tiny_action = torch.full((1, 6), 1e-50, dtype=torch.float64)
    mapped = renderer.forward_euler_mapped(
        endpoint.pred_original_sample,
        bases,
        sample=sample,
        native_model_output=native_output,
        sigma_from=3.0,
        sigma_to=2.0,
        prediction_type="epsilon",
        scheduler_update=endpoint.nominal_update,
        prompt_embedding=torch.zeros(1, 32, dtype=sample.dtype),
        state_features=torch.zeros(1, 16, dtype=sample.dtype),
        action=tiny_action,
    )
    assert calls, "a representable float64 action must not alias the zero path"
    assert torch.any(mapped.rendered.coefficients != 0)
