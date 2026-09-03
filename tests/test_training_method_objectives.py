import copy
from dataclasses import replace

import pytest
import torch

from AttentionGuidance.latent_renderer import RendererCondition, RendererObservation
from latent_renderer_training.methods.common import (
    ScoredRollout,
    dpo_objective,
    opd_objective,
    rl_objective,
    search_distill_objective,
)
from latent_renderer_training.methods.runner import (
    OPTIMIZATION_SEEDS,
    _batches,
    _blocks,
    _prompt_records,
    _proposal_noise,
)
from latent_renderer_training.preferences import PreferenceLabelProvenance
from latent_renderer_training.renderer import (
    CALIBRATION_STATE_COUNT,
    EulerNativeFrameV1,
    FrameCalibration,
)
from latent_renderer_training.rollout import (
    BranchTrajectory,
    DecisionProposal,
    RolloutCollection,
    Transition,
)


def _renderer() -> EulerNativeFrameV1:
    calibration = FrameCalibration(
        (True,) * 6,
        CALIBRATION_STATE_COUNT,
        (1.0,) * 6,
        (1.0,) * 6,
        manifest_sha256="a" * 64,
        source_sha256="b" * 64,
        state_provenance_sha256="c" * 64,
    )
    return EulerNativeFrameV1(calibration=calibration, latent_channels=2)


def _frame(renderer: EulerNativeFrameV1, step_index: int, offset: float):
    generator = torch.Generator().manual_seed(1000 + step_index + int(offset * 10))
    clean = torch.randn(1, 2, 5, 5, generator=generator) + offset
    native = torch.randn(1, 2, 5, 5, generator=generator)
    bases = torch.randn(1, 6, 2, 5, 5, generator=generator)
    observation = RendererObservation(
        latents_before_step=clean,
        pred_original_sample=clean,
        scheduler_update=native,
        step_index=step_index,
        timestep=torch.tensor([float(49 - step_index)]),
        normalized_timestep=torch.tensor([step_index / 49.0]),
    )
    condition = RendererCondition(
        bases=bases,
        prompt_embedding=torch.randn(1, 32, generator=generator),
        state_features=torch.randn(1, 16, generator=generator),
    )
    state = renderer.prepare_state(
        observation,
        condition,
        sigma_from=1.0,
        sigma_to=0.8,
        native_model_output=torch.zeros_like(clean),
    )
    assert bool(state.diagnostics.valid.all())
    return state


def _scored(
    behavior: EulerNativeFrameV1,
    reference: EulerNativeFrameV1,
    *,
    plus_reward: float = 1.0,
    minus_reward: float = 0.0,
) -> ScoredRollout:
    branches = {
        name: BranchTrajectory(name) for name in ("plus", "minus", "anchor")
    }
    proposals = []
    for position, step_index in enumerate((8, 24, 40)):
        states = {
            branch: _frame(behavior, step_index, position + branch_index / 10.0)
            for branch_index, branch in enumerate(("plus", "minus", "anchor"))
        }
        means = {
            branch: behavior.action_parameters(state).detach()
            for branch, state in states.items()
        }
        noise = torch.full_like(means["anchor"], 0.4 + 0.1 * position)
        actions = {}
        for branch, signed_noise in (
            ("plus", noise),
            ("minus", -noise),
            ("anchor", torch.zeros_like(noise)),
        ):
            distribution = behavior.action_distribution(means[branch])
            action, pre_squash = distribution.rsample(signed_noise)
            output = behavior.apply_coefficients(states[branch], action)
            update = states[branch].native_update + states[branch].kappa.reshape(
                -1, 1, 1, 1
            ) * output.residual
            reference_mean = reference.action_parameters(states[branch]).detach()
            transition = Transition(
                state=states[branch],
                action=action.detach(),
                next_state={"branch": branch, "step": step_index + 1},
                native_transition=states[branch].native_update.detach(),
                rendered_transition=update.detach(),
                log_prob=(
                    None
                    if branch == "anchor"
                    else distribution.log_prob_from_pre_squash(pre_squash).detach()
                ),
                step_index=step_index,
                pre_squash=pre_squash.detach(),
                behavior_mean=means[branch],
                reference_mean=reference_mean,
            )
            branches[branch].append(transition)
            actions[branch] = (action.detach(), pre_squash.detach())
        proposals.append(
            DecisionProposal(
                step_index=step_index,
                mean=means["anchor"],
                noise=noise,
                plus_action=actions["plus"][0],
                minus_action=actions["minus"][0],
                anchor_action=actions["anchor"][0],
                plus_mean=means["plus"],
                minus_mean=means["minus"],
                anchor_mean=means["anchor"],
                plus_pre_squash=actions["plus"][1],
                minus_pre_squash=actions["minus"][1],
                anchor_pre_squash=actions["anchor"][1],
            )
        )
    collection = RolloutCollection(
        branches=branches,
        proposals=proposals,
        prefix_steps=8,
        total_steps=50,
        terminal_states={name: {"branch": name} for name in branches},
    )
    rewards = {
        "plus": torch.tensor([plus_reward]),
        "minus": torch.tensor([minus_reward]),
        "anchor": torch.tensor([0.25]),
    }
    label = PreferenceLabelProvenance.from_rewards(
        rollout_id="unit-rollout",
        prompt_id="unit-prompt",
        generation_seed=2026090101,
        split="train",
        plus_reward=float(rewards["plus"].double()),
        minus_reward=float(rewards["minus"].double()),
        reward_location=0.25,
        reward_scale=0.5,
        plus_image_sha256="1" * 64,
        minus_image_sha256="2" * 64,
        reward_statistics_sha256="3" * 64,
        reward_config_sha256="4" * 64,
        reward_preprocess_sha256="5" * 64,
    )
    return ScoredRollout(collection, rewards, label)


def _frozen_copy(renderer: EulerNativeFrameV1) -> EulerNativeFrameV1:
    result = copy.deepcopy(renderer)
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    result.eval()
    return result


def _last_gradient(renderer: EulerNativeFrameV1, loss: torch.Tensor) -> torch.Tensor:
    renderer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = renderer.policy[-1].bias.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    return gradient.detach().clone()


def test_search_distill_uses_the_terminal_winner_and_ties_have_zero_loss():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    rollout = _scored(reference, reference, plus_reward=1.0, minus_reward=0.0)
    plus_gradient = _last_gradient(
        renderer, search_distill_objective(renderer, reference, [rollout])
    )

    renderer = _renderer()
    reference = _frozen_copy(renderer)
    rollout = _scored(reference, reference, plus_reward=0.0, minus_reward=1.0)
    minus_gradient = _last_gradient(
        renderer, search_distill_objective(renderer, reference, [rollout])
    )
    assert not torch.equal(plus_gradient, minus_gradient)

    renderer = _renderer()
    reference = _frozen_copy(renderer)
    tie = _scored(reference, reference, plus_reward=1.0, minus_reward=1.0)
    loss = search_distill_objective(renderer, reference, [tie])
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert torch.equal(_last_gradient(renderer, loss), torch.zeros(6))


def test_dpo_label_swap_reverses_the_policy_gradient():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    rollout = _scored(reference, reference, plus_reward=1.0, minus_reward=0.0)
    positive = _last_gradient(renderer, dpo_objective(renderer, reference, [rollout]))

    renderer = _renderer()
    reference = _frozen_copy(renderer)
    rollout = _scored(reference, reference, plus_reward=0.0, minus_reward=1.0)
    negative = _last_gradient(renderer, dpo_objective(renderer, reference, [rollout]))
    torch.testing.assert_close(positive, -negative, atol=2e-6, rtol=2e-6)


def test_rl_uses_per_decision_ratios_and_zeroes_tied_advantages():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    rollout = _scored(reference, reference, plus_reward=1.0, minus_reward=0.0)
    loss, ratios, kl = rl_objective(renderer, reference, [rollout])
    assert ratios.shape == (2, 3)
    torch.testing.assert_close(ratios, torch.ones_like(ratios))
    torch.testing.assert_close(kl, torch.zeros_like(kl))
    gradient = _last_gradient(renderer, loss)
    assert torch.count_nonzero(gradient) > 0

    renderer = _renderer()
    reference = _frozen_copy(renderer)
    tied = _scored(reference, reference, plus_reward=1.0, minus_reward=1.0)
    tie_loss, _ratios, _kl = rl_objective(renderer, reference, [tied])
    torch.testing.assert_close(tie_loss, torch.zeros_like(tie_loss))
    assert torch.equal(_last_gradient(renderer, tie_loss), torch.zeros(6))


def test_pair_objectives_reject_rewards_changed_after_labeling():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    rollout = _scored(reference, reference, plus_reward=1.0, minus_reward=0.0)
    rollout.rewards["plus"].fill_(-1.0)

    with pytest.raises(ValueError, match="label rewards"):
        dpo_objective(renderer, reference, [rollout])
    with pytest.raises(ValueError, match="label rewards"):
        search_distill_objective(renderer, reference, [rollout])
    with pytest.raises(ValueError, match="label rewards"):
        rl_objective(renderer, reference, [rollout])


def test_preference_label_rejects_changed_normalization_or_outcome():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    label = _scored(reference, reference).preference_label.to_dict()
    label["reward_scale"] = 0.75
    with pytest.raises(ValueError, match="normalized rewards"):
        PreferenceLabelProvenance.from_mapping(label)

    label = _scored(reference, reference).preference_label.to_dict()
    label["label"] = "minus"
    with pytest.raises(ValueError, match="tie rule"):
        PreferenceLabelProvenance.from_mapping(label)


def test_opd_teacher_is_detached_and_trains_on_student_anchor_states():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    teacher = _frozen_copy(renderer)
    with torch.no_grad():
        teacher.policy[-1].bias.fill_(0.35)
    rollout = _scored(reference, reference)
    loss = opd_objective(renderer, teacher, reference, [rollout])
    assert float(loss.detach()) > 0
    gradient = _last_gradient(renderer, loss)
    assert torch.count_nonzero(gradient) > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_opd_objective_rejects_teacher_out_of_eval_mode():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    teacher = _frozen_copy(renderer)
    teacher.train()
    rollout = _scored(reference, reference)

    with pytest.raises(ValueError, match="eval mode"):
        opd_objective(renderer, teacher, reference, [rollout])


def test_opd_objective_rejects_teacher_calibration_or_action_contract_drift():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    rollout = _scored(reference, reference)

    teacher = _frozen_copy(renderer)
    teacher.calibration = replace(teacher.calibration, manifest_sha256="d" * 64)
    with pytest.raises(ValueError, match="calibration hashes"):
        opd_objective(renderer, teacher, reference, [rollout])

    teacher = _frozen_copy(renderer)
    teacher.contract = replace(teacher.contract, pre_squash_sigma=0.5)
    with pytest.raises(ValueError, match="action contracts"):
        opd_objective(renderer, teacher, reference, [rollout])


def test_opd_objective_rejects_teacher_state_storage_alias():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    teacher = _frozen_copy(renderer)
    # Keep the teacher frozen while deliberately sharing one parameter's
    # backing storage with the trainable renderer.
    with torch.no_grad():
        teacher.policy[0].weight.data = renderer.policy[0].weight.data
    rollout = _scored(reference, reference)

    with pytest.raises(ValueError, match="storage aliases"):
        opd_objective(renderer, teacher, reference, [rollout])


def test_objectives_reject_reference_state_drift():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    rollout = _scored(reference, reference)
    with torch.no_grad():
        reference.policy[-1].bias.add_(0.1)
    with pytest.raises(ValueError, match="recorded reference mean"):
        dpo_objective(renderer, reference, [rollout])


def _selected_rows():
    rows = []
    for source in ("four_k_lsdb", "pixverve_95k"):
        for stratum in (
            "nature", "urban", "people", "food", "artwork", "cgi",
            "animals", "architecture",
        ):
            for fold in range(4):
                rows.append(
                    {
                        "id": f"{source}:{stratum}:train:{fold}",
                        "model_prompt": f"{source} {stratum} training prompt {fold}",
                        "selected_split": "train",
                        "source": source,
                        "stratum": stratum,
                        "fold": fold,
                    }
                )
            for index in range(2):
                rows.append(
                    {
                        "id": f"{source}:{stratum}:validation:{index}",
                        "model_prompt": f"{source} {stratum} validation prompt {index}",
                        "selected_split": "validation",
                        "source": source,
                        "stratum": stratum,
                        "fold": None,
                    }
                )
    return rows


def test_pair_schedule_is_exact_and_proposal_noise_is_replayable():
    records = _prompt_records(reversed(_selected_rows()))
    train = _blocks(records, split="train")
    validation = _blocks(records, split="validation")
    assert len(train) == 128
    assert len(validation) == 64
    assert len(_batches(train)) == 32
    assert len({block.stable_id for block in (*train, *validation)}) == 192

    block = train[0]
    first = _proposal_noise(
        block,
        8,
        optimization_seed=OPTIMIZATION_SEEDS[0],
        active_mask=(True, False, True, True, False, True),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    replay = _proposal_noise(
        block,
        8,
        optimization_seed=OPTIMIZATION_SEEDS[0],
        active_mask=(True, False, True, True, False, True),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    other_seed = _proposal_noise(
        block,
        8,
        optimization_seed=OPTIMIZATION_SEEDS[1],
        active_mask=(True, False, True, True, False, True),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(first, replay)
    assert not torch.equal(first, other_seed)
    assert torch.equal(first[:, [1, 4]], torch.zeros(1, 2))
