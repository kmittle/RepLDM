from __future__ import annotations

from dataclasses import dataclass, replace
import io

import torch
from torch import nn

from AttentionGuidance.latent_renderer import (
    RendererCondition,
    RendererObservation,
    predict_euler_no_churn_prev_sample,
    prepare_euler_clean_endpoint,
)
from latent_renderer_training.collector import (
    EulerNativeRolloutCollector,
    EulerRolloutStep,
    EulerTransitionResult,
)
from latent_renderer_training.renderer import (
    CALIBRATION_STATE_COUNT,
    EulerNativeFrameV1,
    FrameCalibration,
)


CHECKPOINT_HASH = "9" * 64
DECISIONS = (8, 24, 40)


@dataclass(frozen=True)
class _State:
    latents: torch.Tensor
    step_index: int = 0
    total_steps: int = 50
    branch: str = "prefix"
    checkpoint_hash: str = CHECKPOINT_HASH
    action_history: tuple[object, ...] = ()


class _UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, latents: torch.Tensor, index: int) -> torch.Tensor:
        self.forward_calls += 1
        return 0.03 * latents + 0.001 * float(index + 1)


def _renderer() -> EulerNativeFrameV1:
    calibration = FrameCalibration(
        (True, True, False, False, False, False),
        CALIBRATION_STATE_COUNT,
        (1.0,) * 6,
        (1.0,) * 6,
        manifest_sha256="a" * 64,
        source_sha256="b" * 64,
        state_provenance_sha256="c" * 64,
    )
    return EulerNativeFrameV1(
        calibration=calibration,
        latent_channels=2,
        hidden_dim=8,
        depth=1,
        prompt_dim=0,
        state_dim=0,
        timestep_dim=4,
    )


class _Harness:
    def __init__(self) -> None:
        self.unet = _UNet()
        self.observed: list[int] = []
        self.transitioned: list[dict[str, object]] = []
        pattern_a = torch.tensor(
            [[1.0, -1.0, 1.0, -1.0]] * 4,
        )
        pattern_b = torch.tensor(
            [[1.0] * 4, [-1.0] * 4, [1.0] * 4, [-1.0] * 4],
        )
        self.bases = torch.zeros(1, 6, 2, 4, 4)
        self.bases[0, 0, 0] = pattern_a
        self.bases[0, 0, 1] = pattern_b
        self.bases[0, 1, 0] = pattern_b
        self.bases[0, 1, 1] = -pattern_a

    def observe(self, state: _State, index: int) -> EulerRolloutStep:
        self.observed.append(index)
        native_output = self.unet(state.latents, index)
        sigma_from = float(50 - index)
        sigma_to = float(49 - index)
        endpoint = prepare_euler_clean_endpoint(
            state.latents,
            native_output,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            prediction_type="epsilon",
        )
        observation = RendererObservation(
            latents_before_step=state.latents.detach(),
            pred_original_sample=endpoint.pred_original_sample.detach(),
            scheduler_update=endpoint.nominal_update.detach(),
            step_index=index,
            timestep=torch.tensor(float(index)),
            normalized_timestep=torch.tensor(index / 49.0),
        )
        return EulerRolloutStep(
            observation=observation,
            condition=RendererCondition(bases=self.bases.detach()),
            native_model_output=native_output.detach(),
            native_prev_sample=(state.latents + endpoint.nominal_update).detach(),
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            prediction_type="epsilon",
            clean_update_gain=endpoint.clean_update_gain.detach(),
        )

    def transition(
        self,
        state: _State,
        index: int,
        model_output: torch.Tensor,
        step: EulerRolloutStep,
    ) -> EulerTransitionResult:
        action = getattr(self, "pending_action", None)
        latent = predict_euler_no_churn_prev_sample(
            state.latents,
            model_output,
            sigma_from=step.sigma_from,
            sigma_to=step.sigma_to,
            prediction_type=step.prediction_type,
        )
        self.transitioned.append(
            {
                "index": index,
                "action": None if action is None else action.detach().clone(),
                "native_bytes": torch.equal(model_output, step.native_model_output),
                "branch": state.branch,
                "checkpoint_hash": state.checkpoint_hash,
            }
        )
        history = state.action_history
        if action is not None:
            history = history + (action.detach().cpu().tolist(),)
        next_state = replace(
            state,
            latents=latent,
            step_index=index + 1,
            action_history=history,
        )
        return EulerTransitionResult(state=next_state, latent=latent)

    def transition_with_action(
        self,
        state: _State,
        index: int,
        model_output: torch.Tensor,
        step: EulerRolloutStep,
        *,
        action: torch.Tensor | None,
    ) -> EulerTransitionResult:
        self.pending_action = action
        try:
            return self.transition(state, index, model_output, step)
        finally:
            del self.pending_action

    def clear_records(self) -> None:
        self.observed.clear()
        self.transitioned.clear()


class _ActionAwareCollector(EulerNativeRolloutCollector):
    """Exercise adapter-style action forwarding without constructing SDXL."""

    def _transition(
        self,
        state: _State,
        index: int,
        model_output: torch.Tensor,
        step: EulerRolloutStep,
        *,
        action: torch.Tensor | None,
    ) -> EulerTransitionResult:
        return self._test_harness.transition_with_action(
            state,
            index,
            model_output,
            step,
            action=action,
        )


def _setup() -> tuple[_ActionAwareCollector, _Harness, _State]:
    harness = _Harness()
    collector = _ActionAwareCollector(
        _renderer(),
        decision_indices=DECISIONS,
        registered_decision_indices=DECISIONS,
        total_steps=50,
        observe_fn=harness.observe,
        transition_fn=harness.transition,
        physical_unet=harness.unet,
    )
    collector._test_harness = harness
    latents = torch.linspace(-1.0, 1.0, 32).reshape(1, 2, 4, 4)
    return collector, harness, _State(latents=latents)


def test_f0_anchor_is_exact_fifty_step_c0_trace_with_independent_cache() -> None:
    collector, harness, initial = _setup()
    trace = collector.collect_f0_anchor_trace(initial)

    assert trace.stats.observe_calls == 50
    assert trace.stats.transition_calls == 50
    assert trace.stats.decision_transitions == 3
    assert trace.stats.auxiliary_transitions == 47
    assert trace.stats.verified_unet_forwards == 50
    assert harness.unet.forward_calls == 50
    assert harness.observed == list(range(50))
    assert set(trace.cached_decisions) == set(DECISIONS)
    assert trace.terminal_state.step_index == 50
    assert trace.terminal_state.branch == "anchor"
    assert trace.terminal_state.checkpoint_hash == CHECKPOINT_HASH

    for record in harness.transitioned:
        index = record["index"]
        if index in DECISIONS:
            action = record["action"]
            assert isinstance(action, torch.Tensor)
            assert torch.count_nonzero(action) == 0
        else:
            assert record["action"] is None
        assert record["native_bytes"] is True

    assert trace.initial_state.latents.data_ptr() != initial.latents.data_ptr()
    cache_ptrs = []
    for index, cached in trace.cached_decisions.items():
        assert cached.state.step_index == index
        assert cached.checkpoint_hash == CHECKPOINT_HASH
        assert cached.state.latents.data_ptr() != initial.latents.data_ptr()
        assert (
            cached.state.latents.data_ptr()
            != cached.step.observation.latents_before_step.data_ptr()
        )
        assert (
            cached.step.observation.latents_before_step.data_ptr()
            != cached.frame.sample.data_ptr()
        )
        assert bool(cached.frame.diagnostics.valid.all())
        cache_ptrs.append(cached.state.latents.data_ptr())
    assert len(set(cache_ptrs)) == 3


def test_cached_suffix_budgets_zero_later_decisions_and_preserve_provenance() -> None:
    collector, harness, initial = _setup()
    trace = collector.collect_f0_anchor(initial)
    harness.clear_records()
    starting_forwards = harness.unet.forward_calls
    expected = {8: (41, 42), 24: (25, 26), 40: (9, 10)}
    results = []

    for decision, (observe_calls, transition_calls) in expected.items():
        harness.clear_records()
        action = torch.tensor([[0.20, -0.15, 0.0, 0.0, 0.0, 0.0]])
        result = collector.replay_f0_suffix(
            trace,
            action,
            decision_index=decision,
            branch=f"target-plus-{decision}",
        )
        results.append(result)

        assert result.stats.observe_calls == observe_calls
        assert result.stats.transition_calls == transition_calls
        assert result.stats.verified_unet_forwards == observe_calls
        assert harness.observed == list(range(decision + 1, 50))
        assert decision not in harness.observed
        assert result.initial_state.branch == f"target-plus-{decision}"
        assert result.terminal_state.branch == f"target-plus-{decision}"
        assert result.terminal_state.checkpoint_hash == CHECKPOINT_HASH
        assert result.terminal_state.step_index == 50
        assert result.initial_state.latents.data_ptr() != trace.cache[decision].state.latents.data_ptr()
        assert result.terminal_state.latents.data_ptr() != trace.cache[decision].state.latents.data_ptr()
        assert torch.equal(result.applied_action, action)

        selected = harness.transitioned[0]
        assert selected["index"] == decision
        assert torch.equal(selected["action"], action)
        for later in (value for value in DECISIONS if value > decision):
            record = next(item for item in harness.transitioned if item["index"] == later)
            assert isinstance(record["action"], torch.Tensor)
            assert torch.count_nonzero(record["action"]) == 0
            assert record["native_bytes"] is True
        assert all(item["branch"] == f"target-plus-{decision}" for item in harness.transitioned)
        assert all(item["checkpoint_hash"] == CHECKPOINT_HASH for item in harness.transitioned)

    assert sum(item.stats.observe_calls for item in results) == 75
    assert sum(item.stats.transition_calls for item in results) == 78
    assert harness.unet.forward_calls - starting_forwards == 75


def test_invalid_cached_frame_forces_exact_native_selected_transition() -> None:
    collector, harness, initial = _setup()
    trace = collector.collect_f0_anchor_trace(initial)
    cached = trace.cache[24]
    invalid_diagnostics = replace(
        cached.frame.diagnostics,
        valid=torch.zeros_like(cached.frame.diagnostics.valid),
    )
    invalid = replace(
        cached,
        frame=replace(cached.frame, diagnostics=invalid_diagnostics),
    )
    action = torch.tensor([[0.25, -0.20, 0.0, 0.0, 0.0, 0.0]])

    harness.clear_records()
    invalid_result = collector.replay_cached_suffix(
        invalid,
        action,
        branch="invalid-frame",
    )
    selected = harness.transitioned[0]
    assert selected["native_bytes"] is True
    assert torch.count_nonzero(selected["action"]) == 0
    assert torch.equal(
        invalid_result.selected_transition.rendered_transition,
        invalid_result.selected_transition.native_transition,
    )
    assert torch.equal(invalid_result.requested_action, action)
    assert torch.count_nonzero(invalid_result.applied_action) == 0
    assert bool(invalid_result.fallback_mask.all())

    harness.clear_records()
    native_result = collector.replay_cached_suffix(
        cached,
        torch.zeros_like(action),
        branch="native-control",
    )
    assert torch.equal(
        invalid_result.terminal_state.latents,
        native_result.terminal_state.latents,
    )
    assert cached.state.latents.data_ptr() != invalid_result.initial_state.latents.data_ptr()


def test_f0_cache_can_be_serialized_and_replayed_after_restart() -> None:
    collector, harness, initial = _setup()
    trace = collector.collect_f0_anchor_trace(initial)
    payload = io.BytesIO()
    torch.save(trace, payload)
    payload.seek(0)
    restored = torch.load(payload, weights_only=False)

    assert set(restored.cached_decisions) == set(DECISIONS)
    assert restored.checkpoint_hash == CHECKPOINT_HASH
    harness.clear_records()
    result = collector.replay_f0_suffix(
        restored,
        torch.zeros(1, 6),
        decision_index=40,
        branch="restored-suffix",
    )
    assert result.stats.observe_calls == 9
    assert result.stats.transition_calls == 10
    assert result.terminal_state.branch == "restored-suffix"
