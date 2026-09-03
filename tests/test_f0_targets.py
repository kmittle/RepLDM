import copy
from dataclasses import replace
import io

import pytest
import torch

from latent_renderer_training.f0_targets import (
    F0_DECISION_INDICES,
    F0_SOURCES,
    F0_STRATA,
    F0_TRAINING_SEEDS,
    F0TargetRow,
    f0_crossfit_splits,
    f0_objective,
    f0_target_from_payload,
    f0_target_to_payload,
    validate_f0_rows,
)
from tests.test_training_method_objectives import _frame, _renderer


def _frozen_copy(renderer):
    result = copy.deepcopy(renderer)
    result.eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def _row(
    renderer,
    *,
    prompt_id="prompt-0",
    seed=F0_TRAINING_SEEDS[0],
    decision=8,
    source=F0_SOURCES[0],
    stratum=F0_STRATA[0],
    fold=0,
    pair_weight=1.0,
    target_offset=0.2,
    valid=True,
):
    state = _frame(renderer, decision, 0.0)
    with torch.no_grad():
        anchor = renderer.action_parameters(state).detach()
        reference = anchor.clone()
        active = torch.tensor(renderer.active_mask, dtype=torch.bool)
        gradient = torch.zeros_like(anchor)
        gradient[..., active] = 1.0
        gradient = gradient / torch.linalg.vector_norm(gradient[..., active], dim=-1, keepdim=True)
        plus_u = anchor + target_offset * gradient
        minus_u = anchor - target_offset * gradient

        def transition(value):
            action = renderer.deterministic_action_from_mean(value)
            output = renderer.apply_coefficients(state, action)
            return (
                state.native_update
                + state.kappa.reshape(-1, 1, 1, 1) * output.residual
            ).detach()

        plus = transition(plus_u)
        minus = transition(minus_u)
    return F0TargetRow(
        prompt_id=prompt_id,
        generation_seed=seed,
        decision_index=decision,
        split="train",
        source=source,
        stratum=stratum,
        fold=fold,
        state=state,
        anchor_u=anchor,
        plus_u=plus_u,
        minus_u=minus_u,
        reward_gradient=gradient,
        plus_transition=plus,
        minus_transition=minus,
        reference_u=reference,
        pair_weight=pair_weight,
        valid=valid,
    )


def test_f0_target_detaches_and_does_not_alias_inputs():
    renderer = _renderer()
    state = _frame(renderer, 8, 0.0)
    anchor = renderer.action_parameters(state).detach()
    row = _row(renderer)
    original = row.anchor_u.clone()
    anchor.add_(10.0)
    state.native_update.add_(10.0)
    assert torch.equal(row.anchor_u, original)
    assert not torch.equal(row.state.native_update, state.native_update)
    assert not row.state.native_update.requires_grad
    assert len(row.record_sha256) == 64


def test_f0_target_weights_only_round_trip_binds_every_tensor():
    row = _row(_renderer())
    buffer = io.BytesIO()
    torch.save(f0_target_to_payload(row), buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = f0_target_from_payload(loaded)
    assert restored.record_sha256 == row.record_sha256
    assert torch.equal(restored.state.raw_bases, row.state.raw_bases)
    assert torch.equal(restored.plus_transition, row.plus_transition)

    loaded["state"]["raw_bases"][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="record hash"):
        f0_target_from_payload(loaded)


def test_f0_target_rejects_out_of_trust_and_inactive_coordinates():
    renderer = _renderer()
    with pytest.raises(ValueError, match="trust radius"):
        _row(renderer, target_offset=0.6)
    row = _row(renderer)
    state = replace(
        row.state,
        diagnostics=replace(
            row.state.diagnostics,
            active_mask=(True, True, True, True, True, False),
        ),
    )
    with pytest.raises(ValueError, match="inactive coordinate"):
        replace(row, state=state)


def test_f0_objective_is_native_transition_loss_and_has_policy_gradient():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    row = _row(renderer, pair_weight=1.0)
    initial = f0_objective(renderer, reference, [row])
    assert initial.item() > 0
    initial.backward()
    gradient = renderer.policy[-1].bias.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    renderer.zero_grad(set_to_none=True)
    invalid = _row(renderer, valid=False)
    zero = f0_objective(renderer, reference, [invalid])
    assert zero.item() == 0.0
    zero.backward()
    assert torch.count_nonzero(renderer.policy[-1].bias.grad) == 0


def test_f0_objective_rejects_reference_drift_and_action_regression_shortcut():
    renderer = _renderer()
    reference = _frozen_copy(renderer)
    row = _row(renderer)
    with torch.no_grad():
        reference.policy[-1].bias.add_(0.1)
    with pytest.raises(ValueError, match="reference mean"):
        f0_objective(renderer, reference, [row])

    reference = _frozen_copy(renderer)
    changed_target = row.plus_transition + 1.0
    changed = copy.copy(row)
    object.__setattr__(changed, "plus_transition", changed_target)
    assert f0_objective(renderer, reference, [changed]) > f0_objective(
        renderer, reference, [row]
    )


def _complete_training_rows(renderer):
    rows = []
    for source_index, source in enumerate(F0_SOURCES):
        for stratum_index, stratum in enumerate(F0_STRATA):
            for fold in range(4):
                prompt_id = f"{source}-{stratum}-{fold}"
                for seed in F0_TRAINING_SEEDS:
                    for decision in F0_DECISION_INDICES:
                        rows.append(
                            _row(
                                renderer,
                                prompt_id=prompt_id,
                                seed=seed,
                                decision=decision,
                                source=source,
                                stratum=stratum,
                                fold=fold,
                                target_offset=0.0,
                            )
                        )
    return rows


def test_f0_crossfit_matrix_is_exact_and_stratified():
    renderer = _renderer()
    rows = _complete_training_rows(renderer)
    assert len(validate_f0_rows(rows, split="train")) == 384
    splits = f0_crossfit_splits(rows)
    assert [item.held_out_fold for item in splits] == [0, 1, 2, 3]
    assert all(len(item.fit_rows) == 288 for item in splits)
    assert all(len(item.held_out_rows) == 96 for item in splits)
    held_out_ids = [
        {row.prompt_id for row in item.held_out_rows} for item in splits
    ]
    assert all(len(values) == 16 for values in held_out_ids)
    assert not any(
        held_out_ids[left] & held_out_ids[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )


def test_f0_matrix_rejects_missing_duplicate_and_unstratified_rows():
    renderer = _renderer()
    rows = _complete_training_rows(renderer)
    with pytest.raises(ValueError, match="exactly 384"):
        validate_f0_rows(rows[:-1], split="train")
    with pytest.raises(ValueError, match="duplicate"):
        validate_f0_rows([*rows[:-1], rows[0]], split="train")

    changed = list(rows)
    victim = changed[0]
    object.__setattr__(victim, "fold", 1)
    with pytest.raises(ValueError, match="metadata changes|not stratified"):
        validate_f0_rows(changed, split="train")
