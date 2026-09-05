"""Formal construction of one immutable F0 reward-gradient target.

This module is the narrow bridge between a cached strict-C0 Euler decision,
the budgeted SDXL reward adapter, and the detached target format used by F0
fitting.  Candidate backtracking is deliberately tensor-only: image decoding
and reward evaluation happen once at the anchor and never influence candidate
selection.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import inspect
import math
from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from .artifacts import module_state_sha256
from .collector import CachedEulerDecision, F0_DECISION_INDICES, F0_TOTAL_STEPS
from .f0_targets import (
    F0_TRAINING_SEEDS,
    F0_VALIDATION_SEEDS,
    F0TargetRow,
)
from .f0_metrics import build_f0_metric_action
from .operations import (
    LedgeredOperationExecutor,
    OperationContext,
    OperationReceipt,
    operation_output_sha256,
    tensor_operation_output_sha256,
)
from .renderer import (
    EulerFrameDiagnostics,
    EulerFrameState,
    EulerNativeFrameV1,
    tensor_sha256,
)
from .sdxl_adapter import SdxlEulerState, SdxlEulerTrainingAdapter
from .teachers import TargetStepConfig, construct_reward_targets


F0_TARGET_STEP_CONFIG = TargetStepConfig(
    eta_target=0.25,
    target_steps=2,
    trust_radius_u=0.50,
    backtracking=(1.0, 0.5, 0.25, 0.125),
    epsilon_grad=1e-12,
)

_MOMENT_RTOL = 2e-4
_MOMENT_ATOL = 2e-5
_CAP_ATOL = 1e-5


def _validate_decode_image(image: Any, state: SdxlEulerState, *, label: str) -> None:
    """Apply the same RGB ``[0, 1]`` contract as the native reward adapter."""
    if (
        not isinstance(image, Tensor)
        or not image.is_floating_point()
        or image.ndim != 4
        or image.shape[0] != state.conditioning.batch_size
        or image.shape[1] != 3
        or image.shape[2] <= 0
        or image.shape[3] <= 0
        or not torch.isfinite(image).all()
        or torch.any(image.detach() < 0.0)
        or torch.any(image.detach() > 1.0)
    ):
        raise RuntimeError(
            f"{label} must be a finite floating-point RGB NCHW batch in [0, 1]"
        )


def _validate_reward_scores(reward: Any, state: SdxlEulerState, *, label: str) -> None:
    """Require exactly one finite floating-point reward for each prompt."""
    if (
        not isinstance(reward, Tensor)
        or not reward.is_floating_point()
        or reward.ndim != 1
        or reward.shape[0] != state.conditioning.batch_size
        or not torch.isfinite(reward).all()
    ):
        raise RuntimeError(f"{label} must be one finite floating-point score per image")


def _is_unmodified_bound_method(
    adapter: Any, name: str, implementation: Any
) -> bool:
    """Only dispatch the native path for the exact bound implementation.

    An instance attribute can shadow a class method without changing
    ``type(adapter).__dict__``.  Inspecting the bound method prevents such an
    override from bypassing the legacy receipt validation path.
    """
    method = getattr(adapter, name, None)
    return (
        getattr(method, "__self__", None) is adapter
        and getattr(method, "__func__", None) is implementation
    )


@dataclass(frozen=True)
class TerminalAnchorRewardReceipt:
    """A detached scalar paired with its native executor capability."""

    reward: Tensor
    decode_receipt: OperationReceipt
    reward_receipt: OperationReceipt

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reward, Tensor)
            or self.reward.numel() != 1
            or not self.reward.is_floating_point()
            or self.reward.requires_grad
            or not torch.isfinite(self.reward).all()
        ):
            raise ValueError("terminal anchor reward must be one detached finite tensor")
        if type(self.decode_receipt) is not OperationReceipt or type(
            self.reward_receipt
        ) is not OperationReceipt:
            raise TypeError("terminal anchor reward requires decode and reward receipts")
        object.__setattr__(self, "reward", self.reward.detach().clone())


def _finite_rows(value: Tensor, *, batch: int) -> Tensor:
    if not isinstance(value, Tensor) or value.shape[0] != batch:
        raise ValueError("candidate diagnostics have an invalid batch shape")
    return torch.isfinite(value).reshape(batch, -1).all(dim=1)


def _same_frame(left: EulerFrameState, right: EulerFrameState) -> bool:
    for field in fields(EulerFrameState):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if field.name == "diagnostics":
            for diagnostic in fields(EulerFrameDiagnostics):
                left_item = getattr(left_value, diagnostic.name)
                right_item = getattr(right_value, diagnostic.name)
                if isinstance(left_item, Tensor):
                    if not isinstance(right_item, Tensor) or not torch.equal(
                        left_item.detach(), right_item.detach()
                    ):
                        return False
                elif left_item != right_item:
                    return False
        elif isinstance(left_value, Tensor):
            if not isinstance(right_value, Tensor) or not torch.equal(
                left_value.detach(), right_value.detach()
            ):
                return False
        elif left_value != right_value:
            return False
    return True


def _require_zero_history(state: SdxlEulerState, decision_index: int) -> None:
    expected = sum(index < decision_index for index in F0_DECISION_INDICES)
    if len(state.action_history) != expected:
        raise ValueError("cached C0 action history differs from the F0 schedule")
    for action in state.action_history:
        try:
            value = torch.as_tensor(action)
        except (TypeError, ValueError) as exc:
            raise ValueError("cached C0 action history is malformed") from exc
        if value.shape != (1, 6) or not torch.isfinite(value).all() or torch.count_nonzero(value):
            raise ValueError("cached C0 action history must contain strict zero actions")


def _require_cached_state(
    cached: CachedEulerDecision,
    renderer: EulerNativeFrameV1,
    adapter: Any,
) -> SdxlEulerState:
    if not isinstance(cached, CachedEulerDecision):
        raise TypeError("cached must be a CachedEulerDecision")
    if not isinstance(cached.state, SdxlEulerState):
        raise TypeError("cached F0 state must be SdxlEulerState")
    state = cached.state
    if state.conditioning.batch_size != 1 or cached.frame.sample.shape[0] != 1:
        raise ValueError("F0 target construction requires one logical state")
    if state.total_steps != F0_TOTAL_STEPS:
        raise ValueError("cached F0 state must use the registered 50-step schedule")
    if state.step_index != cached.decision_index:
        raise ValueError("cached state scheduler position differs from its decision")
    if state.branch != "anchor":
        raise ValueError("cached F0 state must come from the strict C0 anchor branch")
    if state.split not in {"train", "validation"}:
        raise ValueError("cached F0 state has an unsupported split")
    seeds = F0_TRAINING_SEEDS if state.split == "train" else F0_VALIDATION_SEEDS
    if state.conditioning.generation_seeds[0] not in seeds:
        raise ValueError("cached generation seed differs from the registered F0 split")
    if cached.checkpoint_hash is None or state.checkpoint_hash != cached.checkpoint_hash:
        raise ValueError("cached state and decision checkpoint hashes differ")
    if not torch.equal(state.latents.detach(), cached.frame.sample.detach()):
        raise ValueError("cached state latents differ from the renderer frame")
    _require_zero_history(state, cached.decision_index)

    if cached.frame_contract_hash != renderer.frame_contract_hash:
        raise ValueError("cached renderer frame contract differs from C0")
    if cached.calibration_hash != renderer.calibration_hash:
        raise ValueError("cached renderer calibration differs from C0")
    if tuple(cached.frame.diagnostics.active_mask) != tuple(renderer.active_mask):
        raise ValueError("cached frame action mask differs from C0")

    rebuilt = renderer.prepare_state(
        cached.step.observation,
        cached.step.condition,
        sigma_from=cached.step.sigma_from,
        sigma_to=cached.step.sigma_to,
        native_model_output=cached.step.native_model_output,
        prediction_type=cached.step.prediction_type,
    )
    if not _same_frame(cached.frame, rebuilt):
        raise ValueError("cached renderer frame differs from its frozen observation")

    executor = getattr(adapter, "operation_executor", None)
    binding = None if executor is None else getattr(executor, "authorization_binding", None)
    if binding is not None:
        contract_hash = getattr(binding, "contract_hash", None)
        if cached.run_contract != contract_hash:
            raise ValueError("cached run contract differs from the reward adapter")
        binding.validate_current(component=renderer)
    return state


def _require_c0_pair(
    renderer: EulerNativeFrameV1,
    reference_renderer: EulerNativeFrameV1,
    cached: CachedEulerDecision,
) -> tuple[Tensor, Tensor]:
    if not isinstance(renderer, EulerNativeFrameV1) or not isinstance(
        reference_renderer, EulerNativeFrameV1
    ):
        raise TypeError("F0 target construction requires EulerNativeFrameV1 renderers")
    if renderer is reference_renderer:
        raise ValueError("C0 renderer and reference must be distinct objects")
    if renderer.frame_contract_hash != reference_renderer.frame_contract_hash:
        raise ValueError("C0 renderer and reference frame contracts differ")
    if renderer.calibration_hash != reference_renderer.calibration_hash:
        raise ValueError("C0 renderer and reference calibrations differ")
    if renderer.contract.to_dict() != reference_renderer.contract.to_dict():
        raise ValueError("C0 renderer and reference action contracts differ")
    if not renderer.preserve_moments or not reference_renderer.preserve_moments:
        raise ValueError("F0 requires fixed-moment C0 renderers")
    if reference_renderer.training or any(
        parameter.requires_grad for parameter in reference_renderer.parameters()
    ):
        raise ValueError("C0 reference renderer must be frozen in eval mode")
    if any(
        left is right
        for left, right in zip(renderer.parameters(), reference_renderer.parameters())
    ):
        raise RuntimeError("C0 renderer and reference parameters alias")
    for left, right in zip(renderer.parameters(), reference_renderer.parameters()):
        if (
            left.numel()
            and right.numel()
            and left.device == right.device
            and left.untyped_storage().data_ptr()
            == right.untyped_storage().data_ptr()
        ):
            raise RuntimeError("C0 renderer and reference parameter storage aliases")

    renderer_hash = module_state_sha256(renderer)
    reference_hash = module_state_sha256(reference_renderer)
    if renderer_hash != reference_hash or renderer_hash != cached.checkpoint_hash:
        raise ValueError("C0 renderer state differs from reference or cached checkpoint")

    with torch.no_grad():
        anchor = renderer.action_parameters(cached.frame).detach()
        reference = reference_renderer.action_parameters(cached.frame).detach()
    if anchor.shape != (1, renderer.contract.num_slots) or reference.shape != anchor.shape:
        raise ValueError("C0 action parameters have an invalid shape")
    if not torch.equal(anchor, torch.zeros_like(anchor)) or not torch.equal(
        reference, torch.zeros_like(reference)
    ):
        raise ValueError("F0 target construction requires a strict zero-action C0")
    renderer.contract.validate_action(renderer.deterministic_action_from_mean(anchor))
    return anchor, reference


def _verified_terminal_anchor_reward(
    value: TerminalAnchorRewardReceipt,
    *,
    cached: CachedEulerDecision,
    state: SdxlEulerState,
    adapter: SdxlEulerTrainingAdapter,
) -> float:
    if not isinstance(value, TerminalAnchorRewardReceipt):
        raise TypeError(
            "terminal_anchor_reward must be a TerminalAnchorRewardReceipt"
        )
    executor = adapter.operation_executor
    if not isinstance(executor, LedgeredOperationExecutor):
        raise RuntimeError(
            "terminal anchor reward verification requires the formal operation executor"
        )
    binding = executor.authorization_binding
    binding.validate_current()
    contract = binding.contract
    if contract.get("method") != "f0" or cached.run_contract != binding.contract_hash:
        raise ValueError("terminal reward uses a different F0 run contract")

    action_history = torch.zeros(
        len(F0_DECISION_INDICES), 1, 6, dtype=torch.float32
    ).tolist()
    metric_action = build_f0_metric_action(
        branch="anchor",
        target_record_sha256=None,
        renderer_action_history=action_history,
    )
    common = {
        "split": state.split,
        "prompt": state.conditioning.prompt_ids[0],
        "seed": state.conditioning.generation_seeds[0],
        "step": F0_TOTAL_STEPS - 1,
        "prefix": state.prefix,
        "branch": "anchor",
        "action": metric_action,
        "checkpoint_hash": str(cached.checkpoint_hash),
    }
    decode_context = OperationContext(**common)
    decode_receipt = value.decode_receipt
    executor.validate_receipt(
        decode_receipt,
        kind="vae_decode",
        output_hash=decode_receipt.output_hash,
        context=decode_context,
        scalar_or_gradient="tensor",
    )

    reward_receipt = value.reward_receipt
    reward_source = reward_receipt.context
    if reward_source.image_hash is None or reward_source.cached_parent is None:
        raise ValueError("terminal reward receipt does not bind its decoded image")
    expected_output = operation_output_sha256(value.reward)
    if (
        reward_source.cached_parent != decode_receipt.output_hash
        or tensor_operation_output_sha256(reward_source.image_hash)
        != decode_receipt.output_hash
    ):
        raise ValueError("terminal reward image differs from its VAE decode output")
    reward_context = OperationContext(
        **common,
        image_hash=reward_source.image_hash,
        cached_parent=decode_receipt.output_hash,
    )
    executor.validate_receipt(
        reward_receipt,
        kind="reward_forward",
        output_hash=expected_output,
        context=reward_context,
        scalar_or_gradient="scalar",
        parent_reservation_id=decode_receipt.reservation_id,
    )
    return float(value.reward.detach().double().cpu().reshape(()))


def _candidate_validator(
    renderer: EulerNativeFrameV1,
    frame: EulerFrameState,
):
    """Build the reward-blind structural validator used by backtracking."""

    def validate(candidate_u: Tensor) -> Tensor:
        batch = candidate_u.shape[0]
        action = renderer.deterministic_action_from_mean(candidate_u)
        valid = renderer.contract.action_valid_mask(action)
        rendered = renderer.apply_coefficients(frame, action)
        valid = valid & frame.diagnostics.valid & rendered.diagnostics.valid
        valid = valid & (rendered.coefficients == action).reshape(batch, -1).all(dim=1)

        for value in (
            action,
            rendered.guided_x0,
            rendered.residual,
            rendered.coefficients,
            rendered.diagnostics.angle,
            rendered.diagnostics.angle_cap_multiplier,
            rendered.diagnostics.scheduler_cap_multiplier,
            rendered.diagnostics.mapped_update_ratio,
        ):
            valid = valid & _finite_rows(value, batch=batch)

        clean = frame.clean_latent.float()
        guided = rendered.guided_x0.float()
        clean_mean = clean.mean(dim=(-2, -1))
        guided_mean = guided.mean(dim=(-2, -1))
        clean_centered = clean - clean_mean[..., None, None]
        guided_centered = guided - guided_mean[..., None, None]
        clean_energy = clean_centered.square().sum(dim=(-2, -1))
        guided_energy = guided_centered.square().sum(dim=(-2, -1))
        mean_ok = torch.isclose(
            guided_mean,
            clean_mean,
            rtol=_MOMENT_RTOL,
            atol=_MOMENT_ATOL,
        ).all(dim=1)
        energy_ok = torch.isclose(
            guided_energy,
            clean_energy,
            rtol=_MOMENT_RTOL,
            atol=_MOMENT_ATOL,
        ).all(dim=1)

        diagnostics = rendered.diagnostics
        cap_ok = (
            (diagnostics.scheduler_cap_multiplier > 0)
            & (diagnostics.scheduler_cap_multiplier <= 1.0 + _CAP_ATOL)
            & (diagnostics.angle_cap_multiplier > 0)
            & (diagnostics.angle_cap_multiplier <= 1.0 + _CAP_ATOL)
            & (diagnostics.mapped_update_ratio <= renderer.max_update_ratio + _CAP_ATOL)
            & (diagnostics.angle <= renderer.theta_max + _CAP_ATOL)
        )
        transition = frame.native_update.to(rendered.residual.dtype) + frame.kappa.to(
            device=rendered.residual.device, dtype=rendered.residual.dtype
        ).reshape(-1, 1, 1, 1) * rendered.residual
        return valid & mean_ok & energy_ok & cap_ok & _finite_rows(
            transition, batch=batch
        )

    return validate


def _transition_from_u(
    renderer: EulerNativeFrameV1,
    frame: EulerFrameState,
    value: Tensor,
) -> Tensor:
    action = renderer.deterministic_action_from_mean(value)
    rendered = renderer.apply_coefficients(frame, action)
    kappa = frame.kappa.to(
        device=rendered.residual.device, dtype=rendered.residual.dtype
    ).reshape(-1, 1, 1, 1)
    return (
        frame.native_update.to(dtype=rendered.residual.dtype)
        + kappa * rendered.residual
    ).detach()


def _decode_target_with_receipt(
    adapter: SdxlEulerTrainingAdapter,
    state: SdxlEulerState,
    *,
    action_metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[Tensor, OperationReceipt]:
    """Decode one target through the formal receipt API.

    A few test/integration adapters override the legacy ``decode`` method to
    count calls.  The native adapter uses ``decode_with_receipt`` directly;
    overridden adapters receive the same durable receipt through the guarded
    private handoff populated by the base method.
    """
    native = _is_unmodified_bound_method(
        adapter, "decode", SdxlEulerTrainingAdapter.decode
    ) and _is_unmodified_bound_method(
        adapter, "decode_with_receipt", SdxlEulerTrainingAdapter.decode_with_receipt
    )
    if native:
        image, receipt = adapter.decode_with_receipt(
            state, action_metadata=action_metadata
        )
    else:
        previous_metadata = getattr(adapter, "_pending_decode_action_metadata", None)
        previous_receipt = getattr(adapter, "_last_decode_receipt", None)
        adapter._pending_decode_action_metadata = action_metadata
        adapter._last_decode_receipt = None
        try:
            image = adapter.decode(state)
        finally:
            adapter._pending_decode_action_metadata = previous_metadata
        receipt = getattr(adapter, "_last_decode_receipt", None)
        adapter._last_decode_receipt = previous_receipt
    if type(receipt) is not OperationReceipt:
        raise RuntimeError("F0 decode did not return a durable VAE receipt")
    executor = getattr(adapter, "operation_executor", None)
    if not isinstance(executor, LedgeredOperationExecutor):
        raise RuntimeError("F0 decode receipt validation requires the formal executor")
    _validate_decode_image(
        image, state, label="overridden F0 decode output"
    )
    expected_output_hash = operation_output_sha256(image)
    expected_context = state.operation_context(action_metadata=action_metadata)
    # Authenticate the durable ledger pair for both dispatch paths.  Native
    # wrappers already validate internally; repeating this check also binds
    # the returned value to the expected model, context, and operation kind.
    executor.validate_receipt(
        receipt,
        kind="vae_decode",
        output_hash=expected_output_hash,
        context=expected_context,
        scalar_or_gradient="tensor",
    )
    return image, receipt


def _reward_target_with_receipt(
    adapter: SdxlEulerTrainingAdapter,
    state: SdxlEulerState,
    image: Tensor,
    *,
    parent_receipt: OperationReceipt,
    action_metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[Tensor, OperationReceipt]:
    """Score one target while preserving the VAE parent capability."""
    executor = getattr(adapter, "operation_executor", None)
    if not isinstance(executor, LedgeredOperationExecutor):
        raise RuntimeError("F0 reward receipt validation requires the formal executor")

    # Authenticate the parent and validate the exact image before dispatching
    # to an adapter override.  Legacy adapters can perform arbitrary work in
    # ``reward``; an invalid parent or image must not consume that operation.
    _validate_decode_image(image, state, label="F0 reward input image")
    image_hash = tensor_sha256(image)
    image_output_hash = operation_output_sha256(image)
    decode_context = state.operation_context(action_metadata=action_metadata)
    executor.validate_receipt(
        parent_receipt,
        kind="vae_decode",
        output_hash=image_output_hash,
        context=decode_context,
        scalar_or_gradient="tensor",
    )

    native = _is_unmodified_bound_method(
        adapter, "reward", SdxlEulerTrainingAdapter.reward
    ) and _is_unmodified_bound_method(
        adapter, "reward_with_receipt", SdxlEulerTrainingAdapter.reward_with_receipt
    )
    if native:
        reward, receipt = adapter.reward_with_receipt(
            state,
            image,
            parent_receipt=parent_receipt,
            action_metadata=action_metadata,
        )
    else:
        previous_parent = getattr(adapter, "_pending_reward_parent", None)
        previous_metadata = getattr(adapter, "_pending_reward_action_metadata", None)
        previous_receipt = getattr(adapter, "_last_reward_receipt", None)
        adapter._pending_reward_parent = parent_receipt
        adapter._pending_reward_action_metadata = action_metadata
        adapter._last_reward_receipt = None
        try:
            reward = adapter.reward(state, image)
        finally:
            adapter._pending_reward_parent = previous_parent
            adapter._pending_reward_action_metadata = previous_metadata
        receipt = getattr(adapter, "_last_reward_receipt", None)
        adapter._last_reward_receipt = previous_receipt
    if type(receipt) is not OperationReceipt:
        raise RuntimeError(
            "overridden F0 reward did not return a durable reward receipt"
        )
    _validate_reward_scores(
        reward, state, label="overridden F0 reward output"
    )
    reward_context = state.operation_context(
        action_metadata=action_metadata,
        image_hash=image_hash,
        cached_parent=image_output_hash,
    )
    # Authenticate the child after the callback and bind it to the exact
    # returned reward tensor and the already-authenticated decode parent.
    executor.validate_receipt(
        receipt,
        kind="reward_forward",
        output_hash=operation_output_sha256(reward),
        context=reward_context,
        scalar_or_gradient="scalar",
        parent_reservation_id=parent_receipt.reservation_id,
    )
    return reward, receipt


def _reward_gradient_with_metadata(
    adapter: SdxlEulerTrainingAdapter,
    state: SdxlEulerState,
    reward: Tensor,
    inputs: Tensor,
    *,
    image_hash: str,
    action_metadata: Mapping[str, Any],
) -> Tensor:
    """Invoke the gradient API with the exact child operation context.

    Legacy instrumented adapters may still expose the pre-metadata signature;
    they receive the same authenticated receipt through the existing image
    binding and are handled by the executor's compatibility check.
    """
    method = adapter.reward_gradient
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_metadata = "action_metadata" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {"image_hash": image_hash}
    if accepts_metadata:
        kwargs["action_metadata"] = action_metadata
    return method(state, reward, inputs, **kwargs)


def construct_f0_target(
    cached: CachedEulerDecision,
    renderer: EulerNativeFrameV1,
    reference_renderer: EulerNativeFrameV1,
    adapter: SdxlEulerTrainingAdapter,
    *,
    prompt_id: str,
    source: str,
    stratum: str,
    fold: int | None,
    terminal_anchor_reward: TerminalAnchorRewardReceipt,
    reward_location: float,
    reward_scale: float,
) -> F0TargetRow:
    """Construct one detached F0 target with exactly one reward backward.

    The adapter owns the only formal gradient operation.  This function never
    invokes ``torch.autograd.grad`` directly and never evaluates candidates
    with a reward.
    """

    if not isinstance(adapter, SdxlEulerTrainingAdapter):
        raise TypeError("adapter must be SdxlEulerTrainingAdapter")
    for name in (
        "decode",
        "reward",
        "decode_with_receipt",
        "reward_with_receipt",
        "reward_gradient",
    ):
        if not callable(getattr(adapter, name, None)):
            raise TypeError(f"adapter must provide a callable {name}")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError("prompt_id must be a non-empty string")
    if (
        isinstance(reward_location, bool)
        or not isinstance(reward_location, (int, float))
        or not math.isfinite(float(reward_location))
    ):
        raise ValueError("reward_location must be finite")
    if (
        isinstance(reward_scale, bool)
        or not isinstance(reward_scale, (int, float))
        or not math.isfinite(float(reward_scale))
        or float(reward_scale) <= 0
    ):
        raise ValueError("reward_scale must be finite and positive")
    state = _require_cached_state(cached, renderer, adapter)
    if state.conditioning.prompt_ids[0] != prompt_id:
        raise ValueError("prompt_id differs from the cached logical state")
    anchor_u, reference_u = _require_c0_pair(
        renderer, reference_renderer, cached
    )
    terminal_anchor_value = _verified_terminal_anchor_reward(
        terminal_anchor_reward,
        cached=cached,
        state=state,
        adapter=adapter,
    )

    reward_context: dict[str, Any] = {}

    def clean_from_u(value: Tensor) -> Tensor:
        if "anchor_action" in reward_context:
            raise RuntimeError("F0 clean endpoint was constructed more than once")
        action = renderer.deterministic_action_from_mean(value)
        rendered = renderer.apply_coefficients(cached.frame, action)
        reward_context["anchor_action"] = action
        return rendered.guided_x0

    def anchor_reward(guided_x0: Tensor) -> Tensor:
        if "reward" in reward_context:
            raise RuntimeError("F0 anchor reward was evaluated more than once")
        action = reward_context.get("anchor_action")
        if not isinstance(action, Tensor):
            raise RuntimeError("F0 reward is missing its anchor action")
        reward_state = replace(
            state,
            latents=guided_x0,
            action_history=(
                *state.action_history,
                action.detach().float().cpu().tolist(),
            ),
        )
        # Formal F0 rewards must retain the exact VAE operation as their
        # durable parent.  The receipt-aware adapter wrappers still dispatch
        # through the ordinary methods, so subclasses retain instrumentation
        # without creating a second decode or reward call.
        action_history = [
            *state.action_history,
            action.detach().float().cpu().tolist(),
        ]
        # Metric contexts use the fixed three-decision representation even
        # when the cached state is before the later decisions.  Future C0
        # decisions are explicit zero actions, matching the gate's canonical
        # target history.
        action_history.extend(
            [[[0.0] * 6] for _ in range(len(F0_DECISION_INDICES) - len(action_history))]
        )
        action_metadata = build_f0_metric_action(
            branch="anchor",
            target_record_sha256=None,
            renderer_action_history=action_history,
        )
        image, decode_receipt = _decode_target_with_receipt(
            adapter,
            reward_state,
            action_metadata=action_metadata,
        )
        reward, reward_receipt = _reward_target_with_receipt(
            adapter,
            reward_state,
            image,
            parent_receipt=decode_receipt,
            action_metadata=action_metadata,
        )
        reward_context.update(
            state=reward_state,
            image_hash=tensor_sha256(image),
            reward=reward,
            decode_receipt=decode_receipt,
            reward_receipt=reward_receipt,
            action_metadata=action_metadata,
        )
        return reward

    def anchor_gradient(reward: Tensor, value: Tensor) -> Tensor:
        if reward_context.get("gradient_used"):
            raise RuntimeError("F0 anchor reward gradient was evaluated more than once")
        reward_state = reward_context.get("state")
        image_hash = reward_context.get("image_hash")
        action_metadata = reward_context.get("action_metadata")
        if not isinstance(reward_state, SdxlEulerState) or not isinstance(
            image_hash, str
        ) or not isinstance(action_metadata, Mapping):
            raise RuntimeError("F0 reward gradient is missing anchor provenance")
        reward_context["gradient_used"] = True
        return _reward_gradient_with_metadata(
            adapter,
            reward_state,
            reward,
            value,
            image_hash=image_hash,
            action_metadata=action_metadata,
        )

    pair = construct_reward_targets(
        anchor_u,
        clean_from_u,
        anchor_reward,
        reward_gradient=anchor_gradient,
        config=F0_TARGET_STEP_CONFIG,
        contract=renderer.contract,
        candidate_validator=_candidate_validator(renderer, cached.frame),
    )
    if not isinstance(pair.valid, Tensor) or pair.valid.shape != (1,):
        raise RuntimeError("F0 target step returned invalid row flags")
    valid = bool(pair.valid.item())

    active = torch.as_tensor(
        renderer.active_mask, device=anchor_u.device, dtype=anchor_u.dtype
    )
    gradient = pair.gradient_u.detach().to(dtype=anchor_u.dtype) * active
    gradient_norm = torch.linalg.vector_norm(gradient.flatten(1), dim=-1, keepdim=True)
    valid = valid and bool(
        torch.isfinite(gradient).all()
        and (gradient_norm[:, 0] > F0_TARGET_STEP_CONFIG.epsilon_grad).all()
    )
    if valid:
        gradient = gradient / gradient_norm.reshape(-1, 1)
        plus_u = pair.plus_u.detach()
        minus_u = pair.minus_u.detach()
        plus_transition = _transition_from_u(renderer, cached.frame, plus_u)
        minus_transition = _transition_from_u(renderer, cached.frame, minus_u)
    else:
        gradient = torch.zeros_like(anchor_u)
        plus_u = anchor_u.detach().clone()
        minus_u = anchor_u.detach().clone()
        plus_transition = cached.frame.native_update.detach().clone()
        minus_transition = cached.frame.native_update.detach().clone()

    reward_value = reward_context.get("reward")
    if not isinstance(reward_value, Tensor) or reward_value.numel() != 1:
        raise RuntimeError("F0 anchor reward must contain exactly one score")
    local_anchor_value = float(reward_value.detach().double().cpu().reshape(()))
    if not math.isfinite(local_anchor_value):
        raise RuntimeError("F0 anchor reward must be finite")
    pair_weight = min(
        1.0,
        max(
            0.0,
            0.5
            + 0.5
            * (terminal_anchor_value - float(reward_location))
            / float(reward_scale),
        ),
    )

    return F0TargetRow(
        prompt_id=prompt_id,
        generation_seed=state.conditioning.generation_seeds[0],
        decision_index=cached.decision_index,
        split=state.split,
        source=source,
        stratum=stratum,
        fold=fold,
        state=cached.frame,
        anchor_u=anchor_u,
        plus_u=plus_u,
        minus_u=minus_u,
        reward_gradient=gradient.detach(),
        plus_transition=plus_transition,
        minus_transition=minus_transition,
        reference_u=reference_u,
        pair_weight=pair_weight,
        valid=valid,
    )


__all__ = [
    "F0_TARGET_STEP_CONFIG",
    "TerminalAnchorRewardReceipt",
    "construct_f0_target",
]
