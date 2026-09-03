"""Typed SDXL/Euler runtime used by formal latent-renderer training.

The production inference pipeline remains unchanged and under ``no_grad``.
This adapter reuses its prompt, latent, U-Net, scheduler, and VAE components to
expose one branch-safe Euler state machine to every training method.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence
import weakref

import torch
from torch import Tensor, nn

from AttentionGuidance.latent_renderer import (
    RendererCondition,
    RendererObservation,
    predict_euler_no_churn_prev_sample,
    prepare_euler_clean_endpoint,
)
from InferencePipelines.cfg_batch import (
    expand_cfg_latents,
    expand_cfg_time_ids,
    split_cfg_noise_pred,
)

from .collector import (
    EulerNativeRolloutCollector,
    EulerRolloutStep,
    EulerTransitionResult,
)
from .operations import (
    LedgeredOperationExecutor,
    OperationContext,
    OperationReceipt,
    operation_output_sha256,
)
from .preferences import PreferenceLabelProvenance
from .renderer import tensor_sha256


def _require_tensor(
    value: Any,
    name: str,
    *,
    ndim: Optional[int] = None,
    detached: bool = True,
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if detached and value.requires_grad:
        raise ValueError(f"{name} must be detached")
    return value


def _module_device(module: nn.Module) -> torch.device:
    for value in (*tuple(module.parameters()), *tuple(module.buffers())):
        return value.device
    return torch.device("cpu")


def _module_dtype(module: nn.Module) -> torch.dtype:
    for value in (*tuple(module.parameters()), *tuple(module.buffers())):
        if value.is_floating_point():
            return value.dtype
    raise ValueError("module has no floating-point parameters or buffers")


def _freeze(module: nn.Module, name: str) -> None:
    if not isinstance(module, nn.Module):
        raise TypeError(f"{name} must be a torch.nn.Module")
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def _tuple_strings(value: str | Sequence[str], name: str) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")
    return values


def _tuple_seeds(value: int | Sequence[int], count: int) -> tuple[int, ...]:
    values = (value,) if isinstance(value, int) and not isinstance(value, bool) else tuple(value)
    if len(values) != count:
        raise ValueError("generation_seeds must match the prompt batch")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in values):
        raise ValueError("generation_seeds must contain non-negative integers")
    return values


def _action_record(value: Optional[Tensor], history: tuple[Any, ...]) -> list[Any]:
    if value is None:
        return list(history)
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError("action must be a floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError("action contains non-finite values")
    return [*history, value.detach().float().cpu().tolist()]


@dataclass(frozen=True)
class SdxlPromptConditioning:
    """Detached SDXL conditioning in canonical CFG row order."""

    prompt_ids: tuple[str, ...]
    prompts: tuple[str, ...]
    generation_seeds: tuple[int, ...]
    prompt_embeds: Tensor
    pooled_prompt_embeds: Tensor
    add_text_embeds: Tensor
    add_time_ids: Tensor
    do_classifier_free_guidance: bool

    def __post_init__(self) -> None:
        batch = len(self.prompt_ids)
        if batch <= 0 or len(self.prompts) != batch or len(self.generation_seeds) != batch:
            raise ValueError("conditioning identifiers, prompts, and seeds must align")
        if any(not isinstance(value, str) or not value for value in self.prompt_ids):
            raise ValueError("prompt_ids must contain non-empty strings")
        if any(not isinstance(value, str) or not value for value in self.prompts):
            raise ValueError("prompts must contain non-empty strings")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.generation_seeds
        ):
            raise ValueError("generation_seeds must contain non-negative integers")
        if not isinstance(self.do_classifier_free_guidance, bool):
            raise TypeError("do_classifier_free_guidance must be boolean")
        branch_rows = 2 if self.do_classifier_free_guidance else 1
        for name, value in (
            ("prompt_embeds", self.prompt_embeds),
            ("add_text_embeds", self.add_text_embeds),
            ("add_time_ids", self.add_time_ids),
        ):
            tensor = _require_tensor(value, name, detached=True)
            if tensor.shape[0] != branch_rows * batch:
                raise ValueError(f"{name} has the wrong CFG batch size")
        pooled = _require_tensor(
            self.pooled_prompt_embeds,
            "pooled_prompt_embeds",
            ndim=2,
            detached=True,
        )
        if pooled.shape[0] != batch:
            raise ValueError("pooled_prompt_embeds must contain positive rows only")

    @property
    def batch_size(self) -> int:
        return len(self.prompt_ids)

    def clone(self) -> "SdxlPromptConditioning":
        return replace(
            self,
            prompt_embeds=self.prompt_embeds.detach().clone(),
            pooled_prompt_embeds=self.pooled_prompt_embeds.detach().clone(),
            add_text_embeds=self.add_text_embeds.detach().clone(),
            add_time_ids=self.add_time_ids.detach().clone(),
        )


@dataclass(frozen=True)
class SdxlEulerState:
    """One immutable branch state; scheduler position is explicit, not global."""

    latents: Tensor
    conditioning: SdxlPromptConditioning
    step_index: int
    total_steps: int
    split: str
    branch: str
    prefix: int
    checkpoint_hash: str
    action_history: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        latent = _require_tensor(self.latents, "latents", ndim=4, detached=False)
        if latent.shape[0] != self.conditioning.batch_size:
            raise ValueError("latent and conditioning batches differ")
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or isinstance(self.total_steps, bool)
            or not isinstance(self.total_steps, int)
            or self.total_steps <= 0
            or self.step_index < 0
            or self.step_index > self.total_steps
        ):
            raise ValueError("state has an invalid scheduler position")
        for name in ("split", "branch"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.prefix, bool) or not isinstance(self.prefix, int) or self.prefix < 0:
            raise ValueError("prefix must be a non-negative integer")
        if (
            not isinstance(self.checkpoint_hash, str)
            or len(self.checkpoint_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.checkpoint_hash)
        ):
            raise ValueError("checkpoint_hash must be a lowercase SHA-256 hash")
        try:
            json.dumps(self.action_history, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("action_history must contain finite JSON values") from exc

    def operation_context(
        self,
        *,
        action: Optional[Tensor] = None,
        action_metadata: Optional[Mapping[str, Any]] = None,
        image_hash: Optional[str] = None,
        cached_parent: Optional[str] = None,
    ) -> OperationContext:
        if self.conditioning.batch_size != 1:
            raise RuntimeError("formal ledger operations require one prompt-seed block")
        if action is not None and action_metadata is not None:
            raise ValueError("operation context cannot combine an action and action metadata")
        if action_metadata is not None and not isinstance(action_metadata, Mapping):
            raise TypeError("operation action metadata must be a mapping")
        action_value = (
            dict(action_metadata)
            if action_metadata is not None
            else _action_record(action, self.action_history)
        )
        return OperationContext(
            split=self.split,
            prompt=self.conditioning.prompt_ids[0],
            seed=self.conditioning.generation_seeds[0],
            step=min(self.step_index, self.total_steps - 1),
            prefix=self.prefix,
            branch=self.branch,
            action=action_value,
            checkpoint_hash=self.checkpoint_hash,
            image_hash=image_hash,
            cached_parent=cached_parent,
        )


class SdxlEulerTrainingAdapter:
    """One-call CFG observation, real Euler transition, decode, and reward API."""

    def __init__(
        self,
        pipeline: Any,
        basis_provider: Any,
        *,
        total_steps: int,
        guidance_scale: float,
        vae_scaling_factor: Optional[float] = None,
        decision_indices: Sequence[int] = (8, 24, 40),
        operation_executor: Optional[LedgeredOperationExecutor] = None,
        reward_model: Optional[nn.Module] = None,
        guidance_rescale: float = 0.0,
    ) -> None:
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if (
            isinstance(guidance_scale, bool)
            or not isinstance(guidance_scale, (int, float))
            or not math.isfinite(float(guidance_scale))
            or guidance_scale <= 1.0
        ):
            raise ValueError("the registered SDXL adapter requires finite CFG greater than one")
        if guidance_rescale != 0.0:
            raise ValueError("the first training contract freezes guidance_rescale at zero")
        indices = tuple(decision_indices)
        if (
            not indices
            or any(isinstance(value, bool) or not isinstance(value, int) for value in indices)
            or tuple(sorted(indices)) != indices
            or len(set(indices)) != len(indices)
            or any(value < 0 or value >= total_steps for value in indices)
        ):
            raise ValueError("decision_indices must be unique, ordered, and inside the schedule")
        for name in ("unet", "scheduler", "vae", "encode_prompt", "prepare_latents", "_get_add_time_ids"):
            if not hasattr(pipeline, name):
                raise TypeError(f"pipeline is missing {name}")
        if not callable(basis_provider):
            raise TypeError("basis_provider must be callable")
        if not callable(getattr(basis_provider, "capture_forward", None)):
            raise TypeError("basis_provider must expose capture_forward()")
        if operation_executor is not None and not isinstance(
            operation_executor, LedgeredOperationExecutor
        ):
            raise TypeError("operation_executor must be LedgeredOperationExecutor or None")
        self.pipeline = pipeline
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self._scheduler_template = pipeline.scheduler
        self.basis_provider = basis_provider
        self.total_steps = total_steps
        self.decision_indices = indices
        self.guidance_scale = float(guidance_scale)
        self.guidance_rescale = float(guidance_rescale)
        self.operation_executor = operation_executor
        self.reward_model = reward_model
        self._reward_receipts: dict[
            int, tuple[weakref.ReferenceType[Tensor], OperationReceipt]
        ] = {}
        # The receipt-aware wrappers dispatch through the ordinary overridable
        # methods so instrumented adapters cannot accidentally execute a second
        # decode/reward operation.  These transient fields carry the formal
        # parent context across that dispatch boundary.
        self._last_decode_receipt: Optional[OperationReceipt] = None
        self._last_reward_receipt: Optional[OperationReceipt] = None
        self._pending_decode_action_metadata: Optional[Mapping[str, Any]] = None
        self._pending_reward_parent: Optional[OperationReceipt] = None
        self._pending_reward_action_metadata: Optional[Mapping[str, Any]] = None
        _freeze(self.unet, "pipeline.unet")
        _freeze(self.vae, "pipeline.vae")
        if reward_model is not None:
            _freeze(reward_model, "reward_model")
        config_factor = getattr(getattr(self.vae, "config", None), "scaling_factor", None)
        if (
            config_factor is None
            or not math.isfinite(float(config_factor))
            or float(config_factor) <= 0
        ):
            raise ValueError("pinned VAE has an invalid scaling factor")
        if vae_scaling_factor is not None:
            if (
                isinstance(vae_scaling_factor, bool)
                or not isinstance(vae_scaling_factor, (int, float))
                or not math.isfinite(float(vae_scaling_factor))
                or not math.isclose(
                    float(config_factor),
                    float(vae_scaling_factor),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError("configured VAE scaling factor differs from the pinned VAE")
        self.vae_scaling_factor = float(config_factor)
        if (
            bool(getattr(getattr(self.vae, "config", None), "force_upcast", False))
            and _module_dtype(self.vae) != torch.float32
        ):
            raise ValueError(
                "a force_upcast VAE must be loaded permanently in float32 for training"
            )
        self.device = _module_device(self.unet)
        reference = self._build_scheduler(self.device)
        try:
            from diffusers import EulerDiscreteScheduler
        except ImportError as exc:
            raise RuntimeError("diffusers is required for SDXL renderer training") from exc
        if type(reference) is not EulerDiscreteScheduler:
            raise ValueError("the registered training scheduler is EulerDiscreteScheduler")
        self.timesteps = reference.timesteps.detach().clone()
        sigmas = getattr(reference, "sigmas", None)
        if not isinstance(sigmas, Tensor) or len(sigmas) != len(self.timesteps) + 1:
            raise ValueError("Euler scheduler must expose one sigma pair per step")
        self.sigmas = sigmas.detach().clone()
        prediction_type = getattr(getattr(reference, "config", None), "prediction_type", None)
        if prediction_type not in {"epsilon", "sample", "original_sample", "v_prediction"}:
            raise ValueError("Euler scheduler has an unsupported prediction_type")
        self.prediction_type = str(prediction_type)
        schedule_payload = {
            "timesteps": self.timesteps.detach().float().cpu().tolist(),
            "sigmas": self.sigmas.detach().float().cpu().tolist(),
            "total_steps": self.total_steps,
            "prediction_type": self.prediction_type,
        }
        self.schedule_hash = hashlib.sha256(
            json.dumps(
                schedule_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        scheduler_config_payload = {
            "class": (
                f"{type(reference).__module__}.{type(reference).__qualname__}"
            ),
            "config": dict(reference.config),
        }
        try:
            scheduler_config_bytes = json.dumps(
                scheduler_config_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("Euler scheduler config must contain canonical JSON values") from exc
        self.scheduler_config_hash = hashlib.sha256(scheduler_config_bytes).hexdigest()
        if self.operation_executor is not None:
            contract = self.operation_executor.authorization_binding.contract
            if int(contract["nfe"]) != self.total_steps:
                raise ValueError("adapter total_steps differs from the run contract")
            if contract["scheduler_config_sha256"] != self.scheduler_config_hash:
                raise ValueError("Euler scheduler config differs from the run contract")
            if contract["scheduler_schedule_sha256"] != self.schedule_hash:
                raise ValueError("Euler schedule differs from the run contract")
            if contract["prediction_type"] != self.prediction_type:
                raise ValueError("Euler prediction_type differs from the run contract")
            if contract["do_classifier_free_guidance"] is not True:
                raise ValueError("run contract disabled classifier-free guidance")
            if float(contract["guidance_scale"]) != self.guidance_scale:
                raise ValueError("guidance_scale differs from the run contract")
            if float(contract["guidance_rescale"]) != self.guidance_rescale:
                raise ValueError("guidance_rescale differs from the run contract")
            if tuple(contract["decision_indices"]) != self.decision_indices:
                raise ValueError("decision_indices differ from the run contract")

    def _preflight(self) -> None:
        if self.operation_executor is not None:
            self.operation_executor._preflight()

    def _validate_state_runtime(self, state: SdxlEulerState) -> None:
        if not isinstance(state, SdxlEulerState):
            raise TypeError("state must be SdxlEulerState")
        if state.total_steps != self.total_steps:
            raise ValueError("state total_steps differs from the adapter")
        unet_dtype = _module_dtype(self.unet)
        if state.latents.device != self.device or state.latents.dtype != unet_dtype:
            raise ValueError("state latent device or dtype differs from the frozen U-Net")
        if state.conditioning.do_classifier_free_guidance is not True:
            raise ValueError("the registered adapter requires classifier-free guidance")
        for name, value in (
            ("prompt_embeds", state.conditioning.prompt_embeds),
            ("pooled_prompt_embeds", state.conditioning.pooled_prompt_embeds),
            ("add_text_embeds", state.conditioning.add_text_embeds),
            ("add_time_ids", state.conditioning.add_time_ids),
        ):
            if value.device != self.device or value.dtype != unet_dtype:
                raise ValueError(
                    f"conditioning {name} device or dtype differs from the frozen U-Net"
                )
        completed_decisions = sum(
            decision < state.step_index for decision in self.decision_indices
        )
        if len(state.action_history) != completed_decisions:
            raise ValueError("state action history differs from its scheduler position")

    def _validate_step_runtime(
        self,
        state: SdxlEulerState,
        step_index: int,
        step: EulerRolloutStep,
        scheduler: Any,
    ) -> None:
        if step.prediction_type != self.prediction_type:
            raise ValueError("observation prediction_type differs from the adapter")
        if step.metadata.get("schedule_sha256") != self.schedule_hash:
            raise ValueError("observation schedule hash differs from the adapter")
        if step.metadata.get("step_index") != step_index:
            raise ValueError("observation metadata has a stale step index")
        if not torch.equal(
            step.observation.latents_before_step, state.latents.detach()
        ):
            raise ValueError("observation latent differs from the explicit state")
        expected_timestep = scheduler.timesteps[step_index]
        actual_timestep = torch.as_tensor(
            step.observation.timestep,
            device=expected_timestep.device,
            dtype=expected_timestep.dtype,
        )
        if actual_timestep.numel() != 1 or not torch.equal(
            actual_timestep.reshape(()), expected_timestep.reshape(())
        ):
            raise ValueError("observation timestep differs from the Euler schedule")
        for name, actual, expected in (
            ("sigma_from", step.sigma_from, scheduler.sigmas[step_index]),
            ("sigma_to", step.sigma_to, scheduler.sigmas[step_index + 1]),
        ):
            actual_tensor = torch.as_tensor(
                actual, device=expected.device, dtype=expected.dtype
            )
            if actual_tensor.numel() != 1 or not torch.equal(
                actual_tensor.reshape(()), expected.reshape(())
            ):
                raise ValueError(f"observation {name} differs from the Euler schedule")

    def make_collector(
        self,
        renderer,
        *,
        reference_renderer=None,
        preserve_graph: bool = False,
    ):
        """Construct the only formal collector shape supported by this adapter."""
        binding = (
            None
            if self.operation_executor is None
            else self.operation_executor.authorization_binding
        )
        return EulerNativeRolloutCollector(
            renderer,
            decision_indices=self.decision_indices,
            registered_decision_indices=self.decision_indices,
            total_steps=self.total_steps,
            adapter=self,
            reference_renderer=reference_renderer,
            preserve_graph=preserve_graph,
            run_contract=None if binding is None else binding.run_contract,
            authorization_binding=binding,
        )

    def _build_scheduler(self, device: torch.device):
        scheduler_type = type(self._scheduler_template)
        from_config = getattr(scheduler_type, "from_config", None)
        if callable(from_config):
            scheduler = from_config(self._scheduler_template.config)
        else:
            scheduler = copy.deepcopy(self._scheduler_template)
        scheduler.set_timesteps(self.total_steps, device=device)
        return scheduler

    def _scheduler(self, device: torch.device):
        scheduler = self._build_scheduler(device)
        if not torch.equal(scheduler.timesteps.detach().cpu(), self.timesteps.detach().cpu()):
            raise RuntimeError("Euler timestep schedule changed after adapter construction")
        if not torch.equal(scheduler.sigmas.detach().cpu(), self.sigmas.detach().cpu()):
            raise RuntimeError("Euler sigma schedule changed after adapter construction")
        return scheduler

    def _execute(
        self,
        kind: str,
        state: SdxlEulerState,
        callback,
        *,
        action: Optional[Tensor] = None,
        image_hash: Optional[str] = None,
        cached_parent: Optional[str] = None,
        scalar_or_gradient: str = "tensor",
        output_hasher=None,
    ):
        if self.operation_executor is None:
            return callback()
        kwargs = {"scalar_or_gradient": scalar_or_gradient}
        if output_hasher is not None:
            kwargs["output_hasher"] = output_hasher
        return self.operation_executor.execute(
            kind,
            state.operation_context(
                action=action,
                image_hash=image_hash,
                cached_parent=cached_parent,
            ),
            callback,
            **kwargs,
        )

    def initial_state(
        self,
        *,
        prompts: str | Sequence[str],
        prompt_ids: str | Sequence[str],
        generation_seeds: int | Sequence[int],
        checkpoint_hash: str,
        split: str,
        height: int = 1024,
        width: int = 1024,
        negative_prompt: Optional[str | Sequence[str]] = None,
        original_size: Optional[tuple[int, int]] = None,
        target_size: Optional[tuple[int, int]] = None,
        negative_original_size: Optional[tuple[int, int]] = None,
        negative_target_size: Optional[tuple[int, int]] = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        negative_crops_coords_top_left: tuple[int, int] = (0, 0),
        initial_noise: Optional[Tensor] = None,
    ) -> SdxlEulerState:
        """Encode one prompt batch and create its deterministic initial latent."""
        self._preflight()
        prompt_values = _tuple_strings(prompts, "prompts")
        id_values = _tuple_strings(prompt_ids, "prompt_ids")
        if len(id_values) != len(prompt_values):
            raise ValueError("prompt_ids must match prompts")
        seeds = _tuple_seeds(generation_seeds, len(prompt_values))
        if self.operation_executor is not None and len(prompt_values) != 1:
            raise ValueError("formal training initializes one prompt-seed block at a time")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 8
            for value in (height, width)
        ):
            raise ValueError("height and width must be positive multiples of eight")
        device = self.device
        with torch.no_grad():
            encoded = self.pipeline.encode_prompt(
                prompt=list(prompt_values),
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
            )
        if not isinstance(encoded, (tuple, list)) or len(encoded) != 4:
            raise RuntimeError("SDXL encode_prompt returned an unsupported value")
        positive, negative, pooled, negative_pooled = encoded
        for name, value in (
            ("positive prompt embeddings", positive),
            ("negative prompt embeddings", negative),
            ("positive pooled embeddings", pooled),
            ("negative pooled embeddings", negative_pooled),
        ):
            _require_tensor(value, name, detached=True)
        batch = len(prompt_values)
        if any(value.shape[0] != batch for value in (positive, negative, pooled, negative_pooled)):
            raise RuntimeError("SDXL prompt embedding batches are misaligned")
        prompt_embeds = torch.cat((negative, positive), dim=0).to(device).detach()
        add_text_embeds = torch.cat((negative_pooled, pooled), dim=0).to(device).detach()
        positive_size = original_size or (height, width)
        positive_target = target_size or (height, width)
        positive_time = self.pipeline._get_add_time_ids(
            positive_size,
            crops_coords_top_left,
            positive_target,
            dtype=positive.dtype,
        )
        if negative_original_size is not None or negative_target_size is not None:
            if negative_original_size is None or negative_target_size is None:
                raise ValueError("negative original and target sizes must be supplied together")
            negative_time = self.pipeline._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=positive.dtype,
            )
        else:
            negative_time = positive_time
        add_time_ids = expand_cfg_time_ids(
            torch.cat((negative_time, positive_time), dim=0).to(device),
            batch,
            True,
        ).detach()
        conditioning = SdxlPromptConditioning(
            prompt_ids=id_values,
            prompts=prompt_values,
            generation_seeds=seeds,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled.to(device).detach(),
            add_text_embeds=add_text_embeds,
            add_time_ids=add_time_ids,
            do_classifier_free_guidance=True,
        )
        generators = []
        for seed in seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            generators.append(generator)
        generator_value: Any = generators[0] if len(generators) == 1 else generators
        with torch.no_grad():
            latents = self.pipeline.prepare_latents(
                batch,
                int(self.unet.config.in_channels),
                height,
                width,
                positive.dtype,
                device,
                generator_value,
                initial_noise,
            ).detach()
        return SdxlEulerState(
            latents=latents,
            conditioning=conditioning,
            step_index=0,
            total_steps=self.total_steps,
            split=split,
            branch="prefix",
            prefix=0,
            checkpoint_hash=checkpoint_hash,
        )

    def state_from_tensors(
        self,
        latents: Tensor,
        conditioning: SdxlPromptConditioning,
        *,
        checkpoint_hash: str,
        split: str = "train",
        branch: str = "prefix",
        prefix: int = 0,
        step_index: int = 0,
    ) -> SdxlEulerState:
        """Construct a state from already audited tensors for replay and tests."""
        self._preflight()
        return SdxlEulerState(
            latents=latents,
            conditioning=conditioning,
            step_index=step_index,
            total_steps=self.total_steps,
            split=split,
            branch=branch,
            prefix=prefix,
            checkpoint_hash=checkpoint_hash,
        )

    def clone_state(
        self,
        state: SdxlEulerState,
        *,
        branch: Optional[str] = None,
        preserve_graph: bool = False,
    ) -> SdxlEulerState:
        """Clone tensors and scheduler position so branches cannot alias state."""
        if not isinstance(state, SdxlEulerState):
            raise TypeError("state must be SdxlEulerState")
        latent = state.latents.clone()
        if not preserve_graph:
            latent = latent.detach()
        return replace(
            state,
            latents=latent,
            conditioning=state.conditioning.clone(),
            branch=state.branch if branch is None else branch,
            action_history=copy.deepcopy(state.action_history),
        )

    def observe(self, state: SdxlEulerState, step_index: int) -> EulerRolloutStep:
        """Run exactly one frozen U-Net CFG observation at the explicit state."""
        self._preflight()
        self._validate_state_runtime(state)
        if step_index != state.step_index or step_index >= self.total_steps:
            raise ValueError("observation step differs from the explicit scheduler state")
        scheduler = self._scheduler(state.latents.device)
        timestep = scheduler.timesteps[step_index]
        model_input = expand_cfg_latents(
            state.latents, state.conditioning.do_classifier_free_guidance
        )
        model_input = scheduler.scale_model_input(model_input, timestep)
        added = {
            "text_embeds": state.conditioning.add_text_embeds,
            "time_ids": state.conditioning.add_time_ids,
        }
        physical_calls = 0

        def count_forward(_module: nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
            nonlocal physical_calls
            physical_calls += 1

        handle = self.unet.register_forward_hook(count_forward)

        def forward() -> Tensor:
            capture = self.basis_provider.capture_forward()
            with capture if capture is not None else nullcontext():
                output = self.unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=state.conditioning.prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added,
                    return_dict=False,
                )
            if not isinstance(output, (tuple, list)) or len(output) != 1:
                raise RuntimeError("SDXL U-Net must return one tuple output")
            return output[0]

        try:
            with torch.no_grad():
                prediction = self._execute("unet_forward", state, forward)
        finally:
            handle.remove()
        if physical_calls != 1:
            raise RuntimeError(
                "SDXL observation must execute exactly one physical U-Net forward; "
                f"observed {physical_calls}"
            )
        _require_tensor(prediction, "U-Net prediction", ndim=4, detached=True)
        if state.conditioning.do_classifier_free_guidance:
            negative, positive = split_cfg_noise_pred(prediction)
            prediction = negative + self.guidance_scale * (positive - negative)
        if prediction.shape != state.latents.shape:
            raise RuntimeError("CFG prediction does not match the latent batch")
        sigma_from = scheduler.sigmas[step_index]
        sigma_to = scheduler.sigmas[step_index + 1]
        endpoint = prepare_euler_clean_endpoint(
            state.latents,
            prediction,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            prediction_type=self.prediction_type,
        )
        native_prev = predict_euler_no_churn_prev_sample(
            state.latents,
            prediction,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            prediction_type=self.prediction_type,
        )
        observation = RendererObservation(
            latents_before_step=state.latents.detach(),
            pred_original_sample=endpoint.pred_original_sample.detach(),
            scheduler_update=endpoint.nominal_update.detach(),
            step_index=step_index,
            timestep=timestep.detach(),
            normalized_timestep=state.latents.new_tensor(
                step_index / max(self.total_steps - 1, 1)
            ).detach(),
            pooled_prompt_embeds=state.conditioning.pooled_prompt_embeds.detach(),
        )
        with torch.no_grad():
            condition = self.basis_provider(observation)
        if not isinstance(condition, RendererCondition):
            raise TypeError("basis_provider must return RendererCondition")
        condition = RendererCondition(
            bases=condition.bases.detach(),
            prompt_embedding=(
                None
                if condition.prompt_embedding is None
                else condition.prompt_embedding.detach()
            ),
            state_features=(
                None if condition.state_features is None else condition.state_features.detach()
            ),
        )
        return EulerRolloutStep(
            observation=observation,
            condition=condition,
            native_model_output=prediction.detach(),
            native_prev_sample=native_prev.detach(),
            sigma_from=sigma_from.detach(),
            sigma_to=sigma_to.detach(),
            prediction_type=self.prediction_type,
            clean_update_gain=endpoint.clean_update_gain.detach(),
            metadata={
                "schema": "repldm.sdxl_euler_observation.v1",
                "schedule_sha256": self.schedule_hash,
                "prompt_ids": list(state.conditioning.prompt_ids),
                "branch": state.branch,
                "step_index": step_index,
                "physical_unet_forwards": physical_calls,
            },
        )

    def transition(
        self,
        state: SdxlEulerState,
        step_index: int,
        model_output: Tensor,
        step: EulerRolloutStep,
        *,
        action: Optional[Tensor] = None,
    ) -> EulerTransitionResult:
        """Advance one branch through exactly one real Euler scheduler step."""
        self._preflight()
        if not isinstance(state, SdxlEulerState) or not isinstance(step, EulerRolloutStep):
            raise TypeError("transition requires SdxlEulerState and EulerRolloutStep")
        self._validate_state_runtime(state)
        if step_index != state.step_index or step_index != step.observation.step_index:
            raise ValueError("transition step differs from state or observation")
        if not isinstance(model_output, Tensor) or model_output.shape != state.latents.shape:
            raise ValueError("model_output must match state latents")
        if (
            model_output.device != state.latents.device
            or model_output.dtype != state.latents.dtype
            or not torch.isfinite(model_output).all()
        ):
            raise ValueError("model_output device, dtype, or values are invalid")
        if action is not None and (
            not isinstance(action, Tensor)
            or action.shape != (state.conditioning.batch_size, 6)
            or action.device != state.latents.device
            or not action.is_floating_point()
            or not torch.isfinite(action).all()
        ):
            raise ValueError("decision action must be a finite (batch, 6) tensor")
        is_decision = step_index in self.decision_indices
        if is_decision != (action is not None):
            raise ValueError(
                "an action is required exactly at registered decision indices"
            )
        scheduler = self._scheduler(state.latents.device)
        self._validate_step_runtime(state, step_index, step, scheduler)
        timestep = scheduler.timesteps[step_index]
        # This initializes the scheduler's internal index using the public API.
        scheduler.scale_model_input(state.latents, timestep)

        def advance() -> Tensor:
            output = scheduler.step(
                model_output,
                timestep,
                state.latents,
                s_churn=0.0,
                return_dict=False,
            )
            if not isinstance(output, (tuple, list)) or len(output) < 1:
                raise RuntimeError("Euler scheduler returned an unsupported value")
            return output[0]

        latent = self._execute(
            "scheduler_step", state, advance, action=action
        )
        if not isinstance(latent, Tensor) or latent.shape != state.latents.shape:
            raise RuntimeError("Euler scheduler returned an invalid latent")
        history = tuple(_action_record(action, state.action_history))
        next_state = replace(
            state,
            latents=latent,
            step_index=step_index + 1,
            action_history=history,
        )
        return EulerTransitionResult(state=next_state, latent=latent)

    def _decode_forward(
        self,
        state: SdxlEulerState,
        *,
        action_metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Tensor, Optional[OperationReceipt]]:
        """Run the shared VAE decode path and retain its native receipt."""
        self._preflight()
        if not isinstance(state, SdxlEulerState):
            raise TypeError("state must be SdxlEulerState")

        def decode_call() -> Tensor:
            vae_device = _module_device(self.vae)
            vae_dtype = _module_dtype(self.vae)
            latent = state.latents.to(device=vae_device, dtype=vae_dtype)
            decoded = self.vae.decode(
                latent / self.vae_scaling_factor,
                return_dict=False,
            )
            if not isinstance(decoded, (tuple, list)) or len(decoded) != 1:
                raise RuntimeError("SDXL VAE must return one tuple output")
            return (decoded[0] / 2.0 + 0.5).clamp(0.0, 1.0)

        if self.operation_executor is None:
            image = decode_call()
            receipt = None
        else:
            image, receipt = self.operation_executor.execute_with_receipt(
                "vae_decode",
                state.operation_context(action_metadata=action_metadata),
                decode_call,
            )
        _require_tensor(image, "decoded image", ndim=4, detached=False)
        return image, receipt

    def decode(self, state: SdxlEulerState) -> Tensor:
        """Decode a terminal latent with the frozen differentiable SDXL VAE."""
        image, receipt = self._decode_forward(
            state, action_metadata=getattr(
                self, "_pending_decode_action_metadata", None
            )
        )
        self._last_decode_receipt = receipt
        return image

    def decode_with_receipt(
        self,
        state: SdxlEulerState,
        *,
        action_metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Tensor, OperationReceipt]:
        """Return a formal VAE image and its executor-sealed durable receipt."""
        if self.operation_executor is None:
            raise RuntimeError(
                "decode_with_receipt requires a formal ledgered operation executor"
            )
        image, receipt = self._decode_forward(
            state, action_metadata=action_metadata
        )
        self._last_decode_receipt = receipt
        if receipt is None:
            raise RuntimeError("formal VAE decode omitted its operation receipt")
        return image, receipt

    def _reward_forward(
        self,
        state: SdxlEulerState,
        image: Tensor,
        *,
        parent_receipt: Optional[OperationReceipt] = None,
        action_metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Tensor, Optional[OperationReceipt]]:
        """Run the shared reward-forward path and retain its native receipt."""
        self._preflight()
        if not isinstance(state, SdxlEulerState):
            raise TypeError("state must be SdxlEulerState")
        if self.reward_model is None:
            raise RuntimeError("reward_model is required for reward evaluation")
        _require_tensor(image, "reward image", ndim=4, detached=False)
        if image.shape[0] != state.conditioning.batch_size:
            raise ValueError("reward image batch differs from prompts")

        def reward_call() -> Tensor:
            scorer = getattr(self.reward_model, "score_tensor", None)
            value = (
                scorer(image, state.conditioning.prompts)
                if callable(scorer)
                else self.reward_model(image, state.conditioning.prompts)
            )
            if not isinstance(value, Tensor):
                raise TypeError("tensor reward model must return a Tensor")
            if value.ndim == 2 and value.shape[1] == 1:
                value = value[:, 0]
            if value.ndim != 1 or value.shape[0] != image.shape[0]:
                raise ValueError("tensor reward model must return one score per image")
            if not torch.isfinite(value).all():
                raise RuntimeError("tensor reward model returned non-finite scores")
            return value

        image_hash = tensor_sha256(image)
        if self.operation_executor is None:
            if parent_receipt is not None:
                raise ValueError("an informal reward cannot claim a VAE parent receipt")
            return reward_call(), None
        value, receipt = self.operation_executor.execute_with_receipt(
            "reward_forward",
            state.operation_context(
                action_metadata=action_metadata,
                image_hash=image_hash,
                cached_parent=operation_output_sha256(image),
            ),
            reward_call,
            scalar_or_gradient="scalar",
            parent_receipt=parent_receipt,
        )
        return value, receipt

    def reward(self, state: SdxlEulerState, image: Tensor) -> Tensor:
        """Evaluate the frozen tensor-native reward without detaching image gradients."""
        value, receipt = self._reward_forward(
            state,
            image,
            parent_receipt=getattr(self, "_pending_reward_parent", None),
            action_metadata=getattr(
                self, "_pending_reward_action_metadata", None
            ),
        )
        self._last_reward_receipt = receipt
        if isinstance(value, Tensor) and value.requires_grad:
            if receipt is None:
                return value
            self._reward_receipts[id(value)] = (weakref.ref(value), receipt)
        return value

    def reward_with_receipt(
        self,
        state: SdxlEulerState,
        image: Tensor,
        *,
        parent_receipt: OperationReceipt,
        action_metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Tensor, OperationReceipt]:
        """Return a formal reward durably descended from one VAE decode."""
        if self.operation_executor is None:
            raise RuntimeError(
                "reward_with_receipt requires a formal ledgered operation executor"
            )
        self.operation_executor.validate_receipt(
            parent_receipt,
            kind="vae_decode",
            output_hash=operation_output_sha256(image),
            context=state.operation_context(action_metadata=action_metadata),
            scalar_or_gradient="tensor",
        )
        value, receipt = self._reward_forward(
            state,
            image,
            parent_receipt=parent_receipt,
            action_metadata=action_metadata,
        )
        self._last_reward_receipt = receipt
        if isinstance(value, Tensor) and value.requires_grad and receipt is not None:
            self._reward_receipts[id(value)] = (weakref.ref(value), receipt)
        if receipt is None:
            raise RuntimeError("formal reward execution omitted its operation receipt")
        return value, receipt

    def reward_gradient(
        self,
        state: SdxlEulerState,
        reward: Tensor,
        inputs: Tensor,
        *,
        image_hash: Optional[str] = None,
        action_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tensor:
        """Take one explicitly budgeted reward gradient with respect to inputs."""
        self._preflight()
        if not isinstance(reward, Tensor) or reward.ndim != 1:
            raise ValueError("reward must contain one score per image")
        if not isinstance(inputs, Tensor) or not inputs.requires_grad:
            raise ValueError("reward-gradient inputs must require gradients")
        reward_hash = operation_output_sha256(reward.detach())
        parent_receipt = None
        if self.operation_executor is not None:
            if image_hash is None:
                raise ValueError("formal reward gradients must bind the decoded image hash")
            entry = self._reward_receipts.pop(id(reward), None)
            if entry is None or entry[0]() is not reward:
                raise RuntimeError(
                    "reward gradient is not descended from a successful reward forward"
                )
            parent_receipt = entry[1]
            context = state.operation_context(
                action_metadata=action_metadata,
                image_hash=image_hash,
            )
            parent_context = parent_receipt.context
            if (
                parent_receipt.kind != "reward_forward"
                or parent_receipt.output_hash != reward_hash
                or parent_receipt.image_hash != image_hash
                or any(
                    getattr(parent_context, field) != getattr(context, field)
                    for field in (
                        "split", "prompt", "seed", "step", "prefix", "branch",
                        "action", "checkpoint_hash",
                    )
                )
            ):
                raise RuntimeError(
                    "reward gradient context differs from its reward-forward receipt"
                )

        def backward_call() -> Tensor:
            gradient = torch.autograd.grad(
                reward.sum(),
                inputs,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("reward gradient is missing or non-finite")
            return gradient

        if self.operation_executor is None:
            return backward_call()
        value, _receipt = self.operation_executor.execute_with_receipt(
            "reward_backward",
            state.operation_context(
                action_metadata=action_metadata,
                image_hash=image_hash,
                cached_parent=reward_hash,
            ),
            backward_call,
            scalar_or_gradient="gradient",
            parent_receipt=parent_receipt,
        )
        return value

    def preference_label(
        self,
        state: SdxlEulerState,
        plus_reward: Tensor,
        minus_reward: Tensor,
        *,
        rollout_id: str,
        reward_statistics: Mapping[str, Any],
        reward_statistics_sha256: str,
        plus_image_sha256: str,
        minus_image_sha256: str,
        reward_config_sha256: str,
        reward_preprocess_sha256: str,
        tie_epsilon: float = 1e-6,
    ) -> Mapping[str, Any]:
        """Ledger one immutable label under the frozen reward normalization."""
        self._preflight()
        if plus_reward.numel() != 1 or minus_reward.numel() != 1:
            raise ValueError("preference_label requires scalar branch rewards")
        if tie_epsilon != 1e-6:
            raise ValueError("tie_epsilon differs from the registered protocol")
        if not isinstance(reward_statistics, Mapping):
            raise TypeError("reward_statistics must be the frozen statistics mapping")
        if self.operation_executor is not None:
            contract = self.operation_executor.authorization_binding.contract
            expected = {
                "reward_statistics_sha256": reward_statistics_sha256,
                "reward_config_sha256": reward_config_sha256,
                "reward_preprocess_sha256": reward_preprocess_sha256,
            }
            for field, supplied in expected.items():
                if contract.get(field) != supplied:
                    raise ValueError(f"{field} differs from the authorized run")

        def label_call() -> Mapping[str, Any]:
            plus = float(plus_reward.detach().cpu())
            minus = float(minus_reward.detach().cpu())
            if not math.isfinite(plus) or not math.isfinite(minus):
                raise RuntimeError("preference rewards must be finite")
            if state.conditioning.batch_size != 1:
                raise RuntimeError("formal preference labels require one prompt-seed block")
            label = PreferenceLabelProvenance.from_rewards(
                rollout_id=rollout_id,
                prompt_id=state.conditioning.prompt_ids[0],
                generation_seed=state.conditioning.generation_seeds[0],
                split=state.split,
                plus_reward=plus,
                minus_reward=minus,
                reward_location=reward_statistics.get("location"),
                reward_scale=reward_statistics.get("scale"),
                plus_image_sha256=plus_image_sha256,
                minus_image_sha256=minus_image_sha256,
                reward_statistics_sha256=reward_statistics_sha256,
                reward_config_sha256=reward_config_sha256,
                reward_preprocess_sha256=reward_preprocess_sha256,
                tie_epsilon=tie_epsilon,
            )
            return label.to_dict()

        return self._execute(
            "label",
            state,
            label_call,
            cached_parent=operation_output_sha256(
                (plus_reward.detach(), minus_reward.detach())
            ),
            scalar_or_gradient="logical_label",
        )


__all__ = [
    "SdxlEulerState",
    "SdxlEulerTrainingAdapter",
    "SdxlPromptConditioning",
]
