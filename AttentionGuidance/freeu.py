"""Validated, re-entrant FreeU schedules for frozen diffusion backbones.

FreeU itself is an established backbone/skip reweighting baseline.  This
module only provides a small, auditable schedule abstraction so a future
state-conditioned controller can be compared with the fixed baseline without
mutating UNet state between rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol, Sequence, Tuple, Union

import torch


class _FreeUModel(Protocol):
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float) -> None:
        ...

    def disable_freeu(self) -> None:
        ...


@dataclass(frozen=True)
class FreeUParameters:
    """One FreeU setting in the order used by diffusers."""

    s1: float
    s2: float
    b1: float
    b2: float

    def __post_init__(self) -> None:
        values = (self.s1, self.s2, self.b1, self.b2)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("FreeU parameters must be finite")
        # Keep schedules inside a conservative trust region.  Values outside
        # this range are generally either a sign error or an uncontrolled
        # replacement for changing CFG/NFE and need explicit code changes.
        if not 0.0 <= float(self.s1) <= 2.0 or not 0.0 <= float(self.s2) <= 2.0:
            raise ValueError("FreeU skip scales must be in [0, 2]")
        if not 0.0 <= float(self.b1) <= 3.0 or not 0.0 <= float(self.b2) <= 3.0:
            raise ValueError("FreeU backbone scales must be in [0, 3]")

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "FreeUParameters":
        if len(values) != 4:
            raise ValueError("FreeU parameters require [s1, s2, b1, b2]")
        return cls(*(float(value) for value in values))

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (float(self.s1), float(self.s2), float(self.b1), float(self.b2))


@dataclass(frozen=True)
class FreeUKnot:
    """A normalized denoising position and its FreeU setting."""

    position: float
    parameters: FreeUParameters

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.position)) or not 0.0 <= float(self.position) <= 1.0:
            raise ValueError("FreeU knot positions must be finite and in [0, 1]")


class FreeUSchedule:
    """Piecewise-linear, re-entrant FreeU schedule.

    ``position=0`` is the first (highest-noise) denoising step and ``position=1``
    is the final step.  A single knot is a constant baseline.  Interpolation is
    performed in Python scalars before passing values to diffusers, so repeated
    rollouts are deterministic and do not consume an iterator.
    """

    def __init__(self, knots: Iterable[Union[FreeUKnot, Tuple[float, Sequence[float]]]]):
        normalized = []
        for knot in knots:
            if isinstance(knot, FreeUKnot):
                normalized.append(knot)
            else:
                if len(knot) != 2:
                    raise ValueError("each FreeU knot requires (position, parameters)")
                normalized.append(
                    FreeUKnot(float(knot[0]), FreeUParameters.from_sequence(knot[1]))
                )
        if not normalized:
            raise ValueError("FreeU schedule requires at least one knot")
        normalized.sort(key=lambda item: item.position)
        if any(a.position == b.position for a, b in zip(normalized, normalized[1:])):
            raise ValueError("FreeU knot positions must be unique")
        if normalized[0].position != 0.0 or normalized[-1].position != 1.0:
            raise ValueError("FreeU schedule must include knots at positions 0 and 1")
        self._knots = tuple(normalized)

    @classmethod
    def constant(cls, parameters: Sequence[float]) -> "FreeUSchedule":
        value = FreeUParameters.from_sequence(parameters)
        return cls(((0.0, value.as_tuple()), (1.0, value.as_tuple())))

    @property
    def knots(self) -> Tuple[FreeUKnot, ...]:
        return self._knots

    def at(self, position: float) -> FreeUParameters:
        position = float(position)
        if not math.isfinite(position):
            raise ValueError("FreeU schedule position must be finite")
        position = min(max(position, 0.0), 1.0)
        for left, right in zip(self._knots, self._knots[1:]):
            if position <= right.position:
                span = right.position - left.position
                weight = 0.0 if span == 0 else (position - left.position) / span
                return FreeUParameters(
                    *(a + weight * (b - a) for a, b in zip(left.parameters.as_tuple(), right.parameters.as_tuple()))
                )
        return self._knots[-1].parameters

    def apply(self, unet: _FreeUModel, step_index: int, total_steps: int) -> FreeUParameters:
        if int(total_steps) <= 0:
            raise ValueError("total_steps must be positive")
        if int(step_index) < 0 or int(step_index) >= int(total_steps):
            raise ValueError("step_index must be in [0, total_steps)")
        position = float(step_index) / max(int(total_steps) - 1, 1)
        parameters = self.at(position)
        enable = getattr(unet, "enable_freeu", None)
        if enable is None or not callable(enable):
            raise TypeError("the denoiser does not expose diffusers' enable_freeu API")
        enable(*parameters.as_tuple())
        return parameters

    def disable(self, unet: _FreeUModel) -> None:
        disable = getattr(unet, "disable_freeu", None)
        if disable is None or not callable(disable):
            raise TypeError("the denoiser does not expose diffusers' disable_freeu API")
        disable()

    def to_record(self) -> dict:
        return {
            "knots": [
                {"position": knot.position, "parameters": list(knot.parameters.as_tuple())}
                for knot in self._knots
            ]
        }


def match_channel_moments(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Match per-channel spatial mean and RMS without changing tensor shape.

    The projection is an inference-time guard for structural interventions.  It
    preserves the candidate's spatial arrangement while removing first/second
    moment changes that can otherwise appear as global contrast or saturation.
    Computation uses float32 for half-precision inputs and returns the original
    dtype.
    """

    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have identical shapes")
    if candidate.ndim < 3:
        raise ValueError("moment matching requires a channel and spatial dimension")
    if not candidate.is_floating_point() or not reference.is_floating_point():
        raise TypeError("moment matching requires floating-point tensors")

    work_dtype = torch.float32 if candidate.dtype in (torch.float16, torch.bfloat16) else candidate.dtype
    candidate_work = candidate.to(dtype=work_dtype)
    reference_work = reference.to(dtype=work_dtype)
    spatial_dims = tuple(range(2, candidate.ndim))
    reference_mean = reference_work.mean(dim=spatial_dims, keepdim=True)
    candidate_mean = candidate_work.mean(dim=spatial_dims, keepdim=True)
    reference_centered = reference_work - reference_mean
    candidate_centered = candidate_work - candidate_mean
    reference_rms = reference_centered.square().mean(dim=spatial_dims, keepdim=True).sqrt()
    candidate_rms = candidate_centered.square().mean(dim=spatial_dims, keepdim=True).sqrt()
    finite_rms = torch.finfo(work_dtype).eps
    scale = torch.where(
        candidate_rms > finite_rms,
        reference_rms / candidate_rms.clamp_min(finite_rms),
        torch.ones_like(candidate_rms),
    )
    projected = candidate_centered * scale + reference_mean
    return projected.to(dtype=candidate.dtype)


class MomentPreservingFreeUController:
    """Apply FreeU's structural transform while preserving feature moments.

    Diffusers applies FreeU inside each up block through a module-level helper.
    Forward pre-hooks let this controller apply the same backbone/skip
    transform, then match each transformed feature's spatial mean and RMS to
    its unmodified value.  The scheduler trajectory is therefore untouched;
    only the feature-space structural redistribution remains.
    """

    _ATTRIBUTE = "_repldm_moment_preserving_freeu_controller"

    def __init__(self, unet) -> None:
        self.unet = unet
        self._parameters = FreeUParameters(1.0, 1.0, 1.0, 1.0)
        self._handles = []
        try:
            from diffusers.utils.torch_utils import fourier_filter
        except ImportError as exc:  # pragma: no cover - diffusers is an runtime dependency
            raise RuntimeError("moment-preserving FreeU requires diffusers' fourier_filter") from exc
        self._fourier_filter = fourier_filter
        blocks = getattr(unet, "up_blocks", None)
        if not blocks:
            raise TypeError("the denoiser must expose non-empty up_blocks")
        for block_index, block in enumerate(blocks):
            resolution_index = int(getattr(block, "resolution_idx", block_index))
            register = getattr(block, "register_forward_pre_hook", None)
            if register is None or not callable(register):
                raise TypeError("up blocks must support forward pre-hooks")
            handle = register(
                lambda module, args, kwargs, index=resolution_index: self._pre_hook(
                    module, args, kwargs, index
                ),
                with_kwargs=True,
            )
            self._handles.append(handle)
        setattr(unet, self._ATTRIBUTE, self)

    @classmethod
    def clear(cls, unet) -> None:
        existing = getattr(unet, cls._ATTRIBUTE, None)
        if existing is not None:
            existing.disable()

    def apply(
        self, schedule: FreeUSchedule, step_index: int, total_steps: int
    ) -> FreeUParameters:
        if not isinstance(schedule, FreeUSchedule):
            raise TypeError("schedule must be a FreeUSchedule")
        if int(total_steps) <= 0 or not 0 <= int(step_index) < int(total_steps):
            raise ValueError("step_index must be in [0, total_steps)")
        self._parameters = schedule.at(
            float(step_index) / max(int(total_steps) - 1, 1)
        )
        # The controller performs the transform itself; never leave
        # diffusers' in-place FreeU helper active at the same time.
        self.unet.disable_freeu()
        return self._parameters

    def _pre_hook(self, module, args, kwargs, resolution_index):
        if resolution_index not in (0, 1):
            return args, kwargs
        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
            hidden_in_args = False
        elif args:
            hidden_states = args[0]
            hidden_in_args = True
        else:
            return args, kwargs
        if "res_hidden_states_tuple" in kwargs:
            skip_states = kwargs["res_hidden_states_tuple"]
            skips_in_args = False
        elif len(args) > 1:
            skip_states = args[1]
            skips_in_args = True
        else:
            return args, kwargs

        backbone_scale = self._parameters.b1 if resolution_index == 0 else self._parameters.b2
        skip_scale = self._parameters.s1 if resolution_index == 0 else self._parameters.s2
        half_channels = int(hidden_states.shape[1]) // 2
        if half_channels > 0:
            scaled = hidden_states[:, :half_channels] * backbone_scale
            scaled = match_channel_moments(scaled, hidden_states[:, :half_channels])
            hidden_states = torch.cat((scaled, hidden_states[:, half_channels:]), dim=1)

        transformed_skips = []
        for skip in skip_states:
            filtered = self._fourier_filter(skip, threshold=1, scale=skip_scale)
            transformed_skips.append(match_channel_moments(filtered, skip))
        transformed_skips = tuple(transformed_skips)

        new_args = args
        new_kwargs = dict(kwargs)
        if hidden_in_args:
            args_list = list(new_args)
            args_list[0] = hidden_states
            if skips_in_args:
                args_list[1] = transformed_skips
            new_args = tuple(args_list)
        else:
            new_kwargs["hidden_states"] = hidden_states
        if not skips_in_args:
            new_kwargs["res_hidden_states_tuple"] = transformed_skips
        return new_args, new_kwargs

    def disable(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if getattr(self.unet, self._ATTRIBUTE, None) is self:
            delattr(self.unet, self._ATTRIBUTE)
        self.unet.disable_freeu()
