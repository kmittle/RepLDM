"""Validated, re-entrant FreeU schedules for frozen diffusion backbones.

FreeU itself is an established backbone/skip reweighting baseline.  This
module only provides a small, auditable schedule abstraction so a future
state-conditioned controller can be compared with the fixed baseline without
mutating UNet state between rollouts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterable, Iterator, Protocol, Sequence, Tuple, Union

import torch


DIFFUSERS_FREEU_IMPLEMENTATION = "diffusers_constant_v1"
PAPER_FREEU_IMPLEMENTATION = "paper_adaptive_3676d36"
FREEU_IMPLEMENTATIONS = frozenset(
    {DIFFUSERS_FREEU_IMPLEMENTATION, PAPER_FREEU_IMPLEMENTATION}
)
PAPER_FREEU_SOURCE_COMMIT = "3676d3652a44101f9cca030c33f82756dab249d7"
PAPER_FREEU_SDXL_PARAMETERS = (0.9, 0.2, 1.3, 1.4)
PAPER_FREEU_PORT_DIFFUSERS_VERSION = "0.32.1"


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


def apply_paper_freeu(
    resolution_idx: int,
    hidden_states: torch.Tensor,
    res_hidden_states: torch.Tensor,
    **freeu_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the spatially adaptive FreeU transform from the paper code.

    The author's README normalizes the channel-mean backbone activation over
    space before scaling the first half of the channels. Diffusers and the
    author's runnable demo instead use a constant multiplier. The zero-range
    branch below is the only numerical safeguard added to the published
    formula; it leaves a spatially constant backbone unchanged.
    """

    if hidden_states.ndim != 4 or res_hidden_states.ndim != 4:
        raise ValueError("paper FreeU requires NCHW backbone and skip tensors")
    if hidden_states.shape[0] != res_hidden_states.shape[0]:
        raise ValueError("paper FreeU backbone and skip batch sizes must match")
    hidden_channels = int(hidden_states.shape[1])
    if hidden_channels == 1280:
        suffix = "1"
    elif hidden_channels == 640:
        suffix = "2"
    else:
        return hidden_states, res_hidden_states
    backbone_scale = float(freeu_kwargs[f"b{suffix}"])
    skip_scale = float(freeu_kwargs[f"s{suffix}"])
    if not math.isfinite(backbone_scale) or not math.isfinite(skip_scale):
        raise ValueError("paper FreeU scales must be finite")

    spatial_mean = hidden_states.mean(dim=1, keepdim=True)
    flat_mean = spatial_mean.flatten(1)
    spatial_min = flat_mean.amin(dim=1).reshape(-1, 1, 1, 1)
    spatial_max = flat_mean.amax(dim=1).reshape(-1, 1, 1, 1)
    spatial_range = spatial_max - spatial_min
    safe_range = torch.where(
        spatial_range > 0,
        spatial_range,
        torch.ones_like(spatial_range),
    )
    normalized = (spatial_mean - spatial_min) / safe_range
    modulation = 1.0 + (backbone_scale - 1.0) * normalized
    half_channels = int(hidden_states.shape[1]) // 2
    if half_channels:
        hidden_states = torch.cat(
            (
                hidden_states[:, :half_channels] * modulation,
                hidden_states[:, half_channels:],
            ),
            dim=1,
        )

    try:
        from diffusers.utils.torch_utils import fourier_filter
    except ImportError as exc:  # pragma: no cover - diffusers is a runtime dependency
        raise RuntimeError("paper FreeU requires diffusers' Fourier filter") from exc
    res_hidden_states = fourier_filter(
        res_hidden_states, threshold=1, scale=skip_scale
    )
    return hidden_states, res_hidden_states


@contextmanager
def installed_freeu_implementation(implementation: str) -> Iterator[dict]:
    """Install and count one process-local FreeU tensor operator."""

    implementation = str(implementation)
    if implementation not in FREEU_IMPLEMENTATIONS:
        raise ValueError(
            f"unsupported FreeU implementation {implementation!r}; "
            f"choose one of {sorted(FREEU_IMPLEMENTATIONS)}"
        )
    try:
        import diffusers
        from diffusers.models.unets import unet_2d_blocks
    except ImportError as exc:  # pragma: no cover - diffusers is a runtime dependency
        raise RuntimeError("FreeU requires diffusers' 2D U-Net blocks") from exc
    runtime_version = str(getattr(diffusers, "__version__", "unknown"))
    if (
        implementation == PAPER_FREEU_IMPLEMENTATION
        and runtime_version != PAPER_FREEU_PORT_DIFFUSERS_VERSION
    ):
        raise RuntimeError(
            "paper FreeU operator port requires diffusers "
            f"{PAPER_FREEU_PORT_DIFFUSERS_VERSION}, got {runtime_version}"
        )
    original = unet_2d_blocks.apply_freeu
    target = (
        apply_paper_freeu
        if implementation == PAPER_FREEU_IMPLEMENTATION
        else original
    )
    runtime = {
        "implementation": implementation,
        "operator_calls_total": 0,
        "resolution_idx_call_counts": {},
        "hidden_channel_call_counts": {},
        "resolution_channel_call_counts": {},
        "operator_effect_call_counts": {},
    }

    def counted_apply_freeu(
        resolution_idx: int,
        hidden_states: torch.Tensor,
        res_hidden_states: torch.Tensor,
        **freeu_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(int(resolution_idx))
        runtime["operator_calls_total"] += 1
        counts = runtime["resolution_idx_call_counts"]
        counts[key] = counts.get(key, 0) + 1
        hidden_channels = int(hidden_states.shape[1])
        channel_key = str(hidden_channels)
        channel_counts = runtime["hidden_channel_call_counts"]
        channel_counts[channel_key] = channel_counts.get(channel_key, 0) + 1
        joint_key = f"{key}:{hidden_channels}"
        joint_counts = runtime["resolution_channel_call_counts"]
        joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
        if implementation == PAPER_FREEU_IMPLEMENTATION:
            effect = (
                "b1_s1"
                if hidden_channels == 1280
                else "b2_s2"
                if hidden_channels == 640
                else "no_op"
            )
        else:
            effect = "b1_s1" if int(resolution_idx) == 0 else (
                "b2_s2" if int(resolution_idx) == 1 else "no_op"
            )
        effect_counts = runtime["operator_effect_call_counts"]
        effect_counts[effect] = effect_counts.get(effect, 0) + 1
        return target(
            resolution_idx,
            hidden_states,
            res_hidden_states,
            **freeu_kwargs,
        )

    unet_2d_blocks.apply_freeu = counted_apply_freeu
    try:
        yield runtime
    finally:
        unet_2d_blocks.apply_freeu = original


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
