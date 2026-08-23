"""Typed inference-time control interface for Attention Guidance."""
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from torch import Tensor

from .types import (
    BandScales,
    MOMENT_TANGENT_MODES,
    RESIDUAL_MODES,
    ResidualMode,
    Scale,
)


@dataclass(frozen=True)
class GuidanceObservation:
    """State exposed after one scheduler step and before Attention Guidance."""

    step_index: int
    t_index: int
    timestep: Tensor
    alpha_t: Tensor
    latents: Tensor
    denoising_update: Tensor
    pooled_prompt_embeds: Optional[Tensor] = None


@dataclass(frozen=True)
class GuidanceAction:
    """A scalar or low/mid/high action with an optional residual geometry."""

    scale: Optional[Scale] = None
    band_scales: Optional[BandScales] = None
    max_update_ratio: Optional[Scale] = None
    residual_mode: ResidualMode = "raw"

    def __post_init__(self) -> None:
        if (self.scale is None) == (self.band_scales is None):
            raise ValueError("exactly one of scale or band_scales must be provided")
        if self.residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"unsupported residual_mode {self.residual_mode!r}")
        if self.band_scales is not None and self.residual_mode != "raw":
            raise ValueError("frequency-band actions only support residual_mode='raw'")
        if (
            self.max_update_ratio is not None
            and self.residual_mode in MOMENT_TANGENT_MODES
        ):
            raise ValueError(
                "max_update_ratio is not defined for moment-tangent geodesic updates"
            )


class GuidanceController(Protocol):
    def __call__(self, observation: GuidanceObservation) -> Optional[GuidanceAction]:
        ...


class ConstantGuidanceController:
    """Return the same action at every sampling step."""

    def __init__(
        self,
        *,
        scale: Optional[Scale] = None,
        band_scales: Optional[BandScales] = None,
        max_update_ratio: Optional[Scale] = None,
        residual_mode: ResidualMode = "raw",
    ) -> None:
        self.action = GuidanceAction(
            scale=scale,
            band_scales=band_scales,
            max_update_ratio=max_update_ratio,
            residual_mode=residual_mode,
        )

    def __call__(self, observation: GuidanceObservation) -> GuidanceAction:
        return self.action


class ScheduleGuidanceController:
    """Index a scalar or three-band schedule in sampling order (early to late)."""

    def __init__(
        self,
        *,
        scale_schedule: Optional[Sequence[Scale]] = None,
        band_schedule: Optional[Sequence[BandScales]] = None,
        max_update_ratio: Optional[Scale] = None,
        residual_mode: ResidualMode = "raw",
    ) -> None:
        if (scale_schedule is None) == (band_schedule is None):
            raise ValueError("exactly one schedule must be provided")
        self.scale_schedule = scale_schedule
        self.band_schedule = band_schedule
        self.max_update_ratio = max_update_ratio
        self.residual_mode = residual_mode
        if residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"unsupported residual_mode {residual_mode!r}")
        if band_schedule is not None and residual_mode != "raw":
            raise ValueError("frequency-band schedules only support residual_mode='raw'")
        if (
            max_update_ratio is not None
            and residual_mode in MOMENT_TANGENT_MODES
        ):
            raise ValueError(
                "max_update_ratio is not defined for moment-tangent geodesic updates"
            )

    def __call__(self, observation: GuidanceObservation) -> GuidanceAction:
        schedule = self.scale_schedule if self.scale_schedule is not None else self.band_schedule
        if observation.step_index >= len(schedule):
            raise IndexError(
                f"schedule has {len(schedule)} steps, got step_index={observation.step_index}"
            )
        value = schedule[observation.step_index]
        if self.scale_schedule is not None:
            return GuidanceAction(
                scale=value,
                max_update_ratio=self.max_update_ratio,
                residual_mode=self.residual_mode,
            )
        return GuidanceAction(
            band_scales=value,
            max_update_ratio=self.max_update_ratio,
            residual_mode=self.residual_mode,
        )
