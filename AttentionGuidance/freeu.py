"""Validated, re-entrant FreeU schedules for frozen diffusion backbones.

FreeU itself is an established backbone/skip reweighting baseline.  This
module only provides a small, auditable schedule abstraction so a future
state-conditioned controller can be compared with the fixed baseline without
mutating UNet state between rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Protocol, Sequence, Tuple, Union


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
