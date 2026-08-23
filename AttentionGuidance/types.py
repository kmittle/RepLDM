"""Shared type definitions for Attention Guidance actions and operators."""
from typing import Literal, Sequence, Union

from torch import Tensor


Scale = Union[float, Tensor]
BandScales = Union[Sequence[Scale], Tensor]
ResidualMode = Literal[
    "raw",
    "mean_centered",
    "moment_tangent",
    "moment_tangent_rescaled",
]
RESIDUAL_MODES = frozenset(
    {"raw", "mean_centered", "moment_tangent", "moment_tangent_rescaled"}
)
