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
    "trajectory_cone_tangent",
    "trajectory_cone_tangent_rescaled",
]
MOMENT_TANGENT_MODES = frozenset(
    {
        "moment_tangent",
        "moment_tangent_rescaled",
        "trajectory_cone_tangent",
        "trajectory_cone_tangent_rescaled",
    }
)
TRAJECTORY_CONE_MODES = frozenset(
    {"trajectory_cone_tangent", "trajectory_cone_tangent_rescaled"}
)
RESIDUAL_MODES = frozenset(
    {"raw", "mean_centered", *MOMENT_TANGENT_MODES}
)
