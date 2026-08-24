"""CPU-safe proxy for CFG with orthogonal error correction (CFG-OEC).

This module intentionally contains no pipeline or scheduler integration.  It
implements the prediction-level interface registered in ``S8_CFG_EC_SMOKE_DESIGN.md``
so that the history, batch, and no-op contracts can be tested without model
weights.  The extrapolation horizon is one local step in a *normalized* time
coordinate; it is not an assertion of Euler-sigma physical equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class CFGECConfig:
    """Configuration for the history-only proxy.

    ``guidance_scale`` is the ordinary CFG scale.  ``alignment_threshold``
    gates the proxy correction when the proxy error cosine is below the
    threshold, matching CFG-OEC Eq. (13)--(15).  ``blend`` interpolates the
    gated OEC unconditional prediction with the original unconditional one;
    it is an explicit strength control, so zero is an exact CFG identity.

    The paper writes ``2 * prediction_current - prediction_previous`` for
    adjacent, equally spaced steps.  This implementation computes the same
    expression through a normalized-time finite difference.  By default the
    interval must be exactly one.  Non-unit intervals are accepted only with
    ``allow_normalized_time_proxy=True`` and should be treated as an explicit
    smoke-test ablation, not as a scheduler-consistent extrapolation.
    """

    guidance_scale: float
    alignment_threshold: float
    blend: float
    allow_normalized_time_proxy: bool = False
    time_tolerance: float = 1e-6
    projection_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        guidance_scale = float(self.guidance_scale)
        threshold = float(self.alignment_threshold)
        blend = float(self.blend)
        tolerance = float(self.time_tolerance)
        projection_epsilon = float(self.projection_epsilon)
        if not math.isfinite(guidance_scale):
            raise ValueError("guidance_scale must be finite")
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("alignment_threshold must be finite and in [-1, 1]")
        if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be finite and in [0, 1]")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("time_tolerance must be finite and non-negative")
        if not math.isfinite(projection_epsilon) or projection_epsilon <= 0.0:
            raise ValueError("projection_epsilon must be finite and positive")

    def to_record(self) -> Dict[str, Any]:
        """Return a JSON-safe configuration record."""

        return {
            "guidance_scale": float(self.guidance_scale),
            "alignment_threshold": float(self.alignment_threshold),
            "blend": float(self.blend),
            "allow_normalized_time_proxy": bool(self.allow_normalized_time_proxy),
            "time_tolerance": float(self.time_tolerance),
            "projection_epsilon": float(self.projection_epsilon),
        }


@dataclass(frozen=True)
class CFGECDiagnostics:
    """Finite, row-wise diagnostics returned by :func:`correct_cfg_prediction`.

    Tuple fields have one value per effective sample (the leading tensor
    dimension).  ``history_valid`` is false for the first step or an explicit
    no-op, and ``applied_rows`` records which rows passed the OEC gate.
    """

    history_valid: bool
    applied_rows: Tuple[bool, ...]
    alignment_cosine: Tuple[float, ...]
    correction_norm_ratio: Tuple[float, ...]
    proxy_norm_ratio: Tuple[float, ...]
    time_delta: float
    effective_blend: Tuple[float, ...]
    reason: str

    def to_record(self) -> Dict[str, Any]:
        """Return a JSON-safe record and reject accidental non-finite values."""

        record: Dict[str, Any] = {
            "history_valid": bool(self.history_valid),
            "applied_rows": [bool(value) for value in self.applied_rows],
            "alignment_cosine": [float(value) for value in self.alignment_cosine],
            "correction_norm_ratio": [float(value) for value in self.correction_norm_ratio],
            "proxy_norm_ratio": [float(value) for value in self.proxy_norm_ratio],
            "time_delta": float(self.time_delta),
            "effective_blend": [float(value) for value in self.effective_blend],
            "reason": str(self.reason),
        }
        numeric = (
            record["alignment_cosine"]
            + record["correction_norm_ratio"]
            + record["proxy_norm_ratio"]
            + record["effective_blend"]
            + [record["time_delta"]]
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("CFG-EC diagnostics contain a non-finite value")
        return record


def _ordinary_cfg(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Compute CFG in the same branch used for no-history/no-op parity."""

    return unconditional + (conditional - unconditional) * guidance_scale


def _validate_prediction_pair(
    name: str,
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
) -> None:
    if not isinstance(unconditional, torch.Tensor) or not isinstance(conditional, torch.Tensor):
        raise TypeError(f"{name} predictions must be torch.Tensor instances")
    if unconditional.shape != conditional.shape:
        raise ValueError(f"{name} unconditional and conditional shapes must match")
    if unconditional.ndim < 1:
        raise ValueError(f"{name} predictions must have a leading batch dimension")
    if unconditional.shape[0] <= 0:
        raise ValueError(f"{name} predictions must contain at least one batch row")
    if not unconditional.is_floating_point() or not conditional.is_floating_point():
        raise TypeError(f"{name} predictions must be floating-point tensors")
    if unconditional.dtype != conditional.dtype:
        raise ValueError(f"{name} predictions must have the same dtype")
    if unconditional.device != conditional.device:
        raise ValueError(f"{name} predictions must be on the same device")
    if not bool(torch.isfinite(unconditional).all()) or not bool(torch.isfinite(conditional).all()):
        raise ValueError(f"{name} predictions must be finite")


def _as_finite_scalar(name: str, value: Any) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a scalar") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _row_norm(flat: torch.Tensor) -> torch.Tensor:
    return flat.float().norm(dim=1)


def correct_cfg_prediction(
    current_unconditional: torch.Tensor,
    current_conditional: torch.Tensor,
    previous_unconditional: Optional[torch.Tensor],
    previous_conditional: Optional[torch.Tensor],
    *,
    current_time: Optional[float],
    previous_time: Optional[float],
    config: CFGECConfig,
) -> Tuple[torch.Tensor, CFGECDiagnostics]:
    """Return a proxy-OEC corrected CFG prediction.

    Parameters
    ----------
    current_unconditional, current_conditional:
        Current denoiser predictions shaped ``(B, ...)``.
    previous_unconditional, previous_conditional:
        The immediately preceding pair for the same effective samples.  Both
        must be ``None`` on the first step; that call returns ordinary CFG
        exactly and marks ``history_valid=False``.
    current_time, previous_time:
        Finite, monotonically decreasing *normalized* timestep or sigma
        coordinates.  With the default config, ``previous_time-current_time``
        must equal one.  Opting into non-unit intervals only changes the
        finite-difference normalization; it does not claim physical scheduler
        equivalence.
    config:
        Frozen CFG scale, OEC alignment threshold, and correction blend.

    Formula (registered here, not a pipeline implementation):

    ``hat{p} = p_cur + ((p_cur-p_prev)/(q_cur-q_prev))*(q_cur-q_prev)``
    ``A = c_cur-hat{c}``, ``B = u_cur-hat{u}``
    ``B_perp = B - <A,B>/<A,A> * A``
    ``u_bar = hat{u} + B_perp``
    ``s = cos(A,B)``
    ``u_oec = (1-s) * u_bar + s * u_cur`` when ``s < threshold``
    ``u_out = u_cur + blend * (u_oec-u_cur)``
    ``cfg_out = u_out + guidance_scale * (c_cur-u_out)``.

    The row-wise projection is undefined for a zero conditional proxy error;
    such rows are retained as exact ordinary CFG rows and reported as not
    applied.  No random numbers, model calls, scheduler steps, or cache state
    are created by this pure function.
    """

    if not isinstance(config, CFGECConfig):
        raise TypeError("config must be a CFGECConfig")
    _validate_prediction_pair("current", current_unconditional, current_conditional)
    baseline = _ordinary_cfg(
        current_unconditional, current_conditional, float(config.guidance_scale)
    )
    if not bool(torch.isfinite(baseline).all()):
        raise ValueError("ordinary CFG prediction is non-finite")

    # This is deliberately before any history arithmetic.  It is the exact
    # identity branch required for paired PNG/hash parity.
    if float(config.blend) == 0.0:
        batch = int(current_unconditional.shape[0])
        diagnostics = CFGECDiagnostics(
            history_valid=False,
            applied_rows=(False,) * batch,
            alignment_cosine=(0.0,) * batch,
            correction_norm_ratio=(0.0,) * batch,
            proxy_norm_ratio=(0.0,) * batch,
            time_delta=0.0,
            effective_blend=(0.0,) * batch,
            reason="zero_blend",
        )
        diagnostics.to_record()
        return baseline, diagnostics

    if previous_unconditional is None and previous_conditional is None:
        batch = int(current_unconditional.shape[0])
        diagnostics = CFGECDiagnostics(
            history_valid=False,
            applied_rows=(False,) * batch,
            alignment_cosine=(0.0,) * batch,
            correction_norm_ratio=(0.0,) * batch,
            proxy_norm_ratio=(0.0,) * batch,
            time_delta=0.0,
            effective_blend=(0.0,) * batch,
            reason="no_history",
        )
        diagnostics.to_record()
        return baseline, diagnostics
    if previous_unconditional is None or previous_conditional is None:
        raise ValueError(
            "previous unconditional and conditional predictions must be both present or both None"
        )

    # Check cross-pair metadata before finite reductions so a mismatched
    # device is rejected even for backends that cannot reduce meta tensors.
    if not isinstance(previous_unconditional, torch.Tensor) or not isinstance(
        previous_conditional, torch.Tensor
    ):
        raise TypeError("previous predictions must be torch.Tensor instances")
    if previous_unconditional.shape != current_unconditional.shape:
        raise ValueError("current and previous prediction shapes must match")
    if previous_unconditional.dtype != current_unconditional.dtype:
        raise ValueError("current and previous predictions must have the same dtype")
    if previous_unconditional.device != current_unconditional.device:
        raise ValueError("current and previous predictions must be on the same device")
    _validate_prediction_pair("previous", previous_unconditional, previous_conditional)

    current_q = _as_finite_scalar("current_time", current_time)
    previous_q = _as_finite_scalar("previous_time", previous_time)
    time_delta = current_q - previous_q
    if time_delta >= 0.0:
        raise ValueError("previous_time must be greater than current_time")
    if abs(time_delta) <= float(config.time_tolerance):
        raise ValueError("normalized time interval is too small")
    if not config.allow_normalized_time_proxy and not math.isclose(
        abs(time_delta), 1.0, rel_tol=0.0, abs_tol=float(config.time_tolerance)
    ):
        raise ValueError(
            "non-unit normalized time interval is rejected; set "
            "allow_normalized_time_proxy=True for the registered ablation"
        )

    # Work in fp32 for the projection and diagnostics, then cast once at the
    # output boundary.  The no-op branches above never take this path.
    work_dtype = torch.float32
    shape = current_unconditional.shape
    current_u = current_unconditional.to(dtype=work_dtype).reshape(shape[0], -1)
    current_c = current_conditional.to(dtype=work_dtype).reshape(shape[0], -1)
    previous_u = previous_unconditional.to(dtype=work_dtype).reshape(shape[0], -1)
    previous_c = previous_conditional.to(dtype=work_dtype).reshape(shape[0], -1)

    # Explicit normalized-time finite differences.  The extrapolation horizon
    # is one local interval, yielding the paper's 2*current-previous proxy in
    # any opted-in normalized coordinate.
    derivative_u = (current_u - previous_u) / time_delta
    derivative_c = (current_c - previous_c) / time_delta
    proxy_u = current_u + derivative_u * time_delta
    proxy_c = current_c + derivative_c * time_delta
    a = current_c - proxy_c
    b = current_u - proxy_u
    a_norm = _row_norm(a)
    b_norm = _row_norm(b)
    dot = (a * b).sum(dim=1)
    denominator = a_norm * b_norm
    cosine = torch.where(
        denominator > float(config.projection_epsilon),
        dot / denominator.clamp_min(float(config.projection_epsilon)),
        torch.zeros_like(dot),
    ).clamp(min=-1.0, max=1.0)
    valid = a_norm > float(config.projection_epsilon)
    valid = valid & (b_norm > float(config.projection_epsilon))
    gate = valid & (cosine < float(config.alignment_threshold))

    projection_coeff = dot / a_norm.square().clamp_min(float(config.projection_epsilon))
    b_perp = b - projection_coeff[:, None] * a
    corrected_unconditional = proxy_u + b_perp

    # OEC's dynamic mix uses s directly.  ``blend`` scales the complete
    # dynamic correction and makes a registered zero-strength identity.
    dynamic_weight = torch.where(gate, cosine, torch.ones_like(cosine))
    # Do not multiply a non-finite ungated projection by zero: ``0 * inf``
    # would poison an otherwise valid no-op row.  Ungated rows are defined to
    # be exact ordinary CFG rows, so replace their candidate before mixing.
    safe_corrected_unconditional = torch.where(
        gate[:, None], corrected_unconditional, current_u
    )
    oec_unconditional = (1.0 - dynamic_weight)[:, None] * safe_corrected_unconditional
    oec_unconditional = oec_unconditional + dynamic_weight[:, None] * current_u
    applied_weight = torch.where(
        gate, float(config.blend) * (1.0 - cosine), torch.zeros_like(cosine)
    )
    output_u = current_u + float(config.blend) * (oec_unconditional - current_u)
    output = (output_u + float(config.guidance_scale) * (current_c - output_u)).reshape(shape)
    if not bool(torch.isfinite(output).all()):
        raise ValueError("CFG-EC corrected prediction is non-finite")
    output = output.to(dtype=current_unconditional.dtype)
    if not bool(torch.isfinite(output).all()):
        raise ValueError("CFG-EC output cast is non-finite")

    baseline_work = baseline.to(dtype=work_dtype).reshape(shape[0], -1)
    correction_norm = _row_norm(output.reshape(shape[0], -1).to(dtype=work_dtype) - baseline_work)
    baseline_norm = _row_norm(baseline_work).clamp_min(float(config.projection_epsilon))
    proxy_norm = _row_norm((proxy_u - current_u))
    current_norm = _row_norm(current_u).clamp_min(float(config.projection_epsilon))
    correction_ratio = correction_norm / baseline_norm
    proxy_ratio = proxy_norm / current_norm
    diagnostics = CFGECDiagnostics(
        history_valid=True,
        applied_rows=tuple(bool(value) for value in gate.cpu().tolist()),
        alignment_cosine=tuple(float(value) for value in cosine.cpu().tolist()),
        correction_norm_ratio=tuple(float(value) for value in correction_ratio.cpu().tolist()),
        proxy_norm_ratio=tuple(float(value) for value in proxy_ratio.cpu().tolist()),
        time_delta=float(time_delta),
        effective_blend=tuple(float(value) for value in applied_weight.cpu().tolist()),
        reason="applied" if bool(gate.any()) else "alignment_gate",
    )
    diagnostics.to_record()
    return output, diagnostics


# A descriptive alias keeps the smoke name close to the experiment matrix.
apply_cfg_ec_proxy = correct_cfg_prediction


__all__ = [
    "CFGECConfig",
    "CFGECDiagnostics",
    "apply_cfg_ec_proxy",
    "correct_cfg_prediction",
]
