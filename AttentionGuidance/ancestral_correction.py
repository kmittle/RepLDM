"""Bounded Euler-to-ancestral ablations for Euler sampling.

The correction is deliberately kept outside the denoiser.  Given the Euler
predictor output and the scheduler's current noise levels, it interpolates
between the deterministic Euler transition and the corresponding
Euler-Ancestral transition.  Only the endpoints have exact scheduler
semantics; intermediate ``mix`` values are a registered bounded ablation.
This keeps the frozen U-Net, CFG, timestep sequence, and initial noise
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch


_NOISE_MODES = frozenset(("sqrt", "linear", "none"))


@dataclass(frozen=True)
class TrajectoryCorrectionConfig:
    """Configuration for an Euler-compatible ancestral ablation.

    ``mix=0`` is an exact no-op.  With no trust cap, ``mix=1`` and
    ``noise_mode='sqrt'`` gives the Euler-Ancestral transition for an
    epsilon-prediction scheduler.  Intermediate values are bounded
    interpolation controls rather than a new SDE. ``max_correction_ratio``
    bounds the correction norm relative to the ordinary Euler update on each
    sample.
    """

    mix: float
    noise_mode: str = "sqrt"
    max_correction_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        mix = float(self.mix)
        if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
            raise ValueError("trajectory correction mix must be finite and in [0, 1]")
        if self.noise_mode not in _NOISE_MODES:
            raise ValueError(
                "trajectory correction noise_mode must be one of 'sqrt', 'linear', or 'none'"
            )
        if self.max_correction_ratio is not None:
            ratio = float(self.max_correction_ratio)
            if not math.isfinite(ratio) or ratio < 0.0:
                raise ValueError("max_correction_ratio must be finite and non-negative")

    def to_record(self) -> dict:
        return {
            "mix": float(self.mix),
            "noise_mode": str(self.noise_mode),
            "max_correction_ratio": (
                None
                if self.max_correction_ratio is None
                else float(self.max_correction_ratio)
            ),
        }


@dataclass(frozen=True)
class TrajectoryCorrectionDiagnostics:
    """Small JSON-safe summary emitted for every corrected scheduler step."""

    step_index: int
    mix: float
    sigma_from: float
    sigma_to: float
    sigma_up: float
    raw_correction_norm_ratio: float
    applied_correction_norm_ratio: float
    capped: bool

    def to_record(self) -> dict:
        return {
            "step_index": int(self.step_index),
            "mix": float(self.mix),
            "sigma_from": float(self.sigma_from),
            "sigma_to": float(self.sigma_to),
            "sigma_up": float(self.sigma_up),
            "raw_correction_norm_ratio": float(self.raw_correction_norm_ratio),
            "applied_correction_norm_ratio": float(self.applied_correction_norm_ratio),
            "capped": bool(self.capped),
        }


def _prediction_type(scheduler) -> str:
    config = getattr(scheduler, "config", None)
    if config is None:
        return "epsilon"
    if isinstance(config, dict):
        return str(config.get("prediction_type", "epsilon"))
    return str(getattr(config, "prediction_type", "epsilon"))


def _randn_like(sample: torch.Tensor, generator=None) -> torch.Tensor:
    """Draw noise while accepting both device-local and CPU generators."""

    kwargs = {
        "size": tuple(sample.shape),
        "device": sample.device,
        "dtype": sample.dtype,
    }
    try:
        return torch.randn(generator=generator, **kwargs)
    except RuntimeError as exc:
        # A CPU generator is commonly supplied while the pipeline holds the
        # latents on CUDA.  Draw on the generator's device, then transfer.
        if not isinstance(generator, torch.Generator):
            raise
        generator_device = getattr(generator, "device", torch.device("cpu"))
        if generator_device == sample.device:
            raise exc
        return torch.randn(generator=generator, device=generator_device, dtype=sample.dtype, size=tuple(sample.shape)).to(
            sample.device
        )


def _batch_norm(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        return value.abs().reshape(1)
    return value.float().reshape(value.shape[0], -1).norm(dim=1)


def apply_ancestral_correction(
    *,
    scheduler,
    sample: torch.Tensor,
    pred_original_sample: torch.Tensor,
    euler_prev_sample: torch.Tensor,
    step_index: int,
    config: TrajectoryCorrectionConfig,
    generator=None,
) -> Tuple[torch.Tensor, TrajectoryCorrectionDiagnostics]:
    """Apply one bounded Euler-to-ancestral transition.

    The caller must invoke the ordinary scheduler step first and pass its
    ``pred_original_sample`` and ``prev_sample``.  Only Euler-compatible
    schedulers exposing a monotonically decreasing ``sigmas`` sequence and
    epsilon prediction are accepted.  The ordinary scheduler output is
    returned unchanged when ``config.mix == 0``; in particular, no RNG state
    is consumed in that case.
    """

    if not isinstance(config, TrajectoryCorrectionConfig):
        raise TypeError("config must be a TrajectoryCorrectionConfig")
    if sample.shape != pred_original_sample.shape or sample.shape != euler_prev_sample.shape:
        raise ValueError("sample, pred_original_sample, and euler_prev_sample must have identical shapes")
    if not sample.is_floating_point():
        raise TypeError("trajectory correction requires floating-point tensors")
    step_index = int(step_index)

    # This branch is intentionally before scheduler/tensor arithmetic.  It is
    # the parity contract used by paired evaluation and regression tests.
    if float(config.mix) == 0.0:
        return euler_prev_sample, TrajectoryCorrectionDiagnostics(
            step_index=step_index,
            mix=0.0,
            sigma_from=0.0,
            sigma_to=0.0,
            sigma_up=0.0,
            raw_correction_norm_ratio=0.0,
            applied_correction_norm_ratio=0.0,
            capped=False,
        )

    if _prediction_type(scheduler) != "epsilon":
        raise ValueError("ancestral correction currently requires prediction_type='epsilon'")
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or len(sigmas) <= step_index + 1:
        raise ValueError("scheduler must expose sigmas through the requested next step")
    if step_index < 0:
        raise ValueError("step_index must be non-negative")

    work_dtype = torch.float32
    sample_work = sample.to(dtype=work_dtype)
    x0_work = pred_original_sample.to(dtype=work_dtype)
    euler_work = euler_prev_sample.to(dtype=work_dtype)
    sigma_from = torch.as_tensor(sigmas[step_index], device=sample.device, dtype=work_dtype)
    sigma_to = torch.as_tensor(sigmas[step_index + 1], device=sample.device, dtype=work_dtype)
    if not bool(torch.isfinite(sigma_from) and torch.isfinite(sigma_to)):
        raise ValueError("scheduler sigmas must be finite")
    if float(sigma_from) <= 0.0 or float(sigma_to) < 0.0 or float(sigma_to) > float(sigma_from):
        raise ValueError("scheduler sigmas must satisfy 0 < sigma_to <= sigma_from")

    sigma_up = torch.sqrt(
        torch.clamp(sigma_to.square() * (sigma_from.square() - sigma_to.square()) / sigma_from.square(), min=0.0)
    )
    sigma_down = torch.sqrt(torch.clamp(sigma_to.square() - sigma_up.square(), min=0.0))
    derivative = (sample_work - x0_work) / sigma_from
    ancestral_drift = sample_work + derivative * (sigma_down - sigma_from)
    drift_delta = ancestral_drift - euler_work

    if config.noise_mode == "sqrt":
        noise_scale = math.sqrt(float(config.mix))
    elif config.noise_mode == "linear":
        noise_scale = float(config.mix)
    else:
        noise_scale = 0.0
    if noise_scale:
        # Euler-Ancestral draws in the model/output dtype.  Matching that
        # choice keeps the mix=1 comparison meaningful for fp16 SDXL latents.
        noise = _randn_like(euler_prev_sample, generator=generator).to(dtype=work_dtype)
        raw_correction = float(config.mix) * drift_delta + sigma_up * noise_scale * noise
    else:
        raw_correction = float(config.mix) * drift_delta

    euler_update_norm = _batch_norm(euler_work - sample_work)
    raw_norm = _batch_norm(raw_correction)
    raw_ratio = raw_norm / euler_update_norm.clamp_min(torch.finfo(work_dtype).eps)
    applied = raw_correction
    capped = False
    if config.max_correction_ratio is not None:
        limit = float(config.max_correction_ratio) * euler_update_norm
        scale = torch.where(raw_norm > limit, limit / raw_norm.clamp_min(torch.finfo(work_dtype).eps), torch.ones_like(raw_norm))
        view = (slice(None),) + (None,) * (raw_correction.ndim - 1)
        applied = raw_correction * scale[view]
        capped = bool(torch.any(raw_norm > limit))
    applied_ratio = _batch_norm(applied) / euler_update_norm.clamp_min(torch.finfo(work_dtype).eps)
    corrected = (euler_work + applied).to(dtype=euler_prev_sample.dtype)
    diagnostics = TrajectoryCorrectionDiagnostics(
        step_index=step_index,
        mix=float(config.mix),
        sigma_from=float(sigma_from.detach().cpu()),
        sigma_to=float(sigma_to.detach().cpu()),
        sigma_up=float(sigma_up.detach().cpu()),
        raw_correction_norm_ratio=float(raw_ratio.mean().detach().cpu()),
        applied_correction_norm_ratio=float(applied_ratio.mean().detach().cpu()),
        capped=capped,
    )
    return corrected, diagnostics


__all__ = [
    "TrajectoryCorrectionConfig",
    "TrajectoryCorrectionDiagnostics",
    "apply_ancestral_correction",
]
