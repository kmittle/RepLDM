# S7 Novelty Audit: Scheduler and Spectral Claims

**Cutoff:** 2026-08-25. This is a literature/protocol note, not an experiment
result. The underlying arXiv API audit was retrieved on 2026-08-24 and is
recorded in `SCHEDULER_SPECTRAL_NOVELTY_AUDIT_2026.md`. No GPU run is
authorized by this file.

## Motivation

S7 tests a bounded correction around a frozen diffusion sampler. The tempting
claim is "scheduler-consistent ancestral correction" or "spectral trajectory
improvement." That wording is too broad. Recent work already treats the
sampler, stochasticity, trajectory geometry, and latent spectrum as design
objects. The only credible claim is a narrow causal result under the exact
frozen SDXL/U-Net, scheduler, and compute budget used here.

## Nearby Threats

- **CPS (2509.05952), Precise (2605.23522), and LC-GRPO (2608.05600):** finite-
  step ODE/SDE mismatch, coefficient-preserving noise, and ODE Euler plus
  Langevin correction are already covered. Ancestral noise or a stochastic
  interpolation is not a new RL sampler.
- **SlerpFlow (2607.21326) and SGPS (2512.23232):** spherical velocity repair,
  caching, and low-NFE trajectory-gradient correction are already reported.
  A trust cap or one clean-latent step is not sufficient novelty.
- **SPA (2607.22091), SpectralDiT (2606.18765), Frequency-Forcing
  (2604.20902), SEGA (2605.22668), and SPAE (2608.01306):** FFT priors,
  timestep-conditioned low/high residuals, wavelet coarse-to-fine forcing,
  frequency-aware attention, and spectral latent adaptation are covered.
  A spectral band or schedule cannot be the contribution.
- **CFG distortion analysis (2602.00716):** variance shrinkage and negative-
  guidance windows are known controls; changing CFG/energy needs a diversity
  and variance witness.

## S7 Design Constraints

1. Run and report four separate conditions: native
   `EulerAncestralDiscreteScheduler`; deterministic Euler with `sigma_up=0`;
   the registered `mix` interpolation; and an RL rollout sampler (CPS,
   Precise, or LC-GRPO-style). `mix=1,sqrt` is the only registered exact
   ancestral endpoint; intermediate mixes are ablations unless an SDE
   derivation is supplied.
2. Freeze the installed `diffusers` version, scheduler config hash, timestep
   list, `sigma_up/down`, prediction type, CFG, initial-noise seed, and actual
   U-Net call count. Record the current provenance mismatch: environments use
   `diffusers==0.32.1`, while the package declarations request `~=0.21.4`.
   Report noise norms, latent moments, wall time, peak memory, and reward
   variance. Equal nominal steps are not equal compute.
3. A scheduler claim must beat both native ancestral and deterministic Euler on
   held-out prompts, and must not lose to matched-NFE DPM-Solver++ or UniPC.
   Otherwise label the result a sampler choice and do not call it a new
   correction.

## FRLA Constraints

SARA, iREPA, sREPA, SGA, and SSVAE already cover autocorrelation, cosine
self-similarity, Gram structure, and local correlation. FRLA may therefore
claim only the **interaction** between one fixed rank-compatible descriptor and
one scheduler-coordinate clean-latent injection. The development matrix must
include no-op, pointwise and full-Gram descriptors, shuffled tokens,
L2-matched random directions, Euclidean injection, MOG/DiffRGD geometry, SPA
or fixed DCT/wavelet controls, and an equal-NFE extra-U-Net upper bound.

Use a preregistered difference-in-differences interaction, not a point gain:

```text
I = (FRLA - descriptor-only)
    - (scheduler-injection-only - no-op)
```

Charge feature hooks, backward/FFT work, reward calls, wall time, and memory.
Spectral, VIV, or latent-moment improvements without independent
TOPIQ/HPS/CLIP, OCR/count/layout/detail, diversity, and pixel-safety witnesses
do not establish a mechanism.

## Stop Rule

S7 must first clear the registered fixed-action gate: TOPIQ-NR delta `>=
+0.005`, crossed prompt/seed 95% interval above zero, Holm-adjusted prompt
sign-flip significance, HPSv2/CLIP non-inferiority, finite outputs, moment and
trust-cap bounds, and clipping/saturation guards. If it only beats deterministic
Euler, classify it as an ancestral/sampler effect. If FRLA has no positive
interaction or any independent witness fails, report the simplest surviving
baseline and stop S8. Do not add a frequency band, scheduler blend, negative
guidance window, reward weight, distillation stage, or RL controller after
held-out scores are visible.
