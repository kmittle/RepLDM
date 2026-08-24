# Scheduler-Consistent and Latent-Spectral Novelty Audit

**Cutoff:** 2026-08-25. This is a read-only literature audit; it adds no
experiment, score, or GPU authorization. The arXiv API query
`https://export.arxiv.org/api/query?id_list=2606.18765,2608.01306,2604.20902,2607.22091,2605.22668,2607.21326,2605.23522,2507.21802,2509.05952,2608.05600,2602.00716,2512.23232&max_results=30`
returned the records below (feed updated `2026-08-24T23:55:04Z`).

## Bottom Line

Three novelty shortcuts are closed.

1. **Euler-Ancestral matching is a reference condition, not a method.** CPS,
   Precise, and LC-GRPO show that finite-step ODE/SDE noise, coefficient
   preservation, and train/test sampler mismatch change both quality and RL
   rewards. SlerpFlow covers spherical trajectory correction and cached
   velocity. A gain over deterministic Euler, or an interpolation called
   ``ancestral'', cannot be called a new scheduler correction.
2. **Spectral residuals and spectral schedules are already crowded.** SPA,
   SpectralDiT, Frequency-Forcing, SEGA, SPAE, and SSVAE cover FFT priors,
   timestep-conditioned low/high residuals, soft coarse-to-fine forcing,
   frequency-aware attention, spectral autoencoders, and local-correlation
   regularization. FRLA cannot claim a first spectral or coarse-to-fine
   mechanism.
3. **The S8 space is only a causal comparison.** A fixed relational descriptor
   plus scheduler-coordinate clean-latent injection remains a possible
   combination, but only if its interaction survives native Euler-Ancestral,
   DPM/UniPC, spectral, geometric, random, and extra-call controls at matched
   actual NFE and wall time. Otherwise report the simpler sampler or spectral
   control and stop before RL.

## Novelty Threat Matrix

| Work (arXiv record) | API-verified mechanism | Threat and required control |
|---|---|---|
| [CPS, 2509.05952v4](https://arxiv.org/abs/2509.05952v4), 2025-09-07 (revised 2025-12-08) | Coefficients-Preserving Sampling reformulates flow SDE sampling to remove excess noise artifacts and preserve nominal signal/noise coefficients for RL. | Coefficient-preserving stochastic transitions are not new. Compare native Euler-Ancestral, deterministic Euler, CPS, and any S8 stochastic arm with identical noise seeds, coefficients, timesteps, and actual U-Net calls. |
| [Precise, 2605.23522v1](https://arxiv.org/abs/2605.23522v1), 2026-05-22 | Treats the sampler as part of the RL policy; balances exploration and stability and freezes a clean-latent posterior mean for SDE-consistent discretization. | No first SDE-consistent sampler, clean-mean freeze, or exploration schedule. Report solver class/config hash, stochastic schedule, log-prob convention, reward variance, and wall time before attributing a gain to FRLA/RL. |
| [LC-GRPO, 2608.05600v1](https://arxiv.org/abs/2608.05600v1), 2026-08-06 | Each rollout takes an inference-aligned ODE Euler step, then one Langevin correction targeting the marginal; the score is recovered from flow velocity. | Predictor-corrector ODE/Langevin alignment is covered. Include an ODE-Euler plus Langevin control and distinguish it from native Euler-Ancestral; a stochastic correction is not automatically an SDE derivation. |
| [SlerpFlow, 2607.21326v1](https://arxiv.org/abs/2607.21326v1), 2026-07-23 | Uses spherical linear interpolation to correct rectified-flow velocity directions and caches corrected velocity for later steps at first-order cost. | Spherical trajectory correction, curvature-aware velocity repair, and cache amortization are already claimed. Compare spherical/geodesic, norm-only, and uncached controls; do not call scheduler-coordinate geometry new. |
| [Distortion analysis, 2602.00716v4](https://arxiv.org/abs/2602.00716v4), 2026-01-31 (revised 2026-05-08) | Derives CFG-induced variance shrinkage and proposes a negative-guidance window that improves separability and diversity. | A timestep guidance schedule or variance-preserving window is not novel. Add a fixed negative-guidance-window and diversity/variance control when CFG or energy is changed. |
| [SPA, 2607.22091v1](https://arxiv.org/abs/2607.22091v1), 2026-07-24 | Fits a timestep/model spectral prior offline and uses FFT-gradient guidance on intermediate predictions, including SDXL and flow models. | Spectral prior, FFT guidance, and exposure-bias correction are direct baselines. Freeze the prior on training data and report FFT/backward cost, radial spectrum gap, and image quality separately. |
| [SpectralDiT, 2606.18765v1](https://arxiv.org/abs/2606.18765v1), 2026-06-17 | Adds a timestep-conditioned low/high spectral residual to a DiT MLP branch with a zero-initialized additive gate; reports 0.6% FLOP and 1.36% parameter overhead. | Zero-init spectral residuals and timestep-conditioned low/high gates are not new. Use a parameter/FLOP-matched zero-init spectral control; separate its trained-backbone result from a frozen one-call S8 result. |
| [Frequency-Forcing, 2604.20902v1](https://arxiv.org/abs/2604.20902v1), 2026-04-21 | Uses a lightweight learnable wavelet-packet low-frequency scratchpad as an asynchronous soft forcing stream while preserving the base flow path. | Coarse-to-fine forcing, wavelet guidance, and path-preserving frequency ordering are covered. Compare fixed DCT, fixed wavelet, and learned-wavelet upper bounds and charge the auxiliary stream. |
| [SEGA, 2605.22668v1](https://arxiv.org/abs/2605.22668v1), 2026-05-21 | Dynamically scales RoPE attention components according to latent spatial-frequency energy for training-free resolution extrapolation. | Frequency-conditioned attention scaling is not new. At high resolution include SEGA-style attention scaling or state that the DiT/RoPE arm is architecture-inapplicable; do not compare it as a free quality gain. |
| [SPAE, 2608.01306v1](https://arxiv.org/abs/2608.01306v1), 2026-08-02 | Uses a compact bottleneck and channel-wise masking to suppress high-frequency components and decouple semantic/detail information in pretrained visual latents. | Spectral latent adaptation, high-frequency suppression, and channel decoupling are already covered. Keep the SDXL VAE frozen for S8; an SPAE-like adapter is a separately trained representation upper bound with reconstruction and moment checks. |
| [SGPS, 2512.23232v1](https://arxiv.org/abs/2512.23232v1), 2025-12-29 | Uses SURE-gradient trajectory corrections and PCA noise estimation for low-NFE posterior sampling in inverse problems. | Risk-aware trajectory correction and low-NFE gradient updates are not firsts. Use a dummy-gradient/SURE-style correction only as a task-matched control; do not transfer inverse-problem scores to text-to-image. |

## Euler-Ancestral Matching Protocol

The S7/S8 manifest must distinguish four conditions rather than collapsing them
into one ``ancestral'' label:

1. **Native endpoint:** `EulerAncestralDiscreteScheduler` with the exact
   installed `diffusers` version, scheduler config hash, timestep list,
   `sigma_up`/`sigma_down`, prediction type, CFG, and one seeded noise draw per
   transition.
2. **Deterministic endpoint:** the same Euler drift with `sigma_up=0` and no
   correction. This is the no-stochasticity baseline, not a matched ancestral
   sampler.
3. **Registered interpolation:** any `mix` or drift/noise blend, with its
   analytic formula, random draw, and whether it preserves the scheduler's
   coefficients. Intermediate mixes are ablations unless a valid SDE derivation
   is recorded; `mix=1,sqrt` is the only candidate for an exact ancestral
   endpoint in the current protocol.
4. **RL rollout sampler:** CPS, Precise, or LC-GRPO-style ODE-plus-noise
   transition, with the policy likelihood/log-prob convention and reward
   variance recorded. It is a training sampler control, not an inference claim.

All four use the same `(prompt, seed)`, initial latent, CFG, resolution, and
actual U-Net evaluation count. Report `diffusers==0.32.1` in the installed
`diff_attn`/`repldm_eval` environments and the declared `~=0.21.4` dependency
as a provenance mismatch. Record latency, peak memory, noise norms, final
latent moments, diversity, and independent quality/preference witnesses. A
candidate must beat native Euler-Ancestral and deterministic Euler on held-out
prompts before it can be described as a correction rather than a sampler
choice; losing to DPM-Solver++ or UniPC closes the scheduler claim.

## Latent-Spectral Control Protocol

Spectral evidence must separate representation changes, inference guidance,
and backbone training:

- **Frozen-VAE inference:** no-op, fixed radial DCT/FFT projection, SPA prior
  guidance, fixed wavelet projection, and an L2-matched random projection. No
  VAE or U-Net weights change; charge FFT/backward cost and trust-cap updates.
- **Small learned upper bounds:** SpectralDiT-like zero-init low/high residual
  and SPAE/SSVAE-like VAE adapters are trained only on the training split,
  with parameter/FLOP-matched random adapters. They cannot authorize a frozen
  one-call claim and must report reconstruction/LPIPS, latent moments/scale,
  channel covariance, and high-frequency excess.
- **High-resolution attention control:** include SEGA/FreeU where the backbone
  supports it; otherwise mark it architecture-inapplicable rather than silently
  dropping the threat.

For every arm report radial DCT slope and spectrum gap by timestep, high-
frequency energy, coarse-to-fine ordering, VIV/velocity ambiguity as an
exploratory diagnostic, native-resolution quality, OCR/count/layout/detail
metrics, clipping/saturation/contrast guards, diversity, latency, and peak
memory. A spectral proxy improvement without an independent image or human
gain is a failed mechanism result.

## S8 Decision Gate

The fixed FRLA action remains conditional on S7. Before any coefficient or
controller search, it must clear the existing `+0.005` TOPIQ-NR threshold,
crossed prompt/seed confidence interval, Holm sign-flip test, HPSv2/CLIP
non-inferiority, moment/trust/clipping/saturation guards, and the positive
descriptor-by-injection interaction defined in
`FRLA_S8_COMBINATION_AUDIT_2026.md`. The comparison must include native
Euler-Ancestral, DPM-Solver++, UniPC, SPA, a fixed DCT/wavelet arm,
MOG/DiffRGD geometry, shuffled/random directions, and an equal-NFE extra-U-Net
upper bound.

If FRLA only beats deterministic Euler, classify the result as an ancestral
sampler effect. If SPA/SpectralDiT/Frequency-Forcing controls match it, classify
the result as spectral correction. If the interaction is absent or any
independent witness fails, stop S8 and do not begin distillation or RL. No
additional scheduler blend, frequency band, negative-guidance window, or
reward weight may be introduced after held-out scores are visible.
