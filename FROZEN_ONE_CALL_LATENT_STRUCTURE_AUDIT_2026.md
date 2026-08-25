# Frozen One-Call Latent Structure Audit (2024-2026)

**Audit cutoff:** 2026-08-25. This is a literature and experiment-design audit,
not an experiment result. Reported gains below were checked against paper full
texts but have not been reproduced in RepLDM. Most 2026 records are preprints
and must be refreshed before submission.

## Scope and Call Accounting

The target is a frozen SDXL U-Net/VAE whose ordinary classifier-free-guidance
(CFG) forward already produces all features used by a small, non-attention
structural renderer. "One call" means one ordinary denoiser invocation per
scheduled step; conditional and unconditional branch-equivalent evaluations
must still be counted even when batched. A feature hook, FFT, or auxiliary MLP
does not add a denoiser call, but its FLOPs, latency, and memory are not free.
Any second denoiser, VAE decode, external encoder/reward evaluation, backbone
backward, or inner optimization is an extra-compute upper bound.

For comparison, use three cost classes:

- **C0:** ordinary CFG denoiser only, with tensor operations or in-forward edits;
- **C1:** ordinary denoiser plus a learned auxiliary module, but no extra
  denoiser/backbone backward; and
- **CX:** extra model evaluations, gradients, decodes, or optimization steps.

## Reviewer Verdict

The component-level novelty space is closed. Small frozen-backbone activation
modules, zero initialization, timestep gates, internal U-Net correction,
frequency-separated CFG, spectral priors, manifold-aware guidance, and
scheduler correction all have direct precedents. The remaining publishable
question is causal and combination-specific:

> Can a zero-initialized, sub-1M module read detached non-attention structural
> features already produced by the ordinary SDXL CFG forward and emit a bounded
> scheduler-coordinate clean-latent residual whose gain depends specifically on
> the interaction between structural description and injection?

The conference-to-journal connection is the shared principle of reprogramming a
frozen model through its native latent trajectory. Attention is one realization,
not a constraint on the journal method. Writing can explain this continuity, but
cannot substitute for a positive causal interaction, transfer, or human evidence.

## Closest Work and Required Consequences

| Work | Verified mechanism and cost class | Consequence for this project |
|---|---|---|
| [SteeringDiffusion, 2605.01653v1](https://arxiv.org/abs/2605.01653v1) | Freezes SD1.5/SDXL U-Nets; a prompt MLP emits a shared code for zero-initialized FiLM/AdaGN modulation with timestep gating. The default SD1.5 `k=16` arm has 0.88M trainable parameters and adds about 3 ms (C1). Validation is style control, not generic quality. | This is the closest learned-module precedent. Match parameters, training steps/data, layer placement, gate, and runtime with a Steering-style FiLM baseline. Do not claim first small module, frozen intervention, zero-init adapter, or one-call learned control. |
| [DUNE, 2607.09753v1](https://arxiv.org/abs/2607.09753v1) | Detects early abrupt changes in SDXL U-Net h-space using a score-normalized EMA and suppresses selected channels inside the same forward; it is training-free and single-pass (C0). | This is the closest non-attention internal correction. Reproduce detect-only, suppress-only, shuffled-mask, density-matched random-mask, and full DUNE arms. A learned feature reader must beat it. |
| [FreSca, 2504.02154v3](https://arxiv.org/abs/2504.02154v3) and [FDG, 2506.19713v1](https://arxiv.org/abs/2506.19713v1) | Decompose the ordinary CFG conditional-unconditional difference into low/high frequency bands and rescale them. Both are training-free; both demonstrate SDXL without another denoiser (C0). | Low/high spectral residuals, structure/detail allocation, and frequency-dependent CFG are not novel. Reproduce both operators or justify one faithful implementation plus a fixed DCT/wavelet factorial. |
| [SPA, 2607.22091v1](https://arxiv.org/abs/2607.22091v1) | Fits a timestep/channel spectral prior offline, then applies an analytic FFT-gradient correction. It uses no extra neural evaluation or backbone backpropagation and reports `+3.86%` SDXL runtime (C0). | Compare a training-split-fitted SPA prior, shuffled prior, and no prior. Report spectral error separately from image quality; charge prior fitting and FFT cost. |
| [Guidance Interval, 2404.07724](https://arxiv.org/abs/2404.07724), [CFG++, 2406.08070](https://arxiv.org/abs/2406.08070), and [APG, 2410.02416](https://arxiv.org/abs/2410.02416) | Restrict guidance to useful noise levels, constrain CFG toward the data manifold, or decompose/rescale/momentum-filter its parallel and orthogonal components. They support SDXL or SDXL-Lightning and add little tensor cost (C0). | A timestep schedule, projection, rescaling, momentum, or manifold language is not sufficient novelty. Tune these controls on the same development budget. |
| [MOG/Auto-MOG, 2603.11509v1](https://arxiv.org/abs/2603.11509v1) | Applies closed-form Riemannian preconditioning and dynamic guidance-energy balancing to existing CFG quantities. SDXL latency is reported as `1.01x-1.08x` (C0). | Match actual CFG outputs, energy, and NFE. Moment retraction, tangent projection, trust caps, and adaptive strength are constraints or baselines, not contributions alone. |
| [Noise Level Correction, 2412.05488v3](https://arxiv.org/abs/2412.05488v3) | Trains a correction network on features from a frozen U-Net encoder and adjusts the sampler noise level. ImageNet uses a 234M corrector versus a 2109M denoiser; Algorithm 1 evaluates it in the sampling loop and the paper reports about 6% overhead (C1). It is not an SDXL result. | A smaller network around a frozen denoiser and learned scheduler-state correction are occupied. Include its lookup-table analogue and a prompt/timestep-only scalar MLP before attributing gains to structural features. |
| [SHIFT, 2604.09213v1](https://arxiv.org/abs/2604.09213v1) and [CAT, 2603.03163v1](https://arxiv.org/abs/2603.03163v1) | SHIFT adds contrastively estimated, state-gated directions to FLUX text/intermediate activations. CAT uses a zero-output-initialized residual MLP plus geometry-aware gating for Z-Image safety steering. Both intervene during the ordinary forward (C0/C1), but target DiT/AR concept control. | Activation vectors, nonlinear transport, state gating, and zero-init residual maps are not firsts. Treat them as architecture/task-adjacent controls, not evidence of SDXL quality improvement. |
| [LaRender, 2508.07647v1](https://arxiv.org/abs/2508.07647v1) | Performs literal training-free latent rendering on SDXL through object masks, an occlusion graph, and cross-attention compositing. | The term "latent renderer" is occupied. LaRender is an attention- and layout-specific boundary, not a generic-quality baseline for prompts without object graphs. |
| [Representation Guidance, 2601.22468v1](https://arxiv.org/abs/2601.22468v1), [DIAMOND, 2602.00883v1](https://arxiv.org/abs/2602.00883v1), [DiffRGD, 2606.28417v2](https://arxiv.org/abs/2606.28417v2), and [PG-MAP, 2606.22958v1](https://arxiv.org/abs/2606.22958v1) | Use a pretrained representation projector/encoder and latent gradients, clean-latent artifact gradients, spherical inner-loop RGD, or MAP optimization (CX). Their original tasks and backbones differ. | These are equal-quality or equal-wall-time upper bounds, never matched one-call baselines. Report every encoder, decode, backward, and inner iteration. |

[ASCED, 2503.16218v1](https://arxiv.org/abs/2503.16218v1) is also a
training-free phase-aware score-anomaly detector with targeted noise correction,
but its original evaluation is not SDXL. [Readout Guidance,
2312.02150](https://arxiv.org/abs/2312.02150) is an important pre-window
precedent for training small readout heads on frozen diffusion features.

## What Structural Papers Do and Do Not Establish

[Improving the Diffusability of Autoencoders,
2502.14831v3](https://arxiv.org/abs/2502.14831v3) identifies excess latent
high-frequency energy and fine-tunes AE scale equivariance to restore a more
diffusion-friendly frequency profile. [EQ-VAE,
2502.09509v3](https://arxiv.org/abs/2502.09509v3) fine-tunes pretrained AEs so
scaling and rotation act more equivariantly in latent space. These results make
spectral and equivariant structure plausible mechanisms, but they change the
representation on which a generator is trained; they do not validate a frozen
SDXL inference residual.

[iREPA, 2512.10794v1](https://arxiv.org/abs/2512.10794v1) finds pairwise patch
self-similarity more predictive of representation-aligned generation than global
linear-probe accuracy. [SGA, 2605.20808v1](https://arxiv.org/abs/2605.20808v1)
aligns spatial Gram structure instead of pointwise features. [Diffusing in the
Right Space, 2606.03578v1](https://arxiv.org/abs/2606.03578v1) compares spatial,
spectral, equivariant, distributional, semantic, and velocity-ambiguity factors
and finds diffusability multi-factorial. These are strong motivations for a
relational descriptor, but correlation or training-time alignment is not evidence
that an inference-time direction causes better images. FreeU remains the native
U-Net structural baseline linking this direction to the conference narrative.

## Claims to Abandon

Do not claim the first latent renderer, frozen-backbone intervention, small
diffusion controller, zero-initialized adapter, one-call guidance, internal-feature
correction, spectral/coarse-to-fine residual, relational descriptor, equivariant
latent method, manifold-preserving update, or scheduler-aware correction. Do not
claim that a better DCT/Gram statistic proves the mechanism. The potential
contribution is only the complete SDXL operator and its verified causal
interaction at a stricter parameter/call budget than the closest controls.

## Admissible Operator and Causal Test

Let `h_t` be a registered non-attention U-Net feature from the ordinary CFG
forward, `S` a pointwise or relational descriptor, and `R_theta` a sub-1M
renderer:

```text
s_t       = S(stop_gradient(h_t))
r_t       = R_theta(s_t, prompt_embedding, t)
x0_t'     = MomentRetract(x0_t + TrustCap(r_t, rho * ||delta_sched_t||))
u_t       = SchedulerMap_t(x0_t' - x0_t)
x_(t-1)'  = x_(t-1)_base + u_t
```

`R_theta` and every output projection start at zero; scale zero must be bitwise
equal to the base pipeline. `SchedulerMap_t` must use the installed scheduler's
actual coefficients rather than assuming that a clean-latent displacement has
unit effect. No gradient may enter the U-Net. Register the feature block,
descriptor, cap, moments, timestep gate, parameter count, and scheduler mapping
before quality scores are read.

The primary mechanism statistic is the preregistered interaction

```text
I = (full - descriptor_only) - (injection_only - no_op).
```

A positive `full - no_op` mean is insufficient when `I <= 0`, when shuffled
features match it, or when a prompt/timestep-only model explains the gain.

## Required Experiment Matrix

Use common `(prompt, seed)` blocks, randomized action order, prompt-disjoint
development/validation/test splits, frozen checkpoints, and identical initial
latents. The minimum matrix is:

1. **Reference:** no-op, conference TFSA, tuned constant CFG, 2-3-stage CFG,
   Guidance Interval, CFG++, APG, FDG/FreSca, MOG/Auto-MOG, DUNE, and SPA.
2. **Fixed structure:** DCT, wavelet, pointwise feature statistics, relational
   lag/Gram descriptors, fixed searched coefficients, and FreeU; each paired
   with shuffled features and L2/norm-matched random directions.
3. **Causal factorial:** no descriptor/no injection, descriptor-only,
   injection-only, and full descriptor plus scheduler injection. Compare naive
   Euclidean, clean-latent, and scheduler-mapped injection.
4. **Learned controls:** prompt/timestep-only scalar MLP, parameter-matched
   pointwise head, SteeringDiffusion-style FiLM/AdaGN, random frozen head, and
   shuffled-descriptor training. Match parameters, examples, optimizer steps,
   augmentation, and early stopping.
5. **Upper bounds:** one extra U-Net call, SPA/backward wall-time match,
   representation-gradient guidance, and a clean-latent artifact-gradient arm,
   all clearly labelled CX.

Run deterministic Euler, native Euler-Ancestral, DPM-Solver++, and UniPC with
actual branch-equivalent NFE matched. Record scheduler/config hashes, timesteps,
random draws, U-Net and auxiliary calls, FFTs/backwards, wall time, peak memory,
action norms/angles, clean-latent and scheduler-update ratios, channel moments,
spectral/relational diagnostics, clipping, saturation, diversity, and off-target
drift. If the method wins only on deterministic Euler, classify it as a sampler
effect. Transfer claims require native 1024 resolution, the repository's
high-resolution path, ControlNet, and a second backbone after the fixed gate.

## Statistical Gate and RL Stop Rule

Before training any renderer or RL policy, a fixed action must achieve all of:

- TOPIQ-NR delta `>= +0.005` against no-op, with a crossed prompt/seed 95%
  interval above zero and Holm-adjusted prompt sign-flip significance;
- HPSv2 and CLIP non-inferiority, plus an independent OCR/count/layout/detail
  or blinded-human structural witness;
- finite outputs, every update within the trust cap, channel-moment drift
  `<= 1%`, and the registered clipping, saturation, diversity, and off-target
  guards; and
- a positive descriptor-by-injection interaction that beats pointwise,
  shuffled, random, DUNE, FDG/FreSca, SPA, MOG, and tuned CFG controls.

Failure closes the learned-renderer route; do not tune another feature block,
frequency band, lag, gate, or reward after held-out scores. If the fixed gate
passes, compare fixed search, prompt-wise oracle, search-then-distill, and the
small supervised renderer before RL. RL is justified only by held-out headroom
over search/distillation after amortized GPU hours and by independent human and
structural evidence. Otherwise the simpler fixed or distilled controller is the
scientific result.
