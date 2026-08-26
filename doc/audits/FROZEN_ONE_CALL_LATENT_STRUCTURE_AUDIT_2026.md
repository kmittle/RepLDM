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
| [DUNE, 2607.09753v1](https://arxiv.org/abs/2607.09753v1) | Detects early abrupt changes in SDXL U-Net h-space using a score-normalized EMA and suppresses selected channels inside the same forward; it is training-free and single-pass (C0). | This is the closest non-attention internal correction. Reproduce detect-only, suppress-only, shuffled-mask, density-matched random-mask, and full DUNE arms. Also include a same-hook, compute-matched temporal-anomaly control so a gain cannot be attributed merely to reading abrupt changes at `h`. A learned feature reader must beat it. |
| [SPARE, 2608.01990v1](https://arxiv.org/abs/2608.01990v1), [sREPA, 2605.16949v1](https://arxiv.org/abs/2605.16949v1), and [SGA, 2605.20808v1](https://arxiv.org/abs/2605.20808v1) | Match pairwise token affinities or spatial Gram matrices during diffusion/flow training. SPARE uses true clean VAE latents as a parameter-free target; sREPA uses an external teacher and names local relation matching as future work. None performs frozen-SDXL inference transport. | Pairwise affinity, self-similarity, and Gram structure are occupied components; local-relation matching is explicitly anticipated. A feature-specific claim must beat uniform-local and predicted-clean latent-affinity controls on separately generated trajectories; the latter is not an exact SPARE reproduction. |
| [Bilateral filtering](https://doi.org/10.1109/ICCV.1998.710815) and [guided filtering](https://doi.org/10.1109/TPAMI.2012.213) | Classical edge-aware operators average local values using content- or guide-dependent weights. The proposed `W z - z` is a negative random-walk graph-Laplacian residual of this general form. | Do not claim the local transport operator or feature-guided smoothing itself. Uniform, predicted-clean self-guided, and matched-random graphs are required to attribute a gain to frozen U-Net relations and scheduler placement. |
| [FreSca, 2504.02154v3](https://arxiv.org/abs/2504.02154v3) and [FDG, 2506.19713v1](https://arxiv.org/abs/2506.19713v1) | Decompose the ordinary CFG conditional-unconditional difference into low/high frequency bands and rescale them. Both are training-free; both demonstrate SDXL without another denoiser (C0). | Low/high spectral residuals, structure/detail allocation, and frequency-dependent CFG are not novel. Reproduce both operators or justify one faithful implementation plus a fixed DCT/wavelet factorial. |
| [SPA, 2607.22091v1](https://arxiv.org/abs/2607.22091v1) | Fits a timestep/channel spectral prior offline, then applies an analytic FFT-gradient correction. It uses no extra neural evaluation or backbone backpropagation and reports `+3.86%` SDXL runtime (C0). | Compare a training-split-fitted SPA prior, shuffled prior, and no prior. Report spectral error separately from image quality; charge prior fitting and FFT cost. |
| [Guidance Interval, 2404.07724](https://arxiv.org/abs/2404.07724), [CFG++, 2406.08070](https://arxiv.org/abs/2406.08070), and [APG, 2410.02416](https://arxiv.org/abs/2410.02416) | Restrict guidance to useful noise levels, constrain CFG toward the data manifold, or decompose/rescale/momentum-filter its parallel and orthogonal components. They support SDXL or SDXL-Lightning and add little tensor cost (C0). | A timestep schedule, projection, rescaling, momentum, or manifold language is not sufficient novelty. Tune these controls on the same development budget. |
| [MOG/Auto-MOG, 2603.11509v1](https://arxiv.org/abs/2603.11509v1) | Applies closed-form Riemannian preconditioning and dynamic guidance-energy balancing to existing CFG quantities. SDXL latency is reported as `1.01x-1.08x` (C0). | Match actual CFG outputs, energy, and NFE. Moment retraction, tangent projection, trust caps, and adaptive strength are constraints or baselines, not contributions alone. |
| [Noise Level Correction, 2412.05488v3](https://arxiv.org/abs/2412.05488v3) | Trains a correction network on features from a frozen U-Net encoder and adjusts the sampler noise level. ImageNet uses a 234M corrector versus a 2109M denoiser; Algorithm 1 evaluates it in the sampling loop and the paper reports about 6% overhead (C1). It is not an SDXL result. | A smaller network around a frozen denoiser and learned scheduler-state correction are occupied. Include its lookup-table analogue and a prompt/timestep-only scalar MLP before attributing gains to structural features. |
| [Diffusion Controller, 2603.06981v1](https://arxiv.org/abs/2603.06981v1) | Freezes SD1.4 and adds a 12M latent U-Net side network to the pretrained score at every step; it trains with SFT, HPSv2 reward-weighted loss, or PPO (C1). Evaluation sweeps the side-network strength and reports the best HPSv2 result. | This directly occupies frozen-backbone latent side correction and reward/RL training. Include DiffCon-style structured and naive free-residual upper bounds, freeze inference strength before test, and use independent metrics plus human evaluation. |
| [Hierarchical Variational Policies, 2605.21661v1](https://arxiv.org/abs/2605.21661v1) | Keeps the denoiser fixed and learns an initial-noise policy plus Gaussian per-step additive latent controls. Its inverse-problem controllers are 51M/92M DiTs (C1). | Per-step latent control and deterministic/stochastic policies are occupied. Any claim here must rest on sub-1M structure constraints, exact scheduler coordinates, open-domain SDXL evidence, and equal reward-query/compute controls. |
| [SHIFT, 2604.09213v1](https://arxiv.org/abs/2604.09213v1) and [CAT, 2603.03163v1](https://arxiv.org/abs/2603.03163v1) | SHIFT adds contrastively estimated, state-gated directions to FLUX text/intermediate activations. CAT uses a zero-output-initialized residual MLP plus geometry-aware gating for Z-Image safety steering. Both intervene during the ordinary forward (C0/C1), but target DiT/AR concept control. | Activation vectors, nonlinear transport, state gating, and zero-init residual maps are not firsts. Treat them as architecture/task-adjacent controls, not evidence of SDXL quality improvement. |
| [LaRender, 2508.07647v1](https://arxiv.org/abs/2508.07647v1) | Performs literal training-free latent rendering on SDXL through object masks, an occlusion graph, and cross-attention compositing. | The term "latent renderer" is occupied. LaRender is an attention- and layout-specific boundary, not a generic-quality baseline for prompts without object graphs. |
| [Representation Guidance, 2601.22468v1](https://arxiv.org/abs/2601.22468v1), [DIAMOND, 2602.00883v1](https://arxiv.org/abs/2602.00883v1), [DiffRGD, 2606.28417v2](https://arxiv.org/abs/2606.28417v2), and [PG-MAP, 2606.22958v1](https://arxiv.org/abs/2606.22958v1) | Use a pretrained representation projector/encoder and latent gradients, clean-latent artifact gradients, spherical inner-loop RGD, or MAP optimization (CX). Their original tasks and backbones differ. | These are equal-quality or equal-wall-time upper bounds, never matched one-call baselines. Report every encoder, decode, backward, and inner iteration. |

[ASCED, 2503.16218v1](https://arxiv.org/abs/2503.16218v1) is also a
training-free phase-aware score-anomaly detector with targeted noise correction,
but its original evaluation is not SDXL. [Readout Guidance,
2312.02150](https://arxiv.org/abs/2312.02150) is an important pre-window
precedent for training small readout heads on frozen diffusion features.

## Reproduction Readiness and Canonicality

Repository availability is not sufficient evidence of a paper-faithful
baseline. The following source audit was frozen on 2026-08-25 before any new
quality result was observed:

| Baseline | Pinned implementation evidence | Current decision |
|---|---|---|
| Tuned CFG | Native RepLDM operation; no artifact or dependency. Use the same frozen backbone, scheduler, NFE, and prompt/seed blocks. | **GO.** Freeze a development-only scale grid and selection rule; evaluate the selected scale only on later prompt- and seed-disjoint data. |
| SPA | Author MIT repository [`26f3b3b9`](https://github.com/SonyResearch/SPA/tree/26f3b3b9aad9d49242e730a4ee85d96c51d41215) includes SDXL code and a LAION-fitted prior. The paper uses DDIM 30 steps, `(eta,a)=(0.2,5)`, and a pre-sampler correction; its script uses 50 steps, `(0.2,3)`, and subtracts the gradient from the completed `x_(t-1)`. | **Qualified GO only as pinned official-code behavior.** Run an independent paired DDIM comparison and name the paper/script/placement differences. Do not call it an exact paper reproduction or transplant it silently to Euler. |
| FreSca | Author MIT repository [`02e1939d`](https://github.com/WikiChao/FreSca/tree/02e1939d8d697e66fa7ae3d9e7461aefef0bbd86) provides only a generic CFG operator. Its center-row/column cutoff and default `0.2` differ from the paper's radial cumulative-energy `r0=0.9`; there is no runnable SDXL path. | **Operator-only.** A local port must be labelled as such and cannot stand in for the reported SDXL result. |
| FDG | No author repository. Diffusers v0.35.0 has an Apache-2.0 Laplacian-pyramid guider, but it operates on raw model predictions in its modular SDXL path; the paper requires conversion to `x0` and reports multiple incompatible SDXL scale triplets. | **No-go as a paper result.** First implement and unit-test the complete prediction-type round trip; a diffusers port is only a prediction-space surrogate. |
| DUNE | No author repository, commit, or license. The paper fixes `p=0.9`, EMA `gamma=0.7`, SDXL `kappa=0.3`, a three-step warm-up, and an early 40% phase, but does not identify an exact hook or executable U-Net operator. | **No-go.** A paper-derived approximation is not a faithful baseline. |
| MOG/Auto-MOG | Repository [`a2a3a649`](https://github.com/zexiJia/MoG/tree/a2a3a649de22971c8f5fe0c821ca8bde0d9c0f88) contains Apache-2.0 code, but is not linked by arXiv v1. It uses `lambda_perp=5`, clamp `[1,20]`, and no feature hook; the paper uses anisotropy `10`, clamp `[0,50]`, and a combined score/feature metric. | **No-go for paper Auto-MOG.** Repository behavior may be reported only as a separately named surrogate. |
| Guidance Interval / APG | Guidance Interval's Apache-2.0 author code does not implement its reported SDXL path. APG has no author repository; the diffusers guider acts on noise predictions with defaults that differ from the paper's SDXL `x0` configuration. | **No-go as direct SDXL reproductions.** Transparent local ports require frozen equations, prediction coordinates, and unit parity before GPU use. |

The installed generation environment uses Python `3.11.10`, torch `2.5.1`,
diffusers `0.32.1`, transformers `4.47.1`, and has no `kornia`; the package files
still declare the older diffusers `~0.21.4` stack. Until an exact evaluation lock
is committed, sidecar runtime versions are authoritative and this mismatch is a
reproduction blocker. Upgrading to the v0.35 guider stack would also change the
transformers contract. Therefore baseline code must be vendored or independently
implemented with source/license attribution and tensor-level parity tests, not
introduced by an environment-wide upgrade. Missing faithful baselines block a
learned-method claim; they do not authorize weak approximations merely to keep a
GPU occupied.

## What Structural Papers Do and Do Not Establish

[Improving the Diffusability of Autoencoders,
2502.14831v3](https://arxiv.org/abs/2502.14831v3) identifies excess latent
high-frequency energy, but its main intervention is a downsampled
RGB/latent decoder reconstruction loss that encourages scale equivariance, not
a direct DCT penalty. [EQ-VAE,
2502.09509v3](https://arxiv.org/abs/2502.09509v3) fine-tunes pretrained AEs so
scaling and rotation act equivariantly rather than invariantly in latent space.
Both works retrain the downstream generator on the changed representation.
These results make spectral and equivariant structure plausible mechanisms;
they do not validate a frozen SDXL inference residual or a test-time low-pass.

[iREPA, 2512.10794v1](https://arxiv.org/abs/2512.10794v1) finds pairwise patch
self-similarity more predictive of representation-aligned generation than global
linear-probe accuracy. [SGA, 2605.20808v1](https://arxiv.org/abs/2605.20808v1)
aligns spatial Gram structure instead of pointwise features. [SPARE,
2608.01990v1](https://arxiv.org/abs/2608.01990v1) directly aligns diffusion-token
affinities to true clean-latent affinities during training, while sREPA explicitly
proposes local relation matching as a scaling direction. [Diffusing in the Right
Space, 2606.03578v1](https://arxiv.org/abs/2606.03578v1) compares spatial,
spectral, equivariant, distributional, semantic, and velocity-ambiguity factors
and finds diffusability multi-factorial. These are strong motivations for a
relational descriptor, but correlation or training-time alignment is not evidence
that an inference-time direction causes better images. FreeU remains the native
U-Net structural baseline linking this direction to the conference narrative.

## Claims to Abandon

Do not claim the first latent renderer, frozen-backbone intervention, small
diffusion controller, zero-initialized adapter, one-call guidance, internal-feature
correction, spectral/coarse-to-fine residual, relational descriptor, local graph
filter, equivariant latent method, manifold-preserving update, or scheduler-aware correction. Do not
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
   Guidance Interval, CFG++, APG, FDG/FreSca, MOG/Auto-MOG, DUNE, a
   same-hook compute-matched temporal-anomaly control, and SPA.
2. **Fixed structure:** DCT, wavelet, pointwise feature statistics, relational
   lag/Gram descriptors, fixed searched coefficients, and FreeU; pair feature-
   affinity transport with uniform-local, predicted-clean-affinity, shuffled-
   feature, and L2/norm-matched random controls on real trajectories.
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
