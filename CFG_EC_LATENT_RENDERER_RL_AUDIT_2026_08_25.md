# CFG-EC, Latent Renderer, and Diffusion-RL Audit

**Cutoff:** 2026-08-25. The records below were checked against the official
arXiv pages and PDFs. They are novelty boundaries and control requirements,
not evidence that another paper's scores transfer to frozen SDXL.

For this note, **CFG-EC** means any frozen-backbone correction to the
conditional-minus-unconditional guidance vector, its latent residual, or the
resulting scheduler update, including scalar, channel/spatial, reward-gradient,
or learned corrections. If the project uses a narrower definition, it must be
registered before experiments rather than after scores are visible.

## Executive Verdict

No individual ingredient in CFG-EC, a latent renderer, or diffusion-RL is a
safe novelty claim. Dynamic CFG already covers adaptive scalar guidance;
training-free latent reward gradients cover intermediate trajectory steering;
Langevin predictor-correctors cover train/inference sampler correction; and
stage-aware rewards, adaptive episode length, path-space estimators, trajectory
distillation, and Jacobian shortcuts cover the obvious RL contributions.

The only defensible claim is a narrowly defined causal combination: a
zero-initialized, low-capacity structural residual that changes a frozen SDXL
trajectory in scheduler coordinates at matched actual NFE and compute. It must
beat simpler CFG, sampler, fixed-operator, and search-then-distill controls. A
point reward gain or a better deterministic-Euler score is not enough.

## Fresh Records and Boundaries

| Work | Mechanism verified from the paper | Boundary for this project and required control |
|---|---|---|
| [PAST, 2608.06794v1](https://arxiv.org/abs/2608.06794v1), 2026-08-07 | Adds a denoising-progress intrinsic reward, dynamic KL/extrinsic-intrinsic balancing, and prompt-adaptive episode termination from noise and attention-based semantic alignment. | Adaptive horizon, intrinsic reward, prompt difficulty, and exploration/convergence balancing are not CFG-EC or RL novelty. Compare complete, fixed, oracle early-stop, and PAST-style horizons at equal quality and amortized wall time; charge the LLM/attention evaluator and report final quality at equal NFE. |
| [SGPO, 2608.06768v1](https://arxiv.org/abs/2608.06768v1), 2026-08-07 | Uses SNR and semantic-change signals to divide denoising into chaotic, structure-stable, and convergence stages, with different per-step objectives instead of copying the terminal reward to every step. | Stage-aware credit assignment and reward backfilling are covered. Include uniform backfill, SNR-only, semantic-only, frozen stage boundaries, and SGPO-style stage objectives; perturb boundaries and match rollouts, reward calls, and NFE. |
| [Latent Reward Registers, 2608.03929v3](https://arxiv.org/abs/2608.03929v3), 2026-08-14 | Appends learnable position-free register tokens as a read-only path to a frozen DiT. Registers predict terminal preference from noisy latents without changing hidden states/velocity; RG-OPD distills dense reward gradients and RGS applies magnitude-matched inference corrections. | This directly rules out "first latent reward," "first frozen latent guidance," and "first reward-gradient renderer." It is DiT/register-specific, so an SDXL transfer must be demonstrated rather than assumed. Compare terminal reward, noisy-latent reward, read-only register, hidden-state intervention, random-token, and RGS-style gradient arms at matched NFE and backward/reward cost. |
| [REST, 2608.09226v1](https://arxiv.org/abs/2608.09226v1), 2026-08-10 | Attaches a decoupled few-step student to an arbitrary RL teacher and distills segment-wise scored trajectories. Advantage-Modulated Distillation gives signed weights to preferred and low-reward rollouts, without extra rollouts or a separate dataset. | RL-plus-distillation, scored-trajectory reuse, and advantage-weighted compression are already covered. The null model must include fixed-action search, vanilla trajectory distillation, RTDMD/REST-style AMD, and sequential RL-then-distill, with teacher/student NFE and amortized training cost reported. |
| [Unified path-space view, 2608.14430v1](https://arxiv.org/abs/2608.14430v1), 2026-08-14 | Derives reverse-trajectory likelihood-ratio and forward reward-matching losses from one regularized path-space objective; gives a value-gradient form, multi-sample KDE estimator, and scale-bounded weights. | Changing from reverse GRPO to forward matching is not an estimator novelty. Report the exact SDE/ODE, likelihood convention, value-gradient/KDE choice, weight clipping, variance, and common-rollout comparison; do not claim a new credit-assignment principle from notation. |
| [LC-GRPO, 2608.05600v1](https://arxiv.org/abs/2608.05600v1), 2026-08-06 | Takes an inference-aligned ODE Euler step, then one Langevin correction targeting the marginal. The score is recovered from flow velocity, yielding an isotropic Gaussian with tractable policy likelihood. | Predictor-corrector, ODE/SDE bridging, and train/inference-gap correction are covered. Compare native Euler-Ancestral, deterministic Euler, Euler-Maruyama, ODE-plus-Langevin with matched noise, and any CFG-EC correction at equal actual NFE, randomness, likelihood accounting, and wall time. |
| [JAGG, 2607.17572v3](https://arxiv.org/abs/2607.17572v3), 2026-07-25 | Interpolates intermediate-step Jacobians from two endpoints and aggregates upstream signals, reducing `W` full DiT backward passes to two per group when velocity is locally linear; cosine routing selects eligible groups. | Backward caching/interpolation and efficient diffusion-RL gradients are not novelty. If used, compare exact per-step autograd, endpoint-only, JAGG interpolation, finite-difference/SPSA, group sizes, and routing thresholds using gradient cosine/norm error, quality, memory, and end-to-end time. |

The papers use SD1.5/SDXL, SD3.5, Flux, Qwen-Image, or DiT/flow settings in
different combinations, but none by itself validates a frozen one-call SDXL
latent renderer. That architectural difference narrows a claim; it does not
restore a component-level first.

## CFG-EC Controls

Freeze the U-Net/VAE, prompt/seed manifest, CFG base scale, timestep list,
initial noise, scheduler config hash, and image resolution. The fixed inference
factorial must include:

1. no-op/native CFG, a predeclared constant scalar grid, a 2-3-stage scalar
   schedule, and a Dynamic-CFG/CFG-interval-style policy;
2. CFG-EC applied to the guidance vector, to `pred_original_sample`, and in
   scheduler-update coordinates, with norm/energy-matched versions;
3. shuffled conditional features, L2-matched random directions, a detached
   feature statistic, and an equal-NFE extra-U-Net or reward-gradient upper
   bound; and
4. native Euler-Ancestral, deterministic Euler, DPM-Solver++, UniPC, and an
   LC-GRPO/CPS/Precise-compatible sampler when the model and likelihood permit.

Record conditional/unconditional branch order, action norm and angle,
`pred_original_sample` drift, scheduler-update ratio, random draws, actual
U-Net/feature/reward/backward calls, latency, and peak memory. If CFG-EC only
beats deterministic Euler, classify it as a sampler effect. If a scalar or
random direction matches it, classify it as guidance amplitude/direction
selection rather than a structural renderer.

## Latent-Renderer and RL Controls

For a renderer, compare no-op, the conference operator, fixed searched basis,
parameter-matched random basis, shuffled structural features, pointwise versus
relational descriptors, read-only reward-register/RGS-style correction, and a
separately trained AE/U-Net adapter as an upper bound. Keep zero initialization,
latent moments, scheduler trust caps, clipping/saturation, equivariance,
high-frequency energy, OCR/count/layout/detail, diversity, and off-target drift
as predeclared diagnostics. A reward proxy cannot be the sole quality witness.

For RL, the required ladder is fixed-action search -> prompt-wise oracle ->
search-then-distill -> REST/RTDMD-style scored-trajectory distillation ->
stage-wise SGPO/PAST controls -> standard GRPO/DDPO -> proposed policy. Use
shared-prefix antithetic rollouts, independent reward witnesses, crossed
prompt/seed confidence intervals, and amortized GPU-hour/latency accounting.
JAGG is an efficiency ablation, not permission to reduce the rollout budget
while claiming an algorithmic gain. The path-space work requires reporting both
reverse and forward estimators whenever the proposed loss changes that axis.

## Stop Rules

Before any learned renderer or RL, the fixed action must meet the registered
TOPIQ-NR delta `>= +0.005`, crossed prompt/seed interval above zero,
Holm-adjusted prompt sign-flip significance, HPSv2/CLIP non-inferiority, and
moment/trust/clipping/saturation/diversity/off-target guards. It must also show
a positive descriptor-by-injection interaction against descriptor-only and
injection-only controls. If that interaction is absent, stop the renderer
route. If RL does not beat the fixed search-then-distill policy on independent
witnesses after amortizing training cost, report the simpler policy and stop.
Do not tune stages, horizons, reward weights, CFG schedules, or sampler mixes
after seeing held-out scores.
