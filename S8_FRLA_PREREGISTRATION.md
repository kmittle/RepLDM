# S8/FRLA Preregistration

**Status (2026-08-25): conditional proposal only.** S8 may run only if the
registered S7 fixed-action gate passes on its held-out validation block. This
document adds no actions to the active S7 queue, authorizes no GPU run, and
contains no score or selected result. A failed S7 gate closes S8, including
distillation and RL.

## Hypothesis And Scope

Frozen Relational Latent Alignment (FRLA) tests whether a small, deterministic
relational correction to the predicted-clean SDXL latent improves structure at
matched NFE and compute. The conference connection is the shared principle of
reprogramming a frozen model-native latent trajectory; FRLA is not claimed as a
new gradient method, latent renderer, relational loss, or RL estimator. DUNE,
DIAMOND, DiffRGD, SARA, SGA, LaRender, and SATeCo define relevant controls and
novelty boundaries.

## Fixed Operator

Use a frozen SDXL U-Net and one ordinary conditional denoiser call per
scheduler transition. Capture one detached decoder feature from the registered
`up_blocks.0` block, deterministically reduce it to four channels, and resize
both it and `pred_original_sample` to `16 x 16`. For fixed lags
`(1,0), (0,1), (1,1), (2,0), (0,2)`, compute local cosine autocorrelation
discrepancy and take exactly one latent-only gradient step. The YAML must freeze
`eta`, feature block, reduction/projection seed, scheduler, CFG, resolution,
NFE, and trust-cap ratio before any score is read. Use `torch.enable_grad()` only
on a cloned, detached clean latent; never backpropagate through the U-Net.

Inject the residual after the scheduler update using the scheduler-safe form
`prev_sample + (guided_x0 - pred_original_sample)`. Apply the fixed
moment/covariance retraction and scheduler-update trust cap. Record descriptor
error, residual/update ratios, latent moments, backward/FLOP cost, wall time,
peak memory, and all clipping/saturation guards.

## Six-Basis Controls

Use the registered order in `AttentionGuidance/latent_renderer.py`:

```text
[semantic, spectral-low, spectral-mid, spectral-high,
 FreeU-backbone-minus-skip, Laplacian]
```

The paired block includes no-op, the fixed six-basis action, FRLA, shuffled
feature tokens, detached/no-gradient (and pointwise feature) controls, an
L2-matched random direction, an isotropic shell/covariance control, and an
equal-cost extra-U-Net-call upper bound. Controls use the same prompts, seeds,
initial noise, VAE, CFG, scheduler, and actual U-Net-call budget unless the arm
explicitly tests the extra-call upper bound. No coefficient, lag, or reward
search is permitted on the final split.

## Sampler Provenance

The installed evaluation environments `diff_attn` and `repldm_eval` currently
report `diffusers==0.32.1`; `pyproject.toml` and `requirements.txt` declare
`diffusers~=0.21.4`. This mismatch is recorded, not silently resolved. For
native Euler-Ancestral, DPM-Solver++, and UniPC, record the exact installed
version, scheduler class/config hash, solver order, timestep list, prediction
type, CFG, initial-noise seed, actual U-Net call count, wall time, and peak
memory. Compare at matched actual NFE, not nominal loop steps. DPM/UniPC settings
are frozen before selection and are never tuned against S8 scores.

## Sequential Gates

1. **Fixed-action gate.** FRLA must first beat no-op and the conference expert
   on a prompt-disjoint validation set with TOPIQ-NR delta `>= +0.005`, a
   crossed prompt/seed 95% CI above zero, Holm-adjusted prompt sign-flip
   significance, HPSv2/CLIP non-inferiority, and all moment, trust-cap,
   clipping, and saturation guards. A loss against native solver references
   is reported as a sampler/compute result, not a renderer win.
2. **Search gate.** Only after gate 1, search the predeclared fixed FRLA/basis
   coefficients with common random numbers. Freeze the winner and estimate the
   per-prompt oracle gap on validation data. If the gap is within seed
   uncertainty, use a global or two/three-stage action and stop.
3. **Distillation gate.** Fit a tiny deterministic search-then-distill
   controller on training prompts only, with the same renderer inputs and
   parameter/FLOP budget as RL. Evaluate once on unseen prompts and charge
   teacher search and training cost amortized over deployment prompts.
4. **RL necessity gate.** RL is allowed only if antithetic, shared-prefix RL
   beats the frozen distiller on unseen prompts, independent reward witnesses,
   human/patch/color-normalized metrics, and amortized compute. If RL improves
   only its training reward, loses a robustness witness, or fails to recover
   training cost, reject it as unnecessary or reward hacking and report the
   distiller. Preserve untouched final prompts/seeds for the last test.

All exclusions, hashes, solver manifests, and paired records are committed
before generation. Any non-finite output or missing paired record terminates the
route; no post-hoc reward, action, threshold, or paper framing can reopen it.
