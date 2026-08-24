# Latent Renderer Protocol

Status: registration-only, independent of the failed S5 Attention Guidance
extension. This document freezes the first mechanism audit; it does not
authorize RL training or reuse S5 images as supervision.

## Hypothesis and Scope

The testable hypothesis is that a small, structure-conditioned network can
allocate a bounded residual in the predicted-clean latent of a frozen SDXL
denoiser. The journal contribution, if any, is the learned constrained
operator and its evidence, not the use of RL, a scalar schedule, or a generic
adapter. A result that only changes colour or saturation is a failure.

The conference-to-journal link is the shared principle of reprogramming a
model-native latent trajectory. The journal method must compete from a fresh
no-op baseline and may not be described as a rescue of S5.

## Fixed Operator

At each ordinary scheduler step, form the candidate bases in this order:

1. reciprocal semantic transport of the scheduler's `pred_original_sample`;
2. smooth low, mid, and high Fourier residuals;
3. a projected backbone-minus-skip (FreeU-style) feature residual; and
4. a local Laplacian residual.

The renderer predicts six bounded coefficients from basis statistics,
normalized timestep, pooled prompt features, and compact denoising-state
features, and the candidate model adds a D4-symmetrized depthwise-separable
spatial head. The coefficient-only model is a required control, not the main
neural-renderer claim. The renderer produces `guided_x0` by projecting the
combined residual to the fixed-mean/fixed-variance tangent space, applying a
scheduler-update trust-region cap, and using the sphere geodesic. The next
latent is always

```text
prev_sample + (guided_x0 - pred_original_sample)
```

so the scheduler remains the source of the diffusion transition. The default
renderer is zero-initialized, uses no extra UNet call, has a `0.05` update-norm
ratio cap, and preserves channel moments. Every run records coefficients,
norm ratios, moment errors, parameters, FLOPs, latency, and peak memory.

## Sequential Gates

**LR-0 mechanism audit.** Synthetic fields must pass partition, graph,
equivariance, moment, gradient, and scheduler-injection tests. On a new,
prompt-disjoint development split, compare no-op, fixed-basis search,
spectral-only, semantic-only, FreeU-only, Laplacian-only, a matched random
renderer, and the frozen conference expert at identical NFE. Freeze the basis
amplitudes before reading held-out scores.

**LR-1 static headroom.** A fixed renderer action must improve TOPIQ-NR by at
least `+0.005`, have a crossed-bootstrap 95% interval excluding zero and a
Holm-adjusted prompt sign-flip `p < 0.05`, remain non-inferior on HPSv2 and
CLIP, and stay within the clipping/saturation guards. The fixed montage must
show a structural, text, counting, positional, or detail improvement.

**LR-2 granularity and adaptivity.** Compare two/three-stage coefficients with
a free per-step vector. Estimate the per-prompt oracle gap on validation data;
if it is within seed variance on two witnesses, use one global action. Search-
then-distill is the required learning baseline.

**LR-3 RL necessity.** RL is allowed only if a distilled renderer leaves
measurable held-out headroom over fixed search, search-then-distill, and an
equal-budget black-box controller. RL must use antithetic grouped rollouts and
shared prefixes; its optimizer is not a novelty claim. Otherwise report the
simpler method and stop.

## Data, Evaluation, and Stop Rules

The frozen prompt manifest is
`eval-pipeline/prompts/latent_renderer_manifest.json`. It records the source
revision/hash, selection key, excluded rows, and exact hashes for these
disjoint files:

```text
train      eval-pipeline/prompts/latent_renderer_train.csv       (48 rows)
validation eval-pipeline/prompts/latent_renderer_validation.csv  (24 rows)
test       eval-pipeline/prompts/latent_renderer_test.csv        (24 rows)
```

Each split has 8/4/4 prompts per challenge across six PartiPrompts challenges.
The manifest and split files must be committed before any image is generated;
the test file is never used for coefficient or architecture selection. All
prompts used by S0--S5 and their derivative outputs are excluded. Seeds
`0,42,123` are reserved for final evaluation and cannot be used to choose
coefficients. TOPIQ-NR is held out from training; report it with HPSv2, CLIP
alignment, ImageReward, patch/detail witnesses, OCR/counting checks, DCT
statistics, LPIPS/diversity, pixel guards, and a blinded human pairwise test
before making a TPAMI claim. Use prompt/seed crossed bootstrap intervals,
prompt sign-flip tests, and within-family Holm correction.

Any non-finite output, violated moment or trust bound, missing paired record,
or failure of LR-1 closes the learned/RL path. No angle, top-k, reward, or
controller sweep may be used to overturn a failed gate.

## Implementation Boundary

`AttentionGuidance/latent_renderer.py` contains only the reusable basis
construction, constrained renderer, diagnostics, and scheduler-safe injection
primitives. It is not wired into default generation, and no checkpoint or RL
training result is claimed by this registration. The frozen YAML companion is
`eval-pipeline/configs/latent_renderer_mechanism_audit.yaml`.

The inference hook is exercised by
`eval-pipeline/latent_renderer_smoke.py` (basis-construction plumbing) and
`eval-pipeline/latent_renderer_structural_smoke.py` (the real SDXL provider).
The structural provider captures `up_blocks.0` backbone/skip tensors and
`up_blocks.0...attn1` Q/K from the ordinary UNet forward, deterministically
reduces features to four latent channels, and emits the six bases in the order
above. A cached-SDXL run at 1024x1024 produced exact no-op hash parity and a
distinct fixed-probe hash, with finite moment/trust diagnostics. This is
wiring evidence only; the probe is not trained, scored, or used to choose any
later action.

The first reproducible fixed-action grid is
`eval-pipeline/configs/latent_renderer_fixed_lr1.yaml`; its coefficients are
registered search candidates, not selected results. The generation harness
accepts these only as `latent_renderer_fixed` actions and records the provider
diagnostics alongside each paired sidecar.

### Fixed-action selection rule

This rule is frozen before any LR-1 score is inspected. On the train split, keep
`no_ag` as a candidate and select the action with the largest paired mean
`HPSv2` delta subject to all of the following: paired `CLIP cosine` delta is not
below `-0.005`, mean clipped-fraction delta is at most `+0.001`, mean
saturation delta is at most `+0.005`, and every record is finite and within the
registered moment/trust bounds. Ties whose HPSv2 95% intervals overlap are
resolved by the smaller mean renderer update ratio, then by the fixed YAML
order. The train split is not used to claim efficacy and `TOPIQ-NR` is not used
for this selection. Freeze the resulting action and run the validation split
once; LR-1 can proceed only if validation satisfies the TOPIQ, non-inferiority,
guard, and qualitative gates above. If no non-`no_ag` action satisfies the
train rule, record `no_ag` and close the learned/RL path without a validation
search. No amplitude, metric, or tie-breaker may be changed after viewing
train scores.
