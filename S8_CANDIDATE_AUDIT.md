# Candidate Audit: Frozen SDXL Journal Extension

**Purpose.** This is an independent design memo, not an experiment result or a
request to start a GPU job. It narrows the journal extension to hypotheses that
can be falsified after the registered S7 sampler gate. The conference method's
soft connection is the use of a frozen model's internal denoising structure to
propose a bounded latent change. That connection does not make the following
components novel by themselves.

## Fixed Evaluation Contract

Every candidate uses the same SDXL checkpoint, VAE, prompt-disjoint splits,
resolution, CFG, timestep list, precision, initial seeds, and NFE as no-op.
Native Euler, native Euler-Ancestral, and a deterministic DDIM or DPM-Solver++
reference are retained. Report U-Net calls, latent backward calls, FLOPs,
wall time, throughput, peak VRAM, and no-op output hashes. CFG++, APG, TAG,
SPA, DUNE, FreeU, and the conference TFSA action are mandatory where their
interfaces are defined. Batch-2 CFG pairing is a prerequisite: negative rows,
positive rows, and latent replication must correspond exactly.

The primary witness remains TOPIQ-NR. A candidate needs a pre-registered gain
of at least +0.005, a crossed-bootstrap 95% interval whose lower bound is above
zero, and a prompt-level paired sign-flip test with Holm-adjusted p < 0.05.
HPSv2 and CLIP must be non-inferior. Clipping may increase by at most 0.001
and saturation by at most 0.005. OCR, counting, spatial relations,
GenEval/T2I-CompBench, LPIPS or Vendi diversity, contrast, and sharpness must
not regress. A gain against no-op alone is not evidence: it must survive a
matched-cost or matched-norm control.

## Candidate A: CFG-Consensus Common-Mode Corrector (C3R)

### Hypothesis

One CFG forward produces unconditional and conditional feature states, `h_u`
and `h_c`. Their common component is `h_m=(h_u+h_c)/2`; their differential
component is `h_d=h_c-h_u`. A local disagreement score

```text
u_i = norm(P h_d[i]) / (norm(P h_m[i]) + eps)
```

may identify structural uncertainty. Apply a detached, latent-only correction
to the common-mode structural component, while projecting the latent residual
away from the CFG semantic differential and capping its norm relative to the
scheduler update. The fixed operator should be a one-step `pred_original_sample`
correction; no U-Net backward is allowed in the first gate.

The average is not automatically meaningful: cross-attention can put the two
branch features on different scales or affine coordinates. Any branch
normalization or whitening must be fixed on the training split and reported
explicitly, with raw-average, normalized-average, and branch-norm controls.
Likewise, the latent CFG differential is not guaranteed to be a semantic
tangent. Treat projection into its orthogonal complement as a hypothesis, not a
semantic-preservation theorem.

### Boundary and overlap

CFG-OEC (arXiv:2511.14075) already orthogonalizes conditional/unconditional
prediction errors using adjacent-step proxies. CFG++, APG, TAG, CADE, and the
2026 CFG benchmark (arXiv:2608.16786) already cover prediction projections and
attention perturbations. Readout Guidance (2312.02150), DUNE (2607.09753), and
SSG (2607.29122) cover frozen internal features and lightweight self-guidance.
Therefore the allowed claim is only a *fixed CFG-branch common-mode structural
test with semantic-differential protection*. Do not claim first relational,
branch-aware, latent, or training-free guidance.

### Minimum smoke

Implement detached feature capture and a latent descriptor on two prompts and
two seeds before any search:

* no-op and conference TFSA;
* C3R with disagreement gate and semantic projection;
* no gate, no semantic projection, and differential-only corrections;
* shuffled conditional/unconditional rows, shuffled features, random mask, and
  equal-norm dummy latent gradient;
* CFG-OEC, CFG++, APG, and native Euler-Ancestral references.

The zero coefficient must be byte-identical. Log the gate distribution,
cosine between correction and CFG differential, scheduler-update ratio, latent
moments, and branch row IDs. A branch permutation that changes the output is a
necessary sanity check; a permutation that improves it is a likely bug or
shortcut.

Before scoring image quality, calibrate `u_i` against an independent witness on
training prompts: for example, the discrepancy between the ordinary step and
an extra-NFE reference step, a held-out artifact detector, or a known OCR,
counting, or spatial-relation failure. The gate must predict that witness
better than a random mask and a scalar timestep-only gate (pre-register an
AUROC or rank-correlation threshold). If disagreement only tracks token
saliency or CFG scale and does not predict structural error, close C3R even if
one metric improves.

### Gate

Run the fixed operator on the development split only. It must clear the common
quality contract above at matched NFE and beat a matched-norm dummy correction.
If it fails, close C3R and all renderer/RL variants that use its target. If it
passes, compare a fixed searched coefficient, a schedule-only coefficient, a
matched random feature adapter, and the best CFG-OEC result before learning
anything. A renderer is useful only if it improves over the fixed searched
operator on held-out prompts.

## Candidate B: Spectral-Relational Bounded Renderer (SRBR)

### Hypothesis

SPA (arXiv:2607.22091) corrects a sampler-specific power-spectrum mismatch;
FRLA-style local relational descriptors test spatial organization. A narrow
combined hypothesis is that the spectrum correction should be active only when
detached U-Net relations indicate an inconsistent structural state. The first
implementation is not a free four-channel network. It emits bounded
coefficients over fixed bases:

```text
low/mid/high DCT residuals
local Laplacian or edge residual
FreeU-style backbone-minus-skip residual
five local cosine lags from a rank-compatible decoder feature
```

The residual is applied to the scheduler's clean-latent prediction with a
per-step trust cap and a fixed-moment retraction. A zero-initialized basis and
coefficient-only version are required controls. This tests whether structure
and spectrum are complementary, not whether a larger neural renderer can fit a
reward.

When both gradients are combined, normalize each component before the trust
cap and report their cosine. Additive, equal-norm, and gradient-orthogonal
combinations are separate actions; otherwise a gain can be explained entirely
by doubling the update magnitude.

### Boundary and overlap

SPA already provides FFT guidance with a 3-4% overhead and an official SDXL
implementation. SGA (2605.20808), SARA (2503.08253), sREPA (2605.16949), and
iREPA (2512.10794) cover Gram, autocorrelation, and relational alignment.
DUNE covers frozen-SDXL internal-latent suppression; DiffRGD (2606.28417)
covers manifold-preserving latent updates. The allowed claim is a conditional,
rank-compatible combination under a scheduler trust cap. Never claim first
spectral guidance, first relational alignment, or first frozen latent
renderer.

SPA's released prior is tied to model, resolution, and scheduler. Its DDIM
SDXL spline cannot be reused for Euler or Euler-Ancestral. An exact DDIM arm
may use the released prior; an Euler arm must collect clean-latent spectra on
training images with the actual scheduler and fit a new prior. This collection
cost is part of the method, not an invisible preprocessing step.

### Minimum smoke

Use identical prompts, seeds, and scheduler states for:

* no-op, SPA-only, relation-only, and SRBR (additive and orthogonal variants);
* pointwise feature match, full Gram, local-lag subset, shuffled feature, and
  random-feature controls;
* radial DCT projection, low-pass projection, norm-rescale, and matched FFT
  backward-cost dummy gradient;
* no trust cap, scheduler-update cap, and equal-norm isotropic shell; and
* DUNE, FreeU, CFG-OEC, and native sampler references.

Measure spectrum error, relation error, gradient/update cosine, DCT/SEC/LNC,
latent mean/variance/covariance/KL, clipping/saturation, and actual FFT and
backward cost. A full Gram loss is an ablation only: four latent channels and
wider U-Net features are rank-incompatible.

The relation target must be detached and independently audited. Compare its
error against a shuffled-feature target, a cross-timestep target, and a
pointwise feature target. A relation gain that disappears under feature
shuffling, or a relation error that is not associated with the independent
structural witness, is self-referential rather than causal.

### Gate

The fixed SRBR action must beat both SPA-only and relation-only on a new
prompt-disjoint split, not just no-op, while satisfying the common quality and
cost contract. If only the spectral term helps, report SPA and close SRBR. If
only a matched dummy gradient helps, close the structural claim. If the fixed
operator passes, a 0.1M or 1M renderer may predict coefficients, but it must
beat fixed searched coefficients and a parameter-matched random convolution.
No RL is authorized by a positive fixed SRBR result alone.

## Candidate C: Search-Then-Distill, Then Constrained RL (SD-RL)

### Hypothesis

The learning question is whether a tiny state-conditioned renderer can select
safe residual coefficients across denoising stages, rather than whether RL is a
new optimizer. The state includes timestep, scheduler update, CFG disagreement,
feature energy, DCT bands, prior action, and prompt embedding. The action is a
bounded coefficient vector over the fixed basis from Candidate A or B.

### Boundary and overlap

LeSAMP (2607.23488), AdaGen (2603.06993), and related work already learns
prompt/timestep sampling schedules. CRD (2603.14128), DRM (2605.25661),
Latent Reward Registers (2608.03929), BranchGRPO/SGPO, and recent path-space
distillation cover dense step rewards, KL anchoring, and on-policy training.
LP-DS (2606.01151) covers compact noise-space perturbations with a Lagrangian
trust region. AGD (2503.07274) and SSG (2607.29122) cover lightweight frozen
backbone adapters. Thus no contribution may be claimed for “RL on a frozen
diffusion model,” “stepwise reward,” “latent renderer,” or “trust-region RL.”

### Required staging and fair baselines

1. **Teacher search:** on training prompts only, search bounded basis
   coefficients with common random numbers and antithetic pairs; cache feature
   tensors and scheduler states.
2. **Distillation:** fit a zero-initialized 0.1M/1M renderer to teacher actions
   with action, moment, equivariance, spectral, and trust-cap losses.
3. **Constrained RL:** only if distillation beats the fixed searched action,
   use shared-prefix grouped rollouts, antithetic actions, group-relative
   advantages, and a KL penalty to the distilled policy. Keep rollout count,
   U-Net calls, and wall time equal to baselines.
4. **Deployment distillation:** distill any stochastic policy to a deterministic
   mean renderer and re-evaluate on unseen prompts.

Required baselines are fixed searched coefficients, a schedule-only policy, a
context-free coefficient head, a parameter-matched random adapter, offline
distillation without RL, RL from scratch, reward backpropagation, and the
closest published methods (CRD/DRM/LRR/LeSAMP/SSG or AGD where compatible).
TOPIQ-NR or any reward used for selection must be held out from training; report
all raw reward components, independent evaluator scores, and human pairwise
preference.

### Gate

The renderer must first beat fixed search and offline distillation under the
same compute budget. RL must then produce an additional held-out gain of at
least +0.005 over distillation with a positive crossed-bootstrap interval,
non-inferior HPS/CLIP, no guard violation, and no more than the registered
rollout/backward budget. If the action is effectively one global scalar or one
prompt-level choice, label it contextual-bandit/search rather than sequential
RL. A failed fixed-action or distillation gate permanently closes this route;
increasing reward weight, capacity, or rollout count is not a valid rescue.

## Candidate Ranking and Next Queue

After S7 resolves, the strict priority is:

1. C3R fixed smoke, with CFG-OEC as the strongest prediction-level baseline.
2. SRBR fixed factorization, only if its SPA and relation components each pass
   their own controls; collect the actual Euler SPA prior before image scoring.
3. SD-RL, only after fixed and distilled actions pass on held-out prompts.
4. StructFlow structured-source transfer, documented separately in
   `S8_STRUCTURED_SOURCE_AUDIT.md` and treated as a low-priority negative
   control because frozen SDXL prior shift is severe.

While S7 occupies the GPU, CPU work can prepare action manifests, branch-order
regression tests, deterministic masks, prior metadata, and scorer manifests.
Do not reserve a GPU for RL in advance. Each queue transition must be gated by
the preceding result and its pre-registered confidence interval.
