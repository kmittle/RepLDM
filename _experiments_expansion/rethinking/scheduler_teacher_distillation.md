# Scheduler-Teacher Distillation (Conditional Proposal)

## Why This Is A Separate Route

The current S7 action changes an Euler transition toward an ancestral
transition.  If it wins, the gain may come from the sampler rather than from a
new structural renderer.  This proposal makes that distinction explicit: a
strong native sampler is used only to produce paired teacher transitions, and a
small renderer is tested on whether it can reproduce the useful residual with
one ordinary U-Net call at deployment.

This is a conditional proposal.  It cannot start until S7 passes its fixed
action and validation gates.  It is not a new solver, distillation objective,
or RL estimator.  DPM-Solver++, UniPC, REST, and RTDMD are direct controls and
set the novelty boundary.

## Teacher Data

For each prompt, seed, timestep, and initial noise, run the frozen SDXL model
with an Euler baseline and one frozen teacher sampler.  Record the scheduler
configuration hash, installed `diffusers` version, actual U-Net call count,
and the teacher transition in the same latent coordinates.  The target is
`teacher_prev_sample - euler_prev_sample`, together with the teacher's
predicted-clean latent when available.  Teacher generation is never used on
the final test prompts.

The primary teacher must be a native sampler at matched NFE.  A DPM-Solver++ or
UniPC arm with extra model calls is an upper bound, not a fair one-call claim.
Ancestral noise, solver order, timestep list, CFG, VAE precision, and random
number generator are frozen before any scores are inspected.

## Renderer And Controls

The renderer consumes only one Euler denoiser state: `x_t`, `x0_hat`, the
scheduler update, timestep features, prompt features, and detached U-Net
structure summaries.  Its output is a zero-initialized, six-basis residual in
scheduler-update coordinates with a strict post-cast trust cap.  Compare:

* no correction and the fixed teacher residual;
* a coefficient-only renderer and a small state-conditioned renderer;
* shuffled/detached features and an L2-matched random renderer;
* a schedule-only policy and an equal-cost extra-U-Net teacher upper bound.

All arms use the same prompts, seeds, CFG, VAE, resolution, and deployment NFE.
Report teacher-search cost and renderer training cost amortized per image.

## Gates

1. A fixed teacher residual must beat Euler on a new development split with
   TOPIQ-NR delta at least `+0.005`, a crossed-bootstrap interval above zero,
   Holm-adjusted prompt sign-flip significance, HPSv2/CLIP non-inferiority, and
   clipping/saturation/contrast guards.
2. A deterministic renderer must beat its fixed teacher coefficients and all
   matched-capacity controls on unseen prompts.  If it only matches the
   teacher, report sampler distillation and stop.
3. Online RL is allowed only if it beats the frozen distiller on unseen prompts
   under independent patch/color-normalized/human witnesses and recovers its
   teacher-search and training cost.  KL/reference drift, diversity, OCR,
   counting, spatial accuracy, and per-step off-target drift are mandatory.

If the fixed teacher action loses to native Euler-Ancestral, DPM-Solver++, or
UniPC references, classify the result as a sampler effect and close the
renderer/RL route.  No reward or controller tuning may reopen a failed gate.
