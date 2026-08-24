# Scheduler-Consistent Trajectory Correction

## Motivation

The conference method changes a frozen diffusion trajectory after the U-Net
prediction.  Searches over attention, semantic transport, fixed latent bases,
and FreeU did not show guarded held-out headroom, so training an RL policy was
not justified.  This experiment tests a different soft connection to that
principle: preserve the same frozen model and initial noise, but correct the
Euler transition using the analytical ancestral transition already implied by
the scheduler.

This is motivated by the predictor-corrector pattern in LC-GRPO
([2608.05600](https://arxiv.org/abs/2608.05600)), trajectory correction in
CreFlow ([2605.14274](https://arxiv.org/abs/2605.14274)), and the path-space view
of diffusion policies ([2608.14430](https://arxiv.org/abs/2608.14430)). Those
papers mean this implementation must not claim a new RL estimator.  The narrow
question here is whether a scheduler-consistent, bounded correction is a useful
fixed action before any renderer or RL is considered.

## Method

For an epsilon-prediction Euler step with noise levels `sigma_from` and
`sigma_to`, define

```text
sigma_up^2   = sigma_to^2 (sigma_from^2 - sigma_to^2) / sigma_from^2
sigma_down^2 = sigma_to^2 - sigma_up^2
derivative   = (x_t - x_0_hat) / sigma_from
ancestral    = x_t + derivative (sigma_down - sigma_from)
raw          = mix (ancestral - euler_prev) + sigma_up sqrt(mix) epsilon
```

The optional trust cap bounds `||raw|| / ||euler_prev - x_t||` per sample.
`mix=0` returns the scheduler output without arithmetic or RNG consumption;
without a cap, `mix=1` is numerically the Euler-Ancestral transition.  The
hook is Stage-1-only, standalone, and records every correction/update norm.

## Development Result

The four-prompt probe used an explicitly matched Euler scheduler and checked
that `mix=0` produced identical PNG bytes.  Relative to Euler, mean TOPIQ-NR
changes were `+0.005616`, `+0.013206`, `+0.021842`, and `+0.024682` for mixes
`0.25`, `0.50`, `0.75`, and `1.00`, respectively.  This is exploratory only:
the per-prompt TOPIQ changes ranged from `-0.026565` to `+0.065755`, and the
largest mixes increased clipping and saturation on some prompts.  No
inferential claim or renderer/RL decision can be made from four prompts.

The newly registered development manifest has 11 prompt-disjoint prompts,
two seeds, a native Euler-Ancestral reference, and mixes `0.25/0.50/0.75`.  The source TSV hash and verified
PartiPrompts commit `5a657978134374ce28973948331b319adef164bd` are recorded in
the YAML and metadata.  It is a development gate, not the journal test set.

## Reviewer Gate

Before proposing a state-conditioned renderer, the fixed action must be
selected on development data and then beat `no_correction` on a newly frozen,
larger prompt-disjoint validation split.  The primary endpoint remains
TOPIQ-NR with crossed prompt/seed bootstrap and prompt sign-flip tests;
HPSv2/CLIP are witnesses, while clipping, saturation, contrast, colorfulness,
and sharpness are guards.  The native Euler-Ancestral result is a required
sampler control; DPM-Solver++ and UniPC at the same NFE are required before a
publication claim.  Holm correction and a pre-registered minimum effect are
required.  If the correction fails these guards, this route closes and no RL
is trained.  If it passes, only then will a small state-conditioned renderer
be searched and distilled; RL is allowed only if it beats both the fixed search
and the distilled controller under the same held-out protocol.
