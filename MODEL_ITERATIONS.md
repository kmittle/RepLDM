# Model Iteration Ledger

This ledger separates hypotheses fixed before generation from conclusions written after scoring. Generated artifacts remain under ignored `outputs/`; every run records the producing Git commit.

## Iteration Summary

| ID | Candidate | Status | Decision |
|---|---|---|---|
| S0 | Constant scalar TFSA residual | Invalidated | Scale increases saturation and degrades TOPIQ-NR; low scale converges to no-AG. |
| S1 | Low/mid/high spectral residual gains | Invalidated | Best action, `mid_only_0.004`, is non-superior to no-AG; seed-CV finds no adaptive headroom. |
| S2 | Moment-Tangent Attention Guidance (MTAG) | Registered | Test whether spatial rearrangement can be retained without per-channel moment drift. |

S0 and S1 evidence is reported in `EXPERIMENT_RESULTS.md`. Their action spaces must not be reused for RL training.

## S2 Mechanistic Hypothesis

The conference TFSA update mixes spatial token aggregation with changes in latent channel mean and variance. The latter provides a direct route to clipping, contrast, and saturation changes. For each sample and channel, let

```text
d = TFSA(z) - z
x = z - mean_spatial(z)
v = center_spatial(d) - <center_spatial(d), x> x / ||x||^2
theta = scale ||v|| / ||x||
z' = mean_spatial(z) + cos(theta)x + sin(theta)||x||v/||v||.
```

This is an exponential-map step on the sphere of centered latents. It preserves the input channel mean and variance up to numerical rounding while retaining a TFSA-derived spatial direction. `moment_tangent_rescaled` matches `||v||` to the raw residual norm before the map, separating geometry from simple attenuation.

The closest attention-guidance work (GAG, NAG, ASAG) changes attention extrapolation, normalization, or the negative attention branch. MTAG instead constrains RepLDM's external post-scheduler latent update on a fixed-moment manifold. Prompt/timestep RL work such as LeSAMP and AdaGen overlaps future policy learning, not this operator; adaptive control is not itself a novelty claim.

## S2 Registered Smoke

- Commit first, then generate `smoke.csv` with seed `0` on one GPU so all actions in a pair share hardware.
- Compare `raw`, `mean_centered`, `moment_tangent`, and energy-matched moment tangent at scales `0.001`, `0.002`, and `0.004`; probe `0.008` only for tangent modes. Include no-AG and the conference expert.
- Smoke is for correctness and scale-range rejection, not efficacy claims. Reject an action on non-finite output, unchanged nonzero output, TOPIQ loss greater than `0.05`, or clipped-fraction increase greater than `0.01` on either prompt.
- If a tangent family remains viable, retain a contiguous scale interval for the 12-prompt, 3-seed development pilot. Do not select an isolated scale from smoke.

## S2 Development Gate

TOPIQ-NR is primary. Report HPSv2, ImageReward, patch-IR, CLIP alignment, aesthetic score, clipped fraction, saturation, contrast, colorfulness, and sharpness for every registered action. Use paired same-device blocks, crossed prompt/seed bootstrap intervals, prompt sign-flip tests, and Holm correction.

A candidate enters RL work only if it meets all of the following on the frozen development grid:

1. Mean TOPIQ-NR gain over no-AG is at least `+0.005`, its 95% CI excludes zero, and the within-metric Holm-adjusted test is below `0.05`.
2. It also beats the conference expert and best raw scalar under the same generation budget.
3. HPSv2 and CLIP are non-inferior; clipped fraction and saturation do not exceed no-AG by more than `0.001` and `0.005`, respectively.
4. A fixed montage shows a plausible structural/detail change rather than global color or contrast manipulation.

Passing this gate licenses a new prompt-disjoint confirmation set and high-resolution Stage-2 transfer test. It does not itself establish a journal claim. Failure kills S2 and triggers a new operator hypothesis rather than reward tuning or controller complexity.
