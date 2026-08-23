# Model Iteration Ledger

This ledger separates hypotheses fixed before generation from conclusions written after scoring. Generated artifacts remain under ignored `outputs/`; every run records the producing Git commit.

## Iteration Summary

| ID | Candidate | Status | Decision |
|---|---|---|---|
| S0 | Constant scalar TFSA residual | Invalidated | Scale increases saturation and degrades TOPIQ-NR; low scale converges to no-AG. |
| S1 | Low/mid/high spectral residual gains | Invalidated | Best action, `mid_only_0.004`, is non-superior to no-AG; seed-CV finds no adaptive headroom. |
| S2 | Moment-Tangent Attention Guidance (MTAG) | Invalidated | It removes moment drift but does not improve quality or preference. |
| S3 | Trajectory-Cone Moment Guidance (TCMG) | Development grid frozen | Cone is active and non-catastrophic at low scales; no efficacy claim yet. |

S0-S2 evidence is reported in `EXPERIMENT_RESULTS.md`. Their action spaces must not be reused for RL training.

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

## S2 Smoke Outcome

Run: `outputs/exp_moment_tangent/smoke_2prompt_1seed_v1`, produced by commit `b9f3fbc` on GPU 1. All 32 expected PNG/JSON pairs are valid 1024² RGB images, contain complete action metadata, and have distinct PNG hashes. No-AG, conference expert, and raw scales `0.001/0.002/0.004` exactly reproduce the corresponding historical PNG SHA-256 hashes on the same GPU.

The two-prompt means below are descriptive only:

| Action | Δ TOPIQ | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| raw 0.001 | +0.001310 | +0.000366 | +0.001281 | +0.008493 |
| mean-centered 0.001 | -0.000746 | +0.001953 | -0.002757 | +0.005103 |
| tangent 0.002 | -0.007128 | +0.005005 | -0.004189 | -0.000084 |
| tangent-rescaled 0.002 | -0.006053 | +0.001465 | +0.003750 | +0.003801 |

The fixed montage shows no consistent structural winner. Moment tangent suppresses the raw update's contrast/saturation route, but the energy-matched variant introduces a flat-design wall-object artifact. Per the registered catastrophic thresholds, tangent `0.008`, tangent-rescaled `0.004/0.008`, and mean-centered `0.004` are removed. The development grid is frozen in `eval-pipeline/configs/moment_tangent_development.yaml`; retained intervals are mean-centered `0.001–0.002`, tangent `0.001–0.004`, and tangent-rescaled `0.001–0.002`. Raw `0.001` is the previously established best scalar control. The 12-prompt set remains development data because it informed S2; any passing action still requires prompt-disjoint confirmation.

## S2 Development Outcome

Run: `outputs/exp_moment_tangent/development_12prompt_3seed_v1`, 12 prompts × 3 seeds × 10 actions, produced by commit `65ba734`. All 360 records are complete and paired; 108 no-AG/expert/raw control PNGs exactly reproduce prior same-GPU hashes.

No action passes the registered gate. The least-negative TOPIQ result is mean-centered `0.001`: `-0.000946`, 95% CI `[-0.005246,+0.003537]`. Moment tangent `0.001/0.002/0.004` gives `-0.002316/-0.002268/-0.004424`; energy-matched `0.001` is significantly worse after within-metric Holm correction (`-0.003763`, CI `[-0.005925,-0.001483]`, adjusted `p=0.025920`). HPSv2 changes for every S2 action are within `±0.001`; CLIP does not improve.

The mechanism itself is supported: raw `0.001` increases saturation by `+0.013449`, whereas the three unscaled tangent actions change it by `-0.000862/-0.000997/-0.001767`, with clipped fraction within `±0.001`. Thus the fixed-moment constraint removes the color/contrast route, but the remaining spatial TFSA direction has no measurable benefit. TOPIQ-selected per-prompt seed-CV is `+0.001791` vs no-AG, CI `[-0.004392,+0.007529]`, while increasing clipping and saturation. HPS-selected per-prompt actions underperform the global expert by `-0.002082`, CI `[-0.003872,-0.000434]`. The fixed montage shows composition changes but no stable structural repair. S2 is closed; its scales must not be tuned further and it cannot seed RL training.

## S3 Registered Hypothesis

S2 shows that moment drift is harmful but also that unconditional TFSA spatial transport is not reliably useful. S3 tests whether the harmful part is the component that opposes the frozen model's own denoising transition. Let `P_T` be the S2 fixed-moment tangent projection, `d = TFSA(z)-z`, and `g` the scheduler update that produced `z`. Per sample and latent channel:

```text
v = P_T(d),  q = P_T(g)
c = v - min(<v,q>, 0) q / ||q||^2
z' = Exp_z(scale c).
```

Both `v` and `q` lie in the same tangent space, so the half-space projection keeps the fixed-moment guarantee and enforces `<c,q> >= 0`. It is parameter-free, state-dependent, uses no additional UNet call, and has a falsifiable role: if opposing components caused S2's loss, TCMG should beat plain moment tangent at matched scale. An energy-matched variant separates cone geometry from attenuation.

GAG's parallel/orthogonal decomposition concerns sparse-versus-dense cross-attention guidance. TCMG instead constrains an external self-attention latent update against the scheduler's realized transition on a fixed-moment manifold. The decomposition alone is not claimed as generic novelty; the journal claim requires the combined operator and subsequent learned policy to survive baselines.

Registered smoke: `eval-pipeline/configs/trajectory_cone_smoke.yaml`, two prompts, seed `0`, GPU 1. It includes no-AG, expert, raw `0.001`, plain tangent `0.002`, TCMG `0.001/0.002/0.004/0.008`, and energy-matched TCMG `0.001/0.002/0.004`. Apply the same non-finite/no-op, TOPIQ `-0.05`, and clipping `+0.01` rejection rules as S2. Smoke only fixes a contiguous scale range. A 12×3 development run is allowed only if at least one TCMG interval survives; the S2 gate thresholds remain unchanged.

## S3 Smoke Outcome

Run: `outputs/exp_trajectory_cone/smoke_2prompt_1seed_v1`, produced by commit `0e00fe6` on GPU 1. All 22 records are complete, finite, and pixel-distinct. Eight no-AG/expert/raw/plain-tangent controls exactly reproduce S2 hashes, so the tangent refactor has no integration regression. TCMG differs from plain tangent at the PNG level, showing that the cone is active on real scheduler trajectories.

Descriptive two-prompt means are uniformly negative on TOPIQ: TCMG `0.001/0.002/0.004` gives `-0.006597/-0.007909/-0.010637`; the energy-matched `0.001/0.002` gives `-0.010521/-0.008155`. TCMG `0.002` has HPSv2 `+0.005859`, but its two-prompt interval is uninformative and ImageReward is `-0.065652`. The fixed montage shows state-dependent composition changes without a consistent structural repair.

TCMG `0.008` is removed because one prompt exceeds the `+0.01` clipping limit; energy-matched `0.004` is removed for the same reason. The contiguous non-catastrophic ranges are TCMG `0.001–0.004` and energy-matched TCMG `0.001–0.002`. These ranges, plus no-AG, expert, raw `0.001`, and plain tangent `0.002`, are frozen in `eval-pipeline/configs/trajectory_cone_development.yaml`. The smoke does not provide efficacy evidence.
