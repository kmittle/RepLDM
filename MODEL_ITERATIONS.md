# Model Iteration Ledger

This ledger separates hypotheses fixed before generation from conclusions written after scoring. Generated artifacts remain under ignored `outputs/`; every run records the producing Git commit.

## Iteration Summary

| ID | Candidate | Status | Decision |
|---|---|---|---|
| S0 | Constant scalar TFSA residual | Invalidated | Scale increases saturation and degrades TOPIQ-NR; low scale converges to no-AG. |
| S1 | Low/mid/high spectral residual gains | Invalidated | Best action, `mid_only_0.004`, is non-superior to no-AG; seed-CV finds no adaptive headroom. |
| S2 | Moment-Tangent Attention Guidance (MTAG) | Invalidated | It removes moment drift but does not improve quality or preference. |
| S3 | Trajectory-Cone Moment Guidance (TCMG) | Invalidated | Cone geometry is active but does not improve quality, preference, or adaptive headroom. |
| S4 | 2048² Stage-2 target-domain audit | Invalidated | Stage 2 preserves the negative ranking; no static action beats no-AG or licenses RL. |
| S5 | Scheduler-Consistent Reciprocal Semantic Transport (SCRST) | Registered | Change the proposal source to conditional UNet self-attention and scheduler-returned predicted-clean latents. |

S0-S4 evidence is reported in `EXPERIMENT_RESULTS.md`. Their action spaces must not be reused for RL training.

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

## S3 Development Outcome

Run: `outputs/exp_trajectory_cone/development_12prompt_3seed_v1`, 12 prompts × 3 seeds × 9 actions, produced by commit `b0aa343`. All 324 PNG/JSON pairs are valid 1024² RGB images; every prompt/seed block contains all nine actions on one GPU, and each GPU produced 81 records. All 144 no-AG/expert/raw/plain-tangent controls exactly reproduce their S2 PNG hashes.

No TCMG action passes the registered gate. Relative to no-AG:

| Action | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| cone 0.001 | -0.002519 [-0.005576,-0.000152] | -0.000553 | -0.000175 | -0.000812 |
| cone 0.002 | -0.001740 [-0.005274,+0.001283] | +0.000085 | +0.000050 | -0.000385 |
| cone 0.004 | -0.004220 [-0.010200,+0.001771] | +0.000665 | +0.000600 | -0.000368 |
| cone-rescaled 0.001 | -0.003178 [-0.005424,-0.000840] | +0.000173 | +0.000085 | -0.000642 |
| cone-rescaled 0.002 | -0.005843 [-0.010720,-0.000922] | +0.000346 | +0.000998 | +0.000022 |

At matched scale, cone `0.002` improves TOPIQ over plain tangent `0.002` by only `+0.000528`, CI `[-0.000412,+0.001596]`; HPSv2 changes by `+0.000031`. Thus the half-space projection can remove an opposing component, but the removed component does not explain S2's efficacy failure.

TOPIQ-selected seed-CV chooses no-AG globally in all three folds. Its per-prompt result is `+0.001120` vs no-AG, CI `[-0.004272,+0.006516]`, while clipping rises `+0.001262` and saturation `+0.012097`. HPS-selected per-prompt actions lose `-0.001604` to the global expert, CI `[-0.003455,-0.000190]`, and substantially worsen both guards. The complete seed-0 montage at `outputs/exp_trajectory_cone/development_12prompt_3seed_v1/figs/all_actions_seed0.png` shows state-dependent composition changes but no stable structural or text repair. S3 is closed without another scale sweep and cannot seed RL.

## S4 Registered Stage-2 Audit

S0-S3 evaluated the 1024² Stage-1 image, while RepLDM's central product claim concerns the final high-resolution image after resampling. S4 is a target-domain diagnostic to test that mismatch; it is not permission to bypass the failed RL gates or treat reused development prompts as confirmation data. Attention Guidance remains Stage-1-only, so S4 measures whether Stage 2 amplifies or reverses its downstream effect.

Before batch generation, Stage-2 correctness must pass on `prompts/stage2_smoke.csv`, seed `0`, one GPU, and `configs/stage2_engineering_smoke.yaml`. Stage-2 noise must use the task generator rather than global RNG; normal 2048² decode must place the VAE and latents on the execution device when phase offload is enabled. The two duplicate no-AG actions must have identical final PNG hashes, the expert must differ, all outputs must be 2048² RGB, metadata must record Stage-2 settings, and a repeated run must reproduce hashes. Failure blocks the pilot.

The frozen pilot is 12 prompts × 3 seeds × 5 actions = 180 final 2048² images using `configs/stage2_transfer_pilot.yaml`: no-AG, conference expert, raw `0.001`, plain tangent `0.002`, and cone `0.002`. These form a mechanistic ladder fixed from prior evidence; no high-resolution scale search is allowed. Same prompt/seed actions must share a GPU and task-seeded Stage-2 noise.

TOPIQ-NR is primary. A TCMG result can reopen the method only if its gain over no-AG is at least `+0.005`, the 95% CI excludes zero, the within-metric Holm-adjusted test is below `0.05`, and direct paired comparisons also beat expert, raw, and plain tangent. HPSv2 and CLIP cosine must have lower CI bounds above `-0.003` and `-0.005`; clipped fraction and saturation mean deltas must stay below `+0.001` and `+0.005`. Report ImageReward, patch-IR, aesthetic, contrast, colorfulness, sharpness, all failed actions, and a fixed montage. A pass still requires prompt-disjoint confirmation and blinded high-resolution preference/detail crops. If no action passes, close Stage-1 Attention Guidance for this target pipeline before inventing a learned controller.

## S4 Engineering Outcome

Runs `outputs/exp_stage2_transfer/engineering_smoke_v1` and `engineering_smoke_repeat_v1` were produced independently from commit `f18b6b1` on GPU 1. Both completed the normal 2048² encoder/decoder path for all three actions without OOM or device mismatch; observed peak GPU memory use was approximately 21.2 GB. Every output is a valid nonblank 2048² RGB PNG with complete Stage-2 metadata.

Within each run, `no_ag` and `no_ag_repeat` have the same SHA-256 (`3187f72c...0930a55`), while `conference_expert` is distinct (`a513da66...8efddf`). Across fresh processes and output directories, all three action hashes reproduce exactly. The task generator therefore controls both sampling phases, the pipeline remains reentrant after CPU phase offload, and the engineering gate authorizes the already-frozen five-action pilot without changing actions or thresholds.

## S4 Pilot Outcome

Run: `outputs/exp_stage2_transfer/pilot_12prompt_3seed_v1`, 12 prompts × 3 seeds × 5 actions, produced by commit `a31cb75`. All 180 final images are valid 2048² RGB PNGs; 36 prompt/seed blocks are complete and same-device, each GPU produced 45 records, all within-block action hashes differ, and the two smoke controls reproduce exactly. Strict scoring produced all 14 configured outputs for every image.

No action improves the primary endpoint over no-AG:

| Action | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ CLIP | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|---:|
| conference expert | -0.020052 [-0.037713,-0.003316] | +0.004754 | -0.004435 | +0.010235 | +0.044811 |
| raw 0.001 | -0.005907 [-0.011608,-0.000872] | +0.001333 | -0.004632 | +0.002392 | +0.014894 |
| plain tangent 0.002 | -0.002836 [-0.006932,+0.001057] | +0.000756 | -0.003126 | -0.000102 | -0.000058 |
| cone 0.002 | -0.002091 [-0.006817,+0.002305] | +0.001133 | -0.003167 | -0.000059 | +0.000653 |

Cone `0.002` is only `+0.000745` above matched plain tangent, CI `[-0.001212,+0.002815]`. It is `+0.003816` above raw, CI `[-0.001550,+0.009215]`, and `+0.017961` above expert, CI `[+0.002937,+0.033567]`; these are reductions in harm, not efficacy, because cone remains below no-AG and fails the preregistered CLIP non-inferiority bound. Patch-IR also decreases by `-0.017641`, with no local-detail confirmation signal.

TOPIQ seed-CV produces `-0.002148` for the global policy and `-0.003216` for per-prompt selection versus no-AG. HPS selection gives a nominal `+0.005022`, CI `[+0.000604,+0.009759]`, but TOPIQ falls `-0.012735`, clipping rises `+0.001673`, and saturation rises `+0.028848`; this is proxy/guard conflict, not adaptive headroom. The fixed montage at `outputs/exp_stage2_transfer/pilot_12prompt_3seed_v1/figs/all_actions_seed0.png` shows stronger expert/raw color and composition changes without stable structure or text repair. Stage 2 therefore does not rescue Stage-1 Attention Guidance. S4 and the target-domain exception are closed without high-resolution scale tuning.

## Post-S4 Algorithm Bar

No S5 claim is registered. Four failed operator families plus the high-resolution audit are evidence against continuing to transform the same raw four-channel noisy-latent TFSA residual. The next admissible hypothesis must change the proposal source, not its scale, schedule, frequency weights, geometric projection, or controller.

A defensible candidate is scheduler-consistent predicted-clean semantic transport: obtain the scheduler's predicted clean sample `x0`, derive spatial transport from frozen UNet self-attention features rather than four-channel latent dot products, apply a fixed-moment transport to `x0`, then reconstruct the next latent while holding the predicted noise component fixed. This specifically tests whether S0-S4 failed because TFSA was computed on noisy, semantically weak latents. Before implementation it requires a full related-work audit and an exact scheduler-reconstruction test.

Any such operator must first use a new prompt-disjoint development set and fixed static actions. It must compare raw-latent TFSA, predicted-clean TFSA, semantic-feature transport, no-AG, conference expert, and an equal-compute post-step control; if it adds a UNet call, it also needs an equal-call score-correction baseline. The same `+0.005` TOPIQ gate, preference/alignment non-inferiority, pixel guards, high-resolution transfer, and fixed qualitative views apply. Only a static winner can justify search, distillation, or RL; another null result closes Attention Guidance as the journal-extension axis.

## S5 Related-Work Decision

`S5_RELATED_WORK.md` records the full-text audit frozen on 24 August 2026. PLADIS and GAG already cover training-free, single-forward, sparse-versus-dense attention contrast and are mandatory same-NFE baselines. SAG, PAG, SEG, and ASAG form an extra-prediction upper-bound group. Predicted `x_0`, internal features, attention extrapolation, projection, adaptive schedules, and RL are all individually overlapped. The only claim under test is the complete scheduler-consistent reciprocal semantic transport operator; no component-level novelty is claimed.

## S5 Registered Mechanistic Hypothesis

S0-S4 may fail because four-channel noisy-latent dot products are semantically weak. S5 reads the standard conditional branch of

```text
up_blocks.0.attentions.0.transformer_blocks.0.attn1
```

without replacing its processor or changing its output. At 1024² this layer has a 32x32 grid, 20 heads, and 1280 channels. A frozen diagnostic found nonlocal mass beyond four tokens of `0.8433/0.7424/0.7132/0.6930/0.6163` at timesteps `801/601/401/201/1`, but normalized entropy remained `0.9000/0.8636/0.8414/0.8202/0.7876`. Direct dense transport therefore risks global averaging and motivates reciprocal mutual-neighbor sparsification.

For the normal conditional forward, let

```text
A = mean_h softmax(Q_h K_h^T / sqrt(d))
R_ij = sqrt(A_ij A_ji)
W = row_normalize(R restricted to mutual top-16 edges plus the diagonal)
c_t = 1 - mean_i H(A_i) / log(N).
```

Area-downsample the scheduler-returned `x_0` to the attention grid, form `d = upsample(W x_0 - x_0)`, and project `d` per latent channel onto the fixed-mean, fixed-variance tangent space. Move on that sphere by the fixed angle `theta * c_t`; call the result `guided_x_0`. The only scheduler integration allowed is

```text
step_output = scheduler.step(...)
u = guided_x_0 - step_output.pred_original_sample
latents = step_output.prev_sample + u.
```

The scheduler is Euler with epsilon prediction. S5 must not reconstruct `x_0` or the Euler step independently: the fp16 discrepancy from doing so can reach `0.03125`. `theta=0` must exactly reproduce no-AG. The operator uses no backpropagation and no additional UNet evaluation.

## S5 Frozen Data and Controls

`eval-pipeline/prompts/s5_development.csv` contains 12 new prompts, two each from PartiPrompts challenges Complex, Fine-grained Detail, Properties & Positioning, Quantity, Writing & Symbols, and Perspective. `s5_smoke.csv` contains two different prompts. They are disjoint from all prior repository prompt CSVs. Source commit is `5a657978134374ce28973948331b319adef164bd`; rows were selected before generation by the SHA-256 procedure recorded in the evaluation README. Prompts cannot be replaced after viewing images.

Required controls are no-AG, the conference expert, raw noisy-latent TFSA, plain predicted-clean TFSA, predicted-clean reciprocal latent affinity, reciprocal semantic affinity, and the same semantic graph with rows and columns jointly permuted before application. The permutation preserves graph values and compute while destroying spatial alignment. Also include CFG-only 5.0, official PLADIS (`alpha=1.5`, scale `2.0`), and official GAG (`lambda=10`, `eta=15`, `zeta=0`). PLADIS/GAG run with CFG 5.0 as registered by their papers; all RepLDM controls retain CFG 7.5. Report NFE, wall time, and peak GPU memory.

## S5 Registered Smoke and Engineering Gates

`eval-pipeline/configs/s5_smoke.yaml` freezes top-k at `16`, the semantic layer above, matched control angle `0.02`, and a semantic angle range `0.005/0.01/0.02/0.04`. Run two prompts, seed `0`, on one GPU. Smoke is only for implementation and catastrophic-range rejection; it must not rank actions or choose an isolated angle. Remove only a contiguous extreme when any prompt has non-finite output, a nonzero no-op, TOPIQ below no-AG by more than `0.05`, or clipped fraction above no-AG by more than `0.01`.

Before interpreting smoke, unit and real-model checks must show: scheduler-returned `x_0` is used; no-AG and zero-angle PNG hashes match exactly; channel means and variances are preserved in `guided_x_0`; reciprocal support is mutual and row-stochastic; the spatial permutation is deterministic; repeat runs reproduce hashes; and every sidecar records latency, peak memory, normalized affinity entropy, transport confidence, and transport-to-scheduler update norm ratio.

## S5 Development Gate

After smoke, freeze the complete contiguous non-catastrophic semantic interval and all matched controls in a new YAML before development generation. Run the 12 frozen prompts with seeds `0,42,123`, keeping every prompt/seed action block on one GPU. TOPIQ-NR is primary. Report HPSv2, ImageReward, patch-IR, CLIP alignment, aesthetic score, clipped fraction, saturation, contrast, colorfulness, sharpness, latency, memory, and every failed action. Use crossed prompt/seed bootstrap intervals, prompt sign-flip tests, and within-metric Holm correction.

An S5 action advances only if all conditions hold:

1. TOPIQ gain over no-AG is at least `+0.005`, its 95% CI excludes zero, and its within-metric Holm-adjusted test is below `0.05`.
2. It directly beats the conference expert, raw TFSA, matched latent-affinity and permuted-graph controls, CFG 5.0, PLADIS, and GAG under the same NFE accounting.
3. HPSv2 and CLIP are non-inferior; clipped fraction and saturation do not exceed no-AG by more than `0.001` and `0.005`.
4. A fixed all-action montage shows real structure, counting, position, text, or detail improvement rather than global color, contrast, or sharpness manipulation.

A pass licenses prompt-disjoint confirmation and Stage-2 transfer, not RL. A null result closes Attention Guidance as the journal extension; no angle, top-k, layer, schedule, reward, or controller sweep follows.
