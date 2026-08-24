# 2026 Literature Audit: Latent Trajectory and Diffusion-RL Extensions

**Audit date:** 2026-08-25. Titles, first-submission dates, and mechanism
summaries below were checked against the official arXiv abstract/API pages.
The 2026 entries are preprints; an arXiv posting is not evidence of peer
review or a state-of-the-art result.

## Bottom Line

"A small learned latent renderer", adaptive per-step control, dense rewards,
spectral regularization, and RL for diffusion are each already represented in
the literature. The defensible RepLDM extension is narrower: a **frozen SDXL
U-Net, one denoiser call per step, structure-conditioned basis residual,
scheduler-consistent injection, zero initialization, and an explicit update
trust cap**, evaluated at matched NFE/compute. The fixed operator must clear a
prompt-disjoint gate before any learned renderer or RL is run. Do not claim a
first latent renderer, first adaptive schedule, first structural intervention,
or first dense-reward estimator.

## Novelty and Control Matrix

### Direct inference-time overlap

| Work (arXiv first date) | Verified mechanism | Collision with the proposed claim | Required control / boundary |
|---|---|---|---|
| [FreeU, 2309.11497](https://arxiv.org/abs/2309.11497) (2023-09-20) | Reweights U-Net backbone and skip features at inference. | High for a backbone/skip structural story. | Mandatory baseline; report same NFE and wall time. |
| [DC-Solver, 2409.03755](https://arxiv.org/abs/2409.03755) (2024-09-05) | Dynamic compensation for predictor-corrector misalignment without extra denoiser calls. | High for scheduler-error compensation. | Matched no-extra-NFE control; include native Euler-Ancestral, DPM-Solver++, and UniPC. |
| [InfSplign, 2512.17851](https://arxiv.org/abs/2512.17851) (2025-12-19) | Cross-attention losses steer noise every denoising step for spatial alignment. | High for frozen-backbone latent intervention. | Compare on spatial prompts; distinguish cross-attention object placement from self-attention structural residuals. |
| [LaRender, 2508.07647](https://arxiv.org/abs/2508.07647) (2025-08-11) | Training-free latent volume rendering with occlusion graphs. | High for the word "renderer"; task is object occlusion. | Use as a task-specific baseline only when masks/occlusion are evaluated; do not claim first latent rendering. |
| [DIAMOND, 2602.00883](https://arxiv.org/abs/2602.00883) (2026-01-31) | Reconstructs a clean sample and corrects the trajectory to reduce artifacts. | High for clean-latent residual/trajectory correction. | Report a matched clean-latent correction and artifact metrics. |
| [DiffRGD, 2606.28417](https://arxiv.org/abs/2606.28417) (2026-06-25) | Riemannian gradient updates preserve a Gaussian-induced latent manifold. | High for moment/geometric constrained guidance. | Compare spherical/geodesic and norm-only projections; measure latent moments and update ratios. |
| [DUNE, 2607.09753](https://arxiv.org/abs/2607.09753) (2026-07-04) | Detects abrupt internal-latent deviations and suppresses selected entries. | High for internal-UNet structural intervention. | Include detect/suppress and shuffled-feature controls; avoid "first internal latent refinement." |
| [SPA, 2607.22091](https://arxiv.org/abs/2607.22091) (2026-07-24) | FFT guidance aligns intermediate spectra to an offline prior. | High for a spectral latent residual. | Include fixed radial-DCT/FFT projection and report frequency gain separately from image quality. |
| [SATeCo, 2403.17000](https://arxiv.org/abs/2403.17000) (2024-03-25) | Trains spatial/temporal adapters while freezing a diffusion UNet/VAE for video SR. | Medium: frozen-backbone feature adapter, different task. | Matched small adapter/random adapter; scope claims to text-to-image SDXL. |
| [Noise Level Correction, 2412.05488](https://arxiv.org/abs/2412.05488) (2024-12-07) | Learns a correction network for estimated noise level/manifold distance. | Medium for scheduler-state correction. | Compare noise-level calibration and keep extra-network compute explicit. |
| [Test-time Alignment, 2501.05803](https://arxiv.org/abs/2501.05803) (2025-01-10) | SMC samples a reward-aligned target at inference without fine-tuning. | Medium/high for test-time latent optimization. | Equalize number of model calls and report diversity/cost; no reward-only selection. |

### Adaptive schedules and diffusion-RL overlap

| Work (arXiv first date) | Verified mechanism | Collision / required interpretation |
|---|---|---|
| [AdaNAT, 2409.00342](https://arxiv.org/abs/2409.00342) (2024-08-31) | Learns sample-specific generation policies for token image generation. | Adaptive policy is not novel; it is an adjacent architecture/task control. |
| [Dynamic CFG, 2509.16131](https://arxiv.org/abs/2509.16131) (2025-09-19) | Uses online latent evaluations and greedy per-step CFG selection. | Mandatory dynamic-scalar baseline; RepLDM must explain that its action changes a TFSA/structural residual, not CFG. |
| [LeSAMP, 2607.23488](https://arxiv.org/abs/2607.23488) (2026-07-26) | RL emits prompt-conditioned, timestep-varying sampling parameters. | Directly rules out "learned adaptive schedule" as the contribution. |
| [BranchGRPO, 2509.06040](https://arxiv.org/abs/2509.06040) (2025-09-07) | Shared-prefix branching, pruning, and depth-wise advantages. | Branching/shared-prefix rollouts are a baseline design, not a new estimator. |
| [DRM, 2605.25661](https://arxiv.org/abs/2605.25661) (2026-05-25) | A frozen diffusion model supplies step-wise reward estimates on noisy latents. | Direct dense-latent reward overlap; compare before proposing a reward register/critic. |
| [DiNa-LRM, 2602.11146](https://arxiv.org/abs/2602.11146) (2026-02-11) | Learns a noise-calibrated preference model directly on noisy diffusion states. | Directly covers a latent reward/critic; a renderer cannot claim the first latent-space reward signal. |
| [DenseGRPO, 2601.20218](https://arxiv.org/abs/2601.20218) (2026-01-28) / [SDPO, 2411.11727](https://arxiv.org/abs/2411.11727) (2024-11-18) | Predicts step-wise reward gains from intermediate clean states and optimizes dense reward differences. | Dense credit assignment and clean-state reward prediction are mandatory controls, not new RL contributions. |
| [STAR, 2606.17979](https://arxiv.org/abs/2606.17979) (2026-06-16) | Allocates group-relative advantages over text-attention regions and timesteps. | Spatial/temporal credit assignment is covered; an RL renderer needs a different causal operator. |
| [SGPO, 2608.06768](https://arxiv.org/abs/2608.06768) (2026-08-07) | Stage-specific objectives address temporal mismatch and reward shortcuts. | Stage-aware RL is a control; explicitly compare stage partitions and reward backfilling. |
| [JAGG, 2607.17572](https://arxiv.org/abs/2607.17572) (2026-07-20) | Aggregates Jacobians to reduce GRPO backward cost. | Efficiency of diffusion-RL backprop is covered; do not claim compute novelty from cached Jacobians. |
| [CRD, 2603.14128](https://arxiv.org/abs/2603.14128) (2026-03-14) | Centered reward distillation stabilizes forward-process RL. | Required reward-distillation baseline if online learning is attempted. |
| [GeMPO, 2603.10250](https://arxiv.org/abs/2603.10250) (2026-03-10) | General monotone measure reweighting uses negative samples. | Compare against measure matching before inventing a new reweighting rule. |
| [MARBLE, 2605.06507](https://arxiv.org/abs/2605.06507) (2026-05-07) | Balances multiple reward aspects instead of a fixed weighted sum. | A composite reward is not novel; report raw components and Pareto trade-offs. |
| [RTDMD, 2605.26108](https://arxiv.org/abs/2605.26108) (2026-05-25) / [REST, 2608.09226](https://arxiv.org/abs/2608.09226) (2026-08-10) | Reward-tilted distribution matching and scored-trajectory distillation. | Search-then-distill must be a first-class control; RL is only justified if it beats it at equal cost. |
| [Path-space view, 2608.14430](https://arxiv.org/abs/2608.14430) (2026-08-14) | Unifies reverse-trajectory and forward-matching diffusion-RL estimators. | No estimator novelty from changing notation; state the exact path-space difference. |
| [GDRO, 2601.02036](https://arxiv.org/abs/2601.02036) (2026-01-05) | Offline group-level reward optimization for deterministic rectified-flow models, without ODE-to-SDE conversion. | A sampler-independent/offline group objective is already covered; report it before online rollouts. |
| [LC-GRPO, 2608.05600](https://arxiv.org/abs/2608.05600) (2026-08-06) / [MixGRPO, 2507.21802](https://arxiv.org/abs/2507.21802) (2025-07-29) / [CPS, 2509.05952](https://arxiv.org/abs/2509.05952) (2025-09-07) | ODE/SDE mixing, Langevin correction, or coefficient-preserving stochastic sampling. | Mandatory sampler controls for any stochastic trajectory claim; current S7 mixes are interpolation controls, not a derived SDE. |

### Latent representation and reward robustness

| Work (arXiv first date) | Verified mechanism | Implication |
|---|---|---|
| [Improving the Diffusability of Autoencoders, 2502.14831](https://arxiv.org/abs/2502.14831) (2025-02-20) | Finds excess latent high-frequency energy and modifies the autoencoder for diffusability. | A spectral latent claim needs frozen-VAE controls, reconstruction/moment checks, and an AE fine-tune upper bound. |
| [EQ-VAE, 2502.09509](https://arxiv.org/abs/2502.09509) (2025-02-13) | Regularizes transformation **equivariance** (scale/rotation), not invariance. | Measure equivariance error and content preservation separately; do not call this an invariance loss. |
| [Diffusing in the Right Space, 2606.03578](https://arxiv.org/abs/2606.03578) (2026-06-02) | Systematically compares semantic separability, equivariance, uniformity, spatial structure, and spectral smoothness as diffusability factors. | No single DCT statistic can be presented as a proven causal mechanism. |
| [Understanding Reward Hacking, 2601.03468](https://arxiv.org/abs/2601.03468) (2026-01-06) | Shows artifacts persist across reward ensembles and proposes an artifact reward regularizer. | Artifact detector, independent human preference, and pixel safety guards are mandatory. |
| [Pref-GRPO, 2508.20751](https://arxiv.org/abs/2508.20751) (2025-08-28) | Replaces normalized pointwise rewards with within-group pairwise preference wins to reduce reward hacking. | Pairwise preference and score-normalization controls are required for any RL comparison. |
| [DDRL, 2512.04332](https://arxiv.org/abs/2512.04332) (2025-12-03) | Anchors RL with forward KL to an off-policy data distribution to reduce quality loss/over-stylization/diversity collapse. | A KL-to-reference or data-regularized baseline is required for online RL. |
| [D2-Align, 2512.24146](https://arxiv.org/abs/2512.24146) (2025-12-30) | Corrects reward directions to mitigate preference-mode collapse. | Report diversity and style concentration; high reward alone is insufficient. |
| [Reward mechanics, 2606.02884](https://arxiv.org/abs/2606.02884) (2026-06-01) | Attributes reward hacking to finite-particle Doob-h estimation and derives damping. | Test reward damping/temperature and best-of-n controls before claiming a new anti-hacking loss. |
| [NoiseTilt, 2606.18066](https://arxiv.org/abs/2606.18066) (2026-06-16) | Keeps the reverse mean fixed and tilts the noise term using a whitened reward gradient. | A latent residual that changes the mean must be compared with a noise-compatible guidance control. |
| [HALO, 2502.06812](https://arxiv.org/abs/2502.06812) (2025-02-04) | Distills GPT-4o patch rewards and combines local/video rewards with Gran-DPO for video. | Patch rewards are a validated *evaluation/training design*, but the paper is video; do not transfer its claim without image patch validation. |
| [TextNorm, 2404.01863](https://arxiv.org/abs/2404.01863) (2024-04-02) / [VisionReward, 2412.21059](https://arxiv.org/abs/2412.21059) (2024-12-30) | Confidence-aware and multi-dimensional human-preference scoring. | Use independent, criterion-level witnesses rather than one scalar reward. |
| [Z-Reward, 2606.09076](https://arxiv.org/abs/2606.09076) (2026-06-08) / [DiT-Reward, 2606.23626](https://arxiv.org/abs/2606.23626) (2026-06-22) | Distributional rubric scores or generative-DiT representations for reward modeling. | A single CLIP/ImageReward score cannot certify a TPAMI gain; report score uncertainty and model-family holdouts. |

## Defensible Claim and Required Controls

The only plausible novelty left is the *combination and causal test*: a
low-capacity basis renderer conditioned on frozen SDXL decoder/self-attention
structure, injected in scheduler-update coordinates with a zero-init identity
and a measured trust cap. It must be compared to (1) no-op and conference
TFSA, (2) FreeU, (3) fixed searched basis coefficients, (4) matched random and
shuffled-feature renderers, (5) dynamic CFG/LeSAMP-style scalar schedules,
(6) a clean-latent/DIAMOND-like correction, and (7) an equal-cost extra U-Net
call. Report NFE, wall time, peak memory, parameters/FLOPs, latent moments,
spectral statistics, clipping/saturation, OCR/count/spatial accuracy, paired
human preference, and diversity. A learned policy cannot rescue a failed fixed
operator.

## Falsifiable Fixed-Action Hypotheses

These are **conditional follow-ups** and must not be appended to the running
S7 queue or used to reinterpret its selector.

### H1: Registered S7 action has sampler-independent headroom

On the frozen development manifest, then on a new prompt-disjoint validation
manifest, the already registered deterministic drift action (`mix=0.25`,
`noise_mode=none`) improves TOPIQ-NR by at least `+0.005` with a crossed-
bootstrap 95% interval above zero, remains non-inferior on HPSv2/CLIP, and
passes clipping/saturation guards **against both no-correction and native
Euler-Ancestral** at identical NFE/seed/noise. If it only beats Euler but not
the native/DPM-Solver++/UniPC references, classify it as a sampler effect and
close the route. Do not add sigma-horizon or new noise modes to this queue.

### H2: A fixed structural basis beats capacity-matched controls

Only if H1 passes, freeze one development-selected six-coordinate coefficient
vector in the registered order `[semantic, spectral-low, spectral-mid,
spectral-high, FreeU-backbone-minus-skip, Laplacian]` and record it in YAML.
The zero-init structural basis must then beat no-op, conference TFSA, the best
fixed scalar schedule, and a parameter/FLOP-matched random or shuffled-feature
renderer on TOPIQ-NR with a positive crossed-bootstrap interval, while
preserving latent channel moments within `1%`, keeping the scheduler update
ratio below the preregistered cap, and remaining non-inferior on HPSv2/CLIP and
human preference. Matching the random renderer or a searched fixed basis
falsifies the learned-structure claim; no RL is then authorized.

### H3: Spectral/equivariant latent structure transfers to image quality

Only after the fixed renderer gate, test one frozen-VAE, fixed radial-DCT
projection and one separately trained EQ-VAE-style adapter as an upper bound.
The projection is accepted only if it improves coarse-to-fine ordering and
held-out structural/detail metrics without worsening native-resolution
reconstruction/LPIPS, latent mean/variance by more than `1%`, OCR/count/spatial
accuracy, or human preference. A gain in DCT slope/high-frequency excess with
no quality gain, or a gain that disappears under a matched random projection,
falsifies the spectral mechanism. The EQ-VAE arm must be described as
equivariance regularization, not invariance.

## Immediate Queue-Safe Next Steps

1. Leave the current S7 queue, watcher, and external GPU jobs untouched.
2. After completion, verify manifest/config/prompt hashes and inspect paired
   action scores before any paper narrative or cherry-pick.
3. If H1 fails, close scheduler correction and do not start a renderer/RL run.
4. If H1 passes, run a CPU structural smoke, then one frozen development
   renderer action with the controls above; only a passed fixed gate permits
   search-then-distill, and only a subsequent held-out win over distillation
   permits online RL.
5. Use patch/multi-crop and color-normalized scores as independent witnesses,
   not as the sole training reward. A reward increase accompanied by
   saturation, artifact rate, style concentration, or diversity loss is a
   failed result regardless of scalar score.
