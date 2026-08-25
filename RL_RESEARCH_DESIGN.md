# Structural Latent Rendering: RL Research Design

This document is a proposal for the journal extension. It does not change the
registered S5 experiment or authorize an RL run. The first decision gate is a
fresh scheduler-native static result; a policy or a larger network must not be
used to rescue the failed S5 or legacy LR-1 action families.

## Positioning

The conference contribution is a training-free latent reprogramming operator:
the frozen diffusion model's internal structure proposes a latent residual.
The journal extension can make that principle learnable without claiming that
RL, adaptive schedules, or attention extrapolation are new. The proposed claim
is narrower and testable:

> A low-capacity, structure-conditioned latent renderer can learn safe residuals
> that improve the frozen model's coarse-to-fine trajectory at matched NFE and
> compute, while preserving latent moments and spatial behavior.

If S5 fails, this becomes a separately registered latent-renderer project; it
must not be written as a successful continuation of Attention Guidance.

## Literature Constraints

* **FreeU (arXiv:2309.11497)** attributes denoising mainly to U-Net backbone
  features and high-frequency detail mainly to skip features. Its backbone/skip
  reweighting is a mandatory inference baseline and motivates feature probes,
  not a novelty claim.
* **Improving the Diffusability of Autoencoders (ICML 2025,
  arXiv:2502.14831)** finds excess latent high-frequency energy and trains a
  scale-equivariant decoder with downsampled reconstruction. **EQ-VAE (ICML
  2025, arXiv:2502.09509)** regularizes latent spatial equivariance. Both are
  adjacent autoencoder interventions; initially we keep the SDXL VAE frozen,
  measure their spectral/equivariance diagnostics, and treat an AE fine-tune as
  a separate upper-bound arm.
* PLADIS, GAG, SAG/PAG/SEG/ASAG, dynamic-CFG/AdaGen/LeSAMP, BranchGRPO,
  DRM/SGPO, and recent path-space diffusion-RL work cover sparse attention,
  extra denoiser calls, schedules, and policy optimization. They must be
  baselines or explicit scope boundaries, not components of a novelty claim.
* **Diffusion Controller (arXiv:2603.06981)** already freezes a diffusion
  backbone and trains a 12M latent U-Net side network for per-step structured
  score correction with SFT, reward-weighted learning, or PPO. **HVP
  (arXiv:2605.21661)** learns initial-noise and per-step additive control
  policies around a frozen denoiser. Therefore neither a latent side network,
  frozen-backbone reward training, nor per-step RL is a novelty claim. A
  DiffCon-style free residual head and LoRA/DRaFT are required upper bounds.

The 2026 full-text audit tightens this boundary further. The unified path-space
view (arXiv:2608.14430) treats reverse-trajectory gradients and forward reward
matching as one objective; SGPO (2608.06768) assigns different objectives to
SNR/semantic stages; JAGG (2607.17572) approximates intermediate Jacobians;
CRD (2603.14128), GeMPO (2603.10250), MARBLE (2605.06507), RTDMD
(2605.26108), and REST (2608.09226) cover centered/anchored reward matching,
generalized reweighting, multi-reward gradient balancing, and scored-trajectory
distillation. Consequently, the first learning comparison must be
search-then-distill with an anchored, multi-reward objective. A later RL arm may
use antithetic shared-prefix rollouts, but its estimator, stage partition, and
reward aggregation are controls to be compared, not claimed contributions.

Two additional overlaps are important for the word "renderer": **LaRender
(arXiv:2508.07647)** already performs training-free volumetric compositing of
object-wise cross-attention features in latent space, using masks and an
occlusion graph. **SATeCo (arXiv:2403.17000)** freezes a pretrained UNet/VAE
and trains small spatial/temporal feature adapters. The proposed method is not
the first latent renderer, frozen-backbone adapter, or learned diffusion
controller. Its only defensible target is a sub-1M bounded residual in exact
scheduler coordinates whose structural mechanism is causally verified.
Non-attention latent/frequency descriptors are primary; self-attention and
decoder features are ablations because their semantic graph cost is substantial.
LaRender-style compositing, FreeU, DiffCon-style residuals, and matched random
adapters must be included wherever their task is well-defined.

The second supplied paper is also more precisely described as **equivariance**
regularization: EQ-VAE asks transformed inputs to produce correspondingly
transformed latents/reconstructions, rather than forcing all transformed views
to share an invariant code. This distinction determines the renderer test:
measure both equivariance error and unwanted content change instead of calling
one scalar an "invariance" score.

### Representation-side factorial (post-LR-1 only)

The current LR-1 gate deliberately freezes the SDXL VAE, so a positive result
can be attributed to the renderer. If LR-1 passes, register a separate,
prompt-disjoint representation study before touching validation or test:

| Arm | VAE/AE | Renderer | Purpose |
|---|---|---|---|
| R0 | frozen SDXL VAE | frozen winner | causal reference |
| R1 | frozen VAE + fixed radial-DCT projection | frozen winner | non-learned spectral control |
| R2 | AE adapter with high-frequency penalty | frozen winner | diffusability hypothesis |
| R3 | AE adapter with transform-equivariance loss | frozen winner | EQ-VAE hypothesis |
| R4 | R2/R3 adapter + renderer | frozen winner | interaction, not the primary claim |

R2 uses a radial-frequency penalty on the encoded latent, a reconstruction
term, and a channel-moment/scaling constraint; R3 adds transformed
input/latent/reconstruction consistency and measures equivariance error rather
than an invariance proxy. Train only on the registered train images, keep the
diffusion denoiser, scheduler, NFE, CFG, and renderer weights fixed, and match
the adapter parameter/FLOP budget to a random convolutional control. The AE
arms must preserve the scheduler's latent scale and channel moments; otherwise
they are representation changes, not fair renderer comparisons.

For every arm report native-resolution reconstruction/LPIPS, CLIP alignment,
radial DCT slope and high-frequency excess, equivariance error, latent
mean/variance drift, coarse-to-fine denoising order, latency, memory, and
parameter count. A spectral or equivariant AE that improves a proxy while
losing reconstruction, alignment, or the fixed montage is a failed arm. These
experiments are forbidden from rescuing a failed LR-1 gate and cannot authorize
RL; they only establish whether the journal contribution is the constrained
renderer, the representation, or their interaction.

## Candidate Renderer

The first implementation is frozen in `AttentionGuidance/latent_renderer.py`
and registered separately in `LATENT_RENDERER_PROTOCOL.md`. It has a
zero-initialized basis allocator plus an optional D4-symmetrized,
depthwise-separable spatial head; the coefficient-only path is retained as a
matched control. This keeps the initial identity, parameter count, residual
geometry, and scheduler injection auditable before any learned weights or RL
are introduced.
The companion YAML is a registration manifest only and must not be passed to
the legacy image-generation action loader.

The inference-only pipeline hook is now explicit: a
`RendererBasisProvider` consumes tensors from one ordinary denoiser forward and
returns `RendererCondition`. `StructuralUNetBasisProvider` is the historical
attention/decoder implementation; its Q/K graph is now an ablation rather than
the default learned path. It is mutually exclusive with the old guidance paths
and limited to Stage 1. Cached-SDXL smokes reproduce the no-renderer hash with a
zero renderer and change it with a fixed non-zero probe; this validates plumbing
only, not image quality or learned behavior.

At step `t`, one frozen U-Net evaluation returns noise `epsilon_t` and exposes a
small set of decoder backbone/skip features (B_t^l,S_t^l), self-attention
affinity (G_t), and the scheduler's predicted clean latent \(\hat x_0^t\). The
renderer sees only detached structural tensors and normalized timestep/prompt
features; let \(\bar x_{t-1}\) denote the ordinary Euler output:

```text
z_t = R_phi(x_t, x0_hat_t, B_t, S_t, G_t, t, pooled_text)
y_t = MomentGeodesic(x0_hat_t, z_t, theta_t)
g_t = 1 - sigma_next / sigma_t
x_{t-1} = xbar_{t-1} + g_t * (y_t - x0_hat_t)
```

The displayed gain is exact for no-churn Euler with the current state fixed.
Production reconstructs the native epsilon/sample/v prediction and invokes
`scheduler.step` once; DPM multistep methods fail closed because their history
must contain the guided output. `MomentGeodesic` is the existing
fixed-mean/fixed-variance operation, not an unconstrained post-hoc clip. The
per-step cap and diagnostics are measured after scheduler mapping. The renderer
is latent-space only; it is not a second VAE decoder and never predicts pixels.

The primary implementation should be a **basis renderer**, which limits the
learned output to bounded coefficients over interpretable bases:

1. low/mid/high DCT or Fourier residuals of \(\hat x_0^t\);
2. low/high bands of the native scheduler update;
3. a local Laplacian/edge residual; and
4. reciprocal semantic transport and FreeU-style feature differences as
   explicit attention/decoder ablations.

A depthwise 3x3 latent stem, low-rank 1x1 feature adapters, and a FiLM/timestep
head should fit 0.1M, 1M, and 5M parameter variants. A free-form four-channel
renderer is a capacity stress test, not the main claim. Every variant records
exact parameters, FLOPs, latency, memory, and the frozen U-Net parameter count.

The renderer should be equivariant under known spatial transforms. During
training, use a held-out scale/rotation/translation augmentation and penalize
\(\|R_\phi(\tau s)-\tau R_\phi(s)\|\), while separately measuring DCT slope,
high-frequency excess, channel moments, and coarse-to-fine ordering. This is an
inference-side analogue of the motivation in EQ-VAE, not a claim to reproduce
EQ-VAE.

## RL Protocol

RL is staged rather than used as the first source of supervision:

1. **Teacher collection:** on training-only prompts, search bounded basis
   coefficients with common random numbers and antithetic pairs. Cache U-Net
   features and scheduler states.
2. **Distillation:** fit the small renderer to teacher coefficients/residuals
   with moment, equivariance, spectral, and action-bound losses. Compare a
   context-free head and a shuffled-feature head.
3. **Constrained online RL:** a policy emits bounded coefficients (or renderer
   residual parameters) at each denoising step. Use grouped rollouts with a
   shared prefix, antithetic actions, group-relative advantages, and a KL
   penalty to the distilled policy. BranchGRPO/step-wise-RL implementations
   are comparison points; the optimizer itself is not claimed as novel.
4. **Deployment distillation:** if stochastic RL wins, distill its mean policy
   into a deterministic renderer and re-evaluate at identical NFE.

The state is Markov only after including (x_t), timestep, scheduler update,
attention entropy/confidence, feature energy, DCT band energies, prior action,
and prompt embedding. A scalar schedule-only policy is a required baseline.
If the renderer merely selects one global action per prompt, report it as a
contextual-bandit/search baseline rather than overstating it as sequential RL.

The reward is constrained multi-objective, with no single metric used as both
training target and primary test statistic:

```text
R = preference + alignment + structure
    - lambda_clip * clip_violation
    - lambda_sat  * saturation_violation
    - lambda_hf   * latent_high_frequency_excess
    - lambda_mom  * moment_error
    - lambda_cost * renderer_compute
```

Non-finite images, excessive clipping, or violated moment/norm bounds are hard
episode failures. TOPIQ-NR is the registered S5 primary metric and must remain
held out from renderer training (or be replaced by a disjoint evaluator).
Report every raw reward component, reward-model ensemble disagreement, and
human pairwise preference; do not report only the composite reward.

## Experiment Matrix

The sequence is fixed before viewing test images:

* **Static gate:** S5 SCRST, no-AG, conference expert, raw/clean latent TFSA,
  permuted graph, PLADIS, GAG, and FreeU at equal NFE.
* **Mechanism audit:** DCT profiles across timesteps, backbone/skip spectra,
  equivariance under scale/rotation/translation, moment preservation, and
  scheduler reconstruction error. Use synthetic fields with known frequencies
  to validate each renderer basis.
* **Training split:** prompt-disjoint train/validation/test sets stratified by
  complex composition, quantity, text, perspective, and fine detail. Seeds
  `0,42,123` are never used for policy selection on the final test set.
* **Capacity/compute controls:** fixed basis, 0.1M/1M/5M renderers, a random
  convolutional renderer with matched parameters, a full-UNet LoRA upper
  bound, and an equal-cost extra-UNet-call baseline. Keep NFE, resolution,
  CFG, sampler, VAE, and rollout budget identical.
* **Ablations:** remove graph, remove FreeU features, remove spectral loss,
  remove equivariance loss, remove moment geometry, shuffled features, fixed
  schedule, offline distillation only, and RL from scratch.
* **Evaluation:** TOPIQ-NR (primary), HPSv2, ImageReward, CLIP alignment,
  aesthetic score, OCR/text exact match, counting and spatial-relation
  accuracy, DCT/high-frequency statistics, LPIPS/diversity, clipping,
  saturation, contrast, channel moments, latency, peak memory, parameters,
  and FLOPs. Re-run the best frozen policy on unseen prompts and 2048x2048/target
  resolution. Add a blinded randomized human comparison with a prespecified
  sample size and confidence interval.

Use crossed prompt/seed bootstrap intervals, paired sign-flip tests, and
within-family Holm correction. A gain that disappears under new seeds,
high-resolution decoding, human preference, or the matched random renderer is
not a method gain. A renderer that beats no-AG but loses to a fixed searched
basis is a failed learning claim.

## Paper Logic

The paper can connect the versions through the shared principle of **reprogramming
the latent trajectory using model-native structure**: the conference version
provides a hand-designed, training-free operator; the journal version studies
whether a constrained low-capacity renderer can learn that operator and adapt
it safely. The bridge is conceptual, not a claim that the old scalar residual
is the new method. The main figure should show the frozen denoiser, structural
feature probes, constrained renderer, and scheduler-consistent injection. A
negative result, all compute controls, and the exact stopping rule belong in
the paper; TPAMI reviewers will treat their omission as evidence of reward
hacking or uncontrolled capacity.
