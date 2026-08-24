# Structural Latent Rendering: RL Research Design

This document is a proposal for the journal extension. It does not change the
registered S5 experiment or authorize an RL run. The first decision gate is
the S5 static result; a policy or a larger network must not be used to rescue a
failed Attention Guidance hypothesis.

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

Two additional overlaps are important for the word "renderer": **LaRender
(arXiv:2508.07647)** already performs training-free volumetric compositing of
object-wise cross-attention features in latent space, using masks and an
occlusion graph. **SATeCo (arXiv:2403.17000)** freezes a pretrained UNet/VAE
and trains small spatial/temporal feature adapters. The proposed method is not
the first latent renderer or the first frozen-backbone adapter: it is a
different, narrower object that renders a bounded residual for the scheduler
transition from self-attention and decoder structure, without object masks,
extra prompt branches, or a second denoiser call. LaRender-style compositing,
FreeU, and matched random adapters must be included wherever their task is
well-defined.

The second supplied paper is also more precisely described as **equivariance**
regularization: EQ-VAE asks transformed inputs to produce correspondingly
transformed latents/reconstructions, rather than forcing all transformed views
to share an invariant code. This distinction determines the renderer test:
measure both equivariance error and unwanted content change instead of calling
one scalar an "invariance" score.

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

At step `t`, one frozen U-Net evaluation returns noise `epsilon_t` and exposes a
small set of decoder backbone/skip features (B_t^l,S_t^l), self-attention
affinity (G_t), and the scheduler's own predicted clean latent
\(\hat x_0^t\). Let \(\bar x_{t-1}\) be the scheduler output before rendering.
The renderer sees only these tensors and normalized timestep/prompt features:

```text
z_t = R_phi(x_t, x0_hat_t, B_t, S_t, G_t, t, pooled_text)
y_t = MomentGeodesic(x0_hat_t, z_t, theta_t)
x_{t-1} = xbar_{t-1} + (y_t - x0_hat_t)
```

`MomentGeodesic` is the existing fixed-mean/fixed-variance tangent operation,
not an unconstrained post-hoc clip. A per-step norm budget is enforced before
the geodesic map and is reported as a scheduler-update ratio. The renderer is
latent-space only; it is not a second VAE decoder and never predicts pixels.

The primary implementation should be a **basis renderer**, which limits the
learned output to bounded coefficients over interpretable bases:

1. reciprocal semantic transport of \(\hat x_0^t\);
2. low/mid/high DCT residuals;
3. a FreeU-style backbone-minus-skip feature projection; and
4. a local Laplacian/edge residual.

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
