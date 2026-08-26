# Representation-Side Research Audit

This note records the representation direction suggested for a later journal
extension. It is an audit and preregistration boundary, not evidence that the
route works on RepLDM.

## What The Literature Establishes

- [Improving the Diffusability of Autoencoders](https://arxiv.org/abs/2502.14831)
  suppresses latent high frequencies with scale-equivariant/downsampled
  reconstruction losses. The autoencoder is fine-tuned and the downstream
  DiT is trained on the resulting latent distribution; this does not validate
  a zero-shot frequency projection for a frozen SDXL U-Net/VAE.
- [EQ-VAE](https://arxiv.org/abs/2502.09509) reports that explicit latent
  equivariance can collapse the representation. Its effective implicit loss
  also fine-tunes the autoencoder and retrains/evaluates a model in that new
  latent space. Reconstruction, moments, scaling, and distribution matching
  are therefore mandatory checks here.
- [Diffusing in the Right Space](https://arxiv.org/abs/2606.03578) finds that
  velocity ambiguity and semantic/spatial diagnostics are more stable across
  model families than spectral smoothness alone. DCT slope cannot be the sole
  training objective.
- [SPA](https://arxiv.org/abs/2607.22091) already covers timestep-conditioned
  FFT-spectrum guidance, so a pure spectral correction needs it as a direct
  baseline and cannot be claimed as novel.
- [SGA](https://arxiv.org/abs/2605.20808) and [SARA](https://arxiv.org/abs/2503.08253)
  already use spatial self-similarity or cosine/Gram relations as alignment
  signals. [DUNE](https://arxiv.org/abs/2607.09753) is a particularly relevant
  frozen-SDXL, training-free latent intervention baseline. FRLA therefore cannot
  claim first relational alignment or first frozen latent manipulation; its
  narrow test is the fixed, rank-compatible local descriptor attached to a
  detached U-Net feature and a bounded `pred_original_sample` update.

## Candidate And Gate

Only after S7's fixed-action gate is resolved may we test a tiny, zero-initialized
post-hoc latent adapter that preserves the original SDXL latent coordinates.
The first study must be representation-only: frozen U-Net, VAE, scheduler,
CFG, NFE, and initial noise; no RL and no direct VAE replacement. Compare
no-op, fixed radial-DCT projection, ordinary reconstruction fine-tuning,
scale-equivariant implicit fine-tuning, and a parameter-matched random adapter.
Measure reconstruction/LPIPS, latent moments and scale, DCT/SEC, equivariance
error, VIV/trajectory diagnostics, TOPIQ/HPS/CLIP, and pixel guards on a new
prompt-disjoint split. Include SPA, FreeU, native scheduler, and matched-cost
controls. A fixed adapter must beat no-op and SPA with positive confidence,
without a saturation/contrast/sharpness shortcut and across CFG/NFE (plus a
second backbone), before any distillation or RL controller is authorized.

## Candidate After The S7 Gate: FRLA

The most promising non-spectral candidate from the follow-up audit is **Frozen
Relational Latent Alignment (FRLA)**. It is only a registered hypothesis until
S7 is resolved. In the ordinary conditional U-Net forward, capture one decoder
feature map and downsample it, together with `pred_original_sample`, to a fixed
`16 x 16` token grid. The primary descriptor is five fixed local cosine
autocorrelations at lags `(1,0)`, `(0,1)`, `(1,1)`, `(2,0)`, and `(0,2)`; a full
Gram matrix is only an ablation because the four-channel latent and a wider
feature map have incompatible ranks. Take one latent-only gradient step that
reduces the lag-wise discrepancy, then inject the clean-latent residual after
the scheduler step with a scheduler-update trust cap and a channel-covariance
(`Mahalanobis`) retraction. This adds no U-Net call; the extra descriptor and
backward cost and peak memory must be recorded.

The motivation is relational rather than spectral: iREPA (arXiv:2512.10794),
SARA (arXiv:2503.08253), sREPA (arXiv:2605.16949), and *Diffusing in the Right
Space* (arXiv:2606.03578) point to spatial self-similarity and semantic
separability as useful structure signals. DiffRGD (arXiv:2606.28417) and
LatSearch (arXiv:2603.14526) show that frozen-base latent manipulation is
already an active area; their extra inner loops or candidate trajectories are
controls and novelty boundaries, not claims for FRLA.

The preregistered comparison must include no-op, conference TFSA, SPA, FreeU,
FRLA, shuffled/detached features, pointwise feature matching, a
relation-preserving-only projection, an isotropic shell, and a matched dummy
latent-gradient control. Report TOPIQ-NR, HPSv2, CLIP, OCR/count/spatial
probes, LPIPS/diversity, Gram error, latent moments, DCT/SEC, runtime/FLOPs,
and clipping/saturation/contrast/sharpness guards. A fixed FRLA action must
clear the same `+0.005`, crossed-bootstrap, sign-flip/Holm, and guard gates on
new prompts before any search, distillation, or RL is allowed. If it fails,
close the representation route rather than tuning the relation target.

Engineering controls are part of the preregistration. The current S7 jobs use
one prompt and one seed per block, but any batched FRLA implementation must
pair CFG rows as `[all negative, all positive]` and replicate latents with
`torch.cat([latents, latents])`; `repeat_interleave(2)` silently misaligns
prompts when the batch size exceeds one. Report the feature row used for the
conditional signal and test this pairing with a deterministic multi-prompt
smoke. VIV-style diagnostics must be labelled exploratory for SDXL Euler,
because the published derivation is for Flow Matching; use SEC/LNC/LDS/CDS/SRSS
as the scheduler-agnostic structural report.
