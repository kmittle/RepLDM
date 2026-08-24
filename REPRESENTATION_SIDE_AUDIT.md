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

