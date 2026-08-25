# Literature Survey: Frozen Latent-Trajectory Extensions

**Cutoff:** 2026-08-25. This is a research memo, not evidence that any
reported result transfers to RepLDM. The 2026 entries are arXiv preprints and
must be refreshed before submission.

## Representation and Coarse-to-Fine Structure

[Improving the Diffusability of Autoencoders](https://arxiv.org/abs/2502.14831)
finds excess high-frequency energy in modern AE latents and uses
scale-equivariant decoder regularization. It fine-tunes the AE and trains a
downstream generator in the changed latent distribution. [EQ-VAE](https://arxiv.org/abs/2502.09509)
regularizes transformation *equivariance* (scale and rotation), rather than
invariance, and also changes the AE before generator training. These papers
motivate measuring DCT profiles and equivariance, but do not validate a
zero-shot projection of a frozen SDXL latent.

[FreSca](https://arxiv.org/abs/2504.02154), [FDG](https://arxiv.org/abs/2506.19713),
[SPA](https://arxiv.org/abs/2607.22091), and [SpectralDiT](https://arxiv.org/abs/2606.18765)
already cover frequency-decoupled CFG, FFT-prior guidance, and zero-initialized
timestep-conditioned spectral residuals. [iREPA](https://arxiv.org/abs/2512.10794),
[SARA](https://arxiv.org/abs/2503.08253), [SGA](https://arxiv.org/abs/2605.20808),
and [sREPA](https://arxiv.org/abs/2605.16949) show that spatial relations can
matter more than pointwise or global semantic alignment, but their primary
setting is training-time representation alignment. [Diffusing in the Right
Space](https://arxiv.org/abs/2606.03578) finds diffusability is multifactorial
and proposes velocity irreducible variance (VIV); a DCT slope alone is not a
causal mechanism.

## Frozen Inference and Scheduler Controls

[FreeU](https://arxiv.org/abs/2309.11497), [PAG](https://arxiv.org/abs/2403.17377),
[PLADIS](https://arxiv.org/abs/2503.07677), [GAG](https://arxiv.org/abs/2603.02531),
[APG](https://arxiv.org/abs/2410.02416), and [Guidance Interval](https://arxiv.org/abs/2404.07724)
occupy the attention, feature, projection, and timestep-guidance space.
[SteeringDiffusion](https://arxiv.org/abs/2605.01653) is especially close:
it freezes SD1.5/SDXL, uses a small prompt-conditioned zero-init FiLM/AdaGN
module, and gates it by timestep. [LaRender](https://arxiv.org/abs/2508.07647)
occupies the literal “latent renderer” term for object masks and occlusion.
[DUNE](https://arxiv.org/abs/2607.09753), [DIAMOND](https://arxiv.org/abs/2602.00883),
[DiffRGD](https://arxiv.org/abs/2606.28417), and [PG-MAP](https://arxiv.org/abs/2606.22958)
cover internal-latent suppression, clean-sample trajectory correction,
Riemannian updates, and test-time latent optimization.

For Euler, a clean-endpoint residual must be mapped through the actual
scheduler: with noise level `sigma_t` and next level `sigma_next`, its exact
no-churn gain is `g_t = 1 - sigma_next / sigma_t`. Reconstruct the scheduler's
native prediction and call `scheduler.step` once. A unit-gain post-step nudge
is not equivalent. Compare native Euler-Ancestral, deterministic Euler,
DPM-Solver++ and UniPC at matched actual NFE, random draws, wall time, and
memory. A gain only over deterministic Euler is a sampler result.

## Diffusion-RL Boundary

[DDPO](https://arxiv.org/abs/2305.13301), [DPOK](https://arxiv.org/abs/2305.16381),
[Flow-GRPO](https://arxiv.org/abs/2505.05470), and [DanceGRPO](https://arxiv.org/abs/2505.07818)
optimize generator policies or weights. [Dynamic CFG](https://arxiv.org/abs/2509.16131)
uses online feedback to search per-step guidance, while
[LeSAMP](https://arxiv.org/abs/2607.23488) trains a policy for prompt- and
timestep-dependent sampling parameters. [DRM](https://arxiv.org/abs/2605.25661),
[DenseGRPO](https://arxiv.org/abs/2601.20218), and [SDPO](https://arxiv.org/abs/2411.11727)
provide noisy/clean intermediate rewards. [CRD](https://arxiv.org/abs/2603.14128)
adds within-prompt centering and a KL anchor;
[RTDMD](https://arxiv.org/abs/2605.26108) combines reward-tilted distribution
matching with RL, while [REST](https://arxiv.org/abs/2608.09226) distills
reward-selected trajectories. [BranchGRPO](https://arxiv.org/abs/2509.06040) covers
shared-prefix branching, and [Designing RL for Diffusion](https://arxiv.org/abs/2608.14430)
unifies reverse-trajectory and forward-matching objectives. Thus RL, dense
credit, adaptive schedules, and search-to-distill are not individual claims.

## Narrow Hypothesis and Falsification

The only defensible extension is a zero-initialized, sub-1M structural basis
renderer that reads detached features from the ordinary frozen SDXL forward,
emits a bounded residual in scheduler clean-endpoint coordinates, and obeys a
trust cap and moment bound. Its novelty is the tested interaction of descriptor
and scheduler injection, not any ingredient alone.

Before learning, freeze a prompt-disjoint action and require TOPIQ-NR delta
`>= +0.005`, a crossed prompt/seed interval above zero, Holm-adjusted prompt
significance, HPSv2/CLIP non-inferiority, structural/detail evidence, and
clipping, saturation, diversity, moment, and off-target guards. Include no-op,
conference TFSA, fixed scalar/2-stage schedules, SPA/FreSca/FDG, FreeU, pointwise
and relational descriptors, shuffled features, L2-matched random directions,
and an equal-cost extra-U-Net upper bound. Only after a fixed action passes
should common-random-number antithetic search, search-then-distill, and then
RL be attempted. RL must beat fixed search and distillation on unseen prompts
after amortized GPU cost, with a KL/reference anchor and independent
native-resolution, patch, and human witnesses. If the fixed interaction fails,
the scientifically correct result is a negative extension; RL must not be used
to rescue it.
