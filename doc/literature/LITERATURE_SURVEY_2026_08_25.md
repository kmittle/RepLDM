# Literature Survey: Frozen Latent-Trajectory Extensions

**Cutoff:** 2026-08-26. This is a research memo, not evidence that any
reported result transfers to RepLDM. Links use moving arXiv records; pin the
reviewed version and refresh version, withdrawal, and venue status before
submission, especially for the July-August 2026 entries.

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
and [sREPA](https://arxiv.org/abs/2605.16949) provide evidence, in their
training-time settings, that spatial relations can outperform pointwise or
global semantic alignment. [SPARE](https://arxiv.org/abs/2608.01990v1) goes
further: during from-scratch SiT training it matches all non-self within-image
affinities, plus same-position cross-image affinities, to true clean VAE latent
affinities without an encoder or projection head. It does not perform frozen-
model inference transport; sREPA mentions local relation matching only as
future work. Together they establish pairwise-affinity prior art and explicitly
anticipate local matching, requiring uniform-local and predicted-clean-affinity
controls here. [Diffusing in the Right Space](https://arxiv.org/abs/2606.03578)
finds diffusability is multifactorial and proposes velocity irreducible
variance (VIV); we therefore do not treat a DCT slope as a sufficient causal
explanation.

The proposed residual `W z - z` is also the negative random-walk graph
Laplacian, with feature-conditioned local weights closely related to
[bilateral filtering](https://doi.org/10.1109/ICCV.1998.710815) and
[guided filtering](https://doi.org/10.1109/TPAMI.2012.213). Graph smoothing is
therefore not a novelty claim. The open question is whether an ordinary frozen
U-Net supplies a better guide than uniform or self-guided predicted-clean
weights when the same operator is mapped into scheduler coordinates.

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
Riemannian updates, and joint conditioning/latent test-time optimization
(latent-only for PG-MAP's flow setting).

For deterministic Euler with churn disabled, suppose a renderer adds
`delta_x0` to the clean-sample estimate at `sigma_t > 0`. At the next scheduled
level `sigma_next`, the exact endpoint change is
`(1 - sigma_next / sigma_t) * delta_x0`. Reconstruct the scheduler's native
prediction and call `scheduler.step` exactly once; a unit-gain post-step nudge
is not equivalent. Do not reuse this mapping for Euler-Ancestral, DPM-Solver++,
or UniPC: derive each intervention in that scheduler's native coordinates,
including stochastic and multistep state. Compare schedulers at matched actual
NFE, then report wall time and memory separately (and matched-wall-time results
where useful), using common initial noise and stochastic draws where
applicable. A gain only over deterministic Euler is a sampler result.

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
reward-scored rollout trajectories. [BranchGRPO](https://arxiv.org/abs/2509.06040) covers
shared-prefix branching, and [Designing RL for Diffusion](https://arxiv.org/abs/2608.14430)
unifies reverse-trajectory and forward-matching objectives. Thus RL, dense
credit, adaptive schedules, and search-to-distill are not individual claims.

Two full texts released immediately before this update further narrow the
learning claim. [DiffusionOPSD](https://arxiv.org/abs/2608.24646v1) collects
low-noise queries from a frozen behavior policy, converts its prediction to a
clean-output anchor, constructs norm-bounded positive and negative targets with
image-reward gradients, detaches those targets, and fits them before refreshing
the behavior policy by EMA. It explicitly measures target-construction gain and
finite-fitting gain because a locally better target can be realized poorly after
a parameter update. [Reward-based Velocity Matching
(RVM)](https://arxiv.org/abs/2608.23664v1) replaces trajectory likelihood ratios
with signed reward-weighted velocity regression plus a reference or EMA anchor;
its analysis places RAM and DiffusionNFT inside the same update family and finds
reward and anchor choices more consequential than several loss variants.

Their reported results have not been reproduced here, and both primarily update
the generative model rather than RepLDM's sub-1M side renderer. Nevertheless,
reward-to-clean-target self-distillation, trajectory-free native-coordinate
matching, signed reward regression, and EMA/reference anchoring are now occupied
components. If the relational static and replay gates pass, the first learning
matrix must compare the frozen search teacher with an OPSD-style bounded-target
distillation control and must report target gain separately from realized
renderer gain. Decoder/reward backpropagation is a `CX` training cost. RVM's
rectified-flow velocity equations cannot be copied into SDXL epsilon-prediction
Euler; an exact coordinate derivation and anchored native-prediction baseline
would be required. Neither paper permits reward learning to rescue a relational
family that failed its preregistered fixed-action gate.

## Narrow Hypothesis and Falsification

The best-supported candidate hypothesis is a zero-initialized, sub-1M
structural basis renderer that reads detached features from the ordinary frozen
SDXL forward, emits a bounded residual in scheduler clean-endpoint coordinates,
and obeys trust and moment bounds. Size, zero initialization, a frozen backbone,
and structural or spectral control are already covered. A defensible
differentiation would require a significant descriptor-by-scheduler-coordinate
interaction under one-call accounting and feature-affinity gains over both a
uniform local graph and a predicted-clean latent-affinity graph. It is not
established novelty in advance of those tests and a matched
SteeringDiffusion-style adapter.

Before learning, freeze a prompt-disjoint action and require TOPIQ-NR delta
`>= +0.005`, a crossed prompt/seed interval above zero, Holm-adjusted prompt
significance, HPSv2/CLIP non-inferiority, structural/detail evidence, and
clipping, saturation, diversity, moment, and off-target guards. Include no-op,
conference TFSA, fixed scalar/2-stage schedules, SPA/FreSca/FDG, FreeU, pointwise
and relational descriptors, shuffled features, uniform-local and predicted-
clean-affinity graphs, per-step RMS- and trust-cap-matched random directions,
and a compute-matched extra-U-Net reference. Only
after a fixed action passes should common-random-number search,
search-then-distill, and then RL be attempted; use antithetic pairs only if they
demonstrably reduce estimator variance. RL must beat fixed search and
distillation on unseen prompts after amortized GPU cost, with a KL/reference
anchor and independent native-resolution, patch, and human witnesses. Failure
closes the currently registered positive `+0.02` fixed-action family and bars
RL from rescuing that family on the same evidence. It does not falsify every
signed, scheduled, or learned renderer; any distinct family requires an
independent rationale, new preregistration, and untouched prompt splits.

The `+0.005` and non-inferiority margins are preregistered decision thresholds,
not perceptual calibrations. Calibrate them against repeated human judgments
before a paper claim. Treat the current 33-prompt, three-seed grid as a screen:
report sensitivity across its 11 challenge groups, then use larger external
prompt suites and more seeds for any generalization claim.
