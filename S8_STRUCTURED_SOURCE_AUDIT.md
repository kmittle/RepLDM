# S8 Structured-Source Audit

**Status:** hypothesis and experiment-registration draft; no result is implied.
This audit concerns *Spatially-Grounded Flow Matching: Structured Source
Distributions for Image Generation* (StructFlow, arXiv:2608.15452, 2026-08-16)
and adjacent source-prior work such as *Better Source, Better Flow*
(arXiv:2602.05951). It is separate from the registered S7 sampler correction.

## Positioning and Novelty Boundary

StructFlow changes the source distribution at the start of a flow-matching
trajectory. It is not a latent renderer, an attention correction, or evidence
that SDXL's VAE latent has excessive high-frequency energy. The paper trains
the generator (and post-trains SANA) with the structured source; a frozen SDXL
U-Net was not validated. Correlated video-noise priors and learned
condition-dependent source distributions also predate this experiment. A
defensible RepLDM claim, if any, is much narrower:

> A covariance-preserving, spatially correlated initial-noise perturbation is
> a useful training-free ablation for a frozen SDXL Euler trajectory.

This is a source-robustness test, not a new structured latent representation.

The connection to the supplied autoencoder papers is indirect. Diffusability
(2502.14831) and EQ-VAE (2502.09509) change the clean latent representation or
its training objective. StructFlow changes the spatial covariance of x_T. It
does not suppress DCT high frequencies in z_0, and a lower DCT slope after
sampling would not establish that the VAE became more diffusible. Report these
as distinct axes.

## Mathematical Construction

Let M be a local-region mask of shape H x W with region labels 1..K. For each
region k, sample an anchor a_k ~ N(0, I_C). At each location (i,j), sample
u_ij ~ N(0, I_C) and construct:

```text
e_ij = (a[M(i,j)] + lambda * u_ij) / sqrt(1 + lambda^2)
rho  = 1 / (1 + lambda^2)
```

Each element has marginal N(0, I_C). Two positions in the same region have
correlation rho; positions in different regions are uncorrelated. lambda =
infinity (rho = 0) is the exact i.i.d. no-op. The paper uses SLIC/Voronoi-like
masks and may sample at full image resolution before downsampling to the VAE
grid. For SDXL, the initial latent at 1024px is 4 x 128 x 128; the first
smoke should use a deterministic latent-grid Voronoi or block mask, with no
segmentation network.

Do not alter per-step Euler-Ancestral noise in the first study. Initial-source
correlation and scheduler noise are separate factors. Record empirical mean,
per-channel variance, spatial autocorrelation, covariance eigenvalues, and a
Gaussian KL proxy for every source.

## SDXL Euler Risks

1. **Training-distribution shift.** SDXL was trained with factorized Gaussian
   noise. StructFlow's gains rely on training or post-training the flow model
   with correlated sources; a frozen U-Net may reject or wash out the
   correlation. This is the primary reason this route is lower priority than
   CFG-OEC.
2. **Scheduler mismatch.** StructFlow reports DDIM/Euler flow integration, not
   Euler-Ancestral's stochastic transition. Initial correlation may disappear
   after the first ancestral noise injection. Measure autocorrelation after
   every step; do not infer persistence from x_T.
3. **Mask artifacts.** Grid, SLIC, or Voronoi boundaries can become visible.
   Mask geometry must be hashed and fixed for paired seeds; compare several
   geometries and a mask-permuted control.
4. **Prior distortion despite unit moments.** Unit marginal variance does not
   imply an SDXL-compatible prior. Report covariance/KL, norm, DCT, and output
   diversity. A quality gain caused by reduced effective noise must be matched
   by a colored-Gaussian control with the same covariance spectrum.
5. **Wrong mechanism attribution.** A source-spectrum change is not evidence
   for AE high-frequency suppression, equivariance, or a common-mode CFG
   correction. These hypotheses need separate diagnostics.
6. **Batch and seed pairing.** Every action must receive identical prompt/seed
   pairs and the same mask realization where applicable. The current pipeline's
   CFG row order must be regression-tested at batch 2 before interpreting any
   branch-dependent result.

## Minimal Smoke Actions

The source transform is an inference-only action applied immediately after
initial noise creation. It adds no U-Net calls and no scheduler steps. Use a
fresh smoke manifest rather than extending S7's registered YAML.

| ID | Source | Purpose |
| --- | --- | --- |
| `source_iid_noop` | i.i.d. N(0,I) | Byte/hash and reference baseline |
| `source_block_rho005` | 8 x 8 blocks, rho=0.05 | Weak local correlation |
| `source_block_rho010` | 8 x 8 blocks, rho=0.10 | Stronger local correlation |
| `source_voronoi_rho005` | deterministic 256-seed Voronoi mask | Non-grid locality |
| `source_colored_matched` | Gaussian field with matched covariance/eigenvalues but no region labels | Covariance-only control |
| `source_mask_shuffle` | same anchors and rho, spatially permuted mask | Tests whether locality, not correlation alone, matters |
| `euler_ancestral_reference` | native scheduler | S7 reference, non-selectable |

Use two prompts and two seeds for a plumbing smoke, then the registered
development prompts and seeds only if all invariants pass. The sidecar must
include mask hash, rho, region count and size, source covariance summary,
scheduler, NFE, CFG, and exact seed. Verify that rho=0 is byte-identical to
the no-op path and that all sources are finite.

## Required Reviewer Controls

At minimum, compare native Euler, native Euler-Ancestral, and (on a separate
exactly reproducible arm) DDIM. Pair every method by prompt, seed, and initial
latent. Include:

* matched-norm i.i.d. resampling, low-pass/colored Gaussian noise, and a global
  anchor source;
* block, Voronoi, and shuffled-mask geometries at equal empirical covariance;
* rho monotonicity and an antithetic source pair;
* no-op hash, source covariance/KL, latent norm and moments, and
  autocorrelation trajectories r_t;
* TOPIQ-NR, HPSv2, CLIP, LPIPS/Vendi diversity, clipping, saturation,
  contrast/sharpness, OCR, counting, and spatial-relation/GenEval or
  T2I-CompBench probes; and
* NFE, U-Net calls, wall time, throughput, peak VRAM, and mask-generation
  cost.

The strongest reviewer counterexample is "this merely chooses a favorable
noise seed or lowers high-frequency noise." The covariance-matched and
mask-shuffled controls, paired seeds, DCT/SEC diagnostics, and diversity guard
are mandatory responses. A second counterexample is "the paper's post-training
result was silently transferred to a frozen U-Net"; explicitly report the
frozen setting as an out-of-distribution test. A small post-trained adapter may
be an upper-bound arm later, but it cannot rescue a failed fixed-source gate or
authorize RL.

## Go/No-Go Gate

**Smoke gate:** exact no-op hash, finite images, correct moments, source
covariance within its registered target, and no CFG batch mispairing. Failure
closes the implementation.

**Development gate:** on prompt-disjoint held-out prompts, require TOPIQ-NR
gain >= 0.005, crossed-bootstrap 95% CI lower bound > 0, paired prompt-level
sign-flip with Holm p < 0.05, HPSv2/CLIP non-inferiority, clipping increase <=
0.001, saturation increase <= 0.005, and no regression on OCR/counting/spatial
probes or diversity. The structured action must also beat the
covariance-matched and mask-shuffled controls; beating only the i.i.d. no-op is
insufficient.

**Transfer gate:** repeat the frozen winner at a second resolution, at least
one other sampler, and ideally a second backbone. If autocorrelation vanishes
within the first few Euler-Ancestral steps, or gains disappear under matched
covariance, close the route. Do not tune rho, mask count, reward, renderer
capacity, or RL after a failed gate.

## Priority and Queue Recommendation

The recommended order after S7 is:

1. **CFG-OEC/CFG-EC:** highest priority. It is a fixed prediction-level
   correction, has no extra NFE, and can be ported to Euler/Euler-Ancestral
   with cached branch predictions. It is a strong baseline and a narrow,
   falsifiable operator, not an RL novelty claim.
2. **SPA:** second. The official implementation and SDXL prior make it
   reproducible, but an Euler prior must be collected for the actual scheduler;
   the published DDIM spline cannot be reused.
3. **DG-CFG:** third. It is zero-cost and theoretically motivated, but its
   published formula is VP/DDIM-specific. An Euler sigma derivation and exact
   transport smoke are prerequisites.
4. **Structured source / StructFlow transfer:** exploratory last. It is the
   cheapest smoke but has the largest frozen-model distribution-shift risk and
   the weakest novelty for a TPAMI extension.

While S7 is occupying the GPU, prepare CPU-side manifests, deterministic mask
generators, covariance-statistics code, and SPA prior metadata. Once S7 reaches
its go/no-go decision, run the CFG-OEC smoke first; only queue the structured
source smoke if the fixed correction route remains scientifically open. A
positive structured-source smoke authorizes a larger fixed-source study, not
distillation or RL.
